"""第 9 章 · 极简代码 2：一个文件的现代 Transformer，在 CPU 上训练几分钟就能写出"莎士比亚腔"

组件和主线模型（zero/model.py，Qwen3 结构）一一对应：
  Pre-Norm RMSNorm、RoPE、QK-Norm、因果多头注意力、SwiGLU、残差、共享 embedding、无 bias。
为了只讲一个想法，这里没有 GQA、KV cache、YaRN、混合精度、分布式（第 10、14、15 章）。

数据：assets/tiny_corpus/shakespeare.txt，按字节（byte）切分 → 词表 256，
      损失（nat/字节）÷ ln2 就是第 7 章的 bits-per-byte。
运行：uv run python chapters/09-modern-transformer/code/02_tiny_transformer.py
      （约 3–5 分钟；训练结果存到 code/out/tiny_transformer.pt，第 3、4 个脚本和视频会读它）
"""

import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

ROOT = Path(__file__).resolve().parents[3]
CKPT = Path(__file__).resolve().parent / "out" / "tiny_transformer.pt"


@dataclass
class Config:
    vocab_size: int = 256     # 字节级：每个字节就是一个 token
    dim: int = 128            # d_model：残差流的宽度
    n_layers: int = 4
    n_heads: int = 4          # head_dim = 128 / 4 = 32
    ffn_dim: int = 352        # SwiGLU 中间维度 ≈ 8/3 · dim，取 32 的倍数
    seq_len: int = 128        # 上下文长度 T
    rope_theta: float = 10000.0
    eps: float = 1e-6


# ── 1. RMSNorm（第 6 章）：只缩放、不平移 ────────────────────────────────────
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps, self.weight = eps, nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return self.weight * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


# ── 2. RoPE：按位置把 q、k 的每一对维度旋转一个角度 ──────────────────────────
def rope_cos_sin(head_dim, seq_len, theta=10000.0):
    inv_freq = theta ** (-torch.arange(0, head_dim, 2).float() / head_dim)  # ω_i = θ^(-2i/d)
    angles = torch.outer(torch.arange(seq_len).float(), inv_freq)           # (T, d/2)：m·ω_i
    angles = torch.cat([angles, angles], dim=-1)                            # (T, d)
    return angles.cos(), angles.sin()


def apply_rope(x, cos, sin):
    """维度 i 与 i + d/2 配成一对，做二维旋转：(a, b) → (a·cos − b·sin, b·cos + a·sin)。"""
    a, b = x.chunk(2, dim=-1)
    return x * cos + torch.cat([-b, a], dim=-1) * sin


# ── 3. 注意力（第 8 章）+ QK-Norm + RoPE ─────────────────────────────────────
class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.h, self.hd = cfg.n_heads, cfg.dim // cfg.n_heads
        self.wq = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.wk = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.wv = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.wo = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.q_norm = RMSNorm(self.hd, cfg.eps)  # QK-Norm：每个头的 q、k 先归一化
        self.k_norm = RMSNorm(self.hd, cfg.eps)

    def forward(self, x, cos, sin):
        B, T, D = x.shape
        q = self.q_norm(self.wq(x).view(B, T, self.h, self.hd)).transpose(1, 2)  # (B, H, T, hd)
        k = self.k_norm(self.wk(x).view(B, T, self.h, self.hd)).transpose(1, 2)
        v = self.wv(x).view(B, T, self.h, self.hd).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.hd)                   # (B, H, T, T)
        mask = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
        scores = scores.masked_fill(~mask, float("-inf"))                        # 因果 mask
        out = scores.softmax(-1) @ v                                             # (B, H, T, hd)
        return self.wo(out.transpose(1, 2).reshape(B, T, D))


# ── 4. SwiGLU 前馈：down( silu(gate(x)) ⊙ up(x) ) ────────────────────────────
class SwiGLU(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.w_gate = nn.Linear(cfg.dim, cfg.ffn_dim, bias=False)
        self.w_up = nn.Linear(cfg.dim, cfg.ffn_dim, bias=False)
        self.w_down = nn.Linear(cfg.ffn_dim, cfg.dim, bias=False)

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


# ── 5. Block：Pre-Norm + 残差 ────────────────────────────────────────────────
class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attn_norm, self.attn = RMSNorm(cfg.dim, cfg.eps), Attention(cfg)
        self.ffn_norm, self.ffn = RMSNorm(cfg.dim, cfg.eps), SwiGLU(cfg)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.attn_norm(x), cos, sin)  # 位置之间交换信息
        x = x + self.ffn(self.ffn_norm(x))              # 每个位置各自加工
        return x


# ── 6. 整个模型：embedding → N 个 Block → RMSNorm → lm_head（与 embedding 共享）──
class TinyTransformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.layers = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.dim, cfg.eps)
        self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # 共享 embedding：同一个矩阵用两次
        cos, sin = rope_cos_sin(cfg.dim // cfg.n_heads, cfg.seq_len, cfg.rope_theta)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        for name, p in self.named_parameters():  # 与 zero 相同的初始化
            if p.dim() == 2:
                std = 0.02 / math.sqrt(2 * cfg.n_layers) if name.endswith(("wo.weight", "w_down.weight")) else 0.02
                nn.init.normal_(p, 0.0, std)

    def forward(self, idx):                     # idx: (B, T) 整数
        T = idx.shape[1]
        x = self.tok_emb(idx)                   # (B, T, D)
        for layer in self.layers:
            x = layer(x, self.cos[:T], self.sin[:T])
        return self.lm_head(self.norm(x))       # (B, T, V)：每个位置对下一个字节的打分


@torch.no_grad()
def generate(model, prompt: bytes, n_new: int, temperature=0.8, seed=0) -> str:
    g = torch.Generator().manual_seed(seed)
    idx = torch.tensor([list(prompt)], dtype=torch.long)
    for _ in range(n_new):  # 每生成一个字节，都把整段重新算一遍 —— 慢！第 10 章用 KV cache 解决
        logits = model(idx[:, -model.cfg.seq_len:])[:, -1] / temperature
        nxt = torch.multinomial(logits.softmax(-1), 1, generator=g)
        idx = torch.cat([idx, nxt], dim=1)
    return bytes(idx[0].tolist()).decode("utf-8", errors="replace")


def load_data():
    raw = (ROOT / "assets/tiny_corpus/shakespeare.txt").read_bytes()
    data = torch.tensor(list(raw), dtype=torch.long)
    n = int(0.9 * len(data))
    return data[:n], data[n:]


def get_batch(data, cfg, batch_size, g):
    ix = torch.randint(len(data) - cfg.seq_len - 1, (batch_size,), generator=g)
    x = torch.stack([data[i:i + cfg.seq_len] for i in ix])
    y = torch.stack([data[i + 1:i + cfg.seq_len + 1] for i in ix])  # 目标 = 输入右移一位
    return x, y


@torch.no_grad()
def eval_bpb(model, data, cfg, n_batches=20):
    g = torch.Generator().manual_seed(123)
    losses = [F.cross_entropy(model(x).flatten(0, 1), y.flatten()).item()
              for x, y in (get_batch(data, cfg, 32, g) for _ in range(n_batches))]
    return sum(losses) / len(losses) / math.log(2)  # nat/字节 → bit/字节


def train(steps=1200, batch_size=32, lr=3e-3, warmup=100, seed=1337):
    torch.manual_seed(seed)
    cfg = Config()
    model = TinyTransformer(cfg)
    train_data, val_data = load_data()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"参数量 {n_params:,}（其中 embedding {model.tok_emb.weight.numel():,}，与 lm_head 共享）")
    prompt = b"ROMEO:\n"
    before = generate(model, prompt, 200)
    print("── 训练前的样本 ──\n" + before)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)

    def lr_at(s):  # 第 6 章：线性 warmup + 余弦衰减到 10%
        return lr * min(1.0, (s + 1) / warmup) * (0.1 + 0.45 * (1 + math.cos(math.pi * s / steps)))
    g = torch.Generator().manual_seed(seed)
    history, train_curve, t0 = [], [], time.time()
    for step in range(steps + 1):
        if step % 200 == 0 or step == steps:
            model.eval()
            val = eval_bpb(model, val_data, cfg)
            model.train()
            history.append({"step": step, "val_bpb": val, "time": time.time() - t0})
            print(f"step {step:5d} | val {val:.3f} bit/字节 | {time.time() - t0:5.0f}s")
        if step == steps:
            break
        for group in opt.param_groups:
            group["lr"] = lr_at(step)
        x, y = get_batch(train_data, cfg, batch_size, g)
        loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten())
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        train_curve.append(loss.item() / math.log(2))
    model.eval()
    after = generate(model, prompt, 400)
    print("── 训练后的样本 ──\n" + after)
    CKPT.parent.mkdir(exist_ok=True)
    torch.save({"config": asdict(cfg), "model": model.state_dict(), "history": history, "train_curve": train_curve,
                "before": before, "after": after, "n_params": n_params}, CKPT)
    print(f"已保存 {CKPT.relative_to(ROOT)}")


def load_trained():
    ck = torch.load(CKPT, weights_only=False)
    model = TinyTransformer(Config(**ck["config"]))
    model.load_state_dict(ck["model"])
    return model.eval(), ck


if __name__ == "__main__":
    torch.set_num_threads(1)  # 单线程：小模型上多线程收益小，机器繁忙时反而慢很多
    train()
