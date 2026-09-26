"""第 22 章 · 极简代码 2：一个"每层可以有不同窗口"的小 Transformer + 两个任务

和第 9、10 章同一套结构（Pre-Norm RMSNorm、RoPE、SwiGLU、共享 embedding），只改注意力的掩码：
  - windows[l] = None：第 l 层是全注意力（能看见全部过去）；
  - windows[l] = W   ：第 l 层是滑动窗口（只看最近 W 个位置，含自己）；
  - topk = k（只在 05 里用）：每个 query 只保留分数最高的 k 个键 —— 稀疏注意力的最简形式。

三种配置（都是 4 层）：
  full       = [None, None, None, None]
  sliding    = [W, W, W, W]                  （Mistral 7B v0.1 的做法）
  interleave = [W, W, W, None]               （3 局部 : 1 全局，OLMo 3 的比例）

两个任务：
  - 语言建模：assets/tiny_corpus/shakespeare.txt，字符级；
  - 大海捞针（needle）：一串随机"填充字符"里藏着一个"针"字符，最后一个位置要说出它是哪个。
    针离最后一个位置的距离 d 均匀随机，于是能按 d 统计准确率，看滑动窗口在哪里失效。

权重缓存在 code/out/*.pt（已被 .gitignore 忽略），03–05 直接加载。
运行：uv run python chapters/22-local-sparse-attention/code/02_swa_model.py   （训练全部 6 个模型，约 5 分钟）
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
CORPUS = ROOT / "assets" / "tiny_corpus" / "shakespeare.txt"
OUT = Path(__file__).resolve().parent / "out"
torch.set_num_threads(1)  # 构建环境里多个任务共享 CPU；读者本机可以删掉这行

W = 16  # 两个任务统一的窗口大小
VARIANTS = {
    "full": [None, None, None, None],
    "sliding": [W, W, W, W],
    "interleave": [W, W, W, None],
}


# ── 模型 ────────────────────────────────────────────────────────────────────
@dataclass
class Config:
    vocab_size: int
    windows: list = field(default_factory=lambda: [None] * 4)  # 每层的窗口，None = 全注意力
    dim: int = 96
    n_heads: int = 4
    ffn_dim: int = 256
    max_seq_len: int = 512

    @property
    def n_layers(self) -> int:
        return len(self.windows)

    @property
    def head_dim(self) -> int:
        return self.dim // self.n_heads


def window_mask(q_pos: torch.Tensor, k_pos: torch.Tensor, window: int | None) -> torch.Tensor:
    """(Tq, Tk) 布尔掩码：键在查询之前（含自己），且（有窗口时）距离 < window。"""
    q, k = q_pos[:, None], k_pos[None, :]
    mask = k <= q
    if window is not None:
        mask &= (q - k) < window
    return mask


class KVCache:
    """极简 KV cache：每层一个列表。滑动窗口层每次拼接后只留最近 W 个位置（截断），
    所以这些层的缓存大小封顶在 W；全注意力层照常越存越多。"""

    def __init__(self, n_layers: int) -> None:
        self.k = [None] * n_layers
        self.v = [None] * n_layers
        self.pos = [None] * n_layers  # 每个缓存槽对应的全局位置（用来构造掩码）

    def append(self, layer: int, k, v, pos, window: int | None):
        if self.k[layer] is not None:
            k = torch.cat([self.k[layer], k], dim=2)
            v = torch.cat([self.v[layer], v], dim=2)
            pos = torch.cat([self.pos[layer], pos])
        # 先把"旧 + 新"全部返回给这一步的注意力用，再截断存起来
        keep = slice(None) if window is None else slice(-window, None)
        self.k[layer], self.v[layer], self.pos[layer] = k[:, :, keep], v[:, :, keep], pos[keep]
        return k, v, pos

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self.k + self.v if t is not None)


class RMSNorm(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.w = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return self.w * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)


def rope_tables(head_dim: int, max_len: int, theta: float = 10000.0):
    freqs = 1.0 / theta ** (torch.arange(0, head_dim, 2).float() / head_dim)
    ang = torch.outer(torch.arange(max_len).float(), freqs)
    return ang.cos(), ang.sin()


def apply_rope(x, cos, sin):  # x: (B, H, T, D)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(-2)


class Attention(nn.Module):
    def __init__(self, c: Config, window: int | None) -> None:
        super().__init__()
        self.c, self.window = c, window
        self.topk: int | None = None  # 05 里临时打开
        self.wqkv = nn.Linear(c.dim, 3 * c.dim, bias=False)
        self.wo = nn.Linear(c.dim, c.dim, bias=False)

    def forward(self, x, cos, sin, pos, cache=None, layer=0):
        B, T, _ = x.shape
        c = self.c
        q, k, v = self.wqkv(x).view(B, T, 3, c.n_heads, c.head_dim).permute(2, 0, 3, 1, 4)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        k_pos = pos
        if cache is not None:  # 旧 K/V + 新 K/V；滑动窗口层之后会被截断到最近 W 个
            k, v, k_pos = cache.append(layer, k, v, pos, self.window)
        att = q @ k.transpose(-2, -1) / math.sqrt(c.head_dim)  # (B, H, T, S)
        att = att.masked_fill(~window_mask(pos, k_pos, self.window), float("-inf"))
        if self.topk is not None and self.topk < att.shape[-1]:
            # 稀疏注意力（最简形式）：每个 query 只留分数最高的 k 个键，其余当作看不见
            kth = att.topk(self.topk, dim=-1).values[..., -1:]
            att = att.masked_fill(att < kth, float("-inf"))
        out = (att.softmax(-1) @ v).transpose(1, 2).reshape(B, T, c.dim)
        return self.wo(out)


class Block(nn.Module):
    def __init__(self, c: Config, window: int | None) -> None:
        super().__init__()
        self.n1, self.attn, self.n2 = RMSNorm(c.dim), Attention(c, window), RMSNorm(c.dim)
        self.w_gate = nn.Linear(c.dim, c.ffn_dim, bias=False)
        self.w_up = nn.Linear(c.dim, c.ffn_dim, bias=False)
        self.w_down = nn.Linear(c.ffn_dim, c.dim, bias=False)

    def forward(self, x, cos, sin, pos, cache, layer):
        x = x + self.attn(self.n1(x), cos, sin, pos, cache, layer)
        h = self.n2(x)
        return x + self.w_down(F.silu(self.w_gate(h)) * self.w_up(h))


class TinyLM(nn.Module):
    def __init__(self, c: Config) -> None:
        super().__init__()
        self.c = c
        self.emb = nn.Embedding(c.vocab_size, c.dim)
        nn.init.normal_(self.emb.weight, std=0.02)
        self.blocks = nn.ModuleList(Block(c, w) for w in c.windows)
        self.norm = RMSNorm(c.dim)
        cos, sin = rope_tables(c.head_dim, c.max_seq_len)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, ids, cache=None, start: int = 0):
        T = ids.shape[1]
        pos = torch.arange(start, start + T)
        cos, sin = self.cos[start : start + T], self.sin[start : start + T]
        x = self.emb(ids)
        for layer, blk in enumerate(self.blocks):
            x = blk(x, cos, sin, pos, cache, layer)
        return self.norm(x) @ self.emb.weight.T

    def set_topk(self, k: int | None) -> None:
        for blk in self.blocks:
            blk.attn.topk = k


# ── 任务 1：字符级语言建模 ──────────────────────────────────────────────────
class CharData:
    def __init__(self) -> None:
        text = CORPUS.read_text(encoding="utf-8")
        self.chars = sorted(set(text))
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        ids = torch.tensor([self.stoi[ch] for ch in text], dtype=torch.long)
        n = int(len(ids) * 0.9)
        self.train, self.val = ids[:n], ids[n:]

    def encode(self, s: str) -> list[int]:
        return [self.stoi[ch] for ch in s]

    def decode(self, ids) -> str:
        return "".join(self.chars[i] for i in ids)

    def batch(self, split: str, bsz: int, seq: int, g: torch.Generator):
        data = self.train if split == "train" else self.val
        ix = torch.randint(len(data) - seq - 1, (bsz,), generator=g)
        x = torch.stack([data[i : i + seq] for i in ix])
        y = torch.stack([data[i + 1 : i + seq + 1] for i in ix])
        return x, y


# ── 任务 2：大海捞针 ─────────────────────────────────────────────────────────
N_NEEDLE, N_FILL = 8, 32  # 词表：0..7 是"针"，8..39 是填充，40 是"提问"
QUERY = N_NEEDLE + N_FILL
NEEDLE_VOCAB = QUERY + 1
NEEDLE_T = 96  # 序列长度：针与提问的距离 d ∈ [1, 95]


def needle_batch(bsz: int, g: torch.Generator, d: torch.Tensor | None = None):
    """返回 (x, 答案, 距离)。x 的最后一个位置是"提问"，答案是藏在距离 d 处的针。"""
    x = torch.randint(N_NEEDLE, N_NEEDLE + N_FILL, (bsz, NEEDLE_T), generator=g)
    x[:, -1] = QUERY
    if d is None:
        d = torch.randint(1, NEEDLE_T, (bsz,), generator=g)
    ans = torch.randint(0, N_NEEDLE, (bsz,), generator=g)
    x[torch.arange(bsz), NEEDLE_T - 1 - d] = ans
    return x, ans, d


# ── 训练与评估 ──────────────────────────────────────────────────────────────
def train(task: str, windows: list, steps: int, seed: int = 0, verbose: bool = True) -> TinyLM:
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)  # 同一个种子 → 三种配置看到完全相同的数据
    if task == "lm":
        data = CharData()
        model = TinyLM(Config(len(data.chars), windows))
        lr, bsz, seq = 3e-3, 16, 128
    else:
        model = TinyLM(Config(NEEDLE_VOCAB, windows, dim=64, ffn_dim=128))
        lr, bsz = 2e-3, 32
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
    t0 = time.time()
    for step in range(steps + 1):
        for pg in opt.param_groups:  # warmup + cosine（第 6 章）
            pg["lr"] = lr * min(1, (step + 1) / 100) * 0.5 * (1 + math.cos(math.pi * step / steps))
        if task == "lm":
            x, y = data.batch("train", bsz, seq, g)
            loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten())
        else:
            x, ans, _ = needle_batch(bsz, g)
            loss = F.cross_entropy(model(x)[:, -1], ans)  # 只在"提问"位置算 loss
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if verbose and step % 250 == 0:
            print(f"    step {step:5d}  loss {loss.item():.3f}  ({time.time() - t0:.0f}s)")
    return model.eval()


STEPS = {"lm": 600, "needle": 600}


def load_or_train(task: str, variant: str, verbose: bool = True) -> TinyLM:
    windows = VARIANTS[variant]
    path = OUT / f"{task}_{variant}_W{W}_s{STEPS[task]}.pt"
    if task == "lm":
        c = Config(len(CharData().chars), windows)
    else:
        c = Config(NEEDLE_VOCAB, windows, dim=64, ffn_dim=128)
    if path.exists():
        model = TinyLM(c)
        model.load_state_dict(torch.load(path, weights_only=True))
        return model.eval()
    if verbose:
        print(f"  训练 {task} / {variant}（{STEPS[task]} 步，只需一次，之后从 {path.name} 加载）")
    model = train(task, windows, STEPS[task], verbose=verbose)
    OUT.mkdir(exist_ok=True)
    torch.save(model.state_dict(), path)
    return model


@torch.no_grad()
def lm_val_loss(model: TinyLM, seq: int = 128, n_batches: int = 20) -> float:
    data = CharData()
    g = torch.Generator().manual_seed(1234)
    losses = []
    for _ in range(n_batches):
        x, y = data.batch("val", 32, seq, g)
        losses.append(F.cross_entropy(model(x).flatten(0, 1), y.flatten()).item())
    return sum(losses) / len(losses)


@torch.no_grad()
def needle_accuracy(model: TinyLM, n_per_d: int = 64) -> torch.Tensor:
    """对每个距离 d = 1..NEEDLE_T-1 各测 n_per_d 条，返回每个 d 的准确率。"""
    g = torch.Generator().manual_seed(4321)
    accs = []
    for d in range(1, NEEDLE_T):
        x, ans, _ = needle_batch(n_per_d, g, d=torch.full((n_per_d,), d))
        accs.append((model(x)[:, -1].argmax(-1) == ans).float().mean())
    return torch.stack(accs)


if __name__ == "__main__":
    for task in ("lm", "needle"):
        for variant in VARIANTS:
            m = load_or_train(task, variant)
            n = sum(p.numel() for p in m.parameters())
            print(f"{task:>6} / {variant:<10} windows={m.c.windows}  参数 {n / 1e3:.0f}K")
