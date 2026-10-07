"""Chapter 9 · Minimal code 2: a modern Transformer in one file. After a few minutes of training
on a CPU, it writes text in a "Shakespeare style".

Each component has a matching component in the main-line model (zero/model.py, Qwen3 structure):
  Pre-Norm RMSNorm, RoPE, QK-Norm, causal multi-head attention, SwiGLU, residual connections,
  tied embeddings, no bias.
To show one idea at a time, this file has no GQA, KV cache, YaRN, mixed precision,
or distributed training (Chapters 10, 14, 15).

Data: assets/tiny_corpus/shakespeare.txt, split into bytes → a vocabulary of 256.
      The loss (nat/byte) ÷ ln2 is the bits-per-byte of Chapter 7.
Run: uv run python chapters/09-modern-transformer/code/02_tiny_transformer.py
     (about 11 min of CPU time on 1 thread; to save time, add --steps 400. The script saves the
     result in code/out/tiny_transformer.pt. Scripts 3 and 4 and the video read this file.)
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
    vocab_size: int = 256     # byte level: each byte is one token
    dim: int = 128            # d_model: the width of the residual stream
    n_layers: int = 4
    n_heads: int = 4          # head_dim = 128 / 4 = 32
    ffn_dim: int = 352        # SwiGLU hidden dimension ≈ 8/3 · dim, a multiple of 32
    seq_len: int = 128        # context length T
    rope_theta: float = 10000.0
    eps: float = 1e-6


# ── 1. RMSNorm (Chapter 6): scale only, no shift ─────────────────────────────
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps, self.weight = eps, nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return self.weight * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


# ── 2. RoPE: rotate each dimension pair of q, k by an angle that depends on the position ──
def rope_cos_sin(head_dim, seq_len, theta=10000.0):
    inv_freq = theta ** (-torch.arange(0, head_dim, 2).float() / head_dim)  # ω_i = θ^(-2i/d)
    angles = torch.outer(torch.arange(seq_len).float(), inv_freq)           # (T, d/2): m·ω_i
    angles = torch.cat([angles, angles], dim=-1)                            # (T, d)
    return angles.cos(), angles.sin()


def apply_rope(x, cos, sin):
    """Dimension i and dimension i + d/2 make a pair. Rotate the pair in 2D: (a, b) → (a·cos − b·sin, b·cos + a·sin)."""
    a, b = x.chunk(2, dim=-1)
    return x * cos + torch.cat([-b, a], dim=-1) * sin


# ── 3. Attention (Chapter 8) + QK-Norm + RoPE ───────────────────────────────
class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.h, self.hd = cfg.n_heads, cfg.dim // cfg.n_heads
        self.wq = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.wk = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.wv = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.wo = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.q_norm = RMSNorm(self.hd, cfg.eps)  # QK-Norm: normalize q and k of each head first
        self.k_norm = RMSNorm(self.hd, cfg.eps)

    def forward(self, x, cos, sin):
        B, T, D = x.shape
        q = self.q_norm(self.wq(x).view(B, T, self.h, self.hd)).transpose(1, 2)  # (B, H, T, hd)
        k = self.k_norm(self.wk(x).view(B, T, self.h, self.hd)).transpose(1, 2)
        v = self.wv(x).view(B, T, self.h, self.hd).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.hd)                   # (B, H, T, T)
        mask = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
        scores = scores.masked_fill(~mask, float("-inf"))                        # causal mask
        out = scores.softmax(-1) @ v                                             # (B, H, T, hd)
        return self.wo(out.transpose(1, 2).reshape(B, T, D))


# ── 4. SwiGLU feed-forward: down( silu(gate(x)) ⊙ up(x) ) ─────────────────────
class SwiGLU(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.w_gate = nn.Linear(cfg.dim, cfg.ffn_dim, bias=False)
        self.w_up = nn.Linear(cfg.dim, cfg.ffn_dim, bias=False)
        self.w_down = nn.Linear(cfg.ffn_dim, cfg.dim, bias=False)

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


# ── 5. Block: Pre-Norm + residual connection ─────────────────────────────────
class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attn_norm, self.attn = RMSNorm(cfg.dim, cfg.eps), Attention(cfg)
        self.ffn_norm, self.ffn = RMSNorm(cfg.dim, cfg.eps), SwiGLU(cfg)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.attn_norm(x), cos, sin)  # exchange information between positions
        x = x + self.ffn(self.ffn_norm(x))              # process each position separately
        return x


# ── 6. The full model: embedding → N Blocks → RMSNorm → lm_head (tied to the embedding) ──
class TinyTransformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.layers = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.dim, cfg.eps)
        self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # tied embeddings: use the same matrix two times
        cos, sin = rope_cos_sin(cfg.dim // cfg.n_heads, cfg.seq_len, cfg.rope_theta)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        for name, p in self.named_parameters():  # the same initialization as zero
            if p.dim() == 2:
                std = 0.02 / math.sqrt(2 * cfg.n_layers) if name.endswith(("wo.weight", "w_down.weight")) else 0.02
                nn.init.normal_(p, 0.0, std)

    def forward(self, idx):                     # idx: (B, T) integers
        T = idx.shape[1]
        x = self.tok_emb(idx)                   # (B, T, D)
        for layer in self.layers:
            x = layer(x, self.cos[:T], self.sin[:T])
        return self.lm_head(self.norm(x))       # (B, T, V): scores for the next byte at each position


@torch.no_grad()
def generate(model, prompt: bytes, n_new: int, temperature=0.8, seed=0) -> str:
    g = torch.Generator().manual_seed(seed)
    idx = torch.tensor([list(prompt)], dtype=torch.long)
    for _ in range(n_new):  # for each new byte, calculate the full sequence again: slow! Chapter 10 fixes this with a KV cache
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
    y = torch.stack([data[i + 1:i + cfg.seq_len + 1] for i in ix])  # target = input shifted by one position
    return x, y


@torch.no_grad()
def eval_bpb(model, data, cfg, n_batches=20):
    g = torch.Generator().manual_seed(123)
    losses = [F.cross_entropy(model(x).flatten(0, 1), y.flatten()).item()
              for x, y in (get_batch(data, cfg, 32, g) for _ in range(n_batches))]
    return sum(losses) / len(losses) / math.log(2)  # nat/byte → bit/byte


def train(steps=1200, batch_size=32, lr=3e-3, warmup=100, seed=1337):
    torch.manual_seed(seed)
    cfg = Config()
    model = TinyTransformer(cfg)
    train_data, val_data = load_data()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,} (embedding: {model.tok_emb.weight.numel():,}, tied with lm_head)")
    prompt = b"ROMEO:\n"
    before = generate(model, prompt, 200)
    print("── Sample before training ──\n" + before)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)

    def lr_at(s):  # Chapter 6: linear warmup + cosine decay to 10%
        return lr * min(1.0, (s + 1) / warmup) * (0.1 + 0.45 * (1 + math.cos(math.pi * s / steps)))
    g = torch.Generator().manual_seed(seed)
    history, train_curve, t0 = [], [], time.time()
    for step in range(steps + 1):
        if step % 200 == 0 or step == steps:
            model.eval()
            val = eval_bpb(model, val_data, cfg)
            model.train()
            history.append({"step": step, "val_bpb": val, "time": time.time() - t0})
            print(f"step {step:5d} | val {val:.3f} bit/byte | {time.time() - t0:5.0f}s")
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
    print("── Sample after training ──\n" + after)
    CKPT.parent.mkdir(exist_ok=True)
    torch.save({"config": asdict(cfg), "model": model.state_dict(), "history": history, "train_curve": train_curve,
                "before": before, "after": after, "n_params": n_params}, CKPT)
    print(f"Saved {CKPT.relative_to(ROOT)}")


def load_trained():
    ck = torch.load(CKPT, weights_only=False)
    model = TinyTransformer(Config(**ck["config"]))
    model.load_state_dict(ck["model"])
    return model.eval(), ck


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1200, help="number of training steps; use 400 to save time")
    ap.add_argument("--threads", type=int, default=1, help="number of CPU threads; use a larger value when the machine is idle")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)  # default 1 thread: on a busy machine, more threads are much slower
    train(steps=args.steps)
