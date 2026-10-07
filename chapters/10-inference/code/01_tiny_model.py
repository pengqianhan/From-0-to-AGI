"""Chapter 10 · Minimal code 1: a small model that generates text (with a KV cache and a GQA option)

The structure is the same as in Chapter 9 (Pre-Norm RMSNorm, RoPE, SwiGLU, causal attention).
This file has its own copy, so it does not depend on the Chapter 9 files. It adds two things
from this chapter:
  - forward(ids, cache): with a KVCache, the model calculates q/k/v only for the new tokens.
    It reads the old K/V from the cache.
  - n_kv_heads: the number of K/V heads can be smaller than the number of query heads
    (GQA / MQA). Then several query heads share one set of K/V.

Data: assets/tiny_corpus/shakespeare.txt, character level (a vocabulary of 65 characters).
Training for 600 steps on 1 CPU thread takes about 1 to 2 minutes.
The script saves the trained weights in code/out/*.pt (.gitignore ignores *.pt).
The scripts after this one load these weights.
Run: uv run python chapters/10-inference/code/01_tiny_model.py
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
CORPUS = ROOT / "assets" / "tiny_corpus" / "shakespeare.txt"
OUT = Path(__file__).resolve().parent / "out"
# One thread: more threads help little on a very small model.
# With one thread, the timing is more stable and the results are easier to reproduce.
torch.set_num_threads(1)


# ── Data: character level ───────────────────────────────────────────────────
class CharData:
    def __init__(self) -> None:
        text = CORPUS.read_text(encoding="utf-8")
        self.chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        ids = torch.tensor([self.stoi[c] for c in text], dtype=torch.long)
        n = int(len(ids) * 0.9)
        self.train, self.val = ids[:n], ids[n:]

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def encode(self, s: str) -> list[int]:
        return [self.stoi[c] for c in s]

    def decode(self, ids) -> str:
        return "".join(self.chars[i] for i in ids)

    def batch(self, split: str, bsz: int, seq: int, g: torch.Generator):
        data = self.train if split == "train" else self.val
        ix = torch.randint(len(data) - seq - 1, (bsz,), generator=g)
        x = torch.stack([data[i : i + seq] for i in ix])
        y = torch.stack([data[i + 1 : i + seq + 1] for i in ix])
        return x, y


# ── KV cache (minimal version: one list per layer; torch.cat appends the new K/V) ──
class KVCache:
    def __init__(self, n_layers: int) -> None:
        self.k: list[torch.Tensor | None] = [None] * n_layers
        self.v: list[torch.Tensor | None] = [None] * n_layers

    def __len__(self) -> int:  # the number of positions in the cache
        return 0 if self.k[0] is None else self.k[0].shape[2]

    def append(self, layer: int, k: torch.Tensor, v: torch.Tensor):
        if self.k[layer] is not None:
            k = torch.cat([self.k[layer], k], dim=2)  # (B, n_kv_heads, T_past + T_new, head_dim)
            v = torch.cat([self.v[layer], v], dim=2)
        self.k[layer], self.v[layer] = k, v
        return k, v

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self.k + self.v if t is not None)


# ── Model ───────────────────────────────────────────────────────────────────
@dataclass
class Config:
    vocab_size: int = 65
    dim: int = 128
    n_layers: int = 4
    n_heads: int = 4
    n_kv_heads: int = 4  # = n_heads is MHA; 1 is MQA; a value between the two is GQA
    ffn_dim: int = 384
    max_seq_len: int = 1024

    @property
    def head_dim(self) -> int:
        return self.dim // self.n_heads


class RMSNorm(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.w = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return self.w * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)


def rope_tables(head_dim: int, max_len: int, theta: float = 10000.0):
    freqs = 1.0 / theta ** (torch.arange(0, head_dim, 2).float() / head_dim)
    ang = torch.outer(torch.arange(max_len).float(), freqs)  # (max_len, head_dim/2)
    return ang.cos(), ang.sin()


def apply_rope(x, cos, sin):  # x: (B, H, T, D); cos/sin: (T, D/2)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return out.flatten(-2)


class Attention(nn.Module):
    def __init__(self, c: Config) -> None:
        super().__init__()
        self.c = c
        self.wq = nn.Linear(c.dim, c.n_heads * c.head_dim, bias=False)
        self.wk = nn.Linear(c.dim, c.n_kv_heads * c.head_dim, bias=False)  # GQA: narrower K/V projections
        self.wv = nn.Linear(c.dim, c.n_kv_heads * c.head_dim, bias=False)
        self.wo = nn.Linear(c.n_heads * c.head_dim, c.dim, bias=False)

    def forward(self, x, cos, sin, cache: KVCache | None, layer: int):
        B, T, _ = x.shape
        c = self.c
        q = self.wq(x).view(B, T, c.n_heads, c.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, c.n_kv_heads, c.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, c.n_kv_heads, c.head_dim).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)  # the cache keeps K after the rotation
        if cache is not None:
            k, v = cache.append(layer, k, v)  # old K/V + new K/V
        S = k.shape[2]  # total visible length = past + now
        g = c.n_heads // c.n_kv_heads
        k, v = k.repeat_interleave(g, dim=1), v.repeat_interleave(g, dim=1)  # g query heads share one set
        att = q @ k.transpose(-2, -1) / math.sqrt(c.head_dim)  # (B, H, T, S)
        # Causal mask: the global position of new token i is S-T+i.
        # It can see only the keys at positions ≤ its own position.
        i = torch.arange(T)[:, None] + (S - T)
        j = torch.arange(S)[None, :]
        att = att.masked_fill(j > i, float("-inf")).softmax(-1)
        out = (att @ v).transpose(1, 2).reshape(B, T, -1)
        return self.wo(out)


class Block(nn.Module):
    def __init__(self, c: Config) -> None:
        super().__init__()
        self.n1, self.attn = RMSNorm(c.dim), Attention(c)
        self.n2 = RMSNorm(c.dim)
        self.w_gate = nn.Linear(c.dim, c.ffn_dim, bias=False)
        self.w_up = nn.Linear(c.dim, c.ffn_dim, bias=False)
        self.w_down = nn.Linear(c.ffn_dim, c.dim, bias=False)

    def forward(self, x, cos, sin, cache, layer):
        x = x + self.attn(self.n1(x), cos, sin, cache, layer)
        h = self.n2(x)
        return x + self.w_down(F.silu(self.w_gate(h)) * self.w_up(h))


class TinyLM(nn.Module):
    def __init__(self, c: Config) -> None:
        super().__init__()
        self.c = c
        self.emb = nn.Embedding(c.vocab_size, c.dim)
        nn.init.normal_(self.emb.weight, std=0.02)  # shared embedding: the initial logits must be small
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.n_layers))
        self.norm = RMSNorm(c.dim)
        cos, sin = rope_tables(c.head_dim, c.max_seq_len)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, ids, cache: KVCache | None = None):
        T = ids.shape[1]
        start = len(cache) if cache is not None else 0  # the position of the first new token
        cos, sin = self.cos[start : start + T], self.sin[start : start + T]
        x = self.emb(ids)
        for layer, blk in enumerate(self.blocks):
            x = blk(x, cos, sin, cache, layer)
        return self.norm(x) @ self.emb.weight.T  # shared embedding → logits (B, T, V)


# ── Training (with a weight cache: a later run loads the trained weights) ───
def train(c: Config, steps: int = 600, bsz: int = 16, seq: int = 64, lr: float = 3e-3,
          seed: int = 0, verbose: bool = True):
    data = CharData()
    torch.manual_seed(seed)
    model = TinyLM(c)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
    g = torch.Generator().manual_seed(seed)  # same seed → all variants see the data in the same order
    warm = 100
    t0 = time.time()
    for step in range(steps + 1):
        for pg in opt.param_groups:  # warmup + cosine (Chapter 6)
            pg["lr"] = lr * min(1, (step + 1) / warm) * 0.5 * (1 + math.cos(math.pi * step / steps))
        x, y = data.batch("train", bsz, seq, g)
        loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten())
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if verbose and step % 100 == 0:
            print(f"  step {step:5d}  train loss {loss.item():.3f}  ({time.time() - t0:.0f}s)")
    return model.eval()


@torch.no_grad()
def val_loss(model: TinyLM, n_batches: int = 20, seq: int = 64) -> float:
    data = CharData()
    g = torch.Generator().manual_seed(1234)
    tot = 0.0
    for _ in range(n_batches):
        x, y = data.batch("val", 32, seq, g)
        tot += F.cross_entropy(model(x).flatten(0, 1), y.flatten()).item()
    return tot / n_batches


def load_or_train(n_kv_heads: int = 4, steps: int = 600, seed: int = 0,
                  verbose: bool = True) -> TinyLM:
    c = Config(vocab_size=CharData().vocab_size, n_kv_heads=n_kv_heads)
    path = OUT / (f"tiny_kv{n_kv_heads}_s{steps}" + ("" if seed == 0 else f"_seed{seed}") + ".pt")
    if path.exists():
        model = TinyLM(c)
        model.load_state_dict(torch.load(path, weights_only=True))
        return model.eval()
    if verbose:
        print(f"Training the small model with n_kv_heads={n_kv_heads}, seed {seed} ({steps} steps; "
              f"only once, after that the scripts load {path.name})")
    model = train(c, steps=steps, seed=seed, verbose=verbose)
    OUT.mkdir(exist_ok=True)
    torch.save(model.state_dict(), path)
    return model


if __name__ == "__main__":
    data = CharData()
    model = load_or_train(4)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Config: {asdict(model.c)}, head_dim={model.c.head_dim}")
    print(f"Parameters: {n_params / 1e6:.2f}M, vocabulary: {data.vocab_size} characters")
    print(f"Validation loss {val_loss(model):.3f} nats/character (a uniform random guess gives "
          f"ln {data.vocab_size} = {math.log(data.vocab_size):.3f})")
    # The most basic generation: at each step, give the full sequence to the model,
    # take the logits at the last position, and pick the character with the largest logit.
    ids = data.encode("ROMEO:\n")
    with torch.no_grad():
        for _ in range(120):
            logits = model(torch.tensor([ids]))[0, -1]
            ids.append(int(logits.argmax()))
    print("Greedy generation (prompt 'ROMEO:\\n'):\n" + data.decode(ids))
