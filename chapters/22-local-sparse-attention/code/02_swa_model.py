"""Chapter 22 · Minimal code 2: a small Transformer in which each layer can have a different window, and two tasks.

The structure is the same as in Chapters 9 and 10 (Pre-Norm RMSNorm, RoPE, SwiGLU, tied embeddings).
Only the attention mask changes:
  - windows[l] = None: layer l uses full attention (it sees all of the past).
  - windows[l] = W   : layer l uses a sliding window (it sees only the last W positions, itself included).
  - topk = k (used only in 05): each query keeps only the k keys with the highest scores.
    This is the simplest form of sparse attention.

Three configurations (all with 4 layers):
  full       = [None, None, None, None]
  sliding    = [W, W, W, W]                  (the method of Mistral 7B v0.1)
  interleave = [W, W, W, None]               (3 local : 1 global, the ratio of OLMo 3)

Two tasks:
  - Language modeling: assets/tiny_corpus/shakespeare.txt, character level.
  - Needle in a haystack (needle): a sequence of random "filler characters" hides one "needle" character.
    At the last position, the model must tell which needle it is. The distance d from the needle to the
    last position is uniformly random. Thus we can measure the accuracy for each d and see where the
    sliding window fails.

The weights are cached in code/out/*.pt (.gitignore ignores them). Scripts 03–05 load them directly.
Run: uv run python chapters/22-local-sparse-attention/code/02_swa_model.py   (trains all 6 models: about 12 minutes on an idle CPU, up to 40 minutes on a busy one)
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
torch.set_num_threads(1)  # many jobs share the CPU of the build machine; on your computer, you can remove this line

W = 16  # the same window size for the two tasks
VARIANTS = {
    "full": [None, None, None, None],
    "sliding": [W, W, W, W],
    "interleave": [W, W, W, None],
}


# ── Model ──────────────────────────────────────────────────────────────────
@dataclass
class Config:
    vocab_size: int
    windows: list = field(default_factory=lambda: [None] * 4)  # window of each layer; None = full attention
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
    """(Tq, Tk) boolean mask: the key is not after the query and, if there is a window, the distance is < window."""
    q, k = q_pos[:, None], k_pos[None, :]
    mask = k <= q
    if window is not None:
        mask &= (q - k) < window
    return mask


class KVCache:
    """Minimal KV cache: one list for each layer. After each concatenation, a sliding-window layer keeps
    only the last W positions (truncation). Thus the cache of these layers has a maximum size of W.
    A full-attention layer continues to grow as usual."""

    def __init__(self, n_layers: int) -> None:
        self.k = [None] * n_layers
        self.v = [None] * n_layers
        self.pos = [None] * n_layers  # the global position of each cache slot (to build the mask)

    def append(self, layer: int, k, v, pos, window: int | None):
        if self.k[layer] is not None:
            k = torch.cat([self.k[layer], k], dim=2)
            v = torch.cat([self.v[layer], v], dim=2)
            pos = torch.cat([self.pos[layer], pos])
        # Give all of "old + new" to the attention of this step. Then truncate and store.
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
        self.topk: int | None = None  # 05 sets it for a short time
        self.wqkv = nn.Linear(c.dim, 3 * c.dim, bias=False)
        self.wo = nn.Linear(c.dim, c.dim, bias=False)

    def forward(self, x, cos, sin, pos, cache=None, layer=0):
        B, T, _ = x.shape
        c = self.c
        q, k, v = self.wqkv(x).view(B, T, 3, c.n_heads, c.head_dim).permute(2, 0, 3, 1, 4)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        k_pos = pos
        if cache is not None:  # old K/V + new K/V; for a sliding-window layer, the cache then keeps only the last W
            k, v, k_pos = cache.append(layer, k, v, pos, self.window)
        att = q @ k.transpose(-2, -1) / math.sqrt(c.head_dim)  # (B, H, T, S)
        att = att.masked_fill(~window_mask(pos, k_pos, self.window), float("-inf"))
        if self.topk is not None and self.topk < att.shape[-1]:
            # Sparse attention (simplest form): each query keeps only the k keys with the highest scores; the others become invisible
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


# ── Task 1: character-level language modeling ──────────────────────────────
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


# ── Task 2: needle in a haystack ───────────────────────────────────────────
N_NEEDLE, N_FILL = 8, 32  # vocabulary: 0..7 are "needles", 8..39 are fillers, 40 is the "question"
QUERY = N_NEEDLE + N_FILL
NEEDLE_VOCAB = QUERY + 1
NEEDLE_T = 96  # sequence length: the distance d from the needle to the question is in [1, 95]


def needle_batch(bsz: int, g: torch.Generator, d: torch.Tensor | None = None):
    """Return (x, answer, distance). The last position of x is the "question". The answer is the needle at distance d."""
    x = torch.randint(N_NEEDLE, N_NEEDLE + N_FILL, (bsz, NEEDLE_T), generator=g)
    x[:, -1] = QUERY
    if d is None:
        d = torch.randint(1, NEEDLE_T, (bsz,), generator=g)
    ans = torch.randint(0, N_NEEDLE, (bsz,), generator=g)
    x[torch.arange(bsz), NEEDLE_T - 1 - d] = ans
    return x, ans, d


# ── Training and evaluation ────────────────────────────────────────────────
def train(task: str, windows: list, steps: int, seed: int = 0, verbose: bool = True) -> TinyLM:
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)  # the same seed → the three configurations see exactly the same data
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
        for pg in opt.param_groups:  # warmup + cosine (Chapter 6)
            pg["lr"] = lr * min(1, (step + 1) / 100) * 0.5 * (1 + math.cos(math.pi * step / steps))
        if task == "lm":
            x, y = data.batch("train", bsz, seq, g)
            loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten())
        else:
            x, ans, _ = needle_batch(bsz, g)
            loss = F.cross_entropy(model(x)[:, -1], ans)  # calculate the loss only at the "question" position
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
        print(f"  Train {task} / {variant} ({STEPS[task]} steps, only once; later runs load {path.name})")
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
    """Test n_per_d samples at each distance d = 1..NEEDLE_T-1. Return the accuracy for each d."""
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
            print(f"{task:>6} / {variant:<10} windows={m.c.windows}  parameters {n / 1e3:.0f}K")
