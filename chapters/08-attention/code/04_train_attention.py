"""Chapter 8 · Minimal code 4: train a one-layer attention model on a small corpus,
and see where it "looks"

Three models. They differ only in one step: how they mix the earlier tokens. All else is the same
(character level, Tiny Shakespeare):

    bigram     h = emb(x) + pos                   only the current character (Chapter 7 bigram + position)
    average    h = h + Wo · uniform_mean(Wv h)    prefix average: sees the context, all characters equal
    attention  h = h + MultiHeadAttention(h)      Q·K sets the weights (this chapter)
    logits = lm_head(h)

Each model trains in about 20–40 seconds on the CPU. After training, the script prints the
validation loss (nats and bits-per-byte). Then it draws the attention matrix of the attention model
on a sample text as a character heat map, and measures where each head looks on average.

Run: uv run python chapters/08-attention/code/04_train_attention.py
"""

from __future__ import annotations

import importlib.util
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

HERE = Path(__file__).resolve().parent
CORPUS = HERE.parents[2] / "assets" / "tiny_corpus" / "shakespeare.txt"

# For a model this small, one thread is fastest. The overhead of thread scheduling is larger
# than the computation, especially when the machine is busy
torch.set_num_threads(1)

_spec = importlib.util.spec_from_file_location("attn02", HERE / "02_attention_from_scratch.py")
attn02 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(attn02)

BLOCK = 64  # context length T
C = 64  # channels
HEADS = 4  # number of heads; d = 16 per head
BATCH = 32
STEPS = 2000
LR = 3e-3
SAMPLE = "First Citizen:\nBefore we proceed any further, hear me speak."


def load_data():
    text = CORPUS.read_text(encoding="utf-8")
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    n = int(0.9 * len(data))
    return chars, stoi, data[:n], data[n:]


def get_batch(data: torch.Tensor, g: torch.Generator):
    ix = torch.randint(len(data) - BLOCK - 1, (BATCH,), generator=g)
    x = torch.stack([data[i : i + BLOCK] for i in ix])
    y = torch.stack([data[i + 1 : i + BLOCK + 1] for i in ix])
    return x, y


class TinyLM(nn.Module):
    def __init__(self, vocab: int, mode: str) -> None:
        super().__init__()
        self.mode = mode
        self.tok = nn.Embedding(vocab, C)
        self.pos = nn.Embedding(BLOCK, C)  # position vectors (Chapter 9 replaces them with RoPE)
        if mode == "attention":
            self.mix = attn02.MultiHeadAttention(C, HEADS)
        elif mode == "average":
            self.wv = nn.Linear(C, C, bias=False)
            self.wo = nn.Linear(C, C, bias=False)
        self.head = nn.Linear(C, vocab)

    def forward(self, idx: torch.Tensor, return_weights: bool = False):
        T = idx.shape[1]
        h = self.tok(idx) + self.pos(torch.arange(T))
        w = None
        if self.mode == "attention":
            out, w = self.mix(h, return_weights=True)
            h = h + out  # residual connection (Chapter 6)
        elif self.mode == "average":
            W = torch.tril(torch.ones(T, T))
            W = W / W.sum(1, keepdim=True)  # fixed uniform weights
            h = h + self.wo(W @ self.wv(h))
        logits = self.head(h)
        return (logits, w) if return_weights else logits


@torch.no_grad()
def evaluate(model: TinyLM, data: torch.Tensor, batches: int = 40) -> float:
    model.eval()
    g = torch.Generator().manual_seed(1234)  # all three models use the same validation batches
    losses = []
    for _ in range(batches):
        x, y = get_batch(data, g)
        losses.append(F.cross_entropy(model(x).flatten(0, 1), y.flatten()).item())
    model.train()
    return sum(losses) / len(losses)


def train(mode: str, steps: int = STEPS, seed: int = 0, log_every: int = 500):
    torch.manual_seed(seed)
    chars, _, train_data, val_data = load_data()
    model = TinyLM(len(chars), mode)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    g = torch.Generator().manual_seed(seed)
    hist = []
    for step in range(steps + 1):
        if step % log_every == 0:
            hist.append((step, evaluate(model, val_data)))
        if step == steps:
            break
        x, y = get_batch(train_data, g)
        loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten())
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model, hist


@torch.no_grad()
def attention_on(model: TinyLM, text: str) -> torch.Tensor:
    """Return the attention weights (H, T, T)."""
    _, stoi, _, _ = load_data()
    idx = torch.tensor([[stoi[c] for c in text]])
    model.eval()
    _, w = model(idx, return_weights=True)
    return w[0]


@torch.no_grad()
def head_profile(model: TinyLM, data: torch.Tensor, batches: int = 20) -> torch.Tensor:
    """How much weight each head puts on average on the position k steps back.

    Returns (H, 4). The columns are k = 0, 1, 2, ≥3.
    """
    model.eval()
    g = torch.Generator().manual_seed(99)
    acc = torch.zeros(HEADS, 4)
    T = BLOCK
    offset = torch.arange(T)[:, None] - torch.arange(T)[None, :]  # query position − key position
    for _ in range(batches):
        x, _ = get_batch(data, g)
        _, w = model(x, return_weights=True)  # (B, H, T, T)
        w = w[:, :, 8:, :].mean(0)  # skip the first 8 positions (their context is too short)
        off = offset[8:]
        for k in range(3):
            acc[:, k] += (w * (off == k)).sum(-1).mean(-1)
        acc[:, 3] += (w * (off >= 3)).sum(-1).mean(-1)
    return acc / batches


def show_char(c: str) -> str:
    return {"\n": "⏎", " ": "␣"}.get(c, c)


def heatmap(w: torch.Tensor, text: str) -> str:
    """Draw (T, T) weights as a character heat map.

    Row = query (current position), column = key (attended position).
    """
    shades = " .:-=+*#%@"
    lines = ["    " + "".join(show_char(c) for c in text)]
    for i, c in enumerate(text):
        row = "".join(
            shades[min(9, int(w[i, j] * 10))] if j <= i else " " for j in range(len(text))
        )
        lines.append(f"  {show_char(c)} {row}")
    return "\n".join(lines)


def main() -> None:
    chars, _, _, val_data = load_data()
    print(
        f"Corpus: Tiny Shakespeare, {len(chars)} distinct characters (all ASCII, 1 character = 1 byte); "
        f"context T={BLOCK}, C={C}, H={HEADS}, training {STEPS} steps × batch {BATCH}\n"
    )
    models, results = {}, []
    for mode in ("bigram", "average", "attention"):
        t0 = time.time()
        model, hist = train(mode)
        dt = time.time() - t0
        models[mode] = model
        final = hist[-1][1]
        n_params = sum(p.numel() for p in model.parameters())
        results.append((mode, n_params, final, final / math.log(2), dt))
        curve = "  ".join(f"{s}:{v:.3f}" for s, v in hist)
        print(f"[{mode:>9}] validation loss (step:nats) {curve}   time {dt:.0f}s")

    print("\n| Model | Parameters | Validation loss (nats/char) | bits-per-byte |")
    print("|---|---:|---:|---:|")
    for mode, n, loss, bpb, _ in results:
        print(f"| {mode} | {n:,} | {loss:.3f} | {bpb:.3f} |")

    model = models["attention"]
    prof = head_profile(model, val_data)
    print("\nAttention model: where each head puts its weight on average (validation set, query positions ≥ 8)")
    print("| Head | Self (k=0) | 1 back | 2 back | Earlier (k≥3) |")
    print("|---|---:|---:|---:|---:|")
    for h in range(HEADS):
        print(f"| {h} | " + " | ".join(f"{v:.2f}" for v in prof[h]) + " |")

    w = attention_on(model, SAMPLE)
    text = SAMPLE
    for h in range(HEADS):
        print(
            f"\nHead {h} on the sample text (row = current character, column = attended character; "
            "denser symbol = more weight, ␣ is a space, ⏎ is a newline)"
        )
        print(heatmap(w[h, 15:, 15:], text[15:]))  # draw only the second line, so the width stays readable
    # Position by position: where heads 1 and 0 put the largest weight in the last 16 characters
    # ("k back" = k positions back)
    for h in (1, 0):
        print(f"\nHead {h} on {text[-16:]!r}: the character with the largest weight at each position:")
        items = []
        for i in range(len(text) - 16, len(text)):
            j = int(torch.argmax(w[h, i, : i + 1]))
            items.append(
                f"{show_char(text[i])}→{i - j} back '{show_char(text[j])}' {float(w[h, i, j]):.2f}"
            )
        for k in range(0, 16, 4):
            print("  " + "   ".join(items[k : k + 4]))


if __name__ == "__main__":
    main()
