"""Chapter 23 · Minimal code 5: associative recall — pure linear falls behind, the hybrid recovers

Task (a small version of MQAR from Zoology): the first half of the sequence has N random "key value" pairs.
The second half gives some of these keys again and again.
At the next position, the model must give the value of that key:

    k₇ v₃₁  k₂ v₉  k₄₀ v₁₂ … │ k₂ v₉  k₇ v₃₁  k₂ v₉ …
    └── N key-value pairs ──┘ └─ queries: score only the positions after a "key" ─┘

A correct answer needs an **exact** memory of the N key-value pairs.
This is the weak point of a fixed-size state (the capacity table in Section 5).
We compare 4 small 2-layer models (the letters are as in 04_hybrid_lm.py): AA (pure attention),
LL (pure naive linear), GG (pure Gated DeltaNet), GA (1 Gated DeltaNet layer + 1 full-attention layer).
The head_dim of the linear layers is small on purpose (16), so the state capacity is clearly too small.
Then we can see the differences.

Training: 600 steps for each model; each step takes a random N ∈ [4, 24]; the same data order
(on one CPU thread, a few minutes to about 15 min for each model). The weights are cached in code/out/*.pt.
Run: uv run python chapters/23-linear-attention-hybrid/code/05_associative_recall.py
"""

from __future__ import annotations

import importlib.util
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

torch.set_num_threads(1)
HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
_spec = importlib.util.spec_from_file_location("hybrid_lm", HERE / "04_hybrid_lm.py")
lm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lm)

N_KEYS, N_VALS = 64, 64  # keys: 0..63, values: 64..127
PAD = N_KEYS + N_VALS  # 128
VOCAB = PAD + 1
L = 64  # model input length
TRAIN_N = (4, 24)
EVAL_N = [4, 8, 12, 16, 20, 24]
PATTERNS = ["AA", "LL", "GG", "GA"]
DIM, HEADS = 64, 4  # head_dim = 16


def make_batch(bsz: int, N: int, g: torch.Generator):
    """Return (x, y, mask): x and y have the shape (bsz, L).
    mask marks the scored positions (the input is a query key, the target is its value)."""
    keys = torch.argsort(torch.rand(bsz, N_KEYS, generator=g), dim=1)[:, :N]  # N different keys in each sequence
    vals = torch.randint(N_VALS, (bsz, N), generator=g) + N_KEYS
    Q = (L - 2 * N) // 2
    qi = torch.randint(N, (bsz, Q), generator=g)
    qk, qv = keys.gather(1, qi), vals.gather(1, qi)
    pairs = torch.stack([keys, vals], -1).flatten(1)  # k v k v …
    queries = torch.stack([qk, qv], -1).flatten(1)
    seq = torch.full((bsz, L + 1), PAD)
    seq[:, : 2 * N] = pairs
    seq[:, 2 * N : 2 * N + 2 * Q] = queries
    mask = torch.zeros(bsz, L, dtype=torch.bool)
    mask[:, 2 * N : 2 * N + 2 * Q : 2] = True
    return seq[:, :-1], seq[:, 1:], mask


def train(pattern: str, steps: int = 600, bsz: int = 64, lr: float = 3e-3, seed: int = 0,
          verbose: bool = True):
    path = OUT / f"recall_{pattern}_s{steps}_seed{seed}.pt"
    torch.manual_seed(seed)
    model = lm.TinyLM(VOCAB, pattern, dim=DIM, n_heads=HEADS, ffn=2 * DIM, attn_conv=4)
    if path.exists():
        model.load_state_dict(torch.load(path, weights_only=True))
        return model.eval()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    g = torch.Generator().manual_seed(seed)
    t0 = time.time()
    for step in range(steps + 1):
        for pg in opt.param_groups:
            pg["lr"] = lr * min(1, (step + 1) / 100) * 0.5 * (1 + math.cos(math.pi * step / steps))
        N = int(torch.randint(TRAIN_N[0], TRAIN_N[1] + 1, (1,), generator=g))
        x, y, m = make_batch(bsz, N, g)
        loss = F.cross_entropy(model(x)[m], y[m])
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if verbose and step % 250 == 0:
            print(f"  [{pattern}] step {step:4d}  loss {loss.item():.3f}  ({time.time() - t0:.0f}s)")
    OUT.mkdir(exist_ok=True)
    torch.save(model.state_dict(), path)
    return model.eval()


@torch.no_grad()
def accuracy(model, N: int, n: int = 256) -> float:
    x, y, m = make_batch(n, N, torch.Generator().manual_seed(1000 + N))
    return (model(x).argmax(-1)[m] == y[m]).float().mean().item()


def state_numbers(pattern: str) -> str:
    """What each layout stores at inference (input length L=64):
    an A layer stores 2·L·DIM K/V numbers; a linear layer stores H·dh·dh."""
    dh = DIM // HEADS
    parts = []
    for c in pattern:
        parts.append(f"KV {2 * L * DIM}" if c == "A" else f"state {HEADS * dh * dh}")
    return " + ".join(parts)


def results(verbose: bool = False) -> dict:
    acc = {}
    for p in PATTERNS:
        model = train(p, verbose=verbose)
        acc[p] = [accuracy(model, N) for N in EVAL_N]
    return {"patterns": PATTERNS, "N": EVAL_N, "acc": acc}


def main() -> None:
    r = results(verbose=True)
    print(f"\n== Associative recall accuracy (2 layers, width 64, linear head_dim 16; 256 sequences for each N; random guess = 1/{N_VALS}) ==")
    print(f"{'Arch':>4} | " + " | ".join(f"N={n:>2}" for n in EVAL_N) + " | numbers stored per layer at inference")
    for p in PATTERNS:
        print(f"{p:>4} | " + " | ".join(f"{a:>4.0%}" for a in r["acc"][p]) + f" | {state_numbers(p)}")


if __name__ == "__main__":
    main()
