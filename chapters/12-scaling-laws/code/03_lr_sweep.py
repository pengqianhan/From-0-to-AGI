"""Chapter 12 · Minimal code 3: the first step of a ladder experiment. Find a good learning rate for each size.

"Small experiments can predict large results" is true only when each small model has well-tuned
hyperparameters. Badly tuned small models distort the scaling law (Lourie et al. 2026, arXiv:2608.11859).
Thus the ladder experiment starts with a learning-rate sweep:

  1. 4 sizes (about 10K → 200K non-embedding parameters) × 5 learning rates. Each run uses the same number of tokens.
  2. For each size, take the learning rate η* with the lowest validation loss.
  3. Fit η*(N) = c · N^(-k) (a straight line on log axes). Extrapolate it to larger models (the fourth script).

The model is the TinyTransformer from Chapter 9 (byte-level, vocabulary of 256).
The data is assets/tiny_corpus/shakespeare.txt.
The learning-rate schedule is WSD (Chapter 6): warmup → constant → linear decay to 0 in the last 20%.

Run: uv run python chapters/12-scaling-laws/code/03_lr_sweep.py
     (single thread, about 7 min measured on a shared CPU. The results are cached in out/ch12/lr_sweep.json.
      The next run reads the cache. Add --fresh to train again.)
This is a "tiny-configuration demo": tens of thousands of parameters and about 130K tokens.
The results show only the method. They do not represent the main-line model.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "out" / "ch12"
_spec = importlib.util.spec_from_file_location(
    "tiny_tf", ROOT / "chapters/09-modern-transformer/code/02_tiny_transformer.py"
)
tt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tt)

SEQ_LEN = 64
BATCH = 8  # small batch: more steps for the same tokens; at this scale, the model learns faster (measured)
TOKENS_PER_STEP = SEQ_LEN * BATCH  # 512 bytes per step
WARMUP = 20

# Ladder: (name, dim, number of layers). head_dim is always 16.
# The SwiGLU hidden size is about 8/3·dim, rounded to a multiple of 16.
LADDER = [("s1", 16, 2), ("s2", 32, 2), ("s3", 48, 3), ("s4", 64, 4)]
HELD_OUT = ("s5", 96, 4)  # N is 2.1× the largest ladder size; use it only to test the extrapolation
SWEEP_LRS = [2.5e-3, 5e-3, 1e-2, 2e-2, 4e-2]
SWEEP_TOKENS = 131_072  # 256 steps


def make_config(dim: int, n_layers: int):
    ffn = 16 * round(8 / 3 * dim / 16)
    return tt.Config(dim=dim, n_layers=n_layers, n_heads=dim // 16, ffn_dim=ffn, seq_len=SEQ_LEN)


def make_model(dim: int, n_layers: int):
    """The model from Chapter 9 with one change: the input embedding and lm_head do not share weights.

    At this scale, shared weights keep the model for a long time on a plateau where it only
    predicts frequent bytes (measured). The curves then have too much noise to fit a law.
    The tiny configuration of zero does not share them for the same reason
    (see the comment in configs/tiny/pretrain.toml).
    """
    model = tt.TinyTransformer(make_config(dim, n_layers))
    model.lm_head.weight = nn.Parameter(torch.randn(model.cfg.vocab_size, dim) * 0.02)
    return model


def non_embedding_params(dim: int, n_layers: int) -> int:
    """N in the scaling law: the parameters in matrix products.

    This is all parameters except the input embedding lookup, and it includes lm_head.
    Then 6N is the training FLOPs per token (without the attention term), as in 01_flops.py.
    """
    model = make_model(dim, n_layers)
    return sum(p.numel() for p in model.parameters()) - model.tok_emb.weight.numel()


def load_bytes():
    raw = (ROOT / "assets/tiny_corpus/shakespeare.txt").read_bytes()
    data = torch.tensor(list(raw), dtype=torch.long)
    n = int(0.9 * len(data))
    return data[:n], data[n:]


def batch(data, g):
    ix = torch.randint(len(data) - SEQ_LEN - 1, (BATCH,), generator=g)
    x = torch.stack([data[i : i + SEQ_LEN] for i in ix])
    y = torch.stack([data[i + 1 : i + SEQ_LEN + 1] for i in ix])
    return x, y


@torch.no_grad()
def val_bpb(model, val, n_batches=24) -> float:
    """Validation bits per byte (24 fixed batches; all runs use the same batches)."""
    g = torch.Generator().manual_seed(123)
    model.eval()
    losses = [F.cross_entropy(model(x).flatten(0, 1), y.flatten()).item()
              for x, y in (batch(val, g) for _ in range(n_batches))]
    model.train()
    return float(np.mean(losses) / math.log(2))


def train_step(model, opt, x, y, lr):
    for group in opt.param_groups:
        group["lr"] = lr
    loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten())
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()


def adamw(model, lr):
    return torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)


def train_wsd_branches(dim, n_layers, lr, budgets, seed=0, data=None, make_opt=adamw):
    """Get the results of several token budgets from one training run (WSD branches).

    Trunk: after warmup, keep the peak learning rate until 0.8 × the largest budget.
    For each budget D_k: at 0.8·D_k on the trunk, copy the model and the optimizer, add a linear
    decay of 0.2·D_k, and measure the validation loss after the decay. Then 3 budgets cost about
    1.15× the compute of the largest budget, not 1.75×.
    """
    train, val = data or load_bytes()
    torch.manual_seed(seed)
    model = make_model(dim, n_layers)
    opt = make_opt(model, lr)
    g = torch.Generator().manual_seed(seed)
    steps = {D: D // TOKENS_PER_STEP for D in budgets}
    branch_at = {int(0.8 * s): D for D, s in steps.items()}
    trunk_end = max(branch_at)
    results = {}
    for step in range(trunk_end + 1):
        if step in branch_at:  # branch: copy the current state and add a decay
            D = branch_at[step]
            m2 = copy.deepcopy(model)
            o2 = make_opt(m2, lr)
            o2.load_state_dict(copy.deepcopy(opt.state_dict()))
            g2 = torch.Generator().manual_seed(seed * 1000 + D % 997)
            n_decay = steps[D] - step
            for i in range(n_decay):
                train_step(m2, o2, *batch(train, g2), lr * (1 - (i + 1) / n_decay))
            results[D] = val_bpb(m2, val)
        if step == trunk_end:
            break
        warm = min(1.0, (step + 1) / WARMUP)
        train_step(model, opt, *batch(train, g), lr * warm)
    return results


def fit_power_law(x, y):
    """Least squares for log y = log c − k·log x. Returns (c, k)."""
    slope, intercept = np.polyfit(np.log(x), np.log(y), 1)
    return float(np.exp(intercept)), float(-slope)


def run_sweep(fresh=False):
    """For each size, sweep SWEEP_LRS first. If the best value is at an edge of the grid, add one value
    2× farther out on that side. Continue until the best point is inside the grid.

    (A best value at the edge means that the true optimum is outside the grid. This is the most common
    mistake in tuning.) The cache stores each result, so an interrupted run can continue.
    """
    path = OUT / "lr_sweep.json"
    rows = [] if fresh or not path.exists() else json.loads(path.read_text())["rows"]
    done = {(r["size"], r["lr"]) for r in rows}
    data, t0 = None, time.time()
    for name, dim, L in LADDER:
        n = non_embedding_params(dim, L)
        grid = list(SWEEP_LRS)
        while True:
            for lr in grid:
                if (name, lr) in done:
                    continue
                data = data or load_bytes()
                bpb = train_wsd_branches(dim, L, lr, [SWEEP_TOKENS], data=data)[SWEEP_TOKENS]
                rows.append({"size": name, "dim": dim, "layers": L, "N": n, "lr": lr, "val_bpb": bpb})
                done.add((name, lr))
                print(f"  {name} N={n:>7,}  lr={lr:<8g} val {bpb:.4f} bit/byte  ({time.time() - t0:4.0f}s)")
                OUT.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps({"tokens": SWEEP_TOKENS, "rows": rows}, indent=1))
            mine = sorted((r["lr"], r["val_bpb"]) for r in rows if r["size"] == name)
            lrs = [lr for lr, _ in mine]
            best = min(mine, key=lambda t: t[1])[0]
            if best == lrs[0] and best / 2 >= 1e-4:
                grid = [best / 2]
            elif best == lrs[-1] and best * 2 <= 0.16:
                grid = [best * 2]
            else:
                break
    return {"tokens": SWEEP_TOKENS, "rows": rows}


def best_lrs(sweep):
    """The best learning rate of each size.

    The grid has steps of 2×, which is too coarse. Take the lowest point and its two neighbors,
    fit a parabola in log(η), and take its vertex.
    """
    best = {}
    for name in dict.fromkeys(r["size"] for r in sweep["rows"]):
        mine = sorted((r for r in sweep["rows"] if r["size"] == name), key=lambda r: r["lr"])
        i = min(range(len(mine)), key=lambda j: mine[j]["val_bpb"])
        row = dict(mine[i])
        row["grid_lr"] = row["lr"]
        if 0 < i < len(mine) - 1:
            x = np.log([mine[j]["lr"] for j in (i - 1, i, i + 1)])
            y = [mine[j]["val_bpb"] for j in (i - 1, i, i + 1)]
            a, b, _ = np.polyfit(x, y, 2)
            row["lr"] = float(np.exp(np.clip(-b / (2 * a), x[0], x[2])))
        best[name] = row
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fresh", action="store_true", help="ignore the cache and train again")
    args = ap.parse_args()
    torch.set_num_threads(1)  # one thread is fastest on a shared CPU; remove this line on an idle computer
    print(f"Learning-rate sweep: {SWEEP_TOKENS:,} bytes (tokens) per run, batch {BATCH}×{SEQ_LEN}")
    sweep = run_sweep(args.fresh)
    table = {}
    for r in sweep["rows"]:
        table.setdefault((r["size"], r["N"]), {})[r["lr"]] = r["val_bpb"]
    all_lrs = sorted({r["lr"] for r in sweep["rows"]})
    print("\nValidation loss (bit/byte). One row per size. * = best for that size, - = not run:")
    print(f"{'size':>4} {'N':>8} | " + " ".join(f"{lr:>8g}" for lr in all_lrs))
    best = best_lrs(sweep)
    for (name, n), row in table.items():
        cells = [f"{row[lr]:>7.4f}{'*' if lr == best[name]['grid_lr'] else ' '}" if lr in row else f"{'-':>8}"
                 for lr in all_lrs]
        print(f"{name:>4} {n:>8,} | " + " ".join(cells))
    print("η* after parabola interpolation: " + ", ".join(f"{s} {best[s]['lr']:.4g}" for s, _, _ in LADDER))
    ns = [best[s]["N"] for s, _, _ in LADDER]
    lrs = [best[s]["lr"] for s, _, _ in LADDER]
    c, k = fit_power_law(ns, lrs)
    n5 = non_embedding_params(*HELD_OUT[1:])
    print(f"\nFit η*(N) = {c:.3g} · N^(-{k:.3f})")
    print(f"Extrapolate to the held-out size {HELD_OUT[0]} (N={n5:,}): η* ≈ {c * n5 ** -k:.4g}")
    fixed = best[LADDER[0][0]]["grid_lr"]
    print(f"\nIf all sizes use the learning rate {fixed:g} from the smallest model (tuning only on the small model):")
    for (name, n), row in table.items():
        print(f"  {name} N={n:>7,}  {row[fixed]:.4f}  worse than tuned by {row[fixed] - best[name]['val_bpb']:+.4f}")


if __name__ == "__main__":
    main()
