"""Chapter 12 · Minimal code 4: a mini ladder experiment. Fit L(N, D) with small models, extrapolate to
a larger model that the fit did not see, then train that model to check the prediction.

The steps (the same as the "extrapolated prediction" of gate 1, but tens of thousands of times smaller):
  1. Read the learning-rate sweep of the third script. Each size uses its own best learning rate η*.
  2. 4 sizes × 3 token budgets (WSD branches: one training run gives 3 points). The two small sizes get
     one more "overtrained" budget → 14 points.
  3. Fit L(N, D) = E + A/N^α + B/D^β (a grid search over α and β; at each grid point, linear least
     squares gives E, A, and B).
  4. Resample by size for a bootstrap. This gives the 95% interval of the extrapolation.
  5. Use the power law η*(N) to extrapolate the learning rate of a larger model (N is 2.1× the largest
     ladder size). Train that model and compare the prediction with the result.
     As a control, do one more run with the learning rate of the largest ladder size. This shows how much
     error the learning-rate extrapolation itself causes.

Run: uv run python chapters/12-scaling-laws/code/04_mini_ladder.py
     (run 03_lr_sweep.py first. Single thread, about 15 min measured on a shared CPU. The results are cached
      in out/ch12/ladder.json. Add --fresh to train again.)
This is a "tiny-configuration demo": 10K to 500K parameters and a few hundred thousand bytes.
It tests the method. It gives no numbers for the main-line model.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("lr_sweep", HERE / "03_lr_sweep.py")
sw = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sw)

BUDGETS = [65_536, 131_072, 262_144]  # 128 / 256 / 512 steps
# Overtrained region: the main-line model has about 600 tokens/parameter. The ladder must have points with
# much more than 20 tokens/parameter. Without them, the fit cannot find the exponent for N.
# The two smallest sizes get one more budget of 1M bytes (about 100 and 33 tokens/parameter). It is cheap.
EXTRA = {"s1": [1_048_576], "s2": [1_048_576]}


# ── Fit: variable projection. For a fixed (α, β), L = E + A·x + B·y is linear in (E, A, B) ─────────
GRID = np.arange(0.02, 1.501, 0.02)  # search grid for α and β


def fit_lnd(N, D, L, grid=GRID):
    N, D, L = (np.asarray(v, dtype=float) for v in (N, D, L))
    n, d = N / N.mean(), D / D.mean()  # normalize first: a very small N^-α makes the equations ill-conditioned
    a, b = (g.ravel() for g in np.meshgrid(grid, grid, indexing="ij"))
    X = np.stack([np.ones((len(a), len(L))), n[None] ** -a[:, None], d[None] ** -b[:, None]], axis=2)
    w = 1 / L  # weight by the relative error
    Xw, Lw = X * w[None, :, None], L * w
    coef = np.linalg.solve(np.einsum("gni,gnj->gij", Xw, Xw), np.einsum("gni,n->gi", Xw, Lw)[..., None])[..., 0]
    sse = ((Lw[None] - np.einsum("gni,gi->gn", Xw, coef)) ** 2).sum(1)
    sse[(coef < 0).any(1)] = np.inf  # E, A, and B must all be non-negative
    i = int(np.argmin(sse))
    E, A, B = coef[i]
    return {"E": E, "A": A * N.mean() ** a[i], "B": B * D.mean() ** b[i], "alpha": a[i], "beta": b[i]}


def predict(f, N, D):
    return f["E"] + f["A"] / np.asarray(N, float) ** f["alpha"] + f["B"] / np.asarray(D, float) ** f["beta"]


def bootstrap(points, n_boot=200, seed=0):
    """Resample by size with replacement and fit again.

    The points of one size come from the same training run, so they are correlated.
    """
    rng = np.random.default_rng(seed)
    sizes = sorted({p["N"] for p in points})
    fits = []
    while len(fits) < n_boot:
        pick = rng.choice(len(sizes), len(sizes))
        if len(set(pick)) < 3:  # with fewer than 3 different sizes, the fit cannot find the exponent for N
            continue
        sample = [p for k in pick for p in points if p["N"] == sizes[k]]
        fits.append(fit_lnd(*zip(*[(p["N"], p["D"], p["loss"]) for p in sample])))
    return fits


def run_ladder(fresh=False):
    path = sw.OUT / "ladder.json"
    result = None if fresh or not path.exists() else json.loads(path.read_text())
    sweep = sw.run_sweep()
    best = sw.best_lrs(sweep)
    data, t0 = sw.load_bytes(), time.time()
    if result is None:
        c, k = sw.fit_power_law([best[s]["N"] for s, _, _ in sw.LADDER], [best[s]["lr"] for s, _, _ in sw.LADDER])
        points = []
        for name, dim, L in sw.LADDER:
            N, lr = sw.non_embedding_params(dim, L), best[name]["lr"]
            for D, bpb in sw.train_wsd_branches(dim, L, lr, BUDGETS, data=data).items():
                points.append({"size": name, "N": N, "D": D, "loss": bpb, "lr": lr})
                print(f"  ladder {name} N={N:>7,} D={D:>9,} lr={lr:.4g}: {bpb:.4f}  ({time.time() - t0:4.0f}s)", flush=True)
        name, dim, L = sw.HELD_OUT
        N5 = sw.non_embedding_params(dim, L)
        lr5 = float(c * N5**-k)  # the learning rate is also extrapolated: do not tune the held-out model
        lr_alt = best[sw.LADDER[-1][0]]["lr"]  # control: use the learning rate of the largest ladder size
        runs = {}
        for tag, lr in [("law", lr5), ("reuse", lr_alt)]:
            runs[tag] = []
            for D, bpb in sw.train_wsd_branches(dim, L, lr, BUDGETS, data=data).items():
                runs[tag].append({"size": name, "N": N5, "D": D, "loss": bpb, "lr": lr})
                print(f"  held-out {name} N={N5:>7,} D={D:>9,} lr={lr:.4g}: {bpb:.4f}  ({time.time() - t0:4.0f}s)", flush=True)
        result = {"budgets": BUDGETS, "lr_law": {"c": c, "k": k}, "points": points,
                  "held_out": runs["law"], "held_out_reuse_lr": runs["reuse"]}
        path.write_text(json.dumps(result, indent=1))
    have = {(p["size"], p["D"]) for p in result["points"]}
    for name, dim, L in sw.LADDER:  # extra points in the overtrained region (they can run later; old points stay)
        todo = [D for D in EXTRA.get(name, []) if (name, D) not in have]
        if todo:
            N, lr = sw.non_embedding_params(dim, L), best[name]["lr"]
            for D, bpb in sw.train_wsd_branches(dim, L, lr, todo, data=data).items():
                result["points"].append({"size": name, "N": N, "D": D, "loss": bpb, "lr": lr})
                print(f"  ladder {name} N={N:>7,} D={D:>9,} lr={lr:.4g}: {bpb:.4f}  ({time.time() - t0:4.0f}s)", flush=True)
            path.write_text(json.dumps(result, indent=1))
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fresh", action="store_true", help="ignore the cache and train the ladder again")
    args = ap.parse_args()
    torch.set_num_threads(1)
    res = run_ladder(args.fresh)
    pts, held = res["points"], res["held_out"]
    f = fit_lnd([p["N"] for p in pts], [p["D"] for p in pts], [p["loss"] for p in pts])
    print(f"\nFit ({len(pts)} ladder points): L(N, D) = {f['E']:.3f} + {f['A']:.4g}/N^{f['alpha']:.2f} + {f['B']:.4g}/D^{f['beta']:.2f}")
    print(f"{'size':>4} {'N':>8} {'D':>9} {'D/N':>5} | {'actual':>7} {'fit':>7} {'error':>7}")
    for p in sorted(pts, key=lambda p: (p["N"], p["D"])):
        q = predict(f, p["N"], p["D"])
        print(f"{p['size']:>4} {p['N']:>8,} {p['D']:>9,} {p['D'] / p['N']:>5.0f} | {p['loss']:>7.4f} {q:>7.4f} "
              f"{(q - p['loss']) / p['loss']:>+7.2%}")
    fits = bootstrap(pts)
    lr_law = res["lr_law"]
    print(f"\nLearning-rate extrapolation: η*(N) = {lr_law['c']:.3g}·N^(-{lr_law['k']:.3f}) → the held-out model uses {held[0]['lr']:.4g}")
    print(f"Extrapolate to the held-out size {held[0]['size']} (N = {held[0]['N']:,}, {held[0]['N'] / max(p['N'] for p in pts):.1f}× the largest ladder size):")
    print(f"{'D':>8} | {'pred.':>7} {'95% interval':>15} | {'actual':>7} {'error':>7}")
    for h, h2 in zip(held, res["held_out_reuse_lr"]):
        q = predict(f, h["N"], h["D"])
        boots = [predict(g, h["N"], h["D"]) for g in fits]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        print(f"{h['D']:>8,} | {q:>7.4f} {lo:>7.4f}–{hi:<7.4f} | {h['loss']:>7.4f} {(q - h['loss']) / h['loss']:>+7.2%}"
              f"   (with the s4 learning rate {h2['lr']:.4g}: {h2['loss']:.4f}, error {(q - h2['loss']) / h2['loss']:+.2%})")
    C = 6 * max(p["N"] for p in pts) * max(BUDGETS)
    G = (f["alpha"] * f["A"] / (f["beta"] * f["B"])) ** (1 / (f["alpha"] + f["beta"]))
    n_opt = G * (C / 6) ** (f["beta"] / (f["alpha"] + f["beta"]))
    print(f"\nThis fit says: at compute C = {C:.3g}, the optimum is N ≈ {n_opt:,.0f}, D ≈ {C / 6 / n_opt:,.0f}"
          f" ({C / 6 / n_opt / n_opt:.1f} tokens/parameter). This ratio is for small models on byte-level data. "
          "Do not use it for the main-line model.")


if __name__ == "__main__":
    main()
