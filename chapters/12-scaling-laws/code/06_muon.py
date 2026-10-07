"""Chapter 12 · Minimal code 6: Muon. "Orthogonalize" the update of each matrix, and compare with AdamW
for the same number of tokens.

For each hidden-layer weight matrix, Muon does this:
    M ← μ·M + G                 (momentum)
    U ← NS5(μ·M + G)            (5 Newton–Schulz steps change G = UΣVᵀ into about UVᵀ:
                                 the step size is the same in all directions)
    W ← W·(1 − ηλ) − η·0.2·√max(m,n)·U   (0.2·√max(m,n) makes the RMS of the update similar to AdamW,
                                          so the AdamW learning rate still works)
The embedding, lm_head, and RMSNorm still use AdamW (the grouping of Kimi K2, GLM-4.5, and DeepSeek-V4).

Experiment: the sizes s2 and s3 of the ladder, with the same 131,072 bytes as script 03. Sweep 3 learning
rates for Muon, and compare with the best tuned AdamW from script 03. This is a "tiny-configuration demo":
results on tens of thousands of parameters do not transfer directly to 0.7B.

Run: uv run python chapters/12-scaling-laws/code/06_muon.py
     (run 03_lr_sweep.py first. Single thread, about 5 min measured on a shared CPU. The results are cached
      in out/ch12/muon.json.)
Production implementation: zero/train/muon.py (MuonAdamW + build_muon_optimizer), tests in tests/test_muon.py.
"""

import argparse
import importlib.util
import json
import math
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("lr_sweep", HERE / "03_lr_sweep.py")
sw = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sw)

MUON_LRS = [2.5e-3, 5e-3, 1e-2]
SIZES = ["s2", "s3"]


def newton_schulz5(G, steps=5):
    """G → about UVᵀ. The coefficients come from the reference implementation of Keller Jordan.

    5 steps move the singular values to about 1.
    """
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G / (G.norm() + 1e-7)  # the iteration converges only when the largest singular value is ≤ 1
    tall = X.shape[0] > X.shape[1]
    if tall:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        X = a * X + (b * A + c * A @ A) @ X
    return X.T if tall else X


class Muon(torch.optim.Optimizer):
    """Minimal version: groups with muon=True use Muon. The other groups use the AdamW formula from torch."""

    def __init__(self, groups, lr, wd=0.1, momentum=0.95):
        super().__init__(groups, dict(lr=lr, wd=wd, momentum=momentum, muon=False))
        self.adam = torch.optim.AdamW([p for g in self.param_groups if not g["muon"] for p in g["params"]],
                                      lr=lr, betas=(0.9, 0.95), weight_decay=0.0)

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            if not g["muon"]:
                continue
            for p in g["params"]:
                buf = self.state[p].setdefault("m", torch.zeros_like(p))
                buf.mul_(g["momentum"]).add_(p.grad)
                U = newton_schulz5(p.grad + g["momentum"] * buf)  # Nesterov
                p.mul_(1 - g["lr"] * g["wd"])
                p.add_(U, alpha=-g["lr"] * 0.2 * math.sqrt(max(p.shape)))
        for ga in self.adam.param_groups:  # the AdamW part follows the schedule with the same learning rate
            ga["lr"] = self.param_groups[0]["lr"]
        self.adam.step()

    def zero_grad(self, set_to_none=True):
        super().zero_grad(set_to_none)

    def state_dict(self):  # copies the optimizer state at a branch
        return {"muon": super().state_dict(), "adam": self.adam.state_dict()}

    def load_state_dict(self, sd):
        super().load_state_dict(sd["muon"])
        self.adam.load_state_dict(sd["adam"])


def make_muon(model, lr):
    hidden = [p for n, p in model.named_parameters() if n.startswith("layers.") and p.ndim == 2]
    other = [p for n, p in model.named_parameters() if not (n.startswith("layers.") and p.ndim == 2)]
    return Muon([{"params": hidden, "muon": True}, {"params": other, "muon": False}], lr=lr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fresh", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(0)
    G = torch.randn(64, 32)
    s = torch.linalg.svdvals(newton_schulz5(G))
    print(f"NS5 check: singular values of a random 64×32 gradient {torch.linalg.svdvals(G / G.norm()).min():.3f}–"
          f"{torch.linalg.svdvals(G / G.norm()).max():.3f} → after orthogonalization {s.min():.3f}–{s.max():.3f}")

    sweep = sw.run_sweep()
    best = sw.best_lrs(sweep)
    path = sw.OUT / "muon.json"
    rows = [] if args.fresh or not path.exists() else json.loads(path.read_text())
    done = {(r["size"], r["lr"]) for r in rows}
    data = sw.load_bytes()
    for name, dim, L in sw.LADDER:
        if name not in SIZES:
            continue
        grid = list(MUON_LRS)
        while True:  # as in 03: if the best value is at an edge of the grid, add a value 2× farther out
            for lr in grid:
                if (name, lr) in done:
                    continue
                bpb = sw.train_wsd_branches(dim, L, lr, [sw.SWEEP_TOKENS], data=data, make_opt=make_muon)[sw.SWEEP_TOKENS]
                rows.append({"size": name, "N": best[name]["N"], "lr": lr, "val_bpb": bpb})
                done.add((name, lr))
                print(f"  Muon {name} lr={lr:g}: {bpb:.4f}", flush=True)
                path.write_text(json.dumps(rows, indent=1))
            mine = sorted((r["lr"], r["val_bpb"]) for r in rows if r["size"] == name)
            top = min(mine, key=lambda t: t[1])[0]
            if top == mine[0][0] and top / 2 >= 1e-4:
                grid = [top / 2]
            elif top == mine[-1][0] and top * 2 <= 0.16:
                grid = [top * 2]
            else:
                break
    print(f"\nSame {sw.SWEEP_TOKENS:,} bytes, validation loss (bit/byte):")
    print(f"{'size':>4} {'N':>8} | {'AdamW best':>14} | {'Muon best':>14} | diff")
    for name in SIZES:
        mu = min((r for r in rows if r["size"] == name), key=lambda r: r["val_bpb"])
        ad = best[name]
        print(f"{name:>4} {ad['N']:>8,} | {ad['val_bpb']:.4f} (η={ad['grid_lr']:<6g}) | {mu['val_bpb']:.4f} (η={mu['lr']:<6g}) | "
              f"{mu['val_bpb'] - ad['val_bpb']:+.4f}")


if __name__ == "__main__":
    main()
