"""第 12 章 · 极简代码 4：迷你阶梯实验 —— 用小模型拟合 L(N, D)，外推到没见过的大一号模型，再真的训练它来核对

步骤（和闸门 1 的"外推预测"一模一样，只是缩小了几万倍）：
  1. 读第 3 个脚本的学习率扫描，每个尺寸用自己的最优学习率 η*；
  2. 4 个尺寸 × 3 个 token 预算（WSD 分叉，一次训练得到 3 个点）→ 12 个 (N, D, loss)；
  3. 拟合 L(N, D) = E + A/N^α + B/D^β（网格搜索 α、β，每个格点上 E、A、B 用线性最小二乘解出）；
  4. 按尺寸重采样做 bootstrap，得到外推的 95% 区间；
  5. 用 η*(N) 的幂律外推出大一号模型（N 是阶梯最大的 2.3 倍）的学习率，真的训练它，比较预测与实际。

运行：uv run python chapters/12-scaling-laws/code/04_mini_ladder.py
      （需要先跑 03_lr_sweep.py；单线程约 6–10 分钟，结果缓存在 out/ch12/ladder.json，加 --fresh 重跑）
这是"极小配置演示"：1 万到 50 万参数、几十万字节。它验证的是方法，不是主线模型的任何数字。
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

BUDGETS = [65_536, 131_072, 262_144]  # 128 / 256 / 512 步


# ── 拟合：变量投影。固定 (α, β) 后 L = E + A·x + B·y 对 (E, A, B) 是线性的 ─────────
def fit_lnd(N, D, L, grid=np.arange(0.05, 1.501, 0.01)):
    N, D, L = (np.asarray(v, dtype=float) for v in (N, D, L))
    n, d = N / N.mean(), D / D.mean()  # 先归一化，避免 N^-α 太小让方程病态
    a, b = (g.ravel() for g in np.meshgrid(grid, grid, indexing="ij"))
    X = np.stack([np.ones((len(a), len(L))), n[None] ** -a[:, None], d[None] ** -b[:, None]], axis=2)
    w = 1 / L  # 按相对误差加权
    Xw, Lw = X * w[None, :, None], L * w
    coef = np.linalg.solve(np.einsum("gni,gnj->gij", Xw, Xw), np.einsum("gni,n->gi", Xw, Lw))
    sse = ((Lw[None] - np.einsum("gni,gi->gn", Xw, coef)) ** 2).sum(1)
    sse[(coef < 0).any(1)] = np.inf  # E、A、B 都必须非负
    i = int(np.argmin(sse))
    E, A, B = coef[i]
    return {"E": E, "A": A * N.mean() ** a[i], "B": B * D.mean() ** b[i], "alpha": a[i], "beta": b[i]}


def predict(f, N, D):
    return f["E"] + f["A"] / np.asarray(N, float) ** f["alpha"] + f["B"] / np.asarray(D, float) ** f["beta"]


def bootstrap(points, n_boot=300, seed=0):
    """按尺寸（一组 3 个点，来自同一次训练，彼此相关）有放回地重采样，重新拟合。"""
    rng = np.random.default_rng(seed)
    sizes = sorted({p["N"] for p in points})
    fits = []
    while len(fits) < n_boot:
        pick = rng.choice(len(sizes), len(sizes))
        if len(set(pick)) < 3:  # 少于 3 个不同尺寸时，N 方向的指数定不下来
            continue
        sample = [p for k in pick for p in points if p["N"] == sizes[k]]
        fits.append(fit_lnd(*zip(*[(p["N"], p["D"], p["loss"]) for p in sample])))
    return fits


def run_ladder(fresh=False):
    path = sw.OUT / "ladder.json"
    if path.exists() and not fresh:
        return json.loads(path.read_text())
    sweep = sw.run_sweep()
    best = sw.best_lrs(sweep)
    c, k = sw.fit_power_law([best[s]["N"] for s, _, _ in sw.LADDER], [best[s]["lr"] for s, _, _ in sw.LADDER])
    data, t0, points = sw.load_bytes(), time.time(), []
    for name, dim, L in sw.LADDER:
        N, lr = sw.non_embedding_params(dim, L), best[name]["lr"]
        for D, bpb in sw.train_wsd_branches(dim, L, lr, BUDGETS, data=data).items():
            points.append({"size": name, "N": N, "D": D, "loss": bpb, "lr": lr})
            print(f"  阶梯 {name} N={N:>7,} D={D:>7,} lr={lr:g}: {bpb:.4f}  ({time.time() - t0:4.0f}s)")
    name, dim, L = sw.HELD_OUT
    N5 = sw.non_embedding_params(dim, L)
    lr5 = float(c * N5**-k)  # 学习率也是外推出来的，不对留出模型调参
    held = []
    for D, bpb in sw.train_wsd_branches(dim, L, lr5, BUDGETS, data=data).items():
        held.append({"size": name, "N": N5, "D": D, "loss": bpb, "lr": lr5})
        print(f"  留出 {name} N={N5:>7,} D={D:>7,} lr={lr5:.4g}: {bpb:.4f}  ({time.time() - t0:4.0f}s)")
    result = {"budgets": BUDGETS, "lr_law": {"c": c, "k": k}, "points": points, "held_out": held}
    path.write_text(json.dumps(result, indent=1))
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fresh", action="store_true", help="忽略缓存，重新训练阶梯")
    args = ap.parse_args()
    torch.set_num_threads(1)
    res = run_ladder(args.fresh)
    pts, held = res["points"], res["held_out"]
    f = fit_lnd([p["N"] for p in pts], [p["D"] for p in pts], [p["loss"] for p in pts])
    print(f"\n拟合（12 个阶梯点）：L(N, D) = {f['E']:.3f} + {f['A']:.4g}/N^{f['alpha']:.2f} + {f['B']:.4g}/D^{f['beta']:.2f}")
    print(f"{'尺寸':>4} {'N':>8} {'D':>8} | {'实际':>7} {'拟合':>7} {'误差':>7}")
    for p in pts:
        q = predict(f, p["N"], p["D"])
        print(f"{p['size']:>4} {p['N']:>8,} {p['D']:>8,} | {p['loss']:>7.4f} {q:>7.4f} {(q - p['loss']) / p['loss']:>+7.2%}")
    fits = bootstrap(pts)
    lr_law = res["lr_law"]
    print(f"\n学习率外推：η*(N) = {lr_law['c']:.3g}·N^(-{lr_law['k']:.3f}) → 留出模型用 {held[0]['lr']:.4g}")
    print(f"外推到留出尺寸 {held[0]['size']}（N = {held[0]['N']:,}，阶梯最大的 {held[0]['N'] / max(p['N'] for p in pts):.1f} 倍）：")
    print(f"{'D':>8} | {'预测':>7} {'95% 区间':>15} | {'实际':>7} {'误差':>7}")
    for h in held:
        q = predict(f, h["N"], h["D"])
        boots = [predict(g, h["N"], h["D"]) for g in fits]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        print(f"{h['D']:>8,} | {q:>7.4f} {lo:>7.4f}–{hi:<7.4f} | {h['loss']:>7.4f} {(q - h['loss']) / h['loss']:>+7.2%}")
    C = 6 * max(p["N"] for p in pts) * max(BUDGETS)
    G = (f["alpha"] * f["A"] / (f["beta"] * f["B"])) ** (1 / (f["alpha"] + f["beta"]))
    n_opt = G * (C / 6) ** (f["beta"] / (f["alpha"] + f["beta"]))
    print(f"\n这条拟合说：算力 C = {C:.3g} 时最优是 N ≈ {n_opt:,.0f}、D ≈ {C / 6 / n_opt:,.0f}"
          f"（{C / 6 / n_opt / n_opt:.1f} token/参数）——小尺度、字节级数据上的比例，不能搬到主线模型上。")


if __name__ == "__main__":
    main()
