"""阶梯实验的分析（runs/ladder-3090/README.md）。输入是 sweep.py collect 写出的 results/<档>.csv。

    uv run python runs/ladder-3090/analyze.py frontier --scale e5m      # 分数分布、最好的配置、逐轮淘汰的代价
    uv run python runs/ladder-3090/analyze.py nql --scale e5m           # noisy quadratic limit：调够了没有
    uv run python runs/ladder-3090/analyze.py sensitivity --scale e5m   # 每个超参数 vs 最终 val_bpb
    uv run python runs/ladder-3090/analyze.py fit --fit e5m e11m e24m e44m --holdout e83m   # L(N, D) 与外推

图写到 runs/ladder-3090/figures/，数字写到 runs/ladder-3090/results/*.json。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
HERE = REPO / "runs" / "ladder-3090"
RES, FIG = HERE / "results", HERE / "figures"
HP = ["tokens_per_step", "lr", "beta1", "beta2", "warmup_frac", "weight_decay", "rope_theta"]
LOG_HP = {"tokens_per_step", "lr", "warmup_frac", "weight_decay", "rope_theta"}

# 图的样式（dataviz 规范：浅色底、淡网格、细线；配色经 validate_palette.js 验证）
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#1f1f1e", "#6b6b68", "#e6e6e3"
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#b4b4b0"
SIZE_RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]  # 有序：小 → 大（--ordinal 通过）


def plt_setup():  # noqa: ANN201
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK, "text.color": INK, "xtick.color": MUTED,
        "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.6,
        "font.size": 9, "legend.frameon": False,
        "font.sans-serif": ["Noto Sans CJK JP", "Noto Sans CJK SC", "WenQuanYi Zen Hei", "DejaVu Sans"],
        "axes.unicode_minus": False,
    })
    return plt


def load(scale: str) -> list[dict]:
    path = RES / f"{scale}.csv"
    if not path.exists():
        raise SystemExit(f"{path} 不存在：先跑 sweep.py collect --scale {scale}")
    rows = []
    with path.open() as f:
        for r in csv.DictReader(f):
            for k in [*HP, "n_eff", "val_bpb", "tokens", "steps", "k"]:
                r[k] = float(r[k])
            r["k"] = int(r["k"])
            rows.append(r)
    return rows


def scores(rows: list[dict], phase: str, k: int) -> dict[str, float]:
    return {r["id"]: r["val_bpb"] for r in rows if r["phase"] == phase and r["k"] == k}


def configs(rows: list[dict]) -> dict[str, dict]:
    return {r["id"]: {h: r[h] for h in HP} for r in rows}


# ---------------------------------------------------------------- frontier


def cmd_frontier(args: argparse.Namespace) -> None:
    rows = load(args.scale)
    cfg = configs(rows)
    final = scores(rows, "decay", 8)
    n_all = sum(1 for line in (HERE / "configs" / f"{args.scale}.jsonl").read_text().splitlines() if line.strip())
    ys = np.array(sorted(final.values()))
    best = min(final, key=final.get)
    out = {
        "scale": args.scale, "configs": n_all, "finished": len(final),
        "best_id": best, "best_val_bpb": final[best], "best_config": cfg[best],
        "quantiles": {q: float(np.quantile(ys, q)) for q in (0.0, 0.1, 0.25, 0.5)},
    }
    print(f"{args.scale}：{n_all} 组里 {len(final)} 组训完（其余发散或未跑）")
    print(f"最好 {best}：val_bpb {final[best]:.4f}  " + "  ".join(f"{h}={cfg[best][h]:.3g}" for h in HP))
    print("分位数：" + "  ".join(f"{q:.0%} {v:.4f}" for q, v in out["quantiles"].items()))

    # 逐轮淘汰的代价：所有配置都训满了，就能模拟"如果按 2/8、4/8 处的稳定段成绩淘汰，会选出谁"
    s2, s4 = scores(rows, "stable", 2), scores(rows, "stable", 4)
    if s2 and s4:
        def top(ids: list[str], score: dict[str, float], frac: float) -> list[str]:
            ranked = sorted(ids, key=lambda i: score.get(i, math.inf))
            return ranked[: max(1, round(frac * len(ids)))]

        ids = [i for i in cfg if i in s2]
        r1 = top(ids, s2, 0.5)
        r2 = top(r1, s4, 0.5)
        sh_best = min((i for i in r2 if i in final), key=final.get)
        top5 = sorted(final, key=final.get)[:5]
        out["successive_halving"] = {
            "survivors_round1": len(r1), "survivors_round2": len(r2), "picked": sh_best,
            "picked_val_bpb": final[sh_best], "regret_bpb": final[sh_best] - final[best],
            "regret_rel": final[sh_best] / final[best] - 1,
            "global_top5_survived": sum(i in r2 for i in top5),
        }
        sh = out["successive_halving"]
        print(f"逐轮淘汰模拟：{len(ids)} → {len(r1)} → {len(r2)} 组，选中 {sh_best}（{final[sh_best]:.4f}），"
              f"比真正最好的差 {sh['regret_bpb']:.4f}（{sh['regret_rel']:.2%}）；全局前 5 名留下 {sh['global_top5_survived']} 个")
    RES.mkdir(exist_ok=True)
    (RES / f"{args.scale}_frontier.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))


# ---------------------------------------------------------------- noisy quadratic limit


def cmd_nql(args: argparse.Namespace) -> None:
    from opda.parametric import NoisyQuadraticDistribution

    rows = load(args.scale)
    plt = plt_setup()
    targets = [("decay", 8, "衰减后 8/8（完整训练）")] + [("stable", k, f"稳定段 {k}/8") for k in (4, 8)]
    out = {}
    fig, axes = plt.subplots(len(args.tails), len(targets), figsize=(3.4 * len(targets), 2.8 * len(args.tails)),
                             squeeze=False, sharey="row")
    for j, (phase, k, title) in enumerate(targets):
        ys = np.array(sorted(scores(rows, phase, k).values()))
        if len(ys) < 20:
            continue
        for i, tail in enumerate(args.tails):
            thr = float(np.quantile(ys, tail))
            fit = NoisyQuadraticDistribution.fit(
                ys, limits=(-np.inf, thr), constraints={"c": (1, len(HP)), "convex": True},
                generator=np.random.default_rng(0),
            )
            in_tail = ys[ys <= thr]
            ecdf = np.arange(1, len(in_tail) + 1) / len(ys)
            ks = float(np.max(np.abs(fit.cdf(in_tail) - ecdf)))
            key = f"{phase}_{k}_tail{tail:g}"
            out[key] = {"n": len(ys), "n_tail": len(in_tail), "threshold": thr, "a_best": float(fit.a),
                        "b": float(fit.b), "c_effective_hp": float(fit.c), "o_noise": float(fit.o), "ks_tail": ks}
            print(f"{title:14s} 尾部 {tail:>4.0%}（{len(in_tail):3d} 点）：最好可达 a={fit.a:.4f}，"
                  f"有效超参数个数 c={fit.c:.2f}，噪声 o={fit.o:.4f}，尾部 KS={ks:.3f}")
            ax = axes[i][j]
            grid = np.linspace(in_tail.min() - 0.01, thr, 200)
            ax.step(in_tail, ecdf, where="post", color=BLUE, label="随机搜索（经验分布）")
            ax.plot(grid, fit.cdf(grid), color=ORANGE, label="noisy quadratic 拟合")
            ax.set_title(f"{title}，最好的 {tail:.0%}", fontsize=9)
            ax.set_xlabel("val_bpb")
            if j == 0:
                ax.set_ylabel("累积比例")
    axes[0][0].legend(loc="upper left", fontsize=8)
    fig.suptitle(f"{args.scale}：分数分布的尾部与 noisy quadratic limit（{len(ys)} 组配置）", fontsize=10)
    fig.tight_layout()
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / f"{args.scale}_nql.png", dpi=150)
    (RES / f"{args.scale}_nql.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"图：{FIG / f'{args.scale}_nql.png'}")


# ---------------------------------------------------------------- 敏感性


def cmd_sensitivity(args: argparse.Namespace) -> None:
    rows = load(args.scale)
    cfg = configs(rows)
    final = scores(rows, "decay", 8)
    ids = sorted(final, key=final.get)
    good = set(ids[: max(1, len(ids) // 4)])
    plt = plt_setup()
    fig, axes = plt.subplots(2, 4, figsize=(13, 5.6), sharey=True)
    lo, hi = final[ids[0]], float(np.quantile(list(final.values()), 0.9))
    for ax, h in zip(axes.flat, HP):
        for group, color, label in ((False, GRAY, "其余配置"), (True, BLUE, "最好的 25%")):
            xs = [cfg[i][h] for i in ids if (i in good) == group]
            ys = [final[i] for i in ids if (i in good) == group]
            ax.scatter(xs, ys, s=16, color=color, label=label, edgecolors=SURFACE, linewidths=0.6)
        if h in LOG_HP:
            ax.set_xscale("log", base=2 if h in ("tokens_per_step", "rope_theta") else 10)
        ax.set_xlabel(h)
        ax.set_ylim(lo - 0.02 * (hi - lo), hi)
    axes.flat[-1].axis("off")
    axes[0][0].set_ylabel("最终 val_bpb（越低越好）")
    axes[1][0].set_ylabel("最终 val_bpb（越低越好）")
    axes[0][0].legend(loc="upper right", fontsize=8)
    fig.suptitle(f"{args.scale}：每个超参数与最终 val_bpb（纵轴截在 90% 分位，发散的配置不显示）", fontsize=10)
    fig.tight_layout()
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / f"{args.scale}_sensitivity.png", dpi=150)
    print(f"图：{FIG / f'{args.scale}_sensitivity.png'}")


# ---------------------------------------------------------------- scaling law


def frontier_points(scale: str, kmin: int = 3) -> list:
    """每个 (N, D) 预算下的最好成绩：衰减后的点（D = 实际训练的 token 数）；丢掉 1/8、2/8 两个最短的预算（论文的做法）。"""
    from zero.tools.fit_scaling import Point

    rows = load(scale)
    pts = []
    for k in range(kmin, 9):
        cand = [r for r in rows if r["phase"] == "decay" and r["k"] == k]
        if cand:
            b = min(cand, key=lambda r: r["val_bpb"])
            pts.append(Point(N=b["n_eff"], D=b["tokens"], loss=b["val_bpb"], name=f"{scale}/{b['id']}/k{k}", group=scale))
    return pts


def cmd_fit(args: argparse.Namespace) -> None:
    from zero.tools.fit_scaling import bootstrap_chinchilla, fit_chinchilla

    fit_pts = [p for s in args.fit for p in frontier_points(s)]
    hold_pts = [p for s in args.holdout for p in frontier_points(s)]
    law = fit_chinchilla(fit_pts, param_count="flops", tie_exponents=args.tie)
    boot = bootstrap_chinchilla(fit_pts, n=args.boot, param_count="flops", tie_exponents=args.tie)

    def rel_err(pts: list) -> list[float]:
        return [float(law.predict(p.N, p.D) / p.loss - 1) for p in pts]

    out = {"law": law.to_dict(), "fit_scales": args.fit, "holdout_scales": args.holdout,
           "fit_points": len(fit_pts), "fit_max_rel_err": max(map(abs, rel_err(fit_pts)))}
    print(f"拟合：{law.to_dict()}")
    print(f"拟合点 {len(fit_pts)} 个，最大相对误差 {out['fit_max_rel_err']:.2%}")
    if hold_pts:
        errs = rel_err(hold_pts)
        out["holdout_rel_err"] = {p.name: e for p, e in zip(hold_pts, errs)}
        out["holdout_max_rel_err"] = max(map(abs, errs))
        print(f"留出 {args.holdout}：最大相对误差 {out['holdout_max_rel_err']:.2%}（闸门标准 < 1%）")
    # 外推到主线（两种序列长度口径都给；阶梯是 2048）
    from zero.config import load_model_config
    from zero.model import estimate_flops_per_token

    main = load_model_config(REPO / "configs" / "main" / "pretrain.toml")
    for T in (2048, 4096):
        n = estimate_flops_per_token(main, T) / 6
        pred = float(law.predict(n, args.main_tokens))
        lo, hi = boot.interval(lambda f, n=n: float(f.predict(n, args.main_tokens)))
        out[f"main_T{T}"] = {"n_eff": n, "tokens": args.main_tokens, "val_bpb": pred, "ci95": [lo, hi]}
        print(f"外推主线（N_eff {n / 1e9:.2f}B @T={T}，{args.main_tokens / 1e9:.0f}B token）：val_bpb {pred:.4f}，"
              f"95% 区间 [{lo:.4f}, {hi:.4f}]")

    plt = plt_setup()
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    scales = [*args.fit, *args.holdout]
    for i, s in enumerate(scales):
        pts = frontier_points(s, kmin=1)
        color = SIZE_RAMP[min(i, len(SIZE_RAMP) - 1)]
        held = s in args.holdout
        D = np.array([p.D for p in pts])
        ax.scatter(D, [p.loss for p in pts], s=28, color=SURFACE if held else color, edgecolors=color,
                   linewidths=1.4, zorder=3, label=f"{s}（N_eff {pts[0].N / 1e6:.1f}M）{'，留出' if held else ''}")
        grid = np.geomspace(D.min() * 0.8, D.max() * 1.6, 100)
        ax.plot(grid, law.predict(pts[0].N, grid), color=color, linestyle="--" if held else "-", zorder=2)
    ax.set_xscale("log")
    ax.set_xlabel("训练 token 数 D")
    ax.set_ylabel("验证集 bits-per-byte（每个预算下最好的配置）")
    ax.set_title("L(N, D) 拟合：实心点参与拟合，空心点留出检验；1/8、2/8 预算不参与拟合", fontsize=10)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / "scaling_law.png", dpi=150)
    RES.mkdir(exist_ok=True)
    (RES / "scaling_law.json").write_text(json.dumps(out, indent=2, ensure_ascii=False, default=float))
    print(f"图：{FIG / 'scaling_law.png'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("frontier", "nql", "sensitivity"):
        p = sub.add_parser(name)
        p.add_argument("--scale", required=True)
        if name == "nql":
            p.add_argument("--tails", type=float, nargs="+", default=[0.07, 0.2])
    p = sub.add_parser("fit")
    p.add_argument("--fit", nargs="+", required=True)
    p.add_argument("--holdout", nargs="*", default=[])
    p.add_argument("--tie", action="store_true", help="α = β（论文里的一种细化）")
    p.add_argument("--boot", type=int, default=200)
    p.add_argument("--main-tokens", type=float, default=400e9)
    args = ap.parse_args()
    {"frontier": cmd_frontier, "nql": cmd_nql, "sensitivity": cmd_sensitivity, "fit": cmd_fit}[args.cmd](args)


if __name__ == "__main__":
    main()
