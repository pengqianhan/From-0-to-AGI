"""拟合 scaling law 并外推（对应第 12 章；闸门 1 的"外推预测"用它）。

两种形式：

1. Chinchilla 形式（Hoffmann et al. 2022）：

       L(N, D) = E + A / N^α + B / D^β

   N 是模型规模（默认非 embedding 参数，可选总参数或"每 token FLOPs / 6"），D 是训练 token 数。
   拟合方法是"变量投影"：对一个 (α, β) 网格，每个格点上 E、A、B 是线性的，用非负最小二乘直接解出来，
   取误差最小的格点，再在它周围细化两轮。只依赖 NumPy，没有局部极小值和初值的烦恼
   （Epoch AI 复现 Chinchilla 时发现原文的优化器提前停止、系数有偏：arXiv:2404.10102）。

2. 算力幂律：L(C) = E + a · (C / C₀)^(−γ)，C = 6ND（或 FLOPs/token × D）。用于算力最优前沿上的点。

不确定度：按"组"（默认同一个 N 的所有点算一组——WSD 分叉出来的点彼此相关）有放回地重采样，
重新拟合，取外推值的 2.5% / 97.5% 分位数作为 95% 区间（Delphi 的做法：重采样 IsoFLOP 桶，而不是单个运行）。

输入：zero 训练器写出的运行目录（`<out_dir>/log.jsonl` + `<out_dir>/ckpt/*/meta.json`），
或者一个 JSONL 点文件（每行 {"N": ..., "D": ..., "loss": ..., "name": ...}）。

    uv run python -m zero.tools.fit_scaling --run out/ladder/l20m_d1 --run out/ladder/l20m_d2 ... \\
        --holdout out/ladder/l300m --target-config configs/main/pretrain.toml --target-D 400B \\
        --bootstrap 200 --out runs/ladder/fit.json

还提供 `fit_loss_to_score`：把"验证 loss → 基准分数"拟合成一条从随机水平升起的 S 形曲线，
闸门 1 用它把外推出的 loss 换算成基准分数（两步法，见 Delphi 与 Llama 3 技术报告）。
"""

from __future__ import annotations

import argparse
import itertools
import json
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from zero.config import ModelConfig, load_model_config
from zero.tools.estimate_cost import parse_count

PARAM_COUNTS = ("non_embedding", "total", "flops")


# ---------------------------------------------------------------------------
# 数据点
# ---------------------------------------------------------------------------


@dataclass
class Point:
    N: float  # 模型规模（口径见 param_count）
    D: float  # 训练 token 数
    loss: float  # 最终验证 loss（WSD 衰减完的那一次评估）
    name: str = ""
    group: str = ""  # bootstrap 的重采样单位；默认按 N 分组

    def key(self) -> str:
        return self.group or f"N={self.N:.6g}"


def model_size(cfg: ModelConfig, param_count: str = "non_embedding", seq_len: int = 0) -> float:
    """按口径返回 N。"flops" = 每 token 训练 FLOPs / 6（含注意力与 lm_head，相当于"有效参数"）。"""
    from zero.model import count_params, estimate_flops_per_token

    if param_count == "non_embedding":
        return float(count_params(cfg)["non_embedding"])
    if param_count == "total":
        return float(count_params(cfg)["total"])
    if param_count == "flops":
        return estimate_flops_per_token(cfg, seq_len or cfg.max_seq_len) / 6.0
    raise ValueError(f"param_count 只能是 {PARAM_COUNTS}，当前 {param_count!r}")


def _read_log(run_dir: Path) -> dict[str, Any]:
    log = run_dir / "log.jsonl"
    if not log.exists():
        raise FileNotFoundError(f"{log} 不存在（运行目录应是训练配置里的 train.out_dir）")
    last = None
    for line in log.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("val_loss") is not None:
            last = rec
    if last is None:
        raise ValueError(f"{log} 里没有 val_loss（训练时 eval_every 要 > 0）")
    return last


def _config_from_run(run_dir: Path) -> tuple[ModelConfig, int]:
    """从最新 checkpoint 的 meta.json 里读训练时的配置。"""
    from zero.config import _from_dict  # 与 load_config 相同的校验逻辑
    from zero.train.checkpoint import find_latest

    ckpt = find_latest(run_dir / "ckpt")
    if ckpt is None:
        raise FileNotFoundError(f"{run_dir}/ckpt 里没有 checkpoint；请用 --run 目录:配置.toml 指定配置")
    meta = json.loads((ckpt / "meta.json").read_text())
    conf = meta["config"]
    cfg = _from_dict(ModelConfig, conf["model"], "[model]")
    return cfg, int(conf["train"]["data"]["seq_len"])


def load_run(
    spec: str, param_count: str = "non_embedding", group: str = ""
) -> Point:
    """spec = "运行目录" 或 "运行目录:配置.toml"。取最后一次验证 loss 与当时的 token 数。"""
    run, _, config = spec.partition(":")
    run_dir = Path(run)
    rec = _read_log(run_dir)
    if config:
        from zero.config import read_toml

        cfg = load_model_config(config)
        seq_len = int(read_toml(config).get("data", {}).get("seq_len", cfg.max_seq_len))
    else:
        cfg, seq_len = _config_from_run(run_dir)
    n = model_size(cfg, param_count, seq_len)
    return Point(N=n, D=float(rec["tokens"]), loss=float(rec["val_loss"]), name=run_dir.name, group=group)


def load_points(path: str | Path) -> list[Point]:
    pts = []
    for line in Path(path).read_text().splitlines():
        if line.strip():
            d = json.loads(line)
            pts.append(Point(float(d["N"]), float(d["D"]), float(d["loss"]), d.get("name", ""), d.get("group", "")))
    return pts


# ---------------------------------------------------------------------------
# L(N, D) = E + A/N^α + B/D^β
# ---------------------------------------------------------------------------


@dataclass
class ChinchillaFit:
    E: float
    A: float
    B: float
    alpha: float
    beta: float
    rmse: float = 0.0  # 拟合点上的均方根残差（loss 单位）
    max_rel_err: float = 0.0  # 拟合点上最大的相对误差
    n_points: int = 0
    param_count: str = "non_embedding"

    def predict(self, N: Any, D: Any) -> Any:
        N, D = np.asarray(N, dtype=float), np.asarray(D, dtype=float)
        return self.E + self.A / N**self.alpha + self.B / D**self.beta

    def compute_optimal(self, C: float) -> tuple[float, float]:
        """C = 6ND 约束下 loss 最小的 (N, D)：N_opt = G·(C/6)^(β/(α+β))，G = (αA/βB)^(1/(α+β))。"""
        a, b = self.alpha, self.beta
        G = (a * self.A / (b * self.B)) ** (1 / (a + b))
        N = G * (C / 6) ** (b / (a + b))
        return N, C / (6 * N)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ChinchillaFit:
        keys = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in keys})


_SUBSETS = [s for r in (3, 2, 1) for s in itertools.combinations(range(3), r)]


def _nnls3_batched(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """一批 3 变量的非负加权最小二乘：X (G, n, 3)，y (n,)，w (n,)。

    3 个变量只有 7 种"哪些变量为正"的组合，逐个解、保留可行解里误差最小的那个——精确且全部向量化。
    返回 coef (G, 3) 与加权残差平方和 sse (G,)。
    """
    G = X.shape[0]
    Xw = X * np.sqrt(w)[None, :, None]
    yw = y * np.sqrt(w)
    best_sse = np.full(G, np.inf)
    best = np.zeros((G, 3))
    for s in _SUBSETS:
        idx = list(s)
        Xs = Xw[:, :, idx]
        XtX = np.einsum("gni,gnj->gij", Xs, Xs) + 1e-12 * np.eye(len(idx))
        Xty = np.einsum("gni,n->gi", Xs, yw)
        coef = np.linalg.solve(XtX, Xty[..., None])[..., 0]
        resid = yw[None, :] - np.einsum("gni,gi->gn", Xs, coef)
        sse = (resid**2).sum(1)
        ok = (coef >= 0).all(1) & (sse < best_sse)
        best_sse = np.where(ok, sse, best_sse)
        full = np.zeros((G, 3))
        full[:, idx] = coef
        best[ok] = full[ok]
    return best, best_sse


def fit_chinchilla(
    points: Sequence[Point],
    alpha_range: tuple[float, float] = (0.02, 2.0),
    beta_range: tuple[float, float] = (0.02, 2.0),
    grid: int = 80,
    tie_exponents: bool = False,
    param_count: str = "non_embedding",
) -> ChinchillaFit:
    """变量投影 + 网格细化。残差按 1/L 加权，相当于拟合相对误差。"""
    if len(points) < (4 if tie_exponents else 5):
        raise ValueError(f"至少需要 {4 if tie_exponents else 5} 个点，当前 {len(points)} 个")
    N = np.array([p.N for p in points], dtype=float)
    D = np.array([p.D for p in points], dtype=float)
    y = np.array([p.loss for p in points], dtype=float)
    # 先把 N、D 除以几何平均值，避免 N^-α 数值太小导致方程病态；最后再换算回原单位
    N0, D0 = float(np.exp(np.log(N).mean())), float(np.exp(np.log(D).mean()))
    n, d = N / N0, D / D0
    w = 1.0 / y**2

    def solve(alphas: np.ndarray, betas: np.ndarray):
        X = np.stack(
            [np.ones((len(alphas), len(y))), n[None, :] ** -alphas[:, None], d[None, :] ** -betas[:, None]],
            axis=2,
        )
        coef, sse = _nnls3_batched(X, y, w)
        i = int(np.argmin(sse))
        return alphas[i], betas[i], coef[i], sse[i]

    lo_a, hi_a = alpha_range
    lo_b, hi_b = beta_range
    a_grid = np.linspace(lo_a, hi_a, grid)
    b_grid = np.linspace(lo_b, hi_b, grid)
    if tie_exponents:
        A_, B_ = a_grid, a_grid
    else:
        A_, B_ = (g.ravel() for g in np.meshgrid(a_grid, b_grid, indexing="ij"))
    a, b, coef, _ = solve(A_, B_)
    step_a, step_b = a_grid[1] - a_grid[0], b_grid[1] - b_grid[0]
    for _ in range(3):  # 在最优格点周围细化
        fa = np.linspace(max(a - step_a, 1e-4), a + step_a, 21)
        fb = np.linspace(max(b - step_b, 1e-4), b + step_b, 21)
        if tie_exponents:
            A_, B_ = fa, fa
        else:
            A_, B_ = (g.ravel() for g in np.meshgrid(fa, fb, indexing="ij"))
        a, b, coef, _ = solve(A_, B_)
        step_a, step_b = step_a / 10, step_b / 10
    E, An, Bn = (float(c) for c in coef)
    fit = ChinchillaFit(
        E=E, A=An * N0**a, B=Bn * D0**b, alpha=float(a), beta=float(b),
        n_points=len(points), param_count=param_count,
    )
    pred = fit.predict(N, D)
    fit.rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
    fit.max_rel_err = float(np.max(np.abs(pred - y) / y))
    return fit


# ---------------------------------------------------------------------------
# L(C) = E + a (C/C0)^-γ
# ---------------------------------------------------------------------------


@dataclass
class PowerLawFit:
    E: float
    a: float
    gamma: float
    C0: float
    rmse: float = 0.0

    def predict(self, C: Any) -> Any:
        return self.E + self.a * (np.asarray(C, dtype=float) / self.C0) ** -self.gamma


def fit_power_law(
    C: Sequence[float], L: Sequence[float], with_floor: bool = True, gamma_range=(0.005, 1.5), grid=600
) -> PowerLawFit:
    """对 γ 网格搜索；每个 γ 上 E、a 线性（非负）求解。with_floor=False 时 E 固定为 0（纯幂律）。"""
    C_ = np.asarray(C, dtype=float)
    y = np.asarray(L, dtype=float)
    C0 = float(np.exp(np.log(C_).mean()))
    gammas = np.linspace(*gamma_range, grid)
    for _ in range(3):
        cols = [np.ones((len(gammas), len(y)))] if with_floor else []
        cols.append((C_ / C0)[None, :] ** -gammas[:, None])
        X = np.stack(cols, axis=2)
        if with_floor:
            X3 = np.concatenate([X, np.zeros_like(X[..., :1])], axis=2)
            coef, sse = _nnls3_batched(X3, y, 1.0 / y**2)
            coef = coef[:, :2]
        else:
            num = (X[..., 0] * y[None, :] / y**2).sum(1)
            den = (X[..., 0] ** 2 / y**2).sum(1)
            coef = (num / den)[:, None]
            sse = (((y[None, :] - coef * X[..., 0]) ** 2) / y**2).sum(1)
        i = int(np.argmin(sse))
        g = gammas[i]
        span = gammas[1] - gammas[0]
        gammas = np.linspace(max(g - span, 1e-5), g + span, 41)
    E = float(coef[i, 0]) if with_floor else 0.0
    a = float(coef[i, -1])
    fit = PowerLawFit(E=E, a=a, gamma=float(g), C0=C0)
    fit.rmse = float(np.sqrt(np.mean((fit.predict(C_) - y) ** 2)))
    return fit


# ---------------------------------------------------------------------------
# bootstrap 与"loss → 分数"
# ---------------------------------------------------------------------------


@dataclass
class Bootstrap:
    fits: list[ChinchillaFit] = field(default_factory=list)

    def interval(self, fn: Callable[[ChinchillaFit], float], q=(2.5, 97.5)) -> tuple[float, float]:
        vals = np.array([fn(f) for f in self.fits])
        vals = vals[np.isfinite(vals)]
        return float(np.percentile(vals, q[0])), float(np.percentile(vals, q[1]))


def bootstrap_chinchilla(
    points: Sequence[Point], n: int = 200, seed: int = 0, **fit_kwargs: Any
) -> Bootstrap:
    """按组（默认同一个 N）有放回重采样后重拟合。组太少导致点数不够时跳过这次重采样。"""
    rng = np.random.default_rng(seed)
    groups: dict[str, list[Point]] = {}
    for p in points:
        groups.setdefault(p.key(), []).append(p)
    keys = list(groups)
    need = 4 if fit_kwargs.get("tie_exponents") else 5
    fit_kwargs.setdefault("grid", 40)
    out = Bootstrap()
    tries = 0
    while len(out.fits) < n and tries < 5 * n:
        tries += 1
        pick = rng.choice(len(keys), size=len(keys), replace=True)
        sample = [p for k in pick for p in groups[keys[k]]]
        if len({p.N for p in sample}) < 2 or len({p.D for p in sample}) < 2 or len(sample) < need:
            continue
        out.fits.append(fit_chinchilla(sample, **fit_kwargs))
    return out


@dataclass
class ScoreFit:
    """score = chance + (top − chance) · sigmoid(k · (L₀ − L))：loss 越低分数越高，从随机水平起步。"""

    chance: float
    top: float
    L0: float
    k: float
    rmse: float = 0.0

    def predict(self, loss: Any) -> Any:
        z = self.k * (self.L0 - np.asarray(loss, dtype=float))
        return self.chance + (self.top - self.chance) / (1 + np.exp(-z))


def fit_loss_to_score(
    losses: Sequence[float], scores: Sequence[float], chance: float, top: float = 1.0
) -> ScoreFit:
    """网格搜索 (L₀, k)。chance 是随机猜的分数（四选一 = 0.25），top 是满分。

    提醒：点太少、或都挤在随机水平附近时，这条曲线几乎不受约束——
    Delphi 的做法是先对"软指标"（正确答案的对数概率）做 scaling law，再用一批公开模型拟合这条 S 形映射。
    """
    x = np.asarray(losses, dtype=float)
    y = np.asarray(scores, dtype=float)
    best = None
    L0s = np.linspace(x.min() - 2.0, x.max() + 2.0, 400)
    ks = np.geomspace(0.1, 100, 300)
    for L0 in L0s:
        z = ks[:, None] * (L0 - x[None, :])
        pred = chance + (top - chance) / (1 + np.exp(-z))
        sse = ((pred - y[None, :]) ** 2).sum(1)
        i = int(np.argmin(sse))
        if best is None or sse[i] < best[0]:
            best = (float(sse[i]), float(L0), float(ks[i]))
    _, L0, k = best
    fit = ScoreFit(chance=chance, top=top, L0=L0, k=k)
    fit.rmse = float(np.sqrt(np.mean((fit.predict(x) - y) ** 2)))
    return fit


# ---------------------------------------------------------------------------
# 命令行
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="拟合 L(N, D) 并外推（第 12 章 / 闸门 1）")
    ap.add_argument("--run", action="append", default=[], help="运行目录[:配置.toml]，可重复")
    ap.add_argument("--points", help="JSONL 点文件：每行 {N, D, loss[, name, group]}")
    ap.add_argument("--holdout", action="append", default=[], help="只用来检验、不参与拟合的运行目录[:配置]")
    ap.add_argument("--param-count", default="non_embedding", choices=PARAM_COUNTS)
    ap.add_argument("--tie-exponents", action="store_true", help="强制 α = β（点少时更稳）")
    ap.add_argument("--target-config", help="要外推的目标模型配置（例：configs/main/pretrain.toml）")
    ap.add_argument("--target-N", help="或直接给 N（口径同 --param-count），如 605.6M")
    ap.add_argument("--target-D", action="append", default=[], help="目标 token 数，可重复，如 400B")
    ap.add_argument("--bootstrap", type=int, default=200, help="bootstrap 次数（0 = 不做）")
    ap.add_argument("--out", help="把拟合结果写成 JSON（plan_budget --fit 可以读）")
    args = ap.parse_args(argv)

    points = [load_run(s, args.param_count) for s in args.run]
    if args.points:
        points += load_points(args.points)
    if not points:
        ap.error("需要 --run 或 --points")
    fit = fit_chinchilla(points, tie_exponents=args.tie_exponents, param_count=args.param_count)
    print(f"拟合 {len(points)} 个点（N 口径：{args.param_count}）")
    print(f"  L(N, D) = {fit.E:.4f} + {fit.A:.4g}/N^{fit.alpha:.4f} + {fit.B:.4g}/D^{fit.beta:.4f}")
    print(f"  拟合残差 RMSE {fit.rmse:.4f}，最大相对误差 {fit.max_rel_err:.2%}")
    print(f"{'运行':<24} {'N':>12} {'D':>12} {'实际':>8} {'拟合':>8} {'误差':>8}")
    for p in points:
        pred = float(fit.predict(p.N, p.D))
        print(f"{p.name[:24]:<24} {p.N:>12.4g} {p.D:>12.4g} {p.loss:>8.4f} {pred:>8.4f} {(pred - p.loss) / p.loss:>+8.2%}")
    N_opt, D_opt = fit.compute_optimal(1e21)
    print(f"  拟合给出的算力最优比例（C=1e21）：{D_opt / N_opt:.1f} token/参数")

    boot = bootstrap_chinchilla(points, args.bootstrap, tie_exponents=args.tie_exponents,
                                param_count=args.param_count) if args.bootstrap > 0 else None
    result: dict[str, Any] = {"fit": fit.to_dict(), "points": [asdict(p) for p in points]}

    holdouts = [load_run(s, args.param_count) for s in args.holdout]
    if holdouts:
        print("留出检验（没参与拟合）：")
        result["holdout"] = []
        for p in holdouts:
            pred = float(fit.predict(p.N, p.D))
            ci = boot.interval(lambda f, p=p: float(f.predict(p.N, p.D))) if boot else (float("nan"),) * 2
            print(f"  {p.name}: 实际 {p.loss:.4f}，外推 {pred:.4f}（95% 区间 {ci[0]:.4f}–{ci[1]:.4f}），"
                  f"误差 {(pred - p.loss) / p.loss:+.2%}")
            result["holdout"].append({**asdict(p), "pred": pred, "ci95": ci})

    target_N = None
    if args.target_config:
        from zero.config import read_toml

        cfg = load_model_config(args.target_config)
        seq_len = int(read_toml(args.target_config).get("data", {}).get("seq_len", cfg.max_seq_len))
        target_N = model_size(cfg, args.param_count, seq_len)
    elif args.target_N:
        target_N = parse_count(args.target_N)
    if target_N is not None:
        result["targets"] = []
        for d in args.target_D or ["20x"]:
            D = 20 * target_N if d == "20x" else parse_count(d)
            pred = float(fit.predict(target_N, D))
            ci = boot.interval(lambda f, D=D: float(f.predict(target_N, D))) if boot else (float("nan"),) * 2
            print(f"外推：N={target_N:.4g}, D={D:.4g} → loss {pred:.4f}（95% 区间 {ci[0]:.4f}–{ci[1]:.4f}）")
            result["targets"].append({"N": target_N, "D": D, "pred": pred, "ci95": ci})
        print("  提醒：外推倍数越大区间越宽；只有与阶梯同一配方（数据、分词器、超参规则）时才有意义。")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2, ensure_ascii=False))
        print(f"已写出 {args.out}")


if __name__ == "__main__":
    main()
