"""Fit a scaling law and extrapolate (Chapter 12; the "extrapolated prediction" of Gate 1 uses it).

There are two forms:

1. The Chinchilla form (Hoffmann et al. 2022):

       L(N, D) = E + A / N^α + B / D^β

   N is the model size (default: non-embedding parameters; you can also use total parameters, or
   "FLOPs per token / 6"). D is the number of training tokens.
   The fit method is "variable projection". On a grid of (α, β), E, A, and B are linear at each
   grid point, so non-negative least squares gives them directly. The method takes the grid point
   with the smallest error, then refines the grid around it two more times. It uses only NumPy.
   It has no problems with local minima or initial values. (When Epoch AI reproduced Chinchilla,
   they found that the original optimizer stopped too early and the coefficients had a bias:
   arXiv:2404.10102.)

2. The compute power law: L(C) = E + a · (C / C₀)^(−γ), C = 6ND (or FLOPs/token × D).
   Use it for points on the compute-optimal frontier.

Uncertainty: resample "groups" with replacement and fit again. By default, all points with the
same N are one group, because the points that branch from one WSD run are correlated. The 2.5% and
97.5% quantiles of the extrapolated value give the 95% interval. (Delphi does the same: it
resamples IsoFLOP buckets, not single runs.)

Input: run directories that the zero trainer writes (`<out_dir>/log.jsonl` +
`<out_dir>/ckpt/*/meta.json`), or a JSONL file of points (one {"N": ..., "D": ..., "loss": ...,
"name": ...} on each line).

    uv run python -m zero.tools.fit_scaling --run out/ladder/l20m_d1 --run out/ladder/l20m_d2 ... \\
        --holdout out/ladder/l300m --target-config configs/main/pretrain.toml --target-D 400B \\
        --bootstrap 200 --out runs/ladder/fit.json

The module also has `fit_loss_to_score`. It fits "validation loss → benchmark score" with an S-shaped
curve that starts at the chance level. Gate 1 uses it to convert the extrapolated loss into a
benchmark score (the two-step method; see Delphi and the Llama 3 technical report).
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
# Data points
# ---------------------------------------------------------------------------


@dataclass
class Point:
    N: float  # model size (param_count gives the definition)
    D: float  # number of training tokens
    loss: float  # final validation loss (the evaluation after the WSD decay)
    name: str = ""
    group: str = ""  # the resampling unit of the bootstrap; by default one group for each N

    def key(self) -> str:
        return self.group or f"N={self.N:.6g}"


def model_size(cfg: ModelConfig, param_count: str = "non_embedding", seq_len: int = 0) -> float:
    """Return N for the given definition.

    "flops" = training FLOPs per token / 6 (includes attention and lm_head; an "effective parameter count").
    """
    from zero.model import count_params, estimate_flops_per_token

    if param_count == "non_embedding":
        return float(count_params(cfg)["non_embedding"])
    if param_count == "total":
        return float(count_params(cfg)["total"])
    if param_count == "flops":
        return estimate_flops_per_token(cfg, seq_len or cfg.max_seq_len) / 6.0
    raise ValueError(f"param_count must be one of {PARAM_COUNTS}, got {param_count!r}")


def _read_log(run_dir: Path) -> dict[str, Any]:
    log = run_dir / "log.jsonl"
    if not log.exists():
        raise FileNotFoundError(f"{log} does not exist (the run directory must be train.out_dir from the training config)")
    last = None
    for line in log.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("val_loss") is not None:
            last = rec
    if last is None:
        raise ValueError(f"{log} has no val_loss (set eval_every > 0 for training)")
    return last


def _config_from_run(run_dir: Path) -> tuple[ModelConfig, int]:
    """Read the training config from meta.json of the latest checkpoint."""
    from zero.config import _from_dict  # the same validation as load_config
    from zero.train.checkpoint import find_latest

    ckpt = find_latest(run_dir / "ckpt")
    if ckpt is None:
        raise FileNotFoundError(
            f"{run_dir}/ckpt has no checkpoint; give the config with --run dir:config.toml"
        )
    meta = json.loads((ckpt / "meta.json").read_text())
    conf = meta["config"]
    cfg = _from_dict(ModelConfig, conf["model"], "[model]")
    return cfg, int(conf["train"]["data"]["seq_len"])


def load_run(spec: str, param_count: str = "non_embedding", group: str = "") -> Point:
    """spec = "run_dir" or "run_dir:config.toml". Take the last validation loss and the token count at that time."""
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
    return Point(
        N=n, D=float(rec["tokens"]), loss=float(rec["val_loss"]), name=run_dir.name, group=group
    )


def load_points(path: str | Path) -> list[Point]:
    pts = []
    for line in Path(path).read_text().splitlines():
        if line.strip():
            d = json.loads(line)
            pts.append(
                Point(
                    float(d["N"]),
                    float(d["D"]),
                    float(d["loss"]),
                    d.get("name", ""),
                    d.get("group", ""),
                )
            )
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
    rmse: float = 0.0  # root-mean-square residual on the fit points (in loss units)
    max_rel_err: float = 0.0  # largest relative error on the fit points
    n_points: int = 0
    param_count: str = "non_embedding"
    at_boundary: bool = False  # an exponent is at the edge of the search range: the fit is not fixed; do not trust the extrapolation

    def predict(self, N: Any, D: Any) -> Any:
        N, D = np.asarray(N, dtype=float), np.asarray(D, dtype=float)
        return self.E + self.A / N**self.alpha + self.B / D**self.beta

    def compute_optimal(self, C: float) -> tuple[float, float]:
        """The (N, D) with the smallest loss for C = 6ND: N_opt = G·(C/6)^(β/(α+β)), G = (αA/βB)^(1/(α+β))."""
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
    """A batch of non-negative weighted least squares with 3 variables: X (G, n, 3), y (n,), w (n,).

    With 3 variables, there are only 7 combinations of "which variables are positive". Solve each
    one, and keep the feasible solution with the smallest error. This is exact and fully vectorized.
    Return coef (G, 3) and the weighted sum of squared residuals sse (G,).
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
    """Variable projection + grid refinement. The weight of each residual is 1/L, so the fit uses relative errors."""
    if len(points) < (4 if tie_exponents else 5):
        raise ValueError(f"Need at least {4 if tie_exponents else 5} points, got {len(points)}")
    N = np.array([p.N for p in points], dtype=float)
    D = np.array([p.D for p in points], dtype=float)
    y = np.array([p.loss for p in points], dtype=float)
    # Divide N and D by their geometric means first. Else N^-α is very small and the equations are
    # ill-conditioned. At the end, convert back to the original units.
    N0, D0 = float(np.exp(np.log(N).mean())), float(np.exp(np.log(D).mean()))
    n, d = N / N0, D / D0
    w = 1.0 / y**2

    def solve(alphas: np.ndarray, betas: np.ndarray):
        X = np.stack(
            [
                np.ones((len(alphas), len(y))),
                n[None, :] ** -alphas[:, None],
                d[None, :] ** -betas[:, None],
            ],
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
    for _ in range(3):  # refine around the best grid point
        fa = np.linspace(max(a - step_a, lo_a), min(a + step_a, hi_a), 21)
        fb = np.linspace(max(b - step_b, lo_b), min(b + step_b, hi_b), 21)
        if tie_exponents:
            A_, B_ = fa, fa
        else:
            A_, B_ = (g.ravel() for g in np.meshgrid(fa, fb, indexing="ij"))
        a, b, coef, _ = solve(A_, B_)
        step_a, step_b = step_a / 10, step_b / 10
    E, An, Bn = (float(c) for c in coef)
    fit = ChinchillaFit(
        E=E,
        A=An * N0**a,
        B=Bn * D0**b,
        alpha=float(a),
        beta=float(b),
        n_points=len(points),
        param_count=param_count,
    )
    tol = 1e-3
    fit.at_boundary = bool(min(a - lo_a, hi_a - a) < tol or min(b - lo_b, hi_b - b) < tol)
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
    C: Sequence[float],
    L: Sequence[float],
    with_floor: bool = True,
    gamma_range=(0.005, 1.5),
    grid=600,
) -> PowerLawFit:
    """Grid search over γ; for each γ, solve for E and a linearly (non-negative).

    With with_floor=False, E is fixed at 0 (a pure power law).
    """
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
# Bootstrap and "loss → score"
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
    """Resample groups (by default: same N) with replacement, then fit again.

    If a sample has too few groups and thus too few points, skip it.
    """
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
    """score = chance + (top − chance) · sigmoid(k · (L₀ − L)): a lower loss gives a higher score, from the chance level up."""

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
    """Grid search over (L₀, k). chance is the score of a random guess (1 of 4 = 0.25); top is the full score.

    Note: with too few points, or with all points near the chance level, the data almost does not
    constrain this curve. Delphi first fits a scaling law to a "soft metric" (the log-probability of
    the correct answer). Then it fits this S-shaped mapping with a set of public models.
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
# Command line
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Fit L(N, D) and extrapolate (Chapter 12 / Gate 1)")
    ap.add_argument("--run", action="append", default=[], help="run_dir[:config.toml]; can repeat")
    ap.add_argument("--points", help="JSONL file of points: one {N, D, loss[, name, group]} on each line")
    ap.add_argument(
        "--holdout", action="append", default=[], help="run_dir[:config] used only for the held-out test, not for the fit"
    )
    ap.add_argument("--param-count", default="non_embedding", choices=PARAM_COUNTS)
    ap.add_argument("--tie-exponents", action="store_true", help="force α = β (more stable with few points)")
    ap.add_argument(
        "--target-config", help="config of the target model for the extrapolation (example: configs/main/pretrain.toml)"
    )
    ap.add_argument("--target-N", help="or give N directly (same definition as --param-count), for example 605.6M")
    ap.add_argument(
        "--target-D", action="append", default=[], help="target number of tokens; can repeat; for example 400B"
    )
    ap.add_argument("--bootstrap", type=int, default=200, help="number of bootstrap samples (0 = no bootstrap)")
    ap.add_argument("--out", help="write the fit result as JSON (plan_budget --fit can read it)")
    args = ap.parse_args(argv)

    points = [load_run(s, args.param_count) for s in args.run]
    if args.points:
        points += load_points(args.points)
    if not points:
        ap.error("give --run or --points")
    fit = fit_chinchilla(points, tie_exponents=args.tie_exponents, param_count=args.param_count)
    print(f"Fitted {len(points)} points (N definition: {args.param_count})")
    print(f"  L(N, D) = {fit.E:.4f} + {fit.A:.4g}/N^{fit.alpha:.4f} + {fit.B:.4g}/D^{fit.beta:.4f}")
    print(f"  Fit residual RMSE {fit.rmse:.4f}, max relative error {fit.max_rel_err:.2%}")
    if fit.at_boundary:
        print(
            "  ⚠ An exponent is at the edge of the search range: the data does not fix the law (common causes: learning rate not tuned, too narrow a range of sizes or budgets). Do not trust the extrapolation."
        )
    print(f"{'run':<24} {'N':>12} {'D':>12} {'actual':>8} {'fit':>8} {'error':>8}")
    for p in points:
        pred = float(fit.predict(p.N, p.D))
        print(
            f"{p.name[:24]:<24} {p.N:>12.4g} {p.D:>12.4g} {p.loss:>8.4f} {pred:>8.4f} {(pred - p.loss) / p.loss:>+8.2%}"
        )
    N_opt, D_opt = fit.compute_optimal(1e21)
    print(f"  Compute-optimal ratio from the fit (C=1e21): {D_opt / N_opt:.1f} tokens/parameter")

    boot = (
        bootstrap_chinchilla(
            points, args.bootstrap, tie_exponents=args.tie_exponents, param_count=args.param_count
        )
        if args.bootstrap > 0
        else None
    )
    result: dict[str, Any] = {"fit": fit.to_dict(), "points": [asdict(p) for p in points]}

    holdouts = [load_run(s, args.param_count) for s in args.holdout]
    if holdouts:
        print("Held-out test (not used in the fit):")
        result["holdout"] = []
        for p in holdouts:
            pred = float(fit.predict(p.N, p.D))
            ci = (
                boot.interval(lambda f, p=p: float(f.predict(p.N, p.D)))
                if boot
                else (float("nan"),) * 2
            )
            print(
                f"  {p.name}: actual {p.loss:.4f}, extrapolated {pred:.4f} (95% interval {ci[0]:.4f}–{ci[1]:.4f}), "
                f"error {(pred - p.loss) / p.loss:+.2%}"
            )
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
            ci = (
                boot.interval(lambda f, D=D: float(f.predict(target_N, D)))
                if boot
                else (float("nan"),) * 2
            )
            print(
                f"Extrapolation: N={target_N:.4g}, D={D:.4g} → loss {pred:.4f} (95% interval {ci[0]:.4f}–{ci[1]:.4f})"
            )
            result["targets"].append({"N": target_N, "D": D, "pred": pred, "ci95": ci})
        print(
            "  Note: a larger extrapolation factor gives a wider interval. The result has meaning only with the same recipe as the ladder (data, tokenizer, hyperparameter rules)."
        )
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2, ensure_ascii=False))
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
