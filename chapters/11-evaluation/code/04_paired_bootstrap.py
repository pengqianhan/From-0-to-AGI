"""Paired bootstrap: two models have a score difference on the same questions.
How sure can we be that the difference is not luck? (GOAL.md 3.2, item 5)

    uv run python chapters/11-evaluation/code/04_paired_bootstrap.py

1. Write a paired bootstrap from zero. It gives a 95% confidence interval and an
   "ahead / tie / behind" decision.
2. Use the real per-question results of the smoke test (tiny-configuration demo) as an example,
   and do a parity check with zero/eval/bootstrap.py.
3. Paired vs unpaired: on the same data, how wide is each interval?
4. Number of questions: the true difference is 3 percentage points.
   Can we detect it with 30, 300, or 3000 questions?
5. Multiple comparisons: compare two identical models 6 times.
   What is the probability of at least one result that is "not a tie"?
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

AHEAD, TIE, BEHIND = "ahead", "tie", "behind"

# ---------------------------------------------------------------------------
# 1. Implementation from zero
# ---------------------------------------------------------------------------


def paired_bootstrap(a, b, n_boot: int = 10_000, seed: int = 0, confidence: float = 0.95):  # noqa: ANN001, ANN201
    """a, b: per-question scores of two models on the same questions (correct = 1, wrong = 0).
    Return (difference, lower bound, upper bound, decision, all d*)."""
    d = np.asarray(a, float) - np.asarray(b, float)  # per-question difference
    n = len(d)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))  # each row: draw n question indices with replacement (both models use the same indices)
    stats = d[idx].mean(axis=1)  # mean difference d* of each resample
    alpha = 1 - confidence
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])  # percentile method
    decision = AHEAD if lo > 0 else BEHIND if hi < 0 else TIE  # on which side of 0 is the full interval?
    return float(d.mean()), float(lo), float(hi), decision, stats


def unpaired_bootstrap(a, b, n_boot: int = 10_000, seed: int = 0):  # noqa: ANN001, ANN201
    """Control: resample the questions for each model independently.
    This discards the pairing information ("the same question")."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    rng = np.random.default_rng(seed)
    sa = a[rng.integers(0, len(a), size=(n_boot, len(a)))].mean(axis=1)
    sb = b[rng.integers(0, len(b), size=(n_boot, len(b)))].mean(axis=1)
    lo, hi = np.quantile(sa - sb, [0.025, 0.975])
    return float(lo), float(hi)


# ---------------------------------------------------------------------------
# 2. Real per-question results of the smoke test (tiny-configuration demo: a tiny model with about
#    1.3M parameters. Its scores are near random. They only show that the code path works.)
#    They come from out/smoke/eval/results.json (made by `uv run python -m zero.smoke`).
#    out/ is not in git, so we copy the results here. If the file exists locally, the script checks it.
# ---------------------------------------------------------------------------
SMOKE = {
    ("sft", "toy_mc"): "100001010000000001000000100000",
    ("dpo", "toy_mc"): "100001010000100001000000100000",
    ("grpo", "toy_mc"): "101001010010100011000000100000",
    ("sft", "tool_dev"): "000000010001000000010000000001",
    ("dpo", "tool_dev"): "000000010001000000000000000001",
    ("grpo", "tool_dev"): "000000010001000000000000000001",
}
SMOKE_JSON = Path(__file__).resolve().parents[3] / "out/smoke/eval/results.json"


def smoke_scores(model: str, task: str) -> np.ndarray:
    return np.array([int(c) for c in SMOKE[(model, task)]], float)


def check_against_file() -> str:
    if not SMOKE_JSON.exists():
        return "(out/smoke/eval/results.json does not exist locally; check skipped)"
    res = json.loads(SMOKE_JSON.read_text())["results"]
    key = {"toy_mc": "correct", "tool_dev": "call_exact"}
    for (m, t), s in SMOKE.items():
        got = "".join(str(int(it[key[t]])) for it in res[m][t]["items"])
        if got != s:
            # The fixed results above come from the smoke test before the fix of the tool-call scorer
            # (Chapter 19, Section 6). The text uses them. A later run of out/smoke scores more strictly.
            # (Training on a different machine also gives different per-question results.)
            # Thus a mismatch is normal. It is not an error.
            return ("\n(Note: out/smoke is from a run after the scorer fix. It is different from the "
                    f"pre-fix data that the text uses. For example, the per-question results of {m} on {t} "
                    "do not match. Below, we continue to use the fixed pre-fix data.)")
    return "(checked per question: same as out/smoke/eval/results.json)"


def zero_parity(a, b, n_boot: int, seed: int) -> str:  # noqa: ANN001
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repository root, so that we can import zero
    try:
        from zero.eval.bootstrap import paired_bootstrap as zpb
    except ImportError:
        return "(zero package not found; parity check skipped)"
    z = zpb(a, b, n_boot=n_boot, seed=seed)
    diff, lo, hi, dec, _ = paired_bootstrap(a, b, n_boot=n_boot, seed=seed)
    same = (abs(z.diff - diff) < 1e-12 and abs(z.ci_low - lo) < 1e-12
            and abs(z.ci_high - hi) < 1e-12 and z.decision == dec)
    return f"zero.eval.bootstrap.paired_bootstrap gives [{z.ci_low:+.3f}, {z.ci_high:+.3f}] {z.decision}; " \
           f"{'same digits as this file' if same else 'NOT the same as this file!'}"


# ---------------------------------------------------------------------------
# "Models" for the simulation: question i has difficulty p_i.
# The probability that model m answers question i correctly = clip(p_i + delta_m).
# ---------------------------------------------------------------------------


def simulate_pair(rng: np.random.Generator, n: int, acc_a: float, acc_b: float):  # noqa: ANN201
    p = rng.beta(2, 2, size=n)  # question difficulty with mean 0.5
    u = rng.random(n)  # one random number per question: both models tend to fail on hard questions (correlation)
    a = (u < np.clip(p + acc_a - 0.5, 0, 1)).astype(float)
    b = (u < np.clip(p + acc_b - 0.5, 0, 1)).astype(float)
    flip = rng.random(n) < 0.3  # on 30% of the questions, the models answer independently, so they are not fully in sync
    b[flip] = (rng.random(flip.sum()) < np.clip(p[flip] + acc_b - 0.5, 0, 1)).astype(float)
    return a, b


def simulations() -> dict:
    """Three simulations: paired vs unpaired, number of questions, multiple comparisons.
    The video also uses this function."""
    rng = np.random.default_rng(0)
    out: dict = {}
    a, b = simulate_pair(rng, 300, 0.55, 0.50)
    _, lo, hi, dec, _ = paired_bootstrap(a, b)
    out["corr"] = float(np.corrcoef(a, b)[0, 1])
    out["paired"], out["paired_decision"] = (lo, hi), dec
    out["unpaired"] = unpaired_bootstrap(a, b)
    out["by_n"] = []
    for n in (30, 300, 3000):
        widths, wins = [], 0
        for _ in range(100):
            a, b = simulate_pair(rng, n, 0.53, 0.50)
            _, lo, hi, dec, _ = paired_bootstrap(a, b, n_boot=500, seed=int(rng.integers(1 << 30)))
            widths.append(hi - lo)
            wins += dec == AHEAD
        out["by_n"].append((n, float(np.mean(widths)), wins / 100))
    trials, any_hit, single_hit = 300, 0, 0
    for _ in range(trials):
        hits = []
        for _ in range(6):
            a, b = simulate_pair(rng, 30, 0.5, 0.5)
            hits.append(paired_bootstrap(a, b, n_boot=500, seed=int(rng.integers(1 << 30)))[3] != TIE)
        single_hit += hits[0]
        any_hit += any(hits)
    out.update(trials=trials, single_hit=single_hit / trials, any_hit=any_hit / trials)
    return out


def main() -> None:
    # --- a small example that you can calculate by hand ---
    a = np.array([1, 1, 0, 1, 0, 1, 1, 0, 1, 1], float)
    b = np.array([1, 0, 0, 1, 0, 1, 0, 0, 1, 1], float)
    print("Small example to calculate by hand: 10 questions, model A has 7 correct, model B has 5 correct")
    d = (a - b).astype(int).tolist()
    print(f"  per-question difference d = A − B = {d}, mean {np.mean(d):+.2f}")
    rng = np.random.default_rng(1)
    for k in range(3):
        idx = rng.integers(0, 10, size=10)
        print(f"  resample {k + 1}: question indices {idx.tolist()} → d* = {np.mean(np.array(d)[idx]):+.2f}")
    diff, lo, hi, dec, _ = paired_bootstrap(a, b)
    print(f"  10000 resamples: 95% interval [{lo:+.2f}, {hi:+.2f}] → {dec}\n")

    # --- smoke test (tiny-configuration demo) ---
    print("[Tiny-configuration demo] Per-question results of the smoke test, baseline = sft, "
          "2000 resamples, seed 0 (the same as configs/tiny/eval.toml)",
          check_against_file())
    print("| Comparison | Task | Questions | Model | Baseline | Difference | 95% interval | Decision |")
    print("|---|---|---:|---:|---:|---:|---|---|")
    for m in ("dpo", "grpo"):
        for t in ("toy_mc", "tool_dev"):
            x, y = smoke_scores(m, t), smoke_scores("sft", t)
            diff, lo, hi, dec, _ = paired_bootstrap(x, y, n_boot=2000, seed=0)
            print(f"| {m} vs sft | {t} | {len(x)} | {x.mean():.3f} | {y.mean():.3f} | {diff:+.3f} | "
                  f"[{lo:+.3f}, {hi:+.3f}] | {dec} |")
    x, y = smoke_scores("grpo", "toy_mc"), smoke_scores("sft", "toy_mc")
    print("Parity check:", zero_parity(x, y, 2000, 0))
    n10 = int(((x - y) != 0).sum())
    print(f"grpo vs sft on toy_mc: the two models differ on only {n10} of 30 questions "
          "(on all of them, grpo is correct and sft is wrong). "
          "The full difference comes from these few questions.\n")

    # --- three simulations (one random-number stream in a fixed order, so the numbers are reproducible) ---
    sim = simulations()
    print(f"Paired vs unpaired (simulation: 300 questions, correlation of the two models' results {sim['corr']:.2f})")
    lo, hi = sim["paired"]
    ulo, uhi = sim["unpaired"]
    print(f"  paired:   [{lo:+.3f}, {hi:+.3f}], width {hi - lo:.3f} → {sim['paired_decision']}")
    print(f"  unpaired: [{ulo:+.3f}, {uhi:+.3f}], width {uhi - ulo:.3f} → "
          f"{AHEAD if ulo > 0 else BEHIND if uhi < 0 else TIE}\n")
    print("Are there enough questions? True skill 0.53 vs 0.50 (3 percentage points). "
          "100 simulations each. We count how often the decision is 'ahead'.")
    print("| Questions | Mean interval width | Fraction decided 'ahead' |")
    print("|---:|---:|---:|")
    for n, width, win in sim["by_n"]:
        print(f"| {n} | {width:.3f} | {win:.2f} |")
    print(f"\nMultiple comparisons: two models with exactly the same true skill, 30 questions, {sim['trials']} simulations")
    print(f"  fraction with 'ahead/behind' in 1 comparison: {sim['single_hit']:.2f}")
    print(f"  fraction with at least 1 'ahead/behind' in 6 comparisons (like the smoke-test table): {sim['any_hit']:.2f}")
    print("  → The pre-registration must specify in advance 'which score decides the main conclusion'. "
          "All other scores are only descriptive.")


if __name__ == "__main__":
    main()
