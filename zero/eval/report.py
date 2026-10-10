"""Evaluation results → Markdown tables (Chapters 11 and 20).

- `results_table(results)`: one row for each model, one column for each "task / metric".
- `comparison_table(comparisons)`: one row for each "model vs. baseline" paired bootstrap result
  and its decision.
- `write_report(path, ...)`: writes the two tables into one Markdown file. The model card and the
  course text use this file directly.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

METRICS_BY_TYPE = {
    "mc": ["acc", "acc_norm"],
    "gen": ["em"],
    "tool": ["call_exact", "call_reward", "format_ok", "answer_ok"],
    "fc": ["call_exact", "call_reward", "format_ok"],
}


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.3f}" if abs(v) < 1000 else f"{v:,.0f}"
    return str(v)


def results_table(results: Mapping[str, Mapping[str, Mapping[str, Any]]]) -> str:
    """results[model][task] = {"type": ..., metric: value, "n": number of items}."""
    cols: list[tuple[str, str]] = []
    for per_task in results.values():
        for task, r in per_task.items():
            for m in METRICS_BY_TYPE.get(r.get("type", ""), []):
                if (task, m) not in cols and m in r:
                    cols.append((task, m))
    header = "| Model | " + " | ".join(f"{t} {m}" for t, m in cols) + " |"
    sep = "|---|" + "---:|" * len(cols)
    rows = [header, sep]
    for model, per_task in results.items():
        cells = [_fmt(per_task.get(t, {}).get(m, "—")) for t, m in cols]
        rows.append(f"| {model} | " + " | ".join(cells) + " |")
    ns = sorted({(t, r.get("n")) for pt in results.values() for t, r in pt.items()})
    rows.append("")
    rows.append("Number of items: " + ", ".join(f"{t} n={n}" for t, n in ns))
    return "\n".join(rows)


def comparison_table(comparisons: Sequence[Mapping[str, Any]]) -> str:
    if not comparisons:
        return "(no paired comparisons)"
    rows = [
        "| Model | Baseline | Task | Metric | Model score | Baseline score | Difference | 95% CI | Decision |",
        "|---|---|---|---|---:|---:|---:|---|---|",
    ]
    for c in comparisons:
        rows.append(
            f"| {c['model']} | {c['baseline']} | {c['task']} | {c['metric']} | {c['mean_a']:.3f} | "
            f"{c['mean_b']:.3f} | {c['diff']:+.3f} | [{c['ci_low']:+.3f}, {c['ci_high']:+.3f}] | {c['decision']} |"
        )
    return "\n".join(rows)


def write_report(
    path: str | os.PathLike,
    results: Mapping[str, Mapping[str, Mapping[str, Any]]],
    comparisons: Sequence[Mapping[str, Any]] = (),
    title: str = "Evaluation results",
    notes: Sequence[str] = (),
) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    parts = [
        f"# {title}",
        "",
        "## Scores",
        "",
        results_table(results),
        "",
        "## Paired bootstrap comparisons",
        "",
    ]
    parts.append(comparison_table(comparisons))
    parts += [
        "",
        "Decision rule: if the full 95% confidence interval of the difference is above 0, the decision is "
        '"ahead". If the full interval is below 0, the decision is "behind". If the interval contains 0, '
        'the decision is "tie" (GOAL.md 3.2).',
    ]
    for n in notes:
        parts += ["", f"- {n}"]
    p.write_text("\n".join(parts) + "\n", encoding="utf-8")
    return p
