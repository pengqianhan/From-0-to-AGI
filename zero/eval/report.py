"""评测结果 → Markdown 表格（对应第 11、20 章）。

- `results_table(results)`：行是模型，列是"任务/指标"；
- `comparison_table(comparisons)`：每行一个"模型 vs 基线"的配对 bootstrap 结果与判定；
- `write_report(path, ...)`：两张表写进一个 Markdown 文件（模型卡和课程正文直接引用）。
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
}


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.3f}" if abs(v) < 1000 else f"{v:,.0f}"
    return str(v)


def results_table(results: Mapping[str, Mapping[str, Mapping[str, Any]]]) -> str:
    """results[模型][任务] = {"type": ..., 指标: 值, "n": 题数}。"""
    cols: list[tuple[str, str]] = []
    for per_task in results.values():
        for task, r in per_task.items():
            for m in METRICS_BY_TYPE.get(r.get("type", ""), []):
                if (task, m) not in cols and m in r:
                    cols.append((task, m))
    header = "| 模型 | " + " | ".join(f"{t} {m}" for t, m in cols) + " |"
    sep = "|---|" + "---:|" * len(cols)
    rows = [header, sep]
    for model, per_task in results.items():
        cells = [_fmt(per_task.get(t, {}).get(m, "—")) for t, m in cols]
        rows.append(f"| {model} | " + " | ".join(cells) + " |")
    ns = sorted({(t, r.get("n")) for pt in results.values() for t, r in pt.items()})
    rows.append("")
    rows.append("题数：" + "，".join(f"{t} n={n}" for t, n in ns))
    return "\n".join(rows)


def comparison_table(comparisons: Sequence[Mapping[str, Any]]) -> str:
    if not comparisons:
        return "（没有配对比较）"
    rows = [
        "| 模型 | 基线 | 任务 | 指标 | 模型得分 | 基线得分 | 差值 | 95% CI | 判定 |",
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
    title: str = "评测结果",
    notes: Sequence[str] = (),
) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    parts = [
        f"# {title}",
        "",
        "## 得分",
        "",
        results_table(results),
        "",
        "## 配对 bootstrap 比较",
        "",
    ]
    parts.append(comparison_table(comparisons))
    parts += [
        "",
        "判定规则：差值的 95% 置信区间整体大于 0 为“超过”，整体小于 0 为“落后”，跨过 0 为“持平”（GOAL.md 3.2）。",
    ]
    for n in notes:
        parts += ["", f"- {n}"]
    p.write_text("\n".join(parts) + "\n", encoding="utf-8")
    return p
