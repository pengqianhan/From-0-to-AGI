"""Chapter 20 · Minimal code 4: make a model card from the evaluation results JSON

A model card is the README.md on the front page of a model repository. It starts with a block of
YAML metadata (license, language, data sets, tags, ...). Hugging Face uses the metadata for search
and display. The text for people comes after it. GOAL.md 3.5 says that our model card must contain:
the training data and licenses, the recipe and cost of each stage, the preregistration protocol,
all evaluation results (also the items where we are behind), the decontamination check, and the
known limitations.

The rule of this script: **fill in only numbers that have a source**. The script makes the
evaluation tables directly from the results JSON. The format is the same as results.json from
zero/eval/harness.py: results[model][task] = {metric: value, n: number of items}, and
comparisons = the decisions of the paired bootstrap. Where no data exists, the script always
writes "TBD after training". It does not leave the cell empty, and it does not estimate a number.

    uv run python chapters/20-release/code/04_model_card.py                 # reads out/smoke by default (tiny-configuration demo)
    uv run python chapters/20-release/code/04_model_card.py --results path/to/results.json --out card.md

`uv run python -m zero.smoke` makes out/smoke. If this folder does not exist, the script prints a
skeleton with "TBD after training" in all places.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TODO = "TBD after training"

# Metadata: fill in each item at release. None means "not decided yet"; the card shows "undecided"
META = {
    "name": "zero-0.7b (working name)",
    "license": None,  # weight license: the author decides (see the options in the "License" section of the chapter)
    "language": ["zh", "en"],
    "library_name": "transformers",
    "pipeline_tag": "text-generation",
    "tags": ["function-calling", "tool-use", "from-scratch", "gguf"],
    "datasets": [],   # at release, add the HF data set IDs, for example FineWeb-Edu for pretraining
    "architecture": "Dense Transformer compatible with Qwen3ForCausalLM (Pre-Norm RMSNorm, SwiGLU, RoPE, GQA, QK-Norm, shared embedding)",
    "params": "689.5M (provisional shape in configs/main/pretrain.toml)",
}

STAGES = ["Pretraining", "Mid-training", "Long context", "SFT", "Distillation", "DPO", "GRPO"]


def yaml_front_matter(meta: dict) -> str:
    lines = ["---", f"license: {meta['license'] or 'other  # undecided'}"]
    for key in ("language", "tags", "datasets"):
        if meta[key]:
            lines.append(f"{key}:")
            lines += [f"- {v}" for v in meta[key]]
    lines += [f"library_name: {meta['library_name']}", f"pipeline_tag: {meta['pipeline_tag']}", "---"]
    return "\n".join(lines)


def results_table(results: dict) -> str:
    cols: list[tuple[str, str]] = []
    for per_task in results.values():
        for task, r in per_task.items():
            for k, v in r.items():
                if isinstance(v, (int, float)) and k != "n" and (task, k) not in cols:
                    cols.append((task, k))
    out = ["| Model | " + " | ".join(f"{t} / {m}" for t, m in cols) + " |",
           "|---|" + "---:|" * len(cols)]
    for model, per_task in results.items():
        cells = [f"{per_task.get(t, {}).get(m, float('nan')):.3f}" for t, m in cols]
        out.append(f"| {model} | " + " | ".join(cells) + " |")
    ns = sorted({f"{t} n={r.get('n')}" for pt in results.values() for t, r in pt.items()})
    return "\n".join(out) + "\n\nNumber of items: " + ", ".join(ns)


def comparison_table(comps: list[dict]) -> str:
    out = ["| Ours | Opponent | Benchmark | Metric | Ours | Opponent | Difference | 95% CI | Decision |",
           "|---|---|---|---|---:|---:|---:|---|---|"]
    for c in comps:
        out.append(f"| {c['model']} | {c['baseline']} | {c['task']} | {c['metric']} | {c['mean_a']:.3f} | "
                   f"{c['mean_b']:.3f} | {c['diff']:+.3f} | [{c['ci_low']:+.3f}, {c['ci_high']:+.3f}] | "
                   f"**{c['decision']}** |")
    tally = {d: sum(c["decision"] == d for c in comps) for d in ("ahead", "tie", "behind")}
    out.append("")
    out.append(f"Total: ahead {tally['ahead']}, tie {tally['tie']}, behind {tally['behind']} (all items are listed; none are selected).")
    return "\n".join(out)


def stage_table(summary: dict | None) -> str:
    rows = ["| Stage | Tokens / steps | Key metrics | Cost |", "|---|---|---|---|"]
    by_name = {s["stage"]: s for s in (summary or {}).get("stages", [])}
    alias = {"Pretraining": "pretrain", "Mid-training": "midtrain", "SFT": "sft", "Distillation": "distill",
             "DPO": "dpo", "GRPO": "grpo"}
    for st in STAGES:
        s = by_name.get(alias.get(st, ""))
        if s is None:
            rows.append(f"| {st} | {TODO} | {TODO} | {TODO} |")
            continue
        metrics = "; ".join(f"{k}={v}" for k, v in s.items()
                           if k not in ("stage", "status", "seconds", "steps"))
        rows.append(f"| {st} | {s.get('steps', '—')} steps | {metrics} | CPU {s['seconds']:.0f}s, $0 |")
    return "\n".join(rows)


def build_card(results_json: dict | None, summary: dict | None, demo: bool) -> str:
    parts = [yaml_front_matter(META), "", f"# {META['name']}", ""]
    if demo:
        parts += ["> ⚠️ **Tiny-configuration demo**: the numbers below come from a CPU smoke test (`zero.smoke`) of a model",
                  "> with about 1.3M parameters. They only show that the pipeline works. **They are not results of the main-line model.**", ""]
    parts += [
        "## Model summary", "",
        f"- Architecture: {META['architecture']}",
        f"- Parameters: {META['params']}",
        "- Purpose: a small Chinese-English model for tool calling (function calling). It runs on a laptop with llama.cpp / Ollama",
        f"- Weight license: {META['license'] or 'undecided (the author decides)'}",
        "- Released files: Base, SFT, final version, and key intermediate checkpoints (HF safetensors); GGUF (Q8_0, Q4_K_M)",
        "",
        "## Usage", "",
        "```python",
        "from transformers import AutoModelForCausalLM, AutoTokenizer",
        "tok = AutoTokenizer.from_pretrained(\"<repo-name>\")",
        "model = AutoModelForCausalLM.from_pretrained(\"<repo-name>\")",
        "ids = tok.apply_chat_template(messages, tools=tools, add_generation_prompt=True, return_tensors=\"pt\")",
        "```",
        "",
        "```bash",
        "llama-cli -m zero-Q4_K_M.gguf          # llama.cpp; Ollama can load the same GGUF directly",
        "vllm serve <repo-name> --enable-auto-tool-choice --tool-call-parser hermes",
        "```",
        "",
        "## Training data and licenses", "",
        "| Data set | Stage | Tokens | License | Attribution requirement |", "|---|---|---|---|---|",
        f"| {TODO} | {TODO} | {TODO} | {TODO} | {TODO} |",
        "",
        "Teacher model (distillation): name, version, and whether the license allows training other models on its outputs: " + TODO + ".",
        "",
        "## Recipe and cost of each stage", "",
        stage_table(summary), "",
        "For the full configs, see `configs/main/`. For the cost details, see `runs/ledger.md`.", "",
        "## Preregistration", "",
        "We registered the evaluation protocol before training: `eval/PREREGISTRATION.md` (registration commit: " + TODO + ").",
        "Decision rule (paired bootstrap, 95% confidence interval): the full interval > 0 → ahead; the full interval < 0 → behind; the interval crosses 0 → tie.", "",
        "## Evaluation results (all items, also the items where we are behind)", "",
    ]
    if results_json:
        parts += [results_table(results_json["results"]), "",
                  "### Paired comparisons", "", comparison_table(results_json.get("comparisons", [])), ""]
    else:
        parts += [f"{TODO} (BFCL, a Chinese tool-calling benchmark, general benchmarks; one row for each opponent and each benchmark)", ""]
    parts += [
        "We show the official scores of the opponents next to our results, but we do not use them for the comparison.", "",
        "### Opponents added after release", "",
        f"Models of the same size released after the freeze date: {TODO} (we list them here, also if they are better than ours).", "",
        "## Decontamination check", "",
        f"- 13-gram overlap: training data (including synthetic data from the teacher) vs all evaluation sets, hit rate {TODO}",
        f"- Overlap of tool function names / argument schemas with BFCL and other evaluation sets: removed items {TODO}", "",
        "## Known limitations", "",
        "- The model has few parameters and limited knowledge. It makes up answers to factual questions.",
        "- We evaluated tool calling only on the preregistered benchmarks and in our own environment. Real tools and arguments can have a different distribution.",
        f"- Other: {TODO}", "",
        "## Citation and acknowledgments", "",
        "Code and course: <https://github.com/…/From-0-to-AGI> (" + TODO + ")",
    ]
    return "\n".join(parts) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(ROOT / "out/smoke/eval/results.json"))
    ap.add_argument("--summary", default=str(ROOT / "out/smoke/summary.json"))
    ap.add_argument("--out", default="", help="write to this file; by default, print to the terminal")
    args = ap.parse_args()
    res_p, sum_p = Path(args.results), Path(args.summary)
    results = json.loads(res_p.read_text(encoding="utf-8")) if res_p.exists() else None
    summary = json.loads(sum_p.read_text(encoding="utf-8")) if sum_p.exists() else None
    demo = "smoke" in str(res_p) and results is not None
    card = build_card(results, summary, demo)
    if args.out:
        Path(args.out).write_text(card, encoding="utf-8")
        print(f"Model card → {args.out}")
    else:
        print(card)
    n_todo = card.count(TODO)
    print(f"<!-- {n_todo} '{TODO}' placeholders are left. Fill in all of them before release. -->")


if __name__ == "__main__":
    main()
