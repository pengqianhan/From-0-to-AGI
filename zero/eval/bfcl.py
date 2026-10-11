"""BFCL adapter: evaluate our exported model with the official Berkeley Function Calling Leaderboard (Chapters 11 and 20).

**Status (2026-10-10): checked against the source code of `bfcl-eval` 2026.3.23 (PyPI), not run yet.**
A run needs vLLM and a GPU. What the source shows, and how this module follows it:

- Registries: `bfcl_eval.constants.model_config` has `local_inference_model_map` and
  `MODEL_CONFIG_MAPPING` (a merged dict, built at import). The CLI (`bfcl_eval.__main__`) imports
  `MODEL_CONFIG_MAPPING` by reference, so adding our entry to that same dict before `cli()` runs works.
  `ModelConfig` fields: model_name, display_name, url, org, license, model_handler, input_price,
  output_price, is_fc_model, underscore_to_dot.
- `function` in `_format_prompt(messages, function)` is the raw BFCL list: `{"name", "description",
  "parameters"}` with **Python type names** (`"dict"`, `"float"`, `"tuple"`). The built-in
  `QwenFCHandler` writes it as it is (`json.dumps(tool)`). Our handler wraps each function as
  `{"type": "function", "function": ...}` and converts the type names to JSON schema
  (`fc_tasks.to_json_schema`), because all our training data uses that form. This is part of our
  model's template, and the preregistration states it.
- `QwenFCHandler._pre_query_processing_prompting` adds **no** BFCL system prompt (FC models use their
  own template), and `_extract_tool_calls` needs exactly `<tool_call>\n{json}\n</tool_call>`, the format
  of `zero.post.chat.format_tool_call`.
- Per-item results: `score/<model>/<group>/BFCL_v4_<category>_score.json` has a header line
  (`accuracy`, `correct_count`, `total_count`) and then **only the failed items** (with `id`).
  `result/<model>/<group>/BFCL_v4_<category>_result.json` has every generated item. So the per-item
  correctness for the paired bootstrap is "ids in the result file − ids in the score file"
  (`per_item_results`).

Local check without vLLM: `fc_tasks.read_bfcl_dir(<bfcl_eval>/data)` converts the shipped data; with
the output equal to the gold answer, all 3,641 single-turn items score exact under `fc_tasks.score_fc`
(`runs/POSTTRAIN_PLAN.md`, section 6.3). This checks the scorer only: BFCL is a test set, so it is not
used to select checkpoints (eval/PREREGISTRATION.md, section 8).

Why an adapter is necessary: BFCL runs local models in "prompt mode". The evaluation framework
makes the prompt itself (`_format_prompt`), and then calls `/v1/completions` of vLLM / SGLang.
Its built-in `QwenFCHandler` has a hand-written copy of the Qwen3 template. With thinking off, it
puts an empty `<think>\\n\\n</think>\\n\\n` after the generation prompt, but our model never saw
the thinking format. So this module registers a `ZeroFCHandler`. It inherits from `QwenFCHandler`
(to reuse its `<tool_call>` parsing). It replaces only the prompt construction with **the
chat_template in the export directory** (`tokenizer.apply_chat_template`). Then the evaluation
prompt is identical, character for character, to the training prompt.

Usage (Step 2, on a GPU machine):

    uv pip install bfcl-eval==2026.3.23 vllm            # the version is frozen in eval/PREREGISTRATION.md
    # 1) Generate (the framework starts a vLLM server; or start the server yourself and add --skip-server-setup)
    uv run python -m zero.eval.bfcl generate --hf-dir out/main/hf_final --name zero-0.7b-FC \\
        --test-category simple_python,multiple,parallel,parallel_multiple,irrelevance,live,multi_turn \\
        --backend vllm --num-gpus 1
    # 2) Score
    uv run python -m zero.eval.bfcl evaluate --hf-dir out/main/hf_final --name zero-0.7b-FC \\
        --test-category simple_python,multiple,parallel,parallel_multiple,irrelevance,live,multi_turn
    # 3) Per-category accuracy from the per-item results; paired bootstrap against an opponent
    uv run python -m zero.eval.bfcl collect --project-root $BFCL_PROJECT_ROOT --name zero-0.7b-FC
    uv run python -m zero.eval.bfcl compare --project-root $BFCL_PROJECT_ROOT --name zero-0.7b-FC \\
        --opponent Qwen/Qwen3-0.6B-FC

The opponent models (Qwen3.5-0.8B and others) use the built-in BFCL handler names directly (each
with its official template), with the same version and the same decoding parameters (BFCL default
temperature=0.001). The per-item results are in `$BFCL_PROJECT_ROOT/score/<model>/`.
For the paired bootstrap, use `zero.eval.bootstrap.paired_bootstrap`.

Still to check at the first GPU run: the generate / evaluate commands end to end with vLLM, and that
`apply_chat_template` of the exported tokenizer gives the same text as `zero.post.chat.render`
(`tests/test_chat.py` checks this with transformers, but not inside BFCL).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


def bfcl_available() -> bool:
    try:
        import bfcl_eval  # noqa: F401
    except ImportError:
        return False
    return True


def make_handler_class():  # noqa: ANN201
    """Return the ZeroFCHandler class (bfcl-eval must be installed)."""
    from bfcl_eval.model_handler.local_inference.qwen_fc import QwenFCHandler

    from zero.post.envs.fc_tasks import to_json_schema

    class ZeroFCHandler(QwenFCHandler):
        def _format_prompt(
            self, messages: list[dict[str, Any]], function: list[dict[str, Any]]
        ) -> str:
            tools = [
                {"type": "function", "function": to_json_schema(f.get("function", f))}
                for f in function
            ]
            return self.tokenizer.apply_chat_template(
                messages, tools=tools or None, add_generation_prompt=True, tokenize=False
            )

    return ZeroFCHandler


def register_zero_model(name: str, hf_dir: str | os.PathLike, display_name: str = "") -> None:
    """Register our model in the model table of BFCL (only in this process)."""
    from bfcl_eval.constants import model_config as mc

    cfg = mc.ModelConfig(
        model_name=str(hf_dir),
        display_name=display_name or f"{name} (FC)",
        url="https://github.com/(fill in after the release)",
        org="From-0-to-AGI",
        license="apache-2.0",
        model_handler=make_handler_class(),
        input_price=None,
        output_price=None,
        is_fc_model=True,
        underscore_to_dot=False,
    )
    mc.local_inference_model_map[name] = cfg
    mc.MODEL_CONFIG_MAPPING[name] = cfg


def run_cli(cmd: str, hf_dir: str, name: str, extra: list[str]) -> None:
    """Register the model, then call the command-line entry point of bfcl (typer app)."""
    if not bfcl_available():
        raise SystemExit(
            "bfcl-eval is not installed: uv pip install bfcl-eval==<version frozen in the preregistration>"
        )
    register_zero_model(name, hf_dir)
    from bfcl_eval.__main__ import cli

    argv = [cmd, "--model", name, *extra]
    if cmd == "generate":
        argv += ["--local-model-path", str(hf_dir)]
    sys.argv = ["bfcl", *argv]
    cli()


def _read_jsonl(p: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in p.read_text("utf-8").splitlines() if x.strip()]


def per_item_results(project_root: str | os.PathLike, model: str) -> dict[str, dict[str, float]]:
    """{category: {test id: 1.0 correct / 0.0 wrong}} of one model.

    BFCL writes every generated item to the result file and only the failed items to the score file,
    so correct = ids of the result file − ids of the score file. A model name with "/" is stored with "_".
    """
    root = Path(project_root)
    mdir = model.replace("/", "_")
    out: dict[str, dict[str, float]] = {}
    for res in sorted((root / "result" / mdir).rglob("BFCL_v*_*_result.json")):
        cat = res.name.split("_", 2)[2].rsplit("_result.json", 1)[0]
        ids = [r["id"] for r in _read_jsonl(res)]
        rel = res.relative_to(root / "result" / mdir).parent
        score = root / "score" / mdir / rel / res.name.replace("_result.json", "_score.json")
        if not score.exists():
            continue  # generated, not evaluated yet
        rows = _read_jsonl(score)
        failed = {r["id"] for r in rows[1:] if "id" in r}
        out[cat] = {i: 0.0 if i in failed else 1.0 for i in ids}
        header = rows[0] if rows else {}
        if header.get("total_count") not in (None, len(ids)):
            raise ValueError(
                f"{score}: total_count {header['total_count']} != {len(ids)} items in {res}"
            )
    return out


def collect_scores(project_root: str | os.PathLike, model: str) -> dict[str, Any]:
    """Accuracy and item count per category (from the per-item results)."""
    items = per_item_results(project_root, model)
    return {
        c: {"accuracy": sum(v.values()) / max(len(v), 1), "n": len(v)} for c, v in items.items()
    }


def compare_models(
    project_root: str | os.PathLike,
    ours: str,
    theirs: str,
    weights: dict[str, float] | None = None,
    n_boot: int = 10_000,
    seed: int = 0,
) -> dict[str, Any]:
    """Stratified paired bootstrap of two models on the categories that both have (E1).

    weights: the category weights of the preregistration; default: equal weights.
    """
    from zero.eval.bootstrap import stratified_paired_bootstrap

    a, b = per_item_results(project_root, ours), per_item_results(project_root, theirs)
    cats = sorted(set(a) & set(b) & set(weights or a))
    a_by, b_by = {}, {}
    for c in cats:
        common = sorted(set(a[c]) & set(b[c]))
        if len(common) != len(a[c]) or len(common) != len(b[c]):
            raise ValueError(
                f"{c}: the two models have different items ({len(a[c])} vs {len(b[c])})"
            )
        a_by[c] = [a[c][i] for i in common]
        b_by[c] = [b[c][i] for i in common]
    w = {c: (weights or {}).get(c, 1.0) for c in cats}
    r = stratified_paired_bootstrap(a_by, b_by, w, n_boot=n_boot, seed=seed)
    return {"categories": cats, "weights": w, **r.__dict__}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description="BFCL adapter (Step 2; checked against bfcl-eval 2026.3.23)"
    )
    ap.add_argument("command", choices=["generate", "evaluate", "collect", "compare"])
    ap.add_argument("--hf-dir", default="")
    ap.add_argument("--name", default="zero-FC")
    ap.add_argument("--project-root", default=os.environ.get("BFCL_PROJECT_ROOT", ""))
    ap.add_argument("--opponent", default="", help="compare: the BFCL model name of the opponent")
    args, extra = ap.parse_known_args(argv)
    if args.command == "collect":
        print(
            json.dumps(collect_scores(args.project_root, args.name), ensure_ascii=False, indent=2)
        )
        return
    if args.command == "compare":
        res = compare_models(args.project_root, args.name, args.opponent)
        print(json.dumps(res, ensure_ascii=False, indent=2, default=str))
        return
    run_cli(args.command, args.hf_dir, args.name, extra)


if __name__ == "__main__":
    main()
