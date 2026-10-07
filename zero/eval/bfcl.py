"""BFCL adapter: evaluate our exported model with the official Berkeley Function Calling Leaderboard (Chapters 11 and 20).

**Status: not verified yet.** This machine has no GPU, and it cannot download models (only the
evaluation data). The code below follows the source code of `bfcl-eval` version 2026.3.23 (PyPI).
Remove this sentence after the first successful run on a GPU machine in Step 2.

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
    # 3) Read the scores (BFCL writes CSV files under $BFCL_PROJECT_ROOT/score/) and convert them to our report format
    uv run python -m zero.eval.bfcl collect --score-dir $BFCL_PROJECT_ROOT/score --name zero-0.7b-FC

The opponent models (Qwen3.5-0.8B and others) use the built-in BFCL handler names directly (each
with its official template), with the same version and the same decoding parameters (BFCL default
temperature=0.001). The per-item results are in `$BFCL_PROJECT_ROOT/score/<model>/`.
For the paired bootstrap, use `zero.eval.bootstrap.paired_bootstrap`.

To be verified (check each item at the first run in Step 2):
- the names of the two registries `local_inference_model_map` / `MODEL_CONFIG_MAPPING`, and the
  fields of `ModelConfig`;
- the structure of `function` in `_format_prompt(messages, function)` (this code wraps it as
  `{"type": "function", "function": f}`);
- the file names and column names of the score CSV files (`collect` parses them loosely).
"""

from __future__ import annotations

import argparse
import csv
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
    """Return the ZeroFCHandler class (bfcl-eval must be installed). Not verified yet."""
    from bfcl_eval.model_handler.local_inference.qwen_fc import QwenFCHandler

    class ZeroFCHandler(QwenFCHandler):
        def _format_prompt(
            self, messages: list[dict[str, Any]], function: list[dict[str, Any]]
        ) -> str:
            tools = [
                f if "function" in f else {"type": "function", "function": f} for f in function
            ]
            return self.tokenizer.apply_chat_template(
                messages, tools=tools or None, add_generation_prompt=True, tokenize=False
            )

    return ZeroFCHandler


def register_zero_model(name: str, hf_dir: str | os.PathLike, display_name: str = "") -> None:
    """Register our model in the model table of BFCL (only in this process). Not verified yet."""
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
    """Register the model, then call the command-line entry point of bfcl (typer app). Not verified yet."""
    if not bfcl_available():
        raise SystemExit("bfcl-eval is not installed: uv pip install bfcl-eval==<version frozen in the preregistration>")
    register_zero_model(name, hf_dir)
    from bfcl_eval.__main__ import cli

    argv = [cmd, "--model", name, *extra]
    if cmd == "generate":
        argv += ["--local-model-path", str(hf_dir)]
    sys.argv = ["bfcl", *argv]
    cli()


def collect_scores(score_dir: str | os.PathLike, name: str) -> dict[str, Any]:
    """Find the row of `name` in the CSV files of the BFCL score directory, and return {column name: value}. Loose parsing, to be verified."""
    out: dict[str, Any] = {}
    for p in sorted(Path(score_dir).rglob("*.csv")):
        with open(p, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if any(name in str(v) for v in row.values()):
                    out[p.stem] = row
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="BFCL adapter (for Step 2, not verified yet)")
    ap.add_argument("command", choices=["generate", "evaluate", "collect"])
    ap.add_argument("--hf-dir", default="")
    ap.add_argument("--name", default="zero-FC")
    ap.add_argument("--score-dir", default="")
    args, extra = ap.parse_known_args(argv)
    if args.command == "collect":
        print(json.dumps(collect_scores(args.score_dir, args.name), ensure_ascii=False, indent=2))
        return
    run_cli(args.command, args.hf_dir, args.name, extra)


if __name__ == "__main__":
    main()
