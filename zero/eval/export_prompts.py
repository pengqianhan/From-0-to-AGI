"""Export the questions of the preregistered general benchmarks for decontamination (Chapters 11 and 16).

    uv run --extra data python -m zero.eval.export_prompts --out data/eval/prompts
    # → data/eval/prompts/<benchmark>.jsonl, one {"question": ...} per item

`zero.post.sft_data` (`decontam_texts`) and `zero.data.decontam` read these files: a training
conversation whose user text shares a 13-gram with a question is dropped (GOAL.md 3.2). The tool-call
benchmarks (BFCL, ACEBench) are exported by `zero.post.envs.fc_tasks export` instead.

The table follows section 2.2 of eval/PREREGISTRATION.md. The Hugging Face ids and splits are those of
the draft; **check them when the preregistration is frozen** (data revision, subset). Each spec lists
candidate question fields; the first field present in a row is used (CMMLU writes "Question").
For decontamination the exact split matters less than coverage: when in doubt, export more splits.
Needs the `datasets` library and access to Hugging Face (`uv sync --extra data`).
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PromptSpec:
    name: str
    hf_id: str
    splits: tuple[str, ...]
    fields: tuple[str, ...]
    configs: tuple[
        str, ...
    ] = ()  # empty: all configs of the data set (subjects of MMLU / C-Eval / CMMLU)


SPECS: tuple[PromptSpec, ...] = (
    PromptSpec("mmlu_redux", "edinburgh-dawg/mmlu-redux-2.0", ("test",), ("question",)),
    PromptSpec("mmlu_pro", "TIGER-Lab/MMLU-Pro", ("test", "validation"), ("question",)),
    PromptSpec("ceval", "ceval/ceval-exam", ("val", "dev", "test"), ("question",)),
    PromptSpec("cmmlu", "haonan-li/cmmlu", ("test", "dev"), ("Question", "question")),
    PromptSpec("gsm8k", "openai/gsm8k", ("test",), ("question",), ("main",)),
    PromptSpec("math500", "HuggingFaceH4/MATH-500", ("test",), ("problem",)),
    PromptSpec("humaneval_plus", "evalplus/humanevalplus", ("test",), ("prompt",)),
    PromptSpec("mbpp_plus", "evalplus/mbppplus", ("test",), ("prompt", "text")),
    PromptSpec("ifeval", "google/IFEval", ("train",), ("prompt",)),
)

Loader = Callable[[str, str | None, str], Iterable[dict[str, Any]]]


def _hf_loader(
    hf_id: str, config: str | None, split: str
) -> Iterable[dict[str, Any]]:  # pragma: no cover
    from datasets import load_dataset

    return load_dataset(hf_id, config, split=split)


def _hf_configs(hf_id: str) -> list[str | None]:  # pragma: no cover
    from datasets import get_dataset_config_names

    names = get_dataset_config_names(hf_id)
    return list(names) or [None]


def export_spec(
    spec: PromptSpec,
    out_dir: str | Path,
    loader: Loader = _hf_loader,
    list_configs: Callable[[str], list[str | None]] = _hf_configs,
) -> dict[str, Any]:
    """Write <out_dir>/<name>.jsonl. Missing splits are skipped and reported."""
    configs: list[str | None] = list(spec.configs) or list_configs(spec.hf_id)
    seen: set[str] = set()
    missing: list[str] = []
    rows = []
    for c in configs:
        for split in spec.splits:
            try:
                data = loader(spec.hf_id, c, split)
            except (ValueError, KeyError, FileNotFoundError):
                missing.append(f"{c}/{split}")
                continue
            for r in data:
                q = next(
                    (r[f] for f in spec.fields if isinstance(r.get(f), str) and r[f].strip()), None
                )
                if q is None or q in seen:
                    continue
                seen.add(q)
                rows.append({"question": q, "config": c, "split": split})
    p = Path(out_dir) / f"{spec.name}.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return {
        "name": spec.name,
        "hf_id": spec.hf_id,
        "n": len(rows),
        "missing": missing,
        "out": str(p),
    }


def main(argv: list[str] | None = None) -> None:  # pragma: no cover
    ap = argparse.ArgumentParser(description="Export benchmark questions for decontamination")
    ap.add_argument("--out", default="data/eval/prompts")
    ap.add_argument("--only", nargs="*", default=[], help="names from SPECS")
    args = ap.parse_args(argv)
    for spec in SPECS:
        if args.only and spec.name not in args.only:
            continue
        print(json.dumps(export_spec(spec, args.out), ensure_ascii=False))


if __name__ == "__main__":  # pragma: no cover
    main()
