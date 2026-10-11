"""Import an open Qwen3 dense model into a zero checkpoint (proxy base for post-training).

    uv run python -m zero.tools.import_hf <HF folder> out/proxy/base \
        --check-config configs/proxy/sft.toml

Why: the post-training pipeline (`zero.post.*`) starts from a zero checkpoint (`train.init_from`).
Before our own base model is ready, we run the same pipeline on an open base with the same
architecture (Qwen3-0.6B-Base: dense, GQA, QK-Norm, tied embeddings; see `runs/POSTTRAIN_PLAN.md`).
This tool writes `<out>/step_00000000/{model.pt, meta.json}` + `<out>/tokenizer.json`, so that every
stage reads it like one of our own checkpoints, with its own tokenizer.

The Qwen3 tokenizer has all special tokens that our chat template needs (`<|endoftext|>`,
`<|im_start|>`, `<|im_end|>`, `<tool_call>`, `<tool_response>`, `<think>`), so `zero.post.chat`
renders the same text for it. The tool checks this.

`--check-config`: compare the [model] section of a post-training config with the imported model, so
that a wrong shape fails here and not in the middle of a run.
"""

from __future__ import annotations

import argparse
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any

from zero.config import ModelConfig

NEEDED_TOKENS = (
    "<|endoftext|>",
    "<|im_start|>",
    "<|im_end|>",
    "<tool_call>",
    "</tool_call>",
    "<tool_response>",
    "</tool_response>",
)
SHAPE_FIELDS = (
    "vocab_size",
    "dim",
    "n_layers",
    "n_heads",
    "n_kv_heads",
    "head_dim",
    "ffn_dim",
    "rope_theta",
    "norm_eps",
    "qk_norm",
    "tie_embeddings",
)


def compare_shape(a: ModelConfig, b: ModelConfig) -> list[str]:
    """The fields of SHAPE_FIELDS that differ, as "field: a != b"."""
    return [
        f"{k}: {getattr(a, k)} != {getattr(b, k)}"
        for k in SHAPE_FIELDS
        if getattr(a, k) != getattr(b, k)
    ]


def import_hf(
    hf_dir: str | Path,
    out_dir: str | Path,
    max_seq_len: int | None = None,
    check_config: str | Path | None = None,
) -> dict[str, Any]:
    """HF Qwen3 folder → zero checkpoint folder. Returns {"ckpt", "tokenizer", "model"}."""
    import torch

    from zero.hf import load_from_hf_qwen3
    from zero.model import Transformer
    from zero.tokenizer import Tokenizer
    from zero.train.checkpoint import save_checkpoint

    hf_dir, out_dir = Path(hf_dir), Path(out_dir)
    tok_src = hf_dir / "tokenizer.json"
    if not tok_src.exists():
        raise FileNotFoundError(f"{tok_src} not found (a fast tokenizer.json is needed)")
    tok = Tokenizer.load(tok_src)
    missing = [t for t in NEEDED_TOKENS if tok.token_to_id(t) is None]
    if missing:
        raise ValueError(
            f"The tokenizer has no {missing}; the chat template of zero.post.chat needs them"
        )

    model = load_from_hf_qwen3(hf_dir)
    cfg = model.config
    if max_seq_len is not None and max_seq_len != cfg.max_seq_len:
        cfg = ModelConfig(**{**asdict(cfg), "max_seq_len": max_seq_len})
        sd = model.state_dict()
        model = Transformer(cfg)
        model.load_state_dict(sd)
    if check_config is not None:
        from zero.config import load_model_config

        diff = compare_shape(load_model_config(check_config), cfg)
        if diff:
            raise ValueError(f"{check_config} [model] does not match {hf_dir}: " + "; ".join(diff))

    out_dir.mkdir(parents=True, exist_ok=True)
    tok_dst = out_dir / "tokenizer.json"
    shutil.copyfile(tok_src, tok_dst)
    meta = {
        "config": {"model": asdict(cfg), "train": {"data": {"tokenizer": str(tok_dst)}}},
        "imported_from": str(hf_dir),
    }
    ckpt = save_checkpoint(out_dir, 0, model.to(torch.float32), meta=meta)
    return {"ckpt": str(ckpt), "tokenizer": str(tok_dst), "model": asdict(cfg)}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description="Import an HF Qwen3 dense model into a zero checkpoint"
    )
    ap.add_argument("hf_dir")
    ap.add_argument("out_dir")
    ap.add_argument(
        "--max-seq-len",
        type=int,
        default=None,
        help="RoPE table length (default: from config.json)",
    )
    ap.add_argument(
        "--check-config", default=None, help="a post-training config whose [model] must match"
    )
    args = ap.parse_args(argv)
    r = import_hf(args.hf_dir, args.out_dir, args.max_seq_len, args.check_config)
    print(f"checkpoint: {r['ckpt']}\ntokenizer:  {r['tokenizer']}")
    print("[model]")
    for k in SHAPE_FIELDS + ("max_seq_len",):
        print(f"{k} = {r['model'][k]!r}".replace("True", "true").replace("False", "false"))


if __name__ == "__main__":
    main()
