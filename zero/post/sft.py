"""Supervised fine-tuning (SFT): conversations → tokens + loss mask → packing → training (Chapter 16).

    uv run python -m zero.post.sft --config configs/tiny/sft.toml
    uv run torchrun --standalone --nproc_per_node=8 -m zero.post.sft --config configs/main/sft.toml
    # nproc_per_node=1 was verified on one RTX 3090 (2026-10, see runs/2026-10-01-gpu0-check/). 8 GPUs are not verified yet.

Steps:

1. **Data**: JSONL, one `{"messages": [...], "tools": [...]}` on each line (format: see zero/DESIGN.md).
   If the config sets `[sft] generate_train = N` and the file does not exist, `zero/post/envs/tool_env.py`
   generates N tool-calling conversations (gold solution trajectories) as toy data.
2. **Rendering + loss mask**: `zero.post.chat.render(..., tokenizer=tok)` gives the ids and a mask with
   one value per token. Only the assistant output (with tool calls and `<|im_end|>`) has mask=1.
3. **Packing**: put several conversations into windows of length seq_len+1. **Do not cut a
   conversation** (first-fit). The rest of the window is `<|endoftext|>` padding with mask=0. A
   conversation that is longer than the window is dropped and counted.
   Packing removes most of the padding. The cost: the conversations in one window can "see" each other
   (there is no document masking, the same as nanochat and other implementations). In Step 2, the
   varlen interface of FlashAttention can isolate them. This is not verified on a GPU yet.
4. **Training**: use the pretraining `Trainer` again (`[data] format = "sft"` makes it read
   `MaskedWindowLoader`). The targets with mask=0 are -100, and the cross-entropy ignores them.
   `[train] init_from` points to the mid-training checkpoint.

Note: with gradient accumulation, each micro-batch takes the mean over its own assistant tokens, and
then the result is the mean over the micro-batches. This is a little different from "a mean over the
full batch, weighted by the token count". Common implementations do the same, and the effect is small.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from zero.config import Config, DataSourceConfig
from zero.post.chat import render
from zero.post.common import (
    SLOW_WAIT_MIN,
    load_post_config,
    read_jsonl,
    run_on_rank0,
    write_jsonl,
)
from zero.tokenizer import Tokenizer


@dataclass
class SFTConfig:
    train_jsonl: str = ""  # training conversations, JSONL
    val_jsonl: str = ""  # validation conversations, JSONL (can be empty)
    shard_dir: str = ""  # the packed windows go here (<shard_dir>/train.bin/.mask, val.bin/.mask)
    generate_train: int = 0  # if >0 and train_jsonl does not exist, tool_env generates this many conversations
    generate_val: int = 0
    env_seed: int = 0
    enable_thinking: bool = False
    overwrite: bool = False  # pack again if the window files already exist


# ---------------------------------------------------------------------------
# Encoding and packing
# ---------------------------------------------------------------------------


def encode_example(
    example: dict[str, Any], tok: Tokenizer, enable_thinking: bool = False
) -> tuple[list[int], list[bool]]:
    """One conversation → (ids, mask), with one <|endoftext|> (mask=0) at the end as a separator."""
    ids, mask = render(
        example["messages"], example.get("tools"), tokenizer=tok, enable_thinking=enable_thinking
    )
    return list(ids) + [tok.eot_id], list(mask) + [False]  # type: ignore[arg-type]


def pack_examples(
    examples: Iterable[tuple[list[int], list[bool]]],
    window: int,
    pad_id: int,
    lookback: int = 64,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """First-fit packing. Return tokens (N, window) uint32, mask (N, window) uint8, and statistics.

    Each sample goes into the first window that has space, among the last `lookback` windows that are
    not full. If no window has space, start a new window. Drop the samples without assistant tokens and
    the samples that are longer than the window."""
    bins: list[tuple[list[int], list[bool]]] = []
    open_idx: list[int] = []
    stats = {"examples": 0, "dropped_too_long": 0, "dropped_no_target": 0}
    for ids, mask in examples:
        if len(ids) > window:
            stats["dropped_too_long"] += 1
            continue
        if not any(mask[1:]):
            stats["dropped_no_target"] += 1
            continue
        stats["examples"] += 1
        placed = False
        for j in open_idx[-lookback:]:
            b_ids, b_mask = bins[j]
            if len(b_ids) + len(ids) <= window:
                b_ids.extend(ids)
                b_mask.extend(mask)
                placed = True
                if window - len(b_ids) < 16:
                    open_idx.remove(j)
                break
        if not placed:
            bins.append((list(ids), list(mask)))
            open_idx.append(len(bins) - 1)
    tokens = np.full((len(bins), window), pad_id, dtype=np.uint32)
    masks = np.zeros((len(bins), window), dtype=np.uint8)
    for i, (b_ids, b_mask) in enumerate(bins):
        tokens[i, : len(b_ids)] = b_ids
        masks[i, : len(b_mask)] = b_mask
    stats["windows"] = len(bins)
    stats["target_tokens"] = int(masks[:, 1:].sum())
    stats["fill_ratio_pct"] = int(100 * sum(len(b[0]) for b in bins) / max(len(bins) * window, 1))
    return tokens, masks, stats


def write_windows(tokens: np.ndarray, masks: np.ndarray, path_bin: Path) -> None:
    path_bin.parent.mkdir(parents=True, exist_ok=True)
    tokens.astype(np.uint32).tofile(path_bin)
    masks.astype(np.uint8).tofile(path_bin.with_suffix(".mask"))


def build_sft_shards(
    jsonl: str | os.PathLike,
    tok: Tokenizer,
    seq_len: int,
    out_bin: Path,
    enable_thinking: bool = False,
) -> dict[str, int]:
    rows = read_jsonl(jsonl)
    enc = (encode_example(r, tok, enable_thinking) for r in rows)
    tokens, masks, stats = pack_examples(enc, seq_len + 1, tok.eot_id)
    if len(tokens) == 0:
        raise ValueError(f"{jsonl}: no windows after packing (are all samples too long? seq_len={seq_len})")
    write_windows(tokens, masks, out_bin)
    meta = {"source": str(jsonl), "tokenizer_hash": tok.hash(), "seq_len": seq_len, **stats}
    out_bin.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    return stats


def env_conversations(n: int, seed: int, split: str) -> list[dict[str, Any]]:
    """Use the tool environment to generate n gold-solution conversations (SFT toy data).

    train and dev do not overlap.
    """
    from zero.post.envs.tool_env import make_splits, reference_messages

    train, dev = make_splits(n if split == "train" else 0, n if split != "train" else 0, seed)
    tasks = train if split == "train" else dev
    return [{"messages": reference_messages(t), "tools": t.tools, "task_id": t.id} for t in tasks]


def prepare_sft_data(cfg: Config, sft: SFTConfig, log: Callable[[str], None] = print) -> None:
    """Generate (optional) → encode and pack → write the window paths into cfg.train.data."""
    d = cfg.train.data
    tok = Tokenizer.load(d.tokenizer)
    shard_dir = Path(sft.shard_dir or Path(cfg.train.out_dir) / "data")
    for split, jsonl, n in (
        ("train", sft.train_jsonl, sft.generate_train),
        ("val", sft.val_jsonl, sft.generate_val),
    ):
        if not jsonl:
            continue
        if not Path(jsonl).exists():
            if n <= 0:
                raise FileNotFoundError(f"[sft] {jsonl} not found, and generate_{split} is not set")
            rows = env_conversations(n, sft.env_seed, "train" if split == "train" else "dev")
            write_jsonl(jsonl, rows)
            log(f"[sft] tool_env generated {len(rows)} {split} conversations → {jsonl}")
        out_bin = shard_dir / f"{split}.bin"
        if out_bin.exists() and not sft.overwrite:
            meta = json.loads(out_bin.with_suffix(".json").read_text())
            if meta["tokenizer_hash"] != tok.hash() or meta["seq_len"] != d.seq_len:
                raise ValueError(
                    f"{out_bin} does not match the current tokenizer or seq_len. Set [sft] overwrite = true to pack again"
                )
        else:
            stats = build_sft_shards(jsonl, tok, d.seq_len, out_bin, sft.enable_thinking)
            log(f"[sft] {split}: {stats}")
        if split == "train":
            d.sources = [DataSourceConfig(name="sft", path=str(out_bin), weight=1.0)]
        else:
            d.val = str(out_bin)
    if not d.sources:
        raise ValueError("[sft] needs train_jsonl")


def run_sft(
    src: str | os.PathLike | dict[str, Any],
    overrides: Sequence[str] | None = None,
    log: Callable[[str], None] | None = None,
) -> list[dict[str, Any]]:
    from zero.train.dist import cleanup, init_distributed
    from zero.train.trainer import Trainer

    cfg, sec = load_post_config(src, {"sft": SFTConfig}, overrides)
    if cfg.train.data.format != "sft":
        raise ValueError('The SFT config needs [data] format = "sft"')
    # Rank 0 packs the data (minutes for a large mixture) while the others wait (a slow wait), and
    # an error on rank 0 stops every rank.
    info = init_distributed(cfg.train.device, slow_wait_min=SLOW_WAIT_MIN)
    try:
        run_on_rank0(info, lambda: prepare_sft_data(cfg, sec["sft"], log or print))
        if not info.is_main:
            # only fills in the paths: the files exist already (overwrite = false, or every rank would pack again)
            prepare_sft_data(cfg, replace(sec["sft"], overwrite=False), lambda _: None)
        os.makedirs(cfg.train.out_dir, exist_ok=True)
        return Trainer(cfg, info, log).train()
    finally:
        cleanup()


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="SFT (Chapter 16)")
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = ap.parse_args(argv)
    run_sft(args.config, args.set)


if __name__ == "__main__":
    main()
