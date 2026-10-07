"""Shards: tokenize the text and write it into uint32 binary shards (Chapter 14).

Pretraining cannot tokenize the text at each step (too slow). So first tokenize all text, and write
the token ids in sequence:

    tokens of document 1 … <|endoftext|> tokens of document 2 … <|endoftext|> …

An <|endoftext|> follows each document. From it, the model learns "the text ends here".
The token sequence is split into several files `<name>_<idx>.bin` (`np.uint32`, no file header,
so memmap can open them directly). A separate `<name>.json` holds the metadata: the tokenizer hash,
the number of tokens in each shard, the source, and more.

Why uint32 and not uint16: uint16 can hold at most 65535, but the vocabulary of this course can be
larger than 64K (a measurement in Chapter 13 decides it).
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from zero.tokenizer import Tokenizer

TOKEN_DTYPE = np.uint32


def write_shards(
    texts: Iterable[str],
    tokenizer: Tokenizer,
    out_dir: str | os.PathLike,
    name: str,
    shard_tokens: int = 100_000_000,
    source: str = "",
    batch_docs: int = 256,
    extra_meta: dict[str, Any] | None = None,
) -> list[Path]:
    """Tokenize and write the shards. Each shard has at most shard_tokens tokens.

    A document can continue across shards; the loader does not care about document boundaries.
    Returns the list of .bin paths that it wrote. It also writes `<out_dir>/<name>.json`.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    eot = tokenizer.eot_id
    paths: list[Path] = []
    shard_meta: list[dict[str, Any]] = []
    buf = np.empty(shard_tokens, dtype=TOKEN_DTYPE)
    fill = 0
    n_docs = 0
    total = 0

    def flush() -> None:
        nonlocal fill
        if fill == 0:
            return
        p = out / f"{name}_{len(paths):05d}.bin"
        tmp = p.with_suffix(".bin.tmp")
        buf[:fill].tofile(tmp)
        os.replace(tmp, p)  # atomic write: the file is either complete or not there
        paths.append(p)
        shard_meta.append({"file": p.name, "num_tokens": int(fill)})
        fill = 0

    def add(ids: list[int]) -> None:
        nonlocal fill, total
        arr = np.asarray(ids, dtype=TOKEN_DTYPE)
        pos = 0
        while pos < len(arr):
            take = min(len(arr) - pos, shard_tokens - fill)
            buf[fill : fill + take] = arr[pos : pos + take]
            fill += take
            pos += take
            if fill == shard_tokens:
                flush()
        total += len(arr)

    batch: list[str] = []

    def process(batch: list[str]) -> None:
        nonlocal n_docs
        for ids in tokenizer.encode_batch(batch):
            add([*ids, eot])
            n_docs += 1

    for t in texts:
        batch.append(t)
        if len(batch) >= batch_docs:
            process(batch)
            batch = []
    if batch:
        process(batch)
    flush()

    meta = {
        "name": name,
        "dtype": "uint32",
        "tokenizer_hash": tokenizer.hash(),
        "vocab_size": tokenizer.vocab_size,
        "eot_id": eot,
        "num_tokens": int(total),
        "num_documents": n_docs,
        "source": source,
        "shards": shard_meta,
        **(extra_meta or {}),
    }
    tmp = out / f"{name}.json.tmp"
    with open(tmp, "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    os.replace(tmp, out / f"{name}.json")
    return paths


def read_shard(path: str | os.PathLike) -> np.memmap:
    """Open a shard as a read-only memmap (the full file is not read into memory)."""
    return np.memmap(path, dtype=TOKEN_DTYPE, mode="r")


def load_metadata(path: str | os.PathLike) -> dict[str, Any]:
    """Read <name>.json. You can also give a .bin file: the function finds the json with the same prefix."""
    p = Path(path)
    if p.suffix == ".bin":
        stem = p.stem.rsplit("_", 1)[0]
        p = p.with_name(f"{stem}.json")
    with open(p) as f:
        return json.load(f)
