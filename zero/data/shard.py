"""分片：把文本分词后写成 uint32 的二进制分片（对应第 14 章）。

预训练时不能每步现场分词（太慢），所以先把全部文本分好词，按顺序写成一串 token id：

    文档1的token … <|endoftext|> 文档2的token … <|endoftext|> …

每篇文档后面跟一个 <|endoftext|>，模型由此学会"一篇文章到这里结束"。
这串 token 切成若干个文件 `<name>_<idx>.bin`（`np.uint32`，没有文件头，可以直接 memmap），
另写一个 `<name>.json` 记录元数据：分词器哈希、每个分片的 token 数、来源等。

为什么用 uint32 而不是 uint16：uint16 最多表示 65535，而本课的词表可能超过 64K（第 13 章实测决定）。
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
    """分词并写分片；每个分片最多 shard_tokens 个 token（文档可能跨分片，加载器不在意文档边界）。

    返回写出的 .bin 路径列表，同时写 `<out_dir>/<name>.json`。
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
        os.replace(tmp, p)  # 原子写入：要么是完整文件，要么没有
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
    """只读 memmap 打开一个分片（不会把整个文件读进内存）。"""
    return np.memmap(path, dtype=TOKEN_DTYPE, mode="r")


def load_metadata(path: str | os.PathLike) -> dict[str, Any]:
    """读取 <name>.json；也可以传某个 .bin，自动找同名前缀的 json。"""
    p = Path(path)
    if p.suffix == ".bin":
        stem = p.stem.rsplit("_", 1)[0]
        p = p.with_name(f"{stem}.json")
    with open(p) as f:
        return json.load(f)
