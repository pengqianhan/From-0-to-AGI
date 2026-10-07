"""Make "tokenizer + training/validation shards" from raw text in one step (Chapters 13 and 14; mainly for the tiny smoke test).

The data pipeline for real training is: download → clean → dedup → quality → decontam → shard.
Each step runs and is checked separately. The tiny configuration must "train with one command",
so `prepare_data(cfg.train.data)` joins these steps:

1. Read each text file in `[data.prepare] raw_files` (file name = source name). Split it at blank
   lines and join the paragraphs into documents.
2. Clean, and do exact deduplication.
3. If `data.tokenizer` does not exist, train a byte-level BPE on all raw files.
4. For each source, shuffle the documents with a fixed seed. Use the val_fraction part as the
   validation set. Write `<out_dir>/<source>_train_*.bin` and `<out_dir>/<source>_val_*.bin`.

If the shards already exist, skip them (if the tokenizer hash is different, raise an error).
"""

from __future__ import annotations

import glob
import random
from collections.abc import Callable
from pathlib import Path

from zero.config import DataConfig
from zero.data.clean import clean_document, split_into_documents
from zero.data.dedup import exact_dedup
from zero.data.shard import load_metadata, write_shards
from zero.tokenizer import Tokenizer, train_bpe


def _raw_files(patterns: list[str]) -> list[Path]:
    files: list[Path] = []
    for pat in patterns:
        matches = sorted(glob.glob(pat))
        if not matches:
            raise FileNotFoundError(f"[data.prepare] raw_files not found: {pat}")
        files.extend(Path(m) for m in matches)
    return files


def prepare_data(cfg: DataConfig, seed: int = 0, log: Callable[[str], None] = print) -> None:
    p = cfg.prepare
    if p is None:
        return
    files = _raw_files(p.raw_files)
    out = Path(p.out_dir)
    tok_path = Path(cfg.tokenizer)

    if tok_path.exists():
        tok = Tokenizer.load(tok_path)
    else:
        log(f"[prepare] training the tokenizer, vocab_size={p.vocab_size}, corpus {[f.name for f in files]}")
        tok = train_bpe(files, vocab_size=p.vocab_size)
        tok.save(tok_path)
        log(f"[prepare] tokenizer saved to {tok_path}")

    for f in files:
        name = f.stem
        meta_path = out / f"{name}_train.json"
        if meta_path.exists() and (out / f"{name}_val.json").exists():
            meta = load_metadata(meta_path)
            if meta["tokenizer_hash"] != tok.hash():
                raise ValueError(f"{meta_path} was made with a different tokenizer. Delete {out} and make it again")
            continue
        docs = [
            d
            for d in (
                clean_document(x) for x in split_into_documents(f.read_text("utf-8"), p.doc_chars)
            )
            if d
        ]
        docs = [docs[i] for i in exact_dedup(docs)]
        random.Random(f"{seed}-{name}").shuffle(docs)
        n_val = max(1, int(len(docs) * p.val_fraction))
        val, train = docs[:n_val], docs[n_val:]
        write_shards(train, tok, out, f"{name}_train", source=str(f))
        write_shards(val, tok, out, f"{name}_val", source=str(f))
        log(f"[prepare] {name}: {len(train)} training documents, {len(val)} validation documents")
