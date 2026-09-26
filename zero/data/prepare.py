"""从原始文本一步做出"分词器 + 训练/验证分片"（对应第 13、14 章；主要给 tiny 冒烟用）。

正式训练的数据流水线是：下载 → clean → dedup → quality → decontam → shard，每步单独跑、单独检查。
tiny 配置为了"一条命令就能训练"，把这些步骤串成 `prepare_data(cfg.train.data)`：

1. 读 `[data.prepare] raw_files` 里的每个文本文件（文件名 = 来源名），按空行切段拼成文档；
2. 清洗、精确去重；
3. `data.tokenizer` 不存在时，用全部原始文件训练一个 byte-level BPE；
4. 每个来源按固定种子打乱文档，切出 val_fraction 做验证集，写成
   `<out_dir>/<来源>_train_*.bin` 和 `<out_dir>/<来源>_val_*.bin`。

分片已经存在时直接跳过（分词器哈希不一致会报错）。
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
            raise FileNotFoundError(f"[data.prepare] raw_files 找不到：{pat}")
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
        log(f"[prepare] 训练分词器 vocab_size={p.vocab_size}，语料 {[f.name for f in files]}")
        tok = train_bpe(files, vocab_size=p.vocab_size)
        tok.save(tok_path)
        log(f"[prepare] 分词器已保存到 {tok_path}")

    for f in files:
        name = f.stem
        meta_path = out / f"{name}_train.json"
        if meta_path.exists() and (out / f"{name}_val.json").exists():
            meta = load_metadata(meta_path)
            if meta["tokenizer_hash"] != tok.hash():
                raise ValueError(f"{meta_path} 是用另一个分词器切的，请删掉 {out} 后重新生成")
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
        log(f"[prepare] {name}: {len(train)} 篇训练文档, {len(val)} 篇验证文档")
