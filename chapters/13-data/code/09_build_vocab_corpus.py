"""把 zero.data.download 下载的样本整理成 08_vocab_size.py 用的测量语料（第 10 节）。

    uv run python -m zero.data.download --config configs/vocab/download.toml --out <raw>
    uv run python chapters/13-data/code/09_build_vocab_corpus.py --raw <raw> --out <corpus>
    uv run python chapters/13-data/code/08_vocab_size.py --corpus <corpus> --refs --train-mb 30

<raw> 下每个来源一个目录（JSONL 分片 + _manifest.json，带出处和数据集 commit）。本脚本：
1. 按来源名把文档归到三种"语言"：英文（FineWeb-Edu、DCLM、FineMath）、中文（FineWeb-2、Ultra-FineWeb）、
   代码（UltraData-Code-L2 的 9 种编程语言）；
2. 每种语言内部把文档打乱（固定种子）——08 只取训练文件的前 N 字节，打乱后前缀里各来源的比例才和下载
   比例一致（下载量本身按主线配比分配，见 configs/vocab/download.toml）；
3. 先切出验证集（按字节数，与正文原来的验证集大小相当），其余是训练集；文档之间空一行；
4. 写 manifest.json：每种语言、每个来源的文档数和字节数，数据集 repo / config / split / revision，
   以及输出文件的 sha256，作为这份语料的完整记录。

只读取文本，不做任何过滤和改写：这些数据集在发布前已经做过清洗，测分词器要的就是它们原本的样子。
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
from pathlib import Path

SEED = 0
LANG_OF = {  # 来源名前缀 → 语言
    "fineweb-edu": "en",
    "dclm-baseline": "en",
    "finemath": "en",
    "fineweb-2-zh": "zh",
    "ultra-fineweb-zh": "zh",
    "ultradata-code-": "code",
}
VAL_BYTES = {"en": 1_750_000, "zh": 800_000, "code": 1_600_000}  # 与正文原来的验证集大小相当


def lang_of(source: str) -> str:
    for prefix, lang in LANG_OF.items():
        if source == prefix or (prefix.endswith("-") and source.startswith(prefix)):
            return lang
    raise KeyError(f"不认识的来源 {source!r}，先在 LANG_OF 里登记")


def read_source(d: Path) -> tuple[list[tuple[str, str]], dict]:
    """一个来源目录 → [(来源名, 正文)]，以及它的下载清单。"""
    manifest = json.loads((d / "_manifest.json").read_text("utf-8"))
    docs = []
    for shard in sorted(d.glob("*.jsonl.gz")):
        with gzip.open(shard, "rt", encoding="utf-8") as f:
            for line in f:
                text = json.loads(line)["text"].strip()
                if text:
                    docs.append((d.name, text))
    return docs, manifest


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--raw", type=Path, required=True, help="zero.data.download 的 --out 目录")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    by_lang: dict[str, list[tuple[str, str]]] = {"en": [], "zh": [], "code": []}
    sources = {}
    for d in sorted(p for p in args.raw.iterdir() if (p / "_manifest.json").exists()):
        docs, m = read_source(d)
        by_lang[lang_of(d.name)] += docs
        spec = m["spec"]
        sources[d.name] = {
            "lang": lang_of(d.name),
            "repo": spec["repo"],
            "config": spec["config"],
            "split": spec["split"],
            "revision": spec["revision"],
            "license": m["license"],
            "docs": len(docs),
            "bytes": sum(len(t.encode("utf-8")) for _, t in docs),
        }

    args.out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    summary = {}
    print(f"{'语言':<6}{'集合':<6}{'文档':>8}{'字节':>14}   各来源字节占比")
    for lang, docs in by_lang.items():
        rng.shuffle(docs)
        val, train, nval = [], [], 0
        for src, text in docs:
            if nval < VAL_BYTES[lang]:
                val.append((src, text))
                nval += len(text.encode("utf-8"))
            else:
                train.append((src, text))
        summary[lang] = {}
        for split, part in (("train", train), ("val", val)):
            path = args.out / f"{lang}_{split}.txt"
            path.write_text("\n\n".join(t for _, t in part) + "\n", encoding="utf-8")
            per_src: dict[str, int] = {}
            for src, t in part:
                per_src[src] = per_src.get(src, 0) + len(t.encode("utf-8"))
            total = sum(per_src.values())
            summary[lang][split] = {
                "file": path.name,
                "docs": len(part),
                "bytes": total,
                "sha256": sha256(path),
                "bytes_by_source": per_src,
            }
            share = "、".join(f"{s} {b / total:.0%}" for s, b in sorted(per_src.items(), key=lambda x: -x[1]))
            print(f"{lang:<6}{split:<6}{len(part):>8,}{total:>14,}   {share}")

    record = {"seed": SEED, "val_bytes": VAL_BYTES, "sources": sources, "corpus": summary}
    (args.out / "manifest.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), "utf-8")
    print(f"\n写入 {args.out}（manifest.json 记录了每个来源的 commit 和每个文件的 sha256）")


if __name__ == "__main__":
    main()
