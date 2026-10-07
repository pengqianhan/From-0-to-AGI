"""Turn the samples from zero.data.download into the measurement corpus for 08_vocab_size.py (Section 10).

    uv run python -m zero.data.download --config configs/vocab/download.toml --out <raw>
    uv run python chapters/13-data/code/09_build_vocab_corpus.py --raw <raw> --out <corpus>
    uv run python chapters/13-data/code/08_vocab_size.py --corpus <corpus> --refs --train-mb 30

<raw> has one directory per source (JSONL shards + _manifest.json, with the provenance and the
data set commit). This script:
1. Puts the documents into three "languages" by the source name: English (FineWeb-Edu, DCLM,
   FineMath), Chinese (FineWeb-2, Ultra-FineWeb), code (the 9 programming languages of
   UltraData-Code-L2).
2. Shuffles the documents in each language (fixed seed). 08 takes only the first N bytes of the
   training file. After the shuffle, the share of each source in this prefix is the same as in
   the download. (The download amounts follow the main-line mixture; see
   configs/vocab/download.toml.)
3. Cuts the validation set first (by bytes, about the size of the original validation set in the
   text). The rest is the training set. An empty line separates the documents.
4. Writes manifest.json: the documents and bytes of each language and each source, the data set
   repo / config / split / revision, and the sha256 of each output file. This is the full record
   of the corpus.

The script only reads the text. It does not filter or rewrite anything: the data sets were
cleaned before release, and the tokenizer measurement needs them in their original form.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
from pathlib import Path

SEED = 0
LANG_OF = {  # source name prefix → language
    "fineweb-edu": "en",
    "dclm-baseline": "en",
    "finemath": "en",
    "fineweb-2-zh": "zh",
    "ultra-fineweb-zh": "zh",
    "ultradata-code-": "code",
}
VAL_BYTES = {"en": 1_750_000, "zh": 800_000, "code": 1_600_000}  # about the size of the original validation set in the text


def lang_of(source: str) -> str:
    for prefix, lang in LANG_OF.items():
        if source == prefix or (prefix.endswith("-") and source.startswith(prefix)):
            return lang
    raise KeyError(f"unknown source {source!r}; add it to LANG_OF first")


def read_source(d: Path) -> tuple[list[tuple[str, str]], dict]:
    """One source directory → [(source name, text)], and its download manifest."""
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
    ap.add_argument("--raw", type=Path, required=True, help="the --out directory of zero.data.download")
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
    print(f"{'Lang':<6}{'Split':<6}{'Docs':>8}{'Bytes':>14}   Share of bytes per source")
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
            share = ", ".join(f"{s} {b / total:.0%}" for s, b in sorted(per_src.items(), key=lambda x: -x[1]))
            print(f"{lang:<6}{split:<6}{len(part):>8,}{total:>14,}   {share}")

    record = {"seed": SEED, "val_bytes": VAL_BYTES, "sources": sources, "corpus": summary}
    (args.out / "manifest.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), "utf-8")
    print(f"\nWrote {args.out} (manifest.json records the commit of each source and the sha256 of each file)")


if __name__ == "__main__":
    main()
