"""重新生成 assets/tiny_corpus/ 的脚本（记录语料是怎么来的，便于复查与复现）。

    git clone --depth 1 https://github.com/chinese-poetry/chinese-poetry.git /tmp/chinese-poetry
    curl -sSLo /tmp/shakespeare.txt \
        https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
    uv run python assets/tiny_corpus/build_corpus.py --poetry-dir /tmp/chinese-poetry \
        --shakespeare /tmp/shakespeare.txt

输出三个纯文本文件，文档之间用空行分隔：
- shakespeare.txt       英文（原样复制）
- chinese_poetry.txt    中文：论语、诗经、水墨唐诗、宋词三百首，再用全宋词补足到约 1.2MB（均为简体）
- code.txt              代码：本仓库 zero/ 下的部分 Python 文件
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

HERE = Path(__file__).parent
REPO = HERE.parent.parent
ZH_BUDGET_BYTES = 1_200_000
CODE_BUDGET_BYTES = 300_000


def record_to_text(r: dict) -> str:
    head = [
        r[k]
        for k in ("title", "chapter", "section", "author", "rhythmic")
        if isinstance(r.get(k), str) and r[k]
    ]
    body = []
    for k in ("paragraphs", "content"):
        v = r.get(k)
        if isinstance(v, list):
            body.extend(x for x in v if isinstance(x, str))
    if isinstance(r.get("prologue"), str):
        body.append(r["prologue"])
    return "\n".join(head + body).strip()


def build_chinese(poetry: Path) -> str:
    files = [
        poetry / "论语" / "lunyu.json",
        poetry / "诗经" / "shijing.json",
        poetry / "水墨唐诗" / "shuimotangshi.json",
        poetry / "宋词" / "宋词三百首.json",
    ] + [poetry / "宋词" / f"ci.song.{i}.json" for i in range(0, 22000, 1000)]
    docs: list[str] = []
    size = 0
    for f in files:
        for r in json.loads(f.read_text("utf-8")):
            t = record_to_text(r)
            if not t:
                continue
            docs.append(t)
            size += len(t.encode("utf-8")) + 2
            if size >= ZH_BUDGET_BYTES:
                return "\n\n".join(docs) + "\n"
    return "\n\n".join(docs) + "\n"


def build_code() -> str:
    parts: list[str] = []
    size = 0
    for f in sorted((REPO / "zero").rglob("*.py")):
        rel = f.relative_to(REPO)
        text = f"# ===== {rel} =====\n" + f.read_text("utf-8")
        if size + len(text.encode("utf-8")) > CODE_BUDGET_BYTES:
            break
        parts.append(text)
        size += len(text.encode("utf-8"))
    return "\n\n".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--poetry-dir", required=True, type=Path)
    ap.add_argument("--shakespeare", required=True, type=Path)
    args = ap.parse_args()
    shutil.copyfile(args.shakespeare, HERE / "shakespeare.txt")
    (HERE / "chinese_poetry.txt").write_text(build_chinese(args.poetry_dir), "utf-8")
    (HERE / "code.txt").write_text(build_code(), "utf-8")
    for name in ("shakespeare.txt", "chinese_poetry.txt", "code.txt"):
        print(name, (HERE / name).stat().st_size, "bytes")


if __name__ == "__main__":
    main()
