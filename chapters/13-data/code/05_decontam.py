"""去污染：训练文档与考题的 n-gram 重叠检查（GPT-3 用 13-gram，Llama 3 用 8-gram 做污染分析）。

做法：把每道考题规范化（小写、去标点；中文每个字算一个词）后切成 n-gram，放进一个集合；
扫描每篇训练文档的 n-gram，只要有一个在集合里，就认为这篇文档"见过考题"，整篇删掉。

    uv run python chapters/13-data/code/05_decontam.py
"""

from __future__ import annotations

import importlib.util
import re
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


TOKEN = re.compile(r"[一-鿿]|[a-z0-9]+")


def norm_tokens(text: str) -> list[str]:
    """小写 → 只留字母数字串和单个汉字：大小写、标点、空白的差别全部抹掉。"""
    return TOKEN.findall(text.lower())


def ngrams(toks: list[str], n: int) -> set[tuple[str, ...]]:
    return {tuple(toks[i : i + n]) for i in range(len(toks) - n + 1)}


def build_index(eval_items: list[dict], n: int) -> dict[tuple[str, ...], set[str]]:
    index: dict[tuple[str, ...], set[str]] = {}
    for item in eval_items:
        for g in ngrams(norm_tokens(item["question"]), n):
            index.setdefault(g, set()).add(item["id"])
    return index


def find_contaminated(docs: list[dict], eval_items: list[dict], n: int = 13) -> dict[int, set[str]]:
    """返回 {文档下标: 撞上的考题 id 集合}。"""
    index = build_index(eval_items, n)
    hits = {}
    for i, d in enumerate(docs):
        found = set()
        for g in ngrams(norm_tokens(d["text"]), n):
            found |= index.get(g, set())
        if found:
            hits[i] = found
    return hits


def main() -> None:
    crawl_mod = _load("01_noisy_crawl")
    q = _load("04_quality_classifier")
    crawl = crawl_mod.build_crawl()
    docs = q.quality_filter(q.after_dedup())
    eval_items = crawl["eval"]
    leaked = [d for d in docs if d["kind"] == "contaminated"]
    variants = Counter(d["leak"]["variant"] for d in leaked)
    print(f"质量过滤后 {len(docs)} 篇，其中真的夹带了考题的 {len(leaked)} 篇：{dict(variants)}")
    print(f"考题 {len(eval_items)} 道\n")

    print(f"{'n':>4}{'标记的文档':>8}{'原样':>6}{'改大小写标点':>10}{'改写':>6}{'其它命中':>10}")
    for n in [5, 8, 13, 20]:
        hits = find_contaminated(docs, eval_items, n)
        flagged = [docs[i] for i in hits]
        caught = Counter(d["leak"]["variant"] for d in flagged if d["kind"] == "contaminated")
        fp = sum(d["kind"] != "contaminated" for d in flagged)
        print(f"{n:>4}{len(flagged):>8}{caught['verbatim']:>4}/{variants['verbatim']:<2}"
              f"{caught['case_punct']:>7}/{variants['case_punct']:<3}"
              f"{caught['paraphrase']:>4}/{variants['paraphrase']:<3}{fp:>9}")

    hits = find_contaminated(docs, eval_items, 13)
    clean = [d for i, d in enumerate(docs) if i not in hits]
    idx13 = build_index(eval_items, 13)
    for i in hits:
        if docs[i]["kind"] != "contaminated":
            g = next(g for g in ngrams(norm_tokens(docs[i]["text"]), 13) if g in idx13)
            print(f"n=13 的'其它命中'举例：{''.join(g)!r}——这首词在语料里本来就出现了两次"
                  "（《宋词三百首》和《全宋词》各一份），考题取自其中一份：这是真泄漏，不是误报")
            break
    print(f"\n13-gram 去污染：{len(docs)} → {len(clean)} 篇；"
          f"被命中的考题 {len(set().union(*hits.values()))} 道（写进模型卡）")
    fp_docs = [docs[i] for i in find_contaminated(docs, eval_items, 5) if docs[i]["kind"] != "contaminated"]
    if fp_docs:
        i = next(iter(find_contaminated(fp_docs, eval_items, 5)))
        idx5 = build_index(eval_items, 5)
        g = next(g for g in ngrams(norm_tokens(fp_docs[i]["text"]), 5) if g in idx5)
        print(f"n=5 的'其它命中'举例：共享的 5-gram 是 {' '.join(g)!r}（常见搭配，不是泄漏）")


if __name__ == "__main__":
    main()
