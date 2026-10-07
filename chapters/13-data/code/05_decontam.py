"""Decontamination: an n-gram overlap check between the training documents and the test
questions (GPT-3 uses 13-grams; Llama 3 uses 8-grams for its contamination analysis).

Method: normalize each test question (lowercase, remove punctuation; each Chinese character is
one word), split it into n-grams, and put them in a set. Then scan the n-grams of each training
document. If one n-gram is in the set, the document "saw a test question". Remove the full
document.

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
    """Lowercase → keep only alphanumeric strings and single Chinese characters.

    This removes all differences in case, punctuation, and whitespace.
    """
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
    """Return {document index: set of the ids of the test questions that it matches}."""
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
    print(f"After the quality filter: {len(docs)} documents; {len(leaked)} of them really contain a test question: {dict(variants)}")
    print(f"Test questions: {len(eval_items)}\n")

    print(f"{'n':>4}{'Flagged':>8}{'Exact':>6}{'Format':>10}{'Para.':>6}{'Other':>10}")
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
            print(f"Example of an 'other hit' at n=13: {''.join(g)!r}. This poem occurs two times in the corpus"
                  " (one copy in 'Three Hundred Song Ci Poems' and one in 'Complete Song Ci'). The test"
                  " question comes from one copy: this is a real leak, not a false positive")
            break
    print(f"\n13-gram decontamination: {len(docs)} → {len(clean)} documents; "
          f"test questions hit: {len(set().union(*hits.values()))} (write this in the model card)")
    fp_docs = [docs[i] for i in find_contaminated(docs, eval_items, 5) if docs[i]["kind"] != "contaminated"]
    if fp_docs:
        i = next(iter(find_contaminated(fp_docs, eval_items, 5)))
        idx5 = build_index(eval_items, 5)
        g = next(g for g in ngrams(norm_tokens(fp_docs[i]["text"]), 5) if g in idx5)
        print(f"Example of an 'other hit' at n=5: the shared 5-gram is {' '.join(g)!r} (a common phrase, not a leak)")


if __name__ == "__main__":
    main()
