"""Language identification + heuristic filters: some cheap statistical rules remove the pages
that are clearly not normal text.

The rules come from three papers (the thresholds are the values in the papers):
- Gopher (Rae et al. 2021, Appendix A): word count, mean word length, symbol ratio, fraction of
  words with a letter, stop words, duplicate lines, top 2-gram;
- C4 (Raffel et al. 2020): remove the full page if it contains "lorem ipsum" or a curly bracket;
- FineWeb (Penedo et al. 2024, Section 3.6): lines that end with punctuation ≤ 12%, characters
  in duplicate lines ≥ 10%, short lines ≥ 67%.
Chinese has no spaces between words. Thus one Chinese character counts as one "word", and the
stop words are Chinese function words.

    uv run python chapters/13-data/code/02_heuristic_filter.py
"""

from __future__ import annotations

import importlib.util
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))  # the repository root, so that we can import zero


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


crawl_mod = _load("01_noisy_crawl")

CJK = re.compile(r"[一-鿿]")
WORD = re.compile(r"[一-鿿]|[^\s一-鿿]+")
EN_STOP = {"the", "be", "to", "of", "and", "that", "have", "with"}  # the 8 stop words of Gopher
ZH_STOP = set("的了是在和有也就不都而与之其以")
END_PUNCT = tuple(".!?\"'。！？”’…」』")


def lang_id(text: str) -> str:
    """Minimal language identification: more than 30% Chinese characters is Chinese, more than
    50% ASCII letters is English, all else is "other".
    (FineWeb uses the fastText lid.176 model and keeps a page only if the English score ≥ 0.65.)"""
    chars = [c for c in text if not c.isspace()]
    n = max(len(chars), 1)
    if sum(bool(CJK.match(c)) for c in chars) / n > 0.3:
        return "zh"
    if sum(c.isascii() and c.isalpha() for c in chars) / n > 0.5:
        return "en"
    return "other"


def heuristic_reasons(text: str) -> list[str]:
    """Return the rules that this document breaks (an empty list = the document passes)."""
    words = WORD.findall(text)
    n = len(words)
    lines = [ln for ln in text.split("\n") if ln.strip()]
    nl = max(len(lines), 1)
    is_zh = lang_id(text) == "zh"
    bad = []
    # ---- Gopher quality rules ----
    if not 50 <= n <= 100_000:
        bad.append("gopher_word_count")
    if not is_zh and n and not 3 <= sum(map(len, words)) / n <= 10:
        bad.append("gopher_mean_word_len")
    if (text.count("#") + text.count("...") + text.count("…")) / max(n, 1) > 0.1:
        bad.append("gopher_symbol_ratio")
    if n and sum(bool(re.search(r"[^\W\d_]", w)) for w in words) / n < 0.8:
        bad.append("gopher_alpha_words")
    stop = ZH_STOP if is_zh else EN_STOP
    if sum(w.lower() in stop for w in words) < 2:
        bad.append("gopher_stop_words")
    # ---- Gopher repetition rules ----
    counts = Counter(lines)
    if sum(c for c in counts.values() if c > 1) / nl > 0.3:
        bad.append("gopher_dup_lines")
    if n >= 2:
        (gram, c), = Counter(zip(words, words[1:])).most_common(1)
        if c * (len(gram[0]) + len(gram[1])) / max(sum(map(len, words)), 1) > 0.2:
            bad.append("gopher_top_2gram")
    # ---- C4 ----
    if "lorem ipsum" in text.lower() or "{" in text:
        bad.append("c4_lorem_or_curly")
    # ---- FineWeb ----
    if sum(ln.rstrip().endswith(END_PUNCT) for ln in lines) / nl <= 0.12:
        bad.append("fineweb_line_punct")
    dup_chars = sum(len(ln) * c for ln, c in counts.items() if c > 1)
    if dup_chars / max(sum(map(len, lines)), 1) >= 0.1:
        bad.append("fineweb_dup_line_chars")
    if not is_zh and sum(len(ln.strip()) <= 30 for ln in lines) / nl >= 0.67:
        bad.append("fineweb_short_lines")
    return bad


def heuristic_filter(docs: list[dict]) -> tuple[list[dict], Counter, Counter]:
    """Return (kept documents, removed count per "first broken rule", removed count per type)."""
    kept, by_rule, by_kind = [], Counter(), Counter()
    for d in docs:
        reasons = ["lang_other"] if lang_id(d["text"]) == "other" else heuristic_reasons(d["text"])
        if reasons:
            by_rule[reasons[0]] += 1
            by_kind[d["kind"]] += 1
        else:
            kept.append(d)
    return kept, by_rule, by_kind


def main() -> None:
    docs = crawl_mod.build_crawl()["docs"]
    kept, by_rule, by_kind = heuristic_filter(docs)
    before = crawl_mod.kind_table(docs)
    print(f"Heuristic filter: {len(docs)} → {len(kept)} documents\n")
    print(f"{'Type':<14}{'Before':>6}{'Drop':>6}{'Rate':>8}")
    for k in crawl_mod.KINDS:
        print(f"{k:<14}{before[k]:>6}{by_kind[k]:>6}{by_kind[k] / before[k]:>8.0%}")
    print("\nCount by the first broken rule:")
    for rule, c in by_rule.most_common():
        print(f"  {rule:<24}{c:>5}")
    # Show a good document that the rules removed by mistake
    bad_good = [d for d in docs if d["kind"] == "good" and d not in kept]
    lang = Counter(d["lang"] for d in bad_good)
    print(f"\nGood documents removed by mistake: {len(bad_good)} ({lang['en']} English, {lang['zh']} Chinese). Example:")
    ex = next(d for d in bad_good if d["lang"] == "en")
    print("  Rules:", heuristic_reasons(ex["text"]))
    print("  " + ex["text"][:160].replace("\n", "⏎"))

    from zero.data.quality import quality_check  # the production code, on the same documents

    n_prod = sum(lang_id(d["text"]) != "other" and quality_check(d["text"]).keep for d in docs)
    print(f"\nReference: quality_check in zero/data/quality.py keeps {n_prod} documents of the same set")


if __name__ == "__main__":
    main()
