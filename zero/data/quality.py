"""Quality filtering: heuristic rules + a classifier interface (Chapter 13).

There are two types of methods. Industry usually uses both together:

1. **Heuristic rules** (implemented here): based on the quality and repetition rules of Gopher
   (Rae et al. 2021, arXiv:2112.11446, Appendix A), and the three new rules of FineWeb
   (Penedo et al. 2024, arXiv:2406.17557). Each rule is a cheap statistic + a threshold. The rules
   remove pages that are "clearly not normal text": navigation bars, keyword lists, garbled text,
   long repeated parts. Chinese has no spaces between words, so each CJK character counts as one
   "word", and the stop words are common Chinese function words.
   The thresholds were tuned on English web pages. Before we use them on Chinese text, an
   experiment in Chapter 13 must check them again (to be verified).

2. **Classifier scores** (only the interface is defined here): FineWeb-Edu asks a large model to
   give hundreds of thousands of web pages an "educational value" score from 0 to 5. Then it trains
   a small classifier that scores all data, and keeps only the pages with high scores. Step 1 has
   no GPU. So this module gives only the `QualityClassifier` protocol and a toy implementation that
   scores by keywords, `KeywordClassifier`. Step 2 will connect a real classifier (for example
   fastText or a small BERT).
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

_CJK_RE = re.compile(r"[㐀-䶿一-鿿豈-﫿]")
_WORD_RE = re.compile(r"[㐀-䶿一-鿿豈-﫿]|[^\s㐀-䶿一-鿿豈-﫿]+")
_ALPHA_RE = re.compile(r"[^\W\d_]", re.UNICODE)

EN_STOP_WORDS = frozenset(
    {"the", "be", "to", "of", "and", "that", "have", "with"}
)  # the 8 words that Gopher uses
ZH_STOP_WORDS = frozenset("的了是在和有也就不都而与之其以")  # common Chinese function words (chosen for this course, to be verified)
_END_PUNCT = (".", "!", "?", '"', "'", "。", "！", "？", "”", "’", "…", "」", "』")


def words(text: str) -> list[str]:
    """Split the text into "words": each CJK character is one word; other text is split at white space."""
    return _WORD_RE.findall(text)


def cjk_ratio(text: str) -> float:
    non_space = sum(1 for c in text if not c.isspace())
    return len(_CJK_RE.findall(text)) / max(non_space, 1)


@dataclass
class QualityThresholds:
    """The default values come from Gopher Appendix A and Section 3 of the FineWeb paper."""

    min_words: int = 50
    max_words: int = 100_000
    min_mean_word_len: float = 3.0  # applies only to non-CJK text
    max_mean_word_len: float = 10.0
    max_symbol_word_ratio: float = 0.1  # number of "#" and "..." / number of words
    max_bullet_lines_frac: float = 0.9  # fraction of lines that start with a bullet
    max_ellipsis_lines_frac: float = 0.3  # fraction of lines that end with an ellipsis
    min_alpha_words_frac: float = 0.8  # fraction of words with at least one letter
    min_stop_words: int = 2
    # Gopher repetition rules
    max_dup_line_frac: float = 0.3
    max_dup_line_char_frac: float = 0.2
    max_top_2gram_char_frac: float = 0.2
    # New rules of FineWeb
    min_line_punct_frac: float = 0.12  # fraction of lines that end with punctuation
    max_short_line_frac: float = 0.67  # fraction of lines with <= 30 characters
    short_line_chars: int = 30


@dataclass
class QualityResult:
    keep: bool
    reasons: list[str] = field(default_factory=list)
    stats: dict[str, float] = field(default_factory=dict)


def _top_ngram_char_frac(ws: list[str], n: int) -> float:
    if len(ws) < n:
        return 0.0
    grams = Counter(tuple(ws[i : i + n]) for i in range(len(ws) - n + 1))
    ((gram, count),) = grams.most_common(1)
    total_chars = sum(len(w) for w in ws)
    return count * sum(len(w) for w in gram) / max(total_chars, 1)


def quality_check(text: str, th: QualityThresholds | None = None) -> QualityResult:
    """Run all heuristic rules on one document. Return if we keep it, the names of the failed rules, and the statistics."""
    th = th or QualityThresholds()
    reasons: list[str] = []
    ws = words(text)
    n = len(ws)
    lines = [ln for ln in text.split("\n") if ln.strip()]
    n_lines = max(len(lines), 1)
    is_cjk = cjk_ratio(text) > 0.3

    stats: dict[str, float] = {"words": n, "lines": len(lines), "cjk_ratio": cjk_ratio(text)}

    if n < th.min_words:
        reasons.append("too_few_words")
    if n > th.max_words:
        reasons.append("too_many_words")
    if not is_cjk and n:
        mean_len = sum(len(w) for w in ws) / n
        stats["mean_word_len"] = mean_len
        if not (th.min_mean_word_len <= mean_len <= th.max_mean_word_len):
            reasons.append("mean_word_length")
    symbols = text.count("#") + text.count("...") + text.count("…")
    stats["symbol_word_ratio"] = symbols / max(n, 1)
    if stats["symbol_word_ratio"] > th.max_symbol_word_ratio:
        reasons.append("symbol_word_ratio")
    bullets = sum(1 for ln in lines if ln.lstrip().startswith(("•", "-", "*", "·", "●", "▪")))
    if bullets / n_lines > th.max_bullet_lines_frac:
        reasons.append("bullet_lines")
    ellipsis = sum(1 for ln in lines if ln.rstrip().endswith(("...", "…")))
    if ellipsis / n_lines > th.max_ellipsis_lines_frac:
        reasons.append("ellipsis_lines")
    alpha = sum(1 for w in ws if _ALPHA_RE.search(w))
    stats["alpha_words_frac"] = alpha / max(n, 1)
    if n and stats["alpha_words_frac"] < th.min_alpha_words_frac:
        reasons.append("alpha_words")
    stop = ZH_STOP_WORDS if is_cjk else EN_STOP_WORDS
    n_stop = sum(1 for w in ws if w.lower() in stop)
    if n_stop < th.min_stop_words:
        reasons.append("stop_words")

    # Repetition
    line_counts = Counter(lines)
    dup_lines = sum(c for c in line_counts.values() if c > 1)
    dup_line_chars = sum(len(ln) * c for ln, c in line_counts.items() if c > 1)
    stats["dup_line_frac"] = dup_lines / n_lines
    if stats["dup_line_frac"] > th.max_dup_line_frac:
        reasons.append("dup_lines")
    if dup_line_chars / max(sum(len(ln) for ln in lines), 1) > th.max_dup_line_char_frac:
        reasons.append("dup_line_chars")
    stats["top_2gram_char_frac"] = _top_ngram_char_frac(ws, 2)
    if stats["top_2gram_char_frac"] > th.max_top_2gram_char_frac:
        reasons.append("top_2gram")

    # FineWeb
    punct_lines = sum(1 for ln in lines if ln.rstrip().endswith(_END_PUNCT))
    stats["line_punct_frac"] = punct_lines / n_lines
    if stats["line_punct_frac"] < th.min_line_punct_frac:
        reasons.append("line_punct")
    short = sum(1 for ln in lines if len(ln.strip()) <= th.short_line_chars)
    stats["short_line_frac"] = short / n_lines
    if not is_cjk and stats["short_line_frac"] > th.max_short_line_frac:
        # A Chinese line of 30 characters already holds much information, so this rule applies only to non-CJK text
        reasons.append("short_lines")

    return QualityResult(keep=not reasons, reasons=reasons, stats=stats)


def quality_filter(texts: Sequence[str], th: QualityThresholds | None = None) -> list[int]:
    """Return the indices of the documents that pass all rules."""
    return [i for i, t in enumerate(texts) if quality_check(t, th).keep]


# ---------------------------------------------------------------------------
# Classifier interface (Step 2 connects a real model)
# ---------------------------------------------------------------------------


class QualityClassifier(Protocol):
    """Quality classifier protocol: give each document a score (higher is better).

    The plan for Step 2 (as in FineWeb-Edu):
    1. Use a large open-weight model whose license allows it to give about 500,000 samples an
       "educational value" score from 0 to 5.
    2. Train a linear regression head (or fastText) on its embeddings, and score all data.
    3. Keep the documents with a score >= 3 (an ablation with small models in Chapter 13 decides the threshold).
    """

    def score(self, texts: Sequence[str]) -> list[float]: ...


class KeywordClassifier:
    """Toy classifier: the score is the frequency of "educational keywords". Only for tests and to show the interface."""

    def __init__(
        self, keywords: Sequence[str] = ("定理", "证明", "例如", "because", "therefore", "example")
    ):
        self.keywords = [k.lower() for k in keywords]

    def score(self, texts: Sequence[str]) -> list[float]:
        out = []
        for t in texts:
            low = t.lower()
            hits = sum(low.count(k) for k in self.keywords)
            out.append(min(5.0, 5.0 * hits / max(len(words(t)) / 50, 1)))
        return out


def filter_by_classifier(
    texts: Sequence[str], classifier: QualityClassifier, threshold: float = 3.0
) -> list[int]:
    scores = classifier.score(texts)
    return [i for i, s in enumerate(scores) if s >= threshold]
