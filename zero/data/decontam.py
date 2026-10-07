"""Decontamination: check the n-gram overlap between training data and evaluation sets (Chapter 13, GOAL.md 3.2 item 6).

If evaluation items (or their answers) occur in the training data, the model scores are not reliable.
The standard method (GPT-3, Llama, and others used it):

1. Split each evaluation item into n-grams (default: 13 "words", the value of the GPT-3 paper).
2. Scan the training documents. If a document shares one or more n-grams with an item, it is a "hit".
3. Remove each training document with a hit, or cut out the overlapping part.
   Write the statistics into the model card.

Definition of a "word": first change the text to lowercase and remove the punctuation. Split English
text at white space. Each Chinese character is one word.
Chinese items are often short. We must measure if a window of 13 Chinese characters is too strict
or too loose (to be verified). For this reason, n is a parameter.

The code keeps an 8-byte blake2b hash of each n-gram in a set. This uses much less memory than
strings, and the hash is the same in all processes.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

_TOKEN_RE = re.compile(r"[㐀-䶿一-鿿豈-﫿]|[a-z0-9]+")


def normalize_tokens(text: str) -> list[str]:
    """Lowercase → keep only alphanumeric strings and single Chinese characters (drop all punctuation and white space)."""
    return _TOKEN_RE.findall(text.lower())


def _hash_gram(gram: Sequence[str]) -> int:
    return int.from_bytes(
        hashlib.blake2b(" ".join(gram).encode("utf-8"), digest_size=8).digest(), "little"
    )


def ngram_hashes(text: str, n: int = 13) -> set[int]:
    toks = normalize_tokens(text)
    if len(toks) < n:
        # An item shorter than n: use the full item as one gram (if not, we can never find it)
        return {_hash_gram(toks)} if toks else set()
    return {_hash_gram(toks[i : i + n]) for i in range(len(toks) - n + 1)}


@dataclass
class ContaminationHit:
    doc_index: int
    eval_set: str
    eval_index: int
    overlap: int  # number of shared n-grams


@dataclass
class ContaminationReport:
    n: int
    num_docs: int
    hits: list[ContaminationHit] = field(default_factory=list)

    @property
    def contaminated_docs(self) -> list[int]:
        return sorted({h.doc_index for h in self.hits})

    def eval_items_hit(self, eval_set: str) -> list[int]:
        return sorted({h.eval_index for h in self.hits if h.eval_set == eval_set})

    def summary(self) -> str:
        sets = sorted({h.eval_set for h in self.hits})
        parts = [f"{s} items: {len(self.eval_items_hit(s))}" for s in sets]
        return (
            f"{self.n}-gram decontamination: {len(self.contaminated_docs)}/{self.num_docs} training documents overlap with the evaluation sets"
            + (f" ({', '.join(parts)})" if parts else "")
        )


class NgramIndex:
    """N-gram index of the evaluation sets: hash -> [(evaluation set name, item index), ...].

    For an item with >= n words, the index keeps the n-gram hashes. For a shorter item (fewer than
    n words in total), it keeps the full normalized text, and the check looks for it as a substring.
    If not, we can never find a short item.
    """

    def __init__(self, n: int = 13) -> None:
        self.n = n
        self.index: dict[int, list[tuple[str, int]]] = {}
        self.short: list[tuple[str, int, str]] = []

    def add_eval_set(self, name: str, texts: Iterable[str]) -> None:
        for i, t in enumerate(texts):
            toks = normalize_tokens(t)
            if len(toks) >= self.n:
                for h in ngram_hashes(t, self.n):
                    self.index.setdefault(h, []).append((name, i))
            elif toks:
                self.short.append((name, i, " ".join(toks)))

    def check(self, text: str) -> dict[tuple[str, int], int]:
        """Return the hits of this document: {(evaluation set, item index): number of shared n-grams}."""
        toks = normalize_tokens(text)
        hits: dict[tuple[str, int], int] = {}
        if len(toks) >= self.n:
            grams = {_hash_gram(toks[i : i + self.n]) for i in range(len(toks) - self.n + 1)}
            for g in grams:
                for key in self.index.get(g, ()):
                    hits[key] = hits.get(key, 0) + 1
        if self.short:
            joined = " " + " ".join(toks) + " "
            for name, i, s in self.short:
                if f" {s} " in joined:
                    hits[(name, i)] = hits.get((name, i), 0) + 1
        return hits


def find_contamination(
    train_texts: Sequence[str], eval_sets: dict[str, Sequence[str]], n: int = 13
) -> ContaminationReport:
    """Check the n-gram overlap of each training document with each evaluation set."""
    index = NgramIndex(n)
    for name, texts in eval_sets.items():
        index.add_eval_set(name, texts)
    report = ContaminationReport(n=n, num_docs=len(train_texts))
    for d, text in enumerate(train_texts):
        for (name, i), overlap in index.check(text).items():
            report.hits.append(ContaminationHit(d, name, i, overlap))
    return report


def decontaminate(
    train_texts: Sequence[str], eval_sets: dict[str, Sequence[str]], n: int = 13
) -> tuple[list[int], ContaminationReport]:
    """Remove all training documents that overlap with an evaluation set. Return (indices that we keep, report)."""
    report = find_contamination(train_texts, eval_sets, n)
    bad = set(report.contaminated_docs)
    return [i for i in range(len(train_texts)) if i not in bad], report
