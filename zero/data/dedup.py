"""Deduplication: exact hashes + MinHash LSH near-deduplication (Chapter 13).

Web corpora contain very many duplicates (reposts, template pages, mirror sites). Duplicate data
makes the model memorize text, wastes compute, and makes evaluation contamination worse.
Deduplication has two levels:

1. **Exact deduplication**: hash the full normalized text, and keep only the first document of each
   hash. This is cheap, but it cannot find two documents that differ by one character.
2. **Near-deduplication (MinHash + LSH)**:
   - Split each document into a set of character n-grams (shingles). The similarity of two
     documents is the Jaccard index = |A∩B| / |A∪B|.
   - MinHash: use num_perm random hash functions. Each function takes the minimum hash value over
     the set. The result is a signature vector. The probability that two signatures are equal at
     one position is exactly the Jaccard index (this is the core property of MinHash).
   - LSH: split the signature into `bands` bands of `rows` positions each. If one band is fully
     equal, the two documents become a "candidate pair". A pair with similarity s becomes a
     candidate with the probability 1 - (1 - s^rows)^bands. This is an S-shaped curve. Its
     inflection point is at about (1/bands)^(1/rows), and this is the effective threshold.
   - For each candidate pair, the Jaccard index estimated from the signatures is checked again.
     Pairs above the threshold are joined into clusters with union-find. Each cluster keeps one document.

The code uses character n-grams, not word n-grams. Then Chinese (no spaces between words) and
English use the same code. The implementation is pure NumPy, for teaching and for small to medium
data sizes. For terabytes of data in Step 2, use the same algorithm in tools such as datatrove.
"""

from __future__ import annotations

import hashlib
import multiprocessing
from collections import defaultdict
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from zero.data.clean import normalize_text

_MERSENNE_PRIME = np.uint64((1 << 61) - 1)
_MAX_HASH = np.uint64((1 << 32) - 1)


def text_hash(text: str, normalize: bool = True) -> str:
    if normalize:
        text = normalize_text(text)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def exact_dedup(texts: Sequence[str], normalize: bool = True) -> list[int]:
    """Return the indices to keep (from each group of identical documents, keep only the first one)."""
    seen: set[str] = set()
    keep = []
    for i, t in enumerate(texts):
        h = text_hash(t, normalize)
        if h not in seen:
            seen.add(h)
            keep.append(i)
    return keep


def shingles(text: str, ngram: int = 5) -> set[str]:
    """The set of character n-grams (first normalize, change to lowercase, and collapse white space)."""
    t = " ".join(normalize_text(text).lower().split())
    if len(t) <= ngram:
        return {t} if t else set()
    return {t[i : i + ngram] for i in range(len(t) - ngram + 1)}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def _hash32(s: str) -> int:
    # A stable 32-bit hash. Do not use the built-in hash(): it uses a different salt in each process.
    return int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=4).digest(), "little")


class MinHasher:
    """MinHash signature: h_i(x) = ((a_i * x + b_i) mod p) & 0xFFFFFFFF, minimum over the set.

    a_i < 2^31 and x < 2^32, so the product is < 2^63. It does not overflow in uint64.
    """

    def __init__(self, num_perm: int = 128, ngram: int = 5, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        self.num_perm = num_perm
        self.ngram = ngram
        self.a = rng.integers(1, 1 << 31, size=num_perm, dtype=np.uint64)
        self.b = rng.integers(0, 1 << 31, size=num_perm, dtype=np.uint64)

    def signature(self, text: str) -> np.ndarray:
        sh = shingles(text, self.ngram)
        if not sh:
            return np.full(self.num_perm, _MAX_HASH, dtype=np.uint64)
        x = np.fromiter((_hash32(s) for s in sh), dtype=np.uint64, count=len(sh))
        hv = (x[:, None] * self.a[None, :] + self.b[None, :]) % _MERSENNE_PRIME
        return (hv & _MAX_HASH).min(axis=0)


def estimate_jaccard(sig_a: np.ndarray, sig_b: np.ndarray) -> float:
    return float(np.mean(sig_a == sig_b))


def lsh_threshold(bands: int, rows: int) -> float:
    """Approximate inflection point (1/b)^(1/r) of the S-shaped LSH curve."""
    return (1.0 / bands) ** (1.0 / rows)


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            # The smaller index becomes the root: each cluster keeps the document that occurs first
            if ra < rb:
                self.parent[rb] = ra
            else:
                self.parent[ra] = rb


def _signatures(args: tuple[int, int, int, list[str]]) -> np.ndarray:
    """MinHash signatures of a batch of documents.

    With multiple processes, each child process computes one batch. The same seed gives the same a and b,
    so the result is identical, bit for bit, to one process.
    """
    num_perm, ngram, seed, texts = args
    hasher = MinHasher(num_perm=num_perm, ngram=ngram, seed=seed)
    if not texts:
        return np.zeros((0, num_perm), np.uint64)
    return np.stack([hasher.signature(t) for t in texts])


def near_dedup(
    texts: Sequence[str],
    threshold: float = 0.8,
    num_perm: int = 128,
    bands: int = 16,
    ngram: int = 5,
    seed: int = 0,
    n_jobs: int = 1,
) -> tuple[list[int], list[list[int]]]:
    """MinHash LSH near-deduplication.

    Returns (keep, clusters). keep holds the indices to keep. clusters holds all duplicate clusters
    with size >= 2 (the first index of each cluster is the one that we keep).
    With n_jobs > 1, multiple processes compute the signatures. This is the slowest step: pure Python
    hashes each n-gram, at about 1 MB/s in one process.
    The bucketing and the second check still run in the main process. The result is identical to one process.
    """
    if num_perm % bands != 0:
        raise ValueError(f"num_perm={num_perm} must be divisible by bands={bands}")
    rows = num_perm // bands
    batch = 2000
    if n_jobs > 1 and len(texts) > batch:
        jobs = [(num_perm, ngram, seed, list(texts[i : i + batch])) for i in range(0, len(texts), batch)]
        # spawn, not fork: the pipeline process already has numpy / tokenizers threads, and after a fork the child can deadlock
        with ProcessPoolExecutor(n_jobs, mp_context=multiprocessing.get_context("spawn")) as ex:
            sigs = np.concatenate(list(ex.map(_signatures, jobs)))
    else:
        sigs = _signatures((num_perm, ngram, seed, list(texts)))

    uf = _UnionFind(len(texts))
    for band in range(bands):
        buckets: dict[bytes, list[int]] = defaultdict(list)
        chunk = sigs[:, band * rows : (band + 1) * rows]
        for i in range(len(texts)):
            buckets[chunk[i].tobytes()].append(i)
        for members in buckets.values():
            if len(members) < 2:
                continue
            # Check each pair in the bucket again (buckets are usually small). Skip pairs that are already in the same cluster.
            for jj, j in enumerate(members[1:], start=1):
                for i in members[:jj]:
                    if uf.find(i) != uf.find(j) and estimate_jaccard(sigs[i], sigs[j]) >= threshold:
                        uf.union(i, j)

    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(len(texts)):
        groups[uf.find(i)].append(i)
    keep = sorted(groups)
    clusters = [sorted(g) for g in groups.values() if len(g) > 1]
    return keep, clusters


def dedup(texts: Sequence[str], near: bool = True, **near_kwargs: float) -> list[int]:
    """Exact deduplication first, then (optional) near-deduplication. Return the original indices that we keep."""
    keep = exact_dedup(texts)
    if not near:
        return keep
    sub = [texts[i] for i in keep]
    keep2, _ = near_dedup(sub, **near_kwargs)  # type: ignore[arg-type]
    return [keep[i] for i in keep2]
