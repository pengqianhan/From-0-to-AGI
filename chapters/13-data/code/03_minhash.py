"""Deduplication: exact hashes + MinHash LSH written from scratch (the core is about 60 lines).
The script also checks the formula of the S-curve.

1. Exact dedup: normalize the whitespace and calculate a hash. Keep only the first document of
   each hash.
2. Near dedup: document → set of character 5-grams → MinHash signature (num_perm "minimum
   hashes") → split into b bands of r rows. If one band is identical, the pair becomes a candidate
   pair → check the candidate with the Jaccard estimate from the signatures → join the documents
   into clusters with union-find, and keep one document per cluster.
   The probability that a pair with similarity s becomes a candidate: P(s) = 1 − (1 − s^r)^b.

    uv run python chapters/13-data/code/03_minhash.py
"""

from __future__ import annotations

import hashlib
import importlib.util
import zlib
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ======================= MinHash LSH (written from scratch) =======================
PRIME = np.uint64((1 << 61) - 1)  # Mersenne prime; hash function h(x) = low 32 bits of ((a·x + b) mod p)
MASK = np.uint64((1 << 32) - 1)


def normalize(text: str) -> str:
    return " ".join(text.lower().split())


def shingles(text: str, n: int = 5) -> np.ndarray:
    """Set of character n-grams. crc32 changes each n-gram into a 32-bit integer.

    The same code works for Chinese and English.
    """
    t = normalize(text)
    grams = {t[i : i + n] for i in range(max(len(t) - n + 1, 1))}
    return np.array(sorted(zlib.crc32(g.encode()) for g in grams), dtype=np.uint64)


class MinHash:
    def __init__(self, num_perm: int = 128, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        self.a = rng.integers(1, 1 << 31, size=num_perm, dtype=np.uint64)
        self.b = rng.integers(0, 1 << 31, size=num_perm, dtype=np.uint64)

    def signature(self, x: np.ndarray) -> np.ndarray:
        # Apply each hash function to each element of the set and take the minimum:
        # P(this position is equal for two sets) = Jaccard.
        # The low 32 bits are necessary. When a·x + b < p, "mod p" does nothing, so h increases
        # monotonically with x. Then all hash functions select the same minimum element, and the
        # positions of the signature are not independent (the S-curve does not match).
        return (((x[:, None] * self.a + self.b) % PRIME) & MASK).min(axis=0)


def lsh_candidates(sigs: np.ndarray, bands: int) -> set[tuple[int, int]]:
    """Split the signatures into `bands` bands.

    Two documents with one identical band become a candidate pair.
    """
    rows = sigs.shape[1] // bands
    pairs = set()
    for b in range(bands):
        buckets = defaultdict(list)
        for i, s in enumerate(sigs[:, b * rows : (b + 1) * rows]):
            buckets[s.tobytes()].append(i)
        for members in buckets.values():
            pairs.update((i, j) for k, i in enumerate(members) for j in members[k + 1 :])
    return pairs


def near_dedup(texts: list[str], num_perm: int = 128, bands: int = 16, threshold: float = 0.8):
    """Return (kept indices, list of duplicate clusters)."""
    mh = MinHash(num_perm)
    sigs = np.stack([mh.signature(shingles(t)) for t in texts])
    parent = list(range(len(texts)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j in lsh_candidates(sigs, bands):
        if (sigs[i] == sigs[j]).mean() >= threshold:  # check: the Jaccard estimate from the signatures
            ri, rj = find(i), find(j)
            parent[max(ri, rj)] = min(ri, rj)  # keep the first document of each cluster
    groups = defaultdict(list)
    for i in range(len(texts)):
        groups[find(i)].append(i)
    return sorted(groups), [g for g in groups.values() if len(g) > 1]


def p_candidate(s: float, bands: int, rows: int) -> float:
    return 1 - (1 - s**rows) ** bands


# ================================================================================


def exact_dedup(texts: list[str]) -> list[int]:
    seen, keep = set(), []
    for i, t in enumerate(texts):
        h = hashlib.sha1(normalize(t).encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            keep.append(i)
    return keep


def redundant(docs: list[dict]) -> int:
    """Number of redundant copies: for documents with an origin, each origin counts its
    documents minus 1."""
    c = Counter(d["origin"] for d in docs if "origin" in d)
    return sum(v - 1 for v in c.values())


def s_curve_check(num_perm: int = 128, bands: int = 16, n_pairs: int = 300, seed: int = 1) -> list:
    """Make set pairs with a known Jaccard. Count how many pairs become candidates, and compare
    with the formula."""
    rng = np.random.default_rng(seed)
    mh = MinHash(num_perm, seed=7)
    rows = num_perm // bands
    out = []
    for s in [0.3, 0.5, 0.6, 0.7, 0.8, 0.9]:
        # |A| = |B| = 400 with k shared elements: J = k / (800 − k) → k = 800·s / (1 + s)
        k = round(800 * s / (1 + s))
        hit = 0
        for _ in range(n_pairs):
            u = rng.choice(2**31, size=800 - k, replace=False).astype(np.uint64)
            A, B = u[:400], np.concatenate([u[:k], u[400:]])
            sa, sb = mh.signature(A), mh.signature(B)
            hit += any((sa[b * rows : (b + 1) * rows] == sb[b * rows : (b + 1) * rows]).all()
                       for b in range(bands))
        out.append((k / (800 - k), hit / n_pairs, p_candidate(k / (800 - k), bands, rows)))
    return out


def main() -> None:
    crawl_mod = _load("01_noisy_crawl")
    heur = _load("02_heuristic_filter")
    docs = crawl_mod.build_crawl()["docs"]
    docs, _, _ = heur.heuristic_filter(docs)

    print("S-curve of LSH: P(candidate) = 1 − (1 − s^r)^b")
    configs = [(16, 8, "ch13/zero default"), (14, 8, "FineWeb"), (20, 5, "comparison"), (450, 20, "RefinedWeb")]
    print(f"{'s':>6}" + "".join(f"{f'b={b},r={r}':>13}" for b, r, _ in configs))
    for s in [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9]:
        print(f"{s:>6}" + "".join(f"{p_candidate(s, b, r):>13.3f}" for b, r, _ in configs))
    print("Inflection point (1/b)^(1/r): " + ", ".join(f"{n} {(1 / b) ** (1 / r):.3f}" for b, r, n in configs))

    print("\nMeasured (128 hashes, 16 bands × 8 rows, 1000 pairs for each similarity):")
    print(f"{'Jaccard':>8}{'Measured':>12}{'Formula':>8}")
    for s, emp, th in s_curve_check(n_pairs=1000, seed=3):
        print(f"{s:>8.3f}{emp:>12.3f}{th:>8.3f}")

    texts = [d["text"] for d in docs]
    keep = exact_dedup(texts)
    docs2 = [docs[i] for i in keep]
    keep2, clusters = near_dedup([d["text"] for d in docs2])
    docs3 = [docs2[i] for i in keep2]
    print(f"\nDedup: {len(docs)} documents → (exact) {len(docs2)} → (MinHash) {len(docs3)}")
    print("  Redundant copies = Σ over origins (documents with this origin − 1). A document and its"
          " reposts have the same origin. Before dedup, after exact dedup, after MinHash:")
    for name, ds in [("Before", docs), ("Exact", docs2), ("MinHash", docs3)]:
        print(f"    {name:<10}{redundant(ds):>5} redundant copies")
    wrong = [c for c in clusters if len({docs2[i].get("origin", ("x", i)) for i in c}) > 1]
    print(f"  {len(clusters)} duplicate clusters; clusters that join different origins by mistake: {len(wrong)}")
    # Look at one cluster: the true Jaccard and the estimate from the signatures
    c = clusters[0]
    a, b = shingles(docs2[c[0]]["text"]), shingles(docs2[c[1]]["text"])
    true_j = len(np.intersect1d(a, b)) / len(np.union1d(a, b))
    mh = MinHash()
    est = (mh.signature(a) == mh.signature(b)).mean()
    print(f"  Example: two documents in a cluster, {docs2[c[0]]['kind']} / {docs2[c[1]]['kind']}, "
          f"true Jaccard {true_j:.3f}, estimate from the signatures {est:.3f}")


if __name__ == "__main__":
    main()
