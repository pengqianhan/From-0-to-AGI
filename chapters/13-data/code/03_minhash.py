"""去重：精确哈希 + 从零写的 MinHash LSH（核心约 60 行），并验证 S 曲线公式。

1. 精确去重：规范化空白后算哈希，一样的只留第一篇；
2. 近似去重：文档 → 字符 5-gram 集合 → MinHash 签名（num_perm 个"最小哈希"）→ 切成 b 段、每段 r 行，
   任何一段完全相同就成为候选对 → 用签名估计的 Jaccard 复核 → 并查集连成簇，每簇只留一篇。
   相似度为 s 的一对成为候选的概率：P(s) = 1 − (1 − s^r)^b。

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


# ============================ MinHash LSH（从零写） ============================
PRIME = np.uint64((1 << 61) - 1)  # 梅森素数：哈希函数 h(x) = ((a·x + b) mod p) 的低 32 位
MASK = np.uint64((1 << 32) - 1)


def normalize(text: str) -> str:
    return " ".join(text.lower().split())


def shingles(text: str, n: int = 5) -> np.ndarray:
    """字符 n-gram 集合，每个 n-gram 用 crc32 变成一个 32 位整数（中英文同一套代码）。"""
    t = normalize(text)
    grams = {t[i : i + n] for i in range(max(len(t) - n + 1, 1))}
    return np.array(sorted(zlib.crc32(g.encode()) for g in grams), dtype=np.uint64)


class MinHash:
    def __init__(self, num_perm: int = 128, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        self.a = rng.integers(1, 1 << 31, size=num_perm, dtype=np.uint64)
        self.b = rng.integers(0, 1 << 31, size=num_perm, dtype=np.uint64)

    def signature(self, x: np.ndarray) -> np.ndarray:
        # 每个哈希函数作用在集合的每个元素上，取最小值：P(两个集合这一位相等) = Jaccard。
        # 取低 32 位很关键：a·x + b 没超过 p 时"mod p"什么也没做，h 对 x 单调递增，
        # 所有哈希函数会选中同一个最小元素，签名的各位就不再独立（S 曲线会对不上）。
        return (((x[:, None] * self.a + self.b) % PRIME) & MASK).min(axis=0)


def lsh_candidates(sigs: np.ndarray, bands: int) -> set[tuple[int, int]]:
    """签名切成 bands 段，某一段完全相同的两篇文档成为候选对。"""
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
    """返回 (保留的下标, 重复簇列表)。"""
    mh = MinHash(num_perm)
    sigs = np.stack([mh.signature(shingles(t)) for t in texts])
    parent = list(range(len(texts)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j in lsh_candidates(sigs, bands):
        if (sigs[i] == sigs[j]).mean() >= threshold:  # 复核：签名估计的 Jaccard
            ri, rj = find(i), find(j)
            parent[max(ri, rj)] = min(ri, rj)  # 每簇保留最早出现的那篇
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
    """多余副本数：有出处的文档里，同一出处出现了几次就多出几份。"""
    c = Counter(d["origin"] for d in docs if "origin" in d)
    return sum(v - 1 for v in c.values())


def s_curve_check(num_perm: int = 128, bands: int = 16, n_pairs: int = 300, seed: int = 1) -> list:
    """造已知 Jaccard 的集合对，数一数有多少对真的成了候选：和公式比。"""
    rng = np.random.default_rng(seed)
    mh = MinHash(num_perm, seed=7)
    rows = num_perm // bands
    out = []
    for s in [0.3, 0.5, 0.6, 0.7, 0.8, 0.9]:
        # |A| = |B| = 400，共享 k 个元素：J = k / (800 − k) → k = 800·s / (1 + s)
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

    print("LSH 的 S 曲线：P(成为候选) = 1 − (1 − s^r)^b")
    configs = [(16, 8, "本章/zero 默认"), (14, 8, "FineWeb"), (20, 5, "对比"), (450, 20, "RefinedWeb")]
    print(f"{'s':>6}" + "".join(f"{f'b={b},r={r}':>13}" for b, r, _ in configs))
    for s in [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9]:
        print(f"{s:>6}" + "".join(f"{p_candidate(s, b, r):>13.3f}" for b, r, _ in configs))
    print("拐点 (1/b)^(1/r)：" + "，".join(f"{n} {(1 / b) ** (1 / r):.3f}" for b, r, n in configs))

    print("\n实测（128 个哈希、16 段 × 8 行，每个相似度 1000 对）：")
    print(f"{'Jaccard':>8}{'实测候选率':>12}{'公式':>8}")
    for s, emp, th in s_curve_check(n_pairs=1000, seed=3):
        print(f"{s:>8.3f}{emp:>12.3f}{th:>8.3f}")

    texts = [d["text"] for d in docs]
    keep = exact_dedup(texts)
    docs2 = [docs[i] for i in keep]
    keep2, clusters = near_dedup([d["text"] for d in docs2])
    docs3 = [docs2[i] for i in keep2]
    print(f"\n去重：{len(docs)} 篇 →（精确）{len(docs2)} →（MinHash）{len(docs3)}")
    print("  多余副本 = 同一出处的文档数 − 1，加总（一篇文档和它的转载算同一出处）：")
    for name, ds in [("去重前", docs), ("精确去重后", docs2), ("MinHash 后", docs3)]:
        print(f"    {name:<10}{redundant(ds):>5} 篇多余副本")
    wrong = [c for c in clusters if len({docs2[i].get("origin", ("x", i)) for i in c}) > 1]
    print(f"  {len(clusters)} 个重复簇，其中把不同出处误合并的：{len(wrong)} 个")
    # 看一个簇：真实 Jaccard 与签名估计
    c = clusters[0]
    a, b = shingles(docs2[c[0]]["text"]), shingles(docs2[c[1]]["text"])
    true_j = len(np.intersect1d(a, b)) / len(np.union1d(a, b))
    mh = MinHash()
    est = (mh.signature(a) == mh.signature(b)).mean()
    print(f"  例：簇里的两篇 {docs2[c[0]]['kind']} / {docs2[c[1]]['kind']}，"
          f"真实 Jaccard {true_j:.3f}，签名估计 {est:.3f}")


if __name__ == "__main__":
    main()
