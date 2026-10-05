"""去重：精确哈希 + MinHash LSH 近似去重（对应第 13 章）。

网页语料里重复极多（转载、模板页、镜像站）。重复数据会让模型"背书"、浪费算力，还会放大
评测污染。两层去重：

1. **精确去重**：规范化后的全文做哈希，相同哈希只留第一篇。便宜，但差一个字就认不出来。
2. **近似去重（MinHash + LSH）**：
   - 把文档切成字符 n-gram 集合（shingles），两篇文档的相似度用 Jaccard = |A∩B| / |A∪B| 衡量；
   - MinHash：用 num_perm 个随机哈希函数，每个函数取集合里的最小哈希值，得到一个签名向量。
     两个签名某一位相等的概率恰好等于 Jaccard（这是 MinHash 的核心性质）；
   - LSH：把签名切成 bands 段、每段 rows 位。只要有一段完全相同就成为"候选对"。
     相似度为 s 的一对成为候选的概率是 1 - (1 - s^rows)^bands，是一条 S 形曲线，
     拐点大约在 (1/bands)^(1/rows)，这就是有效阈值；
   - 候选对再用签名估计的 Jaccard 复核，超过阈值的用并查集连成簇，每簇只留一篇。

用字符 n-gram 而不是词 n-gram，是为了让中文（没有空格分词）和英文用同一套代码。
纯 NumPy 实现，适合讲解和中小规模；第二步处理 TB 级数据时用 datatrove 等工具的同款算法。
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
    """返回要保留的下标（每组完全相同的文档只留第一篇）。"""
    seen: set[str] = set()
    keep = []
    for i, t in enumerate(texts):
        h = text_hash(t, normalize)
        if h not in seen:
            seen.add(h)
            keep.append(i)
    return keep


def shingles(text: str, ngram: int = 5) -> set[str]:
    """字符 n-gram 集合（先规范化、转小写、压缩空白）。"""
    t = " ".join(normalize_text(text).lower().split())
    if len(t) <= ngram:
        return {t} if t else set()
    return {t[i : i + ngram] for i in range(len(t) - ngram + 1)}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def _hash32(s: str) -> int:
    # 稳定的 32 位哈希（Python 自带的 hash() 每个进程加盐不同，不能用）
    return int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=4).digest(), "little")


class MinHasher:
    """MinHash 签名：h_i(x) = ((a_i * x + b_i) mod p) & 0xFFFFFFFF，取集合上的最小值。

    a_i < 2^31、x < 2^32，乘积 < 2^63，在 uint64 里不会溢出。
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
    """LSH S 形曲线的近似拐点 (1/b)^(1/r)。"""
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
            # 让下标小的当根：每簇保留最早出现的那篇
            if ra < rb:
                self.parent[rb] = ra
            else:
                self.parent[ra] = rb


def _signatures(args: tuple[int, int, int, list[str]]) -> np.ndarray:
    """一批文档的 MinHash 签名（多进程时每个子进程算一批；同样的种子 → 同样的 a、b → 与单进程逐位相同）。"""
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
    """MinHash LSH 近似去重。

    返回 (keep, clusters)：keep 是保留的下标；clusters 是所有大小 >= 2 的重复簇（每簇第一个是被保留的）。
    n_jobs > 1 时用多进程计算签名（最耗时的一步，纯 Python 逐个 n-gram 哈希，单进程约 1 MB/s）；
    分桶和复核仍在主进程里做，结果与单进程完全相同。
    """
    if num_perm % bands != 0:
        raise ValueError(f"num_perm={num_perm} 必须能被 bands={bands} 整除")
    rows = num_perm // bands
    batch = 2000
    if n_jobs > 1 and len(texts) > batch:
        jobs = [(num_perm, ngram, seed, list(texts[i : i + batch])) for i in range(0, len(texts), batch)]
        # spawn 而不是 fork：流水线进程里已有 numpy / tokenizers 的线程，fork 之后子进程可能死锁
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
            # 桶内两两复核（桶一般很小）；已经在同一簇的跳过
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
    """先精确去重，再（可选）近似去重；返回保留的原始下标。"""
    keep = exact_dedup(texts)
    if not near:
        return keep
    sub = [texts[i] for i in keep]
    keep2, _ = near_dedup(sub, **near_kwargs)  # type: ignore[arg-type]
    return [keep[i] for i in keep2]
