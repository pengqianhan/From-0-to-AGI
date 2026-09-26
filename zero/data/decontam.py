"""去污染：训练数据与评测集的 n-gram 重叠检查（对应第 13 章，GOAL.md 3.2 第 6 条）。

如果评测题目（或答案）出现在训练数据里，模型分数就不可信。标准做法（GPT-3、Llama 等都用过）：

1. 把每道评测题切成 n-gram（默认 13 个"词"，GPT-3 论文的取值）；
2. 扫描训练文档，只要某篇文档和某道题共享任何一个 n-gram，就认为"撞了"；
3. 撞了的训练文档要么删掉，要么把重叠片段挖掉；统计结果写进模型卡。

"词"的定义：先转小写、去掉标点，英文按空白切，中文每个汉字算一个词。
中文题目往往很短，13 个汉字的窗口偏严格还是偏宽松需要实测（待核实），所以 n 可调。

n-gram 用 blake2b 取 8 字节哈希存进 set，内存比存字符串小得多，而且跨进程稳定。
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

_TOKEN_RE = re.compile(r"[㐀-䶿一-鿿豈-﫿]|[a-z0-9]+")


def normalize_tokens(text: str) -> list[str]:
    """小写 → 只保留字母数字串和单个汉字（标点、空白全部丢掉）。"""
    return _TOKEN_RE.findall(text.lower())


def _hash_gram(gram: Sequence[str]) -> int:
    return int.from_bytes(
        hashlib.blake2b(" ".join(gram).encode("utf-8"), digest_size=8).digest(), "little"
    )


def ngram_hashes(text: str, n: int = 13) -> set[int]:
    toks = normalize_tokens(text)
    if len(toks) < n:
        # 比 n 还短的题目：整题作为一个 gram（否则永远查不出来）
        return {_hash_gram(toks)} if toks else set()
    return {_hash_gram(toks[i : i + n]) for i in range(len(toks) - n + 1)}


@dataclass
class ContaminationHit:
    doc_index: int
    eval_set: str
    eval_index: int
    overlap: int  # 共享的 n-gram 个数


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
        parts = [f"{s}: {len(self.eval_items_hit(s))} 题" for s in sets]
        return (
            f"{self.n}-gram 去污染：{len(self.contaminated_docs)}/{self.num_docs} 篇训练文档与评测集重叠"
            + (f"（{'，'.join(parts)}）" if parts else "")
        )


class NgramIndex:
    """评测集的 n-gram 索引：hash -> [(评测集名, 题号), ...]。

    长度 >= n 的评测题存 n-gram 哈希；更短的题目（整题不足 n 个词）存规范化后的整句，
    检查时用子串包含判断（否则短题永远查不出来）。
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
        """返回这篇文档撞到的 {(评测集, 题号): 共享 n-gram 数}。"""
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
    """检查每篇训练文档与各评测集的 n-gram 重叠。"""
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
    """删掉所有与评测集有重叠的训练文档，返回 (保留的下标, 报告)。"""
    report = find_contamination(train_texts, eval_sets, n)
    bad = set(report.contaminated_docs)
    return [i for i in range(len(train_texts)) if i not in bad], report
