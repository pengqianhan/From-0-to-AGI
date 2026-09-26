"""质量过滤：启发式规则 + 分类器接口（对应第 13 章）。

两类方法，业界通常叠加使用：

1. **启发式规则**（这里实现）：参考 Gopher（Rae et al. 2021, arXiv:2112.11446, 附录 A）的
   质量与重复规则，以及 FineWeb（Penedo et al. 2024, arXiv:2406.17557）新增的三条规则。
   每条规则都是一个便宜的统计量 + 阈值，专门干掉"明显不是正常文章"的页面：导航栏、
   关键词堆砌、乱码、大段重复。中文没有空格分词，"词"按 CJK 单字计，停用词用中文常用虚词。
   阈值是英文网页上调出来的，用在中文上之前需要在第 13 章的实验里复核（待核实）。

2. **分类器打分**（这里只定义接口）：FineWeb-Edu 的做法是让大模型给几十万个网页的
   "教育价值"打 0–5 分，再训练一个小分类器给全量数据打分，只留高分页面。第一步没有 GPU，
   这里只提供 `QualityClassifier` 协议和一个按关键词打分的玩具实现 `KeywordClassifier`，
   第二步再接入真正的分类器（例如 fastText 或小型 BERT）。
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
)  # Gopher 用的 8 个
ZH_STOP_WORDS = frozenset("的了是在和有也就不都而与之其以")  # 中文常用虚词（本课自定，待核实）
_END_PUNCT = (".", "!", "?", '"', "'", "。", "！", "？", "”", "’", "…", "」", "』")


def words(text: str) -> list[str]:
    """把文本切成"词"：CJK 按单字，其他按空白分隔。"""
    return _WORD_RE.findall(text)


def cjk_ratio(text: str) -> float:
    non_space = sum(1 for c in text if not c.isspace())
    return len(_CJK_RE.findall(text)) / max(non_space, 1)


@dataclass
class QualityThresholds:
    """默认值取自 Gopher 附录 A 与 FineWeb 论文第 3 节。"""

    min_words: int = 50
    max_words: int = 100_000
    min_mean_word_len: float = 3.0  # 只对非 CJK 文本生效
    max_mean_word_len: float = 10.0
    max_symbol_word_ratio: float = 0.1  # "#" 与 "..." 的数量 / 词数
    max_bullet_lines_frac: float = 0.9  # 以项目符号开头的行占比
    max_ellipsis_lines_frac: float = 0.3  # 以省略号结尾的行占比
    min_alpha_words_frac: float = 0.8  # 至少含一个字母的词占比
    min_stop_words: int = 2
    # Gopher 重复规则
    max_dup_line_frac: float = 0.3
    max_dup_line_char_frac: float = 0.2
    max_top_2gram_char_frac: float = 0.2
    # FineWeb 新增规则
    min_line_punct_frac: float = 0.12  # 以标点结尾的行占比
    max_short_line_frac: float = 0.67  # 长度 <= 30 字符的行占比
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
    """对一篇文档跑全部启发式规则，返回是否保留、未通过的规则名和统计量。"""
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

    # 重复
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
        # 中文一行 30 个字符信息量已经很大，这条规则只用于非 CJK 文本
        reasons.append("short_lines")

    return QualityResult(keep=not reasons, reasons=reasons, stats=stats)


def quality_filter(texts: Sequence[str], th: QualityThresholds | None = None) -> list[int]:
    """返回通过全部规则的下标。"""
    return [i for i, t in enumerate(texts) if quality_check(t, th).keep]


# ---------------------------------------------------------------------------
# 分类器接口（第二步接入真实模型）
# ---------------------------------------------------------------------------


class QualityClassifier(Protocol):
    """质量分类器协议：给每篇文档打一个分（越高越好）。

    第二步的计划（参照 FineWeb-Edu）：
    1. 用许可证允许的开放权重大模型给约 50 万篇样本打 0–5 分"教育价值"；
    2. 在其 embedding 上训练一个线性回归头（或 fastText），对全量数据打分；
    3. 保留分数 >= 3 的文档（阈值由第 13 章的小模型消融实验决定）。
    """

    def score(self, texts: Sequence[str]) -> list[float]: ...


class KeywordClassifier:
    """玩具分类器：按"教育类关键词"出现频率打分，只用于测试和演示接口。"""

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
