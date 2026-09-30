"""语言识别 + 启发式过滤：几条便宜的统计规则，干掉"明显不是正常文章"的页面。

规则来自三篇论文（阈值用原文的值）：
- Gopher（Rae et al. 2021，附录 A）：词数、平均词长、符号比例、含字母的词占比、停用词、重复行、重复 2-gram；
- C4（Raffel et al. 2020）：含 "lorem ipsum" 或花括号的页面整篇丢掉；
- FineWeb（Penedo et al. 2024，第 3.6 节）：以标点结尾的行 ≤ 12%、重复行字符 ≥ 10%、短行 ≥ 67%。
中文没有空格分词：一个汉字算一个"词"，停用词换成中文虚词。

    uv run python chapters/13-data/code/02_heuristic_filter.py
"""

from __future__ import annotations

import importlib.util
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))  # 仓库根目录，才能 import zero


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


crawl_mod = _load("01_noisy_crawl")

CJK = re.compile(r"[一-鿿]")
WORD = re.compile(r"[一-鿿]|[^\s一-鿿]+")
EN_STOP = {"the", "be", "to", "of", "and", "that", "have", "with"}  # Gopher 的 8 个停用词
ZH_STOP = set("的了是在和有也就不都而与之其以")
END_PUNCT = tuple(".!?\"'。！？”’…」』")


def lang_id(text: str) -> str:
    """极简语言识别：汉字占比 > 30% 是中文，ASCII 字母占比 > 50% 是英文，否则"其他"。
    （FineWeb 用 fastText 的 lid.176 模型，英文得分 ≥ 0.65 才保留。）"""
    chars = [c for c in text if not c.isspace()]
    n = max(len(chars), 1)
    if sum(bool(CJK.match(c)) for c in chars) / n > 0.3:
        return "zh"
    if sum(c.isascii() and c.isalpha() for c in chars) / n > 0.5:
        return "en"
    return "other"


def heuristic_reasons(text: str) -> list[str]:
    """返回这篇文档违反的规则（空列表 = 通过）。"""
    words = WORD.findall(text)
    n = len(words)
    lines = [ln for ln in text.split("\n") if ln.strip()]
    nl = max(len(lines), 1)
    is_zh = lang_id(text) == "zh"
    bad = []
    # ---- Gopher 质量规则 ----
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
    # ---- Gopher 重复规则 ----
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
    """返回 (保留的文档, 按"第一条违反的规则"计的删除数, 按类型计的删除数)。"""
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
    print(f"启发式过滤：{len(docs)} → {len(kept)} 篇\n")
    print(f"{'类型':<14}{'过滤前':>6}{'删掉':>6}{'删除率':>8}")
    for k in crawl_mod.KINDS:
        print(f"{k:<14}{before[k]:>6}{by_kind[k]:>6}{by_kind[k] / before[k]:>8.0%}")
    print("\n按第一条违反的规则计：")
    for rule, c in by_rule.most_common():
        print(f"  {rule:<24}{c:>5}")
    # 误杀的好文档长什么样
    bad_good = [d for d in docs if d["kind"] == "good" and d not in kept]
    lang = Counter(d["lang"] for d in bad_good)
    print(f"\n被误杀的好文档 {len(bad_good)} 篇（英文 {lang['en']}，中文 {lang['zh']}），例如：")
    ex = next(d for d in bad_good if d["lang"] == "en")
    print("  规则：", heuristic_reasons(ex["text"]))
    print("  " + ex["text"][:160].replace("\n", "⏎"))

    from zero.data.quality import quality_check  # 生产级实现，同一批文档上对照

    n_prod = sum(lang_id(d["text"]) != "other" and quality_check(d["text"]).keep for d in docs)
    print(f"\n对照：zero/data/quality.py 的 quality_check 在同一批文档上保留 {n_prod} 篇")


if __name__ == "__main__":
    main()
