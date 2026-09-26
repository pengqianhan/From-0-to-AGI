"""选择题的两种判分：对数似然（log-likelihood）挑选项 vs 生成后精确匹配（exact match）。

    uv run python chapters/11-evaluation/code/02_loglik_vs_generate.py

为了几秒内跑完、结果完全可复现，这里的"语言模型"不是神经网络，而是一个**最长后缀匹配的字符级
n-gram 模型**（思路同 Infini-gram）：预测下一个字时，在"训练语料"里找和当前上下文末尾最长的
那段匹配，看它后面接的是什么字，再和更短的匹配插值平滑。它没有任何推理能力，只会"背"语料——
正好用来说明：评测分数有多依赖语料里见过什么（污染）、提示词长什么样（格式敏感性）。

本文件定义三样东西，后面的 03、05 用 importlib 复用：
1. `build_world()`：一个玩具"世界"——事实表、训练语料、32 道四选一考题（其中 12 道原题被泄漏进语料）；
2. `SuffixLM`：上面说的字符级最长后缀模型，`logprob(context, continuation)` 返回 log P(续写 | 上下文)；
3. 两个判分器：`score_mc_loglik`（acc 与 acc_norm）和 `score_generate_em`（贪心生成 + 精确匹配）。
"""

from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# 1. 玩具世界：事实 → 语料 + 考题
# ---------------------------------------------------------------------------

# (类别, 完形填空的题干, 答案, 问答式问题)
FACTS = [
    ("capital", "日本的首都是", "东京", "日本的首都是哪座城市？"),
    ("capital", "法国的首都是", "巴黎", "法国的首都是哪座城市？"),
    ("capital", "英国的首都是", "伦敦", "英国的首都是哪座城市？"),
    ("capital", "德国的首都是", "柏林", "德国的首都是哪座城市？"),
    ("capital", "意大利的首都是", "罗马", "意大利的首都是哪座城市？"),
    ("capital", "俄罗斯的首都是", "莫斯科", "俄罗斯的首都是哪座城市？"),
    ("capital", "加拿大的首都是", "渥太华", "加拿大的首都是哪座城市？"),
    ("capital", "澳大利亚的首都是", "堪培拉", "澳大利亚的首都是哪座城市？"),
    ("capital", "韩国的首都是", "首尔", "韩国的首都是哪座城市？"),
    ("capital", "埃及的首都是", "开罗", "埃及的首都是哪座城市？"),
    ("capital", "印度的首都是", "新德里", "印度的首都是哪座城市？"),
    ("capital", "西班牙的首都是", "马德里", "西班牙的首都是哪座城市？"),
    ("capital", "泰国的首都是", "曼谷", "泰国的首都是哪座城市？"),
    ("capital", "巴西的首都是", "巴西利亚", "巴西的首都是哪座城市？"),
    ("capital", "阿根廷的首都是", "布宜诺斯艾利斯", "阿根廷的首都是哪座城市？"),
    ("capital", "美国的首都是", "华盛顿", "美国的首都是哪座城市？"),
    ("author", "《红楼梦》的作者是", "曹雪芹", "《红楼梦》是谁写的？"),
    ("author", "《西游记》的作者是", "吴承恩", "《西游记》是谁写的？"),
    ("author", "《水浒传》的作者是", "施耐庵", "《水浒传》是谁写的？"),
    ("author", "《三国演义》的作者是", "罗贯中", "《三国演义》是谁写的？"),
    ("author", "《呐喊》的作者是", "鲁迅", "《呐喊》是谁写的？"),
    ("author", "《背影》的作者是", "朱自清", "《背影》是谁写的？"),
    ("author", "《边城》的作者是", "沈从文", "《边城》是谁写的？"),
    ("author", "《骆驼祥子》的作者是", "老舍", "《骆驼祥子》是谁写的？"),
    ("planet", "离太阳最近的行星是", "水星", "离太阳最近的行星是哪一颗？"),
    ("planet", "太阳系里最大的行星是", "木星", "太阳系里最大的行星是哪一颗？"),
    ("planet", "以光环闻名的行星是", "土星", "以光环闻名的行星是哪一颗？"),
    ("planet", "被称为红色星球的行星是", "火星", "被称为红色星球的行星是哪一颗？"),
    ("element", "水的化学式是", "H2O", "水的化学式是什么？"),
    ("element", "食盐的主要成分是", "氯化钠", "食盐的主要成分是什么？"),
    ("element", "空气中含量最多的气体是", "氮气", "空气中含量最多的气体是什么？"),
    ("element", "植物光合作用放出的气体是", "氧气", "植物光合作用放出的气体是什么？"),
]
# 这些事实只出现在"练习题"里，不进考卷（教会模型"问：…答：…"这种格式）
EXERCISE_FACTS = [
    ("capital", "中国的首都是", "北京", "中国的首都是哪座城市？"),
    ("capital", "越南的首都是", "河内", "越南的首都是哪座城市？"),
    ("author", "《家》的作者是", "巴金", "《家》是谁写的？"),
    ("author", "《围城》的作者是", "钱锺书", "《围城》是谁写的？"),
    ("planet", "离太阳最远的行星是", "海王星", "离太阳最远的行星是哪一颗？"),
    ("element", "金刚石由哪种元素组成？答案是", "碳", "金刚石由哪种元素组成？"),
]
N_LEAKED = 12  # 32 道考题里有 12 道"原题 + 答案"被泄漏进训练语料
QA_PREFIX, QA_MID = "问：", "\n答："  # 语料里练习题的格式


@dataclass
class Item:
    id: str
    category: str
    stem: str  # 完形填空题干："日本的首都是"
    question: str  # 问答式："日本的首都是哪座城市？"
    choices: list[str]
    answer: int  # 正确选项下标
    leaked: bool = False


@dataclass
class World:
    corpus: str
    items: list[Item]
    docs: list[str] = field(default_factory=list)  # 语料按"文档"切开（05 去污染用）


def build_world(seed: int = 0) -> World:
    rng = random.Random(seed)
    by_cat: dict[str, list[str]] = defaultdict(list)
    for cat, _, ans, _ in FACTS + EXERCISE_FACTS:
        by_cat[cat].append(ans)

    items = []
    leaked_ids = set(rng.sample(range(len(FACTS)), N_LEAKED))
    for i, (cat, stem, ans, q) in enumerate(FACTS):
        wrong = rng.sample([a for a in by_cat[cat] if a != ans], 3)  # 同类别的干扰项
        choices = wrong + [ans]
        rng.shuffle(choices)
        items.append(
            Item(f"q{i:02d}", cat, stem, q, choices, choices.index(ans), i in leaked_ids)
        )

    docs = []
    # (a) "课本"：考题对应的事实，一半按题干的语序写（"日本的首都是东京。"），
    #     三成倒过来写（"东京，就是日本的首都。"），两成课本里根本没有。练习题的事实都按题干语序写。
    kinds = ["direct"] * 16 + ["inverted"] * 10 + ["absent"] * 6
    rng.shuffle(kinds)
    for (_cat, stem, ans, _), kind in zip(FACTS, kinds):
        if kind == "direct":
            docs += [f"{stem}{ans}。", f"我们都知道，{stem}{ans}。"]
        elif kind == "inverted":
            docs += [f"{ans}，就是{stem[:-1]}。", f"大家都说{ans}就是{stem[:-1]}。"]
    for _cat, stem, ans, _ in EXERCISE_FACTS:
        docs += [f"{stem}{ans}。", f"我们都知道，{stem}{ans}。"]
    # 常见的名字在语料里出现得更多（真实语料也是这样）：给模型一点"热门答案"偏好
    for name in ("巴黎", "北京", "伦敦", "鲁迅", "火星"):
        docs += [f"很多人都听说过{name}。", f"关于{name}的书很多。"]
    # (b) "练习册"：问答格式——练习题，外加被泄漏的考题原文
    for _, _, ans, q in EXERCISE_FACTS:
        docs.append(f"{QA_PREFIX}{q}{QA_MID}{ans}\n")
    for it in items:
        if it.leaked:
            docs.append(f"{QA_PREFIX}{it.question}{QA_MID}{it.choices[it.answer]}\n")
    rng.shuffle(docs)
    return World(corpus="\n".join(docs) + "\n", items=items, docs=docs)


# ---------------------------------------------------------------------------
# 2. 字符级最长后缀模型
# ---------------------------------------------------------------------------


class SuffixLM:
    """P(c | 上下文)：从长度 0 的后缀开始，一路插值到语料里能匹配上的最长后缀（最多 max_len 个字）。

    P_0(c) = (count(c) + 1) / (N + V)                         —— 加一平滑的单字分布
    P_l(c) = (n(s_l, c) + beta · P_{l-1}(c)) / (n(s_l) + beta) —— s_l 是上下文末尾 l 个字
    某个 s_l 在语料里没出现过，就停在上一层。
    """

    def __init__(self, corpus: str, max_len: int = 32, beta: float = 1.0) -> None:
        self.max_len, self.beta = max_len, beta
        self.uni = Counter(corpus)
        self.vocab = sorted(self.uni)
        self.n_chars = len(corpus)
        # next_counts[后缀] = Counter(下一个字)
        self.next_counts: dict[str, Counter] = defaultdict(Counter)
        for i in range(1, len(corpus)):
            for length in range(1, min(max_len, i) + 1):
                self.next_counts[corpus[i - length : i]][corpus[i]] += 1

    def _dist_prob(self, context: str, c: str) -> float:
        p = (self.uni.get(c, 0) + 1) / (self.n_chars + len(self.vocab) + 1)  # +1：未登录字
        for length in range(1, min(self.max_len, len(context)) + 1):
            nxt = self.next_counts.get(context[-length:])
            if not nxt:
                break
            p = (nxt.get(c, 0) + self.beta * p) / (sum(nxt.values()) + self.beta)
        return p

    def logprob(self, context: str, continuation: str) -> float:
        """log P(continuation | context) = Σ log P(第 t 个字 | 上下文 + 前 t-1 个字)。"""
        total = 0.0
        for t, c in enumerate(continuation):
            total += math.log(self._dist_prob(context + continuation[:t], c))
        return total

    def greedy(self, context: str, max_new: int = 10, stop: str = "。\n") -> str:
        out = ""
        for _ in range(max_new):
            ctx = context + out
            c = max(self.vocab, key=lambda ch: self._dist_prob(ctx, ch))
            if c in stop:
                break
            out += c
        return out


# ---------------------------------------------------------------------------
# 3. 两种判分
# ---------------------------------------------------------------------------


def cloze_prompt(it: Item) -> str:
    return it.stem


def qa_prompt(it: Item) -> str:
    return f"{QA_PREFIX}{it.question}{QA_MID}"


def score_mc_loglik(lm: SuffixLM, items: list[Item], prompt_fn) -> dict:  # noqa: ANN001
    """对每个选项算 log P(选项 | 提示词)，取最大者。acc_norm 先除以选项的 UTF-8 字节数。"""
    per = []
    for it in items:
        ctx = prompt_fn(it)
        lps = [lm.logprob(ctx, ch) for ch in it.choices]
        norms = [lp / len(ch.encode("utf-8")) for lp, ch in zip(lps, it.choices)]
        pred = max(range(len(lps)), key=lambda i: lps[i])
        pred_n = max(range(len(norms)), key=lambda i: norms[i])
        per.append(
            {"id": it.id, "lps": lps, "correct": float(pred == it.answer),
             "correct_norm": float(pred_n == it.answer), "leaked": it.leaked}
        )
    n = len(per)
    return {
        "acc": sum(p["correct"] for p in per) / n,
        "acc_norm": sum(p["correct_norm"] for p in per) / n,
        "items": per,
    }


def score_generate_em(lm: SuffixLM, items: list[Item], prompt_fn) -> dict:  # noqa: ANN001
    """不给选项，让模型贪心生成，去掉首尾空白后和标准答案逐字比较。"""
    per = []
    for it in items:
        out = lm.greedy(prompt_fn(it)).strip()
        per.append({"id": it.id, "output": out,
                    "correct": float(out == it.choices[it.answer]), "leaked": it.leaked})
    return {"em": sum(p["correct"] for p in per) / len(per), "items": per}


def split_acc(per: list[dict], key: str = "correct") -> tuple[float, float]:
    """(泄漏题上的正确率, 干净题上的正确率)"""
    leak = [p[key] for p in per if p["leaked"]]
    clean = [p[key] for p in per if not p["leaked"]]
    return sum(leak) / len(leak), sum(clean) / len(clean)


def main() -> None:
    world = build_world()
    lm = SuffixLM(world.corpus)
    items = world.items
    print(f"语料 {len(world.corpus)} 字，{len(world.docs)} 篇文档；考题 {len(items)} 道"
          f"（泄漏 {sum(it.leaked for it in items)} 道）；随机猜的期望正确率 0.25\n")

    # 一道题的完整打分过程：挑一道 acc 与 acc_norm 判得不一样的题
    cloze = score_mc_loglik(lm, items, cloze_prompt)
    diff = [p["id"] for p in cloze["items"] if p["correct"] != p["correct_norm"]]
    it = next(x for x in items if x.id == diff[0])
    print(f"acc 与 acc_norm 判得不一样的题：{diff}")
    print(f"例题 {it.id}：{it.stem}____   选项 {it.choices}   答案 {it.choices[it.answer]}")
    print("| 选项 | log P(选项｜题干) | 字节数 | 除以字节数 |")
    print("|---|---:|---:|---:|")
    for ch in it.choices:
        lp = lm.logprob(cloze_prompt(it), ch)
        nb = len(ch.encode("utf-8"))
        print(f"| {ch} | {lp:.2f} | {nb} | {lp / nb:.3f} |")

    print("\n同一个模型、同一批题，两种判分方式：\n")
    print("| 提示词格式 | 判分 | 全部 32 题 | 泄漏的 12 题 | 干净的 20 题 |")
    print("|---|---|---:|---:|---:|")
    for name, fn in (("完形填空", cloze_prompt), ("问答", qa_prompt)):
        r = score_mc_loglik(lm, items, fn)
        lk, cl = split_acc(r["items"])
        print(f"| {name} | 对数似然 acc | {r['acc']:.3f} | {lk:.3f} | {cl:.3f} |")
        lk, cl = split_acc(r["items"], "correct_norm")
        print(f"| {name} | 对数似然 acc_norm | {r['acc_norm']:.3f} | {lk:.3f} | {cl:.3f} |")
        g = score_generate_em(lm, items, fn)
        lk, cl = split_acc(g["items"])
        print(f"| {name} | 生成 + 精确匹配 | {g['em']:.3f} | {lk:.3f} | {cl:.3f} |")
    g = score_generate_em(lm, items, qa_prompt)
    wrong = [p for p in g["items"] if not p["correct"]][:3]
    print("\n生成式判错的几个例子（问答格式）：")
    for p in wrong:
        t = next(x for x in items if x.id == p["id"])
        print(f"  {t.question} → 模型写出 {p['output']!r}，标准答案 {t.choices[t.answer]!r}")


if __name__ == "__main__":
    main()
