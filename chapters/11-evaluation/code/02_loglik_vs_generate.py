"""Two ways to score a multiple-choice question: pick the choice by log-likelihood, or generate and use exact match.

    uv run python chapters/11-evaluation/code/02_loglik_vs_generate.py

The script must run in a few seconds and give fully reproducible results. Thus the "language model"
here is not a neural network. It is a **character-level n-gram model with longest-suffix match**
(the same idea as Infini-gram). To predict the next character, it finds the longest part of the
"training corpus" that matches the end of the context. It looks at the character that follows.
Then it interpolates with shorter matches for smoothing. The model cannot reason. It can only
"memorize" the corpus. This makes it a good tool to show how much an evaluation score depends on
what the corpus contains (contamination) and on the form of the prompt (format sensitivity).

This file defines three things. Files 03 and 05 load them again with importlib:
1. `build_world()`: a toy "world": a table of facts, a training corpus, and 32 four-choice
   questions. The original text of 12 questions leaked into the corpus.
2. `SuffixLM`: the character-level longest-suffix model above.
   `logprob(context, continuation)` returns log P(continuation | context).
3. Two scorers: `score_mc_loglik` (acc and acc_norm) and `score_generate_em`
   (greedy generation + exact match).
"""

from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# 1. Toy world: facts → corpus + exam questions
# ---------------------------------------------------------------------------

# (category, cloze stem, answer, question in QA form)
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
# These facts occur only in the "exercises", not in the exam.
# They teach the model the QA format of the workbook (QA_PREFIX + question + QA_MID + answer).
EXERCISE_FACTS = [
    ("capital", "中国的首都是", "北京", "中国的首都是哪座城市？"),
    ("capital", "越南的首都是", "河内", "越南的首都是哪座城市？"),
    ("author", "《家》的作者是", "巴金", "《家》是谁写的？"),
    ("author", "《围城》的作者是", "钱锺书", "《围城》是谁写的？"),
    ("planet", "离太阳最远的行星是", "海王星", "离太阳最远的行星是哪一颗？"),
    ("element", "金刚石由哪种元素组成？答案是", "碳", "金刚石由哪种元素组成？"),
]
N_LEAKED = 12  # for 12 of the 32 questions, "question + answer" leaked into the training corpus
QA_PREFIX, QA_MID = "问：", "\n答："  # format of the exercises in the corpus: "Q:" and "A:" in Chinese


@dataclass
class Item:
    id: str
    category: str
    stem: str  # cloze stem, for example "The capital of Japan is" (in Chinese)
    question: str  # QA form, for example "Which city is the capital of Japan?" (in Chinese)
    choices: list[str]
    answer: int  # index of the correct choice
    leaked: bool = False


@dataclass
class World:
    corpus: str
    items: list[Item]
    docs: list[str] = field(default_factory=list)  # the corpus split into "documents" (05 uses them for decontamination)


def build_world(seed: int = 0) -> World:
    rng = random.Random(seed)
    by_cat: dict[str, list[str]] = defaultdict(list)
    for cat, _, ans, _ in FACTS + EXERCISE_FACTS:
        by_cat[cat].append(ans)

    items = []
    leaked_ids = set(rng.sample(range(len(FACTS)), N_LEAKED))
    for i, (cat, stem, ans, q) in enumerate(FACTS):
        wrong = rng.sample([a for a in by_cat[cat] if a != ans], 3)  # distractors from the same category
        choices = wrong + [ans]
        rng.shuffle(choices)
        items.append(
            Item(f"q{i:02d}", cat, stem, q, choices, choices.index(ans), i in leaked_ids)
        )

    docs = []
    # (a) The "textbook": the facts behind the exam questions. Half of them use the word order of
    #     the stem ("The capital of Japan is Tokyo."). 30% use the inverse order
    #     ("Tokyo is the capital of Japan."). 20% are not in the textbook.
    #     All facts of the exercises use the word order of the stem.
    kinds = ["direct"] * 16 + ["inverted"] * 10 + ["absent"] * 6
    rng.shuffle(kinds)
    for (_cat, stem, ans, _), kind in zip(FACTS, kinds):
        if kind == "direct":
            docs += [f"{stem}{ans}。", f"我们都知道，{stem}{ans}。"]
        elif kind == "inverted":
            docs += [f"{ans}，就是{stem[:-1]}。", f"大家都说{ans}就是{stem[:-1]}。"]
    for _cat, stem, ans, _ in EXERCISE_FACTS:
        docs += [f"{stem}{ans}。", f"我们都知道，{stem}{ans}。"]
    # Common names occur more often in the corpus (real corpora are the same).
    # This gives the model a small bias toward "popular answers".
    for name in ("巴黎", "北京", "伦敦", "鲁迅", "火星"):
        docs += [f"很多人都听说过{name}。", f"关于{name}的书很多。"]
    # (b) The "workbook": QA format. It has the exercises and the original text of the leaked questions.
    for _, _, ans, q in EXERCISE_FACTS:
        docs.append(f"{QA_PREFIX}{q}{QA_MID}{ans}\n")
    for it in items:
        if it.leaked:
            docs.append(f"{QA_PREFIX}{it.question}{QA_MID}{it.choices[it.answer]}\n")
    rng.shuffle(docs)
    return World(corpus="\n".join(docs) + "\n", items=items, docs=docs)


# ---------------------------------------------------------------------------
# 2. Character-level longest-suffix model
# ---------------------------------------------------------------------------


class SuffixLM:
    """P(c | context): start from the suffix of length 0. Interpolate up to the longest suffix
    that matches in the corpus (at most max_len characters).

    P_0(c) = (count(c) + 1) / (N + V)                         —— add-one smoothed unigram distribution
    P_l(c) = (n(s_l, c) + beta · P_{l-1}(c)) / (n(s_l) + beta) —— s_l is the last l characters of the context
    If an s_l does not occur in the corpus, stop at the level before it.
    """

    def __init__(self, corpus: str, max_len: int = 32, beta: float = 1.0) -> None:
        self.max_len, self.beta = max_len, beta
        self.uni = Counter(corpus)
        self.vocab = sorted(self.uni)
        self.n_chars = len(corpus)
        # next_counts[suffix] = Counter(next character)
        self.next_counts: dict[str, Counter] = defaultdict(Counter)
        for i in range(1, len(corpus)):
            for length in range(1, min(max_len, i) + 1):
                self.next_counts[corpus[i - length : i]][corpus[i]] += 1

    def _dist_prob(self, context: str, c: str) -> float:
        p = (self.uni.get(c, 0) + 1) / (self.n_chars + len(self.vocab) + 1)  # +1: unknown character
        for length in range(1, min(self.max_len, len(context)) + 1):
            nxt = self.next_counts.get(context[-length:])
            if not nxt:
                break
            p = (nxt.get(c, 0) + self.beta * p) / (sum(nxt.values()) + self.beta)
        return p

    def logprob(self, context: str, continuation: str) -> float:
        """log P(continuation | context) = Σ log P(character t | context + the first t-1 characters)."""
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
# 3. Two scorers
# ---------------------------------------------------------------------------


def cloze_prompt(it: Item) -> str:
    return it.stem


def qa_prompt(it: Item) -> str:
    return f"{QA_PREFIX}{it.question}{QA_MID}"


def score_mc_loglik(lm: SuffixLM, items: list[Item], prompt_fn) -> dict:  # noqa: ANN001
    """Calculate log P(choice | prompt) for each choice and take the largest.
    acc_norm first divides by the number of UTF-8 bytes of the choice."""
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
    """Give no choices. Let the model generate greedily. Strip the whitespace at the start and
    the end, then compare the output with the reference answer character by character."""
    per = []
    for it in items:
        out = lm.greedy(prompt_fn(it)).strip()
        per.append({"id": it.id, "output": out,
                    "correct": float(out == it.choices[it.answer]), "leaked": it.leaked})
    return {"em": sum(p["correct"] for p in per) / len(per), "items": per}


def split_acc(per: list[dict], key: str = "correct") -> tuple[float, float]:
    """(accuracy on the leaked questions, accuracy on the clean questions)"""
    leak = [p[key] for p in per if p["leaked"]]
    clean = [p[key] for p in per if not p["leaked"]]
    return sum(leak) / len(leak), sum(clean) / len(clean)


def main() -> None:
    world = build_world()
    lm = SuffixLM(world.corpus)
    items = world.items
    print(f"Corpus: {len(world.corpus)} characters, {len(world.docs)} documents; "
          f"exam: {len(items)} questions ({sum(it.leaked for it in items)} leaked); "
          "expected accuracy of a random guess: 0.25\n")

    # The full scoring of one question: pick a question where acc and acc_norm do not agree
    cloze = score_mc_loglik(lm, items, cloze_prompt)
    diff = [p["id"] for p in cloze["items"] if p["correct"] != p["correct_norm"]]
    it = next(x for x in items if x.id == diff[0])
    print(f"Questions where acc and acc_norm do not agree: {diff}")
    print(f"Example {it.id}: {it.stem}____   choices {it.choices}   answer {it.choices[it.answer]}")
    print("| Choice | log P(choice ∣ stem) | Bytes | Divided by bytes |")
    print("|---|---:|---:|---:|")
    for ch in it.choices:
        lp = lm.logprob(cloze_prompt(it), ch)
        nb = len(ch.encode("utf-8"))
        print(f"| {ch} | {lp:.2f} | {nb} | {lp / nb:.3f} |")

    print("\nSame model, same questions, two scoring methods:\n")
    print("| Prompt format | Scoring | All 32 | Leaked 12 | Clean 20 |")
    print("|---|---|---:|---:|---:|")
    for name, fn in (("Cloze", cloze_prompt), ("QA", qa_prompt)):
        r = score_mc_loglik(lm, items, fn)
        lk, cl = split_acc(r["items"])
        print(f"| {name} | log-likelihood acc | {r['acc']:.3f} | {lk:.3f} | {cl:.3f} |")
        lk, cl = split_acc(r["items"], "correct_norm")
        print(f"| {name} | log-likelihood acc_norm | {r['acc_norm']:.3f} | {lk:.3f} | {cl:.3f} |")
        g = score_generate_em(lm, items, fn)
        lk, cl = split_acc(g["items"])
        print(f"| {name} | generate + exact match | {g['em']:.3f} | {lk:.3f} | {cl:.3f} |")
    g = score_generate_em(lm, items, qa_prompt)
    wrong = [p for p in g["items"] if not p["correct"]][:3]
    print("\nSome wrong answers from generation (QA format):")
    for p in wrong:
        t = next(x for x in items if x.id == p["id"])
        print(f"  {t.question} → model wrote {p['output']!r}, reference answer {t.choices[t.answer]!r}")


if __name__ == "__main__":
    main()
