"""大海捞针（needle-in-a-haystack）：长上下文的冒烟测试（对应第 15 章）。

在一段很长的无关文字（"草堆"）的某个深度插入一句"针"：

    The pass key for <key> is <7 位数字>.

最后问模型这个数字是多少，看它能不能从上下文里找回来。长度 × 深度扫一遍，得到一张表。
写法参照 Kamradt（2023）的原始测试、passkey 检索（Mohtashami & Jaggi 2023）与 RULER
（Hsieh et al. 2024, arXiv:2404.06654）的 S-NIAH：针是"key → 7 位数字"，草堆默认用 RULER 的
重复噪声句子，也可以换成任意真实文本。

两种打分：
- **生成式**（与 RULER 一样按召回判分）：贪心生成若干 token，里面出现正确数字就算对；
- **似然式**：正确数字的平均负对数似然（NLL），并和"对照提示词"比较——对照提示词结构完全相同，
  只是针里的数字换成另一个随机数。`nll_gain = NLL(对照) − NLL(真针)` > 0 说明模型确实在用针里的信息。
  极小模型的生成式得分几乎一定是 0，似然式指标能看出更细的差别。

大海捞针只是**冒烟测试**：通过了不代表长上下文能力好（RULER 发现很多模型 NIAH 接近满分，
复杂任务却随长度明显下降），没通过则一定有问题。正式评测用 RULER 等基准（第 11、20 章）。

    uv run python -m zero.tools.needle --model out/tiny/midtrain/ckpt --lengths 64,128,240 --depths 0,0.5,1
"""

from __future__ import annotations

import argparse
import random
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import torch
import torch.nn.functional as F

# RULER 的噪声草堆（沿用 Mohtashami & Jaggi 2023 的 passkey 句子）
NOISE_EN = [
    "The grass is green.",
    "The sky is blue.",
    "The sun is yellow.",
    "Here we go.",
    "There and back again.",
]
NOISE_ZH = ["草是绿的。", "天是蓝的。", "太阳是黄的。", "我们出发吧。", "去了又回来。"]
KEYS = [
    "apple", "river", "window", "garden", "rocket", "violin", "harbor", "lantern",
    "meadow", "puzzle", "silver", "thunder", "canyon", "pepper", "falcon", "marble",
]  # fmt: skip
KEYS_ZH = [
    "苹果",
    "河流",
    "窗户",
    "花园",
    "火箭",
    "提琴",
    "港口",
    "灯笼",
    "草地",
    "拼图",
    "白银",
    "雷声",
]


class TokenizerLike(Protocol):
    def encode(self, text: str) -> list[int]: ...

    def decode(self, ids: Sequence[int]) -> str: ...


@dataclass
class NeedleCase:
    """一道大海捞针题。prompt_ids 的长度恰好等于 context_tokens。"""

    prompt_ids: list[int]
    control_ids: list[int]  # 对照：针里的数字换成 decoy，其余逐 token 相同
    answer: str
    answer_ids: list[int]
    decoy: str
    key: str
    depth: float  # 请求的深度（0 = 开头，1 = 紧挨着问题）
    needle_pos: int  # 针的第一个 token 在 prompt_ids 里的下标
    context_tokens: int


def _texts(lang: str, key: str, number: str) -> tuple[str, str, str]:
    """(针, 问题, 答案文本)。问题以"...is"结尾，Base 模型可以直接接着写出数字（续写式提问）。

    句式比 RULER 的短（小模型的上下文只有几百个 token，问题本身不能太长）。
    """
    if lang == "zh":
        needle = f"{key}的密码是{number}。"
        question = f"\n问：{key}的密码是多少？答：{key}的密码是"
        return needle, question, number
    needle = f" The pass key for {key} is {number}."
    question = f"\nWhat is the pass key for {key}? The pass key for {key} is"
    return needle, question, f" {number}"


def _sentences(lang: str, haystack_text: str | None) -> list[str]:
    if haystack_text is None:
        parts = NOISE_ZH if lang == "zh" else NOISE_EN
    else:
        parts = [p.strip() for p in re.split(r"(?<=[.!?。！？\n])", haystack_text) if p.strip()]
        if not parts:
            raise ValueError("haystack_text 里切不出句子")
    return parts if lang == "zh" else [" " + p for p in parts]  # 英文句子之间留空格


def make_case(
    tokenizer: TokenizerLike,
    context_tokens: int,
    depth: float,
    seed: int = 0,
    lang: str = "en",
    haystack_text: str | None = None,
) -> NeedleCase:
    """构造一道题：草堆 + 在 depth 处插入的针 + 问题，总长度恰好 context_tokens 个 token。

    各段分别编码再拼接（而不是整段一起编码），这样长度可以精确控制，针的位置也精确可知。
    """
    if not 0.0 <= depth <= 1.0:
        raise ValueError(f"depth 必须在 [0, 1]，当前 {depth}")
    rng = random.Random(seed)
    key = rng.choice(KEYS_ZH if lang == "zh" else KEYS)
    number = str(rng.randint(1_000_000, 9_999_999))
    decoy = number
    while decoy == number:
        decoy = str(rng.randint(1_000_000, 9_999_999))
    needle, question, answer_text = _texts(lang, key, number)
    decoy_needle, _, _ = _texts(lang, key, decoy)
    needle_ids = tokenizer.encode(needle)
    decoy_ids = tokenizer.encode(decoy_needle)
    question_ids = tokenizer.encode(question)
    budget = context_tokens - len(needle_ids) - len(question_ids)
    if budget < 0 or len(decoy_ids) != len(needle_ids):
        if budget < 0:
            raise ValueError(
                f"context_tokens={context_tokens} 太短：针 + 问题就要 {len(needle_ids) + len(question_ids)} 个 token"
            )
        # 极少数情况下两个数字切出来的 token 数不同：换一个种子重来，保证对照提示词逐位对齐
        return make_case(tokenizer, context_tokens, depth, seed + 7919, lang, haystack_text)

    # 草堆：从随机的一句开始循环填充，记录句子边界，最后截到恰好 budget 个 token
    sents = _sentences(lang, haystack_text)
    start = rng.randrange(len(sents))
    filler: list[int] = []
    bounds = [0]
    i = start
    while len(filler) < budget:
        filler.extend(tokenizer.encode(sents[i % len(sents)]))
        bounds.append(min(len(filler), budget))
        i += 1
    filler = filler[:budget]
    target = depth * budget
    ins = min(bounds, key=lambda b: (abs(b - target), b))  # 离目标最近的句子边界
    prompt = filler[:ins] + needle_ids + filler[ins:] + question_ids
    control = filler[:ins] + decoy_ids + filler[ins:] + question_ids
    assert len(prompt) == context_tokens == len(control)
    return NeedleCase(
        prompt_ids=prompt,
        control_ids=control,
        answer=number,
        answer_ids=tokenizer.encode(answer_text),
        decoy=decoy,
        key=key,
        depth=depth,
        needle_pos=ins,
        context_tokens=context_tokens,
    )


def score_text(generated: str, answer: str) -> float:
    """召回式判分（与 RULER 相同）：生成文本里出现完整的正确数字就得 1 分。"""
    return 1.0 if answer in generated else 0.0


@torch.no_grad()
def answer_nll(
    model: torch.nn.Module, prompt_ids: Sequence[int], answer_ids: Sequence[int]
) -> float:
    """在 prompt 之后，正确答案各 token 的平均负对数似然（nat/token）。"""
    device = next(model.parameters()).device
    ids = torch.tensor([list(prompt_ids) + list(answer_ids)], dtype=torch.long, device=device)
    logits = model(ids[:, :-1])
    n = len(answer_ids)
    tgt = ids[0, -n:]
    return F.cross_entropy(logits[0, -n:].float(), tgt).item()


@dataclass
class CellResult:
    length: int
    depth: float
    accuracy: float
    nll: float  # 正确答案的平均 NLL
    nll_control: float  # 同一答案在对照提示词下的平均 NLL
    samples: list[str] = field(default_factory=list)

    @property
    def nll_gain(self) -> float:
        return self.nll_control - self.nll


def _default_generate(model: Any, tokenizer: TokenizerLike, max_new_tokens: int) -> Callable:
    from zero.generate import generate

    def fn(prompt_ids: list[int]) -> str:
        return tokenizer.decode(generate(model, prompt_ids, max_new_tokens, temperature=0.0))

    return fn


def run_grid(
    model: Any,
    tokenizer: TokenizerLike,
    lengths: Sequence[int],
    depths: Sequence[float],
    n: int = 5,
    seed: int = 0,
    lang: str = "en",
    haystack_text: str | None = None,
    max_new_tokens: int = 12,
    generate_fn: Callable[[list[int]], str] | None = None,
    with_nll: bool = True,
) -> list[CellResult]:
    """长度 × 深度扫一遍，每格 n 道题。

    generate_fn(prompt_ids) -> 文本：默认用 zero.generate 贪心解码；也可以传别的（比如 HF 模型、测试用的假模型）。
    with_nll=False 时不算似然（model 不是 zero Transformer 时用）。
    """
    gen = generate_fn or _default_generate(model, tokenizer, max_new_tokens)
    out = []
    for L in lengths:
        for d in depths:
            accs, nlls, ctrls, samples = [], [], [], []
            for k in range(n):
                case = make_case(tokenizer, L, d, seed=seed * 100_003 + k, lang=lang,
                                 haystack_text=haystack_text)  # fmt: skip
                text = gen(case.prompt_ids)
                accs.append(score_text(text, case.answer))
                samples.append(text)
                if with_nll:
                    nlls.append(answer_nll(model, case.prompt_ids, case.answer_ids))
                    ctrls.append(answer_nll(model, case.control_ids, case.answer_ids))
            mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")  # noqa: E731
            out.append(CellResult(L, d, mean(accs), mean(nlls), mean(ctrls), samples))
    return out


def format_grid(results: Sequence[CellResult]) -> str:
    """打印成"长度 × 深度"的两张表：生成式准确率、似然增益 nll_gain。"""
    lengths = sorted({r.length for r in results})
    depths = sorted({r.depth for r in results})
    cell = {(r.length, r.depth): r for r in results}
    lines = []
    for title, get, fmt in [
        ("生成式准确率（贪心解码里出现正确数字的比例）", lambda r: r.accuracy, "{:>8.2f}"),
        ("似然增益 nll_gain = NLL(对照) − NLL(真针)，> 0 表示模型在用针的信息", lambda r: r.nll_gain,
         "{:>+8.3f}"),
    ]:  # fmt: skip
        lines.append(title)
        lines.append("   长度\\深度" + "".join(f"{d:>8.2f}" for d in depths))
        for L in lengths:
            row = "".join(fmt.format(get(cell[(L, d)])) for d in depths if (L, d) in cell)
            lines.append(f"   {L:>9}" + row)
        lines.append("")
    return "\n".join(lines)


def _parse_list(s: str, typ: type) -> list:
    return [typ(x) for x in s.split(",") if x.strip()]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="大海捞针：长度 × 深度的检索冒烟测试")
    ap.add_argument("--model", required=True, help="zero checkpoint 目录或导出的 HF 目录")
    ap.add_argument(
        "--tokenizer", default=None, help="分词器路径（默认从 checkpoint 的 meta 里读）"
    )
    ap.add_argument("--lengths", default="128,256", help="上下文长度（token），逗号分隔")
    ap.add_argument("--depths", default="0,0.25,0.5,0.75,1", help="针的深度，逗号分隔")
    ap.add_argument("--n", type=int, default=5, help="每格几道题")
    ap.add_argument("--lang", choices=["en", "zh"], default="en")
    ap.add_argument("--haystack", default=None, help="用这个文本文件当草堆（默认用噪声句子）")
    ap.add_argument("--max-new-tokens", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    from zero.post.common import load_policy

    torch.set_num_threads(1)
    model, tok = load_policy(args.model, args.tokenizer)
    model.eval()
    max_len = model.config.max_seq_len
    lengths = []
    for L in _parse_list(args.lengths, int):
        if L + args.max_new_tokens > max_len:
            print(
                f"跳过长度 {L}：加上生成的 {args.max_new_tokens} 个 token 超过 max_seq_len={max_len}"
            )
        else:
            lengths.append(L)
    hay = open(args.haystack, encoding="utf-8").read() if args.haystack else None
    res = run_grid(model, tok, lengths, _parse_list(args.depths, float), n=args.n, seed=args.seed,
                   lang=args.lang, haystack_text=hay, max_new_tokens=args.max_new_tokens)  # fmt: skip
    print(f"模型 {args.model}（max_seq_len={max_len}，rope_theta={model.config.rope_theta}，"
          f"rope_scaling={model.config.rope_scaling}）；每格 {args.n} 道题\n")  # fmt: skip
    print(format_grid(res))
    first = res[0] if res else None
    if first is not None and first.samples:
        print(f"示例（长度 {first.length}、深度 {first.depth}）的生成：{first.samples[0]!r}")


if __name__ == "__main__":
    main()
