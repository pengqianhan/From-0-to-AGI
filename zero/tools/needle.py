"""Needle in a haystack: a smoke test for long context (Chapter 15).

The test puts one sentence (the "needle") at some depth in a long text with no relation to it
(the "haystack"):

    The pass key for <key> is <7-digit number>.

At the end, the test asks the model for the number. It checks if the model can find the number
in the context. A sweep over length × depth gives a table.
The design follows the original test of Kamradt (2023), passkey retrieval
(Mohtashami & Jaggi 2023), and S-NIAH from RULER (Hsieh et al. 2024, arXiv:2404.06654).
The needle is "key → 7-digit number". By default, the haystack is the repeated noise sentences of
RULER. You can also use any real text.

There are two scores:
- **Generation** (recall score, as in RULER): greedy decoding generates some tokens. The answer is
  correct if the correct number is in them.
- **Likelihood**: the mean negative log-likelihood (NLL) of the correct number, compared with a
  "control prompt". The control prompt has the same structure; only the number in the needle is a
  different random number. `nll_gain = NLL(control) − NLL(true needle)` > 0 shows that the model
  really uses the information in the needle.
  For a very small model, the generation score is almost always 0. The likelihood metric shows
  smaller differences.

Needle in a haystack is only a **smoke test**. A pass does not show good long-context ability.
(RULER found that many models get almost full NIAH scores, but their scores on complex tasks
decrease much as the length increases.) A fail always shows a problem. For a real evaluation, use
benchmarks such as RULER (Chapters 11 and 20).

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

# The noise haystack of RULER (it uses the passkey sentences of Mohtashami & Jaggi 2023)
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
    """One needle-in-a-haystack question. The length of prompt_ids is exactly context_tokens."""

    prompt_ids: list[int]
    control_ids: list[int]  # control: the number in the needle is the decoy; all other tokens are the same
    answer: str
    answer_ids: list[int]
    decoy: str
    key: str
    depth: float  # requested depth (0 = start, 1 = immediately before the question)
    needle_pos: int  # index of the first needle token in prompt_ids
    context_tokens: int


def _texts(lang: str, key: str, number: str) -> tuple[str, str, str]:
    """(needle, question, answer text). The question ends with "...is", so a base model can
    continue with the number (a completion-style question).

    The sentences are shorter than in RULER: the context of a small model has only a few hundred
    tokens, so the question must not be long.
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
            raise ValueError("Cannot split haystack_text into sentences")
    return parts if lang == "zh" else [" " + p for p in parts]  # English sentences need a space between them


def make_case(
    tokenizer: TokenizerLike,
    context_tokens: int,
    depth: float,
    seed: int = 0,
    lang: str = "en",
    haystack_text: str | None = None,
) -> NeedleCase:
    """Make one question: haystack + needle at depth + question, exactly context_tokens tokens in total.

    Encode each part separately and then join them (do not encode the full text at once).
    Then the length is exact, and the position of the needle is known exactly.
    """
    if not 0.0 <= depth <= 1.0:
        raise ValueError(f"depth must be in [0, 1], got {depth}")
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
                f"context_tokens={context_tokens} is too short: needle + question need {len(needle_ids) + len(question_ids)} tokens"
            )
        # In rare cases, the two numbers give different token counts. Try again with a new seed,
        # so that the control prompt aligns token by token.
        return make_case(tokenizer, context_tokens, depth, seed + 7919, lang, haystack_text)

    # Haystack: start at a random sentence and repeat the sentences. Record the sentence
    # boundaries. At the end, cut to exactly budget tokens.
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
    ins = min(bounds, key=lambda b: (abs(b - target), b))  # the sentence boundary nearest to the target
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
    """Recall score (the same as RULER): 1 point if the generated text contains the full correct number."""
    return 1.0 if answer in generated else 0.0


@torch.no_grad()
def answer_nll(
    model: torch.nn.Module, prompt_ids: Sequence[int], answer_ids: Sequence[int]
) -> float:
    """Mean negative log-likelihood of the correct answer tokens after the prompt (nat/token)."""
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
    nll: float  # mean NLL of the correct answer
    nll_control: float  # mean NLL of the same answer after the control prompt
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
    """Sweep length × depth, with n questions in each cell.

    generate_fn(prompt_ids) -> text: by default, greedy decoding with zero.generate. You can give
    a different function (for example an HF model, or a fake model for tests).
    with_nll=False skips the likelihood (use it when model is not a zero Transformer).
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
    """Format two "length × depth" tables: generation accuracy and likelihood gain nll_gain."""
    lengths = sorted({r.length for r in results})
    depths = sorted({r.depth for r in results})
    cell = {(r.length, r.depth): r for r in results}
    lines = []
    for title, get, fmt in [
        ("Generation accuracy (fraction of greedy outputs that contain the correct number)", lambda r: r.accuracy, "{:>8.2f}"),
        ("Likelihood gain nll_gain = NLL(control) − NLL(true needle); > 0 means that the model uses the needle", lambda r: r.nll_gain,
         "{:>+8.3f}"),
    ]:  # fmt: skip
        lines.append(title)
        lines.append("length\\depth" + "".join(f"{d:>8.2f}" for d in depths))
        for L in lengths:
            row = "".join(fmt.format(get(cell[(L, d)])) for d in depths if (L, d) in cell)
            lines.append(f"   {L:>9}" + row)
        lines.append("")
    return "\n".join(lines)


def _parse_list(s: str, typ: type) -> list:
    return [typ(x) for x in s.split(",") if x.strip()]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Needle in a haystack: a length × depth retrieval smoke test")
    ap.add_argument("--model", required=True, help="zero checkpoint directory or exported HF directory")
    ap.add_argument(
        "--tokenizer", default=None, help="tokenizer path (default: from the checkpoint meta)"
    )
    ap.add_argument("--lengths", default="128,256", help="context lengths (tokens), comma-separated")
    ap.add_argument("--depths", default="0,0.25,0.5,0.75,1", help="needle depths, comma-separated")
    ap.add_argument("--n", type=int, default=5, help="questions per cell")
    ap.add_argument("--lang", choices=["en", "zh"], default="en")
    ap.add_argument("--haystack", default=None, help="use this text file as the haystack (default: noise sentences)")
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
                f"Skip length {L}: with {args.max_new_tokens} generated tokens, it is longer than max_seq_len={max_len}"
            )
        else:
            lengths.append(L)
    hay = open(args.haystack, encoding="utf-8").read() if args.haystack else None
    res = run_grid(model, tok, lengths, _parse_list(args.depths, float), n=args.n, seed=args.seed,
                   lang=args.lang, haystack_text=hay, max_new_tokens=args.max_new_tokens)  # fmt: skip
    print(f"Model {args.model} (max_seq_len={max_len}, rope_theta={model.config.rope_theta}, "
          f"rope_scaling={model.config.rope_scaling}); {args.n} questions per cell\n")  # fmt: skip
    print(format_grid(res))
    first = res[0] if res else None
    if first is not None and first.samples:
        print(f"Example output (length {first.length}, depth {first.depth}): {first.samples[0]!r}")


if __name__ == "__main__":
    main()
