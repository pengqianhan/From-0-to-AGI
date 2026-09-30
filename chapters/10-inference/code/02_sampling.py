"""第 10 章 · 极简代码 2：解码策略——贪心、温度、top-k、top-p

同一个模型、同一段提示词，"怎么从概率分布里挑下一个字"决定了生成的样子。
这里拿 01 训练好的小模型，打印真实的下一个字符分布，看温度、top-k、top-p 怎么改它，
再用不同策略各生成一段文字。
运行：uv run python chapters/10-inference/code/02_sampling.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # dataclass 需要模块已登记
    spec.loader.exec_module(mod)
    return mod


tiny = _load("tiny_model", "01_tiny_model.py")


def filtered_probs(logits: torch.Tensor, temperature: float = 1.0, top_k: int = 0,
                   top_p: float = 1.0) -> torch.Tensor:
    """把 logits 变成"真正用来抽样"的分布：先除以温度再 softmax，然后按 top-k / top-p 截断并重新归一化。"""
    probs = torch.softmax(logits / temperature, dim=-1)  # 第 5 章的带温度 softmax
    if top_k > 0:  # 只留概率最大的 k 个
        kth = torch.topk(probs, top_k).values[-1]
        probs = torch.where(probs >= kth, probs, 0.0)
    if top_p < 1.0:  # nucleus：从大到小累加，刚好够 top_p 就停
        sorted_p, idx = torch.sort(probs, descending=True)
        before = torch.cumsum(sorted_p, 0) - sorted_p  # 加上自己之前的累计概率
        keep = torch.zeros_like(probs, dtype=torch.bool)
        keep[idx] = before < top_p  # 保证至少留下最大的那个
        probs = torch.where(keep, probs, 0.0)
    return probs / probs.sum()


def sample_next(logits: torch.Tensor, temperature: float = 1.0, top_k: int = 0,
                top_p: float = 1.0, g: torch.Generator | None = None) -> int:
    if temperature == 0:  # 贪心：永远挑最大的
        return int(logits.argmax())
    probs = filtered_probs(logits, temperature, top_k, top_p)
    return int(torch.multinomial(probs, 1, generator=g))


@torch.no_grad()
def next_logits(model, data, prompt: str) -> torch.Tensor:
    return model(torch.tensor([data.encode(prompt)]))[0, -1]


@torch.no_grad()
def generate(model, data, prompt: str, n: int, seed: int = 0, **kw) -> str:
    """最朴素的生成循环（每步重算整段；03 会把它换成 KV cache 版）。"""
    g = torch.Generator().manual_seed(seed)
    ids = data.encode(prompt)
    for _ in range(n):
        logits = model(torch.tensor([ids]))[0, -1]
        ids.append(sample_next(logits, g=g, **kw))
    return data.decode(ids[len(data.encode(prompt)):])


def distinct_4gram(text: str) -> float:
    """不重复的 4 字符片段占比：越低说明越在原地打转。"""
    grams = [text[i : i + 4] for i in range(len(text) - 3)]
    return len(set(grams)) / max(1, len(grams))


def show(ch: str) -> str:
    return {"\n": "\\n", " ": "␣"}.get(ch, ch)


PROMPT = "ROMEO:\nI will "
CONTEXTS = {"确定": "KING RICHARD III:\nWhat is th", "不确定": "ROMEO:\n"}
STRATEGIES = [
    ("贪心 (T=0)", dict(temperature=0)),
    ("T=0.5", dict(temperature=0.5)),
    ("T=1.0", dict(temperature=1.0)),
    ("T=1.0, top-k=5", dict(temperature=1.0, top_k=5)),
    ("T=1.0, top-p=0.9", dict(temperature=1.0, top_p=0.9)),
    ("T=1.5", dict(temperature=1.5)),
]


if __name__ == "__main__":
    model, data = tiny.load_or_train(4), tiny.CharData()
    logits = next_logits(model, data, PROMPT)

    print(f"提示词 {PROMPT!r} 之后，下一个字符的真实分布（前 8 名）：")
    order = torch.argsort(logits, descending=True)[:8]
    print("  温度     " + "  ".join(f"{show(data.chars[i]):>5}" for i in order) + "    熵(nats)")
    for T in (0.5, 1.0, 1.5):
        p = filtered_probs(logits, T)
        ent = -(p * p.clamp_min(1e-12).log()).sum()
        print(f"  T={T:<4}  " + "  ".join(f"{p[i]:5.3f}" for i in order) + f"    {ent:.2f}")

    print("\ntop-k 固定留 k 个；top-p 按累计概率留，个数随上下文变化：")
    for name, ctx in CONTEXTS.items():
        lg = next_logits(model, data, ctx)
        p = torch.softmax(lg, -1)
        top1 = int(p.argmax())
        n90 = int((filtered_probs(lg, 1.0, top_p=0.9) > 0).sum())
        print(f"  {name}：{ctx!r:32} 最可能 {show(data.chars[top1])!r} p={p[top1]:.3f}  "
              f"top-p=0.9 留下 {n90} 个字符，top-k=5 永远留 5 个")

    print("\n同一提示词，不同解码策略各生成 200 个字符（种子 0）：")
    for name, kw in STRATEGIES:
        text = generate(model, data, PROMPT, 200, seed=0, **kw)
        print(f"\n[{name}]  不重复 4-gram 占比 {distinct_4gram(text):.2f}")
        print("  " + text[:110].replace("\n", "\n  "))
