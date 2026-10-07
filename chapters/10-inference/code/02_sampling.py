"""Chapter 10 · Minimal code 2: decoding strategies (greedy, temperature, top-k, top-p)

Same model, same prompt: the method that picks the next character from the probability
distribution decides what the generated text looks like.
This script uses the small model that 01 trained. It prints the real distribution of the next
character and shows how temperature, top-k, and top-p change it.
Then it generates one text with each strategy.
Run: uv run python chapters/10-inference/code/02_sampling.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # dataclass needs the module in sys.modules
    spec.loader.exec_module(mod)
    return mod


tiny = _load("tiny_model", "01_tiny_model.py")


def filtered_probs(logits: torch.Tensor, temperature: float = 1.0, top_k: int = 0,
                   top_p: float = 1.0) -> torch.Tensor:
    """Change logits into the distribution that we sample from.

    Divide by the temperature and apply softmax. Then cut with top-k / top-p and normalize again.
    """
    probs = torch.softmax(logits / temperature, dim=-1)  # softmax with temperature (Chapter 5)
    if top_k > 0:  # keep only the k most probable tokens
        kth = torch.topk(probs, top_k).values[-1]
        probs = torch.where(probs >= kth, probs, 0.0)
    if top_p < 1.0:  # nucleus: add from the largest down; stop when the sum reaches top_p
        sorted_p, idx = torch.sort(probs, descending=True)
        before = torch.cumsum(sorted_p, 0) - sorted_p  # cumulative probability before this token
        keep = torch.zeros_like(probs, dtype=torch.bool)
        keep[idx] = before < top_p  # this always keeps the largest one
        probs = torch.where(keep, probs, 0.0)
    return probs / probs.sum()


def sample_next(logits: torch.Tensor, temperature: float = 1.0, top_k: int = 0,
                top_p: float = 1.0, g: torch.Generator | None = None) -> int:
    if temperature == 0:  # greedy: always pick the largest one
        return int(logits.argmax())
    probs = filtered_probs(logits, temperature, top_k, top_p)
    return int(torch.multinomial(probs, 1, generator=g))


@torch.no_grad()
def next_logits(model, data, prompt: str) -> torch.Tensor:
    return model(torch.tensor([data.encode(prompt)]))[0, -1]


@torch.no_grad()
def generate(model, data, prompt: str, n: int, seed: int = 0, **kw) -> str:
    """The most basic generation loop.

    It calculates the full sequence again at each step. 03 replaces it with a KV cache version.
    """
    g = torch.Generator().manual_seed(seed)
    ids = data.encode(prompt)
    for _ in range(n):
        logits = model(torch.tensor([ids]))[0, -1]
        ids.append(sample_next(logits, g=g, **kw))
    return data.decode(ids[len(data.encode(prompt)):])


def distinct_4gram(text: str) -> float:
    """Fraction of distinct 4-character pieces. A lower value means that the text repeats itself more."""
    grams = [text[i : i + 4] for i in range(len(text) - 3)]
    return len(set(grams)) / max(1, len(grams))


def show(ch: str) -> str:
    return {"\n": "\\n", " ": "␣"}.get(ch, ch)


PROMPT = "ROMEO:\nI will "
# The Chinese video (video/scenes.py) uses these labels as keys, so they stay in Chinese.
# LABEL_EN gives the English names for the printed output.
CONTEXTS = {"确定": "KING RICHARD III:\nWhat is th", "不确定": "ROMEO:\n"}
STRATEGIES = [
    ("贪心 (T=0)", dict(temperature=0)),
    ("T=0.5", dict(temperature=0.5)),
    ("T=1.0", dict(temperature=1.0)),
    ("T=1.0, top-k=5", dict(temperature=1.0, top_k=5)),
    ("T=1.0, top-p=0.9", dict(temperature=1.0, top_p=0.9)),
    ("T=1.5", dict(temperature=1.5)),
]
LABEL_EN = {"确定": "certain", "不确定": "uncertain", "贪心 (T=0)": "greedy (T=0)"}


if __name__ == "__main__":
    model, data = tiny.load_or_train(4), tiny.CharData()
    logits = next_logits(model, data, PROMPT)

    print(f"Real distribution of the next character after the prompt {PROMPT!r} (top 8):")
    order = torch.argsort(logits, descending=True)[:8]
    print("  temp    " + "  ".join(f"{show(data.chars[i]):>5}" for i in order) + "    entropy (nats)")
    for T in (0.5, 1.0, 1.5):
        p = filtered_probs(logits, T)
        ent = -(p * p.clamp_min(1e-12).log()).sum()
        print(f"  T={T:<4}  " + "  ".join(f"{p[i]:5.3f}" for i in order) + f"    {ent:.2f}")

    print("\ntop-k always keeps k tokens. top-p keeps tokens by cumulative probability, "
          "so the number changes with the context:")
    for name, ctx in CONTEXTS.items():
        lg = next_logits(model, data, ctx)
        p = torch.softmax(lg, -1)
        top1 = int(p.argmax())
        n90 = int((filtered_probs(lg, 1.0, top_p=0.9) > 0).sum())
        print(f"  {LABEL_EN[name]}: {ctx!r:32} most likely {show(data.chars[top1])!r} p={p[top1]:.3f}  "
              f"top-p=0.9 keeps {n90} characters, top-k=5 always keeps 5")

    print("\nSame prompt, 200 characters from each decoding strategy (seed 0):")
    for name, kw in STRATEGIES:
        text = generate(model, data, PROMPT, 200, seed=0, **kw)
        print(f"\n[{LABEL_EN.get(name, name)}]  distinct 4-gram ratio {distinct_4gram(text):.2f}")
        print("  " + text[:110].replace("\n", "\n  "))
