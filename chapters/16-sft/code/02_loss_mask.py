"""Chapter 16 · Minimal code 2: loss mask — calculate the loss only on what the assistant says.

Three parts:
  1. A toy "tool call" task and a character-level tokenizer (each special token has one id).
     Scripts 3 and 4 also use them.
  2. Render one conversation and get a per-token mask: 1 for assistant tokens, 0 for all others.
     Print the result.
  3. On the same batch of data, compare the loss "on assistant tokens only" with the loss
     "on all tokens". Then show how to average the loss over samples of different lengths
     (mean per token vs mean per sample).

Run: uv run python chapters/16-sft/code/02_loss_mask.py
"""

from __future__ import annotations

import importlib.util
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

torch.set_num_threads(1)  # the build machine shares its CPU between many jobs; on your computer, you can remove this line
HERE = Path(__file__).resolve().parent


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


tmpl = load("ch16_chat_template", HERE / "01_chat_template.py")  # render() from script 1

# ── Toy task: two tools, question templates + cities / numbers ──────────────────
SYSTEM = "Tools: get_weather(city), add(a, b)"
TRAIN_CITIES = ["Paris", "London", "Tokyo", "Berlin", "Madrid", "Rome", "Cairo", "Dublin",
                "Oslo", "Vienna", "Prague", "Lisbon", "Athens", "Warsaw", "Seoul", "Delhi",
                "Lima", "Quito", "Sydney", "Boston", "Denver", "Austin", "Miami", "Chicago",
                "Toronto", "Munich", "Milan", "Naples", "Geneva", "Zurich"]
TEST_CITIES = ["Venice", "Hanoi", "Dallas", "Nairobi", "Bogota", "Havana", "Kyoto", "Porto",
               "Helsinki", "Manila"]  # never in the training data: the test measures "copying", not memorized city names
WEATHER_Q = ["What is the weather in {c}?", "How is the weather in {c} today?",
             "Is it raining in {c}?", "Weather for {c}, please."]
ADD_Q = ["What is {a} plus {b}?", "Add {a} and {b}.", "What is {a} + {b}?", "Compute {a}+{b}."]


def make_example(rng: random.Random, split: str = "train") -> dict:
    """One (system, user, assistant) conversation. The assistant reply is one tool call."""
    if rng.random() < 0.5:
        c = rng.choice(TRAIN_CITIES if split == "train" else TEST_CITIES)
        q, call = rng.choice(WEATHER_Q).format(c=c), {"name": "get_weather", "arguments": {"city": c}}
    else:
        a, b = rng.randint(0, 99), rng.randint(0, 99)
        q, call = rng.choice(ADD_Q).format(a=a, b=b), {"name": "add", "arguments": {"a": a, "b": b}}
    return {"messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": q},
                         {"role": "assistant", "content": "", "tool_calls": [call]}],
            "call": call}


# ── Character-level tokenizer: Shakespeare characters of Chapter 10 + new characters + special tokens
SPECIALS = ["<|im_start|>", "<|im_end|>", "<tool_call>", "</tool_call>"]
_SPECIAL_RE = "(" + "|".join(s.replace("|", r"\|") for s in SPECIALS) + ")"


class ChatTok:
    """base_chars: the vocabulary of the base model (Chapter 10: the 65 characters in Shakespeare).

    The SFT data has characters that the base model never saw ({ } " _ digits, ...) and special
    tokens. They go at the end of the vocabulary.
    """

    def __init__(self, base_chars: list[str]) -> None:
        import re

        self._re = re
        extra = sorted({ch for ch in '{}"_<>/|0123456789+()' if ch not in base_chars})
        self.itos = list(base_chars) + extra + SPECIALS
        self.stoi = {s: i for i, s in enumerate(self.itos)}
        self.n_base = len(base_chars)
        self.im_end = self.stoi["<|im_end|>"]

    def encode(self, text: str) -> list[int]:
        ids = []
        for part in self._re.split(_SPECIAL_RE, text):
            if part in self.stoi and part in SPECIALS:
                ids.append(self.stoi[part])  # a special token is one id; it is never split
            else:
                ids.extend(self.stoi[ch] for ch in part)
        return ids

    def decode(self, ids) -> str:
        return "".join(self.itos[i] for i in ids)


def encode_with_mask(tok: ChatTok, messages, add_generation_prompt: bool = False):
    """Render → (ids, mask). Encode segment by segment, so the mask follows the segments.

    This is correct here because character-level tokens never merge across segment boundaries.
    """
    ids, mask = [], []
    for text, _role, train in tmpl.render(messages, add_generation_prompt=add_generation_prompt):
        t = tok.encode(text)
        ids += t
        mask += [train] * len(t)
    return ids, mask


def shakespeare_chars() -> list[str]:
    text = (HERE.parents[2] / "assets" / "tiny_corpus" / "shakespeare.txt").read_text("utf-8")
    return sorted(set(text))


def masked_targets(ids: list[int], mask: list[bool], use_mask: bool = True):
    """x = ids[:-1], y = ids[1:]. Positions of y that are not in the loss become -100.

    -100 is the ignore_index of cross_entropy. Note mask[1:]: position t predicts token t+1.
    The mask of the predicted token decides if the position is in the loss.
    """
    x = torch.tensor(ids[:-1])
    y = torch.tensor(ids[1:])
    if use_mask:
        y = torch.where(torch.tensor(mask[1:]), y, torch.full_like(y, -100))
    return x, y


if __name__ == "__main__":
    tok = ChatTok(shakespeare_chars())
    print(f"Vocabulary: {tok.n_base} base characters + {len(tok.itos) - tok.n_base - len(SPECIALS)} new characters"
          f" + {len(SPECIALS)} special tokens = {len(tok.itos)}")
    ex = make_example(random.Random(0))
    ids, mask = encode_with_mask(tok, ex["messages"])
    print("\nOne toy sample (the tokens in [square brackets] are in the loss):")
    out, inside = "", False
    for i, m in zip(ids, mask):
        if m and not inside:
            out += "["
        if not m and inside:
            out += "]"
        inside = m
        out += tok.decode([i])
    print(out + ("]" if inside else ""))
    print(f"{len(ids)} tokens in total, {sum(mask)} in the loss ({100 * sum(mask) / len(ids):.0f}%)")

    # The real sample (the tool conversation of script 1), counted in characters: the assistant part is small
    segs = tmpl.render(tmpl.MESSAGES, tmpl.TOOLS)
    n = sum(len(s) for s, _, _ in segs)
    k = sum(len(s) for s, _, t in segs if t)
    print(f"\nReal tool conversation from script 1: {n} characters, {k} in the loss ({100 * k / n:.1f}%)")

    # ── Same batch of data, two losses ───────────────────────────────────────
    # A random embedding + linear layer replaces the model (the demo shows the algorithm; no training)
    torch.manual_seed(0)
    V = len(tok.itos)
    emb, head = torch.nn.Embedding(V, 16), torch.nn.Linear(16, V)
    rng = random.Random(1)
    batch = [encode_with_mask(tok, make_example(rng)["messages"]) for _ in range(4)]
    tot_all, tot_asst, n_all, n_asst = 0.0, 0.0, 0, 0
    for ids, mask in batch:
        x, y_full = masked_targets(ids, mask, use_mask=False)
        _, y_mask = masked_targets(ids, mask, use_mask=True)
        logits = head(emb(x))
        nll = F.cross_entropy(logits, y_full, reduction="none")  # -log p at each position
        keep = y_mask != -100
        tot_all, n_all = tot_all + nll.sum().item(), n_all + len(nll)
        tot_asst, n_asst = tot_asst + nll[keep].sum().item(), n_asst + int(keep.sum())
        # ignore_index gives the mean over the keep positions only
        assert torch.allclose(F.cross_entropy(logits, y_mask, ignore_index=-100), nll[keep].mean())
    print(f"\nSmall random model, 4 samples: mean loss on all tokens {tot_all / n_all:.3f}"
          f" ({n_all} positions), on assistant tokens only {tot_asst / n_asst:.3f} ({n_asst} positions)")
    print("ignore_index=-100 gives the same result as the manual 'mean over assistant positions only' ✓")

    # ── How to average over samples of different lengths ──────────────────────
    # Sample A has 3 assistant tokens, each with loss 2.0. Sample B has 30, each with loss 1.0.
    a, b = torch.full((3,), 2.0), torch.full((30,), 1.0)
    per_token = torch.cat([a, b]).mean()
    per_sample = (a.mean() + b.mean()) / 2
    print(f"\nMean per token: {per_token:.3f}; mean of each sample, then mean over samples: {per_sample:.3f}"
          "\n(With gradient accumulation, if each micro-batch takes its own mean, the first value"
          " silently becomes the second.)")
