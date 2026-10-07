"""Chapter 17 · Minimal code 5: why logits KD needs the same vocabulary, and how many parameters a new vocabulary costs

1. Two tokenizers cut the same sentence. The number of tokens and the boundaries are different:
   position t is not the same "next word", and logit dimension v is not the same token.
   Thus we cannot compare the two distributions position by position.
   The kd_loss of zero raises an error when the shapes are different.
2. The main-line model (configs/main/pretrain.toml: 28 layers, width 1280, shared embedding) uses
   its own trained vocabulary of 65,536 tokens. If we change to the Qwen tokenizer, so that we can do
   logits KD from a Qwen teacher, how many more parameters does the model get?

Run: uv run python chapters/17-distillation/code/05_shared_vocab.py      (about 5 s)
"""

import dataclasses
import sys
from pathlib import Path

import torch

torch.set_num_threads(1)
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from zero.config import load_config  # noqa: E402
from zero.model import count_params  # noqa: E402
from zero.post.distill import kd_loss  # noqa: E402

# Vocabulary sizes (read from the config.json of each model in 2026-09): Qwen3-0.6B 151,936; Qwen3.5-0.8B 248,320
VOCABS = {"own BPE (main line)": 65_536, "Qwen3 tokenizer": 151_936, "Qwen3.5 tokenizer": 248_320}
CAP = 800_000_000  # GOAL.md 3.3: the main-line model has at most 0.8B parameters


def toy_tokenize(text: str, vocab: list[str]) -> list[str]:
    """Longest-match tokenization (only to show that "different vocabularies cut the text differently")."""
    out, i = [], 0
    while i < len(text):
        for L in range(min(4, len(text) - i), 0, -1):
            if text[i:i + L] in vocab or L == 1:
                out.append(text[i:i + L])
                i += L
                break
    return out


def main() -> None:
    text = "今天天气很好"  # data: "The weather is very good today"
    a = toy_tokenize(text, ["今天", "天气"])
    b = toy_tokenize(text, ["今天天", "气很", "好"])
    print(f"1) The same sentence \"{text}\" (\"The weather is very good today\"), two vocabularies:")
    print(f"   The student vocabulary cuts it into {len(a)} tokens: {a}")
    print(f"   The teacher vocabulary cuts it into {len(b)} tokens: {b}")
    print("   The number of positions and the boundaries are different. At step 2, the student must predict '天气', "
          "but the teacher predicts '气很'. There are no distributions to compare position by position.")
    try:
        kd_loss(torch.randn(1, len(a), 65_536), torch.randn(1, len(b), 151_936), torch.ones(1, len(a), dtype=torch.bool))
    except ValueError as e:
        print(f"   zero.post.distill.kd_loss raises an error: {e}")

    cfg = load_config(ROOT / "configs" / "main" / "pretrain.toml").model
    print(f"\n2) The shape of the main-line model stays the same ({cfg.n_layers} layers, width {cfg.dim}, shared embedding); only the vocabulary changes:")
    print(f"   {'tokenizer':<22}{'vocab':>9}{'embedding':>14}{'total params':>14}  ≤ 0.8B?")
    base = None
    for name, v in VOCABS.items():
        c = count_params(dataclasses.replace(cfg, vocab_size=v))
        base = base or c["total"]
        ok = "yes" if c["total"] <= CAP else "no"
        extra = "" if c["total"] == base else f"(+{(c['total'] - base) / 1e6:.1f}M)"
        print(f"   {name:<20}{v:>9,}{c['embedding'] / 1e6:>12.1f}M{c['total'] / 1e6:>12.1f}M  {ok} {extra}")


if __name__ == "__main__":
    main()
