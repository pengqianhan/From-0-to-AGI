"""Chapter 22 · Minimal code 5: the core idea of sparse attention. Select k keys by content, not by position.

Take the **full-attention** models that 02 trained. At inference, each query sees only k keys.
Two selection methods with the same budget:
  - The k most recent keys (by position): change each layer for a short time into a sliding window of size k.
  - The k keys with the highest scores (by content): calculate all q·k scores first, keep only the top k,
    then apply softmax.
There is no retraining. We measure how much the needle-in-a-haystack accuracy and the language-modeling
loss become worse.

Note: to select the top k, this script first calculates all the scores. Thus it saves no compute.
Real sparse attention (DSA of DeepSeek, MSA of MiniMax, and others) uses a much cheaper "indexer" to give
the scores. Then it calculates exact attention on the selected keys.
Run: uv run python chapters/22-local-sparse-attention/code/05_topk_sparse.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

torch.set_num_threads(1)
_spec = importlib.util.spec_from_file_location(
    "swa_model", Path(__file__).resolve().parent / "02_swa_model.py"
)
m = importlib.util.module_from_spec(_spec)
sys.modules["swa_model"] = m  # dataclass must find its module in sys.modules
_spec.loader.exec_module(m)


def set_recent(model, k: int | None) -> None:
    for blk in model.blocks:
        blk.attn.window = k


def main() -> None:
    needle = m.load_or_train("needle", "full")
    lm = m.load_or_train("lm", "full")
    rf = 4 * (m.W - 1)
    print("Full-attention models; at inference, each query keeps only k keys (no retraining)")
    print(
        f"{'Selection':<16}{'k':>4}{f'needle<{m.W}':>11}{f'needle>{rf}':>11}{'all d':>10}{'LM loss':>9}"
    )
    base_acc = m.needle_accuracy(needle)
    base_loss = m.lm_val_loss(lm)
    print(
        f"{'No limit (full)':<16}{'-':>4}{base_acc[: m.W - 1].mean():>11.1%}"
        f"{base_acc[rf:].mean():>11.1%}{base_acc.mean():>10.1%}{base_loss:>9.3f}"
    )
    for k in (4, 8, 16):
        for name in ("k most recent", "k highest scores"):
            for model in (needle, lm):
                if name == "k most recent":
                    set_recent(model, k)
                else:
                    model.set_topk(k)
            acc = m.needle_accuracy(needle)
            loss = m.lm_val_loss(lm)
            for model in (needle, lm):
                set_recent(model, None)
                model.set_topk(None)
            print(
                f"{name:<16}{k:>4}{acc[: m.W - 1].mean():>11.1%}{acc[rf:].mean():>11.1%}"
                f"{acc.mean():>10.1%}{loss:>9.3f}"
            )


if __name__ == "__main__":
    main()
