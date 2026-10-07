"""Chapter 22 · Minimal code 4: the KV cache of a sliding-window layer has a maximum size.

Use the language model that 02 trained to generate 300 characters. Compare:
  - No cache: at each step, run the full sequence through the model again.
  - Truncated cache: a sliding-window layer keeps only the K/V of the last W positions
    (see KVCache.append in 02).
The two methods must generate the same text, character for character. The script also prints how the
cache size changes with the generated length.

Run: uv run python chapters/22-local-sparse-attention/code/04_bounded_cache.py
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


@torch.no_grad()
def generate(model, ids: list[int], n: int, use_cache: bool, report_at=()):
    ids = list(ids)
    cache = m.KVCache(model.c.n_layers) if use_cache else None
    sizes = {}
    nxt_in, start = torch.tensor([ids]), 0
    for _ in range(n):
        if use_cache:
            logits = model(nxt_in, cache, start)
            start += nxt_in.shape[1]
        else:
            logits = model(torch.tensor([ids]))
        ids.append(int(logits[0, -1].argmax()))
        nxt_in = torch.tensor([[ids[-1]]])
        if use_cache and start in report_at:  # start = the number of positions in the cache
            sizes[start] = cache.nbytes()
    return ids, sizes


def main() -> None:
    data = m.CharData()
    prompt = data.encode("ROMEO:\n")
    report_at = (16, 64, 128, 256, 306)
    print(f"Prompt 'ROMEO:\\n', greedy generation of 300 characters; window W = {m.W}\n")
    print(
        f"{'Config':<11}{'Same as no cache':>18}   KV cache bytes (cached "
        + " / ".join(str(t) for t in report_at)
        + " positions)"
    )
    texts = {}
    for variant in m.VARIANTS:
        model = m.load_or_train("lm", variant)
        a, sizes = generate(model, prompt, 300, use_cache=True, report_at=report_at)
        b, _ = generate(model, prompt, 300, use_cache=False)
        texts[variant] = data.decode(a)
        print(f"{variant:<11}{str(a == b):>18}   " + " / ".join(f"{sizes[t]:,}" for t in report_at))
    print("\nText that sliding generated (first 200 characters):\n" + texts["sliding"][:200])


if __name__ == "__main__":
    main()
