"""Chapter 10 · Minimal code 5: GQA / MQA (several query heads share one set of K/V)

The same small model (4 query heads). Only n_kv_heads changes:
  4 = MHA (each query head has its own K/V); 2 = GQA (each 2 query heads share one set);
  1 = MQA (all query heads share one set).
All variants use the same seed, the same data order, and the same 600 training steps.
The script compares the validation loss, the number of parameters, the KV cache size,
and the generation speed.
It also trains MHA again with a different random seed. This shows how much the loss changes
"only because of a new seed". Do not trust a difference that is smaller than this number.
The first run trains 3 new models (about 4–5 minutes on 1 thread). After that, the script
loads them from code/out/ and finishes in some tens of seconds.
Run: uv run python chapters/10-inference/code/05_gqa.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


tiny = _load("tiny_model", "01_tiny_model.py")
kvc = _load("kv_cache_demo", "03_kv_cache.py")

VARIANTS = [("MHA", 4), ("GQA", 2), ("MQA", 1)]
GEN = 512


def run():
    data = tiny.CharData()
    prompt = data.encode(kvc.PROMPT)
    rows = []
    for name, kv in VARIANTS:
        model = tiny.load_or_train(kv)
        attn = sum(p.numel() for n, p in model.named_parameters() if ".attn." in n)
        total = sum(p.numel() for p in model.parameters())
        out, _, cache = kvc.generate_cached(model, prompt, GEN, temperature=0)
        t = kvc.best_time(lambda m=model: kvc.generate_cached(m, prompt, GEN, temperature=0))
        rows.append(dict(name=name, kv=kv, val=tiny.val_loss(model), attn=attn, total=total,
                         cache=cache.nbytes(), time=t, sample=data.decode(out[:60])))
    noise = tiny.val_loss(tiny.load_or_train(4, seed=1))
    return rows, noise


if __name__ == "__main__":
    rows, noise = run()
    mha = rows[0]
    print(f"4 query heads, head_dim 32, 600 training steps; attention / total = parameter counts; "
          f"KV cache measured after {GEN} generated characters (FP32)")
    print("  scheme KV  val loss  attention     total  KV cache      512 chars (s)")
    for r in rows:
        print(f"  {r['name']}   {r['kv']:3d}   {r['val']:7.3f}   {r['attn']:8,d}  {r['total']:8,d}  "
              f"{r['cache']:9,d} B ({r['cache'] / mha['cache']:.2f}×)   {r['time']:5.2f}")
    print(f"  control: MHA trained again with random seed 1, validation loss {noise:.3f}"
          f" (difference from seed 0: {abs(noise - mha['val']):.3f})")
    for r in rows:
        print(f"  {r['name']} start of the greedy generation: {r['sample']!r}")
