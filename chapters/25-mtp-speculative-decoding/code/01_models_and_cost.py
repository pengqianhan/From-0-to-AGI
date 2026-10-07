"""Chapter 25 · Minimal code 1: the target model, the draft model, and "verifying k tokens
costs the same as generating 1".

- Target model: reuse the small model that Chapter 10 trained (4 layers, width 128,
  65 characters at the character level, 0.86M parameters).
- Draft model: the same structure and **the same tokenizer** (the same 65 characters), but only
  1 layer and width 64. We train it on the same corpus. Speculative decoding requires that the two
  vocabularies are identical: the target model must score the token ids from the draft directly.

Then the script measures one thing: the KV cache already exists, and the target model feeds
T new tokens in one forward pass. How long does that take?
In the decode phase, the model feeds only 1 token at a time, and the compute units are not fully
used (Chapters 10 and 21). Thus feeding 1 token and feeding 5 tokens take about the same time.
This is the physical condition that makes speculative decoding work.

The weights are cached in code/out/*.pt (.gitignore ignores them). If the Chapter 10 weights do not
exist, the script trains them first.
Run: uv run python chapters/25-mtp-speculative-decoding/code/01_models_and_cost.py
"""

from __future__ import annotations

import importlib.util
import statistics
import sys
import time
from pathlib import Path

import torch

torch.set_num_threads(1)  # Many jobs share the build machine; 1 thread is more stable. You can remove this.
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = HERE / "out"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # A dataclass must find its own module in sys.modules.
    spec.loader.exec_module(mod)
    return mod


# Reuse TinyLM / Config / CharData / KVCache / train from Chapter 10 without changes.
ch10 = _load("ch10_tiny_model", ROOT / "chapters" / "10-inference" / "code" / "01_tiny_model.py")
KVCache = ch10.KVCache
DRAFT_CFG = dict(dim=64, n_layers=1, n_heads=2, n_kv_heads=2, ffn_dim=192)


def load_target():
    """The 4-layer small model from Chapter 10 (MHA, 600 steps)."""
    return ch10.load_or_train(n_kv_heads=4, steps=600, verbose=True)


def load_draft(steps: int = 1500):
    """The draft model with 1 layer and width 64: the same character-level data, the same vocabulary."""
    c = ch10.Config(vocab_size=ch10.CharData().vocab_size, **DRAFT_CFG)
    path = OUT / f"draft_1x64_s{steps}.pt"
    model = ch10.TinyLM(c)
    if path.exists():
        model.load_state_dict(torch.load(path, weights_only=True))
        return model.eval()
    print(f"Training the draft model ({steps} steps, only once; later runs load {path.name})")
    model = ch10.train(c, steps=steps, seed=0, verbose=True)
    OUT.mkdir(exist_ok=True)
    torch.save(model.state_dict(), path)
    return model


def truncate(cache: KVCache, n: int) -> None:
    """Roll back the KV cache: keep only the first n positions (discard the K/V of rejected drafts)."""
    for layer in range(len(cache.k)):
        if cache.k[layer] is not None:
            cache.k[layer] = cache.k[layer][:, :, :n]
            cache.v[layer] = cache.v[layer][:, :, :n]


@torch.no_grad()
def forward_time(model, ctx_len: int, n_new: int, reps: int = 60) -> float:
    """Time of one forward pass that feeds n_new new tokens, with ctx_len positions in the cache
    (milliseconds, median).

    We use time.process_time() (the CPU time of this process), not the wall-clock time. On the build
    machine, tens of jobs share 4 cores, so the wall-clock time shows mostly the wait for a CPU.
    The CPU time shows the work of the forward pass itself (with 1 thread, the two should be equal)."""
    g = torch.Generator().manual_seed(0)
    cache = KVCache(model.c.n_layers)
    model(torch.randint(0, model.c.vocab_size, (1, ctx_len), generator=g), cache)
    new = torch.randint(0, model.c.vocab_size, (1, n_new), generator=g)
    times = []
    for _ in range(reps):
        t0 = time.process_time()
        model(new, cache)
        times.append(time.process_time() - t0)
        truncate(cache, ctx_len)  # Go back to the same start point each time.
    return statistics.median(times) * 1e3


def n_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    target, draft = load_target(), load_draft()
    data = ch10.CharData()
    print(f"Target model: {n_params(target):,} parameters, validation loss {ch10.val_loss(target):.3f}")
    print(f"Draft model:  {n_params(draft):,} parameters, validation loss {ch10.val_loss(draft):.3f}")
    print(
        f"Parameter ratio {n_params(target) / n_params(draft):.0f} : 1; both use the same vocabulary of {data.vocab_size} characters"
    )

    print(
        "\nKV cache with 200 positions; time of one forward pass that feeds T new tokens (1 thread, process CPU time, median):"
    )
    print(f"{'T':>4} {'target ms':>12} {'vs T=1':>9} {'draft ms':>12}")
    base = None
    for T in (1, 2, 3, 5, 9, 17):
        # Measure the two models in turns: this reduces the effect of load changes on the comparison.
        tt = forward_time(target, 200, T)
        td = forward_time(draft, 200, T)
        base = base or tt
        print(f"{T:4d} {tt:12.2f} {tt / base:8.2f}× {td:12.2f}")
    c = forward_time(draft, 200, 1) / forward_time(target, 200, 1)
    print(f"\nCost coefficient c = draft step / target step ≈ {c:.2f}")
    print("(Many jobs share this machine, so the times change from run to run. Look only at the order of magnitude and the trend.)")
