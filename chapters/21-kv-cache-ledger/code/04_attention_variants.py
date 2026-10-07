"""Chapter 21 · Minimal code 4: one small model, five kinds of attention, the same number of steps.

MHA (4 KV heads) / GQA (2) / MQA (1) / MLA (48-dim latent vector, the same cache size as MQA) /
MLA (16-dim latent vector).
All other parts are the same: the character-level Shakespeare model of Chapter 10
(4 layers, width 128, 4 query heads, head_dim 32), the same data order, 600 steps,
AdamW + warmup + cosine. Each variant trains with 3 random seeds. The script reports the mean
and the range of the validation loss. It also reports the cache bytes that it measures
after it generates 512 characters (FP32).

Note: this is a tiny experiment, with about a million parameters and a few minutes of training.
It only shows that the code is correct and what the approximate trend is.
It does not give conclusions for large models. Appendix D of the DeepSeek-V2 paper and
Section 2.1 of the GLM-5 report make careful comparisons.
The first run trains 15 small models (about 20–30 min on one thread). The weights are cached in
code/out/*.pt. After that, the script finishes in some tens of seconds.
Run: uv run python chapters/21-kv-cache-ledger/code/04_attention_variants.py
"""

from __future__ import annotations

import importlib.util
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

torch.set_num_threads(1)
HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
CH10 = HERE.parents[1] / "10-inference" / "code"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


mla_mod = _load("mla_demo", HERE / "03_mla.py")
tiny = mla_mod.tiny
kvc = _load("kv_cache_demo", CH10 / "03_kv_cache.py")

# name, n_kv_heads, MLA latent dimension (None = standard attention)
VARIANTS = [("MHA", 4, None), ("GQA", 2, None), ("MQA", 1, None),
            ("MLA-48", 4, 48), ("MLA-16", 4, 16)]
SEEDS = (0, 1, 2)
ROPE_DIM = 16
GEN = 512


def build(n_kv: int, rank: int | None) -> tiny.TinyLM:
    c = tiny.Config(vocab_size=tiny.CharData().vocab_size, n_kv_heads=n_kv)
    model = tiny.TinyLM(c)
    if rank is not None:  # replace the attention of each layer with MLA: key = nope 32 + rope 16, value 32
        for blk in model.blocks:
            blk.attn = mla_mod.MLA(c.dim, c.n_heads, rank, c.head_dim, ROPE_DIM, c.head_dim,
                                   c.max_seq_len)
    return model


def train(n_kv: int, rank: int | None, seed: int, steps: int = 600, bsz: int = 16,
          seq: int = 64, lr: float = 3e-3):
    """The same training loop as 01_tiny_model.train in Chapter 10. Only `build` makes the model."""
    data = tiny.CharData()
    torch.manual_seed(seed)
    model = build(n_kv, rank)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
    g = torch.Generator().manual_seed(seed)  # same seed → all variants see the same data order
    for step in range(steps + 1):
        for pg in opt.param_groups:  # warmup + cosine
            pg["lr"] = lr * min(1, (step + 1) / 100) * 0.5 * (1 + math.cos(math.pi * step / steps))
        x, y = data.batch("train", bsz, seq, g)
        loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten())
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    return model.eval()


def load_or_train(name: str, n_kv: int, rank: int | None, seed: int):
    path = OUT / f"{name}_s{seed}.pt"
    if path.exists():
        model = build(n_kv, rank)
        model.load_state_dict(torch.load(path, weights_only=True))
        return model.eval()
    t0 = time.time()
    model = train(n_kv, rank, seed)
    OUT.mkdir(exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"  trained {name} seed {seed}: {time.time() - t0:.0f}s", flush=True)
    return model


def run() -> list[dict]:
    res_path = OUT / "results.pt"  # save the results as .pt too (.gitignore ignores it); the video reads this file
    if res_path.exists():
        return torch.load(res_path, weights_only=False)
    data = tiny.CharData()
    prompt = data.encode(kvc.PROMPT)
    rows = []
    for name, n_kv, rank in VARIANTS:
        losses, models = [], []
        for s in SEEDS:
            m = load_or_train(name, n_kv, rank, s)
            losses.append(tiny.val_loss(m))
            models.append(m)
        m = models[0]
        with torch.no_grad():
            out_c, _, cache = kvc.generate_cached(m, prompt, GEN, temperature=0)
            out_n, _ = kvc.generate_naive(m, prompt, 100, temperature=0)  # parity check: cached version = naive version
        attn = sum(p.numel() for n, p in m.named_parameters() if ".attn." in n)
        per_layer = rank + ROPE_DIM if rank else 2 * n_kv * m.c.head_dim
        rows.append(dict(name=name, n_kv=n_kv, rank=rank, losses=losses,
                         mean=sum(losses) / len(losses), per_layer=per_layer,
                         per_token_bytes=per_layer * m.c.n_layers * 4, cache=cache.nbytes(),
                         attn=attn, total=sum(p.numel() for p in m.parameters()),
                         same=out_c[:100] == out_n, sample=data.decode(out_c[:48])))
    torch.save(rows, res_path)
    return rows


if __name__ == "__main__":
    rows = run()
    mha = rows[0]
    print(f"4 layers, 4 query heads, head_dim 32, 600 steps × 3 seeds; "
          f"cache measured after {GEN} generated characters (FP32)")
    print("  Variant  Per layer/pos Per token  Measured KV cache   Attn params    Mean"
          "  [loss of 3 seeds]       Cached=naive")
    for r in rows:
        ls = " ".join(f"{x:.3f}" for x in r["losses"])
        print(f"  {r['name']:7} {r['per_layer']:6d} numbers  {r['per_token_bytes']:6,d} B  "
              f"{r['cache']:9,d} B ({r['cache'] / mha['cache']:.3f}×)  {r['attn']:9,d}   "
              f"{r['mean']:.3f}  [{ls}]   {r['same']}")
    spread = max(max(r["losses"]) - min(r["losses"]) for r in rows)
    print(f"  With another seed, the loss of one variant changes by up to {spread:.3f}. "
          "Do not trust a difference between variant means that is smaller than this.")
    for r in rows:
        print(f"  {r['name']:7} start of greedy output: {r['sample']!r}")
