"""Chapter 24 · Minimal code 3: the same small language model with an MoE FFN, compared with dense FFNs.

Model: the character-level Shakespeare model of Chapter 10 (4 layers, width 128, 4 heads,
Pre-Norm RMSNorm + RoPE). Only the FFN changes:

  稠密-384       (Dense-384): SwiGLU, width 384      — the baseline for active parameters (≈ compute per token)
  稠密-1536      (Dense-1536): SwiGLU, width 1536    — about the same total parameters as the MoE
  MoE-无均衡     (MoE-no-balance): 8 experts × width 192, top 2 (active width 384), no load balancing
  MoE-辅助损失   (MoE-aux-loss): the same + L_aux, α = 0.01
  MoE-无辅助损失 (MoE-aux-free): the same + bias balancing, γ = 0.01
  细粒度+共享    (Fine-grained+shared): 16 experts × width 96, top 3, + 1 shared expert of width 96
                 (active width 384), bias balancing

The variant names stay in Chinese: they are keys in the result cache, and the video reads them.
The script prints the English names in parentheses (VARIANT_EN).
All MoE variants use sigmoid scores + top-k normalization. This is the DeepSeek-V3 method. Its
ablation also compares the two balancing methods with this setting.
All variants use the same data order, the same number of steps, and AdamW + warmup + cosine.
Each variant runs with 2 random seeds. During training, we record the expert load of each layer every 50 steps.

Note: this is a tiny experiment with about a million parameters and a few minutes of training.
It only shows that the code is correct and approximately what the effects look like.
It does not support conclusions about large models. The serious comparisons are
Table 5 of the DeepSeek-V3 report and Figure 5 of the Kimi K2 report.
The first run trains 12 small models (about 30–40 minutes on one thread).
The results are cached in code/out/moe_runs/.
To train in several parallel processes, give variant names: python 03_train_compare.py 稠密-384 MoE-无均衡 …
Then run the script without arguments to get the summary.
Run: uv run python chapters/24-mixture-of-experts/code/03_train_compare.py
"""

from __future__ import annotations

import importlib.util
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(1)
HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
CH10 = HERE.parents[1] / "10-inference" / "code" / "01_tiny_model.py"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


tiny = _load("tiny_ch10", CH10)          # the data, attention, RMSNorm, and RoPE come from Chapter 10
moe_mod = _load("moe_ch24", HERE / "02_moe_layer.py")

# name → FFN constructor arguments. The names stay in Chinese: they are cache keys, and the video reads them.
VARIANTS = {
    "稠密-384": dict(kind="dense", hidden=384),
    "稠密-1536": dict(kind="dense", hidden=1536),
    "MoE-无均衡": dict(kind="moe", E=8, K=2, hidden=192, balance="none"),
    "MoE-辅助损失": dict(kind="moe", E=8, K=2, hidden=192, balance="aux"),
    "MoE-无辅助损失": dict(kind="moe", E=8, K=2, hidden=192, balance="free"),
    "细粒度+共享": dict(kind="moe", E=16, K=3, hidden=96, shared=1, balance="free"),
}
# Display names for the printed output
VARIANT_EN = {
    "稠密-384": "Dense-384",
    "稠密-1536": "Dense-1536",
    "MoE-无均衡": "MoE-no-balance",
    "MoE-辅助损失": "MoE-aux-loss",
    "MoE-无辅助损失": "MoE-aux-free",
    "细粒度+共享": "Fine-grained+shared",
}
SEEDS = (0, 1)
STEPS = 800
LOG_EVERY = 50
SHOW_LAYER = 1  # the layer that the printout and the video show


def make_ffn(v: dict, dim: int) -> nn.Module:
    if v["kind"] == "dense":
        return moe_mod.Expert(dim, v["hidden"])
    return moe_mod.MoE(dim, v["E"], v["K"], v["hidden"], n_shared=v.get("shared", 0),
                       score="sigmoid", balance=v["balance"], aux_coef=0.01, bias_speed=0.01)


class Block(nn.Module):
    def __init__(self, c, ffn: nn.Module) -> None:
        super().__init__()
        self.n1, self.attn = tiny.RMSNorm(c.dim), tiny.Attention(c)
        self.n2, self.ffn = tiny.RMSNorm(c.dim), ffn

    def forward(self, x, cos, sin):
        x = x + self.attn(self.n1(x), cos, sin, None, 0)
        return x + self.ffn(self.n2(x))


class LM(nn.Module):
    def __init__(self, v: dict) -> None:
        super().__init__()
        c = self.c = tiny.Config(vocab_size=tiny.CharData().vocab_size)
        self.emb = nn.Embedding(c.vocab_size, c.dim)
        nn.init.normal_(self.emb.weight, std=0.02)
        self.blocks = nn.ModuleList(Block(c, make_ffn(v, c.dim)) for _ in range(c.n_layers))
        self.norm = tiny.RMSNorm(c.dim)
        cos, sin = tiny.rope_tables(c.head_dim, c.max_seq_len)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, ids):
        T = ids.shape[1]
        x = self.emb(ids)
        for b in self.blocks:
            x = b(x, self.cos[:T], self.sin[:T])
        return self.norm(x) @ self.emb.weight.T

    def moes(self):
        return [b.ffn for b in self.blocks if isinstance(b.ffn, moe_mod.MoE)]


def ffn_params(model: LM) -> tuple[int, int]:
    """(FFN total parameters, FFN active parameters), summed over the 4 layers."""
    tot = act = 0
    for b in model.blocks:
        f = b.ffn
        n = moe_mod.n_params(f)
        tot += n
        if isinstance(f, moe_mod.MoE):
            n -= (f.E - f.K) * moe_mod.n_params(f.experts[0])
        act += n
    return tot, act


@torch.no_grad()
def val_loss(model: LM, n_batches: int = 20, seq: int = 64) -> float:
    data = tiny.CharData()
    g = torch.Generator().manual_seed(1234)
    model.eval()
    tot = 0.0
    for _ in range(n_batches):
        x, y = data.batch("val", 32, seq, g)
        tot += F.cross_entropy(model(x).flatten(0, 1), y.flatten()).item()
    model.train()
    return tot / n_batches


def train(name: str, seed: int, steps: int = STEPS, bsz: int = 16, seq: int = 64,
          lr: float = 3e-3) -> dict:
    v = VARIANTS[name]
    data = tiny.CharData()
    torch.manual_seed(seed)
    model = LM(v)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
    g = torch.Generator().manual_seed(seed)  # same seed → all variants see the same data order
    loads, curve = [], []
    t0 = time.time()
    for step in range(steps + 1):
        for pg in opt.param_groups:  # warmup + cosine (Chapter 6)
            pg["lr"] = lr * min(1, (step + 1) / 100) * 0.5 * (1 + math.cos(math.pi * step / steps))
        x, y = data.batch("train", bsz, seq, g)
        ce = F.cross_entropy(model(x).flatten(0, 1), y.flatten())
        aux = sum((m.aux for m in model.moes()), torch.zeros(()))
        opt.zero_grad()
        (ce + aux).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        for m in model.moes():
            if m.balance == "free":
                m.update_bias()          # aux-loss-free balancing: after each step, use the load of this step
        if step % LOG_EVERY == 0 and model.moes():
            loads.append([(m.load / m.load.sum()).tolist() for m in model.moes()])
        if step % 100 == 0:
            curve.append((step, ce.item()))
    tot, act = ffn_params(model)
    return dict(name=name, seed=seed, val=val_loss(model), loads=loads, curve=curve,
                ffn_total=tot, ffn_active=act, total=moe_mod.n_params(model),
                secs=time.time() - t0)


def run(only: list[str] | None = None) -> list[dict]:
    """Train each (variant, seed) pair once and save the result as code/out/moe_runs/*.pt
    (.gitignore ignores *.pt). After that, read the cache. If `only` gives variant names, train only
    these. You can run several processes in parallel, then run once without arguments for the summary."""
    runs = OUT / "moe_runs"
    runs.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in SEEDS:
        for i, name in enumerate(VARIANTS):
            path = runs / f"v{i}_s{seed}.pt"
            if path.exists():
                rows.append(torch.load(path, weights_only=False))
                continue
            if only is not None and name not in only:
                continue
            r = train(name, seed)
            print(f"  trained {VARIANT_EN[name]} seed {seed}: {r['secs']:.0f}s, validation loss {r['val']:.3f}", flush=True)
            torch.save(r, path)
            rows.append(r)
    return rows


def imbalance(frac: list[float]) -> float:
    """Load of the busiest expert ÷ mean load (1 = perfect balance; with E experts, the maximum is E/K)."""
    return max(frac) * len(frac)


def summarize(rows: list[dict]) -> list[dict]:
    out = []
    for name in VARIANTS:
        rs = [r for r in rows if r["name"] == name]
        vals = [r["val"] for r in rs]
        s = dict(name=name, vals=vals, mean=sum(vals) / len(vals), ffn_total=rs[0]["ffn_total"],
                 ffn_active=rs[0]["ffn_active"], total=rs[0]["total"])
        if rs[0]["loads"]:
            # the last recorded load: "max/mean" of each layer; take the mean and the max over 4 layers × 2 seeds
            ims = [imbalance(layer) for r in rs for layer in r["loads"][-1]]
            s["imb_mean"], s["imb_max"] = sum(ims) / len(ims), max(ims)
            dead = [sum(v < 0.01 for v in layer) for r in rs for layer in r["loads"][-1]]
            s["dead"] = max(dead)  # number of experts with a load < 1% (the maximum in one layer)
        out.append(s)
    return out


if __name__ == "__main__":
    only = sys.argv[1:] or None
    rows = run(only)
    if only is not None:
        sys.exit(0)
    print(f"Character-level language model, 4 layers, width 128; {STEPS} steps × {len(SEEDS)} seeds; "
          f"FFN parameters are summed over the 4 layers")
    print(f"  {'variant':20}{'FFN total':>11}{'FFN act.':>10}{'model':>10}   val loss mean [seeds]"
          f"  load max/mean (mean, worst)  idle experts")
    for s in summarize(rows):
        vals = " ".join(f"{v:.3f}" for v in s["vals"])
        load = (f"{s['imb_mean']:.2f}, {s['imb_max']:.2f}" if "imb_mean" in s else "  —")
        dead = f"{s['dead']}" if "dead" in s else "—"
        print(f"  {VARIANT_EN.get(s['name'], s['name']):20}{s['ffn_total']:11,d}{s['ffn_active']:10,d}{s['total']:10,d}"
              f"   {s['mean']:.3f} [{vals}]      {load:>12}              {dead}")
    spread = max(max(s["vals"]) - min(s["vals"]) for s in summarize(rows))
    print(f"  Same variant, other seed: the validation loss differs by up to {spread:.3f}. "
          f"Do not trust a smaller difference.")
    print(f"\n  \"max/mean\" load of each layer: start of training (step 0) → end (step {STEPS}), seed 0")
    for name in ("MoE-无均衡", "MoE-辅助损失", "MoE-无辅助损失"):
        r = next(r for r in rows if r["name"] == name and r["seed"] == 0)
        pairs = "  ".join(f"layer {i} {imbalance(a):.2f}→{imbalance(b):.2f}"
                          for i, (a, b) in enumerate(zip(r["loads"][0], r["loads"][-1])))
        print(f"  {VARIANT_EN[name]:16}{pairs}")
    print(f"\n  Expert load share in layer {SHOW_LAYER} (counted from 0) during training "
          f"(seed 0; uniform = 0.125):")
    for name in ("MoE-无均衡", "MoE-辅助损失", "MoE-无辅助损失"):
        r = next(r for r in rows if r["name"] == name and r["seed"] == 0)
        print(f"  {VARIANT_EN[name]}")
        for i in (0, 2, 6, len(r["loads"]) - 1):
            frac = " ".join(f"{v:.2f}" for v in r["loads"][i][SHOW_LAYER])
            print(f"    step {i * LOG_EVERY:4d}: [{frac}]  max/mean "
                  f"{imbalance(r['loads'][i][SHOW_LAYER]):.2f}")
