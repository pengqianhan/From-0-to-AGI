"""Chapter 15 · Minimal code 3: a small model trained at length 64. Can it read 128 and 256?

1. Train the minimal Transformer of Chapter 9 (byte level, head_dim 32, θ = 10K) on Shakespeare,
   with sequences of length 64 only.
2. Do not train more. Calculate the loss on validation sequences of length 64 / 128 / 256 with
   four RoPE settings:
     (a) change nothing    (b) position interpolation PI (÷4)
     (c) YaRN (s = 4)      (d) larger base frequency (θ = 100K)
3. Fine-tune each setting for a short time on sequences of length 256 (150 steps, about 1/10 of
   pretraining). Then evaluate again.

All settings change only the RoPE cos/sin. No learnable parameter changes. In evaluation, one model
uses the same cos/sin for all lengths ("static scaling").
Run: uv run python chapters/15-midtraining-long-context/code/03_context_extension.py
     (about 11 min of CPU time on one thread; the wall time is longer on a busy machine. The results
     are cached in code/out/context_extension.pt, and the video reads this file. Add --fresh to run again.)
"""

import argparse
import copy
import importlib.util
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

torch.set_num_threads(1)

HERE = Path(__file__).resolve().parent
CACHE = HERE / "out" / "context_extension.pt"
_spec = importlib.util.spec_from_file_location(
    "tiny_transformer", HERE.parents[1] / "09-modern-transformer/code/02_tiny_transformer.py")
tt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tt)
_spec2 = importlib.util.spec_from_file_location("yarn", HERE / "02_yarn_from_scratch.py")
yarn = importlib.util.module_from_spec(_spec2)
_spec2.loader.exec_module(yarn)

TRAIN_LEN, MAX_LEN, S = 64, 256, 4.0
EVAL_LENS = (64, 128, 256)
BUCKETS = ((0, 64), (64, 128), (128, 256))
HEAD_DIM, THETA, ABF_THETA = 32, 10_000.0, 100_000.0


def rope_tables(variant: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the cos/sin tables of length MAX_LEN. mscale multiplies cos/sin directly (as in zero),
    so q·k becomes mscale² times larger."""
    mscale = 1.0
    if variant == "none":
        w = yarn.rope_inv_freq(HEAD_DIM, THETA)
    elif variant == "pi":
        w = yarn.pi_inv_freq(HEAD_DIM, THETA, S)
    elif variant == "yarn":
        w, _, mscale = yarn.yarn_inv_freq(HEAD_DIM, THETA, S, TRAIN_LEN)
    elif variant == "abf":
        w = yarn.rope_inv_freq(HEAD_DIM, ABF_THETA)
    else:
        raise ValueError(variant)
    angles = torch.outer(torch.arange(MAX_LEN, dtype=torch.float32), w)
    angles = torch.cat([angles, angles], dim=-1)
    return angles.cos() * mscale, angles.sin() * mscale


def set_rope(model, variant: str):
    model.cos, model.sin = rope_tables(variant)
    return model


def batches(data, seq_len, batch_size, steps, seed):
    g = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        ix = torch.randint(len(data) - seq_len - 1, (batch_size,), generator=g)
        x = torch.stack([data[i:i + seq_len] for i in ix])
        y = torch.stack([data[i + 1:i + seq_len + 1] for i in ix])
        yield x, y


def train(model, data, seq_len, batch_size, steps, lr, warmup, seed, log_every=0):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
    curve = []
    model.train()
    for step, (x, y) in enumerate(batches(data, seq_len, batch_size, steps, seed)):
        # warmup + cosine decay to 10% (Chapter 6)
        f = min(1.0, (step + 1) / warmup) * (0.1 + 0.45 * (1 + math.cos(math.pi * step / steps)))
        for gr in opt.param_groups:
            gr["lr"] = lr * f
        loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten())
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        curve.append(loss.item())
        if log_every and (step + 1) % log_every == 0:
            print(f"   step {step + 1:5d} | train loss {sum(curve[-log_every:]) / log_every:.3f}")
    model.eval()
    return curve


@torch.no_grad()
def per_position_loss(model, val, n_windows=256, seed=7) -> torch.Tensor:
    """Mean loss at each position (nat/byte) on n_windows validation sequences of length MAX_LEN.
    Shape (MAX_LEN,). With causal attention, the prediction at position p sees only [0, p].
    Thus "the loss at length L" is the mean of the first L positions."""
    g = torch.Generator().manual_seed(seed)
    ix = torch.randint(len(val) - MAX_LEN - 1, (n_windows,), generator=g)
    total = torch.zeros(MAX_LEN)
    for chunk in ix.split(32):
        x = torch.stack([val[i:i + MAX_LEN] for i in chunk])
        y = torch.stack([val[i + 1:i + MAX_LEN + 1] for i in chunk])
        loss = F.cross_entropy(model(x).transpose(1, 2), y, reduction="none")  # (B, T)
        total += loss.sum(0)
    return total / n_windows


def summarize(pos_loss: torch.Tensor) -> dict:
    return {
        "len": {L: pos_loss[:L].mean().item() for L in EVAL_LENS},
        "bucket": {f"{a}-{b}": pos_loss[a:b].mean().item() for a, b in BUCKETS},
        "curve": pos_loss.tolist(),
    }


def run(fresh: bool = False, pre_steps: int = 1500, ft_steps: int = 150) -> dict:
    if CACHE.exists() and not fresh:
        return torch.load(CACHE, weights_only=False)
    torch.manual_seed(1337)
    train_data, val_data = tt.load_data()
    cfg = tt.Config(seq_len=MAX_LEN)            # cos/sin up to 256; training uses only the first 64 positions
    base = tt.TinyTransformer(cfg)
    set_rope(base, "none")
    t0 = time.time()
    print(f"Pretraining: length {TRAIN_LEN}, {pre_steps} steps, batch 32")
    pre_curve = train(base, train_data, TRAIN_LEN, 32, pre_steps, lr=3e-3, warmup=100, seed=1, log_every=300)
    print(f"   time {time.time() - t0:.0f}s")

    results = {"zero_shot": {}, "finetuned": {}, "ft_curves": {}}
    variants = ["none", "pi", "yarn", "abf"]
    for v in variants:
        results["zero_shot"][v] = summarize(per_position_loss(set_rope(base, v), val_data))
    for v in variants:
        t0 = time.time()
        m = set_rope(copy.deepcopy(base), v)
        # All fine-tuning runs use the same data order (seed=2). Only the RoPE setting is different.
        results["ft_curves"][v] = train(m, train_data, MAX_LEN, 8, ft_steps, lr=1e-3, warmup=10, seed=2)
        results["finetuned"][v] = summarize(per_position_loss(m, val_data))
        print(f"   fine-tune {v:<5} time {time.time() - t0:.0f}s")
    set_rope(base, "none")
    results.update(pre_curve=pre_curve, pre_steps=pre_steps, ft_steps=ft_steps,
                   tokens_pre=pre_steps * 32 * TRAIN_LEN, tokens_ft=ft_steps * 8 * MAX_LEN)
    CACHE.parent.mkdir(exist_ok=True)
    torch.save(results, CACHE)
    return results


NAMES = {"none": "(a) no change", "pi": "(b) PI (÷4)", "yarn": "(c) YaRN", "abf": "(d) ABF θ=100K"}


def report(r: dict) -> None:
    print(f"\nPretraining {r['tokens_pre']:,} tokens (length {TRAIN_LEN}); fine-tuning {r['tokens_ft']:,} tokens for each setting (length {MAX_LEN})")
    for key, title in [("zero_shot", "No training, only a new RoPE"), ("finetuned", f"After {r['ft_steps']} fine-tuning steps at length {MAX_LEN}")]:
        print(f"\n{title}: validation loss (nat/byte, lower is better)")
        print(f"   {'setting':<16}" + "".join(f"{'L=' + str(L):>9}" for L in EVAL_LENS)
              + "".join(f"{'pos ' + k:>12}" for k in r[key]["none"]["bucket"]))
        for v, name in NAMES.items():
            s = r[key][v]
            print(f"   {name:<16}" + "".join(f"{s['len'][L]:>9.3f}" for L in EVAL_LENS)
                  + "".join(f"{x:>12.3f}" for x in s["bucket"].values()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fresh", action="store_true", help="ignore the cache and train again")
    args = ap.parse_args()
    report(run(fresh=args.fresh))
