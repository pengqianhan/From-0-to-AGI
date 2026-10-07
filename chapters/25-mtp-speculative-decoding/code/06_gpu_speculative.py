"""Chapter 25 · GPU measurements: decode waits for data, so "verify k tokens at once" is almost free.
How much faster is speculative decoding on a GPU?

All times in the main text come from a CPU with 1 thread. This script does the same experiments on
a GPU (CUDA is necessary):
  1. The KV cache has 200 positions. How long does one forward pass of the target take for T new tokens?
     - The small target model of this chapter (0.86M parameters, FP32).
     - The same TinyLM structure made larger, to about 4.1B parameters (width 8192, 6 layers, BF16):
       the "scaled-up target". One read of its weights is 8.3 GB, so it is in the
       bandwidth-limited region of decode. The scaled-up model has random initialization. We measure
       only time here, and the time of a forward pass does not depend on the values of the weights.
  2. The trained target + draft of this chapter run greedy speculative decoding on the GPU
     (4 prompts × 200 characters, with speculative_greedy from 02). Are the outputs still
     token-for-token identical on the GPU? The script also records how many drafts each round accepts.
  3. Scaled-up target + scaled-up draft (1 layer, width 1024, about 1/300 of the target): use the real
     number of accepted drafts of each round from part 2, and "replay" all forward passes,
     synchronizations, and rollbacks of speculative decoding. Measure the wall-clock speedup and
     compare it with the formula (1 − α^{k+1}) / ((1 − α)(1 + k·c)). (Two models with random weights
     cannot "guess well" or "guess badly", so the number accepted is copied from the round of the real
     models. To limit the total time, we replay only the first 2 prompts.)

The Chapter 10 attention builds the causal mask with torch.arange on the CPU. Here we change only
this one place so that the mask is built on the GPU. We reuse all other code without changes.
Run: uv run python chapters/25-mtp-speculative-decoding/code/06_gpu_speculative.py
(Run 01 first, so that the target / draft weights are cached in out/. Then this script takes about
1–2 minutes and needs about 10 GB of GPU memory.)
"""

from __future__ import annotations

import importlib.util
import math
import statistics
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
DEV = "cuda"
BIG_TARGET = dict(dim=8192, n_layers=6, n_heads=64, n_kv_heads=8, ffn_dim=22016)
BIG_DRAFT = dict(dim=1024, n_layers=1, n_heads=8, n_kv_heads=8, ffn_dim=2816)
KS = (1, 2, 3, 4, 6, 8)


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def patch_attention(ch10) -> None:
    """An exact copy of the Chapter 10 Attention.forward. Only the arange of the causal mask is built on the device of x."""

    def forward(self, x, cos, sin, cache, layer):
        B, T, _ = x.shape
        c = self.c
        q = self.wq(x).view(B, T, c.n_heads, c.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, c.n_kv_heads, c.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, c.n_kv_heads, c.head_dim).transpose(1, 2)
        q, k = ch10.apply_rope(q, cos, sin), ch10.apply_rope(k, cos, sin)
        if cache is not None:
            k, v = cache.append(layer, k, v)
        S = k.shape[2]
        g = c.n_heads // c.n_kv_heads
        k, v = k.repeat_interleave(g, dim=1), v.repeat_interleave(g, dim=1)
        att = q @ k.transpose(-2, -1) / math.sqrt(c.head_dim)
        i = torch.arange(T, device=x.device)[:, None] + (S - T)  # ← The only change: device=x.device
        j = torch.arange(S, device=x.device)[None, :]
        att = att.masked_fill(j > i, float("-inf")).softmax(-1)
        return self.wo((att @ v).transpose(1, 2).reshape(B, T, -1))

    ch10.Attention.forward = forward


class OnGPU(torch.nn.Module):
    """Move the token ids from the CPU to the GPU, then run the forward pass.

    Then greedy_generate / speculative_greedy from 02 work without changes."""

    def __init__(self, model) -> None:
        super().__init__()
        self.model, self.c = model, model.c

    def forward(self, ids, cache=None):
        return self.model(ids.to(DEV), cache)


def build_big(ch10, vocab_size: int, cfg: dict):
    """Build the scaled-up model directly on the GPU and directly in BF16.

    With 4B parameters, building in FP32 first and converting after that uses more than 24 GB of GPU memory."""
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    with torch.device(DEV):
        model = ch10.TinyLM(ch10.Config(vocab_size=vocab_size, **cfg))
    torch.set_default_dtype(old)
    return model.to(torch.bfloat16).eval()  # Convert the RoPE tables to BF16 too: the same precision as the activations


def n_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


def n_bytes(model) -> int:
    return sum(p.numel() * p.element_size() for p in model.parameters())


def launch_us(n: int = 2000) -> float:
    """Microseconds for Python to launch one minimal GPU operation on this machine.

    The operation is an addition on one element, so the GPU itself uses almost no time."""
    x = torch.zeros(1, device=DEV)
    for _ in range(100):
        x.add_(1)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        x.add_(1)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1e6


@torch.no_grad()
def forward_ms(model, ctx: int, n_new: int, reps: int = 30) -> float:
    """Time of one forward pass that feeds n_new new tokens, with ctx positions in the cache
    (milliseconds, CUDA event timing, median)."""
    g = torch.Generator(device=DEV).manual_seed(0)
    cache = m1.KVCache(model.c.n_layers)
    model(torch.randint(0, model.c.vocab_size, (1, ctx), generator=g, device=DEV), cache)
    new = torch.randint(0, model.c.vocab_size, (1, n_new), generator=g, device=DEV)
    times = []
    for i in range(reps + 5):  # The first 5 runs are warmup
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        model(new, cache)
        end.record()
        torch.cuda.synchronize()
        if i >= 5:
            times.append(start.elapsed_time(end))
        m1.truncate(cache, ctx)  # Go back to the same start point each time
    return statistics.median(times)


def wall(fn, *args):
    """Wall-clock time (seconds). Wait for the GPU to finish before and after."""
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn(*args)
    torch.cuda.synchronize()
    return time.perf_counter() - t0, out


@torch.no_grad()
def replay(target, draft, prompt: list[int], k: int, accepted: list[int]) -> int:
    """Replay each step of speculative_greedy from 02 with the given number accepted per round.

    The steps: the draft feeds the missing tokens and guesses k tokens; the target verifies them in one
    forward pass; take the argmax (this also waits for the GPU); roll back the two caches.
    Only the number accepted does not come from a comparison: the function copies it."""
    seq = list(prompt)
    tc, dc = m1.KVCache(target.c.n_layers), m1.KVCache(draft.c.n_layers)
    for m in accepted:
        logits = draft(torch.tensor([seq[len(dc) :]]), dc)[0, -1]
        drafts = []
        for i in range(k):
            drafts.append(int(logits.argmax()))
            if i < k - 1:
                logits = draft(torch.tensor([[drafts[-1]]]), dc)[0, -1]
        choice = (
            target(torch.tensor([seq[len(tc) :] + drafts]), tc)[0, -(k + 1) :].argmax(-1).tolist()
        )
        seq += drafts[:m] + [choice[m]]
        m1.truncate(tc, len(seq) - 1)
        m1.truncate(dc, min(len(dc), len(seq) - 1))
    return len(seq) - len(prompt)


def greedy_all(model, P, N):
    return [m2.greedy_generate(model, p, N) for p in P]


def replay_all(target, draft, P, k, traces_k):
    return sum(replay(target, draft, p, k, tr) for p, tr in zip(P, traces_k))


def expected_tokens(alpha: float, k: int) -> float:
    return (1 - alpha ** (k + 1)) / (1 - alpha)


def alpha_of(traces_k: list[list[int]], k: int) -> float:
    """Per-token acceptance rate = number accepted ÷ number of drafts that were compared (the same as in 02)."""
    acc = sum(m for tr in traces_k for m in tr)
    examined = sum(m + (1 if m < k else 0) for tr in traces_k for m in tr)
    return acc / examined


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("This script needs a CUDA GPU. If you do not have a GPU, skip it: the main text shows one run on an RTX 3090.")
        sys.exit(0)
    t_start = time.perf_counter()
    torch.manual_seed(0)
    m1 = _load("ch25_models", "01_models_and_cost.py")
    m2 = _load("ch25_greedy", "02_greedy_speculative.py")
    ch10 = m1.ch10
    patch_attention(ch10)
    print(
        f"GPU: {torch.cuda.get_device_name(0)}, PyTorch {torch.__version__}, CUDA {torch.version.cuda}"
    )

    target, draft = m1.load_target().to(DEV), m1.load_draft().to(DEV)  # Weights from the cache of 01 (FP32)
    big_t = build_big(ch10, target.c.vocab_size, BIG_TARGET)
    big_d = build_big(ch10, target.c.vocab_size, BIG_DRAFT)
    print(
        f"Target of this chapter {n_params(target) / 1e6:.2f}M parameters (FP32, {n_bytes(target) / 1e6:.1f} MB); "
        f"scaled-up target {n_params(big_t) / 1e9:.2f}B parameters (BF16, {n_bytes(big_t) / 1e9:.2f} GB), "
        f"scaled-up draft {n_params(big_d) / 1e6:.1f}M parameters (1/{n_params(big_t) / n_params(big_d):.0f} of the target)"
    )
    print(f"On this machine, Python needs about {launch_us():.1f} µs to launch one minimal GPU operation")

    # ── 1. One forward pass that feeds T tokens ──
    print(
        "\n── 1. KV cache with 200 positions: time of one forward pass that feeds T new tokens (CUDA event, median of 30) ──"
    )
    print(f"{'T':>4} {'target ms':>11} {'vs T=1':>8} {'big target ms':>13} {'vs T=1':>8}")
    base_s = base_b = None
    for T in (1, 2, 4, 8, 16, 64, 256):
        ts, tb = forward_ms(target, 200, T), forward_ms(big_t, 200, T)
        base_s, base_b = base_s or ts, base_b or tb
        print(f"{T:4d} {ts:11.3f} {ts / base_s:7.2f}× {tb:13.3f} {tb / base_b:7.2f}×")
    print(
        f"Scaled-up T=1: {n_bytes(big_t) / 1e9:.2f} GB of weights / {base_b:.2f} ms = effective bandwidth "
        f"{n_bytes(big_t) / base_b / 1e6:.0f} GB/s; target of this chapter T=1: {n_bytes(target) / 1e6:.1f} MB / "
        f"{base_s:.2f} ms = {n_bytes(target) / base_s / 1e6:.1f} GB/s"
    )

    # ── 2. Target + draft of this chapter: real speculative decoding on the GPU (correctness + number accepted per round) ──
    P, N = m2.prompts(), 200
    T_, D_ = OnGPU(target), OnGPU(draft)
    base_out = greedy_all(T_, P, N)
    print(f"\n── 2. Target + draft of this chapter on the GPU (FP32, greedy, 4 prompts × {N} characters) ──")
    print(f"{'k':>2} {'identical outputs':>12} {' accept α':>7} {' tok/round':>7} {'forward passes':>9}")
    traces = {}
    for k in KS:
        trace: list = []
        res = [m2.speculative_greedy(T_, D_, p, N, k, trace) for p in P]
        same = all(r[0] == b for r, b in zip(res, base_out))
        # trace joins the prompts in order: cut it by the number of rounds of each prompt to get the counts per round
        cuts = [0]
        for r in res:
            cuts.append(cuts[-1] + r[1]["rounds"])
        traces[k] = [[m for _, _, m in trace[a:b]] for a, b in zip(cuts, cuts[1:])]
        rounds = cuts[-1]
        print(
            f"{k:2d} {str(same):>17} {alpha_of(traces[k], k):9.3f} "
            f"{(sum(map(sum, traces[k])) + rounds) / rounds:10.2f} "
            f"{rounds:14d}"
        )

    # ── 3. Scaled-up models: replay with the real number accepted per round ──
    BT, BD = OnGPU(big_t), OnGPU(big_d)
    P3 = P[:2]
    c_big = forward_ms(big_d, 200, 1) / forward_ms(big_t, 200, 1)
    replay(BT, BD, P3[0], 3, traces[3][0][:5])  # Warmup
    tb_big, ts_big = [], {k: [] for k in KS}
    for _ in range(3):
        tb_big.append(wall(greedy_all, BT, P3, N)[0])
        for k in KS:
            ts_big[k].append(wall(replay_all, BT, BD, P3, k, traces[k][: len(P3)])[0])
    tbb = statistics.median(tb_big)
    print(
        f"\n── 3. Scaled-up target ({n_params(big_t) / 1e9:.2f}B) + scaled-up draft ({n_params(big_d) / 1e6:.0f}M), "
        f"BF16, replay with the counts per round of the first {len(P3)} prompts of part 2, wall-clock median of 3 ──"
    )
    print(f"Normal greedy decoding of {len(P3) * N} tokens: {tbb:.2f} s; cost coefficient c ≈ {c_big:.3f}")
    print(f"{'k':>2} {' accept α':>7} {' tok/round':>7} {'wall (s)':>7} {' speedup':>6} {'predicted':>7}")
    for k in KS:
        a = alpha_of(traces[k][: len(P3)], k)
        rounds = sum(len(tr) for tr in traces[k][: len(P3)])
        ts = statistics.median(ts_big[k])
        print(
            f"{k:2d} {a:9.3f} {(sum(map(sum, traces[k][: len(P3)])) + rounds) / rounds:10.2f} "
            f"{ts:8.2f} {tbb / ts:7.2f}× {expected_tokens(a, k) / (1 + k * c_big):8.2f}×"
        )
    print(f"\nTotal time {time.perf_counter() - t_start:.0f} s")
