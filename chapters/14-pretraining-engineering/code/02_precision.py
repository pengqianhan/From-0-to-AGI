"""Chapter 14 · Minimal code 2: FP32, BF16, FP16. What do we lose with fewer bits?

A floating-point number = sign bit + exponent bits + mantissa bits:
    FP32: 1 + 8 + 23     BF16: 1 + 8 + 7      FP16: 1 + 5 + 10
The exponent bits set the range (how large and how small a number can be).
The mantissa bits set the precision (the distance between two adjacent numbers).
BF16 keeps the 8 exponent bits of FP32 (the same range) and cuts only the mantissa.
FP16 cuts both, so its range is much smaller.

The script shows this with real numbers:
  ① the bit layout, maximum value, smallest normal number, and machine epsilon of the three formats;
  ② add 0.01 10,000 times (the correct answer is 100): a low-precision accumulator "gets stuck";
  ③ weight update w ← w − η·g: a BF16 weight cannot take a small step → so we need FP32 master weights;
  ④ stochastic rounding: on average, a BF16 accumulator no longer gets stuck;
  ⑤ the range problem of FP16: large numbers overflow to inf, and small gradients underflow to 0
     (so FP16 training needs loss scaling, but BF16 does not);
  ⑥ PyTorch autocast: matmuls run in BF16, but parameters and gradients stay in FP32.

Run: uv run python chapters/14-pretraining-engineering/code/02_precision.py   (a few seconds)
"""

import struct

import torch

torch.manual_seed(0)
FORMATS = {"fp32": (torch.float32, 8, 23), "bf16": (torch.bfloat16, 8, 7), "fp16": (torch.float16, 5, 10)}


def bits(x: float, dtype: torch.dtype) -> str:
    """Convert x to dtype. Then show its bits as "sign | exponent | mantissa"."""
    t = torch.tensor(x, dtype=dtype)
    if dtype == torch.float32:
        raw = struct.unpack(">I", struct.pack(">f", float(t)))[0]
        n, e = 32, 8
    else:
        raw = int(t.view(torch.int16).item()) & 0xFFFF
        n, e = 16, 8 if dtype == torch.bfloat16 else 5
    b = format(raw, f"0{n}b")
    return f"{b[0]} | {b[1:1 + e]} | {b[1 + e:]}"


def repeat_add(value: float, n: int, dtype: torch.dtype) -> float:
    acc = torch.zeros((), dtype=dtype)
    v = torch.tensor(value, dtype=dtype)
    for _ in range(n):
        acc = acc + v  # each result is rounded back to dtype
    return float(acc)


def to_bf16_stochastic(x: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """Stochastic rounding to BF16: add a random number in [0, 2^16) to the low 16 bits of FP32, then truncate.

    The nearer x is to the BF16 number above it, the higher the probability to round up.
    Thus the expected value is exactly x.
    """
    raw = x.float().view(torch.int32)
    noise = torch.randint(0, 1 << 16, raw.shape, generator=gen, dtype=torch.int32)
    return ((raw + noise) & ~0xFFFF).view(torch.float32).to(torch.bfloat16)


def main():
    torch.set_num_threads(1)
    print("① Bit layout (sign | exponent | mantissa) and range")
    x = 3.14159
    print(f"  {'':5} {'bits of π':<42} {'stored value':>12} {'max':>10} {'min normal':>11} {'epsilon':>9}")
    for name, (dt, _, _) in FORMATS.items():
        fi = torch.finfo(dt)
        stored = float(torch.tensor(x, dtype=dt))
        print(f"  {name:5} {bits(x, dt):<42} {stored:>12.7f} {fi.max:>10.3g} {fi.tiny:>11.3g} {fi.eps:>9.3g}")
    print("  epsilon = distance from 1 to the next number = 2^-(mantissa bits): FP32 2^-23, BF16 2^-7, FP16 2^-10")

    print("\n② Add 0.01 10,000 times (the correct answer is 100)")
    for name, (dt, _, _) in FORMATS.items():
        print(f"  {name} accumulator: {repeat_add(0.01, 10_000, dt):.4f}")
    print("  BF16 has only 7 mantissa bits: in [4, 8), two adjacent numbers are 4 × 2^-7 = 0.03125 apart.")
    print("  0.01 is less than half of this gap, so each sum rounds back to the old value. The accumulator stays at 4; FP16 stays at 32 for the same reason")

    print("\n③ Weight update: w = 1.0, each step w ← w − η·g with η·g = 1e-3, 1000 steps (the correct answer is 0.0)")
    for name, dt in [("bf16 weight", torch.bfloat16), ("fp32 weight", torch.float32)]:
        w = torch.tensor(1.0, dtype=dt)
        for _ in range(1000):
            w = w - torch.tensor(1e-3, dtype=dt)
        print(f"  {name}: {float(w):.4f}")
    print("  → The optimizer must keep FP32 master weights. Use BF16 only for the forward/backward computation")

    print("\n④ Stochastic rounding: BF16 accumulator + stochastic rounding (0.01 × 10,000, 5 seeds)")
    results = []
    for seed in range(5):
        gen = torch.Generator().manual_seed(seed)
        acc = torch.zeros((), dtype=torch.bfloat16)
        for _ in range(10_000):
            acc = to_bf16_stochastic(acc.float() + 0.01, gen)
        results.append(float(acc))
    print("  " + "  ".join(f"{r:.2f}" for r in results) + f"   mean {sum(results) / 5:.2f}")

    print("\n⑤ The range problem of FP16 (BF16 for comparison)")
    for v in (70000.0, 1e-8):
        print(f"  {v:>8g} → fp16: {float(torch.tensor(v, dtype=torch.float16)):<10g}"
              f" bf16: {float(torch.tensor(v, dtype=torch.bfloat16)):g}")
    grads = torch.randn(100_000) * 1e-6  # a batch of very small gradients
    for name, dt in [("fp16", torch.float16), ("bf16", torch.bfloat16)]:
        zero_frac = (grads.to(dt) == 0).float().mean().item()
        print(f"  Gradients with std 1e-6 converted to {name}: {zero_frac:.1%} become 0")
    scaled = (grads * 1024).to(torch.float16)
    print(f"  FP16, multiplied by 1024 before the conversion (loss scaling): {(scaled == 0).float().mean().item():.1%} become 0")

    print("\n⑥ autocast: matmuls run in BF16, but parameters and gradients stay in FP32")
    lin = torch.nn.Linear(1024, 1024, bias=False)
    xin = torch.randn(64, 1024)
    ref = lin(xin)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = lin(xin)
    out.float().pow(2).mean().backward()
    rel = ((out.float() - ref).norm() / ref.norm()).item()
    print(f"  Output dtype: {out.dtype}; parameter dtype: {lin.weight.dtype}; gradient dtype: {lin.weight.grad.dtype}")
    print(f"  Relative error against the FP32 result: {rel:.2e} (the size of the BF16 epsilon is {torch.finfo(torch.bfloat16).eps:.1e})")


if __name__ == "__main__":
    main()
