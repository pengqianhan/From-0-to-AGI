"""Chapter 20 · Minimal code 1: block-wise weight quantization, INT8 / INT4 from scratch

A weight matrix W in fp16 uses 16 bits for each number. Quantization approximates W with fewer bits:

    Blocks: along each row, each group of 32 numbers is one block.
    Each block has one scale factor: scale = max|w| / qmax (qmax is 127 for INT8 and 7 for INT4).
    Store the integer q = round(w / scale). To use the weight, multiply back: ŵ = scale · q.

Each block stores its scale in fp16. Over the 32 numbers, this adds 16/32 = 0.5 bit for each number:
    INT8 blocks → 8 + 0.5 = 8.5 bits/weight (the same format as Q8_0 in llama.cpp: 32 int8 + 1 fp16)
    INT4 blocks → 4 + 0.5 = 4.5 bits/weight (the same idea as Q4_0)

Why use blocks? One outlier makes the scale of its block large. Then the other numbers in the block
can use only a few of the integer levels. A smaller block has fewer numbers that one outlier can damage.
This script compares 4 methods: one scale for the whole matrix (per-tensor), one scale for each row
(per-row), one scale for each 32 numbers (block-32), and the two-level scale of K-quants. The Q4_K idea:
a super-block has 256 numbers and 8 sub-blocks. Each sub-block has a 6-bit scale and a 6-bit min.
The super-block stores the "scale of the scales" in fp16.

Run: uv run python chapters/20-release/code/01_blockwise_quant.py
"""

from __future__ import annotations

import numpy as np

BLOCK = 32


# ── Symmetric quantization: ŵ = scale · q, q ∈ [−qmax, qmax] ─────────────────
def quant_sym(w: np.ndarray, bits: int, block: int) -> tuple[np.ndarray, np.ndarray]:
    """Split w (rows × columns) into blocks of `block` numbers along the last axis.

    Return (integers q, the scale of each block).
    """
    qmax = 2 ** (bits - 1) - 1                      # INT8 → 127, INT4 → 7
    blocks = w.reshape(-1, block)                   # each row is one block
    absmax = np.abs(blocks).max(axis=1, keepdims=True)
    scale = (absmax / qmax).astype(np.float16).astype(np.float32)  # store the scale in fp16
    scale[scale == 0] = 1.0
    q = np.clip(np.round(blocks / scale), -qmax, qmax).astype(np.int8)  # q = round(w / scale)
    return q, scale


def dequant_sym(q: np.ndarray, scale: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    return (q.astype(np.float32) * scale).reshape(shape)  # ŵ = scale · q


# ── Asymmetric + two-level scale (the Q4_K idea): ŵ = d·sc_j · q − dmin·m_j ──
def kstyle_superblock(cols: int) -> int:
    """A super-block has 256 numbers. If the row length is not a multiple of 256
    (the small model of Chapter 10 has dim=128), use one super-block for the whole row."""
    return 256 if cols % 256 == 0 else cols


def quant_kstyle(w: np.ndarray, bits: int = 4, sub: int = 32) -> np.ndarray:
    """Return the dequantized matrix. Each sub-block (32 numbers): ŵ = scale_j · q + min_j,
    with q ∈ [0, 2^bits − 1]. scale_j and −min_j are each quantized to a 6-bit integer.
    These integers are multiplied by the fp16 factors d / dmin of the super-block (256 numbers)."""
    levels = 2**bits - 1
    sup = kstyle_superblock(w.shape[1])
    x = w.reshape(-1, sup // sub, sub)                       # (super-block, sub-block, 32)
    lo = np.minimum(x.min(axis=2), 0.0)                      # min_j ≤ 0 (the same as ggml)
    hi = x.max(axis=2)
    scale = (hi - lo) / levels                               # scale_j of the sub-block
    neg_min = -lo                                            # store as a non-negative number m_j
    d = (scale.max(axis=1, keepdims=True) / 63).astype(np.float16).astype(np.float32)
    dmin = (neg_min.max(axis=1, keepdims=True) / 63).astype(np.float16).astype(np.float32)
    d[d == 0], dmin[dmin == 0] = 1.0, 1.0
    sc = np.clip(np.round(scale / d), 1, 63)                 # 6-bit sub-block scale
    mn = np.clip(np.round(neg_min / dmin), 0, 63)            # 6-bit sub-block min
    s_hat = (d * sc)[..., None]
    m_hat = (dmin * mn)[..., None]
    q = np.clip(np.round((x + m_hat) / s_hat), 0, levels)    # 4-bit unsigned integer
    return (s_hat * q - m_hat).reshape(w.shape)


# ── For each method: how to quantize, and how many bits each weight uses ──────
def fake_quant(w: np.ndarray, scheme: str) -> np.ndarray:
    """Quantize, then dequantize (fake quant). Return an approximate matrix with the shape of w."""
    rows, cols = w.shape
    if scheme == "fp16":
        return w.astype(np.float16).astype(np.float32)
    if scheme == "int4-kquant":
        return quant_kstyle(w)
    bits, how = scheme.split("-")  # for example "int8-block32"
    bits = int(bits[3:])
    block = {"tensor": rows * cols, "row": cols, "block32": BLOCK}[how]
    q, s = quant_sym(w, bits, block)
    return dequant_sym(q, s, w.shape)


def bits_per_weight(scheme: str, shape: tuple[int, int]) -> float:
    rows, cols = shape
    n = rows * cols
    if scheme == "fp16":
        return 16.0
    if scheme == "int4-kquant":
        sup = kstyle_superblock(cols)                   # 4.5 for 256, the same as block_q4_K in ggml
        return 4 + (sup // 32 * (6 + 6) + 2 * 16) / sup
    bits, how = scheme.split("-")
    bits = int(bits[3:])
    n_scales = {"tensor": 1, "row": rows, "block32": n // BLOCK}[how]
    return bits + 16 * n_scales / n


SCHEMES = [
    "fp16",
    "int8-tensor", "int8-row", "int8-block32",
    "int4-tensor", "int4-row", "int4-block32", "int4-kquant",
]


def rel_err(w: np.ndarray, w_hat: np.ndarray) -> float:
    return float(np.linalg.norm(w - w_hat) / np.linalg.norm(w))


def make_weight(rows: int = 1024, cols: int = 1024, seed: int = 0) -> np.ndarray:
    """Make a matrix that looks like a weight matrix: Gaussian with mean 0 and std 0.02.
    Then multiply 0.1% of the elements by 10. These elements are the outliers."""
    rng = np.random.default_rng(seed)
    w = rng.normal(0, 0.02, size=(rows, cols)).astype(np.float32)
    mask = rng.random((rows, cols)) < 0.001
    w[mask] *= 10
    return w


def table(w: np.ndarray, x: np.ndarray) -> list[dict]:
    """For each method: bits/weight, matrix size, relative weight error, relative error of y = x·Wᵀ."""
    y = x @ w.T
    rows = []
    for s in SCHEMES:
        w_hat = fake_quant(w, s)
        bpw = bits_per_weight(s, w.shape)
        rows.append(dict(scheme=s, bpw=bpw, kib=w.size * bpw / 8 / 1024,
                         w_err=rel_err(w, w_hat), y_err=rel_err(y, x @ w_hat.T)))
    return rows


def show_block(w: np.ndarray, n: int = 8) -> dict:
    """Show the first block in detail: original values, scale, integers, dequantized values (INT8 and INT4)."""
    blk = w[0, :BLOCK]
    out = {"w": blk[:n].tolist()}
    for bits in (8, 4):
        q, s = quant_sym(blk[None, :], bits, BLOCK)
        out[f"int{bits}"] = dict(scale=float(s[0, 0]), q=q[0, :n].tolist(),
                                 w_hat=(q[0, :n] * s[0, 0]).tolist())
    return out


def main() -> None:
    w = make_weight()
    x = np.random.default_rng(1).normal(size=(64, w.shape[1])).astype(np.float32)
    print(f"Weight matrix {w.shape[0]}×{w.shape[1]}, std={w.std():.4f}, max|w|={np.abs(w).max():.4f}"
          f" (with {int((np.abs(w) > 0.1).sum())} outliers with |w|>0.1)\n")

    b = show_block(w)
    print("First 8 numbers of the first block:", " ".join(f"{v:+.4f}" for v in b["w"]))
    for bits in (8, 4):
        r = b[f"int{bits}"]
        print(f"  INT{bits}: scale={r['scale']:.6f}  q=", r["q"])
        print("        ŵ =", " ".join(f"{v:+.4f}" for v in r["w_hat"]))
    print()

    print(f"{'Scheme':<14}{'bits/w':>9}{'Size KiB':>10}{'Weight err':>14}{'Output err':>14}")
    for r in table(w, x):
        print(f"{r['scheme']:<14}{r['bpw']:>9.3f}{r['kib']:>10.0f}{r['w_err']:>14.2%}{r['y_err']:>14.2%}")
    print("\nHow to read: for INT4, one outlier makes the scale of the whole row or matrix large. "
          "Blocks cost 0.5 bit more for the scales, but the error becomes much smaller.")


if __name__ == "__main__":
    main()
