"""第 20 章 · 极简代码 1：分块（block-wise）权重量化，从零写 INT8 / INT4

一个权重矩阵 W 用 fp16 存，每个数 16 bit。量化 = 用更少的 bit 近似它：

    分块：沿每一行，每 32 个数一块（block）；
    每块一个缩放系数 scale = max|w| / qmax（qmax：INT8 取 127，INT4 取 7）；
    存整数 q = round(w / scale)，用的时候再乘回去：ŵ = scale · q。

每块的 scale 用 fp16 存，摊到每个数上是 16/32 = 0.5 bit：
    INT8 分块 → 8 + 0.5 = 8.5 bit/权重（和 llama.cpp 的 Q8_0 同一种格式：32 个 int8 + 1 个 fp16）
    INT4 分块 → 4 + 0.5 = 4.5 bit/权重（和 Q4_0 同一种思路）

为什么要分块？一个离群值（outlier）会把整块的 scale 撑大，块里其他数就只剩很少几个格子可用。
块越小，一个离群值能"连累"的数就越少。本脚本对比：整张矩阵一个 scale（per-tensor）、每行一个
（per-row）、每 32 个一个（block-32），以及 K-quant 的两级 scale（Q4_K 的思路：256 个数一个超级块，
里面 8 个子块各有一个 6 bit 的 scale 和 min，超级块再用 fp16 存"scale 的 scale"）。

运行：uv run python chapters/20-release/code/01_blockwise_quant.py
"""

from __future__ import annotations

import numpy as np

BLOCK = 32


# ── 对称量化：ŵ = scale · q，q ∈ [−qmax, qmax] ────────────────────────────
def quant_sym(w: np.ndarray, bits: int, block: int) -> tuple[np.ndarray, np.ndarray]:
    """把 w（行 × 列）沿最后一维每 block 个数分一块，返回 (整数 q, 每块的 scale)。"""
    qmax = 2 ** (bits - 1) - 1                      # INT8 → 127，INT4 → 7
    blocks = w.reshape(-1, block)                   # 每行是一块
    absmax = np.abs(blocks).max(axis=1, keepdims=True)
    scale = (absmax / qmax).astype(np.float16).astype(np.float32)  # scale 用 fp16 存
    scale[scale == 0] = 1.0
    q = np.clip(np.round(blocks / scale), -qmax, qmax).astype(np.int8)  # q = round(w / scale)
    return q, scale


def dequant_sym(q: np.ndarray, scale: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    return (q.astype(np.float32) * scale).reshape(shape)  # ŵ = scale · q


# ── 非对称 + 两级 scale（Q4_K 的思路）：ŵ = d·sc_j · q − dmin·m_j ───────────
def kstyle_superblock(cols: int) -> int:
    """超级块 256 个数；行长不是 256 的倍数时（第 10 章小模型 dim=128）退成整行一个超级块。"""
    return 256 if cols % 256 == 0 else cols


def quant_kstyle(w: np.ndarray, bits: int = 4, sub: int = 32) -> np.ndarray:
    """返回反量化后的矩阵。每个子块（32 个数）：ŵ = scale_j · q + min_j，q ∈ [0, 2^bits − 1]；
    scale_j 和 −min_j 再各自量化成 6 bit 整数，乘上超级块（256 个数）的 fp16 系数 d / dmin。"""
    levels = 2**bits - 1
    sup = kstyle_superblock(w.shape[1])
    x = w.reshape(-1, sup // sub, sub)                       # (超级块, 子块, 32)
    lo = np.minimum(x.min(axis=2), 0.0)                      # min_j ≤ 0（和 ggml 一样）
    hi = x.max(axis=2)
    scale = (hi - lo) / levels                               # 子块的 scale_j
    neg_min = -lo                                            # 存成非负数 m_j
    d = (scale.max(axis=1, keepdims=True) / 63).astype(np.float16).astype(np.float32)
    dmin = (neg_min.max(axis=1, keepdims=True) / 63).astype(np.float16).astype(np.float32)
    d[d == 0], dmin[dmin == 0] = 1.0, 1.0
    sc = np.clip(np.round(scale / d), 1, 63)                 # 6 bit 的子块 scale
    mn = np.clip(np.round(neg_min / dmin), 0, 63)            # 6 bit 的子块 min
    s_hat = (d * sc)[..., None]
    m_hat = (dmin * mn)[..., None]
    q = np.clip(np.round((x + m_hat) / s_hat), 0, levels)    # 4 bit 无符号整数
    return (s_hat * q - m_hat).reshape(w.shape)


# ── 每种方案：怎么量化、每个权重摊到几 bit ────────────────────────────────
def fake_quant(w: np.ndarray, scheme: str) -> np.ndarray:
    """量化再反量化（fake quant），返回和 w 同形状的近似矩阵。"""
    rows, cols = w.shape
    if scheme == "fp16":
        return w.astype(np.float16).astype(np.float32)
    if scheme == "int4-kquant":
        return quant_kstyle(w)
    bits, how = scheme.split("-")  # 例如 "int8-block32"
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
        sup = kstyle_superblock(cols)                   # 256 时 = 4.5，与 ggml 的 block_q4_K 一致
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
    """造一个"像权重"的矩阵：均值 0、标准差 0.02 的高斯，再把 0.1% 的元素放大 10 倍当离群值。"""
    rng = np.random.default_rng(seed)
    w = rng.normal(0, 0.02, size=(rows, cols)).astype(np.float32)
    mask = rng.random((rows, cols)) < 0.001
    w[mask] *= 10
    return w


def table(w: np.ndarray, x: np.ndarray) -> list[dict]:
    """每种方案：bit/权重、矩阵大小、权重相对误差、输出 y = x·Wᵀ 的相对误差。"""
    y = x @ w.T
    rows = []
    for s in SCHEMES:
        w_hat = fake_quant(w, s)
        bpw = bits_per_weight(s, w.shape)
        rows.append(dict(scheme=s, bpw=bpw, kib=w.size * bpw / 8 / 1024,
                         w_err=rel_err(w, w_hat), y_err=rel_err(y, x @ w_hat.T)))
    return rows


def show_block(w: np.ndarray, n: int = 8) -> dict:
    """把第一块拆开看：原值、scale、整数、反量化值（INT8 和 INT4 各一次）。"""
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
    print(f"权重矩阵 {w.shape[0]}×{w.shape[1]}，std={w.std():.4f}，max|w|={np.abs(w).max():.4f}"
          f"（含 {int((np.abs(w) > 0.1).sum())} 个 |w|>0.1 的离群值）\n")

    b = show_block(w)
    print("第一块的前 8 个数：", " ".join(f"{v:+.4f}" for v in b["w"]))
    for bits in (8, 4):
        r = b[f"int{bits}"]
        print(f"  INT{bits}: scale={r['scale']:.6f}  q=", r["q"])
        print("        ŵ =", " ".join(f"{v:+.4f}" for v in r["w_hat"]))
    print()

    print(f"{'方案':<14}{'bit/权重':>9}{'大小 KiB':>10}{'权重相对误差':>14}{'输出相对误差':>14}")
    for r in table(w, x):
        print(f"{r['scheme']:<14}{r['bpw']:>9.3f}{r['kib']:>10.0f}{r['w_err']:>14.2%}{r['y_err']:>14.2%}")
    print("\n读法：同样是 INT4，一个离群值会把整行/整张矩阵的 scale 撑大；"
          "分块后多花 0.5 bit 存 scale，误差明显变小。")


if __name__ == "__main__":
    main()
