"""第 8 章 · 极简代码 3：为什么要除以 √d

q、k 的每个分量都是均值 0、方差 1 的随机数时，点积 q·k = Σ qᵢkᵢ 是 d 个"方差为 1"的项相加，
方差是 d。d 越大，分数越分散，softmax 就越接近 one-hot（几乎只看一个位置），
梯度也几乎为 0。除以 √d 把方差拉回 1，和 d 无关。

只用 NumPy，CPU 上一两秒跑完。
运行：uv run python chapters/08-attention/code/03_why_sqrt_d.py
"""

import numpy as np

DIMS = [16, 64, 256, 1024]
T = 16  # 每个查询面对 16 个键
TRIALS = 2000  # 重复次数


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def stats(d: int, scale: bool, rng: np.random.Generator) -> dict:
    q = rng.normal(size=(TRIALS, 1, d))
    k = rng.normal(size=(TRIALS, T, d))
    scores = (q * k).sum(-1)  # (TRIALS, T)：每行是一个查询对 T 个键的分数
    if scale:
        scores = scores / np.sqrt(d)
    p = softmax(scores)
    entropy = -(p * np.log(p + 1e-30)).sum(-1)
    # softmax 的雅可比矩阵 J = diag(p) − ppᵀ；它的大小决定了有多少梯度能传回分数
    jac = np.einsum("ni,ij->nij", p, np.eye(T)) - np.einsum("ni,nj->nij", p, p)
    return {
        "var": scores.var(),
        "max_w": p.max(-1).mean(),  # 最大权重（越接近 1 越像 one-hot）
        "eff_n": np.exp(entropy).mean(),  # "有效关注个数" = e^熵，均匀时 = T
        "jac": np.linalg.norm(jac, axis=(1, 2)).mean(),
    }


def main() -> None:
    rng = np.random.default_rng(0)
    print(f"q、k 每个分量 ~ N(0, 1)；每个查询对 T = {T} 个键做 softmax，重复 {TRIALS} 次取平均\n")
    header = f"{'d':>6} | {'分数方差':>8} {'最大权重':>8} {'有效个数':>8} {'梯度大小':>8}"
    for scale in (False, True):
        print("不缩放：softmax(q·k)" if not scale else "\n缩放后：softmax(q·k / √d)")
        print(header)
        for d in DIMS:
            s = stats(d, scale, rng)
            print(
                f"{d:>6} | {s['var']:>10.1f} {s['max_w']:>10.3f} {s['eff_n']:>10.2f} {s['jac']:>10.3f}"
            )
    print(
        "\n（有效个数 = e^熵：权重均匀分给 16 个位置时是 16，全压在一个位置上时是 1。"
        "梯度大小 = softmax 雅可比矩阵的 Frobenius 范数。）"
    )


if __name__ == "__main__":
    main()
