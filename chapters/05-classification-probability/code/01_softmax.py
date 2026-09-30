"""第 5 章 · 极简代码 1：softmax —— 把一组分数变成概率分布

只用 NumPy，CPU 上瞬间跑完。
运行：uv run python chapters/05-classification-probability/code/01_softmax.py
"""

import numpy as np


def softmax_naive(z: np.ndarray) -> np.ndarray:
    """按定义写：p_k = exp(z_k) / Σ_j exp(z_j)。logits 一大就溢出。"""
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def softmax(z: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """数值稳定版：先减去最大值，再取 exp。
    softmax(z) = softmax(z − c) 对任意常数 c 成立，取 c = max(z) 让最大的指数恰好是 e⁰ = 1。
    temperature（温度）T：先把 logits 除以 T。T < 1 更尖锐，T > 1 更平坦（第 10 章采样会用）。
    """
    z = np.asarray(z, dtype=np.float64) / temperature
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


# 本章贯穿的例子：一张图该归到 猫 / 狗 / 鸟 哪一类？模型输出三个分数（logits）
CLASSES = ["猫", "狗", "鸟"]
LOGITS = np.array([2.0, 1.0, -1.0])


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    p = softmax(LOGITS)
    print("logits（任意实数）   :", LOGITS)
    print("exp(logits)          :", np.exp(LOGITS))
    print("softmax（概率）      :", p, " 求和 =", p.sum())
    print("朴素版算 logits + 100:", softmax_naive(LOGITS + 100), "（加同一个常数，结果不变）")

    print("\n—— 数值稳定性 ——")
    big = LOGITS * 500  # [1000, 500, -500]
    with np.errstate(over="ignore", invalid="ignore"):
        print("朴素版 softmax([1000, 500, -500]) =", softmax_naive(big), " ← exp(1000) 溢出成 inf")
    print("稳定版 softmax([1000, 500, -500]) =", softmax(big))
    print("float64 能表示的最大 exp 指数约为", np.log(np.finfo(np.float64).max).round(1),
          "；float32 约为", np.log(np.finfo(np.float32).max).round(1))

    print("\n—— 温度 T：logits 先除以 T ——")
    for t in [0.5, 1.0, 2.0, 10.0]:
        print(f"T = {t:>4}：", softmax(LOGITS, temperature=t))
