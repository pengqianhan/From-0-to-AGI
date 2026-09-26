"""第 3 章 · 极简代码 2：激活函数，以及"一个 ReLU = 一个折点"

1. 常见激活函数在几个点上的取值：ReLU、sigmoid、tanh（历史上常用）、SiLU、GELU（现代大模型的 SwiGLU 里用 SiLU）。
2. ReLU 拼积木：|x| = ReLU(x) + ReLU(−x)；"帐篷" = ReLU(x+1) − 2·ReLU(x) + ReLU(x−1)。
   每个 ReLU 只贡献一个折点，把它们按权重加起来，就能拼出弯折的形状。
运行：uv run python chapters/03-neural-network/code/02_activations.py
"""

import math

import numpy as np


def relu(z):
    return np.maximum(0.0, z)                        # ReLU(z) = max(0, z)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))                  # σ(z) = 1 / (1 + e^(−z))


def tanh(z):
    return np.tanh(z)


def silu(z):
    return z * sigmoid(z)                            # SiLU(z) = z · σ(z)，也叫 Swish


def gelu(z):
    # GELU(z) = z · Φ(z)，Φ 是标准正态分布的累积分布函数
    return z * 0.5 * (1.0 + np.vectorize(math.erf)(z / math.sqrt(2.0)))


ACTIVATIONS = {"ReLU": relu, "sigmoid": sigmoid, "tanh": tanh, "SiLU": silu, "GELU": gelu}


def hinge(x, w: float, b: float, v: float):
    """一个隐藏单元对输出的贡献：v · ReLU(w·x + b)。折点在 x = −b / w。"""
    return v * relu(w * x + b)


if __name__ == "__main__":
    zs = np.array([-3.0, -1.0, 0.0, 1.0, 3.0])
    print("1) 激活函数在几个点上的取值")
    print("   z        " + "".join(f"{z:>8.1f}" for z in zs))
    for name, f in ACTIVATIONS.items():
        print(f"   {name:<8} " + "".join(f"{v:>8.3f}" for v in f(zs)))

    x = np.linspace(-2, 2, 9)
    print("\n2) 用 ReLU 拼形状（x 从 −2 到 2）")
    rows = [
        ("x", x),
        ("|x| = ReLU(x)+ReLU(−x)", hinge(x, 1, 0, 1) + hinge(x, -1, 0, 1)),
        ("帐篷 = ReLU(x+1)−2ReLU(x)+ReLU(x−1)",
         hinge(x, 1, 1, 1) + hinge(x, 1, 0, -2) + hinge(x, 1, -1, 1)),
    ]
    for name, vals in rows:
        print(f"   {name}")
        print("      " + "".join(f"{v:>6.1f}" for v in vals))
    print("   → 两个 ReLU 拼出 |x|（1 个折点）；三个 ReLU 拼出帐篷（折点在 −1、0、1）。")
