"""第 6 章 · 极简代码 1：一个 30 层的普通 MLP，信号在层与层之间怎么走？

不训练，只看"刚初始化的那一刻"：把一批随机输入送进去，逐层打印
  - 前向：每一层输出（激活值）的标准差；
  - 反向：损失对每一层激活值的梯度（误差信号）的标准差。
对比几种初始化：权重标准差取 1.0、0.01、0.02，以及按输入维度缩放的 Xavier（sqrt(1/fan_in)）
和 Kaiming（sqrt(2/fan_in)）。
运行：uv run python chapters/06-training-stability/code/01_signal_propagation.py
"""

import math

import torch

torch.set_num_threads(1)  # 小矩阵用单线程反而更快，结果也更可复现

DEPTH, WIDTH, BATCH = 30, 256, 512


def make_weights(std: float, depth: int = DEPTH, width: int = WIDTH, seed: int = 0):
    """depth 个 width×width 的权重矩阵，元素 ~ N(0, std²)。"""
    g = torch.Generator().manual_seed(seed)
    return [(torch.randn(width, width, generator=g) * std).requires_grad_() for _ in range(depth)]


def layer_stats(std: float, depth: int = DEPTH, width: int = WIDTH, seed: int = 0):
    """返回 (每层激活标准差列表, 每层梯度标准差列表)。

    网络：h_l = ReLU(h_{l-1} · W_l)，没有偏置、没有归一化、没有残差。
    损失：每个样本的输出和一个固定随机方向做内积再求和（只为了让梯度流回来）。
    "梯度"指 ∂loss/∂h_l：误差信号传回第 l 层时还剩多大。
    """
    ws = make_weights(std, depth, width, seed)
    g = torch.Generator().manual_seed(seed + 1)
    x = torch.randn(BATCH, width, generator=g)
    h, hs = x, []
    for w in ws:
        h = torch.relu(h @ w)          # 一层：线性 + ReLU
        h.retain_grad()                # 让 PyTorch 保留中间层的梯度
        hs.append(h)
    probe = torch.randn(width, generator=g) / math.sqrt(width)
    loss = (h @ probe).sum()
    loss.backward()
    act_std = [t.std().item() for t in hs]
    grad_std = [t.grad.std().item() for t in hs]
    return act_std, grad_std


def kaiming_std(fan_in: int = WIDTH) -> float:
    """ReLU 网络的"刚好"初始化：Var(W) = 2 / fan_in。"""
    return math.sqrt(2 / fan_in)


INITS = {
    "std = 1.0": 1.0,
    "std = 0.01": 0.01,
    "std = 0.02": 0.02,                       # 大模型配置里常见的 initializer_range
    "Xavier": math.sqrt(1 / WIDTH),           # Var = 1/fan_in：没考虑 ReLU 砍掉一半
    "Kaiming": kaiming_std(),                 # Var = 2/fan_in
}


if __name__ == "__main__":
    shown = [1, 5, 10, 15, 20, 25, 30]
    print(f"{DEPTH} 层、每层宽 {WIDTH} 的 ReLU MLP；输入的标准差 = 1\n")
    for name, std in INITS.items():
        act, grad = layer_stats(std)
        print(f"── 初始化 {name}（权重标准差 {std:.4g}）")
        print("   层号      " + "".join(f"{i:>10d}" for i in shown))
        print("   激活 std  " + "".join(f"{act[i - 1]:>10.3g}" for i in shown))
        print("   梯度 std  " + "".join(f"{grad[i - 1]:>10.3g}" for i in shown))
        print(f"   第 1 层梯度 / 第 30 层梯度 = {grad[0] / grad[-1]:.3g}\n")

    # 为什么 Kaiming 取 2/fan_in？每层 std 的放大倍数 ≈ std · sqrt(fan_in / 2)
    for name, std in INITS.items():
        print(f"{name:>11s}：每层放大倍数 ≈ {std * math.sqrt(WIDTH / 2):.3f}，"
              f"30 层之后 ≈ {(std * math.sqrt(WIDTH / 2)) ** DEPTH:.3g}")
