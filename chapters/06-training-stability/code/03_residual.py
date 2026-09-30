"""第 6 章 · 极简代码 3：残差连接 —— 给信号修一条直通的高速路

同样 30 个块，每块 f(h) = ReLU(RMSNorm(h) · W1) · W2（Pre-Norm 写法），比较：
  - 普通堆叠：h ← f(h)
  - 残差连接：h ← h + f(h)
看三件事：
  1. 不同输入在深层还"分得清"吗（两两之间的余弦相似度，越接近 1 越分不清）；
  2. 残差流（residual stream）的尺度怎么随层数变化，输出投影 W2 要不要按层数缩小；
  3. 误差信号传回第 1 块时还剩多少、还"像不像"第 30 块收到的那个信号。
运行：uv run python chapters/06-training-stability/code/03_residual.py
"""

import importlib.util
import math
from pathlib import Path

import torch

torch.set_num_threads(1)

_spec = importlib.util.spec_from_file_location(
    "norm", Path(__file__).with_name("02_normalization.py"))
norm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(norm)

DEPTH, WIDTH, BATCH = 30, 256, 256


def mean_cosine(h: torch.Tensor) -> float:
    """一批样本两两之间的平均余弦相似度。"""
    u = h / h.norm(dim=-1, keepdim=True)
    sim = u @ u.T
    n = h.shape[0]
    return ((sim.sum() - n) / (n * (n - 1))).item()


def run(residual: bool, w2_scale: float = 1.0, depth: int = DEPTH, seed: int = 0):
    """返回每块之后的 (余弦相似度, 残差流 std, 误差信号 std, 第1块与第30块误差信号的相似度)。"""
    g = torch.Generator().manual_seed(seed)
    w1s = [torch.randn(WIDTH, WIDTH, generator=g) * math.sqrt(2 / WIDTH) for _ in range(depth)]
    w2s = [torch.randn(WIDTH, WIDTH, generator=g) * math.sqrt(1 / WIDTH) * w2_scale
           for _ in range(depth)]
    gamma = torch.ones(WIDTH)
    x = torch.randn(BATCH, WIDTH, generator=torch.Generator().manual_seed(seed + 1))
    x.requires_grad_()
    h, hs = x, []
    for w1, w2 in zip(w1s, w2s):
        f = torch.relu(norm.rms_norm(h, gamma) @ w1) @ w2
        h = h + f if residual else f       # ← 唯一的区别：有没有 "h +"
        h.retain_grad()
        hs.append(h)
    probe = torch.randn(WIDTH, generator=torch.Generator().manual_seed(seed + 2)) / WIDTH ** 0.5
    (norm.rms_norm(h, gamma) @ probe).sum().backward()
    cos = [mean_cosine(t.detach()) for t in hs]
    std = [t.std().item() for t in hs]
    grad = [t.grad.std().item() for t in hs]
    # 同一个样本，第 1 块收到的误差信号和第 30 块收到的，方向有多像？
    g1, gL = hs[0].grad, hs[-1].grad
    same_dir = torch.nn.functional.cosine_similarity(g1, gL, dim=-1).mean().item()
    return cos, std, grad, same_dir


VARIANTS = {
    "普通堆叠": dict(residual=False),
    "残差（W2 不缩放）": dict(residual=True),
    "残差（W2 × 1/√(2L)）": dict(residual=True, w2_scale=1 / math.sqrt(2 * DEPTH)),
}


if __name__ == "__main__":
    x = torch.randn(BATCH, WIDTH, generator=torch.Generator().manual_seed(1))
    print(f"输入本身：两两余弦相似度 = {mean_cosine(x):.3f}（随机向量几乎互相垂直）\n")
    shown = [1, 10, 20, 30]
    for name, kw in VARIANTS.items():
        cos, std, grad, same_dir = run(**kw)
        print(f"── {name}")
        print("   块号          " + "".join(f"{i:>9d}" for i in shown))
        print("   余弦相似度    " + "".join(f"{cos[i - 1]:>9.3f}" for i in shown))
        print("   残差流 std    " + "".join(f"{std[i - 1]:>9.3f}" for i in shown))
        print(f"   误差信号：第 1 块 / 第 30 块的大小 = {grad[0] / grad[-1]:.3f}，"
              f"方向相似度 = {same_dir:.3f}\n")
