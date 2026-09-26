"""第 6 章 · 极简代码 2：归一化 —— LayerNorm 和 RMSNorm

1. 手写两种归一化，用一个 4 维向量看它们各做了什么；
2. 把 RMSNorm 插进第 1 个脚本的 30 层网络（每层先归一化再乘矩阵），
   看初始化再"错"，信号也不再塌缩或爆炸。
运行：uv run python chapters/06-training-stability/code/02_normalization.py
"""

import importlib.util
from pathlib import Path

import torch

torch.set_num_threads(1)

_spec = importlib.util.spec_from_file_location(
    "signal", Path(__file__).with_name("01_signal_propagation.py"))
signal = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(signal)


def layer_norm(x, gamma, beta, eps=1e-6):
    """LayerNorm：减均值、除标准差，再乘 γ 加 β。每个样本（最后一维）单独算。"""
    mu = x.mean(-1, keepdim=True)
    var = ((x - mu) ** 2).mean(-1, keepdim=True)
    return (x - mu) / torch.sqrt(var + eps) * gamma + beta


def rms_norm(x, gamma, eps=1e-6):
    """RMSNorm：不减均值，只除以均方根 RMS(x) = sqrt(mean(x²))，再乘 γ。"""
    rms = torch.sqrt((x * x).mean(-1, keepdim=True) + eps)
    return x / rms * gamma


def normed_layer_stats(std: float, depth: int = signal.DEPTH, width: int = signal.WIDTH,
                       seed: int = 0):
    """和 01 一样的网络，但每层变成 h_l = ReLU(RMSNorm(h_{l-1}) · W_l)。"""
    ws = signal.make_weights(std, depth, width, seed)
    gamma = torch.ones(width)
    g = torch.Generator().manual_seed(seed + 1)
    x = torch.randn(signal.BATCH, width, generator=g)
    h, hs = x, []
    for w in ws:
        h = torch.relu(rms_norm(h, gamma) @ w)
        h.retain_grad()
        hs.append(h)
    probe = torch.randn(width, generator=g) / width ** 0.5
    (rms_norm(h, gamma) @ probe).sum().backward()  # 输出前也归一化一次（对应 Pre-Norm 的最终 norm）
    # 归一化后的激活尺度（下一层真正"看到"的输入）永远是 1；这里报告的是它的输入 h_l
    return [t.std().item() for t in hs], [t.grad.std().item() for t in hs]


if __name__ == "__main__":
    x = torch.tensor([2.0, 4.0, 6.0, 8.0])
    one, zero = torch.ones(4), torch.zeros(4)
    ln, rn = layer_norm(x, one, zero), rms_norm(x, one)
    print("输入 x                 =", x.tolist())
    print(f"LayerNorm(x)           = {[round(v, 3) for v in ln.tolist()]}"
          f"   均值 {ln.mean():.3f}，RMS {ln.pow(2).mean().sqrt():.3f}")
    print(f"RMSNorm(x)             = {[round(v, 3) for v in rn.tolist()]}"
          f"   均值 {rn.mean():.3f}，RMS {rn.pow(2).mean().sqrt():.3f}")
    xc = x - x.mean()
    print(f"x 先减掉均值后：LayerNorm 与 RMSNorm 的最大差 = "
          f"{(layer_norm(xc, one, zero) - rms_norm(xc, one)).abs().max():.2e}")
    print(f"把 x 放大 100 倍：RMSNorm 输出的最大变化 = "
          f"{(rms_norm(100 * x, one) - rn).abs().max():.2e}（尺度不变性）")
    print("可学习参数（宽 d）：LayerNorm 有 γ、β 共 2d 个，RMSNorm 只有 γ 共 d 个\n")

    shown = [1, 10, 20, 30]
    print(f"{signal.DEPTH} 层 ReLU MLP，每层前面加 RMSNorm：")
    print("   初始化       " + "".join(f"{'第' + str(i) + '层激活':>12s}" for i in shown)
          + "   第1层梯度/第30层梯度")
    for name in ["std = 1.0", "std = 0.01", "std = 0.02", "Kaiming"]:
        act, grad = normed_layer_stats(signal.INITS[name])
        print(f"   {name:<10s}  " + "".join(f"{act[i - 1]:>15.3g}" for i in shown)
              + f"   {grad[0] / grad[-1]:>12.3g}")
