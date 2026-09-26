"""第 9 章 · 极简代码 5：QK-Norm —— 不管 q、k 长多大，注意力分数都被管住

注意力分数 = q·k / √d。训练中 q、k 的长度（范数）可能越长越大，分数跟着变大，
softmax 就会"一边倒"（几乎全部权重压在一个位置上），梯度变得很尖、很不稳定。
QK-Norm 在做点积之前，先对每个头的 q、k 各做一次 RMSNorm（第 6 章），长度被拉回固定尺度。

实验：同一组随机 q、k，整体放大 s 倍（模拟训练中范数变大），看
  - 最大注意力分数（logit）
  - softmax 后最大的那个权重、注意力分布的熵（越小越"一边倒"）
运行：uv run python chapters/09-modern-transformer/code/05_qk_norm.py
"""

import math

import torch

torch.manual_seed(0)
T, D = 16, 32  # 16 个位置，head_dim = 32


def rms_norm(x, eps=1e-6):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


def stats(q, k):
    logits = q @ k.T / math.sqrt(D)                    # (T, T) 注意力分数
    p = logits.softmax(-1)
    entropy = -(p * p.clamp_min(1e-12).log()).sum(-1).mean()  # 每行的熵，取平均
    return logits.abs().max().item(), p.max(-1).values.mean().item(), entropy.item()


q0, k0 = torch.randn(T, D), torch.randn(T, D)
print(f"均匀分布（完全不挑）的熵 = ln {T} = {math.log(T):.2f}\n")
print(f"{'放大倍数 s':>10} | {'不加 QK-Norm：最大分数':>20} {'平均最大权重':>10} {'熵':>6} |"
      f" {'加 QK-Norm：最大分数':>18} {'平均最大权重':>10} {'熵':>6}")
for s in [1, 2, 4, 8, 16]:
    q, k = q0 * s, k0 * s
    a = stats(q, k)
    b = stats(rms_norm(q), rms_norm(k))
    print(f"{s:>10} | {a[0]:>22.1f} {a[1]:>14.3f} {a[2]:>6.2f} | {b[0]:>20.1f} {b[1]:>14.3f} {b[2]:>6.2f}")
