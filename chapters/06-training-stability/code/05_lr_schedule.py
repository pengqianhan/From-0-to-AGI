"""第 6 章 · 极简代码 5：学习率调度（warmup + 余弦 / WSD）和梯度裁剪

这里只定义函数并打印数值；真正拿它们训练网络在 06、07 两个脚本里。
运行：uv run python chapters/06-training-stability/code/05_lr_schedule.py
"""

import math


def warmup_cosine(step: int, total: int, peak: float, warmup: int, min_ratio: float = 0.1) -> float:
    """线性 warmup 到 peak，然后按余弦曲线降到 peak × min_ratio。"""
    if step < warmup:
        return peak * (step + 1) / warmup                       # warmup：0 → peak
    p = (step - warmup) / max(1, total - warmup)                # 衰减进度 0 → 1
    return peak * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * p)))


def wsd(step: int, total: int, peak: float, warmup: int, decay_frac: float = 0.2,
        min_ratio: float = 0.0) -> float:
    """Warmup-Stable-Decay：warmup 到 peak，保持不变，最后 decay_frac 的步数线性降到 peak × min_ratio。"""
    decay_start = int(total * (1 - decay_frac))
    if step < warmup:
        return peak * (step + 1) / warmup                       # W：warmup
    if step < decay_start:
        return peak                                             # S：stable，恒定
    p = (step - decay_start) / max(1, total - decay_start)      # D：decay，线性降
    return peak * (1 - (1 - min_ratio) * p)


def clip_by_global_norm(grads, max_norm: float = 1.0):
    """梯度裁剪：所有参数的梯度拼成一个大向量，范数超过 max_norm 就整体等比例缩小。

    方向不变，只限制长度。返回裁剪前的范数（训练时常把它记到日志里）。
    """
    total = math.sqrt(sum(float((g * g).sum()) for g in grads))
    scale = min(1.0, max_norm / (total + 1e-6))
    for g in grads:
        g *= scale
    return total


if __name__ == "__main__":
    total, peak, warm = 1000, 3e-3, 100
    print(f"总步数 {total}，峰值学习率 {peak}，warmup {warm} 步")
    print("  步数   warmup+余弦   WSD（最后 20% 线性降到 0）")
    for s in [0, 50, 99, 100, 300, 500, 700, 799, 800, 900, 999]:
        print(f"{s:>6d}   {warmup_cosine(s, total, peak, warm):.2e}      {wsd(s, total, peak, warm):.2e}")

    import numpy as np
    g = [np.array([3.0, 4.0]), np.array([12.0])]               # 全局范数 = √(9+16+144) = 13
    before = clip_by_global_norm(g, max_norm=1.0)
    after = math.sqrt(sum(float((x * x).sum()) for x in g))
    print(f"\n梯度裁剪：裁剪前全局范数 {before:.1f}，裁剪后 {after:.3f}；"
          f"各分量 {[round(float(v), 4) for x in g for v in x]}（方向不变）")
