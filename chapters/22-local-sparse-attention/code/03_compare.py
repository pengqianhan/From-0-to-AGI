"""第 22 章 · 极简代码 3：全注意力 vs 滑动窗口 vs 局部-全局交替，三组真实数字

  1. 语言建模验证集 loss（字符级 Shakespeare）——三者差不多；
  2. KV cache：训练长度 128 和假想的 4096 两种上下文下，各要缓存多少个位置；
  3. 大海捞针：按"针"与提问的距离 d 分三段统计准确率——这里才看出滑动窗口的问题。

先运行 02_swa_model.py 训练（或者直接运行本脚本，缺的模型会自动训练）。
运行：uv run python chapters/22-local-sparse-attention/code/03_compare.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

torch.set_num_threads(1)
_spec = importlib.util.spec_from_file_location(
    "swa_model", Path(__file__).resolve().parent / "02_swa_model.py"
)
m = importlib.util.module_from_spec(_spec)
sys.modules["swa_model"] = m  # dataclass 需要能在 sys.modules 里找到所在模块
_spec.loader.exec_module(m)


def cached_positions(windows: list, T: int) -> int:
    """每层缓存的位置数之和：全注意力层存 T 个，滑动窗口层存 min(W, T) 个。"""
    return sum(T if w is None else min(w, T) for w in windows)


def bucket_means(acc: torch.Tensor, n_layers: int) -> list[float]:
    """把 d = 1..95 的准确率分三段：窗口内、全靠多层接力的区间、超出多层接力的区间。"""
    rf = n_layers * (m.W - 1)  # 只用滑动窗口时 L 层的感受野
    d = torch.arange(1, len(acc) + 1)
    segs = [d < m.W, (d >= m.W) & (d <= rf), d > rf]
    return [acc[s].mean().item() for s in segs]


def main() -> None:
    rf = 4 * (m.W - 1)
    print(f"窗口 W = {m.W}，4 层；只用滑动窗口时的理论感受野 = 4 × (W − 1) = {rf}\n")
    print(f"{'配置':<11}{'各层窗口':<22}{'LM 验证 loss':>12}{'KV 位置@128':>12}{'KV 位置@4096':>13}")
    for variant, windows in m.VARIANTS.items():
        model = m.load_or_train("lm", variant)
        loss = m.lm_val_loss(model)
        print(f"{variant:<11}{str(windows):<22}{loss:>12.3f}"
              f"{cached_positions(windows, 128):>12}{cached_positions(windows, 4096):>13}")

    print(f"\n大海捞针准确率（8 选 1，瞎猜 = 12.5%；序列长 {m.NEEDLE_T}，每个距离测 64 条）")
    print(f"{'配置':<11}{f'd < {m.W}（窗口内）':>16}{f'{m.W} ≤ d ≤ {rf}（接力）':>20}"
          f"{f'd > {rf}（够不着）':>16}")
    curves = {}
    for variant in m.VARIANTS:
        model = m.load_or_train("needle", variant)
        acc = m.needle_accuracy(model)
        curves[variant] = acc
        a, b, c = bucket_means(acc, 4)
        print(f"{variant:<11}{a:>16.1%}{b:>20.1%}{c:>16.1%}")

    # 更细的曲线：每 8 个距离取一次平均，方便画图（视频里用的就是这组数）
    print("\n按距离分段（每段 8 个距离的平均准确率）：")
    edges = list(range(1, m.NEEDLE_T, 8))
    print("  d 从      :", " ".join(f"{e:>4}" for e in edges))
    for variant, acc in curves.items():
        vals = [acc[e - 1 : e + 7].mean().item() for e in edges]
        print(f"  {variant:<10}:", " ".join(f"{v:>4.0%}" for v in vals))


if __name__ == "__main__":
    main()
