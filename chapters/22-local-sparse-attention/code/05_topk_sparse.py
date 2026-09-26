"""第 22 章 · 极简代码 5：稀疏注意力的核心想法 —— 按内容挑 k 个键，而不是按位置

拿 02 里训练好的**全注意力**模型，推理时让每个 query 只看 k 个键，两种挑法预算相同：
  - 最近 k 个（按位置）：就是把每层临时改成窗口为 k 的滑动窗口；
  - 分数最高的 k 个（按内容）：先算出全部 q·k 分数，只留前 k 个再做 softmax。
不重新训练，看大海捞针准确率和语言建模 loss 各掉了多少。

注意：这里为了挑 top-k 先把全部分数算了一遍，所以一点也不省算力；真实的稀疏注意力
（DeepSeek 的 DSA、MiniMax 的 MSA 等）用一个便宜得多的"索引器"来打分，再对选中的键做精确注意力。
运行：uv run python chapters/22-local-sparse-attention/code/05_topk_sparse.py
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


def set_recent(model, k: int | None) -> None:
    for blk in model.blocks:
        blk.attn.window = k


def main() -> None:
    needle = m.load_or_train("needle", "full")
    lm = m.load_or_train("lm", "full")
    rf = 4 * (m.W - 1)
    print("全注意力模型，推理时每个 query 只留 k 个键（不重新训练）")
    print(f"{'挑法':<16}{'k':>4}{f'捞针 d<{m.W}':>11}{f'捞针 d>{rf}':>11}{'全部距离':>10}{'LM loss':>9}")
    base_acc = m.needle_accuracy(needle)
    base_loss = m.lm_val_loss(lm)
    print(f"{'不限制（全注意力）':<16}{'-':>4}{base_acc[: m.W - 1].mean():>11.1%}"
          f"{base_acc[rf:].mean():>11.1%}{base_acc.mean():>10.1%}{base_loss:>9.3f}")
    for k in (4, 8, 16):
        for name in ("最近 k 个", "分数最高的 k 个"):
            for model in (needle, lm):
                if name == "最近 k 个":
                    set_recent(model, k)
                else:
                    model.set_topk(k)
            acc = m.needle_accuracy(needle)
            loss = m.lm_val_loss(lm)
            for model in (needle, lm):
                set_recent(model, None)
                model.set_topk(None)
            print(f"{name:<16}{k:>4}{acc[: m.W - 1].mean():>11.1%}{acc[rf:].mean():>11.1%}"
                  f"{acc.mean():>10.1%}{loss:>9.3f}")


if __name__ == "__main__":
    main()
