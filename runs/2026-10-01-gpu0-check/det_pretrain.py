"""`zero.train.pretrain` 的确定性版本：先打开 PyTorch 的确定性算法，再调用原入口（不改 zero 的代码）。

    torchrun --standalone --nproc_per_node=1 runs/2026-10-01-gpu0-check/det_pretrain.py --config ... [--set ...]

GPU 上默认有几处非确定性（FlashAttention 反向的 dQ 用原子加、embedding 反向的 index_add 等），
同一条命令跑两次，loss 从第 2 步起就有 1e-5 级差异，在大学习率的早期阶段会被放大。
为了把"续训是否精确恢复"和"GPU 本身的非确定性"分开，第 3 项另用这个入口跑一遍。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
torch.use_deterministic_algorithms(True)

from zero.train.pretrain import main  # noqa: E402

if __name__ == "__main__":
    main()
