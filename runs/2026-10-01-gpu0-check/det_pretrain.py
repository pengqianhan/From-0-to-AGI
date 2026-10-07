"""Deterministic version of `zero.train.pretrain`.

It turns on the deterministic algorithms of PyTorch, then calls the original entry point.
The code in zero does not change.

    torchrun --standalone --nproc_per_node=1 runs/2026-10-01-gpu0-check/det_pretrain.py --config ... [--set ...]

On a GPU, some operations are not deterministic by default (the FlashAttention backward pass adds dQ
with atomic adds, the embedding backward pass uses index_add, and others). If you run the same command
two times, the loss differs by about 1e-5 from step 2. In the early phase with a large learning rate,
training makes this difference larger. Item 3 runs one more time with this entry point. This separates
"resume restores the exact state" from "the GPU itself is not deterministic".
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
