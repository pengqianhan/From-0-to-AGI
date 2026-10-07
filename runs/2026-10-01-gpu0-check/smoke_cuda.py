"""Part of Stage 6, item 9: run the training stages of `zero.smoke` on CUDA (BF16 autocast).

All `configs/tiny/*.toml` files set `device = "cpu"` and `dtype = "fp32"`. Thus on a machine with a GPU,
`python -m zero.smoke` still does all training on the CPU. This script does not change a configuration file.
When it reads a configuration, it changes only `[train] device` to "cuda" and `dtype` to "auto"
(on CUDA, "auto" is BF16 autocast). All other behavior is the same as zero.smoke:

    CUDA_VISIBLE_DEVICES=0 UV_NO_SYNC=1 uv run python runs/2026-10-01-gpu0-check/smoke_cuda.py \
        --out out/gpu0-check/smoke_cuda

Stages that run on CUDA: pretrain, midtrain, SFT, distillation (student + local teacher), DPO,
GRPO (sampling + update). In zero.smoke, eval / export / demo already use
`load_policy(..., device="cpu")`, so they still run on the CPU.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import zero.smoke as smoke  # noqa: E402

_orig_load_tiny = smoke.load_tiny


def load_tiny_cuda(name: str, out: str) -> dict:
    d = _orig_load_tiny(name, out)
    if "train" in d:
        d["train"]["device"] = "cuda"
        d["train"]["dtype"] = "auto"
    return d


if __name__ == "__main__":
    import torch

    assert torch.cuda.device_count() == 1, "Only 1 GPU may be visible (CUDA_VISIBLE_DEVICES=0)"
    smoke.load_tiny = load_tiny_cuda
    raise SystemExit(smoke.main(sys.argv[1:]))
