"""阶段 6 第 9 项的一部分：把 `zero.smoke` 的训练阶段搬到 CUDA 上跑（BF16 autocast）。

`configs/tiny/*.toml` 都写死了 `device = "cpu"`、`dtype = "fp32"`，所以在有 GPU 的机器上直接跑
`python -m zero.smoke`，训练仍然全部在 CPU 上。本脚本不改任何配置文件，只在读配置时把
`[train] device` 换成 "cuda"、`dtype` 换成 "auto"（CUDA 上即 BF16 autocast），其余与 zero.smoke 完全相同：

    CUDA_VISIBLE_DEVICES=0 UV_NO_SYNC=1 uv run python runs/2026-10-01-gpu0-check/smoke_cuda.py \
        --out out/gpu0-check/smoke_cuda

走 CUDA 的阶段：pretrain、midtrain、SFT、蒸馏（学生 + 本地教师）、DPO、GRPO（采样 + 更新）。
eval / export / demo 在 zero.smoke 里本来就用 `load_policy(..., device="cpu")`，仍在 CPU 上。
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

    assert torch.cuda.device_count() == 1, "只允许看到 1 张卡（CUDA_VISIBLE_DEVICES=0）"
    smoke.load_tiny = load_tiny_cuda
    raise SystemExit(smoke.main(sys.argv[1:]))
