"""First multi-GPU step: does NCCL work between these GPUs, and what is the measured all-reduce bus bandwidth?

    CUDA_VISIBLE_DEVICES=0,2 UV_NO_SYNC=1 uv run torchrun --standalone --nproc_per_node=2 \
        runs/2026-10-01-gpu0-check/nccl_probe.py

Each rank holds a 256 MiB FP32 tensor with the value rank + 1. After one warmup call, the script runs
all-reduce 10 times in sequence. The result must be (1 + 2) · 2^10 = 3072 (2 GPUs).
The bus bandwidth uses the NCCL definition: 2 · (n − 1) / n · bytes / time.
"""

from __future__ import annotations

import os
import time

import torch
import torch.distributed as dist


def main() -> None:
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", rank))
    n = dist.get_world_size()
    x = torch.full((64 * 2**20,), float(rank + 1), device="cuda")  # 256 MiB
    dist.all_reduce(x)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(10):
        dist.all_reduce(x)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / 10
    busbw = 2 * (n - 1) / n * x.numel() * 4 / dt / 1e9
    print(
        f"rank {rank}: {torch.cuda.get_device_name(rank)}, result {x[0].item():.0f}, "
        f"all-reduce of 256 MiB took {dt * 1e3:.1f} ms, bus bandwidth {busbw:.1f} GB/s",
        flush=True,
    )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
