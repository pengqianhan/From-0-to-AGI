"""多卡第一步：NCCL 能不能在这几张卡之间跑通，all-reduce 的实测总线带宽是多少。

    CUDA_VISIBLE_DEVICES=0,2 UV_NO_SYNC=1 uv run torchrun --standalone --nproc_per_node=2 \
        runs/2026-10-01-gpu0-check/nccl_probe.py

每个 rank 放一个 256 MiB 的 FP32 张量，值为 rank + 1；预热一次后连续 all-reduce 10 次。
结果应为 (1 + 2) · 2^10 = 3072（2 卡）。总线带宽按 NCCL 的口径：2 · (n − 1) / n · 字节数 / 耗时。
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
        f"rank {rank}：{torch.cuda.get_device_name(rank)}，结果 {x[0].item():.0f}，"
        f"all-reduce 256 MiB 用时 {dt * 1e3:.1f} ms，总线带宽 {busbw:.1f} GB/s",
        flush=True,
    )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
