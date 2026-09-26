"""分布式训练的初始化与封装（对应第 14 章）。

用 `torchrun --nproc_per_node=8 -m zero.train.pretrain --config ...` 启动时，torchrun 会给每个进程
设置环境变量 RANK / LOCAL_RANK / WORLD_SIZE / MASTER_ADDR / MASTER_PORT。这里读这些变量：

- 没有这些变量：单进程，world_size = 1；
- 有 GPU：后端用 NCCL，每个进程绑定一张卡（LOCAL_RANK）；
- 只有 CPU：后端用 gloo（`tests/test_ddp_cpu.py` 就是这样在 CPU 上测 DDP 的）。

两种数据并行：
- **DDP**：每张卡一份完整模型，反向传播时对梯度做 all-reduce 求平均。0.6–0.8B 的模型 + AdamW
  在 80GB 卡上放得下，DDP 最简单；
- **FSDP2**（`torch.distributed.fsdp.fully_shard`）：把参数、梯度、优化器状态切分到各卡，
  省显存，代价是更多通信。模型更大或上下文更长时用。
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch import nn


@dataclass
class DistInfo:
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    device: torch.device = torch.device("cpu")
    backend: str = ""

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    @property
    def is_distributed(self) -> bool:
        return self.world_size > 1


def pick_device(pref: str = "auto", local_rank: int = 0) -> torch.device:
    if pref == "cpu" or (pref == "auto" and not torch.cuda.is_available()):
        return torch.device("cpu")
    if not torch.cuda.is_available():
        raise RuntimeError("配置要求 device=cuda，但这台机器没有可用的 CUDA")
    return torch.device("cuda", local_rank)


def init_distributed(device_pref: str = "auto") -> DistInfo:
    """根据 torchrun 的环境变量初始化进程组；单进程时什么都不做。"""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = pick_device(device_pref, local_rank)
    backend = ""
    if world_size > 1:
        backend = "nccl" if device.type == "cuda" else "gloo"
        if device.type == "cuda":
            torch.cuda.set_device(device)  # 尚未在 GPU 上验证
        if not dist.is_initialized():
            dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    return DistInfo(
        rank=rank, local_rank=local_rank, world_size=world_size, device=device, backend=backend
    )


def cleanup() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def all_reduce_mean(t: torch.Tensor) -> torch.Tensor:
    """跨 rank 求平均（单进程时原样返回）。"""
    if dist.is_available() and dist.is_initialized():
        t = t.clone()
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        t /= dist.get_world_size()
    return t


def wrap_model(model: nn.Module, info: DistInfo, parallel: str = "ddp") -> nn.Module:
    """多进程时用 DDP 或 FSDP2 包装模型；单进程原样返回。"""
    if not info.is_distributed:
        return model
    if parallel == "ddp":
        from torch.nn.parallel import DistributedDataParallel as DDP

        device_ids = [info.local_rank] if info.device.type == "cuda" else None
        return DDP(model, device_ids=device_ids)
    if parallel == "fsdp":
        # 尚未在 GPU 上验证：FSDP2 按 Block 逐层切分，再对整个模型切分剩余参数（embedding、最后的 norm）。
        # 混合精度：参数以 bf16 参与计算，梯度 reduce 用 fp32。
        from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

        mp = None
        if info.device.type == "cuda":
            mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
        kwargs = {"mp_policy": mp} if mp is not None else {}
        for layer in model.layers:  # type: ignore[attr-defined]
            fully_shard(layer, **kwargs)
        fully_shard(model, **kwargs)
        return model
    raise ValueError(f"未知的并行方式：{parallel}")


def unwrap_model(model: nn.Module) -> nn.Module:
    """去掉 DDP / torch.compile 的包装，拿到原始的 Transformer。"""
    if hasattr(model, "module"):
        model = model.module  # DDP
    if hasattr(model, "_orig_mod"):
        model = model._orig_mod  # torch.compile
    return model
