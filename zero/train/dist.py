"""Initialization and wrappers for distributed training (Chapter 14).

When you start with `torchrun --nproc_per_node=8 -m zero.train.pretrain --config ...`, torchrun sets
the environment variables RANK / LOCAL_RANK / WORLD_SIZE / MASTER_ADDR / MASTER_PORT for each process.
This module reads these variables:

- No variables: one process, world_size = 1.
- GPUs are available: the backend is NCCL, and each process uses one GPU (LOCAL_RANK).
- Only CPU: the backend is gloo (`tests/test_ddp_cpu.py` uses this to test DDP on CPU).

Two types of data parallelism:
- **DDP**: each GPU has a full copy of the model. In the backward pass, an all-reduce computes
  the mean of the gradients. A 0.6–0.8B model + AdamW fits on an 80GB GPU, and DDP is the simplest.
- **FSDP2** (`torch.distributed.fsdp.fully_shard`): shards the parameters, gradients, and optimizer
  state across the GPUs. This saves GPU memory, but it needs more communication.
  Use it for larger models or longer contexts.
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
        raise RuntimeError("The config requires device=cuda, but CUDA is not available on this machine")
    return torch.device("cuda", local_rank)


def init_distributed(device_pref: str = "auto", timeout_min: float | None = None) -> DistInfo:
    """Initialize the process group from the torchrun environment variables.

    With one process, do nothing. timeout_min: the timeout of the collectives (default of PyTorch:
    10 minutes for NCCL). Stages where rank 0 prepares data while the others wait set it longer.
    """
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = pick_device(device_pref, local_rank)
    backend = ""
    if world_size > 1:
        backend = "nccl" if device.type == "cuda" else "gloo"
        if device.type == "cuda":
            # Runs only with many GPUs. Verified on 2×RTX 3090 (PCIe) (2026-10, see runs/2026-10-01-gpu0-check/).
            torch.cuda.set_device(device)
        if not dist.is_initialized():
            kwargs = {}
            if timeout_min is not None:
                from datetime import timedelta

                kwargs["timeout"] = timedelta(minutes=timeout_min)
            dist.init_process_group(backend=backend, rank=rank, world_size=world_size, **kwargs)
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
    """Compute the mean across ranks. With one process, return the tensor unchanged."""
    if dist.is_available() and dist.is_initialized():
        t = t.clone()
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        t /= dist.get_world_size()
    return t


def wrap_model(model: nn.Module, info: DistInfo, parallel: str = "ddp") -> nn.Module:
    """Wrap the model in DDP or FSDP2 when there are many processes. With one process, return it unchanged."""
    if not info.is_distributed:
        return model
    if parallel == "ddp":
        from torch.nn.parallel import DistributedDataParallel as DDP

        device_ids = [info.local_rank] if info.device.type == "cuda" else None
        return DDP(model, device_ids=device_ids)
    if parallel == "fsdp":
        # FSDP2 shards each Block as one unit. Then it shards the remaining parameters of the full
        # model (embedding, final norm).
        # Verified on 2×RTX 3090 (PCIe) (2026-10, see runs/2026-10-01-gpu0-check/): the shards are
        # really on 2 GPUs. Over 50 steps, the loss differed from DDP on the same data by a maximum of
        # 0.10% (lr 3e-4). The 1-GPU path (an NCCL process group with world_size=1) is also verified.
        # The throughput with 8 GPUs and NVLink is not measured on GPU yet.
        # Mixed precision: the parameters compute in bf16, and the gradient reduce uses fp32.
        from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

        mp = None
        if info.device.type == "cuda":
            mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
        kwargs = {"mp_policy": mp} if mp is not None else {}
        for layer in model.layers:  # type: ignore[attr-defined]
            fully_shard(layer, **kwargs)
        fully_shard(model, **kwargs)
        return model
    raise ValueError(f"Unknown parallel mode: {parallel}")


def unwrap_model(model: nn.Module) -> nn.Module:
    """Remove the DDP / torch.compile wrappers and return the original Transformer."""
    if hasattr(model, "module"):
        model = model.module  # DDP
    if hasattr(model, "_orig_mod"):
        model = model._orig_mod  # torch.compile
    return model
