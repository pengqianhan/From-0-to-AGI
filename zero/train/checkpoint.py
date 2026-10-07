"""Checkpoint: save and restore the full training state (Chapter 14).

The goal: after a resume, the loss is the same as in a run without interruption (GOAL.md 9.1).
The model weights alone are not sufficient for this. The checkpoint also stores more state:

| Content | File | Written by |
|---|---|---|
| model weights | model.pt | rank 0 |
| optimizer state (AdamW first and second moments) | optim.pt | rank 0 |
| step, tokens trained, config, learning-rate schedule state | meta.json | rank 0 |
| data loader position + the state of all random number generators | rank{r}.pt | each rank writes its own file |

Directory layout:

    <ckpt_dir>/step_00001000/{model.pt, optim.pt, meta.json, rank0.pt, rank1.pt, ...}
    <ckpt_dir>/latest            # text file that contains the directory name of the latest checkpoint

**Atomic write**: first write to a temporary directory `.tmp_step_xxx`. When all files are complete,
`os.replace` renames it to the final directory. Then update `latest`.
Sometimes the system kills the training while it writes a checkpoint (frequent on preemptible instances).
Then only a temporary directory stays, and `latest` still points to the last complete checkpoint.
A resume never reads a half-written file.
"""

from __future__ import annotations

import json
import os
import random
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from zero.train.dist import DistInfo, barrier, unwrap_model


def _step_dir_name(step: int) -> str:
    return f"step_{step:08d}"


def rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        # Save and restore of the CUDA RNG state: verified on one RTX 3090 (2026-10, see runs/2026-10-01-gpu0-check/).
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        # Verified on one RTX 3090 (2026-10, see runs/2026-10-01-gpu0-check/).
        torch.cuda.set_rng_state_all(state["cuda"])


def _model_state_dict(model: nn.Module, parallel: str, info: DistInfo) -> dict[str, torch.Tensor]:
    if info.is_distributed and parallel == "fsdp":
        # With FSDP2, each rank has only a shard of the parameters. First gather them into a full
        # state_dict. This is a collective operation: all ranks must call it.
        # The 1-GPU path and 2 GPUs are verified on RTX 3090 (2026-10, see runs/2026-10-01-gpu0-check/:
        # 1-GPU load_policy reads a 2-GPU FSDP checkpoint back in strict mode).
        from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict

        return get_model_state_dict(
            model, options=StateDictOptions(full_state_dict=True, cpu_offload=True)
        )
    return {k: v.detach().cpu() for k, v in unwrap_model(model).state_dict().items()}


def _optim_state_dict(
    model: nn.Module, optimizer: torch.optim.Optimizer, parallel: str, info: DistInfo
) -> Any:
    if info.is_distributed and parallel == "fsdp":
        # The 1-GPU path and 2 GPUs are verified on RTX 3090 (2026-10, see runs/2026-10-01-gpu0-check/).
        from torch.distributed.checkpoint.state_dict import (
            StateDictOptions,
            get_optimizer_state_dict,
        )

        return get_optimizer_state_dict(
            model, optimizer, options=StateDictOptions(full_state_dict=True, cpu_offload=True)
        )
    return optimizer.state_dict()


def save_checkpoint(
    ckpt_dir: str | os.PathLike,
    step: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    loader: Any | None = None,
    meta: dict[str, Any] | None = None,
    info: DistInfo | None = None,
    parallel: str = "ddp",
    keep_last: int = 0,
) -> Path:
    """Save one checkpoint and return the final directory. With many processes, all ranks must call it."""
    info = info or DistInfo()
    root = Path(ckpt_dir)
    final = root / _step_dir_name(step)
    tmp = root / f".tmp_{_step_dir_name(step)}"
    if info.is_main:
        root.mkdir(parents=True, exist_ok=True)
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True)
    barrier()

    model_sd = _model_state_dict(model, parallel, info)
    optim_sd = (
        _optim_state_dict(model, optimizer, parallel, info) if optimizer is not None else None
    )
    if info.is_main:
        torch.save(model_sd, tmp / "model.pt")
        if optim_sd is not None:
            torch.save(optim_sd, tmp / "optim.pt")
        full_meta = {
            "step": step,
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "world_size": info.world_size,
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            **(meta or {}),
        }
        with open(tmp / "meta.json", "w") as f:
            json.dump(full_meta, f, indent=2, ensure_ascii=False, default=str)
    rank_state = {"rng": rng_state(), "loader": loader.state_dict() if loader is not None else None}
    torch.save(rank_state, tmp / f"rank{info.rank}.pt")
    barrier()

    if info.is_main:
        if final.exists():
            shutil.rmtree(final)
        os.replace(tmp, final)
        latest_tmp = root / "latest.tmp"
        latest_tmp.write_text(final.name)
        os.replace(latest_tmp, root / "latest")
        if keep_last > 0:
            prune_checkpoints(root, keep_last)
    barrier()
    return final


def list_checkpoints(ckpt_dir: str | os.PathLike) -> list[Path]:
    root = Path(ckpt_dir)
    if not root.exists():
        return []
    return sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("step_"))


def prune_checkpoints(ckpt_dir: str | os.PathLike, keep_last: int) -> None:
    for p in list_checkpoints(ckpt_dir)[:-keep_last]:
        shutil.rmtree(p, ignore_errors=True)


def find_latest(ckpt_dir: str | os.PathLike) -> Path | None:
    """ckpt_dir can be the checkpoint root directory (read `latest`) or one step_xxx directory."""
    root = Path(ckpt_dir)
    if (root / "model.pt").exists():
        return root
    latest = root / "latest"
    if latest.exists():
        p = root / latest.read_text().strip()
        if (p / "meta.json").exists():
            return p
    ckpts = [p for p in list_checkpoints(root) if (p / "meta.json").exists()]
    return ckpts[-1] if ckpts else None


def load_model_weights(
    path: str | os.PathLike, model: nn.Module, strict: bool = True
) -> dict[str, Any]:
    """Load only the model weights and return meta.

    Mid-training uses this to continue from a pretraining checkpoint.
    """
    ckpt = find_latest(path)
    if ckpt is None:
        raise FileNotFoundError(f"No checkpoint found in {path}")
    sd = torch.load(ckpt / "model.pt", map_location="cpu", weights_only=True)
    unwrap_model(model).load_state_dict(sd, strict=strict)
    with open(ckpt / "meta.json") as f:
        return json.load(f)


def load_checkpoint(
    path: str | os.PathLike,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    loader: Any | None = None,
    info: DistInfo | None = None,
    parallel: str = "ddp",
    restore_rng: bool = True,
) -> dict[str, Any]:
    """Restore the full training state and return meta (it contains the step)."""
    info = info or DistInfo()
    ckpt = find_latest(path)
    if ckpt is None:
        raise FileNotFoundError(f"No checkpoint found in {path}")
    with open(ckpt / "meta.json") as f:
        meta = json.load(f)
    if meta.get("world_size", 1) != info.world_size and loader is not None:
        raise ValueError(
            f"The checkpoint has world_size={meta.get('world_size')}, but the current world_size is "
            f"{info.world_size}. The data loader cannot restore its exact position"
        )

    model_sd = torch.load(ckpt / "model.pt", map_location="cpu", weights_only=True)
    if info.is_distributed and parallel == "fsdp":
        # The 1-GPU path and 2 GPUs are verified on RTX 3090 (2026-10, see runs/2026-10-01-gpu0-check/:
        # a 2-GPU FSDP resume from a checkpoint is bitwise identical to a run without interruption).
        from torch.distributed.checkpoint.state_dict import (
            StateDictOptions,
            set_model_state_dict,
            set_optimizer_state_dict,
        )

        opts = StateDictOptions(full_state_dict=True, broadcast_from_rank0=False)
        set_model_state_dict(model, model_sd, options=opts)
        if optimizer is not None and (ckpt / "optim.pt").exists():
            optim_sd = torch.load(ckpt / "optim.pt", map_location="cpu", weights_only=False)
            set_optimizer_state_dict(model, optimizer, optim_sd, options=opts)
    else:
        unwrap_model(model).load_state_dict(model_sd)
        if optimizer is not None and (ckpt / "optim.pt").exists():
            optimizer.load_state_dict(
                torch.load(ckpt / "optim.pt", map_location="cpu", weights_only=False)
            )
    if scheduler is not None and meta.get("scheduler") is not None:
        scheduler.load_state_dict(meta["scheduler"])

    rank_file = ckpt / f"rank{info.rank}.pt"
    if rank_file.exists():
        rank_state = torch.load(rank_file, map_location="cpu", weights_only=False)
        if loader is not None and rank_state.get("loader") is not None:
            loader.load_state_dict(rank_state["loader"])
        if restore_rng:
            set_rng_state(rank_state["rng"])
    meta["path"] = str(ckpt)
    return meta
