"""Checkpoint：保存与恢复训练的全部状态（对应第 14 章）。

要做到"断点续训后的 loss 与不中断时一致"（GOAL.md 9.1），光存模型权重不够，还要存：

| 内容 | 文件 | 谁来写 |
|---|---|---|
| 模型权重 | model.pt | rank 0 |
| 优化器状态（AdamW 的一阶、二阶矩） | optim.pt | rank 0 |
| 步数、已训练 token 数、配置、学习率调度状态 | meta.json | rank 0 |
| 数据加载器位置 + 各种随机数发生器状态 | rank{r}.pt | 每个 rank 各写一份 |

目录结构：

    <ckpt_dir>/step_00001000/{model.pt, optim.pt, meta.json, rank0.pt, rank1.pt, ...}
    <ckpt_dir>/latest            # 文本文件，内容是最新一个 checkpoint 的目录名

**原子写入**：先写到临时目录 `.tmp_step_xxx`，全部写完再 `os.replace` 改名成正式目录，最后更新 `latest`。
训练在写 checkpoint 的中途被杀掉（抢占式实例常有），留下的只是一个临时目录，`latest` 仍然指向
上一个完整的 checkpoint，不会读到写了一半的文件。
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
        state["cuda"] = torch.cuda.get_rng_state_all()  # 尚未在 GPU 上验证
    return state


def set_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])  # 尚未在 GPU 上验证


def _model_state_dict(model: nn.Module, parallel: str, info: DistInfo) -> dict[str, torch.Tensor]:
    if info.is_distributed and parallel == "fsdp":
        # 尚未在 GPU 上验证：FSDP2 下每个 rank 只有一片参数，要先聚合成完整的 state_dict（集合通信，所有 rank 都要调用）
        from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict

        return get_model_state_dict(
            model, options=StateDictOptions(full_state_dict=True, cpu_offload=True)
        )
    return {k: v.detach().cpu() for k, v in unwrap_model(model).state_dict().items()}


def _optim_state_dict(
    model: nn.Module, optimizer: torch.optim.Optimizer, parallel: str, info: DistInfo
) -> Any:
    if info.is_distributed and parallel == "fsdp":
        # 尚未在 GPU 上验证
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
    """保存一个 checkpoint，返回最终目录。多进程时所有 rank 都要调用。"""
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
    """ckpt_dir 可以是 checkpoint 根目录（读 latest）或者某个 step_xxx 目录本身。"""
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
    """只加载模型权重（中期训练从预训练 checkpoint 接着训时用），返回 meta。"""
    ckpt = find_latest(path)
    if ckpt is None:
        raise FileNotFoundError(f"在 {path} 找不到 checkpoint")
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
    """恢复全部训练状态，返回 meta（含 step）。"""
    info = info or DistInfo()
    ckpt = find_latest(path)
    if ckpt is None:
        raise FileNotFoundError(f"在 {path} 找不到 checkpoint")
    with open(ckpt / "meta.json") as f:
        meta = json.load(f)
    if meta.get("world_size", 1) != info.world_size and loader is not None:
        raise ValueError(
            f"checkpoint 是 {meta.get('world_size')} 卡存的，现在是 {info.world_size} 卡，数据位置无法精确恢复"
        )

    model_sd = torch.load(ckpt / "model.pt", map_location="cpu", weights_only=True)
    if info.is_distributed and parallel == "fsdp":
        # 尚未在 GPU 上验证
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
