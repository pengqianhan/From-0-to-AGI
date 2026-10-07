"""Run 2-process DDP on CPU with the gloo backend. Do a step-by-step parity check against
one process with a doubled batch (Chapter 14).

The data loader makes sure that "2 GPUs with B samples each" and "1 GPU with 2B samples" see the
same global samples. DDP computes the mean of the gradients. Thus the loss at each step and the
final parameters must be the same for both runs (only the rounding error from a different
floating-point sum order is different).
The test also checks the checkpoint save with many processes (each rank writes its own data loader state).
"""

from __future__ import annotations

import json
import os
import socket
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from conftest import small_train_config, write_random_shards

from zero.train.dist import DistInfo, cleanup, init_distributed
from zero.train.trainer import Trainer


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _worker(rank: int, world_size: int, port: int, out_dir: str, src: dict[str, str]) -> None:
    os.environ.update(
        {
            "RANK": str(rank),
            "LOCAL_RANK": str(rank),
            "WORLD_SIZE": str(world_size),
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(port),
        }
    )
    torch.set_num_threads(1)
    try:
        cfg = small_train_config(
            Path(out_dir), src, micro_batch_size=2, grad_accum_steps=2, max_steps=6
        )
        info = init_distributed("cpu")
        assert info.world_size == world_size and info.backend == "gloo"
        trainer = Trainer(cfg, info, log=lambda _: None)
        hist = trainer.train()
        if rank == 0:
            torch.save(
                {"losses": [r["loss"] for r in hist], "state": trainer.raw_model.state_dict()},
                Path(out_dir) / "result.pt",
            )
    finally:
        cleanup()


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(), reason="no gloo backend"
)
def test_ddp_two_processes_matches_single(tmp_path: Path) -> None:
    src = {"a": write_random_shards(tmp_path / "data", "a", 2, 4000, 64, 0)}
    ddp_dir = tmp_path / "ddp"
    try:
        mp.spawn(_worker, args=(2, _free_port(), str(ddp_dir), src), nprocs=2, join=True)
    except Exception as e:  # pragma: no cover - skip (not fail) if the environment does not support many processes
        if "Address already in use" in str(e) or "EADDRINUSE" in str(e):
            pytest.skip(f"Cannot start the gloo process group: {e}")
        raise
    ddp = torch.load(ddp_dir / "result.pt", weights_only=True)

    # One-process reference: micro_batch doubled, all else the same.
    cfg = small_train_config(
        tmp_path / "single", src, micro_batch_size=4, grad_accum_steps=2, max_steps=6
    )
    trainer = Trainer(cfg, DistInfo(), log=lambda _: None)
    hist = trainer.train()

    torch.testing.assert_close(
        torch.tensor(ddp["losses"]), torch.tensor([r["loss"] for r in hist]), rtol=1e-5, atol=1e-5
    )
    for k, v in trainer.raw_model.state_dict().items():
        torch.testing.assert_close(ddp["state"][k], v, rtol=1e-4, atol=1e-5)

    # Checkpoint with many processes: one data loader state for each rank.
    last = ddp_dir / "ckpt" / "step_00000006"
    assert (last / "rank0.pt").exists() and (last / "rank1.pt").exists()
    assert json.loads((last / "meta.json").read_text())["world_size"] == 2
