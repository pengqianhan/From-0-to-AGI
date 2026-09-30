"""在 CPU 上用 gloo 后端跑 2 进程 DDP，与单进程（batch 翻倍）逐步对拍（第 14 章）。

数据加载器保证"2 卡各 B 条"与"1 卡 2B 条"看到同一批全局样本；DDP 对梯度求平均，
所以两种跑法每一步的 loss 和最终参数都应该一致（只差浮点求和顺序带来的舍入误差）。
同时顺带测试多进程下的 checkpoint 保存（每个 rank 各写一份数据加载器状态）。
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
    not dist.is_available() or not dist.is_gloo_available(), reason="没有 gloo 后端"
)
def test_ddp_two_processes_matches_single(tmp_path: Path) -> None:
    src = {"a": write_random_shards(tmp_path / "data", "a", 2, 4000, 64, 0)}
    ddp_dir = tmp_path / "ddp"
    try:
        mp.spawn(_worker, args=(2, _free_port(), str(ddp_dir), src), nprocs=2, join=True)
    except Exception as e:  # pragma: no cover - 环境不支持多进程时跳过而不是报错
        if "Address already in use" in str(e) or "EADDRINUSE" in str(e):
            pytest.skip(f"无法启动 gloo 进程组：{e}")
        raise
    ddp = torch.load(ddp_dir / "result.pt", weights_only=True)

    # 单进程参照：micro_batch 翻倍，其余相同
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

    # 多进程 checkpoint：每个 rank 一份数据加载器状态
    last = ddp_dir / "ckpt" / "step_00000006"
    assert (last / "rank0.pt").exists() and (last / "rank1.pt").exists()
    assert json.loads((last / "meta.json").read_text())["world_size"] == 2
