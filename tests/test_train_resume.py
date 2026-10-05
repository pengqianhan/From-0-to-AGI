"""断点续训：训练 N 步 == 训练 k 步、存盘、重新加载、再训练 N-k 步（loss 逐位相同）。"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from zero.train.checkpoint import find_latest, list_checkpoints
from zero.train.dist import DistInfo
from zero.train.trainer import Trainer


def _sources(tmp_path: Path, random_shards) -> dict[str, str]:  # noqa: ANN001
    return {
        "a": random_shards(tmp_path / "data", "a", 2, 3000, 64, 0),
        "b": random_shards(tmp_path / "data", "b", 1, 3000, 64, 1),
    }


def test_resume_matches_uninterrupted(tmp_path: Path, random_shards, make_config) -> None:  # noqa: ANN001
    src = _sources(tmp_path, random_shards)
    val = random_shards(tmp_path / "data", "val", 1, 2000, 64, 2)

    # 不中断：一口气训 8 步
    cfg_a = make_config(tmp_path / "run_a", src, val)
    trainer_a = Trainer(cfg_a, DistInfo(), log=lambda _: None)
    hist_a = trainer_a.train()
    params_a = {k: v.clone() for k, v in trainer_a.raw_model.state_dict().items()}

    # 中断：训到第 5 步时"崩溃"（最后一个 checkpoint 在第 4 步），然后用新进程状态续训
    cfg_b = make_config(tmp_path / "run_b", src, val)
    Trainer(cfg_b, DistInfo(), log=lambda _: None).train(stop_at=5)
    assert find_latest(cfg_b.train.checkpoint_dir).name == "step_00000004"
    torch.manual_seed(999)  # 故意打乱全局随机数状态，续训必须把它恢复回来
    trainer_b = Trainer(cfg_b, DistInfo(), log=lambda _: None)
    assert trainer_b.step == 4
    hist_b = trainer_b.train()

    losses_a = {r["step"]: r["loss"] for r in hist_a}
    losses_b = {r["step"]: r["loss"] for r in hist_b}
    assert set(losses_b) == {5, 6, 7, 8}
    for s in losses_b:
        assert losses_a[s] == losses_b[s], (s, losses_a[s], losses_b[s])
    vals_a = [r["val_loss"] for r in hist_a if r["step"] == 8]
    vals_b = [r["val_loss"] for r in hist_b if r["step"] == 8]
    assert vals_a == vals_b
    for k, v in trainer_b.raw_model.state_dict().items():
        assert torch.equal(v, params_a[k]), k
    assert trainer_b.tokens_seen == trainer_a.tokens_seen

    # JSONL 日志
    lines = (tmp_path / "run_a" / "log.jsonl").read_text().strip().split("\n")
    rec = json.loads(lines[-1])
    assert rec["step"] == 8 and "tok_per_s" in rec and "mixture_counts" in rec


def test_stop_step_then_continue_matches_uninterrupted(tmp_path: Path, random_shards, make_config) -> None:  # noqa: ANN001
    # 阶梯实验的逐轮淘汰：先用 train.stop_step 训到 checkpoint 处停下，留下的配置再接着训到底
    src = _sources(tmp_path, random_shards)
    val = random_shards(tmp_path / "data", "val", 1, 2000, 64, 2)
    trainer_a = Trainer(make_config(tmp_path / "run_a", src, val), DistInfo(), log=lambda _: None)
    hist_a = trainer_a.train()

    cfg_b = make_config(tmp_path / "run_b", src, val)
    cfg_b.train.stop_step = 4  # 与 checkpoint.every 对齐
    first = Trainer(cfg_b, DistInfo(), log=lambda _: None).train()
    assert [r["step"] for r in first][-1] == 4
    assert find_latest(cfg_b.train.checkpoint_dir).name == "step_00000004"
    cfg_b.train.stop_step = 0
    trainer_b = Trainer(cfg_b, DistInfo(), log=lambda _: None)
    hist_b = trainer_b.train()
    losses_a = {r["step"]: r["loss"] for r in hist_a}
    assert {r["step"]: r["loss"] for r in hist_b} == {s: losses_a[s] for s in (5, 6, 7, 8)}
    for k, v in trainer_b.raw_model.state_dict().items():
        assert torch.equal(v, trainer_a.raw_model.state_dict()[k]), k


def test_loss_decreases_and_seed_reproducible(tmp_path: Path, random_shards, make_config) -> None:  # noqa: ANN001
    src = {"a": random_shards(tmp_path / "data", "a", 1, 5000, 64, 3)}
    h1 = Trainer(
        make_config(tmp_path / "r1", src, max_steps=30), DistInfo(), log=lambda _: None
    ).train()
    h2 = Trainer(
        make_config(tmp_path / "r2", src, max_steps=30), DistInfo(), log=lambda _: None
    ).train()
    assert [r["loss"] for r in h1] == [r["loss"] for r in h2]
    assert h1[-1]["loss"] < h1[0]["loss"] - 0.5


def test_checkpoint_pruning_and_atomic_layout(tmp_path: Path, random_shards, make_config) -> None:  # noqa: ANN001
    src = {"a": random_shards(tmp_path / "data", "a", 1, 3000, 64, 4)}
    cfg = make_config(tmp_path / "r", src, max_steps=12)
    cfg.train.checkpoint.every = 2
    cfg.train.checkpoint.keep_last = 2
    Trainer(cfg, DistInfo(), log=lambda _: None).train()
    ckpts = list_checkpoints(cfg.train.checkpoint_dir)
    assert [p.name for p in ckpts] == ["step_00000010", "step_00000012"]
    root = Path(cfg.train.checkpoint_dir)
    assert (root / "latest").read_text() == "step_00000012"
    assert not list(root.glob(".tmp_*"))
    assert {p.name for p in ckpts[-1].iterdir()} == {
        "model.pt",
        "optim.pt",
        "meta.json",
        "rank0.pt",
    }
