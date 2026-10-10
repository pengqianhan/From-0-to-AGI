"""Data parallelism of DPO / GRPO / on-policy distillation: 2 CPU processes (gloo) == 1 process.

The batch sizes in the configs are global, each rank divides its loss by the global normalizer, and
`LoopState.optimizer_step` sums the gradients over the ranks. Samples are seeded by their global
index. So 2 processes must give the same per-step losses and the same final weights as 1 process on
the same global batch (only the floating-point summation order differs). See `zero/post/common.py`.
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

from tests.conftest import post_config
from zero.train.checkpoint import find_latest


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _run(stage: str, cfg: dict) -> list[dict]:
    torch.set_num_threads(1)
    if stage == "grpo":
        from zero.post.grpo import run_grpo

        return run_grpo(cfg, log=lambda _: None)
    if stage == "opd":
        from zero.post.opd import run_opd

        return run_opd(cfg, log=lambda _: None)
    from zero.post.dpo import run_dpo  # "dpo" and "dpo_gen"

    return run_dpo(cfg, log=lambda _: None)


def _worker(rank: int, world: int, port: int, stage: str, cfg: dict, out: str) -> None:
    os.environ.update(
        {
            "RANK": str(rank),
            "LOCAL_RANK": str(rank),
            "WORLD_SIZE": str(world),
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(port),
        }
    )
    hist = _run(stage, cfg)
    if rank == 0:
        Path(out).write_text(json.dumps(hist))


@pytest.fixture(scope="module")
def teacher_ckpt(tmp_path_factory, chat_tok, chat_tok_path):  # noqa: ANN001, ANN201
    """A tiny model with other weights (same tokenizer): an OPD teacher that differs from the student."""
    import dataclasses

    from tests.conftest import TINY_POST_MODEL
    from zero.config import ModelConfig
    from zero.model import Transformer
    from zero.train.checkpoint import save_checkpoint

    torch.manual_seed(7)
    cfg = ModelConfig(vocab_size=chat_tok.vocab_size, **TINY_POST_MODEL)
    root = tmp_path_factory.mktemp("teacher") / "ckpt"
    meta = {
        "config": {
            "model": dataclasses.asdict(cfg),
            "train": {"data": {"tokenizer": str(chat_tok_path)}},
        }
    }
    save_checkpoint(root, 0, Transformer(cfg), meta=meta)
    return root


def _configs(stage: str, tmp_path: Path, tok, tok_path, ckpt, teacher) -> dict:  # noqa: ANN001
    from zero.post.common import write_jsonl
    from zero.post.sft import env_conversations

    if stage == "grpo":
        return post_config(
            tmp_path,
            tok_path,
            ckpt,
            tok.vocab_size,
            grpo={
                "group_size": 3,
                "prompts_per_step": 4,
                "max_new_tokens": 8,
                "kl_coef": 0.05,
                "n_train_tasks": 20,
                "ppo_epochs": 2,
                "forward_batch": 2,
            },
        )
    if stage == "opd":
        write_jsonl(tmp_path / "sft.jsonl", env_conversations(6, 0, "train"))
        return post_config(
            tmp_path,
            tok_path,
            ckpt,
            tok.vocab_size,
            opd={
                "prompts_per_step": 4,
                "max_new_tokens": 8,
                "n_tool_tasks": 20,
                "forward_batch": 2,
                "teachers": [
                    {"name": "chat", "path": str(teacher), "prompts": str(tmp_path / "sft.jsonl")},
                    {"name": "tool", "path": str(teacher), "prompts": "tool_env"},
                ],
            },
        )
    # DPO: precomputed reference, 3 pairs per micro-batch, 2 micro-steps → 6 different pairs per step
    from zero.post.chat import format_tool_call
    from zero.post.envs.tool_env import generate_tasks

    rows = []
    for t in generate_tasks(12, seed=4):
        good = (
            "\n".join(format_tool_call(c) for c in t.gold_calls) if t.gold_calls else t.gold_answer
        )
        rows.append(
            {
                "messages": t.messages,
                "tools": t.tools,
                "chosen": {"role": "assistant", "content": good},
                "rejected": {"role": "assistant", "content": "I don't know."},
            }
        )
    write_jsonl(tmp_path / "prefs.jsonl", rows)
    dpo = {"train_jsonl": str(tmp_path / "prefs.jsonl"), "beta": 0.5}
    if stage == "dpo_gen":  # the pairs are sampled by the policy first, split over the ranks
        dpo = {"train_jsonl": str(tmp_path / "gen.jsonl"), "beta": 0.5, "generate_pairs": 7,
               "samples_per_prompt": 2, "max_new_tokens": 8}
    d = post_config(tmp_path, tok_path, ckpt, tok.vocab_size, dpo=dpo)
    d["train"].update({"micro_batch_size": 3, "grad_accum_steps": 2, "max_steps": 3})
    return d


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(), reason="no gloo backend"
)
@pytest.mark.parametrize("stage", ["grpo", "opd", "dpo", "dpo_gen"])
def test_two_processes_match_one(
    stage: str, tmp_path: Path, chat_tok, chat_tok_path, tiny_ckpt, teacher_ckpt
) -> None:  # noqa: ANN001
    one = _configs(stage, tmp_path / "one", chat_tok, chat_tok_path, tiny_ckpt, teacher_ckpt)
    two = json.loads(json.dumps(one).replace(str(tmp_path / "one"), str(tmp_path / "two")))
    (tmp_path / "two").mkdir(parents=True, exist_ok=True)
    for f in (tmp_path / "one").glob("*.jsonl"):  # the same input files in the second directory
        (tmp_path / "two" / f.name).write_text(f.read_text())

    h1 = _run(stage, one)
    out = tmp_path / "hist2.json"
    try:
        mp.spawn(_worker, args=(2, _free_port(), stage, two, str(out)), nprocs=2, join=True)
    except Exception as e:  # pragma: no cover - skip if the environment cannot start 2 processes
        if "Address already in use" in str(e) or "EADDRINUSE" in str(e):
            pytest.skip(f"Cannot start the gloo process group: {e}")
        raise
    h2 = json.loads(out.read_text())

    if stage == "dpo_gen":  # 2 ranks sampled the same pairs as 1 process
        g1, g2 = (tmp_path / d / "gen.jsonl" for d in ("one", "two"))
        assert g1.read_text() == g2.read_text() and len(g1.read_text().splitlines()) == 7
    assert len(h1) == len(h2) == one["train"]["max_steps"]
    for a, b in zip(h1, h2):
        assert a["loss"] == pytest.approx(b["loss"], rel=1e-4, abs=1e-6), (stage, a, b)
        for k in ("reward_mean", "kl", "acc", "resp_len"):
            if k in a:
                assert a[k] == pytest.approx(b[k], rel=1e-4, abs=1e-6), (stage, k)
    s1 = torch.load(find_latest(tmp_path / "one" / "run" / "ckpt") / "model.pt", weights_only=True)
    s2 = torch.load(find_latest(tmp_path / "two" / "run" / "ckpt") / "model.pt", weights_only=True)
    for k in s1:
        assert torch.allclose(s1[k], s2[k], atol=1e-5, rtol=1e-4), (stage, k)
    # the weights really changed (the test would pass trivially otherwise)
    s0 = torch.load(find_latest(tiny_ckpt) / "model.pt", weights_only=True)
    assert any(not torch.allclose(s0[k], s1[k]) for k in s0)


def _distill_worker(rank: int, world: int, port: int, cfg: dict) -> None:
    os.environ.update({"RANK": str(rank), "LOCAL_RANK": str(rank), "WORLD_SIZE": str(world),
                       "MASTER_ADDR": "127.0.0.1", "MASTER_PORT": str(port)})
    torch.set_num_threads(1)
    from zero.post.distill import run_distill

    run_distill(cfg, log=lambda _: None)


def _distill_config(tmp_path: Path, tok, tok_path, ckpt) -> dict:  # noqa: ANN001
    from zero.post.common import write_jsonl
    from zero.post.sft import env_conversations

    write_jsonl(tmp_path / "sft.jsonl", env_conversations(8, 0, "train"))
    d = post_config(
        tmp_path, tok_path, ckpt, tok.vocab_size,
        data={"format": "sft", "seq_len": 512},
        teacher={"backend": "local", "path": str(ckpt), "name": "self", "max_new_tokens": 12},
        distill={"out_jsonl": str(tmp_path / "kd.jsonl"), "n_tasks": 3, "samples_per_task": 2,
                 "mix_sft_jsonl": str(tmp_path / "sft.jsonl"), "mix_sft_max": 8, "kd_alpha": 0.5},
    )
    d["model"]["max_seq_len"] = 512
    return d


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason="no gloo backend")
def test_distill_two_processes(tmp_path: Path, chat_tok, chat_tok_path, tiny_ckpt) -> None:  # noqa: ANN001
    """--generate-only writes the teacher data in one process; then 2 ranks train with DDP (with logits KD):
    rank 0 mixes and packs, rank 1 waits."""
    from zero.post.distill import run_distill

    d = _distill_config(tmp_path, chat_tok, chat_tok_path, tiny_ckpt)
    meta = run_distill(d, log=lambda _: None, generate_only=True)["meta"]
    assert (tmp_path / "kd.jsonl").exists() and not (tmp_path / "run").exists()  # no training yet
    mp.spawn(_distill_worker, args=(2, _free_port(), d), nprocs=2, join=True)
    assert find_latest(tmp_path / "run" / "ckpt").name == "step_00000002"
    summary = json.loads((tmp_path / "run" / "distill_summary.json").read_text())
    assert summary["meta"]["n_verified"] == meta["n_verified"]  # the training run reused the data


def test_distill_refuses_to_generate_under_torchrun(tmp_path: Path, chat_tok, chat_tok_path, tiny_ckpt) -> None:  # noqa: ANN001
    """Regression: rank 0 used to generate for hours while the other ranks waited in a barrier."""
    from zero.post.common import load_post_config
    from zero.post.distill import DistillConfig, TeacherConfig, _distill
    from zero.train.dist import DistInfo

    d = _distill_config(tmp_path, chat_tok, chat_tok_path, tiny_ckpt)
    cfg, sec = load_post_config(d, {"teacher": TeacherConfig, "distill": DistillConfig})
    with pytest.raises(RuntimeError, match="--generate-only"):
        _distill(cfg, sec["teacher"], sec["distill"], DistInfo(world_size=2), lambda _: None)
    assert not (tmp_path / "kd.jsonl").exists()


def _rank0_worker(rank: int, world: int, port: int, out: str) -> None:
    from zero.post.common import run_on_rank0
    from zero.train.dist import cleanup, init_distributed

    os.environ.update({"RANK": str(rank), "LOCAL_RANK": str(rank), "WORLD_SIZE": str(world),
                       "MASTER_ADDR": "127.0.0.1", "MASTER_PORT": str(port)})
    info = init_distributed("cpu", timeout_min=1)

    def fail() -> None:
        raise ValueError("bad data")

    try:
        try:
            run_on_rank0(info, fail)
            res = "no error"
        except Exception as e:  # noqa: BLE001
            res = f"{type(e).__name__}: {e}"
        Path(f"{out}.{rank}").write_text(res)
        Path(f"{out}.ok{rank}").write_text(str(run_on_rank0(info, lambda: 42)))
    finally:
        cleanup()


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason="no gloo backend")
def test_run_on_rank0_stops_every_rank(tmp_path: Path) -> None:
    out = str(tmp_path / "res")
    mp.spawn(_rank0_worker, args=(2, _free_port(), out), nprocs=2, join=True)
    assert Path(f"{out}.0").read_text() == "ValueError: bad data"
    assert Path(f"{out}.1").read_text().startswith("RuntimeError: rank 0 failed")
    assert Path(f"{out}.ok0").read_text() == "42" and Path(f"{out}.ok1").read_text() == "None"
