"""Offline difficulty filter (Chapter 19): keep rule, outputs, and 2 processes == 1 process."""

from __future__ import annotations

import json
import os
import socket
from pathlib import Path

import pytest
import torch.distributed as dist
import torch.multiprocessing as mp

from zero.post.difficulty import keep_decision, run_filter
from zero.post.envs.fc_tasks import FCTask, load_fc_tasks

WEATHER = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    },
}


def _tasks(path: Path, n: int = 6) -> str:
    rows = [
        FCTask(f"t{i}", [WEATHER], [{"role": "user", "content": f"Weather in C{i}?"}],
               [{"name": "get_weather", "arguments": {"city": f"C{i}"}}]).to_dict()
        for i in range(n)
    ]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return str(path)


def test_keep_decision() -> None:
    stats = [{"id": "a", "pass": 0.0}, {"id": "b", "pass": 0.25}, {"id": "c", "pass": 1.0}, {"id": "d", "pass": 0.875}]
    keep, why = keep_decision(stats, 1 / 8, 7 / 8, 0.0, seed=0)
    assert keep == {"b", "d"} and why == {"kept": 2, "too_easy": 1, "too_hard": 1}
    keep, why = keep_decision(stats, 1 / 8, 7 / 8, 1.0, seed=0)
    assert keep == {"a", "b", "d"} and why["kept_hard"] == 1


def test_run_filter_outputs(tmp_path: Path, tiny_ckpt) -> None:  # noqa: ANN001
    src = _tasks(tmp_path / "tasks.jsonl")
    out = tmp_path / "kept.jsonl"
    meta = run_filter(str(tiny_ckpt), src, str(out), k=3, keep_hard=1.0, max_new_tokens=6, device="cpu", log=lambda _: None)
    stats = [json.loads(x) for x in Path(f"{out}.stats.jsonl").read_text().splitlines()]
    assert [s["id"] for s in stats] == [f"t{i}" for i in range(6)]
    assert all(0.0 <= s["pass"] <= 1.0 and -1.0 <= s["mean_reward"] <= 1.0 for s in stats)
    assert meta["n_tasks"] == 6 and sum(meta["pass_histogram"].values()) == 6
    assert meta["settings"]["min_pass"] == pytest.approx(1 / 3)
    # a random tiny model never answers exactly: everything is "too hard", and keep_hard=1 keeps it all
    assert meta.get("kept_hard", 0) + meta.get("kept", 0) == len(load_fc_tasks(out))
    assert not list(tmp_path.glob("kept.jsonl.part*"))  # the per-rank parts are merged and removed


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _worker(rank: int, world: int, port: int, ckpt: str, src: str, out: str) -> None:
    import torch

    torch.set_num_threads(1)
    os.environ.update({"RANK": str(rank), "LOCAL_RANK": str(rank), "WORLD_SIZE": str(world),
                       "MASTER_ADDR": "127.0.0.1", "MASTER_PORT": str(port)})
    run_filter(ckpt, src, out, k=3, keep_hard=0.5, max_new_tokens=6, device="cpu", log=lambda _: None)


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason="no gloo backend")
def test_two_processes_same_result(tmp_path: Path, tiny_ckpt) -> None:  # noqa: ANN001
    src = _tasks(tmp_path / "tasks.jsonl", n=5)
    one, two = tmp_path / "one.jsonl", tmp_path / "two.jsonl"
    run_filter(str(tiny_ckpt), src, str(one), k=3, keep_hard=0.5, max_new_tokens=6, device="cpu", log=lambda _: None)
    mp.spawn(_worker, args=(2, _free_port(), str(tiny_ckpt), src, str(two)), nprocs=2, join=True)
    assert Path(f"{one}.stats.jsonl").read_text() == Path(f"{two}.stats.jsonl").read_text()
    assert one.read_text() == two.read_text()
