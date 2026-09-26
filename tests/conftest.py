"""pytest 公共设置与小工具。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

# CPU 被其他进程占用时，多线程会互相抢核，反而慢几十倍；测试统一用单线程，结果也更稳定
torch.set_num_threads(1)

REPO = Path(__file__).resolve().parent.parent
TINY_CORPUS = REPO / "assets" / "tiny_corpus"


def write_random_shards(
    out_dir: Path, name: str, n_shards: int, tokens_per_shard: int, vocab: int, seed: int
) -> str:
    """写几个随机 token 分片（带一点可学习的结构：下一个 token 常常是当前 token + 1），返回 glob。"""
    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n_shards):
        x = rng.integers(0, vocab, size=tokens_per_shard)
        mask = rng.random(tokens_per_shard) < 0.7
        for t in range(1, tokens_per_shard):
            if mask[t]:
                x[t] = (x[t - 1] + 1) % vocab
        x.astype(np.uint32).tofile(out_dir / f"{name}_{i:05d}.bin")
    return str(out_dir / f"{name}_*.bin")


@pytest.fixture(scope="session")
def tiny_texts() -> dict[str, str]:
    """从 tiny_corpus 里各取一小段中文、英文、代码。"""
    return {
        "en": (TINY_CORPUS / "shakespeare.txt").read_text("utf-8")[:60_000],
        "zh": (TINY_CORPUS / "chinese_poetry.txt").read_text("utf-8")[:30_000],
        "code": (TINY_CORPUS / "code.txt").read_text("utf-8")[:60_000],
    }


@pytest.fixture
def random_shards():  # noqa: ANN201
    """返回 write_random_shards 函数本身，方便测试里调用。"""
    return write_random_shards


def small_train_config(
    out_dir: Path, sources: dict[str, str], val: str = "", **train_overrides: object
):  # noqa: ANN201
    """构造一个几秒钟就能训完的 Config（随机分片数据、极小模型、CPU、FP32）。"""
    from zero.config import config_from_dict

    train = {
        "seed": 0,
        "max_steps": 8,
        "micro_batch_size": 4,
        "grad_accum_steps": 2,
        "device": "cpu",
        "dtype": "fp32",
        "out_dir": str(out_dir),
        "eval_every": 4 if val else 0,
        "eval_batches": 2,
    }
    train.update(train_overrides)
    return config_from_dict(
        {
            "model": {
                "vocab_size": 64,
                "dim": 32,
                "n_layers": 2,
                "n_heads": 4,
                "n_kv_heads": 2,
                "ffn_dim": 64,
                "max_seq_len": 64,
            },
            "train": train,
            "data": {
                "seq_len": 32,
                "val": val,
                "sources": [
                    {"name": k, "path": v, "weight": 1.0 + i}
                    for i, (k, v) in enumerate(sources.items())
                ],
            },
            "optim": {"lr": 3e-3, "weight_decay": 0.1},
            "schedule": {"kind": "cosine", "warmup_steps": 2, "min_lr_ratio": 0.1},
            "checkpoint": {"every": 4, "keep_last": 0},
            "logging": {"every": 1},
        }
    )


@pytest.fixture
def make_config():  # noqa: ANN201
    return small_train_config
