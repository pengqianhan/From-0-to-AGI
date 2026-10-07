"""Shared pytest setup and small helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

# When other processes use the CPU, many threads compete for the cores and can be tens of times slower.
# All tests use one thread. This also makes the results more stable.
torch.set_num_threads(1)

REPO = Path(__file__).resolve().parent.parent
TINY_CORPUS = REPO / "assets" / "tiny_corpus"


def write_random_shards(
    out_dir: Path, name: str, n_shards: int, tokens_per_shard: int, vocab: int, seed: int
) -> str:
    """Write some random token shards and return their glob.

    The shards have a small learnable structure: the next token is often the current token + 1.
    """
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
    """Take a short piece of Chinese, English, and code from tiny_corpus."""
    return {
        "en": (TINY_CORPUS / "shakespeare.txt").read_text("utf-8")[:60_000],
        "zh": (TINY_CORPUS / "chinese_poetry.txt").read_text("utf-8")[:30_000],
        "code": (TINY_CORPUS / "code.txt").read_text("utf-8")[:60_000],
    }


@pytest.fixture
def random_shards():  # noqa: ANN201
    """Return the write_random_shards function itself, so that tests can call it."""
    return write_random_shards


def small_train_config(
    out_dir: Path, sources: dict[str, str], val: str = "", **train_overrides: object
):  # noqa: ANN201
    """Make a Config that trains in a few seconds (random shard data, tiny model, CPU, FP32)."""
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


# ---------------------------------------------------------------------------
# Shared by the post-training tests: a small tokenizer that saw the chat / tool-calling format,
# and a checkpoint of a tiny model with random initialization
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def chat_tok_path(tmp_path_factory, tiny_texts):  # noqa: ANN001, ANN201
    from zero.post.chat import render_text
    from zero.post.envs.tool_env import generate_tasks, reference_messages
    from zero.tokenizer import train_bpe

    convs = [render_text(reference_messages(t), t.tools) for t in generate_tasks(300, seed=5)]
    tok = train_bpe([tiny_texts["en"][:20000], tiny_texts["zh"][:10000], *convs], vocab_size=600)
    path = tmp_path_factory.mktemp("chat_tok") / "tokenizer.json"
    tok.save(path)
    return path


@pytest.fixture(scope="session")
def chat_tok(chat_tok_path):  # noqa: ANN001, ANN201
    from zero.tokenizer import Tokenizer

    return Tokenizer.load(chat_tok_path)


TINY_POST_MODEL = {
    "dim": 32,
    "n_layers": 2,
    "n_heads": 4,
    "n_kv_heads": 2,
    "ffn_dim": 64,
    "max_seq_len": 1024,
    "tie_embeddings": False,
}


@pytest.fixture(scope="session")
def tiny_ckpt(tmp_path_factory, chat_tok, chat_tok_path):  # noqa: ANN001, ANN201
    """Save a tiny model with random initialization as a zero checkpoint.

    The meta records the tokenizer path, so load_policy can read the checkpoint directly.
    """
    from zero.config import ModelConfig
    from zero.model import Transformer
    from zero.train.checkpoint import save_checkpoint

    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=chat_tok.vocab_size, **TINY_POST_MODEL)
    model = Transformer(cfg)
    root = tmp_path_factory.mktemp("tiny_ckpt") / "ckpt"
    import dataclasses

    save_checkpoint(
        root,
        0,
        model,
        meta={
            "config": {
                "model": dataclasses.asdict(cfg),
                "train": {"data": {"tokenizer": str(chat_tok_path)}},
            }
        },
    )
    return root


def post_config(tmp_path: Path, tok_path: Path, init_from: Path, vocab: int, **sections):  # noqa: ANN003, ANN201
    """Make a config dict for the post-training stages (the model shape is the same as tiny_ckpt)."""
    d = {
        "model": {"vocab_size": vocab, **TINY_POST_MODEL},
        "train": {
            "seed": 0,
            "max_steps": 2,
            "micro_batch_size": 2,
            "device": "cpu",
            "dtype": "fp32",
            "out_dir": str(tmp_path / "run"),
            "init_from": str(init_from),
            "eval_every": 0,
        },
        "data": {"format": "none", "tokenizer": str(tok_path), "seq_len": 1024},
        "optim": {"lr": 1e-3, "weight_decay": 0.0},
        "schedule": {"kind": "constant", "warmup_steps": 0},
        "checkpoint": {"every": 0, "keep_last": 1},
        "logging": {"every": 1},
    }
    for k, v in sections.items():
        if k in d and isinstance(v, dict):
            d[k].update(v)
        else:
            d[k] = v
    return d
