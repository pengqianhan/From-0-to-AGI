"""SFT (Chapter 16).

The loss mask is equal to a hand calculation. Packing does not cut a conversation. The window loader
can resume. A few training steps end to end.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from tests.conftest import post_config
from zero.config import ModelConfig
from zero.data.loader import MaskedWindowLoader
from zero.model import Transformer
from zero.post.sft import encode_example, env_conversations, pack_examples, run_sft, write_windows


def test_masked_loss_matches_hand_computation() -> None:
    """The positions with -100 in y do not count: loss = mean of -log p at the assistant positions."""
    torch.manual_seed(0)
    m = Transformer(
        ModelConfig(
            vocab_size=11, dim=16, n_layers=1, n_heads=2, n_kv_heads=1, ffn_dim=16, max_seq_len=16
        )
    )
    x = torch.tensor([[1, 5, 7, 2, 3, 4]])
    y_full = torch.tensor([[5, 7, 2, 3, 4, 9]])
    keep = torch.tensor([[False, True, True, False, False, True]])
    y = torch.where(keep, y_full, torch.full_like(y_full, -100))
    got = m.loss(x, y)
    logp = F.log_softmax(m(x).float(), dim=-1)[0]
    hand = -(logp[1, 7] + logp[2, 2] + logp[5, 9]) / 3
    torch.testing.assert_close(got, hand)


def test_pack_examples_keeps_conversations_whole() -> None:
    ex = [
        ([1] * 5, [False, True, True, True, True]),
        ([2] * 7, [False] * 3 + [True] * 4),
        ([3] * 4, [False, False, True, True]),
        ([4] * 20, [True] * 20),
        ([5] * 3, [False] * 3),
    ]
    toks, masks, stats = pack_examples(ex, window=10, pad_id=0)
    assert (
        stats["dropped_too_long"] == 1
        and stats["dropped_no_target"] == 1
        and stats["examples"] == 3
    )
    assert toks.shape == (2, 10)
    # first window: 5 × 1 + 4 × 3 + 1 padding; second window: 7 × 2 + 3 padding
    assert toks[0].tolist() == [1] * 5 + [3] * 4 + [0]
    assert masks[0].tolist() == [0, 1, 1, 1, 1, 0, 0, 1, 1, 0]
    assert toks[1].tolist() == [2] * 7 + [0] * 3
    for row in toks:  # all tokens of one conversation are in one window
        for v in (1, 2, 3):
            assert (row == v).sum() in (0, {1: 5, 2: 7, 3: 4}[v])


def test_window_loader_targets_and_resume(tmp_path: Path) -> None:
    toks = np.arange(4 * 9, dtype=np.uint32).reshape(4, 9)
    masks = np.zeros((4, 9), dtype=np.uint8)
    masks[:, 5:] = 1
    write_windows(toks, masks, tmp_path / "train.bin")
    ld = MaskedWindowLoader(str(tmp_path / "train.bin"), seq_len=8, batch_size=2, seed=1)
    x, y = ld.next_batch()
    assert x.shape == (2, 8)
    for i in range(2):
        row = x[i].tolist()
        assert (y[i, :4] == -100).all() and y[i, 4:].tolist() == [row[0] + k for k in range(5, 9)]
    state = ld.state_dict()
    a = ld.next_batch()
    ld2 = MaskedWindowLoader(str(tmp_path / "train.bin"), seq_len=8, batch_size=2, seed=1)
    ld2.load_state_dict(state)
    b = ld2.next_batch()
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


def test_encode_example_mask_matches_render(chat_tok) -> None:  # noqa: ANN001
    ex = env_conversations(3, seed=0, split="train")[0]
    ids, mask = encode_example(ex, chat_tok)
    assert ids[-1] == chat_tok.eot_id and mask[-1] is False
    txt = chat_tok.decode([i for i, m in zip(ids, mask) if m])
    assert txt.endswith("<|im_end|>") and txt.count("<|im_end|>") == sum(
        m["role"] == "assistant" for m in ex["messages"]
    )


def test_run_sft_end_to_end_and_resume(tmp_path: Path, chat_tok, chat_tok_path, tiny_ckpt) -> None:  # noqa: ANN001
    d = post_config(
        tmp_path,
        chat_tok_path,
        tiny_ckpt,
        chat_tok.vocab_size,
        data={"format": "sft", "seq_len": 512},
        train={"max_steps": 2, "eval_every": 2, "eval_batches": 1},
        checkpoint={"every": 1, "keep_last": 0},
        sft={
            "train_jsonl": str(tmp_path / "train.jsonl"),
            "val_jsonl": str(tmp_path / "val.jsonl"),
            "shard_dir": str(tmp_path / "data"),
            "generate_train": 12,
            "generate_val": 4,
        },
    )
    d["model"]["max_seq_len"] = 512
    hist = run_sft(d, log=lambda _: None)
    assert (
        hist[-1]["step"] == 2 and np.isfinite(hist[-1]["loss"]) and hist[-1]["val_loss"] is not None
    )
    assert (tmp_path / "data" / "train.mask").exists()
    # resume: same folder, max_steps set to 3, so only 1 more step
    d["train"]["max_steps"] = 3
    hist2 = run_sft(d, log=lambda _: None)
    assert [h["step"] for h in hist2] == [3]


def test_sft_requires_format() -> None:
    with pytest.raises(ValueError):
        run_sft({"model": {}, "train": {}, "data": {"format": "none"}}, log=lambda _: None)
