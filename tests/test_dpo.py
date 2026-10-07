"""DPO (Chapter 18).

The loss is equal to a hand calculation. The sequence log probability counts only the response.
A few steps end to end.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from tests.conftest import post_config
from zero.post.chat import encode_prompt_response
from zero.post.dpo import PreferencePair, batch_logps, dpo_loss, encode_pair, run_dpo


def test_dpo_loss_hand_computed() -> None:
    pc = torch.tensor([-1.0, -3.0])
    pr = torch.tensor([-2.0, -1.0])
    rc = torch.tensor([-1.5, -2.0])
    rr = torch.tensor([-1.5, -2.0])
    loss, m = dpo_loss(pc, pr, rc, rr, beta=0.5)
    # pair 1: β·((−1+1.5) − (−2+1.5)) = 0.5 → −log σ(0.5); pair 2: β·((−3+2) − (−1+2)) = −1 → −log σ(−1)
    hand = (math.log(1 + math.exp(-0.5)) + math.log(1 + math.exp(1.0))) / 2
    assert loss.item() == pytest.approx(hand, abs=1e-6)
    assert m["acc"] == 0.5
    assert m["margin"] == pytest.approx((0.5 - 1.0) / 2)
    assert m["chosen_reward"] == pytest.approx((0.25 - 0.5) / 2)
    # with policy == ref, the loss is log 2; the gradient pushes chosen up and rejected down
    x = torch.zeros(3, requires_grad=True)
    y = torch.zeros(3, requires_grad=True)
    loss, _ = dpo_loss(x, y, torch.zeros(3), torch.zeros(3), beta=0.1)
    assert loss.item() == pytest.approx(math.log(2))
    loss.backward()
    assert (x.grad < 0).all() and (y.grad > 0).all()


def test_batch_logps_only_counts_response(chat_tok, tiny_ckpt) -> None:  # noqa: ANN001
    from zero.post.common import load_policy

    model, tok = load_policy(tiny_ckpt)
    msgs = [{"role": "user", "content": "hi"}]
    c_ids, c_mask = encode_prompt_response(msgs, {"role": "assistant", "content": "hello"}, tok)
    r_ids, r_mask = encode_prompt_response(msgs, {"role": "assistant", "content": "no"}, tok)
    pair = PreferencePair(c_ids, c_mask, r_ids, r_mask)
    with torch.no_grad():
        c, r = batch_logps(model, [pair], tok.eot_id, torch.device("cpu"))
        logp = F.log_softmax(model(torch.tensor([c_ids[:-1]])).float(), -1)[0]
        hand = sum(logp[t - 1, c_ids[t]] for t in range(1, len(c_ids)) if c_mask[t])
    assert c.item() == pytest.approx(float(hand), abs=1e-4)
    assert sum(c_mask) == len(tok.encode("hello<|im_end|>"))
    assert (
        encode_pair(
            {
                "messages": msgs,
                "chosen": {"role": "assistant", "content": "x" * 50},
                "rejected": {"role": "assistant", "content": "y"},
            },
            tok,
            max_len=10,
        )
        is None
    )


@pytest.mark.parametrize("ref_mode", ["precompute", "online"])
def test_run_dpo_end_to_end(
    tmp_path: Path, chat_tok, chat_tok_path, tiny_ckpt, ref_mode: str
) -> None:  # noqa: ANN001
    d = post_config(
        tmp_path,
        chat_tok_path,
        tiny_ckpt,
        chat_tok.vocab_size,
        dpo={
            "train_jsonl": str(tmp_path / "prefs.jsonl"),
            "generate_pairs": 4,
            "samples_per_prompt": 2,
            "max_new_tokens": 12,
            "beta": 0.1,
            "ref_mode": ref_mode,
        },
        train={"max_steps": 3},
    )
    hist = run_dpo(d, log=lambda _: None)
    assert hist[0]["loss"] == pytest.approx(math.log(2), abs=1e-5)  # first step: policy == ref
    assert hist[-1]["step"] == 3 and hist[-1]["loss"] < hist[0]["loss"]
    assert (tmp_path / "run" / "ckpt" / "latest").exists()
