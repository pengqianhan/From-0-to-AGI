"""GRPO (Chapter 19).

Group advantages, clipped loss, KL, and the two aggregation methods are equal to hand calculations.
Sampling, and a few steps end to end.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from tests.conftest import post_config
from zero.post.grpo import group_advantages, grpo_loss, run_grpo


def test_group_advantages_hand_computed() -> None:
    r = torch.tensor([[1.0, 0.0, 0.0, 1.0], [0.5, 0.5, 0.5, 0.5], [2.0, 0.0, 1.0, 1.0]])
    a = group_advantages(r, eps=0.0 + 1e-12)
    s1 = math.sqrt(4 * 0.25 / 3)  # unbiased standard deviation
    assert a[0].tolist() == pytest.approx([0.5 / s1, -0.5 / s1, -0.5 / s1, 0.5 / s1], abs=1e-5)
    assert a[1].tolist() == [0.0, 0.0, 0.0, 0.0]  # a zero-variance group gives no gradient signal
    s3 = math.sqrt((1 + 1 + 0 + 0) / 3)
    assert a[2].tolist() == pytest.approx([1 / s3, -1 / s3, 0, 0], abs=1e-5)
    assert group_advantages(r, scale=False)[2].tolist() == [1.0, -1.0, 0.0, 0.0]


def _example():  # noqa: ANN202
    logp = torch.log(torch.tensor([[1.5, 0.9, 1.0], [0.5, 1.1, 1.0]]))
    old = torch.zeros(2, 3)
    adv = torch.tensor([1.0, -1.0])
    mask = torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.bool)
    return logp, old, adv, mask


def test_grpo_loss_clip_hand_computed() -> None:
    logp, old, adv, mask = _example()
    # sequence 1 (A=+1): ρ=1.5 → max(−1.5, −1.2) = −1.2; ρ=0.9 → −0.9; token 3 does not count
    # sequence 2 (A=−1): ρ=0.5 → max(0.5, 0.8) = 0.8; ρ=1.1 → 1.1; ρ=1.0 → 1.0
    per = [[-1.2, -0.9], [0.8, 1.1, 1.0]]
    loss, m = grpo_loss(logp, old, adv, mask, clip_eps=0.2)
    assert loss.item() == pytest.approx(sum(sum(p) for p in per) / 5, abs=1e-6)
    loss, _ = grpo_loss(logp, old, adv, mask, clip_eps=0.2, loss_agg="seq_mean_token_mean")
    assert loss.item() == pytest.approx((sum(per[0]) / 2 + sum(per[1]) / 3) / 2, abs=1e-6)
    assert m["clip_frac"] == pytest.approx(2 / 5)  # 1.5 and 0.5 are clipped
    # clip-higher: with ε_high=0.6, 1.5 is not clipped
    loss, _ = grpo_loss(logp, old, adv, mask, clip_eps=0.2, clip_eps_high=0.6)
    assert loss.item() == pytest.approx((-1.5 - 0.9 + 0.8 + 1.1 + 1.0) / 5, abs=1e-6)


def test_grpo_loss_kl_hand_computed() -> None:
    logp, old, adv, mask = _example()
    ref = torch.zeros(2, 3)
    loss, m = grpo_loss(logp, old, adv, mask, clip_eps=0.2, ref_logp=ref, kl_coef=0.1)
    rs = [[1.5, 0.9], [0.5, 1.1, 1.0]]
    kl = [1 / r + math.log(r) - 1 for row in rs for r in row]  # k3: exp(ref−logp) − (ref−logp) − 1
    base = (-1.2 - 0.9 + 0.8 + 1.1 + 1.0) / 5
    assert loss.item() == pytest.approx(base + 0.1 * sum(kl) / 5, abs=1e-6)
    assert m["kl"] == pytest.approx(sum(kl) / 5, abs=1e-6)


def test_grpo_gradient_on_policy_and_chunking() -> None:
    """With ρ ≡ 1 (right after sampling) and token_mean, the gradient of each response token is −A / total tokens.

    The sum over the chunks is equal to the full batch.
    """
    logp = torch.randn(4, 5).clamp(-3, 0).requires_grad_(True)
    old = logp.detach().clone()
    adv = torch.tensor([1.0, -0.5, 0.0, 2.0])
    mask = torch.rand(4, 5) > 0.3
    loss, _ = grpo_loss(logp, old, adv, mask)
    loss.backward()
    n = mask.sum()
    torch.testing.assert_close(logp.grad, (-adv[:, None] / n) * mask)
    whole, _ = grpo_loss(logp, old, adv, mask)
    parts = [
        grpo_loss(logp[s], old[s], adv[s], mask[s], num_tokens=float(n))[0]
        for s in (slice(0, 2), slice(2, 4))
    ]
    torch.testing.assert_close(sum(parts), whole)
    parts = [
        grpo_loss(logp[s], old[s], adv[s], mask[s], loss_agg="seq_mean_token_mean", num_seqs=4.0)[0]
        for s in (slice(0, 2), slice(2, 4))
    ]
    torch.testing.assert_close(
        sum(parts), grpo_loss(logp, old, adv, mask, loss_agg="seq_mean_token_mean")[0]
    )


def test_sample_group_appends_im_end(monkeypatch, chat_tok, tiny_ckpt) -> None:  # noqa: ANN001
    import zero.generate
    from zero.post.common import load_policy
    from zero.post.envs.tool_env import generate_tasks
    from zero.post.grpo import sample_group

    model, tok = load_policy(tiny_ckpt)
    monkeypatch.setattr(zero.generate, "generate", lambda *a, **k: [[5, 6], [7] * 8])
    _, resps = sample_group(model, tok, generate_tasks(1)[0], 2, 8, 1.0, 1.0, 0)
    assert resps == [[5, 6, tok.im_end_id], [7] * 8]  # a response that stopped early gets <|im_end|> back; a truncated one does not


def test_run_grpo_end_to_end(tmp_path: Path, chat_tok, chat_tok_path, tiny_ckpt) -> None:  # noqa: ANN001
    d = post_config(
        tmp_path,
        chat_tok_path,
        tiny_ckpt,
        chat_tok.vocab_size,
        grpo={
            "group_size": 4,
            "prompts_per_step": 2,
            "max_new_tokens": 12,
            "kl_coef": 0.05,
            "n_train_tasks": 20,
            "ppo_epochs": 2,
            "forward_batch": 3,
        },
    )
    hist = run_grpo(d, log=lambda _: None)
    h = hist[-1]
    assert h["step"] == 2
    for k in (
        "reward_mean",
        "format_rate",
        "call_rate",
        "resp_len",
        "kl",
        "clip_frac",
        "zero_std_groups",
    ):
        assert k in h and math.isfinite(h[k])
    assert 0 < h["resp_len"] <= 13


def test_run_grpo_drops_prompts_that_leave_no_room(tmp_path: Path, chat_tok, chat_tok_path, tiny_ckpt) -> None:  # noqa: ANN001
    """Regression: a prompt longer than max_seq_len − max_new_tokens used to reach the model mid-run.

    The long prompt here fits in max_seq_len (1024) but not with max_new_tokens = 8 more: a check
    against max_seq_len alone would keep it.
    """
    import json

    from tests.conftest import fc_task_with_prompt_len
    from zero.post.envs.fc_tasks import FCTask

    weather = {"type": "function", "function": {"name": "get_weather", "parameters": {
        "type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}
    rows = [
        FCTask(f"t{i}", [weather], [{"role": "user", "content": q}],
               [{"name": "get_weather", "arguments": {"city": "Paris"}}]).to_dict()
        for i, q in enumerate(["Weather in Paris?", "Paris weather, please."])
    ]
    rows.append(fc_task_with_prompt_len(chat_tok, weather, 1024 - 8 + 1, 1023).to_dict())
    (tmp_path / "tasks.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    d = post_config(
        tmp_path, chat_tok_path, tiny_ckpt, chat_tok.vocab_size,
        grpo={"group_size": 2, "prompts_per_step": 2, "max_new_tokens": 8,
              "task_files": [str(tmp_path / "tasks.jsonl")]},
    )
    d["train"]["max_steps"] = 1
    logs: list[str] = []
    assert len(run_grpo(d, log=logs.append)) == 1
    assert any("dropped 1 of 3 tasks" in m for m in logs)
    d["grpo"]["prompts_per_step"] = 3
    with pytest.raises(ValueError, match="usable tasks"):
        run_grpo(d, log=lambda _: None)
