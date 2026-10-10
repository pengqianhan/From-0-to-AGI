"""Cross-stage on-policy distillation (Chapter 17) and the HF import of the proxy base.

The sampled loss has the gradient of the reverse KL in expectation (exact enumeration). A run with the
student as its own teacher has KL 0. Two teachers, both losses, end to end. A teacher with a different
tokenizer is refused. An imported HF model gives the same logits as the original.
"""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path

import pytest
import torch

from tests.conftest import TINY_POST_MODEL, post_config
from zero.post.distill import reverse_kl_loss
from zero.post.opd import conversation_prompt, pick_prompts, run_opd, sampled_opd_loss
from zero.train.checkpoint import find_latest


def test_sampled_loss_is_reverse_kl_gradient_in_expectation() -> None:
    torch.manual_seed(0)
    V = 7
    s_logits = torch.randn(V, requires_grad=True)
    t_logits = torch.randn(V)
    # Exact gradient of KL(p_S ‖ p_T) at one position
    kl = reverse_kl_loss(s_logits.view(1, 1, V), t_logits.view(1, 1, V), torch.ones(1, 1))
    (g_exact,) = torch.autograd.grad(kl, s_logits)
    # E_{y ~ p_S}[∇ sampled loss], by enumeration of all y
    p_s = torch.softmax(s_logits.detach(), -1)
    g_exp = torch.zeros(V)
    for y in range(V):
        lp_s = torch.log_softmax(s_logits, -1)[y].view(1, 1)
        lp_t = torch.log_softmax(t_logits, -1)[y].view(1, 1)
        (g,) = torch.autograd.grad(sampled_opd_loss(lp_s, lp_t, torch.ones(1, 1), 1.0), s_logits)
        g_exp += p_s[y] * g
    assert torch.allclose(g_exp, g_exact, atol=1e-6)


def test_sampled_loss_zero_when_teacher_equals_student() -> None:
    lp = torch.log(torch.tensor([[0.2, 0.5, 0.9]])).requires_grad_(True)
    loss = sampled_opd_loss(lp, lp.detach(), torch.tensor([[True, True, False]]), 2.0)
    loss.backward()
    assert float(loss.detach()) == 0.0 and torch.all(lp.grad == 0)


def test_conversation_prompt_drops_last_assistant() -> None:
    row = {
        "messages": [
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "b"},
            {"role": "user", "content": "c"},
            {"role": "assistant", "content": "d"},
        ],
        "tools": [{"type": "function"}],
    }
    p = conversation_prompt(row)
    assert [m["content"] for m in p.messages] == ["a", "b", "c"] and p.tools == row["tools"]
    assert conversation_prompt({"messages": [{"role": "assistant", "content": "x"}]}) is None
    assert len(conversation_prompt({"messages": [{"role": "user", "content": "q"}]}).messages) == 1
    # regression: a conversation that ends with a user turn keeps that turn (it is the one to answer)
    open_turn = {"messages": row["messages"][:3]}
    assert [m["content"] for m in conversation_prompt(open_turn).messages] == ["a", "b", "c"]
    # a tool trajectory: the prompt ends with the tool result, the final answer is dropped
    traj = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "", "tool_calls": [{"name": "f", "arguments": {}}]},
        {"role": "tool", "content": "{}"},
        {"role": "assistant", "content": "done"},
    ]
    assert [m["role"] for m in conversation_prompt({"messages": traj}).messages] == [
        "user",
        "assistant",
        "tool",
    ]
    assert conversation_prompt({"messages": [{"role": "system", "content": "s"}]}) is None


def test_pick_prompts_follows_weights() -> None:
    import random

    picks = pick_prompts([[0] * 5, [0] * 3], [0.0, 1.0], 20, random.Random(0))
    assert all(k == 1 and 0 <= i < 3 for k, i in picks)


@pytest.fixture(scope="module")
def other_ckpt(tmp_path_factory, chat_tok, chat_tok_path):  # noqa: ANN001, ANN201
    """A second tiny model (different weights, same tokenizer) as a teacher."""
    from zero.config import ModelConfig
    from zero.model import Transformer
    from zero.train.checkpoint import save_checkpoint

    torch.manual_seed(1)
    cfg = ModelConfig(vocab_size=chat_tok.vocab_size, **TINY_POST_MODEL)
    root = tmp_path_factory.mktemp("other_ckpt") / "ckpt"
    save_checkpoint(
        root,
        0,
        Transformer(cfg),
        meta={
            "config": {
                "model": dataclasses.asdict(cfg),
                "train": {"data": {"tokenizer": str(chat_tok_path)}},
            }
        },
    )
    return root


def _opd(tmp_path: Path, tok_path, init, vocab, teachers, **opd):  # noqa: ANN001, ANN003, ANN202
    return post_config(
        tmp_path,
        tok_path,
        init,
        vocab,
        opd={
            "prompts_per_step": 3,
            "max_new_tokens": 10,
            "n_tool_tasks": 20,
            "forward_batch": 2,
            "teachers": teachers,
            **opd,
        },
    )


def test_self_teacher_has_zero_kl(tmp_path: Path, chat_tok, chat_tok_path, tiny_ckpt) -> None:  # noqa: ANN001
    d = _opd(
        tmp_path,
        chat_tok_path,
        tiny_ckpt,
        chat_tok.vocab_size,
        [{"name": "self", "path": str(tiny_ckpt)}],
    )
    d["train"]["max_steps"] = 1
    h = run_opd(d, log=lambda _: None)[0]
    assert h["kl"] == pytest.approx(0.0, abs=1e-6)
    assert h["loss"] == pytest.approx(0.0, abs=1e-6) and h["grad_norm"] == pytest.approx(
        0.0, abs=1e-6
    )


@pytest.mark.parametrize("loss", ["full_kl", "sampled"])
def test_run_opd_two_teachers_end_to_end(
    tmp_path: Path, chat_tok, chat_tok_path, tiny_ckpt, other_ckpt, loss: str
) -> None:  # noqa: ANN001
    from zero.post.common import write_jsonl
    from zero.post.sft import env_conversations

    write_jsonl(tmp_path / "sft.jsonl", env_conversations(5, 0, "train"))
    d = _opd(
        tmp_path,
        chat_tok_path,
        tiny_ckpt,
        chat_tok.vocab_size,
        [
            {
                "name": "chat",
                "path": str(other_ckpt),
                "prompts": str(tmp_path / "sft.jsonl"),
                "weight": 1.0,
            },
            {"name": "tool", "path": str(tiny_ckpt), "prompts": "tool_env", "weight": 1.0},
        ],
        loss=loss,
        prompts_per_step=6,
    )
    hist = run_opd(d, log=lambda _: None)
    h = hist[-1]
    assert h["step"] == 2 and h["n_seqs"] == 6
    for k in ("loss", "kl", "resp_len", "eos_rate", "grad_norm"):
        assert math.isfinite(h[k])
    assert any(k.startswith("kl/") for k in h)
    assert 0 < h["resp_len"] <= 11
    assert find_latest(tmp_path / "run" / "ckpt").name == "step_00000002"
    assert (tmp_path / "run" / "opd_summary.json").exists()


def test_opd_refuses_teacher_with_other_tokenizer(
    tmp_path: Path, chat_tok, chat_tok_path, tiny_ckpt, tiny_texts
) -> None:  # noqa: ANN001
    from zero.tokenizer import train_bpe

    other = train_bpe([tiny_texts["en"][:5000]], vocab_size=chat_tok.vocab_size)
    other.save(tmp_path / "other.json")
    d = _opd(
        tmp_path,
        tmp_path / "other.json",
        tiny_ckpt,
        chat_tok.vocab_size,
        [{"name": "t", "path": str(tiny_ckpt)}],
    )
    with pytest.raises(ValueError, match="different tokenizer"):
        run_opd(d, log=lambda _: None)


def test_import_hf_roundtrip(tmp_path: Path, chat_tok, tiny_ckpt) -> None:  # noqa: ANN001
    from zero.hf import export_to_hf_qwen3
    from zero.post.common import load_policy
    from zero.tools.import_hf import import_hf

    model, tok = load_policy(tiny_ckpt)
    hf_dir = export_to_hf_qwen3(model, None, tmp_path / "hf", tokenizer=tok, dtype=torch.float32)
    r = import_hf(hf_dir, tmp_path / "base")
    m2, t2 = load_policy(tmp_path / "base")
    assert t2.hash() == tok.hash()
    x = torch.randint(0, tok.vocab_size, (1, 12))
    with torch.no_grad():
        assert torch.allclose(model(x), m2(x), atol=1e-5)
    assert Path(r["tokenizer"]).exists()

    # --check-config: a config with a different shape is refused before any training
    bad = tmp_path / "bad.toml"
    bad.write_text(
        "[model]\n"
        + "".join(
            f"{k} = {v!r}\n".replace("False", "false").replace("True", "true")
            for k, v in {**TINY_POST_MODEL, "vocab_size": tok.vocab_size, "dim": 64}.items()
        )
    )
    with pytest.raises(ValueError, match="dim"):
        import_hf(hf_dir, tmp_path / "base2", check_config=bad)
