"""MTP（zero/arch/mtp.py，第 25 章）的正确性：形状、损失与手算一致、梯度流动、因果性、
MTP 自推测解码（贪心）与主模型贪心解码逐字相同。"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from zero.arch.mtp import MTPTransformer, mtp_loss, mtp_speculative_generate
from zero.config import ModelConfig
from zero.generate import generate


def _model(n_mtp: int = 1, seed: int = 0) -> MTPTransformer:
    torch.manual_seed(seed)
    cfg = ModelConfig(
        vocab_size=41,
        dim=32,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        head_dim=8,
        ffn_dim=64,
        max_seq_len=96,
        init_std=0.2,
    )
    return MTPTransformer(cfg, n_mtp=n_mtp)


def test_shapes() -> None:
    model = _model(n_mtp=2)
    tokens = torch.randint(0, 41, (3, 12))
    logits, mtp_logits = model(tokens)
    assert logits.shape == (3, 12, 41)
    assert [t.shape for t in mtp_logits] == [(3, 11, 41), (3, 10, 41)]
    # 主模型 logits 与 Transformer.forward 完全一致（MTP 不改变主模型）
    assert torch.allclose(logits, model.model(tokens), atol=1e-6)


def test_loss_matches_hand_computation() -> None:
    """深度 1：MTP 在位置 i 的目标是 tokens[i+2]；总损失 = 主损失 + λ · MTP 损失。"""
    model = _model()
    x = torch.randint(0, 41, (2, 10))
    tokens, targets = x[:, :-1], x[:, 1:]
    total, main, losses = mtp_loss(model, tokens, targets, lam=0.3)
    logits, (lg1,) = model(tokens)
    hand_main = F.cross_entropy(logits.reshape(-1, 41), targets.reshape(-1))
    hand_mtp = F.cross_entropy(lg1.reshape(-1, 41), x[:, 2:].reshape(-1))  # 下下个 token
    assert torch.isfinite(total)
    assert torch.allclose(main, hand_main, atol=1e-6)
    assert torch.allclose(losses[0], hand_mtp, atol=1e-6)
    assert torch.allclose(total, hand_main + 0.3 * hand_mtp, atol=1e-6)


def test_mtp_is_causal_and_uses_next_token_embedding() -> None:
    """位置 i 的 MTP 输出只依赖 tokens[0..i+1]：改动 i+2 之后的 token 不影响它，改动 i+1 会影响。"""
    model = _model().eval()
    a = torch.randint(0, 41, (1, 12))
    b = a.clone()
    b[0, 7:] = (b[0, 7:] + 1) % 41
    _, (la,) = model(a)
    _, (lb,) = model(b)
    assert torch.allclose(la[0, :6], lb[0, :6], atol=1e-6)  # 位置 0..5 只看到 tokens[..6]
    assert not torch.allclose(la[0, 6], lb[0, 6])  # 位置 6 用了 Emb(tokens[7])


def test_gradient_flows_to_mtp_and_shared_weights() -> None:
    model = _model()
    x = torch.randint(0, 41, (2, 10))
    _, _, losses = mtp_loss(model, x[:, :-1], x[:, 1:])
    losses[0].backward()
    mtp = model.mtp[0]
    for p in (mtp.eh_proj.weight, mtp.block.attn.wq.weight, mtp.enorm.weight, mtp.hnorm.weight):
        assert p.grad is not None and p.grad.abs().sum() > 0
    # 共享的 embedding / 输出头、以及主模型的层都收到了 MTP 损失的梯度（训练信号"加密"）
    assert model.model.tok_emb.weight.grad.abs().sum() > 0
    assert model.model.lm_head.weight.grad.abs().sum() > 0
    assert model.model.layers[0].attn.wq.weight.grad.abs().sum() > 0


def test_training_steps_reduce_loss() -> None:
    model = _model()
    torch.manual_seed(0)
    x = torch.randint(0, 41, (4, 17))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    first = None
    for _ in range(30):
        total, _, _ = mtp_loss(model, x[:, :-1], x[:, 1:])
        opt.zero_grad()
        total.backward()
        opt.step()
        first = first if first is not None else total.item()
    assert total.item() < first


def test_self_speculative_greedy_matches_main_greedy() -> None:
    model = _model().eval()
    prompt = [3, 9, 27, 4, 12]
    ref = generate(model.model, prompt, 30, temperature=0.0)
    res = mtp_speculative_generate(model, prompt, 30, temperature=0.0)
    assert res.tokens == ref
    assert res.rounds + res.accepted + 1 >= len(ref)  # prefill 给 1 个，每轮 1 + 接受数


def test_self_speculative_sampling_runs() -> None:
    model = _model()
    res = mtp_speculative_generate(model, [1, 2, 3], 25, temperature=1.0, top_p=0.95, seed=3)
    again = mtp_speculative_generate(model, [1, 2, 3], 25, temperature=1.0, top_p=0.95, seed=3)
    assert res.tokens == again.tokens and len(res.tokens) == 25
    assert 0.0 <= res.acceptance_rate <= 1.0
