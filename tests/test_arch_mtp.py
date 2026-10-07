"""Correctness of MTP (zero/arch/mtp.py, Chapter 25): shapes, loss agrees with a hand computation,
gradient flow, causality, and MTP self-speculative decoding (greedy) is identical to greedy
decoding of the main model.
"""

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
    # The main-model logits are identical to Transformer.forward (MTP does not change the main model).
    assert torch.allclose(logits, model.model(tokens), atol=1e-6)


def test_loss_matches_hand_computation() -> None:
    """Depth 1: the MTP target at position i is tokens[i+2]; total loss = main loss + λ · MTP loss."""
    model = _model()
    x = torch.randint(0, 41, (2, 10))
    tokens, targets = x[:, :-1], x[:, 1:]
    total, main, losses = mtp_loss(model, tokens, targets, lam=0.3)
    logits, (lg1,) = model(tokens)
    hand_main = F.cross_entropy(logits.reshape(-1, 41), targets.reshape(-1))
    hand_mtp = F.cross_entropy(lg1.reshape(-1, 41), x[:, 2:].reshape(-1))  # the token after the next
    assert torch.isfinite(total)
    assert torch.allclose(main, hand_main, atol=1e-6)
    assert torch.allclose(losses[0], hand_mtp, atol=1e-6)
    assert torch.allclose(total, hand_main + 0.3 * hand_mtp, atol=1e-6)


def test_mtp_is_causal_and_uses_next_token_embedding() -> None:
    """The MTP output at position i depends only on tokens[0..i+1].

    A change of tokens from i+2 on has no effect on it. A change of token i+1 has an effect.
    """
    model = _model().eval()
    a = torch.randint(0, 41, (1, 12))
    b = a.clone()
    b[0, 7:] = (b[0, 7:] + 1) % 41
    _, (la,) = model(a)
    _, (lb,) = model(b)
    assert torch.allclose(la[0, :6], lb[0, :6], atol=1e-6)  # positions 0..5 see only tokens[..6]
    assert not torch.allclose(la[0, 6], lb[0, 6])  # position 6 uses Emb(tokens[7])


def test_gradient_flows_to_mtp_and_shared_weights() -> None:
    model = _model()
    x = torch.randint(0, 41, (2, 10))
    _, _, losses = mtp_loss(model, x[:, :-1], x[:, 1:])
    losses[0].backward()
    mtp = model.mtp[0]
    for p in (mtp.eh_proj.weight, mtp.block.attn.wq.weight, mtp.enorm.weight, mtp.hnorm.weight):
        assert p.grad is not None and p.grad.abs().sum() > 0
    # The shared embedding / output head and the main-model layers all get gradients from the MTP loss
    # (a "denser" training signal).
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
    assert res.rounds + res.accepted + 1 >= len(ref)  # prefill gives 1; each round gives 1 + number accepted


def test_self_speculative_sampling_runs() -> None:
    model = _model()
    res = mtp_speculative_generate(model, [1, 2, 3], 25, temperature=1.0, top_p=0.95, seed=3)
    again = mtp_speculative_generate(model, [1, 2, 3], 25, temperature=1.0, top_p=0.95, seed=3)
    assert res.tokens == again.tokens and len(res.tokens) == 25
    assert 0.0 <= res.acceptance_rate <= 1.0
