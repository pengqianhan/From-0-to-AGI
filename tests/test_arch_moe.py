"""Correctness tests of MoE (zero/arch/moe.py, Chapter 24)."""

from __future__ import annotations

import math

import pytest
import torch

from zero.arch.moe import (
    MoEConfig,
    MoEFFN,
    aux_balance_loss,
    moe_transformer,
    router_scores,
    select_experts,
)
from zero.config import ModelConfig
from zero.model import SwiGLU


def _naive_moe(m: MoEFFN, x: torch.Tensor) -> torch.Tensor:
    """Naive implementation, one token and one selected expert at a time:
    y_t = Σ_s FFN_s(x_t) + Σ_{i∈TopK} g_i · FFN_i(x_t).
    """
    c = m.cfg
    x2 = x.reshape(-1, c.dim)
    scores = router_scores(m.router(x2), c.score_func)
    out = torch.zeros_like(x2)
    for t in range(x2.shape[0]):
        choose = scores[t] + m.expert_bias
        top = sorted(range(c.n_experts), key=lambda e: -float(choose[e].detach()))[: c.top_k]
        g = torch.stack([scores[t, e] for e in top])
        if c.norm_topk_prob:
            g = g / g.sum()
        g = g * c.routed_scaling_factor
        for gi, e in zip(g, top):
            h = x2[t]
            y = (torch.nn.functional.silu(h @ m.w_gate[e]) * (h @ m.w_up[e])) @ m.w_down[e]
            out[t] += gi * y
        if m.shared is not None:
            out[t] += m.shared(x2[t])
    return out.reshape(x.shape)


def test_one_expert_top1_equals_dense_swiglu() -> None:
    torch.manual_seed(0)
    cfg = MoEConfig(dim=16, n_experts=1, top_k=1, expert_dim=24)
    moe = MoEFFN(cfg).double()
    dense = SwiGLU(16, 24).double()
    with torch.no_grad():  # nn.Linear stores (out, in); the stacked weights are (in, out)
        dense.w_gate.weight.copy_(moe.w_gate[0].T)
        dense.w_up.weight.copy_(moe.w_up[0].T)
        dense.w_down.weight.copy_(moe.w_down[0].T)
    x = torch.randn(3, 5, 16, dtype=torch.float64)
    assert torch.allclose(moe(x), dense(x), atol=1e-12)


@pytest.mark.parametrize(
    "kw",
    [
        dict(score_func="softmax", top_k=2),
        dict(score_func="sigmoid", top_k=3, routed_scaling_factor=2.5),
        dict(score_func="softmax", top_k=2, norm_topk_prob=False, n_shared_experts=1),
        dict(score_func="sigmoid", top_k=2, n_shared_experts=2, bias_update_speed=0.01),
    ],
)
def test_sorted_dispatch_matches_naive_loop(kw: dict) -> None:
    torch.manual_seed(1)
    cfg = MoEConfig(dim=12, n_experts=6, expert_dim=10, init_std=0.3, **kw)
    m = MoEFFN(cfg).double()
    with torch.no_grad():
        m.expert_bias.copy_(torch.linspace(-0.2, 0.2, 6))  # make the bias really change some selections
    x = torch.randn(2, 7, 12, dtype=torch.float64)
    assert torch.allclose(m(x), _naive_moe(m, x), atol=1e-12)


def test_bias_changes_selection_but_not_gate_values() -> None:
    scores = torch.tensor([[0.9, 0.8, 0.1]])
    idx, w = select_experts(scores, 1, bias=torch.tensor([0.0, 0.0, 1.0]), norm_topk_prob=False)
    assert idx.tolist() == [[2]]  # the bias moves the expert with 0.1 to the first place
    assert torch.allclose(w, torch.tensor([[0.1]]))  # the weight is still the original score


def test_aux_loss_hand_example() -> None:
    # 4 tokens, 2 experts, top-1. The softmax probabilities are below: the first 3 tokens select expert 0,
    # and the last one selects expert 1.
    probs = torch.tensor([[0.9, 0.1], [0.8, 0.2], [0.7, 0.3], [0.4, 0.6]])
    idx = probs.topk(1, dim=-1).indices
    # f = N/(K·T)·counts = 2/4·[3, 1] = [1.5, 0.5]; P = column mean = [0.7, 0.3]
    # L = α·(1.5·0.7 + 0.5·0.3) = α·1.2
    loss = aux_balance_loss(probs, idx, n_experts=2, coef=0.01)
    assert math.isclose(float(loss), 0.012, rel_tol=1e-6)
    # With perfect balance, L = α.
    even = torch.tensor([[0.6, 0.4], [0.4, 0.6]])
    assert math.isclose(
        float(aux_balance_loss(even, even.topk(1).indices, 2, 0.01)), 0.01, rel_tol=1e-6
    )


def test_aux_loss_gradient_pushes_down_overloaded_expert() -> None:
    logits = torch.zeros(8, 4, requires_grad=True)
    with torch.no_grad():
        logits[:, 0] += 1.0  # expert 0 is overloaded
    scores = router_scores(logits, "softmax")
    idx = scores.topk(1, dim=-1).indices
    aux_balance_loss(scores, idx, 4, 1.0).backward()
    # Gradient descent makes the logit of expert 0 smaller and the others larger.
    assert (logits.grad[:, 0] > 0).all() and (logits.grad[:, 1:] < 0).all()


def test_bias_update_direction() -> None:
    cfg = MoEConfig(dim=4, n_experts=4, top_k=1, expert_dim=4, bias_update_speed=0.1)
    m = MoEFFN(cfg)
    m.load_accum.copy_(torch.tensor([10.0, 2.0, 2.0, 2.0]))  # mean 4: expert 0 is overloaded, the others are underloaded
    m.update_bias()
    assert torch.allclose(m.expert_bias, torch.tensor([-0.1, 0.1, 0.1, 0.1]))
    assert m.load_accum.sum() == 0


def test_bias_balancing_flattens_skewed_router() -> None:
    """The router is fixed and strongly biased toward expert 0.

    Bias updates alone (no weight training) can make the load flat.
    """
    torch.manual_seed(0)
    cfg = MoEConfig(
        dim=8, n_experts=4, top_k=1, expert_dim=4, score_func="sigmoid", bias_update_speed=0.01
    )
    m = MoEFFN(cfg).train()
    with torch.no_grad():
        m.router.weight.normal_(std=0.3)
        m.router.weight[0] += 0.5
    x = torch.randn(512, 8).abs()  # positive inputs: the scores of expert 0 are systematically higher
    with torch.no_grad():
        m(x)
        first = m.last_load.max() / m.last_load.mean()
        for _ in range(300):
            m(x)
            m.update_bias()
        m(x)
    last = m.last_load.max() / m.last_load.mean()
    assert first > 2.0 and last < 1.2


def test_capacity_drops_and_large_capacity_is_dropless() -> None:
    torch.manual_seed(2)
    x = torch.randn(40, 8)
    base = MoEFFN(MoEConfig(dim=8, n_experts=4, top_k=2, expert_dim=8, init_std=0.5))
    tight = MoEFFN(
        MoEConfig(dim=8, n_experts=4, top_k=2, expert_dim=8, init_std=0.5, capacity_factor=0.5)
    )
    loose = MoEFFN(
        MoEConfig(dim=8, n_experts=4, top_k=2, expert_dim=8, init_std=0.5, capacity_factor=4.0)
    )
    tight.load_state_dict(base.state_dict())
    loose.load_state_dict(base.state_dict())
    y = base(x)
    tight(x)
    assert tight.capacity(40) == 10 and tight.last_dropped == 80 - 4 * 10
    assert torch.allclose(loose(x), y) and loose.last_dropped == 0


def test_moe_transformer_trains_and_counts_params() -> None:
    mc = ModelConfig(
        vocab_size=50, dim=32, n_layers=3, n_heads=4, n_kv_heads=2, ffn_dim=64, max_seq_len=32
    )
    cfg = MoEConfig.from_model_config(
        mc,
        n_experts=4,
        top_k=2,
        expert_dim=16,
        n_shared_experts=1,
        aux_loss_coef=0.01,
        bias_update_speed=1e-3,
    )
    torch.manual_seed(0)
    model = moe_transformer(mc, cfg, first_dense=1)
    assert len(model.moe_layers()) == 2
    pc = model.param_counts()
    assert pc["total"] - pc["active"] == 2 * (4 - 2) * 3 * 32 * 16
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    tokens = torch.randint(0, 50, (4, 16))
    losses = []
    for _ in range(5):
        loss = model.loss(tokens[:, :-1], tokens[:, 1:])
        opt.zero_grad()
        loss.backward()
        opt.step()
        model.after_step()
        losses.append(loss.item())
    assert all(math.isfinite(v) for v in losses) and losses[-1] < losses[0]
    assert model.moe_layers()[0].router.weight.grad is not None
    assert model.load_stats().shape == (2, 4)
    assert "layers.1.ffn.expert_bias" in model.state_dict()


def test_moe_forward_backward_under_bf16_autocast() -> None:
    """With BF16 autocast, the expert output is BF16 and the accumulation buffer is FP32.

    The precisions must agree before index_add_. (Found in 2026-10 during a CUDA parity check on
    RTX 3090; BF16 autocast on CPU also causes the problem.)
    """
    mc = ModelConfig(
        vocab_size=50, dim=32, n_layers=2, n_heads=4, n_kv_heads=2, ffn_dim=64, max_seq_len=32
    )
    cfg = MoEConfig.from_model_config(mc, n_experts=4, top_k=2, expert_dim=16, n_shared_experts=1)
    torch.manual_seed(0)
    model = moe_transformer(mc, cfg, first_dense=1)
    x = torch.randint(0, 50, (2, 17))
    ref = model.loss(x[:, :-1], x[:, 1:]).item()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        loss = model.loss(x[:, :-1], x[:, 1:])
    loss.backward()
    assert math.isfinite(loss.item()) and abs(loss.item() - ref) < 0.05
    assert model.moe_layers()[0].w_gate.grad is not None
