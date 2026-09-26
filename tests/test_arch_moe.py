"""MoE（zero/arch/moe.py，第 24 章）的正确性测试。"""

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
    """逐 token、逐被选专家的朴素实现：y_t = Σ_s FFN_s(x_t) + Σ_{i∈TopK} g_i · FFN_i(x_t)。"""
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
    with torch.no_grad():  # nn.Linear 存的是 (out, in)，堆叠权重是 (in, out)
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
        m.expert_bias.copy_(torch.linspace(-0.2, 0.2, 6))  # 让偏置真的改变一些选择
    x = torch.randn(2, 7, 12, dtype=torch.float64)
    assert torch.allclose(m(x), _naive_moe(m, x), atol=1e-12)


def test_bias_changes_selection_but_not_gate_values() -> None:
    scores = torch.tensor([[0.9, 0.8, 0.1]])
    idx, w = select_experts(scores, 1, bias=torch.tensor([0.0, 0.0, 1.0]), norm_topk_prob=False)
    assert idx.tolist() == [[2]]  # 偏置把 0.1 的专家推到了第一
    assert torch.allclose(w, torch.tensor([[0.1]]))  # 权重仍是原始分数


def test_aux_loss_hand_example() -> None:
    # 4 个 token、2 个专家、top-1；softmax 概率如下，前 3 个 token 选专家 0，最后一个选专家 1
    probs = torch.tensor([[0.9, 0.1], [0.8, 0.2], [0.7, 0.3], [0.4, 0.6]])
    idx = probs.topk(1, dim=-1).indices
    # f = N/(K·T)·counts = 2/4·[3, 1] = [1.5, 0.5]；P = 列平均 = [0.7, 0.3]
    # L = α·(1.5·0.7 + 0.5·0.3) = α·1.2
    loss = aux_balance_loss(probs, idx, n_experts=2, coef=0.01)
    assert math.isclose(float(loss), 0.012, rel_tol=1e-6)
    # 完全均衡时 L = α
    even = torch.tensor([[0.6, 0.4], [0.4, 0.6]])
    assert math.isclose(
        float(aux_balance_loss(even, even.topk(1).indices, 2, 0.01)), 0.01, rel_tol=1e-6
    )


def test_aux_loss_gradient_pushes_down_overloaded_expert() -> None:
    logits = torch.zeros(8, 4, requires_grad=True)
    with torch.no_grad():
        logits[:, 0] += 1.0  # 专家 0 过载
    scores = router_scores(logits, "softmax")
    idx = scores.topk(1, dim=-1).indices
    aux_balance_loss(scores, idx, 4, 1.0).backward()
    # 梯度下降会让专家 0 的 logit 变小、其余变大
    assert (logits.grad[:, 0] > 0).all() and (logits.grad[:, 1:] < 0).all()


def test_bias_update_direction() -> None:
    cfg = MoEConfig(dim=4, n_experts=4, top_k=1, expert_dim=4, bias_update_speed=0.1)
    m = MoEFFN(cfg)
    m.load_accum.copy_(torch.tensor([10.0, 2.0, 2.0, 2.0]))  # 平均 4：专家 0 过载，其余欠载
    m.update_bias()
    assert torch.allclose(m.expert_bias, torch.tensor([-0.1, 0.1, 0.1, 0.1]))
    assert m.load_accum.sum() == 0


def test_bias_balancing_flattens_skewed_router() -> None:
    """路由器固定且严重偏向专家 0，只靠偏置更新（不训练任何权重）也能把负载拉平。"""
    torch.manual_seed(0)
    cfg = MoEConfig(
        dim=8, n_experts=4, top_k=1, expert_dim=4, score_func="sigmoid", bias_update_speed=0.01
    )
    m = MoEFFN(cfg).train()
    with torch.no_grad():
        m.router.weight.normal_(std=0.3)
        m.router.weight[0] += 0.5
    x = torch.randn(512, 8).abs()  # 正输入：专家 0 的分数系统性偏高
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
