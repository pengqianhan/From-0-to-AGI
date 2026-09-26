"""Muon 优化器（zero/train/muon.py，第 12 章）的 CPU 测试。GPU 路径（BF16 NS5、DDP）尚未在 GPU 上验证。"""

from __future__ import annotations

import copy

import torch

from zero.config import ModelConfig, OptimConfig
from zero.model import Transformer
from zero.train.muon import (
    MuonAdamW,
    build_muon_optimizer,
    split_params_for_muon,
    zeropower_via_newtonschulz5,
)


def test_newton_schulz_orthogonalizes() -> None:
    torch.manual_seed(0)
    for shape in [(64, 32), (32, 64), (48, 48)]:
        G = torch.randn(*shape)
        X = zeropower_via_newtonschulz5(G, steps=5)
        s = torch.linalg.svdvals(X)
        # 参考系数刻意不收敛到精确的 1，而是在 5 步内把全部奇异值推到大约 [0.7, 1.2]
        assert s.min() > 0.5 and s.max() < 1.3, (shape, s.min(), s.max())
        U, _, Vh = torch.linalg.svd(G, full_matrices=False)
        polar = U @ Vh  # 精确的"正交化"结果
        cos = (X * polar).sum() / (X.norm() * polar.norm())
        assert cos > 0.95


def test_rms_matched_update_scale() -> None:
    """adjust="rms" 时，单步更新的 RMS ≈ lr × 0.2（与 AdamW 的典型更新同量级，可以沿用 AdamW 学习率）。"""
    torch.manual_seed(0)
    W = torch.nn.Parameter(torch.randn(96, 256) * 0.02)
    before = W.detach().clone()
    opt = MuonAdamW([{"params": [W], "use_muon": True}], lr=0.01, weight_decay=0.0)
    W.grad = torch.randn_like(W)
    opt.step()
    rms = ((W.detach() - before) ** 2).mean().sqrt().item()
    assert 0.6 * 0.002 < rms < 1.4 * 0.002, rms


def test_param_split_follows_tech_reports() -> None:
    cfg = ModelConfig(
        vocab_size=128, dim=32, n_layers=2, n_heads=4, n_kv_heads=2, ffn_dim=64, max_seq_len=16
    )
    model = Transformer(cfg)
    groups = split_params_for_muon(model)
    names = {id(p): n for n, p in model.named_parameters()}
    muon = {names[id(p)] for p in groups["muon"]}
    assert all(n.startswith("layers.") and ("attn.w" in n or "ffn.w" in n) for n in muon)
    assert len(muon) == 2 * (4 + 3)  # 每层 wq/wk/wv/wo + w_gate/w_up/w_down
    other = {names[id(p)] for p in groups["adam_no_decay"]}
    assert "tok_emb.weight" in other and any("norm" in n for n in other)
    total = sum(len(v) for v in groups.values())
    assert total == len({id(p) for p in model.parameters()})  # 共享 embedding 不重复


def _train(model, opt, steps=30, seed=0):
    g = torch.Generator().manual_seed(seed)
    losses = []
    for _ in range(steps):
        x = torch.randint(0, 64, (4, 17), generator=g)
        loss = model.loss(x[:, :-1], x[:, 1:] // 2)  # 可学的规律：目标 = 输入的一半
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses


def test_muon_trains_and_resumes_exactly() -> None:
    torch.manual_seed(0)
    cfg = ModelConfig(
        vocab_size=64, dim=32, n_layers=2, n_heads=4, n_kv_heads=2, ffn_dim=64, max_seq_len=16
    )
    model = Transformer(cfg)
    ocfg = OptimConfig(lr=0.02, weight_decay=0.01)
    opt = build_muon_optimizer(model, ocfg)
    losses = _train(model, opt, steps=40)
    assert losses[-1] < losses[0] - 0.5, (losses[0], losses[-1])

    # checkpoint 往返：从中途的 state_dict 续训，与不中断的结果逐位一致
    m2 = copy.deepcopy(model)
    o2 = build_muon_optimizer(m2, ocfg)
    o2.load_state_dict(copy.deepcopy(opt.state_dict()))
    a = _train(model, opt, steps=5, seed=1)
    b = _train(m2, o2, steps=5, seed=1)
    assert a == b
