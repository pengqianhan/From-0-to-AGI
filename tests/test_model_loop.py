"""Looped Transformer (n_loops in ModelConfig; runs/ladder-3090/README.md, stage 7).

- n_loops = 1 is the normal model (same parameters, same init, same logits).
- Middle loop with gates: g = 0 is the same as no loop, g = 1 is a plain loop of the middle layers.
- Whole-stack loop: all layers 3 times with shared weights.
- FLOPs count each layer application; the four N_eff numbers of the preregistration.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from zero.config import ConfigError, ModelConfig, OptimConfig, load_config
from zero.kv_cache import KVCache
from zero.model import Transformer, count_params, estimate_flops_per_token
from zero.train.checkpoint import load_model_weights, save_checkpoint
from zero.train.trainer import build_optimizer

REPO = Path(__file__).resolve().parents[1]


def tiny(**kw: object) -> ModelConfig:
    base = dict(vocab_size=97, dim=32, n_layers=4, n_heads=4, n_kv_heads=2, ffn_dim=64, max_seq_len=64)
    base.update(kw)
    return ModelConfig(**base)  # type: ignore[arg-type]


def make(cfg: ModelConfig, seed: int = 0) -> Transformer:
    torch.manual_seed(seed)
    return Transformer(cfg).eval()


def run_layers(model: Transformer, layers: list, h: torch.Tensor, T: int) -> torch.Tensor:
    cos, sin = model.rope(0, T)
    for layer in layers:
        h = layer(h, cos, sin)
    return h


def head(model: Transformer, h: torch.Tensor) -> torch.Tensor:
    return model.lm_head(model.norm(h))


TOKENS = torch.randint(0, 97, (2, 16), generator=torch.Generator().manual_seed(1))


def test_n_loops_1_is_the_normal_model() -> None:
    a = make(tiny())
    b = make(tiny(n_loops=1, loop_start=0, loop_end=None, loop_gate=False))
    sa, sb = a.state_dict(), b.state_dict()
    assert sa.keys() == sb.keys() and not any("loop" in k for k in sa)
    assert all(torch.equal(sa[k], sb[k]) for k in sa)
    with torch.no_grad():
        ref = head(a, run_layers(a, list(a.layers), a.tok_emb(TOKENS), 16))
        assert torch.equal(a(TOKENS), ref)
        assert torch.equal(b(TOKENS), ref)
    cfg = tiny()
    assert cfg.layer_applications == cfg.n_layers
    d, L, V, T = 32, 4, 97, 64
    q, kv, f = 32, 16, 64
    assert estimate_flops_per_token(cfg, T) == 6 * (L * (2 * d * q + 2 * d * kv + 3 * d * f) + d * V) + 12 * L * q * T


def test_middle_loop_gate_0_equals_no_loop_and_gate_1_a_plain_loop() -> None:
    lm = make(tiny(n_loops=3, loop_start=1, loop_end=3, loop_gate=True))
    dense = make(tiny(), seed=5)
    sd = {k: v for k, v in lm.state_dict().items() if not k.startswith("loop_gates")}
    dense.load_state_dict(sd)
    assert len(lm.loop_gates) == 2 and all(torch.equal(g, torch.zeros(32)) for g in lm.loop_gates)
    with torch.no_grad():
        assert torch.equal(lm(TOKENS), dense(TOKENS))  # gate 0 at init: each pass sees p again
        for g in lm.loop_gates:
            g.fill_(1.0)
        L = list(lm.layers)
        h = run_layers(lm, L[:1], lm.tok_emb(TOKENS), 16)
        for _ in range(3):
            h = run_layers(lm, L[1:3], h, 16)
        ref = head(lm, run_layers(lm, L[3:], h, 16))
        torch.testing.assert_close(lm(TOKENS), ref, atol=1e-5, rtol=1e-5)
        # a general gate: x_r = p + g_r * (m_{r-1} - p)
        torch.manual_seed(3)
        for g in lm.loop_gates:
            g.copy_(torch.randn(32))
        p = run_layers(lm, L[:1], lm.tok_emb(TOKENS), 16)
        m = run_layers(lm, L[1:3], p, 16)
        for r in range(2):
            m = run_layers(lm, L[1:3], p + lm.loop_gates[r] * (m - p), 16)
        torch.testing.assert_close(lm(TOKENS), head(lm, run_layers(lm, L[3:], m, 16)), atol=0, rtol=0)


def test_whole_stack_loop_runs_all_layers_three_times() -> None:
    lf = make(tiny(n_loops=3))
    assert lf.loop_gates is None and not any("loop" in k for k in lf.state_dict())
    with torch.no_grad():
        h = lf.tok_emb(TOKENS)
        for _ in range(3):
            h = run_layers(lf, list(lf.layers), h, 16)
        assert torch.equal(lf(TOKENS), head(lf, h))


def test_init_divides_by_layer_applications() -> None:
    cfg = tiny(dim=256, ffn_dim=512, n_heads=4, n_kv_heads=2, n_loops=3)  # 12 applications
    m = make(cfg)
    for w in (m.layers[0].attn.wo.weight, m.layers[0].ffn.w_down.weight):
        assert abs(float(w.detach().std()) / (0.02 / math.sqrt(2 * 12)) - 1) < 0.05
    assert abs(float(m.layers[0].attn.wq.weight.detach().std()) / 0.02 - 1) < 0.05


def test_flops_count_layer_applications() -> None:
    def per_layer(c: ModelConfig, T: int) -> float:
        return 6 * (2 * c.dim * c.q_dim + 2 * c.dim * c.kv_dim + 3 * c.dim * c.ffn_dim) + 12 * c.q_dim * T

    for kw, apps in [(dict(n_loops=3), 12), (dict(n_loops=3, loop_start=1, loop_end=3, loop_gate=True), 8),
                     (dict(n_loops=2, loop_start=2), 6)]:
        c = tiny(**kw)
        assert c.layer_applications == apps
        assert estimate_flops_per_token(c, 64) == apps * per_layer(c, 64) + 6 * c.dim * c.vocab_size
        m = make(c)
        assert count_params(c)["total"] == m.num_params()  # stored parameters (+ gates), not applications
    assert count_params(tiny(n_loops=3, loop_start=1, loop_end=3, loop_gate=True))["total"] == (
        count_params(tiny())["total"] + 2 * 32
    )


@pytest.mark.parametrize(
    "scale,n_eff",
    [("e24m_lf3", 38_207_488), ("e24m_lm3", 31_064_064), ("e44m_lf3", 82_673_664), ("e44m_lm3", 57_114_624)],
)
def test_ladder_loop_configs_match_preregistered_n_eff(scale: str, n_eff: int) -> None:
    # README stage 7: 38.21M / 31.06M / 82.67M / 57.11M
    cfg = load_config(REPO / "configs" / "ladder3090" / f"{scale}.toml").model
    dense = load_config(REPO / "configs" / "ladder3090" / f"{scale.split('_')[0]}.toml").model
    assert estimate_flops_per_token(cfg, 2048) / 6 == n_eff
    # only the loop fields differ from the dense scale
    diff = {k for k in vars(cfg) if getattr(cfg, k) != getattr(dense, k)}
    assert diff <= {"n_loops", "loop_start", "loop_end", "loop_gate"}


def test_kv_cache_with_loop_raises() -> None:
    m = make(tiny(n_loops=3))
    cache = KVCache.from_config(m.config, batch_size=2, device="cpu", dtype=torch.float32)
    with pytest.raises(NotImplementedError):
        m(TOKENS, kv_cache=cache)


def test_activation_checkpointing_same_gradients() -> None:
    cfg = tiny(n_loops=3, loop_start=1, loop_end=3, loop_gate=True)
    grads = []
    for ckpt in (False, True):
        m = make(cfg).train()
        with torch.no_grad():
            for g in m.loop_gates:
                g.fill_(0.5)
        m.activation_checkpointing = ckpt
        m.loss(TOKENS[:, :-1], TOKENS[:, 1:]).backward()
        grads.append({k: p.grad.clone() for k, p in m.named_parameters()})
    assert grads[0]["loop_gates.0"].abs().sum() > 0
    for k in grads[0]:
        torch.testing.assert_close(grads[0][k], grads[1][k], atol=1e-6, rtol=1e-5)


def test_gate_has_no_weight_decay_and_checkpoint_keeps_it(tmp_path: Path) -> None:
    m = make(tiny(n_loops=3, loop_start=1, loop_end=3, loop_gate=True))
    opt = build_optimizer(m, OptimConfig(weight_decay=0.1), torch.device("cpu"))
    no_decay = next(g for g in opt.param_groups if g["weight_decay"] == 0.0)
    assert all(any(p is q for q in no_decay["params"]) for p in m.loop_gates)
    decay = next(g for g in opt.param_groups if g["weight_decay"] > 0)
    assert not any(p is q for p in m.loop_gates for q in decay["params"])
    with torch.no_grad():
        for g in m.loop_gates:
            g.copy_(torch.randn(32))
    save_checkpoint(tmp_path / "ckpt", 10, m, opt)
    m2 = make(tiny(n_loops=3, loop_start=1, loop_end=3, loop_gate=True), seed=9)
    load_model_weights(tmp_path / "ckpt", m2)
    for a, b in zip(m.loop_gates, m2.loop_gates):
        assert torch.equal(a, b)
    with torch.no_grad():
        assert torch.equal(m(TOKENS), m2(TOKENS))


@pytest.mark.parametrize(
    "kw",
    [dict(n_loops=0), dict(n_loops=3, loop_start=3, loop_end=3), dict(n_loops=3, loop_end=5),
     dict(n_loops=3, loop_start=-1), dict(loop_gate=True)],
)
def test_bad_loop_config_is_rejected(kw: dict) -> None:
    with pytest.raises(ConfigError):
        tiny(**kw).validate()


def test_hf_export_of_looped_model_raises(tmp_path: Path) -> None:
    from zero.hf import config_to_hf_qwen3, export_to_hf_qwen3

    m = make(tiny(n_loops=3))  # adds no parameters: without the check it would export silently
    with pytest.raises(NotImplementedError):
        config_to_hf_qwen3(m.config)
    with pytest.raises(NotImplementedError):
        export_to_hf_qwen3(m, None, tmp_path / "hf")
    assert not (tmp_path / "hf" / "model.safetensors").exists()
