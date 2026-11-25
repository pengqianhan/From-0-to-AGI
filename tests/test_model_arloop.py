"""Attention residuals of ARLoop (attn_res in ModelConfig; runs/ladder-3090/README.md, stage 8).

- The sources that each sublayer gets match tables written by hand from the reference code
  (OSU-NLP-Group/ARLoop, commit 16c8eaa: `_run_loops`, `_select_values`, `_maybe_close_block`).
  The tables are hardcoded here; the expected tensors are built from the recorded sublayer
  updates, not from the model's own routing code.
- The mixing equals an independent copy of `naive_attnres` (flash-linear-attention).
- At init (q = 0) the input of a sublayer is the plain mean of its sources.
- Dense, lf3 and lm3 models are unchanged (compared with zero/model.py before stage 8).
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from zero.config import ConfigError, ModelConfig, OptimConfig, load_config
from zero.kv_cache import KVCache
from zero.model import Transformer, attn_res_mix, count_params, estimate_flops_per_token
from zero.train.checkpoint import load_model_weights, save_checkpoint
from zero.train.trainer import build_optimizer

REPO = Path(__file__).resolve().parents[1]
# the last commit before stage 8 (the stage-8 preregistration): the code without attn_res and the seed key
BEFORE_STAGE8 = "e3a5ec7"
TOKENS = torch.randint(0, 97, (2, 16), generator=torch.Generator().manual_seed(1))


def tiny(**kw: object) -> ModelConfig:
    base = dict(vocab_size=97, dim=32, n_layers=4, n_heads=4, n_kv_heads=2, ffn_dim=64, max_seq_len=64)
    base.update(kw)
    return ModelConfig(**base)  # type: ignore[arg-type]


def ar(n_layers: int = 4, block: int = 2, window: int | None = 3, **kw: object) -> ModelConfig:
    return tiny(n_layers=n_layers, n_loops=3, attn_res=True, attn_res_block_layers=block, attn_res_window=window, **kw)


def make(cfg: ModelConfig, seed: int = 0) -> Transformer:
    torch.manual_seed(seed)
    return Transformer(cfg).eval()


def randomize_routing(m: Transformer, seed: int = 7) -> None:
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for q in [*m.attn_res_queries, m.final_attn_res_query]:
            q.copy_(torch.randn(q.shape, generator=g))
        for n in [*m.attn_res_norms, m.final_attn_res_norm]:
            n.weight.copy_(1 + 0.3 * torch.randn(n.weight.shape, generator=g))


# ---------------------------------------------------------------- independent reference


def naive_attnres(query: torch.Tensor, residuals: list[torch.Tensor], rms_weight: torch.Tensor,
                  rms_eps: float = 1e-6, scale: float = 1.0) -> torch.Tensor:
    """Copy of the math of fla.ops.attnres.naive.naive_attnres (output_rms_weight=None)."""
    shape = residuals[0].shape
    D = shape[-1]
    stacked = torch.stack(tuple(r.reshape(-1, D) for r in residuals), dim=0)
    v = stacked.float()
    k = F.rms_norm(v, (D,), rms_weight.flatten().float(), rms_eps)
    p = torch.einsum("lnd,d->ln", k, query.flatten().float() * scale).softmax(dim=0)
    o = torch.einsum("ln,lnd->nd", p, v).view(shape)
    return o.to(stacked.dtype)


# ---------------------------------------------------------------- source tables (by hand)

# n_layers = 4, n_loops = 3, 2 layers per block (e24m). Sublayer 24 = the final output.
TABLE_4_2_WINDOW3 = {
    **{0: ["E", "E"]},
    **{i: ["E", "P1"] for i in range(1, 5)},
    **{i: ["E", "B1", "P2"] for i in range(5, 9)},
    **{i: ["E", "B1", "B2", "P3"] for i in range(9, 13)},
    **{i: ["E", "B2", "B3", "P4"] for i in range(13, 17)},
    **{i: ["E", "B3", "B4", "P5"] for i in range(17, 21)},
    **{i: ["E", "B4", "B5", "P6"] for i in range(21, 25)},
}
TABLE_4_2_FULL = {
    **{i: TABLE_4_2_WINDOW3[i] for i in range(0, 13)},
    **{i: ["E", "B1", "B2", "B3", "P4"] for i in range(13, 17)},
    **{i: ["E", "B1", "B2", "B3", "B4", "P5"] for i in range(17, 21)},
    **{i: ["E", "B1", "B2", "B3", "B4", "B5", "P6"] for i in range(21, 25)},
}
# n_layers = 6, n_loops = 3, 3 layers per block (e44m). Sublayer 36 = the final output.
TABLE_6_3_WINDOW3 = {
    **{0: ["E", "E"]},
    **{i: ["E", "P1"] for i in range(1, 7)},
    **{i: ["E", "B1", "P2"] for i in range(7, 13)},
    **{i: ["E", "B1", "B2", "P3"] for i in range(13, 19)},
    **{i: ["E", "B2", "B3", "P4"] for i in range(19, 25)},
    **{i: ["E", "B3", "B4", "P5"] for i in range(25, 31)},
    **{i: ["E", "B4", "B5", "P6"] for i in range(31, 37)},
}
TABLE_6_3_FULL = {
    **{i: TABLE_6_3_WINDOW3[i] for i in range(0, 19)},
    **{i: ["E", "B1", "B2", "B3", "P4"] for i in range(19, 25)},
    **{i: ["E", "B1", "B2", "B3", "B4", "P5"] for i in range(25, 31)},
    **{i: ["E", "B1", "B2", "B3", "B4", "B5", "P6"] for i in range(31, 37)},
}


def record(m: Transformer) -> tuple[list[tuple[int, list[torch.Tensor], torch.Tensor]], list[torch.Tensor]]:
    """Run a forward pass; record (i, sources, routed input) of each routing call and each sublayer update."""
    routes: list = []
    updates: list[torch.Tensor] = []
    orig = m.attn_res_route

    def route(i: int, values: list[torch.Tensor]) -> torch.Tensor:
        out = orig(i, values)
        routes.append((i, [v.detach().clone() for v in values], out.detach().clone()))
        return out

    m.attn_res_route = route  # type: ignore[method-assign]
    hooks = [mod.register_forward_hook(lambda _m, _i, o: updates.append(o.detach().clone()))
             for layer in m.layers for mod in (layer.attn, layer.ffn)]
    try:
        with torch.no_grad():
            m(TOKENS)
    finally:
        for h in hooks:
            h.remove()
        del m.attn_res_route
    return routes, updates


def expected(label: str, i: int, E: torch.Tensor, u: list[torch.Tensor], subs_per_block: int) -> torch.Tensor:
    """E; B_k = sum of the updates of block k (+ E for k = 1); P_k at sublayer i = what block k has so far."""
    if label == "E":
        return E
    k = int(label[1:])
    lo = (k - 1) * subs_per_block
    hi = k * subs_per_block if label[0] == "B" else i
    acc = E if k == 1 else None
    for j in range(lo, hi):
        acc = u[j] if acc is None else acc + u[j]
    assert acc is not None
    return acc


@pytest.mark.parametrize(
    "n_layers,block,window,table",
    [(4, 2, 3, TABLE_4_2_WINDOW3), (4, 2, None, TABLE_4_2_FULL), (6, 3, 3, TABLE_6_3_WINDOW3),
     (6, 3, None, TABLE_6_3_FULL)],
)
def test_sources_match_the_hand_table(n_layers: int, block: int, window: int | None, table: dict) -> None:
    m = make(ar(n_layers, block, window))
    randomize_routing(m)
    routes, u = record(m)
    n_sub = 2 * 3 * n_layers
    assert len(u) == n_sub and [r[0] for r in routes] == list(range(n_sub + 1))
    assert sorted(table) == list(range(n_sub + 1))
    with torch.no_grad():
        E = m.tok_emb(TOKENS)
    for i, values, _ in routes:
        want = [expected(lab, i, E, u, 2 * block) for lab in table[i]]
        assert len(values) == len(want), f"sublayer {i}: {len(values)} sources, table {table[i]}"
        for lab, v, w in zip(table[i], values, want):
            torch.testing.assert_close(v, w, atol=1e-5, rtol=1e-5, msg=f"sublayer {i} source {lab}")
    # the sources are different tensors (the check above can tell B2 from P3 and so on)
    _, vals, _ = routes[12]
    assert min((a - b).abs().max() for a in vals for b in vals if a is not b) > 1e-3


def test_routing_equals_naive_attnres() -> None:
    m = make(ar())
    randomize_routing(m)
    routes, _ = record(m)
    for i, values, out in routes:
        if i == len(m.attn_res_queries):
            q, w = m.final_attn_res_query, m.final_attn_res_norm.weight
        else:
            q, w = m.attn_res_queries[i], m.attn_res_norms[i].weight
        torch.testing.assert_close(out, naive_attnres(q.detach(), values, w.detach(), 1e-6), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_attn_res_mix_matches_naive_values_and_gradients(dtype: torch.dtype) -> None:
    g = torch.Generator().manual_seed(0)
    vals = [(torch.randn(2, 5, 16, generator=g) * s).to(dtype).requires_grad_() for s in (1.0, 3.0, 0.2, 7.0)]
    q = torch.randn(16, generator=g).requires_grad_()
    w = (1 + 0.5 * torch.randn(16, generator=g)).requires_grad_()
    a = attn_res_mix(q, w, 1e-6, vals)
    b = naive_attnres(q, vals, w, 1e-6)
    assert a.dtype == b.dtype == dtype
    tol = dict(atol=1e-5, rtol=1e-5) if dtype == torch.float32 else dict(atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(a, b, **tol)
    if dtype == torch.float32:
        ga = torch.autograd.grad(a.square().sum(), [q, w, *vals])
        gb = torch.autograd.grad(b.square().sum(), [q, w, *vals])
        for x, y in zip(ga, gb):
            torch.testing.assert_close(x, y, atol=1e-3, rtol=1e-3)  # fp32 sums in another order


def test_init_routes_the_plain_mean() -> None:
    m = make(ar())
    assert all(torch.equal(q, torch.zeros(32)) for q in [*m.attn_res_queries, m.final_attn_res_query])
    assert all(torch.equal(n.weight, torch.ones(32)) for n in [*m.attn_res_norms, m.final_attn_res_norm])
    routes, _ = record(m)
    for _, values, out in routes:
        torch.testing.assert_close(out, torch.stack(values).mean(0), atol=1e-6, rtol=1e-6)


def test_hand_forward_at_init() -> None:
    """A forward pass written without the model's routing code: plain means of the hand-table sources."""
    m = make(ar())
    with torch.no_grad():
        cos, sin = m.rope(0, 16)
        E = m.tok_emb(TOKENS)
        u: list[torch.Tensor] = []
        for i in range(24):
            x = torch.stack([expected(lab, i, E, u, 4) for lab in TABLE_4_2_WINDOW3[i]]).mean(0)
            layer = m.layers[(i // 2) % 4]
            u.append(layer.attn(layer.attn_norm(x), cos, sin) if i % 2 == 0 else layer.ffn(layer.ffn_norm(x)))
        x = torch.stack([expected(lab, 24, E, u, 4) for lab in TABLE_4_2_WINDOW3[24]]).mean(0)
        torch.testing.assert_close(m(TOKENS), m.lm_head(m.norm(x)), atol=1e-5, rtol=1e-5)


def test_parameters_count_init_and_no_weight_decay(tmp_path: Path) -> None:
    cfg = ar()
    m = make(cfg)
    n_sub = 2 * 4 * 3
    assert len(m.attn_res_queries) == len(m.attn_res_norms) == n_sub
    lf3 = make(tiny(n_loops=3))
    assert m.num_params() == count_params(cfg)["total"] == lf3.num_params() + (n_sub + 1) * 2 * 32
    ar_params = [*m.attn_res_queries, m.final_attn_res_query, *(n.weight for n in m.attn_res_norms),
                 m.final_attn_res_norm.weight]
    assert all(p.dim() == 1 for p in ar_params) and len({id(p) for p in ar_params}) == 2 * (n_sub + 1)
    opt = build_optimizer(m, OptimConfig(weight_decay=0.1), torch.device("cpu"))
    no_decay = {id(p) for g in opt.param_groups if g["weight_decay"] == 0.0 for p in g["params"]}
    decay = {id(p) for g in opt.param_groups if g["weight_decay"] > 0 for p in g["params"]}
    assert all(id(p) in no_decay and id(p) not in decay for p in ar_params)
    # init_weights puts q back to 0 and w back to 1
    randomize_routing(m)
    m.init_weights()
    assert all(torch.equal(q, torch.zeros(32)) for q in [*m.attn_res_queries, m.final_attn_res_query])
    assert all(torch.equal(n.weight, torch.ones(32)) for n in [*m.attn_res_norms, m.final_attn_res_norm])


def test_same_seed_gives_the_lf3_weights() -> None:
    """The new parameters use no random numbers: with the same seed, the shared weights equal lf3's."""
    a, b = make(ar()), make(tiny(n_loops=3))
    sa, sb = a.state_dict(), b.state_dict()
    assert set(sb) < set(sa)
    assert all(torch.equal(sa[k], sb[k]) for k in sb)
    assert {k.split(".")[0] for k in set(sa) - set(sb)} == {
        "attn_res_queries", "attn_res_norms", "final_attn_res_query", "final_attn_res_norm"}


def test_activation_checkpointing_same_gradients() -> None:
    grads = []
    for ckpt in (False, True):
        m = make(ar()).train()
        randomize_routing(m)
        m.activation_checkpointing = ckpt
        m.loss(TOKENS[:, :-1], TOKENS[:, 1:]).backward()
        grads.append({k: p.grad.clone() for k, p in m.named_parameters() if p.grad is not None})
    assert grads[0].keys() == grads[1].keys()
    assert grads[0]["attn_res_queries.5"].abs().sum() > 0 and grads[0]["final_attn_res_query"].abs().sum() > 0
    for k in grads[0]:
        torch.testing.assert_close(grads[0][k], grads[1][k], atol=1e-6, rtol=1e-5)


def test_bf16_autocast_keeps_sources_in_the_residual_dtype() -> None:
    m = make(ar())
    randomize_routing(m)
    routes = []
    orig = m.attn_res_route
    m.attn_res_route = lambda i, v: routes.append([x.dtype for x in v]) or orig(i, v)  # type: ignore[method-assign]
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        logits = m(TOKENS)
    assert logits.dtype == torch.bfloat16 and torch.isfinite(logits.float()).all()
    assert len(routes) == 25 and all(d == torch.float32 for ds in routes for d in ds)


def test_checkpoint_round_trip(tmp_path: Path) -> None:
    m = make(ar())
    randomize_routing(m)
    opt = build_optimizer(m, OptimConfig(weight_decay=0.1), torch.device("cpu"))
    save_checkpoint(tmp_path / "ckpt", 10, m, opt)
    m2 = make(ar(), seed=9)
    load_model_weights(tmp_path / "ckpt", m2)
    for a, b in zip([*m.attn_res_queries, m.final_attn_res_query], [*m2.attn_res_queries, m2.final_attn_res_query]):
        assert torch.equal(a, b)
    with torch.no_grad():
        assert torch.equal(m(TOKENS), m2(TOKENS))


def test_kv_cache_and_hf_export_raise(tmp_path: Path) -> None:
    from zero.hf import config_to_hf_qwen3, export_to_hf_qwen3

    for cfg in (ar(), tiny(attn_res=True, attn_res_block_layers=2)):  # also without a loop
        m = make(cfg)
        cache = KVCache.from_config(m.config, batch_size=2, device="cpu", dtype=torch.float32)
        with pytest.raises(NotImplementedError):
            m(TOKENS, kv_cache=cache)
        with pytest.raises(NotImplementedError):
            config_to_hf_qwen3(m.config)
        with pytest.raises(NotImplementedError):
            export_to_hf_qwen3(m, None, tmp_path / "hf")
        assert not (tmp_path / "hf" / "model.safetensors").exists()


@pytest.mark.parametrize(
    "kw",
    [dict(n_loops=3, attn_res=True, attn_res_block_layers=0),
     dict(n_loops=3, attn_res=True, attn_res_block_layers=3),
     dict(n_loops=3, attn_res=True, attn_res_block_layers=2, attn_res_window=0),
     dict(n_loops=3, loop_start=1, loop_end=3, attn_res=True, attn_res_block_layers=2),
     dict(n_loops=3, loop_gate=True, attn_res=True, attn_res_block_layers=2),
     dict(n_loops=3, attn_res_block_layers=2),
     dict(n_loops=3, attn_res_window=3)],
)
def test_bad_attn_res_config_is_rejected(kw: dict) -> None:
    with pytest.raises(ConfigError):
        tiny(**kw).validate()


@pytest.mark.parametrize(
    "scale,lf3,n_eff,block,window",
    [("e24m_ar3", "e24m_lf3", 38_207_488, 2, 3), ("e24m_arf3", "e24m_lf3", 38_207_488, 2, None),
     ("e44m_ar3", "e44m_lf3", 82_673_664, 3, 3), ("e44m_arf3", "e44m_lf3", 82_673_664, 3, None)],
)
def test_ladder_arloop_configs(scale: str, lf3: str, n_eff: int, block: int, window: int | None) -> None:
    cfg = load_config(REPO / "configs" / "ladder3090" / f"{scale}.toml").model
    base = load_config(REPO / "configs" / "ladder3090" / f"{lf3}.toml").model
    assert estimate_flops_per_token(cfg, 2048) / 6 == estimate_flops_per_token(base, 2048) / 6 == n_eff
    assert (cfg.attn_res, cfg.attn_res_block_layers, cfg.attn_res_window) == (True, block, window)
    diff = {k for k in vars(cfg) if getattr(cfg, k) != getattr(base, k)}
    assert diff == {"attn_res", "attn_res_block_layers"} | ({"attn_res_window"} if window else set())
    assert count_params(cfg)["total"] == count_params(base)["total"] + (2 * cfg.layer_applications + 1) * 2 * cfg.dim


# ---------------------------------------------------------------- existing models unchanged


def _head_module(path: str, name: str) -> object:
    src = subprocess.run(["git", "show", f"{BEFORE_STAGE8}:{path}"], cwd=REPO, capture_output=True, text=True, check=True).stdout
    spec = importlib.util.spec_from_loader(name, loader=None)
    assert spec is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # dataclasses need the module in sys.modules
    exec(compile(src, f"{BEFORE_STAGE8}:{path}", "exec"), mod.__dict__)
    return mod


@pytest.mark.parametrize(
    "kw",
    [dict(), dict(n_loops=3), dict(n_loops=3, loop_start=1, loop_end=3, loop_gate=True)],
    ids=["dense", "lf3", "lm3"],
)
def test_existing_models_unchanged(kw: dict) -> None:
    old = _head_module("zero/model.py", "_head_zero_model")
    cfg = tiny(**kw)
    torch.manual_seed(0)
    a = old.Transformer(cfg).eval()  # type: ignore[attr-defined]
    torch.manual_seed(0)
    b = Transformer(cfg).eval()
    sa, sb = a.state_dict(), b.state_dict()
    assert list(sa) == list(sb) and all(torch.equal(sa[k], sb[k]) for k in sa)
    if cfg.loop_gate:
        with torch.no_grad():
            for ga, gb in zip(a.loop_gates, b.loop_gates):
                ga.fill_(0.4)
                gb.fill_(0.4)
    with torch.no_grad():
        assert torch.equal(a(TOKENS), b(TOKENS))
    a.train()
    b.train()
    la, lb = a.loss(TOKENS[:, :-1], TOKENS[:, 1:]), b.loss(TOKENS[:, :-1], TOKENS[:, 1:])
    la.backward()
    lb.backward()
    assert torch.equal(la, lb)
    assert all(torch.equal(pa.grad, pb.grad) for pa, pb in zip(a.parameters(), b.parameters()))
    assert old.count_params(cfg) == count_params(cfg)  # type: ignore[attr-defined]
    assert old.estimate_flops_per_token(cfg, 2048) == estimate_flops_per_token(cfg, 2048)  # type: ignore[attr-defined]
