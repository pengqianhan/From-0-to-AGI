"""Chapter 23: correctness tests of linear attention / Gated DeltaNet / the hybrid model.

- Recurrent form == chunked form (linear attention, with decay, delta rule, gated delta rule;
  with an initial state, with a length that is not a multiple of the chunk size).
- The gated delta rule agrees with "the most basic matrix form"
  S_t = α_t (I − β_t k_t k_tᵀ) S_{t−1} + β_t k_t v_tᵀ.
- Layer: input in segments (with state) == input in one pass; parity check against the weights
  of the HF Qwen3.5 GatedDeltaNet.
- Hybrid model: generation with the state cache == full recomputation at each step;
  the relation between cache size and length.
"""

from __future__ import annotations

import pytest
import torch

from zero.arch.linear_attention import (
    FULL_ATTENTION,
    LINEAR_ATTENTION,
    GatedDeltaNet,
    HybridCache,
    HybridConfig,
    HybridTransformer,
    LinearAttention,
    cache_bytes_per_sequence,
    chunk_gated_delta_rule,
    chunk_linear_attention,
    generate_greedy,
    hybrid_layer_types,
    l2norm,
    recurrent_gated_delta_rule,
    recurrent_linear_attention,
)


def _inputs(B=2, H=3, T=37, dk=8, dv=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(B, H, T, dk, generator=g)
    k = l2norm(torch.randn(B, H, T, dk, generator=g))
    v = torch.randn(B, H, T, dv, generator=g)
    logdecay = -torch.rand(B, H, T, generator=g) * 0.5
    beta = torch.rand(B, H, T, generator=g)
    s0 = torch.randn(B, H, dk, dv, generator=g) * 0.3
    return q, k, v, logdecay, beta, s0


@pytest.mark.parametrize("use_decay", [False, True])
@pytest.mark.parametrize("chunk", [1, 8, 16, 64])
def test_linear_attention_chunk_equals_recurrent(use_decay: bool, chunk: int) -> None:
    q, k, v, g, _, s0 = _inputs()
    g = g if use_decay else None
    o1, s1 = recurrent_linear_attention(q, k, v, g, initial_state=s0)
    o2, s2 = chunk_linear_attention(q, k, v, g, chunk_size=chunk, initial_state=s0)
    torch.testing.assert_close(o1, o2, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(s1, s2, atol=1e-5, rtol=1e-5)


def test_linear_attention_equals_masked_parallel_form() -> None:
    """Without decay: O = (QKᵀ ⊙ M) V. The associativity of matrix multiplication replaces
    the T×T matrix with a d×d state.
    """
    q, k, v, _, _, _ = _inputs()
    T = q.shape[2]
    mask = torch.ones(T, T).tril()
    ref = ((q @ k.transpose(-1, -2)) * mask) @ v
    o, _ = recurrent_linear_attention(q, k, v)
    torch.testing.assert_close(o, ref, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("use_decay", [False, True])
@pytest.mark.parametrize("chunk", [1, 5, 16, 64])
def test_gated_delta_chunk_equals_recurrent(use_decay: bool, chunk: int) -> None:
    q, k, v, g, beta, s0 = _inputs(seed=1)
    g = g if use_decay else None
    o1, s1 = recurrent_gated_delta_rule(q, k, v, g, beta, initial_state=s0)
    o2, s2 = chunk_gated_delta_rule(q, k, v, g, beta, chunk_size=chunk, initial_state=s0)
    torch.testing.assert_close(o1, o2, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(s1, s2, atol=1e-4, rtol=1e-4)


def test_gated_delta_matches_naive_matrix_reference() -> None:
    """Use the explicit matrices at each step: S_t = α_t (I − β_t k_t k_tᵀ) S_{t−1} + β_t k_t v_tᵀ, o_t = S_tᵀ q_t."""
    q, k, v, g, beta, s0 = _inputs(B=1, H=2, T=20, seed=2)
    dk = q.shape[-1]
    o, s = recurrent_gated_delta_rule(q, k, v, g, beta, initial_state=s0)
    for h in range(2):
        S = s0[0, h].clone()
        for t in range(20):
            kt, vt, bt, at = k[0, h, t], v[0, h, t], beta[0, h, t], g[0, h, t].exp()
            S = at * (torch.eye(dk) - bt * torch.outer(kt, kt)) @ S + bt * torch.outer(kt, vt)
            torch.testing.assert_close(o[0, h, t], S.T @ q[0, h, t], atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(s[0, h], S, atol=1e-5, rtol=1e-5)


def test_delta_rule_overwrites_same_key() -> None:
    """Write v1 and then v2 with the same key (β=1): the delta rule reads v2,
    basic linear attention reads v1+v2.
    """
    k = l2norm(torch.randn(1, 1, 1, 8)).repeat(1, 1, 2, 1)
    v = torch.randn(1, 1, 2, 4)
    beta = torch.ones(1, 1, 2)
    _, S_delta = recurrent_gated_delta_rule(k, k, v, None, beta)
    _, S_lin = recurrent_linear_attention(k, k, v)
    read = k[0, 0, 0]
    torch.testing.assert_close(S_delta[0, 0].T @ read, v[0, 0, 1], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(S_lin[0, 0].T @ read, v[0, 0, 0] + v[0, 0, 1], atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("cls", [LinearAttention, GatedDeltaNet])
@pytest.mark.parametrize("use_decay", [False, True])
def test_layer_chunk_recurrent_and_streaming_agree(cls, use_decay: bool) -> None:
    torch.manual_seed(0)
    layer = cls(
        32,
        num_k_heads=2,
        num_v_heads=4,
        head_k_dim=8,
        head_v_dim=8,
        use_decay=use_decay,
        chunk_size=8,
    ).eval()
    x = torch.randn(2, 23, 32)
    with torch.no_grad():
        full_chunk = layer(x, mode="chunk")
        full_rec = layer(x, mode="recurrent")
        # Three segments: a prefill of 10 tokens (chunked) + one token (recurrent) + 12 more tokens (chunked).
        cache = HybridCache(kv=None, kv_slot={})
        parts = [
            layer(x[:, :10], cache=cache),
            layer(x[:, 10:11], cache=cache),
            layer(x[:, 11:], cache=cache),
        ]
    torch.testing.assert_close(full_chunk, full_rec, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(torch.cat(parts, 1), full_chunk, atol=1e-5, rtol=1e-4)
    # The state size does not depend on the sequence length.
    st = cache.linear[0]
    assert st.recurrent is not None and st.recurrent.shape == (2, 4, 8, 8)
    assert st.conv is not None and st.conv.shape == (2, layer.conv_dim, 3)


def test_gated_deltanet_matches_hf_qwen3_5() -> None:
    """Copy the weights of HF transformers Qwen3_5GatedDeltaNet into the zero GatedDeltaNet. The outputs agree."""
    mod = pytest.importorskip("transformers.models.qwen3_5.modeling_qwen3_5")
    cfg_mod = pytest.importorskip("transformers.models.qwen3_5.configuration_qwen3_5")
    cfg = cfg_mod.Qwen3_5TextConfig(
        hidden_size=48,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=8,
        linear_value_head_dim=12,
        linear_conv_kernel_dim=4,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=12,
        intermediate_size=64,
        vocab_size=100,
        layer_types=hybrid_layer_types(4, 4),
    )
    torch.manual_seed(0)
    hf = mod.Qwen3_5GatedDeltaNet(cfg, layer_idx=0).eval()
    with torch.no_grad():
        for p in hf.parameters():  # randomize all weights (norm and dt_bias too) to prevent a lucky match "because the value is 1"
            p.copy_(torch.randn_like(p) * 0.3)
    ours = GatedDeltaNet(48, 2, 4, 8, 12, conv_kernel=4, use_decay=True, chunk_size=16).eval()
    missing, unexpected = ours.load_state_dict(hf.state_dict(), strict=True)
    assert not missing and not unexpected
    x = torch.randn(2, 29, 48)
    with torch.no_grad():
        ref = hf(x)
        out = ours(x)
    torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-4)


def _tiny_hybrid(
    interval: int = 2, kind: str = "gated_deltanet", seed: int = 0
) -> HybridTransformer:
    torch.manual_seed(seed)
    cfg = HybridConfig(
        vocab_size=61,
        dim=32,
        n_layers=4,
        n_heads=4,
        n_kv_heads=2,
        ffn_dim=64,
        max_seq_len=128,
        init_std=0.1,
        full_attention_interval=interval,
        linear_kind=kind,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_chunk_size=8,
    )
    return HybridTransformer(cfg).eval()


@pytest.mark.parametrize("kind", ["gated_deltanet", "linear"])
@pytest.mark.parametrize("interval", [2, 4, 99])
def test_hybrid_cached_generation_equals_recompute(kind: str, interval: int) -> None:
    model = _tiny_hybrid(interval, kind)
    prompt = [3, 14, 15, 9, 26, 5, 35, 8, 9, 7, 9]
    a = generate_greedy(model, prompt, 30, use_cache=True)
    b = generate_greedy(model, prompt, 30, use_cache=False)
    assert a == b and len(a) == 30


def test_hybrid_chunked_prefill_matches_full_forward() -> None:
    model = _tiny_hybrid(2)
    tokens = torch.randint(0, 61, (2, 45), generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        full = model(tokens)
        cache = model.new_cache(2, 64)
        a = model(tokens[:, :20], cache, 0)
        b = model(tokens[:, 20:21], cache, 20)
        c = model(tokens[:, 21:], cache, 21)
    torch.testing.assert_close(torch.cat([a, b, c], 1), full, atol=1e-4, rtol=1e-4)


def test_layer_types_and_cache_accounting() -> None:
    assert hybrid_layer_types(8, 4) == [LINEAR_ATTENTION] * 3 + [FULL_ATTENTION] + [
        LINEAR_ATTENTION
    ] * 3 + [FULL_ATTENTION]
    assert hybrid_layer_types(3, 1) == [FULL_ATTENTION] * 3
    model = _tiny_hybrid(4)
    assert model.full_layers == [3]
    cache = model.new_cache(1, 100)
    assert cache.kv is not None and cache.kv.k.shape[0] == 1  # KV is allocated only for the 1 full attention layer
    lt = hybrid_layer_types(24, 4)
    short = cache_bytes_per_sequence(lt, 1000, 2, 256, 16, 128, 128, conv_dim=6144)
    long = cache_bytes_per_sequence(lt, 2000, 2, 256, 16, 128, 128, conv_dim=6144)
    assert long["kv"] == 2 * short["kv"] and long["linear_state"] == short["linear_state"]


def test_hybrid_trains() -> None:
    """A few gradient descent steps must decrease the loss
    (the gradient goes through the triangular solve in the chunked operator).
    """
    model = _tiny_hybrid(2).train()
    x = torch.randint(0, 61, (4, 33), generator=torch.Generator().manual_seed(0))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    losses = []
    for _ in range(15):
        loss = model.loss(x[:, :-1], x[:, 1:])
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0] - 0.5


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("kind", ["gated_deltanet", "linear"])
def test_generate_greedy_on_cuda(kind: str) -> None:
    """When the model is on CUDA, generate_greedy must also put the inputs and the cache on CUDA
    (found on RTX 3090 in 2026-10).
    """
    model = _tiny_hybrid(2, kind).cuda()
    prompt = [3, 14, 15, 9, 26, 5, 35, 8, 9, 7, 9]
    a = generate_greedy(model, prompt, 20, use_cache=True)
    b = generate_greedy(model, prompt, 20, use_cache=False)
    assert a == b and len(a) == 20
