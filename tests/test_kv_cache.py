"""KV cache consistency: generation with and without the cache must give exactly the same result (GOAL.md 9.1)."""

from __future__ import annotations

import torch

from zero.config import ModelConfig
from zero.generate import generate, sample_next
from zero.kv_cache import KVCache
from zero.model import Transformer


def tiny_model(seed: int = 0, **kw) -> Transformer:
    cfg = dict(
        vocab_size=97,
        dim=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        ffn_dim=128,
        max_seq_len=96,
        init_std=0.1,
    )
    cfg.update(kw)
    torch.manual_seed(seed)
    return Transformer(ModelConfig(**cfg)).eval()


def test_greedy_cached_equals_uncached() -> None:
    model = tiny_model()
    prompt = [5, 17, 42, 3, 8]
    a = generate(model, prompt, 40, temperature=0.0, use_cache=True)
    b = generate(model, prompt, 40, temperature=0.0, use_cache=False)
    assert a == b and len(a) == 40


def test_greedy_batch_and_yarn() -> None:
    model = tiny_model(
        rope_scaling={"type": "yarn", "factor": 2.0, "original_max_position_embeddings": 32},
        n_kv_heads=1,
    )
    prompt = torch.randint(0, 97, (3, 7), generator=torch.Generator().manual_seed(0))
    a = generate(model, prompt, 30, temperature=0.0, use_cache=True)
    b = generate(model, prompt, 30, temperature=0.0, use_cache=False)
    assert a == b and len(a) == 3


def test_sampling_with_seed_cached_equals_uncached() -> None:
    model = tiny_model(1)
    prompt = [1, 2, 3]
    a = generate(model, prompt, 25, temperature=0.8, top_p=0.9, use_cache=True, seed=123)
    b = generate(model, prompt, 25, temperature=0.8, top_p=0.9, use_cache=False, seed=123)
    c = generate(model, prompt, 25, temperature=0.8, top_p=0.9, use_cache=True, seed=123)
    assert a == b == c


def test_chunked_prefill_matches_full_forward() -> None:
    """The mask must also be correct with history and more than one new token at once (chunked prefill)."""
    model = tiny_model(2)
    tokens = torch.randint(0, 97, (2, 50), generator=torch.Generator().manual_seed(3))
    with torch.no_grad():
        full = model(tokens)
        cache = KVCache.from_config(model.config, batch_size=2)
        parts = [
            model(tokens[:, :20], cache, 0),
            model(tokens[:, 20:21], cache, 20),
            model(tokens[:, 21:], cache, 21),
        ]
    torch.testing.assert_close(torch.cat(parts, dim=1), full, rtol=1e-5, atol=1e-5)


def test_eos_stops_generation() -> None:
    model = tiny_model()
    prompt = [5, 17, 42]
    ref = generate(model, prompt, 20, temperature=0.0)
    eos = ref[4]
    out = generate(model, prompt, 20, temperature=0.0, eos_id=eos)
    assert out == ref[: ref.index(eos)]


def test_sample_next_top_p_and_greedy() -> None:
    logits = torch.tensor([[0.0, 5.0, 4.9, -10.0]])
    assert sample_next(logits, temperature=0).item() == 1
    g = torch.Generator().manual_seed(0)
    # With a very small top_p, only the token with the highest probability remains
    for _ in range(20):
        assert sample_next(logits, temperature=1.0, top_p=0.1, generator=g).item() == 1
    # With top_p=0.99, both high-probability tokens occur; token 3, with a very small probability, never occurs
    seen = {sample_next(logits, 1.0, 0.99, g).item() for _ in range(200)}
    assert seen <= {1, 2, 0} and {1, 2} <= seen


def test_kv_cache_nbytes() -> None:
    cfg = ModelConfig(
        vocab_size=10, dim=64, n_layers=3, n_heads=4, n_kv_heads=2, ffn_dim=64, max_seq_len=100
    )
    cache = KVCache.from_config(cfg, batch_size=2)
    # 2(K,V) × layers × batch × KV heads × length × head_dim × 4 bytes
    assert cache.nbytes() == 2 * 3 * 2 * 2 * 100 * 16 * 4
