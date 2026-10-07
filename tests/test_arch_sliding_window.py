"""Correctness tests of sliding window / local-global attention (zero/arch/sliding_window.py, Chapter 22)."""

from __future__ import annotations

import torch

from zero.arch.sliding_window import (
    FULL,
    SLIDING,
    SlidingWindowKVCache,
    convert_to_sliding_window,
    generate_greedy,
    kv_cache_bytes,
    make_layer_types,
    sliding_window_mask,
)
from zero.config import ModelConfig
from zero.kv_cache import KVCache
from zero.model import Transformer


def tiny_model(seed: int = 0, **kw) -> Transformer:
    cfg = dict(
        vocab_size=61,
        dim=48,
        n_layers=4,
        n_heads=4,
        n_kv_heads=2,
        ffn_dim=96,
        max_seq_len=128,
        init_std=0.1,
    )
    cfg.update(kw)
    torch.manual_seed(seed)
    return Transformer(ModelConfig(**cfg)).eval()


def test_mask_matches_definition() -> None:
    pos = torch.arange(6)
    m = sliding_window_mask(pos, pos, window=3)
    expected = torch.tensor(
        [
            [1, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0],
            [0, 1, 1, 1, 0, 0],
            [0, 0, 1, 1, 1, 0],
            [0, 0, 0, 1, 1, 1],
        ],
        dtype=torch.bool,
    )
    assert torch.equal(m, expected)
    # Each query sees a maximum of window positions (itself included); None is the normal causal mask.
    assert int(m.sum(1).max()) == 3
    assert torch.equal(sliding_window_mask(pos, pos, None), torch.tril(torch.ones(6, 6)).bool())
    # An empty slot (position -1) is never visible; the keys can be in any order.
    k_pos = torch.tensor([4, -1, 2, 3])
    assert sliding_window_mask(torch.tensor([4]), k_pos, 2).tolist() == [[True, False, False, True]]


def test_layer_types_patterns() -> None:
    assert make_layer_types(6, global_every=6) == [SLIDING] * 5 + [FULL]  # Gemma 3: 5:1
    assert make_layer_types(4, global_every=2) == [SLIDING, FULL, SLIDING, FULL]  # gpt-oss: 1:1
    assert make_layer_types(3, all_sliding=True) == [SLIDING] * 3  # Mistral 7B v0.1
    assert make_layer_types(3) == [FULL] * 3


def test_window_covering_sequence_equals_full_attention() -> None:
    model = tiny_model()
    tokens = torch.randint(0, 61, (2, 40), generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        ref = model(tokens)
        convert_to_sliding_window(model, make_layer_types(4, all_sliding=True), window=40)
        out = model(tokens)
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)


def test_small_window_changes_output_only_beyond_window() -> None:
    """Window W, L layers: the outputs at positions < W are identical to full attention
    (they see the same tokens). Only after that do they start to differ.
    """
    model = tiny_model(1)
    tokens = torch.randint(0, 61, (1, 50), generator=torch.Generator().manual_seed(2))
    with torch.no_grad():
        ref = model(tokens)
        convert_to_sliding_window(model, make_layer_types(4, all_sliding=True), window=8)
        out = model(tokens)
    torch.testing.assert_close(out[:, :8], ref[:, :8], atol=1e-5, rtol=1e-5)
    assert (out[:, 8:] - ref[:, 8:]).abs().max() > 1e-3


def test_bounded_cache_generation_equals_uncached() -> None:
    for layer_types in (make_layer_types(4, all_sliding=True), make_layer_types(4, global_every=2)):
        model = tiny_model(2)
        convert_to_sliding_window(model, layer_types, window=6)
        prompt = [3, 14, 15, 9, 26, 5, 35, 8, 9, 7, 9, 3]  # longer than the window: the prefill already writes around the ring
        cache = SlidingWindowKVCache.from_model(model, batch_size=1, max_seq_len=128)
        a = generate_greedy(model, prompt, 60, cache=cache)
        b = generate_greedy(model, prompt, 60, cache=None)
        assert a == b and len(a) == 60


def test_chunked_prefill_with_ring_buffer_matches_full_forward() -> None:
    model = tiny_model(3)
    convert_to_sliding_window(model, make_layer_types(4, global_every=4), window=5)
    tokens = torch.randint(0, 61, (2, 45), generator=torch.Generator().manual_seed(4))
    cache = SlidingWindowKVCache.from_model(model, batch_size=2, max_seq_len=64)
    with torch.no_grad():
        ref = model(tokens)
        parts, pos = [], 0
        for n in (17, 1, 3, 24):  # some chunks are larger than the window, some are smaller
            parts.append(model(tokens[:, pos : pos + n], kv_cache=cache, start_pos=pos))
            pos += n
    torch.testing.assert_close(torch.cat(parts, dim=1), ref, atol=1e-5, rtol=1e-5)


def test_plain_kv_cache_also_works() -> None:
    """A normal (not truncated) KVCache gives the same result.

    It only stores more K/V outside the window that attention does not use.
    """
    model = tiny_model(4)
    convert_to_sliding_window(model, make_layer_types(4, global_every=2), window=4)
    tokens = torch.randint(0, 61, (1, 30), generator=torch.Generator().manual_seed(5))
    cache = KVCache.from_config(model.config, batch_size=1, max_seq_len=30)
    with torch.no_grad():
        ref = model(tokens)
        a = model(tokens[:, :20], kv_cache=cache, start_pos=0)
        b = model(tokens[:, 20:], kv_cache=cache, start_pos=20)
    torch.testing.assert_close(torch.cat([a, b], dim=1), ref, atol=1e-5, rtol=1e-5)


def test_cache_bytes_bounded_by_window() -> None:
    model = tiny_model()
    convert_to_sliding_window(model, make_layer_types(4, global_every=4), window=8)
    cache = SlidingWindowKVCache.from_model(model, batch_size=1, max_seq_len=128)
    per_pos = 2 * 2 * 12 * 4  # K+V × kv heads × head_dim × fp32
    assert cache.nbytes() == (3 * 8 + 1 * 128) * per_pos
    assert cache.nbytes() == kv_cache_bytes(
        make_layer_types(4, global_every=4), 8, 128, 2, 12, bytes_per_elem=4
    )
