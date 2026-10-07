"""Correctness of speculative decoding (zero/arch/speculative.py, Chapter 25):
greedy output is identical to greedy decoding of the target model; with draft = target, the
acceptance rate is 100%; after a cache rollback, the continuation agrees with a full recomputation;
the distribution of one-step rejection sampling equals the target distribution.
"""

from __future__ import annotations

import pytest
import torch

from zero.arch.speculative import (
    expected_speedup,
    expected_tokens_per_round,
    prompt_lookup_draft,
    rollback,
    speculative_generate,
    verify,
    warp_probs,
)
from zero.config import ModelConfig
from zero.generate import generate
from zero.kv_cache import KVCache
from zero.model import Transformer


def _model(seed: int, dim: int = 64, n_layers: int = 2) -> Transformer:
    torch.manual_seed(seed)
    cfg = ModelConfig(
        vocab_size=53,
        dim=dim,
        n_layers=n_layers,
        n_heads=4,
        n_kv_heads=2,
        head_dim=dim // 4,
        ffn_dim=2 * dim,
        max_seq_len=128,
        init_std=0.2,  # a larger init makes the distribution of the random model "sharper", so the greedy outputs differ more
    )
    return Transformer(cfg).eval()


PROMPT = [3, 17, 42, 5, 8, 8, 30]


@pytest.mark.parametrize("k", [1, 3, 6])
def test_greedy_matches_target_greedy(k: int) -> None:
    target, draft = _model(0), _model(1, dim=32, n_layers=1)
    ref = generate(target, PROMPT, 40, temperature=0.0)
    res = speculative_generate(target, draft, PROMPT, 40, k=k, temperature=0.0)
    assert res.tokens == ref
    assert res.rounds <= len(ref)  # the target model does not do more forward passes than normal decoding
    assert res.rounds + res.accepted >= len(ref)  # each round = number accepted + 1 correction/bonus token


def test_prompt_lookup_greedy_matches_target_greedy() -> None:
    target = _model(0)
    prompt = [1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2]
    ref = generate(target, prompt, 30, temperature=0.0)
    res = speculative_generate(target, None, prompt, 30, k=4, temperature=0.0)
    assert res.tokens == ref


def test_draft_equal_target_accepts_everything() -> None:
    target = _model(0)
    for temp in (0.0, 1.0):
        res = speculative_generate(target, target, PROMPT, 33, k=4, temperature=temp, seed=0)
        assert len(res.tokens) == 33
        assert res.acceptance_rate == 1.0
        assert res.accepted == res.proposed
        # Each round: k drafts + 1 bonus token; 33 = 6 rounds × 5 + 3 in the last round.
        assert res.accepted_per_round[:6] == [4] * 6


def test_cache_rollback_then_continue_matches_full_forward() -> None:
    model = _model(0)
    seq = torch.tensor([PROMPT + [11, 12, 13, 14]])
    full = model(seq)
    cache = KVCache.from_config(model.config, 1, 64)
    model(seq[:, :7], kv_cache=cache, start_pos=0)
    model(torch.tensor([[40, 41, 42, 43]]), kv_cache=cache, start_pos=7)  # the "rejected drafts" go into the cache
    rollback(cache, 7)
    assert cache.seq_len == 7
    cont = model(seq[:, 7:], kv_cache=cache, start_pos=7)  # overwrite with the correct tokens
    assert torch.allclose(cont, full[:, 7:], atol=1e-5)


def test_sampling_is_reproducible_and_valid() -> None:
    target, draft = _model(0), _model(1, dim=32, n_layers=1)
    a = speculative_generate(target, draft, PROMPT, 30, k=3, temperature=0.8, top_p=0.9, seed=7)
    b = speculative_generate(target, draft, PROMPT, 30, k=3, temperature=0.8, top_p=0.9, seed=7)
    assert a.tokens == b.tokens and len(a.tokens) == 30
    assert all(0 <= t < 53 for t in a.tokens)
    assert 0.0 <= a.acceptance_rate <= 1.0


def test_verify_preserves_target_distribution() -> None:
    """One position: sample the draft from q; after verify, the token distribution must equal p
    (law of large numbers, TV < 0.01).
    """
    torch.manual_seed(0)
    V, N = 5, 40_000
    p_logits = torch.randn(2, V) * 1.5
    q = torch.softmax(torch.randn(V) * 1.5, -1)
    p = warp_probs(p_logits[0], 1.0)
    g = torch.Generator().manual_seed(1)
    xs = torch.multinomial(q, N, replacement=True, generator=g).tolist()
    counts = torch.zeros(V)
    acc = 0
    for x in xs:
        m, new = verify(p_logits, [x], q[None], 1.0, 1.0, g)
        counts[new[0]] += 1
        acc += m
    assert 0.5 * (counts / N - p).abs().sum() < 0.01
    assert abs(acc / N - torch.minimum(p, q).sum()) < 0.01  # acceptance rate = Σ min(p, q)


def test_warp_probs_top_p_matches_generate_rule() -> None:
    logits = torch.tensor([3.0, 2.0, 1.0, 0.0])
    p = warp_probs(logits, 1.0, top_p=0.7)
    full = torch.softmax(logits, -1)
    assert p[2] == 0 and p[3] == 0  # the cumulative probability of the first two is already more than 0.7
    assert torch.allclose(p[:2], full[:2] / full[:2].sum())


def test_prompt_lookup_and_formulas() -> None:
    assert prompt_lookup_draft([7, 8, 9, 1, 7, 8], k=2) == [9, 1]
    assert prompt_lookup_draft([1, 2, 3], k=2) == []
    assert expected_tokens_per_round(0.0, 5) == 1.0
    assert abs(expected_tokens_per_round(0.8, 4) - (1 - 0.8**5) / 0.2) < 1e-12
    assert expected_speedup(1.0, 4, 0.0) == 5.0
