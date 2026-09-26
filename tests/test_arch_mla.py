"""MLA（zero/arch/mla.py，第 21 章）的正确性：吸收路径 = 显式路径；带缓存生成 = 不带缓存生成。"""

from __future__ import annotations

import torch

from zero.arch.mla import MLAAttention, MLACache, MLAConfig, generate_greedy, mla_transformer
from zero.config import ModelConfig


def _cfgs(q_lora_rank: int | None = None) -> tuple[ModelConfig, MLAConfig]:
    mc = ModelConfig(
        vocab_size=97, dim=64, n_layers=2, n_heads=4, n_kv_heads=4, head_dim=16,
        ffn_dim=128, max_seq_len=96, init_std=0.1,
    )
    return mc, MLAConfig.from_model_config(mc, kv_lora_rank=24, qk_rope_head_dim=8,
                                           q_lora_rank=q_lora_rank)


def test_absorbed_equals_naive() -> None:
    _, cfg = _cfgs(q_lora_rank=32)
    torch.manual_seed(0)
    attn = MLAAttention(cfg).double()
    x = torch.randn(2, 11, cfg.dim, dtype=torch.float64)
    attn.absorb = False
    a = attn(x)
    attn.absorb = True
    b = attn(x)
    assert torch.allclose(a, b, atol=1e-10)


def test_chunked_prefill_with_cache_matches_full_forward() -> None:
    mc, cfg = _cfgs()
    torch.manual_seed(1)
    model = mla_transformer(mc, cfg).eval()
    tokens = torch.randint(0, mc.vocab_size, (1, 30))
    full = model(tokens)
    cache = MLACache.from_config(cfg, mc.n_layers, 1, 30)
    parts = [model(tokens[:, :12], kv_cache=cache, start_pos=0),
             model(tokens[:, 12:13], kv_cache=cache, start_pos=12),
             model(tokens[:, 13:], kv_cache=cache, start_pos=13)]
    assert torch.allclose(full, torch.cat(parts, dim=1), atol=1e-5)


def test_cached_generation_identical() -> None:
    mc, cfg = _cfgs(q_lora_rank=32)
    torch.manual_seed(2)
    model = mla_transformer(mc, cfg)
    prompt = [5, 17, 42, 3, 8]
    a = generate_greedy(model, prompt, 40, cfg, use_cache=True)
    b = generate_greedy(model, prompt, 40, cfg, use_cache=False)
    assert a == b and len(a) == 40


def test_cache_bytes_formula() -> None:
    mc, cfg = _cfgs()
    cache = MLACache.from_config(cfg, mc.n_layers, batch_size=3, max_seq_len=50)
    # 每层每位置 kv_lora_rank + qk_rope_head_dim 个数，FP32 4 字节
    assert cache.nbytes() == mc.n_layers * 3 * 50 * (24 + 8) * 4


def test_mla_trains() -> None:
    mc, cfg = _cfgs()
    torch.manual_seed(3)
    model = mla_transformer(mc, cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    x = torch.randint(0, mc.vocab_size, (4, 24))
    first = None
    for _ in range(30):
        loss = model.loss(x[:, :-1], x[:, 1:])
        first = first if first is not None else loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < first * 0.7
