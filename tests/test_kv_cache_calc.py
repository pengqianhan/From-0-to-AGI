"""KV cache 计算器（zero/tools/kv_cache_calc.py，第 21 章）：公式与真实分配对拍，各种层类型按预期记账。"""

from __future__ import annotations

import torch

from zero.arch.mla import MLACache, MLAConfig
from zero.config import load_model_config
from zero.kv_cache import KVCache
from zero.tools.kv_cache_calc import (
    breakdown,
    fixed_state_bytes,
    kv_bytes_per_token,
    kv_cache_bytes,
    main,
)


def test_matches_real_kvcache_for_zero_configs() -> None:
    for path in ("configs/tiny/pretrain.toml", "configs/main/pretrain.toml"):
        cfg = load_model_config(path)
        for dtype, nb in ((torch.float32, 4), (torch.bfloat16, 2)):
            # meta 设备：只记形状不真正分配内存，主线配置也瞬间完成
            cache = KVCache.from_config(
                cfg, batch_size=2, max_seq_len=1000, device="meta", dtype=dtype
            )
            assert kv_cache_bytes(cfg, 1000, batch=2, dtype_bytes=nb) == cache.nbytes()
            assert kv_cache_bytes(path, 1000, batch=2, dtype_bytes=nb) == cache.nbytes()


def test_main_model_numbers() -> None:
    # 28 层 × 8 个 KV 头 × head_dim 128 × 2（K 和 V）× 2 字节 = 114,688 字节 = 112 KiB
    assert kv_bytes_per_token("configs/main/pretrain.toml") == 114_688
    assert kv_cache_bytes("configs/main/pretrain.toml", 32768) == 114_688 * 32768


def test_mla_matches_mlacache() -> None:
    hf = {
        "num_hidden_layers": 3,
        "num_attention_heads": 4,
        "hidden_size": 64,
        "num_key_value_heads": 4,
        "kv_lora_rank": 24,
        "qk_rope_head_dim": 8,
        "qk_nope_head_dim": 16,
        "v_head_dim": 16,
    }
    cfg = MLAConfig(
        dim=64, n_heads=4, kv_lora_rank=24, qk_nope_head_dim=16, qk_rope_head_dim=8, v_head_dim=16
    )
    cache = MLACache.from_config(cfg, 3, batch_size=2, max_seq_len=77, device="meta")
    assert kv_cache_bytes(hf, 77, batch=2, dtype_bytes=4) == cache.nbytes()


def test_deepseek_v3_mla() -> None:
    # 数字来自 deepseek-ai/DeepSeek-V3 的 config.json：61 层，kv_lora_rank 512，qk_rope_head_dim 64
    hf = {
        "num_hidden_layers": 61,
        "num_attention_heads": 128,
        "num_key_value_heads": 128,
        "hidden_size": 7168,
        "kv_lora_rank": 512,
        "qk_rope_head_dim": 64,
        "qk_nope_head_dim": 128,
        "v_head_dim": 128,
    }
    assert kv_bytes_per_token(hf) == 61 * 576 * 2  # 70,272 字节 ≈ 68.6 KiB


def test_sliding_window_and_hybrid() -> None:
    # gpt-oss 风格：一半层是 128 窗口的滑动注意力
    hf = {
        "num_hidden_layers": 4,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "sliding_window": 128,
        "layer_types": ["sliding_attention", "full_attention"] * 2,
    }
    per_layer = 2 * 2 * 16
    assert kv_cache_bytes(hf, 1000, dtype_bytes=1) == 2 * per_layer * 1000 + 2 * per_layer * 128
    # Qwen3.5 风格：3 层线性注意力 + 1 层全注意力，只有全注意力层随长度增长
    q = {
        "text_config": {
            "num_hidden_layers": 4,
            "num_attention_heads": 8,
            "num_key_value_heads": 2,
            "head_dim": 32,
            "layer_types": ["linear_attention"] * 3 + ["full_attention"],
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 2,
            "linear_key_head_dim": 8,
            "linear_value_head_dim": 8,
            "linear_conv_kernel_dim": 4,
            "mamba_ssm_dtype": "float32",
        }
    }
    assert kv_cache_bytes(q, 500) == 1 * 2 * 2 * 32 * 500 * 2
    state = 2 * 8 * 8 + 3 * (2 * 2 * 8 + 2 * 8)  # 递推状态 + 短卷积缓存（每层）
    assert fixed_state_bytes(q) == 3 * state * 4
    r = breakdown(q, 500)
    assert r["by_kind"]["linear"]["layers"] == 3 and r["by_kind"]["full"]["layers"] == 1


def test_mistral_params_json_and_all_sliding() -> None:
    # Mistral 原生 params.json 字段名
    p = {
        "n_layers": 2,
        "n_heads": 4,
        "n_kv_heads": 4,
        "dim": 64,
        "head_dim": 16,
        "kv_lora_rank": 32,
        "qk_rope_head_dim": 8,
    }
    assert kv_bytes_per_token(p, dtype_bytes=1) == 2 * 40
    # Mistral-7B-v0.1 风格：没有 layer_types，但设置了 sliding_window → 全部层滑动
    m = {
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 1,
        "hidden_size": 64,
        "sliding_window": 10,
    }
    assert kv_cache_bytes(m, 100, dtype_bytes=1) == 2 * (2 * 16) * 10


def test_cli_runs(capsys) -> None:
    main(["configs/main/pretrain.toml", "--seq", "32768"])
    out = capsys.readouterr().out
    assert "112.00 KiB" in out and "3.50 GiB" in out
