"""zero.Transformer 与 Hugging Face 官方 Qwen3ForCausalLM 的 logits 对拍（GOAL.md 9.1）。

随机初始化一个很小的 Qwen3（GQA + QK-Norm + SwiGLU + RMSNorm + RoPE），把权重搬进 zero，
同一批输入的 logits 必须一致。覆盖：共享/不共享 embedding、默认 RoPE、YaRN 缩放、head_dim != dim/n_heads。
"""

from __future__ import annotations

import pytest
import torch

transformers = pytest.importorskip("transformers")
from transformers import Qwen3Config, Qwen3ForCausalLM  # noqa: E402

from zero.hf import config_from_hf_qwen3, load_from_hf_qwen3  # noqa: E402
from zero.model import count_params  # noqa: E402

CASES = {
    "tied_default_rope": dict(tie_word_embeddings=True),
    "untied": dict(tie_word_embeddings=False),
    "yarn": dict(
        tie_word_embeddings=True,
        rope_theta=10000.0,
        rope_scaling={"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 32},
    ),
    "yarn_custom_beta": dict(
        tie_word_embeddings=True,
        rope_theta=50000.0,
        rope_scaling={
            "rope_type": "yarn",
            "factor": 2.5,
            "original_max_position_embeddings": 48,
            "beta_fast": 16.0,
            "beta_slow": 2.0,
        },
    ),
    "mha_big_head": dict(
        num_key_value_heads=4, head_dim=48
    ),  # 无 GQA，且 n_heads*head_dim != hidden
}


def make_hf(**overrides) -> Qwen3ForCausalLM:
    kw = dict(
        vocab_size=211,
        hidden_size=64,
        intermediate_size=160,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        max_position_embeddings=160,
        rms_norm_eps=1e-6,
    )
    kw.update(overrides)
    torch.manual_seed(0)
    model = Qwen3ForCausalLM(Qwen3Config(**kw)).eval()
    # 随机化所有权重（包括 RMSNorm），让对拍更严格
    with torch.no_grad():
        for p in model.parameters():
            if p.dim() > 1:
                p.normal_(0, 0.08)
            else:
                p.uniform_(0.5, 1.5)
    return model


@pytest.mark.parametrize("case", list(CASES))
def test_logits_match_hf(case: str) -> None:
    hf = make_hf(**CASES[case])
    zero_model = load_from_hf_qwen3(hf).eval()
    tokens = torch.randint(0, 211, (2, 150), generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        ref = hf(tokens).logits
        out = zero_model(tokens)
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


def test_config_roundtrip_and_param_count() -> None:
    hf = make_hf(tie_word_embeddings=False)
    cfg = config_from_hf_qwen3(hf.config)
    assert cfg.dim == 64 and cfg.n_kv_heads == 2 and not cfg.tie_embeddings
    hf_params = sum(p.numel() for p in hf.parameters())
    zero_model = load_from_hf_qwen3(hf)
    assert zero_model.num_params() == hf_params == count_params(cfg)["total"]


def test_loss_matches_hf() -> None:
    hf = make_hf()
    zero_model = load_from_hf_qwen3(hf).eval()
    tokens = torch.randint(0, 211, (2, 40), generator=torch.Generator().manual_seed(2))
    with torch.no_grad():
        ref = hf(tokens, labels=tokens).loss  # HF 内部会把 labels 右移一位
        ours = zero_model.loss(tokens[:, :-1], tokens[:, 1:])
    torch.testing.assert_close(ours, ref, rtol=1e-5, atol=1e-5)


def test_load_from_state_dict_requires_config() -> None:
    hf = make_hf()
    with pytest.raises(ValueError):
        load_from_hf_qwen3(hf.state_dict())
    m = load_from_hf_qwen3(hf.state_dict(), config_from_hf_qwen3(hf.config))
    assert m.num_params() > 0
