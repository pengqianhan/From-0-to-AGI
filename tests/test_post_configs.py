"""The post-training / evaluation configs in configs/tiny and configs/main parse.

The model shape is the same as in the upstream stage (Chapters 16–20).
"""

from __future__ import annotations

import dataclasses

import pytest

from zero.config import load_config
from zero.eval.harness import load_eval_config
from zero.post.common import load_post_config
from zero.post.distill import DistillConfig, TeacherConfig
from zero.post.dpo import DPOConfig
from zero.post.grpo import GRPOConfig
from zero.post.opd import OPDConfig
from zero.post.sft import SFTConfig

SECTIONS = {
    "sft": {"sft": SFTConfig},
    "distill": {"teacher": TeacherConfig, "distill": DistillConfig},
    "dpo": {"dpo": DPOConfig},
    "grpo": {"grpo": GRPOConfig},
    "opd": {"opd": OPDConfig},
}
SHAPE = (
    "vocab_size",
    "dim",
    "n_layers",
    "n_heads",
    "n_kv_heads",
    "head_dim",
    "ffn_dim",
    "tie_embeddings",
)


@pytest.mark.parametrize("tier,upstream", [("tiny", "pretrain"), ("main", "pretrain")])
@pytest.mark.parametrize("stage", list(SECTIONS))
def test_post_configs_parse_and_match_shape(tier: str, upstream: str, stage: str) -> None:
    cfg, sec = load_post_config(f"configs/{tier}/{stage}.toml", SECTIONS[stage])
    base = load_config(f"configs/{tier}/{upstream}.toml").model
    for k in SHAPE:
        assert getattr(cfg.model, k) == getattr(base, k), (tier, stage, k)
    assert cfg.train.data.format == ("sft" if stage in ("sft", "distill") else "none")
    if tier == "tiny":
        assert cfg.train.cpu_threads == 1
    if stage == "distill" and tier == "main":
        assert sec["teacher"].license_allows_distillation is False  # refuse to run until the license is verified


QWEN3_06B = {  # Qwen/Qwen3-0.6B-Base config.json
    "vocab_size": 151936,
    "dim": 1024,
    "n_layers": 28,
    "n_heads": 16,
    "n_kv_heads": 8,
    "head_dim": 128,
    "ffn_dim": 3072,
    "tie_embeddings": True,
}


@pytest.mark.parametrize("stage", list(SECTIONS))
def test_proxy_configs_inherit_main_and_use_qwen3_shape(stage: str) -> None:
    cfg, sec = load_post_config(f"configs/proxy/{stage}.toml", SECTIONS[stage])
    main, main_sec = load_post_config(f"configs/main/{stage}.toml", SECTIONS[stage])
    for k, v in QWEN3_06B.items():
        assert getattr(cfg.model, k) == v, (stage, k)
    # Same recipe as the main line: only the model, the tokenizer, and the paths change
    assert cfg.train.optim == main.train.optim and cfg.train.max_steps == main.train.max_steps
    assert cfg.train.data.tokenizer == "out/proxy/base/tokenizer.json"
    assert cfg.train.out_dir.startswith("out/proxy/") and cfg.train.init_from.startswith("out/proxy/")
    if stage == "opd":
        assert [t.path for t in sec["opd"].teachers] == ["out/proxy/distill/ckpt", "out/proxy/grpo/ckpt"]
        assert dataclasses.replace(sec["opd"], teachers=[]) == dataclasses.replace(main_sec["opd"], teachers=[])


@pytest.mark.parametrize("stage", list(SECTIONS))
def test_weak_configs_match_pretrain_checkpoint(stage: str) -> None:
    cfg, sec = load_post_config(f"configs/weak/{stage}.toml", SECTIONS[stage])
    main, _ = load_post_config(f"configs/main/{stage}.toml", SECTIONS[stage])
    pre = load_config("configs/main/pretrain.toml").model
    for k in (*SHAPE, "rope_theta", "max_seq_len"):
        assert getattr(cfg.model, k) == getattr(pre, k), (stage, k)
    assert cfg.train.optim == main.train.optim and cfg.train.max_steps == main.train.max_steps
    assert cfg.train.data.seq_len <= pre.max_seq_len
    assert cfg.train.out_dir.startswith("out/weak/") and cfg.train.init_from.startswith("out/weak/")
    if stage == "opd":
        assert all(t.path.startswith("out/weak/") for t in sec["opd"].teachers)


@pytest.mark.parametrize("tier", ["tiny", "main", "proxy", "weak"])
def test_eval_configs_parse(tier: str) -> None:
    ec = load_eval_config(f"configs/{tier}/eval.toml")
    assert ec.models and ec.baseline in {m.name for m in ec.models}


def test_unknown_section_rejected(tmp_path) -> None:  # noqa: ANN001
    from zero.config import ConfigError

    with pytest.raises(ConfigError, match="Unknown config section"):
        load_post_config({"dpoo": {}}, {"dpo": DPOConfig})
    with pytest.raises(ConfigError, match="did you mean 'beta'"):
        load_post_config({"dpo": {"betta": 0.1}, "data": {"format": "none"}}, {"dpo": DPOConfig})
