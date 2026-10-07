"""The post-training / evaluation configs in configs/tiny and configs/main parse.

The model shape is the same as in the upstream stage (Chapters 16–20).
"""

from __future__ import annotations

import pytest

from zero.config import load_config
from zero.eval.harness import load_eval_config
from zero.post.common import load_post_config
from zero.post.distill import DistillConfig, TeacherConfig
from zero.post.dpo import DPOConfig
from zero.post.grpo import GRPOConfig
from zero.post.sft import SFTConfig

SECTIONS = {
    "sft": {"sft": SFTConfig},
    "distill": {"teacher": TeacherConfig, "distill": DistillConfig},
    "dpo": {"dpo": DPOConfig},
    "grpo": {"grpo": GRPOConfig},
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


@pytest.mark.parametrize("tier", ["tiny", "main"])
def test_eval_configs_parse(tier: str) -> None:
    ec = load_eval_config(f"configs/{tier}/eval.toml")
    assert ec.models and ec.baseline in {m.name for m in ec.models}


def test_unknown_section_rejected(tmp_path) -> None:  # noqa: ANN001
    from zero.config import ConfigError

    with pytest.raises(ConfigError, match="Unknown config section"):
        load_post_config({"dpoo": {}}, {"dpo": DPOConfig})
    with pytest.raises(ConfigError, match="did you mean 'beta'"):
        load_post_config({"dpo": {"betta": 0.1}, "data": {"format": "none"}}, {"dpo": DPOConfig})
