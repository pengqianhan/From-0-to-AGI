"""Config: all configs/*.toml can be read, validation catches common errors, and the parameter formula agrees with the real model."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from zero.config import ConfigError, config_from_dict, load_config, load_model_config
from zero.model import Transformer, count_params, estimate_flops_per_token

REPO = Path(__file__).resolve().parent.parent
ALL_CONFIGS = sorted((REPO / "configs").rglob("*.toml"))


def _sections(stage: str):  # noqa: ANN202
    from zero.post.distill import DistillConfig, TeacherConfig
    from zero.post.dpo import DPOConfig
    from zero.post.grpo import GRPOConfig
    from zero.post.opd import OPDConfig
    from zero.post.sft import SFTConfig

    return {
        "sft": {"sft": SFTConfig},
        "distill": {"teacher": TeacherConfig, "distill": DistillConfig},
        "dpo": {"dpo": DPOConfig},
        "grpo": {"grpo": GRPOConfig},
        "opd": {"opd": OPDConfig},
    }[stage]


POST_SECTIONS = {s: (lambda s=s: _sections(s)) for s in ("sft", "distill", "dpo", "grpo", "opd")}


@pytest.mark.parametrize("path", ALL_CONFIGS, ids=lambda p: str(p.relative_to(REPO)))
def test_all_configs_load(path: Path) -> None:
    if path.name == "base.toml":
        load_model_config(path)  # the shared ladder config has only some of the fields
        return
    if path.stem == "data":  # data pipeline config (Chapter 13, zero/data/pipeline.py; tests/test_pipeline.py tests it)
        from zero.data.pipeline import load_pipeline_config

        load_pipeline_config(path)
        return
    if path.stem == "download":  # a config only for the downloader (such as configs/vocab/download.toml, the vocabulary measurement of Chapter 13)
        import tomllib

        from zero.data.download import check_license, specs_from_config

        specs = specs_from_config(tomllib.loads(path.read_text("utf-8")))
        assert specs
        for s in specs:  # each source must be registered in zero/data/sources.py, with a verified license
            check_license(s)
        return
    if path.stem == "sft_data":  # the SFT data mixture (zero/post/sft_data.py; tests/test_sft_data.py tests it)
        from zero.post.sft_data import load_sft_data_config

        load_sft_data_config(path)
        return
    if path.stem == "eval":  # an evaluation config has only [eval] (tests/test_post_configs.py tests it)
        from zero.eval.harness import load_eval_config

        load_eval_config(path)
        return
    if path.stem in POST_SECTIONS:  # post-training configs add a section for each stage
        from zero.post.common import load_post_config

        cfg, _ = load_post_config(path, POST_SECTIONS[path.stem]())
    else:
        cfg = load_config(path)
    # parameter formula == the model that we really build (built on the meta device, so no memory is allocated)
    with torch.device("meta"):
        model = Transformer(cfg.model)
    assert model.num_params() == count_params(cfg.model)["total"]
    assert model.num_params(non_embedding=True) == count_params(cfg.model)["non_embedding"]


def test_main_model_within_budget() -> None:
    cfg = load_config(REPO / "configs/main/pretrain.toml")
    total = count_params(cfg.model)["total"]
    assert 0.6e9 <= total <= 0.8e9, total


def test_midtrain_inherits_and_overrides() -> None:
    cfg = load_config(REPO / "configs/tiny/midtrain.toml")
    base = load_config(REPO / "configs/tiny/pretrain.toml")
    assert cfg.model.dim == base.model.dim
    assert cfg.model.rope_scaling["type"] == "yarn"
    assert cfg.train.data.seq_len == 2 * base.train.data.seq_len
    assert cfg.train.init_from


def test_overrides() -> None:
    cfg = load_config(
        REPO / "configs/tiny/pretrain.toml", ["train.max_steps=7", "model.rope_theta=1e6"]
    )
    assert cfg.train.max_steps == 7 and cfg.model.rope_theta == 1e6


def _minimal(**model) -> dict:
    return {
        "model": {
            "vocab_size": 10,
            "dim": 16,
            "n_layers": 1,
            "n_heads": 2,
            "n_kv_heads": 1,
            "ffn_dim": 16,
            **model,
        },
        "data": {"seq_len": 8, "sources": [{"name": "a", "path": "x_*.bin"}]},
    }


def test_validation_errors() -> None:
    config_from_dict(_minimal())
    with pytest.raises(ConfigError, match="did you mean .n_layers."):
        config_from_dict(_minimal(n_layer=3))  # wrong spelling: the hint gives the correct field
    with pytest.raises(ConfigError, match="divisible"):
        config_from_dict(_minimal(n_heads=3, n_kv_heads=2, head_dim=8))
    with pytest.raises(ConfigError, match="integer"):
        config_from_dict(_minimal(dim="16"))
    with pytest.raises(ConfigError, match="yarn"):
        config_from_dict(_minimal(rope_scaling={"type": "linear", "factor": 2.0}))
    bad = _minimal()
    bad["optimm"] = {}
    with pytest.raises(ConfigError, match="optim"):
        config_from_dict(bad)
    bad = _minimal()
    bad["data"]["seq_len"] = 4096
    with pytest.raises(ConfigError, match="max_seq_len"):
        config_from_dict(bad)


def test_flops_formula() -> None:
    cfg = load_model_config(REPO / "configs/main/pretrain.toml")
    c = count_params(cfg)
    fpt = estimate_flops_per_token(cfg, 4096)
    # 6N term (N includes the lm_head matmul, not the norms) + attention term
    n_matmul = (
        c["non_embedding"]
        - cfg.n_layers * (2 * cfg.dim + 2 * cfg.head_dim)
        - cfg.dim
        + c["embedding"]
    )
    assert fpt == pytest.approx(6 * n_matmul + 12 * cfg.n_layers * cfg.q_dim * 4096)
