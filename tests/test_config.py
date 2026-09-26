"""配置：所有 configs/*.toml 都能读、校验能抓住常见错误、参数量公式与真实模型一致。"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from zero.config import ConfigError, config_from_dict, load_config, load_model_config
from zero.model import Transformer, count_params, estimate_flops_per_token

REPO = Path(__file__).resolve().parent.parent
ALL_CONFIGS = sorted((REPO / "configs").rglob("*.toml"))


@pytest.mark.parametrize("path", ALL_CONFIGS, ids=lambda p: str(p.relative_to(REPO)))
def test_all_configs_load(path: Path) -> None:
    if path.name == "base.toml":
        load_model_config(path)  # 阶梯公共配置只有部分字段
        return
    cfg = load_config(path)
    # 参数量公式 == 真实构建的模型（在 meta 设备上构建，不分配内存）
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
    with pytest.raises(ConfigError, match="你是不是想写 .n_layers."):
        config_from_dict(_minimal(n_layer=3))  # 拼错：提示正确字段
    with pytest.raises(ConfigError, match="整除"):
        config_from_dict(_minimal(n_heads=3, n_kv_heads=2, head_dim=8))
    with pytest.raises(ConfigError, match="整数"):
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
    # 6N 项（N 含 lm_head 的矩阵乘、不含 norm）+ 注意力项
    n_matmul = (
        c["non_embedding"]
        - cfg.n_layers * (2 * cfg.dim + 2 * cfg.head_dim)
        - cfg.dim
        + c["embedding"]
    )
    assert fpt == pytest.approx(6 * n_matmul + 12 * cfg.n_layers * cfg.q_dim * 4096)
