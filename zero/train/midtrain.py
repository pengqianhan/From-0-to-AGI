"""中期训练 / 退火 / 长上下文扩展入口（对应第 15 章）。

中期训练在预训练权重的基础上接着训，和预训练共用 `zero/train/trainer.py`，靠配置表达三类变化：

1. **换数据混合**：`[[data.sources]]` 换一套来源和权重（加大高质量数据、加入指令与工具调用格式数据）；
2. **学习率衰减段**：`[schedule] kind = "wsd"`、`decay_frac = 1.0` 就是"从峰值线性降到最低"
   的纯衰减段——配合预训练的 WSD 稳定段，相当于把 WSD 的最后一段单独拿出来做；
3. **长上下文扩展**：加大 `data.seq_len` 与 `model.max_seq_len`，同时改 RoPE：
   - 调大 `rope_theta`（ABF，adjusted base frequency），或
   - 加 `model.rope_scaling = {type = "yarn", factor = ..., original_max_position_embeddings = ...}`。
   RoPE 的 cos/sin 不是可训练参数，按新配置重新计算即可，其余权重原样加载。

必须设置 `train.init_from`（预训练的 checkpoint 目录）。模型形状（层数、维度、词表）必须与之一致，
只有 RoPE 相关字段和 max_seq_len 可以不同。中断后重跑同一条命令，会从本次运行自己的 checkpoint 续训。

    uv run python -m zero.train.midtrain --config configs/tiny/midtrain.toml
"""

from __future__ import annotations

import json

from zero.config import Config, ConfigError, load_config
from zero.train.checkpoint import find_latest
from zero.train.pretrain import parse_args
from zero.train.trainer import run_training

# 中期训练允许改变的模型字段
_MUTABLE_MODEL_FIELDS = {"rope_theta", "rope_scaling", "max_seq_len"}


def check_compatible(cfg: Config) -> list[str]:
    """检查与 init_from 的模型结构是否兼容，返回"变化了什么"的说明列表。"""
    if not cfg.train.init_from:
        raise ConfigError("中期训练需要设置 [train] init_from（预训练 checkpoint 目录）")
    ckpt = find_latest(cfg.train.init_from)
    if ckpt is None:
        raise ConfigError(f"init_from={cfg.train.init_from} 里找不到 checkpoint，先跑预训练")
    with open(ckpt / "meta.json") as f:
        old = json.load(f).get("config", {})
    changes = []
    old_model = old.get("model", {})
    new_model = cfg.to_dict()["model"]
    for k, v in new_model.items():
        if k in old_model and old_model[k] != v:
            if k not in _MUTABLE_MODEL_FIELDS:
                raise ConfigError(
                    f"[model] {k} 与预训练 checkpoint 不同（{old_model[k]} → {v}），中期训练不能改变模型形状"
                )
            changes.append(f"model.{k}: {old_model[k]} → {v}")
    old_data = old.get("train", {}).get("data", {})
    new_data = cfg.to_dict()["train"]["data"]
    if old_data.get("seq_len") != new_data["seq_len"]:
        changes.append(f"data.seq_len: {old_data.get('seq_len')} → {new_data['seq_len']}")
    old_mix = {s["name"]: s["weight"] for s in old_data.get("sources", [])}
    new_mix = {s["name"]: s["weight"] for s in new_data["sources"]}
    if old_mix != new_mix:
        changes.append(f"数据混合: {old_mix} → {new_mix}")
    return changes


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv, description="中期训练 / 长上下文扩展")
    cfg = load_config(args.config, args.set)
    if args.print_config:
        print(json.dumps(cfg.to_dict(), indent=2, ensure_ascii=False))
        return
    for c in check_compatible(cfg):
        print(f"[midtrain] {c}")
    run_training(cfg)


if __name__ == "__main__":
    main()
