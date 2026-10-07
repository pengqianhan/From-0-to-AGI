"""Entry point for mid-training / annealing / long-context extension (Chapter 15).

Mid-training continues from the pretrained weights. It uses the same `zero/train/trainer.py` as
pretraining. The config gives three types of change:

1. **New data mixture**: `[[data.sources]]` sets a new list of sources and weights
   (more high-quality data, plus data in the instruction and tool-call formats).
2. **Learning-rate decay phase**: `[schedule] kind = "wsd"` with `decay_frac = 1.0` is a pure decay
   phase: a linear decrease from the peak to the minimum. With the WSD stable phase of pretraining,
   this is the last phase of WSD as a separate run.
3. **Long-context extension**: increase `data.seq_len` and `model.max_seq_len`, and change RoPE:
   - increase `rope_theta` (ABF, adjusted base frequency), or
   - add `model.rope_scaling = {type = "yarn", factor = ..., original_max_position_embeddings = ...}`.
   The RoPE cos/sin are not trainable parameters. The model computes them again from the new config,
   and all other weights load unchanged.

You must set `train.init_from` (the pretraining checkpoint directory). The model shape (layers,
dimensions, vocabulary) must be the same as in that checkpoint. Only the RoPE fields and max_seq_len
can be different. After an interruption, run the same command again. The run resumes from its own
checkpoint.

    uv run python -m zero.train.midtrain --config configs/tiny/midtrain.toml
"""

from __future__ import annotations

import json

from zero.config import Config, ConfigError, load_config
from zero.train.checkpoint import find_latest
from zero.train.pretrain import parse_args
from zero.train.trainer import run_training

# Model fields that mid-training can change.
_MUTABLE_MODEL_FIELDS = {"rope_theta", "rope_scaling", "max_seq_len"}


def check_compatible(cfg: Config) -> list[str]:
    """Check that the model structure is compatible with init_from.

    Return a list of descriptions of what changed.
    """
    if not cfg.train.init_from:
        raise ConfigError("Mid-training needs [train] init_from (the pretraining checkpoint directory)")
    ckpt = find_latest(cfg.train.init_from)
    if ckpt is None:
        raise ConfigError(f"No checkpoint found in init_from={cfg.train.init_from}. Run pretraining first")
    with open(ckpt / "meta.json") as f:
        old = json.load(f).get("config", {})
    changes = []
    old_model = old.get("model", {})
    new_model = cfg.to_dict()["model"]
    for k, v in new_model.items():
        if k in old_model and old_model[k] != v:
            if k not in _MUTABLE_MODEL_FIELDS:
                raise ConfigError(
                    f"[model] {k} is different from the pretraining checkpoint ({old_model[k]} → {v}). "
                    f"Mid-training cannot change the model shape"
                )
            changes.append(f"model.{k}: {old_model[k]} → {v}")
    old_data = old.get("train", {}).get("data", {})
    new_data = cfg.to_dict()["train"]["data"]
    if old_data.get("seq_len") != new_data["seq_len"]:
        changes.append(f"data.seq_len: {old_data.get('seq_len')} → {new_data['seq_len']}")
    old_mix = {s["name"]: s["weight"] for s in old_data.get("sources", [])}
    new_mix = {s["name"]: s["weight"] for s in new_data["sources"]}
    if old_mix != new_mix:
        changes.append(f"data mixture: {old_mix} → {new_mix}")
    return changes


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv, description="Mid-training / long-context extension")
    cfg = load_config(args.config, args.set)
    if args.print_config:
        print(json.dumps(cfg.to_dict(), indent=2, ensure_ascii=False))
        return
    for c in check_compatible(cfg):
        print(f"[midtrain] {c}")
    run_training(cfg)


if __name__ == "__main__":
    main()
