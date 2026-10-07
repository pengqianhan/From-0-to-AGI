"""Entry point for pretraining (Chapter 14).

One process (CPU smoke test):

    uv run python -m zero.train.pretrain --config configs/tiny/pretrain.toml

One machine with many GPUs (step 2):

    uv run torchrun --standalone --nproc_per_node=8 -m zero.train.pretrain --config configs/main/pretrain.toml

`--set section.key=value` overrides a config value for one run.
For example: `--set train.max_steps=20 --set data.seq_len=64`.
After an interruption, run the same command again. The run resumes automatically from the latest
checkpoint in `<out_dir>/ckpt`.
"""

from __future__ import annotations

import argparse
import json

from zero.config import load_config
from zero.train.trainer import run_training


def parse_args(argv: list[str] | None = None, description: str = "Pretraining") -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--config", required=True, help="TOML config file")
    ap.add_argument(
        "--set", action="append", default=[], metavar="KEY=VALUE", help="override a config value (you can use it more than once)"
    )
    ap.add_argument("--print-config", action="store_true", help="print the merged config only, do not train")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    cfg = load_config(args.config, args.set)
    if args.print_config:
        print(json.dumps(cfg.to_dict(), indent=2, ensure_ascii=False))
        return
    run_training(cfg)


if __name__ == "__main__":
    main()
