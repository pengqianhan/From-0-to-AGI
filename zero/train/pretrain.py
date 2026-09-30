"""预训练入口（对应第 14 章）。

单进程（CPU 冒烟）：

    uv run python -m zero.train.pretrain --config configs/tiny/pretrain.toml

单机多卡（第二步）：

    uv run torchrun --standalone --nproc_per_node=8 -m zero.train.pretrain --config configs/main/pretrain.toml

`--set section.key=value` 可以临时覆盖配置，例如 `--set train.max_steps=20 --set data.seq_len=64`。
中断后用同一条命令重跑，会自动从 `<out_dir>/ckpt` 里最新的 checkpoint 续训。
"""

from __future__ import annotations

import argparse
import json

from zero.config import load_config
from zero.train.trainer import run_training


def parse_args(argv: list[str] | None = None, description: str = "预训练") -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--config", required=True, help="TOML 配置文件")
    ap.add_argument(
        "--set", action="append", default=[], metavar="KEY=VALUE", help="覆盖配置项，可重复"
    )
    ap.add_argument("--print-config", action="store_true", help="只打印合并后的配置，不训练")
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
