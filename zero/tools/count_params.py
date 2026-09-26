"""统计模型参数量（对应第 12 章）。

    uv run python -m zero.tools.count_params configs/main/pretrain.toml [configs/ladder/*.toml ...]

按公式计算，不分配内存，0.8B 的配置也是瞬间出结果；`tests/test_config.py` 会用真实构建的模型核对公式。
"""

from __future__ import annotations

import argparse

from zero.config import load_model_config
from zero.model import count_params, estimate_flops_per_token


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="统计配置对应的参数量")
    ap.add_argument("configs", nargs="+")
    args = ap.parse_args(argv)
    header = f"{'配置':<40}{'总参数':>12}{'embedding':>12}{'非embedding':>13}{'每层':>10}{'FLOPs/token@max_seq':>22}"
    print(header)
    for path in args.configs:
        cfg = load_model_config(path)
        c = count_params(cfg)
        fpt = estimate_flops_per_token(cfg, cfg.max_seq_len)
        print(
            f"{path:<40}{c['total'] / 1e6:>11.2f}M{c['embedding'] / 1e6:>11.2f}M"
            f"{c['non_embedding'] / 1e6:>12.2f}M{c['per_layer'] / 1e6:>9.2f}M{fpt:>22.4g}"
        )


if __name__ == "__main__":
    main()
