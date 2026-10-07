"""Count the parameters of a model (Chapter 12).

    uv run python -m zero.tools.count_params configs/main/pretrain.toml [configs/ladder/*.toml ...]

The tool uses formulas and does not allocate memory. Thus it gives the result immediately,
also for a 0.8B config. `tests/test_config.py` checks the formulas against models that it builds.
"""

from __future__ import annotations

import argparse

from zero.config import load_model_config
from zero.model import count_params, estimate_flops_per_token


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Count the parameters of each config")
    ap.add_argument("configs", nargs="+")
    args = ap.parse_args(argv)
    header = f"{'config':<40}{'total':>12}{'embedding':>12}{'non-embed':>13}{'per layer':>10}{'FLOPs/token@max_seq':>22}"
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
