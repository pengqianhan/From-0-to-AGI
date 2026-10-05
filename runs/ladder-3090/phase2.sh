#!/usr/bin/env bash
# 第 2 阶段：e5m 的 128 组全部训满（只用 GPU0；所有配置训满是为了验证逐轮淘汰的代价），
# 再给最好的 8 组做 3/8–7/8 的衰减分叉，最后汇总成 runs/ladder-3090/results/e5m.csv。
# 脱离会话运行：setsid nohup bash runs/ladder-3090/phase2.sh > out/ladder3090/phase2.log 2>&1 &
set -u
cd "$(dirname "$0")/../.."
export UV_NO_SYNC=1
date -Is
uv run python runs/ladder-3090/sweep.py run --scale e5m --until 1 --gpus 0 --per-gpu 1
uv run python runs/ladder-3090/sweep.py branch --scale e5m --top 8 --ks 3 4 5 6 7 --gpus 0 --per-gpu 1
uv run python runs/ladder-3090/sweep.py collect --scale e5m
date -Is
