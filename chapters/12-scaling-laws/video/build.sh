#!/usr/bin/env bash
# 渲染第 12 章视频。加 --preview 渲染 480p 样片。
# 画面里的迷你阶梯数据来自 out/ch12/*.json（由 code/03、04、06 生成；没有缓存时会先训练，单线程约 20 分钟）。
set -euo pipefail
cd "$(dirname "$0")/../../.."
uv run --extra video python -m video_kit.build "chapters/12-scaling-laws" "$@"
