#!/usr/bin/env bash
# 渲染第 19 章视频。加 --preview 渲染 480p 样片。
set -euo pipefail
cd "$(dirname "$0")/../../.."
uv run --extra video python -m video_kit.build "chapters/19-reinforcement-learning" "$@"
