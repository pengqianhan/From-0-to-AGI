#!/usr/bin/env bash
# Render the video for Chapter 23. Add --preview to render a 480p sample.
set -euo pipefail
cd "$(dirname "$0")/../../.."
uv run --extra video python -m video_kit.build "chapters/23-linear-attention-hybrid" "$@"
