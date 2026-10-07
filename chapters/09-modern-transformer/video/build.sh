#!/usr/bin/env bash
# Render the video of Chapter 9. Add --preview to render a 480p sample.
set -euo pipefail
cd "$(dirname "$0")/../../.."
uv run --extra video python -m video_kit.build "chapters/09-modern-transformer" "$@"
