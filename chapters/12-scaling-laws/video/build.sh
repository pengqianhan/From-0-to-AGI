#!/usr/bin/env bash
# Render the video of Chapter 12. Add --preview to render a 480p sample.
# The mini-ladder data in the frames comes from out/ch12/*.json (made by code/03, 04, 06).
# If there is no cache, the build trains first (about 20 min on one thread).
set -euo pipefail
cd "$(dirname "$0")/../../.."
uv run --extra video python -m video_kit.build "chapters/12-scaling-laws" "$@"
