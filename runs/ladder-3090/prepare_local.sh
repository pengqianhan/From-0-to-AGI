#!/usr/bin/env bash
# 流水线跑完后：把分片和分词器从 NAS 拷到本地盘（训练时读本地更快，也不受 NAS 波动影响），再拼固定验证集。
set -euo pipefail
cd "$(dirname "$0")/../.."
NAS=/mnt/DataSets/phan635/From-0-to-AGI/ladder/pretrain
mkdir -p data/ladder3090
rsync -a --include='*.bin' --include='*.json' --exclude='*' "$NAS/" data/ladder3090/
UV_NO_SYNC=1 uv run python runs/ladder-3090/build_val.py
du -sh data/ladder3090
