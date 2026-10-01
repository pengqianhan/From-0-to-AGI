#!/usr/bin/env bash
# 第 0 阶段：并行下载 configs/ladder3090/download.toml 里的全部来源（每个来源一个进程，8 个并发）。
# datasets 流式读取后解释器退出会报 PyGILState_Release（退出码 134，数据已写完），所以不看退出码，
# 下载完用 _manifest.json 核对（见 runs/2026-10-01-vocab-corpus 的"已知问题"）。可重复运行：已完成的来源会续传或跳过。
set -u
cd "$(dirname "$0")/../.."
OUT=${OUT:-/mnt/DataSets/phan635/From-0-to-AGI/ladder/raw}
mkdir -p "$OUT/_logs"
grep '^name = ' configs/ladder3090/download.toml | sed 's/name = "\(.*\)"/\1/' |
  xargs -P "${JOBS:-8}" -I{} sh -c \
    "CUDA_VISIBLE_DEVICES= UV_NO_SYNC=1 uv run python -m zero.data.download --config configs/ladder3090/download.toml \
       --sources {} --out '$OUT' > '$OUT/_logs/{}.log' 2>&1; true"
echo "全部来源跑完：$(date -Is)"
