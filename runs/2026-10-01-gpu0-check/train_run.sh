#!/usr/bin/env bash
# 用 torchrun 跑一次预训练，同时每 0.5 秒记录一次所用 GPU 的显存占用（nvidia-smi，含 CUDA 上下文与缓存分配器）。
#   flock <gpu0.lock> bash runs/2026-10-01-gpu0-check/train_run.sh <名字> <配置> [--set k=v ...]
# 输出：out/gpu0-check/<名字>/（log.jsonl、checkpoint）、out/gpu0-check/<名字>.log、<名字>.gpumem
# 环境变量 DET=1 时改用 det_pretrain.py（torch.use_deterministic_algorithms）
set -euo pipefail
cd "$(dirname "$0")/../.."
GPU=${GPU:-0}
NPROC=${NPROC:-1}
export CUDA_VISIBLE_DEVICES=$GPU UV_NO_SYNC=1
name=$1
cfg=$2
shift 2
out=out/gpu0-check/$name
rm -rf "$out" "$out.log" "$out.gpumem"
nvidia-smi -i "$GPU" --query-gpu=index,memory.used --format=csv,noheader,nounits -lms 500 > "$out.gpumem" &
smi=$!
start=$(date +%s)
set +e
entry=(-m zero.train.pretrain)
[ -n "${DET:-}" ] && entry=(runs/2026-10-01-gpu0-check/det_pretrain.py)
uv run torchrun --standalone --nproc_per_node="$NPROC" "${entry[@]}" --config "$cfg" \
  --set train.out_dir="\"$out\"" "$@" > "$out.log" 2>&1
rc=$?
set -e
kill "$smi" 2>/dev/null || true
end=$(date +%s)
peak=$(awk -F', ' '{if ($2 > p[$1]) p[$1] = $2} END {for (i in p) printf "GPU%s %s MiB  ", i, p[i]}' "$out.gpumem")
echo "[$name] 物理 GPU $GPU，nproc=$NPROC，退出码 $rc，墙钟 $((end - start)) s，nvidia-smi 显存峰值：$peak"
grep -E "Model parameters|step +[0-9]+/|checkpoint saved|Resumed training|Error|error" "$out.log" | tail -30
exit $rc
