#!/usr/bin/env bash
# Run one pretraining with torchrun. At the same time, record the memory use of the GPU every 0.5 s
# (nvidia-smi; this includes the CUDA context and the caching allocator).
#   flock <gpu0.lock> bash runs/2026-10-01-gpu0-check/train_run.sh <name> <config> [--set k=v ...]
# Output: out/gpu0-check/<name>/ (log.jsonl, checkpoint), out/gpu0-check/<name>.log, <name>.gpumem
# With the environment variable DET=1, the script uses det_pretrain.py (torch.use_deterministic_algorithms)
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
echo "[$name] physical GPU $GPU, nproc=$NPROC, exit code $rc, wall-clock time $((end - start)) s, nvidia-smi peak GPU memory: $peak"
grep -E "Model parameters|step +[0-9]+/|checkpoint saved|Resumed training|Error|error" "$out.log" | tail -30
exit $rc
