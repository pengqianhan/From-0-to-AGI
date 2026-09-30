#!/usr/bin/env bash
# 阶段 6 第 3 项：真实 kill -9 之后续训，与不中断的对照运行、以及同一次运行被杀前的轨迹逐步比较 loss。
#   flock <gpu0.lock> bash runs/2026-10-01-gpu0-check/resume_check.sh           # 单卡 GPU0，默认内核
#   flock <gpu0.lock> env DET=1 bash runs/2026-10-01-gpu0-check/resume_check.sh   # 单卡 GPU0，确定性算法
# 环境变量：GPU（CUDA_VISIBLE_DEVICES，默认 0）、NPROC（每机进程数，默认 1）、DET=1（torch.use_deterministic_algorithms）、
#   TAG（输出目录名）、STEPS / EVERY / KILL_AT（默认 200 / 100 / 150）。其余参数原样传给训练命令。
# 对照：STEPS 步不中断；实验：同一条命令跑到 KILL_AT 时 kill -9 torchrun 与全部 worker（模拟抢占），再执行同一条命令
# （自动从最近的 checkpoint 续训）。数据顺序、学习率、优化器与随机数状态都应从 checkpoint 精确恢复。
set -euo pipefail
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES=${GPU:-0} UV_NO_SYNC=1
NPROC=${NPROC:-1}
STEPS=${STEPS:-200}
EVERY=${EVERY:-100}
KILL_AT=${KILL_AT:-150}
OUT=out/gpu0-check/${TAG:-resume${DET:+_det}}
ENTRY=(-m zero.train.pretrain)
[ -n "${DET:-}" ] && ENTRY=(runs/2026-10-01-gpu0-check/det_pretrain.py)
EXTRA=("$@")
rm -rf "$OUT"
mkdir -p "$(dirname "$OUT")"
run() {
  uv run torchrun --standalone --nproc_per_node="$NPROC" "${ENTRY[@]}" \
    --config runs/2026-10-01-gpu0-check/configs/l20m_tiny.toml \
    --set train.max_steps="$STEPS" --set checkpoint.every="$EVERY" --set checkpoint.keep_last=0 \
    --set schedule.warmup_steps=20 --set train.eval_every="$EVERY" --set logging.every=1 \
    --set train.out_dir="\"$1\"" "${EXTRA[@]}"
}

echo "== 对照：不中断 $STEPS 步（GPU=$CUDA_VISIBLE_DEVICES，nproc=$NPROC，DET=${DET:-0}，额外参数：${EXTRA[*]:-无}）"
run "$OUT/ref" > "$OUT.ref.log" 2>&1 || { cat "$OUT.ref.log"; exit 1; }
grep -E "step +(1|$EVERY|$KILL_AT|$STEPS)/" "$OUT.ref.log" || true

echo "== 实验：跑到 step $KILL_AT 时 kill -9"
mkdir -p "$OUT/kill"
bash -c "$(declare -p ENTRY EXTRA NPROC STEPS EVERY); $(declare -f run); run $OUT/kill" > "$OUT.kill1.log" 2>&1 &
PID=$!
until grep -q "\"step\": $KILL_AT," "$OUT/kill/log.jsonl" 2>/dev/null; do sleep 0.2; done
# torchrun 用 start_new_session 启动 worker，杀进程组杀不到它；按命令行里唯一的输出目录把 torchrun 和 worker 一起 kill -9
pkill -9 -f "pretrain.*$OUT/kill" || true
wait "$PID" 2>/dev/null || true
while pgrep -f "pretrain.*$OUT/kill" > /dev/null; do sleep 0.2; done
echo "已 kill -9；日志最后一步：$(tail -1 "$OUT/kill/log.jsonl" | cut -c1-60)"
ls "$OUT/kill/ckpt"

echo "== 同一条命令重跑（自动续训）"
run "$OUT/kill" > "$OUT.kill2.log" 2>&1
grep -E "续训|step +$STEPS/" "$OUT.kill2.log" | head -5 || true

uv run python - "$OUT" "$STEPS" "$EVERY" "$KILL_AT" <<'EOF'
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
steps, every, kill_at = map(int, sys.argv[2:5])
recs = [json.loads(line) for line in open(out / "kill/log.jsonl")]
ref = {r["step"]: r for r in map(json.loads, open(out / "ref/log.jsonl"))}
# kill 目录的 log.jsonl 里 (resume_from, kill_at] 这些步出现两次（被杀前一次、续训后一次）
first, last = {}, {}
for r in recs:
    first.setdefault(r["step"], r)
    last[r["step"]] = r
resume_from = (kill_at - 1) // every * every
seg = range(resume_from + 1, kill_at + 1)
d_before = max(abs(first[s]["loss"] - ref[s]["loss"]) for s in range(1, kill_at + 1))
# 同一次运行：续训后 vs 被杀前（起点是同一个 checkpoint，排除了两次独立运行之间的 GPU 非确定性）
d_same = max(abs(first[s]["loss"] - last[s]["loss"]) for s in seg)
n_exact = next((s for s in seg if first[s]["loss"] != last[s]["loss"]), kill_at + 1) - (resume_from + 1)
d_after = max(abs(last[s]["loss"] - ref[s]["loss"]) for s in range(resume_from + 1, steps + 1))
lr_same = all(abs(last[s]["lr"] - ref[s]["lr"]) < 1e-12 for s in range(resume_from + 1, steps + 1))
d_val = abs(last[steps]["val_loss"] - ref[steps]["val_loss"])
res = {
    "resume_from": resume_from,
    "ref_vs_prekill_max_abs_diff_1_to_kill": d_before,
    "same_run_resumed_vs_prekill_max_abs_diff": d_same,
    "bitwise_identical_steps_after_resume": n_exact,
    "segment_len": len(seg),
    "resumed_vs_ref_max_abs_diff": d_after,
    "lr_identical": lr_same,
    "val_loss_ref_vs_resumed": [ref[steps]["val_loss"], last[steps]["val_loss"], d_val],
}
print(f"对照 vs 被杀前（两次独立运行）1–{kill_at} 步最大绝对差：{d_before:.3e}")
print(f"同一次运行：续训后 {seg.start}–{kill_at} 步与被杀前的最大绝对差 {d_same:.3e}；续训后连续 {n_exact}/{len(seg)} 步逐位相同")
print(f"续训后 {resume_from + 1}–{steps} 步与对照的最大绝对差：{d_after:.3e}（学习率逐步相同：{lr_same}）")
print(f"step {steps} val_loss：对照 {ref[steps]['val_loss']:.6f}，续训 {last[steps]['val_loss']:.6f}，差 {d_val:.3e}")
json.dump(res, open(out / "result.json", "w"), indent=1)
EOF
