#!/usr/bin/env bash
# Stage 6, item 3: resume after a real kill -9. Compare the loss at each step with an uninterrupted
# control run, and with the trajectory of the same run before the kill.
#   flock <gpu0.lock> bash runs/2026-10-01-gpu0-check/resume_check.sh           # 1 GPU (GPU0), default kernels
#   flock <gpu0.lock> env DET=1 bash runs/2026-10-01-gpu0-check/resume_check.sh   # 1 GPU (GPU0), deterministic algorithms
# Environment variables: GPU (CUDA_VISIBLE_DEVICES, default 0), NPROC (processes per machine, default 1),
#   DET=1 (torch.use_deterministic_algorithms), TAG (name of the output directory),
#   STEPS / EVERY / KILL_AT (default 200 / 100 / 150). All other arguments go to the training command unchanged.
# Control: STEPS steps without interruption. Experiment: the same command runs to KILL_AT, then kill -9 stops
# torchrun and all workers (this simulates preemption). Then the same command runs again (it resumes
# automatically from the latest checkpoint). The data order, learning rate, optimizer state, and random
# number state must all come back exactly from the checkpoint.
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

echo "== Control: $STEPS steps without interruption (GPU=$CUDA_VISIBLE_DEVICES, nproc=$NPROC, DET=${DET:-0}, extra arguments: ${EXTRA[*]:-none})"
run "$OUT/ref" > "$OUT.ref.log" 2>&1 || { cat "$OUT.ref.log"; exit 1; }
grep -E "step +(1|$EVERY|$KILL_AT|$STEPS)/" "$OUT.ref.log" || true

echo "== Experiment: kill -9 at step $KILL_AT"
mkdir -p "$OUT/kill"
bash -c "$(declare -p ENTRY EXTRA NPROC STEPS EVERY); $(declare -f run); run $OUT/kill" > "$OUT.kill1.log" 2>&1 &
PID=$!
until grep -q "\"step\": $KILL_AT," "$OUT/kill/log.jsonl" 2>/dev/null; do sleep 0.2; done
# torchrun starts the workers with start_new_session, so a kill of the process group does not reach them.
# The output directory is unique in the command line: use it to kill -9 torchrun and the workers together.
pkill -9 -f "pretrain.*$OUT/kill" || true
wait "$PID" 2>/dev/null || true
while pgrep -f "pretrain.*$OUT/kill" > /dev/null; do sleep 0.2; done
echo "Stopped with kill -9; last step in the log: $(tail -1 "$OUT/kill/log.jsonl" | cut -c1-60)"
ls "$OUT/kill/ckpt"

echo "== Run the same command again (automatic resume)"
run "$OUT/kill" > "$OUT.kill2.log" 2>&1
grep -E "Resumed training|step +$STEPS/" "$OUT.kill2.log" | head -5 || true

uv run python - "$OUT" "$STEPS" "$EVERY" "$KILL_AT" <<'EOF'
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
steps, every, kill_at = map(int, sys.argv[2:5])
recs = [json.loads(line) for line in open(out / "kill/log.jsonl")]
ref = {r["step"]: r for r in map(json.loads, open(out / "ref/log.jsonl"))}
# In log.jsonl of the kill directory, the steps in (resume_from, kill_at] occur two times
# (one time before the kill, one time after the resume)
first, last = {}, {}
for r in recs:
    first.setdefault(r["step"], r)
    last[r["step"]] = r
resume_from = (kill_at - 1) // every * every
seg = range(resume_from + 1, kill_at + 1)
d_before = max(abs(first[s]["loss"] - ref[s]["loss"]) for s in range(1, kill_at + 1))
# Same run: after the resume vs before the kill. Both start from the same checkpoint, so the GPU
# non-determinism between two independent runs has no effect.
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
print(f"Control vs before the kill (two independent runs), steps 1–{kill_at}, max absolute difference: {d_before:.3e}")
print(f"Same run: steps {seg.start}–{kill_at} after the resume vs before the kill, max absolute difference {d_same:.3e}; "
      f"{n_exact}/{len(seg)} consecutive steps bit-identical after the resume")
print(f"After the resume, steps {resume_from + 1}–{steps} vs control, max absolute difference: {d_after:.3e} "
      f"(learning rate identical at each step: {lr_same})")
print(f"step {steps} val_loss: control {ref[steps]['val_loss']:.6f}, resumed {last[steps]['val_loss']:.6f}, difference {d_val:.3e}")
json.dump(res, open(out / "result.json", "w"), indent=1)
EOF
