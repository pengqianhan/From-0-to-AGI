#!/usr/bin/env bash
# 阶段 6 第 5 项（单卡版）：torch.compile 开 / 关各跑 60 步（l60m 形状 + tiny 数据），比较 loss 与吞吐。
#   flock <gpu0.lock> bash runs/2026-10-01-gpu0-check/compile_check.sh        # 默认内核：关 ×2、开 ×1
#   flock <gpu0.lock> env PART=det bash runs/2026-10-01-gpu0-check/compile_check.sh   # 确定性算法：关、开各 1 次（再汇总）
# GPU 默认内核有非确定性（见第 3 项），所以另跑两次"关 compile"看同一配置两次运行本身差多少，
# 再用 DET=1（确定性算法）各跑一次开 / 关，把"编译带来的数值差"和"运行间噪声"分开。
set -euo pipefail
cd "$(dirname "$0")/../.."
common=(--set train.max_steps=60 --set schedule.warmup_steps=10 --set train.eval_every=0
        --set checkpoint.every=0 --set logging.every=1)
cfg=runs/2026-10-01-gpu0-check/configs/l60m_tiny.toml
T=runs/2026-10-01-gpu0-check/train_run.sh
if [ "${PART:-}" != det ]; then
  bash $T compile_off_a "$cfg" "${common[@]}" --set train.compile=false
  bash $T compile_off_b "$cfg" "${common[@]}" --set train.compile=false
  bash $T compile_on "$cfg" "${common[@]}" --set train.compile=true
  exit 0
fi
DET=1 bash $T det_compile_off "$cfg" "${common[@]}" --set train.compile=false || true
DET=1 bash $T det_compile_on "$cfg" "${common[@]}" --set train.compile=true || true
UV_NO_SYNC=1 uv run python - <<'PY'
import json
from pathlib import Path
o = Path("out/gpu0-check")
def load(n):
    p = o / n / "log.jsonl"
    return {r["step"]: r for r in map(json.loads, open(p))} if p.exists() else None
runs = {n: load(n) for n in ("compile_off_a", "compile_off_b", "compile_on", "det_compile_off", "det_compile_on")}
def diff(a, b):
    if runs[a] is None or runs[b] is None:
        return None
    return max(abs(runs[a][s]["loss"] - runs[b][s]["loss"]) for s in range(1, 61))
def tps(n):
    r = runs[n]
    return None if r is None else sum(r[s]["tok_per_s"] for s in range(21, 61)) / 40
res = {
    "eager_a_vs_eager_b": diff("compile_off_a", "compile_off_b"),
    "compile_vs_eager_a": diff("compile_on", "compile_off_a"),
    "det_compile_vs_det_eager": diff("det_compile_on", "det_compile_off"),
    "final_loss": {n: (r[60]["loss"] if r else None) for n, r in runs.items()},
    "tok_per_s_steps21_60": {n: tps(n) for n in runs},
    "mfu_steps21_60": {n: (sum(r[s]["mfu"] for s in range(21, 61)) / 40 if r else None) for n, r in runs.items()},
    "step1_s": {n: (r[1]["elapsed_s"] if r else None) for n, r in runs.items()},
}
print(json.dumps(res, indent=1, ensure_ascii=False))
(o / "compile_check.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
PY
