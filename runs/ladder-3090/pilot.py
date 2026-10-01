"""第 1 阶段试点（只用 GPU0）：在随机 token 数据上测吞吐、显存、单卡并发和衰减分叉的正确性。

    uv run python runs/ladder-3090/pilot.py synth     # 造随机 token 分片（放在 data/ladder3090/，跑完删掉）
    uv run python runs/ladder-3090/pilot.py speed     # A/B/C：各档的 tok/s、显存、单卡并发
    uv run python runs/ladder-3090/pilot.py branch    # D：分叉的衰减 == 从头直接训练（loss 逐步对照）
    uv run python runs/ladder-3090/pilot.py clean     # 删掉随机数据

吞吐与数据内容无关，所以不用等真实数据；随机 token 上 loss 学不下去（停在 ln 65536 ≈ 11.09），只拿来对照。
"""

from __future__ import annotations

import json
import os
import shutil
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data" / "ladder3090"
OUT = REPO / "out" / "ladder3090" / "pilot"
SOURCES = ["fineweb-edu", "dclm-baseline", "fineweb-2-zh", "ultra-fineweb-zh", "ultradata-code", "finemath"]
GPU = "0"  # 试点只用 GPU0


def cmd_synth() -> None:
    if any(DATA.glob("*.bin")) and not (DATA / "SYNTHETIC").exists():
        raise SystemExit(f"{DATA} 里已经有真实数据，不覆盖")
    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / "SYNTHETIC").write_text("试点用的随机 token，不是真实数据；runs/ladder-3090/pilot.py clean 删除\n")
    rng = np.random.default_rng(0)
    for s in SOURCES:
        rng.integers(0, 65536, size=40_000_000, dtype=np.uint32).tofile(DATA / f"{s}_train_000.bin")
    rng.integers(0, 65536, size=1024 * 2048 + 1, dtype=np.uint32).tofile(DATA / "ladder_val.bin")
    print(f"随机分片写到 {DATA}")


def cmd_clean() -> None:
    if (DATA / "SYNTHETIC").exists():
        shutil.rmtree(DATA)
        print(f"删除 {DATA}")


def pretrain(scale: str, out: Path, sets: dict) -> list[str]:
    base = {"train.out_dir": str(out), "data.tokenizer": "", "train.eval_every": 0, "checkpoint.every": 100000}
    cmd = [sys.executable, "-m", "zero.train.pretrain", "--config", str(REPO / "configs/ladder3090" / f"{scale}.toml")]
    for k, v in {**base, **sets}.items():
        cmd += ["--set", f"{k}={v}"]
    return cmd


def gpu_mem_mib() -> int:
    r = subprocess.run(["nvidia-smi", "-i", GPU, "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True)
    return int(r.stdout.strip() or 0)


def run_group(cmds: list[tuple[str, list[str], Path]]) -> tuple[list[float], int, float]:
    """同时启动一组进程（同一张卡），返回各自稳定段的 tok/s、显存峰值（MiB）、墙钟时间。"""
    peak = [gpu_mem_mib()]
    stop = threading.Event()

    def poll() -> None:
        while not stop.is_set():
            peak[0] = max(peak[0], gpu_mem_mib())
            time.sleep(1)

    th = threading.Thread(target=poll, daemon=True)
    th.start()
    t0 = time.time()
    procs = []
    for _, cmd, out in cmds:
        shutil.rmtree(out, ignore_errors=True)
        out.mkdir(parents=True)
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": GPU, "UV_NO_SYNC": "1"}
        procs.append(subprocess.Popen(cmd, cwd=REPO, env=env, stdout=(out / "stdout.log").open("w"),
                                      stderr=subprocess.STDOUT))
    for p in procs:
        p.wait()
    wall = time.time() - t0
    stop.set()
    th.join()
    speeds = []
    for (name, _, out), p in zip(cmds, procs):
        rows = [json.loads(x) for x in (out / "log.jsonl").read_text().splitlines()] if (out / "log.jsonl").exists() else []
        tps = [r["tok_per_s"] for r in rows if "tok_per_s" in r]
        if p.returncode != 0 or not tps:
            print(f"  {name}: 失败（退出码 {p.returncode}），看 {out}/stdout.log")
            speeds.append(float("nan"))
        else:
            speeds.append(statistics.median(tps[len(tps) // 2 :]))  # 后一半：跳过 compile 与预热
    return speeds, peak[0] - 45, wall  # 45 MiB 是空闲时的底数


def cmd_speed() -> None:
    results = []

    def one(label: str, scale: str, micro: int, compile_: bool, k: int = 1, steps: int = 40) -> None:
        tps = 2**17 if scale in ("e5m", "e11m") else 2**16
        sets = {"train.max_steps": steps, "train.micro_batch_size": micro,
                "train.grad_accum_steps": tps // 2048 // micro, "train.compile": str(compile_).lower(),
                "logging.every": 2, "schedule.warmup_steps": 2}
        cmds = [(f"{label}#{i}", pretrain(scale, OUT / "speed" / f"{label}_{i}", sets), OUT / "speed" / f"{label}_{i}")
                for i in range(k)]
        speeds, mem, wall = run_group(cmds)
        row = {"label": label, "scale": scale, "micro": micro, "compile": compile_, "procs": k,
               "tok_s_each": [round(s) if s == s else None for s in speeds],
               "tok_s_total": round(sum(s for s in speeds if s == s)),
               "gpu_mem_mib": mem, "wall_s": round(wall)}
        results.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    only = set(sys.argv[2:])  # 可以只跑某几组：pilot.py speed B C
    if not only or "A" in only:  # A：e5m 的 micro batch × compile（micro 16 在 24GB 上 OOM：FP32 logits 太大）
        for micro in (4, 8):
            for c in (False, True):
                one(f"e5m_m{micro}_{'c' if c else 'e'}", "e5m", micro, c)
    if not only or "B" in only:  # B：e5m 单卡并发（micro 4 + compile 显存最省，约 7.6GB）
        for k in (2, 3):
            one(f"e5m_m4_c_x{k}", "e5m", 4, True, k)
    if not only or "C" in only:  # C：更大的档（micro 4 + compile）
        for scale in ("e11m", "e24m", "e44m", "e83m"):
            one(f"{scale}_m4_c", scale, 4, True, steps=30)
    prev = json.loads((OUT / "speed.json").read_text()) if (OUT / "speed.json").exists() else []
    (OUT / "speed.json").write_text(json.dumps(prev + results, ensure_ascii=False, indent=1))
    print(f"写入 {OUT / 'speed.json'}")


def cmd_branch() -> None:
    """D：主干训到 64 步（每 8 步一个 checkpoint，衰减从第 64 步开始），从第 32 步的 checkpoint 分叉出
    max_steps = 40、从第 32 步开始衰减的分支；对照：从头直接训练同样的 max_steps = 40、decay_frac = 0.2。"""
    sys.path.insert(0, str(REPO / "runs" / "ladder-3090"))
    from sweep import with_decay  # noqa: PLC0415

    common = {"train.micro_batch_size": 4, "train.grad_accum_steps": 4, "train.compile": "false",
              "schedule.warmup_steps": 4, "logging.every": 1, "train.eval_every": 8}
    m, f = with_decay(64)
    trunk = OUT / "branch" / "trunk"
    run_group([("trunk", pretrain("e5m", trunk, {**common, "train.max_steps": m, "schedule.decay_frac": f,
                                                   "train.stop_step": 64, "checkpoint.every": 8}), trunk)])
    mb, fb = with_decay(32)
    br = OUT / "branch" / "branch_4"
    shutil.rmtree(br, ignore_errors=True)
    (br / "ckpt").mkdir(parents=True)
    shutil.copytree(trunk / "ckpt" / "step_00000032", br / "ckpt" / "step_00000032")
    (br / "ckpt" / "latest").write_text("step_00000032")
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": GPU, "UV_NO_SYNC": "1"}
    subprocess.run(pretrain("e5m", br, {**common, "train.max_steps": mb, "schedule.decay_frac": fb,
                                        "checkpoint.every": 100000}),
                   cwd=REPO, env=env, stdout=(br / "stdout.log").open("w"), stderr=subprocess.STDOUT, check=True)
    direct = OUT / "branch" / "direct"
    run_group([("direct", pretrain("e5m", direct, {**common, "train.max_steps": mb, "schedule.decay_frac": fb}),
                direct)])

    def losses(d: Path) -> dict[int, tuple[float, float]]:
        rows = [json.loads(x) for x in (d / "log.jsonl").read_text().splitlines()]
        return {r["step"]: (r["loss"], r["lr"]) for r in rows if "loss" in r}

    a, b = losses(br), losses(direct)
    print(f"分支 max_steps={mb}、decay_frac={fb:.4f}（衰减从第 32 步开始）")
    print(" 步   分支 loss / lr            直接训练 loss / lr")
    worst = 0.0
    for s in sorted(a):
        worst = max(worst, abs(a[s][0] - b[s][0]))
        print(f"{s:3d}   {a[s][0]:.6f} / {a[s][1]:.2e}     {b[s][0]:.6f} / {b[s][1]:.2e}")
    print(f"最大 loss 差 {worst:.2e}（GPU 上 BF16 的非确定性允许 1e-3 量级；学习率必须逐步相同）")


if __name__ == "__main__":
    {"synth": cmd_synth, "speed": cmd_speed, "branch": cmd_branch, "clean": cmd_clean}[sys.argv[1]]()
