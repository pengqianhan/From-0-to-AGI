"""训练监控的旁路进程：读 out/ladder3090 下各运行的 log.jsonl，不改训练代码、不占 GPU。

    setsid nohup .venv-monitor/bin/python runs/ladder-3090/monitor.py >> out/ladder3090/monitor.log 2>&1 < /dev/null &
    .venv-monitor/bin/python runs/ladder-3090/monitor.py --once --no-upload     # 只算一次、打印状态，不联网

做两件事：
1. 把正在训练的运行的指标写进 Trackio（项目 ladder-3090）。默认只存在本机；设了 MONITOR_TRACKIO_SPACE 才同步到 Hugging Face Space。
2. 每隔几分钟把一份精简的状态（status.json）传到私有的 Hugging Face 数据集，HTML 报告的"实时监控"一节读它。

数据情况：训练日志每 10 步记一次各数据来源累计抽到的序列数（mixture_counts）。相邻两条记录相减，
就是这 10 步的 batch 里各来源各有多少条序列。来源对应的内容类型见 SOURCES。这是按来源分的，
不是对每篇文档做主题分类。

环境：单独的 .venv-monitor（uv venv .venv-monitor && uv pip install --python .venv-monitor/bin/python trackio），
不动训练用的 .venv。停止：kill $(cat out/ladder3090/monitor.pid)。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "out" / "ladder3090"
STATE = OUT / "monitor_state.json"
PROJECT = "ladder-3090"
# Trackio 面板默认只存在本机（服务器上跑 `trackio show --project ladder-3090` 看）。
# 设了 MONITOR_TRACKIO_SPACE=<用户>/<名字> 才同步到 Hugging Face Space：私有的 Gradio Space 需要 PRO 订阅，
# 免费的静态 Space 只能是公开的。
SPACE = os.environ.get("MONITOR_TRACKIO_SPACE") or None
STATUS_REPO = "pengqian/ladder-3090-status"
ACTIVE_S = 15 * 60  # 日志在这段时间内更新过，就算"正在训练"

# 数据来源 → 内容类型。weight 是 configs/ladder3090/base.toml 里的配比
SOURCES = [
    {"key": "fineweb-edu", "label": "英文 · 教育类网页", "kind": "英文", "weight": 0.35},
    {"key": "dclm-baseline", "label": "英文 · 通用网页", "kind": "英文", "weight": 0.10},
    {"key": "fineweb-2-zh", "label": "中文 · 通用网页", "kind": "中文", "weight": 0.20},
    {"key": "ultra-fineweb-zh", "label": "中文 · 高质量网页", "kind": "中文", "weight": 0.10},
    {"key": "ultradata-code", "label": "代码", "kind": "代码", "weight": 0.15},
    {"key": "finemath", "label": "数学（英文）", "kind": "数学", "weight": 0.10},
]
KEYS = [s["key"] for s in SOURCES]


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str) -> None:
    print(f"[{now()}] {msg}", flush=True)


def run_logs() -> list[Path]:
    """所有正式运行的 log.jsonl（主干和 branch_k），不含试点目录。"""
    out = []
    for p in list(OUT.glob("*/*/log.jsonl")) + list(OUT.glob("*/*/branch_*/log.jsonl")):
        if not p.relative_to(OUT).parts[0].startswith("pilot"):
            out.append(p)
    return out


def run_name(p: Path) -> str:
    parts = p.relative_to(OUT).parts[:-1]  # (档, 配置号[, branch_k])
    return "/".join(parts[1:]) if len(parts) > 1 else parts[0]


def read_records(p: Path) -> list[dict]:
    recs: dict[int, dict] = {}
    try:
        for line in p.read_text().splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue  # 正在写的半行
            recs[r["step"]] = r  # 续训会重复同一步：取后一条
    except OSError:
        return []
    return [recs[k] for k in sorted(recs)]


def max_steps(p: Path) -> int | None:
    try:
        head = (p.parent / "stdout.log").read_text()[:4000]
    except OSError:
        return None
    m = re.findall(r"step\s+\d+/(\d+)", head)
    return int(m[-1]) if m else None


def gpu() -> dict | None:
    q = "utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw"
    r = subprocess.run(["nvidia-smi", "-i", "0", f"--query-gpu={q}", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True, check=False)
    try:
        u, mu, mt, t, pw = [float(x) for x in r.stdout.strip().split(",")]
    except ValueError:
        return None
    return {"util": u, "mem_used_mib": mu, "mem_total_mib": mt, "temp_c": t, "power_w": pw}


def scheduler() -> dict | None:
    logs = [p for p in OUT.glob("*.log") if p.name not in ("monitor.log",) and not p.name.startswith("pilot")]
    if not logs:
        return None
    p = max(logs, key=lambda x: x.stat().st_mtime)
    lines = [ln for ln in p.read_text(errors="replace").splitlines() if ln.strip() and "] $ " not in ln]
    return {"name": p.name, "updated": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(p.stat().st_mtime)),
            "lines": [ln.replace(str(REPO) + "/", "")[:200] for ln in lines[-8:]]}


def bucket(points: list[tuple[int, float]], n: int = 240) -> list[list[float]]:
    """把曲线压到最多 n 个点：每一段取平均。"""
    if len(points) <= n:
        return [[s, round(v, 4)] for s, v in points]
    size = len(points) / n
    out = []
    for i in range(n):
        seg = points[int(i * size):int((i + 1) * size)]
        if seg:
            out.append([seg[-1][0], round(sum(v for _, v in seg) / len(seg), 4)])
    return out


def mix_intervals(recs: list[dict]) -> list[dict]:
    """相邻两条记录的 mixture_counts 之差：这段步数里各来源各抽了多少条序列。"""
    out = []
    prev: dict | None = None
    for r in recs:
        c = r.get("mixture_counts")
        if not c:
            continue
        if prev is not None and r["step"] > prev["step"]:
            d = {k: c.get(k, 0) - prev["c"].get(k, 0) for k in KEYS}
            if all(v >= 0 for v in d.values()) and sum(d.values()) > 0:
                out.append({"from": prev["step"], "to": r["step"], "counts": d})
        prev = {"step": r["step"], "c": c}
    return out


def run_status(p: Path) -> dict:
    recs = read_records(p)
    last = recs[-1]
    total = max_steps(p)
    tps = [r["tok_per_s"] for r in recs[-30:] if r.get("tok_per_s")]
    tok_s = sorted(tps)[len(tps) // 2] if tps else None
    tokens_per_step = last["tokens"] / last["step"] if last["step"] else None
    eta = (total - last["step"]) * tokens_per_step / tok_s if total and tok_s and tokens_per_step else None
    iv = mix_intervals(recs)
    cum = last.get("mixture_counts") or {}
    n = sum(cum.get(k, 0) for k in KEYS) or 1
    return {
        "name": run_name(p), "scale": p.relative_to(OUT).parts[0],
        "step": last["step"], "max_steps": total, "tokens": last["tokens"], "tok_per_s": tok_s, "eta_s": eta,
        "loss": last["loss"], "lr": last["lr"], "grad_norm": last["grad_norm"], "max_mem_gb": last.get("max_mem_gb"),
        "elapsed_s": last.get("elapsed_s"),
        "updated": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(p.stat().st_mtime)),
        "loss_curve": bucket([(r["step"], r["loss"]) for r in recs if r["step"] >= 50]),
        "lr_curve": bucket([(r["step"], r["lr"]) for r in recs], 120),
        "gnorm_curve": bucket([(r["step"], r["grad_norm"]) for r in recs if r["step"] >= 50], 120),
        "toks_curve": bucket([(r["step"], r["tok_per_s"]) for r in recs if r["step"] >= 50], 120),
        "val": [[r["step"], round(r["val_bpb"], 4)] for r in recs if "val_bpb" in r],
        "mix_last": iv[-1] if iv else None,
        "mix_recent": [[x["to"], *[x["counts"][k] for k in KEYS]] for x in iv[-60:]],
        "mix_cum": {k: cum.get(k, 0) / n for k in KEYS},
        "seqs_seen": sum(cum.get(k, 0) for k in KEYS),
    }


def finished_runs(logs: list[Path], skip: Path | None, n: int = 10) -> list[dict]:
    out = []
    for p in sorted(logs, key=lambda x: -x.stat().st_mtime):
        if p == skip:
            continue
        recs = read_records(p)
        vals = [r for r in recs if "val_bpb" in r]
        if not vals:
            continue
        out.append({"name": run_name(p), "scale": p.relative_to(OUT).parts[0], "step": recs[-1]["step"],
                    "val_bpb": round(vals[-1]["val_bpb"], 4), "val_step": vals[-1]["step"],
                    "ended": time.strftime("%Y-%m-%d %H:%M", time.localtime(p.stat().st_mtime))})
        if len(out) == n:
            break
    return out


def collect() -> tuple[dict, Path | None]:
    logs = run_logs()
    recent = [p for p in logs if time.time() - p.stat().st_mtime < ACTIVE_S and p.stat().st_size > 0]
    active = max(recent, key=lambda x: x.stat().st_mtime) if recent else None
    status = {
        "updated": now(), "updated_ms": int(time.time() * 1000), "sources": SOURCES,
        "gpu": gpu(), "scheduler": scheduler(),
        "run": run_status(active) if active else None,
        "recent": finished_runs(logs, active),
        "trackio": f"https://huggingface.co/spaces/{SPACE}" if SPACE else None,
    }
    return status, active


class Tracker:
    """把新出现的日志行写进 Trackio。每个运行一个 Trackio run；写到第几步记在 monitor_state.json 里。"""

    def __init__(self) -> None:
        self.state = json.loads(STATE.read_text()) if STATE.exists() else {}
        self.cur: str | None = None

    def push(self, p: Path) -> None:
        import trackio

        name = f"{p.relative_to(OUT).parts[0]}/{run_name(p)}"
        if self.cur != name:
            if self.cur is not None:
                trackio.finish()
            kw = {"space_id": SPACE, "private": True} if SPACE else {}
            trackio.init(project=PROJECT, name=name, resume="allow", auto_log_gpu=False, auto_log_cpu=False, **kw)
            self.cur = name
            log(f"Trackio run：{name}")
        done = self.state.get(name, 0)
        recs = read_records(p)
        prev = None
        n = 0
        for r in recs:
            c = r.get("mixture_counts")
            if r["step"] > done:
                m = {"train/loss": r["loss"], "train/lr": r["lr"], "train/grad_norm": r["grad_norm"],
                     "perf/tok_per_s": r["tok_per_s"], "perf/max_mem_gb": r.get("max_mem_gb"),
                     "train/tokens": r["tokens"]}
                if "val_bpb" in r:
                    m["val/bpb"] = r["val_bpb"]
                if c and prev:
                    d = {k: c.get(k, 0) - prev.get(k, 0) for k in KEYS}
                    tot = sum(d.values())
                    if tot > 0 and min(d.values()) >= 0:
                        for k in KEYS:
                            m[f"data/{k}"] = d[k] / tot
                trackio.log({k: v for k, v in m.items() if v is not None}, step=r["step"])
                n += 1
            if c:
                prev = c
        if recs:
            self.state[name] = recs[-1]["step"]
            STATE.write_text(json.dumps(self.state))
        if n:
            log(f"Trackio：{name} 新写 {n} 条（到第 {recs[-1]['step']} 步）")


def upload(status: dict) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(STATUS_REPO, repo_type="dataset", private=True, exist_ok=True)
    data = json.dumps(status, ensure_ascii=False, separators=(",", ":")).encode()
    api.upload_file(path_or_fileobj=data, path_in_repo="status.json", repo_id=STATUS_REPO, repo_type="dataset",
                    commit_message=f"status {status['updated']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--no-upload", action="store_true", help="不传 status.json，不写 Trackio")
    ap.add_argument("--every", type=float, default=60, help="每隔多少秒读一次日志")
    ap.add_argument("--upload-every", type=float, default=180, help="每隔多少秒传一次 status.json")
    args = ap.parse_args()
    if args.once:
        status, _ = collect()
        if not args.no_upload:
            upload(status)
        s = json.dumps(status, ensure_ascii=False)
        print(s[:3000] + (" …" if len(s) > 3000 else ""))
        return
    (OUT / "monitor.pid").write_text(str(os.getpid()))
    tracker = None if args.no_upload else Tracker()
    last_up = 0.0
    squashed = time.time()
    log("监控开始")
    while True:
        try:
            status, active = collect()
            (OUT / "status.json").write_text(json.dumps(status, ensure_ascii=False))
            if tracker and active:
                try:
                    tracker.push(active)
                except Exception as e:  # noqa: BLE001  网络或 Trackio 出错不应让监控退出
                    log(f"Trackio 出错（下次再试）：{type(e).__name__}: {e}")
            if not args.no_upload and time.time() - last_up >= args.upload_every:
                try:
                    upload(status)
                    last_up = time.time()
                    if time.time() - squashed > 86400:  # 每天把提交历史压成一条
                        from huggingface_hub import HfApi

                        HfApi().super_squash_history(STATUS_REPO, repo_type="dataset")
                        squashed = time.time()
                except Exception as e:  # noqa: BLE001
                    log(f"上传出错（下次再试）：{type(e).__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            log(f"读日志出错：{type(e).__name__}: {e}")
        time.sleep(args.every)


if __name__ == "__main__":
    main()
