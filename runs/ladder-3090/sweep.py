"""阶梯实验的搜索工具（runs/ladder-3090/README.md 第 4–5 节）。

    # 1. 采样配置（写进 runs/ladder-3090/configs/<档>.jsonl，可重复执行：只追加到 n 组）
    uv run python runs/ladder-3090/sweep.py sample --scale e5m --n 128
    # 2. 训练：--until 是稳定段的进度（2/8 = 第 2 个 checkpoint；1 = 训完整条主干，含 8/8 的衰减）
    uv run python runs/ladder-3090/sweep.py run --scale e5m --until 1 --gpus 0 --per-gpu 4
    #    逐轮淘汰：先全部训到 2/8，再按 2/8 处的 val_bpb 留前一半训到 4/8，再留前四分之一训完
    uv run python runs/ladder-3090/sweep.py run --scale e11m --until 2/8 --gpus 0 --per-gpu 3
    uv run python runs/ladder-3090/sweep.py run --scale e11m --until 4/8 --top 0.5 --rank-at 2/8 --gpus 0
    uv run python runs/ladder-3090/sweep.py run --scale e11m --until 1 --top 0.25 --rank-at 4/8 --gpus 0
    # 3. 衰减分叉：只给最好的几组做（从第 k/8 个 checkpoint 接一段多 25% 步数的线性衰减）
    uv run python runs/ladder-3090/sweep.py branch --scale e5m --top 8 --ks 3 4 5 6 7 --gpus 0 --per-gpu 4
    # 4. 汇总成表（runs/ladder-3090/results/<档>.csv）；status 只打印进度
    uv run python runs/ladder-3090/sweep.py collect --scale e5m

约定（与论文 arXiv 2608.11859 一致）：
- 每个配置的总 token 预算 D = 32 × N_eff；稳定段 S 步（8 的倍数），在 S/8、2S/8……S 处存 checkpoint 并评估；
- 主干的 max_steps = S + round(S/4)，decay_frac 让衰减恰好从第 S 步开始（= 8/8 的衰减分支）；
  第 k/8 的分支从第 c = kS/8 步的 checkpoint 接出，max_steps = c + round(c/4)，衰减从 c 开始；
- 评估用固定验证集（build_val.py），指标 val_bpb。所有运行共用同一份数据和同一个数据顺序（train.seed 固定）。
- 搜索空间（7 个超参数）同论文图 16，batch 换成 token 数 2^15–2^21（序列 2048 时 16–1024 条）。
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import json
import math
import os
import random
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

HERE = REPO / "runs" / "ladder-3090"
OUT = REPO / "out" / "ladder3090"
SEQ = 2048
TOKENS_PER_PARAM = 32
# 每档的 micro batch（每次前向的序列数）：65,536 词表的 FP32 logits 是 micro × 2048 × 65536 × 4 字节，
# 小模型的显存几乎全花在这里。试点阶段按实测调整
MICRO = {"e5m": 4, "e11m": 4, "e24m": 4, "e44m": 4, "e83m": 4, "e166m": 2}


def n_eff(scale: str) -> float:
    from zero.config import load_config
    from zero.model import estimate_flops_per_token

    cfg = load_config(REPO / "configs" / "ladder3090" / f"{scale}.toml")
    return estimate_flops_per_token(cfg.model, SEQ) / 6


# ---------------------------------------------------------------- 采样


def log_uniform(rng: random.Random, lo: float, hi: float) -> float:
    return math.exp(rng.uniform(math.log(lo), math.log(hi)))


def logit_uniform(rng: random.Random, lo: float, hi: float) -> float:
    logit = lambda p: math.log(p / (1 - p))  # noqa: E731
    z = rng.uniform(logit(lo), logit(hi))
    return 1 / (1 + math.exp(-z))


def sample_full(rng: random.Random) -> dict:
    """论文图 16 的搜索分布（batch 按 token 数）。"""
    return {
        "tokens_per_step": 2 ** rng.randint(15, 21),
        "lr": log_uniform(rng, 1e-5, 1e-1),
        "beta1": logit_uniform(rng, 0.7, 0.999),
        "beta2": logit_uniform(rng, 0.9, 0.9999),
        "warmup_frac": log_uniform(rng, 1e-3, 1 / 8),
        "weight_decay": log_uniform(rng, 1e-4, 1.0),
        "rope_theta": log_uniform(rng, 2.0**10, 2.0**20),
    }


# 收窄的搜索空间（e24m 起一半配置从这里抽）：在"变换后的尺度"上（batch 取 log2、正数取 ln、
# β 取 logit）描述每个超参数的取值区间。full 空间的区间即论文图 16
FULL_Z = {
    "tokens_per_step": (15.0, 21.0),
    "lr": (math.log(1e-5), math.log(1e-1)),
    "beta1": (math.log(0.7 / 0.3), math.log(0.999 / 0.001)),
    "beta2": (math.log(0.9 / 0.1), math.log(0.9999 / 0.0001)),
    "warmup_frac": (math.log(1e-3), math.log(1 / 8)),
    "weight_decay": (math.log(1e-4), math.log(1.0)),
    "rope_theta": (math.log(2.0**10), math.log(2.0**20)),
}
# 收窄后每侧至少留这么宽（变换尺度上）：学习率、warmup、weight decay、RoPE θ 至少 ×/÷ 4，batch 至少 ×/÷ 2
MIN_HALF = {"tokens_per_step": 1.0, "lr": math.log(4), "beta1": 1.0, "beta2": 1.0,
            "warmup_frac": math.log(4), "weight_decay": math.log(4), "rope_theta": math.log(4)}
# 最优值会随模型规模系统性移动的超参数：对 log N_eff 做线性回归、外推到目标档
TRENDED = ("lr", "tokens_per_step")


def to_z(h: str, v: float) -> float:
    if h == "tokens_per_step":
        return math.log2(v)
    if h in ("beta1", "beta2"):
        return math.log(v / (1 - v))
    return math.log(v)


def from_z(h: str, z: float) -> float:
    if h == "tokens_per_step":
        return float(2 ** int(round(z)))
    if h in ("beta1", "beta2"):
        return 1 / (1 + math.exp(-z))
    return math.exp(z)


def narrow_ranges(target: str, sources: list[str]) -> dict[str, list[float]]:
    """按小档的结果给目标档收窄搜索区间：每个小档取最终 val_bpb 最好的 10%（至少 5 组）。
    TRENDED 里的超参数对 log N_eff 做最小二乘、外推到目标档，半宽取残差标准差的 2 倍；
    其余取均值 ± 2 倍标准差。每侧至少 MIN_HALF，并截在完整空间之内。"""
    import numpy as np

    xs, tops = [], []
    for sc in sources:
        rows = list(csv.DictReader((HERE / "results" / f"{sc}.csv").open()))
        final = {r["id"]: float(r["val_bpb"]) for r in rows if r["phase"] == "decay" and r["k"] == "8"}
        cfgs = {c["id"]: c for c in load_configs(sc)}
        k = max(5, round(0.1 * len(final)))
        for i in sorted(final, key=final.get)[:k]:
            tops.append(cfgs[i])
            xs.append(math.log(n_eff(sc)))
    x = np.array(xs)
    target_x = math.log(n_eff(target))
    out = {}
    for h, (lo_full, hi_full) in FULL_Z.items():
        z = np.array([to_z(h, float(c[h])) for c in tops])
        if h in TRENDED and len(set(xs)) >= 2:
            slope, icept = np.polyfit(x, z, 1)
            center = icept + slope * target_x
            spread = float(np.std(z - (icept + slope * x)))
        else:
            center, spread = float(z.mean()), float(z.std())
        half = max(2 * spread, MIN_HALF[h])
        lo, hi = max(lo_full, center - half), min(hi_full, center + half)
        # 中心贴着完整空间的边界、被截掉一截时，向里补足最小宽度（只在真的被截断时做，避免浮点误差误触发）
        if hi - lo < 2 * MIN_HALF[h] - 1e-9:
            if center - half < lo_full:
                lo, hi = lo_full, min(hi_full, lo_full + 2 * MIN_HALF[h])
            elif center + half > hi_full:
                lo, hi = max(lo_full, hi_full - 2 * MIN_HALF[h]), hi_full
        out[h] = [float(lo), float(hi)]
    return out


def sample_in(rng: random.Random, ranges: dict[str, list[float]]) -> dict:
    hp = {}
    for h, (lo, hi) in ranges.items():
        if h == "tokens_per_step":
            hp[h] = 2 ** rng.randint(math.ceil(lo - 1e-9), math.floor(hi + 1e-9))
        else:
            hp[h] = from_z(h, rng.uniform(lo, hi))
    return hp


def configs_path(scale: str) -> Path:
    return HERE / "configs" / f"{scale}.jsonl"


def load_configs(scale: str) -> list[dict]:
    p = configs_path(scale)
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()] if p.exists() else []


def cmd_sample(args: argparse.Namespace) -> None:
    existing = load_configs(args.scale)
    path = configs_path(args.scale)
    path.parent.mkdir(parents=True, exist_ok=True)
    ranges = None
    if args.space in ("narrow", "ext"):
        if not args.sources:
            raise SystemExit(f"--space {args.space} 需要 --from <档…>")
        ranges = narrow_ranges(args.scale, args.sources)
        if args.space == "ext":
            # 边界扩展：这一档最好的配置用的是搜索空间里最小的 batch（2^15），最优可能在边界外，
            # 于是 batch 往下扩两档（2^13–2^15），其余超参数在这一档自己最好的配置附近收窄
            ranges["tokens_per_step"] = [13.0, 15.0]
        rpath = HERE / "results" / f"{args.scale}_{args.space}_ranges.json"
        if len(existing) < args.n:
            rpath.parent.mkdir(parents=True, exist_ok=True)
            rpath.write_text(json.dumps({"from": args.sources, "ranges_z": ranges, "ranges": {
                h: [from_z(h, lo), from_z(h, hi)] for h, (lo, hi) in ranges.items()}}, indent=2))
    elif args.space != "full":
        raise SystemExit(f"未实现的搜索空间 {args.space!r}")
    with path.open("a") as f:
        for i in range(len(existing), args.n):
            # 每组配置的种子只由 (档, 编号, space) 决定：追加采样不会改动已有的配置
            rng = random.Random(f"{args.scale}/{i}/{args.space}")
            hp = sample_full(rng) if ranges is None else sample_in(rng, ranges)
            rec = {"id": f"{args.scale}-{i:03d}", "scale": args.scale, "space": args.space, **hp}
            if ranges is not None:
                rec["narrow_from"] = args.sources
            f.write(json.dumps(rec) + "\n")
    print(f"{path}：共 {max(args.n, len(existing))} 组")


# ---------------------------------------------------------------- 步数与命令


@dataclass(frozen=True)
class Plan:
    stable_steps: int  # S
    micro: int
    accum: int
    warmup_steps: int

    @property
    def every(self) -> int:
        return self.stable_steps // 8

    def step_at(self, frac: Fraction) -> int:
        return int(frac * self.stable_steps)


def plan_for(cfg: dict, neff: float) -> Plan:
    tps = cfg["tokens_per_step"]
    micro = MICRO[cfg["scale"]]
    seqs = tps // SEQ
    micro = min(micro, seqs)
    S = max(8, round(TOKENS_PER_PARAM * neff / tps / 8) * 8)
    return Plan(S, micro, seqs // micro, max(1, round(S * cfg["warmup_frac"])))


def with_decay(c: int) -> tuple[int, float]:
    """从第 c 步开始衰减、多训 25%：返回 (max_steps, decay_frac)，使 wsd() 里 decay_start 恰好等于 c。"""
    m = c + round(c / 4)
    return m, (m - c) / m


def run_dir(cfg: dict, branch: int | None = None) -> Path:
    d = OUT / cfg["scale"] / cfg["id"]
    return d if branch is None else d / f"branch_{branch}"


def train_cmd(cfg: dict, plan: Plan, out_dir: Path, max_steps: int, decay_frac: float,
              stop_step: int, eval_every: int, ckpt_every: int) -> list[str]:
    s = {
        "train.out_dir": str(out_dir),
        "train.max_steps": max_steps,
        "train.stop_step": stop_step,
        "train.micro_batch_size": plan.micro,
        "train.grad_accum_steps": plan.accum,
        "train.eval_every": eval_every,
        "checkpoint.every": ckpt_every,
        "schedule.warmup_steps": plan.warmup_steps,
        "schedule.decay_frac": repr(decay_frac),
        "optim.lr": repr(cfg["lr"]),
        "optim.beta1": repr(cfg["beta1"]),
        "optim.beta2": repr(cfg["beta2"]),
        "optim.weight_decay": repr(cfg["weight_decay"]),
        "model.rope_theta": repr(cfg["rope_theta"]),
    }
    cmd = [sys.executable, "-m", "zero.train.pretrain", "--config",
           str(REPO / "configs" / "ladder3090" / f"{cfg['scale']}.toml")]
    for k, v in s.items():
        cmd += ["--set", f"{k}={v}"]
    return cmd


# ---------------------------------------------------------------- 读进度与结果


def read_log(out_dir: Path) -> list[dict]:
    p = out_dir / "log.jsonl"
    if not p.exists():
        return []
    rows = []
    for line in p.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:  # 被杀掉时最后一行可能写了一半
            pass
    return rows


def val_at(out_dir: Path) -> dict[int, float]:
    """step → val_bpb；发散时可能写出 NaN/inf，一律丢掉（排序时 NaN 会把顺序搅乱）。"""
    return {r["step"]: r["val_bpb"] for r in read_log(out_dir)
            if isinstance(r.get("val_bpb"), (int, float)) and math.isfinite(r["val_bpb"])}


def failed(out_dir: Path) -> bool:
    """训练进程非零退出（多半是 loss 发散，训练器会抛 FloatingPointError）时写下的标记。"""
    return (out_dir / "FAILED").exists()


def last_ckpt_step(out_dir: Path) -> int:
    from zero.train.checkpoint import find_latest

    p = find_latest(out_dir / "ckpt")
    return int(p.name.split("_")[1]) if p is not None else 0


def reached(cfg: dict, plan: Plan, step: int) -> bool:
    """训到过这一步：看日志里有没有这一步之后的评估，而不只看 checkpoint——prune 会删掉 checkpoint，
    只看 checkpoint 会把已经训完的配置从头再训一遍（2026-10-03 重启后发生过）。"""
    return last_ckpt_step(run_dir(cfg)) >= step or any(s >= step for s in val_at(run_dir(cfg)))


def finished_trunk(cfg: dict, plan: Plan) -> bool:
    m, _ = with_decay(plan.stable_steps)
    return m in val_at(run_dir(cfg))


# ---------------------------------------------------------------- 调度


@dataclass
class Job:
    name: str
    cmd: list[str]
    out_dir: Path


def run_jobs(jobs: list[Job], gpus: list[str], per_gpu: int, dry: bool = False) -> None:
    """每张卡最多并发 per_gpu 个训练进程；一个结束就补下一个。可以随时 Ctrl-C，重跑时从 checkpoint 接着训。"""
    if dry:
        for j in jobs:
            print(j.name, " ".join(j.cmd))
        return
    slots = [g for g in gpus for _ in range(per_gpu)]
    running: dict[str, tuple[subprocess.Popen, Job, float]] = {}
    pending = list(jobs)
    t0 = time.time()
    done = n_failed = 0

    def stop_children(*_: object) -> None:  # 调度器被中断时一起结束训练进程，免得重跑时两个进程写同一个目录
        for p, _, _ in running.values():
            p.terminate()
        for p, _, _ in running.values():
            p.wait()
        raise SystemExit("调度器被中断，已结束正在训练的进程（重跑会从 checkpoint 接着训）")

    signal.signal(signal.SIGTERM, stop_children)
    signal.signal(signal.SIGINT, stop_children)
    while pending or running:
        for slot_key in [f"{g}#{i}" for i, g in enumerate(slots)]:
            if slot_key not in running and pending:
                job = pending.pop(0)
                job.out_dir.mkdir(parents=True, exist_ok=True)
                env = {**os.environ, "CUDA_VISIBLE_DEVICES": slot_key.split("#")[0], "UV_NO_SYNC": "1"}
                log = (job.out_dir / "stdout.log").open("a")
                p = subprocess.Popen(job.cmd, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
                running[slot_key] = (p, job, time.time())
        time.sleep(5)
        for k in list(running):
            p, job, ts = running[k]
            if p.poll() is not None:
                del running[k]
                ok = p.returncode == 0
                done += ok
                n_failed += not ok
                if not ok:
                    tail = (job.out_dir / "stdout.log").read_text(errors="replace").splitlines()[-5:]
                    (job.out_dir / "FAILED").write_text(f"退出码 {p.returncode}，{time.strftime('%F %T')}\n"
                                                        + "\n".join(tail) + "\n")
                print(f"[{time.strftime('%H:%M:%S')}] {'完成' if ok else f'失败({p.returncode})'} {job.name} "
                      f"{(time.time() - ts) / 60:.1f} 分钟 | 已完成 {done}、失败 {n_failed}、剩余 {len(pending)}、"
                      f"运行中 {len(running)} | 总用时 {(time.time() - t0) / 3600:.2f} 小时", flush=True)


class scheduler_lock:  # noqa: N801
    """同一档同时只允许一个调度器（两个调度器会把同一个配置启动两次、写同一个目录）。"""

    def __init__(self, scale: str, dry: bool) -> None:
        self.path, self.dry, self.f = OUT / scale / ".scheduler.lock", dry, None

    def __enter__(self) -> None:
        if self.dry:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.f = self.path.open("w")
        try:
            fcntl.flock(self.f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit(f"{self.path} 被另一个调度器占着") from None

    def __exit__(self, *_: object) -> None:
        if self.f is not None:
            self.f.close()


def select(cfgs: list[dict], plans: dict[str, Plan], top: float, rank_at: Fraction | None) -> list[dict]:
    """按 rank_at 处的稳定段 val_bpb 留前 top（比例 <1 或个数 ≥1）；没有这个点的配置排最后。"""
    if rank_at is None or top >= len(cfgs):
        return cfgs
    def score(c: dict) -> float:
        if failed(run_dir(c)):
            return math.inf
        return val_at(run_dir(c)).get(plans[c["id"]].step_at(rank_at), math.inf)
    ranked = sorted(cfgs, key=score)
    k = max(1, round(top * len(cfgs))) if top < 1 else int(top)
    return ranked[:k]


def cmd_run(args: argparse.Namespace) -> None:
    neff = n_eff(args.scale)
    cfgs = load_configs(args.scale)
    if args.ids:
        cfgs = [c for c in cfgs if c["id"] in set(args.ids)]
    if args.only_space:
        cfgs = [c for c in cfgs if c["space"] == args.only_space]
    plans = {c["id"]: plan_for(c, neff) for c in cfgs}
    until = Fraction(args.until)
    cfgs = select(cfgs, plans, args.top, Fraction(args.rank_at) if args.rank_at else None)
    jobs = []
    for c in cfgs:
        if failed(run_dir(c)) and not args.retry_failed:
            continue
        pl = plans[c["id"]]
        m, f = with_decay(pl.stable_steps)
        stop = 0 if until >= 1 else pl.step_at(until)
        if finished_trunk(c, pl) or (stop and reached(c, pl, stop)):
            continue  # 已经训到了
        jobs.append(Job(c["id"], train_cmd(c, pl, run_dir(c), m, f, stop, pl.every, pl.every), run_dir(c)))
    n_failed = sum(failed(run_dir(c)) for c in cfgs)
    print(f"{args.scale}（N_eff {neff / 1e6:.2f}M）：{len(cfgs)} 组里 {len(jobs)} 组要训到 {args.until}"
          f"（已失败 {n_failed} 组{'，重试' if args.retry_failed else '，跳过'}）")
    with scheduler_lock(args.scale, args.dry_run):
        run_jobs(jobs, args.gpus, args.per_gpu, args.dry_run)


def cmd_branch(args: argparse.Namespace) -> None:
    neff = n_eff(args.scale)
    cfgs = [c for c in load_configs(args.scale) if finished_trunk(c, plan_for(c, neff))]
    plans = {c["id"]: plan_for(c, neff) for c in cfgs}
    best = sorted(cfgs, key=lambda c: val_at(run_dir(c))[with_decay(plans[c["id"]].stable_steps)[0]])
    jobs = []
    for c in best[: args.top]:
        pl = plans[c["id"]]
        for k in args.ks:
            start = pl.step_at(Fraction(k, 8))
            m, f = with_decay(start)
            bdir = run_dir(c, k)
            if m in val_at(bdir):
                continue
            src = run_dir(c) / "ckpt" / f"step_{start:08d}"
            dst_root = bdir / "ckpt"
            if not (dst_root / src.name).exists():
                dst_root.mkdir(parents=True, exist_ok=True)
                shutil.copytree(src, dst_root / src.name)
                (dst_root / "latest").write_text(src.name)
            # 分支只在最后评估一次，不再存 checkpoint
            jobs.append(Job(f"{c['id']}/b{k}", train_cmd(c, pl, bdir, m, f, 0, m, m), bdir))
    print(f"{args.scale}：最好的 {min(args.top, len(best))} 组 × 分支 {args.ks} → {len(jobs)} 个分支要训")
    with scheduler_lock(args.scale, args.dry_run):
        run_jobs(jobs, args.gpus, args.per_gpu, args.dry_run)


def cmd_collect(args: argparse.Namespace) -> None:
    neff = n_eff(args.scale)
    rows = []
    for c in load_configs(args.scale):
        pl = plan_for(c, neff)
        tps = c["tokens_per_step"]
        vals = val_at(run_dir(c))
        m, _ = with_decay(pl.stable_steps)
        for k in range(1, 9):
            step = pl.step_at(Fraction(k, 8))
            if step in vals:
                rows.append({**c, "n_eff": neff, "k": k, "phase": "stable", "steps": step,
                             "tokens": step * tps, "val_bpb": vals[step]})
            decayed_dir = run_dir(c) if k == 8 else run_dir(c, k)
            dstep = m if k == 8 else with_decay(step)[0]
            dv = val_at(decayed_dir).get(dstep)
            if dv is not None:
                rows.append({**c, "n_eff": neff, "k": k, "phase": "decay", "steps": dstep,
                             "tokens": dstep * tps, "val_bpb": dv})
    out = HERE / "results" / f"{args.scale}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        # 列取所有行的并集：narrow/ext 配置多一个 narrow_from（列表，写成 "+" 连接的字符串）
        fields = list(dict.fromkeys(k for r in rows for k in r))
        rows = [{k: "+".join(v) if isinstance(v, list) else v for k, v in r.items()} for r in rows]
        with out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
    print(f"{out}：{len(rows)} 行")


def cmd_prune(args: argparse.Namespace) -> None:
    """一档做完衰减分叉并汇总之后清理 checkpoint（全部保留的话五档要 300GB 以上）：
    最终 val_bpb 最好的 keep_top 组保留主干最后一步的权重（model.pt + meta.json，供以后评测），其余全部删除。"""
    neff = n_eff(args.scale)
    cfgs = load_configs(args.scale)
    final_step = {c["id"]: with_decay(plan_for(c, neff).stable_steps)[0] for c in cfgs}
    finals = {c["id"]: val_at(run_dir(c)).get(final_step[c["id"]]) for c in cfgs}
    finals = {k: v for k, v in finals.items() if v is not None}
    keep = set(sorted(finals, key=finals.get)[: args.keep_top])
    freed = 0
    for c in cfgs:
        d = run_dir(c)
        for root in [d / "ckpt", *(b / "ckpt" for b in d.glob("branch_*"))]:
            if not root.exists():
                continue
            for step_dir in [x for x in root.iterdir() if x.is_dir()]:
                if c["id"] in keep and root == d / "ckpt" and step_dir.name == f"step_{final_step[c['id']]:08d}":
                    victims = [f for f in step_dir.iterdir() if f.name not in ("model.pt", "meta.json")]
                else:
                    victims = [step_dir]
                for v in victims:
                    size = sum(f.stat().st_size for f in v.rglob("*")) if v.is_dir() else v.stat().st_size
                    freed += size
                    if not args.dry_run:
                        shutil.rmtree(v) if v.is_dir() else v.unlink()
    print(f"{args.scale}：{'将' if args.dry_run else '已'}释放 {freed / 1e9:.1f} GB；保留最好的 {len(keep)} 组的最终权重")


def cmd_status(args: argparse.Namespace) -> None:
    neff = n_eff(args.scale)
    for c in load_configs(args.scale):
        pl = plan_for(c, neff)
        vals = val_at(run_dir(c))
        best = min(vals.values()) if vals else float("nan")
        print(f"{c['id']}  S={pl.stable_steps:6d}  tok/步={c['tokens_per_step']:>8,}  lr={c['lr']:.2e}  "
              f"ckpt={last_ckpt_step(run_dir(c)):6d}  最好 val_bpb={best:.4f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("sample")
    p.add_argument("--scale", required=True)
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--space", default="full", choices=["full", "narrow", "ext"])
    p.add_argument("--from", dest="sources", nargs="*", default=[], help="narrow：按哪些小档的结果收窄")
    for name in ("run", "branch"):
        p = sub.add_parser(name)
        p.add_argument("--scale", required=True)
        p.add_argument("--gpus", nargs="+", default=["0"])
        p.add_argument("--per-gpu", type=int, default=1)
        p.add_argument("--dry-run", action="store_true")
        if name == "run":
            p.add_argument("--until", default="1")
            p.add_argument("--top", type=float, default=math.inf)
            p.add_argument("--rank-at", default=None)
            p.add_argument("--ids", nargs="*")
            p.add_argument("--retry-failed", action="store_true")
            p.add_argument("--only-space", default=None, help="只训这一类配置（如 ext），逐轮淘汰也只在它们之间排")
        else:
            p.add_argument("--top", type=int, default=8)
            p.add_argument("--ks", type=int, nargs="+", default=[3, 4, 5, 6, 7])
    for name in ("collect", "status"):
        p = sub.add_parser(name)
        p.add_argument("--scale", required=True)
    p = sub.add_parser("prune")
    p.add_argument("--scale", required=True)
    p.add_argument("--keep-top", type=int, default=8)
    p.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    {"sample": cmd_sample, "run": cmd_run, "branch": cmd_branch, "collect": cmd_collect,
     "status": cmd_status, "prune": cmd_prune}[args.cmd](args)


if __name__ == "__main__":
    main()
