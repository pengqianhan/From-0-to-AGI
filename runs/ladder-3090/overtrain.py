"""过训练点（runs/ladder-3090/README.md "第 6 阶段"）：让 D 和 N 分开变化，把 L(N, D) 的下限 E 定下来。

    setsid nohup env UV_NO_SYNC=1 LADDER_GPUS=0 .venv/bin/python runs/ladder-3090/overtrain.py \\
        >> out/ladder3090/overtrain.log 2>&1 < /dev/null &
    .venv/bin/python runs/ladder-3090/overtrain.py --plan     # 只打印计划的点（N、D、步数），不训练

阶梯里每个点的 D/N 都在 15–40，这里在 e11m、e24m、e44m 上补训到 D/N 64–256，每档只用它最好的配置。
配置写在 configs/<档>_ot.jsonl，结果汇总到 results/<档>_ot.csv。每一步都幂等，中断后重跑本脚本接着走。

步骤与事先定好的规则（结果记在 results/decisions.json 的 ot_lr）：
1. e11m 在 D/N = 128 处比较三个学习率：×0.5、×2 两条独立主干，×1 是 D/N = 256 主干在 4/8 处的分支。
   三者的稳定段步数、warmup 步数完全相同，只有学习率不同。
2. 规则：×0.5、×2 里较好的一个比 ×1 好超过 0.5%（相对）就采用它的倍数，否则用 ×1。
   e24m（D/N = 128 主干 + 4/8 处分支 = 64）和 e44m（D/N = 64）都乘上这个倍数。
3. 最后把 e11m 的 ×1 主干接着训到 D/N = 256。
4. 汇总、按 README 里预注册的方式拟合（analyze.py ot-fit）。

warmup 固定为按阶梯原来的预算（32 × N_eff 的稳定段）算出的步数，与阶梯里的老点同一个配方。
"""

from __future__ import annotations

import argparse
import json
import sys
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import orchestrate as orc  # noqa: E402
import sweep as sw  # noqa: E402

BASE = {"e11m": "e11m-106", "e24m": "e24m-045", "e44m": "e44m-002"}  # 每档最好的配置
HP = ["tokens_per_step", "lr", "beta1", "beta2", "warmup_frac", "weight_decay", "rope_theta"]
LR_MULTS = (0.5, 2.0)
LR_MIN_GAIN = 0.005
GPUS = orc._CARDS if "LADDER_GPUS" in orc.os.environ else ["0"]

# (配置号, 档, 稳定段是 U 的几倍, 学习率倍数)：U = 1.6 × 阶梯的稳定段，主干训完是 1.25 × 稳定段，
# 所以 U 的主干到 D/N = 64，2U 到 128，4U 到 256；4/8 处的分支到一半
E11M = [("e11m-ot-000", 4, 1.0), ("e11m-ot-001", 2, 0.5), ("e11m-ot-002", 2, 2.0)]
LATER = {"e24m": [("e24m-ot-000", 2)], "e44m": [("e44m-ot-000", 1)]}


def base_cfg(scale: str) -> dict:
    return next(c for c in sw.load_configs(scale) if c["id"] == BASE[scale])


def make(scale: str, cid: str, units: int, lr_mult: float) -> dict:
    b = base_cfg(scale)
    p32 = sw.plan_for(b, sw.n_eff(scale))
    u = 8 * round(1.6 * p32.stable_steps / 8)
    return {**{k: b[k] for k in HP}, "id": cid, "scale": scale, "space": "ot", "base": b["id"],
            "lr": b["lr"] * lr_mult, "lr_mult": lr_mult, "stable_steps": units * u,
            "warmup_steps": p32.warmup_steps}


def write(scale: str, cfgs: list[dict]) -> None:
    path = sw.configs_path(f"{scale}_ot")
    have = {c["id"] for c in sw.load_configs(f"{scale}_ot")}
    with path.open("a") as f:
        for c in cfgs:
            if c["id"] not in have:
                f.write(json.dumps(c) + "\n")


def planned_points(lr_mult: float = 1.0) -> list[dict]:
    """要训出来的点：(档, N, D)。D 与学习率无关，预注册的预测只用到它们。"""
    cfgs = [make("e11m", *e) for e in E11M]
    cfgs += [make(s, cid, u, lr_mult) for s, es in LATER.items() for cid, u in es]
    out = []
    for c in cfgs:
        n = sw.n_eff(c["scale"])
        pl = sw.plan_for(c, n)
        ends = [sw.with_decay(pl.stable_steps)[0]]
        if c["id"] in ("e11m-ot-000", "e24m-ot-000"):
            ends.append(sw.with_decay(pl.step_at(Fraction(4, 8)))[0])
        for m in ends:
            d = m * c["tokens_per_step"]
            out.append({"scale": c["scale"], "id": c["id"], "lr_mult": c["lr_mult"], "N": n, "D": d,
                        "D_over_N": d / n, "steps": m, "warmup_steps": pl.warmup_steps})
    return out


def final_val(cfg: dict, branch: int | None = None) -> float | None:
    pl = sw.plan_for(cfg, sw.n_eff(cfg["scale"]))
    start = pl.stable_steps if branch is None else pl.step_at(Fraction(branch, 8))
    return sw.val_at(sw.run_dir(cfg, branch)).get(sw.with_decay(start)[0])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--no-e44m", action="store_true", help="不训 e44m 的 D/N = 64（省约 7.7 GPU·小时）")
    args = ap.parse_args()
    if args.no_e44m:
        LATER.pop("e44m")
    if args.plan:
        for p in planned_points():
            print(f"{p['id']:12s} lr×{p['lr_mult']:<4} N={p['N'] / 1e6:6.2f}M D={p['D'] / 1e9:6.3f}B "
                  f"D/N={p['D_over_N']:6.1f} 步数={p['steps']:6d} warmup={p['warmup_steps']}")
        return
    g = ["--gpus", *GPUS]
    orc.log(f"过训练点开始（GPU {GPUS}）")
    write("e11m", [make("e11m", *e) for e in E11M])
    # 1. e11m 学习率检查（D/N = 128）
    orc.sweep("run", "--scale", "e11m_ot", "--ids", "e11m-ot-001", "e11m-ot-002", "--until", "1", *g)
    orc.sweep("run", "--scale", "e11m_ot", "--ids", "e11m-ot-000", "--until", "4/8", *g)
    orc.sweep("branch", "--scale", "e11m_ot", "--ids", "e11m-ot-000", "--ks", "4", "--top", "1", *g)
    c = {x["id"]: x for x in sw.load_configs("e11m_ot")}
    v = {1.0: final_val(c["e11m-ot-000"], 4), 0.5: final_val(c["e11m-ot-001"]), 2.0: final_val(c["e11m-ot-002"])}
    if any(x is None for x in v.values()):
        raise SystemExit(f"e11m 学习率检查缺结果：{v}")
    best = min(LR_MULTS, key=lambda m: v[m])
    mult = best if v[best] < v[1.0] * (1 - LR_MIN_GAIN) else 1.0
    orc.decide("ot_lr", mult, f"e11m D/N=128：lr×0.5 {v[0.5]:.4f}、×1 {v[1.0]:.4f}、×2 {v[2.0]:.4f}；"
               f"较好的 ×{best} 比 ×1 {'好' if v[best] < v[1.0] else '差'} {abs(v[best] / v[1.0] - 1):.2%}"
               f"（阈值 {LR_MIN_GAIN:.1%}）→ e24m、e44m 用 lr×{mult}")
    mult = json.loads((orc.RES / "decisions.json").read_text())["ot_lr"]["value"]  # 重跑时沿用第一次的决定
    # 2. e24m、e44m
    for s, es in LATER.items():
        write(s, [make(s, cid, u, mult) for cid, u in es])
    orc.sweep("run", "--scale", "e24m_ot", "--until", "1", *g)
    orc.sweep("branch", "--scale", "e24m_ot", "--ids", "e24m-ot-000", "--ks", "4", "--top", "1", *g)
    if "e44m" in LATER:
        orc.sweep("run", "--scale", "e44m_ot", "--until", "1", *g)
    # 3. e11m ×1 主干接着训到 D/N = 256
    orc.sweep("run", "--scale", "e11m_ot", "--ids", "e11m-ot-000", "--until", "1", *g)
    # 4. 汇总与拟合
    for s in [f"{x}_ot" for x in ("e11m", *LATER)]:
        orc.sweep("collect", "--scale", s)
    orc.analyze("ot-fit")
    orc.log("过训练点完成")


if __name__ == "__main__":
    main()
