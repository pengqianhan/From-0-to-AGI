"""循环 Transformer 对照实验（runs/ladder-3090/README.md "第 7 阶段"）的调度。

    setsid nohup env UV_NO_SYNC=1 LADDER_GPUS=0 .venv/bin/python runs/ladder-3090/loop.py \\
        >> out/ladder3090/loop.log 2>&1 < /dev/null &
    .venv/bin/python runs/ladder-3090/loop.py --plan     # 只打印计划的点（N_eff、D、步数、warmup），不训练

四个循环的"档"：e24m_lf3、e24m_lm3、e44m_lf3、e44m_lm3（模型见 configs/ladder3090/<档>.toml）。
配置写在 configs/<档>.jsonl，运行目录 out/ladder3090/<档>/<配置号>，结果汇总到 results/<档>.csv。
每一步都幂等，中断后重跑本脚本接着走。

每个配置沿用稠密那一档最好的配置（e24m-045、e44m-002）的全部超参数；稳定段步数、warmup 步数直接写明，
等于稠密那一次的（按稠密的 N_eff 算），所以每个点的 D 与稠密点一一对应。

步骤与事先定好的规则（结果记在 results/decisions.json）：
1. e24m 上每种结构（先 lf3、后 lm3）训 ×0.5、×1、×2 三条主干到 4/8，各做 4/8 的衰减分支；
   ×0.5、×2 中较好的一个比 ×1 好超过 0.5%（相对）就取它的倍数，否则取 ×1（loop_lr_lf3、loop_lr_lm3）。
   选中的主干接着训到 8/8（含衰减）。
2. e44m（先 lm3、后 lf3）用同一种结构选中的倍数，只训一组：主干到 8/8 + 4/8 的衰减分支。
   loss 不是有限数（发散）就学习率减半、换一个配置号重训一次，并记录（loop_lr_retry_<档>）。
3. 汇总四档，analyze.py loop 出对比表。

预注册没有写到的情况（这里事先定下，见 README 的报告）：e24m 的 ×0.5 或 ×2 发散算"没被采用"，照常比较；
×1 发散、选中的主干在 4/8 之后发散、e44m 减半后又发散、任何不是 loss 发散的失败（比如显存不够）都让这一块
停下来，等人看过再决定。停下的只是这一块和依赖它的 e44m（e44m_<结构> 要用 loop_lr_<结构>），另一种结构照常训；
最后照常汇总、分析，再以非零退出码结束。人看过之后删掉对应运行目录里的 FAILED 标记（或改配置）再重跑本脚本。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import orchestrate as orc  # noqa: E402
import sweep as sw  # noqa: E402

BASE = {"e24m": "e24m-045", "e44m": "e44m-002"}  # 稠密每档最好的配置
HP = ["tokens_per_step", "lr", "beta1", "beta2", "warmup_frac", "weight_decay", "rope_theta"]
STRUCTS = ("lf3", "lm3")
LR_CHECK = [(0, 1.0), (1, 0.5), (2, 2.0)]  # e24m 学习率检查：(配置号, 倍数)
LR_MULTS = (0.5, 2.0)
LR_MIN_GAIN = 0.005
GPUS = orc._CARDS if "LADDER_GPUS" in orc.os.environ else ["0"]
# 预注册里写的数（README 第 7 阶段），代码算出来的不一致就不开训
PREREG_NEFF = {"e24m_lf3": 38.21e6, "e24m_lm3": 31.06e6, "e44m_lf3": 82.67e6, "e44m_lm3": 57.11e6}
PREREG_PLAN = {"e24m": (11680, 233), "e44m": (21648, 2347)}  # (稳定段步数, warmup 步数)


def base_cfg(dense: str) -> dict:
    return next(c for c in sw.load_configs(dense) if c["id"] == BASE[dense])


def make(dense: str, struct: str, idx: int, lr_mult: float, **extra: object) -> dict:
    b = base_cfg(dense)
    pl = sw.plan_for(b, sw.n_eff(dense))  # 稠密的 N_eff：步数与 warmup 和稠密那一次完全相同
    scale = f"{dense}_{struct}"
    return {**{k: b[k] for k in HP}, "id": f"{scale}-{idx:03d}", "scale": scale, "space": "loop", "base": b["id"],
            "lr": b["lr"] * lr_mult, "lr_mult": lr_mult, "stable_steps": pl.stable_steps,
            "warmup_steps": pl.warmup_steps, **extra}


def write(scale: str, cfgs: list[dict]) -> None:
    path = sw.configs_path(scale)
    have = {c["id"] for c in sw.load_configs(scale)}
    with path.open("a") as f:
        for c in cfgs:
            if c["id"] not in have:
                f.write(json.dumps(c) + "\n")


def check_prereg() -> None:
    """N_eff、步数、warmup 必须与预注册一致（差一点都说明模型或配置不是预注册的那一个）。"""
    bad = []
    for s, v in PREREG_NEFF.items():
        if abs(sw.n_eff(s) - v) > 0.006e6:
            bad.append(f"{s} N_eff {sw.n_eff(s) / 1e6:.3f}M ≠ 预注册 {v / 1e6:.2f}M")
    for d, (S, w) in PREREG_PLAN.items():
        c = make(d, "lf3", 0, 1.0)
        if (c["stable_steps"], c["warmup_steps"], c["tokens_per_step"]) != (S, w, 65536):
            bad.append(f"{d} 稳定段 {c['stable_steps']}、warmup {c['warmup_steps']}、每步 {c['tokens_per_step']} token "
                       f"≠ 预注册 {S}、{w}、65536")
    if bad:
        raise SystemExit("与预注册不一致，停止：" + "；".join(bad))


def planned_points() -> list[dict]:
    """要训出来的点。e24m 每种结构的三个学习率都有 4/8 点；8/8 只有选中的那条（计划里记在 ×1 名下）。"""
    out = []
    for dense in ("e24m", "e44m"):
        for st in STRUCTS:
            cfgs = [make(dense, st, i, m) for i, m in LR_CHECK] if dense == "e24m" else [make(dense, st, 0, 1.0)]
            n = sw.n_eff(f"{dense}_{st}")
            for c in cfgs:
                pl = sw.plan_for(c, n)
                ends = [(4, sw.with_decay(pl.step_at(Fraction(4, 8)))[0])]
                if c["lr_mult"] == 1.0:
                    ends.append((8, sw.with_decay(pl.stable_steps)[0]))
                for k, m in ends:
                    d = m * c["tokens_per_step"]
                    out.append({"scale": c["scale"], "id": c["id"], "lr_mult": c["lr_mult"], "k": k, "N": n, "D": d,
                                "D_over_N": d / n, "steps": m, "stable_steps": pl.stable_steps,
                                "warmup_steps": pl.warmup_steps})
    return out


def plan_steps() -> dict[str, int]:
    """每档要训的总步数（主干 + 分支），用来估 GPU 时间。"""
    tot = {}
    for dense in ("e24m", "e44m"):
        S = PREREG_PLAN[dense][0]
        half = S // 2
        branch = sw.with_decay(half)[0] - half  # 4/8 分支：只训衰减段
        full = sw.with_decay(S)[0]
        for st in STRUCTS:
            # e24m：三条主干到 4/8 + 三个分支 + 选中的接着训完；e44m：一条主干训完 + 一个分支
            tot[f"{dense}_{st}"] = 3 * (half + branch) + (full - half) if dense == "e24m" else full + branch
    return tot


def final_val(cfg: dict, branch: int | None = None) -> float | None:
    pl = sw.plan_for(cfg, sw.n_eff(cfg["scale"]))
    start = pl.stable_steps if branch is None else pl.step_at(Fraction(branch, 8))
    return sw.val_at(sw.run_dir(cfg, branch)).get(sw.with_decay(start)[0])


def failure(cfg: dict, branches: tuple[int, ...] = ()) -> str | None:
    """None = 没有失败；"diverged" = loss 不是有限数（训练器抛 FloatingPointError）；"other" = 别的失败。"""
    dirs = [sw.run_dir(cfg), *(sw.run_dir(cfg, k) for k in branches)]
    bad = [d for d in dirs if sw.failed(d)]
    if not bad:
        return None
    for d in bad:
        text = (d / "FAILED").read_text(errors="replace")
        log = d / "stdout.log"
        if log.exists():
            text += "\n".join(log.read_text(errors="replace").splitlines()[-60:])
        if "Loss diverged" not in text and "FloatingPointError" not in text:
            return "other"
    return "diverged"


def decisions() -> dict:
    path = orc.RES / "decisions.json"
    return json.loads(path.read_text()) if path.exists() else {}


def decision(key: str) -> object:
    return decisions()[key]["value"]


def lr_check(st: str, g: list[str]) -> None:
    """e24m 上的学习率检查（规则 1），然后把选中的主干训完。"""
    scale = f"e24m_{st}"
    orc.log(f"===== {scale}：学习率检查 =====")
    cfgs = [make("e24m", st, i, m) for i, m in LR_CHECK]
    write(scale, cfgs)
    ids = [c["id"] for c in cfgs]
    orc.sweep("run", "--scale", scale, "--ids", *ids, "--until", "4/8", *g)
    ok = [c["id"] for c in cfgs if not sw.failed(sw.run_dir(c))]
    if ok:
        orc.sweep("branch", "--scale", scale, "--ids", *ok, "--ks", "4", "--top", len(ok), *g)
    v: dict[float, float] = {}
    for c in cfgs:
        f = failure(c, (4,))
        if f == "other" or (f == "diverged" and c["lr_mult"] == 1.0):
            raise SystemExit(f"{c['id']}（lr×{c['lr_mult']}）失败（{f}），预注册没有这种情况的规则：停下来等人看")
        val = math.inf if f == "diverged" else final_val(c, 4)
        if val is None:
            raise SystemExit(f"{scale} 学习率检查缺结果：{c['id']} 没有 4/8 衰减分支的 val_bpb")
        v[c["lr_mult"]] = val
    best = min(LR_MULTS, key=lambda m: v[m])
    mult = best if v[best] < v[1.0] * (1 - LR_MIN_GAIN) else 1.0
    fmt = lambda x: "发散" if math.isinf(x) else f"{x:.4f}"  # noqa: E731
    orc.decide(f"loop_lr_{st}", mult,
               f"{scale} 4/8 衰减分支（D = 0.478B）：lr×0.5 {fmt(v[0.5])}、×1 {fmt(v[1.0])}、×2 {fmt(v[2.0])}；"
               f"较好的 ×{best} 比 ×1 {'好' if v[best] < v[1.0] else '差'} "
               f"{'—' if math.isinf(v[best]) else f'{abs(v[best] / v[1.0] - 1):.2%}'}"
               f"（阈值 {LR_MIN_GAIN:.1%}）→ {scale}、e44m_{st} 用 lr×{mult}")
    mult = decision(f"loop_lr_{st}")  # 重跑时沿用第一次的决定
    chosen = next(c for c in cfgs if c["lr_mult"] == mult)
    orc.sweep("run", "--scale", scale, "--ids", chosen["id"], "--until", "1", *g)
    if failure(chosen):
        raise SystemExit(f"{chosen['id']} 在 4/8 之后失败（{failure(chosen)}），预注册没有这种情况的规则：停下来等人看")


def e44m(st: str, g: list[str]) -> None:
    """e44m 用 e24m 选中的倍数训一组（规则 2）；发散就学习率减半、换配置号重训一次。"""
    scale = f"e44m_{st}"
    orc.log(f"===== {scale} =====")
    mult = decision(f"loop_lr_{st}")
    cfg = make("e44m", st, 0, mult)
    write(scale, [cfg])
    for attempt in (0, 1):
        orc.sweep("run", "--scale", scale, "--ids", cfg["id"], "--until", "1", *g)
        if not failure(cfg):
            orc.sweep("branch", "--scale", scale, "--ids", cfg["id"], "--ks", "4", "--top", 1, *g)
        f = failure(cfg, (4,))
        if f is None:
            missing = [k for k in (4, None) if final_val(cfg, k) is None]
            if missing:
                raise SystemExit(f"{cfg['id']} 训完了却缺结果：" + "、".join("4/8 分支" if k else "8/8" for k in missing))
            return
        if f == "other" or attempt == 1:
            raise SystemExit(f"{cfg['id']}（lr×{cfg['lr_mult']}）失败（{f}）：停下来等人看")
        orc.decide(f"loop_lr_retry_{scale}", mult / 2,
                   f"{cfg['id']}（lr×{mult}，lr {cfg['lr']:.4g}）loss 发散 → 学习率减半（lr×{mult / 2}）"
                   f"用 {scale}-001 重训一次")
        cfg = make("e44m", st, 1, decision(f"loop_lr_retry_{scale}"), retry_of=cfg["id"])
        write(scale, [cfg])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true")
    args = ap.parse_args()
    check_prereg()
    if args.plan:
        for p in planned_points():
            print(f"{p['id']:14s} lr×{p['lr_mult']:<4} k={p['k']}/8 N_eff={p['N'] / 1e6:6.2f}M D={p['D'] / 1e9:6.3f}B "
                  f"D/N_eff={p['D_over_N']:5.1f} 步数={p['steps']:6d}（稳定段 {p['stable_steps']}）"
                  f"warmup={p['warmup_steps']}")
        print("（e24m 的 k=8 点属于学习率检查选中的那一条，这里记在 ×1 名下）")
        steps = plan_steps()
        print("总步数（主干 + 分支，每步 65,536 token）：" + "、".join(f"{s} {n:,}" for s, n in steps.items())
              + f"；合计 {sum(steps.values()):,} 步 = {sum(steps.values()) * 65536 / 1e9:.2f}B token")
        return
    g = ["--gpus", *GPUS]
    orc.log(f"循环 Transformer 对照实验开始（GPU {GPUS}）")
    stopped: dict[str, str] = {}

    def attempt(name: str, fn: object, *a: object) -> None:
        try:
            fn(*a)  # type: ignore[operator]
        except SystemExit as e:  # 只停这一块，别的照常（见文件头）
            stopped[name] = str(e)
            orc.log(f"!!! {name} 停下：{e}")

    for st in STRUCTS:  # 先 e24m：lf3、lm3
        attempt(f"e24m_{st}", lr_check, st, g)
    for st in ("lm3", "lf3"):  # 再 e44m：lm3、lf3
        if f"e24m_{st}" in stopped and f"loop_lr_{st}" not in decisions():
            stopped[f"e44m_{st}"] = f"e24m_{st} 停下了，还没有 loop_lr_{st}"
            orc.log(f"!!! e44m_{st} 跳过：{stopped[f'e44m_{st}']}")
            continue
        attempt(f"e44m_{st}", e44m, st, g)
    for s in PREREG_NEFF:
        orc.sweep("collect", "--scale", s)
    orc.analyze("loop")
    if stopped:
        raise SystemExit("循环 Transformer 对照实验没有全部完成，停下的：" + "；".join(f"{k}（{v}）" for k, v in stopped.items()))
    orc.log("循环 Transformer 对照实验完成")


if __name__ == "__main__":
    main()
