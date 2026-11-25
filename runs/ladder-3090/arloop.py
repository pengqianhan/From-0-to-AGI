"""ARLoop 注意力残差实验（runs/ladder-3090/README.md "第 8 阶段"）的调度。

    setsid nohup env UV_NO_SYNC=1 LADDER_GPUS=0 .venv/bin/python runs/ladder-3090/arloop.py \\
        >> out/ladder3090/arloop.log 2>&1 < /dev/null &
    .venv/bin/python runs/ladder-3090/arloop.py --plan     # 只打印计划的点和估计的 GPU 时间，不训练、不写文件

新的"档"：e24m_ar3、e44m_ar3（滑动窗口）、e24m_arf3、e44m_arf3（完整历史），模型见 configs/ladder3090/<档>.toml。
另外在 e24m_lf3 这一档里加两个换种子的重训：e24m_lf3-s1338、e24m_lf3-s1339（配置里多一个 seed 键，
sweep.train_cmd 会加上 --set train.seed=…；原来的 e24m_lf3-000 是种子 1337）。
配置写在 configs/<档>.jsonl，运行目录 out/ladder3090/<档>/<配置号>。每一步都幂等，中断后重跑本脚本接着走。

每个配置沿用第 7 阶段 lf3 的做法：稠密最好配置（e24m-045、e44m-002）的全部超参数，稳定段步数、warmup 步数
与稠密那一次相同，所以每个点的 D 与 lf3、稠密点一一对应。

顺序与规则（预注册的表；决定记在 results/decisions.json）：
1. e24m_ar3：×0.5、×1、×2 三条主干到 4/8，各做 4/8 的衰减分支；×0.5、×2 中较好的一个比 ×1 好超过 0.5%
   就取它的倍数，否则取 ×1（arloop_lr）。选中的主干接着训到 8/8。
2. e44m_ar3：用 arloop_lr，主干到 8/8 + 4/8 的衰减分支。发散就学习率减半、换配置号重训一次（arloop_lr_retry_e44m_ar3）。
3. e24m_lf3 的两个种子（1338、1339）：lr×1（与 e24m_lf3-000 相同），主干到 8/8 + 4/8 的衰减分支。
4. e24m_arf3：用 arloop_lr，一组，主干到 8/8 + 4/8 的衰减分支。
5. e44m_arf3：用 arloop_lr，一组，同上；发散就学习率减半重训一次（arloop_lr_retry_e44m_arf3）。
然后汇总四个新档（sweep.py collect），analyze.py arloop 出对比表。

失败情况的处理与第 7 阶段相同（loop.py 文件头）：e24m_ar3 的 ×0.5 或 ×2 发散算"没被采用"；×1 发散、选中的主干
在 4/8 之后发散、e44m 减半后又发散、e24m 的单组运行（种子重训、e24m_arf3）发散、任何不是 loss 发散的失败
都让这一块停下来等人决定。依赖 arloop_lr 的块（2、4、5）在第 1 块没有做出决定时跳过；种子重训不依赖它，照常训。
最后照常汇总、分析，再以非零退出码结束。

e24m_lf3 这一档不在这里 collect：那样会改写第 7 阶段的 results/e24m_lf3.csv。analyze.py arloop 直接读运行目录的日志。
"""

from __future__ import annotations

import argparse
import math
import sys
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import loop  # noqa: E402
import orchestrate as orc  # noqa: E402
import sweep as sw  # noqa: E402

AR_SCALES = ("e24m_ar3", "e44m_ar3", "e24m_arf3", "e44m_arf3")
SEEDS = (1338, 1339)  # e24m_lf3 的重训；原来的 e24m_lf3-000 是 1337（base.toml 的 train.seed）
ORIG_SEED = 1337
LR_CHECK = loop.LR_CHECK  # [(配置号, 倍数)]：×1、×0.5、×2
LR_MULTS = loop.LR_MULTS
LR_MIN_GAIN = loop.LR_MIN_GAIN
GPUS = loop.GPUS
# 预注册里写的数（README 第 8 阶段），代码算出来的不一致就不开训
PREREG_NEFF = {"e24m_ar3": 38_207_488, "e24m_arf3": 38_207_488, "e44m_ar3": 82_673_664, "e44m_arf3": 82_673_664}
PREREG_MODEL = {  # (attn_res_block_layers, attn_res_window)；None = 完整历史
    "e24m_ar3": (2, 3), "e24m_arf3": (2, None), "e44m_ar3": (3, 3), "e44m_arf3": (3, None)}
PREREG_PLAN = loop.PREREG_PLAN  # 稳定段步数、warmup 步数与第 7 阶段相同
# 估 GPU 时间用的吞吐（tok/s，第 100 步之后的中位数）：开训前的试点（out/ladder3090/pilot_arloop，各 200 步）实测
PILOT_TOK_S = {"e24m_ar3": 85_664, "e24m_arf3": 85_230, "e44m_ar3": 52_440, "e44m_arf3": 51_221}
# 第 7 阶段 lf3：试点（out/ladder3090/pilot_loop，同样算法）与正式训练主干（loop_result.json）的吞吐。
# 正式训练比试点慢（e24m 91k → 84k），所以第二种估计 = 试点 × (lf3 实测 / lf3 试点)
LF3_PILOT_TOK_S = {"e24m": 91_041, "e44m": 58_532}
LF3_TOK_S = {"e24m": 83_864, "e44m": 58_534}


def make(scale: str, idx: int, lr_mult: float, **extra: object) -> dict:
    """新档的配置（与 loop.make 相同，只是 space 记为 arloop）。"""
    dense, struct = scale.split("_")
    return {**loop.make(dense, struct, idx, lr_mult), "space": "arloop", **extra}


def make_seed(seed: int) -> dict:
    """e24m_lf3 换种子重训：与 e24m_lf3-000 只差 id、seed、space。"""
    return {**loop.make("e24m", "lf3", 0, 1.0), "id": f"e24m_lf3-s{seed}", "space": "arloop", "seed": seed}


def write(scale: str, cfgs: list[dict]) -> None:
    loop.write(scale, cfgs)


def check_prereg() -> None:
    """N_eff、块的大小、窗口、步数、warmup、种子必须与预注册一致。"""
    from zero.config import load_config

    bad = []
    for s, v in PREREG_NEFF.items():
        if sw.n_eff(s) != v:
            bad.append(f"{s} N_eff {sw.n_eff(s):,.0f} ≠ 预注册 {v:,}")
        lf3 = f"{s.split('_')[0]}_lf3"
        if sw.n_eff(s) != sw.n_eff(lf3):
            bad.append(f"{s} N_eff 与 {lf3} 不同")
        m = load_config(sw.REPO / "configs" / "ladder3090" / f"{s}.toml").model
        want = PREREG_MODEL[s]
        if not (m.attn_res and m.n_loops == 3 and (m.attn_res_block_layers, m.attn_res_window) == want):
            bad.append(f"{s} 的模型（attn_res={m.attn_res}、n_loops={m.n_loops}、块 {m.attn_res_block_layers} 层、"
                       f"窗口 {m.attn_res_window}）≠ 预注册（块 {want[0]} 层、窗口 {want[1]}）")
    for d, (S, w) in PREREG_PLAN.items():
        for s in (f"{d}_ar3", f"{d}_arf3"):
            c = make(s, 0, 1.0)
            if (c["stable_steps"], c["warmup_steps"], c["tokens_per_step"]) != (S, w, 65536):
                bad.append(f"{s} 稳定段 {c['stable_steps']}、warmup {c['warmup_steps']}、每步 {c['tokens_per_step']} token "
                           f"≠ 预注册 {S}、{w}、65536")
    base_seed = load_config(sw.REPO / "configs" / "ladder3090" / "e24m_lf3.toml").train.seed
    if base_seed != ORIG_SEED:
        bad.append(f"e24m_lf3 的 train.seed = {base_seed} ≠ 1337")
    orig = next((c for c in sw.load_configs("e24m_lf3") if c["id"] == "e24m_lf3-000"), None)
    if orig is not None and (orig.get("lr_mult") != 1.0 or "seed" in orig):
        bad.append("e24m_lf3-000 不是种子 1337、lr×1 的那一组")
    dec = loop.decisions().get("loop_lr_lf3")
    if dec is not None and dec["value"] != 1.0:
        bad.append(f"第 7 阶段 lf3 选的是 lr×{dec['value']}，不是 ×1")
    if bad:
        raise SystemExit("与预注册不一致，停止：" + "；".join(bad))


def final_val(cfg: dict, branch: int | None = None) -> float | None:
    return loop.final_val(cfg, branch)


failure = loop.failure


def decisions() -> dict:
    return loop.decisions()


def decision(key: str) -> object:
    return decisions()[key]["value"]


def lr_check(g: list[str]) -> None:
    """e24m_ar3 上的学习率检查（规则 1），然后把选中的主干训完。"""
    scale = "e24m_ar3"
    orc.log(f"===== {scale}：学习率检查 =====")
    cfgs = [make(scale, i, m) for i, m in LR_CHECK]
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
    orc.decide("arloop_lr", mult,
               f"{scale} 4/8 衰减分支（D = 0.478B）：lr×0.5 {fmt(v[0.5])}、×1 {fmt(v[1.0])}、×2 {fmt(v[2.0])}；"
               f"较好的 ×{best} 比 ×1 {'好' if v[best] < v[1.0] else '差'} "
               f"{'—' if math.isinf(v[best]) else f'{abs(v[best] / v[1.0] - 1):.2%}'}"
               f"（阈值 {LR_MIN_GAIN:.1%}）→ e24m_ar3、e44m_ar3、e24m_arf3、e44m_arf3 用 lr×{mult}")
    mult = decision("arloop_lr")  # 重跑时沿用第一次的决定
    chosen = next(c for c in cfgs if c["lr_mult"] == mult)
    orc.sweep("run", "--scale", scale, "--ids", chosen["id"], "--until", "1", *g)
    if failure(chosen):
        raise SystemExit(f"{chosen['id']} 在 4/8 之后失败（{failure(chosen)}），预注册没有这种情况的规则：停下来等人看")


def single(scale: str, cfg: dict, g: list[str], retry: bool) -> None:
    """一组：主干到 8/8 + 4/8 的衰减分支。retry = True（e44m）时发散就学习率减半、换配置号重训一次。"""
    orc.log(f"===== {scale}：{cfg['id']}（lr×{cfg['lr_mult']}）=====")
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
        if f == "other" or not retry or attempt == 1:
            raise SystemExit(f"{cfg['id']}（lr×{cfg['lr_mult']}）失败（{f}）：停下来等人看")
        key = f"arloop_lr_retry_{scale}"
        orc.decide(key, cfg["lr_mult"] / 2,
                   f"{cfg['id']}（lr×{cfg['lr_mult']}，lr {cfg['lr']:.4g}）loss 发散 → 学习率减半"
                   f"（lr×{cfg['lr_mult'] / 2}）用 {scale}-001 重训一次")
        cfg = make(scale, 1, decision(key), retry_of=cfg["id"])
        write(scale, [cfg])


def e44m_ar3(g: list[str]) -> None:
    single("e44m_ar3", make("e44m_ar3", 0, decision("arloop_lr")), g, retry=True)


def seeds(g: list[str]) -> None:
    """e24m_lf3 换种子重训（估计运行间的波动 s）。两个种子都训完再判失败。"""
    scale = "e24m_lf3"
    cfgs = [make_seed(s) for s in SEEDS]
    write(scale, cfgs)
    orc.log(f"===== {scale}：种子 {SEEDS} =====")
    ids = [c["id"] for c in cfgs]
    orc.sweep("run", "--scale", scale, "--ids", *ids, "--until", "1", *g)
    ok = [c["id"] for c in cfgs if not sw.failed(sw.run_dir(c))]
    if ok:
        orc.sweep("branch", "--scale", scale, "--ids", *ok, "--ks", "4", "--top", len(ok), *g)
    bad = [f"{c['id']}（{failure(c, (4,))}）" for c in cfgs if failure(c, (4,))]
    if bad:
        raise SystemExit("种子重训失败：" + "、".join(bad) + "，预注册没有这种情况的规则：停下来等人看")
    for c in cfgs:
        missing = [k for k in (4, None) if final_val(c, k) is None]
        if missing:
            raise SystemExit(f"{c['id']} 训完了却缺结果：" + "、".join("4/8 分支" if k else "8/8" for k in missing))


def arf3(dense: str, g: list[str]) -> None:
    scale = f"{dense}_arf3"
    single(scale, make(scale, 0, decision("arloop_lr")), g, retry=dense == "e44m")


def planned_points() -> list[dict]:
    """要训出来的点（e24m_ar3 的 8/8 只有选中的那条，计划里记在 ×1 名下）。"""
    out = []
    jobs = [(make("e24m_ar3", i, m), True) for i, m in LR_CHECK] + [(make("e44m_ar3", 0, 1.0), False)]
    jobs += [(make_seed(s), False) for s in SEEDS] + [(make("e24m_arf3", 0, 1.0), False), (make("e44m_arf3", 0, 1.0), False)]
    for c, _ in jobs:
        n = sw.n_eff(c["scale"])
        pl = sw.plan_for(c, n)
        ends = [(4, sw.with_decay(pl.step_at(Fraction(4, 8)))[0])]
        if c["lr_mult"] == 1.0:
            ends.append((8, sw.with_decay(pl.stable_steps)[0]))
        for k, m in ends:
            d = m * c["tokens_per_step"]
            out.append({"scale": c["scale"], "id": c["id"], "lr_mult": c["lr_mult"], "seed": c.get("seed"), "k": k,
                        "N": n, "D": d, "D_over_N": d / n, "steps": m, "stable_steps": pl.stable_steps,
                        "warmup_steps": pl.warmup_steps})
    return out


def plan_steps() -> dict[str, int]:
    """每一块要训的总步数（主干 + 分支），用来估 GPU 时间。"""
    tot = {}
    for dense in ("e24m", "e44m"):
        S = PREREG_PLAN[dense][0]
        half = S // 2
        one = sw.with_decay(S)[0] + sw.with_decay(half)[0] - half  # 一组：主干训完 + 4/8 分支
        if dense == "e24m":
            tot["e24m_ar3"] = 3 * sw.with_decay(half)[0] + (sw.with_decay(S)[0] - half)
            tot["e24m_lf3（种子 ×2）"] = 2 * one
            tot["e24m_arf3"] = one
        else:
            tot["e44m_ar3"] = one
            tot["e44m_arf3"] = one
    return tot


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true")
    args = ap.parse_args()
    check_prereg()
    if args.plan:
        for p in planned_points():
            print(f"{p['id']:15s} lr×{p['lr_mult']:<4} 种子 {p['seed'] or '—':<5} k={p['k']}/8 "
                  f"N_eff={p['N'] / 1e6:6.2f}M D={p['D'] / 1e9:6.3f}B D/N_eff={p['D_over_N']:5.1f} "
                  f"步数={p['steps']:6d}（稳定段 {p['stable_steps']}）warmup={p['warmup_steps']}")
        print("（e24m_ar3 的 k=8 点属于学习率检查选中的那一条，这里记在 ×1 名下；e24m_lf3-000 是第 7 阶段的，已有）")
        steps = plan_steps()
        print("总步数（主干 + 分支，每步 65,536 token）：" + "、".join(f"{s} {n:,}" for s, n in steps.items())
              + f"；合计 {sum(steps.values()):,} 步 = {sum(steps.values()) * 65536 / 1e9:.2f}B token")
        for name, adj in (("按试点吞吐", False), ("按试点 × lf3 实测/试点", True)):
            tok_s = {s: v * (LF3_TOK_S[s[:4]] / LF3_PILOT_TOK_S[s[:4]] if adj else 1) for s, v in PILOT_TOK_S.items()}
            tok_s["e24m_lf3（种子 ×2）"] = LF3_TOK_S["e24m"]  # 种子重训：第 7 阶段 lf3 的实测
            hrs = {s: n * 65536 / tok_s[s] / 3600 for s, n in steps.items()}
            print(f"GPU·小时（{name}；种子重训按 lf3 实测）：" + "、".join(f"{s} {h:.1f}" for s, h in hrs.items())
                  + f"；合计 {sum(hrs.values()):.1f}")
        return
    g = ["--gpus", *GPUS]
    orc.log(f"ARLoop 注意力残差实验开始（GPU {GPUS}）")
    stopped: dict[str, str] = {}

    def attempt(name: str, fn: object, *a: object) -> None:
        try:
            fn(*a)  # type: ignore[operator]
        except SystemExit as e:  # 只停这一块，别的照常（见文件头）
            stopped[name] = str(e)
            orc.log(f"!!! {name} 停下：{e}")

    def needs_lr(name: str, fn: object, *a: object) -> None:
        if "arloop_lr" not in decisions():
            stopped[name] = "e24m_ar3 停下了，还没有 arloop_lr"
            orc.log(f"!!! {name} 跳过：{stopped[name]}")
            return
        attempt(name, fn, *a)

    attempt("e24m_ar3", lr_check, g)
    needs_lr("e44m_ar3", e44m_ar3, g)
    attempt("e24m_lf3 种子", seeds, g)
    needs_lr("e24m_arf3", arf3, "e24m", g)
    needs_lr("e44m_arf3", arf3, "e44m", g)
    for s in AR_SCALES:
        orc.sweep("collect", "--scale", s)
    orc.analyze("arloop")
    if stopped:
        raise SystemExit("ARLoop 注意力残差实验没有全部完成，停下的：" + "；".join(f"{k}（{v}）" for k, v in stopped.items()))
    orc.log("ARLoop 注意力残差实验完成")


if __name__ == "__main__":
    main()
