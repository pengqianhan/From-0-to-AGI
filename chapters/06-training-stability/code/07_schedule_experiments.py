"""第 6 章 · 极简代码 7：学习率调度、warmup、梯度裁剪各自在什么时候起作用

用 06 的网络和训练循环做三组实验：
  1. 余弦 vs WSD vs 恒定学习率；以及 WSD 的"随时收尾"：同一条 stable 轨迹，
     在第 400 步和第 800 步各分出一个 100 步的衰减，和"事先定好总长"的余弦比；
  2. 高学习率压力测试：去掉 warmup、去掉 RMSNorm 会怎样；
  3. 梯度裁剪：训练中途混进 5 批坏数据（输入放大 30 倍）。
运行：uv run python chapters/06-training-stability/code/07_schedule_experiments.py   （约 1.5 分钟）
"""

import copy
import importlib.util
import math
import time
from pathlib import Path

import torch

torch.set_num_threads(1)

_spec = importlib.util.spec_from_file_location("ablation", Path(__file__).with_name("06_ablation.py"))
ab = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ab)
sched = ab.sched


def clone(model, state):
    """复制一份模型和优化器状态（包括数据流的随机数状态），用来从同一个检查点分叉。"""
    g = torch.Generator()
    g.set_state(state["data"].get_state())
    st = dict(step=state["step"], data=g, m=[t.clone() for t in state["m"]],
              v=[t.clone() for t in state["v"]])
    return copy.deepcopy(model), st


def wsd_branching(cfg=ab.FULL, decay=100, branch_at=(400, 800)):
    """一条 stable 轨迹，在 branch_at 的每个点分出一个 decay 步的线性衰减。

    返回 {分叉点: (stable 末尾的验证损失, 衰减后的验证损失, 衰减段的训练损失曲线)}，
    以及主干（stable）的训练损失曲线。
    """
    peak, warm = cfg["lr"], cfg["warmup"]
    model = ab.init_model(cfg)
    state = ab.new_state(model, cfg)
    stable_lr = lambda s: peak * min(1.0, (s + 1) / warm)          # noqa: E731
    out, trunk, done = {}, [], 0
    for b in branch_at:
        trunk += ab.train_steps(model, state, cfg, stable_lr, b - done)
        done = b
        m2, s2 = clone(model, state)
        before = ab.evaluate(m2, cfg)[0]
        total = b + decay
        lr_decay = lambda s, total=total: sched.wsd(s, total, peak, warm, decay_frac=decay / total)  # noqa: E731
        tail = ab.train_steps(m2, s2, cfg, lr_decay, decay)
        out[b] = (before, ab.evaluate(m2, cfg)[0], [h[0] for h in tail])
    return out, [h[0] for h in trunk]


def cosine_run(total: int, cfg=ab.FULL):
    r = ab.run(dict(cfg, schedule="cosine"), steps=total)
    return r["val_loss"], r["losses"]


def stress(cfg, lr):
    r = ab.run(dict(cfg, lr=lr))
    early = [x for x in r["losses"][:150] if math.isfinite(x)]
    return max(early) if early else float("nan"), r


if __name__ == "__main__":
    t0 = time.time()
    fmt = ab.fmt
    print("一、学习率调度（全套配置，800 步）")
    for name in ["cosine", "wsd", "const"]:
        r = ab.run(dict(ab.FULL, schedule=name))
        print(f"  {name:<7s} 验证损失 {fmt(r['val_loss'])}")

    print("\n  WSD 的随时收尾：一条 stable 主干，在第 400、800 步各分叉衰减 100 步")
    branches, _ = wsd_branching()
    for b, (before, after, _) in branches.items():
        cos, _ = cosine_run(b + 100)
        print(f"    总长 {b + 100:>4d} 步：stable 末尾 {fmt(before)} → 衰减后 {fmt(after)}"
              f"   对照：专门跑一次 {b + 100} 步的余弦 {fmt(cos)}")
    print("    WSD 总共训练 400+100+400+100 = 1000 步；两次余弦要 500+900 = 1400 步")

    print("\n二、高学习率压力测试：η = 0.3（正常是 0.003）")
    print("  前 150 步里的最大训练损失 / 800 步后的验证损失")
    rows = [("全套", ab.FULL), ("去掉 warmup", dict(ab.FULL, warmup=0)),
            ("去掉 RMSNorm", dict(ab.FULL, norm=False))]
    for name, cfg in rows:
        peak, r = stress(cfg, lr=0.3)
        peak_s = fmt(peak) if peak < 1e3 else f"{peak:.1e}"
        print(f"  {name:<12s}{peak_s:>10s} / {fmt(r['val_loss'])}" + ("（发散）" if r["diverged"] else ""))

    print("\n三、梯度裁剪：第 300–304 步混进 5 批坏数据（输入放大 30 倍）")
    bad = tuple(range(300, 305))
    for norm_on in [True, False]:
        for clip in [1.0, None]:
            cfg = dict(ab.FULL, norm=norm_on, clip=clip)
            r = ab.run(cfg, bad_steps=bad)
            after = max(r["losses"][305:345])
            print(f"  RMSNorm {'开' if norm_on else '关'}、裁剪 {'开' if clip else '关'}："
                  f"坏数据之后 40 步内最大训练损失 {fmt(after)}，最终验证损失 {fmt(r['val_loss'])}"
                  f"，坏数据那几步的梯度范数最大 {max(r['grad_norms'][300:305]):.1f}")
    print(f"\n用时 {time.time() - t0:.0f} 秒")
