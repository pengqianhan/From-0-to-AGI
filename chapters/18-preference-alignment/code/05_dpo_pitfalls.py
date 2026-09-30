"""第 18 章 · 极简代码 5：DPO 的几个坑（沿用 04 的小模型和任务）

① 学习率太大：margin 冲得很高，模型却坏了——留出题答对的概率掉、采样出来的回答连格式都不对。
   zero 的冒烟测试踩过同一个坑（configs/tiny/dpo.toml 的注释）：lr = 5e-4 时 24 步 margin 冲到 4.8，
   接下来 GRPO 阶段的格式正确率从 0.16 掉到 0；改成 5e-5 较稳。
② chosen 的概率也会一起掉：DPO 只要求"chosen 比 rejected 涨得多（或掉得少）"，
   没要求 chosen 本身变大。当 rejected 和 chosen 很像（错答案只差 1 或 2）时，
   在这个玩具上 chosen 的 log 概率一路下降，留出题反而变差。
③ 过拟合：训练集上的隐式奖励准确率远高于留出题。

运行：uv run python chapters/18-preference-alignment/code/05_dpo_pitfalls.py   （CPU 时间约 1 分钟；机器繁忙时墙钟几分钟）
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

torch.set_num_threads(1)  # 构建机多任务共享 CPU（本机可删）

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("toy04", HERE / "04_toy_dpo.py")
toy = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(toy)

LRS = (1e-4, 1e-3, 1e-2, 5e-2)


def lr_sweep() -> list[dict]:
    ref = toy.sft_model()
    base = toy.evaluate(ref, ref)
    rows = [{"lr": 0.0, "margin": 0.0, "acc": 0.0, "logp_w": float("nan"), **base}]
    for lr in LRS:
        pol, hh = toy.run_dpo(beta=0.1, lr=lr, steps=150, log_every=150)
        rows.append({"lr": lr, "margin": hh[-1]["margin"], "acc": hh[-1]["acc"],
                     "logp_w": hh[-1]["logp_w"], **toy.evaluate(pol, ref)})
    return rows


def near_miss() -> dict:
    ref = toy.sft_model("near")
    base = toy.evaluate(ref, ref, mistakes="near")
    pol, hist = toy.run_dpo(beta=0.1, lr=1e-3, steps=150, log_every=25, mistakes="near")
    return {"base": base, "hist": hist, "after": toy.evaluate(pol, ref, mistakes="near")}


def main() -> None:
    rows = lr_sweep()
    print("① 扫学习率（β = 0.1，150 步；第一行 lr = 0 是 SFT 参考模型本身）：")
    print(f"   {'lr':>6} | {'margin':>7} | {'训练acc':>6} | {'留出答对概率':>10} | {'采样格式正确':>10} | {'采样答对':>7}")
    for r in rows:
        print(f"   {r['lr']:>6.0e} | {r['margin']:>+7.2f} | {r['acc']:>7.2f} | {r['p_correct']:>12.3f} | "
              f"{r['format']:>12.3f} | {r['sample_acc']:>8.3f}")
    print("   margin 越大 ≠ 越好：lr 太大时，模型为了把 rejected 压下去，把整片'数字'的概率都压坏了。")

    nm = near_miss()
    print("\n② 错答案只差 1 或 2（chosen 与 rejected 很像），β = 0.1，lr = 1e-3：")
    print(f"   {'步':>4} | {'margin':>7} | {'acc':>5} | {'log π(chosen)':>13} | {'log π(rejected)':>15}")
    for h in nm["hist"]:
        print(f"   {h['step']:>4} | {h['margin']:>+7.3f} | {h['acc']:.2f} | {h['logp_w']:>13.3f} | {h['logp_l']:>15.3f}")
    b, a = nm["base"], nm["after"]
    print(f"   留出题：答对的概率 {b['p_correct']:.3f} → {a['p_correct']:.3f} | 采样答对 {b['sample_acc']:.3f} → "
          f"{a['sample_acc']:.3f} | 格式 {b['format']:.3f} → {a['format']:.3f}")
    print("   损失在降、margin 在涨、训练 acc 在涨——可 chosen 的概率也在掉，留出题变差了。")
    print(f"\n③ 训练集隐式奖励 acc {nm['hist'][-1]['acc']:.2f} vs 留出题 {a['pref_acc']:.2f}（①里 lr = 1e-3："
          f"训练 {rows[2]['acc']:.2f} vs 留出 {rows[2]['pref_acc']:.2f}）：只看训练集指标会高估效果。")


if __name__ == "__main__":
    main()
