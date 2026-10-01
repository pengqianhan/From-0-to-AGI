"""配对 bootstrap：两个模型在同一组题上的分差，有多大把握不是运气？（GOAL.md 3.2 第 5 条）

    uv run python chapters/11-evaluation/code/04_paired_bootstrap.py

1. 从零写配对 bootstrap，给出 95% 置信区间和"超过 / 持平 / 落后"判定；
2. 用冒烟测试（极小配置演示）里真实的逐题结果做例子，并和 zero/eval/bootstrap.py 对拍；
3. 配对 vs 不配对：同样的数据，区间宽多少；
4. 题数：同样 3 个百分点的真实差距，30 / 300 / 3000 题各能不能判出来；
5. 多重比较：两个一模一样的模型比 6 次，至少一次"不是持平"的概率。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

AHEAD, TIE, BEHIND = "超过", "持平", "落后"

# ---------------------------------------------------------------------------
# 1. 从零实现
# ---------------------------------------------------------------------------


def paired_bootstrap(a, b, n_boot: int = 10_000, seed: int = 0, confidence: float = 0.95):  # noqa: ANN001, ANN201
    """a、b：同一组题上两个模型的逐题得分（对 = 1，错 = 0）。返回 (差值, 下界, 上界, 判定, 所有 d*)。"""
    d = np.asarray(a, float) - np.asarray(b, float)  # 逐题分差
    n = len(d)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))  # 每一行：有放回地抽 n 道题的下标（两个模型用同一组下标）
    stats = d[idx].mean(axis=1)  # 每次重抽的平均分差 d*
    alpha = 1 - confidence
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])  # 百分位法
    decision = AHEAD if lo > 0 else BEHIND if hi < 0 else TIE  # 整个区间在 0 的哪一边
    return float(d.mean()), float(lo), float(hi), decision, stats


def unpaired_bootstrap(a, b, n_boot: int = 10_000, seed: int = 0):  # noqa: ANN001, ANN201
    """对照：两个模型各自独立地重抽题目（丢掉了"同一道题"的配对信息）。"""
    a, b = np.asarray(a, float), np.asarray(b, float)
    rng = np.random.default_rng(seed)
    sa = a[rng.integers(0, len(a), size=(n_boot, len(a)))].mean(axis=1)
    sb = b[rng.integers(0, len(b), size=(n_boot, len(b)))].mean(axis=1)
    lo, hi = np.quantile(sa - sb, [0.025, 0.975])
    return float(lo), float(hi)


# ---------------------------------------------------------------------------
# 2. 冒烟测试的真实逐题结果（极小配置演示：约 1.3M 参数的 tiny 模型，分数接近随机，只说明代码通路）
#    来自 out/smoke/eval/results.json（`uv run python -m zero.smoke` 的产物；out/ 不进 git，所以抄在这里，
#    本地有这个文件时会自动核对一遍）
# ---------------------------------------------------------------------------
SMOKE = {
    ("sft", "toy_mc"): "100001010000000001000000100000",
    ("dpo", "toy_mc"): "100001010000100001000000100000",
    ("grpo", "toy_mc"): "101001010010100011000000100000",
    ("sft", "tool_dev"): "000000010001000000010000000001",
    ("dpo", "tool_dev"): "000000010001000000000000000001",
    ("grpo", "tool_dev"): "000000010001000000000000000001",
}
SMOKE_JSON = Path(__file__).resolve().parents[3] / "out/smoke/eval/results.json"


def smoke_scores(model: str, task: str) -> np.ndarray:
    return np.array([int(c) for c in SMOKE[(model, task)]], float)


def check_against_file() -> str:
    if not SMOKE_JSON.exists():
        return "（本地没有 out/smoke/eval/results.json，跳过核对）"
    res = json.loads(SMOKE_JSON.read_text())["results"]
    key = {"toy_mc": "correct", "tool_dev": "call_exact"}
    for (m, t), s in SMOKE.items():
        got = "".join(str(int(it[key[t]])) for it in res[m][t]["items"])
        if got != s:
            # 上面写死的是修复工具调用判分器（第 19 章第 6 节）之前那次冒烟测试的逐题结果，正文用的就是它；
            # 之后重跑的 out/smoke 判分更严（换一台机器训练，逐题结果也会不同），对不上是正常的，不算错误。
            return ("\n（注意：out/smoke 是修复判分器之后重跑的结果，与正文使用的修复前数据不同"
                    f"——例如 {m} 的 {t} 逐题结果不一致；下面继续用写死的修复前数据）")
    return "（已与 out/smoke/eval/results.json 逐题核对一致）"


def zero_parity(a, b, n_boot: int, seed: int) -> str:  # noqa: ANN001
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # 仓库根目录，才能 import zero
    try:
        from zero.eval.bootstrap import paired_bootstrap as zpb
    except ImportError:
        return "（没找到 zero 包，跳过对拍）"
    z = zpb(a, b, n_boot=n_boot, seed=seed)
    diff, lo, hi, dec, _ = paired_bootstrap(a, b, n_boot=n_boot, seed=seed)
    same = (abs(z.diff - diff) < 1e-12 and abs(z.ci_low - lo) < 1e-12
            and abs(z.ci_high - hi) < 1e-12 and z.decision == dec)
    return f"zero.eval.bootstrap.paired_bootstrap 给出 [{z.ci_low:+.3f}, {z.ci_high:+.3f}] {z.decision}，" \
           f"与本文件{'逐位一致' if same else '不一致！'}"


# ---------------------------------------------------------------------------
# 模拟用的"模型"：题目有难度 p_i，模型 m 在第 i 题答对的概率 = clip(p_i + delta_m)
# ---------------------------------------------------------------------------


def simulate_pair(rng: np.random.Generator, n: int, acc_a: float, acc_b: float):  # noqa: ANN201
    p = rng.beta(2, 2, size=n)  # 平均 0.5 的题目难度
    u = rng.random(n)  # 同一道题用同一个随机数：难题两个模型都容易错（相关性）
    a = (u < np.clip(p + acc_a - 0.5, 0, 1)).astype(float)
    b = (u < np.clip(p + acc_b - 0.5, 0, 1)).astype(float)
    flip = rng.random(n) < 0.3  # 30% 的题两个模型各自独立作答，别让两者完全同步
    b[flip] = (rng.random(flip.sum()) < np.clip(p[flip] + acc_b - 0.5, 0, 1)).astype(float)
    return a, b


def simulations() -> dict:
    """配对 vs 不配对、题数、多重比较三个模拟（视频也用这个函数）。"""
    rng = np.random.default_rng(0)
    out: dict = {}
    a, b = simulate_pair(rng, 300, 0.55, 0.50)
    _, lo, hi, dec, _ = paired_bootstrap(a, b)
    out["corr"] = float(np.corrcoef(a, b)[0, 1])
    out["paired"], out["paired_decision"] = (lo, hi), dec
    out["unpaired"] = unpaired_bootstrap(a, b)
    out["by_n"] = []
    for n in (30, 300, 3000):
        widths, wins = [], 0
        for _ in range(100):
            a, b = simulate_pair(rng, n, 0.53, 0.50)
            _, lo, hi, dec, _ = paired_bootstrap(a, b, n_boot=500, seed=int(rng.integers(1 << 30)))
            widths.append(hi - lo)
            wins += dec == AHEAD
        out["by_n"].append((n, float(np.mean(widths)), wins / 100))
    trials, any_hit, single_hit = 300, 0, 0
    for _ in range(trials):
        hits = []
        for _ in range(6):
            a, b = simulate_pair(rng, 30, 0.5, 0.5)
            hits.append(paired_bootstrap(a, b, n_boot=500, seed=int(rng.integers(1 << 30)))[3] != TIE)
        single_hit += hits[0]
        any_hit += any(hits)
    out.update(trials=trials, single_hit=single_hit / trials, any_hit=any_hit / trials)
    return out


def main() -> None:
    # --- 手算级的小例子 ---
    a = np.array([1, 1, 0, 1, 0, 1, 1, 0, 1, 1], float)
    b = np.array([1, 0, 0, 1, 0, 1, 0, 0, 1, 1], float)
    print("手算小例子：10 道题，模型 A 对 7 道、模型 B 对 5 道")
    d = (a - b).astype(int).tolist()
    print(f"  逐题分差 d = A − B = {d}，平均 {np.mean(d):+.2f}")
    rng = np.random.default_rng(1)
    for k in range(3):
        idx = rng.integers(0, 10, size=10)
        print(f"  第 {k + 1} 次重抽题号 {idx.tolist()} → d* = {np.mean(np.array(d)[idx]):+.2f}")
    diff, lo, hi, dec, _ = paired_bootstrap(a, b)
    print(f"  重抽 10000 次：95% 区间 [{lo:+.2f}, {hi:+.2f}] → {dec}\n")

    # --- 冒烟测试（极小配置演示）---
    print("【极小配置演示】冒烟测试的逐题结果，基线 = sft，2000 次重抽、种子 0（与 configs/tiny/eval.toml 相同）",
          check_against_file())
    print("| 比较 | 任务 | 题数 | 模型 | 基线 | 差值 | 95% 区间 | 判定 |")
    print("|---|---|---:|---:|---:|---:|---|---|")
    for m in ("dpo", "grpo"):
        for t in ("toy_mc", "tool_dev"):
            x, y = smoke_scores(m, t), smoke_scores("sft", t)
            diff, lo, hi, dec, _ = paired_bootstrap(x, y, n_boot=2000, seed=0)
            print(f"| {m} vs sft | {t} | {len(x)} | {x.mean():.3f} | {y.mean():.3f} | {diff:+.3f} | "
                  f"[{lo:+.3f}, {hi:+.3f}] | {dec} |")
    x, y = smoke_scores("grpo", "toy_mc"), smoke_scores("sft", "toy_mc")
    print("对拍：", zero_parity(x, y, 2000, 0))
    n10 = int(((x - y) != 0).sum())
    print(f"grpo 对 sft 的 toy_mc：30 题里只有 {n10} 题两者结果不同（全是 grpo 对、sft 错）——"
          "分差完全来自这几道题。\n")

    # --- 三个模拟（同一个随机数流，顺序固定，数字可复现）---
    sim = simulations()
    print(f"配对 vs 不配对（模拟：300 题，两个模型的对错相关系数 {sim['corr']:.2f}）")
    lo, hi = sim["paired"]
    ulo, uhi = sim["unpaired"]
    print(f"  配对：  [{lo:+.3f}, {hi:+.3f}]，宽 {hi - lo:.3f} → {sim['paired_decision']}")
    print(f"  不配对：[{ulo:+.3f}, {uhi:+.3f}]，宽 {uhi - ulo:.3f} → "
          f"{AHEAD if ulo > 0 else BEHIND if uhi < 0 else TIE}\n")
    print("题数够不够：真实水平 0.53 vs 0.50（差 3 个百分点），各模拟 100 次，看判成'超过'的比例")
    print("| 题数 | 平均区间宽度 | 判成'超过'的比例 |")
    print("|---:|---:|---:|")
    for n, width, win in sim["by_n"]:
        print(f"| {n} | {width:.3f} | {win:.2f} |")
    print(f"\n多重比较：两个真实水平完全相同的模型，30 题，模拟 {sim['trials']} 次")
    print(f"  比 1 次就判出'超过/落后'的比例：{sim['single_hit']:.2f}")
    print(f"  比 6 次（像冒烟测试那张表）至少有 1 次判出'超过/落后'的比例：{sim['any_hit']:.2f}")
    print("  → 预注册要事先指定'主结论看哪一个分数'，其余只作描述。")


if __name__ == "__main__":
    main()
