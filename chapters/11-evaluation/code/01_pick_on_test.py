"""为什么要"先定考卷"：在测试集上挑模型，分数会被系统性地抬高（Goodhart 定律的最小演示）。

设定：K 个"模型"（可以想成 K 个 checkpoint、K 组超参、K 种提示词）真实水平完全一样，
在一个 n = 200 题的测试集上各考一次。如果我们看着测试分数挑最高的那个并报告它的分数，
报告值会高于真实水平——哪怕没有任何一个模型真的更好。

    uv run python chapters/11-evaluation/code/01_pick_on_test.py

做法（纯 NumPy，几秒）：
- 每道题有自己的难度 p_i（从 Beta(2, 2) 抽，平均 0.5）：所有模型在同一道题上答对的概率都是 p_i，
  所以模型之间的对错是相关的（真实的模型也是如此：难题大家都错）；
- 每个模型在每道题上独立地按 p_i 抛一次硬币；
- "在测试集上挑"：报告 K 个里测试分最高的那个；
- "用开发集挑"：在另一份同分布的开发集上挑，再去测试集上考一次——报告值是无偏的。
"""

from __future__ import annotations

import numpy as np

N_TEST = 200  # 测试集题数
N_TRIALS = 2000  # 重复整个"实验"的次数，取平均
TRUE_ACC = 0.5  # 所有模型的真实水平（Beta(2,2) 的均值）


def one_trial(rng: np.random.Generator, k: int) -> tuple[float, float]:
    """返回 (在测试集上挑出来的报告分, 用开发集挑出来再上测试集的报告分)。"""
    p_test = rng.beta(2, 2, size=N_TEST)  # 测试题的难度
    p_dev = rng.beta(2, 2, size=N_TEST)  # 开发集：同分布的另一批题
    test_scores = (rng.random((k, N_TEST)) < p_test).mean(axis=1)  # K 个模型的测试分
    dev_scores = (rng.random((k, N_TEST)) < p_dev).mean(axis=1)
    picked_on_test = test_scores.max()  # 看着测试分挑：报告的就是最大值
    picked_on_dev = test_scores[dev_scores.argmax()]  # 开发集上挑，测试集只考一次
    return float(picked_on_test), float(picked_on_dev)


def pick_table(ks=(1, 3, 10, 30)) -> list[tuple[int, float, float]]:  # noqa: ANN001
    """[(K, 在测试集上挑的平均报告值, 用开发集挑的平均报告值), ...]（视频也用这个函数）。"""
    rng = np.random.default_rng(0)
    rows = []
    for k in ks:
        res = np.array([one_trial(rng, k) for _ in range(N_TRIALS)])
        on_test, on_dev = res.mean(axis=0)
        rows.append((k, float(on_test), float(on_dev)))
    return rows


def main() -> None:
    print(f"真实水平：所有模型都是 {TRUE_ACC:.3f}；测试集 {N_TEST} 题；每行重复 {N_TRIALS} 次取平均\n")
    print("| 候选个数 K | 在测试集上挑（报告值） | 虚高 | 用开发集挑、测试集只考一次 |")
    print("|---:|---:|---:|---:|")
    for k, on_test, on_dev in pick_table():
        print(f"| {k} | {on_test:.3f} | {on_test - TRUE_ACC:+.3f} | {on_dev:.3f} |")
    print(
        "\n结论：候选越多，'在测试集上挑出来的最好成绩'越虚高；把挑选挪到开发集上，测试集只考一次，"
        "报告值就回到真实水平。这就是 GOAL.md 第 11 节'不要用预注册的测试基准调超参或挑 checkpoint'。"
    )


if __name__ == "__main__":
    main()
