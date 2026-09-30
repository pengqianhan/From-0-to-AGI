"""第 19 章 · 01：REINFORCE —— 用"对数导数技巧"从打分里算梯度（只用 NumPy）

一个 5 臂老虎机（bandit）：每只手臂给一个固定的奖励（加一点噪声）。
策略 π_θ = softmax(θ)，目标 J(θ) = E_{a~π}[r(a)] = Σ_a π(a) r(a)。

1. 对数导数技巧：∇J = Σ_a r(a) ∇π(a) = Σ_a π(a) r(a) ∇log π(a) = E_{a~π}[ r(a) ∇log π(a) ]
   —— 不需要知道 r 的公式，只要能采样、能打分。先和精确梯度对拍。
2. 基线（baseline）：E[(r − b) ∇log π] 对任何常数 b 都等于 ∇J（因为 E[∇log π] = 0），
   但方差可以差很多。把所有奖励平移 +10，看方差怎么爆掉。
3. 训练：同样的学习率，有/无基线，走 300 步，看谁先学会拉最好的那只手臂。

    uv run python chapters/19-reinforcement-learning/code/01_reinforce_bandit.py
"""

from __future__ import annotations

import numpy as np

REWARDS = np.array([0.2, 0.5, 0.9, 0.4, 0.1])  # 第 2 号手臂最好
NOISE = 0.1


def softmax(theta: np.ndarray) -> np.ndarray:
    z = np.exp(theta - theta.max())
    return z / z.sum()


def grad_log_pi(theta: np.ndarray, a: int) -> np.ndarray:
    """∇_θ log softmax(θ)[a] = onehot(a) − π。"""
    g = -softmax(theta)
    g[a] += 1.0
    return g


def exact_grad(theta: np.ndarray, rewards: np.ndarray) -> np.ndarray:
    """J = Σ π_a r_a，∂J/∂θ_k = π_k (r_k − J)：知道 r 的公式时才算得出来。"""
    pi = softmax(theta)
    return pi * (rewards - pi @ rewards)


def reinforce_samples(theta, rewards, n, rng, baseline=0.0):
    """n 个单样本 REINFORCE 估计：(r − b) ∇log π(a)。返回 (n, K)。"""
    pi = softmax(theta)
    acts = rng.choice(len(pi), size=n, p=pi)
    r = rewards[acts] + NOISE * rng.standard_normal(n)
    g = np.eye(len(pi))[acts] - pi  # 每一行都是 grad_log_pi(theta, a)，向量化
    return (r - baseline)[:, None] * g, r


def train(rewards, lr, steps, batch, rng, use_baseline):
    """每步采 batch 个动作；有基线时用这一批的平均奖励作基线（GRPO 的"组均值"就是它）。"""
    theta = np.zeros(len(rewards))
    best = int(np.argmax(rewards))
    hist = []
    for _ in range(steps):
        pi = softmax(theta)
        acts = rng.choice(len(pi), size=batch, p=pi)
        r = rewards[acts] + NOISE * rng.standard_normal(batch)
        b = r.mean() if use_baseline else 0.0
        g = np.zeros_like(theta)
        for a, ri in zip(acts, r):
            g += (ri - b) * grad_log_pi(theta, a)
        theta = theta + lr * g / batch  # 梯度**上升**：让 J 变大
        hist.append(softmax(theta)[best])
    return np.array(hist)


def main() -> None:
    rng = np.random.default_rng(0)
    theta = np.array([0.3, -0.2, 0.0, 0.5, -0.4])

    print("== 1. 对数导数技巧：采样估计 vs 精确梯度 ==")
    g_exact = exact_grad(theta, REWARDS)
    for n in (100, 10_000, 1_000_000):
        g_hat = reinforce_samples(theta, REWARDS, n, rng)[0].mean(0)
        print(f"  N={n:>9,}  估计 {np.array2string(g_hat, precision=4)}  "
              f"与精确梯度的最大差 {np.abs(g_hat - g_exact).max():.4f}")
    print(f"  精确梯度      {np.array2string(g_exact, precision=4)}")

    print("\n== 2. 基线不改变期望，只改变方差（单样本估计，N=200,000） ==")
    print(f"  {'奖励':<14}{'基线 b':<12}{'均值与精确梯度的最大差':<22}{'方差（各分量求和）':>10}")
    var_table = {}
    for shift in (0.0, 10.0):
        rw = REWARDS + shift
        J = softmax(theta) @ rw
        for name, b in (("0", 0.0), ("J = E[r]", J)):
            s, _ = reinforce_samples(theta, rw, 200_000, rng, baseline=b)
            err = np.abs(s.mean(0) - exact_grad(theta, rw)).max()
            var = s.var(0).sum()
            var_table[(shift, name)] = var
            label = "原始 r" if shift == 0 else "r + 10"
            print(f"  {label:<14}{name:<12}{err:<22.4f}{var:>10.4f}")
    ratio = var_table[(10.0, "0")] / var_table[(10.0, "J = E[r]")]
    print(f"  → 奖励整体 +10 时，不减基线的方差是减基线的 {ratio:.0f} 倍")

    print("\n== 3. 训练 300 步（学习率 0.5，每步 8 个样本，奖励 +10，5 个种子平均） ==")
    rw = REWARDS + 10.0
    for use_b in (False, True):
        curves = np.stack([train(rw, 0.5, 300, 8, np.random.default_rng(s), use_b)
                           for s in range(5)])
        mean = curves.mean(0)
        tag = "有基线（批均值）" if use_b else "无基线"
        marks = "  ".join(f"第{t:>3}步 {mean[t - 1]:.2f}" for t in (1, 50, 100, 300))
        print(f"  {tag:<10} π(最好的手臂)：{marks}")


if __name__ == "__main__":
    main()
