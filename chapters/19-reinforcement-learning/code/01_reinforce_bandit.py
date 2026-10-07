"""Chapter 19 · 01: REINFORCE. Use the "log-derivative trick" to get a gradient from scores (NumPy only).

A 5-armed bandit: each arm gives a fixed reward (plus a little noise).
The policy is π_θ = softmax(θ). The objective is J(θ) = E_{a~π}[r(a)] = Σ_a π(a) r(a).

1. Log-derivative trick: ∇J = Σ_a r(a) ∇π(a) = Σ_a π(a) r(a) ∇log π(a) = E_{a~π}[ r(a) ∇log π(a) ]
   We do not need the formula for r. We only need to sample and to score. First, do a parity check
   against the exact gradient.
2. Baseline: for each constant b, E[(r − b) ∇log π] is equal to ∇J (because E[∇log π] = 0).
   But the variance can be very different. Shift all rewards by +10 and see the variance explode.
3. Training: use the same learning rate, with and without a baseline, for 300 steps.
   See which one learns first to pull the best arm.

    uv run python chapters/19-reinforcement-learning/code/01_reinforce_bandit.py
"""

from __future__ import annotations

import numpy as np

REWARDS = np.array([0.2, 0.5, 0.9, 0.4, 0.1])  # arm 2 is the best
NOISE = 0.1


def softmax(theta: np.ndarray) -> np.ndarray:
    z = np.exp(theta - theta.max())
    return z / z.sum()


def grad_log_pi(theta: np.ndarray, a: int) -> np.ndarray:
    """∇_θ log softmax(θ)[a] = onehot(a) − π."""
    g = -softmax(theta)
    g[a] += 1.0
    return g


def exact_grad(theta: np.ndarray, rewards: np.ndarray) -> np.ndarray:
    """J = Σ π_a r_a, ∂J/∂θ_k = π_k (r_k − J). We can calculate this only when we know the formula for r."""
    pi = softmax(theta)
    return pi * (rewards - pi @ rewards)


def reinforce_samples(theta, rewards, n, rng, baseline=0.0):
    """n single-sample REINFORCE estimates: (r − b) ∇log π(a). Returns shape (n, K)."""
    pi = softmax(theta)
    acts = rng.choice(len(pi), size=n, p=pi)
    r = rewards[acts] + NOISE * rng.standard_normal(n)
    g = np.eye(len(pi))[acts] - pi  # each row is grad_log_pi(theta, a), vectorized
    return (r - baseline)[:, None] * g, r


def train(rewards, lr, steps, batch, rng, use_baseline):
    """Sample `batch` actions per step. With a baseline, the baseline is the mean reward of this batch
    (this is the "group mean" of GRPO)."""
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
        theta = theta + lr * g / batch  # gradient **ascent**: make J larger
        hist.append(softmax(theta)[best])
    return np.array(hist)


def main() -> None:
    rng = np.random.default_rng(0)
    theta = np.array([0.3, -0.2, 0.0, 0.5, -0.4])

    print("== 1. Log-derivative trick: sampled estimate vs exact gradient ==")
    g_exact = exact_grad(theta, REWARDS)
    for n in (100, 10_000, 1_000_000):
        g_hat = reinforce_samples(theta, REWARDS, n, rng)[0].mean(0)
        print(f"  N={n:>9,}  estimate {np.array2string(g_hat, precision=4)}  "
              f"max diff vs exact {np.abs(g_hat - g_exact).max():.4f}")
    print(f"  exact gradient        {np.array2string(g_exact, precision=4)}")

    print("\n== 2. A baseline changes the variance, not the expectation (single-sample estimates, N=200,000) ==")
    print(f"  {'Reward':<14}{'Baseline b':<12}{'Max |mean - exact|':<22}{'Sum of var':>10}")
    var_table = {}
    for shift in (0.0, 10.0):
        rw = REWARDS + shift
        J = softmax(theta) @ rw
        for name, b in (("0", 0.0), ("J = E[r]", J)):
            s, _ = reinforce_samples(theta, rw, 200_000, rng, baseline=b)
            err = np.abs(s.mean(0) - exact_grad(theta, rw)).max()
            var = s.var(0).sum()
            var_table[(shift, name)] = var
            label = "raw r" if shift == 0 else "r + 10"
            print(f"  {label:<14}{name:<12}{err:<22.4f}{var:>10.4f}")
    ratio = var_table[(10.0, "0")] / var_table[(10.0, "J = E[r]")]
    print(f"  → With all rewards +10, the variance without a baseline is {ratio:.0f}× the variance with a baseline")

    print("\n== 3. Train for 300 steps (learning rate 0.5, 8 samples per step, rewards +10, mean of 5 seeds) ==")
    rw = REWARDS + 10.0
    for use_b in (False, True):
        curves = np.stack([train(rw, 0.5, 300, 8, np.random.default_rng(s), use_b)
                           for s in range(5)])
        mean = curves.mean(0)
        tag = "batch-mean baseline" if use_b else "no baseline        "
        marks = "  ".join(f"step {t:>3} {mean[t - 1]:.2f}" for t in (1, 50, 100, 300))
        print(f"  {tag:<10} π(best arm): {marks}")


if __name__ == "__main__":
    main()
