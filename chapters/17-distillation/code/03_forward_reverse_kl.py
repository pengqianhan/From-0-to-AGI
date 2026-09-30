"""第 17 章 · 极简代码 3：前向 KL 与反向 KL——"覆盖所有可能" vs "挑一个模式"

教师分布 p 是双峰的（两个高斯的混合，左峰稍高）；学生 q 只能是**一个**高斯（参数 μ、σ），装不下两个峰。
在同一个一维网格上，分别用梯度下降最小化：

  前向 KL(p ‖ q) = Σ p·(log p − log q)   —— logits 蒸馏 / 序列级蒸馏（在教师的样本上做最大似然）优化的方向
  反向 KL(q ‖ p) = Σ q·(log q − log p)   —— 在线策略蒸馏（学生自己采样、教师打分）优化的方向

看两者停在哪：前向 KL 把 q 摊开盖住两个峰（mode covering），连两峰之间教师几乎不给概率的"山谷"也分到不少；
反向 KL 让 q 缩到其中一个峰上（mode seeking），另一个峰被整个放弃。

运行：uv run python chapters/17-distillation/code/03_forward_reverse_kl.py      （约 3 秒）
"""

import math

import torch

torch.set_num_threads(1)

GRID = torch.linspace(-6, 6, 1201)
DX = float(GRID[1] - GRID[0])
# 教师：0.6·N(−2, 0.6²) + 0.4·N(2.5, 0.8²)
MODES = [(0.6, -2.0, 0.6), (0.4, 2.5, 0.8)]
VALLEY = (-0.5, 1.0)  # 两峰之间的山谷


def normal_pdf(x: torch.Tensor, mu, sigma) -> torch.Tensor:
    return torch.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * math.sqrt(2 * math.pi))


def teacher_p() -> torch.Tensor:
    p = sum(w * normal_pdf(GRID, m, s) for w, m, s in MODES)
    return p / (p.sum() * DX)


def student_q(mu: torch.Tensor, log_sigma: torch.Tensor) -> torch.Tensor:
    q = normal_pdf(GRID, mu, log_sigma.exp())
    return q / (q.sum() * DX)


def kl(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """KL(a ‖ b) 在网格上的数值积分；加 1e-30 防止 log 0。"""
    return (a * ((a + 1e-30).log() - (b + 1e-30).log())).sum() * DX


def fit(direction: str, mu0: float, sigma0: float = 1.0, steps: int = 3000, lr: float = 0.02):
    p = teacher_p()
    mu = torch.tensor(mu0, requires_grad=True)
    log_sigma = torch.tensor(math.log(sigma0), requires_grad=True)
    opt = torch.optim.Adam([mu, log_sigma], lr=lr)
    for _ in range(steps):
        q = student_q(mu, log_sigma)
        loss = kl(p, q) if direction == "forward" else kl(q, p)
        opt.zero_grad()
        loss.backward()
        opt.step()
    q = student_q(mu, log_sigma).detach()
    return float(mu.detach()), float(log_sigma.detach().exp()), q


def mass(q: torch.Tensor, lo: float, hi: float) -> float:
    m = (GRID >= lo) & (GRID <= hi)
    return float(q[m].sum() * DX)


def run_all() -> dict:
    """返回画图用的数据（视频 scenes.py 也调用它）。"""
    p = teacher_p()
    out = {"grid": GRID.tolist(), "p": p.tolist(), "fits": []}
    for direction, mu0 in [("forward", 0.0), ("reverse", -1.0), ("reverse", 1.5)]:
        mu, sigma, q = fit(direction, mu0)
        out["fits"].append({
            "direction": direction, "mu0": mu0, "mu": mu, "sigma": sigma, "q": q.tolist(),
            "valley_q": mass(q, *VALLEY),
            "left_q": mass(q, -6, VALLEY[0]), "right_q": mass(q, VALLEY[1], 6),
            "forward_kl": float(kl(p, q)), "reverse_kl": float(kl(q, p)),
        })
    out["valley_p"] = mass(p, *VALLEY)
    out["left_p"] = mass(p, -6, VALLEY[0])
    out["right_p"] = mass(p, VALLEY[1], 6)
    return out


def main() -> None:
    r = run_all()
    print("教师 p = 0.6·N(−2, 0.6²) + 0.4·N(2.5, 0.8²)；学生 q = N(μ, σ²)")
    print(f"教师的概率质量：左峰区 {r['left_p']:.3f}，山谷 [{VALLEY[0]}, {VALLEY[1]}] {r['valley_p']:.3f}，右峰区 {r['right_p']:.3f}")
    print(f"\n{'最小化':<16}{'起点 μ0':>8}{'→ μ':>8}{'σ':>7}{'左峰区':>8}{'山谷':>8}{'右峰区':>8}{'KL(p‖q)':>10}{'KL(q‖p)':>10}")
    for f in r["fits"]:
        name = "前向 KL(p‖q)" if f["direction"] == "forward" else "反向 KL(q‖p)"
        print(f"{name:<14}{f['mu0']:>8.1f}{f['mu']:>8.2f}{f['sigma']:>7.2f}{f['left_q']:>8.3f}{f['valley_q']:>8.3f}"
              f"{f['right_q']:>8.3f}{f['forward_kl']:>10.3f}{f['reverse_kl']:>10.3f}")
    print("\n前向 KL：q 必须在 p > 0 的每个地方都给概率（否则 log q → −∞），于是摊开盖住两峰，山谷也分到很多。")
    print("反向 KL：q 在 p ≈ 0 的地方给概率会被重罚，于是缩进一个峰里；落到哪个峰取决于起点。")


if __name__ == "__main__":
    main()
