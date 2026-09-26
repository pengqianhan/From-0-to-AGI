"""第 18 章 · 极简代码 3：DPO 的推导，一步一步用数字验证

推导的四步（README 第 5 节）：
  (1) 对任意 π：E_π[r] − β·KL(π||π_ref) = β·log Z − β·KL(π||π*)，其中 π* = π_ref·exp(r/β)/Z
      → KL ≥ 0，所以 π* 就是最优解；
  (2) 反解：r(y) = β·log(π*(y)/π_ref(y)) + β·log Z；
  (3) 代入 Bradley–Terry：β·log Z 在 r(y_w) − r(y_l) 里消掉；
  (4) 得到 DPO 损失 −log σ(β·[(log π(y_w) − log π_ref(y_w)) − (log π(y_l) − log π_ref(y_l))])。
然后：
  ⑤ 从零写的 DPO 损失与 zero/post/dpo.py 的 dpo_loss 对拍（数值与梯度）；
  ⑥ 梯度 = β·σ(−h)：隐式奖励排错得越离谱，推得越用力；
  ⑦ 在 02 的老虎机上只用偏好对训练 DPO（没有奖励模型、不采样），看它是否走到 RLHF 的闭式解 π*。

运行：uv run python chapters/18-preference-alignment/code/03_dpo_derivation.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

torch.set_num_threads(1)  # 构建机多任务共享 CPU（本机可删）

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))  # 让脚本能 import 仓库根目录下的 zero
from zero.post.dpo import dpo_loss as zero_dpo_loss  # noqa: E402

_spec = importlib.util.spec_from_file_location("rlhf02", HERE / "02_rlhf_kl.py")
rlhf02 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rlhf02)


def dpo_loss(pi_w: torch.Tensor, pi_l: torch.Tensor, ref_w: torch.Tensor, ref_l: torch.Tensor,
             beta: float) -> torch.Tensor:
    """从零写的 DPO 损失。输入是每个回答的序列 log 概率 (B,)。"""
    h = beta * ((pi_w - ref_w) - (pi_l - ref_l))  # 隐式奖励之差
    return -F.logsigmoid(h).mean()                 # −log σ(h)


def step1_identity(beta: float = 0.5) -> float:
    """(1)：随机取 5 个 π，检查 E[r] − β·KL(π||π_ref) 与 β·log Z − β·KL(π||π*) 的最大差。"""
    g = torch.Generator().manual_seed(0)
    K = 6
    logp_ref = torch.log_softmax(torch.randn(K, generator=g, dtype=torch.float64), 0)
    r = torch.randn(K, generator=g, dtype=torch.float64)
    log_z = torch.logsumexp(logp_ref + r / beta, 0)
    logp_star = logp_ref + r / beta - log_z
    worst = 0.0
    for _ in range(5):
        logp = torch.log_softmax(2 * torch.randn(K, generator=g, dtype=torch.float64), 0)
        p = logp.exp()
        lhs = (p * r).sum() - beta * (p * (logp - logp_ref)).sum()
        rhs = beta * log_z - beta * (p * (logp - logp_star)).sum()
        worst = max(worst, float((lhs - rhs).abs()))
    return worst


def step23_invert(beta: float = 0.5) -> tuple[float, float, float]:
    """(2)(3)：β·log(π*/π_ref) 与 r 只差同一个常数 β·log Z；两两之差完全一样。"""
    g = torch.Generator().manual_seed(1)
    K = 6
    logp_ref = torch.log_softmax(torch.randn(K, generator=g, dtype=torch.float64), 0)
    r = torch.randn(K, generator=g, dtype=torch.float64)
    logp_star = torch.log_softmax(logp_ref + r / beta, 0)
    implicit = beta * (logp_star - logp_ref)
    offset = r - implicit                        # 应该处处等于 β·log Z
    diff_pairs = (implicit[:, None] - implicit[None, :]) - (r[:, None] - r[None, :])
    log_z = float(torch.logsumexp(logp_ref + r / beta, 0))
    return float(offset.max() - offset.min()), float(diff_pairs.abs().max()), abs(float(offset[0]) - beta * log_z)


def step5_parity() -> tuple[float, float]:
    """⑤：与 zero.post.dpo.dpo_loss 对拍，损失与梯度的最大差。"""
    g = torch.Generator().manual_seed(2)
    B = 16
    ins = [(-30 + 10 * torch.randn(B, generator=g)).requires_grad_(True) for _ in range(2)]
    refs = [-30 + 10 * torch.randn(B, generator=g) for _ in range(2)]
    ours = dpo_loss(ins[0], ins[1], refs[0], refs[1], beta=0.1)
    g_ours = torch.autograd.grad(ours, ins)
    theirs, _ = zero_dpo_loss(ins[0], ins[1], refs[0], refs[1], beta=0.1)
    g_theirs = torch.autograd.grad(theirs, ins)
    loss_diff = float((ours - theirs).abs().detach())
    grad_diff = max(float((a - b).abs().max()) for a, b in zip(g_ours, g_theirs))
    return loss_diff, grad_diff


def step6_gradient(beta: float = 0.1) -> list[tuple[float, float, float]]:
    """⑥：单个偏好对，∂L/∂log π(y_w) 与 −β·σ(−h) 对比。"""
    rows = []
    for h_target in (-4.0, -2.0, 0.0, 2.0, 4.0):
        pi_w = torch.tensor([h_target / beta], requires_grad=True)
        pi_l = torch.tensor([0.0], requires_grad=True)
        loss = dpo_loss(pi_w, pi_l, torch.zeros(1), torch.zeros(1), beta)
        gw, gl = torch.autograd.grad(loss, [pi_w, pi_l])
        rows.append((h_target, float(gw), float(-beta * torch.sigmoid(torch.tensor(-h_target)))))
        assert abs(float(gl) + float(gw)) < 1e-7  # rejected 的梯度大小相同、方向相反
    return rows


def step7_bandit(beta: float = 0.5, n_pairs: int = 20000, steps: int = 400) -> dict:
    """⑦：02 的 8 个候选。偏好对由 Bradley–Terry(r) 标注；DPO 只看偏好对，最后和 π* 比较。"""
    r = rlhf02.proxy_reward()
    logp_ref = rlhf02.LOGP_REF
    K = len(r)
    g = torch.Generator().manual_seed(3)
    a = torch.randint(0, K, (n_pairs,), generator=g)
    b = torch.randint(0, K, (n_pairs,), generator=g)
    keep = a != b
    a, b = a[keep], b[keep]
    a_wins = torch.rand(len(a), generator=g) < torch.sigmoid(r[a] - r[b])
    yw, yl = torch.where(a_wins, a, b), torch.where(a_wins, b, a)

    theta = logp_ref.clone().requires_grad_(True)  # 策略从 π_ref 出发
    opt = torch.optim.Adam([theta], lr=0.05)
    for _ in range(steps):
        logp = torch.log_softmax(theta, 0)
        loss = dpo_loss(logp[yw], logp[yl], logp_ref[yw], logp_ref[yl], beta)
        opt.zero_grad()
        loss.backward()
        opt.step()
    logp_dpo = torch.log_softmax(theta.detach(), 0)
    logp_star = rlhf02.optimal_policy(r, beta)
    return {
        "n_pairs": len(yw),
        "kl_dpo_star": rlhf02.kl(logp_dpo, logp_star),
        "kl_ref_star": rlhf02.kl(logp_ref, logp_star),
        "p_dpo": logp_dpo.exp(),
        "p_star": logp_star.exp(),
        "p_ref": logp_ref.exp(),
    }


def main() -> None:
    print("(1) E[r] − β·KL(π||π_ref)  =  β·log Z − β·KL(π||π*)   对 5 个随机 π 的最大差：",
          f"{step1_identity():.1e}")
    print("    → 右边第一项与 π 无关，第二项 ≥ 0 且只在 π = π* 时为 0：π* 就是 RLHF 目标的最优解。")
    spread, pair_diff, z_err = step23_invert()
    print("(2) 反解 r = β·log(π*/π_ref) + β·log Z：")
    print(f"    r − β·log(π*/π_ref) 在 6 个回答上的极差 {spread:.1e}（是同一个常数），与 β·log Z 的差 {z_err:.1e}")
    print(f"(3) 两两之差 [隐式奖励差] − [真实奖励差] 的最大值 {pair_diff:.1e} → 代入 Bradley–Terry 时 Z 消掉了")

    loss_diff, grad_diff = step5_parity()
    print("\n⑤ 从零写的 DPO 损失 vs zero/post/dpo.py 的 dpo_loss（16 对随机 log 概率，β = 0.1）：")
    print(f"    损失差 {loss_diff:.1e}，梯度最大差 {grad_diff:.1e}")

    print("\n⑥ 梯度的大小 = β·σ(−h)，h = 隐式奖励差（β = 0.1，单个偏好对）：")
    print(f"    {'h':>5} | {'∂L/∂log π(y_w)':>15} | {'−β·σ(−h)':>9}")
    for h, gw, formula in step6_gradient():
        print(f"    {h:>+5.0f} | {gw:>15.5f} | {formula:>9.5f}")
    print("    h < 0（隐式奖励把 rejected 排在前面）时推得最用力；h 很大时几乎不推——自带'错多少改多少'。")

    res = step7_bandit()
    print(f"\n⑦ 老虎机上的 DPO（{res['n_pairs']} 个偏好对，按 Bradley–Terry(r) 标注，β = 0.5）：")
    print(f"    KL(π_ref || π*) = {res['kl_ref_star']:.4f}  →  KL(π_DPO || π*) = {res['kl_dpo_star']:.4f}")
    print(f"    {'回答':<10} {'π_ref':>6} {'π_DPO':>6} {'π*':>6}")
    for i, n in enumerate(rlhf02.NAMES):
        print(f"    {n:<10} {res['p_ref'][i]:>6.3f} {res['p_dpo'][i]:>6.3f} {res['p_star'][i]:>6.3f}")
    print("    没有奖励模型、没有采样，只靠偏好对上的一个分类损失，走到了 RLHF 的同一个最优解附近。")


if __name__ == "__main__":
    main()
