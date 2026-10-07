"""Chapter 18 · Minimal code 3: the DPO derivation, checked with numbers step by step.

The four steps of the derivation (README section 5):
  (1) For any π: E_π[r] − β·KL(π||π_ref) = β·log Z − β·KL(π||π*), where π* = π_ref·exp(r/β)/Z.
      → KL ≥ 0, so π* is the optimal solution.
  (2) Solve for r: r(y) = β·log(π*(y)/π_ref(y)) + β·log Z.
  (3) Put r into Bradley–Terry: β·log Z cancels in r(y_w) − r(y_l).
  (4) The result is the DPO loss −log σ(β·[(log π(y_w) − log π_ref(y_w)) − (log π(y_l) − log π_ref(y_l))]).
Then:
  ⑤ Parity check of our DPO loss from scratch against dpo_loss in zero/post/dpo.py (value and gradient).
  ⑥ Gradient = β·σ(−h): the worse the implicit reward ranks a pair, the harder the push.
  ⑦ Train DPO on the bandit of 02 with preference pairs only (no reward model, no sampling).
     Check if DPO gets to the closed-form RLHF solution π*.

Run: uv run python chapters/18-preference-alignment/code/03_dpo_derivation.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

torch.set_num_threads(1)  # The build machine shares its CPU between jobs (you can remove this line).

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))  # Lets the script import zero from the repository root
from zero.post.dpo import dpo_loss as zero_dpo_loss  # noqa: E402

_spec = importlib.util.spec_from_file_location("rlhf02", HERE / "02_rlhf_kl.py")
rlhf02 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rlhf02)


def dpo_loss(pi_w: torch.Tensor, pi_l: torch.Tensor, ref_w: torch.Tensor, ref_l: torch.Tensor,
             beta: float) -> torch.Tensor:
    """DPO loss from scratch. The inputs are the sequence log-probabilities of each answer, shape (B,)."""
    h = beta * ((pi_w - ref_w) - (pi_l - ref_l))  # difference of the implicit rewards
    return -F.logsigmoid(h).mean()                 # −log σ(h)


def step1_identity(beta: float = 0.5) -> float:
    """(1): for 5 random π, find the largest difference between E[r] − β·KL(π||π_ref) and β·log Z − β·KL(π||π*)."""
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
    """(2)(3): β·log(π*/π_ref) and r differ only by the same constant β·log Z. All pairwise differences are equal."""
    g = torch.Generator().manual_seed(1)
    K = 6
    logp_ref = torch.log_softmax(torch.randn(K, generator=g, dtype=torch.float64), 0)
    r = torch.randn(K, generator=g, dtype=torch.float64)
    logp_star = torch.log_softmax(logp_ref + r / beta, 0)
    implicit = beta * (logp_star - logp_ref)
    offset = r - implicit                        # must be β·log Z everywhere
    diff_pairs = (implicit[:, None] - implicit[None, :]) - (r[:, None] - r[None, :])
    log_z = float(torch.logsumexp(logp_ref + r / beta, 0))
    return float(offset.max() - offset.min()), float(diff_pairs.abs().max()), abs(float(offset[0]) - beta * log_z)


def step5_parity() -> tuple[float, float]:
    """⑤: parity check against zero.post.dpo.dpo_loss. Largest difference of the loss and of the gradient."""
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
    """⑥: one preference pair. Compare ∂L/∂log π(y_w) with −β·σ(−h)."""
    rows = []
    for h_target in (-4.0, -2.0, 0.0, 2.0, 4.0):
        pi_w = torch.tensor([h_target / beta], requires_grad=True)
        pi_l = torch.tensor([0.0], requires_grad=True)
        loss = dpo_loss(pi_w, pi_l, torch.zeros(1), torch.zeros(1), beta)
        gw, gl = torch.autograd.grad(loss, [pi_w, pi_l])
        rows.append((h_target, float(gw), float(-beta * torch.sigmoid(torch.tensor(-h_target)))))
        assert abs(float(gl) + float(gw)) < 1e-7  # the gradient of rejected has the same size and the opposite sign
    return rows


def step7_bandit(beta: float = 0.5, n_pairs: int = 20000, steps: int = 400) -> dict:
    """⑦: the 8 candidates of 02. Bradley–Terry(r) labels the preference pairs.

    DPO sees only the preference pairs. At the end, we compare the result with π*.
    """
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

    theta = logp_ref.clone().requires_grad_(True)  # the policy starts from π_ref
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
    print("(1) E[r] − β·KL(π||π_ref)  =  β·log Z − β·KL(π||π*)   largest difference for 5 random π:",
          f"{step1_identity():.1e}")
    print("    → The first term on the right does not depend on π. The second term is ≥ 0, and 0 only at π = π*. "
          "So π* is the optimal solution of the RLHF objective.")
    spread, pair_diff, z_err = step23_invert()
    print("(2) Solve for r: r = β·log(π*/π_ref) + β·log Z:")
    print(f"    range of r − β·log(π*/π_ref) over 6 answers: {spread:.1e} (it is one constant); "
          f"difference from β·log Z: {z_err:.1e}")
    print(f"(3) largest pairwise [implicit reward difference] − [true reward difference]: {pair_diff:.1e} "
          "→ Z cancels in Bradley–Terry")

    loss_diff, grad_diff = step5_parity()
    print("\n⑤ Our DPO loss from scratch vs dpo_loss in zero/post/dpo.py (16 pairs of random log-probabilities, β = 0.1):")
    print(f"    loss difference {loss_diff:.1e}, largest gradient difference {grad_diff:.1e}")

    print("\n⑥ Size of the gradient = β·σ(−h), h = implicit reward difference (β = 0.1, one preference pair):")
    print(f"    {'h':>5} | {'∂L/∂log π(y_w)':>15} | {'−β·σ(−h)':>9}")
    for h, gw, formula in step6_gradient():
        print(f"    {h:>+5.0f} | {gw:>15.5f} | {formula:>9.5f}")
    print("    The push is strongest at h < 0 (the implicit reward ranks rejected first). At large h, there is almost no push. "
          "The size of the correction follows the size of the error.")

    res = step7_bandit()
    print(f"\n⑦ DPO on the bandit ({res['n_pairs']} preference pairs, labeled with Bradley–Terry(r), β = 0.5):")
    print(f"    KL(π_ref || π*) = {res['kl_ref_star']:.4f}  →  KL(π_DPO || π*) = {res['kl_dpo_star']:.4f}")
    print(f"    {'answer':<10} {'π_ref':>6} {'π_DPO':>6} {'π*':>6}")
    for i, n in enumerate(rlhf02.NAMES):
        print(f"    {n:<10} {res['p_ref'][i]:>6.3f} {res['p_dpo'][i]:>6.3f} {res['p_star'][i]:>6.3f}")
    print("    No reward model and no sampling: one classification loss on preference pairs "
          "gets near the same optimal solution as RLHF.")


if __name__ == "__main__":
    main()
