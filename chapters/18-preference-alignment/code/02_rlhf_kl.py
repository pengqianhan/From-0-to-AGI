"""Chapter 18 · Minimal code 2: the RLHF objective, the KL leash, and how PPO solves it.

The RLHF objective (for one prompt):
    max_π  E_{y~π}[ r(y) ] − β · KL(π || π_ref)
r is the score from the reward model. π_ref is the SFT model. This objective has a closed-form
optimal solution:
    π*(y) = π_ref(y) · exp(r(y)/β) / Z
To make it easy to see, we reduce the "answers" to 8 candidates (a multi-armed bandit). Each
candidate has a true quality q and a length ℓ. The reward model scores them with the coefficients
that it learned in 01. It gives extra score to long answers, also to a 900-token "padded" answer.

① Sweep β: see how the proxy reward (the reward-model score), the true quality, and the KL change.
   This shows reward hacking and the KL leash.
② Start from π_ref and solve the same objective with PPO (sampling + clipped ratio + baseline).
   Check if PPO gets to the closed-form solution.

Run: uv run python chapters/18-preference-alignment/code/02_rlhf_kl.py
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

torch.set_num_threads(1)  # The build machine shares its CPU between jobs (you can remove this line).

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("bt01", HERE / "01_bradley_terry.py")
bt01 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bt01)

# 8 candidate answers: (name, true quality q, length ℓ in 100 tokens, logit of the SFT model).
# "concise" and "detailed" are the two correct answers. "padded 900" has 900 tokens of filler.
CANDIDATES = [
    ("concise", 1.4, 1.2, 1.0),
    ("detailed", 1.2, 2.5, 0.8),
    ("okay", 0.6, 1.0, 1.5),
    ("mediocre", 0.2, 1.5, 1.5),
    ("off-topic", -0.8, 1.0, 0.5),
    ("wrong", -1.5, 0.8, 0.5),
    ("verbose", 0.0, 4.0, 0.0),
    ("padded 900", -0.5, 9.0, -2.5),
]
NAMES = [c[0] for c in CANDIDATES]
Q = torch.tensor([c[1] for c in CANDIDATES])
FEATS = torch.tensor([[c[1], c[2]] for c in CANDIDATES])
LOGP_REF = torch.log_softmax(torch.tensor([c[3] for c in CANDIDATES]), 0)


def proxy_reward() -> torch.Tensor:
    """Score the 8 candidates with the reward model from 01 (it learned 'long = good')."""
    rm, _ = bt01.train_rm()
    with torch.no_grad():
        return rm(FEATS)


def optimal_policy(r: torch.Tensor, beta: float) -> torch.Tensor:
    """Closed-form optimal solution: log π* = log π_ref + r/β − log Z."""
    return torch.log_softmax(LOGP_REF + r / beta, 0)


def kl(logp: torch.Tensor, logq: torch.Tensor) -> float:
    return float((logp.exp() * (logp - logq)).sum())


def objective(logp: torch.Tensor, r: torch.Tensor, beta: float) -> float:
    return float((logp.exp() * r).sum()) - beta * kl(logp, LOGP_REF)


def ppo(r: torch.Tensor, beta: float, iters: int = 400, batch: int = 64, epochs: int = 4,
        clip: float = 0.2, lr: float = 0.05, seed: int = 0,
        trace_at: tuple[int, ...] | None = None) -> tuple[torch.Tensor, list[tuple]]:
    """PPO in the InstructGPT style (reduced to a bandit):
    - Each iteration samples `batch` answers from the current policy.
      Reward = r(y) − β·(log π_old(y) − log π_ref(y)) (a KL penalty for each sample).
    - Advantage A = reward − V. V is a scalar "value" baseline, learned with MSE.
      Here there is only one prompt, so V is one number.
    - Each batch gives `epochs` updates. The ratio ρ = π_θ/π_old is clipped to [1−ε, 1+ε]."""
    g = torch.Generator().manual_seed(seed)
    theta = LOGP_REF.clone().requires_grad_(True)  # logits of the policy; start from SFT
    v = torch.zeros(1, requires_grad=True)          # value baseline
    opt = torch.optim.Adam([theta, v], lr=lr)
    trace = []
    trace_at = trace_at or (0, 25, 100, iters)
    for it in range(iters + 1):
        with torch.no_grad():
            logp_old = torch.log_softmax(theta, 0)
            if it in trace_at:
                trace.append((it, objective(logp_old, r, beta), kl(logp_old, LOGP_REF)))
            y = torch.multinomial(logp_old.exp(), batch, replacement=True, generator=g)
            reward = r[y] - beta * (logp_old[y] - LOGP_REF[y])
        for _ in range(epochs):
            logp = torch.log_softmax(theta, 0)
            adv = (reward - v).detach()
            ratio = torch.exp(logp[y] - logp_old[y])
            surr = torch.minimum(ratio * adv, ratio.clamp(1 - clip, 1 + clip) * adv)
            loss = -surr.mean() + 0.5 * ((reward - v) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    return torch.log_softmax(theta.detach(), 0), trace


def main() -> None:
    r = proxy_reward()
    print("8 candidate answers (reward model = r = w_q·q + w_len·ℓ, learned in 01):")
    print(f"   {'answer':<10} {'true q':>9} {'len ℓ':>6} {'RM score':>9} {'π_ref':>7}")
    for i, n in enumerate(NAMES):
        print(f"   {n:<10} {Q[i]:>9.1f} {FEATS[i, 1]:>6.1f} {r[i]:>9.2f} {LOGP_REF[i].exp():>7.3f}")
    print("   Favorite of the reward model:", NAMES[int(r.argmax())], "; highest true quality:", NAMES[int(Q.argmax())])

    print("\n① Closed-form optimal solution π* ∝ π_ref·exp(r/β), sweep β:")
    print(f"   {'β':>6} | {'E[RM score]':>11} | {'E[true q]':>10} | {'KL(π||π_ref)':>12} | most likely answer")
    for beta in (100.0, 2.0, 1.0, 0.5, 0.25, 0.1, 0.03):
        lp = optimal_policy(r, beta)
        p = lp.exp()
        top = int(p.argmax())
        print(f"   {beta:>6} | {float((p * r).sum()):>11.3f} | {float((p * Q).sum()):>10.3f} | "
              f"{kl(lp, LOGP_REF):>12.3f} | {NAMES[top]} ({p[top]:.2f})")
    print("   Large β: almost no change (still SFT). Medium β: highest true quality. "
          "β too small: all on 'padded'. The proxy score is highest, but the true quality goes down.")

    beta = 0.5
    lp_star = optimal_policy(r, beta)
    lp_ppo, trace = ppo(r, beta)
    print(f"\n② PPO starts from π_ref and solves the same objective (β = {beta}, 64 samples per iteration, "
          f"4 epochs, ε = 0.2):")
    print(f"   {'iter':>4} | {'obj E[r]−β·KL':>13} | {'KL(π||π_ref)':>12}")
    for it, obj, k in trace:
        print(f"   {it:>4} | {obj:>13.4f} | {k:>12.4f}")
    print(f"   Objective of the closed-form solution = {objective(lp_star, r, beta):.4f}, KL = {kl(lp_star, LOGP_REF):.4f}")
    print(f"   Distance from the PPO result to the closed-form solution: KL(π_PPO || π*) = {kl(lp_ppo, lp_star):.5f}")
    print("   Probabilities (π_ref → π_PPO / π*):")
    for i, n in enumerate(NAMES):
        print(f"     {n:<10} {LOGP_REF[i].exp():.3f} → {lp_ppo[i].exp():.3f} / {lp_star[i].exp():.3f}")
    print("\nSummary: PPO needs sampling, a value baseline, and clipping to find π* step by step. "
          "But π* has a closed-form solution. This is the starting point of DPO (03).")


if __name__ == "__main__":
    main()
