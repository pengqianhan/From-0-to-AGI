"""Chapter 17 · Minimal code 3: forward KL and reverse KL. "Cover all modes" vs "pick one mode"

The teacher distribution p has two peaks (a mix of two Gaussians; the left peak is a little higher).
The student q can only be **one** Gaussian (parameters μ, σ), so it cannot fit both peaks.
On the same 1D grid, we use gradient descent to minimize each of these:

  forward KL(p ‖ q) = Σ p·(log p − log q)   -- the direction that logits KD and sequence-level KD
                                               (maximum likelihood on teacher samples) optimize
  reverse KL(q ‖ p) = Σ q·(log q − log p)   -- the direction that on-policy distillation
                                               (the student samples, the teacher scores) optimizes

See where each one stops. Forward KL spreads q over both peaks (mode covering). Then the "valley"
between the peaks, where the teacher gives almost no probability, also gets a large share.
Reverse KL shrinks q onto one of the peaks (mode seeking). It gives up the other peak completely.

Run: uv run python chapters/17-distillation/code/03_forward_reverse_kl.py      (about 3 s)
"""

import math

import torch

torch.set_num_threads(1)

GRID = torch.linspace(-6, 6, 1201)
DX = float(GRID[1] - GRID[0])
# teacher: 0.6·N(−2, 0.6²) + 0.4·N(2.5, 0.8²)
MODES = [(0.6, -2.0, 0.6), (0.4, 2.5, 0.8)]
VALLEY = (-0.5, 1.0)  # the valley between the two peaks


def normal_pdf(x: torch.Tensor, mu, sigma) -> torch.Tensor:
    return torch.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * math.sqrt(2 * math.pi))


def teacher_p() -> torch.Tensor:
    p = sum(w * normal_pdf(GRID, m, s) for w, m, s in MODES)
    return p / (p.sum() * DX)


def student_q(mu: torch.Tensor, log_sigma: torch.Tensor) -> torch.Tensor:
    q = normal_pdf(GRID, mu, log_sigma.exp())
    return q / (q.sum() * DX)


def kl(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Numerical integral of KL(a ‖ b) on the grid. The 1e-30 prevents log 0."""
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
    """Return the data for the plots (the video scenes.py also calls this function)."""
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
    print("Teacher p = 0.6·N(−2, 0.6²) + 0.4·N(2.5, 0.8²); student q = N(μ, σ²)")
    print(f"Probability mass of the teacher: left-peak region {r['left_p']:.3f}, valley [{VALLEY[0]}, {VALLEY[1]}] {r['valley_p']:.3f}, right-peak region {r['right_p']:.3f}")
    print(f"\n{'minimize':<16}{'start μ0':>8}{'→ μ':>8}{'σ':>7}{'left':>8}{'valley':>8}{'right':>8}{'KL(p‖q)':>10}{'KL(q‖p)':>10}")
    for f in r["fits"]:
        name = "forward KL(p‖q) " if f["direction"] == "forward" else "reverse KL(q‖p) "
        print(f"{name:<14}{f['mu0']:>8.1f}{f['mu']:>8.2f}{f['sigma']:>7.2f}{f['left_q']:>8.3f}{f['valley_q']:>8.3f}"
              f"{f['right_q']:>8.3f}{f['forward_kl']:>10.3f}{f['reverse_kl']:>10.3f}")
    print("\nForward KL: q must give probability at each point where p > 0 (else log q → −∞). "
          "So q spreads over both peaks, and the valley also gets a lot.")
    print("Reverse KL: q gets a large penalty for probability where p ≈ 0. So q shrinks into one peak; "
          "the start point decides which peak.")


if __name__ == "__main__":
    main()
