"""Chapter 25 · Minimal code 4: the speedup formula. Acceptance rate α, draft length k, draft cost c.

Assume that each draft token is accepted with the probability α (independently).
One round produces at most k+1 tokens:
  The 1st token always occurs (correction or bonus). The 2nd token needs the acceptance of the
  1st draft (probability α). The 3rd token needs the acceptance of the first two drafts (α²), and so on.
  E[output per round] = 1 + α + α² + … + α^k = (1 − α^{k+1}) / (1 − α)   (Leviathan et al. 2023, Eq. (1))
The cost of one round = 1 forward pass of the target + k forward passes of the draft
= (1 + k·c) "target steps", with c = draft step / target step:
  speedup = (1 − α^{k+1}) / ((1 − α)(1 + k·c))                            (Theorem 3.8)
Condition: the target model verifies k+1 positions in one pass as fast as it generates 1 token
(this is approximately true when decode is limited by memory bandwidth).

Run: uv run python chapters/25-mtp-speculative-decoding/code/04_speedup_formula.py
"""

from __future__ import annotations


def expected_tokens(alpha: float, k: int) -> float:
    return (1 - alpha ** (k + 1)) / (1 - alpha)


def speedup(alpha: float, k: int, c: float) -> float:
    return expected_tokens(alpha, k) / (1 + k * c)


def best_k(alpha: float, c: float, k_max: int = 32) -> tuple[int, float]:
    return max(((k, speedup(alpha, k, c)) for k in range(1, k_max + 1)), key=lambda t: t[1])


if __name__ == "__main__":
    alphas = (0.5, 0.6, 0.7, 0.8, 0.9)
    print("Expected number of tokens per round E = (1 − α^{k+1}) / (1 − α):")
    print("  α \\ k " + "".join(f"{k:>7d}" for k in (1, 2, 3, 4, 6, 8)) + "    k→∞")
    for a in alphas:
        row = "".join(f"{expected_tokens(a, k):7.2f}" for k in (1, 2, 3, 4, 6, 8))
        print(f"  {a:4.2f}  {row}  {1 / (1 - a):6.2f}")

    for c in (0.0, 0.05, 0.2):
        print(f"\nSpeedup (c = {c}):")
        print("  α \\ k " + "".join(f"{k:>7d}" for k in (1, 2, 3, 4, 6, 8)) + "   best k")
        for a in alphas:
            row = "".join(f"{speedup(a, k, c):7.2f}" for k in (1, 2, 3, 4, 6, 8))
            kb, sb = best_k(a, c)
            best = f"larger k is better (limit {1 / (1 - a):.2f}×)" if c == 0 else f"k={kb} ({sb:.2f}×)"
            print(f"  {a:4.2f}  {row}   {best}")

    print("\nCompare with the numbers in the papers:")
    print(
        f"  Leviathan et al. Table 1: α=0.8, k=5, c=0 → {speedup(0.8, 5, 0):.2f}× (paper 3.69×); "
        f"α=0.9, k=10 → {speedup(0.9, 10, 0):.2f}× (paper 6.86×)"
    )
    for a in (0.85, 0.90):
        print(
            f"  DeepSeek-V3 MTP self-speculation (k=1): acceptance rate of the 2nd token {a:.2f} → "
            f"{expected_tokens(a, 1):.2f} tokens per forward pass (the paper reports 1.8× TPS)"
        )
    print("\nRule for the best k: a higher α and a smaller c make more guesses useful. With a low α, more guesses only waste the compute of the draft.")
