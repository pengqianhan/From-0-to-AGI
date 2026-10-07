"""Chapter 17 · Minimal code 1: soft labels, temperature, the KD loss and its gradient, and a parity check with zero

1. A one-hot label vs the soft label of the teacher: how much "information" is in each label (entropy, in bits).
2. Temperature τ: divide the teacher logits by τ, then apply softmax. The distribution becomes flatter,
   and the "dark knowledge" (the relative sizes of the wrong answers) becomes visible.
3. KD loss = τ² · KL(p_T^τ ‖ p_S^τ). Derive the gradient by hand, ∂L/∂z_S = τ · (p_S^τ − p_T^τ),
   and test it with central differences.
4. Parity check with the production code zero.post.distill.kd_loss (also the top-k version).

Run: uv run python chapters/17-distillation/code/01_soft_labels.py      (about 2 s)
"""

import sys
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(1)
ROOT = Path(__file__).resolve().parents[3]  # repository root, to import zero

# A small "next word" example. The context is "今天天气很" ("Today the weather is very"),
# and the vocabulary has only 6 candidates: 好 good, 热 hot, 冷 cold, 晴 sunny, 猫 cat, 跑 run.
VOCAB = ["好", "热", "冷", "晴", "猫", "跑"]
TEACHER_LOGITS = np.array([4.0, 3.0, 2.2, 1.8, -2.0, -3.0])  # teacher: 好 > 热 > 冷 > 晴 >> 猫, 跑
STUDENT_LOGITS = np.array([2.0, 0.5, 1.5, -0.5, 0.0, -1.0])  # a student that has not learned well yet
TEMPS = [0.5, 1.0, 2.0, 4.0]


def softmax(z: np.ndarray, tau: float = 1.0) -> np.ndarray:
    z = z / tau
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def entropy_bits(p: np.ndarray) -> float:
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum()) + 0.0  # +0.0 changes -0.0 to 0.0


def kd_loss(z_s: np.ndarray, z_t: np.ndarray, tau: float) -> float:
    """τ² · KL(p_T^τ ‖ p_S^τ) = τ² · Σ_v p_T(v) · (log p_T(v) − log p_S(v))."""
    p_t, p_s = softmax(z_t, tau), softmax(z_s, tau)
    return float(tau * tau * (p_t * (np.log(p_t) - np.log(p_s))).sum())


def kd_grad(z_s: np.ndarray, z_t: np.ndarray, tau: float) -> np.ndarray:
    """∂L/∂z_S = τ² · (1/τ) · (p_S^τ − p_T^τ) = τ · (p_S^τ − p_T^τ)."""
    return tau * (softmax(z_s, tau) - softmax(z_t, tau))


def numeric_grad(z_s: np.ndarray, z_t: np.ndarray, tau: float, h: float = 1e-5) -> np.ndarray:
    g = np.zeros_like(z_s)
    for k in range(len(z_s)):
        e = np.zeros_like(z_s)
        e[k] = h
        g[k] = (kd_loss(z_s + e, z_t, tau) - kd_loss(z_s - e, z_t, tau)) / (2 * h)
    return g


def main() -> None:
    # ── 1. one-hot vs soft label ─────────────────────────────────────────
    onehot = np.eye(len(VOCAB))[0]
    p_t = softmax(TEACHER_LOGITS)
    print("1) Two labels for the same position (context: 今天天气很 __)")
    print("   token  " + "  ".join(f"{w:>5}" for w in VOCAB))
    print("   one-hot" + "  ".join(f"{x:6.3f}" for x in onehot))
    print("   teacher" + "  ".join(f"{x:6.3f}" for x in p_t))
    print(f"   Entropy: one-hot = {entropy_bits(onehot):.3f} bit, teacher soft label = {entropy_bits(p_t):.3f} bit")
    print("   one-hot only says 'the answer is 好'. The soft label also says '热, 冷, 晴 are possible; 猫, 跑 are not'.")

    # ── 2. Temperature ──────────────────────────────────────────────────
    print("\n2) Temperature τ makes the teacher distribution flatter (p_T^τ = softmax(z_T / τ))")
    print("   τ     " + "  ".join(f"{w:>5}" for w in VOCAB) + "   entropy  p(晴)/p(猫)")
    for tau in TEMPS:
        p = softmax(TEACHER_LOGITS, tau)
        ratio = p[3] / p[4]
        print(f"   {tau:<4} " + "  ".join(f"{x:6.3f}" for x in p) + f"   {entropy_bits(p):6.3f}  {ratio:8.1f}")
    print("   A larger τ gives more probability to the low-ranked candidates. Their order (晴 > 猫) does not change.")

    # ── 3. KD loss and gradient ─────────────────────────────────────────
    print("\n3) KD loss L = τ²·KL(p_T^τ ‖ p_S^τ), gradient ∂L/∂z_S = τ·(p_S^τ − p_T^τ)")
    for tau in [1.0, 2.0]:
        ana = kd_grad(STUDENT_LOGITS, TEACHER_LOGITS, tau)
        num = numeric_grad(STUDENT_LOGITS, TEACHER_LOGITS, tau)
        print(f"   τ={tau}: L = {kd_loss(STUDENT_LOGITS, TEACHER_LOGITS, tau):.4f}")
        print("     analytic  " + "  ".join(f"{x:+.4f}" for x in ana))
        print("     numerical " + "  ".join(f"{x:+.4f}" for x in num))
        print(f"     max difference {np.abs(ana - num).max():.2e}")
    ce_grad = softmax(STUDENT_LOGITS) - onehot
    print("   Compare: gradient of the hard-label cross-entropy, p_S − onehot = " + "  ".join(f"{x:+.4f}" for x in ce_grad))
    print("   The hard label pushes only '好' up and pushes all others down. KD pushes each token up or down as the teacher says.")

    # Why multiply by τ²: for a large τ, p ≈ 1/V + z/(Vτ), so the gradient ≈ (z_S − z_T)/(V·τ²) · τ²
    print("\n   Why multiply by τ²: without τ², the gradient size decreases as 1/τ² (below: gradient norm without τ²)")
    for tau in [1.0, 2.0, 4.0, 8.0]:
        g = kd_grad(STUDENT_LOGITS, TEACHER_LOGITS, tau) / (tau * tau)
        print(f"     τ={tau:<4} |∇| = {np.linalg.norm(g):.4f}   × τ² = {np.linalg.norm(g) * tau * tau:.4f}")

    # ── 4. Parity check with zero ───────────────────────────────────────
    sys.path.insert(0, str(ROOT))
    from zero.post.distill import kd_loss as zero_kd_loss

    torch.manual_seed(0)
    b, t, v = 2, 5, 11
    zs, zt = torch.randn(b, t, v) * 2, torch.randn(b, t, v) * 2
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.bool)
    print("\n4) Parity check with zero.post.distill.kd_loss (random logits, 2×5 positions, vocabulary 11, 2 positions masked)")
    for tau in [1.0, 2.0]:
        ours = np.mean([kd_loss(zs[i, j].numpy(), zt[i, j].numpy(), tau)
                        for i in range(b) for j in range(t) if mask[i, j]])
        prod = float(zero_kd_loss(zs, zt, mask, temperature=tau))
        print(f"   τ={tau}: minimal {ours:.6f}   zero {prod:.6f}   difference {abs(ours - prod):.1e}")
    # top-k: the teacher keeps only the k most probable tokens and normalizes them again; the student does not
    k = 3
    ours_k = []
    for i in range(b):
        for j in range(t):
            if not mask[i, j]:
                continue
            idx = np.argsort(-zt[i, j].numpy())[:k]
            p_t_k = softmax(zt[i, j].numpy()[idx])
            logp_s = np.log(softmax(zs[i, j].numpy()))[idx]
            ours_k.append(float((p_t_k * (np.log(p_t_k) - logp_s)).sum()))
    prod_k = float(zero_kd_loss(zs, zt, mask, topk=k))
    print(f"   top-{k}: minimal {np.mean(ours_k):.6f}   zero {prod_k:.6f}   difference {abs(np.mean(ours_k) - prod_k):.1e}")


if __name__ == "__main__":
    main()
