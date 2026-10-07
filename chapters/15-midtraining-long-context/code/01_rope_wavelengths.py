"""Chapter 15 · Minimal code 1: the "wavelength" of each RoPE dimension pair. Why can a model not read
text that is longer than its training length?

RoPE puts the head_dim dimensions into d/2 pairs. At position m, pair i turns by m·ω_i radians:
    ω_i = θ^(-2i/d)              (rotation speed, i = 0 .. d/2-1)
    λ_i = 2π / ω_i = 2π·θ^(2i/d) (wavelength: the number of tokens for one full turn)
In the training length L, pair i turns L / λ_i times. A pair that does not make one full turn sees
only a part of the circle during training. In longer text, it meets "angles that it never saw".

This script uses head_dim = 128 of the main-line model. It compares θ = 10K, 500K (Llama 3),
and 1M (the long-context stage of Qwen3).
Run: uv run python chapters/15-midtraining-long-context/code/01_rope_wavelengths.py
"""

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]   # repository root, to import zero

HEAD_DIM = 128          # main-line model (configs/main/pretrain.toml)
TRAIN_LEN = 4096        # main-line pretraining length
TARGET_LEN = 32768      # target length of the long-context stage
BASES = [10_000.0, 500_000.0, 1_000_000.0]


def inv_freq(head_dim: int, theta: float) -> np.ndarray:
    """ω_i = θ^(-2i/d). Same as the default branch of zero.model.compute_rope_inv_freq."""
    i = np.arange(0, head_dim, 2) / head_dim          # 2i/d
    return theta ** (-i)


def wavelengths(head_dim: int, theta: float) -> np.ndarray:
    return 2 * math.pi / inv_freq(head_dim, theta)    # λ_i = 2π/ω_i


def fmt(x: float) -> str:
    if x >= 1e6:
        return f"{x / 1e6:.1f}M"
    if x >= 1e4:
        return f"{x / 1e3:.0f}K"
    return f"{x:.1f}"


def main() -> None:
    print(f"head_dim = {HEAD_DIM}, {HEAD_DIM // 2} dimension pairs; training length {TRAIN_LEN}, target length {TARGET_LEN}\n")

    # 1) Wavelengths of some dimension pairs (unit: tokens)
    picks = [0, 8, 16, 24, 32, 40, 48, 56, 63]
    print("1) Wavelength λ_i of each dimension pair (tokens for one full turn)")
    print("   i   " + "".join(f"{'θ=' + fmt(b):>12}" for b in BASES))
    for i in picks:
        row = "".join(f"{fmt(wavelengths(HEAD_DIM, b)[i]):>12}" for b in BASES)
        print(f"  {i:>3} {row}")

    # 2) How many pairs do not make one full turn in the training length
    print(f"\n2) Number of pairs that do not make one full turn in length L (λ_i > L), of {HEAD_DIM // 2} pairs")
    print("   L       " + "".join(f"{'θ=' + fmt(b):>12}" for b in BASES))
    for L in [TRAIN_LEN, TARGET_LEN]:
        row = "".join(f"{int((wavelengths(HEAD_DIM, b) > L).sum()):>12}" for b in BASES)
        print(f"  {L:>6}  {row}")

    # 3) At 32K, which pairs get to "angles that training did not show""
    print("\n3) At 32K: number of pairs that turn to angles not seen in training (θ=10K, length 4K)")
    w_train = inv_freq(HEAD_DIM, 1e4)
    settings = {
        "no change, θ=10K": w_train,
        "ABF: θ = 1M": inv_freq(HEAD_DIM, 1e6),
        "PI: all ω ÷ 8": w_train / 8,
    }
    for name, w in settings.items():
        n = unseen_pairs(w_train, w, TRAIN_LEN, TARGET_LEN)
        print(f"   {name:<16} {n:>3} pairs; two adjacent tokens differ by {w[0]:.3f} rad on the fastest pair")
    print("   Note: a larger base removes the \"unseen\" angles. But each angle means a new distance (part 4), so the model must train more.")
    print("   PI also makes the fastest pair 8× slower. Then the model cannot easily tell near positions apart.")

    # 4) After a larger base: how many times slower each pair turns
    print("\n4) θ from 10K to 1M: how many times slower each dimension pair turns")
    w_old, w_new = inv_freq(HEAD_DIM, 1e4), inv_freq(HEAD_DIM, 1e6)
    for i in [0, 16, 32, 48, 63]:
        print(f"   i={i:>2}: slower by {w_old[i] / w_new[i]:>7.1f}×")
    print("   (pair i = 0 does not change; a later pair becomes slower by more, at most about 100×)")

    # 5) The cost of long context: compute per training token of the main-line model (zero's formula)
    sys.path.insert(0, str(ROOT))
    from zero.config import load_model_config
    from zero.model import estimate_flops_per_token

    cfg = load_model_config(ROOT / "configs/main/pretrain.toml")
    print("\n5) Compute per training token of the main-line model (forward + backward, zero.model.estimate_flops_per_token)")
    base = estimate_flops_per_token(cfg, 0)
    for T in [4096, 32768]:
        total = estimate_flops_per_token(cfg, T)
        print(f"   seq len {T:>6}: {total / 1e9:5.2f} GFLOP/token, attention QKᵀ and AV are {(total - base) / total:.0%} of it")
    ratio = estimate_flops_per_token(cfg, 32768) / estimate_flops_per_token(cfg, 4096)
    print(f"   For the same number of tokens, training with 32K sequences costs {ratio:.2f}× more than with 4K")


def unseen_pairs(w_train: np.ndarray, w_new: np.ndarray, train_len: int, new_len: int) -> int:
    """In training, pair i sees the angles [0, L_train·ω_i]. A pair that made one full turn saw all angles.
    At new_len, the largest angle is new_len·ω'_i. An angle outside the training range is "unseen"."""
    seen = train_len * w_train
    full_circle = seen >= 2 * math.pi              # one full turn in training: all angles seen
    return int(((~full_circle) & (new_len * w_new > seen + 1e-9)).sum())


if __name__ == "__main__":
    main()
