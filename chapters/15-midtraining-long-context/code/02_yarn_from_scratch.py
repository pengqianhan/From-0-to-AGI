"""Chapter 15 · Minimal code 2: position interpolation (PI) and YaRN from scratch, with a parity check
against zero and Hugging Face.

Three methods extend the context from L to s·L. All three change only the RoPE rotation speeds ω_i
(no learnable parameter changes):
  - Adjusted base frequency (ABF): θ → θ'. Calculate all ω_i = θ'^(-2i/d) again.
  - Position interpolation (PI): divide all ω_i by s. This is the same as changing position m to m/s.
  - YaRN: put the pairs into three zones by "how many turns the pair makes in the training length":
      pairs that turn many times (high frequency): keep them as they are (extrapolation);
      pairs that do not make one full turn (low frequency): divide by s (interpolation, as in PI);
      pairs between the two: a linear ramp.
    YaRN also multiplies cos/sin by mscale = 0.1·ln(s) + 1. This is the same as multiplying the
    attention logits by mscale², which makes the softmax "sharper".

Run: uv run python chapters/15-midtraining-long-context/code/02_yarn_from_scratch.py
"""

import math
import sys
from pathlib import Path

import torch

torch.set_num_threads(1)
ROOT = Path(__file__).resolve().parents[3]   # repository root, to import zero


def rope_inv_freq(head_dim: int, theta: float) -> torch.Tensor:
    """ω_i = θ^(-2i/d), i = 0 .. d/2-1."""
    return theta ** (-torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)


def pi_inv_freq(head_dim: int, theta: float, s: float) -> torch.Tensor:
    """Position interpolation: make all dimension pairs s times slower."""
    return rope_inv_freq(head_dim, theta) / s


def yarn_inv_freq(
    head_dim: int, theta: float, s: float, orig_len: int, beta_fast: float = 32, beta_slow: float = 1
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Return (the new ω_i, the weight keep_i of "keep the original frequency" for each pair, mscale).

    The method is the same as in the official YaRN code, Hugging Face, and zero. First, find the pair
    index that makes exactly β turns in orig_len. Then make a linear ramp in the pair index between
    the two indices.
    """
    w = rope_inv_freq(head_dim, theta)

    def dim_with_turns(turns: float) -> float:
        # Solve orig_len·ω_i / (2π) = turns for the pair index i (i can be a fraction)
        return head_dim * math.log(orig_len / (turns * 2 * math.pi)) / (2 * math.log(theta))

    low = max(math.floor(dim_with_turns(beta_fast)), 0)                # pairs before it: > β_fast turns
    high = min(math.ceil(dim_with_turns(beta_slow)), head_dim - 1)     # pairs after it: < β_slow turns
    if low == high:
        high += 0.001
    i = torch.arange(head_dim // 2, dtype=torch.float32)
    ramp = ((i - low) / (high - low)).clamp(0, 1)   # 0 = high-frequency zone, 1 = low-frequency zone
    keep = 1 - ramp                                  # weight of the original frequency
    new_w = keep * w + (1 - keep) * w / s            # high: no change; low: ÷ s; between: a mix
    mscale = 0.1 * math.log(s) + 1.0                 # √(1/t) = 0.1·ln(s) + 1
    return new_w, keep, mscale


def yarn_inv_freq_paper(head_dim, theta, s, orig_len, alpha=1.0, beta=32.0):
    """Equations (10)–(13) of the paper, as written: the ramp is linear in the "number of turns r",
    not in the pair index. Only for comparison."""
    w = rope_inv_freq(head_dim, theta)
    r = orig_len * w / (2 * math.pi)                 # number of turns in the training length
    gamma = ((r - alpha) / (beta - alpha)).clamp(0, 1)
    return (1 - gamma) * w / s + gamma * w


def show_tiny_table() -> None:
    d, theta, s, L = 32, 10_000.0, 4.0, 64
    w = rope_inv_freq(d, theta)
    wy, keep, m = yarn_inv_freq(d, theta, s, L)
    print(f"1) Setup of the tiny experiment in this chapter: head_dim={d}, θ={theta:.0f}, training length {L}, extend s={s:.0f}× to {int(s * L)}")
    print("   i        λ turns in L    keep wt  ω (orig)    ω (PI)  ω (YaRN)")
    for i in range(d // 2):
        lam = 2 * math.pi / w[i].item()
        print(f"  {i:>2} {lam:>8.1f} {L / lam:>10.2f} {keep[i].item():>10.2f} "
              f"{w[i].item():>9.5f} {w[i].item() / s:>9.5f} {wy[i].item():>9.5f}")
    print(f"   YaRN mscale = 0.1·ln({s:.0f}) + 1 = {m:.4f} (the same as multiplying the attention logits by {m * m:.4f})")


def show_zones(head_dim: int, theta: float, s: float, L: int, name: str) -> None:
    _, keep, m = yarn_inv_freq(head_dim, theta, s, L)
    n_keep = int((keep == 1).sum())
    n_interp = int((keep == 0).sum())
    print(f"   {name:<34} keep {n_keep:>2} pairs, ramp {head_dim // 2 - n_keep - n_interp:>2} pairs, "
          f"full interpolation {n_interp:>2} pairs; mscale {m:.3f}")


def check_against_zero_and_hf() -> None:
    sys.path.insert(0, str(ROOT))
    from zero.model import compute_rope_inv_freq

    cases = [
        (32, 10_000.0, 4.0, 64, 32.0, 1.0),        # the tiny experiment of this chapter
        (32, 10_000.0, 2.0, 128, 32.0, 1.0),       # configs/tiny/midtrain.toml
        (128, 1_000_000.0, 4.0, 32768, 32.0, 1.0),  # Qwen3 model card recommends: 32K → 128K
        (64, 50_000.0, 2.5, 48, 16.0, 2.0),        # the custom β in tests/test_model_hf_parity.py
    ]
    try:
        from transformers import Qwen3Config
        from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding
    except ImportError:  # pragma: no cover
        Qwen3Config = None
    print("\n3) Parity check: this script vs zero.model.compute_rope_inv_freq vs Qwen3RotaryEmbedding in transformers")
    for d, theta, s, L, bf, bs in cases:
        mine, _, m = yarn_inv_freq(d, theta, s, L, bf, bs)
        scaling = dict(factor=s, original_max_position_embeddings=L, beta_fast=bf, beta_slow=bs)
        ref, m_ref = compute_rope_inv_freq(d, theta, scaling)
        line = (f"   d={d:<3} θ={theta:<9.0f} s={s:<4} L={L:<6} β=({bf:.0f},{bs:.0f}) | "
                f"max diff vs zero {(mine - ref).abs().max().item():.1e}, mscale diff {abs(m - m_ref):.1e}")
        if Qwen3Config is not None:
            cfg = Qwen3Config(hidden_size=d * 2, num_attention_heads=2, head_dim=d,
                              max_position_embeddings=int(s * L), rope_theta=theta,
                              rope_scaling={"rope_type": "yarn", **scaling})
            hf = Qwen3RotaryEmbedding(cfg)
            line += (f"; max diff vs HF {(mine - hf.inv_freq).abs().max().item():.1e}, "
                     f"mscale diff {abs(m - hf.attention_scaling):.1e}")
        print(line)


def main() -> None:
    show_tiny_table()
    print("\n2) The three zones (β_fast=32, β_slow=1): keep a pair with ≥ 32 turns in training, fully interpolate a pair with < 1 turn")
    show_zones(32, 10_000.0, 4.0, 64, "tiny experiment, 64 → 256")
    show_zones(128, 10_000.0, 8.0, 4096, "θ=10K, 4K → 32K (YaRN only)")
    show_zones(128, 1_000_000.0, 4.0, 32768, "θ=1M, 32K → 128K (Qwen3 method)")
    check_against_zero_and_hf()

    paper = yarn_inv_freq_paper(32, 10_000.0, 4.0, 64)
    code, _, _ = yarn_inv_freq(32, 10_000.0, 4.0, 64)
    rel = ((paper - code).abs() / code).max().item()
    print(f"\n4) Largest relative difference in this chapter's setup, paper formula (ramp linear in turns) vs official code (ramp linear in pair index): {rel:.1%}")
    print("   The two methods differ only in the ramp zone. The official YaRN code, transformers, and zero all use the second one.")

    print("\n5) mscale = 0.1·ln(s) + 1 for different extension factors s")
    for s in [2, 4, 8, 16, 32, 40]:
        print(f"   s={s:>2}: {0.1 * math.log(s) + 1:.4f}")


if __name__ == "__main__":
    main()
