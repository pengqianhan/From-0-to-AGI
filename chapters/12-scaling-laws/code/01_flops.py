"""Chapter 12 · Minimal code 1: where C ≈ 6ND comes from. Count the floating-point operations with PyTorch.

The compute to train on one token = forward pass + backward pass:
  - Forward pass: each parameter does one multiply-add (2 FLOPs) → 2N.
  - Backward pass: one gradient for the input and one gradient for the weights, 2N each → 4N
    (Chapter 4: the backward pass of y = Wx calculates two matrix products, Wᵀg and g xᵀ).
  - Total: 6N. Attention also has two matrix products "without parameters", QKᵀ and AV:
    12·d_attn·T for each layer and each token.
Thus  FLOPs per token = 6·N_matmul + 12·L·d_attn·T,   total compute C ≈ 6ND
(the attention term is small when the sequences are short).

This script uses torch.utils.flop_counter.FlopCounterMode to count the matrix-product FLOPs of one
forward and backward pass of the small model from Chapter 9. It compares the count with the formula.
Then it uses the same formula for the main-line model (the shape in configs/main/pretrain.toml).

Run: uv run python chapters/12-scaling-laws/code/01_flops.py   (a few seconds)
"""

import importlib.util
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.flop_counter import FlopCounterMode

ROOT = Path(__file__).resolve().parents[3]
_spec = importlib.util.spec_from_file_location(
    "tiny_tf", ROOT / "chapters/09-modern-transformer/code/02_tiny_transformer.py"
)
tt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tt)


def matmul_params(cfg) -> int:
    """Parameters in matrix products: 4·d² (attention) + 3·d·ffn (SwiGLU) in each layer, plus lm_head (d·V)."""
    d = cfg.dim
    return cfg.n_layers * (4 * d * d + 3 * d * cfg.ffn_dim) + d * cfg.vocab_size


def formula_flops_per_token(n_matmul: int, n_layers: int, d_attn: int, T: int) -> float:
    return 6 * n_matmul + 12 * n_layers * d_attn * T


def measured_flops_per_token(cfg, batch=2) -> float:
    model = tt.TinyTransformer(cfg)
    x = torch.randint(0, cfg.vocab_size, (batch, cfg.seq_len))
    counter = FlopCounterMode(display=False)
    with counter:
        loss = F.cross_entropy(model(x).flatten(0, 1), x.flatten())
        loss.backward()
    return counter.get_total_flops() / (batch * cfg.seq_len)


def main():
    torch.set_num_threads(1)
    print("① Measured vs formula (TinyTransformer from Chapter 9, byte-level vocabulary of 256)")
    print(f"{'dim':>4} {'L':>2} {'T':>5} | {'N_matmul':>9} {'6N':>10} {'attn term':>9} {'formula':>10} {'measured':>10}")
    for dim, L, T in [(64, 2, 64), (128, 4, 128), (128, 4, 512)]:
        cfg = tt.Config(dim=dim, n_layers=L, n_heads=dim // 16, ffn_dim=16 * round(8 / 3 * dim / 16), seq_len=T)
        n = matmul_params(cfg)
        attn = 12 * L * dim * T
        f = formula_flops_per_token(n, L, dim, T)
        m = measured_flops_per_token(cfg)
        print(f"{dim:>4} {L:>2} {T:>5} | {n:>9,} {6 * n:>10,} {attn:>9,} {f:>10,.0f} {m:>10,.0f}")
    print("  → The last two columns are equal. FlopCounterMode counts only matrix products. "
          "Element-wise operations (RMSNorm, softmax, SiLU) are less than 1%, and the formula does not count them.")

    print("\n② Main-line model (shape from configs/main/pretrain.toml: dim 1280, 28 layers, 16 heads × 128, FFN 3584, "
          "vocabulary 65,536)")
    d, L, q_dim, kv_dim, ffn, V, T = 1280, 28, 16 * 128, 8 * 128, 3584, 65536, 4096
    per_layer = d * q_dim + 2 * d * kv_dim + q_dim * d + 3 * d * ffn  # GQA: K and V have only 8 heads
    n_matmul = L * per_layer + d * V  # lm_head shares its weights with the embedding, but its matrix product still counts
    n_total = L * (per_layer + 2 * d + 2 * 128) + d + d * V  # add the weights of RMSNorm and QK-Norm
    attn = 12 * L * q_dim * T
    fpt = 6 * n_matmul + attn
    print(f"  Total parameters N = {n_total / 1e6:.1f}M, parameters in matrix products N_matmul = {n_matmul / 1e6:.1f}M")
    print(f"  Per token: 6·N_matmul = {6 * n_matmul:.4g}, attention term 12·L·d·T = {attn:.4g} ({attn / fpt:.1%} of the total)")
    print(f"  Total {fpt:.4g} FLOPs/token; rough estimate 6·N_total = {6 * n_total:.4g} (too low by {1 - 6 * n_total / fpt:.1%})")
    D = 400e9
    print(f"  Train on {D / 1e9:.0f}B tokens: C = {fpt * D:.3g} FLOPs (rough estimate 6ND = {6 * n_total * D:.3g})")


if __name__ == "__main__":
    main()
