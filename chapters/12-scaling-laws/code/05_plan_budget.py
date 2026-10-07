"""Chapter 12 · Minimal code 5: how many tokens $5,000 can buy. Budget arithmetic for the main-line model.

Only arithmetic, no training. Three formulas (01_flops.py derives the first one):
    FLOPs per token = 6·N_matmul + 12·L·d_attn·T
    GPU-hours       = FLOPs per token × number of tokens / (peak of one GPU × MFU) / 3600
    cost            = GPU-hours × price per GPU-hour
The inverse: number of tokens = budget × peak × MFU × 3600 / (price × FLOPs per token).

Assumptions (the default values in GOAL.md 3.4 and zero/tools/estimate_cost.py; update all of them
after the measurements in step 2):
    H100 SXM dense BF16 peak 989.5 TFLOPS (to be verified), $2.5/GPU-hour, 8 GPUs, MFU 0.4.
Production version: uv run python -m zero.tools.plan_budget --config configs/main/pretrain.toml --mfu 0.4 --mfu 0.5

Run: uv run python chapters/12-scaling-laws/code/05_plan_budget.py   (less than 1 second)
"""

PEAK = 989.5e12  # FLOP/s, H100 SXM dense BF16 (to be verified)
PRICE = 2.5  # US dollars per GPU-hour
GPUS = 8
BUDGET = 5000.0  # GOAL.md 3.4: pretraining costs about $5,000
VOCAB = 65536


def shape(dim, n_layers, n_heads=16, n_kv=8, head_dim=128, ffn=3584):
    """Parameters and FLOPs per token for the main-line structure (GQA + shared embedding)."""
    q, kv = n_heads * head_dim, n_kv * head_dim
    per_layer = dim * q + 2 * dim * kv + q * dim + 3 * dim * ffn
    n_matmul = n_layers * per_layer + dim * VOCAB  # the matrix product of lm_head still counts
    n_total = n_layers * (per_layer + 2 * dim + 2 * head_dim) + dim + dim * VOCAB
    return n_total, n_matmul, q


def flops_per_token(n_matmul, n_layers, q_dim, T):
    return 6 * n_matmul + 12 * n_layers * q_dim * T


def tokens_for(budget, fpt, mfu):
    return budget * PEAK * mfu * 3600 / (PRICE * fpt)


def main():
    candidates = {  # name: (dim, number of layers, FFN)
        "main 0.69B": (1280, 28, 3584),
        "24L 0.60B": (1280, 24, 3584),
        "narrow 0.51B": (1024, 28, 3072),
    }
    print(f"Budget ${BUDGET:,.0f}, {GPUS}×H100 (peak {PEAK / 1e12:.1f} TFLOPS, to be verified), ${PRICE}/GPU-hour → {BUDGET / PRICE:,.0f} GPU-hours in total")
    print(f"{'candidate':<12} {'params':>7} {'seq':>5} {'MFU':>4} | {'token':>6} {'tok/param':>9} {'days':>7}")
    for name, (dim, L, ffn) in candidates.items():
        n_total, n_matmul, q = shape(dim, L, ffn=ffn)
        for T, mfu in [(4096, 0.4), (4096, 0.5), (2048, 0.4)]:
            fpt = flops_per_token(n_matmul, L, q, T)
            D = tokens_for(BUDGET, fpt, mfu)
            days = BUDGET / PRICE / GPUS / 24
            print(f"{name:<12} {n_total / 1e9:>6.2f}B {T:>5} {mfu:>4.1f} | {D / 1e9:>5.0f}B {D / n_total:>9.0f} {days:>7.1f}")
    n_total, n_matmul, q = shape(1280, 28)
    fpt = flops_per_token(n_matmul, 28, q, 4096)
    cost_500b = 500e9 * fpt / (PEAK * 0.4) / 3600 * PRICE
    print(f"\nCompare: the original plan of 500B tokens (MFU 0.4, sequence 4096) costs ${cost_500b:,.0f}, {cost_500b / BUDGET - 1:.0%} over the $5K limit.")
    for mfu in (0.4, 0.45, 0.5, 0.55):
        print(f"  MFU {mfu:.2f}: $5K buys {tokens_for(BUDGET, fpt, mfu) / 1e9:>4.0f}B tokens; 500B costs "
              f"${500e9 * fpt / (PEAK * mfu) / 3600 * PRICE:,.0f}")


if __name__ == "__main__":
    main()
