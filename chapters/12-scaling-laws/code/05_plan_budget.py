"""第 12 章 · 极简代码 5：$5,000 能买多少 token —— 主线模型的预算算术

纯算术，不训练。三条公式（01_flops.py 推过第一条）：
    每 token FLOPs = 6·N_matmul + 12·L·d_attn·T
    卡时           = 每 token FLOPs × token 数 / (单卡峰值 × MFU) / 3600
    费用           = 卡时 × 每卡时单价
反过来：token 数 = 预算 × 峰值 × MFU × 3600 / (单价 × 每 token FLOPs)。

假设（GOAL.md 3.4 与 zero/tools/estimate_cost.py 的默认值；都要在第二步实测后更新）：
    H100 SXM 稠密 BF16 峰值 989.5 TFLOPS（待核实）、$2.5/卡时、8 卡、MFU 0.4。
生产级版本：uv run python -m zero.tools.plan_budget --config configs/main/pretrain.toml --mfu 0.4 --mfu 0.5

运行：uv run python chapters/12-scaling-laws/code/05_plan_budget.py   （不到 1 秒）
"""

PEAK = 989.5e12  # FLOP/s，H100 SXM 稠密 BF16（待核实）
PRICE = 2.5  # 美元 / 卡时
GPUS = 8
BUDGET = 5000.0  # GOAL.md 3.4：预训练约 $5,000
VOCAB = 65536


def shape(dim, n_layers, n_heads=16, n_kv=8, head_dim=128, ffn=3584):
    """主线结构（GQA + 共享 embedding）下的参数与每 token FLOPs。"""
    q, kv = n_heads * head_dim, n_kv * head_dim
    per_layer = dim * q + 2 * dim * kv + q * dim + 3 * dim * ffn
    n_matmul = n_layers * per_layer + dim * VOCAB  # lm_head 的乘法照算
    n_total = n_layers * (per_layer + 2 * dim + 2 * head_dim) + dim + dim * VOCAB
    return n_total, n_matmul, q


def flops_per_token(n_matmul, n_layers, q_dim, T):
    return 6 * n_matmul + 12 * n_layers * q_dim * T


def tokens_for(budget, fpt, mfu):
    return budget * PEAK * mfu * 3600 / (PRICE * fpt)


def main():
    candidates = {  # 名字: (dim, 层数, FFN)
        "主线 0.69B": (1280, 28, 3584),
        "少 4 层 0.60B": (1280, 24, 3584),
        "窄一档 0.51B": (1024, 28, 3072),
    }
    print(f"预算 ${BUDGET:,.0f}，{GPUS}×H100（峰值 {PEAK / 1e12:.1f} TFLOPS，待核实），${PRICE}/卡时 → 共 {BUDGET / PRICE:,.0f} 卡时")
    print(f"{'候选':<12} {'参数':>7} {'序列':>5} {'MFU':>4} | {'token':>6} {'token/参数':>9} {'8 卡天数':>7}")
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
    print(f"\n对照：原计划 500B token（MFU 0.4、序列 4096）要 ${cost_500b:,.0f}，超出 $5K 线 {cost_500b / BUDGET - 1:.0%}。")
    for mfu in (0.4, 0.45, 0.5, 0.55):
        print(f"  MFU {mfu:.2f}：$5K 买 {tokens_for(BUDGET, fpt, mfu) / 1e9:>4.0f}B token；500B 要 "
              f"${500e9 * fpt / (PEAK * mfu) / 3600 * PRICE:,.0f}")


if __name__ == "__main__":
    main()
