"""第 12 章 · 极简代码 1：C ≈ 6ND 是怎么来的 —— 用 PyTorch 自己数一遍浮点运算

训练一个 token 的算力 = 前向 + 反向：
  - 前向：每个参数参与一次乘加（2 FLOPs）→ 2N；
  - 反向：对输入求梯度一次、对权重求梯度一次，各 2N → 4N（第 4 章：y = Wx 的反向要算 Wᵀg 和 g xᵀ 两个矩阵乘）；
  - 合计 6N；再加上注意力里 QKᵀ 和 AV 这两个"没有参数"的矩阵乘：每层每 token 12·d_attn·T。
所以  每 token FLOPs = 6·N_matmul + 12·L·d_attn·T，   总算力 C ≈ 6ND（注意力项在短序列时很小）。

这里用 torch.utils.flop_counter.FlopCounterMode 把第 9 章小模型一次前向 + 反向的矩阵乘 FLOPs
真的数出来，和公式对比；再用同一个公式算主线模型（configs/main/pretrain.toml 的形状）。

运行：uv run python chapters/12-scaling-laws/code/01_flops.py   （几秒）
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
    """参与矩阵乘的参数：每层 attention 4·d² + SwiGLU 3·d·ffn，再加 lm_head（d·V）。"""
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
    print("① 实测 vs 公式（第 9 章 TinyTransformer，字节级词表 256）")
    print(f"{'dim':>4} {'层':>2} {'T':>5} | {'N_matmul':>9} {'6N':>10} {'注意力项':>9} {'公式合计':>10} {'实测':>10}")
    for dim, L, T in [(64, 2, 64), (128, 4, 128), (128, 4, 512)]:
        cfg = tt.Config(dim=dim, n_layers=L, n_heads=dim // 16, ffn_dim=16 * round(8 / 3 * dim / 16), seq_len=T)
        n = matmul_params(cfg)
        attn = 12 * L * dim * T
        f = formula_flops_per_token(n, L, dim, T)
        m = measured_flops_per_token(cfg)
        print(f"{dim:>4} {L:>2} {T:>5} | {n:>9,} {6 * n:>10,} {attn:>9,} {f:>10,.0f} {m:>10,.0f}")
    print("  → 两列完全相等：FlopCounterMode 只数矩阵乘，RMSNorm、softmax、SiLU 这些逐元素运算不到 1%，公式也不算它们。")

    print("\n② 主线模型（configs/main/pretrain.toml 的形状：dim 1280、28 层、16 头 × 128、FFN 3584、词表 65,536）")
    d, L, q_dim, kv_dim, ffn, V, T = 1280, 28, 16 * 128, 8 * 128, 3584, 65536, 4096
    per_layer = d * q_dim + 2 * d * kv_dim + q_dim * d + 3 * d * ffn  # GQA：K、V 只有 8 个头
    n_matmul = L * per_layer + d * V  # lm_head 与 embedding 共享权重，但输出投影的乘法照算
    n_total = L * (per_layer + 2 * d + 2 * 128) + d + d * V  # 加上 RMSNorm、QK-Norm 的权重
    attn = 12 * L * q_dim * T
    fpt = 6 * n_matmul + attn
    print(f"  总参数 N = {n_total / 1e6:.1f}M，参与矩阵乘的 N_matmul = {n_matmul / 1e6:.1f}M")
    print(f"  每 token：6·N_matmul = {6 * n_matmul:.4g}，注意力项 12·L·d·T = {attn:.4g}（占 {attn / fpt:.1%}）")
    print(f"  合计 {fpt:.4g} FLOPs/token；粗算 6·N_total = {6 * n_total:.4g}（低估 {1 - 6 * n_total / fpt:.1%}）")
    D = 400e9
    print(f"  训练 {D / 1e9:.0f}B token：C = {fpt * D:.3g} FLOPs（粗算 6ND = {6 * n_total * D:.3g}）")


if __name__ == "__main__":
    main()
