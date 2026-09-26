"""第 15 章 · 极简代码 2：从零实现位置内插（PI）和 YaRN，并与 zero / Hugging Face 对拍

三种把上下文从 L 扩到 s·L 的办法，都只改 RoPE 的转速 ω_i（不动任何可学参数）：
  - 调大基频（ABF）：θ → θ'，所有 ω_i = θ'^(-2i/d) 重新算一遍；
  - 位置内插（PI）：所有 ω_i 都除以 s，相当于把位置 m 压成 m/s；
  - YaRN：按"训练长度内转了多少圈"分三段——
      转得多（高频）的维度对：原样保留（外推）；
      转不满一圈（低频）的维度对：除以 s（内插，同 PI）；
      中间：线性过渡。
    另外把 cos/sin 乘上 mscale = 0.1·ln(s) + 1（等价于把注意力 logits 乘 mscale²，让 softmax 更"尖"）。

运行：uv run python chapters/15-midtraining-long-context/code/02_yarn_from_scratch.py
"""

import math
import sys
from pathlib import Path

import torch

torch.set_num_threads(1)
ROOT = Path(__file__).resolve().parents[3]   # 仓库根目录，用来 import zero


def rope_inv_freq(head_dim: int, theta: float) -> torch.Tensor:
    """ω_i = θ^(-2i/d)，i = 0 .. d/2-1。"""
    return theta ** (-torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)


def pi_inv_freq(head_dim: int, theta: float, s: float) -> torch.Tensor:
    """位置内插：所有维度对一起放慢 s 倍。"""
    return rope_inv_freq(head_dim, theta) / s


def yarn_inv_freq(
    head_dim: int, theta: float, s: float, orig_len: int, beta_fast: float = 32, beta_slow: float = 1
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """返回 (新的 ω_i, 每个维度对"保留原频率"的权重 keep_i, mscale)。

    写法与 YaRN 官方代码、Hugging Face、zero 一致：先算出"在 orig_len 内恰好转 β 圈"的维度编号，
    再在两个编号之间按维度编号做线性斜坡。
    """
    w = rope_inv_freq(head_dim, theta)

    def dim_with_turns(turns: float) -> float:
        # 解 orig_len·ω_i / (2π) = turns，得到维度对编号 i（可以是小数）
        return head_dim * math.log(orig_len / (turns * 2 * math.pi)) / (2 * math.log(theta))

    low = max(math.floor(dim_with_turns(beta_fast)), 0)                # 比它靠前：转了 > β_fast 圈
    high = min(math.ceil(dim_with_turns(beta_slow)), head_dim - 1)     # 比它靠后：转了 < β_slow 圈
    if low == high:
        high += 0.001
    i = torch.arange(head_dim // 2, dtype=torch.float32)
    ramp = ((i - low) / (high - low)).clamp(0, 1)   # 0 = 高频段，1 = 低频段
    keep = 1 - ramp                                  # 保留原频率的权重
    new_w = keep * w + (1 - keep) * w / s            # 高频不动、低频 ÷ s、中间混合
    mscale = 0.1 * math.log(s) + 1.0                 # √(1/t) = 0.1·ln(s) + 1
    return new_w, keep, mscale


def yarn_inv_freq_paper(head_dim, theta, s, orig_len, alpha=1.0, beta=32.0):
    """论文公式 (10)–(13) 的字面写法：斜坡对"圈数 r"线性，而不是对维度编号线性。只用来对比。"""
    w = rope_inv_freq(head_dim, theta)
    r = orig_len * w / (2 * math.pi)                 # 训练长度内转了几圈
    gamma = ((r - alpha) / (beta - alpha)).clamp(0, 1)
    return (1 - gamma) * w / s + gamma * w


def show_tiny_table() -> None:
    d, theta, s, L = 32, 10_000.0, 4.0, 64
    w = rope_inv_freq(d, theta)
    wy, keep, m = yarn_inv_freq(d, theta, s, L)
    print(f"1) 本章小实验的设置：head_dim={d}，θ={theta:.0f}，训练长度 {L}，扩展 s={s:.0f} 倍到 {int(s * L)}")
    print("   i   波长λ   训练内圈数   保留权重   ω(原)     ω(PI)     ω(YaRN)")
    for i in range(d // 2):
        lam = 2 * math.pi / w[i].item()
        print(f"  {i:>2} {lam:>8.1f} {L / lam:>10.2f} {keep[i].item():>10.2f} "
              f"{w[i].item():>9.5f} {w[i].item() / s:>9.5f} {wy[i].item():>9.5f}")
    print(f"   YaRN 的 mscale = 0.1·ln({s:.0f}) + 1 = {m:.4f}（注意力 logits 相当于乘 {m * m:.4f}）")


def show_zones(head_dim: int, theta: float, s: float, L: int, name: str) -> None:
    _, keep, m = yarn_inv_freq(head_dim, theta, s, L)
    n_keep = int((keep == 1).sum())
    n_interp = int((keep == 0).sum())
    print(f"   {name:<34} 原样保留 {n_keep:>2} 对，过渡 {head_dim // 2 - n_keep - n_interp:>2} 对，"
          f"完全内插 {n_interp:>2} 对；mscale {m:.3f}")


def check_against_zero_and_hf() -> None:
    sys.path.insert(0, str(ROOT))
    from zero.model import compute_rope_inv_freq

    cases = [
        (32, 10_000.0, 4.0, 64, 32.0, 1.0),        # 本章小实验
        (32, 10_000.0, 2.0, 128, 32.0, 1.0),       # configs/tiny/midtrain.toml
        (128, 1_000_000.0, 4.0, 32768, 32.0, 1.0),  # Qwen3 模型卡推荐：32K → 128K
        (64, 50_000.0, 2.5, 48, 16.0, 2.0),        # tests/test_model_hf_parity.py 的自定义 β
    ]
    try:
        from transformers import Qwen3Config
        from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding
    except ImportError:  # pragma: no cover
        Qwen3Config = None
    print("\n3) 对拍：本脚本 vs zero.model.compute_rope_inv_freq vs transformers 的 Qwen3RotaryEmbedding")
    for d, theta, s, L, bf, bs in cases:
        mine, _, m = yarn_inv_freq(d, theta, s, L, bf, bs)
        scaling = dict(factor=s, original_max_position_embeddings=L, beta_fast=bf, beta_slow=bs)
        ref, m_ref = compute_rope_inv_freq(d, theta, scaling)
        line = (f"   d={d:<3} θ={theta:<9.0f} s={s:<4} L={L:<6} β=({bf:.0f},{bs:.0f}) | "
                f"与 zero 最大差 {(mine - ref).abs().max().item():.1e}，mscale 差 {abs(m - m_ref):.1e}")
        if Qwen3Config is not None:
            cfg = Qwen3Config(hidden_size=d * 2, num_attention_heads=2, head_dim=d,
                              max_position_embeddings=int(s * L), rope_theta=theta,
                              rope_scaling={"rope_type": "yarn", **scaling})
            hf = Qwen3RotaryEmbedding(cfg)
            line += (f"；与 HF 最大差 {(mine - hf.inv_freq).abs().max().item():.1e}，"
                     f"mscale 差 {abs(m - hf.attention_scaling):.1e}")
        print(line)


def main() -> None:
    show_tiny_table()
    print("\n2) 三段的划分（β_fast=32、β_slow=1，即训练内转满 32 圈以上保留、不满 1 圈完全内插）")
    show_zones(32, 10_000.0, 4.0, 64, "本章小实验 64 → 256")
    show_zones(128, 10_000.0, 8.0, 4096, "θ=1 万，4K → 32K（只用 YaRN）")
    show_zones(128, 1_000_000.0, 4.0, 32768, "θ=100 万，32K → 128K（Qwen3 做法）")
    check_against_zero_and_hf()

    paper = yarn_inv_freq_paper(32, 10_000.0, 4.0, 64)
    code, _, _ = yarn_inv_freq(32, 10_000.0, 4.0, 64)
    rel = ((paper - code).abs() / code).max().item()
    print(f"\n4) 论文公式（斜坡对圈数线性）与官方代码（斜坡对维度编号线性）在本章设置下的最大相对差：{rel:.1%}")
    print("   两种写法只在过渡段不同；YaRN 官方代码、transformers 和 zero 用的都是后者。")

    print("\n5) 不同扩展倍数 s 下的 mscale = 0.1·ln(s) + 1")
    for s in [2, 4, 8, 16, 32, 40]:
        print(f"   s={s:>2}: {0.1 * math.log(s) + 1:.4f}")


if __name__ == "__main__":
    main()
