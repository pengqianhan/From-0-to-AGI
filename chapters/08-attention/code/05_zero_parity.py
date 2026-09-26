"""第 8 章 · 极简代码 5：极简注意力 ↔ 主线模型 zero/model.py 的 Attention 对拍

zero 的 Attention 比极简版多了三样东西：QK-Norm（第 9 章）、RoPE（第 9 章）、GQA（第 10 章）。
把前两样"关掉"（qk_norm=False；cos=1、sin=0 让 RoPE 变成恒等变换），同一组权重的输出必须一致。
再打开 GQA：2 个 K/V 头给 4 个查询头共用，等价于把每个 K/V 头复制一份再做普通多头注意力。

运行：uv run python chapters/08-attention/code/05_zero_parity.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))  # 让脚本能 import 仓库根目录下的 zero

from zero.config import ModelConfig  # noqa: E402
from zero.model import Attention  # noqa: E402

_spec = importlib.util.spec_from_file_location("attn02", HERE / "02_attention_from_scratch.py")
attn02 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(attn02)


def zero_attention(C: int, H: int, n_kv: int) -> Attention:
    cfg = ModelConfig(
        vocab_size=10,
        dim=C,
        n_layers=1,
        n_heads=H,
        n_kv_heads=n_kv,
        ffn_dim=4 * C,
        qk_norm=False,
        max_seq_len=64,
    )
    return Attention(cfg).eval()


def parity() -> dict:
    """返回两项对拍的最大差，以及两种注意力的参数量和 Wk 形状。"""
    torch.manual_seed(0)
    B, T, C, H = 2, 10, 32, 4
    d = C // H
    x = torch.randn(B, T, C)
    cos, sin = torch.ones(T, d), torch.zeros(T, d)  # RoPE 恒等：x·cos + rotate(x)·sin = x

    # ① 普通多头注意力（MHA）：K/V 头数 = 查询头数
    ours = attn02.MultiHeadAttention(C, H).eval()
    prod = zero_attention(C, H, n_kv=H)
    prod.load_state_dict(ours.state_dict())  # 参数名一样：wq、wk、wv、wo
    with torch.no_grad():
        mha_diff = float((ours(x) - prod(x, cos, sin)).abs().max())

    # ② GQA：4 个查询头共享 2 组 K/V
    gqa = zero_attention(C, H, n_kv=2)
    with torch.no_grad():
        y_gqa = gqa(x, cos, sin)
        # 手工版：K/V 各复制一份，头 0、1 用第 0 组，头 2、3 用第 1 组，再做普通注意力
        q = gqa.wq(x).view(B, T, H, d).transpose(1, 2)
        k = gqa.wk(x).view(B, T, 2, d).transpose(1, 2).repeat_interleave(H // 2, dim=1)
        v = gqa.wv(x).view(B, T, 2, d).transpose(1, 2).repeat_interleave(H // 2, dim=1)
        out, _ = attn02.attention(q, k, v, causal=True)
        y_manual = gqa.wo(out.transpose(1, 2).reshape(B, T, C))
    return {
        "mha_diff": mha_diff,
        "gqa_diff": float((y_gqa - y_manual).abs().max()),
        "wk_mha": tuple(prod.wk.weight.shape),
        "wk_gqa": tuple(gqa.wk.weight.shape),
        "n_mha": sum(p.numel() for p in prod.parameters()),
        "n_gqa": sum(p.numel() for p in gqa.parameters()),
    }


def main() -> None:
    r = parity()
    print(
        f"① 极简 MultiHeadAttention vs zero.model.Attention（关掉 QK-Norm 和 RoPE）最大差：{r['mha_diff']:.1e}"
    )
    print(
        f"② GQA 的 Wk 形状 {r['wk_gqa']}（MHA 是 {r['wk_mha']}）：K/V 投影和 KV cache 都只有一半大"
    )
    print(f'   zero 的 GQA vs 手工"复制 K/V 再做多头注意力" 最大差：{r["gqa_diff"]:.1e}')
    print(f"   注意力参数量：MHA {r['n_mha']}，GQA {r['n_gqa']}")


if __name__ == "__main__":
    main()
