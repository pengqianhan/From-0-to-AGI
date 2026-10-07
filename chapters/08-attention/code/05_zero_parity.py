"""Chapter 8 · Minimal code 5: parity check,
minimal attention ↔ Attention in the main-line model zero/model.py

The zero Attention has three more parts than the minimal version: QK-Norm (Chapter 9), RoPE (Chapter 9),
and GQA (Chapter 10). Turn off the first two (qk_norm=False; cos=1, sin=0 make RoPE the identity).
Then the same weights must give the same output.
Then turn on GQA: 4 query heads share 2 K/V heads. This is the same as a copy of each K/V head,
followed by normal multi-head attention.

Run: uv run python chapters/08-attention/code/05_zero_parity.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))  # lets the script import zero from the repository root

from zero.config import ModelConfig  # noqa: E402
from zero.model import Attention  # noqa: E402

# One thread is fastest for small tensors. Many jobs share the CPU of the build machine.
# You can remove this line on your computer.
torch.set_num_threads(1)

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
    """Return the max differences of the two parity checks.

    Also return the parameter counts and the Wk shapes of the two attention types.
    """
    torch.manual_seed(0)
    B, T, C, H = 2, 10, 32, 4
    d = C // H
    x = torch.randn(B, T, C)
    cos, sin = torch.ones(T, d), torch.zeros(T, d)  # RoPE becomes the identity: x·cos + rotate(x)·sin = x

    # ① Normal multi-head attention (MHA): number of K/V heads = number of query heads
    ours = attn02.MultiHeadAttention(C, H).eval()
    prod = zero_attention(C, H, n_kv=H)
    prod.load_state_dict(ours.state_dict())  # the parameter names are the same: wq, wk, wv, wo
    with torch.no_grad():
        mha_diff = float((ours(x) - prod(x, cos, sin)).abs().max())

    # ② GQA: 4 query heads share 2 K/V groups
    gqa = zero_attention(C, H, n_kv=2)
    with torch.no_grad():
        y_gqa = gqa(x, cos, sin)
        # Manual version: copy K and V once. Heads 0 and 1 use group 0, heads 2 and 3 use group 1.
        # Then do normal attention
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
        f"① Minimal MultiHeadAttention vs zero.model.Attention (QK-Norm and RoPE off), max difference: {r['mha_diff']:.1e}"
    )
    print(
        f"② GQA Wk shape {r['wk_gqa']} (MHA: {r['wk_mha']}): the K/V projections and the KV cache are half the size"
    )
    print(f'   zero GQA vs manual "copy K/V, then multi-head attention", max difference: {r["gqa_diff"]:.1e}')
    print(f"   Attention parameters: MHA {r['n_mha']}, GQA {r['n_gqa']}")


if __name__ == "__main__":
    main()
