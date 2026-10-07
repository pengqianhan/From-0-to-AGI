"""Chapter 21 · Minimal code 3: Multi-head Latent Attention (MLA).

GQA saves cache because it stores fewer KV heads. MLA uses a different method. It compresses
the K and V of all heads together into one low-rank latent vector c_KV. The cache stores only
c_KV, plus a small RoPE key that carries the position:

    c_KV = RMSNorm(W_DKV · x)            ← cached (kv_lora_rank numbers)
    k_R  = RoPE(W_KR · x)                ← cached (qk_rope_head_dim numbers, shared by all heads)
    k_i  = [W_UK,i · c_KV ; k_R]         key of head i: part without position + shared position part
    v_i  =  W_UV,i · c_KV
    q_i  = [q_i^C ; RoPE(q_i^R)]

At inference, we can also "absorb": q_iᵀ k_j = (W_UK,iᵀ q_i^C)ᵀ c_KV,j + q_i^Rᵀ k_R,j.
Thus we do not reconstruct K and V. The query attends to the cache directly in the latent space.
At the end, a multiplication by W_UV projects the result back.

The forward interface is the same as Attention in 01_tiny_model.py of Chapter 10.
Thus MLA can replace it in TinyLM directly (script 04 does this).
Run: uv run python chapters/21-kv-cache-ledger/code/03_mla.py
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn

torch.set_num_threads(1)
CH10 = Path(__file__).resolve().parents[2] / "10-inference" / "code"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


tiny = _load("tiny_model", CH10 / "01_tiny_model.py")  # the small model of Chapter 10: RMSNorm, RoPE, KVCache


class MLA(nn.Module):
    def __init__(self, dim: int, n_heads: int, kv_lora_rank: int, nope_dim: int, rope_dim: int,
                 v_dim: int, max_len: int = 1024) -> None:
        super().__init__()
        self.h, self.r, self.dn, self.dr, self.dv = n_heads, kv_lora_rank, nope_dim, rope_dim, v_dim
        self.wq = nn.Linear(dim, n_heads * (nope_dim + rope_dim), bias=False)
        self.wkv_a = nn.Linear(dim, kv_lora_rank + rope_dim, bias=False)  # down-projection: x → [c_KV ; k_R]
        self.kv_norm = tiny.RMSNorm(kv_lora_rank)
        self.wkv_b = nn.Linear(kv_lora_rank, n_heads * (nope_dim + v_dim), bias=False)  # [W_UK; W_UV]
        self.wo = nn.Linear(n_heads * v_dim, dim, bias=False)
        cos, sin = tiny.rope_tables(rope_dim, max_len)  # rotate only the small rope part of the dimensions
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.absorb = False

    def forward(self, x, cos=None, sin=None, cache=None, layer: int = 0):
        B, T, _ = x.shape
        h, r, dn, dr, dv = self.h, self.r, self.dn, self.dr, self.dv
        start = 0 if cache is None or cache.k[layer] is None else cache.k[layer].shape[2]
        cs, sn = self.cos[start : start + T], self.sin[start : start + T]

        q = self.wq(x).view(B, T, h, dn + dr).transpose(1, 2)  # (B, H, T, dn+dr)
        q_nope, q_pe = q.split([dn, dr], dim=-1)
        q_pe = tiny.apply_rope(q_pe, cs, sn)
        c_kv, k_pe = self.wkv_a(x).split([r, dr], dim=-1)
        c_kv = self.kv_norm(c_kv)  # (B, T, r)        ← the latent vector to cache
        k_pe = tiny.apply_rope(k_pe[:, None], cs, sn)  # (B, 1, T, dr)  ← the RoPE key to cache
        c_kv = c_kv[:, None]  # (B, 1, T, r): use the KVCache of Chapter 10 with "1 head"
        if cache is not None:
            c_kv, k_pe = cache.append(layer, c_kv, k_pe)  # the cache holds only the latent vector and the RoPE key
        S = c_kv.shape[2]
        w = self.wkv_b.weight.view(h, dn + dv, r)
        w_uk, w_uv = w[:, :dn], w[:, dn:]  # (H, dn, r), (H, dv, r)

        if self.absorb:  # absorb: project the query into the latent space; dot product with the cache
            q_lat = torch.einsum("bhtd,hdr->bhtr", q_nope, w_uk)
            att = q_lat @ c_kv.transpose(-2, -1) + q_pe @ k_pe.transpose(-2, -1)  # (B, H, T, S)
        else:  # explicit: first reconstruct the K of each head
            k_nope = torch.einsum("bxsr,hdr->bhsd", c_kv, w_uk)  # (B, H, S, dn)
            k = torch.cat([k_nope, k_pe.expand(-1, h, -1, -1)], dim=-1)
            att = torch.cat([q_nope, q_pe], dim=-1) @ k.transpose(-2, -1)
        att = att / math.sqrt(dn + dr)
        i = torch.arange(T)[:, None] + (S - T)
        j = torch.arange(S)[None, :]
        att = att.masked_fill(j > i, float("-inf")).softmax(-1)

        if self.absorb:  # weighted average in the latent space, then project back with W_UV
            out = torch.einsum("bhts,bxsr->bhtr", att, c_kv)
            out = torch.einsum("bhtr,hvr->bhtv", out, w_uv)
        else:
            v = torch.einsum("bxsr,hvr->bhsv", c_kv, w_uv)  # (B, H, S, dv)
            out = att @ v
        return self.wo(out.transpose(1, 2).reshape(B, T, h * dv))


def per_token_elems(kind: str, n_heads: int, head_dim: int, n_kv: int = 0, r: int = 0,
                    dr: int = 0) -> int:
    """Numbers cached per layer per position."""
    return r + dr if kind == "MLA" else 2 * n_kv * head_dim


if __name__ == "__main__":
    torch.manual_seed(0)
    mla = MLA(dim=128, n_heads=4, kv_lora_rank=32, nope_dim=32, rope_dim=16, v_dim=32).double()
    x = torch.randn(2, 20, 128, dtype=torch.float64)

    print("1. Absorbed path = explicit path (same weights, float64)")
    mla.absorb = False
    a = mla(x)
    mla.absorb = True
    b = mla(x)
    print(f"   max difference {(a - b).abs().max().item():.1e}")

    print("2. Token by token with a cache = one forward pass over the full sequence")
    cache = tiny.KVCache(1)
    steps = [mla(x[:, :8], cache=cache)] + [mla(x[:, t : t + 1], cache=cache) for t in range(8, 20)]
    print(f"   max difference {(torch.cat(steps, 1) - b).abs().max().item():.1e}; shapes in the cache: "
          f"latent {tuple(cache.k[0].shape)}, RoPE key {tuple(cache.v[0].shape)}")

    print("3. Numbers cached per layer per position")
    rows = [("Small model MHA (4 KV heads × 32)", per_token_elems("MHA", 4, 32, n_kv=4)),
            ("Small model GQA (2 KV heads)", per_token_elems("GQA", 4, 32, n_kv=2)),
            ("Small model MQA (1 KV head)", per_token_elems("MQA", 4, 32, n_kv=1)),
            ("Small model MLA (r=32, RoPE 16)", per_token_elems("MLA", 4, 32, r=32, dr=16)),
            ("DeepSeek-V3 if MHA (128 heads, K192+V128)", 128 * (192 + 128)),
            ("DeepSeek-V3 MLA (512 + 64)", per_token_elems("MLA", 128, 128, r=512, dr=64))]
    for name, n in rows:
        print(f"   {name:42} {n:6d}")
    print(f"   DeepSeek-V3: MLA is only {576 / (128 * 320):.2%} of MHA (about 1/{128 * 320 / 576:.0f})")
