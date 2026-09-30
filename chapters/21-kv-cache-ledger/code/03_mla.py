"""第 21 章 · 极简代码 3：多头潜在注意力 MLA（Multi-head Latent Attention）

GQA 靠"少存几个 KV 头"省缓存；MLA 换了一个思路：把所有头的 K、V 一起压成一个低秩潜向量 c_KV，
缓存里只存它（外加一小段带位置的 RoPE key）：

    c_KV = RMSNorm(W_DKV · x)            ← 缓存（kv_lora_rank 个数）
    k_R  = RoPE(W_KR · x)                ← 缓存（qk_rope_head_dim 个数，所有头共享）
    k_i  = [W_UK,i · c_KV ; k_R]         第 i 个头的 key：不带位置的部分 + 共享的位置部分
    v_i  =  W_UV,i · c_KV
    q_i  = [q_i^C ; RoPE(q_i^R)]

推理时还能"吸收"：q_iᵀ k_j = (W_UK,iᵀ q_i^C)ᵀ c_KV,j + q_i^Rᵀ k_R,j，
所以不用把 K、V 还原出来，直接让 query 在潜空间里和缓存做注意力，最后再乘 W_UV 投回去。

forward 的接口和第 10 章 01_tiny_model.py 的 Attention 相同，可以直接换进 TinyLM（04 号脚本就这么做）。
运行：uv run python chapters/21-kv-cache-ledger/code/03_mla.py
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


tiny = _load("tiny_model", CH10 / "01_tiny_model.py")  # 第 10 章的小模型：RMSNorm、RoPE、KVCache


class MLA(nn.Module):
    def __init__(self, dim: int, n_heads: int, kv_lora_rank: int, nope_dim: int, rope_dim: int,
                 v_dim: int, max_len: int = 1024) -> None:
        super().__init__()
        self.h, self.r, self.dn, self.dr, self.dv = n_heads, kv_lora_rank, nope_dim, rope_dim, v_dim
        self.wq = nn.Linear(dim, n_heads * (nope_dim + rope_dim), bias=False)
        self.wkv_a = nn.Linear(dim, kv_lora_rank + rope_dim, bias=False)  # 下投影：x → [c_KV ; k_R]
        self.kv_norm = tiny.RMSNorm(kv_lora_rank)
        self.wkv_b = nn.Linear(kv_lora_rank, n_heads * (nope_dim + v_dim), bias=False)  # [W_UK; W_UV]
        self.wo = nn.Linear(n_heads * v_dim, dim, bias=False)
        cos, sin = tiny.rope_tables(rope_dim, max_len)  # 只给 rope 那一小段维度转位置
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
        c_kv = self.kv_norm(c_kv)  # (B, T, r)        ← 要缓存的潜向量
        k_pe = tiny.apply_rope(k_pe[:, None], cs, sn)  # (B, 1, T, dr)  ← 要缓存的 RoPE key
        c_kv = c_kv[:, None]  # (B, 1, T, r)：借用第 10 章的 KVCache，"1 个头"
        if cache is not None:
            c_kv, k_pe = cache.append(layer, c_kv, k_pe)  # 缓存里只有潜向量和 RoPE key
        S = c_kv.shape[2]
        w = self.wkv_b.weight.view(h, dn + dv, r)
        w_uk, w_uv = w[:, :dn], w[:, dn:]  # (H, dn, r)、(H, dv, r)

        if self.absorb:  # 吸收：query 投进潜空间，直接和缓存点积
            q_lat = torch.einsum("bhtd,hdr->bhtr", q_nope, w_uk)
            att = q_lat @ c_kv.transpose(-2, -1) + q_pe @ k_pe.transpose(-2, -1)  # (B, H, T, S)
        else:  # 显式：先把每个头的 K 还原出来
            k_nope = torch.einsum("bxsr,hdr->bhsd", c_kv, w_uk)  # (B, H, S, dn)
            k = torch.cat([k_nope, k_pe.expand(-1, h, -1, -1)], dim=-1)
            att = torch.cat([q_nope, q_pe], dim=-1) @ k.transpose(-2, -1)
        att = att / math.sqrt(dn + dr)
        i = torch.arange(T)[:, None] + (S - T)
        j = torch.arange(S)[None, :]
        att = att.masked_fill(j > i, float("-inf")).softmax(-1)

        if self.absorb:  # 在潜空间里加权平均，再用 W_UV 投回去
            out = torch.einsum("bhts,bxsr->bhtr", att, c_kv)
            out = torch.einsum("bhtr,hvr->bhtv", out, w_uv)
        else:
            v = torch.einsum("bxsr,hvr->bhsv", c_kv, w_uv)  # (B, H, S, dv)
            out = att @ v
        return self.wo(out.transpose(1, 2).reshape(B, T, h * dv))


def per_token_elems(kind: str, n_heads: int, head_dim: int, n_kv: int = 0, r: int = 0,
                    dr: int = 0) -> int:
    """每层每个位置缓存多少个数。"""
    return r + dr if kind == "MLA" else 2 * n_kv * head_dim


if __name__ == "__main__":
    torch.manual_seed(0)
    mla = MLA(dim=128, n_heads=4, kv_lora_rank=32, nope_dim=32, rope_dim=16, v_dim=32).double()
    x = torch.randn(2, 20, 128, dtype=torch.float64)

    print("1. 吸收路径 = 显式路径（同一组权重、float64）")
    mla.absorb = False
    a = mla(x)
    mla.absorb = True
    b = mla(x)
    print(f"   最大差异 {(a - b).abs().max().item():.1e}")

    print("2. 带缓存逐个喂 = 一次性整段前向")
    cache = tiny.KVCache(1)
    steps = [mla(x[:, :8], cache=cache)] + [mla(x[:, t : t + 1], cache=cache) for t in range(8, 20)]
    print(f"   最大差异 {(torch.cat(steps, 1) - b).abs().max().item():.1e}；缓存里存的形状："
          f"潜向量 {tuple(cache.k[0].shape)}，RoPE key {tuple(cache.v[0].shape)}")

    print("3. 每层每个位置缓存多少个数")
    rows = [("小模型 MHA（4 个 KV 头 × 32）", per_token_elems("MHA", 4, 32, n_kv=4)),
            ("小模型 GQA（2 个 KV 头）", per_token_elems("GQA", 4, 32, n_kv=2)),
            ("小模型 MQA（1 个 KV 头）", per_token_elems("MQA", 4, 32, n_kv=1)),
            ("小模型 MLA（r=32，RoPE 16）", per_token_elems("MLA", 4, 32, r=32, dr=16)),
            ("DeepSeek-V3 若用 MHA（128 头，K 192 + V 128）", 128 * (192 + 128)),
            ("DeepSeek-V3 的 MLA（512 + 64）", per_token_elems("MLA", 128, 128, r=512, dr=64))]
    for name, n in rows:
        print(f"   {name:42} {n:6d}")
    print(f"   DeepSeek-V3：MLA 只有 MHA 的 {576 / (128 * 320):.2%}（约 1/{128 * 320 / 576:.0f}）")
