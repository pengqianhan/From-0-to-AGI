"""多头潜在注意力 MLA（Multi-head Latent Attention，对应第 21 章）。实验模块，不用于主线模型。

MLA 由 DeepSeek-V2（arXiv:2405.04434）提出，DeepSeek-V3、Kimi K2、GLM-5、Mistral Large 3 采用。
和 GQA "少存几个头" 不同，MLA 把所有头的 K、V **联合压缩**成一个低秩潜向量 c_KV，缓存里只存它：

    c_KV = RMSNorm(W_DKV · h)              # (kv_lora_rank,)      ← 缓存
    k_R  = RoPE(W_KR · h)                  # (qk_rope_head_dim,)  ← 缓存（所有头共享一个）
    k_i  = [W_UK,i · c_KV ; k_R]           # 第 i 个头的 key = 不带位置的部分 + 共享的 RoPE 部分
    v_i  =  W_UV,i · c_KV
    q_i  = [W_UQ,i · h ; RoPE(W_QR,i · h)]

每层每个位置只存 kv_lora_rank + qk_rope_head_dim 个数（DeepSeek-V3：512 + 64 = 576），
而同样 128 个头的 MHA 要存 2 × 128 × 128 = 32,768 个。

**为什么 RoPE 要"解耦"**：如果把 RoPE 直接转在 k_i = W_UK,i · c_KV 上，位置相关的旋转矩阵就夹在
W_UQ 和 W_UK 中间，下面的"吸收"技巧就不成立了。所以让一小段维度（qk_rope_head_dim）专门带位置，
其余维度不带位置。

**吸收（absorb）技巧**：推理时不必把 K、V 从 c_KV 还原出来。因为
    q_iᵀ k_j = (W_UK,iᵀ q_i^C)ᵀ c_KV,j + q_i^Rᵀ k_R,j
    Σ_j p_ij v_j = W_UV,i (Σ_j p_ij c_KV,j)
先把 query 投影进潜空间，直接和缓存里的潜向量做注意力，最后再用 W_UV 投回去。
这样每个头都在 576 维上做点积——形式上就是"所有头共享一组 K/V"的 MQA（GLM-5 报告称之为
MLA 的 MQA 模式）。本模块两条路径都实现：`absorb=False` 显式还原 K/V（训练/prefill 用），
`absorb=True` 走吸收路径（decode 用）；`tests/test_arch_mla.py` 验证两者一致、
带缓存与不带缓存生成完全一致。

接口与 `zero.model.Attention` 相同：`forward(x, cos, sin, kv_cache, layer_idx, start_pos)`，
可以直接替换进 `zero.model.Transformer`（见 `mla_transformer`）。区别是 MLA 只对
qk_rope_head_dim 维做 RoPE，所以用自己的 RoPE 表，忽略传进来的 cos/sin。

行业实现：DeepSeek 开源的 FlashMLA（GPU decode kernel）、vLLM / SGLang 的 MLA 后端。
本文件只追求可读和正确：CUDA 上前向 / 反向与 CPU 对拍、潜向量缓存生成与重算逐字相同已在 RTX 3090 上验证
（2026-10，见 runs/2026-10-01-gpu0-check/），尚未在 GPU 上验证性能。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from zero.config import ModelConfig
from zero.model import RMSNorm, RotaryEmbedding, Transformer, apply_rope


@dataclass
class MLAConfig:
    """MLA 超参。字段名与 Hugging Face DeepSeek-V3 的 config.json 一致。"""

    dim: int
    n_heads: int
    kv_lora_rank: int  # 潜向量维度 d_c（DeepSeek-V3：512）
    qk_nope_head_dim: int  # 每个头 key/query 中不带位置的部分（DeepSeek-V3：128）
    qk_rope_head_dim: int  # 带 RoPE 的部分，key 这一段所有头共享（DeepSeek-V3：64）
    v_head_dim: int  # 每个头 value 的维度（DeepSeek-V3：128）
    q_lora_rank: int | None = None  # query 也压缩（省训练激活，不省 KV cache）；None 表示不压缩
    max_seq_len: int = 2048
    rope_theta: float = 10000.0
    norm_eps: float = 1e-6

    @property
    def qk_head_dim(self) -> int:
        return self.qk_nope_head_dim + self.qk_rope_head_dim

    @property
    def cache_dim(self) -> int:
        """每层每个位置缓存多少个数。"""
        return self.kv_lora_rank + self.qk_rope_head_dim

    @classmethod
    def from_model_config(
        cls,
        cfg: ModelConfig,
        kv_lora_rank: int,
        qk_rope_head_dim: int | None = None,
        q_lora_rank: int | None = None,
    ) -> MLAConfig:
        """沿用一个 GQA 配置的宽度、头数和 head_dim，只把注意力换成 MLA。"""
        assert cfg.head_dim is not None
        rope = qk_rope_head_dim if qk_rope_head_dim is not None else cfg.head_dim // 2
        return cls(
            dim=cfg.dim,
            n_heads=cfg.n_heads,
            kv_lora_rank=kv_lora_rank,
            qk_nope_head_dim=cfg.head_dim,
            qk_rope_head_dim=rope,
            v_head_dim=cfg.head_dim,
            q_lora_rank=q_lora_rank,
            max_seq_len=cfg.max_seq_len,
            rope_theta=cfg.rope_theta,
            norm_eps=cfg.norm_eps,
        )


class MLACache:
    """MLA 的缓存：每层只存潜向量 c_KV 和共享的 RoPE key，预分配（同 zero.kv_cache.KVCache）。

    形状：latent (n_layers, batch, max_seq_len, kv_lora_rank)，k_rope (n_layers, batch, max_seq_len, rope_dim)。
    """

    def __init__(
        self,
        n_layers: int,
        batch_size: int,
        max_seq_len: int,
        kv_lora_rank: int,
        rope_dim: int,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.latent = torch.zeros(
            (n_layers, batch_size, max_seq_len, kv_lora_rank), device=device, dtype=dtype
        )
        self.k_rope = torch.zeros(
            (n_layers, batch_size, max_seq_len, rope_dim), device=device, dtype=dtype
        )
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size

    @classmethod
    def from_config(
        cls,
        cfg: MLAConfig,
        n_layers: int,
        batch_size: int,
        max_seq_len: int | None = None,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> MLACache:
        return cls(
            n_layers,
            batch_size,
            max_seq_len or cfg.max_seq_len,
            cfg.kv_lora_rank,
            cfg.qk_rope_head_dim,
            device,
            dtype,
        )

    def update(
        self, layer_idx: int, start_pos: int, c_kv: torch.Tensor, k_rope: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """写入 [start_pos, start_pos+T)，返回这一层 0..start_pos+T 的全部潜向量与 RoPE key。"""
        bsz, t, _ = c_kv.shape
        end = start_pos + t
        if end > self.max_seq_len:
            raise ValueError(f"MLA cache 溢出：需要 {end} 个位置，只预分配了 {self.max_seq_len}")
        self.latent[layer_idx, :bsz, start_pos:end] = c_kv.to(self.latent.dtype)
        self.k_rope[layer_idx, :bsz, start_pos:end] = k_rope.to(self.k_rope.dtype)
        return self.latent[layer_idx, :bsz, :end], self.k_rope[layer_idx, :bsz, :end]

    def nbytes(self) -> int:
        return (
            self.latent.numel() * self.latent.element_size()
            + self.k_rope.numel() * self.k_rope.element_size()
        )


def _causal_mask(seqlen: int, kv_len: int, device: torch.device) -> torch.Tensor | None:
    """第 i 个新 token 能看见位置 <= past + i（past = kv_len - seqlen）。seqlen=1 时不需要掩码。"""
    if seqlen == 1:
        return None
    past = kv_len - seqlen
    i = torch.arange(seqlen, device=device)[:, None]
    j = torch.arange(kv_len, device=device)[None, :]
    return j <= (past + i)


class MLAAttention(nn.Module):
    """多头潜在注意力。参数命名参照 DeepSeek-V3（wq_a/wq_b、wkv_a、wkv_b、wo）。"""

    def __init__(self, cfg: MLAConfig, absorb: bool | None = None) -> None:
        super().__init__()
        self.cfg = cfg
        #: None = 自动：有缓存（推理）时走吸收路径，没有缓存（训练）时显式还原 K/V
        self.absorb = absorb
        h, dq = cfg.n_heads, cfg.qk_head_dim
        if cfg.q_lora_rank:
            self.wq_a = nn.Linear(cfg.dim, cfg.q_lora_rank, bias=False)
            self.q_norm = RMSNorm(cfg.q_lora_rank, cfg.norm_eps)
            self.wq_b = nn.Linear(cfg.q_lora_rank, h * dq, bias=False)
        else:
            self.wq = nn.Linear(cfg.dim, h * dq, bias=False)
        # 下投影：一次得到潜向量 c_KV 和共享的 RoPE key（DeepSeek 里叫 kv_a_proj_with_mqa）
        self.wkv_a = nn.Linear(cfg.dim, cfg.kv_lora_rank + cfg.qk_rope_head_dim, bias=False)
        self.kv_norm = RMSNorm(cfg.kv_lora_rank, cfg.norm_eps)
        # 上投影：潜向量 → 每个头的 k_nope 与 v（W_UK 与 W_UV 拼在一起）
        self.wkv_b = nn.Linear(
            cfg.kv_lora_rank, h * (cfg.qk_nope_head_dim + cfg.v_head_dim), bias=False
        )
        self.wo = nn.Linear(h * cfg.v_head_dim, cfg.dim, bias=False)
        self.rope = RotaryEmbedding(cfg.qk_rope_head_dim, cfg.max_seq_len, cfg.rope_theta)
        self.scale = 1.0 / math.sqrt(dq)

    def _query(self, x: torch.Tensor) -> torch.Tensor:
        if self.cfg.q_lora_rank:
            return self.wq_b(self.q_norm(self.wq_a(x)))
        return self.wq(x)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor | None = None,
        sin: torch.Tensor | None = None,
        kv_cache: MLACache | None = None,
        layer_idx: int = 0,
        start_pos: int = 0,
    ) -> torch.Tensor:
        del cos, sin  # MLA 只对 rope 子维度做旋转，用自己的 RoPE 表
        c = self.cfg
        bsz, seqlen, _ = x.shape
        h, dn, dr, dv = c.n_heads, c.qk_nope_head_dim, c.qk_rope_head_dim, c.v_head_dim
        cos_r, sin_r = self.rope(start_pos, seqlen)

        q = self._query(x).view(bsz, seqlen, h, dn + dr).transpose(1, 2)  # (B, H, T, dn+dr)
        q_nope, q_pe = q.split([dn, dr], dim=-1)
        q_pe = apply_rope(q_pe, cos_r, sin_r)

        kv = self.wkv_a(x)
        c_kv, k_pe = kv.split([c.kv_lora_rank, dr], dim=-1)
        c_kv = self.kv_norm(c_kv)  # (B, T, r)
        k_pe = apply_rope(k_pe[:, None], cos_r, sin_r)[:, 0]  # (B, T, dr)，所有头共享

        if kv_cache is not None:
            c_kv, k_pe = kv_cache.update(layer_idx, start_pos, c_kv, k_pe)
            c_kv, k_pe = c_kv.to(x.dtype), k_pe.to(x.dtype)
        kv_len = c_kv.shape[1]
        mask = _causal_mask(seqlen, kv_len, x.device)

        absorb = self.absorb if self.absorb is not None else kv_cache is not None
        if absorb:
            out = self._absorbed(q_nope, q_pe, c_kv, k_pe, mask)
        else:
            out = self._naive(q_nope, q_pe, c_kv, k_pe, mask)
        return self.wo(out.transpose(1, 2).reshape(bsz, seqlen, h * dv))

    def _naive(self, q_nope, q_pe, c_kv, k_pe, mask) -> torch.Tensor:
        """显式还原每个头的 K、V，再做标准注意力（训练和 prefill 时算力更划算）。"""
        c = self.cfg
        bsz, s, _ = c_kv.shape
        kv = self.wkv_b(c_kv).view(bsz, s, c.n_heads, c.qk_nope_head_dim + c.v_head_dim)
        k_nope, v = kv.transpose(1, 2).split([c.qk_nope_head_dim, c.v_head_dim], dim=-1)
        k = torch.cat([k_nope, k_pe[:, None].expand(-1, c.n_heads, -1, -1)], dim=-1)
        q = torch.cat([q_nope, q_pe], dim=-1)
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=self.scale)

    def _absorbed(self, q_nope, q_pe, c_kv, k_pe, mask) -> torch.Tensor:
        """吸收路径：把 W_UK 并进 query、W_UV 并进输出，直接在潜空间里做注意力。"""
        c = self.cfg
        w = self.wkv_b.weight.view(c.n_heads, c.qk_nope_head_dim + c.v_head_dim, c.kv_lora_rank)
        w_uk, w_uv = w[:, : c.qk_nope_head_dim], w[:, c.qk_nope_head_dim :]
        q_lat = torch.einsum("bhtd,hdr->bhtr", q_nope, w_uk)  # query 投进潜空间
        scores = torch.einsum("bhtr,bsr->bhts", q_lat, c_kv)  # 与缓存的潜向量直接点积
        scores = scores + torch.einsum("bhtd,bsd->bhts", q_pe, k_pe)  # 加上位置部分
        scores = scores * self.scale
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        # 在至少 float32 上做 softmax（BF16 推理时避免精度问题）
        acc = torch.promote_types(scores.dtype, torch.float32)
        p = torch.softmax(scores, dim=-1, dtype=acc).to(q_nope.dtype)
        o_lat = torch.einsum("bhts,bsr->bhtr", p, c_kv)  # 在潜空间里加权平均
        return torch.einsum("bhtr,hvr->bhtv", o_lat, w_uv)  # 最后再投回每个头的 value 空间


# ---------------------------------------------------------------------------
# 实验用：把 zero.model.Transformer 的注意力换成 MLA
# ---------------------------------------------------------------------------


def mla_transformer(model_cfg: ModelConfig, mla_cfg: MLAConfig) -> Transformer:
    """构建一个 Transformer，每层的注意力换成 MLAAttention，其余（RMSNorm、SwiGLU、共享 embedding）不变。"""
    model = Transformer(model_cfg)
    for layer in model.layers:
        layer.attn = MLAAttention(mla_cfg)
    model.init_weights()  # 让新模块也按同样规则初始化（wo 缩小 1/sqrt(2L)）
    return model


@torch.no_grad()
def generate_greedy(
    model: Transformer,
    prompt_ids: list[int],
    max_new_tokens: int,
    mla_cfg: MLAConfig,
    use_cache: bool = True,
) -> list[int]:
    """贪心生成（batch=1）。use_cache=False 时每步重算整段，用来和缓存版对拍。"""
    model.eval()
    device = next(model.parameters()).device
    ids = list(prompt_ids)
    cache = None
    if use_cache:
        cache = MLACache.from_config(
            mla_cfg, len(model.layers), 1, len(ids) + max_new_tokens, device=device
        )
        logits = model(torch.tensor([ids], device=device), kv_cache=cache, start_pos=0)
    out: list[int] = []
    for _ in range(max_new_tokens):
        if not use_cache:
            logits = model(torch.tensor([ids], device=device))
        nxt = int(logits[0, -1].argmax())
        out.append(nxt)
        if use_cache:
            logits = model(torch.tensor([[nxt]], device=device), kv_cache=cache, start_pos=len(ids))
        ids.append(nxt)
    return out
