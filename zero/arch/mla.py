"""Multi-head Latent Attention, MLA (Chapter 21). Experiment module; the main-line model does not use it.

DeepSeek-V2 introduced MLA (arXiv:2405.04434). DeepSeek-V3, Kimi K2, GLM-5, and Mistral Large 3 use it.
GQA stores fewer heads. MLA is different: it **jointly compresses** the K and V of all heads into one
low-rank latent vector c_KV, and the cache stores only this vector:

    c_KV = RMSNorm(W_DKV · h)              # (kv_lora_rank,)      ← cached
    k_R  = RoPE(W_KR · h)                  # (qk_rope_head_dim,)  ← cached (one, shared by all heads)
    k_i  = [W_UK,i · c_KV ; k_R]           # key of head i = part without position + shared RoPE part
    v_i  =  W_UV,i · c_KV
    q_i  = [W_UQ,i · h ; RoPE(W_QR,i · h)]

For each layer and position, the cache stores only kv_lora_rank + qk_rope_head_dim numbers
(DeepSeek-V3: 512 + 64 = 576). MHA with the same 128 heads stores 2 × 128 × 128 = 32,768 numbers.

**Why RoPE must be "decoupled"**: if RoPE rotates k_i = W_UK,i · c_KV directly, a position-dependent
rotation matrix sits between W_UQ and W_UK. Then the "absorb" trick below does not work.
Thus a small set of dimensions (qk_rope_head_dim) carries the position, and the other dimensions
carry no position.

**The absorb trick**: at inference, it is not necessary to recover K and V from c_KV, because
    q_iᵀ k_j = (W_UK,iᵀ q_i^C)ᵀ c_KV,j + q_i^Rᵀ k_R,j
    Σ_j p_ij v_j = W_UV,i (Σ_j p_ij c_KV,j)
First project the query into the latent space. Then do attention directly with the cached latent
vectors. At the end, use W_UV to project back.
Then each head computes dot products in 576 dimensions. In form, this is MQA where "all heads share
one set of K/V" (the GLM-5 report calls it the MQA mode of MLA). This module implements both paths:
`absorb=False` recovers K/V explicitly (for training/prefill), and `absorb=True` uses the absorb path
(for decode). `tests/test_arch_mla.py` checks that the two paths agree, and that generation with
and without the cache is identical.

The interface is the same as `zero.model.Attention`: `forward(x, cos, sin, kv_cache, layer_idx, start_pos)`.
Thus it can replace the attention in `zero.model.Transformer` directly (see `mla_transformer`).
One difference: MLA applies RoPE only to qk_rope_head_dim dimensions. Thus it uses its own RoPE
table and ignores the cos/sin arguments.

Industry implementations: FlashMLA (open-source GPU decode kernel by DeepSeek), and the MLA backends
of vLLM / SGLang. This file has only two goals: easy to read and correct. On RTX 3090, the forward /
backward parity check of CUDA against CPU passed, and generation with the latent cache is identical
to recomputation (2026-10, see runs/2026-10-01-gpu0-check/). The performance is not verified on GPU yet.
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
    """MLA hyperparameters. The field names are the same as in the Hugging Face DeepSeek-V3 config.json."""

    dim: int
    n_heads: int
    kv_lora_rank: int  # latent vector dimension d_c (DeepSeek-V3: 512)
    qk_nope_head_dim: int  # part of each head's key/query without position (DeepSeek-V3: 128)
    qk_rope_head_dim: int  # part with RoPE; all heads share this part of the key (DeepSeek-V3: 64)
    v_head_dim: int  # value dimension of each head (DeepSeek-V3: 128)
    q_lora_rank: int | None = None  # also compress the query (saves training activations, not KV cache); None = no compression
    max_seq_len: int = 2048
    rope_theta: float = 10000.0
    norm_eps: float = 1e-6

    @property
    def qk_head_dim(self) -> int:
        return self.qk_nope_head_dim + self.qk_rope_head_dim

    @property
    def cache_dim(self) -> int:
        """The number of cached values for each layer and position."""
        return self.kv_lora_rank + self.qk_rope_head_dim

    @classmethod
    def from_model_config(
        cls,
        cfg: ModelConfig,
        kv_lora_rank: int,
        qk_rope_head_dim: int | None = None,
        q_lora_rank: int | None = None,
    ) -> MLAConfig:
        """Keep the width, number of heads, and head_dim of a GQA config. Replace only the attention with MLA."""
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
    """The MLA cache. Each layer stores only the latent vector c_KV and the shared RoPE key.

    The memory is preallocated (as in zero.kv_cache.KVCache).
    Shapes: latent (n_layers, batch, max_seq_len, kv_lora_rank), k_rope (n_layers, batch, max_seq_len, rope_dim).
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
        """Write [start_pos, start_pos+T). Return all latent vectors and RoPE keys of this layer for 0..start_pos+T."""
        bsz, t, _ = c_kv.shape
        end = start_pos + t
        if end > self.max_seq_len:
            raise ValueError(f"MLA cache overflow: needs {end} positions, but only {self.max_seq_len} are preallocated")
        self.latent[layer_idx, :bsz, start_pos:end] = c_kv.to(self.latent.dtype)
        self.k_rope[layer_idx, :bsz, start_pos:end] = k_rope.to(self.k_rope.dtype)
        return self.latent[layer_idx, :bsz, :end], self.k_rope[layer_idx, :bsz, :end]

    def nbytes(self) -> int:
        return (
            self.latent.numel() * self.latent.element_size()
            + self.k_rope.numel() * self.k_rope.element_size()
        )


def _causal_mask(seqlen: int, kv_len: int, device: torch.device) -> torch.Tensor | None:
    """New token i can see positions <= past + i (past = kv_len - seqlen). With seqlen=1, no mask is necessary."""
    if seqlen == 1:
        return None
    past = kv_len - seqlen
    i = torch.arange(seqlen, device=device)[:, None]
    j = torch.arange(kv_len, device=device)[None, :]
    return j <= (past + i)


class MLAAttention(nn.Module):
    """Multi-head latent attention. The parameter names follow DeepSeek-V3 (wq_a/wq_b, wkv_a, wkv_b, wo)."""

    def __init__(self, cfg: MLAConfig, absorb: bool | None = None) -> None:
        super().__init__()
        self.cfg = cfg
        #: None = automatic: with a cache (inference), use the absorb path; without a cache (training),
        #: recover K/V explicitly.
        self.absorb = absorb
        h, dq = cfg.n_heads, cfg.qk_head_dim
        if cfg.q_lora_rank:
            self.wq_a = nn.Linear(cfg.dim, cfg.q_lora_rank, bias=False)
            self.q_norm = RMSNorm(cfg.q_lora_rank, cfg.norm_eps)
            self.wq_b = nn.Linear(cfg.q_lora_rank, h * dq, bias=False)
        else:
            self.wq = nn.Linear(cfg.dim, h * dq, bias=False)
        # Down-projection: get the latent vector c_KV and the shared RoPE key in one step
        # (DeepSeek calls it kv_a_proj_with_mqa).
        self.wkv_a = nn.Linear(cfg.dim, cfg.kv_lora_rank + cfg.qk_rope_head_dim, bias=False)
        self.kv_norm = RMSNorm(cfg.kv_lora_rank, cfg.norm_eps)
        # Up-projection: latent vector → k_nope and v of each head (W_UK and W_UV concatenated).
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
        del cos, sin  # MLA rotates only the rope sub-dimensions, so it uses its own RoPE table.
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
        k_pe = apply_rope(k_pe[:, None], cos_r, sin_r)[:, 0]  # (B, T, dr), shared by all heads

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
        """Recover K and V of each head explicitly, then do standard attention.

        This costs less compute for training and prefill.
        """
        c = self.cfg
        bsz, s, _ = c_kv.shape
        kv = self.wkv_b(c_kv).view(bsz, s, c.n_heads, c.qk_nope_head_dim + c.v_head_dim)
        k_nope, v = kv.transpose(1, 2).split([c.qk_nope_head_dim, c.v_head_dim], dim=-1)
        k = torch.cat([k_nope, k_pe[:, None].expand(-1, c.n_heads, -1, -1)], dim=-1)
        q = torch.cat([q_nope, q_pe], dim=-1)
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=self.scale)

    def _absorbed(self, q_nope, q_pe, c_kv, k_pe, mask) -> torch.Tensor:
        """Absorb path: merge W_UK into the query and W_UV into the output. Do attention in the latent space."""
        c = self.cfg
        w = self.wkv_b.weight.view(c.n_heads, c.qk_nope_head_dim + c.v_head_dim, c.kv_lora_rank)
        w_uk, w_uv = w[:, : c.qk_nope_head_dim], w[:, c.qk_nope_head_dim :]
        q_lat = torch.einsum("bhtd,hdr->bhtr", q_nope, w_uk)  # project the query into the latent space
        scores = torch.einsum("bhtr,bsr->bhts", q_lat, c_kv)  # dot product directly with the cached latent vectors
        scores = scores + torch.einsum("bhtd,bsd->bhts", q_pe, k_pe)  # add the position part
        scores = scores * self.scale
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        # Do softmax in at least float32 (this prevents precision problems in BF16 inference).
        acc = torch.promote_types(scores.dtype, torch.float32)
        p = torch.softmax(scores, dim=-1, dtype=acc).to(q_nope.dtype)
        o_lat = torch.einsum("bhts,bsr->bhtr", p, c_kv)  # weighted average in the latent space
        return torch.einsum("bhtr,hvr->bhtv", o_lat, w_uv)  # at the end, project back to the value space of each head


# ---------------------------------------------------------------------------
# For experiments: replace the attention of zero.model.Transformer with MLA
# ---------------------------------------------------------------------------


def mla_transformer(model_cfg: ModelConfig, mla_cfg: MLAConfig) -> Transformer:
    """Build a Transformer with MLAAttention in each layer.

    All other parts (RMSNorm, SwiGLU, shared embedding) do not change.
    """
    model = Transformer(model_cfg)
    for layer in model.layers:
        layer.attn = MLAAttention(mla_cfg)
    model.init_weights()  # Initialize the new modules with the same rules (wo scaled by 1/sqrt(2L)).
    return model


@torch.no_grad()
def generate_greedy(
    model: Transformer,
    prompt_ids: list[int],
    max_new_tokens: int,
    mla_cfg: MLAConfig,
    use_cache: bool = True,
) -> list[int]:
    """Greedy generation (batch=1).

    With use_cache=False, recompute the full sequence at each step. This is a parity check
    against the cached version.
    """
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
