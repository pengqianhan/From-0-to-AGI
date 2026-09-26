"""主线模型：与 Qwen3 稠密模型结构兼容的 Decoder-only Transformer（对应第 8、9、10、15 章）。

组件（都是"共识技术"，见 GOAL.md 3.3）：

- `RMSNorm`：只做缩放不做平移的归一化，Pre-Norm 放在每个子层之前；
- `RotaryEmbedding`：RoPE 旋转位置编码，含 YaRN 长上下文缩放（第 15 章）；
- `Attention`：GQA（多个查询头共享一组 K/V）+ QK-Norm（对每个头的 q、k 做 RMSNorm）
  + PyTorch 的 `scaled_dot_product_attention`（在 GPU 上会自动选 FlashAttention 内核）
  + 可选 KV cache（第 10 章）；
- `SwiGLU`：门控前馈网络 down(silu(gate(x)) * up(x))；
- `Block`：x + attn(norm(x))，再 x + ffn(norm(x))；
- `Transformer`：embedding → N 个 Block → 最后的 RMSNorm → lm_head（可与 embedding 共享权重）。

所有线性层都没有 bias。参数命名与 Hugging Face `Qwen3ForCausalLM` 的对应关系见 `zero/hf.py`，
`tests/test_model_hf_parity.py` 保证两者的 logits 一致。
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from zero.config import ModelConfig
from zero.kv_cache import KVCache

# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------


class RMSNorm(nn.Module):
    """y = x / sqrt(mean(x^2) + eps) * weight。

    与 HF 的 Qwen3RMSNorm 完全一致：先转 float32 算归一化，转回输入精度后再乘权重。
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x.to(dtype)


# ---------------------------------------------------------------------------
# RoPE（含 YaRN）
# ---------------------------------------------------------------------------


def compute_rope_inv_freq(
    head_dim: int, theta: float, scaling: dict[str, Any] | None = None
) -> tuple[torch.Tensor, float]:
    """返回 (inv_freq, attention_scaling)。

    默认 RoPE：inv_freq_i = theta^(-2i/d)，i = 0..d/2-1。

    YaRN（Peng et al. 2023, arXiv:2309.00071），实现与 transformers 的 `_compute_yarn_parameters` 一致：
    - 高频维度（波长远小于原训练长度，转了很多圈）保持原频率 —— "外推"；
    - 低频维度（波长比原训练长度还长）把频率除以 factor —— "内插"；
    - 中间用线性斜坡过渡，边界由 beta_fast / beta_slow（单位：圈数）决定；
    - cos/sin 额外乘 attention_scaling = 0.1 * ln(factor) + 1，补偿内插后注意力分布变"平"。
    """
    dim = head_dim
    pos_freqs = theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
    if scaling is None:
        return 1.0 / pos_freqs, 1.0

    factor = float(scaling["factor"])
    orig_max = int(scaling["original_max_position_embeddings"])
    beta_fast = float(scaling.get("beta_fast") or 32)
    beta_slow = float(scaling.get("beta_slow") or 1)
    attention_factor = scaling.get("attention_factor")
    if attention_factor is None:
        attention_factor = 1.0 if factor <= 1 else 0.1 * math.log(factor) + 1.0

    def correction_dim(num_rotations: float) -> float:
        # 在原训练长度内恰好转 num_rotations 圈的维度编号（可以是小数）
        return (dim * math.log(orig_max / (num_rotations * 2 * math.pi))) / (2 * math.log(theta))

    low = max(math.floor(correction_dim(beta_fast)), 0)
    high = min(math.ceil(correction_dim(beta_slow)), dim - 1)
    if low == high:
        high += 0.001  # 防止除零
    ramp = torch.clamp((torch.arange(dim // 2, dtype=torch.float32) - low) / (high - low), 0, 1)
    extrapolation_weight = 1 - ramp  # 1 = 完全保持原频率（高频），0 = 完全内插（低频）

    inv_freq_extrapolation = 1.0 / pos_freqs
    inv_freq_interpolation = 1.0 / (factor * pos_freqs)
    inv_freq = (
        inv_freq_interpolation * (1 - extrapolation_weight)
        + inv_freq_extrapolation * extrapolation_weight
    )
    return inv_freq, float(attention_factor)


class RotaryEmbedding(nn.Module):
    """预计算 [0, max_seq_len) 每个位置的 cos/sin，形状 (max_seq_len, head_dim)。

    采用 HF/Qwen 的"前后两半配对"写法：维度 i 与 i + d/2 组成一对旋转（而不是相邻两维），
    这样才能和官方权重对上。
    """

    def __init__(
        self,
        head_dim: int,
        max_seq_len: int,
        theta: float = 10000.0,
        scaling: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.scaling = scaling
        cos, sin = self._build()
        # 非持久 buffer：不进 state_dict，改了 theta / YaRN 之后重新构建即可（第 15 章长上下文扩展）
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def _build(self) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq, attn_scaling = compute_rope_inv_freq(self.head_dim, self.theta, self.scaling)
        t = torch.arange(self.max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)  # (T, d/2)
        emb = torch.cat((freqs, freqs), dim=-1)  # (T, d)
        return emb.cos() * attn_scaling, emb.sin() * attn_scaling

    def reset_buffers(self, device: torch.device | str) -> None:
        """在真实设备上重新计算 cos/sin（模型在 meta 设备上构建、to_empty 之后调用）。"""
        with torch.device(device):
            cos, sin = self._build()
        self.cos, self.sin = cos, sin

    def forward(self, start_pos: int, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        end = start_pos + seq_len
        if end > self.max_seq_len:
            raise ValueError(f"位置 {end} 超过了 RoPE 预计算的 max_seq_len={self.max_seq_len}")
        return self.cos[start_pos:end], self.sin[start_pos:end]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: (B, H, T, D)；cos/sin: (T, D)。"""
    cos = cos.to(x.dtype)
    sin = sin.to(x.dtype)
    return x * cos + rotate_half(x) * sin


# ---------------------------------------------------------------------------
# Attention / FFN / Block
# ---------------------------------------------------------------------------


class Attention(nn.Module):
    """GQA + QK-Norm + SDPA，可选 KV cache。"""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        assert config.head_dim is not None
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.wq = nn.Linear(config.dim, config.q_dim, bias=False)
        self.wk = nn.Linear(config.dim, config.kv_dim, bias=False)
        self.wv = nn.Linear(config.dim, config.kv_dim, bias=False)
        self.wo = nn.Linear(config.q_dim, config.dim, bias=False)
        # QK-Norm：在每个头的 head_dim 上做 RMSNorm（权重各头共享），在 RoPE 之前
        self.q_norm = RMSNorm(self.head_dim, config.norm_eps) if config.qk_norm else nn.Identity()
        self.k_norm = RMSNorm(self.head_dim, config.norm_eps) if config.qk_norm else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: KVCache | None = None,
        layer_idx: int = 0,
        start_pos: int = 0,
    ) -> torch.Tensor:
        bsz, seqlen, _ = x.shape
        q = self.q_norm(self.wq(x).view(bsz, seqlen, self.n_heads, self.head_dim)).transpose(1, 2)
        k = self.k_norm(self.wk(x).view(bsz, seqlen, self.n_kv_heads, self.head_dim)).transpose(
            1, 2
        )
        v = self.wv(x).view(bsz, seqlen, self.n_kv_heads, self.head_dim).transpose(1, 2)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        if kv_cache is not None:
            k, v = kv_cache.update(layer_idx, start_pos, k, v)
            k, v = k.to(q.dtype), v.to(q.dtype)

        kv_len = k.shape[2]
        past = kv_len - seqlen
        if past == 0:
            # 没有历史：标准因果掩码
            attn_mask, is_causal = None, seqlen > 1
        elif seqlen == 1:
            # 只有一个新 token：它可以看见全部历史，不需要掩码
            attn_mask, is_causal = None, False
        else:
            # 有历史又一次喂多个 token（例如分块 prefill）：第 i 个新 token 能看见位置 <= past + i
            i = torch.arange(seqlen, device=x.device)[:, None]
            j = torch.arange(kv_len, device=x.device)[None, :]
            attn_mask, is_causal = j <= (past + i), False

        # GQA：enable_gqa 让 SDPA 自己广播 K/V 头，不必 repeat_interleave 出多份拷贝
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            is_causal=is_causal,
            enable_gqa=self.n_kv_heads != self.n_heads,
        )
        out = out.transpose(1, 2).reshape(bsz, seqlen, self.n_heads * self.head_dim)
        return self.wo(out)


class SwiGLU(nn.Module):
    """FFN(x) = W_down( silu(W_gate x) * (W_up x) )。"""

    def __init__(self, dim: int, ffn_dim: int) -> None:
        super().__init__()
        self.w_gate = nn.Linear(dim, ffn_dim, bias=False)
        self.w_up = nn.Linear(dim, ffn_dim, bias=False)
        self.w_down = nn.Linear(ffn_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class Block(nn.Module):
    """Pre-Norm 残差块：x = x + attn(norm(x))；x = x + ffn(norm(x))。"""

    def __init__(self, config: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.attn_norm = RMSNorm(config.dim, config.norm_eps)
        self.attn = Attention(config)
        self.ffn_norm = RMSNorm(config.dim, config.norm_eps)
        self.ffn = SwiGLU(config.dim, config.ffn_dim)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: KVCache | None = None,
        start_pos: int = 0,
    ) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), cos, sin, kv_cache, self.layer_idx, start_pos)
        x = x + self.ffn(self.ffn_norm(x))
        return x


# ---------------------------------------------------------------------------
# Transformer
# ---------------------------------------------------------------------------


class Transformer(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        config.validate()
        assert config.head_dim is not None
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList([Block(config, i) for i in range(config.n_layers)])
        self.norm = RMSNorm(config.dim, config.norm_eps)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight
        self.rope = RotaryEmbedding(
            config.head_dim, config.max_seq_len, config.rope_theta, config.rope_scaling
        )
        self.init_weights()

    # ---- 初始化 ----
    @torch.no_grad()
    def init_weights(self) -> None:
        """正态初始化 N(0, init_std)；两个"写回残差流"的投影（wo、w_down）再除以 sqrt(2 * n_layers)，
        让残差流的方差不随层数增长（GPT-2 / OLMo 等的常见做法）。RMSNorm 权重为 1。"""
        std = self.config.init_std
        out_std = std / math.sqrt(2 * self.config.n_layers)
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                s = out_std if name.endswith(("attn.wo", "ffn.w_down")) else std
                nn.init.normal_(module.weight, mean=0.0, std=s)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=std)
            elif isinstance(module, RMSNorm):
                nn.init.ones_(module.weight)

    # ---- 前向 ----
    def forward(
        self, tokens: torch.Tensor, kv_cache: KVCache | None = None, start_pos: int = 0
    ) -> torch.Tensor:
        """tokens: (B, T) int64 → logits: (B, T, vocab_size)。

        有 kv_cache 时，tokens 是从位置 start_pos 开始的新 token，历史 K/V 从缓存里读。
        """
        _, seqlen = tokens.shape
        cos, sin = self.rope(start_pos, seqlen)
        h = self.tok_emb(tokens)
        for layer in self.layers:
            h = layer(h, cos, sin, kv_cache, start_pos)
        h = self.norm(h)
        return self.lm_head(h)

    def loss(
        self, tokens: torch.Tensor, targets: torch.Tensor, ignore_index: int = -100
    ) -> torch.Tensor:
        """语言模型交叉熵（在 float32 上算）。targets 里等于 ignore_index 的位置不计入（SFT 的 loss mask 用）。"""
        logits = self(tokens)
        return cross_entropy_loss(logits, targets, ignore_index)

    # ---- 统计 ----
    def num_params(self, non_embedding: bool = False) -> int:
        """参数量（共享权重只算一次）。non_embedding=True 时去掉 token embedding
        （以及不共享时的 lm_head），即 Kaplan 等人 scaling law 里的 N。"""
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.tok_emb.weight.numel()
            if not self.config.tie_embeddings:
                n -= self.lm_head.weight.numel()
        return n

    def flops_per_token(self, seq_len: int | None = None) -> float:
        return estimate_flops_per_token(self.config, seq_len or self.config.max_seq_len)


def cross_entropy_loss(
    logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = -100
) -> torch.Tensor:
    return F.cross_entropy(
        logits.float().view(-1, logits.size(-1)), targets.reshape(-1), ignore_index=ignore_index
    )


def count_params(config: ModelConfig) -> dict[str, int]:
    """不分配内存地数参数：返回 total / embedding / non_embedding 以及各部分明细。"""
    assert config.head_dim is not None
    d, L = config.dim, config.n_layers
    attn = d * config.q_dim + 2 * d * config.kv_dim + config.q_dim * d
    qk_norm = 2 * config.head_dim if config.qk_norm else 0
    ffn = 3 * d * config.ffn_dim
    norms = 2 * d
    per_layer = attn + qk_norm + ffn + norms
    emb = config.vocab_size * d
    head = 0 if config.tie_embeddings else config.vocab_size * d
    total = emb + head + L * per_layer + d
    return {
        "total": total,
        "embedding": emb + head,
        "non_embedding": total - emb - head,
        "per_layer": per_layer,
        "attention_per_layer": attn + qk_norm,
        "ffn_per_layer": ffn,
    }


def estimate_flops_per_token(config: ModelConfig, seq_len: int) -> float:
    """训练时每个 token 的浮点运算量（前向 + 反向），用于 MFU 和成本估算。

    = 6 * N_matmul + 12 * L * q_dim * T

    - N_matmul：所有参与矩阵乘法的参数，包括 lm_head（即使与 embedding 共享，输出投影的乘法照样要算），
      不包括 embedding 查表和 RMSNorm；前向每个参数 2 FLOPs、反向 4 FLOPs，合计 6（Kaplan 2020）；
    - 注意力里 QK^T 和 AV 两次矩阵乘：前向每 token 4 * q_dim * T，乘 3 得前向+反向 12 * q_dim * T
      （PaLM 附录 B 的写法；这里没有为因果掩码减半，与 PaLM / nanochat 的口径一致）。
    """
    assert config.head_dim is not None
    d, L = config.dim, config.n_layers
    n_matmul = L * (
        d * config.q_dim + 2 * d * config.kv_dim + config.q_dim * d + 3 * d * config.ffn_dim
    )
    n_matmul += d * config.vocab_size
    return 6.0 * n_matmul + 12.0 * L * config.q_dim * seq_len
