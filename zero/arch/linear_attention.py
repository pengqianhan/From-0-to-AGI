"""线性注意力与混合架构（对应第 23 章）：LinearAttention、GatedDeltaNet 与"3 线性 + 1 全注意力"的混合模型。

第五部分的实验模块，**不用于主线模型**。章节里的极简版在 `chapters/23-linear-attention-hybrid/code/`，
这里是同一套数学的"生产级"写法，结构与 Hugging Face 的 Qwen3.5 / Qwen3-Next 实现
（`transformers/models/qwen3_5/modeling_qwen3_5.py` 的 `Qwen3_5GatedDeltaNet`）对齐，参数名也相同，
`tests/test_arch_linear_attention.py` 把 HF 层的权重搬进来对拍输出。

核心思想（张量布局一律是 (B, H, T, D)，状态 S 的形状是 (B, H, d_k, d_v)）：

- 线性注意力：去掉 softmax 以后，o_t = q_tᵀ S_t，S_t = S_{t-1} + k_t v_tᵀ ——
  一个固定大小的状态代替了随长度增长的 KV cache；
- 衰减（decay / gate）：S_t = α_t S_{t-1} + k_t v_tᵀ，α_t = exp(g_t) ∈ (0, 1]，让旧信息逐渐遗忘
  （RetNet / GLA / Mamba-2 一族的共同骨架；这里用 Mamba-2 式的"每头一个标量门"）；
- delta 规则（DeltaNet）：先读出 S 对 k_t 的"旧答案" Sᵀk_t，只把差值写回：
  S_t = S_{t-1} + β_t k_t (v_t − S_{t-1}ᵀ k_t)ᵀ，即"覆盖"而不是"累加"；
- Gated DeltaNet：两者相乘，S_t = α_t (I − β_t k_t k_tᵀ) S_{t-1} + β_t k_t v_tᵀ。

每种算子都有两种等价写法：
- `recurrent_*`：逐 token 递推，O(1) 状态，推理 decode 用；
- `chunk_*`：分块并行（块内用矩阵乘、块间传状态），训练和 prefill 用；delta 规则的块内部分
  需要解一个下三角方程组（UT 变换，Yang et al. 2024, arXiv:2406.06484）。

纯 PyTorch 实现，只为可读和对拍。真正训练时应换成 flash-linear-attention（fla-org）的 Triton kernel
（`fla.ops.gated_delta_rule.chunk_gated_delta_rule` / `fused_recurrent_gated_delta_rule`），
以及 causal-conv1d 的 CUDA kernel。# 尚未在 GPU 上验证（fla 未安装）；本文件的纯 PyTorch 版在 CUDA 上能跑，
与 CPU 对拍一致（RTX 3090，2026-10，见 runs/2026-10-01-gpu0-check/）
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn

from zero.config import ConfigError, ModelConfig
from zero.kv_cache import KVCache
from zero.model import Attention, RMSNorm, RotaryEmbedding, SwiGLU, cross_entropy_loss

FULL_ATTENTION = "full_attention"
LINEAR_ATTENTION = "linear_attention"


# ---------------------------------------------------------------------------
# 层类型：每隔几层放一层全注意力
# ---------------------------------------------------------------------------


def hybrid_layer_types(n_layers: int, full_attention_interval: int = 4) -> list[str]:
    """Qwen3-Next / Qwen3.5 的排法：第 i 层（从 0 数）当 (i + 1) % interval == 0 时是全注意力。

    interval=4 就是 3:1（三层线性 + 一层全注意力）；interval=1 是纯全注意力；
    interval > n_layers 是纯线性注意力。
    """
    if n_layers <= 0 or full_attention_interval <= 0:
        raise ValueError("n_layers 和 full_attention_interval 必须是正整数")
    return [
        FULL_ATTENTION if (i + 1) % full_attention_interval == 0 else LINEAR_ATTENTION
        for i in range(n_layers)
    ]


def l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """与 HF / fla 一致的 L2 归一化：x / sqrt(sum(x²) + eps)。"""
    return x * torch.rsqrt((x * x).sum(dim=-1, keepdim=True) + eps)


# ---------------------------------------------------------------------------
# 算子：递推形式与分块形式（输入都已经做好归一化/缩放，内部统一用 float32）
# ---------------------------------------------------------------------------


def _zeros_state(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    b, h, _, dk = q.shape
    return torch.zeros(b, h, dk, v.shape[-1], dtype=torch.float32, device=q.device)


def recurrent_linear_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """(带衰减的)线性注意力，逐 token 递推。

    q, k: (B, H, T, Dk)；v: (B, H, T, Dv)；g: (B, H, T) 对数衰减（≤ 0），None 表示不衰减。
    返回 (o: (B, H, T, Dv), 最终状态 S: (B, H, Dk, Dv))。
    """
    q, k, v = q.float(), k.float(), v.float()
    S = _zeros_state(q, v) if initial_state is None else initial_state.float().clone()
    out = torch.empty_like(v)
    for t in range(q.shape[2]):
        if g is not None:
            S = S * g[:, :, t].float().exp()[..., None, None]  # S ← α_t S
        S = S + k[:, :, t, :, None] * v[:, :, t, None, :]  # S ← S + k_t v_tᵀ
        out[:, :, t] = (q[:, :, t, :, None] * S).sum(-2)  # o_t = q_tᵀ S
    return out, S


def chunk_linear_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor | None = None,
    chunk_size: int = 64,
    initial_state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """与 `recurrent_linear_attention` 数学上等价的分块并行形式。

    记块内累计对数衰减 b_i = Σ_{s≤i} g_s，则块内第 i 个位置：
        o_i = e^{b_i} q_iᵀ S_0 + Σ_{j≤i} e^{b_i − b_j} (q_i·k_j) v_j
        S_C = e^{b_C} S_0 + Σ_j e^{b_C − b_j} k_j v_jᵀ
    指数里永远是"后减前"（≤ 0），不会溢出。
    """
    B, H, T, _ = q.shape
    q, k, v = q.float(), k.float(), v.float()
    g = torch.zeros(B, H, T, device=q.device) if g is None else g.float()
    C = chunk_size
    pad = (-T) % C
    if pad:  # 补零：k=0、g=0 的位置对状态没有任何影响
        q, k, v = (F.pad(x, (0, 0, 0, pad)) for x in (q, k, v))
        g = F.pad(g, (0, pad))
    n = (T + pad) // C
    q, k, v = (x.reshape(B, H, n, C, x.shape[-1]) for x in (q, k, v))
    b = g.reshape(B, H, n, C).cumsum(-1)
    causal = torch.ones(C, C, dtype=torch.bool, device=q.device).tril()
    S = _zeros_state(q[:, :, 0], v[:, :, 0]) if initial_state is None else initial_state.float()
    out = torch.empty_like(v)
    for c in range(n):
        qc, kc, vc, bc = q[:, :, c], k[:, :, c], v[:, :, c], b[:, :, c]
        decay = (bc[..., :, None] - bc[..., None, :]).masked_fill(~causal, float("-inf")).exp()
        intra = (qc @ kc.transpose(-1, -2)) * decay  # 块内：带衰减、带因果掩码的 QKᵀ
        out[:, :, c] = (qc * bc.exp()[..., None]) @ S + intra @ vc
        last = bc[..., -1:]
        S = S * last.exp()[..., None] + (kc * (last - bc).exp()[..., None]).transpose(-1, -2) @ vc
    return out.reshape(B, H, n * C, -1)[:, :, :T], S


def recurrent_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor | None,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gated DeltaNet 的逐 token 递推（g=None 时就是 DeltaNet）。

        S ← α_t S                       （衰减，α_t = e^{g_t}）
        u_t = β_t (v_t − Sᵀ k_t)         （只写入"新值与旧答案的差"）
        S ← S + k_t u_tᵀ                 （合起来：S_t = α_t (I − β_t k_t k_tᵀ) S_{t−1} + β_t k_t v_tᵀ）
        o_t = Sᵀ q_t
    beta: (B, H, T)，取值 (0, 1)。
    """
    q, k, v, beta = q.float(), k.float(), v.float(), beta.float()
    S = _zeros_state(q, v) if initial_state is None else initial_state.float().clone()
    out = torch.empty_like(v)
    for t in range(q.shape[2]):
        if g is not None:
            S = S * g[:, :, t].float().exp()[..., None, None]
        k_t = k[:, :, t]
        old = (S * k_t[..., None]).sum(-2)  # Sᵀ k_t：状态对 k_t 的"旧答案"
        u = (v[:, :, t] - old) * beta[:, :, t, None]
        S = S + k_t[..., None] * u[..., None, :]
        out[:, :, t] = (S * q[:, :, t, :, None]).sum(-2)
    return out, S


def chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor | None,
    beta: torch.Tensor,
    chunk_size: int = 64,
    initial_state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """与 `recurrent_gated_delta_rule` 等价的分块形式（UT 变换 / WY 表示）。

    块内每个位置写入的"修正值" u_i 依赖前面所有 u_j：
        u_i = β_i (v_i − e^{b_i} S_0ᵀ k_i − Σ_{j<i} e^{b_i − b_j} (k_i·k_j) u_j)
    这是一个下三角线性方程组 (I + A) U = β⊙V − (β⊙e^{b}⊙K) S_0，
    其中 A_ij = β_i e^{b_i − b_j} (k_i·k_j)（j < i）。解出 U 之后，块内输出和块末状态
    与带衰减的线性注意力完全同形，只是把 v 换成了 u。
    """
    B, H, T, _ = q.shape
    q, k, v, beta = q.float(), k.float(), v.float(), beta.float()
    g = torch.zeros(B, H, T, device=q.device) if g is None else g.float()
    C = chunk_size
    pad = (-T) % C
    if pad:  # 补零：β=0、k=0 的位置既不写入也不影响别人
        q, k, v = (F.pad(x, (0, 0, 0, pad)) for x in (q, k, v))
        g, beta = F.pad(g, (0, pad)), F.pad(beta, (0, pad))
    n = (T + pad) // C
    q, k, v = (x.reshape(B, H, n, C, x.shape[-1]) for x in (q, k, v))
    b = g.reshape(B, H, n, C).cumsum(-1)
    beta = beta.reshape(B, H, n, C)
    causal = torch.ones(C, C, dtype=torch.bool, device=q.device).tril()
    eye = torch.eye(C, device=q.device)
    S = _zeros_state(q[:, :, 0], v[:, :, 0]) if initial_state is None else initial_state.float()
    out = torch.empty_like(v)
    for c in range(n):
        qc, kc, vc, bc, bt = q[:, :, c], k[:, :, c], v[:, :, c], b[:, :, c], beta[:, :, c]
        decay = (bc[..., :, None] - bc[..., None, :]).masked_fill(~causal, float("-inf")).exp()
        kb = kc * bt[..., None]
        A = ((kb @ kc.transpose(-1, -2)) * decay).tril(-1)  # 严格下三角
        M = eye + A
        # 把 U 拆成"与 S_0 无关的部分"和"读 S_0 的部分"：U = W_v − W_k S_0
        W_v = torch.linalg.solve_triangular(M, vc * bt[..., None], upper=False)
        W_k = torch.linalg.solve_triangular(M, kb * bc.exp()[..., None], upper=False)
        u = W_v - W_k @ S
        intra = (qc @ kc.transpose(-1, -2)) * decay
        out[:, :, c] = (qc * bc.exp()[..., None]) @ S + intra @ u
        last = bc[..., -1:]
        S = S * last.exp()[..., None] + (kc * (last - bc).exp()[..., None]).transpose(-1, -2) @ u
    return out.reshape(B, H, n * C, -1)[:, :, :T], S


# ---------------------------------------------------------------------------
# 缓存：全注意力层用 KV cache（随长度增长），线性层用固定大小的状态
# ---------------------------------------------------------------------------


@dataclass
class LinearState:
    """一层线性注意力的推理状态：递推状态 S 和短卷积的最后 K−1 个输入。大小与序列长度无关。"""

    recurrent: torch.Tensor | None = None  # (B, H_v, d_k, d_v)，float32
    conv: torch.Tensor | None = None  # (B, conv_dim, K − 1)

    def nbytes(self) -> int:
        return sum(
            t.numel() * t.element_size() for t in (self.recurrent, self.conv) if t is not None
        )


@dataclass
class HybridCache:
    """混合模型的推理缓存。

    - `kv`：只为全注意力层预分配的 `KVCache`（层数 = 全注意力层数），`kv_slot[layer_idx]` 给出槽位；
    - `linear[layer_idx]`：线性层的 `LinearState`，第一次前向时创建。
    """

    kv: KVCache | None
    kv_slot: dict[int, int]
    linear: dict[int, LinearState] = field(default_factory=dict)

    def state(self, layer_idx: int) -> LinearState:
        return self.linear.setdefault(layer_idx, LinearState())

    def nbytes(self) -> dict[str, int]:
        kv = self.kv.nbytes() if self.kv is not None else 0
        lin = sum(s.nbytes() for s in self.linear.values())
        return {"kv": kv, "linear_state": lin, "total": kv + lin}


# ---------------------------------------------------------------------------
# 层
# ---------------------------------------------------------------------------


class GatedRMSNorm(nn.Module):
    """RMSNorm(x) * silu(z)：先归一化再乘门（与 HF 的 Qwen3_5RMSNormGated 一致）。"""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        x = self.weight * x.to(dtype)
        return (x * F.silu(gate.float())).to(dtype)


class _LinearMixer(nn.Module):
    """LinearAttention 与 GatedDeltaNet 的公共骨架（参数名与 HF Qwen3_5GatedDeltaNet 相同）：

    x ─ in_proj_qkv ─ 因果短卷积 + SiLU ─ 拆成 q, k, v ─ L2 归一化 q, k（q 再乘 1/√d_k）
      ─ in_proj_a ─ g = −exp(A_log) · softplus(a + dt_bias)   （Mamba-2 式的每头对数衰减）
      ─ in_proj_b ─ β = sigmoid(b)                            （只有 delta 规则用）
      ─ 核心算子（递推 / 分块）─ GatedRMSNorm(·, in_proj_z(x)) ─ out_proj
    """

    use_delta: bool = False

    def __init__(
        self,
        dim: int,
        num_k_heads: int,
        num_v_heads: int,
        head_k_dim: int,
        head_v_dim: int,
        conv_kernel: int = 4,
        use_decay: bool = True,
        chunk_size: int = 64,
        norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if num_v_heads % num_k_heads != 0:
            raise ValueError(f"num_v_heads={num_v_heads} 必须是 num_k_heads={num_k_heads} 的整数倍")
        self.num_k_heads, self.num_v_heads = num_k_heads, num_v_heads
        self.head_k_dim, self.head_v_dim = head_k_dim, head_v_dim
        self.key_dim, self.value_dim = num_k_heads * head_k_dim, num_v_heads * head_v_dim
        self.conv_kernel = conv_kernel
        self.use_decay = use_decay
        self.chunk_size = chunk_size
        self.conv_dim = 2 * self.key_dim + self.value_dim

        self.in_proj_qkv = nn.Linear(dim, self.conv_dim, bias=False)
        self.in_proj_z = nn.Linear(dim, self.value_dim, bias=False)
        if self.use_delta:
            self.in_proj_b = nn.Linear(dim, num_v_heads, bias=False)
        if use_decay:
            self.in_proj_a = nn.Linear(dim, num_v_heads, bias=False)
            self.dt_bias = nn.Parameter(torch.ones(num_v_heads))
            self.A_log = nn.Parameter(torch.zeros(num_v_heads))
        if conv_kernel > 0:
            # 深度可分离的因果卷积：每个通道只看自己过去 K 个位置（权重形状与 nn.Conv1d(groups=C) 相同）
            self.conv1d = nn.Conv1d(
                self.conv_dim,
                self.conv_dim,
                conv_kernel,
                groups=self.conv_dim,
                bias=False,
                padding=conv_kernel - 1,
            )
        self.norm = GatedRMSNorm(head_v_dim, norm_eps)
        self.out_proj = nn.Linear(self.value_dim, dim, bias=False)
        self.reset_gate_parameters()

    @torch.no_grad()
    def reset_gate_parameters(self) -> None:
        """Mamba-2 式初始化：A ~ U(1, 16)；dt ~ logU(1e-3, 0.1)，dt_bias = softplus⁻¹(dt)。"""
        if not self.use_decay:
            return
        self.A_log.copy_(torch.empty_like(self.A_log).uniform_(1, 16).log())
        dt = torch.exp(
            torch.rand_like(self.dt_bias) * (math.log(0.1) - math.log(1e-3)) + math.log(1e-3)
        )
        self.dt_bias.copy_(dt + torch.log(-torch.expm1(-dt)))

    # ---- 短卷积（带缓存） ----
    def _short_conv(self, x: torch.Tensor, state: LinearState | None) -> torch.Tensor:
        """x: (B, C, T) → (B, C, T)。有缓存时把上一段最后 K−1 个输入接在前面，保证分段喂和一次喂结果相同。"""
        if self.conv_kernel <= 0:
            return x
        K = self.conv_kernel
        if state is not None and state.conv is not None:
            prev = state.conv.to(x.dtype)
        else:
            prev = x.new_zeros(x.shape[0], x.shape[1], K - 1)
        xx = torch.cat([prev, x], dim=-1)
        if state is not None:
            state.conv = xx[:, :, -(K - 1) :].detach().clone() if K > 1 else None
        out = F.conv1d(xx, self.conv1d.weight, groups=self.conv_dim)
        return F.silu(out)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor | None = None,
        sin: torch.Tensor | None = None,
        cache: HybridCache | None = None,
        layer_idx: int = 0,
        start_pos: int = 0,
        mode: str = "auto",
    ) -> torch.Tensor:
        """x: (B, T, dim) → (B, T, dim)。签名与 `zero.model.Attention` 对齐；cos/sin 不用
        （Qwen3.5 的线性层不加 RoPE，位置信息来自递推顺序和短卷积）。

        mode: "auto"（T=1 用递推、否则分块）、"recurrent"、"chunk"。
        """
        bsz, seqlen, _ = x.shape
        state = cache.state(layer_idx) if cache is not None else None
        qkv = self._short_conv(self.in_proj_qkv(x).transpose(1, 2), state).transpose(1, 2)
        q, k, v = torch.split(qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
        q = q.reshape(bsz, seqlen, self.num_k_heads, self.head_k_dim).transpose(1, 2)
        k = k.reshape(bsz, seqlen, self.num_k_heads, self.head_k_dim).transpose(1, 2)
        v = v.reshape(bsz, seqlen, self.num_v_heads, self.head_v_dim).transpose(1, 2)
        q = l2norm(q.float()) * self.head_k_dim**-0.5
        k = l2norm(k.float())
        rep = self.num_v_heads // self.num_k_heads
        if rep > 1:  # 类似 GQA：多个 v 头共享一组 q/k 头
            q, k = q.repeat_interleave(rep, dim=1), k.repeat_interleave(rep, dim=1)

        g = None
        if self.use_decay:
            a = self.in_proj_a(x).float()
            g = (-self.A_log.float().exp() * F.softplus(a + self.dt_bias)).transpose(1, 2)
        init = state.recurrent if state is not None else None
        use_recurrent = mode == "recurrent" or (mode == "auto" and seqlen == 1)
        if self.use_delta:
            beta = torch.sigmoid(self.in_proj_b(x).float()).transpose(1, 2)
            if use_recurrent:
                o, S = recurrent_gated_delta_rule(q, k, v, g, beta, init)
            else:
                o, S = chunk_gated_delta_rule(q, k, v, g, beta, self.chunk_size, init)
        else:
            if use_recurrent:
                o, S = recurrent_linear_attention(q, k, v, g, init)
            else:
                o, S = chunk_linear_attention(q, k, v, g, self.chunk_size, init)
        if state is not None:
            state.recurrent = S.detach()

        z = self.in_proj_z(x).reshape(bsz, seqlen, self.num_v_heads, self.head_v_dim)
        o = self.norm(o.transpose(1, 2).to(x.dtype), z)
        return self.out_proj(o.reshape(bsz, seqlen, self.value_dim))


class LinearAttention(_LinearMixer):
    """线性注意力：S_t = α_t S_{t−1} + k_t v_tᵀ。use_decay=False 时 α_t ≡ 1，就是最朴素的线性注意力
    （Katharopoulos et al. 2020 去掉分母归一化、换成 L2 归一化 + 输出 RMSNorm 的现代写法）。"""

    use_delta = False


class GatedDeltaNet(_LinearMixer):
    """Gated DeltaNet（Yang, Kautz, Hatamizadeh 2024, arXiv:2412.06464），Qwen3-Next / Qwen3.5 的线性层。
    use_decay=False 时退化为 DeltaNet。"""

    use_delta = True


# ---------------------------------------------------------------------------
# 混合模型
# ---------------------------------------------------------------------------


@dataclass
class HybridConfig(ModelConfig):
    """在 `ModelConfig` 上加线性层的超参。字段名尽量与 HF Qwen3.5 的 text_config 对应：
    full_attention_interval、layer_types、linear_num_key_heads、linear_num_value_heads、
    linear_key_head_dim、linear_value_head_dim、linear_conv_kernel_dim。"""

    layer_types: list[str] | None = None  # None 时由 full_attention_interval 生成
    full_attention_interval: int = 4
    linear_kind: str = "gated_deltanet"  # "gated_deltanet" | "deltanet" | "gated_linear" | "linear"
    linear_num_key_heads: int = 4
    linear_num_value_heads: int = 4
    linear_key_head_dim: int = 32
    linear_value_head_dim: int = 32
    linear_conv_kernel_dim: int = 4
    linear_chunk_size: int = 64

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.layer_types is None:
            self.layer_types = hybrid_layer_types(self.n_layers, self.full_attention_interval)

    def validate(self) -> None:
        super().validate()
        assert self.layer_types is not None
        if len(self.layer_types) != self.n_layers:
            raise ConfigError(
                f"[model] layer_types 有 {len(self.layer_types)} 项，n_layers={self.n_layers}"
            )
        bad = set(self.layer_types) - {FULL_ATTENTION, LINEAR_ATTENTION}
        if bad:
            raise ConfigError(f"[model] layer_types 里有未知类型 {sorted(bad)}")
        if self.linear_kind not in ("gated_deltanet", "deltanet", "gated_linear", "linear"):
            raise ConfigError(f"[model] 未知的 linear_kind={self.linear_kind!r}")


def build_linear_layer(config: HybridConfig) -> _LinearMixer:
    kind = config.linear_kind
    cls = GatedDeltaNet if kind in ("gated_deltanet", "deltanet") else LinearAttention
    return cls(
        config.dim,
        config.linear_num_key_heads,
        config.linear_num_value_heads,
        config.linear_key_head_dim,
        config.linear_value_head_dim,
        conv_kernel=config.linear_conv_kernel_dim,
        use_decay=kind in ("gated_deltanet", "gated_linear"),
        chunk_size=config.linear_chunk_size,
        norm_eps=config.norm_eps,
    )


class HybridBlock(nn.Module):
    """Pre-Norm 残差块，token mixer 是全注意力（zero.model.Attention）或线性层。"""

    def __init__(self, config: HybridConfig, layer_idx: int) -> None:
        super().__init__()
        assert config.layer_types is not None
        self.layer_idx = layer_idx
        self.layer_type = config.layer_types[layer_idx]
        self.attn_norm = RMSNorm(config.dim, config.norm_eps)
        self.attn: nn.Module = (
            Attention(config) if self.layer_type == FULL_ATTENTION else build_linear_layer(config)
        )
        self.ffn_norm = RMSNorm(config.dim, config.norm_eps)
        self.ffn = SwiGLU(config.dim, config.ffn_dim)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cache: HybridCache | None = None,
        start_pos: int = 0,
    ) -> torch.Tensor:
        h = self.attn_norm(x)
        if self.layer_type == FULL_ATTENTION:
            kv = cache.kv if cache is not None else None
            slot = cache.kv_slot[self.layer_idx] if cache is not None else 0
            h = self.attn(h, cos, sin, kv, slot, start_pos)
        else:
            h = self.attn(h, cache=cache, layer_idx=self.layer_idx, start_pos=start_pos)
        x = x + h
        return x + self.ffn(self.ffn_norm(x))


class HybridTransformer(nn.Module):
    """与 `zero.model.Transformer` 同接口的混合模型：forward(tokens, cache=None, start_pos=0) -> logits。"""

    def __init__(self, config: HybridConfig) -> None:
        super().__init__()
        config.validate()
        assert config.head_dim is not None and config.layer_types is not None
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList([HybridBlock(config, i) for i in range(config.n_layers)])
        self.norm = RMSNorm(config.dim, config.norm_eps)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight
        self.rope = RotaryEmbedding(
            config.head_dim, config.max_seq_len, config.rope_theta, config.rope_scaling
        )
        self.full_layers = [i for i, t in enumerate(config.layer_types) if t == FULL_ATTENTION]
        self.init_weights()

    @torch.no_grad()
    def init_weights(self) -> None:
        """与 zero.model.Transformer 相同：N(0, std)，写回残差流的投影再除以 sqrt(2L)；门控参数单独初始化。"""
        std = self.config.init_std
        out_std = std / math.sqrt(2 * self.config.n_layers)
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                s = out_std if name.endswith(("attn.wo", "attn.out_proj", "ffn.w_down")) else std
                nn.init.normal_(module.weight, mean=0.0, std=s)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=std)
            elif isinstance(module, nn.Conv1d):
                nn.init.normal_(module.weight, mean=0.0, std=1.0 / math.sqrt(module.kernel_size[0]))
            elif isinstance(module, (RMSNorm, GatedRMSNorm)):
                nn.init.ones_(module.weight)
            elif isinstance(module, _LinearMixer):
                module.reset_gate_parameters()

    def new_cache(
        self,
        batch_size: int,
        max_seq_len: int | None = None,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> HybridCache:
        """只给全注意力层分配 KV cache；线性层的状态在第一次前向时创建。"""
        c = self.config
        assert c.head_dim is not None
        kv = None
        if self.full_layers:
            kv = KVCache(
                len(self.full_layers),
                batch_size,
                max_seq_len or c.max_seq_len,
                c.n_kv_heads,
                c.head_dim,
                device,
                dtype,
            )
        return HybridCache(kv=kv, kv_slot={layer: s for s, layer in enumerate(self.full_layers)})

    def forward(
        self, tokens: torch.Tensor, cache: HybridCache | None = None, start_pos: int = 0
    ) -> torch.Tensor:
        _, seqlen = tokens.shape
        cos, sin = self.rope(start_pos, seqlen)
        h = self.tok_emb(tokens)
        for layer in self.layers:
            h = layer(h, cos, sin, cache, start_pos)
        return self.lm_head(self.norm(h))

    def loss(
        self, tokens: torch.Tensor, targets: torch.Tensor, ignore_index: int = -100
    ) -> torch.Tensor:
        return cross_entropy_loss(self(tokens), targets, ignore_index)


@torch.no_grad()
def generate_greedy(
    model: HybridTransformer, prompt: list[int], max_new_tokens: int, use_cache: bool = True
) -> list[int]:
    """贪心生成（返回新 token）。use_cache=False 时每步把整段序列重算一遍，专门用来对拍。"""
    # 输入和缓存放到模型所在的设备上（2026-10 在 RTX 3090 上发现原来固定在 CPU，模型在 CUDA 上时报错）
    device = next(model.parameters()).device
    ids = list(prompt)
    if not use_cache:
        for _ in range(max_new_tokens):
            ids.append(int(model(torch.tensor([ids], device=device))[0, -1].argmax()))
        return ids[len(prompt) :]
    cache = model.new_cache(1, len(prompt) + max_new_tokens, device=device)
    logits = model(torch.tensor([ids], device=device), cache, 0)  # prefill：分块形式
    for _ in range(max_new_tokens):
        nxt = int(logits[0, -1].argmax())
        ids.append(nxt)
        # decode：递推形式，O(1) 状态
        logits = model(torch.tensor([[nxt]], device=device), cache, len(ids) - 1)
    return ids[len(prompt) :]


def cache_bytes_per_sequence(
    layer_types: list[str],
    seq_len: int,
    n_kv_heads: int,
    head_dim: int,
    linear_num_value_heads: int,
    linear_key_head_dim: int,
    linear_value_head_dim: int,
    conv_dim: int = 0,
    conv_kernel: int = 4,
    kv_bytes: int = 2,
    state_bytes: int = 4,
) -> dict[str, int]:
    """一条序列的推理缓存字节数（第 21 章的账本加上线性层）：

    - 全注意力层：2 × KV 头数 × head_dim × 序列长 × kv_bytes（随长度线性增长）；
    - 线性层：H_v × d_k × d_v × state_bytes（递推状态，常用 FP32）+ conv_dim × (K−1) × kv_bytes（与长度无关）。
    """
    n_full = sum(t == FULL_ATTENTION for t in layer_types)
    n_lin = len(layer_types) - n_full
    kv = n_full * 2 * n_kv_heads * head_dim * seq_len * kv_bytes
    state = n_lin * (
        linear_num_value_heads * linear_key_head_dim * linear_value_head_dim * state_bytes
        + conv_dim * max(conv_kernel - 1, 0) * kv_bytes
    )
    return {"kv": kv, "linear_state": state, "total": kv + state}
