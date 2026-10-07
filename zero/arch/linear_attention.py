"""Linear attention and hybrid architecture (Chapter 23): LinearAttention, GatedDeltaNet, and
a hybrid model with "3 linear + 1 full attention" layers.

Experiment module for Part 5. **The main-line model does not use it.** The minimal version of
the chapter is in `chapters/23-linear-attention-hybrid/code/`. This file is the "production"
version of the same math. Its structure matches the Hugging Face Qwen3.5 / Qwen3-Next
implementation (`Qwen3_5GatedDeltaNet` in `transformers/models/qwen3_5/modeling_qwen3_5.py`),
and the parameter names are the same. `tests/test_arch_linear_attention.py` copies the weights of
the HF layer into this layer and does a parity check of the outputs.

Core ideas (the tensor layout is always (B, H, T, D); the state S has the shape (B, H, d_k, d_v)):

- Linear attention: without softmax, o_t = q_tᵀ S_t, S_t = S_{t-1} + k_t v_tᵀ.
  A state with a fixed size replaces the KV cache that grows with the length.
- Decay (gate): S_t = α_t S_{t-1} + k_t v_tᵀ, α_t = exp(g_t) ∈ (0, 1]. The model gradually forgets
  old information (the common frame of the RetNet / GLA / Mamba-2 family; this file uses the
  Mamba-2-style "one scalar gate for each head").
- Delta rule (DeltaNet): first read the "old answer" Sᵀk_t of S for k_t, then write back only the
  difference: S_t = S_{t-1} + β_t k_t (v_t − S_{t-1}ᵀ k_t)ᵀ. This "overwrites" and does not "add".
- Gated DeltaNet: the product of the two, S_t = α_t (I − β_t k_t k_tᵀ) S_{t-1} + β_t k_t v_tᵀ.

Each operator has two equivalent forms:
- `recurrent_*`: token-by-token recurrence with an O(1) state, for inference decode.
- `chunk_*`: chunked parallel form (matrix multiplication inside a chunk, state passed between
  chunks), for training and prefill. The intra-chunk part of the delta rule must solve a
  lower-triangular system of equations (UT transform, Yang et al. 2024, arXiv:2406.06484).

Pure PyTorch implementation, only for readability and parity checks. For real training, replace it
with the Triton kernels of flash-linear-attention (fla-org)
(`fla.ops.gated_delta_rule.chunk_gated_delta_rule` / `fused_recurrent_gated_delta_rule`),
and the CUDA kernel of causal-conv1d. # Not verified on GPU yet (fla is not installed). The pure
PyTorch version of this file runs on CUDA and agrees with the CPU parity check
(RTX 3090, 2026-10, see runs/2026-10-01-gpu0-check/).
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
# Layer types: one full attention layer after every few layers
# ---------------------------------------------------------------------------


def hybrid_layer_types(n_layers: int, full_attention_interval: int = 4) -> list[str]:
    """The Qwen3-Next / Qwen3.5 layout: layer i (counted from 0) is full attention when
    (i + 1) % interval == 0.

    interval=4 gives 3:1 (three linear layers + one full attention layer). interval=1 gives pure
    full attention. interval > n_layers gives pure linear attention.
    """
    if n_layers <= 0 or full_attention_interval <= 0:
        raise ValueError("n_layers and full_attention_interval must be positive integers")
    return [
        FULL_ATTENTION if (i + 1) % full_attention_interval == 0 else LINEAR_ATTENTION
        for i in range(n_layers)
    ]


def l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """L2 normalization, the same as in HF / fla: x / sqrt(sum(x²) + eps)."""
    return x * torch.rsqrt((x * x).sum(dim=-1, keepdim=True) + eps)


# ---------------------------------------------------------------------------
# Operators: recurrent form and chunked form (the inputs are already normalized/scaled;
# all internal computation uses float32)
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
    """Linear attention (with optional decay), token-by-token recurrence.

    q, k: (B, H, T, Dk); v: (B, H, T, Dv); g: (B, H, T) log decay (≤ 0), None = no decay.
    Return (o: (B, H, T, Dv), final state S: (B, H, Dk, Dv)).
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
    """Chunked parallel form, mathematically equivalent to `recurrent_linear_attention`.

    Let b_i = Σ_{s≤i} g_s be the cumulative log decay inside the chunk. Then for position i
    in the chunk:
        o_i = e^{b_i} q_iᵀ S_0 + Σ_{j≤i} e^{b_i − b_j} (q_i·k_j) v_j
        S_C = e^{b_C} S_0 + Σ_j e^{b_C − b_j} k_j v_jᵀ
    The exponent is always "later minus earlier" (≤ 0), so it cannot overflow.
    """
    B, H, T, _ = q.shape
    q, k, v = q.float(), k.float(), v.float()
    g = torch.zeros(B, H, T, device=q.device) if g is None else g.float()
    C = chunk_size
    pad = (-T) % C
    if pad:  # Zero padding: positions with k=0, g=0 have no effect on the state.
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
        intra = (qc @ kc.transpose(-1, -2)) * decay  # inside the chunk: QKᵀ with decay and causal mask
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
    """Token-by-token recurrence of Gated DeltaNet (with g=None, this is DeltaNet).

        S ← α_t S                       (decay, α_t = e^{g_t})
        u_t = β_t (v_t − Sᵀ k_t)         (write only "the difference between the new value and the old answer")
        S ← S + k_t u_tᵀ                 (together: S_t = α_t (I − β_t k_t k_tᵀ) S_{t−1} + β_t k_t v_tᵀ)
        o_t = Sᵀ q_t
    beta: (B, H, T), values in (0, 1).
    """
    q, k, v, beta = q.float(), k.float(), v.float(), beta.float()
    S = _zeros_state(q, v) if initial_state is None else initial_state.float().clone()
    out = torch.empty_like(v)
    for t in range(q.shape[2]):
        if g is not None:
            S = S * g[:, :, t].float().exp()[..., None, None]
        k_t = k[:, :, t]
        old = (S * k_t[..., None]).sum(-2)  # Sᵀ k_t: the "old answer" of the state for k_t
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
    """Chunked form, equivalent to `recurrent_gated_delta_rule` (UT transform / WY representation).

    The "correction" u_i that each position in the chunk writes depends on all earlier u_j:
        u_i = β_i (v_i − e^{b_i} S_0ᵀ k_i − Σ_{j<i} e^{b_i − b_j} (k_i·k_j) u_j)
    This is a lower-triangular linear system (I + A) U = β⊙V − (β⊙e^{b}⊙K) S_0,
    where A_ij = β_i e^{b_i − b_j} (k_i·k_j) (j < i). After we solve for U, the output inside the
    chunk and the state at the chunk end have exactly the same form as in linear attention
    with decay. The only change is u in place of v.
    """
    B, H, T, _ = q.shape
    q, k, v, beta = q.float(), k.float(), v.float(), beta.float()
    g = torch.zeros(B, H, T, device=q.device) if g is None else g.float()
    C = chunk_size
    pad = (-T) % C
    if pad:  # Zero padding: positions with β=0, k=0 do not write and do not affect other positions.
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
        A = ((kb @ kc.transpose(-1, -2)) * decay).tril(-1)  # strictly lower triangular
        M = eye + A
        # Split U into "the part that does not depend on S_0" and "the part that reads S_0": U = W_v − W_k S_0
        W_v = torch.linalg.solve_triangular(M, vc * bt[..., None], upper=False)
        W_k = torch.linalg.solve_triangular(M, kb * bc.exp()[..., None], upper=False)
        u = W_v - W_k @ S
        intra = (qc @ kc.transpose(-1, -2)) * decay
        out[:, :, c] = (qc * bc.exp()[..., None]) @ S + intra @ u
        last = bc[..., -1:]
        S = S * last.exp()[..., None] + (kc * (last - bc).exp()[..., None]).transpose(-1, -2) @ u
    return out.reshape(B, H, n * C, -1)[:, :, :T], S


# ---------------------------------------------------------------------------
# Cache: full attention layers use a KV cache (it grows with the length); linear layers use a fixed-size state
# ---------------------------------------------------------------------------


@dataclass
class LinearState:
    """Inference state of one linear attention layer: the recurrent state S and the last K−1 inputs
    of the short convolution. The size does not depend on the sequence length.
    """

    recurrent: torch.Tensor | None = None  # (B, H_v, d_k, d_v), float32
    conv: torch.Tensor | None = None  # (B, conv_dim, K − 1)

    def nbytes(self) -> int:
        return sum(
            t.numel() * t.element_size() for t in (self.recurrent, self.conv) if t is not None
        )


@dataclass
class HybridCache:
    """Inference cache of the hybrid model.

    - `kv`: a `KVCache` preallocated only for the full attention layers (number of layers =
      number of full attention layers). `kv_slot[layer_idx]` gives the slot.
    - `linear[layer_idx]`: the `LinearState` of a linear layer, created in the first forward pass.
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
# Layers
# ---------------------------------------------------------------------------


class GatedRMSNorm(nn.Module):
    """RMSNorm(x) * silu(z): first normalize, then multiply by the gate (the same as HF Qwen3_5RMSNormGated)."""

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
    """The common frame of LinearAttention and GatedDeltaNet
    (the parameter names are the same as in HF Qwen3_5GatedDeltaNet):

    x ─ in_proj_qkv ─ causal short convolution + SiLU ─ split into q, k, v ─ L2-normalize q, k (q also × 1/√d_k)
      ─ in_proj_a ─ g = −exp(A_log) · softplus(a + dt_bias)   (Mamba-2-style log decay for each head)
      ─ in_proj_b ─ β = sigmoid(b)                            (only the delta rule uses it)
      ─ core operator (recurrent / chunked) ─ GatedRMSNorm(·, in_proj_z(x)) ─ out_proj
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
            raise ValueError(f"num_v_heads={num_v_heads} must be an integer multiple of num_k_heads={num_k_heads}")
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
            # Depthwise causal convolution: each channel sees only its own last K positions
            # (the weight shape is the same as nn.Conv1d(groups=C)).
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
        """Mamba-2-style initialization: A ~ U(1, 16); dt ~ logU(1e-3, 0.1), dt_bias = softplus⁻¹(dt)."""
        if not self.use_decay:
            return
        self.A_log.copy_(torch.empty_like(self.A_log).uniform_(1, 16).log())
        dt = torch.exp(
            torch.rand_like(self.dt_bias) * (math.log(0.1) - math.log(1e-3)) + math.log(1e-3)
        )
        self.dt_bias.copy_(dt + torch.log(-torch.expm1(-dt)))

    # ---- Short convolution (with cache) ----
    def _short_conv(self, x: torch.Tensor, state: LinearState | None) -> torch.Tensor:
        """x: (B, C, T) → (B, C, T).

        With a cache, put the last K−1 inputs of the previous segment in front. Then the result is
        the same if the input comes in segments or in one pass.
        """
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
        """x: (B, T, dim) → (B, T, dim). The signature matches `zero.model.Attention`.
        cos/sin are not used (the Qwen3.5 linear layers have no RoPE; the position information comes
        from the recurrence order and the short convolution).

        mode: "auto" (recurrent for T=1, otherwise chunked), "recurrent", "chunk".
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
        if rep > 1:  # Like GQA: many v heads share one set of q/k heads.
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
    """Linear attention: S_t = α_t S_{t−1} + k_t v_tᵀ. With use_decay=False, α_t ≡ 1: this is the
    most basic linear attention (the modern form of Katharopoulos et al. 2020, without the
    denominator normalization, with L2 normalization + output RMSNorm instead).
    """

    use_delta = False


class GatedDeltaNet(_LinearMixer):
    """Gated DeltaNet (Yang, Kautz, Hatamizadeh 2024, arXiv:2412.06464), the linear layer of
    Qwen3-Next / Qwen3.5. With use_decay=False, it becomes DeltaNet.
    """

    use_delta = True


# ---------------------------------------------------------------------------
# Hybrid model
# ---------------------------------------------------------------------------


@dataclass
class HybridConfig(ModelConfig):
    """`ModelConfig` plus the hyperparameters of the linear layers. The field names match the
    HF Qwen3.5 text_config where possible: full_attention_interval, layer_types,
    linear_num_key_heads, linear_num_value_heads, linear_key_head_dim, linear_value_head_dim,
    linear_conv_kernel_dim.
    """

    layer_types: list[str] | None = None  # if None, made from full_attention_interval
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
                f"[model] layer_types has {len(self.layer_types)} items, n_layers={self.n_layers}"
            )
        bad = set(self.layer_types) - {FULL_ATTENTION, LINEAR_ATTENTION}
        if bad:
            raise ConfigError(f"[model] layer_types contains unknown types {sorted(bad)}")
        if self.linear_kind not in ("gated_deltanet", "deltanet", "gated_linear", "linear"):
            raise ConfigError(f"[model] Unknown linear_kind={self.linear_kind!r}")


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
    """Pre-Norm residual block. The token mixer is full attention (zero.model.Attention) or a linear layer."""

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
    """Hybrid model with the same interface as `zero.model.Transformer`:
    forward(tokens, cache=None, start_pos=0) -> logits.
    """

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
        """The same as zero.model.Transformer: N(0, std); the projections that write back to the
        residual stream are also divided by sqrt(2L). The gate parameters have their own initialization.
        """
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
        """Allocate a KV cache only for the full attention layers.

        The states of the linear layers are created in the first forward pass.
        """
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
    """Greedy generation (return the new tokens).

    With use_cache=False, recompute the full sequence at each step. This is only for parity checks.
    """
    # Put the inputs and the cache on the device of the model. (Found on RTX 3090 in 2026-10: the old
    # code always used CPU, which caused an error when the model was on CUDA.)
    device = next(model.parameters()).device
    ids = list(prompt)
    if not use_cache:
        for _ in range(max_new_tokens):
            ids.append(int(model(torch.tensor([ids], device=device))[0, -1].argmax()))
        return ids[len(prompt) :]
    cache = model.new_cache(1, len(prompt) + max_new_tokens, device=device)
    logits = model(torch.tensor([ids], device=device), cache, 0)  # prefill: chunked form
    for _ in range(max_new_tokens):
        nxt = int(logits[0, -1].argmax())
        ids.append(nxt)
        # decode: recurrent form, O(1) state
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
    """Inference cache bytes for one sequence (the ledger of Chapter 21, plus the linear layers):

    - full attention layer: 2 × number of KV heads × head_dim × sequence length × kv_bytes
      (grows linearly with the length);
    - linear layer: H_v × d_k × d_v × state_bytes (recurrent state, usually FP32)
      + conv_dim × (K−1) × kv_bytes (does not depend on the length).
    """
    n_full = sum(t == FULL_ATTENTION for t in layer_types)
    n_lin = len(layer_types) - n_full
    kv = n_full * 2 * n_kv_heads * head_dim * seq_len * kv_bytes
    state = n_lin * (
        linear_num_value_heads * linear_key_head_dim * linear_value_head_dim * state_bytes
        + conv_dim * max(conv_kernel - 1, 0) * kv_bytes
    )
    return {"kv": kv, "linear_state": state, "total": kv + state}
