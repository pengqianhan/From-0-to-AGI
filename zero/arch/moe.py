"""Mixture-of-experts (MoE) feed-forward layer (Chapter 24): routing, top-k, shared experts,
auxiliary loss, auxiliary-loss-free bias balancing, load statistics.

Experiment module for Part 5. **The main-line model does not use it** (the main line is a dense
model, see GOAL.md 3.3).

Structure (written to match the HF implementations of DeepSeek-V3 / Qwen3-MoE / Mixtral):

    y = Σ_{shared expert s} FFN_s(x) + Σ_{i ∈ TopK} g_i · FFN_i(x)

- Router: one linear layer `dim → n_experts`. The score function is softmax (Mixtral, Qwen3-MoE,
  gpt-oss) or sigmoid (DeepSeek-V3, GLM-4.5, Kimi K2).
- Expert selection: top-k of `score + bias b`. The bias changes only "which experts are selected".
  It does not change the gate weights g (DeepSeek-V3, Eq. 16).
- Gate weights: the original scores of the selected experts, optionally normalized
  (`norm_topk_prob`), then multiplied by `routed_scaling_factor`.
- Experts: each expert is one SwiGLU. The weights of all experts are stacked into 3D tensors (E, …).
  In the forward pass, the code sorts the tokens by expert and does one matrix multiplication for
  each segment (a loop on CPU; on GPU, replace it with grouped GEMM, see "Production practice" below).
- Two methods of load balancing. You can use both at the same time:
  1. Auxiliary loss (Switch / GShard style): L_aux = α · Σ_i f_i · P_i,
     f_i = N/(K·T) · (number of tokens sent to expert i), P_i = mean routing probability over
     the T tokens. With perfect balance, L_aux = α.
  2. Auxiliary-loss-free (DeepSeek-V3): after each step, update the bias from the load of the
     full batch: b_i ← b_i + γ · sign(mean load − load_i). The bias of an overloaded expert
     decreases, and the bias of an underloaded expert increases.
- Capacity factor: optional. Each expert processes a maximum of ⌈cf · T · K / E⌉ tokens.
  The layer drops the extra tokens: the output of that expert for the token is 0, and the residual
  passes through as usual. The default None = no dropped tokens (dropless).

Production practice (this file has only two goals: easy to read and correct;
the paths below are **not verified on GPU yet**):
- Expert computation: on GPU, use grouped GEMM / MegaBlocks-style block-sparse matrix
  multiplication to compute all experts in one kernel.
- Expert parallelism (EP): the experts are on many GPUs, and an all-to-all sends the tokens to
  them and gets the results back. With distributed training, `update_bias` must all_reduce the
  load first (the code does this, but it is not verified on many GPUs).

`tests/test_arch_moe.py` checks these points: with 1 expert + top-1, the layer is identical to
`zero.model.SwiGLU`; the sorted-segment implementation agrees with a naive per-token loop;
the auxiliary loss agrees with a hand computation; the bias moves in the correct direction and
can make a skewed routing flat.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from zero.config import ModelConfig
from zero.model import SwiGLU, Transformer, cross_entropy_loss

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class MoEConfig:
    """The field names match the HF configs where possible:
    n_experts ↔ n_routed_experts / num_experts / num_local_experts,
    top_k ↔ num_experts_per_tok, expert_dim ↔ moe_intermediate_size,
    score_func ↔ scoring_func, aux_loss_coef ↔ router_aux_loss_coef / aux_loss_alpha.
    """

    dim: int
    n_experts: int = 8
    top_k: int = 2
    expert_dim: int = 256
    n_shared_experts: int = 0
    shared_expert_dim: int | None = None  # default n_shared_experts × expert_dim (as in DeepSeek)
    score_func: Literal["softmax", "sigmoid"] = "softmax"
    norm_topk_prob: bool = True  # normalize the weights of the selected experts again to sum to 1
    routed_scaling_factor: float = 1.0  # 2.5 in DeepSeek-V3
    aux_loss_coef: float = 0.0  # α; 0 = no auxiliary loss
    bias_update_speed: float = 0.0  # γ; 0 = no auxiliary-loss-free bias balancing
    capacity_factor: float | None = None  # None = dropless
    init_std: float = 0.02

    def __post_init__(self) -> None:
        if not 1 <= self.top_k <= self.n_experts:
            raise ValueError(f"top_k={self.top_k} must be in 1..n_experts={self.n_experts}")
        if self.shared_expert_dim is None:
            self.shared_expert_dim = self.n_shared_experts * self.expert_dim

    @classmethod
    def from_model_config(cls, mc: ModelConfig, **kw) -> MoEConfig:
        kw.setdefault("init_std", mc.init_std)
        return cls(dim=mc.dim, **kw)

    def params(self) -> dict[str, int]:
        """Parameter count of one MoE layer: total (all experts) and active (used for each token)."""
        per_expert = 3 * self.dim * self.expert_dim
        shared = 3 * self.dim * self.shared_expert_dim if self.shared_expert_dim else 0
        router = self.dim * self.n_experts
        return {
            "total": self.n_experts * per_expert + shared + router,
            "active": self.top_k * per_expert + shared + router,
        }


# ---------------------------------------------------------------------------
# Routing and loss (pure functions, easy to test and to explain)
# ---------------------------------------------------------------------------


def router_scores(logits: torch.Tensor, score_func: str) -> torch.Tensor:
    """logits: (T, E) → scores (T, E), computed in float32."""
    logits = logits.float()
    if score_func == "softmax":
        return logits.softmax(dim=-1)
    if score_func == "sigmoid":
        return logits.sigmoid()
    raise ValueError(f"Unknown score_func: {score_func}")


def select_experts(
    scores: torch.Tensor,
    top_k: int,
    bias: torch.Tensor | None = None,
    norm_topk_prob: bool = True,
    scaling: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Take the top-k of (score + bias). The gate weights are the original scores
    (the bias only controls "which experts are selected").

    Return idx (T, K) int64 and weight (T, K) float32.
    """
    choose = scores if bias is None else scores + bias
    idx = choose.topk(top_k, dim=-1).indices
    weight = scores.gather(-1, idx)
    if norm_topk_prob:
        weight = weight / weight.sum(dim=-1, keepdim=True).clamp_min(1e-20)
    return idx, weight * scaling


def aux_balance_loss(
    scores: torch.Tensor, idx: torch.Tensor, n_experts: int, coef: float
) -> torch.Tensor:
    """Switch / GShard-style load-balancing loss (same form as DeepSeek-V3 Eq. 17–20,
    over the T tokens of this batch):

        f_i = N / (K·T) · #{t : i ∈ TopK_t}      (not differentiable, only a "weight")
        P_i = 1/T · Σ_t s'_{i,t}, s' = scores normalized per row (softmax is already normalized)
        L   = α · Σ_i f_i · P_i                   (with perfect balance f_i = 1, P_i = 1/N, L = α)

    The gradient goes only through P: if an expert gets more tokens (large f_i), the loss pushes
    its routing probability down more strongly.
    Note: in the HF Mixtral / Qwen3-MoE implementations, f is not divided by K. Their value is
    K times the value here, so you cannot copy the coefficient directly.
    """
    T, K = idx.shape
    probs = scores / scores.sum(dim=-1, keepdim=True)
    counts = torch.bincount(idx.reshape(-1), minlength=n_experts).to(probs.dtype)
    f = counts * n_experts / (K * T)
    P = probs.mean(dim=0)
    return coef * (f * P).sum()


# ---------------------------------------------------------------------------
# MoE feed-forward layer
# ---------------------------------------------------------------------------


class MoEFFN(nn.Module):
    """An MoE feed-forward layer that can replace `zero.model.SwiGLU` directly:
    forward(x: (..., dim)) -> (..., dim).

    After the forward pass, you can read:
    - `last_aux_loss`: the auxiliary loss of this forward pass (a tensor with gradient in training
      when aux_loss_coef > 0, otherwise 0).
    - `last_load`: the number of tokens for each expert in this forward pass (E,) (before drops);
      `last_dropped`: the number of assignments that the capacity limit dropped.
    - `load_accum`: the load accumulated since the last `update_bias()` (it accumulates in training).
    """

    def __init__(self, cfg: MoEConfig) -> None:
        super().__init__()
        self.cfg = cfg
        E, d, h = cfg.n_experts, cfg.dim, cfg.expert_dim
        self.router = nn.Linear(d, E, bias=False)
        # SwiGLU weights stacked by expert: x (n, d) @ w_gate[e] (d, h) → (n, h)
        self.w_gate = nn.Parameter(torch.empty(E, d, h))
        self.w_up = nn.Parameter(torch.empty(E, d, h))
        self.w_down = nn.Parameter(torch.empty(E, h, d))
        self.shared = SwiGLU(d, cfg.shared_expert_dim) if cfg.shared_expert_dim else None
        # Bias for auxiliary-loss-free balancing: not a parameter (no gradient), but it must be in
        # the state_dict (a resume continues to use it).
        self.register_buffer("expert_bias", torch.zeros(E))
        self.register_buffer("load_accum", torch.zeros(E), persistent=False)
        self.last_aux_loss: torch.Tensor = torch.zeros(())
        self.last_load: torch.Tensor = torch.zeros(E)
        self.last_dropped: int = 0
        self.init_weights()

    @torch.no_grad()
    def init_weights(self, std: float | None = None, out_std: float | None = None) -> None:
        std = self.cfg.init_std if std is None else std
        out_std = std if out_std is None else out_std
        nn.init.normal_(self.router.weight, std=std)
        nn.init.normal_(self.w_gate, std=std)
        nn.init.normal_(self.w_up, std=std)
        nn.init.normal_(self.w_down, std=out_std)
        if self.shared is not None:
            nn.init.normal_(self.shared.w_gate.weight, std=std)
            nn.init.normal_(self.shared.w_up.weight, std=std)
            nn.init.normal_(self.shared.w_down.weight, std=out_std)

    def capacity(self, n_tokens: int) -> int | None:
        cf = self.cfg.capacity_factor
        if cf is None:
            return None
        return max(1, math.ceil(cf * n_tokens * self.cfg.top_k / self.cfg.n_experts))

    def route(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """x: (T, d) → (scores (T,E), idx (T,K), weight (T,K))."""
        c = self.cfg
        scores = router_scores(self.router(x), c.score_func)
        # The bias always takes part in the selection (it stays 0 if it is not updated).
        # DeepSeek-V3 set γ to 0 for the last 500B tokens, but continued to use the learned bias.
        idx, weight = select_experts(
            scores, c.top_k, self.expert_bias, c.norm_topk_prob, c.routed_scaling_factor
        )
        return scores, idx, weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c = self.cfg
        shape = x.shape
        x = x.reshape(-1, c.dim)
        T = x.shape[0]
        scores, idx, weight = self.route(x)

        # ---- Load statistics and loss ----
        counts = torch.bincount(idx.reshape(-1), minlength=c.n_experts)
        self.last_load = counts.detach().float()
        if self.training:
            self.load_accum += self.last_load
        if self.training and c.aux_loss_coef > 0:
            self.last_aux_loss = aux_balance_loss(scores, idx, c.n_experts, c.aux_loss_coef)
        else:
            self.last_aux_loss = torch.zeros((), device=x.device)

        # ---- Dispatch: sort the T·K (token, expert) assignments by expert; each expert gets one contiguous segment ----
        flat_e = idx.reshape(-1)
        flat_t = torch.arange(T, device=x.device).repeat_interleave(c.top_k)
        flat_w = weight.reshape(-1)
        order = torch.argsort(flat_e, stable=True)  # stable sort: keep the token order inside each expert
        flat_e, flat_t, flat_w = flat_e[order], flat_t[order], flat_w[order]
        cap = self.capacity(T)
        if cap is not None:
            # Rank of each assignment inside its expert (from 0). Drop the ranks ≥ capacity.
            starts = torch.cumsum(counts, 0) - counts
            rank = torch.arange(flat_e.numel(), device=x.device) - starts[flat_e]
            keep = rank < cap
            self.last_dropped = int((~keep).sum())
            flat_e, flat_t, flat_w = flat_e[keep], flat_t[keep], flat_w[keep]
            counts = torch.bincount(flat_e, minlength=c.n_experts)
        else:
            self.last_dropped = 0

        # ---- Expert computation + combine ----
        # Loop over the experts. It runs with CUDA + BF16 (RTX 3090, 2026-10, see runs/2026-10-01-gpu0-check/),
        # but it is slow. On GPU, replace it with grouped GEMM (not implemented, not verified on GPU yet).
        out = torch.zeros_like(x)
        xs = x[flat_t]
        for e, seg in enumerate(
            torch.split(torch.arange(flat_t.numel(), device=x.device), counts.tolist())
        ):
            if seg.numel() == 0:
                continue
            h = xs[seg]
            y = (F.silu(h @ self.w_gate[e]) * (h @ self.w_up[e])) @ self.w_down[e]
            # With BF16 autocast, y is BF16 and out is FP32 (same precision as x). index_add_ needs the
            # same type for both: first convert to the precision of out, then add (no-op in FP32 training).
            # Found on RTX 3090 in 2026-10, see runs/2026-10-01-gpu0-check/.
            out.index_add_(0, flat_t[seg], (y * flat_w[seg, None].to(y.dtype)).to(out.dtype))
        if self.shared is not None:
            out = out + self.shared(x)
        return out.reshape(shape)

    @torch.no_grad()
    def update_bias(self) -> None:
        """Auxiliary-loss-free balancing (DeepSeek-V3, Section 2.1.2). Call it once after each training step.

        b_i ← b_i + γ · sign(mean load − load_i): the bias of an overloaded expert (load > mean)
        decreases by γ, and the bias of an underloaded expert increases by γ.
        The load comes from `load_accum`, accumulated since the last call (with gradient
        accumulation, it covers the full global batch).
        """
        g = self.cfg.bias_update_speed
        if g <= 0:
            self.load_accum.zero_()
            return
        load = self.load_accum.clone()
        if dist.is_available() and dist.is_initialized():  # not verified on many GPUs yet
            dist.all_reduce(load)
        self.expert_bias += g * torch.sign(load.mean() - load)
        self.load_accum.zero_()


# ---------------------------------------------------------------------------
# Put it into the Transformer
# ---------------------------------------------------------------------------


class MoETransformer(Transformer):
    """The MoE version of the main-line `Transformer`. The first `first_dense` layers keep the
    dense SwiGLU. The FFN of all other layers becomes MoEFFN
    (first_k_dense_replace = 3 in DeepSeek-V3, 1 in Kimi K2, 3 in GLM-4.5).

    Two differences from the main line: `loss()` automatically adds the auxiliary losses of all
    layers, and you call `after_step()` after each optimizer step to update the auxiliary-loss-free
    bias. (The main-line `zero.train.trainer.Trainer` does not call `after_step` yet. Connect it in
    step 2 if you need it to train an MoE. The small experiments of this chapter use their own
    training loop.)
    """

    def __init__(self, config: ModelConfig, moe_cfg: MoEConfig, first_dense: int = 0) -> None:
        super().__init__(config)
        self.moe_cfg = moe_cfg
        self.first_dense = first_dense
        out_std = config.init_std / math.sqrt(2 * config.n_layers)
        for i, layer in enumerate(self.layers):
            if i >= first_dense:
                layer.ffn = MoEFFN(moe_cfg)
                layer.ffn.init_weights(config.init_std, out_std)

    def moe_layers(self) -> list[MoEFFN]:
        return [layer.ffn for layer in self.layers if isinstance(layer.ffn, MoEFFN)]

    def aux_loss(self) -> torch.Tensor:
        losses = [m.last_aux_loss for m in self.moe_layers()]
        return torch.stack(losses).sum() if losses else torch.zeros(())

    def loss(
        self, tokens: torch.Tensor, targets: torch.Tensor, ignore_index: int = -100
    ) -> torch.Tensor:
        logits = self(tokens)
        return cross_entropy_loss(logits, targets, ignore_index) + self.aux_loss()

    @torch.no_grad()
    def after_step(self) -> None:
        for m in self.moe_layers():
            m.update_bias()

    def load_stats(self) -> torch.Tensor:
        """Fraction of tokens for each expert in each layer in the last forward pass.

        Shape (number of MoE layers, E). Each row sums to 1.
        """
        loads = torch.stack([m.last_load for m in self.moe_layers()])
        return loads / loads.sum(dim=-1, keepdim=True).clamp_min(1)

    def param_counts(self) -> dict[str, int]:
        """total: all parameters (a shared embedding counts once).
        active: the parameters that each token actually uses (experts that are not selected do not
        count; the embedding lookup counts, as in most model cards).
        """
        total = self.num_params()
        inactive = sum(
            (m.cfg.n_experts - m.cfg.top_k) * 3 * m.cfg.dim * m.cfg.expert_dim
            for m in self.moe_layers()
        )
        return {"total": total, "active": total - inactive}


def moe_transformer(
    model_cfg: ModelConfig, moe_cfg: MoEConfig, first_dense: int = 0
) -> MoETransformer:
    return MoETransformer(model_cfg, moe_cfg, first_dense)
