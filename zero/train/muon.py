"""Muon optimizer: Muon for the hidden-layer matrices, AdamW for all other parameters
(the verified result in "Frontier and consensus" of Chapter 12).

The Muon update of one weight matrix W (Jordan et al. 2024, https://kellerjordan.github.io/posts/muon/):

    M ← μ·M + G                      # momentum (G is the gradient)
    U ← NS5(μ·M + G)                 # Nesterov; NS5 = 5 Newton–Schulz iterations that "orthogonalize" the matrix:
                                      #   G = U Σ Vᵀ  →  about U Vᵀ (all singular values move to about 1)
    W ← W·(1 − η·λ) − η·s·U          # decoupled weight decay + update; s is the scale factor

Intuition: AdamW scales each element separately. Muon makes the update of the full matrix "flat",
so the step size is the same in all directions. Then the model also learns in the directions of
small singular values (directions that other gradient directions often hide).

There are two common choices for the scale factor s:
- "rms" (default): s = 0.2·√max(m, n). This makes the update RMS ≈ 0.2, about the same as a typical
  AdamW update RMS. Thus you can keep the AdamW learning rate and weight decay
  (Moonlight / Kimi K2 do this, arXiv:2502.16982; GLM-4.5 uses 0.2, DeepSeek-V4 uses 0.18).
- "spectral": s = √max(1, m/n) (the Keller Jordan reference implementation). You must tune the
  learning rate separately. It is usually about 10 times larger than for AdamW.

Which parameters use Muon: the 2D weights in the Transformer blocks (attention wq/wk/wv/wo, the
three SwiGLU matrices). The embedding, lm_head, and RMSNorm weights use AdamW. The technical
reports of Kimi K2, GLM-4.5, and DeepSeek-V4 all use these groups.

Distributed training:
- DDP: each GPU gets the full gradient after the all-reduce. Each GPU computes the same NS5,
  so the results are the same. The only cost is the repeated computation.
- FSDP: the parameters are sharded, but NS5 needs the full matrix. This implementation does not
  support FSDP (it raises an error on a DTensor). DeepSeek-V4 puts the matrices into ZeRO buckets,
  one matrix at a time. For a 0.7B model on one machine with 8 GPUs, DDP is sufficient.
BF16 NS5 on CUDA is verified on one RTX 3090 (2026-10, see runs/2026-10-01-gpu0-check/:
the relative difference from CPU FP32 is about 2%, and the training loss with `optim.name = "muon"`
decreases normally). The numerical consistency with DDP is verified on 2×RTX 3090 (PCIe):
after 50 steps, the parameters on the two GPUs are bitwise identical. The loss difference from an
equivalent 1-GPU run has the same order of magnitude as a 1-GPU control with a different sum order.

Connection to the trainer: set `optim.name = "muon"` in the config. OptimConfig has the field
`name: str = "adamw"`, and trainer.build_optimizer calls build_muon_optimizer(model, cfg, device)
when the name is "muon" (see the Chapter 12 README).
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn

# Tuned quintic coefficients from the Keller Jordan reference implementation. They do not converge to
# the exact UVᵀ. They move the singular values into ~[0.7, 1.2] in the fewest steps.
NS_COEFFS = (3.4445, -4.7750, 2.0315)


def zeropower_via_newtonschulz5(
    G: torch.Tensor, steps: int = 5, dtype: torch.dtype | None = None
) -> torch.Tensor:
    """Approximately orthogonalize the matrix G (≈ U Vᵀ).

    Default dtype: bfloat16 on CUDA (as in the reference implementation), float32 on CPU.
    """
    assert G.ndim == 2, "Muon handles only 2D matrices"
    a, b, c = NS_COEFFS
    if dtype is None:
        dtype = torch.bfloat16 if G.is_cuda else torch.float32
    X = G.to(dtype)
    transposed = G.size(0) > G.size(1)
    if transposed:  # Make X Xᵀ the smaller of the two square matrices.
        X = X.T
    X = X / (X.norm() + 1e-7)  # The iteration converges only if the largest singular value is ≤ 1.
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X.to(G.dtype)


class MuonAdamW(torch.optim.Optimizer):
    """One optimizer object for two types of parameter groups.

    Groups with `use_muon=True` use Muon. All other groups use AdamW.
    Thus the trainer, the learning-rate schedule (it writes group["lr"] for each group), and the
    checkpoint (state_dict) do not need to know about two optimizers.
    """

    def __init__(
        self,
        param_groups: list[dict[str, Any]],
        lr: float = 3e-4,
        weight_decay: float = 0.1,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        adjust: str = "rms",
        rms_scale: float = 0.2,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
    ) -> None:
        if adjust not in ("rms", "spectral"):
            raise ValueError(f"adjust must be rms or spectral, got {adjust!r}")
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            adjust=adjust,
            rms_scale=rms_scale,
            betas=betas,
            eps=eps,
            use_muon=False,
        )
        super().__init__(param_groups, defaults)
        for g in self.param_groups:
            if g["use_muon"]:
                for p in g["params"]:
                    if p.ndim != 2:
                        raise ValueError(f"A Muon group can contain only 2D matrices, got shape {tuple(p.shape)}")

    @staticmethod
    def _scale(shape: torch.Size, adjust: str, rms_scale: float) -> float:
        m, n = shape
        if adjust == "rms":
            # The NS5 output has RMS ≈ 1/√max(m,n). Multiply to get RMS ≈ rms_scale.
            return rms_scale * math.sqrt(max(m, n))
        return math.sqrt(max(1.0, m / n))

    @torch.no_grad()
    def step(self, closure: Any = None) -> Any:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for g in self.param_groups:
            lr, wd = g["lr"], g["weight_decay"]
            for p in g["params"]:
                if p.grad is None:
                    continue
                if type(p).__name__ == "DTensor" or type(p.grad).__name__ == "DTensor":
                    raise NotImplementedError("Muon needs the full matrix. Sharded FSDP parameters are not supported. Use DDP")
                grad = p.grad
                state = self.state[p]
                if g["use_muon"]:
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    buf = state["momentum_buffer"]
                    buf.mul_(g["momentum"]).add_(grad)
                    update = grad.add(buf, alpha=g["momentum"]) if g["nesterov"] else buf
                    update = zeropower_via_newtonschulz5(update, g["ns_steps"])
                    update = update * self._scale(p.shape, g["adjust"], g["rms_scale"])
                    p.mul_(1 - lr * wd)
                    p.add_(update, alpha=-lr)
                else:  # Standard AdamW (decoupled weight decay).
                    beta1, beta2 = g["betas"]
                    if "step" not in state:
                        state["step"] = torch.zeros((), dtype=torch.float32)
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                    state["step"] += 1
                    t = float(state["step"])
                    state["exp_avg"].lerp_(grad, 1 - beta1)
                    state["exp_avg_sq"].mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                    denom = (state["exp_avg_sq"] / (1 - beta2**t)).sqrt_().add_(g["eps"])
                    p.mul_(1 - lr * wd)
                    p.addcdiv_(state["exp_avg"], denom, value=-lr / (1 - beta1**t))
        return loss


def split_params_for_muon(
    model: nn.Module, decay_embeddings: bool = False
) -> dict[str, list[nn.Parameter]]:
    """Make groups by parameter name.

    2D weights in the blocks → muon; embedding / lm_head → adam (decay is optional);
    1D weights (norm) → adam without decay.
    """
    groups: dict[str, list[nn.Parameter]] = {"muon": [], "adam_decay": [], "adam_no_decay": []}
    seen: set[int] = set()
    for name, p in model.named_parameters():
        if not p.requires_grad or id(p) in seen:  # A shared embedding occurs only once.
            continue
        seen.add(id(p))
        is_embedding = "tok_emb" in name or "lm_head" in name
        if p.ndim == 2 and not is_embedding:
            groups["muon"].append(p)
        elif p.ndim == 2 and decay_embeddings:
            groups["adam_decay"].append(p)
        else:
            groups["adam_no_decay"].append(p)
    return groups


def build_muon_optimizer(
    model: nn.Module, cfg: Any, device: torch.device | None = None
) -> MuonAdamW:
    """Same signature as trainer.build_optimizer.

    cfg is an OptimConfig: lr, weight_decay, beta1, beta2, eps, decay_embeddings.
    Optional fields: muon_momentum / muon_ns_steps / muon_adjust / muon_rms_scale
    (if a field is missing, use the default value).
    """
    groups = split_params_for_muon(model, getattr(cfg, "decay_embeddings", False))
    wd = cfg.weight_decay
    param_groups = [
        {"params": groups["muon"], "use_muon": True, "weight_decay": wd},
        {"params": groups["adam_decay"], "use_muon": False, "weight_decay": wd},
        {"params": groups["adam_no_decay"], "use_muon": False, "weight_decay": 0.0},
    ]
    param_groups = [g for g in param_groups if g["params"]]
    return MuonAdamW(
        param_groups,
        lr=cfg.lr,
        weight_decay=wd,
        momentum=getattr(cfg, "muon_momentum", 0.95),
        ns_steps=getattr(cfg, "muon_ns_steps", 5),
        adjust=getattr(cfg, "muon_adjust", "rms"),
        rms_scale=getattr(cfg, "muon_rms_scale", 0.2),
        betas=(cfg.beta1, cfg.beta2),
        eps=cfg.eps,
    )
