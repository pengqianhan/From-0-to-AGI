"""Muon 优化器：隐藏层的矩阵用 Muon，其余参数用 AdamW（对应第 12 章"前沿与共识"的核实结果）。

Muon（Jordan et al. 2024，https://kellerjordan.github.io/posts/muon/）对一个权重矩阵 W 的更新：

    M ← μ·M + G                      # 动量（G 是梯度）
    U ← NS5(μ·M + G)                 # Nesterov；NS5 = 5 步 Newton–Schulz 迭代，把矩阵"正交化"：
                                      #   G = U Σ Vᵀ  →  约等于 U Vᵀ（所有奇异值拉到 1 附近）
    W ← W·(1 − η·λ) − η·s·U          # 解耦权重衰减 + 更新；s 是尺度因子

直觉：AdamW 对每个元素单独缩放；Muon 把整个矩阵的更新"拉平"成各个方向步长相同，
小奇异值方向（梯度里容易被淹没的方向）也能学到东西。

尺度因子 s 有两种常见取法：
- "rms"（默认）：s = 0.2·√max(m, n)，让更新的 RMS ≈ 0.2，与 AdamW 的典型更新 RMS 相当，
  于是可以直接沿用 AdamW 的学习率与权重衰减（Moonlight / Kimi K2 的做法，arXiv:2502.16982；
  GLM-4.5 取 0.2，DeepSeek-V4 取 0.18）；
- "spectral"：s = √max(1, m/n)（Keller Jordan 参考实现），学习率要单独调，通常比 AdamW 大一个量级。

哪些参数用 Muon：Transformer 块里的二维权重（注意力 wq/wk/wv/wo、SwiGLU 三个矩阵）。
embedding、lm_head、RMSNorm 权重用 AdamW——Kimi K2、GLM-4.5、DeepSeek-V4 的技术报告都是这样分组的。

分布式：
- DDP：每张卡拿到的是 all-reduce 之后的完整梯度，各自算同样的 NS5，结果一致，只是有重复计算；
- FSDP：参数被切片，NS5 需要完整矩阵，本实现不支持（遇到 DTensor 直接报错）。
  DeepSeek-V4 的做法是按矩阵把 ZeRO 分桶，单机 8 卡、0.7B 模型用 DDP 就够。
以上 GPU 路径（CUDA 上 BF16 的 NS5、DDP 下的数值一致性）**尚未在 GPU 上验证**。

接入训练器（zero/train/trainer.py 不在本章改动范围内，接入需要两处小改，见第 12 章 README）：
    OptimConfig 加字段 `name: str = "adamw"`；build_optimizer 开头加一行
    `if cfg.name == "muon": return build_muon_optimizer(model, cfg, device)`
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn

# Keller Jordan 参考实现里调好的五次多项式系数：不追求收敛到精确的 UVᵀ，而是用最少的步数把奇异值推到 ~[0.7, 1.2]
NS_COEFFS = (3.4445, -4.7750, 2.0315)


def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5, dtype: torch.dtype | None = None) -> torch.Tensor:
    """把矩阵 G 近似正交化（≈ U Vᵀ）。dtype 默认：CUDA 上 bfloat16（参考实现），CPU 上 float32。"""
    assert G.ndim == 2, "Muon 只处理二维矩阵"
    a, b, c = NS_COEFFS
    if dtype is None:
        dtype = torch.bfloat16 if G.is_cuda else torch.float32
    X = G.to(dtype)
    transposed = G.size(0) > G.size(1)
    if transposed:  # 让 X Xᵀ 是较小的那个方阵
        X = X.T
    X = X / (X.norm() + 1e-7)  # 先把最大奇异值压到 ≤ 1，迭代才收敛
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X.to(G.dtype)


class MuonAdamW(torch.optim.Optimizer):
    """一个优化器对象管两类参数组：`use_muon=True` 的组走 Muon，其余走 AdamW。

    这样训练器、学习率调度（按组写 group["lr"]）、checkpoint（state_dict）都不用区分两种优化器。
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
            raise ValueError(f"adjust 只能是 rms/spectral，当前 {adjust!r}")
        defaults = dict(
            lr=lr, weight_decay=weight_decay, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps,
            adjust=adjust, rms_scale=rms_scale, betas=betas, eps=eps, use_muon=False,
        )
        super().__init__(param_groups, defaults)
        for g in self.param_groups:
            if g["use_muon"]:
                for p in g["params"]:
                    if p.ndim != 2:
                        raise ValueError(f"Muon 组里只能放二维矩阵，拿到形状 {tuple(p.shape)}")

    @staticmethod
    def _scale(shape: torch.Size, adjust: str, rms_scale: float) -> float:
        m, n = shape
        if adjust == "rms":
            return rms_scale * math.sqrt(max(m, n))  # NS5 输出的 RMS ≈ 1/√max(m,n)，乘回来得到 RMS ≈ rms_scale
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
                    raise NotImplementedError("Muon 需要完整矩阵，FSDP 切片参数暂不支持；请用 DDP")
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
                else:  # 标准 AdamW（解耦权重衰减）
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


def split_params_for_muon(model: nn.Module, decay_embeddings: bool = False) -> dict[str, list[nn.Parameter]]:
    """按名字分组：块内二维权重 → muon；embedding / lm_head → adam（是否衰减可选）；一维（norm）→ adam 不衰减。"""
    groups: dict[str, list[nn.Parameter]] = {"muon": [], "adam_decay": [], "adam_no_decay": []}
    seen: set[int] = set()
    for name, p in model.named_parameters():
        if not p.requires_grad or id(p) in seen:  # 共享 embedding 只出现一次
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


def build_muon_optimizer(model: nn.Module, cfg: Any, device: torch.device | None = None) -> MuonAdamW:
    """与 trainer.build_optimizer 同签名。cfg 是 OptimConfig：lr、weight_decay、beta1、beta2、eps、decay_embeddings；
    可选字段 muon_momentum / muon_ns_steps / muon_adjust / muon_rms_scale（没有就用默认值）。"""
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
