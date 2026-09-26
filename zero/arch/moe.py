"""混合专家（MoE）前馈层（第 24 章）：路由、top-k、共享专家、辅助损失、无辅助损失的偏置均衡、负载统计。

第五部分的实验模块，**不用于主线模型**（主线是稠密模型，见 GOAL.md 3.3）。

结构（与 DeepSeek-V3 / Qwen3-MoE / Mixtral 的 HF 实现对得上的写法）：

    y = Σ_{共享专家 s} FFN_s(x) + Σ_{i ∈ TopK} g_i · FFN_i(x)

- 路由器：一个 `dim → n_experts` 的线性层，打分函数可选 softmax（Mixtral、Qwen3-MoE、gpt-oss）
  或 sigmoid（DeepSeek-V3、GLM-4.5、Kimi K2）；
- 选专家：按 `分数 + 偏置 b` 取 top-k（偏置只影响"选谁"，不影响门控权重 g，DeepSeek-V3 式 16）；
- 门控权重：被选中专家的原始分数，可选归一化（`norm_topk_prob`），再乘 `routed_scaling_factor`；
- 专家：每个专家是一个 SwiGLU。权重按专家堆叠成 (E, …) 的三维张量，前向时把 token 按专家排序、
  分段做矩阵乘（CPU 上用循环；GPU 上应换成 grouped GEMM，见下方"生产实践"）；
- 负载均衡两种做法，可以同时开：
  1. 辅助损失（Switch / GShard 式）：L_aux = α · Σ_i f_i · P_i，
     f_i = N/(K·T) · (分到专家 i 的 token 数)，P_i = 路由概率在 T 个 token 上的平均；完全均衡时 L_aux = α；
  2. 无辅助损失（DeepSeek-V3）：每步结束后按整批的负载更新偏置
     b_i ← b_i + γ · sign(平均负载 − 负载_i)，过载的专家偏置变小、欠载的变大；
- 容量因子（capacity factor）：可选。每个专家最多处理 ⌈cf · T · K / E⌉ 个 token，多出的被丢弃
  （这个 token 在该专家上的输出记为 0，残差照常传下去）。默认 None = 不丢 token（dropless）。

生产实践（本文件只追求可读和正确，以下路径**尚未在 GPU 上验证**）：
- 专家计算：GPU 上用 grouped GEMM / MegaBlocks 式块稀疏矩阵乘，一次 kernel 算完所有专家；
- 专家并行（expert parallelism, EP）：专家分布在多张卡上，token 经 all-to-all 发过去再收回来；
  `update_bias` 在分布式下要先 all_reduce 负载（这里已写，但未在多卡上验证）。

`tests/test_arch_moe.py` 保证：1 个专家 + top-1 时与 `zero.model.SwiGLU` 完全一致；排序分段的实现与逐 token
的朴素循环一致；辅助损失与手算一致；偏置朝正确方向更新并能把一个偏斜的路由拉平。
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
# 配置
# ---------------------------------------------------------------------------


@dataclass
class MoEConfig:
    """字段名尽量与 HF 配置对应：n_experts ↔ n_routed_experts / num_experts / num_local_experts，
    top_k ↔ num_experts_per_tok，expert_dim ↔ moe_intermediate_size，
    score_func ↔ scoring_func，aux_loss_coef ↔ router_aux_loss_coef / aux_loss_alpha。"""

    dim: int
    n_experts: int = 8
    top_k: int = 2
    expert_dim: int = 256
    n_shared_experts: int = 0
    shared_expert_dim: int | None = None  # 默认 n_shared_experts × expert_dim（DeepSeek 的写法）
    score_func: Literal["softmax", "sigmoid"] = "softmax"
    norm_topk_prob: bool = True  # 被选中专家的权重再归一化到和为 1
    routed_scaling_factor: float = 1.0  # DeepSeek-V3 为 2.5
    aux_loss_coef: float = 0.0  # α；0 表示不加辅助损失
    bias_update_speed: float = 0.0  # γ；0 表示不做无辅助损失的偏置均衡
    capacity_factor: float | None = None  # None = dropless
    init_std: float = 0.02

    def __post_init__(self) -> None:
        if not 1 <= self.top_k <= self.n_experts:
            raise ValueError(f"top_k={self.top_k} 必须在 1..n_experts={self.n_experts} 之间")
        if self.shared_expert_dim is None:
            self.shared_expert_dim = self.n_shared_experts * self.expert_dim

    @classmethod
    def from_model_config(cls, mc: ModelConfig, **kw) -> MoEConfig:
        kw.setdefault("init_std", mc.init_std)
        return cls(dim=mc.dim, **kw)

    def params(self) -> dict[str, int]:
        """一层 MoE 的参数量：total（全部专家）与 active（每个 token 实际用到的）。"""
        per_expert = 3 * self.dim * self.expert_dim
        shared = 3 * self.dim * self.shared_expert_dim if self.shared_expert_dim else 0
        router = self.dim * self.n_experts
        return {
            "total": self.n_experts * per_expert + shared + router,
            "active": self.top_k * per_expert + shared + router,
        }


# ---------------------------------------------------------------------------
# 路由与损失（纯函数，方便单测和讲解）
# ---------------------------------------------------------------------------


def router_scores(logits: torch.Tensor, score_func: str) -> torch.Tensor:
    """logits: (T, E) → 分数 (T, E)，在 float32 上算。"""
    logits = logits.float()
    if score_func == "softmax":
        return logits.softmax(dim=-1)
    if score_func == "sigmoid":
        return logits.sigmoid()
    raise ValueError(f"未知的 score_func: {score_func}")


def select_experts(
    scores: torch.Tensor,
    top_k: int,
    bias: torch.Tensor | None = None,
    norm_topk_prob: bool = True,
    scaling: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """按 (分数 + 偏置) 取 top-k；门控权重取原始分数（偏置只管"选谁"）。

    返回 idx (T, K) int64、weight (T, K) float32。
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
    """Switch / GShard 式负载均衡损失（写法同 DeepSeek-V3 式 17–20，作用在这一批 T 个 token 上）：

        f_i = N / (K·T) · #{t : i ∈ TopK_t}      （不可导，只是"权重"）
        P_i = 1/T · Σ_t s'_{i,t}，s' = 每行归一化后的分数（softmax 本来就归一化）
        L   = α · Σ_i f_i · P_i                   （完全均衡时 f_i = 1、P_i = 1/N，L = α）

    梯度只经过 P：哪个专家分到的 token 多（f_i 大），就更用力地压低它的路由概率。
    注意：HF 的 Mixtral / Qwen3-MoE 实现里 f 没有除以 K，数值是这里的 K 倍，系数不能直接照搬。
    """
    T, K = idx.shape
    probs = scores / scores.sum(dim=-1, keepdim=True)
    counts = torch.bincount(idx.reshape(-1), minlength=n_experts).to(probs.dtype)
    f = counts * n_experts / (K * T)
    P = probs.mean(dim=0)
    return coef * (f * P).sum()


# ---------------------------------------------------------------------------
# MoE 前馈层
# ---------------------------------------------------------------------------


class MoEFFN(nn.Module):
    """可直接替换 `zero.model.SwiGLU` 的 MoE 前馈层：forward(x: (..., dim)) -> (..., dim)。

    前向之后可以读：
    - `last_aux_loss`：本次前向的辅助损失（训练且 aux_loss_coef > 0 时是带梯度的张量，否则 0）；
    - `last_load`：本次前向每个专家分到的 token 数 (E,)（丢弃前）；`last_dropped`：被容量丢掉的分配数；
    - `load_accum`：自上次 `update_bias()` 以来累计的负载（训练时累加）。
    """

    def __init__(self, cfg: MoEConfig) -> None:
        super().__init__()
        self.cfg = cfg
        E, d, h = cfg.n_experts, cfg.dim, cfg.expert_dim
        self.router = nn.Linear(d, E, bias=False)
        # 按专家堆叠的 SwiGLU 权重：x (n, d) @ w_gate[e] (d, h) → (n, h)
        self.w_gate = nn.Parameter(torch.empty(E, d, h))
        self.w_up = nn.Parameter(torch.empty(E, d, h))
        self.w_down = nn.Parameter(torch.empty(E, h, d))
        self.shared = SwiGLU(d, cfg.shared_expert_dim) if cfg.shared_expert_dim else None
        # 无辅助损失均衡的偏置：不是参数（不吃梯度），但要进 state_dict（续训要接着用）
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
        """x: (T, d) → (scores (T,E), idx (T,K), weight (T,K))。"""
        c = self.cfg
        scores = router_scores(self.router(x), c.score_func)
        # 偏置总是参与选择（不更新时它保持为 0）：DeepSeek-V3 最后 500B token 把 γ 设成 0，
        # 但已经学到的偏置照样使用
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

        # ---- 负载统计与损失 ----
        counts = torch.bincount(idx.reshape(-1), minlength=c.n_experts)
        self.last_load = counts.detach().float()
        if self.training:
            self.load_accum += self.last_load
        if self.training and c.aux_loss_coef > 0:
            self.last_aux_loss = aux_balance_loss(scores, idx, c.n_experts, c.aux_loss_coef)
        else:
            self.last_aux_loss = torch.zeros((), device=x.device)

        # ---- 分发：把 T·K 个 (token, 专家) 分配按专家排序，每个专家拿到连续的一段 ----
        flat_e = idx.reshape(-1)
        flat_t = torch.arange(T, device=x.device).repeat_interleave(c.top_k)
        flat_w = weight.reshape(-1)
        order = torch.argsort(flat_e, stable=True)  # 稳定排序：同一专家内保持 token 原顺序
        flat_e, flat_t, flat_w = flat_e[order], flat_t[order], flat_w[order]
        cap = self.capacity(T)
        if cap is not None:
            # 每个分配在自己专家里的名次（0 起）；名次 ≥ 容量的被丢弃
            starts = torch.cumsum(counts, 0) - counts
            rank = torch.arange(flat_e.numel(), device=x.device) - starts[flat_e]
            keep = rank < cap
            self.last_dropped = int((~keep).sum())
            flat_e, flat_t, flat_w = flat_e[keep], flat_t[keep], flat_w[keep]
            counts = torch.bincount(flat_e, minlength=c.n_experts)
        else:
            self.last_dropped = 0

        # ---- 专家计算 + 合并（combine）----
        # CPU 版：逐专家循环。GPU 上应换成 grouped GEMM（尚未在 GPU 上验证）。
        out = torch.zeros_like(x)
        xs = x[flat_t]
        for e, seg in enumerate(torch.split(torch.arange(flat_t.numel(), device=x.device),
                                            counts.tolist())):
            if seg.numel() == 0:
                continue
            h = xs[seg]
            y = (F.silu(h @ self.w_gate[e]) * (h @ self.w_up[e])) @ self.w_down[e]
            out.index_add_(0, flat_t[seg], y * flat_w[seg, None].to(y.dtype))
        if self.shared is not None:
            out = out + self.shared(x)
        return out.reshape(shape)

    @torch.no_grad()
    def update_bias(self) -> None:
        """无辅助损失均衡（DeepSeek-V3 2.1.2 节）：每个训练步结束后调用一次。

        b_i ← b_i + γ · sign(平均负载 − 负载_i)：过载（负载 > 平均）的专家偏置减 γ，欠载的加 γ。
        负载取自上次调用以来累计的 `load_accum`（梯度累积时覆盖整个全局 batch）。
        """
        g = self.cfg.bias_update_speed
        if g <= 0:
            self.load_accum.zero_()
            return
        load = self.load_accum.clone()
        if dist.is_available() and dist.is_initialized():  # 尚未在多卡上验证
            dist.all_reduce(load)
        self.expert_bias += g * torch.sign(load.mean() - load)
        self.load_accum.zero_()


# ---------------------------------------------------------------------------
# 装进 Transformer
# ---------------------------------------------------------------------------


class MoETransformer(Transformer):
    """主线 `Transformer` 的 MoE 版本：前 `first_dense` 层保留稠密 SwiGLU，其余层的 FFN 换成 MoEFFN
    （DeepSeek-V3 的 first_k_dense_replace = 3、Kimi K2 = 1、GLM-4.5 = 3）。

    与主线相比多两件事：`loss()` 自动加上各层的辅助损失；每个优化器 step 之后调用 `after_step()`
    更新无辅助损失的偏置。（主线 `zero.train.trainer.Trainer` 尚未接入 `after_step`，第二步如需用它训练
    MoE 再接；本章小实验用自己的训练循环。）
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
        """最近一次前向每层每个专家分到的 token 占比，形状 (MoE 层数, E)，每行和为 1。"""
        loads = torch.stack([m.last_load for m in self.moe_layers()])
        return loads / loads.sum(dim=-1, keepdim=True).clamp_min(1)

    def param_counts(self) -> dict[str, int]:
        """total：全部参数（共享 embedding 只算一次）；active：每个 token 实际参与计算的参数
        （未被选中的专家不算；embedding 查表算在内，与多数模型卡口径一致）。"""
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
