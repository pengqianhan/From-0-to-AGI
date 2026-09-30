"""第 24 章 · 极简代码 2：从零写一个 MoE 层（路由 → top-k → 专家 → 加权合并）+ 两种负载均衡

MoE 层 = 一个路由器 + N 个专家（每个专家就是第 9 章的 SwiGLU）+ 可选的共享专家：

    s   = sigmoid(x · W_r)  或  softmax(x · W_r)          # 打分：每个专家一个分数
    选  = TopK(s + b)                                     # 选 K 个专家；偏置 b 只影响"选谁"
    g   = s[选] / Σ s[选]                                 # 门控权重：被选中专家的分数，归一化
    y   = Σ_共享 FFN(x) + Σ_{i∈选} g_i · FFN_i(x)          # 只算被选中的专家

负载均衡：
  - 辅助损失（Switch / GShard）：L_aux = α · Σ_i f_i · P_i，f_i = N/(K·T) × 分到 i 的 token 数，
    P_i = 路由概率的平均；完全均衡时 L_aux = α；
  - 无辅助损失（DeepSeek-V3）：每步之后 b_i ← b_i + γ · sign(平均负载 − 负载_i)。

本文件的 __main__ 做四个小检查：参数账、与逐 token 朴素实现对拍、辅助损失手算、偏置把偏斜的路由拉平。
第 3 个脚本 03_train_compare.py 用这里的 MoE 训练小语言模型。
运行：uv run python chapters/24-mixture-of-experts/code/02_moe_layer.py
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(1)


class Expert(nn.Module):
    """一个专家 = 一个 SwiGLU（第 9 章）：W_down(SiLU(W_gate x) ⊙ W_up x)。"""

    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.w_gate = nn.Linear(dim, hidden, bias=False)
        self.w_up = nn.Linear(dim, hidden, bias=False)
        self.w_down = nn.Linear(hidden, dim, bias=False)

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class MoE(nn.Module):
    def __init__(self, dim: int, n_experts: int, top_k: int, hidden: int, n_shared: int = 0,
                 score: str = "sigmoid", balance: str = "none", aux_coef: float = 0.01,
                 bias_speed: float = 0.01) -> None:
        super().__init__()
        self.E, self.K, self.score = n_experts, top_k, score
        self.balance, self.aux_coef, self.bias_speed = balance, aux_coef, bias_speed
        self.router = nn.Linear(dim, n_experts, bias=False)
        self.experts = nn.ModuleList(Expert(dim, hidden) for _ in range(n_experts))
        self.shared = Expert(dim, hidden * n_shared) if n_shared else None
        self.register_buffer("bias", torch.zeros(n_experts))  # 无辅助损失均衡用，不吃梯度
        self.load = torch.zeros(n_experts)  # 最近一次前向每个专家分到的 token 数
        self.aux = torch.zeros(())

    def forward(self, x):
        shape = x.shape
        x = x.reshape(-1, shape[-1])                                 # (T, d)
        logits = self.router(x)                                      # (T, E)
        s = logits.sigmoid() if self.score == "sigmoid" else logits.softmax(-1)
        idx = (s + self.bias).topk(self.K, dim=-1).indices           # (T, K) 选专家：偏置只管选
        g = s.gather(-1, idx)                                        # (T, K) 门控：原始分数
        g = g / g.sum(-1, keepdim=True)                              # 归一化到和为 1

        out = self.shared(x) if self.shared is not None else torch.zeros_like(x)
        for e, expert in enumerate(self.experts):
            tok, slot = (idx == e).nonzero(as_tuple=True)            # 哪些 token 选了专家 e
            if tok.numel():
                out.index_add_(0, tok, g[tok, slot, None] * expert(x[tok]))   # 加权合并

        self.load = torch.bincount(idx.flatten(), minlength=self.E).float()
        self.aux = torch.zeros(())
        if self.training and self.balance == "aux":
            p = s / s.sum(-1, keepdim=True)                          # 每行归一化的路由概率
            f = self.load * self.E / (self.K * x.shape[0])           # f_i（均衡时 = 1）
            self.aux = self.aux_coef * (f * p.mean(0)).sum()         # L_aux = α · Σ f_i · P_i
        return out.reshape(shape)

    @torch.no_grad()
    def update_bias(self, load: torch.Tensor | None = None) -> None:
        """无辅助损失均衡：过载的专家 b 减 γ，欠载的加 γ（每个训练步之后调一次）。"""
        load = self.load if load is None else load
        self.bias += self.bias_speed * torch.sign(load.mean() - load)


def naive_moe(m: MoE, x: torch.Tensor) -> torch.Tensor:
    """逐个 token 算，用来对拍上面"按专家分组"的写法。"""
    out = []
    for t in x.reshape(-1, x.shape[-1]):
        logit = m.router(t)
        s = logit.sigmoid() if m.score == "sigmoid" else logit.softmax(-1)
        top = torch.argsort(s + m.bias, descending=True)[: m.K]
        g = s[top] / s[top].sum()
        y = m.shared(t) if m.shared is not None else torch.zeros_like(t)
        out.append(y + sum(gi * m.experts[int(e)](t) for gi, e in zip(g, top)))
    return torch.stack(out).reshape(x.shape)


def n_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def dropped(load: torch.Tensor, capacity_factor: float, k: int) -> float:
    """容量因子：每个专家最多处理 ⌈cf · T·K / E⌉ 个分配，多出来的被丢弃。返回被丢弃的比例。"""
    total = float(load.sum())
    cap = math.ceil(capacity_factor * total / len(load))
    return float((load - cap).clamp_min(0).sum()) / total


def skewed_router_demo(steps=(0, 10, 30, 100, 300), seed: int = 0):
    """固定一个偏向专家 0 的路由器，不训练任何权重，只做偏置更新，看负载怎么变。"""
    torch.manual_seed(seed)
    m = MoE(dim=16, n_experts=8, top_k=2, hidden=8, balance="free", bias_speed=0.01)
    with torch.no_grad():
        m.router.weight.normal_(std=0.3)
        m.router.weight[0] += 0.4
        m.router.weight[1] += 0.2
    x = torch.randn(1024, 16).abs()  # 正的输入：专家 0、1 的分数系统性偏高
    rows = []
    with torch.no_grad():
        for step in range(max(steps) + 1):
            m(x)
            if step in steps:
                frac = m.load / m.load.sum()
                rows.append(dict(step=step, frac=frac.tolist(),
                                 max_over_mean=float(m.load.max() / m.load.mean()),
                                 dropped=dropped(m.load, 1.25, m.K),
                                 bias=m.bias.tolist()))
            m.update_bias()
    return rows


if __name__ == "__main__":
    d = 128
    dense = Expert(d, 384)
    moe = MoE(d, n_experts=8, top_k=2, hidden=192)
    fg = MoE(d, n_experts=16, top_k=3, hidden=96, n_shared=1)
    print("1) 参数账（d = 128；active = 每个 token 实际用到的）")
    for name, m, active in [
        ("稠密 SwiGLU，宽 384", dense, n_params(dense)),
        ("MoE 8 专家 × 宽 192，选 2", moe, 2 * n_params(moe.experts[0]) + n_params(moe.router)),
        ("细粒度 16 × 宽 96，选 3 + 1 共享", fg,
         3 * n_params(fg.experts[0]) + n_params(fg.shared) + n_params(fg.router)),
    ]:
        print(f"  {name:28} 总参数 {n_params(m):8,d}   激活参数 {active:7,d}")

    print("\n2) 按专家分组的写法 vs 逐 token 朴素写法")
    torch.manual_seed(0)
    for score in ("softmax", "sigmoid"):
        m = MoE(16, n_experts=6, top_k=2, hidden=8, n_shared=1, score=score).double()
        with torch.no_grad():
            m.bias.copy_(torch.linspace(-0.1, 0.1, 6).double())
        x = torch.randn(3, 5, 16, dtype=torch.float64)
        diff = (m(x) - naive_moe(m, x)).abs().max().item()
        print(f"  {score:7} 最大差异 {diff:.1e}")

    print("\n3) 辅助损失手算：4 个 token、2 个专家、top-1")
    m = MoE(2, n_experts=2, top_k=1, hidden=1, score="softmax", balance="aux", aux_coef=0.01)
    with torch.no_grad():  # 让 softmax 概率恰好是 [0.9,0.1] [0.8,0.2] [0.7,0.3] [0.4,0.6]
        m.router.weight.copy_(torch.eye(2))
    p = torch.tensor([[0.9, 0.1], [0.8, 0.2], [0.7, 0.3], [0.4, 0.6]])
    m.train()(p.log())
    print(f"  负载 {m.load.tolist()}，f = [1.5, 0.5]，P = [0.7, 0.3] → "
          f"L = 0.01 × (1.5×0.7 + 0.5×0.3) = 0.012；代码算出 {m.aux.item():.4f}")

    print("\n4) 无辅助损失均衡：路由器固定偏向专家 0、1，只更新偏置（γ = 0.01），8 专家选 2")
    for r in skewed_router_demo():
        frac = " ".join(f"{v:.2f}" for v in r["frac"])
        print(f"  更新 {r['step']:3d} 次  负载占比 [{frac}]  最大/平均 {r['max_over_mean']:.2f}"
              f"  容量因子 1.25 时丢弃 {r['dropped']:.1%}")
