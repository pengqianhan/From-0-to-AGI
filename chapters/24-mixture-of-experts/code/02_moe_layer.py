"""Chapter 24 · Minimal code 2: write an MoE layer from zero (router → top-k → experts → weighted
combination) + two load-balancing methods.

MoE layer = one router + N experts (each expert is the SwiGLU of Chapter 9) + optional shared experts:

    s   = sigmoid(x · W_r)  or  softmax(x · W_r)          # score: one score for each expert
    sel = TopK(s + b)                                     # select K experts; the bias b changes only the selection
    g   = s[sel] / Σ s[sel]                               # gate weights: scores of the selected experts, normalized
    y   = Σ_shared FFN(x) + Σ_{i∈sel} g_i · FFN_i(x)       # calculate only the selected experts

Load balancing:
  - Auxiliary loss (Switch / GShard): L_aux = α · Σ_i f_i · P_i, where
    f_i = N/(K·T) × (number of tokens sent to i), and P_i = the mean routing probability.
    With a perfect balance, L_aux = α.
  - Auxiliary-loss-free (DeepSeek-V3): after each step, b_i ← b_i + γ · sign(mean load − load_i).

The __main__ block does four small checks: the parameter ledger, a parity check against a naive
token-by-token version, the auxiliary loss by hand, and a bias that balances a skewed router.
The third script, 03_train_compare.py, trains small language models with the MoE from this file.
Run: uv run python chapters/24-mixture-of-experts/code/02_moe_layer.py
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(1)


class Expert(nn.Module):
    """One expert = one SwiGLU (Chapter 9): W_down(SiLU(W_gate x) ⊙ W_up x)."""

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
        self.register_buffer("bias", torch.zeros(n_experts))  # for aux-loss-free balancing; gets no gradient
        self.load = torch.zeros(n_experts)  # tokens that each expert got in the last forward pass
        self.aux = torch.zeros(())

    def forward(self, x):
        shape = x.shape
        x = x.reshape(-1, shape[-1])                                 # (T, d)
        logits = self.router(x)                                      # (T, E)
        s = logits.sigmoid() if self.score == "sigmoid" else logits.softmax(-1)
        idx = (s + self.bias).topk(self.K, dim=-1).indices           # (T, K) select experts: the bias only selects
        g = s.gather(-1, idx)                                        # (T, K) gates: the original scores
        g = g / g.sum(-1, keepdim=True)                              # normalize to a sum of 1

        out = self.shared(x) if self.shared is not None else torch.zeros_like(x)
        for e, expert in enumerate(self.experts):
            tok, slot = (idx == e).nonzero(as_tuple=True)            # the tokens that selected expert e
            if tok.numel():
                out.index_add_(0, tok, g[tok, slot, None] * expert(x[tok]))   # weighted combination

        self.load = torch.bincount(idx.flatten(), minlength=self.E).float()
        self.aux = torch.zeros(())
        if self.training and self.balance == "aux":
            p = s / s.sum(-1, keepdim=True)                          # routing probabilities, normalized per row
            f = self.load * self.E / (self.K * x.shape[0])           # f_i (= 1 when balanced)
            self.aux = self.aux_coef * (f * p.mean(0)).sum()         # L_aux = α · Σ f_i · P_i
        return out.reshape(shape)

    @torch.no_grad()
    def update_bias(self, load: torch.Tensor | None = None) -> None:
        """Aux-loss-free balancing: subtract γ from b of an overloaded expert, and add γ to b of an
        underloaded expert. Call it once after each training step."""
        load = self.load if load is None else load
        self.bias += self.bias_speed * torch.sign(load.mean() - load)


def naive_moe(m: MoE, x: torch.Tensor) -> torch.Tensor:
    """Calculate token by token. This is the parity check for the "grouped by expert" version above."""
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
    """Capacity factor: each expert processes at most ⌈cf · T·K / E⌉ assignments and drops the rest.
    Return the fraction that the experts drop."""
    total = float(load.sum())
    cap = math.ceil(capacity_factor * total / len(load))
    return float((load - cap).clamp_min(0).sum()) / total


def skewed_router_demo(steps=(0, 10, 30, 100, 300), seed: int = 0):
    """Fix a router that prefers expert 0. Train no weights, only update the bias, and record the load."""
    torch.manual_seed(seed)
    m = MoE(dim=16, n_experts=8, top_k=2, hidden=8, balance="free", bias_speed=0.01)
    with torch.no_grad():
        m.router.weight.normal_(std=0.3)
        m.router.weight[0] += 0.4
        m.router.weight[1] += 0.2
    x = torch.randn(1024, 16).abs()  # positive inputs: the scores of experts 0 and 1 are always higher
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
    print("1) Parameter ledger (d = 128; active = the parameters that each token uses)")
    for name, m, active in [
        ("dense SwiGLU, width 384", dense, n_params(dense)),
        ("MoE 8×192, top 2", moe, 2 * n_params(moe.experts[0]) + n_params(moe.router)),
        ("fine 16×96, top 3 + 1 shared", fg,
         3 * n_params(fg.experts[0]) + n_params(fg.shared) + n_params(fg.router)),
    ]:
        print(f"  {name:28} total {n_params(m):8,d}   active {active:7,d}")

    print("\n2) Grouped by expert vs naive token by token")
    torch.manual_seed(0)
    for score in ("softmax", "sigmoid"):
        m = MoE(16, n_experts=6, top_k=2, hidden=8, n_shared=1, score=score).double()
        with torch.no_grad():
            m.bias.copy_(torch.linspace(-0.1, 0.1, 6).double())
        x = torch.randn(3, 5, 16, dtype=torch.float64)
        diff = (m(x) - naive_moe(m, x)).abs().max().item()
        print(f"  {score:7} max difference {diff:.1e}")

    print("\n3) Auxiliary loss by hand: 4 tokens, 2 experts, top-1")
    m = MoE(2, n_experts=2, top_k=1, hidden=1, score="softmax", balance="aux", aux_coef=0.01)
    with torch.no_grad():  # make the softmax probabilities exactly [0.9,0.1] [0.8,0.2] [0.7,0.3] [0.4,0.6]
        m.router.weight.copy_(torch.eye(2))
    p = torch.tensor([[0.9, 0.1], [0.8, 0.2], [0.7, 0.3], [0.4, 0.6]])
    m.train()(p.log())
    print(f"  load {m.load.tolist()}, f = [1.5, 0.5], P = [0.7, 0.3] → "
          f"L = 0.01 × (1.5×0.7 + 0.5×0.3) = 0.012; the code gives {m.aux.item():.4f}")

    print("\n4) Aux-loss-free balancing: a fixed router that prefers experts 0 and 1; "
          "update only the bias (γ = 0.01); 8 experts, top 2")
    for r in skewed_router_demo():
        frac = " ".join(f"{v:.2f}" for v in r["frac"])
        print(f"  {r['step']:3d} updates  load share [{frac}]  max/mean {r['max_over_mean']:.2f}"
              f"  dropped at capacity factor 1.25 {r['dropped']:.1%}")
