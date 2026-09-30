"""第 18 章 · 极简代码 2：RLHF 的目标、KL 缰绳，以及 PPO 怎么解它

RLHF 的目标（对一个提示词）：
    max_π  E_{y~π}[ r(y) ] − β · KL(π || π_ref)
r 是奖励模型打的分，π_ref 是 SFT 模型。这个目标有闭式最优解：
    π*(y) = π_ref(y) · exp(r(y)/β) / Z
为了看清楚，把"回答"缩成 8 个候选（一个多臂老虎机）：每个候选有真实质量 q 和长度 ℓ，
奖励模型用 01 学到的系数打分，它会给长回答加分——包括一个 900 token 的"注水"回答。

① 扫 β：看代理奖励（奖励模型的分）、真实质量、KL 怎么变——这就是 reward hacking 与 KL 缰绳。
② 用 PPO（采样 + 裁剪比例 + 基线）从 π_ref 出发去解同一个目标，看它是否走到闭式解。

运行：uv run python chapters/18-preference-alignment/code/02_rlhf_kl.py
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

torch.set_num_threads(1)  # 构建机多任务共享 CPU（本机可删）

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("bt01", HERE / "01_bradley_terry.py")
bt01 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bt01)

# 8 个候选回答：(名字, 真实质量 q, 长度 ℓ/百 token, SFT 模型给它的 logit)
CANDIDATES = [
    ("简洁正确", 1.4, 1.2, 1.0),
    ("详细正确", 1.2, 2.5, 0.8),
    ("还行", 0.6, 1.0, 1.5),
    ("一般", 0.2, 1.5, 1.5),
    ("跑题", -0.8, 1.0, 0.5),
    ("错误", -1.5, 0.8, 0.5),
    ("啰嗦", 0.0, 4.0, 0.0),
    ("注水 900 字", -0.5, 9.0, -2.5),
]
NAMES = [c[0] for c in CANDIDATES]
Q = torch.tensor([c[1] for c in CANDIDATES])
FEATS = torch.tensor([[c[1], c[2]] for c in CANDIDATES])
LOGP_REF = torch.log_softmax(torch.tensor([c[3] for c in CANDIDATES]), 0)


def proxy_reward() -> torch.Tensor:
    """用 01 训练出来的奖励模型给 8 个候选打分（它学到了'长 = 好'）。"""
    rm, _ = bt01.train_rm()
    with torch.no_grad():
        return rm(FEATS)


def optimal_policy(r: torch.Tensor, beta: float) -> torch.Tensor:
    """闭式最优解：log π* = log π_ref + r/β − log Z。"""
    return torch.log_softmax(LOGP_REF + r / beta, 0)


def kl(logp: torch.Tensor, logq: torch.Tensor) -> float:
    return float((logp.exp() * (logp - logq)).sum())


def objective(logp: torch.Tensor, r: torch.Tensor, beta: float) -> float:
    return float((logp.exp() * r).sum()) - beta * kl(logp, LOGP_REF)


def ppo(r: torch.Tensor, beta: float, iters: int = 400, batch: int = 64, epochs: int = 4,
        clip: float = 0.2, lr: float = 0.05, seed: int = 0,
        trace_at: tuple[int, ...] | None = None) -> tuple[torch.Tensor, list[tuple]]:
    """InstructGPT 式的 PPO（缩到老虎机上）：
    - 每轮用当前策略采样 batch 个回答，奖励 = r(y) − β·(log π_old(y) − log π_ref(y))（逐样本的 KL 惩罚）；
    - 优势 A = 奖励 − V，V 是一个标量"价值"基线（这里只有一个提示词，V 就是一个数），用 MSE 学；
    - 同一批样本更新 epochs 次，比例 ρ = π_θ/π_old 被裁剪到 [1−ε, 1+ε]。"""
    g = torch.Generator().manual_seed(seed)
    theta = LOGP_REF.clone().requires_grad_(True)  # 策略的 logits，从 SFT 出发
    v = torch.zeros(1, requires_grad=True)          # 价值基线
    opt = torch.optim.Adam([theta, v], lr=lr)
    trace = []
    trace_at = trace_at or (0, 25, 100, iters)
    for it in range(iters + 1):
        with torch.no_grad():
            logp_old = torch.log_softmax(theta, 0)
            if it in trace_at:
                trace.append((it, objective(logp_old, r, beta), kl(logp_old, LOGP_REF)))
            y = torch.multinomial(logp_old.exp(), batch, replacement=True, generator=g)
            reward = r[y] - beta * (logp_old[y] - LOGP_REF[y])
        for _ in range(epochs):
            logp = torch.log_softmax(theta, 0)
            adv = (reward - v).detach()
            ratio = torch.exp(logp[y] - logp_old[y])
            surr = torch.minimum(ratio * adv, ratio.clamp(1 - clip, 1 + clip) * adv)
            loss = -surr.mean() + 0.5 * ((reward - v) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    return torch.log_softmax(theta.detach(), 0), trace


def main() -> None:
    r = proxy_reward()
    print("8 个候选回答（奖励模型 = 01 学到的 r = w_q·q + w_len·ℓ）：")
    print(f"   {'回答':<10} {'真实质量 q':>9} {'长度ℓ':>6} {'奖励模型分':>9} {'π_ref':>7}")
    for i, n in enumerate(NAMES):
        print(f"   {n:<10} {Q[i]:>9.1f} {FEATS[i, 1]:>6.1f} {r[i]:>9.2f} {LOGP_REF[i].exp():>7.3f}")
    print("   奖励模型最爱：", NAMES[int(r.argmax())], "；真实质量最高：", NAMES[int(Q.argmax())])

    print("\n① 闭式最优解 π* ∝ π_ref·exp(r/β)，扫 β：")
    print(f"   {'β':>6} | {'E[奖励模型分]':>11} | {'E[真实质量]':>10} | {'KL(π||π_ref)':>12} | 概率最大的回答")
    for beta in (100.0, 2.0, 1.0, 0.5, 0.25, 0.1, 0.03):
        lp = optimal_policy(r, beta)
        p = lp.exp()
        top = int(p.argmax())
        print(f"   {beta:>6} | {float((p * r).sum()):>11.3f} | {float((p * Q).sum()):>10.3f} | "
              f"{kl(lp, LOGP_REF):>12.3f} | {NAMES[top]}（{p[top]:.2f}）")
    print("   β 大：几乎不动（还是 SFT）；β 适中：真实质量最高；β 太小：全押'注水'——代理分最高，真实质量反而下降。")

    beta = 0.5
    lp_star = optimal_policy(r, beta)
    lp_ppo, trace = ppo(r, beta)
    print(f"\n② PPO 从 π_ref 出发解同一个目标（β = {beta}，每轮 64 个样本，4 个 epoch，ε = 0.2）：")
    print(f"   {'轮':>4} | {'目标 E[r]−β·KL':>13} | {'KL(π||π_ref)':>12}")
    for it, obj, k in trace:
        print(f"   {it:>4} | {obj:>13.4f} | {k:>12.4f}")
    print(f"   闭式最优解的目标值 = {objective(lp_star, r, beta):.4f}，KL = {kl(lp_star, LOGP_REF):.4f}")
    print(f"   PPO 结果与闭式解的距离 KL(π_PPO || π*) = {kl(lp_ppo, lp_star):.5f}")
    print("   概率对比（π_ref → π_PPO / π*）：")
    for i, n in enumerate(NAMES):
        print(f"     {n:<10} {LOGP_REF[i].exp():.3f} → {lp_ppo[i].exp():.3f} / {lp_star[i].exp():.3f}")
    print("\n小结：PPO 要靠采样、价值基线、裁剪一步步摸到 π*；而 π* 其实有闭式解——这就是 DPO 的出发点（03）。")


if __name__ == "__main__":
    main()
