"""第 25 章 · 极简代码 3：采样时的推测解码——拒绝采样为什么一点都不改变分布

目标模型给出分布 p，草稿模型给出分布 q，草稿按 q 抽出一个 token x：
  - 以概率 min(1, p(x)/q(x)) 接受 x；
  - 被拒绝时，从"残差分布" p'(x) = max(0, p(x) − q(x)) / Σ max(0, p − q) 重新抽一个。
结论：最后得到的 token 恰好服从 p。一次接受的概率是 α = Σ_x min(p(x), q(x)) = 1 − TV(p, q)。

本脚本做三件事：
  1. 小词表玩具：一步拒绝采样抽 20 万次，和 p 比（TV 距离 + 卡方检验）；对照"直接用草稿的样本"；
  2. 完整算法（k 个草稿 + 奖励 token）在一对马尔可夫链"语言模型"上生成长度 3 的序列 6 万次，
     和目标模型的精确联合分布比（4³ = 64 个格子的卡方检验）；
  3. 真实小模型（第 10 章目标 + 1 层草稿），温度 1：实测接受率 vs 公式 Σ min(p, q)，以及速度。

运行：uv run python chapters/25-mtp-speculative-decoding/code/03_speculative_sampling.py
"""

from __future__ import annotations

import importlib.util
import statistics
import sys
import time
from pathlib import Path

import torch

torch.set_num_threads(1)
HERE = Path(__file__).resolve().parent


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ── 核心：一次"接受或拒绝" ────────────────────────────────────────────────────
def accept_or_resample(p: torch.Tensor, q: torch.Tensor, x: int, g: torch.Generator):
    """p, q: (V,) 概率；x ~ q。返回 (是否接受, 最终 token)。"""
    if torch.rand((), generator=g) < torch.clamp(p[x] / q[x], max=1.0):  # 以 min(1, p/q) 接受
        return True, x
    residual = torch.clamp(p - q, min=0)  # 残差分布 max(0, p − q)，再归一化
    return False, int(torch.multinomial(residual / residual.sum(), 1, generator=g))


def speculative_step(p_rows, q_rows, drafts, g):
    """一轮验证。p_rows: (k+1, V) 目标在每个位置的分布；q_rows: (k, V) 草稿的分布；drafts: k 个草稿。
    返回 (接受个数 m, 本轮新增的 token 列表，长度 m+1)。"""
    k = len(drafts)
    for i in range(k):
        ok, tok = accept_or_resample(p_rows[i], q_rows[i], drafts[i], g)
        if not ok:
            return i, drafts[:i] + [tok]  # 第 i 个被拒：前 i 个 + 残差分布里重抽的 1 个
    bonus = int(torch.multinomial(p_rows[k], 1, generator=g))  # 全部接受：白送 1 个
    return k, drafts + [bonus]


# ── 统计工具（不依赖 scipy）────────────────────────────────────────────────
def tv(a: torch.Tensor, b: torch.Tensor) -> float:
    return 0.5 * float((a - b).abs().sum())


def chi_square(counts: torch.Tensor, probs: torch.Tensor) -> tuple[float, int, float]:
    """皮尔逊卡方检验：返回 (统计量, 自由度, p 值)。期望次数太小的格子合并，避免近似失效。"""
    n = counts.sum()
    exp = probs * n
    big = exp >= 5
    obs_b, exp_b = counts[big].double(), exp[big].double()
    if (~big).any():  # 小格子合并成一个
        obs_b = torch.cat([obs_b, counts[~big].sum().double().view(1)])
        exp_b = torch.cat([exp_b, exp[~big].sum().double().view(1)])
    stat = float(((obs_b - exp_b) ** 2 / exp_b).sum())
    df = len(obs_b) - 1
    pval = float(torch.special.gammaincc(torch.tensor(df / 2.0), torch.tensor(stat / 2.0)))
    return stat, df, pval


def toy_single_step(V: int = 6, N: int = 200_000):
    torch.manual_seed(4)  # p、q 由全局随机数生成，固定下来（种子 4：残差落在 3 个 token 上，好画）
    g = torch.Generator().manual_seed(0)
    p = torch.distributions.Dirichlet(torch.ones(V)).sample()
    q = torch.distributions.Dirichlet(torch.ones(V)).sample()
    xs = torch.multinomial(q, N, replacement=True, generator=g).tolist()
    out, acc = torch.zeros(V), 0
    for x in xs:
        ok, tok = accept_or_resample(p, q, x, g)
        out[tok] += 1
        acc += ok
    naive = torch.bincount(torch.tensor(xs), minlength=V).float()
    return dict(
        p=p,
        q=q,
        spec=out / N,
        naive=naive / N,
        accept=acc / N,
        alpha=float(torch.minimum(p, q).sum()),
        chi=chi_square(out, p),
        chi_naive=chi_square(naive, p),
        residual=torch.clamp(p - q, min=0),
    )


def toy_markov(V: int = 4, L: int = 3, k: int = 2, N: int = 60_000):
    """目标 P、草稿 Q 都是 V×V 的转移矩阵（"下一个 token 只看上一个 token"的语言模型）。"""
    torch.manual_seed(1)
    P = torch.distributions.Dirichlet(torch.ones(V) * 0.5).sample((V,))
    Q = torch.distributions.Dirichlet(torch.ones(V) * 0.5).sample((V,))
    g = torch.Generator().manual_seed(2)
    counts = torch.zeros(V**L)
    rounds = 0
    for _ in range(N):
        seq = [0]  # 起始 token 固定为 0
        while len(seq) - 1 < L:
            drafts, q_rows = [], []
            for _ in range(k):  # 草稿自回归地抽 k 个
                q_rows.append(Q[(drafts or seq)[-1]])
                drafts.append(int(torch.multinomial(q_rows[-1], 1, generator=g)))
            ctx = seq + drafts  # 目标"一次前向"：每个位置的分布
            p_rows = torch.stack([P[ctx[len(seq) - 1 + i]] for i in range(k + 1)])
            _, new = speculative_step(p_rows, torch.stack(q_rows), drafts, g)
            seq += new
            rounds += 1
        idx = 0
        for t in seq[1 : L + 1]:  # 只看前 L 个生成的 token
            idx = idx * V + t
        counts[idx] += 1
    # 目标模型的精确联合分布
    exact = torch.ones(1)
    prev = torch.zeros(1, dtype=torch.long)
    for _ in range(L):
        exact = (exact[:, None] * P[prev]).reshape(-1)
        prev = torch.arange(V).repeat(len(prev))
    return dict(
        tv=tv(counts / N, exact),
        chi=chi_square(counts, exact),
        cells=V**L,
        tokens_per_round=N * L / rounds,
    )


# ── 真实小模型上的采样推测解码 ───────────────────────────────────────────────
m1 = _load("ch25_models", "01_models_and_cost.py")
KVCache, truncate = m1.KVCache, m1.truncate


@torch.no_grad()
def sample_generate(model, prompt, n_new, temperature, g):
    cache = KVCache(model.c.n_layers)
    logits = model(torch.tensor([prompt]), cache)[0, -1]
    out = []
    for _ in range(n_new):
        nxt = int(torch.multinomial(torch.softmax(logits / temperature, -1), 1, generator=g))
        out.append(nxt)
        logits = model(torch.tensor([[nxt]]), cache)[0, -1]
    return out


@torch.no_grad()
def speculative_sample(target, draft, prompt, n_new, k, temperature, g):
    seq = list(prompt)
    tc, dc = KVCache(target.c.n_layers), KVCache(draft.c.n_layers)
    st = dict(rounds=0, accepted=0, examined=0, alpha_sum=0.0)
    while len(seq) - len(prompt) < n_new:
        logits = draft(torch.tensor([seq[len(dc) :]]), dc)[0, -1]
        drafts, q_rows = [], []
        for i in range(k):
            q_rows.append(torch.softmax(logits / temperature, -1))
            drafts.append(int(torch.multinomial(q_rows[-1], 1, generator=g)))
            if i < k - 1:
                logits = draft(torch.tensor([[drafts[-1]]]), dc)[0, -1]
        p_logits = target(torch.tensor([seq[len(tc) :] + drafts]), tc)[0, -(k + 1) :]
        p_rows = torch.softmax(p_logits / temperature, -1)
        m, new = speculative_step(p_rows, torch.stack(q_rows), drafts, g)
        # 理论接受率 Σ min(p, q)：在每个被比较过的位置上累加
        for i in range(min(m + 1, k)):
            st["alpha_sum"] += float(torch.minimum(p_rows[i], q_rows[i]).sum())
        seq += new
        st["rounds"] += 1
        st["accepted"] += m
        st["examined"] += min(m + 1, k)
        truncate(tc, len(seq) - 1)
        truncate(dc, min(len(dc), len(seq) - 1))
    return seq[len(prompt) :][:n_new], st


if __name__ == "__main__":
    r = toy_single_step()
    fmt = lambda t: "[" + ", ".join(f"{v:.3f}" for v in t.tolist()) + "]"  # noqa: E731
    print("── 1. 一步拒绝采样（词表 6，抽 200,000 次）──")
    print(f"目标 p        = {fmt(r['p'])}")
    print(f"草稿 q        = {fmt(r['q'])}")
    print(f"残差 max(0,p−q) = {fmt(r['residual'])}（归一化前）")
    print(
        f"推测采样的结果 = {fmt(r['spec'])}  TV(结果, p) = {tv(r['spec'], r['p']):.4f}  "
        f"卡方 {r['chi'][0]:.1f}（自由度 {r['chi'][1]}），p 值 {r['chi'][2]:.2f}"
    )
    print(
        f"直接用草稿样本 = {fmt(r['naive'])}  TV(结果, p) = {tv(r['naive'], r['p']):.4f}  "
        f"卡方 {r['chi_naive'][0]:.0f}，p 值 {r['chi_naive'][2]:.1e}"
    )
    print(
        f"实测接受率 {r['accept']:.4f}，公式 Σmin(p,q) = {r['alpha']:.4f}，"
        f"1 − TV(p,q) = {1 - tv(r['p'], r['q']):.4f}"
    )

    print(
        "\n── 2. 完整算法（k=2 个草稿 + 奖励 token），马尔可夫链玩具，生成长度 3 的序列 60,000 次 ──"
    )
    mk = toy_markov()
    print(
        f"与目标模型精确联合分布（{mk['cells']} 个格子）：TV = {mk['tv']:.4f}，"
        f"卡方 {mk['chi'][0]:.1f}（自由度 {mk['chi'][1]}），p 值 {mk['chi'][2]:.2f}；"
        f"平均每轮产出 {mk['tokens_per_round']:.2f} 个 token"
    )

    print("\n── 3. 真实小模型，温度 1.0（4 段提示词 × 200 个字符）──")
    target, draft = m1.load_target(), m1.load_draft()
    m2 = _load("ch25_greedy", "02_greedy_speculative.py")
    P, N, T = m2.prompts(), 200, 1.0
    t_base = []
    t_spec, sts = {}, {}
    ks = (1, 3, 5)
    for rep in range(3):
        t0 = time.process_time()
        for i, p in enumerate(P):
            sample_generate(target, p, N, T, torch.Generator().manual_seed(100 * rep + i))
        t_base.append(time.process_time() - t0)
        for k in ks:
            agg = dict(rounds=0, accepted=0, examined=0, alpha_sum=0.0)
            t0 = time.process_time()
            for i, p in enumerate(P):
                _, s = speculative_sample(
                    target, draft, p, N, k, T, torch.Generator().manual_seed(100 * rep + i)
                )
                for key in agg:
                    agg[key] += s[key]
            t_spec.setdefault(k, []).append(time.process_time() - t0)
            if rep == 0:
                sts[k] = agg
    tb = statistics.median(t_base)
    print(f"普通采样 CPU 时间 {tb:.2f} s（中位数）")
    print(f"{'k':>2} {'实测接受率':>9} {'公式 Σmin(p,q) 均值':>17} {'每轮产出':>8} {'加速比':>7}")
    for k in ks:
        s = sts[k]
        print(
            f"{k:2d} {s['accepted'] / s['examined']:10.3f} {s['alpha_sum'] / s['examined']:18.3f} "
            f"{(s['accepted'] + s['rounds']) / s['rounds']:9.2f} "
            f"{tb / statistics.median(t_spec[k]):6.2f}×"
        )
