"""Chapter 25 · Minimal code 3: speculative decoding with sampling. Why rejection sampling does
not change the distribution at all.

The target model gives a distribution p, and the draft model gives a distribution q.
The draft samples a token x from q:
  - Accept x with the probability min(1, p(x)/q(x)).
  - If x is rejected, sample a new token from the "residual distribution"
    p'(x) = max(0, p(x) − q(x)) / Σ max(0, p − q).
Result: the final token follows p exactly. The probability of one acceptance is
α = Σ_x min(p(x), q(x)) = 1 − TV(p, q).

This script does three things:
  1. A toy with a small vocabulary: sample 200,000 times with one step of rejection sampling and
     compare with p (TV distance + chi-square test). Reference: "use the draft samples directly".
  2. The full algorithm (k drafts + bonus token) on a pair of Markov-chain "language models":
     generate sequences of length 3, 60,000 times, and compare with the exact joint distribution of
     the target model (a chi-square test on 4³ = 64 cells).
  3. The real small models (the Chapter 10 target + the 1-layer draft), temperature 1:
     measured acceptance rate vs the formula Σ min(p, q), and the speed.

Run: uv run python chapters/25-mtp-speculative-decoding/code/03_speculative_sampling.py
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


# ── Core: one "accept or reject" ─────────────────────────────────────────────────
def accept_or_resample(p: torch.Tensor, q: torch.Tensor, x: int, g: torch.Generator):
    """p, q: (V,) probabilities; x ~ q. Returns (accepted or not, final token)."""
    if torch.rand((), generator=g) < torch.clamp(p[x] / q[x], max=1.0):  # Accept with min(1, p/q)
        return True, x
    residual = torch.clamp(p - q, min=0)  # Residual distribution max(0, p − q), then normalize
    return False, int(torch.multinomial(residual / residual.sum(), 1, generator=g))


def speculative_step(p_rows, q_rows, drafts, g):
    """One round of verification.

    p_rows: (k+1, V) distributions of the target at each position; q_rows: (k, V) distributions
    of the draft; drafts: k drafts.
    Returns (number accepted m, list of the new tokens of this round, length m+1)."""
    k = len(drafts)
    for i in range(k):
        ok, tok = accept_or_resample(p_rows[i], q_rows[i], drafts[i], g)
        if not ok:
            return i, drafts[:i] + [tok]  # Draft i is rejected: the first i drafts + 1 new sample from the residual
    bonus = int(torch.multinomial(p_rows[k], 1, generator=g))  # All accepted: 1 more token free
    return k, drafts + [bonus]


# ── Statistics tools (no scipy needed) ─────────────────────────────────────────
def tv(a: torch.Tensor, b: torch.Tensor) -> float:
    return 0.5 * float((a - b).abs().sum())


def chi_square(counts: torch.Tensor, probs: torch.Tensor) -> tuple[float, int, float]:
    """Pearson chi-square test: returns (statistic, degrees of freedom, p-value).

    Cells with a very small expected count are merged, because the approximation fails for them."""
    n = counts.sum()
    exp = probs * n
    big = exp >= 5
    obs_b, exp_b = counts[big].double(), exp[big].double()
    if (~big).any():  # Merge the small cells into one
        obs_b = torch.cat([obs_b, counts[~big].sum().double().view(1)])
        exp_b = torch.cat([exp_b, exp[~big].sum().double().view(1)])
    stat = float(((obs_b - exp_b) ** 2 / exp_b).sum())
    df = len(obs_b) - 1
    pval = float(torch.special.gammaincc(torch.tensor(df / 2.0), torch.tensor(stat / 2.0)))
    return stat, df, pval


def toy_single_step(V: int = 6, N: int = 200_000):
    torch.manual_seed(4)  # p and q come from the global RNG; fix it (seed 4: the residual is on 3 tokens, good for a plot)
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
    """Target P and draft Q are both V×V transition matrices.

    They are language models in which the next token depends only on the previous token."""
    torch.manual_seed(1)
    P = torch.distributions.Dirichlet(torch.ones(V) * 0.5).sample((V,))
    Q = torch.distributions.Dirichlet(torch.ones(V) * 0.5).sample((V,))
    g = torch.Generator().manual_seed(2)
    counts = torch.zeros(V**L)
    rounds = 0
    for _ in range(N):
        seq = [0]  # The start token is always 0
        while len(seq) - 1 < L:
            drafts, q_rows = [], []
            for _ in range(k):  # The draft samples k tokens autoregressively
                q_rows.append(Q[(drafts or seq)[-1]])
                drafts.append(int(torch.multinomial(q_rows[-1], 1, generator=g)))
            ctx = seq + drafts  # "One forward pass" of the target: the distribution at each position
            p_rows = torch.stack([P[ctx[len(seq) - 1 + i]] for i in range(k + 1)])
            _, new = speculative_step(p_rows, torch.stack(q_rows), drafts, g)
            seq += new
            rounds += 1
        idx = 0
        for t in seq[1 : L + 1]:  # Use only the first L generated tokens
            idx = idx * V + t
        counts[idx] += 1
    # The exact joint distribution of the target model
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


# ── Speculative decoding with sampling on the real small models ─────────────────
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
        # Theoretical acceptance rate Σ min(p, q): add it up at each position that was compared
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
    print("── 1. One step of rejection sampling (vocabulary 6, 200,000 samples) ──")
    print(f"target p        = {fmt(r['p'])}")
    print(f"draft q         = {fmt(r['q'])}")
    print(f"residual max(0,p−q) = {fmt(r['residual'])} (before normalization)")
    print(
        f"speculative     = {fmt(r['spec'])}  TV(result, p) = {tv(r['spec'], r['p']):.4f}  "
        f"chi-square {r['chi'][0]:.1f} (df {r['chi'][1]}), p-value {r['chi'][2]:.2f}"
    )
    print(
        f"draft samples   = {fmt(r['naive'])}  TV(result, p) = {tv(r['naive'], r['p']):.4f}  "
        f"chi-square {r['chi_naive'][0]:.0f}, p-value {r['chi_naive'][2]:.1e}"
    )
    print(
        f"measured acceptance rate {r['accept']:.4f}, formula Σmin(p,q) = {r['alpha']:.4f}, "
        f"1 − TV(p,q) = {1 - tv(r['p'], r['q']):.4f}"
    )

    print(
        "\n── 2. Full algorithm (k=2 drafts + bonus token), Markov-chain toy, 60,000 sequences of length 3 ──"
    )
    mk = toy_markov()
    print(
        f"vs the exact joint distribution of the target ({mk['cells']} cells): TV = {mk['tv']:.4f}, "
        f"chi-square {mk['chi'][0]:.1f} (df {mk['chi'][1]}), p-value {mk['chi'][2]:.2f}; "
        f"mean output per round {mk['tokens_per_round']:.2f} tokens"
    )

    print("\n── 3. Real small models, temperature 1.0 (4 prompts × 200 characters) ──")
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
    print(f"CPU time of normal sampling {tb:.2f} s (median)")
    print(f"{'k':>2} {'measured α':>9} {'formula Σ min(p,q)':>17} {'tok/round':>8} {'speedup':>7}")
    for k in ks:
        s = sts[k]
        print(
            f"{k:2d} {s['accepted'] / s['examined']:10.3f} {s['alpha_sum'] / s['examined']:18.3f} "
            f"{(s['accepted'] + s['rounds']) / s['rounds']:9.2f} "
            f"{tb / statistics.median(t_spec[k]):6.2f}×"
        )
