"""Chapter 25 · Minimal code 2: greedy speculative decoding. The output is token-for-token
identical to the target model, but faster.

Each round:
  1. The draft model guesses k tokens autoregressively (cheap: the draft is small).
  2. The target model feeds "the last confirmed token + k drafts" in one forward pass
     (k+1 positions get scores in parallel).
  3. Compare from left to right. If draft d_i equals the argmax of the target at that position,
     accept it. At the first difference, stop and use the argmax of the target (correction).
     If all k drafts are correct, the target gives one more token free at the last position (bonus).
  4. The rejected draft tokens are already in the KV caches of both models. Roll them back (truncate).
Each round produces at least 1 token (correction or bonus) and at most k+1 tokens.
The target model runs only once per round.

Run: uv run python chapters/25-mtp-speculative-decoding/code/02_greedy_speculative.py
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


m1 = _load("ch25_models", "01_models_and_cost.py")
KVCache, truncate = m1.KVCache, m1.truncate


@torch.no_grad()
def greedy_generate(model, prompt: list[int], n_new: int) -> list[int]:
    """Baseline: normal greedy decoding of the target model with a KV cache. One forward pass, one token per step."""
    cache = KVCache(model.c.n_layers)
    logits = model(torch.tensor([prompt]), cache)[0, -1]
    out = []
    for _ in range(n_new):
        nxt = int(logits.argmax())
        out.append(nxt)
        logits = model(torch.tensor([[nxt]]), cache)[0, -1]
    return out


@torch.no_grad()
def speculative_greedy(target, draft, prompt: list[int], n_new: int, k: int, trace=None):
    """Greedy speculative decoding. Returns (new tokens, statistics).

    If trace is a list, the function records (drafts, answers of the target, number accepted) for each round."""
    seq = list(prompt)
    tc, dc = KVCache(target.c.n_layers), KVCache(draft.c.n_layers)
    stats = dict(rounds=0, proposed=0, accepted=0, examined=0)
    while len(seq) - len(prompt) < n_new:
        # ── 1. Draft: first feed the tokens that are not in its cache yet, then guess k tokens one by one ──
        logits = draft(torch.tensor([seq[len(dc) :]]), dc)[0, -1]
        drafts = []
        for i in range(k):
            drafts.append(int(logits.argmax()))
            if i < k - 1:
                logits = draft(torch.tensor([[drafts[-1]]]), dc)[0, -1]
        # ── 2. Target: verify all k drafts in one forward pass (plus the confirmed tokens not in its cache) ──
        feed = seq[len(tc) :] + drafts
        p_logits = target(torch.tensor([feed]), tc)[0, -(k + 1) :]  # The last k+1 positions
        choice = p_logits.argmax(-1).tolist()  # choice[i] = the answer of the target at the position of draft i
        # ── 3. Accept from left to right; stop at the first difference ──
        m = 0
        while m < k and drafts[m] == choice[m]:
            m += 1
        seq += drafts[:m] + [choice[m]]  # m drafts + 1 correction (the bonus token when m == k)
        if trace is not None:
            trace.append((drafts, choice, m))
        stats["rounds"] += 1
        stats["proposed"] += k
        stats["accepted"] += m
        stats["examined"] += m + (1 if m < k else 0)  # Number of drafts that were compared
        # ── 4. Roll back: keep in the cache only the positions that are confirmed and are not the last one ──
        truncate(tc, len(seq) - 1)
        truncate(dc, min(len(dc), len(seq) - 1))
    return seq[len(prompt) :][:n_new], stats


def prompts(n: int = 4, length: int = 40) -> list[list[int]]:
    """Cut a few pieces from the validation set to use as prompts."""
    data = m1.ch10.CharData()
    val = data.val.tolist()
    step = len(val) // (n + 1)
    return [val[(i + 1) * step : (i + 1) * step + length] for i in range(n)]


def timed(fn, *args) -> tuple[float, object]:
    """Returns (CPU time of this process, result).

    On a shared machine, the wall-clock time shows mostly the wait in the queue. See the note in 01."""
    t0 = time.process_time()
    out = fn(*args)
    return time.process_time() - t0, out


def expected_tokens(alpha: float, k: int) -> float:
    """Formula (1) of Leviathan et al.: each round gives (1 − α^{k+1}) / (1 − α) tokens on average."""
    return (1 - alpha ** (k + 1)) / (1 - alpha)


if __name__ == "__main__":
    target, draft = m1.load_target(), m1.load_draft()
    data = m1.ch10.CharData()
    P = prompts()
    N = 200

    # ── Correctness: greedy speculative decoding must give the same tokens as greedy decoding of the target ──
    base_out = [greedy_generate(target, p, N) for p in P]
    for k in (1, 3, 5, 8):
        same = all(speculative_greedy(target, draft, p, N, k)[0] == b for p, b in zip(P, base_out))
        print(f"k={k}: {len(P)} prompts × {N} tokens, identical to greedy decoding of the target model: {same}")
    print("\nExample (prompt + generated text):\n" + data.decode(P[0]) + "|" + data.decode(base_out[0][:120]))

    # ── Acceptance rate and speed: measure 3 times in turns and take the median. This reduces the effect of CPU load changes. ──
    ks = (1, 2, 3, 4, 5, 6, 8)
    t_base, t_spec, st = [], {k: [] for k in ks}, {}
    for _ in range(3):
        t_base.append(sum(timed(greedy_generate, target, p, N)[0] for p in P))
        for k in ks:
            tot, agg = 0.0, dict(rounds=0, proposed=0, accepted=0, examined=0)
            for p in P:
                dt, (_, s) = timed(speculative_greedy, target, draft, p, N, k)
                tot += dt
                for key in agg:
                    agg[key] += s[key]
            t_spec[k].append(tot)
            st[k] = agg
    tb = statistics.median(t_base)
    c = m1.forward_time(draft, 200, 1) / m1.forward_time(target, 200, 1)
    print(
        f"\nNormal greedy decoding of the target model: {len(P) * N} tokens in {tb:.2f} s (median); cost coefficient c ≈ {c:.2f}"
    )
    print(
        f"{'k':>2} {'accept α':>13} {'tok/round':>13} {'eq. (1)':>8} "
        f"{'target fwd':>11} {'CPU s':>7} {'speedup':>7} {'theory':>8}"
    )
    for k in ks:
        s = st[k]
        alpha = s["accepted"] / s["examined"]
        per_round = (s["accepted"] + s["rounds"]) / s["rounds"]  # Per round = number accepted + 1
        ts = statistics.median(t_spec[k])
        pred = expected_tokens(alpha, k) / (k * c + 1)
        print(
            f"{k:2d} {alpha:13.3f} {per_round:13.2f} {expected_tokens(alpha, k):8.2f} "
            f"{s['rounds']:11d} {ts:7.2f} {tb / ts:6.2f}× {pred:7.2f}×"
        )
    print(
        "('target fwd' is the count for one measurement of 4 prompts × 200 tokens; normal decoding needs 800."
        "\n  The times are process CPU times and still change. Look only at the trend of the speedup."
        "\n  The prediction assumes that verifying k+1 tokens costs the same as generating 1.)"
    )
