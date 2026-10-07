"""Speculative decoding (Chapter 25). Experiment module; the main-line training does not use it.

In the decode phase, the model gets only 1 token at a time. The compute units are not fully used,
and the time goes into reading the weights and the KV cache (Chapters 10 and 21).
Thus one forward pass of the target model on k+1 positions takes about the same time as
generating 1 token. Speculative decoding uses this fact:

1. A cheap draft proposes k tokens autoregressively.
2. The target model gets all of them in one pass and gives the distributions p_1..p_{k+1}
   for k+1 positions.
3. Verify the drafts x_i (draft distribution q_i) one at a time, from left to right:
   - Greedy (temperature <= 0): accept x_i if x_i == argmax p_i.
   - Sampling: accept with probability min(1, p_i(x_i) / q_i(x_i)). On a rejection, sample
     a new token from the residual distribution norm(max(0, p_i − q_i)), and end the round.
   If all k drafts are accepted, sample one more "bonus" token from p_{k+1}.
4. The rejected drafts are already in the KV caches of both models. Roll back: only move the
   "valid length" back. `zero.kv_cache.KVCache.update` overwrites from start_pos and returns
   only [0, start_pos+T), so nothing reads the old data that stays after that.

**The quality does not change**: with greedy decoding, the output is identical to greedy decoding
of the target model alone. With sampling, the distribution of each token is exactly the
distribution of the target model (after the same temperature / top-p processing)
(Leviathan et al. 2023, Appendix A.1; Chen et al. 2023).
The probability of one acceptance is α = Σ_x min(p(x), q(x)).

There are two sources of drafts:
- another small model with the same vocabulary (`draft=Transformer`);
- `draft=None`: prompt lookup (n-gram). Find a segment in the existing text that is the same as
  the last n tokens, and use the tokens after it as the draft. The draft distribution is one-hot,
  so the acceptance probability in sampling is p(x).
For the MTP module as the draft (self-speculative), see `mtp_speculative_generate` in
`zero/arch/mtp.py`. It uses `verify` from this file.

Only batch = 1 is supported. Batched speculative decoding must handle the "ragged" sequences that
come from different numbers of accepted tokens. vLLM / SGLang do this in the scheduler.
This file has only two goals: easy to read and correct. On RTX 3090, greedy output on CUDA is
identical to the target model (2026-10, see runs/2026-10-01-gpu0-check/). The performance is not
verified on GPU yet.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import torch

from zero.kv_cache import KVCache
from zero.model import Transformer


@dataclass
class SpeculativeResult:
    """Generation result and acceptance statistics."""

    tokens: list[int]  # new tokens (no prompt; no eos if eos occurs)
    rounds: int = 0  # number of verification forward passes of the target model (no prefill)
    proposed: int = 0  # total number of tokens that the draft proposed
    accepted: int = 0  # number of accepted drafts
    examined: int = 0  # number of drafts actually compared (drafts after the first rejection do not count)
    accepted_per_round: list[int] = field(default_factory=list)

    @property
    def acceptance_rate(self) -> float:
        """Estimate of the per-token acceptance rate α: accepted / compared.

        This is the maximum likelihood estimate if α is i.i.d.
        """
        return self.accepted / self.examined if self.examined else 0.0

    @property
    def tokens_per_round(self) -> float:
        """Mean number of tokens for each forward pass of the target model (1 for normal decoding)."""
        return (self.accepted + self.rounds) / self.rounds if self.rounds else 0.0


def expected_tokens_per_round(alpha: float, k: int) -> float:
    """Formula (1) of Leviathan et al. 2023: (1 − α^{k+1}) / (1 − α)."""
    if alpha >= 1.0:
        return float(k + 1)
    return (1 - alpha ** (k + 1)) / (1 - alpha)


def expected_speedup(alpha: float, k: int, c: float) -> float:
    """Theorem 3.8 of Leviathan et al. 2023: (1 − α^{k+1}) / ((1 − α)(k·c + 1)),
    c = time of one draft step / time of one target step.

    Assumption: the target model verifies k+1 positions as fast as it generates 1 token.
    """
    return expected_tokens_per_round(alpha, k) / (k * c + 1)


def warp_probs(logits: torch.Tensor, temperature: float, top_p: float = 1.0) -> torch.Tensor:
    """logits (..., V) → probabilities after temperature and top-p
    (the same distribution as `zero.generate.sample_next`).

    The guarantee of speculative decoding is for the "processed target distribution":
    p and q must use the same processing.
    """
    probs = torch.softmax(logits.float() / temperature, dim=-1)
    if top_p < 1.0:
        sorted_probs, idx = torch.sort(probs, dim=-1, descending=True)
        keep_sorted = (torch.cumsum(sorted_probs, dim=-1) - sorted_probs) < top_p
        keep = torch.zeros_like(keep_sorted).scatter(-1, idx, keep_sorted)
        probs = probs * keep
        probs = probs / probs.sum(dim=-1, keepdim=True)
    return probs


def verify(
    p_logits: torch.Tensor,
    drafts: Sequence[int],
    q_probs: torch.Tensor | None,
    temperature: float,
    top_p: float = 1.0,
    generator: torch.Generator | None = None,
) -> tuple[int, list[int]]:
    """One round of verification.

    p_logits: (k+1, V) logits of the target model at "each draft position + the last position";
    drafts:   k draft tokens;
    q_probs:  (k, V) draft distributions (after the same temperature / top-p); None means the draft
              is deterministic (one-hot).
    Return (number of accepted tokens m, new tokens of this round: m drafts + 1 correction or bonus token).
    """
    k = len(drafts)
    if temperature <= 0:
        choice = p_logits.argmax(dim=-1).tolist()
        m = 0
        while m < k and drafts[m] == choice[m]:
            m += 1
        return m, list(drafts[:m]) + [choice[m]]

    p = warp_probs(p_logits, temperature, top_p)
    for i, x in enumerate(drafts):
        if q_probs is None:
            q_i = torch.zeros_like(p[i])
            q_i[x] = 1.0
        else:
            q_i = q_probs[i].float()
        ratio = p[i, x] / q_i[x] if q_i[x] > 0 else torch.tensor(0.0)
        if torch.rand((), generator=generator, device=p.device) < torch.clamp(ratio, max=1.0):
            continue  # accept
        residual = torch.clamp(p[i] - q_i, min=0.0)
        if residual.sum() <= 0:  # if p == q (numerically), the residual is 0; a rejection cannot occur then, so sample from p
            residual = p[i]
        tok = int(torch.multinomial(residual / residual.sum(), 1, generator=generator))
        return i, list(drafts[:i]) + [tok]
    bonus = int(torch.multinomial(p[k], 1, generator=generator))
    return k, list(drafts) + [bonus]


def prompt_lookup_draft(seq: Sequence[int], k: int, max_ngram: int = 3) -> list[int]:
    """Prompt lookup (n-gram) draft. Search seq from the end for a segment that is the same as
    the last n tokens (n from large to small).

    Return a maximum of k tokens after that segment, or an empty list if there is no match.
    The cost is almost zero (c ≈ 0).
    """
    L = len(seq)
    for n in range(min(max_ngram, L - 1), 0, -1):
        tail = list(seq[L - n :])
        for start in range(L - n - 1, -1, -1):  # the most recent match first
            if list(seq[start : start + n]) == tail:
                cont = list(seq[start + n : start + n + k])
                if cont:
                    return cont
    return []


def rollback(cache: KVCache, length: int) -> None:
    """Roll back the KV cache to the first length positions.

    A preallocated cache does not need to clear data. It only records the valid length.
    """
    cache.seq_len = min(cache.seq_len, length)


@torch.no_grad()
def speculative_generate(
    target: Transformer,
    draft: Transformer | None,
    prompt_ids: Sequence[int] | torch.Tensor,
    max_new_tokens: int,
    k: int = 4,
    temperature: float = 0.0,
    top_p: float = 1.0,
    seed: int | None = None,
    eos_id: int | None = None,
    max_ngram: int = 3,
) -> SpeculativeResult:
    """Speculative decoding for the target model with a draft model
    (or with prompt lookup if draft=None). batch = 1.

    temperature <= 0: greedy. The output is identical to
    `zero.generate.generate(target, ..., temperature=0)`.
    temperature > 0: sampling. The distribution of each token is the same as the distribution of
    the target model (same temperature / top-p). But the sampled sequence is different from
    `generate` (the random numbers are used in a different way).
    """
    if k < 1:
        raise ValueError("k must be at least 1")
    if draft is not None and draft.config.vocab_size != target.config.vocab_size:
        raise ValueError("The draft model and the target model must use the same vocabulary")
    prompt = (
        prompt_ids.flatten().tolist() if isinstance(prompt_ids, torch.Tensor) else list(prompt_ids)
    )
    if not prompt:
        raise ValueError("The prompt must not be empty")
    device = next(target.parameters()).device
    dtype = next(target.parameters()).dtype
    max_len = min(len(prompt) + max_new_tokens + k + 1, target.config.max_seq_len)
    if draft is not None:
        max_len = min(max_len, draft.config.max_seq_len)
    max_new_tokens = min(max_new_tokens, max_len - len(prompt) - 1)
    gen = None
    if seed is not None:
        gen = torch.Generator(device=device)
        gen.manual_seed(seed)

    t_cache = KVCache.from_config(target.config, 1, max_len, device, dtype)
    d_cache = KVCache.from_config(draft.config, 1, max_len, device, dtype) if draft else None
    t_len = d_len = 0  # number of valid positions in the two caches
    seq = list(prompt)
    res = SpeculativeResult(tokens=[])
    models = [target] + ([draft] if draft is not None else [])
    was_training = [m.training for m in models]
    for m in models:
        m.eval()
    try:
        while len(seq) - len(prompt) < max_new_tokens:
            remaining = max_new_tokens - (len(seq) - len(prompt))
            kk = min(k, remaining - 1, max_len - len(seq) - 1)  # keep a position for the correction/bonus token

            # ── 1. The draft proposes kk tokens ──
            drafts: list[int] = []
            q_rows: list[torch.Tensor] = []
            if kk > 0 and draft is not None:
                assert d_cache is not None
                inp = torch.tensor([seq[d_len:]], device=device)
                logits = draft(inp, kv_cache=d_cache, start_pos=d_len)[0, -1]
                d_len = len(seq)
                for i in range(kk):
                    if temperature <= 0:
                        x = int(logits.argmax())
                    else:
                        q = warp_probs(logits, temperature, top_p)
                        q_rows.append(q)
                        x = int(torch.multinomial(q, 1, generator=gen))
                    drafts.append(x)
                    if i < kk - 1:
                        logits = draft(
                            torch.tensor([[x]], device=device), kv_cache=d_cache, start_pos=d_len
                        )[0, -1]
                        d_len += 1
            elif kk > 0:
                drafts = prompt_lookup_draft(seq, kk, max_ngram)

            # ── 2. One forward pass of the target model verifies them ──
            n = len(drafts)
            inp = torch.tensor([seq[t_len:] + drafts], device=device)
            p_logits = target(inp, kv_cache=t_cache, start_pos=t_len)[0, -(n + 1) :]
            q_probs = torch.stack(q_rows) if q_rows else None
            m, new = verify(p_logits, drafts, q_probs, temperature, top_p, gen)

            # ── 3. Bookkeeping, append the new tokens, roll back the caches ──
            res.rounds += 1
            res.proposed += n
            res.accepted += m
            res.examined += m + (1 if m < n else 0)
            res.accepted_per_round.append(m)
            seq += new
            t_len = len(seq) - 1  # the last new token is not in the cache yet
            rollback(t_cache, t_len)
            if d_cache is not None:
                d_len = min(d_len, len(seq) - 1)
                rollback(d_cache, d_len)
            if eos_id is not None and eos_id in new:
                break
    finally:
        for mdl, was in zip(models, was_training):
            mdl.train(was)

    out = seq[len(prompt) :][:max_new_tokens]
    if eos_id is not None and eos_id in out:
        out = out[: out.index(eos_id)]
    res.tokens = out
    return res
