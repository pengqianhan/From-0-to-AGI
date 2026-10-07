"""Chapter 19 · 02: GRPO from scratch. Imitation reaches its limit; a verifier continues the teaching
(PyTorch, about 1 min on a CPU).

Task: add two one-digit numbers, a + b. Output the digits of the answer, then <eos>
(for example, 7+5 → "1" "2" <eos>). One function can tell if the answer is correct.
This is a "verifiable reward".

1. **Imitation (SFT)**: the teacher is a model that "is not good at carries". It answers all problems
   without a carry correctly. On problems with a carry, only 30% of its answers are correct.
   In 70%, it forgets to write the carry (7+5 becomes "2"). The student learns from the teacher's
   demos, and at the end it also learns the teacher's errors.
2. **GRPO**: sample G answers for the same problem and score them with the verifier. Normalize
   within the group to get the advantage A_i = (r_i − mean(r)) / std(r). Then do a clipped policy
   gradient (plus a k3 KL to the SFT model). There is no value model (critic): the baseline is
   "the mean score of the other answers to the same problem".
3. Parity check with the production code: on the same batch of data, the advantages here are equal to
   zero.post.grpo.group_advantages, and the loss is equal to zero.post.grpo.grpo_loss.

    uv run python chapters/19-reinforcement-learning/code/02_grpo_from_scratch.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(1)  # the build machine shares its CPU between many jobs; you can remove this line

EOS, BOS, V, T = 10, 11, 11, 3  # vocabulary: 0–9 are digits, 10 is <eos>; generate at most 3 tokens
PROMPTS = [(a, b) for a in range(10) for b in range(10)]


def target(a: int, b: int) -> list[int]:
    return [int(c) for c in str(a + b)] + [EOS]


def carry(a: int, b: int) -> bool:
    return a + b >= 10


class TinyPolicy(nn.Module):
    """A tiny autoregressive policy: the next-token distribution depends only on (a, b, position, previous token)."""

    def __init__(self, d: int = 32, h: int = 128):
        super().__init__()
        self.ea, self.eb = nn.Embedding(10, d), nn.Embedding(10, d)
        self.ep, self.eprev = nn.Embedding(T, d), nn.Embedding(12, d)
        self.mlp = nn.Sequential(nn.Linear(d, h), nn.GELU(), nn.Linear(h, V))

    def forward(self, a, b, prev):  # a, b: (B,); prev: (B, T), the "previous token" at each position
        pos = torch.arange(prev.shape[1])
        x = self.ea(a)[:, None] + self.eb(b)[:, None] + self.ep(pos)[None] + self.eprev(prev)
        return self.mlp(x)  # (B, T, V)


def token_logps(model, a, b, seq):
    """seq: (B, T) response tokens (padded with <eos> after <eos>) → log π(y_t | ·) at each position."""
    prev = torch.cat([torch.full_like(seq[:, :1], BOS), seq[:, :-1]], dim=1)
    logits = model(a, b, prev)
    return torch.log_softmax(logits, -1).gather(-1, seq[..., None]).squeeze(-1)


def response_mask(seq):
    """Mask of the response tokens: up to and including the first <eos>."""
    is_eos = (seq == EOS).int()
    before = torch.cumsum(is_eos, 1) - is_eos  # number of <eos> tokens before this position
    return before == 0


@torch.no_grad()
def sample(model, a, b, greedy=False, gen=None):
    seq = torch.full((len(a), T), EOS, dtype=torch.long)
    done = torch.zeros(len(a), dtype=torch.bool)
    for t in range(T):
        prevs = torch.cat([torch.full((len(a), 1), BOS), seq[:, :t]], 1)
        prevs = torch.cat([prevs, torch.full((len(a), T - t - 1), EOS)], 1)
        logits = model(a, b, prevs)[:, t]
        nxt = logits.argmax(-1) if greedy else torch.multinomial(F.softmax(logits, -1), 1, generator=gen)[:, 0]
        nxt = torch.where(done, torch.full_like(nxt, EOS), nxt)
        seq[:, t] = nxt
        done |= nxt == EOS
    return seq


def verify(a: int, b: int, seq: list[int]) -> float:
    """Verifiable reward: 1 if the digits before <eos> are exactly a + b and the output has an <eos>; else 0."""
    if EOS not in seq:
        return 0.0
    return 1.0 if seq[: seq.index(EOS) + 1] == target(a, b) else 0.0


# ---------------------------------------------------------------------------
# The two core functions of GRPO (parity check with zero/post/grpo.py)
# ---------------------------------------------------------------------------


def group_advantages(rewards: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """rewards: (P, G) → A = (r − group mean) / (group std + eps). If a group is all correct or all wrong, A is all 0."""
    mean = rewards.mean(1, keepdim=True)
    std = rewards.std(1, keepdim=True)  # unbiased std, the same as TRL / verl / zero
    return (rewards - mean) / (std + eps)


def grpo_loss(logp, old_logp, ref_logp, adv, mask, eps_clip=0.2, beta=0.02):
    """Per token: −min(ρA, clip(ρ, 1−ε, 1+ε)A) + β·k3. Then take the mean over all response tokens (token_mean)."""
    ratio = torch.exp(logp - old_logp)  # ρ_t = π_θ / π_old
    A = adv[:, None]
    per_tok = torch.maximum(-ratio * A, -torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * A)
    d = ref_logp - logp
    kl = torch.exp(d) - d - 1  # k3 estimate: not negative, its expectation is KL(π_θ ‖ π_ref)
    per_tok = per_tok + beta * kl
    m = mask.float()
    loss = (per_tok * m).sum() / m.sum()
    kl = kl.detach()
    clipped = ((ratio < 1 - eps_clip) | (ratio > 1 + eps_clip)).float()
    return loss, float((kl * m).sum() / m.sum()), float((clipped * m).sum() / m.sum())


# ---------------------------------------------------------------------------


def teacher_demo(a: int, b: int, gen: torch.Generator) -> list[int]:
    """A teacher that is not good at carries: on carry problems, it forgets the carry 70% of the time (7+5 → "2")."""
    if carry(a, b) and torch.rand(1, generator=gen).item() < 0.7:
        return [(a + b) % 10, EOS]
    return target(a, b)


def pad(seq: list[int]) -> list[int]:
    return seq + [EOS] * (T - len(seq))


def evaluate(model, gen):
    a = torch.tensor([p[0] for p in PROMPTS])
    b = torch.tensor([p[1] for p in PROMPTS])
    g = sample(model, a, b, greedy=True).tolist()
    greedy = [verify(x, y, s) for (x, y), s in zip(PROMPTS, g)]
    rep = 20  # sampled accuracy: 20 samples for each problem
    s = sample(model, a.repeat(rep), b.repeat(rep), gen=gen).tolist()
    samp = [verify(x, y, q) for (x, y), q in zip(PROMPTS * rep, s)]
    car = [i for i, p in enumerate(PROMPTS) if carry(*p)]
    return {
        "greedy": sum(greedy) / len(greedy),
        "greedy_carry": sum(greedy[i] for i in car) / len(car),
        "sampled": sum(samp) / len(samp),
    }


def sft(model, gen, steps=600, n_demo=20):
    data = [(a, b, pad(teacher_demo(a, b, gen))) for a, b in PROMPTS for _ in range(n_demo)]
    teacher_acc = sum(verify(a, b, s) for a, b, s in data) / len(data)
    A = torch.tensor([d[0] for d in data])
    B = torch.tensor([d[1] for d in data])
    S = torch.tensor([d[2] for d in data])
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    for _ in range(steps):
        idx = torch.randint(len(data), (256,), generator=gen)
        lp = token_logps(model, A[idx], B[idx], S[idx])
        m = response_mask(S[idx]).float()
        loss = -(lp * m).sum() / m.sum()  # SFT: cross-entropy only on the response tokens
        opt.zero_grad()
        loss.backward()
        opt.step()
    return teacher_acc


def grpo(model, ref, gen, steps=60, P=16, G=8, lr=1e-3, mu=2, log_every=5, crosscheck=None):
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    hist, example = [], None
    for step in range(1, steps + 1):
        idx = torch.randint(len(PROMPTS), (P,), generator=gen)
        a = torch.tensor([PROMPTS[i][0] for i in idx]).repeat_interleave(G)  # G copies of each problem
        b = torch.tensor([PROMPTS[i][1] for i in idx]).repeat_interleave(G)
        seq = sample(model, a, b, gen=gen)  # 1. sample
        r = torch.tensor([verify(x, y, s) for x, y, s in zip(a.tolist(), b.tolist(), seq.tolist())])
        rewards = r.view(P, G)  # 2. score
        adv = group_advantages(rewards).view(-1)  # 3. advantages, normalized within each group
        mask = response_mask(seq)
        with torch.no_grad():
            old = token_logps(model, a, b, seq)  # π_old: the policy at sampling time
            reflp = token_logps(ref, a, b, seq)  # π_ref: the SFT model (frozen)
        if example is None:  # keep the first group with both correct and wrong samples, for the video
            for j in range(P):
                if rewards[j].std() > 0 and carry(int(a[j * G]), int(b[j * G])):
                    example = {
                        "prompt": [int(a[j * G]), int(b[j * G])],
                        "responses": [
                            "".join(str(t) for t in s[: s.index(EOS)]) if EOS in s else "…"
                            for s in seq[j * G : (j + 1) * G].tolist()
                        ],
                        "rewards": rewards[j].tolist(),
                        "adv": group_advantages(rewards[j : j + 1])[0].tolist(),
                    }
                    break
        for _ in range(mu):  # 4. update μ times on the same batch; from the 2nd update, ρ ≠ 1 and clipping can act
            logp = token_logps(model, a, b, seq)
            loss, kl, clip = grpo_loss(logp, old, reflp, adv, mask)
            if crosscheck is not None and step == 1:
                crosscheck(rewards, adv, logp, old, reflp, mask, loss)
                crosscheck = None
            opt.zero_grad()
            loss.backward()
            opt.step()
        hist.append({"step": step, "reward": float(r.mean()), "kl": kl, "clip": clip,
                     "zero_std": float((rewards.std(1) == 0).float().mean())})
        if step % log_every == 0:
            ev = evaluate(model, gen)
            hist[-1].update(ev)
            print(f"  step {step:>3} | batch mean reward {r.mean():.2f} | greedy acc {ev['greedy']:.2f}"
                  f" (carry {ev['greedy_carry']:.2f}) | sampled acc {ev['sampled']:.2f}"
                  f" | KL {kl:.3f} | clip ratio {clip:.2f} | zero-std groups {hist[-1]['zero_std']:.2f}")
    return hist, example


def crosscheck_with_zero(rewards, adv, logp, old, ref, mask, loss):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # the repository root, so that we can import zero
    from zero.post.grpo import group_advantages as z_adv
    from zero.post.grpo import grpo_loss as z_loss

    d_adv = (z_adv(rewards).view(-1) - adv).abs().max().item()
    z, _ = z_loss(logp, old, adv, mask, clip_eps=0.2, ref_logp=ref, kl_coef=0.02, loss_agg="token_mean")
    print(f"  Parity check with zero.post.grpo: max advantage diff {d_adv:.2e}, loss {loss.item():+.6f} vs {z.item():+.6f}"
          f" (diff {abs(loss.item() - z.item()):.2e})")


def run(seed: int = 0, verbose: bool = True):
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    model = TinyPolicy()
    teacher_acc = sft(model, gen)
    ev_sft = evaluate(model, gen)
    if verbose:
        print(f"== 1. Imitation: accuracy of the teacher demos {teacher_acc:.3f} (all correct without a carry, only 30% with a carry) ==")
        print(f"  After SFT: greedy accuracy {ev_sft['greedy']:.2f} (carry problems {ev_sft['greedy_carry']:.2f}), "
              f"sampled accuracy {ev_sft['sampled']:.2f}")
        print("\n== 2. GRPO: G=8, 16 problems per step, ε=0.2, β=0.02, μ=2 ==")
    ref = TinyPolicy()
    ref.load_state_dict(model.state_dict())
    ref.requires_grad_(False)
    hist, example = grpo(model, ref, gen, crosscheck=crosscheck_with_zero if verbose else None)
    hist.insert(0, {"step": 0, **ev_sft})
    return {"teacher_acc": teacher_acc, "sft": ev_sft, "hist": hist, "example": example}


def main() -> None:
    out = run()
    ex = out["example"]
    print(f"\n== 3. One group of samples (step 1, problem {ex['prompt'][0]}+{ex['prompt'][1]}) ==")
    for resp, r, A in zip(ex["responses"], ex["rewards"], ex["adv"]):
        print(f"  answer {resp:>3}  reward {r:.0f}  advantage {A:+.2f}")


if __name__ == "__main__":
    main()
