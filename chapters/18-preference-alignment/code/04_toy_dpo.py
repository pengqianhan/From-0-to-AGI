"""Chapter 18 · Minimal code 4: DPO on a small language model.

Task: the prompt is "a+b=" (a and b are 0–9). The answer is the digits of the sum, and then ";".
① "SFT": train a small GRU language model on data of **mixed quality**. For the same prompt, 40% of
   the demonstrations are correct, and 60% are random wrong numbers. The model learns the format,
   and it also learns to "answer wrong often". This model is the reference model π_ref.
② Preference data: 70 prompts, 4 pairs each (chosen = the correct answer, rejected = a wrong
   answer that π_ref can write). The other 30 prompts are held out. Training never sees them.
③ DPO: see how the implicit-reward margin, the accuracy, and the log-probabilities of chosen and
   rejected change during training. On the held-out prompts: "the probability of the correct
   answer" and the fraction of well-formed samples.
④ Sweep β.
For pitfalls (a learning rate that is too large, the probability of chosen goes down too), see
05_dpo_pitfalls.py.

Run: uv run python chapters/18-preference-alignment/code/04_toy_dpo.py   (about 30 s of CPU time; a few minutes of wall time on a busy machine)
"""

from __future__ import annotations

import copy
import random
from functools import lru_cache

import torch
import torch.nn.functional as F

torch.set_num_threads(1)  # The build machine shares its CPU between jobs (you can remove this line).

VOCAB = list("0123456789+=;")
STOI = {c: i for i, c in enumerate(VOCAB)}
EOS = STOI[";"]
PROMPTS = [(a, b) for a in range(10) for b in range(10)]
_rng = random.Random(0)
_order = PROMPTS[:]
_rng.shuffle(_order)
TRAIN_PROMPTS, HELDOUT_PROMPTS = _order[:70], _order[70:]
P_CORRECT_SFT = 0.4


def encode(s: str) -> list[int]:
    return [STOI[c] for c in s]


def wrong_answers(a: int, b: int, mistakes: str = "random") -> list[int]:
    """Wrong answers. "random": any wrong number in 0–18. "near": off by only 1 or 2 (05 uses it to show a pitfall)."""
    if mistakes == "near":
        return [a + b + d for d in (-2, -1, 1, 2) if a + b + d >= 0]
    return [x for x in range(19) if x != a + b]


class TinyLM(torch.nn.Module):
    """Character-level GRU language model (about 20k parameters).

    If you use a Transformer here, no line of the DPO code below changes.
    """

    def __init__(self, d: int = 32, h: int = 64) -> None:
        super().__init__()
        self.emb = torch.nn.Embedding(len(VOCAB), d)
        self.rnn = torch.nn.GRU(d, h, batch_first=True)
        self.head = torch.nn.Linear(h, len(VOCAB))

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(self.emb(ids))
        return self.head(out)


def pad(seqs: list[list[int]]) -> tuple[torch.Tensor, torch.Tensor]:
    L = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), L), EOS, dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, : len(s)] = torch.tensor(s)
    return ids, torch.tensor([len(s) for s in seqs])


def response_logps(model: TinyLM, prompts: list[tuple[int, int]], answers: list[int]) -> torch.Tensor:
    """For each "prompt + answer", sum the log-probabilities **of the answer part only**.

    The prompt does not count, the same as in zero.
    """
    seqs, starts = [], []
    for (a, b), ans in zip(prompts, answers):
        p = encode(f"{a}+{b}=")
        seqs.append(p + encode(f"{ans};"))
        starts.append(len(p))
    ids, lens = pad(seqs)
    logp = torch.log_softmax(model(ids[:, :-1]), -1).gather(-1, ids[:, 1:, None]).squeeze(-1)
    pos = torch.arange(ids.shape[1] - 1)[None, :] + 1  # logp[:, t] predicts token t+1
    mask = (pos >= torch.tensor(starts)[:, None]) & (pos < lens[:, None])
    return (logp * mask).sum(-1)


@lru_cache(maxsize=2)
def sft_model(mistakes: str = "random") -> TinyLM:
    """① SFT on demonstrations of mixed quality: 40% correct, 60% wrong."""
    torch.manual_seed(0)
    g = random.Random(1)
    model = TinyLM()
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    for _ in range(1500):
        batch = [g.choice(PROMPTS) for _ in range(64)]
        ans = [a + b if g.random() < P_CORRECT_SFT else g.choice(wrong_answers(a, b, mistakes))
               for a, b in batch]
        loss = -response_logps(model, batch, ans).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model.eval()


def make_pairs(mistakes: str = "random", seed: int = 0, per_prompt: int = 4) -> list[tuple[tuple[int, int], int, int]]:
    """② Preference pairs: (prompt, chosen = the correct answer, rejected = a mistake that π_ref can make)."""
    g = random.Random(seed)
    return [((a, b), a + b, g.choice(wrong_answers(a, b, mistakes)))
            for a, b in TRAIN_PROMPTS for _ in range(per_prompt)]


@torch.no_grad()
def sample(model: TinyLM, prompts: list[tuple[int, int]], n: int, seed: int, max_new: int = 4) -> list[str]:
    """Sample at temperature 1. Return the generated strings (without the prompt)."""
    g = torch.Generator().manual_seed(seed)
    outs = []
    for a, b in prompts:
        ids = torch.tensor([encode(f"{a}+{b}=")] * n)
        gen = [""] * n
        done = torch.zeros(n, dtype=torch.bool)
        for _ in range(max_new):
            probs = torch.softmax(model(ids)[:, -1], -1)
            nxt = torch.multinomial(probs, 1, generator=g)
            ids = torch.cat([ids, nxt], 1)
            for i in range(n):
                if not done[i]:
                    gen[i] += VOCAB[int(nxt[i])]
            done |= nxt[:, 0] == EOS
            if done.all():
                break
        outs.extend(gen)
    return outs


def well_formed(s: str) -> bool:
    return s.endswith(";") and 1 <= len(s) - 1 <= 2 and s[:-1].isdigit()


@torch.no_grad()
def evaluate(model: TinyLM, ref: TinyLM, n_samples: int = 20, mistakes: str = "random") -> dict:
    """Held-out prompts: probability of the correct answer, fractions of well-formed and correct samples,
    and the preference accuracy of the implicit reward (independent of β: only the sign counts)."""
    prompts = HELDOUT_PROMPTS
    p_correct = response_logps(model, prompts, [a + b for a, b in prompts]).exp().mean()
    outs = sample(model, prompts, n_samples, seed=123)
    truth = [f"{a + b};" for a, b in prompts for _ in range(n_samples)]
    fmt = sum(well_formed(s) for s in outs) / len(outs)
    acc = sum(s == t for s, t in zip(outs, truth)) / len(outs)
    g = random.Random(7)
    wrong = [g.choice(wrong_answers(a, b, mistakes)) for a, b in prompts]
    right = [a + b for a, b in prompts]
    h = (response_logps(model, prompts, right) - response_logps(ref, prompts, right)) - (
        response_logps(model, prompts, wrong) - response_logps(ref, prompts, wrong))
    return {"p_correct": float(p_correct), "format": fmt, "sample_acc": acc,
            "pref_acc": float((h > 0).float().mean())}


def run_dpo(beta: float = 0.1, lr: float = 1e-3, steps: int = 150, bsz: int = 32, seed: int = 0,
            log_every: int = 0, mistakes: str = "random") -> tuple[TinyLM, list[dict]]:
    """③ DPO training. The log-probabilities of the reference model are computed before training
    (ref_mode = "precompute" in zero)."""
    ref = sft_model(mistakes)
    policy = copy.deepcopy(ref).train()
    pairs = make_pairs(mistakes)
    prompts = [p for p, _, _ in pairs]
    yw = [w for _, w, _ in pairs]
    yl = [l for _, _, l in pairs]
    with torch.no_grad():
        ref_w, ref_l = response_logps(ref, prompts, yw), response_logps(ref, prompts, yl)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    g = torch.Generator().manual_seed(seed)
    hist = []
    for step in range(steps + 1):
        if log_every and step % log_every == 0:
            with torch.no_grad():
                pw, pl = response_logps(policy, prompts, yw), response_logps(policy, prompts, yl)
            h = beta * ((pw - ref_w) - (pl - ref_l))
            hist.append({"step": step, "loss": float(-F.logsigmoid(h).mean()),
                         "margin": float(h.mean()), "acc": float((h > 0).float().mean()),
                         "logp_w": float(pw.mean()), "logp_l": float(pl.mean())})
        if step == steps:
            break
        idx = torch.randint(0, len(pairs), (bsz,), generator=g)
        pw = response_logps(policy, [prompts[i] for i in idx], [yw[i] for i in idx])
        pl = response_logps(policy, [prompts[i] for i in idx], [yl[i] for i in idx])
        h = beta * ((pw - ref_w[idx]) - (pl - ref_l[idx]))  # implicit reward difference
        loss = -F.logsigmoid(h).mean()                       # DPO loss
        opt.zero_grad()
        loss.backward()
        opt.step()
    return policy.eval(), hist


def main_results() -> dict:
    """All numbers of 04 (the video scenes.py also calls this function)."""
    ref = sft_model()
    base = evaluate(ref, ref)
    policy, hist = run_dpo(beta=0.1, lr=1e-3, steps=150, log_every=25)
    after = evaluate(policy, ref)
    sweep = []
    for beta in (0.03, 0.1, 0.3, 1.0):
        pol, hh = (policy, [hist[0], hist[-1]]) if beta == 0.1 else run_dpo(beta=beta, log_every=150)
        e = evaluate(pol, ref)
        sweep.append({"beta": beta, "loss": hh[-1]["loss"], "margin": hh[-1]["margin"],
                      "d_logp_w": hh[-1]["logp_w"] - hh[0]["logp_w"],
                      "d_logp_l": hh[-1]["logp_l"] - hh[0]["logp_l"], "p_correct": e["p_correct"],
                      "sample_acc": e["sample_acc"]})
    return {"base": base, "hist": hist, "after": after, "n_pairs": len(make_pairs()), "beta_sweep": sweep}


def main() -> None:
    R = main_results()
    base, hist, after = R["base"], R["hist"], R["after"]
    print(f"① SFT reference model ({P_CORRECT_SFT:.0%} of the demonstrations are correct) on 30 held-out prompts:")
    print(f"   P(correct) {base['p_correct']:.3f} | well-formed samples {base['format']:.3f} | "
          f"correct samples {base['sample_acc']:.3f}")
    print(f"\n② Preference data: {R['n_pairs']} pairs (70 prompts × 4), chosen = correct answer, rejected = wrong answer")
    print("③ DPO (β = 0.1, lr = 1e-3, 32 pairs per step, 150 steps), metrics on the training set:")
    print(f"   {'step':>4} | {'loss':>6} | {'margin':>7} | {'acc':>5} | {'log π(chosen)':>13} | {'log π(rejected)':>15}")
    for h in hist:
        print(f"   {h['step']:>4} | {h['loss']:.4f} | {h['margin']:>+7.3f} | {h['acc']:.2f} | "
              f"{h['logp_w']:>13.3f} | {h['logp_l']:>15.3f}")
    print(f"   held-out: P(correct) {base['p_correct']:.3f} → {after['p_correct']:.3f} | "
          f"correct samples {base['sample_acc']:.3f} → {after['sample_acc']:.3f} | "
          f"well-formed {base['format']:.3f} → {after['format']:.3f} | implicit reward ranks correctly {after['pref_acc']:.2f}")

    print("\n④ Sweep β (lr = 1e-3, 150 steps; values at the last step):")
    print("   margin = difference of the implicit rewards (β is included). margin/β = difference of the log ratios: "
          "how far the policy moved from π_ref. P(correct) is on the held-out prompts.")
    print(f"   {'β':>5} | {'loss':>7} | {'margin':>7} | {'margin/β':>8} | {'Δlog π(chosen)':>14} | {'Δlog π(rejected)':>16} | {'P(correct)':>10}")
    for r in R["beta_sweep"]:
        print(f"   {r['beta']:>5} | {r['loss']:>7.4f} | {r['margin']:>+7.3f} | {r['margin'] / r['beta']:>+8.2f} | "
              f"{r['d_logp_w']:>+14.3f} | {r['d_logp_l']:>+16.3f} | {r['p_correct']:>10.3f}")


if __name__ == "__main__":
    main()
