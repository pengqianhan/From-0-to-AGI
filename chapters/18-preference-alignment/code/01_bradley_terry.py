"""Chapter 18 · Minimal code 1: the Bradley–Terry reward model.

A preference pair only says "a is better than b". It does not give a score. The Bradley–Terry model
assumes that each answer has a hidden score r, and
    P(a ≻ b) = σ(r(a) − r(b))
To train a reward model, maximize the log of this probability. This is the binary cross-entropy of
Chapter 5. The label is always "chosen wins".

Toy setup: two features describe each answer: the quality q and the length ℓ (unit: 100 tokens).
The labeler decides with r_label = 1.5·q + 0.4·ℓ, with noise (the labeler picks the winner with
probability σ). Thus, **the labeler prefers long answers a little**.
We only see "who won". We check if the reward model can learn the two coefficients back.

Run: uv run python chapters/18-preference-alignment/code/01_bradley_terry.py
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

torch.set_num_threads(1)  # The build machine shares its CPU between jobs (you can remove this line).

W_LABEL = torch.tensor([1.5, 0.4])  # The hidden score of the labeler: 1.5·quality + 0.4·length


def make_answers(n: int, g: torch.Generator) -> torch.Tensor:
    """Features of n answers, shape (n, 2): quality q ~ N(0,1), length ℓ ~ U(0.2, 4) (in 100 tokens)."""
    q = torch.randn(n, generator=g)
    length = 0.2 + 3.8 * torch.rand(n, generator=g)
    return torch.stack([q, length], dim=1)


def make_pairs(n_pairs: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (chosen features, rejected features), each (n_pairs, 2).

    The labeler picks the winner with the Bradley–Terry probability.
    """
    g = torch.Generator().manual_seed(seed)
    a, b = make_answers(n_pairs, g), make_answers(n_pairs, g)
    p_a_wins = torch.sigmoid(a @ W_LABEL - b @ W_LABEL)  # P(a ≻ b) = σ(r(a) − r(b))
    a_wins = torch.rand(n_pairs, generator=g) < p_a_wins
    chosen = torch.where(a_wins[:, None], a, b)
    rejected = torch.where(a_wins[:, None], b, a)
    return chosen, rejected


class LinearRM(torch.nn.Module):
    """The simplest reward model: r(x) = w·x + c.

    A real reward model is "an LLM + a head that outputs a scalar". The loss is the same.
    """

    def __init__(self) -> None:
        super().__init__()
        self.lin = torch.nn.Linear(2, 1)
        torch.nn.init.zeros_(self.lin.weight)  # Start with no knowledge: all answers get the same score, loss = log 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin(x).squeeze(-1)


def bt_loss(r_chosen: torch.Tensor, r_rejected: torch.Tensor) -> torch.Tensor:
    """Bradley–Terry negative log-likelihood: −log σ(r_w − r_l)."""
    return -F.logsigmoid(r_chosen - r_rejected).mean()


def train_rm(steps: int = 600, lr: float = 0.1, seed: int = 0) -> tuple[LinearRM, list[dict]]:
    torch.manual_seed(seed)
    cw, cl = make_pairs(4000, seed=1)
    rm = LinearRM()
    c0 = float(rm.lin.bias.detach())
    opt = torch.optim.Adam(rm.parameters(), lr=lr)
    hist = []
    for step in range(steps + 1):
        loss = bt_loss(rm(cw), rm(cl))
        if step in (0, 10, 50, 200, steps):
            w = rm.lin.weight.detach()[0]
            hist.append({"step": step, "loss": float(loss.detach()), "w_q": float(w[0]), "w_len": float(w[1])})
        opt.zero_grad()
        loss.backward()
        opt.step()
    rm.bias_moved = abs(float(rm.lin.bias.detach()) - c0)  # type: ignore[attr-defined]
    return rm, hist


def accuracy(r_fn, cw: torch.Tensor, cl: torch.Tensor) -> float:  # noqa: ANN001
    return float((r_fn(cw) > r_fn(cl)).float().mean())


def main() -> None:
    print("① σ changes a 'score difference' into a 'win probability': P(a ≻ b) = σ(r(a) − r(b))")
    print("   r(a) − r(b):", "  ".join(f"{d:+.0f}" for d in (-4, -2, -1, 0, 1, 2, 4)))
    print("   P(a ≻ b)   :", "  ".join(f"{torch.sigmoid(torch.tensor(float(d))):.2f}" for d in (-4, -2, -1, 0, 1, 2, 4)))
    print("   Note: only the 'difference' goes into the formula. Add 100 to all scores, and no probability changes.")

    rm, hist = train_rm()
    print("\n② Train a linear reward model r(x) = w_q·quality + w_len·length + c on 4000 preference pairs")
    print(f"   {'step':>4} | {'loss':>6} | {'w_q':>6} | {'w_len':>6}")
    for h in hist:
        print(f"   {h['step']:>4} | {h['loss']:.4f} | {h['w_q']:6.3f} | {h['w_len']:6.3f}")
    print(f"   True coefficients of the labeler: w_q = {W_LABEL[0]:.1f}, w_len = {W_LABEL[1]:.1f}")
    print(f"   Change of the bias c during training: {rm.bias_moved:.1e} "
          "(the gradient is always 0: the constant cancels in the 'difference')")

    cw, cl = make_pairs(4000, seed=2)  # held-out set
    with torch.no_grad():
        acc_rm = accuracy(rm, cw, cl)
        acc_oracle = accuracy(lambda x: x @ W_LABEL, cw, cl)
        acc_quality = accuracy(lambda x: x[:, 0], cw, cl)
    print("\n③ Held-out set: fraction of pairs where the score of chosen is higher")
    print(f"   learned reward model                : {acc_rm:.3f}")
    print(f"   score of the labeler (upper limit)  : {acc_oracle:.3f}  ← the labels have noise, so the limit is not 1")
    print(f"   quality q only                      : {acc_quality:.3f}")

    # ④ The loss is the binary cross-entropy.
    with torch.no_grad():
        d = rm(cw) - rm(cl)
        ours = bt_loss(rm(cw), rm(cl))
        bce = F.binary_cross_entropy_with_logits(d, torch.ones_like(d))
    print("\n④ Bradley–Terry loss = binary cross-entropy with logit r_w − r_l and label always 1 (Chapter 5)")
    print(f"   −log σ(r_w − r_l) = {float(ours):.6f}   BCEWithLogits = {float(bce):.6f}")
    print(f"\nSummary: the reward model also learned 'length' as a good property (w_len ≈ {hist[-1]['w_len']:.2f}). "
          "It learned the preference of the labeler exactly, and also the bias. "
          "Next (02): what happens when a policy pushes hard for this score?")


if __name__ == "__main__":
    main()
