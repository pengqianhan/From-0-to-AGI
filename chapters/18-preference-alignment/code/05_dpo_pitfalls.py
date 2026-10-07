"""Chapter 18 · Minimal code 5: pitfalls of DPO (the same small model and task as 04).

① Learning rate too large: the margin goes very high, but the model breaks. The probability of
   the correct answer on held-out prompts goes down, and many samples do not have the correct format.
   The smoke test of zero had the same problem (see the comments in configs/tiny/dpo.toml): with
   lr = 5e-4, the margin went to 4.8 in 24 steps. Then the format accuracy in the GRPO stage fell
   from 0.16 to 0. With 5e-5, training was more stable.
② The probability of chosen can go down too. DPO only asks that "chosen goes up more (or goes down
   less) than rejected". It does not ask that chosen itself goes up. When rejected and chosen are
   similar (the wrong answer is off by only 1 or 2), the log-probability of chosen goes down
   all the time in this toy, and the held-out results become worse.
③ Overfitting: the implicit-reward accuracy on the training set is much higher than on held-out prompts.

Run: uv run python chapters/18-preference-alignment/code/05_dpo_pitfalls.py   (about 1 min of CPU time; a few minutes of wall time on a busy machine)
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

torch.set_num_threads(1)  # The build machine shares its CPU between jobs (you can remove this line).

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("toy04", HERE / "04_toy_dpo.py")
toy = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(toy)

LRS = (1e-4, 1e-3, 1e-2, 5e-2)


def lr_sweep() -> list[dict]:
    ref = toy.sft_model()
    base = toy.evaluate(ref, ref)
    rows = [{"lr": 0.0, "margin": 0.0, "acc": 0.0, "logp_w": float("nan"), **base}]
    for lr in LRS:
        pol, hh = toy.run_dpo(beta=0.1, lr=lr, steps=150, log_every=150)
        rows.append({"lr": lr, "margin": hh[-1]["margin"], "acc": hh[-1]["acc"],
                     "logp_w": hh[-1]["logp_w"], **toy.evaluate(pol, ref)})
    return rows


def near_miss() -> dict:
    ref = toy.sft_model("near")
    base = toy.evaluate(ref, ref, mistakes="near")
    pol, hist = toy.run_dpo(beta=0.1, lr=1e-3, steps=150, log_every=25, mistakes="near")
    return {"base": base, "hist": hist, "after": toy.evaluate(pol, ref, mistakes="near")}


def main() -> None:
    rows = lr_sweep()
    print("① Sweep the learning rate (β = 0.1, 150 steps; the first row, lr = 0, is the SFT reference model itself).\n"
          "   P(correct) is on the held-out prompts. 'well-formed' and 'correct' are fractions of samples:")
    print(f"   {'lr':>6} | {'margin':>7} | {'tr. acc':>6} | {'  P(correct)':>10} | {' well-formed':>10} | {' correct':>7}")
    for r in rows:
        print(f"   {r['lr']:>6.0e} | {r['margin']:>+7.2f} | {r['acc']:>7.2f} | {r['p_correct']:>12.3f} | "
              f"{r['format']:>12.3f} | {r['sample_acc']:>8.3f}")
    print("   Larger margin ≠ better. When lr is too large, the model pushes rejected down "
          "and damages the probabilities of all 'digit' tokens.")

    nm = near_miss()
    print("\n② The wrong answers are off by only 1 or 2 (chosen and rejected are similar), β = 0.1, lr = 1e-3:")
    print(f"   {'step':>4} | {'margin':>7} | {'acc':>5} | {'log π(chosen)':>13} | {'log π(rejected)':>15}")
    for h in nm["hist"]:
        print(f"   {h['step']:>4} | {h['margin']:>+7.3f} | {h['acc']:.2f} | {h['logp_w']:>13.3f} | {h['logp_l']:>15.3f}")
    b, a = nm["base"], nm["after"]
    print(f"   held-out: P(correct) {b['p_correct']:.3f} → {a['p_correct']:.3f} | correct samples {b['sample_acc']:.3f} → "
          f"{a['sample_acc']:.3f} | well-formed {b['format']:.3f} → {a['format']:.3f}")
    print("   The loss goes down, the margin goes up, and the training acc goes up. "
          "But the probability of chosen also goes down, and the held-out results become worse.")
    print(f"\n③ Implicit-reward acc: training set {nm['hist'][-1]['acc']:.2f} vs held-out {a['pref_acc']:.2f} (in ①, lr = 1e-3: "
          f"training {rows[2]['acc']:.2f} vs held-out {rows[2]['pref_acc']:.2f}). Training metrics alone overestimate the result.")


if __name__ == "__main__":
    main()
