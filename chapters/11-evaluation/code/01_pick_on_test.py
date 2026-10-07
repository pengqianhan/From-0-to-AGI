"""Why we "set the exam first": a pick on the test set makes the score too high on average.

This file is a minimal demo of Goodhart's law.

Setup: K "models" have exactly the same true skill. (Think of K checkpoints, K sets of
hyperparameters, or K prompts.) Each model takes the same test set of n = 200 questions once.
We look at the test scores, pick the highest one, and report its score.
The reported value is higher than the true skill, but no model is really better.

    uv run python chapters/11-evaluation/code/01_pick_on_test.py

Method (NumPy only, a few seconds):
- Each question has its own difficulty p_i (drawn from Beta(2, 2), mean 0.5). All models answer
  the same question correctly with probability p_i. Thus the results of the models are correlated.
  Real models are the same: all models fail on hard questions.
- Each model flips one coin per question, with probability p_i.
- "Pick on the test set": report the highest test score of the K models.
- "Pick on the dev set": pick on a different dev set from the same distribution, then take the
  test once. The reported value is unbiased.
"""

from __future__ import annotations

import numpy as np

N_TEST = 200  # number of questions in the test set
N_TRIALS = 2000  # repeat the full "experiment" this many times and take the mean
TRUE_ACC = 0.5  # true skill of all models (the mean of Beta(2,2))


def one_trial(rng: np.random.Generator, k: int) -> tuple[float, float]:
    """Return (reported score when we pick on the test set,
    reported score when we pick on the dev set and then take the test once)."""
    p_test = rng.beta(2, 2, size=N_TEST)  # difficulty of the test questions
    p_dev = rng.beta(2, 2, size=N_TEST)  # dev set: different questions from the same distribution
    test_scores = (rng.random((k, N_TEST)) < p_test).mean(axis=1)  # test scores of the K models
    dev_scores = (rng.random((k, N_TEST)) < p_dev).mean(axis=1)
    picked_on_test = test_scores.max()  # pick by the test score: the reported value is the maximum
    picked_on_dev = test_scores[dev_scores.argmax()]  # pick on the dev set; take the test only once
    return float(picked_on_test), float(picked_on_dev)


def pick_table(ks=(1, 3, 10, 30)) -> list[tuple[int, float, float]]:  # noqa: ANN001
    """[(K, mean reported value with the pick on the test set,
    mean reported value with the pick on the dev set), ...]. The video also uses this function."""
    rng = np.random.default_rng(0)
    rows = []
    for k in ks:
        res = np.array([one_trial(rng, k) for _ in range(N_TRIALS)])
        on_test, on_dev = res.mean(axis=0)
        rows.append((k, float(on_test), float(on_dev)))
    return rows


def main() -> None:
    print(f"True skill: all models are {TRUE_ACC:.3f}; test set: {N_TEST} questions; "
          f"each row is the mean of {N_TRIALS} trials\n")
    print("| Candidates K | Pick on test set (reported) | Inflation | Pick on dev set, test once |")
    print("|---:|---:|---:|---:|")
    for k, on_test, on_dev in pick_table():
        print(f"| {k} | {on_test:.3f} | {on_test - TRUE_ACC:+.3f} | {on_dev:.3f} |")
    print(
        "\nConclusion: with more candidates, 'the best score picked on the test set' is more inflated. "
        "Pick on the dev set and take the test only once: then the reported value goes back to the true skill. "
        "This is the rule in GOAL.md Section 11: 'Do not use the pre-registered test benchmarks "
        "to tune hyperparameters or pick checkpoints'."
    )


if __name__ == "__main__":
    main()
