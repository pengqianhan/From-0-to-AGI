"""Prompt format sensitivity: same model, same questions. Change only the "look" of the prompt.
Then the highest score can be several times the lowest score.

    uv run python chapters/11-evaluation/code/03_prompt_sensitivity.py

The script uses the toy world and the character-level longest-suffix model from 02 again.
It changes only the prompt format: 6 formats with exactly the same meaning.
All formats use log-likelihood scoring.
Real large models are not this extreme, but the direction is the same. Sclar et al. (ICLR 2024)
found that on LLaMA-2-13B, accuracy can differ by up to 76 percentage points between formats
with the same meaning. Thus the pre-registration must fix the template word for word.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _load(name: str):  # noqa: ANN202
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # dataclass needs to find the module in sys.modules
    spec.loader.exec_module(mod)
    return mod


toy = _load("02_loglik_vs_generate")
LETTERS = "ABCD"

# name → (prompt function, how to write the choices).
# "letter" means that the model outputs the letter of the choice, not the text of the choice.
FORMATS = {
    "Cloze": (lambda it: it.stem, "text"),
    "Cloze + trailing space": (lambda it: it.stem + " ", "text"),
    "QA (workbook format)": (lambda it: f"问：{it.question}\n答：", "text"),
    "QA (English labels)": (lambda it: f"Q: {it.question}\nA: ", "text"),
    "QA (colon → space)": (lambda it: f"问 {it.question}\n答 ", "text"),
    "Letter choice": (
        lambda it: f"问：{it.question}\n"
        + "".join(f"{LETTERS[i]}. {c}\n" for i, c in enumerate(it.choices))
        + "答：",
        "letter",
    ),
}


def score_format(lm, items, prompt_fn, mode: str) -> list[dict]:  # noqa: ANN001
    per = []
    for it in items:
        ctx = prompt_fn(it)
        options = list(LETTERS[: len(it.choices)]) if mode == "letter" else it.choices
        lps = [lm.logprob(ctx, o) for o in options]
        pred = max(range(len(lps)), key=lambda i: lps[i])
        per.append({"id": it.id, "correct": float(pred == it.answer), "leaked": it.leaked,
                    "pred": pred})
    return per


def run_formats() -> list[dict]:
    """Return the accuracy of each format on three groups: all, leaked, and clean questions.
    The video also uses this function."""
    world = toy.build_world()
    lm = toy.SuffixLM(world.corpus)
    rows = []
    for name, (fn, mode) in FORMATS.items():
        per = score_format(lm, world.items, fn, mode)
        leak, clean = toy.split_acc(per)
        preds = [p["pred"] for p in per]
        rows.append({
            "name": name,
            "acc": sum(p["correct"] for p in per) / len(per),
            "leaked": leak,
            "clean": clean,
            "most_common_pred_share": max(preds.count(k) for k in range(4)) / len(preds),
        })
    return rows


def main() -> None:
    rows = run_formats()
    print("Same model (the character-level longest-suffix model from 02), same 32 four-choice questions, "
          "same log-likelihood scoring. Only the prompt format changes:\n")
    print("| Prompt format | All 32 | Leaked 12 | Clean 20 | Share of most-picked position |")
    print("|---|---:|---:|---:|---:|")
    for r in rows:
        print(f"| {r['name']} | {r['acc']:.3f} | {r['leaked']:.3f} | {r['clean']:.3f} | "
              f"{r['most_common_pred_share']:.2f} |")
    accs = [r["acc"] for r in rows]
    print(f"\nHighest {max(accs):.3f}, lowest {min(accs):.3f}, difference {max(accs) - min(accs):.3f}"
          " (the expected value of a random guess is 0.25)")
    print("Last column: with letter choice, the model almost always picks the same letter. "
          "A small model cannot yet 'read the choices and give a letter'. "
          "Thus we usually evaluate a small base model with cloze-style log-likelihood.")


if __name__ == "__main__":
    main()
