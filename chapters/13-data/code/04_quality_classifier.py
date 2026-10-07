"""Model-based quality filter (toy version): an "expensive annotator" labels a small part of the
data. Then a cheap classifier, trained on these labels, gives a score to all documents.

FineWeb-Edu: Llama-3-70B-Instruct gave a score of 0–5 for "educational value" to about 460K
web pages. A small classifier (Snowflake-arctic-embed-m + a linear regression head) learned
from these labels. Then it gave a score to all 15T tokens, and only scores ≥ 3 stayed.
DCLM is similar. The positive examples are instruction data and highly voted ELI5 answers, and
the classifier is fastText.

Here, at a small scale:
- "Annotator": the true labels that we have (kind good/contaminated is good, all others are
  bad). **In a real pipeline, a large model gives these scores.** We have no large model, so we
  use the labels. This is the only "cheat" in this script.
- The "annotator" scores only 250 documents. The classifier predicts the other documents.
- Classifier: 8 hand-made features + logistic regression (gradient descent in NumPy). The most
  useful feature is "how familiar the adjacent word pairs are" (the same idea as the CCNet
  filter with KenLM perplexity).

    uv run python chapters/13-data/code/04_quality_classifier.py
"""

from __future__ import annotations

import importlib.util
import math
import random
import re
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


CJK = re.compile(r"[一-鿿]")
# Fraction of lines that end with punctuation, mean word length, stop-word fraction,
# distinct-word fraction, bigram familiarity, symbol fraction, log length, Chinese-character fraction
FEATURES = ["punct", "wordlen", "stop", "unique", "bigram", "symbol", "loglen", "cjk"]
GOOD_KINDS = ("good", "contaminated", "exact_dup", "near_dup")
STOP = set("的了是在和有也就不都而与之其以") | {"the", "be", "to", "of", "and", "that", "have", "with",
                                            "i", "you", "my", "a", "in", "is", "it", "not"}


def tokens(text: str) -> list[str]:
    """Split Chinese into characters and English at whitespace (the same "words" as in the
    heuristic rules)."""
    return re.findall(r"[一-鿿]|[^\s一-鿿]+", text.lower())


def bigrams(text: str) -> list[tuple[str, str]]:
    out = []
    for ln in text.split("\n"):
        t = tokens(ln)
        out += list(zip(t, t[1:]))
    return out


def features(text: str, ref: Counter, own: Counter | None = None) -> list[float]:
    """ref: the count of each adjacent word pair in the good documents. own: the count that this
    document adds itself. A document in the labeled set must subtract its own counts. If not,
    "familiarity" is too high on the training set, and the threshold that the classifier learns
    is wrong for new documents."""
    t = tokens(text)
    n = max(len(t), 1)
    lines = [ln for ln in text.split("\n") if ln.strip()]
    bg = bigrams(text)
    non_space = max(sum(not c.isspace() for c in text), 1)
    return [
        sum(ln.rstrip().endswith(tuple(".!?。！？，,;:；：")) for ln in lines) / max(len(lines), 1),
        sum(map(len, t)) / n,
        sum(w in STOP for w in t) / n,
        len(set(t)) / n,
        sum(ref[x] - (own[x] if own else 0) > 0 for x in bg) / max(len(bg), 1),  # was this adjacent word pair seen before?
        sum(not c.isalnum() and not c.isspace() for c in text) / non_space,
        math.log(len(text) + 1),
        len(CJK.findall(text)) / non_space,
    ]


def train_logreg(X: np.ndarray, y: np.ndarray, lr: float = 0.5, steps: int = 2000, l2: float = 1e-3):
    """Logistic regression: p = sigmoid(X·w + b). Minimize the cross-entropy (Chapter 5) with
    full-batch gradient descent."""
    w, b = np.zeros(X.shape[1]), 0.0
    for _ in range(steps):
        p = 1 / (1 + np.exp(-(X @ w + b)))
        w -= lr * (X.T @ (p - y) / len(y) + l2 * w)
        b -= lr * float(np.mean(p - y))
    return w, b


def classify(docs: list[dict], n_annotate: int = 250, seed: int = 0) -> dict:
    """The annotator labels n_annotate documents → train logistic regression → score all documents.

    Return the scores, the labels, the weights, and more.
    """
    # The one repost that stays after dedup has good content. Bad documents = pages that are not
    # language, such as word salad
    label = np.array([d["kind"] in GOOD_KINDS for d in docs], dtype=float)
    rng = random.Random(seed)
    idx = list(range(len(docs)))
    rng.shuffle(idx)
    ann, rest = idx[:n_annotate], idx[n_annotate:]
    ref = Counter(b for i in ann if label[i] == 1 for b in bigrams(docs[i]["text"]))
    ann_good = {i for i in ann if label[i] == 1}
    X = np.array([features(d["text"], ref, Counter(bigrams(d["text"])) if i in ann_good else None)
                  for i, d in enumerate(docs)])
    mu, sd = X[ann].mean(0), X[ann].std(0) + 1e-9
    Xs = (X - mu) / sd

    w, b = train_logreg(Xs[ann], label[ann])
    p = 1 / (1 + np.exp(-(Xs @ w + b)))
    return {"p": p, "label": label, "ann": ann, "rest": rest, "w": w}


def after_dedup() -> list[dict]:
    """01 → 02 → 03: the documents of the noisy crawl after the heuristic filter and dedup."""
    crawl_mod = _load("01_noisy_crawl")
    heur = _load("02_heuristic_filter")
    mh = _load("03_minhash")
    docs = crawl_mod.build_crawl()["docs"]
    docs, _, _ = heur.heuristic_filter(docs)
    docs = [docs[i] for i in mh.exact_dedup([d["text"] for d in docs])]
    keep, _ = mh.near_dedup([d["text"] for d in docs])
    return [docs[i] for i in keep]


def quality_filter(docs: list[dict], threshold: float = 0.5) -> list[dict]:
    r = classify(docs)
    return [d for d, p in zip(docs, r["p"]) if p >= threshold]


def main() -> None:
    crawl_mod = _load("01_noisy_crawl")
    docs = after_dedup()
    r = classify(docs)
    p, label, ann, rest, w = r["p"], r["label"], r["ann"], r["rest"], r["w"]
    print(f"After heuristics + dedup: {len(docs)} documents, {int((1 - label).sum())} of them bad: "
          f"{dict(Counter(d['kind'] for d in docs if d['kind'] not in GOOD_KINDS))}")
    print(f"\nThe annotator labels {len(ann)} documents ({int((1 - label[ann]).sum())} of them bad). "
          "Train logistic regression. Weights (standardized):")
    for name, wi in sorted(zip(FEATURES, w), key=lambda x: -abs(x[1])):
        print(f"  {name:<8}{wi:+.2f}")

    rr = np.array(rest)
    print(f"\nOn the {len(rest)} documents without a label, with different thresholds"
          " (remove if score < threshold; Caught = bad documents removed, Good lost = good documents removed):")
    print(f"{'Thresh':>6}{'Caught':>10}{'Miss':>6}{'Good lost':>10}{'Prec.':>8}{'Recall':>8}")
    for th in [0.3, 0.5, 0.7, 0.9]:
        bad = p[rr] < th
        tp = int((bad & (label[rr] == 0)).sum())
        fp = int((bad & (label[rr] == 1)).sum())
        fn = int((~bad & (label[rr] == 0)).sum())
        print(f"{th:>6}{tp:>10}{fn:>6}{fp:>10}{tp / max(tp + fp, 1):>8.2f}{tp / max(tp + fn, 1):>8.2f}")
    pred_bad = p < 0.5

    kept = [d for d, bad in zip(docs, pred_bad) if not bad]
    print(f"\nQuality filter: {len(docs)} → {len(kept)} documents")
    before, after = Counter(d["kind"] for d in docs), Counter(d["kind"] for d in kept)
    for k in crawl_mod.KINDS:
        if before[k]:
            print(f"  {k:<14}{before[k]:>5} → {after[k]:>5}")
    # The document with the lowest score
    worst = int(np.argmin(p))
    print(f"\nDocument with the lowest score (p={p[worst]:.3f}, {docs[worst]['kind']}): "
          + docs[worst]["text"][:90].replace("\n", "⏎"))


if __name__ == "__main__":
    main()
