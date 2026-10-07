"""Chapter 5 · Minimal code 4: a language model = classification over the vocabulary.

"Look at the previous character, guess the next character" is a classification problem:
the number of classes = the vocabulary size.
Here we use the simplest model: a V×V table of logits (Chapter 7 explains this bigram model).
We train it with the same softmax + cross-entropy + gradient p − onehot as in 02.
Then we convert the loss to bits and to perplexity.
Uses only NumPy. It runs in 1 to 2 seconds on a CPU.
Run: uv run python chapters/05-classification-probability/code/04_next_token.py
"""

import importlib.util
from pathlib import Path

import numpy as np

_spec = importlib.util.spec_from_file_location("ce_mod", Path(__file__).with_name("02_cross_entropy.py"))
ce = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ce)

# The corpus is data, so it stays in Chinese. One Chinese character = one token = one class.
# Meaning: "We start from a straight line. A straight line cannot fit a curve, so we have neural
# networks. A neural network outputs a set of scores; we change the scores into probabilities, and
# cross-entropy measures how wrong the probabilities are. A language model does the same: look at
# the previous characters, guess the next character. To guess the next character is to do one
# classification over all characters."
TEXT = (
    "我们从一条直线出发。直线拟合不了曲线，所以有了神经网络。"
    "神经网络输出一组分数，把分数变成概率，交叉熵衡量概率给错了多少。"
    "语言模型做的事情也一样：看前面的字，猜下一个字。"
    "猜下一个字，就是在所有的字里做一次分类。"
)


def build_dataset(text: str):
    vocab = sorted(set(text))
    stoi = {c: i for i, c in enumerate(vocab)}
    ids = np.array([stoi[c] for c in text])
    return vocab, stoi, ids[:-1], ids[1:]  # input: the current character; label: the next character


def train_bigram(x: np.ndarray, y: np.ndarray, V: int, lr: float = 50.0, steps: int = 500):
    """logits = W[x]: each "current character" has one row of V scores.

    This is the same as onehot(x) @ W, a linear classifier.
    """
    W = np.zeros((V, V))  # all zeros → the initial prediction is uniform, loss = ln V
    log = []
    for step in range(steps + 1):
        logits = W[x]
        if step in (0, 10, 50, 100, 500):
            log.append((step, ce.cross_entropy(logits, y)))
        g = ce.ce_grad(logits, y)       # (p − onehot) / N
        np.add.at(W, x, -lr * g)        # add −lr × gradient to the row of each current character
    return W, log


def count_mle_loss(x: np.ndarray, y: np.ndarray, V: int) -> float:
    """The closed-form maximum-likelihood solution: count the frequencies.

    p(next character | current character) = count / total count.
    The bigram that gradient descent trains moves toward this loss.
    """
    counts = np.zeros((V, V))
    np.add.at(counts, (x, y), 1)
    p = counts / counts.sum(axis=1, keepdims=True)
    return float(-np.log(p[x, y]).mean())


def to_bits_and_ppl(nats: float):
    """nats (natural log) → bits (log base 2). Perplexity = e^loss = 2^bits."""
    return nats / np.log(2), float(np.exp(nats))


if __name__ == "__main__":
    vocab, stoi, x, y = build_dataset(TEXT)
    V = len(vocab)
    print(f"Corpus: {len(TEXT)} characters. Vocabulary: V = {V} different characters. "
          f"Training samples: {len(x)} pairs (current character → next character)")
    print(f"Loss of a uniform guess = ln V = {np.log(V):.4f} nats\n")

    W, log = train_bigram(x, y, V)
    print("  step loss (nats) loss (bits) perplexity")
    for step, nats in log:
        bits, ppl = to_bits_and_ppl(nats)
        print(f"{step:>6}   {nats:9.4f}   {bits:9.4f}   {ppl:8.2f}")

    print(f"Comparison: loss from counting frequencies (the closed-form maximum-likelihood solution)"
          f" = {count_mle_loss(x, y, V):.4f} nats")

    print("\nWhat the model learned: probability of the next character, given the current character (top 3)")
    # Three characters from TEXT: "spirit" (first half of "nerve"), "next",
    # and "divide" (first half of "score" and of "classification").
    for c in ["神", "下", "分"]:
        p = ce.softmax(W[stoi[c]])
        top = np.argsort(-p)[:3]
        print(f'  "{c}" → ' + ", ".join(f"{vocab[i]} {p[i]:.2f}" for i in top))

    print("\nAt the scale of a real vocabulary:")
    for name, v in [("this example", V), ("GPT-2 BPE vocabulary", 50257)]:
        print(f"  {name}: V = {v}, loss of a uniform guess ln V = {np.log(v):.2f} nats = {np.log2(v):.2f} bits, "
              f"perplexity = {v}")
