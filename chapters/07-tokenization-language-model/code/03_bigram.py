"""Chapter 7 · Minimal code 3: a bigram language model from counts, and bits-per-byte

A language model gives the probability distribution of the next token from the text before it.
A bigram model looks only at the previous token:
    p(x_t | x_<t) ≈ p(x_t | x_{t-1}) = (count[x_{t-1}, x_t] + α) / (Σ_j count[x_{t-1}, j] + α·V)
"Count, then divide by the total" is the maximum-likelihood solution (Chapter 5, 04_next_token.py, shows this).
We add α for smoothing. Without it, one pair in the validation set that training did not see
gets probability 0, and the loss becomes infinite.

Evaluation: cross-entropy (nats/token) → bits/token → perplexity → bits-per-byte.
bpb = total loss (nats) / (ln 2 × total bytes that these tokens cover). It does not depend on the
tokenizer, so we can compare models that use different tokenizers.
Run: uv run python chapters/07-tokenization-language-model/code/03_bigram.py
"""

import importlib.util
import math
from pathlib import Path

import numpy as np

_spec = importlib.util.spec_from_file_location("bpe_mod", Path(__file__).with_name("02_bpe.py"))
bpe_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bpe_mod)

ALPHA = 0.03  # add-α smoothing: chosen from 1, 0.3, 0.1, 0.03, 0.01 on [-40000:-20000] of each corpus (not used for training or validation)


class ByteTok:
    """Byte level: id = UTF-8 byte."""
    V = 256

    def encode(self, s: str) -> list[int]:
        return list(s.encode("utf-8"))

    def token_bytes(self, i: int) -> bytes:
        return bytes([i])


class CharTok:
    """Character level: the vocabulary is the characters in the training set + 1 <unk> (all unseen characters map to it)."""

    def __init__(self, train_text: str) -> None:
        self.chars = sorted(set(train_text))
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.V = len(self.chars) + 1

    def encode(self, s: str) -> list[int]:
        return [self.stoi.get(c, self.V - 1) for c in s]

    def token_bytes(self, i: int) -> bytes:
        return self.chars[i].encode("utf-8") if i < self.V - 1 else b"?"


class BPETok:
    """The BPE that we wrote by hand in the second script."""

    def __init__(self, bpe) -> None:
        self.bpe, self.V = bpe, len(bpe.vocab)

    def encode(self, s: str) -> list[int]:
        return self.bpe.encode(s)

    def token_bytes(self, i: int) -> bytes:
        return self.bpe.vocab[i]


def count_bigrams(ids: list[int], V: int) -> np.ndarray:
    a = np.asarray(ids)
    flat = a[:-1] * V + a[1:]                                 # encode (previous, next) as one integer
    return np.bincount(flat, minlength=V * V).reshape(V, V).astype(np.float64)  # counts[previous, next]


def bigram_prob(counts: np.ndarray, prev, nxt, alpha: float = ALPHA):
    """p(next = nxt | current = prev) = (count + α) / (row total + α·V). prev and nxt can be arrays."""
    V = counts.shape[0]
    return (counts[prev, nxt] + alpha) / (counts.sum(axis=1)[prev] + alpha * V)


def evaluate(counts: np.ndarray, ids: list[int], n_bytes: list[int]) -> dict:
    """Calculate the cross-entropy on a token sequence. n_bytes[i] is the number of bytes of token i in the original text."""
    x, y = np.array(ids[:-1]), np.array(ids[1:])
    nats = -np.log(bigram_prob(counts, x, y))                 # −ln p(correct next token) at each position
    total_nats, total_bytes = nats.sum(), sum(n_bytes[1:])   # count only the tokens that the model predicts
    ce = total_nats / len(y)                                  # nats / token
    return {
        "tokens": len(y),
        "nats": ce,
        "bits": ce / math.log(2),                             # bits / token
        "ppl": math.exp(ce),                                  # perplexity = e^nats = 2^bits
        "bpb": total_nats / (math.log(2) * total_bytes),     # bits / byte
    }


def run(tok, train_text: str, val_text: str, n_bytes_fn) -> dict:
    tr, va = tok.encode(train_text), tok.encode(val_text)
    counts = count_bigrams(tr, tok.V)
    return {**evaluate(counts, va, n_bytes_fn(va)), "counts": counts}


def sample(counts: np.ndarray, tok, start: int, n: int, seed: int = 0) -> str:
    """Sample from the counts with no smoothing: the next token can only be one that followed the current token in training."""
    rng = np.random.default_rng(seed)
    ids = [start]
    for _ in range(n):
        row = counts[ids[-1]]
        ids.append(int(rng.choice(len(row), p=row / row.sum())))  # sample from p(next | current)
    return b"".join(tok.token_bytes(i) for i in ids).decode("utf-8", errors="replace")


def splits(key: str) -> tuple[str, str]:
    """Bigram training uses the full corpus without its last 40,000 characters. Validation uses the last 20,000 characters (as in 02)."""
    t = (bpe_mod.CORPUS / bpe_mod.FILES[key]).read_text("utf-8")
    return t[:-40_000], t[-20_000:]


if __name__ == "__main__":
    train, _ = bpe_mod.load_splits()
    bpe = bpe_mod.BPE().train("\n".join(train.values()), vocab_size=1024)  # the tokenizer trains only on 60,000 characters of each corpus

    print(f"Bigram training: each corpus without its last 40,000 characters. Validation: the last 20,000 characters. Smoothing α = {ALPHA}\n")
    print(f"{'corpus':<5}{' tokenizer':<14}{'V':>6}{'val tokens':>12}{'nats/tok':>10}{'bits/tok':>10}"
          f"{'ppl':>9}{'bpb':>8}")
    samples = {}
    for k in ["en", "zh", "code"]:
        tr, va = splits(k)
        char = CharTok(tr)
        n_unk = sum(i == char.V - 1 for i in char.encode(va))
        for name, tok in [("byte", ByteTok()), ("char", char), ("BPE", BPETok(bpe))]:
            if isinstance(tok, CharTok):  # count <unk> with the true number of bytes of the original character
                nb = lambda ids, v=va: [len(c.encode()) for c in v]  # noqa: E731
            else:
                nb = lambda ids, t=tok: [len(t.token_bytes(i)) for i in ids]  # noqa: E731
            r = run(tok, tr, va, nb)
            if name != "byte":
                samples[(k, name)] = sample(r["counts"], tok, tok.encode("\n")[0], 50 if name == "char" else 30, seed=1)
            print(f"{k:<6}{name:<12}{tok.V:>8,}{r['tokens']:>11,}{r['nats']:>10.3f}{r['bits']:>10.3f}"
                  f"{r['ppl']:>9.1f}{r['bpb']:>8.3f}")
        print(f"      (character level: {n_unk} characters in the validation set did not occur in training; all map to <unk>)\n")

    print("For comparison: a uniform random guess of one byte = 8 bits/byte.\n")

    print("Samples from the bigram model (it looks only at the previous token; sampling uses the counts with no smoothing):")
    for (k, name), text in samples.items():
        if k != "code":
            print(f"  [{k} · {name}] {text.strip()!r}")
