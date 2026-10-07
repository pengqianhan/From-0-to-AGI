"""Chapter 7 · Minimal code 2: byte-level BPE by hand (about 100 lines)

BPE (Byte Pair Encoding) training has one rule: **merge the most frequent pair of adjacent symbols
in the corpus into a new symbol, again and again**.
  1. Pre-tokenization: a regex splits the text into chunks (runs of letters, single digits,
     runs of punctuation, whitespace). A merge never crosses a chunk boundary.
  2. Each chunk becomes a sequence of UTF-8 bytes. The initial vocabulary is the 256 bytes.
  3. Count all pairs of adjacent ids. Merge the most frequent pair into a new id (256, 257, ...).
     Do this again until the vocabulary has the necessary size.
To encode new text, replay the merges in the order of learning. To decode, join the bytes of
all ids and decode them as UTF-8.
Run: uv run python chapters/07-tokenization-language-model/code/02_bpe.py
"""

import re
import time
from collections import Counter, defaultdict
from pathlib import Path

CORPUS = Path(__file__).resolve().parents[3] / "assets" / "tiny_corpus"
FILES = {"en": "shakespeare.txt", "zh": "chinese_poetry.txt", "code": "code.txt"}

# Pre-tokenization regex: a simplified version of the GPT-2 / Qwen idea. Python `re` has no \p{L},
# so [^\W\d_] stands for "letter" (Chinese characters included). The alternatives are:
# English contraction | run of letters with an optional leading space | one digit
# | run of punctuation (underscore included) with an optional leading space | whitespace.
# Digits are split one by one, as in Qwen.
PATTERN = re.compile(r"'(?:s|t|re|ve|m|ll|d)| ?[^\W\d_]+|\d| ?(?:[^\s\w]|_)+|\s+(?!\S)|\s+")


def pretokenize(text: str) -> list[str]:
    return PATTERN.findall(text)


def merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """Replace each adjacent occurrence of pair in ids with new_id."""
    out, i = [], 0
    while i < len(ids):
        if i + 1 < len(ids) and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class BPE:
    def __init__(self) -> None:
        self.merges: dict[tuple[int, int], int] = {}          # (a, b) -> new id, in the order of learning
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        self.history: list[tuple[int, bytes, int]] = []      # (new id, bytes of the new token, count at the merge)
        self.cache: dict[str, list[int]] = {}                # chunk → ids, so that encode does each chunk only once

    def train(self, text: str, vocab_size: int) -> "BPE":
        words = Counter(pretokenize(text))                   # keep one copy of each chunk, with its count
        seqs = [list(w.encode("utf-8")) for w in words]
        freqs = list(words.values())
        stats: Counter = Counter()                           # adjacent pair -> weighted count
        where = defaultdict(set)                             # adjacent pair -> indices of the chunks that contain it
        for i, s in enumerate(seqs):
            for p in zip(s, s[1:]):
                stats[p] += freqs[i]
                where[p].add(i)
        for new_id in range(256, vocab_size):
            if not stats:
                break
            pair = max(stats, key=stats.get)                 # the most frequent adjacent pair
            count = stats[pair]
            if count < 2:
                break
            self.merges[pair] = new_id
            self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]
            self.history.append((new_id, self.vocab[new_id], count))
            touched = set()
            for i in where.pop(pair):                        # update only the chunks that the merge changes
                s, f = seqs[i], freqs[i]
                for p in zip(s, s[1:]):
                    stats[p] -= f
                    touched.add(p)
                s = seqs[i] = merge(s, pair, new_id)
                for p in zip(s, s[1:]):
                    stats[p] += f
                    where[p].add(i)
            for p in touched:                                # remove the pairs whose count fell to 0
                if stats[p] <= 0:
                    del stats[p]
        return self

    def _encode_chunk(self, chunk: str) -> list[int]:
        ids = list(chunk.encode("utf-8"))
        while len(ids) >= 2:
            # Of all adjacent pairs, do the merge that training learned first (the same order as in training).
            pair = min(zip(ids, ids[1:]), key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            ids = merge(ids, pair, self.merges[pair])
        return ids

    def encode(self, text: str) -> list[int]:
        out: list[int] = []
        for chunk in pretokenize(text):
            if chunk not in self.cache:
                self.cache[chunk] = self._encode_chunk(chunk)
            out.extend(self.cache[chunk])
        return out

    def decode(self, ids: list[int]) -> str:
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")


def show(b: bytes) -> str:
    """Show the bytes of a token: as text if they decode as UTF-8 (a space shows as ␣), else as hex."""
    try:
        return b.decode("utf-8").replace(" ", "␣").replace("\n", "↵")
    except UnicodeDecodeError:
        return "".join(f"\\x{c:02x}" for c in b)


def load_splits(n_train: int = 60_000, n_val: int = 20_000) -> tuple[dict, dict]:
    """Train on the first n_train characters of each corpus. Validate on the last n_val characters (no overlap)."""
    train, val = {}, {}
    for key, f in FILES.items():
        t = (CORPUS / f).read_text("utf-8")
        train[key], val[key] = t[:n_train], t[-n_val:]
    return train, val


if __name__ == "__main__":
    train, val = load_splits()
    text = "\n".join(train.values())
    print(f"Training text: 60,000 characters each of English, Chinese, and code; {len(text.encode()):,} bytes in total")
    print("Pre-tokenization example:", pretokenize("To be, or not to be? 学而时习之。x = 2026\n"))

    t0 = time.time()
    bpe = BPE().train(text, vocab_size=1024)
    print(f"Trained {len(bpe.merges)} merges (vocabulary 256 → {len(bpe.vocab)}) in {time.time() - t0:.1f} s\n")

    print("The first 20 merges:")
    for new_id, b, count in bpe.history[:20]:
        print(f"  {new_id}: {show(b):<10} occurs {count:,} times")
    print("\nSome longer tokens that training learned later:")
    longest = sorted(bpe.history, key=lambda h: -len(h[1]))[:10]
    print("  " + "  ".join(show(b) for _, b, _ in longest))

    print("\nEncoding examples:")
    for s in ["To be, or not to be", "学而时习之，不亦说乎", "def forward(self, x):"]:
        ids = bpe.encode(s)
        assert bpe.decode(ids) == s
        print(f"  {s!r}: {len(s.encode())} bytes → {len(ids)} tokens:",
              " | ".join(show(bpe.vocab[i]) for i in ids))

    print("\nCompression on the validation set (the last 20,000 characters of each corpus, not seen in training):")
    print("  corpus    bytes    tokens   bytes/token")
    for k, v in val.items():
        ids = bpe.encode(v)
        assert bpe.decode(ids) == v                          # encode, then decode, must give back the original text
        nb = len(v.encode())
        print(f"  {k:<6} {nb:>8,} {len(ids):>9,}   {nb / len(ids):8.2f}")
