"""Chapter 7 · Minimal code 1: split by character or by byte?

A model knows only integers. There are two direct ways to change text into a sequence of integers:
  - character level: each Unicode character gets one id. The vocabulary is all the characters in the corpus.
  - byte level: encode the text with UTF-8 first. Each byte (0–255) gets one id. The vocabulary is always 256.
The script counts the vocabulary size and the sequence length of the two methods
on the three corpora in assets/tiny_corpus.
Run: uv run python chapters/07-tokenization-language-model/code/01_chars_and_bytes.py
"""

from pathlib import Path

CORPUS = Path(__file__).resolve().parents[3] / "assets" / "tiny_corpus"
FILES = {"English": "shakespeare.txt", "Chinese": "chinese_poetry.txt", "Code": "code.txt"}


def load(name: str) -> str:
    return (CORPUS / name).read_text("utf-8")


def char_ids(text: str):
    """Character level: the vocabulary is the characters that occur, sorted by code point."""
    vocab = sorted(set(text))
    stoi = {c: i for i, c in enumerate(vocab)}
    return [stoi[c] for c in text], vocab


def byte_ids(text: str) -> list[int]:
    """Byte level: each byte of the UTF-8 encoding is the id (0–255). No vocabulary is necessary."""
    return list(text.encode("utf-8"))


if __name__ == "__main__":
    for s in ["A", "é", "学", "🤖"]:
        b = s.encode("utf-8")
        print(f"'{s}'  code point U+{ord(s):04X}  UTF-8, {len(b)} byte(s): {list(b)}")

    s = "学而时习之"
    print(f"\n'{s}': {len(s)} ids at character level, {len(byte_ids(s))} ids at byte level: {byte_ids(s)}")
    assert bytes(byte_ids(s)).decode("utf-8") == s  # the byte sequence gives back the text with no loss

    print("\nCorpus     Characters       Bytes   Char V   Byte V Byte/char")
    for label, f in FILES.items():
        text = load(f)
        ids, vocab = char_ids(text)
        nb = len(byte_ids(text))
        print(f"{label:<10} {len(ids):>10,} {nb:>11,} {len(vocab):>8,} {256:>8}  {nb / len(ids):8.2f}")

    # The problem of the character level: what about characters that training did not see?
    zh = load("chinese_poetry.txt")
    cut = int(len(zh) * 0.9)
    seen = set(zh[:cut])
    unseen = [c for c in zh[cut:] if c not in seen]
    print(f"\nChinese, first 90% as the training set: {len(unseen)} characters ({len(set(unseen))} distinct)"
          f" in the last 10% do not occur in the training set, for example {''.join(sorted(set(unseen))[:10])}")
    print("The byte level does not have this problem: all text uses only the 256 possible bytes.")
