"""Tokenizer: encode then decode gives the original text; special tokens; compression statistics (GOAL.md 9.1)."""

from __future__ import annotations

from pathlib import Path

import pytest

from zero.tokenizer import (
    DEFAULT_SPECIAL_TOKENS,
    ENDOFTEXT,
    IM_END,
    IM_START,
    TOOL_CALL_START,
    Tokenizer,
    train_bpe,
)


@pytest.fixture(scope="module")
def tok(tiny_texts: dict[str, str]) -> Tokenizer:
    return train_bpe(list(tiny_texts.values()), vocab_size=1024)


SAMPLES = [
    "To be, or not to be, that is the question.",
    "学而时习之，不亦说乎？有朋自远方来，不亦乐乎？",
    "def add(a: int, b: int) -> int:\n    return a + b  # 求和\n\n\tprint(add(1, 2))\n",
    "混合 mixed 文本 with 数字 12345.678 和 emoji 🤖🎉，还有 café 与全角空格　。",
    "",
    "   leading and trailing spaces   \n\n\n",
]


@pytest.mark.parametrize("text", SAMPLES)
def test_roundtrip(tok: Tokenizer, text: str) -> None:
    assert tok.decode(tok.encode(text)) == text


def test_special_tokens(tok: Tokenizer) -> None:
    assert tok.vocab_size == 1024
    # The special tokens are at the start of the vocabulary, with fixed ids
    for i, t in enumerate(DEFAULT_SPECIAL_TOKENS):
        assert tok.special_id(t) == i
    assert tok.eot_id == 0 and tok.im_start_id == tok.special_id(IM_START)
    text = f"{IM_START}user\n你好{IM_END}{TOOL_CALL_START}{ENDOFTEXT}"
    ids = tok.encode(text)
    for t in (IM_START, IM_END, TOOL_CALL_START, ENDOFTEXT):
        assert tok.special_id(t) in ids
    assert tok.decode(ids) == text
    assert tok.decode(ids, skip_special_tokens=True) == "user\n你好"


def test_digits_split_individually(tok: Tokenizer) -> None:
    ids = tok.encode("2026")
    assert len(ids) == 4


def test_bytes_per_token(tok: Tokenizer, tiny_texts: dict[str, str]) -> None:
    stats = {k: tok.bytes_per_token([v[:5000]]) for k, v in tiny_texts.items()}
    # The merges are learned: each token covers more than 1 byte on average.
    # A Chinese character has 3 bytes, so the compression must be higher
    for v in stats.values():
        assert v > 1.5
    assert stats["zh"] > 2.0


def test_save_load_and_hash(tok: Tokenizer, tmp_path: Path) -> None:
    p = tok.save(tmp_path)
    assert p.name == "tokenizer.json"
    tok2 = Tokenizer.load(tmp_path)
    s = SAMPLES[3]
    assert tok2.encode(s) == tok.encode(s)
    assert tok2.hash() == tok.hash()


def test_hf_autotokenizer_compat(tok: Tokenizer, tmp_path: Path) -> None:
    transformers = pytest.importorskip("transformers")
    tok.save_hf(tmp_path)
    hf = transformers.AutoTokenizer.from_pretrained(str(tmp_path))
    for s in SAMPLES[:4]:
        assert hf.encode(s, add_special_tokens=False) == tok.encode(s)
    assert hf.eos_token == ENDOFTEXT


def test_train_from_files(tmp_path: Path, tiny_texts: dict[str, str]) -> None:
    f = tmp_path / "a.txt"
    f.write_text(tiny_texts["zh"][:5000], "utf-8")
    t = train_bpe([f], vocab_size=400)
    assert t.decode(t.encode("学而时习之")) == "学而时习之"
    with pytest.raises(ValueError):
        train_bpe(["abc"], vocab_size=100)
