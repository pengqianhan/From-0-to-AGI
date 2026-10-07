"""Text cleaning: normalization and simple line-by-line processing (Chapter 13).

Cleaning is the first step of the pipeline. Its goal: the same content becomes the same bytes.
Only then can the deduplication step work correctly. The rules:

- Apply Unicode NFC normalization. Change special white space (for example the full-width space)
  to a normal space.
- Change all line breaks to \\n. Remove control characters (keep \\n and \\t) and zero-width characters.
- Remove the white space at the end of each line. Change 3 or more blank lines in sequence to 2.
- Remove literal strings that look like special tokens of the tokenizer (for example "<|endoftext|>").
  See the notes in zero/tokenizer.py.

These rules are conservative: they change only the "format", and they do not remove "content".
quality.py removes documents because of their content.
"""

from __future__ import annotations

import re
import unicodedata

# Control characters: all of C0/C1 except \t and \n
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
# Zero-width characters and BOM
_ZERO_WIDTH_RE = re.compile(r"[​‌‍⁠﻿]")
# Characters that "look like a space" → normal space
_SPACE_RE = re.compile(r"[   -   　]")
_MANY_BLANK_LINES_RE = re.compile(r"\n{3,}")
# Literal special tokens of the form <|xxx|>
_SPECIAL_TOKEN_RE = re.compile(r"<\|[a-zA-Z0-9_]+\|>")


def normalize_text(text: str) -> str:
    """Normalize a text. The function is idempotent: a second call does not change the result."""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _CONTROL_RE.sub("", text)
    text = _ZERO_WIDTH_RE.sub("", text)
    text = _SPACE_RE.sub(" ", text)
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    text = _MANY_BLANK_LINES_RE.sub("\n\n", text)
    return text.strip()


def strip_special_tokens(text: str) -> str:
    """Remove literal special tokens such as <|endoftext|>, so that the tokenizer does not read them as real special tokens."""
    return _SPECIAL_TOKEN_RE.sub("", text)


def clean_document(text: str, min_chars: int = 1) -> str | None:
    """Clean one document completely. Return None if the result is too short (< min_chars)."""
    text = normalize_text(strip_special_tokens(text))
    if len(text) < min_chars:
        return None
    return text


def split_into_documents(text: str, doc_chars: int = 4000) -> list[str]:
    """Split a large text without document boundaries (for example a .txt file of tiny_corpus) at blank lines.

    Then join the paragraphs into "documents" of about doc_chars characters.
    """
    paragraphs = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
    docs: list[str] = []
    cur: list[str] = []
    cur_len = 0
    for p in paragraphs:
        if cur and cur_len + len(p) > doc_chars:
            docs.append("\n\n".join(cur))
            cur, cur_len = [], 0
        cur.append(p)
        cur_len += len(p) + 2
    if cur:
        docs.append("\n\n".join(cur))
    return docs
