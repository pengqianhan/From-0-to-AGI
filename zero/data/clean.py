"""文本清洗：规范化与简单的逐行处理（对应第 13 章）。

清洗是流水线的第一步，目标是"同样的内容变成同样的字节"，这样后面的去重才有效：

- Unicode NFC 规范化；全角空格等特殊空白统一成普通空格；
- 换行统一为 \\n，去掉控制字符（保留 \\n 和 \\t）、零宽字符；
- 每行去掉行尾空白，连续 3 个以上空行压成 2 个；
- 去掉会和分词器特殊 token 撞车的字面字符串（如 "<|endoftext|>"），见 zero/tokenizer.py 的说明。

这些规则都很保守——只改"格式"，不删"内容"。按内容删文档是 quality.py 的事。
"""

from __future__ import annotations

import re
import unicodedata

# 控制字符：C0/C1 里除 \t \n 之外的全部
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
# 零宽字符与 BOM
_ZERO_WIDTH_RE = re.compile(r"[​‌‍⁠﻿]")
# 各种"看起来像空格"的字符 → 普通空格
_SPACE_RE = re.compile(r"[   -   　]")
_MANY_BLANK_LINES_RE = re.compile(r"\n{3,}")
# 形如 <|xxx|> 的特殊 token 字面量
_SPECIAL_TOKEN_RE = re.compile(r"<\|[a-zA-Z0-9_]+\|>")


def normalize_text(text: str) -> str:
    """规范化一段文本（幂等：再调用一次结果不变）。"""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _CONTROL_RE.sub("", text)
    text = _ZERO_WIDTH_RE.sub("", text)
    text = _SPACE_RE.sub(" ", text)
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    text = _MANY_BLANK_LINES_RE.sub("\n\n", text)
    return text.strip()


def strip_special_tokens(text: str) -> str:
    """去掉 <|endoftext|> 之类的特殊 token 字面量，避免被分词器当成真的特殊 token。"""
    return _SPECIAL_TOKEN_RE.sub("", text)


def clean_document(text: str, min_chars: int = 1) -> str | None:
    """完整清洗一篇文档；清洗后太短（< min_chars）返回 None。"""
    text = normalize_text(strip_special_tokens(text))
    if len(text) < min_chars:
        return None
    return text


def split_into_documents(text: str, doc_chars: int = 4000) -> list[str]:
    """把一个没有文档边界的大文本（比如 tiny_corpus 的 .txt）按空行切段，再拼成约 doc_chars 长的"文档"。"""
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
