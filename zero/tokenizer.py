"""Tokenizer: byte-level BPE (Chapters 7 and 13).

It uses the Hugging Face `tokenizers` library (written in Rust; training on GBs of text takes only
a few minutes). The design is the same as in GPT-4 and the Qwen series:

1. **Normalization**: Unicode NFC (it combines characters such as "é = e + ◌́" into one code point).
2. **Pre-tokenization**: a regex splits the text into "words" (runs of letters, single digits,
   runs of punctuation, whitespace). BPE merges do not cross these boundaries.
   Each digit is a separate piece (`\\p{N}`). This helps arithmetic (Qwen and Llama 3 do the same).
3. **Byte level**: each "word" becomes UTF-8 bytes first. All 256 bytes are in the initial
   vocabulary, so the tokenizer can encode any text. [UNK] never occurs.
4. **BPE merges**: again and again, merge the most frequent pair of adjacent symbols in the corpus
   into a new symbol, until the vocabulary reaches vocab_size.

The special tokens (chat, tool calls, reserved slots) are at the start of the vocabulary, with
fixed ids:

    0 <|endoftext|>  document separator / EOS of pretraining
    1 <|im_start|>   2 <|im_end|>          chat template (ChatML, the same as Qwen)
    3 <tool_call>    4 </tool_call>        tool call
    5 <tool_response> 6 </tool_response>   tool response
    7 <think>        8 </think>            thinking
    9–15 <|reserved_0|> … <|reserved_6|>  reserved for new special tokens (no change of vocabulary size)

Note: encode matches the special tokens directly in the input text (the behavior of `tokenizers`).
Pretraining web pages sometimes contain literal strings such as "<|endoftext|>". The data cleaning
must remove them first (see zero/data/clean.py).

`bytes_per_token(texts)` measures the compression: the mean number of UTF-8 bytes per token.
A larger value means fewer tokens. Chapter 13 uses it to compare vocabulary sizes on Chinese,
English, and code.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable, Sequence
from pathlib import Path

from tokenizers import Regex, decoders, models, normalizers, pre_tokenizers, trainers
from tokenizers import Tokenizer as HFTokenizer

ENDOFTEXT = "<|endoftext|>"
IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
TOOL_CALL_START = "<tool_call>"
TOOL_CALL_END = "</tool_call>"
TOOL_RESPONSE_START = "<tool_response>"
TOOL_RESPONSE_END = "</tool_response>"
THINK_START = "<think>"
THINK_END = "</think>"
NUM_RESERVED = 7

DEFAULT_SPECIAL_TOKENS: list[str] = [
    ENDOFTEXT,
    IM_START,
    IM_END,
    TOOL_CALL_START,
    TOOL_CALL_END,
    TOOL_RESPONSE_START,
    TOOL_RESPONSE_END,
    THINK_START,
    THINK_END,
] + [f"<|reserved_{i}|>" for i in range(NUM_RESERVED)]

# The same pre-tokenization regex as the Qwen2/Qwen3 tokenizers (from their tokenizer.json)
PRETOKENIZE_REGEX = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*"
    r"|\s*[\r\n]+|\s+(?!\S)|\s+"
)


def _build_empty_bpe() -> HFTokenizer:
    tok = HFTokenizer(models.BPE(byte_fallback=False))
    tok.normalizer = normalizers.NFC()
    tok.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Split(Regex(PRETOKENIZE_REGEX), behavior="isolated", invert=False),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
        ]
    )
    tok.decoder = decoders.ByteLevel()
    return tok


class Tokenizer:
    """A thin wrapper around `tokenizers.Tokenizer`: it shows only the interface that this course needs."""

    def __init__(self, hf_tokenizer: HFTokenizer) -> None:
        self._tok = hf_tokenizer
        self.special_tokens: dict[str, int] = {
            t.content: i for i, t in hf_tokenizer.get_added_tokens_decoder().items() if t.special
        }

    # ---- Encode and decode ----
    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text, add_special_tokens=False).ids

    def encode_with_offsets(self, text: str) -> tuple[list[int], list[tuple[int, int]]]:
        """Encode, and return the character span [start, end) of each token in the text (the chat template uses it for the loss mask)."""
        enc = self._tok.encode(text, add_special_tokens=False)
        return enc.ids, [tuple(o) for o in enc.offsets]  # type: ignore[misc]

    def encode_batch(self, texts: Sequence[str]) -> list[list[int]]:
        return [e.ids for e in self._tok.encode_batch(list(texts), add_special_tokens=False)]

    def decode(self, ids: Sequence[int], skip_special_tokens: bool = False) -> str:
        return self._tok.decode(list(ids), skip_special_tokens=skip_special_tokens)

    def id_to_token(self, idx: int) -> str | None:
        return self._tok.id_to_token(idx)

    def token_to_id(self, token: str) -> int | None:
        return self._tok.token_to_id(token)

    def special_id(self, token: str) -> int:
        if token not in self.special_tokens:
            raise KeyError(f"Not a special token: {token} (special tokens: {list(self.special_tokens)})")
        return self.special_tokens[token]

    @property
    def vocab_size(self) -> int:
        return self._tok.get_vocab_size(with_added_tokens=True)

    @property
    def eot_id(self) -> int:
        """<|endoftext|>: the separator between pretraining documents."""
        return self.special_id(ENDOFTEXT)

    @property
    def eos_id(self) -> int:
        """End-of-generation token of the base model (<|endoftext|>, as in Qwen3 Base; a chat model uses <|im_end|>)."""
        return self.eot_id

    @property
    def im_start_id(self) -> int:
        return self.special_id(IM_START)

    @property
    def im_end_id(self) -> int:
        return self.special_id(IM_END)

    # ---- Statistics ----
    def bytes_per_token(self, texts: Iterable[str]) -> float:
        """Compression: total UTF-8 bytes / total tokens."""
        texts = list(texts)
        n_bytes = sum(len(t.encode("utf-8")) for t in texts)
        n_tokens = sum(len(ids) for ids in self.encode_batch(texts))
        return n_bytes / max(n_tokens, 1)

    def hash(self) -> str:
        """Hash of the tokenizer content.

        The shard metadata stores it. This prevents training a model for tokenizer B on data that
        tokenizer A made.
        """
        return hashlib.sha256(self._tok.to_str().encode("utf-8")).hexdigest()[:16]

    # ---- Save and load ----
    def save(self, path: str | os.PathLike) -> Path:
        """Save as tokenizer.json. If path is a directory, write path/tokenizer.json."""
        p = Path(path)
        if p.suffix != ".json":
            p.mkdir(parents=True, exist_ok=True)
            p = p / "tokenizer.json"
        else:
            p.parent.mkdir(parents=True, exist_ok=True)
        self._tok.save(str(p))
        return p

    @classmethod
    def load(cls, path: str | os.PathLike) -> Tokenizer:
        p = Path(path)
        if p.is_dir():
            p = p / "tokenizer.json"
        return cls(HFTokenizer.from_file(str(p)))

    def save_hf(
        self,
        out_dir: str | os.PathLike,
        model_max_length: int = 32768,
        eos_token: str = ENDOFTEXT,
        chat_template: str | None = None,
    ) -> None:
        """Write the files that `AutoTokenizer` of transformers can read directly.

        chat_template: the Jinja chat template (see CHAT_TEMPLATE in zero/post/chat.py). It goes
        into tokenizer_config.json. `tokenizer.apply_chat_template(...)`, vLLM, and the llama.cpp
        conversion script read it from there.
        """
        out = Path(out_dir)
        self.save(out / "tokenizer.json")
        added = {
            str(i): {
                "content": t,
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True,
            }
            for t, i in sorted(self.special_tokens.items(), key=lambda kv: kv[1])
        }
        tok_cfg = {
            "tokenizer_class": "PreTrainedTokenizerFast",
            "added_tokens_decoder": added,
            "additional_special_tokens": [
                t for t in self.special_tokens if t not in (ENDOFTEXT, eos_token)
            ],
            "bos_token": None,
            "eos_token": eos_token,
            "pad_token": ENDOFTEXT,
            "unk_token": None,
            "model_max_length": model_max_length,
            "clean_up_tokenization_spaces": False,
            "split_special_tokens": False,
        }
        if chat_template is not None:
            tok_cfg["chat_template"] = chat_template
        with open(out / "tokenizer_config.json", "w") as f:
            json.dump(tok_cfg, f, indent=2, ensure_ascii=False)
        with open(out / "special_tokens_map.json", "w") as f:
            json.dump({"eos_token": eos_token, "pad_token": ENDOFTEXT}, f, indent=2)


def train_bpe(
    texts_or_files: Iterable[str | os.PathLike],
    vocab_size: int,
    special_tokens: Sequence[str] | None = None,
    min_frequency: int = 2,
) -> Tokenizer:
    """Train a byte-level BPE tokenizer.

    texts_or_files: an iterable of `pathlib.Path` (read as a text file) or `str` (the text itself).
    vocab_size: the final vocabulary size (includes the 256 bytes and the special tokens).
    special_tokens: default `DEFAULT_SPECIAL_TOKENS`; they go at the start of the vocabulary.
    """
    special = list(special_tokens) if special_tokens is not None else list(DEFAULT_SPECIAL_TOKENS)
    if ENDOFTEXT not in special:
        special = [ENDOFTEXT, *special]
    if vocab_size < 256 + len(special):
        raise ValueError(
            f"vocab_size={vocab_size} is too small; it must be at least 256 bytes + {len(special)} special tokens"
        )

    tok = _build_empty_bpe()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )

    def iterate() -> Iterable[str]:
        for item in texts_or_files:
            if isinstance(item, os.PathLike):
                with open(item, encoding="utf-8") as f:
                    # Read in chunks, so a large file does not go into memory at once
                    while chunk := f.read(1 << 20):
                        yield chunk
            else:
                yield item

    tok.train_from_iterator(iterate(), trainer=trainer)
    return Tokenizer(tok)
