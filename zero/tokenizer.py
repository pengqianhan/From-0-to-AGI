"""分词器：byte-level BPE（对应第 7、13 章）。

基于 Hugging Face 的 `tokenizers` 库（Rust 实现，训练 GB 级语料也只要几分钟）。设计与 GPT-4 / Qwen
系列一致：

1. **规范化**：Unicode NFC（把"é = e + ◌́"这类组合字符合成一个码位）；
2. **预切分**：先用正则把文本切成"词"（字母串、单个数字、标点串、空白），BPE 合并不跨越这些边界。
   数字按单个切（`\\p{N}`），这对算术有帮助（Qwen、Llama 3 都这么做）；
3. **字节级**：每个"词"先变成 UTF-8 字节，256 个字节都在初始词表里，所以任何文本都能编码，
   永远不会出现 [UNK]；
4. **BPE 合并**：反复把语料里最常见的相邻两个符号合成一个新符号，直到词表达到 vocab_size。

特殊 token（对话、工具调用、预留位）放在词表最前面，id 固定：

    0 <|endoftext|>  文档分隔 / 预训练的 EOS
    1 <|im_start|>   2 <|im_end|>          对话模板（ChatML，与 Qwen 一致）
    3 <tool_call>    4 </tool_call>        工具调用
    5 <tool_response> 6 </tool_response>   工具返回
    7 <think>        8 </think>            思考
    9–15 <|reserved_0|> … <|reserved_6|>  预留，以后要加新特殊 token 时不用改词表大小

注意：特殊 token 在 encode 时会从原文里直接匹配出来（`tokenizers` 的行为）。预训练网页里偶尔会出现
"<|endoftext|>" 这样的字面字符串，数据清洗时应当先去掉（见 zero/data/clean.py）。

`bytes_per_token(texts)` 统计压缩率：平均每个 token 覆盖多少个 UTF-8 字节，越大说明越省 token，
第 13 章用它比较不同词表大小在中文、英文、代码上的表现。
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

# 与 Qwen2/Qwen3 分词器相同的预切分正则（来自其 tokenizer.json）
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
    """对 `tokenizers.Tokenizer` 的薄封装：只暴露本课需要的接口。"""

    def __init__(self, hf_tokenizer: HFTokenizer) -> None:
        self._tok = hf_tokenizer
        self.special_tokens: dict[str, int] = {
            t.content: i for i, t in hf_tokenizer.get_added_tokens_decoder().items() if t.special
        }

    # ---- 编解码 ----
    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text, add_special_tokens=False).ids

    def encode_with_offsets(self, text: str) -> tuple[list[int], list[tuple[int, int]]]:
        """编码并返回每个 token 在原文里的字符区间 [start, end)（对话模板算 loss mask 用）。"""
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
            raise KeyError(f"不是特殊 token：{token}（已有：{list(self.special_tokens)}）")
        return self.special_tokens[token]

    @property
    def vocab_size(self) -> int:
        return self._tok.get_vocab_size(with_added_tokens=True)

    @property
    def eot_id(self) -> int:
        """<|endoftext|>：预训练文档之间的分隔符。"""
        return self.special_id(ENDOFTEXT)

    @property
    def eos_id(self) -> int:
        """Base 模型的生成结束符（与 Qwen3 Base 一致用 <|endoftext|>；对话模型用 <|im_end|>）。"""
        return self.eot_id

    @property
    def im_start_id(self) -> int:
        return self.special_id(IM_START)

    @property
    def im_end_id(self) -> int:
        return self.special_id(IM_END)

    # ---- 统计 ----
    def bytes_per_token(self, texts: Iterable[str]) -> float:
        """压缩率：总 UTF-8 字节数 / 总 token 数。"""
        texts = list(texts)
        n_bytes = sum(len(t.encode("utf-8")) for t in texts)
        n_tokens = sum(len(ids) for ids in self.encode_batch(texts))
        return n_bytes / max(n_tokens, 1)

    def hash(self) -> str:
        """分词器内容的哈希（写进分片元数据，防止拿 A 分词器切的数据去训 B 分词器的模型）。"""
        return hashlib.sha256(self._tok.to_str().encode("utf-8")).hexdigest()[:16]

    # ---- 存取 ----
    def save(self, path: str | os.PathLike) -> Path:
        """存成 tokenizer.json。path 是目录时写到 path/tokenizer.json。"""
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
        """写出 transformers 的 `AutoTokenizer` 能直接读的文件。

        chat_template：Jinja 对话模板（见 zero/post/chat.py 的 CHAT_TEMPLATE），写进 tokenizer_config.json，
        `tokenizer.apply_chat_template(...)`、vLLM、llama.cpp 转换脚本都从这里读。"""
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
    """训练 byte-level BPE。

    texts_or_files：可迭代对象，元素是 `pathlib.Path`（当作文本文件读）或 `str`（当作文本本身）。
    vocab_size：最终词表大小（含 256 个字节和特殊 token）。
    special_tokens：默认 `DEFAULT_SPECIAL_TOKENS`，放在词表最前面。
    """
    special = list(special_tokens) if special_tokens is not None else list(DEFAULT_SPECIAL_TOKENS)
    if ENDOFTEXT not in special:
        special = [ENDOFTEXT, *special]
    if vocab_size < 256 + len(special):
        raise ValueError(
            f"vocab_size={vocab_size} 太小，至少要 256 个字节 + {len(special)} 个特殊 token"
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
                    # 按块读，避免一次把大文件读进内存
                    while chunk := f.read(1 << 20):
                        yield chunk
            else:
                yield item

    tok.train_from_iterator(iterate(), trainer=trainer)
    return Tokenizer(tok)
