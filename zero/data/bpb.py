"""bits-per-byte（bpb）评估：与分词器无关的验证指标（对应第 7、13 章）。

每 token 的 loss 不能跨分词器比较：词表大的分词器一个 token 覆盖更多字节，每个 token 当然更难猜。
bpb 把总损失摊到原始文本的**字节**上：

    bpb = Σ_t (−ln p(y_t | x_≤t)) / (ln 2 × Σ_t bytes(y_t))

- 分子：所有被预测的目标 token 的负对数似然（nats）之和；
- 分母：这些目标 token 在 UTF-8 下一共覆盖多少字节，再乘 ln 2 把 nats 换成 bits；
- 特殊 token（`<|endoftext|>`、`<|im_start|>` 等）记 0 字节，并且**不计入分子**——它们不对应任何原文；
- 标签为 ignore_index（默认 -100，或任意负数）的位置也不计入。

写法参照 nanochat 的 `evaluate_bpb`（nanochat/loss_eval.py）：先用 `token_byte_lengths(tokenizer)`
预先算好"每个 token id → 字节数"的查找表（长度 = 词表大小），评估时 `token_bytes[y]` 一次查表。

用法（训练循环里每隔 eval_every 步调用一次）：

    token_bytes = token_byte_lengths(tokenizer).to(device)
    bpb = evaluate_bpb(model, val_loader, token_bytes, steps=eval_batches)

多卡时分子、分母分别 all_reduce 求和再相除（不是对各卡的 bpb 取平均）。
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F


@lru_cache(maxsize=1)
def _byte_decoder() -> dict[str, int]:
    """GPT-2 byte-level 的反向映射：可见字符 → 原始字节（与 `tokenizers` 的 ByteLevel 一致）。"""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1))
    bs += list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


def token_bytes_of(token: str) -> bytes:
    """byte-level BPE 词表里的一个 token 字符串（如 'Ġthe'）→ 它代表的原始字节（b' the'）。"""
    dec = _byte_decoder()
    return bytes(dec[c] for c in token)


def token_byte_lengths(tokenizer: Any, device: torch.device | str = "cpu") -> torch.Tensor:
    """每个 token id 覆盖的 UTF-8 字节数，形状 (vocab_size,)，int64；特殊 token 为 0。

    tokenizer 可以是 `zero.tokenizer.Tokenizer`（或任何有 `vocab_size`、`id_to_token`、
    `special_tokens` 的对象，token 字符串是 byte-level 表示），也可以直接传一个
    `Sequence[bytes | None]`（第 i 个元素是 token i 的字节；None 表示特殊 token）。

    注意：不能用 `decode([i])` 来量长度——单个 token 可能只是半个汉字，decode 会得到替换字符 '�'（3 字节），
    长度就错了。这里直接把 byte-level 字符映射回原始字节。
    """
    if isinstance(tokenizer, Sequence) and not isinstance(tokenizer, str | bytes):
        lengths = [0 if b is None else len(b) for b in tokenizer]
        return torch.tensor(lengths, dtype=torch.int64, device=device)
    specials = set(getattr(tokenizer, "special_tokens", {}).values())
    lengths = []
    for i in range(tokenizer.vocab_size):
        tok = tokenizer.id_to_token(i)
        if tok is None or i in specials:
            lengths.append(0)
        else:
            lengths.append(len(token_bytes_of(tok)))
    return torch.tensor(lengths, dtype=torch.int64, device=device)


@dataclass
class BpbStats:
    """评估的原始累计量：方便多次调用后合并，或者分语料分别报告。"""

    nats: float  # 目标 token 的负对数似然之和（自然对数）
    bytes: int  # 这些目标 token 覆盖的字节数
    tokens: int  # 计入的目标 token 数（不含特殊 token 与 ignore 位置）

    @property
    def bpb(self) -> float:
        return self.nats / (math.log(2) * self.bytes) if self.bytes > 0 else float("inf")

    @property
    def nats_per_token(self) -> float:
        return self.nats / self.tokens if self.tokens > 0 else float("inf")

    @property
    def bytes_per_token(self) -> float:
        return self.bytes / self.tokens if self.tokens > 0 else 0.0


def _batches(
    loader_or_batches: Any, steps: int | None
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """统一两种输入：有 next_batch() 的加载器（PackedDataLoader / MixtureLoader），或 (x, y) 的可迭代对象。"""
    if hasattr(loader_or_batches, "next_batch"):
        if steps is None:
            raise ValueError("传加载器时必须给 steps（加载器是无限循环的）")
        for _ in range(steps):
            yield loader_or_batches.next_batch()
        return
    it: Iterable = loader_or_batches
    for k, batch in enumerate(it):
        if steps is not None and k >= steps:
            return
        yield batch


def _logits(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    out = model(x)
    return out[0] if isinstance(out, tuple) else out


@torch.no_grad()
def bpb_stats(
    model: torch.nn.Module,
    loader_or_batches: Any,
    token_bytes: torch.Tensor,
    steps: int | None = None,
    ignore_index: int = -100,
    autocast: Any = None,
) -> BpbStats:
    """累计 nats、字节数、token 数（多卡时已 all_reduce 求和）。

    model(x) 返回 logits (B, T, V)（zero.model.Transformer 的约定；返回 tuple 时取第一个）。
    autocast：可选的上下文管理器工厂（如 Trainer.autocast），logits 一律转成 float32 再算交叉熵。
    """
    was_training = model.training
    model.eval()
    device = token_bytes.device
    total_nats = torch.zeros((), dtype=torch.float64, device=device)
    total_bytes = torch.zeros((), dtype=torch.int64, device=device)
    total_tokens = torch.zeros((), dtype=torch.int64, device=device)
    try:
        for x, y in _batches(loader_or_batches, steps):
            x, y = x.to(device), y.to(device)
            if autocast is not None:
                with autocast():
                    logits = _logits(model, x)
            else:
                logits = _logits(model, x)
            y = y.reshape(-1)
            valid = (y >= 0) & (y != ignore_index)
            y_safe = torch.where(valid, y, torch.zeros_like(y))  # 负数标签不能拿去查表 / 算交叉熵
            nats = F.cross_entropy(
                logits.float().reshape(-1, logits.size(-1)), y_safe, reduction="none"
            )
            nb = torch.where(valid, token_bytes[y_safe], torch.zeros_like(y_safe))
            counted = nb > 0  # 特殊 token（0 字节）和 ignore 位置都不计入分子
            total_nats += (nats.double() * counted).sum()
            total_bytes += nb.sum()
            total_tokens += counted.sum()
    finally:
        model.train(was_training)
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        for t in (total_nats, total_bytes, total_tokens):
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return BpbStats(float(total_nats.item()), int(total_bytes.item()), int(total_tokens.item()))


def evaluate_bpb(
    model: torch.nn.Module,
    loader_or_batches: Any,
    token_bytes: torch.Tensor,
    steps: int | None = None,
    ignore_index: int = -100,
    autocast: Any = None,
) -> float:
    """验证集 bits-per-byte（越低越好）。参数见 `bpb_stats`。"""
    return bpb_stats(model, loader_or_batches, token_bytes, steps, ignore_index, autocast).bpb
