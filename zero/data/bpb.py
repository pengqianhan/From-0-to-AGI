"""Bits-per-byte (bpb) evaluation: a validation metric that does not depend on the tokenizer (Chapters 7 and 13).

You cannot compare the loss per token across tokenizers. A tokenizer with a large vocabulary covers
more bytes with one token, so each of its tokens is more difficult to predict.
bpb divides the total loss by the number of **bytes** of the original text:

    bpb = Σ_t (−ln p(y_t | x_≤t)) / (ln 2 × Σ_t bytes(y_t))

- Numerator: the sum of the negative log-likelihoods (in nats) of all predicted target tokens.
- Denominator: the number of UTF-8 bytes that these target tokens cover, multiplied by ln 2
  to change nats to bits.
- Special tokens (`<|endoftext|>`, `<|im_start|>`, and others) count as 0 bytes, and they are
  **not in the numerator**. They do not correspond to any original text.
- Positions with the label ignore_index (default -100, or any negative number) are also not counted.

The method follows `evaluate_bpb` in nanochat (nanochat/loss_eval.py). First,
`token_byte_lengths(tokenizer)` makes a lookup table "token id → number of bytes"
(length = vocabulary size). During the evaluation, `token_bytes[y]` is one table lookup.

Usage (call it every eval_every steps in the training loop):

    token_bytes = token_byte_lengths(tokenizer).to(device)
    bpb = evaluate_bpb(model, val_loader, token_bytes, steps=eval_batches)

With multiple GPUs, all_reduce sums the numerator and the denominator separately, and then the code
divides them. It does not average the bpb values of the GPUs.
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
    """Reverse map of the GPT-2 byte-level encoding: visible character → original byte (same as ByteLevel in `tokenizers`)."""
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
    """A token string from a byte-level BPE vocabulary (for example 'Ġthe') → the original bytes that it represents (b' the')."""
    dec = _byte_decoder()
    return bytes(dec[c] for c in token)


def token_byte_lengths(tokenizer: Any, device: torch.device | str = "cpu") -> torch.Tensor:
    """The number of UTF-8 bytes that each token id covers. Shape (vocab_size,), int64. Special tokens get 0.

    `tokenizer` can be a `zero.tokenizer.Tokenizer`, or any object with `vocab_size`, `id_to_token`,
    and `special_tokens` whose token strings use the byte-level form. It can also be a
    `Sequence[bytes | None]`: element i holds the bytes of token i, and None marks a special token.

    Note: do not use `decode([i])` to measure the length. One token can be half of a Chinese
    character. Then decode gives the replacement character '�' (3 bytes), and the length is wrong.
    This function maps the byte-level characters directly back to the original bytes.
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
    """Raw totals of an evaluation. You can merge the totals of several calls, or report each corpus separately."""

    nats: float  # sum of the negative log-likelihoods of the target tokens (natural log)
    bytes: int  # number of bytes that these target tokens cover
    tokens: int  # number of counted target tokens (no special tokens, no ignore positions)

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
    """Accept two input types: a loader with next_batch() (PackedDataLoader / MixtureLoader), or an iterable of (x, y)."""
    if hasattr(loader_or_batches, "next_batch"):
        if steps is None:
            raise ValueError("With a loader, you must give steps (the loader never stops)")
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
    """Add up the nats, bytes, and tokens. With multiple GPUs, all_reduce sums them.

    model(x) returns logits (B, T, V), as zero.model.Transformer does. If it returns a tuple,
    the first element is used.
    autocast: an optional context-manager factory (for example Trainer.autocast). The logits are
    always changed to float32 before the cross-entropy.
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
            y_safe = torch.where(valid, y, torch.zeros_like(y))  # a negative label cannot index the table or go into the cross-entropy
            nats = F.cross_entropy(
                logits.float().reshape(-1, logits.size(-1)), y_safe, reduction="none"
            )
            nb = torch.where(valid, token_bytes[y_safe], torch.zeros_like(y_safe))
            counted = nb > 0  # special tokens (0 bytes) and ignore positions are not in the numerator
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
    """Bits-per-byte on the validation set (lower is better). See `bpb_stats` for the parameters."""
    return bpb_stats(model, loader_or_batches, token_bytes, steps, ignore_index, autocast).bpb
