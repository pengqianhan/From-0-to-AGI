"""Text generation: sampling + autoregressive decoding with a KV cache (Chapter 10).

- `sample_next(logits, temperature, top_p)`: select the next token from the logits of the last
  position. temperature=0 means greedy (take the maximum). top_p < 1 means nucleus sampling: sort
  the tokens by probability, from high to low, and sample only from the small set of tokens whose
  cumulative probability just reaches top_p.
- `generate(model, prompt_ids, max_new_tokens, ...)`: first send the full prompt (prefill). After
  that, send only one new token at each step (decode), and read the K/V from the cache.
  With `use_cache=False`, each step computes the full sequence again. This is slow but has the
  simplest logic. `tests/test_kv_cache.py` uses it to verify that the cached version gives
  exactly the same output.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import torch

from zero.kv_cache import KVCache
from zero.model import Transformer


def sample_next(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_p: float = 1.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """logits: (B, V) → id of the next token, shape (B,)."""
    if temperature <= 0:
        return logits.argmax(dim=-1)
    probs = torch.softmax(logits.float() / temperature, dim=-1)
    if top_p < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
        cum = torch.cumsum(sorted_probs, dim=-1)
        # Keep the tokens whose "cumulative probability before this token" is < top_p.
        # This always keeps at least the token with the highest probability.
        keep = (cum - sorted_probs) < top_p
        sorted_probs = sorted_probs * keep
        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
        choice = torch.multinomial(sorted_probs, 1, generator=generator)
        return sorted_idx.gather(-1, choice).squeeze(-1)
    return torch.multinomial(probs, 1, generator=generator).squeeze(-1)


def _as_batch(
    prompt_ids: Sequence[int] | torch.Tensor, device: torch.device
) -> tuple[torch.Tensor, bool]:
    if isinstance(prompt_ids, torch.Tensor):
        t = prompt_ids.to(device=device, dtype=torch.long)
        if t.dim() == 1:
            return t[None, :], True
        return t, False
    return torch.tensor([list(prompt_ids)], dtype=torch.long, device=device), True


@torch.no_grad()
def generate_stream(
    model: Transformer,
    prompt_ids: Sequence[int] | torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_p: float = 1.0,
    use_cache: bool = True,
    eos_id: int | None = None,
    seed: int | None = None,
) -> Iterator[torch.Tensor]:
    """Yield the new tokens step by step (shape (B,)), so a command line can print while it generates.

    Stop at eos (when batch=1).
    """
    device = next(model.parameters()).device
    tokens, _ = _as_batch(prompt_ids, device)
    bsz, prompt_len = tokens.shape
    max_len = model.config.max_seq_len
    if prompt_len + max_new_tokens > max_len:
        max_new_tokens = max_len - prompt_len
    generator = None
    if seed is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)

    was_training = model.training
    model.eval()
    try:
        cache = None
        if use_cache:
            dtype = next(model.parameters()).dtype
            cache = KVCache.from_config(
                model.config,
                bsz,
                max_seq_len=prompt_len + max_new_tokens,
                device=device,
                dtype=dtype,
            )
        finished = torch.zeros(bsz, dtype=torch.bool, device=device)
        seq = tokens
        pos = 0
        next_input = tokens
        for _ in range(max_new_tokens):
            if use_cache:
                logits = model(next_input, kv_cache=cache, start_pos=pos)
                pos += next_input.shape[1]
            else:
                logits = model(seq)
            nxt = sample_next(logits[:, -1, :], temperature, top_p, generator)
            if eos_id is not None:
                nxt = torch.where(finished, torch.full_like(nxt, eos_id), nxt)
                finished |= nxt == eos_id
            yield nxt
            seq = torch.cat([seq, nxt[:, None]], dim=1)
            next_input = nxt[:, None]
            if eos_id is not None and bool(finished.all()):
                break
    finally:
        model.train(was_training)


def generate(
    model: Transformer,
    prompt_ids: Sequence[int] | torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_p: float = 1.0,
    use_cache: bool = True,
    eos_id: int | None = None,
    seed: int | None = None,
) -> list[int] | list[list[int]]:
    """Generate and return the **new tokens** (without the prompt and without eos).

    If prompt_ids is a list[int] or a 1D tensor, return list[int]. If it is a (B, T) tensor,
    return list[list[int]].
    """
    _, single = _as_batch(prompt_ids, torch.device("cpu"))
    steps = list(
        generate_stream(
            model, prompt_ids, max_new_tokens, temperature, top_p, use_cache, eos_id, seed
        )
    )
    if not steps:
        return [] if single else [[] for _ in range(len(prompt_ids))]
    out = torch.stack(steps, dim=1).tolist()
    results = []
    for row in out:
        if eos_id is not None and eos_id in row:
            row = row[: row.index(eos_id)]
        results.append(row)
    return results[0] if single else results
