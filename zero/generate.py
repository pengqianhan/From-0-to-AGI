"""文本生成：采样 + 带 KV cache 的自回归解码（对应第 10 章）。

- `sample_next(logits, temperature, top_p)`：从最后一个位置的 logits 里选下一个 token。
  temperature=0 表示贪心（取最大值）；top_p < 1 表示 nucleus 采样：按概率从大到小排，
  只在累计概率刚好达到 top_p 的那一小撮 token 里抽。
- `generate(model, prompt_ids, max_new_tokens, ...)`：先把整段提示词喂进去（prefill），
  之后每步只喂一个新 token（decode），K/V 从缓存里读。`use_cache=False` 时每一步都重算整段序列，
  速度慢但逻辑最简单——`tests/test_kv_cache.py` 用它来验证缓存版的输出完全一致。
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
    """logits: (B, V) → 下一个 token 的 id，形状 (B,)。"""
    if temperature <= 0:
        return logits.argmax(dim=-1)
    probs = torch.softmax(logits.float() / temperature, dim=-1)
    if top_p < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
        cum = torch.cumsum(sorted_probs, dim=-1)
        # 保留"加上自己之前的累计概率 < top_p"的 token：保证至少留下概率最大的那个
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
    """逐步产出新 token（形状 (B,)），适合命令行里边生成边打印。遇到 eos（batch=1 时）停止。"""
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
    """生成并返回**新 token**（不含提示词，也不含 eos）。

    prompt_ids 是 list[int] 或一维张量时返回 list[int]；是 (B, T) 张量时返回 list[list[int]]。
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
