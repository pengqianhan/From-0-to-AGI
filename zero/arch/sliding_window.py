"""Sliding window attention and local-global interleaving (Chapter 22).
Experiment module for Part 5. **The main-line model does not use it.**

Three parts:

- `sliding_window_mask(...)`: a boolean mask. A query at global position p can see only the keys
  at positions j with `p - window < j <= p` (window=None gives the normal causal mask).
  The convention is the same as in Hugging Face (the `sliding_window` field of
  Mistral / Gemma / gpt-oss): the window **includes the token itself**, and each token sees
  a maximum of window positions.
- `make_layer_types(n_layers, global_every)`: makes the list of layer types. The values are the same
  as in HF `config.layer_types` ("sliding_attention" / "full_attention"). `global_every=6` is the
  Gemma 3 ratio of 5 local : 1 global. `global_every=2` is the 1:1 interleaving of Gemma 2 / gpt-oss.
  `global_every=4` is the 3:1 ratio of OLMo 3.
- `SlidingWindowAttention`: a subclass of `zero.model.Attention`. The parameter names are identical,
  so you can copy the weights directly. Only the mask changes. The matching `SlidingWindowKVCache`
  keeps only the latest window positions for sliding window layers (a ring buffer).
  Thus the KV cache of these layers does not grow with the sequence length.

Usage:

    model = Transformer(cfg)
    layer_types = make_layer_types(cfg.n_layers, global_every=4)
    convert_to_sliding_window(model, layer_types, window=128)   # replace the attention of each layer in place
    cache = SlidingWindowKVCache.from_model(model, batch_size=1, max_seq_len=4096)
    out = generate_greedy(model, prompt, 100, cache=cache)

The CPU implementation uses PyTorch SDPA + a custom boolean mask. The equivalents for production
inference (all **not verified on GPU yet**): `flash_attn_func(..., causal=True,
window_size=(window - 1, 0))` of FlashAttention 2/3; the `sliding_window` mask_mod of PyTorch
FlexAttention; vLLM / transformers read `sliding_window` and `layer_types`, then automatically select
the matching kernel and per-layer cache. `tests/test_arch_sliding_window.py` does the parity checks.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F

from zero.config import ModelConfig
from zero.kv_cache import KVCache
from zero.model import Attention, Transformer, apply_rope

SLIDING = "sliding_attention"
FULL = "full_attention"


# ---------------------------------------------------------------------------
# Mask and layer types
# ---------------------------------------------------------------------------


def sliding_window_mask(
    q_pos: torch.Tensor, k_pos: torch.Tensor, window: int | None
) -> torch.Tensor:
    """q_pos: (Tq,) global positions of the queries; k_pos: (Tk,) global positions of the keys
    (any order; a negative value is an empty slot).

    Return a (Tq, Tk) boolean mask, True = visible. Condition: `0 <= k <= q` and
    (with a window) `q - k < window`.
    """
    q = q_pos[:, None]
    k = k_pos[None, :]
    mask = (k >= 0) & (k <= q)
    if window is not None:
        mask &= (q - k) < window
    return mask


def make_layer_types(
    n_layers: int, global_every: int | None = None, all_sliding: bool = False
) -> list[str]:
    """Return the type of each layer: sliding window or full attention.

    - `global_every=k`: layers k, 2k, 3k, ... (counted from 1) are full attention, all others are
      sliding window. This is the same convention as `(i + 1) % sliding_window_pattern == 0` in HF Gemma 3.
    - `all_sliding=True`: all layers are sliding window (as in Mistral 7B v0.1).
    - Neither argument: all layers are full attention.
    """
    if all_sliding:
        return [SLIDING] * n_layers
    if global_every is None:
        return [FULL] * n_layers
    if global_every < 1:
        raise ValueError(f"global_every must be >= 1, got {global_every}")
    return [FULL if (i + 1) % global_every == 0 else SLIDING for i in range(n_layers)]


# ---------------------------------------------------------------------------
# KV cache: sliding window layers use a ring buffer
# ---------------------------------------------------------------------------


class SlidingWindowKVCache:
    """A per-layer KV cache. Full attention layers preallocate max_seq_len positions.
    Sliding window layers allocate only window slots.

    Ring buffer: the K/V of position p go into slot `p % window`. They overwrite the old values
    from window steps before. K already has RoPE before the write (the position information is
    already "rotated" into the vector), and attention does not depend on the order of the keys.
    Thus the slot order is not important. The cache only records which position each slot holds now
    (`pos`, -1 for an empty slot). The mask uses this information.

    `update` returns one more value than `zero.kv_cache.KVCache`: `k_pos` (the global position of each key).
    """

    def __init__(
        self,
        layer_types: Sequence[str],
        window: int,
        batch_size: int,
        max_seq_len: int,
        n_kv_heads: int,
        head_dim: int,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.layer_types = list(layer_types)
        self.window = window
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.k: list[torch.Tensor] = []
        self.v: list[torch.Tensor] = []
        self.pos: list[torch.Tensor] = []
        for t in self.layer_types:
            slots = min(window, max_seq_len) if t == SLIDING else max_seq_len
            shape = (batch_size, n_kv_heads, slots, head_dim)
            self.k.append(torch.zeros(shape, device=device, dtype=dtype))
            self.v.append(torch.zeros(shape, device=device, dtype=dtype))
            self.pos.append(torch.full((slots,), -1, dtype=torch.long, device=device))

    @classmethod
    def from_model(
        cls,
        model: Transformer,
        batch_size: int,
        max_seq_len: int | None = None,
        dtype: torch.dtype | None = None,
    ) -> SlidingWindowKVCache:
        layer_types, window = get_layer_types(model)
        cfg = model.config
        assert cfg.head_dim is not None
        param = next(model.parameters())
        return cls(
            layer_types,
            window or (max_seq_len or cfg.max_seq_len),
            batch_size,
            max_seq_len or cfg.max_seq_len,
            cfg.n_kv_heads,
            cfg.head_dim,
            device=param.device,
            dtype=dtype or param.dtype,
        )

    def update(
        self, layer_idx: int, start_pos: int, k: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Write the new K/V for positions [start_pos, start_pos+T). Return (K, V, k_pos) for this attention call.

        Sliding window layers: return "old values in the cache + new values of this call".
        The code concatenates the new values before it writes them into the ring buffer. Thus a
        chunked prefill with more than window tokens in one chunk is also correct. Then the code
        writes back only the last window values.
        """
        bsz, _, t, _ = k.shape
        end = start_pos + t
        if end > self.max_seq_len:
            raise ValueError(f"KV cache overflow: needs {end} positions, max_seq_len={self.max_seq_len}")
        new_pos = torch.arange(start_pos, end, device=k.device)
        ck, cv, cpos = self.k[layer_idx], self.v[layer_idx], self.pos[layer_idx]
        k = k.to(ck.dtype)
        v = v.to(cv.dtype)
        if self.layer_types[layer_idx] == FULL:
            ck[:bsz, :, start_pos:end] = k
            cv[:bsz, :, start_pos:end] = v
            cpos[start_pos:end] = new_pos
            return ck[:bsz, :, :end], cv[:bsz, :, :end], cpos[:end]

        slots = ck.shape[2]
        valid = cpos >= 0
        k_all = torch.cat([ck[:bsz][:, :, valid], k], dim=2)
        v_all = torch.cat([cv[:bsz][:, :, valid], v], dim=2)
        pos_all = torch.cat([cpos[valid], new_pos])
        keep = slice(max(0, t - slots), t)  # Only the last `slots` new positions must stay.
        idx = new_pos[keep] % slots
        ck[:bsz, :, idx] = k[:, :, keep]
        cv[:bsz, :, idx] = v[:, :, keep]
        cpos[idx] = new_pos[keep]
        return k_all, v_all, pos_all

    def nbytes(self) -> int:
        """Total bytes of K and V (sliding window layers have only window slots)."""
        return sum(t.numel() * t.element_size() for t in self.k + self.v)


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------


class SlidingWindowAttention(Attention):
    """The same parameters as `zero.model.Attention` (wq/wk/wv/wo/q_norm/k_norm), plus a window.

    With window=None, this is a normal full attention layer (a "global layer" in a local-global model).
    """

    def __init__(self, config: ModelConfig, window: int | None) -> None:
        super().__init__(config)
        self.window = window

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: SlidingWindowKVCache | KVCache | None = None,
        layer_idx: int = 0,
        start_pos: int = 0,
    ) -> torch.Tensor:
        bsz, seqlen, _ = x.shape
        q = self.q_norm(self.wq(x).view(bsz, seqlen, self.n_heads, self.head_dim)).transpose(1, 2)
        k = self.k_norm(self.wk(x).view(bsz, seqlen, self.n_kv_heads, self.head_dim))
        k = k.transpose(1, 2)
        v = self.wv(x).view(bsz, seqlen, self.n_kv_heads, self.head_dim).transpose(1, 2)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        q_pos = torch.arange(start_pos, start_pos + seqlen, device=x.device)
        if isinstance(kv_cache, SlidingWindowKVCache):
            k, v, k_pos = kv_cache.update(layer_idx, start_pos, k, v)
        elif kv_cache is not None:  # Normal KVCache: the key positions are 0..kv_len-1.
            k, v = kv_cache.update(layer_idx, start_pos, k, v)
            k_pos = torch.arange(k.shape[2], device=x.device)
        else:
            k_pos = q_pos
        k, v = k.to(q.dtype), v.to(q.dtype)

        mask = sliding_window_mask(q_pos, k_pos, self.window)
        # A custom boolean mask uses the generic SDPA path. It runs on CUDA, but with a mask the Flash
        # kernel is not available: on RTX 3090 the training throughput was about half of full attention
        # with the same size (2026-10). To really skip the blocks outside the window on GPU, use the
        # window_size of FlashAttention or FlexAttention (not verified on GPU yet).
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, enable_gqa=self.n_kv_heads != self.n_heads
        )
        out = out.transpose(1, 2).reshape(bsz, seqlen, self.n_heads * self.head_dim)
        return self.wo(out)


def convert_to_sliding_window(
    model: Transformer, layer_types: Sequence[str], window: int
) -> Transformer:
    """Replace the attention of each layer of model with SlidingWindowAttention in place.

    Keep the existing weights. Return model.
    """
    if len(layer_types) != len(model.layers):
        raise ValueError(f"layer_types has {len(layer_types)} items, but the model has {len(model.layers)} layers")
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    for block, t in zip(model.layers, layer_types, strict=True):
        if t not in (SLIDING, FULL):
            raise ValueError(f"Unknown layer type {t!r}, use {SLIDING!r} / {FULL!r}")
        old = block.attn
        new = SlidingWindowAttention(model.config, window if t == SLIDING else None)
        new.load_state_dict(old.state_dict())
        new.to(device=next(old.parameters()).device, dtype=next(old.parameters()).dtype)
        block.attn = new
    return model


def get_layer_types(model: Transformer) -> tuple[list[str], int | None]:
    """Read the type of each layer and the window size (of the sliding window layers).

    A layer that is not converted counts as full attention.
    """
    types, window = [], None
    for block in model.layers:
        w = getattr(block.attn, "window", None)
        types.append(SLIDING if w is not None else FULL)
        window = w if w is not None else window
    return types, window


# ---------------------------------------------------------------------------
# Generation (greedy; for the parity check of "bounded cache" against "no cache")
# ---------------------------------------------------------------------------


@torch.no_grad()
def generate_greedy(
    model: Transformer,
    prompt_ids: Sequence[int],
    max_new_tokens: int,
    cache: SlidingWindowKVCache | None = None,
) -> list[int]:
    """Generate max_new_tokens new tokens greedily.

    With cache=None, run the full sequence through the model again at each step.
    """
    was_training = model.training
    model.eval()
    try:
        device = next(model.parameters()).device
        seq = torch.tensor([list(prompt_ids)], dtype=torch.long, device=device)
        out: list[int] = []
        pos, nxt_in = 0, seq
        for _ in range(max_new_tokens):
            if cache is not None:
                logits = model(nxt_in, kv_cache=cache, start_pos=pos)  # type: ignore[arg-type]
                pos += nxt_in.shape[1]
            else:
                logits = model(seq)
            nxt = logits[:, -1].argmax(-1, keepdim=True)
            out.append(int(nxt))
            seq = torch.cat([seq, nxt], dim=1)
            nxt_in = nxt
        return out
    finally:
        model.train(was_training)


def kv_cache_bytes(
    layer_types: Sequence[str],
    window: int,
    seq_len: int,
    n_kv_heads: int,
    head_dim: int,
    bytes_per_elem: int = 2,
    batch_size: int = 1,
) -> int:
    """Theoretical KV cache bytes: full attention layers store seq_len positions,
    sliding window layers store min(window, seq_len) positions.
    """
    per_pos = 2 * n_kv_heads * head_dim * bytes_per_elem * batch_size
    positions = sum(seq_len if t == FULL else min(window, seq_len) for t in layer_types)
    return positions * per_pos
