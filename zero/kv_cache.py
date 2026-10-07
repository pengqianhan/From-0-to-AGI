"""KV cache: in inference, keep the K/V that are already computed, so the next token does not
compute them again (Chapters 10 and 21).

In autoregressive generation, the attention at step t uses the K and V of all earlier positions.
Without a cache, each step sends the full sequence through the model again, at a cost of O(T^2).
With a cache, each step computes q/k/v only for the new token, at a cost of O(T).

This implementation "preallocates" the memory: at the start, it allocates
(layers, batch, KV heads, maximum length, head_dim). After that, it only writes into this memory
and does not concatenate (`torch.cat` allocates new memory at each step). The cache stores the
number of **K/V heads** (with GQA, fewer than the query heads). This is why GQA makes the KV cache
smaller. Chapter 21 calculates the numbers.

The stored K already has QK-Norm and the RoPE rotation, so the model can use it directly.
"""

from __future__ import annotations

import torch

from zero.config import ModelConfig


class KVCache:
    """One preallocated K/V cache for each layer.

    Shape: k[layer] and v[layer] are both (batch, n_kv_heads, max_seq_len, head_dim).
    The `start_pos` of the caller sets the position (the same as in
    `Transformer.forward(tokens, kv_cache, start_pos)`).
    """

    def __init__(
        self,
        n_layers: int,
        batch_size: int,
        max_seq_len: int,
        n_kv_heads: int,
        head_dim: int,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        shape = (n_layers, batch_size, n_kv_heads, max_seq_len, head_dim)
        self.k = torch.zeros(shape, device=device, dtype=dtype)
        self.v = torch.zeros(shape, device=device, dtype=dtype)
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.seq_len = 0  # largest written position + 1 (only a record, for debugging)

    @classmethod
    def from_config(
        cls,
        config: ModelConfig,
        batch_size: int,
        max_seq_len: int | None = None,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> KVCache:
        assert config.head_dim is not None
        return cls(
            n_layers=config.n_layers,
            batch_size=batch_size,
            max_seq_len=max_seq_len or config.max_seq_len,
            n_kv_heads=config.n_kv_heads,
            head_dim=config.head_dim,
            device=device,
            dtype=dtype,
        )

    def update(
        self, layer_idx: int, start_pos: int, k: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Write the new k/v (shape (B, n_kv_heads, T, head_dim)) to [start_pos, start_pos+T).

        Return all K/V of this layer from 0 to start_pos+T (a view, not a copy).
        """
        bsz, _, t, _ = k.shape
        end = start_pos + t
        if end > self.max_seq_len:
            raise ValueError(f"KV cache overflow: {end} positions are necessary, but only {self.max_seq_len} are preallocated")
        if bsz > self.batch_size:
            raise ValueError(f"batch={bsz} is larger than the KV cache batch_size={self.batch_size}")
        self.k[layer_idx, :bsz, :, start_pos:end] = k.to(self.k.dtype)
        self.v[layer_idx, :bsz, :, start_pos:end] = v.to(self.v.dtype)
        self.seq_len = max(self.seq_len, end)
        return self.k[layer_idx, :bsz, :, :end], self.v[layer_idx, :bsz, :, :end]

    def reset(self) -> None:
        self.seq_len = 0

    def nbytes(self) -> int:
        """Bytes that the cache uses (K and V together)."""
        return self.k.numel() * self.k.element_size() * 2
