"""KV cache：推理时把算过的 K/V 存起来，下一个 token 不用重算（对应第 10、21 章）。

自回归生成时，第 t 步的注意力要用到前面所有位置的 K 和 V。不缓存的话每一步都要把整段
序列重新过一遍模型，代价是 O(T^2)；缓存以后每一步只算新 token 的 q/k/v，代价 O(T)。

这里的实现是"预分配"的：一开始就按 (层数, batch, kv 头数, 最大长度, head_dim) 分配好
显存，之后只往里写，不做拼接（`torch.cat` 每步都会重新分配内存）。注意缓存的是 **K/V 头**
的数量（GQA 下比查询头少），这正是 GQA 能省 KV cache 的原因，第 21 章会算这笔账。

写入的 K 是已经做过 QK-Norm 和 RoPE 旋转的，所以读出来可以直接用。
"""

from __future__ import annotations

import torch

from zero.config import ModelConfig


class KVCache:
    """每层一块预分配的 K/V 缓存。

    形状：k[layer], v[layer] 都是 (batch, n_kv_heads, max_seq_len, head_dim)。
    位置由调用方的 `start_pos` 决定（和 `Transformer.forward(tokens, kv_cache, start_pos)` 一致）。
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
        self.seq_len = 0  # 已经写入的最大位置 + 1（仅作记录，方便调试）

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
        """把新算出的 k/v（形状 (B, n_kv_heads, T, head_dim)）写到 [start_pos, start_pos+T)，
        返回这一层从 0 到 start_pos+T 的全部 K/V（视图，不拷贝）。"""
        bsz, _, t, _ = k.shape
        end = start_pos + t
        if end > self.max_seq_len:
            raise ValueError(f"KV cache 溢出：需要 {end} 个位置，只预分配了 {self.max_seq_len}")
        if bsz > self.batch_size:
            raise ValueError(f"batch={bsz} 超过了 KV cache 的 batch_size={self.batch_size}")
        self.k[layer_idx, :bsz, :, start_pos:end] = k.to(self.k.dtype)
        self.v[layer_idx, :bsz, :, start_pos:end] = v.to(self.v.dtype)
        self.seq_len = max(self.seq_len, end)
        return self.k[layer_idx, :bsz, :, :end], self.v[layer_idx, :bsz, :, :end]

    def reset(self) -> None:
        self.seq_len = 0

    def nbytes(self) -> int:
        """缓存占用的字节数（K 和 V 合计）。"""
        return self.k.numel() * self.k.element_size() * 2
