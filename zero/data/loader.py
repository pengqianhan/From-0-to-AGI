"""打包数据加载器：从 uint32 分片里切出定长训练样本（对应第 14 章）。

**打包（packing）**：预训练不按文档对齐，而是把所有 token 看成一条长河，切成长度 seq_len+1 的片段，
x = 片段[:-1]，y = 片段[1:]。相邻片段重叠 1 个 token（步长 seq_len），所以每个 token 恰好当一次
"预测目标"。文档边界处的 <|endoftext|> 让模型知道上下文换了，一个片段里可能包含好几篇文档的结尾和开头。

**确定性与断点续训**：样本的顺序完全由 (seed, epoch) 决定：
- 每个 epoch 先打乱分片顺序，再打乱每个分片内部的片段顺序（两级打乱，不需要给全部片段建一个巨大的排列）；
- 于是"第 g 个全局样本是哪个分片的哪个片段"可以直接算出来（随机访问）。
加载器的全部状态就是一个整数 `consumed`（本 rank 已经取了多少个样本），`state_dict()` 存它，
`load_state_dict()` 恢复它，续训后的 batch 与不中断时逐字节相同。

**分布式**：第 g 个全局样本分给 rank = g % world_size。每步所有 rank 合起来正好取走连续的
batch_size * world_size 个全局样本，所以"2 卡各 B 条"与"1 卡 2B 条"看到的是同一批数据
（`tests/test_ddp_cpu.py` 用这一点来对拍 DDP）。
"""

from __future__ import annotations

import glob
import os
from collections import OrderedDict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from zero.data.shard import read_shard


def resolve_shards(paths: str | os.PathLike | Sequence[str | os.PathLike]) -> list[Path]:
    """glob 或路径列表 → 排好序的 .bin 文件列表。"""
    items = [paths] if isinstance(paths, str | os.PathLike) else list(paths)
    out: list[Path] = []
    for item in items:
        matches = sorted(glob.glob(str(item)))
        if not matches:
            raise FileNotFoundError(f"找不到分片：{item}")
        out.extend(Path(m) for m in matches if m.endswith(".bin"))
    if not out:
        raise FileNotFoundError(f"没有 .bin 分片：{paths}")
    return out


def to_torch_batch(
    arr: np.ndarray, device: torch.device | str
) -> tuple[torch.Tensor, torch.Tensor]:
    t = torch.from_numpy(arr.astype(np.int64))
    device = torch.device(device)
    if device.type == "cuda":
        t = t.pin_memory().to(device, non_blocking=True)
    else:
        t = t.to(device)
    return t[:, :-1], t[:, 1:]


class PackedDataLoader:
    def __init__(
        self,
        paths: str | os.PathLike | Sequence[str | os.PathLike],
        seq_len: int,
        batch_size: int,
        rank: int = 0,
        world_size: int = 1,
        seed: int = 0,
        shuffle: bool = True,
        device: torch.device | str = "cpu",
    ) -> None:
        self.paths = resolve_shards(paths)
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self.shuffle = shuffle
        self.device = device
        self.shards = [read_shard(p) for p in self.paths]
        # 每个分片能切出多少个 (seq_len+1) 片段（步长 seq_len）
        self.n_chunks = np.array(
            [max((len(s) - 1) // seq_len, 0) for s in self.shards], dtype=np.int64
        )
        self.total_chunks = int(self.n_chunks.sum())
        if self.total_chunks < world_size:
            raise ValueError(
                f"数据太少：{self.total_chunks} 个长度 {seq_len}+1 的片段，不够分给 {world_size} 个 rank"
            )
        self.consumed = 0
        self._epoch_cache: tuple[int, np.ndarray, np.ndarray] | None = None
        self._perm_cache: OrderedDict[tuple[int, int], np.ndarray] = OrderedDict()

    # ---- 索引计算 ----
    def _epoch_layout(self, epoch: int) -> tuple[np.ndarray, np.ndarray]:
        if self._epoch_cache is None or self._epoch_cache[0] != epoch:
            if self.shuffle:
                order = np.random.default_rng([self.seed, epoch]).permutation(len(self.shards))
            else:
                order = np.arange(len(self.shards))
            cum = np.cumsum(self.n_chunks[order])
            self._epoch_cache = (epoch, order, cum)
        return self._epoch_cache[1], self._epoch_cache[2]

    def _chunk_perm(self, epoch: int, shard: int) -> np.ndarray | None:
        if not self.shuffle:
            return None
        key = (epoch, shard)
        if key not in self._perm_cache:
            perm = np.random.default_rng([self.seed, epoch, shard, 1]).permutation(
                int(self.n_chunks[shard])
            )
            self._perm_cache[key] = perm
            if len(self._perm_cache) > 8:
                self._perm_cache.popitem(last=False)
        return self._perm_cache[key]

    def locate(self, global_index: int) -> tuple[int, int, int]:
        """第 global_index 个全局样本 → (epoch, 分片号, 片段号)。"""
        epoch, pos = divmod(global_index, self.total_chunks)
        order, cum = self._epoch_layout(epoch)
        j = int(np.searchsorted(cum, pos, side="right"))
        shard = int(order[j])
        within = pos - (int(cum[j - 1]) if j > 0 else 0)
        perm = self._chunk_perm(epoch, shard)
        chunk = int(perm[within]) if perm is not None else within
        return epoch, shard, chunk

    def sample(self, local_index: int) -> np.ndarray:
        """本 rank 的第 local_index 个样本（长度 seq_len+1 的 token 数组）。"""
        g = local_index * self.world_size + self.rank
        _, shard, chunk = self.locate(g)
        start = chunk * self.seq_len
        return np.asarray(self.shards[shard][start : start + self.seq_len + 1])

    # ---- 取数据 ----
    def next_samples(self, n: int) -> np.ndarray:
        out = np.stack([self.sample(self.consumed + i) for i in range(n)])
        self.consumed += n
        return out

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        """返回 (x, y)，形状都是 (batch_size, seq_len)，int64。"""
        return to_torch_batch(self.next_samples(self.batch_size), self.device)

    def __iter__(self) -> PackedDataLoader:
        return self

    def __next__(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.next_batch()

    @property
    def epoch(self) -> int:
        return (self.consumed * self.world_size + self.rank) // self.total_chunks

    @property
    def num_tokens(self) -> int:
        return int(sum(len(s) for s in self.shards))

    # ---- 断点续训 ----
    def state_dict(self) -> dict[str, Any]:
        return {
            "consumed": self.consumed,
            "seed": self.seed,
            "seq_len": self.seq_len,
            "rank": self.rank,
            "world_size": self.world_size,
            "total_chunks": self.total_chunks,
            "shards": [p.name for p in self.paths],
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        for key in ("seed", "seq_len", "world_size", "total_chunks"):
            if state[key] != getattr(self, key):
                raise ValueError(
                    f"加载器状态不匹配：{key} 保存时是 {state[key]}，现在是 {getattr(self, key)}。"
                    "改了数据、序列长度或卡数之后不能精确续训。"
                )
        self.consumed = int(state["consumed"])

    def reset(self) -> None:
        self.consumed = 0
