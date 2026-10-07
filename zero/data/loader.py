"""Packed data loader: cut fixed-length training samples from uint32 shards (Chapter 14).

**Packing**: pretraining does not align samples with documents. It treats all tokens as one long
stream and cuts it into chunks of length seq_len+1: x = chunk[:-1], y = chunk[1:]. Two adjacent
chunks overlap by 1 token (the stride is seq_len), so each token is a "prediction target" exactly
once. The <|endoftext|> at each document boundary tells the model that the context changed.
One chunk can contain the end and the start of several documents.

**Determinism and resume**: only (seed, epoch) decides the order of the samples:
- In each epoch, first shuffle the order of the shards, then shuffle the order of the chunks in
  each shard. With these two levels, we do not need one very large permutation of all chunks.
- So we can compute directly "which chunk of which shard is global sample g" (random access).
The full state of the loader is one integer, `consumed` (the number of samples that this rank took).
`state_dict()` saves it and `load_state_dict()` restores it. After a resume, the batches are
identical, byte for byte, to a run without interruption.

**Distributed**: global sample g goes to rank = g % world_size. In each step, all ranks together
take exactly batch_size * world_size consecutive global samples. So "2 GPUs with B rows each" and
"1 GPU with 2B rows" see the same data (`tests/test_ddp_cpu.py` uses this for a parity check of DDP).
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
    """A glob or a list of paths → a sorted list of .bin files."""
    items = [paths] if isinstance(paths, str | os.PathLike) else list(paths)
    out: list[Path] = []
    for item in items:
        matches = sorted(glob.glob(str(item)))
        if not matches:
            raise FileNotFoundError(f"Shards not found: {item}")
        out.extend(Path(m) for m in matches if m.endswith(".bin"))
    if not out:
        raise FileNotFoundError(f"No .bin shards: {paths}")
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
        # Number of (seq_len+1) chunks in each shard (stride seq_len)
        self.n_chunks = np.array(
            [max((len(s) - 1) // seq_len, 0) for s in self.shards], dtype=np.int64
        )
        self.total_chunks = int(self.n_chunks.sum())
        if self.total_chunks < world_size:
            raise ValueError(
                f"Not enough data: {self.total_chunks} chunks of length {seq_len}+1 are too few for {world_size} ranks"
            )
        self.consumed = 0
        self._epoch_cache: tuple[int, np.ndarray, np.ndarray] | None = None
        self._perm_cache: OrderedDict[tuple[int, int], np.ndarray] = OrderedDict()

    # ---- index computation ----
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
        """Global sample global_index → (epoch, shard index, chunk index)."""
        epoch, pos = divmod(global_index, self.total_chunks)
        order, cum = self._epoch_layout(epoch)
        j = int(np.searchsorted(cum, pos, side="right"))
        shard = int(order[j])
        within = pos - (int(cum[j - 1]) if j > 0 else 0)
        perm = self._chunk_perm(epoch, shard)
        chunk = int(perm[within]) if perm is not None else within
        return epoch, shard, chunk

    def sample(self, local_index: int) -> np.ndarray:
        """Sample local_index of this rank (a token array of length seq_len+1)."""
        g = local_index * self.world_size + self.rank
        _, shard, chunk = self.locate(g)
        start = chunk * self.seq_len
        return np.asarray(self.shards[shard][start : start + self.seq_len + 1])

    # ---- get data ----
    def next_samples(self, n: int) -> np.ndarray:
        out = np.stack([self.sample(self.consumed + i) for i in range(n)])
        self.consumed += n
        return out

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (x, y). Both have the shape (batch_size, seq_len), int64."""
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

    # ---- resume ----
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
                    f"The loader state does not match: {key} was {state[key]} when saved, now it is {getattr(self, key)}. "
                    "After a change of the data, the sequence length, or the number of GPUs, an exact resume is not possible."
                )
        self.consumed = int(state["consumed"])

    def reset(self) -> None:
        self.consumed = 0


class MaskedWindowLoader:
    """For SFT: read "fixed-length windows + loss mask" (Chapter 16).

    zero/post/sft.py packs several conversations into windows of length seq_len+1. It does not split
    a conversation across two windows, and it fills the remaining positions with <|endoftext|>.
    It writes two files:

        <name>.bin   np.uint32, n_windows × (seq_len+1) tokens
        <name>.mask  np.uint8, the same length; 1 = token of the assistant output (in the loss), 0 = all others

    `next_batch()` returns (x, y): x = window[:-1], y = window[1:]. Target positions with mask 0
    become -100 (the ignore_index of the cross-entropy). So `Transformer.loss` and the training loop
    compute the loss only on the assistant tokens, without changes.

    Only (seed, epoch) decides the order. The state is one integer, `consumed`, so a resume is as
    exact as with PackedDataLoader. With multiple GPUs, global sample g goes to rank g % world_size.
    """

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
        ignore_index: int = -100,
    ) -> None:
        self.paths = resolve_shards(paths)
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self.shuffle = shuffle
        self.device = device
        self.ignore_index = ignore_index
        w = seq_len + 1
        toks, masks = [], []
        for p in self.paths:
            t = np.fromfile(p, dtype=np.uint32)
            m = np.fromfile(p.with_suffix(".mask"), dtype=np.uint8)
            if len(t) % w != 0 or len(m) != len(t):
                raise ValueError(
                    f"{p}: the length {len(t)} is not a multiple of the window length {w}, or the mask length does not match (did seq_len change?)"
                )
            toks.append(t.reshape(-1, w))
            masks.append(m.reshape(-1, w))
        self.tokens = np.concatenate(toks)
        self.masks = np.concatenate(masks).astype(bool)
        self.total_chunks = len(self.tokens)
        if self.total_chunks < world_size:
            raise ValueError(f"Only {self.total_chunks} SFT windows: too few for {world_size} ranks")
        self.consumed = 0
        self._perm: tuple[int, np.ndarray] | None = None

    def _index(self, g: int) -> int:
        epoch, pos = divmod(g, self.total_chunks)
        if not self.shuffle:
            return pos
        if self._perm is None or self._perm[0] != epoch:
            self._perm = (
                epoch,
                np.random.default_rng([self.seed, epoch, 7]).permutation(self.total_chunks),
            )
        return int(self._perm[1][pos])

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        idx = [
            self._index((self.consumed + i) * self.world_size + self.rank)
            for i in range(self.batch_size)
        ]
        self.consumed += self.batch_size
        tok = self.tokens[idx].astype(np.int64)
        y = np.where(self.masks[idx][:, 1:], tok[:, 1:], self.ignore_index)
        x_t = torch.from_numpy(np.ascontiguousarray(tok[:, :-1]))
        y_t = torch.from_numpy(np.ascontiguousarray(y))
        device = torch.device(self.device)
        return x_t.to(device), y_t.to(device)

    def __iter__(self) -> MaskedWindowLoader:
        return self

    def __next__(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.next_batch()

    def state_dict(self) -> dict[str, Any]:
        return {
            "consumed": self.consumed,
            "seed": self.seed,
            "seq_len": self.seq_len,
            "world_size": self.world_size,
            "total_chunks": self.total_chunks,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        for key in ("seed", "seq_len", "world_size", "total_chunks"):
            if state[key] != getattr(self, key):
                raise ValueError(
                    f"The loader state does not match: {key} was {state[key]} when saved, now it is {getattr(self, key)}"
                )
        self.consumed = int(state["consumed"])
