"""Sampling from multiple sources with a fixed mixture (Chapters 13 and 15).

Pretraining mixes sources such as web text, code, math, and Chinese text in fixed proportions.
Mid-training uses a different mixture (more high-quality data, plus data in the instruction and
tool-call formats). The method here is "one draw per row":

- For each row of a batch, draw a source by its weight. Then take the next sample from the
  `PackedDataLoader` of that source.
- Only (seed, rank, row index) decide the draw. The rows are in blocks of 1024. For each block,
  `default_rng([seed, rank, block index])` draws all rows at one time. So the state is only
  "the number of rows drawn" + the state of the loader of each source. A resumed run gives the
  same bytes as an uninterrupted run.

The loader of each source is already split by rank (see loader.py). Each rank draws on its own,
and the ranks do not repeat samples.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import torch

from zero.data.loader import PackedDataLoader, to_torch_batch

_BLOCK = 1024


class MixtureSampler:
    """Deterministic weighted draws: the result of draw i depends only on (seed, rank, i)."""

    def __init__(self, weights: Mapping[str, float], seed: int = 0, rank: int = 0) -> None:
        if not weights:
            raise ValueError("weights must not be empty")
        self.names = list(weights)
        w = np.array([float(weights[n]) for n in self.names])
        if (w <= 0).any():
            raise ValueError(f"All weights must be > 0: {dict(weights)}")
        self.probs = w / w.sum()
        self.seed = seed
        self.rank = rank
        self.drawn = 0
        self._block: tuple[int, np.ndarray] | None = None

    def choice_at(self, i: int) -> str:
        b, off = divmod(i, _BLOCK)
        if self._block is None or self._block[0] != b:
            rng = np.random.default_rng([self.seed, self.rank, b])
            self._block = (b, rng.choice(len(self.names), size=_BLOCK, p=self.probs))
        return self.names[int(self._block[1][off])]

    def draw(self, n: int) -> list[str]:
        out = [self.choice_at(self.drawn + k) for k in range(n)]
        self.drawn += n
        return out


class MixtureLoader:
    """Mix several `PackedDataLoader`s by weight into one loader. The interface is the same as PackedDataLoader."""

    def __init__(
        self,
        loaders: Mapping[str, PackedDataLoader],
        weights: Mapping[str, float],
        batch_size: int,
        seed: int = 0,
        rank: int = 0,
        device: torch.device | str = "cpu",
    ) -> None:
        missing = set(weights) - set(loaders)
        if missing:
            raise ValueError(f"These sources have a weight but no loader: {sorted(missing)}")
        self.loaders = dict(loaders)
        self.sampler = MixtureSampler(weights, seed=seed, rank=rank)
        self.batch_size = batch_size
        self.device = device
        seq_lens = {ld.seq_len for ld in self.loaders.values()}
        if len(seq_lens) != 1:
            raise ValueError(f"All sources must have the same seq_len, got {seq_lens}")
        self.seq_len = seq_lens.pop()
        self.counts: dict[str, int] = dict.fromkeys(self.loaders, 0)  # samples taken from each source (for statistics)

    def next_samples(self, n: int) -> np.ndarray:
        rows = []
        for name in self.sampler.draw(n):
            rows.append(self.loaders[name].next_samples(1)[0])
            self.counts[name] += 1
        return np.stack(rows)

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        return to_torch_batch(self.next_samples(self.batch_size), self.device)

    def __iter__(self) -> MixtureLoader:
        return self

    def __next__(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.next_batch()

    def state_dict(self) -> dict[str, Any]:
        return {
            "drawn": self.sampler.drawn,
            "seed": self.sampler.seed,
            "names": self.sampler.names,
            "probs": self.sampler.probs.tolist(),
            "counts": dict(self.counts),
            "loaders": {k: v.state_dict() for k, v in self.loaders.items()},
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state["names"] != self.sampler.names or not np.allclose(
            state["probs"], self.sampler.probs
        ):
            raise ValueError(
                "The mixture is different from the saved one, so an exact resume is not possible "
                "(to change the mixture for mid-training, start from a new loader state)"
            )
        self.sampler.drawn = int(state["drawn"])
        self.counts = dict(state["counts"])
        for k, v in state["loaders"].items():
            self.loaders[k].load_state_dict(v)
