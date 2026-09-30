"""多来源按比例混合采样（对应第 13、15 章）。

预训练要把网页、代码、数学、中文等来源按一定比例混在一起；中期训练会换一套比例
（加大高质量数据、加入指令和工具调用格式数据）。这里的做法是"逐行抽签"：

- batch 里的每一行，先按权重抽一个来源，再从那个来源的 `PackedDataLoader` 取下一个样本；
- 抽签结果只由 (seed, rank, 行号) 决定：行号每 1024 行一组，每组用 `default_rng([seed, rank, 组号])`
  一次抽完。于是状态只需要记"已经抽了多少行" + 各来源加载器自己的状态，续训逐字节一致。

每个来源的加载器都按 rank 分好了片（见 loader.py），不同 rank 各自抽签、互不重复。
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import torch

from zero.data.loader import PackedDataLoader, to_torch_batch

_BLOCK = 1024


class MixtureSampler:
    """确定性的加权抽签：第 i 次抽签的结果只取决于 (seed, rank, i)。"""

    def __init__(self, weights: Mapping[str, float], seed: int = 0, rank: int = 0) -> None:
        if not weights:
            raise ValueError("weights 不能为空")
        self.names = list(weights)
        w = np.array([float(weights[n]) for n in self.names])
        if (w <= 0).any():
            raise ValueError(f"权重必须 > 0：{dict(weights)}")
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
    """把多个 `PackedDataLoader` 按权重混成一个加载器，接口与 PackedDataLoader 相同。"""

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
            raise ValueError(f"这些来源有权重但没有加载器：{sorted(missing)}")
        self.loaders = dict(loaders)
        self.sampler = MixtureSampler(weights, seed=seed, rank=rank)
        self.batch_size = batch_size
        self.device = device
        seq_lens = {ld.seq_len for ld in self.loaders.values()}
        if len(seq_lens) != 1:
            raise ValueError(f"各来源的 seq_len 必须相同，实际 {seq_lens}")
        self.seq_len = seq_lens.pop()
        self.counts: dict[str, int] = dict.fromkeys(self.loaders, 0)  # 各来源已取的样本数（统计用）

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
                "混合比例和保存时不同，不能精确续训（中期训练换比例请从新的加载器状态开始）"
            )
        self.sampler.drawn = int(state["drawn"])
        self.counts = dict(state["counts"])
        for k, v in state["loaders"].items():
            self.loaders[k].load_state_dict(v)
