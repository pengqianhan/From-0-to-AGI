"""学习率调度：cosine 与 WSD（对应第 6、12、15 章）。

所有调度都返回一个"倍率" m(step) ∈ [0, 1]，实际学习率 = 峰值 lr × m(step)。

- **warmup**：前 warmup_steps 步从接近 0 线性升到 1。刚开始参数是随机的，梯度方向很乱，
  直接用大学习率容易发散；
- **cosine**：warmup 之后按余弦曲线从 1 降到 min_lr_ratio（GPT-3、Llama 的默认选择）。
  缺点：总步数必须事先定好，中途想多训一些就得重来；
- **WSD（Warmup-Stable-Decay）**：warmup → 长时间保持峰值 → 最后 decay_frac 比例的步数快速衰减。
  好处是"稳定段"的任何 checkpoint 都可以接一小段衰减得到一个可用模型，想多训就从稳定段接着训，
  很适合阶梯实验和"预训练 → 中期训练（退火）"的分阶段做法（由 MiniCPM 提出，arXiv:2404.06395；
  采用方见第 12 章的来源列表）。
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

from zero.config import ScheduleConfig


def warmup_factor(step: int, warmup_steps: int) -> float:
    if warmup_steps <= 0:
        return 1.0
    return min(1.0, (step + 1) / warmup_steps)


def cosine_with_warmup(
    step: int, warmup_steps: int, total_steps: int, min_lr_ratio: float = 0.1
) -> float:
    if step < warmup_steps:
        return warmup_factor(step, warmup_steps)
    decay_steps = max(total_steps - warmup_steps, 1)
    progress = min(1.0, (step - warmup_steps) / decay_steps)
    return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))


def wsd(
    step: int,
    warmup_steps: int,
    total_steps: int,
    decay_frac: float = 0.2,
    min_lr_ratio: float = 0.0,
    decay_shape: str = "linear",
) -> float:
    decay_steps = int(round(total_steps * decay_frac))
    decay_start = total_steps - decay_steps
    if step < warmup_steps:
        return warmup_factor(step, warmup_steps)
    if step < decay_start or decay_steps == 0:
        return 1.0
    progress = min(1.0, (step - decay_start + 1) / decay_steps)
    if decay_shape == "linear":
        f = 1 - progress
    elif decay_shape == "cosine":
        f = 0.5 * (1 + math.cos(math.pi * progress))
    elif decay_shape == "sqrt":
        f = 1 - math.sqrt(progress)
    else:
        raise ValueError(f"未知的 decay_shape：{decay_shape}")
    return min_lr_ratio + (1 - min_lr_ratio) * f


def lr_multiplier_fn(cfg: ScheduleConfig, total_steps: int) -> Callable[[int], float]:
    if cfg.kind == "cosine":
        return lambda s: cosine_with_warmup(s, cfg.warmup_steps, total_steps, cfg.min_lr_ratio)
    if cfg.kind == "wsd":
        return lambda s: wsd(
            s, cfg.warmup_steps, total_steps, cfg.decay_frac, cfg.min_lr_ratio, cfg.decay_shape
        )
    if cfg.kind == "constant":
        return lambda s: warmup_factor(s, cfg.warmup_steps)
    raise ValueError(f"未知的调度：{cfg.kind}")


class LRScheduler:
    """给优化器的每个参数组设置 lr = base_lr × m(step)。状态只有 step，存进 checkpoint。"""

    def __init__(
        self, optimizer: Any, cfg: ScheduleConfig, base_lr: float, total_steps: int
    ) -> None:
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.fn = lr_multiplier_fn(cfg, total_steps)
        self.step_count = 0

    def lr_at(self, step: int) -> float:
        return self.base_lr * self.fn(step)

    def apply(self, step: int) -> float:
        """把第 step 步的学习率写进优化器，返回这个学习率。"""
        self.step_count = step
        lr = self.lr_at(step)
        for group in self.optimizer.param_groups:
            group["lr"] = lr * group.get("lr_scale", 1.0)
        return lr

    def state_dict(self) -> dict[str, Any]:
        return {"step_count": self.step_count, "base_lr": self.base_lr}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.step_count = int(state["step_count"])
