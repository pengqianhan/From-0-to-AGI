"""Learning-rate schedules: cosine and WSD (Chapters 6, 12, and 15).

Each schedule returns a multiplier m(step) ∈ [0, 1]. The learning rate is the peak lr × m(step).

- **warmup**: in the first warmup_steps steps, the multiplier increases linearly from about 0 to 1.
  At the start, the parameters are random and the gradient directions are noisy.
  A large learning rate at this time can make the training diverge.
- **cosine**: after warmup, the multiplier follows a cosine curve from 1 down to min_lr_ratio.
  GPT-3 and Llama use this schedule. Its problem: you must set the total number of steps
  before you start. To train for more steps, you must start again.
- **WSD (Warmup-Stable-Decay)**: warmup, then a long stable phase at the peak, then a fast decay
  in the last decay_frac of the steps. You can take any checkpoint from the stable phase and add
  a short decay to get a usable model. To train for more steps, continue from the stable phase.
  This is good for ladder experiments and for the "pretraining → mid-training (annealing)" stages.
  MiniCPM introduced WSD (arXiv:2404.06395). Chapter 12 lists the models that use it.
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
        raise ValueError(f"Unknown decay_shape: {decay_shape}")
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
    raise ValueError(f"Unknown schedule: {cfg.kind}")


class LRScheduler:
    """Set lr = base_lr × m(step) for each parameter group of the optimizer.

    The only state is the step. The checkpoint stores it.
    """

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
        """Write the learning rate of this step into the optimizer, and return the learning rate."""
        self.step_count = step
        lr = self.lr_at(step)
        for group in self.optimizer.param_groups:
            group["lr"] = lr * group.get("lr_scale", 1.0)
        return lr

    def state_dict(self) -> dict[str, Any]:
        return {"step_count": self.step_count, "base_lr": self.base_lr}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.step_count = int(state["step_count"])
