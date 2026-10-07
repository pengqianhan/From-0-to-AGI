"""Chapter 14 · Minimal code 7: resume from a checkpoint. Restore "all" state to get a bit-identical loss.

Pretraining runs for days or weeks, and problems will occur: preemption, a broken GPU, a power failure,
or a loss spike that needs a rollback. Thus we save a checkpoint at regular intervals. After a crash,
we continue from the latest checkpoint. The standard for "continue" is: each step after the resume
is bit-identical to a run without an interruption.

To get this result, the model weights are not sufficient. The full state of a training run is:
  1. the model parameters
  2. the optimizer state (AdamW m and v, and the step count t for the bias correction)
  3. the position of the learning-rate scheduler
  4. the position in the data (which samples are in the next batch)
  5. the state of the random number generators (all RNGs for dropout, data shuffling, and so on)
  6. counters such as the step count

This script: first, train 30 steps without an interruption. Then train again and "crash" at step 17
(the latest checkpoint is at step 15). Restore from the checkpoint into new objects, train to step 30,
and compare the loss of each step. At the end, "forget" one item of the state at a time and measure the difference.

Run: uv run python chapters/14-pretraining-engineering/code/07_resume.py   (a few seconds)
"""

import io
import math

import torch
from torch import nn

STEPS, SAVE_AT, CRASH_AT = 30, 15, 17


class Run:
    """All mutable state of one training run is in this object."""

    def __init__(self):
        torch.manual_seed(0)
        self.model = nn.Sequential(nn.Linear(32, 128), nn.GELU(), nn.Dropout(0.1), nn.Linear(128, 8))
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=3e-3, weight_decay=0.1)
        warmup = 5
        self.sched = torch.optim.lr_scheduler.LambdaLR(
            self.opt, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * s / STEPS)))
        self.data_gen = torch.Generator().manual_seed(1234)    # the data loader's own RNG (which samples to draw)
        g = torch.Generator().manual_seed(42)
        self.X = torch.randn(4096, 32, generator=g)
        self.Y = (self.X[:, :8] * 2 + 0.3 * torch.randn(4096, 8, generator=g)).argmax(1)
        self.step = 0

    def train_step(self) -> float:
        idx = torch.randint(0, len(self.X), (64,), generator=self.data_gen)
        loss = nn.functional.cross_entropy(self.model(self.X[idx]), self.Y[idx])  # dropout uses the global RNG
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        self.sched.step()
        self.step += 1
        return loss.item()

    def state_dict(self) -> dict:
        return {
            "model": self.model.state_dict(),
            "optim": self.opt.state_dict(),
            "sched": self.sched.state_dict(),
            "data_rng": self.data_gen.get_state(),
            "torch_rng": torch.get_rng_state(),
            "step": self.step,
        }

    def load_state_dict(self, sd: dict, skip: str = "") -> None:
        self.model.load_state_dict(sd["model"])
        if skip != "optim":
            self.opt.load_state_dict(sd["optim"])
        if skip != "sched":
            self.sched.load_state_dict(sd["sched"])
        if skip != "data_rng":
            self.data_gen.set_state(sd["data_rng"])
        if skip != "torch_rng":
            torch.set_rng_state(sd["torch_rng"])
        self.step = sd["step"]


def resume(ckpt: bytes, skip: str = "") -> list[float]:
    torch.manual_seed(999)                  # simulate a "new process": the global RNG state is different
    run = Run()                             # a new model, optimizer, scheduler, and data loader
    run.load_state_dict(torch.load(io.BytesIO(ckpt), weights_only=False), skip)
    return [run.train_step() for _ in range(run.step, STEPS)]


def main():
    torch.set_num_threads(1)
    ref_run = Run()
    ref = [ref_run.train_step() for _ in range(STEPS)]

    run = Run()
    ckpt = None
    for _ in range(CRASH_AT):
        run.train_step()
        if run.step == SAVE_AT:
            buf = io.BytesIO()
            torch.save(run.state_dict(), buf)   # production code writes to disk (first to a temporary folder, then an atomic rename)
            ckpt = buf.getvalue()
    print(f"Crash at step {CRASH_AT}; the latest checkpoint is at step {SAVE_AT} ({len(ckpt):,} bytes)")
    resumed = resume(ckpt)

    print(f"\n① Restore all state: loss of steps {SAVE_AT + 1}–{STEPS}")
    print(f"  {'#':>3} {'no crash':>10} {'resumed':>10}  same?")
    for s in (SAVE_AT + 1, SAVE_AT + 2, CRASH_AT + 1, STEPS):
        a, b = ref[s - 1], resumed[s - SAVE_AT - 1]
        print(f"  {s:>3} {a:>10.6f} {b:>10.6f}  {'bit-identical' if a == b else 'different'}")
    same = all(a == b for a, b in zip(ref[SAVE_AT:], resumed))
    print(f"  All {STEPS - SAVE_AT} steps bit-identical: {same}")

    print("\n② Do not restore one item. Max loss difference from the run without a crash, over the 15 resumed steps:")
    labels = {
        "optim": "optimizer (m, v, t = 0)",
        "sched": "schedule (warmup again)",
        "data_rng": "data order (from start)",
        "torch_rng": "global RNG (new dropout)",
    }
    for key, label in labels.items():
        diff = max(abs(a - b) for a, b in zip(ref[SAVE_AT:], resume(ckpt, skip=key)))
        print(f"  {label:<24} max diff {diff:.2e}")


if __name__ == "__main__":
    main()
