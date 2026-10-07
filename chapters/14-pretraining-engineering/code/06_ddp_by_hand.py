"""Chapter 14 · Minimal code 6: data parallelism (DDP) by hand. It is only the mean of the gradients.

Data parallelism: each GPU has a full copy of the model and gets a different part of the data.
After the backward pass, all GPUs calculate the mean of their gradients (all-reduce).
Thus each GPU updates with the same gradient, and the parameters always stay the same on all GPUs.

Why is this correct? The gradient of the mean loss of a batch = the mean of the gradients of its
sub-batches (differentiation is linear).
Thus "2 processes, B samples each, then the mean" == "1 process with 2B samples"
== "1 process, two times B samples, gradients added (gradient accumulation)".

The script does four things:
  ① one process, 2B samples at a time (the reference);
  ② one process, gradient accumulation: two micro batches of B samples each. Divide each loss by 2
     before backward, and the gradients add up automatically
     (Chapter 1: PyTorch adds gradients by default, so each step needs zero_grad. Here we use this property);
  ③ start 2 real processes (torch.distributed, gloo backend, runs on a CPU). Each process uses its own
     B samples. After backward, it calls all_reduce(gradient) / 2 by hand;
  ④ simulate a ring all-reduce of 4 GPUs in one process, and count how much data each GPU sends.

Run: uv run python chapters/14-pretraining-engineering/code/06_ddp_by_hand.py   (10-20 s)
"""

import os
import socket
import tempfile

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

B, STEPS, LR = 8, 20, 0.05


def make_model() -> nn.Module:
    torch.manual_seed(0)  # all processes use the same seed → the initial parameters are the same
    return nn.Sequential(nn.Linear(16, 64), nn.Tanh(), nn.Linear(64, 1)).double()


def batch(step: int) -> tuple[torch.Tensor, torch.Tensor]:
    """The global batch of this step (2B samples). All processes make the same batch, then each takes its half."""
    g = torch.Generator().manual_seed(1000 + step)
    x = torch.randn(2 * B, 16, generator=g, dtype=torch.float64)
    y = torch.sin(x.sum(1, keepdim=True))
    return x, y


def loss_fn(model, x, y):
    return ((model(x) - y) ** 2).mean()


def run_single(accum: int) -> tuple[list[float], torch.Tensor]:
    """accum = 1: 2B samples at a time; accum = 2: two micro batches with gradient accumulation."""
    model = make_model()
    opt = torch.optim.SGD(model.parameters(), lr=LR)
    losses = []
    for step in range(STEPS):
        x, y = batch(step)
        total = 0.0
        for xs, ys in zip(x.chunk(accum), y.chunk(accum)):
            loss = loss_fn(model, xs, ys) / accum    # divide by the accumulation steps: the sum = mean gradient of the large batch
            loss.backward()                          # the gradient is added into .grad
            total += loss.item()
        opt.step()
        opt.zero_grad()                              # set to zero only at the end of the step
        losses.append(total)
    return losses, torch.cat([p.detach().flatten() for p in model.parameters()])


def worker(rank: int, world: int, port: int, out: str) -> None:
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    torch.set_num_threads(1)
    dist.init_process_group("gloo", rank=rank, world_size=world)
    model = make_model()
    opt = torch.optim.SGD(model.parameters(), lr=LR)
    losses = []
    for step in range(STEPS):
        x, y = batch(step)
        xs, ys = x.chunk(world)[rank], y.chunk(world)[rank]   # each process uses only its own part
        loss = loss_fn(model, xs, ys)
        loss.backward()
        for p in model.parameters():                          # all of DDP is here:
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)     #   add the gradients of all processes
            p.grad /= world                                   #   divide by the number of processes = mean
        opt.step()
        opt.zero_grad()
        lt = loss.detach().clone()
        dist.all_reduce(lt)                                   # the loss in the log is also the mean
        losses.append(lt.item() / world)
    if rank == 0:
        torch.save({"losses": losses, "params": torch.cat([p.detach().flatten() for p in model.parameters()])}, out)
    dist.destroy_process_group()


def ring_all_reduce(bufs: list[np.ndarray]) -> tuple[list[np.ndarray], int]:
    """Simulate an all-reduce of N GPUs in a ring. The data of each GPU is cut into N chunks.

    ① reduce-scatter: N−1 rounds. In each round, each GPU sends one chunk to its right neighbor,
       and the neighbor adds it to its own chunk → each GPU has one chunk of the full sum;
    ② all-gather: N−1 more rounds. The full chunks go around the ring once → each GPU has the full sum.
    Return the results and the number of elements that each GPU sends.
    """
    n = len(bufs)
    chunks = [np.array_split(b.copy(), n) for b in bufs]
    sent = 0
    for r in range(n - 1):                                  # reduce-scatter
        msgs = [(i, (i - r) % n, chunks[i][(i - r) % n].copy()) for i in range(n)]
        for i, c, data in msgs:
            chunks[(i + 1) % n][c] += data
        sent += len(msgs[0][2])
    for r in range(n - 1):                                  # all-gather
        msgs = [(i, (i + 1 - r) % n, chunks[i][(i + 1 - r) % n].copy()) for i in range(n)]
        for i, c, data in msgs:
            chunks[(i + 1) % n][c] = data
        sent += len(msgs[0][2])
    return [np.concatenate(c) for c in chunks], sent


def main():
    torch.set_num_threads(1)
    print(f"① / ② / ③: the same MLP and the same data, {STEPS} training steps (float64)")
    ref_losses, ref_params = run_single(accum=1)
    acc_losses, acc_params = run_single(accum=2)
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "ddp.pt")
        mp.spawn(worker, args=(2, port, out), nprocs=2, join=True)
        ddp = torch.load(out)
    print(f"  {'#':>3} {'one batch 2B':>12} {'grad accum 2×B':>14} {'2 procs all-reduce':>18}")
    for s in (0, 1, 2, STEPS - 1):
        print(f"  {s + 1:>3} {ref_losses[s]:>12.6f} {acc_losses[s]:>14.6f} {ddp['losses'][s]:>18.6f}")
    print(f"  Max difference of the final parameters: gradient accumulation {(acc_params - ref_params).abs().max().item():.1e}, "
          f"2 processes {(ddp['params'] - ref_params).abs().max().item():.1e}")

    print("\n④ Ring all-reduce (4 GPUs, 1,000,000 gradients on each GPU)")
    rng = np.random.default_rng(0)
    n, size = 4, 1_000_000
    bufs = [rng.standard_normal(size) for _ in range(n)]
    out_bufs, sent = ring_all_reduce(bufs)
    truth = sum(bufs)
    print(f"  Max difference between the result of each GPU and the direct sum: {max(np.abs(o - truth).max() for o in out_bufs):.1e}")
    print(f"  Each GPU sends {sent:,} numbers = 2·(N−1)/N × {size:,}. This is almost independent of the number of GPUs")
    P = 689_518_848
    print(f"  Main-line model, {P / 1e6:.1f}M parameters: FP32 gradients {4 * P / 2**30:.2f} GiB. With an 8-GPU ring all-reduce, "
          f"each GPU sends {2 * 7 / 8 * 4 * P / 2**30:.2f} GiB per step")


if __name__ == "__main__":
    main()
