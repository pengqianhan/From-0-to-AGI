"""第 14 章 · 极简代码 6：数据并行（DDP）手写版 —— 梯度求平均，就这么简单

数据并行（data parallelism）：每张卡一份完整模型，各吃一份不同的数据，反向之后把所有卡的梯度
求平均（all-reduce），于是每张卡用同一个梯度更新，参数永远保持一致。

它为什么对？一个 batch 的平均损失对参数的梯度 = 各个子 batch 梯度的平均（求导是线性的）。
所以"2 个进程各算 B 条再平均" == "1 个进程算 2B 条" == "1 个进程分两次各算 B 条再累加（梯度累积）"。

这里做四件事：
  ① 单进程，一次吃 2B 条（参照）；
  ② 单进程，梯度累积：两个 micro batch 各 B 条，loss 先除以 2 再 backward，梯度自动累加
     （第 1 章说过：PyTorch 默认累加梯度，所以平时每步要 zero_grad —— 这里正是利用了这一点）；
  ③ 真的起 2 个进程（torch.distributed，gloo 后端，CPU 就能跑），每个进程吃自己那 B 条，
     backward 之后手动 all_reduce(梯度) / 2；
  ④ 在一个进程里模拟 4 张卡的环形 all-reduce（ring all-reduce），数一数每张卡发出去多少数据。

运行：uv run python chapters/14-pretraining-engineering/code/06_ddp_by_hand.py   （十几秒）
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
    torch.manual_seed(0)  # 每个进程用同一个种子 → 初始参数完全相同
    return nn.Sequential(nn.Linear(16, 64), nn.Tanh(), nn.Linear(64, 1)).double()


def batch(step: int) -> tuple[torch.Tensor, torch.Tensor]:
    """第 step 步的全局 batch（2B 条）。所有进程都能算出同一批，再各取一半。"""
    g = torch.Generator().manual_seed(1000 + step)
    x = torch.randn(2 * B, 16, generator=g, dtype=torch.float64)
    y = torch.sin(x.sum(1, keepdim=True))
    return x, y


def loss_fn(model, x, y):
    return ((model(x) - y) ** 2).mean()


def run_single(accum: int) -> tuple[list[float], torch.Tensor]:
    """accum = 1：一次吃 2B 条；accum = 2：分两个 micro batch，梯度累积。"""
    model = make_model()
    opt = torch.optim.SGD(model.parameters(), lr=LR)
    losses = []
    for step in range(STEPS):
        x, y = batch(step)
        total = 0.0
        for xs, ys in zip(x.chunk(accum), y.chunk(accum)):
            loss = loss_fn(model, xs, ys) / accum    # 除以累积步数：累加结果 = 大 batch 的平均梯度
            loss.backward()                          # 梯度累加进 .grad
            total += loss.item()
        opt.step()
        opt.zero_grad()                              # 一步结束才清零
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
        xs, ys = x.chunk(world)[rank], y.chunk(world)[rank]   # 每个进程只吃自己那一份
        loss = loss_fn(model, xs, ys)
        loss.backward()
        for p in model.parameters():                          # DDP 的全部秘密：
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)     #   把所有进程的梯度加起来
            p.grad /= world                                   #   再除以进程数 = 平均
        opt.step()
        opt.zero_grad()
        lt = loss.detach().clone()
        dist.all_reduce(lt)                                   # 日志里的 loss 也取平均
        losses.append(lt.item() / world)
    if rank == 0:
        torch.save({"losses": losses, "params": torch.cat([p.detach().flatten() for p in model.parameters()])}, out)
    dist.destroy_process_group()


def ring_all_reduce(bufs: list[np.ndarray]) -> tuple[list[np.ndarray], int]:
    """模拟 N 张卡围成一圈做 all-reduce。每张卡的数据切成 N 块：
    ① reduce-scatter：N−1 轮，每轮每张卡把一块发给右邻居，邻居加到自己那块上 → 每张卡各有一块完整的和；
    ② all-gather：再 N−1 轮，把完整的块沿环传一圈 → 每张卡都有全部的和。
    返回结果和每张卡发出的元素个数。"""
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
    print(f"① / ② / ③：同一个 MLP、同样的数据，训练 {STEPS} 步（float64）")
    ref_losses, ref_params = run_single(accum=1)
    acc_losses, acc_params = run_single(accum=2)
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "ddp.pt")
        mp.spawn(worker, args=(2, port, out), nprocs=2, join=True)
        ddp = torch.load(out)
    print(f"  {'步':>3} {'一次 2B 条':>12} {'梯度累积 2×B':>14} {'2 进程 all-reduce':>18}")
    for s in (0, 1, 2, STEPS - 1):
        print(f"  {s + 1:>3} {ref_losses[s]:>12.6f} {acc_losses[s]:>14.6f} {ddp['losses'][s]:>18.6f}")
    print(f"  最终参数最大差：梯度累积 {(acc_params - ref_params).abs().max().item():.1e}，"
          f"2 进程 {(ddp['params'] - ref_params).abs().max().item():.1e}")

    print("\n④ 环形 all-reduce（4 张卡，每张卡 1,000,000 个梯度）")
    rng = np.random.default_rng(0)
    n, size = 4, 1_000_000
    bufs = [rng.standard_normal(size) for _ in range(n)]
    out_bufs, sent = ring_all_reduce(bufs)
    truth = sum(bufs)
    print(f"  每张卡结果与直接求和的最大差：{max(np.abs(o - truth).max() for o in out_bufs):.1e}")
    print(f"  每张卡发出 {sent:,} 个数 = 2·(N−1)/N × {size:,}，和卡数几乎无关")
    P = 689_518_848
    print(f"  主线模型 {P / 1e6:.1f}M 参数：FP32 梯度 {4 * P / 2**30:.2f} GiB，8 卡环形 all-reduce "
          f"每卡每步发送 {2 * 7 / 8 * 4 * P / 2**30:.2f} GiB")


if __name__ == "__main__":
    main()
