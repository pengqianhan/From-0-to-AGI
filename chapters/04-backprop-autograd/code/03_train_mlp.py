"""第 4 章 · 极简代码 3：用自己写的 autograd 训练一个两层隐藏层的 MLP

任务：拟合曲线 y = sin(x)，x ∈ [−3, 3]，20 个点。网络是 MLP(1, [8, 8, 1])，激活函数 tanh。
和第 3 章相比，这里没有任何一行手推的梯度：前向算出 loss，调用 loss.backward()，所有参数的梯度就都有了。
整个训练在 CPU 上约半分钟（标量 autograd 很慢，这正是本章最后要讲的问题）。

运行：uv run python chapters/04-backprop-autograd/code/03_train_mlp.py
"""

import importlib.util
import math
import random
import time
from pathlib import Path

_spec = importlib.util.spec_from_file_location("engine", Path(__file__).with_name("01_engine.py"))
engine = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(engine)
Value, MLP = engine.Value, engine.MLP


def make_data(n: int = 20):
    xs = [-3 + 6 * i / (n - 1) for i in range(n)]
    ys = [math.sin(x) for x in xs]
    return xs, ys


GRID = [-3 + 6 * i / 60 for i in range(61)]


def mse(net, xs, ys) -> Value:
    """L = 1/N · Σ (ŷ_i − y_i)²，整个式子都由 Value 组成，所以它就是一张计算图。"""
    preds = [net([Value(x)]) for x in xs]
    return sum(((p - y) ** 2 for p, y in zip(preds, ys, strict=True)), Value(0.0)) * (1 / len(xs))


def count_nodes(root: Value) -> int:
    seen, stack = set(), [root]
    while stack:
        v = stack.pop()
        if v not in seen:
            seen.add(v)
            stack.extend(v._prev)
    return len(seen)


def train(steps: int = 500, lr: float = 0.1, seed: int = 0, snapshot_at=(), verbose=False):
    """返回 (net, 每一步的损失, {步数: 这一步更新前在 GRID 上的预测})。"""
    random.seed(seed)
    xs, ys = make_data()
    net = MLP(1, [8, 8, 1], act="tanh")
    losses, snaps = [], {}
    for step in range(steps + 1):
        loss = mse(net, xs, ys)           # 1. 前向：顺便记录下整张计算图
        losses.append(loss.data)
        if step in snapshot_at:           # 记下这一步的预测曲线（画图用，不参与训练）
            snaps[step] = [net([Value(x)]).data for x in GRID]
        if step == steps:
            break                         # 最后一步只评估，不更新
        net.zero_grad()                   # 2. 清梯度（因为 backward 里用的是 +=）
        loss.backward()                   # 3. 反向：自动算出全部参数的梯度
        for p in net.parameters():        # 4. 更新：p ← p − η · ∂L/∂p
            p.data -= lr * p.grad
        if verbose and step in (0, 1, 10, 50, 100, 200, 300, 400):
            print(f"{step:>5}  {loss.data:9.5f}")
    return net, losses, snaps


if __name__ == "__main__":
    xs, ys = make_data()
    random.seed(0)
    probe = MLP(1, [8, 8, 1])
    loss = mse(probe, xs, ys)
    print(f"MLP(1, [8, 8, 1])：{len(probe.parameters())} 个参数")
    print(f"一次前向（20 个样本）记录下的计算图有 {count_nodes(loss)} 个 Value 节点\n")

    print(" 步数       损失")
    t0 = time.perf_counter()
    net, losses, _ = train(verbose=True)
    dt = time.perf_counter() - t0
    print(f"{500:>5}  {losses[-1]:9.5f}")
    print(f"\n500 步用时 {dt:.1f} 秒，平均每步 {dt / 500 * 1000:.0f} 毫秒")

    print("\n  x       sin(x)   预测")
    for x, y in list(zip(xs, ys, strict=True))[::4]:
        print(f"{x:6.2f}  {y:8.3f}  {net([Value(x)]).data:7.3f}")
