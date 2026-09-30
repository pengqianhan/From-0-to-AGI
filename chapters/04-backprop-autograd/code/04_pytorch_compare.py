"""第 4 章 · 从极简到生产级：同一个 MLP，用 PyTorch autograd 再算一遍

1. 对拍梯度：把 Value 版 MLP 的初始权重原样拷进 PyTorch，算同一个损失，逐个参数比较梯度；
2. 对拍训练：两边都用 SGD（lr = 0.1）训练 500 步，比较最终损失；
3. 比速度：标量计算图 vs 张量计算图；
4. torch.autograd.gradcheck：PyTorch 自带的梯度检验；
5. 向量-雅可比积（VJP）：张量版的反向传播到底在算什么。

运行：uv run python chapters/04-backprop-autograd/code/04_pytorch_compare.py
"""

import importlib.util
import math
import random
import time
from pathlib import Path

import torch
from torch import nn

HERE = Path(__file__).parent
_spec = importlib.util.spec_from_file_location("train_mlp", HERE / "03_train_mlp.py")
train_mlp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(train_mlp)
Value, MLP = train_mlp.Value, train_mlp.MLP

torch.set_num_threads(1)               # 小矩阵用单线程最快，计时也更稳定
torch.set_default_dtype(torch.float64)  # Value 用的是 Python float（双精度），对拍时两边精度一致


def to_torch(net) -> nn.Sequential:
    """把 Value 版 MLP 的权重拷进 nn.Linear。nn.Linear 的 weight 形状是 (输出, 输入)。"""
    mods = []
    for i, layer in enumerate(net.layers):
        lin = nn.Linear(len(layer.neurons[0].w), len(layer.neurons))
        with torch.no_grad():
            lin.weight.copy_(torch.tensor([[w.data for w in n.w] for n in layer.neurons]))
            lin.bias.copy_(torch.tensor([n.b.data for n in layer.neurons]))
        mods.append(lin)
        if i < len(net.layers) - 1:
            mods.append(nn.Tanh())
    return nn.Sequential(*mods)


def value_grads_as_tensors(net) -> list[torch.Tensor]:
    """按 PyTorch 参数的顺序（每层 weight、bias）整理 Value 版的梯度。"""
    out = []
    for layer in net.layers:
        out.append(torch.tensor([[w.grad for w in n.w] for n in layer.neurons]))
        out.append(torch.tensor([n.b.grad for n in layer.neurons]))
    return out


def time_value_step(net, xs, ys, reps: int) -> float:
    t0 = time.perf_counter()
    for _ in range(reps):
        loss = train_mlp.mse(net, xs, ys)
        net.zero_grad()
        loss.backward()
    return (time.perf_counter() - t0) / reps


def time_torch_step(model, x, y, reps: int) -> float:
    t0 = time.perf_counter()
    for _ in range(reps):
        loss = ((model(x) - y) ** 2).mean()
        model.zero_grad()
        loss.backward()
    return (time.perf_counter() - t0) / reps


if __name__ == "__main__":
    xs, ys = train_mlp.make_data()
    x = torch.tensor(xs).unsqueeze(1)          # 形状 (20, 1)：一次处理整批样本
    y = torch.tensor(ys).unsqueeze(1)

    # ── 1. 对拍梯度 ───────────────────────────────────────────────────────────
    random.seed(0)                             # 和 03_train_mlp.py 的 train() 同一个初始化
    net = MLP(1, [8, 8, 1])
    model = to_torch(net)
    loss_v = train_mlp.mse(net, xs, ys)
    net.zero_grad()
    loss_v.backward()
    loss_t = ((model(x) - y) ** 2).mean()      # nn.MSELoss() 算的是同一个式子
    model.zero_grad()
    loss_t.backward()
    print(f"初始损失：Value = {loss_v.data:.12f}，PyTorch = {loss_t.item():.12f}")
    names = [n for n, _ in model.named_parameters()]
    max_diff = 0.0
    for name, p, g in zip(names, model.parameters(), value_grads_as_tensors(net), strict=True):
        ok = torch.allclose(p.grad, g, rtol=1e-9, atol=1e-12)
        diff = (p.grad - g).abs().max().item()
        max_diff = max(max_diff, diff)
        print(f"  {name:<9} 形状 {str(tuple(p.shape)):<7} allclose = {ok}  最大差 {diff:.1e}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  共 {n_params} 个参数，梯度最大差 {max_diff:.1e}")

    # ── 2. 对拍训练 ───────────────────────────────────────────────────────────
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    for _ in range(500):
        loss_t = ((model(x) - y) ** 2).mean()
        opt.zero_grad()
        loss_t.backward()
        opt.step()
    final_t = ((model(x) - y) ** 2).mean().item()
    _, losses, _ = train_mlp.train(steps=500, lr=0.1)
    print(f"\nSGD 500 步后的损失：Value = {losses[-1]:.10f}，PyTorch = {final_t:.10f}")

    # ── 3. 速度：一次"前向 + 反向" ────────────────────────────────────────────
    print("\n一次前向 + 反向的耗时：")
    print("  样本数   Value（标量图）   PyTorch（张量图）   倍数")
    for n in (20, 200):
        xs_n = [-3 + 6 * i / (n - 1) for i in range(n)]
        ys_n = [math.sin(v) for v in xs_n]
        tv = time_value_step(net, xs_n, ys_n, reps=10)
        xt, yt = torch.tensor(xs_n).unsqueeze(1), torch.tensor(ys_n).unsqueeze(1)
        time_torch_step(model, xt, yt, reps=20)   # 预热
        tt = time_torch_step(model, xt, yt, reps=500)
        print(f"  {n:>6}   {tv * 1000:10.1f} ms   {tt * 1000:12.3f} ms   {tv / tt:8.0f}×")

    # ── 4. torch.autograd.gradcheck：拿数值梯度检验 autograd，和 02_grad_check.py 同一个思路 ──
    params = {k: v.detach().clone().requires_grad_(True) for k, v in model.named_parameters()}

    def loss_of_params(*ps):
        out = torch.func.functional_call(model, dict(zip(params, ps, strict=True)), (x,))
        return ((out - y) ** 2).mean()
    ok = torch.autograd.gradcheck(loss_of_params, tuple(params.values()), eps=1e-6, atol=1e-6)
    print(f"\ntorch.autograd.gradcheck（对全部 {n_params} 个参数）：{ok}")

    # ── 5. 向量-雅可比积：Y = X·W 的反向，就是把上游梯度 G 乘以 Wᵀ ─────────────
    torch.manual_seed(0)
    X = torch.randn(4, 3, requires_grad=True)
    W = torch.randn(3, 2)
    Y = X @ W
    G = torch.randn(4, 2)                      # 上游传回来的梯度 ∂L/∂Y
    (gX,) = torch.autograd.grad(Y, X, grad_outputs=G)
    print(f"VJP：autograd 给出的 ∂L/∂X 与 G·Wᵀ 一致：{torch.allclose(gX, G @ W.T)}"
          f"（完整雅可比矩阵有 {Y.numel()}×{X.numel()} = {Y.numel() * X.numel()} 个数，从不需要显式构造）")
