"""第 3 章 · 从极简到生产级：同一个两层 MLP 的 PyTorch 标准写法

和 03_mlp_numpy.py 做的事一样，区别是：
- 网络用 nn.Sequential(nn.Linear, nn.ReLU, nn.Linear) 搭，不用自己写矩阵乘法；
- 梯度不再手推，loss.backward() 自动求（第 4 章会亲手实现它）；
- 更新交给 torch.optim.SGD。
两个实验：
1. 对拍：把 NumPy 版的初始参数原样拷进 PyTorch，用 float64、同样的学习率和步数，损失应该逐位一致；
2. 用 PyTorch 默认的初始化和 float32 从头训练，看能不能达到差不多的拟合效果。
运行：uv run python chapters/03-neural-network/code/04_pytorch_version.py
"""

import importlib.util
from pathlib import Path

import torch
from torch import nn

_spec = importlib.util.spec_from_file_location(
    "mlp_numpy", Path(__file__).with_name("03_mlp_numpy.py")
)
mlp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mlp)

HIDDEN = 64


def make_model(hidden: int = HIDDEN) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(1, hidden),    # Z = X·W1ᵀ + b1（PyTorch 的 weight 形状是 (输出, 输入)）
        nn.ReLU(),               # A = ReLU(Z)
        nn.Linear(hidden, 1),    # Ŷ = A·W2ᵀ + b2
    )


def fit(model: nn.Module, x: torch.Tensor, y: torch.Tensor, lr: float, steps: int,
        report=(0, 1000, 5000), verbose: bool = True) -> float:
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    for step in range(steps + 1):
        y_hat = model(x)              # 1. 前向
        loss = loss_fn(y_hat, y)      # 2. 损失
        if verbose and (step in report or step == steps):
            print(f"   第 {step:>5} 步：损失 = {loss.item():.4f}")
        if step == steps:
            return loss.item()
        optimizer.zero_grad()         # 3. 清梯度
        loss.backward()               # 4. 反向：autograd 自动求所有参数的梯度
        optimizer.step()              # 5. 更新


def run_match(verbose: bool = True):
    """对拍：NumPy 的初始参数拷进 PyTorch（float64），返回 (PyTorch 损失, NumPy 损失)。"""
    x_np, y_np = mlp.make_data()
    torch.set_default_dtype(torch.float64)
    x, y = torch.tensor(x_np), torch.tensor(y_np)
    model = make_model()
    p0 = mlp.init_params(HIDDEN, seed=0)
    with torch.no_grad():
        model[0].weight.copy_(torch.tensor(p0["W1"].T))   # (1, H) → (H, 1)
        model[0].bias.copy_(torch.tensor(p0["b1"]))
        model[2].weight.copy_(torch.tensor(p0["W2"].T))   # (H, 1) → (1, H)
        model[2].bias.copy_(torch.tensor(p0["b2"]))
    loss_torch = fit(model, x, y, lr=mlp.LR, steps=mlp.STEPS, verbose=verbose)
    loss_numpy = mlp.train(x_np, y_np, HIDDEN)[1][-1]
    torch.set_default_dtype(torch.float32)
    return loss_torch, loss_numpy


def run_default(verbose: bool = True):
    """PyTorch 默认初始化 + float32，从头训练。返回 (最终损失, 参数量)。"""
    x_np, y_np = mlp.make_data()
    torch.manual_seed(0)
    x, y = torch.tensor(x_np, dtype=torch.float32), torch.tensor(y_np, dtype=torch.float32)
    model = make_model()
    loss = fit(model, x, y, lr=mlp.LR, steps=mlp.STEPS, verbose=verbose)
    return loss, sum(p.numel() for p in model.parameters())


def main() -> None:
    print(f"1) 对拍：NumPy 的初始参数拷进 PyTorch（float64），宽 {HIDDEN}，SGD 学习率 {mlp.LR}")
    loss_torch, loss_numpy = run_match()
    print(f"   PyTorch {loss_torch:.10f}  vs  手推梯度 NumPy {loss_numpy:.10f}，"
          f"差 = {abs(loss_torch - loss_numpy):.1e}")

    print(f"\n2) PyTorch 默认初始化 + float32，宽 {HIDDEN}，同样训练 {mlp.STEPS} 步")
    _, n = run_default()
    print(f"   参数量 {n}（W1: {HIDDEN}，b1: {HIDDEN}，W2: {HIDDEN}，b2: 1）")


if __name__ == "__main__":
    main()
