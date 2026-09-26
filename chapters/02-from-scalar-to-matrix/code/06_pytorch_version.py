"""第 2 章 · 从极简到生产级：多元线性回归的 PyTorch 标准写法

和 04_multivariate_regression.py 做的事完全一样：
- 模型是 nn.Linear(3, 1)，一次吃进整个 batch：输入 (N, 3) → 输出 (N, 1)；
- 注意 PyTorch 把权重存成 (out, in) = (1, 3)，前向算的是 x @ weight.T + bias；
- 训练循环和第 1 章的五行一模一样：前向 → 损失 → 清梯度 → 反向 → 更新。
最后和手写矩阵梯度的 NumPy 版对拍。
运行：uv run python chapters/02-from-scalar-to-matrix/code/06_pytorch_version.py
"""

import importlib.util
from pathlib import Path

import numpy as np
import torch
from torch import nn

_spec = importlib.util.spec_from_file_location(
    "multivariate_regression", Path(__file__).with_name("04_multivariate_regression.py")
)
reg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(reg)


def train_torch(Xs: np.ndarray, y: np.ndarray, dtype=torch.float64, lr=0.1, steps=200,
                verbose=False):
    x = torch.tensor(Xs, dtype=dtype)          # (N, 3)
    t = torch.tensor(y, dtype=dtype)           # (N, 1)

    model = nn.Linear(in_features=3, out_features=1, dtype=dtype)
    with torch.no_grad():                      # 和 NumPy 版同一个起点：W = 0，b = 0
        model.weight.zero_()
        model.bias.zero_()

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for step in range(steps):
        y_hat = model(x)                       # 1. 前向：(N, 3) → (N, 1)
        loss = loss_fn(y_hat, t)               # 2. 损失
        if verbose and step in (0, 10, 50):
            print(f"  第 {step:>3} 步：损失 = {loss.item():.3f}")
        optimizer.zero_grad()                  # 3. 清梯度
        loss.backward()                        # 4. 反向：autograd 算出 ∂L/∂W = 2/N·Xᵀ(ŷ−y)
        optimizer.step()                       # 5. 更新
    return model


def main() -> None:
    torch.manual_seed(0)
    X, y = reg.make_data()
    Xs, mu, sigma = reg.standardize(X)

    # nn.Linear 的前向就是 x @ weight.T + bias
    layer = nn.Linear(3, 1, dtype=torch.float64)
    xb = torch.tensor(Xs[:4])
    manual = xb @ layer.weight.T + layer.bias
    print(f"nn.Linear(3, 1)：weight 形状 {tuple(layer.weight.shape)}，bias 形状 {tuple(layer.bias.shape)}")
    print(f"输入 batch {tuple(xb.shape)} → 输出 {tuple(layer(xb).shape)}；"
          f"与 x @ weight.T + bias 的最大差 {(layer(xb) - manual).abs().max().item():.1e}\n")

    print("PyTorch 训练（float64）：")
    model = train_torch(Xs, y, verbose=True)
    W_t = model.weight.detach().numpy().T      # (3, 1)，转成和 NumPy 版一样的形状
    b_t = model.bias.detach().numpy()

    W_n, b_n, _ = reg.gradient_descent(Xs, y, lr=0.1, steps=200)[-1]
    print("\n200 步后（标准化空间）：")
    print(f"  PyTorch  W = {W_t.ravel().round(3)}，b = {b_t.round(3)}")
    print(f"  NumPy    W = {W_n.ravel().round(3)}，b = {b_n.round(3)}")
    diff64 = max(np.abs(W_t - W_n).max(), np.abs(b_t - b_n).max())
    print(f"  float64 最大差 = {diff64:.1e}")

    m32 = train_torch(Xs, y, dtype=torch.float32)
    diff32 = max(np.abs(m32.weight.detach().numpy().T - W_n).max(),
                 np.abs(m32.bias.detach().numpy() - b_n).max())
    print(f"  float32（PyTorch 默认精度）最大差 = {diff32:.1e}")

    W_o, b_o = reg.to_original_units(W_t, b_t, mu, sigma)
    print(f"\n换算回原始单位：w = {W_o.ravel().round(3)}，b = {b_o.round(3)}")


if __name__ == "__main__":
    main()
