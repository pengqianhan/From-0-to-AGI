"""第 1 章 · 从极简到生产级：同一件事的 PyTorch 标准写法

和 01_fit_line.py 做的事完全一样，区别是：
- 梯度不再手算，由 autograd 自动求（第 4 章会讲它怎么做到的）；
- 模型用 nn.Linear（一个输入、一个输出，权重就是 a，偏置就是 b）；
- 参数更新交给 torch.optim.SGD；
- 训练循环的五行骨架（前向 → 损失 → 清梯度 → 反向 → 更新），和训练大模型时一模一样。
运行：uv run python chapters/01-linear-regression/code/03_pytorch_version.py
"""

import importlib.util
from pathlib import Path

import torch
from torch import nn

_spec = importlib.util.spec_from_file_location(
    "fit_line", Path(__file__).with_name("01_fit_line.py")
)
fit_line = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fit_line)


def main() -> None:
    x_np, y_np = fit_line.make_data()
    x = torch.tensor(x_np, dtype=torch.float32).unsqueeze(1)  # 形状 (50, 1)
    y = torch.tensor(y_np, dtype=torch.float32).unsqueeze(1)

    model = nn.Linear(in_features=1, out_features=1)
    with torch.no_grad():  # 和极简版用同样的起点，方便对比
        model.weight.fill_(-1.0)
        model.bias.fill_(4.0)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    loss_fn = nn.MSELoss()

    for step in range(201):
        y_hat = model(x)              # 1. 前向：算预测
        loss = loss_fn(y_hat, y)      # 2. 损失
        if step in (0, 10, 50, 200):  # 打印的是这一步更新之前的参数和损失
            a, b = model.weight.item(), model.bias.item()
            print(f"第 {step:>3} 步：a = {a:.3f}, b = {b:.3f}, 损失 = {loss.item():.4f}")
        optimizer.zero_grad()         # 3. 清掉上一步的梯度
        loss.backward()               # 4. 反向：autograd 自动算 ∂L/∂a、∂L/∂b
        optimizer.step()              # 5. 更新：a ← a − η·∂L/∂a

    # 对拍：和手算梯度的极简版逐步一致
    a_ref, b_ref, _ = fit_line.gradient_descent(x_np, y_np, lr=0.05, steps=201)[-1]
    a, b = model.weight.item(), model.bias.item()
    print(f"\n201 步更新后：PyTorch a = {a:.3f}, b = {b:.3f}；手算梯度 a = {a_ref:.3f}, b = {b_ref:.3f}")


if __name__ == "__main__":
    main()
