"""第 5 章 · 从极简到生产级：F.cross_entropy 与 PyTorch 标准训练循环

1. 对拍：torch.nn.functional.cross_entropy(logits, y) 和我们的 NumPy 版逐位一致。
   注意它吃的是 logits（不是概率）：内部把 log-softmax 和负对数似然融合在一起，数值稳定。
2. 数值稳定：logits = [1000, 500, −500] 时，先 softmax 再取 log 会得到 inf，F.cross_entropy 不会。
3. label_smoothing：把 onehot 目标"抹平"一点，和手写公式对拍。
4. 用 nn.Sequential + torch.optim.SGD 训练同一个螺旋分类器，和 NumPy 手写反向传播的结果对拍。
5. 语言模型里的写法：logits 形状 (B, T, V)，拉平成 (B·T, V) 再算交叉熵。
运行：uv run python chapters/05-classification-probability/code/05_pytorch_version.py
"""

import importlib.util
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ce = _load("ce_mod", "02_cross_entropy.py")
clf = _load("clf_mod", "03_train_classifier.py")


def main() -> None:
    torch.manual_seed(0)
    torch.set_num_threads(1)  # 张量很小，单线程反而更快（多线程的调度开销比计算本身还大）

    print("—— 1. F.cross_entropy 对拍 NumPy ——")
    rng = np.random.default_rng(0)
    z_np, y_np = rng.normal(0, 3, size=(8, 5)), rng.integers(0, 5, size=8)
    z, y = torch.tensor(z_np, requires_grad=True), torch.tensor(y_np)
    loss = F.cross_entropy(z, y)  # 输入 logits，不是 softmax 之后的概率
    loss.backward()
    print(f"PyTorch：{loss.item():.10f}   NumPy：{ce.cross_entropy(z_np, y_np):.10f}")
    print(f"autograd 梯度 vs 手写 (p − onehot)/N：最大差 "
          f"{np.abs(z.grad.numpy() - ce.ce_grad(z_np, y_np)).max():.2e}")
    same = F.nll_loss(F.log_softmax(z, dim=-1), y).item()
    print(f"等价写法 nll_loss(log_softmax(z))：{same:.10f}")

    print("\n—— 2. 数值稳定 ——")
    zb = torch.tensor([[1000.0, 500.0, -500.0]])
    yb = torch.tensor([2])
    naive = -torch.log(torch.exp(zb[0, 2]) / torch.exp(zb).sum())
    print(f"朴素写法 −log(exp(z_y)/Σexp(z))：{naive.item()}")
    print(f"F.cross_entropy（融合 log-softmax）：{F.cross_entropy(zb, yb).item()}")

    print("\n—— 3. label_smoothing = 0.1 ——")
    eps, K = 0.1, z_np.shape[1]
    ls = F.cross_entropy(z.detach(), y, label_smoothing=eps).item()
    logp = ce.log_softmax(z_np)
    target = np.full_like(logp, eps / K)            # 每类先分到 ε/K
    target[np.arange(len(y_np)), y_np] += 1 - eps   # 正确类别再加 1 − ε
    manual = float(-(target * logp).sum(axis=1).mean())
    print(f"PyTorch：{ls:.10f}   手写 −Σ q_k log p_k（q = 抹平后的目标）：{manual:.10f}")

    print("\n—— 4. 用 PyTorch 训练螺旋分类器，对拍 NumPy 手写版 ——")
    X_np, Y_np = clf.make_spirals()
    X, Y = torch.tensor(X_np), torch.tensor(Y_np)
    model = nn.Sequential(nn.Linear(2, 64), nn.ReLU(), nn.Linear(64, 3)).double()
    p0 = clf.init_params()  # 和 NumPy 版用同一组初始参数
    with torch.no_grad():
        model[0].weight.copy_(torch.tensor(p0["W1"].T))
        model[0].bias.copy_(torch.tensor(p0["b1"]))
        model[2].weight.copy_(torch.tensor(p0["W2"].T))
        model[2].bias.copy_(torch.tensor(p0["b2"]))
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)

    for step in range(3001):
        logits = model(X)                    # 1. 前向：得到 logits
        loss = F.cross_entropy(logits, Y)    # 2. 交叉熵（内部做 log-softmax）
        if step in (0, 500, 3000):
            acc = (logits.argmax(1) == Y).double().mean().item()
            print(f"第 {step:>4} 步：交叉熵 = {loss.item():.4f}，准确率 = {acc:.1%}")
        if step == 3000:
            final_loss = loss.item()
            break
        optimizer.zero_grad()                # 3. 清梯度
        loss.backward()                      # 4. 反向：autograd 自动算出 (p − onehot)/N 并往回传
        optimizer.step()                     # 5. 更新

    _, log_np, _ = clf.train(X_np, Y_np, loss="ce", steps=3000)
    print(f"NumPy 手写版第 3000 步：交叉熵 = {log_np[-1][1]:.4f}，准确率 = {log_np[-1][2]:.1%}")
    print(f"两者第 3000 步的交叉熵之差：{abs(final_loss - log_np[-1][1]):.2e}")

    print("\n—— 5. 语言模型里的写法 ——")
    B, T, V = 2, 4, 50257  # 2 条序列 × 每条 4 个位置 × GPT-2 词表大小
    logits = torch.zeros(B, T, V)  # 全 0 = 均匀乱猜
    targets = torch.randint(0, V, (B, T))
    lm_loss = F.cross_entropy(logits.view(-1, V), targets.view(-1))
    print(f"logits {tuple(logits.shape)} → 拉平成 {(B * T, V)}；损失 = {lm_loss.item():.4f}，"
          f"ln V = {np.log(V):.4f}")


if __name__ == "__main__":
    main()
