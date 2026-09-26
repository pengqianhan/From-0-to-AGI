"""第 5 章 · 极简代码 3：用 MLP + softmax + 交叉熵给三类螺旋数据分类

数据在代码里生成（三条交错的螺旋臂，经典的 CS231n 玩具数据），不需要下载。
模型：2 → 64（ReLU）→ 3 的两层 MLP（第 3 章），反向传播手写（第 4 章），全批量梯度下降。
同样的网络、同样的学习率，分别用交叉熵和 MSE 训练，对比结果。
只用 NumPy，CPU 上几秒跑完。
运行：uv run python chapters/05-classification-probability/code/03_train_classifier.py
"""

import importlib.util
from pathlib import Path

import numpy as np

_spec = importlib.util.spec_from_file_location("ce_mod", Path(__file__).with_name("02_cross_entropy.py"))
ce = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ce)


def make_spirals(n_per_class: int = 100, n_classes: int = 3, noise: float = 0.2, seed: int = 0):
    """三条螺旋臂：第 k 类的点沿半径 r ∈ [0, 1] 向外转，角度起点错开 4 弧度。"""
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for k in range(n_classes):
        r = np.linspace(0.0, 1.0, n_per_class)
        t = np.linspace(k * 4.0, (k + 1) * 4.0, n_per_class) + rng.normal(0, noise, n_per_class)
        xs.append(np.stack([r * np.sin(t), r * np.cos(t)], axis=1))
        ys.append(np.full(n_per_class, k))
    return np.concatenate(xs), np.concatenate(ys)


def init_params(hidden: int = 64, n_in: int = 2, n_out: int = 3, seed: int = 0,
                out_std: float = 0.01) -> dict:
    """out_std：输出层权重的标准差。默认很小 → 初始 logits ≈ 0 → 初始预测≈均匀分布。
    调大它（比如 10），模型一开始就会给出很"自信"的随机预测——用来演示 MSE 的问题。
    """
    rng = np.random.default_rng(seed)
    return {
        "W1": rng.normal(0, np.sqrt(2 / n_in), (n_in, hidden)), "b1": np.zeros(hidden),
        "W2": rng.normal(0, out_std, (hidden, n_out)), "b2": np.zeros(n_out),
    }


def forward(params: dict, X: np.ndarray):
    h_pre = X @ params["W1"] + params["b1"]
    h = np.maximum(h_pre, 0)                 # ReLU
    logits = h @ params["W2"] + params["b2"]  # z：每个类别一个分数
    return logits, (X, h_pre, h)


def backward(params: dict, cache, dlogits: np.ndarray) -> dict:
    X, h_pre, h = cache
    dh = dlogits @ params["W2"].T
    dh_pre = dh * (h_pre > 0)
    return {"W1": X.T @ dh_pre, "b1": dh_pre.sum(0), "W2": h.T @ dlogits, "b2": dlogits.sum(0)}


def accuracy(params: dict, X: np.ndarray, y: np.ndarray) -> float:
    return float((forward(params, X)[0].argmax(1) == y).mean())


def train(X, y, loss: str = "ce", lr: float = 1.0, steps: int = 3000, hidden: int = 64,
          snapshot_steps=(), seed: int = 0, out_std: float = 0.01):
    """全批量梯度下降。loss="ce" 用交叉熵，"mse" 用 softmax 概率对 onehot 的均方误差。
    返回 (最终参数, 日志 [(步数, 交叉熵, 准确率)], {步数: 参数快照})。
    """
    params = init_params(hidden, seed=seed, out_std=out_std)
    grad_fn = ce.ce_grad if loss == "ce" else ce.mse_grad
    log, snaps = [], {}
    for step in range(steps + 1):
        logits, cache = forward(params, X)
        if step % 100 == 0 or step in snapshot_steps:
            log.append((step, ce.cross_entropy(logits, y), accuracy(params, X, y)))
        if step in snapshot_steps:
            snaps[step] = {k: v.copy() for k, v in params.items()}
        if step == steps:
            break
        grads = backward(params, cache, grad_fn(logits, y))  # dlogits = (p − onehot)/N
        for k in params:
            params[k] -= lr * grads[k]
    return params, log, snaps


def train_linear(X, y, lr: float = 1.0, steps: int = 3000):
    """对照：没有隐藏层的 softmax 回归（logits = XW + b），只能画直线边界。返回 (W, b)。"""
    W, b = np.zeros((2, 3)), np.zeros(3)
    for _ in range(steps):
        g = ce.ce_grad(X @ W + b, y)
        W -= lr * X.T @ g
        b -= lr * g.sum(0)
    return W, b


if __name__ == "__main__":
    X, y = make_spirals()
    print(f"数据：{len(X)} 个点，{X.shape[1]} 维，{y.max() + 1} 类（每类 {np.sum(y == 0)} 个）")
    print(f"初始损失应当约等于 ln 3 = {np.log(3):.4f}（什么都不懂时均匀猜）\n")

    params, log, _ = train(X, y, loss="ce")
    print("交叉熵训练（lr = 1.0，全批量）")
    print("  步数    交叉熵   训练准确率")
    for step, loss, acc in log:
        if step in (0, 100, 200, 500, 1000, 2000, 3000):
            print(f"{step:>6}   {loss:7.4f}    {acc:6.1%}")

    Xt, yt = make_spirals(seed=1)  # 同一分布、不同随机种子：另造一份测试集
    print(f"测试集（另一份螺旋数据）准确率：{accuracy(params, Xt, yt):.1%}")

    print("\n对照 1：没有隐藏层的线性 softmax 分类器")
    W_lin, b_lin = train_linear(X, y)
    acc_lin = ((X @ W_lin + b_lin).argmax(1) == y).mean()
    print(f"  3000 步后训练准确率：{acc_lin:.1%}（直线边界切不开螺旋）")

    for title, out_std in [("对照 2：正常初始化（初始预测≈均匀）", 0.01),
                           ("对照 3：输出层权重标准差放大到 10（一开始就自信地乱猜）", 10.0)]:
        print(f"\n{title}，同一个 MLP、同样 lr = 1.0，交叉熵 vs MSE（softmax 概率 vs onehot）")
        print("  步数   CE 版交叉熵  CE 版准确率 | MSE 版交叉熵 MSE 版准确率")
        _, log_ce, _ = train(X, y, loss="ce", out_std=out_std)
        _, log_mse, _ = train(X, y, loss="mse", out_std=out_std)
        for (s, l1, a1), (_, l2, a2) in zip(log_ce, log_mse, strict=True):
            if s in (0, 100, 500, 800, 1000, 3000):
                print(f"{s:>6}   {l1:9.4f}   {a1:9.1%}  | {l2:9.4f}   {a2:9.1%}")
    # 卡住的原因：训练 500 步后，有多少样本仍然"自信地错"（给正确类别的概率 < 1%）？
    for loss in ["ce", "mse"]:
        p500, _, _ = train(X, y, loss=loss, steps=500, out_std=10.0)
        p_true = ce.softmax(forward(p500, X)[0])[np.arange(len(y)), y]
        print(f"对照 3 · {loss.upper():>3} 训练 500 步后：p(正确) < 1% 的样本 {np.sum(p_true < 0.01)} 个")
