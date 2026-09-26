"""第 4 章 · 极简代码 1：约 150 行的标量自动微分引擎（autograd）+ 一个小 MLP

思路来自 Andrej Karpathy 的 micrograd（https://github.com/karpathy/micrograd，MIT 许可），
这里是按本课的讲法重新写的版本。只用 Python 标准库。

核心只有两件事：
1. 前向：每做一次运算，就新建一个 Value 节点，记住"我是由谁、用什么运算算出来的"——计算图就这样被记录下来；
2. 反向：从损失出发，按拓扑序的逆序走一遍计算图，每个节点把自己的梯度乘上"局部导数"，
   用 += 累加到它的输入节点上（链式法则）。

运行：uv run python chapters/04-backprop-autograd/code/01_engine.py
"""

from __future__ import annotations

import math
import random


class Value:
    """一个标量节点：存数值 data、梯度 grad，以及"怎么把梯度传给输入"的 _backward。"""

    def __init__(self, data: float, _children: tuple = (), _op: str = ""):
        self.data = float(data)
        self.grad = 0.0                   # ∂L/∂(这个节点)，反向传播前是 0
        self._backward = lambda: None     # 叶子节点（输入、参数）没有东西要往回传
        self._prev = _children            # 计算图的边：这个节点由哪些节点算出来
        self._op = _op                    # 运算名，只用于打印和画图

    # ── 基本运算：每个运算 = 前向算数值 + 一个局部求导的 _backward ──────────────
    def __add__(self, other) -> Value:
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():                  # ∂(a+b)/∂a = 1，∂(a+b)/∂b = 1
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other) -> Value:
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():                  # ∂(a·b)/∂a = b，∂(a·b)/∂b = a
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __pow__(self, k: float) -> Value:
        assert isinstance(k, (int, float)), "只支持常数次幂"
        out = Value(self.data**k, (self,), f"**{k}")

        def _backward():                  # ∂(a^k)/∂a = k·a^(k−1)
            self.grad += k * self.data ** (k - 1) * out.grad
        out._backward = _backward
        return out

    def relu(self) -> Value:
        out = Value(max(0.0, self.data), (self,), "relu")

        def _backward():                  # 正数时导数 1，负数时导数 0
            self.grad += (self.data > 0) * out.grad
        out._backward = _backward
        return out

    def tanh(self) -> Value:
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")

        def _backward():                  # ∂tanh(a)/∂a = 1 − tanh²(a)
            self.grad += (1 - t * t) * out.grad
        out._backward = _backward
        return out

    def exp(self) -> Value:
        out = Value(math.exp(self.data), (self,), "exp")

        def _backward():                  # ∂e^a/∂a = e^a
            self.grad += out.data * out.grad
        out._backward = _backward
        return out

    def log(self) -> Value:
        out = Value(math.log(self.data), (self,), "log")

        def _backward():                  # ∂ln(a)/∂a = 1/a
            self.grad += (1 / self.data) * out.grad
        out._backward = _backward
        return out

    # ── 反向传播 ──────────────────────────────────────────────────────────────
    def backward(self) -> None:
        # 1. 拓扑排序：保证一个节点排在所有"用到它的节点"之前
        topo, visited = [], set()

        def build(v: Value) -> None:
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)
        build(self)
        # 2. 起点：∂L/∂L = 1；然后按拓扑序的逆序，每个节点把梯度传给它的输入
        self.grad = 1.0
        for v in reversed(topo):
            v._backward()

    # ── 其余运算都能用上面几个拼出来 ─────────────────────────────────────────────
    def __neg__(self) -> Value:                return self * -1
    def __sub__(self, other) -> Value:         return self + (-other)
    def __truediv__(self, other) -> Value:     return self * other**-1
    def __radd__(self, other) -> Value:        return self + other
    def __rsub__(self, other) -> Value:        return other + (-self)
    def __rmul__(self, other) -> Value:        return self * other
    def __rtruediv__(self, other) -> Value:    return other * self**-1

    def __repr__(self) -> str:
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"


# ── 用 Value 搭一个多层感知机（MLP），和第 3 章的结构一样 ──────────────────────────
class Neuron:
    """一个神经元：act(w·x + b)。"""

    def __init__(self, nin: int, act: str = "tanh"):
        s = nin**-0.5                     # 让初始输出的尺度不随输入个数变大
        self.w = [Value(random.uniform(-s, s)) for _ in range(nin)]
        self.b = Value(0.0)
        self.act = act

    def __call__(self, x: list) -> Value:
        z = sum((wi * xi for wi, xi in zip(self.w, x, strict=True)), self.b)
        return {"tanh": z.tanh, "relu": z.relu, "linear": lambda: z}[self.act]()

    def parameters(self) -> list[Value]:
        return self.w + [self.b]


class Layer:
    """一层 = 若干个并排的神经元，对应第 2 章的 y = xW + b。"""

    def __init__(self, nin: int, nout: int, act: str):
        self.neurons = [Neuron(nin, act) for _ in range(nout)]

    def __call__(self, x: list) -> list[Value]:
        return [n(x) for n in self.neurons]

    def parameters(self) -> list[Value]:
        return [p for n in self.neurons for p in n.parameters()]


class MLP:
    """例如 MLP(1, [16, 16, 1])：输入 1 维，两个 16 维隐藏层，输出 1 维（最后一层不加激活）。"""

    def __init__(self, nin: int, nouts: list[int], act: str = "tanh"):
        sizes = [nin] + nouts
        self.layers = [Layer(sizes[i], sizes[i + 1], act if i < len(nouts) - 1 else "linear")
                       for i in range(len(nouts))]

    def __call__(self, x: list) -> Value:
        for layer in self.layers:
            x = layer(x)
        return x[0] if len(x) == 1 else x

    def parameters(self) -> list[Value]:
        return [p for layer in self.layers for p in layer.parameters()]

    def zero_grad(self) -> None:
        for p in self.parameters():
            p.grad = 0.0


if __name__ == "__main__":
    # 一个小例子：L = (a·b + c)²，a=2, b=−3, c=10（视频里画的就是这张计算图）
    a, b, c = Value(2.0), Value(-3.0), Value(10.0)
    d = a * b          # d = −6
    e = d + c          # e = 4
    L = e**2           # L = 16
    L.backward()
    print("前向：d = a·b =", d.data, "  e = d + c =", e.data, "  L = e² =", L.data)
    print("反向：∂L/∂e =", e.grad, " ∂L/∂d =", d.grad,
          " ∂L/∂a =", a.grad, " ∂L/∂b =", b.grad, " ∂L/∂c =", c.grad)

    # 分叉（fan-out）：x 被用了两次，梯度要把两条路加起来
    x = Value(3.0)
    y = x * x + x      # dy/dx = 2x + 1 = 7
    y.backward()
    print("\n分叉：y = x·x + x，x = 3 → ∂y/∂x =", x.grad, "（手算 2x + 1 = 7）")

    random.seed(0)
    net = MLP(1, [16, 16, 1])
    print(f"\nMLP(1, [16, 16, 1]) 共有 {len(net.parameters())} 个参数，每个都是一个 Value")
