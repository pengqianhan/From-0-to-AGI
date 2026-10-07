"""Chapter 4 · Minimal code 1: a scalar autograd engine in about 150 lines, and a small MLP.

The idea comes from micrograd by Andrej Karpathy (https://github.com/karpathy/micrograd, MIT license).
This version is rewritten to match the method of this course. It uses only the Python standard library.

The core has two parts:
1. Forward pass: each operation makes a new Value node that records its inputs and its operation (the graph).
2. Backward pass: go from the loss through the graph in reverse topological order. Each node multiplies its
   gradient by the local derivative and adds (+=) the result to the gradients of its inputs (the chain rule).

Run: uv run python chapters/04-backprop-autograd/code/01_engine.py
"""

from __future__ import annotations

import math
import random


class Value:
    """A scalar node. It keeps the value `data`, the gradient `grad`, and `_backward`, which sends the gradient to the inputs."""

    def __init__(self, data: float, _children: tuple = (), _op: str = ""):
        self.data = float(data)
        self.grad = 0.0                   # ∂L/∂(this node); 0 before the backward pass
        self._backward = lambda: None     # a leaf node (input, parameter) has nothing to send back
        self._prev = _children            # graph edges: the nodes that this node comes from
        self._op = _op                    # operation name, only for printing and drawing

    # ── Basic operations: each one = a forward value + a _backward with the local derivative ──
    def __add__(self, other) -> Value:
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():                  # ∂(a+b)/∂a = 1, ∂(a+b)/∂b = 1
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other) -> Value:
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():                  # ∂(a·b)/∂a = b, ∂(a·b)/∂b = a
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __pow__(self, k: float) -> Value:
        assert isinstance(k, (int, float)), "the exponent must be a constant number"
        out = Value(self.data**k, (self,), f"**{k}")

        def _backward():                  # ∂(a^k)/∂a = k·a^(k−1)
            self.grad += k * self.data ** (k - 1) * out.grad
        out._backward = _backward
        return out

    def relu(self) -> Value:
        out = Value(max(0.0, self.data), (self,), "relu")

        def _backward():                  # derivative 1 for a positive input, 0 for a negative input
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

    # ── Backward pass ─────────────────────────────────────────────────────────
    def backward(self) -> None:
        # 1. Topological sort: each node comes before all nodes that use it.
        topo, visited = [], set()

        def build(v: Value) -> None:
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)
        build(self)
        # 2. Start with ∂L/∂L = 1. Then, in reverse topological order, each node sends its gradient to its inputs.
        self.grad = 1.0
        for v in reversed(topo):
            v._backward()

    # ── The other operations are combinations of the operations above ──────────
    def __neg__(self) -> Value:                return self * -1
    def __sub__(self, other) -> Value:         return self + (-other)
    def __truediv__(self, other) -> Value:     return self * other**-1
    def __radd__(self, other) -> Value:        return self + other
    def __rsub__(self, other) -> Value:        return other + (-self)
    def __rmul__(self, other) -> Value:        return self * other
    def __rtruediv__(self, other) -> Value:    return other * self**-1

    def __repr__(self) -> str:
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"


# ── A multilayer perceptron (MLP) made of Value nodes, with the same structure as in Chapter 3 ──
class Neuron:
    """One neuron: act(w·x + b)."""

    def __init__(self, nin: int, act: str = "tanh"):
        s = nin**-0.5                     # the scale of the initial output does not grow with the number of inputs
        self.w = [Value(random.uniform(-s, s)) for _ in range(nin)]
        self.b = Value(0.0)
        self.act = act

    def __call__(self, x: list) -> Value:
        z = sum((wi * xi for wi, xi in zip(self.w, x, strict=True)), self.b)
        return {"tanh": z.tanh, "relu": z.relu, "linear": lambda: z}[self.act]()

    def parameters(self) -> list[Value]:
        return self.w + [self.b]


class Layer:
    """One layer = some neurons side by side. It is the y = xW + b of Chapter 2."""

    def __init__(self, nin: int, nout: int, act: str):
        self.neurons = [Neuron(nin, act) for _ in range(nout)]

    def __call__(self, x: list) -> list[Value]:
        return [n(x) for n in self.neurons]

    def parameters(self) -> list[Value]:
        return [p for n in self.neurons for p in n.parameters()]


class MLP:
    """Example: MLP(1, [16, 16, 1]) has a 1-D input, two 16-D hidden layers, and a 1-D output (no activation in the last layer)."""

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
    # A small example: L = (a·b + c)², a=2, b=−3, c=10 (the video draws this computational graph)
    a, b, c = Value(2.0), Value(-3.0), Value(10.0)
    d = a * b          # d = −6
    e = d + c          # e = 4
    L = e**2           # L = 16
    L.backward()
    print("Forward:  d = a·b =", d.data, "  e = d + c =", e.data, "  L = e² =", L.data)
    print("Backward: ∂L/∂e =", e.grad, " ∂L/∂d =", d.grad,
          " ∂L/∂a =", a.grad, " ∂L/∂b =", b.grad, " ∂L/∂c =", c.grad)

    # Fan-out: y uses x more than one time, so the gradient of x is the sum over all paths
    x = Value(3.0)
    y = x * x + x      # dy/dx = 2x + 1 = 7
    y.backward()
    print("\nFan-out: y = x·x + x, x = 3 → ∂y/∂x =", x.grad, "(by hand: 2x + 1 = 7)")

    random.seed(0)
    net = MLP(1, [8, 8, 1])
    print(f"\nMLP(1, [8, 8, 1]) has {len(net.parameters())} parameters. Each parameter is a Value.")
