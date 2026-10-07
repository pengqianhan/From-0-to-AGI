# 第 4 章：反向传播与自动微分 —— 让计算机替你求导

[English](README.md) · **中文**

> **目标**：读完这一章，你能手写一个约 150 行的标量自动微分引擎。你能用它训练一个多层感知机。你还能用数值梯度和 PyTorch 证明：引擎算出的每一个梯度都是对的。

📺 **本章视频**：还没有发布。要在本机渲染，运行 `bash chapters/04-backprop-autograd/video/build.sh`。
🧪 **本章自检**：学完后，在 Claude Code 里输入 `/ch04-backprop`。

---

上一章我们用两层 MLP 拟合了 `y = sin(2x)`。为了训练它，我们手工推导了 6 行梯度公式。推导从损失往回走，穿过第二层、穿过 ReLU，再到第一层。最后还要用数值梯度检查一遍，才能信任这些公式。这一章要解决的问题是：**网络一变，梯度就要重新推导。能不能让计算机自己算出梯度？**

答案是**自动微分**（automatic differentiation，autograd）。上一章的算法从后往前、逐层应用链式法则。这个算法叫**反向传播**（backpropagation）。它是自动微分在神经网络里的具体形式。第 1 章里 PyTorch 有一行 `loss.backward()`。这一章你要亲手写出这一行背后的机制。

## 1. 问题：手工推导梯度不能扩展

回头看我们推导过的梯度：

- **第 1 章**：`ŷ = a·x + b`，两个参数，两行公式：`∂L/∂a = 2/N · Σ(ŷ_i − y_i)·x_i`，`∂L/∂b = 2/N · Σ(ŷ_i − y_i)`。
- **第 3 章**：两层网络 `A = ReLU(X·W1 + b1)`，`Ŷ = A·W2 + b2`。我们手工推导了 6 行。合成一个公式，第一层权重的梯度是：

```
∂L/∂W1 = Xᵀ · [ (2/N · (Ŷ − Y) · W2ᵀ) ⊙ 1(X·W1 + b1 > 0) ]
```

每加一层，公式就多套一层。把 ReLU 换成 tanh，`1(· > 0)` 这一项要重新推导。把均方误差换成下一章的交叉熵，最外面的一项要重新推导。更严重的问题是：**推导错了很难发现**。公式里写错一个转置，代码照样能运行，只是训练没有进展。

课程最后要训练的大语言模型有几十亿个参数、几十层、几百种不同的运算。手工推导是不可能的。

出路来自一个观察：**不管网络多复杂，它都由少数几种简单运算组合而成。**这些运算是加法、乘法、幂、tanh、指数、对数等。每种简单运算的导数我们都会求。如果计算机也能自动求出组合后的导数，问题就解决了。负责"组合"的数学工具就是链式法则。

## 2. 链式法则：沿路径把局部导数相乘

设想一组齿轮。`u = 3x`：x 转一点，u 转 3 倍。`y = u²`：在 x = 2、u = 6 的位置，u 转一点，y 转 `2u = 12` 倍。那么 x 转一点，y 转多少？3 × 12 = **36 倍**。

这就是**链式法则**（chain rule）：

```
dy/dx = dy/du · du/dx
```

用代码验证（`01_engine.py` 里的 `Value` 类马上会讲到）：

```python
x = Value(2.0); u = x * 3; y = u**2
y.backward()
x.grad   # 36.0
```

链式法则的关键是：**每一段只需要知道自己的一小步。**`u = 3x` 这一段只知道"我把变化放大 3 倍"。它不需要知道后面接的是平方还是别的运算。这种只看一步的导数叫**局部导数**（local derivative）。

一个变量可以通过**多条路径**影响输出。这时要把每条路径的贡献**加起来**（多元链式法则）。在第 5 节，这条规则会变成代码里一个关键的 `+=`。

## 3. 计算图：前向传播时记下每一步

把一个式子拆成最小的运算，把每个中间结果作为一个节点，就得到一张**计算图**（computational graph）。以本章的例子为例：

```
L = (a·b + c)², a = 2, b = −3, c = 10
```

拆开是三步：`d = a·b`，`e = d + c`，`L = e²`。

```
a ──┐
    (×)──→ d ──┐
b ──┘          (+)──→ e ──→ (²)──→ L
c ─────────────┘
```

**前向传播**（forward pass）时，数值从左往右流：`d = −6`，`e = 4`，`L = 16`。

我们的引擎在前向传播时多做一件事：**每做一次运算，就新建一个节点，记下它的输入节点和运算。**算完一遍，整张图就记录下来了。下面是 `Value` 类的全部数据：

```python
class Value:
    def __init__(self, data, _children=(), _op=""):
        self.data = float(data)
        self.grad = 0.0                   # ∂L/∂(this node); 0 before the backward pass
        self._backward = lambda: None     # sends the gradient to the inputs (a leaf node has none)
        self._prev = _children            # graph edges: the nodes that this node comes from
        self._op = _op
```

## 4. 反向传播：上游梯度 × 局部导数

现在要算 L 对每个节点的梯度。我们记 `v̄ = ∂L/∂v`（读作 "v bar"），从右往左走：

| 节点 | 局部导数 | 梯度 = 上游梯度 × 局部导数 |
|---|---|---|
| L | — | `L̄ = 1`（L 对自己的导数） |
| e（L = e²） | `∂L/∂e = 2e = 8` | `ē = 1 × 8 = 8` |
| d、c（e = d + c） | `∂e/∂d = 1`，`∂e/∂c = 1` | `d̄ = 8`，`c̄ = 8` |
| a、b（d = a·b） | `∂d/∂a = b = −3`，`∂d/∂b = a = 2` | `ā = 8 × (−3) = −24`，`b̄ = 8 × 2 = 16` |

运行 `uv run python chapters/04-backprop-autograd/code/01_engine.py`，输出完全一致：

```
Forward:  d = a·b = -6.0   e = d + c = 4.0   L = e² = 16.0
Backward: ∂L/∂e = 8.0  ∂L/∂d = 8.0  ∂L/∂a = -24.0  ∂L/∂b = 16.0  ∂L/∂c = 8.0
```

每一步做的都是**同一个动作**：拿到上游传来的梯度，乘以本节点的局部导数，再传给输入。加法节点的局部导数是 1，所以它把梯度**原样传给**两个输入。乘法节点的局部导数是**另一个输入的值**，所以 `ā = d̄ · b`。

这就是反向传播的全部数学。剩下的都是工程：怎样把它写成代码。

## 5. 每种运算只写一次：`_backward`

自动微分的秘密是：**每种运算的局部导数只需要写一次。**以乘法为例：

```python
def __mul__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data * other.data, (self, other), "*")   # forward: calculate the value, record the two inputs

    def _backward():                  # ∂(a·b)/∂a = b, ∂(a·b)/∂b = a
        self.grad += other.data * out.grad
        other.grad += self.data * out.grad
    out._backward = _backward
    return out
```

`out.grad` 就是"上游梯度"，`other.data` 就是"局部导数"。我们的引擎实现了下表中的运算。每种运算都是"前向一行 + 局部导数一行"：

| 运算 | 前向 | 局部导数 | 代码里的那一行 |
|---|---|---|---|
| 加法 | `c = a + b` | `∂c/∂a = 1`，`∂c/∂b = 1` | `self.grad += out.grad` |
| 乘法 | `c = a·b` | `∂c/∂a = b`，`∂c/∂b = a` | `self.grad += other.data * out.grad` |
| 常数次幂 | `c = aᵏ` | `k·aᵏ⁻¹` | `self.grad += k * self.data ** (k - 1) * out.grad` |
| ReLU | `c = max(0, a)` | a > 0 时为 1，否则为 0 | `self.grad += (self.data > 0) * out.grad` |
| tanh | `c = tanh a` | `1 − c²` | `self.grad += (1 - t * t) * out.grad` |
| 指数 | `c = eᵃ` | `c` | `self.grad += out.data * out.grad` |
| 对数 | `c = ln a` | `1/a` | `self.grad += (1 / self.data) * out.grad` |

减法、除法、取负不需要单独写导数。它们可以用上面几种运算组合出来，例如 `a − b = a + (b × −1)`，`a / b = a × b⁻¹`：

```python
def __sub__(self, other):      return self + (-other)
def __truediv__(self, other):  return self * other**-1
```

## 6. 分叉：梯度为什么要用 `+=`

上面每一行写的都是 `+=`，不是 `=`。为什么？

看这个例子：`y = x·x + x`，x = 3。图里**用了 x 三次**：乘法的两个输入各一次，加法一次。

```
x ──┬──→ (×) ──→ m = x·x ──┐
    ├──→ (×)                (+)──→ y
    └───────────────────────┘
```

反向传播时，三条路径各带回一份梯度。乘法的两个输入各带回 `x × 1 = 3`，加法这条路径带回 `1`。x 的真实梯度是它们的**和**：`3 + 3 + 1 = 7`。这正好等于手算的 `dy/dx = 2x + 1 = 7`。`01_engine.py` 的输出：

```
Fan-out: y = x·x + x, x = 3 → ∂y/∂x = 7.0 (by hand: 2x + 1 = 7)
```

如果写成 `=`，后到的梯度会**覆盖**先到的梯度，只剩最后一条路径的贡献。这个问题在实际中会出现。真实网络里到处都有分叉：第一层的每个神经元都用到同一个输入，下一层的每个神经元都用到一个隐藏单元的输出。在第 8 节的梯度检验里，我们会故意把 `+=` 改成 `=`，看看会产生什么错误。

这也回答了第 1 章留下的一个问题：**PyTorch 为什么默认把梯度累加起来，每一步都要先调用 `optimizer.zero_grad()`？**因为反向传播本来就要靠累加才能正确工作。框架无法区分"同一次反向传播里另一条路径传来的梯度"和"上一步留下来的梯度"。所以训练循环在每次反向传播前都要把梯度清零。在我们的 MLP 里，这一步是 `net.zero_grad()`。

## 7. 拓扑排序：反向传播时的顺序

最后一个问题：反向传播时，按什么顺序处理节点？

规则是：**一个节点必须先收齐下游传回的全部梯度，才能把梯度传给上游。**在 `y = x·x + x` 里，假设 x 只收到加法这条路径的 1，就把梯度传给上游。那么它的梯度就是错的。

做法是先对计算图做一次**拓扑排序**（topological sort）。排序保证每个节点都排在所有用到它的节点之前。然后**倒过来**走一遍。处理一个节点时，所有用到它的节点都已经处理完，梯度也都已经加进来。整个 `backward` 只有这几行：

```python
def backward(self):
    topo, visited = [], set()
    def build(v):
        if v not in visited:
            visited.add(v)
            for child in v._prev:
                build(child)
            topo.append(v)            # add a node only after all its inputs
    build(self)
    self.grad = 1.0                   # start: ∂L/∂L = 1
    for v in reversed(topo):          # reverse topological order: collect all gradients, then send them upstream
        v._backward()
```

到这里，引擎就完整了。它包括 `Value` 类和三个小类：`Neuron`、`Layer`、`MLP`。它们的结构和第 3 章一样：一个神经元计算 `act(w·x + b)`；一层是若干个并排的神经元，对应第 2 章的 `y = xW + b`。整个文件 194 行，去掉空行和注释约 125 行。完整代码见 [`code/01_engine.py`](code/01_engine.py)。

> 思路来自 Andrej Karpathy 的 [micrograd](https://github.com/karpathy/micrograd)（MIT 许可）。本章代码按本课的讲法重新编写，接口和 micrograd 类似。强烈推荐看他的配套视频。他用两个半小时从零写出这个引擎。

## 8. 梯度检验：怎样知道梯度算得对

第 1 章的任务 2 介绍过**梯度检验**（gradient check）：把参数往上移一点、往下移一点，看损失变化多少：

```
∂L/∂p ≈ (L(p + ε) − L(p − ε)) / (2ε)
```

这个**数值梯度**又慢又不够精确。但它**不会有推导错误**，因为它只用到前向计算。把它和 autograd 的结果比较，是检验反向传播最可靠的方法。

```bash
uv run python chapters/04-backprop-autograd/code/02_grad_check.py
```

`02_grad_check.py` 先做两组数值检验（ε = 10⁻⁶）。每组运行两次：一次用正确的引擎，一次用故意改坏的引擎。相对误差按整个梯度向量计算：`‖g_auto − g_num‖ / (‖g_auto‖ + ‖g_num‖)`。

| 检验对象 | 引擎 | 最大绝对误差 | 相对误差 |
|---|---|---:|---:|
| 一个用到全部运算的表达式（3 个输入） | 正确（`+=`） | 8.7 × 10⁻¹¹ | 1.8 × 10⁻¹¹ |
| MLP(1, [8, 8, 1]) 的损失（97 个参数） | 正确（`+=`） | 2.0 × 10⁻¹⁰ | 1.8 × 10⁻¹⁰ |
| 同一个表达式 | 有 bug（`=`） | 2.6 | 0.97 |
| 同一个 MLP | 有 bug（`=`） | 0.66 | 0.93 |

这个表达式是 `((x·y + eᶻ).ln() − x/z).tanh() · x + relu(y²) + (x − 3)³/10`。三个输入的梯度逐位一致：

```
  [expression]  autograd: +2.452530, -2.590708, +0.123690
                numerical: +2.452530, -2.590708, +0.123690
```

最后，`02_grad_check.py` 用 `Value` 重建了第 3 章的 ReLU 网络（宽 8、同一组初始参数、同样的 100 个点）。它把 autograd 的梯度和第 3 章手工推导的 6 行公式对拍：

```
Parity check with the hand-derived gradients of Chapter 3 (ReLU MLP, width 8, 100 points):
  W1  shape (1, 8)  max difference 4.2e-17
  b1  shape (8,)    max difference 6.9e-17
  W2  shape (8, 1)  max difference 8.9e-16
  b2  shape (1,)    max difference 1.1e-16
```

上一章手工推导的 `d_W1 = x.T @ d_z` 等一串公式，现在一行 `backward()` 就能得到。两者的差别只有浮点舍入误差。

读法：正确引擎的相对误差在 10⁻¹⁰ 量级。这是中心差分本身的截断误差和舍入误差，说明两者一致。**有 bug 的引擎相对误差接近 1，说明梯度基本是错的。**请注意一个有意思的现象：有 bug 的代码照样能运行，MLP 照样能"训练"。如果不做梯度检验，你可能只会觉得"这个网络怎么学不好"。

以后给引擎加任何新运算，都要先通过梯度检验。PyTorch 也自带同样的工具，见本章末的"从极简代码到生产级代码"。

## 9. 用引擎训练 MLP

现在用我们自己的引擎从零训练一个网络。第 3 章的网络要训练 20000 步。我们的标量引擎很慢（第 10 节讲原因），所以换一个小一点的任务：拟合 `y = sin(x)`，x 在 [−3, 3] 上均匀取 20 个点。网络是 `MLP(1, [8, 8, 1])`：输入 1 维，**两个**隐藏层，每层 8 个神经元，激活函数是 tanh，共 97 个参数。和第 3 章相比，它多了一层，也换了激活函数。如果手工推导，这两处都要重新推导。用我们的引擎，一个公式都不用推导。

训练循环和第 1 章一样，只是求梯度的那一步换成了 `loss.backward()`：

```python
for step in range(steps + 1):
    loss = mse(net, xs, ys)           # 1. forward pass: also records the full computational graph
    net.zero_grad()                   # 2. set the gradients to zero (backward uses +=)
    loss.backward()                   # 3. backward pass: calculates the gradients of all parameters
    for p in net.parameters():        # 4. update: p ← p − η · ∂L/∂p
        p.data -= lr * p.grad
```

（完整代码里，最后一步只评估、不更新，见 [`code/03_train_mlp.py`](code/03_train_mlp.py)。）运行：

```bash
uv run python chapters/04-backprop-autograd/code/03_train_mlp.py
```

学习率 0.1，全批量梯度下降，500 步：

| 步数 | 0 | 1 | 10 | 50 | 100 | 200 | 300 | 400 | 500 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 损失 | 0.7163 | 0.4323 | 0.1111 | 0.0837 | 0.0478 | 0.0189 | 0.0145 | 0.0115 | 0.0094 |

训练后的预测（每隔 4 个点取一个）：

| x | sin(x) | 预测 |
|---:|---:|---:|
| −3.00 | −0.141 | −0.323 |
| −1.74 | −0.986 | −0.872 |
| −0.47 | −0.456 | −0.560 |
| 0.79 | 0.710 | 0.803 |
| 2.05 | 0.886 | 0.761 |

模型已经学到了曲线的大致形状。在边缘处（x = −3），误差还比较大。多训练一些步、调整学习率都能改善，这一点留给引导问题。我们也试过把学习率提高到 0.2、0.3。这时损失会来回跳动，这就是第 1 章讲的"步子太大"。这里的重点是：**从头到尾，我们没有手工推导任何一个梯度**。

## 10. 代价：标量计算图很慢

`03_train_mlp.py` 开头打印了这个数字：

```
One forward pass (20 samples) records a computational graph with 3720 Value nodes
```

每个权重和输入的乘积、每一次加法，都是一个 Python 对象和一个闭包。这个小网络有 20 个样本、97 个参数，每一步就要建 3720 个节点。训练 500 步用了 12.3 秒（每步约 25 毫秒；这是一次实测，随机器负载变化）。对几十亿参数的模型，这种方法完全行不通。

真实框架做的是**同一件事，但每个节点是一整个张量**。一次矩阵乘法 `Y = X·W` 是计算图里的**一个**节点，它的 `_backward` 也是矩阵乘法。沿用第 2 章的形状规则，设上游梯度为 `G = ∂L/∂Y`（形状和 Y 相同），那么：

```
∂L/∂X = G · Wᵀ        ∂L/∂W = Xᵀ · G        ∂L/∂b = sum of G over the rows
```

这叫**向量-雅可比积**（vector-Jacobian product，VJP）。**雅可比矩阵**（Jacobian）是所有输出对所有输入的偏导数排成的矩阵。`Y` 对 `X` 的完整雅可比矩阵很大：`X` 是 4×3、`Y` 是 4×2 时，它就有 8 × 12 = 96 个数；在真实模型里，这是一个天文数字。但反向传播**从来不需要构造这个矩阵**。它只需要"上游梯度 × 雅可比矩阵"的结果，而这个结果正好是一次矩阵乘法。`04_pytorch_compare.py` 的最后一部分验证了：`torch.autograd.grad(Y, X, grad_outputs=G)` 等于 `G @ W.T`。

原理和我们的标量引擎相同。前向传播记录计算图。反向传播按逆拓扑序进行，把上游梯度乘以局部导数，在分叉处累加。唯一的区别是："乘以局部导数"从一次标量乘法变成了一次矩阵乘法，由高度优化的 C++/CUDA 代码完成。

> **为什么是"反向"？**训练时我们只有**一个**输出（损失），却有**很多**输入（参数）。从输出往回走一遍，就能得到损失对全部参数的梯度，代价只是前向传播的几倍。另一种方法是从输入往前推（前向模式自动微分）。前向模式每走一遍，只能得到对一个输入的导数。对几十亿个参数，前向模式要走几十亿遍。这就是所有深度学习框架都用反向模式的原因。

## 11. 小结

- **手工推导梯度不能扩展**：模型一变就要重新推导，推导错了还很难发现。
- **链式法则**：复合函数的导数 = 沿路径的局部导数相乘；多条路径的贡献相加。
- **计算图**：前向传播时，每做一次运算就记一个节点（值、输入、运算）。
- **反向传播**：从 `L̄ = 1` 出发，按逆拓扑序，每个节点用 `+=` 把"上游梯度 × 局部导数"加到输入上。
- **自动微分**：每种运算的局部导数只写一次，任意组合都能自动求出梯度。
- **梯度检验**：用数值梯度对拍，是发现反向传播 bug 最可靠的方法。
- **代价**：标量计算图太慢；真实框架在张量上做同样的事（VJP）。

---

## 从极简代码到生产级代码

用 PyTorch 写同一个 MLP，就是标准的 `nn.Linear` + `torch.autograd`。[`code/04_pytorch_compare.py`](code/04_pytorch_compare.py) 把 Value 版 MLP 的初始权重原样复制到 `nn.Sequential` 里，然后逐项对拍：

```bash
uv run python chapters/04-backprop-autograd/code/04_pytorch_compare.py
```

```python
model = nn.Sequential(nn.Linear(1, 8), nn.Tanh(), nn.Linear(8, 8), nn.Tanh(), nn.Linear(8, 1))
loss = ((model(x) - y) ** 2).mean()   # x has shape (20, 1): one pass for the full batch of samples
model.zero_grad()
loss.backward()                        # torch.autograd: backpropagation on tensors
```

**1. 逐个参数对拍梯度**（两边都用双精度）：

| 参数 | 形状 | allclose | 最大差 |
|---|---|---|---:|
| `0.weight` | (8, 1) | True | 8.3 × 10⁻¹⁷ |
| `0.bias` | (8,) | True | 3.1 × 10⁻¹⁷ |
| `2.weight` | (8, 8) | True | 5.6 × 10⁻¹⁷ |
| `2.bias` | (8,) | True | 7.5 × 10⁻¹⁷ |
| `4.weight` | (1, 8) | True | 4.2 × 10⁻¹⁷ |
| `4.bias` | (1,) | True | 5.6 × 10⁻¹⁷ |

97 个参数的梯度最大差是 8.3 × 10⁻¹⁷，这是双精度浮点的舍入误差。两边的初始损失都是 0.716258950201。

**2. 对拍训练**：两边都用 SGD、学习率 0.1 训练 500 步。两边的最终损失都是 **0.0093662407**（小数点后 10 位一致）。

**3. 速度**（一次前向传播 + 一次反向传播，一次实测；数字随机器负载变化，但倍数的量级不变）：

| 样本数 | Value 引擎（标量图） | PyTorch（张量图） | 倍数 |
|---:|---:|---:|---:|
| 20 | 43.7 ms | 0.355 ms | 约 120× |
| 200 | 775.3 ms | 0.414 ms | 约 1900× |

样本数增加到 10 倍时，标量引擎的耗时至少也增加到 10 倍，因为节点数线性增长。这次实测约为 18 倍；2026-10 换一台服务器重新运行，约为 10 倍。PyTorch 的耗时几乎不变，因为多出来的样本只是让矩阵多了几行。（这次实测时，机器上还有别的任务在运行。同一个 Value 训练脚本单独运行时，每步约 25 毫秒。）

**4. `torch.autograd.gradcheck`**：PyTorch 自带的梯度检验。它的原理和我们的 `02_grad_check.py` 相同：把有限差分得到的数值梯度和 autograd 的解析梯度比较。官方文档提醒，它要用双精度输入。我们对全部 97 个参数运行它，结果是 `True`。

下表列出生产级代码多做了什么，以及原因：

| 极简代码（`01_engine.py`） | 生产级代码（`torch.autograd`） | 原因 |
|---|---|---|
| 每个标量一个节点 | 每个张量运算一个节点 | 节点数从"参数 × 样本"的量级降到"层数"的量级，Python 开销几乎消失。 |
| `_backward` 是标量乘法 | 反向传播是 VJP（`G·Wᵀ`、`Xᵀ·G`），调用 BLAS / CUDA kernel | 矩阵乘法能用上 CPU 的 SIMD 和 GPU 的并行计算。 |
| 递归的拓扑排序 | C++ 实现的反向引擎，按依赖关系调度 | 图再深也不会耗尽 Python 的递归栈（见动手任务 3）。 |
| 所有中间值一直保留 | 只保存反向传播需要的中间结果，默认在反向传播后释放计算图 | 节省显存。第 14 章的激活检查点（activation checkpointing）更进一步。 |
| 七种运算 | 上千种运算，每种都有注册好的反向函数 | 自定义运算可以用 `torch.autograd.Function` 自己写 forward / backward。 |
| 梯度检验要自己写 | `torch.autograd.gradcheck` / `gradgradcheck` | 同一个思路，框架内置。 |
| `net.zero_grad()` | `optimizer.zero_grad()` | 原因相同：梯度默认累加。 |

后面的每一章都会写新的运算（第 8 章的注意力、第 9 章的 RMSNorm 等）。对这些运算，我们直接用 PyTorch 的 autograd，不再手写反向传播。但你现在知道 `loss.backward()` 里面发生了什么。

## 采用方与来源

反向模式自动微分是深度学习框架的**事实标准**（GOAL.md 第 2.1 节的 B 类）。所有主流开源模型都用它训练：

- **PyTorch**：`torch.autograd`，[Autograd mechanics](https://docs.pytorch.org/docs/stable/notes/autograd.html)。Llama、Qwen、DeepSeek、OLMo 等开源模型的官方训练 / 推理代码都基于 PyTorch。
- **JAX**：`jax.grad` / `jax.vjp`，[Automatic differentiation](https://docs.jax.dev/en/latest/automatic-differentiation.html)。Gemma 的官方实现基于 JAX。
- **TensorFlow**：`tf.GradientTape`，[Introduction to gradients and automatic differentiation](https://www.tensorflow.org/guide/autodiff)。

> 待核实：上面列出了各模型家族的训练代码基于哪个框架。这些说法来自对各项目公开仓库的一般印象。写作时没有逐一打开技术报告核对。这一点不影响本章内容，因为本章只讲原理。

---

## 引导问题

向 Claude Code 提出这些问题。一直问到你能用自己的话讲清楚答案：

1. 在 `L = (a·b + c)²` 的例子里，假设 a 同时出现在 c 的位置（`L = (a·b + a)²`）。计算图会变成什么样？手算 `∂L/∂a`，再用 `Value` 验证。
2. 为什么梯度检验用中心差分 `(L(p+ε) − L(p−ε)) / 2ε`，而不用单边差分 `(L(p+ε) − L(p)) / ε`？ε 取得太大会怎样？取得太小会怎样？（可以改 `02_grad_check.py` 里的 `EPS` 试试。）
3. ReLU 在 0 点不可导。我们的代码在 `data == 0` 时给出的导数是 0。这会造成问题吗？梯度检验在这一点上会怎样？
4. 反向模式一次反向传播得到"一个输出对所有输入"的梯度。前向模式一次得到"所有输出对一个输入"的导数。什么情况下前向模式更划算？
5. 训练 500 步后，x = −3 附近拟合得最差。原因是模型容量不够、训练步数不够，还是学习率不合适？设计实验验证你的猜测。
6. 反向传播要用到前向传播时的中间值（例如乘法里的 `other.data`）。一个几十层的大模型，前向传播时要保存多少中间值？这和训练时的显存占用有什么关系？

## 动手任务

每个任务都要运行代码，并查看结果。

**任务 1（基础）**：手算 `L = (a·b + c)²` 在 `a = 1, b = 2, c = −1` 时 L 对 a、b、c 的梯度。先画计算图，再从右往左填梯度。然后改 `01_engine.py` 末尾的例子，验证你的结果。

**任务 2（核心）**：给 `Value` 加一个 `sigmoid` 运算：前向是 `s = 1 / (1 + e⁻ᵃ)`，局部导数是 `s·(1 − s)`。把它加进 `02_grad_check.py` 中 `expression_case` 的表达式，确认梯度检验通过。然后在 `03_train_mlp.py` 里，把 MLP 的激活函数先换成 `"relu"`，再换成你的 `sigmoid`。比较 500 步后的损失。

**任务 3（挑战）**：把 `04_pytorch_compare.py` 速度测试里的样本数从 200 改成 2000。你会看到 `RecursionError`：递归的拓扑排序在深图上耗尽了栈（损失是 2000 项串起来的加法）。把 `backward` 里的 `build` 改写成用栈的非递归版本。用 `02_grad_check.py` 确认引擎仍然正确，再测 2000 个样本的耗时。进一步：写一个只支持矩阵乘法、加法、tanh、求平均的 `Tensor` 类。反向传播用第 10 节的 VJP 公式，并和 PyTorch 对拍梯度。

---

## 本章参考文献

- Andrej Karpathy. *micrograd*（MIT 许可，本章引擎的原型）：<https://github.com/karpathy/micrograd>
- Andrej Karpathy. *The spelled-out intro to neural networks and backpropagation: building micrograd*（视频）：<https://www.youtube.com/watch?v=VMj-3S1tku0>
- Stanford CS231n 课程笔记 *Backpropagation, Intuitions*（"分叉处梯度相加"的直观讲解）：<https://cs231n.github.io/optimization-2/>
- Baydin, Pearlmutter, Radul, Siskind. *Automatic Differentiation in Machine Learning: a Survey*, JMLR 18(153), 2018：<https://www.jmlr.org/papers/volume18/17-468/17-468.pdf>（arXiv：<https://arxiv.org/abs/1502.05767>）
- Rumelhart, Hinton, Williams. *Learning representations by back-propagating errors*, Nature 323, 1986：<https://www.nature.com/articles/323533a0>
- 3Blue1Brown. *Backpropagation calculus*（有中文字幕版）：<https://www.3blue1brown.com/lessons/backpropagation-calculus>
- PyTorch 文档 *Autograd mechanics*：<https://docs.pytorch.org/docs/stable/notes/autograd.html>；`torch.autograd.gradcheck`：<https://docs.pytorch.org/docs/stable/generated/torch.autograd.gradcheck.gradcheck.html>

**下一章**：有了自动微分，网络想加多深都可以。但到目前为止，我们只做过"预测一个数"的回归。如果要判断一张图是猫还是狗、下一个词是哪一个，输出就变成了"每个类别的概率"。这时均方误差不再合适。第 5 章，我们讲分类与概率：softmax、交叉熵，以及为什么分类不用均方误差。
