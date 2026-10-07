# 第 3 章：非线性与神经网络 —— 用折线拼出曲线

[English](README.md) · **中文**

> **目标**：读完这一章，你能讲清楚为什么线性层不管叠多少层都还是线性的。你能画出一个 ReLU 隐藏单元对应的折线。你还能用梯度下降训练一个两层 MLP，拟合 `y = sin(2x)` 这条曲线。

📺 **本章视频**：还没有发布。要在本机渲染，运行 `bash chapters/03-neural-network/video/build.sh`。
🧪 **本章自检**：学完后，在 Claude Code 里输入 `/ch03-neural-network`。

---

上一章我们把输入从一个数扩展成一组数，把模型写成 `Ŷ = XW + b`，还弄清了矩阵乘法的形状规则。但不管 W 多大，这个模型仍然是**线性**的。输入的变化翻倍，输出的变化也翻倍。画成图，它永远是直线或平面。这一章要回答一个问题：**数据是弯的，直线拟合不了，怎么办？**

答案是神经网络。它的关键只是一个小改动：在两个线性层之间，插入一个非线性的"激活函数"。这一章我们会看到，为什么这个小改动就够了。我们还会看到它在图上的样子：**很多段直线拼成的一条曲线**。

## 1. 直觉：直线拟合不了曲线

第 1 章的任务 3 留了一个问题：如果数据是一条复杂的曲线，我们每次都要猜公式吗？这一章用一条具体的曲线来回答它。

本章的数据：x 在 [−3, 3] 上均匀取 100 个点，`y = sin(2x)`。这是一条上下起伏、有四个弯的波浪线。我们没有加噪声，因为这里只想看一个问题：模型能不能弯？

先用第 1 章的方法：在所有直线里，找均方误差最小的那一条（最小二乘解析解）。运行：

```bash
uv run python chapters/03-neural-network/code/01_linear_is_not_enough.py
```

```
1) Fit y = sin(2x) with a straight line
   Best straight line: y = -0.165·x + 0.000, MSE = 0.4341
   Reference: variance of y = 0.5178 (the MSE if we always predict the mean)
```

最好的直线几乎是平的，MSE 是 0.4341。一个什么都不学、总是预测平均值的模型，MSE 是 0.5178，也差不了多少。**原因不是训练得不够，也不是学习率没调好。**这已经是所有直线里最好的一条。问题出在模型本身：直线弯不过来。

## 2. 多叠几层线性层？没用

一个自然的想法是：一层不够，就叠两层。按第 2 章的写法（每一行是一个样本，`h ← h·W + b`），两层线性层是：

```
Ŷ = (X·W1 + b1)·W2 + b2
```

把括号乘开：

```
Ŷ = X·(W1·W2) + (b1·W2 + b2)
  = X·W + b        where W = W1·W2, b = b1·W2 + b2
```

`W1·W2` 还是一个矩阵，`b1·W2 + b2` 还是一个向量。**两层线性层等于一层线性层。**叠多少层都一样。一层一层合并下去，结果还是 `X·W + b`。线性函数的复合还是线性函数。

代码里的 `collapse` 函数做的就是这个合并（完整代码见 [`code/01_linear_is_not_enough.py`](code/01_linear_is_not_enough.py)）：

```python
def collapse(layers):
    W, b = layers[0]
    for W_next, b_next in layers[1:]:
        W, b = W @ W_next, b @ W_next + b_next   # (X·W + b)·W' + b' = X·(W·W') + (b·W' + b')
    return W, b
```

随机造两个只有线性层的网络。先逐层计算输出，不走捷径。再用合并后的一层计算一遍，然后对比：

```
2) Stacked linear layers are still one linear layer
   2 layers 1→8→1: 25 parameters, merged y = 0.375·x +2.045, max difference layer-by-layer vs merged = 8.9e-16
   3 layers 1→8→8→1: 97 parameters, merged y = -0.145·x -3.210, max difference layer-by-layer vs merged = 4.4e-15
```

差在 10⁻¹⁵ 这个量级，只是浮点数的舍入误差。在数学上，两个结果完全相等。不管是 25 个参数还是 97 个参数，画出来都只是一条直线。

还有更直接的证据。用梯度下降训练一个两层线性、中间宽度为 8 的网络（第 4 节的代码，去掉激活函数）。1000 步后，损失降到 0.4341。之后损失**完全不动**，和最好的直线完全相同（见第 7 节表格第一行）。参数再多，模型也只是一条直线。

## 3. 激活函数：在两层之间加一个弯

问题在于每一部分都是线性的。解决办法只需要一个小改动：在两层之间，让每一个数都经过一个**非线性**函数。这个函数叫**激活函数**（activation function）。最常用的激活函数是 **ReLU**（Rectified Linear Unit，修正线性单元）：

```
ReLU(z) = max(0, z)
```

ReLU 把小于 0 的数变成 0，大于 0 的数原样保留。它的图像是一条在 0 处有一个**折点**（kink）的线。有两点要注意：

- ReLU **逐元素**作用。它对矩阵里的每个数分别计算，不会把不同的数混在一起。所以它不改变形状：`(N, H)` 进，`(N, H)` 出。
- 有了这一个折点，第 2 节的推导（乘开括号、合并成一层）就走不通了。`ReLU(X·W1 + b1)·W2` 没法写成 `X·(某个矩阵)`，因为矩阵乘法做不出折点。

激活函数不止一种。运行：

```bash
uv run python chapters/03-neural-network/code/02_activations.py
```

```
1) Values of the activation functions at some points
   z            -3.0    -1.0     0.0     1.0     3.0
   ReLU        0.000   0.000   0.000   1.000   3.000
   sigmoid     0.047   0.269   0.500   0.731   0.953
   tanh       -0.995  -0.762   0.000   0.762   0.995
   SiLU       -0.142  -0.269   0.000   0.731   2.858
   GELU       -0.004  -0.159   0.000   0.841   2.996
```

| 激活函数 | 公式 | 说明 |
|---|---|---|
| sigmoid | `σ(z) = 1 / (1 + e^(−z))` | 早期神经网络常用。S 形，把输入压到 (0, 1)。两头太平：z 稍微离开 0，梯度就接近 0。用它的深层网络很难训练 |
| tanh | `tanh(z)` | 也是 S 形，把输入压到 (−1, 1)，以 0 为中心。但两头同样太平 |
| ReLU | `max(0, z)` | 2010 年前后流行起来（Nair & Hinton 2010；Glorot 等 2011）。函数形式简单，正半边的梯度恒为 1 |
| SiLU（也叫 Swish） | `z · σ(z)` | ReLU 的平滑版。z 是大正数时约等于 z，是大负数时约等于 0。0 附近是一段光滑的弯 |
| GELU | `z · Φ(z)`，Φ 是标准正态分布的累积分布函数 | 另一个平滑版，形状和 SiLU 很接近 |

今天的大模型用后两个。千问（Qwen）、DeepSeek、Llama 的前馈层用 **SwiGLU**，Gemma 用 **GeGLU**。它们是"门控"（gated）结构：一路经过 SiLU（或 GELU），再和另一路逐元素相乘。门控的细节在第 9 章讲。这一章只要知道：**SiLU 和 GELU 是 ReLU 的平滑版，作用完全相同：提供非线性。**下文都用 ReLU，因为它的折点最容易看清。

## 4. 两层 MLP：线性 → ReLU → 线性

把线性层、激活函数、线性层依次连起来，就得到最简单的神经网络：两层的**多层感知机**（Multi-Layer Perceptron，MLP）。

```
Z = X·W1 + b1        (N, 1) → (N, H)     layer 1: linear
A = ReLU(Z)          (N, H)              activation: element by element
Ŷ = A·W2 + b2        (N, H) → (N, 1)     layer 2: linear
L = mean((Ŷ − Y)²)                       loss: the same mean squared error as in Chapter 1
```

用第 2 章的规则可以核对形状。`X` 是 `(N, 1)`，`W1` 是 `(1, H)`，`b1` 是 `(H,)`。`W2` 是 `(H, 1)`，`b2` 是 `(1,)`。中间这 H 个数叫**隐藏单元**（hidden unit），也常叫神经元。H 叫网络的**宽度**（width）。参数一共 `H + H + H + 1 = 3H + 1` 个。

代码里对应的是下面三行（完整代码见 [`code/03_mlp_numpy.py`](code/03_mlp_numpy.py)）：

```python
def forward(p, x, act="relu"):
    z = x @ p["W1"] + p["b1"]         # Z = X·W1 + b1
    a = act_fn(z, act)                # A = ReLU(Z)
    y_hat = a @ p["W2"] + p["b2"]     # Ŷ = A·W2 + b2
    return y_hat, (z, a)
```

`act="linear"` 时，`act_fn` 原样返回输入。这就是第 2 节那个"两层线性"的网络。代码是同一份，只差一个激活函数。

## 5. 一个 ReLU 隐藏单元 = 一个折点

MLP 为什么能弯？只看第 j 个隐藏单元对输出的贡献。输入是一个数 x 时，这个单元先算 `w·x + b`（w、b 是 W1、b1 的第 j 个数）。然后经过 ReLU。再乘上第二层的权重 v（W2 的第 j 个数）：

```
v · ReLU(w·x + b)
```

这个贡献的图像是一条**折线**：只有一个折点的线。一边是平的（ReLU 输出 0 的那一侧），另一边是斜的。折点在 `w·x + b = 0` 处，也就是 **`x = −b/w`**。

三个参数各自这样改变折线：

- 改 b：折点向左或向右移动。
- 改 w：斜的一边变陡或变缓。w 的正负决定斜的一边在右边还是在左边。
- 改 v：整条线缩放。v 为负数时，整条折线上下翻转。

网络的输出就是 H 条这样的折线之和，再加上 b2：

```
ŷ(x) = Σⱼ vⱼ · ReLU(wⱼ·x + bⱼ) + b2
```

折线相加还是由直线段组成的线（分段线性函数，piecewise-linear function），只是折点变多了。`02_activations.py` 的第二部分用几个 ReLU 手工拼出形状：

```
2) Build shapes from ReLUs (x from −2 to 2)
   x
        -2.0  -1.5  -1.0  -0.5   0.0   0.5   1.0   1.5   2.0
   |x| = ReLU(x)+ReLU(−x)
         2.0   1.5   1.0   0.5   0.0   0.5   1.0   1.5   2.0
   tent = ReLU(x+1)−2ReLU(x)+ReLU(x−1)
         0.0   0.0   0.0   0.5   1.0   0.5   0.0   0.0   0.0
```

两个 ReLU 拼出绝对值 |x|（一个 V 形）。三个 ReLU（折点在 −1、0、1，中间那个乘 −2）拼出一个"帐篷"（tent）：只在 [−1, 1] 里鼓起一个包，别处都是 0。有了帐篷，就能在任何位置堆出任何高度的包。包足够多、足够窄，它们的和就能紧贴任何一条连续曲线。

这就是**万能近似定理**（universal approximation theorem）的意思。只要隐藏单元足够多，一个隐藏层的网络就能把连续函数逼近到任意精度（Cybenko 1989；Hornik 1991；Leshno 等 1993 证明了对 ReLU 这类非多项式激活函数也成立）。

注意，这个定理只保证这样一组参数存在。它没有说梯度下降一定找得到。它也没有说需要多少个隐藏单元。下面用实验来看。

## 6. 训练：手推梯度 + 梯度下降

训练的四步和第 1 章相同：模型、损失、梯度、更新。唯一的新问题是梯度。参数现在分在两层里。要算损失对 W1 的梯度，必须"穿过"第二层和 ReLU。

方法是从损失出发往回走，一层一层地用链式法则：

```python
def gradients(p, x, y, act="relu"):
    n = len(x)
    y_hat, (z, a) = forward(p, x, act)
    loss = float(np.mean((y_hat - y) ** 2))
    d_yhat = 2 * (y_hat - y) / n          # ∂L/∂Ŷ: the same as 2/N·(ŷ − y) in Chapter 1
    d_W2 = a.T @ d_yhat                   # ∂L/∂W2 = Aᵀ · ∂L/∂Ŷ
    d_b2 = d_yhat.sum(axis=0)             # ∂L/∂b2 = Σ ∂L/∂Ŷ
    d_a = d_yhat @ p["W2"].T              # ∂L/∂A  = ∂L/∂Ŷ · W2ᵀ
    d_z = d_a * act_grad(z, act)          # ∂L/∂Z  = ∂L/∂A ⊙ ReLU′(Z)
    d_W1 = x.T @ d_z                      # ∂L/∂W1 = Xᵀ · ∂L/∂Z
    d_b1 = d_z.sum(axis=0)                # ∂L/∂b1 = Σ ∂L/∂Z
    return loss, {"W1": d_W1, "b1": d_b1, "W2": d_W2, "b2": d_b2}
```

请读一下这几行：

- 第二层 `Ŷ = A·W2 + b2` 就是一个线性回归，只是输入从 X 换成了 A。所以 `∂L/∂W2 = Aᵀ·∂L/∂Ŷ`。这个式子和第 1 章的"残差乘输入"形式相同。
- 要把误差传回隐藏层，就乘上 `W2ᵀ`。每个隐藏单元分到的"责任"，和它连到输出的权重成正比。
- 经过 ReLU 时，乘上 `ReLU′(Z)`。z > 0 时导数是 1，z < 0 时导数是 0。**没有被激活的单元，梯度变成 0**，这一步它的参数不动。（z 恰好等于 0 时导数没有定义，按惯例取 0。实际中几乎碰不到这种情况。）
- 第一层又是一个"线性回归"。它的输入是 X，误差是 `∂L/∂Z`。

这种从后往前、逐层应用链式法则的算法叫**反向传播**（backpropagation）。第 4 章会系统地讲它，并让计算机自动完成。这里我们先手推梯度，再用第 1 章任务 2 的**数值梯度**检查有没有推错：

```
1) Gradient check (width 8): manual vs numerical gradients, max difference = 1.1e-10
```

差是 10⁻¹⁰，所以手推的公式是对的。更新规则和第 1 章完全相同：

```python
for k in p:
    p[k] -= lr * g[k]             # θ ← θ − η · ∂L/∂θ, the same as in Chapter 1
```

初始化也要说明一下。W1 从标准正态分布中抽取。b1 的取法让每个隐藏单元的折点 `−b1/W1` 一开始就随机分布在 [−3, 3] 里。W2 的尺度按 `1/√H` 缩小。初始化有很多讲究，第 6 章再系统地讲。

## 7. 实验：宽度 2、8、64

运行：

```bash
uv run python chapters/03-neural-network/code/03_mlp_numpy.py
```

设置是：学习率 0.01，全量梯度下降 20000 步，随机种子 0（约 25 秒）：

| 模型 | 参数量 | 第 0 步 | 第 1000 步 | 第 5000 步 | 第 20000 步 |
|---|---:|---:|---:|---:|---:|
| 两层线性（无激活），宽 8 | 25 | 2.0895 | 0.4341 | 0.4341 | **0.4341** |
| ReLU，宽 2 | 7 | 0.5175 | 0.3245 | 0.3155 | **0.3155** |
| ReLU，宽 8 | 25 | 0.6509 | 0.2731 | 0.0350 | **0.0152** |
| ReLU，宽 64 | 193 | 2.7760 | 0.1169 | 0.0133 | **0.0009** |
| 对照：最好的直线 | 2 | | | | 0.4341 |

请看这几个结果：

- **同样是 25 个参数，有没有 ReLU 差别很大。**两层线性停在 0.4341（就是直线）。加了 ReLU，损失降到 0.0152，小了将近 30 倍。区别只在中间那一个 `max(0, z)`。
- **宽度 2 只比直线好一点。**两个隐藏单元最多给出两个折点、三段直线。而 sin(2x) 在 [−3, 3] 上有四个弯，三段直线怎么拼都拼不出来。
- **宽度 8 有了大致形状，宽度 64 几乎完美。**折点越多，弯就越圆滑。损失从 0.0152 降到 0.0009。

一次实验可能是运气。换 5 个随机种子（0–4）再跑一遍：

```
3) 5 random seeds (0–4), loss after 20000 steps
   width  2: 0.3155  0.4337  0.3866  0.3874  0.3866   median 0.3866
   width  8: 0.0152  0.0628  0.0064  0.0146  0.0333   median 0.0152
   width 64: 0.0009  0.0005  0.0003  0.0009  0.0002   median 0.0005
```

规律是稳定的：越宽越好。但宽度 8 对种子很敏感（从 0.0064 到 0.0628，差将近 10 倍）。有两件事影响结果：折点从哪里出发，以及有没有单元"死掉"。死掉的单元在所有数据上都有 z < 0，所以梯度恒为 0，再也不动。宽度 64 有大量冗余的折点，所以结果稳定得多。这也给出了一个朴素的直觉：现代大模型为什么宁宽勿窄。

最后，把宽度 8 的网络拆开看：

```
4) The width-8 network: each hidden unit is one kink
   Kink positions x = −b1/W1: -2.00  -0.79  0.45  0.86  2.20  2.21  2.53  2.99
   Max amplitude of one hinge piece in the data range = 12.17 (the network output has an amplitude of only 1.20: the pieces cancel each other)
   Max difference between (sum of pieces + b2) and the network output = 2.2e-15
```

- 8 条折线加起来就是网络的输出，误差只有浮点舍入。第 5 节的公式 `ŷ = Σ vⱼ·ReLU(wⱼx + bⱼ) + b2` 不是比喻，而是恒等式。
- 折点的位置是梯度下降自己找出来的。我们没有告诉它任何关于正弦的信息。有几个折点挤在右边 2.2–3.0 一带。这说明这 8 个单元没有被"均匀地"用好。这正是宽度 8 不够稳定的原因之一。
- 单条折线的幅度能到 12，而输出只有 1.2：各条折线在互相抵消。网络学到的不是"每个单元负责一小段"这种整齐的分工，而是一组加起来恰好正确的数。

## 8. 小结

- **线性叠加还是线性**：`(X·W1 + b1)·W2 + b2 = X·(W1·W2) + (b1·W2 + b2)`。参数再多，也只是一条直线。
- **激活函数**提供非线性，逐元素作用。ReLU = `max(0, z)`。sigmoid 和 tanh 是早期的选择。现代大模型用 ReLU 的平滑版 SiLU 和 GELU（放在 SwiGLU / GeGLU 里，第 9 章讲）。
- **一个 ReLU 隐藏单元 = 一个折点**，折点在 `x = −b/w`。两层 MLP 的输出 = H 条折线之和 + b2。
- **宽度**决定网络能拼出多少个折点。在 sin(2x) 上：宽 2 → 0.3155，宽 8 → 0.0152，宽 64 → 0.0009。
- **训练方法没变**：还是模型、损失、梯度、更新四步。梯度用链式法则从后往前推（反向传播，第 4 章细讲）。

---

## 从极简代码到生产级代码

在极简代码里，网络的前向传播和梯度都是我们手写的矩阵运算。第 1–6 章的生产级代码就是 PyTorch 的标准写法，在 [`code/04_pytorch_version.py`](code/04_pytorch_version.py)：

```python
model = nn.Sequential(
    nn.Linear(1, 64),    # Z = X·W1ᵀ + b1 (the PyTorch weight has the shape (out, in))
    nn.ReLU(),           # A = ReLU(Z)
    nn.Linear(64, 1),    # Ŷ = A·W2ᵀ + b2
)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

for step in range(20001):
    y_hat = model(x)              # 1. forward pass
    loss = loss_fn(y_hat, y)      # 2. loss
    optimizer.zero_grad()         # 3. set the gradients to zero
    loss.backward()               # 4. backward pass: autograd calculates the gradients of all parameters
    optimizer.step()              # 5. update
```

训练循环的五行和第 1 章**完全相同**，变的只是 `model`。运行：

```bash
uv run python chapters/03-neural-network/code/04_pytorch_version.py
```

```
1) Parity check: NumPy initial parameters copied into PyTorch (float64), width 64, SGD learning rate 0.01
   step     0: loss = 2.7760
   step  1000: loss = 0.1169
   step  5000: loss = 0.0133
   step 20000: loss = 0.0009
   PyTorch 0.0009144016  vs  NumPy with manual gradients 0.0009144016, difference = 6.2e-18

2) Default PyTorch initialization + float32, width 64, same training for 20000 steps
   step     0: loss = 0.4166
   step  1000: loss = 0.0927
   step  5000: loss = 0.0320
   step 20000: loss = 0.0019
   193 parameters (W1: 64, b1: 64, W2: 64, b2: 1)
```

- **对拍**：把 NumPy 版的初始参数原样拷进 PyTorch（注意转置），两边都用 float64。20000 步之后，损失差 6.2 × 10⁻¹⁸。这说明手推的梯度和 autograd 算出的梯度每一步都一致。
- **默认初始化也能达到差不多的效果**：换成 PyTorch 自带的初始化和 float32，20000 步后损失是 0.0019。它和手工初始化的 0.0009 在同一个量级。

下表列出生产级代码改变了什么，以及原因：

| 极简代码 | 生产级代码 | 原因 |
|---|---|---|
| `x @ W1 + b1` | `nn.Linear(1, 64)` | 模块负责创建、初始化和登记参数。`model.parameters()` 一次拿到所有参数，交给优化器 |
| `np.maximum(0, z)` | `nn.ReLU()` | 激活函数也是一个模块。换成 `nn.SiLU()` 或 `nn.GELU()` 只改一行 |
| 手写三行，依次连起来 | `nn.Sequential(...)` | 按顺序连接子模块。层数多了以后，写成自定义的 `nn.Module`（第 9 章的 Transformer 就是这样） |
| 手推 6 行梯度公式 | `loss.backward()` | 两层已经要推 6 行。几十层、上百种运算时，手推不现实。第 4 章你会自己实现 autograd |
| 自己的初始化规则 | `nn.Linear` 的默认初始化（Kaiming 均匀分布一类） | 默认值对多数网络够用。为什么尺度要按输入维度缩放，第 6 章讲 |
| `p[k] -= lr * g[k]` | `optimizer.step()` | 第 6 章换成 AdamW 时，训练循环一行都不用改 |

## 采用方与来源

本章的主流技术里，只有一项需要核实"谁在用"：现代大模型前馈层里的激活函数。ReLU、sigmoid、tanh 只作为历史铺垫来讲（共识规则 C：必要铺垫）。

| 技术 | 采用方（头部开源模型家族） | 来源 |
|---|---|---|
| SwiGLU（门控 + SiLU） | Llama | LLaMA 论文 2.2 节"SwiGLU activation function"：<https://arxiv.org/abs/2302.13971> |
| | Qwen | Qwen3 技术报告，模型架构一节（GQA、SwiGLU、RoPE、RMSNorm）：<https://arxiv.org/abs/2505.09388> |
| | DeepSeek | DeepSeek LLM 技术报告，架构一节（RMSNorm、SwiGLU）：<https://arxiv.org/abs/2401.02954> |
| GeGLU（门控 + GELU） | Gemma | Gemma 技术报告，模型架构一节（GeGLU activations）：<https://arxiv.org/abs/2403.08295> |

SwiGLU 本身来自 Shazeer 2020 *GLU Variants Improve Transformer*（<https://arxiv.org/abs/2002.05202>）。第 9 章会逐项核实并展开。

## 引导问题

向 Claude Code 提出这些问题。一直问到你能用自己的话讲清楚答案：

1. 把 ReLU 换成 `f(z) = 2z + 1`，网络还能拟合 sin(2x) 吗？换成 `f(z) = z²` 呢？什么样的函数能当激活函数？（提示：搜索"万能近似定理"对激活函数的要求。）
2. 一个隐藏单元在所有 100 个数据点上都有 `z < 0`。它的梯度是多少？训练会怎样？这个问题叫"死 ReLU"（dying ReLU）。SiLU 和 GELU 为什么能缓解它？
3. 两层 MLP 的输出是折线。这说明在 x = 10（训练数据范围之外）时，网络的预测是什么样子？预测会继续像正弦一样起伏吗？
4. 宽度 64 的网络有 193 个参数，数据只有 100 个点。参数比数据还多，为什么没出问题？如果数据有噪声，会怎样？
5. 输入从 1 维变成 d 维时，一个 ReLU 隐藏单元的"折点"变成了什么几何形状？（提示：在二维里，`w·x + b = 0` 是一条直线。）

## 动手任务

每个任务都要运行代码，并查看结果。

**任务 1（基础）**：在 `03_mlp_numpy.py` 里把学习率从 0.01 改成 0.03。重新训练宽 8 和宽 64 的网络。和原来的结果对比，损失怎样变化？如果出现 `nan`，说明发生了什么？（回想第 1 章的临界学习率。）

**任务 2（核心）**：把 `act_fn` 和 `act_grad` 改成 tanh（导数是 `1 − tanh²(z)`）。先用 `numerical_gradients` 做梯度检验。再训练宽 8 和宽 64 的网络，和 ReLU 的结果比较。tanh 网络的输出还是折线吗？

**任务 3（挑战）**：不用梯度下降，**手工**设计一个宽度不超过 12 的 ReLU 网络来逼近 sin(2x)。在 [−3, 3] 上取 12 个等距的点，让网络的折线正好穿过这些点上的 sin(2x) 值。（提示：从左往右，每到一个折点，斜率要改变多少？这个改变量就是这个单元的 v·|w|。）算出它的 MSE。把它和梯度下降训练出的宽 8、宽 64 网络对比。

---

## 本章参考文献

- Goodfellow, Bengio, Courville. *Deep Learning*，第 6 章"深度前馈网络"（6.1 节用 XOR 讲线性模型为什么不够，6.3 节讲隐藏单元与 ReLU）：<https://www.deeplearningbook.org/contents/mlp.html>
- Michael Nielsen. *Neural Networks and Deep Learning*，第 4 章"神经网络可以计算任何函数的可视化证明"：<http://neuralnetworksanddeeplearning.com/chap4.html>
- 3Blue1Brown. *But what is a neural network?*（有中文字幕版）：<https://www.3blue1brown.com/lessons/neural-networks>
- Cybenko (1989). *Approximation by superpositions of a sigmoidal function*：<https://doi.org/10.1007/BF02551274>
- Leshno, Lin, Pinkus, Schocken (1993). *Multilayer feedforward networks with a nonpolynomial activation function can approximate any function*：<https://doi.org/10.1016/S0893-6080(05)80131-5>
- Nair & Hinton (2010). *Rectified Linear Units Improve Restricted Boltzmann Machines*：<https://www.cs.toronto.edu/~hinton/absps/reluICML.pdf>
- Glorot, Bordes, Bengio (2011). *Deep Sparse Rectifier Neural Networks*：<https://proceedings.mlr.press/v15/glorot11a.html>
- Hendrycks & Gimpel (2016). *Gaussian Error Linear Units (GELUs)*：<https://arxiv.org/abs/1606.08415>
- Elfwing, Uchibe, Doya (2017). *Sigmoid-Weighted Linear Units for Neural Network Function Approximation in Reinforcement Learning*（SiLU）：<https://arxiv.org/abs/1702.03118>
- Shazeer (2020). *GLU Variants Improve Transformer*（SwiGLU、GeGLU）：<https://arxiv.org/abs/2002.05202>
- PyTorch 官方教程 *Build the Neural Network*（`nn.Sequential`、`nn.Linear`、`nn.ReLU`）：<https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html>
- [Mathematical theory of deep learning](https://arxiv.org/abs/2407.18384)（`references.md` 已收录。其中万能近似的章节适合想看严格证明的读者）

**下一章**：这一章的两层网络，我们已经要手推 6 行梯度，还要用数值梯度检查一遍。几十层、上百种运算的网络呢？手推梯度工作量大，也容易出错。第 4 章，我们从链式法则出发，亲手写一个一百多行的自动微分（autograd）小工具。然后用它重新训练这一章的 MLP。之后你就知道 `loss.backward()` 里面发生了什么。
