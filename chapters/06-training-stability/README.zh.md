# 第 6 章：让训练稳定 —— 初始化、归一化、残差连接、AdamW 与学习率调度

[English](README.md) · **中文**

> **目标**：读完这一章，你能讲清楚一个几十层的网络为什么"训不动"。你能亲手用初始化、RMSNorm、残差连接、AdamW、warmup + 衰减、梯度裁剪把它修好。你还能用对比实验指出每个技巧各自修好了什么。

📺 **本章视频**：还没有发布。要在本机渲染，运行 `bash chapters/06-training-stability/video/build.sh`。
🧪 **本章自检**：学完后，在 Claude Code 里输入 `/ch06-training-stability`。

---

上一章我们用 softmax 和交叉熵训练了一个分类器。到这里，训练模型的四步（模型、损失、梯度、更新）已经全部就位，网络也能自动计算梯度。但我们用过的网络都只有两三层。大语言模型常有几十层：Qwen3-0.6B 有 28 层，Llama 3 405B 有 126 层。

这一章解决一个问题：**网络一加深，训练就失败，怎么办？** 我们先看训练怎样失败。然后用六个技巧逐个修好它，现在所有大模型都在用这六个技巧。

第 1 章留下的问题也在这里收尾。学习率太大会发散。那么训练大模型时，学习率该怎样安排？

## 1. 问题：30 层之后，信号去哪了

先做一个最简单的实验。搭一个 30 层的 MLP，每层宽 256。每层都是"乘矩阵，再过 ReLU"：

```
h_l = ReLU(h_{l-1} · W_l)
```

不训练这个网络。把一批随机输入（标准差 1）送进去，逐层看两个值。前向传播时，看每层输出的标准差。反向传播时，看误差信号（error signal）`∂L/∂h_l` 的标准差。运行：

```bash
uv run python chapters/06-training-stability/code/01_signal_propagation.py
```

权重按 `N(0, 1)` 初始化。输出如下（节选；脚本打印完整输出）：

| 层号 | 1 | 10 | 20 | 30 |
|---|---:|---:|---:|---:|
| 激活 std | 9.35 | 2.59 × 10¹⁰ | 7.66 × 10²⁰ | 2.27 × 10³¹ |
| 误差信号 std | 1.79 × 10²⁹ | 6.48 × 10¹⁹ | 2.33 × 10⁹ | 0.0622 |

每过一层，数值放大约 11 倍。30 层之后，数值变成 10³¹。反向传播时，误差信号从第 30 层往回走，同样每层都在放大。到第 1 层，它已经是 10²⁹。用这样的梯度更新参数，第一步就会溢出成 NaN。

现在把权重改小，取 `N(0, 0.01²)`。激活从 0.0935 一路缩小到第 30 层的 2.27 × 10⁻²⁹。误差信号在第 1 层只剩 1.79 × 10⁻²⁹。这叫**梯度消失**（vanishing gradient）：前面的层几乎收不到学习信号。第一种数值极大的情况叫**梯度爆炸**（exploding gradient）。

原因是基本的算术。一层就是一次乘法，30 层就是连乘 30 次。设每层的放大倍数稍微偏离 1，比如 1.1 或 0.9。连乘 30 次，就是 17 倍或 0.04 倍。如果偏离得多，结果就是极大或极小的数。**深度把任何一点尺度偏差都放大成指数级的偏差。**

这一章的六个技巧都在回答同一个问题：怎样让信号在几十层里既不爆炸也不消失？怎样让每次参数更新的步子始终可控？

## 2. 初始化：让每一层的尺度保持不变

### 2.1 直觉与公式

设输入 `h` 的每个分量方差是 `v`。把 `h` 乘上一个 `fan_in × fan_out` 的随机矩阵，矩阵元素的方差是 `σ²`。每个输出分量是 `fan_in` 项之和，所以方差变成 `fan_in · σ² · v`。要让方差不变，就要 `fan_in · σ² = 1`。这就是 **Xavier 初始化**（Xavier initialization，Glorot & Bengio, 2010）的思路。

ReLU 把一半的值变成 0，方差再减半。补回这个 1/2，就得到 **Kaiming 初始化**（Kaiming initialization，He et al., 2015）：

```
Var(W) = 2 / fan_in，也就是 std = sqrt(2 / fan_in)
```

对应的代码只有一行：

```python
def kaiming_std(fan_in: int = WIDTH) -> float:
    return math.sqrt(2 / fan_in)      # Var(W) = 2 / fan_in
```

### 2.2 实验

同一个 30 层网络，换五种初始化（`01_signal_propagation.py` 的输出）：

| 初始化 | 权重 std | 每层放大倍数 | 第 30 层激活 std | 误差信号 第 1 层 / 第 30 层 |
|---|---:|---:|---:|---:|
| std = 1.0 | 1 | 11.3 | 2.27 × 10³¹ | 2.87 × 10³⁰ |
| std = 0.01 | 0.01 | 0.113 | 2.27 × 10⁻²⁹ | 2.87 × 10⁻²⁸ |
| std = 0.02 | 0.02 | 0.226 | 2.44 × 10⁻²⁰ | 1.54 × 10⁻¹⁹ |
| Xavier，sqrt(1/fan_in) | 0.0625 | 0.707 | 1.71 × 10⁻⁵ | 3.46 × 10⁻⁵ |
| **Kaiming，sqrt(2/fan_in)** | **0.0884** | **1.000** | **0.56** | **0.801** |

"每层放大倍数"是理论值 `std · sqrt(fan_in / 2)`。它的 30 次方就能解释表里的数量级。Kaiming 让放大倍数正好是 1。30 层之后，激活 std 还有 0.56；误差信号从第 30 层传回第 1 层，还剩 80%。Xavier 只差一个因子 2，每层放大 0.707 倍。30 层之后，信号只剩十万分之几。

**初始化的尺度必须根据 fan_in 和激活函数来算，不能随手写一个数。**

### 2.3 那么大模型为什么都用 0.02？

打开开源模型的配置文件，常见 `"initializer_range": 0.02`。Qwen3-0.6B、Gemma 3 1B、OLMo 2、SmolLM3 的 `config.json` 里都是 0.02。OLMo 2 的技术报告也明确写道：所有参数用均值 0、标准差 0.02 的正态分布初始化。

但上表里 std = 0.02 的普通网络，30 层后信号只剩 10⁻²⁰。这不矛盾，因为大模型不是普通堆叠的网络。它有**归一化**和**残差连接**（下面两节）。有了这两样，信号的尺度不再依赖每层权重正好等于 `sqrt(2/fan_in)`。一个固定的小 std 就够用。另外，宽度为 1024 时 `sqrt(1/1024) ≈ 0.031`，所以 0.02 和 Xavier 的量级本来也差不多。

大模型的初始化常常还多一个细节：**写回残差流的那些输出投影，再按层数缩小**。（残差流（residual stream）是 `h` 穿过各个块的主干通路，见第 4 节。）GPT-2 这样做，nanoGPT 沿用了这个做法：这些矩阵用 `0.02 / sqrt(2 · 层数)`。原因在第 4 节的实验里能看到。

注意，并非所有模型都这样做。OLMo 2 报告说，他们从"按层缩放"的初始化换回了统一的 0.02。在他们的实验里，后者更稳。DeepSeek-V3 的技术报告则写的是：所有参数用标准差 0.006 初始化。

## 3. 归一化：LayerNorm → RMSNorm

### 3.1 直觉

初始化只管"开局"。训练开始后权重在变，每层的尺度会慢慢漂移。更彻底的办法是：**每一层都把输入强行拉回标准尺度**，不管前面发生了什么。这就是**归一化**（normalization）。

### 3.2 公式与代码

**LayerNorm**（Ba et al., 2016）作用于每个样本的向量 `x`（长度 d）。它先减均值，再除以标准差。然后乘一个可学习的缩放 `γ`，加一个偏移 `β`：

```
LayerNorm(x) = (x − μ) / sqrt(σ² + ε) · γ + β，   μ = mean(x)，σ² = mean((x − μ)²)
```

**RMSNorm**（Zhang & Sennrich, 2019）去掉"减均值"这一步和 `β`，只除以均方根：

```
RMSNorm(x) = x / sqrt(mean(x²) + ε) · γ
```

```python
def layer_norm(x, gamma, beta, eps=1e-6):
    mu = x.mean(-1, keepdim=True)
    var = ((x - mu) ** 2).mean(-1, keepdim=True)
    return (x - mu) / torch.sqrt(var + eps) * gamma + beta

def rms_norm(x, gamma, eps=1e-6):
    rms = torch.sqrt((x * x).mean(-1, keepdim=True) + eps)
    return x / rms * gamma
```

运行 `02_normalization.py`。它用一个 4 维向量显示两者的差别：

| | 输出 | 均值 | RMS |
|---|---|---:|---:|
| 输入 x | [2, 4, 6, 8] | | |
| LayerNorm(x) | [−1.342, −0.447, 0.447, 1.342] | 0.000 | 1.000 |
| RMSNorm(x) | [0.365, 0.73, 1.095, 1.461] | 0.913 | 1.000 |

两者都把尺度拉到 1。唯一的区别是 LayerNorm 还把均值移到 0。脚本还验证了两件事。第一，输入先减掉均值后，两者的输出完全相同（最大差 0）。第二，把 x 放大 100 倍，RMSNorm 的输出几乎不变（最大变化 1.19 × 10⁻⁷）。这个性质叫**尺度不变性**（scale invariance）。

### 3.3 RMSNorm 丢掉了什么，为什么够用

RMSNorm 丢掉了"平移不变性"。输入整体加一个常数，LayerNorm 的输出不变，RMSNorm 的输出会变。RMSNorm 论文的假设是：让训练稳定的是**缩放不变性**（把尺度拉回来），减均值的作用不大。

实验里，RMSNorm 的效果和 LayerNorm 相当。但它少算一个均值，少存一组 `β`（参数从 2d 个降到 d 个）。论文报告，RMSNorm 在不同模型和框架上快 7%–64%。今天的主流开源大模型基本都用 RMSNorm。采用方列在本章末尾。

### 3.4 效果

把 RMSNorm 插进第 1 节的 30 层网络。每层先归一化，再乘矩阵：`h_l = ReLU(RMSNorm(h_{l-1}) · W_l)`。

| 初始化 | 第 1 层激活 | 第 10 层 | 第 20 层 | 第 30 层 | 误差信号 第 1 层 / 第 30 层 |
|---|---:|---:|---:|---:|---:|
| std = 1.0 | 9.33 | 10.1 | 9.7 | 9.92 | 1.33 |
| std = 0.01 | 0.0933 | 0.101 | 0.097 | 0.0992 | 1.33 |
| std = 0.02 | 0.187 | 0.201 | 0.194 | 0.198 | 1.33 |
| Kaiming | 0.825 | 0.89 | 0.858 | 0.877 | 1.33 |

不管初始化是 1.0 还是 0.01，信号在 30 层里都保持同一个量级。误差信号也不再指数级变化。归一化把一个脆弱的要求（"每层的尺度必须正好对"）变成了"每层自动对齐尺度"。

### 3.5 放在哪里：Pre-Norm

归一化放在哪里也很重要。原始 Transformer 把 LayerNorm 放在残差相加**之后**（Post-Norm）。现在的主流做法是放在每个子层的**输入**上（**Pre-Norm**），并在最后输出前再加一个 norm：

```
Post-Norm：h ← Norm(h + f(h))
Pre-Norm： h ← h + f(Norm(h))，   输出前：Norm(h)
```

Xiong et al.（2020）分析过这个问题。初始化时，Post-Norm 靠近输出层的梯度很大，必须用 warmup 才能训练。Pre-Norm 的梯度在各层大小相近。第 9 章的 Transformer 块是 Pre-Norm 结构，本章的实验网络也是。也有例外：OLMo 2 把 RMSNorm 放在注意力和 MLP 的**输出**上，再把结果加回残差流。OLMo 2 报告说，这样配合 QK-Norm 更稳。

## 4. 残差连接：给信号一条直通的路

### 4.1 直觉与公式

归一化管住了尺度，但深层网络还有另一个问题。**残差连接**（residual connection，He et al., 2016）只加了一个加号：

```
普通堆叠：h ← f(h)
残差连接：h ← h + f(h)
```

每一块只需要学"在当前值上改一点"。反向传播时，链式法则给出：

```
∂h_{l+1}/∂h_l = I + ∂f/∂h_l
```

那个 `I`（单位矩阵）的意思是：无论 `f` 学成什么样，梯度总有一条原样直通的路。30 块叠起来，梯度从输出到第 1 块，至少有一条"全走 I"的路径。这条路径不经过任何矩阵乘法。

### 4.2 实验

`03_residual.py` 搭 30 个块，每块是 `f(h) = ReLU(RMSNorm(h) · W1) · W2`。脚本比较三种写法。除了尺度，我们还看两个指标：

- **余弦相似度**（cosine similarity）：一批随机输入两两之间有多像。输入本身是 0.000（互相垂直）。越接近 1，说明网络越把不同输入"看成同一个东西"。
- **误差信号的方向相似度**：对同一个样本，第 1 块收到的误差信号和第 30 块收到的误差信号，方向有多像。

| 写法 | 第 30 块余弦相似度 | 第 30 块残差流 std | 误差信号 第 1 / 第 30 块（大小） | 方向相似度 |
|---|---:|---:|---:|---:|
| 普通堆叠 | 0.953 | 0.987 | 1.250 | −0.002 |
| 残差（W2 不缩放） | 0.549 | 5.545 | 3.795 | 0.256 |
| **残差（W2 × 1/√(2L)）** | **0.110** | **1.223** | **1.170** | **0.824** |

这张表有三个发现：

1. **普通堆叠即使加了 RMSNorm，尺度正常，信息却丢了。** 30 块之后，所有输入的表示两两相似度高达 0.953。网络几乎分不清谁是谁。误差信号传回第 1 块时，大小没问题（1.25）。但它的方向和第 30 块完全无关（−0.002），像被 30 个随机矩阵搅乱了。
2. **加了残差，信息保住了，误差信号也直通了。** 按层数缩小 W2 之后，相似度只有 0.110。第 1 块和第 30 块的误差信号，方向相似度是 0.824。这就是"直通的路"。
3. **残差流越加越大，所以输出投影要缩小。** 每块往残差流里加一份 `f(h)`。30 块之后，std 涨到 5.5，大约是 `sqrt(1 + 30)`。把 W2 乘以 `1/sqrt(2L)` 后，std 只到 1.22。这就是第 2.3 节"残差投影按层数缩小"的由来。

## 5. 优化器：SGD → 动量 → Adam → AdamW

信号能流通了，下一个问题是参数怎样走。第 1 章的 SGD 对所有参数用同一个学习率。问题在于，不同参数的"坡度"可能差几个数量级。

### 5.1 一个陡峭程度差 1000 倍的碗

`04_optimizers.py` 用一个四维的碗 `L(θ) = ½ Σ hᵢ θᵢ²`。四个方向的曲率（curvature）是 `h = [100, 10, 1, 0.1]`，起点是 `θ = [1, 1, 1, 1]`。第 1 章讲过，SGD 的学习率必须小于 `2 / 最大曲率 = 0.02`。否则，最陡的方向会发散。

**SGD**：`θ ← θ − η · g`

**动量**（momentum）：把梯度累积成"速度"。在平的方向上，步子一步步加快。在来回震荡的方向上，正负梯度互相抵消。

```
v ← β · v + g，   θ ← θ − η · v
```

**Adam**（Kingma & Ba, 2014）同时记两个滑动平均：梯度的滑动平均 `m`（方向），梯度平方的滑动平均 `v`（尺度）。**Adam 把每个参数的步长除以这个参数自己的梯度尺度**：

```
m ← β₁ m + (1 − β₁) g
v ← β₂ v + (1 − β₂) g²
m̂ = m / (1 − β₁ᵗ)，  v̂ = v / (1 − β₂ᵗ)          （偏差修正：m、v 从 0 起步，开头几步偏小）
θ ← θ − η · m̂ / (√v̂ + ε)
```

```python
m = b1 * m + (1 - b1) * g                   # first moment: direction
v = b2 * v + (1 - b2) * g * g               # second moment: scale
m_hat, v_hat = m / (1 - b1**t), v / (1 - b2**t)
step = -lr * m_hat / (np.sqrt(v_hat) + eps) # each parameter: divide by its own gradient scale
```

输出如下：

| | 第 1 步 \|Δθ₁\| | \|Δθ₂\| | \|Δθ₃\| | \|Δθ₄\| | 200 步后损失 | 损失 < 0.001 用了 |
|---|---:|---:|---:|---:|---:|---:|
| SGD（η = 0.019） | 1.9000 | 0.1900 | 0.0190 | 0.0019 | 2.36 × 10⁻² | 超过 200 步 |
| 动量（η = 0.019，β = 0.9） | 1.9000 | 0.1900 | 0.0190 | 0.0019 | 4.91 × 10⁻⁶ | 109 步 |
| Adam（η = 0.05） | 0.0500 | 0.0500 | 0.0500 | 0.0500 | 2.94 × 10⁻⁷ | 48 步 |

SGD 第一步在最陡的方向走了 1.9，在最平的方向只走了 0.0019。两者差 1000 倍。最陡的方向限制了学习率，所以 200 步后，最平的方向还停在 0.68。动量让平的方向越走越快，109 步收敛。

Adam 第一步让每个参数都正好走 η = 0.05。原因是：第一步的 `m̂ / √v̂` 等于 `g / |g| = ±1`。除法消去了梯度的大小，只剩方向。Adam 48 步就收敛。

Transformer 需要的正是"每个参数自己定步长"这个性质。Zhang et al.（2024）发现，Transformer 里的各个参数块（嵌入层、注意力的 Q/K/V、MLP）曲率差别很大。SGD 用同一个学习率应付不了，Adam 能。反过来，在各层结构相近的网络上，SGD 能和 Adam 效果相同。第 8 节的小 MLP 就是这种情况。

### 5.2 AdamW：把权重衰减"解耦"出来

**权重衰减**（weight decay）让参数每步往 0 缩一点，防止权重无限增长。传统写法是在损失里加 `½ λ ‖θ‖²`，也就是在梯度里加 `λθ`。这叫 L2 正则。在 SGD 里，这两种说法等价。

在 Adam 里，两者不等价。Adam 把梯度里的 `λθ` 和真实梯度一起除以 `√v`。**梯度大的参数，衰减被除以一个大数，几乎不衰减。梯度小的参数，衰减被放大。** AdamW（Loshchilov & Hutter, 2017）把衰减从梯度里拿出来，单独做：

```
θ ← θ − η · λ · θ               （解耦的权重衰减：不经过 √v）
θ ← θ − η · m̂ / (√v̂ + ε)        （正常的 Adam 更新）
```

`04_optimizers.py` 的第二个实验用两组参数，初值都是 1。损失不提供任何信号，梯度只是噪声。A 组噪声标准差 0.01，B 组 10。取 λ = 0.1、η = 0.01，跑 3000 步。理想情况下，权重衰减应该把两组同样地拉到 `(1 − ηλ)^3000 = 0.050`：

| | A 组（梯度小） | B 组（梯度大） |
|---|---:|---:|
| Adam + L2 正则 | 0.000 | 0.753 |
| AdamW（解耦） | 0.047 | 0.056 |

Adam + L2 下，同一个 λ，A 组被压到 0，B 组几乎没动。AdamW 下，两组都在理论值 0.050 附近。**AdamW 让衰减的力度只由 λ 决定，和梯度大小无关。** 这就是今天几乎所有开源大模型预训练都用 AdamW 的原因。常见超参是 β₁ = 0.9、β₂ = 0.95、λ = 0.1（Llama 2、DeepSeek-V3、SmolLM3 的报告都用这组数）。

### 5.3 哪些参数不做权重衰减

实践中，参数通常分成两组：**矩阵做权重衰减，一维参数（RMSNorm 的 γ、偏置）不做。** γ 的作用是给归一化后的信号定尺度，默认值是 1。把它往 0 拉没有道理。nanoGPT 的 `configure_optimizers` 就按 `p.dim() >= 2` 分组。

OLMo 2 更进一步，把**词嵌入**也排除在衰减之外。他们发现，衰减让嵌入的范数越来越小。结果早期层的梯度反而变大，尖峰变多。SmolLM3 报告说沿用了这个做法。本章代码只衰减矩阵：

```python
if p.dim() >= 2:                   # decay only the matrices, not γ
    p.mul_(1 - lr * cfg["wd"])     # decoupled weight decay
p -= lr * m_hat / (v_hat.sqrt() + 1e-8)
```

## 6. 学习率调度：warmup + 余弦，以及 WSD

第 1 章讲过，学习率超过临界值，训练就会发散。训练大模型时，学习率不是一个固定的数，而是一条随步数变化的曲线。

### 6.1 为什么开头要 warmup

训练刚开始时，参数是随机的。Adam 的 `v` 只见过几个梯度，估计不准。这时直接用峰值学习率，每个参数第一步都会走满 η（上一节刚看到）。整个网络被大幅推动，容易出现损失尖峰（loss spike，损失突然升高）。**warmup** 让学习率在前几百到几千步里，从接近 0 线性升到峰值。

### 6.2 为什么结尾要衰减

学习率大时，参数在谷底附近来回跳，损失降不下去。把学习率慢慢降下来，参数才能落进谷底。最常用的是**余弦衰减**（cosine decay，Loshchilov & Hutter 的 SGDR）。warmup 之后，学习率按余弦曲线从峰值降到峰值的 10%。Llama 2、Llama 3、OLMo 2 都用"线性 warmup + 余弦衰减"。

```python
def warmup_cosine(step, total, peak, warmup, min_ratio=0.1):
    if step < warmup:
        return peak * (step + 1) / warmup                  # warmup: 0 → peak
    p = (step - warmup) / max(1, total - warmup)           # decay progress 0 → 1
    return peak * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * p)))
```

### 6.3 WSD：先保持恒定，最后再降

余弦衰减有一个麻烦：曲线形状取决于总步数。所以**必须事先定好训练多长**。训到一半想多训一些，就得重新开始。

**WSD（Warmup-Stable-Decay）**由 MiniCPM（Hu et al., 2024）明确提出。warmup 之后，学习率**保持不变**（stable）。只在最后一小段（常见 10%–20%）快速降下来（decay）。stable 段的任何一个 checkpoint 都可以接一段衰减，得到一个"训练完成"的模型。想多训，就从 stable 段接着训。

```python
def wsd(step, total, peak, warmup, decay_frac=0.2, min_ratio=0.0):
    decay_start = int(total * (1 - decay_frac))
    if step < warmup:
        return peak * (step + 1) / warmup                  # W: warmup
    if step < decay_start:
        return peak                                        # S: stable, constant
    p = (step - decay_start) / max(1, total - decay_start)
    return peak * (1 - (1 - min_ratio) * p)                # D: decay, linear
```

`05_lr_schedule.py` 打印两条曲线（总步数 1000，峰值 3 × 10⁻³，warmup 100 步）：

| 步数 | 0 | 50 | 100 | 300 | 500 | 700 | 800 | 900 | 999 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| warmup + 余弦 | 3.0e−5 | 1.5e−3 | 3.0e−3 | 2.7e−3 | 1.9e−3 | 9.8e−4 | 6.2e−4 | 3.8e−4 | 3.0e−4 |
| WSD | 3.0e−5 | 1.5e−3 | 3.0e−3 | 3.0e−3 | 3.0e−3 | 3.0e−3 | 3.0e−3 | 1.5e−3 | 1.5e−5 |

现在很多开源模型都用这种"长时间恒定，最后衰减"的形状。SmolLM3 明确写的是 WSD（2000 步 warmup，最后 10% 线性降到 0）。Kimi K2 报告明确写用 WSD（前 10T token 恒定 2e−4，之后用 5.5T token 余弦降到 2e−5）。DeepSeek-V3 没有用 WSD 这个名字，但调度是同一个形状（2K 步 warmup，恒定到 10T token，再用 4.3T token 余弦降到 2.2e−5）。

> **注意：**本章训练类实验的数字，来自课程构建机上的一次 CPU 运行。不同的机器、不同版本的底层数学库，浮点运算的顺序略有不同。训练几百步后，这些微小差异会被放大。你本机跑出的数字，可能从小数点后第二、三位开始就不一样。请以下文不依赖具体数值的结论为准。2026-10 在另一台服务器上的复跑对照见 [runs/2026-10-01-gpu0-check/chapters-01-06.md](../../runs/2026-10-01-gpu0-check/chapters-01-06.zh.md)。

`07_schedule_experiments.py` 用第 8 节的网络比较三种调度（800 步，验证损失越低越好）：

| 调度 | 验证损失 |
|---|---:|
| warmup + 余弦 | 0.775 |
| WSD（最后 20% 线性降到 0） | 0.769 |
| 恒定学习率（只有 warmup） | 0.806 |

最后都降下来的两种调度结果相近，不衰减的明显更差。再看 WSD 怎样"随时收尾"。跑一条恒定学习率的主干。在第 400 步和第 800 步，各分出一个 100 步的衰减。把结果和"事先定好总长"的余弦对比：

| 总长 | stable 末尾 | 衰减 100 步后 | 专门跑一次余弦 |
|---:|---:|---:|---:|
| 500 步 | 0.887 | 0.801 | 0.796 |
| 900 步 | 0.806 | 0.755 | 0.760 |

衰减那 100 步，验证损失降了 0.05–0.09，追平了专门跑的余弦。WSD 一共只训了 400 + 100 + 400 + 100 = 1000 步，两次余弦要 500 + 900 = 1400 步。Hägele et al.（2024）在 1B、8B 模型上做了同样的比较，结论一致。恒定学习率加一段短衰减，能追平调好的余弦，还能省下大量做 scaling 实验的算力。第 12 章的阶梯实验和第 15 章的"退火"都会用到这一点。

## 7. 梯度裁剪：给步长加一个上限

即使一切都设好了，训练中偶尔也会遇到一批"坏数据"。例如 OLMo 2 报告里提到的长串重复字符。这时梯度突然变得巨大。**梯度裁剪**（gradient clipping，Pascanu et al., 2013）把所有参数的梯度拼成一个大向量。这个向量的范数超过上限（大模型几乎都用 1.0）时，裁剪把整个向量等比例缩小。方向不变。

```
‖g‖ = sqrt(Σ 所有梯度分量²)，   g ← g · min(1, max_norm / ‖g‖)
```

```python
def clip_by_global_norm(grads, max_norm=1.0):
    total = math.sqrt(sum(float((g * g).sum()) for g in grads))
    scale = min(1.0, max_norm / (total + 1e-6))
    for g in grads:
        g *= scale
    return total
```

`07_schedule_experiments.py` 在第 300–304 步混进 5 批坏数据（输入放大 30 倍）：

| | 坏数据时梯度范数最大 | 之后 40 步内最大训练损失 | 最终验证损失 |
|---|---:|---:|---:|
| RMSNorm 开，裁剪开 | 1.1 | 1.096 | 0.776 |
| RMSNorm 开，裁剪关 | 1.1 | 1.096 | 0.776 |
| RMSNorm 关，裁剪开 | 95.9 | 1.118 | 0.747 |
| RMSNorm 关，裁剪关 | 105.9 | 1.977 | 0.773 |

有 RMSNorm 时，第一个 norm 直接抵消了放大 30 倍的输入，梯度范数只有 1.1。裁不裁剪都一样。没有 RMSNorm 时，梯度范数冲到 100 左右。不裁剪，之后的训练损失跳到 1.977；裁剪后，只到 1.118。

**裁剪是最后一道安全措施。** 正常训练时，它基本不起作用。出问题时，它把这一步的步长限制住。它几乎没有代价，所以是大模型训练的标准做法。

## 8. 对比实验：同一个网络，把技巧逐个打开

现在把所有零件装到一起。`06_ablation.py` 的任务如下。输入是 16 维随机向量，标签由一个固定的"老师网络"给出（10 类）。每一步都采 128 个新样本，像预训练一样，数据不会用完。另留 2048 个样本计算验证损失。随机猜的交叉熵约为 ln 10 = 2.30。

学生网络有 12 块（24 个线性层），宽 32，训练 800 步。所有技巧都是手写的，每个都能单独开关。

```bash
uv run python chapters/06-training-stability/code/06_ablation.py      # about 1.5 min
```

**逐个加上技巧**（SGD 那几行在学习率 0.01 / 0.03 / 0.1 下各跑一次，取最好的）：

| 配置 | 学习率 | 验证损失 | 准确率 | 结论 | 三个学习率的验证损失 |
|---|---:|---:|---:|---|---|
| A 普通深网络（std = 1 初始化 + SGD） | — | nan | — | 发散 | 全部 nan |
| B + Kaiming 初始化 | 0.1 | 1.931 | 0.279 | 学得很慢 | 2.301 / 2.095 / 1.931 |
| C + 残差连接 | 0.03 | 0.821 | 0.705 | 成功 | 0.823 / 0.821 / nan |
| D + RMSNorm（Pre-Norm） | 0.1 | 0.811 | 0.700 | 成功 | 0.838 / 0.814 / 0.811 |
| E SGD → AdamW（恒定学习率） | 0.003 | 0.807 | 0.697 | 成功 | |
| F + warmup + 余弦衰减 | 0.003 | 0.775 | 0.711 | 成功 | |
| G + 梯度裁剪（= 全套） | 0.003 | 0.775 | 0.708 | 成功 | |

**从全套里每次只拿掉一个**：

| 配置 | 验证损失 | 准确率 | 结论 |
|---|---:|---:|---|
| 全套，但去掉残差 | 1.462 | 0.515 | 学得很慢 |
| 全套，但去掉 RMSNorm | 0.747 | 0.726 | 成功 |
| 全套，但初始化用 std = 1 | 0.825 | 0.697 | 成功 |
| 全套，但学习率恒定（无 warmup / 衰减） | 0.803 | 0.704 | 成功 |

（"结论"按验证损失粗分：nan 为发散，大于 2.0 为没学会，1.0–2.0 为学得很慢，小于 1.0 为成功。B 行的 1.931 离 2.0 的分界很近。2026-10 在另一台服务器上复跑同一份代码，得到 2.035（准确率 0.168），按同一规则就成了"没学会"。两次运行说明的是同一件事：只换初始化，24 层的普通堆叠几乎学不动。）

把这两张表和前面几节对照着读：

- **初始化决定训练能不能起步。** A 的初始损失就高达 6 × 10¹⁵。三个学习率下，损失都在第 2 步变成 NaN。换成 Kaiming（B）后不再发散。但 24 层的普通堆叠几乎学不动：最好的结果也只在 2.0 上下，离随机猜的 2.30 不远。
- **残差决定深网络能不能学。** 加上残差（C），验证损失从 2.0 上下直接降到 0.82。反过来，全套里只拿掉残差，损失就退回 1.462（复跑是 1.310），仍在"学得很慢"一档。残差是本实验里影响最大的开关。这和第 4 节"普通堆叠分不清输入"的发现一致。
- **归一化带来的是稳健性，不一定是更低的损失。** C 在学习率 0.1 时发散，D 加了 RMSNorm 后在 0.1 下照样能训。全套用 std = 1 的"坏"初始化，也能训到 0.825。但在这个小网络上，初始化和学习率都调好时，去掉 RMSNorm 的损失反而略低（0.747）。RMSNorm 的价值要在下面的压力测试里看。
- **AdamW 在这里和调好的 SGD 效果相同。** E 的 0.807 和 D 的 0.811 差不多。这个 MLP 每层结构相同，正是第 5.1 节说的"SGD 能和 Adam 效果相同"的情况。Adam 有两个优势。第一，不用逐个调学习率。第二，在 Transformer 这类参数块差异大的结构上也有效。
- **学习率衰减稳定地带来提升。** F 比 E 好 0.03。全套换成恒定学习率，也回到 0.803。
- **梯度裁剪在正常训练里不起作用**（F 和 G 一样）。它是为第 7 节那种意外准备的。

**高学习率压力测试**：把学习率从 0.003 放大 100 倍，到 0.3（`07_schedule_experiments.py`）：

| 配置 | 前 150 步最大训练损失 | 800 步后验证损失 |
|---|---:|---:|
| 全套 | 2.723 | 0.892 |
| 去掉 warmup | 6.178 | 0.890 |
| 去掉 RMSNorm | 1.1 × 10¹⁰ | 0.991 |

这里才看出 warmup 和 RMSNorm 的作用。去掉 warmup，开头的损失冲到 6.2。去掉 RMSNorm，损失尖峰冲到 10¹⁰ 以上。（复跑是 2.4 × 10¹²。尖峰这么高时，具体量级对舍入很敏感。）训练最后恢复了，但验证损失差了一截。

我们这个玩具模型只有几万个参数，这些尖峰还能自己恢复。大模型有几十亿个参数，要训练几个月。对这样的模型，一次尖峰可能意味着回滚到几天前的 checkpoint 重来。所以大模型把这些安全措施全部带上。

## 9. 小结

| 技巧 | 解决什么问题 | 本章的证据 |
|---|---|---|
| 按 fan_in 缩放的初始化 | 连乘导致信号指数级爆炸或消失 | 30 层后激活从 10³¹ / 10⁻²⁹ 变成 0.56 |
| RMSNorm（Pre-Norm） | 训练中尺度漂移；对初始化和学习率敏感 | 任何初始化下 30 层信号都稳定；η = 0.3 时尖峰从 10¹⁰ 以上降到 2.7 |
| 残差连接 | 深层"分不清输入"，梯度被搅乱 | 相似度 0.953 → 0.110；验证损失约 2.0 → 0.82 |
| AdamW | 参数间尺度差异大；衰减力度被梯度大小扭曲 | 48 步 vs SGD 超过 200 步；两组衰减 0.047 / 0.056 |
| warmup + 余弦 / WSD | 开头猛冲出尖峰；结尾落不进谷底 | 去掉 warmup 尖峰 6.2；衰减带来 0.03–0.09 的提升 |
| 梯度裁剪 | 偶发的坏数据产生巨大梯度 | 坏数据后损失尖峰 1.977 → 1.118 |

---

## 从极简代码到生产级代码

第 1–6 章的生产级代码，就是 PyTorch 的标准写法。同一套"全套"配置在 [`code/08_pytorch_version.py`](code/08_pytorch_version.py) 里：

```python
class Block(nn.Module):
    def __init__(self, width, hidden):
        super().__init__()
        self.norm = nn.RMSNorm(width, eps=1e-6)
        self.fc1 = nn.Linear(width, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, width, bias=False)    # output projection of the residual branch

    def forward(self, x):
        return x + self.fc2(F.relu(self.fc1(self.norm(x))))

decay = [p for p in model.parameters() if p.dim() >= 2]
no_decay = [p for p in model.parameters() if p.dim() < 2]
opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1},
                         {"params": no_decay, "weight_decay": 0.0}],
                        lr=3e-3, betas=(0.9, 0.95), eps=1e-8)
scheduler = torch.optim.lr_scheduler.LambdaLR(
    opt, lambda s: warmup_cosine(s, total, 1.0, warmup))   # multiplier function, peak = 1

for step in range(total):
    loss = F.cross_entropy(model(x), y)
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    opt.step()
    scheduler.step()
```

| 极简代码（手写） | PyTorch 标准写法 | 说明 |
|---|---|---|
| `rms_norm(x, gamma)` | `nn.RMSNorm(width, eps=1e-6)` | 公式相同；PyTorch 2.4 起自带。也有 `nn.LayerNorm` |
| `torch.randn(...) * std` | `nn.init.normal_(p, std=0.02)`，输出投影用 `0.02/√(2L)` | 脚本里的 `init_llm_style` 是大模型的常见写法 |
| 手写 m、v、偏差修正、解耦衰减 | `torch.optim.AdamW` | 更新公式完全一样；GPU 上还有融合内核 |
| `if p.dim() >= 2` 才衰减 | 两个参数组，`weight_decay` 分别设 0.1 和 0 | 优化器按组管理超参 |
| `warmup_cosine` / `wsd` 函数 | `LambdaLR(opt, 倍率函数)`，每步调用 `scheduler.step()` | 调度器的状态要和 checkpoint 一起保存（第 14 章） |
| `clip_by_global_norm` | `torch.nn.utils.clip_grad_norm_` | 返回裁剪前的范数，训练日志里常记录它 |

**对拍**：脚本把 06 手写版的初始权重搬进 PyTorch 版。两者用同一条数据流跑 200 步，逐步比较损失：

| 精度 | 逐步损失的最大差 | 第 200 步损失（PyTorch / 手写） |
|---|---:|---|
| float64 | 8.9 × 10⁻¹⁶ | 0.9041 / 0.9041 |
| float32 | 1.3 × 10⁻² | 0.8676 / 0.8659 |

双精度下两者逐位一致，说明公式完全相同。单精度下，两边的运算顺序略有不同。每步产生 10⁻⁷ 量级的舍入差异，训练过程把它逐步放大。200 步后，差到 10⁻²。（复跑是 3 × 10⁻⁴。最终放大到多少，取决于机器和 PyTorch 选用的内核。）

这本身也是一堂"训练稳定性"课。训练是一个会放大微小扰动的动力系统。所以对拍要用高精度。"固定随机种子就能复现"也只在同一台机器、同一套代码上成立。（两种精度下的随机数据不同，所以两行的损失值本身不同。）

脚本最后用大模型风格的 std = 0.02 初始化，把全套完整训练 800 步。余弦和 WSD 的验证损失都在 0.80 左右（构建机上是 0.80028 和 0.80029，复跑是 0.80101 和 0.79763）。差别在机器间的随机波动之内，分不出哪个更好。

**在主线代码里**：从第 7 章起，生产级代码进入 `zero/`。本章这些技巧在主线里的位置如下，第 14 章会详细讲：

- `zero/model.py`：`RMSNorm`（与 Hugging Face 的 Qwen3 实现一致：先转 float32，再归一化）。`Transformer.init_weights()` 用 `N(0, init_std)` 初始化，再把两个写回残差流的投影除以 `sqrt(2 · n_layers)`。
- `zero/train/schedule.py`：`cosine_with_warmup`、`wsd`（衰减形状支持线性、余弦和 1−sqrt）、`LRScheduler`（把倍率写进优化器，并把状态存进 checkpoint）。
- `zero/train/trainer.py`：`build_optimizer` 按"矩阵 / 一维参数"分两组。一个配置项决定嵌入是否衰减（对应 OLMo 2 的做法）。`Trainer.train` 每步调用 `clip_grad_norm_`，同时把梯度范数写进日志。

---

## 前沿观察

**Muon 优化器**：Muon 对每个矩阵的更新做正交化（让更新矩阵的奇异值都相等）。Kimi K2 报告引用其前作 Moonlight 的实验：同样的算力和模型大小下，Muon 明显优于 AdamW。Kimi K2 用改进版 MuonClip 预训练了 15.5T token。GOAL.md 把 Muon 列为"待核实（可能已达共识）"。

本章核实过这些主力开源模型：Llama、Qwen3、DeepSeek-V3、OLMo 2、SmolLM3。截至本章写作时，它们的公开报告仍然用 AdamW。所以本章正文只讲 AdamW。Muon 是否满足"3 个独立家族采用"，留待第 12 章核实。

---

## 采用方与来源

以下都是技术报告、模型卡或官方 `config.json` 里的明确表述。

| 技术 | 采用方（至少 3 个家族） | 来源 |
|---|---|---|
| RMSNorm | Llama 2（"pre-normalization using RMSNorm"）；Qwen3（"RMSNorm with pre-normalization"）；gpt-oss（"root mean square normalization"）；OLMo 2（从非参数 LayerNorm 换成 RMSNorm）；DeepSeek-V3、Gemma 3 的 config 里有 `rms_norm_eps` | [Llama 2 §2.2](https://arxiv.org/abs/2307.09288)、[Qwen3 §2](https://arxiv.org/abs/2505.09388)、[gpt-oss 模型卡 §2.2](https://arxiv.org/abs/2508.10925)、[OLMo 2 §2.1](https://arxiv.org/abs/2501.00656)、[DeepSeek-V3 config](https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/config.json)、[Gemma 3 1B config](https://huggingface.co/google/gemma-3-1b-it/blob/main/config.json) |
| Pre-Norm | Llama 2；Qwen3；gpt-oss（"Pre-LN placement"）。反例：OLMo 2 把 norm 放在子层输出上 | 同上；[Xiong et al. 2020](https://arxiv.org/abs/2002.04745) |
| AdamW | Llama 2（β = 0.9/0.95，λ = 0.1）；Llama 3；DeepSeek-V3（β = 0.9/0.95，λ = 0.1）；OLMo 2；SmolLM3（β = 0.9/0.95，λ = 0.1）。反例：Kimi K2 用 MuonClip | [Llama 2](https://arxiv.org/abs/2307.09288)、[Llama 3 §3.4.1](https://arxiv.org/abs/2407.21783)、[DeepSeek-V3 §4.2](https://arxiv.org/abs/2412.19437)、[OLMo 2 §3.4](https://arxiv.org/abs/2501.00656)、[SmolLM3 博客](https://github.com/huggingface/blog/blob/main/smollm3.md)、[Kimi K2 §2.1](https://arxiv.org/abs/2507.20534) |
| 嵌入 / 一维参数不做权重衰减 | OLMo 2（嵌入不衰减）；SmolLM3（沿用 OLMo 2）；nanoGPT（一维参数不衰减） | [OLMo 2 §3.4.2](https://arxiv.org/abs/2501.00656)、[SmolLM3 博客](https://github.com/huggingface/blog/blob/main/smollm3.md)、[nanoGPT model.py](https://github.com/karpathy/nanoGPT/blob/master/model.py) |
| warmup + 余弦衰减 | Llama 2（2000 步 warmup，降到峰值的 10%）；Llama 3（8000 步 warmup，降到 8e−7）；OLMo 2（2000 步 warmup，降到 10%） | [Llama 2 §2.2](https://arxiv.org/abs/2307.09288)、[Llama 3 §3.4.1](https://arxiv.org/abs/2407.21783)、[OLMo 2 §2.3](https://arxiv.org/abs/2501.00656) |
| WSD（warmup-恒定-衰减） | MiniCPM（提出者）；SmolLM3（最后 10% 线性降到 0）；Kimi K2（报告明确写 WSD）；DeepSeek-V3（同一形状：恒定到 10T token 后余弦衰减，没有用 WSD 这个名字） | [MiniCPM §4](https://arxiv.org/abs/2404.06395)、[SmolLM3 博客](https://github.com/huggingface/blog/blob/main/smollm3.md)、[Kimi K2 §2.5](https://arxiv.org/abs/2507.20534)、[DeepSeek-V3 §4.2](https://arxiv.org/abs/2412.19437) |
| 梯度裁剪（范数 1.0） | Llama 2；DeepSeek-V3；OLMo 2；SmolLM3 | [Llama 2](https://arxiv.org/abs/2307.09288)、[DeepSeek-V3 §4.2](https://arxiv.org/abs/2412.19437)、[OLMo 2 表 3](https://arxiv.org/abs/2501.00656)、[SmolLM3 博客](https://github.com/huggingface/blog/blob/main/smollm3.md) |
| 初始化 std 0.02 | Qwen3、Gemma 3、SmolLM3 的 config 写 `initializer_range: 0.02`；OLMo 2 报告明确写 0.02。注意：DeepSeek-V3 的 config 写 0.02，报告正文写的是 0.006 | [Qwen3-0.6B config](https://huggingface.co/Qwen/Qwen3-0.6B/blob/main/config.json)、[Gemma 3 1B config](https://huggingface.co/google/gemma-3-1b-it/blob/main/config.json)、[SmolLM3-3B config](https://huggingface.co/HuggingFaceTB/SmolLM3-3B/blob/main/config.json)、[OLMo 2 §3.2](https://arxiv.org/abs/2501.00656) |

待核实：Qwen3 技术报告只写了"第二阶段加快学习率衰减"，没有公布具体的调度形状。所以 Qwen3 没有列进 warmup + 余弦或 WSD 的采用方。Gemma 3 的 norm 位置，本章没有核实。

---

## 引导问题

向 Claude Code 提出这些问题。一直问到你能用自己的话讲清楚答案：

1. 看 01 的表里 std = 1.0 和 std = 0.01 两种初始化的"第 1 层 / 第 30 层误差信号比"。两个比值的量级正好互为倒数。ReLU 网络有什么性质，使得只改权重尺度时，整张表等比例缩放？
2. Kaiming 初始化里的"2"来自 ReLU。如果激活函数换成 tanh，这个系数应该是多少？为什么？（提示：tanh 在 0 附近的斜率是 1。）
3. RMSNorm 不减均值。在什么样的输入上，RMSNorm 和 LayerNorm 的输出差别最大？构造一个例子，让 `02_normalization.py` 打印出来。
4. 03 的实验里，普通堆叠的网络尺度正常，误差信号的大小也正常，却仍然"学不动"。"梯度大小正常"和"梯度有用"是一回事吗？
5. Adam 的第一步为什么让每个参数都正好走 η？这和 warmup 有什么关系？
6. WSD 的衰减阶段只占 10%–20% 的步数，却能把损失降下一大截。请解释原因。结合第 1 章的画面："学习率太大时，参数在谷底附近来回跳"。

## 动手任务

每个任务都要运行代码，并查看结果。

**任务 1（基础）**：在 `01_signal_propagation.py` 里，把激活函数从 ReLU 换成 tanh。重跑五种初始化。现在哪一种最接近"每层不放大也不缩小"？和引导问题 2 的答案对得上吗？

**任务 2（核心）**：在 `06_ablation.py` 里，把 `BLOCKS` 从 12 改成 24（48 个线性层）。重跑两张表。哪些配置的结论变了？残差和 RMSNorm 的重要性是变大还是变小？（运行时间会翻倍，约 3 分钟。）

**任务 3（挑战）**：`05_lr_schedule.py` 里的 WSD 用线性衰减。Hägele et al.（2024）发现，`1 − sqrt(进度)` 形状的衰减更好。在 `wsd` 函数里加一个 `shape` 参数来实现它。再用 `07_schedule_experiments.py` 的实验，比较"线性"和"1 − sqrt"两种衰减的验证损失。差别有论文里说的那么明显吗？如果没有，可能是什么原因？

---

## 读完第一部分，就可以开始 CS336

到这里，第一部分"从一条直线开始"就结束了。我们从 `y = ax + b` 出发，走过了矩阵、神经网络、反向传播、softmax 与交叉熵。最后在这一章，我们把深网络训稳了。这正是斯坦福 [CS336: Language Modeling from Scratch](https://cs336.stanford.edu/)（Spring 2026）默认你已经掌握的前置知识。CS336 第一讲从分词开始。作业 1 要求你从零写出 BPE、Transformer、AdamW 和训练循环。

本章手写的 RMSNorm、AdamW、余弦调度和梯度裁剪，会在那里原样再出现一次。从第 7 章起，本课每章末尾都有一节"想深入：CS336"，指向对应的讲座和作业。你可以跟着本课往下读。也可以现在就打开 CS336 的课程页，两边对照着学。

---

## 本章参考文献

- Glorot & Bengio. *Understanding the difficulty of training deep feedforward neural networks*（Xavier 初始化），AISTATS 2010：<https://proceedings.mlr.press/v9/glorot10a.html>
- He et al. *Delving Deep into Rectifiers*（Kaiming 初始化），2015：<https://arxiv.org/abs/1502.01852>
- He et al. *Deep Residual Learning for Image Recognition*（残差连接），2015：<https://arxiv.org/abs/1512.03385>
- Ba, Kiros, Hinton. *Layer Normalization*，2016：<https://arxiv.org/abs/1607.06450>
- Zhang & Sennrich. *Root Mean Square Layer Normalization*，NeurIPS 2019：<https://arxiv.org/abs/1910.07467>
- Xiong et al. *On Layer Normalization in the Transformer Architecture*（Pre-LN 与 warmup），ICML 2020：<https://arxiv.org/abs/2002.04745>
- Kingma & Ba. *Adam: A Method for Stochastic Optimization*，2014：<https://arxiv.org/abs/1412.6980>
- Loshchilov & Hutter. *Decoupled Weight Decay Regularization*（AdamW），2017：<https://arxiv.org/abs/1711.05101>
- Loshchilov & Hutter. *SGDR: Stochastic Gradient Descent with Warm Restarts*（余弦调度），2016：<https://arxiv.org/abs/1608.03983>
- Pascanu, Mikolov, Bengio. *On the difficulty of training recurrent neural networks*（梯度裁剪），2012：<https://arxiv.org/abs/1211.5063>
- Zhang et al. *Why Transformers Need Adam: A Hessian Perspective*，NeurIPS 2024：<https://arxiv.org/abs/2402.16788>
- Hu et al. *MiniCPM: Unveiling the Potential of Small Language Models with Scalable Training Strategies*（WSD），2024：<https://arxiv.org/abs/2404.06395>
- Hägele et al. *Scaling Laws and Compute-Optimal Training Beyond Fixed Training Durations*（恒定学习率 + 冷却），NeurIPS 2024：<https://arxiv.org/abs/2405.18392>
- OLMo Team. *2 OLMo 2 Furious*（初始化、norm 位置、嵌入不衰减等稳定性细节），2024：<https://arxiv.org/abs/2501.00656>
- Karpathy. nanoGPT（`configure_optimizers` 的参数分组、残差投影的缩放初始化、`get_lr` 余弦调度）：<https://github.com/karpathy/nanoGPT>
- 各模型的技术报告与配置：见上方"采用方与来源"表
- CS336（Spring 2026）：<https://cs336.stanford.edu/>

**下一章**：第一部分到此结束。我们手里已经有了一套能把深网络训稳的工具。第二部分开始构建真正的语言模型。第一个问题是：模型只认识数字，文字要怎样变成数字？按字、按词，还是按字节？第 7 章，我们从语言建模和分词讲起，亲手写一个 BPE 分词器。
