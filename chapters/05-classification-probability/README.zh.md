# 第 5 章：分类与概率 —— 从"预测一个数"到"预测哪一类"

[English](README.md) · **中文**

> **目标**：读完这一章，你能手写 softmax 和交叉熵（cross-entropy），并推出两者合起来的梯度就是 `p − onehot`。你能用数字说明分类为什么不用 MSE。你还能解释"语言模型就是一个在词表上做分类的分类器"。

📺 **本章视频**：还没有发布。要在本机渲染，运行 `bash chapters/05-classification-probability/video/build.sh`。
🧪 **本章自检**：学完后，在 Claude Code 里输入 `/ch05-classification`。

---

上一章我们亲手写了自动微分（automatic differentiation）。不管网络怎样搭，只要能写出前向传播，autograd 就能自动算出梯度。但到目前为止，我们的模型都只输出**一个数**：房价，或者曲线上的 y 值。损失也一直是均方误差。

这一章要解决的问题是：**如果答案不是一个数，而是"哪一类"，该怎么办？** 这张图是猫、狗还是鸟？这封邮件是不是垃圾邮件？还有和这门课关系最大的一个问题：**这句话的下一个字是什么？**

这一章讲完，大语言模型的训练目标就完整地出现了。后面还有二十多章，内容会复杂得多。但模型最后一层做的事，都是这一章的内容。

## 1. 直觉：为什么不能直接"预测类别编号"

最偷懒的做法是：把猫、狗、鸟编号成 0、1、2，然后像第 1 章那样用回归（regression）预测这个数。

这样做马上会出现问题：

- **编号带来了不存在的顺序。** 预测值 1.5 是什么意思？"介于狗和鸟之间"？如果模型在猫和鸟之间拿不准，回归会输出它们的平均值 1。而 1 恰好是"狗"。
- **我们想要的是"有多确定"。** "70% 是猫、26% 是狗、4% 是鸟"比一个编号有用得多。我们可以设阈值，可以排序，还可以在第 10 章里按概率采样出下一个字。

所以换一个思路：**有几类，模型就输出几个数，每类一个分数**。然后把这组分数变成一个概率分布。

## 2. logits 与 softmax：把分数变成概率

模型的最后一层（第 2、3 章的 `h @ W + b`）为每一类输出一个任意实数。这些数叫作 **logits**。logit 可正可负，加起来也不等于 1。贯穿全章的例子是：

```
logits z = [2.0, 1.0, −1.0]    # 猫、狗、鸟
```

要把 logits 变成概率，需要满足两个条件：每个数都是正的，并且加起来等于 1。**softmax** 用最直接的办法满足这两点。先取指数，结果一定为正；再除以总和，结果一定归一：

```
p_k = exp(z_k) / Σ_j exp(z_j)
```

运行 [`code/01_softmax.py`](code/01_softmax.py)：

```bash
uv run python chapters/05-classification-probability/code/01_softmax.py
```

| | 猫 | 狗 | 鸟 |
|---|---:|---:|---:|
| logits `z` | 2.0 | 1.0 | −1.0 |
| `exp(z)` | 7.3891 | 2.7183 | 0.3679 |
| softmax `p` | 0.7054 | 0.2595 | 0.0351 |

请记住下面几个性质：

- **保序**：分数越高，概率越大。softmax 是 argmax 的一个"软"版本。它不是只挑出最大的那一项，而是按指数的比例分配概率。这就是名字的由来。
- **只看差值**：给所有 logits 加同一个常数，结果不变。原因是 `exp(z + c) = exp(z)·exp(c)`，`exp(c)` 在分子和分母中约掉。代码给三个 logits 都加上 100，算出的结果还是 `[0.7054, 0.2595, 0.0351]`。
- **指数放大差距**：logits 差 1，概率差 e ≈ 2.7 倍；logits 差 3，概率差约 20 倍。

**数值稳定：先减最大值。** 指数增长得太快。在 float32 里，x 超过约 88.7 时，`exp(x)` 就溢出成 `inf`（float64 是 709.8）。训练大模型时，logits 达到几十、上百并不少见。把 logits 放大到 `[1000, 500, −500]`，按定义直接算会得到 `[nan, 0, 0]`。

解决办法就是上面"只看差值"那条性质。先把所有 logits 减去最大值。这样最大的那一项变成 `exp(0) = 1`，永远不会溢出：

```python
def softmax(z, temperature=1.0):
    z = np.asarray(z, dtype=np.float64) / temperature
    z = z - z.max(axis=-1, keepdims=True)   # subtract the maximum: same result, but no overflow
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)
```

对 `[1000, 500, −500]`，稳定版给出 `[1, 0, 0]`，结果正确。所有深度学习框架里的 softmax 都这样写。

## 3. 最大似然：让正确答案的概率尽量大

有了概率，训练要优化的目标就很自然了：**模型给正确答案的概率越大越好**。

对一个样本，正确类别是 y，模型给它的概率是 `p_y`。假设样本之间相互独立。那么对整个数据集，模型"猜中所有正确答案"的概率是每个样本概率的乘积：

```
似然 = Π_i p_{i, y_i}
```

把它看成参数的函数，找出让它最大的参数。这个方法叫作**最大似然估计**（maximum likelihood estimation，MLE）。

直接算乘积有一个实际问题：它会**下溢**（underflow）。代码算了一个例子，假设每个样本都给正确答案 0.9 的概率：

| 样本数 | 似然 0.9ⁿ | 对数似然 n·ln 0.9 |
|---:|---:|---:|
| 10 | 3.487 × 10⁻¹ | −1.05 |
| 100 | 2.656 × 10⁻⁵ | −10.54 |
| 1000 | 1.748 × 10⁻⁴⁶ | −105.36 |
| 10000 | 2.470 × 10⁻³²³ | −1053.61 |

10000 个样本时，乘积已经接近 float64 能表示的最小正数。而一个语言模型的训练集有上万亿个 token。所以要取对数：**log 把乘积变成求和**。log 单调递增，所以最大化似然和最大化对数似然是同一件事。再加一个负号，变成"越小越好"；再除以 N 取平均。这样就得到**负对数似然**（negative log-likelihood，NLL）：

```
L = −1/N · Σ_i log p_{i, y_i}
```

## 4. 交叉熵，以及它简洁的梯度

上面这个 NLL 有一个更常用的名字：**交叉熵**。一般地，真实分布 q 和模型分布 p 的交叉熵是 `H(q, p) = −Σ_k q_k log p_k`。在分类任务里，真实分布是 onehot：正确类别为 1，其余为 0。这时求和里只剩下一项 `−log p_y`，正好就是 NLL。所以在分类里，**"最大似然"、"最小化负对数似然"、"最小化交叉熵"说的是同一件事**。

回到猫、狗、鸟的例子。下表列出不同正确答案下的损失：

| 正确答案 | 模型给它的概率 p | 损失 −ln p |
|---|---:|---:|
| 猫 | 0.7054 | 0.3490 |
| 狗 | 0.2595 | 1.3490 |
| 鸟 | 0.0351 | 3.3490 |
| （三类均匀地猜） | 1/3 | 1.0986 |

猜对且有把握，损失就小。正确答案的概率越低，损失越大，而且没有上限：p → 0 时，−ln p → ∞。最后一行也很有用：**一个什么都没学的模型，损失应该约等于 ln(类别数)**。这是检查训练代码有没有写错的第一个对拍点。后面螺旋数据的初始损失 1.0919 ≈ ln 3，语言模型的初始损失 ≈ ln(词表大小)，都是这个道理。

实现时同样要注意数值稳定。不要先算 softmax 再取 log：概率下溢成 0 时，log 会得到 −∞。应该直接算 **log-softmax**：`log p_k = z_k − logsumexp(z)`，同样先减最大值：

```python
def log_softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=-1, keepdims=True))

def cross_entropy(logits, y):
    return -log_softmax(logits)[np.arange(len(y)), y].mean()
```

**梯度：`p − onehot`。** 这是全章最重要的一个结论。对一个样本，把 softmax 代入损失：

```
L = −log p_y = −z_y + log Σ_j exp(z_j)
```

对第 k 个 logit 求导。第一项只在 k = y 时贡献 −1。第二项的导数是 `exp(z_k) / Σ_j exp(z_j)`，正好是 `p_k`。所以：

```
∂L/∂z_k = p_k − [k = y]      即   ∂L/∂z = p − onehot(y)
```

求导时，softmax 和 log 的复杂部分全部互相抵消了。剩下的是"模型的预测减去正确答案"。它和第 1 章 MSE 的残差 `ŷ − y` 形式相同。代码里只有一行：

```python
def ce_grad(logits, y):
    p = softmax(logits)
    p[np.arange(len(y)), y] -= 1.0      # p − onehot
    return p / len(y)                  # divide by N because the loss is a mean
```

用第 1、4 章的方法，以中心差分做**梯度检验**（gradient check）（[`code/02_cross_entropy.py`](code/02_cross_entropy.py)）：

| | 猫 | 狗 | 鸟 |
|---|---:|---:|---:|
| 解析梯度 `p − onehot`（正确答案 = 猫） | −0.2946 | 0.2595 | 0.0351 |
| 数值梯度（中心差分） | −0.2946 | 0.2595 | 0.0351 |

最大差是 2.18 × 10⁻¹²。换成随机的 8 个样本 × 5 类，最大差是 3.63 × 10⁻¹¹。推导是对的。

读一下这个梯度。梯度下降把正确类别的 logit 往上推：梯度为负，减去它就是增大。推的力度是 `1 − p_y`，也就是离 1 还差多少。梯度下降把每个错误类别的 logit 往下压，力度正是模型错给这个类别的概率 `p_k`。

## 5. 为什么分类不用 MSE

有了概率以后，能不能继续用 MSE，让 `p` 去逼近 onehot，也就是 `L = Σ_k (p_k − t_k)²`？可以算，也能训练。但它有一个严重的弱点：**模型"自信地错了"的时候，MSE 几乎不给梯度。**

做一个实验。真实类别是 0。让模型把错误类别 1 的 logit 抬到 s，其余 logit 为 0。s 越大，模型越确信那个错误答案。下表是两种损失对 logits 的梯度大小（`02_cross_entropy.py` 第 4 部分）：

| 错误类别 logit s | p(正确) | CE 损失 | ‖CE 梯度‖ | MSE 损失 | ‖MSE 梯度‖ |
|---:|---:|---:|---:|---:|---:|
| 0 | 3.33 × 10⁻¹ | 1.099 | 0.8165 | 0.6667 | 5.44 × 10⁻¹ |
| 2 | 1.07 × 10⁻¹ | 2.240 | 1.1954 | 1.4290 | 5.08 × 10⁻¹ |
| 4 | 1.77 × 10⁻² | 4.036 | 1.3769 | 1.8959 | 1.23 × 10⁻¹ |
| 6 | 2.47 × 10⁻³ | 6.005 | 1.4090 | 1.9852 | 1.83 × 10⁻² |
| 8 | 3.35 × 10⁻⁴ | 8.001 | 1.4135 | 1.9980 | 2.51 × 10⁻³ |
| 10 | 4.54 × 10⁻⁵ | 10.000 | 1.4141 | 1.9997 | 3.40 × 10⁻⁴ |

两列梯度的变化方向正好相反：

- **交叉熵**：错得越多，梯度越大，最后稳定在 √2 ≈ 1.414（就是 `p − onehot` ≈ `[−1, 1, 0]` 的长度）。损失也随 s 线性增长，模型一直"感到疼"。
- **MSE**：错得越多，梯度反而越小。s = 10 时梯度只剩 3.4 × 10⁻⁴，比交叉熵小 4000 多倍。MSE 损失的上限是 2（两个概率各差 1）。到了上限，损失曲面是一片平地。

原因在链式法则里。MSE 的梯度要穿过 softmax 的导数 `∂p/∂z = diag(p) − ppᵀ`。这个矩阵里每一项都乘着某个 `p_k`。当正确类别的概率已经接近 0 时，这个因子让梯度也接近 0。

这和 sigmoid 在两端饱和（saturation）、导数趋于 0 是同一类现象。（第 3 章见过 sigmoid 的形状。）交叉熵里的 `log` 正好抵消了 softmax 的指数。于是梯度就是 `p − onehot`，不会饱和。

> **注意：**本章训练类实验的数字来自课程构建机上的一次 CPU 运行。不同机器、不同版本的底层数学库，浮点运算的顺序略有不同。训练几百步会把这些微小差异放大。你在本机跑出的数字，可能从小数点后第二、三位开始就不一样。请以下文中不依赖具体数值的结论为准。2026-10 在另一台服务器上的复跑对照见 [runs/2026-10-01-gpu0-check/chapters-01-06.md](../../runs/2026-10-01-gpu0-check/chapters-01-06.zh.md)。

这个现象不只出现在一张表里。在第 6 节的螺旋分类器上，我们故意把输出层权重的标准差放大到 10，让模型一开始就"自信地乱猜"。然后分别用交叉熵和 MSE 训练（同一个网络，同样的学习率 1.0）。

下表前三列是课程构建机上的输出。最后一列是 2026-10 在另一台服务器上用同一份代码复跑的 MSE 结果。那次复跑中，交叉熵一列和构建机每一位都相同。

| 步数 | 交叉熵训练：准确率 | MSE 训练：准确率 | MSE 训练：准确率（另一台服务器复跑） |
|---:|---:|---:|---:|
| 0 | 30.7% | 30.7% | 30.7% |
| 100 | 85.7% | 65.3% | 53.3% |
| 500 | 97.3% | 65.3% | 96.7% |
| 800 | 98.0% | 66.0% | 99.3% |
| 1000 | 99.0% | 95.3% | 99.3% |
| 3000 | 99.3% | 99.3% | 99.0% |

在构建机上，MSE 版在 65% 附近**卡住了 700 到 800 步**。训练 500 步时，交叉熵版里"给正确答案的概率不到 1%"的样本是 0 个，MSE 版有 103 个。三分之一的数据自信地错着，却几乎推不动参数。

换到另一台服务器上，MSE 版卡住的时间短得多。100 步时准确率只有 53.3%，有 136 个样本 p(正确) < 1%。但 200 到 300 步后它就走了出来，500 步时这样的样本只剩 7 个（交叉熵版仍是 0 个）。

这本身就是一课：**小实验对数值细节很敏感**。两台机器的矩阵乘法只在舍入上有微小差别。可是 MSE 在饱和区的梯度被压到接近 0，哪些样本先"翻过来"就由这点舍入差决定。所以卡住的时间从 700–800 步变成了 200–300 步。交叉熵的梯度不饱和，它在两台机器上的轨迹每一位都相同。

不要把"卡了多少步"当成结论。两次运行都成立的结论有两个。第一，交叉熵从第一步就稳定地推动参数。第二，MSE 起步明显更慢：最初有一大批样本自信地错着，却推不动参数。这正是上面那张梯度表描述的情形。

两次运行中，MSE 最后都走了出来（这个问题的规模很小）。但在大模型上，你付不起这几百步。

如果初始化正常（初始预测接近均匀），两者都能学会。3000 步后准确率都是 99.3%，只是 MSE 版的交叉熵高一点（0.0490 vs 0.0293）。所以更准确的说法是：**MSE 在分类上不是"不能用"，而是在最需要纠正的时候推动力最弱**。

Golik 等人 2013 年在语音识别上做过系统的对比，结论也是：随机初始化时，用平方误差训练的网络收敛不到好的解。另外，交叉熵有"最大似然"这个清楚的概率解释。因此交叉熵成了分类的默认损失。

## 6. 训练一个分类器：三条螺旋

现在把所有部件组装起来。数据是三条交错的螺旋臂（CS231n 的经典玩具数据），每类 100 个点。数据在代码里生成，不用下载。

直线切不开这些螺旋，所以模型用第 3 章的两层 MLP：2 → 64（ReLU）→ 3。反向传播照第 4 章手写，只是最后一层的梯度换成了 `p − onehot`：

```python
logits, cache = forward(params, X)             # z = ReLU(X W1 + b1) W2 + b2
grads = backward(params, cache, ce_grad(logits, y))   # start the backward pass from dlogits = (p − onehot)/N
for k in params:
    params[k] -= lr * grads[k]
```

运行 [`code/03_train_classifier.py`](code/03_train_classifier.py)（CPU 上几秒）：

```bash
uv run python chapters/05-classification-probability/code/03_train_classifier.py
```

| 步数 | 交叉熵 | 训练准确率 |
|---:|---:|---:|
| 0 | 1.0919 | 30.7% |
| 100 | 0.2641 | 93.0% |
| 200 | 0.1513 | 96.3% |
| 500 | 0.0813 | 99.0% |
| 1000 | 0.0526 | 99.3% |
| 3000 | 0.0293 | 99.3% |

- **初始损失 1.0919 ≈ ln 3 = 1.0986。** 输出层初始化得很小，模型一开始几乎在均匀地猜。这和第 4 节的预期一致。
- **另造一份测试数据**（同样的螺旋，不同的随机噪声），准确率是 99.0%。
- **对照：去掉隐藏层。** 只用线性的 softmax 分类器（`logits = XW + b`，也叫多项逻辑回归），3000 步后准确率只有 54.0%。每两类之间的边界都是直线，切不开螺旋。分类问题同样需要第 3 章的非线性。

视频展示了这个过程。平面上每一点都涂成模型预测的类别的颜色。一开始，几乎整个平面是同一种颜色。随着训练，三种颜色的区域慢慢卷成三条螺旋。

## 7. 温度：让分布更尖或更平

softmax 还有一个调节参数：先把 logits 除以一个**温度**（temperature）T，再做 softmax。

| 温度 T | 猫 | 狗 | 鸟 |
|---:|---:|---:|---:|
| 0.5 | 0.8789 | 0.1189 | 0.0022 |
| 1.0 | 0.7054 | 0.2595 | 0.0351 |
| 2.0 | 0.5465 | 0.3315 | 0.1220 |
| 10.0 | 0.3780 | 0.3420 | 0.2800 |

T 越小，差距越大，分布越"尖"。T → 0 时，softmax 就是 argmax。T 越大，差距越小，分布越"平"。T → ∞ 时，它变成均匀分布。

训练时 T 一般取 1。温度真正发挥作用是在第 10 章：语言模型生成文字时，温度决定是"总挑最可能的字"还是"更大胆地尝试"。

## 8. 语言模型就是一个在词表上做分类的分类器

这是本章最重要的一节。**语言模型做的事，就是"看前面的字，猜下一个字"**。把"下一个字"当成类别，这就是一个分类问题，类别数就是**词表**（vocabulary）的大小 V。

模型最后一层对每个候选 token 输出一个 logit。softmax 给出下一个 token 的概率分布。损失就是正确的下一个 token 的交叉熵。整个过程就是这些。

[`code/04_next_token.py`](code/04_next_token.py) 用一小段中文演示这件事。每个汉字是一个 token。模型是最简单的一种：一张 V × V 的表，"当前字"那一行就是下一个字的 logits（第 7 章会正式讲这个 **bigram** 模型）。训练用的正是上面的 `cross_entropy` 和 `ce_grad`，一行都没改：

```bash
uv run python chapters/05-classification-probability/code/04_next_token.py
```

语料有 104 个字，词表 V = 65，训练样本 103 对：

| 步数 | 损失（nats） | 损失（bits） | 困惑度 |
|---:|---:|---:|---:|
| 0 | 4.1744 | 6.0224 | 65.00 |
| 10 | 0.9503 | 1.3709 | 2.59 |
| 50 | 0.5027 | 0.7253 | 1.65 |
| 500 | 0.4622 | 0.6668 | 1.59 |

对照：直接数频率，`p(下一个字 | 当前字) = 出现次数 / 总次数`，得到的损失是 0.4585。梯度下降正在逼近这个值。**对 bigram 来说，"数频率"就是最大似然的解析解**，就像第 1 章的最小二乘解。模型学到的东西也和数频率的结果一致：「分」后面是「数」的概率是 0.67，是「类」的概率是 0.33。原因是语料里"分数"出现了两次，"分类"出现了一次。

表里有三种单位，表示的是同一个量：

- **nats**：用自然对数 ln 算的交叉熵，就是我们一直在算的数。
- **bits**：换成以 2 为底，`bits = nats / ln 2`。它的含义是"平均每个 token 还需要多少个二进制位来描述"。损失越低，模型压缩得越好。第 7 章会用一个相关的指标 **bits-per-byte**（BPB）。它把 bits 平摊到原始文本的每个字节上，这样用不同分词器的模型也能公平比较。
- **困惑度**（perplexity）：`exp(nats) = 2^bits`。直觉是"模型相当于在几个候选里均匀地犹豫"。第 0 步困惑度是 65，正好等于词表大小，相当于在 65 个字里随机地猜。训练后困惑度是 1.59，相当于每一步只在一两个候选之间犹豫。

换到真实规模：GPT-2 的词表有 50257 个 token。一个未经训练的模型，损失应该在 ln 50257 = 10.82 nats（15.62 bits）附近。以后你看训练日志时，如果第一步的 loss 在 10.8 左右，就知道模型在"均匀地猜"，代码大概率没写错。

## 9. 小结

- **logits**：模型给每一类一个任意实数分数。
- **softmax**：`p_k = exp(z_k)/Σexp(z_j)`，把分数变成概率分布。实现时先减最大值。
- **最大似然**：让正确答案的概率尽量大。取 log 把连乘变成求和，得到**负对数似然 = 交叉熵** `−log p_y`。
- **梯度**：`∂L/∂z = p − onehot`，预测减答案。
- **不用 MSE**：自信地错时，MSE 的梯度会消失，交叉熵的梯度不会。
- **温度**：logits 除以 T，控制分布的尖锐程度。
- **语言模型**：在词表上做分类，损失就是交叉熵。nats、bits、困惑度是同一个量的三种写法。

---

## 从极简代码到生产级代码

第 1–6 章的生产级代码就是 PyTorch 的标准写法。见 [`code/05_pytorch_version.py`](code/05_pytorch_version.py)：

```python
model = nn.Sequential(nn.Linear(2, 64), nn.ReLU(), nn.Linear(64, 3))
optimizer = torch.optim.SGD(model.parameters(), lr=1.0)

for step in range(3000):
    logits = model(X)                    # 1. forward pass: get the logits (no softmax!)
    loss = F.cross_entropy(logits, Y)    # 2. cross-entropy: log-softmax + NLL, fused inside
    optimizer.zero_grad()                # 3. set the gradients to zero
    loss.backward()                      # 4. backward pass: autograd gets (p − onehot)/N
    optimizer.step()                     # 5. update
```

运行结果（全部用 float64 对拍）：

| 对拍项 | PyTorch | 我们的 NumPy 版 |
|---|---:|---:|
| `F.cross_entropy`，随机 8 样本 × 5 类 | 3.1764553511 | 3.1764553511 |
| 梯度：autograd vs 手写 `(p − onehot)/N` | 最大差 6.94 × 10⁻¹⁸ | — |
| `label_smoothing=0.1` vs 手写 `−Σ q_k log p_k` | 3.1633053590 | 3.1633053590 |
| 螺旋分类器 3000 步后的交叉熵（同一初始化） | 0.0293 | 0.0293（差 7.85 × 10⁻¹¹） |
| logits `[1000, 500, −500]`、正确答案 = 第 3 类 | 1500.0 | 朴素写法 `−log(softmax)` 得到 `inf` |

下表列出生产级代码比极简代码多做了什么，以及原因：

| 极简代码 | 生产级代码 | 原因 |
|---|---|---|
| `softmax` 再 `−log`（或手写 `log_softmax`） | `F.cross_entropy(logits, y)` 直接接收 logits | 它把 log-softmax 和 NLL 融合成一个算子，用 logsumexp 保证数值稳定。**最常见的 bug 是先做了 softmax 再传进去。** 这个 bug 不会报错，只会让训练悄悄地变差。 |
| 手写 `ce_grad` | `loss.backward()` | autograd 自动得到同样的 `p − onehot`（上表中逐位一致）。 |
| 目标是 onehot | `label_smoothing=ε` | 它把目标变成"正确类 1 − ε + ε/K，其余各类 ε/K"。这样可以防止模型把 logit 推到无穷大、过度自信（Szegedy 等 2016）。它是可选项，本课主线模型的预训练不用它。 |
| 每个样本都算损失 | `ignore_index=-100` | 某些位置不算损失。第 16 章 SFT 的 loss mask 就靠它：只在回答部分算损失，不在提问部分算。 |
| logits 形状 `(N, K)` | 语言模型里 logits 的形状是 `(B, T, V)`，拉平成 `(B·T, V)` | 每条序列的每个位置都是一次分类。nanoGPT 和 minimind 的模型代码里都有这一行：`F.cross_entropy(logits.view(-1, V), targets.view(-1), ignore_index=...)` |
| float64 | 混合精度训练（BF16） | 矩阵乘法用 BF16。但 PyTorch 自动混合精度会把 `cross_entropy`、`log_softmax`、`softmax` 自动提升到 float32 计算，因为指数和对数对精度很敏感（第 14 章）。 |

脚本最后一段就是语言模型里的写法：logits 的形状是 `(2, 4, 50257)`，全为 0（均匀地猜）。拉平后算出的损失是 10.8249，正好等于 ln 50257。

---

## 引导问题

向 Claude Code 提出这些问题。一直问到你能用自己的话讲清楚答案：

1. 二分类时，softmax 只有两个 logits `z₁, z₂`。证明 `p₁ = sigmoid(z₁ − z₂)`。这说明 sigmoid 和 softmax 是什么关系？
2. 为什么 `F.cross_entropy` 要求传入 logits，而不是概率？如果你传入 softmax 之后的概率，会发生什么？（提示：这时 softmax 做了两次。试着改一下 `05_pytorch_version.py`，看看损失和准确率。）
3. 交叉熵 `H(q, p)` 和熵 `H(q)` 的差叫作 **KL 散度**（KL divergence）。在分类任务里 q 是 onehot，它的熵是多少？这说明最小化交叉熵和最小化 KL 散度是什么关系？（第 17 章的 logits 蒸馏和第 18 章的 DPO 会用到 KL 散度。）
4. 训练集上的交叉熵能降到 0 吗？要降到 0，logits 需要变成什么样？这和 label smoothing 想解决的问题有什么关系？
5. 困惑度 1.59 的直觉是"在 1.59 个候选里均匀地犹豫"。如果一个语言模型在中文上的困惑度是 20，在代码上是 3，这说明什么？为什么用不同分词器的模型不能直接比较困惑度？（这是第 7 章引入 bits-per-byte 的动机。）

## 动手任务

每个任务都要运行代码，并查看结果。

**任务 1（基础）**：在 `01_softmax.py` 里验证：T → 0 时，softmax 趋近于 argmax 的 onehot；T → ∞ 时，softmax 趋近于均匀分布。分别取 T = 0.01 和 T = 1000，打印结果。再解释为什么 `softmax(z / T)` 里 T 不能取 0。

**任务 2（核心）**：在 `03_train_classifier.py` 里，把类别数改成 5（`make_spirals(n_classes=5)`，网络输出也改成 5）。先预测初始损失应该是多少，再运行代码验证。然后记录训练 3000 步后的准确率。如果准确率明显变差，试试加大隐藏层或增加训练步数。

**任务 3（挑战）**：`02_cross_entropy.py` 里的 MSE 对照是"softmax 概率 vs onehot"。另一种常见的错误做法是"直接拿 logits 去回归 onehot"（不经过 softmax，`L = Σ(z_k − t_k)²`）。在螺旋数据上用这种损失训练，看准确率能到多少。然后用本章的概念解释：它的问题和"softmax + MSE"的问题是同一种吗？

---

## 本章参考文献

- Goodfellow, Bengio, Courville. *Deep Learning*，第 5.5 节"最大似然估计"、第 6.2.2 节"softmax 输出单元"：<https://www.deeplearningbook.org/contents/ml.html>、<https://www.deeplearningbook.org/contents/mlp.html>
- Stanford CS231n. *Putting it together: Minimal Neural Network Case Study*（三类螺旋数据、softmax 线性分类器 vs 两层网络）：<https://cs231n.github.io/neural-networks-case-study/>
- Golik, Doetsch, Ney. *Cross-Entropy vs. Squared Error Training: a Theoretical and Experimental Comparison*，Interspeech 2013：<https://www.isca-archive.org/interspeech_2013/golik13_interspeech.html>
- Szegedy et al. *Rethinking the Inception Architecture for Computer Vision*（label smoothing 的出处），2016：<https://arxiv.org/abs/1512.00567>
- Hinton, Vinyals, Dean. *Distilling the Knowledge in a Neural Network*（带温度的 softmax，第 17 章蒸馏会再用到），2015：<https://arxiv.org/abs/1503.02531>
- Radford et al. *Language Models are Unsupervised Multitask Learners*（GPT-2，词表 50257）：<https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf>
- PyTorch 文档 `torch.nn.functional.cross_entropy`（输入是 unnormalized logits；`label_smoothing`、`ignore_index` 参数）：<https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.cross_entropy.html>
- PyTorch 文档 *Automatic Mixed Precision*（"CUDA Ops that can autocast to float32" 列表含 `cross_entropy`、`log_softmax`、`softmax`）：<https://docs.pytorch.org/docs/stable/amp.html>
- [nanoGPT](https://github.com/karpathy/nanoGPT) 的 `model.py`、[minimind](https://github.com/jingyaogong/minimind) 的 `model/model_minimind.py`：语言模型的损失都是一行 `F.cross_entropy`

**下一章**：螺旋分类器只有两层，训练过程很顺利。可是如果把网络加深到十几层、几十层呢？你会发现损失要么完全不动，要么突然变成 NaN。本章 MSE 的"梯度消失"只是这类问题的一小部分。第 6 章介绍初始化、归一化、残差连接、AdamW 和学习率调度。这些方法让深层网络能够训练。
