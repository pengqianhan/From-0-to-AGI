# 第 8 章：注意力 —— 每个位置自己决定看哪里

[English](README.md) · **中文**

> **目标**：读完这一章，你能从"对前面的词取加权平均"出发，一步步推出缩放点积注意力 `softmax(QKᵀ/√d)V`。你能说清楚 Q、K、V、除以 √d、因果 mask、多头各自解决什么问题。你能写出每一步张量的形状。你还能亲手实现一个因果多头注意力，它的输出和 PyTorch 官方函数一致。

📺 **本章视频**：还没有发布。要在本机渲染，运行 `bash chapters/08-attention/video/build.sh`。
🧪 **本章自检**：学完后，在 Claude Code 里输入 `/ch08-attention`。

---

上一章我们把文字变成了 token，训练了一个 bigram 语言模型。我们用 bits-per-byte 衡量它猜得有多好。bigram 有一个根本的局限：**它猜下一个 token 时只看前一个 token**。这一章要解决的问题是：**怎样让模型看见整段前文，并且自己决定看哪里？** 答案是注意力（attention）。注意力是今天所有大语言模型的核心部件。

## 1. 问题：只看前一个 token 不够

想猜 `hear me speak` 后面是什么，只看最后一个字母 `k` 不够。第 7 章的 bigram 只记得"k 后面常跟什么"。而前面的 `hear me` 告诉我们：有人在说话，在请求什么。

难点在于：前文（上下文，context）的长度不固定。但后面的预测层（第 5 章的 softmax 分类器）只接受固定长度的向量。怎样把任意长的前文变成一个向量？

**以前的做法：循环神经网络（recurrent neural network，RNN）。** RNN 从左往右读。每读一个 token，就把它和上一步的状态合成新状态：`h_t = f(h_{t−1}, x_t)`。这里只把 RNN 当作动机来讲。它有两个问题，注意力正好解决这两个问题：

1. **状态的大小是固定的。** 前文再长，也要放进同样大小的 `h`，早期的信息容易丢失。Bahdanau 等人（2014）在机器翻译里指出：把整句压成一个固定长度的向量是瓶颈。
2. **只能一步一步算。** 第 10 个 token 必须等前 9 个算完。《Attention Is All You Need》（Vaswani 等人，2017）在开头指出：这种"天生的顺序性"（inherently sequential nature）让训练无法在序列内部并行。

注意力换了一个思路：**不压缩。每个位置直接看前面所有位置，按需取用。**

## 2. 最朴素的办法：把前面的向量取平均

先做最简单的事。设有 T 个 token，embedding 查表已经把每个 token 变成 C 维向量。把这些向量按行排成矩阵 `X`（形状 `(T, C)`）。让位置 t 的输出等于前 t+1 个向量的平均（包含位置 t 自己，不包含未来的位置）：

```
out_t = (x_0 + x_1 + … + x_t) / (t + 1)
```

有一个经典技巧能一次算完所有位置。把平均的系数排成一个**下三角矩阵**（lower-triangular matrix）`W`：第 t 行的前 t+1 个元素是 `1/(t+1)`，其余是 0。这样一次矩阵乘法 `W @ X` 就同时算出了所有位置的平均。第 2 章讲过，矩阵乘法就是"每一行和每一列做点积"。这里 `W` 的第 t 行正好是"前 t+1 个向量各取 1/(t+1)"。

```bash
uv run python chapters/08-attention/code/01_average_to_attention.py
```

[`code/01_average_to_attention.py`](code/01_average_to_attention.py) 用 5 个 2 维向量演示。我们假设它们是"我 爱 吃 苹 果"这 5 个 token：

```python
def uniform_weights(T):
    w = np.tril(np.ones((T, T)))                 # lower triangle of ones
    return w / w.sum(axis=1, keepdims=True)      # divide each row by its sum → each row sums to 1

mat = uniform_weights(T) @ x                     # one matrix multiplication = prefix averages at all positions
```

输出如下：

```
① Uniform weight matrix W (lower-triangular, each row sums to 1):
[[1.   0.   0.   0.   0.  ]
 [0.5  0.5  0.   0.   0.  ]
 [0.33 0.33 0.33 0.   0.  ]
 [0.25 0.25 0.25 0.25 0.  ]
 [0.2  0.2  0.2  0.2  0.2 ]]
② Max difference between W @ x and the loop version: 2.0e-17
③ Max difference between softmax(all-zero scores + causal mask) and W: 0.0e+00
```

请再看一下第③行。同一个 `W` 还可以用另一种方法得到。

先给每对位置一个分数（这里全是 0）。把上三角（未来）的分数填成 `−∞`。再对每一行做第 5 章的 softmax。因为 `e^{−∞} = 0`，所以未来位置的权重正好是 0。其余位置的分数相同，所以平分权重。

**"分数 → mask → softmax → 加权平均"这个骨架，就是注意力的全部结构。** 剩下的问题只有一个：分数怎样算？

## 3. 让权重由数据决定：点积相似度

平均的问题是：每个 token 的重要程度相同。在真实的文本里，有的 token 很关键，有的无关紧要。所以让数据决定分数：两个位置越"相关"，分数越高。

第 2 章讲过点积（dot product）的几何意义：两个向量的方向越一致，点积越大。所以最直接的分数就是 `x_t · x_s`。一次算出所有位置对，就是 `X @ Xᵀ`（形状 `(T, T)`）：

```python
def dot_product_weights(x):
    scores = x @ x.T                   # scores[t, s] = x_t · x_s
    return causal_softmax(scores)      # fill the upper triangle with −∞, then softmax on each row
```

输出（同一组 5 个向量）：

```
   Weights after causal mask + softmax (each row sums to 1):
[[1.   0.   0.   0.   0.  ]
 [0.41 0.59 0.   0.   0.  ]
 [0.28 0.23 0.48 0.   0.  ]
 [0.06 0.14 0.04 0.76 0.  ]
 [0.1  0.05 0.09 0.01 0.75]]
   5/5 positions give the largest weight to themselves: x·x = |x|² is often the largest, so we need two different projections, Q and K
```

权重不再均匀了，这一步的方向是对的。但出现了一个新问题：**每个位置最关注的都是自己**。原因是 `x·x = |x|²`：一个向量和自己的方向总是完全一致。可是预测下一个 token 时，我们常常要找的是"和我不一样、但对我有用"的 token。

## 4. Q、K、V：同一个输入，三个角色

解决办法是：给每个 token 准备三个不同的向量。三个向量都用第 2 章的线性变换 `y = XW` 得到：

```
Q = X W_q     query: what am I looking for?
K = X W_k     key: which features do I have, for others to match?
V = X W_v     value: what do I give you if you attend to me?
```

三个向量的含义：查询（query）表示"我在找什么"；键（key）表示"我有什么特征，供别人匹配"；值（value）表示"如果你关注我，我交给你什么"。

位置 t 对位置 s 的分数改为 **t 的查询和 s 的键的点积** `q_t · k_s`。加权平均的对象也从 `x` 换成 `v`。`W_q`、`W_k`、`W_v` 都是训练学到的参数（形状 `(C, C)`）。模型可以学会"元音去找前面的辅音""空格后的字母去找上一个词"这类匹配规则，而不再只能找自己。

打个比方：Q 是你在搜索框里输入的关键词，K 是每篇文章的标签，V 是文章的正文。关键词和标签越匹配，你从那篇文章拿走的内容越多。但这里的"匹配"是软的：每篇文章都拿一点，比例由 softmax 决定。

## 5. 缩放点积注意力，以及为什么除以 √d

一次算出所有查询和所有键的点积，结果就是 `Q Kᵀ`。整个注意力可以写成一行（Vaswani 等人 2017，式 1）：

```
Attention(Q, K, V) = softmax( Q Kᵀ / √d + mask ) V
```

`d` 是 q、k 的维度。代码只有几行（[`code/02_attention_from_scratch.py`](code/02_attention_from_scratch.py)）：

```python
def attention(q, k, v, causal=True):
    d = q.shape[-1]
    scores = q @ k.transpose(-2, -1) / math.sqrt(d)            # QKᵀ / √d
    if causal:
        T = q.shape[-2]
        future = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(future, float("-inf"))      # future positions → −∞
    weights = torch.softmax(scores, dim=-1)                     # each row sums to 1
    return weights @ v, weights                                 # weighted average of V
```

**为什么要除以 √d？** 原论文给出了这样的解释。假设 q、k 的每个分量都是均值 0、方差 1 的独立随机数。那么 `q·k = Σ q_i k_i` 是 d 个方差为 1 的项相加，方差就是 d。d 越大，分数之间差得越开，softmax 就越接近 one-hot（几乎只看一个位置），梯度也越小。除以 √d 之后，方差回到 1。

[`code/03_why_sqrt_d.py`](code/03_why_sqrt_d.py) 直接测量这一点：每个查询面对 16 个键，重复 2000 次。

```bash
uv run python chapters/08-attention/code/03_why_sqrt_d.py
```

| d | 不缩放：分数方差 | 最大权重 | 有效个数 | 梯度大小 | 缩放后：分数方差 | 最大权重 | 有效个数 | 梯度大小 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 15.9 | 0.672 | 2.91 | 0.285 | 1.0 | 0.241 | 10.93 | 0.303 |
| 64 | 64.7 | 0.842 | 1.66 | 0.176 | 1.0 | 0.243 | 10.81 | 0.304 |
| 256 | 254.4 | 0.921 | 1.29 | 0.102 | 1.0 | 0.248 | 10.73 | 0.305 |
| 1024 | 1026.8 | 0.957 | 1.14 | 0.058 | 1.0 | 0.245 | 10.75 | 0.305 |

- "有效个数"是 `e^熵`。权重均匀分给 16 个位置时是 16，全部集中在一个位置上时是 1。
- "梯度大小"是 softmax 雅可比矩阵 `diag(p) − ppᵀ` 的 Frobenius 范数。它衡量分数上的梯度有多少能传过 softmax。

不缩放时，d = 1024 的分数方差约为 1027，和理论值 d 吻合。最大权重是 0.957，有效个数是 1.14：注意力几乎只看一个位置。梯度大小只有缩放后的五分之一左右。缩放后，四种 d 的数字几乎相同。**除以 √d，让注意力的"软硬程度"和头的维度无关。** 真实模型的 head_dim 通常是 64 或 128（例如 gpt-oss 每个头 64 维，DeepSeek-V3 每个头 128 维）。所以这一步是必需的。

## 6. 因果 mask：不许偷看答案

语言模型的任务是用前文预测下一个 token。训练时，我们把整段序列一次输入模型，同时让每个位置预测它的下一个 token（第 7 章）。如果位置 t 能看到位置 t+1，它就能直接"抄到答案"。所以每个位置只能看自己和之前的位置。

我们把分数矩阵的上三角填成 `−∞`，这叫**因果 mask**（causal mask）。原论文的说法是："masking out (setting to −∞) all values in the input of the softmax which correspond to illegal connections"。

`02_attention_from_scratch.py` 做了一个因果性检验：把一段序列最后 3 个位置的输入换成别的随机数，前 5 个位置的输出必须完全不变。

```
New inputs at positions 5–7: max output change at positions 0–4: 0.0e+00; at positions 5–7: 0.35
```

## 7. 多头注意力：同时有几种看法

一组注意力权重只能表达一种"看法"：每个位置按一种规则分配注意力。但一个 token 可能同时需要两种信息，例如"前一个字是什么"和"这个词从哪里开始"。**多头注意力**（multi-head attention）把 C 维切成 H 份，每份 `d = C / H` 维。每份独立做一遍注意力。最后把各份拼接起来，再乘一个输出矩阵 `W_o`：

```python
q = q.view(B, T, self.H, self.d).transpose(1, 2)    # (B, T, C) → (B, H, T, d)
out, w = attention(q, k, v, causal=True)            # each head calculates on its own
out = out.transpose(1, 2).reshape(B, T, C)          # concatenate the heads again
out = self.wo(out)                                  # output projection
```

把"头"这个维度移到前面，当作批的一部分。这样所有头就能在一次矩阵乘法里并行算完。下表是每一步的形状（B = 2，T = 8，C = 32，H = 4，是脚本实际打印的值）：

| 步骤 | 形状 | 含义 |
|---|---|---|
| 输入 `x` | (2, 8, 32) | (B, T, C) |
| `q = x @ Wq` | (2, 8, 32) | (B, T, C) |
| 切头 | (2, 4, 8, 8) | (B, H, T, d) |
| 权重 `softmax(QKᵀ/√d)` | (2, 4, 8, 8) | (B, H, T, T) |
| 每个头的输出 `w @ v` | (2, 4, 8, 8) | (B, H, T, d) |
| 拼接各头 | (2, 8, 32) | (B, T, C) |
| 输出 `@ Wo` | (2, 8, 32) | (B, T, C) |

（这里 T 和 d 恰好都是 8，不要混淆：权重是 T × T，每个头的输出是 T × d。）

参数量是 `4 × C² = 4096`（Wq、Wk、Wv、Wo 各是 C × C）。**参数量和头数无关**：多头不增加参数，只是换了一种组织方式。原论文用 8 个头，每个头 64 维。

## 8. 对拍：和 PyTorch 官方实现结果一致

```bash
uv run python chapters/08-attention/code/02_attention_from_scratch.py
```

脚本把同一组权重分别交给我们手写的 `attention` 和 PyTorch 的 `F.scaled_dot_product_attention(q, k, v, is_causal=True)`：

```
Max difference from F.scaled_dot_product_attention(is_causal=True): 9.7e-08
Parameters: 4096 = 4 × C² = 4 × 32² (Wq, Wk, Wv, Wo; does not depend on the number of heads)
```

差在 10⁻⁷ 量级，就是 float32 的舍入误差。确认两者一致之后，后面的代码就可以放心地用官方函数。官方函数在 GPU 上还有快得多的实现（见"从极简代码到生产级代码"）。

## 9. 训练一个单层注意力模型，看它到底在看哪里

公式推完了，下面真正训练一个模型。[`code/04_train_attention.py`](code/04_train_attention.py) 在 Tiny Shakespeare（`assets/tiny_corpus/shakespeare.txt`，约 1.1 MB，65 种字符，全是 ASCII，所以 1 字符 = 1 字节）上训练三个字符级模型。

三个模型只在"怎样混合前文"这一步不同，其余部分完全相同。它们用同样的 embedding（加一个可学习的位置向量，第 9 章换成 RoPE）和同样的输出层。它们还用同样的 AdamW、同样的 2000 步、同样的数据顺序。

```python
h = self.tok(idx) + self.pos(torch.arange(T))
if self.mode == "attention":
    out, w = self.mix(h, return_weights=True)      # MultiHeadAttention from this chapter
    h = h + out                                    # residual connection (Chapter 6)
elif self.mode == "average":
    W = torch.tril(torch.ones(T, T))
    W = W / W.sum(1, keepdim=True)                 # fixed uniform weights
    h = h + self.wo(W @ self.wv(h))
logits = self.head(h)
```

```bash
uv run python chapters/08-attention/code/04_train_attention.py     # about 1 minute on one CPU thread
```

（脚本里有一行 `torch.set_num_threads(1)`。模型这么小，多线程的调度开销比计算本身还大。在我们的构建机上，机器繁忙时，4 线程每步比单线程慢了约 100 倍。）

| 模型 | 参数量 | 验证损失（nats/字符） | bits-per-byte |
|---|---:|---:|---:|
| bigram（只看当前字符 + 位置） | 12,481 | 2.487 | 3.588 |
| 均匀平均（前缀平均） | 20,673 | 2.463 | 3.553 |
| 注意力（4 头，每头 16 维） | 28,865 | 2.083 | 3.005 |

bits-per-byte 就是第 7 章的指标：损失除以 ln 2（这里 1 字符 = 1 字节）。请看这几个结果：

- **均匀平均几乎没有用。** 它多了 8192 个参数，损失只从 2.487 降到 2.463。模型看得见前文，但每个字的权重相同，模型没法从均匀混合的结果里挑出有用的信息。
- **注意力把损失降到 2.083。** 参数只再多 8192 个，每个字节少用 0.58 个比特。看得见前文还不够，**关键是会挑**。
- 注意力模型在 2000 步时损失还在下降（1500 步 2.111 → 2000 步 2.083），多训练一会儿还会更好。这里停下，只是为了让脚本一分钟跑完。

**它到底在看哪里？** 脚本在验证集上统计了每个头平均把多少权重放在"往前数 k 个"的位置上：

| 头 | 自己 (k=0) | 前 1 个 | 前 2 个 | 更早 (k≥3) |
|---|---:|---:|---:|---:|
| 0 | 0.10 | 0.17 | 0.50 | 0.23 |
| 1 | 0.21 | 0.62 | 0.09 | 0.08 |
| 2 | 0.14 | 0.32 | 0.26 | 0.28 |
| 3 | 0.23 | 0.57 | 0.10 | 0.10 |

四个头自己分了工：头 1 和头 3 主要看前 1 个字符，头 0 主要看前 2 个字符，头 2 分散得最开。

没有人告诉模型要这样做。多个头合起来，相当于模型自己拼出了一个"看前两三个字符"的模型（有点像 trigram），而且权重还会随内容变化。

脚本还把每个注意力矩阵画成了字符热力图（heat map）。下面是头 1 在 `First Citizen:\nBefore we proceed any further, hear me speak.` 最后一段上的节选。行是当前字符，列是被看的字符。符号越密，权重越大：` .:-=+*#%@` 依次对应 0–0.1、0.1–0.2……0.9–1.0。`␣` 是空格。

```
    Before␣we␣proceed␣any␣further,␣hear␣me␣speak.
  ,                             @
  ␣                               @
  h                               @
  e                               :+.
  a                                 @
  r                                  @
  ␣                                    %
  m                                    @
  e                                     @
  ␣                                       @
  s                                       @
  p                                        #.
  e                                       . #
  a                                        -.-.
  k                                          =+
  .                                           .#
```

大部分行的 `@` 都落在对角线左边一格：头 1 是一个"看前一个字符"的头。但它的偏移不是固定不变的。脚本逐个位置打印了权重最大的字符（`→k back` 表示往前数 k 个）：

```
Head 1 on ', hear me speak.': the character with the largest weight at each position:
  ,→1 back 'r' 1.00   ␣→0 back '␣' 0.94   h→1 back '␣' 0.96   e→1 back 'h' 0.60
  a→1 back 'e' 0.90   r→1 back 'a' 0.99   ␣→0 back '␣' 0.89   m→1 back '␣' 0.99
  e→1 back 'm' 0.99   ␣→0 back '␣' 0.93   s→1 back '␣' 1.00   p→1 back 's' 0.79
  e→1 back 'p' 0.70   a→1 back 'e' 0.37   k→1 back 'a' 0.50   .→1 back 'k' 0.76

Head 0 on ', hear me speak.': the character with the largest weight at each position:
  ,→43 back 'i' 0.89   ␣→7 back 'u' 0.28   h→2 back ',' 0.66   e→2 back '␣' 0.50
  a→2 back 'h' 0.38   r→0 back 'r' 0.36   ␣→2 back 'a' 0.27   m→3 back 'a' 0.61
  e→1 back 'm' 0.83   ␣→6 back 'e' 0.27   s→2 back 'e' 0.54   p→0 back 'p' 0.77
  e→2 back 's' 0.87   a→2 back 'p' 0.99   k→1 back 'a' 0.61   .→56 back 's' 0.32
```

- **头 1 的规则随内容变化。** 在字母上，它几乎只看前一个字符（`r→a` 0.99、`m→␣` 0.99）。但一到**空格**，它就改看空格自己（0.94、0.89、0.93）。同一个头、同样的相对位置，权重却不同。这正是"由数据决定权重"的含义。
- **头 0 大多看前 2 个字符**（`e→s` 0.87、`a→p` 0.99）。但在逗号和句号上，它把大部分权重放在序列开头附近。逗号的最大权重（0.89）落在 43 个字符之前、`First` 里的 `i` 上。在标点处，这个头似乎没有什么可取的，就把权重放在开头。这和"前沿观察"里的 attention sink 现象看起来很像，但我们没有在这个小模型上进一步验证。

视频里用紫色热力图展示了头 1 和头 0 在 `further, hear me speak.` 上的完整矩阵。

也要说清楚这个模型的局限。它只有一层，没有前馈网络（feed-forward network），只训练了一分钟。它学不到语法，也不理解文本。它只发现了一件事：离得近的字符最有用。第 9 章把注意力放进完整的 Transformer 块、叠很多层之后，才会出现更复杂的模式。

## 10. 小结

- **问题**：bigram 只看前一个 token。RNN 把前文压进固定大小的状态，而且只能顺序计算。
- **骨架**：输出是前文向量的加权平均。一次下三角矩阵乘法就能算完所有位置。步骤是：分数 → 因果 mask（上三角填 −∞）→ softmax → 加权平均。
- **Q、K、V**：三个线性投影。分数是查询和键的点积，平均的对象是值。
- **缩放**：`q·k` 的方差约为 d。除以 √d 后约为 1，softmax 不再退化成 one-hot。
- **多头**：把 C 维切成 H 份，各自算注意力，拼接后乘 `W_o`。形状是 `(B,T,C) → (B,H,T,d) → (B,H,T,T) → (B,T,C)`。参数量 4C² 与头数无关。
- **实测**：从零实现与官方 SDPA 的最大差是 9.7e-08。单层注意力把验证损失从 2.487（bigram）降到 2.083，均匀平均只降到 2.463。

---

## GPU 实测（单张 RTX 3090）

> 上面正文里的数字都来自 CPU 运行。本节在一张 NVIDIA GeForce RTX 3090（24 GB 显存，Ampere 架构）上实测。规格表给出的值：BF16 张量核稠密峰值约 71 TFLOPS，FP32 约 35.6 TFLOPS，显存带宽约 936 GB/s。
>
> 环境：PyTorch 2.11.0+cu128、CUDA 12.8，2026 年 10 月。服务器把这张卡的功耗上限设成了 240 W（出厂默认 350 W）。持续满载时，这张卡会降频。所以算力、带宽的绝对值比满功耗的 3090 低，相对关系更可靠。没有 GPU 可以跳过本节。

运行：

```bash
uv run python chapters/08-attention/code/06_gpu_sdpa_backends.py
```

形状取主线模型的注意力：batch 1、16 个查询头、head_dim 128、BF16、因果 mask。"手写"就是第 5 节的 `attention` 函数，原样搬到 GPU 上。另外三列是 `F.scaled_dot_product_attention`，用 `sdpa_kernel` 分别锁定 math、efficient、flash 三个后端。耗时是 CUDA event 计时的中位数（毫秒）：

| T | 手写 `attention` | SDPA math | SDPA efficient | SDPA flash |
|---:|---:|---:|---:|---:|
| 512 | 0.22 | 0.50 | 0.09 | 0.08 |
| 1K | 0.59 | 1.59 | 0.23 | 0.18 |
| 2K | 2.56 | 6.74 | 0.70 | 0.42 |
| 4K | 11.06 | 28.19 | 1.73 | 1.36 |
| 8K | 44.36 | 110.76 | 6.60 | 4.96 |
| 16K | 170.74 | 显存不够 | 29.21 | 23.50 |
| 32K | 显存不够 | 显存不够 | 119.55 | 102.95 |

下表是同一次运行的峰值额外显存（MiB）。它是调用过程中比调用前多占的最高值，不含 q、k、v 本身。第二列作为参照：16 个头的 T × T 分数矩阵按 BF16 存储有多大。

| T | T × T 分数矩阵 | 手写 `attention` | SDPA math | SDPA efficient | SDPA flash |
|---:|---:|---:|---:|---:|---:|
| 512 | 8 | 18 | 53 | 2 | 2 |
| 1K | 32 | 69 | 180 | 4 | 4 |
| 2K | 128 | 268 | 656 | 8 | 8 |
| 4K | 512 | 1,056 | 2,496 | 16 | 16 |
| 8K | 2,048 | 4,192 | 9,728 | 32 | 33 |
| 16K | 8,192 | 16,704 | 显存不够 | 64 | 65 |
| 32K | 32,768 | 显存不够 | 显存不够 | 128 | 130 |

脚本最后换成 GQA 的形状（16 个查询头共享 8 组 K/V，T = 4096），并打开 `enable_gqa=True`。flash 能跑：它和"先把 K/V 复制成 16 份再算"的结果最大差是 7.8e-03（BF16 舍入量级）。它的额外显存是 16 MiB，而先复制 K/V 要 48 MiB。

efficient 不能跑（`No available kernel`；PyTorch 给的原因是它要求 q、k、v 的头数相同）。math 能跑。不指定后端时，MHA 和 GQA 默认选的都是 flash。

第二张表就是引导问题 2 里那个 `T × T` 的实物。手写版多占的显存几乎正好是分数矩阵的两倍：`QKᵀ/√d` 和 softmax 的结果各存一份。T 翻一倍，这部分显存涨 4 倍。16K 时是 16.3 GiB；照这个比例，32K 要 64 GiB，24 GB 的卡放不下。

efficient 和 flash 只多占"输出"那么一点（16K 时 64 MiB），随 T 线性增长。原因是它们分块计算，从不把 T × T 的矩阵写进显存。

但 FlashAttention 并没有少算。T 翻一倍，flash 的耗时照样涨约 4 倍，因为计算量还是 T²。它在 16K 时比手写版快 7 倍多（23.5 ms 对 170.7 ms）。它省下的是把 T × T 矩阵写进显存再读出来的来回（原理在第 14 章）。

一个意外的发现：SDPA 的 math 后端比我们手写的还慢，也更费显存。4K 时，math 要 28 ms、2.4 GiB，手写版要 11 ms、1.0 GiB。math 先把 BF16 输入转成 float32 再算，换来的是精度。在脚本开头的对拍里，math 和 flash 与 float32 参考的最大差都是 7.3e-03。手写的 BF16 版是 1.6e-02。

下面表格里"SDPA 在 CUDA 上会自动选用 FlashAttention-2"这一条，在这张卡、BF16 下成立。GQA 也一样走 flash，不用真的复制 K/V。

## 从极简代码到生产级代码

主线模型的注意力在 [`zero/model.py`](../../zero/model.py) 的 `Attention` 类里。它的骨架和本章的 `MultiHeadAttention` 相同：三个投影 `wq`、`wk`、`wv`，缩放点积，因果 mask，多头，输出投影 `wo`。参数名也相同。下面是 `Attention.forward` 的核心（省略了 transpose，有几行做了合并）：

```python
q = self.q_norm(self.wq(x).view(bsz, seqlen, self.n_heads, self.head_dim))      # QK-Norm
k = self.k_norm(self.wk(x).view(bsz, seqlen, self.n_kv_heads, self.head_dim))   # GQA: fewer K/V heads
v = self.wv(x).view(bsz, seqlen, self.n_kv_heads, self.head_dim)
q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)                          # RoPE
if kv_cache is not None:
    k, v = kv_cache.update(layer_idx, start_pos, k, v)                           # KV cache
out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=is_causal,
                                     enable_gqa=self.n_kv_heads != self.n_heads)
return self.wo(out.reshape(bsz, seqlen, self.n_heads * self.head_dim))
```

| 极简代码（本章） | 生产级代码（`zero/model.py` 的 `Attention`） | 为什么 |
|---|---|---|
| `wq`、`wk`、`wv` 三个 `nn.Linear(C, C)` | 同样是三个独立的投影（没有融合成一个大矩阵），但 `wk`、`wv` 的输出是 `kv_dim = n_kv_heads × head_dim`；全部没有 bias | 参数名和形状与 Hugging Face 的 Qwen3 一一对应，才能直接加载官方权重做对拍。不用 bias：Qwen3 去掉了 Qwen2 的 QKV bias，OLMo 2 也不用任何 bias |
| 头数 H，每个头 `d = C/H` | `n_heads` 个查询头，`n_kv_heads` 个 K/V 头；`head_dim` 可以不等于 `dim / n_heads` | **GQA**（分组查询注意力，grouped-query attention）：几个查询头共用一组 K/V，KV cache 按比例变小。第 10 章讲 |
| 直接用 q、k | `q_norm`、`k_norm`：对每个头的 q、k 在 head_dim 上做 RMSNorm | **QK-Norm**：防止注意力分数过大导致训练发散（Qwen3、OLMo 2、Gemma 3 都用）。第 9 章讲 |
| 可学习的位置向量，加在输入上（在 `04` 里） | `apply_rope`：对 q、k 做旋转位置编码 | **RoPE**：位置信息直接进入 q·k，只和相对距离有关。第 9 章讲 |
| 手写 `softmax(QKᵀ/√d)` + `masked_fill` | `F.scaled_dot_product_attention(..., is_causal=..., enable_gqa=...)` | 在 CUDA 上用 BF16/FP16 时，SDPA 自动选用 FlashAttention-2 内核，不把 T × T 的权重矩阵写进显存。`enable_gqa` 让 SDPA 自己广播 K/V 头，不用复制。（memory-efficient 内核不支持 GQA，所以 GQA 只能走 Flash；FP32 时退回 math 后端。）已在 RTX 3090 上验证：BF16 + `enable_gqa=True` 默认就走 Flash（见本章"GPU 实测"与 [runs/2026-10-01-gpu0-check](../../runs/2026-10-01-gpu0-check/README.zh.md) 第 1 节）。CPU 上用的是 PyTorch 的 C++ 参考实现 |
| 一次处理整段序列 | 可选的 `kv_cache`：推理时把算过的 K/V 存起来。有历史时，按 `start_pos` 构造 mask（单个新 token 不需要 mask，分块 prefill 用显式的布尔 mask） | 生成时每步只算新 token 的 q/k/v。第 10 章讲 |

**对拍**：[`code/05_zero_parity.py`](code/05_zero_parity.py) 把本章 `MultiHeadAttention` 的权重原样加载进 `zero.model.Attention`。它关掉 QK-Norm，并传入 `cos = 1、sin = 0`，让 RoPE 变成恒等变换。然后它打开 GQA，和"手工复制 K/V"比较：

```bash
uv run python chapters/08-attention/code/05_zero_parity.py
```

```
① Minimal MultiHeadAttention vs zero.model.Attention (QK-Norm and RoPE off), max difference: 8.9e-08
② GQA Wk shape (16, 32) (MHA: (32, 32)): the K/V projections and the KV cache are half the size
   zero GQA vs manual "copy K/V, then multi-head attention", max difference: 8.9e-08
   Attention parameters: MHA 4096, GQA 3072
```

在整个模型层面，[`tests/test_model_hf_parity.py`](../../tests/test_model_hf_parity.py) 随机初始化一个小的 Hugging Face `Qwen3ForCausalLM`。它用 GQA（4 个查询头 / 2 个 K/V 头）+ QK-Norm + RoPE；测试也覆盖了无 GQA、YaRN、`head_dim ≠ dim / n_heads` 的情形。测试把权重搬进 `zero`，要求 logits 在 1e-5 以内一致（`uv run pytest tests/test_model_hf_parity.py`）。所以"本章的极简版 ↔ zero 的 `Attention` ↔ 官方 Qwen3 实现"是一条每一环都对得上的链。

---

## 前沿观察

**注意力 sink / softmax 分母里的可学习偏置**：标准 softmax 强制每行权重加起来等于 1，即使当前位置"什么都不需要看"。gpt-oss 给每个头的 softmax 分母加了一个可学习的偏置，让注意力可以"不看任何 token"（模型卡 §2.2，引用了 "Attention is off by one" 和 attention sinks 的工作）。本章核实过的其他家族（Qwen3、Llama 3、OLMo 2、Gemma 3、DeepSeek-V3）的报告里没有这一项。所以这里只提一句，正文仍讲标准 softmax。

---

## 采用方与来源

多头缩放点积因果注意力是所有 decoder-only 大模型的基本部件。各家的差别只在变体（GQA、MLA、局部/全局交替等，在第 10、21–23 章讲）。以下内容都是技术报告或模型卡里的明确表述。

| 技术 | 采用方 | 来源 |
|---|---|---|
| 缩放点积注意力 `softmax(QKᵀ/√d_k)V`、多头 + 输出投影、解码器里用 −∞ 屏蔽未来位置 | 提出者：Transformer（h = 8，d_k = 64） | [Vaswani et al. 2017 §3.2](https://arxiv.org/abs/1706.03762) |
| 多头因果自注意力（GQA 形式） | Qwen3（"Grouped Query Attention"，如 Qwen3-0.6B 为 16 个查询头 / 8 个 KV 头）；Llama 3（"standard, dense Transformer"，GQA 8 个 KV 头，8B 为 32 个注意力头）；Gemma 3（"decoder-only transformer … Grouped-Query Attention"）；gpt-oss（每层 64 个 64 维查询头，GQA 8 个 KV 头） | [Qwen3 §2 表 1](https://arxiv.org/abs/2505.09388)、[Llama 3 §3.2 表 3](https://arxiv.org/abs/2407.21783)、[Gemma 3 §2](https://arxiv.org/abs/2503.19786)、[gpt-oss 模型卡 §2.2](https://arxiv.org/abs/2508.10925) |
| 多头因果自注意力（标准 MHA 形式） | OLMo 2 7B / 13B（32/32、40/40 "MHA"；32B 换成 GQA 40/8） | [OLMo 2 §2.1 表 3](https://arxiv.org/abs/2501.00656) |
| 多头因果自注意力（MLA 形式） | DeepSeek-V3：128 个头，每头 128 维，输出仍是 `Σ_j softmax_j(q·k/√(d_h + d_h^R)) v`，再乘 `W_O` | [DeepSeek-V3 §2.1.1 式 (10)(11)、§4.2](https://arxiv.org/abs/2412.19437) |
| QK-Norm（本章只预告） | Qwen3（"introduce QK-Norm … to ensure stable training"）；OLMo 2；Gemma 3（"replace the soft-capping of Gemma 2 with QK-norm"） | 同上 |
| FlashAttention（通过 SDPA 使用，行业标准） | gpt-oss（"leverage the Flash Attention algorithms"）；PyTorch SDPA 在 CUDA 上自动选用 FlashAttention-2 | [gpt-oss 模型卡 §2.4](https://arxiv.org/abs/2508.10925)、[Dao et al. 2022](https://arxiv.org/abs/2205.14135)、PyTorch `scaled_dot_product_attention` 文档 |

待核实：无。

---

## 引导问题

带着这些问题去问 Claude Code，直到你能用自己的话讲清楚：

1. 如果去掉因果 mask 直接训练语言模型，训练损失会怎样？验证时（真的逐个 token 生成时）又会怎样？为什么？可以让 Claude Code 帮你在 `04_train_attention.py` 里加一个开关试试。
2. 注意力的计算量里有一项是 `T × T`。上下文从 4K 扩到 128K，这一项涨多少倍？这和第 14 章的 FlashAttention、第 22–23 章的滑动窗口和线性注意力有什么关系？
3. 注意力本身对输入顺序"无感"：把前文打乱，同一个查询得到的加权平均一样吗？那模型怎样知道顺序？（`04` 里的 `self.pos` 是做什么的？第 9 章的 RoPE 为什么更好？）
4. 为什么多头注意力的参数量和头数无关？如果头数多到每个头只有 1 维，会发生什么？（提示：原论文表 3 的 (A) 行做过这个实验。）
5. `05_zero_parity.py` 里 GQA 把 K/V 头从 4 个减到 2 个。省下来的是哪些参数？推理时省下来的又是什么？（第 10 章会算这笔账。）

## 动手任务

每个任务都要真的运行代码，看到结果。

**任务 1（基础）**：在 `01_average_to_attention.py` 里改一下点积分数：先让 `x` 分别乘两个不同的随机矩阵 `Wq`、`Wk`，再做点积（`(x @ Wq) @ (x @ Wk).T`）。再数一次"有几个位置最关注自己"。结果变了吗？这说明 Q、K 分开有什么好处？

**任务 2（核心）**：在 `02_attention_from_scratch.py` 的 `attention` 函数里去掉 `/ math.sqrt(d)`。把 `C` 改成 256，`H` 改成 1（单头 256 维），打印第一个头的权重矩阵。每行最大的权重大约是多少？和第 5 节的表格对得上吗？

**任务 3（挑战）**：在 `04_train_attention.py` 里加第四种模式 `"no_mask"`：注意力不加因果 mask。训练后比较它的训练损失和验证损失（验证时也不加 mask）。然后写一个"真正逐字生成"的函数，比较 `attention` 和 `no_mask` 两个模型生成的文本。损失低的那个生成得更好吗？为什么？

---

## 想深入：CS336

本章对应斯坦福 [CS336: Language Modeling from Scratch](https://cs336.stanford.edu/)（Spring 2026）：

- **第 3 讲：架构与超参（Architectures and hyperparameters）**：从原始 Transformer 讲到现代大模型的各种架构选择。这些选择包括注意力的变体（MHA、GQA 等）、归一化和位置编码。本章是其中"注意力"这一块的入门，其余部分在第 9 章。
- **作业 1（Basics）**：要求从零实现这些部分：`scaled_dot_product_attention`、因果多头自注意力（带 RoPE 和不带 RoPE 两个版本）、Transformer 块和完整的语言模型。代码要通过单元测试。接口见作业仓库 [stanford-cs336/assignment1-basics](https://github.com/stanford-cs336/assignment1-basics) 的 `tests/adapters.py` 里的 `run_scaled_dot_product_attention`、`run_multihead_self_attention` 等。具体要求以当期作业说明为准。本章的 `02_attention_from_scratch.py` 可以直接当作这一部分的热身。

课程页有每一讲的讲义和 YouTube 录像。

---

## 本章参考文献

- Vaswani et al. *Attention Is All You Need*，NeurIPS 2017（缩放点积注意力、√d_k 的方差解释、多头、因果 mask）：<https://arxiv.org/abs/1706.03762>
- Bahdanau, Cho, Bengio. *Neural Machine Translation by Jointly Learning to Align and Translate*（注意力的起源；固定长度向量是瓶颈），2014：<https://arxiv.org/abs/1409.0473>
- Ainslie et al. *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints*，2023：<https://arxiv.org/abs/2305.13245>
- Dao et al. *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*，2022：<https://arxiv.org/abs/2205.14135>
- Qwen Team. *Qwen3 Technical Report*，2025：<https://arxiv.org/abs/2505.09388>
- Llama Team. *The Llama 3 Herd of Models*，2024：<https://arxiv.org/abs/2407.21783>
- OLMo Team. *2 OLMo 2 Furious*，2024：<https://arxiv.org/abs/2501.00656>
- DeepSeek-AI. *DeepSeek-V3 Technical Report*（MLA），2024：<https://arxiv.org/abs/2412.19437>
- Gemma Team. *Gemma 3 Technical Report*，2025：<https://arxiv.org/abs/2503.19786>
- OpenAI. *gpt-oss-120b & gpt-oss-20b Model Card*，2025：<https://arxiv.org/abs/2508.10925>
- Karpathy. nanoGPT（`CausalSelfAttention` 的写法；用下三角矩阵求前缀平均的技巧来自配套视频 *Let's build GPT: from scratch, in code, spelled out*）：<https://github.com/karpathy/nanoGPT>、<https://github.com/karpathy/ng-video-lecture>
- *Understanding Transformers and Attention Mechanisms: An Introduction for Applied Mathematicians*（`references.md` 已收录，适合想看更数学化推导的读者）：<https://arxiv.org/pdf/2604.00965>
- CS336 作业 1 仓库：<https://github.com/stanford-cs336/assignment1-basics>
- CS336（Spring 2026）：<https://cs336.stanford.edu/>

**下一章**：注意力会搬运和混合信息，但它几乎不做"计算"：每个位置拿到的只是别人值向量的加权平均。而且我们的单层模型只看得出"前一两个字符"这种浅层模式。第 9 章，我们给注意力配上前馈网络（SwiGLU）、RMSNorm、残差连接和 RoPE。我们把很多层叠起来，搭出一个完整的现代 Transformer，并训练它生成像样的文字。
