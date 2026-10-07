# 第 24 章：混合专家（MoE）—— 参数多出几十倍，每个 token 的算力不变

[English](README.md) · **中文**

> **目标**：读完这一章，你能按一个模型的 `config.json` 算出它的总参数和激活参数。你能写出 MoE 层的路由（打分 → top-k → 归一化 → 加权求和），并和逐 token 的朴素实现对拍。你能讲清楚路由为什么会坍缩，以及辅助损失和 DeepSeek-V3 的"无辅助损失偏置"各自怎样把负载拉平。你还能用显存和算力两笔账解释两件事。几百亿参数以上的大模型几乎都是 MoE。1B 以内的小模型几乎都是稠密的。

📺 **本章视频**：还没有发布。要在本机渲染，运行 `bash chapters/24-mixture-of-experts/video/build.sh`。
🧪 **本章自检**：学完后，在 Claude Code 里输入 `/ch24-moe`。

---

前三章一直在给**注意力**（attention）省钱。GQA 和 MLA 让每个位置少存一点。滑动窗口让一些层只看附近的位置。线性注意力把全部历史压进一个固定大小的状态。这一章解决另一半问题：**模型想要更多参数（更多知识），但每个 token 的算力不能跟着涨。怎么办？**

答案在 Transformer 的另一个子层：前馈网络（feed-forward network，FFN）。把一个大 FFN 拆成许多个小 FFN，叫作**专家**（expert）。再让一个小小的**路由器**（router）给每个 token 只挑其中几个。这就是**混合专家**（mixture of experts，MoE）。DeepSeek-V3 有 6710 亿参数，每个 token 只用 370 亿。Kimi K2 有一万亿参数，每个 token 只用 326 亿。

MoE 的想法听起来不难，难点在"挑"上。路由器和模型一起训练，它会偏向某些专家。所以**负载均衡**（load balancing）是这一章的主角。我们从零写一个 MoE 层，在 CPU 上训练几个小语言模型，看路由坍缩是什么样子。然后看两种均衡办法怎样把它拉回来。

本章代码（全部在 CPU 上运行）：

```bash
uv run python chapters/24-mixture-of-experts/code/01_param_ledger.py    # parameter ledger: the FFN share; total/active parameters of 9 MoE models (seconds)
uv run python chapters/24-mixture-of-experts/code/02_moe_layer.py       # MoE layer from zero: parity check, auxiliary loss by hand, bias balancing (seconds)
uv run python chapters/24-mixture-of-experts/code/03_train_compare.py   # dense vs MoE, small language models with three balancing setups (slow the first time, then reads the cache)
uv run python chapters/24-mixture-of-experts/code/04_small_vs_large.py  # why small models seldom use MoE: lineups, same-data comparison, memory and reads (seconds)
```

## 1. 直觉：参数都在 FFN 里

第 9 章讲过，每个 Transformer Block 有两个子层。注意力在位置之间搬运信息。FFN（SwiGLU）在每个位置内部加工信息。先数一数两个子层各占多少参数（[`code/01_param_ledger.py`](code/01_param_ledger.py) 第 1 部分）：

| 模型 | 每层注意力 | 每层 FFN | FFN 占比 |
|---|---:|---:|---:|
| Qwen3-8B（d = 4096，FFN 宽 12288） | 41.9M | 151.0M | **78.3%** |
| 主线模型（d = 1280，FFN 宽 3584） | 7.9M | 13.8M | 63.6% |

前向传播时，每个参数大约贡献 2 次浮点运算（一次乘、一次加）。所以这张表也是"每个 token 的矩阵乘算力花在哪"的账：**大头在 FFN**。

这就引出一个矛盾。Scaling law（第 12 章）告诉我们，参数越多，模型越强。但在稠密模型里，参数多一倍，每个 token 的算力也多一倍。训练贵一倍，推理也慢一倍。能不能**让参数和算力脱钩**？

MoE 的回答：把一个宽 FFN 换成 N 个窄一些的 FFN（专家）。每个 token 只经过其中 K 个。

- **总参数**（total parameters）：N 个专家全部算上。它决定模型能"装"多少知识，也决定**显存要装多少权重**。
- **激活参数**（active / activated parameters）：每个 token 实际用到的部分（K 个专家 + 注意力 + embedding 等）。它决定**每个 token 的算力**。

模型的名字里就写着这两个数。Qwen3-235B-A22B 表示"总参数 235B，激活（Activated）22B"。Qwen3.5-35B-A3B 表示"总参数 35B，激活 3B"。

## 2. 真实模型的账本

拿来几个 MoE 模型的 `config.json`，按"embedding + 每层注意力 + 每层 MoE（路由器 + 全部专家 + 共享专家）+ lm_head"数一遍（[`code/01_param_ledger.py`](code/01_param_ledger.py) 第 2 部分）：

| 模型 | 专家数 | 每 token 选 | 共享专家 | 专家宽度 | 总参数（算 / 官方） | 激活参数（算 / 官方） | 激活比 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Mixtral-8x7B（2023.12） | 8 | 2 | 0 | 14336 | 46.7B / 47B | 12.9B / 13B | 27.6% |
| Llama-4-Scout | 16 | 1 | 1 | 8192 | 107.8B / 109B | 17.2B / 17B | 15.9% |
| Qwen3-30B-A3B | 128 | 8 | 0 | 768 | 30.5B / 30.5B | 3.35B / 3.3B | 11.0% |
| Qwen3-235B-A22B | 128 | 8 | 0 | 1536 | 235.1B / 235B | 22.2B / 22B | 9.4% |
| GLM-4.5 | 160 | 8 | 1 | 1536 | 352.8B / 355B | 33.6B / 32B | 9.5% |
| DeepSeek-V3 | 256 | 8 | 1 | 2048 | 671.0B / 671B | 37.6B / 37B | 5.6% |
| gpt-oss-120b | 128 | 4 | 0 | 2880 | 116.8B / 116.83B | 5.71B / 5.13B | 4.9% |
| Kimi K2 | 384 | 8 | 1 | 2048 | 1026.4B / 1.04T | 32.9B / 32.6B | 3.2% |

我们数出来的数和官方公布的数几乎都对得上。差的那一点来自不同的统计口径（脚本最后逐条列出）。

GLM-4.5 的官方数不含 embedding 和输出层，但含 1 个 MTP 层。去掉 embedding 和输出层后，我们得到 351.2B / 32.1B。gpt-oss 的"激活"不含输入 embedding。去掉它后，正好是 5.13B / 3.61B。Llama 4 的 109B 含视觉编码器。Kimi K2 我们少算了约 14B（1.3%），原因**待核实**。

读这张表，能看到三个规律：

1. **路由专家占了总参数的 90%–99%。** 注意力、embedding 这些"每个 token 都要用"的部分只是零头。
2. **激活比越来越低。** Mixtral 从 8 个专家里选 2 个（28%），Kimi K2 已经是 384 选 8（3%）。Kimi K2 报告里的"稀疏度 scaling law"说的就是这件事。固定激活参数（也就是固定算力），专家总数越多，验证损失越低。他们选了稀疏度 48 = 384/8，作为性能和基础设施复杂度的折中。
3. **专家越来越"细"。** Mixtral 一个专家宽 14336，比稠密模型的 FFN 还宽。DeepSeek-V3 一个专家只有 2048，Qwen3-30B-A3B 只有 768。第 7 节讲原因。

## 3. 路由：打分、选 K 个、加权求和

### 3.1 公式

一个 MoE 层（对单个 token 的向量 x）：

```
s_i = sigmoid(x · e_i)    or    s = softmax(x · W_r)          # 1. score: one score for each expert
S   = TopK(s + b, K)                                          # 2. select K experts (b is a bias, see Section 5; for now, b = 0)
g_i = s_i / Σ_{j∈S} s_j        (i ∈ S)                        # 3. gate weights: scores of the selected experts, normalized to a sum of 1
y   = Σ_shared FFN_s(x)  +  Σ_{i∈S} g_i · FFN_i(x)             # 4. calculate only the selected experts, weighted sum
```

- **路由器**是一个 `d → N` 的线性层。DeepSeek 把它的每一行叫作专家的"质心向量" e_i。路由器的参数少到可以忽略（DeepSeek-V3 每层 7168 × 256 ≈ 1.8M）。
- **打分函数**有两类。softmax：Mixtral、Qwen3-MoE、gpt-oss。sigmoid：DeepSeek-V3、GLM-4.5、Kimi K2、MiniMax-M2、Nemotron 3、Llama 4（Llama 4 的 HF 实现对 top-1 的 logit 取 sigmoid）。softmax 让专家之间互相竞争（分数加起来等于 1）。sigmoid 让每个专家独立打分，再在选中的几个里归一化。
- **"先 softmax 再挑 K 个归一化"和"先挑 K 个 logit 再 softmax"结果相同。** Mixtral 论文写的是 `Softmax(TopK(x·W_g))`。Qwen3-MoE 的配置是 softmax 后 `norm_topk_prob: true`。两者算出的 g 完全相同，因为 e^{l_i}/Σ_{j∈S} e^{l_j} 在两种写法下相同。
- 有的模型还乘一个常数 `routed_scaling_factor`（DeepSeek-V3、GLM-4.5 是 2.5）。它只调整路由部分输出的尺度。

### 3.2 极简代码

[`code/02_moe_layer.py`](code/02_moe_layer.py) 的核心是公式的直译：

```python
logits = self.router(x)                                      # (T, E)
s = logits.sigmoid() if self.score == "sigmoid" else logits.softmax(-1)
idx = (s + self.bias).topk(self.K, dim=-1).indices           # (T, K) select experts: the bias only selects
g = s.gather(-1, idx)                                        # (T, K) gates: the original scores
g = g / g.sum(-1, keepdim=True)                              # normalize to a sum of 1

out = self.shared(x) if self.shared is not None else torch.zeros_like(x)
for e, expert in enumerate(self.experts):
    tok, slot = (idx == e).nonzero(as_tuple=True)            # the tokens that selected expert e
    if tok.numel():
        out.index_add_(0, tok, g[tok, slot, None] * expert(x[tok]))   # weighted combination
```

注意，循环是**按专家**，不是按 token。每个专家把选了它的 token 收集起来，用一次矩阵乘算完。然后 `index_add_` 按权重把结果加回各自的位置。这就是 MoE 实现里的两步：**分发**（dispatch）和**合并**（combine）。脚本把它和逐 token 的朴素写法对拍：

```
2) Grouped by expert vs naive token by token
  softmax max difference 2.2e-16
  sigmoid max difference 1.4e-16
```

参数账（d = 128；这些正是第 6 节小实验用的配置）：

| FFN | 总参数 | 激活参数 |
|---|---:|---:|
| 稠密 SwiGLU，宽 384 | 147,456 | 147,456 |
| MoE：8 专家 × 宽 192，选 2 | 590,848 | 148,480 |
| 细粒度：16 专家 × 宽 96，选 3 + 1 个共享 | 628,736 | 149,504 |

激活参数几乎相同（多出的一千来个是路由器）。总参数是稠密 FFN 的 4 倍多。

## 4. 问题：路由坍缩

路由器和模型一起训练，这会形成一个正反馈循环。一开始，某个专家分到的 token 稍多一点。→ 它得到的梯度更多，学得更好。→ 路由器更倾向于选它。→ 它分到的 token 更多……最后，少数几个专家包揽大部分 token，其余专家几乎闲置。这叫**路由坍缩**（routing collapse）。Shazeer 等人 2017 年就指出了这个问题。

坍缩有两个代价：

- **浪费参数**：闲置专家的参数白占显存。模型实际上退化成了一个更小的模型。
- **拖慢训练和推理**：大模型的专家分布在不同的 GPU 上（**专家并行**，expert parallelism，第 9 节）。最忙的专家所在的卡决定整层的耗时。如果设了容量上限，还会丢 token（第 5.3 节）。

> **注意：**本章训练类实验的数字来自课程构建机上的一次 CPU 运行。不同机器、不同版本的底层数学库，浮点运算的顺序略有不同。训练几百步后，这些微小差异会被放大。你本机跑出的数字可能从小数点后第二三位开始就不一样。请以下文不依赖具体数值的结论为准。2026-10 在另一台服务器上的复跑对照见 [runs/2026-10-01-gpu0-check/chapters-24-26.md](../../runs/2026-10-01-gpu0-check/chapters-24-26.md)。

在本章的小实验里（第 6 节），不加任何均衡的 MoE，第 1 层的负载在训练中这样变化：

| step | 第 1 层（从 0 数）8 个专家的负载占比（均匀是 0.125） | 最大 / 平均 |
|---:|---|---:|
| 0 | 0.30 0.00 0.02 0.19 0.11 0.03 0.35 0.01 | 2.76 |
| 100 | 0.27 0.09 **0.42** 0.06 0.03 0.01 0.01 0.12 | 3.34 |
| 300 | 0.24 0.08 **0.45** 0.05 0.05 **0.00** 0.02 0.12 | 3.61 |
| 800 | 0.22 0.08 **0.45** 0.06 0.05 **0.00** 0.02 0.11 | 3.63 |

（8 选 2 时，"最大/平均"的上限是 8/2 = 4。）第 3 个专家从 2% 涨到 45%。第 6 个专家从 3% 跌到 0：训练结束时，它基本不被选中。

我们要如实说明两点。第一，**随机初始化的路由器一开始就不均衡**（step 0 已经是 2.76）。原因是同一层的 token 向量方向相近，某几个专家的分数系统性偏高。第二，在这个 800 步的小实验里，并不是每一层都越来越坏。4 层的"最大/平均"从 3.24、2.76、2.59、2.14 变成 2.25、3.63、2.46、2.79。但**没有任何一层自己回到均衡**。

## 5. 负载均衡

### 5.1 办法一：辅助损失（Switch / GShard）

最经典的做法是在语言模型的损失上再加一项**辅助损失**（auxiliary loss），惩罚不均衡（GShard 2020、Switch Transformer 2021）。本章用 DeepSeek-V3 报告式 17–20 的写法。Switch 原文的写法是它在 K = 1 时的特例。

```
f_i = N / (K·T) · (number of tokens in these T tokens that selected expert i)   # actual load (= 1 when balanced)
P_i = (1/T) · Σ_t s'_{i,t}                                                     # mean router probability of expert i (s' normalized per row)
L_aux = α · Σ_i f_i · P_i                                                       # = α with a perfect balance
```

f 是离散的计数，不可导，只当"权重"用。梯度只经过 P：哪个专家的 f 大（过载），梯度就更用力地压低它的路由概率。代码只有三行：

```python
p = s / s.sum(-1, keepdim=True)                          # routing probabilities, normalized per row
f = self.load * self.E / (self.K * x.shape[0])           # f_i (= 1 when balanced)
self.aux = self.aux_coef * (f * p.mean(0)).sum()         # L_aux = α · Σ f_i · P_i
```

手算一个例子：4 个 token、2 个专家、top-1。路由概率是 [0.9, 0.1]、[0.8, 0.2]、[0.7, 0.3]、[0.4, 0.6]。前 3 个 token 选专家 1，最后一个选专家 2。所以 f = 2/4 × [3, 1] = [1.5, 0.5]，P = 列平均 = [0.7, 0.3]。L = α × (1.5 × 0.7 + 0.5 × 0.3) = **1.2α**，比均衡时的 α 大。脚本算出来是 0.0120（α = 0.01）。

辅助损失的毛病是：它的梯度和语言模型本身的梯度互相拉扯。α 太小，管不住坍缩；α 太大，又会损害模型质量。DeepSeek-V3 报告引用了 Wang 等人（2024）的实验，正是拿这一点作为改进的动机。

> **注意：**系数不能照搬。不同实现的常数不一样。Hugging Face 的 Mixtral / Qwen3-MoE 实现里，f 没有除以 K，算出来是本章写法的 K 倍。配置里的 `router_aux_loss_coef`（Mixtral 0.02、Qwen3 0.001、Llama 4 0.001、OLMoE 0.01）要配合各自的实现来读。

### 5.2 办法二：无辅助损失的偏置（DeepSeek-V3）

DeepSeek-V3 换了个思路：**不改损失，直接改"选谁"**。每个专家有一个偏置 b_i。路由器只在挑 top-K 时把偏置加到分数上（第 3.1 节公式第 2 行）。门控权重 g 仍然用原始分数 s。偏置不是参数，没有梯度。每个训练步结束后，代码按这一步整批的负载更新偏置：

```
b_i ← b_i + γ · sign(mean load − load_i)       # subtract γ for an overloaded expert, add γ for an underloaded expert
```

```python
@torch.no_grad()
def update_bias(self, load=None):
    load = self.load if load is None else load
    self.bias += self.bias_speed * torch.sign(load.mean() - load)
```

偏置只影响"选谁"，不影响"输出多少"。所以它不会干扰语言模型的梯度。过载的专家会慢慢被"挤"出一部分 token，欠载的专家会慢慢被"推"进来一些 token。

[`code/02_moe_layer.py`](code/02_moe_layer.py) 第 4 部分做了一个最干净的演示。它固定一个严重偏心的路由器：8 个专家，选 2 个。我们人为调高专家 1、2 的路由器权重（代码里的编号是 0、1），输入全是正数。演示**不训练任何权重，只更新偏置**（γ = 0.01）：

| 偏置更新次数 | 各专家负载占比 | 最大 / 平均 | 容量因子 1.25 时丢弃 |
|---:|---|---:|---:|
| 0 | 0.50 0.10 0.00 0.32 0.00 0.08 0.00 0.00 | 4.00 | 50.3% |
| 10 | 0.48 0.12 0.04 0.12 0.06 0.12 0.01 0.04 | 3.87 | 32.8% |
| 30 | 0.13 0.12 0.12 0.13 0.13 0.12 0.13 0.13 | 1.05 | 0.0% |
| 300 | 0.13 0.12 0.12 0.13 0.13 0.12 0.13 0.13 | 1.05 | 0.0% |

（8 专家选 2，均匀负载是每个 0.125。"最大/平均"的上限是 E/K = 4，第 0 步已经顶格。）

DeepSeek-V3 的实际设置（报告 4.2 节）：γ = 0.001。最后 500B token 把 γ 设为 0，但继续使用已经学到的偏置。**另外，DeepSeek-V3 保留了一个极小的序列级辅助损失**，α = 0.0001，"只为防止单条序列内的极端不均衡"。所以"无辅助损失"不是完全没有辅助损失，而是**主要靠偏置**。GLM-4.5（偏置更新率 0.001 + 序列级损失 0.0001）和 NVIDIA Nemotron 3 Nano（更新率 1e-3 + 标准负载均衡损失 1e-4）的配方几乎一样。

**它为什么更好？** DeepSeek-V3 报告表 5 在同样的数据、同样的结构下对照两种办法（都用 sigmoid + top-K 归一化）。在总参数 15.7B 的模型上，Pile 测试集 BPB 从 0.727 降到 0.724。在 228.7B 的模型上，从 0.656 降到 0.652。多数下游基准也更好。

报告 4.5.3 节的分析更有意思：关键不在"有没有损失"，而在**均衡的范围**。序列级的辅助损失要求每条序列内部都均衡，专家就不能按领域分工。偏置法只要求整个 batch 均衡。作者又试了一个"batch 级的辅助损失"。在 1B 的 MoE 上，验证损失分别是 2.258（序列级）、2.253（偏置法）、2.253（batch 级损失），后两者打平。

千问 Qwen3 用的"全局 batch 负载均衡损失"（Qiu 等 2025）走的就是这条路。它仍用辅助损失，但在全局 batch 上统计。

### 5.3 容量因子与"不丢 token"

早期的 MoE（GShard、Switch Transformer）要让每个专家的计算形状固定。所以它们给每个专家设一个**容量**（capacity）：最多处理 `容量因子 × T·K / N` 个 token。超出容量的 token 直接跳过这个专家：这个专家对它的输出记为 0，残差连接照常把它往下传。这叫 **token dropping**（丢 token）。上表最后一列就是这个效应。负载偏心时，容量因子 1.25 会丢掉一半的分配；均衡之后，一个都不丢。

今天的主流是**不丢 token**（dropless）。MegaBlocks（Gale 等 2022）把 MoE 写成块稀疏矩阵乘，专家分到多少 token 就算多少。DeepSeek-V3 报告明确写了"训练和推理都不丢 token"，前提正是负载足够均衡。本章的生产级代码两种方式都支持（`capacity_factor=None` 表示 dropless）。

## 6. 小实验：同一个小模型，稠密 vs MoE，三种均衡

[`code/03_train_compare.py`](code/03_train_compare.py) 用第 10 章的字符级莎士比亚小模型（4 层、宽 128、4 个头，Pre-Norm RMSNorm + RoPE），只换 FFN：

| 方案 | FFN 结构 | 说明 |
|---|---|---|
| 稠密-384 | SwiGLU 宽 384 | 激活参数（≈ 每个 token 的算力）的基准 |
| 稠密-1536 | SwiGLU 宽 1536 | 总参数与 MoE 相当 |
| MoE-无均衡 | 8 专家 × 宽 192，选 2 | 激活宽度 2 × 192 = 384，与稠密-384 算力相同 |
| MoE-辅助损失 | 同上 + L_aux（α = 0.01） | |
| MoE-无辅助损失 | 同上 + 偏置（γ = 0.01） | |
| 细粒度+共享 | 16 专家 × 宽 96，选 3，+ 1 个宽 96 的共享专家，偏置 | 激活宽度 4 × 96 = 384 |

所有 MoE 都用 sigmoid 打分 + top-K 归一化，和 DeepSeek-V3 报告表 5 的对照设定一致。所有方案用同样的数据顺序、800 步、AdamW + warmup + 余弦衰减。每种方案跑 2 个随机种子。γ 取 0.01，比 DeepSeek-V3 的 0.001 大十倍。因为我们只训练 800 步，偏置需要走得快一些。

运行结果（`uv run python chapters/24-mixture-of-experts/code/03_train_compare.py`；FFN 参数为 4 层合计）：

| 方案 | FFN 总参数 | FFN 激活参数 | 验证损失均值 [种子 0, 1] | 负载 最大/平均（4 层 × 2 种子的均值，最坏） | 负载 < 1% 的专家（单层最多） |
|---|---:|---:|---|---:|---:|
| 稠密-384 | 589,824 | 589,824 | 1.793 [1.800, 1.787] | — | — |
| 稠密-1536 | 2,359,296 | 2,359,296 | 1.815 [1.815, 1.816] | — | — |
| MoE-无均衡 | 2,363,392 | 593,920 | 1.780 [1.760, 1.799] | 2.61, 3.63 | 1 |
| MoE-辅助损失 | 2,363,392 | 593,920 | 1.786 [1.772, 1.799] | 1.14, 1.26 | 0 |
| MoE-无辅助损失 | 2,363,392 | 593,920 | 1.786 [1.774, 1.797] | 1.12, 1.20 | 0 |
| 细粒度+共享 | 2,514,944 | 598,016 | 1.777 [1.778, 1.776] | 1.15, 1.22 | 0 |

同一方案换种子，验证损失最多相差 0.039。第 1 层的负载在训练中的变化（种子 0）：

| step | 无均衡 | 辅助损失（α = 0.01） | 偏置（γ = 0.01） |
|---:|---:|---:|---:|
| 0 | 2.76 | 2.76 | 2.76 |
| 100 | 3.34 | 1.76 | 1.08 |
| 300 | 3.61 | 1.34 | 1.18 |
| 800 | 3.63 | 1.05 | 1.20 |

能读出的结论（和不能读出的）：

- **两种负载均衡办法都有效。** 训练结束时，两者的"最大/平均"都在 1.05–1.26 之间，没有闲置专家。不加均衡时，最坏的一层是 3.63，有专家完全闲置。在这个设置下，偏置法拉平得更快：第 1 层 100 步就到 1.08，辅助损失要到 300–800 步。这与"偏置每步都直接作用在选择上"一致。但速度取决于 γ 和 α 的取值，不能推广。
- **验证损失的差别都在噪声内。** 五个激活算力相同的方案（稠密-384 和四种 MoE）落在 1.777–1.793 之间，差距小于种子间的 0.039。我们**不能**据此说 MoE 更好，也不能说负载均衡提高了质量。在 800 步、字符级数据上，不均衡带来的"浪费参数"还来不及体现为损失的差距。
- **"总参数相同"的稠密-1536 反而最差**（1.815，两个种子都是）。宽 4 倍的稠密 FFN，每个 token 的算力也是 4 倍。但在同样的 800 步、同样的学习率下，它没有训练得更好。在这个规模下，数据量和步数才是瓶颈。请记住：MoE 的"同算力、更多参数"只有在数据足够多、训练足够久时才会变成质量。（DeepSeekMoE、Qwen3 的报告里都用了上万亿 token。）

**这是百万参数、几分钟训练的极小实验。** 它只能说明"代码是对的、现象大致如何"，不能外推到大模型。认真的对比请看三个来源。DeepSeek-V3 报告表 5 对比了两种均衡办法。DeepSeekMoE 论文对比了粗粒度和细粒度专家、有无共享专家。Kimi K2 报告图 5 展示了稀疏度的影响。

## 7. 细粒度专家 + 共享专家（DeepSeekMoE）

第 2 节的表里，今天的大模型和 Mixtral 有两个最大的结构差别。两者都来自 DeepSeekMoE（Dai 等 2024）。

**细粒度专家**（fine-grained experts）：把每个专家切成 m 份（宽度变成 1/m），同时让每个 token 选 m 倍数量的专家。总参数和激活参数都不变，但**可能的组合数大大增加**。论文里的例子：16 个专家选 2 个，只有 C(16,2) = 120 种组合。每个专家切成 4 份、64 个选 8 个，有 C(64,8) ≈ 44 亿种组合。组合越多，每个 token 越能"凑"出最合适的一组专家，每个专家也能更专一。（论文图 3：同参数、同激活参数下，切得越细，效果越好。）

代价是：专家太小以后，矩阵乘的效率变低，路由和通信的开销变大。所以专家不会无限细下去。（DeepSeekMoE 16B 就因为效率没有再切细。）

**共享专家**（shared experts）：留出一两个专家，让**每个 token 都经过**，不参与路由。论文的动机是：不同的专家都需要一些通用知识（语法、常见词）。没有共享专家时，很多路由专家会把这些知识重复学一遍。有了共享专家，路由专家就能更专心地分工。

论文里有一个很能说明问题的消融。把 2B 模型的共享专家去掉，换成多激活一个路由专家（算力不变）。Pile 损失从 1.808 升到 2.414。

但**不是所有家族都用共享专家**，这是一条光谱（2026-09 核实）：

| 有共享专家 | 没有共享专家 |
|---|---|
| DeepSeek-V3 / V3.2（1 个）、Kimi K2（1 个）、Kimi K3（2 个）、GLM-4.5 / GLM-5（1 个）、Llama 4（1 个）、Qwen3.5-MoE（1 个，外加一个 sigmoid 门）、Mistral Large 3（1 个）、Nemotron 3 Nano（报告写 2 个；配置里是 1 个宽度加倍的共享专家，二者等价） | Qwen3-MoE（报告原文："Unlike Qwen2.5-MoE, the Qwen3-MoE design excludes shared experts"）、gpt-oss、Mixtral、MiniMax-M2（`shared_intermediate_size: 0`）、小米 MiMo-V2-Flash（`n_shared_experts: null`）、OLMoE |

千问自己的选择也变过几次：Qwen2.5-MoE 有共享专家，Qwen3-MoE 去掉了，Qwen3.5 又加回来了（模型卡："8 Routed + 1 Shared"）。所以更准确的说法是：**细粒度专家是共识；共享专家是大多数家族的选择，但不是必需品**。

本章小实验里的"细粒度+共享"方案（16 选 3 + 1 个共享）验证损失是 1.777。它和其他同算力方案的差距同样在噪声内。它两个种子之间的差距也随机器而变。第 6 节表里是 1.778 和 1.776，几乎一样。在另一台服务器上复跑，结果是 1.757 和 1.778，相差 0.021。只凭两个种子，说明不了它更稳定，也说明不了它更不稳定。

## 8. 为什么大模型都用，小模型很少用

先看产品线（[`code/04_small_vs_large.py`](code/04_small_vs_large.py) 第 1 部分，官方仓库名，2026-09）：

| 家族 | 稠密 | MoE（总 / 激活） |
|---|---|---|
| Qwen3 | 0.6B、1.7B、4B、8B、14B、32B | 30B-A3B（30.5 / 3.3）、235B-A22B |
| Qwen3.5 | 0.8B、2B、4B、9B、27B | 35B-A3B、122B-A10B、397B-A17B |
| gpt-oss | — | 20b（20.9 / 3.6）、120b（116.8 / 5.1） |
| Llama 4 | — | Scout（109 / 17）、Maverick（400 / 17） |
| Nemotron 3 | — | Nano 30B-A3B（31.6 / 3.2） |

同一个家族里，几十 B 以下几乎全是稠密模型，MoE 最小也在 20B 总参数以上。为什么？有三笔账。

**第一笔：MoE 该和谁比？** Qwen3 技术报告表 5 用同样的预训练数据，训练了 Qwen3-30B-A3B 和稠密的 Qwen3-14B：

| Base 模型 | MMLU | MMLU-Pro | GSM8K | MATH | EvalPlus | MGSM | 总 / 激活 |
|---|---:|---:|---:|---:|---:|---:|---|
| Qwen3-14B（稠密） | 81.05 | 61.03 | 92.49 | 62.02 | 72.23 | 79.20 | 14B / 14B |
| Qwen3-30B-A3B（MoE） | 81.38 | 61.49 | 91.81 | 59.04 | 71.45 | 79.11 | 30B / 3B |

两者成绩相当。MoE 的**激活参数只有稠密模型的约五分之一**（每个 token 的算力省了四倍多），但**总参数是稠密模型的两倍多**。换句话说，MoE 是"用显存换算力"。

**第二笔：显存装的是总参数。** 权重必须全部驻留在内存里，包括这一步没有被任何 token 选中的专家：

| 模型 | 激活 | BF16 权重 | 4-bit 权重 |
|---|---:|---:|---:|
| Qwen3.5-0.8B（稠密） | 0.8B | 1.6 GB | 0.4 GB |
| Qwen3.5-9B（稠密） | 9.0B | 18.0 GB | 4.5 GB |
| Qwen3-30B-A3B | 3.3B | 61.0 GB | 15.2 GB |

一台 8 GB 内存的手机，用 4-bit 量化也装不下 30B 级的 MoE。**在端侧，内存是最紧的约束，同样的内存放一个稠密模型更划算。** 在几千张卡的集群上，显存可以靠增加 GPU 解决，最贵的是算力和时间。在那里，MoE 是很好的选择。

**第三笔：读多少权重，取决于 batch。** 解码受显存带宽限制（第 21 章）。MoE 每一步要读的，是这一批 token 选中的那些专家。按均匀路由估计，一层里被至少一个 token 选中的专家比例 = 1 − (1 − K/N)^B：

| 模型 | batch 1 | batch 4 | batch 16 | batch 64 | batch 256 |
|---|---:|---:|---:|---:|---:|
| Qwen3-30B-A3B（128 选 8） | 6.2% | 22.8% | 64.4% | 98.4% | 100.0% |
| DeepSeek-V3（256 选 8） | 3.1% | 11.9% | 39.8% | 86.9% | 100.0% |
| Mixtral（8 选 2） | 25.0% | 68.4% | 99.0% | 100.0% | 100.0% |

单个用户（batch 1）只读激活的那一小部分。所以**只要内存装得下**，在本地跑 30B-A3B 这类模型，解码速度接近 3B 的稠密模型。服务端 batch 很大时，几乎每个专家都要读。MoE 省下的是每个 token 的算力，不是显存带宽。

小模型还有几条特有的劣势。每个 token 的 FFN 本来就小，再切成几十个专家，每个专家的矩阵乘小到 GPU 跑不满。路由、分发、合并的额外开销在耗时里的占比变大。

另外，**同尺寸比较按总参数算**（GOAL.md 第 3.2 节的对手清单就是这样定的）。一个 0.8B 总参数的 MoE，激活参数只有一两亿，质量更接近一个一两亿参数的稠密模型。这就是主线模型保持稠密的原因（GOAL.md 3.3）。

> **注意：**也有反例。AllenAI 的 OLMoE-1B-7B（总 7B、激活 1B，64 专家选 8）是一个完全开放的小 MoE，研究者常把它当作基线。但本章核实过的主力小模型（Qwen3 的 0.6–14B、Qwen3.5 的 0.8–27B）都是稠密的。

## 9. 专家并行（简述）

大 MoE 的专家放不进一张卡。于是我们把不同的专家放在不同的 GPU 上。这叫**专家并行**（expert parallelism，EP）。

每一层 MoE 的前向传播变成三步：

1. **dispatch**（分发）：all-to-all 通信把每个 token 发到它选中的专家所在的卡。
2. 各卡计算自己的专家。
3. **combine**（合并）：再用一次 all-to-all 把结果发回来，加权合并。

专家并行带来两个新问题：

- **通信**：DeepSeek-V3 的 256 个路由专家分布在 8 个节点的 64 张卡上。它用"节点受限路由"让每个 token 最多发往 4 个节点：先按每个节点内最高分之和选节点，再在这些节点里选专家。DeepSeek 还专门写了 all-to-all kernel，让通信与计算重叠。Kimi K2 用了尽量小的 EP = 16，让计算能完全遮住通信。
- **负载**：同一层的所有卡要互相等，最忙的专家决定整层的耗时。所以负载均衡不只是"别浪费参数"，更是吞吐问题。DeepSeek-V3 推理时还把高负载的专家复制几份（"冗余专家"），每 10 分钟按线上统计调整一次。

单卡上的对应问题是：不同专家分到的 token 数不同，没法写成一个规整的大矩阵乘。GPU 上的做法是 **grouped GEMM**，或 MegaBlocks 式的块稀疏矩阵乘。本章的 CPU 代码用 Python 循环代替。

概念上，grouped GEMM 一次调用算完一组形状不同的矩阵乘。底层是否真的融合成一个 kernel，取决于实现和硬件。本章"GPU 实测"一节用 profiler 看到：在 RTX 3090 上，PyTorch 的 `F.grouped_mm` 实际是逐专家发了 128 个 cuBLAS GEMM。

## 10. 小结

- **FFN 是参数和算力的大头**（Qwen3-8B 每层 78%）。MoE 把 FFN 拆成 N 个专家，每个 token 只用 K 个。这让**总参数（显存、知识容量）和激活参数（每个 token 的算力）脱钩**。DeepSeek-V3 激活 5.6%，Kimi K2 只有 3.2%。
- **路由**：线性层打分（softmax 或 sigmoid）→ 分数加偏置后选 top-K → 选中的分数归一化，作为门控权重 → 加权求和。实现上按专家分组做 dispatch / combine。
- **负载均衡**：不做均衡，路由就会坍缩。辅助损失 α·Σf_i·P_i 靠梯度压低过载专家，但会干扰主损失。DeepSeek-V3 的偏置法只改"选谁"，每步按 batch 负载做 ±γ 更新。关键在于按整个 batch 均衡，而不是按每条序列均衡。今天的主流是不丢 token。
- **细粒度专家**（更多、更小的专家）是共识。多数家族采用**共享专家**，但 Qwen3-MoE、gpt-oss、Mixtral、MiniMax-M2 没有。这是一条光谱。
- **MoE 用显存换算力。** 在集群上训练和服务几百 B 以上的模型时，MoE 很划算。在内存最紧的端侧，同样的内存放稠密模型更好。所以 1B 级的主流小模型都是稠密的，主线模型也是。

---

## GPU 实测（单张 RTX 3090）

> **注意：**上面正文里的数字都来自 CPU 运行。本节换到一张 NVIDIA GeForce RTX 3090 上实测（24 GB 显存，Ampere 架构）。它的规格表给出：BF16 张量核稠密峰值约 71 TFLOPS，FP32 约 35.6 TFLOPS，显存带宽约 936 GB/s。环境：PyTorch 2.11.0+cu128、CUDA 12.8，2026 年 10 月。
>
> 服务器把这张卡的功耗上限设成了 240 W（出厂默认 350 W）。持续满载时，这张卡会降频。所以算力和带宽的绝对值比满功耗的 3090 偏低，看相对关系更可靠。没有 GPU 可以跳过本节。

运行：

```bash
uv run python chapters/24-mixture-of-experts/code/05_gpu_moe.py
```

一个 MoE 层一次处理 8192 个 token（相当于训练或 prefill 时的一个批次）。d = 1024，每个专家宽 2048，每个 token 选 2 个专家。激活宽度是 4096，和一个宽 4096 的稠密 SwiGLU 算力相同。专家数 E 从 8 加到 128。

我们比较两种写法。第一种是第 3.2 节的**按专家循环**（把 `02_moe_layer.py` 的 `MoE.forward` 原样搬上 GPU）。第二种是**排序分段 + `F.grouped_mm`**：把 token 按专家排好序，每个矩阵用一次调用算完所有专家。这样整个前向传播不用停下来等 GPU。

权重随机初始化，用 BF16。用 CUDA event 计时，取 30 次的中位数。每个 E 都先对拍。两种写法输出的最大差异小于最大输出的 2%（BF16 舍入误差的量级）。

算力相同的稠密 SwiGLU（宽 4096，13M 参数）：4.27 ms，48.3 TFLOPS。

| 专家数 E | 总参数 | 每 token GFLOP | 每个专家平均分到的 token | 按专家循环 ms | 排序 + grouped_mm ms | 循环 ÷ grouped | grouped 实测 TFLOPS |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 50M | 0.0252 | 2048 | 6.45 | 6.03 | 1.1× | 34.2 |
| 16 | 101M | 0.0252 | 1024 | 7.15 | 6.32 | 1.1× | 32.7 |
| 32 | 201M | 0.0252 | 512 | 9.14 | 6.56 | 1.4× | 31.5 |
| 64 | 403M | 0.0253 | 256 | 15.45 | 8.74 | 1.8× | 23.7 |
| 128 | 805M | 0.0254 | 128 | 28.79 | 10.05 | 2.9× | 20.7 |

这里可以直接看到第 1 节说的"让参数和算力脱钩"。总参数翻了 16 倍，每个 token 的 FLOPs 基本不变（多出来的零头是路由器）。分组写法的耗时也只从 6.03 ms 涨到 10.05 ms。

但"FLOPs 不变"不等于"时间不变"。E = 8 时，MoE 已经比同算力的稠密 FFN 慢四成，因为路由、排序、分发、合并都要时间。E = 128 时，每个专家平均只分到 128 个 token。矩阵乘小到喂不饱 GPU，实测算力从 48.3 掉到 20.7 TFLOPS。这就是第 7 节说的"专家太小以后，矩阵乘的效率变低"。

按专家写 Python 循环，专家越多越吃亏：E = 128 时慢 2.9 倍。每个专家都要调用一次 `nonzero`，停下来等 GPU 告诉它分到了哪些 token。（这台机器上，Python 每下发一个算子还要约 6.4 µs。）这部分耗时花在 CPU 上，会随机器负载浮动。

一个出乎意料的细节：用 profiler 看，`F.grouped_mm` 在这张 3090 上发出 129 个 kernel。其中 128 个是逐专家的 cuBLAS 矩阵乘，没有融合成一个 kernel。（第 9 节说"取决于实现和硬件"，这里就是一例。）`F.grouped_mm` 快在循环写在 C++ 里，不用等 GPU。真正把一组矩阵乘合成一个 kernel 的，是 MegaBlocks、vLLM / SGLang 的 fused MoE 这类专门实现。

## 从极简代码到生产级代码

| 极简代码（`code/`） | 生产级代码（`zero/arch/moe.py`） | 多做了什么，为什么 |
|---|---|---|
| `02_moe_layer.py` 的 `MoE`：用 `nn.ModuleList` 装专家，按专家循环 | `MoEFFN`：专家权重按专家**堆叠**成 `(E, d, h)` 三维张量；token 按专家**稳定排序**后分段计算 | 堆叠权重是 grouped GEMM 和专家并行分片的前提（按第 0 维切给不同的卡）。排序后，每个专家的 token 是连续的，方便换成 GPU kernel。（CUDA + BF16 的前向和反向传播已在 RTX 3090 上验证。这次验证还修了 autocast 下 `index_add_` 的精度 bug，见 [runs/2026-10-01-gpu0-check](../../runs/2026-10-01-gpu0-check/README.md) 第 12 节。grouped GEMM kernel 尚未接入。） |
| 固定 sigmoid 或 softmax、top-k 归一化 | `MoEConfig`：`score_func`、`norm_topk_prob`、`routed_scaling_factor`、`n_shared_experts` / `shared_expert_dim`，字段名对齐 HF 配置 | 能表达不同的配方：DeepSeek-V3（sigmoid、归一化、×2.5、1 个共享专家）、Qwen3-MoE（softmax、归一化、无共享专家）、Mixtral 等 |
| 偏置用这一步的负载更新 | `load_accum` 累计自上次 `update_bias()` 以来的负载；分布式训练时先做 `all_reduce` | 梯度累积时，要按整个全局 batch 的负载更新（DeepSeek-V3："monitoring the expert load on the whole batch"）。多卡路径**尚未在 GPU 上验证** |
| 偏置只在 `balance="free"` 时参与 | 偏置总是参与选择（不更新时保持为 0），并作为 buffer 进入 `state_dict` | DeepSeek-V3 最后 500B token 把 γ 设为 0，但继续用已学到的偏置。续训时必须恢复偏置 |
| 无 | `capacity_factor`：可选的容量上限，统计 `last_dropped`；默认 `None` = dropless | 对照 Switch / GShard 的丢 token 做法 |
| 在训练循环里手动加 `m.aux`、调用 `update_bias` | `MoETransformer`（`moe_transformer(model_cfg, moe_cfg, first_dense)`）：前 `first_dense` 层保留稠密 FFN（DeepSeek-V3 为 3、Kimi K2 为 1）；`loss()` 自动加上各层的辅助损失；`after_step()` 更新偏置；`load_stats()`、`param_counts()` | 直接复用主线的注意力、RMSNorm 和初始化规则。主线的 `Trainer` 尚未调用 `after_step`（第二步如果要用 `Trainer` 训练 MoE，再接入） |
| 无 | 行业实现：HF transformers 的 `Qwen3MoeSparseMoeBlock`、`DeepseekV3MoE`（含 `e_score_correction_bias`、分组 top-k）；MegaBlocks；DeepSeek 开源的 DeepEP（专家并行通信库）；vLLM / SGLang 的 fused MoE kernel | 本课的 MoE 只追求可读和正确。真正训练和上线，要用这些实现 |

**对拍**（`uv run pytest tests/test_arch_moe.py`；本机 12 项全部通过，约 5 秒）：

- 1 个专家 + top-1 时，`MoEFFN` 与 `zero.model.SwiGLU` 在 float64 下差异 < 1e-12。
- 排序分段的实现与逐 token 朴素循环完全一致（softmax / sigmoid、有无归一化、`routed_scaling_factor`、共享专家、非零偏置，4 种组合）。
- 偏置改变选择，但不改变门控权重。辅助损失与手算一致（1.2α；均衡时 = α），梯度方向压低过载专家。
- 偏置更新的方向正确。只靠偏置更新，就能把一个固定偏斜路由器的负载从"最大/平均 > 2"拉到 < 1.2。
- 容量因子：丢弃数与手算一致。容量足够大时，输出与 dropless 相同。
- `MoETransformer` 能训练，辅助损失有梯度，偏置进入 `state_dict`，激活参数计数正确。

---

## 前沿观察

> **稀疏度还在往上走。** Kimi K2 报告的"稀疏度 scaling law"（固定激活参数，专家越多，损失越低）给出了继续加专家的理由。Kimi K3 的配置是 896 个专家选 16 个，外加 2 个共享专家。Qwen3.5-397B-A17B 是 512 选 10。但专家越多，路由、通信和负载均衡越难。Kimi K2 报告也明确说，稀疏度 48 是"性能与基础设施复杂度的折中"。最优稀疏度会停在哪里，目前没有共识。

> **均衡的"范围"比"手段"更重要？** DeepSeek-V3 报告 4.5.3 节发现，batch 级辅助损失和偏置法效果打平。千问用全局 batch 的辅助损失，DeepSeek 系用偏置，两条路都在大模型上成功了。它们在工程上各有利弊（偏置法不干扰梯度；辅助损失不需要额外的状态）。哪个更好，还没有定论。

---

## 采用方与来源

| 技术 | 采用方（主力版本） | 来源 |
|---|---|---|
| MoE（细粒度，数十到数百个专家） | DeepSeek（V3：256 选 8）、Qwen（Qwen3：128 选 8；Qwen3.5：512 选 10）、Kimi（K2：384 选 8）、GLM（4.5：160 选 8）、gpt-oss（120b：128 选 4）、Llama 4、MiniMax（M2：256 选 8）、Mistral（Large 3：128 选 4）、NVIDIA Nemotron 3（128 选 6） | 各技术报告与 `config.json`（下方链接） |
| 共享专家 | DeepSeek-V3、Kimi K2/K3、GLM-4.5/5、Llama 4、Qwen3.5、Mistral Large 3、Nemotron 3；**不用**：Qwen3-MoE、gpt-oss、Mixtral、MiniMax-M2、MiMo-V2-Flash | DeepSeekMoE arXiv:2401.06066；Qwen3 报告第 2 节；各配置 |
| 辅助损失（Switch / GShard 式） | Mixtral（`router_aux_loss_coef` 0.02）、Qwen3（全局 batch 负载均衡损失，0.001）、Llama 4（0.001）、OLMoE（0.01） | Switch Transformer arXiv:2101.03961；GShard arXiv:2006.16668；Qwen3 报告；各配置 |
| **无辅助损失的偏置均衡** | **报告明确写出**：DeepSeek-V3（2.1.2、4.2 节，γ = 0.001）、GLM-4.5（2.1、2.4 节："loss-free balance routing"，更新率 0.001）、NVIDIA Nemotron 3 Nano（2.4 节："DeepSeek's aux-loss-free load balancing strategy"，更新率 1e-3）；**配置可见**：Kimi K2 与 K3（`topk_method: noaux_tc`）、GLM-5（`noaux_tc`）、DeepSeek-V3.2（`noaux_tc`）、MiniMax-M2（`use_routing_bias: true`、`e_score_correction_bias`）、小米 MiMo-V2-Flash（`noaux_tc`） | DeepSeek-V3 arXiv:2412.19437；GLM-4.5 arXiv:2508.06471；Nemotron 3 Nano arXiv:2512.20848；Wang 等 arXiv:2408.15664；各配置 |
| 不丢 token（dropless） | DeepSeek-V3（报告 2.1.2 节"No Token-Dropping"）；Mixtral 发布时向 vLLM 提交了集成 MegaBlocks kernel 的实现（Mixtral 论文第 1 节） | arXiv:2412.19437；MegaBlocks arXiv:2211.15841 |

**共识判断（GOAL.md 2.1）**：

- **MoE（细粒度专家）**：DeepSeek、Qwen、Kimi、GLM、gpt-oss、Llama、MiniMax、Mistral、Nemotron 九个家族的旗舰都是 MoE。这满足规则 A，所以进正文。
- **共享专家**：DeepSeek、Kimi、GLM、Llama、Qwen3.5、Mistral、Nemotron 等 7 个家族采用。这满足规则 A，所以进正文。但 Qwen3-MoE、gpt-oss、MiniMax-M2 等不用。所以正文按"光谱"讲，不说成必需。
- **无辅助损失的负载均衡**（GOAL.md 表中列为"待核实"）：在技术报告里明确写出的，有 DeepSeek、GLM（智谱）、NVIDIA 三个彼此独立的家族。另有 Kimi、MiniMax、小米三家的官方配置显示了同样的机制。（Kimi K2 沿用 DeepSeek-V3 架构，独立性稍弱，但 Kimi K3 仍在用。）**这满足规则 A，本章把它放进正文。** 同时如实说明：它不是唯一答案。Qwen、Llama、Mixtral、OLMoE 仍用辅助损失（Qwen3 用全局 batch 版本）。而且"无辅助损失"的采用者都还保留一个极小的序列级辅助损失。

**模型配置**（2026-09 通过 Hugging Face 读取）：
[DeepSeek-V3](https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/config.json)、
[DeepSeek-V3.2](https://huggingface.co/deepseek-ai/DeepSeek-V3.2/blob/main/config.json)、
[Kimi-K2-Instruct](https://huggingface.co/moonshotai/Kimi-K2-Instruct/blob/main/config.json)、
[Kimi-K3](https://huggingface.co/moonshotai/Kimi-K3/blob/main/config.json)、
[Qwen3-235B-A22B](https://huggingface.co/Qwen/Qwen3-235B-A22B/blob/main/config.json)、
[Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B/blob/main/config.json)（模型卡：30.5B / 3.3B）、
[Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B/blob/main/config.json)、
[Qwen3.5-35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B/blob/main/config.json)（模型卡："8 Routed + 1 Shared"）、
[Qwen3.5-397B-A17B（读自 FP8 版）](https://huggingface.co/Qwen/Qwen3.5-397B-A17B-FP8/blob/main/config.json)、
[GLM-4.5](https://huggingface.co/zai-org/GLM-4.5/blob/main/config.json)、
[GLM-5](https://huggingface.co/zai-org/GLM-5/blob/main/config.json)、
[gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b/blob/main/config.json)、
[gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b/blob/main/config.json)、
[Mixtral-8x7B-v0.1](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1/blob/main/config.json)、
[Llama-4-Scout（unsloth 镜像）](https://huggingface.co/unsloth/Llama-4-Scout-17B-16E-Instruct/blob/main/config.json)、
[MiniMax-M2](https://huggingface.co/MiniMaxAI/MiniMax-M2/blob/main/config.json)、
[MiMo-V2-Flash](https://huggingface.co/XiaomiMiMo/MiMo-V2-Flash/blob/main/config.json)、
[NVIDIA-Nemotron-3-Nano-30B-A3B](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/blob/main/config.json)、
[Mistral-Large-3 的 params.json](https://huggingface.co/mistralai/Mistral-Large-3-675B-Instruct-2512/blob/main/params.json)、
[OLMoE-1B-7B](https://huggingface.co/allenai/OLMoE-1B-7B-0924/blob/main/config.json)。

说明：Meta 官方的 Llama 4 仓库需要申请权限，所以本章读取的是 unsloth 镜像（它的 `base_model` 指向官方仓库）。Llama 4 的共享专家见 HF transformers 的 `Llama4TextMoe.shared_expert`（本构建环境无法访问 Meta 博客）。本章没有核实 MiniMax-M2 的总参数 / 激活参数，只引用了它的路由配置。按配置数出的 Kimi K2 总参数比报告少约 14B，**待核实**。

---

## 引导问题

向 Claude Code 提出这些问题。一直问到你能用自己的话讲清楚答案：

1. DeepSeek-V3 的门控权重用原始分数 s，选专家用 s + b。如果门控权重也用 s + b，会出什么问题？（提示：偏置会怎样混进语言模型的梯度？）
2. 专家数从 8 增加到 256 时，softmax 打分和 sigmoid 打分各自会遇到什么问题？为什么 sigmoid + 选中后归一化更适合细粒度专家？
3. 辅助损失里的 f_i 是不可导的计数。那么梯度怎样让过载专家分到的 token "变少"？对第 5.1 节的手算例子求 ∂L/∂(logit)，看它的符号。
4. 本章第 8 节的表说，batch 64 时 Qwen3-30B-A3B 几乎每个专家都要读。那 MoE 在服务端到底省了什么？batch 很大时，和一个 3B 稠密模型比，解码每一步的时间主要差在哪里？
5. 细粒度把专家切得越来越小。为什么不切到每个专家只有一行（宽度 1）？从矩阵乘效率、路由器参数和通信三方面想一想。
6. 本章小实验里，MoE 和两个稠密基线的差距，与种子之间的波动相比，哪个更大？如果要认真比较，你会怎样设计实验（数据量、步数、指标）？参考 DeepSeekMoE 论文第 4 节的做法。

## 动手任务

每个任务都要运行代码，并查看结果。

**任务 1（基础）**：在 Hugging Face 上挑一个本章没算过的 MoE 模型（例如 Qwen3.5-122B-A10B 的文本部分、GLM-4.5-Air、OLMoE-1B-7B）。读它的 `config.json`，把字段加进 `01_param_ledger.py` 的 `MOE`。算出总参数和激活参数，和模型卡对照。说清楚差异来自哪种口径。

**任务 2（核心）**：在 `03_train_compare.py` 里加两个方案：(a) 辅助损失 α = 0.1；(b) 偏置 γ = 0.001（DeepSeek-V3 的值）。各训练一次（可以只跑种子 0）。比较训练结束时的"最大/平均"负载和验证损失。α 太大时，损失有没有变差？γ 太小时，800 步够不够把负载拉平？

**任务 3（挑战）**：用 `zero/arch/moe.py` 的 `moe_transformer` 搭一个和主线结构相同的小 MoE（`configs/tiny` 的尺寸）。写一个训练循环（每步 `loss()` → `backward` → `step` → `after_step()`）。在 `assets/tiny_corpus` 上训练几百步，每 50 步打印 `load_stats()`。然后把 `capacity_factor` 设成 1.0，记录 `last_dropped` 在训练中的变化。负载均衡之后，丢弃率是不是趋近于 0？

---

## 想深入：CS336

本章对应斯坦福 CS336（Spring 2026）<https://cs336.stanford.edu/>：

- **第 4 讲：注意力的替代方案与 MoE。** 讲义与录像见课程页。（本构建环境访问不了课程页。这一讲具体覆盖了哪些 MoE 细节，以讲义为准。）本章的路由、top-k、负载均衡、细粒度 / 共享专家是这一讲 MoE 部分的核心内容。`02_moe_layer.py`、`03_train_compare.py` 可以当作它的最小可运行版本。
- **CS336 未深入**：本章有两部分是本课结合主线模型的决策补充的。第 2 节按官方配置逐项核对总参数 / 激活参数。第 8 节给出"为什么小模型少用 MoE"的三笔账。专家并行的工程细节（all-to-all、节点受限路由、冗余专家），可以接着看 CS336 第 7、8 讲（并行）和 DeepSeek-V3 报告第 3 节。

---

## 本章参考文献

- Shazeer et al. *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer*（MoE 层与路由坍缩问题），2017：<https://arxiv.org/abs/1701.06538>
- Lepikhin et al. *GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding*（top-2 路由、容量、辅助损失、专家并行），2020：<https://arxiv.org/abs/2006.16668>
- Fedus, Zoph, Shazeer. *Switch Transformers*（top-1 路由、负载均衡损失、容量因子），2021：<https://arxiv.org/abs/2101.03961>
- Gale et al. *MegaBlocks: Efficient Sparse Training with Mixture-of-Experts*（dropless、块稀疏矩阵乘），2022：<https://arxiv.org/abs/2211.15841>
- Jiang et al. *Mixtral of Experts*，2024：<https://arxiv.org/abs/2401.04088>
- Dai et al. *DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models*（细粒度专家、共享专家），2024：<https://arxiv.org/abs/2401.06066>
- Wang et al. *Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts*，2024：<https://arxiv.org/abs/2408.15664>
- DeepSeek-AI. *DeepSeek-V3 Technical Report*（2.1.2 节 DeepSeekMoE 与无辅助损失均衡；3.2、3.4 节专家并行与部署；4.5.2–4.5.3 节消融），2024：<https://arxiv.org/abs/2412.19437>
- Qwen Team. *Qwen3 Technical Report*（第 2 节 MoE 结构：去掉共享专家、全局 batch 负载均衡；表 5 MoE 与稠密对照），2025：<https://arxiv.org/abs/2505.09388>
- Kimi Team. *Kimi K2: Open Agentic Intelligence*（2.3 节架构与稀疏度 scaling law；2.4 节专家并行），2025：<https://arxiv.org/abs/2507.20534>
- GLM-4.5 Team. *GLM-4.5: Agentic, Reasoning, and Coding (ARC) Foundation Models*（2.1、2.4 节），2025：<https://arxiv.org/abs/2508.06471>
- OpenAI. *gpt-oss-120b & gpt-oss-20b Model Card*（表 1 参数分解；2.2 节 MoE），2025：<https://arxiv.org/abs/2508.10925>
- NVIDIA. *Nemotron 3 Nano: Open, Efficient Mixture-of-Experts Hybrid Mamba-Transformer Model for Agentic Reasoning*（2.1、2.4 节），2025：<https://arxiv.org/abs/2512.20848>
- Muennighoff et al. *OLMoE: Open Mixture-of-Experts Language Models*，2024：<https://arxiv.org/abs/2409.02060>
- Hugging Face transformers 的 MoE 实现（`models/qwen3_moe`、`models/deepseek_v3`、`models/llama4`）：<https://github.com/huggingface/transformers/tree/main/src/transformers/models>
- DeepSeek. DeepEP（专家并行通信库）：<https://github.com/deepseek-ai/DeepEP>
- [CS336](https://cs336.stanford.edu/) 第 4 讲（注意力的替代方案与 MoE）
- [1.5 万字速通 LLM 主流模型结构（Llama、Qwen、GLM、DeepSeek…）](https://zhuanlan.zhihu.com/p/2060741715095560795)：各家 MoE 结构的横向对比（`references.md` 已收录）
- [Marin 535B-A23B 训练直播](https://wandb.ai/marin-community/marin_moe/reports/535B-A23B-18T-Token-Hero-Run-Scaling-Ladder--VmlldzoxNzc2MDM5Ng)：一个完全公开的大 MoE 训练过程与阶梯实验（`references.md` 已收录）
- 模型配置链接见上方"采用方与来源"。

**下一章**：DeepSeek-V3、GLM-4.5 的配置里都有一个 `num_nextn_predict_layers: 1`。它们在训练时不只预测下一个 token，还多预测一个。第 25 章讲多 token 预测（MTP），以及它怎样和推测解码结合，让生成快将近一倍，而且不损失质量。
