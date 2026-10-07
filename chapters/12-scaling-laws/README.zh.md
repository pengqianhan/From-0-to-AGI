# 第 12 章：Scaling law 与实验设计 —— 先用小模型算清楚，再花钱

[English](README.md) · **中文**

> **目标**：读完这一章，你能用 C ≈ 6ND 算出一次训练要多少算力和钱。你能说清 Chinchilla 的"每个参数约 20 个 token"是什么意思，以及小模型为什么要"过训练"。你能亲手跑一个迷你阶梯实验。先给每个尺寸调好学习率，再拟合 L(N, D)，外推到大一号的模型，最后真的训练它来对答案。最后，你能讲出主线模型为什么定为 0.69B × 约 400B token，以及闸门 1 的报告要回答什么。

📺 **本章视频**：还没有发布。要在本机渲染，运行 `bash chapters/12-scaling-laws/video/build.sh`。
🧪 **本章自检**：学完后，在 Claude Code 里输入 `/ch12-scaling-laws`。

---

上一章我们先定好了考卷：考哪些基准、和谁比、怎样才算"超过"。这一章解决一个现实问题：**主线模型的预训练只有大约 $5,000，而且只有一次机会**。模型该做多大？训练多少 token？学习率取多少？花钱之前，怎么知道这笔钱能换来什么？

这些问题不能靠"试试看"回答，因为一次完整预训练要 8 张 H100 跑十天。好在语言模型有一个有用的规律：**损失随算力的变化很平滑**。它平滑到可以用一串便宜的小模型，提前算出大模型的结果。这个规律叫 **scaling law**（缩放定律）。这一章把它用在实验设计上。

本章代码如下。所有脚本都在 CPU 上运行。脚本 3、4、6 会把结果缓存在 `out/ch12/`，再跑时直接读缓存。

```bash
uv run python chapters/12-scaling-laws/code/01_flops.py         # where 6ND comes from: measured vs formula (a few seconds)
uv run python chapters/12-scaling-laws/code/02_chinchilla.py    # Chinchilla arithmetic: optimal allocation, overtraining, inference cost (<1 s)
uv run python chapters/12-scaling-laws/code/03_lr_sweep.py      # mini ladder, step 1: sweep the learning rate for each size (about 7 min, one thread)
uv run python chapters/12-scaling-laws/code/04_mini_ladder.py   # fit L(N,D), extrapolate, held-out test (about 15 min, one thread)
uv run python chapters/12-scaling-laws/code/05_plan_budget.py   # how many tokens $5,000 buys (<1 s)
uv run python chapters/12-scaling-laws/code/06_muon.py          # Muon vs AdamW (about 5 min, one thread)
```

> **注意：**这些时间是在一台被多个任务共享的 CPU 上实测的。空闲的笔记本会快几倍。

## 1. 算账：C ≈ 6ND

先学会算账。第 4 章讲过，线性层 y = Wx 的反向传播要做**两个**矩阵乘。一个对输入求梯度（Wᵀg，传给前一层）。另一个对权重求梯度（g xᵀ，给优化器用）。把它们和前向传播加在一起：

| 阶段 | 每个参数、每个 token | 来源 |
|---|---|---|
| 前向传播 | 1 次乘加 = 2 FLOPs | y = Wx |
| 反向传播：对输入 | 2 FLOPs | Wᵀg |
| 反向传播：对权重 | 2 FLOPs | g xᵀ |
| **合计** | **6 FLOPs** | |

所以，训练 N 个参数的模型、用 D 个 token，总算力（compute）是：

```
C ≈ 6 · N · D
```

这是整章最重要的一行公式。只差一件事。第 8 章的注意力里还有两个**没有参数**的矩阵乘：QKᵀ 和 AV。它们每层、每个 token 的前向是 4·d_attn·T，加上反向是 12·d_attn·T（T 是序列长度）。完整的公式是：

```
每 token FLOPs = 6·N_matmul + 12·L·d_attn·T
```

`N_matmul` 是参与矩阵乘的参数。它包括输出层 lm_head，不包括 embedding 查表，因为查表不做乘法。这个数不是估计，可以直接数出来。`01_flops.py` 用 PyTorch 的 `FlopCounterMode`，把第 9 章小模型一次前向 + 反向的矩阵乘全部数了一遍：

```python
def formula_flops_per_token(n_matmul: int, n_layers: int, d_attn: int, T: int) -> float:
    return 6 * n_matmul + 12 * n_layers * d_attn * T
```

| dim | 层 | T | N_matmul | 6N | 注意力项 | 公式合计 | 实测 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | 2 | 64 | 116,736 | 700,416 | 98,304 | 798,720 | 798,720 |
| 128 | 4 | 128 | 811,008 | 4,866,048 | 786,432 | 5,652,480 | 5,652,480 |
| 128 | 4 | 512 | 811,008 | 4,866,048 | 3,145,728 | 8,011,776 | 8,011,776 |

最后两列逐位相等。（RMSNorm、softmax、SiLU 这些逐元素运算不到 1%，计数器和公式都不算它们。）再用同一个公式算主线模型（`configs/main/pretrain.toml`：dim 1280、28 层、16 个查询头 × 128、FFN 3584、词表 65,536）：

```
Total parameters N = 689.5M, parameters in matrix products N_matmul = 689.4M
Per token: 6·N_matmul = 4.137e+09, attention term 12·L·d·T = 2.819e+09 (40.5% of the total)
Total 6.955e+09 FLOPs/token; rough estimate 6·N_total = 4.137e+09 (too low by 40.5%)
Train on 400B tokens: C = 2.78e+21 FLOPs (rough estimate 6ND = 1.65e+21)
```

有两个细节要记住：

- **序列越长，注意力越贵。**主线的 q_dim（2048）比 dim（1280）还宽。序列长度为 4096 时，注意力占了四成。缩短预训练序列是一个省钱的办法（第 9 节）。
- **计数口径（counting convention）。**这个公式没有为因果掩码减半（PaLM、nanochat 和 `zero` 都这么算）。实际运行时，FlashAttention 会跳过被遮住的一半。所以同一次训练，按不同口径报出的 MFU（model FLOPs utilization，模型实际用到的 GPU 峰值算力的比例）能差不少。Porian et al. 2024 专门讨论过这件事。**预算要按实测的 token/s 算，不要只看 MFU 这个数。**

## 2. Scaling law：损失是算力的幂律

用不同的数据量训练不同大小的模型，把结果画在"算力 – 损失"的对数坐标上。你会看到一幅很规整的图（视频 S04）。每个固定大小的模型，损失先快速下降，然后被模型容量卡住，曲线变平。所有曲线的下沿是"给定算力能达到的最好损失"。这条下沿在对数坐标里几乎是一条直线，也就是一个**幂律**（power law）。

**历史**：Kaplan et al.（2020，OpenAI）第一次系统地测出了这个规律。他们的结论是：算力增加时，模型应该长得比数据快，N_opt ∝ C^0.73。GPT-3 就是按这个思路设计的（175B 参数，只训练了 300B token）。

**Chinchilla**（Hoffmann et al. 2022，DeepMind）重做了实验，用一个公式描述整张图：

```
L(N, D) = E + A / N^α + B / D^β
```

- E：数据本身的不确定性。模型再大、数据再多，也降不下这一部分。
- A/N^α：模型太小的代价。
- B/D^β：数据太少的代价。

给定算力 C = 6ND，把 D = C/(6N) 代进公式，对 N 求最小值，就得到"算力最优"（compute-optimal）的分配。结论和 Kaplan 很不一样：**模型和数据应该同比例增长，大约每个参数 20 个 token**。Chinchilla 70B 用 1.4T token 训练，打败了比它大 4 倍的 Gopher。

`02_chinchilla.py` 用两组公开系数做这个算术。这里用的是 Epoch AI 的复现版本（Besiroglu et al. 2024）。他们发现 Chinchilla 原文的拟合有偏差，原因是优化器提前停止了。复现后的系数和 Chinchilla 实际采用的 20 token/参数一致：

```python
def compute_optimal(C, E, A, B, alpha, beta):
    """Closed-form solution: N_opt = G · (C/6)^(β/(α+β)), with G = (αA / βB)^(1/(α+β))."""
    G = (alpha * A / (beta * B)) ** (1 / (alpha + beta))
    N = G * (C / 6) ** (beta / (alpha + beta))
    return N, C / (6 * N)
```

| 算力 C | N_opt | D_opt | token/参数 |
|---:|---:|---:|---:|
| 1e19 | 0.26B | 6.4B | 24.2 |
| 1e21 | 2.78B | 60.0B | 21.6 |
| 1.65e21（主线的量级） | 3.59B | 76.6B | 21.3 |
| 1e23 | 29.45B | 566.0B | 19.2 |
| 1e25 | 312.07B | 5340.7B | 17.1 |

（同一段代码用 Chinchilla 原文的系数算，C=1e23 时是 78 token/参数。这就是复现者指出的偏差。）

**为什么 Kaplan 和 Chinchilla 结论不同？**Porian et al.（2024）用 900 多次训练把分歧拆开了。Kaplan 没有把最后一层（lm_head）的算力算进去。warmup 对小模型太长。还有，**小模型的超参没有按尺寸调好**。做了这三项修正之后，结果和 Chinchilla 吻合。请记住最后一条，第 4 节还会遇到它。

> **绝对数值不能搬**：这些系数对应的是 Chinchilla 的数据和分词器。能搬到我们身上的是"形状"：最优比例的量级，以及过训练的代价。我们自己的系数要用自己的阶梯实验拟合（第 5、7 节）。

## 3. 为什么小模型要"过训练"

但在现实中，几乎没有小模型按每个参数 20 个 token 来训练：

| 模型 | 参数 | 训练 token | token/参数 |
|---|---:|---:|---:|
| Chinchilla | 70B | 1.4T | 20 |
| 本课主线（计划） | 0.69B | 约 400B | 约 580 |
| Puro-2B | 约 2B | 1.4T | 约 700 |
| Llama 3 8B | 8B | 15T | 约 1,875 |
| MobileLLM-R1-950M | 0.95B | 4.2T | 约 4,400 |
| Qwen3-0.6B | 0.6B | 36T | 约 60,000 |

原因是 Chinchilla 只算了**训练**的账。模型只训练一次，却要被使用上亿次。每生成一个 token 大约要 2N FLOPs。把推理也算进总账（Sardana & Frankle 2023 的思路），目标就变成"达到某个损失，训练 + 推理的总算力最小"：

```python
            total = 6 * N * D + 2 * N * D_inf
```

`02_chinchilla.py` 第 ③ 部分（目标损失取主线的损失，Epoch 系数）：

| 一生要生成的 token | 总算力最小的 N | D | token/参数 |
|---:|---:|---:|---:|
| 0（只算训练） | 2.25B | 49B | 22 |
| 1e12 | 0.84B | 228B | 271 |
| 1e13 | 0.56B | 901B | 1,610 |
| 1e14 | 0.44B | 4,184B | 9,498 |

要服务的 token 越多，越该用小模型、训练更久。这就是 Qwen3、Llama 3、MobileLLM 这些小模型"过训练"（overtraining）几百到几万倍的原因。

过训练也有代价。在我们的预算（C = 6 × 0.69B × 400B ≈ 1.65e21）下，脚本输出：

```
Compute-optimal:  N = 3.60B, D = 77B → L = 2.2636
Main-line choice: N = 0.69B, D = 400B (580 tokens/parameter) → L = 2.3426
Cost of overtraining: the loss is higher by 0.0790 (3.5%); for the same loss, the optimal allocation needs only 40% of the compute
```

Delphi 的报告给出了一个大尺度上的同类数字。过训练 10 倍时，算力最优的配置只要约 1/6 的算力，就能达到同样的损失。（他们提醒，这个数外推得很远，可信度低于主结论。）这是一笔价格明确的交易。我们选 0.69B，是因为目标是端侧可用、与 Qwen3.5-0.8B 同级比较（GOAL.md 3.3）。我们不是因为它"最优"才选它。

## 4. 用小实验拟合：先调参，再拟合

现在自己做一遍阶梯实验（scaling ladder：一串尺寸逐级变大的模型）。所有结果都是**极小配置演示**。模型是第 9 章的 TinyTransformer（字节级，词表 256），有 1 万到 50 万参数。数据是 `assets/tiny_corpus/shakespeare.txt`。它验证的是方法，不代表主线模型的任何数字。

> **注意：**关于数字。本章训练类实验的数字来自课程构建机上的一次 CPU 运行。不同机器、不同版本的底层数学库，浮点运算的顺序略有不同。训练几百步后，这些微小差异会被放大。你本机的数字可能从小数点后第二、三位开始就不一样。请以下文中不依赖具体数值的结论为准。2026-10 在另一台服务器上的复跑对照见 [runs/2026-10-01-gpu0-check/chapters-11-15.md](../../runs/2026-10-01-gpu0-check/chapters-11-15.md)。

| 尺寸 | dim | 层 | 非 embedding 参数 N |
|---|---:|---:|---:|
| s1 | 16 | 2 | 10,896 |
| s2 | 32 | 2 | 31,968 |
| s3 | 48 | 3 | 95,664 |
| s4 | 64 | 4 | 217,792 |
| s5（留出） | 96 | 4 | 467,936 |

这里的 N 是参与矩阵乘的参数（含 lm_head，不含 embedding 查表）。它和第 1 节的 6N 是同一个口径。Porian et al. 发现，"不算 lm_head"正是 Kaplan 偏差的来源之一。另外还有两个根据实测做的小调整：

- 输入 embedding 和 lm_head 不共享权重。在这个尺度上，共享会让模型在"只会猜高频字节"的平台期卡很久。曲线的噪声会大到拟合不出规律。
- batch 用 8×64。同样的 token 数可以走更多步，在这个尺度上学得更快。

**第一步是调学习率。**Lourie et al.（2026，arXiv:2608.11859）的结论是本章最重要的警告。小模型对超参很敏感，没调好的小实验会把 scaling law 扭歪。只要调得足够好，4M 参数的模型上就已经有清晰的规律。`03_lr_sweep.py` 给每个尺寸扫 5 个学习率（相邻两个差 2 倍）。**最优值落在网格边上时，就向那一侧再扩一格**，直到最优值在网格内部。然后在最优点和两个邻居上拟合一条抛物线，取顶点：

```
Validation loss (bit/byte). One row per size. * = best for that size, - = not run:
size        N |  0.00125   0.0025    0.005     0.01     0.02     0.04
  s1   10,896 |        -  3.5129   3.3643   3.3191   3.2778*  3.3372
  s2   31,968 |        -  3.2625   3.1996   3.1669*  3.2353   3.3323
  s3   95,664 |  3.3094   3.1940*  3.2385   3.3055   3.3848   3.4342
  s4  217,792 |  3.2189   3.1401*  3.2746   3.3019   3.3805   3.3845
η* after parabola interpolation: s1 0.01879, s2 0.00885, s3 0.002915, s4 0.002284

Fit η*(N) = 18.6 · N^(-0.744)
Extrapolate to the held-out size s5 (N=467,936): η* ≈ 0.001123
```

最优学习率随模型变大一路往下走。假设你偷懒，所有尺寸都用最小模型调出来的 0.02：

```
  s1 N= 10,896  3.2778  worse than tuned by +0.0000
  s2 N= 31,968  3.2353  worse than tuned by +0.0684
  s3 N= 95,664  3.3848  worse than tuned by +0.1908
  s4 N=217,792  3.3805  worse than tuned by +0.2405
```

这时 s3、s4 反而比 s1、s2 **更差**。拿这组数据去拟合，结论会是"模型做大没用"。这个结论完全错误。这正是 Kaplan 与 Chinchilla 分歧的来源之一。它也是后面"主线进度"里 zero 流水线演示失败的原因。

学习率之外的超参怎么随尺度变？目前的共识做法是"用小实验拟合"：

- DeepSeek LLM 拟合了最优学习率和最优 batch 随算力变化的幂律（η_opt = 0.3118·C^−0.125，B_opt = 0.292·C^0.327）。
- Qwen3 报告说，它用 scaling law 预测每个模型的学习率调度和 batch。
- Llama 4 用自研的 MetaP，让超参跨宽度、深度、batch、训练长度迁移。

具体的迁移理论（μP 及其变体）还没有形成共识。它们放在章末的"前沿观察"里。

## 5. 迷你阶梯：拟合 L(N, D)，再外推

每个尺寸有了 η\* 之后，`04_mini_ladder.py` 训练阶梯。它用一个省算力的技巧：**WSD 分叉**（WSD branch）。第 6 章的 WSD 调度在稳定段的学习率是常数。所以每个尺寸只训练一条主干。在 0.8·D_k 处复制一份主干，接一段 0.2·D_k 的衰减，就得到预算 D_k 的结果：

```python
    for step in range(trunk_end + 1):
        if step in branch_at:  # branch: copy the current state and add a decay
            D = branch_at[step]
            m2 = copy.deepcopy(model)
            ...
            for i in range(n_decay):
                train_step(m2, o2, *batch(train, g2), lr * (1 - (i + 1) / n_decay))
            results[D] = val_bpb(m2, val)
```

这样 3 个预算只花约 1.15 倍最大预算的算力（而不是 1.75 倍）。主线的预训练配置也是这样设计的（`decay_frac = 0`，衰减留给第 15 章的中期训练）。

每个尺寸有 3 个预算（64K、128K、256K 字节），两个最小的尺寸再补一个 1M 字节的预算。**阶梯必须覆盖主线所在的"过训练"区。**否则，N 方向的指数定不下来。第一次只跑了 12 个点，拟合出 α ≈ 0.05。这几乎是在说"N 不重要"，因为所有点都在数据不够的区域。补上这两个点后，α 才不再贴着 0。但下面会看到，α 仍然没有被定准。

拟合用"变量投影"（variable projection）。固定 (α, β) 时，L 对 (E, A, B) 是线性的。程序对 α、β 做网格搜索，在每个格点解一个 3 元最小二乘，得到 E、A、B：

```python
    X = np.stack([np.ones((len(a), len(L))), n[None] ** -a[:, None], d[None] ** -b[:, None]], axis=2)
    ...
    sse[(coef < 0).any(1)] = np.inf  # E, A, and B must all be non-negative
```

结果（14 个点，课程构建机）：

```
Fit (14 ladder points): L(N, D) = 2.557 + 19.23/N^0.52 + 520.6/D^0.58
size        N         D   D/N |  actual     fit   error
  s1   10,896    65,536     6 |  3.4853  3.5478  +1.79%
  s1   10,896 1,048,576    96 |  2.9307  2.8781  -1.80%
  s2   31,968 1,048,576    33 |  2.7765  2.8125  +1.30%
  s4  217,792   262,144     1 |  2.9258  2.9644  +1.32%
  ... (the full 14 rows are in the script output; maximum error 1.80%)
```

拟合点的最大误差在 2% 上下（这一次 1.80%，下面的复跑 2.40%）。这比 Delphi（< 0.5%）和闸门 1 的标准（< 1%）都松。几万参数、几十万字节的训练本身噪声就大。这是如实的结果。

**同一套实验，换一台机器，α 差了将近一倍。**2026-10，我们在另一台服务器上用同样的代码、同样的种子复跑。拟合结果是：

```
课程构建机：L(N, D) = 2.557 + 19.23/N^0.52 + 520.6/D^0.58   （最大误差 1.80%）
复跑服务器：L(N, D) = 2.565 + 1721/N^0.98  + 427.2/D^0.56   （最大误差 2.40%）
```

E 和 β 几乎没变，N 方向的指数 α 却从 0.52 跳到了 0.98。代码没有 bug：在同一台机器上重跑，输出逐字节相同。差别的唯一来源是浮点运算的顺序。`03` 里高学习率那几格的损失在第三、四位上不同。抛物线插值出的 η\* 跟着变（s1 从 0.0188 变成 0.0214）。阶梯按新的学习率训练，上表能对上的 4 个点又各自漂了 0.1%–2%。

这么小的扰动就能把 α 推开一倍。这说明这个指数**本来就没有被数据定住**。原因有三层：

- **点太少**：告诉我们"模型变大、损失降多快"的只有 4 个尺寸（跨度 20 倍），其中只有 2 个有过训练区的点。A、α 要从这几个点里定出来，还要和 E 分账。
- **模型太小、噪声太大**：单个种子、几万参数、几十万字节。每个点自带 1%–2% 的噪声（两台机器之间同一个点的漂移就有这么大）。而 N 从 1 万变到 22 万，损失只降 2%–8%（取决于预算）。水平差能看出来，"降得多快"的曲率却淹没在噪声里。Lourie et al.（2026）发现，小模型对超参很敏感，规律只在"调到头"的前沿上才出现。他们的阶梯从 4M 参数起步：每个尺寸只试 4 组或 16 组超参时，规律根本看不出来；要试 256 组才拟合得准。我们的模型比 4M 小约 20–400 倍，每个尺寸只扫了 5–6 个学习率。
- **参数之间能互相抵消（可辨识性差，poor identifiability）**：α 变大时，A 和 E 跟着变大。这时在拟合范围内画出的曲线几乎一样。用复跑的 14 个点试一下：把 α 固定在 0.1 到 1.5 之间的任意一个值，重新拟合其余参数。相对误差的均方根只在 1.23%–1.40% 之间变化：α = 0.52 时是 1.27%，最优的 α = 0.98 时是 1.23%。差别远小于噪声。下面的 bootstrap 也说明了这一点：按尺寸重采样 200 次，α 的 95% 区间是 0.02–1.50，盖满了整个搜索网格。

指数没定住时，离数据近的预测影响不大，离数据远的结论就全变了。同样用复跑的点，把 α 固定在 0.2 到 1.5 之间：

- 下面的留出检验只外推 2.1 倍，三个预算的预测误差都在 ±2.5% 以内。
- 外推到 N ≈ 470 万（阶梯最大尺寸的 21 倍，D = 262,144），预测的损失从 2.84 变到 2.98，差了 5%。
- "每个参数该配多少 token"由 β/(α+β) 决定，也跟着 α 变（本节最后一行输出：127 还是 68）。

这正是本章"外推与留出检验"要讲的道理，也是主线阶梯（第 7 节）这样设计的原因。主线阶梯用大得多的模型：`configs/ladder/` 的 23M–319M，都在 Lourie et al. 看到规律的 4M 以上，与主线用同一个词表。每个尺寸**先把学习率扫到最优值在网格内部，再拟合**。外推值给出 bootstrap 区间。并且**一定留一个没参与拟合的尺寸做检验**。

判断拟合靠不靠得住，要看留出误差和区间，不看指数"像不像 0.5"。Lourie et al. 也提醒：外推越远，抽样误差被放得越大，规律在数据附近才最可靠。

**关键一步是留出检验**（held-out test）。用 η\*(N) 的幂律外推出 s5 的学习率 0.00112（s5 的 N 是阶梯最大尺寸的 2.1 倍）。**不对它调参。**真的训练它，再和拟合的外推值对答案。不确定度用 bootstrap 估计：按尺寸有放回地抽样，重新拟合 200 次。（同一次训练分叉出的点彼此相关，要整组重采样。）

```
Extrapolate to the held-out size s5 (N = 467,936, 2.1× the largest ladder size):
       D |   pred.    95% interval |  actual   error
  65,536 |  3.4165  3.3912–3.4535  |  3.4465  -0.87%   (with the s4 learning rate 0.002284: 3.3770, error +1.17%)
 131,072 |  3.1393  3.1062–3.1829  |  3.1495  -0.32%   (with the s4 learning rate 0.002284: 3.1264, error +0.41%)
 262,144 |  2.9538  2.9123–3.0049  |  2.9243  +1.01%   (with the s4 learning rate 0.002284: 2.9209, error +1.13%)
```

三个预算的误差是 −0.87%、−0.32%、+1.01%，都落在 95% 区间内。复跑服务器上是 −1.02%、−0.15%、+1.51%，误差的量级一样。但 65,536 那一格的实际值 3.4623 比区间上界 3.4585 略高，**落在了区间外**。

两台机器都成立的结论是：外推 2.1 倍，误差在 ±1.5% 左右，和拟合点自身的误差是同一量级。bootstrap 区间只包含"换一组尺寸重新拟合"的不确定度。它不包含留出那次训练自身的噪声（种子、浮点）。所以区间偏窄，真值偶尔落在外面并不奇怪。写报告时，要把落在区间外的点如实写出来。

括号里是对照。如果 s5 直接沿用 s4 的学习率，实际损失在短预算上更低一些（两台机器都是如此）。这说明学习率的幂律外推在最短的预算上偏保守了。学习率规则本身也会出错，这正是 Delphi 第一次失败的原因（下一节）。

最后一行输出值得一看：

```
This fit says: at compute C = 3.43e+11, the optimum is N ≈ 21,194, D ≈ 2,693,780 (127.1 tokens/parameter).
```

复跑服务器上，这一行是 N ≈ 28,997、D ≈ 1,968,928（67.9 token/参数）。两次都远离 20，彼此却也差了近一倍。最优分配 N_opt ∝ C^(β/(α+β)) 由指数的比值决定。α 没定住，它也定不住。

能放心说的只有一点：在字节级小数据上，算力最优比例不是 20。**最优比例依赖数据和分词器。**（DeepSeek LLM 也发现，数据质量越高，算力越该往模型大小上分。）所以主线必须用自己的数据、自己的分词器跑阶梯，不能直接套用 Chinchilla 的数字。

## 6. 外推的实践：Delphi

大尺度上也是这样吗？Marin 团队的 **Delphi**（`references.md` 已收录）是一个很好的真实案例：

- **做法**：先用一个能穷举调参的参考小模型定出"配方"（recipe）。配方是一个函数，它把算力映射成完整的训练配置（宽度、深度、batch、学习率、β、权重衰减、初始化……）。然后在 3e18–3e20 FLOPs 的 7 个预算上做 IsoFLOP 扫描（同一算力下换不同的 N 与 D）。取每个预算的最优点，拟合 L(C) 幂律。
- **第一次失败**：小预算上拟合得很干净，但 1e22 的留出运行偏了 2.5%，1e23 直接发散。问题出在长训练上。学习率随 batch 按 √B 放大，却没有随训练长度下调，所以数据多的大运行学习率太大。权重衰减固定为 0.1，也不随尺度变。
- **修配方，而不是修曲线**：学习率乘上训练长度修正 (T₀/T)^0.3（来自 Apple 的 Complete(d)P）。优化器换成 AdamH，它把权重约束在固定范数的球面上（权重衰减从配方里消失）。同样 7 个预算重新拟合后，1e21、1e22、1e23 的留出误差是 +0.5%、+0.2%、+0.2%。这是外推到拟合范围的 300 倍之外。
- **不确定度**：对 7 个桶的最优点做 bootstrap 重采样，95% 区间从 1e21 的 ±0.5% 放宽到 1e23 的 ±4%。三个种子之间的差异只有约 0.1%。
- **下游分数**：准确率这种"硬"指标在小模型上贴着随机水平（四选一的 MMLU 就是 25%），看不出进步。Delphi 先对**软指标**（正确选项的对数概率、参考答案的 bits-per-byte）做 scaling law。再用一批公开模型拟合"软指标 → 硬分数"的 S 形映射。两者复合，就能预测大模型的 MMLU、HumanEval、GSM8K。（Llama 3 技术报告用的是同样的两步法。）

从 Delphi 和我们的迷你阶梯，可以总结出外推的四条纪律：**配方固定、先调参、留出检验、区间如实报告**。检验不通过时，修配方重跑，不在原拟合上打补丁。

## 7. 闸门 1：花大钱之前交给人决定

GOAL.md 3.4 的闸门 1，就是把上面的方法用在主线上。第二步有了 GPU 之后的阶梯协议，写在 [`runs/ladder/README.md`](../../runs/ladder/README.md)（**第一步不花 GPU 钱**）：

- **尺寸与预算**：`configs/ladder/` 的 4 个尺寸（总参数 23M、71M、160M、319M，与主线同一个词表）。每个尺寸有 20×、80×、320× token/参数三个预算（l300m 只跑 20×、80×）。运行用 WSD 分叉。l20m–l150m 用来拟合，**l300m 留出**。
- **先扫学习率**：每个尺寸 5 个学习率，最优值在边上就外扩。把 η\*(N) 外推给 l300m 和主线。必要时加上 Delphi 的训练长度修正。
- **估算**：一轮阶梯（含分叉）约 $340。加上学习率扫描和种子重复，约 $450–550（假设 MFU 0.3）。这在第 12–13 章 $1,200 的预算内。单次超过 $100 的运行要先报批。

闸门 1 报告（模板：[`runs/gate1_report_template.md`](../../runs/gate1_report_template.md)）必须回答三件事：

1. **损失外推**：主线在计划 token 数下的验证损失，以及 95% 区间。拟合残差、留出误差、外推倍数全部列出（`zero.tools.fit_scaling`）。
2. **基准分数外推**：用两步法，即软指标的 scaling law + 公开模型拟合的 S 形映射（`fit_loss_to_score`）。阶梯规模下接近随机的基准不外推，如实写"无信号"。只用自己的开发集，不碰预注册的测试基准。
3. **配方验证**：(a) 把后训练配方套在 2–3 个阶梯 Base 模型上，拟合"Base 损失 → 工具调用得分"，并外推到主线。(b) 把配方套在一个现成的同尺寸开源 Base 模型上，看配方本身的上限（只验证，不发布）。

**预测达不到硬目标，就不开始预训练。**先调整配方或目标，不硬花钱。

## 8. 两个"待核实"的训练技术：Muon 与 FP8

GOAL.md 2.1 把 Muon 优化器和 FP8 训练列为"待核实"。我们按共识规则核实。规则要求至少 3 个独立的头部开源模型家族，在主力版本的技术报告里明确采用。核实结果如下，出处见"采用方与来源"。

- **Muon：达到共识，进正文。**Kimi K2（MuonClip）、GLM-4.5、DeepSeek-V4 三家的技术报告都明确用 Muon 训练主力模型。另外还有 Moonlight（与 Kimi 同属 Moonshot）、Puro-2B（MuonH）、nanochat 等。
- **FP8 训练：达到共识。**采用方是 DeepSeek-V3（细粒度 FP8 混合精度）、Llama 4（官方博客：用 FP8 预训练 Behemoth）、NVIDIA Nemotron-H（56B 全程 FP8 预训练）。另外还有 Puro-2B（分块 FP8）。FP8 属于第 14 章"预训练工程"的内容，这里只讨论它对预算的影响。

**Muon 是什么**：对每个隐藏层的权重矩阵，Muon 先像 SGD 一样积累动量。再用 5 步 Newton–Schulz 迭代把更新矩阵"正交化"：G = UΣVᵀ 变成约 UVᵀ，所有奇异值拉到 1 附近。直觉是：AdamW 逐元素缩放，更新矩阵往往被少数几个方向主导。Muon 让所有方向的步长相同，梯度里容易被淹没的方向也能学到东西。

embedding、lm_head、RMSNorm 仍然用 AdamW（三家的分组方式一致）。更新乘上 0.2·√max(m, n) 后，RMS 和 AdamW 相当，于是可以沿用 AdamW 的学习率（Moonlight / Kimi K2 的做法；GLM-4.5 取 0.2，DeepSeek-V4 取 0.18）。

```python
def newton_schulz5(G, steps=5):
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G / (G.norm() + 1e-7)  # the iteration converges only when the largest singular value is ≤ 1
    ...
    for _ in range(steps):
        A = X @ X.T
        X = a * X + (b * A + c * A @ A) @ X
```

`06_muon.py` 在迷你阶梯的 s2、s3 上，和第 4 节**调好的** AdamW 比较，数据同样是 131K 字节。（Muon 也扫了学习率，最优值在网格内部。）

```
NS5 check: singular values of a random 64×32 gradient 0.059–0.290 → after orthogonalization 0.682–1.042

Same 131,072 bytes, validation loss (bit/byte):
size        N |     AdamW best |      Muon best | diff
  s2   31,968 | 3.1669 (η=0.01  ) | 2.9945 (η=0.02  ) | -0.1724
  s3   95,664 | 3.1940 (η=0.0025) | 2.9350 (η=0.01  ) | -0.2590
```

差距很大，但**这是几万参数、短训练上的极小配置演示**，不能直接推到 0.7B。Moonlight 报告在算力最优训练下约有 2 倍的算力效率。Puro-2B 报告 MuonH 相对调好的 Muon 还有 1.19 倍。两者都有提升，但远没有这里夸张。生产级实现在 `zero/train/muon.py`（CPU 测试通过，GPU 路径尚未验证）。

**对主线的建议**：在第二步的阶梯里，在 l60m、l150m 上做 Muon 与 AdamW 的正面对比。两边都调学习率，拟合"等效算力倍数"（Puro-2B 的做法）。倍数的区间明确大于 1.1 才采用 Muon，否则保持 AdamW。

**FP8 对预算的影响**：Puro-2B 在 1.7B 模型上测得 1.36 倍吞吐，扣除损失上的代价后净 1.34 倍。NVIDIA NeMo 在 H100 上对 Llama 3 8B 测得约 1.3 倍，模型越大提升越大。0.7B 的矩阵更小，提升可能更低。`zero` 目前没有实现 FP8，也没有在 GPU 上验证。**第 9 节的预算不指望它。**

## 9. 主线决策：尺寸、token 数、超参

`05_plan_budget.py` 把第 1 节的公式反过来用：给定 $5,000，能买多少 token？输出如下。这是节选：脚本还会打印更多候选行，以及 MFU 0.55 的一行。`24L` 是少 4 层（24 层，不是 28 层）。`narrow` 是窄一档（dim 1024、FFN 3072）。`days` 是 8 张卡上的天数。

```
Budget $5,000, 8×H100 (peak 989.5 TFLOPS, to be verified), $2.5/GPU-hour → 2,000 GPU-hours in total
candidate     params   seq  MFU |  token tok/param    days
main 0.69B     0.69B  4096  0.4 |   410B       594    10.4
main 0.69B     0.69B  4096  0.5 |   512B       743    10.4
main 0.69B     0.69B  2048  0.4 |   514B       745    10.4
24L 0.60B      0.60B  4096  0.4 |   472B       783    10.4
narrow 0.51B   0.51B  4096  0.4 |   486B       958    10.4

Compare: the original plan of 500B tokens (MFU 0.4, sequence 4096) costs $6,102, 22% over the $5K limit.
  MFU 0.40: $5K buys  410B tokens; 500B costs $6,102
  MFU 0.45: $5K buys  461B tokens; 500B costs $5,424
  MFU 0.50: $5K buys  512B tokens; 500B costs $4,881
```

**建议（交给第二步的闸门 1 定稿，不是结论）**：

1. **尺寸：保持 0.69B**（`configs/main/pretrain.toml`，总参数 689.5M ≤ 0.8B）。缩小到 0.5–0.6B 能多买 15–20% 的 token。但在第 2 节的曲线上，N 缩小 25% 的损失大于 D 增加 20% 的收益。更重要的是，硬目标要和 Qwen3.5-0.8B 比。模型比对手小得越多，越难赢。
2. **token 数：按 $5K 封顶，计划约 400B**（MFU 0.4 时约 $4.9K，580 token/参数）。预训练用 WSD 的稳定段（不衰减），任何时刻停下都能交给中期训练。所以不必事先赌 MFU。阶段 6 实测吞吐后，如果 $5K 能买到更多（MFU 0.5 时 512B），就训练到预算用完为止。
3. **可选的省钱办法**：预训练序列从 4096 降到 2048。每 token 的 FLOPs 少 20%，同样的钱多买 25% 的 token。（Qwen3、GLM-4.5、DeepSeek-V4 用 4K，SmolLM2 用 2K。长上下文反正要在第 15 章扩到 32K。）代价是主线的超参要在 2048 上重扫，阶梯也要用同一个长度。这个选项值得在阶梯里测，但不是默认。
4. **超参**：峰值学习率由阶梯的 η\*(N) 外推（配置里的 3e-4 只是起点）。β₁ 0.9、β₂ 0.95、权重衰减 0.1、梯度裁剪 1.0、batch 524K token 先沿用。优化器默认 AdamW，Muon 按第 8 节的对比决定。精度用 BF16，FP8 不计入预算。
5. **生产工具**：`uv run python -m zero.tools.plan_budget --config configs/main/pretrain.toml --mfu <实测> --fit runs/ladder/fit.json` 会把预算、token 数和阶梯外推的损失放进同一张表。

## 10. 小结

- **算账**：每 token 训练 FLOPs ≈ 6·N_matmul + 12·L·d_attn·T，总算力 C ≈ 6ND。主线在 4096 长度下，注意力占四成。
- **Chinchilla**：L(N, D) = E + A/N^α + B/D^β。只算训练时，最优约 20 token/参数。Kaplan 的不同结论主要来自小模型没调好和算力口径。
- **过训练**：把推理也算进来，小模型训练得越久越划算。代价是同算力下损失稍高（主线约 3.5%）。
- **先调参，再拟合**：小模型对超参很敏感。最优学习率随尺寸下降。偷懒会得出"做大没用"的错误结论。
- **外推要检验**：WSD 分叉省算力。阶梯要覆盖过训练区。bootstrap 给出区间。留出一个更大的模型对答案。检验不通过就修配方。极小阶梯拟合出的指数很脆弱（同一实验换台机器，α 从 0.52 变成 0.98）。要看留出误差和区间，不看指数本身。
- **主线**：0.69B × 约 400B token，$5K 封顶，WSD 稳定段随时可停。Muon、FP8 都已是共识，但要在阶梯里验证后再用到主线上。

---

## 从极简代码到生产级代码

| 极简代码（`code/`） | 生产级代码（`zero/`、`runs/`） | 多做了什么、为什么 |
|---|---|---|
| `01_flops.py`：手写公式 + 用 `FlopCounterMode` 实测 | `zero/model.py` 的 `estimate_flops_per_token`、`count_params` | 处理 GQA（K、V 头更少）、共享 embedding 时 lm_head 的乘法、YaRN 之后的配置。训练器用它实时报告 MFU。 |
| `05_plan_budget.py`：固定常数的算术 | `zero/tools/plan_budget.py`：`tokens_for_budget`、`plan`、`apply_candidate` | 直接复用 `estimate_cost`（同一张 GPU 峰值表、同一个口径）。能读任意配置，能改形状做候选，能读阶梯拟合来预测损失。`--speedup` 用来讨论"如果 FP8 提速"之类的问题（标注为未验证）。 |
| `04_mini_ladder.py` 的 `fit_lnd`：网格 + 无约束最小二乘，负系数直接丢弃 | `zero/tools/fit_scaling.py`：`fit_chinchilla`、`fit_power_law`、`bootstrap_chinchilla`、`fit_loss_to_score` | 3 变量**非负**最小二乘（枚举 7 种有效集，精确且向量化）。N、D 先归一化，防止病态。两轮网格细化。指数落在边界时报警。可选 α=β。按组 bootstrap。L(C) 幂律。损失 → 分数的 S 形映射（闸门 1 的第二步）。 |
| 手动把点写进 JSON | `fit_scaling --run 目录[:配置] --holdout ... --target-config ... --out fit.json` | 直接读训练器的 `log.jsonl`（取最后一次验证损失与 token 数）。用 checkpoint 里的配置算 N。N 的口径可选非 embedding 参数、总参数或 FLOPs。 |
| `03`、`04` 里手写的 WSD 分叉 | `zero/train/schedule.py` 的 WSD + 从 checkpoint 续训（`runs/ladder/README.md` 第 4 节的命令） | 分叉就是"复制分叉点的 checkpoint，用更小的 `max_steps` 续训"。数据加载器的状态一起恢复，所以分叉看到的是主干接下来的数据。 |
| `06_muon.py` 的极简 `Muon` | `zero/train/muon.py`：`zeropower_via_newtonschulz5`、`MuonAdamW`、`split_params_for_muon`、`build_muon_optimizer` | 一个优化器对象同时管理 Muon 组和 AdamW 组（调度按组写学习率，checkpoint 只有一个 `state_dict`）。两种尺度规则（rms / spectral）。CUDA 上用 BF16 做 NS5。遇到 FSDP 切片的参数直接报错。（CUDA 上的 BF16 NS5，以及 DDP 下各卡结果一致，已在 RTX 3090 上验证，见 [runs/2026-10-01-gpu0-check](../../runs/2026-10-01-gpu0-check/README.md) 第 11、14.1 节。） |

**对拍与测试**：

- `tests/test_fit_scaling.py`：用已知系数（Epoch 复现的量级）生成阶梯数据。拟合必须恢复 E、α、β（误差 < 0.02），且外推误差 < 0.3%。加噪声后，bootstrap 区间要覆盖真值。α=β、L(C) 幂律、S 形映射各自恢复已知参数。CLI 能读训练器格式的日志，并做留出检验。`plan_budget` 算出的 token 数代回 `estimate_cost`，正好等于预算（$5K ↔ 410B，与 RUNBOOK 一致）。
- `tests/test_muon.py`：NS5 的输出与 SVD 的精确正交因子方向一致（余弦 > 0.95），奇异值落在 [0.5, 1.3]。rms 尺度下，单步更新的 RMS ≈ lr × 0.2。参数分组与三家技术报告一致（块内矩阵走 Muon，embedding / norm 走 AdamW）。在小 Transformer 上损失下降。从 `state_dict` 续训与不中断训练逐位一致。

**接入训练器（待主流程合并）**：本章不改动 `zero/train/trainer.py` 与 `zero/config.py`。启用 Muon 需要两处小改。第一，在 `OptimConfig` 里加字段 `name: str = "adamw"`。第二，在 `build_optimizer` 开头加一行 `if cfg.name == "muon": return build_muon_optimizer(model, cfg, device)`。之后用 `--set optim.name=muon` 就能在阶梯里做对比。

## 主线进度

### 极小配置演示：zero 流水线从训练日志到拟合

> 以下是**极小配置演示**：只说明代码能跑通，不代表主线模型的任何结果。

用 `configs/tiny/pretrain.toml`（tiny 语料、2048 词表）跑一个 4 尺寸 × 2 预算的小阶梯。**故意沿用配置里的学习率 3e-3，不做扫描**：

```bash
for spec in "32 1 150" "32 1 300" "64 2 150" "64 2 300" "96 3 150" "96 3 300" "128 4 150" "128 4 300"; do
  set -- $spec
  uv run python -m zero.train.pretrain --config configs/tiny/pretrain.toml \
    --set model.dim=$1 --set model.n_heads=$2 --set model.n_kv_heads=$2 --set model.ffn_dim=$(( $1 * 3 )) \
    --set model.n_layers=2 --set train.max_steps=$3 --set train.eval_every=$3 --set train.eval_batches=16 \
    --set checkpoint.every=0 --set schedule.warmup_steps=20 --set train.out_dir=out/ch12/zero_ladder/d$1_s$3
done
Z=out/ch12/zero_ladder
uv run python -m zero.tools.fit_scaling --run $Z/d32_s150 --run $Z/d32_s300 --run $Z/d64_s150 --run $Z/d64_s300 \
  --run $Z/d96_s150 --run $Z/d96_s300 --holdout $Z/d128_s150 --holdout $Z/d128_s300 --tie-exponents --bootstrap 100
```

输出（验证损失的单位是 nat/token）：

```
Fitted 6 points (N definition: non_embedding)
  L(N, D) = 4.7664 + 2.725e+07/N^2.0000 + 6.361e+10/D^2.0000
  Fit residual RMSE 0.1088, max relative error 3.76%
  ⚠ An exponent is at the edge of the search range: the data does not fix the law (common causes: learning rate not tuned, too narrow a range of sizes or budgets). Do not trust the extrapolation.
run                                 N            D   actual      fit    error
d32_s300                    2.691e+04    6.144e+05   4.9626   4.9725   +0.20%
d64_s300                    1.069e+05    6.144e+05   4.7790   4.9373   +3.31%
d96_s300                    2.402e+05    6.144e+05   5.1282   4.9354   -3.76%
...
Held-out test (not used in the fit):
  d128_s150: actual 5.5341, extrapolated 5.4406 (95% interval 5.2520–5.5005), error -1.69%
  d128_s300: actual 5.1999, extrapolated 4.9351 (95% interval 4.6818–5.0706), error -5.09%
```

流水线跑通了：训练器日志 → 读配置算 N → 拟合 → bootstrap → 留出检验 → JSON。但**拟合本身失败了**，而且这次失败很有教育意义。没调学习率，d96 比 d64 还差（5.13 vs 4.78）。指数撞到搜索边界，工具发出警告。留出误差 −5.09%，落在区间外。这正是第 4 节的结论在生产流水线上的重演：闸门 1 的阶梯必须先扫学习率（`runs/ladder/README.md` 第 3 节）。

### 待 GPU 训练后补充

- 阶梯实验（`runs/ladder/README.md`）：学习率扫描表、L(N, D) 拟合、l300m 留出检验、主线外推损失与区间。
- 基准分数外推（软指标 + S 形映射），以及配方验证 (a)(b)。
- Muon vs AdamW 的阶梯对比与"等效算力倍数"。
- 实测 MFU / token/s，以及最终的 token 数和费用。
- **闸门 1 报告**（`runs/gate1_report_template.md`）与批准记录。花费记入 `runs/ledger.md`。

## 前沿观察

- **μP 与各种超参迁移理论**（μP、Complete(d)P、Llama 4 的 MetaP）：目标是把小模型上调好的学习率和初始化"原样"迁移到大模型。这样大模型就不用再调参。Llama 4 的博客说，MetaP 能让超参跨 batch、宽度、深度和训练长度迁移。Delphi 用了 Complete(d)P 的训练长度修正，MiniCPM 用了 μP。但各家的具体规则彼此不同，细节多数没有公开，所以还不算共识。本课用"阶梯上扫描 + 幂律外推"这个共识做法（GOAL.md 2.1 把 μP 细节列为不讲）。
- **Hyperball 类优化器**（AdamH、MuonH）：把权重矩阵约束在初始化时的范数球面上。这样权重衰减从超参里消失，学习率就是"有效学习率"。Delphi（AdamH）和 Puro-2B（MuonH，报告比调好的 Muon 省 16% 算力）在用，采用方还太少。
- **课程式数据排序 + checkpoint 平均**（Puro-2B 的 CMA）：数据按质量从低到高排，后期用恒定学习率，再平均最后几个 checkpoint。报告说，这比均匀数据 + 衰减省约 2.4 倍成本。这是单一来源的结果。第 15 章也会提到退火与平均。
- **按配方的"成本 scaling law"**（Puro Cost Scaling Law）：固定 2B 模型，把 15 个基准的平均分拟合成成本的对数函数 P = a + b·log₂(C − C_P1)。它用来回答"多少钱能到 Qwen2-1.5B 的水平"（约 $4.4K）。它是固定尺寸、单一配方、往成本更低方向的曲线。作者明确说它不能推广到其他尺寸。这个思路可以借鉴到闸门 1 的"钱 → 分数"报告里。

## 采用方与来源

| 技术 | 采用方（头部开源模型家族） | 来源 |
|---|---|---|
| 用小实验拟合 scaling law 来定尺寸、token 数与超参 | DeepSeek | DeepSeek LLM §3（学习率与 batch 的幂律、IsoFLOP、非 embedding FLOPs/token 口径）：<https://arxiv.org/abs/2401.02954> |
| | Qwen3 | 技术报告 §3.2"we develop scaling laws for optimal hyper-parameters (e.g., learning rate scheduler, and batch size) predictions"：<https://arxiv.org/abs/2505.09388> |
| | Llama | Llama 3 技术报告 §3.2.1（用 scaling law 定旗舰尺寸、用两步法预测下游）：<https://arxiv.org/abs/2407.21783>；Llama 4 博客（MetaP）：<https://ai.meta.com/blog/llama-4-multimodal-intelligence/> |
| | Kimi | Kimi K2 技术报告 §2（架构"derived from empirical scaling law analysis"）：<https://arxiv.org/abs/2507.20534> |
| | Marin（开放配方） | Delphi：<https://openathena.ai/blog/delphi/> |
| 小模型远超 20 token/参数（过训练） | Qwen3 | 0.6B–235B 都用约 36T token：<https://arxiv.org/abs/2505.09388> |
| | Llama 3 | 8B 用 15T token（Delphi 估算约为算力最优的 90 倍）：<https://arxiv.org/abs/2407.21783> |
| | MobileLLM（Meta） | MobileLLM-R1-950M 用 4.2T token：<https://arxiv.org/abs/2509.24945> |
| | Puro-2B（开放配方） | 约 1.4T token、约 700 token/参数（附录 A）：<https://www.alphaxiv.org/abs/2608.27370> |
| Muon 优化器 | Kimi | Kimi K2 §2.1 MuonClip（Muon + 权重衰减 + RMS 匹配 0.2 + QK-Clip）：<https://arxiv.org/abs/2507.20534>；Moonlight：<https://arxiv.org/abs/2502.16982> |
| | GLM | GLM-4.5 §2.4"We employed the Muon optimizer for all parameters except word embedding, bias, and weights for RMSNorm"（更新 RMS 0.2）：<https://arxiv.org/abs/2508.06471> |
| | DeepSeek | DeepSeek-V4 §2.4 与 §4.2.2（嵌入、预测头、RMSNorm 用 AdamW，其余用 Muon；更新 RMS 0.18；混合 Newton–Schulz）：<https://arxiv.org/abs/2606.19348> |
| | 其他 | Puro-2B（MuonH）：<https://www.alphaxiv.org/abs/2608.27370>；nanochat：<https://github.com/karpathy/nanochat>；参考实现：<https://kellerjordan.github.io/posts/muon/> |
| FP8 训练 | DeepSeek | DeepSeek-V3 §3.3（细粒度 FP8 混合精度训练）：<https://arxiv.org/abs/2412.19437> |
| | Llama | Llama 4 博客"we focus on efficient model training by using FP8 precision… pre-training our Llama 4 Behemoth model using FP8"：<https://ai.meta.com/blog/llama-4-multimodal-intelligence/> |
| | NVIDIA Nemotron | Nemotron-H"We pre-trained Nemotron-H-56B-Base on 20 trillion tokens in FP8"：<https://research.nvidia.com/labs/adlr/nemotronh/> |
| | 其他 | Puro-2B（分块 FP8，1.7B 上净 1.34 倍）：<https://www.alphaxiv.org/abs/2608.27370> |
| WSD 学习率调度（阶梯分叉用） | Kimi | Kimi K2 §2.5"the WSD learning rate schedule"：<https://arxiv.org/abs/2507.20534> |
| | DeepSeek | DeepSeek-V4 §4.2.2（学习率大部分时间保持 2.7e-4，最后衰减）：<https://arxiv.org/abs/2606.19348> |
| | MiniCPM（提出者） | <https://arxiv.org/abs/2404.06395>；反例：GLM-4.5 §2.4 报告 WSD 在其设置下更差，改用 cosine |

待核实：SmolLM2 预训练序列长度 2048（第 9 节第 3 条引用，未逐字核对模型卡）。H100 稠密 BF16 峰值 989.5 TFLOPS 与 $2.5/卡时（卡时：GPU-hour；`estimate_cost.py` 标注为待核实）。

## 引导问题

1. 主线在序列长度 4096 时，注意力占每 token FLOPs 的四成。FlashAttention 会跳过因果掩码下被遮住的一半。这时实测的 MFU（按 `zero` 的口径）会偏高还是偏低？预算应该用什么来算？
2. Chinchilla 原文系数给出约 78 token/参数，Epoch 复现给出约 20。"优化器提前停止"只是一个小问题，为什么能让结论差 4 倍？（提示：看 α、β 的比值怎样决定 N_opt ∝ C^(β/(α+β))。）
3. 迷你阶梯第一次只有 12 个点时，拟合出 α ≈ 0.05。为什么所有点都在"数据不够"的区域时，N 方向的指数定不下来？主线的阶梯协议里哪一条是为此设计的？
4. bootstrap 为什么按尺寸整组重采样，而不是把 14 个点各自独立重采样？如果独立重采样，区间会变宽还是变窄？
5. Muon 在迷你阶梯上领先 0.17–0.26 bit/字节，但本章建议"阶梯对比后再定"。列出至少三个理由，说明为什么极小配置上的优势不能直接推到 0.7B。
6. 如果闸门 1 的外推显示工具调用的硬目标达不到，你会先改配方里的哪一项？为什么不是"多花一点钱，多训练一些 token"？

## 动手任务

**任务 1（基础）**：修改 `05_plan_budget.py`。把单价换成你能找到的真实 H100 租价，把 MFU 换成 0.35，重新算 $5K 能买多少 token。再算 Puro-2B 的配置（2B、1.4T token）在 8×H100、MFU 0.4 下要多少钱。和论文里 RTX 5090 集群的 $6.9K 比较，差在哪里？

**任务 2（核心）**：在 `03_lr_sweep.py` 里把 `SWEEP_TOKENS` 改成 262,144，重新扫描（缓存文件换个名字）。看最优学习率怎么变。Delphi 的配方说，学习率应该随训练长度按 (T₀/T)^0.3 下降。你的数据支持这个结论吗？

**任务 3（挑战）**：把 `04_mini_ladder.py` 的留出尺寸换成 dim 128、4 层（N ≈ 80 万，约为阶梯最大尺寸的 3.7 倍）。重新训练，并检验外推。外推倍数变大后，误差和 bootstrap 区间各怎么变？再试试只用 s1、s2、s3 拟合，把 s4 和 s5 都当作留出，画出"外推倍数 – 误差"的关系。

## 想深入：CS336

本章对应 [CS336](https://cs336.stanford.edu/)（Spring 2026）的**第 9、11 讲"Scaling Law"**。这两讲把本章的每一步讲得更深。内容包括：从 Kaplan 到 Chinchilla 的方法之争、IsoFLOP 与参数化拟合、超参怎样随尺度迁移，以及各家技术报告怎样用 scaling law 做决策。**作业 3（Scaling）**要求在有限的算力预算里自己设计实验、拟合 scaling law 并做预测（具体要求以课程页当期说明为准）。本章的迷你阶梯就是它的缩小版，做完本章再做它会顺利很多。课程页有每讲的讲义和录像。

## 本章参考文献

- Kaplan et al. *Scaling Laws for Neural Language Models*，2020：<https://arxiv.org/abs/2001.08361>
- Hoffmann et al. *Training Compute-Optimal Large Language Models*（Chinchilla），2022：<https://arxiv.org/abs/2203.15556>
- Besiroglu et al. *Chinchilla Scaling: A replication attempt*（Epoch AI），2024：<https://arxiv.org/abs/2404.10102>
- Porian et al. *Resolving Discrepancies in Compute-Optimal Scaling of Language Models*，2024：<https://arxiv.org/abs/2406.19146>
- Sardana & Frankle. *Beyond Chinchilla-Optimal: Accounting for Inference in Language Model Scaling Laws*，2023：<https://arxiv.org/abs/2401.00448>
- Lourie, Cho, Ullrich, Lotfi. *Small-Scale Experiments: Are We There Yet?*，2026：<https://arxiv.org/abs/2608.11859>（`references.md` 已收录）
- Held (Marin / Open Athena). *Scaling Laws That Extrapolate 300× Past the Fit*（Delphi），2026：<https://openathena.ai/blog/delphi/>（`references.md` 已收录）
- Luo et al. *PuRo-2B: Poor Lab's Qwen2-1.5B Trained on RTX 5090 within $5090*，2026：<https://www.alphaxiv.org/abs/2608.27370>（`references.md` 已收录）
- DeepSeek-AI. *DeepSeek LLM: Scaling Open-Source Language Models with Longtermism*，2024：<https://arxiv.org/abs/2401.02954>
- Qwen Team. *Qwen3 Technical Report*，2025：<https://arxiv.org/abs/2505.09388>
- Llama Team. *The Llama 3 Herd of Models*，2024：<https://arxiv.org/abs/2407.21783>
- Meta. *The Llama 4 herd*（博客），2025：<https://ai.meta.com/blog/llama-4-multimodal-intelligence/>
- Zhao et al. *MobileLLM-R1*，2025：<https://arxiv.org/abs/2509.24945>
- Jordan et al. *Muon: An optimizer for hidden layers in neural networks*，2024：<https://kellerjordan.github.io/posts/muon/>
- Liu et al. *Muon is Scalable for LLM Training*（Moonlight），2025：<https://arxiv.org/abs/2502.16982>
- Kimi Team. *Kimi K2: Open Agentic Intelligence*，2025：<https://arxiv.org/abs/2507.20534>
- GLM-4.5 Team. *GLM-4.5: Agentic, Reasoning, and Coding (ARC) Foundation Models*，2025：<https://arxiv.org/abs/2508.06471>
- DeepSeek-AI. *DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence*，2026：<https://arxiv.org/abs/2606.19348>
- DeepSeek-AI. *DeepSeek-V3 Technical Report*，2024：<https://arxiv.org/abs/2412.19437>
- NVIDIA. *Nemotron-H: A Family of Accurate, Efficient Hybrid Mamba-Transformer Models*，2025：<https://research.nvidia.com/labs/adlr/nemotronh/>
- NVIDIA. *Faster Training Throughput in FP8 Precision with NVIDIA NeMo*（博客），2025：<https://developer.nvidia.com/blog/faster-training-throughput-in-fp8-precision-with-nvidia-nemo/>
- Hu et al. *MiniCPM*（WSD），2024：<https://arxiv.org/abs/2404.06395>
- [CS336](https://cs336.stanford.edu/) 第 9、11 讲，作业 3

**下一章**：阶梯实验固定了"配方"，而配方里分量最重的一项是数据。第 13 章，我们收集、清洗、去重、配比主线的预训练数据，并训练主线的分词器。词表大小也会反过来改变本章的参数账。
