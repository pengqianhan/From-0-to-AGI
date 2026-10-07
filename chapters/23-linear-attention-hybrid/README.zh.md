# 第 23 章：线性注意力与混合架构 —— 把 KV cache 压缩成一个固定大小的矩阵

[English](README.md) · **中文**

> **目标**：读完这一章，你能从 softmax 注意力出发，推导出线性注意力的递推形式 `S_t = S_{t−1} + v_t k_tᵀ`。你能写出 delta 规则和 Gated DeltaNet 的更新公式，并验证"分块形式 == 递推形式"。Qwen3.5-0.8B 是"3 层线性 + 1 层全注意力"的混合结构。你还能用它的真实配置，算出这种混合结构省下多少 KV cache。最后，你能讲清楚这种模型为什么还要保留那几层全注意力。

📺 **本章视频**：还没有发布。要在本机渲染，运行 `bash chapters/23-linear-attention-hybrid/video/build.sh`。
🧪 **本章自检**：学完后，在 Claude Code 里输入 `/ch23-linear-attention`。

---

上一章我们用滑动窗口限制注意力能看的长度。一部分层只看最近的几千个 token，KV cache 就不再随上下文无限增长。但这些层确实看不到窗口外面的内容。这一章要解决另一个问题：**能不能让一层注意力"看见"全部历史，却只用固定大小的内存？**

答案是**线性注意力**（linear attention）。去掉 softmax，注意力就变成一个 RNN（循环神经网络）：一个 d×d 的状态矩阵压缩保存全部历史。压缩一定有代价：状态会"满"，精确回忆会变差。于是有了两个改进：衰减门和 delta 规则，两者合起来是 Gated DeltaNet。今天工业界的主流答案是混合架构：**少量全注意力 + 大量线性层**。千问 Qwen3.5 连 0.8B 的小模型都用了 3:1 的混合。

本章代码：

```bash
uv run python chapters/23-linear-attention-hybrid/code/01_linear_attention.py   # associativity, KV vs state, one decode step (a few seconds)
uv run python chapters/23-linear-attention-hybrid/code/02_chunked.py            # decay gate + chunkwise parallel form (10 to 20 seconds)
uv run python chapters/23-linear-attention-hybrid/code/03_delta_rule.py         # overwrite, capacity, chunkwise form of Gated DeltaNet (a few seconds)
uv run python chapters/23-linear-attention-hybrid/code/04_hybrid_lm.py          # small language models with four layouts (slow on the first run, then reads the cache)
uv run python chapters/23-linear-attention-hybrid/code/05_associative_recall.py # associative recall: pure linear vs hybrid (slow on the first run)
```

> **注意：**本章所有计时都在一台多个任务共享的 CPU 上用单线程测得。同一个脚本多跑几次，计时可能相差好几倍。表里的计时只用来看**数量级和趋势**。数值结果（误差、损失、准确率）固定了随机种子，可以复现。

## 1. 问题：softmax 注意力必须记住一切

回忆第 8 章的因果注意力。生成第 t 个 token 时，它这样计算：

```
o_t = Σ_{j≤t} softmax_j( q_t·k_j / √d ) · v_j
```

softmax 要对**这一行的全部 t 个分数**一起做归一化。只要缺一个 k_j，就算不出分母。所以模型必须保留前面每个位置的 K 和 V。这就是第 10、21 章的 KV cache，它随上下文长度线性增长。`01_linear_attention.py` 的第 2 部分算了单层单头（d = 64，BF16）的内存：

| 上下文长度 T | KV cache | 线性注意力的状态 |
|---:|---:|---:|
| 128 | 32 KB | 8 KB |
| 1,024 | 256 KB | 8 KB |
| 8,192 | 2,048 KB | 8 KB |
| 65,536 | 16,384 KB | 8 KB |
| 262,144 | 65,536 KB | 8 KB |

右边一列就是这一章的目标：**不管上下文多长，大小都不变。**

## 2. 去掉 softmax：矩阵乘法换个顺序

假设注意力分数不经过 softmax，而是直接用 `φ(q)·φ(k)`。这里 φ 是一个特征映射（feature map），它把向量的元素变成非负数；Katharopoulos 等人 2020 年用的是 `elu(x) + 1`。这样，整段序列的输出就是三个矩阵连乘，再逐元素乘一个因果掩码 M：

```
O = ( φ(Q) φ(K)ᵀ ⊙ M ) V
```

矩阵乘法满足**结合律**（associativity）。不带掩码时，`(φ(Q) φ(K)ᵀ) V = φ(Q) (φ(K)ᵀ V)`。左边要先造一个 T×T 的大矩阵。右边只要一个 d×d 的小矩阵，它和 T 无关。因果掩码的意思是：第 t 个位置只累加位置不超过 t 的项。所以可以边走边算：

```
S_t = S_{t−1} + v_t φ(k_t)ᵀ      （写入：状态 S 是 d_v × d_k 的矩阵）
o_t = S_t φ(q_t)                 （读出）
```

这就是一个**循环神经网络**（recurrent neural network，RNN）。每来一个 token，就把 v 和 k 的外积加进状态，再用 q 去读。每一步的内存和计算量都是常数。代码（`01_linear_attention.py`）：

```python
def linear_attention_recurrent(q, k, v):
    S = torch.zeros(v.shape[1], q.shape[1])
    out = []
    for t in range(q.shape[0]):
        S = S + torch.outer(v[t], phi(k[t]))  # S_t = S_{t-1} + v_t φ(k_t)ᵀ   (write)
        out.append(S @ phi(q[t]))             # o_t = S_t φ(q_t)                (read)
    return torch.stack(out)
```

把它和并行形式对比，并行形式一次算出整个 T×T 矩阵。T = 256、d = 16 时，两种算法输出的**最大相对误差是 4.7e-07**，只是 float32 的舍入误差。softmax 为什么不行？它的分母把一整行绑在一起，拆不开。所以 softmax 注意力没有这种递推形式。

> **注意：**原论文的线性注意力还要除以一个归一化项 `φ(q_t)ᵀ Σ_j φ(k_j)`。后来的工作（GLA、DeltaNet、Qwen3.5 等）大多去掉了这个分母。它们改为对 q、k 做 L2 归一化，并在输出上加一个 RMSNorm，数值更稳定。本章从第 3 节起都用这种现代写法（φ 取恒等映射）。

**推理时快了多少？** `01` 的第 3 部分测了生成一步时的注意力部分：已有 T 个历史 token，再生成下一个 token（单头，d = 64）：

| 已有上下文 | softmax 注意力 | 线性注意力 |
|---:|---:|---:|
| 1,024 | 0.033 ms | 0.030 ms |
| 8,192 | 0.250 ms | 0.030 ms |
| 65,536 | 12.5 ms | 0.030 ms |
| 262,144 | 44.2 ms | 0.030 ms |

softmax 注意力每一步都要读全部 T 个 K、V，所以耗时随 T 增长。线性注意力每一步只用一个 64×64 的状态，所以耗时是常数。（计时波动很大：另一次运行里，262,144 那一格是 19–25 ms。）

## 3. 训练：分块并行形式

推理可以一个 token 一个 token 地走，训练却要一次处理整段序列。逐 token 的 for 循环无法并行，GPU 会闲着。完全并行的形式 `(QKᵀ ⊙ M) V` 又回到了 T×T 的矩阵。折中办法是**分块**（chunkwise）形式。把序列切成长度为 C 的块：

- **块内**：用并行形式，做一次矩阵乘法。
- **块间**：把状态 S 传下去，S 压缩了前面所有的块。块内每个位置再额外读一次 S。

```python
for s in range(0, T, C):
    qc, kc, vc = q[s:s+C], k[s:s+C], v[s:s+C]
    b = g[s:s+C].cumsum(0)                                   # cumulative log decay in the chunk (next section)
    D = (b[:, None] - b[None, :]).masked_fill(~mask, -inf).exp()
    inter = (qc * b.exp()[:, None]) @ S.T                    # read the state from the chunk start
    intra = ((qc @ kc.T) * D) @ vc                           # parallel attention in the chunk
    out[s:s+C] = inter + intra
    S = b[-1].exp() * S + vc.T @ (kc * (b[-1] - b).exp()[:, None])   # compress the full chunk into the state
```

`02_chunked.py` 验证三种算法的数值一致。T = 512 时，递推 vs 并行的最大差是 1.8e-06，递推 vs 分块是 6.0e-07。然后它比较速度（单头，d = 64，CPU 单线程，毫秒）：

| T | 递推（逐 token） | 分块，C = 64 | 完全并行 |
|---:|---:|---:|---:|
| 256 | 33.4 | 4.1 | 3.9 |
| 1,024 | 76.2 | 12.7 | 79.6 |
| 4,096 | 344.1 | 47.2 | 2015.0 |

序列短时，完全并行最快（一次大矩阵乘法）。序列长时，T² 的代价超过其他所有代价。这时分块形式比逐 token 快约 7 倍，比完全并行快 40 多倍。GPU 上的差距更大，因为分块把工作变成了矩阵乘法，这是 GPU 最擅长的运算。**训练和 prefill 用分块形式，decode 用递推形式。** 两者在数学上完全等价。所有线性注意力模型都用这种标准做法。

## 4. 衰减门：学会遗忘

朴素线性注意力有一个明显的问题：它**只会累加**。一万个 token 之前写进去的内容，和最新的内容权重一样。状态里的内容越堆越多。第一个改进是加一个**衰减门**（decay gate）：

```
S_t = α_t · S_{t−1} + v_t k_tᵀ,     α_t ∈ (0, 1]
```

每次写入之前，先把整个旧状态乘上 α_t。`02_chunked.py` 只在第 0 步写入一次，然后看写入的值随距离怎样衰减：

| α | 距离 10 | 距离 100 | 距离 500 |
|---|---:|---:|---:|
| 1.0 | 1.000 | 1.000 | 1.000 |
| 0.99 | 0.904 | 0.366 | 0.007 |
| 0.9 | 0.349 | 0.000 | 0.000 |

这些值正好是 α^距离。更重要的是，在真实模型里，**模型由当前输入算出 α_t**（数据相关，data-dependent）。话题变了时，模型可以把门关小，清掉旧记忆。需要长期记住时，模型可以把门开到接近 1。在分块形式里，衰减只多了上面代码中的 `D` 矩阵和几个 `exp(b)` 因子。这里 b 是对数衰减的累加和。指数里永远是"后减前"，所以不会溢出。

一大批模型都用这个骨架："带衰减的线性递推"。它们的区别只在 α 怎样参数化。RetNet 每个头用一个固定常数。GLA（Gated Linear Attention）用数据相关的向量门。Mamba-2 每个头用一个数据相关的标量门。（Mamba-2 论文把这类状态空间模型和线性注意力统一进同一个框架，叫"结构化状态空间对偶"。）

本章不逐一展开这些模型，只讲它们的共同骨架。在这一族里，主流模型真正采用的是两支：Mamba-2 和 Gated DeltaNet 一族（见下文）。

## 5. 代价：固定大小的记忆会满

这种压缩有代价。把整段历史压进一个固定大小的矩阵，记忆就会"满"。把状态 S 看成一张"键 → 值"的表：写入 (k, v)，之后用同一个 k 去读，希望 `S k ≈ v`。如果 key 两两正交，一个 key 维度为 d_k 的状态最多存 d_k 个互不干扰的键值对。再多，不同的 key 就会互相串扰。

`03_delta_rule.py` 第 2 部分往一个 64×64 的状态里写入 N 个随机的单位向量 key 和随机 value，再逐个读回。它报告相对误差 `‖S k − v‖ / ‖v‖`。0 表示完美读回，1 ≈ 读出的噪声和信号一样大：

| 写入 N 个 | 线性注意力 | delta 规则（下一节） | softmax 注意力（存全部 KV） |
|---:|---:|---:|---:|
| 16 | 0.469 | 0.309 | 0.000 |
| 32 | 0.692 | 0.479 | 0.000 |
| 64 | 0.996 | 0.737 | 0.000 |
| 128 | 1.431 | 0.972 | 0.000 |
| 256 | 1.974 | 1.206 | 0.000 |

softmax 注意力把 K、V 原样存下，用 key 去"查表"。它的误差始终是 0，代价是内存随 N 增长。线性注意力的内存固定，代价是**精确回忆**（exact recall）：写得越多，读回来越不准。这是线性注意力最大的弱点。Zoology（Arora 等 2023）用"多查询联想回忆"（MQAR）任务系统地测过它。第 8.3 节我们也做一个小版本。

## 6. delta 规则：覆盖，而不是累加

线性注意力的写入是盲目的。它不管状态里已经存了什么，直接把 `v kᵀ` 加上去。同一个 key 写两次，读出来是两个值的和。**delta 规则**（delta rule）换了一种写法。（它来自 Widrow–Hoff 的经典学习规则。Schlag 等人 2021 年把它用到线性注意力上，称为 DeltaNet。）它先用 k 读出旧答案 `S k`，再只把"新值与旧答案的差"写回去：

```
S_t = S_{t−1} + β_t (v_t − S_{t−1} k_t) k_tᵀ
    = S_{t−1} (I − β_t k_t k_tᵀ) + β_t v_t k_tᵀ
```

β_t ∈ (0, 1) 是写入强度（模型也由输入算出它）。当 β = 1、‖k‖ = 1 时，写完以后 `S_t k_t = v_t` 恰好成立：新值干净地**覆盖**了旧值。这次更新就是对 S k 与 v 之间的平方误差做了一步梯度下降，学习率是 β。所以人们也把它看作一种"测试时学习"（test-time learning）。`03_delta_rule.py` 第 1 部分用同一个 key 先写 v1 = [1, 0]，再写 v2 = [0, 1]：

| | 用 k 读出 |
|---|---|
| 线性注意力 | [1.0, 1.0]（v1 + v2） |
| delta 规则 | [0.0, 1.0]（v2） |

看上一节的容量表。在每个 N 上，delta 规则的误差都比朴素累加低（N = 64 时 0.737 vs 0.996）。它在写入前先"擦掉" key 方向上的旧内容，所以串扰小了很多。

## 7. Gated DeltaNet：衰减 × delta 规则

两个改进各管一件事。衰减门让旧信息整体淡出（适合"换话题"）。delta 规则精确地改写某个 key 上的内容（适合"更新一条记忆"）。Gated DeltaNet（Yang, Kautz, Hatamizadeh 2024）把两者乘在一起：

```
S_t = α_t · S_{t−1} (I − β_t k_t k_tᵀ) + β_t v_t k_tᵀ
o_t = S_t q_t
```

`03_delta_rule.py` 第 3 部分做了一个流式实验：连续写入 1024 个键值对，只读最近写入的 32 个：

| 写入规则 | 最近 32 个的读回误差 |
|---|---:|
| 线性注意力 | 4.049 |
| 线性 + 固定衰减 α = 0.95 | 0.644 |
| delta 规则 | 0.584 |
| delta + 固定衰减 α = 0.95 | 0.652 |
| 线性 + 换话题时 α = 0 | 0.675 |
| delta + 换话题时 α = 0（数据相关的门） | 0.502 |

我们看到四点。1000 个旧条目淹没了朴素累加（4.05）。只用衰减或只用 delta 规则，都能把误差拉回 0.6 左右。**固定的**衰减叠在 delta 规则上反而略差，因为它把最近的条目也一起衰减了。最好的结果（0.502）来自 delta 规则加上**数据相关**的门：门在"换话题"那一步关到 0。

所以门的价值在于"由输入决定什么时候忘"。这也是 Gated DeltaNet 必须由当前 token 算出 α_t 的原因。不过 0.5 离 softmax 注意力的 0 还很远：固定大小的状态总有容量上限。

**分块形式。** delta 规则比朴素线性注意力多一个难点。块内第 i 个位置往状态里写入一个"修正值" u_i = β_i (v_i − S k_i)。这个值依赖前面位置写入的 u_j，所以不能直接用一次矩阵乘法算出来。把块内的依赖写成矩阵，就得到一个下三角线性方程组：

```
(I + A) U = β ⊙ (V − e^{b} ⊙ K S₀ᵀ),     A_ij = β_i e^{b_i − b_j} (k_i · k_j)   (j < i)
```

用一次三角求解解出 U（Yang 等人 2024 年的"UT 变换"/WY 表示）。剩下的部分和带衰减的线性注意力完全同形，只是把 v 换成 u：

```python
A = (bt[..., :, None] * (kc @ kc.mT) * D).tril(-1)            # A_ij = β_i e^{b_i−b_j} k_i·k_j, j<i
rhs = bt[..., None] * (vc - (kc * b.exp()[..., None]) @ S.mT)  # β_i (v_i − e^{b_i} S₀ k_i)
u = torch.linalg.solve_triangular(eye + A, rhs, upper=False)
out.append((qc * b.exp()[..., None]) @ S.mT + ((qc @ kc.mT) * D) @ u)
S = b[..., -1, None, None].exp() * S + u.mT @ (kc * (b[..., -1:] - b).exp()[..., None])
```

`03_delta_rule.py` 第 4 部分做了验证。测试取 B = 2、H = 3、T = 128、块长 32。分块与递推的最大输出差是 4.8e-07。最终状态差是 3.6e-07。

**真实模型里的 Gated DeltaNet 层**还多几个部分。这里以 Qwen3.5 为准，见 `zero/arch/linear_attention.py`。这些部分是：

- q、k、v 投影之后，先过一个核长为 4 的**因果短卷积**（depthwise 卷积）。每个位置先混合左边相邻几个 token 的信息。这对回忆任务帮助很大。
- 对 q、k 做 **L2 归一化**。它保证 ‖k‖ = 1，delta 规则才稳定。
- α_t 用 Mamba-2 的参数化：`α_t = exp(−e^{A} · softplus(a_t + dt_bias))`。另外，β_t = sigmoid(b_t)。
- 输出过一个**门控 RMSNorm**：RMSNorm(o) ⊙ SiLU(z)。
- 多头。value 头数可以是 q/k 头数的整数倍（Qwen3-Next 有 16 个 q/k 头、32 个 v 头）。

线性层**不加 RoPE**。递推本身就有先后顺序，短卷积也提供局部的位置信息。

## 8. 混合：少量全注意力 + 大量线性层

线性层的记忆有限，全注意力又太贵。工业界的答案是**混合**（hybrid）。大部分层用线性注意力（或状态空间层）。每隔几层放一层全注意力，专门负责精确回忆。

### 8.1 Qwen3.5-0.8B 的真实结构

下面的内容读自 Hugging Face 上 `Qwen/Qwen3.5-0.8B` 的 `config.json`（2026-09）：

```
"full_attention_interval": 4,
"layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention", ... ×6]
"num_key_value_heads": 2, "head_dim": 256,                       # 全注意力层（gated attention）
"linear_num_key_heads": 16, "linear_num_value_heads": 16,
"linear_key_head_dim": 128, "linear_value_head_dim": 128,        # Gated DeltaNet 层
"linear_conv_kernel_dim": 4, "mamba_ssm_dtype": "float32"
```

模型卡写得更直接：`Hidden Layout: 6 × (3 × (Gated DeltaNet → FFN) → 1 × (Gated Attention → FFN))`。`04_hybrid_lm.py` 的最后一部分按这份配置算了一条序列的推理缓存。KV 用 BF16，递推状态按 config 用 FP32：

| 上下文 | KV cache（6 层全注意力） | 线性状态（18 层 Gated DeltaNet） | 合计 | 假如 24 层全是全注意力 |
|---:|---:|---:|---:|---:|
| 4,096 | 48 MiB | 18.6 MiB | 67 MiB | 192 MiB |
| 32,768 | 384 MiB | 18.6 MiB | 403 MiB | 1,536 MiB |
| 262,144 | 3,072 MiB | 18.6 MiB | 3,091 MiB | 12,288 MiB |

只有那 6 层全注意力有 KV cache。另外 18 层 Gated DeltaNet 的状态加起来是 18.6 MiB，与上下文长度无关。上下文越长，节省的比例越接近 4 倍（3:1 的理论上限）。注意，这里的"24 层全注意力"是一个假想的对照：把线性层换成同样配置的全注意力层。它不是某个真实模型。

### 8.2 小实验一：同一个小语言模型，四种结构

`04_hybrid_lm.py` 用第 10 章同款的字符级莎士比亚语料，训练四个 4 层小模型（宽度 128、4 个头、SwiGLU FFN、同样的数据顺序和超参数，各 800 步）。只换每层的 **token mixer**。（token mixer 是层里负责在 token 之间混合信息的部分。）A = softmax 注意力（带 RoPE），L = 朴素线性注意力，G = Gated DeltaNet。L 和 G 都带短卷积和输出归一化。

> **注意：**本章训练类实验的数字来自课程构建机上的一次 CPU 运行。不同的机器、不同版本的底层数学库，浮点运算的顺序略有不同。训练几百步以后，这些微小差异会变大。你本机的数字可能从小数点后第二、三位开始就不一样。请以下文不依赖具体数值的结论为准。2026-10 在另一台服务器上的复跑对照，见 [runs/2026-10-01-gpu0-check/chapters-21-23.md](../../runs/2026-10-01-gpu0-check/chapters-21-23.zh.md)。

| 结构 | 参数量 | 验证集损失（nats/字符） | 另一台服务器复跑（2026-10） | 推理缓存 T=1,024 | T=65,536 |
|---|---:|---:|---:|---:|---:|
| AAAA（纯注意力） | 861,440 | 1.685 | 1.683 | 2,048 KB | 131,072 KB |
| LLLL（纯朴素线性） | 867,712 | 1.754 | 1.760 | 73 KB | 73 KB |
| GGGG（纯 Gated DeltaNet） | 871,872 | 1.661 | 1.648 | 73 KB | 73 KB |
| GGGA（3:1 混合） | 869,264 | 1.649 | 1.652 | 567 KB | 32,823 KB |

（缓存按"KV 用 BF16、线性状态用 FP32"计算，见 `cache_bytes`。）

- 朴素线性注意力最差（1.754）。只会累加的状态，在字符级建模里也吃亏。
- 在这个规模上，Gated DeltaNet 甚至比纯注意力略好（1.661 vs 1.685）。这并不说明它"比注意力强"。800 步、0.87M 参数、128 个字符的上下文，模型主要在学局部拼写。短卷积 + 门控递推恰好擅长这种局部模式。
- 3:1 混合和纯 Gated DeltaNet 几乎一样低（1.649 vs 1.661）。在另一台服务器上复跑是 1.652 vs 1.648，名次反了过来。这个差别在单一种子的随机波动之内。和纯注意力相比，混合的好处是缓存只有约 1/4（T 越长越接近 1/4）。它相对纯线性模型的真正优势是精确回忆，而语言建模的损失看不出这一点。见下一个实验。

**如实说明**：我们只用了一个随机种子。四者的差距最大 0.1 nats，前三名之间只差 0.04。第 10 章观察到的种子波动也在这个量级。所以这里不能据此排出可靠的名次。它只能说明一点：在同样的参数量下，把大部分层换成线性层，**没有让语言建模明显变差**，而缓存省了大半。线性层真正的短板，要用专门的任务才看得出来：这就是下一个实验。

### 8.3 小实验二：联想回忆——纯线性落后，混合追回来

语言模型的损失对"精确回忆"不太敏感：大部分字符靠局部上下文就能猜出来。`05_associative_recall.py` 专门测回忆。序列前半段是 N 个随机的"键 值"对。后半段反复给出其中某个键，模型要答出它对应的值。这是 Zoology 的 MQAR 任务的缩小版。

我们用四种 2 层小模型（宽度 64，4 个头）。线性层的 head_dim 故意取得很小（16），让状态容量明显不够用。为了公平，注意力层的 q/k/v 也带同样的短卷积。每种模型训练 600 步，每步随机取 N ∈ [4, 24]。每个 N 测 256 条序列。随机猜的准确率是 1/64：

| 结构 | N = 4 | N = 8 | N = 12 | N = 16 | N = 20 | N = 24 | 推理时每层要存的数 |
|---|---:|---:|---:|---:|---:|---:|---|
| AA（纯注意力） | 100% | 99% | 97% | 97% | 96% | 94% | KV 8192 + KV 8192 |
| LL（纯朴素线性） | 64% | 40% | 30% | 24% | 19% | 17% | 状态 1024 + 状态 1024 |
| GG（纯 Gated DeltaNet） | 85% | 62% | 48% | 39% | 32% | 29% | 状态 1024 + 状态 1024 |
| GA（1 层 GDN + 1 层全注意力） | 99% | 96% | 91% | 86% | 83% | 79% | 状态 1024 + KV 8192 |

我们得到三个结论，它们和第 5–7 节的容量实验一致：

- **纯线性模型的准确率随 N 单调下降。** 键值对越多，固定大小的状态越装不下。朴素线性注意力在 N = 24 时只剩 17%。
- **Gated DeltaNet 比朴素线性强得多**（每个 N 上都高 12 到 22 个百分点）。覆盖式写入和门控确实让有限的状态用得更好。但它仍然远不如注意力。
- **只要把两层中的一层换成全注意力，回忆能力就回来了大半**（N = 24 时从 29% 升到 79%）。这个混合模型的缓存只比纯注意力的一半多一点。

**如实说明**：这是一个极小规模的实验：2 层、600 步、单一随机种子，而且我们故意把线性层的状态设得很小。纯注意力在 N 大时的优势，会随训练步数和模型大小变化。训练更久时，混合模型和纯注意力之间的差距（79% vs 94%）可能缩小，也可能不变。这里没有做这个实验。真实模型的对比，请看 Zoology、Gated DeltaNet 和 Kimi Linear 论文里的回忆类评测。

### 8.4 为什么是"少量全注意力 + 大量线性层"

把上面的现象合起来：

- **线性层负责"便宜的深度"。** 它们处理局部模式、语法和逐步累积的语义。它们的内存和每步计算都与长度无关。
- **少量全注意力层负责"精确回忆"。** 它们从很远的上下文里原样找回某个名字、数字或代码变量。固定大小的状态恰恰最不擅长这件事。
- **KV cache 只随全注意力层数增长。** 3:1 时约为纯注意力的 1/4，9:1 时约为 1/10。

比例没有标准答案。Qwen3.5、Kimi Linear、Ling-3.0 用 3:1。IBM Granite 4.0-H 用 9:1。NVIDIA Nemotron 3 Nano 在 52 层里只放了 6 层注意力。Falcon-H1 在同一层里把注意力头和 Mamba-2 头并联。每个模型都在"KV cache 能省多少"和"回忆能力保留多少"之间选了自己的点。

也有明确的反例。MiniMax-Text-01（2025 年初）用了 lightning attention（一种线性注意力）与 softmax 注意力 7:1 的混合。它的 config 里，80 层的 `attn_type_list` 每 8 层有 1 层是 softmax。但它的下一代 MiniMax-M2 回到了**每一层都是全注意力**（MiniMax-M2.5 的 `attn_type_list` 全是 1）。MiniMax 官方博客给出了这些原因：

- 在代码、数学、智能体、长链推理和强化学习这些复杂任务上，线性/稀疏注意力的效果还不够稳定。
- 线性注意力的训练和推理基础设施还不成熟，很多实现受限于显存带宽。
- 线性注意力对数值精度更敏感，用低精度存储状态有困难。
- 它和推测解码怎样配合，还是一个没有解决的问题。

所以，混合架构是多家公司已经采用的**工程权衡**，不是没有代价的选择。

## 9. 小结

- softmax 注意力要对一整行分数做归一化，只能保留全部 K、V：KV cache ∝ T。
- 去掉 softmax，换个顺序做矩阵乘法，注意力就变成 RNN：`S_t = S_{t−1} + v_t k_tᵀ`，`o_t = S_t q_t`。状态大小固定。
- 训练用**分块形式**（块内并行，块间传递状态），decode 用**递推形式**。两者在数学上等价。
- 固定大小的状态会满：精确回忆是线性注意力最大的弱点。
- **衰减门** α_t 让旧信息淡出（RetNet / GLA / Mamba-2 一族）。**delta 规则**先读后写、只写差值，实现"覆盖"。两者相乘就是 **Gated DeltaNet**。
- **混合架构**：大量线性层 + 少量全注意力层（Qwen3.5 是 3:1）。KV cache 只随全注意力层增长，全注意力层保证精确回忆。

---

## GPU 实测（单张 RTX 3090）

> **注意：**上面正文里的数字都来自 CPU 运行。本节换到一张 NVIDIA GeForce RTX 3090 上实测：24 GB 显存，Ampere 架构。规格表：BF16 张量核稠密峰值约 71 TFLOPS，FP32 约 35.6 TFLOPS，显存带宽约 936 GB/s。环境：PyTorch 2.11.0+cu128、CUDA 12.8，2026 年 10 月。服务器把这张卡的功耗上限设成了 240 W（出厂默认 350 W）。持续满载时，这张卡会降频。所以算力和带宽的绝对值比满功耗的 3090 低，看相对关系更可靠。没有 GPU 可以跳过本节。

运行：

```bash
uv run python chapters/23-linear-attention-hybrid/code/06_gpu_linear_vs_softmax.py
```

形状取 Qwen3.5-0.8B 的 Gated DeltaNet 层：batch 1、16 个头、d_k = d_v = 128。softmax 注意力也用 16 × 128。线性注意力这边原样调用本章代码：04 的 `linear_chunked`，以及 03 的 `gated_delta_chunked` 和 `gated_delta_recurrent`。脚本只是把调用放进 `with torch.device("cuda")` 里，让函数内部新建的张量也落在 GPU 上。线性注意力是纯 PyTorch、FP32、块长 64，用 Python 逐块循环。softmax 注意力用 PyTorch 自带的 FlashAttention kernel（BF16）。先确认 GPU 上数学没变。T = 1,024 时，Gated DeltaNet 的分块形式与递推形式的最大输出差是 6.0e-07。最终状态差是 4.8e-07。

整段处理 T 个 token（训练 / prefill；毫秒，取中位数；括号里是输入之外额外占用的峰值显存）：

| 序列长 T | softmax（FlashAttention） | 线性注意力·分块 | Gated DeltaNet·分块 | Gated DeltaNet·递推 |
|---:|---:|---:|---:|---:|
| 1,024 | 0.2（4 MiB） | 2.9（17 MiB） | 10.7（19 MiB） | 198.8 |
| 4,096 | 1.3（16 MiB） | 11.2（65 MiB） | 43.1（67 MiB） | 813.5 |
| 16,384 | 21.9（65 MiB） | 48.8（257 MiB） | 193.5（259 MiB） | — |
| 65,536 | 460.1（260 MiB） | 180.7（1,025 MiB） | 757.5（1,027 MiB） | — |

（逐 token 递推太慢，只测了前两行。）

decode 一步：已有 T 个 token 的上下文，再来 1 个。Gated DeltaNet 这边先用分块形式把 T 个 token 真的压成状态，再从这个状态递推一步（毫秒，50 次取中位数）：

| 上下文 T | softmax 的 KV cache | softmax 一步 | Gated DeltaNet 的状态 | Gated DeltaNet 一步 |
|---:|---:|---:|---:|---:|
| 1,024 | 8 MiB | 0.053 | (1, 16, 128, 128)，1 MiB | 0.259 |
| 16,384 | 128 MiB | 0.217 | (1, 16, 128, 128)，1 MiB | 0.245 |
| 65,536 | 512 MiB | 0.705 | (1, 16, 128, 128)，1 MiB | 0.242 |
| 262,144 | 2,048 MiB | 3.408 | (1, 16, 128, 128)，1 MiB | 0.306 |

第二张表是第 1、2 节那两张表在 GPU 上的样子。softmax 注意力的 KV cache 跟着上下文涨到 2 GiB。每一步都要把它整块读一遍，耗时从 0.053 ms 涨到 3.4 ms。Gated DeltaNet 的状态从头到尾是同一个 1 MiB 的 (1, 16, 128, 128) 张量。一步耗时 0.24–0.31 ms，和上下文多长无关。1.6 万个 token 时，两者还差不多（0.217 对 0.245 ms）。6.5 万个 token 时，softmax 这一步已经是 Gated DeltaNet 的 2.9 倍，26 万个 token 时是 11 倍。

第一张表印证了第 3 节的结论"训练和 prefill 用分块形式"。T = 4,096 时，逐 token 递推要 814 ms，分块只要 43 ms，快了约 19 倍。这比 CPU 上的差距（第 3 节的 7 倍）更大。分块形式的耗时随 T 线性增长，FlashAttention 按 T² 增长（16K → 64K 涨了 21 倍）。到 T = 65,536 时，朴素线性注意力的分块形式（181 ms）已经比 FlashAttention（460 ms）快。

出乎意料的是短序列上的差距这么大。T = 1,024 时，本章的 Gated DeltaNet 分块形式比 FlashAttention 慢几十倍（10.7 对 0.2 ms）。decode 一步的耗时也是 1K 上下文时 softmax 的 5 倍。时间几乎全花在 Python 循环的固定开销上：Gated DeltaNet 分块形式每块约 0.7 ms，朴素线性每块约 0.18 ms，不随 T 变。每块要启动十几个小 kernel，Gated DeltaNet 还要做一次三角求解。GPU 本身没有忙起来。这就是生产代码要用 flash-linear-attention 的 Triton kernel 的原因：它把整个循环融合成一个 kernel。（本机没有装 fla，所以没有对比。）

## 从极简代码到生产级代码

生产级代码在 `zero/arch/linear_attention.py`。它是第五部分的实验模块，**不用于主线模型**。它的结构和参数名与 Hugging Face transformers 的 Qwen3.5 实现一致（`transformers/models/qwen3_5/modeling_qwen3_5.py` 的 `Qwen3_5GatedDeltaNet`）。所以两者之间可以直接搬权重。

| 极简代码（`code/`） | 生产级代码（`zero/arch/linear_attention.py`） | 多做了什么、为什么 |
|---|---|---|
| `01` 的 `linear_attention_recurrent`、`02` 的 `recurrent` | `recurrent_linear_attention(q, k, v, g, initial_state)` | 批量、多头 `(B, H, T, D)`。可以接着上一次的状态继续算（decode 需要）。内部用 float32 |
| `02` 的 `chunked`（要求 T 是 C 的整数倍） | `chunk_linear_attention(..., chunk_size, initial_state)` | 自动补零（补的位置 k = 0、g = 0，对状态没有影响）。带初始状态，支持分段 prefill |
| `03` 的 `gated_delta_recurrent` / `gated_delta_chunked` | `recurrent_gated_delta_rule` / `chunk_gated_delta_rule` | 同上。状态布局改成 HF / fla 的 `(d_k, d_v)`（极简代码是 `(d_v, d_k)`，只差一个转置） |
| `04` 的 `LinearMixer`（q/k/v 投影、短卷积、L2 归一化、每头 RMSNorm） | `LinearAttention`、`GatedDeltaNet`（共用 `_LinearMixer`） | 参数名与 HF 相同（`in_proj_qkv`、`in_proj_z`、`in_proj_b`、`in_proj_a`、`conv1d`、`A_log`、`dt_bias`、`norm`、`out_proj`）。**门控 RMSNorm**（输出乘 SiLU(z)）。value 头数可以是 q/k 头数的整数倍。Mamba-2 式的 A、dt 初始化。短卷积带缓存（保存最后 K−1 个输入），分段输入和一次输入的结果相同。`mode="auto"` 时，T = 1 走递推形式，其他长度走分块形式 |
| `04` 的 `TinyLM(pattern="GGGA")` | `HybridConfig`、`HybridTransformer`、`hybrid_layer_types(n_layers, full_attention_interval=4)` | 由配置驱动：`layer_types` 与 HF 的 Qwen3.5 config 同名同义。全注意力层直接复用主线的 `zero.model.Attention`（GQA + QK-Norm + RoPE + SDPA） |
| 无 | `HybridCache`：全注意力层用预分配的 `KVCache`（只为全注意力层分配），线性层用 `LinearState`（递推状态 + 卷积尾巴）；`generate_greedy` | 推理时两种缓存并存。`cache_bytes_per_sequence` 按层记账（第 21 章账本的延伸） |
| `04` 的 `cache_bytes`、`qwen35_cache_mib` | `cache_bytes_per_sequence`；以及第 21 章的 `zero/tools/kv_cache_calc.py`（`layout_from_config` 直接读 HF config 里的 `layer_types` 或 Kimi 的 `linear_attn_config`，线性层按固定状态记账） | 读真实 config，算任意混合模型的缓存，不用手抄层数 |
| 无 | 未实现：Qwen3.5 全注意力层的**输出门**（gated attention，`attn_output_gate: true`）、部分 RoPE（`partial_rotary_factor: 0.25`）、MoE、MTP | 本章只关心 token mixer 的混合方式。这些内容在第 24、25 章和第 26 章的全景里讲 |

**对拍**（parity check）：`tests/test_arch_linear_attention.py`（`uv run pytest tests/test_arch_linear_attention.py`，本机 33 项全部通过，约 10 秒）。这些测试保证以下几点：

- 线性注意力（带/不带衰减）和 gated delta rule（带/不带衰减）：**分块形式 == 递推形式**。测试覆盖块长 1、5、8、16、64，长度不能被块长整除的情况，以及随机初始状态。
- 不衰减的线性注意力 == 掩码并行形式 `(QKᵀ ⊙ M) V`。
- gated delta rule == 朴素参考实现，它每一步都用显式矩阵 `S_t = α_t (I − β_t k_t k_tᵀ) S_{t−1} + β_t k_t v_tᵀ`。同一个 key 写两次时，delta 规则覆盖，线性注意力累加。
- `LinearAttention` / `GatedDeltaNet` 层：分块、递推、带状态分三段输入（10 + 1 + 12 个 token），三者输出一致。状态形状与长度无关。
- **与 HF transformers 的 `Qwen3_5GatedDeltaNet` 对拍**：随机化 HF 层的全部权重，再用 `load_state_dict(strict=True)` 搬进来。输出在 1e-4 内一致。
- `HybridTransformer`（间隔 2、4、99，即 1:1、3:1、纯线性；Gated DeltaNet 与朴素线性两种）：**用状态缓存生成 == 每步全量重算**（贪心生成 30 个 token，完全相同）。分块 prefill（20 + 1 + 24）== 一次性前向传播。长度翻倍时 KV 字节数翻倍，线性状态不变。训练几步后损失下降（梯度能穿过三角求解）。

**真正训练和部署时**，纯 PyTorch 的分块循环太慢。行业的做法是用这些工具：

- **flash-linear-attention（fla-org）**：用 Triton 写的线性注意力 kernel 库。`fla.ops.gated_delta_rule.chunk_gated_delta_rule` / `fused_recurrent_gated_delta_rule` 就是本章两种形式的 GPU 实现。Kimi Linear 的 KDA kernel（`fla.ops.kda`）也开源在里面。装了 fla 和 causal-conv1d 时，HF transformers 的 Qwen3.5 实现会自动换用这些 kernel。否则，它退回到和本章同构的纯 PyTorch 版本。
- **vLLM**：`vllm/model_executor/models/qwen3_next.py`、`qwen3_5.py` 等文件支持这些混合模型。它的混合 KV cache 管理器（见 Hybrid KV Cache Manager 设计文档）为不同类型的层分配不同的缓存。全注意力层按 token 数分配 KV 页。Mamba / 线性层按请求分配固定大小的状态。

我们在 RTX 3090 上验证了 `zero/arch/linear_attention.py` 在 CUDA 上的前向传播、反向传播和生成。验证时顺带修了一个 bug：`generate_greedy` 把输入建在了 CPU 上。在 BF16 下，Gated DeltaNet 的梯度与 FP32 的相对差约 20%。所以纯 PyTorch 分块实现在低精度下不够准。见 [runs/2026-10-01-gpu0-check](../../runs/2026-10-01-gpu0-check/README.zh.md) 第 11、12 节。fla、causal-conv1d 这些 CUDA kernel 没有安装，所以没有验证。

---

## 前沿观察

> **还不算共识的技术，只在这里提一句**
>
> - **纯线性 / 纯状态空间模型**（纯 Mamba、RWKV、RetNet 等）：它们证明了线性递推可以单独撑起一个语言模型。但头部开源模型家族的主力版本没有采用纯线性结构，全部是混合结构。（GOAL.md 2.1 把它们列为"只在前沿观察里提一句"。）
> - **KDA（Kimi Delta Attention）**：Kimi Linear 提出的 Gated DeltaNet 改进版。它把每个头一个标量的衰减门，换成逐通道（每个 key 维度一个）的细粒度门。蚂蚁的 Ling-3.0 也采用了它。目前只有这两家采用，而且 Ling 明确说继承了 Kimi Linear 的设计思路。所以它还不算多家独立的共识。
> - **并联混合**（Falcon-H1）：同一层里注意力头和 Mamba-2 头并行运行，这一层把两者的输出拼接起来。层的类型不交替。目前只有 Falcon-H1 这样做。
> - **比例之争**：头部模型有的用 3:1，有的用 9:1，甚至有的用"全注意力"（MiniMax-M2）。最佳比例是多少、全注意力层放在哪几层，都还没有共识。
> - **稀疏注意力**（DeepSeek DSA/NSA 一类；MiniMax-M3 也用了块稀疏注意力）是另一条"让长上下文变便宜"的路线，见第 22 章。

## 采用方与来源

| 技术 | 采用方（主力版本） | 来源 |
|---|---|---|
| 混合线性注意力：DeltaNet 一族线性层 + 少量全注意力 | **Qwen3-Next-80B-A3B**（48 层 = 12 × [3 × Gated DeltaNet + 1 × Gated Attention]）；**Qwen3.5 全系列**（0.8B、2B、4B、9B、27B、35B-A3B、122B-A10B、397B-A17B 的 config 均为 `full_attention_interval: 4`）；**Kimi Linear 48B-A3B**（KDA : MLA = 3:1，27 层中 7 层 MLA）；**蚂蚁 Ling-3.0-tiny**（3 层 KDA + 1 层 MLA 为一组） | 模型卡与 config.json：[Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B)、[Qwen3.5-27B config](https://huggingface.co/Qwen/Qwen3.5-27B/blob/main/config.json)、[Qwen3.5-397B-A17B config](https://huggingface.co/Qwen/Qwen3.5-397B-A17B/blob/main/config.json)、[Qwen3-Next-80B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct)、[Kimi-Linear-48B-A3B-Instruct](https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct)（技术报告 arXiv:2510.26692）、[Ling-3.0-tiny](https://huggingface.co/inclusionAI/Ling-3.0-tiny) |
| 混合状态空间：Mamba-2 层 + 少量注意力层 | **NVIDIA Nemotron-H**、**Nemotron 3**（Nano 30B-A3B：52 层里 23 层 Mamba-2、23 层 MoE、6 层注意力；Ultra 550B-A55B 同为 `nemotron_h` 结构）；**IBM Granite 4.0-H**（H-Small：40 层里 4 层注意力，官方称 Mamba-2 : Transformer = 9:1）；**TII Falcon-H1**（每层注意力与 Mamba-2 并联） | [Nemotron-H 技术报告 arXiv:2504.03624](https://arxiv.org/abs/2504.03624)、[Nemotron 3 Nano 技术报告 arXiv:2512.20848](https://arxiv.org/abs/2512.20848) 与 [模型卡](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16)（`hybrid_override_pattern`）；[granite-4.0-h-small config](https://huggingface.co/ibm-granite/granite-4.0-h-small/blob/main/config.json) 与 [IBM 发布公告](https://www.ibm.com/new/announcements/ibm-granite-4-0-hyper-efficient-high-performance-hybrid-models)；[Falcon-H1 技术报告 arXiv:2507.22448](https://arxiv.org/abs/2507.22448) |
| 线性注意力（lightning attention）+ softmax 7:1（后放弃） | **MiniMax-Text-01 / M1**（80 层，每 8 层 1 层 softmax）；**MiniMax-M2 / M2.5 回到全注意力** | [MiniMax-01 技术报告 arXiv:2501.08313](https://arxiv.org/abs/2501.08313)、[MiniMax-Text-01 config](https://huggingface.co/MiniMaxAI/MiniMax-Text-01/blob/main/config.json)、[MiniMax-M2.5 config](https://huggingface.co/MiniMaxAI/MiniMax-M2.5/blob/main/config.json)、官方博客 [Why Did M2 End Up as a Full Attention Model?](https://www.minimax.io/news/why-did-m2-end-up-as-a-full-attention-model) |
| 分块形式 + 递推形式的 kernel | flash-linear-attention（HF transformers 的 Qwen3.5 / Qwen3-Next 实现调用它，Kimi 的 KDA kernel 也开源在里面）；vLLM 的混合缓存管理 | <https://github.com/fla-org/flash-linear-attention>；vLLM [Hybrid KV Cache Manager](https://github.com/vllm-project/vllm/blob/main/docs/design/hybrid_kv_cache_manager.md) |

我们按 GOAL.md 2.1 的规则 A 计数。清单内有三个头部家族在主力版本中明确采用了 **"少量全注意力 + 大量线性/状态空间层"的混合结构**：Qwen、Kimi、NVIDIA Nemotron。另外 IBM Granite、Falcon、蚂蚁 Ling 也采用了。这满足共识条件，所以混合结构进正文。**Gated DeltaNet 本身**只有 Qwen 直接采用（Kimi 的 KDA 和 Ling 用的是它的改进版）。所以本章把它作为"DeltaNet 一族线性层"的代表来讲，把 KDA 放在前沿观察。

说明：各模型的层数、比例都读自 Hugging Face 上的 config.json 和模型卡（2026-09 读取）。写作时没有找到 Qwen3.5 技术报告的独立 arXiv 版本。所以本章以官方博客和模型卡为准（**待核实**：是否有正式的技术报告）。

---

## 引导问题

带着这些问题去问 Claude Code，直到你能用自己的话讲清楚：

1. 线性注意力的递推形式里，状态 S 的形状是 d_v × d_k。假如把多头注意力的 H 个头都换成线性注意力，整层的状态有多大？和同样 H 个头、长度为 T 的 KV cache 相比，T 为多少时两者相等？
2. delta 规则为什么要求对 k 做 L2 归一化？如果 ‖k‖ = 2、β = 1，写完以后 S k 等于什么？会不会发散？（提示：看 I − β k kᵀ 的特征值。）
3. 本章说 delta 规则是"对 ‖S k − v‖² 做了一步梯度下降"。自己求一下这个损失对 S 的梯度，验证它和 delta 规则的更新一致。在这个视角下，"衰减门"相当于什么？（提示：权重衰减。）
4. 训练用分块形式，推理用递推形式。如果训练时块长取 1，分块形式退化成什么？如果块长取整个序列长度呢？块长应该怎样选？
5. Qwen3.5 的线性层不加 RoPE，全注意力层只对 head_dim 的 25% 加 RoPE（`partial_rotary_factor: 0.25`）。线性层的位置信息从哪里来？
6. MiniMax-M2 回到了全注意力。假如你要为一个 0.6B、主要做工具调用的小模型（本课的主线模型）选择架构，你会用 3:1 混合还是全注意力？列出你的理由，以及需要做的对比实验。

## 动手任务

每个任务都要运行代码，看到结果。

**任务 1（基础）**：在 `03_delta_rule.py` 的容量实验里，把 key 换成**两两正交**的向量（例如 `torch.linalg.qr` 得到的正交矩阵的行），N 取 16、32、64、65。线性注意力和 delta 规则的读回误差分别是多少？为什么 N = 64 和 N = 65 之间会有突变？

**任务 2（核心）**：在 `05_associative_recall.py` 里加一种结构"GAGG"（全注意力放在第 2 层，而不是最后一层），和"GGGA"比较回忆准确率。再把线性层的 head_dim 翻倍（状态变成 4 倍大）。纯 Gated DeltaNet 的准确率能追上多少？

**任务 3（挑战）**：用 `zero/arch/linear_attention.py` 的 `HybridTransformer` 搭一个和 `04_hybrid_lm.py` 同尺寸的 3:1 混合模型。在同样的语料上训练 800 步，对比验证集损失。然后用 `model.new_cache()` 做带缓存的生成。生成 2000 个字符时，测量 `HybridCache.nbytes()` 里 KV 和线性状态各占多少，并和 `cache_bytes_per_sequence` 的公式核对。

---

## 想深入：CS336

本章对应斯坦福 CS336（Spring 2026）<https://cs336.stanford.edu/>：

- **第 4 讲：注意力的替代方案与 MoE。** 讲义与录像见课程页。这一讲讨论了为什么要找 softmax 注意力的替代品，也讲了线性注意力、状态空间模型这些方向的基本思路。本章的推导（结合律 → 递推 → 分块）和这一讲的线性注意力部分对应。
- **CS336 没有深入讲的内容**：delta 规则、Gated DeltaNet 的分块算法（UT 变换），以及工业混合架构（Qwen3.5 / Kimi Linear / Nemotron）的具体配置。见本章参考文献，尤其是 Songlin Yang 等人的 DeltaNet 与 Gated DeltaNet 论文，以及 flash-linear-attention 仓库。

---

## 本章参考文献

- Katharopoulos, Vyas, Pappas, Fleuret. *Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention*，2020：<https://arxiv.org/abs/2006.16236>
- Schlag, Irie, Schmidhuber. *Linear Transformers Are Secretly Fast Weight Programmers*（delta 规则用于线性注意力），2021：<https://arxiv.org/abs/2102.11174>
- Yang, Wang, Zhang, Shen, Kim. *Parallelizing Linear Transformers with the Delta Rule over Sequence Length*（DeltaNet 的分块并行算法），2024：<https://arxiv.org/abs/2406.06484>
- Yang, Kautz, Hatamizadeh. *Gated Delta Networks: Improving Mamba2 with Delta Rule*，2024：<https://arxiv.org/abs/2412.06464>
- Yang, Wang, Shen, Panda, Kim. *Gated Linear Attention Transformers with Hardware-Efficient Training*（GLA），2023：<https://arxiv.org/abs/2312.06635>
- Dao, Gu. *Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality*（Mamba-2），2024：<https://arxiv.org/abs/2405.21060>
- Sun et al. *Retentive Network: A Successor to Transformer for Large Language Models*（RetNet），2023：<https://arxiv.org/abs/2307.08621>
- Arora et al. *Zoology: Measuring and Improving Recall in Efficient Language Models*（MQAR 联想回忆任务），2023：<https://arxiv.org/abs/2312.04927>
- Qiu et al. *Gated Attention for Large Language Models: Non-linearity, Sparsity, and Attention-Sink-Free*（Qwen 的 gated attention），2025：<https://arxiv.org/abs/2505.06708>
- Kimi Team. *Kimi Linear: An Expressive, Efficient Attention Architecture*，2025：<https://arxiv.org/abs/2510.26692>
- MiniMax. *MiniMax-01: Scaling Foundation Models with Lightning Attention*，2025：<https://arxiv.org/abs/2501.08313>；*Why Did M2 End Up as a Full Attention Model?*：<https://www.minimax.io/news/why-did-m2-end-up-as-a-full-attention-model>
- NVIDIA. *Nemotron-H: A Family of Accurate and Efficient Hybrid Mamba-Transformer Models*，2025：<https://arxiv.org/abs/2504.03624>；*Nemotron 3 Nano*，2025：<https://arxiv.org/abs/2512.20848>
- TII. *Falcon-H1: A Family of Hybrid-Head Language Models Redefining Efficiency and Performance*，2025：<https://arxiv.org/abs/2507.22448>
- IBM. *IBM Granite 4.0: hyper-efficient, high performance hybrid models for enterprise*：<https://www.ibm.com/new/announcements/ibm-granite-4-0-hyper-efficient-high-performance-hybrid-models>
- Qwen Team. Qwen3-Next 与 Qwen3.5 的官方博客与模型卡：<https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct>、<https://huggingface.co/Qwen/Qwen3.5-0.8B>
- 蚂蚁百灵 Ling-3.0-tiny 模型卡：<https://huggingface.co/inclusionAI/Ling-3.0-tiny>（本仓库 `small-llms-under-5b-2026-08-30/` 的调研报告里也提到了它）
- flash-linear-attention：<https://github.com/fla-org/flash-linear-attention>
- Hugging Face transformers 的 Qwen3.5 实现：<https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/modeling_qwen3_5.py>
- vLLM Hybrid KV Cache Manager 设计文档：<https://github.com/vllm-project/vllm/blob/main/docs/design/hybrid_kv_cache_manager.md>
- CS336：<https://cs336.stanford.edu/>

**下一章**：这一章让每个 token 的"记忆"变便宜了：大部分层只需要一个固定大小的状态。但每个 token 经过 FFN 的计算量一点没少，而 FFN 占了模型参数和算力的大头。能不能让模型的参数很多，每个 token 却只用其中一小部分？第 24 章，混合专家（MoE）。
