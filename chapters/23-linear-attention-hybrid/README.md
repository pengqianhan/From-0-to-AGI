# 第 23 章：线性注意力与混合架构 —— 把 KV cache 压成一个固定大小的矩阵

> **一句话目标**：读完这一章，你能从 softmax 注意力出发，推导出线性注意力的递推形式 `S_t = S_{t−1} + v_t k_tᵀ`，写出 delta 规则和 Gated DeltaNet 的更新公式并验证"分块形式 == 递推形式"；还能用真实配置算出 Qwen3.5-0.8B 这种"3 层线性 + 1 层全注意力"的混合结构省下多少 KV cache，并说清楚它为什么还要保留那几层全注意力。

📺 **本章视频**：待发布（本地渲染：`bash chapters/23-linear-attention-hybrid/video/build.sh`）
🧪 **本章自检**：学完后在 Claude Code 里输入 `/ch23-linear-attention`

---

上一章我们用滑动窗口给注意力"限长"：一部分层只看最近的几千个 token，KV cache 就不再随上下文无限增长，但被窗口挡在外面的内容也就真的看不见了。这一章要解决的问题是：**能不能让一层注意力"看见"全部历史，却只花固定大小的内存？** 答案是线性注意力（linear attention）——把 softmax 拿掉，注意力就变成了一个 RNN，全部历史被压进一个 d×d 的状态矩阵。压缩必然有代价：状态会"满"，精确回忆会变差。于是有了两代改进（衰减门、delta 规则，合起来是 Gated DeltaNet），以及今天工业界的主流答案：**少量全注意力 + 大量线性层**的混合架构。千问 Qwen3.5 连 0.8B 的小模型都用了 3:1 的混合。

本章代码：

```bash
uv run python chapters/23-linear-attention-hybrid/code/01_linear_attention.py   # 结合律、KV vs 状态、解码一步（几秒）
uv run python chapters/23-linear-attention-hybrid/code/02_chunked.py            # 衰减门 + 分块并行形式（十几秒）
uv run python chapters/23-linear-attention-hybrid/code/03_delta_rule.py         # 覆盖、容量、Gated DeltaNet 的分块形式（几秒）
uv run python chapters/23-linear-attention-hybrid/code/04_hybrid_lm.py          # 四种结构的小语言模型（首次训练较慢，之后读缓存）
uv run python chapters/23-linear-attention-hybrid/code/05_associative_recall.py # 联想回忆：纯线性 vs 混合（首次训练较慢）
```

> 本章所有计时都在一台被多个任务共享的 CPU 上单线程测得，同一脚本多跑几次会有成倍的波动；表里的计时只用来看**数量级和趋势**。数值结果（误差、loss、准确率）固定随机种子，可复现。

## 1. 问题：softmax 注意力必须记住一切

回忆第 8 章的因果注意力：生成第 t 个 token 时，

```
o_t = Σ_{j≤t} softmax_j( q_t·k_j / √d ) · v_j
```

softmax 要对**这一行的全部 t 个分数**一起做归一化：只要还有一个 k_j 没参与，分母就算不出来。所以前面每个位置的 K、V 都得留着——这就是第 10、21 章的 KV cache，它随上下文长度线性增长。`01_linear_attention.py` 的第 2 部分算了单层单头（d = 64，BF16）的账：

| 上下文长度 T | KV cache | 线性注意力的状态 |
|---:|---:|---:|
| 128 | 32 KB | 8 KB |
| 1,024 | 256 KB | 8 KB |
| 8,192 | 2,048 KB | 8 KB |
| 65,536 | 16,384 KB | 8 KB |
| 262,144 | 65,536 KB | 8 KB |

右边那一列就是这一章要得到的东西：**不管上下文多长，大小都不变。**

## 2. 去掉 softmax：矩阵乘法换个顺序

如果注意力分数不过 softmax，而是直接用 `φ(q)·φ(k)`（φ 是某个把向量变成非负数的特征映射，Katharopoulos 等人 2020 年用的是 `elu(x) + 1`），那么整段序列的输出就是三个矩阵连乘，再乘一个因果掩码 M：

```
O = ( φ(Q) φ(K)ᵀ ⊙ M ) V
```

矩阵乘法满足**结合律**。不带掩码时，`(φ(Q) φ(K)ᵀ) V = φ(Q) (φ(K)ᵀ V)`：左边要先造一个 T×T 的大矩阵，右边只要一个 d×d 的小矩阵，和 T 无关。带上因果掩码，意思是"第 t 个位置只累加它之前的项"，于是可以边走边算：

```
S_t = S_{t−1} + v_t φ(k_t)ᵀ      （写入：状态 S 是 d_v × d_k 的矩阵）
o_t = S_t φ(q_t)                 （读出）
```

这就是一个**循环神经网络（RNN）**：每来一个 token，把 v 和 k 的外积加进状态，再用 q 去读。每一步的内存和计算都是常数。代码（`01_linear_attention.py`）：

```python
def linear_attention_recurrent(q, k, v):
    S = torch.zeros(v.shape[1], q.shape[1])
    out = []
    for t in range(q.shape[0]):
        S = S + torch.outer(v[t], phi(k[t]))  # S_t = S_{t-1} + v_t φ(k_t)ᵀ   （写入）
        out.append(S @ phi(q[t]))             # o_t = S_t φ(q_t)                （读出）
    return torch.stack(out)
```

和一次性算 T×T 矩阵的并行形式对比，T = 256、d = 16 时两种算法输出的**最大相对误差 4.7e-07**——只差 float32 的舍入误差。softmax 为什么不行？因为它的分母把一整行绑在一起，拆不开，所以 softmax 注意力没有这种递推形式。

> 说明：原论文的线性注意力还要除以一个归一化项 `φ(q_t)ᵀ Σ_j φ(k_j)`。后来的工作（GLA、DeltaNet、Qwen3.5 等）大多去掉了这个分母，改为对 q、k 做 L2 归一化、在输出上加一个 RMSNorm，数值更稳定。本章从第 3 节起都用这种现代写法（φ 取恒等映射）。

**推理时快了多少？** `01` 的第 3 部分测了"已经有 T 个历史 token 时，生成下一个 token 的注意力部分"（单头 d = 64）：

| 已有上下文 | softmax 注意力 | 线性注意力 |
|---:|---:|---:|
| 1,024 | 0.033 ms | 0.030 ms |
| 8,192 | 0.250 ms | 0.030 ms |
| 65,536 | 12.5 ms | 0.030 ms |
| 262,144 | 44.2 ms | 0.030 ms |

softmax 注意力每一步要读全部 T 个 K、V，时间随 T 增长；线性注意力每一步只碰一个 64×64 的状态，是常数。（计时波动很大：另一次运行里 262,144 那一格是 19–25 ms。）

## 3. 训练怎么办：分块并行形式

推理可以一个 token 一个 token 地走，训练却要一次处理整段序列。逐 token 的 for 循环没法并行，GPU 会闲着；完全并行的形式 `(QKᵀ ⊙ M) V` 又退回到 T×T 的矩阵。折中办法是**分块（chunkwise）**：把序列切成长度 C 的块，

- **块内**：用并行形式，一次矩阵乘法；
- **块间**：把前面所有块压缩成的状态 S 传下去，块内每个位置再额外读一次 S。

```python
for s in range(0, T, C):
    qc, kc, vc = q[s:s+C], k[s:s+C], v[s:s+C]
    b = g[s:s+C].cumsum(0)                                   # 块内累计对数衰减（下一节）
    D = (b[:, None] - b[None, :]).masked_fill(~mask, -inf).exp()
    inter = (qc * b.exp()[:, None]) @ S.T                    # 读块开始时的状态
    intra = ((qc @ kc.T) * D) @ vc                           # 块内的并行注意力
    out[s:s+C] = inter + intra
    S = b[-1].exp() * S + vc.T @ (kc * (b[-1] - b).exp()[:, None])   # 整块压进状态
```

`02_chunked.py` 验证三种算法数值一致（T = 512：递推 vs 并行最大差 1.8e-06，递推 vs 分块 6.0e-07），再比速度（单头 d = 64，CPU 单线程，毫秒）：

| T | 递推（逐 token） | 分块 C = 64 | 完全并行 |
|---:|---:|---:|---:|
| 256 | 33.4 | 4.1 | 3.9 |
| 1,024 | 76.2 | 12.7 | 79.6 |
| 4,096 | 344.1 | 47.2 | 2015.0 |

短序列时完全并行最快（一次大矩阵乘法）；序列一长，T² 的代价压倒一切，分块形式比逐 token 快约 7 倍、比完全并行快 40 多倍。GPU 上的差距会更大，因为分块把工作变成了 GPU 最擅长的矩阵乘法。**训练和 prefill 用分块形式，decode 用递推形式**，两者数学上完全等价——这是所有线性注意力模型的标准做法。

## 4. 衰减门：学会遗忘

朴素线性注意力有个明显的问题：它**只会累加**。一万个 token 之前写进去的东西，和刚写进去的东西权重一样，状态里越堆越多。第一个改进是加一个**衰减门（decay / gate）**：

```
S_t = α_t · S_{t−1} + v_t k_tᵀ,     α_t ∈ (0, 1]
```

每写入一次之前，先把旧状态整体乘上 α_t。`02_chunked.py` 里只在第 0 步写入一次，然后看它随距离怎么衰减：

| α | 距离 10 | 距离 100 | 距离 500 |
|---|---:|---:|---:|
| 1.0 | 1.000 | 1.000 | 1.000 |
| 0.99 | 0.904 | 0.366 | 0.007 |
| 0.9 | 0.349 | 0.000 | 0.000 |

正好是 α^距离。更关键的是，在真实模型里 **α_t 是由当前输入算出来的**（数据相关，data-dependent）：模型可以在"换了一个话题"时把门关小、清掉旧记忆，在需要长期记住的时候把门开到接近 1。分块形式里，衰减只是多了上面代码中的 `D` 矩阵和几个 `exp(b)` 因子（用对数衰减的累加和 b 表示，指数里永远是"后减前"，不会溢出）。

一大批模型的骨架就是这个"带衰减的线性递推"，区别只在 α 怎么参数化：RetNet 用固定的每头常数，GLA（Gated Linear Attention）用数据相关的向量门，Mamba-2 用数据相关的每头标量门（Mamba-2 论文把这类状态空间模型和线性注意力统一成了同一个框架，"结构化状态空间对偶"）。本章不逐一展开它们，只讲共同骨架；其中被主流模型真正采用的是 Mamba-2 和下面的 Gated DeltaNet 一族。

## 5. 代价：固定大小的记忆会满

天下没有免费的午餐。把整段历史压进一个固定大小的矩阵，记忆就会"满"。把状态 S 看成一张"键 → 值"的表：写入 (k, v)，之后用同一个 k 去读，希望 `S k ≈ v`。如果 key 两两正交，一个 d_k 维的状态最多存 d_k 个互不干扰的键值对；再多，不同 key 就互相串扰。

`03_delta_rule.py` 第 2 部分往一个 64×64 的状态里写入 N 个随机的单位向量 key 和随机 value，再逐个读回，报告相对误差 `‖S k − v‖ / ‖v‖`（0 = 完美读回，1 ≈ 读出来的噪声和信号一样大）：

| 写入 N 个 | 线性注意力 | delta 规则（下一节） | softmax 注意力（存全部 KV） |
|---:|---:|---:|---:|
| 16 | 0.469 | 0.309 | 0.000 |
| 32 | 0.692 | 0.479 | 0.000 |
| 64 | 0.996 | 0.737 | 0.000 |
| 128 | 1.431 | 0.972 | 0.000 |
| 256 | 1.974 | 1.206 | 0.000 |

softmax 注意力把 K、V 原样存着，用 key 去"查表"，误差始终是 0——代价是内存随 N 增长。线性注意力内存固定，代价是**精确回忆（recall）**：写得越多，读回来越糊。这是线性注意力最大的弱点，Zoology（Arora 等 2023）用"多查询联想回忆（MQAR）"任务系统地测过它，第 8.3 节我们也会做一个小版本。

## 6. delta 规则：覆盖，而不是累加

线性注意力的写入是盲目的：不管状态里已经存了什么，直接把 `v kᵀ` 加上去。同一个 key 写两次，读出来是两个值的和。**delta 规则**（delta rule，来自 Widrow–Hoff 的经典学习规则；Schlag 等人 2021 年把它用到线性注意力上，叫 DeltaNet）换了一种写法：先用 k 读出旧答案 `S k`，只把"新值与旧答案的差"写回去：

```
S_t = S_{t−1} + β_t (v_t − S_{t−1} k_t) k_tᵀ
    = S_{t−1} (I − β_t k_t k_tᵀ) + β_t v_t k_tᵀ
```

β_t ∈ (0, 1) 是写入强度（也由输入算出）。当 β = 1、‖k‖ = 1 时，写完以后 `S_t k_t = v_t` 恰好成立——旧值被干净地**覆盖**。它其实是对"让 S k 逼近 v"这个平方误差做了一步梯度下降（学习率 β），所以也被叫作"测试时学习"的一种。`03_delta_rule.py` 第 1 部分：同一个 key 先写 v1 = [1, 0]，再写 v2 = [0, 1]：

| | 用 k 读出 |
|---|---|
| 线性注意力 | [1.0, 1.0]（v1 + v2） |
| delta 规则 | [0.0, 1.0]（v2） |

上一节的容量表里，delta 规则在每个 N 上都比朴素累加的误差低（N = 64 时 0.737 vs 0.996）：写入前先"擦掉"key 方向上的旧内容，串扰小了很多。

## 7. Gated DeltaNet：衰减 × delta 规则

两个改进各管一件事：衰减门让旧信息整体淡出（适合"换话题"），delta 规则精确地改写某个 key 上的内容（适合"更新一条记忆"）。Gated DeltaNet（Yang, Kautz, Hatamizadeh 2024）把它们乘在一起：

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

几个观察：朴素累加被 1000 个旧条目淹没（4.05）；衰减或 delta 任何一个都能把误差拉回 0.6 左右；**固定的**衰减叠在 delta 上反而略差（它连最近的条目也一起衰减了）；而在"换话题"那一步把门关到 0 的**数据相关**门控，配合 delta 规则效果最好（0.502）。这说明门的价值在于"由输入决定什么时候忘"，这也是 Gated DeltaNet 里 α_t 必须由当前 token 算出来的原因。当然，0.5 离 softmax 注意力的 0 还很远——固定大小的状态终究有容量上限。

**分块形式。** delta 规则比朴素线性注意力多了一个麻烦：块内第 i 个位置真正写进状态的"修正值" u_i = β_i (v_i − S k_i) 依赖前面位置写入的 u_j，不能直接一次矩阵乘法算出来。把块内的依赖写成矩阵，是一个下三角线性方程组：

```
(I + A) U = β ⊙ (V − e^{b} ⊙ K S₀ᵀ),     A_ij = β_i e^{b_i − b_j} (k_i · k_j)   (j < i)
```

解出 U（一次三角求解，Yang 等人 2024 年的"UT 变换"/WY 表示），剩下的就和带衰减的线性注意力完全同形，只是把 v 换成 u：

```python
A = (bt[..., :, None] * (kc @ kc.mT) * D).tril(-1)            # A_ij = β_i e^{b_i−b_j} k_i·k_j, j<i
rhs = bt[..., None] * (vc - (kc * b.exp()[..., None]) @ S.mT)  # β_i (v_i − e^{b_i} S₀ k_i)
u = torch.linalg.solve_triangular(eye + A, rhs, upper=False)
out.append((qc * b.exp()[..., None]) @ S.mT + ((qc @ kc.mT) * D) @ u)
S = b[..., -1, None, None].exp() * S + u.mT @ (kc * (b[..., -1:] - b).exp()[..., None])
```

`03_delta_rule.py` 第 4 部分验证：B = 2、H = 3、T = 128、块长 32 时，分块与递推的最大输出差 4.8e-07，最终状态差 3.6e-07。

**真实模型里的 Gated DeltaNet 层**还多了几样东西（以 Qwen3.5 为准，见 `zero/arch/linear_attention.py`）：q、k、v 投影之后先过一个核长为 4 的**因果短卷积**（depthwise conv，让每个位置先混合一下左边相邻几个 token 的信息，这对回忆任务帮助很大）；q、k 做 **L2 归一化**（保证 ‖k‖ = 1，delta 规则才稳定）；α_t 用 Mamba-2 的参数化 `α_t = exp(−e^{A} · softplus(a_t + dt_bias))`；β_t = sigmoid(b_t)；输出过一个**门控 RMSNorm**（RMSNorm(o) ⊙ SiLU(z)）；多头，且 value 头数可以是 q/k 头数的整数倍（Qwen3-Next 是 16 个 q/k 头、32 个 v 头）。线性层**不加 RoPE**——递推本身就有先后顺序，短卷积也提供了局部位置信息。

## 8. 混合：少量全注意力 + 大量线性层

线性层记性有限，全注意力又太贵，工业界的答案是**混合**：大部分层用线性注意力（或状态空间层），每隔几层放一层全注意力，专门负责精确回忆。

### 8.1 Qwen3.5-0.8B 的真实结构

从 Hugging Face 上 `Qwen/Qwen3.5-0.8B` 的 `config.json` 读到（2026-09）：

```
"full_attention_interval": 4,
"layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention", ... ×6]
"num_key_value_heads": 2, "head_dim": 256,                       # 全注意力层（gated attention）
"linear_num_key_heads": 16, "linear_num_value_heads": 16,
"linear_key_head_dim": 128, "linear_value_head_dim": 128,        # Gated DeltaNet 层
"linear_conv_kernel_dim": 4, "mamba_ssm_dtype": "float32"
```

模型卡写得更直白：`Hidden Layout: 6 × (3 × (Gated DeltaNet → FFN) → 1 × (Gated Attention → FFN))`。`04_hybrid_lm.py` 最后一部分按这份配置算了一条序列的推理缓存（KV 用 BF16，递推状态按 config 用 FP32）：

<<QWEN_TABLE>>

KV cache 只跟着那 6 层全注意力走；另外 18 层 Gated DeltaNet 的状态加起来 18.6 MiB，与上下文长度无关。上下文越长，省得越接近 4 倍（3:1 的理论上限）。注意这里的"24 层全注意力"是一个假想对照（把线性层换成同配置的全注意力层），不是某个真实模型。

### 8.2 小实验一：同一个小语言模型，四种结构

`04_hybrid_lm.py` 用第 10 章同款的字符级莎士比亚语料，训练四个 4 层小模型（宽度 128、4 个头、SwiGLU FFN、同样的数据顺序和超参，各 800 步），只换每层的 token mixer：A = softmax 注意力（带 RoPE），L = 朴素线性注意力，G = Gated DeltaNet（L 和 G 都带短卷积和输出归一化）。

<<LM_TABLE>>

### 8.3 小实验二：联想回忆——纯线性掉队，混合找回来

语言模型的 loss 对"精确回忆"不太敏感（大部分字符靠局部上下文就能猜）。`05_associative_recall.py` 专门测回忆：序列前半段是 N 个随机的"键 值"对，后半段反复给出其中的某个键，模型要答出它对应的值（Zoology 的 MQAR 任务的缩小版）。

<<RECALL_TABLE>>

### 8.4 为什么是"少量全注意力 + 大量线性层"

把上面的现象合起来：

- **线性层负责"便宜的深度"**：它们处理局部模式、语法、逐步累积的语义，内存和每步计算都与长度无关；
- **少量全注意力层负责"精确回忆"**：从很远的上下文里原样找回某个名字、数字、代码变量，这恰恰是固定大小的状态最不擅长的；
- **KV cache 只随全注意力层数增长**：3:1 时约为纯注意力的 1/4，9:1 时约 1/10。

比例没有标准答案：Qwen3.5、Kimi Linear、Ling-3.0 是 3:1；IBM Granite 4.0-H 是 9:1；NVIDIA Nemotron 3 Nano 在 52 层里只放了 6 层注意力；Falcon-H1 干脆在同一层里把注意力头和 Mamba-2 头并联。它们在"KV cache 能省多少"和"回忆能力保留多少"之间各取了一个点。

也有明确的反例。MiniMax-Text-01（2025 年初）用了 lightning attention（一种线性注意力）与 softmax 注意力 7:1 的混合（config 里 80 层的 `attn_type_list` 每 8 层有 1 层是 softmax）；但它的下一代 MiniMax-M2 回到了**每一层都是全注意力**（MiniMax-M2.5 的 `attn_type_list` 全是 1）。MiniMax 官方博客解释了原因：在代码、数学、智能体、长链推理和强化学习这些复杂任务上，线性/稀疏注意力的效果还不够稳定；线性注意力的训练和推理基础设施不成熟、很多实现受限于显存带宽；它对数值精度更敏感，低精度存储状态有困难；和推测解码的配合也还是未解决的问题。所以混合架构是一个已经被多家采用的**工程权衡**，而不是免费的午餐。

## 9. 小结

- softmax 注意力要对一整行分数归一化，只能把全部 K、V 留着：KV cache ∝ T。
- 去掉 softmax，矩阵乘法换个顺序，注意力就变成 RNN：`S_t = S_{t−1} + v_t k_tᵀ`，`o_t = S_t q_t`，状态大小固定。
- 训练用**分块形式**（块内并行、块间传状态），decode 用**递推形式**，两者数学等价。
- 固定大小的状态会满：精确回忆是线性注意力最大的弱点。
- **衰减门** α_t 让旧信息淡出（RetNet / GLA / Mamba-2 一族）；**delta 规则**先读后写、只写差值，实现"覆盖"；两者相乘就是 **Gated DeltaNet**。
- **混合架构**：大量线性层 + 少量全注意力层（Qwen3.5 是 3:1），KV cache 只随全注意力层增长，精确回忆由全注意力层兜底。

---

## 从极简到生产级

生产级实现在 `zero/arch/linear_attention.py`（第五部分的实验模块，**不用于主线模型**）。它的结构和参数名与 Hugging Face transformers 的 Qwen3.5 实现（`transformers/models/qwen3_5/modeling_qwen3_5.py` 的 `Qwen3_5GatedDeltaNet`）一致，可以直接搬权重。

| 极简版（`code/`） | 生产级（`zero/arch/linear_attention.py`） | 多做了什么、为什么 |
|---|---|---|
| `01` 的 `linear_attention_recurrent`、`02` 的 `recurrent` | `recurrent_linear_attention(q, k, v, g, initial_state)` | 批量、多头 `(B, H, T, D)`；可以接着上一次的状态继续算（decode 需要）；内部用 float32 |
| `02` 的 `chunked`（要求 T 是 C 的整数倍） | `chunk_linear_attention(..., chunk_size, initial_state)` | 自动补零（补的位置 k = 0、g = 0，对状态没有影响）；带初始状态，支持分段 prefill |
| `03` 的 `gated_delta_recurrent` / `gated_delta_chunked` | `recurrent_gated_delta_rule` / `chunk_gated_delta_rule` | 同上；状态布局改成 HF / fla 的 `(d_k, d_v)`（极简版是 `(d_v, d_k)`，只是转置） |
| `04` 的 `LinearMixer`（q/k/v 投影、短卷积、L2 归一化、每头 RMSNorm） | `LinearAttention`、`GatedDeltaNet`（共用 `_LinearMixer`） | 参数名与 HF 相同（`in_proj_qkv`、`in_proj_z`、`in_proj_b`、`in_proj_a`、`conv1d`、`A_log`、`dt_bias`、`norm`、`out_proj`）；**门控 RMSNorm**（输出乘 SiLU(z)）；value 头数可以是 q/k 头数的整数倍；Mamba-2 式的 A、dt 初始化；短卷积带缓存（保存最后 K−1 个输入），分段喂和一次喂结果相同；`mode="auto"` 时 T = 1 走递推、否则走分块 |
| `04` 的 `TinyLM(pattern="GGGA")` | `HybridConfig`、`HybridTransformer`、`hybrid_layer_types(n_layers, full_attention_interval=4)` | 配置驱动，`layer_types` 与 HF 的 Qwen3.5 config 同名同义；全注意力层直接复用主线的 `zero.model.Attention`（GQA + QK-Norm + RoPE + SDPA） |
| 无 | `HybridCache`：全注意力层用预分配的 `KVCache`（只为全注意力层分配），线性层用 `LinearState`（递推状态 + 卷积尾巴）；`generate_greedy` | 推理时两种缓存并存；`cache_bytes_per_sequence` 按层记账（第 21 章账本的延伸） |
| 无 | 未实现：Qwen3.5 全注意力层的**输出门**（gated attention，`attn_output_gate: true`）、部分 RoPE（`partial_rotary_factor: 0.25`）、MoE、MTP | 本章只关心 token mixer 的混合方式；这些在第 24、25 章和第 26 章的全景里讲 |

**对拍**：`tests/test_arch_linear_attention.py`（`uv run pytest tests/test_arch_linear_attention.py`，本机 33 项全部通过，约 10 秒），保证：

- 线性注意力（带/不带衰减）、gated delta rule（带/不带衰减）的**分块形式 == 递推形式**，块长 1、5、8、16、64，长度不整除块长，带随机初始状态；
- 不衰减的线性注意力 == 掩码并行形式 `(QKᵀ ⊙ M) V`；
- gated delta rule == 逐步用显式矩阵 `S_t = α_t (I − β_t k_t k_tᵀ) S_{t−1} + β_t k_t v_tᵀ` 的朴素参考；同一个 key 写两次时 delta 规则覆盖、线性注意力累加；
- `LinearAttention` / `GatedDeltaNet` 层：分块、递推、分三段带状态喂（10 + 1 + 12 个 token）三者输出一致；状态形状与长度无关；
- **与 HF transformers 的 `Qwen3_5GatedDeltaNet` 对拍**：随机化 HF 层的全部权重后 `load_state_dict(strict=True)` 搬进来，输出在 1e-4 内一致；
- `HybridTransformer`（间隔 2、4、99，即 1:1、3:1、纯线性；Gated DeltaNet 与朴素线性两种）：**状态缓存生成 == 每步全量重算**（贪心 30 个 token 完全相同）；分块 prefill（20 + 1 + 24）== 一次性前向；KV 字节数随长度翻倍、线性状态不变；几步训练 loss 下降（梯度能穿过三角求解）。

**真正训练和部署时**，纯 PyTorch 的分块循环太慢了。行业做法是：

- **flash-linear-attention（fla-org）**：Triton 写的线性注意力 kernel 库，`fla.ops.gated_delta_rule.chunk_gated_delta_rule` / `fused_recurrent_gated_delta_rule` 就是本章两种形式的 GPU 实现，Kimi Linear 的 KDA kernel（`fla.ops.kda`）也开源在里面；HF transformers 的 Qwen3.5 实现在装了 fla 和 causal-conv1d 时会自动换用这些 kernel，否则退回到和本章同构的纯 PyTorch 版本。
- **vLLM**：`vllm/model_executor/models/qwen3_next.py`、`qwen3_5.py` 等支持这些混合模型；它的混合 KV cache 管理器（Hybrid KV Cache Manager 设计文档）为不同类型的层分配不同的缓存：全注意力层按 token 数分配 KV 页，Mamba / 线性层按请求分配固定大小的状态。

`zero/arch/linear_attention.py` 里这些 GPU 路径都标注了"尚未在 GPU 上验证"：本章只在 CPU 上用纯 PyTorch 验证了数学。

---

## 前沿观察

> **不算共识、只在这里提一句的技术**
>
> - **纯线性 / 纯状态空间模型**（纯 Mamba、RWKV、RetNet 等）：它们证明了线性递推可以单独撑起一个语言模型，但头部开源模型家族的主力版本没有采用纯线性结构，全部是混合（GOAL.md 2.1 把它们列在"只在前沿观察一句带过"）。
> - **KDA（Kimi Delta Attention）**：Kimi Linear 提出的 Gated DeltaNet 改进版，把每头一个标量的衰减门换成逐通道（每个 key 维度一个）的细粒度门；蚂蚁 Ling-3.0 也采用了它。目前只有这两家（且 Ling 明确说继承自 Kimi Linear 的设计思路），还不算独立的多家共识。
> - **并联混合**（Falcon-H1）：同一层里注意力头和 Mamba-2 头并行、输出拼接，而不是按层交替。目前只见于 Falcon-H1。
> - **比例之争**：3:1、9:1、甚至"全注意力"（MiniMax-M2）都有头部模型在用，最佳比例和"全注意力层放在哪几层"还没有共识。
> - **稀疏注意力**（DeepSeek DSA/NSA 一类，MiniMax-M3 也用了块稀疏注意力）是另一条"让长上下文变便宜"的路线，见第 22 章。

## 采用方与来源

| 技术 | 采用方（主力版本） | 来源 |
|---|---|---|
| 混合线性注意力：DeltaNet 一族线性层 + 少量全注意力 | **Qwen3-Next-80B-A3B**（48 层 = 12 × [3 × Gated DeltaNet + 1 × Gated Attention]）；**Qwen3.5 全系列**（0.8B、2B、4B、9B、27B、35B-A3B、122B-A10B、397B-A17B 的 config 均为 `full_attention_interval: 4`）；**Kimi Linear 48B-A3B**（KDA : MLA = 3:1，27 层中 7 层 MLA）；**蚂蚁 Ling-3.0-tiny**（3 层 KDA + 1 层 MLA 为一组） | 模型卡与 config.json：[Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B)、[Qwen3.5-27B config](https://huggingface.co/Qwen/Qwen3.5-27B/blob/main/config.json)、[Qwen3.5-397B-A17B config](https://huggingface.co/Qwen/Qwen3.5-397B-A17B/blob/main/config.json)、[Qwen3-Next-80B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct)、[Kimi-Linear-48B-A3B-Instruct](https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct)（技术报告 arXiv:2510.26692）、[Ling-3.0-tiny](https://huggingface.co/inclusionAI/Ling-3.0-tiny) |
| 混合状态空间：Mamba-2 层 + 少量注意力层 | **NVIDIA Nemotron-H**、**Nemotron 3**（Nano 30B-A3B：52 层里 23 层 Mamba-2、23 层 MoE、6 层注意力；Ultra 550B-A55B 同为 `nemotron_h` 结构）；**IBM Granite 4.0-H**（H-Small：40 层里 4 层注意力，官方称 Mamba-2 : Transformer = 9:1）；**TII Falcon-H1**（每层注意力与 Mamba-2 并联） | [Nemotron-H 技术报告 arXiv:2504.03624](https://arxiv.org/abs/2504.03624)、[Nemotron 3 Nano 技术报告 arXiv:2512.20848](https://arxiv.org/abs/2512.20848) 与 [模型卡](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16)（`hybrid_override_pattern`）；[granite-4.0-h-small config](https://huggingface.co/ibm-granite/granite-4.0-h-small/blob/main/config.json) 与 [IBM 发布公告](https://www.ibm.com/new/announcements/ibm-granite-4-0-hyper-efficient-high-performance-hybrid-models)；[Falcon-H1 技术报告 arXiv:2507.22448](https://arxiv.org/abs/2507.22448) |
| 线性注意力（lightning attention）+ softmax 7:1（后放弃） | **MiniMax-Text-01 / M1**（80 层，每 8 层 1 层 softmax）；**MiniMax-M2 / M2.5 回到全注意力** | [MiniMax-01 技术报告 arXiv:2501.08313](https://arxiv.org/abs/2501.08313)、[MiniMax-Text-01 config](https://huggingface.co/MiniMaxAI/MiniMax-Text-01/blob/main/config.json)、[MiniMax-M2.5 config](https://huggingface.co/MiniMaxAI/MiniMax-M2.5/blob/main/config.json)、官方博客 [Why Did M2 End Up as a Full Attention Model?](https://www.minimax.io/news/why-did-m2-end-up-as-a-full-attention-model) |
| 分块形式 + 递推形式的 kernel | flash-linear-attention（被 HF transformers 的 Qwen3.5 / Qwen3-Next 实现调用，Kimi 的 KDA kernel 开源于此）；vLLM 的混合缓存管理 | <https://github.com/fla-org/flash-linear-attention>；vLLM [Hybrid KV Cache Manager](https://github.com/vllm-project/vllm/blob/main/docs/design/hybrid_kv_cache_manager.md) |

按 GOAL.md 2.1 的规则 A 计数：**"少量全注意力 + 大量线性/状态空间层"的混合结构**有 Qwen、Kimi、NVIDIA Nemotron 三个清单内的头部家族在主力版本中明确采用（另有 IBM Granite、Falcon、蚂蚁 Ling），满足共识条件，进正文。**Gated DeltaNet 本身**只有 Qwen 直接采用（Kimi 的 KDA、Ling 是它的改进版），所以本章把它作为"DeltaNet 一族线性层"的代表来讲，把 KDA 放在前沿观察。

说明：各模型的层数、比例都读自 Hugging Face 上的 config.json 和模型卡（2026-09 读取）；Qwen3.5 技术报告写作时未找到独立的 arXiv 版本，以官方博客和模型卡为准（**待核实**是否有正式技术报告）。

---

## 引导问题

带着这些问题去问 Claude Code，直到你能用自己的话讲清楚：

1. 线性注意力的递推形式里，状态 S 的形状是 d_v × d_k。如果把多头注意力的 H 个头都换成线性注意力，整层的状态有多大？和同样 H 个头、长度为 T 的 KV cache 相比，T 为多少时两者相等？
2. 为什么 delta 规则要求 k 做 L2 归一化？如果 ‖k‖ = 2、β = 1，写完以后 S k 等于什么？会不会发散？（提示：看 I − β k kᵀ 的特征值。）
3. 本章说 delta 规则是"对 ‖S k − v‖² 做了一步梯度下降"。试着自己求一下这个损失对 S 的梯度，验证它和 delta 规则的更新一致。那么"衰减门"在这个视角下相当于什么？（提示：权重衰减。）
4. 训练用分块形式、推理用递推形式。如果训练时块长取 1，分块形式退化成什么？如果块长取整个序列长度呢？块长应该怎么选？
5. Qwen3.5 的线性层不加 RoPE，全注意力层只对 head_dim 的 25% 加 RoPE（`partial_rotary_factor: 0.25`）。线性层的位置信息从哪里来？
6. MiniMax-M2 回到了全注意力。如果你要为一个 0.6B、主要做工具调用的小模型（本课的主线模型）选择架构，你会用 3:1 混合还是全注意力？列出你的理由和需要做的对比实验。

## 动手任务

每个任务都要真的运行代码、看到结果。

**任务 1（基础）**：在 `03_delta_rule.py` 的容量实验里，把 key 换成**两两正交**的向量（例如 `torch.linalg.qr` 得到的正交矩阵的行），N 取 16、32、64、65。线性注意力和 delta 规则的读回误差分别是多少？为什么 N = 64 和 65 之间会有突变？

**任务 2（核心）**：在 `05_associative_recall.py` 里加一种结构"GAGG"（全注意力放在第 2 层而不是最后一层），和"GGGA"比较回忆准确率。再试试把线性层的 head_dim 翻倍（状态变成 4 倍大），纯 Gated DeltaNet 的准确率能追上多少？

**任务 3（挑战）**：用 `zero/arch/linear_attention.py` 的 `HybridTransformer` 搭一个和 `04_hybrid_lm.py` 同尺寸的 3:1 混合模型，在同样的语料上训练 800 步，对比验证 loss；然后用 `model.new_cache()` 做缓存生成，测量生成 2000 个字符时 `HybridCache.nbytes()` 里 KV 和线性状态各占多少，并和 `cache_bytes_per_sequence` 的公式核对。

---

## 想深入：CS336

本章对应斯坦福 CS336（Spring 2026）<https://cs336.stanford.edu/>：

- **第 4 讲：注意力的替代方案与 MoE**。讲义与录像见课程页；课上讨论了为什么要找 softmax 注意力的替代品，以及线性注意力、状态空间模型这些方向的基本思路。本章的推导（结合律 → 递推 → 分块）和这一讲的线性注意力部分对应。
- **CS336 未深入**：delta 规则、Gated DeltaNet 的分块（UT 变换）算法，以及 Qwen3.5 / Kimi Linear / Nemotron 这类工业混合架构的具体配置，CS336 没有展开，见本章参考文献（尤其是 Songlin Yang 等人的 DeltaNet 与 Gated DeltaNet 论文，以及 flash-linear-attention 仓库）。

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

**下一章**：这一章让每个 token 的"记忆"变便宜了：大部分层只需要一个固定大小的状态。但每个 token 经过的 FFN 计算一点没少，而 FFN 占了模型参数和算力的大头。能不能让模型参数很多、每个 token 却只用其中一小部分？第 24 章，混合专家（MoE）。
