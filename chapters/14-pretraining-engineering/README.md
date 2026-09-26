# 第 14 章：预训练工程 —— 混合精度、FlashAttention、数据并行与断点续训

> **一句话目标**：读完这一章，你能给主线模型的一步训练算清三本账（算力、显存、时间），说清楚 BF16 混合精度、FlashAttention 的分块 + online softmax、DDP 与 FSDP 各自解决什么问题，并能亲手验证"2 个进程各算一半再平均梯度"和"崩溃后续训"都与不中断的单进程训练逐位一致。

📺 **本章视频**：待发布（本地渲染：`bash chapters/14-pretraining-engineering/video/build.sh`）
🧪 **本章自检**：学完后在 Claude Code 里输入 `/ch14-pretraining`

---

上一章我们准备好了数据和分词器，第 12 章定下了主线模型的尺寸：689.5M 参数、先训 500B token。第 6 章讲过怎么让训练在数学上稳定（初始化、RMSNorm、AdamW、warmup、梯度裁剪）。这一章要解决的问题是：**同一个训练循环，怎么在 8 张 GPU 上跑两周不出事、不浪费钱？**

这不再是数学问题，而是工程问题。算一下就知道分量：主线模型每训练一个 token 要 69.6 亿次浮点运算，500B token 一共 3.5 × 10²¹ 次；8 张 H100 按 40% 的效率跑，要 12.7 天、约 6100 美元。效率每掉 10 个百分点，就要多花一千多美元。这一章的每一个技术，要么是让 GPU 算得更快（混合精度、FlashAttention），要么是让显存装得下（激活检查点、FSDP），要么是让多张卡一起干活（数据并行），要么是让两周的训练出了事也不白跑（loss spike 处理、断点续训）。

> 本章所有代码都在 CPU 上运行。GPU 上的数字（吞吐、MFU、显存）都是**按公式估算**，标注"尚未在 GPU 上验证"，第二步的第一件事就是一次不超过 $50 的 GPU 验证运行（见"主线进度"）。

## 1. 先算账：一步预训练要花多少

第 12 章推过，训练一个 token 的浮点运算量（前向 + 反向）是：

```
每 token FLOPs = 6 · N_matmul + 12 · L · q_dim · T
```

`6N` 来自"前向每个参数一次乘加（2 FLOPs），反向两次（4 FLOPs）"；后一项是注意力里 `QKᵀ` 和 `AV` 两个没有参数的矩阵乘，和序列长度 T 成正比。运行：

```bash
uv run python chapters/14-pretraining-engineering/code/01_step_cost.py
```

```python
per_layer = d * q_dim + 2 * d * kv_dim + q_dim * d + 3 * d * m["ffn_dim"]
n_matmul = L * per_layer + d * m["vocab_size"]  # lm_head 即使和 embedding 共享，乘法照算
return 6 * n_matmul + 12 * L * q_dim * T, n_matmul, n_total
```

主线模型（`configs/main/pretrain.toml`：28 层、dim 1280、16 个查询头 × 128、8 个 KV 头、FFN 3584、词表 65,536、T = 4096）：

| 量 | 数值 |
|---|---:|
| 总参数 N / 参与矩阵乘的 N_matmul | 689.5M / 689.4M |
| 每 token FLOPs | 6.955 × 10⁹ |
| 每步 token（micro batch 8 × 累积 2 × 8 卡 × 4096） | 524,288 |
| 每步 FLOPs | 3.647 × 10¹⁵ |
| 500B token 总 FLOPs | 3.478 × 10²¹ |

换算成 8×H100 上的时间和钱（H100 SXM 稠密 BF16 峰值 989.5 TFLOPS，$2.5/卡时）：

| MFU | 8 卡吞吐（token/s） | 每步秒数 | 天数 | 卡时 | 费用 |
|---:|---:|---:|---:|---:|---:|
| 0.3 | 341,442 | 1.54 | 16.95 | 3,254 | $8,135 |
| 0.4 | 455,256 | 1.15 | 12.71 | 2,441 | $6,102 |
| 0.5 | 569,070 | 0.92 | 10.17 | 1,953 | $4,881 |

表里的 **MFU（Model FLOPs Utilization，模型算力利用率）** 是这一章反复出现的指标，第 6 节细讲。先记住它的量级：PaLM 540B 报告 46.2%，Llama 3 405B 报告 38–43%，Nemotron-4 340B 报告 41–42%。同样的 500B token，MFU 从 0.3 提到 0.5，能省下 3000 多美元。生产级的估算工具是 `zero/tools/estimate_cost.py`，输出与上表 MFU 0.4 那一行一致（2,440.6 卡时、12.71 天、$6,102）。

## 2. 显存里装了什么

算力决定"要多久"，显存决定"能不能跑"。训练一步时，显存里有四类东西：

- **参数**：FP32 主权重，每个 4 字节；
- **梯度**：和参数同形状同精度，4 字节；
- **优化器状态**：AdamW 的一阶矩 m、二阶矩 v（第 6 章），各 4 字节；
- **激活值（activations）**：前向时为反向保存的中间张量，和 micro batch × 序列长度成正比。

前三项加起来就是常说的"**每个参数 16 字节**"。`05_memory.py` 在 CPU 上用 zero 的真实模型（tiny 形状，1,049,984 个参数），走一步 AdamW 后逐项数：

| | 字节 | 每参数 |
|---|---:|---:|
| 参数（FP32） | 4,199,936 | 4 |
| 梯度（FP32） | 4,199,936 | 4 |
| AdamW m（FP32） | 4,199,936 | 4 |
| AdamW v（FP32） | 4,199,936 | 4 |
| 合计 | 16,799,744 | 16 |

激活值怎么数？PyTorch 提供了一个钩子 `torch.autograd.graph.saved_tensors_hooks`，前向时 autograd 每保存一个"反向要用的张量"都会经过它。我们按存储去重，把它们的字节数加起来：

```python
def pack(t):
    key = t.untyped_storage().data_ptr()
    if key not in own and key not in seen:        # 不算参数本身，同一块内存只算一次
        seen[key] = t.untyped_storage().nbytes()
    return t
```

micro batch 4 × 序列 128 = 512 个 token、4 层：

| 情况 | 保存的字节 | 每 token | 相对 |
|---|---:|---:|---:|
| FP32 | 30,251,012 | 59,084 | 100% |
| BF16 autocast | 24,090,628 | 47,052 | 80% |
| BF16 + 激活检查点 | 6,428,676 | 12,556 | 21% |

**激活检查点（activation checkpointing，也叫重算 / rematerialization）**：前向时每个 Transformer 块只保存它的输入，块内部的中间结果全部丢掉；反向算到这一块时，从保存的输入把这一块的前向再算一遍。代价是多做一次前向，约多 1/3 的计算量（6N 变成 8N）；换来的是激活显存从"每层几十个张量"降到"每层一个张量"。

```python
def forward(self, x, cos, sin, kv_cache=None, start_pos=0):
    return checkpoint(self.block, x, cos, sin, use_reentrant=True)   # 只存 x，反向时重算整块
```

同样的账算到主线模型上（生产级计算器 `zero/tools/memory_calc.py`，8×H100 80GB，BF16，seq 4096）：

| micro batch | 激活 | DDP 每卡合计 | FSDP 每卡合计 | DDP + 激活检查点 |
|---:|---:|---:|---:|---:|
| 1 | 11.0 GiB | 26.6 GiB | 14.0 GiB | 16.5 GiB |
| 2 | 21.9 GiB | 39.1 GiB | 26.5 GiB | 19.9 GiB |
| 4 | 43.9 GiB | 64.0 GiB | 51.4 GiB | 26.8 GiB |
| 8 | 87.7 GiB | 113.8 GiB | 101.2 GiB | 40.6 GiB |

参数本身只要 689.5M × 16 字节 = 10.3 GiB，真正的大头是激活：每个 token 每层 92,840 字节，28 层加上输出头，一个 token 约 2.9 MB。**micro batch 8 在 80 GB 的卡上放不下**（这是 eager 模式下的保守上界；`torch.compile` 会融合逐元素运算、少存中间张量，但不太可能省掉 30 GB）。这正是下一招的用武之地。

### 梯度累积：用时间换显存

第 1 章说过，PyTorch 默认把梯度**累加**进 `.grad`，所以每步开始要 `zero_grad()`。反过来利用这个特性：连续跑 k 个 micro batch，每个的 loss 先除以 k 再 backward，梯度自然累加成"大 batch 的平均梯度"，然后才 `optimizer.step()` 和 `zero_grad()`。

```python
for xs, ys in zip(x.chunk(accum), y.chunk(accum)):
    loss = loss_fn(model, xs, ys) / accum    # 除以累积步数：累加结果 = 大 batch 的平均梯度
    loss.backward()                          # 梯度累加进 .grad
opt.step()
opt.zero_grad()                              # 一步结束才清零
```

激活显存只和 micro batch 有关，和累积步数无关。所以主线的每步 524,288 token 可以写成"micro batch 4 × 累积 4 × 8 卡 × 4096"，显存按 micro batch 4 算（64.0 GiB），每步 token 数不变。`06_ddp_by_hand.py` 验证了梯度累积和一次吃大 batch 的结果一致（第 5 节的表）。

## 3. 混合精度：为什么是 BF16

### 3.1 位布局

一个浮点数 = 符号位 + 指数位 + 尾数位。指数位决定**范围**（能表示多大、多小），尾数位决定**精度**（相邻两个数隔多远）。

```
FP32：1 + 8 + 23     BF16：1 + 8 + 7      FP16：1 + 5 + 10
```

`02_precision.py` 把 π 存进三种格式：

| 格式 | π 的二进制位（符号 \| 指数 \| 尾数） | 存下来的值 | 最大值 | 最小正规数 | epsilon |
|---|---|---:|---:|---:|---:|
| FP32 | `0 \| 10000000 \| 10010010000111111010000` | 3.1415901 | 3.4 × 10³⁸ | 1.18 × 10⁻³⁸ | 1.19 × 10⁻⁷ |
| BF16 | `0 \| 10000000 \| 1001001` | 3.1406250 | 3.39 × 10³⁸ | 1.18 × 10⁻³⁸ | 0.00781 |
| FP16 | `0 \| 10000 \| 1001001000` | 3.1406250 | 65,504 | 6.1 × 10⁻⁵ | 0.000977 |

BF16（brain floating point）就是把 FP32 的尾数砍掉 16 位：指数位一样多，所以**范围和 FP32 相同**，只是精度粗（epsilon 2⁻⁷ ≈ 0.0078，约 2–3 位有效十进制数字）。FP16 尾数多 3 位、精度更好，但指数只有 5 位，最大只能到 65,504。

### 3.2 精度不够会怎样

把 0.01 连加一万次，正确答案是 100：

| 累加器 | 结果 |
|---|---:|
| FP32 | 100.0030 |
| BF16 | 4.0000 |
| FP16 | 32.0000 |

BF16 在 [4, 8) 里相邻两数相距 4 × 2⁻⁷ = 0.03125，加上 0.01 不到间隔的一半，每次又被舍回 4，累加器"卡住"了。同样的事会发生在权重更新上：权重 w = 1.0，每步减 10⁻³，走 1000 步，正确答案是 0：BF16 权重始终是 **1.0000**，FP32 权重是 0.0000。学习率乘梯度通常比权重小好几个数量级，BF16 的权重根本吃不下这么小的更新。

所以**混合精度（mixed precision）**的标准做法（Micikevicius et al., 2017）是分工：

- **矩阵乘用低精度算**：GPU 的 Tensor Core 在 BF16 下的吞吐比 FP32 高得多，激活值也省一半显存；
- **主权重、梯度累加、优化器状态保留 FP32**：保证小更新不丢；
- **对精度敏感的运算（归一化、softmax、损失）用 FP32 算**。

PyTorch 用 `torch.autocast` 实现这个分工：在它的作用域里，矩阵乘自动转成 BF16，参数本身不变。`02_precision.py` 的最后一个实验：

| 输出 dtype | 参数 dtype | 梯度 dtype | 与 FP32 结果的相对误差 |
|---|---|---|---:|
| bfloat16 | float32 | float32 | 2.78 × 10⁻³ |

另一种思路是**随机舍入（stochastic rounding）**：舍入时按距离远近随机进位或舍去，期望值等于原数。同样把 0.01 加一万次，BF16 + 随机舍入在 5 个种子下得到 95.50、92.50、98.00、100.00、103.50，平均 97.90：不再卡住，但每次有噪声。主流训练框架用的仍是"FP32 主权重"，这里只作为理解舍入误差的对照。

### 3.3 为什么不用 FP16

FP16 的问题在范围：70,000 存进 FP16 直接变成 `inf`；10⁻⁸ 存进 FP16 变成 0（BF16 能存成 1.00117 × 10⁻⁸）。一批标准差 10⁻⁶ 的小梯度转成 FP16，有 2.4% 变成 0，转成 BF16 一个都不丢。FP16 训练因此要配 **loss scaling**：先把 loss 乘一个大数（例如 1024）再反向，梯度整体放大、躲开下溢，更新前再除回来；乘 1024 之后变成 0 的比例降到 0.0%。但放大倍数要动态调，放大过头又会溢出成 inf，要跳过那一步。BF16 的范围和 FP32 一样，这些麻烦都没有，所以 A100 之后的大模型训练几乎都用 BF16（采用方见文末）。

**FP8**：H100 开始支持 8 位浮点矩阵乘，理论吞吐再翻倍。DeepSeek-V3 用细粒度量化的 FP8 训练了 671B 的 MoE 模型。FP8 是否已满足本课的共识规则，见第 12 章的核实；本章把它放在章末"前沿观察"，主线默认 BF16。

## 4. FlashAttention：注意力慢在"读写"，不在"算"

### 4.1 问题：memory-bound

GPU 有两种内存：容量大但慢的显存（HBM，H100 上 80 GB），和容量很小但快得多的片上缓存（SRAM，每个计算单元几百 KB）。一个运算如果每读一个字节只做很少的计算，时间就耗在搬数据上，叫 **memory-bound（受访存限制）**；反之叫 compute-bound。

朴素注意力正是前者：先算 `S = QKᵀ/√d`（T × T），写回显存；读出来做 softmax 得到 P（T × T），再写回；再读出来乘 V。softmax 本身几乎不花算力，却要把两个 T × T 的矩阵完整地读写好几遍。`04_tiled_attention.py` 最后一段算了主线模型的尺度：T = 4096、16 个头、micro batch 8、BF16，**一层的 S 和 P 就是 8.0 GiB**，28 层为反向保留 P 要 112 GiB。

### 4.2 online softmax：一遍扫描，边走边改正

要避免写出整个 S，就得一块一块地算 softmax。难点在于 softmax 要先知道整行的最大值（数值稳定的写法是减去最大值再取指数），而分块时后面的块还没看到。

**online softmax**（Milakov & Gimelshein, 2018）只扫一遍，维护两个"到目前为止"的量：

```
m_j = max(m_{j−1}, x_j)                                   当前最大值
l_j = l_{j−1} · exp(m_{j−1} − m_j) + exp(x_j − m_j)         以当前最大值为基准的指数和
```

出现新的最大值时，之前的和是按旧的 m 算的，乘一个 `exp(m_旧 − m_新)`（≤ 1）就改正到新基准。扫完之后 `softmax_i = exp(x_i − m_N) / l_N`。

```python
for xi in x:
    m_new = max(m, xi)
    l = l * math.exp(m - m_new) + math.exp(xi - m_new)   # 旧的和按新最大值改正
    m = m_new
```

`03_online_softmax.py` 用 x = [1, 3, 2, 5, 4] 逐步打印：

| j | x_j | m_j | l_j | 发生了什么 |
|---:|---:|---:|---:|---|
| 1 | 1 | 1 | 1.0000 | |
| 2 | 3 | 3 | 1.1353 | 新最大值：旧的和乘 exp(1 − 3) = 0.1353 |
| 3 | 2 | 3 | 1.5032 | |
| 4 | 5 | 5 | 1.2034 | 新最大值：旧的和乘 exp(3 − 5) = 0.1353 |
| 5 | 4 | 5 | 1.5713 | |

最终的 softmax 是 0.0117、0.0861、0.0317、0.6364、0.2341，和标准三遍写法的最大差是 0。换成 10,000 个随机分数（float64），最大差 1.1 × 10⁻¹⁶。

注意力的输出 `Σ softmax_i · v_i` 也能用同样的办法边走边改正：

```
o_j = o_{j−1} · exp(m_{j−1} − m_j) + exp(x_j − m_j) · v_j，   最后输出 o_N / l_N
```

512 个位置、d = 64，与 `softmax(scores) @ V` 的最大差 5.8 × 10⁻¹⁵。

### 4.3 分块注意力

**FlashAttention**（Dao et al., 2022）把上面的想法推广到矩阵块：Q 切成 Br 行一块，K、V 切成 Bc 行一块；外层循环取一块 Q，内层循环逐块取 K、V，在片上缓存里算出 Br × Bc 的小分数块，用 online softmax 更新每一行的 (m, l, O)。完整的 S 和 P 从来不写回显存。

```python
for i0 in range(0, T, br):                       # 外层：一块 Q（Br 行）
    ...
    for j0 in range(0, T, bc):                   # 内层：一块 K、V（Bc 行）
        s = qi @ k[j0:j0 + bc].T * scale         # (Br, Bc) —— 只有这么大
        m_new = torch.maximum(m, s.max(dim=1).values)
        alpha = torch.exp(m - m_new)             # 旧状态的改正系数 exp(m_旧 − m_新)
        p = torch.exp(s - m_new[:, None])
        l = l * alpha + p.sum(dim=1)
        o = o * alpha[:, None] + p @ v[j0:j0 + bc]
        m = m_new
    out[i0:i0 + br] = o / l[:, None]             # 最后才除以 l
    lse[i0:i0 + br] = m + torch.log(l)           # 反向传播只需要存这个
```

因果 mask 还带来一个免费的加速：整块都在对角线右上方的 K、V 块直接跳过。`04_tiled_attention.py` 的验证（T = 512，d = 64，float64，因果）：

| 块大小 Br × Bc | 与朴素注意力的最大差 | 与 PyTorch SDPA 的最大差 | 最大的中间块 |
|---|---:|---:|---:|
| 64 × 64 | 1.0 × 10⁻¹⁵ | 6.7 × 10⁻¹⁶ | 4,096 |
| 128 × 32 | 7.8 × 10⁻¹⁶ | 6.7 × 10⁻¹⁶ | 4,096 |
| 32 × 128 | 1.2 × 10⁻¹⁵ | 4.4 × 10⁻¹⁶ | 4,096 |
| 100 × 70 | 7.8 × 10⁻¹⁶ | 7.8 × 10⁻¹⁶ | 7,000 |

朴素写法的中间张量是 2T² = 524,288 个数，分块后最大的中间块只有几千个。块大小怎么切都对，块大小只影响速度（在 GPU 上由 SRAM 容量决定）。

**反向传播也不存 P**：前向只为每一行存一个数 `lse = m + log l`（logsumexp），反向时按块重算 `P = exp(S − lse)`。脚本验证了前 64 行重算的 P 与 softmax(S) 的最大差 2.8 × 10⁻¹⁶。多算一次 QKᵀ，换来不存 T × T 的矩阵：FlashAttention 同时省了显存和时间，因为省下的读写比多出的计算贵得多。

要强调一点：**FlashAttention 算的是精确注意力，不是近似**。它是同一个数学结果的另一种计算顺序。后续的 FlashAttention-2 改进了并行划分，FlashAttention-3 针对 H100 的异步和 FP8 特性做了优化；它们都是 CUDA/Triton 写的 GPU 内核，我们的 Python 循环只是用来看清算法。

### 4.4 zero 怎么用它

zero 不自己写注意力内核，而是调用 PyTorch 的 `F.scaled_dot_product_attention`（SDPA）：

```python
out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=is_causal,
                                     enable_gqa=self.n_kv_heads != self.n_heads)
```

SDPA 会根据设备、dtype、形状自动挑后端：GPU 上 BF16/FP16 优先走 FlashAttention 内核，不满足条件就退到 memory-efficient 或 math 后端。在 CPU 上，本章用保存张量的钩子确认过：zero 的注意力保存的是 q、k、v、输出和每行一个 logsumexp，没有 T × T 的矩阵（第 2 节的激活公式就是这样数出来的）。GPU 上 `enable_gqa=True` 能否走 Flash 后端**尚未在 GPU 上验证**，是阶段 6 的第一项检查。

## 5. 数据并行：从 DDP 到 FSDP

### 5.1 DDP：梯度求平均

一张卡太慢，就用 8 张。最简单的办法是**数据并行（data parallelism）**：每张卡放一份完整的模型，各吃一份不同的数据，反向之后把所有卡的梯度求平均（all-reduce），每张卡用同一个平均梯度更新，参数永远保持一致。

它为什么对？一个 batch 的平均损失对参数的梯度，等于各个子 batch 梯度的平均，因为求导是线性的。所以"2 个进程各算 B 条再平均" = "1 个进程算 2B 条" = "1 个进程分两次各算 B 条再累加"（梯度累积）。

`06_ddp_by_hand.py` 真的起两个进程（`torch.distributed` 的 gloo 后端，CPU 就能跑），手写 DDP 的核心三行：

```python
for p in model.parameters():                          # DDP 的全部秘密：
    dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)     #   把所有进程的梯度加起来
    p.grad /= world                                   #   再除以进程数 = 平均
```

同一个 MLP、同样的数据，训练 20 步（float64）：

| 步 | 一次 2B 条 | 梯度累积 2 × B | 2 进程 all-reduce |
|---:|---:|---:|---:|
| 1 | 0.532621 | 0.532621 | 0.532621 |
| 2 | 0.413222 | 0.413222 | 0.413222 |
| 3 | 0.440613 | 0.440613 | 0.440613 |
| 20 | 0.511739 | 0.511739 | 0.511739 |

最终参数的最大差：梯度累积 6.9 × 10⁻¹⁷，2 进程 6.9 × 10⁻¹⁷，只差浮点求和顺序带来的舍入。（这个小例子只为对拍，loss 本身降得不多。）

真实的 PyTorch `DistributedDataParallel` 在此基础上做了两个优化：把梯度打包成"桶"（bucket）再通信，并且**边反向边通信**，后面的层梯度一算完就开始 all-reduce，和前面层的反向计算重叠。配合梯度累积时，只在最后一个 micro batch 同步（`no_sync()`），省掉中间的通信。

### 5.2 all-reduce 是怎么做的：环

8 张卡的梯度怎么求和？最直观的办法是都发给一张卡加完再发回去，但那张卡的带宽会成为瓶颈。**环形 all-reduce（ring all-reduce）**把卡排成一圈，每张卡的数据切成 N 块：

1. **reduce-scatter**：N − 1 轮，每轮每张卡把一块发给右邻居，邻居加到自己那块上。结束时每张卡各持有一块"完整的和"；
2. **all-gather**：再 N − 1 轮，把完整的块沿环传一圈，每张卡都拿到全部的和。

`06_ddp_by_hand.py` 在一个进程里模拟 4 张卡、每张 1,000,000 个梯度：结果与直接求和的最大差 1.8 × 10⁻¹⁵，**每张卡发出 1,500,000 个数 = 2·(N−1)/N × 1,000,000**。这个量几乎和卡数无关，这是环形算法的好处。主线模型的 FP32 梯度是 2.57 GiB，8 卡环形 all-reduce 每卡每步要发送 4.50 GiB。H100 机内 NVLink 的 GPU 间带宽是 900 GB/s（Nemotron-4 报告写明），这在机内不是瓶颈；跨机时带宽低一个数量级，通信和计算的重叠就变得关键。

### 5.3 FSDP / ZeRO：把"每卡一份"切开

DDP 的浪费在于：8 张卡各存一份完整的参数、梯度、优化器状态，一模一样。**ZeRO**（Rajbhandari et al., 2019）的想法是把它们切成 8 份，每张卡只管自己那 1/8：

| 方案 | 每卡存什么 | 主线模型每卡的静态显存（8 卡） |
|---|---|---:|
| DDP | 参数 + 梯度 + 优化器，全份；外加 DDP 通信桶 | 12.84 GiB |
| ZeRO-1 | 优化器状态切分 | 8.35 GiB |
| ZeRO-2 | 再切分梯度 | 3.53 GiB |
| ZeRO-3 / FSDP | 再切分参数 | 1.28 GiB |

（`zero/tools/memory_calc.py` 的输出。DDP 一行包含 PyTorch 默认设置下通信桶里多出的一份 FP32 梯度，2.57 GiB。）

切了参数，前向时怎么算？**FSDP（Fully Sharded Data Parallel）**（Zhao et al., 2023）在算到某一层之前，用 all-gather 把这一层的完整参数临时拼出来，算完就扔；反向时再拼一次，算出的梯度用 reduce-scatter 直接送到负责那一份的卡上。通信量约是 DDP 的 1.5 倍，换来参数、梯度、优化器状态都只占 1/N。

对主线模型来说，参数只有 10.3 GiB，DDP 完全放得下，所以 `configs/main/pretrain.toml` 默认 `parallel = "ddp"`。第 2 节的表也说明了：**卡住我们的不是参数，而是激活**，而 FSDP 不切激活（每张卡的激活只取决于它自己的 micro batch）。FSDP 在模型更大或上下文更长（第 15 章的 32K）时才真正需要。zero 里两种都实现了，用 `--set train.parallel=fsdp` 切换。

### 5.4 张量并行、流水线并行：知道就好

模型大到一张卡连一层都放不下时，还有两种切法：

- **张量并行（tensor parallelism，TP）**（Megatron-LM，Shoeybi et al., 2019）：把一个矩阵乘切到多张卡上，每层都要通信，只适合机内高速互联；
- **流水线并行（pipeline parallelism，PP）**：把不同的层放到不同的卡上，micro batch 像流水线一样依次流过，代价是"气泡"（有的卡在等）。

Llama 3 405B 把 TP、PP、上下文并行、FSDP 组合成"4D 并行"。对 0.7B 的主线模型，这些都用不上，所以本课不展开；CS336 第 7、8 讲有深入讲解。

## 6. MFU：离硬件峰值还有多远

**MFU**（PaLM 论文提出）的定义：

```
MFU = 实测吞吐（token/s）× 每 token FLOPs / 硬件峰值 FLOPs
```

分子只算"模型必须做的"运算（6N + 注意力项），不算激活检查点的重算。所以开了重算，MFU 会下降，这是有意的：MFU 衡量的是"训练有多快"，不是"GPU 有多忙"。后者叫 HFU（hardware FLOPs utilization），把重算也算进去。PaLM 540B 的 MFU 是 46.2%，HFU 是 57.8%。

MFU 达不到 100% 的原因有很多：逐元素运算（归一化、激活函数）是 memory-bound 的；通信没有完全和计算重叠；Python 开销和内核启动开销；小矩阵喂不饱 Tensor Core。

zero 的训练日志里直接报 MFU（`zero/train/trainer.py`，GPU 上按设备名查峰值表，CPU 上不报）。为了在 CPU 上也看一眼这个指标，`01_step_cost.py` 实测了本机单线程 FP32 矩阵乘的峰值（1024 × 1024，取最快的一次 48.8 GFLOPS），再读 tiny 预训练日志的吞吐（中位数 2,676 token/s）：

```
MFU = 2,676 token/s × 7.078 × 10⁶ FLOPs/token / 48.8 GFLOPS ≈ 38.8%
```

这是**极小配置演示**：分母是本机单线程 CPU 的实测值，每次运行都会因机器负载浮动（为视频缓存数据时的另一次运行，峰值测得 39.1 GFLOPS、吞吐中位数 2,790 token/s，换算出 50.5%），和 GPU 上的 MFU 不能直接比较。它只说明 MFU 的算法：日志里的 tok/s 乘以每 token FLOPs，除以峰值。

## 7. loss spike：训练中途突然爆炸

### 7.1 现象与原因

**loss spike（损失尖峰）**是指训练曲线平稳下降时，loss 突然跳高。有的几十步后自己恢复，有的一去不回（发散）。PaLM 540B 训练中出现过约 20 次尖峰，而且开着梯度裁剪；小一些的模型上没有出现。OLMo 团队观察到，梯度范数的尖峰往往出现在 loss 尖峰之前，模型越大尖峰越频繁。

目前公开报告里总结出的原因大致有几类：

- **数据**：OLMo 2 发现尖峰发生的 batch 里常有长串重复的 n-gram（比如一长串 `255, 255, 255, …`）；
- **数值尺度失控**：注意力 logits 越来越大（softmax 饱和成 one-hot）、输出层 logits 越来越大、嵌入的范数被权重衰减压得太小导致前几层梯度过大；
- **学习率太大、warmup 太短**：第 1 章"步子迈大就发散"的现象在大模型上的重演。

PaLM 还有一个耐人寻味的发现：把尖峰附近那几批数据拿出来，从更早的 checkpoint 开始训，并不会尖峰。**尖峰是"特定数据 + 特定参数状态"的组合**，不是单纯的坏数据。

### 7.2 应对办法

| 办法 | 做什么 | 采用情况 |
|---|---|---|
| 梯度裁剪（第 6 章） | 全局梯度范数超过 1.0 就等比例缩小 | 几乎所有报告都用，见文末 |
| warmup + 合适的峰值学习率（第 6 章） | 开头不要猛冲 | 所有报告都用 warmup；Llama 3 还提到"早期用较小的 batch 以提高稳定性" |
| QK-Norm（第 9 章） | 对 q、k 做 RMSNorm，注意力 logits 不随激活漂移变大 | Qwen3、Gemma 3、OLMo 2；主线模型已采用 |
| 数据侧过滤 | 去掉长串重复 n-gram 的文档 | OLMo 2（第 13 章） |
| 跳过坏 batch / 回滚 | 从尖峰前的 checkpoint 重启，跳过尖峰附近的数据；或者自动跳过梯度范数异常的那一步 | PaLM（回滚 ~100 步、跳过 200–500 批）；OLMo 2 的训练代码 `SkipStepAdamW`（梯度范数或 loss 超过滑动窗口 6 个标准差就不更新）。不足 3 个家族，放进前沿观察 |
| z-loss | 损失里加 `10⁻⁴ · log²Z`，让 softmax 的归一化项 Z 不要变大 | PaLM、OLMo 2；不足 3 个家族，放进前沿观察 |

好消息是：配方对了，大模型训练可以很平稳。Llama 3 405B 报告"只观察到少数尖峰，不需要人工干预"；DeepSeek-V3 报告"整个训练没有不可恢复的尖峰，也没有回滚"。主线模型已经带上了前三项（梯度裁剪、warmup、QK-Norm），数据侧的过滤在第 13 章做。

### 7.3 所以要能回滚

不管用了多少预防手段，最后的保险都是：**定期存 checkpoint，出事能退回去接着训**。下一节讲怎么让"接着训"真正做到分毫不差。

## 8. 断点续训：恢复"全部"状态

两周的训练里，机器一定会出事。Llama 3 405B 的训练在 54 天里被中断了 466 次，其中 419 次是意外，约 78% 是硬件问题。所以训练必须能随时存档、随时从存档接着跑。

"接着跑"的标准是：**续训后的每一步，和从没中断过逐位相同**。要做到这一点，光存模型权重远远不够。一次训练的全部状态是：

1. 模型参数；
2. 优化器状态（AdamW 的 m、v，以及步数 t，偏差修正要用）；
3. 学习率调度器的位置；
4. 数据读到了哪里；
5. 所有随机数发生器的状态（dropout、数据打乱等）；
6. 步数、已训练 token 数等计数器。

`07_resume.py` 先不中断地训练 30 步；再训练到第 17 步时"崩溃"（最近的 checkpoint 在第 15 步），在全新的对象里恢复、训到第 30 步：

| 步 | 不中断 | 续训 | |
|---:|---:|---:|---|
| 16 | 1.862050 | 1.862050 | 逐位相同 |
| 17 | 1.784434 | 1.784434 | 逐位相同 |
| 18 | 1.828543 | 1.828543 | 逐位相同 |
| 30 | 1.769457 | 1.769457 | 逐位相同 |

全部 15 步逐位相同。然后每次故意"忘掉"一项状态，看续训 15 步里 loss 与不中断的最大差：

| 少恢复了什么 | 最大差 |
|---|---:|
| 优化器状态（m、v、t 从零开始） | 1.13 × 10⁻² |
| 学习率调度（从 warmup 重新开始） | 1.46 × 10⁻¹ |
| 数据位置（重新抽到开头那几批） | 1.25 × 10⁻¹ |
| 全局随机数（dropout 掩码不同） | 1.61 × 10⁻² |

每一项都会让训练偏离原来的轨迹。偏离不一定让模型变差，但它让实验不可复现：你再也分不清一个现象是配方造成的还是续训造成的。

生产级还要多考虑一件事：**写 checkpoint 的中途被杀掉**（抢占式实例常有）。如果直接覆盖旧文件，就会留下一个写了一半的坏存档。zero 的做法是先写到临时目录，全部写完再用 `os.replace` 原子改名，最后更新 `latest` 指针；中途被杀，`latest` 仍然指向上一个完整的 checkpoint。

## 9. 小结

| 问题 | 办法 | 本章的证据 |
|---|---|---|
| 训一次要多少算力和时间 | 每 token 6N + 注意力项；时间 = 总 FLOPs /（峰值 × MFU） | 主线 500B token：3.48 × 10²¹ FLOPs，8×H100 @ MFU 0.4 约 12.7 天、$6,102 |
| 显存装不下 | 16 字节/参数 + 激活；梯度累积、激活检查点 | 实测 16 字节/参数；检查点把激活降到 21%；主线 micro batch 8 估算超 80 GB |
| FP32 太慢、FP16 会溢出 | BF16 算、FP32 存 | BF16 累加卡在 4、权重更新丢失；FP16 70,000 → inf |
| 注意力读写 T × T 矩阵 | 分块 + online softmax，反向存 logsumexp 重算 | 与朴素注意力、SDPA 最大差约 10⁻¹⁵；主线一层省 8 GiB |
| 一张卡太慢 | DDP：梯度 all-reduce 求平均 | 2 进程 = 梯度累积 = 大 batch，参数差 6.9 × 10⁻¹⁷ |
| 每卡存一份太浪费 | ZeRO / FSDP 切分状态 | 主线静态显存每卡 12.84 → 1.28 GiB |
| loss spike | 裁剪、warmup、QK-Norm、数据过滤、回滚 | 公开报告（PaLM、OLMo 2、Llama 3、DeepSeek-V3） |
| 机器会坏 | 保存并恢复全部状态，原子写入 | 续训 15 步逐位相同；少任何一项都会偏离 |

---

## 从极简到生产级

本章的极简代码只讲原理；主线模型的真实训练在 `zero/train/` 里。对应关系：

| 极简版 | 生产级（`zero/`） | 多做了什么、为什么 |
|---|---|---|
| `01_step_cost.py` 的公式 | `zero/model.py` 的 `estimate_flops_per_token`；`zero/tools/estimate_cost.py` | 同一个公式；估算工具带 GPU 峰值表、单价、卡数，输出卡时和费用。H100 峰值 989.5 TFLOPS 在代码里标"待核实"，Nemotron-4 报告写的是"989 teraFLOP/s（bfloat16，不含稀疏）"，与之吻合 |
| `05_memory.py` 数保存的张量 | **本章新增** `zero/tools/memory_calc.py`（`estimate_memory`、`max_micro_batch`，CLI） | 按 zero 的 `Block` 逐项写出每层每 token 的激活公式，支持 DDP / ZeRO-1 / ZeRO-2 / FSDP、激活检查点、BF16 / FP32；`tests/test_memory_calc.py` 保证公式与 `saved_tensors_hooks` 实测**逐字节相等** |
| 手写 `torch.autocast` 实验 | `zero/train/trainer.py` 的 `autocast_context`：CUDA 上 BF16 autocast，参数和优化器状态保持 FP32；`zero/model.py` 的 RMSNorm、交叉熵先转 FP32 再算 | 精度敏感的运算单独保护；CPU 上默认 FP32 |
| `04_tiled_attention.py` 的 Python 循环 | `zero/model.py` 的 `Attention`：`F.scaled_dot_product_attention(..., enable_gqa=...)` | 不自己写内核，交给 PyTorch 选 FlashAttention / memory-efficient 后端。**尚未在 GPU 上验证**实际走的是哪个后端 |
| 手写梯度累积 | `Trainer.train`：`grad_accum_steps` 次前向 + 反向，loss 除以累积步数；DDP 下前几个 micro batch 用 `no_sync()` | 省掉中间的梯度通信 |
| `06_ddp_by_hand.py` 手写 all-reduce | `zero/train/dist.py`：`init_distributed`（读 torchrun 的环境变量，GPU 用 NCCL、CPU 用 gloo）、`wrap_model`（DDP 或 FSDP2 `fully_shard`，BF16 参数 + FP32 梯度规约） | 桶化、通信与反向重叠由 PyTorch DDP 完成；FSDP2 路径**尚未在 GPU 上验证** |
| 数据按进程分片 | `zero/data/loader.py` 的 `PackedDataLoader`：第 g 个全局样本分给 rank g % world_size | 保证"2 卡各 B 条"与"1 卡 2B 条"看到同一批数据，DDP 才能对拍 |
| 第 6 章的 `clip_grad_norm_` | `Trainer.train` 每步裁剪，梯度范数写进日志 | 梯度范数是发现 spike 的第一信号 |
| `01_step_cost.py` 的 MFU | `Trainer._peak_flops` + 日志里的 `tok_per_s`、`mfu` | 按设备名查峰值表；CPU 上不报 |
| `07_resume.py` 的 `state_dict` | `zero/train/checkpoint.py`：`save_checkpoint`（model / optim / meta.json（步数、token 数、调度器、配置）/ 每个 rank 的数据位置 + 全部 RNG）；先写 `.tmp_step_xxx` 再 `os.replace`，最后原子更新 `latest`；`keep_last` 自动清理 | 多进程时每个 rank 各存一份加载器与 RNG 状态；卡数变了会拒绝精确续训（数据位置对不上） |
| — | 激活检查点、z-loss、跳过坏 batch | **zero 目前没有实现**。显存计算器给出了"如果开激活检查点"的估算；是否需要由阶段 6 的实测决定 |

**对拍（都在 CPU 上通过）**：

```bash
uv run pytest tests/test_train_resume.py tests/test_ddp_cpu.py tests/test_memory_calc.py   # 11 passed
```

- `tests/test_train_resume.py`：训 8 步 vs 训到第 5 步"崩溃"、从第 4 步的 checkpoint 续训，第 5–8 步的 loss、验证 loss、最终参数、token 计数**逐位相同**；另测固定种子可复现、checkpoint 的原子布局与清理；
- `tests/test_ddp_cpu.py`：gloo 后端起 2 个进程跑 DDP（micro batch 2、累积 2），与单进程 micro batch 4 逐步对拍，loss 差在 10⁻⁵ 以内，并检查多进程 checkpoint 里每个 rank 各有一份状态；
- `tests/test_memory_calc.py`：手算的参数量（主线 689,518,848）、每层激活字节数（tiny BF16 8,368 / FP32 12,336）、各种切分方式的静态显存，以及公式与真实保存张量的逐字节对拍（含激活检查点）。

---

## 主线进度

### 极小配置演示（CPU，`configs/tiny`，约 1.3M 参数）

> 以下全部是**极小配置演示**：只说明代码能跑通、各项机制按预期生效，不代表主线模型的任何结果。

**单进程预训练**（为了不和其他章节共用输出目录，用 `--set` 换了 `out_dir`）：

```bash
uv run python -m zero.train.pretrain --config configs/tiny/pretrain.toml --set train.out_dir=out/ch14/pretrain
```

```
模型参数 1.31M（非 embedding 0.79M），每步 2048 token，共 200 步，设备 cpu，world_size=1
step      1/200 | loss 7.6507 | lr 1.00e-04 | gnorm 0.98 | 1,774 tok/s
step     25/200 | loss 6.3844 | lr 2.50e-03 | gnorm 0.38 | 2,396 tok/s
step     50/200 | loss 6.3938 | lr 3.00e-03 | gnorm 1.75 | 2,853 tok/s
step     75/200 | loss 6.2778 | lr 3.00e-03 | gnorm 0.58 | 2,790 tok/s
step    100/200 | loss 6.1065 | lr 3.00e-03 | gnorm 0.38 | 2,909 tok/s | val 6.0048
checkpoint 已保存：out/ch14/pretrain/ckpt/step_00000100
step    125/200 | loss 5.9836 | lr 3.00e-03 | gnorm 0.56 | 2,465 tok/s
step    150/200 | loss 5.8692 | lr 3.00e-03 | gnorm 0.33 | 2,835 tok/s
step    175/200 | loss 5.6566 | lr 1.99e-03 | gnorm 0.40 | 2,561 tok/s
step    200/200 | loss 5.4736 | lr 3.00e-04 | gnorm 0.38 | 2,528 tok/s | val 5.7116
checkpoint 已保存：out/ch14/pretrain/ckpt/step_00000200
```

单线程 CPU，200 步用了 155.8 秒（日志里的 `elapsed_s`；机器同时被其他任务占用，吞吐会浮动）。CPU 上 zero 不报 MFU；按第 6 节的算法用本机实测峰值换算约 38.8%（`01_step_cost.py`）。

**断点续训（真实的命令行）**：把上面的输出目录复制一份，删掉第 200 步的 checkpoint，让 `latest` 指回第 100 步，再执行同一条命令：

```
从 out/ch14/resume/ckpt/step_00000100 续训（step 100）
step    125/200 | loss 5.9836 | ...
step    150/200 | loss 5.8692 | ...
step    175/200 | loss 5.6566 | ...
step    200/200 | loss 5.4736 | ... | val 5.7116
```

逐项比较两次运行的 `log.jsonl`：第 125、150、175、200 步的 loss（5.9835896492004395、5.869236469268799、5.6566314697265625、5.473620414733887）与验证 loss 全部逐位相同，第 200 步的 `model.pt` 每个张量都 `torch.equal`。

**2 进程 DDP（torchrun，CPU gloo）**：

```bash
uv run torchrun --standalone --nproc_per_node 2 -m zero.train.pretrain --config configs/tiny/pretrain.toml \
    --set train.out_dir=out/ch14/ddp2 --set train.micro_batch_size=8
```

```
模型参数 1.31M（非 embedding 0.79M），每步 2048 token，共 200 步，设备 cpu，world_size=2
step      1/200 | loss 7.6537 | lr 1.00e-04 | gnorm 0.97 | 3,178 tok/s
step    100/200 | loss 5.9307 | lr 3.00e-03 | gnorm 0.38 | 3,797 tok/s | val 5.9674
step    200/200 | loss 5.4258 | lr 3.00e-04 | gnorm 0.37 | 3,729 tok/s | val 5.5986
```

每卡 micro batch 8，两卡合起来每步仍是 2048 token；checkpoint 里有 `rank0.pt` 和 `rank1.pt`。两卡的吞吐约 3,700 token/s，单卡约 2,500–2,900（同一台负载浮动的机器，只能看量级）。注意这次的 loss 和单进程**不应该**逐位相同：tiny 配置是三个来源的混合（`MixtureLoader`），每个 rank 各自按 `(seed, rank)` 抽签选来源，2 卡看到的数据和 1 卡不是同一批。严格的 DDP 对拍用的是单来源数据，由 `tests/test_ddp_cpu.py` 保证。

**成本与显存估算**（公式估算，**尚未在 GPU 上验证**）：

```bash
uv run python -m zero.tools.estimate_cost --config configs/main/pretrain.toml --tokens 500B --gpu h100-sxm
uv run python -m zero.tools.memory_calc configs/main/pretrain.toml
```

| 项 | 估算 |
|---|---|
| 预训练 500B token，8×H100，MFU 0.4 | 2,440.6 卡时，12.71 天，$6,102（超过 GOAL.md 3.4 给预训练的 ~$5K，需闸门 1 在 token 数、实测 MFU、租价之间定稿） |
| 显存：micro batch 8 × 4096，DDP，eager | 每卡约 113.8 GiB，**超过 80 GB**；micro batch 4 约 64.0 GiB，最大约 5 |
| 建议 | 保持每步 524,288 token，改成 micro batch 4 × 累积 4；或开激活检查点（需要先在 zero 里实现）。以阶段 6 的实测为准 |

### 第二步的第一件事：≤ $50 的 GPU 验证运行（GOAL.md 第 10 节阶段 6）

在花大钱预训练之前，先用 8×H100 跑约 1.5 小时（约 $30，上限 $50），把本章所有"尚未在 GPU 上验证"的路径逐项验证掉。完整命令见 [`runs/RUNBOOK.md`](../../runs/RUNBOOK.md) 第 2 节，和本章的对应关系是：

| 验证项 | 验证什么 | 对应本章 |
|---|---|---|
| 单卡 BF16 + SDPA | 注意力真的走 FlashAttention 内核；`enable_gqa=True` 会不会让它退回别的后端 | 第 4 节 |
| 8 卡 DDP 200 步 | NCCL 通信、loss 正常下降、各卡显存均衡 | 第 5 节 |
| 中途 kill 再续训 | 多卡 + BF16 下续训后数据顺序完全一致、loss 一致（BF16 允许 10⁻³ 级差异） | 第 8 节 |
| FSDP2 | 与 DDP 前 50 步 loss 一致，checkpoint 能被单卡读回 | 第 5.3 节 |
| torch.compile 开关 | 两者 loss 一致，记录提速 | 第 6 节 |
| 实测 MFU | 用日志里的 `mfu` 重跑 `estimate_cost --mfu <实测>`，更新全部预算 | 第 1、6 节 |
| 试探 micro batch | 找到不 OOM 的最大 micro batch，和 `memory_calc` 的预测（约 5）对比，校准公式 | 第 2 节 |

### 待 GPU 训练后补充

- 阶段 6 的实测：MFU、吞吐、每卡显存峰值，与本章估算的对比；
- 主线预训练的真实训练曲线（loss、梯度范数、学习率）、实际花费（记入 `runs/ledger.md`）；
- 训练中遇到的 loss spike、中断与续训记录，以及怎么处理的；
- 如果估算与实测差距大，写清楚差在哪里、为什么。

---

## 前沿观察

**FP8 训练**：H100 起 Tensor Core 支持 8 位浮点矩阵乘，理论吞吐是 BF16 的两倍。DeepSeek-V3 用"细粒度量化（激活 1 × 128、权重 128 × 128 一组缩放）+ 高精度累加"的 FP8 框架训练，报告与 BF16 基线的相对 loss 误差小于 0.25%，同时主权重、梯度仍保留 FP32，AdamW 的矩用 BF16 存。nanochat 的代码里也有 FP8 选项；GOAL.md 3.1 提到的 Puro-2B 同样用了 FP8。FP8 对缩放策略和数值异常值很敏感，实现复杂度远高于 BF16。它是否已满足本课"3 个独立家族明确采用"的规则，由第 12 章核实；本章和主线默认 BF16。

**z-loss 与自动跳过坏步**：z-loss（PaLM 提出，OLMo 2 沿用，系数 10⁻⁴ 量级）约束 softmax 的归一化项；OLMo 2 的训练代码用 `SkipStepAdamW` 在梯度范数或 loss 超过滑动窗口 6 个标准差时跳过这一步更新，PaLM 则是人工回滚并跳过数据。这些做法有效，但本章核实到的明确采用方不足 3 个家族，所以没有进主线默认配置。

---

## 采用方与来源

| 技术 | 采用方（至少 3 个家族） | 来源 |
|---|---|---|
| BF16 混合精度（BF16 计算 + FP32 主权重 / 梯度累加） | Llama 3（"BF16 MFU"；多个 micro batch 的梯度用 FP32 累加、FSDP 的 reduce-scatter 用 FP32）；Nemotron-4 340B（按 bfloat16 峰值 989 TFLOPS 计算 MFU）；DeepSeek-V3（以 BF16 训练为对照基线，FP8 框架中主权重和梯度保留 FP32）；OLMo 2（官方训练脚本 `param_dtype=bfloat16`、`reduce_dtype=float32`） | [Llama 3 §3.3.2 与表 4](https://arxiv.org/abs/2407.21783)、[Nemotron-4 340B §2.3](https://arxiv.org/abs/2406.11704)、[DeepSeek-V3 §3.3 与附录 B.1](https://arxiv.org/abs/2412.19437)、[OLMo-core OLMo-2-0325-32B 训练脚本](https://github.com/allenai/OLMo-core/blob/main/src/scripts/official/OLMo2/OLMo-2-0325-32B-train.py) |
| FlashAttention | 行业标准（GOAL.md 2.1 的 B 类）：PyTorch 的 `scaled_dot_product_attention` 内置 FlashAttention 后端；OLMo 的训练代码 OLMo-core 支持 FlashAttention-2 后端，OLMo 3 的模型配置默认 `attn_backend=flash_2`（OLMo 2 报告 §3.3.3 也提到了 Flash Attention 库）；nanochat 训练用 FlashAttention-3（不可用时退回 SDPA） | [FlashAttention](https://arxiv.org/abs/2205.14135)、[FlashAttention-2](https://arxiv.org/abs/2307.08691)、[FlashAttention-3](https://arxiv.org/abs/2407.08608)、[OLMo-core `transformer/config.py`](https://github.com/allenai/OLMo-core/blob/main/src/olmo_core/nn/transformer/config.py)、[nanochat `flash_attention.py`](https://github.com/karpathy/nanochat/blob/master/nanochat/flash_attention.py) |
| 数据并行 + 状态切分（FSDP / ZeRO） | Llama 3（FSDP 切分优化器状态和梯度）；DeepSeek-V3（ZeRO-1 数据并行）；Gemma 3（优化器状态用 ZeRO-3 的实现切分）；Nemotron-4（分布式优化器，把优化器状态切到数据并行副本上）；OLMo 2（官方脚本用 HSDP，即分组的 FSDP） | [Llama 3 §3.3.2](https://arxiv.org/abs/2407.21783)、[DeepSeek-V3 §3.2](https://arxiv.org/abs/2412.19437)、[Gemma 3 §2.4](https://arxiv.org/abs/2503.19786)、[Nemotron-4 340B §2.3](https://arxiv.org/abs/2406.11704)、[OLMo-core 训练脚本](https://github.com/allenai/OLMo-core/blob/main/src/scripts/official/OLMo2/OLMo-2-0325-32B-train.py) |
| 梯度裁剪（范数 1.0） | Llama 2、DeepSeek-V3、OLMo 2、SmolLM3（第 6 章已核实）；PaLM（全局范数裁剪 1.0） | 见[第 6 章"采用方与来源"](../06-training-stability/README.md#采用方与来源)；[PaLM §5](https://arxiv.org/abs/2204.02311) |
| QK-Norm（训练稳定性） | Qwen3（"to ensure stable training"）；Gemma 3；OLMo 2（"avoids attention logits being too large, which can lead to training loss divergence"） | 见[第 9 章"采用方与来源"](../09-modern-transformer/README.md)；[Qwen3 §2](https://arxiv.org/abs/2505.09388)、[Gemma 3 §2](https://arxiv.org/abs/2503.19786)、[OLMo 2 §2.1、§3.3.2](https://arxiv.org/abs/2501.00656) |
| 激活检查点（重算） | PaLM 540B（rematerialization）；OLMo 2（官方 32B 脚本开 full activation checkpointing）；DeepSeek-V3（反向时重算所有 RMSNorm 和 MLA 上投影）。反例：Llama 3 报告优化后 8K 序列"不需要激活检查点" | [PaLM §4.1](https://arxiv.org/abs/2204.02311)、[OLMo-core 训练脚本](https://github.com/allenai/OLMo-core/blob/main/src/scripts/official/OLMo2/OLMo-2-0325-32B-train.py)、[DeepSeek-V3 §3.2.3](https://arxiv.org/abs/2412.19437)、[Llama 3 §3.3.2](https://arxiv.org/abs/2407.21783) |
| MFU 作为效率指标 | PaLM（提出，46.2%）；Llama 3 405B（38–43%）；Nemotron-4 340B（41.0–42.4%） | [PaLM §4.1 与附录 B](https://arxiv.org/abs/2204.02311)、[Llama 3 表 4](https://arxiv.org/abs/2407.21783)、[Nemotron-4 表 2](https://arxiv.org/abs/2406.11704) |

OLMo-core 训练脚本读自 commit `718d08f`（2026-09-26）。待核实：Qwen3 技术报告没有写训练精度和并行方式；SmolLM3 博客只写了 nanotron 框架和梯度裁剪 1，没有写精度与并行细节。

---

## 引导问题

带着这些问题去问 Claude Code，直到你能用自己的话讲清楚：

1. 本章说"FlashAttention 多算了一次 QKᵀ，却更快"。用"memory-bound"解释：为什么在 GPU 上，多算有时比多读写便宜？什么样的运算属于 compute-bound？
2. online softmax 的改正系数 `exp(m_旧 − m_新)` 为什么总是 ≤ 1？如果不减最大值、直接累加 `exp(x_i)`，BF16 下会出什么问题？（提示：`02_precision.py` 的最大值一栏。）
3. 激活检查点会让 MFU 下降，却可能让训练更快。这两件事怎么同时成立？（提示：PaLM 为什么开了重算？和 micro batch 有什么关系？）
4. DDP 对梯度求平均，而不是求和。如果改成求和、学习率不变，训练会怎样？换成 8 张卡时呢？
5. 第 8 节说"卡数变了，zero 拒绝精确续训"。为什么？如果确实要从 8 卡换到 4 卡接着训，最少要放弃什么？
6. PaLM 发现"尖峰附近那几批数据从更早的 checkpoint 开始训不会尖峰"。这说明尖峰的原因是什么？对"跳过坏数据"这个办法意味着什么？

## 动手任务

每个任务都要真的运行代码、看到结果。

**任务 1（基础）**：用 `zero/tools/memory_calc.py` 回答：主线模型如果把序列长度从 4096 改成 32,768（第 15 章的长上下文），micro batch 1 在 DDP 下每卡要多少显存？开激活检查点呢？FSDP 呢？哪种组合能放进 80 GB？

**任务 2（核心）**：在 `04_tiled_attention.py` 里加一个"反向传播"：用前向存下的 `lse` 按块重算 P，算出 `dV = Pᵀ · dO`，和 PyTorch 自动求导（对朴素注意力 `backward`）得到的 dV 比较，打印最大差。（进阶：再推导 dQ、dK，需要用到 `D_i = Σ_j dO_ij · O_ij`。）

**任务 3（挑战）**：`07_resume.py` 目前在"每步之后"存 checkpoint。改成用梯度累积（每步 2 个 micro batch），并在**两个 micro batch 之间**崩溃。要逐位续训，checkpoint 里还需要多存什么？为什么生产代码（包括 zero）一般只在完整的一步之后存 checkpoint？

---

## 想深入：CS336

本章对应 [CS336（Spring 2026）](https://cs336.stanford.edu/) 的系统部分，也是 CS336 比本课深入得多的地方：

- **第 5 讲 GPU**：GPU 的内存层级（HBM、SRAM）、算术强度与 roofline 模型，为什么注意力是 memory-bound；
- **第 6 讲 Kernel 与 Triton**：怎么用 Triton 写融合内核，本章的分块循环在那里会变成真正的 GPU 程序；
- **第 7、8 讲 并行**：DDP、ZeRO/FSDP、张量并行、流水线并行、序列并行的通信代价与组合方式，本章第 5.4 节一笔带过的内容在这两讲里展开；
- **作业 2（Systems）**：用 Triton 实现 FlashAttention-2 的前向和反向、手写 DDP（含梯度分桶与通信重叠）、实现优化器状态切分，并做性能测试。本章的 `04_tiled_attention.py` 和 `06_ddp_by_hand.py` 是它的 CPU 缩小版。

课程页有每讲的讲义和录像。

---

## 本章参考文献

- Micikevicius et al. *Mixed Precision Training*，2017：<https://arxiv.org/abs/1710.03740>
- Kalamkar et al. *A Study of BFLOAT16 for Deep Learning Training*，2019：<https://arxiv.org/abs/1905.12322>
- Milakov & Gimelshein. *Online normalizer calculation for softmax*，2018：<https://arxiv.org/abs/1805.02867>
- Dao et al. *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*，2022：<https://arxiv.org/abs/2205.14135>
- Dao. *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning*，2023：<https://arxiv.org/abs/2307.08691>
- Shah et al. *FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision*，2024：<https://arxiv.org/abs/2407.08608>
- Chen et al. *Training Deep Nets with Sublinear Memory Cost*（激活检查点），2016：<https://arxiv.org/abs/1604.06174>
- Korthikanti et al. *Reducing Activation Recomputation in Large Transformer Models*，2022：<https://arxiv.org/abs/2205.05198>
- Rajbhandari et al. *ZeRO: Memory Optimizations Toward Training Trillion Parameter Models*，2019：<https://arxiv.org/abs/1910.02054>
- Zhao et al. *PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel*，2023：<https://arxiv.org/abs/2304.11277>
- Shoeybi et al. *Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism*，2019：<https://arxiv.org/abs/1909.08053>
- Chowdhery et al. *PaLM: Scaling Language Modeling with Pathways*（MFU 的定义、loss spike 的回滚与跳过、z-loss、逐位可复现），2022：<https://arxiv.org/abs/2204.02311>
- Llama Team. *The Llama 3 Herd of Models*（4D 并行、BF16 MFU、中断统计），2024：<https://arxiv.org/abs/2407.21783>
- NVIDIA. *Nemotron-4 340B Technical Report*（H100 BF16 峰值、MFU、分布式优化器），2024：<https://arxiv.org/abs/2406.11704>
- DeepSeek-AI. *DeepSeek-V3 Technical Report*（FP8 训练、ZeRO-1、无回滚），2024：<https://arxiv.org/abs/2412.19437>
- OLMo Team. *2 OLMo 2 Furious*（第 3 节：预训练稳定性），2024：<https://arxiv.org/abs/2501.00656>；训练代码 OLMo-core：<https://github.com/allenai/OLMo-core>
- Gemma Team. *Gemma 3 Technical Report*，2025：<https://arxiv.org/abs/2503.19786>
- Wortsman et al. *Small-scale proxies for large-scale Transformer training instabilities*，2023：<https://arxiv.org/abs/2309.14322>
- Karpathy. nanochat（单节点、可读的训练代码，本课 zero 的设计参考）：<https://github.com/karpathy/nanochat>
- CS336（Spring 2026）：<https://cs336.stanford.edu/>

**下一章**：预训练把 500B token 的"通识"灌进了模型，学习率一直停在稳定段。第 15 章做两件事：用高质量数据把学习率退火下来（中期训练），再把上下文从 4K 扩展到 32K，得到主线的 Base 模型，并在闸门 2 检查它是否达到预期。
