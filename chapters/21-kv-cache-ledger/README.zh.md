# 第 21 章：KV cache 的账本 —— 长上下文为什么贵，每个 token 要存多少

[English](README.md) · **中文**

> **目标**：读完这一章，你能说清楚长上下文在 prefill 和 decode 两个阶段分别贵在哪里。你能按层算出任意一个开源模型的 KV cache 有多大（依据它的 `config.json`），包括四种层：全注意力、滑动窗口、混合线性注意力、MLA。你还能讲清楚 MLA 怎样把 K、V 压成一个潜向量，以及推理时怎样"吸收"掉还原 K、V 的步骤。

📺 **本章视频**：还没有发布。要在本机渲染，运行 `bash chapters/21-kv-cache-ledger/video/build.sh`。
🧪 **本章自检**：学完后，在 Claude Code 里输入 `/ch21-kv-cache`。

---

第四部分结束时，我们有了一个会调用工具的主线模型。第五部分换一个视角。这一部分不再训练主线模型，而是看过去两年开源模型的**架构**（architecture）往哪里演进。这些演进几乎都指向同一个方向：**更长的上下文、更小的 KV cache**。

第 10 章第一次算过这笔账。主线模型每个 token 要缓存 112 KiB，一条 32K 的对话就要 3.5 GiB，比模型权重（1.28 GiB）还大。GQA 用 8 个 KV 头代替 16 个，省了一半。

这一章回答两个问题：**长上下文到底贵在哪？KV cache 还能怎样压缩？** 我们先把"贵"拆成两笔账：prefill 的算力和 decode 的显存带宽。然后把第 10 章的公式升级成按层记账的"账本"（ledger），用它去算几个最新的开源模型。最后讲 MQA → GQA → MLA 这条压缩路线，并在 CPU 上把五种注意力放进同一个小模型里比较。

本章代码：

```bash
uv run python chapters/21-kv-cache-ledger/code/01_kv_ledger.py          # ledger: main-line model + 12 public models
uv run python chapters/21-kv-cache-ledger/code/02_prefill_decode.py     # prefill compute / decode bandwidth / number of conversations
uv run python chapters/21-kv-cache-ledger/code/03_mla.py                # minimal MLA: parity check of the absorbed and explicit paths
uv run python chapters/21-kv-cache-ledger/code/04_attention_variants.py # MHA/GQA/MQA/MLA comparison (first run: about half an hour)
```

## 1. 长上下文贵在哪：两个阶段，两种瓶颈

第 10 章讲过，有了 KV cache，推理分成两个阶段。**prefill** 把整段提示词一次输入模型。**decode** 之后每步只输入一个新 token。长上下文在这两个阶段里贵的原因完全不同。

### 1.1 prefill：算力按 T² 增长

prefill 的前向运算量分两部分（`02_prefill_decode.py` 的 `prefill_flops`）：

```python
linear = 2 * n_matmul * T                  # matmul: one multiply-add per parameter per token = 2 operations
attn = 2 * c.n_layers * c.q_dim * T * T    # QKᵀ and AV: 4·q_dim per pair (i, j); the causal mask leaves only T²/2 pairs
```

第一项随 T 线性增长，第二项随 T² 增长。代入主线模型：参与矩阵乘的参数 N = 689.4M，q_dim = 16 × 128 = 2048。按 H100 SXM 的稠密 BF16 峰值 989.5 TFLOPS 算理论耗时：

| 提示词长度 T | 矩阵乘部分（次） | 注意力部分（次） | 注意力占比 | H100 理论下限 |
|---:|---:|---:|---:|---:|
| 1,024 | 1.41 × 10¹² | 1.20 × 10¹¹ | 7.8% | 1.5 ms |
| 4,096 | 5.65 × 10¹² | 1.92 × 10¹² | 25.4% | 7.7 ms |
| 32,768 | 4.52 × 10¹³ | 1.23 × 10¹⁴ | **73.2%** | 170 ms |
| 131,072 | 1.81 × 10¹⁴ | 1.97 × 10¹⁵ | **91.6%** | 2.2 s |

在 4K 以内，注意力只占算力的一小部分。到 128K，九成以上的算力花在注意力上。所以上下文变长时，首 token 延迟（time to first token）会急剧变差。第 22、23 章的滑动窗口和线性注意力就是为了解决这个问题。**KV cache 本身不改变 prefill 的算力。**这一章主要关心下一节的 decode。

### 1.2 decode：受显存带宽限制

decode 每一步只算一个 token。但为了这一个 token，GPU 要把**全部权重读一遍**，还要把这条对话的**整个 KV cache 读一遍**。设 batch 里有 B 条对话，每条已有 T 个 token（`decode_step`）：

```python
flops = B * (2 * n_matmul + 4 * c.n_layers * c.q_dim * T)   # operations to do
w_bytes = n_params * BF16                                     # bytes to read: weights (shared by all B)
kv = B * T * kv_per_token                                     # bytes to read: the KV cache of each conversation
t = max(flops / PEAK_FLOPS, (w_bytes + kv) / HBM_BW)          # theoretical minimum time of one step
```

运算次数 ÷ 读取字节数叫作**算术强度**（arithmetic intensity）。H100 SXM 每秒能算 989.5 万亿次，能读 3.35 TB。两者之比约为 295 次/字节，叫作**脊点**（ridge point）。算术强度低于脊点时，GPU 在等数据，这叫**带宽受限**（memory-bound）。主线模型的情况如下：

| 上下文 T | batch B | 读权重 | 读 KV cache | 算术强度 | 一步理论耗时 | 吞吐（token/s） | 80 GB 装得下？ |
|---:|---:|---:|---:|---:|---:|---:|---|
| 4,096 | 1 | 1.28 GiB | 0.44 GiB | 1.3 | 0.55 ms | 1,812 | 是 |
| 4,096 | 16 | 1.28 GiB | 7.00 GiB | 4.2 | 2.66 ms | 6,026 | 是 |
| 4,096 | 64 | 1.28 GiB | 28.00 GiB | 4.7 | 9.39 ms | 6,819 | 是 |
| 32,768 | 1 | 1.28 GiB | 3.50 GiB | 1.7 | 1.53 ms | 652 | 是 |
| 32,768 | 16 | 1.28 GiB | 56.00 GiB | 2.3 | 18.36 ms | 871 | 是 |
| 32,768 | 64 | 1.28 GiB | 224.00 GiB | 2.4 | 72.21 ms | 886 | **否** |

算术强度只有 1–5，比 295 低两个数量级：decode 完全是带宽受限的。

第 10 章说过："把很多请求拼成一批，读一遍权重就能服务几十个请求。"这个方法在短上下文下有效：4K 时 batch 从 1 增大到 16，吞吐涨了 3.3 倍。**但在长上下文下，这个方法失效了。**32K 时，KV cache（每条 3.5 GiB）远大于权重（1.28 GiB）。每条对话各自读取自己的 KV cache，拼批省不掉这部分读取。batch 从 1 增大到 16，吞吐只涨了 1.3 倍。

所以在长上下文下，decode 的速度几乎只由每个 token 的 KV cache 大小决定。

### 1.3 显存：能同时服务几条对话

还有一个更硬的约束：数据必须放得进显存。用 80 GB 减去权重，剩下的全部留给 KV cache。这里不计激活和碎片，所以结果偏乐观。表中是能同时服务的对话条数：

| 上下文 | GQA（主线，112 KiB/token） | 若是 MHA（224 KiB） | 若换 MLA 512+64（31.5 KiB） |
|---:|---:|---:|---:|
| 4,096 | 167 | 83 | 595 |
| 32,768 | 20 | 10 | 74 |
| 131,072 | 5 | 2 | 18 |

每个 token 的 KV cache 小一半，同一张卡能同时服务的对话就多一倍，decode 时要读的字节也少一半。这就是整个第五部分的动机。

## 2. 账本：按层记账

第 10 章的公式假设每层都一样：

```
KV cache 字节数 = 2 × 层数 × KV 头数 × head_dim × 序列长度 × 每个数的字节数（× batch）
```

可是 2025–2026 年的开源模型，层和层已经不一样了。有的层只看最近 128 个 token（滑动窗口）。有的层根本不存 K、V（线性注意力）。有的层存的不是 K、V，而是一个压缩过的潜向量（MLA）。所以账要按层记：

| 层类型 | 每层每个位置存什么 | 随序列长度怎样增长 | 本课在哪里讲 |
|---|---|---|---|
| 全注意力（MHA / GQA / MQA） | K、V 各 `KV 头数 × head_dim` 个数 | 线性增长 | 第 8、10 章，本章第 4 节 |
| MLA | 潜向量 `kv_lora_rank` + 共享 RoPE key `qk_rope_head_dim` | 线性增长，但每个位置小得多 | 本章第 4 节 |
| 滑动窗口 | 同全注意力 | 最多存 `window` 个位置，之后不再增长 | 第 22 章 |
| 线性注意力（Gated DeltaNet、KDA 等） | 不存 K/V，只有一个固定大小的状态矩阵 | 不增长 | 第 23 章 |

`01_kv_ledger.py` 的核心是下面几行：

```python
def layer_list(m):
    if m.get("kv_lora_rank"):                                  # MLA
        per, kind = m["kv_lora_rank"] + m["qk_rope_head_dim"], "mla"
    else:                                                      # MHA / GQA / MQA
        per, kind = 2 * m["kv_heads"] * m["head_dim"], "full"
    ...

def kv_bytes(m, seq_len, batch=1, nbytes=BF16):
    total = 0
    for kind, per, window in layer_list(m):
        kept = min(seq_len, window) if kind == "sliding" else seq_len  # sliding window: keep at most `window`
        total += per * kept                                            # linear layer: per = 0
    return total * batch * nbytes
```

## 3. 给主线模型和最新模型记账

运行 `01_kv_ledger.py`。先看主线模型（28 层、16 个查询头、8 个 KV 头、head_dim 128、BF16）换用不同注意力时的账：

| 方案 | 每层每位置存 | 每 token | 32K | 128K |
|---|---:|---:|---:|---:|
| MHA（16 个 KV 头） | 4,096 个数 | 224 KiB | 7.00 GiB | 28.00 GiB |
| **GQA（主线，8 个 KV 头）** | **2,048** | **112 KiB** | **3.50 GiB** | **14.00 GiB** |
| MQA（1 个 KV 头） | 256 | 14 KiB | 0.44 GiB | 1.75 GiB |
| 假设换成 MLA（512 + 64） | 576 | 31.5 KiB | 0.98 GiB | 3.94 GiB |

再看 12 个公开模型。所有字段都读自各模型 Hugging Face 仓库的 `config.json`。读取时间是 2026-09，链接见文末"采用方与来源"。表中按 BF16、batch 1 计算，只算随序列增长的部分：

| 模型 | 注意力 | 随长度增长的层 / 总层 | 每 token | 32K | 128K | 对照：同头数全 MHA 的 128K |
|---|---|---:|---:|---:|---:|---:|
| Qwen3-0.6B | GQA 16/8，head_dim 128 | 28 / 28 | 112 KiB | 3.50 GiB | 14.00 GiB | 28.00 GiB |
| Qwen3-8B | GQA 32/8 | 36 / 36 | 144 KiB | 4.50 GiB | 18.00 GiB | 72.00 GiB |
| Llama-3.1-8B | GQA 32/8 | 32 / 32 | 128 KiB | 4.00 GiB | 16.00 GiB | 64.00 GiB |
| gpt-oss-120b | GQA 64/8，head_dim 64；一半的层用 128 的滑动窗口 | 36 / 36（18 层封顶 128） | 72 KiB¹ | 1.13 GiB | 4.50 GiB | 72.00 GiB |
| Qwen3.5-0.8B | GQA 8/2，head_dim 256；每 4 层有 1 层全注意力 | 6 / 24 | 12 KiB | 0.38 GiB | 1.50 GiB | 24.00 GiB |
| Qwen3.5-9B | GQA 16/4，head_dim 256；同上 | 8 / 32 | 32 KiB | 1.00 GiB | 4.00 GiB | 64.00 GiB |
| Qwen3.5-397B-A17B | GQA 32/2，head_dim 256；同上 | 15 / 60 | 30 KiB | 0.94 GiB | 3.75 GiB | 240.00 GiB |
| DeepSeek-V3 / V3.2 | MLA 512 + 64，128 头 | 61 / 61 | 68.6 KiB | 2.14 GiB | 8.58 GiB | 488.00 GiB |
| Kimi-K2 | MLA 512 + 64，64 头 | 61 / 61 | 68.6 KiB | 2.14 GiB | 8.58 GiB | 244.00 GiB |
| GLM-5 | MLA 512 + 64，64 头 | 78 / 78 | 87.8 KiB | 2.74 GiB | 10.97 GiB | 312.00 GiB |
| Mistral-Large-3 | MLA 512 + 64，128 头 | 61 / 61 | 68.6 KiB | 2.14 GiB | 8.58 GiB | 488.00 GiB |
| Kimi-K3 | 24 层 MLA（512 + 64）+ 69 层 KDA 线性注意力 | 24 / 93 | 27 KiB | 0.84 GiB | 3.38 GiB | 558.00 GiB |

¹ gpt-oss 的"每 token"是窗口填满之前的增量。过了 128 个 token 以后，只有 18 层全注意力层还在增长（每 token 36 KiB）。
对照列假设每层都是 MHA、KV 头数 = 查询头数、head_dim 128（Qwen3.5 按 256）。这是一个假想的上限，用来看各种方法省了多少。MLA 模型真实的 K 头宽是 192（128 + 64），按 MHA 算会更大。这里保守地按 128 算。

请注意以下几点：

- **小模型几乎都用 GQA。** 主线模型的 KV 配置与 Qwen3-0.6B 相同，每 token 112 KiB。
- **省缓存有三条路：**
  1. **少存头**：GQA / MQA。所有稠密小模型都在用。
  2. **每个位置存得更少**：MLA。DeepSeek、Kimi、GLM、Mistral 的旗舰 MoE 模型在用。61 层 × 576 个数，只有同头数 MHA 的 1/57。
  3. **少存层或少存位置**：Qwen3.5 只有 1/4 的层是全注意力，其余层是 Gated DeltaNet 线性注意力，状态大小固定。gpt-oss 一半的层只看最近 128 个 token。这是第 22、23 章的主题。
- **三条路可以叠加。** Qwen3.5 是"GQA + 混合线性注意力"，Kimi K3 是"MLA + 混合线性注意力"。Qwen3.5-397B 约有四千亿参数，但 128K 上下文的 KV cache 只有 3.75 GiB。这和 0.7B 的主线模型在 32K 时差不多。
- **线性注意力层也有状态，但状态不随长度增长。** 生产级计算器 `zero/tools/kv_cache_calc.py` 会单独估算它。例如 Qwen3.5-9B 的 24 层 Gated DeltaNet，按 `mamba_ssm_dtype: float32` 估算约 50 MiB，与上下文长度无关。状态和卷积缓存的具体存法依实现而定，所以这是估算值。

## 4. MQA → GQA → MLA

### 4.1 回顾：少存几个头

第 10 章讲过这些方法。在标准多头注意力（**MHA**）中，每个查询头都有自己的 K、V。**MQA**（Shazeer 2019）让所有查询头共用一组 K、V。**GQA**（Ainslie 等 2023）是两者的折中：每组查询头共用一组 K、V。三种方法省缓存的办法相同：**减少 KV 头数**。代价是 K、V 的表达能力变弱：共用一组 K、V 的查询头，看到的是完全相同的 key 和 value。

### 4.2 MLA：把 K、V 一起压成一个潜向量

DeepSeek-V2（2024）提出了**多头潜在注意力**（Multi-head Latent Attention，MLA）。MLA 换了一个思路：它不减少头数。它利用一个事实：每个头的 K、V 都是从同一个输入 x 线性变换得到的。所以 MLA 先把 x 压缩成一个很小的**潜向量**（latent vector）c_KV，再从 c_KV 还原出每个头的 K、V：

```
c_KV = RMSNorm(W_DKV · x)          # 下投影：7168 维 → 512 维（DeepSeek-V3）     ← 缓存
k_i^C = W_UK,i · c_KV              # 上投影：还原第 i 个头的 key（不带位置）
v_i   = W_UV,i · c_KV              # 上投影：还原第 i 个头的 value
```

缓存里只存 c_KV。这叫**低秩联合压缩**（low-rank joint compression）。"低秩"是因为乘积 W_UK · W_DKV 的秩不超过 512。"联合"是因为 K 和 V 共用同一个潜向量。

比较一下两种方法。GQA 规定"128 个头只能用 8 组 K/V"，硬性规定哪些头共享。MLA 规定"128 个头的 K/V 都必须能从同一个 512 维向量算出来"，共享的方式由训练学出来。每个头仍然有自己独立的 W_UK,i、W_UV,i，所以每个头看到的 key、value 各不相同。

对应 `03_mla.py` 中的代码：

```python
self.wkv_a = nn.Linear(dim, kv_lora_rank + rope_dim, bias=False)            # down-projection: x → [c_KV ; k_R]
self.kv_norm = tiny.RMSNorm(kv_lora_rank)
self.wkv_b = nn.Linear(kv_lora_rank, n_heads * (nope_dim + v_dim), bias=False)  # up-projection [W_UK; W_UV]
...
c_kv, k_pe = self.wkv_a(x).split([r, dr], dim=-1)
c_kv = self.kv_norm(c_kv)                        # the latent vector to cache
k_pe = tiny.apply_rope(k_pe[:, None], cs, sn)    # the RoPE key to cache
```

### 4.3 为什么 RoPE 要"解耦"

RoPE（第 9 章）要在 q、k 上乘一个和位置有关的旋转矩阵。假设直接旋转 k_i^C = W_UK,i · c_KV，下一节的"吸收"方法就不成立了。旋转矩阵夹在 W_UQ 和 W_UK 中间，两者无法提前相乘（矩阵乘法不满足交换律）。推理时只能把每个位置的 K 都还原出来再旋转。这样缓存潜向量就省不下任何东西。

DeepSeek 的解法是**解耦 RoPE**（decoupled RoPE）。每个头的 query 和 key 都拆成两段：

```
q_i = [ q_i^C ; RoPE(q_i^R) ]      # 128 维不带位置 + 64 维带位置
k_i = [ k_i^C ; RoPE(k^R)   ]      # k^R 由 x 直接算出，所有头共享一个
```

带位置的那一小段 k^R（`qk_rope_head_dim` = 64）由所有头共享，也要缓存。所以 MLA 每层每个位置缓存 **512 + 64 = 576 个数**。这就是 DeepSeek-V3、Kimi K2、GLM-5、Mistral Large 3 的 `config.json` 里 `kv_lora_rank: 512`、`qk_rope_head_dim: 64` 的含义。同样 128 个头的 MHA（K 每头 192、V 每头 128）要存 128 × 320 = 40,960 个数，MLA 只有它的 1.4%。DeepSeek-V2 论文的说法是：MLA 的缓存相当于只有 2.25 组的 GQA。

### 4.4 吸收（absorb）：推理时不还原 K、V

还原 K、V，意味着每一步都要把缓存里全部 T 个潜向量乘上 W_UK、W_UV。这样做很浪费。注意力可以换一个顺序算：

```
q_iᵀ k_j = (q_i^C)ᵀ W_UK,i c_KV,j + (q_i^R)ᵀ k^R_j
         = (W_UK,iᵀ q_i^C)ᵀ c_KV,j + (q_i^R)ᵀ k^R_j      ← 先把 query 投进潜空间
Σ_j p_ij v_j = W_UV,i ( Σ_j p_ij c_KV,j )                  ← 先在潜空间里加权平均，最后再投回去
```

矩阵乘法满足结合律。所以 W_UK 可以"吸收"进 query 一侧，W_UV 可以吸收进输出一侧。缓存里的潜向量直接参与注意力，K、V 从头到尾都不用还原。`03_mla.py` 实现了两条路径：

```python
if self.absorb:  # absorb: project the query into the latent space; dot product with the cache
    q_lat = torch.einsum("bhtd,hdr->bhtr", q_nope, w_uk)
    att = q_lat @ c_kv.transpose(-2, -1) + q_pe @ k_pe.transpose(-2, -1)
else:            # explicit: first reconstruct the K of each head
    k_nope = torch.einsum("bxsr,hdr->bhsd", c_kv, w_uk)
    ...
```

运行结果（同一组随机权重、float64）：

```
1. Absorbed path = explicit path (same weights, float64)
   max difference 4.4e-16
2. Token by token with a cache = one forward pass over the full sequence
   max difference 2.8e-16; shapes in the cache: latent (2, 1, 20, 32), RoPE key (2, 1, 20, 16)
```

两条路径在数学上等价，差异来自浮点舍入。吸收之后的形式值得注意。每个头拿一个 576 维的 query，与**同一份** 576 维的缓存做点积。这正是 MQA 的形式（所有头共用一组 K/V），只是这组"K/V"是 576 维的潜向量。GLM-5 的报告把它叫作"MLA 的 MQA 模式"。所以可以这样理解 MLA：**训练时像 MHA（每个头有自己的 K、V），推理时像 MQA（只存、只读一份）。**

实际系统常在训练和 prefill 中用显式路径。这两个阶段一次处理很多 token，算力是瓶颈。decode 一次处理一个 token，带宽是瓶颈，所以用吸收路径。

### 4.5 MLA 的代价

MLA 有代价。写作时能查到以下几点：

- **decode 的算力更高。** 吸收之后，每个头在 576 维上做点积，而 GQA 通常是 128 维。因此 GLM-5 把头维从 192 加到 256，把头数减少 1/3，以降低 decode 的算力。Kimi K2 也把头数从 DeepSeek-V3 的 128 减到 64。它的报告说，128K 上下文时，128 个头比 64 个头的推理 FLOPs 多 83%。
- **质量结论依赖设置。** 在 DeepSeek-V2 的消融实验里，MLA 比 MHA 还好（附录 D.2）。但 GLM-5 报告发现，用 Muon 优化器时，576 维潜向量的 MLA 不如 GQA-8。他们改变了优化器的用法（"Muon Split"）以后，MLA 才追平。
- **与一些组件不兼容。** Kimi K2 报告指出，QK-Norm 不能用在 MLA 上，因为推理时从不还原 K。主线模型用了 QK-Norm（第 9 章），这是它继续用 GQA 的原因之一。
- **工程更复杂。** MLA 需要专门的 kernel（例如 DeepSeek 开源的 FlashMLA），也需要推理引擎的支持。

这些代价也解释了目前的采用情况。采用 MLA 的都是几百亿到上万亿参数的旗舰 MoE 模型，而 0.6–9B 的稠密小模型几乎都用 GQA。

## 5. 小实验：同一个小模型，五种注意力

`04_attention_variants.py` 用第 10 章的字符级莎士比亚小模型（4 层、宽 128、4 个查询头、head_dim 32），只更换注意力：

- MHA（4 个 KV 头）、GQA（2 个）、MQA（1 个）。
- MLA-48：潜向量 48 维 + RoPE key 16 维 = 64 个数，**缓存和 MQA 一样大**。
- MLA-16：潜向量 16 维 + 16 = 32 个数，只有 MQA 的一半。

其余部分完全相同：同样的数据顺序、600 步、AdamW + warmup + cosine。每种结构跑 3 个随机种子。缓存大小是生成 512 个字符后**实测**的（FP32）：

```bash
uv run python chapters/21-kv-cache-ledger/code/04_attention_variants.py
```

> **注意：**本章训练类实验的数字来自课程构建机上的一次 CPU 运行。不同机器、不同版本的底层数学库，浮点运算的顺序略有不同。训练几百步后，这些微小差异会被放大。你在本机跑出的数字，可能从小数点后第二、三位开始就不同。请以下文中不依赖具体数值的结论为准。2026-10 在另一台服务器上复跑的对照见 [runs/2026-10-01-gpu0-check/chapters-21-23.md](../../runs/2026-10-01-gpu0-check/chapters-21-23.md)。

| 方案 | 每层每位置缓存 | 每 token（4 层，FP32） | 生成 512 字后实测缓存 | 注意力参数 | 验证 loss 均值 | 三个种子 | 缓存版 = 朴素版 |
|---|---:|---:|---:|---:|---:|---|---|
| MHA | 256 个数 | 4,096 B | 2,150,400 B（1×） | 262,144 | 1.858 | 1.838 / 1.869 / 1.867 | 是 |
| GQA | 128 | 2,048 B | 1,075,200 B（1/2） | 196,608 | 1.856 | 1.867 / 1.849 / 1.852 | 是 |
| MQA | 64 | 1,024 B | 537,600 B（1/4） | 163,840 | 1.860 | 1.860 / 1.858 / 1.862 | 是 |
| MLA-48 | 64（48 + 16） | 1,024 B | 537,600 B（1/4） | 245,952 | 1.887 | 1.878 / 1.888 / 1.895 | 是 |
| MLA-16 | 32（16 + 16） | 512 B | 268,800 B（1/8） | 196,672 | 1.886 | 1.889 / 1.854 / 1.916 | 是 |

先看能确定的部分：

- **缓存大小与账本完全一致。** 实测字节数与"每层每位置个数 × 4 层 × 4 字节 × 525 个位置"完全相等。MLA-48 与 MQA 一样大，MLA-16 只有 MHA 的 1/8。
- **缓存版与朴素版生成的字符相同。** 五种结构都是这样，包括走吸收路径的 MLA。用缓存生成的前 100 个字符，与每步整段重算的结果相同。MHA 种子 0 的 loss 1.838 与第 10 章完全相同，说明训练循环与第 10 章一致。

再看 loss，这里要小心：

- **MHA、GQA、MQA 分不出高下。** 三者的均值在 1.856–1.860 之间，而 MHA 自己换种子就能差 0.031（1.838 对 1.869）。结论和第 10 章一样：在这个规模上，随机性掩盖了 KV 头数的影响。
- **MLA 在这里稍差。** MLA-48 的三个种子（1.878–1.895）全部高于前三种结构的全部九个种子（最高 1.869）。它的均值比缓存同样大的 MQA 高约 0.03。MLA-16 的种子波动很大（1.854–1.916），无法得出结论。
- **不能据此说"MLA 不如 MQA"。** 可能的原因至少有三个。第一，学习率等超参数是按第 10 章的 MHA 定的，没有为 MLA 单独调。第二，这个模型只有 4 个头、head_dim 32。48 维的潜向量相对头宽并不"低秩"，所以 MLA 的结构优势（每个头有独立的上投影）没有多少发挥空间。第三，训练只有 600 步。DeepSeek-V2 在 16B 和 250B 的 MoE 上做了对比，结果是 MLA 优于 MHA。GLM-5 在他们的设置下发现，MLA 需要调整优化器才能追平 GQA-8。**MLA 好不好，取决于规模和训练设置。**这个极小实验只能说明三点：代码是对的，缓存省下来了，质量代价在这个规模上可见但很小。

GOAL.md 第五部分的"第二步"会用生产级的 `zero/arch/mla.py`，在约 1 亿参数的 ladder 配置上重跑这组对比。那时会用多个种子，并为每种结构单独调学习率。到时再下更可靠的结论。

## 6. 小结

- **长上下文的两种贵**：prefill 的注意力算力按 T² 增长（主线模型 32K 时占 73%）。decode 受显存带宽限制（算术强度只有 1–5）。长上下文下，KV cache 比权重还大，拼批也省不掉它的读取。
- **账本**：按层记账。全注意力层和 MLA 层随长度线性增长。滑动窗口层有上限。线性注意力层是常数。
- **三条省缓存的路**：少存头（GQA/MQA）、每个位置存得更少（MLA）、少存层或位置（滑动窗口、混合线性注意力）。三条路可以叠加。
- **MLA**：把 K、V 联合压缩成潜向量 c_KV（512 维），加上共享的 RoPE key（64 维），每层每位置 576 个数。解耦 RoPE 让"吸收"成为可能。吸收后，MLA 训练时像 MHA，推理时像 MQA。
- **代价**：decode 算力更高、与 QK-Norm 不兼容、需要专门的 kernel。目前 MLA 是大 MoE 模型的选择，稠密小模型仍用 GQA。

---

## GPU 实测（单张 RTX 3090）

> **注意：**上面正文里的数字都来自 CPU 运行。本节换到一张 NVIDIA GeForce RTX 3090 上实测：24 GB 显存，Ampere 架构。规格表：BF16 张量核稠密峰值约 71 TFLOPS，FP32 约 35.6 TFLOPS，显存带宽约 936 GB/s。环境：PyTorch 2.11.0+cu128、CUDA 12.8，2026 年 10 月。服务器把这张卡的功耗上限设成了 240 W（出厂默认 350 W）。持续满载时，这张卡会降频。所以算力和带宽的绝对值比满功耗的 3090 低，看相对关系更可靠。没有 GPU 可以跳过本节。

运行：

```bash
uv run python chapters/21-kv-cache-ledger/code/05_gpu_decode_ledger.py
```

脚本按主线模型的形状搭建一个随机权重的 decode 步（BF16）：28 层、宽 1280、16 个查询头、8 个 KV 头、head_dim 128、FFN 3584、词表 65,536。KV cache 预先填满 T 个位置，然后测量"再生成 1 个 token"要多久。"理论下限"直接调用 `02_prefill_decode.py` 的 `decode_step`。公式不变，只把硬件规格换成 3090 的 71 TFLOPS、936 GB/s。"等效带宽"= 实际读取的字节（权重 + KV cache）÷ 实测耗时。脚本用 CUDA Graph 把整步录下来再重放。如果不这样做，T = 1,024 时一步要 7.51 ms，一大半时间花在 Python 逐个发射 kernel 上。作为参照，在同一张卡上把 1 GiB 连续读一遍，实测带宽是 873 GB/s。

主线模型（GQA 16/8），decode 一步（30 次取中位数）：

| 上下文 T | batch B | 读权重 | 读 KV cache | 理论下限 | 实测 | 实测 / 理论 | 等效带宽 | 吞吐（token/s） |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1,024 | 1 | 1.28 GiB | 0.11 GiB | 1.60 ms | 3.40 ms | 2.13× | 440 GB/s | 294 |
| 4,096 | 1 | 1.28 GiB | 0.44 GiB | 1.98 ms | 4.09 ms | 2.07× | 452 GB/s | 244 |
| 4,096 | 16 | 1.28 GiB | 7.00 GiB | 9.50 ms | 17.10 ms | 1.80× | 520 GB/s | 936 |
| 32,768 | 1 | 1.28 GiB | 3.50 GiB | 5.49 ms | 9.25 ms | 1.69× | 555 GB/s | 108 |
| 32,768 | 4 | 1.28 GiB | 14.00 GiB | 17.53 ms | 27.48 ms | 1.57× | 597 GB/s | 146 |
| 131,072 | 1 | 1.28 GiB | 14.00 GiB | 17.53 ms | 27.23 ms | 1.55× | 603 GB/s | 37 |

batch 1 时，从 T = 1,024 到 131,072，KV cache 多读 13.89 GiB，一步多花 23.83 ms，折合 626 GB/s。

同一个上下文（T = 32,768、batch 1），只换注意力。MLA 取 512 + 64，走 `03_mla.py` 的吸收路径：

| 方案 | 每层每位置 | 账本 KV | 实际分配 | 权重 | 理论下限 | 实测 | 等效带宽 |
|---|---:|---:|---:|---:|---:|---:|---:|
| MHA | 4,096 个数 | 7.00 GiB | 7.00 GiB | 1.42 GiB | 9.66 ms | 15.57 ms | 581 GB/s |
| **GQA（主线）** | 2,048 | 3.50 GiB | 3.50 GiB | 1.28 GiB | 5.49 ms | 9.33 ms | 550 GB/s |
| MQA | 256 | 0.44 GiB | 0.44 GiB | 1.16 GiB | 1.84 ms | 4.36 ms | 395 GB/s |
| MLA | 576 | 0.98 GiB | 0.98 GiB | 1.36 GiB | 3.70 ms | 7.75 ms | 447 GB/s |

（权重大小不同，是因为 K/V 投影的形状随方案变化。MLA 的理论下限按实际读取的字节算：算分数时读一遍 576 维潜向量，加权平均时再读一遍前 512 维。FlashMLA 这类融合 kernel 只读一遍。）

这两张表把第 1.2 节的公式变成了实测结果。batch 1 时，上下文从 1K 增长到 128K。每多读 1 GiB KV cache，一步就多花约 1.7 ms。"4 条 32K 的对话"和"1 条 128K 的对话"要读的 KV cache 一样多（都是 14 GiB），一步的耗时也几乎一样（27.48 ms 对 27.23 ms）。decode 只取决于要读多少字节，与这些字节属于几条对话无关。

拼批的规律也和账本一致。4K 时 batch 从 1 增大到 16，吞吐涨了约 3.8 倍（244 → 936）。32K 时 batch 从 1 增大到 4，吞吐只涨了约 1.35 倍（108 → 146）。拼批省不掉 KV cache 的读取。

第 3 节的账本在显存里完全准确（实际分配 7.00 / 3.50 / 0.44 / 0.98 GiB），decode 的耗时也按 KV cache 的大小排序。但 MLA 的缓存只有 GQA 的 28%，一步却只快了约 17%。没有融合 kernel 时，潜向量要读两遍，注意力里的矩阵乘又小又碎。这就是第 4.5 节说 MLA "需要 FlashMLA 这类专门 kernel"的原因。

离理论下限的距离出乎意料。同一张卡纯读能到 873 GB/s，可短上下文时等效带宽只有 440 GB/s。每层的权重矩阵只有 5–18 MB。batch 1 时，一步是一串小的矩阵-向量乘，每一个都不够大，无法把带宽用满。上下文越长，大块连续读取 KV cache 的比重越高，等效带宽才向 600 GB/s 靠近。

## 从极简代码到生产级代码

| 极简代码（`code/`） | 生产级代码 | 多做了什么、为什么 |
|---|---|---|
| `01_kv_ledger.py`：手写的模型字典 + `layer_list` / `kv_bytes` | `zero/tools/kv_cache_calc.py`：`kv_cache_bytes(cfg, seq_len, batch, dtype_bytes)`、`kv_bytes_per_token`、`fixed_state_bytes`、`breakdown`；命令行 `uv run python -m zero.tools.kv_cache_calc configs/main/pretrain.toml --seq 32768` | 直接读取 zero 的 `ModelConfig` / TOML 和 Hugging Face 的 `config.json`。支持多模态模型嵌套的 `text_config`、`layer_types`、Kimi 的 `linear_attn_config`、Mistral 原生 `params.json` 的字段名。自动识别 MLA、滑动窗口、线性注意力层。单独估算线性层的固定状态。`--dtype-bytes 1` 可以算 FP8 KV cache。 |
| `03_mla.py` 的 `MLA`：接口与第 10 章的小模型一致，借用它的 `torch.cat` 式缓存 | `zero/arch/mla.py`：`MLAConfig`（字段名与 DeepSeek-V3 的 config 一致）、`MLAAttention`（接口与 `zero.model.Attention` 相同，可以直接换进 `Transformer`，见 `mla_transformer`）、`MLACache`（预分配潜向量与 RoPE key，`nbytes()`） | 也支持对 query 做低秩压缩（`q_lora_rank`，DeepSeek-V3 为 1536；省训练激活，不省缓存）。有缓存时自动走吸收路径，无缓存时走显式路径 + SDPA。支持分块 prefill。softmax 至少在 float32 上计算。 |
| `04_attention_variants.py`：MHA/GQA/MQA/MLA 同配置对比 | 第二步（可选）：用 `mla_transformer(model_cfg, mla_cfg)` 在约 1 亿参数的 ladder 配置上重跑 | 生产级模块与主线 `Transformer` 共用 RMSNorm、SwiGLU 和训练循环。换注意力只需改一处。 |
| 无 | 行业实现：vLLM 的 PagedAttention（第 10 章）按页管理 KV cache；DeepSeek 开源的 FlashMLA 是 MLA decode 的 GPU kernel；vLLM、SGLang 都有 MLA 后端 | 本课的 MLA 只追求可读和正确。它在 CUDA + BF16 下的正确性已在 RTX 3090 上验证（见 [runs/2026-10-01-gpu0-check](../../runs/2026-10-01-gpu0-check/README.md) 第 11 节）。它的性能没有优化，也还没有在 GPU 上验证（本章"GPU 实测"一节有一组 decode 耗时供参考）。真正上线要用这些专门实现。 |

**对拍**（parity check；`uv run pytest tests/test_kv_cache_calc.py tests/test_arch_mla.py`，本机 12 项全部通过，约 3 秒）：

- 计算器对 `configs/tiny`、`configs/main`，在 FP32 / BF16、batch 2 下，与 `KVCache.from_config(...).nbytes()` 真实分配的字节数完全相等。主线模型每 token 114,688 字节。
- MLA 配置的结果与 `MLACache.nbytes()` 相等。DeepSeek-V3 每 token 61 × 576 × 2 字节。滑动窗口、Qwen3.5 式混合线性注意力、Mistral `params.json` 各有一个手算用例。
- `MLAAttention` 的吸收路径与显式路径在 float64 下差异 < 1e-10。分块 prefill（12 + 1 + 17 个 token）与一次性前向在 1e-5 内一致。带缓存与不带缓存的贪心生成 40 个 token 完全相同。换上 MLA 的 `Transformer` 能正常训练。

`01_kv_ledger.py` 的第 3 部分也直接调用 `zero.tools.kv_cache_calc`，核对了主线模型和 DeepSeek-V3 的数字。

---

## 前沿观察

> **稀疏注意力（DeepSeek DSA）不省 KV cache，省的是算力。** DeepSeek-V3.2 和 GLM-5 在 MLA 之上加了 DeepSeek Sparse Attention。一个轻量的"索引器"（indexer）给历史 token 打分，每个 query 只和得分最高的 2048 个 token（`index_topk: 2048`）做注意力。这把长上下文的注意力算力从 O(T²) 降到接近 O(T·k)。但 KV cache 仍要全部保留，索引器自己还要额外缓存一份小 key。写这一章时，只核实到 DeepSeek 和 GLM 两个家族。第 22 章再次核实后，"学出来的稀疏注意力"这个大方向已有 4 家采用（DeepSeek、GLM-5、MiniMax-M3、美团 LongCat）。这满足 GOAL.md 2.1，所以在第 22 章正文讲。但各家的具体做法（DSA、MSA、LSA 等）还没有收敛，所以具体变体仍算前沿观察。它们省的依然是算力，不是 KV cache。

---

## 采用方与来源

| 技术 | 采用方（主力版本） | 来源 |
|---|---|---|
| GQA | Qwen3（0.6B：16 Q / 8 KV；8B：32 / 8）、Llama 3.1（32 / 8）、gpt-oss（64 / 8）、Qwen3.5 的全注意力层（0.8B：8 / 2）；更多见第 10 章 | 各模型的 `config.json`（下方链接）；Qwen3、Llama 3 技术报告 |
| MLA | **DeepSeek**（V2 提出；V3、V3.2：`kv_lora_rank` 512、`qk_rope_head_dim` 64）、**Kimi**（K2：技术报告 2.3 节"employing MLA"；K3 的全注意力层）、**GLM**（GLM-5：技术报告 2.1 节"Multi-latent Attention"）、**Mistral**（Mistral Large 3：`params.json` 中 `kv_lora_rank` 512、`qk_rope_head_dim` 64，与 DeepSeek-V3 同形；模型卡没有用文字说明） | DeepSeek-V2 arXiv:2405.04434；DeepSeek-V3 arXiv:2412.19437；Kimi K2 arXiv:2507.20534；GLM-5 arXiv:2602.15763；各模型配置 |
| 滑动窗口 / 局部-全局交替 | gpt-oss（`layer_types` 交替，`sliding_window` 128）；更多见第 22 章 | gpt-oss 的 `config.json` |
| 混合线性注意力 | Qwen3.5（`full_attention_interval` 4，其余为 `linear_attention`）、Kimi K3（`linear_attn_config`：24 层全注意力 + 69 层 KDA）；更多见第 23 章 | 各模型的 `config.json` |
| KV cache 分页管理 / MLA kernel | vLLM（PagedAttention，行业标准）；FlashMLA（DeepSeek 开源） | Kwon 等 2023，arXiv:2309.06180；<https://github.com/deepseek-ai/FlashMLA> |

**共识判断（GOAL.md 2.1）**：DeepSeek、Kimi、GLM、Mistral 四个彼此独立的头部家族在主力版本中采用了 MLA。这满足规则 A，所以 MLA 进正文。但要说明它的适用范围：四家都是几百亿到上万亿参数的 MoE 旗舰模型。**稠密小模型没有采用 MLA**（Qwen3、Qwen3.5、Llama 3、Gemma 3、gpt-oss、SmolLM3 都用 GQA 或 MQA）。所以主线模型不用它（GOAL.md 3.3）。MQA 作为 GQA 的端点来讲（规则 C）。在头部家族里，只查到 Gemma 3 1B 用 MQA（见第 10 章）。

**模型配置**（2026-09 通过 Hugging Face 读取）：
[Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B/blob/main/config.json)、
[Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B/blob/main/config.json)、
[Llama-3.1-8B（unsloth 镜像）](https://huggingface.co/unsloth/Meta-Llama-3.1-8B/blob/main/config.json)、
[gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b/blob/main/config.json)、
[Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B/blob/main/config.json)、
[Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B/blob/main/config.json)、
[Qwen3.5-397B-A17B（读自 FP8 版）](https://huggingface.co/Qwen/Qwen3.5-397B-A17B-FP8/blob/main/config.json)、
[DeepSeek-V3](https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/config.json)、
[DeepSeek-V3.2](https://huggingface.co/deepseek-ai/DeepSeek-V3.2/blob/main/config.json)、
[Kimi-K2-Instruct](https://huggingface.co/moonshotai/Kimi-K2-Instruct/blob/main/config.json)、
[GLM-5](https://huggingface.co/zai-org/GLM-5/blob/main/config.json)、
[Mistral-Large-3-675B-Instruct-2512 的 params.json](https://huggingface.co/mistralai/Mistral-Large-3-675B-Instruct-2512/blob/main/params.json)、
[Kimi-K3](https://huggingface.co/moonshotai/Kimi-K3/blob/main/config.json)。

说明：Meta 官方的 `meta-llama/Llama-3.1-8B` 需要申请权限，本章读取不到。数字取自 unsloth 镜像（它的 `_name_or_path` 指向官方仓库），与第 10 章一致。Kimi K3 的 `full_attn_layers` 层号从 1 开始编号（1–93），本章按此解读为 24 层 MLA、69 层 KDA。H100 SXM 的 989.5 TFLOPS 与 `zero/tools/estimate_cost.py` 采用同一口径。3.35 TB/s 和 80 GB 取自 NVIDIA H100 产品页 <https://www.nvidia.com/en-us/data-center/h100/>。

---

## 引导问题

带着这些问题去问 Claude Code，直到你能用自己的话讲清楚：

1. 第 1.2 节说"长上下文下拼批省不掉 KV cache 的读取"。那为什么拼批还能省下权重的读取？试着估算：主线模型在 32K 上下文下，batch 多大时，"读 KV cache"的时间是"读权重"的 10 倍？
2. MLA 的潜向量是 512 维，而 DeepSeek-V3 的隐藏层是 7168 维。如果把潜向量加大到 7168 维，还省缓存吗？还有意义吗？"低秩"这个限制在这里起了什么作用？
3. 为什么 k^R（带 RoPE 的那段 key）由所有头共享一个，而不是每个头一个？如果每个头一个，缓存会变成多少？
4. 吸收之后，MLA 在 decode 时"形式上就是 MQA"。它和真正的 MQA（第 10 章，1 个 KV 头、head_dim 32）区别在哪里？为什么 MLA 的质量可以比 MQA 好得多？
5. Qwen3.5 和 Kimi K3 都是"少量全注意力 + 大量线性注意力"。如果全注意力层也换成 MLA（像 Kimi K3 那样），账本会怎样变？如果 Qwen3.5-9B 这样改，128K 的 KV cache 大约是多少？
6. 本章小实验里，各方案之间的 loss 差距和种子之间的差距是什么关系？如果想认真比较 MLA 和 GQA，你会怎样设计实验？参考 GLM-5 报告表 1 的做法。

## 动手任务

每个任务都要运行代码，并查看结果。

**任务 1（基础）**：在 Hugging Face 上挑一个本章没算过的模型，例如 SmolLM3-3B、Gemma 3 系列，或 MiniMax、GLM 的其他版本。读它的 `config.json`，把字段加进 `01_kv_ledger.py` 的 `MODELS`，算出 32K 和 128K 的 KV cache。再用 `uv run python -m zero.tools.kv_cache_calc 你保存的config.json --seq 131072` 对拍。注意它有没有滑动窗口和 `layer_types`。

**任务 2（核心）**：在 `03_mla.py` 里给 `MLA` 加一个计时实验。随机初始化一个 `n_heads=16, kv_lora_rank=64, rope_dim=16` 的 MLA。先 prefill 1024 个位置，再分别用吸收路径和显式路径 decode 64 步，比较每步耗时。再把缓存长度换成 256 和 4096，观察趋势。解释为什么缓存越长，吸收路径的优势越大。

**任务 3（挑战）**：在 `04_attention_variants.py` 里加一个 "GQA-2 × head_dim 16" 之类的变体，让它的每层每位置缓存也是 64 个数（2 × 2 × 16，和 MQA、MLA-48 一样大）。然后跑 3 个种子。缓存大小相同时，哪种结构的 loss 更低？差距是否超出了种子之间的波动？再把训练步数加倍，看结论变不变。

---

## 想深入：CS336

本章对应斯坦福 CS336（Spring 2026）<https://cs336.stanford.edu/>：

- **第 10 讲：推理**。它从资源核算的角度讲推理的开销：prefill 与 decode、KV cache 的显存与内存带宽，以及让推理更快、更省的各类方法。讲义与录像见课程页。本构建环境访问不了课程页，具体覆盖哪些架构方法，以讲义为准。本章的 `02_prefill_decode.py` 可以看作这种资源核算思路在主线模型上的一次具体演算。

---

## 本章参考文献

- DeepSeek-AI. *DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model*（MLA 的提出：低秩联合压缩、解耦 RoPE、吸收、附录 D 的消融），2024：<https://arxiv.org/abs/2405.04434>
- DeepSeek-AI. *DeepSeek-V3 Technical Report*，2024：<https://arxiv.org/abs/2412.19437>
- Kimi Team. *Kimi K2: Open Agentic Intelligence*（2.3 节：沿用 MLA、头数从 128 减到 64 的理由；QK-Norm 不适用于 MLA），2025：<https://arxiv.org/abs/2507.20534>
- GLM-5 Team. *GLM-5: from Vibe Coding to Agentic Engineering*（2.1 节：MLA 与 GQA-8 的对比、Muon Split、MLA-256；DSA），2026：<https://arxiv.org/abs/2602.15763>
- Mistral AI. *Mistral 3*（Mistral Large 3 发布博客）：<https://mistral.ai/news/mistral-3>
- Shazeer. *Fast Transformer Decoding: One Write-Head is All You Need*（MQA），2019：<https://arxiv.org/abs/1911.02150>
- Ainslie et al. *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints*，2023：<https://arxiv.org/abs/2305.13245>
- Kwon et al. *Efficient Memory Management for Large Language Model Serving with PagedAttention*（vLLM），2023：<https://arxiv.org/abs/2309.06180>
- DeepSeek. FlashMLA（MLA decode kernel）：<https://github.com/deepseek-ai/FlashMLA>
- [CS336](https://cs336.stanford.edu/) 第 10 讲（推理）
- [1.5 万字速通 LLM 主流模型结构（Llama、Qwen、GLM、DeepSeek…）](https://zhuanlan.zhihu.com/p/2060741715095560795)：各家注意力结构的横向对比（`references.md` 已收录）
- [Pretraining a Mini Kimi K3](https://books.vizuara.ai/book/pretraining-a-mini-k3)：Kimi K3 的"MLA + KDA 混合"架构的动手预训练（`references.md` 已收录）
- 模型配置链接见上方"采用方与来源"。

**下一章**：账本里还有两条路没走完。一些层可以只看最近的一段（滑动窗口），另一些层可以完全不存 K、V（线性注意力）。第 22 章先讲局部与稀疏注意力：gpt-oss 为什么一半的层只看 128 个 token？Gemma 为什么大部分层是局部的？模型怎样在"只看附近"的同时，不丢掉远处的信息？
