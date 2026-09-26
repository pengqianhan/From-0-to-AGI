# 第 21 章：KV cache 的账本 —— 长上下文贵在哪，每个 token 该存多少

> **一句话目标**：读完这一章，你能说清楚长上下文在 prefill 和 decode 两个阶段分别贵在哪里；能按层给任意一个开源模型的 `config.json` 算出 KV cache 有多大（全注意力、滑动窗口、线性注意力混合、MLA 四种层）；还能讲清楚 MLA 是怎么把 K、V 压成一个潜向量、又怎么在推理时"吸收"掉还原步骤的。

📺 **本章视频**：待发布（本地渲染：`bash chapters/21-kv-cache-ledger/video/build.sh`）
🧪 **本章自检**：学完后在 Claude Code 里输入 `/ch21-kv-cache`

---

第四部分结束时，我们有了一个会调用工具的主线模型。第五部分换一个视角：不再训练主线模型，而是看过去两年开源模型的**架构**往哪里演进。这些演进几乎都指向同一个方向——**为了更长的上下文、更小的 KV cache**。

第 10 章我们第一次算过这笔账：主线模型每个 token 要缓存 112 KiB，一条 32K 的对话就是 3.5 GiB，比模型权重（1.28 GiB）还大。GQA 让 8 个 KV 头代替 16 个，省了一半。这一章要解决的问题是：**长上下文到底贵在哪？KV cache 还能怎么压？** 我们先把"贵"拆成两笔账（prefill 的算力、decode 的显存带宽），再把第 10 章的公式升级成按层记账的"账本"，拿它去算几个最新的开源模型，最后讲 MQA → GQA → MLA 这条压缩之路，并在 CPU 上把五种注意力放在同一个小模型里比一比。

本章代码：

```bash
uv run python chapters/21-kv-cache-ledger/code/01_kv_ledger.py          # 账本：主线模型 + 12 个公开模型
uv run python chapters/21-kv-cache-ledger/code/02_prefill_decode.py     # prefill 算力 / decode 带宽 / 并发条数
uv run python chapters/21-kv-cache-ledger/code/03_mla.py                # 极简 MLA：吸收路径与显式路径对拍
uv run python chapters/21-kv-cache-ledger/code/04_attention_variants.py # MHA/GQA/MQA/MLA 对比（首次约半小时）
```

## 1. 长上下文贵在哪：两个阶段，两种瓶颈

第 10 章讲过，有了 KV cache，推理分成两段：**prefill**（整段提示词一次喂进去）和 **decode**（之后每步只喂一个新 token）。长上下文在这两段里贵的方式完全不同。

### 1.1 prefill：算力按 T² 涨

prefill 的前向运算量分两部分（`02_prefill_decode.py` 的 `prefill_flops`）：

```python
linear = 2 * n_matmul * T                  # 矩阵乘：每个参数每个 token 一次乘加 = 2 次运算
attn = 2 * c.n_layers * c.q_dim * T * T    # QKᵀ 与 AV：每对 (i, j) 4·q_dim 次，因果掩码只有 T²/2 对
```

第一项随 T 线性增长，第二项随 T² 增长。代入主线模型（参与矩阵乘的参数 N = 689.4M，q_dim = 16 × 128 = 2048），按 H100 SXM 的稠密 BF16 峰值 989.5 TFLOPS 算理论耗时：

| 提示词长度 T | 矩阵乘部分（次） | 注意力部分（次） | 注意力占比 | H100 理论下限 |
|---:|---:|---:|---:|---:|
| 1,024 | 1.41 × 10¹² | 1.20 × 10¹¹ | 7.8% | 1.5 ms |
| 4,096 | 5.65 × 10¹² | 1.92 × 10¹² | 25.4% | 7.7 ms |
| 32,768 | 4.52 × 10¹³ | 1.23 × 10¹⁴ | **73.2%** | 170 ms |
| 131,072 | 1.81 × 10¹⁴ | 1.97 × 10¹⁵ | **91.6%** | 2.2 s |

4K 以内，注意力只是配角；到 128K，九成以上的算力花在注意力上。这是"首 token 延迟"（time to first token）随上下文变长而急剧变差的原因，也是第 22、23 章滑动窗口、线性注意力要解决的问题。**KV cache 本身不改变 prefill 的算力**，这一章主要关心下面的 decode。

### 1.2 decode：卡在显存带宽上

decode 每一步只算一个 token，但为了这一个 token，GPU 要把**全部权重读一遍**，再把这条对话的**整个 KV cache 读一遍**。设 batch 里有 B 条对话、每条已有 T 个 token（`decode_step`）：

```python
flops = B * (2 * n_matmul + 4 * c.n_layers * c.q_dim * T)   # 要算的
w_bytes = n_params * BF16                                     # 要读的：权重（B 条共享）
kv = B * T * kv_per_token                                     # 要读的：每条各自的 KV cache
t = max(flops / PEAK_FLOPS, (w_bytes + kv) / HBM_BW)          # 一步的理论最短时间
```

"算力 ÷ 读取字节"叫**算术强度（arithmetic intensity）**。H100 SXM 每秒能算 989.5 万亿次、能读 3.35 TB，两者之比约 295 次/字节，叫**脊点（ridge point）**：强度低于它，GPU 就在等数据，叫**带宽受限（memory-bound）**。主线模型的实际情况：

| 上下文 T | batch B | 读权重 | 读 KV cache | 算术强度 | 一步理论耗时 | 吞吐（token/s） | 80 GB 装得下？ |
|---:|---:|---:|---:|---:|---:|---:|---|
| 4,096 | 1 | 1.28 GiB | 0.44 GiB | 1.3 | 0.55 ms | 1,812 | 是 |
| 4,096 | 16 | 1.28 GiB | 7.00 GiB | 4.2 | 2.66 ms | 6,026 | 是 |
| 4,096 | 64 | 1.28 GiB | 28.00 GiB | 4.7 | 9.39 ms | 6,819 | 是 |
| 32,768 | 1 | 1.28 GiB | 3.50 GiB | 1.7 | 1.53 ms | 652 | 是 |
| 32,768 | 16 | 1.28 GiB | 56.00 GiB | 2.3 | 18.36 ms | 871 | 是 |
| 32,768 | 64 | 1.28 GiB | 224.00 GiB | 2.4 | 72.21 ms | 886 | **否** |

算术强度只有 1–5，离 295 差两个数量级：decode 完全是带宽受限的。第 10 章说"把很多请求拼成一批，读一遍权重服务几十个请求"——这招在短上下文下有效（4K 时 batch 从 1 到 16，吞吐涨了 3.3 倍），**但在长上下文下失灵了**：32K 时 KV cache（每条 3.5 GiB）远大于权重（1.28 GiB），而 KV cache 是每条对话各读各的，拼批省不掉。batch 从 1 到 16，吞吐只涨了 1.3 倍。

所以长上下文的 decode 速度，几乎就由"每个 token 的 KV cache 有多大"决定。

### 1.3 显存：能同时服务几条对话

还有一个更硬的约束：放不放得下。80 GB 减去权重，全部留给 KV cache（不计激活、碎片，偏乐观）：

| 上下文 | GQA（主线，112 KiB/token） | 若是 MHA（224 KiB） | 若换 MLA 512+64（31.5 KiB） |
|---:|---:|---:|---:|
| 4,096 | 167 条 | 83 条 | 595 条 |
| 32,768 | 20 条 | 10 条 | 74 条 |
| 131,072 | 5 条 | 2 条 | 18 条 |

KV cache 每小一半，同一张卡能同时服务的对话就多一倍，decode 时要读的字节也少一半。这就是整个第五部分的动机。

## 2. 账本：按层记

第 10 章的公式假设每层都一样：

```
KV cache 字节数 = 2 × 层数 × KV 头数 × head_dim × 序列长 × 每个数的字节数（× batch）
```

可是 2025–2026 年的开源模型，层和层已经不一样了。有的层只看最近 128 个 token（滑动窗口），有的层根本不存 K、V（线性注意力），有的层存的不是 K、V 而是一个压缩过的潜向量（MLA）。所以账要按层记：

| 层类型 | 每层每个位置存什么 | 随序列长怎么涨 | 本章之后在哪讲 |
|---|---|---|---|
| 全注意力（MHA / GQA / MQA） | K、V 各 `KV 头数 × head_dim` 个数 | 线性增长 | 第 8、10 章，本章第 4 节 |
| MLA | 潜向量 `kv_lora_rank` + 共享 RoPE key `qk_rope_head_dim` | 线性增长，但每个位置小得多 | 本章第 4 节 |
| 滑动窗口 | 同全注意力 | 最多存 `window` 个位置，之后不再增长 | 第 22 章 |
| 线性注意力（Gated DeltaNet、KDA 等） | 不存 K/V，只有一个固定大小的状态矩阵 | 不增长 | 第 23 章 |

`01_kv_ledger.py` 的核心就这几行：

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
        kept = min(seq_len, window) if kind == "sliding" else seq_len  # 滑动窗口：最多存 window 个
        total += per * kept                                            # 线性层 per = 0
    return total * batch * nbytes
```

## 3. 给主线模型和最新模型记账

运行 `01_kv_ledger.py`。先看主线模型（28 层、16 个查询头、8 个 KV 头、head_dim 128、BF16）换不同注意力时的账：

| 方案 | 每层每位置存 | 每 token | 32K | 128K |
|---|---:|---:|---:|---:|
| MHA（16 个 KV 头） | 4,096 个数 | 224 KiB | 7.00 GiB | 28.00 GiB |
| **GQA（主线，8 个 KV 头）** | **2,048** | **112 KiB** | **3.50 GiB** | **14.00 GiB** |
| MQA（1 个 KV 头） | 256 | 14 KiB | 0.44 GiB | 1.75 GiB |
| 假设换成 MLA（512 + 64） | 576 | 31.5 KiB | 0.98 GiB | 3.94 GiB |

再看 12 个公开模型。字段全部读自各模型 Hugging Face 仓库的 `config.json`（2026-09 读取，链接见文末"采用方与来源"），BF16、batch 1，只算随序列增长的部分：

| 模型 | 注意力 | 随长度增长的层 / 总层 | 每 token | 32K | 128K | 对照：同头数全 MHA 的 128K |
|---|---|---:|---:|---:|---:|---:|
| Qwen3-0.6B | GQA 16/8，head_dim 128 | 28 / 28 | 112 KiB | 3.50 GiB | 14.00 GiB | 28.00 GiB |
| Qwen3-8B | GQA 32/8 | 36 / 36 | 144 KiB | 4.50 GiB | 18.00 GiB | 72.00 GiB |
| Llama-3.1-8B | GQA 32/8 | 32 / 32 | 128 KiB | 4.00 GiB | 16.00 GiB | 64.00 GiB |
| gpt-oss-120b | GQA 64/8，head_dim 64；一半层滑动窗口 128 | 36 / 36（18 层封顶 128） | 72 KiB¹ | 1.13 GiB | 4.50 GiB | 72.00 GiB |
| Qwen3.5-0.8B | GQA 8/2，head_dim 256；每 4 层 1 层全注意力 | 6 / 24 | 12 KiB | 0.38 GiB | 1.50 GiB | 24.00 GiB |
| Qwen3.5-9B | GQA 16/4，head_dim 256；同上 | 8 / 32 | 32 KiB | 1.00 GiB | 4.00 GiB | 64.00 GiB |
| Qwen3.5-397B-A17B | GQA 32/2，head_dim 256；同上 | 15 / 60 | 30 KiB | 0.94 GiB | 3.75 GiB | 240.00 GiB |
| DeepSeek-V3 / V3.2 | MLA 512 + 64，128 头 | 61 / 61 | 68.6 KiB | 2.14 GiB | 8.58 GiB | 488.00 GiB |
| Kimi-K2 | MLA 512 + 64，64 头 | 61 / 61 | 68.6 KiB | 2.14 GiB | 8.58 GiB | 244.00 GiB |
| GLM-5 | MLA 512 + 64，64 头 | 78 / 78 | 87.8 KiB | 2.74 GiB | 10.97 GiB | 312.00 GiB |
| Mistral-Large-3 | MLA 512 + 64，128 头 | 61 / 61 | 68.6 KiB | 2.14 GiB | 8.58 GiB | 488.00 GiB |
| Kimi-K3 | 24 层 MLA 512 + 64 + 69 层 KDA 线性注意力 | 24 / 93 | 27 KiB | 0.84 GiB | 3.38 GiB | 558.00 GiB |

¹ gpt-oss 的"每 token"是窗口填满之前的增量；过了 128 个 token 以后，只有 18 层全注意力层还在涨（每 token 36 KiB）。
对照列按"每层都是 MHA、KV 头数 = 查询头数、head_dim 128（Qwen3.5 按 256）"算，是一个假想的上限，用来感受各种手段省了多少；MLA 模型真实的 K 头宽 192（128 + 64），按 MHA 算会更大，这里保守按 128。

几个值得注意的现象：

- **小模型基本都是 GQA**。主线模型的 KV 配置与 Qwen3-0.6B 完全一样，每 token 112 KiB。
- **省缓存有三条路**：
  1. **少存头**：GQA / MQA，所有稠密小模型都在用；
  2. **每个位置存得更小**：MLA，DeepSeek、Kimi、GLM、Mistral 的旗舰 MoE 模型在用，61 层 × 576 个数，只有同头数 MHA 的 1/57；
  3. **少存层或少存位置**：Qwen3.5 只有 1/4 的层是全注意力（其余是 Gated DeltaNet 线性注意力，状态大小固定），gpt-oss 一半层只看最近 128 个 token。这是第 22、23 章的主题。
- **三条路可以叠加**：Qwen3.5 是"GQA + 混合线性"，Kimi K3 是"MLA + 混合线性"。Qwen3.5-397B 这样的四千亿参数模型，128K 上下文的 KV cache 只有 3.75 GiB，和主线这个 0.7B 模型 32K 时差不多。
- **线性注意力层也有状态**，只是不随长度增长。生产级计算器 `zero/tools/kv_cache_calc.py` 会单独估算它：例如 Qwen3.5-9B 的 24 层 Gated DeltaNet 按 `mamba_ssm_dtype: float32` 估约 50 MiB，与上下文长短无关（状态和卷积缓存的具体存法依实现而定，属于估算）。

## 4. MQA → GQA → MLA

### 4.1 回顾：少存几个头

第 10 章讲过：标准多头注意力（**MHA**）每个查询头都有自己的 K、V；**MQA**（Shazeer 2019）让所有查询头共用一组 K、V；**GQA**（Ainslie 等 2023）折中，分组共用。它们省缓存的办法都是**减少 KV 头数**，代价是 K、V 的表达能力变弱：所有共用一组 K、V 的查询头，看到的是完全一样的 key 和 value。

### 4.2 MLA：把 K、V 一起压成一个潜向量

DeepSeek-V2（2024）提出的**多头潜在注意力（MLA，Multi-head Latent Attention）**换了一个思路：不减少头数，而是注意到"每个头的 K、V 都是从同一个输入 x 线性变换来的"，那就先把 x 压到一个很小的**潜向量（latent）** c_KV，再从它还原出每个头的 K、V：

```
c_KV = RMSNorm(W_DKV · x)          # 下投影：7168 维 → 512 维（DeepSeek-V3）     ← 缓存
k_i^C = W_UK,i · c_KV              # 上投影：还原第 i 个头的 key（不带位置）
v_i   = W_UV,i · c_KV              # 上投影：还原第 i 个头的 value
```

缓存里只存 c_KV。这叫**低秩联合压缩**（low-rank joint compression）："低秩"是因为 W_UK · W_DKV 这个乘积的秩不超过 512；"联合"是因为 K 和 V 共用同一个潜向量。

直觉上可以这样理解：GQA 是"128 个头只许用 8 组 K/V"，硬性规定谁和谁共享；MLA 是"128 个头的 K/V 都必须能从同一个 512 维向量算出来"，共享的方式由训练自己学。每个头仍然有自己独立的 W_UK,i、W_UV,i，所以每个头看到的 key、value 各不相同。

对应 `03_mla.py`：

```python
self.wkv_a = nn.Linear(dim, kv_lora_rank + rope_dim, bias=False)            # 下投影：x → [c_KV ; k_R]
self.kv_norm = tiny.RMSNorm(kv_lora_rank)
self.wkv_b = nn.Linear(kv_lora_rank, n_heads * (nope_dim + v_dim), bias=False)  # 上投影 [W_UK; W_UV]
...
c_kv, k_pe = self.wkv_a(x).split([r, dr], dim=-1)
c_kv = self.kv_norm(c_kv)                        # 要缓存的潜向量
k_pe = tiny.apply_rope(k_pe[:, None], cs, sn)    # 要缓存的 RoPE key
```

### 4.3 为什么 RoPE 要"解耦"

RoPE（第 9 章）要在 q、k 上乘一个和位置有关的旋转矩阵。如果直接转在 k_i^C = W_UK,i · c_KV 上，下一节的"吸收"技巧就不成立了：旋转矩阵夹在 W_UQ 和 W_UK 中间，两者没法提前乘到一起（矩阵乘法不满足交换律），推理时只好把每个位置的 K 都还原出来再转，缓存潜向量就白省了。

DeepSeek 的解法是**解耦 RoPE（decoupled RoPE）**：每个头的 query 和 key 拆成两段，

```
q_i = [ q_i^C ; RoPE(q_i^R) ]      # 128 维不带位置 + 64 维带位置
k_i = [ k_i^C ; RoPE(k^R)   ]      # k^R 由 x 直接算出，所有头共享一个
```

带位置的那一小段 k^R（`qk_rope_head_dim` = 64）所有头共享，也要缓存。所以 MLA 每层每位置缓存 **512 + 64 = 576 个数**，这就是 DeepSeek-V3、Kimi K2、GLM-5、Mistral Large 3 的 `config.json` 里 `kv_lora_rank: 512`、`qk_rope_head_dim: 64` 的含义。同样 128 个头的 MHA（K 每头 192、V 每头 128）要存 128 × 320 = 40,960 个数，MLA 只有它的 1.4%。DeepSeek-V2 论文的写法是：MLA 的缓存相当于只有 2.25 组的 GQA。

### 4.4 吸收（absorb）：推理时不还原 K、V

还原 K、V 意味着每一步都要把缓存里全部 T 个潜向量乘上 W_UK、W_UV，很浪费。注意力分数可以换个顺序算：

```
q_iᵀ k_j = (q_i^C)ᵀ W_UK,i c_KV,j + (q_i^R)ᵀ k^R_j
         = (W_UK,iᵀ q_i^C)ᵀ c_KV,j + (q_i^R)ᵀ k^R_j      ← 先把 query 投进潜空间
Σ_j p_ij v_j = W_UV,i ( Σ_j p_ij c_KV,j )                  ← 先在潜空间里加权平均，最后再投回去
```

矩阵乘法满足结合律，所以 W_UK 可以"吸收"进 query 一侧，W_UV 可以吸收进输出一侧，缓存里的潜向量直接参与注意力，K、V 从头到尾不用还原。`03_mla.py` 两条路径都写了：

```python
if self.absorb:  # 吸收：query 投进潜空间，直接和缓存点积
    q_lat = torch.einsum("bhtd,hdr->bhtr", q_nope, w_uk)
    att = q_lat @ c_kv.transpose(-2, -1) + q_pe @ k_pe.transpose(-2, -1)
else:            # 显式：先把每个头的 K 还原出来
    k_nope = torch.einsum("bxsr,hdr->bhsd", c_kv, w_uk)
    ...
```

运行结果（同一组随机权重、float64）：

```
1. 吸收路径 = 显式路径（同一组权重、float64）
   最大差异 4.4e-16
2. 带缓存逐个喂 = 一次性整段前向
   最大差异 2.8e-16；缓存里存的形状：潜向量 (2, 1, 20, 32)，RoPE key (2, 1, 20, 16)
```

两条路径数学上完全等价，差异是浮点舍入。有意思的是吸收之后的样子：每个头都拿一个 576 维的 query 去和**同一份** 576 维的缓存做点积——形式上正是 MQA（所有头共用一组 K/V），只不过这组"K/V"是 576 维的潜向量。GLM-5 的报告直接把它叫作"MLA 的 MQA 模式"。所以 MLA 可以理解为：**训练时像 MHA（每个头有自己的 K、V），推理时像 MQA（只存、只读一份）**。

实际系统里，训练和 prefill（一次处理很多 token，算力是瓶颈）常用显式路径，decode（一次一个 token，带宽是瓶颈）用吸收路径。

### 4.5 MLA 的代价

MLA 不是免费的午餐，写作时能查到的几点：

- **decode 算力更高**：吸收之后每个头在 576 维上做点积，而 GQA 通常是 128 维。GLM-5 报告因此把头维从 192 加到 256、头数减少 1/3，以降低 decode 算力；Kimi K2 也把头数从 DeepSeek-V3 的 128 减到 64，报告说 128K 上下文时 128 个头比 64 个头的推理 FLOPs 多 83%。
- **质量结论依赖设置**：DeepSeek-V2 的消融里 MLA 比 MHA 还好（附录 D.2）；但 GLM-5 报告在 Muon 优化器下发现 576 维潜向量的 MLA 比不过 GQA-8，要改优化器的用法（"Muon Split"）才追平。
- **和一些组件不兼容**：Kimi K2 报告指出 QK-Norm 用不到 MLA 上，因为推理时 K 从来没被还原出来。主线模型用了 QK-Norm（第 9 章），这是它继续用 GQA 的原因之一。
- **工程更复杂**：需要专门的 kernel（如 DeepSeek 开源的 FlashMLA）和推理引擎支持。

这些也解释了为什么目前采用 MLA 的都是几百亿到上万亿参数的旗舰 MoE 模型，而 0.6–9B 的稠密小模型几乎清一色用 GQA。

## 5. 小实验：同一个小模型，五种注意力

`04_attention_variants.py` 用第 10 章的字符级莎士比亚小模型（4 层、宽 128、4 个查询头、head_dim 32），只换注意力：

- MHA（4 个 KV 头）、GQA（2 个）、MQA（1 个）；
- MLA-48：潜向量 48 维 + RoPE key 16 维 = 64 个数，**缓存和 MQA 一样大**；
- MLA-16：潜向量 16 维 + 16 = 32 个数，只有 MQA 的一半。

其余完全相同：同样的数据顺序、600 步、AdamW + warmup + cosine。每种跑 3 个随机种子。缓存大小是生成 512 个字符后**真实测得**的（FP32）：

<!-- EXP_TABLE -->

## 6. 小结

- **长上下文的两种贵**：prefill 的注意力算力按 T² 涨（主线模型 32K 时占 73%）；decode 被显存带宽卡住（算术强度只有 1–5），长上下文下 KV cache 比权重还大，拼批也省不掉。
- **账本**：按层记——全注意力和 MLA 层随长度线性涨，滑动窗口层封顶，线性注意力层是常数。
- **三条省缓存的路**：少存头（GQA/MQA）、每个位置存得更小（MLA）、少存层或位置（滑动窗口、混合线性），可以叠加。
- **MLA**：K、V 联合压缩成潜向量 c_KV（512 维）+ 共享 RoPE key（64 维），每层每位置 576 个数；解耦 RoPE 是为了能"吸收"；吸收后训练像 MHA、推理像 MQA。
- **代价**：decode 算力更高、和 QK-Norm 不兼容、需要专门 kernel；目前是大 MoE 模型的选择，小稠密模型仍用 GQA。

---

## 从极简到生产级

| 极简版（`code/`） | 生产级 | 多做了什么、为什么 |
|---|---|---|
| `01_kv_ledger.py`：手写的模型字典 + `layer_list` / `kv_bytes` | `zero/tools/kv_cache_calc.py`：`kv_cache_bytes(cfg, seq_len, batch, dtype_bytes)`、`kv_bytes_per_token`、`fixed_state_bytes`、`breakdown`；命令行 `uv run python -m zero.tools.kv_cache_calc configs/main/pretrain.toml --seq 32768` | 直接读 zero 的 `ModelConfig` / TOML、Hugging Face 的 `config.json`（含多模态模型的 `text_config` 嵌套、`layer_types`、Kimi 的 `linear_attn_config`、Mistral 原生 `params.json` 的字段名）；自动识别 MLA、滑动窗口、线性注意力层；单独估算线性层的固定状态；`--dtype-bytes 1` 可以算 FP8 KV cache |
| `03_mla.py` 的 `MLA`：接口对齐第 10 章小模型，借用它的 `torch.cat` 式缓存 | `zero/arch/mla.py`：`MLAConfig`（字段名与 DeepSeek-V3 的 config 一致）、`MLAAttention`（接口与 `zero.model.Attention` 相同，可直接换进 `Transformer`，见 `mla_transformer`）、`MLACache`（预分配潜向量与 RoPE key，`nbytes()`） | 支持 query 也做低秩压缩（`q_lora_rank`，DeepSeek-V3 为 1536，省训练激活、不省缓存）；有缓存时自动走吸收路径、无缓存时走显式路径 + SDPA；支持分块 prefill；softmax 至少在 float32 上算 |
| `04_attention_variants.py`：MHA/GQA/MQA/MLA 同配置对比 | 第二步可选：用 `mla_transformer(model_cfg, mla_cfg)` 在约 1 亿参数的 ladder 配置上重跑 | 生产级模块与主线 `Transformer` 共用 RMSNorm、SwiGLU、训练循环，换注意力只改一处 |
| 无 | 行业实现：vLLM 的 PagedAttention（第 10 章）按页管理 KV cache；DeepSeek 开源的 FlashMLA 是 MLA decode 的 GPU kernel；vLLM、SGLang 都有 MLA 后端 | 本课的 MLA 只追求可读和正确，**尚未在 GPU 上验证性能**；真正上线要用这些专门实现 |

**对拍**（`uv run pytest tests/test_kv_cache_calc.py tests/test_arch_mla.py`，本机 12 项全部通过，约 3 秒）：

- 计算器对 `configs/tiny`、`configs/main`，在 FP32 / BF16、batch 2 下，与 `KVCache.from_config(...).nbytes()` 真实分配的字节数完全相等；主线模型每 token 114,688 字节；
- MLA 配置的结果与 `MLACache.nbytes()` 相等；DeepSeek-V3 每 token 61 × 576 × 2 字节；滑动窗口、Qwen3.5 式混合线性、Mistral `params.json` 各有一个手算用例；
- `MLAAttention` 的吸收路径与显式路径在 float64 下差异 < 1e-10；分块 prefill（12 + 1 + 17 个 token）与一次性前向在 1e-5 内一致；带缓存与不带缓存的贪心生成 40 个 token 完全相同；换上 MLA 的 `Transformer` 能正常训练。

`01_kv_ledger.py` 第 3 部分也直接调用 `zero.tools.kv_cache_calc` 核对了主线模型和 DeepSeek-V3 的数字。

---

## 前沿观察

> **稀疏注意力（DeepSeek DSA）不省 KV cache，省的是算力。** DeepSeek-V3.2 和 GLM-5 在 MLA 之上加了 DeepSeek Sparse Attention：一个轻量的"索引器"给历史 token 打分，每个 query 只和得分最高的 2048 个（`index_topk: 2048`）做注意力。这把长上下文的注意力算力从 O(T²) 降到接近 O(T·k)，但 KV cache 仍要全部保留（索引器自己还要额外缓存一份小 key）。目前明确采用的只有 DeepSeek 和 GLM 两个家族，不满足 GOAL.md 2.1 的"至少 3 家"，第 22 章会再核实一次。

---

## 采用方与来源

| 技术 | 采用方（主力版本） | 来源 |
|---|---|---|
| GQA | Qwen3（0.6B：16 Q / 8 KV；8B：32 / 8）、Llama 3.1（32 / 8）、gpt-oss（64 / 8）、Qwen3.5 的全注意力层（0.8B：8 / 2）；更多见第 10 章 | 各模型 `config.json`（下方链接）；Qwen3、Llama 3 技术报告 |
| MLA | **DeepSeek**（V2 提出；V3、V3.2：`kv_lora_rank` 512、`qk_rope_head_dim` 64）、**Kimi**（K2：技术报告 2.3 节"employing MLA"；K3 的全注意力层）、**GLM**（GLM-5：技术报告 2.1 节"Multi-latent Attention"）、**Mistral**（Mistral Large 3：`params.json` 中 `kv_lora_rank` 512、`qk_rope_head_dim` 64，与 DeepSeek-V3 同形；模型卡未用文字说明） | DeepSeek-V2 arXiv:2405.04434；DeepSeek-V3 arXiv:2412.19437；Kimi K2 arXiv:2507.20534；GLM-5 arXiv:2602.15763；各模型配置 |
| 滑动窗口 / 局部-全局交替 | gpt-oss（`layer_types` 交替、`sliding_window` 128）；更多见第 22 章 | gpt-oss 的 `config.json` |
| 混合线性注意力 | Qwen3.5（`full_attention_interval` 4，其余为 `linear_attention`）、Kimi K3（`linear_attn_config`：24 层全注意力 + 69 层 KDA）；更多见第 23 章 | 各模型 `config.json` |
| KV cache 分页管理 / MLA kernel | vLLM（PagedAttention，行业标准）；FlashMLA（DeepSeek 开源） | Kwon 等 2023，arXiv:2309.06180；<https://github.com/deepseek-ai/FlashMLA> |

**共识判断（GOAL.md 2.1）**：MLA 被 DeepSeek、Kimi、GLM、Mistral 四个彼此独立的头部家族在主力版本中采用，满足规则 A，进正文；但要说明它的适用范围——四家全是几百亿到上万亿参数的 MoE 旗舰，**稠密小模型没有采用**（Qwen3、Qwen3.5、Llama 3、Gemma 3、gpt-oss、SmolLM3 均为 GQA 或 MQA），所以主线模型不用它（GOAL.md 3.3）。MQA 作为 GQA 的端点讲（规则 C），头部家族里只查到 Gemma 3 1B（见第 10 章）。

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

说明：Meta 官方的 `meta-llama/Llama-3.1-8B` 需要申请权限，本章读取不到，数字取自 unsloth 镜像（其 `_name_or_path` 指向官方仓库），与第 10 章一致。Kimi K3 的 `full_attn_layers` 层号从 1 开始编号（1–93），本章按此解读为 24 层 MLA、69 层 KDA。H100 SXM 的 989.5 TFLOPS 与 `zero/tools/estimate_cost.py` 同一口径；3.35 TB/s、80 GB 取自 NVIDIA H100 产品页 <https://www.nvidia.com/en-us/data-center/h100/>。

---

## 引导问题

带着这些问题去问 Claude Code，直到你能用自己的话讲清楚：

1. 第 1.2 节说"长上下文下拼批省不掉 KV cache 的读取"。那为什么拼批还能省下权重的读取？试着估算：主线模型在 32K 上下文下，batch 多大时"读 KV cache"的时间是"读权重"的 10 倍？
2. MLA 的潜向量是 512 维，而 DeepSeek-V3 的隐藏层是 7168 维。如果把潜向量加大到 7168 维，还省缓存吗？还有意义吗？"低秩"这个限制在这里起了什么作用？
3. 为什么 k^R（带 RoPE 的那段 key）要所有头共享一个，而不是每个头一个？如果每个头一个，缓存会变成多少？
4. 吸收之后，MLA 在 decode 时"形式上就是 MQA"。那它和真正的 MQA（第 10 章，1 个 KV 头、head_dim 32）区别在哪里？为什么 MLA 的质量可以比 MQA 好得多？
5. Qwen3.5 和 Kimi K3 都是"少量全注意力 + 大量线性注意力"。如果全注意力层也换成 MLA（像 Kimi K3 那样），账本会怎样变？Qwen3.5-9B 如果这样改，128K 的 KV cache 大概是多少？
6. 本章小实验里各方案的 loss 差距和种子之间的差距是什么关系？如果想认真比较 MLA 和 GQA，你会怎么设计实验？参考 GLM-5 报告表 1 的做法。

## 动手任务

每个任务都要真的运行代码、看到结果。

**任务 1（基础）**：在 Hugging Face 上挑一个本章没算过的模型（例如 SmolLM3-3B、Gemma 3 系列、MiniMax 或 GLM 的其他版本），读它的 `config.json`，把字段加进 `01_kv_ledger.py` 的 `MODELS`，算出 32K 和 128K 的 KV cache；再用 `uv run python -m zero.tools.kv_cache_calc 你保存的config.json --seq 131072` 对拍。注意它有没有滑动窗口、`layer_types`。

**任务 2（核心）**：在 `03_mla.py` 里给 `MLA` 加一个计时实验：随机初始化一个 `n_heads=16, kv_lora_rank=64, rope_dim=16` 的 MLA，先 prefill 1024 个位置，再分别用吸收路径和显式路径 decode 64 步，比较每步耗时；再把缓存长度换成 256、4096 看趋势。解释为什么缓存越长，吸收路径的优势越大。

**任务 3（挑战）**：在 `04_attention_variants.py` 里加一个 "GQA-1 × head_dim 48" 之类的变体，让它的每层每位置缓存也是 64 个数（和 MQA、MLA-48 一样大），然后跑 3 个种子。同样大小的缓存下，哪种结构的 loss 更低？差距是否超出了种子之间的波动？再试着把训练步数加倍，看结论变不变。

---

## 想深入：CS336

本章对应斯坦福 CS336（Spring 2026）<https://cs336.stanford.edu/>：

- **第 10 讲：推理**。从资源核算的角度讲推理的开销：prefill 与 decode、KV cache 的显存与内存带宽，以及让推理更快更省的各类办法（讲义与录像见课程页；本构建环境访问不了课程页，具体覆盖哪些架构手段以讲义为准）。本章的 `02_prefill_decode.py` 可以看作这种资源核算思路在主线模型上的一次具体演算。

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

**下一章**：账本里还有两条路没走完——让一些层只看最近的一段（滑动窗口），或者干脆不存 K、V（线性注意力）。第 22 章先讲局部与稀疏注意力：gpt-oss 为什么一半层只看 128 个 token，Gemma 为什么大部分层是局部的，模型又是怎么在"只看附近"的同时不丢掉远处的信息。
