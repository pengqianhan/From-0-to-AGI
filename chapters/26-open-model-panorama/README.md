# 第 26 章：当前最先进开源模型全景 —— 把整门课的架构放进一棵树

> **一句话目标**：读完这一章，你能拿到任意一个开源旗舰模型的 `config.json`，逐层说出它由哪几种注意力组成、FFN 怎么稀疏、有没有 MTP，算出它的总参数、激活参数和 128K 上下文的 KV cache，并在本章的演化树上指出它的每一个零件来自哪一章、哪些已是共识、哪些还在分化；还能说清楚为什么我们的主线模型只用稠密共识块，以及它的"下一版"最值得加什么。

📺 **本章视频**：待发布（本地渲染：`bash chapters/26-open-model-panorama/video/build.sh`）
🧪 **本章自检**：学完后在 Claude Code 里输入 `/ch26-panorama`

---

上一章我们讲完了 MTP 与推测解码，第五部分的五个零件——KV cache 的账本、局部与稀疏注意力、线性注意力混合、MoE、MTP——都已经在 CPU 上拆开、装回、测过。这一章要解决的问题是：**把这些零件装回真实的模型里，现在（2026 年 9 月）最强的开源模型到底长什么样？哪些做法已经人人都用，哪些还在各走各路？** 我们先确认"最新"是谁，然后画一棵从 GPT-2 长出来的演化树，再拆四个代表模型，把它们和我们的主线模型、标杆 Qwen3.5-0.8B 放进同一张表。最后回头看整门课：从 `y = ax + b` 一路走到这里，每一步都在这棵树上。

本章代码（都在 CPU 上跑，几秒到二十秒）：

```bash
uv run python chapters/26-open-model-panorama/code/01_panorama.py        # 拆解 + 大对比表 + 采用矩阵 + 主线"下一版"的账
uv run python chapters/26-open-model-panorama/code/02_evolution_tree.py  # 架构演化树（视频也用它）
uv run python chapters/26-open-model-panorama/code/03_meta_params.py     # 不下载权重，用 HF 官方模型类数参数
```

所有模型数字都来自 [`code/models.json`](code/models.json)：里面每个字段都是从各模型 Hugging Face 仓库的 `config.json`、模型卡或技术报告里**逐项抄下来**的（2026-09-26 读取），每个模型都带着来源链接。没读到的就不写。

## 1. 先确认"最新"是谁

"最先进"的名单变得很快，凭记忆写一定会错。本章的做法是：在 Hugging Face 上按创建时间列出几家头部机构的仓库（`deepseek-ai`、`Qwen`、`moonshotai`、`openai`、`zai-org`、`MiniMaxAI`），看每家最新的旗舰是什么。2026-09-26 的结果：

| 家族 | 本章用的旗舰 | 同代还有 | 本章角色 |
|---|---|---|---|
| DeepSeek | **DeepSeek-V4-Pro**（1.6T 总参数 / 49B 激活，MIT） | V4-Flash（284B / 13B）、V4-Pro-0813 与 V4-Flash-0731 两次更新、带视觉的 V4.1-Flash | 拆解 |
| 千问 | **Qwen3.8-2.4T-A95B**（2.4T / 95B） | Qwen3.8-27B（稠密）、Qwen3.8-Flash-Next（`model_type: qwen4_exp`，实验结构） | 拆解 |
| Kimi | **Kimi-K3**（2.8T / 104B） | Kimi-K2.6、K2.7-Code | 拆解 |
| OpenAI | **gpt-oss-120b**（117B / 5.1B，Apache-2.0） | gpt-oss-20b；截至 2026-09 没有新一代开放权重 | 拆解（最"保守"的一个） |
| 智谱 GLM | **GLM-5.3**（与 GLM-5.2 同底座） | GLM-5.3-Flash | 同代对照 |
| MiniMax | **MiniMax-M3**（约 428B / 约 23B） | M2.7 等 | 同代对照 |

挑四个拆解：DeepSeek、千问、Kimi 三家代表了 2026 年三条不同的长上下文路线，gpt-oss 则代表"只用最稳的零件"。GLM-5.3 和 MiniMax-M3 放进对比表，用来数"几家在用"。

## 2. 一棵树：整门课的架构演化

运行 `02_evolution_tree.py`，打印出整门课的架构演化树。"在用"那一栏不是手写的，是程序从 `models.json` 里各模型的 config 字段读出来的：

```
GPT-2  [起点，第 8 章]
└─ 现代稠密块（Llama 式）  [起点，第 9 章]
   ├─ Pre-Norm + RMSNorm  [共识，第 6、9 章]｜在用：6 个旗舰全部 + 两个千问小模型
   ├─ RoPE（+ YaRN 扩长）  [共识，第 9、15 章]｜在用：除 Kimi-K3 外的 5 个旗舰
   ├─ SwiGLU  [共识，第 9 章]｜在用：6 个旗舰全部
   ├─ GQA  [共识，第 10 章]｜在用：Qwen3.8、gpt-oss、MiniMax-M3
   │  └─ MLA：K/V 压成潜向量  [共识，第 21 章]｜在用：Kimi-K3、GLM-5.3
   │     └─ CSA/HCA：按 token 压缩的共享 KV  [前沿观察，本章]｜在用：DeepSeek-V4-Pro
   ├─ QK-Norm  [共识，第 9 章]｜在用：DeepSeek-V4-Pro、Qwen3.8、MiniMax-M3
   ├─ 共享输入输出 embedding（小模型）  [共识，第 9 章]｜在用：Qwen3-0.6B、Qwen3.5-0.8B
   ├─ MoE：细粒度 + 共享专家  [共识，第 24 章]｜在用：6 个旗舰全部
   │  ├─ 无辅助损失均衡  [共识，第 24 章]｜在用：DeepSeek-V4-Pro、Kimi-K3、GLM-5.3、MiniMax-M3
   │  └─ Latent MoE（专家在潜空间里算）  [前沿观察，本章]｜在用：Kimi-K3
   ├─ 滑动窗口 / 局部-全局交替  [共识，第 22 章]｜在用：DeepSeek-V4-Pro、gpt-oss-120b
   │  └─ 稀疏注意力（按内容挑 top-k）  [新晋共识（2026 年才达到规则 A），第 22 章、本章]｜在用：DeepSeek-V4-Pro、GLM-5.3、MiniMax-M3
   ├─ 混合线性注意力（约 3:1）  [共识，第 23 章]｜在用：Qwen3.8、Kimi-K3、Qwen3.5-0.8B
   │  └─ KDA（逐通道门控）  [前沿观察，第 23 章]｜在用：Kimi-K3
   ├─ MTP（多预测一个 token）  [共识，第 25 章]｜在用：DeepSeek-V4-Pro、Qwen3.8、GLM-5.3、MiniMax-M3、Qwen3.5-0.8B
   └─ 残差流改造：mHC / AttnRes  [前沿观察，本章]｜在用：DeepSeek-V4-Pro、Kimi-K3
```

（输出里模型名是全称，这里为了排版做了缩写。）

读这棵树的方法：

- **树根是 GPT-2（2019）**。第 8 章的多头注意力、学出来的绝对位置、GELU 的 MLP、LayerNorm，都是它的样子。它的 KV cache 每个 token 36 KiB，最长 1024 个 token。
- **主干是第 9 章的现代稠密块**：Pre-Norm + RMSNorm、RoPE、SwiGLU、GQA、QK-Norm、小模型共享 embedding、不加 bias。今天所有旗舰都还长在这根主干上——6 家全部用 RMSNorm 和门控 FFN，5 家用 RoPE。**我们的主线模型就是这根主干本身**，和 Qwen3-0.6B 同形。
- **六根分枝**，每一根都在回答同一个问题"怎样更大、更长、更便宜"：
  1. **前馈层变稀疏**（第 24 章）：MoE，6 个旗舰全部采用，其中 5 家带共享专家；
  2. **每个位置少存点**（第 10、21 章）：GQA → MLA → DeepSeek-V4 的压缩注意力；
  3. **只看一部分位置**（第 22 章）：滑动窗口按位置挑，稀疏注意力按内容挑；
  4. **干脆不存 K/V**（第 23 章）：线性注意力层 + 少量全注意力层的混合；
  5. **训练目标更密**（第 25 章）：MTP，顺便当推测解码的草稿；
  6. **残差流本身**（2026 年才出现）：DeepSeek 的 mHC、Kimi 的 AttnRes，还在前沿。

## 3. 拆解四个模型

`01_panorama.py` 第 1 部分把每个模型的层画成一行字母（G 全注意力、s 滑动窗口、l 线性注意力、M MLA、c CSA、h HCA、g GQA+块稀疏）：

```
DeepSeek-V4-Pro         61 层：31 HCA + 30 CSA
                           hhchchchchchchchchchchchchchchchchchchchchchchchchchchchchchc
Qwen3.8-2.4T-A95B       92 层：69 GDN + 23 GQA
                           lllGlllGlllG……（23 组）
Kimi-K3                 93 层：69 KDA + 24 MLA
                           lllMlllMlllM……lllMM
gpt-oss-120b            36 层：18 SWA + 18 GQA
                           sGsGsGsGsGsGsGsGsGsGsGsGsGsGsGsGsGsG
主线模型（本课）          28 层：28 GQA
```

同样是"让长上下文变便宜"，四家给出了四种层的排列。

### 3.1 DeepSeek-V4-Pro：离开了自己发明的 MLA

DeepSeek 在 V2 提出了 MLA（第 21 章），V3、V3.2 一直在用。到了 V4，它**换掉了 MLA**。按 V4 技术报告第 2.3 节和 `config.json`：

- **共享 K=V 的 MQA**：`num_key_value_heads: 1`、`head_dim: 512`，每个位置只存一个 512 维向量，它既当 key 又当 value；128 个查询头都去读它。
- **两种压缩注意力交替**（`compress_ratios`）：前两层是 **HCA**（Heavily Compressed Attention），把每 128 个 token 的 KV 压成 1 条，然后对全部压缩条目做稠密注意力；之后 CSA 与 HCA 交替。**CSA**（Compressed Sparse Attention）每 4 个 token 压成 1 条，再用 V3.2 的"闪电索引器"从压缩条目里挑出 top-1024 条（`index_topk: 1024`）做注意力——这就是第 22 章"按内容挑"的稀疏注意力，只不过挑的对象是压缩块。数出来是 30 层 CSA、31 层 HCA。
- **每层还有一个 128 个 token 的滑动窗口分支**（`sliding_window: 128`），补上"压缩块内部看不见最近几个 token"的缺口；还有 attention sink、query 与 KV 条目上的 RMSNorm（相当于 QK-Norm）、只转最后 64 维的 RoPE。
- **FFN**：384 个路由专家选 6 个 + 1 个共享专家，前 3 层用 Hash 路由（`num_hash_layers: 3`），打分函数是 `sqrtsoftplus`，无辅助损失均衡（`topk_method: noaux_tc`）；1 层 MTP。
- **残差流**：mHC（`hc_mult: 4`），把残差流扩成 4 路，层与层之间用一个被约束成"双随机矩阵"的混合矩阵连接（报告 2.2 节）。
- **精度与优化器**：路由专家用 FP4 发布，其余多为 FP8；预训练用 Muon 优化器（模型卡）。

KV cache 怎么记账？把 CSA 层摊到每个原始 token 是 512 / 4 = 128 个数，HCA 层是 512 / 128 = 4 个数，滑动窗口分支封顶 128 个位置。`01_panorama.py` 里的 `dsv4_layout` 直接用生产级 `zero.tools.kv_cache_calc` 的 `KVLayout` / `LayerSpec` 搭出这本账：

```python
for r in c["compress_ratios"][: c["num_hidden_layers"]]:
    if r:                                   # 压缩层：每个 token 摊到 head_dim / r 个数
        layers.append(LayerSpec("full", hd // r))
        if with_indexer and r == 4:         # CSA 层的索引器也要存压缩 key
            layers.append(LayerSpec("full", c["index_head_dim"] // r))
    layers.append(LayerSpec("sliding", hd, window=win))   # 128 窗口的滑动分支（K=V 只存一份）
```

结果：每 token 7.7 KiB，128K 上下文 998.62 MiB——而 V3.2 是 68.6 KiB 和 8.58 GiB。报告说 1M 上下文时 V4-Pro 的 KV cache 只有 V3.2 的 10%；用同一精度（BF16）、两边都算上索引器的 key，我们的账是 **11.5%**（V4-Pro 9.62 GiB 对 V3.2 83.88 GiB），同一个量级。差的那一点来自报告里 KV 条目除 RoPE 维外都用 FP8 存，以及压缩的精确取整。

### 3.2 Qwen3.8-2.4T-A95B：把 0.8B 的结构放大到 2.4T

千问这一代最大的模型，`architectures` 写的是 `Qwen3_5MoeForCausalLM`——**和 Qwen3.5 是同一个架构类**，模型卡也说"Built on the architectural foundation of Qwen3.5"。它和我们的标杆 Qwen3.5-0.8B 是同一套图纸，只是尺寸差了三千倍：

| | Qwen3.5-0.8B | Qwen3.8-2.4T-A95B |
|---|---|---|
| 层 | 24 = 6 ×（3 GDN + 1 全注意力） | 92 = 23 ×（3 GDN + 1 全注意力） |
| 全注意力 | GQA 8 Q / 2 KV，head_dim 256，输出门控 | GQA 64 Q / 4 KV，head_dim 256，输出门控 |
| 线性层（Gated DeltaNet） | 16 个 key 头、16 个 value 头 | 16 个 key 头、128 个 value 头 |
| FFN | 稠密 SwiGLU | 512 专家选 10 + 1 共享；辅助损失 0.001 |
| RoPE | 只转 25% 的维度，θ = 10⁷ | 同 |
| MTP | 1 层 | 1 层（模型卡："trained with multiple steps"） |
| 词表 / 共享 embedding | 248,320 / 是 | 248,320 / 否 |
| 上下文 | 262,144 | 262,144（模型卡：可扩到 1,010,000） |

混合线性注意力让 92 层里只有 23 层需要随长度增长的 KV：每 token 92 KiB，128K 为 11.50 GiB，另有线性层的固定状态约 568 MiB（估算）。注意它的负载均衡用的是**辅助损失**（`router_aux_loss_coef: 0.001`），不是 DeepSeek 式的偏置——第 24 章说过，这两条路都在大模型上成功了。

### 3.3 Kimi-K3：混合线性注意力 + MLA + 更稀疏的 MoE

K3 的 93 层里，第 4、8、……、92 层和最后的第 93 层是**带门控的 MLA**（24 层），其余 69 层是 **KDA**（Kimi Delta Attention，第 23 章：Gated DeltaNet 的逐通道门控版）。这是第 21 章"三条路可以叠加"的例子：MLA 让全注意力层每个位置只存 576 个数，混合线性让只有 1/4 的层需要存。每 token 27 KiB，128K 为 3.38 GiB。

另外几处值得注意：

- **全注意力层不加位置编码**（`mla_use_nope: true`）。Kimi Linear 报告的做法是 MLA 层用 NoPE，位置信息全部交给 KDA 层；这是 6 个旗舰里唯一不用 RoPE 的。
- **更稀疏的 MoE**：896 个专家选 16 个 + 2 个共享专家。模型卡写着"Latent MoE Dimension 3584"：路由专家不在 7168 维的隐藏层上算，而是先投到 3584 维的潜空间。`03_meta_params.py` 手算了路由专家这一块：在潜空间里算是 2.72T，若直接在 7168 维上算会是 5.45T——模型卡的 2.8T 总参数只和前者对得上。
- **新激活函数**：`hidden_act: situ`（模型卡称 SiTU-GLU），仍是门控 FFN 的一种。
- **残差流**：Attention Residuals（`attn_res_block_size: 12`）。
- **没有 MTP**（`num_nextn_predict_layers: 0`），与 K2 相同。
- MXFP4 权重、量化感知训练（模型卡）。

### 3.4 gpt-oss-120b：只用最稳的零件

gpt-oss 是 2025 年 8 月发布的，本章另外五家都比它新，但 OpenAI 至今没有发布新一代开放权重，它仍是这个家族的最新版。它几乎就是"共识块 + MoE + 滑动窗口"的教科书：

- 36 层，滑动窗口（128 个 token）与全注意力 1:1 交替；GQA 64 Q / 8 KV，head_dim 64；每个头一个可学习的 attention sink；
- 128 个专家选 4 个，没有共享专家，用辅助损失（`router_aux_loss_coef: 0.9`）；
- RoPE θ = 150,000，YaRN 把 4K 扩 32 倍到 128K；
- SwiGLU 带截断（`swiglu_limit: 7.0`）；MoE 权重以 MXFP4 发布，单张 80 GB 卡放得下；
- 没有 MLA、没有线性注意力、没有 MTP。

每 token 36 KiB（只有 18 层全注意力层在涨），128K 为 4.50 GiB。

### 3.5 同代另外两家

- **GLM-5.3**：78 层全部是 MLA（512 + 64）+ DeepSeek 式稀疏注意力（`index_topk: 2048`）。GLM-5.2 引入的 IndexShare 让每 4 层稀疏注意力共用一个索引器（config 的 `indexer_types`：21 层 `full`、57 层 `shared`）。256 专家选 8 + 1 共享，无辅助损失均衡，1 层 MTP。每 token 87.8 KiB。
- **MiniMax-M3**：MiniMax-M2 是全注意力（第 23 章），M3 改成 GQA（64 Q / 4 KV）+ **块稀疏注意力 MSA**：前 3 层普通注意力，后 57 层把历史按 128 个 token 分块、每个 query 挑 16 块（`sparse_block_size: 128`、`sparse_topk_blocks: 16`）。模型卡称 1M 上下文时 prefill 快 9 倍、decode 快 15 倍（相对 M2）。块稀疏**不减少 KV cache**：每 token 仍是 120 KiB，128K 为 15.00 GiB，是本章旗舰里最大的。

## 4. 同一张表

`01_panorama.py` 第 2 部分。KV cache 一律用 BF16、batch 1，只算随长度增长的部分；"每 token KV"取长上下文下每多一个 token 增加的量（滑动窗口层早就填满了，不再计入）：

| 模型 | 总参数 | 激活 | 层 | 层的构成 | 词表 | 上下文 | 共享 emb | 每 token KV | 32K KV | 128K KV | 线性层状态 |
|---|---:|---:|---:|---|---:|---:|---|---:|---:|---:|---:|
| GPT-2 | 0.12B | 稠密 | 12 | 12 MHA | 50,257 | 1K | 是 | 36.0 KiB | — | — | — |
| Qwen3-0.6B | 0.60B | 稠密 | 28 | 28 GQA | 151,936 | 40K | 是 | 112.0 KiB | 3.50 GiB | — | — |
| **主线模型（本课）** | **689.5M** | **稠密** | **28** | **28 GQA** | **65,536** | **32K** | **是** | **112.0 KiB** | **3.50 GiB** | — | — |
| Qwen3.5-0.8B（标杆） | 0.75B | 稠密 | 24 | 18 GDN + 6 GQA | 248,320 | 256K | 是 | 12.0 KiB | 384.00 MiB | 1.50 GiB | 19.27 MiB |
| gpt-oss-120b | 117B | 5.1B | 36 | 18 SWA + 18 GQA | 201,088 | 128K | 否 | 36.0 KiB | 1.13 GiB | 4.50 GiB | — |
| DeepSeek-V3.2 | 671B | 37B | 61 | 61 MLA+DSA | 129,280 | 160K | 否 | 68.6 KiB | 2.14 GiB | 8.58 GiB | — |
| DeepSeek-V4-Pro | 1.6T | 49B | 61 | 31 HCA + 30 CSA | 129,280 | 1M | 否 | 7.7 KiB | 255.38 MiB | 998.62 MiB | — |
| Qwen3.8-2.4T-A95B | 2.4T | 95B | 92 | 69 GDN + 23 GQA | 248,320 | 256K | 否 | 92.0 KiB | 2.88 GiB | 11.50 GiB | 568.17 MiB |
| Kimi-K3 | 2.8T | 104B | 93 | 69 KDA + 24 MLA | 163,840 | 1M | 否 | 27.0 KiB | 864.00 MiB | 3.38 GiB | 221.55 MiB |
| GLM-5.3 | 744B | 40B | 78 | 78 MLA+DSA | 154,880 | 1M | 否 | 87.8 KiB | 2.74 GiB | 10.97 GiB | — |
| MiniMax-M3 | ~428B | ~23B | 60 | 3 GQA + 57 GQA+块稀疏 | 200,064 | 1M | 否 | 120.0 KiB | 3.75 GiB | 15.00 GiB | — |

说明：旗舰的总参数 / 激活参数取模型卡；稠密小模型的总参数由 `03_meta_params.py` 数出；GLM-5.3 的模型卡没写参数量，它的 config 形状与 GLM-5 完全一致，按 GLM-5 模型卡记为 744B / 40B。"—"表示超出该模型的上下文（主线模型按长上下文阶段的 32K 记）。线性层状态是估算（按 `mamba_ssm_dtype: float32`，写法因实现而异）。

**参数量是怎么核对的**。`03_meta_params.py` 把 config 交给 Hugging Face transformers 里各家架构的**官方模型类**（`DeepseekV4ForCausalLM`、`Qwen3_5MoeForCausalLM`、`GlmMoeDsaForCausalLM`……），在 `meta` 设备上建模型——只有形状，不分配内存，2.4T 的模型一秒钟建完——然后数参数：

```python
with torch.device("meta"):
    model = AutoModelForCausalLM.from_config(cfg)
total = sum(p.numel() for p in model.parameters())
experts = sum(p.numel() for n, p in model.named_parameters() if ".experts." in n)
active = total - experts * (1 - k / E)      # 每个 token 只用到 E 个路由专家里的 k 个
```

| 模型 | 数出来的总参数 | 激活（含全部） | 不含输入 embedding | 不含整个词表 | 模型卡 |
|---|---:|---:|---:|---:|---|
| gpt-oss-120b | 116.829B | 5.71B | **5.13B** | 4.55B | 117B / 5.1B |
| DeepSeek-V3.2 | 671.878B | 38.40B | 37.48B | 36.55B | 671B / 37B |
| DeepSeek-V4-Pro | 1572.997B | **49.78B** | 48.85B | 47.93B | 1.6T / 49B |
| Qwen3.8-2.4T-A95B | 2419.805B | **95.29B** | 93.25B | 91.22B | 2.4T / 95B |
| GLM-5.3 | 743.377B | 41.25B | **40.30B** | 39.35B | 744B / 40B |
| MiniMax-M3 | 426.175B | 25.96B | 24.73B | **23.50B** | ~428B / ~23B |
| 主线模型 | 0.6895B（zero 公式 = HF `Qwen3ForCausalLM`） | 稠密 | | | |

总参数都和模型卡对得上（DeepSeek 的 HF 类不含 MTP 层，所以略少）。有意思的是**激活参数没有统一口径**：OpenAI 的 5.1B 不算输入 embedding，MiniMax 的约 23B 连输出头也不算，千问的 95B 和 DeepSeek 的 49B 则把 embedding 都算进去了（加粗的是和模型卡最接近的口径）。比较不同家的"激活参数"时，差个一两成可能只是口径不同。

几个值得停下来看的地方：

- **KV cache 和参数量几乎无关了**。2.8T 的 Kimi-K3 在 128K 时的 KV（3.38 GiB）比 0.7B 的主线模型在 32K 时（3.50 GiB）还小；1.6T 的 DeepSeek-V4-Pro 不到 1 GiB。决定 KV 的是**有几层存、每层每个位置存多少**，这正是第五部分的全部内容。
- **六个旗舰没有一个"每层都是普通全注意力"**：要么一半层滑动窗口（gpt-oss），要么 3/4 层线性（千问、Kimi），要么压缩（DeepSeek、MLA 的 GLM），要么稀疏（MiniMax 的稀疏省算力、不省 KV）。
- **小模型这一侧**，Qwen3.5-0.8B 用 3:1 混合把 32K 的 KV 压到 384 MiB，是主线模型（3.50 GiB）的九分之一。这是主线模型"下一版"最有说服力的参照（第 6 节）。
- **词表越来越大**：旗舰在 13 万到 25 万之间；Qwen3.5-0.8B 用了 24.8 万词表还共享 embedding，光词表就占它 7.5 亿参数的三分之一。主线模型的 65,536 是第 13 章为小模型定的折中。

## 5. 什么已成共识，什么还在分化

`01_panorama.py` 第 3 部分从 6 个旗舰的 config 里读出每项技术的采用情况，再补上前几章已经核实过的其他家族，按 GOAL.md 2.1 的规则 A（至少 3 个独立的头部家族）判定：

```
技术                     DS-V4  Qwen3.8   K3  gpt-oss GLM-5.3  M3   家数  前几章核实过的其他采用方 → 判定
RMSNorm 前置               ●      ●      ●      ●      ●      ●     6  ≥3 家，共识
RoPE                       ●      ●      ·      ●      ●      ●     5  ≥3 家，共识
YaRN                       ●      ·      ·      ●      ·      ·     2  ＋Kimi K2、Qwen3、SmolLM3（第 15 章） → 共识
SwiGLU/GLU                 ●      ●      ●      ●      ●      ●     6  ≥3 家，共识
QK-Norm                    ●      ●      ·      ·      ·      ●     3  ≥3 家，共识
滑动窗口                    ●      ·      ·      ●      ·      ·     2  ＋Gemma 3、OLMo 3（第 22 章） → 共识
稀疏注意力                  ●      ·      ·      ·      ●      ●     3  ＋美团 LongCat-2.0（第 22 章） → 共识（2026 新晋）
MoE                        ●      ●      ●      ●      ●      ●     6  ≥3 家，共识
共享专家                    ●      ●      ●      ·      ●      ●     5  ≥3 家，共识
无辅助损失均衡               ●      ·      ●      ·      ●      ●     4  ≥3 家，共识
SwiGLU 截断                ●      ·      ·      ●      ·      ●     3  ≥3 家
MTP                        ●      ●      ·      ·      ●      ●     4  ≥3 家，共识
GQA                        ·      ●      ·      ●      ·      ●     3  ≥3 家，共识
混合线性注意力               ·      ●      ●      ·      ·      ·     2  ＋NVIDIA Nemotron 3（第 23 章） → 共识
MLA                        ·      ·      ●      ·      ●      ·     2  ＋Mistral Large 3（第 21 章） → 共识
压缩注意力 CSA/HCA、MQA 共享 K=V、mHC、AttnRes、NoPE   各 1 家 → 前沿观察
```

（为排版删了几行，完整输出见运行结果。）

把结果分成三类：

**一、稳如磐石的主干。** RMSNorm 前置、门控 FFN（SwiGLU 及其变体）、MoE 在 6 家里是 6/6；RoPE 5/6（唯一的例外 Kimi-K3 是因为它把位置交给了线性层）；共享专家 5/6；MTP 4/6；无辅助损失均衡 4/6。第 9 章的共识块加上第 24、25 章的 MoE 和 MTP，就是 2026 年旗舰的"默认配置"。

**二、"让长上下文变便宜"已经分成三派，每派都过了 3 家的门槛。**

- **混合线性注意力派**：千问（Gated DeltaNet 3:1）、Kimi（KDA 约 3:1）、NVIDIA Nemotron 3（Mamba-2 混合，第 23 章）；
- **稀疏注意力派**：DeepSeek（V3.2 的 DSA → V4 的 CSA）、GLM（DSA + IndexShare）、MiniMax（M3 的 MSA 块稀疏）。第 21 章写作时只核实到 DeepSeek 和 GLM 两家；2026 年 MiniMax-M3 加入（第 22 章另核实了美团 LongCat-2.0），只看本章的 6 个旗舰就已满 3 家，所以树上标成"新晋共识"，与第 22 章的判定一致。但各家的做法并不一样——DSA 按单个 token 挑 top-k，CSA 先压缩再挑，MSA 按 128 个 token 的块挑——**"按内容挑"这个思想已是共识，具体做法还没收敛**，而且没有一家小模型在用；
- **少存 / 压缩派**：MLA（Kimi-K3、GLM-5.3，加上第 21 章的 Mistral Large 3）、滑动窗口（gpt-oss、DeepSeek-V4 的滑动分支，加上第 22 章的 Gemma 3、OLMo 3）。

要注意 MLA 的处境：它的发明者 DeepSeek 在 V4 里换成了压缩注意力 + 共享 K=V 的 MQA。MLA 按计数仍满足规则 A，但它已经不再是"越来越多人用"的那种共识了。

**三、按计数达标、但不值得单独讲的小技巧。** SwiGLU 截断（DeepSeek-V4 为 10、gpt-oss 与 MiniMax-M3 为 7）恰好 3 家，是防止激活值离群的稳定性技巧；另外还有两个旗舰之外的观察：**低精度发布**（gpt-oss 的 MXFP4、DeepSeek-V4 的 FP4 专家、Kimi-K3 的 MXFP4 量化感知训练，都写在模型卡里）和 **Muon 优化器**（DeepSeek-V4 模型卡、Kimi Linear 报告的 MuonClip、GLM-5 报告，见第 21 章）。它们分别属于第 20 章（量化）和第 12 章（优化器）的话题，这里只记下来。

**还在分化的**（各只有 1–2 家，放进文末"前沿观察"）：DeepSeek-V4 的 CSA/HCA 压缩注意力与共享 K=V 的 MQA、残差流改造（mHC、AttnRes）、Latent MoE、KDA 的逐通道门控、全注意力层用 NoPE、n-gram 嵌入表。

**还有一个趋势**：几乎所有人都只对一部分维度做 RoPE。千问转 25%（`partial_rotary_factor: 0.25`），MiniMax 转 50%，DeepSeek 和 GLM 的 MLA / 压缩注意力只转 64 维，只有 gpt-oss 转全部维度，Kimi-K3 的全注意力层干脆不转。第 9 章讲的全维度 RoPE 仍是小模型的标准做法（Qwen3-0.6B、主线模型），但大模型这边已经普遍"留一部分维度不带位置"。

## 6. 为什么主线模型只用稠密共识块，下一版加什么

把主线模型放回这张表，它在一众几千亿、上万亿参数的 MoE 中间显得很"朴素"：28 层全注意力、GQA 16/8、QK-Norm、SwiGLU、RMSNorm、RoPE、共享 embedding，和 Qwen3-0.6B 同一个形状。这是 GOAL.md 3.3 的刻意选择，理由在第五部分每一章里都算过：

| 技术 | 旗舰都在用，为什么主线不用 |
|---|---|
| MoE | MoE 是"用显存换算力"：显存装的是总参数，省的是每个 token 的算力。第 24 章第 8 节的产品线里，同一家族几十 B 以下几乎全是稠密、MoE 最小也在 20B 总参数以上；0.7B 再切成几十个专家，每个专家的矩阵小到 GPU 都吃不饱 |
| MLA | 目前只在几百亿以上的 MoE 旗舰里出现；推理时 K 从不还原，和 QK-Norm 不兼容（第 21 章）；需要专门 kernel |
| 稀疏 / 压缩注意力 | 省的是 10 万 token 以上的注意力算力；主线的目标上下文是 32K；做法还没收敛 |
| 混合线性注意力 | **最值得认真考虑**：Qwen3.5-0.8B 在我们的尺寸上就用了；但它需要额外的 kernel、推测解码的回滚更麻烦（第 25 章引导问题 6），第 23 章的小实验也显示纯线性层在"回忆"类任务上吃亏，工具调用恰恰要精确复制参数名 |
| MTP | 训练时多一个 block，推理时可当草稿；但第 25 章的数字说明它的收益主要在推理速度，不改变"工具调用准不准" |

更根本的一条：**在这个规模和上下文长度下，架构带来的差异远小于数据和后训练带来的差异。** 主线模型要赢的是工具调用，决定胜负的是第 13 章的数据配比、第 16–19 章的 SFT、蒸馏、DPO、GRPO。架构选最稳的，把风险留给数据和训练。

那么"下一版"（第二步训练完、有了真实的基线之后）先试什么？`01_panorama.py` 第 5 部分用生产级计算器给主线模型的几种变体记了账（32K 上下文，BF16）：

| 方案 | 32K 的 KV cache | 另外 |
|---|---:|---|
| 现在：28 层全注意力 GQA | 3.50 GiB | — |
| 3:1 局部-全局（窗口 4096） | 1.20 GiB | 零件现成：`zero/arch/sliding_window.py` |
| 3:1 混合线性注意力（Qwen3.5 式） | 896.00 MiB | 线性层固定状态约 22.48 MiB；零件现成：`zero/arch/linear_attention.py` 的 `HybridTransformer` |

按共识规则和我们的目标排序，我会这样排"下一版"的实验：① **1 层 MTP**（`zero/arch/mtp.py`，代价小、第 25 章已有推测解码代码，对工具调用的长 JSON 输出很实用）；② **3:1 混合线性注意力**（KV 少到四分之一，和 Qwen3.5-0.8B 同一条路，但要先在工具调用的"精确复制"评测上对比）；③ 局部-全局交替作为②的保守备选。MoE、MLA、稀疏注意力在 0.7B、32K 这个点上都不划算，不进下一版。每一项都要先在第 12 章的 ladder 配置上做同算力对比，按第 11 章的预注册方式判定，再决定进不进主线。

## 7. 小结

- **最新的名单**（2026-09）：DeepSeek-V4-Pro、Qwen3.8-2.4T-A95B、Kimi-K3、gpt-oss-120b、GLM-5.3、MiniMax-M3。确认"最新"要现查，不能凭记忆。
- **演化树**：GPT-2 → 现代稠密块（RMSNorm、RoPE、SwiGLU、GQA、QK-Norm）→ 六根分枝：MoE、KV 压缩（GQA → MLA → 压缩注意力）、只看一部分（滑动窗口 → 稀疏）、混合线性注意力、MTP、残差流改造。
- **共识**：稠密块 + MoE（细粒度 + 共享专家）+ MTP 是 2026 年旗舰的默认配置；长上下文分成混合线性、稀疏、少存三派，每派都过了 3 家门槛，但具体做法还在分化。
- **算账**：KV cache 已经和参数量脱钩，2.8T 的 Kimi-K3 在 128K 时比 0.7B 的主线模型 32K 时还小；激活参数的口径各家不同。
- **主线模型**：刻意只用稠密共识块；"下一版"最值得试的是 MTP 和 3:1 混合线性注意力。

---

## 回到起点：整门课是一条路

26 章走完了。回头看，每一章都是在上一章遇到的问题上长出来的（GOAL.md 2.2）：

| 部分 | 我们解决了什么问题 | 在今天的旗舰里是哪一块 |
|---|---|---|
| 第一部分（第 1–6 章） | 直线 → 神经网络 → 反向传播 → 分类 → 让训练稳定 | 每个旗舰的训练循环：模型、损失、梯度、更新，四步一模一样；RMSNorm、残差、AdamW / warmup 至今未变 |
| 第二部分（第 7–10 章） | 分词、注意力、现代 Transformer、推理 | 演化树的主干：6 个旗舰全部长在第 9 章的稠密块上；第 10 章的 KV cache 是第五部分的起点 |
| 第三部分（第 11–15 章） | 评测、Scaling Law、数据、预训练工程、长上下文 | 模型卡上的每一张成绩表、每一个"32T token"、每一个 YaRN 因子 |
| 第四部分（第 16–20 章） | SFT、蒸馏、偏好对齐、强化学习、发布 | DeepSeek-V4 的"专家 SFT + GRPO → 在线策略蒸馏"、各家的 thinking 模式、FP4 / MXFP4 发布 |
| 第五部分（第 21–26 章） | KV cache、局部 / 稀疏、线性混合、MoE、MTP、全景 | 这一章的整棵树 |

第一章说"训练一个大模型，和这一章做的事情在结构上一模一样，只是参数从 2 个变成了几十亿个"。现在可以把数字补上：Kimi-K3 有 2.8 万亿个参数，每个 token 用到其中 1040 亿个，但它每一步做的仍然是算损失、求梯度、减去学习率乘梯度。

**接下来是第二步。** 课程写完了，主线模型的生产级代码也在 CPU 上用极小配置跑通了。拿到 GPU 之后，按 GOAL.md 第 3、10 节的闸门流程真实训练那个 689.5M 的稠密模型：先做一次 ≤ $50 的 GPU 验证运行，再跑 ladder、定超参、预训练、中期训练、长上下文、SFT、蒸馏、DPO、GRPO，最后在预注册的考卷上和 Qwen3.5-0.8B 比工具调用。那时我们再回到这张表，把"主线模型"那一行的"待 GPU 训练后补充"换成真实的数字。

---

## 从极简到生产级

| 极简版（`code/`） | 生产级 / 行业参考实现 | 多做了什么、为什么 |
|---|---|---|
| `01_panorama.py` 的 `layout` / `kv_numbers` | `zero/tools/kv_cache_calc.py`：`layout_from_config`、`kv_cache_bytes`、`fixed_state_bytes`、`breakdown`；命令行 `uv run python -m zero.tools.kv_cache_calc <config.json> --seq 131072` | 本章除 DeepSeek-V4 外的 KV 数字全部直接调用它（它能读 HF 的 `text_config` 嵌套、`layer_types`、Kimi 的 `linear_attn_config`、MLA 字段）；DeepSeek-V4 的 `compress_ratios` 它还不认识，本章用它的 `KVLayout` / `LayerSpec` 自己搭（建议的改进见下） |
| `03_meta_params.py` 的 `meta_count` | `zero/tools/count_params.py`（主线模型，按公式、不分配内存）；HF transformers 的官方模型类（`DeepseekV4ForCausalLM`、`Qwen3_5MoeForCausalLM`、`KimiLinearForCausalLM`、`GptOssForCausalLM`、`GlmMoeDsaForCausalLM`、`MiniMaxM3VLForCausalLM`） | 主线模型：zero 的公式与同形状的 HF `Qwen3ForCausalLM` 在 meta 设备上数出的 0.6895B 完全相同。HF 类是各家架构的参考实现，读它们的 `modeling_*.py` 是理解新架构最快的办法（例如 `DeepseekV4CSACache` 的注释把 CSA 的缓存讲得很清楚） |
| 演化树上的每个分枝 | `zero/arch/`：`mla.py`（第 21 章）、`sliding_window.py`（第 22 章）、`linear_attention.py`（第 23 章，`HybridTransformer`，与 HF `Qwen3_5GatedDeltaNet` 对拍）、`moe.py`（第 24 章）、`mtp.py` 与 `speculative.py`（第 25 章） | 都是"不用于主线"的实验模块，接口与 `zero.model` 相同、可以直接换进主线的 `Transformer`；CUDA 上的正确性已在 RTX 3090 上验证（MoE、线性注意力各修了一个 bug），性能尚未在 GPU 上验证，见 [runs/2026-10-01-gpu0-check](../../runs/2026-10-01-gpu0-check/README.md) 第 11、12 节 |
| 主线模型这一行 | `configs/main/pretrain.toml`、`zero/model.py` | 稠密共识块；`tests/test_model_hf_parity.py` 保证与 HF Qwen3 实现 logits 一致 |

**对拍**：`uv run pytest tests/test_kv_cache_calc.py tests/test_config.py tests/test_arch_*.py`，本机 109 项全部通过（约 44 秒）。本章的三个脚本都直接调用上面这些生产级函数，没有另写一套公式。

**建议对 `zero/` 的改进**（按协作约定只在这里提出，没有改动）：`zero/tools/kv_cache_calc.layout_from_config` 可以识别 DeepSeek-V4 的 `compress_ratios`（4 → CSA、128 → HCA、0 → 纯滑动窗口，共享 K=V、每条 `head_dim` 个数、外加 `sliding_window` 分支，CSA 另有 `index_head_dim` 的索引器 key），逻辑可以照搬本章的 `dsv4_layout`；另外 GPT-2 式字段名（`n_layer`、`n_head`、`n_embd`）也可以加进 `_get` 的别名。

---

## 前沿观察

> **不算共识、只在这里提一句的技术**（各只查到 1–2 个头部家族）
>
> - **压缩注意力 CSA / HCA**（DeepSeek-V4）：把每 4 个或 128 个 token 的 KV 压成一条，再配一个 128 窗口的滑动分支；共享 K=V 的 MQA、head_dim 512。KV cache 只有 V3.2 的约十分之一。目前只有 DeepSeek 一家，而且它是在 MLA 已被多家采用之后换掉 MLA 的——这是本章最值得持续观察的变化。
> - **残差流改造**：DeepSeek-V4 的 mHC（把残差流扩成 4 路，层间混合矩阵约束为双随机矩阵）和 Kimi-K3 的 Attention Residuals（`attn_res_block_size: 12`）。两家思路不同，千问实验性的 Qwen3.8-Flash-Next（`model_type: qwen4_exp`）的 config 里也出现了 `hc_count: 4`，但它不是主力版本。
> - **Latent MoE**（Kimi-K3）：路由专家在 3584 维的潜空间里计算，896 个专家才装得下。
> - **全注意力层用 NoPE**（Kimi Linear / K3）：位置信息完全交给线性注意力层。
> - **n-gram 嵌入表**：DeepSeek-V4.1-Flash 的 config 里有 `engram_*` 字段，Qwen3.8-Flash-Next 有 `ngram_*` 字段，用查表的方式给模型加"记忆"。两家都只出现在非旗舰或实验版本里。
> - **KDA 的逐通道门控**：见第 23 章。
> - **注意力输出门控**（千问的 `attn_output_gate`、Kimi-K3 的 Gated MLA）与 **attention sink**（gpt-oss、DeepSeek-V4）：各 2 家。

---

## 采用方与来源

本章拆解的 6 个旗舰家族里，每项技术的采用方（证据是 config 字段，或注明的代码 / 报告；完整明细见 `01_panorama.py` 第 3 部分输出）：

| 技术 | 本章 6 个旗舰里的采用方 | 前几章补充的采用方 | 判定 |
|---|---|---|---|
| Pre-Norm + RMSNorm | 全部 6 家（`rms_norm_eps`；MiniMax-M3 为 Gemma 式 `1+w`） | — | 共识 |
| RoPE | DeepSeek-V4（只转 64 维）、Qwen3.8（25%）、gpt-oss（全维，θ 150,000）、GLM-5.3（64 维）、MiniMax-M3（50%） | Qwen3、Llama 3 等（第 9 章） | 共识 |
| YaRN | DeepSeek-V4（×16，原长 65,536）、gpt-oss（×32，原长 4,096） | Kimi K2、Qwen3、SmolLM3（第 15 章） | 共识 |
| SwiGLU / 门控 FFN | 全部 6 家（`silu`；MiniMax `swigluoai`；Kimi `situ`） | — | 共识 |
| GQA | Qwen3.8（64/4）、gpt-oss（64/8）、MiniMax-M3（64/4） | Qwen3、Llama 3 等（第 10、21 章） | 共识 |
| QK-Norm | DeepSeek-V4（报告 2.3.3 节）、Qwen3.8（HF `Qwen3_5MoeAttention.q_norm/k_norm`）、MiniMax-M3（`use_qk_norm: true`） | Gemma 3、OLMo 2（第 9 章） | 共识 |
| MLA | Kimi-K3、GLM-5.3（`kv_lora_rank: 512`） | Mistral Large 3、DeepSeek-V3/V3.2（第 21 章） | 共识（DeepSeek-V4 已换掉） |
| 滑动窗口 / 局部-全局 | gpt-oss（1:1，窗口 128）、DeepSeek-V4（每层 128 的滑动分支） | Gemma 3、OLMo 3（第 22 章） | 共识 |
| 稀疏注意力 | DeepSeek-V4（`index_topk: 1024`）、GLM-5.3（`index_topk: 2048` + IndexShare）、MiniMax-M3（MSA，`sparse_topk_blocks: 16`） | DeepSeek-V3.2（第 21 章）、美团 LongCat-2.0（第 22 章） | **2026 新晋共识，做法未收敛** |
| 混合线性注意力 | Qwen3.8（69 GDN : 23）、Kimi-K3（69 KDA : 24） | NVIDIA Nemotron 3（第 23 章）、Qwen3.5 全系列 | 共识 |
| MoE（细粒度） | 全部 6 家：384 选 6、512 选 10、896 选 16、128 选 4、256 选 8、128 选 4 | 第 24 章 | 共识 |
| 共享专家 | DeepSeek-V4（1）、Qwen3.8（1）、Kimi-K3（2）、GLM-5.3（1）、MiniMax-M3（1）；gpt-oss 不用 | 第 24 章 | 共识 |
| 无辅助损失均衡 | DeepSeek-V4、Kimi-K3、GLM-5.3（`noaux_tc`）、MiniMax-M3（`use_routing_bias`）；千问、gpt-oss 用辅助损失 | 第 24 章 | 共识 |
| MTP | DeepSeek-V4、GLM-5.3（`num_nextn_predict_layers: 1`）、Qwen3.8（`mtp_num_hidden_layers: 1`）、MiniMax-M3（`num_nextn_predict_layers: 1`，另有 `num_mtp_modules: 7`）；Kimi-K3、gpt-oss 没有 | 第 25 章 | 共识 |
| 共享输入输出 embedding | 旗舰都不共享；小模型共享：Qwen3-0.6B、Qwen3.5-0.8B | 第 9 章 | 共识（限小模型） |

**模型配置与模型卡**（2026-09-26 通过 Hugging Face 读取）：

- DeepSeek-V4-Pro：[config.json](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/config.json)、[模型卡](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro)、技术报告 [arXiv:2606.19348](https://arxiv.org/abs/2606.19348)（2.3 节 CSA/HCA 与其他细节、2.2 节 mHC、3.5.1 节 KV cache 布局、4.2.1 节模型设置）
- DeepSeek-V3.2：[config.json](https://huggingface.co/deepseek-ai/DeepSeek-V3.2/blob/main/config.json)（671B / 37B 取自 V4 模型卡的 base 模型对比表）
- DeepSeek-V4.1-Flash：[config.json](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/config.json)（`engram_*` 字段）
- Qwen3.8-2.4T-A95B：[config.json](https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B/blob/main/config.json)、[模型卡](https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B)、[官方博客](https://qwen.ai/blog?id=qwen3.8)
- Qwen3.8-Flash-Next：[config.json](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/config.json)（`qwen4_exp`，`hc_count`、`ngram_*`、`indexer_*`）
- Qwen3.5-0.8B：[config.json](https://huggingface.co/Qwen/Qwen3.5-0.8B/blob/main/config.json)；Qwen3-0.6B：[config.json](https://huggingface.co/Qwen/Qwen3-0.6B/blob/main/config.json)
- Kimi-K3：[config.json](https://huggingface.co/moonshotai/Kimi-K3/blob/main/config.json)、[模型卡](https://huggingface.co/moonshotai/Kimi-K3)、[技术报告 PDF](https://github.com/MoonshotAI/Kimi-K3/blob/main/k3_tech_report.pdf)（本章未通读，见"待核实"）；Kimi Linear 报告 [arXiv:2510.26692](https://arxiv.org/abs/2510.26692)（KDA、3:1、MLA 层 NoPE、MuonClip）
- gpt-oss-120b：[config.json](https://huggingface.co/openai/gpt-oss-120b/blob/main/config.json)、[模型卡](https://huggingface.co/openai/gpt-oss-120b)、[OpenAI 模型卡 arXiv:2508.10925](https://arxiv.org/abs/2508.10925)
- GLM-5.3：[config.json](https://huggingface.co/zai-org/GLM-5.3/blob/main/config.json)、[模型卡](https://huggingface.co/zai-org/GLM-5.3)；GLM-5 [config.json](https://huggingface.co/zai-org/GLM-5/blob/main/config.json) 与 [模型卡](https://huggingface.co/zai-org/GLM-5)（744B / 40B）；[GLM-5.2 模型卡](https://huggingface.co/zai-org/GLM-5.2)（IndexShare [arXiv:2603.12201](https://arxiv.org/abs/2603.12201)）；GLM-5 技术报告 [arXiv:2602.15763](https://arxiv.org/abs/2602.15763)
- MiniMax-M3：[config.json](https://huggingface.co/MiniMaxAI/MiniMax-M3/blob/main/config.json)、[模型卡](https://huggingface.co/MiniMaxAI/MiniMax-M3)、技术报告 [arXiv:2606.13392](https://arxiv.org/abs/2606.13392)、[MSA 代码](https://github.com/MiniMax-AI/MSA)
- GPT-2：[config.json](https://huggingface.co/openai-community/gpt2/blob/main/config.json)

**待核实**：

- Kimi-K3 的技术报告（GitHub 上的 PDF）本章没有通读：`mla_use_nope: true` 按 Kimi Linear 报告的做法解读为"MLA 层不用位置编码"；Latent MoE 的具体结构（投影在哪、是否每个专家共享投影）只做了参数量层面的核对（2.72T 与模型卡的 2.8T 相符）；Attention Residuals 的细节未核实。
- GLM-5.3 的总 / 激活参数：模型卡未写，按与 GLM-5 相同的 config 形状沿用 GLM-5 模型卡的 744B / 40B；HF 类数出 743.4B。
- MiniMax-M3 的 `num_mtp_modules: 7` 与 `num_nextn_predict_layers: 1` 同时出现，推理时实际用几层 MTP 未核实。
- DeepSeek-V4 `compress_ratios` 的第 62 项（0）对应的是 MTP 层，是按 HF `DeepseekV4Config` 的映射（0 = 纯滑动窗口）和"61 层 + 1 层 MTP"推断的。
- DeepSeek-V4 的 KV cache 按"每条压缩条目 head_dim 个数、摊到每个 token"估算，没有计入尚未凑满一个压缩块的缓冲 token；报告里的精确存储格式（RoPE 维 BF16、其余 FP8、索引器用 FP4 计算）只在对照里定性提到。
- Qwen3.8 是否有正式技术报告：写作时只找到模型卡与博客链接。

---

## 引导问题

带着这些问题去问 Claude Code，直到你能用自己的话讲清楚：

1. DeepSeek 发明了 MLA，又在 V4 里换掉了它。对照第 21 章 MLA 的代价（decode 算力、与 QK-Norm 不兼容、需要专门 kernel）和本章 CSA/HCA 的做法，你觉得 DeepSeek 换掉它最可能是为了解决哪个问题？V4 的"共享 K=V、head_dim 512"又重新引入了什么代价？
2. Qwen3.8 和 Kimi-K3 都是约 3:1 的混合线性注意力，一个全注意力层用 GQA，一个用 MLA。如果把 Qwen3.8 的 23 层全注意力换成 MLA（512 + 64），128K 的 KV cache 会变成多少？用 `01_panorama.py` 的办法算一算。
3. MiniMax-M3 的块稀疏注意力不省 KV cache，为什么还能让 decode 快 15 倍？（提示：第 21 章 decode 读的是什么；稀疏注意力每一步要读的是全部 KV 还是被挑中的块？）
4. 本章说"激活参数没有统一口径"。如果你要公平地比较 gpt-oss-120b 和 MiniMax-M3 每个 token 的计算量，应该用哪一列？每个 token 的前向计算量大约是激活参数的几倍？
5. 按 GOAL.md 2.1，稀疏注意力到 2026 年才过了 3 家的门槛。你认为"几家都在用同一个思想、但做法各不相同"算不算共识？如果你是这门课的作者，会把它写进正文还是留在前沿观察？
6. 假如主线模型的下一版只能加一项新技术，你会选 MTP、3:1 混合线性注意力，还是局部-全局交替？列出你的理由、预期收益，以及要在第 11 章的考卷上加什么测试来证明它没有伤害工具调用。

## 动手任务

每个任务都要真的运行代码、看到结果。

**任务 1（基础）**：在 Hugging Face 上挑一个本章没收录的模型（例如 DeepSeek-V4-Flash、Qwen3.8-27B、GLM-5.3-Flash、NVIDIA Nemotron 3 或 Gemma 的最新版），读它的 `config.json`，按 `models.json` 的格式加一条（每个字段都要来自你读到的文件，并附链接），重跑 `01_panorama.py` 和 `03_meta_params.py`：它的层由什么组成？总参数和模型卡对得上吗？128K 的 KV cache 是多少？它会改变采用矩阵里哪些技术的"家数"？

**任务 2（核心）**：给 `zero.tools.kv_cache_calc.layout_from_config` 写一个"本地补丁"（先复制到你自己的脚本里，不要改 `zero/`）：让它能直接读 DeepSeek-V4 的 `compress_ratios`，并加一个开关决定算不算 CSA 索引器的 key。用它算 V4-Pro 在 32K、128K、1M 时的 KV cache，和 `01_panorama.py` 的 `dsv4_layout` 对拍；再把 KV 条目按"448 维 FP8 + 64 维 BF16"计字节，看看和报告的"约 10%"是不是更接近了。

**任务 3（挑战）**：设计你自己的"下一代小模型"（总参数 ≤ 0.8B，目标上下文 32K，主打工具调用）：写出它的 config（层数、宽度、每层类型、KV 头数、是否 MTP、词表），用 `zero.tools.kv_cache_calc` 算出 32K 的 KV cache，用 `zero.tools.count_params`（或 HF 类在 meta 设备上）算出总参数；然后对每一项和主线模型不同的选择，写一句"它满足 GOAL.md 2.1 的哪一条、采用方是谁、在 0.8B / 32K 这个点上为什么划算"。自检 Skill `/ch26-panorama` 的第四关会和你讨论这个设计。

---

## 想深入：CS336

本章对应斯坦福 CS336（Spring 2026）<https://cs336.stanford.edu/>：

- **第 3 讲：架构与超参**。现代 Transformer 的各种架构选择（归一化位置、激活函数、位置编码、注意力头的共享方式）和超参的经验取值。本章的演化树主干和"稳如磐石的主干"一节就是这一讲内容在 2026 年旗舰上的落点。讲义与录像见课程页（本构建环境访问不了课程页，具体覆盖范围以讲义为准）。
- **第 4 讲：注意力的替代方案与 MoE**。线性注意力、状态空间模型、稀疏 / 局部注意力与 MoE，对应本章的分枝 1、3、4。
- **CS336 未深入**：按 config 逐项核对 6 个最新旗舰、用 HF 官方类在 meta 设备上数参数、KV cache 的逐层账本、按"至少 3 家"做共识判定、DeepSeek-V4 的压缩注意力与 mHC、Kimi-K3 的 Latent MoE 与 AttnRes——这些是本课结合 2026 年 9 月的模型补充的内容，见本章参考文献。

---

## 本章参考文献

- DeepSeek-AI. *DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence*，2026：<https://arxiv.org/abs/2606.19348>
- Kimi Team. *Kimi Linear: An Expressive, Efficient Attention Architecture*（KDA、3:1 混合、MLA 层 NoPE），2025：<https://arxiv.org/abs/2510.26692>
- Kimi Team. *Kimi K3 Technical Report*：<https://github.com/MoonshotAI/Kimi-K3/blob/main/k3_tech_report.pdf>
- GLM-5 Team. *GLM-5: from Vibe Coding to Agentic Engineering*，2026：<https://arxiv.org/abs/2602.15763>；IndexShare：<https://arxiv.org/abs/2603.12201>
- MiniMax. *MiniMax-M3* 技术报告（MiniMax Sparse Attention），2026：<https://arxiv.org/abs/2606.13392>
- OpenAI. *gpt-oss-120b & gpt-oss-20b Model Card*，2025：<https://arxiv.org/abs/2508.10925>
- Qwen Team. *Qwen3.8-Max* 博客：<https://qwen.ai/blog?id=qwen3.8>
- 演化树上各节点的出处：GPT-2（Radford 等 2019）；Llama <https://arxiv.org/abs/2302.13971>；RMSNorm <https://arxiv.org/abs/1910.07467>；RoPE <https://arxiv.org/abs/2104.09864>；YaRN <https://arxiv.org/abs/2309.00071>；GLU 变体 <https://arxiv.org/abs/2002.05202>；GQA <https://arxiv.org/abs/2305.13245>；QK-Norm <https://arxiv.org/abs/2010.04245>；共享 embedding <https://arxiv.org/abs/1608.05859>；DeepSeekMoE <https://arxiv.org/abs/2401.06066>；无辅助损失均衡 <https://arxiv.org/abs/2408.15664>；MLA（DeepSeek-V2）<https://arxiv.org/abs/2405.04434>；Mistral 7B <https://arxiv.org/abs/2310.06825>；Gemma 2 <https://arxiv.org/abs/2408.00118>；Gated DeltaNet <https://arxiv.org/abs/2412.06464>；MTP（DeepSeek-V3）<https://arxiv.org/abs/2412.19437>
- [1.5 万字速通 LLM 主流模型结构（Llama、Qwen、GLM、DeepSeek…）](https://zhuanlan.zhihu.com/p/2060741715095560795)：各家结构的横向对比，适合配合本章的演化树读（`references.md` 已收录）
- [Pretraining a Mini Kimi K3](https://books.vizuara.ai/book/pretraining-a-mini-k3)：动手预训练一个"KDA + MLA 混合"的小 K3，是第 23、26 章之后最好的动手延伸（`references.md` 已收录）
- [CS336](https://cs336.stanford.edu/) 第 3、4 讲
- 模型配置链接见上方"采用方与来源"。

**下一步**：这是课程的最后一章。第一步（写完课程、生产级代码在 CPU 上跑通）到这里完成；第二步是拿到 GPU 后按闸门流程真实训练主线模型，从 `runs/RUNBOOK.md` 开始。
