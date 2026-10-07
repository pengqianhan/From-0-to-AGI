# 第 22 章：局部与稀疏注意力 —— 看附近，也保留远处的上下文

[English](README.md) · **中文**

> **目标**：读完这一章，你能写出滑动窗口注意力的掩码和有上限的 KV cache，并算出 L 层滑动窗口的感受野。你能根据 `config.json` 里的 `layer_types` 和 `sliding_window`，说清楚 Gemma、gpt-oss、OLMo 3 把多少层换成了局部注意力，省了多少 KV cache。你还能用一个小实验指出纯滑动窗口在哪里失效，以及插入一层全局注意力为什么能修好它。最后，你能讲清楚两种挑键方式的区别：稀疏注意力按内容挑键，滑动窗口按位置挑键。

📺 **本章视频**：还没有发布。要在本机渲染，运行 `bash chapters/22-local-sparse-attention/video/build.sh`。
🧪 **本章自检**：学完后，在 Claude Code 里输入 `/ch22-local-attention`。

---

上一章我们给 KV cache 记了一本账。长上下文贵在两处：prefill 的注意力算力随长度的平方增长；decode 的每一步都要从显存里把全部历史 K/V 读一遍。MQA、GQA、MLA 压缩的是"每个位置存多少"。这一章问另一个问题：**每个位置都必须看全部历史吗？**

最直接的办法是只看附近的一段，这就是**滑动窗口注意力**（sliding window attention，SWA）。它的代价是看不到远处，所以有了**局部-全局交替**（local-global interleaving）：大部分层是局部的，少数层是全局的。再往前一步：与其固定地看"最近的"键，不如让模型自己挑出"最相关的"几个键。这就是**稀疏注意力**（sparse attention）。

本章代码（都在 CPU 上运行）：

```bash
uv run python chapters/22-local-sparse-attention/code/01_masks_and_ledger.py  # masks, receptive field, KV ledger of 4 public models (a few seconds)
uv run python chapters/22-local-sparse-attention/code/02_swa_model.py         # trains 6 small models (slow on the first run, see below; later runs read the cache)
uv run python chapters/22-local-sparse-attention/code/03_compare.py           # full attention / sliding window / interleaving: loss, KV, needle in a haystack
uv run python chapters/22-local-sparse-attention/code/04_bounded_cache.py     # generation with a truncated cache = generation without a cache; the cache size has a maximum
uv run python chapters/22-local-sparse-attention/code/05_topk_sparse.py       # the same k keys: select by position vs select by content
```

`02_swa_model.py` 在单线程 CPU 上训练 6 个小模型：3 个语言模型各 600 步，3 个大海捞针模型各 600 步。本课的构建机上有多个任务共享 CPU，首次运行共用了约 40 分钟。空闲的机器上会快很多。之后的脚本直接读 `code/out/` 里缓存的权重。`03`、`04` 各需一两分钟，`05` 需要几分钟。

## 1. 全注意力的两笔账

先回忆第 8 章的因果注意力。位置 i 的查询（query）和位置 0…i 的每一个键（key）各算一次分数，再对值（value）加权平均。把"哪些位置对要计算"画成一张 T × T 的表，结果是一个下三角：

```
Full causal attention T=10 (■ = visible, 55 pairs)
  ■ · · · · · · · · ·
  ■ ■ · · · · · · · ·
  ■ ■ ■ · · · · · · ·
  ...
  ■ ■ ■ ■ ■ ■ ■ ■ ■ ■
```

两笔账都和长度 T 有关（第 21 章已经细算过）：

- **算力**：要算 `T(T+1)/2` 个分数，随 T² 增长。prefill 一段长提示词时，这一项最终会超过所有矩阵乘法的总和。
- **KV cache**：每层要存全部 T 个位置的 K 和 V。decode 时每生成一个 token，都要从显存里把它们完整读一遍。

问题出在"每个位置都要看全部历史"。可是语言里大部分依赖都很近：一个词最常依赖同一句、同一段里的词。那么，能不能**只看附近**？

## 2. 滑动窗口：每个 token 只看最近 W 个

**滑动窗口注意力**给因果掩码（mask）再加一个条件：位置 i 只能看见满足 `i − W < j ≤ i` 的位置 j，也就是最近 W 个位置（包括它自己）。

```
mask(i, j) = (j ≤ i) 且 (i − j < W)
```

代码里只多一个 `&`（[`code/01_masks_and_ledger.py`](code/01_masks_and_ledger.py)）：

```python
def sliding_mask(T: int, W: int) -> torch.Tensor:
    i = torch.arange(T)[:, None]
    j = torch.arange(T)[None, :]
    return (j <= i) & (i - j < W)  # see only the last W positions (itself included)
```

三角形变成了一条斜着的带子：

```
Sliding window W=4 (■ = visible, 34 pairs)
  ■ · · · · · · · · ·
  ■ ■ · · · · · · · ·
  ■ ■ ■ · · · · · · ·
  ■ ■ ■ ■ · · · · · ·
  · ■ ■ ■ ■ · · · · ·
  · · ■ ■ ■ ■ · · · ·
  ...
  · · · · · · ■ ■ ■ ■
```

带子的宽度固定，所以每个查询的计算量是 O(W)，而不是 O(T)。整张表的面积从 T²/2 变成约 T·W。数一数（同一个脚本的输出）：

| 长度 T | 全因果（对） | W = 4 的滑动窗口（对） | 倍数 |
|---:|---:|---:|---:|
| 1,024 | 524,800 | 4,090 | 128 |
| 32,768 | 536,887,296 | 131,066 | 4,096 |

真实模型的窗口是几百到几千个位置（见下一节），但道理一样：**上下文比窗口长得越多，省得越多**。

> **注意：**本课统一约定：窗口 W 包含自己，每个 token 最多看 W 个位置。这和 Hugging Face 配置里 `sliding_window` 字段的含义一致。不同 kernel 对这个参数的定义可能差一。例如 FlashAttention 的 `window_size=(左, 右)` 指的是"向左再看几个"，对应 W − 1。移植代码时要做对拍（parity check）。

## 3. 感受野：信息一层层接力

只看最近 W 个位置，远处的信息就全部丢了吗？不完全是。在第 1 层，位置 i 从 i−3…i 收集信息。在第 2 层，位置 i 看到位置 i−3，而位置 i−3 在第 1 层已经收集了 i−6…i−3 的信息。**每过一层，信息最多再往前传 W − 1 个位置。**L 层之后，一个 token 理论上能"间接看到" `L × (W − 1)` 个位置之前。

`01_masks_and_ledger.py` 用布尔矩阵连乘算出感受野（receptive field）。`reach[i, j]` 表示位置 j 的信息能否经过前面这些层流到位置 i。每过一层，就乘一次这一层的掩码：

```python
reach = (m.float() @ reach.float()) > 0  # in this layer, i sees k, and k already collected j
```

T = 64、W = 4、6 层时，最后一个 token 在每层之后最远能看到多少个位置之前：

| 层 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---:|---:|---:|---:|---:|---:|
| 全部滑动窗口 | 3 | 6 | 9 | 12 | 15 | 18 |
| 第 3、6 层换成全注意力 | 3 | 6 | 63 | 63 | 63 | 63 |

第一行正好是 `l × 3`。Mistral 7B 的论文用的就是这个论证：窗口 4096、32 层，理论跨度约 13 万个 token。第二行说明另一件事：只要插入一层全注意力，这一层之后就能**一步**看到开头。

但"理论上够得着"不等于"实际学得会"。远处的信息要靠中间的 token 一层层转手，还要挤在有限的隐藏维度里。第 6 节的实验会显示这个边界有多硬。

## 4. KV cache 有上限

滑动窗口还有一个比省算力更实际的好处：**窗口外的 K、V 以后不会再用到，可以直接丢掉**。所以滑动窗口层的 KV cache 封顶在 W 个位置，不再随上下文增长。

极简版（[`code/02_swa_model.py`](code/02_swa_model.py) 的 `KVCache.append`）先把"旧 + 新"交给这一步的注意力使用，再只保留最后 W 个：

```python
keep = slice(None) if window is None else slice(-window, None)
self.k[layer], self.v[layer], self.pos[layer] = k[:, :, keep], v[:, :, keep], pos[keep]
```

K 在写进缓存之前已经做过 RoPE，位置信息已经"转"进了向量。所以缓存只需要额外记住每个槽位对应的全局位置 `pos`，用来构造掩码。

[`code/04_bounded_cache.py`](code/04_bounded_cache.py) 用第 6 节训练好的小语言模型（4 层、W = 16）贪心生成 300 个字符。它比较两种做法："截断缓存"和"不用缓存、每步整段重算"：

| 配置 | 与不用缓存逐字相同 | KV cache 字节数：已缓存 16 / 64 / 128 / 256 / 306 个位置 |
|---|---|---|
| 全注意力 | True | 49,152 / 196,608 / 393,216 / 786,432 / 940,032 |
| 全部滑动窗口 | True | 49,152 / 49,152 / 49,152 / 49,152 / 49,152 |
| 3 局部 + 1 全局 | True | 49,152 / 86,016 / 135,168 / 233,472 / 271,872 |

纯滑动窗口从头到尾都是 49,152 字节（16 个位置 × 4 层 × K、V × 96 维 × 4 字节）。交替配置里只有那一层全局层在增长。生产级实现（见本章后面）用**环形缓冲区**（ring buffer）代替切片：位置 p 写进槽位 `p % W`，新的覆盖最旧的。缓冲区只分配一次，不再拷贝。

## 5. 局部-全局交替：真实模型怎么配

纯滑动窗口在大模型里不多见。最早的 Mistral 7B（v0.1）32 层全部是窗口 4096 的滑动窗口。但后来的 Mistral-7B-v0.3 在 `config.json` 里把 `sliding_window` 设成了 `null`，退回到全注意力。今天的主流是**局部-全局交替**：大部分层用小窗口，每隔几层放一层全注意力，由这些层负责"远处的检索"。这些配置都直接写在 `config.json` 里（`layer_types` 列出每层的类型，`sliding_window` 给出窗口）：

| 模型 | 层的排布 | 窗口 | 依据 |
|---|---|---:|---|
| Gemma 2（9B） | 局部、全局 1:1 交替 | 4096 | 技术报告："alternate between a local sliding window attention and global attention in every other layer"；`config.json` 的 `sliding_window` |
| Gemma 3（27B） | 5 局部 : 1 全局 | 1024 | 技术报告；`sliding_window` 1024（1B 版本为 512，`sliding_window_pattern` 6） |
| Gemma 4（31B） | 5 局部 : 1 全局 | 1024 | `layer_types`（60 层中 10 层全局）；E2B 版本窗口 512、4:1 |
| gpt-oss（20b / 120b） | 1:1 交替 | 128 | `layer_types`；`sliding_window` 128 |
| OLMo 3（7B） | 3 局部 : 1 全局 | 4096 | `layer_types` |
| Ministral 8B（2410） | 1 全局 : 3 局部 | 32768 | `layer_types`（窗口与最大长度相同） |
| Step-3.5-Flash | 1 全局 : 3 局部 | 512 | `layer_types`；`sliding_window` 512 |
| MiMo-V2-Flash | 1 全局 : 5 局部（首层全局） | 128 | `hybrid_layer_pattern`；`sliding_window` 128 |
| Llama 4 Scout | 每 4 层中 3 层用"分块注意力"，1 层全局（NoPE） | 8192（块） | `attention_chunk_size` 8192、`no_rope_layers` |

几点观察：

- **比例和窗口差别很大**：比例从 1:1 到 5:1，窗口从 128 到 4096。Gemma 3 报告（5.2 节）做了消融：局部:全局从 1:1 变到 7:1，窗口从 4096 降到 512，对困惑度的影响都很小。在 32K 上下文下，"全部全局"配置的 KV cache 相当于模型本身内存的 60%；换成 1:3、窗口 1024 后，不到 15%。"困惑度看不出区别"这一点，第 6 节的小实验会再遇到一次。
- **不少模型给局部层和全局层配不同的 RoPE 基频**：Gemma 3 的 `rope_local_base_freq` 是 10000，全局层的 `rope_theta` 是 100 万。Gemma 4、MiMo-V2-Flash、Step-3.5-Flash 也把两者分开设。局部层只看几百个位置，用不到为长距离准备的低频维度。
- **分块注意力**（chunked attention，Llama 4）是滑动窗口的近亲。它把序列切成 8192 一块，块内是因果注意力，块之间互相不看。效果类似，而缓存和 kernel 更规整。

用第 21 章的账本算一下 128K 上下文（BF16、batch 1）下能省多少（`01_masks_and_ledger.py` 的最后一段）：

| 模型（配置） | 假如每层都是全注意力 | 实际配置 | 节省 |
|---|---:|---:|---:|
| Mistral-7B-v0.1（全部滑动，W 4096） | 16.00 GiB | 0.50 GiB | 96.9% |
| Gemma-3-27B（5:1，W 1024） | 62.00 GiB | 10.41 GiB | 83.2% |
| gpt-oss-120b（1:1，W 128） | 9.00 GiB | 4.50 GiB | 50.0% |
| OLMo-3-7B（3:1，W 4096） | 64.00 GiB | 17.50 GiB | 72.7% |

（Mistral 7B 和 OLMo 3 7B 的官方最大长度分别是 32K 和 64K。这里统一按 128K 算，只为比较结构。）规律很直接：**全局层那部分照样随长度线性增长，局部层那部分封顶**。局部层比例越高、窗口越小，省得越多。

## 6. 代价在哪：loss 看不出，大海捞针看得出

[`code/02_swa_model.py`](code/02_swa_model.py) 定义了一个小 Transformer，它的每层可以有不同的窗口。结构与第 9、10 章相同。三种配置都是 4 层、窗口 W = 16：

| 配置 | 各层窗口 | 对应 |
|---|---|---|
| `full` | `[None, None, None, None]` | 全注意力 |
| `sliding` | `[16, 16, 16, 16]` | Mistral 7B v0.1 |
| `interleave` | `[16, 16, 16, None]` | 3 局部 : 1 全局（OLMo 3 的比例） |

注意力里只多了一行掩码，三种配置共用同一份代码：

```python
att = att.masked_fill(~window_mask(pos, k_pos, self.window), float("-inf"))
```

**实验一：字符级语言建模**（Shakespeare，600 步，数据顺序相同）。运行 [`code/03_compare.py`](code/03_compare.py)：

| 配置 | 验证集 loss（nats/字符） | KV 缓存位置数 @128 | @4096 |
|---|---:|---:|---:|
| full | 1.797 | 512 | 16,384 |
| sliding | 1.745 | 64 | 64 |
| interleave | 1.748 | 176 | 4,144 |

三者几乎一样，纯滑动窗口甚至略好。原因是预测下一个字符主要靠附近的上下文。在这么小的模型、这么短的训练里，"只看附近"反而是一个有用的先验。（这个结论**不能**外推到大模型。）如果只看 loss，你会以为滑动窗口没有代价。

**实验二：大海捞针**（needle in a haystack）。序列长 96，全是随机的"填充字符"。其中一个位置藏着一根"针"（8 种之一）。最后一个位置是"提问"，模型要说出针是哪一种。针与提问的距离 d 在 1–95 之间均匀随机，所以可以按距离统计准确率：

```python
x[torch.arange(bsz), NEEDLE_T - 1 - d] = ans  # put the needle at distance d from the question
loss = F.cross_entropy(model(x)[:, -1], ans)  # calculate the loss only at the "question" position
```

同一个脚本的输出（每个距离测 64 条；8 选 1，瞎猜的准确率是 12.5%）：

| 配置 | d < 16（窗口内） | 16 ≤ d ≤ 60（靠接力） | d > 60（够不着） |
|---|---:|---:|---:|
| full | 100.0% | 100.0% | 100.0% |
| sliding | 100.0% | 100.0% | 11.9% |
| interleave | 100.0% | 100.0% | 100.0% |

按距离细分（每段 8 个距离的平均）：

| d 从 | 1 | 9 | 17 | 25 | 33 | 41 | 49 | 57 | 65 | 73 | 81 | 89 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| full | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| sliding | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 55% | 12% | 13% | 11% | 12% |
| interleave | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |

（57 这一段包含 d = 57…64，正好跨过 60 这条线，所以是 55%。）

结果和第 3 节的感受野表完全吻合：

- **纯滑动窗口在 d ≤ 60 时是 100%**：模型确实学会了一层层接力。
- **d 一超过 4 × (16 − 1) = 60，准确率立刻掉到瞎猜的水平**：够不着的位置就是用不上，训练再久也没用。
- **只把最后一层换成全注意力，所有距离都回到 100%**，而 KV cache 只多存这一层。

这就是局部-全局交替的全部道理：**局部层用低成本处理附近，少数全局层负责远距离检索**。这也提醒我们：评测长上下文模型不能只看困惑度，要用专门的检索类评测（大海捞针、RULER 一类，见第 15 章）。

## 7. 稀疏注意力：按内容挑，而不是按位置挑

滑动窗口的问题是挑键的方式太死板：永远是"最近的 W 个"。如果每个查询只负担得起 k 个键，为什么不挑**最相关**的 k 个？这就是**稀疏注意力**的想法。

[`code/05_topk_sparse.py`](code/05_topk_sparse.py) 拿第 6 节训练好的**全注意力**模型做实验。推理时，每个查询只保留 k 个键，不重新训练。脚本比较两种挑法：

```python
kth = att.topk(self.topk, dim=-1).values[..., -1:]  # the k-th largest score of each row
att = att.masked_fill(att < kth, float("-inf"))  # the other keys become invisible
```

| 挑法 | k | 捞针 d < 16 | 捞针 d > 60 | 全部距离 | LM loss |
|---|---:|---:|---:|---:|---:|
| 不限制（全注意力） | – | 100.0% | 100.0% | 100.0% | 1.797 |
| 最近 k 个 | 4 | 29.9% | 12.3% | 15.2% | 1.886 |
| 分数最高的 k 个 | 4 | 100.0% | 100.0% | 100.0% | 1.818 |
| 最近 k 个 | 8 | 54.9% | 12.3% | 19.2% | 1.838 |
| 分数最高的 k 个 | 8 | 100.0% | 100.0% | 100.0% | 1.806 |
| 最近 k 个 | 16 | 100.0% | 12.3% | 26.8% | 1.819 |
| 分数最高的 k 个 | 16 | 100.0% | 100.0% | 100.0% | 1.800 |

（"最近 k 个"在 d < 16 时也不到 100%。这个模型是用全注意力训练的，推理时突然改成窗口 4 或 8，连窗口内的针也会受影响。k = 16 时窗口覆盖了 d < 16，准确率就回到 100%。）

在同样的预算下，按位置挑（最近 k 个）完全找不到远处的针。按内容挑（分数最高的 k 个）在 k = 4 时就能保持 100%，语言建模 loss 也只涨了一点。注意力本来就是"稀疏"的：大部分权重集中在少数几个键上。

但这个演示有作弊的成分：为了挑出 top-k，它先把全部 T 个分数都算了一遍，所以没有省下任何算力。真实系统的做法是：

1. 用一个**很便宜的索引器**（indexer）给历史 token 打分。索引器维度小、头少，甚至可以用低精度。
2. 每个查询选出分数最高的 k 个 token（或 k 个块）。
3. 只对选中的这些 token 做精确的注意力。

**这算不算共识？** 我们用 GOAL.md 2.1 的规则 A：至少 3 个独立的头部家族在主力版本中明确采用。2026 年 9 月核实的结果如下：

| 家族 | 主力版本 | 做法 | 依据 |
|---|---|---|---|
| DeepSeek | V3.2（DSA）、V4（CSA + HCA） | 闪电索引器（64 个索引头、128 维）给每个 token 打分，每个查询选 top-2048（V4-Flash 为 512） | V3.2 模型卡 "DeepSeek Sparse Attention (DSA)"、`index_topk` 2048；V4 模型卡 "Compressed Sparse Attention"、`index_topk` 512 |
| 智谱 GLM | GLM-5 / 5.3 | 直接采用 DSA | GLM-5 模型卡 "GLM-5 also integrates DeepSeek Sparse Attention (DSA)"；`GlmMoeDsaForCausalLM`、`index_topk` 2048 |
| MiniMax | MiniMax-M3 | MSA：128 个 token 为一块，每个 GQA 组选 16 块（2048 个 token） | M3 模型卡 "MiniMax Sparse Attention (MSA)"；`sparse_attention_config` |
| 美团 LongCat | LongCat-2.0 | LSA：在 DSA 的索引器上加分层和跨层共享 | 模型卡 "LongCat Sparse Attention"；`index_topk` 2048 |

这是四个彼此独立的家族。GLM 和 LongCat 明确是在 DeepSeek 的 DSA 基础上改的，但它们是独立的模型家族，按规则计数。**这满足规则 A，所以写进正文。**但它的适用范围要说清楚：

- **四家全是 2000 亿参数以上的 MoE 旗舰**（DeepSeek-V4-Flash 284B、GLM-5 744B、MiniMax-M3 约 428B、LongCat-2.0 1.6T），目标是 100 万级上下文。在 10B 以下的小模型里，只查到 MiniCPM4 / 4.1（InfLLM-V2）。而同一家新出的 MiniCPM5-2B，`config.json` 又回到了普通的 `LlamaForCausalLM`。
- **它省算力，不省 KV cache。**将来哪个 token 会被选中，事先不知道。所以全部 K/V 都得留着，外加索引器自己的一份小 key。这和滑动窗口正好相反。DeepSeek V4 另外加了压缩（`compress_ratios`）和 128 的滑动窗口，才把缓存也压了下来。
- **具体做法还没有收敛**：选 token 还是选块、索引器怎么训练、要不要和压缩结合，各家都不一样（见本章后面的"前沿观察"）。

所以本课的立场是：**"用一个便宜的打分器挑 top-k，再做精确注意力"这个想法是共识，写进正文。具体变体放进前沿观察。主线模型（0.6–0.8B、32K 上下文）不用它。**

## 8. 小结

- **滑动窗口**：`mask = (j ≤ i) & (i − j < W)`，每个 token 只看最近 W 个位置。算力从 O(T²) 降到 O(T·W)，KV cache 封顶在 W。
- **感受野**：L 层滑动窗口理论上能接力 `L × (W − 1)` 个位置。实验里超过这个距离，检索立刻掉到瞎猜的水平。
- **局部-全局交替**：大部分层局部、少数层全局（Gemma 3 为 5:1，OLMo 3 为 3:1，gpt-oss 为 1:1）。全局层负责远距离检索，局部层节省缓存。128K 下 Gemma-3-27B 省 83%。
- **loss 看不出代价，大海捞针看得出**：评测长上下文要用检索类任务。
- **稀疏注意力**：按内容挑 top-k，而不是按位置挑；便宜的索引器 + 精确注意力。DeepSeek、GLM、MiniMax、LongCat 的旗舰都在用。它省算力，不省 KV cache。

---

## GPU 实测（单张 RTX 3090）

> **注意：**上面正文里的数字都来自 CPU 运行。本节换到一张 NVIDIA GeForce RTX 3090 上实测：24 GB 显存，Ampere 架构。规格表：BF16 张量核稠密峰值约 71 TFLOPS，FP32 约 35.6 TFLOPS，显存带宽约 936 GB/s。环境：PyTorch 2.11.0+cu128、CUDA 12.8，2026 年 10 月。
>
> 服务器把这张卡的功耗上限设成了 240 W（出厂默认 350 W）。持续满载时它会降频，所以算力、带宽的绝对值比满功耗的 3090 低，相对关系更可靠。没有 GPU 可以跳过本节。

运行：

```bash
uv run python chapters/22-local-sparse-attention/code/06_gpu_flex_window.py
```

我们在 GPU 上用三种算法计算同一个滑动窗口注意力。条件就是第 2 节的 `(j ≤ i) & (i − j < W)`。设置为 batch 1、16 个头、head_dim 128、BF16：

- **稠密因果 SDPA**：PyTorch 自带的 FlashAttention kernel，`is_causal=True`。窗口外的部分它也照算。它算的是全因果注意力，只作为速度参照。
- **布尔掩码 SDPA**：先造一张 T × T 的布尔掩码，再交给 SDPA 的 memory-efficient kernel（`attn_mask=`）。它先算出窗口外的分数，再把它们掩掉。`zero/arch/sliding_window.py` 现在用的就是这种写法。
- **FlexAttention**：`create_block_mask` 先把 T × T 的表按 128 × 128 切块，并标出整块落在窗口外的块。kernel 直接跳过这些块，只算带子经过的块。第一次调用要用 `torch.compile` 生成 Triton kernel，本机用了 7 秒。

固定 T = 16,384，改变窗口 W（毫秒，10 次取中位数）：

| 窗口 W | FlexAttention 要算的块（占因果全部块） | 稠密因果 SDPA | 布尔掩码 SDPA | FlexAttention | FlexAttention 比稠密快 |
|---:|---:|---:|---:|---:|---:|
| 128 | 3.1% | 22.68 | 70.19 | 0.90 | 25.2× |
| 512 | 7.6% | 23.35 | 70.53 | 2.47 | 9.5× |
| 1,024 | 13.5% | 23.93 | 70.80 | 3.64 | 6.6× |
| 4,096 | 44.8% | 23.25 | 71.48 | 9.37 | 2.5× |
| 16,384（= T，就是全因果） | 100.0% | 23.70 | 73.37 | 19.88 | 1.2× |

W = 1,024 时，除了 q、k、v 之外额外占用的显存：稠密 65 MiB；布尔掩码 SDPA 先要一张 256 MiB 的掩码，运行时再占 576 MiB；FlexAttention 66 MiB。

固定 W = 1,024，改变序列长度 T（毫秒）：

| 序列长度 T | 稠密因果 SDPA | 布尔掩码 SDPA | FlexAttention | FlexAttention 比稠密快 |
|---:|---:|---:|---:|---:|
| 4,096 | 1.62 | 5.00 | 0.94 | 1.7× |
| 16,384 | 22.72 | 71.24 | 3.61 | 6.3× |
| 65,536 | 473.55 | （跳过：布尔掩码本身就要 4 GiB） | 12.06 | 39.3× |

三种算法算的是同一个结果。T = 16,384 时，FlexAttention 与布尔掩码 SDPA 的最大绝对差是 3.9e-03（W = 128 时为 7.8e-03）。W = T 时，稠密因果 SDPA 与布尔掩码 SDPA 也差 3.9e-03。这些差都是 BF16 舍入的量级。

第 2 节说"每个查询的计算量从 O(T) 变成 O(W)"。这组数字说明，它在 GPU 上成立有一个前提：**kernel 必须真的跳过窗口外的块**。FlexAttention 的耗时跟着"要算的块"走：W 从 128 涨到 4,096，耗时从 0.90 ms 涨到 9.37 ms。固定 W = 1,024 时，T 每增大到 4 倍，它的耗时只涨到 3–4 倍（线性）。稠密因果的耗时却涨到 14–21 倍（平方）。到 T = 65,536 时，FlexAttention 已经快了 39 倍。

反过来，布尔掩码"只多一个 `&`"，但在 GPU 上没有省下任何时间。不管窗口多小，它都要 70 ms 左右，是全因果 FlashAttention 的 3 倍。它把 T × T 个分数全算了一遍，连因果掩码去掉的上三角也算了。它还要额外一个 T² 大小的掩码和偏置。有一点出乎意料：W = T 时，FlexAttention（19.88 ms）比 PyTorch 自带的 FlashAttention（23.70 ms）还快一点。FlexAttention 在 3090 + PyTorch 2.11 上直接可用，没有遇到兼容问题。

## 从极简代码到生产级代码

生产级实现在 [`zero/arch/sliding_window.py`](../../zero/arch/sliding_window.py)。它是第五部分的实验模块，**不用于主线模型**：

| 极简代码（`code/`） | 生产级代码（`zero/arch/sliding_window.py`） | 多做了什么、为什么 |
|---|---|---|
| `window_mask(q_pos, k_pos, window)` | `sliding_window_mask(q_pos, k_pos, window)` | 同一个条件。额外处理"空槽"（位置 −1 永远不可见），键可以乱序（环形缓冲区需要这一点）。 |
| 在 `VARIANTS` 里手写每层窗口 | `make_layer_types(n_layers, global_every=..., all_sliding=...)` | 输出和 HF `config.layer_types` 相同的字符串（`"sliding_attention"` / `"full_attention"`）。`global_every=6` 就是 Gemma 3 的 5:1。约定与 HF 的 `(i + 1) % pattern == 0` 一致。 |
| 自带的小 `Attention` | `SlidingWindowAttention(zero.model.Attention)` + `convert_to_sliding_window(model, layer_types, window)` | 继承主线的注意力：GQA、QK-Norm、RoPE、参数名全部相同。可以把任意 `zero` 模型原地改成局部-全局交替，并保留权重。用 PyTorch SDPA 加布尔掩码。 |
| `KVCache.append`：拼接后切片，保留最后 W 个 | `SlidingWindowKVCache`：全局层按 `max_seq_len` 预分配；滑动窗口层只分配 W 个槽位的**环形缓冲区**（位置 p → 槽位 `p % W`），另存每个槽位的位置 | 一次分配，不再拷贝。支持一次输入超过 W 个 token 的分块 prefill（先用"旧 + 新"算注意力，再只写回最后 W 个）。`nbytes()` 与 `kv_cache_bytes(...)` 的公式一致。 |
| 每步整段重算 / 截断缓存 | `generate_greedy(model, prompt, n, cache=None)` | 专门用来对拍"有界缓存"和"不用缓存"。 |
| `01_masks_and_ledger.py` 的 `kv_bytes` | `zero/arch/sliding_window.py` 的 `kv_cache_bytes(layer_types, window, ...)`。按 `config.json` 自动识别层类型的完整账本是第 21 章的 `zero/tools/kv_cache_calc.py`（已支持 `sliding` 层） | 同一个公式：全局层存 T 个位置，滑动窗口层存 min(W, T) 个。 |
| 手写 `q @ kᵀ` + 掩码 | SDPA + 自定义掩码（CPU） | GPU 上要真正跳过窗口外的块，需要 FlashAttention 的 `flash_attn_func(..., causal=True, window_size=(W − 1, 0))`，或 PyTorch FlexAttention 的滑动窗口 `mask_mod`。vLLM / transformers 读到 `sliding_window` 和 `layer_types` 后，会自动选择 kernel 和分层缓存。FlexAttention 的块稀疏滑动窗口已在 RTX 3090 上实测（本章"GPU 实测"一节：T = 65,536 时比稠密因果注意力快约 39 倍）。因为没装 `flash_attn`，FlashAttention 的 `window_size` 尚未验证。`zero/arch/sliding_window.py` 在 CUDA 上仍用布尔掩码 SDPA。它的正确性已验证（[runs/2026-10-01-gpu0-check](../../runs/2026-10-01-gpu0-check/README.zh.md) 第 11 节），但比全注意力还慢。 |

**对拍**：[`tests/test_arch_sliding_window.py`](../../tests/test_arch_sliding_window.py)（`uv run pytest tests/test_arch_sliding_window.py`；本机 8 项全部通过，几秒内跑完）：

- 掩码与定义逐格一致（W = 3 的 6×6 表）。每个查询最多看 W 个位置。空槽不可见，键可以乱序。
- `make_layer_types` 的三种排布：5:1、1:1、全部滑动。
- **窗口 ≥ 序列长度时，滑动窗口模型的 logits 与全注意力完全一致**。窗口为 8 时，前 8 个位置一致，之后开始不同。
- **环形缓存的贪心生成与不用缓存的生成逐 token 相同**（全部滑动、1:1 交替两种；提示词比窗口长；生成 60 个 token）。
- 分块 prefill（17 + 1 + 3 + 24 个 token，块有大有小）与一次性前向传播的 logits 在 1e-5 内一致。
- 用普通的 `zero.kv_cache.KVCache` 也能得到同样的结果（只是多存了用不到的 K/V）。
- `nbytes()` 等于"3 层 × 8 个槽位 + 1 层 × 128 个位置"的公式值。

稀疏注意力没有生产级实现。它需要专门训练的索引器和定制 kernel。（DeepSeek 开源了 FlashMLA 的稀疏版本和 DeepGEMM 的索引器 kernel，MiniMax 开源了 MSA 算子。）这超出了本课 CPU 实验的范围，而且主线模型用不到。

---

## 前沿观察

> **稀疏注意力的具体变体还在分化。** 共识只到"便宜打分 + top-k + 精确注意力"这一层，再往下各家不同。DeepSeek 先提出 **NSA**（Native Sparse Attention：压缩、选块、滑动窗口三个分支相加；2025 年的论文，未见用于旗舰）。之后是 **DSA**（逐 token 的闪电索引器，V3.2），再之后是 **CSA/HCA**（V4：先把 KV 按 4 倍或 128 倍压缩，再做稀疏选择；`compress_ratios` 逐层交替）。
>
> Kimi 提出过 **MoBA**（按块做类似 MoE 的路由），但 Kimi K3 的主力架构是 MLA + KDA 线性注意力混合，没有用 MoBA。面壁的 **InfLLM-V2**（可以在稠密和稀疏之间切换，不加参数）用在了 MiniCPM4 / 4.1，但 MiniCPM5 没有保留。有几个问题还没有定论：选 token 还是选块、索引器怎么训练、要不要和 KV 压缩结合。（DeepSeek 和 MiniMax 都先用全注意力热身，让索引器用 KL 散度拟合主注意力的分布，再切换到稀疏注意力，但细节不同。）

> **注意力汇聚点（attention sink）。** StreamingLLM（Xiao 等 2023）发现：只保留滑动窗口时，模型在丢掉开头几个 token 后会崩溃。原因是很多注意力头把"多余"的注意力堆在序列开头。保留开头几个 token（或加一个可学习的 sink），模型就能稳定地无限流式生成。gpt-oss 在每个注意力头上加了可学习的 sink（见模型卡）。MiMo-V2-Flash 的配置里有 `add_swa_attention_sink_bias`，MiniMax 则报告说 MSA 不需要强制保留开头。它和滑动窗口配合得很紧，但各家做法不同，还不算共识。

---

## 采用方与来源

| 技术 | 采用方（主力版本） | 来源 |
|---|---|---|
| 滑动窗口 + 局部-全局交替 | **Google Gemma**（Gemma 2：1:1、4096；Gemma 3：5:1、1024；Gemma 4：5:1、1024）、**OpenAI gpt-oss**（1:1、128）、**AllenAI OLMo 3**（3:1、4096）、**Mistral**（Mistral 7B v0.1 全部滑动 4096；Ministral 8B 3:1）、**StepFun** Step-3.5-Flash（3:1、512）、**小米 MiMo**-V2-Flash（5:1、128）；Meta Llama 4 用分块注意力（3:1、8192） | Gemma 2 arXiv:2408.00118；Gemma 3 arXiv:2503.19786；gpt-oss 模型卡 arXiv:2508.10925；Mistral 7B arXiv:2310.06825；各模型的 `config.json`（下方链接） |
| 学习型稀疏注意力（索引器 + top-k） | **DeepSeek**（V3.2 DSA、V4 CSA）、**智谱 GLM**（GLM-5、5.3：DSA）、**MiniMax**（M3：MSA）、**美团 LongCat**（2.0：LSA） | DeepSeek-V3.2 模型卡与技术报告；DeepSeek-V4 arXiv:2606.19348；GLM-5 arXiv:2602.15763；MiniMax Sparse Attention arXiv:2606.13392；LongCat-2.0 模型卡 |
| FlashAttention 的窗口支持 | 行业标准 kernel（GOAL.md 2.1 的 B 类） | <https://github.com/Dao-AILab/flash-attention>（`window_size` 参数） |

**共识判断（GOAL.md 2.1）**：

- 滑动窗口 / 局部-全局交替：Gemma、gpt-oss、OLMo、Mistral、StepFun、MiMo 至少六个独立家族在主力版本的配置里明确采用。这满足规则 A。
- 学习型稀疏注意力：DeepSeek、GLM、MiniMax、LongCat 四个独立家族在 2025-09 至 2026-07 的旗舰中明确采用，**满足规则 A，写进正文**。但它仅限 2000 亿参数以上、百万级上下文的 MoE 旗舰。具体变体放进"前沿观察"。第 21 章写作时只核实到 DeepSeek 和 GLM 两家，本章补充了 MiniMax-M3 与 LongCat-2.0（均为 2026 年发布）。

**模型配置与模型卡**（2026-09 通过 Hugging Face 读取）：
[gemma-2-9b](https://huggingface.co/google/gemma-2-9b/blob/main/config.json)、
[gemma-3-1b-pt](https://huggingface.co/google/gemma-3-1b-pt/blob/main/config.json)、
[gemma-3-27b-pt](https://huggingface.co/google/gemma-3-27b-pt/blob/main/config.json)、
[gemma-4-31B](https://huggingface.co/google/gemma-4-31B/blob/main/config.json)、
[gemma-4-E2B-it](https://huggingface.co/google/gemma-4-E2B-it/blob/main/config.json)、
[gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b/blob/main/config.json)、
[gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b/blob/main/config.json)、
[Olmo-3-1025-7B](https://huggingface.co/allenai/Olmo-3-1025-7B/blob/main/config.json)、
[Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1/blob/main/config.json)、
[Mistral-7B-v0.3](https://huggingface.co/mistralai/Mistral-7B-v0.3/blob/main/config.json)、
[Ministral-8B-Instruct-2410](https://huggingface.co/mistralai/Ministral-8B-Instruct-2410/blob/main/config.json)、
[Step-3.5-Flash](https://huggingface.co/stepfun-ai/Step-3.5-Flash/blob/main/config.json)、
[MiMo-V2-Flash](https://huggingface.co/XiaomiMiMo/MiMo-V2-Flash/blob/main/config.json)、
[Llama-4-Scout（unsloth 镜像）](https://huggingface.co/unsloth/Llama-4-Scout-17B-16E-Instruct/blob/main/config.json)、
[DeepSeek-V3.2](https://huggingface.co/deepseek-ai/DeepSeek-V3.2)（[V3.2-Exp config](https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/config.json)）、
[DeepSeek-V4-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/main/config.json)、
[GLM-5](https://huggingface.co/zai-org/GLM-5)（[config](https://huggingface.co/zai-org/GLM-5/blob/main/config.json)）、
[GLM-5.3 config](https://huggingface.co/zai-org/GLM-5.3/blob/main/config.json)、
[MiniMax-M3](https://huggingface.co/MiniMaxAI/MiniMax-M3)（[config](https://huggingface.co/MiniMaxAI/MiniMax-M3/blob/main/config.json)）、
[LongCat-2.0](https://huggingface.co/meituan-longcat/LongCat-2.0)（[config](https://huggingface.co/meituan-longcat/LongCat-2.0/blob/main/config.json)）、
[MiniCPM4.1-8B](https://huggingface.co/openbmb/MiniCPM4.1-8B)、
[MiniCPM5-2B config](https://huggingface.co/openbmb/MiniCPM5-2B/blob/main/config.json)、
[Kimi-K3 config](https://huggingface.co/moonshotai/Kimi-K3/blob/main/config.json)、
[Qwen3.5-0.8B config](https://huggingface.co/Qwen/Qwen3.5-0.8B/blob/main/config.json)。

说明：Gemma 2 的 `config.json` 只有 `sliding_window` 4096，没有 `layer_types`，"1:1 交替"取自技术报告。Gemma 3 27B 的配置里没有写 `sliding_window_pattern`（1B 版本写的是 6），"5:1"取自技术报告和 transformers 的默认值。Llama 4 的官方仓库需要申请权限，数字取自 unsloth 镜像。"3 层分块 + 1 层全局 NoPE"是我们按 `no_rope_layers` 和 transformers 的 Llama 4 实现做的解读。这个解读**待核实**（Meta 没有发布技术报告，只有博客）。Qwen（从 3.5 起用线性注意力混合，见第 23 章）和 Kimi（K3 用 KDA 混合）没有采用滑动窗口或稀疏注意力。

---

## 引导问题

1. 滑动窗口层和全局层要不要用同一个 RoPE 基频？看看 Gemma 3（`rope_local_base_freq` 10000 vs `rope_theta` 100 万）和 gpt-oss 的配置。想一想：一个只看 128 个位置的层，需要 RoPE 里那些"转一圈要几十万个位置"的低频维度吗？
2. 本章的大海捞针里，纯滑动窗口在 d ≤ 60 时是 100%。如果把填充字符换成"会干扰的"内容（比如也出现针的字符），接力还能这么完美吗？为什么信息靠中间 token 转手会越来越"挤"？
3. 两种设计都能省 KV cache："全部层滑动窗口 + 大窗口"和"交替 + 小窗口"。它们各有什么优缺点？为什么 Mistral 后来把滑动窗口关掉了，而 Gemma 用得越来越多？
4. 分块注意力（Llama 4 的 `attention_chunk_size`）和滑动窗口有什么区别？块的边界处会发生什么？它对 KV cache 和 kernel 有什么好处？
5. DSA 这类稀疏注意力不省 KV cache，那它在 decode 时到底省了什么？（提示：第 21 章说 decode 受限于显存带宽。每步要读多少 K/V？）
6. 如果主线模型（0.6–0.8B、32K 上下文）换成 3:1 局部-全局交替，KV cache 能省多少？代价可能在哪里？为什么 GOAL.md 3.3 还是决定不冒这个险？

## 动手任务

**任务 1（基础）**：修改 `01_masks_and_ledger.py` 的 `MODELS`，加上 Gemma-3-1B（26 层、1 个 KV 头、head_dim 256、窗口 512、5:1）和 gpt-oss-20b（24 层、8 个 KV 头、head_dim 64、窗口 128、1:1）。算出 128K 下各省多少。再想一想 Gemma 3 1B 为什么把窗口从 1024 降到 512。

**任务 2（核心）**：在 `02_swa_model.py` 的 `VARIANTS` 里加一个 `"interleave_first": [None, 16, 16, 16]`（全局层放在**第一**层）。重新训练大海捞针模型，和 `[16, 16, 16, None]` 比较。全局层放在哪一层有关系吗？为什么？（训练一个约 5–10 分钟。）

**任务 3（挑战）**：给 `02_swa_model.py` 的 `Attention` 加一个真正的"索引器"：一对维度只有 8 的小投影 `wq_idx`、`wk_idx`。用 `q_idx · k_idx` 挑 top-k，再对选中的键做正常注意力。先冻结全注意力模型，只训练索引器，让它的分数分布拟合主注意力的分布（用 KL 散度，这是 MiniMax MSA 的做法）。然后用 `05_topk_sparse.py` 的评测，看 k = 8 时能保住多少远距离捞针准确率。

---

## 想深入：CS336

本章对应斯坦福 CS336（Spring 2026）<https://cs336.stanford.edu/>：

- **第 4 讲：注意力的替代方案与 MoE。**这一讲说明为什么要替代全注意力，以及有哪些路线（局部/稀疏、线性、状态空间）。本章展开的是其中"局部与稀疏"这一支。讲义与录像见课程页。
- **第 3 讲：架构与超参**横向比较了各家的架构选择，也涉及局部-全局交替。
- 本章后半部分的学习型稀疏注意力（DSA、MSA 等 2025–2026 年的方法），**CS336 没有深入讲**，见本章参考文献。

---

## 本章参考文献

- Child, Gray, Radford, Sutskever. *Generating Long Sequences with Sparse Transformers*，2019：<https://arxiv.org/abs/1904.10509>
- Beltagy, Peters, Cohan. *Longformer: The Long-Document Transformer*（滑动窗口 + 全局 token），2020：<https://arxiv.org/abs/2004.05150>
- Jiang et al. *Mistral 7B*（滑动窗口、滚动缓冲区缓存、理论跨度），2023：<https://arxiv.org/abs/2310.06825>
- Gemma Team. *Gemma 2: Improving Open Language Models at a Practical Size*，2024：<https://arxiv.org/abs/2408.00118>
- Gemma Team. *Gemma 3 Technical Report*（5:1 局部-全局、窗口 1024、KV cache 与困惑度消融），2025：<https://arxiv.org/abs/2503.19786>
- OpenAI. *gpt-oss-120b & gpt-oss-20b Model Card*（交替的带状窗口注意力、可学习的 sink），2025：<https://arxiv.org/abs/2508.10925>
- Xiao et al. *Efficient Streaming Language Models with Attention Sinks*（StreamingLLM），2023：<https://arxiv.org/abs/2309.17453>
- Yuan et al. *Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention*（NSA），2025：<https://arxiv.org/abs/2502.11089>
- Lu et al. *MoBA: Mixture of Block Attention for Long-Context LLMs*，2025：<https://arxiv.org/abs/2502.13189>
- DeepSeek-AI. *DeepSeek-V3.2*（DSA）技术报告：<https://huggingface.co/deepseek-ai/DeepSeek-V3.2/blob/main/assets/paper.pdf>
- DeepSeek-AI. *DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence*，2026：<https://arxiv.org/abs/2606.19348>
- GLM Team. *GLM-5*，2026：<https://arxiv.org/abs/2602.15763>
- Lai et al. *MiniMax Sparse Attention*，2026：<https://arxiv.org/abs/2606.13392>
- MiniCPM Team. *MiniCPM4*，2025：<https://arxiv.org/abs/2506.07900>；*InfLLM-V2*，2025：<https://arxiv.org/abs/2509.24663>
- Dao. FlashAttention（`window_size` 参数）：<https://github.com/Dao-AILab/flash-attention>；PyTorch FlexAttention 博客（滑动窗口 `mask_mod` 示例）：<https://pytorch.org/blog/flexattention/>
- [1.5 万字速通 LLM 主流模型结构（Llama、Qwen、GLM、DeepSeek…）](https://zhuanlan.zhihu.com/p/2060741715095560795)：各家注意力结构的横向对比（`references.md` 已收录）
- 模型配置的链接见上方"采用方与来源"。

**下一章**：滑动窗口让一部分层不再为远处存 K、V，稀疏注意力让每个查询只读一小部分 K、V。但全局层和稀疏层的缓存仍然随长度增长。第 23 章走一条更彻底的路：完全不存 K、V，把全部历史压进一个固定大小的状态里。这就是线性注意力，以及 Qwen3.5 那种"3 层线性 + 1 层全注意力"的混合架构。
