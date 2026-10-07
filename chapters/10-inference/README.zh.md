# 第 10 章：推理 —— 让模型生成文字，并且生成得快

[English](README.md) · **中文**

> **目标**：读完这一章，你能写出带温度、top-k、top-p 的采样函数。你能给一个 Transformer 加上 KV cache，并验证它和不加缓存时生成的结果逐字相同。你还能用 `2 × 层数 × KV 头数 × head_dim × 序列长度 × 字节数` 算出任意模型的 KV cache 有多大、GQA 能省多少。

📺 **本章视频**：还没有发布。要在本机渲染，运行 `bash chapters/10-inference/video/build.sh`。
🧪 **本章自检**：学完后，在 Claude Code 里输入 `/ch10-inference`。

---

上一章我们搭出了一个完整的现代 Transformer：RMSNorm、RoPE、SwiGLU、因果注意力。它训练几分钟就能写出"莎士比亚腔"。但到目前为止，我们只关心**训练**（training）：给模型一整段文本，一次算出所有位置的损失。这一章回答一个新问题：**训练好的模型，怎样真正把话说出来？**

这个问题有两部分。第一部分是"说什么"。模型每一步给出一个概率分布。从中挑字的方法，决定了生成的文字是呆板还是灵活、是通顺还是胡言乱语。

第二部分是"说多快"：最朴素的生成方法里有大量重复计算。KV cache 省掉了这些计算，但带来了新问题：显存。KV cache 太大，于是有了 GQA。

本章所有代码都用一个自带的小模型（`code/01_tiny_model.py`）。它和第 9 章是同一套结构（Pre-Norm RMSNorm、RoPE、SwiGLU、共享 embedding）。为了在 CPU 上一两分钟训完，它改成了字符级（词表有 65 个字符），并去掉了 QK-Norm。它还多了本章的两样东西：KV cache 和 `n_kv_heads`。

```bash
uv run python chapters/10-inference/code/01_tiny_model.py   # train and save the weights (about 1.5 minutes on 1 thread, only once)
```

| 配置 | 数值 |
|---|---|
| 层数 / 宽度 / 查询头数 / head_dim | 4 / 128 / 4 / 32 |
| 参数量 | 861,440（0.86M） |
| 训练 | 600 步，batch 16 × 64 字符，AdamW + warmup + cosine |
| 验证集损失 | 1.838 nats/字符（均匀乱猜是 ln 65 = 4.174） |

> **注意：**本章训练类实验的数字来自课程构建机上的一次 CPU 运行。不同的机器、不同版本的底层数学库，浮点运算的顺序略有不同。训练几百步后，这些微小差异会被放大。你在本机跑出的数字可能从小数点后第二、三位开始就不一样。请以下文中不依赖具体数值的结论为准。2026-10 在另一台服务器上的复跑结果见 [runs/2026-10-01-gpu0-check/chapters-07-10.md](../../runs/2026-10-01-gpu0-check/chapters-07-10.md)。

## 1. 生成是一个循环

语言模型只会做一件事：看前面的全部内容，给"下一个 token"的每个候选打一个分数（logit）。要生成一段话，就把这件事放进循环：

1. 把当前序列输入模型，取**最后一个位置**的 logits。
2. 按某种策略从中挑出一个 token。
3. 把它接到序列末尾。然后回到第 1 步。

这叫**自回归生成**（autoregressive generation）：每一步的输出，都是下一步的输入。`01_tiny_model.py` 的末尾就是最朴素的写法：

```python
ids = data.encode("ROMEO:\n")
for _ in range(120):
    logits = model(torch.tensor([ids]))[0, -1]   # give the full sequence, keep only the last position
    ids.append(int(logits.argmax()))             # pick the largest one and append it
```

这四行代码里有两个问题。第 2 步"挑最大的"好不好？第 1 步"整段输入"快不快？下面逐个讨论。

## 2. 挑哪个字：贪心、温度、top-k、top-p

### 2.1 贪心：稳妥，但会原地打转

每一步都挑概率最大的 token，叫**贪心解码**（greedy decoding）。它看起来最稳妥。但运行 `02_sampling.py`，看看它写出了什么：

```
[greedy (T=0)]  distinct 4-gram ratio 0.27
  the son the such the shall the shall the proves
  The shall the shall the such the shall the shalleend the shand
```

"the shall the shall"：文字在原地打转。我们用"不重复的 4 字符片段占比"粗略衡量重复程度。在 200 个字符里，贪心解码只有 0.27。每一步都选局部最优，拼起来不是全局最好的文字。一旦走进一个循环，下一步最可能的字又把它带回循环里。

大模型上也有这种退化现象。Holtzman 等人 2019 年的论文专门研究了它，标题是《神经文本退化的奇特案例》（*The Curious Case of Neural Text Degeneration*）。

解决办法是**抽样**（sampling）：按模型给出的概率随机抽一个字。概率大的字更容易被抽中，但不是每次都被抽中。

### 2.2 温度：抽样之前先调整分布

第 5 章讲过带温度的 softmax：先把 logits 除以温度 T，再做 softmax。

```
p_i = softmax(z / T)_i
```

这对应 `02_sampling.py` 里的 `filtered_probs`：

```python
probs = torch.softmax(logits / temperature, dim=-1)  # softmax with temperature (Chapter 5)
```

下面是提示词 `"ROMEO:\nI will "` 之后，小模型给出的**真实**下一字符分布（前 8 名，`␣` 是空格）：

| 温度 | t | a | n | s | h | m | b | w | 熵（nats） |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| T = 0.5 | 0.427 | 0.093 | 0.073 | 0.066 | 0.062 | 0.060 | 0.047 | 0.037 | 2.15 |
| T = 1.0 | 0.173 | 0.081 | 0.071 | 0.068 | 0.066 | 0.065 | 0.057 | 0.051 | 3.00 |
| T = 1.5 | 0.106 | 0.064 | 0.059 | 0.057 | 0.056 | 0.055 | 0.051 | 0.047 | 3.39 |

T < 1 放大差距：t 的概率从 0.17 涨到 0.43。分布更尖，生成更保守。T > 1 缩小差距：t 降到 0.11，冷门字符也有了机会。生成更大胆，也更容易胡言乱语。T → 0 就是贪心解码，所以代码在 `temperature == 0` 时直接用 `argmax`。

### 2.3 截掉长尾：top-k 与 top-p

即使温度合适，分布的尾部还有几十个很不靠谱的 token。每个的概率都很小，但加起来并不小。生成几百个 token，迟早会抽中其中一个。一旦抽中，后面的文字就被带偏了。所以常常在抽样前截掉长尾，再重新归一化：

- **top-k**（Fan 等 2018）：只保留概率最大的 k 个。
- **top-p**，也叫 **nucleus 采样**（Holtzman 等 2019）：把概率从大到小排序，累加到刚好不小于 p 为止，只在这一小撮"核"里抽。

```python
if top_k > 0:  # keep only the k most probable tokens
    kth = torch.topk(probs, top_k).values[-1]
    probs = torch.where(probs >= kth, probs, 0.0)
if top_p < 1.0:  # nucleus: add from the largest down; stop when the sum reaches top_p
    sorted_p, idx = torch.sort(probs, descending=True)
    before = torch.cumsum(sorted_p, 0) - sorted_p  # cumulative probability before this token
    keep = torch.zeros_like(probs, dtype=torch.bool)
    keep[idx] = before < top_p  # this always keeps the largest one
    probs = torch.where(keep, probs, 0.0)
return probs / probs.sum()
```

top-p 比 top-k 好在哪里？看两个上下文：

| 上下文 | 最可能的下一个字符 | top-p = 0.9 留下几个 | top-k = 5 留下几个 |
|---|---|---:|---:|
| `"KING RICHARD III:\nWhat is th"`（很确定） | `e`，p = 0.546 | 5 | 5 |
| `"ROMEO:\n"`（台词刚开头，很不确定） | `A`，p = 0.123 | 15 | 5 |

模型有把握时，核很小。模型没把握时，核自动变大。top-k 不管模型有没有把握，永远留 k 个。没把握时，它砍掉了合理的候选；有把握时，它又可能留下离谱的候选。这就是 Holtzman 等人提出 top-p 的理由。实践中两者也常常一起用。

### 2.4 实际效果，和开源模型的默认配置

同一提示词、同一随机种子，每种策略各生成 200 个字符（`02_sampling.py` 的输出，只取开头）：

| 策略 | 不重复 4-gram 占比 | 生成开头 |
|---|---:|---|
| 贪心（T = 0） | 0.27 | `the son the such the shall the shall the proves` |
| T = 0.5 | 0.84 | `there his the prester of thee hand,` |
| T = 1.0 | 0.95 | `thy forful haves: / WeWadHis nears! younk gelst to you` |
| T = 1.0，top-k = 5 | 0.91 | `thy comman haves their soul that` |
| T = 1.0，top-p = 0.9 | 0.94 | `thou more the prevenced can that` |
| T = 1.5 | 0.98 | `thyremhis kiss selted? / His nears! youd Igelst, gMycouuesy` |

一个 86 万参数的字符级模型写不出真正通顺的英文。但趋势很清楚：贪心解码重复；T = 1.5 产生乱码（`gMycouuesy`）；截掉长尾之后（top-k、top-p），怪词明显变少。

不重复占比只衡量"不重复"，衡量不了"通顺"。所以 T = 1.5 的 0.98 不代表它最好。这也是解码参数最终要靠人看、靠评测来定的原因。

真实模型怎样选？开源模型会随权重一起发布 `generation_config.json`。这个文件里是官方推荐的默认解码参数（2026-09 读取）：

| 模型 | 温度 | top-p | top-k |
|---|---:|---:|---:|
| Qwen3-8B | 0.6 | 0.95 | 20 |
| Qwen2.5-7B-Instruct | 0.7 | 0.8 | 20 |
| Llama-3.1-8B-Instruct | 0.6 | 0.9 | — |
| SmolLM3-3B | 0.6 | 0.95 | — |
| Gemma-3-27B-it | —（默认 1.0） | 0.95 | 64 |

"温度略低于 1 + top-p 0.8–0.95"几乎是标准配置。

## 3. 朴素生成为什么慢：一个三角形的重复计算

回到第 1 节那四行代码，看 `model(torch.tensor([ids]))` 这一行。每生成一个字，代码都把**整段**序列重新输入模型。设提示词长 P。生成第 t 个字时，模型要处理 P + t 个位置。生成 n 个字，一共处理这么多个位置：

```
Σ (P + t) ≈ P·n + n²/2   个位置
```

画出来是一个三角形：第 1 步处理 P 个位置，第 2 步处理 P + 1 个，依此类推。本章的提示词 `"ROMEO:\nI will "` 有 14 个字符。生成 512 个字符，要处理 137,984 个位置。

但请仔细想一想：在因果注意力里，位置 i 只看位置 ≤ i 的内容。后面来了新字，**过去位置的所有中间结果都不会变**。三角形里的每一行，除了最后一个新位置，全是上一步算过的东西。

## 4. KV cache：把算过的 K、V 存起来

哪些中间结果值得存？看注意力（第 8 章）。新位置 t 要计算：

```
out_t = softmax(q_t · [k_1, …, k_t]ᵀ / √d) · [v_1, …, v_t]
```

它需要自己的 query `q_t`，以及**所有**位置的 key 和 value。过去位置的 `k_i`、`v_i` 不会变，那就在每层存一份。这就是 **KV cache**。

之后每一步只输入一个新 token。模型算出它的 q、k、v，把 k、v 追加到缓存末尾，再用 q 和缓存里全部的 K、V 做注意力。**Q 不用缓存**：`q_t` 只在第 t 步用一次。

极简实现只改了注意力里的两处（`01_tiny_model.py`）：

```python
q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)  # the cache keeps K after the rotation
if cache is not None:
    k, v = cache.append(layer, k, v)  # old K/V + new K/V (torch.cat)
S = k.shape[2]  # total visible length = past + now
...
# Causal mask: the global position of new token i is S-T+i.
# It can see only the keys at positions ≤ its own position.
i = torch.arange(T)[:, None] + (S - T)
j = torch.arange(S)[None, :]
att = att.masked_fill(j > i, float("-inf")).softmax(-1)
```

另外还有一个细节：RoPE 需要知道新 token 的**绝对位置**。所以 `TinyLM.forward` 用 `start = len(cache)` 取出对应位置的 cos/sin 值。生成循环变成（`03_kv_cache.py`）：

```python
cache = tiny.KVCache(model.c.n_layers)
logits = model(torch.tensor([prompt]), cache)[0, -1]      # prefill: give the full prompt at once
for i in range(n):
    nxt = samp.sample_next(logits, g=g, **kw)
    logits = model(torch.tensor([[nxt]]), cache)[0, -1]  # decode: give only 1 new character
```

生成 512 个字，处理的位置从 137,984 个降到 525 个（14 + 511）。

### 4.1 对拍：结果必须逐字相同

缓存只是"少算重复的东西"，不应该改变任何结果。运行：

```bash
uv run python chapters/10-inference/code/03_kv_cache.py
```

```
1. Parity check: do the cached and the naive versions generate the same 200 characters?
  greedy                 same: True   start: 'the son the such the shall the shall the'
  sample T=1.0 top-p=0.9 same: True   start: 'thou more the prevenced can that\nhave wi'
  max logits difference at the same position: 4.3e-06 (the size of floating-point rounding)
  cache size 872448 bytes = 2 × 4 layers × 4 KV heads × 32 × 213 positions × 4 bytes = 872448
```

采样模式也一致。两边用同一个种子的随机数生成器，每一步看到的分布（在浮点误差内）相同。logits 差 4.3 × 10⁻⁶，原因是两种算法里矩阵乘法的形状不同，累加顺序也不同。

### 4.2 测速

输出相同，快了多少？（单线程 CPU，每项取两次中较快的一次，贪心解码）

| 新生成 | 朴素（秒） | KV cache（秒） | 加速 | 朴素共处理位置 | 缓存共处理位置 |
|---:|---:|---:|---:|---:|---:|
| 64 | 0.33 | 0.16 | 2.1× | 2,912 | 77 |
| 128 | 0.78 | 0.30 | 2.6× | 9,920 | 141 |
| 256 | 2.55 | 0.58 | 4.4× | 36,224 | 269 |
| 512 | 10.47 | 1.34 | 7.8× | 137,984 | 525 |

序列越长，加速越大：朴素版的代价按 n² 增长，缓存版按 n 增长。加速比远小于"处理位置"之比（137,984 / 525 ≈ 263），原因有两个。第一，小模型上每一步都有固定开销（Python 循环、小矩阵乘法的调用）。第二，缓存版每步仍要让新的 q 和全部历史 K 做注意力。

计时受机器负载影响：视频里用的另一次运行是 1.9× / 2.7× / 4.4× / 9.8×。

### 4.3 两个阶段：prefill 与 decode

有了缓存，推理分成两个阶段：

- **prefill**：把整段提示词一次输入模型。所有位置并行计算，同时把 K、V 写进缓存。
- **decode**：之后一次只算一个新 token。

两个阶段处理同样的 256 个位置（`03_kv_cache.py` 第 3 部分）：

```
prefill, 256 at a time:   16.9 ms  →    15168 positions/s
decode,  1 at a time:    691.5 ms  →      370 positions/s
prefill throughput is 41× the decode throughput
```

（视频里用的另一次运行是 19.0 ms 对 585 ms，31 倍。具体倍数取决于机器，量级是几十倍。）decode 每一步只有一个 token 的计算量，却要把全部权重和整个 KV cache 读一遍。硬件大部分时间花在搬数据上，算力用不满。

所以推理服务要把很多用户的请求**拼成一批（batch）一起 decode**：读一遍权重，服务几十个请求。这也意味着 KV cache 要按用户数成倍增加。

## 5. KV cache 的显存账

缓存省了计算，却要占显存。每一层、每个位置都要存一份 K 和一份 V。每份是 `KV 头数 × head_dim` 个数：

```
KV cache 字节数 = 2 × 层数 × KV 头数 × head_dim × 序列长度 × 每个数的字节数（× batch）
```

对应 `04_kv_memory.py` 里的函数：

```python
def kv_bytes(layers, kv_heads, head_dim, seq_len, bytes_per=BF16, batch=1):
    return 2 * layers * kv_heads * head_dim * seq_len * bytes_per * batch
```

代入主线模型的配置（`configs/main/pretrain.toml`：28 层，16 个查询头，8 个 KV 头，head_dim 128，BF16）：

```bash
uv run python chapters/10-inference/code/04_kv_memory.py
```

| 方案 | KV 头 | 每 token | 4K 上下文 | 32K 上下文 |
|---|---:|---:|---:|---:|
| MHA（不共享） | 16 | 224 KiB | 896 MiB | 7.00 GiB |
| **GQA（主线）** | **8** | **112 KiB** | **448 MiB** | **3.50 GiB** |
| MQA（全共享） | 1 | 14 KiB | 56 MiB | 0.44 GiB |

每个 token 是 2 × 28 × 8 × 128 × 2 = 114,688 字节 = 112 KiB。作为对照：主线模型有 689.5M 参数，BF16 权重是 1.28 GiB。**一条 32K 上下文的对话，KV cache 就是权重的 2.7 倍。**同时服务 16 条这样的对话，缓存要 56 GiB（MHA 要 112 GiB）。推理的瓶颈从"算得慢"变成了"存不下"。

## 6. GQA：让几个查询头共享一组 K、V

公式里的层数和 head_dim 都和模型能力直接相关，不好改。序列长度由用户决定，也不能改。剩下的是 **KV 头数**。

标准的**多头注意力**（Multi-Head Attention，MHA）里，每个查询头都有自己的一组 K、V。Shazeer 在 2019 年提出了一个极端做法：所有查询头**共用一组** K、V。这叫**多查询注意力**（Multi-Query Attention，MQA）。它把 KV cache 缩小到原来的 1/头数，但质量会有损失。

Ainslie 等人在 2023 年提出了折中办法：把查询头分成若干组，每组共用一组 K、V。这叫**分组查询注意力**（Grouped-Query Attention，GQA）。论文的实验结论是：GQA 的质量接近 MHA，速度接近 MQA。MHA 和 MQA 是它的两个端点（组数 = 头数，组数 = 1）。

实现上的改动很小。K、V 的投影变窄（输出 `n_kv_heads × head_dim`）。做注意力之前，把每组 K、V 复制给组内的查询头：

```python
self.wk = nn.Linear(c.dim, c.n_kv_heads * c.head_dim, bias=False)  # GQA: narrower K/V projections
...
g = c.n_heads // c.n_kv_heads
k, v = k.repeat_interleave(g, dim=1), v.repeat_interleave(g, dim=1)  # g query heads share one set
```

关键在于复制发生在**缓存之后**：缓存里存的是 `n_kv_heads` 份，而不是 `n_heads` 份。

### 6.1 小实验：同样训练 600 步

`05_gqa.py` 用同样的 4 个查询头，只改 KV 头数。种子、数据顺序、训练步数都相同：

```bash
uv run python chapters/10-inference/code/05_gqa.py   # the first run trains 3 models, about 5 minutes on 1 thread
```

| 方案 | KV 头 | 验证集损失 | 注意力参数 | 总参数 | KV cache（生成 512 字后，FP32） |
|---|---:|---:|---:|---:|---:|
| MHA | 4 | 1.838 | 262,144 | 861,440 | 2,150,400 B（1.00×） |
| GQA | 2 | 1.867 | 196,608 | 795,904 | 1,075,200 B（0.50×） |
| MQA | 1 | 1.860 | 163,840 | 763,136 | 537,600 B（0.25×） |
| 对照：MHA 换随机种子 1 | 4 | 1.869 | | | |

（2026-10 在另一台服务器上复跑的验证集损失：MHA 1.831、GQA 1.865、MQA 1.860，MHA 换种子后 1.861。）

KV cache 严格按 KV 头数成比例缩小。参数也少了一点，因为 K、V 投影变窄了。

脚本也打印了生成 512 个字的时间。但在这个小模型和单线程 CPU 上，这个时间主要反映机器负载。两次运行分别是 1.79 / 1.50 / 1.21 秒和 3.62 / 4.09 / 4.89 秒，连大小顺序都反了。所以这里看不出 GQA 的速度差别。GQA 的速度收益来自 decode 时要读的 KV cache 变少。只有在显存带宽成为瓶颈的 GPU 上、在长上下文和大 batch 时才明显（第 21 章）。

损失呢？MHA 看起来最好。但只换一个随机种子，同样的 MHA 就差了 0.031。这和三者之间的差距（最大 0.029）是同一个量级。

在另一台服务器上复跑，换种子差 0.030，三者之间最大差 0.034，仍是同一个量级。**在这个规模上，GQA/MQA 的质量代价和随机种子带来的波动差不多大，两者分不开。**不能从这张表得出"谁更好"的结论。

要看清质量差距，需要多个种子、更大的模型和更长的训练。GQA 论文做的正是这件事。第 21 章会在 CPU 上更仔细地比较 MHA / GQA / MLA。

### 6.2 公开模型怎么选

这是 `04_kv_memory.py` 的第 2 部分。数字取自各模型 Hugging Face 仓库里的 `config.json`（32K 上下文，BF16，batch 1，按公式把所有层都当作全注意力计算）：

| 模型 | 层 | Q 头 | KV 头 | head_dim | 每 token | 32K 上下文 | 若不共享（MHA） | 省下 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-0.6B | 28 | 16 | 8 | 128 | 112 KiB | 3.50 GiB | 7.00 GiB | 2× |
| SmolLM3-3B | 36 | 16 | 4 | 128 | 72 KiB | 2.25 GiB | 9.00 GiB | 4× |
| Qwen2.5-7B | 28 | 28 | 4 | 128 | 56 KiB | 1.75 GiB | 12.25 GiB | 7× |
| Llama-3.1-8B | 32 | 32 | 8 | 128 | 128 KiB | 4.00 GiB | 16.00 GiB | 4× |
| Qwen3-8B | 36 | 32 | 8 | 128 | 144 KiB | 4.50 GiB | 18.00 GiB | 4× |
| Mistral-7B-v0.1 | 32 | 32 | 8 | 128 | 128 KiB | 4.00 GiB | 16.00 GiB | 4× |
| gpt-oss-20b | 24 | 64 | 8 | 64 | 48 KiB | 1.50 GiB | 12.00 GiB | 8× |
| Gemma-3-27B | 62 | 32 | 16 | 128 | 496 KiB | 15.50 GiB | 31.00 GiB | 2× |
| Llama-3.3-70B | 80 | 64 | 8 | 128 | 320 KiB | 10.00 GiB | 80.00 GiB | 8× |

主线模型的 KV 配置和 Qwen3-0.6B 完全相同（28 层、8 个 KV 头、head_dim 128），所以每 token 的缓存也一样。

下面是几点说明。Mistral 7B 用了 4096 的滑动窗口。gpt-oss 一半的层是窗口为 128 的滑动注意力。Gemma 3 大部分层是窗口为 1024 的局部注意力。这些模型实际的缓存比表里小，这是第 22 章的主题。Gemma 3 的 1B 版本更激进，直接用了 MQA（1 个 KV 头）。

## 7. 小结

- **自回归生成**：算分布 → 挑一个 token → 接到末尾 → 重复。
- **解码策略**：贪心解码会原地打转。温度调整分布的尖锐程度。top-k 固定留 k 个。top-p 按累计概率留，个数随模型的把握自动伸缩。开源模型的默认配置多是温度 0.6–0.7 + top-p 0.8–0.95。
- **KV cache**：过去位置的 K、V 不会变，所以存起来。每步只算新 token。输出逐字不变，代价从 O(n²) 降到 O(n)。
- **prefill / decode**：prefill 并行计算，能用满硬件。decode 一次一个 token，用不满硬件，所以服务要批处理。
- **显存账**：`2 × 层数 × KV 头数 × head_dim × 序列长度 × 字节数`。主线模型每 token 112 KiB，32K 上下文 3.5 GiB，是权重的 2.7 倍。
- **GQA**：多个查询头共享一组 K、V，缓存按比例缩小。MHA 和 MQA 是它的两个端点。

---

## GPU 实测（单张 RTX 3090）

> 上面正文里的数字都来自 CPU 运行。本节换到一张 NVIDIA GeForce RTX 3090 上实测（24 GB 显存，Ampere 架构）。规格表上的数值：BF16 张量核稠密峰值约 71 TFLOPS，FP32 约 35.6 TFLOPS，显存带宽约 936 GB/s。环境：PyTorch 2.11.0+cu128、CUDA 12.8，2026 年 10 月。
>
> 服务器把这张卡的功耗上限设成了 240 W（出厂默认 350 W）。持续满载时它会降频，所以算力和带宽的绝对值比满功耗的 3090 偏低，看相对关系更可靠。没有 GPU 可以跳过本节。

运行：

```bash
uv run python chapters/10-inference/code/06_gpu_prefill_decode.py
```

模型还是 `01_tiny_model.py` 里的 `TinyLM`（脚本原样加载代码）。只把尺寸换成 Llama 3 8B 一层的形状：d = 4096、32 个查询头、8 个 KV 头、FFN 14336。模型叠了 4 层：共 0.87B 参数，BF16 权重 1.63 GiB，随机初始化（只测速度）。按规格带宽，把这些权重从显存读一遍至少要 1.86 ms。

计时分两种。"eager"就是平常那样一个算子一个算子地调用。"CUDA graph"先把一次前向传播的全部 kernel 录下来，再把整张图一次性重放，不经过 Python。它量到的是 GPU 自己干活的时间。

先用 `03_kv_cache.py` 的两个生成函数做贪心生成，看平均每个 token 的耗时（eager，3 次取中位数）：

| 新生成 | 朴素（ms/token） | KV cache（ms/token） | 加速 |
|---:|---:|---:|---:|
| 64 | 6.44 | 5.77 | 1.1× |
| 256 | 9.21 | 5.14 | 1.8× |
| 512 | 13.89 | 5.21 | 2.7× |

再看一次前向传播输入 T 个 token（就是 prefill T 个 token）要多久。后三列按 CUDA graph 的时间算，括号里是占规格峰值的比例：

| T | eager（ms） | CUDA graph（ms） | token/s | 算力 TFLOPS | 读权重 GB/s |
|---:|---:|---:|---:|---:|---:|
| 1 | 4.70 | 2.65 | 378 | 0.7（1%） | 659（70%） |
| 8 | 5.17 | 2.69 | 2,978 | 5.2（7%） | 650（69%） |
| 32 | 4.69 | 3.19 | 10,019 | 17.5（25%） | 547（58%） |
| 128 | 6.78 | 5.95 | 21,507 | 37.6（53%） | 293（31%） |
| 512 | 20.30 | 20.39 | 25,106 | 44.2（62%） | 86（9%） |
| 2048 | 88.13 | 87.99 | 23,275 | 42.2（59%） | 20（2%） |

最后是 decode。每条序列每步只输入 1 个 token，batch 从 1 加到 64（CUDA graph；"读显存"按"全部权重 + 全部 KV cache 各读一遍"计算）：

| 上下文 | batch | 每步（ms） | token/s | KV cache | 读显存 GB/s |
|---:|---:|---:|---:|---:|---:|
| 64 | 1 | 2.77 | 360 | 1 MiB | 630（67%） |
| 64 | 8 | 3.13 | 2,553 | 8 MiB | 560（60%） |
| 64 | 32 | 4.36 | 7,333 | 32 MiB | 408（44%） |
| 64 | 64 | 6.21 | 10,311 | 64 MiB | 292（31%） |
| 512 | 1 | 2.91 | 344 | 8 MiB | 603（64%） |
| 512 | 8 | 4.58 | 1,747 | 64 MiB | 396（42%） |
| 512 | 32 | 10.04 | 3,188 | 256 MiB | 201（21%） |
| 512 | 64 | 17.12 | 3,737 | 512 MiB | 133（14%） |

第二张表在真实硬件上展示了 4.3 节说的"decode 算力用不满"。T 从 1 涨到 32，一次前向传播几乎一样快（2.65 → 3.19 ms）。这段时间里 GPU 主要在把 1.63 GiB 权重从显存搬出来（550–660 GB/s），算力只用了峰值的 1%–25%。这是 decode 的处境。

T 过了一百左右，耗时才随 T 线性增长。这时 token/s 停在 2.3–2.5 万，算力用到峰值的六成左右。这是 prefill 的处境。

拐点落在 32 和 128 之间。这和脚本打印的"峰值算力 ÷ 带宽 ≈ 76 FLOP/字节"一致：一次输入 T 个 token，每个 2 字节的参数要做 2T 次浮点运算，所以算术强度（arithmetic intensity，每读 1 字节做的浮点运算次数）大约就是 T。

第三张表是同一件事的另一面。上下文为 64 时，batch 从 1 加到 64，每步只从 2.77 ms 涨到 6.21 ms。吞吐涨了约 29 倍：读一遍权重，服务了 64 条序列。

但上下文为 512、batch 为 64 时，KV cache 有 512 MiB，每步都要整块读一遍（极简版的 `torch.cat` 和 `repeat_interleave` 还要再复制一次）。吞吐只剩 3,737。缓存开始和权重抢带宽。这就是第 5 节那笔显存账，也是第 21 章的主题。

第一张表里 KV cache 只快了 1.1–2.7 倍，没有 4.2 节单线程 CPU 上那么明显。原因也在第二张表：在 GPU 上，重算几十个 token 和算 1 个 token 几乎一样贵。序列要足够长，朴素版的浪费才显现出来。

出乎意料的是 eager 和 CUDA graph 的差距。T = 1 时，eager 要 4.70 ms，重放只要 2.65 ms。将近一半时间花在 Python 逐个发射 kernel 上（极简模型每层都有几十个小算子：RMSNorm、RoPE、拼接缓存、复制 K/V 等）。vLLM 这类推理引擎在 decode 阶段录制 CUDA graph，就是为了省掉这部分时间。

## 从极简代码到生产级代码

本章的三件事在主线模型里的对应如下：

| 极简代码（`code/`） | 生产级代码（`zero/`） | 多做了什么、为什么 |
|---|---|---|
| `02_sampling.py` 的 `sample_next(logits, temperature, top_k, top_p)`，一次处理一个序列 | `zero/generate.py` 的 `sample_next(logits, temperature, top_p, generator)` | 输入是 `(B, V)`，一次处理一批序列。在 float32 上做 softmax（避免 BF16 推理时的精度问题）。top-p 的"至少留一个"写法相同。zero 目前没有 top-k（第 20 章本地 demo 需要时再加）。 |
| 手写的 `generate_cached` 循环 | `zero/generate.py` 的 `generate(...)` 和 `generate_stream(...)` | 支持 batch、用 `eos_id` 提前停止（已结束的序列补 eos）、用 `seed` 保证可复现。`generate_stream` 边生成边产出，命令行里能一个字一个字地打印。`use_cache=False` 保留朴素路径，专门用于对拍。 |
| `KVCache.append`：每步用 `torch.cat` 拼接 | `zero/kv_cache.py` 的 `KVCache`：按 `(层, batch, n_kv_heads, max_seq_len, head_dim)` **一次性预分配**。`update(layer, start_pos, k, v)` 只往里写。`nbytes()` 报告占用 | `torch.cat` 每一步都要重新分配、拷贝整块缓存，长序列上很浪费。预分配之后，写入是 O(1) 的，显存占用也从一开始就确定。（`04_kv_memory.py` 第 3 部分验证了 `nbytes()` 与公式一致：主线配置 1024 个位置为 117,440,512 字节。） |
| 用 `len(cache)` 决定新 token 的位置 | `Transformer.forward(tokens, kv_cache, start_pos)` 显式传入位置 | 支持**分块 prefill**（chunked prefill）：已有历史时，一次也能输入多个 token。为此 `zero/model.py` 的 `Attention.forward` 单独构造了掩码（`j <= past + i`）。这对第 15 章的长上下文很重要。 |
| 用 `repeat_interleave` 把 K/V 复制 g 份再做注意力 | `zero/model.py` 的 `Attention`：`F.scaled_dot_product_attention(q, k, v, ..., enable_gqa=self.n_kv_heads != self.n_heads)` | SDPA 在内部广播 K/V 头，不必真的复制出 `n_heads` 份张量。同时它能用上 FlashAttention 等融合 kernel（第 14 章）。 |
| 手写 `q @ kᵀ`、掩码、softmax | 同上，SDPA | 数学相同。融合 kernel 更快、更省显存。 |
| 无 | 在 QK-Norm 之后、RoPE 旋转之后再写入缓存；YaRN 长上下文缩放 | 从缓存读出来的 K 可以直接用。与位置相关的计算都在写入前完成。 |

**对拍**：`tests/test_kv_cache.py`（`uv run pytest tests/test_kv_cache.py`，本机 7 项全部通过）保证以下几点：

- 贪心生成：缓存版与 `use_cache=False` 的朴素版生成的 40 个 token 完全一致。
- batch 为 3、MQA（`n_kv_heads=1`）、开启 YaRN 时也完全一致。
- 带温度和 top-p 的**采样**、固定种子时：缓存版、朴素版、再跑一次的缓存版，三者结果一致。
- 分块 prefill（20 + 1 + 29 个 token 分三次输入）的 logits 与一次性前向传播的结果在 1e-5 内一致。
- `eos_id` 能正确截断输出。top-p 很小时，`sample_next` 只留下最大的 token。`KVCache.nbytes()` 等于 2 × 层 × batch × KV 头 × 长度 × head_dim × 4。

**真正上线的服务**不会用这样的循环。行业标准是 **vLLM**。它的 **PagedAttention** 把 KV cache 切成固定大小的"页"。它像操作系统管理虚拟内存那样按需分配这些页，并在请求之间共享。这样就不必为每个请求预留最大长度，避免了浪费。

vLLM 还配合**连续批处理**（continuous batching）：一个请求生成完，下一个请求立刻补进来，而不是等整批都结束。这两种方法一起把 GPU 喂饱。第 20 章发布主线模型时会用到 vLLM，这里不展开。

---

## 采用方与来源

| 技术 | 采用方（主力版本） | 来源 |
|---|---|---|
| KV cache | 所有自回归 Transformer 推理的标准做法：Hugging Face transformers（`use_cache`，各模型 `config.json` 里 `"use_cache": true`）、vLLM、llama.cpp | 行业标准（GOAL.md 2.1 的 B 类）；vLLM 论文 |
| 温度 + top-p（nucleus）采样 | Qwen3（0.6 / 0.95 / top-k 20）、Qwen2.5-Instruct（0.7 / 0.8 / top-k 20）、Llama 3.1 Instruct（0.6 / 0.9）、SmolLM3（0.6 / 0.95）、Gemma 3 it（top-p 0.95 / top-k 64） | 各模型的 `generation_config.json`（2026-09 读取）；Holtzman 等 2019 |
| GQA | Llama 2（34B、70B）与 Llama 3 全系列（8 个 KV 头）；Qwen2 / Qwen2.5 / Qwen3；Mistral 7B；Gemma 3（27B：32 Q / 16 KV）；gpt-oss（64 Q / 8 KV）；SmolLM3（16 Q / 4 KV） | 技术报告：Llama 2、Llama 3、Qwen2、Qwen3、Mistral 7B、Gemma 3；各模型 `config.json` 的 `num_key_value_heads` |
| MQA | Gemma 3 1B（`num_key_value_heads: 1`）；本章作为 GQA 的端点来讲 | `google/gemma-3-1b-pt` 的 `config.json`；Shazeer 2019 |
| PagedAttention / 连续批处理 | vLLM（行业标准推理引擎） | Kwon 等 2023；Orca（Yu 等 2022） |

说明：Llama 官方仓库需要申请权限。所以 Llama-3.1-8B、Llama-3.1-8B-Instruct 和 Llama-3.3-70B 的 `config.json` / `generation_config.json` 读自 unsloth 的镜像仓库（`unsloth/Meta-Llama-3.1-8B` 等，其中 `_name_or_path` 指向 meta-llama 官方仓库）。这些数值与 Llama 3 论文表 3 的"8 个 KV 头"一致。Llama 2 70B 的配置未能直接读取，这里采用 Llama 2 论文的说法（34B、70B 用 GQA），**config 数字待核实**。

---

## 引导问题

向 Claude Code 提出这些问题。一直问到你能用自己的话讲清楚答案：

1. 温度 T 很大时，分布趋向均匀；T → 0 时，分布趋向 argmax。用 softmax 的公式解释原因。如果 logits 里有两个完全相等的最大值，T → 0 时会怎样？
2. 为什么 KV cache 只缓存 K 和 V？为什么不缓存 Q，也不缓存注意力的输出或 FFN 的中间结果？如果模型用的是双向注意力（例如 BERT），KV cache 还成立吗？
3. 缓存里的 K 是做过 RoPE 旋转之后的。如果缓存旋转之前的 K，每步读出来再旋转，结果会一样吗？哪种更划算？
4. decode 阶段"算力用不满"具体是什么意思？请 Claude Code 帮你估算：主线模型 decode 一个 token，要读多少字节的权重和 KV cache？要做多少次乘加？两者的比值是多少？（提示：搜索"算术强度"（arithmetic intensity）和"roofline"。）
5. 本章小实验里，MHA 和 GQA 的损失差距和随机种子带来的差距是同一个量级。如果要认真比较，你会怎样设计实验？需要几个种子、多大的模型？怎样报告误差？
6. 在 GQA 的论文里，GQA 模型是从已有的 MHA checkpoint "上训练"（uptraining）得到的：把一组内几个头的 K、V 投影取平均。为什么取平均？还有别的初始化办法吗？

## 动手任务

每个任务都要运行代码，并查看结果。

**任务 1（基础）**：在 `02_sampling.py` 里找一个你自己的上下文。打印 top-p = 0.5、0.9、0.99 时分别留下多少个字符。再把温度改成 0.7：同样的 top-p，留下的个数怎样变化？解释为什么温度和 top-p 会互相影响。

**任务 2（核心）**：把 `01_tiny_model.py` 的 `KVCache` 改成**预分配**版本。构造时传入 `max_len`，一次性分配形状为 `(n_layers, B, n_kv_heads, max_len, head_dim)` 的张量。`append` 只写入 `[start:start+T]`。用 `03_kv_cache.py` 验证输出仍然逐字一致，再比较生成 512 个字的时间。把你的写法和 `zero/kv_cache.py` 对照。

**任务 3（挑战）**：给 `03_kv_cache.py` 加上 batch 支持：一次生成 8 条不同提示词的续写（提示词先补齐到同样长度，或者全部用同一个提示词、不同种子）。测出 batch = 1、4、8 时每秒生成的总 token 数，验证 4.3 节"批处理能提高 decode 吞吐"的说法。再用 `04_kv_memory.py` 的公式算出每种 batch 下的缓存大小。

---

## 想深入：CS336

本章对应斯坦福 CS336（Spring 2026）<https://cs336.stanford.edu/>：

- **第 10 讲：推理**。它从资源核算的角度讲推理的开销从哪里来（prefill 与 decode、KV cache 的显存与内存带宽），以及让推理更快、更省的各类办法（讲义与录像见课程页）。本章只讲了其中最基础的部分。第 20、21、25 章还会回到这一讲。
- **作业 1（Basics）的解码部分**：在自己训练的 Transformer 上实现带温度和 top-p 的文本生成。这和本章的 `02_sampling.py` 是同一件事。作业仓库：<https://github.com/stanford-cs336/assignment1-basics>

---

## 本章参考文献

- Holtzman, Buys, Du, Forbes, Choi. *The Curious Case of Neural Text Degeneration*（nucleus / top-p 采样），2019：<https://arxiv.org/abs/1904.09751>
- Fan, Lewis, Dauphin. *Hierarchical Neural Story Generation*（top-k 采样），2018：<https://arxiv.org/abs/1805.04833>
- Shazeer. *Fast Transformer Decoding: One Write-Head is All You Need*（MQA），2019：<https://arxiv.org/abs/1911.02150>
- Ainslie et al. *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints*，2023：<https://arxiv.org/abs/2305.13245>
- Kwon et al. *Efficient Memory Management for Large Language Model Serving with PagedAttention*（vLLM），2023：<https://arxiv.org/abs/2309.06180>
- Yu et al. *Orca: A Distributed Serving System for Transformer-Based Generative Models*（迭代级调度 / 连续批处理），OSDI 2022：<https://www.usenix.org/conference/osdi22/presentation/yu>
- Touvron et al. *Llama 2: Open Foundation and Fine-Tuned Chat Models*，2023：<https://arxiv.org/abs/2307.09288>
- Llama Team. *The Llama 3 Herd of Models*，2024：<https://arxiv.org/abs/2407.21783>
- Qwen Team. *Qwen2 Technical Report*，2024：<https://arxiv.org/abs/2407.10671>；*Qwen3 Technical Report*，2025：<https://arxiv.org/abs/2505.09388>
- Jiang et al. *Mistral 7B*，2023：<https://arxiv.org/abs/2310.06825>
- Gemma Team. *Gemma 3 Technical Report*，2025：<https://arxiv.org/abs/2503.19786>
- 模型配置（2026-09 读取）：[Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B/blob/main/config.json)、[Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B/blob/main/config.json)（[generation_config](https://huggingface.co/Qwen/Qwen3-8B/blob/main/generation_config.json)）、[Qwen2.5-7B](https://huggingface.co/Qwen/Qwen2.5-7B/blob/main/config.json)、[Qwen2.5-7B-Instruct generation_config](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/blob/main/generation_config.json)、[Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1/blob/main/config.json)、[gemma-3-1b-pt](https://huggingface.co/google/gemma-3-1b-pt/blob/main/config.json)、[gemma-3-27b-pt](https://huggingface.co/google/gemma-3-27b-pt/blob/main/config.json)、[gemma-3-27b-it generation_config](https://huggingface.co/google/gemma-3-27b-it/blob/main/generation_config.json)、[gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b/blob/main/config.json)、[SmolLM3-3B](https://huggingface.co/HuggingFaceTB/SmolLM3-3B/blob/main/config.json)（[generation_config](https://huggingface.co/HuggingFaceTB/SmolLM3-3B/blob/main/generation_config.json)）、Llama 镜像：[unsloth/Meta-Llama-3.1-8B](https://huggingface.co/unsloth/Meta-Llama-3.1-8B/blob/main/config.json)、[unsloth/Llama-3.1-8B-Instruct generation_config](https://huggingface.co/unsloth/Llama-3.1-8B-Instruct/blob/main/generation_config.json)、[unsloth/Llama-3.3-70B-Instruct](https://huggingface.co/unsloth/Llama-3.3-70B-Instruct/blob/main/config.json)
- vLLM：<https://github.com/vllm-project/vllm>
- [nanoGPT](https://github.com/karpathy/nanoGPT) 的 `generate`（温度 + top-k 的最简写法）、[minimind](https://github.com/jingyaogong/minimind)（带 KV cache 的小模型推理）

**下一章**：第二部分到这里结束。我们有了一个能训练、能快速生成的现代 Transformer。第三部分要训练一个真正的模型：0.6–0.8B 参数、上千亿 token、上万美元的算力。但在花第一块钱之前，要先回答几个问题：怎样判断它好不好、和谁比、怎样比才公平？第 11 章，我们先定考卷。
