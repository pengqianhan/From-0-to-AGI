# 第 15 章：中期训练与长上下文 —— 最后一段怎么训，读不长怎么办

> **一句话目标**：读完这一章，你能说清楚"退火"为什么要在 WSD 的衰减段换上高质量数据，并会用"分叉衰减"低成本地比较两种配比；能算出任意 RoPE 设置下每个维度对的波长，讲清楚模型为什么读不了比训练时更长的文本；能手写 YaRN 的频率插值并和官方实现对拍；知道主线模型怎样从 4K 扩到 32K、闸门 2 要检查什么。

📺 **本章视频**：待发布（本地渲染：`bash chapters/15-midtraining-long-context/video/build.sh`）
🧪 **本章自检**：学完后在 Claude Code 里输入 `/ch15-midtraining`

---

上一章我们把主线模型的预训练搭好了：几千张卡时、几千亿 token，学习率按 WSD 走 warmup 和稳定段（`configs/main/pretrain.toml` 里 `decay_frac = 0`，衰减段故意留着没走）。这一章要解决预训练收尾时的两个问题：

1. **最后一段怎么训？** 学习率降下来的那一小段，对最终模型的影响出奇地大。几乎所有头部开源模型都在这一段换上"最好的数据"——这叫**退火（annealing）**或**中期训练（mid-training）**。
2. **读不长怎么办？** 预训练用的是 4K 长度的片段（注意力的算力随长度平方增长，全程用 32K 太贵）。可主线模型要做工具调用，工具说明加上多轮对话动辄上万 token。一个只见过 4K 的模型，直接读 32K 会崩。办法是改 RoPE 的"转速"再接着训一小段：**调大 RoPE 基频**和 **YaRN**。

这两件事做完，主线 Base 模型就出炉了，接着是**闸门 2**：拿它的真实评测成绩和闸门 1 的预测对比，明显偏低就先诊断，不急着进后训练。

本章代码（都在 CPU 上跑）：

```bash
uv run python chapters/15-midtraining-long-context/code/01_rope_wavelengths.py   # 波长表、没见过的角度、32K 的算力账（1 秒）
uv run python chapters/15-midtraining-long-context/code/02_yarn_from_scratch.py  # 从零写 PI / YaRN，与 zero、HF 对拍（几秒）
uv run python chapters/15-midtraining-long-context/code/03_context_extension.py  # 长度 64 训练 → 读 128/256（单线程约 11 分钟 CPU 时间）
uv run python chapters/15-midtraining-long-context/code/04_anneal_mixture.py     # 分叉衰减：衰减 × 换数据（单线程约 8 分钟 CPU 时间）
```

## 1. 中期训练：学习率最低的那一段，喂最好的数据

### 1.1 直觉：最后学到的，记得最牢

第 6 章做过一个实验：学习率恒定的主干上随时分出一小段衰减，验证损失马上掉一截，追平了事先定好总长的余弦调度。MiniCPM（WSD 的提出者）在 0.036B 模型上看到同样的现象：**衰减段只占约 10% 的步数，损失却急剧下降**。

原因可以用第 1 章的画面理解：学习率大时，参数在谷底附近来回跳，"大方向"学得快，细节落不下来；学习率降下来，参数才慢慢落进谷底。换句话说，**衰减段决定了模型最后停在哪里**，而停在哪里，取决于这一段看的是什么数据。

于是一个很自然的想法：既然最后这一段这么"有分量"，那就把最好、最想让模型学会的数据留到这一段。这就是中期训练：

```
预训练（稳定段）：海量网页为主，学习率恒定          ≈ 90–95% 的算力
中期训练（衰减段）：换成高质量数据 + 数学/代码 + 指令式数据，学习率降到 0   ≈ 5–10%
```

"中期"这个名字是说它在预训练和后训练（SFT、RL）之间：目标仍然是通用的下一个 token 预测，但数据已经开始向"想要的能力"倾斜。

### 1.2 谁在这么做

| 模型 | 做法 | 来源 |
|---|---|---|
| **OLMo 2** | 明确叫 mid-training，占总算力 5–10%：学习率线性降到 0，数据换成 Dolmino Mix（高质量过滤的网页约占一半，外加 FLAN 指令数据、学术论文、维基、StackExchange 问答、合成数学）。7B 用 50B token 做三遍（不同数据顺序）再把权重平均 | [OLMo 2 §2.3、§4](https://arxiv.org/abs/2501.00656) |
| **Llama 3** | 预训练最后的退火：学习率线性降到 0，同时上采样"质量非常高"的数据；还用退火来**评估数据**：把训练到一半的 8B 模型在 40B token 上退火，新数据集占 30% | [Llama 3 §3.1.3、§3.4.3](https://arxiv.org/abs/2407.21783) |
| **SmolLM3** | WSD 的衰减段（10T → 11.1T token）进一步上采样数学和代码，并加入 OpenMathReasoning 等指令与推理数据；之后还有"长上下文 + 推理"的 mid-training | [SmolLM3 博客](https://github.com/huggingface/blog/blob/main/smollm3.md) |
| **MiniCPM** | 衰减段把高质量的 SFT 数据混进预训练数据；对照实验显示"衰减段就加入"比"只在 SFT 阶段加入"好 | [MiniCPM §5、§6.2](https://arxiv.org/abs/2404.06395) |
| **Qwen3** | 第二阶段（约 5T token）提高 STEM、代码、推理和合成数据的比例，并"加快学习率衰减" | [Qwen3 §3.2](https://arxiv.org/abs/2505.09388) |
| **MobileLLM-R1** | 两段各 100B token 的 mid-training，学习率线性降到 0，数据以 Dolmino 为基础再加数学和代码，并用 Llama-3.1-8B 做 logits 蒸馏 | [MobileLLM-R1 §3、附录 A](https://arxiv.org/abs/2509.24945) |
| **Puro-2B** | 第二阶段 960B token 线性衰减，同时把每个来源内部的数据按质量分从低到高排序，越好的越靠后；排序比随机顺序的 15 项平均高 1.18 分 | [Puro-2B §3.4](https://www.alphaxiv.org/abs/2608.27370) |

OLMo 2 的效果最直观（表 9）：7B 模型中期训练前后，10 项评测平均分从 53.0 升到 62.9，GSM8K 从 24.1 升到 67.5。而每一遍中期训练只有 50B token，约为 3.9T 预训练的 1.3%（做了三遍再平均）。

### 1.3 为什么有效：三个理由

1. **低学习率把东西"定"下来**。上面说过，衰减段决定模型停在哪里；这一段看到的数据分布，对最终模型的影响比同样多的 token 放在前面大得多。
2. **好数据少，只够用在刀刃上**。高质量数学、代码、指令数据往往只有几十亿到几百亿 token，摊在几万亿 token 的预训练里，要么占比微不足道，要么被重复很多遍。MiniCPM 正是这么论证的：集中放在衰减段，既不用反复重复小数据集，又赶上了"最有分量"的那段。OLMo 2 的"微退火"实验还发现：目标领域的数据只要**出现**就有用，数学占 10% 和 35% 的结果差不多（GSM8K 子集 61 vs 63.5，退火前是 28.5）。
3. **评估数据变便宜了**。想知道一个新数据集好不好，用不着从头训一个模型：从稳定段的某个 checkpoint 分出一小段衰减，比较"加它"和"不加它"就行。Llama 3 用这个办法给小数据集估值（做法与 Blakeney et al. 2024 类似），OLMo 2 叫它"微退火（microannealing）"。这正是第 6 章"分叉衰减"的用法，只是这回分叉比的是数据。

### 1.4 小实验：衰减 × 换数据

`04_anneal_mixture.py` 用第 9 章的极简 Transformer（字节级）在三种"来源"上训练：英文（莎士比亚）、中文（古诗词）、代码。主干配比是 英 0.45 / 中 0.45 / 代码 0.10——代码是"少而想补强"的那类数据，好比 OLMo 2 里的数学。主干以恒定学习率训 800 步，然后从同一个点分出 4 条支路，各训 200 步：

```python
decay = [PEAK_LR * (1 - (s + 1) / BRANCH_STEPS) for s in range(BRANCH_STEPS)]   # 线性降到 0
const = [PEAK_LR] * BRANCH_STEPS
branches = {
    "A 恒定 + 原配比": (const, MIX_PRETRAIN),
    "B 衰减 + 原配比": (decay, MIX_PRETRAIN),
    "C 恒定 + 新配比": (const, MIX_ANNEAL),     # 新配比：英 0.25 / 中 0.25 / 代码 0.50
    "D 衰减 + 新配比": (decay, MIX_ANNEAL),
}
```

输出（验证集 bits-per-byte，越低越好；单线程约 8 分钟 CPU 时间）：

| | 英文 | 中文 | 代码 | 平均 |
|---|---:|---:|---:|---:|
| 分叉点（主干 800 步） | 2.759 | 3.148 | 2.746 | 2.884 |
| A 恒定 + 原配比 | 2.700 | 3.097 | 2.632 | 2.810 |
| B 衰减 + 原配比 | 2.549 | 3.037 | 2.472 | 2.686 |
| C 恒定 + 新配比 | 2.803 | 3.168 | 2.223 | 2.731 |
| D 衰减 + 新配比 | 2.642 | 3.097 | **2.091** | **2.610** |

以"继续恒定学习率、不换数据"的 A 为基准，代码的 bits-per-byte 下降了：只衰减（B）0.160，只换数据（C）0.409，两者一起（D）0.541。几个值得注意的地方：

- **衰减本身对所有来源都有用**：B 比 A 在英文、中文、代码上都低 0.06–0.16，这是第 6 章看到过的现象；
- **换数据是"有取有舍"**：C 的代码大幅变好，但英文、中文比 A 还差（英文 2.803 vs 2.700），因为它们的份额从 0.45 降到了 0.25；
- **D 把两者叠起来**：代码最好、平均最好；英文、中文不如 B（2.642 vs 2.549、3.097 vs 3.037），但仍然好于或等于 A。真实的中期训练也要做这个权衡——所以 OLMo 2 的 Dolmino 里高质量网页仍占约一半，而不是全换成数学；
- 四条支路一共只多训了 4 × 200 = 800 步，却回答了"衰减值不值、换数据值不值"两个问题。这就是"分叉衰减"作为数据实验工具的价值。

这是单个种子、130 万参数级别的极小实验，只用来演示方法；差距的大小不能外推到主线模型。

### 1.5 主线模型的中期训练加什么

按 GOAL.md 3.3，主线模型的中期训练在高质量网页之外加三类数据（`configs/main/midtrain.toml`，比例待第二步的分叉衰减实验确定）：

- **数学与代码**：FineMath、Stack-Edu（OLMo 2、SmolLM3、Llama 3 都在这一段上采样它们）；
- **指令式数据**：像 OLMo 2 的 FLAN、MiniCPM 的 SFT 数据那样，让 Base 模型提前熟悉"问—答"的格式；
- **工具调用格式数据**：这是主线模型自己的需要（第 16、19 章的目标是工具调用）。让 `<tool_call>{...}</tool_call>` 这种格式在预训练末尾就出现，后训练时模型不用从零学格式。**这一点是本课的设计选择，不是已核实的行业共识**，第二步会用分叉衰减验证它到底有没有用。

两条纪律：中期训练数据同样要做 13-gram 去污染（GOAL.md 3.2）；OLMo 2 为了调中期训练只允许自己看 GSM8K 1319 题里的 200 题，我们更严格——所有决策只看自己的开发集，测试基准只在闸门上跑。

## 2. 长上下文：为什么模型读不长

### 2.1 RoPE 的钟表

第 9 章讲过 RoPE：把 q、k 的 `head_dim` 维两两配成 `d/2` 对，第 i 对在位置 m 旋转 `m·ω_i` 弧度：

```
ω_i = θ^(-2i/d)             转速（θ 是基频，默认 1 万）
λ_i = 2π / ω_i = 2π·θ^(2i/d)  波长：转满一圈需要多少个 token
```

像一块有 64 根指针的钟：第 0 根是秒针，每个 token 转 1 弧度；最后一根比时针还慢。`01_rope_wavelengths.py` 用主线模型的 `head_dim = 128` 算出每对的波长：

```python
def wavelengths(head_dim: int, theta: float) -> np.ndarray:
    return 2 * math.pi / inv_freq(head_dim, theta)    # λ_i = 2π/ω_i
```

输出（节选，单位：token）：

| 维度对 i | θ = 1 万 | θ = 50 万（Llama 3） | θ = 100 万（Qwen3 长上下文） |
|---:|---:|---:|---:|
| 0 | 6.3 | 6.3 | 6.3 |
| 16 | 62.8 | 167.1 | 198.7 |
| 32 | 628.3 | 4,443 | 6,283 |
| 48 | 6,283 | 118K | 199K |
| 63 | 54K | 2.6M | 5.1M |

| 长度 L 内转不满一圈（λ_i > L）的维度对个数（共 64 对） | θ = 1 万 | θ = 50 万 | θ = 100 万 |
|---|---:|---:|---:|
| L = 4096 | 18 | 32 | 33 |
| L = 32768 | 4 | 22 | 24 |

最快那一对的波长永远是 2π ≈ 6.3（每个 token 转 1 弧度），和 θ 无关；θ 只决定后面的指针有多慢。

### 2.2 读不长的两个原因

**原因一：没见过的角度。** 训练长度 4096、θ = 1 万时，有 18 对指针在 4096 个 token 内**转不满一圈**。对这些维度对，模型只见过圆周上的一段弧；读到 32K 时，它们会转到训练中从没出现过的角度——就像一个只见过 0 点到 3 点的人，突然被问"9 点是什么意思"。YaRN 论文的解释是：转不满一圈的维度里其实藏着**绝对位置**信息，一旦越界，模型就不认识了。

**原因二：注意力被摊薄。** softmax 要在所有位置之间分配权重。位置从 4K 变成 32K，候选多了 8 倍，注意力分布会变"平"（熵变大），原本该集中的注意力被摊到大量无关位置上。YaRN 的"温度"就是针对这一点的（3.3 节）。

小实验（`03_context_extension.py`）把问题缩小到 CPU 上：第 9 章的极简 Transformer（字节级，`head_dim = 32`，θ = 1 万），只用长度 **64** 的片段训练 1500 步，然后直接在长度 128、256 的验证片段上算 loss。因果注意力下，位置 p 的预测只看得见 `[0, p]`，所以按位置分段看 loss，就知道"超出训练长度的部分"表现如何：

预训练 3,072,000 个字节（token），然后**不做任何训练**，只换 RoPE 的 cos/sin（验证 loss，nat/字节，越低越好；最后三列按位置分段）：

| 设置 | L = 64 | L = 128 | L = 256 | 位置 0–64 | 位置 64–128 | 位置 128–256 |
|---|---:|---:|---:|---:|---:|---:|
| (a) 什么都不改 | 1.578 | 1.646 | 1.879 | 1.578 | 1.714 | **2.111** |

"什么都不改"时，训练长度以内（位置 0–64）的 loss 是 1.578；一越过 64，loss 就往上走，到位置 128–256 升到 2.111。按理说上下文越长、能参考的内容越多，loss 应该**更低**才对——这就是"读不长"。

（同一张表里 PI、YaRN、调大基频三行放在 3.4 节，讲完三种办法再看。）

## 3. 三种办法：调大基频、位置内插、YaRN

三种办法都**只改 RoPE 的转速**，不动任何可学参数，改完都要在长文本上再训一小段。

### 3.1 调大基频（ABF）

最直接的办法：把 θ 调大，所有指针一起变慢。Xiong et al.（2023，Llama 2 Long）把这个做法叫 **ABF（adjusted base frequency）**。θ 从 1 万调到 100 万，第 i 对慢了 `100^(2i/d)` 倍：

| 维度对 i | 0 | 16 | 32 | 48 | 63 |
|---|---:|---:|---:|---:|---:|
| 慢了多少倍 | 1.0 | 3.2 | 10.0 | 31.6 | 93.1 |

秒针不受影响（近处的位置照样分得清），慢指针被大幅放慢：读到 32K 时，**没有一对指针会转到"θ = 1 万、长度 4K 训练时没见过的角度"**（`01` 的第 3 部分：18 对 → 0 对）。代价是同一个角度对应的距离全变了，所以要在新基频下接着训练。

谁在用：Qwen3 在长上下文阶段"按 Qwen2.5 的做法用 ABF 把基频从 1 万调到 100 万"（`config.json`：`rope_theta: 1000000`）；SmolLM3 分两段扩展，4K→32K 时 θ 调到 150 万，32K→64K 时调到 500 万（`config.json`：`rope_theta: 5000000.0`）；Gemma 3 全局注意力层的基频从 1 万调到 100 万。Llama 3 和 OLMo 2 则从预训练一开始就用 θ = 50 万（Llama 3 报告引用 Xiong et al. 的结论：这个值对 32K 以内有效）。

### 3.2 位置内插（PI）：YaRN 的铺垫

Chen et al.（2023）的位置内插（Position Interpolation, PI）换了个思路：不让位置越界，而是把位置"压扁"——要扩 s 倍，就把位置 m 当成 m/s，等价于所有指针都慢 s 倍：

```python
def pi_inv_freq(head_dim: int, theta: float, s: float) -> torch.Tensor:
    return rope_inv_freq(head_dim, theta) / s      # 所有维度对一起放慢 s 倍
```

越界的问题解决了，但秒针也被放慢了：原来相邻两个 token 在秒针上差 1 弧度，÷8 之后只差 0.125 弧度——**近处的位置变得难分辨**。YaRN 论文的消融里，不微调直接 PI 扩 8 倍，困惑度超过 10（原模型约 4）。Gemma 3 扩到 128K 时用的就是 PI 式的缩放（`rope_scaling: {rope_type: "linear", factor: 8.0}`），配合了续训。本课把 PI 作为理解 YaRN 的铺垫（GOAL.md 2.1 规则 C）。

### 3.3 YaRN：快指针不动，慢指针内插，再调一下温度

YaRN（Peng et al., 2023）综合了上面两种直觉：**只压那些真的会越界的慢指针，不碰快指针**。按"训练长度 L 内转了多少圈" `r_i = L/λ_i` 把维度对分成三段：

- 转了 β_fast = 32 圈以上（高频）：**原样保留**——它们只编码相对距离，本来就不会越界；
- 不满 β_slow = 1 圈（低频）：**完全内插**，÷s，和 PI 一样；
- 中间：线性过渡。

另外把注意力 logits 乘一个温度因子：`√(1/t) = 0.1·ln(s) + 1`，补偿"候选变多、注意力变平"。实现上不用改注意力代码，直接把 cos/sin 乘上这个数（q、k 各乘一次，logits 就乘了它的平方）。`02_yarn_from_scratch.py` 的核心：

```python
low = max(math.floor(dim_with_turns(beta_fast)), 0)                # 比它靠前：转了 > β_fast 圈
high = min(math.ceil(dim_with_turns(beta_slow)), head_dim - 1)     # 比它靠后：转了 < β_slow 圈
ramp = ((i - low) / (high - low)).clamp(0, 1)   # 0 = 高频段，1 = 低频段
keep = 1 - ramp                                  # 保留原频率的权重
new_w = keep * w + (1 - keep) * w / s            # 高频不动、低频 ÷ s、中间混合
mscale = 0.1 * math.log(s) + 1.0                 # √(1/t) = 0.1·ln(s) + 1
```

`02` 打印了本章小实验设置（`head_dim = 32`、θ = 1 万、训练长度 64、s = 4）下每个维度对的处理方式（节选）：

| i | 波长 λ | 训练内圈数 | 保留权重 | ω（原） | ω（PI） | ω（YaRN） |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 6.3 | 10.19 | 1.00 | 1.00000 | 0.25000 | 1.00000 |
| 1 | 11.2 | 5.73 | 0.80 | 0.56234 | 0.14059 | 0.47799 |
| 2 | 19.9 | 3.22 | 0.60 | 0.31623 | 0.07906 | 0.22136 |
| 4 | 62.8 | 1.02 | 0.20 | 0.10000 | 0.02500 | 0.04000 |
| 5 | 111.7 | 0.57 | 0.00 | 0.05623 | 0.01406 | 0.01406 |
| 15 | 35333 | 0.00 | 0.00 | 0.00018 | 0.00004 | 0.00004 |

mscale = 0.1·ln 4 + 1 = 1.1386，注意力 logits 相当于乘 1.2965。注意这里的"圈数"边界要按维度编号取整（`floor` / `ceil`）：训练长度只有 64 时，第 0 对只转了 10 圈，没到 32 圈，但取整后仍被划进"保留"段——这是 YaRN 官方代码的写法，transformers 和 zero 都照此实现。论文公式（10）–（13）把斜坡写成对"圈数"线性，与官方代码在过渡段最多差 57%（`02` 第 4 部分）；以代码为准。

对主线模型的设置（`02` 第 2 部分）：

| 场景 | 原样保留 | 过渡 | 完全内插 | mscale |
|---|---:|---:|---:|---:|
| θ = 1 万，4K → 32K（只用 YaRN） | 21 对 | 25 对 | 18 对 | 1.208 |
| θ = 100 万，32K → 128K（Qwen3 模型卡的做法） | 24 对 | 16 对 | 24 对 | 1.139 |

对拍（`02` 第 3 部分）：从零写的版本与 `zero.model.compute_rope_inv_freq`、transformers 的 `Qwen3RotaryEmbedding` 在 4 组设置下频率最大差 ≤ 6.0×10⁻⁸（float32 舍入），mscale 完全相同。

### 3.4 小实验：微调一小段

零样本（不训练）只是开始。三种办法在真实模型里都要再训：Llama 3 分 6 段从 8K 扩到 128K，约 800B token；Qwen3 的长上下文阶段用了"几千亿" token；DeepSeek-V3 用 YaRN 分两段各训 1000 步（4K→32K→128K）；SmolLM3 两段各 50B token。小实验里每种设置都用长度 256 的片段微调 150 步（预训练 token 数的 1/10），数据顺序完全相同：

先看零样本（不训练，只换 RoPE）的完整表：

| 设置 | L = 64 | L = 128 | L = 256 | 位置 0–64 | 位置 64–128 | 位置 128–256 |
|---|---:|---:|---:|---:|---:|---:|
| (a) 什么都不改 | 1.578 | 1.646 | 1.879 | 1.578 | 1.714 | 2.111 |
| (b) 位置内插 PI（÷4） | 3.461 | 3.531 | 3.561 | 3.461 | 3.602 | 3.590 |
| (c) YaRN（s = 4） | 1.654 | 1.660 | 1.673 | 1.654 | 1.667 | **1.685** |
| (d) 调大基频（θ = 10 万） | 1.626 | 1.643 | 1.851 | 1.626 | 1.660 | 2.058 |

再用长度 256 的片段微调 150 步（307,200 个 token，预训练的 1/10）之后：

| 设置 | L = 64 | L = 128 | L = 256 | 位置 0–64 | 位置 64–128 | 位置 128–256 |
|---|---:|---:|---:|---:|---:|---:|
| (a) 什么都不改 | 1.587 | 1.570 | 1.561 | 1.587 | 1.553 | 1.553 |
| (b) 位置内插 PI | 1.701 | 1.688 | 1.679 | 1.701 | 1.676 | 1.669 |
| (c) YaRN | **1.578** | **1.564** | **1.555** | 1.578 | 1.550 | 1.547 |
| (d) 调大基频 | 1.591 | 1.574 | 1.568 | 1.591 | 1.557 | 1.562 |

怎么读这两张表：

- **零样本，YaRN 最稳**：位置 128–256 的 loss 只有 1.685，几乎和训练长度内一样；代价是长度 64 以内略差（1.654 vs 1.578），因为慢指针被压慢、温度也变了。这正是 Qwen3 模型卡提醒"静态 YaRN 可能影响短文本、只在需要时打开"的原因；
- **零样本，PI 最糟**：连长度 64 以内都从 1.578 涨到 3.461——四倍压缩把秒针也压慢了，近处的位置分不清，模型连短文本都读不好了；
- **零样本，只调大基频几乎没用**（2.058 vs 2.111）：所有角度与距离的对应都变了，模型必须重新适应。它本来就是配合续训的办法；
- **微调之后差距大幅缩小**：YaRN 在三个长度上都最低，"不改"、"调大基频"紧随其后（差 0.006–0.013），PI 仍落后约 0.12；
- **单个种子**：YaRN、不改、调大基频之间零点零几的差距很可能在种子波动以内（动手任务 2 会让你换种子验证）。能放心下结论的只有两点：零样本时 YaRN 远好于其他三种；PI 在同样的微调预算下明显落后。

这个极小实验的规模（长度 64 → 256、1.5 万步都不到）和真实模型（4K → 32K、几十亿 token）差得很远，只用来看清机制，不代表主线模型上哪种办法更好。主线模型的选择（调大基频 + 32K 续训，推理时可再叠加 YaRN）依据的是 3.1–3.3 节列出的头部模型的做法。

## 4. 长上下文的数据与代价

**代价**。注意力里 `QKᵀ` 和 `AV` 的运算量正比于序列长度。`01` 的第 5 部分按 zero 的口径（PaLM 附录 B，不为因果掩码减半）算主线模型每个训练 token 的运算量：

| 序列长 | 每 token 运算量 | 其中注意力（`QKᵀ`、`AV`）占比 |
|---:|---:|---:|
| 4096 | 6.96 GFLOP | 41% |
| 32768 | 26.69 GFLOP | 84% |

同样多的 token，用 32K 序列训练要贵 3.84 倍（FlashAttention 实际会跳过因果掩码的上三角，约省一半注意力运算，但比例关系不变）。

所以没有人全程用长序列训练。Llama 3 的原话是"我们不更早地用长序列训练，因为自注意力的算力随序列长度平方增长"。长上下文都放在预训练的最后，而且只占很小一段。

**数据**。长上下文阶段的数据要真的"长"：

- **天然的长文档**：书、代码仓库、长网页。Qwen3 的长上下文语料里 75% 在 16K–32K token 之间，另外 25% 在 4K–16K——**保留一部分较短的数据**，防止短文本能力退化；
- **合成的长文本任务**：Llama 3 在 SFT 阶段用模型生成长文档问答、层层摘要、"删掉仓库里一个被很多文件引用的源文件，让模型补回来"，发现只要混 0.1% 这类数据就能兼顾长短；
- 不是越多越好：SmolLM3 的消融发现，在自然长度分布之外额外上采样书和代码仓库，并没有进一步提高 RULER、HELMET 的成绩，"用衰减段的配比 + 更长的序列 + 更大的基频"就够了。

**别丢了短文本能力**。Llama 3 判断每一段扩展是否成功的标准有两条：短上下文评测**完全恢复**，大海捞针在该长度下**全部答对**。

## 5. 怎么评长上下文

- **大海捞针（needle-in-a-haystack, NIAH）**（Kamradt, 2023）：在一大段无关文字的某个深度插入一句"针"（比如"某某的密码是 7 位数字"），最后问模型。长度 × 深度扫一遍，画成一张表。它是**冒烟测试**：通不过一定有问题；通过了不代表真的会用长上下文。Llama 3 报告 NIAH 全部答对，DeepSeek-V3 在 128K 以内全部通过。
- **RULER**（Hsieh et al., 2024，NVIDIA）：把 NIAH 扩展成 4 类 13 个合成任务——检索（多种针、多个干扰针）、多跳追踪（变量赋值链）、聚合（找出最常见的词）、长文档问答。测了 17 个模型，**几乎所有模型在普通 NIAH 上接近满分，但在 RULER 上随长度明显下降**；声称支持 32K 以上的模型里只有一半在 32K 时还达标（以 Llama2-7B 在 4K 的 85.6 分为及格线）。Qwen3、Gemma 3、SmolLM3 的报告都用 RULER 报长上下文成绩。

本章给主线模型加了一个大海捞针工具 `zero/tools/needle.py`（见"从极简到生产级"）。除了"生成的文字里有没有正确数字"，它还算一个更细的**似然增益**：把针里的数字换成另一个随机数作对照，看正确答案的负对数似然降了多少——> 0 才说明模型真的在用针里的信息。

## 6. 小结

- **中期训练 / 退火**：WSD 的衰减段只占 5–10% 的算力，却决定模型最后停在哪里；把高质量网页、数学代码、指令式数据集中放在这一段。用"分叉衰减"就能低成本比较配比。
- **为什么读不长**：训练长度内转不满一圈的慢指针，在更长的位置会转到没见过的角度；候选变多，注意力也被摊薄。
- **调大基频**：所有指针一起变慢，快指针几乎不变；要在新基频下续训。
- **YaRN**：快指针不动、慢指针 ÷s、中间过渡，再把 logits 乘 `(0.1·ln s + 1)²`；小实验里零样本就几乎不掉点，续训一小段后仍然最好（差距很小）。
- **代价与评测**：32K 序列每 token 的算力约是 4K 的 3.8 倍，所以只在最后训一小段；大海捞针是冒烟测试，RULER 才是基准。

---

## 从极简到生产级

| 极简版（`code/`） | 生产级（`zero/`、`configs/`、`tests/`） | 多做了什么、为什么 |
|---|---|---|
| `02` 的 `yarn_inv_freq` | `zero/model.py`：`compute_rope_inv_freq(head_dim, theta, scaling)` 返回 `(inv_freq, attention_scaling)`；`RotaryEmbedding` 预计算 `[0, max_seq_len)` 的 cos/sin，并乘上 `attention_scaling` | cos/sin 是**非持久 buffer**，不进 `state_dict`：改 `rope_theta` 或 `rope_scaling` 后重建即可，其余权重原样加载；位置超过 `max_seq_len` 直接报错，不会悄悄外推 |
| `03` 里手动替换 cos/sin | `zero/config.py`：`ModelConfig.rope_theta`、`ModelConfig.rope_scaling`（校验：只支持 `type = "yarn"`，`factor ≥ 1`，必须给 `original_max_position_embeddings`，不认识的字段直接报错）；`zero/hf.py` 导出时把它写成 HF 的 `rope_scaling` / `rope_parameters` | 配置驱动，导出后 transformers、vLLM、llama.cpp 都能按同样的 YaRN 参数推理 |
| `04` 的"从同一个点分叉 + 换配比 + 线性衰减" | `zero/train/midtrain.py`：`init_from` 指向预训练 checkpoint；`check_compatible` 只允许改 `rope_theta`、`rope_scaling`、`max_seq_len`（改层数、维度会直接报错），并打印"变了什么"；`[schedule] kind = "wsd"`、`decay_frac = 1.0` 就是一整段衰减；`[[data.sources]]` 换配比（`zero/data/mixture.py` 按权重采样） | 与预训练共用 `zero/train/trainer.py`（BF16、梯度累积、断点续训、多卡）；中断后重跑同一条命令，从本次运行自己的 checkpoint 续训 |
| `03` 的"长度 256 微调" | `configs/main/longctx.toml`（见下表） | 32K 序列、FSDP、每步 100 万 token |
| 无 | `zero/tools/needle.py`：`make_case`（精确控制长度和针的位置，附对照提示词）、`score_text`、`answer_nll`、`run_grid`、`format_grid`；命令行 `uv run python -m zero.tools.needle --model <ckpt> --lengths ... --depths ...` | 可以评任何 zero checkpoint 或导出的 HF 目录；也可以传自己的 `generate_fn` 评别的模型 |

**对拍**：

- `02` 的第 3 部分：从零写的 YaRN 与 `zero.model.compute_rope_inv_freq`、transformers 的 `Qwen3RotaryEmbedding` 在 4 组设置下（包括 Qwen3 推荐的"32K×4"）频率最大差 ≤ 6×10⁻⁸（float32 舍入），mscale 完全相同；
- [`tests/test_model_hf_parity.py`](../../tests/test_model_hf_parity.py) 的 `yarn`、`yarn_custom_beta` 两个用例：随机初始化的 HF `Qwen3ForCausalLM` 带 YaRN，权重搬进 zero，150 个位置（超过原长度 32 / 48）的 logits 误差在 1e-5 以内；
- [`tests/test_kv_cache.py`](../../tests/test_kv_cache.py) 的 `test_greedy_batch_and_yarn`：开着 YaRN 时，用 KV cache 和不用 KV cache 的贪心生成完全一致；
- [`tests/test_needle.py`](../../tests/test_needle.py)（本章新增，9 项）：题目长度恰好等于请求的长度、针的位置随深度单调、对照提示词只在针的数字处不同；"读得出针"的假模型得满分、不说话的假模型得 0 分；`answer_nll` 与逐 token 手算一致；真实的 zero 模型能跑通整张表。

```bash
uv run pytest tests/test_model_hf_parity.py tests/test_kv_cache.py tests/test_needle.py -q
```

本机结果：`24 passed in 16.73s`（其中本章新增的 `test_needle.py` 9 项）。

**主线配置逐项说明**：

`configs/main/midtrain.toml`（继承 `pretrain.toml`）：

| 设置 | 值 | 为什么 |
|---|---|---|
| `train.init_from` | `out/main/pretrain/ckpt` | 从预训练稳定段的最后一个 checkpoint 接着训 |
| `train.max_steps` | 50000（≈ 26B token） | 约为预训练 500B 的 5%，落在 OLMo 2 的 5–10% 区间下沿；待定 |
| `[schedule]` | `wsd`，`warmup_steps = 0`，`decay_frac = 1.0`，`min_lr_ratio = 0` | 整段就是 WSD 的衰减段：从峰值线性降到 0（OLMo 2、Llama 3、MobileLLM-R1 都是线性降到 0） |
| `[[data.sources]]` | fineweb-edu 0.35、fineweb-2-zh 0.25、stack-edu 0.15、finemath 0.15、instruct-toolcall 0.10 | 数学和代码比预训练的 0.12 / 0.08 翻倍左右，另加 10% 指令与工具调用格式数据；具体比例待第二步用分叉衰减确定 |
| 其余（模型形状、优化器、seq_len 4096） | 继承 | 中期训练不改模型形状，`check_compatible` 会拦住误改 |

`configs/main/longctx.toml`（继承 `pretrain.toml`，从 `midtrain` 的结果接着训）：

| 设置 | 值 | 为什么 |
|---|---|---|
| `model.rope_theta` | 1,000,000 | ABF：1 万 → 100 万，与 Qwen3 的长上下文阶段相同 |
| `model.max_seq_len`、`data.seq_len` | 32768 | 目标长度直接训练（不靠推理时外推） |
| `micro_batch_size × grad_accum × 8 卡 × 32768` | 1 × 4 × 8 × 32768 = 1,048,576 token/步 | 32K 序列的激活很大，每卡一次只放一条 |
| `train.parallel` | `fsdp` | 参数、梯度、优化器状态切到 8 张卡上，给激活腾显存（**尚未在 GPU 上验证**） |
| `train.max_steps` | 4000（≈ 4.2B token） | 待定；参照 DeepSeek-V3 每段 1000 步、SmolLM3 每段 50B token，4.2B 偏保守 |
| `optim.lr`、`[schedule]` | 1e-4，cosine，200 步 warmup，降到 10% | 待定，见下方"待决定的问题" |
| 推理时更长 | 可再加 `model.rope_scaling = {type = "yarn", factor = 4, original_max_position_embeddings = 32768}` | Qwen3 模型卡推荐的做法：32K 原生，YaRN ×4 到 128K；只在需要时开（静态 YaRN 会略微影响短文本） |

**待决定的问题（第二步用小规模实验定）**：`midtrain` 已经把学习率降到 0，`longctx` 又从 1e-4 重新 warmup，相当于两段退火。头部模型的顺序并不统一：Qwen3 把长上下文放在预训练最后一段；Llama 3 先扩长度、最后 40M token 在 128K 长度上退火；DeepSeek-V3 扩长度时直接沿用预训练末尾的学习率 7.3×10⁻⁶；SmolLM3 则"用衰减段的配比 + 更长的序列"。另外，zero 目前没有做跨文档注意力屏蔽（见"前沿观察"），32K 窗口里会拼进很多短文档。

## 主线进度

### 极小配置演示（CPU，`configs/tiny`，约 1.3M 参数）

> 以下是**极小配置演示**：只说明代码能跑通、各项改动按预期生效，不代表主线模型的任何结果。

先跑 tiny 预训练（为了不和其他章节共用输出目录，这里用 `--set` 换了 `out_dir`；直接用默认命令也一样）：

```bash
uv run python -m zero.train.pretrain --config configs/tiny/pretrain.toml --set train.out_dir=out/tiny/ch15/pretrain
uv run python -m zero.train.midtrain --config configs/tiny/midtrain.toml \
    --set train.out_dir=out/tiny/ch15/midtrain --set train.init_from=out/tiny/ch15/pretrain/ckpt
```

```
# 预训练（节选）
模型参数 1.31M（非 embedding 0.79M），每步 2048 token，共 200 步，设备 cpu，world_size=1
step    100/200 | loss 6.1065 | lr 3.00e-03 | gnorm 0.38 | 2,709 tok/s | val 6.0048
step    200/200 | loss 5.4736 | lr 3.00e-04 | gnorm 0.38 | 2,608 tok/s | val 5.7116

# 中期训练
[midtrain] model.rope_scaling: None → {'type': 'yarn', 'factor': 2.0, 'original_max_position_embeddings': 128, 'beta_fast': 32.0, 'beta_slow': 1.0}
[midtrain] data.seq_len: 128 → 256
[midtrain] 数据混合: {'shakespeare': 0.45, 'chinese_poetry': 0.45, 'code': 0.1} → {'shakespeare': 0.3, 'chinese_poetry': 0.6, 'code': 0.1}
从 out/tiny/ch15/pretrain/ckpt 加载模型权重（step 200）
step      1/60 | loss 5.5548 | lr 9.83e-04 | gnorm 0.38 | 1,793 tok/s
step     30/60 | loss 5.4730 | lr 5.00e-04 | gnorm 0.42 | 2,434 tok/s | val 5.6488
step     60/60 | loss 5.4976 | lr 0.00e+00 | gnorm 0.38 | 2,229 tok/s | val 5.6101
```

读这份日志要注意三点：

- `check_compatible` 打印出了这次改了什么：YaRN ×2（原长 128）、序列长度 128 → 256、配比向中文倾斜；
- 学习率从 1e-3 线性降到 0（`decay_frac = 1.0`），正是"整段衰减"；
- 预训练的 val 5.7116 是在长度 128 的片段上算的，中期训练的 val 5.6101 是在长度 256 上算的，**两者不能直接比**。能说明的只是：换了 RoPE、加长了序列之后，训练照常收敛，没有崩。

大海捞针（`uv run python -m zero.tools.needle --model out/tiny/ch15/midtrain/ckpt --lengths 64,128,240 --depths 0,0.5,1 --n 5`）：

```
生成式准确率（贪心解码里出现正确数字的比例）
   长度\深度    0.00    0.50    1.00
          64    0.00    0.00    0.00
         128    0.00    0.00    0.00
         240    0.00    0.00    0.00

似然增益 nll_gain = NLL(对照) − NLL(真针)，> 0 表示模型在用针的信息
   长度\深度    0.00    0.50    1.00
          64  +0.010  -0.012  +0.000
         128  +0.004  +0.006  -0.064
         240  -0.019  -0.038  -0.027
```

一个只训了 260 步、130 万参数的模型当然捞不到针：准确率全是 0，似然增益在 0 附近正负摆动（每格只有 5 题，这些差别都在噪声里）。工具本身的正确性由 `tests/test_needle.py` 保证。

### 待 GPU 训练后补充

- 中期训练与长上下文扩展的真实训练曲线、每段的数据配比与花费、失败与返工；
- 分叉衰减实验：比较中期训练的几种配比（尤其是"工具调用格式数据有没有用"）；
- 32K 下的大海捞针表、RULER（4K–32K）成绩、短上下文评测是否恢复；
- **闸门 2** 的对比报告（下方清单）。

**成本估算**（`zero.tools.estimate_cost`，H100 SXM，MFU 0.4、$2.5/卡时的假设，**尚未在 GPU 上验证**，32K 序列下的实际 MFU 可能更低）：

| 阶段 | token | 每 token 运算量 | 卡时 | 费用 |
|---|---:|---:|---:|---:|
| 中期训练（`midtrain.toml`，4K） | 26.2B | 6.96 GFLOP | 127.9 | $320 |
| 长上下文（`longctx.toml`，32K） | 4.19B | 26.69 GFLOP | 78.5 | $196 |
| 合计 | | | 206.4 | $516（GOAL.md 3.4 预算 $700） |

两项都超过 $100，按 GOAL.md 3.4 须先把估算交给作者批准再开跑。

### 闸门 2 检查清单（GOAL.md 3.4：预训练结束）

- [ ] 用预注册（`eval/PREREGISTRATION.md`）里的 Base 模型少样本基准和我们自己的开发集，评测长上下文扩展后的 Base 模型，报 bootstrap 95% 置信区间；
- [ ] 与闸门 1 的外推预测（loss 与基准分数）逐项对比；**明显偏低时先诊断**（数据、学习率、代码 bug、评测模板），不进后训练；
- [ ] 短上下文能力：长上下文扩展前后的开发集 loss 与少样本成绩对比，确认"完全恢复"（Llama 3 的标准）；
- [ ] 长上下文：`zero.tools.needle` 在 4K / 8K / 16K / 32K × 5 个深度上全部答对；RULER 4K–32K 如实报告；
- [ ] 工具调用格式：在留出的工具调用格式数据上看 loss，确认中期训练里的格式数据被学到；
- [ ] 中期训练与长上下文数据的 13-gram 去污染检查结果存档；
- [ ] 花费记入 `runs/ledger.md`；
- [ ] 不用预注册的测试基准挑 checkpoint 或调配比（GOAL.md 11 节）。

---

## 前沿观察

> **NTK-aware 插值**：YaRN 的前身之一，按 `θ' = θ·s^(d/(d−2))` 换一个更大的基频，把"内插的压力"摊到各个维度（Code Llama 手动把基频调到 100 万也被 YaRN 论文归入这一类）；不同倍数下最优的基频要靠试，细节不进正文。
>
> **跨文档注意力屏蔽（intra-document masking）**：把多篇短文档拼进一个长窗口时，让每篇只看自己。Llama 3 说它"在标准预训练里影响有限，但在超长序列的续训里很重要"，SmolLM3 也采用了；DeepSeek-V3 明确写了没有用。目前核实到的采用方只有两家，暂不进正文；zero 的预训练也还没有实现它，第二步做长上下文前值得先用小实验验证。

---

## 采用方与来源

| 技术 | 采用方（主力版本） | 来源 |
|---|---|---|
| 中期训练 / 退火（衰减段换高质量数据） | **OLMo 2**（mid-training，Dolmino Mix，LR 线性降到 0）；**Llama 3**（退火上采样高质量数据，并用退火评估数据）；**SmolLM3**（衰减段上采样数学、代码并加入指令与推理数据）；**MiniCPM**（衰减段混入 SFT 数据）；**Qwen3**（S2 阶段提高 STEM/代码/推理/合成数据比例并加快 LR 衰减）；**MobileLLM-R1**（两段 mid-training，LR 线性降到 0）；Puro-2B（第二阶段按质量排序的数据课程 + 线性衰减） | [OLMo 2](https://arxiv.org/abs/2501.00656) §2.3、§4；[Llama 3](https://arxiv.org/abs/2407.21783) §3.1.3、§3.4.3；[SmolLM3 博客](https://github.com/huggingface/blog/blob/main/smollm3.md)；[MiniCPM](https://arxiv.org/abs/2404.06395) §5；[Qwen3](https://arxiv.org/abs/2505.09388) §3.2；[MobileLLM-R1](https://arxiv.org/abs/2509.24945) §3、附录 A；[Puro-2B](https://www.alphaxiv.org/abs/2608.27370) §3.4 |
| 调大 RoPE 基频（ABF） | **Qwen3**（长上下文阶段 1 万 → 100 万；`rope_theta: 1000000`）；**SmolLM3**（150 万 → 500 万；`rope_theta: 5000000.0`）；**Gemma 3**（全局层 1 万 → 100 万）；**Llama 3**（从预训练起 θ = 50 万；`rope_theta: 500000.0`）；**OLMo 2**（`rope_theta: 500000`）；gpt-oss（`rope_theta: 150000`） | [Qwen3](https://arxiv.org/abs/2505.09388) §3.2；[SmolLM3 博客](https://github.com/huggingface/blog/blob/main/smollm3.md)；[Gemma 3](https://arxiv.org/abs/2503.19786) §2、§5.3；[Llama 3](https://arxiv.org/abs/2407.21783) §3.2；Xiong et al. 2023 [arXiv:2309.16039](https://arxiv.org/abs/2309.16039)；各模型 `config.json`（下方链接） |
| YaRN | **DeepSeek-V3**（`rope_scaling: {type: yarn, factor: 40, original_max_position_embeddings: 4096, beta_fast: 32, beta_slow: 1}`，两段各 1000 步 4K→32K→128K）；**gpt-oss**（`factor: 32`，`original_max_position_embeddings: 4096`）；**Kimi K2**（`factor: 32`，原长 4096）；**Qwen3**（原生 32K，模型卡推荐 YaRN `factor: 4` 到 128K，RULER 成绩即按此测）；**SmolLM3**（64K 训练，YaRN 外推到 128K） | [YaRN](https://arxiv.org/abs/2309.00071)；[DeepSeek-V3](https://arxiv.org/abs/2412.19437) §4.3；[Qwen3-8B 模型卡](https://huggingface.co/Qwen/Qwen3-8B)；[SmolLM3 博客](https://github.com/huggingface/blog/blob/main/smollm3.md)；各模型 `config.json` |
| 位置内插 PI（铺垫） | Gemma 3（`rope_scaling: {rope_type: linear, factor: 8.0}`，报告写明沿用 Chen et al. 的做法） | [PI](https://arxiv.org/abs/2306.15595)；[Gemma 3](https://arxiv.org/abs/2503.19786) §5.3 |
| 长上下文评测 | NIAH：Llama 3、DeepSeek-V3 的报告；RULER：Qwen3（附录 A.1.1）、Gemma 3（表 15）、SmolLM3（博客） | [Kamradt NIAH](https://github.com/gkamradt/LLMTest_NeedleInAHaystack)；[RULER](https://arxiv.org/abs/2404.06654) |

**共识判断（GOAL.md 2.1）**：中期训练 / 退火、调大基频、YaRN 各有 3 个以上彼此独立的头部家族在主力版本中明确采用，进正文。PI 只作为 YaRN 的铺垫（规则 C）。NTK-aware 插值与跨文档屏蔽放在"前沿观察"。

**模型配置**（2026-09 通过 Hugging Face 读取）：
[Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B/blob/main/config.json)、
[Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B/blob/main/config.json)、
[SmolLM3-3B](https://huggingface.co/HuggingFaceTB/SmolLM3-3B/blob/main/config.json)、
[Llama-3.1-8B（unsloth 镜像）](https://huggingface.co/unsloth/Meta-Llama-3.1-8B/blob/main/config.json)、
[OLMo-2-1124-7B](https://huggingface.co/allenai/OLMo-2-1124-7B/blob/main/config.json)、
[DeepSeek-V3](https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/config.json)、
[gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b/blob/main/config.json)、
[Kimi-K2-Instruct](https://huggingface.co/moonshotai/Kimi-K2-Instruct/blob/main/config.json)、
[gemma-3-4b-pt](https://huggingface.co/google/gemma-3-4b-pt/blob/main/config.json)。

说明：Llama 3.1 的 `rope_scaling` 是 Meta 自己的 `llama3` 类型（`factor: 8`、`low_freq_factor: 1`、`high_freq_factor: 4`，原长 8192），思路与 YaRN 的"按频率分段"相同但公式不同，没有计入 YaRN 的采用方。Kimi K2 的 `config.json` 里 `beta_fast` 与 `beta_slow` 都是 1.0，与 YaRN 论文推荐的 32 / 1 不同，原因待核实。DeepSeek-V3 的 YaRN 只作用于 MLA 里解耦出来的 RoPE key（第 21 章）。Qwen3 的 `config.json` 里 `rope_scaling` 为 `null`，YaRN 是模型卡推荐用户按需打开的。

---

## 引导问题

带着这些问题去问 Claude Code，直到你能用自己的话讲清楚：

1. 中期训练的 token 往往只有预训练的百分之几，为什么效果能这么大？如果把同样的高质量数据均匀撒在整个预训练里，你预期结果会怎样？MiniCPM 的两个论点你同意吗？
2. 本章 `04` 实验里，"只换数据不衰减"（C）和"只衰减不换数据"（B）各带来多少提升？英文、中文为什么在 D 里没有变差（或者变差了多少）？"想补强的能力"与"已有的能力"之间怎么权衡？
3. 把 θ 调大以后，读到 32K 时"没见过的角度"就没有了，为什么还要续训？（提示：同一个角度对应的距离变了。）如果 θ 调到无穷大，会发生什么？
4. PI 把所有指针都放慢 s 倍，YaRN 只放慢慢指针。在 `02` 的表里找出本章小实验中"完全内插"的维度对，解释为什么秒针（i = 0）不能动。
5. YaRN 的温度 `0.1·ln(s) + 1` 是拟合出来的经验公式。试着让 Claude Code 帮你推一推：候选位置从 L 变成 sL 时，均匀注意力的熵增加多少？温度应该往哪个方向调？
6. 大海捞针全部答对的模型，RULER 上可能表现很差。设计一个你认为比 NIAH 更能说明"真的会用长上下文"的任务，它要怎样自动判分？

## 动手任务

每个任务都要真的运行代码、看到结果。

**任务 1（基础）**：修改 `01_rope_wavelengths.py`，把 `BASES` 换成你感兴趣的模型：gpt-oss（θ = 15 万，head_dim 64）、SmolLM3（θ = 500 万，head_dim 128）。它们在各自的原生训练长度内，有多少对指针转不满一圈？

**任务 2（核心）**：在 `03_context_extension.py` 里加一个设置 (e)：YaRN 但**不乘温度**（mscale 固定为 1），零样本和微调后各测一次，和 (c) 比较。温度在零样本时帮了多少？微调之后还重要吗？再换一个随机种子（改 `train(..., seed=...)`），看差距是否超出种子之间的波动。

**任务 3（挑战）**：用 `04_anneal_mixture.py` 的办法做一次"微退火"式的数据估值：把"代码"来源换成 `assets/tiny_corpus/` 以外你自己找的一小份文本（比如一本公版书），只在衰减段加入、占 30%（Llama 3 的做法），看它对自己那一类验证集的 bits-per-byte 有多大帮助、对其他来源有没有伤害。然后用 `uv run python -m zero.tools.needle --model out/tiny/midtrain/ckpt --lengths 64,128,240` 跑一次大海捞针，解释为什么 tiny 模型全是 0。

---

## 想深入：CS336

本章对应斯坦福 CS336（Spring 2026）<https://cs336.stanford.edu/>，**部分覆盖**：

- **第 14 讲：数据（过滤、去重、配比、合成数据）**。数据配比与合成数据的部分，是本章"中期训练换什么数据"的理论背景；作业 4（Data）里的过滤与配比实验可以直接接到本章的分叉衰减上。
- **第 15 讲：中期训练与后训练（SFT/RLHF）**。讲中期训练在整条流水线里的位置，以及它和 SFT 的分界。
- RoPE 的长上下文扩展（ABF、YaRN）与长上下文评测，CS336 未深入，见本章参考文献。

---

## 本章参考文献

- OLMo Team. *2 OLMo 2 Furious*（mid-training、Dolmino Mix、微退火、checkpoint soup），2024：<https://arxiv.org/abs/2501.00656>
- Llama Team. *The Llama 3 Herd of Models*（§3.1.3 退火数据、§3.4.2 长上下文预训练、§3.4.3 退火、§4.3.4 长上下文 SFT），2024：<https://arxiv.org/abs/2407.21783>
- Hugging Face. *SmolLM3: smol, multilingual, long-context reasoner*（博客）：<https://github.com/huggingface/blog/blob/main/smollm3.md>
- Hu et al. *MiniCPM: Unveiling the Potential of Small Language Models with Scalable Training Strategies*（WSD 与"衰减段加入高质量数据"），2024：<https://arxiv.org/abs/2404.06395>
- Qwen Team. *Qwen3 Technical Report*（§3.2 三阶段预训练、附录 A.1.1 RULER），2025：<https://arxiv.org/abs/2505.09388>
- Zhao et al. *MobileLLM-R1: Exploring the Limits of Sub-Billion Language Model Reasoners with Open Training Recipes*，2025：<https://arxiv.org/abs/2509.24945>
- Luo et al. *PuRo-2B: Poor Lab's Qwen2-1.5B Trained on RTX 5090 within $5090*，2026：<https://www.alphaxiv.org/abs/2608.27370>（`references.md` 已收录）
- Blakeney et al. *Does your data spark joy? Performance gains from domain upsampling at the end of training*，2024：<https://arxiv.org/abs/2406.03476>
- Peng et al. *YaRN: Efficient Context Window Extension of Large Language Models*，2023：<https://arxiv.org/abs/2309.00071>
- Chen et al. *Extending Context Window of Large Language Models via Positional Interpolation*，2023：<https://arxiv.org/abs/2306.15595>
- Xiong et al. *Effective Long-Context Scaling of Foundation Models*（ABF），2023：<https://arxiv.org/abs/2309.16039>
- Gemma Team. *Gemma 3 Technical Report*（§5.3 长上下文），2025：<https://arxiv.org/abs/2503.19786>
- DeepSeek-AI. *DeepSeek-V3 Technical Report*（§4.3 长上下文扩展），2024：<https://arxiv.org/abs/2412.19437>
- Hsieh et al. *RULER: What's the Real Context Size of Your Long-Context Language Models?*，2024：<https://arxiv.org/abs/2404.06654>
- Kamradt. *Needle In A Haystack — Pressure Testing LLMs*，2023：<https://github.com/gkamradt/LLMTest_NeedleInAHaystack>
- Hägele et al. *Scaling Laws and Compute-Optimal Training Beyond Fixed Training Durations*（WSD 与分叉衰减，第 6 章已引），2024：<https://arxiv.org/abs/2405.18392>
- [CS336](https://cs336.stanford.edu/) 第 14、15 讲

**下一章**：Base 模型会续写，但还不会"对话"，更不会按格式调用工具。第 16 章讲 SFT：chat template（包括工具调用格式）怎么设计、为什么只在回复部分算 loss，以及怎样把主线 Base 模型第一次变成一个能听指令的助手。
