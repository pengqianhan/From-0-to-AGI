# 第 13 章：数据 —— 从网页到一份能训练的数据集

[English](README.md) · **中文**

> **目标**：读完这一章，你能说出预训练数据从网页到分片要经过哪几步，以及每一步删掉什么。你能手写 MinHash LSH 去重，并算出"相似度为 s 的一对文档被抓到的概率"。你能用小模型的对照实验（配对比较 bits-per-byte）判断一种过滤或一个配比是否有用。你能用 13-gram 检查训练数据有没有见过考题。你还能根据"字节/token"和"算力/字节"两本账，为主线模型选定词表大小。

📺 **本章视频**：还没有发布。要在本机渲染，运行 `bash chapters/13-data/video/build.sh`。
🧪 **本章自检**：学完后，在 Claude Code 里输入 `/ch13-data`。

---

上一章用 scaling law 定下了主线模型的尺寸（约 0.69B 参数）和预训练的 token 预算（约 4000 亿）。这一章回答下一个问题：**这 4000 亿个 token 从哪来、长什么样？** 网上的文字几乎取之不尽。但大部分是导航栏、广告、乱码和一模一样的转载，还混着各种评测集的考题。本章讲怎样把它们变成一份数据集：干净、合法、配比合适、不含考题。然后为这份数据集训练一个合身的分词器（tokenizer）。

本章代码都在 CPU 上运行。前 5 个脚本加起来约 1 分钟。两个消融实验要训练小模型，各需几分钟 CPU 时间。机器繁忙时，墙钟时间会长得多。

```bash
uv run python chapters/13-data/code/01_noisy_crawl.py          # make a "noisy crawl": real documents + seven types of junk + leaked test questions
uv run python chapters/13-data/code/02_heuristic_filter.py     # language identification + Gopher / C4 / FineWeb rules
uv run python chapters/13-data/code/03_minhash.py              # MinHash LSH from scratch; check the S-curve formula
uv run python chapters/13-data/code/04_quality_classifier.py   # toy version of "model-based quality filtering"
uv run python chapters/13-data/code/05_decontam.py             # 13-gram decontamination
uv run python chapters/13-data/code/06_quality_ablation.py     # ablation 1: noisy vs. filtered (about 3 min of CPU time on one thread)
uv run python chapters/13-data/code/07_mixture_ablation.py     # ablation 2: three mixtures (about 4 min of CPU time on one thread)
uv run python chapters/13-data/code/08_vocab_size.py           # vocabulary size: compression × parameters and compute
uv run python -m zero.data.pipeline --config configs/tiny/data.toml   # production pipeline (tiny configuration, about half a minute)
```

## 1. 数据是最大的杠杆

先看两个和主线模型差不多大的例子。

- **MobileLLM-R1**（Meta，2025）：950M 参数。预训练只用了 4.2T token，是 Qwen3 小模型 36T 的 11.7%。但它在多项推理基准（benchmark）上追平或超过 Qwen3-0.6B。论文要检验的假设是"推理能力需要海量数据（10T 以上）"。结论是：精选约 2T token 的开放数据，再按设计好的比例重复采样，就够了。它的"留一法"实验还发现：去掉 FineWeb-Edu（一份按教育价值筛选的网页数据）后，知识、数学、代码三项**一起**变差。通用网页像胶水，把各领域粘在一起。
- **Puro-2B**（清华，2026-08）：2B 参数、约 1.4T token，只用公开数据。它拟合出的成本曲线表明，约 **4.4K 美元**的算力就能追上 Qwen2-1.5B。它的数据配方不靠自己的质量分，而是靠**代理实验（proxy experiment）**。从同一个 Qwen3-0.6B checkpoint 出发，在每个候选数据切片上续训约 8.4B token。然后看 15 项基准组成的"能力向量"。有一个发现很说明问题。把数据集 DCLM 按质量分排序：最前面的切片在 39 个候选切片里排第 3，往后 25% 的切片只排第 11。

Llama 3 的技术报告也说得很直接：和 Llama 2 相比，架构几乎没变。报告写道："性能提升主要来自数据质量和多样性的改进，以及训练规模"。在这个规模下，第 9 章的架构和第 12 章的超参带来的差异，都比不上"喂什么数据"。所以主线模型不冒架构风险（GOAL.md 3.3），把功夫花在数据上。

## 2. 开放数据集和它们的许可证

GOAL.md 3.3 要求："只用许可证允许的公开数据，每个数据集都记录来源和许可证。"下表是主线的候选。每一行都在 Hugging Face 的数据集卡上核对过（2026-09-26）：

| 数据集 | 规模（数据集卡口径） | 语言 | 许可证 | 怎么筛出来的 |
|---|---|---|---|---|
| [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) | 1.3T token（另有 score≥2 的 5.4T 版） | 英 | ODC-By 1.0 | FineWeb（Common Crawl → trafilatura 抽取 → fastText 语言识别 → Gopher/C4/FineWeb 规则 → 每个 dump 内 MinHash）+ 教育价值分类器（Llama-3-70B 标 46 万篇 → 训练小分类器 → 保留 ≥3 分，删掉 92%） |
| [DCLM-baseline 1.0](https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0) | 4T token、3B 篇 | 英 | CC-BY-4.0（卡片注明"仅供研究"，见下文） | RefinedWeb 式启发式规则 → Bloom 过滤器去重 → fastText 分类器（正例：OpenHermes 2.5 指令数据 + ELI5 高赞回答） |
| [FineWeb-2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2)（`cmn_Hani` 子集） | 中文 6.36 亿篇、parquet 1.6TB | 中（及 1000+ 种语言） | ODC-By 1.0 | FineWeb 流水线的多语言版（规则与去重按语言调过），**没有**模型打分 |
| [Ultra-FineWeb](https://huggingface.co/datasets/openbmb/Ultra-FineWeb) | 英 约 1T、中 约 120B token | 英、中 | 页面标 Apache-2.0；中文部分来自多个上游语料（待核实） | 对 FineWeb 和中文 FineWeb-edu-v2 做"验证式"筛选：用小代价的训练实验挑正负样本，再训练 fastText 分类器（MiniCPM4/5 的核心网页数据） |
| [Stack-Edu](https://huggingface.co/datasets/HuggingFaceTB/stack-edu) | 125B token、15 种编程语言 | 代码 | 页面没有许可证字段，指向 The Stack v2 的条款；每个文件带 `detected_licenses`（待核实） | StarCoder2 训练集 → 每种语言一个教育价值分类器（StarEncoder，标注来自 Llama-3-70B）→ 保留 ≥3 分（Java ≥2）。**只含文件 id**，内容要从 Software Heritage 的 S3 取 |
| [FineMath](https://huggingface.co/datasets/HuggingFaceTB/finemath) | FineMath-3+ 34B token（4+ 约 9.6B） | 英 | ODC-By 1.0 | 从 Common Crawl 召回数学页面 → 数学分类器（Llama-3.1-70B 标注）→ ≥3 分 |
| [Nemotron-CC](https://data.commoncrawl.org/contrib/Nemotron/Nemotron-CC/index.html)（v2 在 HF 上需申请） | 6.3T token（4.4T 真实 + 1.9T 合成） | 英（v2 加了多语言问答） | Common Crawl 使用条款；v2 是 NVIDIA 数据协议（允许训练，禁止再分发原始数据） | 三个分类器集成打分，分成 20 档 → 低质量文本改写成维基风格，高质量文本生成问答/摘要/知识列表（第 7 节） |

下面几点提醒，写模型卡时都要面对：

- **许可证不只是一个字段。** ODC-By 要求署名，同时受 Common Crawl 使用条款约束。DCLM 的许可证是 CC-BY-4.0，但数据集卡写着"intended for research use only"。发布模型前，要确认两者的关系。Ultra-FineWeb 页面标 Apache-2.0，可中文部分汇集了 IndustryCorpus2、WuDao、SkyPile、WanJuan、CCI3 等上游语料。这些上游语料的条款各不相同。（Puro-2B 的论文专门说明，其中的 SkyPile 用的是 Skywork 社区许可证。）Nemotron-CC 的论文还提到，他们没把 FineWeb-Edu 分类器放进集成，"因为许可证问题"：这个分类器的训练标注来自 Llama 3。
- **"开放"不等于"能下载"。** Stack-Edu 只给文件 id。内容要按数据集卡的脚本，去 Software Heritage 的 S3 桶取（要 AWS 凭证）。Nemotron-CC-v2 在 HF 上要申请。
- 除 Nemotron-CC 外，这张表的每一行都登记在 [`zero/data/sources.py`](../../zero/data/sources.py) 里。（Nemotron-CC 暂不纳入主线：v2 要申请，而且禁止再分发原始数据。这和"数据配方全部公开"的目标冲突。）许可证还没核实完的来源，下载器默认拒绝下载（第 11 节）。

## 3. 流水线全景，和一份"脏网页"

从 Common Crawl 到训练分片（shard），主流做法是下面这串步骤（FineWeb、DCLM、Nemotron-CC、Llama 3 大同小异）：

```
网页 HTML → 文本抽取 → 语言识别 → 启发式规则 → 精确去重 + 近似去重 → 模型打分过滤
         →（合成改写）→ 去污染 → 分词 → 分片 + 配比 → 训练
```

顺序有讲究：**便宜的步骤放前面**。规则和哈希每篇文档只要几微秒。模型打分要跑神经网络：FineWeb-Edu 给 15T token 打分用了 6000 个 H100 卡时。先用便宜的步骤把数据量减下来，贵的步骤才算得起。

构建环境里下载不了真实的网页数据（数据也太大）。所以 [`01_noisy_crawl.py`](code/01_noisy_crawl.py) 自己造了一份"脏网页"。`assets/tiny_corpus` 里的莎士比亚和宋词当好文档。然后脚本按网页上常见的几类垃圾往里掺。每篇文档都带一个标签，后面每一步都能检查"删得对不对"。真实世界没有这个标签，这正是数据工作难的地方。

| 类型 | 篇数 | 是什么 |
|---|---:|---|
| good | 1015 | 真实文档（每篇约 1500 字符） |
| contaminated | 16 | 好文档中间夹了一道"考题"（12 道原样、2 道改了大小写和标点、2 道改写过） |
| exact_dup | 80 | 原样转载，只有换行符和行尾空格不同 |
| near_dup | 80 | 转载时改了几个词，加上"转载自……""分享到……"这样的页眉页脚 |
| nav | 60 | 导航页：`首页 \| 关于我们 \| 联系方式 …` |
| spam | 40 | 关键词堆砌：`免费下载高清在线观看…`、`cheap best price buy now…` |
| garbled | 30 | 乱码：UTF-8 被当成 Latin-1 解码（`æå®ä¸`），或一堆符号 |
| salad | 80 | 把好文档每一行的词打乱（中文按字），行尾标点留在原位 |

这份脏网页一共 1401 篇、2.76 MB。另外留出英文 84 篇、中文 30 篇干净文档做验证集（validation set）。60 道考题（中英各 30）就是从这些留出文档里摘的。

## 4. 文本抽取、语言识别与启发式规则

**文本抽取**决定了原料的成色。Common Crawl 提供两种格式：WARC（原始 HTML）和 WET（它自己抽好的纯文本）。FineWeb 的消融（ablation）发现：用 trafilatura 从 WARC 重新抽取，训出的模型明显好于直接用 WET（WET 里导航和菜单太多）。Nemotron-CC 发现，换成 jusText 能多保留 28.6% 的高质量 token。

**语言识别**一般用 fastText 的 lid.176 模型。FineWeb 只保留"英文得分 ≥ 0.65"的文档。本章的 `02` 用一个 10 行的替代品：汉字占比 > 30% 算中文，ASCII 字母占比 > 50% 算英文，其余算"其他"并丢掉。乱码页在这一步全部被删掉。

**启发式规则**是一组便宜的统计量加阈值。[`02_heuristic_filter.py`](code/02_heuristic_filter.py) 从零实现了三篇论文里的规则，阈值用原文的值：

```python
if not 50 <= n <= 100_000:                                   bad.append("gopher_word_count")
if sum(w.lower() in stop for w in words) < 2:                bad.append("gopher_stop_words")      # the/be/to/of/and/that/have/with
if sum(c for c in counts.values() if c > 1) / nl > 0.3:     bad.append("gopher_dup_lines")       # fraction of duplicate lines
if "lorem ipsum" in text.lower() or "{" in text:             bad.append("c4_lorem_or_curly")
if sum(ln.rstrip().endswith(END_PUNCT) for ln in lines) / nl <= 0.12:  bad.append("fineweb_line_punct")
```

中文没有空格分词，所以一个汉字算一个"词"。停用词换成"的了是在和有也就不都而与之其以"。运行结果：

| 类型 | 过滤前 | 删掉 | 删除率 |
|---|---:|---:|---:|
| good | 1015 | 177 | 17% |
| contaminated | 16 | 2 | 12% |
| exact_dup / near_dup | 80 / 80 | 21 / 8 | 26% / 10% |
| nav / spam / garbled | 60 / 40 / 30 | 60 / 40 / 30 | **100%** |
| salad | 80 | 12 | 15% |

规则删掉了全部导航页、广告和乱码。但乱序文本有 85% 溜了过去。它的字符分布、行长、标点都和真文档一样，规则看不出"这不是话"。

还有一个数字值得多看一眼：**好文档被误杀了 17%**，其中 152 篇是莎士比亚。原因是 Gopher 的"重复行"规则。剧本里 `QUEEN MARGARET:` 这样的人名行反复出现，重复行占比很容易超过 30%。这些阈值是在网页上调出来的，用在剧本、诗歌、代码上会误伤好文档。生产级的 `zero/data/quality.py` 在同一批文档上也只保留 1110 篇。如果把网页规则用在代码上，tiny 语料里的代码会被删掉 96%（代码没有停用词，行尾也没有句号）。所以流水线按来源选规则：

- 网页：Gopher/C4/FineWeb。
- 代码：Codex 论文的三条（平均行长 > 100、最长行 > 1000、字母数字占比太低）。
- 上游已经过滤过的数据集（FineWeb-Edu、DCLM）：不再重复过滤。

Nemotron-CC 甚至发现，对高质量文档关掉启发式规则，MMLU 还能再高 2 分。规则删掉的不全是坏东西。

## 5. 去重：精确哈希 + MinHash LSH

网页上的重复极多：转载、镜像站、模板页。重复数据浪费算力（compute），还让模型"背书"。（Lee et al. 2021 发现，去重后模型逐字复述训练文本的比例大幅下降。）重复还会放大考题泄漏。去重（deduplication）分两层：

1. **精确去重**：规范化空白后算哈希，同一个哈希只留一篇。这一步便宜，但差一个字就认不出来。
2. **近似去重**：用两篇文档"字符 5-gram 集合"的 **Jaccard 相似度**衡量它们有多像：`J(A, B) = |A∩B| / |A∪B|`。两两比较是 O(N²)，几十亿篇文档比不过来。MinHash 和 LSH 解决这个问题。

**MinHash**：取一个随机哈希函数 h，找出集合里哈希值最小的元素。A∪B 里哈希最小的元素，以 |A∩B|/|A∪B| 的概率落在交集里。这时两个集合的最小值相等。所以：

```
P( min h(A) = min h(B) ) = J(A, B)
```

用 128 个不同的哈希函数，得到 128 个最小值，这就是文档的**签名**。两个签名里相等的位所占的比例，就是 Jaccard 的估计。[`03_minhash.py`](code/03_minhash.py) 的核心不到 60 行：

```python
def signature(self, x):            # x: the 5-grams of a document (crc32 changes each one into an integer)
    return (((x[:, None] * self.a + self.b) % PRIME) & MASK).min(axis=0)
```

最后的 `& MASK`（取低 32 位）不能省。`a·x + b` 没超过素数 p 时，"mod p"什么也没做。于是哈希值随 x 单调递增，128 个哈希函数会选中同一个最小元素，签名的各位就不再独立。我第一次写的时候漏了这一步，结果下面的"实测"和公式对不上：s = 0.3 时，实测候选率是 0.22，公式只有 0.001。

**LSH（局部敏感哈希，locality-sensitive hashing）**：把 128 位签名切成 b 段，每段 r 行（本章是 16 × 8）。只要有一段完全相同，两篇文档就进同一个桶，成为"候选对"。然后再用签名估计的 Jaccard 复核。相似度为 s 的一对成为候选的概率是：

```
P(s) = 1 − (1 − s^r)^b          # 一段全等：s^r；b 段都不全等：(1 − s^r)^b
```

这是一条 S 形曲线，拐点约在 `(1/b)^(1/r)`：

| s | b=16,r=8（本章/zero） | b=14,r=8（FineWeb） | b=450,r=20（RefinedWeb） | 实测（本章设置，每档 1000 对） |
|---:|---:|---:|---:|---:|
| 0.5 | 0.061 | 0.053 | 0.000 | 0.069 |
| 0.6 | 0.237 | 0.211 | 0.016 | 0.202 |
| 0.7 | 0.613 | 0.565 | 0.302 | 0.615 |
| 0.8 | 0.947 | 0.924 | 0.995 | 0.949 |
| 0.9 | 1.000 | 1.000 | 1.000 | 1.000 |

实测和公式吻合。r、b 越大，曲线越陡（RefinedWeb 用了 9000 个哈希），代价是要算、要存的哈希越多。FineWeb 为了省算力，选了 112 个（14 × 8）。它的论文算过：s = 0.75 时被抓到的概率是 77%，s ≥ 0.85 时几乎一定被抓到。

用在脏网页上（启发式过滤之后的 1051 篇）：精确去重后剩 992 篇，MinHash 后剩 923 篇。按"出处"数多余的副本：去重前 130 份，精确去重后 71 份，MinHash 后只剩 2 份。（这两份改动较多，相似度落在阈值以下。）68 个重复簇里，没有一个把不同出处的文档误合并。

FineWeb 的论文里还有一个反直觉的发现：对 96 个 Common Crawl 快照做**全局** MinHash 去重，模型没有变好。老快照里"幸存"下来的那 10% 恰恰是更差的数据。改成在每个快照内部去重，才追上 RefinedWeb。去重的目标是删掉"成千上万份的大簇"，不是追求一份不重。

## 6. 基于模型的质量过滤

规则抓不住"不像话"的文档。主流做法是训练一个便宜的分类器，给每篇文档打分：

- **FineWeb-Edu**：让 Llama-3-70B-Instruct 按"对中小学教育有没有价值"给 46 万个网页打 0–5 分（用加性打分提示词）。在 41 万条标注上训练一个线性回归头（Snowflake-arctic-embed-m 编码器冻结）。然后给全部 15T token 打分，保留 ≥3 分的网页。结果删掉 92%，剩 1.3T token。MMLU、ARC 等知识类基准明显提升。阈值再高，HellaSwag、PIQA 反而下降。
- **DCLM**：fastText 分类器。正例是 OpenHermes 2.5 的指令数据和 ELI5 的高赞回答，负例是随机网页。保留分数最高的约 10%。
- **Nemotron-CC**：发现 FineWeb-Edu 和 DCLM 两个分类器选出的高质量文档只有 10% 重合。于是集成三个分类器（取最大值），分成 20 档。高质量 token 的比例从 9% 提到 25%。
- **Ultra-FineWeb**（MiniCPM4）：同样用 fastText，但正负样本由小代价的训练实验来挑。

[`04_quality_classifier.py`](code/04_quality_classifier.py) 把 FineWeb-Edu 的做法缩小：请"评审员"标 250 篇，训练逻辑回归（第 5 章），再给其余文档打分。本章没有大模型，"评审员"就用我们手里的真实标签代替。这是本脚本唯一"作弊"的地方。8 个手工特征里最有用的是**词对眼熟度**：相邻两个词（中文是两个字）有多少在好文档里出现过。CCNet 用 KenLM 困惑度过滤，也是这个思路。乱序文本的相邻词对几乎都是陌生的。

```
Weights (standardized): bigram +3.31, wordlen +2.01, cjk −0.96, unique −0.60 …
```

（`bigram` 是词对眼熟度，`wordlen` 是平均词长，`cjk` 是汉字占比，`unique` 是不同词占比。）

在没有标注的 673 篇上，阈值决定了"抓得全"和"误伤少"之间的取舍：

| 阈值 | 抓到差文档 | 漏掉 | 误伤好文档 | 精确率 | 召回率 |
|---:|---:|---:|---:|---:|---:|
| 0.3 | 27 | 17 | 0 | 1.00 | 0.61 |
| 0.5 | 31 | 13 | 1 | 0.97 | 0.70 |
| 0.7 | 37 | 7 | 6 | 0.86 | 0.84 |
| 0.9 | 41 | 3 | 66 | 0.38 | 0.93 |

取 0.5：文档从 923 篇降到 871 篇。乱序文本从 68 篇降到 21 篇，好文档只少了 3 篇。阈值不是拍脑袋定的：FineWeb-Edu 在 2、3、4 之间做了消融，选了 3。

## 7. 合成改写：让模型把网页"重写一遍"

好数据会用完。Muennighoff et al.（2023）发现，同一份数据重复 4 遍以内几乎没有损失；重复更多，收益就快速下降。于是出现了**合成改写**（synthetic rephrasing）：用一个语言模型把已有文本换个写法重写，得到新的 token。几家的做法：

- **Nemotron-CC**：用 Mistral NeMo 12B，把低质量文档改写成"维基百科风格"（去掉噪声和错误）。从高质量文档生成多样的问答对、精简版、知识抽取和知识列表。一共合成了 1.9T token。对照实验让 8B 模型训练 1T token：改写低质量数据，平均 +1.5 分；把高质量数据 8 遍重复中的 4 遍换成合成数据，平均再 +0.9 分。
- **Kimi K2**：把知识类文本按不同风格和视角改写。长文档先分块，自回归地改写，再拼回去。它还检查改写与原文是否一致。数学文档改写成"学习笔记"体。同一份维基文本：原文重复 10 遍，SimpleQA 得 23.76；改写 1 次、重复 10 遍，得 27.39；**改写 10 次、每份看 1 遍，得 28.94**。
- **Phi-4**：合成数据占预训练 token 的 40%，另有 15% 是网页改写。消融发现：只用合成数据，知识类基准（TriviaQA）明显变差。所以还要混入网页数据。
- **Qwen3**：用 Qwen2.5、Qwen2.5-Math、Qwen2.5-Coder 合成了"数万亿 token"的教材、问答、指令和代码片段。

合成改写不在本章的极简代码里，因为它需要一个会写字的大模型。风险也要记住。改写可能引入幻觉。（Nemotron-CC 在个别任务上看到下降，并说明没有核对改写的事实准确性。）生成改写的模型，其许可证必须允许"用输出训练其他模型"（第 17 章讲教师模型的许可证）。主线模型第一阶段不做大规模合成。改写数据主要放在中期训练（mid-training，第 15 章），规模由第二步的消融决定。

## 8. 配比与小规模消融

有了几份干净的数据集，还要决定**配比**（mixture）：每一步训练里，英文网页、中文、代码、数学各占多少。没有公式能直接算出来。各家都靠**小规模对照实验**：

| 谁 | 怎么做 |
|---|---|
| Llama 3 | 用小模型做 scaling law 实验，预测大模型在某个配比下的表现，反复迭代。用"退火"给小数据集估值（把训练到一半的 8B 模型在 40B token 上退火，新数据占 30%）。最终配比：通用知识 50%、数学与推理 25%、代码 17%、多语言 8% |
| Qwen3 | 给 30T token 按教育价值、领域、安全等维度打标签，用小代理模型做大量消融，在"实例级"优化配比 |
| OLMo 2 | "微退火（microannealing）"：从预训练好的 checkpoint 分出一小段，比较加或不加某个数据集 |
| MobileLLM-R1 | 留一法：每次去掉一个来源，看代码、数学、知识三类探针集的 loss。再用影响函数估计每个来源的价值，算出配比 |
| Puro-2B | 从同一个 Qwen3-0.6B checkpoint 续训约 8.4B token。候选数据的占比在前 1600 步从 0 线性升到 80%。看 15 项基准的能力向量，再由人定配比 |

核心都是一句话：**同样的模型、同样的算力，只换数据，看结果**。本章用约 50 万参数的小模型做两个这样的实验。两个模型用同一个随机种子：同样的初始化，同样的抽样位置。所以它们的差值就是数据带来的（**配对比较**，第 11 章 bootstrap 的思路）。评测指标用 bits-per-byte：

```python
bpb = (nats * (nbytes > 0)).sum() / (math.log(2) * nbytes.sum())   # 06_quality_ablation.py: bpb_by_hand
```

分子是所有目标 token 的负对数似然（单位是 nats）。分母是这些 token 覆盖的 UTF-8 字节数乘 ln 2。特殊 token 记 0 字节，不计入。第 7 章讲过，bpb 和分词器无关，换了词表也能比。脚本同时调用生产级的 `zero/data/bpb.py`，两者结果一致（用 `assert` 对拍）。

**消融①：脏数据 vs 过滤后**（[`06_quality_ablation.py`](code/06_quality_ablation.py)）。两个模型各训练 200 步 × 16 × 128 = 409,600 个 token。一个用原样的 1401 篇脏网页，一个用走完 02–05 全部步骤后剩下的 856 篇：

| 训练数据 | 文档数 | token 数 | 英文 bpb（种子 0 / 1） | 中文 bpb（种子 0 / 1） | 最后一步的训练 loss（种子 0 / 1） |
|---|---:|---:|---:|---:|---:|
| 原样脏网页 | 1401 | 1,370,713 | 3.126 / 3.215 | 3.400 / 3.494 | 4.546 / 4.715 |
| 过滤后 | 856 | 842,535 | 3.086 / 3.187 | 3.316 / 3.412 | 4.659 / 4.779 |
| **配对差值（脏 − 过滤后）** | | | **0.040 / 0.028** | **0.084 / 0.081** | |

训练 loss 反而是脏数据那边更低，因为导航页、广告、重复文档都很好猜。但在验证集上，过滤后的数据更好。**训练 loss 低不等于模型好。** 比较数据，一定要在同一份干净的验证集上比。另外注意：同一种数据换一个种子，bpb 能差 0.1，比数据带来的差值还大。这正是只能做配对比较的原因，也是真实的消融要用更大的代理模型和多个种子的原因。

**消融②：配比**（[`07_mixture_ablation.py`](code/07_mixture_ablation.py)）。用同一堆干净数据（英文、中文、代码），三种配比各训练一个模型。

每种配比用 2 个种子各训练一次。（同一个种子下，三种配比的初始化相同，是配对比较。）

| 配比（英/中/代码） | 英文 bpb（种子 0 / 1） | 中文 bpb（种子 0 / 1） | 代码 bpb（种子 0 / 1） | 按均衡权重的平均（两种子平均） |
|---|---:|---:|---:|---:|
| 均衡 0.45/0.45/0.10 | 3.192 / 3.155 | 3.486 / 3.525 | 3.760 / 3.744 | 3.381 |
| 英文为主 0.80/0.10/0.10 | 2.928 / 3.018 | 3.818 / 4.057 | 3.652 / 3.794 | 3.482 |
| 代码为主 0.25/0.25/0.50 | 2.928 / 2.943 | 3.316 / 3.348 | 2.576 / 2.612 | 3.080 |

两个种子方向一致的结论有三条：

- **代码 bpb 对配比最敏感。** 代码占 0.50 时，代码 bpb 从约 3.75 降到约 2.59。（代价是代码被看了 2.37 遍。）
- **"英文为主"让英文变好、中文明显变差。** 英文比均衡配比低约 0.20，中文高约 0.43。（中文只被看了 0.08 遍。）
- **有一条结果出乎意料。** "代码为主"在英文和中文上也都比"均衡"好（英文约低 0.24，中文约低 0.17），两个种子都是这样。我们没有可靠的解释。可能是代码帮模型更快学会了通用的局部结构（MobileLLM-R1 也发现代码数据对数学有帮助）。也可能只是这个规模（50 万参数、40 万 token）特有的现象。**这正是代理实验要谨慎外推的原因。** 真实的配比实验要用大得多的代理模型，看下游能力而不只看 bpb，并确认结论不随规模改变。Puro-2B 就发现代码能力和通用能力此消彼长。

哪个配比"最好"，取决于你给各领域的权重。也就是说，先要想清楚模型的目标。这也是 Puro-2B 用"能力向量"而不是单一分数来选数据的原因。

**主线的配比（暂定）**写在 [`configs/main/data.toml`](../../configs/main/data.toml) 里：

| 类别 | 占比 | 来源 | 理由 |
|---|---:|---|---|
| 英文网页 | 0.45 | FineWeb-Edu 0.35 + DCLM 0.10 | 开放的高质量数据绝大多数是英文；网页是"胶水"（MobileLLM-R1 的留一法实验） |
| 中文网页 | 0.30 | FineWeb-2 中文 0.20 + Ultra-FineWeb 中文 0.10 | 双语目标（C-Eval、CMMLU）；Puro-2B 中文只占 9–12%，但中文不是它的目标 |
| 代码 | 0.15 | Stack-Edu | 工具调用就是按 schema 生成结构化文本；代码也帮数学（MobileLLM-R1）；Llama 3 约 17% |
| 数学 | 0.10 | FineMath-3+ | Llama 3 数学与推理占 25%；更多的数学数据放到中期训练 |

按约 400B token 的预算算，每个来源最多看 1.2 遍（FineMath-3+），其余来源都不到 1 遍（逐项见 `configs/main/data.toml` 的注释）。这些比例是**暂定值**。第二步要按上面的方法做代理实验再定（计划与成本见"主线进度"）。

## 9. 去污染：训练数据不能见过考题

如果评测题目出现在训练数据里，分数就不可信。标准做法是 **n-gram 重叠检查**。把每道考题规范化（小写、去标点；中文每个字算一个词），切成 n-gram，放进一个集合。然后扫描每篇训练文档，只要有一个 n-gram 命中，就删掉整篇。GPT-3 用 13-gram。Llama 3 用 8-gram 做污染分析。Phi-4 用 13-gram 加 7-gram 的混合规则，并把"常见 13-gram"（选择题选项的套话）列入白名单。

[`05_decontam.py`](code/05_decontam.py) 的核心只有几行：

```python
def find_contaminated(docs, eval_items, n=13):
    index = build_index(eval_items, n)                      # n-gram of a test question → question ids
    for i, d in enumerate(docs):
        for g in ngrams(norm_tokens(d["text"]), n):        # look up each n-gram of the document
            found |= index.get(g, set())
```

我们用质量过滤后的 871 篇。其中真正夹带考题的有 13 篇：原样 10 篇、改大小写标点 2 篇、改写 1 篇。换不同的 n，结果如下：

| n | 标记的文档 | 原样 | 改大小写标点 | 改写 | 其它命中 |
|---:|---:|---:|---:|---:|---:|
| 5 | 28 | 10/10 | 2/2 | 1/1 | 15 |
| 8 | 19 | 10/10 | 2/2 | 0/1 | 7 |
| 13 | 15 | 10/10 | 2/2 | 0/1 | 3 |
| 20 | 14 | 10/10 | 2/2 | 0/1 | 2 |

三个结论：

1. 规范化让"改大小写和标点"骗不过去。但**改写过的考题，13-gram 抓不到**：每隔几个词换一个词，13 个连续的词就再也对不上。Phi-4 的报告也承认，n-gram 方法挡不住改写。
2. n 太小会误报。n = 5 时，"me to the sight of"这样的常见搭配也算命中。
3. n = 13 的 3 个"其它命中"**不是误报**。宋词《御街行》等 3 首在语料里本来就出现了两次：《宋词三百首》和《全宋词》各一份。考题取自其中一份，所以这是真泄漏。它还说明：文档级去重对"两篇不同文档里有同一段文字"无能为力。

主线的去污染（decontamination）会对照第 11 章预注册里的全部基准（BFCL、C-Eval、CMMLU、GSM8K……），外加我们自己的工具调用开发集。工具调用数据还要检查函数名和 schema（GOAL.md 3.2）。命中的考题数量写进模型卡。

## 10. 训练主线分词器：词表选多大

第 7 章讲过词表大小的取舍：词表越大，同样的文本切出的 token 越少（字节/token 越大）。但 embedding 的参数是 V × d。第 7 章用 2.5 MB 的小语料，只能扫到 32K。要为主线做决定，需要更像样的语料。最好的语料就是**主线将要训练的数据本身**。

[`configs/vocab/download.toml`](../../configs/vocab/download.toml) 用第二步的下载器 `zero.data.download`，从主线的每个预训练来源里抽一小份：

- 英文：FineWeb-Edu、DCLM、FineMath。
- 中文：FineWeb-2（`cmn_Hani`）和 Ultra-FineWeb 中文。
- 代码：MiniCPM5 的 UltraData-Code-L2（9 种编程语言）。主线配置里的 Stack-Edu 只有文件 id，正文要用 AWS 凭证从 Software Heritage 取。所以这里换成正文可以直接下载的同类数据。

配比和主线一致。6 个数据集都固定到 2026-10-01 的 commit，一共 9,361 篇、46 MB 正文。[`09_build_vocab_corpus.py`](code/09_build_vocab_corpus.py) 把它们按语言合并、打乱，切出验证集（英 1.75 MB、中 0.81 MB、代码 1.63 MB），其余做训练文本。

分词器的训练文本按主线配比取（英 55%、中 30%、代码 15%，共 30 MB）。训练文本不能太少。只用 12 MB 时，词表长到约 9 万，出现 2 次以上的相邻对就用完了，更大的词表根本训练不出来。

下载了什么、每个来源下了多少、每个文件的 sha256，都记在 [runs/2026-10-01-vocab-corpus/](../../runs/2026-10-01-vocab-corpus/README.zh.md) 里。语料只在本地做测量，不进仓库，也不用于训练。（这一节的第一版用的是从 GitHub 技术文档拼的语料。写书时，构建环境访问不了 Hugging Face。那一版的结果放在下面表格后面，正好用来看"同分布"的影响有多大。）

代价按主线的形状算（`configs/main/pretrain.toml`：28 层、宽 1280、共享 embedding），只换 V。参数 = 非 embedding 605.6M + V × 1280，**总量不能超过 0.8B**。每 token 的训练算力 ≈ 6 × N_matmul（包括 1280 × V 的 lm_head 矩阵乘）+ 注意力。真正要比的是**读完同样多的文本要花多少算力**：

```
FLOPs / 字节 = (FLOPs / token) ÷ (字节 / token)
```

下面是 `08_vocab_size.py --corpus <dir> --refs --train-mb 30` 的结果。（"加权"按英 0.55 / 中 0.30 / 代码 0.15；FLOPs/字节以 V = 65,536 时每 token 的 FLOPs 为单位；CPU 上 16 线程，7 个词表一共训练约 2 分钟。）

| 分词器 | 词表 V | 英文 | 中文 | 代码 | 加权 字节/token | embedding | 总参数 | 每 token 算力 | **FLOPs / 字节** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| zero BPE | 16,384 | 3.68 | 3.45 | 3.19 | 3.54 | 21.0M | 626.6M | 6.58 GFLOP | 0.2674 |
| zero BPE | 32,768 | 3.99 | 3.86 | 3.46 | 3.87 | 41.9M | 647.6M | 6.70 GFLOP | 0.2489 |
| zero BPE | 49,152 | 4.13 | 4.09 | 3.59 | 4.04 | 62.9M | 668.5M | 6.83 GFLOP | 0.2431 |
| zero BPE | **65,536** | 4.22 | 4.25 | 3.66 | 4.15 | 83.9M | 689.5M | 6.96 GFLOP | **0.2412** |
| zero BPE | 98,304 | 4.31 | 4.46 | 3.76 | 4.27 | 125.8M | 731.5M | 7.21 GFLOP | 0.2425 |
| zero BPE | 131,072 | 4.36 | 4.62 | 3.81 | 4.36 | 167.8M | 773.4M | 7.46 GFLOP | 0.2460 |
| zero BPE | 151,936 | 4.39 | 4.70 | 3.83 | 4.40 | 194.5M | **800.1M（超线）** | 7.62 GFLOP | 0.2491 |
| Qwen2/Qwen3 分词器（参考） | 151,936 | 4.39 | 3.90 | 4.00 | 4.18 | 194.5M | 800.1M | 7.62 GFLOP | 0.2618 |
| Llama 3 分词器（参考） | 128,256 | 4.53 | 3.27 | 4.05 | 4.08 | 164.2M | 769.8M | 7.44 GFLOP | 0.2619 |

（参考分词器由 llama.cpp 的 `ggml-vocab-{qwen2,llama-bpe}.gguf` 重建，并用 llama.cpp 自带的测试用例核对过切分。参数与算力按"放在主线形状上"计算。）

怎么读这张表：

- **压缩率收益递减。** 16K → 32K，加权字节/token 提高 9.5%；64K → 128K 只提高 5.1%。中文涨得最多（中文"词"多，词表大才装得下）。英文和代码在 64K 之后涨得很慢。
- **算力/字节在 64K 最低**，往两边都上升。96K 贵 0.5%，48K 贵 0.8%（词表小，切出的 token 多）。128K 贵 2.0%，151,936 贵 3.3%（lm_head 变大）。而且 Qwen 的词表大小放在这个宽度上，总参数是 800.1M，**超过 0.8B 的上限**。
- **自己训练的分词器在中文上占优，在代码和英文上不占优。** 同样 151,936 的词表，我们的中文是 4.70 字节/token，Qwen 只有 3.90（我们高 20%）。英文两者持平（4.39）。代码反而是 Qwen 高 4.5%。Llama 3 的英文和代码也都比我们 128K 的分词器好（高 3.9% 和 6.2%）。Qwen 和 Llama 的分词器训练数据比我们的 30 MB 大几个数量级，也杂得多。我们唯一明显的优势是中文配比高（30%，而 Qwen 要照顾 119 种语言）。
- **同分布的优势比看上去小。** 第一版测量（下面折叠的表）用的是技术文档，训练文本和验证文本来自同一类文档。那时我们的中文比 Qwen 高 43%，代码也更好。换成真实的网页和代码样本后，中文优势缩到 20%，代码优势消失。在什么数据上测，比测得多精细更重要。
- Tao et al.（2024）的词表 scaling law，对 3B 非词表参数、按 Chinchilla 配比训练的模型，推荐的最优词表约 4 万。主线的非词表参数约 6 亿，但过训练（约 700 token/参数）会把最优值往上推。这和表格"64K 最省算力"的结论方向一致。

<details>
<summary>第一版测量：GitHub 技术文档语料（2026-09，写书时的构建环境访问不了 Hugging Face）</summary>

语料：中文来自《动手学深度学习》中文版、Kubernetes 中文文档、JavaGuide、CS-Notes、Python-100-Days（去掉了代码块和 HTML 注释里的英文原文）。英文来自《动手学深度学习》英文版、Kubernetes 英文文档和 CPython 文档。代码是 CPython 的 Python 与 C 源码。一共约 75 MB（来源见本章参考文献；整理脚本没有保留）。验证集：英 1.75 MB、中 0.77 MB、代码 1.63 MB。

| 分词器 | 词表 V | 英文 | 中文 | 代码 | 加权 字节/token | **FLOPs / 字节** |
|---|---:|---:|---:|---:|---:|---:|
| zero BPE | 16,384 | 4.23 | 4.41 | 3.74 | 4.21 | 0.2245 |
| zero BPE | 32,768 | 4.44 | 4.96 | 3.98 | 4.53 | 0.2129 |
| zero BPE | 49,152 | 4.51 | 5.25 | 4.08 | 4.67 | **0.2103** |
| zero BPE | 65,536 | 4.56 | 5.43 | 4.14 | 4.76 | **0.2103** |
| zero BPE | 98,304 | 4.60 | 5.66 | 4.21 | 4.86 | 0.2134 |
| zero BPE | 131,072 | 4.62 | 5.79 | 4.25 | 4.91 | 0.2182 |
| zero BPE | 151,936 | 4.62 | 5.87 | 4.26 | 4.94 | 0.2217 |
| Qwen2/Qwen3 分词器（参考） | 151,936 | 4.29 | 4.11 | 4.09 | 4.21 | 0.2603 |
| Llama 3 分词器（参考） | 128,256 | 4.38 | 3.75 | 4.18 | 4.16 | 0.2568 |

技术文档比网页"整齐"得多，所有分词器的压缩率都更高。当时的结论是"48K 和 64K 并列最省算力"。这次在主线数据样本上，64K 单独最低，选择不变。

</details>

**主线的词表定为 65,536。**（`configs/main/data.toml` 与 `configs/main/pretrain.toml` 已一致；第 7 章写的"暂定"在这里确认。）理由如下：

- 在主线数据样本上，它的算力/字节最低（两次测量都在最低点上）。
- 它比 48K 多 4.0% 的中文压缩率。中文是我们的弱项数据，token 越省越好。
- embedding 83.9M，只占总参数的 12%。
- 它是 2 的幂次，也是 128 的倍数，GPU 上的矩阵乘很整齐。

第二步正式下载完数据后，再用完整的训练语料测一次。这次的样本只来自各数据集最早的几个分片（2013–2014 年的网页，见 runs 目录的记录）。如果结论变了，再改。

**预切分正则**：第 7 章留了一个问题。zero 的正则和 Qwen2/Qwen3 相同，而 Qwen3.5 把字母串从 `\p{L}+` 改成了 `[\p{L}\p{M}]+`。`\p{M}` 是"组合符号"：天城文、泰文的元音符号，阿拉伯文的变音符号。旧正则会在这些符号处把一个词切碎：

| 句子 | qwen2 正则 | qwen3.5 正则 |
|---|---:|---:|
| 印地语（世界人权宣言第一条） | 36 块，`सभी` 被切成 `सभ` + `ी` | 14 块 |
| 泰语 | 16 块 | 1 块 |
| 中文 / 英文 | 3 / 13 块 | 3 / 13 块（完全相同） |

在主线数据样本的验证集上（每种语言取前 20 万字符），英文和代码的词块完全相同。中文只有 4 处不同：2 处是 emoji 后面的变体选择符 U+FE0F（它属于 \p{M}），另 2 处是混在中文网页里的一段泰文。真实网页从来不是纯净的单一语言。我们用两种正则各训练一个 65,536 词表的分词器，验证集字节/token 在三种语言上完全相同（英 4.217、中 4.254、代码 3.665）。所以**主线采用 qwen3.5 正则**：对中英文和代码零代价，对其他语言更合理，也和最新的 Qwen 一致。llama.cpp 已经支持这个预切分类型（`qwen35`）：导出 GGUF 时，把 `pre_tokenizer` 参数设成 `qwen35`。这个选择只通过配置生效（`configs/main/data.toml` 的 `pretokenize = "qwen3.5"`）。`zero/tokenizer.py` 的默认值不变，已有的测试和 tiny 流水线照旧运行。

## 11. 许可证与出处：每一步都要留下记录

数据集要写进模型卡，也要能被别人复现。生产级流水线在每一步都留下记录：

- 下载器（[`zero/data/download.py`](../../zero/data/download.py)）只下载 `sources.py` 登记过的来源，许可证"待核实"的默认拒绝。每一行 JSON 都带上数据集名、子集、revision、行号和数据集自带的元数据（URL、dump、质量分数、代码文件的 `detected_licenses`）。下载器给每个分片记 sha256，并且可以续传。
- 流水线（[`zero/data/pipeline.py`](../../zero/data/pipeline.py)）把每一步的中间结果落盘成 JSONL，最后写 `manifest.json`。清单包括以下内容：
  - 每个来源每一步后剩多少篇、多少字节（漏斗）；
  - 删除原因和重复簇数；
  - 去污染命中了哪些题；
  - 分词器哈希与各来源的压缩率；
  - 分片的 token 数、配比，以及"按预算要看几遍"；
  - 许可证与来源地址、配置哈希和 git commit。
- 每个分片的元数据里都有分词器哈希（第 7 章）。这样可以防止拿 A 分词器切的数据去训练 B 分词器的模型。

## 12. 小结

- **数据是最大的杠杆**：MobileLLM-R1 用 11.7% 的 token 追平了 Qwen3-0.6B 的推理能力。Puro-2B 用公开数据和代理实验，以几千美元追上 Qwen2-1.5B。
- **流水线**：抽取 → 语言识别 → 启发式规则 → 精确 + 近似去重 → 模型打分 →（合成改写）→ 去污染 → 分词分片。便宜的步骤放前面。规则抓得住格式问题，抓不住"不像话"。规则的阈值依赖领域，要按来源选。
- **MinHash**：P(签名某位相等) = Jaccard。**LSH**：P(候选) = 1 − (1 − s^r)^b，拐点在 (1/b)^(1/r)。
- **模型打分**：大模型标一小部分，训练便宜的分类器给全部数据打分。阈值是精确率和召回率的取舍，要靠消融来定。
- **配比**：没有公式，靠代理模型的对照实验。比较时用配对设计，和与分词器无关的 bpb。
- **去污染**：13-gram 抓得住原样泄漏和改格式的泄漏，抓不住改写。n 太小会误报。
- **分词器**：比较"算力/字节"，并受"总参数 ≤ 0.8B"约束。主线的词表与正则见第 10 节。

---

## 从极简代码到生产级代码

| 极简代码（`code/`） | 生产级代码（`zero/`） | 多做了什么、为什么 |
|---|---|---|
| `02` 的 `lang_id`（汉字/字母占比） | `zero/data/pipeline.py` 的 `langid_stage`（同样的粗检）；第二步换成 fastText lid.176 | 真实网页有上百种语言，需要一个真正的语言识别模型 |
| `02` 的 `heuristic_reasons`（11 条规则） | `zero/data/quality.py` 的 `quality_check`（TOML 的 `[quality]` 段可以覆盖 `QualityThresholds`）；`pipeline.py` 的 `code_quality_reasons`（Codex 的三条） | 阈值可配置；按来源选规则（`heuristics = "web" / "code" / "none"`）；清单按规则统计删除原因 |
| `03` 的 `MinHash` / `lsh_candidates` / `near_dedup` | `zero/data/dedup.py` 的 `MinHasher`、`near_dedup`（同样的"取低 32 位"哈希、同样的并查集）+ `exact_dedup` | `num_perm / bands / ngram / threshold` 可调；来源内去重后可以再跨来源去重一次；单进程实现。第二步的规模要换分布式 MinHash（如 datatrove）。**尚未在大规模数据上验证** |
| `04` 的逻辑回归 | `pipeline.py` 的 `score_stage`：对数据集自带的分数设阈值（`score_field` / `score_min`，如 FineWeb-Edu 的 `int_score`），或任意 `模块:类名` 分类器（`quality.py` 的 `QualityClassifier` 协议） | 主线用上游已经打好的分数；中文质量分类器是第二步的任务 |
| `05` 的 `find_contaminated` | `zero/data/decontam.py` 的 `NgramIndex`（用 8 字节哈希存 n-gram，短题用子串匹配）+ `pipeline.py` 的 `decontam_stage` | 从 JSONL 读评测集（默认取全部字符串字段，**包括工具名和 schema**）；命中明细写进清单 |
| `06` 的 `bpb_by_hand` | `zero/data/bpb.py` 的 `token_byte_lengths`、`evaluate_bpb`、`bpb_stats`（nanochat `evaluate_bpb` 的写法） | 预先算好"token id → 字节数"查找表；特殊 token 与 ignore 位置不计；多卡时分子、分母分别 all_reduce |
| `07` 的逐行按配比抽来源 | `zero/data/mixture.py` 的 `MixtureLoader`（第 14 章） | 抽样只由 (seed, rank, 行号) 决定，续训逐字节一致 |
| `08` 的 `train_tokenizer(..., "qwen3.5")` | `zero/data/pipeline.py` 的 `train_tokenizer`、`PRETOKENIZE_PRESETS`（`qwen2` 与 `zero/tokenizer.py` 的默认值完全相同） | 正则只通过配置切换；按配比从各来源采样训练文本（`sample_bytes`） |
| 无 | `zero/data/download.py` | 第二步的流式下载：许可证闸门、出处字段、分片 sha256、续传、下载量规划（`plan_targets`）。**尚未验证** |
| 无 | `zero/data/pipeline.py` 的 `run_pipeline` + `configs/{tiny,main}/data.toml` | 一条命令跑完 10 个阶段，每个阶段落盘。最后写 `manifest.json`，以及可以贴进 `pretrain.toml` 的 `pretrain_sources.toml` |

**对拍与测试**（`uv run pytest tests/test_bpb.py tests/test_pipeline.py tests/test_download.py tests/test_data.py`）：

- `test_bpb.py`：手算的小例子（4 个 token、logits 全 0 → bpb 正好是 1.0；非均匀 logits 的解析值）；均匀模型的不变量 bpb = log2(V) ÷ (字节/token)；特殊 token、ignore_index、负数标签不计入；与普通交叉熵的换算关系；`token_byte_lengths` 逐 token 加起来等于原文的 UTF-8 字节数（包括含"半个汉字"的 token）。
- `test_pipeline.py`：各阶段的纯函数；qwen2 预设与 `train_bpe` 的哈希完全相同；qwen3.5 预设的往返、数字逐个切、特殊 token id、存取后正则还在；小语料端到端（文本 + JSONL 两种输入、分数阈值、精确重复、泄漏的考题被删、分片能被 `PackedDataLoader` 读出）。
- `test_download.py`：用假数据代替网络。测试出处字段、分片轮换与 sha256、中途停下后续传、Stack-Edu 按 blob_id 取内容、许可证闸门、下载量规划，以及 `configs/main/data.toml` 的下载规格能否解析。
- `06_quality_ablation.py` 每次评估都用 `assert` 检查：手算的 bpb 与 `zero/data/bpb.py` 一致。

**接进训练循环**（已接入 `zero/train/trainer.py`）：`Trainer` 用分词器算一次 `token_byte_lengths`。`Trainer.evaluate` 把同一批验证 batch 交给 `bpb_stats`，把 `val_bpb` 和 `val_loss` 一起写进日志。（SFT 格式的数据或没有分词器时跳过这一步。）tiny 配置上的一次运行记录到 `val_bpb 3.997`（极小配置演示）。

---

## 主线进度

### 极小配置演示（CPU，`configs/tiny/data.toml`，assets/tiny_corpus）

> 以下是**极小配置演示**。它只说明生产级流水线的每一步都真的在工作。2.5 MB 的玩具语料上的比例不代表真实网页数据。

```bash
uv run python -m zero.data.pipeline --config configs/tiny/data.toml
```

下面是本机一次运行的漏斗（CPU 时间约 10 秒；机器繁忙时墙钟时间 48 秒）。每格是该步之后剩下的文档数：

| 来源（许可证） | 读取 | 清洗 | 语言 | 规则 | 打分 | 去重 | 去污染 | 训练 / 验证文档 | 训练 token | 验证集字节/token |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| shakespeare（公有领域） | 615 | 615 | 615 | 528 | 528 | 528 | 528 | 502 / 26 | 351,245 | 2.62 |
| chinese_poetry（MIT） | 225 | 225 | 225 | 211 | 211 | 211 | 211 | 201 / 10 | 422,715 | 2.52 |
| code（待核实：仓库许可证未定） | 81 | 81 | 81 | 81 | 81 | 81 | 81 | 77 / 4 | 73,511 | 2.01 |

- 清单里记录了规则的删除原因。莎士比亚：重复行 70、行尾标点 13、短行 4。诗词：重复行 8，另有 6 篇"含字母的词"不到 80%。（中文按字切词后，每个标点也算一个"词"；短句多的词牌标点密度高。）代码用 Codex 规则，一篇都没删。
- 去重和去污染都删掉 0 篇。tiny 语料本身没有重复文档。（第 9 节那三首重复的宋词分在不同的"文档"里，文档级去重看不出来。）语料也不含工具调用开发集 `tool_dev.jsonl` 的内容（100 道题，含函数名与 schema）。
- 分词器的词表是 2048（qwen2 正则），它的哈希写进每个分片的元数据。`out/tiny/data_pipeline/pretrain_sources.toml` 可以直接贴进 tiny 预训练配置。用它训练 20 步（`zero.train.pretrain`，1.31M 参数），验证 loss 是 6.80。再用 `zero/data/bpb.py` 在三个来源的验证分片上评估，得到 bpb：莎士比亚 3.553、诗词 3.749、代码 5.146。（只训练了 20 步，这些数字只说明接口通了。）

```python
tb = token_byte_lengths(Tokenizer.load("out/tiny/data_pipeline/shards/tokenizer.json"))
val = PackedDataLoader("out/tiny/data_pipeline/shards/code_val_*.bin", seq_len=128, batch_size=8, shuffle=False)
print(bpb_stats(model, val, tb, steps=8).bpb)
```

### 待 GPU 训练后补充

- 第二步：用真实数据运行 `download.py`（先用 `--max-docs 1000` 核对字段）和 `pipeline.py`（第 5 步换成分布式 MinHash）。把真实的漏斗、删除原因、去污染命中写到这里。
- 配比的代理实验：起点是第 12 章阶梯实验 `configs/ladder/l150m.toml` 的中途 checkpoint。每个候选续训约 2B token，候选占比从 0 线性升到 80%。看各领域开发集的 bpb 和小基准。先比来源，再比 3–4 个配比，最后定 FineWeb-2 中文分类器的阈值。`zero.tools.estimate_cost` 估算单个实验 1.8 卡时、约 $4（假设 H100、MFU 0.4、$2.5/卡时，**尚未在 GPU 上验证**）。计划做 20–30 个，连同评测约 $100–200。这在 GOAL.md 3.4 的"第 12–13 章小规模实验 ~$1,200"之内。
- 在真实样本上重测第 10 节的词表表格，确认 V。定稿后，同步 `configs/main/pretrain.toml` 的 `model.vocab_size`。
- 三项待核实许可证的核实结果（Ultra-FineWeb 中文的上游、Stack-Edu、DCLM 的"仅供研究"）。
- 最终数据集的清单（`manifest.json`）随模型卡发布。

---

## 前沿观察

- **自动化的配比优化**：DoReMi、RegMix（用小模型拟合"配比 → loss"的回归）、MobileLLM-R1 的影响函数（AutoMixer）都直接算出配比。它们在各自的论文里有效。但在头部开源模型的技术报告里，最终配比大多仍是"代理实验 + 人工决定"（Llama 3、Puro-2B 都这么写）。Qwen3 只说用了代理模型做实例级优化，没有公开方法。所以这还不算共识。
- **按质量排序的课程学习**：Puro-2B 把每个来源内部按质量分从低到高排序，越好的越靠后，并配合检查点平均。这和第 15 章"中期训练换上最好的数据"是同一个方向。更细的排序方式还在探索中。
- **语义去重**：用句向量聚类，去掉"意思相同、字面不同"的文档（SemDeDup）。Llama 3 在后训练数据上用了它。在预训练数据上，还没有看到多家采用。

---

## 采用方与来源

| 技术 | 采用方（主力版本） | 来源 |
|---|---|---|
| 去重（精确 + MinHash 近似） | Llama 3（URL 级、全局 MinHash 文档级、行级）；Nemotron（全局模糊去重 + 精确子串去重，NeMo Curator）；SmolLM（FineWeb / FineWeb-Edu：每个 dump 内 MinHash，5-gram，14 × 8）；OLMo 2（DCLM-baseline：Bloom 过滤器去重）；Puro-2B（来源内 MinHash）；MiniCPM（Ultra-FineWeb-L1） | [Llama 3 §3.1.1](https://arxiv.org/abs/2407.21783)；[Nemotron-CC §2.1](https://arxiv.org/abs/2412.02595)；[FineWeb §3.4、附录 E](https://arxiv.org/abs/2406.17557)；[DCLM 数据集卡](https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0)；[OLMo 2 §2.4](https://arxiv.org/abs/2501.00656)；[Puro-2B §3.6.2](https://www.alphaxiv.org/abs/2608.27370)；[Ultra-FineWeb 数据集卡](https://huggingface.co/datasets/openbmb/Ultra-FineWeb) |
| 基于模型的质量过滤 | Llama 3（Llama 2 标注 → DistilRoberta 质量、代码、推理分类器）；Qwen3（30T token 按教育价值等多维度打标签）；SmolLM2（FineWeb-Edu、Stack-Edu、FineMath 分类器）；OLMo 2（DCLM fastText + FineWeb-Edu 分类器筛选 Dolmino）；Nemotron（三分类器集成）；MiniCPM4（Ultra-FineWeb fastText）；Phi-4（~10⁶ 条 LLM 标注训练的小分类器） | [Llama 3 §3.1.1](https://arxiv.org/abs/2407.21783)；[Qwen3 §3.1](https://arxiv.org/abs/2505.09388)；[SmolLM2](https://arxiv.org/abs/2502.02737)；[OLMo 2 §4.3](https://arxiv.org/abs/2501.00656)；[Nemotron-CC §2.2](https://arxiv.org/abs/2412.02595)；[Ultra-FineWeb](https://arxiv.org/abs/2505.05427)；[Phi-4 §2.3](https://arxiv.org/abs/2412.08905) |
| 合成改写 / 合成数据 | Kimi K2（知识改写、数学"学习笔记"改写）；Nemotron（Nemotron-CC 的 1.9T 合成 token）；Qwen3（Qwen2.5 系列合成数万亿 token）；Phi-4（合成 40% + 网页改写 15%）；OLMo 2（Dolmino 里 MIND 改写的 TinyGSM）；MiniCPM（Ultra-FineWeb-L3 问答生成与多风格改写） | [Kimi K2 §2.2](https://arxiv.org/abs/2507.20534)；[Nemotron-CC §2.3](https://arxiv.org/abs/2412.02595)；[Qwen3 §3.1](https://arxiv.org/abs/2505.09388)；[Phi-4 §3.2](https://arxiv.org/abs/2412.08905)；[OLMo 2 §4.4](https://arxiv.org/abs/2501.00656)；[Ultra-FineWeb 数据集卡](https://huggingface.co/datasets/openbmb/Ultra-FineWeb) |
| 去污染（n-gram 重叠） | Llama 3（8-gram 污染分析；后训练数据与基准提示词精确匹配去重）；Phi-4（13-gram + 7-gram 混合）；OLMo 2（FLAN 与评测集 n-gram 重叠 ≥10% 的删除）；Puro-2B（SFT 数据 13-gram）；GPT-3（13-gram，方法的出处） | [Llama 3 §5.1、§5.2](https://arxiv.org/abs/2407.21783)；[Phi-4 附录 B](https://arxiv.org/abs/2412.08905)；[OLMo 2 §4.3](https://arxiv.org/abs/2501.00656)；[Puro-2B §3.5](https://www.alphaxiv.org/abs/2608.27370)；[GPT-3 附录 C](https://arxiv.org/abs/2005.14165) |
| 用小规模实验定配比 | Llama 3（scaling law 实验 + 退火估值）；Qwen3（小代理模型）；OLMo 2（微退火）；Phi-4（1T token、7B 规模的配比消融）；MobileLLM-R1（留一法 + 影响函数）；Puro-2B（Qwen3-0.6B 代理续训） | [Llama 3 §3.1.2–3.1.3](https://arxiv.org/abs/2407.21783)；[Qwen3 §3.1](https://arxiv.org/abs/2505.09388)；[OLMo 2 §4.4](https://arxiv.org/abs/2501.00656)；[Phi-4 §3.2](https://arxiv.org/abs/2412.08905)；[MobileLLM-R1 §2](https://arxiv.org/abs/2509.24945)；[Puro-2B §3.6.1](https://www.alphaxiv.org/abs/2608.27370) |
| 启发式规则（Gopher / C4 / FineWeb） | FineWeb（SmolLM 的数据）；DCLM（RefinedWeb 规则，OLMo 2 的数据）；Llama 3（重复 n-gram、脏词、token 分布 KL）；Nemotron（只用在低质量部分） | [Gopher 附录 A](https://arxiv.org/abs/2112.11446)；[C4](https://arxiv.org/abs/1910.10683)；[FineWeb §3.3–3.6](https://arxiv.org/abs/2406.17557)；[Llama 3 §3.1.1](https://arxiv.org/abs/2407.21783) |

待核实：FineWeb-2 中文子集的 token 数（数据集卡只给了文档数和 parquet 大小）；Nemotron-CC-v2 许可条款的原文（只读到了 Puro-2B 论文的转述和 HF 页面的 `license: other`）；DCLM 数据集卡"仅供研究"与 CC-BY-4.0 的关系；Codex 论文第 3.1 节里"字母数字占比太低"的具体阈值（本章取 0.25，论文原文没有给数字，待核实）。

---

## 引导问题

带着这些问题去问 Claude Code，直到你能用自己的话讲清楚：

1. FineWeb 发现，对 96 个快照做全局去重反而更差。为什么"更彻底的去重"会让数据变差？如果你只有一个快照，这个问题还存在吗？
2. `03_minhash.py` 用 128 个哈希、16 × 8。如果改成 32 × 4，S 曲线的拐点移到哪里？对"转载时改了几个词"的文档，和"两篇不同但主题相近的文章"，分别有什么影响？
3. 本章的质量分类器最有用的特征是"词对眼熟度"。如果评审员标注的 250 篇全是莎士比亚、没有宋词，分类器会怎么对待中文文档？真实的 FineWeb-Edu 分类器有没有类似的偏向？（提示：阈值再高，HellaSwag 反而下降。）
4. 两个消融实验只训练了几十万 token。这么小的实验能用来决定主线的配比吗？Puro-2B 和 Llama 3 用什么办法让代理实验更可信？
5. 13-gram 去污染抓不到改写过的考题。除了减小 n，还有什么办法？（提示：Phi-4 的"新题"评测、第 11 章的自建开发集。）
6. 词表从 64K 加到 128K，主线模型的推理速度和 GGUF 文件大小会怎样变？对"在笔记本上运行的工具调用助手"这个目标，哪一边更重要？

## 动手任务

每个任务都要运行代码，看到结果。

**任务 1（基础）**：在 `02_heuristic_filter.py` 里，把 Gopher 的重复行阈值从 0.3 改成 0.5，重新运行。记录莎士比亚的误杀率和导航页的删除率。再想一想：为什么不能只为莎士比亚调阈值？

**任务 2（核心）**：在 `03_minhash.py` 里，把 `near_dedup` 的 `bands` 分别改成 8 和 32（`num_perm` 保持 128），各运行一次。对照 S 曲线表，解释"多余副本"和"误合并"的变化。再在 `01_noisy_crawl.py` 的 `near_copy` 里，把改词比例从 1/60 调到 1/10。MinHash 还能抓到这些副本吗？

**任务 3（挑战）**：给 `05_decontam.py` 加上 Phi-4 的"白名单"思路。先在训练文档里统计出现次数最多的 13-gram（比如出现在 5 篇以上的），检查时跳过它们。再把 n 降到 8。"其它命中"能不能降下来，同时原样泄漏仍然全部抓到？然后用 `06_quality_ablation.py` 的框架，比较"去污染前 / 后"训练的两个模型在考题上的 bpb。泄漏到底让模型在考题上"好"了多少？

---

## 想深入：CS336

本章对应斯坦福 CS336（Spring 2026）<https://cs336.stanford.edu/>：

- **第 13 讲：数据（来源与数据集）**。Common Crawl、网页抽取、各个开放数据集是怎么来的，以及数据的版权与许可问题。这一讲是本章第 2 节的展开。
- **第 14 讲：数据（过滤、去重、配比、合成数据）**。语言识别、质量分类器、MinHash/LSH、配比与合成数据。这一讲和本章第 4–9 节是同一条线，讲得更系统。
- **作业 4（Data）**：从 Common Crawl 的原始数据出发，自己实现 HTML 抽取、语言识别、PII 脱敏、有害内容过滤、质量分类器、精确去重与 MinHash 去重。再用过滤后的数据训练模型，在排行榜上比较验证集 loss。本章的 `02`–`05` 是它的缩小版热身。作业仓库：<https://github.com/stanford-cs336/assignment4-data>

---

## 本章参考文献

- Zhao et al. *MobileLLM-R1: Exploring the Limits of Sub-Billion Language Model Reasoners with Open Training Recipes*，ICLR 2026：<https://arxiv.org/abs/2509.24945>
- Luo et al. *PuRo-2B: Poor Lab's Qwen2-1.5B Trained on RTX 5090 within $5090*，2026：<https://www.alphaxiv.org/abs/2608.27370>
- Penedo et al. *The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale*，2024：<https://arxiv.org/abs/2406.17557>；FineWeb-Edu 数据集卡：<https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu>
- Penedo et al. *FineWeb2: One Pipeline to Scale Them All — Adapting Pre-Training Data Processing to Every Language*，2025：<https://arxiv.org/abs/2506.20920>；数据集卡：<https://huggingface.co/datasets/HuggingFaceFW/fineweb-2>
- Li et al. *DataComp-LM: In search of the next generation of training sets for language models*，2024：<https://arxiv.org/abs/2406.11794>；数据集卡：<https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0>
- Su et al. *Nemotron-CC: Transforming Common Crawl into a Refined Long-Horizon Pretraining Dataset*，ACL 2025：<https://arxiv.org/abs/2412.02595>
- Wang et al. *Ultra-FineWeb: Efficient Data Filtering and Verification for High-Quality LLM Training Data*，2025：<https://arxiv.org/abs/2505.05427>；数据集卡：<https://huggingface.co/datasets/openbmb/Ultra-FineWeb>
- Allal et al. *SmolLM2: When Smol Goes Big — Data-Centric Training of a Small Language Model*（Stack-Edu、FineMath），2025：<https://arxiv.org/abs/2502.02737>；<https://huggingface.co/datasets/HuggingFaceTB/stack-edu>、<https://huggingface.co/datasets/HuggingFaceTB/finemath>
- Llama Team. *The Llama 3 Herd of Models*，2024：<https://arxiv.org/abs/2407.21783>
- Qwen Team. *Qwen3 Technical Report*，2025：<https://arxiv.org/abs/2505.09388>
- Kimi Team. *Kimi K2: Open Agentic Intelligence*，2025：<https://arxiv.org/abs/2507.20534>
- Abdin et al. *Phi-4 Technical Report*，2024：<https://arxiv.org/abs/2412.08905>
- OLMo Team. *2 OLMo 2 Furious*，2025：<https://arxiv.org/abs/2501.00656>
- Maini et al. *Rephrasing the Web (WRAP)*，2024：<https://arxiv.org/abs/2401.16380>
- Rae et al. *Scaling Language Models: Methods, Analysis & Insights from Training Gopher*（附录 A 的质量规则），2021：<https://arxiv.org/abs/2112.11446>
- Raffel et al. *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer*（C4），2019：<https://arxiv.org/abs/1910.10683>
- Wenzek et al. *CCNet: Extracting High Quality Monolingual Datasets from Web Crawl Data*，2019：<https://arxiv.org/abs/1911.00359>
- Lee et al. *Deduplicating Training Data Makes Language Models Better*，2021：<https://arxiv.org/abs/2107.06499>
- Broder. *On the resemblance and containment of documents*（MinHash），1997
- Brown et al. *Language Models are Few-Shot Learners*（GPT-3，附录 C 的 13-gram 去污染），2020：<https://arxiv.org/abs/2005.14165>
- Chen et al. *Evaluating Large Language Models Trained on Code*（Codex，第 3.1 节的代码过滤规则），2021：<https://arxiv.org/abs/2107.03374>
- Muennighoff et al. *Scaling Data-Constrained Language Models*，2023：<https://arxiv.org/abs/2305.16264>
- Tao et al. *Scaling Laws with Vocabulary: Larger Models Deserve Larger Vocabularies*，NeurIPS 2024：<https://arxiv.org/abs/2407.13623>
- Karpathy. nanochat（`nanochat/loss_eval.py` 的 `evaluate_bpb`）：<https://github.com/karpathy/nanochat>
- Qwen3.5-0.8B 的 `tokenizer.json`（预切分正则，2026-09 读取）：<https://huggingface.co/Qwen/Qwen3.5-0.8B/blob/main/tokenizer.json>
- llama.cpp 的预切分类型 `qwen35` 与 vocab 测试文件：<https://github.com/ggml-org/llama.cpp>（`src/llama-vocab.cpp`、`models/ggml-vocab-*.gguf`）
- 第 10 节词表测量用的语料（2026-10，主线预训练来源的样本，记录见 [runs/2026-10-01-vocab-corpus](../../runs/2026-10-01-vocab-corpus/README.zh.md)）：[FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)（`sample-10BT`）、[DCLM-baseline 1.0](https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0)、[FineMath](https://huggingface.co/datasets/HuggingFaceTB/finemath)（`finemath-3plus`）、[FineWeb-2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2)（`cmn_Hani`）、[Ultra-FineWeb](https://huggingface.co/datasets/openbmb/Ultra-FineWeb)（`zh`）、[UltraData-Code](https://huggingface.co/datasets/openbmb/UltraData-Code)（`UltraData-Code-L2`，MiniCPM5 的代码数据）
- 第 10 节第一版词表测量用的语料：[d2l-ai/d2l-zh](https://github.com/d2l-ai/d2l-zh)、[d2l-ai/d2l-en](https://github.com/d2l-ai/d2l-en)、[kubernetes/website](https://github.com/kubernetes/website)（`content/{zh-cn,en}/docs`）、[Snailclimb/JavaGuide](https://github.com/Snailclimb/JavaGuide)、[CyC2018/CS-Notes](https://github.com/CyC2018/CS-Notes)、[jackfrued/Python-100-Days](https://github.com/jackfrued/Python-100-Days)、[python/cpython](https://github.com/python/cpython)（`Lib`、`Objects`、`Doc`）。只在本地做测量，不进仓库，也不用于训练
- CS336 作业 4 仓库：<https://github.com/stanford-cs336/assignment4-data>

**下一章**：数据和分词器都有了。下一步是把几千亿个 token 真正喂进 8 张 GPU。第 14 章讲预训练工程：混合精度、FlashAttention、数据并行与 FSDP、MFU、loss spike 和断点续训。
