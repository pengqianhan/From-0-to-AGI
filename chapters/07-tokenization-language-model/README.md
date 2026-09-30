# 第 7 章：语言建模与分词 —— 从"猜下一个字"到 byte-level BPE

> **一句话目标**：读完这一章，你能手写一个 byte-level BPE 分词器，用它和数次数的 bigram 语言模型算出一段文本的 bits-per-byte，并且能解释为什么比较不同分词器的模型时要看 bpb，而不是看 loss 或困惑度。

📺 **本章视频**：待发布（本地渲染：`bash chapters/07-tokenization-language-model/video/build.sh`）
🧪 **本章自检**：学完后在 Claude Code 里输入 `/ch07-tokenization`

---

第一部分结束时，我们手里有了一整套工具：梯度下降、神经网络、自动微分、softmax 和交叉熵，还有让深网络训得稳的初始化、归一化、残差、AdamW。第 5 章最后已经说过，**语言模型就是一个在词表上做分类的分类器**：看前面的内容，给"下一个 token"的每个候选打分，softmax 变成概率，用交叉熵训练。

但第 5 章偷了个懒：它直接拿"字"当类别，词表是那一小段话里出现过的 65 个字。这一章要解决的问题是：**真实文本要怎么切成 token？** 按字？按字节？还是按词？切法决定了词表有多大、序列有多长、模型有没有"不认识的字"，甚至决定了 loss 这个数还能不能拿来跨模型比较。

这一章我们先把语言建模这件事用概率写清楚，然后亲手写一个 byte-level BPE 分词器——GPT-2、Llama 3、Qwen、DeepSeek 的分词器都是这一类——再用一个只看前 1 个 token 的 bigram 模型，把"评估一个语言模型"的几个数（nats、bits、困惑度、bits-per-byte）全部算一遍。

## 1. 语言建模：给整段文本一个概率

**语言模型（language model）** 要回答的问题是：一段文本 `x_1, x_2, …, x_T` 出现的概率有多大？

直接给"整段话"一个概率是做不到的——可能的句子无穷多。但概率论里有一条**链式法则（chain rule）**，可以把它拆开：

```
p(x_1, x_2, …, x_T) = p(x_1) · p(x_2 | x_1) · p(x_3 | x_1, x_2) · … · p(x_T | x_1, …, x_{T−1})
                    = Π_t p(x_t | x_<t)
```

这不是近似，是恒等式。它把"给整段话打分"变成了 T 个"已知前文，猜下一个 token"的小问题。每个小问题都是第 5 章的分类：类别是词表里的 V 个 token。

取对数，连乘变成求和。训练时我们最小化平均负对数似然，也就是**每个位置上正确的下一个 token 的交叉熵**：

```
L = −(1/T) · Σ_t log p(x_t | x_<t)
```

这就是所有 GPT 类模型的训练目标，叫 **next-token prediction**。从第 8 章的注意力到第 14 章的预训练，模型越来越复杂，但损失始终是这一行。

模型之间的区别在于 `p(x_t | x_<t)` 怎么算。本章用最简单的一种：**只看前 1 个 token**，

```
p(x_t | x_<t) ≈ p(x_t | x_{t−1})
```

这叫 **bigram 模型**。它的能力很有限，但足够让我们把分词和评估讲清楚。

在这之前，先解决更基本的问题：`x_t` 到底是什么？

## 2. 按字符切：简单，但中文词表很大

模型只认识整数。最直接的办法是**字符级（character-level）**：每个不同的字符分配一个 id。第 5 章就是这么做的。

```bash
uv run python chapters/07-tokenization-language-model/code/01_chars_and_bytes.py
```

在本课的极小语料（`assets/tiny_corpus/`）上统计：

| 语料 | 字符数 | UTF-8 字节数 | 字符级词表 | 字节 / 字符 |
|---|---:|---:|---:|---:|
| 英文（莎士比亚） | 1,115,394 | 1,115,394 | 65 | 1.00 |
| 中文（诗词） | 428,876 | 1,200,167 | 5,297 | 2.80 |
| 代码 | 139,005 | 159,673 | 870 | 1.15 |

英文只要 65 个字符就够了（大小写字母、标点、空白）。中文完全不同：一百多万字节的诗词里就有 5,297 个不同的字，而 Unicode 里的汉字有好几万个。字符级有两个麻烦：

1. **词表会很大、而且没有上限**。中英文加代码加 emoji，字符级词表轻松过万，而且永远有下一个没见过的字。
2. **没见过的字怎么办？** 把中文语料前 90% 当训练集，后 10% 里就有 198 个字符（129 种）是训练集里没出现过的，比如"亘""企"。字符级模型只能把它们统统映射成一个 `<unk>`（未知）符号——信息就这么丢了。

## 3. 按字节切：词表固定 256，永远不会遇到"不认识"

计算机存文本时，用的是 **UTF-8** 编码：每个字符变成 1 到 4 个字节（byte，0–255 之间的整数）。

| 字符 | 码位 | UTF-8 字节 |
|---|---|---|
| `A` | U+0041 | `[65]`（1 个字节） |
| `é` | U+00E9 | `[195, 169]`（2 个） |
| `学` | U+5B66 | `[229, 173, 166]`（3 个） |
| `🤖` | U+1F916 | `[240, 159, 164, 150]`（4 个） |

ASCII 字符（英文字母、数字、常用标点）是 1 个字节，**常用汉字是 3 个字节**。于是有了第二种切法——**字节级（byte-level）**：直接拿 UTF-8 字节当 token，词表固定就是 256 个。

```python
def byte_ids(text: str) -> list[int]:
    return list(text.encode("utf-8"))     # 每个字节就是 id（0–255），不需要任何词表
```

"学而时习之"字符级是 5 个 id，字节级是 15 个：`[229, 173, 166, 232, 128, 140, …]`。好处是任何文本——生僻字、emoji、乱码、二进制——都能编码，**永远不会出现 `<unk>`**，而且 `bytes(ids).decode("utf-8")` 能无损还原。坏处也很明显：序列变长了。中文是字符级的 2.8 倍长。第 8 章会看到，注意力的计算量随序列长度平方增长，序列长就是贵。

字符级词表大、有 `<unk>`；字节级词表小、序列长。有没有两全其美的办法？

## 4. Byte-level BPE：把最常见的一对合并成一个新 token

**BPE（Byte Pair Encoding，字节对编码）** 原本是一种数据压缩算法，Sennrich 等人 2015 年把它用到了机器翻译的分词上，GPT-2（2019）把它改成在 UTF-8 字节上运行，这就是 **byte-level BPE**。它的训练过程只有一句话：

> 从 256 个字节出发，**反复把语料里最常见的相邻两个 token 合并成一个新 token**，直到词表达到想要的大小。

比如英文里 `t` 后面经常跟 `h`，就把 `(t, h)` 合并成新 token `th`（id 256）；之后 `th` 后面又经常跟 `e`，再合并成 `the`……每合并一次，词表加 1，语料里的 token 数就减少一些。

最终的分词器有两个性质：

- 初始词表里就有全部 256 个字节，所以**任何文本都能编码**，不会有 `<unk>`（字节级的好处保住了）；
- 常见的片段（常用词、常用汉字、代码关键字）变成一个 token，**序列变短**（字符级的好处也有了）。

### 4.1 手写 BPE

完整代码在 [`code/02_bpe.py`](code/02_bpe.py)，训练部分的核心是这个循环：

```python
words = Counter(pretokenize(text))                   # 相同词块只存一份，记次数
seqs = [list(w.encode("utf-8")) for w in words]      # 每个词块 → 字节序列
...
for new_id in range(256, vocab_size):
    pair = max(stats, key=stats.get)                 # 最常见的相邻对
    self.merges[pair] = new_id                       # 记住这次合并
    self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]   # 新 token 的字节 = 两者拼接
    for i in where.pop(pair):                        # 只更新包含这一对的词块
        ... seqs[i] = merge(seqs[i], pair, new_id)   # 把所有 (a, b) 替换成 new_id，并更新相邻对计数
```

`stats` 是"相邻对 → 出现次数"，`where` 记录每一对出现在哪些词块里，这样每次合并只需要更新受影响的那部分，不用把整个语料重新数一遍。

编码新文本时，**按训练时学到的顺序重放合并**：在当前序列的所有相邻对里，找最早学到的那个合并，先做它，直到没有可合并的对。解码更简单：把每个 id 对应的字节拼起来，再按 UTF-8 解码。

```bash
uv run python chapters/07-tokenization-language-model/code/02_bpe.py
```

训练文本是英文、中文、代码各取开头 60,000 个字符，共 305,302 字节。做 768 次合并（词表 256 → 1024），纯 Python 用时约 4 秒。**最先学到的 20 个合并**（`␣` 表示空格，`\xef\xbc` 是没法单独显示的字节片段）：

| id | 新 token | 次数 | id | 新 token | 次数 |
|---:|---|---:|---:|---|---:|
| 256 | `\xef\xbc` | 7,212 | 266 | `\xe5\x85` | 1,703 |
| 257 | `␣␣` | 6,000 | 267 | `\xe4\xba` | 1,693 |
| 258 | `，` | 4,985 | 268 | `\xe2\x80` | 1,635 |
| 259 | `\xe3\x80` | 4,074 | 269 | `␣t` | 1,595 |
| 260 | `。` | 3,859 | 270 | `子` | 1,429 |
| 261 | `\xe4\xb9` | 2,931 | 271 | `he` | 1,294 |
| 262 | `\xe4\xb8` | 2,901 | 272 | `不` | 1,228 |
| 263 | `\xe5\xad` | 1,871 | 273 | `in` | 1,200 |
| 264 | `␣␣␣␣` | 1,853 | 274 | `\xe4\xbb` | 1,193 |
| 265 | `之` | 1,720 | 275 | `\xe6\x9c` | 1,161 |

这张表很值得多看几眼：

- **第一个合并是 `\xef\xbc`**。它不是一个完整的字，而是全角标点（`，`是 `ef bc 8c`，`：` 是 `ef bc 9a`……）共同的前两个字节。紧接着 258 号就把它和 `\x8c` 合成了完整的 `，`。
- **大量合并是汉字的"半个字"**：`\xe4\xb9`、`\xe4\xb8` 这样的两字节前缀被很多常用汉字共享，所以先被合并；然后才轮到"之""子""不"这样的高频整字。
- **代码里的缩进**（`␣␣`、`␣␣␣␣`）很早就被合并，英文的 `␣t`、`he`、`in` 也名列前茅。
- 后面学到的较长 token 有 `␣return`、`␣import`、`␣Citizen`、`MENENIUS`（莎士比亚剧本里的人名），以及一长串 `----------------`。**BPE 学到什么，完全取决于训练语料里什么多。**

编码的例子：

| 文本 | 字节 | token 数 | 切分结果 |
|---|---:|---:|---|
| `To be, or not to be` | 19 | 7 | `To` `␣be` `,` `␣or` `␣not` `␣to` `␣be` |
| `学而时习之，不亦说乎` | 30 | 12 | `学` `而` `时` `\xe4\xb9` `\xa0` `之` `，` `不` `亦` `\xe8\xaf` `\xb4` `乎` |
| `def forward(self, x):` | 21 | 11 | `def` `␣for` `w` `ard` `(` `self` `,` `␣` `x` `)` `:` |

"习"和"说"在 1024 的小词表里还没有被合并成整字，所以各被切成了两个 token（一个两字节前缀 + 一个字节）。这正是 byte-level BPE 的兜底机制：**没学到的字就退回到字节**，不会丢信息。

### 4.2 预切分：合并不跨越"词"的边界

如果直接在整段文本上跑 BPE，会学到 `e␣t`、`.␣The` 这种跨越单词和标点的 token，很浪费。所以 GPT-2 以来的分词器都先用一个正则表达式做**预切分（pre-tokenization）**，把文本切成字母串、数字、标点串、空白这样的"词块"，BPE 只在词块内部合并：

```python
PATTERN = re.compile(r"'(?:s|t|re|ve|m|ll|d)| ?[^\W\d_]+|\d| ?(?:[^\s\w]|_)+|\s+(?!\S)|\s+")
```

`"To be, or not to be? 学而时习之。x = 2026\n"` 会被切成 `'To' ' be' ',' ' or' ' not' ' to' ' be' '?' ' 学而时习之' '。' 'x' ' =' ' ' '2' '0' '2' '6' '\n'`。注意两点：单词前面的空格和单词粘在一起（所以有 `␣be` 这种 token）；**数字被逐个切开**，"2026"永远是 4 个 token。数字逐个切是 Qwen 的做法，好处是每个数字的表示方式一致，对算术更友好；Llama 3 和 DeepSeek-V3 则是最多 3 位一组（见文末"采用方与来源"）。

### 4.3 验证集上的压缩率

在每份语料结尾 20,000 个字符（训练时没见过）上，1024 词表的手写 BPE：

| 语料 | 字节数 | token 数 | 字节 / token |
|---|---:|---:|---:|
| 英文 | 20,000 | 11,353 | 1.76 |
| 中文 | 55,928 | 32,209 | 1.74 |
| 代码 | 21,946 | 11,236 | 1.95 |

**字节 / token**（bytes per token）是衡量分词器压缩率的常用指标：越大，同一段文本需要的 token 越少。纯字节级是 1.00，这里只做了 768 次合并就到了 1.7–2.0。词表再大会怎样？

## 5. 词表大小：序列长度和 embedding 参数的取舍

[`code/04_vocab_and_zero.py`](code/04_vocab_and_zero.py) 用生产级的训练器（`zero/tokenizer.py`，下文细讲）在整份语料上扫一遍词表大小，看验证集上的字节 / token：

```bash
uv run python chapters/07-tokenization-language-model/code/04_vocab_and_zero.py
```

| 词表 V | 英文 | 中文 | 代码 | 验证集总 token 数 |
|---:|---:|---:|---:|---:|
| 272（不合并） | 1.00 | 1.00 | 1.00 | 97,874 |
| 512 | 1.70 | 1.53 | 1.44 | 63,625 |
| 1,024 | 2.05 | 2.02 | 1.72 | 50,102 |
| 2,048 | 2.40 | 2.47 | 1.98 | 42,048 |
| 4,096 | 2.74 | 2.93 | 2.37 | 35,615 |
| 8,192 | 3.01 | 3.35 | 2.82 | 31,116 |
| 16,384 | 3.24 | 3.78 | 3.20 | 27,840 |
| 32,768 | 3.36 | 4.09 | 3.54 | 25,823 |

（272 = 256 个字节 + 16 个特殊 token。）词表从 272 涨到 1,024，序列长度几乎砍掉一半；再往后每翻一倍，收益越来越小：从 16K 到 32K，总 token 数只少了 7%。**收益递减**是词表大小的第一个规律。

代价在哪？词表里每个 token 都要一个 d 维的向量，**embedding 矩阵的参数量是 V × d**（输入输出共享 embedding 时只算一份，第 9 章会讲）。按各模型 `config.json` 里的数字算：

| 模型 | 词表 V | 宽度 d | embedding 参数 V × d |
|---|---:|---:|---:|
| GPT-2（124M） | 50,257 | 768 | 38.6M |
| Qwen3-0.6B | 151,936 | 1,024 | 155.6M |
| **Qwen3.5-0.8B** | **248,320** | **1,024** | **254.3M** |
| Llama 3.2 1B | 128,256 | 2,048 | 262.7M |
| 本课主线（暂定） | 65,536 | 1,280 | 83.9M |

Qwen3.5-0.8B 光 embedding 就约 2.5 亿参数，对一个标称 0.8B 的模型来说是很大的一块。对大模型来说这点参数不算什么，词表大能省下大量序列长度，很划算；**对小模型来说，embedding 挤占的是本可以放进 Transformer 层的参数**。所以本课主线模型的词表暂定为 65,536，比 Qwen 小得多——这是暂定值，最终由第 13 章在真实的中英代码数据上实测压缩率后决定。

还有两个不那么显眼的代价：词表越大，每个 token 在训练数据里出现的次数越少，罕见 token 的 embedding 训练不充分；输出层的 softmax 也要在 V 个类别上算，V 越大越慢。

## 6. Bigram 语言模型：数一数就是最大似然

有了分词器，终于可以训练语言模型了。bigram 模型的参数是一张 V × V 的表：第 a 行第 b 列表示"当前是 a，下一个是 b"的概率。第 5 章已经验证过，**对 bigram 来说，最大似然的解就是直接数频率**，不需要梯度下降：

```
p(b | a) = count(a, b) / Σ_j count(a, j)
```

完整代码在 [`code/03_bigram.py`](code/03_bigram.py)。数次数只要一行：

```python
def count_bigrams(ids, V):
    a = np.asarray(ids)
    flat = a[:-1] * V + a[1:]                                 # 把 (前一个, 下一个) 编成一个整数
    return np.bincount(flat, minlength=V * V).reshape(V, V)   # counts[前, 后]
```

直接数频率有一个问题：验证集里只要出现一次训练时没见过的组合，概率就是 0，`−log 0` 是无穷大。所以要**平滑（smoothing）**：每个格子都加一个小数 α，

```python
def bigram_prob(counts, prev, nxt, alpha=0.03):
    V = counts.shape[0]
    return (counts[prev, nxt] + alpha) / (counts.sum(axis=1)[prev] + alpha * V)
```

α = 0.03 是在一段既不参与训练、也不参与验证的文本（每份语料倒数第 40,000 到第 20,000 个字符）上，从 1、0.3、0.1、0.03、0.01 里挑的。

从 bigram 里生成文本，就是反复"按当前 token 那一行的概率抽下一个"。训练文本是每份语料除去最后 40,000 个字符（抽样时不加平滑）：

| 语料 · 分词 | 抽样结果（从换行开始） |
|---|---|
| 英文 · 字符 | `In weatee ther?⏎⏎FLES:⏎rure by arort me pre ther t` |
| 英文 · BPE | `MINIUS: I arself.⏎⏎With say.⏎⏎IDUMIRIARYous, all` |
| 中文 · 字符 | `晏殊⏎雪可得见好容。⏎有客长城子⏎张先⏎糜鹿鸣，鞗革有多媚，明。⏎翠绾鬓。⏎报。⏎见桃李藉栽双语。` |
| 中文 · BPE | `凭�。⏎云徊⏎月千花如之终共。⏎拾辱。⏎�` |

它学会了一些局部的东西：英文的"名字 + 冒号 + 换行"剧本格式、中文"作者名单独一行，然后是诗句"的格式；但整体毫无意义，因为**它只看前 1 个 token**。中文 BPE 的结果里还有 `�`：模型生成了一个汉字的前两个字节，下一步却接了一个不能构成合法 UTF-8 的字节。这是 byte-level 模型特有的现象，真实的大模型偶尔也会吐出这种乱码。

## 7. 评估：nats、bits、困惑度，和 bits-per-byte

评估语言模型，就是在没见过的验证文本上算平均交叉熵。第 5 章已经介绍过同一个量的三种写法：

- **nats / token**：用自然对数算的交叉熵 `−(1/T) Σ ln p(x_t | x_{t−1})`；
- **bits / token**：`nats / ln 2`，"平均每个 token 还要用多少个二进制位来描述"；
- **困惑度（perplexity）**：`e^nats = 2^bits`，"相当于在多少个候选里均匀犹豫"。

这三个数都是"每个 token"的。问题来了：**不同分词器的 token 长短不一**。BPE 的一个 token 平均覆盖 1.76 个字节，字节级的一个 token 只有 1 个字节；每个 BPE token 当然更难猜，loss 当然更高，但它一次猜的内容也更多。直接比较每 token 的 loss，就像比较"每步走多远"却不管步子大小。

解决办法是换一个分母：把总损失摊到原始文本的**每个字节**上。

```
bits-per-byte (bpb) = Σ_t (−ln p(x_t | …)) / (ln 2 × 这些 token 覆盖的总字节数)
                    = (bits / token) ÷ (bytes / token)
```

同一段验证文本的字节数是固定的，不管怎么切，所以 **bpb 与分词器无关**，可以横向比较。代码里对应这几行：

```python
nats = -np.log(bigram_prob(counts, x, y))                 # 每个位置的 −ln p(正确的下一个 token)
total_nats, total_bytes = nats.sum(), sum(n_bytes[1:])   # 只统计被预测的那些 token
ce = total_nats / len(y)                                  # nats / token
bits = ce / math.log(2)                                   # bits / token
ppl = math.exp(ce)                                        # 困惑度
bpb = total_nats / (math.log(2) * total_bytes)            # bits / byte
```

`03_bigram.py` 在三份语料的验证集上，用三种分词各训练一个 bigram（BPE 就是 4.1 节那个 1024 词表的分词器）：

| 语料 | 分词 | 词表 V | 验证 token 数 | nats / token | bits / token | 困惑度 | **bpb** |
|---|---|---:|---:|---:|---:|---:|---:|
| 英文 | 字节 | 256 | 19,999 | 2.543 | 3.669 | 12.7 | 3.669 |
| 英文 | 字符 | 66 | 19,999 | 2.543 | 3.668 | 12.7 | 3.668 |
| 英文 | BPE | 1,024 | 11,352 | 3.610 | 5.208 | 37.0 | **2.956** |
| 中文 | 字节 | 256 | 55,927 | 2.810 | 4.054 | 16.6 | 4.054 |
| 中文 | 字符 | 5,172 | 19,999 | 5.507 | 7.945 | 246.4 | **2.841** |
| 中文 | BPE | 1,024 | 32,208 | 3.809 | 5.495 | 45.1 | 3.164 |
| 代码 | 字节 | 256 | 21,945 | 2.567 | 3.704 | 13.0 | 3.704 |
| 代码 | 字符 | 815 | 19,999 | 2.683 | 3.871 | 14.6 | 3.528 |
| 代码 | BPE | 1,024 | 11,235 | 3.844 | 5.546 | 46.7 | **2.839** |

读这张表：

- **只看困惑度会得出完全相反的结论**。英文上字节级 bigram 的困惑度 12.7，BPE 是 37.0，好像字节级好得多；但换算成 bpb，BPE 是 2.956，字节级是 3.669——BPE 反而好 19%。原因是 BPE 的一个 token 覆盖 1.76 个字节，bigram 看"前 1 个 token"其实看到了更多的前文。
- **英文的字节级和字符级几乎一样**：莎士比亚几乎全是 ASCII，一个字符就是一个字节，两者是同一个模型。
- **中文上字符级最好（2.841）**。1024 的 BPE 词表对中文太小了，很多汉字还是被切成字节片段（4.3 节的"习""说"），bigram 的那"1 个 token"的上下文浪费在了半个字上。字符级有 5,172 个字、每个字一个 token，正好对上。不过字符级在这里占了一点便宜：验证集里 58 个没见过的字全记成同一个 `<unk>`，预测 `<unk>` 比预测具体是哪个生僻字容易。
- **所有 bpb 都远低于 8**：均匀乱猜一个字节要 8 bits。一个只看前 1 个 token 的计数表，已经把文本压到了 3 bits/byte 左右。

bpb 不是本章的玩具指标。Karpathy 的 nanochat 在预训练时定期评估的验证指标就是 `val_bpb`，它的 `evaluate_bpb` 函数做的正是上面这件事：累加每个目标 token 的损失和字节数，特殊 token 不计入，最后相除。本课主线模型到第 13 章比较不同词表大小时，也要看 bpb 而不是 loss。

## 8. 小结

- **语言建模**：链式法则把整段文本的概率拆成 `Π p(x_t | x_<t)`；训练目标是每个位置上下一个 token 的交叉熵，就是第 5 章在词表上的分类。
- **字符级**：词表大（中文 5,000+ 字）、有 `<unk>`；**字节级**：词表固定 256，没有 `<unk>`，但中文序列长 2.8 倍。
- **Byte-level BPE**：从 256 个字节出发，反复合并最常见的相邻对；预切分让合并不跨越词块；编码时按学到的顺序重放合并。
- **词表大小**：越大序列越短但收益递减，embedding 参数 V × d 线性增长；小模型尤其要算这笔账。
- **bigram**：只看前 1 个 token，数频率就是最大似然，要加平滑。
- **评估**：nats、bits、困惑度都是"每 token"的，跨分词器不可比；**bits-per-byte** 除以字节数，与分词器无关。

bigram 最大的问题写在它的定义里：**只看前 1 个 token**。"学而时习之，不亦说乎"里，猜"乎"的时候需要知道前面是"不亦说"，猜整句诗的格律需要记得前面好几句。把上下文从 1 个 token 扩到 2 个、3 个，表的大小就是 V³、V⁴，很快就存不下了。下一章的注意力，就是让模型能看到前面全部 token、又不需要一张天文数字大的表的办法。

---

## 从极简到生产级

极简版的每个想法，在主线模型的生产级代码里都有对应：

| 极简版 | 生产级（`zero/`） | 多做了什么、为什么 |
|---|---|---|
| `02_bpe.py` 的 `BPE.train`（纯 Python，305KB 语料 4 秒） | `zero/tokenizer.py` 的 `train_bpe(texts, vocab_size, special_tokens)`，基于 Hugging Face `tokenizers`（Rust） | 主线分词器要在 GB 级语料上训练，纯 Python 太慢；同样 768 次合并，zero 用时 0.74 秒，手写版 4.55 秒（本机实测，受机器负载影响） |
| `PATTERN`：Python `re` 的简化正则，用 `[^\W\d_]` 表示字母 | `PRETOKENIZE_REGEX`：与 Qwen2/Qwen3 的 `tokenizer.json` 完全相同的正则，用 `\p{L}`、`\p{N}` | Unicode 类别更精确；数字 `\p{N}` 逐个切；和 Qwen 保持一致，便于对照 |
| 无规范化 | `normalizers.NFC()` | 把"e + 组合重音符"这类写法合成一个码位，同一个字不会因为写法不同而被切成不同 token（与 Qwen3 一致） |
| 没有特殊 token | 16 个特殊 token 固定在 id 0–15：`<|endoftext|>`、`<|im_start|>`/`<|im_end|>`（对话）、`<tool_call>`/`</tool_call>`、`<tool_response>`/`</tool_response>`（工具调用）、`<think>`/`</think>`、7 个预留位 | 对话和工具调用格式（第 16 章）需要不会被 BPE 拆开的边界标记：zero 把 `<|im_start|>` 编成一个 id（1），手写版会把它拆成 8 个普通 token；预留位让以后加特殊 token 不必改词表大小 |
| 手算 `字节数 / token 数` | `Tokenizer.bytes_per_token(texts)` | 第 13 章用它在中英代码上比较不同词表大小 |
| `03_bigram.py` 里的 `bpb` 公式 | **zero 目前还没有**：`zero/train/trainer.py` 的验证只报告每 token 的 loss。计划按 nanochat `evaluate_bpb` 的写法补上（累加每个目标 token 的损失和字节数，特殊 token 不计） | 第 13 章比较不同词表大小时，loss 不可比，bpb 可比 |
| 在内存里编码一小段文本 | `zero/data/shard.py` 的 `write_shards`：分词后写成 `np.uint32` 分片，**每篇文档后面跟一个 `<|endoftext|>`**；元数据里记分词器哈希 `Tokenizer.hash()` | 预训练不能每步现场分词；`<|endoftext|>` 让模型学会"文档到此结束"；哈希防止拿 A 分词器切的数据去训 B 分词器的模型；uint32 为了词表可能超过 65,535 |
| 无 | `Tokenizer.save_hf(out_dir)` | 导出成 transformers 的 `AutoTokenizer` 能直接读的文件，发布时要用（第 20 章） |

**对拍**：同一份训练文本（三份语料各 60,000 字符）、同样 768 次合并，在验证集上的字节 / token：

| | 英文 | 中文 | 代码 |
|---|---:|---:|---:|
| 手写 BPE（`02_bpe.py`） | 1.76 | 1.74 | 1.95 |
| `zero/tokenizer.py` | 1.86 | 1.79 | 1.97 |

两者都能把验证集逐字节还原，数字切分完全一样（`x = 2026` → `x` `␣=` `␣` `2` `0` `2` `6`）。压缩率不完全相同是预期中的：正则不同（`\p{L}` vs `[^\W\d_]`）、次数相同时选哪一对的规则不同，都会让合并顺序略有差别。

生产级分词器的质量由 `tests/test_tokenizer.py` 保证（`uv run pytest tests/test_tokenizer.py`，本机 12 项全部通过）：中文、英文、代码、emoji、全角空格、首尾空白的**编码再解码还原原文**；特殊 token 的 id 固定且不被拆开；`"2026"` 编成 4 个 token；中英代码上 `bytes_per_token` 都大于 1.5、中文大于 2.0；存取后编码不变、哈希不变；导出后 transformers 的 `AutoTokenizer` 编码结果与 zero 完全一致。

---

## 采用方与来源

| 技术 | 采用方（主力版本） | 来源 |
|---|---|---|
| Byte-level BPE（256 字节打底，无 `<unk>`） | GPT-2；Llama 3（128K 词表：tiktoken 的 100K + 28K 非英语 token）；Qwen（Qwen2/Qwen3/Qwen3.5 的 `tokenizer.json`：`ByteLevel` 预切分 + `BPE` 模型）；DeepSeek-V3（128K，`ByteLevel` + `BPE`）；OLMo 2（`ByteLevel` + `BPE`） | GPT-2 论文第 2.2 节；Llama 3 论文；各模型 Hugging Face 仓库的 `tokenizer.json`（链接见参考文献） |
| 正则预切分 | GPT-2 起；Qwen3、Llama 3.2、DeepSeek-V3、OLMo 2 的 `tokenizer.json` 里都有 `Split` 正则 | 同上 |
| 数字切分 | Qwen3 / Qwen3.5：`\p{N}`（逐个）；Llama 3.2、DeepSeek-V3、OLMo 2：`\p{N}{1,3}`（最多 3 位一组） | 同上（2026-09 读取） |
| 词表大小 | GPT-2 50,257；Llama 3 系 128,256；Qwen3 151,936；Qwen3.5 248,320（`config.json` 的 `vocab_size`） | 各模型 `config.json` |
| bits-per-byte 作为跨分词器的指标 | nanochat（`val_bpb`，`nanochat/loss_eval.py` 的 `evaluate_bpb`） | <https://github.com/karpathy/nanochat> |

Gemma 系列用的是 SentencePiece 分词器，不在上表里；是否也属于"字节兜底的 BPE"请以其技术报告为准（待核实）。

---

## 引导问题

带着这些问题去问 Claude Code，直到你能用自己的话讲清楚：

1. 链式法则 `p(x_1…x_T) = Π p(x_t | x_<t)` 是恒等式，不是近似。那 bigram 的近似发生在哪一步？trigram（看前 2 个 token）的表有多大？V = 65,536 时存得下吗？
2. 为什么 BPE 学到的第一个合并是 `\xef\xbc` 而不是某个汉字？如果训练语料全是英文，第一个合并大概会是什么？试着改 `02_bpe.py` 验证。
3. "困惑度 37 的 BPE 模型比困惑度 12.7 的字节级模型更好"——用 bpb 的定义解释这句话。如果两个模型用的是同一个分词器，比较 loss 和比较 bpb 的结论会不会不同？
4. 本章的中文 bigram 里字符级（bpb 2.841）赢了 1024 词表的 BPE（3.164）。如果把 BPE 词表加到 8,192，你猜会怎样？为什么？
5. Qwen3.5-0.8B 的 embedding 约 2.5 亿参数。假设总参数预算固定在 0.8B，把词表从 248K 减到 65K，省下的参数可以做什么？代价又是什么？
6. 为什么预训练数据里每篇文档后面要加 `<|endoftext|>`？如果不加，模型在生成时会出现什么现象？

## 动手任务

每个任务都要真的运行代码、看到结果。

**任务 1（基础）**：在 `02_bpe.py` 里把 `vocab_size` 从 1024 改成 2048 和 4096，记录训练时间和验证集上三种语料的字节 / token，并检查"习""说"是否变成了一个 token。

**任务 2（核心）**：给 `03_bigram.py` 加一个 **unigram**（完全不看前文，`p(x_t) = count(x_t) / N`）基线，算出三种分词在三份语料上的 bpb，和 bigram 比较："看前 1 个 token"值多少 bits/byte？

**任务 3（挑战）**：把 `03_bigram.py` 里的 BPE 换成 `zero.tokenizer.train_bpe` 训练出的 8,192 词表分词器（在 `04_vocab_and_zero.py` 里可以找到写法），重新算三份语料的 bpb。词表变大后，bigram 的表有 8,192² ≈ 6,700 万格，而训练数据只有一百多万 token——你会看到什么？调一调平滑 α，解释你的发现。这个"数据不够填满表"的问题，就是下一章用神经网络（而不是计数表）来建模的原因之一。

---

## 想深入：CS336

本章对应斯坦福 CS336（Spring 2026）<https://cs336.stanford.edu/>：

- **第 1 讲：概览与分词**。从字符级、字节级、词级讲到 BPE，和本章同一条线，但更系统地讨论了各种分词方案的优缺点。
- **第 2 讲：PyTorch 与资源核算**。本章 5 节的"embedding 参数 = V × d"只是资源核算的一小部分；第 2 讲会教你把一个模型的参数、显存、FLOPs 全部算清楚（本课第 12 章会再用到）。
- **作业 1（Basics）的 BPE 部分**：从零实现 byte-level BPE 的训练与编解码，要求在 TinyStories 和 OpenWebText 上训练并控制时间和内存，比本章的 `02_bpe.py` 要求高得多（需要并行化预切分、增量更新计数）。本章代码可以作为热身。作业仓库：<https://github.com/stanford-cs336/assignment1-basics>

---

## 本章参考文献

- Sennrich, Haddow, Birch. *Neural Machine Translation of Rare Words with Subword Units*（BPE 用于分词），2015：<https://arxiv.org/abs/1508.07909>
- Radford et al. *Language Models are Unsupervised Multitask Learners*（GPT-2，byte-level BPE，第 2.2 节）：<https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf>
- Llama Team, Meta. *The Llama 3 Herd of Models*（128K 词表 = tiktoken 的 100K + 28K 非英语 token），2024：<https://arxiv.org/abs/2407.21783>
- Karpathy. minbpe（最小的 byte-level BPE 实现，含 GPT-4 分词器复现）：<https://github.com/karpathy/minbpe>
- Karpathy. nanochat（`nanochat/loss_eval.py` 的 `evaluate_bpb`）：<https://github.com/karpathy/nanochat>
- Hugging Face `tokenizers` 文档：<https://huggingface.co/docs/tokenizers>
- 各模型的分词器与配置文件（2026-09 读取）：
  - Qwen3-0.6B：<https://huggingface.co/Qwen/Qwen3-0.6B/blob/main/tokenizer.json>、<https://huggingface.co/Qwen/Qwen3-0.6B/blob/main/config.json>
  - Qwen3.5-0.8B：<https://huggingface.co/Qwen/Qwen3.5-0.8B/blob/main/config.json>、<https://huggingface.co/Qwen/Qwen3.5-0.8B/blob/main/tokenizer.json>
  - DeepSeek-V3：<https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/tokenizer.json>
  - Llama 3.2 1B（非官方镜像，官方仓库需申请访问）：<https://huggingface.co/unsloth/Llama-3.2-1B/blob/main/tokenizer.json>
  - OLMo 2：<https://huggingface.co/allenai/OLMo-2-1124-7B/blob/main/tokenizer.json>
  - GPT-2：<https://huggingface.co/openai-community/gpt2/blob/main/config.json>
- UTF-8 编码规则：<https://en.wikipedia.org/wiki/UTF-8>
- CS336 作业 1 仓库：<https://github.com/stanford-cs336/assignment1-basics>
- [nanoGPT](https://github.com/karpathy/nanoGPT)、[minimind](https://github.com/jingyaogong/minimind)：都从字符级 / BPE 分词开始讲语言模型

**下一章**：bigram 只能看前 1 个 token。要猜"不亦说乎"的"乎"，模型得回头看"不亦"；要写出一句押韵的诗，得记得前面好几句。第 8 章，我们从"对前面所有 token 做加权平均"出发，推导出注意力（Attention）——让每个位置自己决定该看前文的哪里。
