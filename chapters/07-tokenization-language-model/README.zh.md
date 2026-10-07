# 第 7 章：语言建模与分词 —— 从"预测下一个字"到 byte-level BPE

[English](README.md) · **中文**

> **目标**：读完这一章，你能手写一个 byte-level BPE 分词器。用这个分词器和一个靠计数得到的 bigram 语言模型，你能算出一段文本的 bits-per-byte。你还能解释：比较使用不同分词器的模型时，为什么要看 bpb，而不看损失或困惑度。

📺 **本章视频**：还没有发布。要在本机渲染，运行 `bash chapters/07-tokenization-language-model/video/build.sh`。
🧪 **本章自检**：学完后，在 Claude Code 里输入 `/ch07-tokenization`。

---

第一部分结束时，我们已经有了一整套工具：梯度下降、神经网络、自动微分、softmax 和交叉熵。我们还有让深层网络稳定训练的方法：初始化、归一化、残差连接和 AdamW。第 5 章最后讲过：**语言模型就是一个在词表上做分类的分类器**。它看前面的内容，给"下一个 token"的每个候选打分。softmax 把分数变成概率，交叉熵用来训练模型。

但第 5 章走了一条捷径。它直接把单个"字"当作类别，词表是一小段话里出现过的 65 个字。这一章要回答一个新问题：**真实文本要怎么切成 token？** 按字？按字节？还是按词？

切法决定了词表的大小和序列的长度。切法还决定模型会不会遇到"不认识的字"。切法甚至决定了能不能用损失来比较不同的模型。

这一章先用概率写清楚语言建模。然后我们手写一个 byte-level BPE 分词器（tokenizer）。GPT-2、Llama 3、Qwen、DeepSeek 的分词器都属于这一类。最后我们用一个 bigram 模型，它只看前 1 个 token。用它，我们把评测语言模型的几个数全部算一遍：nats、bits、困惑度（perplexity）和 bits-per-byte。

## 1. 语言建模：给整段文本一个概率

**语言模型**（language model）回答的问题是：一段文本 `x_1, x_2, …, x_T` 出现的概率有多大？

我们没法直接给整段文本一个概率，因为可能的句子有无穷多。但概率论里有一条**链式法则**（chain rule），可以把这个概率拆开：

```
p(x_1, x_2, …, x_T) = p(x_1) · p(x_2 | x_1) · p(x_3 | x_1, x_2) · … · p(x_T | x_1, …, x_{T−1})
                    = Π_t p(x_t | x_<t)
```

这不是近似，而是恒等式。它把"给整段文本打分"变成了 T 个小问题："已知前文，猜下一个 token"。每个小问题都是第 5 章的分类问题，类别是词表里的 V 个 token。

取对数后，连乘变成求和。训练时，我们最小化平均负对数似然（negative log-likelihood）。它就是**每个位置上正确的下一个 token 的交叉熵**：

```
L = −(1/T) · Σ_t log p(x_t | x_<t)
```

这就是所有 GPT 类模型的训练目标，叫作 **next-token prediction**（预测下一个 token）。从第 8 章的注意力到第 14 章的预训练，模型越来越复杂，但损失始终是这一行。

模型之间的区别在于怎么算 `p(x_t | x_<t)`。本章用最简单的一种：**只看前 1 个 token**：

```
p(x_t | x_<t) ≈ p(x_t | x_{t−1})
```

这叫 **bigram 模型**。它的能力很有限，但足够用来讲清楚分词和评测。

在这之前，先解决一个更基本的问题：`x_t` 到底是什么？

## 2. 按字符切：简单，但中文词表很大

模型只认识整数。最直接的办法是**字符级**（character-level）切分：给每个不同的字符分配一个 id。第 5 章就是这么做的。

```bash
uv run python chapters/07-tokenization-language-model/code/01_chars_and_bytes.py
```

在本课的极小语料（`assets/tiny_corpus/`）上统计，结果如下：

| 语料 | 字符数 | UTF-8 字节数 | 字符级词表 | 字节 / 字符 |
|---|---:|---:|---:|---:|
| 英文（莎士比亚） | 1,115,394 | 1,115,394 | 65 | 1.00 |
| 中文（诗词） | 428,876 | 1,200,167 | 5,297 | 2.80 |
| 代码 | 139,005 | 159,673 | 870 | 1.15 |

英文只要 65 个字符就够了（大小写字母、标点和空白）。中文完全不同。一百多万字节的诗词里就有 5,297 个不同的字，而 Unicode 里的汉字有好几万个。字符级有两个问题：

1. **词表很大，而且没有上限。** 中文、英文、代码再加上 emoji，字符级词表很容易超过一万。而且永远会有下一个没见过的字。
2. **没见过的字怎么办？** 把中文语料的前 90% 当训练集。后 10% 里有 198 个字符（129 种）在训练集里没有出现过，比如"亘""企"。字符级模型只能把它们全部映射成一个 `<unk>`（未知）符号。这些信息就丢了。

## 3. 按字节切：词表固定 256，永远不会遇到"不认识"

计算机存文本时，用的是 **UTF-8** 编码。UTF-8 把每个字符变成 1 到 4 个字节（byte）。一个字节是 0–255 之间的整数。

| 字符 | 码位 | UTF-8 字节 |
|---|---|---|
| `A` | U+0041 | `[65]`（1 个字节） |
| `é` | U+00E9 | `[195, 169]`（2 个） |
| `学` | U+5B66 | `[229, 173, 166]`（3 个） |
| `🤖` | U+1F916 | `[240, 159, 164, 150]`（4 个） |

ASCII 字符（英文字母、数字、常用标点）占 1 个字节，**常用汉字占 3 个字节**。原因是：UTF-8 对 U+0800 到 U+FFFF 之间的码位都用 3 个字节，常用汉字在这个范围里。于是有了第二种切法：**字节级**（byte-level）切分。它直接把 UTF-8 字节当作 token，词表固定是 256 个。

```python
def byte_ids(text: str) -> list[int]:
    return list(text.encode("utf-8"))     # each byte is the id (0–255); no vocabulary is necessary
```

"学而时习之"有 5 个字。它在字符级是 5 个 id，在字节级是 15 个 id（每个字 3 个字节）：`[229, 173, 166, 232, 128, 140, …]`。好处是任何文本都能编码，包括生僻字、emoji、乱码和二进制数据。**永远不会出现 `<unk>`**，而且 `bytes(ids).decode("utf-8")` 能无损还原原文。

字节级的坏处是序列变长了：中文的序列长度是字符级的 2.8 倍。第 8 章会讲到，注意力需要的算力随序列长度的平方增长。序列长，就贵。

字符级词表大，还有 `<unk>`；字节级词表小，但序列长。有没有一种办法能同时得到两者的好处？

## 4. Byte-level BPE：把最常见的一对合并成一个新 token

**BPE**（Byte Pair Encoding，字节对编码）原本是一种数据压缩算法。2015 年，Sennrich 等人把它用到机器翻译的分词上。GPT-2（2019）把它改成在 UTF-8 字节上运行，这就是 **byte-level BPE**。它的训练过程只有一句话：

> 从 256 个字节出发，**反复把语料里最常见的相邻两个 token 合并成一个新 token**，直到词表达到想要的大小。

比如英文里 `t` 后面经常跟着 `h`，就把 `(t, h)` 合并成新 token `th`（id 256）。之后 `th` 后面又经常跟着 `e`，就再合并成 `the`，依此类推。每合并一次，词表加 1，语料里的 token 数就减少一些。

最终的分词器有两个性质：

- 初始词表里有全部 256 个字节，所以**任何文本都能编码**，不会出现 `<unk>`。字节级的好处保住了。
- 常见的片段（常用词、常用汉字、代码关键字）变成一个 token，所以**序列变短**。字符级的好处也有了。

### 4.1 手写 BPE

完整代码在 [`code/02_bpe.py`](code/02_bpe.py)。训练部分的核心是这个循环：

```python
words = Counter(pretokenize(text))                   # keep one copy of each chunk, with its count
seqs = [list(w.encode("utf-8")) for w in words]      # each chunk → a byte sequence
...
for new_id in range(256, vocab_size):
    pair = max(stats, key=stats.get)                 # the most frequent adjacent pair
    self.merges[pair] = new_id                       # remember this merge
    self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]   # bytes of the new token = the two joined
    for i in where.pop(pair):                        # update only the chunks that contain this pair
        ... seqs[i] = merge(seqs[i], pair, new_id)   # replace each (a, b) with new_id, and update the pair counts
```

词块（chunk）是预切分得到的一段文本（见 4.2 节）。`stats` 记录"相邻对 → 出现次数"。`where` 记录每一对出现在哪些词块里。这样每次合并只需要更新受影响的部分，不用把整个语料重新数一遍。

编码新文本时，**按训练时学到的顺序重放合并**。在当前序列的所有相邻对里，找到最早学到的那个合并，先做它。一直做到没有可合并的对为止。解码更简单：把每个 id 对应的字节拼起来，再按 UTF-8 解码。

```bash
uv run python chapters/07-tokenization-language-model/code/02_bpe.py
```

训练文本是英文、中文、代码各取开头 60,000 个字符，共 305,302 字节。做 768 次合并（词表 256 → 1024），纯 Python 用时几秒（取决于机器）。下表是**最先学到的 20 个合并**。`␣` 表示空格，`\xef\xbc` 是没法单独显示成字符的字节片段。

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

这张表值得仔细看：

- **第一个合并是 `\xef\xbc`。** 它不是一个完整的字，而是全角标点共有的前两个字节：`，` 是 `ef bc 8c`，`：` 是 `ef bc 9a`，等等。紧接着，258 号把它和 `\x8c` 合成了完整的 `，`。
- **很多合并是汉字的"半个字"。** 很多常用汉字共享同样的两字节前缀，比如 `\xe4\xb9`、`\xe4\xb8`，所以 BPE 先合并这些前缀。之后才轮到"之""子""不"这样的高频整字。
- **代码里的缩进**（`␣␣`、`␣␣␣␣`）很早就被合并。英文的 `␣t`、`he`、`in` 也排在前面。
- 后面学到的较长 token 有 `␣return`、`␣import`、`␣Citizen`、`MENENIUS`（莎士比亚剧本里的人名），还有一长串 `----------------`。**BPE 学到什么，完全取决于训练语料里什么多。**

编码的例子：

| 文本 | 字节 | token 数 | 切分结果 |
|---|---:|---:|---|
| `To be, or not to be` | 19 | 7 | `To` `␣be` `,` `␣or` `␣not` `␣to` `␣be` |
| `学而时习之，不亦说乎` | 30 | 12 | `学` `而` `时` `\xe4\xb9` `\xa0` `之` `，` `不` `亦` `\xe8\xaf` `\xb4` `乎` |
| `def forward(self, x):` | 21 | 11 | `def` `␣for` `w` `ard` `(` `self` `,` `␣` `x` `)` `:` |

在 1024 的小词表里，"习"和"说"还没有被合并成整字。所以它们各被切成两个 token：一个两字节前缀加一个字节。这就是 byte-level BPE 的兜底机制：**没学到的字就退回到字节**，不会丢信息。

### 4.2 预切分：合并不跨越"词"的边界

如果直接在整段文本上运行 BPE，它会学到 `e␣t`、`.␣The` 这样跨越单词和标点的 token。这些 token 浪费词表的位置。所以 GPT-2 以来的分词器都先用一个正则表达式做**预切分**（pre-tokenization）。正则把文本切成字母串、数字、标点串、空白这样的"词块"，BPE 只在词块内部合并：

```python
PATTERN = re.compile(r"'(?:s|t|re|ve|m|ll|d)| ?[^\W\d_]+|\d| ?(?:[^\s\w]|_)+|\s+(?!\S)|\s+")
```

正则把 `"To be, or not to be? 学而时习之。x = 2026\n"` 切成 `'To' ' be' ',' ' or' ' not' ' to' ' be' '?' ' 学而时习之' '。' 'x' ' =' ' ' '2' '0' '2' '6' '\n'`。注意两点。第一，单词前面的空格和单词粘在一起（所以有 `␣be` 这种 token）。正则把汉字也算作字母，中文词之间又没有空格，所以 `␣学而时习之` 是一个词块。

第二，**数字被逐个切开**，"2026"永远是 4 个 token。逐个切数字是 Qwen 的做法，好处是每个数字的表示方式都一致，对算术更友好。Llama 3 和 DeepSeek-V3 则是最多 3 位一组（见本页末尾的"采用方与来源"）。

### 4.3 验证集上的压缩率

我们在每份语料结尾的 20,000 个字符上测试 1024 词表的手写 BPE。训练时没有见过这些字符。

| 语料 | 字节数 | token 数 | 字节 / token |
|---|---:|---:|---:|
| 英文 | 20,000 | 11,353 | 1.76 |
| 中文 | 55,928 | 32,209 | 1.74 |
| 代码 | 21,946 | 11,236 | 1.95 |

**字节 / token**（bytes per token）是衡量分词器压缩率的常用指标。数值越大，同一段文本需要的 token 越少。纯字节级是 1.00，这里只做了 768 次合并就到了 1.7–2.0。词表再大会怎样？

## 5. 词表大小：序列长度和 embedding 参数的取舍

[`code/04_vocab_and_zero.py`](code/04_vocab_and_zero.py) 用生产级的训练器（`zero/tokenizer.py`，下文细讲）。它在整份语料上扫一遍不同的词表大小，然后测量验证集上的字节 / token：

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

（272 = 256 个字节 + 16 个特殊 token。）词表从 272 增加到 1,024，序列长度几乎减少一半。之后词表每翻一倍，收益越来越小：从 16K 到 32K，总 token 数只少了 7%。**收益递减**是词表大小的第一个规律。

代价在哪里？词表里每个 token 都需要一个 d 维向量，所以 **embedding 矩阵的参数量是 V × d**。（输入和输出共享 embedding 时只算一份，第 9 章会讲。）按各模型 `config.json` 里的数字计算：

| 模型 | 词表 V | 宽度 d | embedding 参数 V × d |
|---|---:|---:|---:|
| GPT-2（124M） | 50,257 | 768 | 38.6M |
| Qwen3-0.6B | 151,936 | 1,024 | 155.6M |
| **Qwen3.5-0.8B** | **248,320** | **1,024** | **254.3M** |
| Llama 3.2 1B | 128,256 | 2,048 | 262.7M |
| 本课主线模型（暂定） | 65,536 | 1,280 | 83.9M |

Qwen3.5-0.8B 仅 embedding 就有约 2.5 亿参数，对一个标称 0.8B 的模型来说是很大的一块。对大模型来说，这些参数不算什么。词表大能省下大量序列长度，很划算。

**对小模型来说，embedding 占用的是本可以放进 Transformer 层的参数。** 所以本课主线模型的词表暂定为 65,536，比 Qwen 小得多。这是暂定值。第 13 章会在真实的中英代码数据上实测压缩率，再决定最终的值。

还有两个不那么显眼的代价。第一，词表越大，每个 token 在训练数据里出现的次数越少，罕见 token 的 embedding 训练不充分。第二，输出层的 softmax 要在 V 个类别上计算，V 越大越慢。

## 6. Bigram 语言模型：计数就是最大似然

有了分词器，就可以训练语言模型了。bigram 模型的参数是一张 V × V 的表：第 a 行第 b 列是"当前 token 是 a 时，下一个 token 是 b"的概率。第 5 章已经验证过：**对 bigram 来说，最大似然（maximum likelihood）的解就是频率**，不需要梯度下降：

```
p(b | a) = count(a, b) / Σ_j count(a, j)
```

完整代码在 [`code/03_bigram.py`](code/03_bigram.py)。计数只要一行：

```python
def count_bigrams(ids, V):
    a = np.asarray(ids)
    flat = a[:-1] * V + a[1:]                                 # encode (previous, next) as one integer
    return np.bincount(flat, minlength=V * V).reshape(V, V)   # counts[previous, next]
```

直接用频率有一个问题。验证集里训练时没见过的组合，概率是 0，而 `−log 0` 是无穷大。这样的组合只要出现一次就够了。所以要做**平滑**（smoothing）：给每个格子都加一个小数 α。

```python
def bigram_prob(counts, prev, nxt, alpha=0.03):
    V = counts.shape[0]
    return (counts[prev, nxt] + alpha) / (counts.sum(axis=1)[prev] + alpha * V)
```

α = 0.03 是从 1、0.3、0.1、0.03、0.01 里挑出来的。挑选用的是一段既不参与训练、也不参与验证的文本：每份语料倒数第 40,000 到倒数第 20,000 个字符。

用 bigram 生成文本，就是反复按当前 token 那一行的概率抽取下一个 token。训练文本是每份语料去掉最后 40,000 个字符。抽样时不加平滑。表中的 `⏎` 表示换行。

| 语料 · 分词 | 抽样结果（从换行开始） |
|---|---|
| 英文 · 字符 | `In weatee ther?⏎⏎FLES:⏎rure by arort me pre ther t` |
| 英文 · BPE | `MINIUS: I arself.⏎⏎With say.⏎⏎IDUMIRIARYous, all` |
| 中文 · 字符 | `晏殊⏎雪可得见好容。⏎有客长城子⏎张先⏎糜鹿鸣，鞗革有多媚，明。⏎翠绾鬓。⏎报。⏎见桃李藉栽双语。` |
| 中文 · BPE | `凭�。⏎云徊⏎月千花如之终共。⏎拾辱。⏎�` |

模型学会了一些局部的东西。英文学会了剧本的"人名 + 冒号 + 换行"格式。中文学会了"作者名单独一行，然后是诗句"的格式。（"晏殊""张先"都是作者名。）但整体没有意义，因为**它只看前 1 个 token**。

中文 BPE 的结果里还有 `�`。模型生成了一个汉字的前两个字节，下一步却接了一个不能构成合法 UTF-8 的字节。这是 byte-level 模型特有的现象。真实的大模型偶尔也会输出这种乱码。

## 7. 评测：nats、bits、困惑度和 bits-per-byte

评测（evaluation）语言模型，就是在没见过的验证文本上计算平均交叉熵。第 5 章已经介绍过同一个量的三种写法：

- **nats / token**：用自然对数计算的交叉熵 `−(1/T) Σ ln p(x_t | x_{t−1})`。
- **bits / token**：`nats / ln 2`，即"平均每个 token 还需要多少个二进制位来描述"。
- **困惑度**：`e^nats = 2^bits`，即"相当于在多少个候选之间均匀犹豫"。

这三个数都是"每个 token"的。问题是：**不同分词器的 token 长短不一**。BPE 的一个 token 平均覆盖 1.76 个字节，字节级的一个 token 只有 1 个字节。BPE 的每个 token 更难猜，所以损失更高；但它一次猜的内容也更多。直接比较每 token 的损失，就像比较两个人每一步有多费力，却不管两人的步子大小不同。

解决办法是换一个分母：把总损失摊到原始文本的**每个字节**上。

```
bits-per-byte (bpb) = Σ_t (−ln p(x_t | …)) / (ln 2 × total bytes that these tokens cover)
                    = (bits / token) ÷ (bytes / token)
```

不管怎么切，同一段验证文本的字节数都是固定的。所以 **bpb 与分词器无关**，可以横向比较。代码里对应这几行：

```python
nats = -np.log(bigram_prob(counts, x, y))                 # −ln p(correct next token) at each position
total_nats, total_bytes = nats.sum(), sum(n_bytes[1:])   # count only the tokens that the model predicts
ce = total_nats / len(y)                                  # nats / token
bits = ce / math.log(2)                                   # bits / token
ppl = math.exp(ce)                                        # perplexity
bpb = total_nats / (math.log(2) * total_bytes)            # bits / byte
```

`03_bigram.py` 在三份语料上，用三种分词各训练一个 bigram，然后在验证集上评测每个模型。（BPE 就是 4.1 节那个 1024 词表的分词器。）

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

- **只看困惑度，会得出完全相反的结论。** 英文上，字节级 bigram 的困惑度是 12.7，BPE 是 37.0，好像字节级好得多。但换算成 bpb，BPE 是 2.956，字节级是 3.669——BPE 反而好 19%。原因是 BPE 的一个 token 覆盖 1.76 个字节，bigram 看的"前 1 个 token"包含了更多前文。
- **英文的字节级和字符级几乎一样。** 莎士比亚的文本几乎全是 ASCII，一个字符就是一个字节，两者是同一个模型。
- **中文上字符级最好（2.841）。** 1024 的 BPE 词表对中文太小，很多汉字还是被切成字节片段（4.1 节的"习""说"）。于是 bigram 那"1 个 token"的上下文浪费在了半个字上。字符级有 5,172 个字，每个字一个 token，正好对上。不过字符级在这里占了一点便宜：验证集里 58 个没见过的字全记成同一个 `<unk>`。预测 `<unk>` 比预测具体是哪个生僻字容易。
- **所有 bpb 都远低于 8。** 均匀乱猜一个字节需要 8 bits。一个只看前 1 个 token 的计数表，已经把文本压到了 3 bits/byte 左右。

bpb 不只是本章的玩具指标。Karpathy 的 nanochat 在预训练时定期评测的验证指标就是 `val_bpb`。它的 `evaluate_bpb` 函数做的正是上面这件事：累加每个目标 token 的损失和字节数，特殊 token 不计入，最后相除。到第 13 章，本课主线模型比较不同词表大小时，也要看 bpb，而不是损失。

## 8. 小结

- **语言建模**：链式法则把整段文本的概率拆成 `Π p(x_t | x_<t)`。训练目标是每个位置上下一个 token 的交叉熵，也就是第 5 章在词表上的分类。
- **字符级**：词表大（中文 5,000+ 字），有 `<unk>`。**字节级**：词表固定 256，没有 `<unk>`，但中文序列长 2.8 倍。
- **Byte-level BPE**：从 256 个字节出发，反复合并最常见的相邻对。预切分让合并不跨越词块。编码时按学到的顺序重放合并。
- **词表大小**：词表越大，序列越短，但收益递减；embedding 参数 V × d 线性增长。小模型尤其要算这笔账。
- **bigram**：只看前 1 个 token。频率就是最大似然的解，但要加平滑。
- **评测**：nats、bits、困惑度都是"每 token"的，不能跨分词器比较。**bits-per-byte** 除以字节数，与分词器无关。

bigram 最大的问题写在它的定义里：**只看前 1 个 token**。再看"学而时习之，不亦说乎"这一句。猜"乎"的时候，需要知道前面是"不亦说"；猜整首诗的格律，需要记得前面好几句。把上下文从 1 个 token 扩到 2 个、3 个，表的大小就是 V³、V⁴，很快就存不下了。下一章的注意力能让模型看到前面全部 token，又不需要一张巨大的表。

---

## 从极简代码到生产级代码

极简代码里的每个想法，在主线模型的生产级代码里都有对应：

| 极简代码 | 生产级代码（`zero/`） | 多做了什么，为什么 |
|---|---|---|
| `02_bpe.py` 的 `BPE.train`（纯 Python，305KB 语料用几秒） | `zero/tokenizer.py` 的 `train_bpe(texts, vocab_size, special_tokens)`，基于 Hugging Face `tokenizers`（Rust） | 主线分词器要在 GB 级语料上训练，纯 Python 太慢。同样做 768 次合并，一次实测 zero 用时 0.74 秒，手写版 4.55 秒；2026-10 换一台服务器复跑，是 1.08 秒对 1.62 秒。语料这么小时，倍数随机器和负载变化很大 |
| `PATTERN`：Python `re` 的简化正则，用 `[^\W\d_]` 表示字母 | `PRETOKENIZE_REGEX`：与 Qwen2/Qwen3 的 `tokenizer.json` 完全相同的正则，用 `\p{L}`、`\p{N}` | Unicode 类别更精确；数字 `\p{N}` 逐个切；和 Qwen 保持一致，便于对照 |
| 无规范化 | `normalizers.NFC()` | 把"e + 组合重音符"这类写法合成一个码位。这样同一个字不会因为写法不同而被切成不同的 token（与 Qwen3 一致） |
| 没有特殊 token | 16 个特殊 token 固定在 id 0–15：`<|endoftext|>`、`<|im_start|>`/`<|im_end|>`（对话）、`<tool_call>`/`</tool_call>`、`<tool_response>`/`</tool_response>`（工具调用）、`<think>`/`</think>`、7 个预留位 | 对话和工具调用格式（第 16 章）需要 BPE 不会拆开的边界标记。zero 把 `<|im_start|>` 编成一个 id（1），手写版会把它拆成 8 个普通 token。有了预留位，以后添加特殊 token 时不必改词表大小 |
| 手算 `字节数 / token 数` | `Tokenizer.bytes_per_token(texts)` | 第 13 章用它在中英代码上比较不同的词表大小 |
| `03_bigram.py` 里的 `bpb` 公式 | **zero 目前还没有**：`zero/train/trainer.py` 的验证只报告每 token 的损失。计划按 nanochat `evaluate_bpb` 的写法补上（累加每个目标 token 的损失和字节数，特殊 token 不计入） | 第 13 章比较不同词表大小时，损失不可比，bpb 可比 |
| 在内存里编码一小段文本 | `zero/data/shard.py` 的 `write_shards`：分词后写成 `np.uint32` 分片，**每篇文档后面跟一个 `<|endoftext|>`**；元数据里记录分词器哈希 `Tokenizer.hash()` | 预训练不能每一步现场分词。`<|endoftext|>` 让模型学会"文档到此结束"。哈希防止用 A 分词器切的数据去训练 B 分词器的模型。用 uint32 是因为词表可能超过 65,535 |
| 无 | `Tokenizer.save_hf(out_dir)` | 导出成 transformers 的 `AutoTokenizer` 能直接读取的文件，发布时要用（第 20 章） |

**对拍**：同一份训练文本（三份语料各 60,000 字符）、同样 768 次合并，验证集上的字节 / token 如下：

| | 英文 | 中文 | 代码 |
|---|---:|---:|---:|
| 手写 BPE（`02_bpe.py`） | 1.76 | 1.74 | 1.95 |
| `zero/tokenizer.py` | 1.86 | 1.79 | 1.97 |

两者都能把验证集逐字节还原，数字切分也完全一样（`x = 2026` → `x` `␣=` `␣` `2` `0` `2` `6`）。压缩率不完全相同，这在预期之中。正则不同（`\p{L}` vs `[^\W\d_]`），次数相同时选哪一对的规则也不同。这两点都会让合并顺序略有差别。

生产级分词器的质量由 `tests/test_tokenizer.py` 保证（`uv run pytest tests/test_tokenizer.py`，本机 12 项全部通过）。测试检查这几点：

- 中文、英文、代码、emoji、全角空格、首尾空白的**编码再解码能还原原文**。
- 特殊 token 的 id 固定，不会被拆开。
- `"2026"` 编成 4 个 token。
- 中英代码上 `bytes_per_token` 都大于 1.5，中文大于 2.0。
- 保存再读取后，编码和哈希都不变。
- 导出后，transformers 的 `AutoTokenizer` 的编码结果与 zero 完全一致。

---

## 采用方与来源

| 技术 | 采用方（主力版本） | 来源 |
|---|---|---|
| Byte-level BPE（以 256 个字节打底，没有 `<unk>`） | GPT-2；Llama 3（128K 词表：tiktoken 的 100K + 28K 非英语 token）；Qwen（Qwen2/Qwen3/Qwen3.5 的 `tokenizer.json`：`ByteLevel` 预切分 + `BPE` 模型）；DeepSeek-V3（128K，`ByteLevel` + `BPE`）；OLMo 2（`ByteLevel` + `BPE`） | GPT-2 论文第 2.2 节；Llama 3 论文；各模型 Hugging Face 仓库里的 `tokenizer.json`（链接见参考文献） |
| 正则预切分 | 从 GPT-2 开始；Qwen3、Llama 3.2、DeepSeek-V3、OLMo 2 的 `tokenizer.json` 里都有 `Split` 正则 | 同上 |
| 数字切分 | Qwen3 / Qwen3.5：`\p{N}`（逐个）；Llama 3.2、DeepSeek-V3、OLMo 2：`\p{N}{1,3}`（最多 3 位一组） | 同上（2026-09 读取） |
| 词表大小 | GPT-2 50,257；Llama 3 系列 128,256；Qwen3 151,936；Qwen3.5 248,320（`config.json` 的 `vocab_size`） | 各模型的 `config.json` |
| 用 bits-per-byte 跨分词器比较 | nanochat（`val_bpb`，`nanochat/loss_eval.py` 的 `evaluate_bpb`） | <https://github.com/karpathy/nanochat> |

Gemma 系列用的是 SentencePiece 分词器，不在上表里。它是否也属于"字节兜底的 BPE"，请以其技术报告为准（待核实）。

---

## 引导问题

带着这些问题去问 Claude Code，直到你能用自己的话讲清楚：

1. 链式法则 `p(x_1…x_T) = Π p(x_t | x_<t)` 是恒等式，不是近似。那么 bigram 的近似发生在哪一步？trigram（看前 2 个 token）的表有多大？V = 65,536 时存得下吗？
2. 为什么 BPE 学到的第一个合并是 `\xef\xbc`，而不是某个汉字？如果训练语料全是英文，第一个合并大概是什么？修改 `02_bpe.py` 来验证。
3. "困惑度 37 的 BPE 模型比困惑度 12.7 的字节级模型更好"——用 bpb 的定义解释这句话。如果两个模型用同一个分词器，比较损失和比较 bpb 的结论会不会不同？
4. 本章的中文 bigram 里，字符级（bpb 2.841）胜过 1024 词表的 BPE（3.164）。如果把 BPE 词表增加到 8,192，你猜会怎样？为什么？
5. Qwen3.5-0.8B 的 embedding 约有 2.5 亿参数。假设总参数预算固定在 0.8B，把词表从 248K 减到 65K。省下的参数可以做什么？代价又是什么？
6. 为什么预训练数据里每篇文档后面要加 `<|endoftext|>`？如果不加，模型生成文本时会出现什么现象？

## 动手任务

每个任务都要运行代码，并看到结果。

**任务 1（基础）**：在 `02_bpe.py` 里把 `vocab_size` 从 1024 改成 2048 和 4096。记录训练时间和三份语料验证集上的字节 / token，并检查"习""说"是否变成了一个 token。

**任务 2（核心）**：给 `03_bigram.py` 加一个 **unigram** 基线。unigram 完全不看前文：`p(x_t) = count(x_t) / N`。算出三种分词在三份语料上的 bpb，和 bigram 比较："看前 1 个 token"能带来多少 bits/byte 的改进？

**任务 3（挑战）**：把 `03_bigram.py` 里的 BPE 换成用 `zero.tokenizer.train_bpe` 训练的 8,192 词表分词器（写法见 `04_vocab_and_zero.py`），重新算三份语料的 bpb。词表变大后，bigram 的表有 8,192² ≈ 6,700 万格，而训练数据只有一百多万 token。你会看到什么？调整平滑系数 α，解释你的发现。这个"数据不够填满表"的问题，是下一章用神经网络（而不是计数表）来建模的原因之一。

---

## 想深入：CS336

本章对应斯坦福 CS336（Spring 2026）<https://cs336.stanford.edu/>：

- **第 1 讲：概览与分词**。它从字符级、字节级、词级讲到 BPE，和本章是同一条线，但更系统地讨论了各种分词方案的优缺点。
- **第 2 讲：PyTorch 与资源核算**。本章第 5 节的"embedding 参数 = V × d"只是资源核算的一小部分。第 2 讲教你把一个模型的参数、显存、FLOPs 全部算清楚（本课第 12 章会再用到）。
- **作业 1（Basics）的 BPE 部分**：从零实现 byte-level BPE 的训练与编解码。作业要求在 TinyStories 和 OpenWebText 上训练，并控制时间和内存。它比本章的 `02_bpe.py` 要求高得多：需要并行化预切分、增量更新计数。本章代码可以作为热身。作业仓库：<https://github.com/stanford-cs336/assignment1-basics>

---

## 本章参考文献

- Sennrich, Haddow, Birch. *Neural Machine Translation of Rare Words with Subword Units*（BPE 用于分词），2015：<https://arxiv.org/abs/1508.07909>
- Radford et al. *Language Models are Unsupervised Multitask Learners*（GPT-2，byte-level BPE，第 2.2 节）：<https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf>
- Llama Team, Meta. *The Llama 3 Herd of Models*（128K 词表 = tiktoken 的 100K + 28K 非英语 token），2024：<https://arxiv.org/abs/2407.21783>
- Karpathy. minbpe（最小的 byte-level BPE 实现，含 GPT-4 分词器的复现）：<https://github.com/karpathy/minbpe>
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
- [nanoGPT](https://github.com/karpathy/nanoGPT)、[minimind](https://github.com/jingyaogong/minimind)：都从字符级或 BPE 分词开始讲语言模型

**下一章**：bigram 只能看前 1 个 token。要猜"不亦说乎"的"乎"，模型得回头看"不亦"；要写出一句押韵的诗，得记得前面好几句。第 8 章，我们从"对前面所有 token 做加权平均"出发，推导出**注意力**（attention）。注意力让每个位置自己决定该看前文的哪里。
