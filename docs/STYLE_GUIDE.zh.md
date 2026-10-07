# 写作规范：英文与中文

[English](STYLE_GUIDE.md) · **中文**

本规范适用于课程的全部文字、代码注释、docstring 和程序输出。

这门课面向全世界的读者。英文是主要语言，每一页也有中文版本。两种语言都以 **ASD-STE100 简化技术英语（STE）**的写作规则为指导。我们采用 STE 的规则，但不逐词套用 STE 词典，因为课程必须使用 attention、tokenizer、梯度这类技术名称。

## 1. 为什么采用 STE 规则

很多读者的母语不是英语，很多读者也会用机器翻译。短而直接的句子更容易读、更容易翻译、也更容易检查。中文同样如此。

## 2. 英文规则

### 2.1 句子

1. 一句话只讲一个主题。
2. 给出指令的句子（步骤句）不超过 20 个词。
3. 给出信息的句子（描述句）不超过 25 个词。
4. 一段不超过 6 句。
5. 段落的第一句写这一段的要点。
6. 用主动语态。写 "The optimizer updates the weights."，不写 "The weights are updated by the optimizer."
7. 陈述事实用一般现在时，陈述测得的结果用一般过去时。
8. 指令用祈使句。写 "Run the script."，不写 "You may want to run the script."
9. 一句只写一个指令。两个动作同时发生时，可以写在一句里。
10. 除表格单元格和标题外，不省略冠词（a、an、the）。

### 2.2 用词

1. 一个意思只用一个词，使用第 6 节术语表里的词。不为了变换说法而换同义词。
2. 用简短、常见的词：use 而不是 utilize，start 而不是 commence，make sure 而不是 ensure，about 而不是 approximately。
3. 不用习语、俚语和玩笑，它们很难翻译。
4. 不用 clearly、obviously、simply、just、easy：读者不一定觉得简单。
5. 正确的名称是技术名称时，就用技术名称。代码名称放在反引号里：`loss.backward()`。
6. 技术术语在一页里第一次出现时，给出简短定义。
7. 读者可能弄不清 it、this、that 指什么时，把名词再写一遍。

### 2.3 类比和例子

1. 只有在类比能帮助读者理解机制时才用。
2. 类比要短、要直接："The loss surface has the shape of a bowl."
3. 类比之后给出准确的说法。

### 2.4 列表、注意和警告

1. 用一个以冒号结尾的完整句子引出列表。
2. 列表的各项结构相同：都是句子，或都是名词短语。
3. 按顺序执行的步骤用编号列表，其他列表用圆点。
4. "注意"写成以 **Note:** 开头的引用块，"警告"写成以 **Warning:** 开头的引用块。

### 2.5 数字和单位

1. 所有测量值用阿拉伯数字："0.25"、"200 steps"、"4.3M parameters"。
2. 数字和单位之间加空格："24 GB"、"15 min"。% 和 × 例外。
3. 小数位数和程序输出相同。
4. 翻译时不改数字。数字来自程序输出，不来自文字。

## 3. 中文规则

中文版遵循同样的原则。中文正文按英文版的结构完整重写，不是逐字翻译。

1. 一句话只讲一件事。描述句一般不超过 40 个字，步骤句一般不超过 30 个字。
2. 一段不超过 6 句。段落的第一句写这一段的要点。
3. 用主动句。写"优化器更新权重"，不写"权重被优化器更新"。
4. 步骤用祈使句："运行这个脚本。"不写"你可以试着运行一下这个脚本"。
5. 一个概念只用一个词，用第 6 节术语表里的写法。不为了变换说法而换同义词。
6. 术语第一次出现时，在括号里给英文：损失（loss）。
7. 不用成语、俗语、网络用语和反问句。不用"显然""其实""非常""简单地"这类词。
8. 类比只在能帮助理解机制时用。类比要短，之后给出准确的说法。
9. 数字、公式、代码、表格里的数值与英文版完全相同。
10. 标点用全角。中文和英文、数字之间加一个空格。数字和单位之间加一个空格（% 和 × 除外）。

## 4. 代码规则

1. 注释、docstring 和程序输出全部用英文，遵循第 2 节的规则。
2. 注释说明代码**为什么**这样做，不重复代码已经写明的内容。
3. 公式能把代码和正文对应起来时，在注释里保留数学符号：`# ∂L/∂a = 2/N · Σ (ŷ_i − y_i) · x_i`。
4. 不改函数、类、变量、文件和配置项的名称。其他文件和视频会导入这些名称。
5. 不改程序逻辑。输出的标签可以变，输出的数字必须不变。
6. 保持输出表格的格式，正文会引用这些表格。

## 5. 文件和语言切换

1. 英文页面是 `README.md`，中文页面是同一目录下的 `README.zh.md`。其他文档同理：`GOAL.md` 和 `GOAL.zh.md`。
2. 两个页面的标题相同、顺序相同。这样课程网站的语言按钮能打开另一种语言的同一节。
3. 标题下第一行链接到另一种语言：`[English](README.md) · **中文**`。
4. 两个页面里指向代码、数据和其他章节的链接相同。中文页面里指向其他章节的链接，指向同一语言的页面：`../02-from-scalar-to-matrix/README.zh.md`。
5. 课程网站用 MkDocs Material 加 `mkdocs-static-i18n` 插件，默认语言是英文。见 `site/README.md`。

## 6. 术语表

请使用下表的写法。表里没有的术语，在两章以上用到之前，先加进这张表（英文版和中文版的表要同时改）。

| English | 中文 | Notes |
|---|---|---|
| model | 模型 | |
| parameter | 参数 | |
| weight | 权重 | |
| bias | 偏置 | |
| prediction | 预测（值） | ŷ |
| loss | 损失 | not "代价" |
| loss function | 损失函数 | |
| mean squared error (MSE) | 均方误差（MSE） | |
| cross-entropy | 交叉熵 | |
| residual | 残差 | ŷ − y; also the residual connection |
| residual connection | 残差连接 | |
| gradient | 梯度 | |
| partial derivative | 偏导数 | |
| chain rule | 链式法则 | |
| gradient descent | 梯度下降 | |
| learning rate | 学习率 | η |
| step | 步 | one optimizer update |
| update | 更新 | |
| converge / convergence | 收敛 | |
| diverge / divergence | 发散 | |
| optimizer | 优化器 | |
| forward pass | 前向传播 | |
| backward pass | 反向传播 | |
| backpropagation | 反向传播 | |
| automatic differentiation (autograd) | 自动微分（autograd） | |
| closed-form solution | 解析解 | |
| least squares | 最小二乘 | |
| Hessian | Hessian（二阶导数矩阵） | |
| eigenvalue | 特征值 | |
| noise | 噪声 | |
| batch | 批 / batch | "batch size" = batch 大小 |
| mini-batch | mini-batch | |
| stochastic gradient descent (SGD) | 随机梯度下降（SGD） | |
| activation function | 激活函数 | |
| neural network | 神经网络 | |
| layer | 层 | |
| token | token | do not translate |
| tokenizer | 分词器 | |
| vocabulary | 词表 | |
| embedding | 嵌入 / embedding | |
| attention | 注意力 | |
| self-attention | 自注意力 | |
| head (attention head) | 头（注意力头） | |
| context length | 上下文长度 | |
| sequence length | 序列长度 | |
| training | 训练 | |
| pretraining | 预训练 | |
| mid-training | 中期训练 | |
| fine-tuning | 微调 | |
| supervised fine-tuning (SFT) | 监督微调（SFT） | |
| distillation | 蒸馏 | |
| reinforcement learning (RL) | 强化学习（RL） | |
| reward | 奖励 | |
| inference | 推理 | |
| evaluation | 评测 | |
| benchmark | 基准 | |
| validation set | 验证集 | |
| test set | 测试集 | |
| overfitting | 过拟合 | |
| scaling law | scaling law | do not translate |
| compute | 算力 | |
| checkpoint | checkpoint | do not translate |
| warmup | warmup | do not translate |
| bits per byte (BPB) | bits-per-byte（BPB） | |
| production code | 生产级代码 | the code in `zero/` |
| minimal code | 极简代码 | the code in `chapters/*/code/` |
| main-line model | 主线模型 | the model that the course trains |
| self-check | 自检 | the `/chNN-…` skills |
| guided questions | 引导问题 | section name |
| hands-on tasks | 动手任务 | section name |
| parity check | 对拍 | two implementations give the same result |
