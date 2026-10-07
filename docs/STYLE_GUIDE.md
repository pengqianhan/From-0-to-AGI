# Style Guide: English and Chinese

**English** · [中文](STYLE_GUIDE.zh.md)

This guide applies to all course text, code comments, docstrings, and printed output.

The course is for readers around the world. English is the main language. Each page also has a Chinese version. Both versions use the writing rules of **ASD-STE100 Simplified Technical English (STE)** as a guide. We use the STE rules. We do not apply the STE dictionary word by word, because the course must use technical names such as "attention", "tokenizer", and "gradient".

## 1. Why we use STE rules

Many readers do not speak English as their first language. Many readers also use machine translation. Short, direct sentences are easier to read, easier to translate, and easier to check. The same is true for the Chinese text.

## 2. Rules for English text

### 2.1 Sentences

1. Write one topic in one sentence.
2. Use a maximum of 20 words in a sentence that gives an instruction (a procedural sentence).
3. Use a maximum of 25 words in a sentence that gives information (a descriptive sentence).
4. Use a maximum of 6 sentences in a paragraph.
5. Start a paragraph with its main idea.
6. Use the active voice. Write "The optimizer updates the weights." Do not write "The weights are updated by the optimizer."
7. Use the simple present tense for facts. Use the simple past tense for results that we measured.
8. Use the imperative for instructions. Write "Run the script." Do not write "You may want to run the script."
9. Write one instruction in one sentence. If two actions occur at the same time, you can write them in one sentence.
10. Do not omit articles ("a", "an", "the"), unless the text is a table cell or a heading.

### 2.2 Words

1. Use one word for one meaning. Use the terms in the glossary (Section 6). Do not use a different word only to add variety.
2. Use short, common words. Write "use", not "utilize". Write "start", not "commence". Write "make sure", not "ensure". Write "about", not "approximately".
3. Do not use idioms, slang, or jokes. They are difficult to translate.
4. Do not use "clearly", "obviously", "simply", "just", or "easy". The reader may not find it easy.
5. Use a technical name when it is the correct name. Write code names in backticks: `loss.backward()`.
6. When you use a technical term for the first time on a page, give a short definition.
7. Do not use "it", "this", or "that" when the reader can be unsure what they refer to. Write the noun again.

### 2.3 Analogies and examples

1. Use an analogy only when it helps the reader to understand a mechanism.
2. Keep the analogy short and literal: "The loss surface has the shape of a bowl."
3. After the analogy, give the exact statement.

### 2.4 Lists, notes, and warnings

1. Introduce a list with a full sentence that ends with a colon.
2. Make all items in a list parallel: all sentences, or all nouns.
3. Use numbered lists for steps that occur in a sequence. Use bullets for other lists.
4. Write a note as a blockquote that starts with **Note:**. Write a warning as a blockquote that starts with **Warning:**.

### 2.5 Numbers and units

1. Use digits for all measured values: "0.25", "200 steps", "4.3M parameters".
2. Put a space between a number and its unit: "24 GB", "15 min". Exception: "%" and "×".
3. Use the same number of decimal places as the program output.
4. Do not change a number in a translation. Numbers come from the program output, not from the text.

## 3. Rules for Chinese text (中文规则)

The Chinese version follows the same principles. The Chinese text is a full rewrite with the same structure as the English text. It is not a word-for-word translation.

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

## 4. Rules for code

1. Write all comments, docstrings, and printed output in English. Use the rules in Section 2.
2. Write a comment to tell the reader **why** the code does something. Do not repeat what the code says.
3. Keep mathematical notation in comments when it links the code to a formula in the text: `# ∂L/∂a = 2/N · Σ (ŷ_i − y_i) · x_i`.
4. Do not change names of functions, classes, variables, files, or configuration keys. Other files and the videos import these names.
5. Do not change the program logic. If a printed label changes, the printed numbers must stay the same.
6. Keep the format of a printed table. The text copies these tables.

## 5. Files and the language switch

1. The English page is `README.md`. The Chinese page is `README.zh.md` in the same folder. The same rule applies to other documents: `GOAL.md` and `GOAL.zh.md`.
2. The two pages have the same headings in the same order. Then the language button on the course website opens the same section in the other language.
3. The first line under the title links to the other language: `**English** · [中文](README.zh.md)`.
4. Links to code, data, and other chapters are the same in both pages. A link to another chapter goes to the page in the same language: `../02-from-scalar-to-matrix/README.zh.md` in a Chinese page.
5. The course website uses MkDocs Material with the `mkdocs-static-i18n` plugin. English is the default language. See `site/README.md`.

## 6. Glossary

Use these terms. If a term is not in the list, add it here before you use it in more than one chapter.

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
