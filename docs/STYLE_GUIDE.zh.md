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
| vector | 向量 | |
| matrix | 矩阵 | |
| matrix multiplication | 矩阵乘法 | |
| dot product | 点积 | |
| transpose | 转置 | |
| shape | 形状 | tensor shape |
| broadcasting | 广播 | |
| vectorization | 向量化 | |
| element-wise multiplication | 元素级乘法 | |
| feature | 特征 | |
| standardization | 标准化 | |
| batch dimension | batch 维 | |
| computational graph | 计算图 | |
| local derivative | 局部导数 | |
| upstream gradient | 上游梯度 | |
| topological sort | 拓扑排序 | |
| gradient check | 梯度检验 | |
| numerical gradient | 数值梯度 | |
| central difference | 中心差分 | |
| reverse mode / forward mode | 反向模式 / 前向模式 | automatic differentiation |
| Jacobian | 雅可比矩阵 | |
| vector-Jacobian product (VJP) | 向量-雅可比积（VJP） | |
| curvature | 曲率 | |
| cosine similarity | 余弦相似度 | |
| power law | 幂律 | |
| multilayer perceptron (MLP) | 多层感知机（MLP） | |
| hidden unit | 隐藏单元 | |
| width | 宽度 | of a layer |
| kink | 折点 | |
| piecewise-linear function | 分段线性函数 | |
| universal approximation theorem | 万能近似定理 | |
| dying ReLU | 死 ReLU | |
| feed-forward network (FFN) | 前馈网络（FFN） | |
| gating | 门控 | |
| initialization | 初始化 | |
| normalization | 归一化 | |
| weight decay | 权重衰减 | |
| gradient clipping | 梯度裁剪 | |
| learning-rate schedule | 学习率调度 | |
| cosine decay | 余弦衰减 | |
| momentum | 动量 | |
| vanishing / exploding gradient | 梯度消失 / 梯度爆炸 | |
| residual stream | 残差流 | |
| loss spike | 损失尖峰（loss spike） | |
| scale invariance | 尺度不变性 | |
| hyperparameter | 超参数 | |
| ablation | 消融 | |
| logits | logits | do not translate |
| perplexity | 困惑度 | |
| maximum likelihood estimation (MLE) | 最大似然估计 | |
| log-likelihood | 对数似然 | |
| negative log-likelihood (NLL) | 负对数似然 | |
| KL divergence | KL 散度 | |
| forward / reverse KL | 前向 / 反向 KL | |
| label smoothing | label smoothing | do not translate |
| temperature | 温度 | |
| underflow | 下溢 | |
| saturation | 饱和 | |
| confidence interval | 置信区间 | |
| pre-tokenization | 预切分 | |
| merge | 合并 | a BPE merge |
| character level / byte level | 字符级 / 字节级 | |
| code point | 码位 | Unicode |
| special token | 特殊 token | |
| smoothing | 平滑 | |
| query / key / value | 查询 / 键 / 值 | Q / K / V |
| causal mask | 因果 mask | |
| multi-head attention | 多头注意力 | |
| multi-query attention (MQA) | 多查询注意力（MQA） | |
| grouped-query attention (GQA) | 分组查询注意力（GQA） | |
| rotary position embedding (RoPE) | 旋转位置编码（RoPE） | |
| tied embeddings | 共享 embedding | not "shared embedding" |
| heat map | 热力图 | |
| autoregressive generation | 自回归生成 | |
| greedy decoding | 贪心解码 | |
| sampling | 采样 | not "抽样" |
| KV cache | KV cache | do not translate |
| prefill / decode | prefill / decode | do not translate |
| query head | 查询头 | |
| chunked prefill | 分块 prefill | |
| continuous batching | 连续批处理 | |
| throughput | 吞吐 | |
| arithmetic intensity | 算术强度 | |
| memory-bound / compute-bound | 带宽受限 / 算力受限 | |
| ledger | 账本 | the KV cache ledger |
| latent vector | 潜向量 | MLA |
| absorption | 吸收 | MLA |
| decoupled RoPE | 解耦 RoPE | |
| sliding window | 滑动窗口 | |
| ring buffer | 环形缓冲区 | |
| linear attention | 线性注意力 | |
| mixture of experts (MoE) | 混合专家（MoE） | |
| router | 路由器 | |
| auxiliary loss | 辅助损失 | |
| speculative decoding | 推测解码 | |
| draft | 草稿 | draft model, draft tokens |
| acceptance rate | 接受率 | |
| mixed precision | 混合精度 | |
| master weights | 主权重 | |
| bit-identical | 逐位相同 | |
| activation checkpointing | 激活检查点 | |
| gradient accumulation | 梯度累积 | |
| data parallelism | 数据并行 | |
| micro batch | micro batch | do not translate |
| GPU memory | 显存 | |
| GPU-hour | 卡时 | |
| wall-clock time | 墙钟时间 | |
| resume (from a checkpoint) | 续训（断点续训） | |
| deduplication (dedup) | 去重 | |
| near-deduplication | 近似去重 | |
| contamination / decontamination | 污染 / 去污染 | |
| data mixture | 配比 | |
| shard | 分片 | |
| manifest | 清单 | |
| provenance | 出处 | |
| heuristic rules | 启发式规则 | |
| synthetic rephrasing | 合成改写 | |
| language identification | 语言识别 | |
| proxy model | 代理模型 | |
| funnel | 漏斗 | |
| compute-optimal | 算力最优 | |
| overtraining | 过训练 | |
| ladder experiment | 阶梯实验 | |
| extrapolation | 外推 | |
| held-out set / held-out test | 留出集 / 留出检验 | |
| learning-rate sweep | 学习率扫描 | |
| recipe | 配方 | training recipe |
| MFU (model FLOPs utilization) | MFU | do not translate |
| annealing | 退火 | |
| stable phase / decay phase | 稳定段 / 衰减段 | WSD |
| branched decay (WSD branch) | 分叉衰减（WSD 分叉） | |
| base frequency | 基频 | RoPE |
| position interpolation (PI) | 位置内插（PI） | |
| needle in a haystack | 大海捞针 | |
| base model | 底座模型 | |
| chat template | 对话模板 | |
| packing | 打包 | |
| cross-contamination | 串门 | between packed documents |
| teacher / student | 教师 / 学生 | |
| soft label | 软标签 | |
| sequence-level distillation | 序列级蒸馏 | |
| on-policy distillation | 在线策略蒸馏 | |
| exposure bias | 暴露偏差 | |
| rejection sampling | 拒绝采样 | |
| preference pair | 偏好对 | |
| reward model (RM) | 奖励模型 | |
| reference model | 参考模型 | |
| implicit reward | 隐式奖励 | |
| reward hacking | reward hacking（奖励作弊） | |
| policy / policy gradient | 策略 / 策略梯度 | |
| baseline | 基线 | |
| advantage | 优势 | |
| importance ratio | 重要性比 | |
| verifier | 验证器 | |
| verifiable reward | 可验证奖励 | |
| entropy collapse | 熵坍缩 | |
| multi-armed bandit | 多臂老虎机 | |
| development set (dev set) | 开发集 | |
| exact match | 精确匹配 | |
| paired bootstrap | 配对 bootstrap | |
| preregistration | 预注册 | |
| primary endpoint | 主终点 | |
| ahead / tie / behind | 超过 / 持平 / 落后 | the values in code are "ahead", "tie", "behind" |
| opponent | 对手 | |
| canary string | canary 字符串 | |
| prompt | 提示词 | |
| thinking mode | 思考模式 | |
| few-shot | 少样本 | |
| tool calling | 工具调用 | |
| smoke test | 冒烟测试 | `zero.smoke` |
| gate (Gate 1, 2, 3) | 闸门（闸门 1、2、3） | |
| hard goal | 硬目标 | not "hard target" |
| model card | 模型卡 | |
| block-wise quantization | 分块量化 | |
| outlier | 离群值 | |
| Step 1 / Step 2 | 第一步 / 第二步 | project phases |
| tiny-configuration demo | 极小配置演示 | section name |
| to be verified | 待核实 | |
| 10k yuan | 万元 | unit of money |
| mask | mask（掩码） | causal mask = 因果 mask |
| sparse attention | 稀疏注意力 | |
| receptive field | 感受野 | |
| local-global interleaving | 局部-全局交替 | |
| attention sink | 注意力汇聚点 | |
| state space model | 状态空间模型 | |
| chunkwise form / recurrent form | 分块形式 / 递推形式 | |
| delta rule | delta 规则 | |
| hybrid architecture | 混合架构 | |
| associative recall | 联想回忆 | |
| recurrent neural network (RNN) | 循环神经网络（RNN） | |
| expert | 专家 | MoE |
| routed / shared expert | 路由专家 / 共享专家 | |
| total / active parameters | 总参数 / 激活参数 | |
| dense model | 稠密模型 | |
| load balancing | 负载均衡 | |
| routing collapse | 路由坍缩 | |
| capacity factor | 容量因子 | |
| expert parallelism (EP) | 专家并行（EP） | |
| multi-token prediction (MTP) | 多 token 预测（MTP） | |
| target model | 目标模型 | speculative decoding |
| bonus token | 额外 token（bonus token） | not "奖励 token" |
| speedup | 加速比 | |
| total variation (TV) distance | 总变差距离 | |
| consensus | 共识 | |
| adopter | 采用方 | |
| evolution tree | 演化树 | |
| post-training | 后训练 | |
| agentic reinforcement learning | 智能体强化学习 | |
| model soup | 模型汤 | |
| chunked cross-entropy | 分块交叉熵 | |
| deterministic algorithms | 确定性算法 | |
| scaling efficiency | 扩展效率 | multi-GPU |
| bus bandwidth | 总线带宽 | NCCL |
| power cap | 功耗上限 | |
| steady state | 稳定后 | throughput; not the WSD "stable phase" |
| code path | 通路 | |
| convention | 口径 | how a number is counted |
| backend | 后端 | |
| match / timing only / mismatch / error | 一致 / 仅计时不同 / 不一致 / 报错 | verification records |
| runbook | 运行手册 | |
| release checklist | 发布清单 | |
| cost ledger | 花费账本 | |
| per-question results | 逐题结果 | |
| gap analysis | 差距分析 | |
| project lead | 项目负责人 | |
| size class | 档位 | |
| benchmark model | 标杆 | not "benchmark" (基准) |
| type I error rate | 第一类错误率 | |
| descriptive report | 描述性报告 | |
| adapter layer | 适配层 | |
| Amendments | 修订记录 | section name in eval/PREREGISTRATION.md |
| narration | 旁白 | video |
| storyboard / shot | 分镜 / 镜头 | video |
| opening | 片头 | video |
| fact list | 事实清单 | video/script.md |
| delivery check | 交付检查 | video |
| final video / sample video | 成片 / 样片 | |
| quick-read text | 速读正文 | |
| streaming read | 流式读取 | |
| extrapolation factor | 外推倍数 | |
| soft metric / hard score | 软指标 / 硬分数 | |
