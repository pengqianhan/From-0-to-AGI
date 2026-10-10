# 第 17 章：蒸馏 —— 小模型向大模型学习

[English](README.md) · **中文**

> **目标**：读完这一章，你能说清楚为什么教师的"软标签"比 one-hot 标签多带信息。你能手推带温度的 KD 损失和它的梯度，并与 `zero` 对拍。你能说明 logits 蒸馏、序列级蒸馏、在线策略蒸馏分别需要什么条件（尤其是"同一个词表"）。你能说明为什么主线模型只能做序列级蒸馏。你会用"多次采样 + 执行验证"筛出干净的工具调用数据。你还能查一个教师模型的许可证，判断它能不能当教师。

📺 **本章视频**：还没有发布。要在本机渲染，运行 `bash chapters/17-distillation/video/build.sh`。
🧪 **本章自检**：学完后，在 Claude Code 里输入 `/ch17-distillation`。

---

上一章我们用 SFT 把底座（base）模型变成了一个助手。助手会按格式回答，会调用工具。我们给它一批"问题 → 标准回答"的对话，只在回答部分计算交叉熵。这一章回答下一个问题：**标准回答从哪里来？什么样的回答教得最好？**

人工写成本太高。小模型自己写，质量又不够好。几乎所有头部开源模型家族给出同一个答案：**找一个更强的模型当教师**。

- **Llama 3.2** 的 1B、3B：Meta 先从 Llama 3.1 8B 剪枝，再把 8B 和 70B 的 logits 作为每个 token 的训练目标。
- **Gemma 2** 的 2B、9B：用知识蒸馏代替"预测下一个 token"来预训练。**Gemma 3** 的所有尺寸都用蒸馏训练。
- **Qwen3** 的 0.6B–14B 和 30B-A3B：这些模型不走旗舰模型的四阶段后训练，改用"强到弱蒸馏"，GPU 时只需要约十分之一。
- **DeepSeek-R1-Distill**：用 DeepSeek-R1 生成的约 80 万条数据，对 Qwen2.5、Llama 3 的底座只做 SFT。1.5B 的学生在数学基准上超过了 GPT-4o。

本章代码（全部在 CPU 上运行）：

```bash
uv run python chapters/17-distillation/code/01_soft_labels.py         # soft labels, temperature, KD gradient, parity check with zero (a few seconds)
uv run python chapters/17-distillation/code/02_toy_distill.py         # toy experiment: hard labels vs sequence-level vs logits distillation (about 2 min of CPU time)
uv run python chapters/17-distillation/code/03_forward_reverse_kl.py  # forward KL vs reverse KL, a distribution with two peaks (a few seconds)
uv run python chapters/17-distillation/code/04_rejection_sampling.py  # the funnel of rejection sampling + execution check (a few seconds)
uv run python chapters/17-distillation/code/05_shared_vocab.py        # why the vocabulary must be the same; the parameter cost of a new vocabulary (a few seconds)
```

## 1. 直觉：一个答案 vs 一整张分布

### 1.1 one-hot 只说"对的是哪个"

回忆第 5 章：训练分类器时，标签是 one-hot：正确的词是 1，其余全是 0。（语言模型也是一个分类器，它在词表上做分类。）设上文是"今天天气很"，训练文本里下一个字是"好"。模型只被告知"好"。

可是"热""冷""晴"也说得通，"猫""跑"则完全说不通。one-hot 标签把这些信息全部丢掉了。它不区分"次优答案"和"荒唐答案"，一律给 0。

### 1.2 教师的软标签把"次优"也告诉你

设旁边有一个训练好的大模型（**教师**，teacher）。在这个位置，教师给出的不是一个字，而是一整张概率分布，也就是**软标签**（soft label）。`01_soft_labels.py` 的例子如下：

| | 好 | 热 | 冷 | 晴 | 猫 | 跑 | 熵 |
|---|---:|---:|---:|---:|---:|---:|---:|
| one-hot | 1.000 | 0 | 0 | 0 | 0 | 0 | 0 bit |
| 教师 p_T | 0.607 | 0.223 | 0.100 | 0.067 | 0.002 | 0.001 | 1.535 bit |

one-hot 标签的熵是 0。在每个训练位置，它只给出一条信息：答案是哪个。教师的分布有 1.5 bit 的熵。它还给出了**相对大小**："热比冷更可能，晴比猫可能得多"。这种信息藏在错误答案的概率里，后来常被叫作**暗知识**（dark knowledge）。

Hinton 等人（2015）举过一个识别手写数字的网络的例子。对某个"2"，网络给"是 3"的概率 10⁻⁶，给"是 7"的概率 10⁻⁹。对另一个"2"，这两个数可能正好反过来。这两个极小的数说明了"这个 2 更像 3 还是更像 7"。

**蒸馏**（knowledge distillation，KD）让小模型（**学生**，student）学习教师的整张分布，而不只是 one-hot 答案。Gemma 2 报告的说法很直接：把每个 token 上的 one-hot 标签换成大模型算出的下一个 token 分布，**让每一步训练收到更丰富的信息**。Gemma 团队用这个方法，让 2B、9B 模型"模拟"在比实际数据多得多的 token 上训练。

### 1.3 温度：把暗知识放大

教师通常很自信。错误答案的概率小到学生可以忽略（上表里"猫"只有 0.002）。Hinton 的方法借用第 5 章的**温度**（temperature）：先把 logits 除以 τ，再做 softmax，`p_T^τ = softmax(z_T / τ)`。

| τ | 好 | 热 | 冷 | 晴 | 猫 | 跑 | 熵 (bit) | p(晴)/p(猫) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.5 | 0.851 | 0.115 | 0.023 | 0.010 | 0.000 | 0.000 | 0.752 | 1998.2 |
| 1 | 0.607 | 0.223 | 0.100 | 0.067 | 0.002 | 0.001 | 1.535 | 44.7 |
| 2 | 0.412 | 0.250 | 0.168 | 0.137 | 0.021 | 0.012 | 2.046 | 6.7 |
| 4 | 0.295 | 0.230 | 0.188 | 0.170 | 0.066 | 0.051 | 2.373 | 2.6 |

τ 越大，排名靠后的候选分到的概率越多，学生能"看见"的差别也越多。但排名始终不变（晴 > 猫）。训练时，学生也用同一个 τ。推理时，τ 回到 1。

## 2. logits 蒸馏：逐位置贴近教师的分布

### 2.1 公式

在同一段文本的每个位置 t，教师和学生各给出一个词表上的分布。logits 蒸馏最小化两个分布之间的 KL 散度。（KL 散度是第 5 章交叉熵的"亲戚"。）

```
L_KD = τ² · KL(p_T^τ ‖ p_S^τ) = τ² · Σ_v p_T^τ(v) · (log p_T^τ(v) − log p_S^τ(v))
```

对 mask 内的位置取平均（和 SFT 一样，只在回答部分计算）。实际训练常把它和普通交叉熵混合：`L = (1 − α)·CE + α·L_KD`。

`KL(p_T ‖ p_S) = H(p_T, p_S) − H(p_T)`。教师的熵 `H(p_T)` 与学生无关。所以对学生来说，**最小化 KL 就是最小化"以教师分布为标签的交叉熵"**。把教师分布换成 one-hot 标签，这个损失就变成普通 SFT 的交叉熵。硬标签只是软标签的一个特例。Gemma 2 报告写的正是这个交叉熵形式。

### 2.2 梯度：从 `p − onehot` 到 `p_S − p_T`

第 5 章推过交叉熵对 logits 的梯度：`p − onehot`。这里照同样的步骤推导。`log p_S^τ(v) = z_S(v)/τ − logsumexp(z_S/τ)`。它对 `z_S(k)` 的导数是 `(1/τ)·([v = k] − p_S^τ(k))`。把这个导数代入 `−Σ_v p_T^τ(v) log p_S^τ(v)`，再利用 `Σ_v p_T^τ(v) = 1`：

```
∂L_KD/∂z_S(k) = τ² · (1/τ) · (p_S^τ(k) − p_T^τ(k)) = τ · (p_S^τ(k) − p_T^τ(k))
```

梯度仍然是"学生的预测减去目标"。只是目标从 one-hot 标签变成了教师的分布。代码里只有一行：

```python
def kd_grad(z_s, z_t, tau):
    return tau * (softmax(z_s, tau) - softmax(z_t, tau))   # ∂L/∂z_S = τ·(p_S^τ − p_T^τ)
```

`01` 的第 3 部分做了中心差分检验。τ = 1、2 时，解析梯度与数值梯度的最大差是 1.2×10⁻¹¹、3.1×10⁻¹¹。下面把 KD 梯度和同一个学生的硬标签梯度对比：

| | 好 | 热 | 冷 | 晴 | 猫 | 跑 |
|---|---:|---:|---:|---:|---:|---:|
| 硬标签 `p_S − onehot` | −0.5231 | +0.1064 | +0.2893 | +0.0391 | +0.0645 | +0.0237 |
| KD（τ = 1）`p_S − p_T` | −0.1301 | −0.1169 | +0.1889 | −0.0281 | +0.0630 | +0.0232 |

硬标签只把"好"往上推，其余一律往下压。KD 发现学生给"热""晴"的概率偏低，所以也把它们往上推（梯度为负）。KD 只往下压学生高估了的"冷""猫""跑"。

**为什么乘 τ²？** 不乘的话，梯度是 `(1/τ)·(p_S^τ − p_T^τ)`。τ 大时，两个分布都接近均匀，差值也按 1/τ 缩小。合起来，梯度按 1/τ² 变小。（`01` 的输出：τ = 1、2、4、8 时，梯度范数是 0.268、0.100、0.035、0.011。）乘上 τ² 后，梯度的大小基本不随 τ 变化。这样换温度时，不用重新调学习率和 α。

τ 很大时，还能看出 KD 的另一面：`p ≈ (1 + z/τ)/V`。在 logits 均值为零时，梯度 ≈ `(z_S − z_T)/V`。这个梯度**直接让学生的 logits 贴近教师的 logits**（Hinton 等人 2015 第 2.1 节）。

### 2.3 与生产级代码对拍

`01` 的最后一部分把极简版和 `zero.post.distill.kd_loss` 放在同一批随机 logits 上运行（2×5 个位置、词表 11、mask 掉 2 个位置）：

| | 极简代码 | zero | 差 |
|---|---:|---:|---:|
| τ = 1 | 4.164090 | 4.164090 | 3.0×10⁻⁷ |
| τ = 2 | 5.177818 | 5.177818 | 3.0×10⁻⁷ |
| top-3 | 4.800477 | 4.800477 | 1.2×10⁻⁷ |

**top-k**：教师只保留概率最大的 k 个 token，并在这 k 个 token 上重新归一化。学生不重新归一化。词表可能有十几万个 token。这时把每个位置的完整教师分布存下来太大。只存 top-k，能把存储减少几个数量级。

Gemma 3 的做法类似，但是随机的：在每个 token 上，**按教师概率抽 256 个 logits**。没抽中的 logits 置 0，再重新归一化。

### 2.4 硬约束：同一个词表

logits 蒸馏逐位置、逐维比较两个分布。所以**两边的第 t 个位置必须是同一个"下一个 token"，第 v 维必须是同一个 token**。也就是说，教师和学生必须用**同一个分词器**。

`05_shared_vocab.py` 用两个玩具词表切同一句"今天天气很好"。学生切成 `今天 | 天气 | 很 | 好`（4 个）。教师切成 `今天天 | 气很 | 好`（3 个）。连位置数都对不上。`zero` 的 `kd_loss` 遇到形状不同时直接报错。`run_distill` 还会比较两边分词器的哈希。

上面四个采用方都满足这一点。Llama 3.2 与 Llama 3.1 用同一个分词器。Gemma 2/3 的所有尺寸共用一个 25.6 万 / 26.2 万的词表。Qwen3 的各尺寸共用一个词表。DeepSeek-R1-Distill 不同：它的学生是 Qwen2.5 和 Llama，分词器和 R1 不同。正因为这样，它**只能做序列级蒸馏**（下一节）。

**这把我们带回第 13 章的词表决定。** 主线模型用自己训练的 65,536 词表（`configs/main/pretrain.toml`）。这个词表和任何开源教师的词表都不同。所以**主线只能做序列级蒸馏**。反过来想：如果当初直接用千问（Qwen）的分词器呢？`05` 保持主线模型的形状（28 层、宽 1280、共享 embedding），只换词表：

| 分词器 | 词表 | embedding | 总参数 | ≤ 0.8B？ |
|---|---:|---:|---:|---|
| 主线（自训 BPE，第 13 章） | 65,536 | 83.9M | 689.5M | 是 |
| Qwen3 分词器 | 151,936 | 194.5M | 800.1M（+110.6M） | 否 |
| Qwen3.5 分词器 | 248,320 | 317.8M | 923.5M（+234.0M） | 否 |

换成 Qwen3 的分词器，就能从 Qwen3 教师做 logits 蒸馏和在线策略蒸馏（第 4 节）。代价是 1.1 亿参数花在 embedding 上，模型刚好越过 0.8B 的上限。我们还得接受一个不是按我们的中英代码配比训练出的词表。这是一个真实的取舍，没有"哪个一定对"。主线选择了前者：小词表、自有分词器、完全开放的配方。所以主线的蒸馏只走序列级。

## 3. 序列级蒸馏：教师写，学生抄

### 3.1 公式：就是 SFT

**序列级蒸馏**（sequence-level KD，Kim & Rush 2016）不看教师的 logits，只看教师**写出来的文本**。对每个提示词，教师生成一段回答 `y ~ p_T(·|x)`。然后学生在这段回答上做普通的 SFT：`L = −log p_S(y|x)`。

它在优化什么？在教师的样本上做最大似然，期望值是 `E_{y~p_T}[−log p_S(y|x)]`。这个值是序列级的交叉熵 `H(p_T, p_S)`。再减去一个与学生无关的常数，就得到**序列级的前向 KL(p_T ‖ p_S)**。

所以它和 logits 蒸馏的目标方向相同，只是用采样代替了整张分布。每条样本只带"教师挑中的那一条路"，信息比整张分布少。但**它对教师只要求"能生成文本"**。任何分词器、任何架构，甚至一个远程 HTTP 接口都行。DeepSeek-R1-Distill 和 Qwen3 小模型的"离线蒸馏"阶段都用这个方法。

### 3.2 玩具实验：同样的步数，四种信号

`02_toy_distill.py` 在莎士比亚文本（`assets/tiny_corpus/shakespeare.txt`，111 万字符）上训练一个字符级语言模型。模型看前 8 个字符，预测下一个字符：

- 教师：MLP，429,665 个参数。在全部训练文本（约 100 万字符）上训练 3000 步。验证集结果是 2.439 bits/char。
- 学生：MLP，8,905 个参数。它**只能看到 2 万字符**的训练文本。（这模拟"学生能拿到的好数据不多"。）我们比较四种训练信号：同样的初始化、2000 步、batch 256、3 个随机种子。

> **注意：**本章训练类实验的数字来自课程构建机上的一次 CPU 运行。不同机器、不同版本的底层数学库，浮点运算的顺序略有不同。训练几百步后，这些微小差异会被放大。所以你本机跑出的数字可能从小数点后第二、三位开始就不同。请以下文不依赖具体数值的结论为准。2026-10 在另一台服务器上的复跑对照见 [runs/2026-10-01-gpu0-check/chapters-16-20.md](../../runs/2026-10-01-gpu0-check/chapters-16-20.zh.md)。

| 训练信号 | 在 2 万训练字符上 | 验证 bits/char（3 个种子平均） | 各种子 |
|---|---:|---:|---|
| A 硬标签（2 万真实字符） | 2.678 | **3.679** | 3.626 / 3.739 / 3.673 |
| B 序列级蒸馏（教师写的 2 万字符） | 3.302 | **3.362** | 3.327 / 3.387 / 3.371 |
| C logits 蒸馏（τ = 2，同样的 2 万字符） | 2.867 | **3.290** | 3.258 / 3.326 / 3.286 |
| D 0.5·硬标签 + 0.5·KD | 2.693 | **3.344** | 3.326 / 3.374 / 3.332 |
| R 参照：硬标签，全部 100 万字符 | 3.123 | 3.143 | 3.114 / 3.170 / 3.145 |

（均匀随机猜测是 log₂65 = 6.022 bits/char。）这张表这样读：

- **A 过拟合了。** 在自己那 2 万字符上，A 是 2.678。在验证集上是 3.679，差了 1 bit。在小数据上，one-hot 标签把学生推向"背答案"。
- **C 最好。** C 用同样的 2 万字符、同样的步数，只把标签换成教师的分布。验证集降到 3.290，比 A 好 0.39 bit，而且三个种子都更好。软标签既带暗知识，又起到正则化的作用。（在训练集上 C 是 2.867，反而不如 A "背得熟"。）
- **B 也明显好于 A**（3.362）。学生没见过真实文本，只抄了教师写的 2 万字符。教师写的是"莎士比亚腔的胡话"（`02` 会打印一段）。但这段文本的统计规律比 2 万字符的真实样本更平滑、更有代表性。
- **D 介于两者之间。** 混入硬标签后，D 比纯 KD 略差。在这个玩具设置里，真实标签只有 2 万字符，所以教师反而更可靠。有时教师不如数据可靠（例如教师本身会犯错）。这时要调大硬标签的比重。
- **R** 说明这个容量的学生的上限在 3.14 左右。蒸馏把"只有 2 万字符"的学生从 3.68 拉到了离"数据管够"只差约 0.15 bit 的地方。

Hinton 等人 2015 第 6 节在语音模型上做过同样的实验。只用 3% 的数据时，硬标签严重过拟合：需要提前停止，准确率 44.5%。软标签不用提前停止，就收敛到 57%。

这也和 Gemma 2 的消融实验方向一致。2B 模型训练 5000 亿 token，从头训练时 3 个基准平均 60.3。用 7B 教师蒸馏时是 67.7。**不要把玩具数字外推到大模型。** 这里的学生是字符级 MLP。实验只说明一点："在数据受限时，软标签比 one-hot 标签多带信息"。

## 4. 前向 KL 与反向 KL，以及在线策略蒸馏

### 4.1 同一个 KL，两个方向

上面两种蒸馏都在最小化**前向 KL**（forward KL）`KL(p_T ‖ p_S)`，期望在**教师**的分布下取。KL 不对称。**反向 KL**（reverse KL）`KL(p_S ‖ p_T) = Σ p_S·(log p_S − log p_T)` 的期望在**学生**的分布下取。有时学生的容量不够，装不下教师的全部行为。这时两个方向的结果差别很大。

`03_forward_reverse_kl.py` 用一个一维的例子说明：教师是双峰分布 `0.6·N(−2, 0.6²) + 0.4·N(2.5, 0.8²)`，学生只能是**一个**高斯分布。

| 最小化 | 起点 μ₀ | 收敛到 μ | σ | 左峰区质量 | 山谷 [−0.5, 1] 质量 | 右峰区质量 |
|---|---:|---:|---:|---:|---:|---:|
| （教师 p） | | | | 0.596 | 0.016 | 0.388 |
| 前向 KL(p ‖ q) | 0.0 | −0.22 | 2.43 | 0.453 | **0.240** | 0.307 |
| 反向 KL(q ‖ p) | −1.0 | −2.00 | 0.60 | **0.994** | 0.006 | 0.000 |
| 反向 KL(q ‖ p) | 1.5 | 2.49 | 0.81 | 0.000 | 0.033 | **0.967** |

- **前向 KL "覆盖所有模式"（mode covering）。** 在 p > 0 的地方，q 必须给概率。否则 `log q → −∞`，惩罚无穷大。所以学生只好摊开，把两个峰都盖住。代价是：它在两峰之间的山谷里放了 24% 的概率。教师几乎不去那里（教师在那里只有 1.6%）。对语言模型来说，这就是"两种说法都学一点，结果说出一句哪种都不是的话"。
- **反向 KL "挑一个模式"（mode seeking）。** q 在 p ≈ 0 的地方给概率，会被重罚。q 在 p 大的地方给不给概率都可以。学生缩进一个峰里，整个放弃另一个峰。落到哪个峰，取决于起点。

### 4.2 在线策略蒸馏：学生自己写，教师逐 token 打分

反向 KL 的期望在学生分布下取。要计算它，**学生必须自己采样**。学生生成一段回答。教师**在学生写的每个 token 上**给出自己的对数概率。训练在每个位置最小化 `log p_S(y_t) − log p_T(y_t)`。这就是**在线策略蒸馏**（on-policy distillation；Agarwal 等人 2023 的 GKD、Gu 等人 2023 的 MiniLLM）。

在线策略蒸馏解决了序列级蒸馏的一个老问题。学生只在教师走过的路上练过。一旦学生早早走错一步，就到了教师从没去过的地方，错误越滚越大（**暴露偏差**，exposure bias）。在线策略蒸馏让学生**在自己会犯错的地方**得到逐 token 的纠正。

Thinking Machines 的博客（Lu 等，2025）把三种方法放在一张表里。SFT 是离线 + 稠密信号。RL 是在线 + 稀疏信号（一整段回答只有一个奖励）。在线策略蒸馏是**在线 + 稠密信号**。实现上可以直接复用 RL 代码，把每个 token 的优势换成 `−(log p_S − log p_T)`。第 19 章讲 GRPO 时会再见到这个结构。

它有多大用？Qwen3 报告的表 21 给出了答案。所有行都从同一个离线蒸馏后的 8B 模型出发：

| Qwen3-8B | AIME'24 | AIME'25 | MATH500 | LiveCodeBench v5 | GPU 时 |
|---|---:|---:|---:|---:|---:|
| 离线蒸馏后 | 55.0 | 42.8 | 92.4 | 42.0 | — |
| + 强化学习 | 67.6 | 55.5 | 94.8 | 52.9 | 17,920 |
| + 在线策略蒸馏 | 74.4 | 65.5 | 97.0 | 60.3 | 1,800 |

只用约十分之一的 GPU 时，所有成绩都高于直接做 RL。

**共识判断（GOAL.md 2.1）**：GOAL 把在线策略蒸馏列为"待核实"。我们逐项核对了技术报告。至少五个彼此独立的头部家族在主力版本里明确采用了它：

- **Qwen3**：小模型强到弱蒸馏的第二阶段（§4.5）。
- **Gemma 2**：SFT 阶段"在学生的分布上从教师蒸馏"，引用了 GKD 和 MiniLLM（§4）。
- **GLM-5**：后训练的最后一步"在线策略跨阶段蒸馏"（§3.5）。
- **小米 MiMo-V2-Flash**：多教师在线策略蒸馏 MOPD（§4）。
- **DeepSeek-V4**：用在线策略蒸馏整个替换了混合 RL 阶段，十多个领域专家当教师（§5.1）。

这满足规则 A，所以在线策略蒸馏**进正文**。

**主线不能用外部教师做它。** 在线策略蒸馏要教师在学生的 token 上打分。和 logits 蒸馏一样，它要求同一个分词器，而我们自训的分词器和所有开源教师都不同。跨分词器的在线策略蒸馏还在研究阶段（见"前沿观察"）。

**主线用的是 GLM-5 的形式：跨阶段自蒸馏**（2026-10-10 确定）。教师是主线**自己**前几个阶段的 checkpoint，分词器天然相同。GRPO 之后，学生在两个提示词池上采样：工具调用任务由 GRPO 的 checkpoint 打分，对话提示词由蒸馏后的 checkpoint 打分。这样学回被 RL 削弱的能力，又不丢掉工具调用（`zero/post/opd.py`，后训练的最后一步；`configs/main/opd.toml`）。`zero/post/distill.py` 里原来的 `on_policy_distill`（单个教师，接在离线蒸馏之后，`on_policy_steps`）留给极小配置演示。

## 5. 拒绝采样：多采几个，只留验证过的

### 5.1 教师也会错

教师不是完美的。它写的工具调用可能参数写错、选错工具、JSON 写坏。它也可能不调工具，直接编一个答案。序列级蒸馏会把这些错误原样教给学生。

几乎所有做教师数据的团队都加了一道**拒绝采样**（rejection sampling）。做法是让教师对同一个问题采样多次，只保留**能被自动验证为正确**的那些。

- **DeepSeek-R1**：推理数据"对每个提示词采样多个回答，只保留正确的"。这一步得到约 60 万条。加上约 20 万条非推理数据，共约 80 万条。R1-Distill 用的就是这批数据。
- **Qwen3**：冷启动数据由 QwQ-32B 对每个问题生成 N 个候选。过滤器去掉六类回答，例如最终答案错误、大量重复、明显靠猜。
- **GLM-5**：推理数据"用拒绝采样合成"。代码与智能体数据先在大量可执行环境里跑出轨迹，再筛选。
- **Llama 3.2**：每一轮对齐都是 SFT → 拒绝采样 → DPO。

工具调用天然适合这样做：调用对不对，**执行一下就知道**。

### 5.2 漏斗

`04_rejection_sampling.py` 在 `zero/post/envs/tool_env.py` 的 200 个训练任务上演示这道筛子。这里的教师是一个**规则模拟器**，不是语言模型。它按固定概率犯几类常见错误。（概率是随手设的，只为展示漏斗。）筛选用的是生产级函数 `zero.post.distill.teacher_trajectories`：

1. 第一轮调用必须拿满分：函数名和规范化后的参数都与标准调用一致。
2. 然后筛选器**真的执行工具**，把结果喂回去。
3. 教师写最终回答。回答必须通过 `score_final_answer`：要点都在，也没有罗列一堆数字碰运气。

| 关卡 | 剩下 | 占比 |
|---|---:|---:|
| 候选（200 个任务 × 8 次） | 1600 | 100% |
| 关 1：格式正确（JSON 完整、标签配对、没伪造工具结果） | 1303 | 81.4% |
| 关 2：调用正确（函数名 + 规范化后的参数；计算器另接受执行结果一致） | 596 | 37.2% |
| 关 3：执行 + 最终回答正确 | 497 | 31.1% |
| 关 4：每个任务最多留 1 条 | 172 | （172/200 个任务有数据） |

按错误类型看：参数写错的 291 条里有 43 条通过。选错工具、JSON 写坏、不调工具直接答、多调一个、胡话，全部被筛掉。不筛的话，68.9% 的候选没有通过验证，其中绝大多数是错的。

**数据宁可少，不可错。** 保留的 172 条都经过了执行验证。它们比混着一大半错误数据的 1600 条有用得多。

另外 28 个任务一条都没留下，其中 21 个是"不需要工具"的任务。原因见 5.3。

再看"参数写错却通过"的 43 条。24 条是 weather_compare，19 条是 weather。它们把"北京"写成"北京市"之类。规范化后与标准参数相同，所以算对是合理的。

判分器修复之前，还有 4 条 weekday 也通过了。它们的日期正好改了 7 天，星期几没变。**答案对、参数错**，只比较执行结果查不出来。现在，除计算器外，判分器一律比较规范化后的参数（`args_equivalent`）。所以这 4 条被筛掉了。验证器只能保证它检查的东西。

### 5.3 验证器的漏洞是真实观察到的

修复判分器之前，这段输出更重要。保留的数据里有 21 条来自"不需要工具"的任务（打招呼、说句鼓励的话）。其中 9 条回答是胡话。例如用户说"讲一句鼓励的话"，助手答"坚下云，气温 28°C。"。**这条回答照样通过了验证。**

原因是：`score_final_answer` 只检查"该出现的要点都出现了"，而这类任务没有要点；`score_tool_calls` 只检查"没有乱调工具"。

这不是编出来的例子。**冒烟测试里唯一通过验证的那条教师数据，就是这一条。** 它在当时的 `out/smoke/distill/teacher.jsonl` 里。摘录在 `video/data/smoke_before_fix.json`（见"主线进度"）。

现在的修法在 `zero/post/envs/tool_env.py`。对不需要工具的任务，正常回答只得 `NO_TOOL_REWARD = 0.5`，并标记为"无法核对"。它不算验证通过，所以不进蒸馏数据（上面 21 个任务保留 0 条）。回答里出现题目中没有的数字，直接得 0 分。

代价是蒸馏数据里没有"该不调就不调"的示范。这部分交给人工审过的 SFT 数据（第 16 章）。以后如果想让教师也生成这类数据，需要一个真正的内容判分（规则或评审模型），并在报告里单列这一类的通过率。

## 6. 教师的许可证

GOAL.md 3.3 的规则：**只用许可证允许"用输出训练其他模型"的开放权重模型当教师**，并记录教师的名称、版本、许可证。

蒸馏数据会和我们的模型一起发布。所以许可证里关于"输出"和"衍生模型"的条款会传到我们身上。2026 年 9 月，我们逐一读了许可证原文。（下表"条款"一列是原文要点，不是法律意见。）

| 教师 | 许可证 | 与"用输出训练其他模型"相关的条款 | 能当主线教师？ |
|---|---|---|---|
| Qwen3 全系列、Qwen3.5 全系列 | Apache-2.0 | 无额外限制（Qwen3 报告："all Qwen3 models are publicly accessible under Apache 2.0"） | ✅ |
| DeepSeek-R1 | MIT | README 明确写着："support commercial use, allow for any modifications and derivative works, including, but not limited to, **distillation for training other LLMs**" | ✅ |
| DeepSeek-V4（Flash / Pro） | MIT | 无额外限制 | ✅ |
| GLM-5 | MIT | 无额外限制 | ✅ |
| MiMo-V2-Flash | MIT | 无额外限制 | ✅ |
| gpt-oss-20b / 120b | Apache-2.0 | 另附一句使用政策：使用须遵守适用法律 | ✅ |
| Gemma 4 | Apache-2.0（Hugging Face 模型卡的 license 字段） | 条款全文页 `gemma_4_license` 在本环境打不开，全文待核实 | 待核实 |
| Gemma 1–3 | Gemma Terms of Use | 用 Gemma 生成的合成数据训练出的模型属于 **"Model Derivatives"**。分发时必须带上使用限制条款，并附上协议全文 | ❌（限制会传给衍生模型，不是 Apache/MIT） |
| Llama 3.1 / 3.2 / 3.3 / 4 | Llama Community License | 允许，但"用 Llama 或其**输出**训练并发布的模型，名字必须以 'Llama' 开头"。还要标注 "Built with Llama"，遵守可接受使用政策。月活超过 7 亿需要另行授权 | ❌（不是 Apache/MIT，且有命名要求） |
| Llama 2、Meta Llama 3（2024-04） | 上面许可证的早期版本 | "You will not use the Llama Materials or any output or results of the Llama Materials to **improve any other large language model**" | ❌（明确禁止） |

两个容易踩的坑：

1. **学生继承的是底座的许可证。** DeepSeek-R1 本身是 MIT。但 DeepSeek-R1-Distill-Llama-8B / 70B 的底座是 Llama 3.1 / 3.3。所以它们受 Llama 许可证约束（R1 的 README 写明了这一点）。拿它们当教师，等于间接用了 Llama 的输出。
2. **许可证会变。** Llama 2 禁止用输出改进其他模型。从 Llama 3.1 起，规则改成"可以，但名字要以 Llama 开头"。每次选定教师，都要按**具体版本**重新读原文。还要把许可证写进蒸馏数据的元数据（`zero` 会检查，见下一节）。

## 7. 小结

- **软标签比 one-hot 标签多带信息**：错误答案之间的相对大小（暗知识）。温度 τ 把它放大。损失乘 τ²，保持梯度的大小。
- **logits 蒸馏**：`τ²·KL(p_T^τ ‖ p_S^τ)`，梯度是 `τ·(p_S^τ − p_T^τ)`。它带的信息最多，但**要求同一个词表**。
- **序列级蒸馏**：教师写，学生做 SFT，等价于序列级的前向 KL。它对教师只要求能生成文本。主线模型有自己的分词器，所以只能走这条路。
- **前向 KL 覆盖所有模式，反向 KL 挑一个模式。** **在线策略蒸馏**让学生自己采样，教师逐 token 打分。Qwen3、Gemma 2、GLM-5、MiMo、DeepSeek-V4 都采用了它。它同样要求同一个词表，所以主线用自己前几个阶段的 checkpoint 当教师（跨阶段，GLM-5 的形式）。
- **拒绝采样 + 执行验证**：多次采样，逐关筛选。数据质量比数量更重要。验证器只能保证它检查的东西。
- **教师许可证**：Apache-2.0 / MIT 可用。Gemma 1–3、Llama 系列的条款会传到学生身上，我们不用。

---

## 从极简代码到生产级代码

| 极简代码（`code/`） | 生产级代码（`zero/`、`configs/`、`tests/`） | 多做了什么、为什么 |
|---|---|---|
| `01` 的 `kd_loss`（单个位置、NumPy） | `zero/post/distill.py`：`kd_loss(student_logits, teacher_logits, mask, temperature, topk)` | 批量形状 (B, T, V)，只在 mask（助手回答）位置上平均。`topk > 0` 时只用教师的前 k 个 token（离线存教师 logits 时省存储）。形状不同时直接报错，并提示"要同一个分词器" |
| `02` 的 C、D 组（`(1−α)·CE + α·KD`） | `DistillTrainer._forward_loss`（`_make_distill_trainer_cls`）：复用通用的 `Trainer`，只改损失。教师冻结，在 `no_grad` 下运行。`extra_metrics` 记录 `ce`、`kd` | 与预训练 / SFT 共用断点续训、日志、BF16、多卡（**多卡尚未在 GPU 上验证**）。`kd_alpha`、`kd_temperature`、`kd_topk` 由配置决定 |
| `02` 的 B 组（教师写、学生抄） | `LocalTeacher`（本地 zero checkpoint 或 Qwen3 结构的 HF 目录，带 KV cache 采样）、`OpenAITeacher`（任何 OpenAI 兼容接口；第二步用 vLLM 启动教师服务；只用标准库 `urllib`）。`build_teacher` 按 `[teacher] backend` 选择 | 教师可以是任何分词器、任何架构，也可以是远程服务。`OpenAITeacher` 负责在我们的工具调用格式和 OpenAI 的 `tool_calls` 之间互相转换（`to_openai_messages`、`openai_message_to_text`） |
| `03` 的反向 KL | `zero/post/opd.py`：`run_opd`，跨阶段在线策略蒸馏。多个教师（自己前几个阶段的 checkpoint），各有提示词池和权重；`loss = "full_kl"`（在整个词表上的精确反向 KL，只算回答位置）或 `"sampled"`（GLM-5 的写法，每个 token 的优势为 `log p_T − log p_S`） | 检查每个教师的分词器哈希。两种损失的梯度期望相同（`tests/test_opd.py` 用穷举验证）。可用 torchrun 多卡数据并行。`distill.py` 里的 `reverse_kl_loss` / `on_policy_distill` 是单教师版本（默认关闭，`on_policy_steps = 0`） |
| 无 | `generate_kd_data` 的 `task_files`（`zero/post/envs/fc_tasks.py` 的真实工具调用任务：只留教师判分完全正确的第一轮）和 `prompt_files`（没有答案的提示词，例如中文指令：丢掉空回答、伪造轮次、工具调用、语言不符的回答） | 用真实数据，不再只有玩具环境；`prompt_files` 是用许可证允许的教师生成中文 SFT 数据的路子。`concurrency` 向教师服务并发请求，输出顺序不受影响 |
| `04` 的漏斗 | `teacher_trajectories`（采样 n 个 → `score_tool_calls == 1` → `execute_safely` 真的执行 → 教师写最终回答 → `score_final_answer`）。`generate_kd_data` 写 `teacher.jsonl` 和 `.meta.json` | 每条数据和元数据都记录**教师的名称、版本、许可证**，以及采样参数、候选数、通过率、筛选规则。`check_license`：非本项目的模型当教师时，配置里必须写 `license_allows_distillation = true`，否则拒绝运行 |
| `05` 的词表检查 | `run_distill` 在生成数据**之前**比较教师与学生分词器的哈希。不同就报错，并提示设 `logits_kd = false` | 不要等教师数据生成完，才发现不能做 logits 蒸馏 |
| 无 | `mix_sft_jsonl` / `mix_sft_max`：把原 SFT 数据混进蒸馏数据 | 教师数据少时，防止遗忘第 16 章学到的东西 |
| 无 | 用 torchrun 运行 `run_distill`：rank 0 生成并打包教师数据，然后所有 rank 用 SFT 的 `Trainer` 训练（DDP） | 学生的多卡训练；用 2 个 CPU 进程测过（`tests/test_post_ddp.py`），尚未在 GPU 上运行 |

**对拍**：

- `01` 的第 4 部分：极简版 KD 损失与 `zero.post.distill.kd_loss` 在 τ = 1、2 和 top-3 下相差 ≤ 3×10⁻⁷（float32 舍入）。
- [`tests/test_distill.py`](../../tests/test_distill.py)（8 项；另有 1 项需要 CUDA）：
  - KD 损失、温度、top-1、反向 KL 与手算一致。
  - mask 外的位置不影响损失。
  - 形状不同时报错。
  - 许可证守卫有效。
  - OpenAI 消息格式能双向转换。
  - **本地假 OpenAI 服务器**当教师。每个任务给一对一错两个回答。执行验证筛掉错的，所以通过率正好 50%，元数据字段完整。
  - 本地自蒸馏能跑通。学生与教师初始相同，所以第一步 KL 为 0。在线策略蒸馏跑 1 步。
  - 分词器不同时报错。
  - 按脚本作答的假教师回答真实工具调用任务和提示词：只留完全正确的第一轮；空回答、伪造轮次、语言不符的回答被丢掉；并发 1 和 4 写出的文件完全相同。
- [`tests/test_opd.py`](../../tests/test_opd.py)（9 项）：sampled 损失的梯度期望等于反向 KL 的梯度（穷举验证）；学生自己当教师时 KL 为 0；两个教师、两种损失端到端跑通；分词器不同的教师被拒绝。

```bash
uv run pytest tests/test_distill.py tests/test_opd.py -q     # 17 passed, 1 skipped (3.9 s on the build machine)
```

**主线配置**（`configs/main/distill.toml`；第二步的默认值，待调；**尚未在 GPU 上验证**）：

| 设置 | 值 | 为什么 |
|---|---|---|
| `[teacher] backend` | `openai`，`base_url = http://localhost:8000/v1` | 教师用 vLLM 启动 OpenAI 兼容服务（`vllm serve <teacher> --enable-auto-tool-choice --tool-call-parser hermes`），和训练解耦 |
| `[teacher] name / version / license` | 待定 / 待定 / 待核实；`license_allows_distillation = false` | 没核实许可证就拒绝运行（GOAL.md 3.3） |
| `[distill] logits_kd` | `false` | 自训的分词器与任何开源教师都不同，只能做序列级蒸馏 |
| `samples_per_task`、`keep_per_task` | 4、1 | 拒绝采样：每个任务采 4 次，最多留 1 条 |
| `n_tasks` | 200,000 | `tool_env`：玩具环境，端到端执行并验证（含工具结果的多轮） |
| `task_files`、`prompt_files`、`concurrency` | 空、空、64 | 真实工具调用任务、没有答案的提示词（例如中文指令）；向 vLLM 服务并发 64 个请求。用哪些文件见 `runs/POSTTRAIN_PLAN.zh.md` 第 6.2、6.4 节 |
| `mix_sft_jsonl`、`mix_sft_max` | `data/sft/train.jsonl`、200,000 | 蒸馏数据和 SFT 数据混合，防止遗忘 |
| `init_from`、`lr`、`max_steps` | SFT 的 checkpoint、3e-5、1500 | 从 SFT 模型出发的一小段续训 |
| `on_policy_steps` | 0 | 主线把在线策略蒸馏作为单独的最后一步，教师是自己的 checkpoint（`configs/main/opd.toml`） |

## 主线进度

### 极小配置演示（CPU，`configs/tiny`，约 1.3M 参数）

> **注意：**本节数字来自修复工具调用判分器（第 19 章第 6 节）**之前**的那次冒烟测试，是当时的真实输出。修复后我们重跑了冒烟测试（`uv run python -m zero.smoke --out out/smoke_final`）。数据、预训练、中期训练、SFT 各阶段的结果完全一致。
>
> 蒸馏中通过验证的样本从 1 条（那句胡话）变为 0 条。下游的 DPO、GRPO 和评测数字随之变化。例如，同一个 SFT 模型的工具调用 call_exact 从 0.133 变为 0.100。模型没变，是判分更严了。GRPO 相对 SFT 仍判"持平"。你自己运行时，以运行结果为准。

> 以下是**极小配置演示**：只说明代码能跑通，不代表主线模型的任何结果。数字来自 `uv run python -m zero.smoke` 的真实输出（`out/smoke/SUMMARY.md`、`out/smoke/distill/`）。本章没有重跑。

**教师是替身。** 这个环境访问不了 huggingface.co，下载不了任何开源权重。所以冒烟测试用 **tiny SFT 模型自己**当教师（自蒸馏 / 拒绝采样微调）。这只验证"教师采样 → 执行验证 → 打包 → 序列级 + logits 蒸馏"这条通路，不说明蒸馏的效果。

| 项目 | 数值 |
|---|---|
| 教师 | `zero-tiny-sft (self-distillation stand-in)`，许可证 "same as this repository"（本项目自己的模型，`check_license` 放行） |
| 任务 × 采样 | 24 个 tool_env 训练任务 × 4 次 = **96 个候选** |
| 通过执行验证 | **1 条**（通过率 1.0%），用时 15.5 秒 |
| 混入 SFT 数据 | 200 条 → 共 201 条，打包成 190 个窗口、9,011 个目标 token |
| 训练 | 20 步，`kd_alpha = 0.5`，τ = 1 |
| 第 1 步 | loss 0.218 = 0.5 × ce 0.437 + 0.5 × kd **0.000**（学生与教师是同一个模型） |
| 第 20 步 | loss **0.269** = 0.5 × ce **0.526** + 0.5 × kd **0.0116** |

对这些结果的说明：

- tiny 模型几乎不会调用工具。96 个候选只有 1 个通过，符合预期。
- **唯一通过的那一条，就是第 5.3 节的假阳性**："讲一句鼓励的话。" → "坚下云，气温 28°C。"。当时的判分器不检查不需要工具的任务的回答内容。这提醒我们：验证器的漏洞在真实流水线里会被命中。这个漏洞随后已修复（见 5.3）。修复后，这一条不会再通过。上表是修复前那次运行的记录。
- 这里的 KD 项是学生对"冻结的 SFT 模型"的 KL。它从 0 开始，慢慢变大。它的作用相当于一个"别离原模型太远"的约束，不是向更强的教师学习。

### 待 GPU 训练后补充

- 教师的最终选择、版本与许可证核实记录。
- 真实的教师数据生成：候选数、各关通过率、按任务类型的通过率、花费。不需要工具的任务单列。
- 去污染：蒸馏数据与全部评测集的 13-gram 检查。还有函数名和 schema 与 BFCL 等评测集的重合检查（GOAL.md 3.2）。
- 蒸馏训练曲线、蒸馏前后在我们自己的开发集上的工具调用成绩、失败与返工。

**第二步的教师方案（待定）**：

1. 从许可证为 Apache-2.0 或 MIT 的开放权重模型里选教师。候选（许可证于 2026-10-10 读自模型元数据）：Qwen3.5-35B-A3B、Qwen3.5-122B-A10B（Apache-2.0），Qwen3-235B-A22B-Instruct-2507（Apache-2.0），gpt-oss-120b（Apache-2.0；OpenAI 另有使用政策），DeepSeek-V4.1-Flash（MIT），GLM-5.2（MIT，中英文）。按工具调用能力、中文能力和推理成本挑，**具体型号由项目负责人定**。可以由多个教师分别生成数据，每条数据记录来自哪个教师。
2. 用 vLLM 在单独的 GPU 上启动 OpenAI 兼容服务。在 `configs/main/distill.toml` 的 `[teacher]` 里填上名称、版本、许可证，并设 `license_allows_distillation = true`。
3. 在真实任务（`task_files`：第一轮判分完全正确）和中文提示词（`prompt_files`）上生成数据。真实工具上的多步轨迹需要可执行的工具，目前只有玩具环境有。给不需要工具的任务补上内容判分。
4. 做序列级蒸馏（`logits_kd = false`），并混入 SFT 数据。
5. 教师数据生成与蒸馏训练的花费计入第 16–19 章的后训练预算（GOAL.md 3.4：约 $1,500，含教师数据生成）。单次预计超过 $100 的运行，先报批。

---

## 前沿观察

> **剪枝 + 蒸馏**：先把大模型剪小（删层、删注意力头、缩窄 FFN），再用原模型当教师，用蒸馏恢复质量。Llama 3.2 的 1B、3B 就是这样做的："从 8B 剪枝，再用 8B 和 70B 的 logits 蒸馏"。NVIDIA 的 Minitron（Nemotron 系列）系统地研究了这条路。明确采用它的头部家族，目前只核实到两家。主线也是从零训练。所以本章不讲。
>
> **跨分词器蒸馏**：教师和学生用不同的分词器，也能做 logits / 在线策略蒸馏。（例如按字节前缀把两边的概率对齐。）如果这个方法成熟，主线就能从千问教师做在线策略蒸馏，而不必换词表。目前只有研究论文（如 arXiv 2607.22334），没有头部家族在主力版本中采用。
>
> **多教师在线策略蒸馏合并领域专家**：MiMo-V2-Flash、DeepSeek-V4 先训练一批领域专家（数学、代码、智能体……）。然后用在线策略蒸馏把它们合进一个学生。GLM-5 用它在多段 RL 之后找回被遗忘的能力。这是在线策略蒸馏（正文）的一种用法。主线采用 GLM-5 的跨阶段形式（教师是自己前几个阶段的 checkpoint，`zero/post/opd.py`）；预算不够训练一批领域专家，所以不做专家合并。

---

## 采用方与来源

| 技术 | 采用方（主力版本） | 来源 |
|---|---|---|
| logits 蒸馏（逐 token 软标签） | **Llama 3.2** 1B/3B（Llama 3.1 8B、70B 的 logits 作为逐 token 目标）；**Gemma 2** 2B/9B（预训练用蒸馏代替下一个 token 预测）；**Gemma 3** 全系列（每个 token 按教师概率采样 256 个 logits）；**Qwen3** 小模型（在线阶段对齐教师 logits）；**DeepSeek-V4**（全词表 logits 的在线策略蒸馏） | [Llama 3.2 模型卡](https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/MODEL_CARD.md)、[Meta 博客](https://ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices/)（"structured pruning in a single shot manner from the Llama 3.1 8B"）；[Gemma 2](https://arxiv.org/abs/2408.00118) §3.2、§5；[Gemma 3](https://arxiv.org/abs/2503.19786) §2.2；[Qwen3](https://arxiv.org/abs/2505.09388) §4、§4.5；[DeepSeek-V4](https://arxiv.org/abs/2606.19348) §5.1.2 |
| 序列级蒸馏（教师数据 SFT） | **DeepSeek-R1-Distill**（约 80 万条 R1 数据，只做 SFT）；**Qwen3** 小模型（离线蒸馏阶段）；**Gemma 2**（SFT 回答"主要由更大的教师合成"） | [DeepSeek-R1](https://arxiv.org/abs/2501.12948) 附录 B.3.3、B.4.3、F；[Qwen3](https://arxiv.org/abs/2505.09388) §4.5；[Gemma 2](https://arxiv.org/abs/2408.00118) §4 |
| 拒绝采样 / 验证后保留 | **DeepSeek-R1**（采样多个，只留正确的）；**Qwen3**（QwQ-32B 生成 N 个候选，六类过滤）；**GLM-5**（拒绝采样 + 可执行环境）；**Llama 3.2**（每轮 SFT → 拒绝采样 → DPO） | [DeepSeek-R1](https://arxiv.org/abs/2501.12948) B.3.3；[Qwen3](https://arxiv.org/abs/2505.09388) §4.1；[GLM-5](https://arxiv.org/abs/2602.15763) §3.1；[Llama 3.2 模型卡](https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/MODEL_CARD.md) |
| 在线策略蒸馏 | **Qwen3**（§4.5、表 21）；**Gemma 2**（"distillation from the teacher on the student's distribution"，§4）；**GLM-5**（在线策略跨阶段蒸馏，§3.5）；**MiMo-V2-Flash**（MOPD，§4.1、§4.4）；**DeepSeek-V4**（OPD 替换混合 RL，§5.1） | [Qwen3](https://arxiv.org/abs/2505.09388)；[Gemma 2](https://arxiv.org/abs/2408.00118)；[GLM-5](https://arxiv.org/abs/2602.15763)；[MiMo-V2-Flash](https://arxiv.org/abs/2601.02780)；[DeepSeek-V4](https://arxiv.org/abs/2606.19348)；方法：[GKD](https://arxiv.org/abs/2306.13649)、[MiniLLM](https://arxiv.org/abs/2306.08543)、[Thinking Machines 博客](https://thinkingmachines.ai/blog/on-policy-distillation/) |
| 教师数据生成服务（vLLM，OpenAI 兼容接口） | 行业标准工具（GOAL.md 2.1 规则 B） | [vLLM](https://github.com/vllm-project/vllm) |

**共识判断（GOAL.md 2.1）**：我们核对了四项技术：logits 蒸馏、序列级蒸馏、拒绝采样、在线策略蒸馏。每一项都有 3 个以上彼此独立的头部家族在主力版本中明确采用，所以都进正文。GOAL 表里写的是"在线策略蒸馏（待核实）"。经核实，它**已达共识**。剪枝 + 蒸馏、跨分词器蒸馏放在"前沿观察"。

**许可证来源**（2026-09 读取）：
[Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B)、[Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B/blob/main/LICENSE)、[Qwen3.5-397B-A17B](https://huggingface.co/Qwen/Qwen3.5-397B-A17B)、
[DeepSeek-R1 README §7](https://github.com/deepseek-ai/DeepSeek-R1#7-license)、[DeepSeek-V4-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash)、
[GLM-5](https://huggingface.co/zai-org/GLM-5)、[MiMo-V2-Flash](https://huggingface.co/XiaomiMiMo/MiMo-V2-Flash)、
[gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b)（及其 `USAGE_POLICY`）、[Gemma 4 E2B](https://huggingface.co/google/gemma-4-E2B-it)、
[Gemma Terms of Use](https://ai.google.dev/gemma/terms)（1.1(e) "Model Derivatives"）、
[Llama 3.2 License](https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/LICENSE)、[Llama 4 License](https://github.com/meta-llama/llama-models/blob/main/models/llama4/LICENSE)、
[Meta Llama 3 License](https://github.com/meta-llama/llama-models/blob/main/models/llama3/LICENSE)、[Llama 2 License](https://github.com/meta-llama/llama-models/blob/main/models/llama2/LICENSE)。

---

## 引导问题

向 Claude Code 提出这些问题。一直问到你能用自己的话讲清楚答案：

1. `02` 里，C（logits 蒸馏）在自己那 2 万字符上的损失（2.867）比 A（2.678）还高，验证集上却好得多。软标签在这里起了什么作用？如果把学生的训练数据从 2 万字符加到 100 万字符，你预期 A 和 C 的差距会怎样变化？
2. 序列级蒸馏等价于序列级的前向 KL。那为什么说它"只带一条路的信息"？如果每个提示词让教师采样 16 条回答，全部拿来训练，它会更接近 logits 蒸馏吗？代价是什么？
3. 对语言模型来说，"mode covering"的坏处具体是什么样子？举一个工具调用的例子：一个调用有两种都正确的写法。学生"各学一半"，会写出什么？反向 KL 又有什么风险（提示：多样性）？
4. 在线策略蒸馏的"奖励"是 `log p_T − log p_S`。为什么它很难被 hack？如果教师自己在某类问题上是错的，学生会怎样？（对照第 19 章的 reward hacking。）
5. 第 5.2 节的 weekday 任务里，有一个调用的日期错了 7 天，星期几却是对的。这个调用通过了执行验证。你会怎样改验证器？改得太严，又会误杀哪些正确的样本？
6. 设你负责第二步选教师。一个是许可证为 Apache-2.0 的中等模型。另一个工具调用更强，但许可证要求"衍生模型的名字以它的名字开头"。你怎么选？需要记录哪些信息，别人日后才能复查这个决定？

## 动手任务

每个任务都要运行代码，并查看结果。

**任务 1（基础）**：在 `01_soft_labels.py` 里，把教师 logits 改成一个"极度自信"的分布（例如 `[10, 1, 0, 0, -5, -5]`）。重新看 τ = 1、2、4 时的熵和 `p(晴)/p(猫)`。τ 多大时，"次优答案"才开始有可见的概率？

**任务 2（核心）**：在 `02_toy_distill.py` 里做两组扫描。(a) 把 `TAU` 换成 1、4，看 C 组的验证集结果怎样变化。(b) 把 `SMALL_N` 换成 5 千和 10 万，看 A 与 C 的差距怎样变化。写一句话结论："学生数据越____，蒸馏的好处越____"。（每次 CPU 时间约 2 分钟。）

**任务 3（挑战）**：给 `tool_env` 里"不需要工具"的任务写一个内容判分。例如：回答里不能出现数字和城市名，长度在 2–40 字之间；或者回答与 `gold_answer` 的字符重合率超过阈值。在 `04_rejection_sampling.py` 里用它替换 `score_final_answer`。看 9 条胡话还剩几条，有没有误杀正常回答。然后思考：这条规则放到真实教师上，会不会误杀大量合理的问候语？

---

## 想深入：CS336

本章对应斯坦福 CS336（Spring 2026）<https://cs336.stanford.edu/>，**部分覆盖**：

- **第 14 讲：数据（过滤、去重、配比、合成数据）**。其中合成数据的部分，是本章"让教师写数据、筛过再用"的背景。作业 4（Data）的过滤流水线可以直接用在教师数据上。
- **第 15 讲：中期训练与后训练（SFT/RLHF）**。讲 SFT 数据从哪里来。教师数据蒸馏是其中最常见的一类。
- logits 蒸馏的温度与梯度、前向 / 反向 KL、在线策略蒸馏、教师许可证，CS336 没有深入讲。见本章参考文献。

---

## 本章参考文献

- Hinton, Vinyals, Dean. *Distilling the Knowledge in a Neural Network*（软标签、温度、τ² 缩放），2015：<https://arxiv.org/abs/1503.02531>
- Kim, Rush. *Sequence-Level Knowledge Distillation*，2016：<https://arxiv.org/abs/1606.07947>
- Agarwal et al. *On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes*（GKD），2023：<https://arxiv.org/abs/2306.13649>
- Gu et al. *MiniLLM: Knowledge Distillation of Large Language Models*（反向 KL），2023：<https://arxiv.org/abs/2306.08543>
- Lu, Thinking Machines Lab. *On-Policy Distillation*（博客），2025-10：<https://thinkingmachines.ai/blog/on-policy-distillation/>
- Gemma Team. *Gemma 2: Improving Open Language Models at a Practical Size*（§3.2 蒸馏、§5 消融），2024：<https://arxiv.org/abs/2408.00118>
- Gemma Team. *Gemma 3 Technical Report*（§2.2 采样 256 个 logits、§3 后训练），2025：<https://arxiv.org/abs/2503.19786>
- Meta. *Llama 3.2 模型卡*（剪枝 + logits 蒸馏）：<https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/MODEL_CARD.md>
- Meta. *Llama 3.2: Revolutionizing edge AI and vision with open, customizable models*（博客，"Lightweight models" 一节），2024-09：<https://ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices/>
- Qwen Team. *Qwen3 Technical Report*（§4.5 强到弱蒸馏、表 21），2025：<https://arxiv.org/abs/2505.09388>
- DeepSeek-AI. *DeepSeek-R1*（附录 B.3.3 拒绝采样、B.4.3 与 F 蒸馏），2025：<https://arxiv.org/abs/2501.12948>
- GLM-5 Team. *GLM-5: from Vibe Coding to Agentic Engineering*（§3.5 在线策略跨阶段蒸馏），2026：<https://arxiv.org/abs/2602.15763>
- Xiaomi LLM-Core. *MiMo-V2-Flash Technical Report*（§4 MOPD），2026：<https://arxiv.org/abs/2601.02780>
- DeepSeek-AI. *DeepSeek-V4*（§5.1 专家训练 + 在线策略蒸馏），2026：<https://arxiv.org/abs/2606.19348>
- Muralidharan et al. *Compact Language Models via Pruning and Knowledge Distillation*（Minitron），2024：<https://arxiv.org/abs/2407.14679>
- 各教师的许可证原文：见"采用方与来源"末尾
- [CS336](https://cs336.stanford.edu/) 第 14、15 讲

**下一章**：蒸馏和 SFT 教会了模型"怎么答"，但它们只给正面例子。模型从没见过"这个回答比那个好在哪里"。第 18 章讲偏好对齐：先用 RLHF（奖励模型 + PPO）搭好框架，再从同一个目标推出更简单的 DPO。DPO 让主线模型在通用对话上更得体。
