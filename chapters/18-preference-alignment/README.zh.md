# 第 18 章：偏好对齐 —— 从 RLHF 到 DPO

[English](README.md) · **中文**

> **目标**：读完这一章，你能从"偏好数据 + Bradley–Terry 模型"出发，写出奖励模型的损失。你能讲清 RLHF 目标 `E[r] − β·KL` 里的 KL"缰绳"为什么必不可少。你能亲手把 RLHF 的最优解推成 DPO 损失，并用代码和 `zero/post/dpo.py` 对拍。你还知道怎样盯住 DPO 的几个坑：学习率、β、"chosen 的概率也会下降"。

📺 **本章视频**：还没有发布。要在本机渲染，运行 `bash chapters/18-preference-alignment/video/build.sh`。
🧪 **本章自检**：学完后，在 Claude Code 里输入 `/ch18-dpo`。

---

上一章我们用蒸馏（distillation）把教师的工具调用本领教给了小模型。先用 SFT 学格式，再学教师写的、通过执行验证的示范。这两步都是**模仿**：示范怎么写，模型就怎么写。

这一章解决另一个问题：**很多时候我们写不出"最好的答案"，却能用少得多的工夫判断"两个答案哪个更好"。**怎样把这种判断变成训练信号？

经典的答案是 **RLHF**（Reinforcement Learning from Human Feedback，基于人类反馈的强化学习）。先训练一个奖励模型给回答打分，再用 PPO 让模型去追高分。InstructGPT 就用了这个方法。

这一章先讲清 RLHF，因为它是第 19 章 GRPO 的铺垫（GOAL.md 2.1 节的规则 C）。然后从**同一个目标**出发，推导出今天开源模型最常用的偏好对齐方法：**DPO**（Direct Preference Optimization，直接偏好优化）。主线模型的第 6 个训练阶段就是 DPO。

本章代码（都在 CPU 上运行，`torch.set_num_threads(1)`）：

```bash
uv run python chapters/18-preference-alignment/code/01_bradley_terry.py    # Bradley–Terry reward model (a few seconds)
uv run python chapters/18-preference-alignment/code/02_rlhf_kl.py          # RLHF objective, KL leash, PPO (a few seconds)
uv run python chapters/18-preference-alignment/code/03_dpo_derivation.py   # check each step of the DPO derivation + parity check with zero (a few seconds)
uv run python chapters/18-preference-alignment/code/04_toy_dpo.py          # DPO on a small language model, sweep β (about 30 s of CPU time)
uv run python chapters/18-preference-alignment/code/05_dpo_pitfalls.py     # learning rate, chosen probability goes down, overfitting (about 1 min of CPU time)
```

## 1. 直觉：模仿到了上限，就让模型学"哪个更好"

SFT 的损失（loss）是"让示范答案的概率最大"。这个损失有两个上限：

1. **示范有多好，模型最多就学到多好。** 让标注员写一首好诗，或者一段没有漏洞的代码解释，又贵又慢，而且写出来的也不一定是最好的。
2. **示范里的错误也会被模型学走。** 在第 7 节（`04`）的小实验里，SFT 数据只有 40% 是对的，模型就照样学会了"六成的时候答错"。

但**判断**比**创作**省力得多：读两首诗，说出哪首更好，谁都能做到。于是有了**偏好数据**（preference data）：对同一个提示词 x 给出两个回答，由裁判挑一个。挑中的叫 **chosen**（记作 y_w，w = win），落选的叫 **rejected**（记作 y_l，l = lose）。

```
{"prompt": "用一句话解释什么是梯度",
 "chosen":   "函数值上升最快的方向，大小是那个方向的坡度。",
 "rejected": "梯度就是梯度的意思。"}
```

裁判可以是人（**人类反馈**，是 InstructGPT、Llama 3 的主要数据）。裁判也可以是一个更强的模型（**AI 反馈**）。Tülu 3、OLMo 2 用 GPT-4o 按 1–5 分给四个回答打分，最高分的回答当 chosen。Nemotron-4 用自己的奖励模型打分。

裁判还可以是一个**能自动判断对错的程序**。Qwen2.5 的离线 DPO 用"执行反馈、答案匹配"区分对的回答和错的回答。主线模型用工具调用环境打分（见"从极简代码到生产级代码"）。

## 2. Bradley–Terry：把"谁赢了"变成概率

偏好数据只说"a 比 b 好"，不给分数。**Bradley–Terry 模型**假设每个回答有一个看不见的分数 r。a 赢的概率由分数差决定：

```
P(a ≻ b) = σ(r(a) − r(b)),     σ(z) = 1 / (1 + e^(−z))
```

`01_bradley_terry.py` 的第 ① 部分打印出 σ 的几个值：

| r(a) − r(b) | −4 | −2 | −1 | 0 | +1 | +2 | +4 |
|---|---:|---:|---:|---:|---:|---:|---:|
| P(a ≻ b) | 0.02 | 0.12 | 0.27 | 0.50 | 0.73 | 0.88 | 0.98 |

后面会用到两个性质。分数相同时，两个回答各有 0.5 的概率赢。**只有差值进入公式**：所有回答的分数同时加 100，概率一个都不变。

## 3. 奖励模型：一个二分类问题

**奖励模型**（reward model，RM）是一个给（提示词，回答）打一个标量分的网络。通常的做法是：取 SFT 模型，去掉最后的词表输出层，换上一个输出 1 个数的线性头。训练奖励模型，就是最大化"chosen 赢"的对数似然：

```
L_RM = −log σ( r(x, y_w) − r(x, y_l) )
```

这就是第 5 章的**二分类交叉熵**（binary cross-entropy）。logit 是两个分数的差，标签永远是 1（"chosen 赢"）。InstructGPT 的奖励模型损失（论文式 (1)）就是这个式子。代码只有一行：

```python
def bt_loss(r_chosen, r_rejected):
    return -F.logsigmoid(r_chosen - r_rejected).mean()     # −log σ(r_w − r_l)
```

**玩具实验**：每个回答用两个特征描述，即质量 q 和长度 ℓ（单位：百 token）。标注员心里按 `1.5·q + 0.4·ℓ` 打分，也就是说**标注员略微偏爱长回答**。标注员按 Bradley–Terry 的概率挑赢家。我们只看得到 4000 对"谁赢了"，用它们训练一个线性奖励模型 `r = w_q·q + w_len·ℓ + c`（权重从 0 开始）：

| 步数 | 损失 | w_q | w_len |
|---:|---:|---:|---:|
| 0 | 0.6931 | 0.000 | 0.000 |
| 10 | 0.4610 | 0.932 | 0.358 |
| 50 | 0.4404 | 1.609 | 0.482 |
| 200 | 0.4397 | 1.493 | 0.449 |
| 600 | 0.4397 | 1.493 | 0.449 |

- **初始损失 0.6931 = ln 2**：所有回答同分，每对里两个回答各有 0.5 的概率赢。这和第 5 章"没学到东西时损失 ≈ ln(类别数)"是同一个对拍点。
- **系数学回来了**：1.493 / 0.449，对应真实的 1.5 / 0.4。
- **偏置 c 完全没动**（训练前后变化 0.0）。常数在 `r_w − r_l` 里抵消，所以它的梯度恒为 0。奖励模型的分数只有相对大小有意义。InstructGPT 在做 RL 之前专门加了一个偏置，把示范答案的平均分归零。
- 在留出集（held-out set）上，统计"奖励模型给 chosen 打分更高"的比例：奖励模型是 0.803，标注员自己的打分是 0.805。**上限不是 1**，因为标注本身有噪声。InstructGPT 报告训练标注员之间的一致率约为 72.6%。
- `−log σ(r_w − r_l)` 与 PyTorch 的 `binary_cross_entropy_with_logits(r_w − r_l, 1)` 都是 0.428393。

**最重要的一点**：奖励模型把"长"也学成了优点（w_len ≈ 0.45）。它如实学会了标注员的偏好，**也学会了偏见**。长度偏好在真实的人类偏好数据和模型偏好数据里普遍存在（Singhal et al. 2023）。

## 4. RLHF：追高分，但拴着缰绳

### 4.1 目标

有了奖励模型，下一步是让策略（policy）π 生成奖励高的回答。策略就是我们要训练的语言模型。但**不能放任策略去追分**。

奖励模型只在它见过的那类回答附近可靠。策略一旦跑到奖励模型没见过的地方，就可能找到"奖励模型打高分、实际很差"的回答。所以目标里要减去一项 KL 散度（KL divergence）。这一项把策略拴在 SFT 模型 π_ref 附近：

```
max_π   E_{y~π}[ r(x, y) ]  −  β · KL( π(·|x) ‖ π_ref(·|x) )
```

KL 项是一根**缰绳**，β 决定缰绳的松紧。InstructGPT 的写法（论文式 (2)）在每个 token 上加 KL 惩罚，β = 0.02。它还额外混入预训练梯度（PPO-ptx）。这些梯度用来减轻模型在公开 NLP 基准上的退化（"对齐税"，alignment tax）。

### 4.2 缰绳太松会怎样：reward hacking

`02_rlhf_kl.py` 把"回答"缩成 8 个候选（一个多臂老虎机，multi-armed bandit）。每个候选的真实质量 q 和长度 ℓ 已知。奖励模型用上一节学到的系数打分。表里括号中是程序输出的英文名：

| 回答 | 真实质量 q | 长度 ℓ | 奖励模型分 | π_ref |
|---|---:|---:|---:|---:|
| 简洁正确（concise） | 1.4 | 1.2 | 2.05 | 0.149 |
| 详细正确（detailed） | 1.2 | 2.5 | 2.33 | 0.122 |
| 还行（okay） | 0.6 | 1.0 | 0.76 | 0.245 |
| 一般（mediocre） | 0.2 | 1.5 | 0.39 | 0.245 |
| 跑题（off-topic） | −0.8 | 1.0 | −1.33 | 0.090 |
| 错误（wrong） | −1.5 | 0.8 | −2.46 | 0.090 |
| 啰嗦（verbose） | 0.0 | 4.0 | 1.21 | 0.055 |
| 注水 900 字（padded 900） | −0.5 | 9.0 | **2.71** | 0.004 |

奖励模型最爱的是"注水 900 字"。训练奖励模型时，回答最长只有 400 token。奖励模型从没见过 900 字的回答，只会顺着"越长越好"外推。下一节会证明，这个目标的最优策略有解析解 `π* ∝ π_ref·exp(r/β)`。先直接看结果：

| β | E[奖励模型分] | E[真实质量] | KL(π‖π_ref) | 概率最大的回答 |
|---:|---:|---:|---:|---|
| 100 | 0.626 | 0.352 | 0.000 | 还行（0.25） |
| 2 | 1.325 | 0.749 | 0.161 | 简洁正确（0.25） |
| 1 | 1.727 | 0.962 | 0.454 | 详细正确（0.35） |
| **0.5** | 2.105 | **1.124** | 0.988 | 详细正确（0.51） |
| 0.25 | 2.293 | 1.061 | 1.501 | 详细正确（0.64） |
| 0.1 | 2.554 | 0.174 | 3.336 | 注水 900 字（0.61） |
| 0.03 | 2.711 | −0.500 | 5.405 | 注水 900 字（1.00） |

β 变小时，奖励模型分**一路上涨**，真实质量却**先升后降**。β = 0.5 时结果最好。β = 0.03 时，策略把全部概率押在注水回答上，真实质量比 SFT 还差。

这种现象就是 **reward hacking**（奖励作弊）。Gao et al. 2022 在真实模型上系统地测过这条曲线："代理分（proxy score）优化得越多，真实分先升后降"。KL 缰绳就是用来防止 reward hacking 的。β 要足够大，让策略不跑出奖励模型可靠的范围。β 又要足够小，让策略真的有改进。

### 4.3 PPO：用采样一步步去解（铺垫）

真实的语言模型没法列举所有回答，只能**采样**。InstructGPT 用 **PPO**（Proximal Policy Optimization，近端策略优化）来解这个目标。PPO 的要点只有四个（第 19 章细讲）：

1. 用当前策略采样一批回答。奖励 = `r(y) − β·(log π_old(y) − log π_ref(y))`（逐样本的 KL 惩罚）。
2. **价值基线**（value baseline）：优势 A = 奖励 − V。V 是学出来的"平均能拿多少分"，用来降低方差。
3. **重要性比**（importance ratio）ρ = π_θ(y)/π_old(y)：同一批样本可以做好几次更新。
4. **裁剪**（clipping）：`min(ρ·A, clip(ρ, 1−ε, 1+ε)·A)` 防止一步走得太远。

`02` 的第 ② 部分在老虎机上跑了一个这样的 PPO（每轮 64 个样本、4 个 epoch、ε = 0.2）。PPO 从 π_ref 出发：

| 轮 | 目标 E[r] − β·KL | KL(π‖π_ref) |
|---:|---:|---:|
| 0 | 0.6071 | 0.0000 |
| 25 | 1.5921 | 0.9437 |
| 100 | 1.6084 | 0.9744 |
| 400 | 1.6111 | 0.9855 |

解析最优解的目标值是 1.6112。400 轮后，PPO 的策略与解析解之间的 KL 只有 0.00021。PPO 能解这个问题，但要让四样东西一起运行：**采样、奖励模型、价值模型、参考模型（reference model）**。训练 7B 模型时，就要同时放下四个大模型。能不能跳过这些？

**InstructGPT 是经典**：SFT → 奖励模型 → PPO 三步。在人工评测（evaluation）中，13 亿参数的 InstructGPT 胜过了 1750 亿参数的 GPT-3。Llama 2 也用"奖励模型 + 拒绝采样 + PPO"。

## 5. 从同一个目标推出 DPO

DPO（Rafailov et al. 2023）的出发点是：**上面那个目标的最优解可以直接写出来**。推导只有四步。`03_dpo_derivation.py` 用数字把每一步都验证了一遍。

**第 1 步：最优策略。** 把目标展开（x 固定，省略不写）：

```
E_π[r] − β·KL(π‖π_ref)
  = Σ_y π(y)·r(y) − β·Σ_y π(y)·log(π(y)/π_ref(y))
  = −β·Σ_y π(y)·[ log π(y) − log π_ref(y) − r(y)/β ]
  = −β·Σ_y π(y)·[ log π(y) − log( π_ref(y)·e^{r(y)/β} / Z ) − log Z ]
  = β·log Z − β·KL(π ‖ π*),         where  π*(y) = π_ref(y)·e^{r(y)/β} / Z,   Z = Σ_y π_ref(y)·e^{r(y)/β}
```

第一项与 π 无关。第二项是 KL ≥ 0，只在 π = π* 时为 0。所以 **π* 就是 RLHF 目标的最优解**：取参考模型的概率，乘上奖励的指数，再归一化。β 越小，π* 对高奖励回答的放大越强。4.2 节的表正好显示了这一点。（`03` 的 (1)：对 5 个随机 π，等式两边的最大差是 2.2×10⁻¹⁶。）

**第 2 步：反解奖励。** 对 π* 的式子取对数，再移项：

```
r(y) = β·log( π*(y) / π_ref(y) ) + β·log Z
```

奖励 = β × "最优策略相对参考模型的对数概率比" + 一个与 y 无关的常数。（`03` 的 (2)：r − β·log(π*/π_ref) 在 6 个回答上的极差是 3.3×10⁻¹⁶，确实是同一个常数 β·log Z。）

**第 3 步：代入 Bradley–Terry，Z 抵消。** Z 要对所有可能的回答求和，没法算。但 Bradley–Terry 只用奖励的**差**：

```
r(y_w) − r(y_l) = β·log( π*(y_w)/π_ref(y_w) ) − β·log( π*(y_l)/π_ref(y_l) )     (the two β·log Z terms cancel)
```

（`03` 的 (3)：隐式奖励差与真实奖励差的最大差是 4.4×10⁻¹⁶。）

**第 4 步：用策略 π_θ 代替 π*，做最大似然。** 在奖励模型的损失 `−log σ(r_w − r_l)` 里，把奖励换成上面的表达式：

```
L_DPO = −log σ( β·[ (log π_θ(y_w) − log π_ref(y_w)) − (log π_θ(y_l) − log π_ref(y_l)) ] )
```

这样，**奖励模型消失了**，采样也不需要了。只要一份偏好数据、一个冻结的参考模型、一个分类式的损失。DPO 论文的副标题说得好：*Your Language Model is Secretly a Reward Model*。`β·log(π_θ/π_ref)` 就是模型自带的**隐式奖励**（implicit reward）。

```python
def dpo_loss(pi_w, pi_l, ref_w, ref_l, beta):          # inputs: sequence log-probability of each answer, (B,)
    h = beta * ((pi_w - ref_w) - (pi_l - ref_l))       # difference of the implicit rewards
    return -F.logsigmoid(h).mean()                     # −log σ(h)
```

这里的 `log π(y|x)` 是**回答里每个 token 的 log 概率之和**（提示词部分不算）。这和第 16 章 SFT 的 loss mask 是同一个做法。

**对拍**：`03` 的 ⑤ 把这个从零写的损失和 `zero/post/dpo.py` 的 `dpo_loss` 放在 16 对随机 log 概率上比较。损失差是 1.2×10⁻⁷（float32 舍入），梯度的最大差是 4.7×10⁻¹⁰。

**DPO 真能到达 RLHF 的最优解吗？** `03` 的 ⑦ 回到 4.2 节的老虎机。它按 Bradley–Terry(r) 标注 17427 个偏好对，再**只用这些偏好对**训练 DPO（没有奖励模型，不采样），β = 0.5：

| 回答 | π_ref | π_DPO | π*（RLHF 最优解） |
|---|---:|---:|---:|
| 简洁正确 | 0.149 | 0.362 | 0.355 |
| 详细正确 | 0.122 | 0.493 | 0.514 |
| 还行 | 0.245 | 0.053 | 0.045 |
| 一般 | 0.245 | 0.026 | 0.021 |
| 跑题 | 0.090 | 0.000 | 0.000 |
| 错误 | 0.090 | 0.000 | 0.000 |
| 啰嗦 | 0.055 | 0.025 | 0.025 |
| 注水 900 字 | 0.004 | 0.040 | 0.041 |

KL(π_DPO‖π*) = 0.0017（起点 KL(π_ref‖π*) = 2.0082）。同一个目标，两条路，到了同一个地方。

## 6. DPO 的梯度在做什么

经过 h 对损失求导：

```
∂L/∂log π_θ(y_w) = −β·σ(−h),      ∂L/∂log π_θ(y_l) = +β·σ(−h)
```

梯度下降减去梯度。所以 **DPO 抬高 chosen、压低 rejected，力度相同，都是 β·σ(−h)**。σ(−h) 是"隐式奖励把这一对排错的概率"。`03` 的 ⑥（β = 0.1）：

| h | ∂L/∂log π(y_w) | −β·σ(−h) |
|---:|---:|---:|
| −4 | −0.09820 | −0.09820 |
| −2 | −0.08808 | −0.08808 |
| 0 | −0.05000 | −0.05000 |
| +2 | −0.01192 | −0.01192 |
| +4 | −0.00180 | −0.00180 |

排错得越严重（h 越负），推得越用力。已经大幅排对的，几乎不推。第 5 章的交叉熵也是这样："还差多少，就推多少"。

**β 的作用**：从推导看，β 是 RLHF 里 KL 缰绳的松紧。从损失看，β 把 log 比值差换算成"奖励"。β 小时，同样的 log 比值差只算很小的奖励，σ 不容易饱和。于是梯度会推着模型离 π_ref 更远。

## 7. 在一个小语言模型上跑 DPO

`04_toy_dpo.py`：提示词是 `a+b=`（a、b ∈ 0–9），回答是和，以 `;` 结束。模型是一个约 2.0 万参数的字符级 GRU 语言模型：

1. **SFT**（参考模型）：示范的**质量参差不齐**。同一道题，40% 的示范是对的，60% 是 0–18 里随便一个错数字。
2. **偏好数据**：70 道题 × 4 对 = 280 对。chosen = 正确答案，rejected = 参考模型会写出的错答案。另外 30 道题留出，训练时从不出现。
3. **DPO**：β = 0.1，lr = 1e-3，Adam，每步 32 对，150 步。参考模型的 log 概率在训练前算好。

> **注意：** 本章训练类实验的数字来自课程构建机上的一次 CPU 运行。不同的机器、不同版本的底层数学库，浮点运算的顺序略有不同。训练几百步后，这些微小差异会被放大。你在本机跑出的数字，可能从小数点后第二三位开始就不一样。请以下文中不依赖具体数值的结论为准。2026-10 在另一台服务器上的复跑对照见 [runs/2026-10-01-gpu0-check/chapters-16-20.md](../../runs/2026-10-01-gpu0-check/chapters-16-20.zh.md)。

SFT 参考模型在 30 道留出题上：答对的概率是 0.369，采样格式正确的比例是 1.000，采样答对的比例是 0.360。DPO 训练中（训练集上）：

| 步 | 损失 | margin | acc | log π(chosen) | log π(rejected) |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.6931 | +0.000 | 0.00 | −1.043 | −3.396 |
| 25 | 0.6512 | +0.087 | 0.93 | −0.662 | −3.881 |
| 50 | 0.6144 | +0.167 | 0.94 | −0.504 | −4.527 |
| 75 | 0.5751 | +0.260 | 0.94 | −0.466 | −5.419 |
| 100 | 0.5353 | +0.362 | 0.94 | −0.482 | −6.455 |
| 125 | 0.4934 | +0.480 | 0.94 | −0.520 | −7.677 |
| 150 | 0.4553 | +0.601 | 0.95 | −0.551 | −8.919 |

（margin = 隐式奖励差的平均值，acc = 隐式奖励把 chosen 排在前面的比例。这两个名字和 `zero` 的日志字段相同。第 0 步时 h 全是 0，所以 acc 记为 0，损失正好是 ln 2。）

在留出题上，**答对的概率从 0.369 升到 0.430**。采样答对从 0.360 升到 0.413，格式正确从 1.000 变为 0.995。隐式奖励排序正确的比例是 0.83。

注意两条曲线的形状。**rejected 一路被压下去**（−3.4 → −8.9）。chosen 先升，后略降（−1.04 → −0.47 → −0.55）。DPO 只管两者之**差**，从不直接要求 chosen 本身变大。到第 8 节，这一点会变成一个坑。

**扫 β**（lr = 1e-3，150 步）。margin/β 是 log 比值差，衡量策略离 π_ref 走了多远：

| β | 最终损失 | margin | margin/β | Δlog π(chosen) | Δlog π(rejected) | 留出答对概率 |
|---:|---:|---:|---:|---:|---:|---:|
| 0.03 | 0.5997 | +0.204 | +6.80 | +0.255 | −6.540 | 0.388 |
| 0.1 | 0.4553 | +0.601 | +6.01 | +0.492 | −5.523 | 0.430 |
| 0.3 | 0.2601 | +1.540 | +5.13 | +0.722 | −4.411 | 0.529 |
| 1.0 | 0.1030 | +2.724 | +2.72 | +0.739 | −1.984 | 0.576 |

β 越小，log 比值差越大（6.80 vs 2.72），策略离参考模型走得越远。但多出来的"路程"几乎都花在把 rejected 往下压，chosen 反而涨得少。在这个玩具上、这 150 步里，β 大的留出效果更好。不要把它推广成"β 越大越好"：这里的 π_ref 很差（只对四成），缰绳没什么价值。

在真实模型里，β 的常见取值在 0.1 附近（DPO 原论文、Zephyr、Llama 3 都用 0.1）。长度归一化（length normalization）DPO 的 β 量纲不同（Tülu 3 用 5）。无论哪种，都要在开发集上扫 β。

## 8. DPO 的坑

`05_dpo_pitfalls.py` 沿用上面的小模型。

**坑 1：学习率太大。**（β = 0.1，150 步；第一行是 SFT 参考模型本身）

| lr | margin | 训练 acc | 留出答对概率 | 采样格式正确 | 采样答对 |
|---:|---:|---:|---:|---:|---:|
| SFT | — | — | 0.369 | 1.000 | 0.360 |
| 1e-4 | +0.05 | 0.95 | 0.432 | 1.000 | 0.447 |
| 1e-3 | +0.60 | 0.95 | 0.430 | 0.995 | 0.413 |
| 1e-2 | +4.26 | 1.00 | 0.071 | 0.473 | 0.063 |
| 5e-2 | +6.61 | 1.00 | 0.031 | 0.300 | 0.032 |

lr = 1e-2 时，margin 冲到 4 以上，训练 acc 是 1.00，看起来"学得最好"。可是采样出来的回答有一大部分连格式都不对，答对的只剩 6%–7%。

学习率大的这两行训练很不稳定，对浮点误差特别敏感。2026-10 在另一台服务器上复跑，lr = 1e-2 一行是 +4.20 / 1.00 / 0.068 / 0.618 / 0.068（格式错的比例从一半多变成近四成）。5e-2 一行是 +5.73 / 1.00 / 0.000 / 0.172 / 0.000。前三行几乎不变。

具体比例会变，方向不变：模型为了把 rejected 的数字压下去，把整片"数字 token"的概率一起压坏了。**margin 高 ≠ 模型好。**

主线的冒烟测试（smoke test）遇到过同一个坑（**极小配置演示**，见 `configs/tiny/dpo.toml` 的注释）。DPO 学习率为 5e-4 时，24 步里 margin 冲到 4.8。接下来的 GRPO 阶段，工具调用格式正确率从 0.16 掉到 0。改成 5e-5 后，训练才稳定。

真实模型的 DPO 学习率通常比 SFT 小一个数量级以上：Zephyr 5e-7，Tülu 3 8B 5e-7、70B 2e-7，OLMo 2 7B 1e-6，Qwen2.5 7e-7。Llama 3 用了 1e-5（另外加了下面说的正则）。主线配置的默认值是 5e-7，待调。

**坑 2：chosen 的概率也会一起掉。** 把错答案换成"只差 1 或 2"，让 chosen 和 rejected 很像，其余不变：

| 步 | margin | acc | log π(chosen) | log π(rejected) |
|---:|---:|---:|---:|---:|
| 0 | +0.000 | 0.00 | −1.045 | −1.984 |
| 25 | +0.050 | 0.70 | −1.131 | −2.574 |
| 50 | +0.107 | 0.74 | −1.134 | −3.144 |
| 75 | +0.156 | 0.79 | −1.138 | −3.639 |
| 100 | +0.201 | 0.79 | −1.254 | −4.201 |
| 125 | +0.258 | 0.79 | −1.442 | −4.960 |
| 150 | +0.357 | 0.84 | −1.662 | −6.168 |

损失在降，margin 在涨，acc 在涨。可是 chosen 的 log 概率一路下滑。在留出题上，答对的概率**从 0.353 降到 0.223**，采样答对从 0.340 降到 0.230，格式正确从 1.000 降到 0.887。DPO 只要求"chosen 比 rejected 掉得少"。被压下去的概率可能流到了别的回答上（包括格式错误的回答）。

大模型上也反复出现这种现象。Nemotron-4 报告"chosen 和 rejected 的似然都持续下降"。Razin et al. 2024 把它叫作**似然位移**（likelihood displacement），并发现 chosen 与 rejected 越像，问题越严重。

常见的补救是**在 chosen 上加一项 SFT（NLL）损失**：Llama 3 的系数是 0.2，Nemotron-4 也加了这一项。Llama 3 还把 chosen 和 rejected 共有的格式 token（header、终止符）从 DPO 损失里屏蔽掉。原因是："同一个 token 既要抬高又要压低"会导致尾部重复或突然终止。

**坑 3：过拟合（overfitting），以及只看训练指标。** 在上面两个实验里，训练集 acc 是 0.95 / 0.84，留出题只有 0.83 / 0.63。Zephyr 报告，DPO 训练一个 epoch 后，训练准确率就到了 100%。一定要在留出的开发集上看真实指标（格式正确率、工具调用得分、对话评测），不要只看训练 margin。

**坑 4：长度偏好。** 第 3 节的奖励模型学会了"长 = 好"，DPO 的隐式奖励也会学到同样的东西。chosen 往往比 rejected 长，模型就学到"写长一点"（Park et al. 2024）。对策包括：长度归一化、在构造偏好对时控制长度、评测时用长度控制的指标（如 AlpacaEval 2 LC）。Tülu 3、OLMo 2 用的就是"长度归一化 DPO"：把 log 概率除以回答长度。

## 9. 小结

- **偏好数据**：同一提示词下成对的 (chosen, rejected)。裁判可以是人、强模型，或能自动打分的程序。
- **Bradley–Terry**：`P(y_w ≻ y_l) = σ(r_w − r_l)`。只有差值有意义。
- **奖励模型**：`−log σ(r_w − r_l)`，就是二分类交叉熵。它会学会偏好，也会学会偏见。
- **RLHF**：`max E[r] − β·KL(π‖π_ref)`。KL 是缰绳，防止 reward hacking。PPO 靠采样 + 价值基线 + 裁剪来解这个目标（第 19 章的铺垫）。
- **DPO**：最优解 `π* = π_ref·e^{r/β}/Z` → 反解 `r = β·log(π*/π_ref) + β·log Z` → 代入 Bradley–Terry，Z 抵消 → `−log σ(β·[Δ_w − Δ_l])`。
- **梯度**：抬高 chosen、压低 rejected，力度是 β·σ(−h)。
- **坑**：学习率太大会破坏格式；chosen 也会掉（加 NLL、屏蔽格式 token）；只看训练 margin 会高估效果；长度偏好。

---

## 从极简代码到生产级代码

同一件事在主线代码里是 [`zero/post/dpo.py`](../../zero/post/dpo.py)。配置在 [`configs/tiny/dpo.toml`](../../configs/tiny/dpo.toml) 和 [`configs/main/dpo.toml`](../../configs/main/dpo.toml)，测试在 [`tests/test_dpo.py`](../../tests/test_dpo.py)。

| 极简代码（`code/`） | 生产级代码（`zero/`） | 多做了什么、为什么 |
|---|---|---|
| `03` 的 `dpo_loss` | `zero.post.dpo.dpo_loss(policy_chosen_logps, policy_rejected_logps, ref_chosen_logps, ref_rejected_logps, beta)` | 公式相同。另外返回 `acc`、`margin`、`chosen_reward`、`rejected_reward` 指标，训练时每步写进日志。要盯住"chosen 也在掉"，就看 `chosen_reward`。 |
| `04` 的 `response_logps`：手工拼接"提示词 + 回答"，mask 掉提示词 | `encode_pair` → `zero.post.chat.encode_prompt_response`：按对话模板渲染（含工具定义、`tool_calls`）。只有助手回复的 token 计入 log 概率。超过 `seq_len` 的偏好对被丢弃 | 和第 16 章 SFT 用同一套模板与 loss mask。这样 DPO 优化的正是推理时模型要生成的那段文字。 |
| 每个回答单独做一次前向传播 | `batch_logps`：把 chosen 和 rejected 拼成一个 batch，右侧补齐，做一次前向传播。`zero.post.common.sequence_token_logprobs` / `token_logprobs` 取每个目标 token 的 log 概率（float32） | 一次前向传播省下一半 kernel 启动。log_softmax 在 float32 里做，避免 BF16 下长回答的求和误差。 |
| 参考 log 概率在训练前算好 | `[dpo] ref_mode = "precompute"`（训练前对全部数据算一遍参考 log 概率）或 `"online"`（每步用一份冻结的拷贝现算） | precompute 省下一整份模型的显存。online 适合边生成数据边训练的情况。 |
| 固定的 280 对 | `make_env_preferences`：当 `generate_pairs > 0` 且文件不存在时，使用工具调用环境 `zero/post/envs/tool_env.py` 的任务。从当前策略采样 `samples_per_prompt` 个回答，用可验证奖励打分。最高分（满分才算）当 chosen，否则用标准解答当 chosen。最低分当 rejected。都是满分时，把标准调用改坏当 rejected | **on-policy 偏好数据**：rejected 是模型自己真会犯的错。不需要人工标注。 |
| Adam、固定学习率 | `zero.post.common.LoopState`：AdamW、warmup + cosine、梯度裁剪、梯度累积、JSONL 日志、checkpoint、断点续训 | 和其他后训练阶段共用一套训练循环。 |
| 单进程 CPU | 单进程；CUDA 上用 BF16 autocast | 多卡 DDP 还没有实现。单卡 CUDA + BF16 的通路已在 RTX 3090 上验证（见 [runs/2026-10-01-gpu0-check](../../runs/2026-10-01-gpu0-check/README.zh.md) 第 9 节）。 |

**对拍**（[`tests/test_dpo.py`](../../tests/test_dpo.py)）：`test_dpo_loss_hand_computed` 用两对手算的例子验证损失、acc、margin。它还检查 policy = ref 时损失为 ln 2，以及梯度抬高 chosen、压低 rejected。`test_batch_logps_only_counts_response` 验证序列 log 概率只计入回复 token，并与逐 token 手算一致。`test_run_dpo_end_to_end` 在 precompute 和 online 两种模式下各跑 3 步，检查第一步损失 = ln 2、之后损失下降、checkpoint 写入磁盘。本章 `03` 的 ⑤ 又把从零写的损失和 `dpo_loss` 在数值与梯度上对了一遍。

```bash
uv run pytest tests/test_dpo.py -q
```

```
....                                                                     [100%]
4 passed in 16.09s
```

**主线配置 `configs/main/dpo.toml` 逐项说明**：

| 设置 | 值 | 为什么 |
|---|---|---|
| `init_from` | `out/main/distill/ckpt` | 流水线顺序是 SFT → 蒸馏 → DPO → GRPO。参考模型 = 蒸馏后的模型。 |
| `[dpo] beta` | 0.1 | 与 DPO 原论文、Zephyr、Llama 3 相同。待用开发集扫。 |
| `[optim] lr` | 5e-7，warmup 50 步，cosine 降到 0 | 在 Zephyr / Tülu 3（5e-7）、Qwen2.5（7e-7）、OLMo 2 7B（1e-6）的区间内。冒烟测试证明学习率太大会破坏工具调用格式。待调。 |
| `micro_batch_size × grad_accum` | 8 × 8 = 每步 64 对 | Zephyr 32，Tülu 3 128。1000 步 ≈ 6.4 万对。 |
| `ref_mode` | `precompute` | 一张卡放得下 0.7B 策略 + 预先算好的参考 log 概率。 |
| `seq_len` | 8192 | 工具说明 + 多轮对话较长，超过的偏好对被丢弃。 |
| `generate_pairs` / `samples_per_prompt` | 0 / 8 | 主线默认读现成的 `data/dpo/prefs.jsonl`。设 `generate_pairs > 0`，就用 tool_env 现场造 on-policy 偏好对。 |

**第二步的偏好数据从哪里来**（GOAL.md 3.3 节："提升通用对话质量"，并保住工具调用格式）：

1. **工具调用：on-policy + 可验证打分**。`make_env_preferences` 已经写好：策略自己采样，tool_env 打分。Qwen2 / Qwen2.5 也是这个思路："对错可以判定的任务，用执行反馈造偏好对"。
2. **通用对话：开放偏好数据集（注意许可证）**：
   - [HelpSteer3](https://huggingface.co/datasets/nvidia/HelpSteer3)：CC-BY-4.0，人工标注，含中文（`language` 列表含 zh）。
   - [UltraFeedback](https://huggingface.co/datasets/openbmb/UltraFeedback)：MIT，GPT-4 打分。二值化版本 [HuggingFaceH4/ultrafeedback_binarized](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized) 也是 MIT（Zephyr 用的就是它）。
   - [Tülu 3 8B 偏好混合](https://huggingface.co/datasets/allenai/llama-3.1-tulu-3-8b-preference-mixture)：ODC-BY-1.0，但数据卡写明"部分子集不可商用"，要逐个子集核对。
   - 这些数据里的回答来自各种模型，裁判多是 GPT-4 / GPT-4o。生成模型与裁判的使用条款是否允许"用输出训练其他模型"，**待核实**（GOAL.md 3.3 节的教师许可证规则同样适用于这里）。
3. **仿 Tülu 3 的合成管线**：用许可证允许的开放模型当裁判（与第 17 章的教师用同一份许可证清单）。裁判给 SFT 模型和几个开放模型的回答打分，最高分当 chosen。
4. 所有偏好数据都要和评测集做 13-gram 去污染（GOAL.md 3.2 节）。

## 主线进度

### 极小配置演示（CPU，`configs/tiny`，约 1.3M 参数）

> 以下是**极小配置演示**。它只说明代码通路是通的，不代表主线模型的任何结果。

> **注意：** 本节数字来自修复工具调用判分器（第 19 章第 6 节）**之前**的那次冒烟测试，是当时的真实输出。修复后重跑（`uv run python -m zero.smoke --out out/smoke_final`），数据、预训练、中期训练、SFT 各阶段完全一致。
>
> 蒸馏中通过验证的样本从 1 条（那句胡话）变为 0 条，下游的 DPO、GRPO 和评测数字随之变化。例如，同一个 SFT 模型的工具调用 call_exact 从 0.133 变为 0.100：模型没变，是判分更严了。GRPO 相对 SFT 仍判"持平"。你自己运行时，以你的运行结果为准。

`uv run python -m zero.smoke` 的 DPO 阶段（`out/smoke/SUMMARY.md`、`out/smoke/dpo/log.jsonl`）。smoke 把 `seq_len` 设为 512，造 32 对，跑 24 步。其余设置与 `configs/tiny/dpo.toml` 相同：β = 0.1，lr = 5e-5，precompute。

- 偏好数据：32 对，全部由 tool_env 现场造。其中 3 对的 chosen 来自策略自身的采样，其余 29 对用标准解答（tiny 模型太弱，很少自己答对）。
- 训练日志：

| 步 | loss | acc | margin | chosen_reward | rejected_reward | lr |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.6931 | 0.00 | +0.000 | +0.000 | +0.000 | 1.25e-5 |
| 5 | 0.6941 | 0.50 | −0.001 | +0.003 | +0.005 | 5.00e-5 |
| 10 | 0.6125 | 1.00 | +0.170 | +0.060 | −0.110 | 4.34e-5 |
| 15 | 0.3797 | 1.00 | +0.802 | +0.011 | −0.790 | 2.75e-5 |
| 20 | 0.3531 | 1.00 | +0.932 | +0.017 | −0.915 | 1.16e-5 |
| 24 | 0.5262 | 0.75 | +0.394 | −0.020 | −0.414 | 5.28e-6 |

- 第 1 步的损失正好是 ln 2（策略 = 参考模型）。之后 rejected_reward 一路变负，chosen_reward 在 0 附近徘徊，最后一步略为负。这和第 8 节"chosen 不一定涨"是同一个现象的迹象（每步只有 4 对，噪声很大，不能下结论）。
- 学习率：最初用 5e-4 时，24 步里 margin 冲到 4.8，接下来的 GRPO 阶段格式正确率从 0.16 掉到 0。于是改成 5e-5（见 `configs/tiny/dpo.toml` 的注释）。
- 之后的 eval 阶段比较的是 SFT 模型与最终的 GRPO 模型，没有单独评测 DPO 模型的效果。

### 待 GPU 训练后补充

- 主线偏好数据的来源、规模、许可证清单与去污染结果。
- β、学习率在开发集上的扫描结果。重点看工具调用格式正确率与通用对话评测，不看训练 margin。
- 训练曲线（loss、acc、margin、chosen_reward / rejected_reward）和花费（记入 `runs/ledger.md`）。
- DPO 前后在开发集上的对比，以及 DPO 对下一步 GRPO 的影响。
- 如果出现"chosen 一起掉"或格式退化：是否需要加 chosen 上的 NLL 项、屏蔽格式 token？（`zero` 目前都没有实现，见下面的说明。）

> **说明：** `zero` 当前实现的是原版 DPO。它没有 Llama 3 的"NLL 正则 + 屏蔽格式 token"，也没有 Tülu 3 的长度归一化。这几项改动都很小。第二步若在开发集上看到对应的问题，再加上它们并做对照实验。

---

## 前沿观察

> **DPO 的变体**：变体层出不穷。IPO（Azar et al. 2023）把 −log σ 换成平方损失，以防过拟合。KTO（Ethayarajh et al. 2024）只需要"好/坏"的单条标注，不必成对。SimPO（Meng et al. 2024）去掉参考模型，用长度归一化的 log 概率当奖励。ORPO（Hong et al. 2024）把偏好项并进 SFT 损失，不要参考模型。
>
> 头部开源模型里只看到零散的采用：例如 SmolLM3 用 APO，Nemotron-4 用自家的 RPO，Tülu 3 比较后选了长度归一化 DPO 而不是 SimPO。按 GOAL.md 2.1 节，这里只简单提一句。

---

## 采用方与来源

| 技术 | 采用方（主力版本） | 来源 |
|---|---|---|
| DPO | **Llama 3**（每轮 SFT 后做 DPO；lr 1e-5、β = 0.1；屏蔽格式 token + 0.2 倍 NLL；比较过 PPO，认为 DPO 更省算力、IFEval 更好）；**Qwen2**（离线 DPO + 在线阶段用奖励模型选最好/最差的回答做 DPO）；**Qwen2.5**（离线 DPO 约 15 万对、lr 7e-7，之后在线 GRPO）；**DeepSeek LLM**（SFT 后做 DPO，lr 5e-6，batch 512）；**Tülu 3 / OLMo 2**（长度归一化 DPO + on-policy 合成偏好数据；Tülu 3 8B：lr 5e-7、β = 5）；**Nemotron-4 340B**（DPO 后再做三轮 RPO，并在 chosen 上加 SFT 损失）；**Zephyr**（在 UltraFeedback 上做 dDPO，lr 5e-7、β = 0.1） | [Llama 3](https://arxiv.org/abs/2407.21783) §4.1.4；[Qwen2](https://arxiv.org/abs/2407.10671) §4.3；[Qwen2.5](https://arxiv.org/abs/2412.15115) §4.2–4.3；[DeepSeek LLM](https://arxiv.org/abs/2401.02954) §4；[Tülu 3](https://arxiv.org/abs/2411.15124) §5；[OLMo 2](https://arxiv.org/abs/2501.00656) §5；[Nemotron-4 340B](https://arxiv.org/abs/2406.11704) §3.3.2；[Zephyr](https://arxiv.org/abs/2310.16944) §4.4 |
| DPO 变体（APO） | **SmolLM3**（APO，"DPO 的一个更稳定的变体"；非推理模式用 Tülu 3 偏好数据，推理模式用 Qwen3-32B 的回答当 chosen、Qwen3-0.6B 的回答当 rejected） | [SmolLM3 博客](https://github.com/huggingface/blog/blob/main/smollm3.md)；[APO](https://arxiv.org/abs/2408.06266) |
| 奖励模型 + PPO（铺垫） | **InstructGPT**（6B 奖励模型、PPO、β = 0.02、PPO-ptx）；Llama 2（奖励模型 + 拒绝采样 + PPO）；Tülu 3 做过 PPO 与 DPO 的对照 | [InstructGPT](https://arxiv.org/abs/2203.02155) §3.5、附录 C.4；[Stiennon et al. 2020](https://arxiv.org/abs/2009.01325)；[PPO](https://arxiv.org/abs/1707.06347) |

**共识判断（GOAL.md 2.1 节）**：Llama、Qwen、DeepSeek、OLMo、Nemotron 至少 5 个彼此独立的头部家族在主力版本里明确采用了 DPO（SmolLM3 用它的变体）。DPO 满足规则 A，进正文。奖励模型 + PPO 是 DPO 推导的起点，也是第 19 章 GRPO 的铺垫（规则 C）。

> **说明：** 更新一代的模型（如 Qwen2.5 之后）把更多后训练预算放到了在线 RL（GRPO，第 19 章）上。DeepSeek-V3、Qwen3 等最新主力版本的后训练是否仍含 DPO，本章没有逐一核实（待核实）。

---

## 引导问题

带着这些问题去问 Claude Code，直到你能用自己的话讲清楚：

1. 奖励模型的偏置为什么梯度恒为 0？InstructGPT 在 RL 之前把示范答案的平均奖励归零。如果没有这一步，PPO 会受影响吗？DPO 需要这一步吗？
2. `02` 的表里，β = 0.1 时"注水"回答的概率是 0.61，而它在 π_ref 下只有 0.004。用 `π* ∝ π_ref·e^{r/β}` 算一算：要多大的奖励差，才能让一个 0.4% 的回答升到 61%？这说明奖励模型在"没见过的区域"犯一点错有多危险。
3. DPO 推导的第 4 步"用 π_θ 代替 π*"隐含了一个假设：策略能表示任意分布，偏好数据覆盖了所有回答。数据只覆盖一小部分回答时，DPO 对"没出现过的回答"的概率有约束吗？这和第 8 节"chosen 也会掉"有什么关系？
4. 为什么说 DPO 是 **off-policy** 的，而 PPO 是 on-policy 的？Llama 3 为什么"主要用最近一轮最好模型采集的偏好数据"？`make_env_preferences` 属于哪一种？
5. 长度归一化 DPO 把 log 概率除以回答长度。它的最优解还和原版 DPO 一样是 `π_ref·e^{r/β}` 吗？为什么 Tülu 3 的 β 要设到 5 这么大？
6. 如果偏好数据里的"裁判"本身就是一个奖励模型（例如 Nemotron-4 用奖励模型给合成回答排序），DPO 和 RLHF 的区别还剩下什么？

## 动手任务

每个任务都要运行代码，看到结果。

**任务 1（基础）**：修改 `01_bradley_terry.py`，把标注员的长度系数从 0.4 改成 −0.4（标注员讨厌长回答）。重新训练奖励模型，再运行 `02`。reward hacking 还会发生吗？最好的 β 变了吗？

**任务 2（核心）**：在 `04_toy_dpo.py` 的 `run_dpo` 里加一项 chosen 上的 NLL 损失：`loss = dpo + α·(−pw.mean() / n_answer_tokens)`，α 取 0.2，仿照 Llama 3。然后用 `05` 的"错答案只差一点"设置重跑。chosen 的 log 概率还会一路掉吗？留出题答对的概率变成多少？

**任务 3（挑战）**：实现长度归一化 DPO（log 概率除以回答 token 数）。构造一份"chosen 总是比 rejected 长"的偏好数据。例如 chosen 是 `12;`，rejected 是 `9;`，同时让一部分正确答案写成 `12!;` 这种更长的形式。分别用原版 DPO 和长度归一化 DPO 训练，比较两个模型生成回答的平均长度。

---

## 想深入：CS336

本章对应斯坦福 CS336（Spring 2026）<https://cs336.stanford.edu/>：

- **第 15 讲：中期训练与后训练（SFT / RLHF）**。讲 InstructGPT 式的 RLHF 流水线、奖励模型、PPO 与 DPO 的关系，是本章第 3–5 节的英文原版。
- **作业 5（Alignment and Reasoning RL）的可选第二部分：DPO**。在真实的小模型和真实的偏好数据上实现 DPO，是本章 `03`、`04` 的放大版。作业 5 的主体（SFT、专家迭代、GRPO）对应第 16、19 章。

---

## 本章参考文献

- Rafailov et al. *Direct Preference Optimization: Your Language Model is Secretly a Reward Model*，2023：<https://arxiv.org/abs/2305.18290>
- Ouyang et al. *Training language models to follow instructions with human feedback*（InstructGPT），2022：<https://arxiv.org/abs/2203.02155>
- Stiennon et al. *Learning to summarize from human feedback*，2020：<https://arxiv.org/abs/2009.01325>
- Schulman et al. *Proximal Policy Optimization Algorithms*，2017：<https://arxiv.org/abs/1707.06347>
- Gao, Schulman, Hilton. *Scaling Laws for Reward Model Overoptimization*，2022：<https://arxiv.org/abs/2210.10760>
- Llama Team. *The Llama 3 Herd of Models*（§4.1.4 DPO），2024：<https://arxiv.org/abs/2407.21783>
- Lambert et al. *Tülu 3: Pushing Frontiers in Open Language Model Post-Training*（§5 偏好微调），2024：<https://arxiv.org/abs/2411.15124>
- OLMo Team. *2 OLMo 2 Furious*（§5 后训练），2024：<https://arxiv.org/abs/2501.00656>
- Qwen Team. *Qwen2 Technical Report*（§4.3），2024：<https://arxiv.org/abs/2407.10671>；*Qwen2.5 Technical Report*（§4.2–4.3），2024：<https://arxiv.org/abs/2412.15115>
- DeepSeek-AI. *DeepSeek LLM: Scaling Open-Source Language Models with Longtermism*（§4），2024：<https://arxiv.org/abs/2401.02954>
- NVIDIA. *Nemotron-4 340B Technical Report*（§3.3.2 DPO 与 RPO），2024：<https://arxiv.org/abs/2406.11704>
- Tunstall et al. *Zephyr: Direct Distillation of LM Alignment*，2023：<https://arxiv.org/abs/2310.16944>
- Hugging Face. *SmolLM3: smol, multilingual, long-context reasoner*（博客）：<https://github.com/huggingface/blog/blob/main/smollm3.md>
- D'Oosterlinck et al. *Anchored Preference Optimization and Contrastive Revisions*（APO），2024：<https://arxiv.org/abs/2408.06266>
- Razin et al. *Unintentional Unalignment: Likelihood Displacement in Direct Preference Optimization*，2024：<https://arxiv.org/abs/2410.08847>
- Pal et al. *Smaug: Fixing Failure Modes of Preference Optimisation with DPO-Positive*，2024：<https://arxiv.org/abs/2402.13228>
- Park et al. *Disentangling Length from Quality in Direct Preference Optimization*，2024：<https://arxiv.org/abs/2403.19159>
- Singhal et al. *A Long Way to Go: Investigating Length Correlations in RLHF*，2023：<https://arxiv.org/abs/2310.03716>
- 变体（前沿观察）：IPO <https://arxiv.org/abs/2310.12036>；KTO <https://arxiv.org/abs/2402.01306>；SimPO <https://arxiv.org/abs/2405.14734>；ORPO <https://arxiv.org/abs/2403.07691>
- 偏好数据集：[UltraFeedback](https://huggingface.co/datasets/openbmb/UltraFeedback)（MIT）、[HelpSteer3](https://huggingface.co/datasets/nvidia/HelpSteer3)（CC-BY-4.0）、[Tülu 3 8B 偏好混合](https://huggingface.co/datasets/allenai/llama-3.1-tulu-3-8b-preference-mixture)（ODC-BY-1.0，部分子集不可商用）
- [Hands-On Modern Reinforcement Learning](https://walkinglabs.github.io/hands-on-modern-rl/preface/intro)（`references.md` 已收录，RL 与 PPO 的中文入门）
- [CS336](https://cs336.stanford.edu/) 第 15 讲、作业 5

**下一章**：偏好对只能告诉模型"哪个更好"，而且数据是事先收集好的。工具调用这类任务有一个更好的条件：对错可以自动判定。那就让模型自己去试、当场拿到分数、从自己的尝试里学习。第 19 章：强化学习 —— 从 PPO 到 GRPO，以及可验证奖励。
