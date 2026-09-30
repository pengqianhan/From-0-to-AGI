# 第 18 章：偏好对齐 —— 从 RLHF 到 DPO

> **一句话目标**：读完这一章，你能从"偏好数据 + Bradley–Terry 模型"出发，写出奖励模型的损失；讲清 RLHF 目标 `E[r] − β·KL` 里 KL 这根"缰绳"为什么必不可少；亲手把 RLHF 的最优解推成 DPO 损失，并用代码和 `zero/post/dpo.py` 对拍；知道 DPO 的学习率、β、"chosen 概率也会掉"这几个坑该怎么盯。

📺 **本章视频**：待发布（本地渲染：`bash chapters/18-preference-alignment/video/build.sh`）
🧪 **本章自检**：学完后在 Claude Code 里输入 `/ch18-dpo`

---

上一章我们用蒸馏把教师的工具调用本领灌进了小模型：先 SFT 学格式，再学教师写的、经过执行验证的示范。这两步的本质都是**模仿**——示范怎么写，模型就怎么写。这一章要解决的问题是：**很多时候我们写不出"最好的答案"，却能轻松判断"两个答案哪个更好"**。怎么把这种判断变成训练信号？

经典答案是 **RLHF（Reinforcement Learning from Human Feedback，基于人类反馈的强化学习）**：先训练一个奖励模型给回答打分，再用 PPO 让模型去追高分（InstructGPT 就是这么做的）。这一章先把 RLHF 讲清楚——它是理解第 19 章 GRPO 的铺垫（GOAL.md 2.1 的规则 C）；然后从**同一个目标**出发，推导出今天开源模型最常用的偏好对齐方法：**DPO（Direct Preference Optimization，直接偏好优化）**。主线模型的第 6 个训练阶段就是 DPO。

本章代码（都在 CPU 上跑，`torch.set_num_threads(1)`）：

```bash
uv run python chapters/18-preference-alignment/code/01_bradley_terry.py    # Bradley–Terry 奖励模型（几秒）
uv run python chapters/18-preference-alignment/code/02_rlhf_kl.py          # RLHF 目标、KL 缰绳、PPO（几秒）
uv run python chapters/18-preference-alignment/code/03_dpo_derivation.py   # DPO 推导逐步验证 + 与 zero 对拍（几秒）
uv run python chapters/18-preference-alignment/code/04_toy_dpo.py          # 小语言模型上的 DPO、扫 β（CPU 时间约半分钟）
uv run python chapters/18-preference-alignment/code/05_dpo_pitfalls.py     # 学习率、chosen 概率下降、过拟合（CPU 时间约 1 分钟）
```

## 1. 直觉：模仿到头了，就让模型学"哪个更好"

SFT 的损失是"让示范答案的概率最大"。它有两个天花板：

1. **示范写得多好，模型最多学到多好。** 让标注员写一首好诗、一段滴水不漏的代码解释，又贵又慢，而且写出来的也未必是最好的。
2. **示范里的毛病也一起学走。** 第 4 节的小实验里，SFT 数据只有 40% 是对的，模型就老老实实学会了"六成时候答错"。

可是**判断**比**创作**容易得多：读两首诗，说出哪首更好，谁都会。于是有了**偏好数据（preference data）**：对同一个提示词 x 给出两个回答，裁判挑一个。挑中的叫 **chosen**（记作 y_w，w = win），落选的叫 **rejected**（y_l，l = lose）。

```
{"prompt": "用一句话解释什么是梯度",
 "chosen":   "函数值上升最快的方向，大小是那个方向的坡度。",
 "rejected": "梯度就是梯度的意思。"}
```

裁判可以是人（**人类反馈**，InstructGPT、Llama 3 的主力数据），也可以是一个更强的模型（**AI 反馈**：Tülu 3、OLMo 2 用 GPT-4o 按 1–5 分给四个回答打分，最高分当 chosen；Nemotron-4 用自己的奖励模型打分），还可以是一个**可以自动判对错的程序**（Qwen2.5 的离线 DPO 用"执行反馈、答案匹配"区分对错回答；主线模型用工具调用环境打分，见"从极简到生产级"）。

## 2. Bradley–Terry：把"谁赢了"变成概率

偏好数据只说"a 比 b 好"，不给分数。**Bradley–Terry 模型**假设每个回答心里有一个看不见的分数 r，a 赢的概率由分数差决定：

```
P(a ≻ b) = σ(r(a) − r(b)),     σ(z) = 1 / (1 + e^(−z))
```

`01_bradley_terry.py` 的第 ① 部分把 σ 的几个值打出来：

| r(a) − r(b) | −4 | −2 | −1 | 0 | +1 | +2 | +4 |
|---|---:|---:|---:|---:|---:|---:|---:|
| P(a ≻ b) | 0.02 | 0.12 | 0.27 | 0.50 | 0.73 | 0.88 | 0.98 |

两个性质后面都会用到：分数一样时是五五开；**只有差值进公式**——所有回答的分数同时加 100，概率一个都不变。

## 3. 奖励模型：一个二分类问题

**奖励模型（reward model, RM）** 就是一个"给 (提示词, 回答) 打一个标量分"的网络：通常是 SFT 模型去掉最后的词表输出层，换一个输出 1 个数的线性头。训练它就是最大化"chosen 赢"的对数似然：

```
L_RM = −log σ( r(x, y_w) − r(x, y_l) )
```

这正是第 5 章的**二分类交叉熵**：logit 是两个分数的差，标签永远是 1（"chosen 赢"）。InstructGPT 的奖励模型损失（论文式 (1)）就是这个式子。代码只有一行：

```python
def bt_loss(r_chosen, r_rejected):
    return -F.logsigmoid(r_chosen - r_rejected).mean()     # −log σ(r_w − r_l)
```

**玩具实验**：每个回答用两个特征描述——质量 q 和长度 ℓ（百 token）。标注员心里按 `1.5·q + 0.4·ℓ` 打分（**略微偏爱长回答**），按 Bradley–Terry 的概率挑赢家。我们只看得到 4000 对"谁赢了"，训练一个线性奖励模型 `r = w_q·q + w_len·ℓ + c`（权重从 0 开始）：

| 步数 | 损失 | w_q | w_len |
|---:|---:|---:|---:|
| 0 | 0.6931 | 0.000 | 0.000 |
| 10 | 0.4610 | 0.932 | 0.358 |
| 50 | 0.4404 | 1.609 | 0.482 |
| 200 | 0.4397 | 1.493 | 0.449 |
| 600 | 0.4397 | 1.493 | 0.449 |

- **初始损失 0.6931 = ln 2**：所有回答同分，每对都是五五开——和第 5 章"没学东西时损失 ≈ ln(类别数)"是同一个对拍点。
- **学回来了**：1.493 / 0.449，对应真实的 1.5 / 0.4。
- **偏置 c 纹丝不动**（训练前后变化 0.0）：常数在 `r_w − r_l` 里消掉了，梯度恒为 0。奖励模型的分数只有"相对大小"有意义，InstructGPT 在做 RL 之前专门加了一个偏置，把示范答案的平均分归零。
- 留出集上"给 chosen 打分更高"的比例：奖励模型 0.803，标注员自己的打分 0.805（**上限不是 1**，因为标注本身有噪声；InstructGPT 报告训练标注员之间的一致率约 72.6%）。
- `−log σ(r_w − r_l)` 与 PyTorch 的 `binary_cross_entropy_with_logits(r_w − r_l, 1)` 都是 0.428393。

**最要紧的一点**：奖励模型把"长"也学成了优点（w_len ≈ 0.45）。它忠实地学会了标注员的偏好，**包括偏见**。长度偏好在真实的人类和模型偏好数据里普遍存在（Singhal et al. 2023）。

## 4. RLHF：追高分，但拴着缰绳

### 4.1 目标

有了奖励模型，下一步是让策略 π（我们要训练的语言模型）去生成奖励高的回答。但**不能放任它追**：奖励模型只在它见过的那类回答附近靠谱，策略一旦跑到没见过的地方，就可能找到"奖励模型打高分、其实很烂"的回答。所以目标里要减去一项 KL 散度，把策略拴在 SFT 模型 π_ref 附近：

```
max_π   E_{y~π}[ r(x, y) ]  −  β · KL( π(·|x) ‖ π_ref(·|x) )
```

KL 就是一根**缰绳**，β 是缰绳的松紧。InstructGPT 的写法（论文式 (2)）是在每个 token 上加 KL 惩罚，β = 0.02；它还额外混入了预训练梯度（PPO-ptx），用来缓解在公开 NLP 基准上的退化（"对齐税"）。

### 4.2 缰绳太松会怎样：reward hacking

`02_rlhf_kl.py` 把"回答"缩成 8 个候选（一个多臂老虎机），真实质量 q 和长度 ℓ 已知，奖励模型用上一节学到的系数打分：

| 回答 | 真实质量 q | 长度 ℓ | 奖励模型分 | π_ref |
|---|---:|---:|---:|---:|
| 简洁正确 | 1.4 | 1.2 | 2.05 | 0.149 |
| 详细正确 | 1.2 | 2.5 | 2.33 | 0.122 |
| 还行 | 0.6 | 1.0 | 0.76 | 0.245 |
| 一般 | 0.2 | 1.5 | 0.39 | 0.245 |
| 跑题 | −0.8 | 1.0 | −1.33 | 0.090 |
| 错误 | −1.5 | 0.8 | −2.46 | 0.090 |
| 啰嗦 | 0.0 | 4.0 | 1.21 | 0.055 |
| 注水 900 字 | −0.5 | 9.0 | **2.71** | 0.004 |

奖励模型最爱的是"注水 900 字"——训练奖励模型时回答最长只有 400 token，900 字它从没见过，只会顺着"越长越好"外推。下一节会证明，这个目标的最优策略有闭式解 `π* ∝ π_ref·exp(r/β)`；先直接看结果：

| β | E[奖励模型分] | E[真实质量] | KL(π‖π_ref) | 概率最大的回答 |
|---:|---:|---:|---:|---|
| 100 | 0.626 | 0.352 | 0.000 | 还行（0.25） |
| 2 | 1.325 | 0.749 | 0.161 | 简洁正确（0.25） |
| 1 | 1.727 | 0.962 | 0.454 | 详细正确（0.35） |
| **0.5** | 2.105 | **1.124** | 0.988 | 详细正确（0.51） |
| 0.25 | 2.293 | 1.061 | 1.501 | 详细正确（0.64） |
| 0.1 | 2.554 | 0.174 | 3.336 | 注水 900 字（0.61） |
| 0.03 | 2.711 | −0.500 | 5.405 | 注水 900 字（1.00） |

奖励模型分随 β 变小**一路上涨**，真实质量却**先升后降**：β = 0.5 时最好，β = 0.03 时策略把全部概率押在注水回答上，真实质量比 SFT 还差。这就是 **reward hacking（奖励投机）**，Gao et al. 2022 在真实模型上系统地测过这条"代理分越优化、真实分先升后降"的曲线。KL 缰绳就是为它准备的：β 要大到不让策略跑出奖励模型靠谱的范围，又要小到让策略真的改进。

### 4.3 PPO：用采样一步步去解（铺垫）

真实的语言模型没法枚举所有回答，只能**采样**。InstructGPT 用 **PPO（Proximal Policy Optimization）** 来解这个目标，要点只有四个（第 19 章细讲）：

1. 用当前策略采样一批回答，奖励 = `r(y) − β·(log π_old(y) − log π_ref(y))`（逐样本的 KL 惩罚）；
2. **价值基线**：优势 A = 奖励 − V，V 是一个学出来的"平均能拿多少分"，用来降方差；
3. **重要性比例** ρ = π_θ(y)/π_old(y)：同一批样本可以更新好几次；
4. **裁剪**：`min(ρ·A, clip(ρ, 1−ε, 1+ε)·A)`，防止一次走太远。

`02` 的第 ② 部分在老虎机上跑了一个这样的 PPO（每轮 64 个样本、4 个 epoch、ε = 0.2），从 π_ref 出发：

| 轮 | 目标 E[r] − β·KL | KL(π‖π_ref) |
|---:|---:|---:|
| 0 | 0.6071 | 0.0000 |
| 25 | 1.5921 | 0.9437 |
| 100 | 1.6084 | 0.9744 |
| 400 | 1.6111 | 0.9855 |

闭式最优解的目标值是 1.6112；400 轮后 PPO 的策略与闭式解只差 KL = 0.00021。PPO 能解，但要**采样、奖励模型、价值模型、参考模型**四样东西一起跑——训练 7B 模型就要同时放四个大模型。能不能跳过这些？

**InstructGPT 是经典**：SFT → 奖励模型 → PPO 三步，13 亿参数的 InstructGPT 在人工评测中胜过 1750 亿参数的 GPT-3。Llama 2 也用"奖励模型 + 拒绝采样 + PPO"。

## 5. 从同一个目标推出 DPO

DPO（Rafailov et al. 2023）的出发点是：**上面那个目标的最优解可以直接写出来**。推导只有四步，每一步 `03_dpo_derivation.py` 都用数字验证了一遍。

**第 1 步：最优策略。** 把目标展开（对一个固定的 x，省略不写）：

```
E_π[r] − β·KL(π‖π_ref)
  = Σ_y π(y)·r(y) − β·Σ_y π(y)·log(π(y)/π_ref(y))
  = −β·Σ_y π(y)·[ log π(y) − log π_ref(y) − r(y)/β ]
  = −β·Σ_y π(y)·[ log π(y) − log( π_ref(y)·e^{r(y)/β} / Z ) − log Z ]
  = β·log Z − β·KL(π ‖ π*),         其中  π*(y) = π_ref(y)·e^{r(y)/β} / Z,   Z = Σ_y π_ref(y)·e^{r(y)/β}
```

第一项与 π 无关；第二项 KL ≥ 0，只在 π = π* 时为 0。所以 **π* 就是 RLHF 目标的最优解**：参考模型的概率，乘上奖励的指数，再归一化。β 越小，奖励高的回答被放大得越猛——正是 4.2 表里看到的。（`03` 的 (1)：对 5 个随机 π，等式两边最大差 2.2×10⁻¹⁶。）

**第 2 步：反解奖励。** 对 π* 的式子取对数、移项：

```
r(y) = β·log( π*(y) / π_ref(y) ) + β·log Z
```

奖励 = β × "最优策略相对参考模型的对数概率比" + 一个与 y 无关的常数。（`03` 的 (2)：r − β·log(π*/π_ref) 在 6 个回答上的极差 3.3×10⁻¹⁶，确实是同一个常数 β·log Z。）

**第 3 步：代入 Bradley–Terry，Z 消掉。** Z 要对所有可能的回答求和，没法算。但 Bradley–Terry 只用奖励的**差**：

```
r(y_w) − r(y_l) = β·log( π*(y_w)/π_ref(y_w) ) − β·log( π*(y_l)/π_ref(y_l) )     （两个 β·log Z 抵消）
```

（`03` 的 (3)：隐式奖励差与真实奖励差的最大差 4.4×10⁻¹⁶。）

**第 4 步：用策略 π_θ 代替 π*，做最大似然。** 奖励模型那一步的损失 `−log σ(r_w − r_l)` 里，把奖励换成上面的表达式：

```
L_DPO = −log σ( β·[ (log π_θ(y_w) − log π_ref(y_w)) − (log π_θ(y_l) − log π_ref(y_l)) ] )
```

就这样，**奖励模型消失了**，采样也不需要了：只要一份偏好数据、一个冻结的参考模型、一个分类式的损失。DPO 论文的副标题说得好：*Your Language Model is Secretly a Reward Model*——`β·log(π_θ/π_ref)` 就是模型自带的**隐式奖励（implicit reward）**。

```python
def dpo_loss(pi_w, pi_l, ref_w, ref_l, beta):          # 输入：每个回答的序列 log 概率 (B,)
    h = beta * ((pi_w - ref_w) - (pi_l - ref_l))       # 隐式奖励之差
    return -F.logsigmoid(h).mean()                     # −log σ(h)
```

这里的 `log π(y|x)` 是**回答里每个 token 的 log 概率之和**（提示词部分不算），和第 16 章 SFT 的 loss mask 是同一回事。

**对拍**：`03` 的 ⑤ 把这个从零写的损失和 `zero/post/dpo.py` 的 `dpo_loss` 放在 16 对随机 log 概率上比较：损失差 1.2×10⁻⁷（float32 舍入），梯度最大差 4.7×10⁻¹⁰。

**DPO 真能到达 RLHF 的最优解吗？** `03` 的 ⑦ 回到 4.2 的老虎机：按 Bradley–Terry(r) 标注 17427 个偏好对，**只用这些偏好对**训练 DPO（没有奖励模型、不采样），β = 0.5：

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

KL(π_DPO‖π*) = 0.0017（起点 KL(π_ref‖π*) = 2.0082）。同一个目标，两条路，走到了同一个地方。

## 6. DPO 的梯度在做什么

对 h 求导：

```
∂L/∂log π_θ(y_w) = −β·σ(−h),      ∂L/∂log π_θ(y_l) = +β·σ(−h)
```

梯度下降减去梯度，所以：**抬高 chosen、压低 rejected，力度相同，都是 β·σ(−h)**。σ(−h) 是"隐式奖励把这对排错的概率"。`03` 的 ⑥（β = 0.1）：

| h | ∂L/∂log π(y_w) | −β·σ(−h) |
|---:|---:|---:|
| −4 | −0.09820 | −0.09820 |
| −2 | −0.08808 | −0.08808 |
| 0 | −0.05000 | −0.05000 |
| +2 | −0.01192 | −0.01192 |
| +4 | −0.00180 | −0.00180 |

排错得越离谱（h 越负）推得越用力，已经排对很多的几乎不推——和第 5 章交叉熵"还差多少就推多少"一模一样。

**β 的作用**：从推导看，β 是 RLHF 里 KL 缰绳的松紧；从损失看，β 把 log 比值差换算成"奖励"。β 小，同样的 log 比值差只算很小的奖励，σ 不容易饱和，模型会被推着离 π_ref 走得更远。

## 7. 在一个小语言模型上跑 DPO

`04_toy_dpo.py`：提示词 `a+b=`（a、b ∈ 0–9），回答是和，以 `;` 结束。一个约 2.0 万参数的字符级 GRU 语言模型：

1. **SFT**（参考模型）：示范**质量参差不齐**——同一道题，40% 的示范是对的，60% 是 0–18 里随便一个错数字。
2. **偏好数据**：70 道题 × 4 对 = 280 对，chosen = 正确答案，rejected = 参考模型会写出的错答案。另外 30 道题留出，训练时从不出现。
3. **DPO**：β = 0.1，lr = 1e-3，Adam，每步 32 对，150 步；参考模型的 log 概率训练前算好。

SFT 参考模型在 30 道留出题上：答对的概率 0.369，采样格式正确 1.000，采样答对 0.360。DPO 训练中（训练集上）：

| 步 | 损失 | margin | acc | log π(chosen) | log π(rejected) |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.6931 | +0.000 | 0.00 | −1.043 | −3.396 |
| 25 | 0.6512 | +0.087 | 0.93 | −0.662 | −3.881 |
| 50 | 0.6144 | +0.167 | 0.94 | −0.504 | −4.527 |
| 75 | 0.5751 | +0.260 | 0.94 | −0.466 | −5.419 |
| 100 | 0.5353 | +0.362 | 0.94 | −0.482 | −6.455 |
| 125 | 0.4934 | +0.480 | 0.94 | −0.520 | −7.677 |
| 150 | 0.4553 | +0.601 | 0.95 | −0.551 | −8.919 |

（margin = 隐式奖励差的平均，acc = 隐式奖励把 chosen 排在前面的比例，和 `zero` 的日志字段同名。第 0 步 h 全是 0，所以 acc 记为 0、损失正好是 ln 2。）

留出题上：**答对的概率 0.369 → 0.430**，采样答对 0.360 → 0.413，格式 1.000 → 0.995，隐式奖励排序正确 0.83。

注意两根曲线的形状：**rejected 一路被压下去**（−3.4 → −8.9），chosen 先升后微降（−1.04 → −0.47 → −0.55）。DPO 只管两者之**差**，从不直接要求 chosen 本身变大——这一点在第 8 节会变成一个坑。

**扫 β**（lr = 1e-3，150 步；margin/β 是 log 比值差，衡量策略离 π_ref 走了多远）：

| β | 最终损失 | margin | margin/β | Δlog π(chosen) | Δlog π(rejected) | 留出答对概率 |
|---:|---:|---:|---:|---:|---:|---:|
| 0.03 | 0.5997 | +0.204 | +6.80 | +0.255 | −6.540 | 0.388 |
| 0.1 | 0.4553 | +0.601 | +6.01 | +0.492 | −5.523 | 0.430 |
| 0.3 | 0.2601 | +1.540 | +5.13 | +0.722 | −4.411 | 0.529 |
| 1.0 | 0.1030 | +2.724 | +2.72 | +0.739 | −1.984 | 0.576 |

β 越小，log 比值差越大（6.80 vs 2.72）——策略离参考模型走得更远，但多出来的"路程"几乎都花在把 rejected 往下压，chosen 涨得反而少。在这个玩具上、这 150 步里，β 大的留出效果更好；不要把它推广成"β 越大越好"：这里 π_ref 很差（只对四成），缰绳并不值钱。真实模型里，β 常见取值是 0.1 附近（DPO 原论文、Zephyr、Llama 3 都是 0.1），长度归一化 DPO 的 β 量纲不同（Tülu 3 用 5），都要用开发集扫。

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

lr = 1e-2 时 margin 冲到 4.26、训练 acc 1.00，看起来"学得最好"；可采样出来的回答一半多连格式都不对。模型为了把 rejected 的数字压下去，把整片"数字 token"的概率一起压坏了。**margin 高 ≠ 模型好。**

主线的冒烟测试踩过同一个坑（**极小配置演示**，`configs/tiny/dpo.toml` 的注释）：DPO 学习率 5e-4 时，24 步里 margin 冲到 4.8，接下来 GRPO 阶段的工具调用格式正确率从 0.16 掉到 0；改成 5e-5 才稳住。真实模型的 DPO 学习率通常比 SFT 小一个数量级以上：Zephyr 5e-7，Tülu 3 8B 5e-7、70B 2e-7，OLMo 2 7B 1e-6，Qwen2.5 7e-7；Llama 3 用了 1e-5（外加下面说的正则）。主线配置默认 5e-7，待调。

**坑 2：chosen 的概率也会一起掉。** 把错答案换成"只差 1 或 2"（chosen 和 rejected 很像），其余不变：

| 步 | margin | acc | log π(chosen) | log π(rejected) |
|---:|---:|---:|---:|---:|
| 0 | +0.000 | 0.00 | −1.045 | −1.984 |
| 25 | +0.050 | 0.70 | −1.131 | −2.574 |
| 50 | +0.107 | 0.74 | −1.134 | −3.144 |
| 75 | +0.156 | 0.79 | −1.138 | −3.639 |
| 100 | +0.201 | 0.79 | −1.254 | −4.201 |
| 125 | +0.258 | 0.79 | −1.442 | −4.960 |
| 150 | +0.357 | 0.84 | −1.662 | −6.168 |

损失在降、margin 在涨、acc 在涨——chosen 的 log 概率却一路下滑，留出题答对的概率 **0.353 → 0.223**，采样答对 0.340 → 0.230，格式 1.000 → 0.887。DPO 只要求"chosen 比 rejected 掉得少"，被压下去的概率可能流到了别处（包括格式错误的回答）。这在大模型上也被反复观察到：Nemotron-4 报告"chosen 和 rejected 的似然都持续下降"，Razin et al. 2024 称之为**似然位移（likelihood displacement）**，并发现 chosen 与 rejected 越像越严重。常见的补救是**在 chosen 上加一项 SFT（NLL）损失**：Llama 3 系数 0.2，Nemotron-4 也加了；Llama 3 还把 chosen 和 rejected 共有的格式 token（header、终止符）从 DPO 损失里屏蔽掉，因为"同一个 token 既要抬高又要压低"会导致尾部重复或突然终止。

**坑 3：过拟合与只看训练指标。** 上面两个实验里，训练集 acc 0.95 / 0.84，留出题只有 0.83 / 0.63。Zephyr 报告 DPO 一个 epoch 后训练准确率就到了 100%。一定要在留出的开发集上看真实指标（格式正确率、工具调用得分、对话评测），而不是训练 margin。

**坑 4：长度偏好。** 第 3 节的奖励模型学会了"长 = 好"，DPO 的隐式奖励同样会学：chosen 往往比 rejected 长，模型就学到"写长一点"（Park et al. 2024）。对策包括长度归一化（Tülu 3、OLMo 2 用的"长度归一化 DPO"：log 概率除以回答长度）、在构造偏好对时控制长度、评测时用长度控制的指标（如 AlpacaEval 2 LC）。

## 9. 小结

- **偏好数据**：同一提示词下 (chosen, rejected) 成对；裁判可以是人、强模型，或能自动判分的程序。
- **Bradley–Terry**：`P(y_w ≻ y_l) = σ(r_w − r_l)`；只有差值有意义。
- **奖励模型**：`−log σ(r_w − r_l)`，就是二分类交叉熵；它会学会偏好，也会学会偏见。
- **RLHF**：`max E[r] − β·KL(π‖π_ref)`；KL 是缰绳，防止 reward hacking；PPO 靠采样 + 价值基线 + 裁剪去解它（第 19 章的铺垫）。
- **DPO**：最优解 `π* = π_ref·e^{r/β}/Z` → 反解 `r = β·log(π*/π_ref) + β·log Z` → 代入 Bradley–Terry，Z 消掉 → `−log σ(β·[Δ_w − Δ_l])`。
- **梯度**：抬 chosen、压 rejected，力度 β·σ(−h)。
- **坑**：学习率太大会坏格式；chosen 也会掉（加 NLL、屏蔽格式 token）；只看训练 margin 会高估；长度偏好。

---

## 从极简到生产级

同一件事在主线代码里是 [`zero/post/dpo.py`](../../zero/post/dpo.py)，配置在 [`configs/tiny/dpo.toml`](../../configs/tiny/dpo.toml) 和 [`configs/main/dpo.toml`](../../configs/main/dpo.toml)，测试在 [`tests/test_dpo.py`](../../tests/test_dpo.py)。

| 极简版（`code/`） | 生产级（`zero/`） | 多做了什么、为什么 |
|---|---|---|
| `03` 的 `dpo_loss` | `zero.post.dpo.dpo_loss(policy_chosen_logps, policy_rejected_logps, ref_chosen_logps, ref_rejected_logps, beta)` | 公式相同；另外返回 `acc`、`margin`、`chosen_reward`、`rejected_reward` 指标，训练时每步打日志（盯"chosen 也在掉"就看 `chosen_reward`） |
| `04` 的 `response_logps`：手工拼 "提示词 + 回答"，mask 掉提示词 | `encode_pair` → `zero.post.chat.encode_prompt_response`：按对话模板（含工具定义、`tool_calls`）渲染，只有助手回复的 token 算进 log 概率；超过 `seq_len` 的偏好对丢弃 | 和第 16 章 SFT 用同一套模板与 loss mask，保证 DPO 优化的正是推理时模型要生成的那段文字 |
| 每个回答单独前向 | `batch_logps`：chosen 和 rejected 拼成一个批次、右侧补齐，一次前向；`zero.post.common.sequence_token_logprobs` / `token_logprobs` 取每个目标 token 的 log 概率（float32） | 一次前向省一半 kernel 启动；log_softmax 在 float32 里做，避免 BF16 下长回答的求和误差 |
| 参考 log 概率训练前算好 | `[dpo] ref_mode = "precompute"`（训练前对全部数据算一遍参考 log 概率）或 `"online"`（每步用一份冻结的拷贝现算） | precompute 省下一整份模型的显存；online 适合数据边生成边训练 |
| 固定的 280 对 | `make_env_preferences`：`generate_pairs > 0` 且文件不存在时，对工具调用环境 `zero/post/envs/tool_env.py` 的任务，从当前策略采样 `samples_per_prompt` 个回答，用可验证奖励打分，最高分（满分才算）当 chosen、否则用标准解答；最低分当 rejected；都满分时把标准调用改坏当 rejected | **on-policy 偏好数据**：rejected 是模型自己真会犯的错；不需要人工标注 |
| Adam、固定学习率 | `zero.post.common.LoopState`：AdamW、warmup + cosine、梯度裁剪、梯度累积、日志 JSONL、checkpoint、断点续训 | 和其他后训练阶段共用一套训练循环 |
| 单进程 CPU | 单进程；CUDA 上用 BF16 autocast | 多卡 DDP 尚未实现；单卡 CUDA + BF16 的通路已在 RTX 3090 上验证（见 [runs/2026-10-01-gpu0-check](../../runs/2026-10-01-gpu0-check/README.md) 第 9 节） |

**对拍**（[`tests/test_dpo.py`](../../tests/test_dpo.py)）：`test_dpo_loss_hand_computed` 用两对手算的例子验证损失、acc、margin，并检查 policy = ref 时损失为 ln 2、梯度抬 chosen 压 rejected；`test_batch_logps_only_counts_response` 验证序列 log 概率只算回复 token，与逐 token 手算一致；`test_run_dpo_end_to_end` 在 precompute / online 两种模式下各跑 3 步，第一步损失 = ln 2、之后下降、checkpoint 落盘。本章 `03` 的 ⑤ 又把从零写的损失和 `dpo_loss` 在数值与梯度上对了一遍。

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
| `init_from` | `out/main/distill/ckpt` | 流水线顺序 SFT → 蒸馏 → DPO → GRPO；参考模型 = 蒸馏后的模型 |
| `[dpo] beta` | 0.1 | 与 DPO 原论文、Zephyr、Llama 3 相同；待用开发集扫 |
| `[optim] lr` | 5e-7，warmup 50 步，cosine 降到 0 | 在 Zephyr / Tülu 3（5e-7）、Qwen2.5（7e-7）、OLMo 2 7B（1e-6）的区间内；冒烟测试证明 lr 太大会伤工具调用格式；待调 |
| `micro_batch_size × grad_accum` | 8 × 8 = 每步 64 对 | Zephyr 32、Tülu 3 128；1000 步 ≈ 6.4 万对 |
| `ref_mode` | `precompute` | 一张卡放得下 0.7B 策略 + 预先算好的参考 log 概率 |
| `seq_len` | 8192 | 工具说明 + 多轮对话较长，超过的偏好对丢弃 |
| `generate_pairs` / `samples_per_prompt` | 0 / 8 | 主线默认读现成的 `data/dpo/prefs.jsonl`；设 `generate_pairs > 0` 则用 tool_env 现场造 on-policy 偏好对 |

**第二步的偏好数据从哪里来**（GOAL.md 3.3："提升通用对话质量"，并保住工具调用格式）：

1. **工具调用：on-policy + 可验证打分**。`make_env_preferences` 已经写好：策略自己采样、tool_env 打分，和 Qwen2 / Qwen2.5 "对错可判定的任务用执行反馈造偏好对" 是同一个思路。
2. **通用对话：开放偏好数据集（注意许可证）**：
   - [HelpSteer3](https://huggingface.co/datasets/nvidia/HelpSteer3)：CC-BY-4.0，人工标注，含中文（`language` 列表含 zh）；
   - [UltraFeedback](https://huggingface.co/datasets/openbmb/UltraFeedback)：MIT，GPT-4 打分；二值化版本 [HuggingFaceH4/ultrafeedback_binarized](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized) 也是 MIT（Zephyr 用的就是它）；
   - [Tülu 3 8B 偏好混合](https://huggingface.co/datasets/allenai/llama-3.1-tulu-3-8b-preference-mixture)：ODC-BY-1.0，但卡片写明"部分子集不可商用"，要逐个子集核对；
   - 这些数据里的回答来自各种模型，裁判多是 GPT-4 / GPT-4o；生成模型与裁判的使用条款是否允许"用输出训练其他模型"，**待核实**（GOAL.md 3.3 的教师许可证规则同样适用于这里）。
3. **仿 Tülu 3 的合成管线**：用许可证允许的开放模型当裁判（与第 17 章的教师同一份许可证清单），对 SFT 模型和几个开放模型的回答打分，最高分当 chosen。
4. 所有偏好数据都要和评测集做 13-gram 去污染（GOAL.md 3.2）。

## 主线进度

### 极小配置演示（CPU，`configs/tiny`，约 1.3M 参数）

> 以下是**极小配置演示**：只说明代码通路是通的，不代表主线模型的任何结果。

> 注：本节数字来自修复工具调用判分器（第 19 章第 6 节）**之前**的那次冒烟测试，是当时的真实输出。修复后重跑（`uv run python -m zero.smoke --out out/smoke_final`），数据、预训练、中期训练、SFT 各阶段完全一致；蒸馏通过验证的样本从 1 条（那句胡话）变为 0 条，下游的 DPO、GRPO 和评测数字随之变化（例如同一个 SFT 模型的工具调用 call_exact 从 0.133 变为 0.100——模型没变，是判分更严了；GRPO 相对 SFT 仍判"持平"）。你自己运行时以运行结果为准。

`uv run python -m zero.smoke` 的 DPO 阶段（`out/smoke/SUMMARY.md`、`out/smoke/dpo/log.jsonl`；smoke 把 `seq_len` 设为 512、造 32 对、跑 24 步，其余同 `configs/tiny/dpo.toml`：β = 0.1、lr = 5e-5、precompute）：

- 偏好数据：32 对，全部由 tool_env 现场造；其中 chosen 来自策略自身采样的 3 对，其余 29 对用标准解答（tiny 模型太弱，很少自己答对）。
- 训练日志：

| 步 | loss | acc | margin | chosen_reward | rejected_reward | lr |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.6931 | 0.00 | +0.000 | +0.000 | +0.000 | 1.25e-5 |
| 5 | 0.6941 | 0.50 | −0.001 | +0.003 | +0.005 | 5.00e-5 |
| 10 | 0.6125 | 1.00 | +0.170 | +0.060 | −0.110 | 4.34e-5 |
| 15 | 0.3797 | 1.00 | +0.802 | +0.011 | −0.790 | 2.75e-5 |
| 20 | 0.3531 | 1.00 | +0.932 | +0.017 | −0.915 | 1.16e-5 |
| 24 | 0.5262 | 0.75 | +0.394 | −0.020 | −0.414 | 5.28e-6 |

- 第 1 步损失正好 ln 2（策略 = 参考模型）；之后 rejected_reward 一路变负、chosen_reward 在 0 附近徘徊，最后一步略为负——和第 8 节"chosen 不一定涨"是同一个现象的迹象（每步只有 4 对，噪声很大，不能下结论）。
- 学习率：最初用 5e-4 时，24 步 margin 冲到 4.8，接下来 GRPO 阶段的格式正确率从 0.16 掉到 0；改成 5e-5（见 `configs/tiny/dpo.toml` 注释）。
- 之后的 eval 阶段比较的是 SFT 与最终 GRPO 模型，没有单独评测 DPO 模型的效果。

### 待 GPU 训练后补充

- 主线偏好数据的来源、规模、许可证清单与去污染结果；
- β、学习率在开发集上的扫描结果（重点看工具调用格式正确率与通用对话评测，不看训练 margin）；
- 训练曲线（loss、acc、margin、chosen_reward / rejected_reward）、花费（记入 `runs/ledger.md`）；
- DPO 前后在开发集上的对比，以及 DPO 对下一步 GRPO 的影响；
- 如果出现"chosen 一起掉"或格式退化：是否需要加 chosen 上的 NLL 项、屏蔽格式 token（`zero` 目前都没有实现，见下面的说明）。

**说明**：`zero` 当前实现的是原版 DPO，没有 Llama 3 的"NLL 正则 + 屏蔽格式 token"，也没有 Tülu 3 的长度归一化。这几项改动都很小，第二步若在开发集上看到对应问题，再加上并做对照实验。

---

## 前沿观察

> **DPO 的变体**：IPO（Azar et al. 2023，把 −log σ 换成平方损失以防过拟合）、KTO（Ethayarajh et al. 2024，只需"好/坏"单条标注而不必成对）、SimPO（Meng et al. 2024，去掉参考模型、用长度归一化的 log 概率当奖励）、ORPO（Hong et al. 2024，把偏好项并进 SFT 损失、不要参考模型）等层出不穷；头部开源模型里只看到零散采用（如 SmolLM3 用 APO，Nemotron-4 用自家的 RPO，Tülu 3 比较后选了长度归一化 DPO 而非 SimPO），按 GOAL.md 2.1 只在这里一句带过。

---

## 采用方与来源

| 技术 | 采用方（主力版本） | 来源 |
|---|---|---|
| DPO | **Llama 3**（每轮 SFT 后做 DPO；lr 1e-5、β = 0.1；屏蔽格式 token + 0.2 倍 NLL；比较过 PPO，认为 DPO 算力更省、IFEval 更好）；**Qwen2**（离线 DPO + 在线阶段用奖励模型选最好/最差回答做 DPO）；**Qwen2.5**（离线 DPO 约 15 万对、lr 7e-7，之后在线 GRPO）；**DeepSeek LLM**（SFT 后 DPO，lr 5e-6，batch 512）；**Tülu 3 / OLMo 2**（长度归一化 DPO + on-policy 合成偏好数据，Tülu 3 8B：lr 5e-7、β = 5）；**Nemotron-4 340B**（DPO 后再三轮 RPO，并在 chosen 上加 SFT 损失）；**Zephyr**（在 UltraFeedback 上 dDPO，lr 5e-7、β = 0.1） | [Llama 3](https://arxiv.org/abs/2407.21783) §4.1.4；[Qwen2](https://arxiv.org/abs/2407.10671) §4.3；[Qwen2.5](https://arxiv.org/abs/2412.15115) §4.2–4.3；[DeepSeek LLM](https://arxiv.org/abs/2401.02954) §4；[Tülu 3](https://arxiv.org/abs/2411.15124) §5；[OLMo 2](https://arxiv.org/abs/2501.00656) §5；[Nemotron-4 340B](https://arxiv.org/abs/2406.11704) §3.3.2；[Zephyr](https://arxiv.org/abs/2310.16944) §4.4 |
| DPO 变体（APO） | **SmolLM3**（APO，"DPO 的一个更稳定的变体"；非推理模式用 Tülu 3 偏好数据，推理模式用 Qwen3-32B 当 chosen、Qwen3-0.6B 当 rejected） | [SmolLM3 博客](https://github.com/huggingface/blog/blob/main/smollm3.md)；[APO](https://arxiv.org/abs/2408.06266) |
| 奖励模型 + PPO（铺垫） | **InstructGPT**（6B 奖励模型、PPO、β = 0.02、PPO-ptx）；Llama 2（奖励模型 + 拒绝采样 + PPO）；Tülu 3 做过 PPO 与 DPO 的对照 | [InstructGPT](https://arxiv.org/abs/2203.02155) §3.5、附录 C.4；[Stiennon et al. 2020](https://arxiv.org/abs/2009.01325)；[PPO](https://arxiv.org/abs/1707.06347) |

**共识判断（GOAL.md 2.1）**：DPO 被 Llama、Qwen、DeepSeek、OLMo、Nemotron 至少 5 个彼此独立的头部家族在主力版本里明确采用（SmolLM3 用其变体），满足规则 A，进正文；奖励模型 + PPO 作为 DPO 的推导起点和第 19 章 GRPO 的铺垫（规则 C）。

**说明**：更新一代的模型（如 Qwen2.5 之后）把更多后训练预算放到了在线 RL（GRPO，第 19 章）上；DeepSeek-V3、Qwen3 等最新主力版本的后训练是否仍含 DPO，本章未逐一核实（待核实）。

---

## 引导问题

带着这些问题去问 Claude Code，直到你能用自己的话讲清楚：

1. 奖励模型的偏置为什么梯度恒为 0？如果 InstructGPT 不在 RL 之前把示范答案的平均奖励归零，PPO 会受影响吗？DPO 需要这一步吗？
2. `02` 的表里，β = 0.1 时"注水"的概率是 0.61，而它在 π_ref 下只有 0.004。用 `π* ∝ π_ref·e^{r/β}` 算一算：要多大的奖励差，才能让一个 0.4% 的回答翻到 61%？这说明奖励模型在"没见过的区域"犯一点错会有多危险。
3. DPO 推导的第 4 步"用 π_θ 代替 π*"其实隐含了一个假设：策略能表示任意分布、偏好数据覆盖了所有回答。数据只覆盖一小部分回答时，DPO 对"没出现过的回答"的概率有约束吗？这和第 8 节"chosen 也会掉"有什么关系？
4. 为什么说 DPO 是 **off-policy** 的，而 PPO 是 on-policy 的？Llama 3 为什么"主要用最近一轮最好模型采集的偏好数据"？`make_env_preferences` 属于哪一种？
5. 长度归一化 DPO 把 log 概率除以回答长度。它和原版 DPO 的最优解还是 `π_ref·e^{r/β}` 吗？为什么 Tülu 3 的 β 要设到 5 这么大？
6. 如果偏好数据里的"裁判"本身就是一个奖励模型（例如 Nemotron-4 用奖励模型给合成回答排序），DPO 和 RLHF 的区别还剩下什么？

## 动手任务

每个任务都要真的运行代码、看到结果。

**任务 1（基础）**：修改 `01_bradley_terry.py`，把标注员的长度系数从 0.4 改成 −0.4（标注员讨厌长回答），重新训练奖励模型，再跑 `02`：reward hacking 还会发生吗？最好的 β 变了吗？

**任务 2（核心）**：在 `04_toy_dpo.py` 的 `run_dpo` 里加一项 chosen 上的 NLL 损失（`loss = dpo + α·(−pw.mean() / 回答 token 数)`，α 取 0.2，仿 Llama 3），然后用 `05` 的"错答案只差一点"设置重跑：chosen 的 log 概率还会一路掉吗？留出题答对的概率变成多少？

**任务 3（挑战）**：实现长度归一化 DPO（log 概率除以回答 token 数），构造一个"chosen 总是比 rejected 长"的偏好数据（例如 chosen 是 `12;`，rejected 是 `9;`，同时让一部分正确答案写成 `12!;` 这种更长的形式），比较原版 DPO 与长度归一化 DPO 训练后模型生成的平均长度。

---

## 想深入：CS336

本章对应斯坦福 CS336（Spring 2026）<https://cs336.stanford.edu/>：

- **第 15 讲：中期训练与后训练（SFT / RLHF）**。讲 InstructGPT 式的 RLHF 流水线、奖励模型、PPO 与 DPO 的关系，是本章第 3–5 节的英文原版。
- **作业 5（Alignment and Reasoning RL）的可选第二部分：DPO**。在真实的小模型和真实偏好数据上实现 DPO——本章 `03`、`04` 的放大版。作业 5 的主体（SFT、专家迭代、GRPO）对应第 16、19 章。

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

**下一章**：偏好对只能告诉模型"哪个更好"，而且数据是事先收集好的。工具调用这类任务有一个更好的条件：对错可以自动判定。那就让模型自己去试、当场拿分数、从自己的尝试里学——第 19 章，强化学习：从 PPO 到 GRPO，以及可验证奖励。
