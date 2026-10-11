# 第 19 章：强化学习 —— 模型从自己的尝试中学习

[English](README.md) · **中文**

> **目标**：读完这一章，你能从"对数导数技巧"推出策略梯度。你能讲清楚 GRPO 为什么不需要价值模型、组内优势怎么算、裁剪和 KL 各管什么。你能给工具调用设计一个可验证奖励。训练曲线"看起来很好"时，你能认出奖励作弊（reward hacking），找到漏洞，并补上守卫。

📺 **本章视频**：还没有发布。要在本机渲染，运行 `bash chapters/19-reinforcement-learning/video/build.sh`。
🧪 **本章自检**：学完后，在 Claude Code 里输入 `/ch19-rl`。

---

上一章我们把模型对齐到人的偏好。我们先讲 RLHF（奖励模型 + PPO）作为铺垫，再从同一个目标推出 DPO。现在，模型会说话，会用工具调用的格式，回答也更讨人喜欢。但它学到的一切都来自**别人给的答案**：SFT 模仿示范，蒸馏模仿老师，DPO 模仿"哪个更好"的标注。

这一章要解决的问题是：**模型会模仿，但不会解题**。示范里没有的能力，它从哪里学？答案是强化学习（reinforcement learning，RL）。模型自己去尝试，一个能自动判断对错的程序（验证器，verifier）给每次尝试打分，得分高的做法变得更常见。这是主线模型的最后一个训练阶段：在工具调用环境里做 GRPO。

## 1. 模仿的上限

先看一个极小的例子（[`code/02_grpo_from_scratch.py`](code/02_grpo_from_scratch.py)）。任务是两个个位数相加。模型先输出答案的数字，再输出 `<eos>`。

老师是一个"不太会进位"的模型：不进位的题全对；进位的题只有 30% 写对，另外 70% 忘了写进位（7+5 写成 "2"）。学生用老师的 2000 条示范做 SFT：

| | 准确率 |
|---|---:|
| 老师示范（2000 条） | 0.684 |
| SFT 学生，采样 | 0.68 |
| SFT 学生，贪心解码 | 0.60 |
| SFT 学生，贪心解码，只看进位题 | **0.11** |

学生学得"很好"：采样准确率和老师几乎相同。老师的错误，学生也照样学会了。进位题上，老师多数时候写错，所以学生最有把握的答案也是错的，贪心解码（greedy decoding）只剩 0.11。**模仿学习的上限就是示范数据。**示范里的每一处偏差，学生都会照样继承。

但请注意一个细节：学生采样时，进位题仍有约 30% 的概率写对。正确答案**在模型的分布里**，只是不是概率最大的那个。如果一个程序能告诉模型"这次对了、那次错了"，就能把这 30% 放大。

判断两个数的和对不对，一行代码就够。这样的奖励叫**可验证奖励**（verifiable reward）。用它做强化学习，60 步之后：

| | SFT 之后 | GRPO 第 5 步 | 第 10 步 | 第 60 步 |
|---|---:|---:|---:|---:|
| 贪心准确率 | 0.60 | 0.96 | 1.00 | 1.00 |
| 贪心准确率（进位题） | 0.11 | 0.91 | 1.00 | 1.00 |
| 采样准确率 | 0.68 | 0.88 | 0.95 | 0.98 |

学生超过了老师。下面几节把中间发生的事逐步拆开讲。

## 2. 策略梯度：分数不可导，梯度从哪来

把语言模型看成一个**策略**（policy）π_θ：给定提示词 x，它按概率生成回答 y。我们想最大化期望奖励：

```
J(θ) = E_{y~π_θ(·|x)} [ r(x, y) ]
```

问题在于 r 是一个判分程序（"答案对不对"）。它对 θ 不可导，梯度无法从它反传回去。**对数导数技巧**（log-derivative trick）绕开了这个问题：

```
∇J = ∇ Σ_y π_θ(y) r(y)
   = Σ_y r(y) ∇π_θ(y)
   = Σ_y π_θ(y) r(y) ∇log π_θ(y)        （因为 ∇π = π · ∇log π）
   = E_{y~π_θ} [ r(y) ∇log π_θ(y) ]
```

最后一行是一个期望，所以可以用采样来估计。让模型生成几个回答，把每个回答的"对数概率的梯度"乘上它的得分，再求平均。r 只作为一个数出现。**r 不需要可导，只需要能打分。**这就是 REINFORCE（Williams，1992）。直觉很直接：得分高的回答，把它的概率往上推；得分低的回答，把它的概率往下压。

对语言模型，log π_θ(y) = Σ_t log π_θ(y_t | x, y_<t)。所以 ∇log π 就是回答里每个 token 的对数概率梯度之和。这和 SFT 交叉熵的梯度完全相同，区别只是每条样本多乘了一个"得分"系数。SFT 可以看成所有样本的系数都等于 1 的特例。（DeepSeekMath 论文第 5 节把 SFT、拒绝采样、DPO、PPO、GRPO 统一写成"梯度系数"的不同选法。）

[`code/01_reinforce_bandit.py`](code/01_reinforce_bandit.py) 在一个 5 臂老虎机（bandit）上验证这件事。策略是 π = softmax(θ)。脚本把采样估计和精确梯度做对拍：

| 样本数 N | 采样估计与精确梯度的最大差 |
|---:|---:|
| 100 | 0.0217 |
| 10,000 | 0.0010 |
| 1,000,000 | 0.0001 |

## 3. 方差与基线

REINFORCE 是无偏的，但方差很大。一个关键的观察：对任意常数 b，

```
E[ (r − b) ∇log π ] = E[ r ∇log π ] − b · E[∇log π] = ∇J       （因为 E[∇log π] = Σ ∇π = ∇1 = 0）
```

减去一个**基线**（baseline）b，期望不变，方差却可以大幅改变。直觉是这样的：假设所有回答都得 10 分左右。不减基线，每个回答都被"往上推"，只是推的幅度略有不同，信号淹没在噪声里。减掉平均分之后，高于平均的回答被推上去，低于平均的被压下来，信号干净得多。`01` 的输出：

| 奖励 | 基线 b | 均值与精确梯度的最大差 | 方差（各分量求和） |
|---|---|---:|---:|
| 原始 r | 0 | 0.0004 | 0.1946 |
| 原始 r | E[r] | 0.0004 | 0.0534 |
| r + 10 | 0 | 0.0076 | 84.70 |
| r + 10 | E[r] | 0.0006 | 0.0533 |

所有奖励加 10 时，不减基线的方差是减基线的 **1589 倍**。用同样的学习率训练 300 步（每步 8 个样本，5 个种子取平均），最好那只手臂的概率：

| | 第 1 步 | 第 50 步 | 第 100 步 | 第 300 步 |
|---|---:|---:|---:|---:|
| 无基线 | 0.21 | 0.20 | 0.20 | 0.40 |
| 基线 = 这一批的平均奖励 | 0.21 | 0.85 | 0.95 | 0.99 |

请记住这个做法："用同一批样本的平均奖励作基线"。GRPO 的核心就是它。

## 4. 从 PPO 到 GRPO

### 4.1 PPO 回顾（第 18 章的铺垫）

PPO（Proximal Policy Optimization）是 RLHF 的标准算法。它在 REINFORCE 上加了两样东西：

1. **价值模型（critic）当基线**：训练一个和策略差不多大的模型 V(s)，预测"从这个位置往后大概还能得多少分"。然后用 GAE 算出每个 token 的优势（advantage）A_t。
2. **裁剪的重要性比**（importance ratio）：PPO 在同一批样本上更新好几次，以节省采样。第二次更新时，策略已经变了。所以 PPO 用 ρ_t = π_θ(y_t)/π_old(y_t) 修正，并把 ρ 限制在 [1−ε, 1+ε] 里，防止一次更新走得太远：

```
ℓ_t = −min( ρ_t · A_t,  clip(ρ_t, 1−ε, 1+ε) · A_t )
```

裁剪（clipping）的效果要按优势的正负分开看。A > 0（好回答）时，ρ 超过 1+ε 就不再有梯度："已经加够了，不要再推"。A < 0（坏回答）时，ρ 低于 1−ε 就不再有梯度："已经压够了"。

问题在于价值模型。它和策略一样大，要多占一份显存，还要多训练一个模型。另外，语言模型的奖励通常只在回答末尾给一个数。让价值模型在每个 token 上都估得准，并不容易（这是 DeepSeekMath 论文 4.1.1 节的说法）。

### 4.2 GRPO：用"同一个提示词的其他回答"当基线

GRPO（Group Relative Policy Optimization，DeepSeekMath，2024）**把价值模型整个去掉**。它对每个提示词采样 G 个回答，用验证器打分得到 r_1..r_G，再在组内归一化，得到优势：

```
A_i = ( r_i − mean(r_1..r_G) ) / std(r_1..r_G)
```

回答 i 里的所有 token 共享同一个 A_i。这就是第 3 节的"批均值基线"，只是按提示词分组。同一道题上，高于平均的回答被推上去，低于平均的被压下来。除以标准差，是把不同题目的信号缩放到同一尺度。下面是 `02` 第 1 步里一组真实的样本（题目 2+8，G = 8）：

| 回答 | 10 | 0 | 0 | 10 | 10 | 0 | 10 | 0 |
|---|---|---|---|---|---|---|---|---|
| 奖励 | 1 | 0 | 0 | 1 | 1 | 0 | 1 | 0 |
| 优势 | +0.94 | −0.94 | −0.94 | +0.94 | +0.94 | −0.94 | +0.94 | −0.94 |

（4 对 4 错时，均值是 0.5，无偏标准差是 0.535，所以优势是 ±0.94。）"0" 就是那个忘了写进位的答案。

对应的代码只有几行：

```python
def group_advantages(rewards, eps=1e-6):          # rewards: (P, G)
    mean = rewards.mean(1, keepdim=True)
    std = rewards.std(1, keepdim=True)            # unbiased std, the same as TRL / verl / zero
    return (rewards - mean) / (std + eps)         # A_i = (r_i − mean) / std

def grpo_loss(logp, old_logp, ref_logp, adv, mask, eps_clip=0.2, beta=0.02):
    ratio = torch.exp(logp - old_logp)                                 # ρ_t = π_θ / π_old
    A = adv[:, None]
    per_tok = torch.maximum(-ratio * A,
                            -torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * A)   # clipping
    d = ref_logp - logp
    kl = torch.exp(d) - d - 1                                          # k3: unbiased KL estimate, never negative
    per_tok = per_tok + beta * kl
    m = mask.float()
    return (per_tok * m).sum() / m.sum()                               # token_mean aggregation
```

完整的训练循环是：取 P 道题 → 每道题复制 G 份并采样 → 验证器打分 → 算组内优势 → 用上面的损失更新 μ 次。`02` 用 P = 16、G = 8、ε = 0.2、β = 0.02、μ = 2。

**一组全对或全错时，优势全为 0，这一组对梯度没有贡献。**在 `02` 的日志里，"零方差组"的比例从第 5 步的 0.56 升到第 20 步的 0.94。模型越学越好，能提供信号的题就越来越少。这是 GRPO 在实践中最常遇到的问题之一。第 5 节的"按难度筛题"就是为它准备的。

### 4.3 三个细节：KL、聚合方式、裁剪什么时候起作用

**KL 约束与 k3 估计。**可以选择加一项 β · KL(π_θ ‖ π_ref)，让策略不要离起点（SFT/DPO 模型）太远。GRPO 把这一项直接加在损失里（RLHF 则是从奖励里扣除）。GRPO 逐 token 用 k3 估计它：

```
KL ≈ π_ref/π_θ − log(π_ref/π_θ) − 1        （Schulman 的 "Approximating KL"，DeepSeekMath 式 4）
```

k3 估计总是非负（x − log x − 1 ≥ 0），期望等于真正的 KL。用可验证奖励时，很多团队把 KL 整个去掉（β = 0）。GLM-4.5 的推理 RL "在 GRPO 框架上去掉 KL 项"，MiMo-7B 和 OLMo 3 也都去掉了。DeepSeek-R1 保留了一个很小的 β = 0.001。

理由是：RLHF 里的 KL 防止策略跑到奖励模型不可信的区域；规则验证器不会"不可信"，限制也就没那么必要。主线配置 `configs/main/grpo.toml` 默认 `kl_coef = 0.0`，tiny 配置用 0.02 演示参考模型的用法。

**聚合方式（aggregation）。**一批回答长短不一，逐 token 的损失怎么平均？原始 GRPO 论文先在每条回答内平均，再在回答之间平均（`seq_mean_token_mean`）。这样，长回答里每个 token 的权重被稀释：一条又长又错的回答，每个 token 受到的惩罚更轻。DAPO 改成所有回答的 token 一起平均（`token_mean`）。

MiMo-7B 的目标函数就是 1/Σ|o_i| 的 token 级平均。OLMo 3 明确采用 token 级损失，以"避免长度偏差"。GLM-4.5 在代码 RL 上比较过两种写法，token 加权平均收敛更快。`zero` 默认 `loss_agg = "token_mean"`。

**裁剪什么时候起作用。**如果每批样本只更新一次（μ = 1），π_old 就是当前策略，ρ ≡ 1，裁剪永远不会触发。但 ∇ρ = ∇log π 不为零，所以梯度照样存在，方法退化成带组基线的 REINFORCE。只有同一批样本更新多次时（μ > 1，或者把大批次拆成多个小批次依次更新），ρ 才会偏离 1。`02` 用 μ = 2，裁剪比例在 0.00–0.02 之间。冒烟测试的 tiny 配置用 `ppo_epochs = 1`，所以日志里的 clip 一直是 0.00。

## 5. GRPO 的各种改进：哪些进正文，哪些还在观察

2025 年以来，出现了一大批 GRPO 的改进。我们按 GOAL.zh.md 第 2.1 节的规则逐项核对：至少 3 个独立的头部开源模型家族在技术报告里明确采用。结论如下：

| 做法 | 谁明确采用（技术报告） | 结论 | zero 里 |
|---|---|---|---|
| 组内基线、不要价值模型（GRPO 本身） | DeepSeek-R1、Qwen3（推理 RL）、GLM-4.5、MiMo-7B、OLMo 3；Kimi K2 用组均值基线，但目标函数是 K1.5 的变体 | **共识** | `group_advantages` |
| token 级聚合 | MiMo-7B、OLMo 3、GLM-4.5（代码 RL）；出处是 DAPO | **共识**（3 家） | `loss_agg = "token_mean"`（默认） |
| 去掉 KL 项 | GLM-4.5、MiMo-7B、OLMo 3；DeepSeek-R1 仍保留 β = 0.001 | **常见选择**（3 家），不是必须 | `kl_coef = 0.0`（main） |
| 按难度筛题：去掉全对或全错的题 | MiMo-7B（动态采样 + 简单题重采样）、OLMo 3（零梯度过滤 + active sampling）、GLM-4.5（按难度的课程学习）、Kimi K2（按 SFT 模型的 pass@k 只留中等难度）、Qwen3（query 要"对冷启动模型可学"） | **共识的原则**；在线动态采样的具体做法各家不同 | 记录 `zero_std_groups`。按开始 RL 的 checkpoint 的 pass@k 离线筛题：`zero/post/difficulty.py`（保留 1/k ≤ pass ≤ (k−1)/k）。训练循环内的在线过滤**未实现** |
| clip-higher（ε_high > ε_low） | MiMo-7B、OLMo 3；出处是 DAPO（字节 Seed） | **待核实**：明确采用的头部家族只找到 2 家 | `clip_eps_high`（main 设 0.28，设 0 即关闭） |
| 不除以标准差（Dr. GRPO） | OLMo 3 | 前沿观察 | `scale_rewards = false` 可切换 |
| 序列级重要性比（GSPO） | Qwen3 的后续版本（GSPO 论文自述） | 前沿观察 | 未实现 |

"待核实"不等于"没用"。在 DAPO 的消融实验里，clip-higher 把 AIME 从 36 提到 38（Qwen2.5-32B 基座），MiMo 和 OLMo 3 也都用了它。只是按本课的规则，第三家的明确证据还没有找到。主线配置把它保留为一个开关，改一项设置就能关掉。第二步用我们自己的开发集（不是预注册的测试基准）决定开不开。

## 6. 可验证奖励：怎么给工具调用打分

奖励的质量决定了强化学习效果的上限。DeepSeek-R1 在推理任务上**只用规则奖励**（准确率奖励 + 格式奖励），明确不用神经网络奖励模型。理由是"神经奖励模型在大规模 RL 中容易被 reward hacking"。Qwen3 的报告也说："设计良好的规则奖励能高精度地判断输出对错，防止 reward hacking"。在这类强化学习里，程序可以检查答案。Tülu 3 给它起名叫 **RLVR**（Reinforcement Learning with Verifiable Rewards）：

| 领域 | 验证器 | 例子 |
|---|---|---|
| 数学 | 抽出最终答案，和标准答案做等价比较 | DeepSeek-R1（答案写在 box 里）、OLMo 3（用 SymPy 比较） |
| 代码 | 在沙箱里运行测试用例 | DeepSeek-R1、MiMo-7B（按测试难度给部分分）、OLMo 3（按通过率给分，或全部通过才给分） |
| 指令遵循 | 每条约束一个检查函数 | Tülu 3、OLMo 3（满足的约束所占的比例） |
| 工具调用 | 格式检查 + 调用与标准调用比对 / 执行结果比对 | GLM-4.5（格式对、且和标准调用完全一致才给 1）、本课的 `tool_env` |

小米开源的 [XiaomiMiMo/verl](https://github.com/XiaomiMiMo/verl) 复现了 MiMo-V2.6 的五类 RL 环境。这五类环境正好展示了验证器的三种主要形态：**可执行测试**（代码：软件工程任务）、**规则检查**（网络安全的漏洞复现、符号音乐创作）、**rubric 判分**（通用知识工作：一组"通过/不通过"的断言，由 LLM 裁判逐条判定）。另外还有网页开发的视觉判分。读它的代码，能看到几条值得照着做的工程原则：

- **环境故障和模型失败分开记。**沙箱超时、容器崩溃这类基础设施错误，要得到一个显眼的无效值，并单独统计。不要悄悄记成 0 分，否则模型会因为环境的问题受到惩罚。
- **判分器本身也会静默退化。**代码注释里记录过一个例子：某个评分脚本因为路径问题找不到依赖，悄悄降级成部分分恒为 0 的打法。只有靠监控才能发现。
- **单独维护一份"防作弊"配置**（配置名里带 antihack）。
- 调试用的随机奖励开关会把真实分数另存一份（`true_reward`）。训练用的奖励和真实分数分开记录。

主线模型的环境在 `zero/post/envs/tool_env.py`。它有 6 个模拟 API：计算器、天气、单位换算、日期加减、日期间隔、星期几。这些 API 全部是确定性的，不联网。约 10% 的题不需要调用工具，考的是模型"不该调用时就不调用"。打分规则：

```
格式错误（JSON 坏了、标签不配对、伪造 <tool_response>、超长、调用过多）→ −1
不该调工具却调了 → −0.5；该调却没调 → 0
否则  0.1（格式分）+ 0.9 × Σ 每个标准调用的得分 / 标准调用数
      − 0.25 × 多余调用数 − 0.5 × 不合 schema 的调用数
      每个标准调用：函数名 + 参数一致 → 1；只有函数名对 → 0.2
不需要工具的题：正常回答 → +0.5（内容无法自动核对）；
      空回复或凭空编出数字 → 0
```

"参数一致"指参数规范化之后逐项相等。日期统一成 ISO 格式，单位和城市名统一写法，数字统一成数值。BFCL 也用这个思路。只有计算器额外接受"执行一致"：判分器真的算一遍，结果相同、且表达式里的数字相同，就算对（`"28-44"` 和 `"28 - 44"` 都算对）。其他工具为什么不接受执行一致，见本节最后。[`code/04_tool_env_rewards.py`](code/04_tool_env_rewards.py) 把几种典型输出交给生产级判分器：

| 输出（题目："帮我算一下 28 - 44 等于多少？"） | 奖励 | 理由 |
|---|---:|---|
| 正确调用 | +1.00 | |
| 标签里的 JSON 坏了 | −1.00 | `tool_call is not valid JSON: Expecting value`（不是合法 JSON） |
| 去掉标签的裸 JSON | −1.00 | 守卫 #8：`Tool call without <tool_call> tags (name / arguments JSON outside the tags)`（标签外出现 name / arguments JSON） |
| 心算出 −16，调用 `calculator("-16")` | +0.28 | 守卫 #2：`Wrong arguments for calculator`。执行结果一样，但表达式里的数字不对，只给函数名分 |
| 同一个调用输出两遍 | +0.75 | 守卫 #1：`Extra calls: 1`。一对一匹配，多余调用扣 0.25 |
| 调用后自己编一段 `<tool_response>` | −1.00 | 守卫 #3：`The output contains <tool_response> (a forged tool result or conversation turn)`（伪造工具结果） |
| 闲聊题（"Thanks a lot."）：正常回答 / 空回复 / 裸 JSON / 乱调用 | +0.5 / 0 / −1 / −0.5 | `Correctly made no tool call (the answer content cannot be checked automatically)` / `Empty answer` / 守卫 #8 的理由 / `Called a tool when no tool was necessary`。正常回答只给 0.5：没有标准答案可以核对 |

每一条守卫（guard）都对应一种"不学能力也能拿分"的捷径。每条守卫在 `tests/test_tool_env.py` 里都有测试。其中第 8 条不是事先想到的，而是训练中**真实发生**之后补上的。这是下一节的故事。

**后来，判分器又修了两处漏洞。**这两处是写第 17 章（蒸馏）时发现的，都来自"只看结果、不看过程"：

1. **结果碰巧相同就算对。**旧版对所有工具都接受"执行一致"。题目问"2024-06-03 是星期几"，模型调用 `weekday("2024-06-10")`。日期错了 7 天，但星期几一样，判分器判为执行一致，给了满分。现在的修法：除计算器外，一律要求规范化后的参数相等（`args_equivalent`）。计算器保留执行匹配，因为它有"数字多重集合一致"的额外检查（守卫 #2）。
2. **不需要工具的题，不检查回答内容。**旧版在闲聊题上只看"有没有乱调用"，正常回答给满分。冒烟测试里，唯一一条"通过验证"的教师样本是乱码：题目是"讲一句鼓励的话。"，回答是"坚下云，气温 28°C。"。现在的修法：正常回答只给 `NO_TOOL_REWARD = 0.5`，并标记为"无法核对"，蒸馏筛选不会把它当成已验证样本。回答里出现题目中没有的数字（凭空编造的事实），直接得 0 分。

两处都有回归测试（`tests/test_tool_env.py` 里的 `test_wrong_args_with_coincident_result_not_full_credit` 和 `test_no_tool_invented_numbers_get_zero`）。教训很直接：**验证器要检查"结果是怎么得到的"，而不只是"结果对不对"**。这也是 BFCL 用 AST 匹配参数、而不是只比较执行结果的原因。

## 7. 奖励作弊：奖励涨了，能力没涨

### 7.1 真实案例：十步之内学会"去掉标签"

写 `zero` 的冒烟测试（smoke test）时，最初的奖励只检查 `<tool_call>` 标签**里面**的内容。tiny 模型（约 1.3M 参数）的调用几乎都是坏 JSON，得 −1 分。GRPO 跑了十步，平均奖励曲线漂亮地上升。检查样本才发现，模型学会了**去掉标签，照样输出 JSON**。没有标签就不算"调用"，也就没有格式错误，得分从 −1 变成 0（"该调却没调"）。格式错误率降了，奖励涨了，模型却一个调用都不会了。

同时还有第二个漏洞：不需要工具的题，只要不调用就给满分。回复为空、写一串乱码，也能得 1 分。

修法就是 `tool_env` 的第 8 条守卫：标签外面出现 `{"name": ...` / `"arguments":` 这样的 JSON，判为格式错误（−1）。闲聊题回复为空，得 0 分而不是 1 分。

### 7.2 在玩具上重演一遍

[`code/03_reward_hacking.py`](code/03_reward_hacking.py) 把这件事缩小重演。玩具版的正确调用是 `<call> 字 d </call> <eos>`（"函数名 + 参数"，d = (a+b) mod 10）。这里的 `字` 是一个 token，代表普通文字或函数名。起点模型和 tiny 模型一样：59% 的输出格式是坏的，格式对的调用只有 5%。两种奖励只差第 8 条守卫。用同一个起点、同一个随机种子，各跑 120 步 GRPO：

| | 起点 | 天真奖励，第 10 步 | 天真奖励，第 120 步 | 修好的奖励，第 10 步 | 修好的奖励，第 120 步 |
|---|---:|---:|---:|---:|---:|
| 训练奖励（这一批的平均） | — | +0.11 | +0.06 | +0.16 | +0.22 |
| 格式错误率 | 0.59 | 0.01 | **0.00** | 0.10 | 0.00 |
| 工具题：格式正确的调用 | 0.05 | 0.01 | **0.00** | 0.87 | 1.00 |
| 裸调用（标签外出现"调用"） | 0.19 | 0.85 | 0.66 | 0.01 | 0.00 |
| 闲聊题：正常回复 | 0.55 | 0.33 | **0.09** | 0.97 | 1.00 |
| 真实成功率（留出的判据） | 0.14 | 0.08 | **0.02** | 0.29 | 0.36 |

天真奖励下，格式错误率从 0.59 降到 0，训练奖励为正。如果只看这两条曲线，你会以为训练很成功。但格式正确的调用从 0.05 降到 0，闲聊题的正常回复从 0.55 降到 0.09，真实成功率从 0.14 降到 0.02。最后一步的样本：

```
naive reward   tool 5+4  →  字 5 <eos>                    reward +0.0   (removed the tags)
               chat #21  →  <eos>                         reward +1.0   (empty reply gets the full score)
fixed reward   tool 5+4  →  <call> 字 7 </call> <eos>     reward +0.1
               chat #21  →  字 <eos>                      reward +1.0
```

作弊为什么来得这么快？看第一个 token 的决策。起点模型的示范里，"开标签"的输出约 96% 是坏的（−1 分）；在天真奖励下，"不开标签"稳定得 0 分。组内优势会把第一个 token 大力推向"不开标签"。一旦不开标签成了习惯，写对格式的样本就很少被采到，也就很少有机会得到奖励。

**策略梯度只会沿着最容易涨分的方向走，它不知道你真正想要什么。**

修好之后，格式问题解决了，真实成功率从 0.14 升到 0.36，但也就到这里。这个极小的模型没有学会 (a+b) mod 10，工具题的答案仍然基本靠猜。最后一批 120 个工具题回答里，约三分之二猜 7，其余分散在 0、8、1、5 等几个数上；上面的样本 5+4 → 7 就是一例。

守卫堵住的是捷径，能力还得靠模型、数据和更长的训练去学。另外请注意：**两种奖励的数值不能互相比较**。修好的奖励在第 120 步只有 +0.22，模型却远好于天真奖励下的模型。换了奖励函数，曲线的高低就不再有意义。

### 7.3 一般的防守办法

1. **格式要严。**宁可把"差不多对"的输出判成格式错误，也不要留出"绕开检查"的缝。GLM-4.5 的函数调用奖励是"格式对、且和标准调用完全一致才给 1，否则给 0"。
2. **惩罚退化输出**：空回复、超长、重复、伪造工具结果、把所有可能的调用都输出一遍。Tülu 3 给没有输出 EOS 的回复 −10 分。Kimi K2 在指令遵循上专门加了一层 hack-check，检测"声称完成了、实际没有完成"的输出。
3. **用留出的验证器和人工抽查。**训练用的奖励和评测用的判据要分开。每隔一段时间看一次样本。要问的不是"平均奖励是多少"，而是"这个输出真的对吗"。MiMo 的代码把训练奖励和 `true_reward` 分开记录，也是这个原因。
4. **优先用规则，而不是模型打分。**DeepSeek-R1 说，神经奖励模型在长时间 RL 中会被利用漏洞，所以第二阶段只在最后 400 步加入偏好奖励。MiMo-7B 说，基座模型在数学题上会利用判分器的漏洞，而运行测试用例的代码题很难被利用。
5. **每发现一个漏洞，就补一条测试。**`tests/test_tool_env.py` 里每条守卫都有正例和反例。

## 8. 训练时盯什么

| 指标 | 健康的样子 | 危险信号 |
|---|---|---|
| 平均奖励 | 稳步上升，有噪声 | 猛涨：先去看样本，可能是作弊 |
| 格式正确率 / 调用率 | 一起上升 | 格式正确率涨、调用率跌（7.2 节的样子） |
| 回复长度 | 推理任务上缓慢变长 | 暴涨（乱码、重复），或塌到极短（空回复） |
| KL（对参考模型） | 缓慢增长 | 突然跳升：策略跑得太远 |
| 熵（entropy） | 缓慢下降或基本稳定 | 快速塌到接近 0：探索没了，DAPO 称为"熵坍缩" |
| 裁剪比例 | 很小（百分之几） | 很大：学习率太大或更新次数太多 |
| 零方差组比例 | 中等 | 接近 1：题太简单或太难，需要换题或筛题 |
| 留出集上的真实成功率 | 跟着奖励一起涨 | 奖励涨、它不涨：reward hacking |

`zero/post/grpo.py` 每步记录这些值（`log.jsonl`）：奖励、格式正确率、调用率、回复长度、KL、裁剪比例、零方差组比例。熵目前没有记录。DAPO 的论文提醒过一个反直觉的现象：训练集上的奖励和验证集准确率的相关性常常很低。所以一定要看留出集。

## 9. 推理模型：RL 让模型学会"多想一会儿"

DeepSeek-R1-Zero 直接在基座模型上用 GRPO 和规则奖励训练（奖励 = 准确率 + 格式，要求把思考写在 `<think>` 里）。它在 AIME 2024 上的 pass@1 从 15.6% 升到 77.9%，同时平均回复长度随训练持续变长。没有人教它，它自己学会了在回答里验证、反思、换一种方法再试。DeepSeek-R1 再加上冷启动 SFT 和多阶段 RL，解决可读性和语言混杂的问题。

Qwen3 的推理 RL 只用了 3995 个"题目 + 验证器"对，在 170 步内把 Qwen3-235B-A22B 的 AIME'24 从 70.1 提到 85.1。

有两点提醒。第一，"回答变长"不全是能力的体现：Dr. GRPO 指出，序列平均的写法本身会偏向更长的错误回答。第二，Qwen3 报告对比了 Qwen3-8B：RL 提升了 pass@1，但 pass@64 没有变。RL 更多是把模型本来就能采到的正确答案变得更常见，这和第 1 节进位题的例子是同一个道理。

对小模型，Qwen3 的报告还发现：从大模型做在线策略蒸馏比直接 RL 效果更好，而且只用约 1/10 的 GPU 时（见第 17 章）。

主线模型的 GRPO 默认不开思考（工具调用要短而准），`max_new_tokens = 512`。

## 10. 小结

- **模仿有上限**：SFT 和蒸馏会照样继承示范里的错误。强化学习只需要一个能打分的验证器，就能把模型分布里"偶尔对"的答案变得更常见。
- **策略梯度**：∇E[r] = E[r · ∇log π]，奖励不必可导。减去基线不改变期望，却能大幅降低方差。
- **GRPO**：每道题采样 G 个回答，A = (r − 组均值)/组标准差，不需要价值模型。再加上裁剪（μ > 1 时起作用）和可选的 k3 KL，用 token 级聚合。全对或全错的组没有梯度。
- **可验证奖励**：数学比答案，代码跑测试，工具调用比 AST 和执行结果。验证器的三种形态是可执行测试、规则检查、rubric 判分。
- **奖励作弊**：模型会沿最容易涨分的方向走。我们在冒烟测试里亲眼看到它在十步之内学会去掉标签。防守靠严格的格式、惩罚退化输出、留出的判据和人工抽查，以及每个漏洞一条测试。

---

## 从极简代码到生产级代码

极简代码和主线代码讲的是同一件事。`02` 在同一批数据上直接调用生产级函数做对拍：**优势最大差 0.00e+00，损失 −0.046469 vs −0.046469**。（`tests/test_grpo.py` 另有手算的小例子，覆盖裁剪、clip-higher、k3 KL 和两种聚合方式。）

| 极简代码（`code/`） | 生产级代码（`zero/`、`configs/`） | 多做了什么、为什么 |
|---|---|---|
| `group_advantages`（02） | `zero/post/grpo.py::group_advantages` | 支持 `scale_rewards = false`（只减均值，Dr. GRPO 的写法）；G = 1 时不会除以 0 |
| `grpo_loss`（02，对称裁剪、token_mean） | `zero/post/grpo.py::grpo_loss` | 支持 `clip_eps_high`（clip-higher）和两种聚合方式；分块前向时传入全批 token 数，保证结果和整批一次算出的相等；返回 KL、裁剪比例、ρ 均值 |
| 自己写的逐 token 采样（02/03） | `sample_group`：`zero.generate` + KV cache，同一提示词复制 G 份，遇到 `<|im_end|>` 停止 | 真实模型要带对话模板、KV cache 和停止符；没用满预算时，把 `<|im_end|>` 补回来参与训练 |
| 一个 `verify` 函数 | `zero/post/envs/tool_env.py::score_tool_calls` | 6 个模拟 API、schema 校验、AST 和执行两种匹配、一对一匹配、8 条防作弊守卫；`generate_tasks` 与冻结的 dev 集互不重叠 |
| （真实任务） | `zero/post/envs/fc_tasks.py::score_fc` + `[grpo] task_files` | **任意** schema 的工具调用任务，来自开源数据集（Hermes、ToolACE 等），分值和防作弊守卫与上一行相同。调用先与可接受答案匹配，只有对不上的调用才检查 schema。把标准答案当作输出时，BFCL v4 单轮 3,641 题和 ACEBench 1,940 题全部判对（这是对判分器的检查，不用它们训练） |
| 03 的天真奖励 / 修好的奖励 | 守卫 #8（`_bare_call`）+ 空回复 0 分 | 冒烟测试里真实观察到的作弊，对应测试 `tests/test_tool_env.py::test_guard_untagged_call_json` |
| 固定学习率、手写循环 | `run_grpo` + `LoopState` | 配置驱动、断点续训、日志（`log.jsonl`）、梯度裁剪、BF16 autocast（GPU 上，尚未验证）；任务选择只由 (seed, step) 决定，以便续训可复现。用 torchrun **多卡数据并行**：`prompts_per_step` 是全局的，各 rank 采样自己那一份（按题目的全局序号定种子），损失按全局 token 数归一，梯度在各 rank 间求和；2 个 CPU 进程与 1 个进程的损失和权重一致（`tests/test_post_ddp.py`） |
| 打印几个指标 | 每步记录奖励、格式正确率、调用率、长度、KL、裁剪比例、零方差组、耗时 | 对应第 8 节的监控表；熵和留出集成功率还没有接进训练循环 |

配置：`configs/tiny/grpo.toml`（CPU 冒烟测试：G = 8，每步 4 道题，β = 0.02，20 步）；`configs/main/grpo.toml`（主线：G = 16，每步 64 道题即 1024 条回复，`max_new_tokens = 512`，ε = 0.2，`clip_eps_high = 0.28`，β = 0，学习率 1e-6，500 步；尚未在 GPU 上验证）。

**吞吐与 verl（GOAL.zh.md 第 3.3 节）**：`zero` 的 GRPO 是一个可读的实现：可以多卡数据并行，但采样没有连续批处理。第二步如果吞吐不够，就改用 [verl](https://github.com/verl-project/verl)（或小米的 [XiaomiMiMo/verl](https://github.com/XiaomiMiMo/verl)）做实际训练。

但要先在同一个小任务上和 `zero` 对拍：用同一个导出的 HF 模型、同一批 `tool_env` 任务（用固定种子导出成 JSONL），以及同样的 G / ε / β / 学习率 / 聚合方式。用 verl 的自定义奖励接口把 `score_tool_calls` 包一层。先比较第一步的优势和损失（同一批样本应该逐位一致），再比较前 50 步的平均奖励曲线。（对应的 verl 配置项名称见 `zero/post/grpo.py` 的模块说明，是按 verl 的文档整理的，待核实。）

## 主线进度

### 极小配置演示（CPU，`configs/tiny`，约 1.3M 参数）

> 本节是**极小配置演示**：只说明代码能跑通、指标按预期记录，不代表主线模型的任何结果。

> **注意：**本节的数字来自修复工具调用判分器（第 19 章第 6 节）**之前**的那次冒烟测试，是当时的真实输出。修复后我们重跑了一次（`uv run python -m zero.smoke --out out/smoke_final`）。数据、预训练、中期训练、SFT 各阶段的结果完全一致。
>
> 蒸馏中通过验证的样本从 1 条（那句乱码）变为 0 条，下游 DPO、GRPO 和评测的数字随之变化。例如，同一个 SFT 模型的工具调用 call_exact 从 0.133 变为 0.100：模型没变，是判分更严了。GRPO 相对 SFT 仍判为"持平"。你自己运行时，以你的运行结果为准。

冒烟测试（`uv run python -m zero.smoke`，结果在 `out/smoke/`）的 GRPO 阶段从 DPO 之后的权重出发，跑了 10 步。每步 4 道题 × G = 8 = 32 条回复，共用 114 秒：

| 步 | 平均奖励 | 格式正确率 | 调用率 | 平均长度 | KL |
|---:|---:|---:|---:|---:|---:|
| 1 | −0.19 | 0.50 | 0.34 | 25.5 | 0.0000 |
| 3 | −0.40 | 0.47 | 0.47 | 39.7 | 0.0017 |
| 5 | −0.04 | 0.72 | 0.72 | 37.7 | 0.0042 |
| 6 | −0.34 | 0.44 | 0.44 | 31.7 | 0.0074 |
| 9 | −0.02 | 0.72 | 0.72 | 29.7 | 0.0121 |
| 10 | −0.07 | 0.62 | 0.62 | 31.7 | 0.0120 |

十步之间，奖励上下跳动（每步只有 32 条回复）。首尾之差不能当作"学到了东西"的证据。评测阶段在 30 道冻结的 tool_dev 题上做配对 bootstrap：GRPO 相对 SFT 的 call_exact 差为 **−0.033，95% CI [−0.100, +0.000]，判定为"持平"**（sft 0.133，grpo 0.100）。

同一张表里，GRPO 在 30 道玩具选择题上比 SFT 高 0.133（CI [+0.033, +0.267]）。但 GRPO 并没有在选择题上训练，起点 DPO 模型在这组题上已经是 0.200。我们不解读这个差异。

复现命令（换一个输出目录，只跑 3 步，在我们的机器上约 1.5 分钟）：

```bash
uv run python -m zero.post.grpo --config configs/tiny/grpo.toml \
  --set train.init_from=out/smoke/dpo/ckpt --set data.tokenizer=out/smoke/tokenizer.json \
  --set train.max_steps=3 --set train.out_dir=/tmp/grpo_tiny
# step 1 | reward -0.186 | format 0.47 | call 0.28 | len 24.8 | kl 0.0000
# step 2 | reward -0.428 | format 0.31 | call 0.31 | len 33.8 | kl 0.0002
# step 3 | reward -0.610 | format 0.22 | call 0.22 | len 40.2 | kl 0.0016
```

第 8 条守卫就是在这个配置上发现的（7.1 节）。

### 待 GPU 训练后补充

- 主线 GRPO 的训练曲线（奖励、格式正确率、调用率、长度、KL、熵），以及开发集上的工具调用得分。
- 与 verl 的对拍结果（同一小任务上第一步的优势和损失、前 50 步的奖励曲线）。
- 开发集上的对比：clip-higher 开/关，β = 0 / 0.02（用来决定主线配置）。
- 训练中新发现的作弊方式和对应的守卫。（第 6 节列出的两处漏洞已在第一步修复；第二步训练中新发现的，继续补在这里。）
- 花费（记入 `runs/ledger.md`）。

第二步开跑前，曾建议给 `zero` 补两件事。离线的那件已完成（2026-10-10）：`zero/post/difficulty.py` 按开始 RL 的 checkpoint 的 pass@k 筛题。仍待做：在训练循环里记录策略熵；按难度的在线过滤（跳过全对或全错的组并补采样）。

## 前沿观察

- **clip-higher（DAPO）**：把裁剪上界 1+ε_high 设得比下界更宽（DAPO 用 0.2 / 0.28）。这样，低概率的"探索" token 能被推得更高，熵坍缩随之缓解。MiMo-7B 和 OLMo 3 采用了它。按本课的规则，还差一家明确采用的头部家族，所以列为待核实；它在主线配置里是一个可以关闭的开关。
- **不除以标准差（Dr. GRPO）**：组内标准差很小的题（几乎全对或全错）会被放大权重，造成"难度偏差"。Dr. GRPO 建议只减均值，同时去掉按回答长度的归一化。OLMo 3 采用了不除以标准差的做法。目前只找到这一家头部模型明确采用。
- **GSPO**：把重要性比从逐 token 的比值，改为整条回答的（长度归一化的）似然比，并在序列级裁剪。Qwen 团队称它让 MoE 模型的 RL 训练更稳定，并用于 Qwen3 的后续版本。其他家族是否采用，尚未核实。
- **异步 / off-policy RL**：把采样和训练解耦，以提高吞吐（OLMo 3 的 in-flight 更新、GLM-4.5 的 slime、MiMo 的 Seamless Rollout）。这需要额外的重要性修正（例如截断重要性采样）。它属于基础设施层面的做法，本课主线不涉及。
- **长程智能体任务重新用上 critic（GLM-5.2，2026-06）**：据机器之心转述的智谱技术博客，GLM-5.2 的长程 RL 阶段把 GRPO 换成了**带价值网络的 PPO**。长任务经过上下文压缩（compaction）后，子轨迹的数量和长度参差不齐，凑不成一组可比较的样本；改由价值网络给出 token 级的优势。论文 *Learning Without Critics?*（arXiv 2511.03527）在经典 RL 环境里得到同样的结论：在没有提前终止的长任务里，不带 critic 的方法比不过带价值函数的 PPO。目前只有一家头部家族这样做（DeepSeek-V4 训练领域专家时仍用 GRPO），所以放在前沿观察。主线保留 GRPO：工具调用任务短，奖励可验证，组内比较成立。来源见 `references.zh.md` 第 19 章。

## 采用方与来源

| 技术 | 采用方（技术报告 / 模型卡） |
|---|---|
| GRPO（组内基线、无价值模型） | DeepSeek：[DeepSeekMath](https://arxiv.org/abs/2402.03300)（提出）、[DeepSeek-R1](https://arxiv.org/abs/2501.12948)；Qwen：[Qwen3 技术报告](https://arxiv.org/abs/2505.09388) 4.2 节；GLM：[GLM-4.5](https://arxiv.org/abs/2508.06471) 3.2 节（去掉 KL 的 GRPO）；OLMo：[OLMo 3](https://arxiv.org/abs/2512.13961) 4.4 节（OlmoRL 基于 GRPO）；小米：[MiMo-7B](https://arxiv.org/abs/2505.07608) 3.3 节 |
| 组均值基线的 RLVR（非 GRPO 原式） | Kimi：[Kimi K2](https://arxiv.org/abs/2507.20534) 3.2.3 节（K1.5 的目标函数） |
| 可验证奖励（RLVR） | [Tülu 3](https://arxiv.org/abs/2411.15124)（命名者，用 PPO）、DeepSeek-R1、Qwen3、Kimi K2（Verifiable Rewards Gym）、OLMo 3、GLM-4.5、MiMo-7B |
| token 级聚合 | MiMo-7B（式 1）、OLMo 3（4.4.1 节）、GLM-4.5（图 7）；出处是 [DAPO](https://arxiv.org/abs/2503.14476) |
| 去掉 KL 项 | GLM-4.5、MiMo-7B、OLMo 3；DAPO |
| 按难度筛题 / 动态采样 | MiMo-7B、OLMo 3、GLM-4.5、Kimi K2、Qwen3（均见上） |
| k3 KL 估计 | DeepSeekMath 式 4、DeepSeek-R1 式 2；[Schulman, Approximating KL](http://joschu.net/blog/kl-approx.html) |
| 工具调用的规则奖励 | GLM-4.5（3.4 节 Function Calling RL）、Qwen3（General RL 的 Agent Ability）、Kimi K2（工具使用环境）；环境设计参考 [XiaomiMiMo/verl](https://github.com/XiaomiMiMo/verl) |
| 待核实 / 前沿 | clip-higher：DAPO、MiMo-7B、OLMo 3；Dr. GRPO：[论文](https://arxiv.org/abs/2503.20783)、OLMo 3；GSPO：[论文](https://arxiv.org/abs/2507.18071)（Qwen3 的后续版本） |

说明：SmolLM3 的后训练用的是 SFT + 偏好优化，没有 RL（OLMo 3 报告的脚注 28 也这么说）。所以不把它列为 GRPO 的采用方。

## 引导问题

带着这些问题去问 Claude Code，直到你能用自己的话讲清楚：

1. 对数导数技巧里，"∇π = π · ∇log π" 这一步为什么让期望可以用采样估计？如果奖励本身依赖 θ（比如用策略模型自己当裁判），推导在哪里会出问题？
2. GRPO 的优势对所有 token 都一样。一个 30 个 token 的回答里只有一个 token 是错的，它和其他 29 个 token 受到的惩罚一样多。这合理吗？过程奖励模型（PRM）想解决什么问题？DeepSeek-R1 为什么说它没有成功？
3. 为什么 μ = 1 时裁剪不起作用，但梯度仍然不为零？把 `02` 的 μ 改成 1 和 4，看裁剪比例怎么变。
4. k3 估计 exp(d) − d − 1 为什么总是非负？和直接用 log π_θ − log π_ref 当 KL 估计相比，两者各有什么好处？
5. 除以组内标准差时，有两道题：一道是"8 个回答里 1 个对"，另一道是"4 个对"。那个对的回答在两道题上的优势分别是多少？这算"难度偏差"吗？
6. 给"查天气，然后用一句话回答"这样的两轮任务打分，只看最终回答够吗？中间的工具调用要不要单独给分？GLM-4.5 的"逐步规则 RL"和"端到端多轮 RL"是怎么分工的？

## 动手任务

**任务 1（基础）**：在 `01_reinforce_bandit.py` 里把基线换成"留一法"（leave-one-out：每个样本的基线是同一批其他样本的平均奖励），比较方差。它和 GRPO 的组均值只差一个常数因子 G/(G−1)（Dr. GRPO 论文附录 A 有推导）。

**任务 2（核心）**：在 `02_grpo_from_scratch.py` 里，把老师在进位题上的正确率从 30% 改成 5% 和 0%，重新运行。0% 时 GRPO 还能学会进位吗？为什么？（提示：看第 1 步有没有"有对有错"的组。）这说明 RL 需要什么样的起点？

**任务 3（挑战）**：给 `03_reward_hacking.py` 的"修好的奖励"再找一个漏洞。设计一种输出：它在修好的奖励下的平均分比 `<call> 字 d </call>` 瞎猜更高，却不是正确行为。然后补一条守卫，重新运行，确认真实成功率没有下降。把同样的思路用到 `zero/post/envs/tool_env.py` 上：读一遍 `score_tool_calls`，找一种现有守卫没有覆盖的作弊方式，写成 `tests/test_tool_env.py` 风格的测试（先在本地试，不要提交）。

## 想深入：CS336

- 第 16 讲：后训练中的 RLVR（可验证奖励的强化学习、策略梯度、GRPO）。
- 作业 5（Alignment and Reasoning RL）：在数学推理任务上实现 SFT、专家迭代和 GRPO，比较不同的损失归一化与基线。可选的第二部分是 DPO（对应第 18 章）。

课程页（讲义与录像）：<https://cs336.stanford.edu/>

## 本章参考文献

- Williams. *Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning* (REINFORCE), 1992：<https://link.springer.com/article/10.1007/BF00992696>
- Schulman et al. *Proximal Policy Optimization Algorithms*, 2017：<https://arxiv.org/abs/1707.06347>
- Shao et al. *DeepSeekMath*（GRPO、k3 KL、统一的梯度系数视角）, 2024：<https://arxiv.org/abs/2402.03300>
- DeepSeek-AI. *DeepSeek-R1*, 2025：<https://arxiv.org/abs/2501.12948>
- Qwen Team. *Qwen3 Technical Report*, 2025：<https://arxiv.org/abs/2505.09388>；*Group Sequence Policy Optimization*：<https://arxiv.org/abs/2507.18071>
- Kimi Team. *Kimi K2: Open Agentic Intelligence*, 2025：<https://arxiv.org/abs/2507.20534>
- GLM-4.5 Team. *GLM-4.5*, 2025：<https://arxiv.org/abs/2508.06471>
- Xiaomi LLM-Core. *MiMo: Unlocking the Reasoning Potential of Language Model*, 2025：<https://arxiv.org/abs/2505.07608>
- XiaomiMiMo/verl（MiMo-V2.6 的五类 RL 环境与判分器）：<https://github.com/XiaomiMiMo/verl>
- Olmo Team. *Olmo 3*, 2025：<https://arxiv.org/abs/2512.13961>
- Lambert et al. *Tülu 3*（RLVR）, 2024：<https://arxiv.org/abs/2411.15124>
- Yu et al. *DAPO*, 2025：<https://arxiv.org/abs/2503.14476>
- Liu et al. *Understanding R1-Zero-Like Training: A Critical Perspective*（Dr. GRPO）, 2025：<https://arxiv.org/abs/2503.20783>
- Schulman. *Approximating KL Divergence*：<http://joschu.net/blog/kl-approx.html>
- Lilian Weng. *Reward Hacking in Reinforcement Learning*, 2024：<https://lilianweng.github.io/posts/2024-11-28-reward-hacking/>
- verl（HybridFlow）：<https://github.com/verl-project/verl>
- Hands-On Modern Reinforcement Learning（references.md）：<https://walkinglabs.github.io/hands-on-modern-rl/preface/intro>
- CS336：<https://cs336.stanford.edu/>

**下一章**：主线模型的训练到此全部结束。第 20 章按预注册协议做最终评测。它在 BFCL 等工具调用基准上和同尺寸的所有公开模型比较，领先幅度超出置信区间才算"超过"。然后把模型量化成 GGUF，做一个能在笔记本电脑上运行的工具调用助手，写好模型卡，如实发布。
