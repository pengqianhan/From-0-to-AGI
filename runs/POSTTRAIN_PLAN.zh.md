# 后训练计划（第 16–19 章 · 阶段 10）

[English](POSTTRAIN_PLAN.md) · **中文**

> 状态：**计划**，2026-10-10。代码与配置已就绪，并在 CPU 上跑通（`uv run python -m zero.smoke`）；**还没有在 GPU 上跑过**。
> 本文件说明后训练怎么做、按什么顺序做、怎样算做成。具体命令的逐条清单在 [`RUNBOOK.zh.md`](RUNBOOK.zh.md) 第 6 节。

## 0. 已经定下的决定

| 决定 | 内容 | 理由 |
|---|---|---|
| 分词器 | 主线用**自训的** 65,536 词表 | 2026-10-10 项目负责人确认 |
| 流程 | SFT → 序列级蒸馏 → [DPO，可选] → GRPO → **跨阶段在线策略蒸馏（OPD）** | 见第 1 节 |
| OPD 的教师 | **只用自己前几个阶段的 checkpoint** | OPD 逐 token 比较两个分布，需要同一个词表。外部开源教师的词表都和我们不同 |
| 外部教师 | 只做**序列级蒸馏**（教师生成文本 → 执行验证 → 当 SFT 数据） | 和词表无关 |
| RL 算法 | GRPO（可验证奖励） | 工具调用的轨迹短、奖励可验证，组内比较成立 |
| PPO | 不进主线，放进第 19 章"前沿观察" | 长程智能体 RL 改用 critic，目前只有 GLM-5.2 一家（见 `references.zh.md` 第 19 章）。主线任务不是长程任务 |
| 两条线解耦 | 底座出来之前，先在开源底座上把后训练跑通 | 见第 2 节的路线 A、B |

## 1. 后训练流程

| # | 阶段 | 模块 | 起点 | 数据 | 解决什么 |
|---|---|---|---|---|---|
| 1 | SFT | `zero.post.sft` | 底座 | `data/sft/*.jsonl`（第 16 章的来源与许可证） | 底座只会续写 → 学会对话模板、工具调用格式、什么时候停 |
| 2 | 序列级蒸馏 | `zero.post.distill` | SFT | 教师生成 + 执行验证的轨迹，混入原 SFT 数据 | 自己的 SFT 数据不够 → 借强教师的解题轨迹 |
| 3 | DPO（可选） | `zero.post.dpo` | 蒸馏 | 在线偏好对 | 会说但不听话 → 偏好对齐。**是否保留由消融决定**（第 4 节） |
| 4 | GRPO | `zero.post.grpo` | DPO（或蒸馏） | 有验证器的工具调用任务 | 会模仿但不会自己做对 → 用可验证奖励强化 |
| 5 | 跨阶段 OPD | `zero.post.opd`（**新增**） | GRPO | 各教师自己的提示词池 | GRPO 让通用能力退化 → 向前几个阶段的 checkpoint 学回来（GLM-5 §3.5） |

第 5 步的默认教师：

| 教师 | checkpoint | 提示词池 | 权重 | 负责找回的能力 |
|---|---|---|---:|---|
| `distill` | 阶段 2 的结果 | SFT 对话的提示词（`data/sft/train.jsonl`） | 0.5 | 通用对话、指令遵循（GRPO 之前的水平） |
| `grpo` | 阶段 4 的结果 | 工具调用任务 | 0.5 | 工具调用（GRPO 学到的能力，防止 OPD 把它冲掉） |

OPD 的损失默认是 `full_kl`：在学生自己采样的每个回答位置上，算整个词表上的反向 KL(p_S ‖ p_T)。教师就在本地，拿完整分布的成本很低。
`sampled` 是 GLM-5 的写法：优势 A_t = log p_T(y_t) − log p_S(y_t)，只需要采样 token 的对数概率；在期望上和反向 KL 的梯度相同（`tests/test_opd.py` 用穷举验证过）。

## 2. 三条路线

| 路线 | 底座 | 配置 | 什么时候跑 | 目的 |
|---|---|---|---|---|
| **A 代理强底座** | Qwen3-0.6B-Base（和主线同构：稠密、GQA、QK-Norm、共享 embedding） | `configs/proxy/*.toml` | **现在就能跑**（只需要 GPU） | 把流程、代码和超参跑通；和官方 Qwen3-0.6B 做同底座对比 |
| **B 自有弱底座** | 自己预训练的中间 checkpoint（同词表、同架构） | `configs/weak/*.toml` | 预训练有了约 100B token 的 checkpoint 之后 | 验证路线 A 的配方在弱底座上还成立；找出要重调的超参 |
| **C 主线** | 闸门 2 之后的 `out/main/longctx` | `configs/main/*.toml` | 闸门 2 通过之后 | 正式模型，按闸门 3 判定 |

三条路线用**同一套代码、同一份配方**。`proxy` 和 `weak` 的配置都用 `base = "../main/<阶段>.toml"` 继承主线，只改模型形状、分词器和路径（`tests/test_post_configs.py` 检查这一点）。所以在 A、B 上调出来的超参，改主线配置一处就够了。

**能共享的数据**：`data/sft/*.jsonl` 和教师数据 `data/distill/teacher.jsonl` 都是文本，和分词器无关，三条路线共用，只生成一次。打包后的窗口（`packed/`）和在线偏好对跟分词器或策略有关，每条路线各一份。

### 路线 A 的局限

- Qwen3-0.6B-Base 预训练了约 36T token，我们的主线约 400B，差约 90 倍。A 上成立的结论，到弱底座上可能不成立，所以必须跑 B。
- 路线 A 的模型用 Qwen 的词表，A 的 checkpoint **不能**当主线的 OPD 教师。
- A 的结果只算"流程验证"，不能写成主线模型的成绩。

### 路线 B 选哪个 checkpoint

- 选预训练 token 数约 100–200B 的 checkpoint。它的强弱更接近主线最终底座，而不是 Qwen。
- 如果它处在 WSD 的平稳段（学习率还很高），先做一次短的分叉衰减（第 15 章；`configs/main/midtrain.toml` 改 `max_steps` 到约 4,000 步 ≈ 2B token，约 $25），再当底座。不衰减的 checkpoint 会低估底座能力。
- 中间 checkpoint 只见过 4,096 长度的窗口，所以 `configs/weak/` 把 `seq_len` 设为 4,096，RoPE 基频保持 10,000。

## 3. 怎样算做成（写进预注册）

### 路线 A：同底座对比

> 同一个 Qwen3-0.6B-Base 起步：**我们的后训练** vs **官方后训练的 Qwen3-0.6B**

- 主终点：预注册里的 E1（BFCL）和 E2（ACEBench 中文）。官方 Qwen3-0.6B 的思考 / 非思考两种模式都测，逐个基准取较高分。
- 判定：`zero.eval.bootstrap` 的配对 bootstrap，按预注册的"超过 / 持平 / 落后"。
- **通过**：E1、E2 都是"持平"或"超过"。
- 每个阶段都要有贡献：后一阶段在 E1 上不能显著落后于前一阶段。落后就先查原因，不往下走。
- 通用组（MMLU-Redux、C-Eval、GSM8K、IFEval 等）如实报告。OPD 之后不能比 SFT 之后显著退化。
- 这个判据要在路线 A 开跑**之前**写进 `eval/PREREGISTRATION.md`。

### 路线 B：配方能迁移

- SFT、蒸馏、GRPO、OPD 每一步在 E1 上都有非负、稳定的提升（配对 bootstrap 不落后于上一阶段）。
- GRPO 有学习信号：`zero_std_groups` 前 50 步的均值低于 0.8。高于 0.8，说明任务对弱底座太难或太简单，要先换任务难度（见第 6 节）。
- 记下和路线 A 相比必须改的超参，写进 `runs/<日期>-weak-*/README.md`。

### 路线 C：闸门 3

按 `RUNBOOK.zh.md` 第 6 节的闸门 3 清单，和全部对手比较。

## 4. 消融（先在 A 上做，结论在 B 上复核一次）

| 消融 | 做法 | 判定 |
|---|---|---|
| DPO 要不要 | GRPO 分别从 `distill` 和 `dpo` 起步，后续步骤相同 | E1/E2 不落后且通用组不差，就**去掉 DPO**（省一个阶段） |
| OPD 要不要 | 比较 `grpo` 和 `opd` 两个 checkpoint | 通用组有提升、E1 不落后，就保留 |
| OPD 的损失 | `--set opd.loss=sampled` vs 默认 `full_kl` | 同样步数下 E1 + 通用组更好的那个 |
| OPD 的教师权重 | `distill` 0.5/0.5 vs 0.7/0.3 | 同上 |
| clip-higher | GRPO `clip_eps_high` 0.28 vs 0 | E1 和 `resp_len` 的稳定性 |

预算紧时，消融用正式步数的一半。

## 5. 超参起点（配置里的默认值，按 A 的结果调）

| 阶段 | 学习率 | 其他 | 先扫什么 |
|---|---|---|---|
| SFT | 5e-5，cosine，warmup 100 | 2,000 步 × 524k token，seq 8,192 | 学习率 {2e-5, 5e-5, 1e-4}，用 25% 的步数比 `val_loss` |
| 蒸馏 | 3e-5 | 1,500 步；混入 SFT 数据防遗忘 | — |
| DPO | 5e-7 | β = 0.1，64 对 / 步 | — |
| GRPO | 1e-6，常数 | G = 16，64 题 / 步，ε 0.2 / 0.28，无 KL，token 均值 | 学习率 {5e-7, 1e-6, 2e-6}，各 100 步 |
| OPD | 1e-6，常数 | 256 题 / 步，每题 1 个样本，200 步 | — |

## 6. 前置工作与已知缺口

**开跑前必须完成（P0）：**

1. **SFT 数据**：流水线代码已完成（2026-10-10，`zero/post/sft_data.py`，见第 6.2 节）。还要做的是逐个核实来源的许可证、下载数据、导出评测题目用于去污染，然后按 token 占比调配比。
2. **教师**：代码已经能用真实数据（第 6.4 节）；**由你**从第 6.4 节核实过的候选里选定教师，在 `configs/main/distill.toml` 填好名称、版本、许可证。
3. **真实的 RL 任务集**：代码已完成（2026-10-10，`zero/post/envs/fc_tasks.py`，见第 6.1 节）。还要做的是下载数据、构建任务文件，并在本地抽查判分结果。
4. **路线 A 的预注册判据**：已作为候选条款写进 `eval/PREREGISTRATION.md` 第 6 节（草案，冻结时由你确认）。评测流水线见第 6.3 节。
5. **开机检查**：租到 GPU 的第一个小时运行（`zero.tools.launch_check`，第 6.4 节），实测每个阶段每步的耗时，用来定第 7 节的预算。

### 6.1 真实工具调用任务（`zero/post/envs/fc_tasks.py`）

**任务格式**：一行一个 JSON。`tools`（OpenAI 函数格式）、`messages`（提示词，可以含历史）、`gold_calls`（下一轮助手应该发出的调用，顺序无关；空列表表示"不应调用"）。每个参数可以有 `alternatives`（所有可接受的值，`""` 表示可以省略），和 BFCL 的 possible answer 格式一致。只判第一轮助手输出，和 `tool_env` 的 GRPO 一样。

**判分**：和 `tool_env.score_tool_calls` 同一套分值和同样的防作弊规则（格式错 −1；伪造工具结果、标签外的调用 JSON、过长、调用过多都算格式错；该调不调 0；不该调却调 −0.5；正确不调 0.5；否则 0.1 + 0.9 × 命中比例 − 0.25 × 多余调用 − 0.5 × 违反 schema 的调用）。新增的是**通用 schema 检查**（未提供的工具、缺必填参数、schema 外的参数、类型错、不在 enum 里）和**通用参数比较**（字符串去空白、忽略大小写；数字按值；省略的参数如果金标值等于默认值，算同一个调用）。

**候选数据源**（许可证在 2026-10-10 从数据卡核实；用之前再核一次）：

| 来源 | 许可证 | 语言 | 转换器 | 备注 |
|---|---|---|---|---|
| NousResearch/hermes-function-calling-v1 | Apache-2.0 | 英 | `hermes` | ShareGPT + `<tool_call>`，单轮和多轮都有 |
| Team-ACE/ToolACE | Apache-2.0 | 英、中 | `toolace` | Python 调用语法 `[Func(a=1)]`；有"缺参数时不该调用"的轮次；**中文工具调用数据的主要来源** |
| glaiveai/glaive-function-calling-v2 | Apache-2.0 | 英 | 先转成 `openai` 格式 | Hermes 里已含清洗过的 5k 子集 |
| Salesforce/xlam-function-calling-60k | 待核实（需申请访问） | 英 | `xlam` | 访问受限，这里读不到数据卡 |

**构建步骤**：

```bash
# 1. 转换 + 校验 + 去重 + 对 BFCL 去污染（工具名重合或 13-gram 重合就丢弃），留 500 条做 dev
uv run python -m zero.post.envs.fc_tasks build \
    --src hermes:data/raw/hermes-function-calling-v1/func-calling-singleturn.json \
    --src hermes:data/raw/hermes-function-calling-v1/func-calling.json \
    --src toolace:data/raw/ToolACE/data.json \
    --exclude-bfcl data/eval/bfcl --exclude-tasks data/eval/acebench_zh.jsonl \
    --out data/rl/fc_all.jsonl --dev-out data/rl/fc_dev.jsonl --dev-size 500
# 2. 分成 SFT 对话和 RL 任务，两部分不重叠（SFT 见过答案的题，RL 时组内全对，没有信号）
uv run python -m zero.post.envs.fc_tasks split data/rl/fc_all.jsonl \
    --sft-out data/sft/fc_sft.jsonl --rl-out data/rl/fc_train.jsonl --sft-frac 0.5
# 3. 抽查：挑几条任务，手写正确 / 错误的输出，看分数是否合理
uv run python -m zero.post.envs.fc_tasks score data/rl/fc_dev.jsonl 0 '<tool_call>{...}</tool_call>'
```

`<out>.meta.json` 记录每个来源的条数、许可证、各种丢弃原因的计数、"不应调用"和并行调用的条数。

**已接入的地方**：`configs/main/grpo.toml` 的 `task_files`，三条路线 OPD 的 `grpo` 教师提示词池，三条路线评测配置里的 `fc_tasks`（dev 集，配对 bootstrap）。tiny 配置仍用 `tool_env`（冒烟测试）。

**还没做的**：
- ACEBench 到本格式的转换器（目前只能用 `--exclude-tasks` 读已经转好的文件）。
- 多步任务（需要可执行的工具）。
- DPO 的在线偏好对和蒸馏的教师任务目前仍来自 `tool_env`。
- 去污染只查了用户文本的 13-gram 和工具名；函数 schema 的近似重复还没查。

### 6.2 SFT 数据流水线（`zero/post/sft_data.py`）

一份配置（`configs/main/sft_data.toml`）列出所有来源，一条命令产出 `data/sft/train.jsonl`、`val.jsonl` 和 `meta.json`：

```bash
uv run python -m zero.post.sft_data --config configs/main/sft_data.toml
```

**支持的格式**：`messages`（Tülu 3、SmolTalk、SmolTalk2，含 SmolTalk2 的 `xml_tools` 和 `custom_instructions`）、`sharegpt`、`alpaca`、`sft`（本项目格式，例如 `fc_tasks split` 的输出）、`tool_env`（自己合成的轨迹）。助手文本里的 `<tool_call>` 会转成结构化的 `tool_calls`，解析失败的整条丢弃。

**许可证按行判定**：Tülu 3、SmolTalk2 每行都带上游子集名（`source` 字段）。`license_by_source` 给每个子集单独指定许可证，`drop_sources` 直接排除。NC 许可证一律丢弃；"待核实"默认丢弃，整个来源都被丢掉时会打出警告。

**清洗与过滤**（每一项都在 `meta.json` 里按来源计数）：特殊 token 注入、空回答、工具调用解析失败、`<think>` 块（主线默认不思考，去掉思考部分，只剩思考的整条丢弃）、语言过滤（`langs`）、超长（按主线分词器计 token）、精确去重、近似去重（第一条用户消息的 MinHash）、13-gram 去污染（评测题目）和 BFCL 工具名去污染。

**配比**：每个来源先过滤、打乱、截到 `max_rows`；然后从总池里切出验证集（在重复之前切，保证验证集不会出现在训练集里）；最后按 `repeat` 上采样。`meta.json` 按来源和语言报告条数、总 token 和助手 token 的占比。**配比按 token 看，不按条数看。**起步目标：工具调用约 25%，中文约 30%，其余是英文通用数据。

**当前配置里的来源**（许可证 2026-10-10 读自数据卡）：

| 来源 | 许可证 | 状态 |
|---|---|---|
| `fc-tool-calls`（Hermes + ToolACE，`fc_tasks split` 的输出） | Apache-2.0（每行自带） | 可用 |
| `tool-env`（自己合成） | 本项目 | 可用 |
| SmolTalk2 的 6 个英文 no_think 子集 | 数据卡无许可证元数据，部分由 Qwen 生成 | **待核实** |
| SmolTalk2 多语言子集的中文部分 | 同上 | **待核实** |
| smoltalk-chinese（OpenCSG） | 元数据写 Apache-2.0，但正文说商用需要邮件取得许可 | **待核实**：商用要先发邮件申请 |
| COIG-CQIA | 数据卡没写许可证，内容来自知乎、豆瓣等 | 不用 |
| Infinity-Instruct（BAAI） | 访问受限，读不到数据卡 | 待你查看 |

**中文通用指令数据是这套配比现在最大的缺口。**许可证干净的中文数据很少。三个办法：
1. 给 OpenCSG 发邮件，申请 smoltalk-chinese 的商用许可。
2. 核实 SmolTalk2 多语言子集的上游条款（Qwen 生成的部分）。
3. 自己生成：用许可证允许的开源教师（Apache-2.0 / MIT）回答中文提示词，走第 17 章的序列级蒸馏流程。这条路最干净，但要花教师推理的钱。

### 6.3 评测流水线（2026-10-10）

在 CPU 上能做的部分都已做完并用真实数据核对过；剩下的只有需要 GPU 和 vLLM 的生成步骤。

**BFCL（E1）**：对照 `bfcl-eval` 2026.3.23 的源码核对了 `zero/eval/bfcl.py` 原来的三个"待核实"项。
- 模型注册表的名字和 `ModelConfig` 字段：与源码一致；运行时注册能被命令行看到。
- `function` 的结构：BFCL 给的是原始函数描述，类型名是 Python 写法（`dict`、`float`）。我们的 handler 把它们换成 JSON schema 写法，与训练数据一致（已写进预注册第 4 节）。
- 逐题结果：BFCL 的 score 文件**只记失败的题**。逐题对错 = result 文件里的全部 id − score 文件里的失败 id。新增 `per_item_results`、`collect`、`compare`（分层配对 bootstrap）。原来按 CSV 松散解析的写法删掉了。

**用真实评测数据核对判分器**：把每道题的标准答案当成模型输出去打分，应该全部判对。
- BFCL v4 单轮 3,641 题：第一版判分器有 264 题不一致，发现四类问题并全部修正——BFCL 的函数描述和它自己的答案矛盾（例如 `year` 标成整数，答案却是 `"dontcare"`）、可选参数的 `null`、嵌套参数的候选答案、超过 5 个的并行调用。修正后 0 不一致。
- ACEBench 中文 967 题、英文 973 题（Normal + Special，不含 Agent）：修正输出长度上限后 0 不一致。ACEBench 自身有 1 题工具名重复（`normal_atom_number_48`）。
- 这些例子都加成了回归测试。

**导出**：
- `fc_tasks export bfcl|acebench`：把评测集导出成任务文件，**只用于去污染**和核对判分器，不用来挑 checkpoint（预注册第 8 节）。工具调用的开发集是从训练来源留出的 `data/rl/fc_dev.jsonl`。
- `zero.eval.export_prompts`：导出通用组基准（MMLU-Redux、MMLU-Pro、C-Eval、CMMLU、GSM8K、MATH-500、HumanEval+、MBPP+、IFEval）的题目，供 SFT 的 13-gram 去污染。需要 `datasets` 和 Hugging Face 访问，这里只用假数据测过；数据集 id 和 split 冻结预注册时再核对。

**还要在 GPU 上做的**：用 vLLM 跑通 BFCL 的 generate / evaluate；确认导出 tokenizer 的 `apply_chat_template` 在 BFCL 里给出的文本和训练时一致；ACEBench 官方评测脚本对新版 vLLM 的适配。

### 6.4 教师数据、多卡、难度筛选、开机检查（2026-10-10）

**真实数据上的教师数据**（`zero/post/distill.py`）：除了玩具环境，还支持 `task_files`（真实工具调用任务：只留教师判分完全正确的第一轮）和 `prompt_files`（没有答案的提示词，例如中文指令：丢掉空回答、伪造轮次、工具调用和语言不符的回答；每行写入 `prompt_license`）。`concurrency` 向教师服务并发请求。**这是补第 6.2 节中文缺口最干净的路子**：用许可证允许的教师回答中文提示词。提示词本身也要有许可证。

**教师候选**（许可证于 2026-10-10 读自模型元数据；**由你来选**）：

| 模型 | 许可证 | 备注 |
|---|---|---|
| Qwen3.5-35B-A3B | Apache-2.0 | 激活 3B，单卡（FP8）就能便宜地起服务 |
| Qwen3.5-122B-A10B | Apache-2.0 | 更强；推理服务商有托管 |
| Qwen3-235B-A22B-Instruct-2507 | Apache-2.0 | 非思考的指令模型 |
| gpt-oss-120b | Apache-2.0 | OpenAI 另有使用政策，用之前要读 |
| DeepSeek-V4.1-Flash | MIT | 总参数 763B，实际上只能用托管服务 |
| GLM-5.2 | MIT | 中英文；工具调用强（MCP-Atlas、Tool-Decathlon） |

用推理服务商托管的教师时，还要读服务商的服务条款。

**多卡**（`zero/post/common.py`）：DPO、GRPO、OPD 都能用 torchrun 多卡数据并行；配置里的批大小是全局的，配方不随卡数改变。2 个 CPU 进程与 1 个进程的逐步损失和最终权重一致（`tests/test_post_ddp.py`）。蒸馏在 rank 0 生成数据后，用 SFT 的 `Trainer`（DDP）训练。**顺手修了一个 DPO 的 bug**：同一步的每个微步都用同样的 `micro_batch_size` 对数据，每步的大部分偏好对从来没有被训练到。

**难度筛选**（`zero/post/difficulty.py`）：用开始 RL 的 checkpoint 对每个任务采样 k 次，保留 1/k ≤ pass ≤ (k−1)/k 的任务（弱底座可以另外保留一部分全错的题）。难度取决于策略，所以每条路线各一份：`data/rl/<路线>/fc_train_filtered.jsonl`。

**开机检查**（`zero/tools/launch_check.py`）：租到 GPU 后的第一个小时。把一条路线的每个阶段各跑几步（在一个启动目录里串起来），给出每步秒数的中位数、显存峰值，以及每个阶段全量运行的预计时长、卡时和花费；教师吞吐用一小份样本实测。开跑前先检查输入是否齐全。它取代了后训练里的"阶段 6 第 9 项"。

**已知风险：**

| 风险 | 表现 | 应对 |
|---|---|---|
| 弱底座上 RL 没有奖励信号 | `zero_std_groups` 接近 1，`reward_mean` 不动 | `zero/post/difficulty.py`：按开始 RL 的 checkpoint 的 pass@k 筛掉全对和全错的题（Kimi K2、OLMo 3 的做法；在线筛选未实现）；加强蒸馏数据 |
| GRPO 奖励作弊 | `call_rate` 下降但 `reward_mean` 上升 | `tool_env.py` 列出的 8 条防作弊规则；真实任务集的判分器也要有同样的测试 |
| OPD 把 GRPO 学到的东西冲掉 | E1 下降，`kl/grpo` 上升 | 调高 `grpo` 教师的权重；减少步数 |
| 采样太慢 | 用上所有卡后 GRPO / OPD 每步仍然太慢（已多卡数据并行，但没有连续批处理） | 先按 `grpo.py` 模块说明和 verl 对拍，再用 verl |
| 学生输出变啰嗦 | `eos_rate` 下降、`resp_len` 上升 | 降学习率；检查 `max_new_tokens` 截断比例 |

## 7. 预算（总额 $1,500，GOAL.md 3.4）

GRPO 和 OPD 的数字是**上限**。用开机检查实测后重算。单次运行超过 $100 先报批，花费记入 `runs/ledger.zh.md`。

| 项目 | 上限 | 说明 |
|---|---:|---|
| 教师数据生成（三条路线共用） | $150 | vLLM 实测吞吐后再定样本数 |
| 路线 A | $550 | SFT ~$15、蒸馏 ~$13、DPO ~$20、GRPO ≤ $150、OPD ≤ $80、消融（半步数）≤ $200、BFCL/ACEBench 评测 ~$30 |
| 路线 B | $280 | 分叉衰减 ~$25、SFT ~$8、蒸馏 ~$7、GRPO ≤ $150、OPD ≤ $80、评测 ~$10 |
| 路线 C | $450 | 一遍完整流程约 $300 + 一次重跑的余量 |
| 机动 | $70 | |

## 8. 路线 A 的执行步骤

```bash
# 0. 下载底座和官方后训练模型（Apache-2.0）
huggingface-cli download Qwen/Qwen3-0.6B-Base --local-dir data/hf/Qwen3-0.6B-Base
huggingface-cli download Qwen/Qwen3-0.6B --local-dir data/hf/Qwen3-0.6B
# 1. 导入成 zero checkpoint，并检查配置里的模型形状
uv run python -m zero.tools.import_hf data/hf/Qwen3-0.6B-Base out/proxy/base --check-config configs/proxy/sft.toml
# 2. 开机检查：每个阶段跑几步 → 每步秒数、显存、预计花费（runs/<日期>-proxy-launch/）
uv run python -m zero.tools.launch_check --track proxy --nproc 8 --price 2.5
# 3. 全量运行（都是多卡数据并行；配置里的批大小是全局的）
uv run torchrun --standalone --nproc_per_node=8 -m zero.post.sft     --config configs/proxy/sft.toml
uv run torchrun --standalone --nproc_per_node=8 -m zero.post.distill --config configs/proxy/distill.toml
uv run torchrun --standalone --nproc_per_node=8 -m zero.post.dpo     --config configs/proxy/dpo.toml
uv run torchrun --standalone --nproc_per_node=8 -m zero.post.difficulty --policy out/proxy/dpo/ckpt \
    --tasks data/rl/fc_train.jsonl --k 8 --out data/rl/proxy/fc_train_filtered.jsonl
uv run torchrun --standalone --nproc_per_node=8 -m zero.post.grpo    --config configs/proxy/grpo.toml
uv run torchrun --standalone --nproc_per_node=8 -m zero.post.opd     --config configs/proxy/opd.toml
# 4. 内部评测（含官方 Qwen3-0.6B）；正式判定用 BFCL / ACEBench（第 3 节）
uv run python -m zero.eval.harness --config configs/proxy/eval.toml
```

路线 B 的命令相同，把 `configs/proxy/` 换成 `configs/weak/`，第 1 步换成把选好的 checkpoint 链接到 `out/weak/base`（见 `configs/weak/sft.toml` 开头的注释）。路线 C 见 `RUNBOOK.zh.md` 第 6 节。

每次运行建一个 `runs/<日期>-<路线>-<阶段>/` 目录，放配置副本、日志摘要（`log.jsonl` 的关键行）和结论。

## 9. 本次新增的代码

| 文件 | 内容 |
|---|---|
| `zero/post/opd.py` | 跨阶段在线策略蒸馏：多个同词表教师、各自的提示词池与权重、`full_kl` / `sampled` 两种损失、断点续训、按教师记录 KL |
| `zero/tools/import_hf.py` | 把 HF Qwen3 稠密模型导入成 zero checkpoint（含分词器），并检查配置的模型形状 |
| `configs/{tiny,main}/opd.toml` | OPD 配置；`configs/main/eval.toml` 加入 `opd` |
| `configs/proxy/*.toml` | 路线 A：Qwen3-0.6B-Base 上的全套后训练配置 |
| `configs/weak/*.toml` | 路线 B：自有中间 checkpoint 上的全套后训练配置 |
| `tests/test_opd.py`、`tests/test_post_configs.py` | 损失的手算 / 穷举校验、端到端、分词器不一致时报错、HF 导入往返一致、三条路线的配置检查 |
| `zero/smoke.py` | 冒烟流程加入 OPD 阶段，导出改为从 OPD checkpoint |
| `zero/post/common.py`（数据并行）、`grpo.py` / `opd.py` / `dpo.py` / `distill.py`（第 6.4 节） | DPO、GRPO、OPD 和蒸馏的多卡运行；DPO 批次修复；真实任务和提示词上的教师数据 |
| `zero/post/difficulty.py`、`zero/tools/launch_check.py`（第 6.4 节） | RL 任务的离线难度筛选；带花费推算的开机检查 |
| `zero/eval/bfcl.py`、`zero/eval/export_prompts.py`、`fc_tasks` 的 ACEBench 转换与 `export`（第 6.3 节） | 评测流水线：BFCL 适配器按源码修正、逐题结果与配对 bootstrap、判分器在 BFCL 和 ACEBench 全量数据上核对、评测题导出（去污染） |
| `zero/post/sft_data.py`、`configs/main/sft_data.toml`（第 6.2 节） | SFT 数据流水线：5 种来源格式、按行判定许可证、清洗、去重、去污染、按 token 统计的配比 |
| `zero/post/envs/fc_tasks.py`（第 6.1 节） | 真实工具调用任务：格式、通用 schema 检查与判分、Hermes / ToolACE / xLAM / OpenAI / BFCL 转换器、去污染、构建与 SFT/RL 切分；GRPO（`task_files`）和评测（`fc_tasks`）已接入 |
