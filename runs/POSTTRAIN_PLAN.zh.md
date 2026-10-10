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

1. **SFT 数据**：`data/sft/train.jsonl`、`val.jsonl`，按第 16 章的来源和许可证构建；对 BFCL / ACEBench 的函数名和 schema 去污染。
2. **教师**：确定序列级蒸馏的教师（Apache-2.0 / MIT），在 `configs/main/distill.toml` 填好名称、版本、许可证。
3. **真实的 RL 任务集**：**这是现在最大的缺口。**`grpo.py` 和 `opd.py` 的工具调用任务目前只来自 `tool_env`，它只有 6 个模拟工具，是玩具环境。正式训练需要带验证器的真实工具调用任务（任意 schema 的 AST 匹配判分）。要先补"从 JSONL 读任务 + 通用判分"的代码，再跑路线 A 的 GRPO。
4. **路线 A 的预注册判据**写进 `eval/PREREGISTRATION.md`（第 3 节）。
5. **阶段 6 第 9 项**：在 GPU 上实测 GRPO 和 OPD 每步的耗时，用来定第 7 节的预算。

**已知风险：**

| 风险 | 表现 | 应对 |
|---|---|---|
| 弱底座上 RL 没有奖励信号 | `zero_std_groups` 接近 1，`reward_mean` 不动 | 按 SFT 模型的 pass@k 筛掉全对和全错的题（Kimi K2、OLMo 3 的做法；在线筛选还没实现，先离线筛）；加强蒸馏数据 |
| GRPO 奖励作弊 | `call_rate` 下降但 `reward_mean` 上升 | `tool_env.py` 列出的 8 条防作弊规则；真实任务集的判分器也要有同样的测试 |
| OPD 把 GRPO 学到的东西冲掉 | E1 下降，`kl/grpo` 上升 | 调高 `grpo` 教师的权重；减少步数 |
| 单进程实现的吞吐不够 | GRPO / OPD 每步太慢 | 先按 `grpo.py` 模块说明和 verl 对拍，再用 verl |
| 学生输出变啰嗦 | `eos_rate` 下降、`resp_len` 上升 | 降学习率；检查 `max_new_tokens` 截断比例 |

## 7. 预算（总额 $1,500，GOAL.md 3.4）

GRPO 和 OPD 的数字是**上限**。阶段 6 第 9 项实测后重算。单次运行超过 $100 先报批，花费记入 `runs/ledger.zh.md`。

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
# 2. 每个阶段先用 --set train.max_steps=20 跑通，再跑全量（RUNBOOK 的"先小后大"）
uv run torchrun --standalone --nproc_per_node=8 -m zero.post.sft --config configs/proxy/sft.toml
uv run python -m zero.post.distill --config configs/proxy/distill.toml
uv run python -m zero.post.dpo     --config configs/proxy/dpo.toml
uv run python -m zero.post.grpo    --config configs/proxy/grpo.toml
uv run python -m zero.post.opd     --config configs/proxy/opd.toml
# 3. 内部评测（含官方 Qwen3-0.6B）；正式判定用 BFCL / ACEBench（第 3 节）
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
