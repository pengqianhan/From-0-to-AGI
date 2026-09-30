# 预注册（PREREGISTRATION）——**草案 / 未冻结**

> **状态：草案（DRAFT），未冻结，不具备登记效力。**
> 按 GOAL.md 3.2：第二步开始、主线预训练之前，确定对手清单和冻结日期，经项目负责人确认后定稿提交，
> **commit 时间即登记时间**。定稿之后任何修改只能在文末"修订记录"里以"日期 + 理由"追加，不能静默改写。
> 本草案里所有"TBD / 待定"项都必须在冻结时填上具体值；标"候选"的是本草案的提议，需要负责人确认。
> 草案里的基准事实（题量、类别、判分方式、版本）由第 11 章（`chapters/11-evaluation/README.md`）核实，截至 2026-09。

GOAL.md 3.2 要求本文件写明五件事，分别在第 2 节（基准及版本）、第 3 节（框架及版本）、第 4 节（提示词、模板、解码参数）、
第 5 节（对手清单与冻结日期）、第 6 节（"超过"的判定标准）。

## 1. 要检验的主张

- **硬目标（专项）**：在第 2.1 节的两个**主终点**上，我们的最终模型**超过**对手清单（`eval/opponents.md`）里的每一个模型，包括 Qwen3.5-0.8B。
- **软目标（通用）**：在第 2.2 节的通用基准上如实报告；与"完全开放配方"模型比较名次；与只开放权重的模型的差距如实写出。
- 我们的模型默认只有非思考模式。

## 2. 基准与版本

### 2.1 专项组（硬目标）

| 主终点 | 基准 | 版本 / 数据快照 | 纳入的类别 | 指标 | 状态 |
|---|---|---|---|---|---|
| **E1** | BFCL | 冻结时的最新版（当前 V4；PyPI `bfcl-eval` 最新 2026.3.23） | 候选：非 Agentic 四段——Non-Live（`simple_python`、`simple_java`、`simple_javascript`、`multiple`、`parallel`、`parallel_multiple`）、Live（`live_simple`、`live_multiple`、`live_parallel`、`live_parallel_multiple`）、相关性（`irrelevance`、`live_irrelevance`、`live_relevance`）、Multi-Turn（`multi_turn_base`、`multi_turn_miss_func`、`multi_turn_miss_param`、`multi_turn_long_context`） | 各段准确率按官方 V4 权重 10 / 10 / 10 / 30 重新归一化后的加权准确率（段内计分规则与官方一致） | 候选，待确认 |
| **E2** | ACEBench（中文） | 仓库 commit TBD（论文 arXiv:2501.12851 v8；仓库 2025-10-29 仍在修正答案；MIT 许可） | `data_zh` 的 Normal（atom、single-turn、multi-turn、similar API、preference）+ Special（incomplete、error、irrelevant） | 按官方口径的准确率（Normal 为 AST 匹配，Special 为规则判定）；两类合并方式 TBD（候选：官方的"样本数平方根"加权） | 候选，待确认 |

只报告、不设为硬目标：

| 基准 | 内容 | 理由 |
|---|---|---|
| BFCL V4 官方总分（含 Agentic 40%） | `web_search_*`、`memory_*` | web search 依赖 SerpAPI 与实时网页，不可复现；memory 类别能否离线复现待核实。与 E1 并列报告 |
| BFCL `format_sensitivity` | 不计分类别 | 描述性报告 |
| ACEBench Agent（中、英） | multi-turn、multi-step | 需要 GPT-4o 扮演用户：闭源、要花钱、不可复现 |
| ACEBench 英文 Normal + Special | — | 描述性报告 |
| τ²-bench | airline、retail、telecom（50 / 115 / 114 题） | 用户由大模型模拟，模拟用户自身出错率 16%–47%（论文表 2）；指标 pass^k，k 与模拟用户模型 TBD |

### 2.2 通用组（如实报告）

| 基准 | 数据 / 子集 | 题量 | 判分 | 状态 |
|---|---|---:|---|---|
| MMLU-Redux | `edinburgh-dawg/mmlu-redux-2.0`（修订号 TBD）；是否剔除 `error_type ≠ ok` 的题：TBD（候选：全部 5,700 题为主，剔除版描述性报告） | 5,700 | 准确率 | 候选 |
| MMLU-Pro | 官方 test | 12,032（10 选 1） | 5-shot CoT 后按官方正则抽取答案 | 候选 |
| C-Eval | **val**（test 答案不公开，自己重跑对手只能用 val） | 1,346 | 准确率 | 候选 |
| CMMLU | 官方 test | 11,528 或 11,582（论文正文与附录不一致，按数据文件核实） | 准确率 | 候选 |
| GSM8K | test | 1,319 | 抽取最终数字，精确匹配 | 候选 |
| MATH-500 | PRM800K 的 500 题子集 | 500 | 答案等价判定；判定器 TBD | 候选 |
| HumanEval+ / MBPP+ | EvalPlus | 164 / 378 | pass@1（贪心） | 候选 |
| IFEval | `google/IFEval` | 541 | 主报 prompt 级 strict，另报其余三个分数 | 候选 |
| Base 模型（闸门 2 用） | HellaSwag、ARC-Easy/Challenge、PIQA、WinoGrande；MMLU / C-Eval / CMMLU 的 5-shot 对数似然版 | — | acc 与 acc_norm（主报哪一个 TBD） | 候选 |

## 3. 评测框架与版本（冻结时填写）

| 框架 | 用途 | 版本 | 备注 |
|---|---|---|---|
| lm-evaluation-harness | 通用组、Base 基准 | TBD（当前 PyPI 最新 0.4.13，2026-08-31） | `--log_samples` 保存逐题结果；已核实 `mmlu_redux`、`mmlu_pro`、`ceval`、`cmmlu`、`gsm8k`、`ifeval`、`hellaswag`、`arc`、`piqa`、`winogrande` 有现成任务；MATH-500 用哪个任务或脚本待核实 |
| bfcl-eval | BFCL | TBD（当前最新 2026.3.23） | 我们的模型用 `zero/eval/bfcl.py` 的 `ZeroFCHandler`（导出目录里的 chat_template）；**尚未验证** |
| ACEBench | E2 | 仓库 commit TBD | 官方 `requirements.txt` 钉了 `vllm==0.6.1.post1`，新模型大概率需要适配层（待核实） |
| EvalPlus | HumanEval+ / MBPP+ | TBD（当前最新 0.3.1） | |
| vLLM | 推理后端 | TBD | 所有模型同一版本 |
| 本仓库 | 配对 bootstrap、报告 | 冻结时的 commit | `zero/eval/bootstrap.py`、`zero/eval/report.py`；E1 需要的分层 / 加权 bootstrap 已实现为 `stratified_paired_bootstrap`，多对手 × 多终点的总判定为 `overall_verdict`（见第 6 节；测试在 `tests/test_eval.py`） |

## 4. 提示词、模板与解码参数

- **模板**：每个模型用各自**官方**的对话 / 工具调用模板（HF `tokenizer_config.json` 里的 `chat_template`，或 BFCL 内置 handler）。
  我们的模型用导出目录里的 `chat_template`（`zero/post/chat.py` 的 `CHAT_TEMPLATE`，与训练时逐字一致，`tests/test_chat.py` 保证）。
  BFCL 对每个模型用 FC 模式还是 Prompt 模式：TBD（候选：有官方 FC handler 的用 FC，否则用 Prompt；两者都有时取较高）。
- **系统提示词**：各基准官方默认；没有官方默认的一律不加。
- **少样本数与示例**：按各基准在所用框架里的默认设置，冻结时逐项写明。
- **思考模式**：对手有思考 / 非思考两种模式时**两种都测**，**逐个基准**取较高分的模式来比（GOAL.md 3.2 第 4 条）。
  理由：哪种模式更强因基准而异（Qwen3.5-0.8B 模型卡：MMLU-Redux 思考 59.5 / 非思考 48.5；IFEval 思考 44.0 / 非思考 52.1）。
  思考模式的最大输出长度：TBD；达到上限仍未给出答案的按错误计（候选），并单独统计比例。
- **解码参数**（全部模型相同，GOAL.md 3.2 第 4 条）：候选为贪心（temperature = 0；BFCL 默认 0.001）；max_new_tokens TBD。
  需要采样的指标（如 pass@k、τ²-bench 的 pass^k）固定 temperature / top_p / 种子 / 样本数。
- **待负责人决定**：Qwen3.5 官方评测用采样（语言类基准 temperature 1.0、top_p 0.95、top_k 20、presence_penalty 1.5），
  模型卡还提示 0.8B 在思考模式下容易陷入循环。统一贪心可能对它不利。候选做法：主比较用统一的贪心；
  另按各模型官方推荐参数再跑一遍，取两者中较高者作为对手分数（把"取较高模式"推广到解码参数）。

## 5. 对手清单与冻结日期

见 `eval/opponents.md`：官方参数量（含 embedding，按发布方口径；多模态模型按语言模型部分）在我们模型 0.7–1.3 倍之间的全部公开权重模型，
冻结日期之前发布的都算；Qwen3.5-0.8B 必比。按当前 `configs/main/pretrain.toml` 的 689.5M，区间为 [482.7M, 896.4M]（最终尺寸由第 12 章决定，冻结时重算）。
冻结日期：**TBD**。发布时补查冻结日之后的新模型，写进"发布后新增对手"一节，即使它们比我们强。

## 6. "超过"的判定标准

- **单个比较**：同一组题上的逐题得分做**配对 bootstrap**（10,000 次重抽，百分位法 95% 置信区间，种子 TBD），差值 = 我们 − 对手
  （对手取第 4 节规则下的较高者）。区间下界 > 0 → **超过**；上界 < 0 → **落后**；跨过 0 → **持平**（`zero/eval/bootstrap.py` 的 `decide`）。
- **E1 的重抽方式**：E1 是按类别加权的分数，重抽在每个类别内部分别进行（分层 bootstrap），再按第 2.1 节的权重合成。实现：`zero/eval/bootstrap.py` 的 `stratified_paired_bootstrap(a_by, b_by, weights, ...)`（只有一个类别时与 `paired_bootstrap` 结果相同）。
- **硬目标成立的条件**（候选）：对 `eval/opponents.md` 冻结清单里的**每一个**对手，E1 与 E2 **都**判为"超过"。
  这是交集-并集检验（intersection-union test）：每个单独比较在 5% 水平上显著，整体结论的第一类错误率不超过 5%，因此不另做多重比较校正。
- **其余所有分数**（各类别、各子集、通用组、只报告的基准）只作描述，不参与"超过"的判定，也不能事后改成主终点。
- 官方公布的分数并列展示，但不作为比较依据。

## 7. 去污染方法

- **n-gram 重叠**：所有训练数据（预训练、中期训练、SFT、教师合成数据、偏好数据、RL 任务）与第 2 节的全部评测集做 13-gram 重叠检查
  （`zero/data/decontam.py`；"词"= 英文字母数字串 / 单个汉字；不足 13 个词的题目整题匹配），命中的训练文档删除或整段屏蔽；统计命中率写进模型卡。
  检查对象（题目 / 题目 + 答案）与中文短题的 n 取值：TBD（第 11 章 `code/05_contamination.py` 演示了短题干"撞车"不等于泄漏）。
- **canary**：扫描训练语料里的已知 canary 字符串（如 BIG-bench 的 GUID），命中即整篇删除。
- **工具调用专项**：训练用的函数名、参数 schema 与 BFCL、ACEBench 比对：同名同参的函数整体剔除并计数；同名不同参的人工复核；记录数量。
- **自建环境**：`zero/post/envs/tool_env.py` 的训练任务与其固定 dev 集按问题文本去重（天气、寒暄两类问题空间太小，无法去重，单列报告）。
- 结果与方法写进模型卡。

## 8. 开发集与测试集的分工

- 挑 checkpoint、调超参、选提示词，只用开发集：`zero/eval/tasks/tool_dev.jsonl`（冻结文件）及 TBD 的通用开发集（候选：C-Eval dev、各基准的 validation split）。
- 第 2 节的测试基准只在闸门（GOAL.md 3.4）和最终评测时跑，每次运行都记入 `runs/ledger.md`。

## 9. 结果报告

- 全部结果（包括落后的项）写进模型卡和第 20 章；逐题结果与评测日志随发布一起公开。
- 没达到硬目标就不宣称"超过"，如实写差距分析。

## 修订记录

（冻结之后的修改只能追加在这里：日期、修改内容、理由。）
