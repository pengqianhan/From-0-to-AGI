# 发布清单（第二步阶段 10 · 闸门 3 与发布）

[English](RELEASE_CHECKLIST.md) · **中文**

> 配套文档：第 20 章 [`chapters/20-release/`](../chapters/20-release/README.zh.md)、[`runs/RUNBOOK.md`](RUNBOOK.zh.md) 第 6 节、
> [`eval/PREREGISTRATION.md`](../eval/PREREGISTRATION.zh.md)。本清单在第一步写好，**所有项目都还没有执行**。
> 逐项打勾。每一项都在"证据"一栏写上文件路径或链接。打不了勾的项写明原因，不跳过。
> 发布前**等项目负责人最后确认**（GOAL.md 第 10 节阶段 10）。

## A. 闸门 3：最终评测（按预注册协议执行，一字不改）

| # | 项目 | 证据 | 状态 |
|---|---|---|---|
| A1 | 确认 `eval/PREREGISTRATION.md` 已定稿，记录登记 commit。冻结后的每次修改都写在"修订记录"里，带日期和理由 | commit hash | [ ] |
| A2 | 评测框架版本与预注册一致（lm-evaluation-harness、bfcl-eval、vLLM、evalplus）。`pip freeze` 的输出存档 | `runs/<日期>-final-eval/env.txt` | [ ] |
| A3 | 评测对象是**发布的那份权重**：导出的 HF 目录（`export_to_hf_qwen3(..., chat=True)`），不是训练 checkpoint。记下 `model.safetensors` 的 sha256 | sha256 | [ ] |
| A4 | 模板：我们的模型用导出目录里的 `chat_template`；对手用各自的官方模板。解码参数与预注册相同 | 评测日志 | [ ] |
| A5 | 专项基准（BFCL + 中文工具调用基准）和通用基准全部跑完，保存**逐题**结果（`--log_samples`） | `eval/results/final/` | [ ] |
| A6 | 对每个对手、每个基准做配对 bootstrap（`zero.eval.bootstrap`，10,000 次，95% CI；对手取思考 / 非思考中较高的一个） | 比较表 | [ ] |
| A7 | 按预注册的判定标准，给每一格写出"超过 / 持平 / 落后"。**全部列出**，不挑选基准 | `zero.eval.report` 生成的表 | [ ] |
| A8 | 硬目标是否成立：按预注册的"硬目标成立条件"逐条判断。不成立就**不宣称"超过"**，并写差距分析 | 结论段 | [ ] |
| A9 | 对手官方公布的分数并列展示，但只作参考 | 表格 | [ ] |
| A10 | 发布后新增对手：查冻结日期之后发布的、尺寸为我们 0.7–1.3 倍的模型。能跑就跑，写进"发布后新增对手"（Opponents added after release）一节，即使它们比我们强 | 清单 + 结果 | [ ] |
| A11 | 测试基准只在这里跑了一次；没有用它们挑 checkpoint 或调超参数（GOAL.md 第 11 节） | 说明 | [ ] |

## B. 去污染

| # | 项目 | 证据 | 状态 |
|---|---|---|---|
| B1 | 全部训练数据（预训练、中期训练、SFT、教师合成数据、偏好数据、RL 任务）与全部评测集做 13-gram 重叠检查（`zero/data/decontam.py`） | 命中率表 | [ ] |
| B2 | 工具调用数据的函数名、参数 schema 与 BFCL 等评测集比对。重合的剔除并计数 | 计数 | [ ] |
| B3 | 结果写进模型卡的"去污染检查"一节（Decontamination check） | 模型卡 | [ ] |

## C. 权重与格式

| # | 项目 | 证据 | 状态 |
|---|---|---|---|
| C1 | HF 目录：`config.json`（`Qwen3ForCausalLM`）、`generation_config.json`、`model.safetensors`（bf16）、`tokenizer.json`、`tokenizer_config.json`（含 `chat_template`） | 目录列表 | [ ] |
| C2 | `transformers` 加载后，logits 与 zero 一致（用 `tests/test_export_hf.py` 的做法，在最终权重上再跑一次） | 最大误差 | [ ] |
| C3 | `apply_chat_template` 的渲染结果与训练时的模板逐字一致（用一个带 tools 的例子） | diff 为空 | [ ] |
| C4 | vLLM：`vllm serve <dir> --enable-auto-tool-choice --tool-call-parser hermes` 能加载；带 tools 的请求返回结构化的 `tool_calls` | 请求 / 响应存档 | [ ] |
| C5 | GGUF：`python -m zero.export.gguf --hf-dir ... --outtype f16`。`llama-tokenize` 与我们的分词器逐 token 一致（中文、英文、代码、工具调用各一条） | 对拍输出 | [ ] |
| C6 | f32（或 f16）GGUF 的贪心输出与 zero 逐 token 一致（含 YaRN 配置） | 对拍输出 | [ ] |
| C7 | 量化：Q8_0、Q4_K_M。用 `llama-perplexity` 在我们自己的开发集上测 PPL 与 KLD（相对 f16），写进模型卡 | PPL / KLD 表 | [ ] |
| C8 | 量化版（至少 Q4_K_M）在工具调用开发集上的得分，与 bf16 对比。掉分明显，就在模型卡里写清楚，并推荐 Q8_0 | 表 | [ ] |
| C9 | Ollama：`Modelfile` 写 `FROM ./zero-Q4_K_M.gguf` + 对话模板。`ollama create` 后能对话 | 录屏 / 日志 | [ ] |
| C10 | 在一台普通笔记本上跑 Q4_K_M（记录型号、内存、系统）：记录内存占用与 token/s | 表 | [ ] |

## D. 本地 demo

| # | 项目 | 证据 | 状态 |
|---|---|---|---|
| D1 | `python -m zero.demo.cli --model <最终 HF 目录> --root <演示目录>`：计算器、日期、文件搜索三类问题各录一段 | 录屏 | [ ] |
| D2 | 失败的例子也录下来（调错工具、参数错误、编造结果），写进模型卡的"已知局限"（Known limitations） | 录屏 | [ ] |
| D3 | 文件搜索只在 `--root` 之内（`tests/test_demo.py` 覆盖了 `../` 和符号链接两种逃逸）。README 提醒用户不要把根目录设成 home | 说明 | [ ] |

## E. 许可证与署名（作者决定，执行者准备材料）

| # | 项目 | 证据 | 状态 |
|---|---|---|---|
| E1 | 权重许可证：作者在第 20 章列出的选项中选定。写进模型卡 YAML 的 `license:` 和仓库的 `LICENSE` | 决定记录 | [ ] |
| E2 | 每个训练数据集：名称、版本、许可证、署名要求（ODC-By 数据要求署名；CC-BY 要求署名；代码数据按原始许可证）。列成表 | 数据表 | [ ] |
| E3 | 蒸馏教师：名称、版本，以及许可证原文里关于"用输出训练其他模型"的条款 | 条款摘录 | [ ] |
| E4 | 代码仓库的许可证与模型的许可证分别写明 | LICENSE 文件 | [ ] |

## F. 发布物（GOAL.md 3.5）

| # | 项目 | 证据 | 状态 |
|---|---|---|---|
| F1 | Hugging Face：Base、SFT、最终版三个仓库（或一个仓库多个分支）。关键的中间 checkpoint 用 `revision` 分支发布（名称写清步数与 token 数） | 链接 | [ ] |
| F2 | GGUF 仓库：Q8_0、Q4_K_M（f16 可选） | 链接 | [ ] |
| F3 | 模型卡：用 `chapters/20-release/code/04_model_card.py` 从评测 JSON 生成骨架，再人工补文字。**所有"TBD after training"（待训练）占位必须清零** | 模型卡 | [ ] |
| F4 | 可复现配方：`configs/main/`、数据处理脚本、训练日志摘要、`runs/ledger.md` 里的花费 | 链接 | [ ] |
| F5 | 评测报告（逐题结果 + 日志）与模型一起公开 | 链接 | [ ] |
| F6 | 课程回填：第 11–20 章的"主线进度"补上真实结果。第 20 章视频里标"极小配置演示"的镜头，换成真实数据重新渲染 | PR | [ ] |
| F7 | 花费记入 `runs/ledger.md` | 行号 | [ ] |
| F8 | **等项目负责人最后确认**，再把仓库设为公开 | 确认记录 | [ ] |

## G. 发布之后

- 开 issue 模板 / 讨论区收集反馈：失败的工具调用例子（附输入、工具定义、输出）、量化版的问题、许可证问题。
- 别人复现出的评测分数与我们不同时，先核对框架版本、模板、解码参数，再更新模型卡。**改动要写明日期和原因**。
- 不根据发布后的反馈悄悄改评测表。新的评测结果以"修订"的形式追加。
