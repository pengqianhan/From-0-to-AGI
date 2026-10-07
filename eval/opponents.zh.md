# 对手清单（候选，**未冻结**）

[English](opponents.md) · **中文**

> 状态：**草案**。在第二步阶段 7 确定冻结日期，经项目负责人确认后定稿（GOAL.md 3.2 第 3 条）。
> 定稿后，任何修改只能以带日期和理由的"修订"追加。

## 规则（GOAL.md 3.2）

- **范围**：官方参数量在我们模型的 **0.7–1.3 倍**之间的全部公开权重模型。冻结日期之前发布的都算。
  参数量含 embedding，按发布方的口径；多模态模型按语言模型部分的参数量计。
- 我们的模型：`configs/main/pretrain.toml`，**689.5M**（`uv run python -m zero.tools.count_params configs/main/pretrain.toml`；
  词表 65,536 为暂定值，第 13 章如果改词表，需要重算）→ 区间 **[482.7M, 896.4M]**。
- **标杆**：不管最终尺寸落在哪一档，**Qwen3.5-0.8B 必须比较**。
- 起点：`small-llms-under-5b-2026-08-30/inventory.md`（2026-08-30 更新）。冻结前还要检查该日期之后的新发布。
- 对手有思考 / 非思考两种模式时，两种都测，以较高分为准。所有对手都由我们用同一套框架重跑。

## 候选清单（按 inventory.md 筛出）

"档位"一栏是 inventory 里的档位名。标"待核实"的行，档位名落在区间内，但确切参数量（含 embedding）要在冻结前按发布方模型卡核对。名义上为 "0.5B" 的模型尤其要核对：区间下界是 482.7M，确切数字可能落在区间外。

| 组织 | 模型 | 档位 | 与 689.5M 之比（按档位） | 备注 |
|---|---|---:|---:|---|
| Alibaba / Qwen | **Qwen3.5-0.8B** | 0.8B | 1.16 | **必比标杆**。多模态，按语言模型部分计（模型卡：Language Model 0.8B，词表 248,320，与输出层共享）。思考 / 非思考都测。模型卡写明**默认非思考**，并提示思考模式下容易陷入循环（第 11 章，已核实 2026-09） |
| Alibaba / Qwen | Qwen3-0.6B | 0.6B | 0.87 | 模型卡：0.6B（不含 embedding 为 0.44B）。思考 / 非思考都测（已核实 2026-09） |
| Alibaba / Qwen | Qwen2.5-0.5B(-Instruct) | 0.49B | 0.71 | 模型卡：0.49B（不含 embedding 为 0.36B）→ 490M，**在区间内**，但离下界 482.7M 很近。最终尺寸变了要重算（已核实 2026-09） |
| Alibaba / Qwen | Qwen2.5-Coder-0.5B | 0.5B | 0.73 | 参数量待核实；代码模型 |
| Alibaba / Qwen | Qwen2-0.5B | 0.5B | 0.73 | 参数量待核实 |
| Alibaba / Qwen | Qwen1.5-0.5B | 0.5B | 0.73 | 参数量待核实 |
| Meta research | MobileLLM-600M | 600M | 0.87 | |
| Liquid AI | LFM2-700M | 700M | 1.02 | 混合架构 |
| CyberAgent | OpenCALM-medium（830M） | 830M | 1.20 | 日语模型 |
| BigScience | BLOOM-560M / BLOOMZ-560M | 560M | 0.81 | 多语言；BLOOMZ 为指令版 |
| Cerebras | Cerebras-GPT-590M | 590M | 0.86 | |
| Tencent | Hunyuan-0.5B | 0.5B | 0.73 | 参数量待核实 |
| OpenBMB | MiniCPM4-0.5B | 0.5B | 0.73 | 参数量待核实 |
| H2O.ai | H2O-Danube3-500M | 0.5B | 0.73 | 参数量待核实 |
| TII | Falcon-H1-0.5B | 0.5B | 0.73 | 混合注意力 / SSM；参数量待核实 |
| MBZUAI | MobiLlama-0.5B | 0.5B | 0.73 | 参数量待核实 |
| Swiss AI | Apertus-v1.1-0.5B | 0.5B | 0.73 | 完全开放；参数量待核实 |
| RWKV Foundation | RWKV 4/5/6/7 各版本 | ~169M–2.9B | — | 需逐个版本核实：是否有落在区间内的尺寸 |

### 区间外、但接近边界（不作为对手，冻结前复核参数量）

| 模型 | 档位 | 比值 | 说明 |
|---|---:|---:|---|
| OpenELM-450M | 450M | 0.65 | 低于下界 |
| Pythia-410M、OpenCALM-small(410M)、InkubaLM-0.4B | ~0.4B | ~0.6 | 低于下界 |
| SmolLM / SmolLM2-360M | 360M | 0.52 | 低于下界 |
| Gemma 3 1B、OLMo (2) 1B、Pythia-1B、MobileLLM-1B、MiniCPM5-1B、Falcon3-1B、PLaMo-2-1B 等 1B 档 | 1B | 1.45 | 高于上界（如确切参数量 < 896.4M，则纳入；冻结前核对） |

## 第二步的重跑计划与成本

- 按 `runs/RUNBOOK.md` 阶段 7 执行。先确定冻结日期与清单，钉死框架版本。再用 1–2 个对手试跑（核实 `zero/eval/bfcl.py` 的待核实项，以及 ACEBench 能否在新版 vLLM 上运行）。
- 每个对手：用官方模板、两种模式（如有）、相同的解码参数，跑 `eval/PREREGISTRATION.md` 第 2 节的全部基准，保存逐题结果。
- 成本估算（RUNBOOK）：约 25 个模型 × 每个约 1 卡时 ≈ $60–200（GOAL.md 3.4 预算约 $200）。单次超过 $100，先报批。花费记入 `runs/ledger.md`。

## 冻结时要补的信息

每个对手：确切参数量与出处链接、发布日期、许可证、HF 仓库与 commit、官方对话 / 工具调用模板、是否有思考模式。
另写明用于 BFCL 的 handler 名称（BFCL 已内置的直接用；没有的写适配层）。
