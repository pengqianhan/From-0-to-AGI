# 第一步完成汇报（GOAL.md 第 10 节阶段 5）

日期：2026-09-30。分支：`claude/transformer-learning-roadmap-euhvze`。

## 1. 课程完成情况

26 章全部完成。每章都有：

- 速读正文（`chapters/NN-*/README.md`，含"从极简到生产级""采用方与来源""引导问题""动手任务"，第 7 章起有"想深入：CS336"）；
- 极简代码（`code/`，CPU 上几秒到几分钟跑完，正文里的数字都来自它的真实输出）；
- 视频源码（`video/script.md` 事实清单 + 分镜旁白，`video/scenes.py`，`video/build.sh`），480p 样片都已渲染并逐镜检查版式，交付检查 `problems` 为空，时长 5–10 分钟（各章时长见根目录 README）；
- 自检 Skill（`.claude/commands/chNN-*.md`）。

26 章的 1080p 成片都已渲染（`bash chapters/NN-*/video/build.sh`），交付检查全部通过：1920×1080、有音轨、峰值约 −6 dB、`problems` 为空，时长 5.2–9.5 分钟；字幕 `video/subtitles.srt` 已提交。MP4 按 GOAL.md 第 7 节不进 git，发布到 B 站 / YouTube 后把链接写回各章 README。

生产级代码（`zero/`）：

| 检查 | 结果 |
|---|---|
| `uv run pytest` | 324 passed |
| `uv run ruff check zero tests` | 通过 |
| `uv run python -m zero.smoke`（CPU，约 1.3M 参数的 tiny 配置） | 10 个阶段全部通过（数据与分词器 → 预训练 → 中期训练 → SFT → 蒸馏 → DPO → GRPO → 评测 → HF/GGUF 导出（llama.cpp 跑通）→ demo），单线程 315 秒 |

## 2. 与 GOAL.md 的偏差（需要你知道）

1. **旁白 TTS**：构建环境的代理拦截了 edge-tts 的 WebSocket，改用离线的 sherpa-onnx + MeloTTS 中英混读模型（MIT）。`video_kit/tts.py` 保留了 edge-tts 后端，网络允许时设 `VIDEO_TTS=edge` 即可切换。**所有旁白都未经人工试听**，发布前需要听一遍（专有名词读法最可能出问题）。
2. **数据**：构建环境访问不了 huggingface.co，极简代码和冒烟测试用的是 `assets/tiny_corpus/` 里的小语料（来源与许可证见 `assets/tiny_corpus/LICENSES.md`）。
3. **冒烟测试的蒸馏教师**是 tiny SFT 模型自己（下载不了开源权重），只验证通路。
4. **阶段 0 试点没有停下来等确认**：按你"直到完成最终的课程，再交给我"的指示一路做完。第 1 章的风格如果需要调整，其余各章会跟着改。
5. **判分器修复**：写第 17 章时发现 `tool_env` 的两处漏洞（"执行结果碰巧相同就给满分"、"不需要工具的题不查回答内容"），已修复并加了回归测试。第 11、16–20 章引用的冒烟测试数字来自修复前的那次运行（各章有注明），修复后重跑的差异见各章"主线进度"。

## 3. 主线模型相对 GOAL.md 的调整（第 12–15 章的结论）

| 项目 | GOAL.md 初稿 | 现在 | 依据 |
|---|---|---|---|
| 预训练 token | 500B | **约 400B**（`max_steps = 762940`） | 500B 在 MFU 0.4 下约 $6.1K，超过 $5K 线；400B 约 $4.9K。WSD 稳定段随时可停，实测 MFU 后在闸门 1 定稿 |
| 每卡 batch | micro 8 × 累积 2 | **micro 4 × 累积 4**（每步仍是 524,288 token） | micro 8 在 DDP 下约 114 GiB/卡，放不下；micro 4 约 64 GiB。另加了激活检查点开关（约 27 GiB） |
| 优化器 | AdamW | AdamW 默认，**Muon 进阶梯实验和 AdamW 正面对比** | Muon 核实为共识（Kimi K2、GLM-4.5、DeepSeek-V4），迷你阶梯上领先 |
| 词表 | 待定 | 65,536，byte-level BPE，Qwen3.5 预切分正则 | 第 13 章的词表扫描与正则对比 |
| 架构 | 稠密 GQA | 不变（MLA、稀疏注意力、混合线性注意力都核实为共识，但对 0.69B / 32K 的收益不足以抵消风险，理由见第 21–23、26 章） | |

共识核实的逐项结论已追加到 GOAL.md 2.1 表格下方。

## 4. 尚未在 GPU 上验证的部分

> **2026-10 更新**：下面这份清单是第一步结束时的记录，保持原样。其中 BF16、FlashAttention（`enable_gqa=True` 默认走 Flash）、`torch.compile`、激活检查点、DDP / FSDP2 与跨卡断点续训（2 卡）、SFT / 蒸馏 / DPO / GRPO 的单卡 CUDA 通路、Muon 已在单张和 2 张 RTX 3090 上验证，见 [`runs/2026-10-01-gpu0-check/`](2026-10-01-gpu0-check/README.md)；32K 在 24GB 的卡上（单卡、2 卡、3 卡 FSDP）都放不下；vLLM、BFCL、ACEBench 仍未验证。

以下代码都在 CPU 上测过逻辑，但**没有在 GPU 上跑过**（`runs/RUNBOOK.md` 第 2 节"阶段 6"逐项列了验证命令和通过标准，预算 ≤ $50）：

- BF16 autocast、SDPA 走 FlashAttention 后端（`enable_gqa=True` 能否走 Flash 待核实）、`torch.compile`；
- 多卡 DDP、FSDP2、断点续训的跨卡一致性；激活检查点在 GPU 上的显存与吞吐；
- 32K 长序列训练（YaRN 配置）的显存；
- SFT / DPO / GRPO 在 GPU 上的通路（DPO、GRPO 目前是单进程实现；GRPO 吞吐不够时换 verl，先对拍）；
- vLLM 加载导出的 HF 模型、`hermes` 工具调用解析器；
- BFCL 的 `ZeroFCHandler`（`zero/eval/bfcl.py`）、ACEBench 适配层；
- Muon 在多卡上的实现（`zero/train/muon.py`）。

## 5. 第二步预计花费（H100 SXM $2.5/卡时，MFU 0.4）

| 阶段 | 内容 | 估算 | 需要批准（> $100） |
|---|---|---:|---|
| 6 | GPU 验证 + 实测 MFU | ≤ $50 | 否 |
| 7 | 对手重跑、预注册定稿 | $60–200 | 是（超 $100 时） |
| 8 | 阶梯实验（4 个尺寸 < $40）+ 消融、配方验证 | ≤ $1,200 | 是 |
| 9 | 预训练约 400B token | ~$4,900 | 是 |
| 9 | 中期训练约 26B token | ~$320 | 是 |
| 9 | 长上下文扩展到 32K 约 4.2B token | ~$196 | 是 |
| 10 | SFT | ~$17 | 否 |
| 10 | 蒸馏、DPO、GRPO | 阶段 6 实测后估算，预算 ~$1,500 | 是 |
| 10 | 最终评测与发布 | ~$200 | 视情况 |
| — | 第五部分约 1 亿参数的架构对比（可选） | ~$400 | 是 |
| **合计** | | **约 $8,500–9,000** | 在 $10K 内，余量约 $1,000 |

预训练的余量很小：实测 MFU 低于 0.391 时，400B token 就会超过 $5K 线，需要在闸门 1 减 token 或换更便宜的卡。

## 6. 需要你决定的事

1. **预注册（`eval/PREREGISTRATION.md` 草案）**：
   - E1 用 BFCL V4 去掉 Agentic（web search 依赖 SerpAPI、不可复现）后按官方权重重新归一化，是否同意？
   - 解码参数：所有模型统一贪心，还是对手用各自推荐的采样参数？
   - 思考模式对手的最大生成长度。
   - 冻结日期与最终对手清单（`eval/opponents.md`）。
2. **长上下文阶段的学习率**：中期训练把学习率退火到 0 之后，长上下文阶段要不要重新 warmup 到 1e-4（Qwen3、Llama 3、DeepSeek-V3、SmolLM3 的做法各不相同，见第 15 章），以及两者的先后顺序。
3. **批准**：中期训练（~$320）和长上下文（~$196）这两项超过 $100，需要你批准；预训练和后训练同理。
4. **模型权重的许可证**（第 20 章列了选项，比如 Apache-2.0）。
5. **蒸馏教师**：许可证允许的候选是 Qwen3 / Qwen3.5 系列（Apache-2.0）、DeepSeek-R1 / V4（MIT）、GLM-5、MiMo-V2-Flash（MIT）、gpt-oss（Apache-2.0）；Gemma 和 Llama 的条款会传到学生身上，不用。主线词表和它们都不同，只能做序列级蒸馏（第 17 章）。
6. **GPU 环境**：阶段 6 需要 8×H100（或同级）约 1.5 小时。

## 7. 仍标注"待核实"的主要事项

- H100 SXM 稠密 BF16 峰值取 989.5 TFLOPS（成本估算的分母）；
- 第 26 章：Kimi-K3 技术报告里的若干细节、GLM-5.3 参数量沿用 GLM-5 的模型卡、MiniMax-M3 的 MTP 配置；
- GRPO 的 clip-higher 是否已达共识；最新旗舰是否仍用 DPO；
- 部分数据集的许可证（SmolTalk2，xLAM / ToolACE / Hermes 数据生成模型的条款，Tülu 3 偏好混合的部分子集不可商用）。
