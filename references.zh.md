# 参考资料

[English](references.md) · **中文**

这个教程不是从零构建的，参考了很多前人的工作。以下是一些重要的参考资料：
- [CS336](https://cs336.stanford.edu/)
- [minimind](https://github.com/jingyaogong/minimind)
- [nanoGPT](https://github.com/karpathy/nanoGPT)
- [Hands-On Modern Reinforcement Learning](https://walkinglabs.github.io/hands-on-modern-rl/preface/intro)
- [Mathematical theory of deep learning](https://arxiv.org/abs/2407.18384)
- [Scaling Laws That Extrapolate 300× Past the Fit](https://openathena.ai/blog/delphi/)
- [12G 显存，够吗？——消费级 GPU 上做 LLM 算法研究的完整路线](https://x.com/dashen_wang/status/2038970877732380924)
- [Reproducing all of Schmidhuber’s papers (1990-2025)](https://github.com/cybertronai/schmidhuber-problems/blob/main/VISUAL_TOUR.md)
- [AI 研究方法的演变](https://github.com/owlman/CS_StudyNotes/blob/master/02_%E5%9F%BA%E7%A1%80%E7%90%86%E8%AE%BA%E5%AD%A6%E4%B9%A0/%E4%BA%BA%E5%B7%A5%E6%99%BA%E8%83%BD/01.%E5%AD%A6%E4%B9%A0%E7%AC%94%E8%AE%B0/AI%20%E7%A0%94%E7%A9%B6%E6%96%B9%E6%B3%95%E7%9A%84%E6%BC%94%E5%8F%98.md)：从 1950 年开始的 AI 研究方法的演变。内容涵盖从符号主义到连接主义，再到现代深度学习的历程。
- [Advanced Natural Language Processing / Spring 2026 from CMU](https://cmu-l3.github.io/anlp-spring2026/)，[code](https://github.com/cmu-l3/anlp-spring2026-code)
- [Maths, CS & AI Compendium](https://github.com/HenryNdubuaku/maths-cs-ai-compendium/tree/main)
- [Pi Book](https://github.com/antinomie-lab/pi-book/tree/main)：这本书的格式和交互方式很适合学习。关于 Pi agent 的内容也可以参考。
- [Pretraining a Mini Kimi K3](https://books.vizuara.ai/book/pretraining-a-mini-k3)：K3 这个新架构的预训练可以参考这里。
- [1.5万字速通LLM主流模型结构（Llama、Qwen、GLM、Deepseek...）](https://zhuanlan.zhihu.com/p/2060741715095560795)：这篇文章详细介绍和比较了主流 LLM 的模型结构，涵盖 Llama、Qwen、GLM、DeepSeek 等模型的架构特点和应用场景。
- [Small-Scale Experiments: Are We There Yet?](https://arxiv.org/pdf/2608.11859)：这篇文章介绍了小规模 LLM 的 scaling law。小模型对超参数极度敏感，这掩盖了 scaling law 的真实存在。只有充分调优后，小规模实验才能可靠地预测大规模结果。
- 数据集：fine-web edu、ultra-fineweb-L1
- [Marin 535B-A23B 训练直播](https://wandb.ai/marin-community/marin_moe/reports/535B-A23B-18T-Token-Hero-Run-Scaling-Ladder--VmlldzoxNzc2MDM5Ng)、[github details](https://github.com/marin-community/marin/issues/8435)、[data composition:](https://storage.googleapis.com/marin-public/held/harrier-k40-cluster-overview/2026.08.18/index.html?revision=uniform-sampling)
- [UNDERSTANDING TRANSFORMERS AND ATTENTIONMECHANISMS: AN INTRODUCTION FOR APPLIEDMATHEMATICIANS](https://arxiv.org/pdf/2604.00965)
- [Puro-2B](https://www.alphaxiv.org/abs/2608.27370)：Puro-2B：穷实验室在 RTX 5090 上以 5090 美元预算训练的 Qwen2-1.5B
- [Awesome Claude Opus 5.5 Videos](https://github.com/athemeroy/awesome-opus-5-5-videos)：用 Claude Opus 5.5 做视频的案例合集。重点看"教育讲解片"路径（Manim/Remotion + TTS）和 [七类可复用提示词模板](https://github.com/athemeroy/awesome-opus-5-5-videos/blob/main/docs/prompt-playbook.zh-CN.md)。每章配套视频的制作流程参考这里。
- [XiaomiMiMo/verl](https://github.com/XiaomiMiMo/verl)：小米 MiMo 开源的智能体强化学习训练代码（基于 verl 0.9.0.dev，Apache-2.0）。它复现 MiMo-V2.6 技术报告第 7 节的 RL 配方，包含五类 RL 环境：代码（可执行测试判分）、网络安全、通用知识工作（rubric 判分）、网页开发（视觉判分）、音乐。它还公开了训练数据 [MiMo-V2.6-RL-oss](https://huggingface.co/datasets/XiaomiMiMo/MiMo-V2.6-RL-oss)。可作为第 19 章 RL 环境与奖励设计的参考，也是第二步实际 RL 训练可评估使用的框架。

## 全开源小模型（权重 + 训练数据 + 训练代码）

收录标准：权重可下载，**并且**预训练数据公开（给出数据集链接，或给出能从公开数据确定性重建的流水线），**并且**训练代码或配方公开；参数量大致 ≤ 10B，重点看 ≤ 5B。所有链接、许可证和发布情况均于 2026-10 逐个打开核实。只满足部分条件的模型放在本节末尾的"部分开源"列表里，并写明缺什么。

### OpenBMB MiniCPM
- **MiniCPM5-1B / MiniCPM5-2B**（1.08B / 2.52B，标准 `LlamaForCausalLM` 稠密结构；1B 于 2026-05、2B 于 2026-09-07 发布；同时放出 Base、Midtrain（仅 2B）、SFT 各阶段 checkpoint）
  - 模型：[MiniCPM5-1B](https://huggingface.co/openbmb/MiniCPM5-1B)、[MiniCPM5-1B-Base](https://huggingface.co/openbmb/MiniCPM5-1B-Base)、[MiniCPM5-2B](https://huggingface.co/openbmb/MiniCPM5-2B)、[MiniCPM5-2B-Base](https://huggingface.co/openbmb/MiniCPM5-2B-Base)、[MiniCPM5-2B-Midtrain](https://huggingface.co/openbmb/MiniCPM5-2B-Midtrain)；合集 [MiniCPM5](https://huggingface.co/collections/openbmb/minicpm5)
  - 预训练数据（UltraData 分级数据体系，合集 [UltraData](https://huggingface.co/collections/openbmb/ultradata)）：1B 用 [Ultra-FineWeb](https://huggingface.co/datasets/openbmb/Ultra-FineWeb)（L2 精选网页，约 1T 英文 + 120B 中文 token；其 L1 底料为 [Ultra-FineWeb-L1](https://huggingface.co/datasets/openbmb/Ultra-FineWeb-L1)）、[Ultra-FineWeb-L3](https://huggingface.co/datasets/openbmb/Ultra-FineWeb-L3)（问答生成 + 多风格改写的合成数据，400B+ 英文 + 200B+ 中文 token，用于衰减阶段）、[UltraData-Math](https://huggingface.co/datasets/openbmb/UltraData-Math)（290B+ token，L1/L2/L3 三级）；2B 在此基础上再加 [UltraX-Preview](https://huggingface.co/datasets/openbmb/UltraX-Preview)（函数调用式逐条编辑精炼的网页数据，5 份各约 20B token）和 [UltraData-Code](https://huggingface.co/datasets/openbmb/UltraData-Code)（L0–L3 分级代码数据）
  - 后训练数据：[UltraData-SFT-2605](https://huggingface.co/datasets/openbmb/UltraData-SFT-2605)（1B-SFT 的全部核心领域 SFT 数据，需在 HF 上申请访问）、[UltraData-SFT-Agent-2609](https://huggingface.co/datasets/openbmb/UltraData-SFT-Agent-2609)（约 50 万条 Agent 轨迹）、[UltraData-RL-2609](https://huggingface.co/datasets/openbmb/UltraData-RL-2609)（2B 的 RL 数据，8 万+ 条可验证奖励任务）；1B 的 RL 另用了公开的 DAPO-Math-17k、TriviaQA、NQ-Open 等，以及未单独发布的合成 RLVR 数据和成对 RLHF 信号
  - 代码 / 配方：[MiniCPM GitHub](https://github.com/OpenBMB/MiniCPM) 的 "Training Recipe" 一节和流程图给出阶段级配方（stable → 短衰减 4K → 长衰减 32K/128K → 中期训练 → SFT → 多个 RL 教师 → 在线策略蒸馏 OPD，衰减、中期训练、SFT 各阶段标了 token 数）。仓库另有微调脚本与 TRL / LLaMA-Factory 等微调 cookbook；数据侧代码有 [UltraX](https://github.com/openbmb/UltraX)、[UltraData-Math](https://github.com/UltraData-OpenBMB/UltraData-Math)。**MiniCPM5 的预训练代码、stable 阶段 token 数和数据配比没有公开**。OpenBMB 的预训练框架 [ForgeTrain](https://github.com/OpenBMB/ForgeTrain)（Apache-2.0）目前只覆盖 MiniCPM4-0.5B/8B，MiniCPM5-1B 版列在路线图上
  - 技术报告：暂无 MiniCPM5 专门报告；数据体系见 [Data Science and Technology Towards AGI Part I: Tiered Data Management](https://arxiv.org/abs/2602.09003)，另有 [Ultra-FineWeb](https://arxiv.org/abs/2505.05427)、[UltraX](https://arxiv.org/abs/2607.08646)、[MiniCPM4](https://arxiv.org/abs/2506.07900)
  - 许可证：模型 Apache-2.0；上述数据集卡均标 Apache-2.0（UltraX 数据卡提醒需同时遵守各来源数据集的许可证）
  - 课程参考：中英双语预训练数据的"L0–L4 分级治理"和合成改写 → 第 13 章；stable + 分段衰减（WSD）→ 第 6/12 章；按上下文长度分段衰减 → 第 15 章；多个 RL 教师 + 在线策略蒸馏 → 第 17/19 章；中文数据量大（Ultra-FineWeb 约 120B 中文 token，L3 合成数据中文 200B+ token），是主线模型中文数据的首选参考 → 主线模型

### Hugging Face Smol
截至 2026-10，HuggingFaceTB 下最新的 Smol 文本模型仍是 SmolLM3，没有查到 SmolLM4 或更新的正式版本。
- **SmolLM3**（3B，2025-07；11.2T token 三阶段预训练 + 4K→32K→64K 长上下文 + 140B token 推理中期训练 + SFT + APO）
  - 模型：[SmolLM3-3B-Base](https://huggingface.co/HuggingFaceTB/SmolLM3-3B-Base)、[SmolLM3-3B](https://huggingface.co/HuggingFaceTB/SmolLM3-3B)、中间 checkpoint [SmolLM3-3B-checkpoints](https://huggingface.co/HuggingFaceTB/SmolLM3-3B-checkpoints)
  - 预训练数据：[SmolLM3 预训练数据合集](https://huggingface.co/collections/HuggingFaceTB/smollm3-pretraining-datasets-685a7353fdc01aecde51b1d9)（FineWeb-Edu、DCLM、FineWeb2 / FineWeb2-HQ、FineMath、MegaMath、Stack-Edu、The Stack v2、dolmino-mix 等 15 个）；后训练数据 [smoltalk2](https://huggingface.co/datasets/HuggingFaceTB/smoltalk2)（Mid / SFT / Preference 三个子集）
  - 代码 / 配方：[huggingface/smollm](https://github.com/huggingface/smollm/tree/main/text/pretraining/smollm3)（5 个 nanotron 配置，写明各阶段数据权重）、[smollm3-configs](https://huggingface.co/datasets/HuggingFaceTB/smollm3-configs)、后训练 [alignment-handbook recipes/smollm3](https://github.com/huggingface/alignment-handbook/tree/main/recipes/smollm3)
  - 技术报告：[SmolLM3 博客](https://huggingface.co/blog/smollm3)（无 arXiv 论文）；配套长文 [The Smol Training Playbook](https://huggingface.co/spaces/HuggingFaceTB/smol-training-playbook)
  - 许可证：模型 Apache-2.0；数据按各来源（FineWeb 系 ODC-BY，DCLM CC-BY-4.0，The Stack v2 需同意条款，stage 3 用到的 natural_reasoning 为 CC-BY-NC-4.0）
  - 注意：配置里的路径是内部 S3 上的分词后数据，只有 stage 1 标了对应的 HF 数据集，后几阶段有少数数据名找不到一一对应的公开仓库
  - 课程参考：多阶段配比（网页 → 加大代码/数学比例）→ 第 13 章；长上下文分段扩展 → 第 15 章；中期训练 + SFT + APO → 第 15/16/18 章；Playbook 里的消融方法和训练排障 → 第 12/14 章
- **SmolLM2**（135M / 360M / 1.7B，2024-10；1.7B 训 11T token、四阶段，360M/135M 训 4T/2T token）
  - 模型：[SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M)、[SmolLM2-360M](https://huggingface.co/HuggingFaceTB/SmolLM2-360M)、[SmolLM2-1.7B](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B)，另有各尺寸的 intermediate-checkpoints 仓库
  - 数据：[FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)、[DCLM-Edu](https://huggingface.co/datasets/HuggingFaceTB/dclm-edu)、[FineMath](https://huggingface.co/datasets/HuggingFaceTB/finemath)、[Stack-Edu](https://huggingface.co/datasets/HuggingFaceTB/stack-edu)（只给 Software Heritage ID，正文需按 The Stack v2 条款另行下载）、[SmolLM-Corpus](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus)（含 Cosmopedia v2）；后训练 [smoltalk](https://huggingface.co/datasets/HuggingFaceTB/smoltalk)
  - 代码 / 配方：[smollm/text/pretraining/smollm2](https://github.com/huggingface/smollm/tree/main/text/pretraining/smollm2)（nanotron 配置，但各阶段配比只写在论文里）、[alignment-handbook recipes/smollm2](https://github.com/huggingface/alignment-handbook/tree/main/recipes/smollm2)
  - 技术报告：[SmolLM2: When Smol Goes Big](https://arxiv.org/abs/2502.02737)
  - 许可证：模型 Apache-2.0；数据 ODC-BY / CC-BY-4.0 / The Stack v2 条款
  - 课程参考：根据中途评测"在线"调整配比的多阶段训练 → 第 13 章；过训练（1.7B 训 11T）→ 第 12 章；小尺寸（135M/360M）适合在单卡上做对照实验 → 主线模型
- **SmolLM**（135M / 360M / 1.7B，2024-07；600B / 600B / 1T token）
  - 模型：[SmolLM-135M](https://huggingface.co/HuggingFaceTB/SmolLM-135M)、[SmolLM-360M](https://huggingface.co/HuggingFaceTB/SmolLM-360M)、[SmolLM-1.7B](https://huggingface.co/HuggingFaceTB/SmolLM-1.7B)
  - 数据：[SmolLM-Corpus](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus)（Cosmopedia v2 28B + FineWeb-Edu-dedup 220B + Python-Edu 4B token，ODC-BY）；实际配置里还混了 open-web-math、StackOverflow 等少量数据
  - 代码 / 配方：[smollm/text/pretraining/smollm1](https://github.com/huggingface/smollm/tree/main/text/pretraining/smollm1)（三个尺寸的 nanotron YAML）
  - 技术报告：[SmolLM 博客](https://huggingface.co/blog/smollm)
  - 许可证：模型 Apache-2.0，数据 ODC-BY
  - 课程参考：数据量小、配置简单，是复现一个"全流程小模型"最容易上手的起点 → 主线模型、第 14 章
- **配套数据与代码**
  - [FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb)（ODC-BY，约 18.5T token；论文 [arXiv 2406.17557](https://arxiv.org/abs/2406.17557)）、[FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)（ODC-BY）、[FineWeb2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2)（ODC-BY，1000+ 语言；论文 [arXiv 2506.20920](https://arxiv.org/abs/2506.20920)）、[Cosmopedia](https://huggingface.co/datasets/HuggingFaceTB/cosmopedia)（Apache-2.0，合成教科书）→ 第 13 章
  - [nanotron](https://github.com/huggingface/nanotron)（Apache-2.0，3D 并行预训练框架）→ 第 14 章；[huggingface/smollm](https://github.com/huggingface/smollm)（Apache-2.0，含 FineWeb-Edu 分类器、FineMath 流水线、去污染与评测脚本）→ 第 11/13 章

### NVIDIA
逐个核实后，NVIDIA 在 10B 以内**没有**能进主表的模型。Nemotron Nano 2 和 Nemotron 3 Nano 4B 只公开了"大部分"预训练数据，而且需要申请访问（见下方"部分开源"）。Nemotron-H、Hymba、Nemotron-Flash、Minitron、Nemotron-Mini 等的模型卡没有给出可下载的完整训练数据。OpenReasoning-Nemotron、OpenCodeReasoning-Nemotron 是在 Qwen2.5 上微调的。它们公开的数据集本身很有参考价值：
- **Nemotron 预训练数据**（Nano 2 一起发布，合计约 6.6T token）
  - 数据：[Nemotron-CC v1](https://data.commoncrawl.org/contrib/Nemotron/Nemotron-CC/index.html)（6.3T token，放在 Common Crawl 上，遵守 Common Crawl 使用条款；论文 [Nemotron-CC](https://arxiv.org/abs/2412.02595)）、[Nemotron-CC-v2](https://huggingface.co/datasets/nvidia/Nemotron-CC-v2)、[Nemotron-CC-Math-v1](https://huggingface.co/datasets/nvidia/Nemotron-CC-Math-v1)（133B token；论文 [arXiv 2508.15096](https://arxiv.org/abs/2508.15096)）、[Nemotron-Pretraining-Code-v1](https://huggingface.co/datasets/nvidia/Nemotron-Pretraining-Code-v1)（GitHub 代码只给 repo / commit / 路径元数据，需要自己重新抓取）、[Nemotron-Pretraining-SFT-v1](https://huggingface.co/datasets/nvidia/Nemotron-Pretraining-SFT-v1)；合集 [Nemotron pre-training dataset](https://huggingface.co/collections/nvidia/nemotron-pre-training-dataset-689d9de36f84279d83786b35)，免申请的样本 [Nemotron-Pretraining-Dataset-sample](https://huggingface.co/datasets/nvidia/Nemotron-Pretraining-Dataset-sample)
  - Nemotron 3 时期新增：[Nemotron-CC-v2.1](https://huggingface.co/datasets/nvidia/Nemotron-CC-v2.1)、[Nemotron-CC-Code-v1](https://huggingface.co/datasets/nvidia/Nemotron-CC-Code-v1)、[Nemotron-Pretraining-Code-v2](https://huggingface.co/datasets/nvidia/Nemotron-Pretraining-Code-v2)（需申请），以及免申请的 [Nemotron-Pretraining-Specialized-v1](https://huggingface.co/datasets/nvidia/Nemotron-Pretraining-Specialized-v1)（270.7B token，CC-BY-4.0，其中 Wiki 改写子集为 CC-BY-SA、科学代码子集为 GFDL）
  - 后训练数据：[Nemotron-Post-Training-Dataset-v1](https://huggingface.co/datasets/nvidia/Nemotron-Post-Training-Dataset-v1)、[Nemotron-Post-Training-Dataset-v2](https://huggingface.co/datasets/nvidia/Nemotron-Post-Training-Dataset-v2)（需申请）、[Llama-Nemotron-Post-Training-Dataset](https://huggingface.co/datasets/nvidia/Llama-Nemotron-Post-Training-Dataset)（均为 CC-BY-4.0）
  - 许可证：带 gated 标记的数据集需同意 "NVIDIA Data Agreement for Model Training"，只允许用于训练模型，不是 CC 或 OSI 许可证
  - 课程参考：Nemotron-CC 的"分类器集成打分 + 对低质量数据做合成改写"→ 第 13 章；Nemotron-CC-Math 的数学网页抽取 → 第 13 章；后训练数据集 → 第 16/17 章
- 训练框架：[Megatron-LM](https://github.com/NVIDIA/Megatron-LM)、[Megatron-Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge)、[NeMo-RL](https://github.com/NVIDIA-NeMo/RL)（均 Apache-2.0）→ 第 14/19 章；配方仓库 [NVIDIA-NeMo/Nemotron](https://github.com/NVIDIA-NeMo/Nemotron) 目前只给 30B-A3B 等大模型的完整流程，并注明只用了开源数据子集，复现不出报告里的分数

### AllenAI
[allenai.org/open-models](https://allenai.org/open-models) 页面较旧（只列 OLMo 2、Molmo、OLMoE App）。最新模型、数据和代码的汇总见 [allenai.org/olmo](https://allenai.org/olmo)。Ai2 的模型都附带全部中间 checkpoint 和训练日志。
- **Olmo 3**（7B，2025-11；另有 32B 不在范围内）：5.9T token 预训练 → Dolmino 100B token 中期训练 → Longmino 50B token 长上下文（65K），后训练分 Think / Instruct / RL-Zero 三条线
  - 模型：[Olmo-3-1025-7B](https://huggingface.co/allenai/Olmo-3-1025-7B)（Base）、[Olmo-3-7B-Think](https://huggingface.co/allenai/Olmo-3-7B-Think)、[Olmo-3-7B-Instruct](https://huggingface.co/allenai/Olmo-3-7B-Instruct)，各带 SFT / DPO 阶段 checkpoint；RL-Zero 系列（Math / Code / IF / General / Mix，直接从 Base 做 RL）
  - 数据：预训练 [dolma3_mix-6T-1025-7B](https://huggingface.co/datasets/allenai/dolma3_mix-6T-1025-7B)（专供复现 7B；部分 olmOCR PDF 文本被替换为 `[REMOVED]`）、中期训练 [dolma3_dolmino_mix-100B-1025](https://huggingface.co/datasets/allenai/dolma3_dolmino_mix-100B-1025)、长上下文 [dolma3_longmino_mix-50B-1025](https://huggingface.co/datasets/allenai/dolma3_longmino_mix-50B-1025)、未配比的原始池 [dolma3_pool](https://huggingface.co/datasets/allenai/dolma3_pool)（约 9.3T）；后训练数据 Dolci 系列（如 [Dolci-Instruct-SFT](https://huggingface.co/datasets/allenai/Dolci-Instruct-SFT)）
  - 代码 / 配方：[OLMo-core](https://github.com/allenai/OLMo-core)（`src/scripts/official/OLMo3/`）、[open-instruct](https://github.com/allenai/open-instruct)、数据重建 [dolma3](https://github.com/allenai/dolma3)、评测 [OLMES](https://github.com/allenai/olmes)
  - 技术报告：[Olmo 3](https://arxiv.org/abs/2512.13961)
  - 许可证：模型 Apache-2.0；Dolma 3 / Dolci 为 ODC-BY（Dolci 部分子集按来源许可证）
  - 课程参考：预训练 → 中期训练 → 长上下文三阶段是本课第 14/15 章的完整对照；RL-Zero 和 Think 线 → 第 19 章；去污染工具 decon → 第 11 章。7B 超出主线尺寸，但它是目前最完整的"全流程复现"范本 → 主线模型
- **OLMo 2**（1B / 7B；7B 于 2024-11，1B 于 2025-04 至 05 间发布；13B、32B 不在范围内）
  - 模型：[OLMo-2-0425-1B](https://huggingface.co/allenai/OLMo-2-0425-1B)（1.48B，4T token）、[OLMo-2-1124-7B](https://huggingface.co/allenai/OLMo-2-1124-7B)（4T token），均有 SFT / DPO / Instruct checkpoint
  - 数据：预训练 [olmo-mix-1124](https://huggingface.co/datasets/allenai/olmo-mix-1124)、中期训练 [dolmino-mix-1124](https://huggingface.co/datasets/allenai/dolmino-mix-1124)、SFT [tulu-3-sft-olmo-2-mixture](https://huggingface.co/datasets/allenai/tulu-3-sft-olmo-2-mixture)
  - 代码 / 配方：[OLMo](https://github.com/allenai/OLMo)（`configs/official-0425/`、`configs/official-1124/` 下有 1B / 7B 的阶段配置；该仓库现已标为弃用，新代码在 OLMo-core）、[open-instruct](https://github.com/allenai/open-instruct)
  - 技术报告：[2 OLMo 2 Furious](https://arxiv.org/abs/2501.00656)（已在第 6 章条目中引用）
  - 许可证：模型 Apache-2.0；olmo-mix / dolmino-mix 为 ODC-BY（其中 DCLM 部分 CC-BY-4.0，StackExchange 部分 CC-BY-SA）
  - 课程参考：RMSNorm + QK-Norm + z-loss 的稳定性配方 → 第 6/9 章；退火阶段换高质量数据再做"模型汤"（多个退火结果取平均）→ 第 15 章；RLVR → 第 19 章。1B 版本和本课主线模型尺寸最接近 → 主线模型
- **OLMo / OLMo 0424 / OLMo 0724**（1B、7B，2024）
  - 模型：[OLMo-1B](https://huggingface.co/allenai/OLMo-1B)、[OLMo-7B](https://huggingface.co/allenai/OLMo-7B)、[OLMo-7B-0424-hf](https://huggingface.co/allenai/OLMo-7B-0424-hf)、[OLMo-1B-0724-hf](https://huggingface.co/allenai/OLMo-1B-0724-hf)、[OLMo-7B-0724-hf](https://huggingface.co/allenai/OLMo-7B-0724-hf)
  - 数据：[Dolma](https://huggingface.co/datasets/allenai/dolma)（v1.5 / v1.7，ODC-BY）；数据工具包 [dolma](https://github.com/allenai/dolma)
  - 代码 / 配方：[OLMo](https://github.com/allenai/OLMo)
  - 技术报告：[OLMo: Accelerating the Science of Language Models](https://arxiv.org/abs/2402.00838)、[Dolma](https://arxiv.org/abs/2402.00159)
  - 许可证：模型 Apache-2.0，数据 ODC-BY
  - 课程参考：每 1000 步一个 checkpoint，适合观察训练动态 → 第 12/14 章；Dolma 的清洗流水线 → 第 13 章
- **OLMoE**（总参数 6.9B、激活 1.3B 的 MoE，2024-09 / 2025-01）
  - 模型：[OLMoE-1B-7B-0924](https://huggingface.co/allenai/OLMoE-1B-7B-0924)、[OLMoE-1B-7B-0125](https://huggingface.co/allenai/OLMoE-1B-7B-0125)
  - 数据：[OLMoE-mix-0924](https://huggingface.co/datasets/allenai/OLMoE-mix-0924)（ODC-BY，主体为 DCLM-Baseline）；0125 版的退火另用 dolmino-mix-1124
  - 代码 / 配方：[allenai/OLMoE](https://github.com/allenai/OLMoE)（Apache-2.0）
  - 技术报告：[OLMoE: Open Mixture-of-Experts Language Models](https://arxiv.org/abs/2409.02060)
  - 许可证：模型 Apache-2.0，数据 ODC-BY
  - 课程参考：全开源的小 MoE，路由、负载均衡和专家数量的消融都有公开数据 → 第 24 章
- **Olmo Hybrid**（7.4B，2026-03）：3 层 Gated DeltaNet 配 1 层注意力的混合架构
  - 模型：[Olmo-Hybrid-7B](https://huggingface.co/allenai/Olmo-Hybrid-7B)（另有 Instruct / Think 的 SFT、DPO 版）
  - 数据：[dolma3_mix-6T](https://huggingface.co/datasets/allenai/dolma3_mix-6T)（5.5T token）+ dolmino 100B + longmino 50B（同 Olmo 3）
  - 代码 / 配方：[OLMo-core](https://github.com/allenai/OLMo-core)（`src/scripts/official/OLMo-hybrid/`）
  - 技术报告：[Olmo Hybrid: From Theory to Practice and Back](https://arxiv.org/abs/2604.03444)
  - 许可证：模型 Apache-2.0，数据 ODC-BY
  - 课程参考：同一套数据下线性注意力混合架构与纯注意力的对照 → 第 23 章
- **Bolmo**（1B / 7B，2025-12）：把 OLMo-2-0425-1B 和 Olmo-3-1025-7B 改造成字节级模型
  - 模型：[Bolmo-1B](https://huggingface.co/allenai/Bolmo-1B)、[Bolmo-7B](https://huggingface.co/allenai/Bolmo-7B)；数据 [bolmo_mix](https://huggingface.co/datasets/allenai/bolmo_mix)（ODC-BY）；代码 [bolmo-core](https://github.com/allenai/bolmo-core)；报告 [arXiv 2512.15586](https://arxiv.org/abs/2512.15586)；模型 Apache-2.0
  - 课程参考：不用 BPE 分词、直接建模字节的做法 → 第 7 章

### 其他
- **nanochat（Karpathy）**（d32 约 1.9B、d34 约 2.2B，2025-10 / 2025-11）：单个 8×H100 节点上从分词器、预训练、SFT、评测到聊天界面的完整最小实现
  - 模型：[nanochat-d32](https://huggingface.co/karpathy/nanochat-d32)、[nanochat-d34](https://huggingface.co/karpathy/nanochat-d34)（原生 `.pt` checkpoint，不是 transformers 格式）
  - 数据：预训练 [fineweb-edu-100b-shuffle](https://huggingface.co/datasets/karpathy/fineweb-edu-100b-shuffle)（2026-03 起仓库默认改用 [climbmix-400b-shuffle](https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle)）；SFT 用 SmolTalk 等公开数据
  - 代码 / 配方：[karpathy/nanochat](https://github.com/karpathy/nanochat)（`runs/speedrun.sh` 一个脚本跑完全流程；只调 `--depth` 一个旋钮，其余超参数按算力最优自动推出）
  - 许可证：MIT
  - 课程参考：和本课主线模型规模、预算最接近的全流程参考实现 → 主线模型；"GPT-2 speedrun"排行榜记录了每项训练改进带来的加速 → 第 14 章
- **MobileLLM-R1（Meta）**（140M / 360M / 950M，2025-09；面向端侧推理的 1B 以下模型）
  - 模型：[MobileLLM-R1-950M](https://huggingface.co/facebook/MobileLLM-R1-950M)、[MobileLLM-R1-950M-base](https://huggingface.co/facebook/MobileLLM-R1-950M-base)、[MobileLLM-R1-360M](https://huggingface.co/facebook/MobileLLM-R1-360M)、[MobileLLM-R1-140M](https://huggingface.co/facebook/MobileLLM-R1-140M)（下载需在 HF 上人工审批）
  - 数据：预训练、中期训练、SFT 用到的 FineWeb-Edu、StarCoderData、FineMath、dolmino-mix-1124、Nemotron-CC-Math-v1、OpenMathReasoning、tulu-3-sft-olmo-2-mixture 等全部是公开数据集，README 给出各阶段配比；不提供打包好的语料，其中 NVIDIA 的数据需申请
  - 代码 / 配方：[facebookresearch/MobileLLM-R1](https://github.com/facebookresearch/MobileLLM-R1)（预训练、中期训练、SFT 的复现代码和数据配比；中间 checkpoint 链接标注"即将提供"）
  - 技术报告：[MobileLLM-R1: Exploring the Limits of Sub-Billion Language Model Reasoners with Open Training Recipes](https://arxiv.org/abs/2509.24945)
  - 许可证：**FAIR 非商用研究许可**；数据按各来源
  - 课程参考：中期训练以 Llama-3.1-8B-Instruct 为教师、最小化 logits 的 KL 做蒸馏 → 第 17 章；1B 以下模型的推理数据配方 → 第 15/16 章
- **K2-Horizon（LLM360 / IFM）**（名义 0.9B / 3.7B / 7B，HF 统计的总参数为 1.08B / 5.06B / 9.0B；2026-09）：0.9B 预训练 5T token，3.7B / 7B 预训练 22.9T token，之后两段中期训练扩到 128K，再做多个 RL 专家 → 权重合并 → 在线策略蒸馏
  - 模型：[K2-Horizon-0.9B](https://huggingface.co/IFM/K2-Horizon-0.9B)、[K2-Horizon-3.7B](https://huggingface.co/IFM/K2-Horizon-3.7B)、[K2-Horizon-7B](https://huggingface.co/IFM/K2-Horizon-7B)（预训练、中期训练、各 RL 专家的 checkpoint 都放在分支里，例如 `pretrain_600000`）
  - 数据：[TxT360-v2](https://huggingface.co/datasets/IFM/TxT360-v2)（CC-BY-4.0），另有 Code-Reasoning、Math-Reasoning、SFT-Reasoning、Pretrain-Behaviors 四个配套数据集
  - 代码 / 配方：[xLLM](https://github.com/ifm-ai/xllm)（Apache-2.0，长上下文训练框架）；模型卡列出了每个阶段的步数、token 数和序列长度，并附 W&B 训练日志
  - 技术报告：模型卡标注"撰写中"，截至 2026-10-01 未发布；已公开的数据是否覆盖全部 22.9T / 5T token 的配比**待核实**
  - 许可证：模型 Apache-2.0；数据 CC-BY-4.0
  - 课程参考：各阶段 token 预算一览表、RL 专家合并后再做在线策略蒸馏 → 第 15/17/19 章；0.9B 版本和主线模型尺寸接近
- **YuLan-Mini（人大 RUC-GSAI）**（2.4B，2024-12；1.08T token）：中英双语，数学和代码较强
  - 模型：[YuLan-Mini](https://huggingface.co/yulan-team/YuLan-Mini)
  - 数据：[YuLan-Mini-Datasets](https://huggingface.co/datasets/yulan-team/YuLan-Mini-Datasets)（含各阶段数据配比）
  - 代码 / 配方：[YuLan-Mini/pretrain](https://github.com/RUC-GSAI/YuLan-Mini/tree/main/pretrain)（`train.py`、各阶段启动脚本、数据预处理与合成代码）
  - 技术报告：[YuLan-Mini: An Open Data-efficient Language Model](https://arxiv.org/abs/2412.17743)
  - 许可证：模型 MIT；数据按各来源
  - 课程参考：只用 1T token 做出强 2B 模型的数据效率配方、训练稳定性处理和退火 → 第 6/13/15 章；中文数据 → 主线模型
- **Instella（AMD）**（3B，2025-03；另有 Long-Instruct、Math 版本）
  - 模型：[Instella-3B](https://huggingface.co/amd/Instella-3B)（另有 Stage1、SFT、Instruct checkpoint）
  - 数据：第一阶段 4.07T token 用 [OLMoE-mix-0924](https://huggingface.co/datasets/allenai/OLMoE-mix-0924)；第二阶段用 dolmino-mix-1124 等公开数据加 AMD 自己合成的 [Instella-GSM8K-synthetic](https://huggingface.co/datasets/amd/Instella-GSM8K-synthetic)
  - 代码 / 配方：[AMD-AIG-AIMA/Instella](https://github.com/AMD-AIG-AIMA/Instella)（含数据准备脚本和配置）
  - 技术报告：[Instella: Fully Open Language Models with Stellar Performance](https://arxiv.org/abs/2511.10628)
  - 许可证：**RESEARCH-ONLY RAIL-MS，仅限研究用途**
  - 课程参考：完全用他人公开数据（Ai2 的数据）在非 NVIDIA 硬件上复现 3B 模型的案例 → 第 14 章
- **Ettin 解码器（JHU）**（17M / 32M / 68M / 150M / 400M / 1B，2025-07；与同配方的编码器成对发布）
  - 模型：如 [ettin-decoder-400m](https://huggingface.co/jhu-clsp/ettin-decoder-400m)、[ettin-decoder-1b](https://huggingface.co/jhu-clsp/ettin-decoder-1b)
  - 数据：[ettin-pretraining-data](https://huggingface.co/datasets/jhu-clsp/ettin-pretraining-data)（另有 extension、decay 阶段数据，以及逐 batch 的训练顺序 [ettin-data-order](https://huggingface.co/datasets/jhu-clsp/ettin-data-order)）
  - 代码 / 配方：[ettin-encoder-vs-decoder](https://github.com/JHU-CLSP/ettin-encoder-vs-decoder)
  - 技术报告：[Seq vs Seq: An Open Suite of Paired Encoders and Decoders](https://arxiv.org/abs/2507.11412)
  - 许可证：MIT
  - 课程参考：同数据、同配方下 6 个尺寸，适合单卡上做 scaling 小实验 → 第 12 章
- **TinyLlama**（1.1B，2023–2024；3T token）
  - 模型：[TinyLlama-1.1B-intermediate-step-1431k-3T](https://huggingface.co/TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T)、[TinyLlama_v1.1](https://huggingface.co/TinyLlama/TinyLlama_v1.1)
  - 数据：SlimPajama（官方仓库 cerebras/SlimPajama-627B 已不可访问，现只有社区重传版 [SlimPajama-627B_Reupload](https://huggingface.co/datasets/gmongaras/SlimPajama-627B_Reupload)）+ [starcoderdata](https://huggingface.co/datasets/bigcode/starcoderdata)（需同意条款）
  - 代码 / 配方：[jzhang38/TinyLlama](https://github.com/jzhang38/TinyLlama)（基于 lit-gpt，含 PRETRAIN.md）
  - 技术报告：[TinyLlama: An Open-Source Small Language Model](https://arxiv.org/abs/2401.02385)
  - 许可证：Apache-2.0
  - 课程参考：1B 模型训到 3T token 的过训练实例、单机多卡的吞吐优化 → 第 12/14 章
- **Pythia（EleutherAI）**（14M–12B，其中 ≤5B 的有 14M / 31M / 70M / 160M / 410M / 1B / 1.4B / 2.8B，每个尺寸有标准版和去重版；2023）
  - 模型：[pythia-1b](https://huggingface.co/EleutherAI/pythia-1b)、[pythia-1b-deduped](https://huggingface.co/EleutherAI/pythia-1b-deduped) 等，每个模型有 154 个 checkpoint 分支（`step0` … `step143000`）
  - 数据：原版 Pile 已因版权问题下架；现在还能拿到按训练顺序排好的分词数据 [pile-deduped-pythia-preshuffled](https://huggingface.co/datasets/EleutherAI/pile-deduped-pythia-preshuffled)、[pile-standard-pythia-preshuffled](https://huggingface.co/datasets/EleutherAI/pile-standard-pythia-preshuffled)，以及原文 [the_pile_deduplicated](https://huggingface.co/datasets/EleutherAI/the_pile_deduplicated)（仍包含 Books3 等有争议的部分）
  - 代码 / 配方：[pythia](https://github.com/EleutherAI/pythia)（各尺寸配置）、[gpt-neox](https://github.com/EleutherAI/gpt-neox)
  - 技术报告：[Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling](https://arxiv.org/abs/2304.01373)
  - 许可证：模型 Apache-2.0；数据按各来源，许可证不统一
  - 课程参考：同一份数据、同一个顺序下 8 个尺寸的完整训练轨迹，最适合做 scaling 和训练动态的课堂实验 → 第 12 章
- **DCLM-Baseline 模型**（1.4B / 6.9B，2024）
  - 模型：[TRI-ML/DCLM-1B](https://huggingface.co/TRI-ML/DCLM-1B)（4.3T token，Apache-2.0）、[apple/DCLM-7B](https://huggingface.co/apple/DCLM-7B)（2.5T token，Apple Sample Code License）
  - 数据：[dclm-baseline-1.0](https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0)（CC-BY-4.0），另混入 StarCoderData 和 [proof-pile-2](https://huggingface.co/datasets/EleutherAI/proof-pile-2)
  - 代码 / 配方：[mlfoundations/dclm](https://github.com/mlfoundations/dclm)（MIT，基于 OpenLM；有各尺度配置，但没有专门标注 7B 正式训练的那一份）
  - 技术报告：[DataComp-LM](https://arxiv.org/abs/2406.11794)
  - 课程参考：固定模型和训练配方、只比较数据过滤方法的"数据竞赛"设计，fastText 质量分类器 → 第 13 章
- **Luciole（OpenLLM-France / LINAGORA）**（1B / 8B，2026-02；约 5T token，法语占 30%）
  - 模型：[Luciole-1B-Base](https://huggingface.co/OpenLLM-France/Luciole-1B-Base)、[Luciole-8B-Base](https://huggingface.co/OpenLLM-France/Luciole-8B-Base)（8B 为 Mamba-注意力混合结构）
  - 数据：[Luciole-Training-Dataset](https://huggingface.co/datasets/OpenLLM-France/Luciole-Training-Dataset)（约 4.65T token，CC-BY-SA-4.0）
  - 代码 / 配方：[Luciole-Training](https://github.com/OpenLLM-France/Luciole-Training)（GPL-3.0，数据处理、分词器、NeMo 预训练、SFT、评测）
  - 技术报告：模型卡标注"coming soon"；前代 Lucie-7B 见 [arXiv 2503.12294](https://arxiv.org/abs/2503.12294)（数据 [Lucie-Training-Dataset](https://huggingface.co/datasets/OpenLLM-France/Lucie-Training-Dataset) 为 CC-BY-NC-SA）
  - 许可证：模型 Apache-2.0；数据 CC-BY-SA-4.0
  - 课程参考：非英语语种占大比例时的配比与分词器设计，可对照中文场景 → 第 7/13 章
- **Marin（Stanford CRFM / marin-community）**（8B，2025-05；12.7T token）
  - 模型：[marin-8b-base](https://huggingface.co/marin-community/marin-8b-base)、[marin-8b-instruct](https://huggingface.co/marin-community/marin-8b-instruct)
  - 数据：全部来自公开数据集（DCLM-Baseline、StarCoderData、ProofPile2、Nemotron-CC、Dolma、dolmino-mix-1124、FineMath 等），各阶段的精确配比写在 [exp600_tootsie.py](https://github.com/marin-community/marin/blob/ee163702c5bc71c9bbba3238db84b6ee86e826a7/experiments/tootsie/exp600_tootsie.py) 里；数据许可证按各来源
  - 代码 / 配方：[marin](https://github.com/marin-community/marin)（Apache-2.0，从数据处理到训练的每个实验都是可执行脚本）、训练框架 [levanter](https://github.com/stanford-crfm/levanter)；[Marin 8B 复盘](https://marin.readthedocs.io/en/latest/reports/marin-8b-retro/) 如实记录了各阶段为何调整以及中途的 bug（RoPE 设置错误等）
  - 许可证：模型 Apache-2.0
  - 课程参考：训练中途根据观测改配比、改学习率的真实决策记录 → 第 12/14 章；同一项目的 Delphi scaling 套件（157M–25B 的 IsoFLOP checkpoint，如 [delphi-1e21-3.4Bparams-46.3Btokens](https://huggingface.co/marin-community/delphi-1e21-3.4Bparams-46.3Btokens)）就是开头通用参考里那篇 Delphi 博客的模型 → 第 12 章
- **Apertus（Swiss AI）**（8B，2025-09；70B 不在范围内）
  - 模型：[Apertus-8B-2509](https://huggingface.co/swiss-ai/Apertus-8B-2509)、[Apertus-8B-Instruct-2509](https://huggingface.co/swiss-ai/Apertus-8B-Instruct-2509)（下载前需点击同意使用政策）
  - 数据：**不直接发布语料，而是发布可确定性重建的流水线**：[pretrain-data](https://github.com/swiss-ai/pretrain-data)（从 FineWeb-Edu、FineWeb-2、DCLM-Edu、FineMath、MegaMath、StarCoder 等公开数据重建）+ 按 2025-01 robots.txt 回溯剔除的域名表（如 [robots-txt-blocked-domains-english](https://huggingface.co/datasets/swiss-ai/robots-txt-blocked-domains-english)）
  - 代码 / 配方：[pretrain-code](https://github.com/swiss-ai/pretrain-code)
  - 技术报告：[Apertus: Democratizing Open and Compliant LLMs for Global Language Environments](https://arxiv.org/abs/2509.14233)
  - 许可证：模型 Apache-2.0；数据按各来源
  - 课程参考：数据合规（按 robots.txt 回溯剔除退出的网站）的做法，以及合规过滤损失多少 token（英文约 8%）→ 第 13 章
- **MAP-Neo（M-A-P）**（7.8B，另有 2B 和 250M / 460M / 980M 的 scaling law 小模型；2024）：中英双语
  - 模型：[neo_7b](https://huggingface.co/m-a-p/neo_7b)（另有 intermediate 和 decay 阶段 checkpoint）、[neo_2b_general](https://huggingface.co/m-a-p/neo_2b_general)
  - 数据：[Matrix](https://huggingface.co/datasets/m-a-p/Matrix)（4.69T token，中英双语）
  - 代码 / 配方：[MAP-NEO](https://github.com/multimodal-art-projection/MAP-NEO)（MIT，基于 Megatron，含数据处理流水线）
  - 技术报告：[MAP-Neo: Highly Capable and Transparent Bilingual Large Language Model Series](https://arxiv.org/abs/2405.19327)
  - 许可证：模型 Apache-2.0；数据集卡标 Apache-2.0
  - 课程参考：少有的全开源中文预训练数据和清洗流水线 → 第 13 章；scaling law 小模型 → 第 12 章
- **Amber / Crystal（LLM360，HF 组织已改名为 IFM）**（6.7B / 7B，2023-12）
  - 模型：[Amber](https://huggingface.co/IFM/Amber)（1.26T token，360 个 checkpoint）、[Crystal](https://huggingface.co/IFM/Crystal)（约 1.4T token，SlimPajama + StarCoderData）
  - 数据：[AmberDatasets](https://huggingface.co/datasets/IFM/AmberDatasets)、[CrystalCoderDatasets](https://huggingface.co/datasets/IFM/CrystalCoderDatasets)（都是按训练顺序排好的分词后数据，ODC-BY）
  - 代码 / 配方：[amber-train](https://github.com/LLM360/amber-train)、[amber-data-prep](https://github.com/LLM360/amber-data-prep)、[crystalcoder-train](https://github.com/LLM360/crystalcoder-train)（基于 Cerebras 硬件，换平台不易复现）、分析工具 [Analysis360](https://github.com/LLM360/Analysis360)
  - 技术报告：[LLM360: Towards Fully Transparent Open-Source LLMs](https://arxiv.org/abs/2312.06550)
  - 许可证：模型 Apache-2.0；数据 ODC-BY
  - 课程参考："全透明发布"应该包含哪些东西（checkpoint、数据顺序、日志、分析代码）→ 第 20 章
- **Comma v0.1（EleutherAI 等，Common Pile）**（7B，2025）：只用公开许可文本训练
  - 模型：[comma-v0.1-1t](https://huggingface.co/common-pile/comma-v0.1-1t)、[comma-v0.1-2t](https://huggingface.co/common-pile/comma-v0.1-2t)
  - 数据：[comma_v0.1_training_dataset](https://huggingface.co/datasets/common-pile/comma_v0.1_training_dataset)（每个来源都是开放许可，各自保留原许可证）；数据处理代码 [r-three/common-pile](https://github.com/r-three/common-pile)
  - 代码 / 配方：用 [Meta Lingua](https://github.com/facebookresearch/lingua) 训练，2T 版的配置文件和训练指标在 [comma-v0.1-2t-checkpoints](https://huggingface.co/common-pile/comma-v0.1-2t-checkpoints)
  - 技术报告：[The Common Pile v0.1](https://arxiv.org/abs/2506.05209)
  - 许可证：模型 Apache-2.0
  - 课程参考：只用开放许可数据能做到什么水平，数据许可证怎么逐源核对 → 第 13 章

### 部分开源（不进主表，写明缺什么）
- **MiniCPM4**（[0.5B](https://huggingface.co/openbmb/MiniCPM4-0.5B) / [8B](https://huggingface.co/openbmb/MiniCPM4-8B)，Apache-2.0）：预训练框架 [ForgeTrain](https://github.com/OpenBMB/ForgeTrain) 和 Ultra-FineWeb 已公开，但其余预训练数据和 SFT 数据（UltraChat v2）没有完整发布
- **Nemotron Nano 2**（[NVIDIA-Nemotron-Nano-9B-v2](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-9B-v2)，由 12B-Base 剪枝蒸馏而来；[arXiv 2508.14444](https://arxiv.org/abs/2508.14444)；NVIDIA Open Model License）：公开了"大部分"预训练数据（约 6.6T token，见上方 NVIDIA 一节），但需同意 NVIDIA 数据协议。GitHub 代码只给元数据，多语言网页抓取数据和两份第三方私有数据未公开。只有 Megatron-LM / NeMo-RL 框架，没有端到端复现配方
- **Nemotron 3 Nano 4B**（[NVIDIA-Nemotron-3-Nano-4B-BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16)，2026-03；NVIDIA Nemotron Open Model License）：用 [Nemotron Elastic](https://arxiv.org/abs/2511.16664) 从 Nano-9B-v2 压缩而来，继承上面的数据缺口；蒸馏所用数据子集未写明，Elastic 路由代码未公开
- **Apertus v1.1**（[0.5B](https://huggingface.co/swiss-ai/Apertus-v1.1-0.5B) / [1.5B](https://huggingface.co/swiss-ai/Apertus-v1.1-1.5B) / [4B](https://huggingface.co/swiss-ai/Apertus-v1.1-4B)，2026-02 至 04；Apache-2.0）：在 8B 的第 5 阶段数据上用 1.7T token 做预训练蒸馏（[arXiv 2605.29128](https://arxiv.org/abs/2605.29128)），但模型卡给的蒸馏代码仓库（swiss-ai/Megatron-LM-Distill）在 2026-10-01 无法访问
- **Tülu 3 8B**（[Llama-3.1-Tulu-3-8B](https://huggingface.co/allenai/Llama-3.1-Tulu-3-8B)，Llama 3.1 许可证）：后训练全开源（[tulu-3-sft-mixture](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture)、[open-instruct](https://github.com/allenai/open-instruct)、[Tülu 3 报告](https://arxiv.org/abs/2411.15124)），但底座是 Meta 的 Llama 3.1，预训练数据不公开；作为 SFT → DPO → RLVR 配方参考 → 第 16/18/19 章
- **StarCoder2**（[3B](https://huggingface.co/bigcode/starcoder2-3b) / 7B / 15B，BigCode OpenRAIL-M；[arXiv 2402.19173](https://arxiv.org/abs/2402.19173)）：[The Stack v2](https://huggingface.co/datasets/bigcode/the-stack-v2) 需申请且只含 Software Heritage ID，正文需另行批量下载；GitHub 仓库只有微调和推理代码，没有预训练配方
- **OpenCoder**（[OpenCoder-1.5B-Base](https://huggingface.co/infly/OpenCoder-1.5B-Base) / 8B；[arXiv 2411.04905](https://arxiv.org/abs/2411.04905)）：代码预训练语料 RefineCode 只发布了元数据（[RefineCode-code-corpus-meta](https://huggingface.co/datasets/OpenCoder-LLM/RefineCode-code-corpus-meta)），模型许可证为自定义
- **Zamba2**（[1.2B](https://huggingface.co/Zyphra/Zamba2-1.2B) / 2.7B / 7B，Zyphra，Apache-2.0）：第一阶段数据 [Zyda-2](https://huggingface.co/datasets/Zyphra/Zyda-2)（ODC-BY）公开，约 50B token 的退火数据和训练代码未公开；2026 年的 ZAYA1-8B 只给了数据类别占比
- **AMD-OLMo-1B**（[AMD-OLMo-1B](https://huggingface.co/amd/AMD-OLMo-1B)，Apache-2.0）：用 Dolma v1.7 的 1.3T token 子集训练，但子集怎么选的没有查到
- **Salamandra**（[2B](https://huggingface.co/BSC-LT/salamandra-2b) / 7B，BSC）：模型卡写明语料"不会发布"，只公开来源列表和训练脚本
- **EuroLLM**（[1.7B](https://huggingface.co/utter-project/EuroLLM-1.7B) / 9B，Apache-2.0）：部分数据（EuroWeb）和 Megatron 分支已公开，合成数学数据、翻译版 Cosmopedia 是否发布未能确认
- **Baguettotron / Monad**（[Baguettotron](https://huggingface.co/PleIAs/Baguettotron) 321M、Monad 56M，Pleias，2025-11，Apache-2.0）：只用全合成数据 [SYNTH](https://huggingface.co/datasets/PleIAs/SYNTH)（CC-BY-4.0）训练，但没有找到训练代码；全合成预训练的例子 → 第 13 章
- **OpenLLaMA / RedPajama-INCITE**（[open_llama_3b_v2](https://huggingface.co/openlm-research/open_llama_3b_v2)、[RedPajama-INCITE-Base-3B-v1](https://huggingface.co/togethercomputer/RedPajama-INCITE-Base-3B-v1)，2023，Apache-2.0）：数据（RedPajama 等）公开，但没有发布精确的训练配置或训练代码
- 另：Cerebras-GPT、BTLM 的官方权重和 SlimPajama 的官方仓库（cerebras/SlimPajama-627B）已无法访问，不再收录

## 课程写作中新增的参考资料（按章节）

以下资料在编写各章时查阅并引用，完整出处见各章 README 的参考文献。

### 第 2 章
- NumPy Broadcasting: https://numpy.org/doc/stable/user/basics.broadcasting.html
- NumPy "What is NumPy? / Why is NumPy fast?": https://numpy.org/doc/stable/user/whatisnumpy.html
- PyTorch torch.nn.Linear: https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html
- 3Blue1Brown, Essence of Linear Algebra: https://www.3blue1brown.com/topics/linear-algebra
- Deep Learning book, ch. 2 Linear Algebra: https://www.deeplearningbook.org/contents/linear_algebra.html
- Dive into Deep Learning §3.1 Linear Regression: https://d2l.ai/chapter_linear-regression/linear-regression.html

### 第 3 章
- Nielsen, Neural Networks and Deep Learning ch4（万能近似的可视化证明）: http://neuralnetworksanddeeplearning.com/chap4.html
- Goodfellow et al., Deep Learning ch6: https://www.deeplearningbook.org/contents/mlp.html
- 3Blue1Brown, Neural networks: https://www.3blue1brown.com/lessons/neural-networks
- Shazeer, GLU Variants Improve Transformer (SwiGLU): https://arxiv.org/abs/2002.05202

### 第 4 章
- micrograd（MIT）: https://github.com/karpathy/micrograd
- Karpathy, The spelled-out intro to neural networks and backpropagation: https://www.youtube.com/watch?v=VMj-3S1tku0
- CS231n backprop notes: https://cs231n.github.io/optimization-2/
- Baydin et al., Automatic Differentiation in Machine Learning: a Survey: https://arxiv.org/abs/1502.05767
- PyTorch Autograd mechanics: https://docs.pytorch.org/docs/stable/notes/autograd.html

### 第 5 章
- Hinton, Vinyals, Dean, Distilling the Knowledge in a Neural Network（softmax 温度）: https://arxiv.org/abs/1503.02531
- Szegedy et al., Rethinking the Inception Architecture（label smoothing）: https://arxiv.org/abs/1512.00567
- CS231n neural network case study（螺旋数据分类）: https://cs231n.github.io/neural-networks-case-study/

### 第 6 章
- MiniCPM（WSD 学习率调度）: https://arxiv.org/abs/2404.06395
- Hägele et al., Scaling Laws and Compute-Optimal Training Beyond Fixed Training Durations（恒定学习率 + 冷却）: https://arxiv.org/abs/2405.18392
- OLMo 2: https://arxiv.org/abs/2501.00656
- SmolLM3 博客: https://huggingface.co/blog/smollm3
- Xiong et al., On Layer Normalization in the Transformer Architecture（Pre-LN）: https://arxiv.org/abs/2002.04745
- Loshchilov & Hutter, Decoupled Weight Decay Regularization（AdamW）: https://arxiv.org/abs/1711.05101

### 第 7 章
- Sennrich et al., Neural Machine Translation of Rare Words with Subword Units（BPE）: https://arxiv.org/abs/1508.07909
- Llama 3 Herd of Models: https://arxiv.org/abs/2407.21783
- Karpathy minbpe: https://github.com/karpathy/minbpe
- CS336 Assignment 1 Basics: https://github.com/stanford-cs336/assignment1-basics

### 第 8 章
- Vaswani et al., Attention Is All You Need: https://arxiv.org/abs/1706.03762
- Bahdanau et al., Neural Machine Translation by Jointly Learning to Align and Translate: https://arxiv.org/abs/1409.0473
- Ainslie et al., GQA: https://arxiv.org/abs/2305.13245
- Dao et al., FlashAttention: https://arxiv.org/abs/2205.14135
- Gemma 3 Technical Report: https://arxiv.org/abs/2503.19786
- gpt-oss-120b & gpt-oss-20b Model Card: https://arxiv.org/abs/2508.10925
- Karpathy, ng-video-lecture（Let's build GPT）: https://github.com/karpathy/ng-video-lecture

### 第 9 章
- Su et al., RoFormer（RoPE）: https://arxiv.org/abs/2104.09864
- Dehghani et al., Scaling Vision Transformers to 22B（QK-Norm）: https://arxiv.org/abs/2302.05442
- Wortsman et al., Small-scale proxies for large-scale Transformer training instabilities: https://arxiv.org/abs/2309.14322
- Press & Wolf, Using the Output Embedding to Improve Language Models: https://arxiv.org/abs/1608.05859
- Qwen3 Technical Report: https://arxiv.org/abs/2505.09388
- DeepSeek-V3 Technical Report: https://arxiv.org/abs/2412.19437

### 第 10 章
- Holtzman et al., The Curious Case of Neural Text Degeneration（top-p）: https://arxiv.org/abs/1904.09751
- Shazeer, Fast Transformer Decoding: One Write-Head is All You Need（MQA）: https://arxiv.org/abs/1911.02150
- Kwon et al., Efficient Memory Management for LLM Serving with PagedAttention（vLLM）: https://arxiv.org/abs/2309.06180
- Orca（continuous batching, OSDI'22）: https://www.usenix.org/conference/osdi22/presentation/yu

### 第 11 章
- MMLU-Redux: https://arxiv.org/abs/2406.04127 ; MMLU-Pro: https://arxiv.org/abs/2406.01574
- C-Eval: https://arxiv.org/abs/2305.08322 ; CMMLU: https://arxiv.org/abs/2306.09212
- EvalPlus（HumanEval+/MBPP+）: https://arxiv.org/abs/2305.01210 ; IFEval: https://arxiv.org/abs/2311.07911
- BFCL: https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard
- ACEBench: https://arxiv.org/abs/2501.12851 （https://github.com/chenchen0103/ACEBench）
- τ²-bench: https://arxiv.org/abs/2506.07982
- Miller, Adding Error Bars to Evals: https://arxiv.org/abs/2411.00640
- Sclar et al., Quantifying Language Models' Sensitivity to Spurious Features in Prompt Design: https://arxiv.org/abs/2310.11324

### 第 12 章
- Besiroglu et al., Chinchilla Scaling: A replication attempt: https://arxiv.org/abs/2404.10102
- Porian et al., Resolving Discrepancies in Compute-Optimal Scaling: https://arxiv.org/abs/2406.19146
- Sardana & Frankle, Beyond Chinchilla-Optimal（推理成本与过训练）: https://arxiv.org/abs/2401.00448
- Moonlight（Muon 规模化）: https://arxiv.org/abs/2502.16982
- GLM-4.5: https://arxiv.org/abs/2508.06471
- Keller Jordan, Muon: https://kellerjordan.github.io/posts/muon/

### 第 13 章
- FineWeb: https://arxiv.org/abs/2406.17557 ; FineWeb2: https://arxiv.org/abs/2506.20920
- DCLM: https://arxiv.org/abs/2406.11794
- Nemotron-CC: https://arxiv.org/abs/2412.02595
- Ultra-FineWeb: https://arxiv.org/abs/2505.05427
- Tao et al., Scaling Laws with Vocabulary: https://arxiv.org/abs/2407.13623
- Lee et al., Deduplicating Training Data Makes Language Models Better: https://arxiv.org/abs/2107.06499
- CS336 Assignment 4 Data: https://github.com/stanford-cs336/assignment4-data

### 第 14 章
- Micikevicius et al., Mixed Precision Training: https://arxiv.org/abs/1710.03740
- Milakov & Gimelshein, Online normalizer calculation for softmax: https://arxiv.org/abs/1805.02867
- FlashAttention-2: https://arxiv.org/abs/2307.08691 ; FlashAttention-3: https://arxiv.org/abs/2407.08608
- Chen et al., Training Deep Nets with Sublinear Memory Cost（激活检查点）: https://arxiv.org/abs/1604.06174
- ZeRO: https://arxiv.org/abs/1910.02054 ; PyTorch FSDP: https://arxiv.org/abs/2304.11277
- PaLM（MFU、loss spike、z-loss）: https://arxiv.org/abs/2204.02311
- OLMo-core: https://github.com/allenai/OLMo-core

### 第 15 章
- YaRN: https://arxiv.org/abs/2309.00071 ; Position Interpolation: https://arxiv.org/abs/2306.15595
- Xiong et al., Effective Long-Context Scaling（ABF）: https://arxiv.org/abs/2309.16039
- RULER: https://arxiv.org/abs/2404.06654 ; Needle in a Haystack: https://github.com/gkamradt/LLMTest_NeedleInAHaystack
- Blakeney et al., Does your data spark joy?（退火阶段的数据）: https://arxiv.org/abs/2406.03476

### 第 16 章
- Touvron et al. *Llama 2: Open Foundation and Fine-Tuned Chat Models*，2023：https://arxiv.org/abs/2307.09288
- Lambert et al. *Tülu 3: Pushing Frontiers in Open Language Model Post-Training*，2024：https://arxiv.org/abs/2411.15124；训练代码 open-instruct：https://github.com/allenai/open-instruct
- Allal et al. *SmolLM2: When Smol Goes Big — Data-Centric Training of a Small Language Model*，2025：https://arxiv.org/abs/2502.02737
- Zhou et al. *LIMA: Less Is More for Alignment*，2023：https://arxiv.org/abs/2305.11206
- Ouyang et al. *Training language models to follow instructions with human feedback*（InstructGPT），2022：https://arxiv.org/abs/2203.02155
- Teknium et al. *Hermes 3 Technical Report*，2024：https://arxiv.org/abs/2408.11857
- Liu et al. *APIGen: Automated Pipeline for Generating Verifiable and Diverse Function-Calling Datasets*，2024：https://arxiv.org/abs/2406.18518
- Liu et al. *ToolACE: Winning the Points of LLM Function Calling*，2024：https://arxiv.org/abs/2409.00920
- Hu et al. *LoRA: Low-Rank Adaptation of Large Language Models*，2021：https://arxiv.org/abs/2106.09685
- OpenAI. ChatML 说明（openai-python v0.28）：https://github.com/openai/openai-python/blob/release-v0.28.0/chatml.md
- Hugging Face TRL 文档：SFT Trainer: https://huggingface.co/docs/trl/sft_trainer、Reducing Memory Usage（打包）: https://huggingface.co/docs/trl/reducing_memory_usage
- 数据卡（2026-09 读取）：tulu-3-sft-mixture: https://huggingface.co/datasets/allenai/tulu-3-sft-mixture、smoltalk: https://huggingface.co/datasets/HuggingFaceTB/smoltalk、smoltalk2: https://huggingface.co/datasets/HuggingFaceTB/smoltalk2、xlam-function-calling-60k: https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k、ToolACE: https://huggingface.co/datasets/Team-ACE/ToolACE、hermes-function-calling-v1: https://huggingface.co/datasets/NousResearch/hermes-function-calling-v1
- 对话模板（2026-09 读取）：Qwen3-0.6B: https://huggingface.co/Qwen/Qwen3-0.6B/blob/main/tokenizer_config.json、Qwen3.5-0.8B: https://huggingface.co/Qwen/Qwen3.5-0.8B/blob/main/chat_template.jinja、SmolLM3-3B: https://huggingface.co/HuggingFaceTB/SmolLM3-3B/blob/main/chat_template.jinja、Hermes-3-Llama-3.1-8B: https://huggingface.co/NousResearch/Hermes-3-Llama-3.1-8B、Olmo-3-7B-Instruct: https://huggingface.co/allenai/Olmo-3-7B-Instruct/blob/main/chat_template.jinja
- nanochat: https://github.com/karpathy/nanochat 的 `scripts/chat_sft.py`；minimind: https://github.com/jingyaogong/minimind 的 SFT 部分（中文小模型的完整 SFT 流程）
- CS336 作业 5：https://github.com/stanford-cs336/assignment5-alignment

### 第 17 章
- Kim & Rush, Sequence-Level Knowledge Distillation: https://arxiv.org/abs/1606.07947
- Agarwal et al., On-Policy Distillation of Language Models（GKD）: https://arxiv.org/abs/2306.13649
- Thinking Machines, On-Policy Distillation: https://thinkingmachines.ai/blog/on-policy-distillation
- Gemma 2: https://arxiv.org/abs/2408.00118
- Minitron（剪枝 + 蒸馏）: https://arxiv.org/abs/2407.14679

### 第 18 章
- Rafailov et al. *Direct Preference Optimization: Your Language Model is Secretly a Reward Model*，2023：https://arxiv.org/abs/2305.18290
- Stiennon et al. *Learning to summarize from human feedback*，2020：https://arxiv.org/abs/2009.01325
- Schulman et al. *Proximal Policy Optimization Algorithms*，2017：https://arxiv.org/abs/1707.06347
- Gao, Schulman, Hilton. *Scaling Laws for Reward Model Overoptimization*，2022：https://arxiv.org/abs/2210.10760
- Qwen Team. *Qwen2 Technical Report*（§4.3），2024：https://arxiv.org/abs/2407.10671；*Qwen2.5 Technical Report*（§4.2–4.3），2024：https://arxiv.org/abs/2412.15115
- DeepSeek-AI. *DeepSeek LLM: Scaling Open-Source Language Models with Longtermism*（§4），2024：https://arxiv.org/abs/2401.02954
- NVIDIA. *Nemotron-4 340B Technical Report*（§3.3.2 DPO 与 RPO），2024：https://arxiv.org/abs/2406.11704
- Tunstall et al. *Zephyr: Direct Distillation of LM Alignment*，2023：https://arxiv.org/abs/2310.16944
- Hugging Face. *SmolLM3: smol, multilingual, long-context reasoner*（博客）：https://github.com/huggingface/blog/blob/main/smollm3.md
- D'Oosterlinck et al. *Anchored Preference Optimization and Contrastive Revisions*（APO），2024：https://arxiv.org/abs/2408.06266
- Razin et al. *Unintentional Unalignment: Likelihood Displacement in Direct Preference Optimization*，2024：https://arxiv.org/abs/2410.08847
- Pal et al. *Smaug: Fixing Failure Modes of Preference Optimisation with DPO-Positive*，2024：https://arxiv.org/abs/2402.13228
- Park et al. *Disentangling Length from Quality in Direct Preference Optimization*，2024：https://arxiv.org/abs/2403.19159
- Singhal et al. *A Long Way to Go: Investigating Length Correlations in RLHF*，2023：https://arxiv.org/abs/2310.03716
- 变体（前沿观察）：IPO https://arxiv.org/abs/2310.12036；KTO https://arxiv.org/abs/2402.01306；SimPO https://arxiv.org/abs/2405.14734；ORPO https://arxiv.org/abs/2403.07691
- 偏好数据集：UltraFeedback: https://huggingface.co/datasets/openbmb/UltraFeedback（MIT）、HelpSteer3: https://huggingface.co/datasets/nvidia/HelpSteer3（CC-BY-4.0）、Tülu 3 8B 偏好混合: https://huggingface.co/datasets/allenai/llama-3.1-tulu-3-8b-preference-mixture（ODC-BY-1.0，部分子集不可商用）

### 第 19 章
- Williams. *Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning* (REINFORCE), 1992：https://link.springer.com/article/10.1007/BF00992696
- Shao et al. *DeepSeekMath*（GRPO、k3 KL、统一的梯度系数视角）, 2024：https://arxiv.org/abs/2402.03300
- DeepSeek-AI. *DeepSeek-R1*, 2025：https://arxiv.org/abs/2501.12948
- Qwen Team. *Qwen3 Technical Report*, 2025：https://arxiv.org/abs/2505.09388；*Group Sequence Policy Optimization*：https://arxiv.org/abs/2507.18071
- Olmo Team. *Olmo 3*, 2025：https://arxiv.org/abs/2512.13961
- Yu et al. *DAPO*, 2025：https://arxiv.org/abs/2503.14476
- Liu et al. *Understanding R1-Zero-Like Training: A Critical Perspective*（Dr. GRPO）, 2025：https://arxiv.org/abs/2503.20783
- Schulman. *Approximating KL Divergence*：http://joschu.net/blog/kl-approx.html
- Lilian Weng. *Reward Hacking in Reinforcement Learning*, 2024：https://lilianweng.github.io/posts/2024-11-28-reward-hacking/
- verl（HybridFlow）：https://github.com/verl-project/verl

### 第 20 章
- GGUF 规范: https://github.com/ggml-org/ggml/blob/master/docs/gguf.md
- llama.cpp k-quants（PR #1684）: https://github.com/ggml-org/llama.cpp/pull/1684
- Mitchell et al., Model Cards for Model Reporting: https://arxiv.org/abs/1810.03993
- Dettmers et al., LLM.int8(): https://arxiv.org/abs/2208.07339 ; Frantar et al., GPTQ: https://arxiv.org/abs/2210.17323

### 第 21 章
- DeepSeek-V2（MLA）: https://arxiv.org/abs/2405.04434
- Kimi K2 Technical Report: https://arxiv.org/abs/2507.20534
- GLM-5 Technical Report: https://arxiv.org/abs/2602.15763
- FlashMLA: https://github.com/deepseek-ai/FlashMLA

### 第 22 章
- Child, Gray, Radford, Sutskever. *Generating Long Sequences with Sparse Transformers*，2019：https://arxiv.org/abs/1904.10509
- Beltagy, Peters, Cohan. *Longformer: The Long-Document Transformer*（滑动窗口 + 全局 token），2020：https://arxiv.org/abs/2004.05150
- Jiang et al. *Mistral 7B*（滑动窗口、滚动缓冲区缓存、理论跨度），2023：https://arxiv.org/abs/2310.06825
- Xiao et al. *Efficient Streaming Language Models with Attention Sinks*（StreamingLLM），2023：https://arxiv.org/abs/2309.17453
- Yuan et al. *Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention*（NSA），2025：https://arxiv.org/abs/2502.11089
- Lu et al. *MoBA: Mixture of Block Attention for Long-Context LLMs*，2025：https://arxiv.org/abs/2502.13189
- DeepSeek-AI. *DeepSeek-V3.2*（DSA）技术报告：https://huggingface.co/deepseek-ai/DeepSeek-V3.2/blob/main/assets/paper.pdf
- DeepSeek-AI. *DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence*，2026：https://arxiv.org/abs/2606.19348
- Lai et al. *MiniMax Sparse Attention*，2026：https://arxiv.org/abs/2606.13392
- MiniCPM Team. *MiniCPM4*，2025：https://arxiv.org/abs/2506.07900；*InfLLM-V2*，2025：https://arxiv.org/abs/2509.24663
- Dao. FlashAttention（`window_size` 参数）：https://github.com/Dao-AILab/flash-attention；PyTorch FlexAttention 博客（滑动窗口 `mask_mod` 示例）：https://pytorch.org/blog/flexattention/

### 第 23 章
- Katharopoulos et al., Transformers are RNNs（线性注意力）: https://arxiv.org/abs/2006.16236
- Yang et al., Parallelizing Linear Transformers with the Delta Rule: https://arxiv.org/abs/2406.06484
- Gated Delta Networks: https://arxiv.org/abs/2412.06464
- Kimi Linear: https://arxiv.org/abs/2510.26692
- MiniMax-01: https://arxiv.org/abs/2501.08313
- Nemotron-H: https://arxiv.org/abs/2504.03624
- flash-linear-attention: https://github.com/fla-org/flash-linear-attention

### 第 24 章
- Shazeer et al. *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer*（MoE 层与路由坍缩问题），2017：https://arxiv.org/abs/1701.06538
- Lepikhin et al. *GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding*（top-2 路由、容量、辅助损失、专家并行），2020：https://arxiv.org/abs/2006.16668
- Fedus, Zoph, Shazeer. *Switch Transformers*（top-1 路由、负载均衡损失、容量因子），2021：https://arxiv.org/abs/2101.03961
- Gale et al. *MegaBlocks: Efficient Sparse Training with Mixture-of-Experts*（dropless、块稀疏矩阵乘），2022：https://arxiv.org/abs/2211.15841
- Jiang et al. *Mixtral of Experts*，2024：https://arxiv.org/abs/2401.04088
- Dai et al. *DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models*（细粒度专家、共享专家），2024：https://arxiv.org/abs/2401.06066
- Wang et al. *Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts*，2024：https://arxiv.org/abs/2408.15664
- NVIDIA. *Nemotron 3 Nano: Open, Efficient Mixture-of-Experts Hybrid Mamba-Transformer Model for Agentic Reasoning*（2.1、2.4 节），2025：https://arxiv.org/abs/2512.20848
- Muennighoff et al. *OLMoE: Open Mixture-of-Experts Language Models*，2024：https://arxiv.org/abs/2409.02060
- Hugging Face transformers 的 MoE 实现（`models/qwen3_moe`、`models/deepseek_v3`、`models/llama4`）：https://github.com/huggingface/transformers/tree/main/src/transformers/models
- DeepSeek. DeepEP（专家并行通信库）：https://github.com/deepseek-ai/DeepEP

### 第 25 章
- Leviathan et al., Fast Inference from Transformers via Speculative Decoding: https://arxiv.org/abs/2211.17192
- Chen et al., Accelerating LLM Decoding with Speculative Sampling: https://arxiv.org/abs/2302.01318
- Gloeckle et al., Better & Faster LLMs via Multi-token Prediction: https://arxiv.org/abs/2404.19737
- MiMo: https://arxiv.org/abs/2505.07608
- CS336 lectures 代码（Spring 2026）: https://github.com/stanford-cs336/lectures

### 第 26 章
- Kimi Team. *Kimi K3 Technical Report*：https://github.com/MoonshotAI/Kimi-K3/blob/main/k3_tech_report.pdf
- GLM-5 Team. *GLM-5: from Vibe Coding to Agentic Engineering*，2026：https://arxiv.org/abs/2602.15763；IndexShare：https://arxiv.org/abs/2603.12201
- Qwen Team. *Qwen3.8-Max* 博客：https://qwen.ai/blog?id=qwen3.8
- 演化树上各节点的出处：GPT-2（Radford 等 2019）；Llama https://arxiv.org/abs/2302.13971；RMSNorm https://arxiv.org/abs/1910.07467；RoPE https://arxiv.org/abs/2104.09864；YaRN https://arxiv.org/abs/2309.00071；GLU 变体 https://arxiv.org/abs/2002.05202；GQA https://arxiv.org/abs/2305.13245；QK-Norm https://arxiv.org/abs/2010.04245；共享 embedding https://arxiv.org/abs/1608.05859；DeepSeekMoE https://arxiv.org/abs/2401.06066；无辅助损失均衡 https://arxiv.org/abs/2408.15664；MLA（DeepSeek-V2）https://arxiv.org/abs/2405.04434；Mistral 7B https://arxiv.org/abs/2310.06825；Gemma 2 https://arxiv.org/abs/2408.00118；Gated DeltaNet https://arxiv.org/abs/2412.06464；MTP（DeepSeek-V3）https://arxiv.org/abs/2412.19437
