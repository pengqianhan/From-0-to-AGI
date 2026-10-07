# From 0 to AGI：从 y = ax + b 到最先进的开源模型

[English](README.md) · **中文**

> *You can outsource your thinking, but you cannot outsource your understanding.*

这门课从一条直线 `y = ax + b` 开始。之后讲到亲手搭建 Transformer、scaling law、预训练工程、SFT、蒸馏、DPO、GRPO 强化学习，最后讲为了更长的上下文、更小的 KV cache 而出现的最新架构。**课程只讲业界已经形成共识的主流技术**（判定规则见 [GOAL.zh.md](GOAL.zh.md) 第 2.1 节）。课程有英文和中文两个版本，每一页都有切换到另一种语言的按钮或链接。

每一章有三个部分：

- **文字速读版**：15–30 分钟读完。顺序是直觉 → 公式 → 极简代码 → 小结。
- **讲解视频**：5–10 分钟。视频里的每个数字都由本章代码真实算出。
- **两层代码**：第一层是**极简代码**，在 CPU 上几秒到几分钟跑完（`chapters/NN-*/code/`）。第二层是同一个想法在主线模型里的**生产级代码**（[`zero/`](zero/DESIGN.zh.md)）。两层代码互相对拍，保证结果一致。

课程的后半部分围绕一个**主线模型**。我们用约 1 万美元的算力，从零训练一个中英双语、约 0.69B 参数的小模型。目标是在**工具调用**上超过同尺寸的所有公开模型（包括 Qwen3.5-0.8B）。通用基准如实报告。生产级代码已经全部写好，并在 CPU 上用极小配置跑通。真实训练在第二步、有 GPU 之后进行（见 [runs/RUNBOOK.md](runs/RUNBOOK.md)）。

## 和斯坦福 CS336 的关系

这门课是 [CS336（Language Modeling from Scratch）](https://cs336.stanford.edu/) 的**入门与导读**。第 1–6 章讲 CS336 默认你已经掌握的深度学习基础。从第 7 章开始，每章末尾有"想深入：CS336"一节，指向 Spring 2026 对应的讲次和作业。这门课多出两条主线：真实训练并发布一个模型的全过程（预注册、闸门、公平评测），以及第五部分的架构演进。

## 怎么学

1. 读章节 README 的**目标**和正文。配合视频效果更好。视频还没有发布，请用下面的命令在本机渲染。
2. 运行本章 `code/` 里的极简代码，和正文里的数字对照。
3. 读"从极简代码到生产级代码"一节，打开 `zero/` 里对应的文件。
4. 带着"引导问题"去问 Claude Code，完成"动手任务"。
5. 在 Claude Code 里输入 `/chNN-...` 做自检（Skill 在 `.claude/commands/`）。

```bash
# 安装依赖（需要 uv：https://docs.astral.sh/uv/）
uv sync                       # 文字 + 代码
uv sync --extra video         # 还要自己渲染视频（另需系统依赖，见下面的"渲染视频前的准备"）

# 运行第 1 章
uv run python chapters/01-linear-regression/code/01_fit_line.py

# 渲染第 1 章的视频（加 --preview 渲染 480p 样片）
bash chapters/01-linear-regression/video/build.sh

# 生产级代码的测试与端到端冒烟测试（CPU）
uv run pytest
uv run python -m zero.smoke
```

**渲染视频前的准备**：

- 安装 `ffmpeg` 和 LaTeX（Manim 用 LaTeX 渲染公式）。
- 字体：安装一种中文字体（WenQuanYi Zen Hei、Noto Sans CJK SC、Source Han Sans SC 任一种）和等宽字体 Noto Sans Mono（用于代码块）。
- 第一次渲染时，程序从 GitHub 下载离线 TTS 模型（sherpa-onnx 的 MeloTTS 中英混读模型）到 `~/.cache/tts`。之后可以离线使用。要换目录，设置环境变量 `VIDEO_TTS_MODEL_DIR`。TTS 默认用满所有 CPU 核。在共享服务器上，请限制线程数，例如 `VIDEO_TTS_THREADS=8`。
- 一章的 1080p 视频在 4 核 CPU 上约需 6–8 分钟。加 `--preview` 渲染 480p 样片更快。视频输出到 `chapters/NN-*/video/out/`（不进 git）。
- 目前视频只有中文版。

## 目录

"视频"一列是 1080p 视频的时长（运行 `bash chapters/NN-*/video/build.sh` 可以重新生成）。视频发布后，我们把链接写进各章 README。

### 第一部分：从一条直线开始（NumPy，CPU）

| 章 | 标题 | 自检 | 视频 |
|---|---|---|---|
| 1 | [y = ax + b —— 从一条直线学会"训练"](chapters/01-linear-regression/README.zh.md) | `/ch01-linear` | 5.2 分钟 |
| 2 | [从标量到矩阵 —— y = XW + b](chapters/02-from-scalar-to-matrix/README.zh.md) | `/ch02-matrix` | 7.4 分钟 |
| 3 | [非线性与神经网络 —— 用折线拼出曲线](chapters/03-neural-network/README.zh.md) | `/ch03-neural-network` | 7.9 分钟 |
| 4 | [反向传播与自动微分 —— 让计算机替你求导](chapters/04-backprop-autograd/README.zh.md) | `/ch04-backprop` | 6.3 分钟 |
| 5 | [分类与概率 —— 从"预测一个数"到"预测哪一类"](chapters/05-classification-probability/README.zh.md) | `/ch05-classification` | 8.7 分钟 |
| 6 | [让训练稳定 —— 初始化、归一化、残差连接、AdamW 与学习率调度](chapters/06-training-stability/README.zh.md) | `/ch06-training-stability` | 9.4 分钟 |

### 第二部分：构建现代 Transformer（PyTorch）

| 章 | 标题 | 自检 | 视频 |
|---|---|---|---|
| 7 | [语言建模与分词 —— 从"预测下一个字"到 byte-level BPE](chapters/07-tokenization-language-model/README.zh.md) | `/ch07-tokenization` | 7.5 分钟 |
| 8 | [注意力 —— 每个位置自己决定看哪里](chapters/08-attention/README.zh.md) | `/ch08-attention` | 7.8 分钟 |
| 9 | [现代 Transformer —— 用注意力搭出一个能写文字的模型](chapters/09-modern-transformer/README.zh.md) | `/ch09-transformer` | 6.3 分钟 |
| 10 | [推理 —— 让模型生成文字，并且生成得快](chapters/10-inference/README.zh.md) | `/ch10-inference` | 9.4 分钟 |

### 第三部分：训练一个真正的模型（主线模型开始）

| 章 | 标题 | 自检 | 视频 |
|---|---|---|---|
| 11 | [评测：先定考卷 —— 考什么、怎么评分、差多少才算赢](chapters/11-evaluation/README.zh.md) | `/ch11-evaluation` | 8.0 分钟 |
| 12 | [Scaling law 与实验设计 —— 先用小模型算清楚，再花钱](chapters/12-scaling-laws/README.zh.md) | `/ch12-scaling-laws` | 7.1 分钟 |
| 13 | [数据 —— 从网页到一份能训练的数据集](chapters/13-data/README.zh.md) | `/ch13-data` | 8.6 分钟 |
| 14 | [预训练工程 —— 混合精度、FlashAttention、数据并行与断点续训](chapters/14-pretraining-engineering/README.zh.md) | `/ch14-pretraining` | 9.6 分钟 |
| 15 | [中期训练与长上下文 —— 最后一段怎么训，更长的文本怎么读](chapters/15-midtraining-long-context/README.zh.md) | `/ch15-midtraining` | 8.8 分钟 |

### 第四部分：后训练 —— 把底座模型变成可用的工具调用模型

| 章 | 标题 | 自检 | 视频 |
|---|---|---|---|
| 16 | [SFT —— 教底座模型回答问题、调用工具](chapters/16-sft/README.zh.md) | `/ch16-sft` | 6.3 分钟 |
| 17 | [蒸馏 —— 小模型向大模型学习](chapters/17-distillation/README.zh.md) | `/ch17-distillation` | 6.4 分钟 |
| 18 | [偏好对齐 —— 从 RLHF 到 DPO](chapters/18-preference-alignment/README.zh.md) | `/ch18-dpo` | 6.5 分钟 |
| 19 | [强化学习 —— 模型从自己的尝试中学习](chapters/19-reinforcement-learning/README.zh.md) | `/ch19-rl` | 6.6 分钟 |
| 20 | [发布 —— 按预注册报告结果，把模型装进笔记本电脑](chapters/20-release/README.zh.md) | `/ch20-release` | 7.4 分钟 |

### 第五部分：架构演进 —— 为了更长的上下文、更小的 KV cache

| 章 | 标题 | 自检 | 视频 |
|---|---|---|---|
| 21 | [KV cache 的账本 —— 长上下文为什么贵，每个 token 要存多少](chapters/21-kv-cache-ledger/README.zh.md) | `/ch21-kv-cache` | 7.0 分钟 |
| 22 | [局部与稀疏注意力 —— 看附近，也保留远处的上下文](chapters/22-local-sparse-attention/README.zh.md) | `/ch22-local-attention` | 6.8 分钟 |
| 23 | [线性注意力与混合架构 —— 把 KV cache 压缩成一个固定大小的矩阵](chapters/23-linear-attention-hybrid/README.zh.md) | `/ch23-linear-attention` | 7.2 分钟 |
| 24 | [混合专家（MoE）—— 参数多出几十倍，每个 token 的算力不变](chapters/24-mixture-of-experts/README.zh.md) | `/ch24-moe` | 6.4 分钟 |
| 25 | [多 token 预测与推测解码 —— 小模型先猜，大模型检查](chapters/25-mtp-speculative-decoding/README.zh.md) | `/ch25-speculative` | 6.4 分钟 |
| 26 | [当前最先进开源模型全景 —— 把整门课的架构放进一棵树](chapters/26-open-model-panorama/README.zh.md) | `/ch26-panorama` | 6.5 分钟 |

## 主线模型进度

| 阶段 | 状态 |
|---|---|
| 第一步：课程 + 生产级代码（在 CPU 上用极小配置跑通，测试通过） | ✅ 完成 |
| 阶段 6：GPU 环境验证（≤ $50） | 🟡 单卡与 2 卡 RTX 3090 版已完成（[runs/2026-10-01-gpu0-check](runs/2026-10-01-gpu0-check/README.md)），8×H100 版还没有做 |
| 阶段 7：重跑对手模型，定稿预注册（[草案](eval/PREREGISTRATION.md)） | ⏳ 等 GPU |
| 阶段 8：阶梯实验与闸门 1（3×RTX 3090 的小规模版本正在进行：[runs/ladder-3090](runs/ladder-3090/README.md)） | 🔄 暂停中 |
| 阶段 9：预训练、中期训练与闸门 2 | ⏳ 等 GPU |
| 阶段 10：后训练、闸门 3 与发布 | ⏳ 等 GPU |
| 阶段 11：把真实结果写回第三、四部分 | ⏳ 等 GPU |

第一步的报告（和计划的差异、没有在 GPU 上验证的部分、第二步的花费、待定的决定）见 [runs/STEP1_REPORT.md](runs/STEP1_REPORT.md)。第二步的操作手册、成本估算和记账表见 [runs/RUNBOOK.md](runs/RUNBOOK.md)、[runs/ledger.md](runs/ledger.md)、[runs/RELEASE_CHECKLIST.md](runs/RELEASE_CHECKLIST.md)。

## 仓库结构

```
chapters/NN-slug/     每章：README.md（英文）、README.zh.md（中文）、code/（极简代码）、video/
zero/                 主线模型的生产级代码（设计说明：zero/DESIGN.zh.md）
configs/              tiny（CPU 冒烟测试）/ ladder（阶梯实验）/ main（主线训练）三档配置
tests/                生产级代码的测试（uv run pytest）
eval/                 预注册草案、对手模型清单
runs/                 第二步的操作手册、记账表、发布清单和实验记录
video_kit/            所有章节共用的视频工具（配色、离线旁白、分镜对齐、字幕与交付检查）
site/                 课程网站（MkDocs；默认英文，有切换中文的按钮）
docs/STYLE_GUIDE.md   中英文写作规则（以 ASD-STE100 为指导）与术语表
docs/CHAPTER_GUIDE.md 章节结构规范
assets/tiny_corpus/   离线极小语料（附许可证说明）
GOAL.md               课程与主线模型的完整目标说明
references.md         参考资料
```

## 说明

- 视频旁白由离线 TTS（sherpa-onnx + MeloTTS 中英混读模型）合成。**还没有人试听过发音和语速。**发布前必须有人逐个试听。
- 第三、四部分标注"极小配置演示"的数字，来自约 1M 参数的极小模型在 CPU 上的真实运行。这些数字只说明代码能跑通，不代表主线模型的效果。
- 第 11、16–20 章引用的冒烟测试数字，来自修复工具调用评分器之前的那次运行。修复后重跑，SFT 及之前各阶段的数字完全相同，蒸馏及之后的数字有变化（各章"主线进度"里有说明）。
- 生产级代码里，CPU 上测不了的路径（多卡、FlashAttention kernel、BF16 等）原来都标注"尚未在 GPU 上验证"。2026 年 10 月，我们在单张和 2 张 RTX 3090 上验证了这些路径（BF16、FlashAttention、DDP / FSDP2、断点续训、compile、后训练通路等）。结果见 [runs/2026-10-01-gpu0-check](runs/2026-10-01-gpu0-check/README.md)，标注已改为具体的验证状态。8×H100 / NVLink 下的吞吐与 MFU、32K 长序列的显存，还要在第二步实测。
