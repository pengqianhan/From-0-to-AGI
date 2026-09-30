# From 0 to AGI：从 y = ax + b 到最先进的开源模型

> *You can outsource your thinking, but you cannot outsource your understanding.*

一门**中文**课程：从一条直线 `y = ax + b` 出发，讲到亲手搭建 Transformer、Scaling Law、预训练工程、SFT / 蒸馏 / DPO / GRPO 强化学习，再到为了更长上下文、更小 KV cache 而演进出的最新架构。**只讲已经形成共识的主流技术**（判定规则见 [GOAL.md](GOAL.md) 2.1）。

每一章有三样东西：

- **文字速读版**：15–30 分钟读完，直觉 → 公式 → 极简代码 → 小结。
- **讲解视频**：5–10 分钟，画面里的数字全部由本章代码真实算出。
- **两层代码**：先是 CPU 上几秒到几分钟跑完的**极简代码**（`chapters/NN-*/code/`），再是同一个想法在主线模型里的**生产级写法**（[`zero/`](zero/DESIGN.md)），两层互相对拍。

课程的后半程贯穿一个**主线模型**：用约 1 万美元算力从零训练一个中英双语、约 0.69B 参数的小模型，目标是在**工具调用**上超过同尺寸的所有公开模型（包括 Qwen3.5-0.8B），通用基准如实报告。生产级代码已经全部写好并在 CPU 上用极小配置跑通；真实训练在第二步、有 GPU 之后进行（见 [runs/RUNBOOK.md](runs/RUNBOOK.md)）。

## 和斯坦福 CS336 的关系

本课定位为 [CS336（Language Modeling from Scratch）](https://cs336.stanford.edu/) 的**中文入门与导读**：第 1–6 章补上 CS336 默认你已经会的深度学习基础；第 7 章起每章末尾有"想深入：CS336"，指向 Spring 2026 对应的讲次和作业。本课多出来的两条主线是：真实训练并发布一个模型的全过程（预注册、闸门、公平评测），以及第五部分的架构演进。

## 怎么学

1. 读章节 README 的**一句话目标**和速读正文（最好配合视频）。
2. 运行本章 `code/` 里的极简代码，对照正文里的数字。
3. 读"从极简到生产级"，打开 `zero/` 里对应的文件。
4. 带着"引导问题"去问 Claude Code，完成"动手任务"。
5. 在 Claude Code 里输入 `/chNN-...` 做自我检验（Skill 在 `.claude/commands/`）。

```bash
# 安装依赖（需要 uv：https://docs.astral.sh/uv/）
uv sync                       # 文字 + 代码
uv sync --extra video         # 还想自己渲染视频（另需 ffmpeg、LaTeX）

# 跑第 1 章
uv run python chapters/01-linear-regression/code/01_fit_line.py

# 渲染第 1 章视频（480p 样片加 --preview）
bash chapters/01-linear-regression/video/build.sh

# 生产级代码的测试与端到端冒烟（CPU）
uv run pytest
uv run python -m zero.smoke
```

## 目录

表中"视频"一列是 480p 样片的时长；1080p 成片发布后会把链接写回各章 README。

### 第一部分：从一条直线开始（NumPy，CPU）

| 章 | 标题 | 自检 | 视频 |
|---|---|---|---|
| 1 | [y = ax + b —— 从一条直线学会"训练"](chapters/01-linear-regression/) | `/ch01-linear` | 5.2 分钟 |
| 2 | [从标量到矩阵 —— y = XW + b](chapters/02-from-scalar-to-matrix/) | `/ch02-matrix` | 7.4 分钟 |
| 3 | [非线性与神经网络 —— 用折线拼出曲线](chapters/03-neural-network/) | `/ch03-neural-network` | 7.9 分钟 |
| 4 | [反向传播与自动微分 —— 让计算机替你求导](chapters/04-backprop-autograd/) | `/ch04-backprop` | 6.3 分钟 |
| 5 | [分类与概率 —— 从"猜一个数"到"猜哪一类"](chapters/05-classification-probability/) | `/ch05-classification` | 8.5 分钟 |
| 6 | [让训练稳定 —— 初始化、归一化、残差、AdamW 与学习率调度](chapters/06-training-stability/) | `/ch06-training-stability` | 9.3 分钟 |

### 第二部分：构建现代 Transformer（PyTorch）

| 章 | 标题 | 自检 | 视频 |
|---|---|---|---|
| 7 | [语言建模与分词 —— 从"猜下一个字"到 byte-level BPE](chapters/07-tokenization-language-model/) | `/ch07-tokenization` | 7.4 分钟 |
| 8 | [注意力 —— 让每个位置自己决定看哪里](chapters/08-attention/) | `/ch08-attention` | 7.7 分钟 |
| 9 | [现代 Transformer —— 把注意力搭成一个会写字的模型](chapters/09-modern-transformer/) | `/ch09-transformer` | 6.3 分钟 |
| 10 | [推理 —— 让模型开口说话，而且说得快](chapters/10-inference/) | `/ch10-inference` | 9.4 分钟 |

### 第三部分：训练一个真正的模型（主线模型开始）

| 章 | 标题 | 自检 | 视频 |
|---|---|---|---|
| 11 | [评测：先定考卷 —— 考什么、怎么判、差多少才算赢](chapters/11-evaluation/) | `/ch11-evaluation` | 8.0 分钟 |
| 12 | [Scaling Law 与实验设计 —— 先用小模型算清楚，再花大钱](chapters/12-scaling-laws/) | `/ch12-scaling-laws` | 6.3 分钟 |
| 13 | [数据 —— 从一堆网页到一份能训练的数据集](chapters/13-data/) | `/ch13-data` | 8.6 分钟 |
| 14 | [预训练工程 —— 混合精度、FlashAttention、数据并行与断点续训](chapters/14-pretraining-engineering/) | `/ch14-pretraining` | 9.4 分钟 |
| 15 | [中期训练与长上下文 —— 最后一段怎么训，读不长怎么办](chapters/15-midtraining-long-context/) | `/ch15-midtraining` | 8.8 分钟 |

### 第四部分：后训练 —— 把底座变成可用的工具调用模型

| 章 | 标题 | 自检 | 视频 |
|---|---|---|---|
| 16 | [SFT —— 把只会续写的底座，教成会回答、会调工具的助手](chapters/16-sft/) | `/ch16-sft` | 6.0 分钟 |
| 17 | [蒸馏 —— 让小模型向大模型学](chapters/17-distillation/) | `/ch17-distillation` | 6.3 分钟 |
| 18 | [偏好对齐 —— 从 RLHF 到 DPO](chapters/18-preference-alignment/) | `/ch18-dpo` | 6.4 分钟 |
| 19 | [强化学习 —— 让模型从自己的尝试里学](chapters/19-reinforcement-learning/) | `/ch19-rl` | 6.6 分钟 |
| 20 | [发布 —— 按预注册交卷，把模型装进笔记本](chapters/20-release/) | `/ch20-release` | 7.4 分钟 |

### 第五部分：架构演进 —— 为了更长的上下文、更小的 KV cache

| 章 | 标题 | 自检 | 视频 |
|---|---|---|---|
| 21 | [KV cache 的账本 —— 长上下文贵在哪，每个 token 该存多少](chapters/21-kv-cache-ledger/) | `/ch21-kv-cache` | 7.1 分钟 |
| 22 | [局部与稀疏注意力 —— 只看附近，也不丢掉远处](chapters/22-local-sparse-attention/) | `/ch22-local-attention` | 6.6 分钟 |
| 23 | [线性注意力与混合架构 —— 把 KV cache 压成一个固定大小的矩阵](chapters/23-linear-attention-hybrid/) | `/ch23-linear-attention` | 7.1 分钟 |
| 24 | [混合专家（MoE）—— 参数翻几十倍，每个 token 的算力不变](chapters/24-mixture-of-experts/) | `/ch24-moe` | 6.3 分钟 |
| 25 | [多 token 预测与推测解码 —— 让小模型先猜，大模型一次改完](chapters/25-mtp-speculative-decoding/) | `/ch25-speculative` | 6.4 分钟 |
| 26 | [当前最先进开源模型全景 —— 把整门课的架构放进一棵树](chapters/26-open-model-panorama/) | `/ch26-panorama` | 6.5 分钟 |

## 主线模型进度

| 阶段 | 状态 |
|---|---|
| 第一步：课程 + 生产级代码（CPU 上极小配置跑通、测试通过） | ✅ 完成 |
| 阶段 6：GPU 环境验证（≤ $50） | ⏳ 等 GPU |
| 阶段 7：对手重跑与预注册定稿（[草案](eval/PREREGISTRATION.md)） | ⏳ 等 GPU |
| 阶段 8：阶梯实验与闸门 1 | ⏳ 等 GPU |
| 阶段 9：预训练、中期训练与闸门 2 | ⏳ 等 GPU |
| 阶段 10：后训练、闸门 3 与发布 | ⏳ 等 GPU |
| 阶段 11：把真实结果回填进第三、四部分 | ⏳ 等 GPU |

第二步的操作手册、成本估算和记账表见 [runs/RUNBOOK.md](runs/RUNBOOK.md)、[runs/ledger.md](runs/ledger.md)、[runs/RELEASE_CHECKLIST.md](runs/RELEASE_CHECKLIST.md)。

## 仓库结构

```
chapters/NN-slug/     每章：README（正文）、code/（极简代码）、video/（脚本、场景、渲染命令、字幕）
zero/                 主线模型的生产级代码（设计说明：zero/DESIGN.md）
configs/              tiny（CPU 冒烟）/ ladder（阶梯实验）/ main（主线训练）三档配置
tests/                生产级代码的测试（uv run pytest）
eval/                 预注册草案、对手清单
runs/                 第二步的运行手册、记账表、发布清单
video_kit/            全课程共用的视频工具（配色、离线旁白、分镜对齐、字幕与交付检查）
docs/CHAPTER_GUIDE.md 章节写作规范
assets/tiny_corpus/   离线极小语料（附许可证说明）
GOAL.md               课程与主线模型的完整目标说明
references.md         参考资料
```

## 说明

- 视频旁白由离线 TTS（sherpa-onnx + MeloTTS 中英混读模型）合成，**发音与语速尚未经人工试听**，发布前需要人工听一遍。
- 第三、四部分里标注"极小配置演示"的数字来自约 1M 参数的极小模型在 CPU 上的真实运行，只说明代码通路正确，不代表主线模型的效果。
- 生产级代码里凡是 CPU 上测不了的路径（多卡、FlashAttention kernel、BF16 等）都标注了"尚未在 GPU 上验证"。
