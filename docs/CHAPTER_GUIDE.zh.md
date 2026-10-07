# 章节写作指南（给执行者和贡献者）

[English](CHAPTER_GUIDE.md) · **中文**

本指南把 [GOAL.zh.md](../GOAL.zh.md) 的要求落成可操作的规范。**第 1 章（`chapters/01-linear-regression/`）是样板。**动手前，先完整读一遍它的 `README.md`（英文）、`README.zh.md`（中文）、`code/`、`video/script.md`、`video/scenes.py`。再读 [STYLE_GUIDE.zh.md](STYLE_GUIDE.zh.md)：它规定两种语言的写作规则，并给出术语表。

## 1. 目录

```
chapters/NN-slug/
├── README.md          # 英文：速读正文 + 从极简代码到生产级代码 + 引导问题 + 动手任务 + (CS336) + 本章参考文献
├── README.zh.md       # 中文版：标题相同，顺序相同
├── code/              # 第一层：极简代码。文件名 01_xxx.py、02_xxx.py …
└── video/
    ├── script.md      # 事实清单 → 分镜与旁白 → 交付检查（中文）
    ├── scenes.py      # Manim 场景，类名 ChapterScene
    ├── build.sh       # 照抄第 1 章，只改章节目录名
    └── subtitles.srt  # 由 build 生成（全片渲染时）
.claude/commands/chNN-<slug>.md   # 自检 Skill（英文）
```

## 2. 章节页面的结构（按顺序）

每章有两个内容相同的页面：`README.md`（英文）和 `README.zh.md`（中文）。两页的标题相同、顺序相同、级别相同。这样课程网站上的语言按钮能打开另一种语言的同一节。两页都遵守 [STYLE_GUIDE.zh.md](STYLE_GUIDE.zh.md)。

1. 第 1 行是标题：`# Chapter N: Title — Subtitle`（英文页）/ `# 第 N 章：标题 —— 副标题`（中文页）。
2. 第 3 行是语言切换行：`**English** · [中文](README.zh.md)` / `[English](README.md) · **中文**`。
3. `> **Goal**: After this chapter, you can …` / `> **目标**：读完这一章，你能……`。写可以检验的动作，不写"了解"。
4. 视频与自检两行（照抄第 1 章的格式）：`📺 **Video**:` / `📺 **本章视频**：` 和 `🧪 **Self-check**:` / `🧪 **本章自检**：`。
5. 衔接段："上一章我们……，这一章要解决的问题是……"（见 GOAL.md 2.2）。
6. **速读正文**：分小节，顺序是 直觉 → 公式 → 极简代码 → 小结。15–30 分钟读完（中文页约 5000–9000 字；英文页内容相同）。
   - 每个公式都能在 `code/` 里找到对应的那一行。正文里用代码片段标出这一行。
   - 正文里引用的所有数字，必须来自运行 `code/` 得到的真实输出（把输出贴成表格）。
   - 两页里的代码片段和程序输出都是英文，因为代码是英文的。
7. **从极简代码到生产级代码**：
   - 第 1–6 章：对应 PyTorch 标准写法（放在 `code/` 里的一个脚本中）。
   - 第 7 章起：对应 `zero/` 里的文件和函数（写相对路径和函数名）。用表格逐条说明"生产级多做了什么、为什么"。再说明对拍方式（哪个测试保证两层结果一致）。
8. **主线进度**（只有第 11–20 章）：第一步写极小配置演示的真实运行结果，并明确标注"极小配置演示"。另外写"待 GPU 训练后补充"。
9. **前沿观察**（可选）：不满足共识规则的技术放这里。用一段话说明：它是什么、谁在用、为什么还不算共识。
10. **采用方与来源**（技术类章节必须有）：正文里的每项主流技术，列出至少 3 个采用它的头部开源模型家族。每个家族附技术报告或模型卡的链接。核实不了的写"待核实"。
11. `## 引导问题`：4–6 个问题，交给读者和 Claude Code 对话。
12. `## 动手任务`：三个任务：**任务 1（基础）**、**任务 2（核心）**、**任务 3（挑战）**。都要能在 CPU 上完成。
13. `## 想深入：CS336`（第 7 章起）：见第 6 节的对应表。
14. `## 本章参考文献`：可点击的链接。优先用 `references.md` 里已有的资料。
15. 末尾用一段话预告下一章：`**下一章**：`。

两页使用下面这些标准名称：

| 英文页 | 中文页 |
|---|---|
| `> **Goal**:` | `> **目标**：` |
| `📺 **Video**:` | `📺 **本章视频**：` |
| `🧪 **Self-check**:` | `🧪 **本章自检**：` |
| Summary | 小结 |
| From minimal code to production code | 从极简代码到生产级代码 |
| Main-line progress | 主线进度 |
| Tiny-configuration demo | 极小配置演示 |
| To be added after GPU training | 待 GPU 训练后补充 |
| Frontier notes | 前沿观察 |
| Adopters and sources | 采用方与来源 |
| Guided questions | 引导问题 |
| Hands-on tasks | 动手任务 |
| **Task 1 (basic)** / **Task 2 (core)** / **Task 3 (challenge)** | **任务 1（基础）** / **任务 2（核心）** / **任务 3（挑战）** |
| Go deeper: CS336 | 想深入：CS336 |
| References | 本章参考文献 |
| `**Next chapter**:` | `**下一章**：` |
| `> **Note:**` | `> **注意：**` |
| `> **Warning:**` | `> **警告：**` |
| to be verified | 待核实 |
| GPU measurements (one RTX 3090) | GPU 实测（单张 RTX 3090） |
| 10k yuan | 万元 |

链接：

- 指向其他章节的链接，指向同一种语言的页面：英文页写 `../02-from-scalar-to-matrix/README.md`，中文页写 `../02-from-scalar-to-matrix/README.zh.md`。
- 指向其他章节"采用方与来源"一节的链接：英文页写 `../NN-slug/README.md#adopters-and-sources`，中文页写 `../NN-slug/README.zh.md#采用方与来源`。
- 指向代码、数据和其他文件的链接，两页相同。

本章研究的中文示例文本（例如分词器的输入），在英文页里保持中文。第一次出现时，在括号里给出英文意思。

写作风格：遵守 [STYLE_GUIDE.zh.md](STYLE_GUIDE.zh.md)。两种语言都以 ASD-STE100 简化技术英语（Simplified Technical English）的规则为指导：

- 一句话只讲一件事。句子要短：英文的步骤句不超过 20 个词，描述句不超过 25 个词；中文的步骤句一般不超过 30 个字，描述句一般不超过 40 个字。
- 用主动句。步骤用祈使句。
- 一个概念只用一个词，用术语表里的写法。
- 直觉优先。不用"显然""易证""其实""非常"这类词（英文不用 "clearly""obviously""simply""easy"）。
- 术语第一次出现时，给一个简短的定义。中文页在括号里给出英文。
- 不要写成教科书。

## 3. 极简代码（`code/`）

- 单文件、几十到一两百行、只依赖 NumPy 或 PyTorch（加 matplotlib 也可以）。
- `uv run python chapters/NN-slug/code/xx.py` 在 CPU 上几分钟内跑完（最好几秒），并打印可读的结果。
- 注释、docstring 和打印输出都用英文（见 [STYLE_GUIDE.zh.md](STYLE_GUIDE.zh.md) 第 4 节）。注释说明代码为什么这样做。不要翻译数据：程序处理的中文示例文本保持原样。
- 固定随机种子，保证正文里贴的数字可以复现。
- PyTorch 脚本开头加 `torch.set_num_threads(1)`。构建环境里多个任务共享 CPU，多线程反而会慢上百倍（读者在自己的电脑上可以删掉这行）。
- 不要写文件到仓库里。如果要输出图片，写到 `chapters/NN-slug/code/out/`（前提是 .gitignore 已经忽略这个目录；否则只打印）。
- 需要数据时：优先在代码内造数据。需要真实文本时，用 `data/tiny/` 里的小语料（由 `zero` 核心代码提供，见 `zero/DESIGN.md`）。**不要**依赖 Hugging Face 下载（本构建环境访问不了 huggingface.co）。
- 章节之间可以复用代码：用 `importlib` 按路径加载其他章节的文件（见第 1 章的 `02_learning_rate.py`）。

## 4. 视频

> **注意：**视频暂时保持中文。旁白、屏幕文字和 `script.md` 都用中文。课程改为以英文为主之后，本节的规则不变。

### 4.1 script.md

照第 1 章的格式：

- **一、事实清单**：一张表，列出视频里的每个主张、公式、数字，以及它的来源（代码输出或文献）和状态。
- **二、分镜与旁白**：每镜一个 `### Sxx 标题`，下面三个字段：`- 画面：`、`- 屏幕文字：`、`- 旁白：`。旁白必须写在 `- 旁白：` 这一行（这一行可以很长，不要换行）。
  - 旁白是给 TTS 念的：用口语，一镜讲一件事。公式、希腊字母、缩写用 `{显示|读法}`，例如 `{ŷ|y hat}`、`{η|eta}`、`{λmax|lambda max}`、`{QKᵀ|Q 乘 K 的转置}`。字幕显示前半部分，TTS 念后半部分。
  - 数字写成读起来更自然的中文（"零点二五"），或者直接写阿拉伯数字（TTS 会读）。
  - **时长**：全片 5–10 分钟。当前 TTS 约每秒念 5.5–6 个汉字，所以旁白总量约 **1900–3000 字**（不含标记）。第 1 章：12 镜、约 2300 字、5.2 分钟。
- **三、交付检查**：一份清单，最后一行写 `> ⚠️ 旁白发音与语速未经人工试听。`

### 4.2 scenes.py

```python
from video_kit import theme
from video_kit.scene import NarratedScene, polyline_in_axes, zh

class ChapterScene(NarratedScene):
    chapter_label = "第 N 章"
    chapter_title = "标题"

    def construct(self):
        with self.shot("S01"):                     # each shot of script.md must occur exactly once
            self.play(*self.set_heading("小标题"), run_time=self.fit(1))
            self.play(Create(x), run_time=self.fit(2))   # fit: not longer than the narration time left in this shot
            self.wait(self.remaining() * 0.3)             # divide the waits by the rhythm of the narration
```

示例里的字符串是中文的屏幕文字：`"第 N 章"`、`"标题"`、`"小标题"`。

API（见 `video_kit/scene.py`）：

- `with self.shot("Sxx")`：插入本镜旁白。退出时自动补齐等待。
- `self.fit(t, reserve=0)`：动画时长，不超过本镜剩余时间。`self.remaining()`：本镜剩余秒数。
- `self.set_heading(text)`：换左上角的标题（返回动画列表，用 `self.play(*...)`）。`set_heading(None)` 清除标题。**不要**自己用 `self.heading()` 叠加标题。
- `zh(text, size, color)`：中文文字。公式用 `MathTex`（LaTeX 可用，但 MathTex 里不能写中文）。
- `polyline_in_axes(axes, [(x, y), ...], color=..., stroke_width=...)`：只画坐标范围内的部分。等高线、轨迹用它，避免线条溢出坐标轴。
- `self.demo_badge()` / `self.show_badge()`：第 11–20 章用到极小配置数据的镜头，右上角必须带"极小配置演示"标注。
- 片头第一镜用 `self.chapter_card()`。
- 代码块用 `code_block(source, size)`（`from video_kit.scene import code_block`）。它保留缩进，返回每行一个 Text 的 VGroup，便于逐行高亮。
- scenes.py 里要重复跑的较慢计算（例如训练小网络），用 `functools.lru_cache`，或把结果缓存到 `video/out/cache.json`。这样每次渲染不用重跑。

版式规则：

- 画面坐标为 x ∈ [−7.1, 7.1]，y ∈ [−4, 4]。**y < −2.75 是字幕区，任何内容都不要放进去。**
- 左上角是标题区（y > 3.0 的左半边）。常用布局：左半边放一个坐标系（中心 x≈−3.4），右半边放公式或说明（中心 x≈3.5）。
- 配色用语义色：`theme.INPUT`（输入/数据，蓝）、`theme.PARAM`（参数，橙）、`theme.GRAD`（梯度/误差，红）、`theme.ATTN`（注意力，紫）、`theme.OUTPUT`（输出/预测，绿）、`theme.HIGHLIGHT`（强调，黄）、`theme.MUTED`（次要）。
- 一镜结束前，把本镜专属的元素 FadeOut，避免堆叠。
- `DecimalNumber` 的数值变化用 `ChangeDecimalToValue`，不要用 `.animate.set_value`。
- 画面里的数字从 `../code/` 里的代码算出来（用 `importlib` 加载），不要手抄。

### 4.3 渲染与自查（必须做）

```bash
bash chapters/NN-slug/video/build.sh --preview    # 480p sample, about 2 minutes
```

- 看终端最后的交付检查（`[build] Delivery check:` 那一行）：`problems` 必须为空，时长必须在 5–10 分钟内（样片也会报告时长）。
- 打开 `video/out/frames/` 里每镜中点的截图，检查版式。可以用 ffmpeg 拼成一张图，再用 Read 工具查看：
  `ffmpeg -i a.png -i b.png -i c.png -i d.png -filter_complex "[0][1]hstack[x];[2][3]hstack[y];[x][y]vstack" grid.png`
- 检查这些问题：文字重叠、超出画面、进入字幕区、标题叠在一起、线条溢出坐标轴。修到干净为止。
- **不要**渲染 1080p 全片（最后由主流程统一渲染）。

## 5. 自检 Skill（`.claude/commands/chNN-<slug>.md`）

照 `.claude/commands/ch01-linear.md` 的写法：

- front matter 里写 `description`：`"Chapter N self-check: <topic> (第 N 章自检：<中文主题>)"`。
- Skill 用英文写。加上第 1 章 Skill 里的 **Language** 段落：按学习者的语言回答；本章的中文正文在 `README.zh.md`；句子简短清楚（见 `docs/STYLE_GUIDE.md`）。
- 4 关检验问题（概念 → 直觉 → 发现问题 → 迁移），每关给出期望回答。
- 最后指向下一章。

## 6. CS336 对应表（Spring 2026，已核对 <https://cs336.stanford.edu/>）

Spring 2026 讲次：1 概览与分词；2 PyTorch 与资源核算；3 架构与超参；4 注意力的替代方案与 MoE；5 GPU/TPU；6 Kernel 与 Triton；7、8 并行；9、11 Scaling Law；10 推理；12 评测；13 数据（来源与数据集）；14 数据（过滤、去重、配比、合成数据）；15 中期训练/后训练（SFT/RLHF）；16 后训练 RLVR；17 对齐与多模态；18、19 客座讲座。作业：A1 Basics、A2 Systems、A3 Scaling、A4 Data、A5 Alignment and Reasoning RL（可选第二部分：DPO）。

| 本课章节 | 写在"想深入：CS336"里的内容 |
|---|---|
| 7 | 第 1 讲（概览与分词）、第 2 讲（资源核算）；作业 1 的 BPE 部分 |
| 8 | 第 3 讲（架构）；作业 1 |
| 9 | 第 3 讲（架构与超参）；作业 1 |
| 10 | 第 10 讲（推理）；作业 1 的解码部分 |
| 11 | 第 12 讲（评测） |
| 12 | 第 9、11 讲（Scaling Law）；作业 3 |
| 13 | 第 13、14 讲（数据）；作业 4 |
| 14 | 第 5 讲（GPU）、第 6 讲（Kernel/Triton）、第 7–8 讲（并行）；作业 2 |
| 15 | 第 14 讲（配比）、第 15 讲（中期训练）；部分覆盖 |
| 16 | 第 15 讲（SFT）；作业 5 |
| 17 | 第 14 讲（合成数据）、第 15 讲；部分覆盖 |
| 18 | 第 15 讲（RLHF）；作业 5 可选第二部分（DPO） |
| 19 | 第 16 讲（RLVR）；作业 5 |
| 20 | 第 10 讲（推理）、第 12 讲（评测） |
| 21、25 | 第 10 讲（推理） |
| 22、23 | 第 4 讲（注意力的替代方案）；超出部分注明"CS336 未深入" |
| 24 | 第 4 讲（MoE） |
| 26 | 第 3、4 讲；超出部分注明"CS336 未深入" |

链接统一写 <https://cs336.stanford.edu/>（课程页有每讲的讲义和 YouTube 录像）。

## 7. 协作约定（多人或多个 agent 并行时）

- 临时文件放在自己章节的 `video/out/` 下，或草稿目录里以章节命名的子目录下。不要和其他 agent 共用同名文件。
- 只改自己负责的章节目录和对应的 Skill 文件。需要改 `video_kit/`、`zero/` 或别人的文件时，在汇报里提出，不要直接改。
- **不要 git commit**（由主流程统一提交）。
- 新找到的参考资料写进本章参考文献，并在汇报里列出。由主流程合并进 `references.md`。
- 汇报里写清：完成了什么、运行了哪些命令及结果、样片时长、还有哪些"待核实"。
