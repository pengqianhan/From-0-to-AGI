# Chapter writing guide (for executors and contributors)

**English** · [中文](CHAPTER_GUIDE.zh.md)

This guide changes the requirements of [GOAL.md](../GOAL.md) into rules that you can apply. **Chapter 1 (`chapters/01-linear-regression/`) is the model.** Before you start, read all of its `README.md` (English), `README.zh.md` (Chinese), `code/`, `video/script.md`, and `video/scenes.py`. Read also [STYLE_GUIDE.md](STYLE_GUIDE.md): it gives the writing rules for both languages and the glossary.

## 1. Folders

```
chapters/NN-slug/
├── README.md          # English: quick-read text + From minimal code to production code + Guided questions + Hands-on tasks + (CS336) + References
├── README.zh.md       # Chinese version: same headings in the same order
├── code/              # level 1: minimal code. File names 01_xxx.py, 02_xxx.py, …
└── video/
    ├── script.md      # fact list → storyboard and narration → delivery check (in Chinese)
    ├── scenes.py      # Manim scenes, class name ChapterScene
    ├── build.sh       # copy it from Chapter 1; change only the chapter folder name
    └── subtitles.srt  # made by the build (in a full render)
.claude/commands/chNN-<slug>.md   # self-check skill (in English)
```

## 2. Structure of the chapter pages (in this order)

Each chapter has two pages with the same content: `README.md` (English) and `README.zh.md` (Chinese). The two pages have the same headings in the same order, at the same levels. Then the language button on the course website opens the same section in the other language. Both pages follow [STYLE_GUIDE.md](STYLE_GUIDE.md).

1. Line 1, the title: `# Chapter N: Title — Subtitle` (English page) / `# 第 N 章：标题 —— 副标题` (Chinese page).
2. Line 3, the language line: `**English** · [中文](README.zh.md)` / `[English](README.md) · **中文**`.
3. `> **Goal**: After this chapter, you can …` / `> **目标**：读完这一章，你能……`. Write an action that a reader can test, not "know about".
4. The video line and the self-check line (copy the format of Chapter 1): `📺 **Video**:` / `📺 **本章视频**：` and `🧪 **Self-check**:` / `🧪 **本章自检**：`.
5. The linking paragraph: "In the last chapter, we … . In this chapter, the problem to solve is … ." (see GOAL.md 2.2).
6. **The quick-read text**: subsections in this order: intuition → formulas → minimal code → Summary. A reader can read it in 15–30 minutes (about 5000–9000 Chinese characters in the Chinese page; the English page has the same content).
   - Each formula has a matching line in `code/`. Show that line in the text with a code snippet.
   - All numbers in the text must come from the real output of `code/` (paste the output as a table).
   - The code snippets and the program output are in English in both pages, because the code is in English.
7. **From minimal code to production code**:
   - Chapters 1–6: the standard PyTorch code (in one script in `code/`).
   - From Chapter 7: the files and functions in `zero/` (write the relative path and the function name). Use a table to tell, item by item, "what the production code does in addition, and why". Also tell how the parity check works (which test makes sure that the two levels give the same result).
8. **Main-line progress** (only Chapters 11–20): in Step 1, write the real results of the tiny-configuration demo, with the clear label "Tiny-configuration demo". Also write "To be added after GPU training".
9. **Frontier notes** (optional): put here the methods that do not satisfy the consensus rule. Write one paragraph: what the method is, who uses it, and why it is not a consensus yet.
10. **Adopters and sources** (necessary in chapters about methods): for each mainstream method in the text, list at least 3 leading open model families that use it. Add a link to the technical report or the model card of each family. If you cannot verify an item, write "to be verified".
11. `## Guided questions`: 4–6 questions for the reader to discuss with Claude Code.
12. `## Hands-on tasks`: three tasks: **Task 1 (basic)**, **Task 2 (core)**, and **Task 3 (challenge)**. A reader must be able to do all of them on a CPU.
13. `## Go deeper: CS336` (from Chapter 7): see the mapping table in Section 6.
14. `## References`: clickable links. First use the sources that are already in `references.md`.
15. At the end, one paragraph that introduces the next chapter: `**Next chapter**:`.

Use these standard names in the two pages:

| English page | Chinese page |
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

Links:

- A link to another chapter goes to the page in the same language: `../02-from-scalar-to-matrix/README.md` in the English page, `../02-from-scalar-to-matrix/README.zh.md` in the Chinese page.
- A link to the section "Adopters and sources" of another chapter: `../NN-slug/README.md#adopters-and-sources` in the English page, `../NN-slug/README.zh.md#采用方与来源` in the Chinese page.
- Links to code, data, and other files are the same in both pages.

Chinese example text that the chapter studies (for example, the input to a tokenizer) stays in Chinese in the English page. Give its English meaning in parentheses at its first use.

Writing style: follow [STYLE_GUIDE.md](STYLE_GUIDE.md). Both languages use the rules of ASD-STE100 Simplified Technical English as a guide:

- Write one topic in one sentence. Use short sentences: in English, a maximum of 20 words for an instruction and 25 words for information; in Chinese, about 30 and 40 characters.
- Use the active voice. Use the imperative for instructions.
- Use one term for one meaning. Use the terms in the glossary.
- Put the intuition first. Do not use words such as "clearly", "obviously", "simply", or "easy" (in Chinese: "显然", "易证", "其实", "非常").
- At the first use of a technical term, give a short definition. In the Chinese page, give the English term in parentheses.
- Do not write a textbook.

## 3. Minimal code (`code/`)

- One file, tens of lines to one or two hundred lines. It depends only on NumPy or PyTorch (matplotlib is also OK).
- `uv run python chapters/NN-slug/code/xx.py` finishes on a CPU in a few minutes (seconds is better), and it prints readable results.
- Write comments, docstrings, and printed output in English (see Section 4 of [STYLE_GUIDE.md](STYLE_GUIDE.md)). A comment tells why the code does something. Do not translate data: Chinese example text that the program processes stays as it is.
- Fix the random seed, so that the numbers in the text are reproducible.
- At the start of a PyTorch script, add `torch.set_num_threads(1)`. In the build environment, many tasks share the CPU. There, multiple threads make the script hundreds of times slower. Readers can remove this line on their own computer.
- Do not write files into the repository. If you must write images, write them to `chapters/NN-slug/code/out/` (only if .gitignore ignores that folder; if not, only print).
- Data: first, make the data in the code. If you need real text, use the small corpus in `data/tiny/` (the `zero` core code supplies it, see `zero/DESIGN.md`). **Do not** depend on downloads from Hugging Face (this build environment cannot access huggingface.co).
- Chapters can reuse the code of other chapters: load a file of another chapter by its path with `importlib` (see `02_learning_rate.py` of Chapter 1).

## 4. Video

> **Note:** The videos stay in Chinese for now. The narration, the on-screen text, and `script.md` are in Chinese. The English-first language policy does not change the rules in this section.

### 4.1 script.md

Use the format of Chapter 1:

- **一、事实清单** (1. Fact list): a table that lists each claim, formula, and number in the video, its source (code output or literature), and its status.
- **二、分镜与旁白** (2. Storyboard and narration): each shot has a heading `### Sxx 标题` (Sxx title), with three fields under it: `- 画面：` (image), `- 屏幕文字：` (on-screen text), and `- 旁白：` (narration). The narration must be on the `- 旁白：` line (the line can be very long; do not break it).
  - The narration is for the TTS: write in spoken style, and cover one thing in each shot. For formulas, Greek letters, and abbreviations, use `{display|reading}`, for example `{ŷ|y hat}`, `{η|eta}`, `{λmax|lambda max}`, `{QKᵀ|Q 乘 K 的转置}`. The subtitle shows the first part, and the TTS reads the second part.
  - Write numbers in Chinese words when that reads more naturally ("零点二五", 0.25), or write digits (the TTS reads them).
  - **Length**: the full video is 5–10 minutes. The current TTS reads about 5.5–6 Chinese characters per second, so the narration has about **1900–3000 characters** in total (without the markup). Chapter 1: 12 shots, about 2300 characters, 5.2 minutes.
- **三、交付检查** (3. Delivery check): a checklist, and as the last line `> ⚠️ 旁白发音与语速未经人工试听。` ("A person has not listened to the pronunciation and the speed of the narration.")

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

The strings in the example are Chinese on-screen text: `"第 N 章"` (Chapter N), `"标题"` (title), and `"小标题"` (subheading).

API (see `video_kit/scene.py`):

- `with self.shot("Sxx")`: inserts the narration of this shot. At exit, it adds the wait that is still necessary.
- `self.fit(t, reserve=0)`: the length of an animation, not longer than the time left in this shot. `self.remaining()`: the seconds left in this shot.
- `self.set_heading(text)`: changes the heading at the top left (it returns a list of animations: use `self.play(*...)`). `set_heading(None)` removes the heading. **Do not** add headings yourself with `self.heading()`.
- `zh(text, size, color)`: Chinese text. For formulas, use `MathTex` (LaTeX works, but a MathTex cannot contain Chinese).
- `polyline_in_axes(axes, [(x, y), ...], color=..., stroke_width=...)`: draws only the part inside the range of the axes. Use it for contour lines and paths, so that lines do not go outside the axes.
- `self.demo_badge()` / `self.show_badge()`: in Chapters 11–20, each shot that uses tiny-configuration data must have the label "极小配置演示" (tiny-configuration demo) at the top right.
- For the first shot (the opening), use `self.chapter_card()`.
- For code blocks, use `code_block(source, size)` (`from video_kit.scene import code_block`). It keeps the indentation and returns a VGroup with one Text for each line, so that you can highlight line by line.
- Slow calculations in scenes.py that run again and again (for example, the training of a small network): use `functools.lru_cache`, or cache the results in `video/out/cache.json`. Then each render does not run them again.

Layout rules:

- The frame coordinates are x ∈ [−7.1, 7.1], y ∈ [−4, 4]. **y < −2.75 is the subtitle area. Do not put any content there.**
- The top left is the heading area (the left half where y > 3.0). A common layout: a coordinate system in the left half (center x≈−3.4), and formulas or explanations in the right half (center x≈3.5).
- Use the semantic colors: `theme.INPUT` (input/data, blue), `theme.PARAM` (parameter, orange), `theme.GRAD` (gradient/error, red), `theme.ATTN` (attention, purple), `theme.OUTPUT` (output/prediction, green), `theme.HIGHLIGHT` (emphasis, yellow), `theme.MUTED` (secondary).
- Before a shot ends, FadeOut the elements that belong only to this shot. Then elements do not pile up.
- To change the value of a `DecimalNumber`, use `ChangeDecimalToValue`. Do not use `.animate.set_value`.
- Calculate the numbers in the images with the code in `../code/` (load it with `importlib`). Do not copy them by hand.

### 4.3 Render and check (necessary)

```bash
bash chapters/NN-slug/video/build.sh --preview    # 480p sample, about 2 minutes
```

- Read the delivery check ("交付检查") at the end of the terminal output. `problems` must be empty. The length must be 5–10 minutes (the sample also reports the length).
- Open the screenshots from the middle of each shot in `video/out/frames/`, and check the layout. You can join them into a grid with ffmpeg and look at the grid with the Read tool:
  `ffmpeg -i a.png -i b.png -i c.png -i d.png -filter_complex "[0][1]hstack[x];[2][3]hstack[y];[x][y]vstack" grid.png`
- Check for these problems: text that overlaps, text outside the frame, content in the subtitle area, headings on top of each other, and lines outside the axes. Fix them until the frames are clean.
- **Do not** render the full 1080p video (the main process renders all videos at the end).

## 5. Self-check skill (`.claude/commands/chNN-<slug>.md`)

Use `.claude/commands/ch01-linear.md` as the model:

- Front matter with a `description`: `"Chapter N self-check: <topic> (第 N 章自检：<中文主题>)"`.
- Write the skill in English. Include the **Language** paragraph of the Chapter 1 skill: use the language of the learner, the Chinese text of the chapter is in `README.zh.md`, and write short, clear sentences (see `docs/STYLE_GUIDE.md`).
- 4 levels of check questions (concept → intuition → find the problem → transfer), with an expected answer for each level.
- At the end, point to the next chapter.

## 6. CS336 mapping table (Spring 2026, checked against <https://cs336.stanford.edu/>)

Spring 2026 lectures: 1 Overview and tokenization; 2 PyTorch and resource accounting; 3 Architectures and hyperparameters; 4 Alternatives to attention, and MoE; 5 GPU/TPU; 6 Kernels and Triton; 7, 8 Parallelism; 9, 11 Scaling laws; 10 Inference; 12 Evaluation; 13 Data (sources and data sets); 14 Data (filtering, deduplication, mixing, synthetic data); 15 Mid-training/post-training (SFT/RLHF); 16 Post-training RLVR; 17 Alignment and multimodality; 18, 19 Guest lectures. Assignments: A1 Basics, A2 Systems, A3 Scaling, A4 Data, A5 Alignment and Reasoning RL (optional part 2: DPO).

| Chapter of this course | Content for "Go deeper: CS336" |
|---|---|
| 7 | Lecture 1 (overview and tokenization), Lecture 2 (resource accounting); the BPE part of Assignment 1 |
| 8 | Lecture 3 (architecture); Assignment 1 |
| 9 | Lecture 3 (architecture and hyperparameters); Assignment 1 |
| 10 | Lecture 10 (inference); the decoding part of Assignment 1 |
| 11 | Lecture 12 (evaluation) |
| 12 | Lectures 9 and 11 (scaling laws); Assignment 3 |
| 13 | Lectures 13 and 14 (data); Assignment 4 |
| 14 | Lecture 5 (GPU), Lecture 6 (kernels/Triton), Lectures 7–8 (parallelism); Assignment 2 |
| 15 | Lecture 14 (mixing), Lecture 15 (mid-training); partly covered |
| 16 | Lecture 15 (SFT); Assignment 5 |
| 17 | Lecture 14 (synthetic data), Lecture 15; partly covered |
| 18 | Lecture 15 (RLHF); optional part 2 of Assignment 5 (DPO) |
| 19 | Lecture 16 (RLVR); Assignment 5 |
| 20 | Lecture 10 (inference), Lecture 12 (evaluation) |
| 21, 25 | Lecture 10 (inference) |
| 22, 23 | Lecture 4 (alternatives to attention); for the rest, write "CS336 does not go deep into this topic" |
| 24 | Lecture 4 (MoE) |
| 26 | Lectures 3 and 4; for the rest, write "CS336 does not go deep into this topic" |

For all links, write <https://cs336.stanford.edu/> (the course page has the slides and the YouTube recording of each lecture).

## 7. Collaboration rules (when many people or agents work in parallel)

- Put temporary files in the `video/out/` folder of your own chapter, or in a subfolder with the name of your chapter in the draft folder. Do not use the same file names as other agents.
- Change only the folder of your own chapter and its skill file. If you must change `video_kit/`, `zero/`, or files of other people, write the change in your report. Do not change these files directly.
- **Do not git commit** (the main process commits everything).
- Add new references to the references of your chapter, and list them in your report. The main process merges them into `references.md`.
- In your report, write clearly: what you finished, the commands that you ran and their results, the length of the sample video, and the items that are still "to be verified".
