# From 0 to AGI: from y = ax + b to state-of-the-art open models

**English** · [中文](README.zh.md)

> *You can outsource your thinking, but you cannot outsource your understanding.*

This course starts with one straight line, `y = ax + b`. It continues to a Transformer that you build yourself, scaling laws, pretraining engineering, SFT, distillation, DPO, GRPO reinforcement learning, and the newest architectures for longer context and a smaller KV cache. **The course teaches only mainstream methods that the field agrees on** (the rule is in [GOAL.md](GOAL.md), Section 2.1). The course is in English and Chinese. Each page has a button or a link to the other language.

Each chapter has three parts:

- **A short text**: You can read it in 15–30 minutes. The order is intuition → formulas → minimal code → summary.
- **A video**: 5–10 minutes. The chapter code calculates every number in the video.
- **Two levels of code**: First, **minimal code** that runs on a CPU in seconds or minutes (`chapters/NN-*/code/`). Second, the **production code** for the same idea in the main-line model ([`zero/`](zero/DESIGN.md)). A parity check makes sure that the two levels give the same result.

The second half of the course follows one **main-line model**. We train a bilingual (Chinese and English) model with about 0.69B parameters from zero, with a compute budget of about 10,000 US dollars. The goal is to be better at **tool calling** than all public models of the same size, Qwen3.5-0.8B included. We report the general benchmarks as they are. The production code is complete, and it runs end to end on a CPU with a tiny configuration. The real training occurs in Step 2, when GPUs are available (see [runs/RUNBOOK.md](runs/RUNBOOK.md)).

## Relation to Stanford CS336

This course is an **introduction and guide** to [CS336 (Language Modeling from Scratch)](https://cs336.stanford.edu/). Chapters 1–6 teach the deep-learning basics that CS336 expects you to know. From Chapter 7, each chapter ends with "Go deeper: CS336", which points to the matching lecture and assignment of Spring 2026. This course adds two main topics: the full process of training and releasing a real model (preregistration, gates, and fair evaluation), and the architecture changes in Part 5.

## How to study

1. Read the **goal** and the text of the chapter README. The video helps. The videos are not published yet, so render them on your computer with the commands below.
2. Run the minimal code in the `code/` folder of the chapter. Compare the output with the numbers in the text.
3. Read "From minimal code to production code" and open the related file in `zero/`.
4. Ask Claude Code the "Guided questions" and do the "Hands-on tasks".
5. Type `/chNN-...` in Claude Code to check yourself (the skills are in `.claude/commands/`).

```bash
# Install the dependencies (uv is necessary: https://docs.astral.sh/uv/)
uv sync                       # text + code
uv sync --extra video         # also render the videos yourself (see "Before you render the videos" below)

# Run Chapter 1
uv run python chapters/01-linear-regression/code/01_fit_line.py

# Render the video of Chapter 1 (add --preview for a 480p sample)
bash chapters/01-linear-regression/video/build.sh

# Tests and end-to-end smoke test of the production code (CPU)
uv run pytest
uv run python -m zero.smoke
```

**Before you render the videos**:

- Install `ffmpeg` and LaTeX (Manim uses LaTeX for formulas).
- Fonts: install a Chinese font (WenQuanYi Zen Hei, Noto Sans CJK SC, or Source Han Sans SC) and the monospace font Noto Sans Mono (for code blocks).
- The first render downloads an offline TTS model (the MeloTTS Chinese–English model of sherpa-onnx) from GitHub to `~/.cache/tts`. After that, it works offline. To use a different folder, set `VIDEO_TTS_MODEL_DIR`. By default, TTS uses all CPU cores. On a shared server, limit the threads, for example with `VIDEO_TTS_THREADS=8`.
- A 1080p video of one chapter takes about 6–8 minutes on a 4-core CPU. A 480p sample with `--preview` is faster. The videos go to `chapters/NN-*/video/out/` (not in git).
- The videos are in Chinese at this time.

## Contents

The "Video" column gives the length of the 1080p video (run `bash chapters/NN-*/video/build.sh` to make it again). When the videos are published, we add the links to each chapter README.

### Part 1: Start from a straight line (NumPy, CPU)

| Chapter | Title | Self-check | Video |
|---|---|---|---|
| 1 | [y = ax + b — Learn "training" from a straight line](chapters/01-linear-regression/README.md) | `/ch01-linear` | 5.2 min |
| 2 | [From scalars to matrices — y = XW + b](chapters/02-from-scalar-to-matrix/README.md) | `/ch02-matrix` | 7.4 min |
| 3 | [Nonlinearity and neural networks — Build a curve from line segments](chapters/03-neural-network/README.md) | `/ch03-neural-network` | 7.9 min |
| 4 | [Backpropagation and automatic differentiation — Let the computer calculate the derivatives](chapters/04-backprop-autograd/README.md) | `/ch04-backprop` | 6.3 min |
| 5 | [Classification and probability — From "predict a number" to "predict a class"](chapters/05-classification-probability/README.md) | `/ch05-classification` | 8.7 min |
| 6 | [Stable training — Initialization, normalization, residual connections, AdamW, and learning-rate schedules](chapters/06-training-stability/README.md) | `/ch06-training-stability` | 9.4 min |

### Part 2: Build a modern Transformer (PyTorch)

| Chapter | Title | Self-check | Video |
|---|---|---|---|
| 7 | [Language modeling and tokenization — From "predict the next character" to byte-level BPE](chapters/07-tokenization-language-model/README.md) | `/ch07-tokenization` | 7.5 min |
| 8 | [Attention — Each position decides where to look](chapters/08-attention/README.md) | `/ch08-attention` | 7.8 min |
| 9 | [The modern Transformer — Build a model that writes text from attention](chapters/09-modern-transformer/README.md) | `/ch09-transformer` | 6.3 min |
| 10 | [Inference — Make the model generate text, and make it fast](chapters/10-inference/README.md) | `/ch10-inference` | 9.4 min |

### Part 3: Train a real model (the main-line model starts)

| Chapter | Title | Self-check | Video |
|---|---|---|---|
| 11 | [Evaluation: set the exam first — What to test, how to score, and what counts as a win](chapters/11-evaluation/README.md) | `/ch11-evaluation` | 8.0 min |
| 12 | [Scaling laws and experiment design — Calculate with small models before you spend money](chapters/12-scaling-laws/README.md) | `/ch12-scaling-laws` | 7.1 min |
| 13 | [Data — From web pages to a training data set](chapters/13-data/README.md) | `/ch13-data` | 8.6 min |
| 14 | [Pretraining engineering — Mixed precision, FlashAttention, data parallelism, and resume from checkpoints](chapters/14-pretraining-engineering/README.md) | `/ch14-pretraining` | 9.6 min |
| 15 | [Mid-training and long context — How to train the last stage, and how to read longer text](chapters/15-midtraining-long-context/README.md) | `/ch15-midtraining` | 8.8 min |

### Part 4: Post-training — Make the base model a useful tool-calling model

| Chapter | Title | Self-check | Video |
|---|---|---|---|
| 16 | [SFT — Teach a base model to answer questions and to call tools](chapters/16-sft/README.md) | `/ch16-sft` | 6.3 min |
| 17 | [Distillation — A small model learns from a large model](chapters/17-distillation/README.md) | `/ch17-distillation` | 6.4 min |
| 18 | [Preference alignment — From RLHF to DPO](chapters/18-preference-alignment/README.md) | `/ch18-dpo` | 6.5 min |
| 19 | [Reinforcement learning — The model learns from its own attempts](chapters/19-reinforcement-learning/README.md) | `/ch19-rl` | 6.6 min |
| 20 | [Release — Report against the preregistration, and run the model on a laptop](chapters/20-release/README.md) | `/ch20-release` | 7.4 min |

### Part 5: Architecture changes — For longer context and a smaller KV cache

| Chapter | Title | Self-check | Video |
|---|---|---|---|
| 21 | [The KV cache budget — Why long context is expensive, and how much each token must store](chapters/21-kv-cache-ledger/README.md) | `/ch21-kv-cache` | 7.0 min |
| 22 | [Local and sparse attention — Look nearby, and keep the distant context](chapters/22-local-sparse-attention/README.md) | `/ch22-local-attention` | 6.8 min |
| 23 | [Linear attention and hybrid architectures — Compress the KV cache into a fixed-size matrix](chapters/23-linear-attention-hybrid/README.md) | `/ch23-linear-attention` | 7.2 min |
| 24 | [Mixture of experts (MoE) — Many more parameters, the same compute for each token](chapters/24-mixture-of-experts/README.md) | `/ch24-moe` | 6.4 min |
| 25 | [Multi-token prediction and speculative decoding — A small model guesses, a large model checks](chapters/25-mtp-speculative-decoding/README.md) | `/ch25-speculative` | 6.4 min |
| 26 | [The state of open models — Put the architectures of the course in one tree](chapters/26-open-model-panorama/README.md) | `/ch26-panorama` | 6.5 min |

## Progress of the main-line model

| Stage | Status |
|---|---|
| Step 1: course + production code (runs on a CPU with a tiny configuration, tests pass) | ✅ Done |
| Stage 6: GPU verification (≤ $50) | 🟡 Done on 1 and 2 RTX 3090 GPUs ([runs/2026-10-01-gpu0-check](runs/2026-10-01-gpu0-check/README.md)); the 8×H100 version is not done yet |
| Stage 7: run the competitor models again and finalize the preregistration ([draft](eval/PREREGISTRATION.md)) | ⏳ Waiting for GPUs |
| Stage 8: scaling-ladder experiments and Gate 1 (a small 3×RTX 3090 version is in progress: [runs/ladder-3090](runs/ladder-3090/README.md)) | 🔄 Paused |
| Stage 9: pretraining, mid-training, and Gate 2 | ⏳ Waiting for GPUs |
| Stage 10: post-training, Gate 3, and release | ⏳ Waiting for GPUs |
| Stage 11: put the real results into Parts 3 and 4 | ⏳ Waiting for GPUs |

The report of Step 1 (differences from the plan, parts not verified on a GPU, cost of Step 2, and open decisions) is in [runs/STEP1_REPORT.md](runs/STEP1_REPORT.md). The runbook, cost estimates, and cost ledger of Step 2 are in [runs/RUNBOOK.md](runs/RUNBOOK.md), [runs/ledger.md](runs/ledger.md), and [runs/RELEASE_CHECKLIST.md](runs/RELEASE_CHECKLIST.md).

## Repository structure

```
chapters/NN-slug/     each chapter: README.md (English), README.zh.md (Chinese), code/ (minimal code), video/
zero/                 production code of the main-line model (design notes: zero/DESIGN.md)
configs/              tiny (CPU smoke test) / ladder (scaling ladder) / main (main-line training) configurations
tests/                tests of the production code (uv run pytest)
eval/                 preregistration draft, list of competitor models
runs/                 runbook, cost ledger, release checklist, and experiment records of Step 2
video_kit/            video tools for all chapters (colors, offline narration, shot alignment, subtitles, checks)
site/                 the course website (MkDocs; English by default, with a button for Chinese)
docs/STYLE_GUIDE.md   writing rules for English and Chinese (based on ASD-STE100), and the glossary
docs/CHAPTER_GUIDE.md rules for the structure of a chapter
assets/tiny_corpus/   a tiny offline corpus (with license notes)
GOAL.md               the full goal of the course and the main-line model
references.md         references
```

## Notes

- An offline TTS model (sherpa-onnx + the MeloTTS Chinese–English model) speaks the narration of the videos. **Nobody has listened to the pronunciation and speed yet.** A person must listen to each video before release.
- In Parts 3 and 4, the numbers marked "tiny-configuration demo" come from real CPU runs of a tiny model with about 1M parameters. They show only that the code works. They do not show the quality of the main-line model.
- The smoke-test numbers in Chapters 11 and 16–20 come from a run before we fixed the tool-calling grader. After the fix, the stages up to SFT gave the same numbers, and the numbers from distillation onward changed (each chapter explains this in "Main-line progress").
- In the production code, each path that a CPU cannot test (multiple GPUs, FlashAttention kernels, BF16, and others) had the label "not verified on a GPU yet". In 2026-10, we verified these paths on 1 and 2 RTX 3090 GPUs (BF16, FlashAttention, DDP / FSDP2, resume from checkpoints, compile, post-training paths, and others). The results are in [runs/2026-10-01-gpu0-check](runs/2026-10-01-gpu0-check/README.md), and the labels now give the exact status. The throughput and MFU on 8×H100 with NVLink, and the memory for 32K sequences, are still open for Step 2.
