# Step 1 completion report (GOAL.md Section 10, Stage 5)

**English** · [中文](STEP1_REPORT.zh.md)

Date: 2026-09-30. Branch: `claude/transformer-learning-roadmap-euhvze`.

## 1. Status of the course

All 26 chapters are complete. Each chapter has these parts:

- **Quick-read text** (`chapters/NN-*/README.md`). It has the sections "From minimal code to production code", "Adopters and sources", "Guided questions", and "Hands-on tasks". From Chapter 7, it also has "Go deeper: CS336".
- **Minimal code** (`code/`). It runs on a CPU in a few seconds to a few minutes. All numbers in the text come from its real output.
- **Video source code**: `video/script.md` (the fact list + the storyboard and the narration), `video/scenes.py`, and `video/build.sh`. All 480p sample videos are rendered, and we checked the layout of each shot. The delivery check gives an empty `problems` list. Each video is 5–10 minutes long (the root README gives the length of each chapter).
- **Self-check skill** (`.claude/commands/chNN-*.md`).

The 1080p final videos of all 26 chapters are rendered (`bash chapters/NN-*/video/build.sh`). All of them passed the delivery check: 1920×1080, an audio track, a peak of about −6 dB, an empty `problems` list, and a length of 5.2–9.5 minutes. The subtitles `video/subtitles.srt` are committed. GOAL.md Section 7 says that the MP4 files do not go into git. After the publication on Bilibili / YouTube, write the links back into the README of each chapter.

Production code (`zero/`):

| Check | Result |
|---|---|
| `uv run pytest` | 324 passed |
| `uv run ruff check zero tests` | Passed |
| `uv run python -m zero.smoke` (CPU, the tiny configuration with about 1.3M parameters) | All 10 stages passed (data and tokenizer → pretraining → mid-training → SFT → distillation → DPO → GRPO → evaluation → HF/GGUF export (runs in llama.cpp) → demo), 315 seconds on a single thread |

## 2. Differences from GOAL.md (you must know these)

1. **Narration TTS**: the proxy of the build environment blocked the WebSocket connection of edge-tts. We changed to the offline sherpa-onnx + MeloTTS model, which reads mixed Chinese and English (MIT). `video_kit/tts.py` keeps the edge-tts backend. When the network allows it, set `VIDEO_TTS=edge` to change back. **No person has listened to the narration yet.** Listen to all of it before the publication. The pronunciation of technical names is the most probable problem.
2. **Data**: the build environment could not access huggingface.co. The minimal code and the smoke test use the small corpus in `assets/tiny_corpus/`. The sources and licenses are in `assets/tiny_corpus/LICENSES.md`.
3. **The distillation teacher of the smoke test** is the tiny SFT model itself, because we could not download open weights. This verifies only the path.
4. **The Stage 0 pilot did not stop to wait for confirmation**: we followed your instruction "continue until the final course is complete, then give it to me". Thus we did all of the work without a stop. If the style of Chapter 1 needs changes, the other chapters will change with it.
5. **Fix of the grader**: when we wrote Chapter 17, we found two bugs in `tool_env`. Bug 1: "full score when the execution result is the same by chance". Bug 2: "for a question that needs no tool, the grader does not check the answer text". We fixed both bugs and added regression tests. The smoke-test numbers in Chapters 11 and 16–20 come from the run before the fix (each chapter says so). For the differences in the rerun after the fix, see "Main-line progress" in each chapter.

## 3. Changes to the main-line model relative to GOAL.md (conclusions of Chapters 12–15)

| Item | GOAL.md first draft | Now | Reason |
|---|---|---|---|
| Pretraining tokens | 500B | **About 400B** (`max_steps = 762940`) | At MFU 0.4, 500B costs about $6.1K, which is above the $5K line. 400B costs about $4.9K. The WSD stable phase can stop at any time. The final value is set at Gate 1, after we measure the MFU |
| Batch for each GPU | micro 8 × accumulation 2 | **micro 4 × accumulation 4** (each step is still 524,288 tokens) | With DDP, micro 8 needs about 114 GiB for each GPU and does not fit. micro 4 needs about 64 GiB. We also added a switch for activation checkpointing (about 27 GiB) |
| Optimizer | AdamW | AdamW is the default. **Muon goes into the ladder experiments for a direct comparison with AdamW** | Muon is verified as a consensus method (Kimi K2, GLM-4.5, DeepSeek-V4). It was ahead on the mini ladder |
| Vocabulary | To be decided | 65,536, byte-level BPE, Qwen3.5 pre-tokenization regex | The vocabulary sweep and the regex comparison of Chapter 13 |
| Architecture | Dense GQA | No change. MLA, sparse attention, and hybrid linear attention are all verified as consensus methods. But for 0.69B / 32K, their benefit is too small for their risk. Chapters 21–23 and 26 give the reasons | |

The item-by-item conclusions of the consensus verification are added below the table in GOAL.md 2.1.

## 4. Parts not verified on a GPU yet

> **2026-10 update**: the list below is the record from the end of Step 1. We keep it as it was. Since then, we verified these items on one and on two RTX 3090 GPUs: BF16, FlashAttention (with `enable_gqa=True`, the default path is Flash), `torch.compile`, activation checkpointing, DDP / FSDP2 and resume across GPUs (2 GPUs), the single-GPU CUDA paths of SFT / distillation / DPO / GRPO, and Muon. See [`runs/2026-10-01-gpu0-check/`](2026-10-01-gpu0-check/README.md). 32K does not fit on GPUs with 24 GB (one GPU, 2 GPUs, or 3 GPUs with FSDP). vLLM, BFCL, and ACEBench are still not verified.

We tested the logic of all code below on a CPU, but the code **did not run on a GPU**. Section 2 "Stage 6" of `runs/RUNBOOK.md` lists the verification command and the pass criterion for each item, with a budget of ≤ $50. The items are:

- BF16 autocast, the FlashAttention backend of SDPA (to be verified: can `enable_gqa=True` use Flash?), and `torch.compile`;
- Multi-GPU DDP, FSDP2, and the consistency of resume across GPUs; the GPU memory and the throughput of activation checkpointing on a GPU;
- The GPU memory of training with 32K long sequences (the YaRN configuration);
- The paths of SFT / DPO / GRPO on a GPU (DPO and GRPO are single-process implementations now; if the GRPO throughput is not enough, change to verl, and do a parity check first);
- vLLM with the exported HF model, and the `hermes` tool-call parser;
- The BFCL `ZeroFCHandler` (`zero/eval/bfcl.py`) and the ACEBench adapter layer;
- The multi-GPU implementation of Muon (`zero/train/muon.py`).

## 5. Expected cost of Step 2 (H100 SXM at $2.5 for each GPU-hour, MFU 0.4)

| Stage | Work | Estimate | Approval necessary (> $100) |
|---|---|---:|---|
| 6 | GPU verification + MFU measurement | ≤ $50 | No |
| 7 | Opponent reruns, final preregistration | $60–200 | Yes (when above $100) |
| 8 | Ladder experiments (4 sizes < $40) + ablations, recipe validation | ≤ $1,200 | Yes |
| 9 | Pretraining, about 400B tokens | ~$4,900 | Yes |
| 9 | Mid-training, about 26B tokens | ~$320 | Yes |
| 9 | Long-context extension to 32K, about 4.2B tokens | ~$196 | Yes |
| 10 | SFT | ~$17 | No |
| 10 | Distillation, DPO, GRPO | Estimate after the measurements of Stage 6; budget ~$1,500 | Yes |
| 10 | Final evaluation and release | ~$200 | It depends |
| — | Part 5: architecture comparison with about 100M parameters (optional) | ~$400 | Yes |
| **Total** | | **About $8,500–9,000** | Inside $10K, with a reserve of about $1,000 |

The reserve for pretraining is small. If the measured MFU is lower than 0.391, 400B tokens cost more than the $5K line. Then, at Gate 1, reduce the number of tokens or change to cheaper GPUs.

## 6. Decisions for you

1. **Preregistration (the draft `eval/PREREGISTRATION.md`)**:
   - For E1, we use BFCL V4 without the Agentic category (its web search depends on SerpAPI and is not reproducible), and normalize again with the official weights. Do you agree?
   - Decoding parameters: greedy decoding for all models, or the recommended sampling parameters of each opponent?
   - The maximum generation length for opponents in thinking mode.
   - The freeze date and the final opponent list (`eval/opponents.md`).
2. **Learning rate of the long-context stage**: mid-training anneals the learning rate to 0. After that, does the long-context stage do a new warmup to 1e-4? Qwen3, Llama 3, DeepSeek-V3, and SmolLM3 do this differently (see Chapter 15). Also decide the order of the two stages.
3. **Approval**: mid-training (~$320) and long context (~$196) each cost more than $100, so they need your approval. The same is true for pretraining and post-training.
4. **License of the model weights**: Chapter 20 lists the options, for example Apache-2.0.
5. **Distillation teacher**: the licenses allow these candidates: the Qwen3 / Qwen3.5 series (Apache-2.0), DeepSeek-R1 / V4 (MIT), GLM-5, MiMo-V2-Flash (MIT), and gpt-oss (Apache-2.0). The terms of Gemma and Llama pass to the student, so we do not use them. The main-line vocabulary is different from all of these vocabularies, so only sequence-level distillation is possible (Chapter 17).
6. **GPU environment**: Stage 6 needs 8×H100 (or equivalent GPUs) for about 1.5 hours.

## 7. Main items that are still marked "to be verified"

- The dense BF16 peak of the H100 SXM: we use 989.5 TFLOPS (the denominator of the cost estimates);
- Chapter 26: some details in the Kimi-K3 technical report; the parameter count of GLM-5.3, which we copy from the model card of GLM-5; the MTP configuration of MiniMax-M3;
- Is clip-higher in GRPO a consensus method yet? Do the newest flagship models still use DPO?
- The licenses of some data sets: SmolTalk2; the terms of the data-generation models of xLAM / ToolACE / Hermes; some subsets of the Tülu 3 preference mixture do not allow commercial use.
