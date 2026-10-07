# GOAL: From 0 to AGI — from y = ax + b to state-of-the-art open models (text + a video for each chapter + a model that really works)

**English** · [中文](GOAL.zh.md)

> This file is the task description for Claude (Claude Code), the executor. Before you start, read these files completely: this file, `README.md`, `references.md`, `build_notes.md`, `chapter01-matrix-basics/`, and `small-llms-under-5b-2026-08-30/inventory.md`.

---

## 1. Goal

Make this repository into a course **in English first, with a Chinese version**. The course goes from `y = ax + b` to the current state-of-the-art open models. **The course teaches only mainstream methods that the field agrees on (consensus methods).** The course has three deliverables:

1. **A quick-read text**: a reader can read each chapter in 15–30 minutes.
2. **A video for each chapter**: 5–10 minutes.
3. **A main-line model that really works**: with about 10,000 US dollars of compute, train a small bilingual (Chinese and English) model from zero. The model must be better at **tool calling (function calling)** than all public models of the same size, Qwen3.5 included. Report the general benchmark scores honestly. Publish everything: the model, the data recipe, the code, and the intermediate checkpoints.

**The work has two steps**:
- **Step 1: finish the course first. No GPU is necessary.** Write all text, videos, and minimal code of the 26 chapters. Also write all production code of the main-line model. Run the production code on a CPU with a tiny configuration, and make sure that the tests pass.
- **Step 2: train the main-line model after I supply the GPUs.** Do the real training with the gate process of Section 3. Then put the real results back into Parts 3 and 4 (see 2.3).

### 1.1 Two levels of code in each chapter: minimal code first, then production code

- **Level 1: minimal code** (`chapters/NN-*/code/`). It teaches only the one core idea of the chapter. Each program is one file with tens of lines to one or two hundred lines. It depends only on NumPy or PyTorch. It finishes on a CPU in seconds to minutes, and it shows a result at the end. The reader first understands this level.
- **Level 2: production code** (`zero/`). This code is the real implementation of the same idea in the main-line model. It reads all settings from configuration files. It supports multiple GPUs on one machine, resume from a checkpoint, logs, and a full set of tests. You can use it directly to train the main-line model with 0.6–0.8B parameters.
- **The two levels must match**: each chapter has a section "From minimal code to production code" at the end of its text. This section tells, item by item, what the production code does in addition to the minimal code, and why. It points to the related files and functions in `zero/`.
- **Part 1 (Chapters 1–6)**: the production version is the standard PyTorch code, for example `torch.autograd`, `nn.Linear`, and `torch.optim.AdamW`. From Chapter 7, the production code goes into `zero/`.

## 2. Course principles

### 2.1 Teach only consensus methods

A method goes into the main text only if it satisfies at least one of these rules:

- **A. Many adopters**: at least 3 independent leading open model families (Qwen, DeepSeek, Llama, Gemma, Kimi, GLM, MiniMax, Mistral, gpt-oss, OLMo, SmolLM, NVIDIA Nemotron, and others) state clearly that they use the method. The statement must be in the technical report or the model card of their main version.
- **B. Industry standard**: the method is a tool or a process that is already the de facto standard, for example FlashAttention, vLLM, and lm-evaluation-harness.
- **C. Necessary background**: the method is a basic idea that is necessary to understand A or B. Teach it only as background. Examples: RNN for "why we need attention", PPO for GRPO, MHA for GQA.

A method that does not satisfy these rules gets one of two treatments. Either put it in a "Frontier notes" box at the end of the chapter: one paragraph that tells what it is, who uses it, and why it is not a consensus yet. Or do not teach it. When you write each chapter, verify each item. At the end of the chapter, list "adopters + source links".

The first classification is below. This table uses what we knew in September 2026. **When you write, you must verify each item again.**

| Category | In the main text | To be verified (possibly a consensus now) | Not taught, or only one sentence in "Frontier notes" |
|---|---|---|---|
| Architecture | Pre-Norm + RMSNorm, SwiGLU, RoPE, GQA, QK-Norm, tied input and output embeddings for small models, MoE (fine-grained experts + shared experts), MLA, MTP, sliding-window / alternating local-global attention, hybrid linear attention (a few full-attention layers + many linear-attention or state-space layers; Qwen3.5 uses a 3:1 Gated DeltaNet hybrid structure even at 0.8B) | Sparse attention (the DeepSeek DSA/NSA type), auxiliary-loss-free MoE load balancing | RWKV, pure Mamba/pure SSM models, RetNet, ALiBi and other positional encodings that are not mainstream |
| Training | AdamW, warmup + cosine/WSD learning-rate schedule, BF16 mixed precision, gradient clipping, FlashAttention, data parallelism/FSDP, deduplication + model-based quality filtering, synthetic rephrased data, annealing/mid-training, RoPE base-frequency adjustment + YaRN long-context extension, fit a scaling law with small experiments to set the hyperparameters | Muon optimizer, FP8 training | The details of μP, the details of NTK-aware interpolation |
| Post-training | SFT (chat template, loss mask), distillation (SFT on teacher data, logits distillation), DPO, RLHF (reward model + PPO, as background), reinforcement learning with verifiable rewards (RLVR) of the GRPO family | On-policy distillation, improvements to GRPO (DAPO, GSPO, and others) | IPO, KTO, SimPO, ORPO, and other DPO variants |
| Inference | KV cache, speculative decoding, PagedAttention (vLLM), weight quantization (GGUF / INT4, and others) | — | — |

> **Note:** "GPO" in the first requirements means **GRPO**.

**Verification results during the writing of Step 1 (September 2026; the details are in "Adopters and sources" of each chapter)**:

| Method | Conclusion | Chapters |
|---|---|---|
| Muon optimizer, FP8 training | They satisfy rule A, so they go into the main text. The ladder experiments compare Muon and AdamW for the main-line model | 12, 14 |
| MLA | It satisfies rule A (DeepSeek, Kimi, GLM-5, Mistral Large 3), but all these models are large MoE models. The main-line model keeps GQA | 21 |
| Learned sparse attention | The general direction satisfies rule A (DeepSeek, GLM-5, MiniMax-M3, Meituan LongCat). It goes into the main text, with its limits. The specific variants (DSA / MSA / LSA, and others) have not converged, so they go into Frontier notes | 22, 26 |
| Hybrid linear attention | In the main text (Qwen, Kimi, NVIDIA). Only Qwen uses the specific operator Gated DeltaNet | 23 |
| Auxiliary-loss-free MoE load balancing | It satisfies rule A (DeepSeek-V3, GLM-4.5, Nemotron 3), so it goes into the main text | 24 |
| MTP | It satisfies rule A, so it goes into the main text. Draft heads of the Medusa / EAGLE type go into Frontier notes | 25 |
| On-policy distillation | It satisfies rule A (Qwen3, Gemma 2, GLM-5, MiMo, DeepSeek-V4), so it goes into the main text. The main-line model cannot use it, because its vocabulary is different | 17 |
| GRPO improvements | GRPO itself is a consensus. Clip-higher is to be verified. Dr. GRPO and GSPO go into Frontier notes | 19 |

### 2.2 Each problem leads to the next method

The main thread of the course is: "The previous method has a problem. The next method solves that problem."

- A straight line cannot fit a curve → neural networks
- Gradients by hand are too much work → automatic differentiation
- A long sequence is too long to remember → attention
- Inference is too slow → KV cache
- The KV cache is too large → GQA, MLA
- The context must be even longer → sliding window, hybrid linear attention
- The model can talk but does not follow instructions → SFT, DPO
- The model can imitate but cannot solve problems → reinforcement learning

Start each chapter with one linking sentence: "In the last chapter, we … . In this chapter, the problem to solve is … ."

### 2.3 The main-line model in the second half of the course

From Chapter 11, each chapter is one training stage of the main-line model (see the column "Main-line output" in Section 4).

- **Step 1 (course phase)**: In each chapter, write the production code for the stage. Run it for real on a CPU with a tiny configuration. The curves and tables in the text and in the video come from this small run. Label them clearly as "Tiny-configuration demo". In the section "Main-line progress" at the end of the chapter, first write "To be added after GPU training".
- **Step 2 (training phase)**: After the real training, add the real data to "Main-line progress": the training curves, the evaluation table, the cost, and the failures and fixes. If necessary, render again the video shots that use real data.
- Never make up results. If you did not run something, write "Not trained yet".

## 3. The main-line model

### 3.1 Why we define the goal this way

"Better than the current best model of the same size on all benchmarks" is not realistic with the course budget. Some reference points:

| Reference | Investment | Result |
|---|---|---|
| [nanochat](https://github.com/karpathy/nanochat) (2026) | 8×H100 for about 2 hours, about $48 | GPT-2 (2019) level |
| [Puro-2B](https://www.alphaxiv.org/abs/2608.27370) (Tsinghua, 2026-08) | 2B, about 1.4T tokens, about $4.4K–6.9K (RTX 5090 cluster + FP8 + much engineering optimization) | General ability about equal to Qwen2-1.5B (2024) |
| [MobileLLM-R1-950M](https://arxiv.org/abs/2509.24945) (Meta, 2025) | 0.95B, 4.2T tokens, 128 GPUs for about two weeks | Equal to or better than Qwen3-0.6B (which has only 60% of its size) only on reasoning tasks. Still behind on GSM8K |
| Qwen3.5-0.8B (2026, the benchmark model that we must beat) | The pretraining of the Qwen3 small models used 36T tokens, plus distillation from the flagship model and large-scale RL. Qwen3.5 is natively multimodal, with a 262K context | — |

Thus the goal has two levels (3.2 gives the decision method):

- **Hard goal (specific)**: on a preregistered group of tool-calling benchmarks, be ahead of all public-weight models of the same size, Qwen3.5-0.8B included.
- **Soft goal (general)**: on general benchmarks, try to be first among the "fully open recipe" models of the same size (data and code are both public). Report the gap to open-weight-only models such as Qwen honestly.

We chose tool calling as the specific area for three reasons:

1. Tool calling is the most useful use case for small models (on-device agents).
2. A program can score automatically if a call is correct. This fits GRPO in Chapter 19.
3. This ability depends mainly on post-training, not on the amount of knowledge from pretraining. Thus it is the area where a limited budget has the best chance to win.

### 3.2 Preregistration and fair evaluation

A claim of "ahead" must be clear before the evaluation, and it must be possible to check it after the evaluation.

1. **Preregistration**: In Step 1, write a draft of `eval/PREREGISTRATION.md`. In Step 2, before the main-line pretraining starts, fix the opponent list and the freeze date. After I confirm, finalize and commit the file. The commit time is the registration time. The file states:
   - the target benchmarks and their versions
   - the evaluation framework and its version number
   - the prompts and templates, and the decoding parameters
   - the opponent list and the freeze date
   - the criterion for "ahead"

   After that, you can make a change only as an "amendment" that you add with a date and a reason. Do not rewrite the file silently.
2. **Benchmarks**:
   - Specific (hard goal): mainly BFCL (the latest version at the freeze date), plus one tool-calling benchmark that includes Chinese (for example ACEBench; verify its current status). Only report agent benchmarks such as τ²-bench. Do not make them a hard goal.
   - General (report honestly): knowledge (MMLU-Redux / MMLU-Pro), Chinese (C-Eval / CMMLU), math (GSM8K / MATH-500), code (HumanEval+ / MBPP+), instruction following (IFEval). For the Base model, also use the common few-shot benchmarks. Chapter 11 fixes the exact list and writes it into the preregistration.
3. **Opponent list**:
   - Scope: all public-weight models with an official parameter count between 0.7 and 1.3 times the size of our model. All models released before the freeze date count. The parameter count includes the embedding, and it uses the count of the publisher. For a multimodal model, use the parameter count of the language-model part. Start the list from `small-llms-under-5b-2026-08-30/inventory.md`.
   - Benchmark model: Qwen3.5-0.8B is always in the comparison, whatever the final size of our model.
   - At release, check again for new models released after the freeze date. Report them honestly in a section "Opponents added after release", even if they are better than our model.
4. **The same ruler for all models**:
   - We run all opponents again ourselves: the same evaluation framework, the official chat/tool-calling template of each model, and the same decoding parameters. Show the official published scores next to our scores, but do not use them for the comparison.
   - If an opponent has a thinking mode and a non-thinking mode, test both modes. For a claim of "ahead", use the **higher score** of the two modes of the opponent.
5. **Decision criterion**: the lead must be larger than the bootstrap 95% confidence interval to count as "ahead". If it is not, we can only say "tie".
6. **Decontamination**: check all training data (including synthetic data from teacher models) against all evaluation sets for n-gram overlap (for example 13-gram). For tool-calling data, also check if function names and schemas overlap with evaluation sets such as BFCL. Write the check method and the results into the model card.

### 3.3 Default technical plan (the experiments of Chapter 12 make the final decision)

- **Size**: a dense model with about 0.6–0.8B parameters, not more than 0.8B. This makes sure that the benchmark model Qwen3.5-0.8B is not smaller than our model.
- **Architecture**: only consensus components: Pre-Norm RMSNorm, SwiGLU, RoPE, GQA, QK-Norm, tied embeddings, no bias. **The main-line model takes no architecture risk**: at this scale and context length, the differences from the architecture are much smaller than the differences from the data and the post-training. Part 5 explains hybrid linear attention, MoE, and MLA fully with small experiments. They do not go into the main-line model.
- **Tokenizer**: a byte-level BPE that we train ourselves (Chinese + English + code). Chapter 13 measures and decides the vocabulary size. For a small model, watch the parameter cost of the vocabulary: Qwen3.5-0.8B has a vocabulary of 248K tokens, and its embedding alone has about 250M parameters.
- **Data**: use only public data with a license that allows this use. Record the source and the license of each data set.
  - Pretraining candidates: FineWeb-Edu, DCLM, FineWeb-2 (Chinese), Ultra-FineWeb, Stack-Edu, FineMath, and synthetic rephrased data.
  - Mid-training: add high-quality data, instruction data, and data in tool-calling format.
- **Training stages**:
  1. Pretraining, about 0.5–1T tokens. The exact number depends on the final size, the GPU price, and the MFU
  2. Mid-training / annealing
  3. Long-context extension to 32K (tool descriptions and multi-turn dialogs need a long context)
  4. SFT
  5. Distillation
  6. DPO
  7. GRPO in a tool-calling environment
- **Optimizer and precision**: AdamW + BF16 by default. If verification shows that Muon and FP8 satisfy the consensus rule, we can use them after a small comparison in Chapter 12.
- **Teacher models**: for distillation, use only open-weight models whose license allows "use of the outputs to train other models", for example models with an Apache-2.0 or MIT license. Record the name, the version, and the license of each teacher.
- **Code**:
  - The main-line training code is in `zero/` (working name). It grows chapter by chapter from Chapter 7. Its design follows nanochat (one node, readable).
  - The training code does not depend on large frameworks as black boxes. The evaluation uses the official or standard evaluation code.
  - Reference and alternative for reinforcement learning: [verl](https://github.com/verl-project/verl) and the Xiaomi version [XiaomiMiMo/verl](https://github.com/XiaomiMiMo/verl) (it publishes the five types of RL environments and graders of MiMo-V2.6). The GRPO in `zero/` stays a readable implementation of our own. In Step 2, if the throughput of our implementation is not sufficient, we can use verl for the real RL training. But first, run the same small task with verl and with `zero`, and make sure that the two give the same result (a parity check).
  - For this budget, one machine with 8×H100 is sufficient. We do not need multiple machines.

### 3.4 Budget and gates

The total budget is about **10,000 US dollars**. At about $2.5 for each H100 GPU-hour, this is about 4000 GPU-hours. The real rental price decides the final number.

| Use | Budget |
|---|---|
| Chapter 11: evaluate the opponent models again | ~$200 |
| Chapters 12–13: small experiments, data ablations, recipe validation | ~$1,200 |
| Chapter 14: pretraining | ~$5,000 |
| Chapter 15: mid-training + long-context extension | ~$700 |
| Chapters 16–19: post-training (including teacher data generation) | ~$1,500 |
| Chapter 20: final evaluation and release | ~$200 |
| Part 5: small architecture experiments | ~$400 |
| Reserve (reruns, incidents) | ~$800 |

- **All steps that cost money are in Step 2**: Step 1 (course phase) spends no money on GPUs. This includes the opponent reruns of Chapter 11 and the ladder experiments of Gate 1. Do all of them after I supply the GPUs. In Step 1, prepare the scripts, configurations, and cost estimates of these experiments, so that they are ready to run.
- **Spending needs approval**: the executor writes the scripts, estimates the cost, monitors the runs, and analyzes the results. Before any single run with an expected cost of more than $100, send me the cost estimate. Start the run only after I approve it. Record all costs in `runs/ledger.md`: the date, the purpose, the GPU-hours, the amount, and a link to the results.
- **Gate 1 (before the large spending)**:
  1. **Extrapolated prediction**: extrapolate from small scaling-ladder experiments. Predict the loss and the benchmark scores that the main-line Base model can reach with the current recipe and within the budget.
  2. **Recipe validation**, in two paths:
     - (a) Apply the post-training recipe to several small Base models from the ladder experiments. Fit the relation "Base quality → tool-calling score" and extrapolate it.
     - (b) Also apply the recipe to an existing open Base model of the same size. This shows the upper limit of the post-training recipe itself. Use (b) only for validation. Do not publish it as a result.
  3. **I decide**: give me the predictions. Start pretraining only if the prediction reaches the hard goal. If it does not, adjust the recipe or the goal first. Do not force the spending.
- **Gate 2 (end of pretraining)**: compare the real evaluation results of the Base model with the prediction of Gate 1. If the results are clearly lower, find the cause first. Then start post-training.
- **Gate 3 (before release)**: do the final evaluation with the preregistered protocol. If the model does not reach the hard goal, do not claim "ahead". Publish the results honestly. The gap analysis itself becomes course content.

### 3.5 Release items

- **Models**: publish on Hugging Face: the Base model, the SFT model, the final version, and the important intermediate checkpoints. Readers without compute can start the experiments of Chapters 16–19 directly from our Base model.
- **Quantized versions**: GGUF / INT4. They run with llama.cpp or Ollama on a normal laptop.
- **Local demo**: a tool-calling demo that runs locally, for example a command-line assistant that can call local tools such as a calculator, a calendar, and file search. This demo is the direct proof that the model "really works".
- **Model card**: the training data and the licenses, the recipe and the cost of each stage, the preregistration protocol, all evaluation results (including the items where we are behind), the decontamination checks, and the known limits.
- **A complete, reproducible recipe**: data-processing scripts, training configurations, and logs.

## 4. Course outline (the executor can change the split of the chapters a little, but must not remove topics)

> For each chapter, write: the goal (one sentence), the prerequisite chapters, the core formulas, the minimal code, the related production code, and the key images for the video. For each chapter of Parts 3 and 4, also write what the chapter gives to the main-line model ("Main-line output"). In the tables, "Code output" means the level-1 minimal code. "Main-line output" means production code that is written and runs on a CPU in Step 1, and the real training result in Step 2.

### Part 1: Start from a straight line (runs on a CPU, NumPy only)

| # | Chapter | Core content | Code output |
|---|---|---|---|
| 1 | y = ax + b | Data, model, loss (MSE), gradients by hand, gradient descent, a learning rate that is too large/too small | Fit a straight line; plot the loss curve and the parameter path |
| 2 | From scalars to matrices | Multivariate linear regression, vectorization, `y = XW + b`, the shape rule (merge the existing `chapter01-matrix-basics`) | A vectorized version; compare its speed with the loop version |
| 3 | Nonlinearity and neural networks | Why stacked linear layers are still linear, activation functions, MLP | A two-layer MLP fits a nonlinear function |
| 4 | Backpropagation and automatic differentiation | Chain rule, computational graph, a micrograd-style autograd written by hand | A scalar autograd of about 150 lines, and an MLP trained with it |
| 5 | Classification and probability | softmax, cross-entropy, maximum likelihood, why classification does not use MSE | Classification on toy data or MNIST |
| 6 | Stable training | Initialization, normalization (LayerNorm → RMSNorm), residual connections, AdamW, warmup + decay | A comparison experiment: the same network with each of these methods on and off |

### Part 2: Build a modern Transformer (change to PyTorch; one GPU or a CPU)

| # | Chapter | Core content | Code output |
|---|---|---|---|
| 7 | Language modeling and tokenization | Character level → byte-level BPE, next-token prediction, bits-per-byte | A BPE written by hand + a bigram language model |
| 8 | Attention | Derive attention from "weighted average", Q/K/V, scaled dot product, causal mask, multiple heads; RNN only as motivation | Attention from zero, and a visualization of the attention matrix |
| 9 | The modern Transformer | Pre-Norm RMSNorm, SwiGLU, RoPE, QK-Norm, tied embeddings; the GPT-2 structure only as a historical comparison; the full data flow of tensor shapes | Train a small model that generates text (TinyStories or a small sample of FineWeb-Edu) |
| 10 | Inference | Sampling (temperature, top-p), KV cache, GQA (the KV cache is too large → several query heads share one KV head) | Add a KV cache and GQA to the model of Chapter 9; measure the speed and the GPU memory |

### Part 3: Train a real model (the main-line model starts)

| # | Chapter | Core content | Main-line output |
|---|---|---|---|
| 11 | Evaluation: set the exam first | What each benchmark tests, rules for a fair comparison, data contamination, confidence intervals, preregistration | `eval/PREREGISTRATION.md` + the rerun scores of the opponent models |
| 12 | Scaling laws and experiment design | `C ≈ 6ND`, Chinchilla, why small models are "overtrained", fit the loss and the hyperparameters with small experiments, extrapolate to predict the final scores (use the two papers in the references: Delphi and "small-scale experiments") | The decision on the size, the token count, and the hyperparameters of the main-line model + the Gate 1 prediction report |
| 13 | Data | Open data sets, deduplication, model-based quality filtering, synthetic rephrasing, mixture ablations, decontamination, training of the main-line tokenizer | The pretraining data set + the tokenizer |
| 14 | Pretraining engineering | Mixed precision, FlashAttention (how it works: tiling + online softmax; how to use it), data parallelism/FSDP, MFU, loss spikes and training stability, resume from a checkpoint | The main-line pretraining itself |
| 15 | Mid-training and long context | Annealing (the decay phase of WSD + high-quality data), RoPE base-frequency adjustment + YaRN extension to 32K | The main-line Base model (Gate 2) |

### Part 4: Post-training — make the base model a useful tool-calling model

| # | Chapter | Core content | Main-line output |
|---|---|---|---|
| 16 | SFT | Chat template (including the tool-calling format), loss mask, instruction data | The SFT model |
| 17 | Distillation | Why small models become stronger through distillation (Llama 3.2, Gemma, the Qwen3 small models, and DeepSeek-R1-Distill all do this); SFT on teacher data, logits distillation, on-policy distillation (to be verified); teacher licenses | Tool-calling distillation data verified by execution + the distilled model |
| 18 | Preference alignment | First RLHF (reward model + PPO, as background), then derive DPO from the same objective | The DPO model (better general chat quality) |
| 19 | Reinforcement learning | GRPO and verifiable rewards; tool-calling environment (sandbox, simulated APIs), reward design, prevention of reward hacking; the design of the environments and graders follows XiaomiMiMo/verl | The final model |
| 20 | Release | Final evaluation with the preregistered protocol, quantization, local deployment, model card, honest report | The model on Hugging Face + the evaluation report + the local demo (Gate 3) |

### Part 5: Architecture changes — for longer context and a smaller KV cache

> This part does comparison experiments on the course code. It does not change the main-line model. In Step 1, use tiny models with millions of parameters on a CPU. In Step 2, you can optionally run them again with about 100M parameters.

| # | Chapter | Core content | Code output |
|---|---|---|---|
| 21 | The KV cache budget | The memory formula `2 × layers × KV heads × head_dim × sequence length × bytes`, prefill and decode, MQA → GQA → MLA | A KV cache calculator (for the main-line model and several new models) + an MHA/GQA/MLA comparison experiment |
| 22 | Local and sparse attention | Sliding window, alternating local-global attention; the verification result of 2.1 decides if sparse attention goes into the main text or into "Frontier notes" | An implementation and a comparison of sliding-window attention |
| 23 | Linear attention and hybrid architectures | softmax attention → the recurrent form of linear attention → Gated DeltaNet; why "a few full-attention layers + many linear layers" (for example, 3:1 in Qwen3.5) | A recurrent implementation of linear attention + a comparison of the hybrid structure with pure attention |
| 24 | Mixture of experts (MoE) | Routing, load balancing, fine-grained experts + shared experts; why all large models use MoE and small models use it less | A small MoE layer, compared with a dense layer of the same compute |
| 25 | Multi-token prediction and speculative decoding | The MTP training objective; why speculative decoding does not decrease the quality | Speculative decoding for the main-line model with a small draft model; measure the speedup |
| 26 | The state of open models | Analyze 3–4 representative models that are the newest at the time of writing (for example, the newest versions of DeepSeek, Qwen, Kimi, and gpt-oss); draw an architecture evolution tree; put the main-line model in the same table | An architecture comparison table + an evolution tree |

> All content about "the newest" (the choice of methods in Chapters 17 and 19, and all of Part 5): when you write it, check the papers, technical reports, and official model cards online (alphaXiv / Hugging Face / official blogs). Do not write model configuration numbers from memory. At the end of the chapter, give a source link for each specific number.

## 5. Relation to Stanford CS336: an introduction and guide

[CS336 (Language Modeling from Scratch)](https://cs336.stanford.edu/) covers most of the topics of Parts 2 to 4 of this course. This course **does not avoid this overlap. It is clearly an introduction and guide to CS336, in English first, with a Chinese version**:

- **Add the prerequisites**: CS336 expects that readers already know deep learning and PyTorch. Its first lecture starts with tokenization. Chapters 1–6 of this course (y = ax + b → backpropagation → training methods) fill this gap. Then readers with no background can reach CS336.
- **Make the entry easier**: each CS336 assignment needs tens of hours and a GPU cluster. For the same topic, this course gives a 15–30 minute quick read and smaller experiments that run on a consumer GPU. It also publishes the intermediate checkpoints of the main-line model.
- **Add two main threads**:
  - First, the full process of training and releasing a real model: preregistration, gates, and fair evaluation.
  - Second, the architecture changes in Part 5.

  In the other direction, CS336 goes deeper into GPU kernels (Triton), distributed training, and data pipelines. In these areas, this course points directly to CS336.
- **A different form**: English first, with a Chinese version; animated explanation videos (the narration is in Chinese for now); guided questions + Claude Code self-check skills.

**From Chapter 7, each chapter ends with a section "Go deeper: CS336".** This section points to the matching lectures and assignments, as the advanced path. The approximate mapping is below. When you write, check the schedule of the **current** term on cs336.stanford.edu. Write the specific lecture titles and links. Do not write lecture numbers from memory.

| Chapter of this course | Matching CS336 content |
|---|---|
| 1–6 | No direct match (they are prerequisites of CS336). Do not add the section to these chapters. Instead, at the end of Chapter 6, write "After Part 1, you can start CS336" |
| 7 Language modeling and tokenization / 9 The modern Transformer | Tokenization, PyTorch and resource accounting, and architecture lectures; Assignment 1 (Basics: BPE, Transformer, AdamW, and training loop by hand) |
| 8 Attention / 10 Inference | Architecture lecture, inference lecture; Assignment 1 |
| 11 Evaluation | Evaluation lecture |
| 12 Scaling laws | Scaling-law lectures; Assignment 3 (Scaling) |
| 13 Data | Data lectures; Assignment 4 (Data) |
| 14 Pretraining engineering | GPU, kernel, and parallel-training lectures; Assignment 2 (Systems, including FlashAttention in Triton) |
| 15 Mid-training and long context | Partly covered (architecture and data lectures) |
| 16 SFT / 18 Preference alignment / 19 Reinforcement learning | Alignment and RL lectures; Assignment 5 (Alignment: SFT, expert iteration, GRPO) |
| 17 Distillation | Partly covered |
| 20 Release | Inference lecture (quantization and other topics) |
| 21 KV cache / 25 Speculative decoding | Inference lecture |
| 22 Local and sparse attention / 23 Linear attention / 26 State of open models | Architecture lectures (partly covered); for the rest, write "CS336 does not go deep into this topic; see the references of this chapter" |
| 24 MoE | MoE lecture |

## 6. Readers and delivery format

- **Readers**: people who know some Python and high-school math, and who want to understand LLMs systematically. They have Claude Code / Codex near them, so they can ask questions at any time.
- **The text is a quick read**: a reader can read the main text of each chapter in 15–30 minutes. After that, the reader can explain what the chapter teaches, why, and how to calculate it. Intuition comes first. The formulas are sufficient, not more. Each formula comes with a small piece of code that runs.
- **Keep the current teaching method**: keep the sequence of the current `README.md`: "goal → starting code → guided questions → hands-on tasks → `/chXX` self-check skill". It becomes the "deeper" part after the quick-read text. Thus each chapter = **quick-read text** (new) + **exploration and practice** (the current style).
- **Purpose of the video**: a 5–10 minute explanation. It follows the same thread as the text. But it uses animation to show shapes, flows, and changes clearly. Examples: matrix multiplication, gradient-descent paths, attention weights, and the growth of the KV cache. The video does not try to look impressive. It must be **correct and understandable**.
- **Readers without compute can follow the course**:
  - Chapters 1–10 run on a CPU or one GPU.
  - Each of Chapters 11–20 gives "a smaller version that finishes in a few minutes". Readers can also use our published intermediate checkpoints, for example to start SFT from our Base model.
  - The authors (the executor + I) do the full main-line training. The readers read about a real process.

## 7. Repository structure

```
From-0-to-AGI/
├── README.md                  # contents and status table of the new outline + relation to CS336 + progress of the main-line model (Chinese: README.zh.md)
├── GOAL.md                    # this file (Chinese: GOAL.zh.md)
├── references.md              # reference library (append only; do not delete or change existing entries)
├── chapters/
│   └── NN-slug/               # example: 01-linear-regression
│       ├── README.md          # English: quick-read text + guided questions + hands-on tasks + Go deeper: CS336 + references
│       ├── README.zh.md       # Chinese version of README.md (same headings in the same order)
│       ├── code/              # level 1: minimal code; runs with uv run, seconds to minutes on a CPU
│       └── video/
│           ├── script.md      # fact list and sources → storyboard → narration for each shot (write this first)
│           ├── scenes.py      # Manim scene source code
│           ├── build.sh       # one command: render the images + synthesize the narration + make the subtitles + merge into MP4
│           └── subtitles.srt  # made from the same timeline
├── zero/                      # level 2: production code of the main-line model (working name); grows chapter by chapter from Chapter 7
├── configs/                   # three configuration levels: tiny (CPU smoke test), ladder (ladder experiments), main (main-line training)
├── tests/                     # unit tests, correctness tests, and the end-to-end smoke test of the production code
├── eval/                      # preregistration file, opponent list, evaluation scripts, and result tables
├── runs/                      # configuration, log summary, and report of each real training run; ledger.md for costs; runbook for Step 2
├── video_kit/                 # video tools that all chapters share: colors, fonts, formula style, TTS wrapper, merge script
└── .claude/commands/chNN-*.md # one self-check skill for each chapter (same style as ch01-matrix); in English, answers in the language of the learner
```

- Move the existing `chapter01-matrix-basics/` into the new structure (it becomes the new Chapter 2). Rename the `/ch01-matrix` skill to match, and keep its content.
- Do not commit MP4 files, model weights, or data sets (add them to `.gitignore`). Publish the finished videos on Bilibili/YouTube, and the models on Hugging Face. Write the links back into the README of the related chapter.
- Manage the dependencies with `uv` and write them into `pyproject.toml`. Put the dependencies for the videos and for GPU training in optional groups, so that they do not affect people who only read the text.

## 8. Video production rules

Use [awesome-opus-5-5-videos](https://github.com/athemeroy/awesome-opus-5-5-videos) in `references.md` as a reference, especially its "educational explainer" path and template type 2 in `docs/prompt-playbook.zh-CN.md`. Some examples in the "educational explainer" path: a derivatives lesson with Manim + edge-tts, an English lesson with Remotion + CosyVoice, and a VAE math video.

The conclusion from these cases: **in an explainer video, the content is more important than the visuals**. Thus we use this reproducible code-rendering pipeline:

> **Note:** The narration of the videos stays in Chinese for now. The English-first language policy does not change the rules in this section.

1. **Technology stack**:
   - Images: **Manim Community**. It is a natural fit for formulas, matrices, coordinate systems, and animations.
   - Narration: TTS, by default an `edge-tts` Chinese voice. Keep an interface to change to a local TTS such as CosyVoice.
   - Make the subtitles and the narration from the same timeline. Merge everything with `ffmpeg` at the end.
   - Output: 1080p, 16:9, 30fps.
2. **Production order for each chapter video (do not skip a step)**:
   1. **Fact list**: list each factual claim, formula, and number in the video, with its source. Mark the items that the author must confirm.
   2. **Storyboard + narration for each shot** (in `script.md`): for each shot, write the time, the main image, the animation, the narration, and the on-screen text. Write the narration in spoken style. One shot covers one thing.
   3. **Low-resolution sample**: first render the first 30–60 seconds at 480p. Check that no formula is hidden, that the subtitles stay inside the frame, and that the animation is not too fast. Fix the problems, then render the full video.
   4. **Full render + delivery check**: the images and the narration are in sync, the subtitles agree with the timeline, and the audio track exists and does not clip.
3. **One visual language** (in `video_kit/`): all chapters use one set of colors (for example: input = blue, parameter = orange, gradient = red, attention weight = purple), fonts, formula style, and opening and closing sequences. The same concept looks the same in the videos of different chapters.
4. **Honest labels**: the model cannot really "listen" to the finished video. Thus, when you deliver each video, add the label **"旁白发音与语速未经人工试听"** ("a person has not listened to the pronunciation and the speed of the narration") at the end of `script.md`. I remove the label after I listen to the video, and then I publish it. Check how the TTS reads technical names (for example Softmax, RoPE, GRPO). If necessary, write them in a form that is easier to read in the narration script.
5. **The videos agree with real data**:
   - First calculate the values in the video with the code in `code/` of the chapter, and then draw them. Examples of such values: the path of one gradient-descent run, or one attention matrix. Do not make them up.
   - In Step 1, the videos of Parts 3 and 4 use real data from runs with the tiny configuration. Label these images "极小配置演示" (tiny-configuration demo). After the training of Step 2, replace these shots with the real training curves, evaluation tables, and failure cases of the main-line model, and render them again.

## 9. Writing rules

- **Language**: English first, with a Chinese version. The English page is `README.md`, and the Chinese page is `README.zh.md` in the same folder. The two pages have the same headings in the same order. Both languages use the rules of ASD-STE100 Simplified Technical English as a guide. Follow [docs/STYLE_GUIDE.md](docs/STYLE_GUIDE.md) and its glossary. In the Chinese page, give the English term in parentheses at its first use, for example "注意力（attention）".
- **Structure of each chapter**: Goal → quick-read text (intuition → formulas → minimal code → Summary) → From minimal code to production code (see 1.1) → Guided questions → Hands-on tasks → Go deeper: CS336 (see Section 5) → References. The Chinese page uses the Chinese names of these sections (see `docs/CHAPTER_GUIDE.md`).
- **"Main-line progress" in Parts 3 and 4**: put this section after "From minimal code to production code". In Step 1, write the results of the run with the tiny configuration (labeled "Tiny-configuration demo") and "To be added after GPU training". In Step 2, add what the real training did, how much it cost, and the results (including failures and rework).
- Each formula must have a matching line in the minimal code of the chapter.
- Start each chapter with the linking sentence (see 2.2).
- Obey the consensus rule of 2.1: the main text teaches only consensus methods. List the adopters and sources at the end of the chapter.
- For references, first use the sources that are already in `references.md` (CS336, nanoGPT, minimind, Hands-On Modern RL, the mainstream architecture comparison article, Mini Kimi K3, Puro-2B, and others). Add new sources to `references.md` as `CLAUDE.md` specifies.
- The minimal code of all chapters must finish on a CPU in a few minutes. This includes the chapters about pretraining, SFT, DPO, and GRPO (use tiny models and toy tasks). You can also give a larger version for one GPU, but it must not be the main version.

### 9.1 Production code standard (`zero/`)

Step 1 has no GPU, so the trust in the production code comes only from the tests. Requirements:

- **Correctness tests** (`tests/`; `uv run pytest` passes completely on a CPU):
  - Model: load the weights of an open model with the same structure into the `zero` model (for example a Qwen3 small model, a dense model with GQA + QK-Norm + SwiGLU + RMSNorm + RoPE). The logits must agree with the official Hugging Face transformers implementation within the tolerance.
  - Inference: generation with a KV cache and generation without a KV cache give exactly the same result.
  - Tokenizer: encode and then decode gives the original text back. Measure the compression ratio on Chinese, English, and code samples.
  - Loss functions: the SFT loss mask, the DPO loss, and the GRPO advantages and loss all agree with small examples calculated by hand.
  - Training: the loss after resume from a checkpoint is the same as the loss without interruption. A fixed random seed gives reproducible results.
  - Tool calling: template rendering and parsing are inverse operations. The reward function gives the expected scores on positive examples, negative examples, and examples with format errors.
- **End-to-end smoke test**: one command (for example `uv run python -m zero.smoke`) runs the full pipeline on a CPU with `configs/tiny`: data processing → tokenizer → pretraining → mid-training → SFT → distillation → DPO → GRPO → evaluation → export to Hugging Face format and GGUF. The total time is about 30 minutes.
- **Ready to scale up**: the same code runs `configs/main` with only a change of configuration. The code must support multiple GPUs on one machine (torchrun, DDP/FSDP), BF16, FlashAttention, resume from a checkpoint, logs, and checkpoint management.
- **Label GPU-only paths honestly**: code that a CPU cannot test, such as multi-GPU communication, FlashAttention kernels, and FP8, must have the label "not verified on GPU yet". The first task of Step 2 is a GPU verification run of ≤$50 (see Section 10).
- **Ready for Step 2**:
  - `runs/RUNBOOK.md` gives, for each stage: the environment preparation, the start commands, the expected throughput, the metrics to watch, and the gate checklist.
  - `zero/tools/estimate_cost.py` estimates the GPU-hours and the cost from the model configuration, the token count, the GPU type, and the price.
- **Code style**: readability first, type annotations. Write comments at important points, docstrings, and printed output in English (see Section 4 of [docs/STYLE_GUIDE.md](docs/STYLE_GUIDE.md)). Use ruff for format and lint checks.

## 10. Execution stages and acceptance

### Step 1: finish the course (no GPU necessary)

**Stage 0: pilot (when it is done, stop and wait for my confirmation before you continue)**
- Make the folder structure of Section 7 and `video_kit/`. Move the existing Chapter 1 into it.
- Make all of **Chapter 1 (y = ax + b)**: the quick-read text, the code, the self-check skill, all video source code, and the rendered MP4.
- Two-level code samples: for Chapter 1, give the NumPy minimal version and the standard PyTorch version. Also make the two levels of code for Chapter 9 early (code only, no text and no video): a small minimal Transformer, plus the production model in `zero/` and a correctness test that "the logits agree with the official implementation". Then I can see early what the two levels of code look like.
- Update `README.md`: replace it with the contents table of the new outline (the status column shows ✅/🔜). At the start, write the relation of this course to CS336 and the goal of the main-line model.
- Acceptance:
  - `uv run python chapters/01-*/code/*.py` all run;
  - `uv run pytest` passes on a CPU;
  - `bash chapters/01-*/video/build.sh` makes the MP4 from zero;
  - I read the text and watch the video, and I accept the style.

**Stage 1: Parts 1 and 2 (Chapters 1–10)**
- Two levels of code for each chapter. `zero/` gets the model, the tokenizer, and inference with a KV cache. They pass the correctness tests that compare them with the official implementations (parity checks).

**Stage 2: Part 3 (Chapters 11–15)**
- Evaluation framework code + a preregistration draft.
- Ladder experiment scripts, the data-processing pipeline (it runs on small sample data), tokenizer training, the pretrainer, mid-training, and long-context extension.
- For each chapter, do a real run on a CPU with `configs/tiny`.

**Stage 3: Part 4 (Chapters 16–20)**
- SFT, distillation (a teacher data generation script; in the smoke test, a small open model that runs on a CPU is a temporary teacher), DPO, GRPO and the tool-calling environment, evaluation, quantized export, and the local demo.
- The demo in the smoke test uses a tiny model. It only needs to run. Bad quality is normal. Say this honestly.

**Stage 4: Part 5 (Chapters 21–26)**
- Do the comparison experiments first on a CPU with tiny models that have millions of parameters. Keep the version with about 100M parameters for Step 2, as an option.

**Stage 5: end of Step 1**
- The end-to-end smoke test passes, and all unit tests pass.
- `runs/RUNBOOK.md` and the cost estimates are ready. In the text of Parts 3 and 4, each "Main-line progress" section says "To be added after GPU training".
- Give me a report: the status of the course, the parts of the production code that are not verified on a GPU yet, and the expected cost of each stage of Step 2.

### Step 2: train the main-line model (after I supply the GPUs)

**Stage 6: GPU environment verification (≤ $50)**
- Do a short training run on real GPUs: verify multiple GPUs, FlashAttention, BF16, and resume from a checkpoint. Measure the throughput and the MFU, and update the cost estimates with these measurements.

**Stage 7: opponent reruns and the final preregistration**
- Fix the opponent list and the freeze date. Run the opponents again.
- **Stop and wait for my confirmation of the preregistration.** Commit it only after I confirm it.

**Stage 8: ladder experiments and Gate 1**
- Small experiments, data ablations, and recipe validation. Write the Gate 1 report.
- **Wait for my approval of the pretraining budget.**

**Stage 9: pretraining, mid-training, and Gate 2**

**Stage 10: post-training, Gate 3, and release**
- Before the release, wait for my final confirmation.

**Stage 11: put the results back into the course**
- Add the real results to "Main-line progress" in Parts 3 and 4. Replace the video shots that use real data with the real curves, and render them again.
- Part 5, optional: run the comparison experiments again with about 100M parameters (budget about $400).

At the end of each stage: commit the code, update the status in `README.md`, and give me a short report. The report includes the chapters that you finished and the money that you spent. It also lists the items that I must confirm: doubtful facts, narration that a person must listen to, and results that do not agree with the expectation. In Step 1, give a report at the end of each of Stages 1–4. Then continue to the next stage without waiting for my confirmation. You must stop at Stage 0, and at the points in Step 2 that say "wait for me".

**Definition of Done for each chapter**
- [ ] The quick-read text is complete. After reading it, the reader can do what the goal says.
- [ ] All minimal code in `code/` runs on a CPU with `uv run` in a few minutes, and the output agrees with the text.
- [ ] From Chapter 7: the related production code in `zero/` is written, the related tests pass on a CPU, and the section "From minimal code to production code" is written.
- [ ] The guided questions and the hands-on tasks are in place, and the `/chNN-*` self-check skill is in place.
- [ ] `README.md` (English) and `README.zh.md` (Chinese) are both complete, with the same headings in the same order. Both follow `docs/STYLE_GUIDE.md`.
- [ ] `video/script.md` has the fact list with sources, the storyboard, and the narration.
- [ ] `video/build.sh` renders the MP4 again. The length is 5–10 minutes. Overlaps, subtitles, and the audio track are checked.
- [ ] All methods in the text satisfy the consensus rule of 2.1. The adopters and sources are listed at the end of the chapter. Methods that do not satisfy the rule are in "Frontier notes".
- [ ] All content about "the newest" has clickable source links.
- [ ] From Chapter 7, each chapter has a section "Go deeper: CS336". Its lectures and assignments are checked against the current schedule, with links.
- [ ] Parts 3 and 4: at the end of Step 1, the results of the tiny-configuration run are in the text, labeled "Tiny-configuration demo", and "Main-line progress" says that the real results come later. At the end of Step 2, the real training results are in the text, and the costs are in `runs/ledger.md`.
- [ ] The status in the contents table of `README.md` is updated.

## 11. What not to do

- Do not write the quick-read text as a textbook to "cover everything". Put the details that do not fit into the guided questions, for the reader to discuss with Claude Code.
- Do not make up paper conclusions, model configuration numbers, or experiment results. If you are not sure, mark the item "to be verified" and tell me.
- Do not start any single run with an expected cost of more than $100 without my approval.
- Do not present results of the tiny configuration as results of the main-line model. Do not call code "verified" if it was not verified on a GPU.
- Do not train on the test data of any evaluation set. Do not use the preregistered test benchmarks to tune hyperparameters or to pick checkpoints. Make these decisions with our own development set. Run the test benchmarks only at the gates and in the final evaluation.
- Do not use data or teacher models whose license does not allow this use.
- If the model does not reach the preregistered criterion, do not claim "ahead". Do not report only the benchmarks that are good for us.
- Do not commit large files such as MP4 files, model weights, or data sets.
- Do not delete or rewrite existing entries in `references.md`.
