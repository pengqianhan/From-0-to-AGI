---
description: "Chapter 14 self-check: pretraining engineering — three budgets (compute, memory, time), BF16 mixed precision, tiles and online softmax in FlashAttention, gradient accumulation and activation checkpointing, DDP and FSDP, MFU, loss spikes, resume from checkpoints (第 14 章自检：预训练工程——算力/显存/时间三本账、BF16 混合精度、FlashAttention 的分块与 online softmax、梯度累积与激活检查点、DDP 与 FSDP、MFU、loss spike、断点续训)"
---

# Chapter 14 self-check: pretraining engineering

The learner typed `/ch14-pretraining`. They finished Chapter 14 (`chapters/14-pretraining-engineering/`). Help them check if they understand it. Do not give them the answers.

**Language**: Use the language of the learner. If the learner writes in Chinese, ask the questions and give feedback in Chinese. The Chinese text of the chapter is in `README.zh.md`. Write short, clear sentences (see `docs/STYLE_GUIDE.md`).

**Method**: Ask the questions below one at a time. Wait for the learner to answer in their own words. Then give feedback: tell them what is correct, what is not correct, and how to make their understanding deeper. Do not ask all the questions at the same time.

---

## Questions (from easy to difficult)

**Level 1: the three budgets**

Ask the learner:
> You train a model with 689.5M parameters. About how much memory do "parameters + gradients + optimizer state" use on each GPU? Why this number? An estimate shows that micro batch 8 × 4096 does not fit on an 80 GB GPU. What is the largest part?

Expected answer: 16 bytes per parameter (FP32 parameter 4, gradient 4, AdamW m and v 4 each), about 10.3 GiB. The largest part is the activations that the forward pass saves for the backward pass. They are proportional to micro batch × sequence length (in the main line, about 92,840 bytes per token per layer; the estimate at micro batch 8 is about 88 GiB). Extra credit: "use gradient accumulation to keep the tokens per step and make the micro batch smaller", or "activation checkpointing".

---

**Level 2: intuition for BF16**

Ask the learner:
> BF16 and FP16 both have 16 bits. Why does large-model training use BF16? If BF16 works, why does the optimizer still keep FP32 weights?

Expected answer: BF16 has 8 exponent bits (the same range as FP32). It does not overflow to inf like FP16, and small gradients do not underflow to 0, so it does not need loss scaling. The cost is only 7 mantissa bits, so the precision is low. A weight update η·g is often several orders of magnitude smaller than the weight, and BF16 rounds it away (`02_precision.py`: w = 1.0 minus 10⁻³, 1000 times, is still 1.0). Thus the master weights, the gradient accumulation, and the optimizer state stay in FP32. Only the matrix multiplications run in BF16.

---

**Level 3: why FlashAttention is correct and fast**

Ask the learner:
> You calculate the softmax block by block. The maximum in the first block is 3. In the second block, the maximum becomes 5. What do you do with the sum of exponentials so far? FlashAttention calculates more than naive attention (the backward pass calculates P again). Why is it faster?

Expected answer: the old sum of exponentials uses 3 as its reference. Multiply it by exp(3 − 5) to correct it to the new reference (also multiply the output O by this factor). After the scan, divide by l. The result is the same as the standard softmax (exact, not an approximation). It is faster because attention is memory-bound. The naive method writes the T×T matrices S and P to GPU memory and reads them back, and most of the time goes into these reads and writes. With tiles, only small blocks are processed in the on-chip cache, and the backward pass stores only one logsumexp per row. The extra calculation is much cheaper than the reads and writes that it saves. If the learner says "FlashAttention is approximate attention", correct them.

---

**Level 4: transfer — data parallelism and resume**

Ask the learner:
> You train with DDP on 8 GPUs and save a checkpoint at step 10,000. Then you can rent only 4 GPUs. (1) To keep the training dynamics the same, how must you change the micro batch per GPU or the gradient accumulation? (2) Why does zero refuse an "exact resume" when the number of GPUs changes? (3) After a resume, the loss is different from the uninterrupted run. Which states do you check, and in which order?

Expected answer: (1) The global batch per step = micro batch × accumulation steps × number of GPUs. When the number of GPUs is halved, double the accumulation steps (or the micro batch). Then the tokens per step stay the same, and the definition of the mean gradient does not change. (2) The data loader splits the data by rank (global sample g goes to rank g % world_size), and each rank saves its own read position. When the number of GPUs changes, "which samples are in the next batch" no longer matches. Thus you must give up bit-identical results (for example, split the data again from one global position). (3) The optimizer state (m, v, step count), the learning-rate scheduler, the data position, all RNG states, and the step and token counters. `07_resume.py` shows that each missing item causes a deviation. Extra credit: the checkpoint must be written atomically (temporary folder + rename).

---

## Rules for feedback

- If the answer is correct: say so. Then ask a deeper "why" question (for example, "Why does activation checkpointing make MFU lower, but can make training faster?").
- If the answer is not correct: do not give the answer. Give a hint, and ask the learner to run the related script: `01_step_cost.py` (the budgets), `02_precision.py` (precision), `03_online_softmax.py` / `04_tiled_attention.py` (FlashAttention), `05_memory.py` (memory), `06_ddp_by_hand.py` (DDP), `07_resume.py` (resume). They can also use `uv run python -m zero.tools.memory_calc configs/main/pretrain.toml` to see the memory budget.
- If the learner says "I do not know": ask them to guess first. A wrong guess is better than no guess.

When the learner passes all four levels, tell them to continue to Chapter 15 (`chapters/15-midtraining-long-context/`). After Chapter 15, they can check themselves with `/ch15-midtraining`.
