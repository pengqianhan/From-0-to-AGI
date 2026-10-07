---
description: "Chapter 12 self-check: scaling laws and experiment design — C≈6ND, Chinchilla, overtraining, tune first then fit, extrapolation and held-out tests, gate 1, budget decisions (第 12 章自检：Scaling Law 与实验设计——C≈6ND、Chinchilla、过训练、先调参再拟合、外推与留出检验、闸门 1、预算决策)"
---

# Chapter 12 self-check: scaling laws and experiment design

The learner typed `/ch12-scaling-laws`. They finished Chapter 12 (`chapters/12-scaling-laws/`). Help them check if they understand it. Do not give them the answers.

**Language**: Use the language of the learner. If the learner writes in Chinese, ask the questions and give feedback in Chinese. The Chinese text of the chapter is in `README.zh.md`. Write short, clear sentences (see `docs/STYLE_GUIDE.md`).

**Method**: Ask the questions below one at a time. Wait for the learner to answer in their own words. Then give feedback: tell them what is correct, what is not correct, and how to make their understanding deeper. Do not ask all the questions at the same time. When a question needs numbers, tell the learner to run the scripts in `chapters/12-scaling-laws/code/` themselves. Do not calculate the numbers for them.

---

## Questions (from easy to difficult)

**Level 1: do the arithmetic (concept)**

Ask the learner:
> Why does training on one token take about 6N floating-point operations? Where do the 2 and the 4 in the "6" come from? The main-line model (0.69B, sequence length 4096) actually needs 6.96 GFLOPs per token. What is the extra 40% above 6N?

Expected answer: in the forward pass, each parameter does one multiply-add = 2 FLOPs. The backward pass calculates one gradient for the input and one for the weights (in Chapter 4, the backward pass of y = Wx is two matrix products) = 4 FLOPs. The total is 6N. The extra part is the two matrix products in attention that have no parameters, QKᵀ and AV: 12·d_attn·T for each layer and each token. It grows with the sequence length (for the main line, with q_dim 2048 and T 4096, it is 40%). Extra credit: this convention does not halve the cost for the causal mask, so the MFU value depends on the convention. `01_flops.py` uses FlopCounterMode, and the measured values are equal to the formula in every digit.

---

**Level 2: Chinchilla and overtraining (intuition)**

Ask the learner:
> Chinchilla says that the compute-optimal ratio is about 20 tokens per parameter. But Qwen3-0.6B trained on 36T tokens (about 60,000 tokens per parameter). Which one is wrong? Why does the main-line model use 0.69B and about 400B tokens, and not the "optimal" 3.6B and 77B for the same compute?

Expected answer: neither is wrong. They answer different questions. Chinchilla minimizes the loss only for the **training** compute. A deployed model also has an inference cost (about 2N FLOPs for each generated token). The more tokens the model must serve, the smaller the model should be and the longer it should train (part ③ of `02_chinchilla.py`). The main line also has hard constraints: size ≤ 0.8B (to compare with Qwen3.5-0.8B in the same class), and usable on devices. The cost is a slightly higher loss than the compute-optimal choice (about 3.5% higher with the Epoch replication coefficients; for the same loss, the optimal allocation needs only about 40% of the compute). This is a trade with a known price.

---

**Level 3: why small experiments can mislead (find the problem)**

Ask the learner:
> In the mini ladder, suppose that all sizes use the learning rate 0.02 that was tuned on the smallest model. What do you see? What wrong conclusion about "does a larger model help" does this give? What is the correct method?

Expected answer: a larger model has a smaller best learning rate. With the learning rate of the small model, the larger models are worse than the small models (`03_lr_sweep.py`: s3 and s4 are 0.19–0.24 bit/byte worse than when tuned, and worse than s1 and s2). A scaling law fitted on these data says "a larger model does not help". This is the conclusion of Lourie et al. (arXiv:2608.11859): small models are very sensitive to hyperparameters, and a badly tuned ladder distorts the scaling law. For each size, sweep the learning rate first (if the best value is at an edge of the grid, extend the grid), and then fit. Extra credit: the first failure of Delphi also came from a learning-rate rule that did not hold for long training. They fixed the recipe, not the fit.

---

**Level 4: transfer (gate 1)**

Ask the learner:
> Step 2 has started, and you have GPUs. You write the gate 1 report: the fit uses ladder sizes l20m–l150m, and l300m is held out. The actual loss of l300m is outside the 95% interval of the extrapolation, 2% too high. What do you do? Besides the loss, which two other questions must the report answer?

Expected answer: do not delete points, and do not patch the old fit. First find where the recipe fails for larger or longer training (the correction of the learning rate for training length, the batch size, the data, the stability). Fix the recipe and run the ladder again (the method of Delphi). The interval of the extrapolation to the main line becomes wider; write this honestly. The report must also have: (1) the benchmark score extrapolation, with the two-step method: first a scaling law of soft metrics (log probability of the correct answer / bits per byte), then an S-shaped mapping "soft metric → hard score" fitted on public models; do not extrapolate a benchmark that is near random; (2) the recipe validation: (a) apply the post-training recipe to ladder Base models and fit "Base quality → tool-calling score"; (b) apply it to an existing open Base model of the same size to see the upper limit of the recipe. If the prediction does not reach the hard goal, do not start pretraining. See `runs/gate1_report_template.md`.

---

## Rules for feedback

- If the answer is correct: say so. Then ask a deeper "why" question. For example: why must the bootstrap resample by size as groups, and not by single points? (The points from the branches of one training run are correlated.)
- If the answer is not correct: do not give the answer. Give a hint. For example, ask the learner to run `02_chinchilla.py` or look at the table of `03_lr_sweep.py`. Then let them think again.
- If the learner says "I do not know": ask them to guess first. A wrong guess is better than no guess.

When the learner passes all four levels, tell them to continue to Chapter 13 (`chapters/13-data/`). The ladder experiment fixes the "recipe", and the most important part of the recipe is the data. The next chapter shows how to collect, clean, deduplicate, and mix the data, and how to train the tokenizer of the main line.
