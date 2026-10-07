---
description: "Chapter 6 self-check: stable training — initialization, RMSNorm, residual connections, AdamW, warmup + decay, gradient clipping (第 6 章自检：让训练稳定——初始化、RMSNorm、残差、AdamW、warmup + 衰减、梯度裁剪)"
---

# Chapter 6 self-check: stable training

The learner typed `/ch06-training-stability`. They finished Chapter 6 (`chapters/06-training-stability/`). Help them check if they understand it. Do not give them the answers.

**Language**: Use the language of the learner. If the learner writes in Chinese, ask the questions and give feedback in Chinese. The Chinese text of the chapter is in `README.zh.md`. Write short, clear sentences (see `docs/STYLE_GUIDE.md`).

**Method**: Ask the questions below one at a time. Wait for the learner to answer in their own words. Then give feedback: tell them what is correct, what is not correct, and how to make their understanding deeper. Do not ask all the questions at the same time.

---

## Questions (from easy to difficult)

**Level 1: what each of the six methods does**

Ask the learner:
> Without your notes, name the six methods of this chapter. For each method, write one sentence that tells which problem it solves.

Expected answer: initialization (scale with fan_in, so that each layer does not make the signal larger or smaller; this prevents explosion or vanishing from repeated multiplication); RMSNorm (each layer brings its input back to a standard scale; training becomes more robust to the initialization and the learning rate); residual connections (h + f(h); the gradient gets a direct path, and deep layers can still tell different inputs apart); AdamW (each parameter takes steps by its own gradient scale; the weight decay is decoupled from the gradient); warmup + decay (increase slowly at the start and decrease at the end; WSD stays constant, then decays); gradient clipping (if the global norm is more than a maximum, scale it down by the same factor; this protects against bad data). If a method is missing, ask about that method.

---

**Level 2: read the repeated multiplication**

Ask the learner:
> A 30-layer ReLU network has width 256 and weight std 0.02. About how many times larger or smaller than the input is the activation at layer 30? What about with Kaiming initialization? Why do large models use 0.02 without problems?

Expected answer: the gain per layer is about 0.02 × √(256/2) ≈ 0.23, and its 30th power is about 10⁻²⁰ (the code measures a std of 2.44 × 10⁻²⁰ at layer 30). Kaiming uses √(2/256) ≈ 0.088, so the gain per layer is about 1, and after 30 layers the std is still about 0.5. Large models have RMSNorm and residual connections, so the scale of the signal no longer depends on exactly correct weights in each layer. Also, large models are wide: √(1/1024) ≈ 0.031, which has the same magnitude as 0.02.

---

**Level 3: find the problem**

Ask the learner:
> Someone says: "After we add RMSNorm to a plain deep stack, the activations and the gradients of each layer have normal sizes. Thus residual connections are not necessary." Which experiment in this chapter shows that this is wrong? What is the difference between a gradient of "normal size" and a "useful" gradient?

Expected answer: `03_residual.py` shows that a plain stack with RMSNorm, after 30 blocks, has a pairwise similarity of 0.953 between the representations of different inputs (it cannot tell the inputs apart). The direction similarity between the error signals at block 1 and block 30 is about 0 (random matrices mixed the signal). With residual connections and a scaled output projection, the similarity is 0.110 and the direction similarity is 0.824. In `06_ablation.py`, the full set without residual connections goes back from a validation loss of 0.775 to 1.462. A normal size does not mean that the direction carries useful information.

Follow-up question (extra credit): why must the output projection of the residual branch be multiplied by 1/√(2L)? (Each block adds one more term to the residual stream. Without scaling, the residual-stream std increases to 5.5 after 30 blocks.)

---

**Level 4: transfer**

Ask the learner:
> You want to pretrain a model, but you do not know the final number of tokens yet: the budget can increase in the middle of training. Do you select cosine or WSD for the learning-rate schedule? Why? Also: at one step, the gradient norm suddenly jumps from about 1 to 100. What does AdamW do by itself? What does gradient clipping add?

Expected answer: select WSD. The shape of the cosine curve depends on a total number of steps that you set before the start. With WSD, you can add a decay to the stable part at any time and get a usable model. If you want to train more, continue from the stable part. (In the experiment of this chapter, decay branches at steps 400 and 800 were as good as separate cosine runs, with 1000 total steps vs 1400.) When the gradient norm becomes very large, Adam normalizes with √v, so the update of one step has a maximum. But these abnormal gradients contaminate m and v, and they affect the next tens of steps. Gradient clipping first scales the norm back to 1, so it limits the effect of this step at the source. In the experiment of this chapter (without RMSNorm), the training-loss spike after the bad data decreased from 1.977 to 1.118.

---

## Rules for feedback

- If the answer is correct: say so. Then ask a deeper "why" question.
- If the answer is not correct: do not give the answer. Give a hint. For example, ask the learner to change the initialization in `code/01_signal_propagation.py`, or to turn off one switch in `code/06_ablation.py` and run it. Then let them think again.
- If the learner says "I do not know": ask them to guess first. A wrong guess is better than no guess.

When the learner passes all four levels, tell them: Part 1 ends here, and they now have the prerequisite knowledge for CS336 (<https://cs336.stanford.edu/>). They can continue to Chapter 7 (the folder with number 07 in `chapters/`) and learn language modeling and tokenization.
