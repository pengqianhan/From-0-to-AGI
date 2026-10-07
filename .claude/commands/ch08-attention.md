---
description: "Chapter 8 self-check: attention — weighted average, Q/K/V, scaled dot product, causal mask, multiple heads and tensor shapes (第 8 章自检：注意力——加权平均、Q/K/V、缩放点积、因果 mask、多头与张量形状)"
---

# Chapter 8 self-check: attention

The learner typed `/ch08-attention`. They finished Chapter 8 (`chapters/08-attention/`). Help them check if they really understand it. Do not give them the answers.

**Language**: Use the language of the learner. If the learner writes in Chinese, ask the questions and give feedback in Chinese. The Chinese text of the chapter is in `README.zh.md`. Write short, clear sentences (see `docs/STYLE_GUIDE.md`).

**Method**: Ask the questions below one at a time. Wait for the learner to answer in their own words. Then give feedback: tell them what is correct, what is not correct, and how to make their understanding deeper. Do not ask all the questions at the same time.

---

## Questions (from easy to difficult)

**Level 1: from the average to attention**

Ask the learner:
> Without your notes, say in one sentence what the output of attention is. Then explain: why can "the average of the vectors of all earlier tokens" be written as a lower-triangular matrix multiplied by X? What does row 3 of this matrix look like (count from 0)?

Expected answer: the output of attention is a weighted average of the value vectors of the context. The data sets the weights, and each row of weights sums to 1. Put the coefficients of the prefix average in a lower-triangular matrix W. Then row t of W @ X is the sum of the first t+1 vectors, each multiplied by 1/(t+1). Row 3 is `[1/4, 1/4, 1/4, 1/4, 0, …]`. Extra credit: "the upper triangle is 0 = do not look at the future".

---

**Level 2: the intuition of Q, K, V and √d**

Ask the learner:
> In Section 3, the score is `x·x` directly, and all 5 positions attend most to themselves. Why? How do two separate projections, Q and K, solve this problem? Also: if head_dim changes from 64 to 1024 and we do not divide by √d, what happens?

Expected answer: `x·x = |x|²`. A vector points in exactly the same direction as itself, so this dot product is often the largest. With different projections `W_q` and `W_k`, the score becomes `q_t·k_s`. The query ("what am I looking for") and the key ("what do I have") can be different, so the model can learn to look for other positions. Without scaling, the variance of `q·k` is about d (`03_why_sqrt_d.py` measured about 1027 at d = 1024). Softmax comes close to one-hot (max weight 0.957), and the gradient becomes smaller. After division by √d, the variance goes back to about 1.

---

**Level 3: causal mask and shapes (find the problem)**

Ask the learner:
> A classmate implemented attention and filled the positions of `torch.tril(..., diagonal=-1)` with −∞ (so the mask hides the lower triangle). The training loss decreases very fast, but the generated text is very bad. What happened? Also: with B=2, T=16, C=64, H=4, what are the shapes of the weight matrix and of the output of each head?

Expected answer: the classmate hid the "past" and opened the "future". In training, the model can see the next token (the answer) directly, so the training loss is very low. In generation, the future does not exist. The model did not learn to predict from the context, so the generated text is bad. (Row 0 can even see only itself, but the main problem is that the model looks at the answer.) Shapes: the weights are (2, 4, 16, 16), that is (B, H, T, T). The output of each head is (2, 4, 16, 16), that is (B, H, T, d), with d = 64/4 = 16. After concatenation, the shape is (2, 16, 64). Extra credit: "the two 16s have different meanings".

---

**Level 4: transfer to production code**

Ask the learner:
> In the `Attention` of `zero/model.py`, the output dimension of `wk` is `n_kv_heads × head_dim`, not `n_heads × head_dim`. At the end, it calls `F.scaled_dot_product_attention(..., enable_gqa=...)`. What does this save, compared with the minimal version of this chapter? Why does `05_zero_parity.py` of this chapter prove that the two are mathematically the same?

Expected answer: this is GQA. Several query heads share one K/V group, so `wk` and `wv` have fewer parameters (4096 → 3072 in the example). During inference, the KV cache also becomes smaller in proportion (Chapter 10). `05_zero_parity.py` first turns off QK-Norm and makes RoPE the identity transformation. Then it loads the weights of the minimal version into zero without changes; the max output difference is 8.9e-08. Then it compares GQA with "a manual copy of K/V, followed by normal multi-head attention"; the max difference is also 8.9e-08. `tests/test_model_hf_parity.py` makes sure that the full model agrees with HF Qwen3. Extra credit: SDPA uses FlashAttention on the GPU. This was verified on an RTX 3090: with BF16 (and `enable_gqa=True` for GQA), the default backend is flash.

---

## Rules for feedback

- If the answer is correct: say so. Then ask a deeper "why" question.
- If the answer is not correct: do not give the answer. Give a hint. For example, ask the learner to change `02_attention_from_scratch.py` or `03_why_sqrt_d.py` and run it. Then let them think again.
- If the learner says "I do not know": ask them to guess first. A wrong guess is better than no guess.

When the learner passes all four levels, tell them to continue to Chapter 9 (`chapters/09-modern-transformer/`). After Chapter 9, they can check themselves with `/ch09-transformer`.
