---
description: "Chapter 25 self-check: multi-token prediction and speculative decoding — decode is memory-bound, draft and one verification pass, greedy output is token-for-token identical, min(1, p/q) and the residual distribution, α and the speedup formula, the MTP module and self-speculation (第 25 章自检：多 token 预测与推测解码——decode 受带宽限制、草稿与一次验证、贪心逐字相同、min(1, p/q) 与残差分布、α 与加速比公式、MTP 模块与自推测)"
---

# Chapter 25 self-check: multi-token prediction and speculative decoding

The learner typed `/ch25-speculative`. They finished Chapter 25 (`chapters/25-mtp-speculative-decoding/`). Help them check if they understand it. Do not give them the answers.

**Language**: Use the language of the learner. If the learner writes in Chinese, ask the questions and give feedback in Chinese. The Chinese text of the chapter is in `README.zh.md`. Write short, clear sentences (see `docs/STYLE_GUIDE.md`).

**Method**: Ask the questions below one at a time. Wait for the learner to answer in their own words. Then give feedback: tell them what is correct, what is not correct, and how to make their understanding deeper. Do not ask all the questions at the same time.

---

## Questions (from easy to difficult)

**Level 1: why is the verification almost free?**

Ask the learner:
> In speculative decoding, the target model verifies k+1 positions in one forward pass. Why does this forward pass take about the same time as one normal decode step that generates 1 token? Is this also true in the prefill phase, which processes thousands of tokens at once?

Expected answer: decode calculates only one token at a time. Each step reads all weights (and the KV cache) from GPU memory, so the arithmetic intensity is very low. (Chapter 21: decode of the main-line model is only 1–5 operations per byte, much lower than the ridge point of about 295 on an H100.) The hardware waits for data. If we feed a few more tokens, the bytes read almost do not change, and only a little compute is added, so the time is about the same. Prefill is already compute-bound. If prefill gets k times more tokens, it needs about k times more time, so the condition of speculative decoding is not true. (This is also the reason why the benefit of speculative decoding is smaller with large batches and high concurrency.) Extra credit: in `01_models_and_cost.py` of this chapter, T increases from 1 to 17, but the time increases by only some tens of percent.

---

**Level 2: why is the greedy output token-for-token identical?**

Ask the learner:
> The draft guessed "t h e _". At these 4 positions, the argmax of the target model is "t h a t". Which tokens does this round produce? What must you do with the KV cache before the next round? Why is the output of the full process token-for-token identical to greedy decoding of the target model?

Expected answer: accept "t h". At position 3, "e" ≠ "a", so "e" is rejected and the answer of the target, "a", replaces it. This round produces 3 tokens, "t h a". The "_" after it and the answer of the target for it are discarded. Roll back the K/V of "e" and "_" in the caches of both models (a preallocated cache only needs to set the valid length back). Each accepted token is equal to the argmax of the target model with the same prefix. The correction is also the argmax of the target. Thus the output is token-for-token identical. The draft decides only "how many steps we move forward at once", not "what we write".

---

**Level 3: why does sampling keep the distribution?**

Ask the learner:
> The vocabulary has only two tokens, A and B. The target is p = (0.3, 0.7), and the draft is q = (0.6, 0.4). Use the rules of speculative sampling. When the draft samples A, with what probability do we accept it? After a rejection, from which distribution do we sample again? Calculate the total probability that the final token is A. What is α, the mean probability of one acceptance?

Expected answer: when the draft samples A, the acceptance probability is min(1, 0.3/0.6) = 0.5. When it samples B, the probability is min(1, 0.7/0.4) = 1, so B is always accepted. The residual distribution max(0, p − q) = (0, 0.3), which is (0, 1) after normalization. Thus after a rejection, the token always changes to B. P(A) = 0.6 × 0.5 = 0.3 = p(A). P(B) = 0.4 × 1 + 0.6 × 0.5 × 1 = 0.7 = p(B). α = Σ min(p, q) = 0.3 + 0.4 = 0.7 = 1 − TV(p, q). If the learner says "we can use the samples of the draft directly", show them the control group in part 1 of `03_speculative_sampling.py` (TV distance 0.43, chi-square p-value 0).

---

**Level 4: transfer — the cost calculation and MTP**

Ask the learner:
> DeepSeek-V3 says that when its MTP module is the draft, the acceptance rate of the second token is 85%–90%, and the decoding speed increases to about 1.8×. Use the speedup formula to explain this number. Also: why is the MTP module a good draft (compared with an independent small model)? Our main-line model (about 0.6B, without MTP) needs a draft. What would you do?

Expected answer: with k = 1, the expected output per forward pass is (1 − α²)/(1 − α) = 1 + α ≈ 1.85–1.9 tokens. After we remove the overhead of the MTP module itself (c > 0, plus the system overhead), a measured 1.8× is reasonable. The MTP module reads the last-layer representation of the main model directly, shares the embedding and the output head, and trains together with the main model. Thus its guesses are accurate (high α), it has only one block (small c), it needs no separate deployed model, and it has the same vocabulary automatically.

The main-line model has no MTP (GOAL 3.3: the main line takes no architecture risk). Possible methods:
- Train a separate small `zero` model with the same tokenizer and the same data as the draft (for example, 50 million to 100 million parameters). It is better to also distill it with the outputs of the main-line model to increase α.
- Use prompt lookup, which costs nothing. When tool calls copy many parameter names and JSON keys, α can be high.
- Freeze the main-line model later and train only one MTP module.

The inference engines (vLLM / SGLang) support all of these methods. But before step 2 measures them on a GPU, we cannot say how much faster they are.

---

## Rules for feedback

- If the answer is correct: say so. Then ask a deeper "why" question. For example: when c increases, why does the best k decrease? When the temperature increases, how does α usually change?
- If the answer is not correct: do not give the answer. Give a hint. For example, ask the learner to change α, k, and c in `04_speedup_formula.py` and run it, or to use a different pair p, q in `03_speculative_sampling.py`. Then let them think again.
- If the learner says "I do not know": ask them to guess first. A wrong guess is better than no guess.

When the learner passes all four levels, tell them to continue to Chapter 26 (`chapters/26-*/`, a panorama of the current state-of-the-art open models). After Chapter 26, they can check themselves with `/ch26-panorama`.
