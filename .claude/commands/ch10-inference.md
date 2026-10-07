---
description: "Chapter 10 self-check: inference — autoregressive generation, temperature and top-p, KV cache, prefill/decode, GQA (第 10 章自检：推理——自回归生成、温度与 top-p、KV cache、prefill/decode、GQA)"
---

# Chapter 10 self-check: inference

The learner typed `/ch10-inference`. They finished Chapter 10 (`chapters/10-inference/`). Help them check if they really understand it. Do not give them the answers.

**Language**: Use the language of the learner. If the learner writes in Chinese, ask the questions and give feedback in Chinese. The Chinese text of the chapter is in `README.zh.md`. Write short, clear sentences (see `docs/STYLE_GUIDE.md`).

**Method**: Ask the questions below one at a time. Wait for the learner to answer in their own words. Then give feedback: tell them what is correct, what is not correct, and how to make their understanding deeper. Do not ask all the questions at the same time.

---

## Questions (from easy to difficult)

**Level 1: concepts (the generation loop and decoding strategies)**

Ask the learner:
> In a few sentences, describe the loop that a language model uses to "generate a text". Then tell me: in which step of the loop do the temperature T, top-k, and top-p act? What does the loop become when T → 0?

Expected answer: the loop is "give the current sequence to the model → take the logits at the last position → use a strategy to pick one token → append it to the end of the sequence → do it again" (autoregressive). All three settings act in the step "pick one token". Temperature divides the logits by T before softmax (a small T makes the distribution sharper, a large T makes it flatter). top-k keeps only the k most probable tokens. top-p adds the probabilities from the largest down, cuts when the sum reaches p, and then normalizes again before sampling. When T → 0, the loop becomes greedy decoding (argmax).

---

**Level 2: intuition (why top-p is "smarter" than top-k)**

Ask the learner:
> In the code of this chapter, after "What is th", top-p=0.9 keeps only 5 characters. At the start of a line of dialogue, it keeps 15. What problem does top-k=5 cause in each of these two cases?

Expected answer: when the model is very certain (the next character is almost certainly e), a fixed 5 can keep some absurd candidates that should be removed. When the model is not certain (many characters are reasonable), only 5 removes reasonable candidates, and the generation becomes monotonous. With top-p, the number of kept tokens changes automatically with the "confidence" of the distribution (the core argument of nucleus sampling, Holtzman et al. 2019). Extra credit: in practice, people often use both together (for example, the default of Qwen3 is top-k 20 + top-p 0.95).

---

**Level 3: find the problem (what the KV cache saves, and what it does not save)**

Ask the learner:
> Without a KV cache, about how many positions does the model process to generate n tokens? And with a KV cache? Why do we cache K and V, but not Q? Also, the KV cache changes one problem into a new problem. What are the two problems?

Expected answer: at step t, the naive version calculates the full sequence again (the prompt + the t generated tokens). In total, it processes about n²/2 positions (the measurement in this chapter for 512 generated characters: 137,984 vs 525). At each step, the cached version calculates only the q/k/v of the new token, about n positions in total. In causal attention, a new token does not change the K and V of past positions, so we can use them again. Q is used only once, when the current token looks at the history, and never again. The cost is GPU memory: `2 × layers × KV heads × head_dim × sequence length × bytes`. The main-line model needs 112 KiB per token and 3.5 GiB for a 32K context, which is 2.7 times the weights. The compute bottleneck becomes a memory bottleneck. Extra credit: the learner can explain the difference between prefill (parallel, uses the compute fully) and decode (one token at a time, does not use the compute fully).

---

**Level 4: transfer (calculate the GQA ledger)**

Ask the learner:
> A model has 32 layers, 32 query heads, head_dim 128, and BF16. Calculate the KV cache of one sequence with an 8K context for MHA, for GQA with 8 KV heads, and for MQA. In GQA with 8 KV heads, how many query heads share one set of K/V? Why do not all models use MQA, which saves the most?

Expected answer: per token = 2 × 32 × KV heads × 128 × 2 bytes. MHA (32 KV heads): 512 KiB per token, 4 GiB for an 8K context. GQA (8 KV heads): 128 KiB per token, 1 GiB for an 8K context. MQA (1 KV head): 16 KiB per token, 128 MiB for an 8K context. In GQA-8, each 4 query heads share one set. MQA has only one set of K/V, so it loses more expressive power, and its training can be less stable. The GQA paper found that group sharing gives a quality near MHA and a speed near MQA. Thus most main models (Llama 3, Qwen3, Mistral, and others) choose the GQA compromise. Extra credit: the learner knows that MLA in Chapter 21 takes a different path with low-rank compression. The learner also knows that, in the small experiment of this chapter, we can draw a conclusion about the loss differences of MHA/GQA/MQA only after we compare them with the difference from "a different random seed".

---

## Rules for feedback

- If the answer is correct: say so. Then ask a deeper "why" question.
- If the answer is not correct: do not give the answer. Give a hint. For example, ask the learner to run `code/02_sampling.py` and look at the distribution, to change `LENGTHS` in `03_kv_cache.py` and look at the speedup, or to change `04_kv_memory.py` and calculate a model that interests them. Then let them think again.
- If the learner says "I do not know": ask them to guess first. A wrong guess is better than no guess.

When the learner passes all four levels, tell them that Part 2 is complete. They can continue to the first chapter of Part 3: Chapter 11, "Evaluation: set the exam first" (`chapters/11-evaluation/`).
