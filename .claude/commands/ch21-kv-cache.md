---
description: "Chapter 21 self-check: the KV cache ledger — prefill compute and decode bandwidth, layer-by-layer ledger, MQA → GQA → MLA, decoupled RoPE and absorption (第 21 章自检：KV cache 的账本——prefill 算力与 decode 带宽、按层记账、MQA → GQA → MLA、解耦 RoPE 与吸收)"
---

# Chapter 21 self-check: the KV cache ledger

The learner typed `/ch21-kv-cache`. They finished Chapter 21 (`chapters/21-kv-cache-ledger/`). Help them check if they understand it. Do not give them the answers.

**Language**: Use the language of the learner. If the learner writes in Chinese, ask the questions and give feedback in Chinese. The Chinese text of the chapter is in `README.zh.md`. Write short, clear sentences (see `docs/STYLE_GUIDE.md`).

**Method**: Ask the questions below one at a time. Wait for the learner to answer in their own words. Then give feedback: tell them what is correct, what is not correct, and how to make their understanding deeper. Do not ask all the questions at the same time.

---

## Questions (from easy to difficult)

**Level 1: calculate the ledger**

Ask the learner:
> Without your notes, write the formula for the memory of the KV cache. Then calculate: a model has 32 layers, 8 KV heads, head_dim 128, and BF16. How many bytes does it store per token? How many GiB does one conversation of 128K tokens need? Suppose that only 1 layer in each 4 layers uses full attention, and the other layers use linear attention. How many GiB is it then?

Expected answer: `2 × layers × KV heads × head_dim × sequence length × bytes`. Per token: 2 × 32 × 8 × 128 × 2 = 131,072 bytes = 128 KiB. At 128K: 16 GiB (this is the configuration of Llama-3.1-8B). With only 8 full-attention layers, it is 1/4, so 4 GiB. The linear layers have only a fixed state that does not depend on the length. The key points are "count layer by layer" and "linear layers do not grow with the length".

---

**Level 2: two kinds of cost**

Ask the learner:
> Why is long context expensive in prefill, and why is it expensive in decode? At a context of 32K, why does a batch of 16 conversations in decode give almost no increase in throughput?

Expected answer: In prefill, the attention compute grows as T² (at 32K, attention uses 73% of the compute of the main-line model). In decode, each step must read the weights and the KV cache from GPU memory. The arithmetic intensity is only a single-digit number, far below the ridge point of the H100 (about 295), so decode is memory-bound. A batch shares the reads of the weights, but each conversation reads its own KV cache. At 32K, each conversation has 3.5 GiB of KV cache, which is more than the weights (1.28 GiB). Thus a batch saves little. Extra credit: "Thus, when the KV cache is half as large, long-context decode is almost twice as fast, and the GPU can serve twice as many conversations at the same time."

---

**Level 3: the two key designs of MLA**

Ask the learner:
> MLA does not cache K and V. It caches a latent vector. At inference, must each step reconstruct the K and V of all past positions? Why can RoPE not be applied directly to the key that is reconstructed from the latent vector?

Expected answer: It does not need to reconstruct them. For the attention score qᵀ W_UK c, we can first calculate W_UKᵀ q (W_UK is "absorbed" into the query). For the output Σ p·W_UV c, we can first take the weighted average in the latent space and then multiply by W_UV (absorbed into the output). This works because matrix multiplication is associative. If RoPE is applied to W_UK c, a rotation matrix that depends on the position is between W_UQ and W_UK. Then the two cannot be combined in advance, and absorption does not work. Thus MLA uses "decoupled RoPE": a separate small set of dimensions (64 in DeepSeek-V3) carries the position. All heads share this part of the key, and the cache must store it too. Thus each layer stores 512 + 64 = 576 numbers per position. Extra credit: "After absorption, the form is equal to MQA."

---

**Level 4: transfer and judgment**

Ask the learner:
> You must make a long-context version of a dense model with 1B parameters. A colleague suggests that you change GQA to MLA. Use the ledger, the costs, and the adoption data from this chapter. How do you evaluate this suggestion? What other methods to save KV cache can you consider together with it?

Expected answer: First, calculate the ledger. The GQA KV cache of a 1B model may already be small. How much does MLA save, and how many more conversations does that give? Then look at the costs. In decode, MLA calculates dot products over 576 dims for each head, so it needs more compute. It does not work with QK-Norm, and it needs special kernels. The GLM-5 report found that in some setups, MLA was not as good as GQA-8 and needed a change to the optimizer. The four families that use MLA now (DeepSeek, Kimi, GLM, Mistral) all use it in large MoE flagships. Small dense models usually use GQA. Other methods: sliding window / alternating local-global layers (Chapter 22), hybrid linear attention (Chapter 23, for example the 3:1 pattern of Qwen3.5), and an FP8 KV cache. Finally, the decision must come from small experiments with several seeds and long-context evaluations, not from impressions.

---

## Rules for feedback

- If the answer is correct: say so. Then ask a deeper "why" question.
- If the answer is not correct: do not give the answer. Give a hint. For example, ask the learner to run `code/01_kv_ledger.py`, `code/02_prefill_decode.py`, or `code/03_mla.py` and look at the output. Then let them think again.
- If the learner says "I do not know": ask them to guess first. A wrong guess is better than no guess.

When the learner passes all four levels, tell them to continue to Chapter 22 (`chapters/22-*/`, local and sparse attention). After Chapter 22, they can check themselves with the matching `/ch22-*` skill.
