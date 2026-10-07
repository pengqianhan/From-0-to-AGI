---
description: "Chapter 23 self-check: linear attention and hybrid architectures — remove the softmax → recurrent state, chunkwise form, delta rule, Gated DeltaNet, 3:1 hybrid (第 23 章自检：线性注意力与混合架构——去掉 softmax → 递推状态、分块形式、delta 规则、Gated DeltaNet、3:1 混合)"
---

# Chapter 23 self-check: linear attention and hybrid architectures

The learner typed `/ch23-linear-attention`. They finished Chapter 23 (`chapters/23-linear-attention-hybrid/`). Help them check if they understand it. Do not give them the answers.

**Language**: Use the language of the learner. If the learner writes in Chinese, ask the questions and give feedback in Chinese. The Chinese text of the chapter is in `README.zh.md`. Write short, clear sentences (see `docs/STYLE_GUIDE.md`).

**Method**: Ask the questions below one at a time. Wait for the learner to answer in their own words. Then give feedback: tell them what is correct, what is not correct, and how to make their understanding deeper. Do not ask all the questions at the same time.

---

## Questions (from easy to difficult)

**Level 1: concept — where does the state come from?**

Ask the learner:
> To generate token t, softmax attention uses all t earlier K and V vectors. Why does attention need only one fixed-size matrix S when we remove the softmax? Write the update formula and the read formula of S. What is the shape of S?

Expected answer: without the softmax, o_t = Σ_{j≤t} (q_t·k_j) v_j = (Σ_{j≤t} v_j k_jᵀ) q_t. The model can add up the sum in the parentheses step by step: S_t = S_{t−1} + v_t k_tᵀ, and o_t = S_t q_t. The shape of S is d_v × d_k, independent of the sequence length. The key word is "associativity of matrix multiplication": (Q Kᵀ) V becomes Q (Kᵀ V). Extra credit: the softmax normalizes a full row, so we cannot split it.

---

**Level 2: intuition — why overwrite, and why forget?**

Ask the learner:
> We write v1 and then v2 with the same key. What does naive linear attention read? What does the delta rule read? Then tell me which other problem the decay gate α_t of Gated DeltaNet solves.

Expected answer: naive linear attention can only add, so it reads v1 + v2. The delta rule first reads the old answer S k. Then it writes only the difference β(v − S k). With β = 1, it reads v2 (overwrite). The decay gate makes old content decay exponentially by α_t. Also, the input decides α_t (data-dependent). Thus the model can close the gate at a "topic change" and clear the old memory. In `03_delta_rule.py`, the delta rule with "α = 0 at the topic change" has the lowest read error. If the learner says only "it forgets old information", ask: what is the difference between a fixed α and a data-dependent α?

---

**Level 3: find the problem — the cost of a fixed size**

Ask the learner:
> In the associative recall experiment of this chapter, the accuracy of the pure linear models drops clearly when the number of key-value pairs grows. The 3:1 hybrid model (with only one full-attention layer) recovers most of the accuracy. Use "capacity" to explain why. Then explain why the industry does not use only linear layers, and does not use only attention. Why does it use "a few full-attention layers + many linear layers"?

Expected answer: a d_v × d_k state can reliably store only about d (order of magnitude) nearly orthogonal key-value pairs. The more it stores, the larger the interference (the capacity table in `03`). Exact recall (retrieval / recall) needs the original K and V. This is exactly the KV cache. In a hybrid, the full-attention layers do the exact recall. The linear layers give most of the depth at a fixed cost. The KV cache grows only with the number of full-attention layers. At 3:1, it is about 1/4 of pure attention (Qwen3.5-0.8B: 6 full-attention layers in 24 layers). Extra credit: MiniMax went from MiniMax-01 (a 7:1 hybrid with linear attention) back to full attention in M2. The reasons include immature infrastructure for linear attention, the difficulty of a low-precision state, and the open problem of speculative decoding. This shows that the hybrid is still an engineering trade-off.

---

**Level 4: transfer — chunkwise form for training, recurrent form for inference**

Ask the learner:
> In training, `zero/arch/linear_attention.py` uses `chunk_gated_delta_rule`. In inference decode, it uses `recurrent_gated_delta_rule`. If only the recurrent form exists, what problem does training have? If only the fully parallel (T×T) form exists, what happens in inference? In the chunkwise form, which extra calculation does the delta rule need, compared with naive linear attention? Why?

Expected answer: with only the recurrent form, training must go token by token in sequence, and the GPU has little parallel work (`02_chunked.py`: at T = 4096, the token-by-token recurrence is about one order of magnitude slower than the chunkwise form). With only the T×T parallel form, inference must store the full history again, and the cost of each step grows with the length. Then linear attention loses its purpose. In the delta rule, the u_i that each position in a chunk writes depends on the earlier u_j. Thus the delta rule must solve a lower-triangular linear system (I + A)U = … (the UT transform). Naive linear attention does not need this step. Extra credit: the parity checks "input in segments == input in one pass" and "generation with the cache == full recalculation".

---

## Rules for feedback

- If the answer is correct: say so. Then ask a deeper "why" question.
- If the answer is not correct: do not give the answer. Give a hint. For example, ask the learner to run `code/03_delta_rule.py` and look at the capacity table, or to change the number of key-value pairs in `code/05_associative_recall.py`. Then let them think again.
- If the learner says "I do not know": ask them to guess first. A wrong guess is better than no guess.

When the learner passes all four levels, tell them to continue to Chapter 24 (mixture of experts, MoE: `chapters/24-mixture-of-experts/`). After Chapter 24, they can check themselves with `/ch24-moe`.
