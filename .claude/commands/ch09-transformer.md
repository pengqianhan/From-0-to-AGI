---
description: "Chapter 9 self-check: the modern Transformer — Pre-Norm RMSNorm, RoPE, SwiGLU, QK-Norm, tied embeddings, tensor-shape data flow (第 9 章自检：现代 Transformer——Pre-Norm RMSNorm、RoPE、SwiGLU、QK-Norm、共享 embedding、张量形状数据流)"
---

# Chapter 9 self-check: the modern Transformer

The learner typed `/ch09-transformer`. They finished Chapter 9 (`chapters/09-modern-transformer/`). Help them check if they really understand it. Do not give them the answers.

**Language**: Use the language of the learner. If the learner writes in Chinese, ask the questions and give feedback in Chinese. The Chinese text of the chapter is in `README.zh.md`. Write short, clear sentences (see `docs/STYLE_GUIDE.md`).

**Method**: Ask the questions below one at a time. Wait for the learner to answer in their own words. Then give feedback: tell them what is correct, what is not correct, and how to make their understanding deeper. Do not ask all the questions at the same time.

---

## Questions (from easy to difficult)

**Level 1: concepts — what is in a Block**

Ask the learner:
> Without your notes, write the two formulas of one Block of a modern Transformer. Say what attention does and what the FFN does. Then explain: where is RMSNorm, and why is the name "Pre-Norm"?

Expected answer: `x = x + Attn(RMSNorm(x))` and `x = x + FFN(RMSNorm(x))`. Attention exchanges information between positions. The FFN processes each position separately (the same parameters apply to each position independently). The normalization is at the input of each sublayer, not after the residual addition. The residual "main road" stays an identity mapping, so the gradient goes directly through a deep network. This is Pre-Norm. Extra credit: "the shape of the residual stream (B, T, d) does not change from start to end".

---

**Level 2: intuition — why RoPE keeps only the relative position**

Ask the learner:
> Why can attention alone not tell "the dog bit the man" from "the man bit the dog"? At position m, RoPE rotates q by m·ω. At position n, it rotates k by n·ω. Why does the dot product depend only on m − n? In the code of this chapter, why are the dot products of the three pairs (3,1), (10,8), and (50,48) exactly the same?

Expected answer: the attention score looks only at the content similarity q·k. A weighted average does not depend on the order, so attention treats the earlier text as a set. After a 2D rotation, the dot product depends only on the angle between the two vectors. q rotates by mω and k rotates by nω, so the angle changes by (m − n)ω, and the absolute positions cancel. All three pairs have a distance of 2, so the dot products are the same (all 2.4426). Extra credit: "the rotation does not change the length", "different dimension pairs rotate at different speeds: the fast pairs distinguish near positions and the slow pairs distinguish far positions", and "RoPE applies only to q and k".

---

**Level 3: find the problem — parameters, scale, and tied embeddings**

Ask the learner (you can split this into separate questions):
> (1) SwiGLU has one more matrix than a normal MLP. Why is the hidden width about 8/3·d and not 4d? (2) In the experiment of this chapter, q and k are 4 times larger. Without QK-Norm, the mean max attention weight becomes 0.923. With QK-Norm, it is always 0.213. What problem does this show, and how does QK-Norm solve it? (3) Qwen3-0.6B ties its embeddings, but Qwen3-8B does not. Why?

Expected answer: (1) Two d×4d matrices have 8d² parameters, and three d×(8/3)d matrices also have 8d². The number of parameters is the same (13.11M for both at d=1280). This is only a starting point. Each team adjusts it for hardware alignment and from experiments (Qwen3-0.6B uses 3d, Llama 3.2 1B uses 4d). (2) The norm of q and k increases → the scores increase → softmax becomes one-sided, and the gradients become sharp and unstable. QK-Norm applies RMSNorm to q and k of each head before the dot product. Then the scale of the scores does not change with the drift of the activations. (The learnable weight of RMSNorm can still make the scores larger slowly, but the parameters control this.) (3) In a small model, the vocabulary matrix is a large fraction (155.6M of 596M in Qwen3-0.6B, 26%), so tied embeddings save many parameters. In a large model, the fraction is small, and untied embeddings give the input and the output more freedom.

---

**Level 4: transfer — parity check on two levels and tensor shapes**

Ask the learner:
> In this chapter, we copy the weights of the minimal model into `zero.Transformer`. The maximum difference of the logits is 1.43×10⁻⁵, not 0. (1) For the two models to match, which parameter in the configuration of zero must have a special value that agrees with the minimal code? (2) Why is the difference not 0, and why is this not a bug? (3) Somebody changes `apply_rope` to pair adjacent dimensions. Can training still work normally? What happens to the parity check? (4) For the main-line model with B=8, T=4096, V=65536, how large is the logits tensor? What problem of Chapter 14 does this suggest?

Expected answer: (1) The minimal code has no GQA, so `n_kv_heads = n_heads` is necessary. (The other settings, such as qk_norm, tie_embeddings, eps, and rope_theta, must also agree.) (2) The minimal code writes softmax(QKᵀ/√d)·V by hand, and zero uses the SDPA of PyTorch. The math is the same, but the order of floating-point operations is different, so the results agree within floating-point error. `assert_close(rtol=1e-5, atol=1e-5)` passes, and greedy generation gives the same bytes. This shows that the structures are the same. (3) Training works normally (the two pairings are mathematically equivalent; only the order of the dimensions is different). But the weights are not interchangeable with the "first half and second half" pairing of zero / HF Qwen3, so the parity check fails. (4) 8×4096×65536 ≈ 2.15 billion numbers, about 8 GiB in fp32. With a large vocabulary and long sequences, the final logits and the T×T attention scores use the most memory. FlashAttention, chunked cross-entropy, and similar methods solve this.

---

## Rules for feedback

- If the answer is correct: say so. Then ask a deeper "why" question (for example, "What occurs if we also rotate v?").
- If the answer is not correct: do not give the answer. Give a hint. Ask the learner to go back to `code/01_position.py`, `code/05_qk_norm.py`, `code/03_shapes.py`, or `code/04_parity_with_zero.py`, change a parameter, and run it. Then let them think again.
- If the learner says "I do not know": ask them to guess first. A wrong guess is better than no guess.

When the learner passes all four levels, tell them to continue to Chapter 10 (`chapters/10-inference/`). After Chapter 10, they can check themselves with `/ch10-inference`. During generation, the model calculates the full sequence again for each byte. Chapter 10 solves this problem with a KV cache and GQA.
