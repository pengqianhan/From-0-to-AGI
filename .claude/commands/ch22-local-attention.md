---
description: "Chapter 22 self-check: local and sparse attention — sliding window, receptive field, local-global interleaving, bounded KV cache, sparse attention that selects the top-k keys by content (第 22 章自检：局部与稀疏注意力——滑动窗口、感受野、局部-全局交替、有界 KV cache、按内容挑 top-k 的稀疏注意力)"
---

# Chapter 22 self-check: local and sparse attention

The learner typed `/ch22-local-attention`. They finished Chapter 22 (`chapters/22-local-sparse-attention/`). Help them check if they understand it. Do not give them the answers.

**Language**: Use the language of the learner. If the learner writes in Chinese, ask the questions and give feedback in Chinese. The Chinese text of the chapter is in `README.zh.md`. Write short, clear sentences (see `docs/STYLE_GUIDE.md`).

**Method**: Ask the questions below one at a time. Wait for the learner to answer in their own words. Then give feedback: tell them what is correct, what is not correct, and how to make their understanding deeper. Do not ask all the questions at the same time.

---

## Questions (from easy to difficult)

**Level 1: concept — what does a sliding window save?**

Ask the learner:
> A model has 36 layers. Half of the layers use a sliding window of 128. The other half use full attention (gpt-oss-120b has this design). The context grows from 8K to 128K. How does the KV cache of the sliding-window layers change? How does the KV cache of the full-attention layers change? And what happens to the attention compute for each new token?

Expected answer: each sliding-window layer stores only the last 128 positions. Its KV cache does not grow with the context (the maximum is W). The KV cache of a full-attention layer grows linearly with the length. For each new token, a sliding-window layer calculates attention with only 128 keys (O(W)). A full-attention layer calculates attention with all of the history (O(T)). Thus, in total, the KV cache is about half (`01_masks_and_ledger.py` calculates 9.00 → 4.50 GiB at 128K). The fraction of local layers sets the size of the saving. Extra credit: "the window includes the token itself, so a token sees at most W positions".

---

**Level 2: intuition — the receptive field**

Ask the learner:
> The window is W = 4. The model has 6 layers, and all of them use a sliding window. From how many positions back can information "indirectly" reach the last token? What changes if layers 3 and 6 use full attention?

Expected answer: each layer moves information back by at most W − 1 = 3 more positions. After 6 layers, this is 6 × 3 = 18 (the receptive-field table of `01_masks_and_ledger.py`). If layer 3 is global, the token sees the start in one step after layer 3 (in the table, the value becomes 63 from layer 3, which is the full sequence). Follow-up question: what does "indirectly" mean? Answer: intermediate tokens must relay the information from layer to layer. The information must also fit into a hidden dimension of limited size. A distance that is reachable in theory is not always a distance that the model can learn.

---

**Level 3: find the problem — why the loss is not sufficient**

Ask the learner:
> In the experiment of this chapter, the all-sliding-window model has almost the same language-modeling loss as the full-attention model. But on the needle in a haystack, its accuracy at long distances falls to the level of a random guess. Why does the loss not show this problem? What does this mean for the evaluation of long-context models?

Expected answer: next-token prediction at the character level (and for most natural text) depends mainly on the nearby context. Long-distance dependencies are only a small part of the mean loss, so the loss of a local model is about the same. But some tasks need one specific piece of information from far away (retrieval, a citation, or a parameter that the text defined earlier in a tool call). If the model cannot reach that position, it cannot use the information. Conclusion: measure long-context ability with dedicated retrieval evaluations (needle in a haystack, RULER, and similar), not only with perplexity. The Gemma 3 report also shows that the fraction of local layers has only a small effect on perplexity. This is why models still keep some global layers.

---

**Level 4: transfer — sparse attention vs sliding window**

Ask the learner:
> A sliding window, DSA of DeepSeek, and MSA of MiniMax all let each query see only a part of the keys. What is the basic difference in how they select keys? Can DSA put a maximum on the KV cache, as a sliding window does? Why do real systems train a separate "indexer", instead of selecting keys directly with q·k as `05_topk_sparse.py` does?

Expected answer: a sliding window selects keys by **position** (the last W). The selection is fixed and needs no calculation. Sparse attention selects keys by **content** (the k keys or k blocks with the highest scores), so it can jump to positions that are far away. In `05_topk_sparse.py`, with the same budget of k keys, selection by content keeps the long-distance needle accuracy, and selection by position does not. DSA cannot put a maximum on the KV cache. Nobody knows in advance which history token a later query will select. Thus the model must keep all K/V (and also the small keys of the indexer). DSA saves compute and memory reads. Selection with q·k must first calculate all the scores, so it saves nothing. An indexer has a small dimension and few heads (and can even use low precision), so its scores are cheap. Then the model calculates exact attention only on the few selected keys.

---

## Rules for feedback

- If the answer is correct: say so. Then ask a deeper "why" question. For example: must a sliding-window layer use the same RoPE base frequency as a global layer? Look at the configurations of Gemma 3 and gpt-oss.
- If the answer is not correct: do not give the answer. Give a hint. For example, ask the learner to change `W` or `VARIANTS` in `02_swa_model.py` and run `03_compare.py` again. Then let them think again.
- If the learner says "I do not know": ask them to guess first. A wrong guess is better than no guess.

When the learner passes all four levels, tell them to continue to Chapter 23 (`chapters/23-linear-attention-hybrid/`). After Chapter 23, they can check themselves with `/ch23-linear-attention`. Chapter 23 takes another path: it stores no K and V at all, and it compresses the history into a state of fixed size. This is linear attention and the hybrid architecture.
