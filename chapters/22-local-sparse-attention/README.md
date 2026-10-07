# Chapter 22: Local and sparse attention — Look nearby, and keep the distant context

**English** · [中文](README.zh.md)

> **Goal**: After this chapter, you can write the mask of sliding window attention and a KV cache that has a maximum size. You can calculate the receptive field of L sliding-window layers. You can read `layer_types` and `sliding_window` in `config.json`. Then you can tell how many layers of Gemma, gpt-oss, and OLMo 3 are local, and how much KV cache this saves. With a small experiment, you can show where a pure sliding window fails, and why one global attention layer fixes the problem. You can also explain the difference between sparse attention, which selects keys by content, and a sliding window, which selects keys by position.

📺 **Video**: Not published yet. To render it on your computer, run `bash chapters/22-local-sparse-attention/video/build.sh`.
🧪 **Self-check**: After the chapter, type `/ch22-local-attention` in Claude Code.

---

In the last chapter, we kept a ledger for the KV cache. Long context is expensive in two places. The attention compute of prefill grows with the square of the length. At each decode step, the model must read all the K/V of the history from GPU memory. MQA, GQA, and MLA decrease "how much each position stores". This chapter asks a different question: **Must each position look at the full history?**

The most direct method is to look only at a nearby segment. This is **sliding window attention (SWA)**. Its cost is that the model cannot see far away. Thus models use **local-global interleaving**: most layers are local, and a few layers are global. One step further: instead of a fixed look at the "most recent" keys, the model can select the few "most relevant" keys itself. This is **sparse attention**.

The code for this chapter (all of it runs on the CPU):

```bash
uv run python chapters/22-local-sparse-attention/code/01_masks_and_ledger.py  # masks, receptive field, KV ledger of 4 public models (a few seconds)
uv run python chapters/22-local-sparse-attention/code/02_swa_model.py         # trains 6 small models (slow on the first run, see below; later runs read the cache)
uv run python chapters/22-local-sparse-attention/code/03_compare.py           # full attention / sliding window / interleaving: loss, KV, needle in a haystack
uv run python chapters/22-local-sparse-attention/code/04_bounded_cache.py     # generation with a truncated cache = generation without a cache; the cache size has a maximum
uv run python chapters/22-local-sparse-attention/code/05_topk_sparse.py       # the same k keys: select by position vs select by content
```

`02_swa_model.py` trains 6 small models on a single CPU thread: 3 language models with 600 steps each, and 3 needle-in-a-haystack models with 600 steps each. On the build machine of this course, many jobs shared the CPU, and the first run took about 40 minutes in total. On an idle computer, it is much faster. The later scripts read the cached weights in `code/out/` directly. `03` and `04` need one or two minutes each, and `05` needs a few minutes.

## 1. The two costs of full attention

First, recall the causal attention of Chapter 8. The query at position i calculates one score with each key at positions 0…i. Then it calculates a weighted average of the values. Draw "which pairs are calculated" as a T × T table. The result is a lower triangle:

```
Full causal attention T=10 (■ = visible, 55 pairs)
  ■ · · · · · · · · ·
  ■ ■ · · · · · · · ·
  ■ ■ ■ · · · · · · ·
  ...
  ■ ■ ■ ■ ■ ■ ■ ■ ■ ■
```

Both costs depend on the length T (Chapter 21 calculated them in detail):

- **Compute**: the model must calculate `T(T+1)/2` scores, so this cost grows as T². When the model prefills a long prompt, this cost becomes larger than all the matrix multiplications together.
- **KV cache**: each layer stores K and V for all T positions. In decode, for each new token, the model must read all of them from GPU memory.

The problem is that each position looks at the full history. But most dependencies in language are short. A word depends most often on words in the same sentence or in the same paragraph. So can a position **look only nearby**?

## 2. Sliding window: each token looks only at the last W

**Sliding window attention** adds one condition to the causal mask. Position i can see only the positions j with `i − W < j ≤ i`. These are the last W positions (position i included).

```
mask(i, j) = (j ≤ i) and (i − j < W)
```

In code, this adds one `&` ([`code/01_masks_and_ledger.py`](code/01_masks_and_ledger.py)):

```python
def sliding_mask(T: int, W: int) -> torch.Tensor:
    i = torch.arange(T)[:, None]
    j = torch.arange(T)[None, :]
    return (j <= i) & (i - j < W)  # see only the last W positions (itself included)
```

The triangle becomes a diagonal band:

```
Sliding window W=4 (■ = visible, 34 pairs)
  ■ · · · · · · · · ·
  ■ ■ · · · · · · · ·
  ■ ■ ■ · · · · · · ·
  ■ ■ ■ ■ · · · · · ·
  · ■ ■ ■ ■ · · · · ·
  · · ■ ■ ■ ■ · · · ·
  ...
  · · · · · · ■ ■ ■ ■
```

The band has a fixed width. Thus the compute for each query is O(W), not O(T). The area of the full table changes from T²/2 to about T·W. Count the pairs (output of the same script):

| Length T | Full causal (pairs) | Sliding window, W = 4 (pairs) | Ratio |
|---:|---:|---:|---:|
| 1,024 | 524,800 | 4,090 | 128 |
| 32,768 | 536,887,296 | 131,066 | 4,096 |

Real models use windows of some hundreds to some thousands of positions (next section). But the principle is the same: **the more the context is longer than the window, the more the window saves**.

> **Note:** In this course, the window W always includes the token itself, so each token sees at most W positions. This agrees with the meaning of the `sliding_window` field in Hugging Face configurations. Different kernels can define the parameter with a difference of one. For example, `window_size=(left, right)` in FlashAttention means "how many more positions to the left", which corresponds to W − 1. When you port code, do a parity check.

## 3. Receptive field: layers relay the information

If a token looks only at the last W positions, is the distant information lost completely? Not completely. In layer 1, position i collects information from i−3…i. In layer 2, position i sees position i−3, and position i−3 already collected information from i−6…i−3 in layer 1. **Each layer moves information back by at most W − 1 more positions.** After L layers, a token can in theory "see indirectly" up to `L × (W − 1)` positions back.

`01_masks_and_ledger.py` calculates this with a product of boolean matrices. `reach[i, j]` tells if the information of position j can flow to position i through the layers so far. Each layer multiplies by its mask once:

```python
reach = (m.float() @ reach.float()) > 0  # in this layer, i sees k, and k already collected j
```

With T = 64, W = 4, and 6 layers, this is the largest distance back that the last token can see after each layer:

| Layer | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---:|---:|---:|---:|---:|---:|
| All sliding window | 3 | 6 | 9 | 12 | 15 | 18 |
| Layers 3 and 6 use full attention | 3 | 6 | 63 | 63 | 63 | 63 |

The first row is exactly `l × 3`. The Mistral 7B paper uses the same argument: with a window of 4096 and 32 layers, the theoretical span is about 130,000 tokens. The second row shows another fact: with only one full-attention layer, the token sees the start in **one step** after that layer.

But "reachable in theory" does not mean "learnable in practice". The distant information must pass through intermediate tokens, layer by layer. It must also fit into a hidden dimension of limited size. The experiment in Section 6 shows how hard this limit is.

## 4. The KV cache has a maximum size

A sliding window has a benefit that is more practical than the saved compute: **the model never uses the K and V outside the window again, so it can discard them**. Thus the KV cache of a sliding-window layer has a maximum of W positions. It does not grow with the context.

The minimal version (`KVCache.append` in [`code/02_swa_model.py`](code/02_swa_model.py)) first gives "old + new" to the attention of this step. Then it keeps only the last W:

```python
keep = slice(None) if window is None else slice(-window, None)
self.k[layer], self.v[layer], self.pos[layer] = k[:, :, keep], v[:, :, keep], pos[keep]
```

K gets RoPE before it goes into the cache, so the position information is already "rotated" into the vector. Thus the cache must only also store the global position `pos` of each slot, to build the mask.

[`code/04_bounded_cache.py`](code/04_bounded_cache.py) uses the small language model that Section 6 trains (4 layers, W = 16). It generates 300 characters greedily. It compares a "truncated cache" with "no cache, recalculate the full sequence at each step":

| Configuration | Same as no cache, character for character | KV cache bytes: 16 / 64 / 128 / 256 / 306 positions cached |
|---|---|---|
| Full attention | True | 49,152 / 196,608 / 393,216 / 786,432 / 940,032 |
| All sliding window | True | 49,152 / 49,152 / 49,152 / 49,152 / 49,152 |
| 3 local + 1 global | True | 49,152 / 86,016 / 135,168 / 233,472 / 271,872 |

The pure sliding window uses 49,152 bytes from start to end (16 positions × 4 layers × K and V × 96 dims × 4 bytes). In the interleaved configuration, only the one global layer grows. The production implementation (see the end of the chapter) uses a **ring buffer** instead of slices. Position p goes into slot `p % W`, and the new entry overwrites the oldest one. The code allocates the buffer once and never copies it.

## 5. Local-global interleaving: how real models configure it

Pure sliding windows are not common in large models. The first Mistral 7B (v0.1) used a sliding window of 4096 in all 32 layers. But the later Mistral-7B-v0.3 sets `sliding_window` to `null` in `config.json`, so it went back to full attention. Today, the main design is **local-global interleaving**. Most layers use a small window, and after every few layers, one full-attention layer does the "retrieval of distant information". The configurations are written directly in `config.json` (`layer_types` lists the type of each layer, and `sliding_window` gives the window):

| Model | Layer layout | Window | Source |
|---|---|---:|---|
| Gemma 2 (9B) | Local and global alternate 1:1 | 4096 | Technical report: "alternate between a local sliding window attention and global attention in every other layer"; `sliding_window` in `config.json` |
| Gemma 3 (27B) | 5 local : 1 global | 1024 | Technical report; `sliding_window` 1024 (the 1B version has 512, `sliding_window_pattern` 6) |
| Gemma 4 (31B) | 5 local : 1 global | 1024 | `layer_types` (10 global layers of 60); the E2B version has a window of 512 and 4:1 |
| gpt-oss (20b / 120b) | Alternate 1:1 | 128 | `layer_types`; `sliding_window` 128 |
| OLMo 3 (7B) | 3 local : 1 global | 4096 | `layer_types` |
| Ministral 8B (2410) | 1 global : 3 local | 32768 | `layer_types` (the window is equal to the maximum length) |
| Step-3.5-Flash | 1 global : 3 local | 512 | `layer_types`; `sliding_window` 512 |
| MiMo-V2-Flash | 1 global : 5 local (the first layer is global) | 128 | `hybrid_layer_pattern`; `sliding_window` 128 |
| Llama 4 Scout | In each group of 4 layers, 3 layers use "chunked attention" and 1 layer is global (NoPE) | 8192 (chunk) | `attention_chunk_size` 8192, `no_rope_layers` |

Some observations:

- **The ratios and windows are very different**: the ratios go from 1:1 to 5:1, and the windows from 128 to 4096. In the ablations of the Gemma 3 report (Section 5.2), the local:global ratio changed from 1:1 to 7:1, and the window decreased from 4096 to 512. In all cases, the effect on perplexity was small. At a 32K context, the KV cache of an "all global" configuration is equal to 60% of the memory of the model itself. With 1:3 and a window of 1024, it is less than 15%. We meet "perplexity shows no difference" again in the small experiment of Section 6.
- **Many models use different RoPE base frequencies for local and global layers**: in Gemma 3, `rope_local_base_freq` is 10000, and the `rope_theta` of the global layers is 1 million. Gemma 4, MiMo-V2-Flash, and Step-3.5-Flash also set the two values separately. A local layer sees only some hundreds of positions. It does not need the low-frequency dimensions that are for long distances.
- **Chunked attention** (Llama 4) is a close relative of the sliding window. It cuts the sequence into chunks of 8192. Inside a chunk, attention is causal, and a token does not see other chunks. The effect is similar, and the cache and the kernel are more regular.

Use the ledger of Chapter 21 to calculate the saving at a 128K context (BF16, batch 1; the last part of `01_masks_and_ledger.py`):

| Model (configuration) | If all layers used full attention | Real configuration | Saving |
|---|---:|---:|---:|
| Mistral-7B-v0.1 (all sliding, W 4096) | 16.00 GiB | 0.50 GiB | 96.9% |
| Gemma-3-27B (5:1, W 1024) | 62.00 GiB | 10.41 GiB | 83.2% |
| gpt-oss-120b (1:1, W 128) | 9.00 GiB | 4.50 GiB | 50.0% |
| OLMo-3-7B (3:1, W 4096) | 64.00 GiB | 17.50 GiB | 72.7% |

(The official maximum lengths of Mistral 7B and OLMo 3 7B are 32K and 64K. Here we use 128K for all models, only to compare the structures.) The rule is direct: **the part of the global layers still grows linearly with the length, and the part of the local layers has a maximum**. A larger fraction of local layers and a smaller window give a larger saving.

## 6. Where is the cost? The loss does not show it, but the needle in a haystack does

[`code/02_swa_model.py`](code/02_swa_model.py) defines a small Transformer in which each layer can have a different window. It has the same structure as in Chapters 9 and 10. The three configurations all have 4 layers and a window of W = 16:

| Configuration | Window of each layer | Similar to |
|---|---|---|
| `full` | `[None, None, None, None]` | Full attention |
| `sliding` | `[16, 16, 16, 16]` | Mistral 7B v0.1 |
| `interleave` | `[16, 16, 16, None]` | 3 local : 1 global (the ratio of OLMo 3) |

The attention has only one more line for the mask. The three configurations use the same code:

```python
att = att.masked_fill(~window_mask(pos, k_pos, self.window), float("-inf"))
```

**Experiment 1: character-level language modeling** (Shakespeare, 600 steps, the same data order). Run [`code/03_compare.py`](code/03_compare.py):

| Configuration | Validation loss (nats/character) | KV cache positions @128 | @4096 |
|---|---:|---:|---:|
| full | 1.797 | 512 | 16,384 |
| sliding | 1.745 | 64 | 64 |
| interleave | 1.748 | 176 | 4,144 |

The three are almost the same, and the pure sliding window is even a little better. The reason is that the prediction of the next character depends mainly on the nearby context. In such a small model with such short training, "look only nearby" is a useful prior. (Do **not** extrapolate this result to large models.) If you look only at the loss, you will think that the sliding window has no cost.

**Experiment 2: needle in a haystack**. The sequence has a length of 96 and contains only random "filler characters". One position hides a "needle" (one of 8 types). The last position is the "question", and the model must tell which needle it is. The distance d between the needle and the question is uniformly random in 1–95. Thus we can measure the accuracy for each distance:

```python
x[torch.arange(bsz), NEEDLE_T - 1 - d] = ans  # put the needle at distance d from the question
loss = F.cross_entropy(model(x)[:, -1], ans)  # calculate the loss only at the "question" position
```

The output of the same script (64 samples at each distance; a random guess among 8 gives 12.5%):

| Configuration | d < 16 (in the window) | 16 ≤ d ≤ 60 (by relay) | d > 60 (out of reach) |
|---|---:|---:|---:|
| full | 100.0% | 100.0% | 100.0% |
| sliding | 100.0% | 100.0% | 11.9% |
| interleave | 100.0% | 100.0% | 100.0% |

By distance (the mean of each group of 8 distances):

| d from | 1 | 9 | 17 | 25 | 33 | 41 | 49 | 57 | 65 | 73 | 81 | 89 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| full | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| sliding | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 55% | 12% | 13% | 11% | 12% |
| interleave | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |

(The group 57 contains d = 57…64. It crosses the line at 60, so its value is 55%.)

The results agree exactly with the receptive-field table of Section 3:

- **The pure sliding window has 100% for d ≤ 60.** The model really learned to relay the information from layer to layer.
- **When d is more than 4 × (16 − 1) = 60, the accuracy falls to a random guess immediately.** If the model cannot reach a position, it cannot use it. Longer training does not help.
- **When only the last layer uses full attention, all distances go back to 100%.** The KV cache stores only this one layer more.

This is the full reason for local-global interleaving: **the local layers process the nearby context cheaply, and a few global layers do the long-distance retrieval**. It is also a reminder: do not evaluate a long-context model only with perplexity. Use dedicated retrieval evaluations (needle in a haystack, RULER, and similar; Chapter 15).

## 7. Sparse attention: select by content, not by position

The problem of the sliding window is that its selection of keys is rigid: it is always "the last W". If each query can pay for only k keys, why not select the **most relevant** k keys? This is the idea of **sparse attention**.

[`code/05_topk_sparse.py`](code/05_topk_sparse.py) takes the **full-attention** models that Section 6 trains. At inference, each query keeps only k keys, and there is no retraining. The script compares two selection methods:

```python
kth = att.topk(self.topk, dim=-1).values[..., -1:]  # the k-th largest score of each row
att = att.masked_fill(att < kth, float("-inf"))  # the other keys become invisible
```

| Selection | k | Needle d < 16 | Needle d > 60 | All distances | LM loss |
|---|---:|---:|---:|---:|---:|
| No limit (full attention) | – | 100.0% | 100.0% | 100.0% | 1.797 |
| k most recent | 4 | 29.9% | 12.3% | 15.2% | 1.886 |
| k highest scores | 4 | 100.0% | 100.0% | 100.0% | 1.818 |
| k most recent | 8 | 54.9% | 12.3% | 19.2% | 1.838 |
| k highest scores | 8 | 100.0% | 100.0% | 100.0% | 1.806 |
| k most recent | 16 | 100.0% | 12.3% | 26.8% | 1.819 |
| k highest scores | 16 | 100.0% | 100.0% | 100.0% | 1.800 |

("k most recent" is below 100% even for d < 16. The model was trained with full attention. When inference suddenly uses a window of 4 or 8, even the needles inside the window are affected. At k = 16, the window covers d < 16, and the accuracy goes back to 100%.)

With the same budget, selection by position (k most recent) cannot find the distant needles at all. Selection by content (k highest scores) keeps 100% already at k = 4, and the language-modeling loss increases only a little. Attention is "sparse" by nature: most of the weight goes to a few keys.

But this demo cheats. To select the top k, it first calculates all T scores, so it saves nothing. Real systems do these steps:

1. A **very cheap indexer** gives a score to each history token. It has a small dimension and few heads, and it can even use low precision.
2. Each query selects the k tokens (or k blocks) with the highest scores.
3. The model calculates exact attention only on the selected tokens.

**Is this a consensus?** We use rule A of GOAL.md 2.1: at least 3 independent leading families use the method explicitly in their main versions. We verified the result in September 2026:

| Family | Main version | Method | Source |
|---|---|---|---|
| DeepSeek | V3.2 (DSA), V4 (CSA + HCA) | A lightning indexer (64 index heads, 128 dims) gives a score to each token; each query selects the top 2048 (512 for V4-Flash) | V3.2 model card "DeepSeek Sparse Attention (DSA)", `index_topk` 2048; V4 model card "Compressed Sparse Attention", `index_topk` 512 |
| Zhipu GLM | GLM-5 / 5.3 | Uses DSA directly | GLM-5 model card "GLM-5 also integrates DeepSeek Sparse Attention (DSA)"; `GlmMoeDsaForCausalLM`, `index_topk` 2048 |
| MiniMax | MiniMax-M3 | MSA: blocks of 128 tokens; each GQA group selects 16 blocks (2048 tokens) | M3 model card "MiniMax Sparse Attention (MSA)"; `sparse_attention_config` |
| Meituan LongCat | LongCat-2.0 | LSA: adds levels and cross-layer sharing to the indexer of DSA | Model card "LongCat Sparse Attention"; `index_topk` 2048 |

There are four independent families. GLM and LongCat explicitly built on the DSA of DeepSeek, but they are independent model families, so they count under the rule. **This satisfies rule A, so the method is in the main text.** But we must give its exact scope:

- **All four are MoE flagships with more than 200 billion parameters** (DeepSeek-V4-Flash 284B, GLM-5 744B, MiniMax-M3 about 428B, LongCat-2.0 1.6T). Their target is a context of about 1 million tokens. Among small models below 10B, we found only MiniCPM4 / 4.1 (InfLLM-V2). And the newer MiniCPM5-2B from the same company uses a plain `LlamaForCausalLM` again in its `config.json`.
- **It saves compute, not KV cache.** Nobody knows in advance which token a later query will select. Thus the model must keep all K/V, and also a small key of the indexer itself. This is the opposite of a sliding window. DeepSeek V4 also adds compression (`compress_ratios`) and a sliding window of 128, and only these make the cache smaller.
- **The specific methods have not converged.** The families differ: select tokens or blocks, how to train the indexer, and whether to combine it with compression (see "Frontier notes" at the end).

Thus the position of this course is: **the idea "select the top k with a cheap scorer, then calculate exact attention" is a consensus, so it is in the main text. The specific variants go into the frontier notes. The main-line model (0.6–0.8B, 32K context) does not use it.**

## 8. Summary

- **Sliding window**: `mask = (j ≤ i) & (i − j < W)`. Each token sees only the last W positions. The compute decreases from O(T²) to O(T·W), and the KV cache has a maximum of W.
- **Receptive field**: L sliding-window layers can relay information over `L × (W − 1)` positions in theory. In the experiment, retrieval beyond this distance falls to a random guess immediately.
- **Local-global interleaving**: most layers are local, and a few layers are global (5:1 for Gemma 3, 3:1 for OLMo 3, 1:1 for gpt-oss). The global layers do long-distance retrieval, and the local layers save cache. At 128K, Gemma-3-27B saves 83%.
- **The loss does not show the cost, but the needle in a haystack does**: evaluate long context with retrieval tasks.
- **Sparse attention**: select the top k by content, not by position. A cheap indexer + exact attention. The flagships of DeepSeek, GLM, MiniMax, and LongCat use it. It saves compute, not KV cache.

---

## GPU measurements (one RTX 3090)

> **Note:** The numbers in the main text above all come from CPU runs. This section uses one NVIDIA GeForce RTX 3090: 24 GB of memory, Ampere architecture. Spec sheet: dense BF16 tensor-core peak about 71 TFLOPS, FP32 about 35.6 TFLOPS, memory bandwidth about 936 GB/s. Environment: PyTorch 2.11.0+cu128, CUDA 12.8, October 2026.
>
> The server sets the power limit of this card to 240 W (the factory default is 350 W). Under a continuous full load, the card decreases its clock speed. Thus the absolute compute and bandwidth are lower than on a 3090 at full power, and the relative values are more reliable. If you have no GPU, skip this section.

Run:

```bash
uv run python chapters/22-local-sparse-attention/code/06_gpu_flex_window.py
```

We calculate the same sliding-window attention on the GPU with three algorithms. The condition is `(j ≤ i) & (i − j < W)` from Section 2. The setup is batch 1, 16 heads, head_dim 128, BF16:

- **Dense causal SDPA**: the FlashAttention kernel of PyTorch, with `is_causal=True`. It also calculates everything outside the window. It calculates full causal attention, so it is only a speed reference.
- **Boolean-mask SDPA**: first build a T × T boolean mask, then give it to the memory-efficient kernel of SDPA (`attn_mask=`). It calculates the scores outside the window first and then masks them. `zero/arch/sliding_window.py` uses this method now.
- **FlexAttention**: `create_block_mask` first divides the T × T table into blocks of 128 × 128 and marks the blocks that are fully outside the window. The kernel skips these blocks and calculates only the blocks on the band. The first call must use `torch.compile` to generate a Triton kernel. On this machine, this took 7 seconds.

Fix T = 16,384 and change the window W (milliseconds, median of 10 runs):

| Window W | Blocks that FlexAttention calculates (fraction of all causal blocks) | Dense causal SDPA | Boolean-mask SDPA | FlexAttention | FlexAttention speedup over dense |
|---:|---:|---:|---:|---:|---:|
| 128 | 3.1% | 22.68 | 70.19 | 0.90 | 25.2× |
| 512 | 7.6% | 23.35 | 70.53 | 2.47 | 9.5× |
| 1,024 | 13.5% | 23.93 | 70.80 | 3.64 | 6.6× |
| 4,096 | 44.8% | 23.25 | 71.48 | 9.37 | 2.5× |
| 16,384 (= T, which is full causal) | 100.0% | 23.70 | 73.37 | 19.88 | 1.2× |

At W = 1,024, the extra GPU memory in addition to q, k, and v was: dense 65 MiB; boolean-mask SDPA first needs a mask of 256 MiB, and then 576 MiB more at run time; FlexAttention 66 MiB.

Fix W = 1,024 and change the sequence length T (milliseconds):

| Sequence length T | Dense causal SDPA | Boolean-mask SDPA | FlexAttention | FlexAttention speedup over dense |
|---:|---:|---:|---:|---:|
| 4,096 | 1.62 | 5.00 | 0.94 | 1.7× |
| 16,384 | 22.72 | 71.24 | 3.61 | 6.3× |
| 65,536 | 473.55 | (skipped: the boolean mask alone needs 4 GiB) | 12.06 | 39.3× |

The three algorithms calculate the same result. At T = 16,384, the maximum absolute difference between FlexAttention and boolean-mask SDPA is 3.9e-03 (7.8e-03 at W = 128). At W = T, dense causal SDPA and boolean-mask SDPA also differ by 3.9e-03. All these differences are at the level of BF16 rounding.

Section 2 says that "the compute for each query changes from O(T) to O(W)". These numbers show that this is true on a GPU only under one condition: **the kernel must really skip the blocks outside the window**. The time of FlexAttention follows the number of blocks to calculate. When W increases from 128 to 4,096, its time increases from 0.90 ms to 9.37 ms. At a fixed W = 1,024, each time T increases 4×, its time increases only 3–4× (linear). The time of dense causal attention increases 14–21× (quadratic), and at T = 65,536 FlexAttention is 39× faster.

In contrast, the boolean mask "adds only one `&`", but it saves nothing on a GPU. For all window sizes, it takes about 70 ms, which is 3× the time of full causal FlashAttention. It calculates all T × T scores, also the upper triangle that the causal mask removes. It also needs an extra mask and bias of size T². A small surprise: at W = T, FlexAttention (19.88 ms) was a little faster than the FlashAttention of PyTorch (23.70 ms). FlexAttention worked on the 3090 with PyTorch 2.11 without changes, and we found no compatibility problems.

## From minimal code to production code

The production implementation is in [`zero/arch/sliding_window.py`](../../zero/arch/sliding_window.py). It is an experimental module of Part 5 and is **not used in the main-line model**:

| Minimal code (`code/`) | Production code (`zero/arch/sliding_window.py`) | What it adds, and why |
|---|---|---|
| `window_mask(q_pos, k_pos, window)` | `sliding_window_mask(q_pos, k_pos, window)` | The same condition. It also handles "empty slots" (position −1 is never visible), and the keys can be in any order (the ring buffer needs this). |
| The windows of each layer are written by hand in `VARIANTS` | `make_layer_types(n_layers, global_every=..., all_sliding=...)` | Gives the same strings as HF `config.layer_types` (`"sliding_attention"` / `"full_attention"`). `global_every=6` is the 5:1 of Gemma 3. The convention agrees with `(i + 1) % pattern == 0` in HF. |
| Its own small `Attention` | `SlidingWindowAttention(zero.model.Attention)` + `convert_to_sliding_window(model, layer_types, window)` | Inherits the main-line attention: GQA, QK-Norm, RoPE, and all parameter names are the same. It can change any `zero` model in place to local-global interleaving and keep the weights. Uses PyTorch SDPA with a boolean mask. |
| `KVCache.append`: concatenate, then slice and keep the last W | `SlidingWindowKVCache`: global layers preallocate `max_seq_len`; a sliding-window layer allocates only a **ring buffer** of W slots (position p → slot `p % W`) and also stores the position of each slot | Allocates once and never copies. Supports chunked prefill with more than W tokens at a time (first calculate attention with "old + new", then write back only the last W). `nbytes()` agrees with the formula of `kv_cache_bytes(...)`. |
| Recalculate the full sequence at each step / truncated cache | `generate_greedy(model, prompt, n, cache=None)` | Made for the parity check of "bounded cache" against "no cache". |
| `kv_bytes` in `01_masks_and_ledger.py` | `kv_cache_bytes(layer_types, window, ...)` in `zero/arch/sliding_window.py`. The full ledger that reads the layer types from `config.json` automatically is `zero/tools/kv_cache_calc.py` of Chapter 21 (it supports `sliding` layers) | The same formula: a global layer stores T positions, and a sliding-window layer stores min(W, T). |
| Hand-written `q @ kᵀ` + mask | SDPA + custom mask (CPU) | On a GPU, to really skip the blocks outside the window, you need `flash_attn_func(..., causal=True, window_size=(W − 1, 0))` of FlashAttention, or a sliding-window `mask_mod` for PyTorch FlexAttention. vLLM and transformers read `sliding_window` and `layer_types` and select the kernel and the per-layer cache automatically. We measured the block-sparse sliding window of FlexAttention on an RTX 3090 (section "GPU measurements" of this chapter: at T = 65,536, it is about 39× faster than dense causal attention). We did not verify the `window_size` of FlashAttention, because `flash_attn` is not installed. `zero/arch/sliding_window.py` still uses boolean-mask SDPA on CUDA. Its correctness is verified (Section 11 of [runs/2026-10-01-gpu0-check](../../runs/2026-10-01-gpu0-check/README.md)), but it is slower than full attention. |

**Parity check**: [`tests/test_arch_sliding_window.py`](../../tests/test_arch_sliding_window.py) (`uv run pytest tests/test_arch_sliding_window.py`; on this machine, all 8 tests passed in a few seconds):

- The mask agrees with the definition cell by cell (a 6×6 table with W = 3). Each query sees at most W positions. Empty slots are invisible, and the keys can be in any order.
- The three layouts of `make_layer_types`: 5:1, 1:1, and all sliding.
- **When the window is at least the sequence length, the logits of the sliding-window model are exactly the same as the logits of full attention.** With a window of 8, the first 8 positions agree, and the positions after them start to differ.
- **Greedy generation with the ring cache gives the same tokens as generation without a cache.** We tested all sliding and 1:1 interleaving, with a prompt longer than the window and 60 generated tokens.
- Chunked prefill (17 + 1 + 3 + 24 tokens, chunks of different sizes) agrees with one full forward pass within 1e-5.
- The plain `zero.kv_cache.KVCache` also gives the same result (it only stores K/V that the model does not use).
- `nbytes()` is equal to the formula "3 layers × 8 slots + 1 layer × 128 positions".

Sparse attention has no production implementation. It needs a specially trained indexer and custom kernels. (DeepSeek released a sparse version of FlashMLA and the indexer kernels of DeepGEMM as open source, and MiniMax released the MSA operator.) This is beyond the scope of the CPU experiments of this course, and the main-line model does not need it.

---

## Frontier notes

> **The specific variants of sparse attention still diverge.** The consensus covers only the level "cheap scoring + top-k + exact attention". Below that level, the families differ. DeepSeek proposed **NSA** first (Native Sparse Attention: the sum of three branches, compression, block selection, and sliding window; a 2025 paper, not seen in a flagship). Then it proposed **DSA** (a lightning indexer for each token, V3.2), and then **CSA/HCA** (V4: compress the KV by 4× or 128×, then select sparsely; `compress_ratios` alternates layer by layer).
>
> Kimi proposed **MoBA** (MoE-like routing over blocks), but the main architecture of Kimi K3 is a hybrid of MLA and KDA linear attention, without MoBA. **InfLLM-V2** from ModelBest (it can switch between dense and sparse, and it adds no parameters) is used in MiniCPM4 / 4.1, but MiniCPM5 did not keep it. Some questions have no answer yet: select tokens or blocks, how to train the indexer, and whether to combine it with KV compression. (DeepSeek and MiniMax both warm up with full attention first. During the warmup, the indexer uses KL divergence to fit the distribution of the main attention. Then they switch to sparse attention, but the details differ.)

> **Attention sink.** StreamingLLM (Xiao et al. 2023) found that a model with only a sliding window collapses after it drops the first few tokens. The reason is that many attention heads put their "extra" attention on the start of the sequence. If the model keeps the first few tokens (or adds a learnable sink), it can generate a stream without end, in a stable way. gpt-oss adds a learnable sink to each attention head (model card). The configuration of MiMo-V2-Flash has `add_swa_attention_sink_bias`, and MiniMax reports that MSA does not need to force-keep the start. The attention sink works closely with the sliding window, but the methods differ, so it is not yet a consensus.

---

## Adopters and sources

| Technique | Adopters (main versions) | Sources |
|---|---|---|
| Sliding window + local-global interleaving | **Google Gemma** (Gemma 2: 1:1, 4096; Gemma 3: 5:1, 1024; Gemma 4: 5:1, 1024), **OpenAI gpt-oss** (1:1, 128), **AllenAI OLMo 3** (3:1, 4096), **Mistral** (Mistral 7B v0.1 all sliding 4096; Ministral 8B 3:1), **StepFun** Step-3.5-Flash (3:1, 512), **Xiaomi MiMo**-V2-Flash (5:1, 128); Meta Llama 4 uses chunked attention (3:1, 8192) | Gemma 2 arXiv:2408.00118; Gemma 3 arXiv:2503.19786; gpt-oss model card arXiv:2508.10925; Mistral 7B arXiv:2310.06825; the `config.json` of each model (links below) |
| Learned sparse attention (indexer + top-k) | **DeepSeek** (V3.2 DSA, V4 CSA), **Zhipu GLM** (GLM-5, 5.3: DSA), **MiniMax** (M3: MSA), **Meituan LongCat** (2.0: LSA) | DeepSeek-V3.2 model card and technical report; DeepSeek-V4 arXiv:2606.19348; GLM-5 arXiv:2602.15763; MiniMax Sparse Attention arXiv:2606.13392; LongCat-2.0 model card |
| Window support in FlashAttention | Industry-standard kernel (class B of GOAL.md 2.1) | <https://github.com/Dao-AILab/flash-attention> (`window_size` parameter) |

**Consensus decision (GOAL.md 2.1)**:

- Sliding window / local-global interleaving: at least six independent families (Gemma, gpt-oss, OLMo, Mistral, StepFun, MiMo) explicitly use it in the configurations of their main versions. This satisfies rule A.
- Learned sparse attention: four independent families (DeepSeek, GLM, MiniMax, LongCat) explicitly use it in flagships from 2025-09 to 2026-07. **This satisfies rule A, so it is in the main text.** But it is limited to MoE flagships with more than 200 billion parameters and contexts of about 1 million tokens. The specific variants are in "Frontier notes". When we wrote Chapter 21, we had verified only two families, DeepSeek and GLM. This chapter adds MiniMax-M3 and LongCat-2.0 (both released in 2026).

**Model configurations and model cards** (read through Hugging Face in 2026-09):
[gemma-2-9b](https://huggingface.co/google/gemma-2-9b/blob/main/config.json),
[gemma-3-1b-pt](https://huggingface.co/google/gemma-3-1b-pt/blob/main/config.json),
[gemma-3-27b-pt](https://huggingface.co/google/gemma-3-27b-pt/blob/main/config.json),
[gemma-4-31B](https://huggingface.co/google/gemma-4-31B/blob/main/config.json),
[gemma-4-E2B-it](https://huggingface.co/google/gemma-4-E2B-it/blob/main/config.json),
[gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b/blob/main/config.json),
[gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b/blob/main/config.json),
[Olmo-3-1025-7B](https://huggingface.co/allenai/Olmo-3-1025-7B/blob/main/config.json),
[Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1/blob/main/config.json),
[Mistral-7B-v0.3](https://huggingface.co/mistralai/Mistral-7B-v0.3/blob/main/config.json),
[Ministral-8B-Instruct-2410](https://huggingface.co/mistralai/Ministral-8B-Instruct-2410/blob/main/config.json),
[Step-3.5-Flash](https://huggingface.co/stepfun-ai/Step-3.5-Flash/blob/main/config.json),
[MiMo-V2-Flash](https://huggingface.co/XiaomiMiMo/MiMo-V2-Flash/blob/main/config.json),
[Llama-4-Scout (unsloth mirror)](https://huggingface.co/unsloth/Llama-4-Scout-17B-16E-Instruct/blob/main/config.json),
[DeepSeek-V3.2](https://huggingface.co/deepseek-ai/DeepSeek-V3.2) ([V3.2-Exp config](https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/config.json)),
[DeepSeek-V4-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/main/config.json),
[GLM-5](https://huggingface.co/zai-org/GLM-5) ([config](https://huggingface.co/zai-org/GLM-5/blob/main/config.json)),
[GLM-5.3 config](https://huggingface.co/zai-org/GLM-5.3/blob/main/config.json),
[MiniMax-M3](https://huggingface.co/MiniMaxAI/MiniMax-M3) ([config](https://huggingface.co/MiniMaxAI/MiniMax-M3/blob/main/config.json)),
[LongCat-2.0](https://huggingface.co/meituan-longcat/LongCat-2.0) ([config](https://huggingface.co/meituan-longcat/LongCat-2.0/blob/main/config.json)),
[MiniCPM4.1-8B](https://huggingface.co/openbmb/MiniCPM4.1-8B),
[MiniCPM5-2B config](https://huggingface.co/openbmb/MiniCPM5-2B/blob/main/config.json),
[Kimi-K3 config](https://huggingface.co/moonshotai/Kimi-K3/blob/main/config.json),
[Qwen3.5-0.8B config](https://huggingface.co/Qwen/Qwen3.5-0.8B/blob/main/config.json).

Notes: the `config.json` of Gemma 2 has only `sliding_window` 4096 and no `layer_types`, so the "1:1 alternation" comes from the technical report. The configuration of Gemma 3 27B does not include `sliding_window_pattern` (the 1B version has 6), so the "5:1" comes from the technical report and the default value in transformers. The official Llama 4 repository needs an access request, so the numbers come from the unsloth mirror. "3 chunked layers + 1 global NoPE layer" is our interpretation of `no_rope_layers` and of the Llama 4 implementation in transformers. This interpretation is **to be verified** (Meta published no technical report, only a blog post). Qwen (hybrid linear attention from 3.5, Chapter 23) and Kimi (KDA hybrid in K3) do not use sliding windows or sparse attention.

---

## Guided questions

1. Must sliding-window layers and global layers use the same RoPE base frequency? Look at the configurations of Gemma 3 (`rope_local_base_freq` 10000 vs `rope_theta` 1 million) and gpt-oss. Think about this: a layer sees only 128 positions. Does it need the low-frequency dimensions of RoPE, which "need hundreds of thousands of positions for one full turn"?
2. In the needle in a haystack of this chapter, the pure sliding window has 100% for d ≤ 60. If the filler characters change to "distracting" content (for example, characters that also appear as needles), is the relay still this perfect? When intermediate tokens pass the information on, why does it become more and more "crowded"?
3. Two designs both save KV cache: "sliding window in all layers + a large window" and "interleaving + a small window". What are the advantages and disadvantages of each? Why did Mistral later turn off the sliding window, while Gemma uses it more and more?
4. What is the difference between chunked attention (`attention_chunk_size` of Llama 4) and a sliding window? What occurs at the chunk boundaries? What are the advantages for the KV cache and for the kernel?
5. Sparse attention such as DSA does not save KV cache. Then what does it save in decode? (Hint: Chapter 21 says that memory bandwidth limits decode. How much K/V must each step read?)
6. If the main-line model (0.6–0.8B, 32K context) changed to 3:1 local-global interleaving, how much KV cache would it save? What could the cost be? Why does GOAL.md 3.3 still decide not to take this risk?

## Hands-on tasks

**Task 1 (basic)**: Change `MODELS` in `01_masks_and_ledger.py`. Add Gemma-3-1B (26 layers, 1 KV head, head_dim 256, window 512, 5:1) and gpt-oss-20b (24 layers, 8 KV heads, head_dim 64, window 128, 1:1). Calculate how much each model saves at 128K. Then think about why Gemma 3 1B decreased the window from 1024 to 512.

**Task 2 (core)**: In `VARIANTS` of `02_swa_model.py`, add `"interleave_first": [None, 16, 16, 16]` (the global layer is the **first** layer). Train the needle-in-a-haystack model again and compare it with `[16, 16, 16, None]`. Does the position of the global layer matter? Why? (One training run takes about 5–10 minutes.)

**Task 3 (challenge)**: Add a real "indexer" to the `Attention` of `02_swa_model.py`: a pair of small projections `wq_idx` and `wk_idx` with only 8 dimensions. Use `q_idx · k_idx` to select the top k, then calculate normal attention on the selected keys. First freeze the full-attention model and train only the indexer: its score distribution must fit the distribution of the main attention (KL divergence, the method of MiniMax MSA). Then use the evaluation of `05_topk_sparse.py` to see how much long-distance needle accuracy it keeps at k = 8.

---

## Go deeper: CS336

This chapter matches Stanford CS336 (Spring 2026) <https://cs336.stanford.edu/>:

- **Lecture 4: Alternatives to attention, and MoE.** It explains why we want to replace full attention, and which paths exist (local/sparse, linear, state space). This chapter expands the "local and sparse" path. The slides and videos are on the course page.
- **Lecture 3: Architectures and hyperparameters** compares the architecture choices of different families. It also covers local-global interleaving.
- **CS336 does not go deep into** the learned sparse attention of the second half of this chapter (DSA, MSA, and other methods from 2025–2026). See the references of this chapter.

---

## References

- Child, Gray, Radford, Sutskever. *Generating Long Sequences with Sparse Transformers*, 2019: <https://arxiv.org/abs/1904.10509>
- Beltagy, Peters, Cohan. *Longformer: The Long-Document Transformer* (sliding window + global tokens), 2020: <https://arxiv.org/abs/2004.05150>
- Jiang et al. *Mistral 7B* (sliding window, rolling buffer cache, theoretical span), 2023: <https://arxiv.org/abs/2310.06825>
- Gemma Team. *Gemma 2: Improving Open Language Models at a Practical Size*, 2024: <https://arxiv.org/abs/2408.00118>
- Gemma Team. *Gemma 3 Technical Report* (5:1 local-global, window 1024, ablations of KV cache and perplexity), 2025: <https://arxiv.org/abs/2503.19786>
- OpenAI. *gpt-oss-120b & gpt-oss-20b Model Card* (alternating banded window attention, learnable sink), 2025: <https://arxiv.org/abs/2508.10925>
- Xiao et al. *Efficient Streaming Language Models with Attention Sinks* (StreamingLLM), 2023: <https://arxiv.org/abs/2309.17453>
- Yuan et al. *Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention* (NSA), 2025: <https://arxiv.org/abs/2502.11089>
- Lu et al. *MoBA: Mixture of Block Attention for Long-Context LLMs*, 2025: <https://arxiv.org/abs/2502.13189>
- DeepSeek-AI. *DeepSeek-V3.2* (DSA) technical report: <https://huggingface.co/deepseek-ai/DeepSeek-V3.2/blob/main/assets/paper.pdf>
- DeepSeek-AI. *DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence*, 2026: <https://arxiv.org/abs/2606.19348>
- GLM Team. *GLM-5*, 2026: <https://arxiv.org/abs/2602.15763>
- Lai et al. *MiniMax Sparse Attention*, 2026: <https://arxiv.org/abs/2606.13392>
- MiniCPM Team. *MiniCPM4*, 2025: <https://arxiv.org/abs/2506.07900>; *InfLLM-V2*, 2025: <https://arxiv.org/abs/2509.24663>
- Dao. FlashAttention (`window_size` parameter): <https://github.com/Dao-AILab/flash-attention>; PyTorch FlexAttention blog (sliding-window `mask_mod` example): <https://pytorch.org/blog/flexattention/>
- [The main LLM architectures in 15,000 characters (Llama, Qwen, GLM, DeepSeek…)](https://zhuanlan.zhihu.com/p/2060741715095560795) (in Chinese): a side-by-side comparison of the attention structures of each family (already in `references.md`)
- The links to the model configurations are in "Adopters and sources" above.

**Next chapter**: A sliding window lets some layers stop storing K and V for distant positions. Sparse attention lets each query read only a small part of K and V. But the cache of the global layers and the sparse layers still grows with the length. Chapter 23 takes a more radical path: it stores no K and V at all, and it compresses the full history into a state of fixed size. This is linear attention, and the hybrid architecture of Qwen3.5 with "3 linear layers + 1 full-attention layer".
