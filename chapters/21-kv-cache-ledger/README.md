# Chapter 21: The KV cache budget — Why long context is expensive, and how much each token must store

**English** · [中文](README.zh.md)

> **Goal**: After this chapter, you can explain why long context is expensive in each of the two phases, prefill and decode. You can calculate the KV cache size layer by layer from the `config.json` of any open model, for four kinds of layers: full attention, sliding window, hybrid linear attention, and MLA. You can also explain how MLA compresses K and V into one latent vector, and how inference "absorbs" the step that reconstructs them.

📺 **Video**: Not published yet. To render it on your computer, run `bash chapters/21-kv-cache-ledger/video/build.sh`.
🧪 **Self-check**: After the chapter, type `/ch21-kv-cache` in Claude Code.

---

At the end of Part 4, we had a main-line model that can call tools. Part 5 changes the point of view. In this part, we do not train the main-line model. Instead, we look at how the **architecture** of open models changed in the last two years. Almost all of these changes go in the same direction: **longer context and a smaller KV cache**.

In Chapter 10, we did this calculation for the first time. The main-line model caches 112 KiB for each token. One conversation of 32K tokens needs 3.5 GiB, which is more than the model weights (1.28 GiB). GQA uses 8 KV heads instead of 16 and saves half.

This chapter answers two questions: **Why is long context expensive? How else can we make the KV cache smaller?** First, we split the cost into two accounts: prefill compute and decode memory bandwidth. Then we change the formula of Chapter 10 into a "ledger" that counts each layer separately. We use this ledger on some of the newest open models. Finally, we follow the compression path MQA → GQA → MLA, and we compare five kinds of attention in the same small model on the CPU.

The code for this chapter:

```bash
uv run python chapters/21-kv-cache-ledger/code/01_kv_ledger.py          # ledger: main-line model + 12 public models
uv run python chapters/21-kv-cache-ledger/code/02_prefill_decode.py     # prefill compute / decode bandwidth / number of conversations
uv run python chapters/21-kv-cache-ledger/code/03_mla.py                # minimal MLA: parity check of the absorbed and explicit paths
uv run python chapters/21-kv-cache-ledger/code/04_attention_variants.py # MHA/GQA/MQA/MLA comparison (first run: about half an hour)
```

## 1. Why long context is expensive: two phases, two bottlenecks

Chapter 10 showed that with a KV cache, inference has two phases. **Prefill** gives the full prompt to the model in one pass. **Decode** then gives only one new token at each step. Long context is expensive in these two phases for completely different reasons.

### 1.1 Prefill: compute grows as T²

The forward operations of prefill have two parts (`prefill_flops` in `02_prefill_decode.py`):

```python
linear = 2 * n_matmul * T                  # matmul: one multiply-add per parameter per token = 2 operations
attn = 2 * c.n_layers * c.q_dim * T * T    # QKᵀ and AV: 4·q_dim per pair (i, j); the causal mask leaves only T²/2 pairs
```

The first term grows linearly with T. The second term grows as T². We use the main-line model: N = 689.4M parameters in matrix multiplications, and q_dim = 16 × 128 = 2048. We calculate the theoretical time at the dense BF16 peak of an H100 SXM, 989.5 TFLOPS:

| Prompt length T | Matmul part (operations) | Attention part (operations) | Attention share | H100 lower bound |
|---:|---:|---:|---:|---:|
| 1,024 | 1.41 × 10¹² | 1.20 × 10¹¹ | 7.8% | 1.5 ms |
| 4,096 | 5.65 × 10¹² | 1.92 × 10¹² | 25.4% | 7.7 ms |
| 32,768 | 4.52 × 10¹³ | 1.23 × 10¹⁴ | **73.2%** | 170 ms |
| 131,072 | 1.81 × 10¹⁴ | 1.97 × 10¹⁵ | **91.6%** | 2.2 s |

Up to 4K, attention is only a small part of the compute. At 128K, more than 90% of the compute goes to attention. This is why the time to first token becomes much worse when the context becomes longer. The sliding windows and the linear attention of Chapters 22 and 23 solve this problem. **The KV cache itself does not change the compute of prefill.** This chapter is mainly about decode, in the next section.

### 1.2 Decode: limited by memory bandwidth

Each decode step calculates only one token. But for this one token, the GPU must **read all the weights once**. It must also **read the full KV cache of the conversation once**. Let the batch have B conversations, each with T tokens already (`decode_step`):

```python
flops = B * (2 * n_matmul + 4 * c.n_layers * c.q_dim * T)   # operations to do
w_bytes = n_params * BF16                                     # bytes to read: weights (shared by all B)
kv = B * T * kv_per_token                                     # bytes to read: the KV cache of each conversation
t = max(flops / PEAK_FLOPS, (w_bytes + kv) / HBM_BW)          # theoretical minimum time of one step
```

Operations ÷ bytes read is the **arithmetic intensity**. Each second, an H100 SXM can do 989.5 trillion operations and read 3.35 TB. The ratio of the two is about 295 operations per byte. This ratio is the **ridge point**. When the intensity is below it, the GPU waits for data. This is **memory-bound**. These are the values for the main-line model:

| Context T | Batch B | Weights read | KV cache read | Arithmetic intensity | Theoretical step time | Throughput (token/s) | Fits in 80 GB? |
|---:|---:|---:|---:|---:|---:|---:|---|
| 4,096 | 1 | 1.28 GiB | 0.44 GiB | 1.3 | 0.55 ms | 1,812 | Yes |
| 4,096 | 16 | 1.28 GiB | 7.00 GiB | 4.2 | 2.66 ms | 6,026 | Yes |
| 4,096 | 64 | 1.28 GiB | 28.00 GiB | 4.7 | 9.39 ms | 6,819 | Yes |
| 32,768 | 1 | 1.28 GiB | 3.50 GiB | 1.7 | 1.53 ms | 652 | Yes |
| 32,768 | 16 | 1.28 GiB | 56.00 GiB | 2.3 | 18.36 ms | 871 | Yes |
| 32,768 | 64 | 1.28 GiB | 224.00 GiB | 2.4 | 72.21 ms | 886 | **No** |

The arithmetic intensity is only 1–5, two orders of magnitude below 295. Decode is fully memory-bound.

Chapter 10 said: "Put many requests into one batch, and one read of the weights serves tens of requests." This method works for short context: at 4K, when the batch grows from 1 to 16, the throughput increases 3.3 times. **But for long context, the method fails.** At 32K, the KV cache (3.5 GiB per conversation) is much larger than the weights (1.28 GiB). Each conversation reads its own KV cache, so a larger batch does not save these reads. When the batch grows from 1 to 16, the throughput increases only 1.3 times.

Thus, for long context, the decode speed depends almost only on the size of the KV cache per token.

### 1.3 Memory: how many conversations fit at the same time

There is also a stricter limit: the data must fit in memory. Take 80 GB, subtract the weights, and give all the rest to the KV cache. This ignores activations and fragmentation, so the result is optimistic. The table gives the number of conversations:

| Context | GQA (main-line, 112 KiB/token) | If MHA (224 KiB) | If MLA 512+64 (31.5 KiB) |
|---:|---:|---:|---:|
| 4,096 | 167 | 83 | 595 |
| 32,768 | 20 | 10 | 74 |
| 131,072 | 5 | 2 | 18 |

When the KV cache per token becomes half as large, the same GPU can serve twice as many conversations at the same time. Decode also reads half as many bytes. This is the motivation for all of Part 5.

## 2. The ledger: count layer by layer

The formula of Chapter 10 assumes that all layers are the same:

```
KV cache bytes = 2 × layers × KV heads × head_dim × sequence length × bytes per number (× batch)
```

But in the open models of 2025–2026, the layers are not all the same. Some layers look only at the most recent 128 tokens (sliding window). Some layers store no K and V at all (linear attention). Some layers store a compressed latent vector instead of K and V (MLA). Thus the ledger must count each layer:

| Layer type | What it stores per layer per position | How it grows with sequence length | Where the course explains it |
|---|---|---|---|
| Full attention (MHA / GQA / MQA) | `KV heads × head_dim` numbers for K, and the same for V | Grows linearly | Chapters 8 and 10; Section 4 of this chapter |
| MLA | Latent vector `kv_lora_rank` + shared RoPE key `qk_rope_head_dim` | Grows linearly, but much less per position | Section 4 of this chapter |
| Sliding window | The same as full attention | Keeps at most `window` positions, then stops growing | Chapter 22 |
| Linear attention (Gated DeltaNet, KDA, and others) | No K/V; only a state matrix of fixed size | Does not grow | Chapter 23 |

The core of `01_kv_ledger.py` is these lines:

```python
def layer_list(m):
    if m.get("kv_lora_rank"):                                  # MLA
        per, kind = m["kv_lora_rank"] + m["qk_rope_head_dim"], "mla"
    else:                                                      # MHA / GQA / MQA
        per, kind = 2 * m["kv_heads"] * m["head_dim"], "full"
    ...

def kv_bytes(m, seq_len, batch=1, nbytes=BF16):
    total = 0
    for kind, per, window in layer_list(m):
        kept = min(seq_len, window) if kind == "sliding" else seq_len  # sliding window: keep at most `window`
        total += per * kept                                            # linear layer: per = 0
    return total * batch * nbytes
```

## 3. The ledger for the main-line model and the newest models

Run `01_kv_ledger.py`. First, look at the main-line model (28 layers, 16 query heads, 8 KV heads, head_dim 128, BF16) with different kinds of attention:

| Variant | Stored per layer per position | Per token | 32K | 128K |
|---|---:|---:|---:|---:|
| MHA (16 KV heads) | 4,096 numbers | 224 KiB | 7.00 GiB | 28.00 GiB |
| **GQA (main-line, 8 KV heads)** | **2,048** | **112 KiB** | **3.50 GiB** | **14.00 GiB** |
| MQA (1 KV head) | 256 | 14 KiB | 0.44 GiB | 1.75 GiB |
| Assumed MLA (512 + 64) | 576 | 31.5 KiB | 0.98 GiB | 3.94 GiB |

Then look at 12 public models. All fields come from the `config.json` in the Hugging Face repository of each model. We read them in 2026-09; the links are in "Adopters and sources" at the end. The values use BF16 and batch 1, and they count only the part that grows with the sequence:

| Model | Attention | Growing layers / all layers | Per token | 32K | 128K | Reference: 128K if all layers are MHA with the same heads |
|---|---|---:|---:|---:|---:|---:|
| Qwen3-0.6B | GQA 16/8, head_dim 128 | 28 / 28 | 112 KiB | 3.50 GiB | 14.00 GiB | 28.00 GiB |
| Qwen3-8B | GQA 32/8 | 36 / 36 | 144 KiB | 4.50 GiB | 18.00 GiB | 72.00 GiB |
| Llama-3.1-8B | GQA 32/8 | 32 / 32 | 128 KiB | 4.00 GiB | 16.00 GiB | 64.00 GiB |
| gpt-oss-120b | GQA 64/8, head_dim 64; half of the layers use a sliding window of 128 | 36 / 36 (18 layers capped at 128) | 72 KiB¹ | 1.13 GiB | 4.50 GiB | 72.00 GiB |
| Qwen3.5-0.8B | GQA 8/2, head_dim 256; 1 full-attention layer in each 4 layers | 6 / 24 | 12 KiB | 0.38 GiB | 1.50 GiB | 24.00 GiB |
| Qwen3.5-9B | GQA 16/4, head_dim 256; the same pattern | 8 / 32 | 32 KiB | 1.00 GiB | 4.00 GiB | 64.00 GiB |
| Qwen3.5-397B-A17B | GQA 32/2, head_dim 256; the same pattern | 15 / 60 | 30 KiB | 0.94 GiB | 3.75 GiB | 240.00 GiB |
| DeepSeek-V3 / V3.2 | MLA 512 + 64, 128 heads | 61 / 61 | 68.6 KiB | 2.14 GiB | 8.58 GiB | 488.00 GiB |
| Kimi-K2 | MLA 512 + 64, 64 heads | 61 / 61 | 68.6 KiB | 2.14 GiB | 8.58 GiB | 244.00 GiB |
| GLM-5 | MLA 512 + 64, 64 heads | 78 / 78 | 87.8 KiB | 2.74 GiB | 10.97 GiB | 312.00 GiB |
| Mistral-Large-3 | MLA 512 + 64, 128 heads | 61 / 61 | 68.6 KiB | 2.14 GiB | 8.58 GiB | 488.00 GiB |
| Kimi-K3 | 24 MLA layers (512 + 64) + 69 KDA linear-attention layers | 24 / 93 | 27 KiB | 0.84 GiB | 3.38 GiB | 558.00 GiB |

¹ For gpt-oss, "per token" is the increase before the window is full. After 128 tokens, only the 18 full-attention layers still grow (36 KiB per token).
The reference column assumes MHA in every layer, with KV heads = query heads and head_dim 128 (256 for Qwen3.5). It is a hypothetical upper limit. It shows how much each method saves. In MLA models, the real K heads have a width of 192 (128 + 64), so MHA would be even larger. Here we use 128 to be conservative.

Note these points:

- **Almost all small models use GQA.** The KV configuration of the main-line model is the same as that of Qwen3-0.6B: 112 KiB per token.
- **There are three ways to save cache:**
  1. **Store fewer heads**: GQA / MQA. All small dense models use this method.
  2. **Store less at each position**: MLA. The flagship MoE models of DeepSeek, Kimi, GLM, and Mistral use it. 61 layers × 576 numbers is only 1/57 of MHA with the same number of heads.
  3. **Store fewer layers or fewer positions**: In Qwen3.5, only 1/4 of the layers use full attention. The other layers use Gated DeltaNet linear attention, which has a state of fixed size. In gpt-oss, half of the layers look only at the most recent 128 tokens. Chapters 22 and 23 explain these methods.
- **You can combine the three ways.** Qwen3.5 uses "GQA + hybrid linear attention", and Kimi K3 uses "MLA + hybrid linear attention". Qwen3.5-397B has about 400 billion parameters, but its KV cache at 128K context is only 3.75 GiB. This is about the same as the 0.7B main-line model at 32K.
- **Linear-attention layers also have a state, but the state does not grow with length.** The production calculator `zero/tools/kv_cache_calc.py` estimates this state separately. For example, the 24 Gated DeltaNet layers of Qwen3.5-9B need about 50 MiB with `mamba_ssm_dtype: float32`, for any context length. How an implementation stores the state and the convolution cache can be different, so this value is an estimate.

## 4. MQA → GQA → MLA

### 4.1 Review: store fewer heads

Chapter 10 explained these methods. In standard multi-head attention (**MHA**), each query head has its own K and V. In **MQA** (Shazeer 2019), all query heads share one set of K and V. **GQA** (Ainslie et al. 2023) is between the two: each group of query heads shares one set of K and V. All three methods save cache in the same way: they **decrease the number of KV heads**. The cost is that K and V can express less. All query heads that share one set of K and V see exactly the same keys and values.

### 4.2 MLA: compress K and V together into one latent vector

DeepSeek-V2 (2024) introduced **Multi-head Latent Attention (MLA)**. MLA uses a different idea: it does not decrease the number of heads. Instead, it uses one fact: the K and V of each head are linear transformations of the same input x. Thus MLA first compresses x into a small **latent vector** c_KV. Then it reconstructs the K and V of each head from c_KV:

```
c_KV = RMSNorm(W_DKV · x)          # down-projection: 7168 dims → 512 dims (DeepSeek-V3)     ← cached
k_i^C = W_UK,i · c_KV              # up-projection: reconstruct the key of head i (no position)
v_i   = W_UV,i · c_KV              # up-projection: reconstruct the value of head i
```

The cache stores only c_KV. This is **low-rank joint compression**. It is "low-rank" because the rank of the product W_UK · W_DKV is at most 512. It is "joint" because K and V share the same latent vector.

Compare the two methods. GQA says: "128 heads can use only 8 groups of K/V." It sets a fixed rule for which heads share. MLA says: "The K/V of all 128 heads must come from the same 512-dim vector." Training learns how the heads share. Each head still has its own W_UK,i and W_UV,i, so each head sees different keys and values.

The matching code in `03_mla.py`:

```python
self.wkv_a = nn.Linear(dim, kv_lora_rank + rope_dim, bias=False)            # down-projection: x → [c_KV ; k_R]
self.kv_norm = tiny.RMSNorm(kv_lora_rank)
self.wkv_b = nn.Linear(kv_lora_rank, n_heads * (nope_dim + v_dim), bias=False)  # up-projection [W_UK; W_UV]
...
c_kv, k_pe = self.wkv_a(x).split([r, dr], dim=-1)
c_kv = self.kv_norm(c_kv)                        # the latent vector to cache
k_pe = tiny.apply_rope(k_pe[:, None], cs, sn)    # the RoPE key to cache
```

### 4.3 Why RoPE must be "decoupled"

RoPE (Chapter 9) multiplies q and k by a rotation matrix that depends on the position. Suppose that we rotate k_i^C = W_UK,i · c_KV directly. Then the "absorb" method of the next section does not work. The rotation matrix is between W_UQ and W_UK, so we cannot multiply the two in advance (matrix multiplication is not commutative). At inference, we must then reconstruct the K at each position and rotate it. Then the cache of latent vectors saves nothing.

The solution of DeepSeek is **decoupled RoPE**. The query and the key of each head have two parts:

```
q_i = [ q_i^C ; RoPE(q_i^R) ]      # 128 dims without position + 64 dims with position
k_i = [ k_i^C ; RoPE(k^R)   ]      # k^R comes directly from x; all heads share one k^R
```

All heads share the small position part k^R (`qk_rope_head_dim` = 64), and the cache must also store it. Thus MLA caches **512 + 64 = 576 numbers** per layer per position. This is the meaning of `kv_lora_rank: 512` and `qk_rope_head_dim: 64` in the `config.json` of DeepSeek-V3, Kimi K2, GLM-5, and Mistral Large 3. MHA with the same 128 heads (K 192 and V 128 per head) stores 128 × 320 = 40,960 numbers. MLA stores only 1.4% of that. The DeepSeek-V2 paper says it this way: the MLA cache is equal to a GQA cache with only 2.25 groups.

### 4.4 Absorb: do not reconstruct K and V at inference

To reconstruct K and V, each step must multiply all T latent vectors in the cache by W_UK and W_UV. This wastes work. We can calculate the attention in a different order:

```
q_iᵀ k_j = (q_i^C)ᵀ W_UK,i c_KV,j + (q_i^R)ᵀ k^R_j
         = (W_UK,iᵀ q_i^C)ᵀ c_KV,j + (q_i^R)ᵀ k^R_j      ← first project the query into the latent space
Σ_j p_ij v_j = W_UV,i ( Σ_j p_ij c_KV,j )                  ← first average in the latent space, then project back
```

Matrix multiplication is associative. Thus W_UK can be "absorbed" into the query side, and W_UV can be absorbed into the output side. The latent vectors in the cache go directly into attention, and K and V are never reconstructed. `03_mla.py` has both paths:

```python
if self.absorb:  # absorb: project the query into the latent space; dot product with the cache
    q_lat = torch.einsum("bhtd,hdr->bhtr", q_nope, w_uk)
    att = q_lat @ c_kv.transpose(-2, -1) + q_pe @ k_pe.transpose(-2, -1)
else:            # explicit: first reconstruct the K of each head
    k_nope = torch.einsum("bxsr,hdr->bhsd", c_kv, w_uk)
    ...
```

The output (the same random weights, float64):

```
1. Absorbed path = explicit path (same weights, float64)
   max difference 4.4e-16
2. Token by token with a cache = one forward pass over the full sequence
   max difference 2.8e-16; shapes in the cache: latent (2, 1, 20, 32), RoPE key (2, 1, 20, 16)
```

The two paths are mathematically equivalent, and the difference is floating-point rounding. The form after absorption is interesting. Each head takes a 576-dim query and calculates a dot product with **the same** 576-dim cache. This is the form of MQA (all heads share one set of K/V), but here the "K/V" is a 576-dim latent vector. The GLM-5 report calls this form "the MQA mode of MLA". Thus you can think of MLA in this way: **in training, it acts like MHA (each head has its own K and V); in inference, it acts like MQA (it stores and reads only one copy).**

Real systems often use the explicit path for training and prefill. These phases process many tokens at once, and compute is the bottleneck. They use the absorbed path for decode, which processes one token at a time, and bandwidth is the bottleneck.

### 4.5 The costs of MLA

MLA has costs. At the time of writing, we found these:

- **Decode needs more compute.** After absorption, each head calculates dot products over 576 dims, but GQA usually uses 128 dims. For this reason, GLM-5 increased the head dimension from 192 to 256 and used 1/3 fewer heads, to decrease the decode compute. Kimi K2 also decreased the number of heads from 128 (DeepSeek-V3) to 64. Its report says that at 128K context, 128 heads need 83% more inference FLOPs than 64 heads.
- **The quality result depends on the setup.** In the ablation of DeepSeek-V2, MLA was better than MHA (Appendix D.2). But the GLM-5 report found that with the Muon optimizer, MLA with a 576-dim latent vector was worse than GQA-8. MLA became equal only after a change to how they used the optimizer ("Muon Split").
- **MLA does not work with some components.** The Kimi K2 report says that QK-Norm cannot be used with MLA, because inference never reconstructs K. The main-line model uses QK-Norm (Chapter 9). This is one reason why the main-line model continues to use GQA.
- **The engineering is more complex.** MLA needs special kernels, such as FlashMLA (open source from DeepSeek), and support in the inference engine.

These costs also explain the current adopters. Only flagship MoE models, with tens of billions to trillions of parameters, use MLA now. Almost all small dense models of 0.6–9B use GQA.

## 5. A small experiment: one small model, five kinds of attention

`04_attention_variants.py` uses the character-level Shakespeare model of Chapter 10 (4 layers, width 128, 4 query heads, head_dim 32). It changes only the attention:

- MHA (4 KV heads), GQA (2), MQA (1).
- MLA-48: a 48-dim latent vector + a 16-dim RoPE key = 64 numbers. **The cache has the same size as in MQA.**
- MLA-16: a 16-dim latent vector + 16 = 32 numbers, only half of MQA.

All other parts are the same: the same data order, 600 steps, AdamW + warmup + cosine. Each variant trains with 3 random seeds. The script **measures** the cache size after it generates 512 characters (FP32):

```bash
uv run python chapters/21-kv-cache-ledger/code/04_attention_variants.py
```

> **Note:** The numbers of the training experiments in this chapter come from one CPU run on the course build machine. Different machines and different versions of the math libraries do floating-point operations in a slightly different order. After some hundred training steps, these small differences become larger. Your numbers can be different from the second or third decimal place. Trust the conclusions below, which do not depend on exact values. For a second run on another server in 2026-10, see [runs/2026-10-01-gpu0-check/chapters-21-23.md](../../runs/2026-10-01-gpu0-check/chapters-21-23.md).

| Variant | Cache per layer per position | Per token (4 layers, FP32) | Measured cache after 512 characters | Attention parameters | Mean validation loss | Three seeds | Cached = naive |
|---|---:|---:|---:|---:|---:|---|---|
| MHA | 256 numbers | 4,096 B | 2,150,400 B (1×) | 262,144 | 1.858 | 1.838 / 1.869 / 1.867 | Yes |
| GQA | 128 | 2,048 B | 1,075,200 B (1/2) | 196,608 | 1.856 | 1.867 / 1.849 / 1.852 | Yes |
| MQA | 64 | 1,024 B | 537,600 B (1/4) | 163,840 | 1.860 | 1.860 / 1.858 / 1.862 | Yes |
| MLA-48 | 64 (48 + 16) | 1,024 B | 537,600 B (1/4) | 245,952 | 1.887 | 1.878 / 1.888 / 1.895 | Yes |
| MLA-16 | 32 (16 + 16) | 512 B | 268,800 B (1/8) | 196,672 | 1.886 | 1.889 / 1.854 / 1.916 | Yes |

First, the results that are certain:

- **The cache sizes agree exactly with the ledger.** The measured bytes are exactly "numbers per layer per position × 4 layers × 4 bytes × 525 positions". MLA-48 is as large as MQA, and MLA-16 is only 1/8 of MHA.
- **The cached version gives the same characters as the naive version.** This is true for all five structures, also for MLA on the absorbed path. The first 100 characters generated with the cache are the same as with a full recalculation at each step. The loss of MHA with seed 0, 1.838, is exactly the same as in Chapter 10. Thus the training loop agrees with Chapter 10.

Then the loss. Be very careful here:

- **MHA, GQA, and MQA show no clear difference.** Their means are between 1.856 and 1.860. But MHA alone changes by 0.031 with a different seed (1.838 vs 1.869). The conclusion is the same as in Chapter 10: at this scale, the random variation hides the effect of the number of KV heads.
- **MLA is a little worse here.** All three seeds of MLA-48 (1.878–1.895) are higher than all nine seeds of the first three structures (the highest is 1.869). Its mean is about 0.03 higher than that of MQA, which has the same cache size. The seeds of MLA-16 vary a lot (1.854–1.916), so they give no conclusion.
- **This result does not show that "MLA is worse than MQA".** There are at least three possible reasons. First, the hyperparameters, such as the learning rate, come from MHA in Chapter 10, and we did not tune them for MLA. Second, this model has only 4 heads with head_dim 32. A 48-dim latent vector is not really "low-rank" compared with the head width. Thus the structural advantage of MLA (each head has its own up-projection) has little effect. Third, the training has only 600 steps. DeepSeek-V2 compared them on MoE models of 16B and 250B, and MLA was better than MHA. In the GLM-5 setup, MLA needed a change to the optimizer to become equal to GQA-8. **Whether MLA is better depends on the scale and the training setup.** This tiny experiment shows only three things: the code is correct, the cache is smaller, and at this scale the quality cost is visible but small.

In Part 5 of GOAL.md, "step 2" runs this comparison again with the production code `zero/arch/mla.py`, on a ladder configuration of about 100 million parameters. It will use several seeds and tune the learning rate for each variant. Then we can make a more reliable conclusion.

## 6. Summary

- **Two costs of long context**: In prefill, the attention compute grows as T² (73% for the main-line model at 32K). Decode is limited by memory bandwidth (the arithmetic intensity is only 1–5). At long context, the KV cache is larger than the weights, and a larger batch does not save its reads.
- **Ledger**: Count layer by layer. Full-attention and MLA layers grow linearly with length. Sliding-window layers stop at a cap. Linear-attention layers are constant.
- **Three ways to save cache**: Store fewer heads (GQA/MQA), store less per position (MLA), or store fewer layers or positions (sliding window, hybrid linear attention). You can combine them.
- **MLA**: MLA compresses K and V jointly into a latent vector c_KV (512 dims) + a shared RoPE key (64 dims). This is 576 numbers per layer per position. Decoupled RoPE makes absorption possible. With absorption, MLA acts like MHA in training and like MQA in inference.
- **Costs**: More decode compute, no QK-Norm, and special kernels. Now large MoE models use MLA, and small dense models still use GQA.

---

## GPU measurements (one RTX 3090)

> **Note:** The numbers in the main text above all come from CPU runs. This section uses one NVIDIA GeForce RTX 3090: 24 GB of memory, Ampere architecture. Spec sheet: dense BF16 tensor-core peak about 71 TFLOPS, FP32 about 35.6 TFLOPS, memory bandwidth about 936 GB/s. Environment: PyTorch 2.11.0+cu128, CUDA 12.8, October 2026. The server sets the power limit of this card to 240 W (the factory default is 350 W). Under a continuous full load, the card decreases its clock speed. Thus the absolute compute and bandwidth are lower than on a 3090 at full power, and the relative values are more reliable. If you have no GPU, skip this section.

Run:

```bash
uv run python chapters/21-kv-cache-ledger/code/05_gpu_decode_ledger.py
```

The script builds one decode step (BF16) with random weights in the shape of the main-line model: 28 layers, width 1280, 16 query heads, 8 KV heads, head_dim 128, FFN 3584, vocabulary 65,536. It fills the KV cache with T positions first. Then it measures the time to generate 1 more token. The "lower bound" calls `decode_step` from `02_prefill_decode.py` directly. The formula is the same; only the hardware specification changes to the 71 TFLOPS and 936 GB/s of the 3090. The "effective bandwidth" is the bytes actually read (weights + KV cache) ÷ the measured time. The script records the full step with a CUDA Graph and replays it. Without the graph, one step at T = 1,024 takes 7.51 ms. More than half of that time goes to Python, which launches the kernels one by one. For reference, one contiguous read of 1 GiB on the same card has a measured bandwidth of 873 GB/s.

Main-line model (GQA 16/8), one decode step (median of 30 runs):

| Context T | Batch B | Weights read | KV cache read | Lower bound | Measured | Measured / lower bound | Effective bandwidth | Throughput (token/s) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1,024 | 1 | 1.28 GiB | 0.11 GiB | 1.60 ms | 3.40 ms | 2.13× | 440 GB/s | 294 |
| 4,096 | 1 | 1.28 GiB | 0.44 GiB | 1.98 ms | 4.09 ms | 2.07× | 452 GB/s | 244 |
| 4,096 | 16 | 1.28 GiB | 7.00 GiB | 9.50 ms | 17.10 ms | 1.80× | 520 GB/s | 936 |
| 32,768 | 1 | 1.28 GiB | 3.50 GiB | 5.49 ms | 9.25 ms | 1.69× | 555 GB/s | 108 |
| 32,768 | 4 | 1.28 GiB | 14.00 GiB | 17.53 ms | 27.48 ms | 1.57× | 597 GB/s | 146 |
| 131,072 | 1 | 1.28 GiB | 14.00 GiB | 17.53 ms | 27.23 ms | 1.55× | 603 GB/s | 37 |

With batch 1, from T = 1,024 to 131,072, the KV cache reads 13.89 GiB more, and one step takes 23.83 ms more. This is 626 GB/s.

At the same context (T = 32,768, batch 1), change only the attention. MLA uses 512 + 64 and the absorbed path of `03_mla.py`:

| Variant | Per layer per position | Ledger KV | Actual allocation | Weights | Lower bound | Measured | Effective bandwidth |
|---|---:|---:|---:|---:|---:|---:|---:|
| MHA | 4,096 numbers | 7.00 GiB | 7.00 GiB | 1.42 GiB | 9.66 ms | 15.57 ms | 581 GB/s |
| **GQA (main-line)** | 2,048 | 3.50 GiB | 3.50 GiB | 1.28 GiB | 5.49 ms | 9.33 ms | 550 GB/s |
| MQA | 256 | 0.44 GiB | 0.44 GiB | 1.16 GiB | 1.84 ms | 4.36 ms | 395 GB/s |
| MLA | 576 | 0.98 GiB | 0.98 GiB | 1.36 GiB | 3.70 ms | 7.75 ms | 447 GB/s |

(The weights are not the same size, because the shape of the K/V projections changes with the variant. The lower bound of MLA uses the bytes actually read. The scores read the 576-dim latent vectors once, and the weighted average reads the first 512 dims again. A fused kernel such as FlashMLA reads them only once.)

These two tables change the formula of Section 1.2 into real measurements. With batch 1, the context grows from 1K to 128K. For each additional 1 GiB of KV cache to read, one step takes about 1.7 ms more. "4 conversations of 32K" and "1 conversation of 128K" read the same amount of KV cache (14 GiB each). Their step times are also almost the same (27.48 ms vs 27.23 ms). Decode depends only on how many bytes it reads, not on how many conversations own these bytes.

The batch results also agree with the ledger. At 4K, from batch 1 to 16, the throughput increases about 3.8 times (244 → 936). At 32K, from batch 1 to 4, it increases only about 1.35 times (108 → 146). A larger batch does not save the KV cache reads.

The ledger of Section 3 is exact in GPU memory (actual allocation 7.00 / 3.50 / 0.44 / 0.98 GiB). The decode times also follow the order of the KV cache sizes. But the MLA cache is only 28% of the GQA cache, and one step is only about 17% faster. Without a fused kernel, the step reads the latent vectors twice, and the matrix multiplications in attention are small and fragmented. This is why Section 4.5 says that MLA needs special kernels such as FlashMLA.

The distance from the lower bound was a surprise. A pure read on the same card reaches 873 GB/s, but at short context the effective bandwidth is only 440 GB/s. The weight matrices of each layer are only 5–18 MB. With batch 1, the step is a chain of small matrix-vector products, and none of them is large enough to use the full bandwidth. When the context grows, the large contiguous reads of the KV cache become a larger part of the step. Then the effective bandwidth moves toward 600 GB/s.

## From minimal code to production code

| Minimal code (`code/`) | Production code | What it adds, and why |
|---|---|---|
| `01_kv_ledger.py`: a hand-written dict of models + `layer_list` / `kv_bytes` | `zero/tools/kv_cache_calc.py`: `kv_cache_bytes(cfg, seq_len, batch, dtype_bytes)`, `kv_bytes_per_token`, `fixed_state_bytes`, `breakdown`; command line `uv run python -m zero.tools.kv_cache_calc configs/main/pretrain.toml --seq 32768` | Reads the `ModelConfig` / TOML of zero and the Hugging Face `config.json` directly. This includes the nested `text_config` of multimodal models, `layer_types`, the `linear_attn_config` of Kimi, and the field names of the native Mistral `params.json`. Detects MLA, sliding-window, and linear-attention layers automatically. Estimates the fixed state of linear layers separately. `--dtype-bytes 1` calculates an FP8 KV cache. |
| `MLA` in `03_mla.py`: its interface matches the small model of Chapter 10, and it uses the `torch.cat`-style cache of that model | `zero/arch/mla.py`: `MLAConfig` (the field names match the DeepSeek-V3 config), `MLAAttention` (the same interface as `zero.model.Attention`; it can go directly into `Transformer`, see `mla_transformer`), `MLACache` (preallocates the latent vectors and the RoPE keys; `nbytes()`) | Also supports low-rank compression of the query (`q_lora_rank`, 1536 in DeepSeek-V3; this saves training activations, not cache). Uses the absorbed path automatically with a cache, and the explicit path + SDPA without a cache. Supports chunked prefill. Calculates softmax in float32 or higher. |
| `04_attention_variants.py`: MHA/GQA/MQA/MLA comparison with the same configuration | Optional step 2: run it again with `mla_transformer(model_cfg, mla_cfg)` on a ladder configuration of about 100 million parameters | The production module shares RMSNorm, SwiGLU, and the training loop with the main-line `Transformer`. To change the attention, change only one place. |
| None | Industry implementations: PagedAttention in vLLM (Chapter 10) manages the KV cache in pages; FlashMLA (open source from DeepSeek) is a GPU kernel for MLA decode; vLLM and SGLang both have MLA backends | The MLA of this course aims only to be readable and correct. Its correctness under CUDA + BF16 was verified on an RTX 3090 (see Section 11 of [runs/2026-10-01-gpu0-check](../../runs/2026-10-01-gpu0-check/README.md)). Its performance is not optimized and not yet verified on a GPU (the "GPU measurements" section of this chapter gives some decode times for reference). A real deployment must use these special implementations. |

**Parity check** (`uv run pytest tests/test_kv_cache_calc.py tests/test_arch_mla.py`; on this machine, all 12 tests passed in about 3 seconds):

- For `configs/tiny` and `configs/main`, in FP32 / BF16 with batch 2, the calculator gives exactly the bytes that `KVCache.from_config(...).nbytes()` really allocates. The main-line model uses 114,688 bytes per token.
- For MLA configurations, the result is equal to `MLACache.nbytes()`. DeepSeek-V3 uses 61 × 576 × 2 bytes per token. The sliding window, the Qwen3.5-style hybrid linear attention, and the Mistral `params.json` each have one test case that we calculated by hand.
- In `MLAAttention`, the absorbed path and the explicit path differ by < 1e-10 in float64. Chunked prefill (12 + 1 + 17 tokens) agrees with one full forward pass within 1e-5. Greedy generation of 40 tokens gives exactly the same tokens with and without a cache. A `Transformer` with MLA trains normally.

Part 3 of `01_kv_ledger.py` also calls `zero.tools.kv_cache_calc` directly to check the numbers for the main-line model and DeepSeek-V3.

---

## Frontier notes

> **Sparse attention (DeepSeek DSA) does not save KV cache. It saves compute.** DeepSeek-V3.2 and GLM-5 add DeepSeek Sparse Attention on top of MLA. A light "indexer" gives a score to each past token. Each query attends only to the 2048 tokens with the highest scores (`index_topk: 2048`). This decreases the attention compute for long context from O(T²) to about O(T·k). But the KV cache must still keep all tokens, and the indexer caches an additional small key. When we wrote this chapter, we had verified this method only in two families, DeepSeek and GLM. Chapter 22 checked again. The general direction, "learned sparse attention", now has 4 adopters (DeepSeek, GLM-5, MiniMax-M3, Meituan LongCat). This satisfies GOAL.md 2.1, so Chapter 22 explains it in the main text. But the specific methods (DSA, MSA, LSA, and others) have not converged, so the specific variants are still frontier notes. These methods still save compute, not KV cache.

---

## Adopters and sources

| Technique | Adopters (main versions) | Sources |
|---|---|---|
| GQA | Qwen3 (0.6B: 16 Q / 8 KV; 8B: 32 / 8), Llama 3.1 (32 / 8), gpt-oss (64 / 8), the full-attention layers of Qwen3.5 (0.8B: 8 / 2); Chapter 10 has more | The `config.json` of each model (links below); the Qwen3 and Llama 3 technical reports |
| MLA | **DeepSeek** (introduced in V2; V3, V3.2: `kv_lora_rank` 512, `qk_rope_head_dim` 64), **Kimi** (K2: technical report Section 2.3, "employing MLA"; the full-attention layers of K3), **GLM** (GLM-5: technical report Section 2.1, "Multi-latent Attention"), **Mistral** (Mistral Large 3: `kv_lora_rank` 512 and `qk_rope_head_dim` 64 in `params.json`, the same shape as DeepSeek-V3; the model card does not describe it in words) | DeepSeek-V2 arXiv:2405.04434; DeepSeek-V3 arXiv:2412.19437; Kimi K2 arXiv:2507.20534; GLM-5 arXiv:2602.15763; the configuration of each model |
| Sliding window / alternating local-global layers | gpt-oss (`layer_types` alternate, `sliding_window` 128); Chapter 22 has more | The `config.json` of gpt-oss |
| Hybrid linear attention | Qwen3.5 (`full_attention_interval` 4, the other layers are `linear_attention`), Kimi K3 (`linear_attn_config`: 24 full-attention layers + 69 KDA layers); Chapter 23 has more | The `config.json` of each model |
| Paged KV cache management / MLA kernel | vLLM (PagedAttention, the industry standard); FlashMLA (open source from DeepSeek) | Kwon et al. 2023, arXiv:2309.06180; <https://github.com/deepseek-ai/FlashMLA> |

**Consensus decision (GOAL.md 2.1)**: Four independent leading families use MLA in their main versions: DeepSeek, Kimi, GLM, and Mistral. This satisfies rule A, so MLA goes into the main text. But we must state its scope. All four are MoE flagships with tens of billions to trillions of parameters. **No small dense model uses MLA** (Qwen3, Qwen3.5, Llama 3, Gemma 3, gpt-oss, and SmolLM3 all use GQA or MQA). Thus the main-line model does not use it (GOAL.md 3.3). We explain MQA as the end point of GQA (rule C). Among the leading families, we found only one model that uses MQA: Gemma 3 1B (see Chapter 10).

**Model configurations** (read through Hugging Face in 2026-09):
[Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B/blob/main/config.json),
[Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B/blob/main/config.json),
[Llama-3.1-8B (unsloth mirror)](https://huggingface.co/unsloth/Meta-Llama-3.1-8B/blob/main/config.json),
[gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b/blob/main/config.json),
[Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B/blob/main/config.json),
[Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B/blob/main/config.json),
[Qwen3.5-397B-A17B (read from the FP8 version)](https://huggingface.co/Qwen/Qwen3.5-397B-A17B-FP8/blob/main/config.json),
[DeepSeek-V3](https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/config.json),
[DeepSeek-V3.2](https://huggingface.co/deepseek-ai/DeepSeek-V3.2/blob/main/config.json),
[Kimi-K2-Instruct](https://huggingface.co/moonshotai/Kimi-K2-Instruct/blob/main/config.json),
[GLM-5](https://huggingface.co/zai-org/GLM-5/blob/main/config.json),
[params.json of Mistral-Large-3-675B-Instruct-2512](https://huggingface.co/mistralai/Mistral-Large-3-675B-Instruct-2512/blob/main/params.json),
[Kimi-K3](https://huggingface.co/moonshotai/Kimi-K3/blob/main/config.json).

Notes: The official `meta-llama/Llama-3.1-8B` of Meta needs an access request, and this chapter could not read it. The numbers come from the unsloth mirror (its `_name_or_path` points to the official repository), the same as in Chapter 10. The layer numbers in `full_attn_layers` of Kimi K3 start from 1 (1–93). This chapter reads them as 24 MLA layers and 69 KDA layers. The 989.5 TFLOPS of the H100 SXM uses the same basis as `zero/tools/estimate_cost.py`. The 3.35 TB/s and the 80 GB come from the NVIDIA H100 product page <https://www.nvidia.com/en-us/data-center/h100/>.

---

## Guided questions

Ask Claude Code these questions. Continue until you can explain the answers in your own words:

1. Section 1.2 says: "For long context, a larger batch does not save the KV cache reads." Then why does a larger batch still save the weight reads? Try to estimate: for the main-line model at 32K context, at which batch size is the time to read the KV cache 10 times the time to read the weights?
2. The MLA latent vector has 512 dims, and the hidden size of DeepSeek-V3 is 7168. If the latent vector grows to 7168 dims, does it still save cache? Is it still useful? What does the "low-rank" limit do here?
3. Why do all heads share one k^R (the part of the key with RoPE), instead of one k^R for each head? If each head had its own k^R, how large would the cache be?
4. After absorption, MLA at decode "has the form of MQA". How is it different from a real MQA (Chapter 10, 1 KV head, head_dim 32)? Why can the quality of MLA be much better than the quality of MQA?
5. Qwen3.5 and Kimi K3 both use "a few full-attention layers + many linear-attention layers". If the full-attention layers also change to MLA (as in Kimi K3), how does the ledger change? If Qwen3.5-9B made this change, about how large would its KV cache be at 128K?
6. In the small experiment of this chapter, how do the loss differences between variants compare with the differences between seeds? To compare MLA and GQA seriously, how would you design the experiment? Use the method of Table 1 in the GLM-5 report as a reference.

## Hands-on tasks

For each task, run the code and look at the result.

**Task 1 (basic)**: On Hugging Face, select a model that this chapter did not calculate. Examples are SmolLM3-3B, the Gemma 3 series, or other versions of MiniMax or GLM. Read its `config.json`. Add its fields to `MODELS` in `01_kv_ledger.py`, and calculate the KV cache at 32K and 128K. Then do a parity check with `uv run python -m zero.tools.kv_cache_calc your_saved_config.json --seq 131072`. Check if the model has a sliding window or `layer_types`.

**Task 2 (core)**: In `03_mla.py`, add a timing experiment for `MLA`. Initialize an MLA randomly with `n_heads=16, kv_lora_rank=64, rope_dim=16`. First, prefill 1024 positions. Then decode 64 steps with the absorbed path, and 64 steps with the explicit path. Compare the time per step. Then change the cache length to 256 and 4096, and look at the trend. Explain why the advantage of the absorbed path grows with the cache length.

**Task 3 (challenge)**: In `04_attention_variants.py`, add a variant such as "GQA-2 × head_dim 16". Its cache per layer per position must also be 64 numbers (2 × 2 × 16, the same as MQA and MLA-48). Then run 3 seeds. With the same cache size, which structure has the lowest loss? Is the difference larger than the variation between seeds? Then double the number of training steps, and check if the conclusion changes.

---

## Go deeper: CS336

This chapter matches Stanford CS336 (Spring 2026) <https://cs336.stanford.edu/>:

- **Lecture 10: Inference.** It discusses the cost of inference from the point of view of resource accounting: prefill and decode, the memory and memory bandwidth of the KV cache, and many methods that make inference faster and cheaper. The slides and videos are on the course page. This build environment cannot open the course page, so check the slides for the exact list of architecture methods. You can think of `02_prefill_decode.py` in this chapter as one concrete calculation of this resource accounting for the main-line model.

---

## References

- DeepSeek-AI. *DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model* (introduces MLA: low-rank joint compression, decoupled RoPE, absorption, and the ablations in Appendix D), 2024: <https://arxiv.org/abs/2405.04434>
- DeepSeek-AI. *DeepSeek-V3 Technical Report*, 2024: <https://arxiv.org/abs/2412.19437>
- Kimi Team. *Kimi K2: Open Agentic Intelligence* (Section 2.3: why they keep MLA and decrease the heads from 128 to 64; QK-Norm does not apply to MLA), 2025: <https://arxiv.org/abs/2507.20534>
- GLM-5 Team. *GLM-5: from Vibe Coding to Agentic Engineering* (Section 2.1: MLA compared with GQA-8, Muon Split, MLA-256; DSA), 2026: <https://arxiv.org/abs/2602.15763>
- Mistral AI. *Mistral 3* (the release blog of Mistral Large 3): <https://mistral.ai/news/mistral-3>
- Shazeer. *Fast Transformer Decoding: One Write-Head is All You Need* (MQA), 2019: <https://arxiv.org/abs/1911.02150>
- Ainslie et al. *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints*, 2023: <https://arxiv.org/abs/2305.13245>
- Kwon et al. *Efficient Memory Management for Large Language Model Serving with PagedAttention* (vLLM), 2023: <https://arxiv.org/abs/2309.06180>
- DeepSeek. FlashMLA (MLA decode kernel): <https://github.com/deepseek-ai/FlashMLA>
- [CS336](https://cs336.stanford.edu/) Lecture 10 (Inference)
- [The main LLM architectures in 15,000 characters (Llama, Qwen, GLM, DeepSeek…)](https://zhuanlan.zhihu.com/p/2060741715095560795) (in Chinese): a side-by-side comparison of the attention structures of each family (already in `references.md`)
- [Pretraining a Mini Kimi K3](https://books.vizuara.ai/book/pretraining-a-mini-k3): hands-on pretraining of the "MLA + KDA hybrid" architecture of Kimi K3 (already in `references.md`)
- The links to the model configurations are in "Adopters and sources" above.

**Next chapter**: The ledger has two paths that we did not finish. Some layers can look only at a recent part of the sequence (sliding window). Other layers can store no K and V at all (linear attention). Chapter 22 first explains local and sparse attention. Why do half of the layers of gpt-oss look at only 128 tokens? Why are most layers of Gemma local? How can a model "look only nearby" and still keep the information that is far away?
