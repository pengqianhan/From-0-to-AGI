# Chapter 10: Inference — Make the model generate text, and make it fast

**English** · [中文](README.zh.md)

> **Goal**: After this chapter, you can write a sampling function with temperature, top-k, and top-p. You can add a KV cache to a Transformer and make sure that it generates exactly the same text as the model without a cache. You can also use `2 × layers × KV heads × head_dim × sequence length × bytes` to calculate the KV cache size of any model and how much GQA saves.

📺 **Video**: Not published yet. To render it on your computer, run `bash chapters/10-inference/video/build.sh`.
🧪 **Self-check**: After the chapter, type `/ch10-inference` in Claude Code.

---

In the last chapter, we built a complete modern Transformer: RMSNorm, RoPE, SwiGLU, and causal attention. After a few minutes of training, it wrote text in the style of Shakespeare. But until now, we looked only at **training**: we give the model a full text and calculate the loss at all positions at the same time. This chapter answers a new question: **how does a trained model actually produce text?**

The question has two parts. The first part is "what to say". At each step, the model gives a probability distribution. The method that picks a character from this distribution decides if the text is dull or varied, and if it is fluent or nonsense.

The second part is "how fast". The most basic generation method does a large amount of repeated calculation. The KV cache removes this calculation, but it causes a new problem: GPU memory. The KV cache is too large, and GQA is the answer to that problem.

All code in this chapter uses its own small model (`code/01_tiny_model.py`). It has the same structure as Chapter 9 (Pre-Norm RMSNorm, RoPE, SwiGLU, tied embeddings). We want to train it on a CPU in 1 to 2 minutes. Thus it works on characters (a vocabulary of 65 characters), and it has no QK-Norm. It also adds two things from this chapter: a KV cache and `n_kv_heads`.

```bash
uv run python chapters/10-inference/code/01_tiny_model.py   # train and save the weights (about 1.5 minutes on 1 thread, only once)
```

| Setting | Value |
|---|---|
| Layers / width / query heads / head_dim | 4 / 128 / 4 / 32 |
| Parameters | 861,440 (0.86M) |
| Training | 600 steps, batch 16 × 64 characters, AdamW + warmup + cosine |
| Validation loss | 1.838 nats/character (a uniform random guess gives ln 65 = 4.174) |

> **Note:** The numbers from the training experiments in this chapter come from one CPU run on the course build machine. Different machines and different versions of the low-level math libraries do floating-point operations in a slightly different order. After some hundred training steps, these small differences become larger. Your numbers can differ from the second or third decimal place. Trust the conclusions below that do not depend on exact values. For a re-run on a different server in 2026-10, see [runs/2026-10-01-gpu0-check/chapters-07-10.md](../../runs/2026-10-01-gpu0-check/chapters-07-10.md).

## 1. Generation is a loop

A language model can do only one thing. It looks at all the content before the current position and gives a score (a logit) to each candidate for the "next token". To generate a text, we do this in a loop:

1. Give the current sequence to the model, and take the logits at the **last position**.
2. Use a strategy to pick one token from these logits.
3. Append the token to the end of the sequence. Then go back to step 1.

This is **autoregressive generation**: the output of each step is the input of the next step. The end of `01_tiny_model.py` shows the most basic form:

```python
ids = data.encode("ROMEO:\n")
for _ in range(120):
    logits = model(torch.tensor([ids]))[0, -1]   # give the full sequence, keep only the last position
    ids.append(int(logits.argmax()))             # pick the largest one and append it
```

These four short lines contain two questions. Is it good to "pick the largest one" (step 2)? Is it fast to "give the full sequence" (step 1)? We look at the two questions one at a time.

## 2. Which character to pick: greedy, temperature, top-k, top-p

### 2.1 Greedy: safe, but it goes around in a loop

**Greedy decoding** picks the token with the highest probability at each step. It seems to be the safest method. But run `02_sampling.py` and look at what it writes:

```
[greedy (T=0)]  distinct 4-gram ratio 0.27
  the son the such the shall the shall the proves
  The shall the shall the such the shall the shalleend the shand
```

"the shall the shall": the text goes around in a loop. We use "the fraction of distinct 4-character pieces" as a rough measure of repetition. In 200 characters, greedy decoding gets only 0.27. Each step makes the best local choice, but a chain of best local choices is not the best text. When the text enters a loop, the most probable next character takes it back into the loop.

Large models show the same degeneration. Holtzman et al. studied it in a 2019 paper with the title *The Curious Case of Neural Text Degeneration*.

The solution is **sampling**: pick a character at random, with the probabilities that the model gives. A character with a high probability has a higher chance, but the sampler does not pick it every time.

### 2.2 Temperature: change the distribution before sampling

Chapter 5 showed softmax with temperature: divide the logits by the temperature T, and then apply softmax.

```
p_i = softmax(z / T)_i
```

This is `filtered_probs` in `02_sampling.py`:

```python
probs = torch.softmax(logits / temperature, dim=-1)  # softmax with temperature (Chapter 5)
```

Below is the **real** distribution of the next character from the small model, after the prompt `"ROMEO:\nI will "` (top 8; `␣` is a space):

| Temperature | t | a | n | s | h | m | b | w | Entropy (nats) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| T = 0.5 | 0.427 | 0.093 | 0.073 | 0.066 | 0.062 | 0.060 | 0.047 | 0.037 | 2.15 |
| T = 1.0 | 0.173 | 0.081 | 0.071 | 0.068 | 0.066 | 0.065 | 0.057 | 0.051 | 3.00 |
| T = 1.5 | 0.106 | 0.064 | 0.059 | 0.057 | 0.056 | 0.055 | 0.051 | 0.047 | 3.39 |

T < 1 makes the differences larger: the probability of t increases from 0.17 to 0.43. The distribution becomes sharper, and the generation becomes more conservative. T > 1 makes the differences smaller: t decreases to 0.11, and rare characters also get a chance. The generation becomes bolder, and it also produces nonsense more often. T → 0 is greedy decoding. Thus the code uses `argmax` directly when `temperature == 0`.

### 2.3 Cut the long tail: top-k and top-p

Even with a good temperature, the tail of the distribution contains tens of very unlikely tokens. Each of them has a very small probability, but their sum is not small. In a generation of some hundred tokens, the sampler will pick one of them sooner or later. When it does, the text after that token goes in a wrong direction. Thus we often cut the long tail before sampling, and then normalize again:

- **top-k** (Fan et al. 2018): keep only the k most probable tokens.
- **top-p**, also called **nucleus sampling** (Holtzman et al. 2019): sort the probabilities from the largest to the smallest. Add them until the sum is at least p. Sample only from this small "nucleus".

```python
if top_k > 0:  # keep only the k most probable tokens
    kth = torch.topk(probs, top_k).values[-1]
    probs = torch.where(probs >= kth, probs, 0.0)
if top_p < 1.0:  # nucleus: add from the largest down; stop when the sum reaches top_p
    sorted_p, idx = torch.sort(probs, descending=True)
    before = torch.cumsum(sorted_p, 0) - sorted_p  # cumulative probability before this token
    keep = torch.zeros_like(probs, dtype=torch.bool)
    keep[idx] = before < top_p  # this always keeps the largest one
    probs = torch.where(keep, probs, 0.0)
return probs / probs.sum()
```

Why is top-p better than top-k? Look at two contexts:

| Context | Most probable next character | Kept by top-p = 0.9 | Kept by top-k = 5 |
|---|---|---:|---:|
| `"KING RICHARD III:\nWhat is th"` (very certain) | `e`, p = 0.546 | 5 | 5 |
| `"ROMEO:\n"` (a line of dialogue starts, very uncertain) | `A`, p = 0.123 | 15 | 5 |

When the model is confident, the nucleus is small. When the model is not confident, the nucleus becomes larger automatically. top-k always keeps k tokens, whether the model is confident or not. When the model is not confident, top-k removes reasonable candidates. When the model is confident, top-k can keep absurd candidates. This is the reason why Holtzman et al. proposed top-p. In practice, people also often use the two methods together.

### 2.4 Real results, and the defaults of open models

We use the same prompt and the same random seed, and generate 200 characters with each strategy (output of `02_sampling.py`, start only):

| Strategy | Distinct 4-gram ratio | Start of the generated text |
|---|---:|---|
| Greedy (T = 0) | 0.27 | `the son the such the shall the shall the proves` |
| T = 0.5 | 0.84 | `there his the prester of thee hand,` |
| T = 1.0 | 0.95 | `thy forful haves: / WeWadHis nears! younk gelst to you` |
| T = 1.0, top-k = 5 | 0.91 | `thy comman haves their soul that` |
| T = 1.0, top-p = 0.9 | 0.94 | `thou more the prevenced can that` |
| T = 1.5 | 0.98 | `thyremhis kiss selted? / His nears! youd Igelst, gMycouuesy` |

A character-level model with 860 thousand parameters cannot write really fluent English. But the trend is clear. Greedy decoding repeats itself. T = 1.5 produces garbage (`gMycouuesy`). After we cut the long tail (top-k, top-p), there are many fewer strange words.

The distinct ratio measures only "no repetition". It does not measure "fluency". Thus the 0.98 of T = 1.5 does not mean that T = 1.5 is the best. For this reason, people must choose the decoding settings by reading the output and by evaluation.

How do real models choose? Open models publish a `generation_config.json` file together with the weights. This file contains the official recommended decoding settings (read in 2026-09):

| Model | Temperature | top-p | top-k |
|---|---:|---:|---:|
| Qwen3-8B | 0.6 | 0.95 | 20 |
| Qwen2.5-7B-Instruct | 0.7 | 0.8 | 20 |
| Llama-3.1-8B-Instruct | 0.6 | 0.9 | — |
| SmolLM3-3B | 0.6 | 0.95 | — |
| Gemma-3-27B-it | — (default 1.0) | 0.95 | 64 |

"A temperature a little below 1 + top-p 0.8–0.95" is almost the standard setting.

## 3. Why naive generation is slow: a triangle of repeated calculation

Go back to the four lines of code in Section 1, and look at `model(torch.tensor([ids]))`. For each new character, the code gives the **full** sequence to the model again. Let the prompt length be P. To generate character t, the model processes P + t positions. To generate n characters, it processes this number of positions in total:

```
Σ (P + t) ≈ P·n + n²/2   positions
```

In a plot, this is a triangle: step 1 processes P positions, step 2 processes P + 1 positions, and so on. The prompt of this chapter, `"ROMEO:\nI will "`, has 14 characters. To generate 512 characters, the model processes 137,984 positions.

But think about it. In causal attention, position i looks only at the positions ≤ i. When a new character arrives, **all intermediate results of the past positions stay the same**. In each row of the triangle, everything except the last new position was already calculated in the previous step.

## 4. KV cache: keep the K and V that you already calculated

Which intermediate results must we keep? Look at attention (Chapter 8). The new position t must calculate:

```
out_t = softmax(q_t · [k_1, …, k_t]ᵀ / √d) · [v_1, …, v_t]
```

It needs its own query `q_t`, and the keys and values of **all** positions. The `k_i` and `v_i` of past positions do not change, so we keep one copy of them in each layer. This is the **KV cache**.

After the first step, each step gives the model only one new token. The model calculates the q, k, v of this token and appends k and v to the end of the cache. Then it does attention between q and all K, V in the cache. **Q needs no cache**: the model uses `q_t` only once, in step t.

The minimal implementation changes only two places in attention (`01_tiny_model.py`):

```python
q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)  # the cache keeps K after the rotation
if cache is not None:
    k, v = cache.append(layer, k, v)  # old K/V + new K/V (torch.cat)
S = k.shape[2]  # total visible length = past + now
...
# Causal mask: the global position of new token i is S-T+i.
# It can see only the keys at positions ≤ its own position.
i = torch.arange(T)[:, None] + (S - T)
j = torch.arange(S)[None, :]
att = att.masked_fill(j > i, float("-inf")).softmax(-1)
```

There is one more detail. RoPE must know the **absolute position** of the new token. Thus `TinyLM.forward` uses `start = len(cache)` to take the cos/sin values of the correct positions. The generation loop becomes (`03_kv_cache.py`):

```python
cache = tiny.KVCache(model.c.n_layers)
logits = model(torch.tensor([prompt]), cache)[0, -1]      # prefill: give the full prompt at once
for i in range(n):
    nxt = samp.sample_next(logits, g=g, **kw)
    logits = model(torch.tensor([[nxt]]), cache)[0, -1]  # decode: give only 1 new character
```

To generate 512 characters, the number of processed positions decreases from 137,984 to 525 (14 + 511).

### 4.1 Parity check: the result must be exactly the same

The cache only removes repeated calculation. It must not change any result. Run:

```bash
uv run python chapters/10-inference/code/03_kv_cache.py
```

```
1. Parity check: do the cached and the naive versions generate the same 200 characters?
  greedy                 same: True   start: 'the son the such the shall the shall the'
  sample T=1.0 top-p=0.9 same: True   start: 'thou more the prevenced can that\nhave wi'
  max logits difference at the same position: 4.3e-06 (the size of floating-point rounding)
  cache size 872448 bytes = 2 × 4 layers × 4 KV heads × 32 × 213 positions × 4 bytes = 872448
```

The sampling mode also gives the same text. Both versions use a random number generator with the same seed, and at each step they see the same distribution (within the floating-point error). The logits differ by 4.3 × 10⁻⁶ for this reason: the matrix multiplications in the two methods have different shapes, so the additions occur in a different order.

### 4.2 Speed

The output is the same. How much faster is it? (1 CPU thread, the faster of 2 runs for each item, greedy)

| New characters | Naive (s) | KV cache (s) | Speedup | Positions processed (naive) | Positions processed (cache) |
|---:|---:|---:|---:|---:|---:|
| 64 | 0.33 | 0.16 | 2.1× | 2,912 | 77 |
| 128 | 0.78 | 0.30 | 2.6× | 9,920 | 141 |
| 256 | 2.55 | 0.58 | 4.4× | 36,224 | 269 |
| 512 | 10.47 | 1.34 | 7.8× | 137,984 | 525 |

The longer the sequence, the larger the speedup. The cost of the naive version increases as n², and the cost of the cached version increases as n. The speedup is much smaller than the ratio of the processed positions (137,984 / 525 ≈ 263). There are two reasons. On a small model, each step has a fixed overhead (the Python loop and the calls of small matrix multiplications). Also, at each step, the cached version must still do attention between the new q and all past K.

The timing depends on the load of the machine: another run, which the video uses, gave 1.9× / 2.7× / 4.4× / 9.8×.

### 4.3 Two phases: prefill and decode

With a cache, inference has two phases:

- **prefill**: give the full prompt to the model at once. The model calculates all positions in parallel and writes their K and V into the cache.
- **decode**: after that, calculate one new token at a time.

We process the same 256 positions in both phases (part 3 of `03_kv_cache.py`):

```
prefill, 256 at a time:   16.9 ms  →    15168 positions/s
decode,  1 at a time:    691.5 ms  →      370 positions/s
prefill throughput is 41× the decode throughput
```

(Another run, which the video uses, gave 19.0 ms vs 585 ms, 31×. The exact ratio depends on the machine, but it is always some tens of times.) Each decode step calculates only one token. But it must read all weights and the full KV cache. The hardware spends most of its time on moving data, and the compute units are not fully used.

For this reason, inference services **put the requests of many users into one batch and decode them together**: one read of the weights serves tens of requests. This also means that the KV cache increases in proportion to the number of users.

## 5. The memory ledger of the KV cache

The cache saves calculation, but it uses GPU memory. Each layer must keep one K and one V for each position. Each of them has `KV heads × head_dim` numbers:

```
KV cache bytes = 2 × layers × KV heads × head_dim × sequence length × bytes per number (× batch)
```

This is the function in `04_kv_memory.py`:

```python
def kv_bytes(layers, kv_heads, head_dim, seq_len, bytes_per=BF16, batch=1):
    return 2 * layers * kv_heads * head_dim * seq_len * bytes_per * batch
```

Use the values of the main-line model (`configs/main/pretrain.toml`: 28 layers, 16 query heads, 8 KV heads, head_dim 128, BF16):

```bash
uv run python chapters/10-inference/code/04_kv_memory.py
```

| Scheme | KV heads | Per token | 4K context | 32K context |
|---|---:|---:|---:|---:|
| MHA (no sharing) | 16 | 224 KiB | 896 MiB | 7.00 GiB |
| **GQA (main line)** | **8** | **112 KiB** | **448 MiB** | **3.50 GiB** |
| MQA (all heads share) | 1 | 14 KiB | 56 MiB | 0.44 GiB |

Each token needs 2 × 28 × 8 × 128 × 2 = 114,688 bytes = 112 KiB. For comparison, the main-line model has 689.5M parameters, and its BF16 weights use 1.28 GiB. **For one conversation with a 32K context, the KV cache is 2.7 times as large as the weights.** To serve 16 such conversations at the same time, the cache needs 56 GiB (112 GiB with MHA). The bottleneck of inference changes from "the calculation is too slow" to "the memory is too small".

## 6. GQA: several query heads share one set of K and V

The number of layers and head_dim in the formula have a direct effect on the capability of the model. Thus we do not want to change them. The user decides the sequence length, so we cannot change it either. The **number of KV heads** is the only term that is left.

In standard **multi-head attention (MHA)**, each query head has its own K and V. In 2019, Shazeer proposed an extreme method: all query heads **share one set** of K and V. This is **multi-query attention (MQA)**. The KV cache becomes smaller by a factor equal to the number of heads, but the quality decreases.

In 2023, Ainslie et al. proposed a compromise. Divide the query heads into groups, and let each group share one set of K and V. This is **grouped-query attention (GQA)**. The experiments in the paper found that GQA has a quality near MHA and a speed near MQA. MHA and MQA are the two end points of GQA (number of groups = number of heads, and number of groups = 1).

The change in the implementation is small. The K and V projections become narrower (their output is `n_kv_heads × head_dim`). Before attention, the code copies each set of K and V to the query heads of its group:

```python
self.wk = nn.Linear(c.dim, c.n_kv_heads * c.head_dim, bias=False)  # GQA: narrower K/V projections
...
g = c.n_heads // c.n_kv_heads
k, v = k.repeat_interleave(g, dim=1), v.repeat_interleave(g, dim=1)  # g query heads share one set
```

The important point is that the copy occurs **after the cache**. The cache keeps `n_kv_heads` copies, not `n_heads` copies.

### 6.1 A small experiment: the same 600 training steps

`05_gqa.py` uses the same 4 query heads and changes only the number of KV heads. The seed, the data order, and the number of training steps are the same:

```bash
uv run python chapters/10-inference/code/05_gqa.py   # the first run trains 3 models, about 5 minutes on 1 thread
```

| Scheme | KV heads | Validation loss | Attention parameters | Total parameters | KV cache (after 512 generated characters, FP32) |
|---|---:|---:|---:|---:|---:|
| MHA | 4 | 1.838 | 262,144 | 861,440 | 2,150,400 B (1.00×) |
| GQA | 2 | 1.867 | 196,608 | 795,904 | 1,075,200 B (0.50×) |
| MQA | 1 | 1.860 | 163,840 | 763,136 | 537,600 B (0.25×) |
| Control: MHA with random seed 1 | 4 | 1.869 | | | |

(Validation loss from a re-run on a different server in 2026-10: MHA 1.831, GQA 1.865, MQA 1.860, and MHA with the other seed 1.861.)

The KV cache decreases exactly in proportion to the number of KV heads. The number of parameters also decreases a little, because the K and V projections are narrower.

The script also prints the time to generate 512 characters. But on this small model and 1 CPU thread, the time shows mostly the load of the machine. Two runs gave 1.79 / 1.50 / 1.21 seconds and 3.62 / 4.09 / 4.89 seconds: even the order is reversed. Thus we cannot see a speed difference from GQA here. The speed benefit of GQA comes from the smaller KV cache that decode must read. It becomes visible only on a GPU where memory bandwidth is the bottleneck, with long contexts and large batches (Chapter 21).

What about the loss? MHA seems to be the best. But a different random seed alone changes the loss of the same MHA by 0.031. This is the same order of magnitude as the differences between the three variants (at most 0.029).

In the re-run on a different server, the seed difference was 0.030, and the largest difference between the three variants was 0.034. Again, these numbers have the same order of magnitude. **At this scale, the quality cost of GQA/MQA is about as large as the variation from the random seed, so we cannot separate the two.** This table does not tell us which variant is better.

To see the quality difference, we need several seeds, a larger model, and longer training. The GQA paper did this. Chapter 21 does a more careful comparison of MHA / GQA / MLA on a CPU.

### 6.2 What public models choose

This is part 2 of `04_kv_memory.py`. The numbers come from the `config.json` of each model in its Hugging Face repository (32K context, BF16, batch 1; the formula counts all layers as full attention):

| Model | Layers | Q heads | KV heads | head_dim | Per token | 32K context | Without sharing (MHA) | Saving |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-0.6B | 28 | 16 | 8 | 128 | 112 KiB | 3.50 GiB | 7.00 GiB | 2× |
| SmolLM3-3B | 36 | 16 | 4 | 128 | 72 KiB | 2.25 GiB | 9.00 GiB | 4× |
| Qwen2.5-7B | 28 | 28 | 4 | 128 | 56 KiB | 1.75 GiB | 12.25 GiB | 7× |
| Llama-3.1-8B | 32 | 32 | 8 | 128 | 128 KiB | 4.00 GiB | 16.00 GiB | 4× |
| Qwen3-8B | 36 | 32 | 8 | 128 | 144 KiB | 4.50 GiB | 18.00 GiB | 4× |
| Mistral-7B-v0.1 | 32 | 32 | 8 | 128 | 128 KiB | 4.00 GiB | 16.00 GiB | 4× |
| gpt-oss-20b | 24 | 64 | 8 | 64 | 48 KiB | 1.50 GiB | 12.00 GiB | 8× |
| Gemma-3-27B | 62 | 32 | 16 | 128 | 496 KiB | 15.50 GiB | 31.00 GiB | 2× |
| Llama-3.3-70B | 80 | 64 | 8 | 128 | 320 KiB | 10.00 GiB | 80.00 GiB | 8× |

The KV configuration of the main-line model is the same as that of Qwen3-0.6B (28 layers, 8 KV heads, head_dim 128). Thus the cache per token is also the same.

Some notes follow. Mistral 7B uses a sliding window of 4096. In gpt-oss, half of the layers use sliding attention with a window of 128. In Gemma 3, most layers use local attention with a window of 1024. For these models, the real cache is smaller than the table shows. This is the topic of Chapter 22. The 1B version of Gemma 3 goes further: it uses MQA (1 KV head).

## 7. Summary

- **Autoregressive generation**: calculate the distribution → pick one token → append it to the end → do it again.
- **Decoding strategies**: greedy decoding goes around in a loop. Temperature controls how sharp the distribution is. top-k always keeps k tokens. top-p keeps tokens by cumulative probability, so the number changes automatically with the confidence of the model. Most open models use a default temperature of 0.6–0.7 + top-p 0.8–0.95.
- **KV cache**: the K and V of past positions do not change, so keep them. Each step calculates only the new token. The output stays exactly the same, and the cost decreases from O(n²) to O(n).
- **prefill / decode**: prefill works in parallel and uses the hardware fully. Decode does one token at a time and does not use the hardware fully, so a service must use batches.
- **Memory ledger**: `2 × layers × KV heads × head_dim × sequence length × bytes`. The main-line model needs 112 KiB per token and 3.5 GiB for a 32K context. This is 2.7 times the size of the weights.
- **GQA**: several query heads share one set of K and V, and the cache becomes smaller in proportion. MHA and MQA are its two end points.

---

## GPU measurements (one RTX 3090)

> All numbers in the main text above come from CPU runs. In this section, we measure on one NVIDIA GeForce RTX 3090 (24 GB of GPU memory, Ampere architecture). Its spec sheet gives a dense BF16 tensor-core peak of about 71 TFLOPS, FP32 of about 35.6 TFLOPS, and a memory bandwidth of about 936 GB/s. Environment: PyTorch 2.11.0+cu128, CUDA 12.8, October 2026.
>
> The server sets the power limit of this card to 240 W (the factory default is 350 W). Under a continuous full load, the card lowers its clock. Thus the absolute compute and bandwidth are lower than on a 3090 at full power, and the relative values are more reliable. If you do not have a GPU, skip this section.

Run:

```bash
uv run python chapters/10-inference/code/06_gpu_prefill_decode.py
```

The model is still `TinyLM` from `01_tiny_model.py` (the script loads the code without changes). Only the size changes to the shape of one layer of Llama 3 8B: d = 4096, 32 query heads, 8 KV heads, FFN 14336. The model has 4 such layers: 0.87B parameters in total, 1.63 GiB of BF16 weights, with random initialization (we measure only the speed). At the spec-sheet bandwidth, one read of these weights from GPU memory takes at least 1.86 ms.

There are two kinds of timing. "eager" is the usual call, one operator at a time. "CUDA graph" first records all kernels of one forward pass, and then replays the full graph at once, without Python. It measures only the time of the GPU work.

First, we use the two generation functions of `03_kv_cache.py` with greedy generation, and look at the mean time per token (eager, median of 3 runs):

| New tokens | Naive (ms/token) | KV cache (ms/token) | Speedup |
|---:|---:|---:|---:|
| 64 | 6.44 | 5.77 | 1.1× |
| 256 | 9.21 | 5.14 | 1.8× |
| 512 | 13.89 | 5.21 | 2.7× |

Next, we measure the time of one forward pass with T tokens (this is a prefill of T tokens). The last three columns use the CUDA graph time. The values in parentheses are the fraction of the spec-sheet peak:

| T | eager (ms) | CUDA graph (ms) | token/s | Compute TFLOPS | Weight read GB/s |
|---:|---:|---:|---:|---:|---:|
| 1 | 4.70 | 2.65 | 378 | 0.7 (1%) | 659 (70%) |
| 8 | 5.17 | 2.69 | 2,978 | 5.2 (7%) | 650 (69%) |
| 32 | 4.69 | 3.19 | 10,019 | 17.5 (25%) | 547 (58%) |
| 128 | 6.78 | 5.95 | 21,507 | 37.6 (53%) | 293 (31%) |
| 512 | 20.30 | 20.39 | 25,106 | 44.2 (62%) | 86 (9%) |
| 2048 | 88.13 | 87.99 | 23,275 | 42.2 (59%) | 20 (2%) |

Last is decode. Each sequence gets only 1 token per step, and the batch increases from 1 to 64 (CUDA graph; "memory read" counts one read of all weights + one read of the full KV cache):

| Context | batch | Per step (ms) | token/s | KV cache | Memory read GB/s |
|---:|---:|---:|---:|---:|---:|
| 64 | 1 | 2.77 | 360 | 1 MiB | 630 (67%) |
| 64 | 8 | 3.13 | 2,553 | 8 MiB | 560 (60%) |
| 64 | 32 | 4.36 | 7,333 | 32 MiB | 408 (44%) |
| 64 | 64 | 6.21 | 10,311 | 64 MiB | 292 (31%) |
| 512 | 1 | 2.91 | 344 | 8 MiB | 603 (64%) |
| 512 | 8 | 4.58 | 1,747 | 64 MiB | 396 (42%) |
| 512 | 32 | 10.04 | 3,188 | 256 MiB | 201 (21%) |
| 512 | 64 | 17.12 | 3,737 | 512 MiB | 133 (14%) |

The second table shows on real hardware what Section 4.3 said: "decode does not use the compute fully". When T increases from 1 to 32, one forward pass takes almost the same time (2.65 → 3.19 ms). During this time, the GPU mainly moves the 1.63 GiB of weights out of GPU memory (550–660 GB/s). It uses only 1%–25% of the peak compute. This is the situation of decode.

Only when T passes about one hundred does the time increase linearly with T. Then token/s stays at 23,000–25,000, and the compute reaches about 60% of the peak. This is the situation of prefill.

The turning point is between 32 and 128. This agrees with the value that the script prints: "peak compute ÷ bandwidth ≈ 76 FLOP/byte". When we give T tokens at once, each parameter of 2 bytes does 2T floating-point operations. Thus the arithmetic intensity (floating-point operations per byte read) is about T.

The third table shows the other side of the same effect. At context 64, the batch increases from 1 to 64, but each step increases only from 2.77 ms to 6.21 ms. The throughput increases by about 29 times: one read of the weights serves 64 sequences.

But at context 512 and batch 64, the KV cache is 512 MiB, and each step must read all of it. (The `torch.cat` and `repeat_interleave` of the minimal code also copy it again.) The throughput is only 3,737. The cache starts to compete with the weights for bandwidth. This is the memory ledger of Section 5, and it is also the topic of Chapter 21.

In the first table, the KV cache is only 1.1–2.7 times faster, much less than on 1 CPU thread in Section 4.2. The second table also gives the reason: on a GPU, to calculate some tens of tokens again costs almost the same as to calculate 1 token. The waste of the naive version becomes visible only when the sequence is long enough.

The difference between eager and CUDA graph was a surprise. At T = 1, eager needs 4.70 ms, and the replay needs only 2.65 ms. Almost half of the time goes to Python, which launches the kernels one at a time. (Each layer of the minimal model has tens of small operators: RMSNorm, RoPE, cache concatenation, K/V copies, and more.) Inference engines such as vLLM record CUDA graphs in the decode phase to remove this cost.

## From minimal code to production code

The table shows where the three topics of this chapter are in the main-line model:

| Minimal code (`code/`) | Production code (`zero/`) | What it adds, and why |
|---|---|---|
| `sample_next(logits, temperature, top_k, top_p)` in `02_sampling.py`, one sequence at a time | `sample_next(logits, temperature, top_p, generator)` in `zero/generate.py` | The input is `(B, V)`, so it processes a batch of sequences at once. It does softmax in float32 (this prevents precision problems in BF16 inference). The "keep at least one token" method of top-p is the same. zero has no top-k now (we add it when the local demo in Chapter 20 needs it). |
| The hand-written `generate_cached` loop | `generate(...)` and `generate_stream(...)` in `zero/generate.py` | It supports batches, an early stop with `eos_id` (it pads the finished sequences with eos), and reproducible results with `seed`. `generate_stream` yields tokens while it generates, so the command line can print one character at a time. `use_cache=False` keeps the naive path only for parity checks. |
| `KVCache.append`: `torch.cat` at each step | `KVCache` in `zero/kv_cache.py`: it **allocates the full cache once in advance** with the shape `(layer, batch, n_kv_heads, max_seq_len, head_dim)`. `update(layer, start_pos, k, v)` only writes into it. `nbytes()` reports the memory use | At each step, `torch.cat` allocates and copies the full cache again, which wastes much time on long sequences. With an allocation in advance, a write is O(1), and the memory use is known from the start. (Part 3 of `04_kv_memory.py` makes sure that `nbytes()` agrees with the formula: 117,440,512 bytes for 1024 positions in the main-line config.) |
| `len(cache)` sets the position of the new token | `Transformer.forward(tokens, kv_cache, start_pos)` gets the position explicitly | It supports **chunked prefill**: it can take several tokens at once when the cache already has history. For this case, `Attention.forward` in `zero/model.py` makes its own mask (`j <= past + i`). This is important for the long context in Chapter 15. |
| `repeat_interleave` copies K/V g times before attention | `Attention` in `zero/model.py`: `F.scaled_dot_product_attention(q, k, v, ..., enable_gqa=self.n_kv_heads != self.n_heads)` | SDPA broadcasts the K/V heads internally, so it does not really make `n_heads` copies of the tensors. It can also use fused kernels such as FlashAttention (Chapter 14). |
| Hand-written `q @ kᵀ`, mask, softmax | Same as above, SDPA | The same math. The fused kernel is faster and uses less memory. |
| None | It writes K into the cache after QK-Norm and after the RoPE rotation; YaRN long-context scaling | The K that it reads from the cache is ready to use. All position-dependent calculation occurs before the write. |

**Parity check**: `tests/test_kv_cache.py` (`uv run pytest tests/test_kv_cache.py`; all 7 tests pass on this machine) makes sure of these points:

- Greedy generation: the cached version and the naive version with `use_cache=False` give exactly the same 40 tokens.
- They are also exactly the same with batch 3, with MQA (`n_kv_heads=1`), and with YaRN.
- **Sampling** with temperature and top-p and a fixed seed: the cached version, the naive version, and a second run of the cached version give the same tokens.
- Chunked prefill (20 + 1 + 29 tokens in three calls) gives the same logits as one full forward pass, within 1e-5.
- `eos_id` cuts the output correctly. `sample_next` keeps only the largest token when top-p is very small. `KVCache.nbytes()` equals 2 × layers × batch × KV heads × length × head_dim × 4.

**A real production service** does not use a loop like this. The industry standard is **vLLM**. Its **PagedAttention** cuts the KV cache into "pages" of a fixed size. It allocates the pages when they are necessary and shares them between requests, as an operating system manages virtual memory. This prevents the waste that occurs when each request reserves memory for the maximum length.

vLLM also uses **continuous batching**: when one request finishes, the next request starts immediately, and the service does not wait for the full batch to finish. Together, these methods keep the GPU busy. Chapter 20 uses vLLM when we release the main-line model. We do not discuss it further here.

---

## Adopters and sources

| Technique | Adopters (main versions) | Sources |
|---|---|---|
| KV cache | The standard method of all autoregressive Transformer inference: Hugging Face transformers (`use_cache`; `"use_cache": true` in the `config.json` of each model), vLLM, llama.cpp | Industry standard (class B in GOAL.md 2.1); the vLLM paper |
| Temperature + top-p (nucleus) sampling | Qwen3 (0.6 / 0.95 / top-k 20), Qwen2.5-Instruct (0.7 / 0.8 / top-k 20), Llama 3.1 Instruct (0.6 / 0.9), SmolLM3 (0.6 / 0.95), Gemma 3 it (top-p 0.95 / top-k 64) | The `generation_config.json` of each model (read in 2026-09); Holtzman et al. 2019 |
| GQA | Llama 2 (34B, 70B) and all Llama 3 models (8 KV heads); Qwen2 / Qwen2.5 / Qwen3; Mistral 7B; Gemma 3 (27B: 32 Q / 16 KV); gpt-oss (64 Q / 8 KV); SmolLM3 (16 Q / 4 KV) | Technical reports: Llama 2, Llama 3, Qwen2, Qwen3, Mistral 7B, Gemma 3; `num_key_value_heads` in the `config.json` of each model |
| MQA | Gemma 3 1B (`num_key_value_heads: 1`); this chapter discusses it as an end point of GQA | The `config.json` of `google/gemma-3-1b-pt`; Shazeer 2019 |
| PagedAttention / continuous batching | vLLM (the industry-standard inference engine) | Kwon et al. 2023; Orca (Yu et al. 2022) |

Notes: the official Llama repositories require an access request. Thus we read the `config.json` / `generation_config.json` of Llama-3.1-8B, Llama-3.1-8B-Instruct, and Llama-3.3-70B from the unsloth mirror repositories (`unsloth/Meta-Llama-3.1-8B` and others; their `_name_or_path` points to the official meta-llama repositories). These values agree with the "8 KV heads" in Table 3 of the Llama 3 paper. We could not read the config of Llama 2 70B directly. Thus we use the statement of the Llama 2 paper (34B and 70B use GQA). **The config numbers are to be verified.**

---

## Guided questions

Ask Claude Code these questions. Continue until you can explain the answers in your own words:

1. When the temperature T is very large, the distribution becomes almost uniform. When T → 0, it becomes argmax. Use the softmax formula to explain why. If the logits have two equal maximum values, what happens when T → 0?
2. Why does the KV cache keep only K and V? Why does it not keep Q, the attention output, or the intermediate results of the FFN? If the model uses bidirectional attention (for example, BERT), does the KV cache still work?
3. The cache keeps K after the RoPE rotation. Suppose that we keep K before the rotation and rotate it each time that we read it. Is the result the same? Which method costs less?
4. What exactly does "decode does not use the compute fully" mean? Ask Claude Code to help you estimate it. To decode one token with the main-line model, how many bytes of weights and KV cache must the hardware read? How many multiply-add operations does it do? What is the ratio? (Hint: search for "arithmetic intensity" and "roofline".)
5. In the small experiment of this chapter, the loss difference between MHA and GQA has the same order of magnitude as the difference from the random seed. For a serious comparison, how do you design the experiment? How many seeds and how large a model do you need? How do you report the error?
6. In the GQA paper, the authors made the GQA models from existing MHA checkpoints with "uptraining": they averaged the K and V projections of the heads in a group. Why the average? Are there other initialization methods?

## Hands-on tasks

For each task, run the code and look at the result.

**Task 1 (basic)**: In `02_sampling.py`, choose a context of your own. Print the number of characters that top-p = 0.5, 0.9, and 0.99 keep. Then change the temperature to 0.7. For the same top-p, how does the number of kept characters change? Explain why temperature and top-p affect each other.

**Task 2 (core)**: Change the `KVCache` in `01_tiny_model.py` to a **pre-allocated** version. The constructor gets `max_len` and allocates one tensor with the shape `(n_layers, B, n_kv_heads, max_len, head_dim)`. `append` only writes into `[start:start+T]`. Use `03_kv_cache.py` to make sure that the output is still exactly the same. Then compare the time to generate 512 characters. Compare your code with `zero/kv_cache.py`.

**Task 3 (challenge)**: Add batch support to `03_kv_cache.py`. Generate continuations of 8 different prompts at once (pad the prompts to the same length first, or use the same prompt with different seeds). Measure the total number of generated tokens per second for batch = 1, 4, and 8. Check the statement in Section 4.3 that "batching increases the decode throughput". Then use the formula in `04_kv_memory.py` to calculate the cache size for each batch.

---

## Go deeper: CS336

This chapter corresponds to Stanford CS336 (Spring 2026) <https://cs336.stanford.edu/>:

- **Lecture 10: Inference**. It uses resource accounting to show where the cost of inference comes from (prefill and decode, the memory and the memory bandwidth of the KV cache). It also shows the methods that make inference faster and cheaper (the slides and the recordings are on the course page). This chapter covers only the most basic parts. Chapters 20, 21, and 25 come back to this lecture.
- **The decoding part of Assignment 1 (Basics)**: implement text generation with temperature and top-p on the Transformer that you trained. This is the same task as `02_sampling.py` in this chapter. Assignment repository: <https://github.com/stanford-cs336/assignment1-basics>

---

## References

- Holtzman, Buys, Du, Forbes, Choi. *The Curious Case of Neural Text Degeneration* (nucleus / top-p sampling), 2019: <https://arxiv.org/abs/1904.09751>
- Fan, Lewis, Dauphin. *Hierarchical Neural Story Generation* (top-k sampling), 2018: <https://arxiv.org/abs/1805.04833>
- Shazeer. *Fast Transformer Decoding: One Write-Head is All You Need* (MQA), 2019: <https://arxiv.org/abs/1911.02150>
- Ainslie et al. *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints*, 2023: <https://arxiv.org/abs/2305.13245>
- Kwon et al. *Efficient Memory Management for Large Language Model Serving with PagedAttention* (vLLM), 2023: <https://arxiv.org/abs/2309.06180>
- Yu et al. *Orca: A Distributed Serving System for Transformer-Based Generative Models* (iteration-level scheduling / continuous batching), OSDI 2022: <https://www.usenix.org/conference/osdi22/presentation/yu>
- Touvron et al. *Llama 2: Open Foundation and Fine-Tuned Chat Models*, 2023: <https://arxiv.org/abs/2307.09288>
- Llama Team. *The Llama 3 Herd of Models*, 2024: <https://arxiv.org/abs/2407.21783>
- Qwen Team. *Qwen2 Technical Report*, 2024: <https://arxiv.org/abs/2407.10671>; *Qwen3 Technical Report*, 2025: <https://arxiv.org/abs/2505.09388>
- Jiang et al. *Mistral 7B*, 2023: <https://arxiv.org/abs/2310.06825>
- Gemma Team. *Gemma 3 Technical Report*, 2025: <https://arxiv.org/abs/2503.19786>
- Model configs (read in 2026-09): [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B/blob/main/config.json), [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B/blob/main/config.json) ([generation_config](https://huggingface.co/Qwen/Qwen3-8B/blob/main/generation_config.json)), [Qwen2.5-7B](https://huggingface.co/Qwen/Qwen2.5-7B/blob/main/config.json), [Qwen2.5-7B-Instruct generation_config](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/blob/main/generation_config.json), [Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1/blob/main/config.json), [gemma-3-1b-pt](https://huggingface.co/google/gemma-3-1b-pt/blob/main/config.json), [gemma-3-27b-pt](https://huggingface.co/google/gemma-3-27b-pt/blob/main/config.json), [gemma-3-27b-it generation_config](https://huggingface.co/google/gemma-3-27b-it/blob/main/generation_config.json), [gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b/blob/main/config.json), [SmolLM3-3B](https://huggingface.co/HuggingFaceTB/SmolLM3-3B/blob/main/config.json) ([generation_config](https://huggingface.co/HuggingFaceTB/SmolLM3-3B/blob/main/generation_config.json)), Llama mirrors: [unsloth/Meta-Llama-3.1-8B](https://huggingface.co/unsloth/Meta-Llama-3.1-8B/blob/main/config.json), [unsloth/Llama-3.1-8B-Instruct generation_config](https://huggingface.co/unsloth/Llama-3.1-8B-Instruct/blob/main/generation_config.json), [unsloth/Llama-3.3-70B-Instruct](https://huggingface.co/unsloth/Llama-3.3-70B-Instruct/blob/main/config.json)
- vLLM: <https://github.com/vllm-project/vllm>
- The `generate` function of [nanoGPT](https://github.com/karpathy/nanoGPT) (the shortest code for temperature + top-k), [minimind](https://github.com/jingyaogong/minimind) (inference of a small model with a KV cache)

**Next chapter**: Part 2 ends here. We have a modern Transformer that we can train and that generates text fast. In Part 3, we train a real model: 0.6–0.8B parameters, more than 100 billion tokens, and more than 10,000 dollars of compute. But before we spend the first dollar, we must answer some questions: how do we know if the model is good, what do we compare it with, and how do we make the comparison fair? In Chapter 11, we set the exam first.
