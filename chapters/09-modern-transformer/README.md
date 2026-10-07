# Chapter 9: The modern Transformer — Build a model that writes text from attention

**English** · [中文](README.zh.md)

> **Goal**: After this chapter, you can draw the complete tensor-shape data flow of one forward pass of a modern Transformer. You can explain what problem each part solves and which line of code it is: Pre-Norm RMSNorm, RoPE, SwiGLU, QK-Norm, and tied embeddings. You can also train a small model on a CPU that writes text in a "Shakespeare style".

📺 **Video**: Not published yet. To render it on your computer, run `bash chapters/09-modern-transformer/video/build.sh`.
🧪 **Self-check**: After the chapter, type `/ch09-transformer` in Claude Code.

---

In the previous chapter, we derived attention from a "weighted average". Each position compares its Q with the K of all positions. Then it uses these similarities as weights to average the V vectors. A causal mask makes sure that each position sees only the past. Each of several heads does this independently. But one attention layer is not yet a language model. How do bytes (or tokens) become vectors? Where does the position information come from? After a position gets information from other positions, how does the model process it? How do we stack dozens of layers so that training does not fail? How do we change the result back into "the probability of the next word"?

This chapter solves one problem: **put attention into a complete language model that we can train**. We add, one at a time, the parts that almost all open large models use today. We write a single-file model of about 200 lines and train it on a CPU for 10 to 20 minutes. We watch it go from random characters to text in the form of a play. At the end, we copy its weights without changes into `zero/`, the production code of the main-line model. This shows that the two levels of code calculate the same thing. And `zero` matches the official implementation of Qwen3 item by item.

## 1. Overview: lookup → N Blocks → scores

First, look at the complete structure. A decoder-only Transformer does three things:

1. **Lookup (embedding)**: The model uses the ID of each token to get one row from a `V × d` table. The row is a `d`-dimensional vector. The minimal model in this chapter splits text into bytes, so `V = 256`. It uses `d = 128`.
2. **N Blocks**: Each Block has two sublayers.
   - **Attention**: exchanges information **between positions**. Position t "reads" from the positions before it.
   - **Feed-forward network (FFN)**: processes **each position separately**. The same parameters apply one nonlinear transformation to each position (the MLP of Chapter 3).
3. **Scores (LM head)**: The model multiplies the vector of the last position by a `d × V` matrix. The result is V scores (logits). After softmax, the scores are the probabilities of the next token (Chapter 5).

Each sublayer uses the same pattern: **Pre-Norm + residual connection** (Chapter 6):

```
x = x + Attention(RMSNorm(x))      # exchange information between positions
x = x + FFN(RMSNorm(x))            # process each position separately
```

This is `Block.forward` in the minimal code [`code/02_tiny_transformer.py`](code/02_tiny_transformer.py):

```python
def forward(self, x, cos, sin):
    x = x + self.attn(self.attn_norm(x), cos, sin)  # exchange information between positions
    x = x + self.ffn(self.ffn_norm(x))              # process each position separately
    return x
```

A useful analogy is the **residual stream**. `x` is a "main road" that goes from the input to the output. Its width is always `d`. Each sublayer reads a copy from the main road (and normalizes the copy first). It calculates a small "change" and adds the change back to the main road. Chapter 6 showed that the gradient can then go through the additions directly to the earlier layers.

Pre-Norm puts the normalization at the input of each sublayer, not on the main road. The main road itself stays a "clean identity mapping", so we can train dozens of layers. The original Transformer (2017) used Post-Norm (`x = Norm(x + sublayer(x))`). From GPT-2 onward, models use Pre-Norm.

The sections below explain each part in the order of the data flow.

## 2. Position: attention cannot see the order

### 2.1 The problem

The attention of Chapter 8 looks only at how similar the contents are. The score is `q·k`, and it does not depend on the position of k. What is the result? Do an experiment ([`code/01_position.py`](code/01_position.py)). Give "狗咬人了" ("the dog bit the man") and "人咬狗了" ("the man bit the dog") to the same causal attention. Look at the output of the last character "了" (a word that marks a completed action):

```bash
uv run python chapters/09-modern-transformer/code/01_position.py
```

| | Max difference of the output at "了", "狗咬人了" vs "人咬狗了" |
|---|---:|
| No position information | 5.96 × 10⁻⁸ (floating-point error: the outputs are the same) |
| With RoPE | 0.133 |

Without position information, attention treats the earlier text as a **set**, not as a **sequence**. A weighted average does not depend on the order of its terms. Thus "who bit whom" makes no difference to attention. A language model must know the order, so we must write the position into the model.

GPT-2 learns one more table, a "position table" (learned absolute position embedding). The model adds row m of this table to position m. This method has two problems. It knows only the positions that it saw during training (1024 for GPT-2). Also, the position information mixes with the content vector, and it becomes weaker in the deep layers. Today, the standard method is **RoPE**.

### 2.2 RoPE: rotate q and k by the position

**Rotary Position Embedding (RoPE)** (Su et al., 2021) uses a geometric idea. Put the dimensions of q and k into pairs. Think of each pair as a 2D vector in a plane. At position m, **rotate this 2D vector by the angle m·ω**.

To rotate one dimension pair `(a, b)` by the angle θ:

```
(a, b)  →  (a·cosθ − b·sinθ,  b·cosθ + a·sinθ)
```

Different dimension pairs rotate at different speeds. The speed of pair i is `ω_i = θ_base^(−2i/d)`. The default `θ_base` is 10000. The first pairs rotate fast (1 radian per position). The last pairs rotate very slowly (one full turn takes thousands of positions). They are like the second hand, the minute hand, and the hour hand of a clock.

Why does this encode the position? The dot product is the key. At position m, q rotates by `mω`. At position n, k rotates by `nω`. The dot product of the two vectors depends only on the **change of the angle between them**, which is `(m − n)ω`:

```
RoPE(q, m) · RoPE(k, n) = a function of (m − n) only
```

The absolute positions cancel, and only the **relative position** stays. This is what language needs: "the previous word" is the same relation at the start of a sentence and in the middle. The script tests this directly (it puts the same pair q, k at different positions):

| m | n | m − n | Dot product after the rotation, q_m·k_n |
|---:|---:|---:|---:|
| 3 | 1 | +2 | 2.4426 |
| 10 | 8 | +2 | 2.4426 |
| 50 | 48 | +2 | 2.4426 |
| 5 | 1 | +4 | 0.8593 |
| 40 | 36 | +4 | 0.8593 |
| 1 | 3 | −2 | 0.7563 |

The three pairs at distance 2 have exactly the same dot product. The two pairs at distance 4 also have the same dot product. The values for −2 and +2 are different, so RoPE can tell "before" from "after". Also, the rotation does not change the length (in the script, `|q| = 2.2184`, and after the rotation to position 37 it is still 2.2184). Thus RoPE changes only the "direction", not the "size". It does not change the scale of the attention scores.

The code has only two functions:

```python
def rope_cos_sin(head_dim, seq_len, theta=10000.0):
    inv_freq = theta ** (-torch.arange(0, head_dim, 2).float() / head_dim)  # ω_i = θ^(-2i/d)
    angles = torch.outer(torch.arange(seq_len).float(), inv_freq)           # (T, d/2): m·ω_i
    angles = torch.cat([angles, angles], dim=-1)                            # (T, d)
    return angles.cos(), angles.sin()

def apply_rope(x, cos, sin):
    """Dimension i and dimension i + d/2 make a pair. Rotate the pair in 2D: (a, b) → (a·cos − b·sin, b·cos + a·sin)."""
    a, b = x.chunk(2, dim=-1)
    return x * cos + torch.cat([-b, a], dim=-1) * sin
```

Note the pairing. Here, dimension i and dimension i + d/2 make a pair ("first half and second half"). The RoFormer paper pairs two adjacent dimensions. The two methods are mathematically equivalent (only the order of the dimensions changes). But **the weights are not interchangeable**. The Hugging Face implementations of Llama and Qwen both use "first half and second half". We do the same, so that the parity check below (in "Parity check on two levels") can match.

RoPE applies only to q and k, not to v. The position changes only "where to look", not "what to read". RoPE also has no learnable parameters. In Chapter 15, for long context, we change `θ_base` and the rotation speed of each dimension (YaRN).

## 3. FFN: SwiGLU

Attention "moves" information. The FFN processes the information after the move. The FFN of GPT-2 is the two-layer MLP of Chapter 3: `W₂ · GELU(W₁ x)`, with a hidden width of `4d`.

Modern models use **SwiGLU** (Shazeer, 2020) instead. SwiGLU is a variant of the **Gated Linear Unit (GLU)**:

```
FFN(x) = W_down · ( SiLU(W_gate · x) ⊙ (W_up · x) )
SiLU(z) = z · sigmoid(z)
```

There are two parallel paths. `W_up x` is the "content", and `SiLU(W_gate x)` is the "gate". The model multiplies them element by element (⊙). A gate near 0 closes its dimension, and a large gate makes its dimension larger. Chapter 3 said that SiLU is a smooth version of ReLU. Gating means that **the input itself decides what to close**, and the decision is smooth. Thus a gated FFN can express more than a plain ReLU.

In the experiments of Shazeer, the GLU variants had a lower perplexity than the ReLU/GELU MLPs with the same number of parameters. LLaMA adopted SwiGLU, and SwiGLU became the standard in practice.

```python
def forward(self, x):
    return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))
```

SwiGLU has one more matrix. Does it then have more parameters? To compensate, the hidden width decreases from `4d` to `8/3·d`. Two `d × 4d` matrices have `8d²` parameters, and three `d × (8/3)d` matrices also have `8d²`. [`code/03_shapes.py`](code/03_shapes.py) calculates 13.11M for both at `d = 1280`. The LLaMA paper says: "we use a dimension of 2/3·4d instead of 4d as in PaLM".

This ratio is **a starting point, not a law**. The minimal model in this chapter uses 352 (`8/3 × 128 ≈ 341`, rounded to a multiple of 32). Qwen3-0.6B uses 3072 (3d), Llama 3.2 1B uses 8192 (4d), and the main-line model uses 3584 (2.8d). Each team chose its value from hardware alignment and experiments.

## 4. QK-Norm: a safety limit for the attention scores

The attention score is `q·k / √d`. During training, the lengths of q and k slowly increase, and the scores increase with them. Then softmax becomes "one-sided": almost all the weight goes to one position. The gradients become sharp and unstable. Loss spikes in the training of large models often have this cause (the OLMo 2 report analyzes it in detail).

**QK-Norm** (Dehghani et al., 2023, first used in a vision Transformer with 22 billion parameters) uses a direct method. Before the dot product, it applies RMSNorm to q and k of **each head**. [`code/05_qk_norm.py`](code/05_qk_norm.py) multiplies the same random q, k by s. This simulates a norm that increases during training:

```bash
uv run python chapters/09-modern-transformer/code/05_qk_norm.py
```

| Scale s | No QK-Norm: max score | Mean max weight | Entropy | With QK-Norm: max score | Mean max weight | Entropy |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 4.9 | 0.222 | 2.42 | 2.8 | 0.213 | 2.43 |
| 2 | 19.5 | 0.667 | 0.99 | 2.8 | 0.213 | 2.43 |
| 4 | 77.8 | 0.923 | 0.21 | 2.8 | 0.213 | 2.43 |
| 8 | 311.3 | 0.996 | 0.02 | 2.8 | 0.213 | 2.43 |
| 16 | 1245.0 | 1.000 | 0.00 | 2.8 | 0.213 | 2.43 |

(The entropy of a uniform distribution over 16 positions is ln 16 = 2.77.) Without QK-Norm, at a scale of 4, already 92% of the attention goes to one position. With QK-Norm, the scale of the scores does not depend on the lengths of q and k. In a real model, RMSNorm has a learnable scale weight, so the scores can still increase. But then the increase comes from weights that the model learns slowly. It does not come from a drift of the activations.

In the code, QK-Norm is two lines, after the split into heads and before RoPE:

```python
q = self.q_norm(self.wq(x).view(B, T, self.h, self.hd)).transpose(1, 2)  # (B, H, T, hd)
k = self.k_norm(self.wk(x).view(B, T, self.h, self.hd)).transpose(1, 2)
```

## 5. Input and output: tied embeddings

The model has one `V × d` matrix at each end. At the input, there is the embedding table (token → vector). At the output, there is the LM head (vector → a score for each token). **Tied embeddings** (Press & Wolf, 2017) use the same matrix for both:

```python
self.lm_head.weight = self.tok_emb.weight  # tied embeddings: use the same matrix two times
```

The intuition: at the input, the word "cat" has a vector. At the output, the more similar the vector of a position is to the vector of "cat", the more the model must predict "cat". Thus one table for two uses makes sense, and it also saves parameters. How many parameters does it save? That depends on the fraction of the model that the vocabulary uses. (The table is the output of `code/03_shapes.py`. For Qwen3-0.6B, the script uses the hyperparameters in the official `config.json`.)

| Model | Vocabulary matrix V×d | Total parameters (tied) | Share of vocabulary | If not tied | Actual |
|---|---:|---:|---:|---:|---|
| Minimal model of this chapter (V=256) | 0.03M | 0.84M | 3.9% | 0.87M | Tied |
| `configs/tiny` (V=2048) | 0.26M | 1.05M | 25.0% | 1.31M | Not tied |
| Qwen3-0.6B (V=151,936) | 155.58M | 596.05M | 26.1% | 751.63M | Tied |
| `configs/main` main line (V=65,536) | 83.89M | 689.52M | 12.2% | 773.40M | Tied |

In a small model, the vocabulary is a large part. Without tied embeddings, Qwen3-0.6B would have 156 million more parameters, 26% more in total. This is why **small models usually tie the embeddings and large models usually do not**. In Table 1 of the Qwen3 technical report, 0.6B, 1.7B, and 4B tie the embeddings; 8B and larger do not. Large models such as DeepSeek-V3 and gpt-oss do not tie them (see the sources at the end). The vocabulary of the main-line model is provisionally 65,536. With tied embeddings, the embedding is 83.9M of 689.5M.

`configs/tiny` is an exception. Its comment gives the reason. At this smoke-test size of 1M parameters, the loss left the plateau of "guess only the frequent words" faster without tied embeddings. Thus the demo configuration does not tie them. Such trade-offs are in the configuration, not fixed in the code.

## 6. The complete tensor-shape data flow

Connect the parts and look at the shape at each step of one forward pass. [`code/03_shapes.py`](code/03_shapes.py) puts hooks (forward hooks) on the minimal model and prints the real shapes (B = 2 sequences, T = 16 bytes):

| Module | Input shape | Output shape | Description |
|---|---|---|---|
| `tok_emb` | (2, 16) | (2, 16, 128) | Integers → vectors |
| `layers.0.attn_norm` | (2, 16, 128) | (2, 16, 128) | RMSNorm does not change the shape |
| `layers.0.attn.wq` | (2, 16, 128) | (2, 16, 128) | Then `view` splits it into 4 heads |
| `layers.0.attn.q_norm` | (2, 16, 4, 32) | (2, 16, 4, 32) | QK-Norm works on the 32 dimensions of each head |
| `layers.0.attn` | (2, 16, 128) | (2, 16, 128) | Internal score matrix (2, 4, 16, 16) |
| `layers.0.ffn.w_gate` / `w_up` | (2, 16, 128) | (2, 16, 352) | SwiGLU first makes it wider |
| `layers.0.ffn.w_down` | (2, 16, 352) | (2, 16, 128) | Then back to d |
| `layers.0` … `layers.3` | (2, 16, 128) | (2, 16, 128) | Each Block has the same input and output shape, so Blocks can stack |
| `norm` | (2, 16, 128) | (2, 16, 128) | The last RMSNorm |
| `lm_head` | (2, 16, 128) | (2, 16, 256) | 256 scores for the next byte at each position |

The same data flow with the sizes of the main-line model (`configs/main/pretrain.toml`, one micro batch per GPU: B = 8, T = 4096):

```
token id                  (8, 4096)
embedding lookup          (8, 4096, 1280)
× 28 layers  RMSNorm      (8, 4096, 1280)
  q = x·Wq → split heads  (8, 16, 4096, 128)
  k, v = x·Wk, x·Wv       (8, 8, 4096, 128)  ← GQA: 8 K/V heads (Chapter 10)
  QK-Norm + RoPE          shape does not change
  attention scores QKᵀ    (8, 16, 4096, 4096)
  weights·V → merge → Wo  (8, 4096, 2048) → (8, 4096, 1280)
  add to residual         (8, 4096, 1280)
  SwiGLU: gate, up        (8, 4096, 3584)
  down → add to residual  (8, 4096, 1280)
final RMSNorm             (8, 4096, 1280)
lm_head → logits          (8, 4096, 65536)
```

Remember two points. First, **the shape of the residual stream does not change from start to end**. Each sublayer "reads a copy, calculates a little, and adds it back". Second, **two places are the most expensive**. The attention scores are T × T. The final logits have 8 × 4096 × 65536 ≈ 2.15G numbers, which need 8.0 GiB in float32. FlashAttention and chunked cross-entropy in Chapter 14 are for these two places.

Also, the q of all heads in the main-line model has 16 × 128 = 2048 dimensions in total. This is wider than d = 1280. Qwen3 allows this (`head_dim` does not have to equal `d / n_heads`). Thus `Wo` maps 2048 → 1280.

## 7. Training: from random bytes to a play

All parts are ready, so now we train. The data is `assets/tiny_corpus/shakespeare.txt` (about 1.1 MB, public domain). We split it into bytes. The first 90% is for training, and the last 10% is for validation. Chapter 7 showed that the cross-entropy per byte divided by ln 2 is the **bits-per-byte**. With a vocabulary of 256, a random guess gives 8 bits/byte.

The training loop is the five lines of Chapter 1, plus AdamW, warmup + cosine decay, and gradient clipping from Chapter 6:

```python
x, y = get_batch(train_data, cfg, batch_size, g)          # y = x shifted by one position
loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten())
opt.zero_grad()
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
opt.step()
```

```bash
uv run python chapters/09-modern-transformer/code/02_tiny_transformer.py
```

The model has 836,992 parameters (4 layers, d = 128, 4 heads, FFN 352). The batch is 32 × 128 bytes, the peak learning rate is 3e-3, and the training takes 1200 steps. The bits-per-byte on the validation set:

| Step | 0 | 200 | 400 | 600 | 800 | 1000 | 1200 |
|---|---:|---:|---:|---:|---:|---:|---:|
| val bits/byte | 8.052 | 2.781 | 2.518 | 2.391 | 2.289 | 2.243 | 2.221 |

> **Note:** The numbers of the training experiments in this chapter come from one CPU run on the course build machine. Different machines and different versions of the low-level math libraries use a slightly different order of floating-point operations. A few hundred training steps make these small differences larger. Thus your numbers can be different from the second or third decimal place. Trust the conclusions below, which do not depend on the exact values. For a second run on a different server in 2026-10, see [runs/2026-10-01-gpu0-check/chapters-07-10.md](../../runs/2026-10-01-gpu0-check/chapters-07-10.md).

Before training (random initialization, sampling at temperature 0.8), the model writes 200 bytes after `ROMEO:\n`. They are invalid UTF-8 and control characters, and the terminal shows random symbols. After training, the model writes 400 bytes (an extract is below). On a different machine, the sampled sentences are different, but the format and the word choice have the same features:

```
ROMEO:
And yield for what be a child.

JULIET:
No determina, hear my master warners:
Whose three live well said an your honour:
I having alonged against thee, good both
Your words of all now, and thou thy death,
She was not had succetting with worm dead!
And say we did sound the friends of England,
And doth to well here dew thy justice king.

Second Servingman:
Then I be the heart of your blood than name
```

The model learned the format of a play (character name in capitals + colon + new line). It learned the spelling of most English words, old English words such as `thou`/`thy`, and an approximate rhythm. The meaning does not make sense yet, and sometimes the model makes new words (`succetting`, `warners`). This is the result of 0.84M parameters after about 4.9M bytes (1200 steps × 32 × 128, about 4.9 passes over the training set). Time on the build machine (4 cores, other jobs at the same time): 14.7 min wall-clock time and 10.8 min CPU time on 1 thread. To save time, add `--steps 400`. (Then the learning-rate schedule uses 400 steps, and the result is a little worse. Trust the numbers of your own run.)

During generation, note this comment in the `generate` function:

```python
for _ in range(n_new):  # for each new byte, calculate the full sequence again: slow! Chapter 10 fixes this with a KV cache
```

## 8. Summary

- **Transformer = lookup → N Blocks → scores**. Each Block = attention (exchanges information between positions) + FFN (processes each position separately). Each sublayer is inside Pre-Norm RMSNorm + a residual connection. The shape of the residual stream (B, T, d) does not change from start to end.
- **RoPE**: rotates each dimension pair of q and k by the position. Then the dot product depends only on the relative position m − n.
- **SwiGLU**: `W_down(SiLU(W_gate x) ⊙ W_up x)`, a gated FFN. A hidden width of about 8/3·d gives the same number of parameters as a 4d MLP.
- **QK-Norm**: applies RMSNorm to q and k of each head before the dot product. Then the attention scores do not increase with the drift of the activations.
- **Tied embeddings**: the input table and the output head use the same matrix. The benefit is large in small models, where the vocabulary is a large fraction of the parameters.

### Historical comparison: GPT-2 (2019) and today

| Part | GPT-2 | Today's standard (Qwen3 / Llama 3 / main line of this course) |
|---|---|---|
| Normalization | LayerNorm (subtracts the mean, divides by the standard deviation, has a bias) | RMSNorm (scale only) |
| Position of the normalization | Pre-Norm (GPT-2 already moved LN to the sublayer input; the original Transformer of 2017 used Post-Norm) | Pre-Norm |
| Position encoding | Learned absolute position table, at most 1024 positions | RoPE, no parameters, encodes the relative position |
| FFN | `4d` two-layer MLP + GELU | SwiGLU, about 8/3·d ~ 4d |
| Attention | Multi-head attention | GQA (Chapter 10) + QK-Norm (Qwen3, Gemma 3, OLMo 2) |
| bias | Linear layers and LayerNorm all have a bias | Most models removed it (the Qwen3 report says that it removed the QKV bias of Qwen2) |
| Tied embeddings | Tied | Small models tie them; most large models do not |

You can check the structure of GPT-2 in two places. One is `GPT2Config` in transformers (`activation_function="gelu_new"`, `n_inner` defaults to 4 × hidden, `tie_word_embeddings=True`). The other is `modeling_gpt2.py` (the `wpe` position table, and `ln_1` before the attention). The gpt-oss model card also says: "Similar to GPT-2 we use Pre-LN placement".

---

## From minimal code to production code

The model code of the main-line model is in [`zero/model.py`](../../zero/model.py). Each part has a matching part in the minimal code:

| Minimal code (`code/02_tiny_transformer.py`) | Production code (`zero/`) | What the production code adds, and why |
|---|---|---|
| `RMSNorm` | `RMSNorm` in `zero/model.py` | Changes to float32 for the normalization, then back to the original precision (more numerically stable in BF16 training). Line by line the same as `Qwen3RMSNorm` in HF |
| `rope_cos_sin` + `apply_rope` | `compute_rope_inv_freq`, `RotaryEmbedding`, `apply_rope` | Precomputes cos/sin as buffers that are not in the checkpoint; supports **YaRN** scaling (long context, Chapter 15); `reset_buffers` builds them again after a device change |
| `Attention` (softmax + mask written by hand) | `Attention` | **GQA** (`n_kv_heads < n_heads`, Chapter 10); `F.scaled_dot_product_attention` (on a GPU with BF16, it selects the FlashAttention kernel automatically, Chapter 14; verified on an RTX 3090, see section 1 of [runs/2026-10-01-gpu0-check](../../runs/2026-10-01-gpu0-check/README.md)); **KV cache** and the mask for chunked prefill (Chapter 10); `head_dim` can be different from `d / n_heads` |
| `SwiGLU` | `SwiGLU` | The same (the parameter names `w_gate/w_up/w_down` are the same on purpose) |
| `Block` | `Block` | Also takes `kv_cache` and `start_pos`; the rest is the same |
| `TinyTransformer` | `Transformer` | `tie_embeddings` is configurable; `loss()` calculates in float32 and supports `ignore_index` (the loss mask of SFT in Chapter 16); `num_params`, `flops_per_token` (for MFU and cost in Chapters 12 and 14) |
| `Config` dataclass, fixed in the file | `ModelConfig` in `zero/config.py` + `configs/*.toml` | All hyperparameters are in TOML files: `configs/tiny` (CPU smoke test), `configs/ladder` (ladder experiments), `configs/main` (main line). The code checks the configuration when it reads it (for example, `n_heads` must be divisible by `n_kv_heads`) |
| Training loop written by hand | `zero/train/` (`python -m zero.train.pretrain`) | Data mixing, BF16, gradient accumulation, DDP/FSDP, resume from a checkpoint, logs, checkpoints (Chapter 14) |
| None | `zero/hf.py` | Converts to and from the Hugging Face Qwen3 format: `load_from_hf_qwen3`, `export_to_hf_qwen3` |

### Parity check on two levels

**Level 1: minimal code = zero.** [`code/04_parity_with_zero.py`](code/04_parity_with_zero.py) builds a `zero.Transformer` with the hyperparameters of the minimal model. It loads the trained weights directly with `load_state_dict` (each parameter name has a match, so strict mode also passes). Then it compares the logits on the same text:

```bash
uv run python chapters/09-modern-transformer/code/04_parity_with_zero.py
```

```
Parameters: minimal 836,992  zero 836,992
1) minimal vs zero: logits shape (1, 66, 256), max absolute difference 1.43e-05, positions with the same next-byte prediction 100%
2) minimal vs transformers.Qwen3ForCausalLM: max absolute difference 1.43e-05
3) greedy generation of 80 bytes, the two sides are identical: True
```

To align the two models, only two configuration points are necessary. First, the minimal code has no GQA, so zero sets `n_kv_heads = n_heads` (each query head has its own K/V; this is normal multi-head attention). Second, the minimal code has no bias, uses tied embeddings, `θ_base = 10000`, and `eps = 1e-6`, and zero uses the same settings. The remaining difference is the order of calculation. The minimal code writes `softmax(QKᵀ/√d)·V` by hand, and zero calls the SDPA of PyTorch. The math is the same, but the order of floating-point operations is different.

Thus the results **agree within floating-point error** (after training, the maximum difference is 1.4 × 10⁻⁵, and `torch.testing.assert_close(rtol=1e-5, atol=1e-5)` passes). They are not identical bit for bit. With random initial weights, the maximum difference is 5.96 × 10⁻⁷. Item 3 also checks a topic of Chapter 10. With a KV cache, zero generates 80 bytes greedily. These bytes are identical to the output of the minimal code, which calculates the full sequence again at each step.

**Level 2: zero = the official Qwen3 implementation.** Item 2 uses `export_to_hf_qwen3` to export the zero model to a Hugging Face folder. Then it loads the folder again with the official `Qwen3ForCausalLM.from_pretrained` of transformers. The logits agree. The systematic evidence for this level is in [`tests/test_model_hf_parity.py`](../../tests/test_model_hf_parity.py). The test initializes a small `Qwen3ForCausalLM` with random weights. (It also randomizes the RMSNorm weights, which makes the parity check stricter.) It copies the weights into zero and covers five cases: tied and untied embeddings, default RoPE, two types of YaRN scaling, and no GQA with `n_heads × head_dim ≠ d`.

In all cases, the error of the logits is less than 1e-5. The test also checks that the number of parameters is exactly the same as in HF, and that the loss agrees with HF:

```bash
uv run pytest -q tests/test_model_hf_parity.py      # on the build machine: 8 passed in 4.24s
```

Thus **zero is an implementation of the Qwen3 structure (dense version)**. After you export a trained model with `zero/hf.py`, transformers, vLLM, and llama.cpp can load it directly (Chapter 20). This is also the real identity of the minimal model in this chapter: a "mini Qwen3" with 4 layers and a vocabulary of 256.

### Production training entry point

In `zero`, a configuration file controls the training of the same model structure:

```bash
uv run python -m zero.train.pretrain --config configs/tiny/pretrain.toml
```

We ran this on the build machine. (We added `--set train.out_dir=<temporary folder>`, so that the run does not overwrite the existing `out/tiny/pretrain` in the repository.) `configs/tiny` has 1.31M parameters (4 layers, d = 128, GQA 4/2, untied embeddings). It uses a BPE tokenizer with a vocabulary of 2048. The tokenizer is trained on three corpora in `assets/tiny_corpus`: Chinese, English, and code. (If the shards and the tokenizer do not exist, `[data.prepare]` in the configuration makes them during the run.) The run mixes the three sources at 0.45 / 0.45 / 0.1, uses 2048 tokens per step, and trains for 200 steps:

| Step | 1 | 25 | 50 | 100 | 150 | 200 |
|---|---:|---:|---:|---:|---:|---:|
| Training loss (nat/token) | 7.651 | 6.384 | 6.394 | 6.107 | 5.869 | 5.474 |
| Validation loss | | | | 6.005 | | 5.712 |

Wall-clock time: 1 min 25 s. Throughput: about 3,650–6,240 tokens/s (1 thread, busy machine). This is a smoke test. It shows only that "the same production code runs from end to end, and the loss decreases". Its loss is per BPE token (vocabulary 2048), so you cannot compare it directly with the bits-per-byte above. The conversion needs the mean number of bytes per token (Chapter 7), and we do not do it here.

## Frontier notes

The items below occur in the newest models. They are not yet a consensus under the rules of this course, so they are not in the main text:

- **Layers without RoPE (NoPE)**: The model card of SmolLM3 says that it uses "GQA and NoPE (3:1)". In its `config.json`, 1 of every 4 layers has no RoPE. The causal mask itself leaks a little position information (position i can see i tokens). Thus a decoder can train even with no position encoding at all. Only a few model families do this now.
- **Normalization on the sublayer output**: OLMo 2 moved RMSNorm from the sublayer input to the sublayer output (`h = x + RMSNorm(Attn(x))`; the report calls it "reordered norm"). Gemma 3 puts it at both the input and the output (the report says "post-norm and pre-norm with RMSNorm"). Thus some teams test "one more norm in addition to Pre-Norm" in large-scale training. But the standard models (Llama, Qwen, DeepSeek, gpt-oss) still use only Pre-Norm.
- **Rotation of only some dimensions (partial RoPE)**: The `config.json` of Qwen3.5-0.8B has `partial_rotary_factor = 0.25`. GLM-4.5-Air uses 0.5.
- **GeGLU**: The Gemma family uses GELU for the gate (`hidden_activation = gelu_pytorch_tanh`). The structure is the same as SwiGLU. Only the activation function is different.

## Adopters and sources

| Technique | Adopters (leading open model families) | Source |
|---|---|---|
| RMSNorm | Llama | LLaMA paper, section 2.2, "Pre-normalization … We use the RMSNorm": <https://arxiv.org/abs/2302.13971> |
| | Qwen3 | Qwen3 technical report, section 2, "RMSNorm with pre-normalization": <https://arxiv.org/abs/2505.09388> |
| | Gemma 3 | Gemma 3 technical report, section 2, "post-norm and pre-norm with RMSNorm": <https://arxiv.org/abs/2503.19786> |
| | OLMo 2 | OLMo 2 report, section 2.1 (changes from a nonparametric LayerNorm to RMSNorm): <https://arxiv.org/abs/2501.00656> |
| | gpt-oss | Model card, section 2.2, "root mean square normalization … before each attention and MoE block": <https://arxiv.org/abs/2508.10925> |
| Pre-Norm | Llama | LLaMA paper, section 2.2, "normalize the input of each transformer sub-layer": <https://arxiv.org/abs/2302.13971> |
| | Qwen3 | Qwen3 technical report, section 2, "with pre-normalization": <https://arxiv.org/abs/2505.09388> |
| | gpt-oss | Model card, section 2.2, "Similar to GPT-2 we use Pre-LN placement": <https://arxiv.org/abs/2508.10925> |
| | DeepSeek-V3 | Technical report, Figure 2 (in each Block, RMSNorm is before the attention and before the FFN): <https://arxiv.org/abs/2412.19437> |
| SwiGLU | Llama | LLaMA paper, section 2.2, "SwiGLU activation function … 2/3·4d": <https://arxiv.org/abs/2302.13971> |
| | Qwen3 | Qwen3 technical report, section 2: <https://arxiv.org/abs/2505.09388>; `config.json` of `Qwen/Qwen3-0.6B` (`hidden_act: silu`) |
| | OLMo 2 | OLMo 2 report, section 2.1, "SwiGLU … approximately 8/3·d": <https://arxiv.org/abs/2501.00656> |
| | gpt-oss | Model card, section 2.2, "The MoE blocks use the gated SwiGLU activation" (a non-standard implementation with clamping and a residual connection): <https://arxiv.org/abs/2508.10925> |
| RoPE | Llama | LLaMA paper, section 2.2, "Rotary Embeddings": <https://arxiv.org/abs/2302.13971> |
| | Qwen3 | Qwen3 technical report, section 2: <https://arxiv.org/abs/2505.09388> |
| | Gemma 3 | Gemma 3 technical report, section 2 (RoPE base frequency 1M for global layers, 10k for local layers): <https://arxiv.org/abs/2503.19786> |
| | DeepSeek-V3 | Technical report, section 2.1.1 (in MLA, a separate decoupled key carries RoPE): <https://arxiv.org/abs/2412.19437> |
| | gpt-oss | Model card, section 2.2, "rotary position embeddings … YaRN": <https://arxiv.org/abs/2508.10925> |
| QK-Norm | Qwen3 | Qwen3 technical report, section 2, "introduce QK-Norm … to ensure stable training": <https://arxiv.org/abs/2505.09388> |
| | Gemma 3 | Gemma 3 technical report, section 2, "we replace the soft-capping of Gemma 2 with QK-norm": <https://arxiv.org/abs/2503.19786> |
| | OLMo 2 | OLMo 2 report, sections 2.1 and 3.3.2, "This avoids attention logits being too large": <https://arxiv.org/abs/2501.00656> |
| Tied embeddings in small models | Qwen3 | Technical report, Table 1: 0.6B / 1.7B / 4B tie the embeddings, 8B and larger do not: <https://arxiv.org/abs/2505.09388>; `config.json` of `Qwen/Qwen3-0.6B`: `tie_word_embeddings: true` |
| | Llama 3.2 | `config.json` of `Llama-3.2-1B`: `tie_word_embeddings: true` (the official repository needs an access request; we read the mirror `unsloth/Llama-3.2-1B`: <https://huggingface.co/unsloth/Llama-3.2-1B/blob/main/config.json>) |
| | SmolLM | `config.json` of `HuggingFaceTB/SmolLM3-3B` and of `SmolLM2-135M`: `tie_word_embeddings: true` (<https://huggingface.co/HuggingFaceTB/SmolLM3-3B>) |
| | Gemma 3 | The `config.json` of `google/gemma-3-1b-it` does not have this field, and `Gemma3TextConfig` in transformers has the default `tie_word_embeddings=True`. Table 1 of the technical report lists only one embedding parameter count of 302M (= 262,144 × 1,152) |

We read all `config.json` files above from the Hugging Face Hub on 2026-09-26. We also checked counterexamples: OLMo-2-0425-1B (`tie_word_embeddings: false`), DeepSeek-V3, and gpt-oss-20b do not tie the embeddings.

## Guided questions

Ask Claude Code these questions. Continue until you can explain the answers in your own words:

1. Change Pre-Norm in the Block to Post-Norm (`x = RMSNorm(x + Attn(x))`). Does the residual stream still have a "clean identity path"? Why does this make a deep network more difficult to train? (Hint: go back to the residual experiment of Chapter 6.)
2. The RoPE dot product depends only on m − n. Then why do different dimension pairs use different rotation speeds? If all dimension pairs use ω = 1, what occurs at position 1 and position 1 + 2π·k?
3. Why does RoPE rotate only q and k, and not v? What occurs if we also rotate v?
4. Tied embeddings make `lm_head.weight` and `tok_emb.weight` the same tensor. In the backward pass, which two places does the gradient of this matrix come from? How must `weight decay` treat this matrix?
5. Change the model of this chapter from byte level to the BPE of Chapter 7 (for example, a vocabulary of 2048). How do the number of parameters, the amount of text in each step, and the bits-per-byte change?
6. To generate 400 bytes, we call the model 400 times. Each call calculates the full earlier sequence again. Which calculations occur more than one time? (This is the starting point of Chapter 10.)

## Hands-on tasks

**Task 1 (basic)**: Run `01_position.py`. Change the pairing in `apply_rope` to "adjacent pairs" (dimensions 0 and 1 are a pair, dimensions 2 and 3 are a pair, and so on). Make sure that the dot product still depends only on m − n. Then think: why does `04_parity_with_zero.py` fail after this change?

**Task 2 (core)**: Do an ablation in `02_tiny_transformer.py`. Change only one thing each time, and train for 400 steps each time (`--steps 400`). Record the validation bits-per-byte: (a) remove RoPE; (b) replace SwiGLU with a `4d` GELU MLP; (c) remove QK-Norm; (d) do not tie the embeddings. Which change has the largest effect? Is the result what you expected? (At this small size, some differences can be inside the noise. Run again with a different random seed.)

**Task 3 (challenge)**: Change the training data to `assets/tiny_corpus/chinese_poetry.txt`. In UTF-8, one Chinese character is 3 bytes. Thus the same context of 128 bytes contains only about 40 characters. After training, do the generated poems often contain half characters (shown as �)? Try a larger `seq_len`, or change to the BPE of Chapter 7. Compare the quality and the speed.

## Go deeper: CS336

This chapter matches **Lecture 3, "Architectures and hyperparameters"** of [CS336](https://cs336.stanford.edu/) (Spring 2026). The lecture covers these choices: the position of the normalization, activation functions and gating, position encoding, FFN width, number of heads, and vocabulary size. It also explains why each model family made its choices. Thus it is an advanced version of the "Adopters" table of this chapter. **Assignment 1 (Basics)** asks you to write BPE, a Transformer language model, AdamW, and a training loop from zero. After this chapter, the assignment is much less work. The course page has the notes and the video of each lecture.

## References

- Vaswani et al. (2017). *Attention Is All You Need*: <https://arxiv.org/abs/1706.03762>
- Radford et al. (2019). *Language Models are Unsupervised Multitask Learners* (GPT-2): <https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf>
- Xiong et al. (2020). *On Layer Normalization in the Transformer Architecture* (Pre-LN vs Post-LN): <https://arxiv.org/abs/2002.04745>
- Zhang & Sennrich (2019). *Root Mean Square Layer Normalization*: <https://arxiv.org/abs/1910.07467>
- Su et al. (2021). *RoFormer: Enhanced Transformer with Rotary Position Embedding*: <https://arxiv.org/abs/2104.09864>
- Shazeer (2020). *GLU Variants Improve Transformer*: <https://arxiv.org/abs/2002.05202>
- Dehghani et al. (2023). *Scaling Vision Transformers to 22 Billion Parameters* (QK-Norm): <https://arxiv.org/abs/2302.05442>
- Wortsman et al. (2023). *Small-scale proxies for large-scale Transformer training instabilities*: <https://arxiv.org/abs/2309.14322>
- Press & Wolf (2017). *Using the Output Embedding to Improve Language Models* (tied embeddings): <https://arxiv.org/abs/1608.05859>
- Touvron et al. (2023). *LLaMA*: <https://arxiv.org/abs/2302.13971>
- Qwen Team (2025). *Qwen3 Technical Report*: <https://arxiv.org/abs/2505.09388>
- Gemma Team (2025). *Gemma 3 Technical Report*: <https://arxiv.org/abs/2503.19786>
- OLMo Team (2025). *2 OLMo 2 Furious*: <https://arxiv.org/abs/2501.00656>
- OpenAI (2025). *gpt-oss-120b & gpt-oss-20b Model Card*: <https://arxiv.org/abs/2508.10925>
- DeepSeek-AI (2024). *DeepSeek-V3 Technical Report*: <https://arxiv.org/abs/2412.19437>
- nanoGPT (single-file training code for the GPT-2 structure; read it side by side with the modern structure of this chapter): <https://github.com/karpathy/nanoGPT>
- minimind (a small model with the Llama structure, trained from zero; in Chinese): <https://github.com/jingyaogong/minimind>
- A quick tour of the main LLM structures in 15,000 Chinese characters (Llama, Qwen, GLM, DeepSeek, …; in Chinese): <https://zhuanlan.zhihu.com/p/2060741715095560795>
- CS336 Language Modeling from Scratch: <https://cs336.stanford.edu/>

**Next chapter**: Our model can write, but it writes slowly. For each new byte, it calculates the full earlier sequence again, and the previous step already did most of this work. Chapter 10 stores the K and V that the model already calculated (KV cache). Then it lets several query heads share one set of K and V (GQA), which makes the cache smaller. It also explains the sampling controls, such as temperature and top-p.
