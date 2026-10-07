# Chapter 23: Linear attention and hybrid architectures — Compress the KV cache into a fixed-size matrix

**English** · [中文](README.zh.md)

> **Goal**: After this chapter, you can start from softmax attention and derive the recurrent form of linear attention, `S_t = S_{t−1} + v_t k_tᵀ`. You can write the update formulas of the delta rule and Gated DeltaNet, and you can verify that "chunkwise form == recurrent form". Qwen3.5-0.8B is a hybrid of "3 linear layers + 1 full-attention layer". You can also use its real configuration to calculate how much KV cache such a hybrid saves. Finally, you can explain why such a model still keeps those few full-attention layers.

📺 **Video**: Not published yet. To render it on your computer, run `bash chapters/23-linear-attention-hybrid/video/build.sh`.
🧪 **Self-check**: After the chapter, type `/ch23-linear-attention` in Claude Code.

---

In the last chapter, we used a sliding window to limit the length that attention reads. Some layers read only the most recent few thousand tokens. Then the KV cache no longer grows without limit as the context grows. But those layers really cannot see the content outside the window. This chapter asks a different question: **Can one attention layer "see" the full history and use only a fixed amount of memory?**

The answer is **linear attention**. Remove the softmax, and attention becomes an RNN (recurrent neural network): one d×d state matrix holds the full history in compressed form. Compression always has a cost: the state becomes "full", and exact recall becomes worse. Thus two improvements followed: the decay gate and the delta rule. Together, they give Gated DeltaNet. The main answer of the industry today is a hybrid architecture: **a few full-attention layers + many linear layers**. Qwen3.5 uses a 3:1 hybrid even in its small 0.8B model.

The code for this chapter:

```bash
uv run python chapters/23-linear-attention-hybrid/code/01_linear_attention.py   # associativity, KV vs state, one decode step (a few seconds)
uv run python chapters/23-linear-attention-hybrid/code/02_chunked.py            # decay gate + chunkwise parallel form (10 to 20 seconds)
uv run python chapters/23-linear-attention-hybrid/code/03_delta_rule.py         # overwrite, capacity, chunkwise form of Gated DeltaNet (a few seconds)
uv run python chapters/23-linear-attention-hybrid/code/04_hybrid_lm.py          # small language models with four layouts (slow on the first run, then reads the cache)
uv run python chapters/23-linear-attention-hybrid/code/05_associative_recall.py # associative recall: pure linear vs hybrid (slow on the first run)
```

> **Note:** We measured all times in this chapter with one thread on a CPU that other jobs shared. If you run the same script several times, the times can differ by a large factor. Use the times in the tables only to see the **order of magnitude and the trend**. The numerical results (errors, losses, accuracies) use fixed random seeds, so you can reproduce them.

## 1. The problem: softmax attention must remember everything

Recall the causal attention of Chapter 8. To generate token t, it calculates:

```
o_t = Σ_{j≤t} softmax_j( q_t·k_j / √d ) · v_j
```

The softmax normalizes **all t scores of this row** together. If one k_j is missing, we cannot calculate the denominator. Thus the model must keep the K and V of every earlier position. This is the KV cache of Chapters 10 and 21, and it grows linearly with the context length. Part 2 of `01_linear_attention.py` calculates the memory for one layer and one head (d = 64, BF16):

| Context length T | KV cache | Linear-attention state |
|---:|---:|---:|
| 128 | 32 KB | 8 KB |
| 1,024 | 256 KB | 8 KB |
| 8,192 | 2,048 KB | 8 KB |
| 65,536 | 16,384 KB | 8 KB |
| 262,144 | 65,536 KB | 8 KB |

The right column is the goal of this chapter: **the size stays the same for any context length.**

## 2. Remove the softmax: change the order of the matrix products

Suppose that the attention scores do not go through a softmax. Instead, each score is `φ(q)·φ(k)`. Here, φ is a feature map that makes the vector entries non-negative; Katharopoulos et al. (2020) used `elu(x) + 1`. Then the output of the full sequence is a product of three matrices, multiplied element-wise by a causal mask M:

```
O = ( φ(Q) φ(K)ᵀ ⊙ M ) V
```

Matrix multiplication is **associative**. Without the mask, `(φ(Q) φ(K)ᵀ) V = φ(Q) (φ(K)ᵀ V)`. The left side first builds a large T×T matrix. The right side needs only a small d×d matrix, which does not depend on T. The causal mask means that position t adds only the terms at positions up to t. Thus we can calculate the output step by step:

```
S_t = S_{t−1} + v_t φ(k_t)ᵀ      (write: the state S is a d_v × d_k matrix)
o_t = S_t φ(q_t)                 (read)
```

This is a **recurrent neural network (RNN)**. For each new token, add the outer product of v and k to the state. Then read the state with q. The memory and the compute of each step are constant. The code (`01_linear_attention.py`):

```python
def linear_attention_recurrent(q, k, v):
    S = torch.zeros(v.shape[1], q.shape[1])
    out = []
    for t in range(q.shape[0]):
        S = S + torch.outer(v[t], phi(k[t]))  # S_t = S_{t-1} + v_t φ(k_t)ᵀ   (write)
        out.append(S @ phi(q[t]))             # o_t = S_t φ(q_t)                (read)
    return torch.stack(out)
```

We compare it with the parallel form, which calculates the full T×T matrix at once. With T = 256 and d = 16, the **maximum relative error** between the outputs of the two algorithms is **4.7e-07**. This is only the float32 rounding error. Why does this not work with the softmax? The denominator of the softmax ties a full row together, and we cannot split it. Thus softmax attention has no recurrent form of this kind.

> **Note:** In the original paper, linear attention also divides by a normalization term `φ(q_t)ᵀ Σ_j φ(k_j)`. Most later work (GLA, DeltaNet, Qwen3.5, and others) removes this denominator. Instead, it applies L2 normalization to q and k, and it adds an RMSNorm to the output. This is numerically more stable. From Section 3 on, this chapter uses this modern form (φ is the identity map).

**How much faster is inference?** Part 3 of `01` measures the attention part of one generation step: T earlier tokens exist, and the model generates the next token (one head, d = 64):

| Existing context | Softmax attention | Linear attention |
|---:|---:|---:|
| 1,024 | 0.033 ms | 0.030 ms |
| 8,192 | 0.250 ms | 0.030 ms |
| 65,536 | 12.5 ms | 0.030 ms |
| 262,144 | 44.2 ms | 0.030 ms |

At each step, softmax attention reads all T K and V vectors, so the time grows with T. At each step, linear attention uses only one 64×64 state, so the time is constant. (The times vary a lot: in another run, the cell for 262,144 was 19–25 ms.)

## 3. Training: the chunkwise parallel form

Inference can go one token at a time. But training processes the full sequence at once. A token-by-token for loop cannot run in parallel, so the GPU waits. The fully parallel form `(QKᵀ ⊙ M) V` goes back to the T×T matrix. The compromise is the **chunkwise** form. Cut the sequence into chunks of length C:

- **In a chunk**: use the parallel form, one matrix multiplication.
- **Between chunks**: pass on the state S, which compresses all earlier chunks. Each position in the chunk also reads S one time.

```python
for s in range(0, T, C):
    qc, kc, vc = q[s:s+C], k[s:s+C], v[s:s+C]
    b = g[s:s+C].cumsum(0)                                   # cumulative log decay in the chunk (next section)
    D = (b[:, None] - b[None, :]).masked_fill(~mask, -inf).exp()
    inter = (qc * b.exp()[:, None]) @ S.T                    # read the state from the chunk start
    intra = ((qc @ kc.T) * D) @ vc                           # parallel attention in the chunk
    out[s:s+C] = inter + intra
    S = b[-1].exp() * S + vc.T @ (kc * (b[-1] - b).exp()[:, None])   # compress the full chunk into the state
```

`02_chunked.py` verifies that the three algorithms give the same numbers. With T = 512, the maximum difference is 1.8e-06 for recurrent vs parallel and 6.0e-07 for recurrent vs chunkwise. Then it compares the speed (one head, d = 64, one CPU thread, ms):

| T | Recurrent (token by token) | Chunkwise, C = 64 | Fully parallel |
|---:|---:|---:|---:|
| 256 | 33.4 | 4.1 | 3.9 |
| 1,024 | 76.2 | 12.7 | 79.6 |
| 4,096 | 344.1 | 47.2 | 2015.0 |

For short sequences, the fully parallel form is the fastest (one large matrix multiplication). For long sequences, the T² cost is larger than all other costs. The chunkwise form is then about 7 times faster than the token-by-token form and more than 40 times faster than the fully parallel form. On a GPU, the difference is larger, because the chunkwise form changes the work into matrix multiplications, which GPUs do best. **Use the chunkwise form for training and prefill, and the recurrent form for decode.** The two forms are mathematically equivalent. All linear-attention models use this standard method.

## 4. The decay gate: learn to forget

Naive linear attention has a clear problem: it **can only add**. Content from 10,000 tokens ago has the same weight as the newest content. The state collects more and more content. The first improvement adds a **decay gate**:

```
S_t = α_t · S_{t−1} + v_t k_tᵀ,     α_t ∈ (0, 1]
```

Before each write, multiply the full old state by α_t. `02_chunked.py` writes only once, at step 0. Then it shows how the written value decays with distance:

| α | Distance 10 | Distance 100 | Distance 500 |
|---|---:|---:|---:|
| 1.0 | 1.000 | 1.000 | 1.000 |
| 0.99 | 0.904 | 0.366 | 0.007 |
| 0.9 | 0.349 | 0.000 | 0.000 |

The values are exactly α^distance. More important: in real models, **the model calculates α_t from the current input** (data-dependent). When the topic changes, the model can make the gate smaller and clear the old memory. When it must remember for a long time, it can open the gate to near 1. In the chunkwise form, the decay adds only the `D` matrix and a few `exp(b)` factors in the code above. Here, b is the cumulative sum of the log decays. The exponent is always "later minus earlier", so it cannot overflow.

Many models share this skeleton, a "linear recurrence with decay". They differ only in how they parameterize α. RetNet uses a fixed constant for each head. GLA (Gated Linear Attention) uses a data-dependent vector gate. Mamba-2 uses a data-dependent scalar gate for each head. (The Mamba-2 paper puts state space models of this type and linear attention into one framework, "structured state space duality".)

This chapter does not discuss each of these models. It discusses only their common skeleton. Of this family, mainstream models really use two branches: Mamba-2 and the Gated DeltaNet family (see below).

## 5. The cost: a fixed-size memory becomes full

This compression has a cost. When we compress the full history into a fixed-size matrix, the memory becomes "full". Think of the state S as a "key → value" table. Write (k, v), then read with the same k. We want `S k ≈ v`. If the keys are pairwise orthogonal, a state with d_k key dimensions can store at most d_k key-value pairs without interference. With more pairs, different keys interfere with each other.

Part 2 of `03_delta_rule.py` writes N random unit-vector keys and random values into a 64×64 state. Then it reads each one back and reports the relative error `‖S k − v‖ / ‖v‖`. Here, 0 = a perfect read, and 1 ≈ the read noise is as large as the signal:

| N written | Linear attention | Delta rule (next section) | Softmax attention (stores all KV) |
|---:|---:|---:|---:|
| 16 | 0.469 | 0.309 | 0.000 |
| 32 | 0.692 | 0.479 | 0.000 |
| 64 | 0.996 | 0.737 | 0.000 |
| 128 | 1.431 | 0.972 | 0.000 |
| 256 | 1.974 | 1.206 | 0.000 |

Softmax attention stores K and V unchanged and "looks up" the table with the key. Its error is always 0, but its memory grows with N. Linear attention has a fixed memory, and it pays with **exact recall**: the more it writes, the less accurate the read. This is the largest weakness of linear attention. Zoology (Arora et al. 2023) measured it systematically with the "multi-query associative recall (MQAR)" task. In Section 8.3, we do a small version of this task.

## 6. The delta rule: overwrite, do not add

Linear attention writes blindly. It adds `v kᵀ` directly and ignores what the state already contains. If we write the same key two times, the read gives the sum of the two values. The **delta rule** writes in a different way. (It comes from the classic Widrow–Hoff learning rule. Schlag et al. used it for linear attention in 2021 and called the result DeltaNet.) First, it uses k to read the old answer `S k`. Then it writes back only "the difference between the new value and the old answer":

```
S_t = S_{t−1} + β_t (v_t − S_{t−1} k_t) k_tᵀ
    = S_{t−1} (I − β_t k_t k_tᵀ) + β_t v_t k_tᵀ
```

β_t ∈ (0, 1) is the write strength (the model also calculates it from the input). When β = 1 and ‖k‖ = 1, `S_t k_t = v_t` is exactly true after the write: the new value cleanly **overwrites** the old value. The update is one step of gradient descent (learning rate β) on the squared error between S k and v. Thus people also call it a kind of "test-time learning". Part 1 of `03_delta_rule.py` writes v1 = [1, 0] and then v2 = [0, 1] with the same key:

| | Read with k |
|---|---|
| Linear attention | [1.0, 1.0] (v1 + v2) |
| Delta rule | [0.0, 1.0] (v2) |

Look at the capacity table of the last section. For each N, the delta rule has a lower error than naive addition (0.737 vs 0.996 at N = 64). Before it writes, it "erases" the old content in the direction of the key. This makes the interference much smaller.

## 7. Gated DeltaNet: decay × delta rule

Each of the two improvements does one job. The decay gate fades out all old information (good for a "topic change"). The delta rule changes exactly the content of one key (good for "update one memory"). Gated DeltaNet (Yang, Kautz, Hatamizadeh 2024) multiplies them together:

```
S_t = α_t · S_{t−1} (I − β_t k_t k_tᵀ) + β_t v_t k_tᵀ
o_t = S_t q_t
```

Part 3 of `03_delta_rule.py` does a streaming experiment. It writes 1024 key-value pairs one after the other and reads only the 32 newest:

| Write rule | Read error of the 32 newest |
|---|---:|
| Linear attention | 4.049 |
| Linear + fixed decay α = 0.95 | 0.644 |
| Delta rule | 0.584 |
| Delta + fixed decay α = 0.95 | 0.652 |
| Linear + α = 0 at the topic change | 0.675 |
| Delta + α = 0 at the topic change (data-dependent gate) | 0.502 |

We see four things. The 1000 old items overwhelm naive addition (4.05). Decay or the delta rule alone brings the error back to about 0.6. A **fixed** decay on top of the delta rule is a little worse, because it also decays the newest items. The best result (0.502) comes from the delta rule with a **data-dependent** gate that closes to 0 at the topic change.

Thus the value of the gate is that "the input decides when to forget". This is also why Gated DeltaNet must calculate α_t from the current token. But 0.5 is still far from the 0 of softmax attention: a fixed-size state always has a capacity limit.

**The chunkwise form.** The delta rule has one more difficulty than naive linear attention. Position i of a chunk writes a "correction" u_i = β_i (v_i − S k_i) into the state. This value depends on the u_j that the earlier positions write, so one matrix multiplication cannot calculate it directly. If we write the dependencies in the chunk as matrices, we get a lower-triangular linear system:

```
(I + A) U = β ⊙ (V − e^{b} ⊙ K S₀ᵀ),     A_ij = β_i e^{b_i − b_j} (k_i · k_j)   (j < i)
```

Solve for U with one triangular solve (the "UT transform" / WY representation of Yang et al. 2024). The rest has exactly the same form as linear attention with decay, but with u in place of v:

```python
A = (bt[..., :, None] * (kc @ kc.mT) * D).tril(-1)            # A_ij = β_i e^{b_i−b_j} k_i·k_j, j<i
rhs = bt[..., None] * (vc - (kc * b.exp()[..., None]) @ S.mT)  # β_i (v_i − e^{b_i} S₀ k_i)
u = torch.linalg.solve_triangular(eye + A, rhs, upper=False)
out.append((qc * b.exp()[..., None]) @ S.mT + ((qc @ kc.mT) * D) @ u)
S = b[..., -1, None, None].exp() * S + u.mT @ (kc * (b[..., -1:] - b).exp()[..., None])
```

Part 4 of `03_delta_rule.py` verifies this. The test uses B = 2, H = 3, T = 128, and a chunk length of 32. The maximum output difference between the chunkwise and recurrent forms is 4.8e-07. The difference of the final states is 3.6e-07.

**A Gated DeltaNet layer in a real model** has some more parts. We follow Qwen3.5 here; see `zero/arch/linear_attention.py`. The parts are:

- After the q, k, v projections, a **short causal convolution** with kernel size 4 (a depthwise convolution). Each position first mixes in the information of a few neighbor tokens on its left. This helps recall tasks a lot.
- **L2 normalization** of q and k. It makes sure that ‖k‖ = 1, which keeps the delta rule stable.
- The Mamba-2 parameterization of α_t: `α_t = exp(−e^{A} · softplus(a_t + dt_bias))`. Also, β_t = sigmoid(b_t).
- A **gated RMSNorm** on the output: RMSNorm(o) ⊙ SiLU(z).
- Multiple heads. The number of value heads can be an integer multiple of the number of q/k heads (Qwen3-Next has 16 q/k heads and 32 v heads).

The linear layers **do not use RoPE**. The recurrence itself has an order in time, and the short convolution also gives local position information.

## 8. Hybrid: a few full-attention layers + many linear layers

Linear layers have a limited memory, and full attention is too expensive. The answer of the industry is a **hybrid**. Most layers use linear attention (or state space layers). After every few layers, there is one full-attention layer, which does the exact recall.

### 8.1 The real architecture of Qwen3.5-0.8B

We read this from the `config.json` of `Qwen/Qwen3.5-0.8B` on Hugging Face (2026-09):

```
"full_attention_interval": 4,
"layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention", ... ×6]
"num_key_value_heads": 2, "head_dim": 256,                       # full-attention layers (gated attention)
"linear_num_key_heads": 16, "linear_num_value_heads": 16,
"linear_key_head_dim": 128, "linear_value_head_dim": 128,        # Gated DeltaNet layers
"linear_conv_kernel_dim": 4, "mamba_ssm_dtype": "float32"
```

The model card says it more directly: `Hidden Layout: 6 × (3 × (Gated DeltaNet → FFN) → 1 × (Gated Attention → FFN))`. The last part of `04_hybrid_lm.py` uses this configuration to calculate the inference cache of one sequence. KV uses BF16, and the recurrent state uses FP32, as the config says:

| Context | KV cache (6 full-attention layers) | Linear state (18 Gated DeltaNet layers) | Total | If all 24 layers were full attention |
|---:|---:|---:|---:|---:|
| 4,096 | 48 MiB | 18.6 MiB | 67 MiB | 192 MiB |
| 32,768 | 384 MiB | 18.6 MiB | 403 MiB | 1,536 MiB |
| 262,144 | 3,072 MiB | 18.6 MiB | 3,091 MiB | 12,288 MiB |

Only the 6 full-attention layers have a KV cache. The states of the other 18 Gated DeltaNet layers total 18.6 MiB, independent of the context length. The longer the context, the closer the saving gets to 4 times (the theoretical limit for 3:1). Note that "24 layers of full attention" is a hypothetical baseline: we replace the linear layers with full-attention layers of the same configuration. It is not a real model.

### 8.2 Small experiment 1: one small language model, four architectures

`04_hybrid_lm.py` uses the same character-level Shakespeare corpus as Chapter 10. It trains four small 4-layer models (width 128, 4 heads, SwiGLU FFN, the same data order and hyperparameters, 800 steps each). Only the **token mixer** of each layer changes. (The token mixer is the part of a layer that mixes information between tokens.) A = softmax attention (with RoPE), L = naive linear attention, G = Gated DeltaNet. L and G both have the short convolution and the output normalization.

> **Note:** The numbers of the training experiments in this chapter come from one CPU run on the build machine of the course. Different machines and different versions of the low-level math libraries do floating-point operations in a slightly different order. After a few hundred training steps, these small differences become larger. Your numbers can differ from the second or third decimal place. Trust the conclusions below that do not depend on exact values. For a rerun on another server in 2026-10, see [runs/2026-10-01-gpu0-check/chapters-21-23.md](../../runs/2026-10-01-gpu0-check/chapters-21-23.md).

| Architecture | Parameters | Validation loss (nats/char) | Rerun on another server (2026-10) | Inference cache, T=1,024 | T=65,536 |
|---|---:|---:|---:|---:|---:|
| AAAA (pure attention) | 861,440 | 1.685 | 1.683 | 2,048 KB | 131,072 KB |
| LLLL (pure naive linear) | 867,712 | 1.754 | 1.760 | 73 KB | 73 KB |
| GGGG (pure Gated DeltaNet) | 871,872 | 1.661 | 1.648 | 73 KB | 73 KB |
| GGGA (3:1 hybrid) | 869,264 | 1.649 | 1.652 | 567 KB | 32,823 KB |

(The cache uses "KV in BF16, linear state in FP32"; see `cache_bytes`.)

- Naive linear attention is the worst (1.754). A state that can only add is a disadvantage even in character-level modeling.
- At this scale, Gated DeltaNet is even a little better than pure attention (1.661 vs 1.685). This does not show that it is "stronger than attention". With 800 steps, 0.87M parameters, and a context of 128 characters, the model learns mainly local spelling. The short convolution + gated recurrence happens to be very good at such local patterns.
- The 3:1 hybrid is almost as low as pure Gated DeltaNet (1.649 vs 1.661). In the rerun on another server, it was 1.652 vs 1.648, so the order changed. The difference is within the random variation of a single seed. Compared with pure attention, the advantage of the hybrid is a cache of only about 1/4 (closer to 1/4 for longer T). Its real advantage over a pure linear model is exact recall, and the language-modeling loss does not show it. See the next experiment.

**Limits of this result**: We used a single random seed. The differences between the four models are at most 0.1 nats, and only 0.04 among the top three. In Chapter 10, we saw seed variation of the same order of magnitude. Thus we cannot make a reliable ranking from them. The result shows only one thing: with the same number of parameters, linear layers in most positions **did not make language modeling clearly worse**, and they saved most of the cache. To see the real weakness of linear layers, we need a special task: the next experiment.

### 8.3 Small experiment 2: associative recall — pure linear falls behind, the hybrid recovers

The loss of a language model is not very sensitive to "exact recall": the local context is enough to guess most characters. `05_associative_recall.py` measures recall directly. The first half of the sequence has N random "key value" pairs. The second half gives some of these keys again and again, and the model must give the value of each key. This is a small version of the MQAR task of Zoology.

We use four small 2-layer models (width 64, 4 heads). The head_dim of the linear layers is small on purpose, 16, so the state capacity is clearly too small. For a fair comparison, the q/k/v of the attention layers also have the same short convolution. We train each model for 600 steps, with a random N ∈ [4, 24] at each step. We test 256 sequences for each N. A random guess has an accuracy of 1/64:

| Architecture | N = 4 | N = 8 | N = 12 | N = 16 | N = 20 | N = 24 | Numbers stored per layer at inference |
|---|---:|---:|---:|---:|---:|---:|---|
| AA (pure attention) | 100% | 99% | 97% | 97% | 96% | 94% | KV 8192 + KV 8192 |
| LL (pure naive linear) | 64% | 40% | 30% | 24% | 19% | 17% | state 1024 + state 1024 |
| GG (pure Gated DeltaNet) | 85% | 62% | 48% | 39% | 32% | 29% | state 1024 + state 1024 |
| GA (1 GDN layer + 1 full-attention layer) | 99% | 96% | 91% | 86% | 83% | 79% | state 1024 + KV 8192 |

We draw three conclusions. They agree with the capacity experiments of Sections 5–7:

- **The accuracy of the pure linear models decreases monotonically with N.** The more key-value pairs there are, the less the fixed-size state can hold. Naive linear attention has only 17% at N = 24.
- **Gated DeltaNet is much stronger than naive linear attention** (12 to 22 percentage points higher at each N). Overwriting and gating really make better use of the limited state. But Gated DeltaNet is still far below attention.
- **Change only one of the two layers to full attention, and most of the recall ability comes back** (29% → 79% at N = 24). The cache of this hybrid model is only a little more than half of the cache of pure attention.

**Limits of this result**: This is a very small experiment: 2 layers, 600 steps, a single random seed, and a linear-layer state that we made small on purpose. The advantage of pure attention at large N changes with the number of training steps and the model size. With longer training, the gap between the hybrid model and pure attention (79% vs 94%) can become smaller, or it can stay the same. We did not test this. For comparisons of real models, see the recall evaluations in the Zoology, Gated DeltaNet, and Kimi Linear papers.

### 8.4 Why "a few full-attention layers + many linear layers"

Put the results above together:

- **The linear layers give "cheap depth".** They process local patterns, grammar, and meaning that accumulates step by step. Their memory and the compute of each step do not depend on the length.
- **A few full-attention layers do the "exact recall".** They find a name, a number, or a code variable unchanged from far back in the context. A fixed-size state is worst at exactly this task.
- **The KV cache grows only with the number of full-attention layers.** At 3:1, it is about 1/4 of pure attention; at 9:1, it is about 1/10.

There is no standard ratio. Qwen3.5, Kimi Linear, and Ling-3.0 use 3:1. IBM Granite 4.0-H uses 9:1. NVIDIA Nemotron 3 Nano has only 6 attention layers in 52 layers. Falcon-H1 puts attention heads and Mamba-2 heads in parallel in the same layer. Each model chooses its own point between "how much KV cache it saves" and "how much recall ability it keeps".

There is also a clear counterexample. MiniMax-Text-01 (early 2025) used a 7:1 hybrid of lightning attention (a type of linear attention) and softmax attention. In its config, the `attn_type_list` of the 80 layers has 1 softmax layer in every 8 layers. But its next generation, MiniMax-M2, went back to **full attention in every layer** (all values in the `attn_type_list` of MiniMax-M2.5 are 1). The official MiniMax blog gives these reasons:

- On complex tasks such as code, math, agents, long-chain reasoning, and reinforcement learning, the quality of linear/sparse attention is not yet stable enough.
- The training and inference infrastructure for linear attention is not mature, and memory bandwidth limits many implementations.
- Linear attention is more sensitive to numerical precision, and it is difficult to store the state in low precision.
- How linear attention works with speculative decoding is still an open problem.

Thus the hybrid architecture is an **engineering trade-off** that several companies use. It does not come for free.

## 9. Summary

- Softmax attention normalizes a full row of scores, so it must keep all K and V: KV cache ∝ T.
- Remove the softmax and change the order of the matrix products, and attention becomes an RNN: `S_t = S_{t−1} + v_t k_tᵀ`, `o_t = S_t q_t`. The state has a fixed size.
- Training uses the **chunkwise form** (parallel in a chunk, the state passes between chunks). Decode uses the **recurrent form**. The two forms are mathematically equivalent.
- A fixed-size state becomes full: exact recall is the largest weakness of linear attention.
- The **decay gate** α_t fades out old information (the RetNet / GLA / Mamba-2 family). The **delta rule** reads before it writes and writes only the difference, so it can "overwrite". Multiply the two, and you get **Gated DeltaNet**.
- **Hybrid architecture**: many linear layers + a few full-attention layers (3:1 in Qwen3.5). The KV cache grows only with the full-attention layers, and the full-attention layers make sure that exact recall still works.

---

## GPU measurements (one RTX 3090)

> **Note:** The numbers in the main text above all come from CPU runs. This section uses one NVIDIA GeForce RTX 3090: 24 GB of GPU memory, Ampere architecture. Spec sheet: dense BF16 tensor-core peak about 71 TFLOPS, FP32 about 35.6 TFLOPS, GPU memory bandwidth about 936 GB/s. Environment: PyTorch 2.11.0+cu128, CUDA 12.8, October 2026. The server sets the power limit of this card to 240 W (the factory default is 350 W). Under a continuous full load, the card decreases its clock speed. Thus the absolute compute and bandwidth are lower than on a 3090 at full power, and the relative values are more reliable. If you have no GPU, skip this section.

Run:

```bash
uv run python chapters/23-linear-attention-hybrid/code/06_gpu_linear_vs_softmax.py
```

The shapes are those of the Gated DeltaNet layer of Qwen3.5-0.8B: batch 1, 16 heads, d_k = d_v = 128. Softmax attention also uses 16 × 128. For linear attention, the script calls the code of this chapter without changes: `linear_chunked` from 04, and `gated_delta_chunked` and `gated_delta_recurrent` from 03. It only puts the calls in `with torch.device("cuda")`, so the tensors that the functions create are also on the GPU. The linear attention is pure PyTorch in FP32, with chunk length 64 and a Python loop over the chunks. Softmax attention uses the FlashAttention kernel of PyTorch (BF16). First, we make sure that the math did not change on the GPU. At T = 1,024, the maximum output difference between the chunkwise and recurrent forms of Gated DeltaNet is 6.0e-07. The difference of the final states is 4.8e-07.

Process all T tokens at once (training / prefill; ms, median; in parentheses: the peak extra GPU memory on top of the inputs):

| Sequence length T | Softmax (FlashAttention) | Linear attention, chunkwise | Gated DeltaNet, chunkwise | Gated DeltaNet, recurrent |
|---:|---:|---:|---:|---:|
| 1,024 | 0.2 (4 MiB) | 2.9 (17 MiB) | 10.7 (19 MiB) | 198.8 |
| 4,096 | 1.3 (16 MiB) | 11.2 (65 MiB) | 43.1 (67 MiB) | 813.5 |
| 16,384 | 21.9 (65 MiB) | 48.8 (257 MiB) | 193.5 (259 MiB) | — |
| 65,536 | 460.1 (260 MiB) | 180.7 (1,025 MiB) | 757.5 (1,027 MiB) | — |

(The token-by-token recurrence is too slow, so we measured only the first two rows.)

One decode step: a context of T tokens exists, and 1 more token comes. For Gated DeltaNet, the script first uses the chunkwise form to really compress the T tokens into the state. Then it does one recurrent step from this state (ms, median of 50 runs):

| Context T | Softmax KV cache | Softmax step | Gated DeltaNet state | Gated DeltaNet step |
|---:|---:|---:|---:|---:|
| 1,024 | 8 MiB | 0.053 | (1, 16, 128, 128), 1 MiB | 0.259 |
| 16,384 | 128 MiB | 0.217 | (1, 16, 128, 128), 1 MiB | 0.245 |
| 65,536 | 512 MiB | 0.705 | (1, 16, 128, 128), 1 MiB | 0.242 |
| 262,144 | 2,048 MiB | 3.408 | (1, 16, 128, 128), 1 MiB | 0.306 |

The second table shows the two tables of Sections 1 and 2 on a GPU. The KV cache of softmax attention grows with the context to 2 GiB. Each step must read all of it, so the time grows from 0.053 ms to 3.4 ms. The state of Gated DeltaNet is the same 1 MiB tensor (1, 16, 128, 128) from start to end. One step takes 0.24–0.31 ms, independent of the context length. At 16K tokens, the two are about the same (0.217 vs 0.245 ms). At 65K tokens, the softmax step takes 2.9 times as long, and at 262K tokens, 11 times as long.

The first table confirms the rule of Section 3: "use the chunkwise form for training and prefill". At T = 4,096, the token-by-token recurrence takes 814 ms, but the chunkwise form takes only 43 ms. That is about 19 times faster, a larger gap than on the CPU (7 times in Section 3). The time of the chunkwise form grows linearly with T. The time of FlashAttention grows as T² (21 times from 16K to 64K). At T = 65,536, the chunkwise form of naive linear attention (181 ms) is already faster than FlashAttention (460 ms).

The surprise is the large gap on short sequences. At T = 1,024, the Gated DeltaNet chunkwise form of this chapter is tens of times slower than FlashAttention (10.7 vs 0.2 ms). One decode step also takes 5 times as long as the softmax step at a 1K context. Almost all the time goes to the fixed overhead of the Python loop: about 0.7 ms per chunk for the Gated DeltaNet chunkwise form and about 0.18 ms per chunk for naive linear attention, independent of T. Each chunk launches more than ten small kernels, and Gated DeltaNet also does one triangular solve. The GPU itself is not busy. This is why production code uses the Triton kernels of flash-linear-attention, which fuse the full loop into one kernel. (fla is not installed on this machine, so we have no comparison.)

## From minimal code to production code

The production code is in `zero/arch/linear_attention.py`. It is an experimental module of Part 5 and is **not used in the main-line model**. Its structure and parameter names match the Qwen3.5 implementation of Hugging Face transformers (`Qwen3_5GatedDeltaNet` in `transformers/models/qwen3_5/modeling_qwen3_5.py`). Thus you can load weights directly from one into the other.

| Minimal code (`code/`) | Production code (`zero/arch/linear_attention.py`) | What it adds, and why |
|---|---|---|
| `linear_attention_recurrent` in `01`, `recurrent` in `02` | `recurrent_linear_attention(q, k, v, g, initial_state)` | Batched and multi-head `(B, H, T, D)`. It can continue from the last state (decode needs this). It uses float32 internally |
| `chunked` in `02` (T must be a multiple of C) | `chunk_linear_attention(..., chunk_size, initial_state)` | It pads with zeros automatically (padded positions have k = 0 and g = 0, so they do not change the state). It takes an initial state, so prefill can run in segments |
| `gated_delta_recurrent` / `gated_delta_chunked` in `03` | `recurrent_gated_delta_rule` / `chunk_gated_delta_rule` | Same as above. The state layout changes to the HF / fla layout `(d_k, d_v)` (the minimal code uses `(d_v, d_k)`; it is only a transpose) |
| `LinearMixer` in `04` (q/k/v projections, short convolution, L2 normalization, RMSNorm per head) | `LinearAttention`, `GatedDeltaNet` (both use `_LinearMixer`) | The parameter names are the same as in HF (`in_proj_qkv`, `in_proj_z`, `in_proj_b`, `in_proj_a`, `conv1d`, `A_log`, `dt_bias`, `norm`, `out_proj`). **Gated RMSNorm** (the output is multiplied by SiLU(z)). The number of value heads can be an integer multiple of the number of q/k heads. Mamba-2 style initialization of A and dt. The short convolution has a cache (it keeps the last K−1 inputs), so input in segments gives the same result as input in one pass. With `mode="auto"`, T = 1 uses the recurrent form, and other lengths use the chunkwise form |
| `TinyLM(pattern="GGGA")` in `04` | `HybridConfig`, `HybridTransformer`, `hybrid_layer_types(n_layers, full_attention_interval=4)` | Driven by the configuration: `layer_types` has the same name and meaning as in the HF Qwen3.5 config. The full-attention layers reuse `zero.model.Attention` of the main line (GQA + QK-Norm + RoPE + SDPA) |
| None | `HybridCache`: the full-attention layers use a preallocated `KVCache` (allocated only for the full-attention layers), and the linear layers use `LinearState` (recurrent state + convolution tail); `generate_greedy` | The two cache types exist together at inference. `cache_bytes_per_sequence` counts layer by layer (an extension of the ledger of Chapter 21) |
| `cache_bytes`, `qwen35_cache_mib` in `04` | `cache_bytes_per_sequence`; also `zero/tools/kv_cache_calc.py` of Chapter 21 (`layout_from_config` reads `layer_types` from the HF config, or the `linear_attn_config` of Kimi, directly, and it counts each linear layer as a fixed state) | It reads a real config and calculates the cache of any hybrid model. You do not copy the layer counts by hand |
| None | Not implemented: the **output gate** of the Qwen3.5 full-attention layers (gated attention, `attn_output_gate: true`), partial RoPE (`partial_rotary_factor: 0.25`), MoE, MTP | This chapter is only about how to mix token mixers. Chapters 24 and 25, and the overview in Chapter 26, discuss these parts |

**Parity check**: `tests/test_arch_linear_attention.py` (`uv run pytest tests/test_arch_linear_attention.py`; all 33 tests passed on this machine in about 10 s). The tests make sure of these points:

- For linear attention (with and without decay) and the gated delta rule (with and without decay): **chunkwise form == recurrent form**. The tests use chunk lengths 1, 5, 8, 16, 64, lengths that are not a multiple of the chunk length, and a random initial state.
- Linear attention without decay == the masked parallel form `(QKᵀ ⊙ M) V`.
- The gated delta rule == a naive reference that uses the explicit matrix `S_t = α_t (I − β_t k_t k_tᵀ) S_{t−1} + β_t k_t v_tᵀ` at each step. When we write the same key two times, the delta rule overwrites and linear attention adds.
- `LinearAttention` / `GatedDeltaNet` layers: the chunkwise form, the recurrent form, and input in three segments with state (10 + 1 + 12 tokens) give the same output. The state shape does not depend on the length.
- **Parity check with `Qwen3_5GatedDeltaNet` of HF transformers**: randomize all weights of the HF layer, then load them with `load_state_dict(strict=True)`. The outputs agree within 1e-4.
- `HybridTransformer` (interval 2, 4, and 99, that is, 1:1, 3:1, and pure linear; both Gated DeltaNet and naive linear): **generation with the state cache == full recalculation at each step** (30 greedy tokens are exactly the same). Chunked prefill (20 + 1 + 24) == one forward pass. The KV bytes double when the length doubles, and the linear state does not change. The loss decreases in a few training steps (the gradient flows through the triangular solve).

**For real training and deployment**, the pure PyTorch chunk loop is too slow. The industry uses these tools:

- **flash-linear-attention (fla-org)**: a library of linear-attention kernels written in Triton. `fla.ops.gated_delta_rule.chunk_gated_delta_rule` / `fused_recurrent_gated_delta_rule` are the GPU implementations of the two forms of this chapter. The KDA kernel of Kimi Linear (`fla.ops.kda`) is also open source there. When fla and causal-conv1d are installed, the Qwen3.5 implementation of HF transformers automatically uses these kernels. Otherwise, it falls back to a pure PyTorch version with the same structure as this chapter.
- **vLLM**: `vllm/model_executor/models/qwen3_next.py`, `qwen3_5.py`, and other files support these hybrid models. Its hybrid KV cache manager (see the Hybrid KV Cache Manager design document) gives different cache types to different layer types. Full-attention layers get KV pages by the number of tokens. Mamba / linear layers get a fixed-size state for each request.

We verified the forward pass, the backward pass, and generation of `zero/arch/linear_attention.py` with CUDA on an RTX 3090. During this check, we also fixed a bug: `generate_greedy` created the input on the CPU. In BF16, the gradients of Gated DeltaNet differ from FP32 by about 20% (relative). Thus the pure PyTorch chunkwise implementation is not accurate enough in low precision. See Sections 11 and 12 of [runs/2026-10-01-gpu0-check](../../runs/2026-10-01-gpu0-check/README.md). The CUDA kernels of fla and causal-conv1d are not installed, so we did not verify them.

---

## Frontier notes

> **Techniques that are not a consensus yet. We mention them only here.**
>
> - **Pure linear / pure state space models** (pure Mamba, RWKV, RetNet, and others): they show that a linear recurrence alone can support a language model. But the main versions of the leading open model families do not use a pure linear architecture. All of them use hybrids. (Section 2.1 of GOAL.md lists these models as "mention in one sentence in the frontier notes".)
> - **KDA (Kimi Delta Attention)**: an improved Gated DeltaNet from Kimi Linear. It replaces the scalar decay gate of each head with a fine-grained gate for each channel (one for each key dimension). Ling-3.0 of Ant Group also uses it. At the moment, only these two companies use it, and Ling says clearly that it follows the design ideas of Kimi Linear. Thus it is not yet an independent consensus of several companies.
> - **Parallel hybrid** (Falcon-H1): attention heads and Mamba-2 heads run in parallel in the same layer, and the layer concatenates their outputs. The layer types do not alternate. At the moment, only Falcon-H1 does this.
> - **The ratio debate**: leading models use 3:1, 9:1, and even "full attention" (MiniMax-M2). There is no consensus yet on the best ratio, or on which layers should be full attention.
> - **Sparse attention** (the DeepSeek DSA/NSA type; MiniMax-M3 also uses block-sparse attention) is another way to make long context cheaper. See Chapter 22.

## Adopters and sources

| Technique | Adopters (main versions) | Sources |
|---|---|---|
| Hybrid linear attention: DeltaNet-family linear layers + a few full-attention layers | **Qwen3-Next-80B-A3B** (48 layers = 12 × [3 × Gated DeltaNet + 1 × Gated Attention]); **the full Qwen3.5 series** (the configs of 0.8B, 2B, 4B, 9B, 27B, 35B-A3B, 122B-A10B, and 397B-A17B all have `full_attention_interval: 4`); **Kimi Linear 48B-A3B** (KDA : MLA = 3:1, 7 MLA layers in 27 layers); **Ant Group Ling-3.0-tiny** (groups of 3 KDA layers + 1 MLA layer) | Model cards and config.json: [Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B), [Qwen3.5-27B config](https://huggingface.co/Qwen/Qwen3.5-27B/blob/main/config.json), [Qwen3.5-397B-A17B config](https://huggingface.co/Qwen/Qwen3.5-397B-A17B/blob/main/config.json), [Qwen3-Next-80B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct), [Kimi-Linear-48B-A3B-Instruct](https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct) (technical report arXiv:2510.26692), [Ling-3.0-tiny](https://huggingface.co/inclusionAI/Ling-3.0-tiny) |
| Hybrid state space: Mamba-2 layers + a few attention layers | **NVIDIA Nemotron-H**, **Nemotron 3** (Nano 30B-A3B: 23 Mamba-2 layers, 23 MoE layers, and 6 attention layers in 52 layers; Ultra 550B-A55B also has the `nemotron_h` architecture); **IBM Granite 4.0-H** (H-Small: 4 attention layers in 40 layers; IBM calls it Mamba-2 : Transformer = 9:1); **TII Falcon-H1** (attention and Mamba-2 in parallel in each layer) | [Nemotron-H technical report arXiv:2504.03624](https://arxiv.org/abs/2504.03624), [Nemotron 3 Nano technical report arXiv:2512.20848](https://arxiv.org/abs/2512.20848) and [model card](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16) (`hybrid_override_pattern`); [granite-4.0-h-small config](https://huggingface.co/ibm-granite/granite-4.0-h-small/blob/main/config.json) and [IBM announcement](https://www.ibm.com/new/announcements/ibm-granite-4-0-hyper-efficient-high-performance-hybrid-models); [Falcon-H1 technical report arXiv:2507.22448](https://arxiv.org/abs/2507.22448) |
| Linear attention (lightning attention) + softmax at 7:1 (later abandoned) | **MiniMax-Text-01 / M1** (80 layers, 1 softmax layer in every 8); **MiniMax-M2 / M2.5 went back to full attention** | [MiniMax-01 technical report arXiv:2501.08313](https://arxiv.org/abs/2501.08313), [MiniMax-Text-01 config](https://huggingface.co/MiniMaxAI/MiniMax-Text-01/blob/main/config.json), [MiniMax-M2.5 config](https://huggingface.co/MiniMaxAI/MiniMax-M2.5/blob/main/config.json), official blog [Why Did M2 End Up as a Full Attention Model?](https://www.minimax.io/news/why-did-m2-end-up-as-a-full-attention-model) |
| Kernels for the chunkwise + recurrent forms | flash-linear-attention (the Qwen3.5 / Qwen3-Next implementations of HF transformers call it; the KDA kernel of Kimi is open source there); the hybrid cache management of vLLM | <https://github.com/fla-org/flash-linear-attention>; vLLM [Hybrid KV Cache Manager](https://github.com/vllm-project/vllm/blob/main/docs/design/hybrid_kv_cache_manager.md) |

We count with rule A of GOAL.md 2.1. Three leading families on the list clearly use **the hybrid architecture "a few full-attention layers + many linear/state space layers"** in their main versions: Qwen, Kimi, and NVIDIA Nemotron. IBM Granite, Falcon, and Ant Group's Ling also use it. This meets the consensus condition, so the hybrid architecture is in the main text. Only Qwen uses **Gated DeltaNet itself** directly (Kimi's KDA and Ling use improved versions of it). Thus this chapter uses Gated DeltaNet as the representative of "DeltaNet-family linear layers", and it puts KDA in the frontier notes.

Note: we read the layer counts and ratios of all models from the config.json files and model cards on Hugging Face (read in 2026-09). At the time of writing, we found no separate arXiv version of a Qwen3.5 technical report. We use the official blog and the model cards (**to be verified**: is there a formal technical report?).

---

## Guided questions

Ask Claude Code these questions. Continue until you can explain the answers in your own words:

1. In the recurrent form of linear attention, the state S has the shape d_v × d_k. Suppose that we change all H heads of multi-head attention to linear attention. How large is the state of the full layer? Compare it with the KV cache of the same H heads at length T. At which T are the two equal?
2. Why does the delta rule need L2 normalization of k? If ‖k‖ = 2 and β = 1, what is S k after the write? Can it diverge? (Hint: look at the eigenvalues of I − β k kᵀ.)
3. This chapter says that the delta rule "does one step of gradient descent on ‖S k − v‖²". Calculate the gradient of this loss with respect to S yourself, and verify that it agrees with the update of the delta rule. In this view, what does the "decay gate" correspond to? (Hint: weight decay.)
4. Training uses the chunkwise form, and inference uses the recurrent form. If the chunk length in training is 1, what does the chunkwise form become? What if the chunk length is the full sequence length? How should you choose the chunk length?
5. The linear layers of Qwen3.5 do not use RoPE. The full-attention layers use RoPE on only 25% of head_dim (`partial_rotary_factor: 0.25`). Where do the linear layers get their position information from?
6. MiniMax-M2 went back to full attention. Suppose that you choose an architecture for a 0.6B small model that mainly calls tools (the main-line model of this course). Do you use a 3:1 hybrid or full attention? List your reasons and the comparison experiments that you must do.

## Hands-on tasks

For each task, run the code and look at the result.

**Task 1 (basic)**: In the capacity experiment of `03_delta_rule.py`, change the keys to **pairwise orthogonal** vectors (for example, the rows of an orthogonal matrix from `torch.linalg.qr`). Use N = 16, 32, 64, and 65. What are the read errors of linear attention and of the delta rule? Why is there a sudden change between N = 64 and N = 65?

**Task 2 (core)**: In `05_associative_recall.py`, add an architecture "GAGG" (full attention in layer 2, not in the last layer). Compare its recall accuracy with "GGGA". Then double the head_dim of the linear layers (the state becomes 4 times larger). How much of the gap can pure Gated DeltaNet close?

**Task 3 (challenge)**: Use `HybridTransformer` from `zero/arch/linear_attention.py` to build a 3:1 hybrid model of the same size as in `04_hybrid_lm.py`. Train it for 800 steps on the same corpus and compare the validation loss. Then generate with the cache from `model.new_cache()`. When you generate 2000 characters, measure how much of `HybridCache.nbytes()` is KV and how much is linear state. Compare the result with the formula of `cache_bytes_per_sequence`.

---

## Go deeper: CS336

This chapter matches Stanford CS336 (Spring 2026) <https://cs336.stanford.edu/>:

- **Lecture 4: Alternatives to attention, and MoE.** The slides and recordings are on the course page. The lecture discusses why we look for alternatives to softmax attention. It also gives the basic ideas of directions such as linear attention and state space models. The derivation of this chapter (associativity → recurrence → chunkwise form) matches the linear-attention part of this lecture.
- **CS336 does not go deep into these topics**: the delta rule, the chunkwise algorithm (UT transform) of Gated DeltaNet, and the exact configurations of industrial hybrid architectures (Qwen3.5 / Kimi Linear / Nemotron). See the references of this chapter, especially the DeltaNet and Gated DeltaNet papers by Songlin Yang et al., and the flash-linear-attention repository.

---

## References

- Katharopoulos, Vyas, Pappas, Fleuret. *Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention*, 2020: <https://arxiv.org/abs/2006.16236>
- Schlag, Irie, Schmidhuber. *Linear Transformers Are Secretly Fast Weight Programmers* (the delta rule for linear attention), 2021: <https://arxiv.org/abs/2102.11174>
- Yang, Wang, Zhang, Shen, Kim. *Parallelizing Linear Transformers with the Delta Rule over Sequence Length* (the chunkwise parallel algorithm of DeltaNet), 2024: <https://arxiv.org/abs/2406.06484>
- Yang, Kautz, Hatamizadeh. *Gated Delta Networks: Improving Mamba2 with Delta Rule*, 2024: <https://arxiv.org/abs/2412.06464>
- Yang, Wang, Shen, Panda, Kim. *Gated Linear Attention Transformers with Hardware-Efficient Training* (GLA), 2023: <https://arxiv.org/abs/2312.06635>
- Dao, Gu. *Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality* (Mamba-2), 2024: <https://arxiv.org/abs/2405.21060>
- Sun et al. *Retentive Network: A Successor to Transformer for Large Language Models* (RetNet), 2023: <https://arxiv.org/abs/2307.08621>
- Arora et al. *Zoology: Measuring and Improving Recall in Efficient Language Models* (the MQAR associative recall task), 2023: <https://arxiv.org/abs/2312.04927>
- Qiu et al. *Gated Attention for Large Language Models: Non-linearity, Sparsity, and Attention-Sink-Free* (the gated attention of Qwen), 2025: <https://arxiv.org/abs/2505.06708>
- Kimi Team. *Kimi Linear: An Expressive, Efficient Attention Architecture*, 2025: <https://arxiv.org/abs/2510.26692>
- MiniMax. *MiniMax-01: Scaling Foundation Models with Lightning Attention*, 2025: <https://arxiv.org/abs/2501.08313>; *Why Did M2 End Up as a Full Attention Model?*: <https://www.minimax.io/news/why-did-m2-end-up-as-a-full-attention-model>
- NVIDIA. *Nemotron-H: A Family of Accurate and Efficient Hybrid Mamba-Transformer Models*, 2025: <https://arxiv.org/abs/2504.03624>; *Nemotron 3 Nano*, 2025: <https://arxiv.org/abs/2512.20848>
- TII. *Falcon-H1: A Family of Hybrid-Head Language Models Redefining Efficiency and Performance*, 2025: <https://arxiv.org/abs/2507.22448>
- IBM. *IBM Granite 4.0: hyper-efficient, high performance hybrid models for enterprise*: <https://www.ibm.com/new/announcements/ibm-granite-4-0-hyper-efficient-high-performance-hybrid-models>
- Qwen Team. Official blogs and model cards of Qwen3-Next and Qwen3.5: <https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct>, <https://huggingface.co/Qwen/Qwen3.5-0.8B>
- Model card of Ling-3.0-tiny (Ant Group, Bailing): <https://huggingface.co/inclusionAI/Ling-3.0-tiny> (the survey report in `small-llms-under-5b-2026-08-30/` of this repository also mentions it)
- flash-linear-attention: <https://github.com/fla-org/flash-linear-attention>
- The Qwen3.5 implementation of Hugging Face transformers: <https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/modeling_qwen3_5.py>
- vLLM Hybrid KV Cache Manager design document: <https://github.com/vllm-project/vllm/blob/main/docs/design/hybrid_kv_cache_manager.md>
- CS336: <https://cs336.stanford.edu/>

**Next chapter**: This chapter made the "memory" of each token cheap: most layers need only a fixed-size state. But the FFN compute for each token did not become smaller, and the FFN has most of the parameters and compute of the model. Can a model have very many parameters but use only a small part of them for each token? Chapter 24: mixture of experts (MoE).
