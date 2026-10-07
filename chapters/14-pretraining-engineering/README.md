# Chapter 14: Pretraining engineering — Mixed precision, FlashAttention, data parallelism, and resume from checkpoints

**English** · [中文](README.zh.md)

> **Goal**: After this chapter, you can calculate three budgets for one training step of the main-line model: compute, memory, and time. You can explain which problem each of these methods solves: BF16 mixed precision, the tiles + online softmax of FlashAttention, DDP, and FSDP. You can also verify by yourself that two methods give results that are bit-identical to an uninterrupted single-process run: "2 processes each calculate one half, then average the gradients", and "resume after a crash".

📺 **Video**: Not published yet. To render it on your computer, run `bash chapters/14-pretraining-engineering/video/build.sh`.
🧪 **Self-check**: After the chapter, type `/ch14-pretraining` in Claude Code.

---

In the last chapter, we prepared the data and the tokenizer. Chapter 12 set the size of the main-line model: 689.5M parameters and about 400B tokens of pretraining. Chapter 6 showed how to make training stable in the mathematical sense (initialization, RMSNorm, AdamW, warmup, gradient clipping). This chapter answers a different question: **how can the same training loop run on 8 GPUs for 10 days or more, with no failures and no wasted money?**

This is not a mathematical problem any more. It is an engineering problem. A short calculation shows its size. The main-line model needs 6.96 billion floating-point operations to train one token, and 400B tokens need 2.8 × 10²¹ operations in total. On 8 H100 GPUs at 40% efficiency, this takes 10.2 days and costs about $4,900. This is just below the $5,000 limit for pretraining in GOAL.md. If the efficiency drops from 40% to 30%, the cost increases by more than $1,600, and the run goes over the limit. Each technique in this chapter does one of these four things:

- It makes the GPU calculate faster (mixed precision, FlashAttention).
- It makes the model fit in GPU memory (activation checkpointing, FSDP).
- It lets many GPUs work together (data parallelism).
- It keeps a run of 10 days or more useful when a problem occurs (loss-spike handling, resume from checkpoints).

> **Note:** The code in the main text runs on a CPU. The H100 numbers in the main text (throughput, MFU, memory) are still **estimates from formulas**. The new section "GPU measurements (one RTX 3090)" measures some of them on a real GPU. For an item-by-item check of the GPU code paths of zero on one and on two RTX 3090 GPUs, see [runs/2026-10-01-gpu0-check](../../runs/2026-10-01-gpu0-check/README.md). A verification run on 8×H100 that costs at most $50 is still the first task of step 2 (see "Main-line progress").

## 1. Do the calculation first: the cost of one pretraining step

Chapter 12 derived the number of floating-point operations (FLOPs) to train one token (forward pass + backward pass):

```
FLOPs per token = 6 · N_matmul + 12 · L · q_dim · T
```

The `6N` term comes from this rule: in the forward pass, each parameter does one multiply-add (2 FLOPs). In the backward pass, it does two (4 FLOPs). The second term is for the two matrix multiplications in attention that have no parameters, `QKᵀ` and `AV`. This term is proportional to the sequence length T. Run:

```bash
uv run python chapters/14-pretraining-engineering/code/01_step_cost.py
```

```python
per_layer = d * q_dim + 2 * d * kv_dim + q_dim * d + 3 * d * m["ffn_dim"]
n_matmul = L * per_layer + d * m["vocab_size"]  # lm_head does its matmul, also when it shares weights with the embedding
return 6 * n_matmul + 12 * L * q_dim * T, n_matmul, n_total
```

The main-line model (`configs/main/pretrain.toml`: 28 layers, dim 1280, 16 query heads × 128, 8 KV heads, FFN 3584, vocabulary 65,536, T = 4096) gives these values:

| Quantity | Value |
|---|---:|
| Total parameters N / parameters in matrix multiplications N_matmul | 689.5M / 689.4M |
| FLOPs per token | 6.955 × 10⁹ |
| Tokens per step (micro batch 4 × accumulation 4 × 8 GPUs × 4096) | 524,288 |
| FLOPs per step | 3.647 × 10¹⁵ |
| Total FLOPs for 400B tokens | 2.782 × 10²¹ |
| Total steps (400B / 524,288 rounded up; this is `max_steps` in the configuration) | 762,940 |

The next table converts this into time and money on 8×H100 (dense BF16 peak of the H100 SXM: 989.5 TFLOPS; $2.5 per GPU-hour):

| MFU | 8-GPU throughput (tokens/s) | Seconds per step | Days | GPU-hours | Cost |
|---:|---:|---:|---:|---:|---:|
| 0.3 | 341,442 | 1.54 | 13.56 | 2,603 | $6,508 |
| 0.4 | 455,256 | 1.15 | 10.17 | 1,953 | $4,881 |
| 0.5 | 569,070 | 0.92 | 8.14 | 1,562 | $3,905 |

Chapter 12 set the budget of 400B tokens. The original plan was 500B tokens. At the same MFU of 0.4, that plan takes 12.71 days and costs $6,102. This is more than the approximately $5K for pretraining in GOAL.md 3.4, so we cut the budget to about 400B tokens. (The last two lines of part ② of the script print this comparison and the MFU threshold below which the cost goes over the limit.)

**MFU (Model FLOPs Utilization)** in the table is a metric that this chapter uses many times. Section 6 explains it in detail. For now, remember its typical size: PaLM 540B reports 46.2%, Llama 3 405B reports 38–43%, and Nemotron-4 340B reports 41–42%. For the same 400B tokens, an increase of MFU from 0.3 to 0.5 saves $2,600. If MFU is only a little below 0.4, the budget goes over the $5K limit. The production estimation tool is `zero/tools/estimate_cost.py`. Its output agrees with the MFU 0.4 row of the table above (1,952.5 GPU-hours, 10.17 days, $4,881).

## 2. What is in GPU memory

Compute decides how long the run takes. GPU memory decides if the run can start at all. During one training step, GPU memory holds four types of data:

- **Parameters**: the FP32 master weights, 4 bytes each.
- **Gradients**: the same shape and precision as the parameters, 4 bytes.
- **Optimizer state**: the first moment m and the second moment v of AdamW (Chapter 6), 4 bytes each.
- **Activations**: the intermediate tensors that the forward pass saves for the backward pass. Their size is proportional to micro batch × sequence length.

The first three items add up to the well-known "**16 bytes per parameter**". `05_memory.py` uses the real model of zero on a CPU (tiny shape, 1,049,984 parameters). After one AdamW step, it counts each item:

| | Bytes | Per parameter |
|---|---:|---:|
| Parameters (FP32) | 4,199,936 | 4 |
| Gradients (FP32) | 4,199,936 | 4 |
| AdamW m (FP32) | 4,199,936 | 4 |
| AdamW v (FP32) | 4,199,936 | 4 |
| Total | 16,799,744 | 16 |

How do we count the activations? PyTorch has a hook, `torch.autograd.graph.saved_tensors_hooks`. In the forward pass, each tensor that autograd saves for the backward pass goes through this hook. We count each storage only once and add up the bytes:

```python
def pack(t):
    key = t.untyped_storage().data_ptr()
    if key not in own and key not in seen:        # do not count the parameters; count each memory block only once
        seen[key] = t.untyped_storage().nbytes()
    return t
```

With micro batch 4 × sequence 128 = 512 tokens and 4 layers:

| Case | Saved bytes | Per token | Relative |
|---|---:|---:|---:|
| FP32 | 30,251,012 | 59,084 | 100% |
| BF16 autocast | 24,090,628 | 47,052 | 80% |
| BF16 + activation checkpointing | 6,428,676 | 12,556 | 21% |

**Activation checkpointing** (also called recomputation or rematerialization): in the forward pass, each Transformer block saves only its input and discards all intermediate results inside the block. When the backward pass reaches this block, it uses the saved input to calculate the forward pass of the block again. The cost is one more forward pass, about 1/3 more compute (6N becomes 8N). The gain: the activation memory decreases from "tens of tensors per layer" to "one tensor per layer".

```python
def forward(self, x, cos, sin, kv_cache=None, start_pos=0):
    return checkpoint(self.block, x, cos, sin, use_reentrant=True)   # save only x; calculate the full block again in the backward pass
```

The same calculation for the main-line model (production calculator `zero/tools/memory_calc.py`, 8×H100 80GB, BF16, seq 4096) gives:

| Micro batch | Activations | DDP total per GPU | FSDP total per GPU | DDP + activation checkpointing |
|---:|---:|---:|---:|---:|
| 1 | 11.0 GiB | 26.6 GiB | 14.0 GiB | 16.5 GiB |
| 2 | 21.9 GiB | 39.1 GiB | 26.5 GiB | 19.9 GiB |
| 4 | 43.9 GiB | 64.0 GiB | 51.4 GiB | 26.8 GiB |
| 8 | 87.7 GiB | 113.8 GiB | 101.2 GiB | 40.6 GiB |

The parameters need only 689.5M × 16 bytes = 10.3 GiB. The largest part is the activations: 92,840 bytes per token per layer. With 28 layers and the output head, one token needs about 2.9 MB. **Micro batch 8 does not fit on an 80 GB GPU.** (This is a conservative upper bound in eager mode. `torch.compile` fuses element-wise operations and saves fewer intermediate tensors, but it probably cannot save 30 GB.) The next method solves this problem.

### Gradient accumulation: use time to save memory

Chapter 1 showed that PyTorch **adds** each new gradient into `.grad` by default. Thus each step must start with `zero_grad()`. Gradient accumulation uses this property on purpose. Run k micro batches, one after the other. Divide the loss of each micro batch by k before `backward`. The gradients then add up to the mean gradient of a large batch. Only after that, call `optimizer.step()` and `zero_grad()`.

```python
for xs, ys in zip(x.chunk(accum), y.chunk(accum)):
    loss = loss_fn(model, xs, ys) / accum    # divide by the accumulation steps: the sum = mean gradient of the large batch
    loss.backward()                          # the gradient is added into .grad
opt.step()
opt.zero_grad()                              # set to zero only at the end of the step
```

The activation memory depends only on the micro batch, not on the number of accumulation steps. Thus the main-line step of 524,288 tokens can be "micro batch 4 × accumulation 4 × 8 GPUs × 4096". The memory is that of micro batch 4 (64.0 GiB), and the number of tokens per step does not change. `06_ddp_by_hand.py` shows that gradient accumulation gives the same result as one large batch (see the table in Section 5).

## 3. Mixed precision: why BF16

### 3.1 Bit layout

A floating-point number = sign bit + exponent bits + mantissa bits. The exponent bits set the **range** (how large and how small a number can be). The mantissa bits set the **precision** (the distance between two adjacent numbers).

```
FP32: 1 + 8 + 23     BF16: 1 + 8 + 7      FP16: 1 + 5 + 10
```

`02_precision.py` stores π in the three formats:

| Format | Bits of π (sign \| exponent \| mantissa) | Stored value | Maximum | Smallest normal number | epsilon |
|---|---|---:|---:|---:|---:|
| FP32 | `0 \| 10000000 \| 10010010000111111010000` | 3.1415901 | 3.4 × 10³⁸ | 1.18 × 10⁻³⁸ | 1.19 × 10⁻⁷ |
| BF16 | `0 \| 10000000 \| 1001001` | 3.1406250 | 3.39 × 10³⁸ | 1.18 × 10⁻³⁸ | 0.00781 |
| FP16 | `0 \| 10000 \| 1001001000` | 3.1406250 | 65,504 | 6.1 × 10⁻⁵ | 0.000977 |

BF16 (brain floating point) is FP32 with 16 fewer mantissa bits. It has the same number of exponent bits, so **its range is the same as the range of FP32**. Only its precision is lower (epsilon 2⁻⁷ ≈ 0.0078, about 2–3 significant decimal digits). FP16 has 3 more mantissa bits and a better precision. But it has only 5 exponent bits, so its maximum is 65,504.

### 3.2 What occurs when the precision is too low

Add 0.01 to an accumulator 10,000 times. The correct answer is 100:

| Accumulator | Result |
|---|---:|
| FP32 | 100.0030 |
| BF16 | 4.0000 |
| FP16 | 32.0000 |

In BF16, two adjacent numbers in [4, 8) are 4 × 2⁻⁷ = 0.03125 apart. 0.01 is less than half of this gap, so each sum rounds back to 4. The accumulator "gets stuck". The same problem occurs in a weight update. Start with the weight w = 1.0, and subtract 10⁻³ in each of 1000 steps. The correct answer is 0. The BF16 weight stays at **1.0000**, and the FP32 weight is 0.0000. The learning rate times the gradient is usually several orders of magnitude smaller than the weight. A BF16 weight cannot take such a small update.

Thus the standard method of **mixed precision** (Micikevicius et al., 2017) divides the work:

- **Do the matrix multiplications in low precision.** The Tensor Cores of a GPU have a much higher throughput in BF16 than in FP32. The activations also use half the memory.
- **Keep the master weights, the gradient accumulation, and the optimizer state in FP32.** Then small updates are not lost.
- **Calculate the operations that are sensitive to precision (normalization, softmax, loss) in FP32.**

PyTorch uses `torch.autocast` for this division of work. Inside its scope, matrix multiplications change to BF16 automatically, and the parameters do not change. The last experiment of `02_precision.py` gives:

| Output dtype | Parameter dtype | Gradient dtype | Relative error against the FP32 result |
|---|---|---|---:|
| bfloat16 | float32 | float32 | 2.78 × 10⁻³ |

A different method is **stochastic rounding**: round up or down at random, with a probability that depends on the distance. The expected value is then equal to the original number. Again, add 0.01 10,000 times. BF16 + stochastic rounding with 5 seeds gives 95.50, 92.50, 98.00, 100.00, and 103.50, with a mean of 97.90. The accumulator no longer gets stuck, but each result has noise. The main training frameworks still use FP32 master weights. We show stochastic rounding here only as a comparison, to help you understand rounding errors.

### 3.3 Why not FP16

The problem of FP16 is its range. 70,000 stored in FP16 becomes `inf`. 10⁻⁸ stored in FP16 becomes 0 (BF16 stores it as 1.00117 × 10⁻⁸). Convert a batch of small gradients with standard deviation 10⁻⁶ to FP16, and 2.4% of them become 0. In BF16, none of them is lost. Thus FP16 training needs **loss scaling**. Multiply the loss by a large number (for example, 1024) before the backward pass. All gradients become larger and do not underflow. Divide them by the same number before the update. After the multiplication by 1024, the fraction of zeros drops to 0.0%. But the scale factor must change during training. If it is too large, values overflow to inf, and that step must be skipped. BF16 has the same range as FP32 and has none of these problems. Thus almost all large-model training since the A100 uses BF16 (see the adopters at the end of the chapter).

**FP8**: starting with the H100, GPUs support 8-bit floating-point matrix multiplication. In theory, this doubles the throughput again. DeepSeek-V3 trained a 671B MoE model in FP8 with fine-grained quantization. Chapter 12 checks if FP8 already meets the consensus rule of this course. This chapter puts FP8 in "Frontier notes" at the end. The main line uses BF16 by default.

## 4. FlashAttention: attention is slow because of reads and writes, not because of calculation

### 4.1 The problem: memory-bound

A GPU has two types of memory. GPU memory (HBM, 80 GB on the H100) is large but slow. The on-chip cache (SRAM, a few hundred KB per compute unit) is very small but much faster. If an operation does only a little calculation for each byte that it reads, its time goes into moving data. Such an operation is **memory-bound**. The opposite is **compute-bound**.

Naive attention is memory-bound. First, it calculates `S = QKᵀ/√d` (T × T) and writes it to GPU memory. Then it reads S, calculates the softmax P (T × T), and writes P back. Then it reads P again to multiply by V. The softmax itself needs almost no compute, but the two T × T matrices are read and written in full several times. The last part of `04_tiled_attention.py` calculates the scale of the main-line model. With T = 4096, 16 heads, micro batch 8, and BF16, **S and P of one layer are 8.0 GiB**. To keep P for the backward pass in 28 layers needs 112 GiB.

### 4.2 Online softmax: one scan, with corrections during the scan

To avoid writing out the full S, we must calculate the softmax block by block. The difficulty: the softmax must first know the maximum of the full row (the numerically stable method subtracts the maximum before the exponential). But when we work block by block, we have not seen the later blocks yet.

**Online softmax** (Milakov & Gimelshein, 2018) scans only once. It keeps two values "up to now":

```
m_j = max(m_{j−1}, x_j)                                   current maximum
l_j = l_{j−1} · exp(m_{j−1} − m_j) + exp(x_j − m_j)         sum of exponentials, relative to the current maximum
```

When a new maximum occurs, the old sum still uses the old m. Multiply it by `exp(m_old − m_new)` (≤ 1) to correct it to the new reference. After the scan, `softmax_i = exp(x_i − m_N) / l_N`.

```python
for xi in x:
    m_new = max(m, xi)
    l = l * math.exp(m - m_new) + math.exp(xi - m_new)   # correct the old sum to the new maximum
    m = m_new
```

`03_online_softmax.py` prints each step for x = [1, 3, 2, 5, 4]:

| j | x_j | m_j | l_j | What occurs |
|---:|---:|---:|---:|---|
| 1 | 1 | 1 | 1.0000 | |
| 2 | 3 | 3 | 1.1353 | New maximum: multiply the old sum by exp(1 − 3) = 0.1353 |
| 3 | 2 | 3 | 1.5032 | |
| 4 | 5 | 5 | 1.2034 | New maximum: multiply the old sum by exp(3 − 5) = 0.1353 |
| 5 | 4 | 5 | 1.5713 | |

The final softmax is 0.0117, 0.0861, 0.0317, 0.6364, and 0.2341. The maximum difference from the standard three-scan method is 0. With 10,000 random scores (float64, standard deviation 10), the maximum difference is 8.3 × 10⁻¹⁵.

The same method corrects the attention output `Σ softmax_i · v_i` during the scan:

```
o_j = o_{j−1} · exp(m_{j−1} − m_j) + exp(x_j − m_j) · v_j,   final output o_N / l_N
```

With 512 positions and d = 64, the maximum difference from `softmax(scores) @ V` is 5.8 × 10⁻¹⁵.

### 4.3 Tiled attention

**FlashAttention** (Dao et al., 2022) applies this idea to blocks of matrices. It cuts Q into blocks of Br rows, and K and V into blocks of Bc rows. The outer loop takes one block of Q. The inner loop takes the blocks of K and V one by one. In the on-chip cache, it calculates a small Br × Bc block of scores and uses online softmax to update (m, l, O) of each row. The full S and P are never written to GPU memory.

```python
for i0 in range(0, T, br):                       # outer loop: one block of Q (Br rows)
    ...
    for j0 in range(0, T, bc):                   # inner loop: one block of K, V (Bc rows)
        s = qi @ k[j0:j0 + bc].T * scale         # (Br, Bc): only this size
        m_new = torch.maximum(m, s.max(dim=1).values)
        alpha = torch.exp(m - m_new)             # correction factor for the old state, exp(m_old − m_new)
        p = torch.exp(s - m_new[:, None])
        l = l * alpha + p.sum(dim=1)
        o = o * alpha[:, None] + p @ v[j0:j0 + bc]
        m = m_new
    out[i0:i0 + br] = o / l[:, None]             # divide by l only at the end
    lse[i0:i0 + br] = m + torch.log(l)           # the backward pass needs to store only this
```

The causal mask also gives a free speedup: the loop skips each K, V block that is fully above the diagonal. The check in `04_tiled_attention.py` (T = 512, d = 64, float64, causal) gives:

| Block size Br × Bc | Max difference from naive attention | Max difference from PyTorch SDPA | Largest intermediate block |
|---|---:|---:|---:|
| 64 × 64 | 1.0 × 10⁻¹⁵ | 6.7 × 10⁻¹⁶ | 4,096 |
| 128 × 32 | 7.8 × 10⁻¹⁶ | 6.7 × 10⁻¹⁶ | 4,096 |
| 32 × 128 | 1.2 × 10⁻¹⁵ | 4.4 × 10⁻¹⁶ | 4,096 |
| 100 × 70 | 7.8 × 10⁻¹⁶ | 7.8 × 10⁻¹⁶ | 7,000 |

The intermediate tensors of the naive method have 2T² = 524,288 numbers. With tiles, the largest intermediate block has only a few thousand. All block sizes give the correct result. The block size changes only the speed (on a GPU, the SRAM capacity sets it).

**The backward pass also does not store P.** The forward pass stores only one number for each row, `lse = m + log l` (the logsumexp). The backward pass calculates `P = exp(S − lse)` again, block by block. The script checks the first 64 rows: the maximum difference between the recalculated P and softmax(S) is 2.8 × 10⁻¹⁶. FlashAttention calculates QKᵀ one more time, and in exchange it does not store a T × T matrix. It saves both memory and time, because the reads and writes that it saves cost much more than the extra calculation.

One important point: **FlashAttention calculates exact attention, not an approximation.** It is a different order of calculation for the same mathematical result. Later, FlashAttention-2 improved the parallel work partitioning, and FlashAttention-3 was optimized for the asynchronous and FP8 features of the H100. All of them are GPU kernels in CUDA/Triton. Our Python loop only shows the algorithm.

### 4.4 How zero uses it

zero does not have its own attention kernel. It calls PyTorch's `F.scaled_dot_product_attention` (SDPA):

```python
out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=is_causal,
                                     enable_gqa=self.n_kv_heads != self.n_heads)
```

SDPA selects a backend automatically from the device, the dtype, and the shapes. On a GPU with BF16/FP16, it prefers the FlashAttention kernel. If the conditions are not met, it falls back to the memory-efficient backend or the math backend. On the CPU, this chapter used the saved-tensor hook for a check. The attention of zero saves only q, k, v, the output, and one logsumexp per row. It saves no T × T matrix. (This is how we counted the activation formula in Section 2.) Can SDPA use the Flash backend on a GPU with `enable_gqa=True`? We verified this on an RTX 3090: with BF16, it uses Flash by default (the memory-efficient backend does not support GQA). See the section "GPU measurements" and Section 1 of [runs/2026-10-01-gpu0-check](../../runs/2026-10-01-gpu0-check/README.md).

## 5. Data parallelism: from DDP to FSDP

### 5.1 DDP: the mean of the gradients

If one GPU is too slow, use 8. The simplest method is **data parallelism**. Each GPU has a full copy of the model and gets a different part of the data. After the backward pass, the GPUs calculate the mean of the gradients of all GPUs (all-reduce). Each GPU updates with the same mean gradient, so the parameters always stay the same on all GPUs.

Why is this correct? The gradient of the mean loss of a batch is equal to the mean of the gradients of its sub-batches, because differentiation is linear. Thus "2 processes with B samples each, then the mean" = "1 process with 2B samples" = "1 process, two times B samples, then the sum" (gradient accumulation).

`06_ddp_by_hand.py` starts two real processes (the gloo backend of `torch.distributed`, which runs on a CPU). The core of DDP by hand is three lines:

```python
for p in model.parameters():                          # all of DDP is here:
    dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)     #   add the gradients of all processes
    p.grad /= world                                   #   divide by the number of processes = mean
```

The same MLP and the same data, 20 training steps (float64):

| Step | One batch of 2B | Gradient accumulation 2 × B | 2-process all-reduce |
|---:|---:|---:|---:|
| 1 | 0.532621 | 0.532621 | 0.532621 |
| 2 | 0.413222 | 0.413222 | 0.413222 |
| 3 | 0.440613 | 0.440613 | 0.440613 |
| 20 | 0.511739 | 0.511739 | 0.511739 |

The maximum difference of the final parameters is 6.9 × 10⁻¹⁷ for gradient accumulation and 6.9 × 10⁻¹⁷ for 2 processes. The only cause is rounding from a different order of floating-point sums. (This small example is only for the parity check. The loss itself does not decrease much.)

The real PyTorch `DistributedDataParallel` adds two optimizations. First, it packs the gradients into "buckets" before it sends them. Second, it **communicates during the backward pass**. When the gradients of a later layer are ready, their all-reduce starts at once and overlaps with the backward computation of the earlier layers. With gradient accumulation, it synchronizes only at the last micro batch (`no_sync()`) and skips the communication in between.

### 5.2 How all-reduce works: a ring

How do 8 GPUs add their gradients? The most direct method sends all gradients to one GPU, adds them there, and sends the result back. But then the bandwidth of that one GPU becomes the bottleneck. **Ring all-reduce** puts the GPUs in a ring and cuts the data of each GPU into N chunks:

1. **reduce-scatter**: N − 1 rounds. In each round, each GPU sends one chunk to its right neighbor, and the neighbor adds it to its own chunk. At the end, each GPU has one chunk of the full sum.
2. **all-gather**: N − 1 more rounds. The full chunks go once around the ring, and each GPU gets the full sum.

`06_ddp_by_hand.py` simulates 4 GPUs with 1,000,000 gradients each in one process. The maximum difference from the direct sum is 1.8 × 10⁻¹⁵. **Each GPU sends 1,500,000 numbers = 2·(N−1)/N × 1,000,000.** This amount is almost independent of the number of GPUs. This is the advantage of the ring algorithm. The FP32 gradients of the main-line model are 2.57 GiB. With an 8-GPU ring all-reduce, each GPU sends 4.50 GiB per step. Inside one H100 machine, the NVLink bandwidth between GPUs is 900 GB/s (the Nemotron-4 report gives this value). Inside one machine, this is not a bottleneck. Between machines, the bandwidth is an order of magnitude lower, and the overlap of communication and computation becomes important.

### 5.3 FSDP / ZeRO: split the copy on each GPU

DDP wastes memory: each of the 8 GPUs stores a full, identical copy of the parameters, the gradients, and the optimizer state. The idea of **ZeRO** (Rajbhandari et al., 2019) is to cut them into 8 parts (shards). Each GPU keeps only its own 1/8:

| Method | What each GPU stores | Static memory per GPU, main-line model (8 GPUs) |
|---|---|---:|
| DDP | Parameters + gradients + optimizer state, all in full; plus the DDP communication buckets | 12.84 GiB |
| ZeRO-1 | The optimizer state is sharded | 8.35 GiB |
| ZeRO-2 | The gradients are also sharded | 3.53 GiB |
| ZeRO-3 / FSDP | The parameters are also sharded | 1.28 GiB |

(Output of `zero/tools/memory_calc.py`. With the PyTorch default settings, the communication buckets hold one more copy of the FP32 gradients, 2.57 GiB. The DDP row includes it.)

If the parameters are sharded, how does the forward pass work? **FSDP (Fully Sharded Data Parallel)** (Zhao et al., 2023) uses all-gather before each layer to build the full parameters of that layer for a short time. After the calculation, it discards them. The backward pass builds them again. A reduce-scatter then sends each calculated gradient directly to the GPU that owns that shard. The communication is about 1.5 times that of DDP. In exchange, the parameters, the gradients, and the optimizer state each use only 1/N of the memory.

For the main-line model, the parameters need only 10.3 GiB, and DDP fits easily. Thus `configs/main/pretrain.toml` uses `parallel = "ddp"` by default. The table in Section 2 also shows this: **the activations limit us, not the parameters**, and FSDP does not shard the activations (the activations of each GPU depend only on its own micro batch). FSDP becomes necessary only for a larger model or a longer context (the 32K of Chapter 15). zero has both. Use `--set train.parallel=fsdp` to change between them.

### 5.4 Tensor parallelism and pipeline parallelism: for your information

When a model is so large that one layer does not fit on one GPU, there are two more ways to split it:

- **Tensor parallelism (TP)** (Megatron-LM, Shoeybi et al., 2019): split one matrix multiplication across many GPUs. Each layer needs communication, so TP is good only with a fast connection inside one machine.
- **Pipeline parallelism (PP)**: put different layers on different GPUs. The micro batches go through the GPUs one after the other, as in a production line. The cost is "bubbles" (some GPUs wait).

Llama 3 405B combines TP, PP, context parallelism, and FSDP into "4D parallelism". The 0.7B main-line model does not need any of these, so this course does not go into detail. Lectures 7 and 8 of CS336 explain them in depth.

## 6. MFU: how far from the hardware peak

The definition of **MFU** (introduced in the PaLM paper) is:

```
MFU = measured throughput (tokens/s) × FLOPs per token / hardware peak FLOPs
```

The numerator counts only the operations that the model must do (6N + the attention term). It does not count the recomputation of activation checkpointing. Thus MFU decreases when recomputation is on. This is on purpose: MFU measures how fast the training is, not how busy the GPU is. The metric for how busy the GPU is has the name HFU (hardware FLOPs utilization). HFU also counts the recomputation. PaLM 540B has an MFU of 46.2% and an HFU of 57.8%.

There are many reasons why MFU does not reach 100%:

- Element-wise operations (normalization, activation functions) are memory-bound.
- Communication does not fully overlap with computation.
- Python and kernel launches add overhead.
- Small matrices cannot keep the Tensor Cores busy.

The training log of zero shows MFU directly (`zero/train/trainer.py`). On a GPU, the trainer looks up the peak in a table by device name. On a CPU, it does not show MFU. To see MFU on a CPU too, `01_step_cost.py` measures the peak FP32 matrix-multiplication speed of one thread (1024 × 1024, fastest run: 48.8 GFLOPS). Then it reads the throughput from the log of the tiny pretraining run (median 2,676 tokens/s):

```
MFU = 2,676 token/s × 7.078 × 10⁶ FLOPs/token / 48.8 GFLOPS ≈ 38.8%
```

This is a **tiny-configuration demo**. The denominator is a measured value of one CPU thread on this computer, and it changes with the machine load in each run. (In 2026-10, we cached data for the video on a different server. There, the median throughput of the tiny pretraining run was 6,148 tokens/s. The peak measured during the rendering was 57.2 GFLOPS, which gives 76.0%. The throughput and the peak were not measured at the same time, so the ratio changes when the machine load changes.) Do not compare this value directly with MFU on a GPU. It only shows how to calculate MFU: multiply tok/s from the log by the FLOPs per token, and divide by the peak.

## 7. Loss spikes: sudden explosions during training

### 7.1 What occurs and why

A **loss spike** is a sudden jump of the loss while the training curve decreases smoothly. Some spikes recover by themselves after tens of steps. Others never recover (divergence). PaLM 540B had about 20 spikes during training, with gradient clipping on. Smaller models did not have them. The OLMo team saw that spikes of the gradient norm often occur before loss spikes, and that larger models have more frequent spikes.

Public reports give these main types of causes:

- **Data**: OLMo 2 found that the batches with spikes often contain long repeated n-grams (for example, a long run of `255, 255, 255, …`).
- **Numerical scales out of control**: attention logits grow larger and larger, and the softmax saturates to one-hot. Output-layer logits grow larger and larger. Weight decay pushes the norm of the embeddings too low, and then the gradients of the first layers become too large.
- **A learning rate that is too large, or a warmup that is too short**: this is the effect of Chapter 1 again, now in a large model. A step that is too large makes training diverge.

PaLM also found something interesting. The team took the batches near a spike and trained on them from an earlier checkpoint. Then no spike occurred. **A spike is the combination of specific data and a specific state of the parameters.** It is not only bad data.

### 7.2 Countermeasures

| Method | What it does | Adoption |
|---|---|---|
| Gradient clipping (Chapter 6) | When the global gradient norm is more than 1.0, scale the gradient down proportionally | Almost all reports use it; see the end of the chapter |
| Warmup + a suitable peak learning rate (Chapter 6) | Do not start too fast | All reports use warmup. Llama 3 also mentions a smaller batch early in training to improve stability |
| QK-Norm (Chapter 9) | Apply RMSNorm to q and k, so that the attention logits do not grow with the drift of the activations | Qwen3, Gemma 3, OLMo 2; the main-line model already uses it |
| Data filtering | Remove documents with long repeated n-grams | OLMo 2 (Chapter 13) |
| Skip bad batches / roll back | Restart from a checkpoint before the spike and skip the data near the spike; or skip a step with an abnormal gradient norm automatically | PaLM (rolled back ~100 steps and skipped 200–500 batches); `SkipStepAdamW` in the OLMo 2 training code (no update when the gradient norm or the loss is more than 6 standard deviations from a moving window). Fewer than 3 families, so it is in Frontier notes |
| z-loss | Add `10⁻⁴ · log²Z` to the loss, so that the softmax normalizer Z does not grow | PaLM, OLMo 2; fewer than 3 families, so it is in Frontier notes |

The good news: with the correct recipe, large-model training can be very smooth. Llama 3 405B reports that it saw only a few spikes and needed no manual intervention. DeepSeek-V3 reports no irrecoverable spikes and no rollbacks during the full training. The main-line model already has the first three items (gradient clipping, warmup, QK-Norm). Chapter 13 does the data filtering.

### 7.3 Thus you must be able to roll back

Whatever preventive methods you use, the final safety net is: **save checkpoints at regular intervals, so that after a problem you can go back and continue training**. The next section shows how to make "continue training" exact in every bit.

## 8. Resume from a checkpoint: restore "all" state

In a run of 10 days or more, the machines will have problems. The training of Llama 3 405B had 466 interruptions in 54 days. 419 of them were unexpected, and about 78% were hardware problems. Thus training must be able to save its state at any time, and to continue from a saved state at any time.

The standard for "continue" is: **each step after the resume is bit-identical to a run without an interruption** (the same in every bit). To get this, the model weights are far from sufficient. The full state of a training run is:

1. The model parameters.
2. The optimizer state (AdamW m and v, and the step count t for the bias correction).
3. The position of the learning-rate scheduler.
4. The position in the data.
5. The state of all random number generators (for dropout, data shuffling, and so on).
6. Counters such as the step count and the number of trained tokens.

`07_resume.py` first trains 30 steps without an interruption. Then it trains again and "crashes" at step 17 (the latest checkpoint is at step 15). It restores the state into new objects and trains to step 30:

| Step | Uninterrupted | Resumed | |
|---:|---:|---:|---|
| 16 | 1.862050 | 1.862050 | Bit-identical |
| 17 | 1.784434 | 1.784434 | Bit-identical |
| 18 | 1.828543 | 1.828543 | Bit-identical |
| 30 | 1.769457 | 1.769457 | Bit-identical |

All 15 steps are bit-identical. Then the script "forgets" one item of the state at a time. It measures the maximum loss difference from the uninterrupted run over the 15 resumed steps:

| What is not restored | Max difference |
|---|---:|
| Optimizer state (m, v, t start from zero) | 1.13 × 10⁻² |
| Learning-rate schedule (warmup starts again) | 1.46 × 10⁻¹ |
| Data position (the first batches are drawn again) | 1.25 × 10⁻¹ |
| Global RNG (different dropout masks) | 1.61 × 10⁻² |

Each item makes the training leave its original path. The change does not always make the model worse. But it makes the experiment not reproducible: you can no longer tell if an effect comes from the recipe or from the resume.

Production code must also handle one more case: **the process is killed while it writes a checkpoint** (this often occurs on preemptible instances). If the code overwrites the old files directly, a half-written, bad checkpoint stays. zero writes to a temporary folder first. When all files are written, it renames the folder atomically with `os.replace`. Last, it updates the `latest` pointer. If the process is killed in the middle, `latest` still points to the last complete checkpoint.

## 9. Summary

| Problem | Solution | Evidence in this chapter |
|---|---|---|
| How much compute and time one run needs | 6N + attention term per token; time = total FLOPs / (peak × MFU) | Main line, about 400B tokens: 2.78 × 10²¹ FLOPs; 8×H100 at MFU 0.4: about 10.2 days, $4,881 |
| The model does not fit in GPU memory | 16 bytes/parameter + activations; gradient accumulation, activation checkpointing | Measured 16 bytes/parameter; checkpointing reduces the activations to 21%; main line at micro batch 8 estimated above 80 GB, so the configuration uses 4 × accumulation 4 |
| FP32 is too slow, FP16 overflows | Calculate in BF16, store in FP32 | BF16 accumulation stuck at 4, weight updates lost; FP16 70,000 → inf |
| Attention reads and writes T × T matrices | Tiles + online softmax; the backward pass stores the logsumexp and recalculates | Max difference from naive attention and SDPA about 10⁻¹⁵; saves 8 GiB per layer in the main line |
| One GPU is too slow | DDP: all-reduce for the mean of the gradients | 2 processes = gradient accumulation = large batch; parameter difference 6.9 × 10⁻¹⁷ |
| One copy on each GPU wastes memory | ZeRO / FSDP shard the state | Main-line static memory per GPU 12.84 → 1.28 GiB |
| Loss spikes | Clipping, warmup, QK-Norm, data filtering, rollback | Public reports (PaLM, OLMo 2, Llama 3, DeepSeek-V3) |
| Machines fail | Save and restore all state, atomic writes | 15 resumed steps bit-identical; each missing item causes a deviation |

---

## GPU measurements (one RTX 3090)

> **Note:** All numbers in the main text above come from CPU runs. This section measures on one NVIDIA GeForce RTX 3090 (24 GB of memory, Ampere architecture). Its spec sheet gives a dense BF16 Tensor Core peak of about 71 TFLOPS, about 35.6 TFLOPS for FP32, and a memory bandwidth of about 936 GB/s. Environment: PyTorch 2.11.0+cu128, CUDA 12.8, October 2026. The server sets the power limit of this GPU to 240 W (factory default 350 W). Under a continuous full load, the GPU lowers its clock. Thus the absolute values of compute and bandwidth are lower than those of a 3090 at full power, and the relations between values are more reliable. If you do not have a GPU, skip this section.

First, what this section is **not**: the 3090 is not an H100. Its BF16 peak is about 1/14 of the H100 peak, and it has only 24 GB of memory. The time, cost, and memory of the main-line model on 8×H100 in the main text are still estimates from formulas. Stage 6 will measure them. This section shows two things. First, the **method**: how to measure these quantities on a real GPU. Second, the **relations of magnitude**: how far the formulas are off on a real GPU, and where the results are unexpected.

Run:

```bash
uv run python chapters/14-pretraining-engineering/code/08_gpu_matmul_attention.py   # matmul throughput, BF16 error, attention memory, SDPA backend (about 15 s)
uv run python chapters/14-pretraining-engineering/code/09_gpu_train_step.py         # one training step of the main-line model: memory, activation checkpointing, MFU (about 1 min, needs 24 GB of GPU memory)
```

All timings warm up first, then use CUDA events, and take the median of several runs.

### Matrix multiplication: how much faster is mixed precision (Section 3)

Part ① of `08`: square matrix multiplication C = A·B, in TFLOPS. The last column is the relative error against the FP64 result for the same pair of 4096 × 4096 random matrices. The spec peak is the dense value from the Ampere GA102 whitepaper (Tensor Cores with FP32 accumulation):

| Format | n = 1024 | 2048 | 4096 | 8192 | Spec peak | Best / peak | Relative error |
|---|---:|---:|---:|---:|---:|---:|---:|
| FP32 | 9.7 | 14.1 | 16.9 | 16.6 | 35.6 | 48% | 1.1 × 10⁻⁶ |
| TF32 | 21.3 | 24.2 | 28.4 | 31.1 | 35.6 | 87% | 2.9 × 10⁻⁴ |
| BF16 | 54.1 | 60.4 | 54.4 | 53.7 | 71.0 | 85% | 2.9 × 10⁻³ |
| FP16 | 55.2 | 58.9 | 52.2 | 51.2 | 71.0 | 83% | 3.6 × 10⁻⁴ |

- **On the same GPU, BF16 is 3.6 times faster than FP32** (60.4 vs 16.9). This is the gain from "do the matrix multiplications in low precision" in Section 3.2. The spec sheet shows only a factor of 2, but the measured factor is larger. The FP32 value of 35.6 is an ideal value. It assumes that both FP32 data paths of Ampere are fully used (and one of them is shared with integer operations). A real matrix multiplication reaches only 48% of it. BF16 on the Tensor Cores reaches 85%. On the H100, the spec sheet gives 989.5 TFLOPS for dense BF16 vs 67 TFLOPS for FP32, a factor of almost 15. Without mixed precision, you discard most of the compute.
- **Small matrices cannot keep the GPU busy.** FP32 at n = 1024 gives only 9.7, less than 60% of the value at n = 4096. One of the MFU losses in Section 6 is "small matrices cannot keep the Tensor Cores busy". This is what it means. In the BF16 and FP16 rows, 4096 and 8192 are even a little lower than 2048. The reason is the power limit of 240 W on this GPU (default 350 W): under a continuous full load, the GPU lowers its clock. The same cell can differ by about 10% between two runs, so look only at the magnitude.
- **TF32** is the "FP32" of the Tensor Cores: it cuts the input mantissa to 10 bits before the multiplication. On this GPU, it is 1.8 times faster than true FP32, with an error of 2.9 × 10⁻⁴. This error is between those of FP16 and BF16. By default, PyTorch does **not** use TF32 for matrix multiplications. Turn it on with `torch.set_float32_matmul_precision("high")`.
- **Error**: the BF16 error of 2.9 × 10⁻³ is 8 times the FP16 error. This agrees with the 3 fewer mantissa bits (2³ = 8). BF16 gives up precision to get range (Section 3.3).

Part ② of `08` takes the **same** Linear and the **same** inputs as `02_precision.py` ⑥ and runs BF16 autocast on the CPU and on the GPU. The relative error against the FP32 result is 2.78 × 10⁻³ on the CPU (the same as in the main text) and 3.25 × 10⁻³ on the GPU. The small extra error comes from a default setting of cuBLAS. For speed, cuBLAS lets BF16 matrix multiplications use BF16 for the intermediate reduction of partial sums (`torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction`, default True). With this setting off, the GPU also gives 2.78 × 10⁻³. With the default settings, only 59.9% of the BF16 outputs of the CPU and the GPU are bit-identical. The same mathematical operation with a different order of additions gives a different last bit. Thus you can expect the "bit-identical" results of Section 8 only on the same machine with the same kernels.

### Attention: FlashAttention saves memory and reads/writes (Section 4)

Part ③ of `08`: 1 sequence, 16 heads × head_dim 128 (the main-line shape), BF16, causal, forward + backward pass. "Naive" is the method of `naive_attention` in `04_tiled_attention.py` (it calculates S and P explicitly). "FlashAttention" is SDPA with only the FlashAttention backend allowed. The peak memory is the highest memory above q, k, and v themselves:

| T | S + P theoretical size | Naive peak memory | FlashAttention peak memory | Naive time | FlashAttention time | Speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 1,024 | 64 MiB | 144 MiB | 32 MiB | 1.41 ms | 0.46 ms | 3.1× |
| 2,048 | 256 MiB | 528 MiB | 64 MiB | 5.73 ms | 1.74 ms | 3.3× |
| 4,096 | 1,024 MiB | 2,080 MiB | 129 MiB | 25.65 ms | 5.08 ms | 5.1× |
| 8,192 | 4,096 MiB | 8,256 MiB | 257 MiB | 100.95 ms | 18.54 ms | 5.4× |
| 16,384 | 16,384 MiB | Out of memory | 514 MiB | — | 74.92 ms | — |

- **Memory**: for the naive method, when T doubles, the memory increases 4 times. The peak is about two times S + P (the backward pass also needs dP and dS of the same size). At T = 16K, one layer does not fit on this 24 GB GPU. For FlashAttention, when T doubles, the memory only doubles. The extra memory is only the output, the gradients of q, k, v, and one logsumexp per row, as Section 4.3 says. The value "S and P of one main-line layer are 8.0 GiB" in Section 4.1 is for micro batch 8. Here, there is 1 sequence: at T = 4096, S + P is 1 GiB, and the measured peak of the naive method is 2 GiB.
- **Time**: for both methods, when T doubles, the time increases 3–4 times. Both do a number of operations of the order of T². FlashAttention does not do many fewer operations. The causal mask lets it skip the upper triangle, but the backward pass calculates QKᵀ again. Together, it does about 60% of the operations of the naive method. (In the forward + backward pass, the naive method does 6 full T × T matrix multiplications. FlashAttention does 7, each at half size.) Most of the 3–5× speedup comes from not writing the T × T matrices to GPU memory and reading them back. The longer T is, the larger the effect. This table is the evidence for "attention is slow because of reads and writes, not because of calculation".

Part ④ of `08` uses the PyTorch profiler to see which CUDA kernels the attention of zero (GQA, `enable_gqa=True`, main-line shape, 1 layer) really calls:

- **BF16 autocast**: the forward pass uses `pytorch_flash::flash_fwd_kernel`. The backward pass uses `flash_bwd_convert_dq_kernel`, `flash_bwd_dot_do_o_kernel`, and `flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel`. This is FlashAttention. When we force one backend at a time, the FlashAttention and cuDNN backends both accept `enable_gqa=True`, and the memory-efficient backend raises an error.
- **FP32**: there is no fused attention kernel, only a separate softmax kernel. FlashAttention supports only FP16 / BF16, and the memory-efficient backend does not support GQA. Thus SDPA falls back to the math backend, which calculates the full T × T matrix. The next subsection shows the cost of this.

### Memory: is memory_calc correct on a real GPU (Section 2)

`09` puts the main-line model (689.5M, the zero Transformer + fused AdamW) on this GPU and really trains some steps with micro batch 1. The parameters + AdamW m and v always use 7.84 GiB (7.71 GiB at 12 bytes/parameter). The gradients exist only from the backward pass to the update. "Formula" is `zero/tools/memory_calc.py` (one GPU, no DDP communication buckets). We compare the memory added at the end of the forward pass with its "activations + BF16 weight copies". We compare the peak of the full step with its total:

| T | Setting | Added at the end of the forward pass | Formula | Peak of the full step | Formula |
|---:|---|---:|---:|---:|---:|
| 1,024 | FP32 | 5.98 GiB | 3.93 GiB | 14.32 GiB | 14.46 GiB |
| 1,024 | BF16 autocast | 4.08 GiB | 4.03 GiB | 12.41 GiB | 14.68 GiB |
| 1,024 | BF16 + activation checkpointing | 0.56 GiB | 0.68 GiB | 10.74 GiB | 11.33 GiB |
| 4,096 | FP32 | Out of memory | 15.73 GiB | Out of memory | 27.01 GiB |
| 4,096 | BF16 autocast | 12.33 GiB | 12.25 GiB | 22.17 GiB | 24.02 GiB |
| 4,096 | BF16 + activation checkpointing | 1.75 GiB | 2.15 GiB | 12.25 GiB | 13.92 GiB |

- **For BF16 activations, the formula is accurate.** At the end of the forward pass, the measurement and the formula differ by only 1–2% (4.08 vs 4.03, 12.33 vs 12.25). The activation formula that we checked byte by byte on the CPU is also approximately correct on CUDA.
- **For the peak of the full step, the formula is conservative.** In the BF16 rows, the measurement is 5–16% lower than the formula. The formula adds the parameters, the gradients, the optimizer state, all activations, and the logits as if they are all in memory at the same time. The real peak occurs during the backward pass, when part of the activations is already freed. The best example is BF16 at T = 4096. The formula gives 24.02 GiB, more than the 23.6 GiB of this GPU, so by the formula it "does not fit". The real peak is 22.17 GiB, and it fits. The main-line memory table in Section 2 is also this type of conservative upper bound.
- **Activation checkpointing**: at T = 4096, the memory added at the end of the forward pass decreases from 12.33 GiB to 1.75 GiB (14%). This is even lower than the 21% of the tiny model in Section 2. The reason: tiny has only 4 layers, so the output layer, which checkpointing does not change, is a larger part. With checkpointing at T = 4096, the formula predicts that this GPU holds at most 3 sequences. The measurement also gives 3 (4 sequences do not fit in memory).
- **FP32 gives an unexpected result.** At T = 1024, the memory added at the end of the forward pass is 5.98 GiB, 2.05 GiB more than the formula. Most of the extra memory is the attention matrices that the math backend saves (see the last subsection). They need 28 layers × 16 heads × 1024² × 4 bytes = 1.75 GiB. The rest is intermediate tensors such as the expanded K and V. We checked the formula on the CPU (in FP32, the CPU SDPA also has a kernel that does not store T × T matrices). On a GPU in FP32, the formula is not correct. At T = 4096, not even one FP32 sequence fits. zero uses BF16 autocast on CUDA by default, so the main line is not affected. But this is the "T × T matrix" problem of Section 4, now on a real GPU.

### Speed: the cost of activation checkpointing, and MFU (Sections 2 and 6)

The speed of the same run follows. MFU uses the BF16 spec peak of 71 TFLOPS and the FLOPs per token of Section 1 (6N + 12·L·q_dim·T). HFU also counts the extra forward pass of the checkpoint recomputation. The last column halves the attention term, because the causal mask skips the upper triangle and it is not calculated. This column is the utilization of "the operations that are really done":

| T | Setting | Per step | tokens/s | MFU | HFU | Utilization with the causal term halved |
|---:|---|---:|---:|---:|---:|---:|
| 1,024 | FP32 | 527 ms | 1,942 | 13.2% | | 12.3% |
| 1,024 | BF16 autocast | 195 ms | 5,239 | 35.7% | | 33.1% |
| 1,024 | BF16 + activation checkpointing | 249 ms | 4,114 | 28.1% | 36.4% | 26.0% |
| 4,096 | BF16 autocast | 723 ms | 5,665 | 55.5% | | 44.2% |
| 4,096 | BF16 + activation checkpointing | 919 ms | 4,455 | 43.6% | 57.1% | 34.8% |

At T = 4096 with checkpointing and a larger micro batch: 2 sequences take 1,721 ms per step, 4,761 tokens/s, MFU 46.6%. 3 sequences take 2,561 ms per step, 4,798 tokens/s, MFU 47.0% (peak of the full step 15.33 and 18.78 GiB; formula 17.37 and 20.82 GiB).

- **Mixed precision**: at T = 1024, BF16 takes 195 ms per step and FP32 takes 527 ms. BF16 is 2.7 times faster. This is a little less than the 3.6 times of the matrix multiplication. The parts that do not use the Tensor Cores (normalization, activation functions, the optimizer update) do not become faster. Also, FP32 has the extra cost of the math-backend attention.
- **Activation checkpointing**: the time per step increases from 723 ms to 919 ms, 27% slower. This agrees with the estimate "one more forward pass, about 1/3 more compute" (the recomputation does not include the output layer, so it is less than 1/3). MFU drops from 55.5% to 43.6%, but the HFU of 57.1% is about the same as the MFU without checkpointing. The GPU is equally busy, but part of the work is recomputation. This is why Section 6 separates MFU and HFU. You can use the saved memory for a larger micro batch: with 3 sequences, MFU increases again to 47.0%. But on this GPU, the throughput with 3 sequences (4,798 tokens/s) is still lower than with 1 sequence without checkpointing (5,665 tokens/s). One sequence already has 4096 tokens, so the matrices are not small. The gain from a larger batch (MFU 43.6% → 47.0%) does not pay for the cost of the recomputation.
- **The size of MFU**: for the main-line model at T = 4096, on one GPU in eager mode, MFU is 55.5%. This looks much higher than the 0.4 that the main text assumes for the H100. But three things must be considered together:
  1. **Convention**: the formula of Section 1 does not halve the attention term for the causal mask (the same as PaLM and nanochat). But FlashAttention really skips the upper triangle. At T = 4096, the attention term is 41% of the FLOPs per token. After halving, the operations that are really done are 5.55 GFLOP/token, not 6.96, and the utilization is 44.2%. The longer the sequence, the more this convention overstates the value. (In the GPU measurements of Chapter 15, one layer at 32K gives almost 100% with this convention.)
  2. **Power limit**: the power limit of this GPU is set to 240 W (default 350 W). Under full load, the GPU lowers its clock, and the best BF16 matrix multiplication was only 60.4 TFLOPS (85% of the spec). Thus the absolute MFU values in the table are low. A 3090 without a power limit gives somewhat higher values.
  3. **The 3090 is not an H100**: by the spec sheets, the H100 has about 14 times the BF16 compute of the 3090. But it has only about 3.6 times the memory bandwidth (3.35 TB/s vs 936 GB/s). The more compute there is relative to bandwidth, the larger the fraction of time in memory-bound parts such as normalization and element-wise operations. Then a high MFU in eager mode is more difficult. This is where `torch.compile`, which fuses element-wise operations, helps. 8-GPU DDP communication also adds cost. Thus the MFU on the H100 must wait for the measurements of stage 6. Do not use the 55.5% here to change the budget.

---

## From minimal code to production code

The minimal code of this chapter only shows the principles. The real training of the main-line model is in `zero/train/`. The table shows how they correspond:

| Minimal code | Production code (`zero/`) | What it adds, and why |
|---|---|---|
| The formula in `01_step_cost.py` | `estimate_flops_per_token` in `zero/model.py`; `zero/tools/estimate_cost.py` | The same formula. The estimation tool has a table of GPU peaks, the price, and the number of GPUs, and gives GPU-hours and cost. The code marks the H100 peak of 989.5 TFLOPS as "to be verified". The Nemotron-4 report gives 989 teraFLOP/s (bfloat16, without sparsity), which agrees |
| `05_memory.py` counts the saved tensors | **New in this chapter**: `zero/tools/memory_calc.py` (`estimate_memory`, `max_micro_batch`, CLI) | Writes the activation formula per layer per token, item by item, for the zero `Block`. Supports DDP / ZeRO-1 / ZeRO-2 / FSDP, activation checkpointing, and BF16 / FP32. `tests/test_memory_calc.py` makes sure that the formula is **equal byte for byte** to the measurement with `saved_tensors_hooks` |
| Hand-written `torch.autocast` experiments | `autocast_context` in `zero/train/trainer.py`: BF16 autocast on CUDA, parameters and optimizer state stay in FP32; RMSNorm and cross-entropy in `zero/model.py` convert to FP32 before they calculate | Separate protection for the operations that are sensitive to precision; FP32 by default on the CPU |
| The Python loop in `04_tiled_attention.py` | `Attention` in `zero/model.py`: `F.scaled_dot_product_attention(..., enable_gqa=...)` | No own kernel; PyTorch selects the backend. Measured on an RTX 3090: BF16 + GQA uses FlashAttention (the memory-efficient backend does not support GQA). See Section 1 of [runs/2026-10-01-gpu0-check](../../runs/2026-10-01-gpu0-check/README.md) |
| Hand-written gradient accumulation | `Trainer.train`: `grad_accum_steps` forward + backward passes, loss divided by the number of accumulation steps; with DDP, `no_sync()` for the first micro batches | Skips the gradient communication in between |
| Hand-written all-reduce in `06_ddp_by_hand.py` | `zero/train/dist.py`: `init_distributed` (reads the torchrun environment variables; NCCL on GPUs, gloo on CPUs), `wrap_model` (DDP or FSDP2 `fully_shard`, BF16 parameters + FP32 gradient reduction) | PyTorch DDP does the bucketing and the overlap of communication with the backward pass. DDP and FSDP2 are verified on 2×RTX 3090 (PCIe): consistency across GPUs, checkpoint gathering, and resume. The throughput on 8 GPUs with NVLink is not measured yet. See Section 14 of [runs/2026-10-01-gpu0-check](../../runs/2026-10-01-gpu0-check/README.md) |
| Data split per process | `PackedDataLoader` in `zero/data/loader.py`: global sample g goes to rank g % world_size | Makes sure that "2 GPUs with B samples each" and "1 GPU with 2B samples" see the same data. Only then can DDP pass the parity check |
| `clip_grad_norm_` of Chapter 6 | `Trainer.train` clips at each step and writes the gradient norm to the log | The gradient norm is the first signal of a spike |
| MFU in `01_step_cost.py` | `Trainer._peak_flops` + `tok_per_s` and `mfu` in the log | Looks up the peak by device name; not shown on a CPU |
| `state_dict` in `07_resume.py` | `zero/train/checkpoint.py`: `save_checkpoint` (model / optim / meta.json (step, token count, scheduler, configuration) / data position + all RNG states of each rank); writes `.tmp_step_xxx` first, then `os.replace`, and last updates `latest` atomically; `keep_last` deletes old checkpoints automatically | With many processes, each rank saves its own loader and RNG state. If the number of GPUs changes, zero refuses an exact resume (the data positions do not match) |
| `checkpoint(self.block, ...)` in `05_memory.py` | `Transformer.forward` in `zero/model.py`: when `train.activation_checkpointing = true`, `torch.utils.checkpoint` wraps each Block (`Trainer` turns it on); the main-line configuration uses `false` by default | Works only in training and without the KV cache. `tests/test_activation_checkpointing.py` makes sure that the gradients are the same with and without it. Verified on an RTX 3090 (at main-line size on 24GB, not even micro batch 1 fits without checkpointing); see Section 7 of [runs/2026-10-01-gpu0-check](../../runs/2026-10-01-gpu0-check/README.md). The measurements of stage 6 will decide if the 80GB H100 needs it |
| — | z-loss, skipping bad batches | **zero does not have these yet** |

**Parity checks (all pass on a CPU)**:

```bash
uv run pytest tests/test_train_resume.py tests/test_ddp_cpu.py tests/test_memory_calc.py   # 11 passed
```

- `tests/test_train_resume.py`: trains 8 steps, and compares this with a run that "crashes" at step 5 and resumes from the checkpoint of step 4. The loss of steps 5–8, the validation loss, the final parameters, and the token count are **bit-identical**. It also tests that a fixed seed is reproducible, and the atomic layout and the cleanup of checkpoints.
- `tests/test_ddp_cpu.py`: starts 2 processes with DDP on the gloo backend (micro batch 2, accumulation 2) and compares each step with one process at micro batch 4. The loss difference is less than 10⁻⁵. It also checks that the multi-process checkpoint has one state for each rank.
- `tests/test_memory_calc.py`: checks the parameter count calculated by hand (main line 689,518,848) and the activation bytes per layer (tiny BF16 8,368 / FP32 12,336). It also checks the static memory for each sharding method. Last, it does a byte-for-byte parity check of the formula against the real saved tensors (also with activation checkpointing).

---

## Main-line progress

### Tiny-configuration demo (CPU, `configs/tiny`, about 1.3M parameters)

> **Note:** All of the following is a **tiny-configuration demo**. It shows only that the code runs and that each mechanism works as expected. It does not represent any result of the main-line model.

> **Note:** About the numbers: the numbers of the training experiments in this chapter come from one CPU run on the course build machine. Different machines and different versions of the low-level math libraries use a slightly different order of floating-point operations. Some hundred training steps make these small differences larger. Your numbers can differ from the second or third decimal place. Trust the conclusions below that do not depend on exact values. For a rerun on a different server in 2026-10, see [runs/2026-10-01-gpu0-check/chapters-11-15.md](../../runs/2026-10-01-gpu0-check/chapters-11-15.md).

**Single-process pretraining** (`--set` changes `out_dir`, so that this chapter does not share an output folder with other chapters):

```bash
uv run python -m zero.train.pretrain --config configs/tiny/pretrain.toml --set train.out_dir=out/ch14/pretrain
```

```
Model parameters 1.31M (non-embedding 0.79M), 2048 tokens per step, 200 steps, device cpu, world_size=1
step      1/200 | loss 7.6507 | lr 1.00e-04 | gnorm 0.98 | 1,774 tok/s
step     25/200 | loss 6.3844 | lr 2.50e-03 | gnorm 0.38 | 2,396 tok/s
step     50/200 | loss 6.3938 | lr 3.00e-03 | gnorm 1.75 | 2,853 tok/s
step     75/200 | loss 6.2778 | lr 3.00e-03 | gnorm 0.58 | 2,790 tok/s
step    100/200 | loss 6.1065 | lr 3.00e-03 | gnorm 0.38 | 2,909 tok/s | val 6.0048
checkpoint saved: out/ch14/pretrain/ckpt/step_00000100
step    125/200 | loss 5.9836 | lr 3.00e-03 | gnorm 0.56 | 2,465 tok/s
step    150/200 | loss 5.8692 | lr 3.00e-03 | gnorm 0.33 | 2,835 tok/s
step    175/200 | loss 5.6566 | lr 1.99e-03 | gnorm 0.40 | 2,561 tok/s
step    200/200 | loss 5.4736 | lr 3.00e-04 | gnorm 0.38 | 2,528 tok/s | val 5.7116
checkpoint saved: out/ch14/pretrain/ckpt/step_00000200
```

On one CPU thread, 200 steps took 155.8 s (`elapsed_s` in the log; other tasks used the machine at the same time, so the throughput changes). On a CPU, zero does not show MFU. With the method of Section 6 and the measured peak of this computer, MFU is about 38.8% (`01_step_cost.py`).

In 2026-10, we ran the same command again on a different server. The first 25 steps were bit-identical. From step 50, the runs started to differ (6.3596; above, it is 6.3938). At step 200, the loss was 5.3342 and val was 5.6004. Two checks below do not depend on exact values, and both passed on both machines. First, after the resume, steps 125–200 are bit-identical to the uninterrupted run. Second, 2-process DDP runs (val 5.6080 at step 200).

**Resume from a checkpoint (the real command line)**: copy the output folder above. Delete the checkpoint of step 200, and set `latest` to point back to step 100. Then run the same command again:

```
Resumed training from out/ch14/resume/ckpt/step_00000100 (step 100)
step    125/200 | loss 5.9836 | ...
step    150/200 | loss 5.8692 | ...
step    175/200 | loss 5.6566 | ...
step    200/200 | loss 5.4736 | ... | val 5.7116
```

Compare the `log.jsonl` of the two runs item by item. The loss at steps 125, 150, 175, and 200 (5.9835896492004395, 5.869236469268799, 5.6566314697265625, 5.473620414733887) and the validation loss are all bit-identical. At step 200, each tensor in `model.pt` passes `torch.equal`.

**2-process DDP (torchrun, CPU gloo)**:

```bash
uv run torchrun --standalone --nproc_per_node 2 -m zero.train.pretrain --config configs/tiny/pretrain.toml \
    --set train.out_dir=out/ch14/ddp2 --set train.micro_batch_size=8
```

```
Model parameters 1.31M (non-embedding 0.79M), 2048 tokens per step, 200 steps, device cpu, world_size=2
step      1/200 | loss 7.6537 | lr 1.00e-04 | gnorm 0.97 | 3,178 tok/s
step    100/200 | loss 5.9307 | lr 3.00e-03 | gnorm 0.38 | 3,797 tok/s | val 5.9674
step    200/200 | loss 5.4258 | lr 3.00e-04 | gnorm 0.37 | 3,729 tok/s | val 5.5986
```

Each process (rank) has micro batch 8, so the two together still have 2048 tokens per step. The checkpoint contains `rank0.pt` and `rank1.pt`. With two processes, the throughput is about 3,700 tokens/s; with one, about 2,500–2,900 (the same machine with a changing load, so look only at the magnitude). Note that the loss of this run **must not** be bit-identical to the single-process run. The tiny configuration is a mixture of three sources (`MixtureLoader`). Each rank selects its sources at random with `(seed, rank)`, so 2 processes see different data from 1 process. The strict DDP parity check uses data from one source, and `tests/test_ddp_cpu.py` does this check.

**Cost and memory estimates** (from formulas, **not verified on a GPU yet**):

```bash
uv run python -m zero.tools.estimate_cost --config configs/main/pretrain.toml --tokens 400B --gpu h100-sxm
uv run python -m zero.tools.memory_calc configs/main/pretrain.toml
```

| Item | Estimate |
|---|---|
| Pretraining 400B tokens, 8×H100, MFU 0.4 | 2.782 × 10²¹ FLOPs, 1,952.5 GPU-hours, 10.17 days, $4,881 (within the ~$5K for pretraining in GOAL.md 3.4, but with a very small margin: if the measured MFU is below 0.391, the cost goes over the limit. Gate 1 must still fix the final values of the token count, the measured MFU, and the rental price) |
| Memory: main-line configuration, micro batch 4 × 4096, DDP, eager | About 64.0 GiB per GPU, so it fits; the maximum micro batch is about 5. Micro batch 8 needs about 113.8 GiB, **more than 80 GB** |
| Configuration | `configs/main/pretrain.toml` already uses micro batch 4 × accumulation 4 (still 524,288 tokens per step, `max_steps` = 762,940). If it still does not fit, turn on `train.activation_checkpointing` (estimate: about 26.8 GiB). The measurements of stage 6 decide |

### The first task of step 2: a GPU verification run of ≤ $50 (GOAL.md Section 10, stage 6)

Before we spend a lot of money on pretraining, we run 8×H100 for about 1.5 hours (about $30, at most $50). This run verifies, one by one, all paths of this chapter that are "not verified on a GPU yet". The full commands are in Section 2 of [`runs/RUNBOOK.md`](../../runs/RUNBOOK.md). The table shows how the items correspond to this chapter:

| Verification item | What it verifies | Section of this chapter |
|---|---|---|
| One GPU, BF16 + SDPA | Attention really uses the FlashAttention kernel; does `enable_gqa=True` make it fall back to a different backend? | Section 4 |
| 8-GPU DDP, 200 steps | NCCL communication, the loss decreases normally, the memory is balanced across GPUs | Section 5 |
| Kill in the middle, then resume | With many GPUs + BF16, after the resume the data order is exactly the same and the loss agrees (BF16 allows differences of the order of 10⁻³) | Section 8 |
| FSDP2 | The loss of the first 50 steps agrees with DDP; one GPU can read the checkpoint back | Section 5.3 |
| torch.compile on/off | Both give the same loss; record the speedup | Section 6 |
| Measured MFU | Use `mfu` from the log to run `estimate_cost --mfu <measured>` again, and update all budgets | Sections 1 and 6 |
| Try micro batch sizes | Find the largest micro batch without OOM, compare it with the prediction of `memory_calc` (about 5), and calibrate the formula | Section 2 |

### To be added after GPU training

- The measurements of stage 6: MFU, throughput, and peak memory per GPU, compared with the estimates of this chapter.
- The real training curves of main-line pretraining (loss, gradient norm, learning rate) and the real cost (recorded in `runs/ledger.md`).
- The loss spikes, interruptions, and resumes during training, and how we handled them.
- If the estimates and the measurements differ much: where they differ, and why.

---

## Frontier notes

**FP8 training**: starting with the H100, the Tensor Cores support 8-bit floating-point matrix multiplication. In theory, the throughput is two times that of BF16. DeepSeek-V3 trained with an FP8 framework: "fine-grained quantization (one scale for each group of 1 × 128 activations or 128 × 128 weights) + high-precision accumulation". It reports a relative loss error of less than 0.25% against a BF16 baseline. The master weights and the gradients still stay in FP32, and the AdamW moments are stored in BF16. The nanochat code also has an FP8 option. Puro-2B, which GOAL.md 3.1 mentions, also uses FP8. FP8 is sensitive to the scaling strategy and to numerical outliers, and it is much more complex to implement than BF16. Chapter 12 checks if FP8 meets the rule of this course, "clearly adopted by 3 independent families". This chapter and the main line use BF16 by default.

**z-loss and automatic skipping of bad steps**: z-loss limits the softmax normalizer. PaLM introduced it, and OLMo 2 also uses it, with a coefficient of the order of 10⁻⁴. The OLMo 2 training code uses `SkipStepAdamW`. It skips the update of a step when the gradient norm or the loss is more than 6 standard deviations from a moving window. PaLM rolls back by hand and skips data. These methods work, but this chapter found clear adopters in fewer than 3 families. Thus they are not in the default main-line configuration.

---

## Adopters and sources

| Technique | Adopters (at least 3 families) | Sources |
|---|---|---|
| BF16 mixed precision (BF16 computation + FP32 master weights / gradient accumulation) | Llama 3 ("BF16 MFU"; gradients of many micro batches accumulated in FP32, FSDP reduce-scatter in FP32); Nemotron-4 340B (MFU calculated with the bfloat16 peak of 989 TFLOPS); DeepSeek-V3 (BF16 training as the baseline; in the FP8 framework, the master weights and gradients stay in FP32); OLMo 2 (official training script: `param_dtype=bfloat16`, `reduce_dtype=float32`) | [Llama 3 §3.3.2 and Table 4](https://arxiv.org/abs/2407.21783), [Nemotron-4 340B §2.3](https://arxiv.org/abs/2406.11704), [DeepSeek-V3 §3.3 and Appendix B.1](https://arxiv.org/abs/2412.19437), [OLMo-core OLMo-2-0325-32B training script](https://github.com/allenai/OLMo-core/blob/main/src/scripts/official/OLMo2/OLMo-2-0325-32B-train.py) |
| FlashAttention | Industry standard (class B in GOAL.md 2.1): PyTorch `scaled_dot_product_attention` has a built-in FlashAttention backend; the OLMo training code OLMo-core supports a FlashAttention-2 backend, and the OLMo 3 model configuration uses `attn_backend=flash_2` by default (the OLMo 2 report, §3.3.3, also mentions the Flash Attention library); nanochat trains with FlashAttention-3 (and falls back to SDPA when it is not available) | [FlashAttention](https://arxiv.org/abs/2205.14135), [FlashAttention-2](https://arxiv.org/abs/2307.08691), [FlashAttention-3](https://arxiv.org/abs/2407.08608), [OLMo-core `transformer/config.py`](https://github.com/allenai/OLMo-core/blob/main/src/olmo_core/nn/transformer/config.py), [nanochat `flash_attention.py`](https://github.com/karpathy/nanochat/blob/master/nanochat/flash_attention.py) |
| Data parallelism + state sharding (FSDP / ZeRO) | Llama 3 (FSDP shards the optimizer state and the gradients); DeepSeek-V3 (ZeRO-1 data parallelism); Gemma 3 (shards the optimizer state with a ZeRO-3 implementation); Nemotron-4 (distributed optimizer: shards the optimizer state across the data-parallel replicas); OLMo 2 (the official script uses HSDP, which is FSDP in groups) | [Llama 3 §3.3.2](https://arxiv.org/abs/2407.21783), [DeepSeek-V3 §3.2](https://arxiv.org/abs/2412.19437), [Gemma 3 §2.4](https://arxiv.org/abs/2503.19786), [Nemotron-4 340B §2.3](https://arxiv.org/abs/2406.11704), [OLMo-core training script](https://github.com/allenai/OLMo-core/blob/main/src/scripts/official/OLMo2/OLMo-2-0325-32B-train.py) |
| Gradient clipping (norm 1.0) | Llama 2, DeepSeek-V3, OLMo 2, SmolLM3 (verified in Chapter 6); PaLM (global norm clipping at 1.0) | See [Chapter 6, "Adopters and sources"](../06-training-stability/README.md#adopters-and-sources); [PaLM §5](https://arxiv.org/abs/2204.02311) |
| QK-Norm (training stability) | Qwen3 ("to ensure stable training"); Gemma 3; OLMo 2 ("avoids attention logits being too large, which can lead to training loss divergence") | See [Chapter 9, "Adopters and sources"](../09-modern-transformer/README.md#adopters-and-sources); [Qwen3 §2](https://arxiv.org/abs/2505.09388), [Gemma 3 §2](https://arxiv.org/abs/2503.19786), [OLMo 2 §2.1, §3.3.2](https://arxiv.org/abs/2501.00656) |
| Activation checkpointing (recomputation) | PaLM 540B (rematerialization); OLMo 2 (the official 32B script turns on full activation checkpointing); DeepSeek-V3 (recomputes all RMSNorm operations and MLA up-projections in the backward pass). Counterexample: Llama 3 reports that after its optimizations, 8K sequences do not need activation checkpointing | [PaLM §4.1](https://arxiv.org/abs/2204.02311), [OLMo-core training script](https://github.com/allenai/OLMo-core/blob/main/src/scripts/official/OLMo2/OLMo-2-0325-32B-train.py), [DeepSeek-V3 §3.2.3](https://arxiv.org/abs/2412.19437), [Llama 3 §3.3.2](https://arxiv.org/abs/2407.21783) |
| MFU as the efficiency metric | PaLM (introduced it, 46.2%); Llama 3 405B (38–43%); Nemotron-4 340B (41.0–42.4%) | [PaLM §4.1 and Appendix B](https://arxiv.org/abs/2204.02311), [Llama 3 Table 4](https://arxiv.org/abs/2407.21783), [Nemotron-4 Table 2](https://arxiv.org/abs/2406.11704) |

We read the OLMo-core training script at commit `718d08f` (2026-09-26). To be verified: the Qwen3 technical report does not give the training precision or the parallelism method. The SmolLM3 blog gives only the nanotron framework and gradient clipping at 1, with no details about precision and parallelism.

---

## Guided questions

Ask Claude Code these questions. Continue until you can explain the answers in your own words:

1. This chapter says that "FlashAttention calculates QKᵀ one more time, but it is faster". Explain this with "memory-bound": why is more calculation sometimes cheaper than more reads and writes on a GPU? Which operations are compute-bound?
2. Why is the correction factor `exp(m_old − m_new)` of online softmax always ≤ 1? If you do not subtract the maximum and add `exp(x_i)` directly, what problem occurs in BF16? (Hint: the maximum column of `02_precision.py`.)
3. Activation checkpointing makes MFU lower, but it can make training faster. How can both be true? (Hint: why did PaLM turn on recomputation? How is it related to the micro batch?)
4. DDP calculates the mean of the gradients, not the sum. If it used the sum with the same learning rate, what would occur in training? And with 8 GPUs?
5. Section 8 says that "if the number of GPUs changes, zero refuses an exact resume". Why? If you must continue training on 4 GPUs instead of 8, what must you give up, at least?
6. PaLM found that "the batches near a spike do not cause a spike when the training starts from an earlier checkpoint". What does this tell you about the cause of spikes? What does it mean for the method "skip bad data"?

## Hands-on tasks

For each task, run the code and look at the result.

**Task 1 (basic)**: Use `zero/tools/memory_calc.py` to answer these questions. Change the sequence length of the main-line model from 4096 to 32,768 (the long context of Chapter 15). How much memory per GPU does micro batch 1 need with DDP? With activation checkpointing? With FSDP? Which combination fits in 80 GB?

**Task 2 (core)**: Add a "backward pass" to `04_tiled_attention.py`. Use the `lse` that the forward pass stored to calculate P again, block by block. Then calculate `dV = Pᵀ · dO`. Compare it with the dV from PyTorch autograd (`backward` on naive attention), and print the maximum difference. (Advanced: also derive dQ and dK. You need `D_i = Σ_j dO_ij · O_ij`.)

**Task 3 (challenge)**: `07_resume.py` now saves a checkpoint "after each step". Change it to use gradient accumulation (2 micro batches per step), and crash **between the two micro batches**. For a bit-identical resume, what more must the checkpoint store? Why does production code (zero too) usually save checkpoints only after a full step?

---

## Go deeper: CS336

This chapter corresponds to the systems part of [CS336 (Spring 2026)](https://cs336.stanford.edu/). In this area, CS336 goes much deeper than this course:

- **Lecture 5, GPUs**: the memory hierarchy of the GPU (HBM, SRAM), arithmetic intensity and the roofline model, and why attention is memory-bound.
- **Lecture 6, Kernels and Triton**: how to write fused kernels in Triton. There, the tiled loop of this chapter becomes a real GPU program.
- **Lectures 7 and 8, Parallelism**: the communication cost and the combinations of DDP, ZeRO/FSDP, tensor parallelism, pipeline parallelism, and sequence parallelism. These two lectures explain in depth what Section 5.4 of this chapter mentions only briefly.
- **Assignment 2 (Systems)**: implement the forward and backward pass of FlashAttention-2 in Triton. Write DDP by hand (with gradient bucketing and overlap of communication). Implement optimizer state sharding, and measure the performance. `04_tiled_attention.py` and `06_ddp_by_hand.py` of this chapter are small CPU versions of it.

The course page has the notes and the recordings of each lecture.

---

## References

- Micikevicius et al. *Mixed Precision Training*, 2017: <https://arxiv.org/abs/1710.03740>
- Kalamkar et al. *A Study of BFLOAT16 for Deep Learning Training*, 2019: <https://arxiv.org/abs/1905.12322>
- Milakov & Gimelshein. *Online normalizer calculation for softmax*, 2018: <https://arxiv.org/abs/1805.02867>
- Dao et al. *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*, 2022: <https://arxiv.org/abs/2205.14135>
- Dao. *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning*, 2023: <https://arxiv.org/abs/2307.08691>
- Shah et al. *FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision*, 2024: <https://arxiv.org/abs/2407.08608>
- Chen et al. *Training Deep Nets with Sublinear Memory Cost* (activation checkpointing), 2016: <https://arxiv.org/abs/1604.06174>
- Korthikanti et al. *Reducing Activation Recomputation in Large Transformer Models*, 2022: <https://arxiv.org/abs/2205.05198>
- Rajbhandari et al. *ZeRO: Memory Optimizations Toward Training Trillion Parameter Models*, 2019: <https://arxiv.org/abs/1910.02054>
- Zhao et al. *PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel*, 2023: <https://arxiv.org/abs/2304.11277>
- Shoeybi et al. *Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism*, 2019: <https://arxiv.org/abs/1909.08053>
- Chowdhery et al. *PaLM: Scaling Language Modeling with Pathways* (definition of MFU, rollback and skipping for loss spikes, z-loss, bitwise reproducibility), 2022: <https://arxiv.org/abs/2204.02311>
- Llama Team. *The Llama 3 Herd of Models* (4D parallelism, BF16 MFU, interruption statistics), 2024: <https://arxiv.org/abs/2407.21783>
- NVIDIA. *Nemotron-4 340B Technical Report* (H100 BF16 peak, MFU, distributed optimizer), 2024: <https://arxiv.org/abs/2406.11704>
- DeepSeek-AI. *DeepSeek-V3 Technical Report* (FP8 training, ZeRO-1, no rollbacks), 2024: <https://arxiv.org/abs/2412.19437>
- OLMo Team. *2 OLMo 2 Furious* (Section 3: pretraining stability), 2024: <https://arxiv.org/abs/2501.00656>; training code OLMo-core: <https://github.com/allenai/OLMo-core>
- Gemma Team. *Gemma 3 Technical Report*, 2025: <https://arxiv.org/abs/2503.19786>
- Wortsman et al. *Small-scale proxies for large-scale Transformer training instabilities*, 2023: <https://arxiv.org/abs/2309.14322>
- Karpathy. nanochat (single-node, readable training code; a design reference for zero in this course): <https://github.com/karpathy/nanochat>
- CS336 (Spring 2026): <https://cs336.stanford.edu/>

**Next chapter**: Pretraining puts the "general knowledge" of about 400B tokens into the model, and the learning rate stays in the stable phase all the time. Chapter 15 does two things. First, it anneals the learning rate with high-quality data (mid-training). Then it extends the context from 4K to 32K. The result is the Base model of the main line, and gate 2 checks if this model meets the expectations.
