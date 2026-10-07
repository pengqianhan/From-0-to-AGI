# Chapter 8: Attention — Each position decides where to look

**English** · [中文](README.zh.md)

> **Goal**: After this chapter, you can start from "a weighted average of the earlier words" and derive scaled dot-product attention `softmax(QKᵀ/√d)V` step by step. You can explain which problem each part solves: Q, K, V, the division by √d, the causal mask, and multiple heads. You can write the tensor shape after each step. You can also implement causal multi-head attention yourself, with the same output as the official PyTorch function.

📺 **Video**: Not published yet. To render it on your computer, run `bash chapters/08-attention/video/build.sh`.
🧪 **Self-check**: After the chapter, type `/ch08-attention` in Claude Code.

---

In the last chapter, we changed text into tokens and trained a bigram language model. We used bits-per-byte to measure how well it predicts. A bigram model has a basic limit: **to predict the next token, it looks only at the previous token**. This chapter solves this problem: **how can the model see all of the earlier text, and decide by itself where to look?** The answer is attention. Attention is the core part of every large language model today.

## 1. The problem: the previous token is not sufficient

To predict what comes after `hear me speak`, the last letter `k` alone is not sufficient. The bigram model of Chapter 7 remembers only "what often comes after k". But the earlier words `hear me` tell us that a person speaks and asks for something.

The difficulty is this: the length of the earlier text (the context) is not fixed. But the prediction layer after it (the softmax classifier of Chapter 5) accepts only a vector of fixed length. How do we change a context of any length into one vector?

**The earlier method: the recurrent neural network (RNN).** An RNN reads from left to right. At each token, it combines the token with the state of the last step into a new state: `h_t = f(h_{t−1}, x_t)`. Here we use the RNN only as a motivation. It has two problems, and attention solves both of them:

1. **The state has a fixed size.** All of the context, however long, must go into an `h` of the same size. Early information is easily lost. In machine translation, Bahdanau et al. (2014) showed that a fixed-length vector for the full sentence is a bottleneck.
2. **It calculates only one step at a time.** Token 10 must wait until the first 9 tokens are done. *Attention Is All You Need* (Vaswani et al., 2017) says at its start that this "inherently sequential nature" prevents parallel training within a sequence.

Attention uses a different idea: **do not compress. Each position looks directly at all earlier positions and takes what it needs.**

## 2. The simplest method: average the earlier vectors

Start with the simplest thing. We have T tokens. An embedding lookup has changed each token into a C-dimensional vector. Put these vectors in the rows of a matrix `X` (shape `(T, C)`). Let the output at position t be the mean of the first t+1 vectors (position t itself is included, future positions are not):

```
out_t = (x_0 + x_1 + … + x_t) / (t + 1)
```

A classic trick does this for all positions. Put the averaging coefficients in a **lower-triangular matrix** `W`. In row t, the first t+1 entries are `1/(t+1)`, and the other entries are 0. Then one matrix multiplication `W @ X` gives the averages at all positions at the same time. Chapter 2 showed that matrix multiplication is "a dot product of each row with each column". Here, row t of `W` is exactly "take 1/(t+1) of each of the first t+1 vectors".

```bash
uv run python chapters/08-attention/code/01_average_to_attention.py
```

[`code/01_average_to_attention.py`](code/01_average_to_attention.py) shows this with 5 vectors of 2 dimensions. We pretend that they are the Chinese tokens "我 爱 吃 苹 果" ("I love eating apples"):

```python
def uniform_weights(T):
    w = np.tril(np.ones((T, T)))                 # lower triangle of ones
    return w / w.sum(axis=1, keepdims=True)      # divide each row by its sum → each row sums to 1

mat = uniform_weights(T) @ x                     # one matrix multiplication = prefix averages at all positions
```

The output is:

```
① Uniform weight matrix W (lower-triangular, each row sums to 1):
[[1.   0.   0.   0.   0.  ]
 [0.5  0.5  0.   0.   0.  ]
 [0.33 0.33 0.33 0.   0.  ]
 [0.25 0.25 0.25 0.25 0.  ]
 [0.2  0.2  0.2  0.2  0.2 ]]
② Max difference between W @ x and the loop version: 2.0e-17
③ Max difference between softmax(all-zero scores + causal mask) and W: 0.0e+00
```

Look again at line ③. The same `W` also comes from a different method.

First, give each pair of positions a score (here, all scores are 0). Fill the scores in the upper triangle (the future) with `−∞`. Then apply the softmax of Chapter 5 to each row. Because `e^{−∞} = 0`, the weights of the future positions are exactly 0. The other positions have the same score, so they share the weight equally.

**This skeleton, "scores → mask → softmax → weighted average", is the full structure of attention.** Only one question is left: how do we calculate the scores?

## 3. Let the data set the weights: dot-product similarity

The problem with the average is that all tokens have the same importance. In real text, some tokens are important and some are not. So let the data set the scores: when two positions are more "related", their score is higher.

Chapter 2 gave the geometric meaning of the dot product: when two vectors point in more similar directions, their dot product is larger. So the most direct score is `x_t · x_s`. For all pairs at once, the scores are `X @ Xᵀ` (shape `(T, T)`):

```python
def dot_product_weights(x):
    scores = x @ x.T                   # scores[t, s] = x_t · x_s
    return causal_softmax(scores)      # fill the upper triangle with −∞, then softmax on each row
```

The output (the same 5 vectors) is:

```
   Weights after causal mask + softmax (each row sums to 1):
[[1.   0.   0.   0.   0.  ]
 [0.41 0.59 0.   0.   0.  ]
 [0.28 0.23 0.48 0.   0.  ]
 [0.06 0.14 0.04 0.76 0.  ]
 [0.1  0.05 0.09 0.01 0.75]]
   5/5 positions give the largest weight to themselves: x·x = |x|² is often the largest, so we need two different projections, Q and K
```

The weights are not uniform now, so this step goes in the correct direction. But a new problem occurs: **each position gives the most attention to itself**. The reason is `x·x = |x|²`: a vector always points in exactly the same direction as itself. But to predict the next token, we often need a token that is "different from me, but useful to me".

## 4. Q, K, V: one input, three roles

The solution is to give each token three different vectors. Each vector comes from the linear transformation `y = XW` of Chapter 2:

```
Q = X W_q     query: what am I looking for?
K = X W_k     key: which features do I have, for others to match?
V = X W_v     value: what do I give you if you attend to me?
```

The score of position t for position s becomes **the dot product of the query of t and the key of s**, `q_t · k_s`. The weighted average also changes: it averages `v`, not `x`. `W_q`, `W_k`, and `W_v` are parameters that training learns (shape `(C, C)`). The model can learn matching rules such as "a vowel looks for the consonant before it" or "a letter after a space looks for the previous word". It no longer finds only itself.

An analogy: Q is the keyword that you type in a search box. K is the tag of each article, and V is the text of the article. When the keyword and a tag match better, you take more content from that article. But here the "match" is soft: you take a little from every article, and softmax sets the proportions.

## 5. Scaled dot-product attention, and why we divide by √d

Calculate all queries against all keys at the same time: the result is `Q Kᵀ`. Then the full attention fits in one line (Vaswani et al. 2017, Equation 1):

```
Attention(Q, K, V) = softmax( Q Kᵀ / √d + mask ) V
```

`d` is the dimension of q and k. The code has only a few lines ([`code/02_attention_from_scratch.py`](code/02_attention_from_scratch.py)):

```python
def attention(q, k, v, causal=True):
    d = q.shape[-1]
    scores = q @ k.transpose(-2, -1) / math.sqrt(d)            # QKᵀ / √d
    if causal:
        T = q.shape[-2]
        future = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(future, float("-inf"))      # future positions → −∞
    weights = torch.softmax(scores, dim=-1)                     # each row sums to 1
    return weights @ v, weights                                 # weighted average of V
```

**Why do we divide by √d?** The original paper gives this explanation. Assume that each component of q and k is an independent random number with mean 0 and variance 1. Then `q·k = Σ q_i k_i` is a sum of d terms with variance 1, so its variance is d. When d is larger, the scores spread farther apart, and softmax comes closer to one-hot (it looks at almost only one position). The gradient also becomes smaller. Division by √d brings the variance back to 1.

[`code/03_why_sqrt_d.py`](code/03_why_sqrt_d.py) measures this directly. Each query has 16 keys, and the test runs 2000 times.

```bash
uv run python chapters/08-attention/code/03_why_sqrt_d.py
```

| d | No scaling: score variance | Max weight | Effective count | Gradient size | With scaling: score variance | Max weight | Effective count | Gradient size |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 15.9 | 0.672 | 2.91 | 0.285 | 1.0 | 0.241 | 10.93 | 0.303 |
| 64 | 64.7 | 0.842 | 1.66 | 0.176 | 1.0 | 0.243 | 10.81 | 0.304 |
| 256 | 254.4 | 0.921 | 1.29 | 0.102 | 1.0 | 0.248 | 10.73 | 0.305 |
| 1024 | 1026.8 | 0.957 | 1.14 | 0.058 | 1.0 | 0.245 | 10.75 | 0.305 |

- The "effective count" is `e^entropy`. It is 16 when the weights are spread evenly over 16 positions, and 1 when all the weight is on one position.
- The "gradient size" is the Frobenius norm of the softmax Jacobian `diag(p) − ppᵀ`. It measures how much of the gradient on the scores can pass through softmax.

Without scaling, the score variance at d = 1024 is about 1027, which agrees with the theoretical value d. The max weight is 0.957 and the effective count is 1.14: attention looks at almost only one position. The gradient size is only about one fifth of the value with scaling. With scaling, the numbers for the four values of d are almost the same. **Division by √d makes the "softness" of attention independent of the head dimension.** In real models, head_dim is usually 64 or 128 (for example, gpt-oss uses 64 dimensions per head, and DeepSeek-V3 uses 128 dimensions per head). Thus this step is necessary.

## 6. Causal mask: do not look at the answer

The task of a language model is to predict the next token from the earlier text. In training, we give the full sequence to the model at one time, and each position predicts its next token at the same time (Chapter 7). If position t can see position t+1, it can "copy the answer" directly. So each position can see only itself and the earlier positions.

We fill the upper triangle of the score matrix with `−∞`. This is the **causal mask**. The original paper describes it as "masking out (setting to −∞) all values in the input of the softmax which correspond to illegal connections".

`02_attention_from_scratch.py` does a causality test. It replaces the inputs at the last 3 positions of a sequence with other random numbers. The outputs at the first 5 positions must not change at all.

```
New inputs at positions 5–7: max output change at positions 0–4: 0.0e+00; at positions 5–7: 0.35
```

## 7. Multi-head attention: several views at the same time

One set of attention weights can express only one "view": each position distributes its attention by one rule. But a token can need two kinds of information at the same time, for example "what is the previous character" and "where does this word start". **Multi-head attention** splits the C dimensions into H parts of `d = C / H` dimensions. Each part does attention independently. Then we concatenate the parts and multiply the result by an output matrix `W_o`:

```python
q = q.view(B, T, self.H, self.d).transpose(1, 2)    # (B, T, C) → (B, H, T, d)
out, w = attention(q, k, v, causal=True)            # each head calculates on its own
out = out.transpose(1, 2).reshape(B, T, C)          # concatenate the heads again
out = self.wo(out)                                  # output projection
```

Move the "head" dimension to the front and treat it as part of the batch. Then all heads run in parallel in one matrix multiplication. The table gives the shape after each step (B = 2, T = 8, C = 32, H = 4, as the script prints them):

| Step | Shape | Meaning |
|---|---|---|
| Input `x` | (2, 8, 32) | (B, T, C) |
| `q = x @ Wq` | (2, 8, 32) | (B, T, C) |
| Split into heads | (2, 4, 8, 8) | (B, H, T, d) |
| Weights `softmax(QKᵀ/√d)` | (2, 4, 8, 8) | (B, H, T, T) |
| Output of each head `w @ v` | (2, 4, 8, 8) | (B, H, T, d) |
| Concatenate the heads | (2, 8, 32) | (B, T, C) |
| Output `@ Wo` | (2, 8, 32) | (B, T, C) |

(Here T and d are both 8. Do not mix them up: the weights are T × T, and the output of each head is T × d.)

The number of parameters is `4 × C² = 4096` (Wq, Wk, Wv, and Wo are each C × C). **It does not depend on the number of heads.** Multiple heads do not add parameters; they only organize the parameters in a different way. The original paper uses 8 heads with 64 dimensions per head.

## 8. Parity check: the same output as the official PyTorch implementation

```bash
uv run python chapters/08-attention/code/02_attention_from_scratch.py
```

The script gives the same weights to our handwritten `attention` and to PyTorch's `F.scaled_dot_product_attention(q, k, v, is_causal=True)`:

```
Max difference from F.scaled_dot_product_attention(is_causal=True): 9.7e-08
Parameters: 4096 = 4 × C² = 4 × 32² (Wq, Wk, Wv, Wo; does not depend on the number of heads)
```

The difference is of the order of 10⁻⁷. This is float32 rounding error. Now we know that the two implementations agree, so the later code can use the official function with confidence. On the GPU, the official function also has much faster implementations (see "From minimal code to production code").

## 9. Train a one-layer attention model and see where it looks

We have derived the formulas. Now we train a real model. [`code/04_train_attention.py`](code/04_train_attention.py) trains three character-level models on Tiny Shakespeare (`assets/tiny_corpus/shakespeare.txt`, about 1.1 MB, 65 distinct characters, all ASCII, so 1 character = 1 byte).

The models differ in only one step: how they mix the earlier text. All other parts are the same. The models use the same embedding (plus a learnable position vector; Chapter 9 replaces it with RoPE) and the same output layer. They also use the same AdamW, the same 2000 steps, and the same data order.

```python
h = self.tok(idx) + self.pos(torch.arange(T))
if self.mode == "attention":
    out, w = self.mix(h, return_weights=True)      # MultiHeadAttention from this chapter
    h = h + out                                    # residual connection (Chapter 6)
elif self.mode == "average":
    W = torch.tril(torch.ones(T, T))
    W = W / W.sum(1, keepdim=True)                 # fixed uniform weights
    h = h + self.wo(W @ self.wv(h))
logits = self.head(h)
```

```bash
uv run python chapters/08-attention/code/04_train_attention.py     # about 1 minute on one CPU thread
```

(The script has the line `torch.set_num_threads(1)`. For a model this small, the overhead of thread scheduling is larger than the computation itself. On our build machine, when the machine was busy, each step with 4 threads was about 100 times slower than with one thread.)

| Model | Parameters | Validation loss (nats/character) | bits-per-byte |
|---|---:|---:|---:|
| bigram (only the current character + position) | 12,481 | 2.487 | 3.588 |
| uniform average (prefix average) | 20,673 | 2.463 | 3.553 |
| attention (4 heads, 16 dimensions per head) | 28,865 | 2.083 | 3.005 |

Bits-per-byte is the metric of Chapter 7: the loss divided by ln 2 (here 1 character = 1 byte). Look at these results:

- **The uniform average helps almost nothing.** It adds 8192 parameters, but the loss decreases only from 2.487 to 2.463. The model can see the context, but every character has the same weight. The model cannot pick the useful information out of an even mix of all characters.
- **Attention decreases the loss to 2.083.** It adds only 8192 more parameters, and it uses 0.58 fewer bits per byte. To see the context is not sufficient. **The important thing is to select.**
- At step 2000, the loss of the attention model still decreases (2.111 at step 1500 → 2.083 at step 2000). More training would make it better. We stop here only so that the script finishes in one minute.

**Where does it really look?** On the validation set, the script measures the mean fraction of weight that each head puts on the position "k steps back":

| Head | Self (k=0) | 1 back | 2 back | Earlier (k≥3) |
|---|---:|---:|---:|---:|
| 0 | 0.10 | 0.17 | 0.50 | 0.23 |
| 1 | 0.21 | 0.62 | 0.09 | 0.08 |
| 2 | 0.14 | 0.32 | 0.26 | 0.28 |
| 3 | 0.23 | 0.57 | 0.10 | 0.10 |

The four heads divide the work among themselves. Heads 1 and 3 look mainly at the previous character. Head 0 looks mainly at the character 2 positions back. Head 2 spreads its weight the most.

Nobody told the model to do this. Together, the heads make a model that "looks at the previous two or three characters" (a little like a trigram). The weights of this model also change with the content.

The script also draws each attention matrix as a character heat map. Below is an excerpt for head 1 on the last part of `First Citizen:\nBefore we proceed any further, hear me speak.` A row is the current character, and a column is the attended character. A denser symbol means a larger weight: ` .:-=+*#%@` stand for 0–0.1, 0.1–0.2, …, 0.9–1.0. `␣` is a space.

```
    Before␣we␣proceed␣any␣further,␣hear␣me␣speak.
  ,                             @
  ␣                               @
  h                               @
  e                               :+.
  a                                 @
  r                                  @
  ␣                                    %
  m                                    @
  e                                     @
  ␣                                       @
  s                                       @
  p                                        #.
  e                                       . #
  a                                        -.-.
  k                                          =+
  .                                           .#
```

In most rows, the `@` is one column left of the diagonal: head 1 is a "look at the previous character" head. But it does not use a rigid fixed offset. For each position, the script prints the character with the largest weight (`→k back` means k positions back):

```
Head 1 on ', hear me speak.': the character with the largest weight at each position:
  ,→1 back 'r' 1.00   ␣→0 back '␣' 0.94   h→1 back '␣' 0.96   e→1 back 'h' 0.60
  a→1 back 'e' 0.90   r→1 back 'a' 0.99   ␣→0 back '␣' 0.89   m→1 back '␣' 0.99
  e→1 back 'm' 0.99   ␣→0 back '␣' 0.93   s→1 back '␣' 1.00   p→1 back 's' 0.79
  e→1 back 'p' 0.70   a→1 back 'e' 0.37   k→1 back 'a' 0.50   .→1 back 'k' 0.76

Head 0 on ', hear me speak.': the character with the largest weight at each position:
  ,→43 back 'i' 0.89   ␣→7 back 'u' 0.28   h→2 back ',' 0.66   e→2 back '␣' 0.50
  a→2 back 'h' 0.38   r→0 back 'r' 0.36   ␣→2 back 'a' 0.27   m→3 back 'a' 0.61
  e→1 back 'm' 0.83   ␣→6 back 'e' 0.27   s→2 back 'e' 0.54   p→0 back 'p' 0.77
  e→2 back 's' 0.87   a→2 back 'p' 0.99   k→1 back 'a' 0.61   .→56 back 's' 0.32
```

- **The rule of head 1 changes with the content.** On letters, it looks almost only at the previous character (`r→a` 0.99, `m→␣` 0.99). But at a **space**, it looks at the space itself (0.94, 0.89, 0.93). The head is the same and the relative position is the same, but the weights are different. This is exactly what "the data sets the weights" means.
- **Head 0 mostly looks 2 characters back** (`e→s` 0.87, `a→p` 0.99). But on the comma and the period, it puts most of its weight near the start of the sequence. For the comma, the top weight (0.89) is on the `i` in `First`, 43 characters back. At punctuation, the head seems to have nothing useful to take, so it puts the weight on the start. This looks similar to the attention sink in "Frontier notes". But we did not examine it further on this small model.

In the video, purple heat maps show the full matrices of heads 1 and 0 on `further, hear me speak.`.

We must also state the limits of this model. It has one layer, no feed-forward network, and only one minute of training. It cannot learn grammar, and it does not understand the text. It found only one fact: nearby characters are the most useful. In Chapter 9, we put attention into a full Transformer block and stack many layers. Only then do more complex patterns appear.

## 10. Summary

- **Problem**: A bigram model looks only at the previous token. An RNN compresses the context into a state of fixed size, and it can only calculate in sequence.
- **Skeleton**: The output is a weighted average of the context vectors. One lower-triangular matrix multiplication calculates all positions at once. The steps are: scores → causal mask (−∞ in the upper triangle) → softmax → weighted average.
- **Q, K, V**: Three linear projections. The score is the dot product of a query and a key. The average is over the values.
- **Scaling**: The variance of `q·k` is about d. After division by √d, it is about 1, and softmax no longer becomes one-hot.
- **Multi-head**: Split C into H parts and run attention on each part. Concatenate the results and multiply by `W_o`. The shapes are `(B,T,C) → (B,H,T,d) → (B,H,T,T) → (B,T,C)`. The parameter count 4C² does not depend on the number of heads.
- **Measurements**: The max difference between the from-scratch implementation and the official SDPA is 9.7e-08. One attention layer decreases the validation loss from 2.487 (bigram) to 2.083. The uniform average reaches only 2.463.

---

## GPU measurements (one RTX 3090)

> All numbers in the text above come from CPU runs. In this section, we measure on one NVIDIA GeForce RTX 3090 (24 GB of GPU memory, Ampere architecture). Its data sheet gives a dense BF16 tensor-core peak of about 71 TFLOPS, FP32 of about 35.6 TFLOPS, and a memory bandwidth of about 936 GB/s.
>
> Environment: PyTorch 2.11.0+cu128, CUDA 12.8, October 2026. The server sets the power limit of this card to 240 W (the factory default is 350 W). Under a continuous full load, the card lowers its clock. Thus the absolute compute and bandwidth are lower than on a 3090 at full power, and the relative values are more reliable. If you do not have a GPU, skip this section.

Run:

```bash
uv run python chapters/08-attention/code/06_gpu_sdpa_backends.py
```

The shapes are those of the attention in the main-line model: batch 1, 16 query heads, head_dim 128, BF16, causal mask. "Handwritten" is the `attention` function from Section 5, moved to the GPU without changes. The other three columns use `F.scaled_dot_product_attention`, with `sdpa_kernel` locked to the math, efficient, or flash backend. The times are medians of CUDA event timing (milliseconds):

| T | Handwritten `attention` | SDPA math | SDPA efficient | SDPA flash |
|---:|---:|---:|---:|---:|
| 512 | 0.22 | 0.50 | 0.09 | 0.08 |
| 1K | 0.59 | 1.59 | 0.23 | 0.18 |
| 2K | 2.56 | 6.74 | 0.70 | 0.42 |
| 4K | 11.06 | 28.19 | 1.73 | 1.36 |
| 8K | 44.36 | 110.76 | 6.60 | 4.96 |
| 16K | 170.74 | out of memory | 29.21 | 23.50 |
| 32K | out of memory | out of memory | 119.55 | 102.95 |

The next table gives the peak extra GPU memory in the same run (MiB). This is the highest memory use during the call, minus the memory use before the call; it does not include q, k, v. The second column is a reference: the size of the T × T score matrices of 16 heads in BF16.

| T | T × T score matrices | Handwritten `attention` | SDPA math | SDPA efficient | SDPA flash |
|---:|---:|---:|---:|---:|---:|
| 512 | 8 | 18 | 53 | 2 | 2 |
| 1K | 32 | 69 | 180 | 4 | 4 |
| 2K | 128 | 268 | 656 | 8 | 8 |
| 4K | 512 | 1,056 | 2,496 | 16 | 16 |
| 8K | 2,048 | 4,192 | 9,728 | 32 | 33 |
| 16K | 8,192 | 16,704 | out of memory | 64 | 65 |
| 32K | 32,768 | out of memory | out of memory | 128 | 130 |

At the end, the script changes to a GQA shape (16 query heads share 8 K/V groups, T = 4096) and sets `enable_gqa=True`. Flash runs. Its max difference from "first copy K/V to 16 heads, then calculate" is 7.8e-03 (BF16 rounding level). Its extra memory is 16 MiB, but the version that copies K/V first needs 48 MiB.

Efficient does not run (`No available kernel`; the reason from PyTorch is that it needs the same number of heads for q, k, and v). Math runs. With no backend given, the default for both MHA and GQA is flash.

The second table shows the real size of the `T × T` term from guided question 2. The extra memory of the handwritten version is almost exactly two times the score matrix. It keeps one copy of `QKᵀ/√d` and one copy of the softmax result. When T doubles, this memory becomes 4 times larger. At 16K, it is 16.3 GiB. At this rate, 32K needs 64 GiB, which does not fit on a 24 GB card.

Efficient and flash use only a little more memory, about the size of the output (64 MiB at 16K). This memory grows linearly with T. The reason is that they calculate in blocks and never write the T × T matrix to GPU memory.

But FlashAttention does not calculate less. When T doubles, the time of flash still grows about 4 times, because the computation is still T². At 16K, flash is more than 7 times faster than the handwritten version (23.5 ms vs 170.7 ms). It saves the round trip: writing the T × T matrix to GPU memory and reading it back (Chapter 14 explains how).

A surprise: the SDPA math backend is slower than our handwritten version, and it uses more memory. At 4K, math needs 28 ms and 2.4 GiB, and the handwritten version needs 11 ms and 1.0 GiB. Math first converts the BF16 inputs to float32, and this gives more precision. In the parity check at the start of the script, math and flash both have a max difference of 7.3e-03 from the float32 reference. The handwritten BF16 version has 1.6e-02.

The table below says that "SDPA automatically selects FlashAttention-2 on CUDA". On this card with BF16, this statement is true, and GQA also uses flash with no real copy of K/V.

## From minimal code to production code

The attention of the main-line model is the `Attention` class in [`zero/model.py`](../../zero/model.py). Its skeleton is the same as the `MultiHeadAttention` of this chapter: three projections `wq`, `wk`, `wv`, scaled dot product, causal mask, multiple heads, and the output projection `wo`. The parameter names are also the same. This is the core of `Attention.forward` (without the transposes, and with a few lines merged):

```python
q = self.q_norm(self.wq(x).view(bsz, seqlen, self.n_heads, self.head_dim))      # QK-Norm
k = self.k_norm(self.wk(x).view(bsz, seqlen, self.n_kv_heads, self.head_dim))   # GQA: fewer K/V heads
v = self.wv(x).view(bsz, seqlen, self.n_kv_heads, self.head_dim)
q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)                          # RoPE
if kv_cache is not None:
    k, v = kv_cache.update(layer_idx, start_pos, k, v)                           # KV cache
out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=is_causal,
                                     enable_gqa=self.n_kv_heads != self.n_heads)
return self.wo(out.reshape(bsz, seqlen, self.n_heads * self.head_dim))
```

| Minimal code (this chapter) | Production code (`Attention` in `zero/model.py`) | Why |
|---|---|---|
| Three `nn.Linear(C, C)`: `wq`, `wk`, `wv` | Also three separate projections (not fused into one large matrix), but the output of `wk` and `wv` is `kv_dim = n_kv_heads × head_dim`. No bias anywhere | The parameter names and shapes match Qwen3 in Hugging Face one to one, so we can load the official weights directly for a parity check. No bias: Qwen3 removed the QKV bias of Qwen2, and OLMo 2 uses no bias at all |
| H heads, `d = C/H` per head | `n_heads` query heads and `n_kv_heads` K/V heads. `head_dim` can be different from `dim / n_heads` | **GQA** (grouped-query attention): several query heads share one K/V group, so the KV cache becomes smaller in proportion. See Chapter 10 |
| Uses q and k directly | `q_norm`, `k_norm`: RMSNorm on the q and k of each head, over head_dim | **QK-Norm**: prevents attention scores that are too large and make training diverge (Qwen3, OLMo 2, and Gemma 3 use it). See Chapter 9 |
| A learnable position vector, added to the input (in `04`) | `apply_rope`: rotary position embedding on q and k | **RoPE**: the position information goes directly into q·k and depends only on the relative distance. See Chapter 9 |
| Handwritten `softmax(QKᵀ/√d)` + `masked_fill` | `F.scaled_dot_product_attention(..., is_causal=..., enable_gqa=...)` | On CUDA with BF16/FP16, SDPA automatically selects the FlashAttention-2 kernel, which does not write the T × T weight matrix to GPU memory. `enable_gqa` lets SDPA broadcast the K/V heads itself, with no copy. (The memory-efficient kernel does not support GQA, so GQA can only use Flash. With FP32, SDPA falls back to the math backend.) Verified on an RTX 3090: BF16 + `enable_gqa=True` uses Flash by default (see "GPU measurements" in this chapter and Section 1 of [runs/2026-10-01-gpu0-check](../../runs/2026-10-01-gpu0-check/README.md)). On the CPU, SDPA uses the PyTorch C++ reference implementation |
| Processes the full sequence at one time | Optional `kv_cache`: during inference, it stores the K/V that it has already calculated. With history, it builds the mask from `start_pos` (one new token needs no mask; chunked prefill uses an explicit Boolean mask) | During generation, each step calculates q/k/v only for the new token. See Chapter 10 |

**Parity check**: [`code/05_zero_parity.py`](code/05_zero_parity.py) loads the weights of this chapter's `MultiHeadAttention` without changes into `zero.model.Attention`. It turns off QK-Norm and passes `cos = 1, sin = 0`, which makes RoPE the identity transformation. Then it turns on GQA and compares it with a manual copy of K/V:

```bash
uv run python chapters/08-attention/code/05_zero_parity.py
```

```
① Minimal MultiHeadAttention vs zero.model.Attention (QK-Norm and RoPE off), max difference: 8.9e-08
② GQA Wk shape (16, 32) (MHA: (32, 32)): the K/V projections and the KV cache are half the size
   zero GQA vs manual "copy K/V, then multi-head attention", max difference: 8.9e-08
   Attention parameters: MHA 4096, GQA 3072
```

For the full model, [`tests/test_model_hf_parity.py`](../../tests/test_model_hf_parity.py) randomly initializes a small Hugging Face `Qwen3ForCausalLM`. It uses GQA with 4 query heads / 2 K/V heads, QK-Norm, and RoPE. The test also covers no GQA, YaRN, and `head_dim ≠ dim / n_heads`. The test moves the weights into `zero` and requires the logits to agree within 1e-5 (`uv run pytest tests/test_model_hf_parity.py`). Thus "the minimal version of this chapter ↔ the zero `Attention` ↔ the official Qwen3 implementation" is a chain in which each link agrees.

---

## Frontier notes

**Attention sink / a learnable bias in the softmax denominator**: The standard softmax forces the weights of each row to sum to 1, even when the current position does not need to look at anything. gpt-oss adds a learnable bias to the softmax denominator of each head, so attention can "look at no token" (model card §2.2, which cites "Attention is off by one" and the work on attention sinks). The reports of the other families that this chapter checked (Qwen3, Llama 3, OLMo 2, Gemma 3, DeepSeek-V3) do not include this item. So we only mention it here, and the main text still uses the standard softmax.

---

## Adopters and sources

Multi-head scaled dot-product causal attention is a basic part of all decoder-only large models. The models differ only in the variants (GQA, MLA, alternating local/global attention, and others; see Chapters 10 and 21–23). All the items below are explicit statements in technical reports or model cards.

| Technique | Adopters | Source |
|---|---|---|
| Scaled dot-product attention `softmax(QKᵀ/√d_k)V`, multiple heads + output projection, −∞ to mask future positions in the decoder | Proposed by: Transformer (h = 8, d_k = 64) | [Vaswani et al. 2017 §3.2](https://arxiv.org/abs/1706.03762) |
| Multi-head causal self-attention (GQA form) | Qwen3 ("Grouped Query Attention"; for example, Qwen3-0.6B has 16 query heads / 8 KV heads); Llama 3 ("standard, dense Transformer", GQA with 8 KV heads, 32 attention heads for 8B); Gemma 3 ("decoder-only transformer … Grouped-Query Attention"); gpt-oss (64 query heads of 64 dimensions per layer, GQA with 8 KV heads) | [Qwen3 §2 Table 1](https://arxiv.org/abs/2505.09388), [Llama 3 §3.2 Table 3](https://arxiv.org/abs/2407.21783), [Gemma 3 §2](https://arxiv.org/abs/2503.19786), [gpt-oss model card §2.2](https://arxiv.org/abs/2508.10925) |
| Multi-head causal self-attention (standard MHA form) | OLMo 2 7B / 13B (32/32 and 40/40 "MHA"; 32B changes to GQA 40/8) | [OLMo 2 §2.1 Table 3](https://arxiv.org/abs/2501.00656) |
| Multi-head causal self-attention (MLA form) | DeepSeek-V3: 128 heads, 128 dimensions per head. The output is still `Σ_j softmax_j(q·k/√(d_h + d_h^R)) v`, multiplied by `W_O` | [DeepSeek-V3 §2.1.1 Eq. (10)(11), §4.2](https://arxiv.org/abs/2412.19437) |
| QK-Norm (only a preview in this chapter) | Qwen3 ("introduce QK-Norm … to ensure stable training"); OLMo 2; Gemma 3 ("replace the soft-capping of Gemma 2 with QK-norm") | Same as above |
| FlashAttention (used through SDPA; industry standard) | gpt-oss ("leverage the Flash Attention algorithms"); PyTorch SDPA automatically selects FlashAttention-2 on CUDA | [gpt-oss model card §2.4](https://arxiv.org/abs/2508.10925), [Dao et al. 2022](https://arxiv.org/abs/2205.14135), PyTorch `scaled_dot_product_attention` documentation |

To be verified: none.

---

## Guided questions

Ask Claude Code these questions. Continue until you can explain the answers in your own words:

1. Train a language model without the causal mask. What happens to the training loss? What happens at validation (when the model really generates one token at a time)? Why? You can ask Claude Code to add a switch to `04_train_attention.py` and try it.
2. The computation of attention has a `T × T` term. When the context grows from 4K to 128K, how many times larger does this term become? How is this related to FlashAttention in Chapter 14, and to sliding windows and linear attention in Chapters 22–23?
3. Attention itself does not "feel" the order of its inputs. If you shuffle the context, does the same query get the same weighted average? Then how does the model know the order? (What does `self.pos` in `04` do? Why is RoPE in Chapter 9 better?)
4. Why does the parameter count of multi-head attention not depend on the number of heads? What happens if there are so many heads that each head has only 1 dimension? (Hint: row (A) of Table 3 in the original paper did this experiment.)
5. In `05_zero_parity.py`, GQA reduces the K/V heads from 4 to 2. Which parameters does this save? What does it save during inference? (Chapter 10 does this calculation.)

## Hands-on tasks

For each task, run the code and look at the result.

**Task 1 (basic)**: In `01_average_to_attention.py`, change the dot-product score. First multiply `x` by two different random matrices `Wq` and `Wk`, then take the dot product (`(x @ Wq) @ (x @ Wk).T`). Count again how many positions attend most to themselves. Did the result change? Which advantage of separate Q and K does this show?

**Task 2 (core)**: In the `attention` function of `02_attention_from_scratch.py`, remove `/ math.sqrt(d)`. Change `C` to 256 and `H` to 1 (one head with 256 dimensions). Print the weight matrix of the first head. About how large is the largest weight in each row? Does it agree with the table in Section 5?

**Task 3 (challenge)**: In `04_train_attention.py`, add a fourth mode `"no_mask"`: attention without the causal mask. After training, compare its training loss and validation loss (do not use the mask at validation either). Then write a function that really generates text one character at a time. Compare the text that the `attention` model and the `no_mask` model generate. Does the model with the lower loss generate better text? Why?

---

## Go deeper: CS336

This chapter corresponds to Stanford [CS336: Language Modeling from Scratch](https://cs336.stanford.edu/) (Spring 2026):

- **Lecture 3: Architectures and hyperparameters**: from the original Transformer to the architecture choices of modern large models. These choices include the attention variants (MHA, GQA, and others), normalization, and position encoding. This chapter is an introduction to the "attention" part. Chapter 9 covers the rest.
- **Assignment 1 (Basics)**: implement these parts from scratch: `scaled_dot_product_attention`, causal multi-head self-attention (two versions, with and without RoPE), the Transformer block, and the full language model. The code must pass the unit tests. See the interfaces such as `run_scaled_dot_product_attention` and `run_multihead_self_attention` in `tests/adapters.py` of the assignment repository [stanford-cs336/assignment1-basics](https://github.com/stanford-cs336/assignment1-basics). The assignment text of the current term gives the exact requirements. You can use `02_attention_from_scratch.py` from this chapter directly as a warm-up for this part.

The course page has the lecture notes and the YouTube recordings of each lecture.

---

## References

- Vaswani et al. *Attention Is All You Need*, NeurIPS 2017 (scaled dot-product attention, the variance explanation of √d_k, multiple heads, causal mask): <https://arxiv.org/abs/1706.03762>
- Bahdanau, Cho, Bengio. *Neural Machine Translation by Jointly Learning to Align and Translate* (the origin of attention; a fixed-length vector is a bottleneck), 2014: <https://arxiv.org/abs/1409.0473>
- Ainslie et al. *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints*, 2023: <https://arxiv.org/abs/2305.13245>
- Dao et al. *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*, 2022: <https://arxiv.org/abs/2205.14135>
- Qwen Team. *Qwen3 Technical Report*, 2025: <https://arxiv.org/abs/2505.09388>
- Llama Team. *The Llama 3 Herd of Models*, 2024: <https://arxiv.org/abs/2407.21783>
- OLMo Team. *2 OLMo 2 Furious*, 2024: <https://arxiv.org/abs/2501.00656>
- DeepSeek-AI. *DeepSeek-V3 Technical Report* (MLA), 2024: <https://arxiv.org/abs/2412.19437>
- Gemma Team. *Gemma 3 Technical Report*, 2025: <https://arxiv.org/abs/2503.19786>
- OpenAI. *gpt-oss-120b & gpt-oss-20b Model Card*, 2025: <https://arxiv.org/abs/2508.10925>
- Karpathy. nanoGPT (the form of `CausalSelfAttention`; the trick of the prefix average with a lower-triangular matrix comes from the companion video *Let's build GPT: from scratch, in code, spelled out*): <https://github.com/karpathy/nanoGPT>, <https://github.com/karpathy/ng-video-lecture>
- *Understanding Transformers and Attention Mechanisms: An Introduction for Applied Mathematicians* (already in `references.md`; for readers who want a more mathematical derivation): <https://arxiv.org/pdf/2604.00965>
- CS336 Assignment 1 repository: <https://github.com/stanford-cs336/assignment1-basics>
- CS336 (Spring 2026): <https://cs336.stanford.edu/>

**Next chapter**: Attention moves and mixes information, but it does almost no "computation": each position gets only a weighted average of the value vectors of other positions. Also, our one-layer model finds only shallow patterns, such as "the previous one or two characters". In Chapter 9, we add a feed-forward network (SwiGLU), RMSNorm, residual connections, and RoPE to attention. We stack many layers to build a full modern Transformer, and we train it to generate readable text.
