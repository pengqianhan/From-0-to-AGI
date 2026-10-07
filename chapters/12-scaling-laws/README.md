# Chapter 12: Scaling laws and experiment design — Calculate with small models before you spend money

**English** · [中文](README.zh.md)

> **Goal**: After this chapter, you can use C ≈ 6ND to calculate how much compute and money one training run needs. You can explain the Chinchilla rule of "about 20 tokens per parameter", and why small models are "overtrained". You can run a mini ladder experiment yourself. First, tune the learning rate for each size. Then fit L(N, D), extrapolate to a larger model, and train that model to check the answer. Finally, you can explain why the main-line model is 0.69B × about 400B tokens, and what the gate 1 report must answer.

📺 **Video**: Not published yet. To render it on your computer, run `bash chapters/12-scaling-laws/video/build.sh`.
🧪 **Self-check**: After the chapter, type `/ch12-scaling-laws` in Claude Code.

---

In the previous chapter, we wrote the exam first: which benchmarks, which models to compare with, and what "better" means. This chapter solves a practical problem: **the pretraining of the main-line model has a budget of about $5,000, and we get only one chance**. How large must the model be? How many tokens must it train on? Which learning rate must it use? Before we spend the money, how can we know what the money will buy?

We cannot answer these questions by trial and error, because one full pretraining run takes 8 H100 GPUs for ten days. But language models have a useful property: **the loss changes very smoothly with compute**. The change is so smooth that a series of cheap small models can calculate the result of a large model in advance. This rule is the **scaling law**. This chapter uses the scaling law to design experiments.

The code of this chapter is below. All scripts run on a CPU. Scripts 3, 4, and 6 cache their results in `out/ch12/`, and the next run reads the cache.

```bash
uv run python chapters/12-scaling-laws/code/01_flops.py         # where 6ND comes from: measured vs formula (a few seconds)
uv run python chapters/12-scaling-laws/code/02_chinchilla.py    # Chinchilla arithmetic: optimal allocation, overtraining, inference cost (<1 s)
uv run python chapters/12-scaling-laws/code/03_lr_sweep.py      # mini ladder, step 1: sweep the learning rate for each size (about 7 min, one thread)
uv run python chapters/12-scaling-laws/code/04_mini_ladder.py   # fit L(N,D), extrapolate, held-out test (about 15 min, one thread)
uv run python chapters/12-scaling-laws/code/05_plan_budget.py   # how many tokens $5,000 buys (<1 s)
uv run python chapters/12-scaling-laws/code/06_muon.py          # Muon vs AdamW (about 5 min, one thread)
```

> **Note:** We measured these times on a CPU that several jobs shared. An idle laptop is a few times faster.

## 1. Do the arithmetic: C ≈ 6ND

First, learn to calculate the cost. Chapter 4 showed that the backward pass of a linear layer y = Wx does **two** matrix products. One product gives the gradient for the input (Wᵀg, which goes to the previous layer). The other product gives the gradient for the weights (g xᵀ, which the optimizer uses). Add them to the forward pass:

| Stage | For each parameter and each token | Source |
|---|---|---|
| Forward pass | 1 multiply-add = 2 FLOPs | y = Wx |
| Backward pass: for the input | 2 FLOPs | Wᵀg |
| Backward pass: for the weights | 2 FLOPs | g xᵀ |
| **Total** | **6 FLOPs** | |

Thus, to train a model with N parameters on D tokens, the total compute is:

```
C ≈ 6 · N · D
```

This is the most important formula of the chapter. Only one part is missing. Attention (Chapter 8) has two matrix products **without parameters**: QKᵀ and AV. For each layer and each token, they cost 4·d_attn·T in the forward pass, and 12·d_attn·T with the backward pass (T is the sequence length). The full formula is:

```
FLOPs per token = 6·N_matmul + 12·L·d_attn·T
```

`N_matmul` is the number of parameters in matrix products. It includes the output layer lm_head. It does not include the embedding lookup, because a lookup does no multiplication. This number is not an estimate: we can count it directly. `01_flops.py` uses `FlopCounterMode` from PyTorch to count all matrix products in one forward and backward pass of the small model from Chapter 9:

```python
def formula_flops_per_token(n_matmul: int, n_layers: int, d_attn: int, T: int) -> float:
    return 6 * n_matmul + 12 * n_layers * d_attn * T
```

| dim | Layers | T | N_matmul | 6N | Attention term | Formula total | Measured |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | 2 | 64 | 116,736 | 700,416 | 98,304 | 798,720 | 798,720 |
| 128 | 4 | 128 | 811,008 | 4,866,048 | 786,432 | 5,652,480 | 5,652,480 |
| 128 | 4 | 512 | 811,008 | 4,866,048 | 3,145,728 | 8,011,776 | 8,011,776 |

The last two columns are equal in every digit. (Element-wise operations such as RMSNorm, softmax, and SiLU are less than 1%. The counter and the formula do not count them.) Then use the same formula for the main-line model (`configs/main/pretrain.toml`: dim 1280, 28 layers, 16 query heads × 128, FFN 3584, vocabulary of 65,536):

```
Total parameters N = 689.5M, parameters in matrix products N_matmul = 689.4M
Per token: 6·N_matmul = 4.137e+09, attention term 12·L·d·T = 2.819e+09 (40.5% of the total)
Total 6.955e+09 FLOPs/token; rough estimate 6·N_total = 4.137e+09 (too low by 40.5%)
Train on 400B tokens: C = 2.78e+21 FLOPs (rough estimate 6ND = 1.65e+21)
```

Remember two details:

- **A longer sequence makes attention more expensive.** In the main line, q_dim (2048) is wider than dim (1280). At a sequence length of 4096, attention is 40% of the cost. A shorter pretraining sequence is one way to save money (Section 9).
- **The counting convention.** This formula does not halve the attention cost for the causal mask (PaLM, nanochat, and `zero` all count this way). In practice, FlashAttention skips the masked half. Thus the same training run can report quite different MFU values (model FLOPs utilization: the fraction of the GPU peak that the model uses) under different conventions. Porian et al. 2024 discuss this problem in detail. **Calculate the budget from the measured tokens/s. Do not rely only on the MFU number.**

## 2. Scaling law: the loss is a power law of compute

Train models of different sizes on different amounts of data. Plot the results with compute and loss on log axes. The plot is very regular (video S04). For each fixed model size, the loss first decreases quickly. Then the capacity of the model stops it, and the curve becomes flat. The lower edge of all curves is "the best loss for a given compute". On log axes, this edge is almost a straight line, which is a **power law**.

**History**: Kaplan et al. (2020, OpenAI) were the first to measure this rule systematically. Their conclusion was that the model should grow faster than the data when compute increases: N_opt ∝ C^0.73. GPT-3 followed this idea (175B parameters, trained on only 300B tokens).

**Chinchilla** (Hoffmann et al. 2022, DeepMind) did the experiments again and described the whole plot with one formula:

```
L(N, D) = E + A / N^α + B / D^β
```

- E: the uncertainty of the data itself. No model size and no amount of data can decrease this part.
- A/N^α: the cost of a model that is too small.
- B/D^β: the cost of too little data.

For a given compute C = 6ND, put D = C/(6N) into the formula and find the N with the minimum loss. The result is the "compute-optimal" allocation. The conclusion is very different from Kaplan: **the model and the data should grow in the same proportion, at about 20 tokens per parameter**. Chinchilla 70B trained on 1.4T tokens and beat Gopher, a model 4 times larger.

`02_chinchilla.py` does this arithmetic with two published sets of coefficients. Here, we use the replication by Epoch AI (Besiroglu et al. 2024). They found that the fit in the original Chinchilla paper has a bias, because the optimizer stopped too early. The replicated coefficients agree with the ratio that Chinchilla actually used, 20 tokens/parameter:

```python
def compute_optimal(C, E, A, B, alpha, beta):
    """Closed-form solution: N_opt = G · (C/6)^(β/(α+β)), with G = (αA / βB)^(1/(α+β))."""
    G = (alpha * A / (beta * B)) ** (1 / (alpha + beta))
    N = G * (C / 6) ** (beta / (alpha + beta))
    return N, C / (6 * N)
```

| Compute C | N_opt | D_opt | Tokens/parameter |
|---:|---:|---:|---:|
| 1e19 | 0.26B | 6.4B | 24.2 |
| 1e21 | 2.78B | 60.0B | 21.6 |
| 1.65e21 (the scale of the main line) | 3.59B | 76.6B | 21.3 |
| 1e23 | 29.45B | 566.0B | 19.2 |
| 1e25 | 312.07B | 5340.7B | 17.1 |

(The same code with the coefficients of the original Chinchilla paper gives 78 tokens/parameter at C=1e23. This is the bias that the replication found.)

**Why do Kaplan and Chinchilla disagree?** Porian et al. (2024) used more than 900 training runs to separate the causes. Kaplan did not count the compute of the last layer (lm_head). The warmup was too long for small models. And **the hyperparameters of the small models were not tuned for each size**. After the three corrections, the results agree with Chinchilla. Remember the last cause: we see it again in Section 4.

> **The absolute values do not transfer**: these coefficients belong to the data and the tokenizer of Chinchilla. What transfers to us is the "shape": the order of magnitude of the optimal ratio, and the cost of overtraining. We must fit our own coefficients with our own ladder experiments (Sections 5 and 7).

## 3. Why small models are "overtrained"

But in practice, almost no small model trains at 20 tokens per parameter:

| Model | Parameters | Training tokens | Tokens/parameter |
|---|---:|---:|---:|
| Chinchilla | 70B | 1.4T | 20 |
| Main line of this course (plan) | 0.69B | about 400B | about 580 |
| Puro-2B | about 2B | 1.4T | about 700 |
| Llama 3 8B | 8B | 15T | about 1,875 |
| MobileLLM-R1-950M | 0.95B | 4.2T | about 4,400 |
| Qwen3-0.6B | 0.6B | 36T | about 60,000 |

The reason is that Chinchilla counts only the cost of **training**. A model trains once, but people use it hundreds of millions of times. Each generated token costs about 2N FLOPs. Add inference to the total cost (the method of Sardana & Frankle 2023). Then the goal becomes "reach a given loss with the smallest total compute for training + inference":

```python
            total = 6 * N * D + 2 * N * D_inf
```

Part ③ of `02_chinchilla.py` (the target loss is the main-line loss; Epoch coefficients):

| Tokens generated in the model's lifetime | N with the smallest total compute | D | Tokens/parameter |
|---:|---:|---:|---:|
| 0 (training only) | 2.25B | 49B | 22 |
| 1e12 | 0.84B | 228B | 271 |
| 1e13 | 0.56B | 901B | 1,610 |
| 1e14 | 0.44B | 4,184B | 9,498 |

The more tokens the model must serve, the smaller the model should be and the longer it should train. This is why small models such as Qwen3, Llama 3, and MobileLLM are "overtrained" by hundreds to tens of thousands of times.

Overtraining also has a price. With our budget (C = 6 × 0.69B × 400B ≈ 1.65e21), the script prints:

```
Compute-optimal:  N = 3.60B, D = 77B → L = 2.2636
Main-line choice: N = 0.69B, D = 400B (580 tokens/parameter) → L = 2.3426
Cost of overtraining: the loss is higher by 0.0790 (3.5%); for the same loss, the optimal allocation needs only 40% of the compute
```

The Delphi report gives a similar number at a large scale. With 10× overtraining, the compute-optimal configuration needs only about 1/6 of the compute to reach the same loss. (They warn that this number comes from a long extrapolation, so it is less reliable than their main result.) This is a trade with a known price. We choose 0.69B because the goal is a model that is usable on devices and compares with Qwen3.5-0.8B in the same class (GOAL.md 3.3). We do not choose it because it is "optimal".

## 4. Fit with small experiments: tune first, then fit

Now do a ladder experiment (a scaling ladder: a series of models of increasing size) yourself. All results are a **tiny-configuration demo**. The model is the TinyTransformer from Chapter 9 (byte-level, vocabulary of 256), with 10K to 500K parameters. The data is `assets/tiny_corpus/shakespeare.txt`. The demo tests the method. It does not represent any number of the main-line model.

> **Note:** About the numbers: the numbers of the training experiments in this chapter come from one CPU run on the course build machine. Different machines and different versions of the low-level math libraries do floating-point operations in a slightly different order. After a few hundred training steps, these small differences become larger. Your numbers can differ from the second or third decimal place. Rely on the conclusions below that do not depend on exact values. For a rerun on another server in 2026-10, see [runs/2026-10-01-gpu0-check/chapters-11-15.md](../../runs/2026-10-01-gpu0-check/chapters-11-15.md).

| Size | dim | Layers | Non-embedding parameters N |
|---|---:|---:|---:|
| s1 | 16 | 2 | 10,896 |
| s2 | 32 | 2 | 31,968 |
| s3 | 48 | 3 | 95,664 |
| s4 | 64 | 4 | 217,792 |
| s5 (held out) | 96 | 4 | 467,936 |

Here, N is the number of parameters in matrix products (with lm_head, without the embedding lookup). This is the same count as 6N in Section 1. Porian et al. found that "not counting lm_head" is one source of the Kaplan bias. Two more small changes come from measurements:

- The input embedding and lm_head do not share weights. At this scale, shared weights keep the model for a long time on a plateau where it "only predicts frequent bytes". The curves then have too much noise to fit a law.
- The batch is 8×64. The same number of tokens then gives more steps, and the model learns faster at this scale.

**The first step is to tune the learning rate.** The conclusion of Lourie et al. (2026, arXiv:2608.11859) is the most important warning of this chapter. Small models are very sensitive to hyperparameters, and badly tuned small experiments distort the scaling law. When the tuning is good enough, a clear law appears already at 4M parameters. `03_lr_sweep.py` sweeps 5 learning rates for each size (each value is 2× the previous value). **If the best value is at an edge of the grid, the script adds one more value on that side**, until the best value is inside the grid. Then it fits a parabola through the best point and its two neighbors, and takes the vertex:

```
Validation loss (bit/byte). One row per size. * = best for that size, - = not run:
size        N |  0.00125   0.0025    0.005     0.01     0.02     0.04
  s1   10,896 |        -  3.5129   3.3643   3.3191   3.2778*  3.3372
  s2   31,968 |        -  3.2625   3.1996   3.1669*  3.2353   3.3323
  s3   95,664 |  3.3094   3.1940*  3.2385   3.3055   3.3848   3.4342
  s4  217,792 |  3.2189   3.1401*  3.2746   3.3019   3.3805   3.3845
η* after parabola interpolation: s1 0.01879, s2 0.00885, s3 0.002915, s4 0.002284

Fit η*(N) = 18.6 · N^(-0.744)
Extrapolate to the held-out size s5 (N=467,936): η* ≈ 0.001123
```

The best learning rate decreases as the model becomes larger. Suppose that you take a shortcut and use 0.02, the value from the smallest model, for all sizes:

```
  s1 N= 10,896  3.2778  worse than tuned by +0.0000
  s2 N= 31,968  3.2353  worse than tuned by +0.0684
  s3 N= 95,664  3.3848  worse than tuned by +0.1908
  s4 N=217,792  3.3805  worse than tuned by +0.2405
```

Then s3 and s4 are **worse** than s1 and s2. A fit to these data concludes that "a larger model does not help". That conclusion is completely wrong. This effect is one source of the disagreement between Kaplan and Chinchilla. It is also the reason why the zero pipeline demo in "Main-line progress" below fails.

How do the other hyperparameters change with scale? The current consensus method is "fit them with small experiments":

- DeepSeek LLM fitted power laws of the best learning rate and the best batch size against compute (η_opt = 0.3118·C^−0.125, B_opt = 0.292·C^0.327).
- The Qwen3 report says that it used scaling laws to predict the learning-rate schedule and the batch size of each model.
- Llama 4 uses MetaP, its own method, to transfer hyperparameters across width, depth, batch size, and training length.

The theories of transfer (μP and its variants) are not a consensus yet. They are in "Frontier notes" at the end of the chapter.

## 5. Mini ladder: fit L(N, D), then extrapolate

When each size has its η\*, `04_mini_ladder.py` trains the ladder. It uses a trick to save compute: **WSD branches**. The WSD schedule of Chapter 6 keeps the learning rate constant in the stable phase. Thus each size trains only one trunk. At 0.8·D_k, copy the trunk and add a decay of 0.2·D_k. The result is the model for budget D_k:

```python
    for step in range(trunk_end + 1):
        if step in branch_at:  # branch: copy the current state and add a decay
            D = branch_at[step]
            m2 = copy.deepcopy(model)
            ...
            for i in range(n_decay):
                train_step(m2, o2, *batch(train, g2), lr * (1 - (i + 1) / n_decay))
            results[D] = val_bpb(m2, val)
```

3 budgets then cost only about 1.15× the compute of the largest budget (not 1.75×). The pretraining configuration of the main line uses the same design (`decay_frac = 0`; the decay occurs in the mid-training of Chapter 15).

Each size has 3 budgets (64K, 128K, and 256K bytes). The two smallest sizes get one more budget of 1M bytes. **The ladder must cover the "overtrained" region where the main line is.** Without that region, the fit cannot find the exponent for N. The first run had only 12 points, and the fit gave α ≈ 0.05. That result almost says "N does not matter", because all points were in the region with too little data. With the two extra points, α is no longer near 0. But, as we see below, the fit still does not find α precisely.

The fit uses "variable projection". For a fixed (α, β), L is linear in (E, A, B). A grid search runs over α and β, and at each grid point a least-squares problem with 3 unknowns gives E, A, and B:

```python
    X = np.stack([np.ones((len(a), len(L))), n[None] ** -a[:, None], d[None] ** -b[:, None]], axis=2)
    ...
    sse[(coef < 0).any(1)] = np.inf  # E, A, and B must all be non-negative
```

Result (14 points, course build machine):

```
Fit (14 ladder points): L(N, D) = 2.557 + 19.23/N^0.52 + 520.6/D^0.58
size        N         D   D/N |  actual     fit   error
  s1   10,896    65,536     6 |  3.4853  3.5478  +1.79%
  s1   10,896 1,048,576    96 |  2.9307  2.8781  -1.80%
  s2   31,968 1,048,576    33 |  2.7765  2.8125  +1.30%
  s4  217,792   262,144     1 |  2.9258  2.9644  +1.32%
  ... (the full 14 rows are in the script output; maximum error 1.80%)
```

The maximum error of the fitted points is about 2% (1.80% in this run, 2.40% in the rerun below). This is looser than Delphi (< 0.5%) and the gate 1 standard (< 1%). Training runs with tens of thousands of parameters and hundreds of thousands of bytes have a lot of noise. This is the honest result.

**The same experiment on a different machine gave an α that is almost 2× different.** In 2026-10, we ran the same code with the same seeds on another server. The fits were:

```
Course build machine: L(N, D) = 2.557 + 19.23/N^0.52 + 520.6/D^0.58   (maximum error 1.80%)
Rerun server:         L(N, D) = 2.565 + 1721/N^0.98  + 427.2/D^0.56   (maximum error 2.40%)
```

E and β almost did not change, but α, the exponent for N, jumped from 0.52 to 0.98. The code has no bug: a rerun on the same machine gives the same output, byte for byte. The only source of the difference is the order of floating-point operations. In `03`, the losses in the cells with high learning rates differ in the third or fourth digit. The η\* from the parabola changes with them (for s1, from 0.0188 to 0.0214). The ladder trains with the new learning rates, and each of the 4 points in the table above moved by 0.1%–2%.

Such a small change can move α by 2×. This shows that the data **did not fix this exponent in the first place**. There are three reasons:

- **Too few points**: only 4 sizes (a range of 20×) tell us how fast the loss decreases as the model becomes larger. Only 2 of them have points in the overtrained region. The fit must find A and α from these few points, and it must also separate them from E.
- **The models are too small and the noise is too large**: one seed, tens of thousands of parameters, and hundreds of thousands of bytes. Each point has 1%–2% noise (the same point moved this much between the two machines). But when N changes from 10K to 220K, the loss decreases by only 2%–8% (it depends on the budget). We can see the difference in level, but the curvature ("how fast the loss decreases") is lost in the noise. Lourie et al. (2026) found that small models are very sensitive to hyperparameters, and that the law appears only at the frontier of full tuning. Their ladder starts at 4M parameters. With only 4 or 16 hyperparameter settings for each size, they could not see the law at all. They needed 256 settings for an accurate fit. Our models are about 20–400 times smaller than 4M, and we swept only 5–6 learning rates for each size.
- **The parameters can cancel each other (poor identifiability)**: when α increases, A and E increase too. The curve in the range of the fit then stays almost the same. Test this with the 14 points of the rerun. Fix α at any value from 0.1 to 1.5, and fit the other parameters again. The root mean square of the relative error changes only between 1.23% and 1.40%. It is 1.27% at α = 0.52, and 1.23% at the best α = 0.98. The difference is much smaller than the noise. The bootstrap below shows the same thing: with 200 resamples by size, the 95% interval of α is 0.02–1.50, which covers the full search grid.

When the exponent is not fixed, predictions near the data change little, but conclusions far from the data change completely. Again with the points of the rerun, fix α between 0.2 and 1.5:

- The held-out test below extrapolates only 2.1×. The prediction errors for all three budgets stay within ±2.5%.
- Extrapolate to N ≈ 4.7M (21× the largest ladder size, D = 262,144). The predicted loss then changes from 2.84 to 2.98, a difference of 5%.
- "How many tokens each parameter should get" depends on β/(α+β), so it also changes with α (the last line of output in this section: 127 or 68).

This is the lesson of "extrapolation and held-out tests" in this chapter. It is also the reason for the design of the main-line ladder (Section 7). That ladder uses much larger models: 23M–319M in `configs/ladder/`, all above the 4M where Lourie et al. saw the law, with the same vocabulary as the main line. For each size, it **sweeps the learning rate until the best value is inside the grid, and only then fits**. It gives bootstrap intervals for the extrapolated values. And it **always keeps one size out of the fit for a test**.

To decide if a fit is reliable, look at the held-out error and the interval, not at whether the exponent "looks like 0.5". Lourie et al. also warn that a longer extrapolation makes the sampling error larger, so the law is most reliable near the data.

**The key step is the held-out test.** Use the power law η\*(N) to extrapolate the learning rate of s5 (N is 2.1× the largest ladder size): 0.00112. **Do not tune this learning rate.** Train the model, and compare the result with the extrapolation of the fit. For the uncertainty, use a bootstrap. Resample by size with replacement, and fit again 200 times. (The points from the branches of one training run are correlated, so resample them as a whole group.)

```
Extrapolate to the held-out size s5 (N = 467,936, 2.1× the largest ladder size):
       D |   pred.    95% interval |  actual   error
  65,536 |  3.4165  3.3912–3.4535  |  3.4465  -0.87%   (with the s4 learning rate 0.002284: 3.3770, error +1.17%)
 131,072 |  3.1393  3.1062–3.1829  |  3.1495  -0.32%   (with the s4 learning rate 0.002284: 3.1264, error +0.41%)
 262,144 |  2.9538  2.9123–3.0049  |  2.9243  +1.01%   (with the s4 learning rate 0.002284: 2.9209, error +1.13%)
```

The errors for the three budgets are −0.87%, −0.32%, and +1.01%, and all of them are inside the 95% interval. On the rerun server, the errors were −1.02%, −0.15%, and +1.51%, which are of the same size. But in the 65,536 cell, the actual value 3.4623 was a little above the upper limit of the interval, 3.4585: it **fell outside the interval**.

The conclusion that holds on both machines: an extrapolation of 2.1× has an error of about ±1.5%, the same size as the error of the fitted points. The bootstrap interval contains only the uncertainty of "fit again with a different set of sizes". It does not contain the noise of the held-out training run itself (seed, floating point). Thus the interval is a little too narrow, and it is not surprising when the true value falls outside it. In a report, write down each point outside the interval honestly.

The values in parentheses are a control. When s5 uses the learning rate of s4, the actual loss is a little lower on the short budgets (on both machines). Thus the power-law extrapolation of the learning rate was a little too conservative on the shortest budget. The learning-rate rule itself can also be wrong. This is the reason why Delphi failed the first time (next section).

The last line of the output is worth a look:

```
This fit says: at compute C = 3.43e+11, the optimum is N ≈ 21,194, D ≈ 2,693,780 (127.1 tokens/parameter).
```

On the rerun server, this line is N ≈ 28,997, D ≈ 1,968,928 (67.9 tokens/parameter). Both runs are far from 20, but they also differ from each other by almost 2×. The optimal allocation N_opt ∝ C^(β/(α+β)) depends on the ratio of the exponents. Because α is not fixed, the allocation is not fixed either.

The only safe statement is this: on small byte-level data, the compute-optimal ratio is not 20. **The optimal ratio depends on the data and the tokenizer.** (DeepSeek LLM also found that with higher-quality data, more of the compute should go to model size.) This is why the main line must run its ladder with its own data and its own tokenizer. It cannot use the Chinchilla numbers directly.

## 6. Extrapolation in practice: Delphi

Is the same true at a large scale? **Delphi** from the Marin team (in `references.md`) is a good real case:

- **Method**: First, use a small reference model that can be tuned exhaustively to define a "recipe". The recipe is a function that maps compute to a full training configuration (width, depth, batch size, learning rate, β, weight decay, initialization, ...). Then run IsoFLOP sweeps (different N and D at the same compute) at 7 budgets from 3e18 to 3e20 FLOPs. Take the best point of each budget and fit a power law L(C).
- **First failure**: the fit at small budgets was clean. But the held-out run at 1e22 was off by 2.5%, and the run at 1e23 diverged. The problem was in long training. The learning rate increased with the batch size as √B, but it did not decrease with the training length. Thus large runs with much data had a learning rate that was too large. The weight decay was a fixed 0.1, and it also did not change with scale.
- **Fix the recipe, not the curve**: Multiply the learning rate by a training-length correction (T₀/T)^0.3 (from Complete(d)P by Apple). Change the optimizer to AdamH, which keeps the weights on a sphere of fixed norm (weight decay is then no longer in the recipe). After a new fit on the same 7 budgets, the held-out errors at 1e21, 1e22, and 1e23 were +0.5%, +0.2%, and +0.2%. This is an extrapolation to more than 300× the range of the fit.
- **Uncertainty**: a bootstrap over the best points of the 7 buckets gives a 95% interval of ±0.5% at 1e21, which widens to ±4% at 1e23. The difference between three seeds is only about 0.1%.
- **Downstream scores**: "hard" metrics such as accuracy stay near the random level for small models (25% for MMLU with four choices), so they show no progress. Delphi first fits a scaling law to **soft metrics** (the log probability of the correct choice, or the bits per byte of the reference answer). Then it uses a set of public models to fit an S-shaped mapping "soft metric → hard score". The combination of the two predicts MMLU, HumanEval, and GSM8K for large models. (The Llama 3 technical report uses the same two-step method.)

From Delphi and our mini ladder, we get four rules for extrapolation: **fix the recipe, tune first, keep a held-out test, and report the interval honestly**. When the test fails, fix the recipe and run again. Do not patch the old fit.

## 7. Gate 1: a person decides before we spend the large budget

Gate 1 in GOAL.md 3.4 applies the methods above to the main line. The ladder protocol for step 2, when GPUs are available, is in [`runs/ladder/README.md`](../../runs/ladder/README.md) (**step 1 spends no money on GPUs**):

- **Sizes and budgets**: the 4 sizes in `configs/ladder/` (23M, 71M, 160M, and 319M total parameters, with the same vocabulary as the main line). Each size has three budgets: 20×, 80×, and 320× tokens/parameter (l300m runs only 20× and 80×). The runs use WSD branches. l20m–l150m are for the fit, and **l300m is held out**.
- **Sweep the learning rate first**: 5 learning rates for each size. If the best value is at an edge, extend the grid. Extrapolate η\*(N) to l300m and to the main line. If necessary, add the training-length correction of Delphi.
- **Estimate**: one round of the ladder (with branches) costs about $340. With the learning-rate sweep and repeated seeds, the cost is about $450–550 (with the assumption MFU 0.3). This is inside the $1,200 budget for Chapters 12–13. A single run that costs more than $100 needs approval first.

The gate 1 report (template: [`runs/gate1_report_template.md`](../../runs/gate1_report_template.md)) must answer three questions:

1. **Loss extrapolation**: the validation loss of the main line at the planned number of tokens, with its 95% interval. List all fit residuals, held-out errors, and extrapolation factors (`zero.tools.fit_scaling`).
2. **Benchmark score extrapolation**: use the two-step method, a scaling law of soft metrics + an S-shaped mapping fitted on public models (`fit_loss_to_score`). Do not extrapolate a benchmark that is near the random level at ladder scale; write "no signal" honestly. Use only our own development set. Do not touch the preregistered test benchmarks.
3. **Recipe validation**: (a) Apply the post-training recipe to 2–3 Base models of the ladder. Fit "Base loss → tool-calling score", and extrapolate it to the main line. (b) Apply the recipe to an existing open Base model of the same size, to see the upper limit of the recipe itself (only for validation, not for release).

**If the prediction does not reach the hard target, do not start pretraining.** First change the recipe or the target. Do not spend the money anyway.

## 8. Two training techniques "to be verified": Muon and FP8

GOAL.md 2.1 lists the Muon optimizer and FP8 training as "to be verified". We apply the consensus rule. The rule requires at least 3 independent leading open model families. Each family must state clearly in the technical report of its main version that it uses the technique. The results are below. The sources are in "Adopters and sources".

- **Muon: consensus reached, so it goes into the main text.** The technical reports of Kimi K2 (MuonClip), GLM-4.5, and DeepSeek-V4 all state that they trained their main models with Muon. Other users are Moonlight (from Moonshot, the same company as Kimi), Puro-2B (MuonH), nanochat, and more.
- **FP8 training: consensus reached.** The adopters are DeepSeek-V3 (fine-grained FP8 mixed precision), Llama 4 (official blog: Behemoth was pretrained in FP8), and NVIDIA Nemotron-H (56B pretrained fully in FP8). Another user is Puro-2B (block-wise FP8). FP8 belongs to Chapter 14, "Pretraining engineering". Here we discuss only its effect on the budget.

**What Muon is**: For each weight matrix of a hidden layer, Muon first accumulates momentum as SGD does. Then 5 Newton–Schulz iterations "orthogonalize" the update matrix: G = UΣVᵀ becomes about UVᵀ, and all singular values move to about 1. The intuition: AdamW scales each element separately, so a few directions often dominate the update matrix. Muon gives all directions the same step size, so directions that are small in the gradient can also learn.

The embedding, lm_head, and RMSNorm still use AdamW (the three families group the parameters in the same way). A multiplication by 0.2·√max(m, n) makes the RMS of the update similar to that of AdamW. Thus the AdamW learning rate still works (the method of Moonlight / Kimi K2; GLM-4.5 uses 0.2, DeepSeek-V4 uses 0.18).

```python
def newton_schulz5(G, steps=5):
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G / (G.norm() + 1e-7)  # the iteration converges only when the largest singular value is ≤ 1
    ...
    for _ in range(steps):
        A = X @ X.T
        X = a * X + (b * A + c * A @ A) @ X
```

`06_muon.py` compares Muon with the **tuned** AdamW of Section 4. It uses s2 and s3 of the mini ladder with the same 131K bytes. (Muon also gets a learning-rate sweep, and its best value is inside the grid.)

```
NS5 check: singular values of a random 64×32 gradient 0.059–0.290 → after orthogonalization 0.682–1.042

Same 131,072 bytes, validation loss (bit/byte):
size        N |     AdamW best |      Muon best | diff
  s2   31,968 | 3.1669 (η=0.01  ) | 2.9945 (η=0.02  ) | -0.1724
  s3   95,664 | 3.1940 (η=0.0025) | 2.9350 (η=0.01  ) | -0.2590
```

The difference is large. But **this is a tiny-configuration demo with tens of thousands of parameters and short training**, and it does not transfer directly to 0.7B. Moonlight reports about 2× compute efficiency in compute-optimal training. Puro-2B reports a further 1.19× for MuonH over a tuned Muon. Both are improvements, but they are much smaller than the result here. The production implementation is in `zero/train/muon.py` (the CPU tests pass; the GPU path is not verified yet).

**Recommendation for the main line**: in the ladder of step 2, compare Muon and AdamW directly on l60m and l150m. Tune the learning rate for both, and fit an "equivalent compute factor" (the method of Puro-2B). Use Muon only if the interval of the factor is clearly above 1.1. Otherwise, keep AdamW.

**Effect of FP8 on the budget**: on a 1.7B model, Puro-2B measured 1.36× throughput, or a net 1.34× after the loss penalty. NVIDIA NeMo measured about 1.3× for Llama 3 8B on H100, and the gain is larger for larger models. A 0.7B model has smaller matrices, so its gain can be smaller. `zero` does not implement FP8 yet, and FP8 is not verified on GPUs. **The budget in Section 9 does not count on FP8.**

## 9. Main-line decisions: size, number of tokens, hyperparameters

`05_plan_budget.py` inverts the formula of Section 1: how many tokens can $5,000 buy? The output is below. It is an excerpt: the script prints more rows for the candidates, and one more line for MFU 0.55. `24L` has 4 fewer layers (24, not 28). `narrow` has dim 1024 and FFN 3072. `days` is the number of days on the 8 GPUs.

```
Budget $5,000, 8×H100 (peak 989.5 TFLOPS, to be verified), $2.5/GPU-hour → 2,000 GPU-hours in total
candidate     params   seq  MFU |  token tok/param    days
main 0.69B     0.69B  4096  0.4 |   410B       594    10.4
main 0.69B     0.69B  4096  0.5 |   512B       743    10.4
main 0.69B     0.69B  2048  0.4 |   514B       745    10.4
24L 0.60B      0.60B  4096  0.4 |   472B       783    10.4
narrow 0.51B   0.51B  4096  0.4 |   486B       958    10.4

Compare: the original plan of 500B tokens (MFU 0.4, sequence 4096) costs $6,102, 22% over the $5K limit.
  MFU 0.40: $5K buys  410B tokens; 500B costs $6,102
  MFU 0.45: $5K buys  461B tokens; 500B costs $5,424
  MFU 0.50: $5K buys  512B tokens; 500B costs $4,881
```

**Recommendations (gate 1 in step 2 makes the final decision; these are not conclusions)**:

1. **Size: keep 0.69B** (`configs/main/pretrain.toml`, 689.5M total parameters ≤ 0.8B). A model of 0.5–0.6B can buy 15–20% more tokens. But on the curve of Section 2, the loss from a 25% smaller N is larger than the gain from 20% more D. More important, the hard goal compares with Qwen3.5-0.8B. The smaller our model is than the competitor, the harder it is to win.
2. **Number of tokens: a limit of $5K, with a plan of about 400B** (about $4.9K at MFU 0.4, 580 tokens/parameter). Pretraining uses the stable phase of WSD (no decay), so we can stop at any time and continue with mid-training. Thus we do not have to bet on the MFU in advance. After we measure the throughput in stage 6, $5K can buy more (512B at MFU 0.5). In that case, train until the budget is used.
3. **An optional way to save money**: reduce the pretraining sequence from 4096 to 2048. The FLOPs per token decrease by 20%, and the same money buys 25% more tokens. (Qwen3, GLM-4.5, and DeepSeek-V4 use 4K, and SmolLM2 uses 2K. The long context must extend to 32K in Chapter 15 anyway.) The cost is that the main-line hyperparameters must be swept again at 2048, and the ladder must use the same length. This option is worth a test in the ladder. It is not the default.
4. **Hyperparameters**: extrapolate the peak learning rate from η\*(N) of the ladder (3e-4 in the configuration is only a starting point). For now, keep β₁ 0.9, β₂ 0.95, weight decay 0.1, gradient clipping 1.0, and batch size 524K tokens. The default optimizer is AdamW; the comparison in Section 8 decides about Muon. The precision is BF16, and the budget does not count on FP8.
5. **Production tool**: `uv run python -m zero.tools.plan_budget --config configs/main/pretrain.toml --mfu <measured> --fit runs/ladder/fit.json` puts the budget, the number of tokens, and the loss extrapolated from the ladder into one table.

## 10. Summary

- **Arithmetic**: the training FLOPs per token ≈ 6·N_matmul + 12·L·d_attn·T, and the total compute C ≈ 6ND. For the main line at length 4096, attention is 40% of the cost.
- **Chinchilla**: L(N, D) = E + A/N^α + B/D^β. When only training counts, the optimum is about 20 tokens/parameter. The different conclusion of Kaplan came mainly from badly tuned small models and a different compute count.
- **Overtraining**: when inference counts too, a longer training of a small model pays off. The cost is a slightly higher loss for the same compute (about 3.5% for the main line).
- **Tune first, then fit**: small models are very sensitive to hyperparameters. The best learning rate decreases with size. A shortcut gives the wrong conclusion that "a larger model does not help".
- **Test each extrapolation**: WSD branches save compute. Cover the overtrained region. Use a bootstrap for the interval. Hold out a larger model to check the answer. If the test fails, fix the recipe. The exponents from a tiny ladder are fragile: the same experiment on another machine moved α from 0.52 to 0.98. Look at the held-out error and the interval, not at the exponent itself.
- **Main line**: 0.69B × about 400B tokens, with a $5K limit and a WSD stable phase that can stop at any time. Muon and FP8 are both consensus, but they must pass a test in the ladder before they go into the main line.

---

## From minimal code to production code

| Minimal code (`code/`) | Production code (`zero/`, `runs/`) | What it adds and why |
|---|---|---|
| `01_flops.py`: formula by hand + measurement with `FlopCounterMode` | `estimate_flops_per_token` and `count_params` in `zero/model.py` | Handles GQA (fewer K and V heads), the lm_head product with tied embeddings, and configurations after YaRN. The trainer uses it to report the MFU in real time. |
| `05_plan_budget.py`: arithmetic with fixed constants | `zero/tools/plan_budget.py`: `tokens_for_budget`, `plan`, `apply_candidate` | Reuses `estimate_cost` directly (the same GPU peak table, the same counting convention). Reads any configuration, can change the shape to make candidates, and can read a ladder fit to predict the loss. `--speedup` supports questions such as "what if FP8 makes training faster" (marked as not verified). |
| `fit_lnd` in `04_mini_ladder.py`: grid + unconstrained least squares; drops negative coefficients | `zero/tools/fit_scaling.py`: `fit_chinchilla`, `fit_power_law`, `bootstrap_chinchilla`, `fit_loss_to_score` | **Non-negative** least squares with 3 variables (enumerates 7 active sets; exact and vectorized). Normalizes N and D first, to prevent an ill-conditioned problem. Two rounds of grid refinement. A warning when an exponent is at the boundary. An option for α=β. Bootstrap by group. The power law L(C). The S-shaped mapping loss → score (step 2 of gate 1). |
| Points written into JSON by hand | `fit_scaling --run DIR[:CONFIG] --holdout ... --target-config ... --out fit.json` | Reads the `log.jsonl` of the trainer directly (the last validation loss and the number of tokens). Calculates N from the configuration in the checkpoint. N can be non-embedding parameters, total parameters, or FLOPs. |
| WSD branches written by hand in `03` and `04` | WSD in `zero/train/schedule.py` + resume from a checkpoint (the commands in Section 4 of `runs/ladder/README.md`) | A branch is "copy the checkpoint at the branch point, and continue training with a smaller `max_steps`". The state of the data loader is restored too, so the branch sees the data that the trunk sees next. |
| Minimal `Muon` in `06_muon.py` | `zero/train/muon.py`: `zeropower_via_newtonschulz5`, `MuonAdamW`, `split_params_for_muon`, `build_muon_optimizer` | One optimizer object manages both the Muon group and the AdamW group (the schedule writes the learning rate for each group; the checkpoint has one `state_dict`). Two scale rules (rms / spectral). NS5 in BF16 on CUDA. An error for parameters that FSDP shards. (BF16 NS5 on CUDA and the same results on all GPUs under DDP are verified on an RTX 3090; see Sections 11 and 14.1 of [runs/2026-10-01-gpu0-check](../../runs/2026-10-01-gpu0-check/README.md).) |

**Parity checks and tests**:

- `tests/test_fit_scaling.py`: generate ladder data from known coefficients (on the scale of the Epoch replication). The fit must recover E, α, and β (error < 0.02), and the extrapolation error must be < 0.3%. With added noise, the bootstrap interval must cover the true values. The α=β option, the power law L(C), and the S-shaped mapping each recover known parameters. The CLI can read logs in the trainer format and do a held-out test. The number of tokens from `plan_budget`, put back into `estimate_cost`, gives exactly the budget ($5K ↔ 410B, as in the RUNBOOK).
- `tests/test_muon.py`: the output of NS5 points in the same direction as the exact orthogonal factor from an SVD (cosine > 0.95), and its singular values are in [0.5, 1.3]. With the rms scale, the RMS of one update ≈ lr × 0.2. The parameter groups agree with the three technical reports (matrices in blocks use Muon; embedding / norm use AdamW). On a small Transformer, the loss decreases. A resume from `state_dict` gives the same digits as a run without interruption.

**Connection to the trainer (waits for a merge into the main code)**: this chapter does not change `zero/train/trainer.py` or `zero/config.py`. To enable Muon, make two small changes. First, add the field `name: str = "adamw"` to `OptimConfig`. Second, add one line at the start of `build_optimizer`: `if cfg.name == "muon": return build_muon_optimizer(model, cfg, device)`. Then `--set optim.name=muon` enables the comparison in the ladder.

## Main-line progress

### Tiny-configuration demo: the zero pipeline from training logs to a fit

> The results below are a **tiny-configuration demo**. They show only that the code runs. They are not results of the main-line model.

Use `configs/tiny/pretrain.toml` (tiny corpus, vocabulary of 2048) to run a small ladder of 4 sizes × 2 budgets. **On purpose, keep the learning rate 3e-3 from the configuration and do no sweep**:

```bash
for spec in "32 1 150" "32 1 300" "64 2 150" "64 2 300" "96 3 150" "96 3 300" "128 4 150" "128 4 300"; do
  set -- $spec
  uv run python -m zero.train.pretrain --config configs/tiny/pretrain.toml \
    --set model.dim=$1 --set model.n_heads=$2 --set model.n_kv_heads=$2 --set model.ffn_dim=$(( $1 * 3 )) \
    --set model.n_layers=2 --set train.max_steps=$3 --set train.eval_every=$3 --set train.eval_batches=16 \
    --set checkpoint.every=0 --set schedule.warmup_steps=20 --set train.out_dir=out/ch12/zero_ladder/d$1_s$3
done
Z=out/ch12/zero_ladder
uv run python -m zero.tools.fit_scaling --run $Z/d32_s150 --run $Z/d32_s300 --run $Z/d64_s150 --run $Z/d64_s300 \
  --run $Z/d96_s150 --run $Z/d96_s300 --holdout $Z/d128_s150 --holdout $Z/d128_s300 --tie-exponents --bootstrap 100
```

Output (the validation loss is in nat/token):

```
Fitted 6 points (N definition: non_embedding)
  L(N, D) = 4.7664 + 2.725e+07/N^2.0000 + 6.361e+10/D^2.0000
  Fit residual RMSE 0.1088, max relative error 3.76%
  ⚠ An exponent is at the edge of the search range: the data does not fix the law (common causes: learning rate not tuned, too narrow a range of sizes or budgets). Do not trust the extrapolation.
run                                 N            D   actual      fit    error
d32_s300                    2.691e+04    6.144e+05   4.9626   4.9725   +0.20%
d64_s300                    1.069e+05    6.144e+05   4.7790   4.9373   +3.31%
d96_s300                    2.402e+05    6.144e+05   5.1282   4.9354   -3.76%
...
Held-out test (not used in the fit):
  d128_s150: actual 5.5341, extrapolated 5.4406 (95% interval 5.2520–5.5005), error -1.69%
  d128_s300: actual 5.1999, extrapolated 4.9351 (95% interval 4.6818–5.0706), error -5.09%
```

The pipeline works: trainer log → read the configuration to calculate N → fit → bootstrap → held-out test → JSON. But **the fit itself failed**, and the failure teaches a useful lesson. The learning rate was not tuned, so d96 is worse than d64 (5.13 vs 4.78). The exponents hit the edge of the search range, and the tool gives a warning. The held-out error is −5.09%, outside the interval. This is the conclusion of Section 4 again, now in the production pipeline: the ladder of gate 1 must sweep the learning rate first (Section 3 of `runs/ladder/README.md`).

### To be added after GPU training

- The ladder experiment (`runs/ladder/README.md`): the learning-rate sweep table, the L(N, D) fit, the l300m held-out test, and the extrapolated loss of the main line with its interval.
- The benchmark score extrapolation (soft metrics + S-shaped mapping), and the recipe validation (a) and (b).
- The ladder comparison of Muon vs AdamW, and the "equivalent compute factor".
- The measured MFU / tokens/s, and the final number of tokens and cost.
- **The gate 1 report** (`runs/gate1_report_template.md`) and the approval record. Record the spending in `runs/ledger.md`.

## Frontier notes

- **μP and other theories of hyperparameter transfer** (μP, Complete(d)P, MetaP of Llama 4): the goal is to transfer the learning rate and the initialization, tuned on a small model, "as is" to a large model. Then the large model needs no tuning. The Llama 4 blog says that MetaP transfers hyperparameters across batch size, width, depth, and training length. Delphi uses the training-length correction of Complete(d)P, and MiniCPM uses μP. But each family uses different specific rules, and most details are not published, so these theories are not a consensus yet. This course uses the consensus method "sweep on the ladder + power-law extrapolation" (GOAL.md 2.1 lists the details of μP as out of scope).
- **Hyperball optimizers** (AdamH, MuonH): they keep each weight matrix on a sphere with its norm at initialization. Weight decay is then no longer a hyperparameter, and the learning rate is the "effective learning rate". Delphi (AdamH) and Puro-2B (MuonH; it reports 16% less compute than a tuned Muon) use them. There are still too few adopters.
- **Curriculum data order + checkpoint averaging** (CMA of Puro-2B): order the data from low to high quality, keep the learning rate constant late in training, and average the last few checkpoints. The report says that this costs about 2.4× less than uniform data + decay. This result has only one source. Chapter 15 also discusses annealing and averaging.
- **A "cost scaling law" for a recipe** (Puro Cost Scaling Law): fix a 2B model, and fit the average score of 15 benchmarks as a log function of the cost: P = a + b·log₂(C − C_P1). It answers the question "how much money reaches the level of Qwen2-1.5B" (about $4.4K). It is a curve toward lower cost for one fixed size and one recipe. The authors state clearly that it does not generalize to other sizes. The idea can help with the "money → score" report of gate 1.

## Adopters and sources

| Technique | Adopters (leading open model families) | Source |
|---|---|---|
| Fit scaling laws with small experiments to set the size, the number of tokens, and the hyperparameters | DeepSeek | DeepSeek LLM §3 (power laws of the learning rate and the batch size, IsoFLOP, non-embedding FLOPs/token count): <https://arxiv.org/abs/2401.02954> |
| | Qwen3 | Technical report §3.2 "we develop scaling laws for optimal hyper-parameters (e.g., learning rate scheduler, and batch size) predictions": <https://arxiv.org/abs/2505.09388> |
| | Llama | Llama 3 technical report §3.2.1 (scaling laws set the flagship size; a two-step method predicts downstream scores): <https://arxiv.org/abs/2407.21783>; Llama 4 blog (MetaP): <https://ai.meta.com/blog/llama-4-multimodal-intelligence/> |
| | Kimi | Kimi K2 technical report §2 (the architecture is "derived from empirical scaling law analysis"): <https://arxiv.org/abs/2507.20534> |
| | Marin (open recipe) | Delphi: <https://openathena.ai/blog/delphi/> |
| Small models far above 20 tokens/parameter (overtraining) | Qwen3 | About 36T tokens for all sizes from 0.6B to 235B: <https://arxiv.org/abs/2505.09388> |
| | Llama 3 | 15T tokens for 8B (Delphi estimates about 90× the compute-optimal amount): <https://arxiv.org/abs/2407.21783> |
| | MobileLLM (Meta) | MobileLLM-R1-950M, 4.2T tokens: <https://arxiv.org/abs/2509.24945> |
| | Puro-2B (open recipe) | About 1.4T tokens, about 700 tokens/parameter (Appendix A): <https://www.alphaxiv.org/abs/2608.27370> |
| Muon optimizer | Kimi | Kimi K2 §2.1 MuonClip (Muon + weight decay + RMS matching 0.2 + QK-Clip): <https://arxiv.org/abs/2507.20534>; Moonlight: <https://arxiv.org/abs/2502.16982> |
| | GLM | GLM-4.5 §2.4 "We employed the Muon optimizer for all parameters except word embedding, bias, and weights for RMSNorm" (update RMS 0.2): <https://arxiv.org/abs/2508.06471> |
| | DeepSeek | DeepSeek-V4 §2.4 and §4.2.2 (AdamW for the embedding, the prediction head, and RMSNorm; Muon for the rest; update RMS 0.18; hybrid Newton–Schulz): <https://arxiv.org/abs/2606.19348> |
| | Others | Puro-2B (MuonH): <https://www.alphaxiv.org/abs/2608.27370>; nanochat: <https://github.com/karpathy/nanochat>; reference implementation: <https://kellerjordan.github.io/posts/muon/> |
| FP8 training | DeepSeek | DeepSeek-V3 §3.3 (fine-grained FP8 mixed-precision training): <https://arxiv.org/abs/2412.19437> |
| | Llama | Llama 4 blog "we focus on efficient model training by using FP8 precision… pre-training our Llama 4 Behemoth model using FP8": <https://ai.meta.com/blog/llama-4-multimodal-intelligence/> |
| | NVIDIA Nemotron | Nemotron-H "We pre-trained Nemotron-H-56B-Base on 20 trillion tokens in FP8": <https://research.nvidia.com/labs/adlr/nemotronh/> |
| | Others | Puro-2B (block-wise FP8, net 1.34× on 1.7B): <https://www.alphaxiv.org/abs/2608.27370> |
| WSD learning-rate schedule (used for the ladder branches) | Kimi | Kimi K2 §2.5 "the WSD learning rate schedule": <https://arxiv.org/abs/2507.20534> |
| | DeepSeek | DeepSeek-V4 §4.2.2 (the learning rate stays at 2.7e-4 for most of the training and decays at the end): <https://arxiv.org/abs/2606.19348> |
| | MiniCPM (proposed WSD) | <https://arxiv.org/abs/2404.06395>; counterexample: GLM-4.5 §2.4 reports that WSD was worse in its setup, and uses cosine instead |

To be verified: the pretraining sequence length 2048 of SmolLM2 (cited in item 3 of Section 9; not checked word for word against the model card). The H100 dense BF16 peak of 989.5 TFLOPS and the price of $2.5/GPU-hour (marked "to be verified" in `estimate_cost.py`).

## Guided questions

1. At a sequence length of 4096, attention is 40% of the FLOPs per token of the main line. FlashAttention skips the masked half under the causal mask. Is the measured MFU (with the `zero` counting convention) then too high or too low? What should you use to calculate the budget?
2. The coefficients of the original Chinchilla paper give about 78 tokens/parameter, and the Epoch replication gives about 20. How can a small problem such as "the optimizer stopped too early" change the conclusion by 4×? (Hint: look at how the ratio of α and β sets N_opt ∝ C^(β/(α+β)).)
3. In the first run of the mini ladder, with only 12 points, the fit gave α ≈ 0.05. Why can the fit not find the exponent for N when all points are in the region with too little data? Which item of the main-line ladder protocol addresses this problem?
4. Why does the bootstrap resample whole sizes as groups, and not each of the 14 points independently? With independent resampling, does the interval become wider or narrower?
5. On the mini ladder, Muon leads by 0.17–0.26 bit/byte, but this chapter recommends "decide after the ladder comparison". Give at least three reasons why an advantage on a tiny configuration does not transfer directly to 0.7B.
6. Suppose that the extrapolation of gate 1 shows that the main line cannot reach the tool-calling hard goal. Which part of the recipe do you change first? Why not "spend a little more money and train on more tokens"?

## Hands-on tasks

**Task 1 (basic)**: In `05_plan_budget.py`, change the price to a real H100 rental price that you can find, and change the MFU to 0.35. Calculate again how many tokens $5K buys. Then calculate the cost of the Puro-2B configuration (2B, 1.4T tokens) on 8×H100 at MFU 0.4. Compare it with the $6.9K for the RTX 5090 cluster in the paper. Where does the difference come from?

**Task 2 (core)**: In `03_lr_sweep.py`, change `SWEEP_TOKENS` to 262,144 and sweep again (use a different name for the cache file). How does the best learning rate change? The Delphi recipe says that the learning rate should decrease with the training length as (T₀/T)^0.3. Do your data support this?

**Task 3 (challenge)**: In `04_mini_ladder.py`, change the held-out size to dim 128 with 4 layers (N ≈ 800K, about 3.7× the largest ladder size). Train it again and test the extrapolation. When the extrapolation factor increases, how do the error and the bootstrap interval change? Then fit with only s1, s2, and s3, and use both s4 and s5 as held-out sizes. Plot the relation "extrapolation factor – error".

## Go deeper: CS336

This chapter matches **Lectures 9 and 11, "Scaling Laws"**, of [CS336](https://cs336.stanford.edu/) (Spring 2026). The two lectures go deeper into each step of this chapter. They cover these topics: the debate about methods from Kaplan to Chinchilla, IsoFLOP and parametric fits, how hyperparameters transfer across scale, and how technical reports use scaling laws to make decisions. **Assignment 3 (Scaling)** asks you to design experiments within a limited compute budget, fit a scaling law, and make predictions (see the course page for the current requirements). The mini ladder of this chapter is a smaller version of this assignment. After this chapter, the assignment will be much easier. The course page has the notes and the videos of each lecture.

## References

- Kaplan et al. *Scaling Laws for Neural Language Models*, 2020: <https://arxiv.org/abs/2001.08361>
- Hoffmann et al. *Training Compute-Optimal Large Language Models* (Chinchilla), 2022: <https://arxiv.org/abs/2203.15556>
- Besiroglu et al. *Chinchilla Scaling: A replication attempt* (Epoch AI), 2024: <https://arxiv.org/abs/2404.10102>
- Porian et al. *Resolving Discrepancies in Compute-Optimal Scaling of Language Models*, 2024: <https://arxiv.org/abs/2406.19146>
- Sardana & Frankle. *Beyond Chinchilla-Optimal: Accounting for Inference in Language Model Scaling Laws*, 2023: <https://arxiv.org/abs/2401.00448>
- Lourie, Cho, Ullrich, Lotfi. *Small-Scale Experiments: Are We There Yet?*, 2026: <https://arxiv.org/abs/2608.11859> (in `references.md`)
- Held (Marin / Open Athena). *Scaling Laws That Extrapolate 300× Past the Fit* (Delphi), 2026: <https://openathena.ai/blog/delphi/> (in `references.md`)
- Luo et al. *PuRo-2B: Poor Lab's Qwen2-1.5B Trained on RTX 5090 within $5090*, 2026: <https://www.alphaxiv.org/abs/2608.27370> (in `references.md`)
- DeepSeek-AI. *DeepSeek LLM: Scaling Open-Source Language Models with Longtermism*, 2024: <https://arxiv.org/abs/2401.02954>
- Qwen Team. *Qwen3 Technical Report*, 2025: <https://arxiv.org/abs/2505.09388>
- Llama Team. *The Llama 3 Herd of Models*, 2024: <https://arxiv.org/abs/2407.21783>
- Meta. *The Llama 4 herd* (blog), 2025: <https://ai.meta.com/blog/llama-4-multimodal-intelligence/>
- Zhao et al. *MobileLLM-R1*, 2025: <https://arxiv.org/abs/2509.24945>
- Jordan et al. *Muon: An optimizer for hidden layers in neural networks*, 2024: <https://kellerjordan.github.io/posts/muon/>
- Liu et al. *Muon is Scalable for LLM Training* (Moonlight), 2025: <https://arxiv.org/abs/2502.16982>
- Kimi Team. *Kimi K2: Open Agentic Intelligence*, 2025: <https://arxiv.org/abs/2507.20534>
- GLM-4.5 Team. *GLM-4.5: Agentic, Reasoning, and Coding (ARC) Foundation Models*, 2025: <https://arxiv.org/abs/2508.06471>
- DeepSeek-AI. *DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence*, 2026: <https://arxiv.org/abs/2606.19348>
- DeepSeek-AI. *DeepSeek-V3 Technical Report*, 2024: <https://arxiv.org/abs/2412.19437>
- NVIDIA. *Nemotron-H: A Family of Accurate, Efficient Hybrid Mamba-Transformer Models*, 2025: <https://research.nvidia.com/labs/adlr/nemotronh/>
- NVIDIA. *Faster Training Throughput in FP8 Precision with NVIDIA NeMo* (blog), 2025: <https://developer.nvidia.com/blog/faster-training-throughput-in-fp8-precision-with-nvidia-nemo/>
- Hu et al. *MiniCPM* (WSD), 2024: <https://arxiv.org/abs/2404.06395>
- [CS336](https://cs336.stanford.edu/), Lectures 9 and 11, Assignment 3

**Next chapter**: The ladder experiment fixes the "recipe", and the most important part of the recipe is the data. In Chapter 13, we collect, clean, deduplicate, and mix the pretraining data of the main line, and we train the tokenizer of the main line. The vocabulary size also changes the parameter count of this chapter.
