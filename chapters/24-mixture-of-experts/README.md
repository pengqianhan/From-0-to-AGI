# Chapter 24: Mixture of experts (MoE) — Many more parameters, the same compute for each token

**English** · [中文](README.zh.md)

> **Goal**: After this chapter, you can calculate the total parameters and the active parameters of a model from its `config.json`. You can write the routing of an MoE layer (score → top-k → normalize → weighted sum) and do a parity check against a naive token-by-token version. You can explain why routing collapses, and how the auxiliary loss and the "auxiliary-loss-free bias" of DeepSeek-V3 balance the load. You can also use two ledgers, memory and compute, to explain two facts. Almost all large models above some tens of billions of parameters are MoE. Almost all small models below 1B are dense.

📺 **Video**: Not published yet. To render it on your computer, run `bash chapters/24-mixture-of-experts/video/build.sh`.
🧪 **Self-check**: After the chapter, type `/ch24-moe` in Claude Code.

---

In the last three chapters, we reduced the cost of **attention**. GQA and MLA store less at each position. A sliding window lets some layers look only at nearby positions. Linear attention compresses the full history into a state of fixed size. This chapter solves the other half of the problem: **the model needs more parameters (more knowledge), but the compute for each token must not increase. What can we do?**

The answer is in the other sublayer of the Transformer, the feed-forward network (FFN). Split one large FFN into many small FFNs (**experts**). Then let a small **router** select only a few of them for each token. This is a **mixture of experts (MoE)**. DeepSeek-V3 has 671 billion parameters, but each token uses only 37 billion. Kimi K2 has 1 trillion parameters, but each token uses only 32.6 billion.

The idea of MoE sounds simple. The difficult part is the selection. We train the router with the model, and the router can prefer some experts. Thus **load balancing** is the main topic of this chapter. We write an MoE layer from zero and train some small language models on a CPU. We see what routing collapse looks like. Then we see how two balancing methods correct it.

The code of this chapter (all of it runs on a CPU):

```bash
uv run python chapters/24-mixture-of-experts/code/01_param_ledger.py    # parameter ledger: the FFN share; total/active parameters of 9 MoE models (seconds)
uv run python chapters/24-mixture-of-experts/code/02_moe_layer.py       # MoE layer from zero: parity check, auxiliary loss by hand, bias balancing (seconds)
uv run python chapters/24-mixture-of-experts/code/03_train_compare.py   # dense vs MoE, small language models with three balancing setups (slow the first time, then reads the cache)
uv run python chapters/24-mixture-of-experts/code/04_small_vs_large.py  # why small models seldom use MoE: lineups, same-data comparison, memory and reads (seconds)
```

## 1. Intuition: the parameters are in the FFN

Chapter 9 showed that each Transformer block has two sublayers. Attention moves information between positions. The FFN (SwiGLU) processes the information inside each position. First, count the parameters of each sublayer ([`code/01_param_ledger.py`](code/01_param_ledger.py), part 1):

| Model | Attention per layer | FFN per layer | FFN share |
|---|---:|---:|---:|
| Qwen3-8B (d = 4096, FFN width 12288) | 41.9M | 151.0M | **78.3%** |
| Main-line model (d = 1280, FFN width 3584) | 7.9M | 13.8M | 63.6% |

In the forward pass, each parameter gives about 2 floating-point operations (one multiplication and one addition). Thus this table also shows where the matrix-multiplication compute for each token goes: **most of it goes to the FFN**.

This causes a conflict. Scaling laws (Chapter 12) tell us that more parameters make a stronger model. But in a dense model, twice the parameters also means twice the compute for each token. Training costs twice as much, and inference is twice as slow. Can we **separate the parameters from the compute**?

The MoE answer: replace one wide FFN with N narrower FFNs (experts). Each token goes through only K of them.

- **Total parameters**: the count includes all N experts. They set how much "knowledge" the model can hold. They also set **how many weights the memory must hold**.
- **Active (activated) parameters**: the part that each token really uses (K experts + attention + embedding, and so on). They set **the compute for each token**.

The model names contain these two numbers. Qwen3-235B-A22B means "235B total, 22B activated". Qwen3.5-35B-A3B means "35B total, 3B activated".

## 2. The ledger of real models

Take the `config.json` of some MoE models. Count the parameters as "embedding + attention in each layer + MoE in each layer (router + all experts + shared experts) + lm_head" ([`code/01_param_ledger.py`](code/01_param_ledger.py), part 2):

| Model | Experts | Selected per token | Shared experts | Expert width | Total (calc / official) | Active (calc / official) | Active ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| Mixtral-8x7B (2023.12) | 8 | 2 | 0 | 14336 | 46.7B / 47B | 12.9B / 13B | 27.6% |
| Llama-4-Scout | 16 | 1 | 1 | 8192 | 107.8B / 109B | 17.2B / 17B | 15.9% |
| Qwen3-30B-A3B | 128 | 8 | 0 | 768 | 30.5B / 30.5B | 3.35B / 3.3B | 11.0% |
| Qwen3-235B-A22B | 128 | 8 | 0 | 1536 | 235.1B / 235B | 22.2B / 22B | 9.4% |
| GLM-4.5 | 160 | 8 | 1 | 1536 | 352.8B / 355B | 33.6B / 32B | 9.5% |
| DeepSeek-V3 | 256 | 8 | 1 | 2048 | 671.0B / 671B | 37.6B / 37B | 5.6% |
| gpt-oss-120b | 128 | 4 | 0 | 2880 | 116.8B / 116.83B | 5.71B / 5.13B | 4.9% |
| Kimi K2 | 384 | 8 | 1 | 2048 | 1026.4B / 1.04T | 32.9B / 32.6B | 3.2% |

Our counts agree with the official numbers in almost all cases. The small differences come from different counting methods (the script lists them one by one at the end).

The official GLM-4.5 numbers do not include the embedding and the output layer, but they include 1 MTP layer. Without the embedding and the output layer, we get 351.2B / 32.1B. The "active" count of gpt-oss does not include the input embedding. Without it, we get exactly 5.13B / 3.61B. The 109B of Llama 4 includes the vision encoder. For Kimi K2, our count is about 14B (1.3%) lower; the cause is **to be verified**.

The table shows three patterns:

1. **The routed experts have 90%–99% of the total parameters.** The parts that every token uses, such as attention and embedding, are only a small remainder.
2. **The active ratio becomes lower and lower.** Mixtral selects 2 of 8 experts (28%). Kimi K2 selects 8 of 384 (3%). The "sparsity scaling law" in the Kimi K2 report describes this trend. At fixed active parameters (thus at fixed compute), more experts in total give a lower validation loss. Kimi chose a sparsity of 48 = 384/8 as a trade-off between performance and infrastructure complexity.
3. **The experts become "finer".** One Mixtral expert has width 14336, which is wider than the FFN of a dense model. One DeepSeek-V3 expert has width 2048 only, and one Qwen3-30B-A3B expert has width 768 only. Section 7 explains why.

## 3. Routing: score, select K, weighted sum

### 3.1 Formula

An MoE layer (for the vector x of one token):

```
s_i = sigmoid(x · e_i)    or    s = softmax(x · W_r)          # 1. score: one score for each expert
S   = TopK(s + b, K)                                          # 2. select K experts (b is a bias, see Section 5; for now, b = 0)
g_i = s_i / Σ_{j∈S} s_j        (i ∈ S)                        # 3. gate weights: scores of the selected experts, normalized to a sum of 1
y   = Σ_shared FFN_s(x)  +  Σ_{i∈S} g_i · FFN_i(x)             # 4. calculate only the selected experts, weighted sum
```

- The **router** is a linear layer `d → N`. DeepSeek calls its rows the "centroid vectors" e_i of the experts. The router has very few parameters (DeepSeek-V3: 7168 × 256 ≈ 1.8M in each layer).
- There are two kinds of **score functions**. Softmax: Mixtral, Qwen3-MoE, gpt-oss. Sigmoid: DeepSeek-V3, GLM-4.5, Kimi K2, MiniMax-M2, Nemotron 3, Llama 4 (the HF implementation of Llama 4 applies a sigmoid to the top-1 logit). With softmax, the experts compete with each other (the scores add up to 1). With sigmoid, each expert gets an independent score, and the layer normalizes only the scores of the selected experts.
- **"Softmax first, then select K and normalize" and "select K logits first, then softmax" give the same result.** The Mixtral paper writes `Softmax(TopK(x·W_g))`. The Qwen3-MoE configuration uses softmax and then `norm_topk_prob: true`. Both give exactly the same g, because e^{l_i}/Σ_{j∈S} e^{l_j} is the same in both forms.
- Some models also multiply by a constant `routed_scaling_factor` (2.5 in DeepSeek-V3 and GLM-4.5). It only changes the scale of the output of the routed part.

### 3.2 Minimal code

The core of [`code/02_moe_layer.py`](code/02_moe_layer.py) is a direct translation of the formula:

```python
logits = self.router(x)                                      # (T, E)
s = logits.sigmoid() if self.score == "sigmoid" else logits.softmax(-1)
idx = (s + self.bias).topk(self.K, dim=-1).indices           # (T, K) select experts: the bias only selects
g = s.gather(-1, idx)                                        # (T, K) gates: the original scores
g = g / g.sum(-1, keepdim=True)                              # normalize to a sum of 1

out = self.shared(x) if self.shared is not None else torch.zeros_like(x)
for e, expert in enumerate(self.experts):
    tok, slot = (idx == e).nonzero(as_tuple=True)            # the tokens that selected expert e
    if tok.numel():
        out.index_add_(0, tok, g[tok, slot, None] * expert(x[tok]))   # weighted combination
```

Look at the loop: it goes **over the experts**, not over the tokens. Each expert collects the tokens that selected it and calculates them with one matrix multiplication. Then `index_add_` adds the weighted results back to their positions. These are the two steps of an MoE implementation: **dispatch** and **combine**. The script does a parity check against a naive token-by-token version:

```
2) Grouped by expert vs naive token by token
  softmax max difference 2.2e-16
  sigmoid max difference 1.4e-16
```

The parameter ledger (d = 128; these are the configurations of the small experiment in Section 6):

| FFN | Total parameters | Active parameters |
|---|---:|---:|
| Dense SwiGLU, width 384 | 147,456 | 147,456 |
| MoE: 8 experts × width 192, select 2 | 590,848 | 148,480 |
| Fine-grained: 16 experts × width 96, select 3 + 1 shared | 628,736 | 149,504 |

The active parameters are almost the same (the extra thousand or so are the router). The total parameters are more than 4 times those of the dense FFN.

## 4. The problem: routing collapse

The router trains together with the model. This causes a positive feedback loop. At the start, one expert gets a few more tokens. → It gets more gradient and learns better. → The router selects it more often. → It gets even more tokens… At the end, a few experts get most of the tokens, and the other experts are almost idle. This is **routing collapse** (Shazeer et al. described this problem in 2017).

Routing collapse has two costs:

- **Wasted parameters**: the parameters of idle experts use memory for nothing. The model becomes, in effect, a smaller model.
- **Slower training and inference**: in a large model, the experts are on different GPUs (**expert parallelism**, Section 9). The GPU with the busiest expert sets the time of the full layer. With a capacity limit, the layer also drops tokens (Section 5.3).

> **Note:** About the numbers: the numbers of the training experiments in this chapter come from one CPU run on the course build machine. Different machines and different versions of the low-level math libraries use a slightly different order of floating-point operations. After some hundred training steps, these small differences become larger. Your numbers can differ from the second or third decimal place. Trust the conclusions below that do not depend on exact values. For a rerun on another server in 2026-10, see [runs/2026-10-01-gpu0-check/chapters-24-26.md](../../runs/2026-10-01-gpu0-check/chapters-24-26.md).

In the small experiment of this chapter (Section 6), an MoE without any balancing has this load in layer 1 during training:

| Step | Load share of the 8 experts in layer 1 (counted from 0; uniform is 0.125) | Max / mean |
|---:|---|---:|
| 0 | 0.30 0.00 0.02 0.19 0.11 0.03 0.35 0.01 | 2.76 |
| 100 | 0.27 0.09 **0.42** 0.06 0.03 0.01 0.01 0.12 | 3.34 |
| 300 | 0.24 0.08 **0.45** 0.05 0.05 **0.00** 0.02 0.12 | 3.61 |
| 800 | 0.22 0.08 **0.45** 0.06 0.05 **0.00** 0.02 0.11 | 3.63 |

(With 2 of 8 experts selected, the upper limit of "max/mean" is 8/2 = 4.) The third expert grows from 2% to 45%. The sixth expert falls from 3% to 0: at the end of training, the router almost never selects it.

We must state two facts honestly. First, **a router with random initialization is not balanced at the start** (step 0 already has 2.76). The token vectors in one layer point in similar directions, so some experts always get higher scores. Second, in this small experiment of 800 steps, not every layer becomes worse. The "max/mean" values of the 4 layers change from 3.24, 2.76, 2.59, 2.14 to 2.25, 3.63, 2.46, 2.79. But **no layer returns to a balance by itself**.

## 5. Load balancing

### 5.1 Method 1: auxiliary loss (Switch / GShard)

The classic method adds an **auxiliary loss** to the language-model loss. The auxiliary loss penalizes an unbalanced load (GShard 2020, Switch Transformer 2021). This chapter uses the form of equations 17–20 in the DeepSeek-V3 report. The original Switch form is the special case K = 1.

```
f_i = N / (K·T) · (number of tokens in these T tokens that selected expert i)   # actual load (= 1 when balanced)
P_i = (1/T) · Σ_t s'_{i,t}                                                     # mean router probability of expert i (s' normalized per row)
L_aux = α · Σ_i f_i · P_i                                                       # = α with a perfect balance
```

f is a discrete count, so it has no gradient. It acts only as a "weight". The gradient goes only through P: if the f of an expert is large (the expert is overloaded), the gradient pushes its routing probability down harder. The code has only three lines:

```python
p = s / s.sum(-1, keepdim=True)                          # routing probabilities, normalized per row
f = self.load * self.E / (self.K * x.shape[0])           # f_i (= 1 when balanced)
self.aux = self.aux_coef * (f * p.mean(0)).sum()         # L_aux = α · Σ f_i · P_i
```

An example by hand: 4 tokens, 2 experts, top-1. The routing probabilities are [0.9, 0.1], [0.8, 0.2], [0.7, 0.3], and [0.4, 0.6]. The first 3 tokens select expert 1, and the last token selects expert 2. Thus f = 2/4 × [3, 1] = [1.5, 0.5], and P = the column means = [0.7, 0.3]. L = α × (1.5 × 0.7 + 0.5 × 0.3) = **1.2α**, which is larger than the α of a balanced load. The script gives 0.0120 (α = 0.01).

The problem of the auxiliary loss: its gradient works against the gradient of the language model. If α is too small, it cannot stop the collapse. If α is too large, it decreases the model quality. The DeepSeek-V3 report cites the experiments of Wang et al. (2024) on this problem as the reason for its new method.

> **Note:** Do not copy the coefficients from one implementation to another. Different implementations use different constants. In the Hugging Face implementations of Mixtral and Qwen3-MoE, f is not divided by K, so the result is K times the result of this chapter. Read each `router_aux_loss_coef` in a configuration (Mixtral 0.02, Qwen3 0.001, Llama 4 0.001, OLMoE 0.01) together with its own implementation.

### 5.2 Method 2: the auxiliary-loss-free bias (DeepSeek-V3)

DeepSeek-V3 uses a different idea: **do not change the loss; change the selection directly**. Each expert gets a bias b_i. The router adds the bias to the scores only when it selects the top-K (line 2 of the formula in Section 3.1). The gate weights g still use the original scores s. The bias is not a parameter and gets no gradient. After each training step, the code updates the bias from the load of the full batch of that step:

```
b_i ← b_i + γ · sign(mean load − load_i)       # subtract γ for an overloaded expert, add γ for an underloaded expert
```

```python
@torch.no_grad()
def update_bias(self, load=None):
    load = self.load if load is None else load
    self.bias += self.bias_speed * torch.sign(load.mean() - load)
```

The bias changes only "which experts the router selects", not "how large the output is". Thus it does not disturb the gradient of the language model. An overloaded expert slowly loses some tokens, and an underloaded expert slowly gets more tokens.

Part 4 of [`code/02_moe_layer.py`](code/02_moe_layer.py) gives the cleanest demonstration. It fixes a router with a strong preference. The router has 8 experts and selects 2. We increase the router weights of experts 1 and 2 by hand (indices 0 and 1 in the code), and all inputs are positive. The demonstration **trains no weights and only updates the bias** (γ = 0.01):

| Bias updates | Load share of each expert | Max / mean | Dropped at capacity factor 1.25 |
|---:|---|---:|---:|
| 0 | 0.50 0.10 0.00 0.32 0.00 0.08 0.00 0.00 | 4.00 | 50.3% |
| 10 | 0.48 0.12 0.04 0.12 0.06 0.12 0.01 0.04 | 3.87 | 32.8% |
| 30 | 0.13 0.12 0.12 0.13 0.13 0.12 0.13 0.13 | 1.05 | 0.0% |
| 300 | 0.13 0.12 0.12 0.13 0.13 0.12 0.13 0.13 | 1.05 | 0.0% |

(8 experts, select 2: a uniform load is 0.125 for each expert. The upper limit of "max/mean" is E/K = 4, and step 0 is already at the limit.)

The actual settings of DeepSeek-V3 (report Section 4.2): γ = 0.001. For the last 500B tokens, γ = 0, but the model continues to use the learned bias. **DeepSeek-V3 also keeps a very small sequence-level auxiliary loss**, α = 0.0001, "only to prevent an extreme imbalance inside a single sequence". Thus "auxiliary-loss-free" does not mean "no auxiliary loss at all". It means **mainly the bias**. GLM-4.5 (bias update rate 0.001 + sequence-level loss 0.0001) and NVIDIA Nemotron 3 Nano (update rate 1e-3 + standard load-balancing loss 1e-4) use almost the same recipe.

**Why is it better?** Table 5 of the DeepSeek-V3 report compares the two methods with the same data and the same architecture (both with sigmoid + top-K normalization). For the model with 15.7B total parameters, the Pile test-set BPB changes from 0.727 to 0.724. For the model with 228.7B, it changes from 0.656 to 0.652. Most downstream benchmarks are also better.

Section 4.5.3 of the report gives a more interesting analysis. The key is not "a loss or no loss", but **the scope of the balance**. A sequence-level auxiliary loss requires a balance inside each sequence, so the experts cannot specialize by domain. The bias method requires a balance only over the full batch. The authors also tried a "batch-level auxiliary loss". On a 1B MoE, the validation losses were 2.258 (sequence-level), 2.253 (bias method), and 2.253 (batch-level loss), so the last two are equal.

The "global-batch load-balancing loss" of Qwen3 (Qiu et al. 2025) follows this path. It still uses an auxiliary loss, but it calculates the statistics over the global batch.

### 5.3 The capacity factor and "no dropped tokens"

Early MoE models (GShard, Switch Transformer) used a fixed shape for the calculation of each expert. Thus each expert has a **capacity**: it processes at most `capacity factor × T·K / N` tokens. A token above the capacity skips this expert. Its output from this expert is 0, and the residual connection passes the token on as usual. This is **token dropping**. The last column of the table above shows this effect. With a skewed load, a capacity factor of 1.25 drops half of the assignments. After the balance, it drops none.

Today, most models **do not drop tokens (dropless)**. MegaBlocks (Gale et al. 2022) writes the MoE as a block-sparse matrix multiplication, so each expert calculates all the tokens that it gets. The DeepSeek-V3 report states that it drops no tokens "in training and inference". The condition is a load that is balanced enough. The production code of this chapter supports both methods (`capacity_factor=None` means dropless).

## 6. Small experiment: the same small model, dense vs MoE, three balancing setups

[`code/03_train_compare.py`](code/03_train_compare.py) uses the character-level Shakespeare model of Chapter 10 (4 layers, width 128, 4 heads, Pre-Norm RMSNorm + RoPE). It changes only the FFN:

| Variant | FFN structure | Notes |
|---|---|---|
| Dense-384 | SwiGLU, width 384 | The baseline for active parameters (≈ compute for each token) |
| Dense-1536 | SwiGLU, width 1536 | About the same total parameters as the MoE |
| MoE-no-balance | 8 experts × width 192, select 2 | Active width 2 × 192 = 384, the same compute as Dense-384 |
| MoE-aux-loss | The same + L_aux (α = 0.01) | |
| MoE-aux-free | The same + bias (γ = 0.01) | |
| Fine-grained+shared | 16 experts × width 96, select 3, + 1 shared expert of width 96, bias | Active width 4 × 96 = 384 |

All MoE variants use sigmoid scores + top-K normalization. This is the same setting as the comparison in Table 5 of the DeepSeek-V3 report. All variants use the same data order, 800 steps, and AdamW + warmup + cosine. Each variant runs with 2 random seeds. γ is 0.01, 10 times the 0.001 of DeepSeek-V3. We train only 800 steps, so the bias must move faster.

Results (`uv run python chapters/24-mixture-of-experts/code/03_train_compare.py`; the FFN parameters are the sum over the 4 layers):

| Variant | FFN total parameters | FFN active parameters | Validation loss mean [seed 0, 1] | Load max/mean (mean over 4 layers × 2 seeds, worst) | Experts with load < 1% (max in one layer) |
|---|---:|---:|---|---:|---:|
| Dense-384 | 589,824 | 589,824 | 1.793 [1.800, 1.787] | — | — |
| Dense-1536 | 2,359,296 | 2,359,296 | 1.815 [1.815, 1.816] | — | — |
| MoE-no-balance | 2,363,392 | 593,920 | 1.780 [1.760, 1.799] | 2.61, 3.63 | 1 |
| MoE-aux-loss | 2,363,392 | 593,920 | 1.786 [1.772, 1.799] | 1.14, 1.26 | 0 |
| MoE-aux-free | 2,363,392 | 593,920 | 1.786 [1.774, 1.797] | 1.12, 1.20 | 0 |
| Fine-grained+shared | 2,514,944 | 598,016 | 1.777 [1.778, 1.776] | 1.15, 1.22 | 0 |

For the same variant, a different seed changes the validation loss by up to 0.039. The load of layer 1 during training (seed 0):

| Step | No balancing | Auxiliary loss (α = 0.01) | Bias (γ = 0.01) |
|---:|---:|---:|---:|
| 0 | 2.76 | 2.76 | 2.76 |
| 100 | 3.34 | 1.76 | 1.08 |
| 300 | 3.61 | 1.34 | 1.18 |
| 800 | 3.63 | 1.05 | 1.20 |

What we can conclude (and what we cannot):

- **Both load-balancing methods work.** At the end of training, "max/mean" is between 1.05 and 1.26 for both methods, and no expert is idle. Without balancing, the worst layer has 3.63, and one expert is fully idle. In this setup, the bias method balances the load faster: layer 1 reaches 1.08 at step 100, but the auxiliary loss needs 300–800 steps. This agrees with the fact that the bias acts directly on the selection at each step. But the speed depends on the values of γ and α, so we cannot make a general conclusion.
- **All differences in validation loss are within the noise.** The five variants with the same active compute (Dense-384 and the four MoE variants) are between 1.777 and 1.793. The differences are smaller than the 0.039 between seeds. Thus we **cannot** say that MoE is better. We also cannot say that load balancing improves the quality. With 800 steps on character-level data, the "wasted parameters" of an unbalanced load do not yet cause a difference in loss.
- **Dense-1536, with "the same total parameters", is the worst** (1.815, with both seeds). A dense FFN that is 4 times wider also uses 4 times the compute for each token. But with the same 800 steps and the same learning rate, it does not train better. At this scale, the amount of data and the number of steps are the bottleneck. Remember this: "more parameters at the same compute" improves the quality of an MoE only with enough data and long enough training. (The DeepSeekMoE and Qwen3 reports use trillions of tokens.)

**This is a tiny experiment with a million parameters and a few minutes of training.** It only shows that the code is correct and approximately what the effects look like. Do not extrapolate it to large models. For serious comparisons, see three sources. Table 5 of the DeepSeek-V3 report compares the two balancing methods. The DeepSeekMoE paper compares coarse and fine-grained experts, with and without shared experts. Figure 5 of the Kimi K2 report shows the effect of sparsity.

## 7. Fine-grained experts + shared experts (DeepSeekMoE)

The table in Section 2 shows two main structural differences between the large models of today and Mixtral. Both come from DeepSeekMoE (Dai et al. 2024).

**Fine-grained experts**: split each expert into m parts (the width becomes 1/m), and let each token select m times as many experts. The total parameters and the active parameters do not change, but **the number of possible combinations becomes much larger**. The example in the paper: 16 experts with 2 selected give only C(16,2) = 120 combinations. Split each expert into 4 parts and select 8 of 64: this gives C(64,8) ≈ 4.4 billion combinations. With more combinations, each token can build a better set of experts, and each expert can specialize more. (Figure 3 of the paper: with the same parameters and the same active parameters, finer experts give better results.)

The cost: when the experts become too small, the matrix multiplications become less efficient, and the routing and communication overhead becomes larger. Thus the experts do not become finer without a limit. (DeepSeekMoE 16B did not split its experts further, because of efficiency.)

**Shared experts**: keep one or two experts that **every token goes through**, without routing. The motivation in the paper: different experts all need some common knowledge (grammar, frequent words). Without shared experts, many routed experts learn this knowledge again and again. With shared experts, the routed experts can specialize more.

An ablation in the paper shows this well. Remove the shared expert of a 2B model, and activate one more routed expert instead (the compute does not change). The Pile loss increases from 1.808 to 2.414.

But **not all model families use shared experts**. Their use is a spectrum (verified in 2026-09):

| With shared experts | Without shared experts |
|---|---|
| DeepSeek-V3 / V3.2 (1), Kimi K2 (1), Kimi K3 (2), GLM-4.5 / GLM-5 (1), Llama 4 (1), Qwen3.5-MoE (1, plus a sigmoid gate), Mistral Large 3 (1), Nemotron 3 Nano (the report gives 2; the configuration has 1 shared expert of double width, which is equivalent) | Qwen3-MoE (the report: "Unlike Qwen2.5-MoE, the Qwen3-MoE design excludes shared experts"), gpt-oss, Mixtral, MiniMax-M2 (`shared_intermediate_size: 0`), Xiaomi MiMo-V2-Flash (`n_shared_experts: null`), OLMoE |

Qwen itself changed its choice more than once. Qwen2.5-MoE had shared experts, Qwen3-MoE removed them, and Qwen3.5 added them again (model card: "8 Routed + 1 Shared"). Thus a more accurate statement is: **fine-grained experts are a consensus; shared experts are the choice of most families, but they are not necessary**.

In the small experiment of this chapter, the Fine-grained+shared variant (select 3 of 16 + 1 shared) has a validation loss of 1.777. Its difference from the other variants with the same compute is also within the noise. The difference between its two seeds also changes with the machine. In the table of Section 6, the values are 1.778 and 1.776, almost the same. A rerun on another server gave 1.757 and 1.778, a difference of 0.021. With only two seeds, we cannot say that it is more stable or less stable.

## 8. Why all large models use MoE, but small models seldom use it

First, look at the lineups ([`code/04_small_vs_large.py`](code/04_small_vs_large.py), part 1; official repository names, 2026-09):

| Family | Dense | MoE (total / active) |
|---|---|---|
| Qwen3 | 0.6B, 1.7B, 4B, 8B, 14B, 32B | 30B-A3B (30.5 / 3.3), 235B-A22B |
| Qwen3.5 | 0.8B, 2B, 4B, 9B, 27B | 35B-A3B, 122B-A10B, 397B-A17B |
| gpt-oss | — | 20b (20.9 / 3.6), 120b (116.8 / 5.1) |
| Llama 4 | — | Scout (109 / 17), Maverick (400 / 17) |
| Nemotron 3 | — | Nano 30B-A3B (31.6 / 3.2) |

In one family, almost all models below some tens of B are dense. The smallest MoE has more than 20B total parameters. Why? There are three ledgers.

**Ledger 1: what should we compare an MoE with?** For Table 5 of the Qwen3 technical report, the Qwen team trained Qwen3-30B-A3B and the dense Qwen3-14B on the same pretraining data:

| Base model | MMLU | MMLU-Pro | GSM8K | MATH | EvalPlus | MGSM | Total / active |
|---|---:|---:|---:|---:|---:|---:|---|
| Qwen3-14B (dense) | 81.05 | 61.03 | 92.49 | 62.02 | 72.23 | 79.20 | 14B / 14B |
| Qwen3-30B-A3B (MoE) | 81.38 | 61.49 | 91.81 | 59.04 | 71.45 | 79.11 | 30B / 3B |

The scores are similar. The MoE has **only about one fifth of the active parameters of the dense model** (the compute for each token is more than 4 times smaller). But **it has more than twice the total parameters**. In other words, an MoE "trades memory for compute".

**Ledger 2: the memory holds the total parameters.** All weights must stay in memory, including the weights of an expert that no token selects in this step:

| Model | Active | BF16 weights | 4-bit weights |
|---|---:|---:|---:|
| Qwen3.5-0.8B (dense) | 0.8B | 1.6 GB | 0.4 GB |
| Qwen3.5-9B (dense) | 9.0B | 18.0 GB | 4.5 GB |
| Qwen3-30B-A3B | 3.3B | 61.0 GB | 15.2 GB |

A phone with 8 GB of memory cannot hold a 30B-class MoE, even with 4-bit quantization. **On a device, memory is the tightest limit, so a dense model makes better use of the same memory.** On a cluster with thousands of GPUs, more GPUs give more memory, and compute and time are the most expensive parts. There, an MoE is a very good choice.

**Ledger 3: the number of weights to read depends on the batch.** Decoding is limited by memory bandwidth (Chapter 21). In each step, an MoE must read the experts that the tokens of this batch selected. With uniform routing, the fraction of experts in one layer that at least one token selects = 1 − (1 − K/N)^B:

| Model | batch 1 | batch 4 | batch 16 | batch 64 | batch 256 |
|---|---:|---:|---:|---:|---:|
| Qwen3-30B-A3B (8 of 128) | 6.2% | 22.8% | 64.4% | 98.4% | 100.0% |
| DeepSeek-V3 (8 of 256) | 3.1% | 11.9% | 39.8% | 86.9% | 100.0% |
| Mixtral (2 of 8) | 25.0% | 68.4% | 99.0% | 100.0% | 100.0% |

One user (batch 1) reads only the small active part. Thus, **if the memory can hold the model**, a local model such as 30B-A3B decodes at about the speed of a 3B dense model. On a server with a large batch, almost every expert must be read. The MoE saves compute for each token, not memory bandwidth.

Small models also have some special disadvantages. The FFN of each token is already small. If we split it into tens of experts, the matrix multiplication of each expert is too small to use the GPU fully. The overhead of routing, dispatch, and combine becomes a larger part of the time.

Also, **a comparison at the same size uses the total parameters** (the list of competitors in Section 3.2 of GOAL.md uses this rule). An MoE with 0.8B total parameters has only one or two hundred million active parameters. Its quality is closer to a dense model with one or two hundred million parameters. This is why the main-line model stays dense (GOAL.md 3.3).

> **Note:** There are exceptions. OLMoE-1B-7B from AllenAI (7B total, 1B active, select 8 of 64 experts) is a fully open small MoE. Researchers often use it as a baseline. But the main small models that this chapter verified (Qwen3 0.6–14B, Qwen3.5 0.8–27B) are all dense.

## 9. Expert parallelism (short overview)

The experts of a large MoE do not fit on one GPU. Thus we put different experts on different GPUs. This is **expert parallelism (EP)**.

The forward pass of each MoE layer then has three steps:

1. **Dispatch**: all-to-all communication sends each token to the GPUs of the experts that it selected.
2. Each GPU calculates its own experts.
3. **Combine**: another all-to-all sends the results back for the weighted combination.

Expert parallelism causes two new problems:

- **Communication**: DeepSeek-V3 puts its 256 routed experts on 64 GPUs in 8 nodes. It uses "node-limited routing", so each token goes to at most 4 nodes. First, it selects the nodes by the sum of the highest scores in each node. Then it selects the experts in these nodes. DeepSeek also wrote its own all-to-all kernels, which overlap with the calculation. Kimi K2 uses the smallest possible EP = 16, so that the calculation can fully hide the communication.
- **Load**: all GPUs of one layer must wait for each other, and the busiest expert sets the time of the full layer. Thus load balancing is not only about "do not waste parameters". It is also a throughput problem. For inference, DeepSeek-V3 also makes several copies of the experts with a high load ("redundant experts"). It adjusts the copies every 10 minutes from statistics of the live service.

On one GPU, the matching problem is this: different experts get different numbers of tokens, so we cannot write one regular large matrix multiplication. GPUs use a **grouped GEMM** or a block-sparse matrix multiplication in the MegaBlocks style. The CPU code of this chapter uses a Python loop instead.

In concept, a grouped GEMM calculates a group of matrix multiplications with different shapes in one call. Whether it really becomes one fused kernel depends on the implementation and the hardware. The section "GPU measurements" of this chapter used a profiler: on an RTX 3090, the PyTorch `F.grouped_mm` actually launches 128 cuBLAS GEMMs, one for each expert.

## 10. Summary

- **The FFN has most of the parameters and the compute** (78% of each layer in Qwen3-8B). MoE splits the FFN into N experts, and each token uses only K. This **separates the total parameters (memory, knowledge capacity) from the active parameters (compute for each token)**. DeepSeek-V3 activates 5.6%, and Kimi K2 only 3.2%.
- **Routing**: a linear layer gives scores (softmax or sigmoid) → add the bias to the scores and select the top-K → normalize the selected scores to get the gate weights → weighted sum. The implementation groups the tokens by expert for dispatch and combine.
- **Load balancing**: without it, routing collapses. The auxiliary loss α·Σf_i·P_i uses the gradient to push down overloaded experts, but it disturbs the main loss. The bias method of DeepSeek-V3 changes only the selection, and updates the bias by ±γ from the batch load after each step. The key is to balance over the full batch, not over each sequence. Today, most models do not drop tokens.
- **Fine-grained experts** (more and smaller experts) are a consensus. Most families use **shared experts**, but Qwen3-MoE, gpt-oss, Mixtral, and MiniMax-M2 do not. Their use is a spectrum.
- **An MoE trades memory for compute.** It is a very good choice to train and serve models with hundreds of B parameters on a cluster. On a device, memory is the tightest limit, and a dense model uses the same memory better. Thus the main small models at the 1B scale are all dense, and so is the main-line model.

---

## GPU measurements (one RTX 3090)

> **Note:** All numbers in the main text above come from CPU runs. This section uses one NVIDIA GeForce RTX 3090: 24 GB of memory, Ampere architecture. Spec sheet: dense BF16 tensor-core peak about 71 TFLOPS, FP32 about 35.6 TFLOPS, memory bandwidth about 936 GB/s. Environment: PyTorch 2.11.0+cu128, CUDA 12.8, October 2026.
>
> The server sets the power limit of this card to 240 W (the factory default is 350 W). Under a continuous full load, the card decreases its clock speed. Thus the absolute compute and bandwidth are lower than on a 3090 at full power, and the relative values are more reliable. If you have no GPU, skip this section.

Run:

```bash
uv run python chapters/24-mixture-of-experts/code/05_gpu_moe.py
```

One MoE layer gets 8192 tokens at a time (the size of one batch in training or prefill). It has d = 1024 and an expert width of 2048, and each token selects 2 experts. The active width is 4096, which is the same compute as one dense SwiGLU of width 4096. The number of experts E grows from 8 to 128.

We compare two versions. The first is the **loop over experts** of Section 3.2 (`MoE.forward` of `02_moe_layer.py`, moved to the GPU without changes). The second is **sort into segments + `F.grouped_mm`**. It sorts the tokens by expert and calculates all experts with one call for each matrix. Thus the full forward pass never stops to wait for the GPU.

The weights have random initialization in BF16. CUDA events measure the time, and we take the median of 30 runs. For each E, a parity check runs first. The max difference between the outputs of the two versions is less than 2% of the max output (the size of BF16 rounding errors).

The dense SwiGLU with the same compute (width 4096, 13M parameters): 4.27 ms, 48.3 TFLOPS.

| Experts E | Total parameters | GFLOP per token | Mean tokens per expert | Loop over experts, ms | Sort + grouped_mm, ms | Loop ÷ grouped | Measured grouped TFLOPS |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 50M | 0.0252 | 2048 | 6.45 | 6.03 | 1.1× | 34.2 |
| 16 | 101M | 0.0252 | 1024 | 7.15 | 6.32 | 1.1× | 32.7 |
| 32 | 201M | 0.0252 | 512 | 9.14 | 6.56 | 1.4× | 31.5 |
| 64 | 403M | 0.0253 | 256 | 15.45 | 8.74 | 1.8× | 23.7 |
| 128 | 805M | 0.0254 | 128 | 28.79 | 10.05 | 2.9× | 20.7 |

Here you can see "separate the parameters from the compute" (Section 1) directly. The total parameters grow 16 times, but the FLOPs per token do not change (the small extra part is the router). The time of the grouped version grows only from 6.03 ms to 10.05 ms.

But "the same FLOPs" does not mean "the same time". At E = 8, the MoE is already 40% slower than the dense FFN with the same compute, because routing, sorting, dispatch, and combine all take time. At E = 128, each expert gets only 128 tokens on average. The matrix multiplications are then too small to keep the GPU busy, and the measured compute falls from 48.3 to 20.7 TFLOPS. This is the effect of Section 7: "the matrix multiplications become less efficient when the experts become too small".

The Python loop over experts becomes worse as the number of experts grows: at E = 128, it is 2.9 times slower. Each expert must call `nonzero` once and stop to wait for the GPU to tell it which tokens it has. (On this machine, Python also needs about 6.4 µs to launch each operation.) This time is spent on the CPU, so it changes with the machine load.

One unexpected detail: the profiler shows that `F.grouped_mm` launches 129 kernels on this 3090. 128 of them are cuBLAS matrix multiplications, one for each expert, and they are not fused into one kernel. (Section 9 said "it depends on the implementation and the hardware". This is an example.) `F.grouped_mm` is fast because its loop is in C++ and does not wait for the GPU. Special implementations, such as MegaBlocks and the fused MoE of vLLM / SGLang, really combine a group of matrix multiplications into one kernel.

## From minimal code to production code

| Minimal code (`code/`) | Production code (`zero/arch/moe.py`) | What it adds, and why |
|---|---|---|
| `MoE` in `02_moe_layer.py`: an `nn.ModuleList` holds the experts; a loop over the experts | `MoEFFN`: the expert weights are **stacked** by expert into 3D tensors `(E, d, h)`; the tokens are **stably sorted** by expert and calculated segment by segment | Stacked weights are necessary for grouped GEMM and for the shards of expert parallelism (split along dimension 0 to different GPUs). After the sort, the tokens of each expert are contiguous, so a GPU kernel can replace the loop. (The forward and backward passes with CUDA + BF16 are verified on an RTX 3090. This check also fixed a precision bug of `index_add_` under autocast; see Section 12 of [runs/2026-10-01-gpu0-check](../../runs/2026-10-01-gpu0-check/README.md). The grouped GEMM kernel is not connected yet.) |
| Fixed sigmoid or softmax, top-k normalization | `MoEConfig`: `score_func`, `norm_topk_prob`, `routed_scaling_factor`, `n_shared_experts` / `shared_expert_dim`; the field names match the HF configurations | It can express different recipes: DeepSeek-V3 (sigmoid, normalization, ×2.5, 1 shared expert), Qwen3-MoE (softmax, normalization, no shared expert), Mixtral, and others |
| The bias update uses the load of this step | `load_accum` adds up the load since the last `update_bias()`; with distributed training, it first does an `all_reduce` | With gradient accumulation, the update must use the load of the full global batch (DeepSeek-V3: "monitoring the expert load on the whole batch"). The multi-GPU path is **not verified on a GPU yet** |
| The bias takes part only when `balance="free"` | The bias always takes part in the selection (it stays 0 when there are no updates), and it is a buffer in the `state_dict` | DeepSeek-V3 sets γ to 0 for the last 500B tokens, but continues to use the learned bias. To resume training, you must restore the bias |
| None | `capacity_factor`: an optional capacity limit, which records `last_dropped`; the default `None` = dropless | For comparison with the token dropping of Switch / GShard |
| Add `m.aux` and call `update_bias` by hand in the training loop | `MoETransformer` (`moe_transformer(model_cfg, moe_cfg, first_dense)`): the first `first_dense` layers keep a dense FFN (3 for DeepSeek-V3, 1 for Kimi K2); `loss()` adds the auxiliary losses of all layers automatically; `after_step()` updates the biases; `load_stats()`, `param_counts()` | It reuses the attention, RMSNorm, and initialization rules of the main line directly. The main-line `Trainer` does not call `after_step` yet (we connect it in Step 2, if we use the `Trainer` to train an MoE) |
| None | Industry implementations: `Qwen3MoeSparseMoeBlock` and `DeepseekV3MoE` in HF transformers (with `e_score_correction_bias` and group top-k); MegaBlocks; DeepEP from DeepSeek (an open expert-parallel communication library); the fused MoE kernels of vLLM / SGLang | The MoE of this course aims only to be readable and correct. For real training and deployment, use these implementations |

**Parity checks** (`uv run pytest tests/test_arch_moe.py`; all 12 tests passed on this machine, in about 5 seconds):

- With 1 expert and top-1, `MoEFFN` and `zero.model.SwiGLU` differ by < 1e-12 in float64.
- The sort-and-segment version gives exactly the same result as a naive token-by-token loop (softmax / sigmoid, with and without normalization, `routed_scaling_factor`, shared experts, nonzero bias; 4 combinations).
- The bias changes the selection but not the gate weights. The auxiliary loss agrees with the calculation by hand (1.2α; = α when balanced), and its gradient pushes down overloaded experts.
- The bias update has the correct direction. Bias updates alone bring the load of a fixed skewed router from "max/mean > 2" to < 1.2.
- Capacity factor: the number of dropped tokens agrees with the calculation by hand. With a sufficiently large capacity, the output is the same as dropless.
- `MoETransformer` trains, the auxiliary loss has a gradient, the bias is in the `state_dict`, and the count of active parameters is correct.

---

## Frontier notes

> **Sparsity continues to increase.** The "sparsity scaling law" of the Kimi K2 report (at fixed active parameters, more experts give a lower loss) is a reason to add more experts. The Kimi K3 configuration selects 16 of 896 experts, plus 2 shared experts. Qwen3.5-397B-A17B selects 10 of 512. But with more experts, routing, communication, and load balancing become more difficult. The Kimi K2 report also states that a sparsity of 48 is a trade-off between performance and infrastructure complexity. There is no consensus yet about where the optimal sparsity will stop.

> **Is the "scope" of the balance more important than the "method"?** Section 4.5.3 of the DeepSeek-V3 report found that a batch-level auxiliary loss and the bias method give equal results. Qwen uses an auxiliary loss over the global batch, and the DeepSeek family uses the bias. Both paths were successful in large models. Each has engineering advantages and disadvantages (the bias method does not disturb the gradient; the auxiliary loss needs no extra state). It is not decided yet which one is better.

---

## Adopters and sources

| Technique | Adopters (main versions) | Sources |
|---|---|---|
| MoE (fine-grained, tens to hundreds of experts) | DeepSeek (V3: 8 of 256), Qwen (Qwen3: 8 of 128; Qwen3.5: 10 of 512), Kimi (K2: 8 of 384), GLM (4.5: 8 of 160), gpt-oss (120b: 4 of 128), Llama 4, MiniMax (M2: 8 of 256), Mistral (Large 3: 4 of 128), NVIDIA Nemotron 3 (6 of 128) | The technical reports and the `config.json` of each model (links below) |
| Shared experts | DeepSeek-V3, Kimi K2/K3, GLM-4.5/5, Llama 4, Qwen3.5, Mistral Large 3, Nemotron 3; **not used by**: Qwen3-MoE, gpt-oss, Mixtral, MiniMax-M2, MiMo-V2-Flash | DeepSeekMoE arXiv:2401.06066; Qwen3 report, Section 2; the configurations |
| Auxiliary loss (Switch / GShard style) | Mixtral (`router_aux_loss_coef` 0.02), Qwen3 (global-batch load-balancing loss, 0.001), Llama 4 (0.001), OLMoE (0.01) | Switch Transformer arXiv:2101.03961; GShard arXiv:2006.16668; Qwen3 report; the configurations |
| **Auxiliary-loss-free bias balancing** | **Stated in the reports**: DeepSeek-V3 (Sections 2.1.2 and 4.2, γ = 0.001), GLM-4.5 (Sections 2.1 and 2.4: "loss-free balance routing", update rate 0.001), NVIDIA Nemotron 3 Nano (Section 2.4: "DeepSeek's aux-loss-free load balancing strategy", update rate 1e-3); **visible in the configurations**: Kimi K2 and K3 (`topk_method: noaux_tc`), GLM-5 (`noaux_tc`), DeepSeek-V3.2 (`noaux_tc`), MiniMax-M2 (`use_routing_bias: true`, `e_score_correction_bias`), Xiaomi MiMo-V2-Flash (`noaux_tc`) | DeepSeek-V3 arXiv:2412.19437; GLM-4.5 arXiv:2508.06471; Nemotron 3 Nano arXiv:2512.20848; Wang et al. arXiv:2408.15664; the configurations |
| No dropped tokens (dropless) | DeepSeek-V3 (report Section 2.1.2, "No Token-Dropping"); at the release of Mixtral, the Mixtral team sent vLLM an implementation with MegaBlocks kernels (Mixtral paper, Section 1) | arXiv:2412.19437; MegaBlocks arXiv:2211.15841 |

**Consensus decision (GOAL.md 2.1)**:

- **MoE (fine-grained experts)**: the flagship models of nine families (DeepSeek, Qwen, Kimi, GLM, gpt-oss, Llama, MiniMax, Mistral, Nemotron) are all MoE. This meets rule A, so the topic is in the main text.
- **Shared experts**: 7 families use them (DeepSeek, Kimi, GLM, Llama, Qwen3.5, Mistral, Nemotron). This meets rule A, so the topic is in the main text. But Qwen3-MoE, gpt-oss, MiniMax-M2, and others do not use them. Thus the main text describes a "spectrum" and does not call shared experts necessary.
- **Auxiliary-loss-free load balancing** (marked "to be verified" in the table in GOAL.md): three independent families state it in their technical reports: DeepSeek, GLM (Zhipu), and NVIDIA. The official configurations of three more families show the same mechanism: Kimi, MiniMax, and Xiaomi. (Kimi K2 reuses the DeepSeek-V3 architecture, so it is less independent, but Kimi K3 still uses the mechanism.) **This meets rule A, so this chapter puts it in the main text.** We also state honestly that it is not the only answer. Qwen, Llama, Mixtral, and OLMoE still use an auxiliary loss (Qwen3 uses the global-batch version). Also, all adopters of "auxiliary-loss-free" balancing still keep a very small sequence-level auxiliary loss.

**Model configurations** (read through Hugging Face in 2026-09):
[DeepSeek-V3](https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/config.json),
[DeepSeek-V3.2](https://huggingface.co/deepseek-ai/DeepSeek-V3.2/blob/main/config.json),
[Kimi-K2-Instruct](https://huggingface.co/moonshotai/Kimi-K2-Instruct/blob/main/config.json),
[Kimi-K3](https://huggingface.co/moonshotai/Kimi-K3/blob/main/config.json),
[Qwen3-235B-A22B](https://huggingface.co/Qwen/Qwen3-235B-A22B/blob/main/config.json),
[Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B/blob/main/config.json) (model card: 30.5B / 3.3B),
[Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B/blob/main/config.json),
[Qwen3.5-35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B/blob/main/config.json) (model card: "8 Routed + 1 Shared"),
[Qwen3.5-397B-A17B (read from the FP8 version)](https://huggingface.co/Qwen/Qwen3.5-397B-A17B-FP8/blob/main/config.json),
[GLM-4.5](https://huggingface.co/zai-org/GLM-4.5/blob/main/config.json),
[GLM-5](https://huggingface.co/zai-org/GLM-5/blob/main/config.json),
[gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b/blob/main/config.json),
[gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b/blob/main/config.json),
[Mixtral-8x7B-v0.1](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1/blob/main/config.json),
[Llama-4-Scout (unsloth mirror)](https://huggingface.co/unsloth/Llama-4-Scout-17B-16E-Instruct/blob/main/config.json),
[MiniMax-M2](https://huggingface.co/MiniMaxAI/MiniMax-M2/blob/main/config.json),
[MiMo-V2-Flash](https://huggingface.co/XiaomiMiMo/MiMo-V2-Flash/blob/main/config.json),
[NVIDIA-Nemotron-3-Nano-30B-A3B](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/blob/main/config.json),
[params.json of Mistral-Large-3](https://huggingface.co/mistralai/Mistral-Large-3-675B-Instruct-2512/blob/main/params.json),
[OLMoE-1B-7B](https://huggingface.co/allenai/OLMoE-1B-7B-0924/blob/main/config.json).

Notes: the official Meta repository of Llama 4 requires an access request. Thus this chapter reads the unsloth mirror (its `base_model` points to the official repository). For the shared expert of Llama 4, see `Llama4TextMoe.shared_expert` in HF transformers (the Meta blog was not accessible from this build environment). This chapter did not verify the total / active parameters of MiniMax-M2; it cites only its routing configuration. Our count of the total parameters of Kimi K2 from the configuration is about 14B lower than the report: **to be verified**.

---

## Guided questions

Ask Claude Code these questions. Continue until you can explain the answers in your own words:

1. DeepSeek-V3 uses the original scores s for the gate weights, but s + b to select the experts. What problem occurs if the gate weights also use s + b? (Hint: how does the bias then enter the gradient of the language model?)
2. With softmax scores and with sigmoid scores, what problem occurs in each case when the number of experts grows from 8 to 256? Why is sigmoid + normalization after the selection a better fit for fine-grained experts?
3. In the auxiliary loss, f_i is a count that has no gradient. Then how does the gradient make an overloaded expert get "fewer" tokens? Calculate ∂L/∂(logit) for the example by hand in Section 5.1, and look at the sign.
4. The table in Section 8 shows that at batch 64, Qwen3-30B-A3B reads almost every expert. Then what does an MoE save on a server? With a very large batch, compare it with a 3B dense model: what causes most of the difference in the time of each decode step?
5. Fine-grained experts become smaller and smaller. Why do we not split them until each expert has only one row (width 1)? Think about three things: the efficiency of matrix multiplication, the router parameters, and the communication.
6. In the small experiment of this chapter, compare the difference between the MoE and the two dense baselines with the variation between seeds. Which one is larger? For a serious comparison, how would you design the experiment (amount of data, steps, metrics)? Use Section 4 of the DeepSeekMoE paper as a reference.

## Hands-on tasks

For each task, run the code and look at the result.

**Task 1 (basic)**: On Hugging Face, select an MoE model that this chapter did not count (for example, the text part of Qwen3.5-122B-A10B, GLM-4.5-Air, or OLMoE-1B-7B). Read its `config.json` and add its fields to `MOE` in `01_param_ledger.py`. Calculate the total and active parameters, and compare them with the model card. Explain which counting method causes the difference.

**Task 2 (core)**: Add two variants to `03_train_compare.py`: (a) auxiliary loss with α = 0.1; (b) bias with γ = 0.001 (the DeepSeek-V3 value). Train each one once (seed 0 only is enough). Compare the "max/mean" load at the end of training and the validation loss. Does the loss become worse when α is too large? When γ is too small, are 800 steps enough to balance the load?

**Task 3 (challenge)**: Use `moe_transformer` from `zero/arch/moe.py` to build a small MoE with the same structure as the main-line model (the sizes of `configs/tiny`). Write a training loop (each step: `loss()` → `backward` → `step` → `after_step()`). Train it for a few hundred steps on `assets/tiny_corpus`, and print `load_stats()` every 50 steps. Then set `capacity_factor` to 1.0 and record how `last_dropped` changes during training. After the load is balanced, does the drop rate go to 0?

---

## Go deeper: CS336

This chapter matches Stanford CS336 (Spring 2026) <https://cs336.stanford.edu/>:

- **Lecture 4: alternatives to attention, and MoE.** The slides and the recording are on the course page. (This build environment could not access the course page. For the exact MoE details of this lecture, use the slides.) The routing, top-k, load balancing, and fine-grained / shared experts of this chapter are the core of the MoE part of this lecture. You can use `02_moe_layer.py` and `03_train_compare.py` as its smallest runnable version.
- **Not covered in depth by CS336**: two parts of this chapter. Section 2 checks the total / active parameters item by item against the official configurations. Section 8 gives the three ledgers of "why small models seldom use MoE". This course adds them for the decisions about the main-line model. For the engineering details of expert parallelism (all-to-all, node-limited routing, redundant experts), continue with CS336 Lectures 7 and 8 (parallelism) and Section 3 of the DeepSeek-V3 report.

---

## References

- Shazeer et al. *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer* (the MoE layer and the routing-collapse problem), 2017: <https://arxiv.org/abs/1701.06538>
- Lepikhin et al. *GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding* (top-2 routing, capacity, auxiliary loss, expert parallelism), 2020: <https://arxiv.org/abs/2006.16668>
- Fedus, Zoph, Shazeer. *Switch Transformers* (top-1 routing, load-balancing loss, capacity factor), 2021: <https://arxiv.org/abs/2101.03961>
- Gale et al. *MegaBlocks: Efficient Sparse Training with Mixture-of-Experts* (dropless, block-sparse matrix multiplication), 2022: <https://arxiv.org/abs/2211.15841>
- Jiang et al. *Mixtral of Experts*, 2024: <https://arxiv.org/abs/2401.04088>
- Dai et al. *DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models* (fine-grained experts, shared experts), 2024: <https://arxiv.org/abs/2401.06066>
- Wang et al. *Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts*, 2024: <https://arxiv.org/abs/2408.15664>
- DeepSeek-AI. *DeepSeek-V3 Technical Report* (Section 2.1.2: DeepSeekMoE and auxiliary-loss-free balancing; Sections 3.2 and 3.4: expert parallelism and deployment; Sections 4.5.2–4.5.3: ablations), 2024: <https://arxiv.org/abs/2412.19437>
- Qwen Team. *Qwen3 Technical Report* (Section 2: the MoE structure, no shared experts, global-batch load balancing; Table 5: MoE vs dense), 2025: <https://arxiv.org/abs/2505.09388>
- Kimi Team. *Kimi K2: Open Agentic Intelligence* (Section 2.3: architecture and the sparsity scaling law; Section 2.4: expert parallelism), 2025: <https://arxiv.org/abs/2507.20534>
- GLM-4.5 Team. *GLM-4.5: Agentic, Reasoning, and Coding (ARC) Foundation Models* (Sections 2.1 and 2.4), 2025: <https://arxiv.org/abs/2508.06471>
- OpenAI. *gpt-oss-120b & gpt-oss-20b Model Card* (Table 1: parameter breakdown; Section 2.2: MoE), 2025: <https://arxiv.org/abs/2508.10925>
- NVIDIA. *Nemotron 3 Nano: Open, Efficient Mixture-of-Experts Hybrid Mamba-Transformer Model for Agentic Reasoning* (Sections 2.1 and 2.4), 2025: <https://arxiv.org/abs/2512.20848>
- Muennighoff et al. *OLMoE: Open Mixture-of-Experts Language Models*, 2024: <https://arxiv.org/abs/2409.02060>
- The MoE implementations in Hugging Face transformers (`models/qwen3_moe`, `models/deepseek_v3`, `models/llama4`): <https://github.com/huggingface/transformers/tree/main/src/transformers/models>
- DeepSeek. DeepEP (expert-parallel communication library): <https://github.com/deepseek-ai/DeepEP>
- [CS336](https://cs336.stanford.edu/) Lecture 4 (alternatives to attention, and MoE)
- [The main LLM architectures in 15,000 characters (Llama, Qwen, GLM, DeepSeek…)](https://zhuanlan.zhihu.com/p/2060741715095560795) (in Chinese): a side-by-side comparison of the MoE structures of each family (already in `references.md`)
- [Marin 535B-A23B training livestream](https://wandb.ai/marin-community/marin_moe/reports/535B-A23B-18T-Token-Hero-Run-Scaling-Ladder--VmlldzoxNzc2MDM5Ng): the fully public training process of a large MoE, with ladder experiments (already in `references.md`)
- For the model-configuration links, see "Adopters and sources" above.

**Next chapter**: The configurations of DeepSeek-V3 and GLM-4.5 both contain `num_nextn_predict_layers: 1`. In training, these models predict not only the next token, but also one more token. Chapter 25 explains multi-token prediction (MTP). It also shows how MTP works together with speculative decoding to make generation almost twice as fast without a loss of quality.
