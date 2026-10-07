# Chapter 25: Multi-token prediction and speculative decoding — A small model guesses, a large model checks

**English** · [中文](README.zh.md)

> **Goal**: After this chapter, you can write the "guess–verify–roll back" loop of speculative decoding. You can verify that with greedy decoding, the output is token-for-token identical to the output of the target model. You can use `min(1, p/q)` and the residual distribution `max(0, p − q)` to derive that "with sampling, the distribution does not change at all". You can verify this with a chi-square test on a small vocabulary. You can estimate the speedup with `(1 − α^{k+1}) / ((1 − α)(1 + k·c))`. You can also explain the structure of the MTP module of DeepSeek-V3, and why it makes training better and can also be the draft at inference.

📺 **Video**: Not published yet. To render it on your computer, run `bash chapters/25-mtp-speculative-decoding/video/build.sh`.
🧪 **Self-check**: After the chapter, type `/ch25-speculative` in Claude Code.

---

In the last chapter, mixture of experts (MoE) gave the model many parameters, but each token uses only a small part of them. This saves compute for each token. But with a dense model and with MoE, generation still produces **one token at a time**. Chapter 10 showed that each decode step calculates only one new token, but it must read all weights and the KV cache from GPU memory. The hardware spends most of its time waiting for data.

This chapter asks: **can a large model produce several tokens in one forward pass, with no change at all in the quality of the output?** The answer is speculative decoding: a cheap draft first guesses a few tokens, and the large model checks all of them in one forward pass. Where does the draft come from? It can be a small model. It can also be a "multi-token prediction" (MTP) module that the large model learns during its own training. After DeepSeek-V3, the main models of Qwen, GLM, MiniMax, and Xiaomi MiMo all have such a module.

The code of this chapter:

```bash
uv run python chapters/25-mtp-speculative-decoding/code/01_models_and_cost.py    # target/draft models + "verifying k costs the same as generating 1"
uv run python chapters/25-mtp-speculative-decoding/code/02_greedy_speculative.py # greedy speculative decoding: identical output + acceptance rate + speed
uv run python chapters/25-mtp-speculative-decoding/code/03_speculative_sampling.py # rejection sampling keeps the distribution: chi-square test + real models
uv run python chapters/25-mtp-speculative-decoding/code/04_speedup_formula.py    # speedup formula and best k (calculation only, instant)
uv run python chapters/25-mtp-speculative-decoding/code/05_mtp.py                # MTP module: training + self-speculative decoding
```

The target model is the small model that Chapter 10 trained (4 layers, width 128, 65 characters at the character level, 0.86M parameters, validation loss 1.838). The draft model has the same structure and **the same vocabulary**, but only 1 layer and width 64 (57,600 parameters, about 1/15 of the target, validation loss 1.966). The first run trains the draft model and caches it in `code/out/`.

> **Note:** The numbers of the training experiments in this chapter come from one CPU run on the course build machine. Different machines and different versions of the math libraries do floating-point operations in a slightly different order. After some hundred training steps, these small differences become larger. Your numbers can be different from the second or third decimal place. Trust the conclusions below, which do not depend on exact values. For a second run on another server in 2026-10, see [runs/2026-10-01-gpu0-check/chapters-24-26.md](../../runs/2026-10-01-gpu0-check/chapters-24-26.md).

> **Note:** We measured all times in this chapter with one thread, on a 4-core machine that many jobs shared (the load was 16–33 during the measurements). To decrease the effect of "waiting in a queue for a CPU", the timing uses the **process CPU time** (`time.process_time()`), not the wall-clock time. But the times still change much. Use the times in the tables only to see the **order of magnitude and the trend**. The acceptance rates, the numbers of forward passes, and the distribution tests use fixed random seeds, so you can reproduce them.

## 1. The problem: decode waits for data, it does not calculate

Remember a number from Chapter 10: for the same small model, the throughput of prefill (256 tokens at a time) was 41 times the throughput of decode (1 token at a time). Chapter 21 calculated the reason. For each token that decode generates, it reads all weights and the full KV cache, but it does very few multiply-adds. The ratio "operations ÷ bytes read" (the arithmetic intensity) is only 1–5, but an H100 needs about 295 to use its full compute. **In the decode phase, the GPU waits for data.**

This leaves an opportunity. The time to read the weights once is fixed. Thus **feeding a few more tokens at a time costs almost no extra time**. The bytes read do not change, and the small extra compute uses units that are idle anyway. `01_models_and_cost.py` measures this on a CPU. The KV cache already has 200 positions. How long does one forward pass of the target model take for T new tokens?

| Tokens fed at a time (T) | 1 | 2 | 3 | 5 | 9 | 17 |
|---|---|---|---|---|---|---|
| Target model time (ms) | 3.55 | 4.13 | 4.44 | 4.60 | 4.85 | 6.08 |
| Relative to T = 1 | 1.00× | 1.16× | 1.25× | 1.30× | 1.36× | 1.71× |
| Draft model time (ms) | 0.88 | 0.88 | 1.05 | 0.92 | 1.07 | 1.06 |

The compute increased 17 times, but the time increased only 71%. (On this very small model and a single-thread CPU, most of the time goes to the fixed overhead of Python and of each PyTorch operator, not to memory bandwidth. But the conclusion has the same shape: **checking a piece of text is much cheaper than writing it**.)

The problem is that we do not know the next few tokens, so we cannot "feed several at a time". We can do it only if something guesses them first.

## 2. Speculative decoding: the draft guesses, the target checks once

Speculative decoding comes from Leviathan et al. 2023. At the same time, Chen et al. 2023 at DeepMind proposed it with the name "speculative sampling". It borrows the idea of "speculative execution" from CPUs: continue on a guess, and undo the work if the guess is wrong. Each round has four steps:

1. **The draft guesses k tokens**: a cheap model generates k tokens `d₁ … d_k` autoregressively.
2. **The target checks once**: feed "the last confirmed token + k drafts" to the target model at once. One forward pass gives the predictions at k+1 positions.
3. **Compare from left to right**: if `d_i` agrees with the answer of the target at that position, accept it. At the first disagreement, stop and use the answer of the target instead (**correction**). If all k drafts are correct, the answer of the target at the last position is one more token, free (**bonus**).
4. **Roll back**: the rejected drafts are already in the KV caches of both models. Discard them.

In `02_greedy_speculative.py`, the core of the greedy version is these lines:

```python
feed = seq[len(tc) :] + drafts  # Confirmed tokens not in the cache yet + k drafts
p_logits = target(torch.tensor([feed]), tc)[0, -(k + 1) :]  # One forward pass, the last k+1 positions
choice = p_logits.argmax(-1).tolist()  # The answer of the target at each position
m = 0
while m < k and drafts[m] == choice[m]:  # Accept from left to right
    m += 1
seq += drafts[:m] + [choice[m]]  # m drafts + 1 correction (or bonus)
truncate(tc, len(seq) - 1)  # Roll back: discard the K/V of the rejected drafts
```

Here is a real example. Video S03 uses it. (After the target model is retrained on another machine, the video can select a different round when it renders.) The prompt is a piece of Shakespeare from the validation set. The draft guessed 4 characters. The target model accepted the first two, did not agree with the third, and used its own answer instead. This round ran the target model only once and produced 3 characters.

**Each round produces at least 1 token** (in the worst case, the first draft is rejected and we get the answer of the target, the same as normal decoding) **and at most k+1 tokens**. The target model runs only once per round. Thus the number of forward passes of the target can only decrease, never increase. The only extra cost is the time of the draft.

## 3. Greedy: why the output is token-for-token identical

First, look at the simplest case: the target model uses greedy decoding (argmax at each step). Each output token of speculative decoding is either "a draft that agrees with the argmax of the target" or "the argmax of the target". **All of them are the token that the target model selects with the same prefix.** Thus the output is **token-for-token identical** to the output of greedy decoding of the target model, step by step. The draft decides only "how many steps we move forward at once", not "what we write".

This statement must be verified with code, because it is easy to get wrong. The most common error is a KV cache that is not fully rolled back. The K/V of the rejected drafts stay in the cache, and the attention later "sees" a history that never occurred. `02` does a parity check with 4 prompts, 200 characters each, and k = 1, 3, 5, 8:

```
k=1: 4 prompts × 200 tokens, identical to greedy decoding of the target model: True
k=3: 4 prompts × 200 tokens, identical to greedy decoding of the target model: True
k=5: 4 prompts × 200 tokens, identical to greedy decoding of the target model: True
k=8: 4 prompts × 200 tokens, identical to greedy decoding of the target model: True
```

The method of rollback depends on the cache implementation. The minimal cache of Chapter 10 concatenates with `torch.cat`, so rollback is a slice: `cache.k[layer] = cache.k[layer][:, :, :n]`. The cache of `zero` is preallocated (Chapter 10). `KVCache.update(layer, start_pos, k, v)` overwrites by position and returns only `[0, start_pos + T)`. Thus rollback does not even move data: set the "valid length" back, and the next write starts from that position.

> **Note:** Strictly, "token-for-token identical" also requires identical floating-point calculations. The target model can get k+1 tokens at once or 1 token at a time. In these two cases, the matrix multiplication can use a different block order, and the rounding of the last bit can be different. If the logits of two candidates are almost equal, the argmax can flip. We did not see this on the FP32 small model of this chapter. The vLLM documentation also states that its implementation is "lossless in the algorithm", but that numerical differences between batch sizes on a GPU can make the outputs not fully identical.

## 4. Sampling: rejection sampling keeps the distribution exactly

Greedy decoding is easy. What about sampling? At some position, the target model gives a distribution `p`, and the draft gives a different distribution `q`. The draft samples a token `x` from `q`. We want the final token to follow `p`, not `q`.

There are only two rules:

1. **Accept** `x` with the probability `min(1, p(x) / q(x))`.
2. If `x` is rejected, sample a new token from the **residual distribution** `p'(x) = max(0, p(x) − q(x)) / Σ max(0, p − q)`, and end this round.

Intuition: for a token with `p(x) ≥ q(x)`, the draft gives "not enough" probability, so we always accept it. For a token with `p(x) < q(x)`, the draft gives "too much", so we remove the extra part in proportion. Where does the removed probability mass go? It goes to the tokens for which the draft gives not enough: the residual distribution has mass only where `p > q`.

**Derivation**: the probability of a final token `x` = "the draft samples x and x is accepted" + "after a rejection, the residual gives x":

```
P(draft samples x and accepts) = q(x) · min(1, p(x)/q(x)) = min(p(x), q(x))
P(rejection)                   = 1 − Σ_x min(p(x), q(x)) = 1 − α
P(x after rejection)           = (1 − α) · max(0, p(x) − q(x)) / (1 − α)  = p(x) − min(p(x), q(x))
Sum of the two terms           = p(x)                                                  ✓
```

(The second-to-last line uses `Σ max(0, p − q) = Σ (p − min(p, q)) = 1 − α`.) We also get an important quantity: **the probability of one acceptance** `α = Σ_x min(p(x), q(x)) = 1 − TV(p, q)`. TV is the total variation distance between the two distributions. The more similar the draft and the target are, the higher α is.

In `03_speculative_sampling.py`, the rule is:

```python
def accept_or_resample(p, q, x, g):
    if torch.rand((), generator=g) < torch.clamp(p[x] / q[x], max=1.0):  # Accept with min(1, p/q)
        return True, x
    residual = torch.clamp(p - q, min=0)  # Residual max(0, p − q)
    return False, int(torch.multinomial(residual / residual.sum(), 1, generator=g))
```

**Experiment 1: one step, 6 tokens, 200,000 samples.** Select a random pair p, q. The draft samples from q, and we accept or resample with the rules above:

| token | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| Target p | 0.123 | 0.310 | 0.055 | 0.230 | 0.197 | 0.085 |
| Draft q | 0.125 | 0.013 | 0.296 | 0.143 | 0.382 | 0.041 |
| Residual max(0, p − q) (before normalization) | 0 | 0.297 | 0 | 0.088 | 0 | 0.044 |
| **Result of speculative sampling** | 0.123 | 0.310 | 0.055 | 0.231 | 0.198 | 0.084 |
| Samples of the draft, used directly | 0.125 | 0.013 | 0.296 | 0.143 | 0.382 | 0.041 |

The TV distance between the result of speculative sampling and p is 0.0013. The chi-square statistic is 4.6 (5 degrees of freedom), and the p-value is 0.46: we cannot see any difference from p. For the samples of the draft used directly, the TV distance is 0.43 and the chi-square p-value is 0. The measured acceptance rate is 0.5722, and the formula Σ min(p, q) gives 0.5714.

**Experiment 2: the full algorithm.** The experiment above verifies only one position. The full speculative decoding also has two more parts: "a later draft is checked only if the earlier drafts were accepted", and "the bonus token when all drafts are accepted". `03` uses a pair of Markov chains as "language models": the next token depends only on the previous token, and the target and the draft are two random transition matrices. With k = 2, it generates sequences of length 3, 60,000 times, and compares them with the exact joint distribution of the target model (4³ = 64 cells).

The TV distance is 0.0083, and the chi-square statistic is 50.5 (49 degrees of freedom, after merging the cells with an expected count below 5), with a p-value of 0.41. Each round produced 1.50 tokens on average. The full algorithm also does not change the distribution.

**Experiment 3: real models, temperature 1.0.** Use the same rules with the target model of Chapter 10 and the 1-layer draft (4 prompts × 200 characters, part 3 of `03`):

| k | Measured acceptance rate | Mean of the formula Σ min(p, q) | Output per round | Speedup (CPU time) |
|---|---|---|---|---|
| 1 | 0.674 | 0.690 | 1.67 | 1.11× |
| 3 | 0.680 | 0.697 | 2.44 | 1.03× |
| 5 | 0.747 | 0.714 | 3.19 | 0.94× |

The measured acceptance rate agrees with "Σ min(p, q) at each position, then the mean". (Each k has only some hundred comparisons, so a difference of 0.02–0.03 is within the sampling error.) The acceptance rate with sampling is a little lower than with greedy decoding (about 0.70–0.72, see Section 5). At temperature 1, the distributions are flatter, and the two models differ more.

The speedup on the CPU is small, and k = 5 is even slower. The reasons are the same as in the greedy experiment of Section 5 (c ≈ 0.2, verification is not free, Python overhead). Also, each position needs an extra softmax and a random sample. In two runs, the CPU time of normal sampling was 2.85 seconds and 2.22 seconds. The speedup changed with it, between 0.9× and 1.1× (the previous run gave 1.13×, 1.12×, and 1.05×). **Look only at the trend of the speed here.**

Two practical details:

- **Apply the same temperature and top-p to both sides.** `p` and `q` are the distributions "after the same temperature and top-p processing". Speculative decoding makes sure that the output follows "the processed target distribution". This is the distribution that you get when you sample with the same settings and without speculative decoding. `warp_probs` in `zero/arch/speculative.py` does this.
- **A looser acceptance rule loses exactness.** To get a higher acceptance rate, some implementations make the rule looser. Examples are the lenience parameter in the appendix of the paper of Leviathan et al., and the typical acceptance of Medusa. They are faster, but the output distribution is no longer exactly the target distribution. This chapter explains only the strict version.

## 5. How much faster: α, k, and c

Assume that each draft token is accepted with the same probability α, independently. In one round, the 1st output token always occurs. The 2nd token needs the acceptance of the 1st draft (probability α). The 3rd token needs the acceptance of the first two drafts (α²), and so on. Thus:

```
Expected output per round E = 1 + α + α² + … + α^k = (1 − α^{k+1}) / (1 − α)
```

The cost of each round is 1 forward pass of the target plus k forward passes of the draft. Let `c` = draft step ÷ target step. **Assume that the target verifies k+1 positions as fast as it generates 1.** Then the speedup is (Leviathan et al. 2023, Theorem 3.8):

```
speedup = (1 − α^{k+1}) / ((1 − α)(1 + k·c))
```

`04_speedup_formula.py` calculates it as a table (here is a part of the table for c = 0.05):

| α \ k | 1 | 2 | 3 | 4 | 6 | 8 | Best k |
|---|---|---|---|---|---|---|---|
| 0.5 | 1.43 | 1.59 | 1.63 | 1.61 | 1.53 | 1.43 | 3 (1.63×) |
| 0.7 | 1.62 | 1.99 | 2.20 | 2.31 | 2.35 | 2.28 | 6 (2.35×) |
| 0.9 | 1.81 | 2.46 | 2.99 | 3.41 | 4.01 | 4.38 | 13 (4.67×) |

Some rules: the higher α is and the smaller c is, the more guesses are useful. When α is low, more guesses only waste the compute of the draft (at α = 0.5, a k above 3 is slower). When c = 0, the upper limit of the speedup is `1/(1 − α)`. The formula agrees with the paper: α = 0.8, k = 5, c = 0 gives 3.69×, which is the number in Table 1 of Leviathan et al.

**Measurement (greedy, 4 × 200 characters, `02_greedy_speculative.py`)**: normal greedy decoding of 800 characters used 2.73 seconds of CPU time. The cost coefficient is c ≈ 0.20.

| k | Per-token acceptance rate α | Output per round (measured) | Formula (1) | Target forward passes (normal decoding: 800) | CPU time (s) | Speedup | Formula prediction |
|---|---|---|---|---|---|---|---|
| 1 | 0.721 | 1.72 | 1.72 | 466 | 2.35 | 1.17× | 1.43× |
| 2 | 0.697 | 2.19 | 2.18 | 368 | 2.24 | 1.22× | 1.56× |
| 3 | 0.722 | 2.64 | 2.62 | 306 | 2.13 | 1.28× | 1.63× |
| 4 | 0.704 | 2.83 | 2.79 | 286 | 2.28 | 1.20× | 1.55× |
| 5 | 0.700 | 2.95 | 2.94 | 275 | 2.47 | 1.10× | 1.47× |
| 6 | 0.707 | 3.10 | 3.12 | 262 | 2.51 | 1.09× | 1.41× |
| 8 | 0.703 | 3.32 | 3.22 | 244 | 2.76 | 0.99× | 1.23× |

("Per-token acceptance rate" = number accepted ÷ number of drafts that were compared. The drafts after the first rejection are not compared, so we do not count them.)

Three things in this table are important:

1. **Formula (1) agrees almost exactly**: the measured output per round differs from `(1 − α^{k+1})/(1 − α)` by less than 0.1. The acceptance rate is 0.70–0.72 for all values of k. Thus "each draft is accepted independently with α" is a good approximation here.
2. **The number of forward passes of the target really decreases**: at k = 3, it goes from 800 to 306. This number does not depend on the hardware.
3. **On the CPU, the speedup is much smaller than the decrease in forward passes**: the fastest, k = 3, is only 1.28× faster. This is even lower than the 1.63× of the formula. All the reasons are in the assumptions of the formula. First, c is not small: one step of the 1-layer draft costs about one fifth of a step of the target. On such a small model, most of the time is the fixed overhead of each operator, and fewer layers and a smaller width do not save much of it.

   Second, verification is not completely free: in the table of Section 1, feeding 4 tokens at once is about 1.3× slower. If we put this into the formula (`2.64 / (1.28 + 3 × 0.2) ≈ 1.40`), we get close to the measurement. The rest of the gap is the overhead of the Python loop, the rollback, and the extra tokens that the draft must feed. With a larger k, the overhead of the draft adds up. At k = 8, speculative decoding is already slower than no speculative decoding.

Thus: **this small CPU experiment can verify the correctness, the acceptance rate, and the number of forward passes, but its speedup does not show the situation on a GPU**.

On real GPU systems, the draft is usually two orders of magnitude smaller than the target, and verifying k+1 tokens is almost really free. c can go below 0.05, but only if the overhead of launching the kernels of each step also becomes small (CUDA Graph, fused kernels, and so on). In the "GPU measurements" section of this chapter, the draft has only 1/322 of the parameters of the target. But when PyTorch launches the operators one by one, c is still 0.078. Leviathan et al. used T5-small (77M) as the draft for T5-XXL (11B) on TPUs. α was 0.62–0.75, and the measured speedup was 2.3–3.4×.

**When speculative decoding is not worth it**: first, when α is too low (the draft is too different from the target, or the sampling temperature is high and the distribution is flat). Second, with **large batches and high concurrency**. A server puts tens of requests into one decode batch, so the arithmetic intensity is already high. "Verifying a few more positions is almost free" is then not true, and the extra verification compute takes compute from other requests. The vLLM documentation states that speculative decoding is mainly for "low-to-medium QPS, memory-bound" workloads. vLLM also has an option that changes the guess length dynamically with the load.

## 6. Where the draft comes from

Speculative decoding has **no correctness requirement on the draft**: even with a very bad draft, the output distribution does not change, and only the speed decreases. Thus the draft can be of many kinds. There is only one hard constraint: **the draft must use the same tokenizer as the target model** (the target must score the token ids from the draft directly). There are three common kinds:

1. **A small model of the same family.** It has the same tokenizer and the same training data, so it "thinks" in a similar way. Leviathan et al. found that α and c have the best balance when the draft is about two orders of magnitude smaller than the target. The inference lecture of CS336 gives the pairs 70B with 8B and 8B with 1B, and recommends distillation to make the draft more similar to the target. Google released dedicated MTP draft models for Gemma 4 E2B, E4B, 12B, 26B-A4B, and 31B (`gemma-4-*-it-assistant`; the draft of 31B has only 4 layers and shares the KV cache with the target model). The model card says "up to about 3× faster, with output quality identical to standard generation".
2. **Prompt lookup (n-gram).** In the existing text, find a piece that matches the last few tokens, and use the tokens after that piece as the draft. The cost is almost zero (c ≈ 0). α is high for tasks that "copy much of the context": rewriting code, summarizing documents, and parameter names that repeat in tool calls. Leviathan et al. even found that a small bigram table makes translation with T5-XXL 1.25× faster. `zero/arch/speculative.py` uses prompt lookup when `draft=None`.
3. **A draft head built into the model.** Do not train and deploy a separate model. Instead, grow a small part on the target model that "guesses the next few tokens". This part can read the last-layer representation of the target model directly. It has much more information than an independent small model, so its guesses are accurate. The most common method of this kind is MTP, in the next section. For Medusa, EAGLE, and other methods, see "Frontier notes" at the end of the chapter.

## 7. MTP: learn one more step in training, get a built-in draft at inference

### 7.1 From "several heads" to "sequential modules"

The first form of multi-token prediction (Gloeckle et al. 2024, Meta) is direct: after a shared trunk, add n independent output heads. Head j predicts the token j positions ahead, and the losses are added. DeepSeek-V3 made a key change: **predict in sequence and keep the full causal chain**. At position i, the MTP module of depth k does this:

```
h'ᵏᵢ = Mₖ [RMSNorm(hᵏ⁻¹ᵢ) ; RMSNorm(Emb(t_{i+k}))]       # representation from depth k−1 + embedding of token i+k
hᵏ   = TRMₖ(h'ᵏ)                                          # one Transformer block (causal attention)
Pᵏ_{i+k+1} = OutHead(hᵏᵢ)                                 # output head shared with the main model
L_MTP = λ/D · Σₖ CrossEntropy(Pᵏ, t)                      # added to the main loss
```

Key points:

- **The embedding and the output head are shared with the main model.** The MTP module itself has only two RMSNorms, one `2d → d` projection `Mₖ`, and one Transformer block. On the small model of this chapter, the MTP module has 246,400 parameters, 29% of the main model (861,440). The reason is that the main model has only 4 layers, so one more block adds a quarter. DeepSeek-V3 has 61 layers, so one MTP module is a much smaller fraction.
- The input of module k contains the embedding of `t_{i+k}`. Thus when the module predicts `t_{i+2}`, it **knows** `t_{i+1}` (the real token in training, and the token that the main model just selected at inference). So it does not "guess two steps ahead blindly". It **takes one more step** from the representation of the main model. This is the reason why it is a very accurate draft.
- DeepSeek-V3 uses D = 1 (it predicts only one more token). λ is 0.3 for the first 10T tokens and 0.1 after that. GLM-4.5 uses the same λ setting (0.3 for the first 15T tokens, 0.1 after that).

### 7.2 Training: a denser signal

The first purpose of MTP is to **make the main model train better**. At each position, the model must give a useful representation for "the next token" and also for "the token after next". The training signal becomes denser (the DeepSeek paper: "densifies the training signals"), and the model can also learn to "plan ahead". The ablation of DeepSeek-V3 (Table 4) trains a pair of models at each of two MoE scales, 15.7B and 228.7B. The only difference in each pair is 1 MTP layer or none. With MTP, most benchmarks improve (for example, for the small model, HumanEval 20.7 → 26.8 and GSM8K 25.4 → 31.4). At inference, the MTP module is discarded, so the cost is exactly the same.

`05_mtp.py` does the same on the small model of Chapter 10: the same initialization, the same data order, the same 600 steps, and only one addition, `0.3 × L_MTP`. The two cached models of Chapter 10 (seeds 0 and 1) are the control group with λ = 0:

| Model | Validation loss | Next-character accuracy | MTP accuracy (character after next) |
|---|---|---|---|
| Seed 0, λ = 0 (Chapter 10) | 1.838 | 0.454 | — |
| Seed 0, λ = 0.3 (with MTP) | 1.821 | 0.456 | 0.464 |
| Seed 1, λ = 0 (Chapter 10) | 1.869 | 0.445 | — |
| Seed 1, λ = 0.3 (with MTP) | 1.857 | 0.449 | 0.458 |

For both seeds, the main model with MTP has a slightly lower validation loss (by 0.017 and 0.012). But Chapter 10 found that **a different random seed alone changes the loss of the same model by 0.031**, which is more than the difference here. Thus this table does **not** show that "MTP makes the main model better". It shows only that we saw no harm at this scale. The conclusion of DeepSeek comes from controlled experiments with hundreds of billions of tokens and from over a billion to hundreds of billions of parameters. A few-minute small experiment like this one cannot reproduce it.

The last column is more interesting. The accuracy of the MTP module for the **character after next** (0.464, 0.458) is about the same as the accuracy of the main model for the **next** character (0.456, 0.449). It is even a little higher. The reason is in the structure. When the module predicts t_{i+2}, it already has the embedding of t_{i+1} and the representation of the main model at position i. It knows one more character than the main model knows when it predicts t_{i+1}.

The core of the MTP module in `05`:

```python
x = self.proj(
    torch.cat([self.enorm(emb_next), self.hnorm(h)], dim=-1)
)  # [RMSNorm(Emb(t_{i+1})); RMSNorm(h_i)] → projection
return self.norm(self.block(x, cos, sin, cache, 0))  # One block + RMSNorm, then multiply by the shared Embᵀ
```

In training, it is one line: `loss = l_main + lam * l_mtp`. The target of `l_mtp` is `y[:, 1:]`. (y is the target sequence shifted left by one position. Shift it left by one more position to get "the token after next".)

### 7.3 Inference: discard it, or use it as a draft

The second use of MTP is **self-speculative decoding**. The main model gives the next token `t_{i+1}`. At the same time, the MTP module uses the representation of the main model and the embedding of `t_{i+1}` to guess `t_{i+2}`. The next forward pass feeds `[t_{i+1}, draft]` together. It verifies the draft and also gets the token after that. With k = 1, each forward pass produces `1 + α` tokens on average.

DeepSeek-V3 reports an acceptance rate of 85%–90% for the second token, and a decoding speed (TPS) of about 1.8×. Put these values into the formula: 1 + α = 1.85–1.90, and after the cost of the MTP module itself, 1.8× is reasonable.

The self-speculation experiment in `05` uses the model of seed 0, greedy decoding, and 4 × 200 characters. The output is **token-for-token identical** to the output of greedy decoding of the main model. The draft acceptance rate is 0.638. The forward passes of the main model decrease from 800 to 486, and each forward pass produces 1.65 characters (= 1 + 0.65, in agreement with `1 + α`).

But the CPU time almost does not change (normal greedy 2.15 seconds, self-speculation 2.22 seconds, 0.97×). The MTP module has a full block. On the 4-layer small model, one step of the module costs more than about a quarter of a step of the main model, and the Python overhead adds more. This cancels the saved forward passes. On the 61-layer DeepSeek-V3, one MTP module is much cheaper relative to the main model, so it gets about 1.8×.

A small implementation detail: the Transformer block in the MTP module also looks at the history, so it has its own KV cache. At position i, it needs "the representation of the main model at i" and "token i+1". Thus it is always one step behind the main model and processes only confirmed positions. **It does not need a rollback.**

### 7.4 Who uses MTP

We checked with rule A of GOAL.md 2.1 (details in "Adopters and sources"):

- DeepSeek: `num_nextn_predict_layers: 1` in V3 and V4.
- Qwen: the Qwen3-Next model card states MTP. The configs of all Qwen3.5 models have `mtp_num_hidden_layers: 1`, even the 0.8B model.
- Zhipu GLM: the GLM-4.5 technical report says "add one MoE layer as the MTP layer, for speculative decoding at inference". The GLM-5 config also has 1 layer.
- MiniMax: the M2 config has `use_mtp: true` and `num_mtp_modules: 3`.
- Xiaomi MiMo: `num_nextn_predict_layers: 1` in MiMo-7B. MiMo-V2-Flash released 3 layers of MTP weights, and its model card says "3× output speed".
- NVIDIA: `num_nextn_predict_layers: 1` in Nemotron 3 Super.

This is far more than 3 families, so MTP satisfies the consensus condition and goes into the main text. There is also a counterexample: the config of Kimi K2 has `num_nextn_predict_layers: 0` (it uses the DeepSeek-V3 structure, but without an MTP layer). Speculative decoding itself is an industry standard in inference engines: vLLM and SGLang both have built-in draft models, n-gram, MTP, EAGLE, and other draft methods.

## 8. Summary

- **Condition**: decode is limited by memory bandwidth. The target model verifies k+1 positions at once in about the same time as it generates 1.
- **Algorithm**: the draft guesses k tokens → one forward pass of the target → accept from left to right, correct at the first error, get a bonus if all are correct → roll back the KV cache. Each round gives 1 to k+1 tokens.
- **Same quality**: with greedy decoding, the output is token-for-token identical. With sampling, accept with `min(1, p/q)` and, after a rejection, resample from `max(0, p − q)`. The output then follows the target distribution exactly. The probability of one acceptance is `α = Σ min(p, q) = 1 − TV(p, q)`.
- **How much faster**: `(1 − α^{k+1}) / ((1 − α)(1 + k·c))`. It is fast only with a high α and a small c. With large batches and high concurrency, the benefit is smaller.
- **Drafts**: a small model with the same tokenizer, prompt lookup, or a draft head built into the model.
- **MTP**: sequential MTP modules (two RMSNorms + projection + one block, with a shared embedding and output head). In training, an extra λ·L_MTP makes the signal denser. At inference, discard the module or use it as a draft (DeepSeek-V3: acceptance rate 85%–90%, about 1.8×).

---

## GPU measurements (one RTX 3090)

> **Note:** All numbers in the main text above come from CPU runs. In this section, we measure on one NVIDIA GeForce RTX 3090 (24 GB of GPU memory, Ampere architecture). Its spec sheet gives a dense BF16 tensor-core peak of about 71 TFLOPS, FP32 of about 35.6 TFLOPS, and a memory bandwidth of about 936 GB/s. Environment: PyTorch 2.11.0+cu128, CUDA 12.8, October 2026. The server sets the power limit of this card to 240 W (the factory default is 350 W). Under a continuous full load, the card lowers its clock. Thus the absolute compute and bandwidth are lower than on a 3090 at full power, and the relative values are more reliable. If you do not have a GPU, skip this section.

Run:

```bash
uv run python chapters/25-mtp-speculative-decoding/code/06_gpu_speculative.py
```

The script uses three models. The first is the small target model of this chapter (0.86M parameters, FP32). The second is the "scaled-up target": the same TinyLM structure, made larger (width 8192, 6 layers, 4.15B parameters, 8.31 GB in BF16). The third is the "scaled-up draft" (width 1024, 1 layer, 12.9M parameters, 1/322 of the target). The scaled-up models have random initialization, and we use them only to measure time: the time of a forward pass does not depend on the values of the weights.

**The condition of Section 1: how long does one forward pass take for T tokens?** (KV cache with 200 positions, CUDA event timing, median of 30.)

| Tokens fed at a time (T) | 1 | 2 | 4 | 8 | 16 | 64 | 256 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Target of this chapter (0.86M, FP32), ms | 3.838 | 3.864 | 3.930 | 3.453 | 3.984 | 3.929 | 3.835 |
| Scaled-up target (4.15B, BF16), ms | 13.385 | 13.876 | 13.815 | 14.180 | 14.597 | 20.189 | 46.902 |
| Scaled-up, relative to T = 1 | 1.00× | 1.04× | 1.03× | 1.06× | 1.09× | 1.51× | 3.50× |

With T = 1, the scaled-up model read its 8.31 GB of weights in 13.39 ms: an effective bandwidth of 621 GB/s. The 3.4 MB of weights of the small target of this chapter give only 0.9 GB/s. One forward pass of the small target on the GPU takes 3.84 ms, which is not faster than the single-thread CPU of Section 1. All the time goes to Python, which launches some hundred small kernels one by one (about 6.5 µs each on this machine).

**Wall-clock speedup of speculative decoding.** First, we move the trained target + draft of this chapter to the GPU and run `speculative_greedy` from `02` (FP32, 4 × 200 characters). For k = 1, 2, 3, 4, 6, and 8, the output is **token-for-token identical** to the output of greedy decoding of the target model. The per-token acceptance rate is 0.702–0.723. (These weights were retrained on another machine. The validation loss of the target model is 1.831, not 1.838, so the acceptance rates differ from the table of Section 5 by 0.01–0.02.)

Then we change to the scaled-up target + scaled-up draft. Between random weights, "guessing well or badly" has no meaning. Thus **the number accepted in each round is copied from the real result of the small models in that round**. All other steps occur again with no step missing: the draft feeds the missing tokens and guesses k tokens, the target verifies once and takes the argmax, and both caches roll back. (To limit the time, we replay only the first 2 prompts. The wall-clock time is the median of 3 runs.) Normal greedy decoding of 400 tokens takes 5.68 s, and the cost coefficient measured on the GPU is c ≈ 0.078:

| k | 1 | 2 | 3 | 4 | 6 | 8 |
|---|---:|---:|---:|---:|---:|---:|
| Per-token acceptance rate α (first 2 prompts) | 0.724 | 0.699 | 0.731 | 0.721 | 0.714 | 0.707 |
| Output per round | 1.72 | 2.18 | 2.67 | 2.97 | 3.27 | 3.35 |
| Wall-clock time (s) | 3.53 | 2.99 | 2.61 | 2.53 | 2.60 | 2.83 |
| **Measured speedup** | **1.61×** | **1.90×** | **2.18×** | **2.25×** | **2.18×** | **2.01×** |
| Formula `(1 − α^{k+1}) / ((1 − α)(1 + k·c))` | 1.60× | 1.89× | 2.15× | 2.20× | 2.15× | 2.01× |

The condition of Section 1 is really true only on the GPU. The 4.15B model with 16 tokens at a time is only 9% slower than with 1 token, because the time goes to moving 8 GB of weights from GPU memory. Only at 64 tokens does it become clearly slower, and at 256 tokens, compute becomes the bottleneck (3.5×). Verification is almost free and c is small, so the formula of Section 5 agrees almost exactly: the measurement and the prediction differ by less than 3%.

At k = 4, speculative decoding is 2.25× faster, the same order of magnitude as the 2.3–3.4× that Leviathan et al. reported on TPUs. The CPU experiment of Section 5 was at most 1.28× faster, about 20% below the formula. The difference comes exactly from the two assumptions "verification is free" and "c is small".

The value of c was a surprise. The draft has only 1/322 of the parameters of the target, but one step takes 1/13 of the time of a step of the target. The reason is that the time of a 1-layer small model is almost all fixed kernel-launch overhead, and this overhead does not depend on the number of parameters. When Python launches the operators one by one, as in this implementation, the number of layers and operators decides c. Inference engines use CUDA Graph and fused kernels to decrease this overhead. Only then can c go lower.

## From minimal code to production code

The production implementation is in `zero/arch/speculative.py` and `zero/arch/mtp.py`. These are experiment modules of Part 5, and the course **does not use them to train the main-line model**.

| Minimal code (`code/`) | Production code (`zero/arch/`) | What it adds, and why |
|---|---|---|
| `speculative_greedy` in `02`, `speculative_sample` in `03`, `TinyLM` of Chapter 10 + KV cache that concatenates | `speculative_generate(target, draft, prompt_ids, max_new_tokens, k, temperature, top_p, seed, eos_id)` in `speculative.py` → `SpeculativeResult` | Uses `zero.model.Transformer` and the preallocated `zero.kv_cache.KVCache`: rollback needs only `rollback(cache, n)`, which sets the valid length back (`KVCache.update` overwrites at `start_pos` and returns only `[0, start_pos + T)`); one code path for greedy decoding and sampling; early stop at `eos_id`; returns the number accepted, the number compared, and the number accepted in each round, with `acceptance_rate` and `tokens_per_round` ready to read; checks that the two models have the same vocabulary |
| `accept_or_resample` and `speculative_step` in `03` | `verify(p_logits, drafts, q_probs, temperature, top_p, generator)` | Supports **top-p**: `warp_probs` applies the same temperature and top-p to p and q (the same distribution as `zero.generate.sample_next`); the draft distribution can be None (a deterministic draft, handled as one-hot, so the acceptance probability is p(x)); a fallback when the residual is numerically 0 |
| None | `prompt_lookup_draft(seq, k, max_ngram)`; `speculative_generate(..., draft=None)` | A zero-cost n-gram draft (the same kind as the `ngram` method of vLLM) |
| The formula in `04` | `expected_tokens_per_round(α, k)`, `expected_speedup(α, k, c)` | The same formula, for the evaluation scripts |
| `MTPHead` in `05` | `MTPModule` in `mtp.py` (`enorm`, `hnorm`, `eh_proj`, `block`, `norm`) | The names agree with the DeepSeek MTP implementations of vLLM / SGLang (the concatenation order is `[enorm(emb); hnorm(h)]`, and the input to MTP is the main-model representation after its last RMSNorm); the block is the main-line `zero.model.Block` (GQA + QK-Norm + RoPE + SwiGLU) |
| The loss line in `train_mtp` of `05` | `MTPTransformer(config, n_mtp)` (the main model is `zero.model.Transformer` without changes) + `mtp_loss(model, tokens, targets, lam)` | Supports D sequential modules (depth k calculates on length T−k) and `ignore_index` (so the SFT loss mask also works); returns the total loss, the main loss, and the loss of each depth, for the logs |
| `mtp_self_speculative` in `05` (greedy) | `mtp_speculative_generate(model, prompt_ids, max_new_tokens, temperature, top_p, seed, eos_id)` | Greedy decoding and sampling; the MTP block has its own preallocated KV cache and processes only confirmed positions, so it needs no rollback; reuses `verify` |

**Parity check** (`uv run pytest tests/test_arch_speculative.py tests/test_arch_mtp.py`; all 17 tests passed on this machine, in about 20 seconds):

- `test_arch_speculative.py`: at k = 1, 3, 6, greedy speculative decoding is **token-for-token identical** to `zero.generate.generate(target, ..., temperature=0)`. The prompt-lookup draft is also identical. When **draft = target** (greedy and sampling), the acceptance rate is 100%, and each round gives exactly k drafts + 1 bonus. **Cache rollback**: write 4 "wrong drafts" into the cache, roll back, and write the correct tokens; the logits agree with one full forward pass within 1e-5. `verify` at one position with 40,000 samples: the TV distance between the result distribution and p is < 0.01, and the acceptance rate differs from Σ min(p, q) by < 0.01. Sampling with a fixed seed is reproducible. The top-p rule agrees with `sample_next`. The edge cases of the formula are tested.
- `test_arch_mtp.py`: shapes (with D = 2, the logits of the two MTP layers have lengths T−1 and T−2), and the main-model logits are exactly the same as the output of `Transformer.forward`. **Depth-1 loss calculated by hand**: MTP loss = cross-entropy on `tokens[i+2]`, and total loss = main loss + λ · MTP loss. **Causality**: a change to the tokens at position 7 and later does not affect the MTP output at positions 0–5, and a change at position 7 affects position 6 (which uses Emb(t₇)). **Gradients** reach the MTP module, the shared embedding / output head, and the layers of the main model. The loss decreases in a few training steps. MTP self-speculative decoding (greedy) is token-for-token identical to greedy decoding of the main model.

**A real production service** does not use a Python loop like this one. This is the industry practice:

- **vLLM**: `--speculative-config` supports draft models (`draft_model`), `ngram`, `mtp`, `eagle`, and other methods (also suffix decoding, MLP speculator, and others). `num_speculative_tokens` is the k of this chapter. Its rejection sampler has dedicated tests for "convergence to the target distribution" and end-to-end tests for "token-for-token identical with greedy decoding". The documentation example for `method: "mtp"` uses Xiaomi MiMo-7B.
- **SGLang**: `speculative_algorithm` can be `EAGLE` / `EAGLE3`, `STANDALONE` (an independent draft model), `NGRAM`, and others. The MTP layers of DeepSeek and other models use the alias `NEXTN` and go through the EAGLE path (`python/sglang/srt/speculative/spec_info.py`). There is also SpecForge, a tool to train EAGLE draft heads.
- The difficulty for a server is **batching**. Each sequence in a batch accepts a different number of tokens, so the KV cache rollback and the input length of the next round are different for each sequence. Servers also often use "tree verification" (verify several candidate paths at once) to increase the output per round. The scheduler of the inference engine does all of this. The batch = 1 implementation of this chapter does not cover it.

The correctness of `zero/arch/speculative.py` and `mtp.py` on CUDA was verified on an RTX 3090 (speculative decoding is token-for-token identical to greedy decoding; see Section 11 of [runs/2026-10-01-gpu0-check](../../runs/2026-10-01-gpu0-check/README.md)). We did not measure the speed of these two `zero` modules on a GPU. The "GPU measurements" section of this chapter measures the scaled-up version of the minimal code.

**How the main-line model uses this chapter.** The main-line model **does not have MTP**. GOAL.md 3.3 says that the main line takes no architecture risk. Also, at the 0.6B scale, there is no reliable public evidence about the effect of MTP on the quality of the main model. At the release in step 2, these methods can make it faster, but first measure them on a GPU:

1. **Train a separate small draft**: use the tokenizer of the main line and the same data to train a `zero` model of about 50 million to 100 million parameters. (Add one configuration in `configs/`; no code change is necessary.) Then do one round of distillation with the outputs of the main-line model (Chapter 17) to increase α.
2. **Prompt lookup**: tool calls copy many function names, parameter names, and JSON keys. Prompt lookup costs nothing, so try it first.
3. **Add MTP later**: freeze the main-line model and train only one `zero.arch.mtp.MTPModule`. The DeepSeek ablation shows that joint training improves the main model. But if we use MTP only as a draft, training it later can also work. This course did not verify this point: **experiment to be done**.

Which method to use, which k, and how much faster each one is on llama.cpp / vLLM: we will write this only after step 2 measures them on real hardware.

---

## Frontier notes

> **Methods that are not consensus. We mention them only here.**
>
> - **Medusa** (Cai et al. 2024): add several independent heads in parallel on the last layer of the target model. Head j predicts the token j positions ahead directly. It does not see the tokens between, like the parallel heads of Gloeckle et al. Then "tree attention" verifies several candidates at once.
> - **EAGLE / EAGLE-2 / EAGLE-3** (Li et al. 2024–2025): a very small autoregressive draft network. Its input is the hidden features of the target model (EAGLE-3 uses features from several layers). With a dynamic draft tree, its acceptance rate is high. It is now one of the most common draft methods in inference engines (the vLLM documentation puts it next to MTP as the methods with the largest benefit). Hugging Face also has many EAGLE-3 draft heads: NVIDIA trained one for gpt-oss-120b, and the community trained them for Llama, Qwen3, Kimi K2.6, MiniMax, and others. But inference companies or the community trained almost all of them **later**. We could not verify 3 leading model families that released EAGLE drafts **themselves** in the technical report or model card of a main version, so EAGLE is here. Its idea is very similar to MTP: the DeepSeek-V3 paper says that "keep the causal chain" is similar to EAGLE. The difference is that the primary purpose of MTP is to improve training.
> - **Tree verification** (SpecInfer, Medusa, EAGLE-2, and others): verify a tree of candidates at once instead of one chain. The expected output per round is higher, but the attention mask and the cache management are more complex.
> - **Looser acceptance rules** (lenience, typical acceptance, and others): they get a higher acceptance rate, but the output distribution is no longer exactly the target distribution.

## Adopters and sources

| Method | Adopters (main versions) | Sources |
|---|---|---|
| MTP (sequential multi-token prediction modules, training objective + inference draft) | **DeepSeek**: V3 (D = 1, λ = 0.3 → 0.1, acceptance rate of the second token 85%–90%, about 1.8× TPS), V4-Pro (`num_nextn_predict_layers: 1`); **Qwen**: Qwen3-Next-80B-A3B (model card: "MTP improves pretraining and makes inference faster"), all Qwen3.5 models (`mtp_num_hidden_layers: 1` in 0.8B); **Zhipu**: GLM-4.5 / 4.5-Air (1 MoE layer as the MTP layer, λ = 0.3 → 0.1), GLM-5 (`num_nextn_predict_layers: 1`); **MiniMax**: M2 (`use_mtp: true`, `num_mtp_modules: 3`); **Xiaomi**: MiMo-7B (`num_nextn_predict_layers: 1`), MiMo-V2-Flash (released 3 layers of MTP weights); **NVIDIA**: Nemotron 3 Super 120B-A12B (`num_nextn_predict_layers: 1`) | [DeepSeek-V3 technical report arXiv:2412.19437](https://arxiv.org/abs/2412.19437) (Sections 2.2, 4.5.1, 5.4.3); config.json: [DeepSeek-V3](https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/config.json), [DeepSeek-V4-Pro](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/config.json), [Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B/blob/main/config.json), [GLM-4.5](https://huggingface.co/zai-org/GLM-4.5/blob/main/config.json), [GLM-5](https://huggingface.co/zai-org/GLM-5/blob/main/config.json), [MiniMax-M2](https://huggingface.co/MiniMaxAI/MiniMax-M2/blob/main/config.json), [MiMo-7B-Base](https://huggingface.co/XiaomiMiMo/MiMo-7B-Base/blob/main/config.json), [Nemotron-3-Super-120B-A12B](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16/blob/main/config.json); model cards: [Qwen3-Next-80B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct), [MiMo-V2-Flash](https://huggingface.co/XiaomiMiMo/MiMo-V2-Flash); [GLM-4.5 technical report arXiv:2508.06471](https://arxiv.org/abs/2508.06471) (Sections 2.1, 2.4); [MiMo-7B technical report arXiv:2505.07608](https://arxiv.org/abs/2505.07608) |
| Speculative decoding (draft model / n-gram / MTP draft + verification with rejection sampling) | Industry standard (class B of GOAL.md 2.1): built into **vLLM** and **SGLang**. Model makers release dedicated drafts: **Google Gemma 4** (E2B, E4B, 12B, 26B-A4B, and 31B each have a `*-it-assistant` MTP draft model; the model card says "up to about 3×, identical quality"). **DeepSeek-V3**, **GLM-4.5**, and **MiMo** state in their technical reports / model cards that they use MTP for speculative decoding | Leviathan et al. 2023; Chen et al. 2023; [vLLM speculative decoding documentation (source file)](https://github.com/vllm-project/vllm/blob/main/docs/features/speculative_decoding/README.md) and [MTP documentation](https://github.com/vllm-project/vllm/blob/main/docs/features/speculative_decoding/mtp.md); [SGLang](https://github.com/sgl-project/sglang); [gemma-4-31B-it-assistant model card](https://huggingface.co/google/gemma-4-31B-it-assistant) |

Notes:

- We read all configs and model cards from Hugging Face in 2026-09. The config.json of Qwen3-Next has no MTP fields (it stores the MTP weights in a different way from Qwen3.5), so we use the explicit statement of the model card.
- The config of Kimi K2 has `num_nextn_predict_layers: 0` (Table 1 of the GLM-4.5 technical report also lists 0 MTP layers), so we do not count Kimi K2 as an adopter.
- This chapter did not check item by item how the MiMo-7B technical report uses MTP (how many layers in pretraining, how many layers at inference, the acceptance rate). It cites only the config and the example in the vLLM documentation: **to be verified**.
- The MTP draft of Gemma 4 is an independent small model (`Gemma4AssistantForCausalLM`, 4 layers, shares the KV cache with the target). It is not exactly the same as a DeepSeek-style "MTP module inside the main model". Here we count it as "speculative decoding + a draft that the model maker releases", not as "a DeepSeek-style MTP module". This chapter did not read the Gemma 4 technical report (the model card links to arXiv:2607.02770). Its description of how MTP is trained is **to be verified**.

---

## Guided questions

Ask Claude Code these questions. Continue until you can explain the answers in your own words:

1. Speculative decoding assumes that "verifying k+1 positions is as fast as generating 1". When is this assumption not true? Ask Claude Code to help you estimate: for the main-line model on an H100, what is the arithmetic intensity of one decode step at batch = 1 and at batch = 64? What is the arithmetic intensity of one verification round at k = 4? (Hint: `02_prefill_decode.py` of Chapter 21.)
2. In the derivation for sampling, what happens if the draft samples a token with `q(x) = 0` (the draft thinks that the token is impossible)? What happens if the target thinks that it is impossible (`p(x) = 0`)? When the temperature → 0, does the sampling rule become the greedy rule?
3. This chapter says "the probability of one acceptance α = 1 − TV(p, q)". Why is it the TV distance and not the KL divergence? If you **train** a draft to make α as high as possible, which loss function must you use? (Hint: search for DistillSpec.)
4. When the MTP module of DeepSeek-V3 predicts `t_{i+2}`, it can see the embedding of `t_{i+1}`. The parallel heads of Gloeckle et al. and Medusa cannot see it. What does this mean for the "denser signal" in training, and for the acceptance rate at inference?
5. In `05_mtp.py`, the "accuracy for the character after next" of the MTP module is about as high as the "next-character accuracy" of the main model. Does this mean that predicting two steps ahead is as easy as predicting one step ahead?
6. Why does the MiniMax blog list "working together with speculative decoding" as one of the unsolved problems of linear attention / hybrid architectures? (Hint: think about rollback. The state of linear attention is a running sum, and the rejected drafts are already added into it. How do you undo them?)

## Hands-on tasks

For each task, run the code and look at the results.

**Task 1 (basic)**: In `04_speedup_formula.py`, set c to the value that `01_models_and_cost.py` measures on your machine. Use the α that `02` measures, and find the best k. Then in `02`, run only this k and the k values on each side of it. Is the fastest k in the measurement the same as the prediction of the formula? If not, find the overhead that the formula does not include.

**Task 2 (core)**: Add a prompt-lookup draft to `02_greedy_speculative.py`: in `seq`, find the most recent piece that matches the last 3 characters, and take the k characters after it. Compare its acceptance rate and number of target forward passes with the 1-layer draft model. Then use a prompt that "repeats" (for example, copy a piece of dialogue two times, and let the model continue with the third copy). How does the difference between the two drafts change? Compare your code with `prompt_lookup_draft` in `zero/arch/speculative.py`.

**Task 3 (challenge)**: Use distillation to increase the acceptance rate of the draft. In `01_models_and_cost.py`, add a training function. Its loss is the KL divergence between the draft distribution and the target distribution (the target model is frozen; this is the logits distillation of Chapter 17). Train for the same number of steps. Before and after distillation, compare the measured acceptance rate at temperature 1.0 (part 3 of `03`) and the acceptance rate with greedy decoding (`02`). Then try the TV distance as the loss. Which loss is better for α?

---

## Go deeper: CS336

This chapter matches Stanford CS336 (Spring 2026) <https://cs336.stanford.edu/>:

- **Lecture 10: Inference.** The speculative sampling part of this lecture (`speculative_sampling()` in the lecture source `lecture_10.py`, see <https://github.com/stanford-cs336/lectures>) starts from "prefill is parallel and decode is limited by bandwidth, so checking is faster than generating". It gives the same proof as Section 4 of this chapter, with a vocabulary of two tokens. It also gives the draft pairs 70B with 8B and 8B with 1B, distillation to make the draft more similar to the target, and two methods that improve the draft, Medusa and EAGLE. Note: in the CS336 lecture notes, **the draft is p and the target is q**. This is the opposite of this chapter (and of the original paper of Leviathan et al.).
- **CS336 does not go deep into**: MTP as a training objective (the sequential MTP module of DeepSeek-V3, the loss weight, the ablation), and the implementation details in inference engines. This lecture does not explain them. See the references of this chapter.

---

## References

- Leviathan, Kalman, Matias. *Fast Inference from Transformers via Speculative Decoding*, ICML 2023: <https://arxiv.org/abs/2211.17192>
- Chen, Borgeaud, Irving, Lespiau, Sifre, Jumper. *Accelerating Large Language Model Decoding with Speculative Sampling*, 2023: <https://arxiv.org/abs/2302.01318>
- Stern, Shazeer, Uszkoreit. *Blockwise Parallel Decoding for Deep Autoregressive Models* (the predecessor of the greedy "guess in parallel, then verify"), NeurIPS 2018: <https://arxiv.org/abs/1811.03115>
- Gloeckle, Idrissi, Rozière, Lopez-Paz, Synnaeve. *Better & Faster Large Language Models via Multi-token Prediction*, ICML 2024: <https://arxiv.org/abs/2404.19737>
- DeepSeek-AI. *DeepSeek-V3 Technical Report*, 2024: <https://arxiv.org/abs/2412.19437>
- GLM-4.5 Team. *GLM-4.5: Agentic, Reasoning, and Coding (ARC) Foundation Models*, 2025: <https://arxiv.org/abs/2508.06471>
- Xiaomi LLM-Core Team. *MiMo: Unlocking the Reasoning Potential of Language Model – From Pretraining to Posttraining*, 2025: <https://arxiv.org/abs/2505.07608>
- Cai et al. *Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads*, 2024: <https://arxiv.org/abs/2401.10774>
- Li, Wei, Zhang, Zhang. *EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty*, 2024: <https://arxiv.org/abs/2401.15077>; *EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test*, 2025: <https://arxiv.org/abs/2503.01840>
- Zhou et al. *DistillSpec: Improving Speculative Decoding via Knowledge Distillation*, 2023: <https://arxiv.org/abs/2310.08461>
- Saxena. *Prompt Lookup Decoding*: <https://github.com/apoorvumang/prompt-lookup-decoding>
- vLLM speculative decoding documentation (source file): <https://github.com/vllm-project/vllm/blob/main/docs/features/speculative_decoding/README.md>; the DeepSeek MTP implementation of vLLM: <https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/deepseek_mtp.py>
- SGLang: <https://github.com/sgl-project/sglang>
- Model configurations and model cards (read in 2026-09): see the links in the "Adopters and sources" table above; [Kimi-K2-Instruct config](https://huggingface.co/moonshotai/Kimi-K2-Instruct/blob/main/config.json) (`num_nextn_predict_layers: 0`)
- Google Research. *Looking back at speculative decoding* (the review blog that the CS336 lecture notes cite): <https://research.google/blog/looking-back-at-speculative-decoding/>
- CS336 Spring 2026 lecture source: <https://github.com/stanford-cs336/lectures>; course page: <https://cs336.stanford.edu/>

**Next chapter**: The architecture experiments of Part 5 end here: a smaller KV cache (GQA, MLA), cheaper long context (sliding window, linear-attention hybrids), less compute (MoE), and faster generation (speculative decoding, MTP). The last chapter puts these pieces back into real models. It takes apart several of the newest open flagship models at the time of writing, draws the evolution tree of the architectures, and puts our main-line model into the same table. Chapter 26: a panorama of the current state-of-the-art open models.
