# Chapter 17: Distillation — A small model learns from a large model

**English** · [中文](README.zh.md)

> **Goal**: After this chapter, you can explain why the "soft label" of a teacher carries more information than a one-hot label. You can derive the KD loss with temperature and its gradient by hand, and do a parity check with `zero`. You can explain what logits distillation, sequence-level distillation, and on-policy distillation each need (especially "the same vocabulary"). You can explain why the main-line model can use only sequence-level distillation. You can use "many samples + execution check" to filter clean tool-call data. You can also read the license of a teacher model and decide if you can use it as a teacher.

📺 **Video**: Not published yet. To render it on your computer, run `bash chapters/17-distillation/video/build.sh`.
🧪 **Self-check**: After the chapter, type `/ch17-distillation` in Claude Code.

---

In the last chapter, we used SFT to change the base model into an assistant. The assistant answers in the correct format and can call tools. We gave it a set of "question → reference answer" conversations and calculated the cross-entropy only on the answers. This chapter answers the next question: **where do the reference answers come from, and which answers teach best?**

Answers that people write cost too much. Answers that the small model writes itself are not good enough. Almost all leading open model families give the same answer: **use a stronger model as the teacher**.

- **Llama 3.2** 1B and 3B: Meta first pruned them from Llama 3.1 8B. Then the logits of 8B and 70B became the training targets at each token.
- **Gemma 2** 2B and 9B: pretrained with knowledge distillation instead of "predict the next token". **Gemma 3** uses distillation for all sizes.
- **Qwen3** 0.6B–14B and 30B-A3B: these models do not use the four-stage post-training of the flagship model. They use "strong-to-weak distillation", which needs only about 1/10 of the GPU hours.
- **DeepSeek-R1-Distill**: about 800,000 samples that DeepSeek-R1 generated, used only for SFT on the base models of Qwen2.5 and Llama 3. The 1.5B student scored higher than GPT-4o on math benchmarks.

The code of this chapter (all of it runs on a CPU):

```bash
uv run python chapters/17-distillation/code/01_soft_labels.py         # soft labels, temperature, KD gradient, parity check with zero (a few seconds)
uv run python chapters/17-distillation/code/02_toy_distill.py         # toy experiment: hard labels vs sequence-level vs logits distillation (about 2 min of CPU time)
uv run python chapters/17-distillation/code/03_forward_reverse_kl.py  # forward KL vs reverse KL, a distribution with two peaks (a few seconds)
uv run python chapters/17-distillation/code/04_rejection_sampling.py  # the funnel of rejection sampling + execution check (a few seconds)
uv run python chapters/17-distillation/code/05_shared_vocab.py        # why the vocabulary must be the same; the parameter cost of a new vocabulary (a few seconds)
```

## 1. Intuition: one answer vs a full distribution

### 1.1 A one-hot label says only "which answer is correct"

Remember Chapter 5. When we train a classifier, the label is one-hot: the correct word is 1, and all other words are 0. (A language model is also a classifier: it classifies over the vocabulary.) Suppose that the context is "今天天气很" ("Today the weather is very"). The next character in the training text is "好" ("good"). Then the model learns only "好".

But "热" ("hot"), "冷" ("cold"), and "晴" ("sunny") also make sense. "猫" ("cat") and "跑" ("run") do not make sense at all. The one-hot label discards all of this information. It does not separate "second-best answers" from "absurd answers": all of them get 0.

### 1.2 The soft label of the teacher also tells you the "second-best" answers

Suppose that a large trained model (the **teacher**) is available. At this position, the teacher does not give one character. It gives a full probability distribution: a **soft label**. Here is the example from `01_soft_labels.py`:

| | 好 | 热 | 冷 | 晴 | 猫 | 跑 | Entropy |
|---|---:|---:|---:|---:|---:|---:|---:|
| one-hot | 1.000 | 0 | 0 | 0 | 0 | 0 | 0 bit |
| teacher p_T | 0.607 | 0.223 | 0.100 | 0.067 | 0.002 | 0.001 | 1.535 bit |

The entropy of the one-hot label is 0. At each training position, it gives only one piece of information: which answer is correct. The teacher distribution has an entropy of 1.5 bit. It also gives the **relative sizes**: "热 is more probable than 冷, and 晴 is much more probable than 猫". This information hides in the probabilities of the wrong answers. People later called it **dark knowledge**.

Hinton et al. (2015) give an example with a network that recognizes handwritten digits. For one "2", the network gives a probability of 10⁻⁶ to "3" and 10⁻⁹ to "7". For another "2", the two numbers can be the other way around. These two very small numbers tell us if this "2" looks more like a 3 or more like a 7.

**Knowledge distillation** (KD) trains a small model (the **student**) to learn the full distribution of the teacher, not only the one-hot answer. The Gemma 2 report states it directly. It replaces the one-hot label at each token with the next-token distribution that the large model calculates. Then **each training step receives richer information**. With this method, the Gemma team let the 2B and 9B models "simulate" training on many more tokens than the real data has.

### 1.3 Temperature: make the dark knowledge larger

A teacher is usually very confident. The probabilities of the wrong answers are so small that the student can ignore them ("猫" has only 0.002 in the table above). The method of Hinton uses the **temperature** from Chapter 5: divide the logits by τ, then apply softmax, `p_T^τ = softmax(z_T / τ)`.

| τ | 好 | 热 | 冷 | 晴 | 猫 | 跑 | Entropy (bit) | p(晴)/p(猫) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.5 | 0.851 | 0.115 | 0.023 | 0.010 | 0.000 | 0.000 | 0.752 | 1998.2 |
| 1 | 0.607 | 0.223 | 0.100 | 0.067 | 0.002 | 0.001 | 1.535 | 44.7 |
| 2 | 0.412 | 0.250 | 0.168 | 0.137 | 0.021 | 0.012 | 2.046 | 6.7 |
| 4 | 0.295 | 0.230 | 0.188 | 0.170 | 0.066 | 0.051 | 2.373 | 2.6 |

A larger τ gives more probability to the low-ranked candidates, so the student can "see" more differences. But the ranking never changes (晴 > 猫). During training, the student uses the same τ. At inference, τ goes back to 1.

## 2. Logits distillation: match the teacher distribution at each position

### 2.1 Formula

At each position t of the same text, the teacher and the student each give a distribution over the vocabulary. Logits distillation minimizes the KL divergence between the two distributions. (The KL divergence is a "relative" of the cross-entropy from Chapter 5.)

```
L_KD = τ² · KL(p_T^τ ‖ p_S^τ) = τ² · Σ_v p_T^τ(v) · (log p_T^τ(v) − log p_S^τ(v))
```

Take the mean over the positions in the mask (as in SFT, only on the answer). In practice, training often mixes this loss with the normal cross-entropy: `L = (1 − α)·CE + α·L_KD`.

`KL(p_T ‖ p_S) = H(p_T, p_S) − H(p_T)`. The teacher entropy `H(p_T)` does not depend on the student. Thus, for the student, **to minimize the KL is the same as to minimize "the cross-entropy with the teacher distribution as the label"**. Replace the teacher distribution with a one-hot label, and the loss becomes the cross-entropy of normal SFT. A hard label is only a special case of a soft label. The Gemma 2 report uses this cross-entropy form.

### 2.2 Gradient: from `p − onehot` to `p_S − p_T`

In Chapter 5, we derived the gradient of the cross-entropy for the logits: `p − onehot`. Use the same steps here. `log p_S^τ(v) = z_S(v)/τ − logsumexp(z_S/τ)`. The derivative for `z_S(k)` is `(1/τ)·([v = k] − p_S^τ(k))`. Put this derivative into `−Σ_v p_T^τ(v) log p_S^τ(v)`, and use `Σ_v p_T^τ(v) = 1`:

```
∂L_KD/∂z_S(k) = τ² · (1/τ) · (p_S^τ(k) − p_T^τ(k)) = τ · (p_S^τ(k) − p_T^τ(k))
```

The gradient is still "the prediction of the student minus the target". Only the target changes, from a one-hot label to the teacher distribution. In the code, it is one line:

```python
def kd_grad(z_s, z_t, tau):
    return tau * (softmax(z_s, tau) - softmax(z_t, tau))   # ∂L/∂z_S = τ·(p_S^τ − p_T^τ)
```

Part 3 of `01` does a central-difference check. At τ = 1 and 2, the maximum difference between the analytic gradient and the numerical gradient is 1.2×10⁻¹¹ and 3.1×10⁻¹¹. Now compare the KD gradient with the hard-label gradient for the same student:

| | 好 | 热 | 冷 | 晴 | 猫 | 跑 |
|---|---:|---:|---:|---:|---:|---:|
| Hard label `p_S − onehot` | −0.5231 | +0.1064 | +0.2893 | +0.0391 | +0.0645 | +0.0237 |
| KD (τ = 1) `p_S − p_T` | −0.1301 | −0.1169 | +0.1889 | −0.0281 | +0.0630 | +0.0232 |

The hard label pushes only "好" up and pushes all other tokens down. KD finds that the student gives too little probability to "热" and "晴". So KD also pushes them up (the gradient is negative). It pushes down only "冷", "猫", and "跑", which the student overestimates.

**Why multiply by τ²?** Without τ², the gradient is `(1/τ)·(p_S^τ − p_T^τ)`. For a large τ, both distributions are close to uniform, and their difference also decreases as 1/τ. Together, the gradient decreases as 1/τ². (The output of `01`: at τ = 1, 2, 4, 8, the gradient norm is 0.268, 0.100, 0.035, 0.011.) With τ², the size of the gradient stays about the same for all τ. Then you do not need to tune the learning rate and α again when you change the temperature.

A large τ also shows another view of KD: `p ≈ (1 + z/τ)/V`. When the logits have a mean of zero, the gradient ≈ `(z_S − z_T)/V`. This gradient **makes the student logits match the teacher logits directly** (Hinton et al. 2015, Section 2.1).

### 2.3 Parity check with the production code

The last part of `01` runs the minimal version and `zero.post.distill.kd_loss` on the same batch of random logits (2×5 positions, vocabulary 11, 2 positions masked):

| | Minimal code | zero | Difference |
|---|---:|---:|---:|
| τ = 1 | 4.164090 | 4.164090 | 3.0×10⁻⁷ |
| τ = 2 | 5.177818 | 5.177818 | 3.0×10⁻⁷ |
| top-3 | 4.800477 | 4.800477 | 1.2×10⁻⁷ |

**top-k**: the teacher keeps only the k most probable tokens and normalizes them again over these k tokens. The student is not normalized again. A vocabulary can have more than 100,000 tokens. Then the full teacher distribution at each position is too large to store. If you store only the top k, you save storage by several orders of magnitude.

Gemma 3 uses a similar method, but a random one. At each token, it **samples 256 logits by the teacher probabilities**. It sets the logits that it did not sample to 0 and normalizes again.

### 2.4 A hard requirement: the same vocabulary

Logits distillation compares the two distributions position by position and dimension by dimension. Thus **position t must be the same "next token" on both sides, and dimension v must be the same token**. In other words, the teacher and the student must use **the same tokenizer**.

`05_shared_vocab.py` uses two toy vocabularies to cut the same sentence "今天天气很好" ("The weather is very good today"). The student cuts it into `今天 | 天气 | 很 | 好` (4 tokens). The teacher cuts it into `今天天 | 气很 | 好` (3 tokens). Even the numbers of positions are different. The `kd_loss` of `zero` raises an error when the shapes are different. `run_distill` also compares the hashes of the two tokenizers.

All four adopters above meet this requirement. Llama 3.2 uses the same tokenizer as Llama 3.1. All sizes of Gemma 2/3 share one vocabulary of 256,000 / 262,000 tokens. All sizes of Qwen3 share one vocabulary. DeepSeek-R1-Distill is different. Its students are Qwen2.5 and Llama, and their tokenizers are different from the tokenizer of R1. For this reason, it **can use only sequence-level distillation** (next section).

**This brings us back to the vocabulary decision of Chapter 13.** The main-line model uses its own trained vocabulary of 65,536 tokens (`configs/main/pretrain.toml`). This vocabulary is different from the vocabulary of every open teacher. Thus **the main line can use only sequence-level distillation**. Now look at it from the other side: what if we had used the Qwen tokenizer from the start? `05` keeps the shape of the main-line model (28 layers, width 1280, tied embeddings) and changes only the vocabulary:

| Tokenizer | Vocabulary | Embedding | Total parameters | ≤ 0.8B? |
|---|---:|---:|---:|---|
| Main line (own trained BPE, Chapter 13) | 65,536 | 83.9M | 689.5M | Yes |
| Qwen3 tokenizer | 151,936 | 194.5M | 800.1M (+110.6M) | No |
| Qwen3.5 tokenizer | 248,320 | 317.8M | 923.5M (+234.0M) | No |

With the Qwen3 tokenizer, we could do logits distillation and on-policy distillation (Section 4) from a Qwen3 teacher. The cost is 110 million parameters in the embedding, and the model goes just over the 0.8B limit. We would also have to accept a vocabulary that was not trained on our mix of Chinese, English, and code. This is a real trade-off, and neither choice is "always correct". The main line chose the first option: a small vocabulary, its own tokenizer, and a fully open recipe. Thus its distillation is only sequence-level.

## 3. Sequence-level distillation: the teacher writes, the student copies

### 3.1 Formula: it is SFT

**Sequence-level distillation** (sequence-level KD, Kim & Rush 2016) does not look at the teacher logits. It looks only at the **text that the teacher writes**. For each prompt, the teacher generates an answer `y ~ p_T(·|x)`. Then the student does normal SFT on this answer: `L = −log p_S(y|x)`.

What does it optimize? Maximum likelihood on teacher samples has the expected value `E_{y~p_T}[−log p_S(y|x)]`. This value is the sequence-level cross-entropy `H(p_T, p_S)`. Subtract a constant that does not depend on the student, and you get the **sequence-level forward KL(p_T ‖ p_S)**.

Thus its objective has the same direction as logits distillation. It only replaces the full distribution with samples. Each sample carries only "the one path that the teacher chose", so it has less information than the full distribution. But **it requires only that the teacher "can generate text"**. Any tokenizer, any architecture, and even a remote HTTP API can work. The "offline distillation" stage of DeepSeek-R1-Distill and of the Qwen3 small models uses this method.

### 3.2 Toy experiment: the same number of steps, four signals

`02_toy_distill.py` trains a character-level language model on Shakespeare text (`assets/tiny_corpus/shakespeare.txt`, 1.11 million characters). The model sees the previous 8 characters and predicts the next one:

- Teacher: an MLP with 429,665 parameters. It trains for 3000 steps on all of the training text (about 1 million characters). Its validation result is 2.439 bits/char.
- Student: an MLP with 8,905 parameters. It **sees only 20,000 characters** of training text. (This simulates "the student has little good data".) We compare four training signals with the same initialization, 2000 steps, batch 256, and 3 random seeds:

> **Note:** The numbers of the training experiments in this chapter come from one CPU run on the course build machine. Different machines and different versions of the low-level math libraries do floating-point operations in a slightly different order. Training for hundreds of steps makes these small differences larger. Thus your numbers can be different from the second or third decimal place. Trust the conclusions below, which do not depend on the exact values. For a rerun on another server in 2026-10, see [runs/2026-10-01-gpu0-check/chapters-16-20.md](../../runs/2026-10-01-gpu0-check/chapters-16-20.md).

| Training signal | On the 20,000 training characters | Validation bits/char (mean of 3 seeds) | Each seed |
|---|---:|---:|---|
| A Hard labels (20,000 real characters) | 2.678 | **3.679** | 3.626 / 3.739 / 3.673 |
| B Sequence-level distillation (20,000 characters that the teacher wrote) | 3.302 | **3.362** | 3.327 / 3.387 / 3.371 |
| C Logits distillation (τ = 2, the same 20,000 characters) | 2.867 | **3.290** | 3.258 / 3.326 / 3.286 |
| D 0.5·hard label + 0.5·KD | 2.693 | **3.344** | 3.326 / 3.374 / 3.332 |
| R Reference: hard labels, all 1,000,000 characters | 3.123 | 3.143 | 3.114 / 3.170 / 3.145 |

(A uniform random guess gives log₂65 = 6.022 bits/char.) How to read this table:

- **A overfits.** On its own 20,000 characters, A gets 2.678. On the validation set, it gets 3.679: a difference of 1 bit. On small data, one-hot labels push the student to "memorize the answers".
- **C is the best.** It uses the same 20,000 characters and the same number of steps. Only the labels change to the teacher distribution. The validation result decreases to 3.290, which is 0.39 bit better than A, and all three seeds are better. The soft labels carry dark knowledge, and they also act as a regularizer. (On the training set, C gets 2.867: it "memorizes" less well than A.)
- **B is also much better than A** (3.362). The student never saw real text. It only copied 20,000 characters that the teacher wrote. The teacher writes "Shakespeare-style nonsense" (`02` prints a sample). But the statistics of this text are smoother and more representative than a real sample of 20,000 characters.
- **D is between the two.** With hard labels in the mix, D is a little worse than pure KD. In this toy setup, the real labels are only 20,000 characters, so the teacher is more reliable. Sometimes the teacher is less reliable than the data (for example, when the teacher itself makes errors). Then increase the weight of the hard labels.
- **R** shows that the limit for a student of this capacity is about 3.14. Distillation moves the student "with only 20,000 characters" from 3.68 to a point that is only about 0.15 bit from "enough data".

Hinton et al. 2015, Section 6, did the same experiment on a speech model. With only 3% of the data, hard labels overfit badly: they needed early stopping, and the accuracy was 44.5%. Soft labels converged to 57% without early stopping.

The result also agrees with the ablation in Gemma 2. A 2B model trained on 500 billion tokens gets a mean of 60.3 on 3 benchmarks when trained from scratch. With distillation from a 7B teacher, it gets 67.7. **Do not extrapolate the toy numbers to large models.** The student here is a character-level MLP. The experiment shows only one point: "with limited data, soft labels carry more information than one-hot labels".

## 4. Forward KL, reverse KL, and on-policy distillation

### 4.1 The same KL, two directions

Both distillation methods above minimize the **forward KL** `KL(p_T ‖ p_S)`. Its expected value is over the **teacher** distribution. The KL is not symmetric. The **reverse KL** `KL(p_S ‖ p_T) = Σ p_S·(log p_S − log p_T)` takes the expected value over the **student** distribution. Sometimes the student does not have enough capacity for all behaviors of the teacher. Then the two directions give very different results.

`03_forward_reverse_kl.py` shows this with a 1D example. The teacher has two peaks, `0.6·N(−2, 0.6²) + 0.4·N(2.5, 0.8²)`, and the student can only be **one** Gaussian.

| Minimize | Start μ₀ | Converges to μ | σ | Mass in left-peak region | Mass in valley [−0.5, 1] | Mass in right-peak region |
|---|---:|---:|---:|---:|---:|---:|
| (teacher p) | | | | 0.596 | 0.016 | 0.388 |
| Forward KL(p ‖ q) | 0.0 | −0.22 | 2.43 | 0.453 | **0.240** | 0.307 |
| Reverse KL(q ‖ p) | −1.0 | −2.00 | 0.60 | **0.994** | 0.006 | 0.000 |
| Reverse KL(q ‖ p) | 1.5 | 2.49 | 0.81 | 0.000 | 0.033 | **0.967** |

- **Forward KL "covers all modes" (mode covering).** Where p > 0, q must give some probability. If it does not, `log q → −∞`, and the penalty is infinite. So the student spreads out and covers both peaks. The cost: it puts 24% of the probability in the valley between the peaks. The teacher almost never goes there (it has only 1.6% there). For a language model, this means: "learn a little of both phrasings, and produce a sentence that is neither of them".
- **Reverse KL "picks one mode" (mode seeking).** If q gives probability where p ≈ 0, q gets a large penalty. Where p is large, q can give probability or not. The student shrinks into one peak and gives up the other peak completely. The start point decides which peak.

### 4.2 On-policy distillation: the student writes, the teacher scores each token

The reverse KL takes the expected value over the student distribution. To calculate it, **the student must sample by itself**. The student generates an answer. The teacher gives its own log-probability **for each token that the student wrote**. Training minimizes `log p_S(y_t) − log p_T(y_t)` at each position. This method is **on-policy distillation** (GKD by Agarwal et al. 2023, MiniLLM by Gu et al. 2023).

On-policy distillation solves an old problem of sequence-level distillation. The student practiced only on the paths that the teacher took. When the student makes an early error, it arrives at a place where the teacher never went. Then the errors grow larger and larger (**exposure bias**). On-policy distillation gives the student a correction at each token, **at the places where the student itself makes errors**.

The Thinking Machines blog (Lu et al., 2025) puts the three methods in one table. SFT is off-policy with a dense signal. RL is on-policy with a sparse signal (one reward for a full answer). On-policy distillation is **on-policy with a dense signal**. In the implementation, you can reuse RL code and replace the advantage of each token with `−(log p_S − log p_T)`. We see this structure again in Chapter 19 with GRPO.

How much does it help? Table 21 of the Qwen3 report shows it. All rows start from the same 8B model after offline distillation:

| Qwen3-8B | AIME'24 | AIME'25 | MATH500 | LiveCodeBench v5 | GPU hours |
|---|---:|---:|---:|---:|---:|
| After offline distillation | 55.0 | 42.8 | 92.4 | 42.0 | — |
| + Reinforcement learning | 67.6 | 55.5 | 94.8 | 52.9 | 17,920 |
| + On-policy distillation | 74.4 | 65.5 | 97.0 | 60.3 | 1,800 |

With about 1/10 of the GPU hours, all scores are higher than with RL directly.

**Consensus decision (GOAL.md 2.1)**: GOAL lists on-policy distillation as "to be verified". We checked the technical reports item by item. At least five independent leading families clearly use it in their main releases:

- **Qwen3**: the second stage of strong-to-weak distillation for small models (§4.5).
- **Gemma 2**: in the SFT stage, "distillation from the teacher on the student's distribution", with citations of GKD and MiniLLM (§4).
- **GLM-5**: the last step of post-training, "on-policy cross-stage distillation" (§3.5).
- **Xiaomi MiMo-V2-Flash**: multi-teacher on-policy distillation, MOPD (§4).
- **DeepSeek-V4**: on-policy distillation replaces the full mixed-RL stage, with more than ten domain experts as teachers (§5.1).

This meets rule A, so on-policy distillation **goes into the main text**.

**The main line cannot use an external teacher for it.** In on-policy distillation, the teacher scores the tokens of the student. Like logits distillation, it requires the same tokenizer, and our own tokenizer is different from every open teacher's. On-policy distillation across tokenizers is still a research topic (see "Frontier notes").

**The main line uses it in the GLM-5 form: cross-stage self-distillation** (decided 2026-10-10). The teachers are the main line's **own** checkpoints of the earlier stages, so the tokenizer is the same by construction. After GRPO, the student samples on two prompt pools: tool-call tasks scored by the GRPO checkpoint, and chat prompts scored by the checkpoint after distillation. It learns back what RL made worse without losing the tool calls (`zero/post/opd.py`, the last post-training stage; `configs/main/opd.toml`). The older `on_policy_distill` in `zero/post/distill.py` (one teacher, after offline distillation, `on_policy_steps`) stays for the tiny demo.

## 5. Rejection sampling: sample several times, keep only the verified ones

### 5.1 The teacher also makes errors

The teacher is not perfect. In a tool call, it can write wrong arguments, choose the wrong tool, or write broken JSON. It can also skip the tool call and invent an answer. Sequence-level distillation teaches all these errors to the student.

Almost all teams that make teacher data add a **rejection sampling** step. The teacher samples the same question several times. The team keeps only the samples that **an automatic check verifies as correct**.

- **DeepSeek-R1**: for reasoning data, "sample several answers for each prompt and keep only the correct ones". This step gave about 600,000 samples. With about 200,000 non-reasoning samples, the total is about 800,000. R1-Distill used this data.
- **Qwen3**: for the cold-start data, QwQ-32B generates N candidates for each question. A filter removes six types of answers, for example a wrong final answer, much repetition, or a clear guess.
- **GLM-5**: the reasoning data is "synthesized with rejection sampling". For code and agent data, the team runs trajectories in many executable environments and then filters them.
- **Llama 3.2**: each round of alignment is SFT → rejection sampling → DPO.

Tool calls are a natural fit for this method: to know if a call is correct, **run it**.

### 5.2 The funnel

`04_rejection_sampling.py` shows this filter on the 200 training tasks of `zero/post/envs/tool_env.py`. The teacher here is a **rule-based simulator**, not a language model. It makes a few common types of errors with fixed probabilities. (We chose the probabilities arbitrarily, only to show the funnel.) The filter is the production function `zero.post.distill.teacher_trajectories`:

1. The first-turn call must get the full score: the function name and the normalized arguments are the same as the gold call.
2. Then the filter **really runs the tool** and gives the result back.
3. The teacher writes the final answer. The answer must pass `score_final_answer`: all key points are present, and it does not list many numbers to get a lucky match.

| Gate | Remaining | Share |
|---|---:|---:|
| Candidates (200 tasks × 8 samples) | 1600 | 100% |
| Gate 1: correct format (complete JSON, matched tags, no fake tool result) | 1303 | 81.4% |
| Gate 2: correct call (function name + normalized arguments; for the calculator, the same execution result is also accepted) | 596 | 37.2% |
| Gate 3: execution + correct final answer | 497 | 31.1% |
| Gate 4: at most 1 for each task | 172 | (172/200 tasks have data) |

Look at the error types. Of the 291 samples with wrong arguments, 43 passed. All samples with the wrong tool, broken JSON, a direct answer with no tool call, one extra call, or nonsense were removed. Without the filter, 68.9% of the candidates did not pass verification, and most of them are wrong.

**Fewer samples are better than wrong samples.** The filter ran and checked each of the 172 kept samples. They are much more useful than 1600 samples in which more than half are wrong.

The other 28 tasks kept no sample. 21 of them are tasks that "need no tool". For the reason, see 5.3.

Look at the 43 samples with "wrong arguments that still passed". 24 are weather_compare and 19 are weather. They write "北京" (Beijing) as "北京市" (Beijing City), or similar. After normalization, they are the same as the gold arguments, so it is correct to accept them.

Before the fix of the scorer, 4 weekday samples also passed. Their date was wrong by exactly 7 days, so the weekday did not change. **The answer is correct, but the arguments are wrong**, and a comparison of the execution results alone cannot find this error. Now, for all tools except the calculator, the scorer compares the normalized arguments (`args_equivalent`). So the filter removes these 4 samples. A verifier guarantees only the things that it checks.

### 5.3 The verifier has a hole, and we really saw it

Before the fix of the scorer, this part of the output was more important. The kept data had 21 samples from tasks that "need no tool" (greetings, a few words of encouragement). 9 of these answers were nonsense. For example, the user said "讲一句鼓励的话" ("Say something encouraging"). The assistant answered "坚下云，气温 28°C。" ("<nonsense words>, temperature 28°C."). **This answer still passed verification.**

The reason: `score_final_answer` checks only that "all expected key points appear", and these tasks have no key points. `score_tool_calls` checks only that there is "no unnecessary tool call".

This is not an invented example. **The only teacher sample that passed verification in the smoke test was exactly this one.** It was in `out/smoke/distill/teacher.jsonl` at that time. An excerpt is in `video/data/smoke_before_fix.json` (see "Main-line progress").

The current fix is in `zero/post/envs/tool_env.py`. For a task that needs no tool, a normal answer gets only `NO_TOOL_REWARD = 0.5` and is marked "cannot be checked". It does not count as verified, so it does not go into the distillation data (the 21 tasks above keep 0 samples). An answer with a number that is not in the question gets 0.

The cost: the distillation data has no examples of "no tool call when no tool is needed". The human-reviewed SFT data (Chapter 16) gives these examples. Later, we may want the teacher to generate this type of data too. Then we need a real content scorer (rules or a judge model), and the report must give the pass rate of this type separately.

## 6. The license of the teacher

The rule of GOAL.md 3.3: **use as a teacher only an open-weight model whose license allows "using its outputs to train other models"**. Record the name, version, and license of the teacher.

We release the distillation data together with our model. Thus the terms of the license about "outputs" and "derivative models" apply to us. In September 2026, we read the original text of each license. (The column "Terms" in the table below gives the main points of the original text. It is not legal advice.)

| Teacher | License | Terms about "using outputs to train other models" | Can it be a main-line teacher? |
|---|---|---|---|
| All Qwen3 models, all Qwen3.5 models | Apache-2.0 | No additional restrictions (Qwen3 report: "all Qwen3 models are publicly accessible under Apache 2.0") | ✅ |
| DeepSeek-R1 | MIT | The README states: "support commercial use, allow for any modifications and derivative works, including, but not limited to, **distillation for training other LLMs**" | ✅ |
| DeepSeek-V4 (Flash / Pro) | MIT | No additional restrictions | ✅ |
| GLM-5 | MIT | No additional restrictions | ✅ |
| MiMo-V2-Flash | MIT | No additional restrictions | ✅ |
| gpt-oss-20b / 120b | Apache-2.0 | One additional usage-policy sentence: use must obey the applicable law | ✅ |
| Gemma 4 | Apache-2.0 (the license field of the Hugging Face model card) | The full-text page `gemma_4_license` did not open in this environment; the full text is to be verified | To be verified |
| Gemma 1–3 | Gemma Terms of Use | A model trained on synthetic data that Gemma generated is a **"Model Derivative"**. Its distribution must include the use restrictions and the full text of the agreement | ❌ (the restrictions pass to derivatives; not Apache/MIT) |
| Llama 3.1 / 3.2 / 3.3 / 4 | Llama Community License | Allowed, but "a released model that is trained with Llama or its **outputs** must have a name that starts with 'Llama'". It must also show "Built with Llama" and obey the Acceptable Use Policy. More than 700 million monthly active users need a separate license | ❌ (not Apache/MIT, and it has a naming requirement) |
| Llama 2, Meta Llama 3 (2024-04) | Earlier versions of the license above | "You will not use the Llama Materials or any output or results of the Llama Materials to **improve any other large language model**" | ❌ (explicitly forbidden) |

Two common traps:

1. **The student inherits the license of its base model.** DeepSeek-R1 itself is MIT. But the base models of DeepSeek-R1-Distill-Llama-8B / 70B are Llama 3.1 / 3.3, so the Llama license applies to them (the R1 README states this). If you use them as teachers, you indirectly use outputs of Llama.
2. **Licenses change.** Llama 2 forbids the use of its outputs to improve other models. From Llama 3.1, the rule became "allowed, but the name must start with Llama". Each time that you choose a teacher, read the original text again for the **exact version**. Write the license into the metadata of the distillation data (`zero` checks it; see the next section).

## 7. Summary

- **Soft labels carry more information than one-hot labels**: the relative sizes of the wrong answers (dark knowledge). The temperature τ makes them larger. Multiply the loss by τ² to keep the size of the gradient.
- **Logits distillation**: `τ²·KL(p_T^τ ‖ p_S^τ)`, with the gradient `τ·(p_S^τ − p_T^τ)`. It carries the most information, but it **requires the same vocabulary**.
- **Sequence-level distillation**: the teacher writes, and the student does SFT. It is equivalent to the sequence-level forward KL. It requires only that the teacher can generate text. The main-line model has its own tokenizer, so this method is the only one that it can use.
- **Forward KL covers all modes; reverse KL picks one mode.** In **on-policy distillation**, the student samples and the teacher scores each token. Qwen3, Gemma 2, GLM-5, MiMo, and DeepSeek-V4 use it. It also requires the same vocabulary, so the main line uses its own earlier checkpoints as teachers (cross-stage, the GLM-5 form).
- **Rejection sampling + execution check**: sample many times and filter gate by gate. Data quality is more important than quantity. A verifier guarantees only the things that it checks.
- **Teacher license**: Apache-2.0 / MIT is OK. The terms of Gemma 1–3 and of the Llama series pass to the student, so we do not use them.

---

## From minimal code to production code

| Minimal code (`code/`) | Production code (`zero/`, `configs/`, `tests/`) | What it adds, and why |
|---|---|---|
| `kd_loss` in `01` (one position, NumPy) | `zero/post/distill.py`: `kd_loss(student_logits, teacher_logits, mask, temperature, topk)` | Batches of shape (B, T, V). The mean is only over the mask positions (the assistant answer). With `topk > 0`, it uses only the top k tokens of the teacher (this saves storage when you store teacher logits offline). Different shapes raise an error with the hint "the same tokenizer is required" |
| Arms C and D of `02` (`(1−α)·CE + α·KD`) | `DistillTrainer._forward_loss` (`_make_distill_trainer_cls`): it reuses the general `Trainer` and changes only the loss. The teacher is frozen and runs under `no_grad`. `extra_metrics` records `ce` and `kd` | It shares resume from checkpoints, logs, BF16, and multi-GPU training with pretraining / SFT (**multi-GPU training is not yet verified on a GPU**). The config sets `kd_alpha`, `kd_temperature`, and `kd_topk` |
| Arm B of `02` (the teacher writes, the student copies) | `LocalTeacher` (a local zero checkpoint or an HF directory with the Qwen3 structure; it samples with a KV cache) and `OpenAITeacher` (any OpenAI-compatible API; in step 2, vLLM serves the teacher; it uses only the standard library `urllib`). `build_teacher` selects one by `[teacher] backend` | The teacher can have any tokenizer and any architecture, or it can be a remote service. `OpenAITeacher` converts between our tool-call format and the OpenAI `tool_calls` (`to_openai_messages`, `openai_message_to_text`) |
| Reverse KL of `03` | `zero/post/opd.py`: `run_opd`, cross-stage on-policy distillation. Several teachers (our own earlier checkpoints), each with its own prompt pool and weight; `loss = "full_kl"` (exact reverse KL over the vocabulary, only at the response positions) or `"sampled"` (the GLM-5 form, advantage `log p_T − log p_S` per token) | The tokenizer hash of every teacher is checked. The two losses have the same gradient in expectation (`tests/test_opd.py` checks it by enumeration). Data parallel with torchrun. `reverse_kl_loss` / `on_policy_distill` in `distill.py` are the one-teacher version (off by default, `on_policy_steps = 0`) |
| None | `generate_kd_data` with `task_files` (real function-calling tasks of `zero/post/envs/fc_tasks.py`: keep the teacher's first turns that score exact) and `prompt_files` (prompts without answers, e.g. Chinese instructions: drop empty answers, forged turns, tool calls, and answers in another language) | Real data instead of only the toy environment; `prompt_files` is the way to make Chinese SFT data with a teacher whose license allows it. `concurrency` sends parallel requests to the teacher server; the output order does not depend on it |
| Funnel of `04` | `teacher_trajectories` (sample n → `score_tool_calls == 1` → `execute_safely` really runs the tool → the teacher writes the final answer → `score_final_answer`). `generate_kd_data` writes `teacher.jsonl` and `.meta.json` | Each sample and the metadata record **the name, version, and license of the teacher**, the sampling parameters, the number of candidates, the pass rate, and the filter rules. `check_license`: when the teacher is not a model of this project, the config must have `license_allows_distillation = true`, or the run stops |
| Vocabulary check of `05` | `run_distill` compares the hashes of the teacher tokenizer and the student tokenizer **before** it generates data. If they are different, it raises an error and suggests `logits_kd = false` | Do not wait until all teacher data is generated to find out that logits distillation is not possible |
| None | `mix_sft_jsonl` / `mix_sft_max`: mix the original SFT data into the distillation data | When the teacher data is small, this prevents the loss of what the model learned in Chapter 16 |
| None | `run_distill` on N GPUs: first `--generate-only` writes the teacher data in one process; then, with torchrun, rank 0 packs it and every rank trains with the SFT `Trainer` (DDP) | Multi-GPU training of the student; tested with 2 CPU processes (`tests/test_post_ddp.py`), not yet on GPUs |

**Parity checks**:

- Part 4 of `01`: the minimal KD loss and `zero.post.distill.kd_loss` differ by ≤ 3×10⁻⁷ at τ = 1, 2 and top-3 (float32 rounding).
- [`tests/test_distill.py`](../../tests/test_distill.py) (8 tests; 1 more needs CUDA):
  - The KD loss, the temperature, top-1, and the reverse KL agree with hand calculations.
  - Positions outside the mask do not change the loss.
  - Different shapes raise an error.
  - The license guard works.
  - The conversion between OpenAI message formats works in both directions.
  - A **local fake OpenAI server** acts as the teacher. It gives one correct and one wrong answer for each task. The execution check removes the wrong answer, so the pass rate is exactly 50%, and all metadata fields are present.
  - Local self-distillation runs. The student and the teacher start identical, so the KL at the first step is 0. On-policy distillation runs for 1 step.
  - Different tokenizers raise an error.
  - A scripted fake teacher answers real function-calling tasks and prompts: only exact first turns are kept; empty answers, forged turns, and answers in the wrong language are dropped; 1 and 4 parallel requests give the same file.
- [`tests/test_opd.py`](../../tests/test_opd.py) (9 tests): the sampled loss has the gradient of the reverse KL in expectation (exact enumeration); a student that is its own teacher has KL 0; two teachers end to end with both losses; a teacher with another tokenizer is refused.

```bash
uv run pytest tests/test_distill.py tests/test_opd.py -q     # 17 passed, 1 skipped (3.9 s on the build machine)
```

**Main-line config** (`configs/main/distill.toml`; default values for step 2; to be tuned; **not yet verified on a GPU**):

| Setting | Value | Why |
|---|---|---|
| `[teacher] backend` | `openai`, `base_url = http://localhost:8000/v1` | vLLM serves the teacher as an OpenAI-compatible service (`vllm serve <teacher> --enable-auto-tool-choice --tool-call-parser hermes`). The teacher is separate from training |
| `[teacher] name / version / license` | to be decided / to be decided / to be verified; `license_allows_distillation = false` | The run stops when the license is not verified (GOAL.md 3.3) |
| `[distill] logits_kd` | `false` | Our own tokenizer is different from the tokenizer of every open teacher, so only sequence-level distillation is possible |
| `samples_per_task`, `keep_per_task` | 4, 1 | Rejection sampling: 4 samples for each task, keep at most 1 |
| `n_tasks` | 200,000 | `tool_env`: the toy environment, executed and checked end to end (multi-turn with tool results) |
| `task_files`, `prompt_files`, `concurrency` | empty, empty, 64 | Real function-calling tasks and prompts without answers (e.g. Chinese instructions); 64 parallel requests to the vLLM server. Which files: `runs/POSTTRAIN_PLAN.md` 6.2 and 6.4 |
| `mix_sft_jsonl`, `mix_sft_max` | `data/sft/train.jsonl`, 200,000 | Mix the distillation data with SFT data to prevent forgetting |
| `init_from`, `lr`, `max_steps` | the SFT checkpoint, 3e-5, 1500 | A short continued training from the SFT model |
| `on_policy_steps` | 0 | The main line does on-policy distillation as a separate last stage with its own checkpoints as teachers (`configs/main/opd.toml`) |

## Main-line progress

### Tiny-configuration demo (CPU, `configs/tiny`, about 1.3M parameters)

> **Note:** The numbers in this section come from the smoke test **before** the fix of the tool-call scorer (Chapter 19, Section 6). They are the real output of that run. After the fix, we ran the smoke test again (`uv run python -m zero.smoke --out out/smoke_final`). The data, pretraining, mid-training, and SFT stages gave exactly the same results.
>
> The distillation samples that passed verification went from 1 (the nonsense sentence) to 0. The downstream DPO, GRPO, and evaluation numbers changed as a result. For example, the tool-call call_exact of the same SFT model went from 0.133 to 0.100. The model did not change; the scoring became stricter. GRPO is still judged a "tie" with SFT. When you run it yourself, trust your own output.

> This is a **tiny-configuration demo**. It shows only that the code runs from end to end. It does not show any result of the main-line model. The numbers come from the real output of `uv run python -m zero.smoke` (`out/smoke/SUMMARY.md`, `out/smoke/distill/`). We did not run it again for this chapter.

**The teacher is a stand-in.** This environment cannot access huggingface.co and cannot download any open weights. Thus the smoke test uses **the tiny SFT model itself** as the teacher (self-distillation / rejection-sampling fine-tuning). This test verifies only the path "teacher sampling → execution check → packing → sequence-level + logits distillation". It does not show the effect of distillation.

| Item | Value |
|---|---|
| Teacher | `zero-tiny-sft (self-distillation stand-in)`, license "same as this repository" (a model of this project, so `check_license` lets it pass) |
| Tasks × samples | 24 tool_env training tasks × 4 = **96 candidates** |
| Passed the execution check | **1** (pass rate 1.0%), time 15.5 s |
| Mixed-in SFT data | 200 samples → 201 in total, packed into 190 windows, 9,011 target tokens |
| Training | 20 steps, `kd_alpha = 0.5`, τ = 1 |
| Step 1 | loss 0.218 = 0.5 × ce 0.437 + 0.5 × kd **0.000** (the student and the teacher are the same model) |
| Step 20 | loss **0.269** = 0.5 × ce **0.526** + 0.5 × kd **0.0116** |

Notes on these results:

- The tiny model is almost unable to call tools. Only 1 of the 96 candidates passed. This result is as expected.
- **The only sample that passed is the false positive of Section 5.3**: "讲一句鼓励的话。" ("Say something encouraging.") → "坚下云，气温 28°C。" ("<nonsense words>, temperature 28°C."). The scorer at that time did not check the content of the answers on tasks that need no tool. This result reminds us that a real pipeline does hit the holes of a verifier. We fixed this hole later (see 5.3). After the fix, this sample does not pass. The table above is the record of the run before the fix.
- The KD term here is the KL of the student to "the frozen SFT model". It starts from 0 and slowly increases. It acts as a constraint: "do not move too far from the original model". It is not learning from a stronger teacher.

### To be added after GPU training

- The final choice of the teacher, its version, and the record of the license check.
- Real teacher data generation: the number of candidates, the pass rate of each gate, the pass rate by task type, and the cost. The tasks that need no tool are a separate item.
- Decontamination: a 13-gram check of the distillation data against all evaluation sets. Also, a check of the overlap of function names and schemas with evaluation sets such as BFCL (GOAL.md 3.2).
- The distillation training curves, the tool-call scores on our own development set before and after distillation, and the failures and reruns.

**The teacher plan for step 2 (to be decided)**:

1. Choose teachers from open-weight models with an Apache-2.0 or MIT license. Candidates, with the license read from the model metadata on 2026-10-10: Qwen3.5-35B-A3B and Qwen3.5-122B-A10B (Apache-2.0), Qwen3-235B-A22B-Instruct-2507 (Apache-2.0), gpt-oss-120b (Apache-2.0; OpenAI also has a usage policy), DeepSeek-V4.1-Flash (MIT), GLM-5.2 (MIT, Chinese and English). Choose by tool-call ability, Chinese ability, and inference cost; **the project lead decides the exact models**. Several teachers can each produce data. Each sample records the teacher that it came from.
2. Use vLLM to run an OpenAI-compatible service on a separate GPU. In `[teacher]` of `configs/main/distill.toml`, fill in the name, version, and license, and set `license_allows_distillation = true`.
3. Generate data on real tasks (`task_files`: first turns that score exact) and on Chinese prompts (`prompt_files`). Multi-step trajectories on real tools need executable tools; they are still only in the toy environment. Add a content scorer for the tasks that need no tool.
4. Do sequence-level distillation (`logits_kd = false`) + mixed-in SFT data.
5. Count the cost of teacher data generation and distillation training in the post-training budget of Chapters 16–19 (GOAL.md 3.4: about $1,500, including teacher data generation). Get approval first for a run that is expected to cost more than $100.

---

## Frontier notes

> **Pruning + distillation**: first make a large model smaller (remove layers, remove attention heads, make the FFN narrower). Then use the original model as the teacher, and recover the quality with distillation. Llama 3.2 1B and 3B were made in this way: "pruned from 8B, then distilled with the logits of 8B and 70B". Minitron from NVIDIA (the Nemotron family) studies this path systematically. We verified only two leading families that clearly use this method. Also, the main line trains from scratch. So this chapter does not cover it.
>
> **Cross-tokenizer distillation**: logits / on-policy distillation when the teacher and the student use different tokenizers. (For example, align the probabilities of the two sides by byte prefixes.) If this method becomes mature, the main line can do on-policy distillation from a Qwen teacher without a change of vocabulary. At this time, only research papers exist (for example arXiv 2607.22334). No leading family uses it in a main release.
>
> **Multi-teacher on-policy distillation to merge domain experts**: MiMo-V2-Flash and DeepSeek-V4 first train a set of domain experts (math, code, agents, ...). Then on-policy distillation merges them into one student. GLM-5 uses it to recover abilities that the model forgot after several RL stages. This method is one use of on-policy distillation (in the main text). The main line uses the GLM-5 cross-stage form (its own earlier checkpoints as teachers, `zero/post/opd.py`); the budget allows no set of domain experts, so it does not merge experts.

---

## Adopters and sources

| Technique | Adopters (main releases) | Sources |
|---|---|---|
| Logits distillation (soft labels at each token) | **Llama 3.2** 1B/3B (the logits of Llama 3.1 8B and 70B are the targets at each token); **Gemma 2** 2B/9B (pretraining with distillation instead of next-token prediction); all **Gemma 3** models (256 logits sampled by teacher probability at each token); **Qwen3** small models (match the teacher logits in the on-policy stage); **DeepSeek-V4** (on-policy distillation with full-vocabulary logits) | [Llama 3.2 model card](https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/MODEL_CARD.md), [Meta blog](https://ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices/) ("structured pruning in a single shot manner from the Llama 3.1 8B"); [Gemma 2](https://arxiv.org/abs/2408.00118) §3.2, §5; [Gemma 3](https://arxiv.org/abs/2503.19786) §2.2; [Qwen3](https://arxiv.org/abs/2505.09388) §4, §4.5; [DeepSeek-V4](https://arxiv.org/abs/2606.19348) §5.1.2 |
| Sequence-level distillation (SFT on teacher data) | **DeepSeek-R1-Distill** (about 800,000 R1 samples, SFT only); **Qwen3** small models (offline distillation stage); **Gemma 2** (the SFT answers are "mainly synthesized by a larger teacher") | [DeepSeek-R1](https://arxiv.org/abs/2501.12948) Appendix B.3.3, B.4.3, F; [Qwen3](https://arxiv.org/abs/2505.09388) §4.5; [Gemma 2](https://arxiv.org/abs/2408.00118) §4 |
| Rejection sampling / keep after verification | **DeepSeek-R1** (sample several, keep only the correct ones); **Qwen3** (QwQ-32B generates N candidates, six types of filters); **GLM-5** (rejection sampling + executable environments); **Llama 3.2** (each round: SFT → rejection sampling → DPO) | [DeepSeek-R1](https://arxiv.org/abs/2501.12948) B.3.3; [Qwen3](https://arxiv.org/abs/2505.09388) §4.1; [GLM-5](https://arxiv.org/abs/2602.15763) §3.1; [Llama 3.2 model card](https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/MODEL_CARD.md) |
| On-policy distillation | **Qwen3** (§4.5, Table 21); **Gemma 2** ("distillation from the teacher on the student's distribution", §4); **GLM-5** (on-policy cross-stage distillation, §3.5); **MiMo-V2-Flash** (MOPD, §4.1, §4.4); **DeepSeek-V4** (OPD replaces mixed RL, §5.1) | [Qwen3](https://arxiv.org/abs/2505.09388); [Gemma 2](https://arxiv.org/abs/2408.00118); [GLM-5](https://arxiv.org/abs/2602.15763); [MiMo-V2-Flash](https://arxiv.org/abs/2601.02780); [DeepSeek-V4](https://arxiv.org/abs/2606.19348); methods: [GKD](https://arxiv.org/abs/2306.13649), [MiniLLM](https://arxiv.org/abs/2306.08543), [Thinking Machines blog](https://thinkingmachines.ai/blog/on-policy-distillation/) |
| Teacher data generation service (vLLM, OpenAI-compatible API) | Industry-standard tool (GOAL.md 2.1 rule B) | [vLLM](https://github.com/vllm-project/vllm) |

**Consensus decision (GOAL.md 2.1)**: we checked four techniques: logits distillation, sequence-level distillation, rejection sampling, and on-policy distillation. For each technique, 3 or more independent leading families clearly use it in their main releases. Thus all four go into the main text. The GOAL table lists "on-policy distillation (to be verified)". Our check shows that it **has reached consensus**. Pruning + distillation and cross-tokenizer distillation are in "Frontier notes".

**License sources** (read in 2026-09):
[Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B), [Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B/blob/main/LICENSE), [Qwen3.5-397B-A17B](https://huggingface.co/Qwen/Qwen3.5-397B-A17B),
[DeepSeek-R1 README §7](https://github.com/deepseek-ai/DeepSeek-R1#7-license), [DeepSeek-V4-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash),
[GLM-5](https://huggingface.co/zai-org/GLM-5), [MiMo-V2-Flash](https://huggingface.co/XiaomiMiMo/MiMo-V2-Flash),
[gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b) (and its `USAGE_POLICY`), [Gemma 4 E2B](https://huggingface.co/google/gemma-4-E2B-it),
[Gemma Terms of Use](https://ai.google.dev/gemma/terms) (1.1(e) "Model Derivatives"),
[Llama 3.2 License](https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/LICENSE), [Llama 4 License](https://github.com/meta-llama/llama-models/blob/main/models/llama4/LICENSE),
[Meta Llama 3 License](https://github.com/meta-llama/llama-models/blob/main/models/llama3/LICENSE), [Llama 2 License](https://github.com/meta-llama/llama-models/blob/main/models/llama2/LICENSE).

---

## Guided questions

Ask Claude Code these questions. Continue until you can explain the answers in your own words:

1. In `02`, the loss of C (logits distillation) on its own 20,000 characters (2.867) is higher than the loss of A (2.678). But C is much better on the validation set. What does the soft label do here? Suppose that the training data of the student increases from 20,000 to 1,000,000 characters. How do you expect the gap between A and C to change?
2. Sequence-level distillation is equivalent to the sequence-level forward KL. Then why do we say that it "carries the information of only one path"? Suppose that the teacher samples 16 answers for each prompt and we train on all of them. Does the method come closer to logits distillation? What is the cost?
3. For a language model, what does the problem of "mode covering" look like in practice? Give a tool-call example: a call has two correct forms. If the student "learns half of each", what does it write? What is the risk of the reverse KL (hint: diversity)?
4. The "reward" of on-policy distillation is `log p_T − log p_S`. Why is it difficult to hack? If the teacher itself is wrong on one type of question, what happens to the student? (Compare with reward hacking in Chapter 19.)
5. In the weekday task of Section 5.2, one call had a date that was wrong by 7 days, but the weekday was correct. This call passed the execution check. How would you change the verifier? If the verifier becomes too strict, what correct samples would it reject?
6. Suppose that you choose the teacher for step 2. One option is a medium model with an Apache-2.0 license. The other option calls tools better, but its license requires that "a derivative model has a name that starts with its name". Which do you choose? What information must you record so that other people can check this decision later?

## Hands-on tasks

For each task, run the code and look at the result.

**Task 1 (basic)**: In `01_soft_labels.py`, change the teacher logits to an "extremely confident" distribution (for example `[10, 1, 0, 0, -5, -5]`). Look again at the entropy and `p(晴)/p(猫)` at τ = 1, 2, 4. At what τ do the "second-best answers" start to get a visible probability?

**Task 2 (core)**: In `02_toy_distill.py`, do two sweeps. (a) Change `TAU` to 1 and to 4, and see how the validation result of arm C changes. (b) Change `SMALL_N` to 5,000 and to 100,000, and see how the gap between A and C changes. Write a one-sentence conclusion: "The ____ the student data, the ____ the benefit of distillation." (Each run takes about 2 min of CPU time.)

**Task 3 (challenge)**: Write a content scorer for the tasks of `tool_env` that need no tool. For example: the answer must not contain numbers or city names, and its length must be 2–40 characters. Or: its character overlap with `gold_answer` must be above a threshold. In `04_rejection_sampling.py`, use it instead of `score_final_answer`. How many of the 9 nonsense answers are left? Does the scorer reject normal answers? Then think: with a real teacher, would this rule reject many reasonable greetings?

---

## Go deeper: CS336

This chapter maps to Stanford CS336 (Spring 2026) <https://cs336.stanford.edu/>, with **partial coverage**:

- **Lecture 14: Data (filtering, deduplication, mixing, synthetic data)**. The synthetic-data part is the background of this chapter: "let the teacher write data, filter it, then use it". You can apply the filter pipeline of Assignment 4 (Data) directly to teacher data.
- **Lecture 15: Mid-training and post-training (SFT/RLHF)**. It discusses where SFT data comes from. Distillation from teacher data is the most common type.
- CS336 does not go deep into the temperature and the gradient of logits distillation, forward / reverse KL, on-policy distillation, or teacher licenses. See the references of this chapter.

---

## References

- Hinton, Vinyals, Dean. *Distilling the Knowledge in a Neural Network* (soft labels, temperature, τ² scaling), 2015: <https://arxiv.org/abs/1503.02531>
- Kim, Rush. *Sequence-Level Knowledge Distillation*, 2016: <https://arxiv.org/abs/1606.07947>
- Agarwal et al. *On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes* (GKD), 2023: <https://arxiv.org/abs/2306.13649>
- Gu et al. *MiniLLM: Knowledge Distillation of Large Language Models* (reverse KL), 2023: <https://arxiv.org/abs/2306.08543>
- Lu, Thinking Machines Lab. *On-Policy Distillation* (blog), 2025-10: <https://thinkingmachines.ai/blog/on-policy-distillation/>
- Gemma Team. *Gemma 2: Improving Open Language Models at a Practical Size* (§3.2 distillation, §5 ablations), 2024: <https://arxiv.org/abs/2408.00118>
- Gemma Team. *Gemma 3 Technical Report* (§2.2 sampling of 256 logits, §3 post-training), 2025: <https://arxiv.org/abs/2503.19786>
- Meta. *Llama 3.2 model card* (pruning + logits distillation): <https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/MODEL_CARD.md>
- Meta. *Llama 3.2: Revolutionizing edge AI and vision with open, customizable models* (blog, section "Lightweight models"), 2024-09: <https://ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices/>
- Qwen Team. *Qwen3 Technical Report* (§4.5 strong-to-weak distillation, Table 21), 2025: <https://arxiv.org/abs/2505.09388>
- DeepSeek-AI. *DeepSeek-R1* (Appendix B.3.3 rejection sampling, B.4.3 and F distillation), 2025: <https://arxiv.org/abs/2501.12948>
- GLM-5 Team. *GLM-5: from Vibe Coding to Agentic Engineering* (§3.5 on-policy cross-stage distillation), 2026: <https://arxiv.org/abs/2602.15763>
- Xiaomi LLM-Core. *MiMo-V2-Flash Technical Report* (§4 MOPD), 2026: <https://arxiv.org/abs/2601.02780>
- DeepSeek-AI. *DeepSeek-V4* (§5.1 expert training + on-policy distillation), 2026: <https://arxiv.org/abs/2606.19348>
- Muralidharan et al. *Compact Language Models via Pruning and Knowledge Distillation* (Minitron), 2024: <https://arxiv.org/abs/2407.14679>
- The original license texts of each teacher: see the end of "Adopters and sources".
- [CS336](https://cs336.stanford.edu/) Lectures 14 and 15

**Next chapter**: Distillation and SFT teach the model "how to answer". But they give only positive examples: the model never sees "why this answer is better than that one". Chapter 18 covers preference alignment. First, we build the framework with RLHF (a reward model + PPO). Then, from the same objective, we derive the simpler DPO. DPO helps the main-line model to behave better in general conversation.
