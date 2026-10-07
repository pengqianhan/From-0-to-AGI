# Chapter 18: Preference alignment — From RLHF to DPO

**English** · [中文](README.zh.md)

> **Goal**: After this chapter, you can start from "preference data + the Bradley–Terry model" and write the loss of a reward model. You can explain why the KL "leash" in the RLHF objective `E[r] − β·KL` is necessary. You can derive the DPO loss from the optimal solution of RLHF by hand, and do a parity check of your code against `zero/post/dpo.py`. You also know how to watch for the pitfalls of DPO: the learning rate, β, and "the probability of chosen also goes down".

📺 **Video**: Not published yet. To render it on your computer, run `bash chapters/18-preference-alignment/video/build.sh`.
🧪 **Self-check**: After the chapter, type `/ch18-dpo` in Claude Code.

---

In the previous chapter, we used distillation to put the tool-calling skill of a teacher into a small model. First, SFT taught the format. Then the model learned from demonstrations that the teacher wrote and that passed an execution check. Both steps are **imitation**: the model writes what the demonstrations show.

This chapter solves a different problem. **Often, we cannot write "the best answer", but we can tell "which of two answers is better" with much less work.** How do we change this judgment into a training signal?

The classic answer is **RLHF (Reinforcement Learning from Human Feedback)**. First, train a reward model that gives each answer a score. Then use PPO to make the model chase high scores. InstructGPT used this method.

This chapter first explains RLHF, because it prepares you for GRPO in Chapter 19 (rule C in GOAL.md section 2.1). Then, from **the same objective**, we derive the preference-alignment method that open models use most today: **DPO (Direct Preference Optimization)**. DPO is the 6th training stage of the main-line model.

The code of this chapter (all scripts run on the CPU, with `torch.set_num_threads(1)`):

```bash
uv run python chapters/18-preference-alignment/code/01_bradley_terry.py    # Bradley–Terry reward model (a few seconds)
uv run python chapters/18-preference-alignment/code/02_rlhf_kl.py          # RLHF objective, KL leash, PPO (a few seconds)
uv run python chapters/18-preference-alignment/code/03_dpo_derivation.py   # check each step of the DPO derivation + parity check with zero (a few seconds)
uv run python chapters/18-preference-alignment/code/04_toy_dpo.py          # DPO on a small language model, sweep β (about 30 s of CPU time)
uv run python chapters/18-preference-alignment/code/05_dpo_pitfalls.py     # learning rate, chosen probability goes down, overfitting (about 1 min of CPU time)
```

## 1. Intuition: when imitation reaches its limit, teach the model "which one is better"

The SFT loss "maximizes the probability of the demonstration answer". This loss has two limits:

1. **The model can only be as good as the demonstrations.** Ask a labeler to write a good poem, or an explanation of some code with no gaps. This is expensive and slow, and the result is not always the best.
2. **The model also learns the errors in the demonstrations.** In the small experiment of section 7 (`04`), only 40% of the SFT data is correct. The model faithfully learns to "answer wrong 60% of the time".

But **judging** is much less work than **creating**. Anyone can read two poems and say which one is better. This gives us **preference data**: for the same prompt x, there are two answers, and a judge picks one. The picked answer is **chosen** (written y_w, w = win). The other answer is **rejected** (y_l, l = lose).

```
{"prompt": "用一句话解释什么是梯度",
 "chosen":   "函数值上升最快的方向，大小是那个方向的坡度。",
 "rejected": "梯度就是梯度的意思。"}
```

(In English: the prompt is "Explain in one sentence what a gradient is". Chosen: "The direction in which the function value increases fastest; its size is the slope in that direction." Rejected: "A gradient means a gradient.")

The judge can be a human (**human feedback**, the main data of InstructGPT and Llama 3). The judge can also be a stronger model (**AI feedback**). Tülu 3 and OLMo 2 use GPT-4o to rate four answers on a scale of 1–5, and the answer with the highest score becomes chosen. Nemotron-4 uses its own reward model for the scores.

The judge can also be **a program that checks correctness automatically**. The offline DPO of Qwen2.5 uses "execution feedback and answer matching" to separate correct and wrong answers. The main-line model uses a tool-calling environment for the scores (see "From minimal code to production code").

## 2. Bradley–Terry: change "who won" into a probability

Preference data only says "a is better than b". It does not give scores. The **Bradley–Terry model** assumes that each answer has a hidden score r. The difference of the scores sets the probability that a wins:

```
P(a ≻ b) = σ(r(a) − r(b)),     σ(z) = 1 / (1 + e^(−z))
```

Part ① of `01_bradley_terry.py` prints some values of σ:

| r(a) − r(b) | −4 | −2 | −1 | 0 | +1 | +2 | +4 |
|---|---:|---:|---:|---:|---:|---:|---:|
| P(a ≻ b) | 0.02 | 0.12 | 0.27 | 0.50 | 0.73 | 0.88 | 0.98 |

We use two properties later. When the two scores are equal, each answer wins with probability 0.5. **Only the difference goes into the formula**: add 100 to the scores of all answers, and no probability changes.

## 3. Reward model: a binary classification problem

A **reward model (RM)** is a network that gives one scalar score to a (prompt, answer) pair. Usually, it is the SFT model without its last layer (the output layer over the vocabulary). A linear head that outputs 1 number replaces that layer. To train the reward model, maximize the log-likelihood that "chosen wins":

```
L_RM = −log σ( r(x, y_w) − r(x, y_l) )
```

This is the **binary cross-entropy** of Chapter 5. The logit is the difference of the two scores, and the label is always 1 ("chosen wins"). The reward-model loss of InstructGPT (equation (1) in the paper) is this formula. The code is one line:

```python
def bt_loss(r_chosen, r_rejected):
    return -F.logsigmoid(r_chosen - r_rejected).mean()     # −log σ(r_w − r_l)
```

**Toy experiment**: two features describe each answer: the quality q and the length ℓ (in 100 tokens). The labeler scores each answer with the hidden formula `1.5·q + 0.4·ℓ`, so **the labeler prefers long answers a little**. The labeler picks the winner with the Bradley–Terry probability. We only see 4000 pairs of "who won". We train a linear reward model `r = w_q·q + w_len·ℓ + c` (the weights start at 0):

| Step | Loss | w_q | w_len |
|---:|---:|---:|---:|
| 0 | 0.6931 | 0.000 | 0.000 |
| 10 | 0.4610 | 0.932 | 0.358 |
| 50 | 0.4404 | 1.609 | 0.482 |
| 200 | 0.4397 | 1.493 | 0.449 |
| 600 | 0.4397 | 1.493 | 0.449 |

- **The initial loss is 0.6931 = ln 2.** All answers have the same score, so each answer of a pair wins with probability 0.5. This is the same check point as in Chapter 5: "a model that learned nothing has loss ≈ ln(number of classes)".
- **The model learned the coefficients back**: 1.493 / 0.449, against the true values 1.5 / 0.4.
- **The bias c does not move at all** (its change during training is 0.0). The constant cancels in `r_w − r_l`, so its gradient is always 0. Only the relative values of the reward-model scores have a meaning. Before RL, InstructGPT added a bias that sets the mean score of the demonstration answers to zero.
- On the held-out set, we count the pairs where the score of chosen is higher. The fraction is 0.803 for the reward model and 0.805 for the labeler's own score. **The upper limit is not 1**, because the labels have noise. InstructGPT reports about 72.6% agreement between its training labelers.
- `−log σ(r_w − r_l)` and PyTorch's `binary_cross_entropy_with_logits(r_w − r_l, 1)` both give 0.428393.

**The most important point**: the reward model also learned "long" as a good property (w_len ≈ 0.45). It faithfully learned the preference of the labeler, **and also the bias**. A preference for length is common in real human and model preference data (Singhal et al. 2023).

## 4. RLHF: chase a high score, but on a leash

### 4.1 The objective

With a reward model, the next step is to make the policy π generate answers with high rewards. The policy is the language model that we train. But **we must not let the policy chase the score freely**.

The reward model is reliable only near the types of answers that it saw. When the policy moves to answers that the reward model never saw, it can find answers that "get a high reward-model score but are bad". Thus, the objective subtracts a KL divergence term. This term keeps the policy near the SFT model π_ref:

```
max_π   E_{y~π}[ r(x, y) ]  −  β · KL( π(·|x) ‖ π_ref(·|x) )
```

The KL term is a **leash**, and β sets how tight the leash is. InstructGPT (equation (2) in the paper) adds the KL penalty at each token, with β = 0.02. It also mixes in pretraining gradients (PPO-ptx). These gradients reduce the drop in scores on public NLP benchmarks (the "alignment tax").

### 4.2 What happens when the leash is too loose: reward hacking

`02_rlhf_kl.py` reduces the "answers" to 8 candidates (a multi-armed bandit). We know the true quality q and the length ℓ of each candidate. The reward model scores them with the coefficients from the previous section. The names are short: "concise" and "detailed" are the two correct answers, and "padded 900" is a 900-token answer full of filler.

| Answer | True quality q | Length ℓ | RM score | π_ref |
|---|---:|---:|---:|---:|
| concise | 1.4 | 1.2 | 2.05 | 0.149 |
| detailed | 1.2 | 2.5 | 2.33 | 0.122 |
| okay | 0.6 | 1.0 | 0.76 | 0.245 |
| mediocre | 0.2 | 1.5 | 0.39 | 0.245 |
| off-topic | −0.8 | 1.0 | −1.33 | 0.090 |
| wrong | −1.5 | 0.8 | −2.46 | 0.090 |
| verbose | 0.0 | 4.0 | 1.21 | 0.055 |
| padded 900 | −0.5 | 9.0 | **2.71** | 0.004 |

The favorite of the reward model is "padded 900". During reward-model training, the longest answer had only 400 tokens. The reward model never saw a 900-token answer, so it extrapolates "longer is better". The next section proves that the optimal policy of this objective has a closed-form solution, `π* ∝ π_ref·exp(r/β)`. First, look at the results:

| β | E[RM score] | E[true quality] | KL(π‖π_ref) | Most likely answer |
|---:|---:|---:|---:|---|
| 100 | 0.626 | 0.352 | 0.000 | okay (0.25) |
| 2 | 1.325 | 0.749 | 0.161 | concise (0.25) |
| 1 | 1.727 | 0.962 | 0.454 | detailed (0.35) |
| **0.5** | 2.105 | **1.124** | 0.988 | detailed (0.51) |
| 0.25 | 2.293 | 1.061 | 1.501 | detailed (0.64) |
| 0.1 | 2.554 | 0.174 | 3.336 | padded 900 (0.61) |
| 0.03 | 2.711 | −0.500 | 5.405 | padded 900 (1.00) |

As β becomes smaller, the reward-model score **increases all the time**. But the true quality **first increases and then decreases**. β = 0.5 gives the best result. At β = 0.03, the policy puts all of its probability on the padded answer, and the true quality is worse than SFT.

This effect is **reward hacking**. Gao et al. 2022 measured this curve systematically on real models: "the more you optimize the proxy score, the true score first increases and then decreases". The KL leash is there to prevent reward hacking. β must be large enough to keep the policy in the range where the reward model is reliable. β must also be small enough to let the policy really improve.

### 4.3 PPO: solve it step by step with sampling (preview)

A real language model cannot list all possible answers. It can only **sample**. InstructGPT uses **PPO (Proximal Policy Optimization)** to solve this objective. PPO has only four key points (Chapter 19 gives the details):

1. Sample a batch of answers with the current policy. Reward = `r(y) − β·(log π_old(y) − log π_ref(y))` (a KL penalty for each sample).
2. **Value baseline**: advantage A = reward − V. V is a learned estimate of "the average score". It reduces the variance.
3. **Importance ratio** ρ = π_θ(y)/π_old(y): one batch of samples can give several updates.
4. **Clipping**: `min(ρ·A, clip(ρ, 1−ε, 1+ε)·A)` prevents a step that goes too far.

Part ② of `02` runs such a PPO on the bandit (64 samples per iteration, 4 epochs, ε = 0.2). PPO starts from π_ref:

| Iteration | Objective E[r] − β·KL | KL(π‖π_ref) |
|---:|---:|---:|
| 0 | 0.6071 | 0.0000 |
| 25 | 1.5921 | 0.9437 |
| 100 | 1.6084 | 0.9744 |
| 400 | 1.6111 | 0.9855 |

The objective value of the closed-form optimal solution is 1.6112. After 400 iterations, the KL between the PPO policy and the closed-form solution is only 0.00021. PPO can solve the problem. But it needs four things that run together: **sampling, a reward model, a value model, and a reference model**. To train a 7B model, you must keep four large models in memory at the same time. Can we skip these?

**InstructGPT is the classic example**: three steps, SFT → reward model → PPO. In human evaluation, the 1.3B-parameter InstructGPT was better than the 175B-parameter GPT-3. Llama 2 also uses "reward model + rejection sampling + PPO".

## 5. Derive DPO from the same objective

The starting point of DPO (Rafailov et al. 2023) is this: **we can write the optimal solution of the objective above directly**. The derivation has only four steps. `03_dpo_derivation.py` checks each step with numbers.

**Step 1: the optimal policy.** Expand the objective (for a fixed x, which we do not write):

```
E_π[r] − β·KL(π‖π_ref)
  = Σ_y π(y)·r(y) − β·Σ_y π(y)·log(π(y)/π_ref(y))
  = −β·Σ_y π(y)·[ log π(y) − log π_ref(y) − r(y)/β ]
  = −β·Σ_y π(y)·[ log π(y) − log( π_ref(y)·e^{r(y)/β} / Z ) − log Z ]
  = β·log Z − β·KL(π ‖ π*),         where  π*(y) = π_ref(y)·e^{r(y)/β} / Z,   Z = Σ_y π_ref(y)·e^{r(y)/β}
```

The first term does not depend on π. The second term is KL ≥ 0, and it is 0 only when π = π*. Thus, **π* is the optimal solution of the RLHF objective**: take the probability of the reference model, multiply it by the exponential of the reward, and normalize. The smaller β is, the more π* amplifies the answers with high rewards. The table in section 4.2 shows this effect. (Part (1) of `03`: for 5 random π, the largest difference between the two sides of the equation is 2.2×10⁻¹⁶.)

**Step 2: solve for the reward.** Take the log of the formula for π* and move the terms:

```
r(y) = β·log( π*(y) / π_ref(y) ) + β·log Z
```

Reward = β × "the log-probability ratio of the optimal policy to the reference model" + a constant that does not depend on y. (Part (2) of `03`: the range of r − β·log(π*/π_ref) over 6 answers is 3.3×10⁻¹⁶. It is the same constant β·log Z.)

**Step 3: put the reward into Bradley–Terry, and Z cancels.** Z is a sum over all possible answers, so we cannot calculate it. But Bradley–Terry uses only the **difference** of the rewards:

```
r(y_w) − r(y_l) = β·log( π*(y_w)/π_ref(y_w) ) − β·log( π*(y_l)/π_ref(y_l) )     (the two β·log Z terms cancel)
```

(Part (3) of `03`: the largest difference between the implicit reward difference and the true reward difference is 4.4×10⁻¹⁶.)

**Step 4: replace π* with the policy π_θ, and do maximum likelihood.** In the reward-model loss `−log σ(r_w − r_l)`, replace the rewards with the expression above:

```
L_DPO = −log σ( β·[ (log π_θ(y_w) − log π_ref(y_w)) − (log π_θ(y_l) − log π_ref(y_l)) ] )
```

With this step, **the reward model disappears**, and sampling is not necessary. You need only a preference data set, a frozen reference model, and a classification-type loss. The subtitle of the DPO paper says it well: *Your Language Model is Secretly a Reward Model*. `β·log(π_θ/π_ref)` is the **implicit reward** that the model contains.

```python
def dpo_loss(pi_w, pi_l, ref_w, ref_l, beta):          # inputs: sequence log-probability of each answer, (B,)
    h = beta * ((pi_w - ref_w) - (pi_l - ref_l))       # difference of the implicit rewards
    return -F.logsigmoid(h).mean()                     # −log σ(h)
```

Here, `log π(y|x)` is **the sum of the log-probabilities of each token in the answer** (the prompt does not count). This is the same idea as the loss mask of SFT in Chapter 16.

**Parity check**: part ⑤ of `03` compares this loss from scratch with `dpo_loss` in `zero/post/dpo.py` on 16 pairs of random log-probabilities. The loss difference is 1.2×10⁻⁷ (float32 rounding), and the largest gradient difference is 4.7×10⁻¹⁰.

**Does DPO really reach the optimal solution of RLHF?** Part ⑦ of `03` goes back to the bandit of section 4.2. It labels 17427 preference pairs with Bradley–Terry(r). Then it trains DPO **with these preference pairs only** (no reward model, no sampling), with β = 0.5:

| Answer | π_ref | π_DPO | π* (RLHF optimal solution) |
|---|---:|---:|---:|
| concise | 0.149 | 0.362 | 0.355 |
| detailed | 0.122 | 0.493 | 0.514 |
| okay | 0.245 | 0.053 | 0.045 |
| mediocre | 0.245 | 0.026 | 0.021 |
| off-topic | 0.090 | 0.000 | 0.000 |
| wrong | 0.090 | 0.000 | 0.000 |
| verbose | 0.055 | 0.025 | 0.025 |
| padded 900 | 0.004 | 0.040 | 0.041 |

KL(π_DPO‖π*) = 0.0017 (at the start, KL(π_ref‖π*) = 2.0082). One objective, two paths, the same destination.

## 6. What the gradient of DPO does

Differentiate the loss through h:

```
∂L/∂log π_θ(y_w) = −β·σ(−h),      ∂L/∂log π_θ(y_l) = +β·σ(−h)
```

Gradient descent subtracts the gradient. Thus: **DPO pushes chosen up and rejected down, with the same strength β·σ(−h)**. σ(−h) is "the probability that the implicit reward ranks this pair in the wrong order". Part ⑥ of `03` (β = 0.1):

| h | ∂L/∂log π(y_w) | −β·σ(−h) |
|---:|---:|---:|
| −4 | −0.09820 | −0.09820 |
| −2 | −0.08808 | −0.08808 |
| 0 | −0.05000 | −0.05000 |
| +2 | −0.01192 | −0.01192 |
| +4 | −0.00180 | −0.00180 |

The worse the ranking error (the more negative h is), the harder the push. A pair that is already in the correct order by a large margin gets almost no push. The cross-entropy of Chapter 5 does the same: "the size of the push follows the size of the remaining error".

**The role of β**: in the derivation, β sets how tight the KL leash of RLHF is. In the loss, β converts the difference of log ratios into a "reward". With a small β, the same difference of log ratios gives only a small reward, and σ does not saturate soon. Thus, the gradient pushes the model farther from π_ref.

## 7. Run DPO on a small language model

`04_toy_dpo.py`: the prompt is `a+b=` (a, b ∈ 0–9). The answer is the sum, and it ends with `;`. The model is a character-level GRU language model with about 20k parameters:

1. **SFT** (the reference model): the demonstrations have **mixed quality**. For the same prompt, 40% of the demonstrations are correct, and 60% are a random wrong number in 0–18.
2. **Preference data**: 70 prompts × 4 pairs = 280 pairs. Chosen = the correct answer. Rejected = a wrong answer that the reference model can write. The other 30 prompts are held out, and training never sees them.
3. **DPO**: β = 0.1, lr = 1e-3, Adam, 32 pairs per step, 150 steps. We calculate the log-probabilities of the reference model before training.

> **Note:** The numbers of the training experiments in this chapter come from one CPU run on the course build machine. Different machines and different versions of the math libraries do floating-point operations in a slightly different order. After a few hundred training steps, these small differences become larger. Your numbers can differ from the second or third decimal place. Trust the conclusions below that do not depend on exact values. For a 2026-10 rerun on a different server, see [runs/2026-10-01-gpu0-check/chapters-16-20.md](../../runs/2026-10-01-gpu0-check/chapters-16-20.md).

The SFT reference model on the 30 held-out prompts: the probability of the correct answer is 0.369, the fraction of well-formed samples is 1.000, and the fraction of correct samples is 0.360. During DPO training (on the training set):

| Step | Loss | margin | acc | log π(chosen) | log π(rejected) |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.6931 | +0.000 | 0.00 | −1.043 | −3.396 |
| 25 | 0.6512 | +0.087 | 0.93 | −0.662 | −3.881 |
| 50 | 0.6144 | +0.167 | 0.94 | −0.504 | −4.527 |
| 75 | 0.5751 | +0.260 | 0.94 | −0.466 | −5.419 |
| 100 | 0.5353 | +0.362 | 0.94 | −0.482 | −6.455 |
| 125 | 0.4934 | +0.480 | 0.94 | −0.520 | −7.677 |
| 150 | 0.4553 | +0.601 | 0.95 | −0.551 | −8.919 |

(margin = the mean difference of the implicit rewards. acc = the fraction of pairs where the implicit reward ranks chosen first. These names are the same as the log fields of `zero`. At step 0, h is 0 everywhere, so acc is 0 and the loss is exactly ln 2.)

On the held-out prompts, **the probability of the correct answer goes from 0.369 to 0.430**. Correct samples go from 0.360 to 0.413, and well-formed samples from 1.000 to 0.995. The implicit reward ranks 0.83 of the pairs correctly.

Look at the shapes of the two curves. **Rejected goes down all the time** (−3.4 → −8.9). Chosen first goes up and then goes down a little (−1.04 → −0.47 → −0.55). DPO only cares about the **difference** of the two. It never asks directly that chosen itself goes up. In section 8, this property becomes a pitfall.

**Sweep β** (lr = 1e-3, 150 steps). margin/β is the difference of the log ratios. It measures how far the policy moved from π_ref:

| β | Final loss | margin | margin/β | Δlog π(chosen) | Δlog π(rejected) | Held-out P(correct) |
|---:|---:|---:|---:|---:|---:|---:|
| 0.03 | 0.5997 | +0.204 | +6.80 | +0.255 | −6.540 | 0.388 |
| 0.1 | 0.4553 | +0.601 | +6.01 | +0.492 | −5.523 | 0.430 |
| 0.3 | 0.2601 | +1.540 | +5.13 | +0.722 | −4.411 | 0.529 |
| 1.0 | 0.1030 | +2.724 | +2.72 | +0.739 | −1.984 | 0.576 |

The smaller β is, the larger the difference of the log ratios (6.80 vs 2.72). The policy moves farther from the reference model. But almost all of the extra "distance" goes into pushing rejected down, and chosen goes up less. In this toy, in these 150 steps, a larger β gave better held-out results. Do not generalize this result to "a larger β is always better". Here, π_ref is bad (only 40% correct), so the leash has little value.

In real models, a common value of β is about 0.1 (the original DPO paper, Zephyr, and Llama 3 all use 0.1). For length-normalized DPO, β has a different scale (Tülu 3 uses 5). In all cases, sweep β on a development set.

## 8. Pitfalls of DPO

`05_dpo_pitfalls.py` uses the same small model as above.

**Pitfall 1: the learning rate is too large.** (β = 0.1, 150 steps; the first row is the SFT reference model itself)

| lr | margin | Train acc | Held-out P(correct) | Well-formed samples | Correct samples |
|---:|---:|---:|---:|---:|---:|
| SFT | — | — | 0.369 | 1.000 | 0.360 |
| 1e-4 | +0.05 | 0.95 | 0.432 | 1.000 | 0.447 |
| 1e-3 | +0.60 | 0.95 | 0.430 | 0.995 | 0.413 |
| 1e-2 | +4.26 | 1.00 | 0.071 | 0.473 | 0.063 |
| 5e-2 | +6.61 | 1.00 | 0.031 | 0.300 | 0.032 |

At lr = 1e-2, the margin goes above 4 and the training acc is 1.00. The model seems to "learn best". But a large part of the sampled answers do not even have the correct format, and only 6%–7% are correct.

The two rows with a large learning rate have very unstable training, and they are very sensitive to floating-point errors. In a 2026-10 rerun on a different server, the lr = 1e-2 row was +4.20 / 1.00 / 0.068 / 0.618 / 0.068: the fraction of samples with a wrong format changed from more than half to almost 40%. The 5e-2 row was +5.73 / 1.00 / 0.000 / 0.172 / 0.000. The first three rows almost did not change.

The exact fractions change, but the direction stays the same. To push the numbers of rejected down, the model damages the probabilities of all "digit tokens" together. **A high margin ≠ a good model.**

The smoke test of the main line had the same problem (**tiny-configuration demo**, see the comments in `configs/tiny/dpo.toml`). With a DPO learning rate of 5e-4, the margin went to 4.8 in 24 steps. In the next GRPO stage, the tool-calling format accuracy fell from 0.16 to 0. Only a change to 5e-5 made the training stable.

For real models, the DPO learning rate is usually more than 10 times smaller than the SFT learning rate: Zephyr 5e-7, Tülu 3 8B 5e-7 and 70B 2e-7, OLMo 2 7B 1e-6, Qwen2.5 7e-7. Llama 3 used 1e-5 (plus the regularization below). The default of the main-line configuration is 5e-7 (to be tuned).

**Pitfall 2: the probability of chosen also goes down.** Change the wrong answers to "off by only 1 or 2", so that chosen and rejected are similar. Keep everything else the same:

| Step | margin | acc | log π(chosen) | log π(rejected) |
|---:|---:|---:|---:|---:|
| 0 | +0.000 | 0.00 | −1.045 | −1.984 |
| 25 | +0.050 | 0.70 | −1.131 | −2.574 |
| 50 | +0.107 | 0.74 | −1.134 | −3.144 |
| 75 | +0.156 | 0.79 | −1.138 | −3.639 |
| 100 | +0.201 | 0.79 | −1.254 | −4.201 |
| 125 | +0.258 | 0.79 | −1.442 | −4.960 |
| 150 | +0.357 | 0.84 | −1.662 | −6.168 |

The loss goes down, the margin goes up, and acc goes up. But the log-probability of chosen goes down all the time. On the held-out prompts, the probability of the correct answer goes **from 0.353 to 0.223**. Correct samples go from 0.340 to 0.230, and well-formed samples from 1.000 to 0.887. DPO only asks that "chosen goes down less than rejected". The probability that DPO pushes down can flow to other answers (also to answers with a wrong format).

Large models show this effect again and again. Nemotron-4 reports that "the likelihoods of both chosen and rejected go down all the time". Razin et al. 2024 call this effect **likelihood displacement**, and they found that it is worse when chosen and rejected are more similar.

A common fix is to **add an SFT (NLL) loss on chosen**: Llama 3 uses a coefficient of 0.2, and Nemotron-4 also adds it. Llama 3 also removes from the DPO loss the format tokens that chosen and rejected share (headers, end tokens). The reason: to "push the same token up and down at the same time" can cause repetition at the end or a sudden stop.

**Pitfall 3: overfitting, and looking only at training metrics.** In the two experiments above, the training acc is 0.95 / 0.84, but the held-out value is only 0.83 / 0.63. Zephyr reports that after one epoch of DPO, the training accuracy was already 100%. Always look at real metrics on a held-out development set (format accuracy, tool-calling score, chat evaluation), not at the training margin.

**Pitfall 4: length preference.** The reward model of section 3 learned "long = good". The implicit reward of DPO learns the same thing. Chosen is often longer than rejected, so the model learns to "write longer" (Park et al. 2024). The fixes include length normalization, length control when you build preference pairs, and length-controlled metrics in evaluation (for example, AlpacaEval 2 LC). Tülu 3 and OLMo 2 use "length-normalized DPO": divide the log-probability by the answer length.

## 9. Summary

- **Preference data**: (chosen, rejected) pairs for the same prompt. The judge can be a human, a strong model, or a program that scores automatically.
- **Bradley–Terry**: `P(y_w ≻ y_l) = σ(r_w − r_l)`. Only the difference has a meaning.
- **Reward model**: `−log σ(r_w − r_l)`, which is the binary cross-entropy. It learns the preference, and it also learns the bias.
- **RLHF**: `max E[r] − β·KL(π‖π_ref)`. The KL term is the leash that prevents reward hacking. PPO solves the objective with sampling + a value baseline + clipping (a preview of Chapter 19).
- **DPO**: optimal solution `π* = π_ref·e^{r/β}/Z` → solve for `r = β·log(π*/π_ref) + β·log Z` → put r into Bradley–Terry, Z cancels → `−log σ(β·[Δ_w − Δ_l])`.
- **Gradient**: push chosen up and rejected down, with strength β·σ(−h).
- **Pitfalls**: a learning rate that is too large damages the format; chosen also goes down (add NLL, mask format tokens); the training margin alone overestimates the result; length preference.

---

## From minimal code to production code

In the main-line code, the same method is in [`zero/post/dpo.py`](../../zero/post/dpo.py). The configurations are [`configs/tiny/dpo.toml`](../../configs/tiny/dpo.toml) and [`configs/main/dpo.toml`](../../configs/main/dpo.toml). The tests are in [`tests/test_dpo.py`](../../tests/test_dpo.py).

| Minimal code (`code/`) | Production code (`zero/`) | What it adds and why |
|---|---|---|
| `dpo_loss` in `03` | `zero.post.dpo.dpo_loss(policy_chosen_logps, policy_rejected_logps, ref_chosen_logps, ref_rejected_logps, beta)` | The same formula. It also returns the metrics `acc`, `margin`, `chosen_reward`, and `rejected_reward`, and training logs them at each step. To watch for "chosen also goes down", look at `chosen_reward`. |
| `response_logps` in `04`: joins "prompt + answer" by hand and masks the prompt | `encode_pair` → `zero.post.chat.encode_prompt_response`: renders the pair with the chat template (with tool definitions and `tool_calls`). Only the tokens of the assistant reply count in the log-probability. It drops preference pairs that are longer than `seq_len` | The same template and loss mask as SFT in Chapter 16. This makes sure that DPO optimizes the exact text that the model generates at inference. |
| One forward pass for each answer | `batch_logps`: joins chosen and rejected into one batch, pads on the right, and does one forward pass. `zero.post.common.sequence_token_logprobs` / `token_logprobs` get the log-probability of each target token (float32) | One forward pass saves half of the kernel launches. log_softmax runs in float32 to prevent summation errors for long answers in BF16. |
| Reference log-probabilities calculated before training | `[dpo] ref_mode = "precompute"` (calculate the reference log-probabilities for all data once, before training) or `"online"` (calculate them at each step with a frozen copy) | precompute saves the memory of one full model. online is for data that training generates while it runs. |
| A fixed set of 280 pairs | `make_env_preferences`: when `generate_pairs > 0` and the file does not exist, it uses the tasks of the tool-calling environment `zero/post/envs/tool_env.py`. It samples `samples_per_prompt` answers from the current policy and scores them with the verifiable reward. The highest score becomes chosen (only a full score counts); otherwise, the reference solution becomes chosen. The lowest score becomes rejected. If all answers have a full score, a broken copy of the reference call becomes rejected | **On-policy preference data**: rejected is an error that the model really makes. No human labels are necessary. |
| Adam, fixed learning rate | `zero.post.common.LoopState`: AdamW, warmup + cosine, gradient clipping, gradient accumulation, JSONL log, checkpoint, resume from checkpoint | The same training loop as the other post-training stages. |
| One CPU process | One process; BF16 autocast on CUDA | Multi-GPU DDP is not implemented yet. We verified the single-GPU CUDA + BF16 path on an RTX 3090 (see section 9 of [runs/2026-10-01-gpu0-check](../../runs/2026-10-01-gpu0-check/README.md)). |

**Parity check** ([`tests/test_dpo.py`](../../tests/test_dpo.py)): `test_dpo_loss_hand_computed` checks the loss, acc, and margin on two pairs that we calculated by hand. It also checks that the loss is ln 2 when policy = ref, and that the gradient pushes chosen up and rejected down. `test_batch_logps_only_counts_response` checks that the sequence log-probability counts only the reply tokens and agrees with a token-by-token calculation by hand. `test_run_dpo_end_to_end` runs 3 steps in each of the two modes, precompute and online. It checks that the first loss = ln 2, that the loss then goes down, and that the checkpoint is saved to disk. Part ⑤ of `03` in this chapter also compares the loss from scratch with `dpo_loss`, for values and gradients.

```bash
uv run pytest tests/test_dpo.py -q
```

```
....                                                                     [100%]
4 passed in 16.09s
```

**The main-line configuration `configs/main/dpo.toml`, item by item**:

| Setting | Value | Why |
|---|---|---|
| `init_from` | `out/main/distill/ckpt` | The pipeline order is SFT → distillation → DPO → GRPO. The reference model = the model after distillation. |
| `[dpo] beta` | 0.1 | The same as the original DPO paper, Zephyr, and Llama 3. To be swept on a development set. |
| `[optim] lr` | 5e-7, warmup 50 steps, cosine to 0 | In the range of Zephyr / Tülu 3 (5e-7), Qwen2.5 (7e-7), and OLMo 2 7B (1e-6). The smoke test showed that a learning rate that is too large damages the tool-calling format. To be tuned. |
| `micro_batch_size × grad_accum` | 8 × 8 = 64 pairs per step | Zephyr 32, Tülu 3 128. 1000 steps ≈ 64,000 pairs. |
| `ref_mode` | `precompute` | One GPU can hold the 0.7B policy + the precomputed reference log-probabilities. |
| `seq_len` | 8192 | Tool descriptions + multi-turn chats are long. The code drops preference pairs that are longer. |
| `generate_pairs` / `samples_per_prompt` | 0 / 8 | By default, the main line reads the existing `data/dpo/prefs.jsonl`. Set `generate_pairs > 0` to make on-policy preference pairs with tool_env during the run. |

**Where the preference data of the second step comes from** (GOAL.md section 3.3: "improve the general chat quality", and keep the tool-calling format):

1. **Tool calling: on-policy + verifiable scores.** `make_env_preferences` is already written. The policy samples answers, and tool_env scores them. Qwen2 / Qwen2.5 use the same idea: "for tasks where a program can check correctness, use execution feedback to make preference pairs".
2. **General chat: open preference data sets (check the licenses)**:
   - [HelpSteer3](https://huggingface.co/datasets/nvidia/HelpSteer3): CC-BY-4.0, human labels, includes Chinese (the `language` list includes zh).
   - [UltraFeedback](https://huggingface.co/datasets/openbmb/UltraFeedback): MIT, GPT-4 scores. The binarized version [HuggingFaceH4/ultrafeedback_binarized](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized) is also MIT (Zephyr uses this version).
   - [Tülu 3 8B preference mixture](https://huggingface.co/datasets/allenai/llama-3.1-tulu-3-8b-preference-mixture): ODC-BY-1.0, but the card says that "some subsets are not for commercial use". Check each subset.
   - The answers in these data sets come from many models, and the judges are mostly GPT-4 / GPT-4o. Do the terms of use of the generating models and of the judges allow "training other models on their outputs"? **To be verified.** (The teacher-license rule of GOAL.md section 3.3 also applies here.)
3. **A synthetic pipeline like the one of Tülu 3**: use an open model whose license allows this use as the judge (the same license list as for the teachers of Chapter 17). The judge scores the answers of the SFT model and of some open models. The highest score becomes chosen.
4. Decontaminate all preference data against the evaluation sets with 13-gram matching (GOAL.md section 3.2).

## Main-line progress

### Tiny-configuration demo (CPU, `configs/tiny`, about 1.3M parameters)

> This section is a **tiny-configuration demo**. It only shows that the code path works. It does not represent any result of the main-line model.

> **Note:** The numbers in this section come from the smoke test **before** the fix of the tool-calling grader (Chapter 19, section 6). They are the real output of that time. After the fix, we ran the smoke test again (`uv run python -m zero.smoke --out out/smoke_final`). The data, pretraining, mid-training, and SFT stages gave exactly the same results.
>
> The number of distillation samples that passed verification changed from 1 (that nonsense sentence) to 0. The downstream DPO, GRPO, and evaluation numbers changed with it. For example, the tool-calling call_exact of the same SFT model changed from 0.133 to 0.100: the model did not change, but the grading became stricter. The verdict for GRPO against SFT was still "tie". When you run it yourself, trust your own result.

The DPO stage of `uv run python -m zero.smoke` (`out/smoke/SUMMARY.md`, `out/smoke/dpo/log.jsonl`). The smoke test sets `seq_len` to 512, makes 32 pairs, and runs 24 steps. Everything else is the same as `configs/tiny/dpo.toml`: β = 0.1, lr = 5e-5, precompute.

- Preference data: 32 pairs, all made by tool_env during the run. In 3 pairs, chosen comes from samples of the policy itself. The other 29 pairs use the reference solution (the tiny model is too weak and seldom answers correctly by itself).
- Training log:

| Step | loss | acc | margin | chosen_reward | rejected_reward | lr |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.6931 | 0.00 | +0.000 | +0.000 | +0.000 | 1.25e-5 |
| 5 | 0.6941 | 0.50 | −0.001 | +0.003 | +0.005 | 5.00e-5 |
| 10 | 0.6125 | 1.00 | +0.170 | +0.060 | −0.110 | 4.34e-5 |
| 15 | 0.3797 | 1.00 | +0.802 | +0.011 | −0.790 | 2.75e-5 |
| 20 | 0.3531 | 1.00 | +0.932 | +0.017 | −0.915 | 1.16e-5 |
| 24 | 0.5262 | 0.75 | +0.394 | −0.020 | −0.414 | 5.28e-6 |

- At step 1, the loss is exactly ln 2 (policy = reference model). After that, rejected_reward becomes more negative all the time. chosen_reward stays near 0, and it is a little negative at the last step. This is a sign of the same effect as "chosen does not always go up" in section 8. (Each step has only 4 pairs, so the noise is large, and we cannot make a conclusion.)
- Learning rate: at first, we used 5e-4. In 24 steps, the margin went to 4.8, and in the next GRPO stage, the format accuracy fell from 0.16 to 0. We changed the learning rate to 5e-5 (see the comments in `configs/tiny/dpo.toml`).
- The eval stage after DPO compares the SFT model and the final GRPO model. It does not evaluate the DPO model separately.

### To be added after GPU training

- The sources, size, license list, and decontamination results of the main-line preference data.
- The sweep results of β and of the learning rate on the development set. Look at the tool-calling format accuracy and the general chat evaluation, not at the training margin.
- Training curves (loss, acc, margin, chosen_reward / rejected_reward) and cost (record it in `runs/ledger.md`).
- A comparison before and after DPO on the development set, and the effect of DPO on the next step, GRPO.
- If "chosen goes down too" or format degradation occurs: do we need an NLL term on chosen, or a mask for format tokens? (`zero` does not implement either of them now. See the note below.)

> **Note:** `zero` currently implements the original DPO. It does not have the "NLL regularization + format-token mask" of Llama 3, and it does not have the length normalization of Tülu 3. These changes are all small. If the second step shows the related problem on the development set, we add them and run a controlled comparison.

---

## Frontier notes

> **Variants of DPO**: many variants appear. IPO (Azar et al. 2023) replaces −log σ with a squared loss to prevent overfitting. KTO (Ethayarajh et al. 2024) needs only single "good/bad" labels, not pairs. SimPO (Meng et al. 2024) removes the reference model and uses the length-normalized log-probability as the reward. ORPO (Hong et al. 2024) puts the preference term into the SFT loss and needs no reference model.
>
> Among the leading open models, we see only scattered adoption. For example, SmolLM3 uses APO, Nemotron-4 uses its own RPO, and Tülu 3 compared the methods and chose length-normalized DPO, not SimPO. Per GOAL.md section 2.1, we only mention them here, in one note.

---

## Adopters and sources

| Technique | Adopters (main versions) | Sources |
|---|---|---|
| DPO | **Llama 3** (DPO after each SFT round; lr 1e-5, β = 0.1; format-token mask + 0.2 × NLL; compared DPO with PPO and found that DPO needs less compute and gives better IFEval); **Qwen2** (offline DPO + an online stage that uses the reward model to pick the best/worst answers for DPO); **Qwen2.5** (offline DPO with about 150,000 pairs, lr 7e-7, then online GRPO); **DeepSeek LLM** (DPO after SFT, lr 5e-6, batch 512); **Tülu 3 / OLMo 2** (length-normalized DPO + on-policy synthetic preference data; Tülu 3 8B: lr 5e-7, β = 5); **Nemotron-4 340B** (DPO, then three rounds of RPO, with an SFT loss on chosen); **Zephyr** (dDPO on UltraFeedback, lr 5e-7, β = 0.1) | [Llama 3](https://arxiv.org/abs/2407.21783) §4.1.4; [Qwen2](https://arxiv.org/abs/2407.10671) §4.3; [Qwen2.5](https://arxiv.org/abs/2412.15115) §4.2–4.3; [DeepSeek LLM](https://arxiv.org/abs/2401.02954) §4; [Tülu 3](https://arxiv.org/abs/2411.15124) §5; [OLMo 2](https://arxiv.org/abs/2501.00656) §5; [Nemotron-4 340B](https://arxiv.org/abs/2406.11704) §3.3.2; [Zephyr](https://arxiv.org/abs/2310.16944) §4.4 |
| DPO variant (APO) | **SmolLM3** (APO, "a more stable variant of DPO"; for the non-reasoning mode, Tülu 3 preference data; for the reasoning mode, Qwen3-32B answers as chosen and Qwen3-0.6B answers as rejected) | [SmolLM3 blog](https://github.com/huggingface/blog/blob/main/smollm3.md); [APO](https://arxiv.org/abs/2408.06266) |
| Reward model + PPO (preview) | **InstructGPT** (6B reward model, PPO, β = 0.02, PPO-ptx); Llama 2 (reward model + rejection sampling + PPO); Tülu 3 compared PPO and DPO | [InstructGPT](https://arxiv.org/abs/2203.02155) §3.5, Appendix C.4; [Stiennon et al. 2020](https://arxiv.org/abs/2009.01325); [PPO](https://arxiv.org/abs/1707.06347) |

**Consensus check (GOAL.md section 2.1)**: at least 5 independent leading families (Llama, Qwen, DeepSeek, OLMo, Nemotron) clearly use DPO in their main versions (SmolLM3 uses a variant). DPO meets rule A, so it is in the main text. Reward model + PPO is the starting point of the DPO derivation and the preview of GRPO in Chapter 19 (rule C).

> **Note:** Newer models (for example, after Qwen2.5) put more of the post-training budget into online RL (GRPO, Chapter 19). Does the post-training of the newest main versions, such as DeepSeek-V3 and Qwen3, still include DPO? This chapter did not check each model (to be verified).

---

## Guided questions

Ask Claude Code these questions. Continue until you can explain the answers in your own words:

1. Why is the gradient of the reward-model bias always 0? Before RL, InstructGPT sets the mean reward of the demonstration answers to zero. Without this step, would PPO change? Does DPO need this step?
2. In the table of `02`, at β = 0.1, the probability of "padded 900" is 0.61, but under π_ref it is only 0.004. Calculate with `π* ∝ π_ref·e^{r/β}`: how large must the reward difference be to change an answer from 0.4% to 61%? The result shows how dangerous a small error of the reward model is "in a region that it never saw".
3. Step 4 of the DPO derivation, "replace π* with π_θ", contains a hidden assumption: the policy can represent any distribution, and the preference data covers all answers. When the data covers only a small part of the answers, does DPO constrain the probability of "answers that never occur"? How is this related to "chosen also goes down" in section 8?
4. Why is DPO **off-policy**, and PPO on-policy? Why does Llama 3 "mainly use preference data that the best model of the latest round collected"? Which type is `make_env_preferences`?
5. Length-normalized DPO divides the log-probability by the answer length. Is its optimal solution still `π_ref·e^{r/β}`, as for the original DPO? Why does Tülu 3 set β as large as 5?
6. Suppose that the "judge" of the preference data is itself a reward model (for example, Nemotron-4 uses a reward model to rank synthetic answers). What difference between DPO and RLHF is left?

## Hands-on tasks

For each task, run the code and look at the result.

**Task 1 (basic)**: In `01_bradley_terry.py`, change the length coefficient of the labeler from 0.4 to −0.4 (the labeler dislikes long answers). Train the reward model again, then run `02`. Does reward hacking still occur? Does the best β change?

**Task 2 (core)**: In `run_dpo` of `04_toy_dpo.py`, add an NLL loss on chosen: `loss = dpo + α·(−pw.mean() / n_answer_tokens)`, with α = 0.2, like Llama 3. Then run it again with the setup of `05` where "the wrong answers differ by only a little". Does the log-probability of chosen still go down all the time? What is the held-out probability of the correct answer now?

**Task 3 (challenge)**: Implement length-normalized DPO (divide the log-probability by the number of answer tokens). Build preference data where "chosen is always longer than rejected". For example, chosen is `12;` and rejected is `9;`, and some correct answers use a longer form such as `12!;`. Train with the original DPO and with length-normalized DPO. Compare the mean length of the answers that each model generates.

---

## Go deeper: CS336

This chapter matches Stanford CS336 (Spring 2026) <https://cs336.stanford.edu/>:

- **Lecture 15: mid-training and post-training (SFT / RLHF).** It covers the RLHF pipeline in the InstructGPT style, reward models, and the relation between PPO and DPO. It is the original English lecture for sections 3–5 of this chapter.
- **Optional part 2 of Assignment 5 (Alignment and Reasoning RL): DPO.** Implement DPO on a real small model and real preference data. It is a larger version of `03` and `04` in this chapter. The main part of Assignment 5 (SFT, expert iteration, GRPO) matches Chapters 16 and 19.

---

## References

- Rafailov et al. *Direct Preference Optimization: Your Language Model is Secretly a Reward Model*, 2023: <https://arxiv.org/abs/2305.18290>
- Ouyang et al. *Training language models to follow instructions with human feedback* (InstructGPT), 2022: <https://arxiv.org/abs/2203.02155>
- Stiennon et al. *Learning to summarize from human feedback*, 2020: <https://arxiv.org/abs/2009.01325>
- Schulman et al. *Proximal Policy Optimization Algorithms*, 2017: <https://arxiv.org/abs/1707.06347>
- Gao, Schulman, Hilton. *Scaling Laws for Reward Model Overoptimization*, 2022: <https://arxiv.org/abs/2210.10760>
- Llama Team. *The Llama 3 Herd of Models* (§4.1.4 DPO), 2024: <https://arxiv.org/abs/2407.21783>
- Lambert et al. *Tülu 3: Pushing Frontiers in Open Language Model Post-Training* (§5 preference fine-tuning), 2024: <https://arxiv.org/abs/2411.15124>
- OLMo Team. *2 OLMo 2 Furious* (§5 post-training), 2024: <https://arxiv.org/abs/2501.00656>
- Qwen Team. *Qwen2 Technical Report* (§4.3), 2024: <https://arxiv.org/abs/2407.10671>; *Qwen2.5 Technical Report* (§4.2–4.3), 2024: <https://arxiv.org/abs/2412.15115>
- DeepSeek-AI. *DeepSeek LLM: Scaling Open-Source Language Models with Longtermism* (§4), 2024: <https://arxiv.org/abs/2401.02954>
- NVIDIA. *Nemotron-4 340B Technical Report* (§3.3.2 DPO and RPO), 2024: <https://arxiv.org/abs/2406.11704>
- Tunstall et al. *Zephyr: Direct Distillation of LM Alignment*, 2023: <https://arxiv.org/abs/2310.16944>
- Hugging Face. *SmolLM3: smol, multilingual, long-context reasoner* (blog): <https://github.com/huggingface/blog/blob/main/smollm3.md>
- D'Oosterlinck et al. *Anchored Preference Optimization and Contrastive Revisions* (APO), 2024: <https://arxiv.org/abs/2408.06266>
- Razin et al. *Unintentional Unalignment: Likelihood Displacement in Direct Preference Optimization*, 2024: <https://arxiv.org/abs/2410.08847>
- Pal et al. *Smaug: Fixing Failure Modes of Preference Optimisation with DPO-Positive*, 2024: <https://arxiv.org/abs/2402.13228>
- Park et al. *Disentangling Length from Quality in Direct Preference Optimization*, 2024: <https://arxiv.org/abs/2403.19159>
- Singhal et al. *A Long Way to Go: Investigating Length Correlations in RLHF*, 2023: <https://arxiv.org/abs/2310.03716>
- Variants (Frontier notes): IPO <https://arxiv.org/abs/2310.12036>; KTO <https://arxiv.org/abs/2402.01306>; SimPO <https://arxiv.org/abs/2405.14734>; ORPO <https://arxiv.org/abs/2403.07691>
- Preference data sets: [UltraFeedback](https://huggingface.co/datasets/openbmb/UltraFeedback) (MIT), [HelpSteer3](https://huggingface.co/datasets/nvidia/HelpSteer3) (CC-BY-4.0), [Tülu 3 8B preference mixture](https://huggingface.co/datasets/allenai/llama-3.1-tulu-3-8b-preference-mixture) (ODC-BY-1.0, some subsets are not for commercial use)
- [Hands-On Modern Reinforcement Learning](https://walkinglabs.github.io/hands-on-modern-rl/preface/intro) (in `references.md`; an introduction to RL and PPO in Chinese)
- [CS336](https://cs336.stanford.edu/) Lecture 15, Assignment 5

**Next chapter**: a preference pair can only tell the model "which one is better", and the data is collected before training. Tasks such as tool calling have a better property: a program can check correctness automatically. So let the model try by itself, get a score at once, and learn from its own attempts. Chapter 19: reinforcement learning — from PPO to GRPO, and verifiable rewards.
