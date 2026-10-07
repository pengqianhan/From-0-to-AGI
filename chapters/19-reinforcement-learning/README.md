# Chapter 19: Reinforcement learning — The model learns from its own attempts

**English** · [中文](README.zh.md)

> **Goal**: After this chapter, you can derive the policy gradient from the "log-derivative trick". You can explain why GRPO does not need a value model, how to calculate the group advantage, and what clipping and KL each control. You can design a verifiable reward for tool calls. When a training curve "looks very good", you can recognize reward hacking, find the loophole, and add a guard.

📺 **Video**: Not published yet. To render it on your computer, run `bash chapters/19-reinforcement-learning/video/build.sh`.
🧪 **Self-check**: After the chapter, type `/ch19-rl` in Claude Code.

---

In the previous chapter, we aligned the model to human preferences. First, we introduced RLHF (a reward model + PPO) as background. Then we derived DPO from the same objective. Now the model can talk, it can use the tool-call format, and its replies are more pleasant. But everything that it learned came from **answers that someone else gave**. SFT copies demonstrations, distillation copies a teacher, and DPO copies labels of "which one is better".

This chapter solves one problem: **the model can imitate, but it cannot solve problems**. Where does it learn a skill that is not in the demonstrations? The answer is reinforcement learning (RL): the model tries by itself. A program that can automatically tell correct from wrong (a verifier) scores each attempt. Then the attempts with high scores become more frequent. This is the last training stage of the main-line model: GRPO in a tool-call environment.

## 1. The ceiling of imitation

First, look at a very small example ([`code/02_grpo_from_scratch.py`](code/02_grpo_from_scratch.py)). The task is to add two one-digit numbers. The model outputs the digits of the answer and then `<eos>`.

The teacher is a model that is "not good at carries". It answers all problems without a carry correctly. On problems with a carry, only 30% of its answers are correct. In the other 70%, it forgets to write the carry (7+5 becomes "2"). The student does SFT on 2000 demonstrations from the teacher:

| | Accuracy |
|---|---:|
| Teacher demonstrations (2000) | 0.684 |
| SFT student, sampling | 0.68 |
| SFT student, greedy decoding | 0.60 |
| SFT student, greedy decoding, carry problems only | **0.11** |

The student learned "well": its sampling accuracy is almost the same as the accuracy of the teacher. It also learned the errors of the teacher exactly. On carry problems, the teacher is wrong most of the time. Thus the most probable answer of the student is also wrong, and greedy decoding gets only 0.11. **The upper limit of imitation learning is the demonstration data.** The student inherits each bias in the demonstrations exactly.

But note one detail. When the student samples, it still answers a carry problem correctly about 30% of the time. The correct answer **is in the distribution of the model**, but it is not the most probable answer. A program can tell the model "this answer was correct, that answer was wrong". Then we can make this 30% larger.

One line of code is sufficient to check if the sum of two numbers is correct. Such a reward is a **verifiable reward**. We use it for reinforcement learning. After 60 steps:

| | After SFT | GRPO step 5 | Step 10 | Step 60 |
|---|---:|---:|---:|---:|
| Greedy accuracy | 0.60 | 0.96 | 1.00 | 1.00 |
| Greedy accuracy (carry problems) | 0.11 | 0.91 | 1.00 | 1.00 |
| Sampling accuracy | 0.68 | 0.88 | 0.95 | 0.98 |

The student is now better than the teacher. The next sections explain each part of what happened.

## 2. Policy gradient: the score has no derivative, so where does the gradient come from?

Think of a language model as a **policy** π_θ. For a prompt x, it generates a response y with some probability. We want to maximize the expected reward:

```
J(θ) = E_{y~π_θ(·|x)} [ r(x, y) ]
```

The problem is that r is a scoring program ("is the answer correct?"). It has no derivative with respect to θ, so no gradient can flow back through it. The **log-derivative trick** goes around this problem:

```
∇J = ∇ Σ_y π_θ(y) r(y)
   = Σ_y r(y) ∇π_θ(y)
   = Σ_y π_θ(y) r(y) ∇log π_θ(y)        (because ∇π = π · ∇log π)
   = E_{y~π_θ} [ r(y) ∇log π_θ(y) ]
```

The last line is an expectation, so we can estimate it with samples. Let the model generate some responses. Multiply the "gradient of the log-probability" of each response by its score, and take the mean. r occurs only as a number. **r does not need a derivative. It only needs to give a score.** This is REINFORCE (Williams, 1992). The intuition is direct: a response with a high score becomes more probable, and a response with a low score becomes less probable.

For a language model, log π_θ(y) = Σ_t log π_θ(y_t | x, y_<t). Thus ∇log π is the sum of the log-probability gradients of all tokens in the response. This is exactly the gradient of the SFT cross-entropy. The only difference is one more factor for each sample: its "score". SFT is the special case in which the factor of every sample is 1. (Section 5 of the DeepSeekMath paper writes SFT, rejection sampling, DPO, PPO, and GRPO as different choices of one "gradient coefficient".)

[`code/01_reinforce_bandit.py`](code/01_reinforce_bandit.py) checks this on a 5-armed bandit. The policy is π = softmax(θ). The script does a parity check of the sampled estimate against the exact gradient:

| Number of samples N | Max difference between the sampled estimate and the exact gradient |
|---:|---:|
| 100 | 0.0217 |
| 10,000 | 0.0010 |
| 1,000,000 | 0.0001 |

## 3. Variance and the baseline

REINFORCE is unbiased, but its variance is large. A key observation: for any constant b,

```
E[ (r − b) ∇log π ] = E[ r ∇log π ] − b · E[∇log π] = ∇J       (because E[∇log π] = Σ ∇π = ∇1 = 0)
```

When we subtract a **baseline** b, the expectation does not change, but the variance can change by a large amount. The intuition: suppose that all responses get a score of about 10. Without a baseline, each response is "pushed up" by a slightly different amount, so the signal is lost in the noise. Subtract the mean score: then the responses above the mean go up, and the responses below the mean go down. The signal is much cleaner. The output of `01`:

| Reward | Baseline b | Max difference between the mean and the exact gradient | Variance (sum over components) |
|---|---|---:|---:|
| Raw r | 0 | 0.0004 | 0.1946 |
| Raw r | E[r] | 0.0004 | 0.0534 |
| r + 10 | 0 | 0.0076 | 84.70 |
| r + 10 | E[r] | 0.0006 | 0.0533 |

When we add 10 to all rewards, the variance without a baseline is **1589×** the variance with a baseline. We train for 300 steps with the same learning rate (8 samples per step, mean of 5 seeds). The probability of the best arm is:

| | Step 1 | Step 50 | Step 100 | Step 300 |
|---|---:|---:|---:|---:|
| No baseline | 0.21 | 0.20 | 0.20 | 0.40 |
| Baseline = mean reward of this batch | 0.21 | 0.85 | 0.95 | 0.99 |

Remember this method: "use the mean reward of the same batch of samples as the baseline". It is the core of GRPO.

## 4. From PPO to GRPO

### 4.1 Review of PPO (background from Chapter 18)

PPO (Proximal Policy Optimization) is the standard algorithm of RLHF. It adds two things to REINFORCE:

1. **A value model (critic) as the baseline**: Train a model V(s) of about the same size as the policy. It predicts "about how much score we can still get from this position". Then GAE calculates the advantage A_t of each token.
2. **A clipped importance ratio**: PPO updates several times on one batch of samples (to save sampling). At the second update, the policy has already changed. Thus PPO corrects with ρ_t = π_θ(y_t)/π_old(y_t). It also limits ρ to [1−ε, 1+ε], so that one update does not go too far:

```
ℓ_t = −min( ρ_t · A_t,  clip(ρ_t, 1−ε, 1+ε) · A_t )
```

Look at the effect of clipping separately for a positive and a negative advantage. When A > 0 (a good response), the gradient stops when ρ goes above 1+ε: "the increase is sufficient, do not push more". When A < 0 (a bad response), the gradient stops when ρ goes below 1−ε: "the decrease is sufficient".

The problem is the value model. It is as large as the policy, so it uses a second share of memory and adds a second model to train. Also, a language model usually gets its reward as one number at the end of the response. It is difficult for the value model to give accurate estimates at each token (this is from Section 4.1.1 of the DeepSeekMath paper).

### 4.2 GRPO: use "the other responses to the same prompt" as the baseline

GRPO (Group Relative Policy Optimization, DeepSeekMath, 2024) **removes the value model completely**. For each prompt, it samples G responses. The verifier scores them as r_1..r_G. Normalization within the group gives the advantage:

```
A_i = ( r_i − mean(r_1..r_G) ) / std(r_1..r_G)
```

All tokens in response i share the same A_i. This is the "batch-mean baseline" from Section 3, but grouped by prompt. On the same problem, the responses above the mean go up, and the responses below the mean go down. The division by the standard deviation puts the signals of different problems on the same scale. Here is a real group of samples from step 1 of `02` (problem 2+8, G = 8):

| Response | 10 | 0 | 0 | 10 | 10 | 0 | 10 | 0 |
|---|---|---|---|---|---|---|---|---|
| Reward | 1 | 0 | 0 | 1 | 1 | 0 | 1 | 0 |
| Advantage | +0.94 | −0.94 | −0.94 | +0.94 | +0.94 | −0.94 | +0.94 | −0.94 |

(With 4 correct and 4 wrong, the mean is 0.5 and the unbiased standard deviation is 0.535, so the advantages are ±0.94.) "0" is the answer that forgot the carry.

The code is only a few lines:

```python
def group_advantages(rewards, eps=1e-6):          # rewards: (P, G)
    mean = rewards.mean(1, keepdim=True)
    std = rewards.std(1, keepdim=True)            # unbiased std, the same as TRL / verl / zero
    return (rewards - mean) / (std + eps)         # A_i = (r_i − mean) / std

def grpo_loss(logp, old_logp, ref_logp, adv, mask, eps_clip=0.2, beta=0.02):
    ratio = torch.exp(logp - old_logp)                                 # ρ_t = π_θ / π_old
    A = adv[:, None]
    per_tok = torch.maximum(-ratio * A,
                            -torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * A)   # clipping
    d = ref_logp - logp
    kl = torch.exp(d) - d - 1                                          # k3: unbiased KL estimate, never negative
    per_tok = per_tok + beta * kl
    m = mask.float()
    return (per_tok * m).sum() / m.sum()                               # token_mean aggregation
```

The full training loop is: take P prompts → copy each prompt G times and sample → score with the verifier → group advantages → update μ times with the loss above. `02` uses P = 16, G = 8, ε = 0.2, β = 0.02, μ = 2.

**When a group is all correct or all wrong, all its advantages are 0, and the group adds nothing to the gradient.** In the log of `02`, the fraction of "zero-variance groups" increases from 0.56 at step 5 to 0.94 at step 20. As the model becomes better, fewer problems give a signal. This is one of the most frequent problems of GRPO in practice. The "difficulty filtering" in Section 5 is for this problem.

### 4.3 Three details: KL, aggregation, and when clipping acts

**KL constraint and the k3 estimate.** As an option, add a term β · KL(π_θ ‖ π_ref). The term keeps the policy near its start point (the SFT/DPO model). GRPO adds the term directly to the loss (RLHF subtracts it from the reward instead). GRPO estimates it per token with k3:

```
KL ≈ π_ref/π_θ − log(π_ref/π_θ) − 1        (Schulman, "Approximating KL"; DeepSeekMath, Eq. 4)
```

The k3 estimate is never negative (x − log x − 1 ≥ 0), and its expectation is equal to the true KL. With verifiable rewards, many teams remove the KL term completely (β = 0). The reasoning RL of GLM-4.5 "removes the KL term from the GRPO framework". MiMo-7B and OLMo 3 also remove it. DeepSeek-R1 keeps a very small β = 0.001.

The reason: in RLHF, the KL term keeps the policy out of regions where the reward model is not reliable. A rule-based verifier does not become "unreliable", so the constraint is less necessary. The main-line configuration `configs/main/grpo.toml` uses `kl_coef = 0.0` by default. The tiny configuration uses 0.02 to show how to use a reference model.

**Aggregation.** The responses in a batch have different lengths. How do we average the per-token loss? The original GRPO paper first takes the mean within each response, then the mean over the responses (`seq_mean_token_mean`). Then each token in a long response gets a smaller weight: in a long, wrong response, each token gets a smaller penalty. DAPO changes this to one mean over the tokens of all responses together (`token_mean`).

The objective of MiMo-7B is a token-level mean with 1/Σ|o_i|. OLMo 3 explicitly uses a token-level loss "to avoid a length bias". GLM-4.5 compared the two methods in code RL, and the token-weighted mean converged faster. `zero` uses `loss_agg = "token_mean"` by default.

**When clipping acts.** Suppose that each batch of samples gets only one update (μ = 1). Then π_old is the current policy, ρ ≡ 1, and clipping never occurs. But ∇ρ = ∇log π is not zero, so there is still a gradient: the method becomes REINFORCE with a group baseline. ρ moves away from 1 only when the same batch gets several updates (μ > 1, or a large batch split into several small batches that update one after another). `02` uses μ = 2, and the clip ratio is between 0.00 and 0.02. The tiny configuration of the smoke test uses `ppo_epochs = 1`, so clip in the log is always 0.00.

## 5. Improvements to GRPO: which go into the main text, and which we still watch

Since 2025, many improvements to GRPO have appeared. We use the rule in GOAL.md, Section 2.1: at least 3 independent leading open-model families must explicitly adopt a method in their technical reports. We checked each item. The results:

| Method | Who adopts it explicitly (technical reports) | Conclusion | In zero |
|---|---|---|---|
| Group baseline, no value model (GRPO itself) | DeepSeek-R1, Qwen3 (reasoning RL), GLM-4.5, MiMo-7B, OLMo 3; Kimi K2 uses a group-mean baseline, but its objective is a variant of K1.5 | **Consensus** | `group_advantages` |
| Token-level aggregation | MiMo-7B, OLMo 3, GLM-4.5 (code RL); origin: DAPO | **Consensus** (3 families) | `loss_agg = "token_mean"` (default) |
| Remove the KL term | GLM-4.5, MiMo-7B, OLMo 3; DeepSeek-R1 still keeps β = 0.001 | **Common choice** (3 families), not required | `kl_coef = 0.0` (main) |
| Filter problems by difficulty: remove problems that are all correct or all wrong | MiMo-7B (dynamic sampling + resampling of easy problems), OLMo 3 (zero-gradient filtering + active sampling), GLM-4.5 (curriculum learning by difficulty), Kimi K2 (keeps only medium difficulty, by the pass@k of the SFT model), Qwen3 (a query must be "learnable for the cold-start model") | **Consensus principle**; the details of online dynamic sampling are different in each family | Only logs `zero_std_groups`; online filtering is **not implemented** (see "Main-line progress") |
| clip-higher (ε_high > ε_low) | MiMo-7B, OLMo 3; origin: DAPO (ByteDance Seed) | **To be verified**: we found only 2 leading families that explicitly adopt it | `clip_eps_high` (main sets 0.28; set 0 to turn it off) |
| No division by the standard deviation (Dr. GRPO) | OLMo 3 | Frontier note | `scale_rewards = false` turns it on |
| Sequence-level importance ratio (GSPO) | Later versions of Qwen3 (as the GSPO paper states) | Frontier note | Not implemented |

"To be verified" does not mean "not useful". In the DAPO ablation, clip-higher increased AIME from 36 to 38 (Qwen2.5-32B base model), and MiMo and OLMo 3 both use it. But by the rule of this course, we have not found clear evidence from a third family yet. The main-line configuration keeps clip-higher as a switch that you can turn off with one setting. In step 2, our own development set (not the preregistered test benchmarks) decides if we turn it on.

## 6. Verifiable rewards: how to score tool calls

The quality of the reward sets the upper limit of reinforcement learning. For reasoning tasks, DeepSeek-R1 **uses only rule-based rewards** (an accuracy reward + a format reward). It explicitly does not use a neural reward model, because "a neural reward model is prone to reward hacking in large-scale RL". The Qwen3 report also says that "well-designed rule-based rewards can judge the correctness of outputs with high precision and prevent reward hacking". In this type of reinforcement learning, a program can check the answer. Tülu 3 gave it the name **RLVR** (Reinforcement Learning with Verifiable Rewards):

| Domain | Verifier | Examples |
|---|---|---|
| Math | Extract the final answer and compare it with the reference answer for equivalence | DeepSeek-R1 (answer in a box), OLMo 3 (SymPy comparison) |
| Code | Run test cases in a sandbox | DeepSeek-R1, MiMo-7B (partial score by test difficulty), OLMo 3 (pass rate, or a score only when all tests pass) |
| Instruction following | One check function for each constraint | Tülu 3, OLMo 3 (fraction of the constraints that the output satisfies) |
| Tool calls | Format check + comparison of the call with the reference call / comparison of the execution results | GLM-4.5 (1 only if the format is correct and the call is exactly the same as the reference call), `tool_env` in this course |

Xiaomi's open-source [XiaomiMiMo/verl](https://github.com/XiaomiMiMo/verl) reproduces the five types of RL environment of MiMo-V2.6. They show the three main forms of a verifier: **executable tests** (code: software-engineering tasks), **rule checks** (vulnerability reproduction in cybersecurity, symbolic music composition), and **rubric grading** (general knowledge work: a set of "pass/fail" assertions that an LLM judge checks one by one). There is also visual grading for web development. The code of this repository shows some engineering principles that are worth copying:

- **Record environment failures and model failures separately.** An infrastructure error, such as a sandbox timeout or a crashed container, gets a clearly visible invalid value and a separate count. Do not silently record it as 0. Otherwise the model gets a penalty for a problem of the environment.
- **The scorer itself can also degrade silently.** A code comment records one case: a scoring script could not find a dependency because of a path problem. The script silently changed to a fallback method in which the partial score was always 0. Only monitoring could find the problem.
- **Keep a separate "anti-hacking" configuration** (the configuration name contains antihack).
- The random-reward switch for debugging saves a separate copy of the true score (`true_reward`). The repository records the training reward and the true score separately.

The environment of the main-line model is in `zero/post/envs/tool_env.py`. It has 6 simulated APIs: calculator, weather, unit conversion, date addition/subtraction, interval between dates, and day of the week. All are deterministic and do not use the network. About 10% of the problems do not need a tool call. These problems test if the model "does not call a tool when it should not". The scoring rules:

```
Format error (broken JSON, unpaired tags, forged <tool_response>, too long, too many calls) → −1
Calls a tool when it should not → −0.5; does not call a tool when it should → 0
Else  0.1 (format score) + 0.9 × Σ score of each reference call / number of reference calls
      − 0.25 × number of extra calls − 0.5 × number of calls that do not match the schema
      Each reference call: function name + arguments match → 1; only the function name matches → 0.2
Problems that need no tool: a normal answer → +0.5 (the content cannot be checked automatically);
      an empty reply or invented numbers → 0
```

"Arguments match" means that each argument is equal after normalization. Dates are in ISO format, units and city names have one standard spelling, and numbers become numeric values. BFCL uses the same idea. Only the calculator also accepts an "execution match": the scorer calculates the expression. If the result is the same and the expression has the same numbers, the call is correct (both `"28-44"` and `"28 - 44"` are correct). The end of this section explains why the other tools do not accept an execution match. [`code/04_tool_env_rewards.py`](code/04_tool_env_rewards.py) gives some typical outputs to the production scorer:

| Output (problem: "帮我算一下 28 - 44 等于多少？" ("Please calculate 28 - 44 for me.")) | Reward | Reason |
|---|---:|---|
| Correct call | +1.00 | |
| Broken JSON inside the tags | −1.00 | `tool_call is not valid JSON: Expecting value` |
| Bare JSON without the tags | −1.00 | Guard #8: `Tool call without <tool_call> tags (name / arguments JSON outside the tags)` |
| Calculates −16 in its head, then calls `calculator("-16")` | +0.28 | Guard #2: `Wrong arguments for calculator`. The execution result is the same, but the numbers in the expression are wrong, so only the function-name score |
| Outputs the same call twice | +0.75 | Guard #1: `Extra calls: 1`. One-to-one matching, −0.25 for the extra call |
| After the call, writes its own `<tool_response>` | −1.00 | Guard #3: `The output contains <tool_response> (a forged tool result or conversation turn)` |
| Chat problem ("Thanks a lot."): normal answer / empty reply / bare JSON / random call | +0.5 / 0 / −1 / −0.5 | `Correctly made no tool call (the answer content cannot be checked automatically)` / `Empty answer` / the guard #8 reason / `Called a tool when no tool was necessary`. A normal answer gets only 0.5: there is no reference answer to check |

Each guard blocks one shortcut that "gets a score without learning the skill". Each guard has tests in `tests/test_tool_env.py`. We did not think of guard 8 in advance. We added it after the hack **really occurred** in training. This is the story of the next section.

**Later, we fixed two more loopholes in the scorer.** We found them when we wrote Chapter 17 (distillation). Both came from "checking only the result, not the process":

1. **A result that was equal by coincidence counted as correct.** The old version accepted an "execution match" for all tools. For the question "What day of the week is 2024-06-03?", the model called `weekday("2024-06-10")`. The date was wrong by 7 days, but the day of the week was the same. The scorer judged it an execution match and gave the full score. The fix: except for the calculator, all tools now need arguments that are equal after normalization (`args_equivalent`). The calculator keeps the execution match, because it has an extra check that "the multiset of numbers is the same" (guard #2).
2. **No check of the answer content on problems that need no tool.** On chat problems, the old version only checked for "random calls" and gave the full score to a normal answer. In the smoke test, the only teacher sample that "passed verification" was nonsense. To "讲一句鼓励的话。" ("Say something encouraging."), it answered "坚下云，气温 28°C。" (meaningless words, then "temperature 28°C."). The fix: a normal answer now gets only `NO_TOOL_REWARD = 0.5` and the mark "cannot be checked". Distillation filtering does not use it as a verified sample. If an answer contains a number that is not in the problem (an invented fact), it gets 0.

Both fixes have regression tests (`test_wrong_args_with_coincident_result_not_full_credit` and `test_no_tool_invented_numbers_get_zero` in `tests/test_tool_env.py`). The lesson is direct: **a verifier must check "how the model got the result", not only "if the result is correct"**. This is also why BFCL matches the arguments with an AST and does not only compare execution results.

## 7. Reward hacking: the reward goes up, but the skill does not

### 7.1 A real case: the model learned to "remove the tags" in 10 steps

When we wrote the smoke test of `zero`, the first reward checked only the content **inside** the `<tool_call>` tags. Almost all calls of the tiny model (about 1.3M parameters) were broken JSON and got −1. GRPO ran for 10 steps, and the mean-reward curve went up nicely. Then we looked at the samples: the model had learned to **remove the tags and output the JSON anyway**. Without tags, the output is not a "call", so there is no format error, and the score changed from −1 to 0 ("should call, but did not"). The format error rate went down and the reward went up, but the model could not make one call.

At the same time, there was a second loophole. On problems that need no tool, any output without a call got the full score. An empty reply or a string of nonsense also got 1.

The fix is guard 8 of `tool_env`. If JSON such as `{"name": ...` / `"arguments":` occurs outside the tags, the output is a format error (−1). An empty reply to a chat problem gets 0, not 1.

### 7.2 A replay on a toy

[`code/03_reward_hacking.py`](code/03_reward_hacking.py) replays this event at a small scale. In the toy version, the correct call is `<call> 字 d </call> <eos>` ("function name + argument", d = (a+b) mod 10). Here `字` (Chinese for "word") is a token for plain text or a function name. The start model is like the tiny model: 59% of its outputs have a broken format, and only 5% are calls with the correct format. The two rewards differ only in guard 8. Each reward runs 120 steps of GRPO from the same start point with the same random seed:

| | Start | Naive reward, step 10 | Naive reward, step 120 | Fixed reward, step 10 | Fixed reward, step 120 |
|---|---:|---:|---:|---:|---:|
| Training reward (mean of this batch) | — | +0.11 | +0.06 | +0.16 | +0.22 |
| Format error rate | 0.59 | 0.01 | **0.00** | 0.10 | 0.00 |
| Tool problems: calls with the correct format | 0.05 | 0.01 | **0.00** | 0.87 | 1.00 |
| Bare calls ("call" outside the tags) | 0.19 | 0.85 | 0.66 | 0.01 | 0.00 |
| Chat problems: normal replies | 0.55 | 0.33 | **0.09** | 0.97 | 1.00 |
| True success rate (held-out criterion) | 0.14 | 0.08 | **0.02** | 0.29 | 0.36 |

With the naive reward, the format error rate goes from 0.59 to 0, and the training reward is positive. If you look only at these two curves, you think that the training was a success. But the calls with the correct format go from 0.05 to 0. The normal replies on chat problems go from 0.55 to 0.09. The true success rate goes from 0.14 to 0.02. Samples from the last step:

```
naive reward   tool 5+4  →  字 5 <eos>                    reward +0.0   (removed the tags)
               chat #21  →  <eos>                         reward +1.0   (empty reply gets the full score)
fixed reward   tool 5+4  →  <call> 字 7 </call> <eos>     reward +0.1
               chat #21  →  字 <eos>                      reward +1.0
```

Why does the hack come so fast? Look at the decision at the first token. In the demonstrations of the start model, about 96% of the outputs that "open a tag" are broken (−1). With the naive reward, "do not open a tag" gets a stable 0. The group advantage pushes the first token strongly toward "do not open a tag". When this choice becomes a habit, the model rarely samples outputs with the correct format, so these outputs rarely get a reward.

**The policy gradient only follows the direction in which the score increases most easily. It does not know what you really want.**

After the fix, the format problem is solved, and the true success rate goes from 0.14 to 0.36. But it stops there. This tiny model did not learn (a+b) mod 10, so its answers on tool problems are still mostly guesses. In the last batch of 120 tool-problem answers, about two thirds guess 7. The rest are spread over a few numbers, such as 0, 8, 1, and 5. The sample above, 5+4 → 7, is one example.

The guards block the shortcuts, but the skill must still come from the model, the data, and longer training. Also note: **you cannot compare the values of the two rewards**. At step 120, the fixed reward is only +0.22, but the model is much better than the model with the naive reward. When you change the reward function, the height of the curve no longer has a meaning.

### 7.3 General defenses

1. **Use a strict format.** Judge an output that is "almost correct" as a format error. This is better than a gap that lets outputs "go around the check". The function-call reward of GLM-4.5 is "1 only if the format is correct and the call is exactly the same as the reference call, else 0".
2. **Penalize degenerate outputs**: empty replies, outputs that are too long, repetition, forged tool results, and outputs that list all possible calls. Tülu 3 gives −10 to a reply without an EOS. Kimi K2 adds a special hack-check layer for instruction following. The layer detects outputs that "say that the task is complete, but it is not".
3. **Use a held-out verifier and human spot checks.** Keep the reward for training separate from the criterion for evaluation. Look at samples at regular intervals. Do not ask "what is the mean reward?". Ask "is this output really correct?". The MiMo code records the training reward and `true_reward` separately for the same reason.
4. **Prefer rules to model-based scores.** DeepSeek-R1 says that a neural reward model gets exploited in long RL training. Thus its second stage adds the preference reward only in the last 400 steps. MiMo-7B says that the base model exploits the scorer on math problems, but code problems that run test cases are difficult to exploit.
5. **For each loophole that you find, add a test.** In `tests/test_tool_env.py`, each guard has positive and negative examples.

## 8. What to monitor during training

| Metric | Healthy | Danger signal |
|---|---|---|
| Mean reward | Increases steadily, with noise | Increases suddenly: look at the samples first, it can be a hack |
| Format-correct rate / call rate | Increase together | Format-correct rate goes up, call rate goes down (as in Section 7.2) |
| Response length | Increases slowly on reasoning tasks | Increases suddenly (nonsense, repetition), or collapses to very short (empty replies) |
| KL (to the reference model) | Increases slowly | Jumps suddenly: the policy has moved too far |
| Entropy | Decreases slowly or stays almost stable | Collapses quickly to near 0: no more exploration; DAPO calls this "entropy collapse" |
| Clip ratio | Very small (a few percent) | Large: the learning rate or the number of updates is too high |
| Fraction of zero-variance groups | Medium | Near 1: the problems are too easy or too difficult; change or filter the problems |
| True success rate on the held-out set | Increases with the reward | The reward goes up, but the success rate does not: reward hacking |

`zero/post/grpo.py` logs these values at each step (`log.jsonl`): reward, format-correct rate, call rate, response length, KL, clip ratio, and the fraction of zero-variance groups. It does not log the entropy now. The DAPO paper warns about a counterintuitive effect. The correlation between the reward on the training set and the accuracy on the validation set is often low. Thus always look at a held-out set.

## 9. Reasoning models: RL teaches the model to "think longer"

DeepSeek-R1-Zero trained directly on the base model with GRPO and rule-based rewards (reward = accuracy + format; the thinking must be inside `<think>`). Its pass@1 on AIME 2024 increased from 15.6% to 77.9%. At the same time, the mean response length increased steadily during training. Nobody taught the model, but it learned by itself to verify, reflect, and try a different method in its answers. DeepSeek-R1 adds cold-start SFT and multi-stage RL to solve the problems of readability and language mixing.

The reasoning RL of Qwen3 used only 3995 "problem + verifier" pairs. In 170 steps, it increased the AIME'24 score of Qwen3-235B-A22B from 70.1 to 85.1.

Two warnings. First, "longer answers" do not always show more capability. Dr. GRPO shows that the sequence-mean method itself prefers longer wrong answers. Second, the Qwen3 report compares results for Qwen3-8B: RL increased pass@1, but pass@64 did not change. RL mostly takes correct answers that the model could already sample and makes them more frequent. This is the same mechanism as the carry example in Section 1.

For small models, the Qwen3 report also found that on-policy distillation from a large model works better than direct RL. On-policy distillation also uses only about 1/10 of the GPU hours (see Chapter 17).

The GRPO of the main-line model does not turn on thinking by default (tool calls must be short and accurate). It uses `max_new_tokens = 512`.

## 10. Summary

- **Imitation has a ceiling**: SFT and distillation inherit the errors in the demonstrations exactly. Reinforcement learning needs only a verifier that can give a score. Then it can make the "sometimes correct" answers in the distribution of the model more frequent.
- **Policy gradient**: ∇E[r] = E[r · ∇log π]. The reward does not need a derivative. A baseline does not change the expectation, and it decreases the variance by a large amount.
- **GRPO**: Sample G responses for each problem. A = (r − group mean)/group std, with no value model. Add clipping (it acts when μ > 1) and an optional k3 KL. Use token-level aggregation. A group that is all correct or all wrong gives no gradient.
- **Verifiable rewards**: For math, compare the answer. For code, run tests. For tool calls, compare the AST and the execution result. The three forms of a verifier are executable tests, rule checks, and rubric grading.
- **Reward hacking**: The model follows the direction in which the score increases most easily. In the smoke test, we saw the model learn to remove the tags in 10 steps. The defenses are a strict format, penalties for degenerate outputs, held-out criteria and human spot checks, and one test for each loophole.

---

## From minimal code to production code

The minimal code and the main-line code do the same thing. `02` calls the production functions directly on the same batch of data for a parity check: **max advantage difference 0.00e+00, loss −0.046469 vs −0.046469**. (`tests/test_grpo.py` also has small examples, calculated by hand, for clipping, clip-higher, k3 KL, and the two aggregation methods.)

| Minimal code (`code/`) | Production code (`zero/`, `configs/`) | What it adds and why |
|---|---|---|
| `group_advantages` (02) | `zero/post/grpo.py::group_advantages` | Supports `scale_rewards = false` (subtract only the mean, the Dr. GRPO method). Does not divide by 0 when G = 1 |
| `grpo_loss` (02, symmetric clipping, token_mean) | `zero/post/grpo.py::grpo_loss` | `clip_eps_high` (clip-higher) and two aggregation methods. With a chunked forward pass, it takes the token count of the full batch, so the result is equal to one pass over the full batch. Returns KL, clip ratio, and the mean of ρ |
| Hand-written per-token sampling (02/03) | `sample_group`: `zero.generate` + KV cache, G copies of the same prompt, stops at `<|im_end|>` | A real model needs the chat template, a KV cache, and a stop token. When a response does not use the full budget, the code adds `<|im_end|>` back and trains on it |
| One `verify` function | `zero/post/envs/tool_env.py::score_tool_calls` | 6 simulated APIs, schema validation, two match methods (AST and execution), one-to-one matching, 8 anti-hacking guards. `generate_tasks` does not overlap with the frozen dev set |
| The naive / fixed reward of 03 | Guard #8 (`_bare_call`) + 0 for an empty reply | The hack that we really saw in the smoke test. Test: `tests/test_tool_env.py::test_guard_untagged_call_json` |
| Fixed learning rate, hand-written loop | `run_grpo` + `LoopState` | Driven by the configuration, resume from checkpoints, logs (`log.jsonl`), gradient clipping, BF16 autocast (on a GPU, not verified yet). Task selection depends only on (seed, step), so a resumed run is reproducible |
| Prints a few metrics | Logs at each step: reward, format-correct rate, call rate, length, KL, clip ratio, zero-variance groups, time | The monitoring table of Section 8. Entropy and the held-out success rate are not in the training loop yet |

Configurations: `configs/tiny/grpo.toml` (CPU smoke test: G = 8, 4 problems per step, β = 0.02, 20 steps); `configs/main/grpo.toml` (main line: G = 16, 64 problems per step, so 1024 responses, `max_new_tokens = 512`, ε = 0.2, `clip_eps_high = 0.28`, β = 0, learning rate 1e-6, 500 steps; not verified on a GPU yet).

**Throughput and verl (GOAL.md, Section 3.3)**: The GRPO of `zero` is a readable single-process implementation. Its sampling has no continuous batching. If the throughput is not sufficient in step 2, use [verl](https://github.com/verl-project/verl) (or Xiaomi's [XiaomiMiMo/verl](https://github.com/XiaomiMiMo/verl)) for the real training.

But first, do a parity check with `zero` on the same small task. Use the same exported HF model, the same batch of `tool_env` tasks (exported to JSONL with a fixed seed), and the same G / ε / β / learning rate / aggregation method. Wrap `score_tool_calls` with the custom reward interface of verl. First, compare the advantages and the loss of the first step (the same batch of samples must give the same values, digit by digit). Then compare the mean-reward curves of the first 50 steps. (The names of the matching verl configuration items are in the module docstring of `zero/post/grpo.py`; we collected them from the verl documentation, to be verified.)

## Main-line progress

### Tiny-configuration demo (CPU, `configs/tiny`, about 1.3M parameters)

> This section is a **tiny-configuration demo**. It shows only that the code runs and logs the metrics as expected. It does not show any result of the main-line model.

> **Note:** The numbers in this section come from the smoke test **before** we fixed the tool-call scorer (Chapter 19, Section 6). They are the real output of that run. We ran the smoke test again after the fix (`uv run python -m zero.smoke --out out/smoke_final`). The data, pretraining, mid-training, and SFT stages gave exactly the same results.
>
> The distillation samples that passed verification changed from 1 (the nonsense sentence) to 0. Thus the downstream numbers of DPO, GRPO, and evaluation changed. For example, the tool-call call_exact of the same SFT model changed from 0.133 to 0.100: the model did not change, but the scoring became stricter. GRPO against SFT is still judged "tie". When you run it yourself, use your own results.

The smoke test (`uv run python -m zero.smoke`, results in `out/smoke/`) ran the GRPO stage for 10 steps from the weights after DPO. Each step had 4 problems × G = 8 = 32 responses, and the stage took 114 seconds:

| Step | Mean reward | Format-correct rate | Call rate | Mean length | KL |
|---:|---:|---:|---:|---:|---:|
| 1 | −0.19 | 0.50 | 0.34 | 25.5 | 0.0000 |
| 3 | −0.40 | 0.47 | 0.47 | 39.7 | 0.0017 |
| 5 | −0.04 | 0.72 | 0.72 | 37.7 | 0.0042 |
| 6 | −0.34 | 0.44 | 0.44 | 31.7 | 0.0074 |
| 9 | −0.02 | 0.72 | 0.72 | 29.7 | 0.0121 |
| 10 | −0.07 | 0.62 | 0.62 | 31.7 | 0.0120 |

Over the 10 steps, the reward goes up and down (each step has only 32 responses). The difference between the first and the last step is not proof that the model "learned something". The evaluation stage does a paired bootstrap on 30 frozen tool_dev problems. The call_exact difference of GRPO against SFT is **−0.033, 95% CI [−0.100, +0.000], judged "tie"** (sft 0.133, grpo 0.100).

In the same table, GRPO is 0.133 higher than SFT on 30 toy multiple-choice questions (CI [+0.033, +0.267]). But GRPO did not train on multiple-choice questions, and the start DPO model already had 0.200 on these questions. We do not interpret this difference.

To reproduce (use a different output folder and run only 3 steps; about 1.5 minutes on our machine):

```bash
uv run python -m zero.post.grpo --config configs/tiny/grpo.toml \
  --set train.init_from=out/smoke/dpo/ckpt --set data.tokenizer=out/smoke/tokenizer.json \
  --set train.max_steps=3 --set train.out_dir=/tmp/grpo_tiny
# step 1 | reward -0.186 | format 0.47 | call 0.28 | len 24.8 | kl 0.0000
# step 2 | reward -0.428 | format 0.31 | call 0.31 | len 33.8 | kl 0.0002
# step 3 | reward -0.610 | format 0.22 | call 0.22 | len 40.2 | kl 0.0016
```

We found guard 8 with this configuration (Section 7.1).

### To be added after GPU training

- The training curves of the main-line GRPO (reward, format-correct rate, call rate, length, KL, entropy) and the tool-call score on the dev set.
- The results of the parity check with verl (advantages/loss of the first step on the same small task, the reward curve of the first 50 steps).
- A comparison on the dev set: clip-higher on/off, and β = 0 / 0.02 (this decides the main-line configuration).
- New hacks found in training and the guards for them. (We fixed the two loopholes in Section 6 in step 1. We add here the new hacks that we find in the step 2 training.)
- Cost (recorded in `runs/ledger.md`).

Before step 2 starts, we suggest two additions to `zero` (this chapter does not change `zero`; we report them). First, log the policy entropy in the training loop. Second, add online filtering by difficulty (skip groups that are all correct or all wrong, and sample more), or at least filter the problems offline by the pass rate of the SFT model.

## Frontier notes

- **clip-higher (DAPO)**: Make the upper clipping bound 1+ε_high wider than the lower bound (DAPO uses 0.2 / 0.28). Then low-probability "exploration" tokens can be pushed higher, and entropy collapse decreases. MiMo-7B and OLMo 3 adopt it. By the rule of this course, one more leading family with explicit adoption is necessary. Thus it is "to be verified", and in the main-line configuration it is a switch that you can turn off.
- **No division by the standard deviation (Dr. GRPO)**: A problem with a very small group std (almost all correct or all wrong) gets a larger weight. This causes a "difficulty bias". Dr. GRPO suggests subtracting only the mean, and also removing the normalization by response length. OLMo 3 adopts "no division by the std". We found only this one leading model that adopts it explicitly.
- **GSPO**: GSPO changes the importance ratio from a per-token ratio to the (length-normalized) likelihood ratio of the full response. It also clips at the sequence level. The Qwen team says that GSPO made the RL training of MoE models stable, and that later versions of Qwen3 use it. Adoption by other families is not verified yet.
- **Asynchronous / off-policy RL**: Decouple sampling from training to increase throughput (in-flight updates in OLMo 3, slime in GLM-4.5, Seamless Rollout in MiMo). This needs an extra importance correction (for example, truncated importance sampling). It is an infrastructure method, and the main line of this course does not use it.

## Adopters and sources

| Technique | Adopters (technical reports / model cards) |
|---|---|
| GRPO (group baseline, no value model) | DeepSeek: [DeepSeekMath](https://arxiv.org/abs/2402.03300) (proposed it), [DeepSeek-R1](https://arxiv.org/abs/2501.12948); Qwen: [Qwen3 technical report](https://arxiv.org/abs/2505.09388), Section 4.2; GLM: [GLM-4.5](https://arxiv.org/abs/2508.06471), Section 3.2 (GRPO without KL); OLMo: [OLMo 3](https://arxiv.org/abs/2512.13961), Section 4.4 (OlmoRL is based on GRPO); Xiaomi: [MiMo-7B](https://arxiv.org/abs/2505.07608), Section 3.3 |
| RLVR with a group-mean baseline (not the original GRPO form) | Kimi: [Kimi K2](https://arxiv.org/abs/2507.20534), Section 3.2.3 (the objective of K1.5) |
| Verifiable rewards (RLVR) | [Tülu 3](https://arxiv.org/abs/2411.15124) (gave the name, uses PPO), DeepSeek-R1, Qwen3, Kimi K2 (Verifiable Rewards Gym), OLMo 3, GLM-4.5, MiMo-7B |
| Token-level aggregation | MiMo-7B (Eq. 1), OLMo 3 (Section 4.4.1), GLM-4.5 (Figure 7); origin: [DAPO](https://arxiv.org/abs/2503.14476) |
| Remove the KL term | GLM-4.5, MiMo-7B, OLMo 3; DAPO |
| Filter problems by difficulty / dynamic sampling | MiMo-7B, OLMo 3, GLM-4.5, Kimi K2, Qwen3 (all above) |
| k3 KL estimate | DeepSeekMath Eq. 4, DeepSeek-R1 Eq. 2; [Schulman, Approximating KL](http://joschu.net/blog/kl-approx.html) |
| Rule-based rewards for tool calls | GLM-4.5 (Section 3.4, Function Calling RL), Qwen3 (Agent Ability in General RL), Kimi K2 (tool-use environments); environment design follows [XiaomiMiMo/verl](https://github.com/XiaomiMiMo/verl) |
| To be verified / frontier | clip-higher: DAPO, MiMo-7B, OLMo 3; Dr. GRPO: [paper](https://arxiv.org/abs/2503.20783), OLMo 3; GSPO: [paper](https://arxiv.org/abs/2507.18071) (later versions of Qwen3) |

Note: the post-training of SmolLM3 uses SFT + preference optimization, without RL (footnote 28 of the OLMo 3 report says the same). Thus we do not list SmolLM3 as a GRPO adopter.

## Guided questions

Ask Claude Code these questions. Continue until you can explain the answers in your own words:

1. In the log-derivative trick, why does the step "∇π = π · ∇log π" let us estimate the expectation with samples? If the reward itself depends on θ (for example, the policy model is its own judge), where does the derivation fail?
2. The GRPO advantage is the same for all tokens. In a response of 30 tokens, only one token is wrong, but it gets the same penalty as the other 29 tokens. Is this reasonable? What problem does a process reward model (PRM) try to solve? Why does DeepSeek-R1 say that the PRM did not succeed?
3. Why does clipping have no effect when μ = 1, but the gradient is still not zero? Change μ in `02` to 1 and to 4, and see how the clip ratio changes.
4. Why is the k3 estimate exp(d) − d − 1 never negative? Compare it with the direct KL estimate log π_θ − log π_ref. What are the benefits of each?
5. Divide by the group std. Take two problems: on one, 1 of 8 responses is correct; on the other, 4 are correct. What is the advantage of the correct response in each problem? Is this a "difficulty bias"?
6. Score a two-turn task: "check the weather, then answer in one sentence". Is it sufficient to check only the final answer? Must the tool call in the middle get its own score? How do the "step-wise rule-based RL" and the "end-to-end multi-turn RL" of GLM-4.5 divide the work?

## Hands-on tasks

**Task 1 (basic)**: In `01_reinforce_bandit.py`, change the baseline to "leave-one-out": the baseline of each sample is the mean reward of the other samples in the same batch. Compare the variance. This baseline differs from the group mean of GRPO only by a constant factor G/(G−1) (Appendix A of the Dr. GRPO paper has the derivation).

**Task 2 (core)**: In `02_grpo_from_scratch.py`, change the accuracy of the teacher on carry problems from 30% to 5% and to 0%, and run again. At 0%, can GRPO still learn the carry? Why? (Hint: at step 1, is there a group with both correct and wrong answers?) What does this tell you about the start point that RL needs?

**Task 3 (challenge)**: Find one more loophole in the "fixed reward" of `03_reward_hacking.py`. Design an output that gets a higher mean score under the fixed reward than random guesses with `<call> 字 d </call>`, but is not the correct behavior. Then add a guard, and run again to make sure that the true success rate does not decrease. Use the same idea on `zero/post/envs/tool_env.py`. Read `score_tool_calls`, find a hack that the current guards do not cover, and write it as a test in the style of `tests/test_tool_env.py` (try it locally first; do not commit it).

## Go deeper: CS336

- Lecture 16: RLVR in post-training (reinforcement learning with verifiable rewards, policy gradient, GRPO).
- Assignment 5 (Alignment and Reasoning RL): implement SFT, expert iteration, and GRPO on math reasoning tasks, and compare different loss normalizations and baselines. The optional second part is DPO (see Chapter 18).

Course page (lecture notes and recordings): <https://cs336.stanford.edu/>

## References

- Williams. *Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning* (REINFORCE), 1992: <https://link.springer.com/article/10.1007/BF00992696>
- Schulman et al. *Proximal Policy Optimization Algorithms*, 2017: <https://arxiv.org/abs/1707.06347>
- Shao et al. *DeepSeekMath* (GRPO, k3 KL, the unified view of gradient coefficients), 2024: <https://arxiv.org/abs/2402.03300>
- DeepSeek-AI. *DeepSeek-R1*, 2025: <https://arxiv.org/abs/2501.12948>
- Qwen Team. *Qwen3 Technical Report*, 2025: <https://arxiv.org/abs/2505.09388>; *Group Sequence Policy Optimization*: <https://arxiv.org/abs/2507.18071>
- Kimi Team. *Kimi K2: Open Agentic Intelligence*, 2025: <https://arxiv.org/abs/2507.20534>
- GLM-4.5 Team. *GLM-4.5*, 2025: <https://arxiv.org/abs/2508.06471>
- Xiaomi LLM-Core. *MiMo: Unlocking the Reasoning Potential of Language Model*, 2025: <https://arxiv.org/abs/2505.07608>
- XiaomiMiMo/verl (the five types of RL environment and the scorers of MiMo-V2.6): <https://github.com/XiaomiMiMo/verl>
- Olmo Team. *Olmo 3*, 2025: <https://arxiv.org/abs/2512.13961>
- Lambert et al. *Tülu 3* (RLVR), 2024: <https://arxiv.org/abs/2411.15124>
- Yu et al. *DAPO*, 2025: <https://arxiv.org/abs/2503.14476>
- Liu et al. *Understanding R1-Zero-Like Training: A Critical Perspective* (Dr. GRPO), 2025: <https://arxiv.org/abs/2503.20783>
- Schulman. *Approximating KL Divergence*: <http://joschu.net/blog/kl-approx.html>
- Lilian Weng. *Reward Hacking in Reinforcement Learning*, 2024: <https://lilianweng.github.io/posts/2024-11-28-reward-hacking/>
- verl (HybridFlow): <https://github.com/verl-project/verl>
- Hands-On Modern Reinforcement Learning (references.md): <https://walkinglabs.github.io/hands-on-modern-rl/preface/intro>
- CS336: <https://cs336.stanford.edu/>

**Next chapter**: This is the end of the training of the main-line model. Chapter 20 does the final evaluation with the preregistered protocol. It compares the model with all public models of the same size on tool-call benchmarks such as BFCL. The model is "ahead" only when its lead is larger than the confidence interval. Then we quantize the model to GGUF, build a tool-call assistant that runs on a laptop, write the model card, and release the model with honest results.
