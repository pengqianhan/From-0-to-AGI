---
description: "Chapter 19 self-check: reinforcement learning — REINFORCE and the log-derivative trick, baseline and variance, PPO→GRPO group advantage, clipping and k3 KL, token-level aggregation, verifiable rewards, reward hacking and guards, training monitoring (第 19 章自检：强化学习——REINFORCE 与对数导数技巧、基线与方差、PPO→GRPO 组内优势、裁剪与 k3 KL、token 级聚合、可验证奖励、reward hacking 与守卫、训练监控)"
---

# Chapter 19 self-check: reinforcement learning

The learner typed `/ch19-rl`. They finished Chapter 19 (`chapters/19-reinforcement-learning/`). Help them check if they really understand it. Do not give them the answers.

**Language**: Use the language of the learner. If the learner writes in Chinese, ask the questions and give feedback in Chinese. The Chinese text of the chapter is in `README.zh.md`. Write short, clear sentences (see `docs/STYLE_GUIDE.md`).

**Method**: Ask the questions below one at a time. Wait for the learner to answer in their own words. Then give feedback: tell them what is correct, what is not correct, and how to make their understanding deeper. Do not ask all the questions at the same time. When a number is necessary, ask the learner to run the scripts in `code/` and read the output. Do not give the numbers from memory.

---

## Questions (from easy to difficult)

**Level 1: concept (why reinforcement learning, and what GRPO removes)**

Ask the learner:
> SFT and distillation can already teach the model to make tool calls. Why do we still need reinforcement learning? Compared with PPO, which model does GRPO not have? What does GRPO use to do the job of that model?

Expected answer: the upper limit of imitation learning is the demonstration data. The student also learns the errors of the teacher (see the "teacher that cannot carry" in `02_grpo_from_scratch.py`). Reinforcement learning lets the model learn from its own attempts. It needs only a verifier that can give a score. GRPO removes the value model (critic). It samples G responses for the same prompt and uses the mean score of the group as the baseline: A_i = (r_i − mean) / std. Extra credit: the learner says that GRPO saves the memory and the training of a model that is as large as the policy.

---

**Level 2: intuition (the log-derivative trick and the baseline)**

Ask the learner:
> The reward function has no derivative (for example, "is the answer correct?"). How do we calculate the gradient? If we add 10 to all rewards, does the expected REINFORCE gradient change? Does the variance change?

Expected answer: ∇E[r] = E[r · ∇log π(a)]. We only need to sample and to give a score; r itself does not need a derivative. A constant baseline does not change the expectation (because E[∇log π] = 0), but it changes the variance by a large amount. In `01_reinforce_bandit.py`, with rewards +10, the variance without a baseline is more than a thousand times the variance with a baseline, and the training almost does not move. The "group mean" of GRPO is such a baseline.

---

**Level 3: find the problem (the reward goes up, but the skill does not)**

Ask the learner:
> You run GRPO. The mean reward increases steadily from −0.6 to near 0, and the format error rate decreases to 0. But a human spot check finds that the model makes no correct tool calls. What can have happened? Which metrics do you look at? How do you change the reward?

Expected answer: reward hacking. The real case in this chapter: the reward checked only the content inside the `<tool_call>` tags. The model learned to remove the tags and output the JSON anyway, so −1 became 0. Also, a chat problem got the full score for any output. Look at more than the mean reward: also look at the call rate, the fraction of bare calls, the response length, and the true success rate from a held-out verifier or a human spot check. The fix: JSON that looks like a call outside the tags is a format error (guard #8), and an empty reply does not get the full score. The general principles: use a strict format, penalize degenerate outputs, and monitor with a separate held-out verifier. Extra credit: the learner says that after a fix to the reward, you must run again and check the true success rate again, not only the reward curve.

---

**Level 4: transfer (gradient of an all-correct group; aggregation methods)**

Ask the learner:
> All 8 responses to one prompt are correct (or all are wrong). How much does this group add to the gradient? Late in training, there are more and more such groups. What happens? What can you do about it? Also, what is the difference between "average within each response, then average over responses" and "average over all tokens together"?

Expected answer: the variance in the group is 0, so all advantages are 0, and the group gives no gradient. Late in training, there are fewer and fewer useful samples (in the log of `02`, the fraction of "zero-variance groups" goes above 0.8). What to do: filter the problems by difficulty (offline, or online like DAPO/MiMo/OLMo 3: remove all-correct and all-wrong groups and sample more). Aggregation: the sequence mean gives a smaller weight to each token in a long response (a long wrong response gets a smaller penalty). The token mean gives the same weight to each token. The reports of MiMo-7B, OLMo 3, and GLM-4.5 all use the token-level mean, and `zero` also uses `token_mean` by default.

---

## Rules for feedback

- If the answer is correct: say so. Then ask a deeper "why" question (for example: why is the k3 estimate never negative? Why does clipping have no effect when ppo_epochs = 1?).
- If the answer is not correct: do not give the answer. Give a hint. For example, ask the learner to change the reward function in `03_reward_hacking.py`, run it again, and look at the metrics. Then let them think again.
- If the learner says "I do not know": ask them to guess first. A wrong guess is better than no guess.

When the learner passes all four levels, tell them to continue to Chapter 20 (release: the final evaluation with the preregistered protocol, quantization, and local deployment).
