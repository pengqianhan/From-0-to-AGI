---
description: "Chapter 18 self-check: preference alignment — preference data, Bradley–Terry, the reward model as binary classification, the RLHF objective and the KL leash, reward hacking, PPO preview, DPO derivation, implicit reward and gradient, β and learning rate, chosen probability goes down (第 18 章自检：偏好对齐——偏好数据、Bradley–Terry、奖励模型即二分类、RLHF 目标与 KL 缰绳、reward hacking、PPO 铺垫、DPO 推导、隐式奖励与梯度、β 与学习率、chosen 概率下降)"
---

# Chapter 18 self-check: preference alignment — from RLHF to DPO

The learner typed `/ch18-dpo`. They finished Chapter 18 (`chapters/18-preference-alignment/`). Help them check if they understand it. Do not give them the answers.

**Language**: Use the language of the learner. If the learner writes in Chinese, ask the questions and give feedback in Chinese. The Chinese text of the chapter is in `README.zh.md`. Write short, clear sentences (see `docs/STYLE_GUIDE.md`).

**Method**: Ask the questions below one at a time. Wait for the learner to answer in their own words. Then give feedback: tell them what is correct, what is not correct, and how to make their understanding deeper. Do not ask all the questions at the same time.

---

## Questions (from easy to difficult)

**Level 1: concepts — preference data and the reward model**

Ask the learner:
> What does one preference pair look like? How does the Bradley–Terry model change it into a probability? Why do we say that training a reward model "is the binary cross-entropy of Chapter 5"?

Expected answer: a pair of answers for the same prompt (chosen y_w, rejected y_l). The judge can be a human, a stronger model, or a program that scores automatically. P(y_w ≻ y_l) = σ(r_w − r_l). The loss is −log σ(r_w − r_l). This is the binary cross-entropy with logit r_w − r_l and a label that is always 1. The initial loss is ln 2. Extra credit: only the score difference has a meaning, so the gradient of the reward-model bias is always 0.

---

**Level 2: intuition — the KL leash**

Ask the learner:
> The RLHF objective is E[r] − β·KL(π‖π_ref). What happens if you set β to 0? In the bandit of `02` in this chapter, why does the reward-model score increase all the time while the true quality first increases and then decreases?

Expected answer: at β = 0, the policy only chases the reward-model score. It moves to regions that the reward model never saw, where the scores are not reliable (reward hacking). In the toy, the reward model learned "long = good". It gives the highest score to the 900-token padded answer that it never saw. When β is too small, the policy puts everything on that answer, and the true quality falls to −0.5. At β = 0.5, the true quality is highest (1.124). The KL term keeps the policy near the SFT model, where the reward model is reliable. Extra credit: InstructGPT uses a KL penalty at each token (β = 0.02). PPO solves this objective with sampling, a value baseline, and a clipped ratio.

---

**Level 3: find the problem — derivation and gradient**

Ask the learner:
> Start from the optimal solution of RLHF and derive the DPO loss. The derivation contains a Z that we cannot calculate. How does Z disappear? Then tell what the DPO gradient does to chosen and to rejected, and what sets its strength.

Expected answer: the optimal solution is π* = π_ref·exp(r/β)/Z, because the objective = β·log Z − β·KL(π‖π*). Solve for r: r = β·log(π*/π_ref) + β·log Z. Put r into Bradley–Terry, which uses only the reward difference, and the two β·log Z terms cancel. Replace π* with π_θ and do maximum likelihood. The result is −log σ(β[(log π_θ(y_w) − log π_ref(y_w)) − (log π_θ(y_l) − log π_ref(y_l))]). The gradient pushes chosen up and rejected down, both with strength β·σ(−h), where h is the implicit reward difference. The worse the ranking error, the harder the push.

Follow-up question:
> During DPO training, the loss goes down and the margin goes up. Does this show that the model became better?

Expected answer: no. DPO only cares about the difference, so the probability of chosen can also go down. In `05` of this chapter, with wrong answers that differ by only a little, the log-probability of chosen went from −1.045 to −1.662, and the held-out probability of the correct answer went from 0.353 to 0.223. Nemotron-4 also reports that both go down. With a learning rate that is too large, the margin goes very high, but the format breaks (lr = 1e-2: margin 4.26, fraction of well-formed samples 0.473). In the main-line smoke test, after DPO with lr = 5e-4, the GRPO format accuracy fell from 0.16 to 0. Look at real metrics on a held-out set. Fixes: add an NLL term on chosen (Llama 3 uses a coefficient of 0.2) and mask the format tokens.

---

**Level 4: transfer — the main-line model**

Ask the learner:
> The DPO of the main-line model must improve the general chat quality and must not damage tool calling. How do you prepare the preference data? What does `make_env_preferences` in `zero/post/dpo.py` do? How do you set the learning rate and β, and which metrics do you watch?

Expected answer: for tool calling, use on-policy preference pairs. The current policy samples several answers, and the verifiable reward of tool_env scores them. The highest score (a full score) becomes chosen; otherwise, the reference solution becomes chosen. The lowest score becomes rejected. For general chat, use open preference data whose license allows it (HelpSteer3 CC-BY-4.0, UltraFeedback MIT; the Tülu 3 mixture has subsets that are not for commercial use). Or use an open model whose license allows it as the judge. Do 13-gram decontamination. The learning rate is much smaller than for SFT (main-line default 5e-7; Zephyr / Tülu 3 5e-7). Start β at 0.1 and sweep it on the development set. Watch the tool-calling format accuracy, the chat evaluation on the development set, and whether chosen_reward also goes down. Do not watch the training margin.

---

## Rules for feedback

- If the answer is correct: say so. Then ask a deeper "why" question.
- If the answer is not correct: do not give the answer. Give a hint. For example, ask the learner to change β in `02_rlhf_kl.py` or the learning rate in `04_toy_dpo.py` and run it. Then let them think again.
- If the learner says "I do not know": ask them to guess first. A wrong guess is better than no guess.

When the learner passes all four levels, tell them to continue to Chapter 19 (`chapters/19-reinforcement-learning/`, reinforcement learning: GRPO and verifiable rewards). After Chapter 19, they can check themselves with `/ch19-rl`.
