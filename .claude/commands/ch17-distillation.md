---
description: "Chapter 17 self-check: distillation — soft labels and dark knowledge, temperature and τ², KD gradient p_S − p_T, the same vocabulary, sequence-level distillation, forward/reverse KL, on-policy distillation, rejection sampling and execution check, teacher license (第 17 章自检：蒸馏——软标签与暗知识、温度与 τ²、KD 梯度 p_S − p_T、同一个词表、序列级蒸馏、前向/反向 KL、在线策略蒸馏、拒绝采样与执行验证、教师许可证)"
---

# Chapter 17 self-check: distillation

The learner typed `/ch17-distillation`. They finished Chapter 17 (`chapters/17-distillation/`). Help them check if they understand it. Do not give them the answers.

**Language**: Use the language of the learner. If the learner writes in Chinese, ask the questions and give feedback in Chinese. The Chinese text of the chapter is in `README.zh.md`. Write short, clear sentences (see `docs/STYLE_GUIDE.md`).

**Method**: Ask the questions below one at a time. Wait for the learner to answer in their own words. Then give feedback: tell them what is correct, what is not correct, and how to make their understanding deeper. Do not ask all the questions at the same time.

The chapter uses a Chinese example. The context is "今天天气很" ("Today the weather is very"). The candidate next characters are 好 (good), 热 (hot), 冷 (cold), 晴 (sunny), 猫 (cat), and 跑 (run). Keep these characters as they are, and give the English meaning when it helps.

---

## Questions (from easy to difficult)

**Level 1: concept — what does a soft label add?**

Ask the learner:
> The context is "今天天气很" ("Today the weather is very"), and the next character in the training text is "好" ("good"). What does the one-hot label tell the student? What does the soft label of the teacher tell the student (好 0.607, 热 0.223, 冷 0.100, 晴 0.067, 猫 0.002, 跑 0.001)? When you increase the temperature τ, how does this distribution change, and what does not change?

Expected answer: the one-hot label says only "the answer is 好", and its entropy is 0. The soft label also says "热, 冷, and 晴 are possible; 猫 and 跑 are not", and it gives their relative sizes (dark knowledge). A larger τ makes the distribution flatter, and the low-ranked candidates get more probability (p(晴)/p(猫) decreases from 44.7 to 2.6 at τ = 4). But the ranking does not change. Extra credit: "to minimize KL(p_T‖p_S) is the same as to minimize the cross-entropy with the teacher distribution as the label; one-hot is a special case of it".

---

**Level 2: intuition — the gradient and τ²**

Ask the learner:
> What is the gradient of the KD loss L = τ²·KL(p_T^τ ‖ p_S^τ) for the student logits? Compare it with the gradient of the cross-entropy from Chapter 5, p − onehot. What is the same, and what is different? Why do we multiply by τ²?

Expected answer: ∂L/∂z_S = τ·(p_S^τ − p_T^τ). It is also "the prediction of the student minus the target"; only the target changes from one-hot to the teacher distribution. Thus KD also pushes up the second-best answers that the student underestimates (热, 晴), instead of pushing all other answers down. Without τ², the size of the gradient decreases as 1/τ². One 1/τ comes from the derivative of z/τ. The other comes from the smaller difference, because both distributions become flatter. With τ², you do not need to tune the learning rate and α again when you change the temperature. Extra credit: "for a very large τ, KD is approximately a direct match of the logits".

---

**Level 3: find the problem — why can the main line use only sequence-level distillation?**

Ask the learner:
> The main-line model uses its own trained vocabulary of 65,536 tokens from Chapter 13. For step 2, we plan to use a Qwen model with an Apache-2.0 license as the teacher. Why can we not do logits distillation or on-policy distillation? If we had used the Qwen tokenizer from the start, what would we get, and what would it cost?

Expected answer: logits distillation compares the two distributions position by position and dimension by dimension. Position t must be the same "next token", and dimension v must be the same token: the tokenizer must be the same. On-policy distillation needs the teacher to score the tokens of the student, so it has the same requirement. Two vocabularies cut the same sentence into different numbers of tokens with different boundaries, so the positions do not align (the kd_loss of zero raises a shape error, and run_distill compares the tokenizer hashes). With the Qwen3 vocabulary of 151,936 tokens, logits and on-policy distillation become possible. But with the main-line shape, the embedding grows from 83.9M to 194.5M, and the total parameters grow from 689.5M to 800.1M. This is over the 0.8B limit. Also, that vocabulary was not trained on our mix of Chinese, English, and code. Thus the main line uses sequence-level distillation: the teacher writes and the student does SFT. The teacher only needs to generate text.

---

**Level 4: transfer — forward/reverse KL, verifiers, and licenses**

Ask the learner (you can ask in two parts):
> (a) For one tool-call question, the teacher has two correct ways to write the call, and the capacity of the student is limited. What can the student learn with sequence-level distillation only (forward KL)? What can it learn with on-policy distillation (reverse KL)?
> (b) You have a set of teacher data in which all samples passed the execution check. You plan to train on it directly. What else must you check? Also: a teacher calls tools very well, but its license requires that "a model trained on its outputs has a name that starts with its name". Can the main line use this teacher?

Expected answer: (a) Forward KL is mode covering. The student must give probability to both forms of the teacher. With limited capacity, it can "learn half of each" and produce a form that is neither of them. (In the two-peak experiment, the valley gets 24% of the probability, but the teacher has only 1.6% there.) Reverse KL is mode seeking. The student picks one form, does it well, and gives up the other. The cost is less diversity, and the start point decides which mode. (b) A verifier guarantees only the things that it checks. For tasks that need no tool, the old verifier checked only "no unnecessary tool call", not the content. (The only sample that passed in the smoke test was "讲一句鼓励的话" ("Say something encouraging") → "坚下云，气温 28°C" ("<nonsense words>, temperature 28°C").) The same execution result does not mean correct arguments (the weekday date that was wrong by 7 days). Also check the pass rate by task type, and do decontamination: 13-gram checks, and BFCL function names / schemas. For licenses: the main line uses only Apache-2.0 / MIT teachers (GOAL.md 3.3). A naming requirement such as the one in the Llama Community License passes to the student, so we do not use such a teacher. Each sample records the name, version, and license of the teacher, and `check_license` stops the run when the license is not confirmed.

---

## Rules for feedback

- If the answer is correct: say so. Then ask a deeper "why" question (for example, "If the teacher itself is wrong on one type of question, what happens in on-policy distillation?").
- If the answer is not correct: do not give the answer. Give a hint, and ask them to run the code again: `01_soft_labels.py` (temperature table, gradient comparison), `03_forward_reverse_kl.py` (two peaks), `04_rejection_sampling.py` (funnel and nonsense answers), `05_shared_vocab.py` (parameter cost).
- If the learner says "I do not know": ask them to guess first. A wrong guess is better than no guess.

When the learner passes all four levels, tell them to continue to Chapter 18 (`chapters/18-preference-alignment/`): preference alignment, from RLHF to DPO.
