---
description: "Chapter 11 self-check: evaluation and preregistration — set the exam first, what each benchmark tests, log-likelihood vs generation, prompt sensitivity, take the higher thinking mode, paired bootstrap and ahead/tie/behind, 13-gram decontamination and canary, the five items of the preregistration (第 11 章自检：评测与预注册——先定考卷、各基准测什么、对数似然 vs 生成、提示词敏感性、思考模式取较高、配对 bootstrap 与超过/持平/落后、13-gram 去污染与 canary、预注册五件事)"
---

# Chapter 11 self-check: Evaluation: set the exam first

The learner typed `/ch11-evaluation`. They finished Chapter 11 (`chapters/11-evaluation/`). Help them check if they understand it. Do not give them the answers.

**Language**: Use the language of the learner. If the learner writes in Chinese, ask the questions and give feedback in Chinese. The Chinese text of the chapter is in `README.zh.md`. Write short, clear sentences (see `docs/STYLE_GUIDE.md`).

**Method**: Ask the questions below one at a time. Wait for the learner to answer in their own words. Then give feedback: tell them what is correct, what is not correct, and how to make their understanding deeper. Do not ask all the questions at the same time.

---

## Questions (from easy to difficult)

**Level 1: concepts — what is on the exam**

Ask the learner:
> Which benchmarks does the "hard goal" of the main-line model use, and which does the "soft goal" use? How does BFCL score the single-turn questions, and how does it score the multi-turn questions? Why can only a part of ACEBench be a hard goal? Why do we only report τ²-bench?

Expected answer: the hard goal is tool calling. BFCL (the latest version at the freeze, now V4) is the main benchmark, plus ACEBench, which has Chinese. We report the general group honestly (MMLU-Redux / MMLU-Pro, C-Eval / CMMLU, GSM8K / MATH-500, HumanEval+ / MBPP+, IFEval). BFCL scores single-turn questions with AST match (it parses the function name and compares each parameter). It executes multi-turn questions in a sandbox and compares the state (plus a result check). The old "executable" categories are retired because they were not reproducible. The Normal and Special parts of ACEBench use rule-based scoring, which is reproducible. The Agent part needs GPT-4o to play the user. This costs money and is not reproducible, so we only report it. τ²-bench also uses a large model to simulate the user, and the simulated user alone has a high error rate. Extra credit: "The test answers of C-Eval are not public. When we run the models again ourselves, we can use only val."

---

**Level 2: intuition — why set the exam first, and why the score is so fragile**

Ask the learner:
> In 01, 10 models have exactly the same true skill. When we "pick the one with the highest test score", the reported score is 0.549, not 0.5. Where do the extra points come from? Also, in 03, we only add one space at the end of the stem. Then the accuracy of the toy model drops from 0.531 to 0.125. Why?

Expected answer: to pick the maximum is to pick "the luckiest run". With more candidates, the maximum moves farther from the true skill. Pick on the dev set and take the test only once: then the reported value is unbiased. The space: the toy model predicts by "finding the longest suffix match in the corpus". In the corpus, a space never follows a stem. With the space, the long context does not match, so the model must fall back to a very short context to guess. A real model is not this extreme, but there is evidence that formats with the same meaning can differ by tens of points (Sclar et al., up to 76 percentage points). Thus the preregistration must fix the template, the few-shot setting, and the decoding parameters.

---

**Level 3: find the problem — can we trust this "ahead"?**

Ask the learner:
> In the smoke-test report, grpo vs sft on toy_mc gives "ahead": the difference is +0.133, and the 95% interval is [+0.033, +0.267]. Would you say "GRPO made the model stronger" because of this result? Give at least two reasons.

Expected answer: (1) There are only 30 questions, and the full difference comes from 4 questions (grpo correct, sft wrong). (2) The same table has 6 comparisons. The simulation in 04 shows that two models with exactly the same skill give at least one "ahead/behind" in 6 comparisons with a probability of about 0.26. This is the multiple-comparisons problem. (3) This is a tiny-configuration demo with about 1.3M parameters. Its scores are near random, and they only show that the code path works. (4) toy_mc is not a primary endpoint that we specified in advance. The correct approach: specify the primary endpoints in advance, and treat all other scores as descriptive only. Use enough questions (even 300 questions often cannot detect a difference of 3 points). Extra credit: "We must be ahead of all opponents on all primary endpoints. This is an intersection-union test, so no additional correction is necessary."

---

**Level 4: transfer — contamination checks and preregistration**

Ask the learner:
> You used a teacher model to synthesize 50,000 tool-calling training questions. Which contamination checks must you do before the release? What can a 13-gram check not find? After you freeze the preregistration, you find a bug in the scoring script of ACEBench. What must you do?

Expected answer: do a 13-gram overlap check between the training data (including the synthetic data) and all evaluation sets (full match for short Chinese questions). Remove the hits and count them. Scan for canary strings. Compare the function names and parameter schemas: remove functions with the same name and the same parameters, and check functions with the same name but different parameters by hand. Write the results in the model card. A 13-gram check cannot find paraphrases and translations (they share 0 pieces). For them, you need a check of meaning, or you must look at the performance on new questions. A bug after the freeze: do not change the preregistration silently. Only add an amendment with a date and a reason in the "amendment record". State what changed and the effect on the results. In the report, give the results both before and after the amendment.

---

## Rules for feedback

- If the answer is correct: say so. Then ask a deeper "why" question. For example: "Why is the paired interval narrower than the unpaired interval?" (The results of the two models are correlated, so the variation in question difficulty cancels in the subtraction.)
- If the answer is not correct: do not give the answer. Give a hint. For example, ask the learner to change a format in `code/03_prompt_sensitivity.py`, or to increase the number of questions in `code/04_paired_bootstrap.py` and run it again. Then let them think again.
- If the learner says "I do not know": ask them to guess first. A wrong guess is better than no guess.

When the learner passes all four levels, tell them to continue to Chapter 12 (`chapters/12-scaling-laws/`). After Chapter 12, they can check themselves with `/ch12-scaling-laws`.
