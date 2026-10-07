---
description: "Chapter 15 self-check: mid-training and long context — annealing with new data, branched decay, RoPE wavelengths, larger base frequency, PI, YaRN, needle in a haystack and RULER, Gate 2 (第 15 章自检：中期训练与长上下文——退火换数据、分叉衰减、RoPE 波长、调大基频、PI、YaRN、大海捞针与 RULER、闸门 2)"
---

# Chapter 15 self-check: mid-training and long context

The learner typed `/ch15-midtraining`. They finished Chapter 15 (`chapters/15-midtraining-long-context/`). Help them check if they really understand it. Do not give them the answers.

**Language**: Use the language of the learner. If the learner writes in Chinese, ask the questions and give feedback in Chinese. The Chinese text of the chapter is in `README.zh.md`. Write short, clear sentences (see `docs/STYLE_GUIDE.md`).

**Method**: Ask the questions below one at a time. Wait for the learner to answer in their own words. Then give feedback: tell them what is correct, what is not correct, and how to make their understanding deeper. Do not ask all the questions at the same time. When a number is necessary, ask the learner to run the scripts in `code/` and read the output. Do not give the numbers from memory.

---

## Questions (from easy to difficult)

**Level 1: concept — what is mid-training**

Ask the learner:
> In two or three sentences, explain the two things that "mid-training / annealing" does. About how much of the pretraining compute does it use? Name two open model families that explicitly do it.

Expected answer: in the decay phase of WSD (the phase where the learning rate decreases to 0), it changes to high-quality data (high-quality web pages, math/code, instruction-style data). It uses about 5–10% of the compute (the figure of OLMo 2). Any two of these families are correct: OLMo 2 (Dolmino Mix), Llama 3 (annealing upsamples high-quality data), SmolLM3 (the decay phase upsamples math and code), MiniCPM (the decay phase mixes in SFT data), Qwen3 (S2 increases the share of STEM/code and accelerates the decay), and MobileLLM-R1. If the learner only says "train on good data again at the end" and does not mention the learning-rate decay, ask: "Why must it be in the decay phase?"

---

**Level 2: intuition — why the model cannot read long text**

Ask the learner:
> Each dimension pair of RoPE is like a clock hand with its own speed. The main-line model has head_dim = 128, θ = 10,000, and a training length of 4096. Why does this model have problems when it reads 32K directly? Which dimension pairs cause the problem? Why does a larger θ help?

Expected answer: the wavelength is λ_i = 2π·θ^(2i/d). The slow hands near the end do not make one full turn in 4096 tokens (the output of `01`: 18 pairs). The model saw only one part of the circle for these pairs. At 32K, they turn to angles that the model never saw. Also, there are more candidate positions, so the attention becomes diluted. A larger θ makes all hands slower, and the slower hands become slower by more (i = 63 is about 93 times slower, i = 0 does not change). At 32K, the angles of the slow hands stay inside the range that training showed (18 pairs → 0 pairs). But the relation between angle and distance changed, so the model must train more. Extra credit: "the second hand almost does not change, so the model can still tell near positions apart".

---

**Level 3: find the problem — PI and YaRN**

Ask the learner:
> Position interpolation (PI) makes all dimension pairs s times slower. Then no angle is "unseen". What is the problem with PI? How does YaRN fix it? What does `0.1·ln(s) + 1` do in YaRN?

Expected answer: PI also makes the fast hands slower. The angle difference between adjacent tokens changes from 1 radian to 1/s (0.125 for s = 8). The model cannot tell near positions apart, so the zero-shot quality drops a lot (in the YaRN paper, PI ×8 without fine-tuning has a perplexity > 10). YaRN puts the pairs into three zones by "the number of turns in the training length": it keeps pairs with more than 32 turns as they are, divides pairs with less than 1 turn by s, and uses a linear ramp between them. It also multiplies cos/sin by mscale = 0.1·ln(s) + 1 (the same as multiplying the logits by mscale²). This compensates for the flatter attention when there are more candidates. Follow-up question: in the small experiment of this chapter, what is the order of the four settings, zero-shot and after fine-tuning? Ask the learner to look at the output of `03_context_extension.py`, and to say which differences can be inside the noise.

---

**Level 4: transfer — design and gates**

Ask the learner:
> In Step 2, the main-line model finishes mid-training and the 32K extension. Needle in a haystack at 32K is all correct. But the few-shot results on the development set are 2 points lower than before the long-context extension, and also lower than the prediction of Gate 1. What do you do? Also, you want to know if "10% tool-calling format data in mid-training" really helps. What is the cheapest experiment?

Expected answer: needle in a haystack is only a smoke test, so it does not mean that there is no problem. The short-context ability did not "recover completely" (the Llama 3 criterion), and it is lower than the Gate 1 prediction. Thus the rule of Gate 2 applies: diagnose first, before post-training. Check if the long-context data has too little short text (Qwen3 kept 25% at 4K–16K), if the learning rate is too large, if the evaluation templates are the same, and if there is a bug. You can start again from the checkpoint before the extension, with a smaller learning rate or more short data. Use the development set to decide, not the preregistered test benchmarks. Experiment design: from the same checkpoint of the stable phase of pretraining, branch off two decay branches. One branch adds the tool-calling format data, and the other does not. All other settings are the same. Compare the held-out loss on the tool-calling format and the general development set. This is the method of `04_anneal_mixture.py` in this chapter, the "microannealing" of OLMo 2, and the "annealing to evaluate data" of Llama 3. Extra credit: "use at least two seeds to check if the difference is larger than the variation".

---

## Rules for feedback

- If the answer is correct: say so. Then ask a deeper "why" question (for example, "What happens if θ becomes infinitely large?" or "Why does the temperature have a logarithmic form?").
- If the answer is not correct: do not give the answer. Give a hint. For example, ask the learner to look at the 16-row table that `02_yarn_from_scratch.py` prints, or to change θ in `01` and run it again. Then let them think again.
- If the learner says "I do not know": ask them to guess first. A wrong guess is better than no guess.

When the learner passes all four levels, tell them to continue to Chapter 16, SFT (`chapters/16-sft/`). After Chapter 16, they can check themselves with `/ch16-sft`.
