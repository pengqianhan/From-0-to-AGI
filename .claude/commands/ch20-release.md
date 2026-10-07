---
description: "Chapter 20 self-check: release — Gate 3 decisions with the preregistration, ahead/tie/behind and honest reports, HF standard format and chat template, block-wise quantization Q8_0/Q4_K/Q4_K_M, memory = parameters × bits, llama.cpp/Ollama/vLLM, parity checks, model card and license (第 20 章自检：发布——闸门 3 按预注册判定、超过/持平/落后与如实报告、HF 标准格式与 chat template、分块量化 Q8_0/Q4_K/Q4_K_M、内存 = 参数 × bit、llama.cpp/Ollama/vLLM、对拍、模型卡与许可证)"
---

# Chapter 20 self-check: Release

The learner typed `/ch20-release`. They finished Chapter 20 (`chapters/20-release/`). Help them check if they understand it. Do not give them the answers.

**Language**: Use the language of the learner. If the learner writes in Chinese, ask the questions and give feedback in Chinese. The Chinese text of the chapter is in `README.zh.md`. Write short, clear sentences (see `docs/STYLE_GUIDE.md`).

**Method**: Ask the questions below one at a time. Wait for the learner to answer in their own words. Then give feedback: tell them what is correct, what is not correct, and how to make their understanding deeper. Do not ask all the questions at the same time.

---

## Questions (from easy to difficult)

**Level 1: the decision rule (concept)**

Ask the learner:
> On BFCL, our model is 1.2 points higher than Qwen3.5-0.8B. The 95% confidence interval of the paired bootstrap is [−0.4, +2.8]. With the preregistration, must this cell say "ahead", "tie", or "behind"? If the opponent has a thinking mode and a non-thinking mode, which score do we use for the comparison?

Expected answer: the interval crosses 0, so the cell can only say "tie". It must not say "slightly ahead". The decision is "ahead" only if the lower bound > 0, and "behind" if the upper bound < 0. We test both modes of the opponent and use the higher score (`compare_to_opponent`). Extra credit: the learner says that the decision rule, the benchmarks, the templates, and the decoding parameters are all frozen before training, and Gate 3 only runs them as written.

---

**Level 2: memory and blocks (intuition)**

Ask the learner:
> A model has 689.5M parameters. About how large is it in bf16, in Q8_0 (8.5 bits), and with Q4_K for all matrices (4.5 bits)? Why is Q8_0 8.5 bits and not 8 bits? Why do we use "one scale for each 32 numbers", and not one scale for the whole matrix?

Expected answer: about 1.28 GiB, 0.68 GiB, and 0.36 GiB (Q4_K_M is about 0.40 GiB, because some tensors use Q6_K). The extra 0.5 bit is one fp16 scale for each block: 16 bits / 32 numbers. With one scale for the whole matrix, one outlier makes the scale large, and all the other small weights are quantized to 0. (On the matrix that the chapter made, the INT4 error is 93% with one scale and 11% with blocks.)

---

**Level 3: find the problem**

Ask the learner:
> A colleague quantizes the final model to Q4_K_M. He tries a few sentences on a laptop and "feels no difference". Now he wants to write "quantization is lossless" in the model card. What is wrong with this statement? How must we verify it before the release? Also, when he exported the HF folder, he used a different chat template that "looks about the same". What happens?

Expected answer: a feeling from a few sentences is not evidence. On the small model of the chapter, 4 bits change the top-1 prediction at about 6% of the positions. On Llama 3 8B, Q4_K_M also increases the PPL from 6.233 to 6.407. Use `llama-perplexity` to measure PPL / KLD on our own development set. Compare the scores of bf16 and Q4_K_M on the tool-calling development set, and write the results in the model card truthfully (checklist items C7–C8). The template must be identical to the training template, character by character. A tool-calling model is very sensitive to the format. A different template can stop it from producing `<tool_call>`, and the evaluation scores also become wrong. The smoke test uses `apply_chat_template` and compares the result with our template character by character.

---

**Level 4: transfer**

Ask the learner:
> One month after our release, a 0.8B model is released after the freeze date. On BFCL, it is clearly ahead of our model. How must we change the model card and the announcements? Also, suppose our weights use Apache-2.0, and the training data contains FineWeb-Edu with ODC-By. Must we do the attribution duty? Where do we write it?

Expected answer: in the section "Opponents added after release", report its comparison with our model truthfully, also if it is stronger. The original conclusion from the preregistration does not change, because it is a claim about the frozen opponent list. But the announcements must not make claims beyond the decision table, such as "the strongest model of its size". When we change the model card, we write the date and the reason. The licenses are two different matters. The author decides the weight license. We must do the attribution duty for the data, whatever license we select. The data table of the model card lists the data sets, their licenses, and the attribution.

---

## Rules for feedback

- If the answer is correct: say so. Then ask a deeper "why" question (for example, "Why does Q4_K_M give more bits to the output layer?").
- If the answer is not correct: do not give the answer. Give a hint. For example, ask the learner to run `code/01_blockwise_quant.py` or `code/02_quantize_tiny_model.py`, change the block size or the number of bits, and look at the result. Then let them think again.
- If the learner says "I do not know": ask them to guess first. A wrong guess is better than no guess.

When the learner passes all four levels, tell them that Part 4 is complete. They can continue to Chapter 21 in Part 5 (`chapters/21-kv-cache-ledger/`). After Chapter 21, they can check themselves with `/ch21-kv-cache`.
