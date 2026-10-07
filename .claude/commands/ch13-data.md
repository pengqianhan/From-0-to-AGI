---
description: "Chapter 13 self-check: data — open data sets and licenses, heuristic filters, MinHash LSH dedup, model-based quality filtering, synthetic rephrasing, mixture ablations, 13-gram decontamination, vocabulary size and bits-per-byte (第 13 章自检：数据——开放数据集与许可证、启发式过滤、MinHash LSH 去重、基于模型的质量过滤、合成改写、配比消融、13-gram 去污染、词表大小与 bits-per-byte)"
---

# Chapter 13 self-check: data

The learner typed `/ch13-data`. They finished Chapter 13 (`chapters/13-data/`). Help them check if they understand it. Do not give them the answers.

**Language**: Use the language of the learner. If the learner writes in Chinese, ask the questions and give feedback in Chinese. The Chinese text of the chapter is in `README.zh.md`. Write short, clear sentences (see `docs/STYLE_GUIDE.md`).

**Method**: Ask the questions below one at a time. Wait for the learner to answer in their own words. Then give feedback: tell them what is correct, what is not correct, and how to make their understanding deeper. Do not ask all the questions at the same time. When a question needs numbers, let the learner run the scripts in `chapters/13-data/code/`. Do not calculate the numbers for them.

---

## Questions (from easy to difficult)

**Level 1: the pipeline (concepts)**

Ask the learner:
> What steps take raw Common Crawl web pages to shards that you can feed to a model? What type of document does each step mainly remove? Which steps are cheap, and which step is the most expensive? Why must the expensive step come later?

Expected answer: text extraction → language identification → heuristic rules (Gopher/C4/FineWeb: word count, symbol ratio, stop words, duplicate lines, line-end punctuation, …) → exact dedup + MinHash near dedup → model-based quality filtering (the educational-value classifier of FineWeb-Edu, the fastText classifier of DCLM) → (optional) synthetic rephrasing → decontamination → tokenization and shards, with a record of the license and provenance of each source. Rules and hashes take a few microseconds per document. Model-based scoring must run a neural network (FineWeb-Edu used 6000 H100 GPU hours to score 15T tokens). Thus the cheap steps make the data smaller first. Extra credit: the learner can tell where each type of junk is removed in the funnel of `01–05`. Navigation pages, spam, and garbled pages are removed by the rules. Reposts are removed by dedup. Most word salad passes the rules and is removed by the classifier.

---

**Level 2: the math of MinHash (intuition + formula)**

Ask the learner:
> The Jaccard similarity of the 5-gram sets of two documents is s. Why is the probability that "the two signatures are equal at one position" exactly s? Split the signature into b bands of r rows. What is the probability that the pair becomes a candidate? With b=16 and r=8, what is it for s=0.8 and for s=0.5? To make the threshold "steeper", how do you change b and r, and what is the cost?

Expected answer: with a random hash, the element of A∪B with the smallest hash is in A∩B with the probability |A∩B|/|A∪B| = s. In that case, the two minimums are equal. The candidate probability is 1 − (1 − s^r)^b. With b=16 and r=8: about 0.947 for s=0.8, and about 0.061 for s=0.5 (the table and the measurement of `03_minhash.py` agree). The inflection point is at about (1/b)^(1/r). When r and b both increase, the curve becomes steeper (RefinedWeb uses 450 × 20). The cost is that you must calculate and store more hashes (FineWeb selected 14 × 8 to save compute). Extra credit: the learner knows why the "take the low 32 bits" line in the code of this chapter is necessary. When a·x+b does not exceed p, the hash increases monotonically with x. Then all hash functions select the same minimum element, and the positions of the signature are no longer independent.

---

**Level 3: find the problem (data ablations and decontamination)**

Ask the learner:
> In `06_quality_ablation.py`, the model trained on the noisy crawl has a lower training loss, but a higher validation bpb. What does this tell you? Also: what type of leak can 13-gram decontamination not catch? What problem occurs if you decrease n to 5? In `05_decontam.py`, why are the "other hits" at n=13 not false positives?

Expected answer: the noisy data contains many duplicates and template-like navigation pages and spam. They are easy to predict, so they decrease the training loss. But they do not help on normal held-out text. A low training loss does not mean a good model. To compare data, look at bpb on the same clean validation set, and check if the difference is larger than the variation from the random seed. 13-grams cannot catch paraphrased test questions (one word changes in every few words, so no 13 consecutive words match). A small n flags common phrases ("me to the sight of") as leaks and removes good documents by mistake. The "other hits" at n=13 occur because the same Song ci poem is in the corpus two times ("Three Hundred Song Ci Poems" and "Complete Song Ci"), and the test question came from one copy. This is a real leak. It also shows that document-level dedup cannot find "paragraph-level duplicates".

---

**Level 4: transfer (the main-line decisions)**

Ask the learner:
> The shape of the main-line model is fixed (28 layers, width 1280, tied embeddings). Only the vocabulary size changes. Why can you not select the vocabulary by "bytes/token" alone? Use the table of Section 10: when the vocabulary increases from 64K to 128K, how much does the compute per token increase, and how much does the compression improve? Together, does "the compute to read the same amount of text" increase or decrease? Why did the main line select its value? Also: the license page of one source says Apache-2.0. Why does `configs/main/data.toml` still mark it as "to be verified"?

Expected answer: a larger vocabulary compresses better, but the embedding parameters V×1280 and the lm_head matmul increase linearly. Compare FLOPs/byte = (FLOPs/token) ÷ (bytes/token). Also, the total parameters must not be more than 0.8B (the Qwen vocabulary of 151,936 at this width is over the limit). For the exact numbers, use the output of `08_vocab_size.py` that the learner runs and Section 10 of the README. The answer must use the two constraints, "compute/byte" and "the parameter limit". It must also mention the conclusion of Tao et al. 2024: the optimal vocabulary of a small model is smaller than that of a large model, but overtraining makes the optimum larger. Licenses: the Chinese part of Ultra-FineWeb collects many upstream corpora with different licenses, so the label on the data set page does not mean that each upstream source permits it. The license of DCLM is CC-BY-4.0, but the card says "research use only". The Stack-Edu page has no license field: you must check the terms of The Stack v2 and the detected_licenses of each file. Thus the downloader refuses "to be verified" sources by default.

---

## Rules for feedback

- If the answer is correct: say so. Then ask a deeper "why" question (for example, "Why did FineWeb find that global dedup made the data worse?").
- If the answer is not correct: do not give the answer. Give a hint. For example, ask the learner to change `bands` in `03_minhash.py` or `n` in `05_decontam.py` and run the script. Then let them think again.
- If the learner says "I do not know": ask them to guess first. A wrong guess is better than no guess.

When the learner passes all four levels, tell them to continue to Chapter 14 (`chapters/14-pretraining-engineering/`). Now they have the data and the tokenizer. The next step is to really feed hundreds of billions of tokens into 8 GPUs.
