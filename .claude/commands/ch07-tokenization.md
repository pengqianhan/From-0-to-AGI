---
description: "Chapter 7 self-check: language modeling and tokenization — chain rule, byte-level BPE, bigram, bits-per-byte (第 7 章自检：语言建模与分词——链式法则、字节级 BPE、bigram、bits-per-byte)"
---

# Chapter 7 self-check: language modeling and tokenization

The learner typed `/ch07-tokenization`. They finished Chapter 7 (`chapters/07-tokenization-language-model/`). Help them check if they understand it. Do not give them the answers.

**Language**: Use the language of the learner. If the learner writes in Chinese, ask the questions and give feedback in Chinese. The Chinese text of the chapter is in `README.zh.md`. Write short, clear sentences (see `docs/STYLE_GUIDE.md`).

**Method**: Ask the questions below one at a time. Wait for the learner to answer in their own words. Then give feedback: tell them what is correct, what is not correct, and how to make their understanding deeper. Do not ask all the questions at the same time.

---

## Questions (from easy to difficult)

**Level 1: concepts**

Ask the learner:
> In one sentence, what does a language model do? Write the formula that splits the probability of a full text into parts. At which step of this formula does the bigram model make an approximation?

Expected answer: a language model gives the probability distribution of the next token from the earlier text (it is a classification over the vocabulary). The chain rule `p(x_1…x_T) = Π p(x_t | x_<t)` is an identity. The bigram model approximates `p(x_t | x_<t)` with `p(x_t | x_{t−1})`: it looks only at the previous token. The training objective is the cross-entropy of the correct token at each position. If the learner says "the chain rule is an approximation", correct them.

---

**Level 2: intuition**

Ask the learner:
> On a mixed corpus of Chinese, English, and code, the first merge of the hand-written BPE is `\xef\xbc`. It is not a full character. Why? In the example text `学而时习之`, the BPE with a vocabulary of 1024 splits the character `习` ("practice") into two tokens. Does this lose information?

Expected answer: BPE looks only at how often a pair of bytes occurs. The UTF-8 encodings of the full-width punctuation marks (`，`, `：`, `；`, and others) share the first two bytes `ef bc`. Together they occur most often, so BPE merges them first. For the same reason, many Chinese characters share a 2-byte prefix. When BPE did not learn `习` as a full character, the character falls back to byte pieces (`\xe4\xb9` + `\xa0`). All 256 bytes are in the vocabulary, so decoding joins the bytes again and restores the text. No information is lost. This is the advantage of the byte level over the character level, which has `<unk>`.

---

**Level 3: find the error**

Ask the learner:
> On the same English validation set, the byte-level bigram has a perplexity of 12.7, and the BPE bigram has 37.0. Someone says: "So the byte-level model is better." What is wrong with this conclusion? How must we compare the two models?

Expected answer: perplexity (and the loss) is "per token". One BPE token covers about 1.76 bytes on average, so each token is harder to predict. We cannot compare the two numbers directly. Compare bits-per-byte: `bpb = total nats / (ln2 × total bytes) = (bits/token) ÷ (bytes/token)`. The number of bytes of the same text does not depend on the tokenizer. In this chapter, we measured a bpb of 2.956 for BPE and 3.669 for the byte level: BPE is about 19% better. Extra credit: "nanochat calculates `val_bpb` in this way and does not count special tokens".

---

**Level 4: transfer**

Ask the learner:
> You must choose a vocabulary for a small bilingual (Chinese and English) model with at most 0.8B parameters and a width of d = 1280. A classmate says: "Use the 248K vocabulary of Qwen3.5. It has the best compression." What do you tell the classmate? Name at least two costs. Also tell how you would design an experiment to decide the vocabulary size.

Expected answer: embedding parameters = V × d, and 248,320 × 1280 ≈ 320M. This takes a large part of the 0.8B budget, and the Transformer layers lose these parameters. With a larger vocabulary, each token has fewer training examples, so the model does not learn rare tokens well. The output softmax also becomes more expensive. The gain in compression decreases as the vocabulary grows (in this chapter, from 16K to 32K, the sequences became only 7% shorter). The experiment: train tokenizers with several values of V on real Chinese, English, and code data, and measure bytes/token. Then train small models with the same compute and compare the validation **bpb** (not the loss). At the same time, calculate the embedding parameters. Chapter 13 does exactly this. The main line of this course uses 65,536 for now.

---

## Rules for feedback

- If the answer is correct: say so. Then ask a deeper "why" question (for example: how large is the table of a trigram model? Why can we not store it?).
- If the answer is not correct: do not give the answer. Give a hint. For example, ask the learner to change `vocab_size` in `code/02_bpe.py`, or to print bits/token and bytes/token in `code/03_bigram.py`. Then let them think again.
- If the learner says "I do not know": ask them to guess first. A wrong guess is better than no guess.

When the learner passes all four levels, tell them to continue to Chapter 8 (`chapters/08-attention/`, attention: each position can see all earlier tokens). After Chapter 8, they can check themselves with `/ch08-attention`.
