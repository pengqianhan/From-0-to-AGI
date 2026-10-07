# Chapter 13: Data — From web pages to a training data set

**English** · [中文](README.zh.md)

> **Goal**: After this chapter, you can name the steps that take pretraining data from web pages to shards, and tell what each step removes. You can write MinHash LSH deduplication by hand and calculate "the probability that a document pair with similarity s is caught". You can use controlled experiments with small models (a paired comparison of bits-per-byte) to decide if a filter or a mixture is useful. You can use 13-grams to check if the training data saw the test questions. You can also select the vocabulary size of the main-line model from two accounts: "bytes per token" and "compute per byte".

📺 **Video**: Not published yet. To render it on your computer, run `bash chapters/13-data/video/build.sh`.
🧪 **Self-check**: After the chapter, type `/ch13-data` in Claude Code.

---

In the previous chapter, we used scaling laws to set the size of the main-line model (about 0.69B parameters) and its pretraining token budget (about 400 billion). This chapter answers the next question: **where do these 400 billion tokens come from, and what do they look like?** The web has an almost unlimited amount of text. But most of it is navigation bars, ads, garbled text, and identical reposts, and it also contains test questions from many evaluation sets. This chapter shows how to change this text into a data set that is clean, legal, well mixed, and free of test questions. Then it trains a tokenizer that fits this data set.

All code in this chapter runs on a CPU. The first 5 scripts take about 1 minute in total. The two ablation experiments train small models and need several minutes of CPU time each. When the machine is busy, the wall-clock time is much longer.

```bash
uv run python chapters/13-data/code/01_noisy_crawl.py          # make a "noisy crawl": real documents + seven types of junk + leaked test questions
uv run python chapters/13-data/code/02_heuristic_filter.py     # language identification + Gopher / C4 / FineWeb rules
uv run python chapters/13-data/code/03_minhash.py              # MinHash LSH from scratch; check the S-curve formula
uv run python chapters/13-data/code/04_quality_classifier.py   # toy version of "model-based quality filtering"
uv run python chapters/13-data/code/05_decontam.py             # 13-gram decontamination
uv run python chapters/13-data/code/06_quality_ablation.py     # ablation 1: noisy vs. filtered (about 3 min of CPU time on one thread)
uv run python chapters/13-data/code/07_mixture_ablation.py     # ablation 2: three mixtures (about 4 min of CPU time on one thread)
uv run python chapters/13-data/code/08_vocab_size.py           # vocabulary size: compression × parameters and compute
uv run python -m zero.data.pipeline --config configs/tiny/data.toml   # production pipeline (tiny configuration, about half a minute)
```

## 1. Data is the largest lever

First, look at two examples of about the same size as the main-line model.

- **MobileLLM-R1** (Meta, 2025): 950M parameters. Its pretraining used only 4.2T tokens. This is 11.7% of the 36T tokens of the small Qwen3 models. But it equals or beats Qwen3-0.6B on many reasoning benchmarks. The paper tests the assumption that "reasoning needs very large amounts of data (more than 10T)". Its conclusion: about 2T tokens of carefully selected open data, resampled at designed ratios, are sufficient. Its "leave-one-out" experiment also found this: when you remove FineWeb-Edu (web data filtered for educational value), knowledge, math, and code **all** become worse. General web data acts like glue that holds the domains together.
- **Puro-2B** (Tsinghua, 2026-08): 2B parameters and about 1.4T tokens, with only public data. Its fitted cost curve says that about **USD 4.4K** of compute is sufficient to match Qwen2-1.5B. Its data recipe does not use its own quality scores. It uses **proxy experiments**. Start from the same Qwen3-0.6B checkpoint, and continue training on each candidate data slice for about 8.4B tokens. Then look at the "capability vector" of 15 benchmarks. One finding shows the effect well. Take the data set DCLM and sort it by quality score. The top slice ranked 3rd of 39 candidate slices, but the slice 25% further down ranked only 11th.

The Llama 3 technical report also says it directly. Compared with Llama 2, the architecture almost did not change. The report says: "the performance improvements come mainly from improvements in data quality and diversity, and from the scale of training". At this scale, the architecture of Chapter 9 and the hyperparameters of Chapter 12 make less difference than "which data we feed". Thus the main-line model takes no architecture risk (GOAL.md 3.3) and puts its effort into the data.

## 2. Open data sets and their licenses

GOAL.md 3.3 requires this: "Use only public data that the license permits. Record the source and the license of each data set." The table shows the candidates for the main line. We checked each row against the data set card on Hugging Face (2026-09-26):

| Data set | Size (as on the data set card) | Language | License | How it was filtered |
|---|---|---|---|---|
| [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) | 1.3T tokens (also a 5.4T version with score ≥2) | English | ODC-By 1.0 | FineWeb (Common Crawl → trafilatura extraction → fastText language identification → Gopher/C4/FineWeb rules → MinHash within each dump) + an educational-value classifier (Llama-3-70B labels 460K pages → train a small classifier → keep scores ≥3, remove 92%) |
| [DCLM-baseline 1.0](https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0) | 4T tokens, 3B documents | English | CC-BY-4.0 (the card says "research use only"; see below) | RefinedWeb-style heuristics → Bloom filter dedup → fastText classifier (positive examples: OpenHermes 2.5 instruction data + highly voted ELI5 answers) |
| [FineWeb-2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) (`cmn_Hani` subset) | Chinese: 636M documents, 1.6TB of parquet | Chinese (and 1000+ languages) | ODC-By 1.0 | Multilingual version of the FineWeb pipeline (rules and dedup tuned for each language), **no** model-based scoring |
| [Ultra-FineWeb](https://huggingface.co/datasets/openbmb/Ultra-FineWeb) | English about 1T, Chinese about 120B tokens | English, Chinese | The page says Apache-2.0; the Chinese part comes from many upstream corpora (to be verified) | "Verification-based" filtering of FineWeb and Chinese FineWeb-edu-v2: low-cost training experiments select the positive and negative examples, then a fastText classifier trains on them (the core web data of MiniCPM4/5) |
| [Stack-Edu](https://huggingface.co/datasets/HuggingFaceTB/stack-edu) | 125B tokens, 15 programming languages | Code | The page has no license field and points to the terms of The Stack v2; each file has `detected_licenses` (to be verified) | StarCoder2 training set → one educational-value classifier per language (StarEncoder, labels from Llama-3-70B) → keep scores ≥3 (Java ≥2). **Contains only file ids**: get the content from the S3 of Software Heritage |
| [FineMath](https://huggingface.co/datasets/HuggingFaceTB/finemath) | FineMath-3+ 34B tokens (4+ about 9.6B) | English | ODC-By 1.0 | Recall math pages from Common Crawl → math classifier (labels from Llama-3.1-70B) → scores ≥3 |
| [Nemotron-CC](https://data.commoncrawl.org/contrib/Nemotron/Nemotron-CC/index.html) (v2 on HF needs an application) | 6.3T tokens (4.4T real + 1.9T synthetic) | English (v2 adds multilingual QA) | Common Crawl terms of use; v2 uses the NVIDIA data agreement (training is permitted; redistribution of the raw data is not) | An ensemble of three classifiers puts the data into 20 buckets → rewrite low-quality text in Wikipedia style; generate QA / summaries / knowledge lists from high-quality text (Section 7) |

Some reminders. You must deal with each of them when you write the model card:

- **A license is more than one field.** ODC-By requires attribution, and the Common Crawl terms of use also apply. The license of DCLM is CC-BY-4.0, but the data set card says "intended for research use only". Before you release a model, make sure how the two statements relate. The Ultra-FineWeb page says Apache-2.0, but its Chinese part collects upstream corpora such as IndustryCorpus2, WuDao, SkyPile, WanJuan, and CCI3. These upstream corpora have different terms. (The Puro-2B paper says specially that SkyPile uses the Skywork community license.) The Nemotron-CC paper also says that they did not put the FineWeb-Edu classifier into the ensemble "because of license issues": the training labels of that classifier come from Llama 3.
- **"Open" does not mean "downloadable".** Stack-Edu gives only file ids. You must use the script on the data set card to get the content from the S3 bucket of Software Heritage (this needs AWS credentials). Nemotron-CC-v2 on HF needs an application.
- All rows of this table except Nemotron-CC are registered in [`zero/data/sources.py`](../../zero/data/sources.py). (Nemotron-CC is not in the main line for now: v2 needs an application, and it does not permit redistribution of the raw data. This conflicts with the goal "publish the full data recipe".) If the license of a source is not fully verified, the downloader refuses to download it by default (Section 11).

## 3. The pipeline, and a "noisy crawl"

From Common Crawl to training shards, the common method is this sequence of steps (FineWeb, DCLM, Nemotron-CC, and Llama 3 are similar):

```
web HTML → text extraction → language identification → heuristic rules → exact dedup + near dedup → model-based filtering
         → (synthetic rephrasing) → decontamination → tokenization → shards + mixture → training
```

The order is important: **put the cheap steps first**. Rules and hashes take a few microseconds per document. Model-based scoring must run a neural network: FineWeb-Edu used 6000 H100 GPU hours to score 15T tokens. The cheap steps make the data smaller first. Then we can pay for the expensive steps.

We cannot download real web data in the build environment (and it is too large). Thus [`01_noisy_crawl.py`](code/01_noisy_crawl.py) makes its own "noisy crawl". Shakespeare and Song ci (Chinese poems of the Song dynasty) from `assets/tiny_corpus` are the good documents. Then the script adds the types of junk that are common on web pages. Each document has a label, so each later step can check if it removed the correct documents. The real world has no such label. This is why data work is difficult.

| Type | Documents | What it is |
|---|---:|---|
| good | 1015 | Real documents (about 1500 characters each) |
| contaminated | 16 | A good document with a "test question" in the middle (12 verbatim, 2 with changed case and punctuation, 2 paraphrased) |
| exact_dup | 80 | Verbatim reposts; only the line ends and the spaces at line ends are different |
| near_dup | 80 | Reposts with some changed words, plus a header and a footer such as "转载自……" ("Reposted from …") and "分享到……" ("Share to …") |
| nav | 60 | Navigation pages: `首页 \| 关于我们 \| 联系方式 …` ("Home \| About us \| Contact …") |
| spam | 40 | Keyword stuffing: `免费下载高清在线观看…` ("free download, HD, watch online …"), `cheap best price buy now…` |
| garbled | 30 | Garbled text: UTF-8 decoded as Latin-1 (`æå®ä¸`), or a block of symbols |
| salad | 80 | The words of each line of a good document in random order (in Chinese, the characters); the punctuation at the line end stays in place |

In total, the crawl has 1401 documents and 2.76 MB. We also hold out 84 English and 30 Chinese clean documents as the validation set. The 60 test questions (30 Chinese, 30 English) come from these held-out documents.

## 4. Text extraction, language identification, and heuristic rules

**Text extraction** sets the quality of the raw material. Common Crawl gives two formats: WARC (the raw HTML) and WET (plain text that Common Crawl extracted). The FineWeb ablations found that text extracted again from WARC with trafilatura trains clearly better models than WET (WET contains too many navigation bars and menus). Nemotron-CC found that jusText keeps 28.6% more high-quality tokens.

**Language identification** usually uses the fastText model lid.176. FineWeb keeps only documents with "English score ≥ 0.65". Script `02` of this chapter uses a 10-line replacement. More than 30% Chinese characters is Chinese, and more than 50% ASCII letters is English. All else is "other" and goes out. All garbled pages are removed at this step.

**Heuristic rules** are a set of cheap statistics with thresholds. [`02_heuristic_filter.py`](code/02_heuristic_filter.py) implements the rules of three papers from scratch, with the threshold values of the papers:

```python
if not 50 <= n <= 100_000:                                   bad.append("gopher_word_count")
if sum(w.lower() in stop for w in words) < 2:                bad.append("gopher_stop_words")      # the/be/to/of/and/that/have/with
if sum(c for c in counts.values() if c > 1) / nl > 0.3:     bad.append("gopher_dup_lines")       # fraction of duplicate lines
if "lorem ipsum" in text.lower() or "{" in text:             bad.append("c4_lorem_or_curly")
if sum(ln.rstrip().endswith(END_PUNCT) for ln in lines) / nl <= 0.12:  bad.append("fineweb_line_punct")
```

Chinese has no spaces between words, so one Chinese character counts as one "word". The stop words become the Chinese function words "的了是在和有也就不都而与之其以". The result:

| Type | Before the filter | Removed | Removal rate |
|---|---:|---:|---:|
| good | 1015 | 177 | 17% |
| contaminated | 16 | 2 | 12% |
| exact_dup / near_dup | 80 / 80 | 21 / 8 | 26% / 10% |
| nav / spam / garbled | 60 / 40 / 30 | 60 / 40 / 30 | **100%** |
| salad | 80 | 12 | 15% |

The rules remove all navigation pages, spam, and garbled pages. But 85% of the word salad passes. Its character distribution, line lengths, and punctuation are the same as in real documents, so the rules cannot see that "this is not language".

One more number deserves a second look: **the rules removed 17% of the good documents by mistake**, and 152 of them are Shakespeare. The cause is the "duplicate lines" rule of Gopher. In a play, name lines such as `QUEEN MARGARET:` occur many times, so the fraction of duplicate lines easily goes above 30%. These thresholds were tuned on web pages, and on plays, poems, and code they remove good documents. The production `zero/data/quality.py` also keeps only 1110 documents of the same set. If you use the web rules on code, they remove 96% of the code in the tiny corpus. (Code has no stop words, and its lines do not end with a period.) Thus the pipeline selects the rules by source:

- Web pages: Gopher/C4/FineWeb.
- Code: the three rules of the Codex paper (mean line length > 100, longest line > 1000, alphanumeric fraction too low).
- Data sets that upstream already filtered (FineWeb-Edu, DCLM): no second filter.

Nemotron-CC even found that MMLU increases by 2 more points when it turns off the heuristic rules for high-quality documents. Not everything that the rules remove is bad.

## 5. Deduplication: exact hashes + MinHash LSH

The web has very many duplicates: reposts, mirror sites, and template pages. Duplicate data wastes compute and makes the model "memorize" text. (Lee et al. 2021 found that dedup greatly decreases the fraction of training text that the model repeats word for word.) Duplicates also make test-question leaks larger. Deduplication (dedup) has two levels:

1. **Exact dedup**: normalize the whitespace, calculate a hash, and keep only one document for each hash. This is cheap, but one different character is sufficient to hide a duplicate.
2. **Near dedup**: measure how similar two documents are with the **Jaccard similarity** of their sets of character 5-grams: `J(A, B) = |A∩B| / |A∪B|`. A comparison of all pairs is O(N²). For billions of documents, this is not possible. MinHash and LSH solve this problem.

**MinHash**: take a random hash function h, and find the element of the set with the smallest hash value. The element with the smallest hash in A∪B is in the intersection with the probability |A∩B|/|A∪B|. In that case, the minimums of the two sets are equal. Thus:

```
P( min h(A) = min h(B) ) = J(A, B)
```

Use 128 different hash functions to get 128 minimums. These are the **signature** of the document. The fraction of equal positions in two signatures is an estimate of the Jaccard similarity. The core of [`03_minhash.py`](code/03_minhash.py) is less than 60 lines:

```python
def signature(self, x):            # x: the 5-grams of a document (crc32 changes each one into an integer)
    return (((x[:, None] * self.a + self.b) % PRIME) & MASK).min(axis=0)
```

The `& MASK` at the end (take the low 32 bits) is necessary. When `a·x + b` is less than the prime p, "mod p" does nothing. Then the hash value increases monotonically with x, and all 128 hash functions select the same minimum element. The positions of the signature are then no longer independent. I forgot this operation in my first version. Then the "measured" values below did not agree with the formula: at s = 0.3, the measured candidate rate was 0.22, but the formula gave only 0.001.

**LSH (locality-sensitive hashing)**: split the 128-position signature into b bands of r rows each (16 × 8 in this chapter). If one band is completely identical, the two documents go into the same bucket and become a "candidate pair". Then the Jaccard estimate from the signatures checks the pair again. The probability that a pair with similarity s becomes a candidate is:

```
P(s) = 1 − (1 − s^r)^b          # one band identical: s^r; no band identical: (1 − s^r)^b
```

This is an S-shaped curve. Its inflection point is at about `(1/b)^(1/r)`:

| s | b=16, r=8 (this chapter / zero) | b=14, r=8 (FineWeb) | b=450, r=20 (RefinedWeb) | Measured (this chapter's setting, 1000 pairs per row) |
|---:|---:|---:|---:|---:|
| 0.5 | 0.061 | 0.053 | 0.000 | 0.069 |
| 0.6 | 0.237 | 0.211 | 0.016 | 0.202 |
| 0.7 | 0.613 | 0.565 | 0.302 | 0.615 |
| 0.8 | 0.947 | 0.924 | 0.995 | 0.949 |
| 0.9 | 1.000 | 1.000 | 1.000 | 1.000 |

The measured values agree with the formula. Larger r and b make the curve steeper (RefinedWeb uses 9000 hashes), but then you must calculate and store more hashes. FineWeb selected 112 hashes (14 × 8) to save compute. Its paper calculates this: a pair with s = 0.75 is caught with a probability of 77%. A pair with s ≥ 0.85 is almost always caught.

On the noisy crawl (1051 documents after the heuristic filter), exact dedup decreases the count to 992 documents, and MinHash decreases it to 923. Count the redundant copies by "origin": 130 before dedup, 71 after exact dedup, and only 2 after MinHash. (These 2 copies have more changes, so their similarity is below the threshold.) None of the 68 duplicate clusters joins documents from different origins by mistake.

The FineWeb paper has one more counter-intuitive finding. **Global** MinHash dedup over 96 Common Crawl snapshots did not make the model better. The 10% of data that "survived" in the old snapshots was the worse data. Only dedup inside each snapshot reached the level of RefinedWeb. The goal of dedup is to remove the "large clusters with thousands of copies". It is not to remove every duplicate.

## 6. Model-based quality filtering

Rules cannot catch documents that are "not language". The common method is to train a cheap classifier that gives a score to each document:

- **FineWeb-Edu**: Llama-3-70B-Instruct gave a score of 0–5 to 460K web pages for "educational value for primary and secondary school" (with an additive scoring prompt). A linear regression head (on a frozen Snowflake-arctic-embed-m encoder) trained on 410K labels. It scored all 15T tokens, and the pages with scores ≥3 stayed. This removed 92% and left 1.3T tokens. Knowledge benchmarks such as MMLU and ARC improved clearly. With a higher threshold, HellaSwag and PIQA decreased.
- **DCLM**: a fastText classifier. The positive examples are OpenHermes 2.5 instruction data and highly voted ELI5 answers. The negative examples are random web pages. It keeps about the top 10% by score.
- **Nemotron-CC**: found that the high-quality documents from the FineWeb-Edu and DCLM classifiers overlap by only 10%. Thus it uses an ensemble of three classifiers (the maximum score) and 20 buckets. This increased the fraction of high-quality tokens from 9% to 25%.
- **Ultra-FineWeb** (MiniCPM4): also fastText, but low-cost training experiments select the positive and negative examples.

[`04_quality_classifier.py`](code/04_quality_classifier.py) makes the FineWeb-Edu method smaller. An "annotator" labels 250 documents, a logistic regression (Chapter 5) trains on them, and then it scores the other documents. This chapter has no large model, so the "annotator" uses the true labels that we have. This is the only "cheat" in the script. The most useful of the 8 hand-made features is **bigram familiarity**: the fraction of adjacent word pairs (in Chinese, two characters) that occur in the good documents. CCNet uses the same idea when it filters with KenLM perplexity. In word salad, almost all adjacent word pairs are unfamiliar.

```
Weights (standardized): bigram +3.31, wordlen +2.01, cjk −0.96, unique −0.60 …
```

(`bigram` is the bigram familiarity, `wordlen` is the mean word length, `cjk` is the fraction of Chinese characters, and `unique` is the fraction of distinct words.)

On the 673 documents without a label, the threshold sets the trade-off between "catch all bad documents" and "remove few good documents":

| Threshold | Bad documents caught | Missed | Good documents removed | Precision | Recall |
|---:|---:|---:|---:|---:|---:|
| 0.3 | 27 | 17 | 0 | 1.00 | 0.61 |
| 0.5 | 31 | 13 | 1 | 0.97 | 0.70 |
| 0.7 | 37 | 7 | 6 | 0.86 | 0.84 |
| 0.9 | 41 | 3 | 66 | 0.38 | 0.93 |

With 0.5, the count goes from 923 to 871 documents. Word salad decreases from 68 to 21 documents, and only 3 good documents are lost. A threshold is not a guess: FineWeb-Edu ran ablations on 2, 3, and 4 and selected 3.

## 7. Synthetic rephrasing: let a model "rewrite" the web

Good data runs out. Muennighoff et al. (2023) found that up to 4 repeats of the same data cause almost no loss. With more repeats, the gain decreases quickly. This led to **synthetic rephrasing**: a language model rewrites existing text in a different form, and this gives new tokens. Some teams do it like this:

- **Nemotron-CC**: uses Mistral NeMo 12B. It rewrites low-quality documents in "Wikipedia style" (without the noise and errors). From high-quality documents, it generates varied QA pairs, condensed versions, knowledge extractions, and knowledge lists. In total, it synthesized 1.9T tokens. A controlled experiment trained an 8B model on 1T tokens. Rephrased low-quality data gave +1.5 points on average. When synthetic data replaced 4 of the 8 repeats of high-quality data, the average increased by +0.9 more points.
- **Kimi K2**: rewrites knowledge text in different styles and from different perspectives. It splits long documents into chunks, rewrites the chunks autoregressively, and joins them again. It also checks if each rewrite agrees with the original. It rewrites math documents as "study notes". On the same Wikipedia text: the original repeated 10 times gave a SimpleQA score of 23.76. One rewrite repeated 10 times gave 27.39. **10 rewrites, each seen 1 time, gave 28.94**.
- **Phi-4**: synthetic data is 40% of the pretraining tokens, and rewritten web pages are another 15%. Its ablation found that with only synthetic data, knowledge benchmarks (TriviaQA) became clearly worse. Thus it still mixes in web data.
- **Qwen3**: used Qwen2.5, Qwen2.5-Math, and Qwen2.5-Coder to synthesize "trillions of tokens" of textbooks, QA, instructions, and code snippets.

Synthetic rephrasing is not in the minimal code of this chapter, because it needs a large model that can write. Also remember the risks. A rewrite can add hallucinations. (Nemotron-CC saw decreases on some tasks, and says that it did not check the factual accuracy of the rewrites.) The license of the model that writes the rewrites must permit "use of the output to train other models" (Chapter 17 discusses the licenses of teacher models). The main-line model does no large-scale synthesis in the first stage. Rephrased data goes mainly into mid-training (Chapter 15), and the ablations of Step 2 set its amount.

## 8. Mixture and small-scale ablations

When we have some clean data sets, we must still decide the **mixture**: the share of English web, Chinese, code, and math in each training step. No formula gives the answer directly. All teams use **small-scale controlled experiments**:

| Who | How |
|---|---|
| Llama 3 | Small models run scaling-law experiments that predict the performance of large models with a given mixture, again and again. "Annealing" estimates the value of small data sets (anneal an 8B model, halfway through its training, on 40B tokens with 30% new data). Final mixture: general knowledge 50%, math and reasoning 25%, code 17%, multilingual 8% |
| Qwen3 | Labels 30T tokens by educational value, domain, safety, and more. Many ablations with small proxy models optimize the mixture "at the instance level" |
| OLMo 2 | "Microannealing": branch a short run from a pretrained checkpoint, and compare the runs with and without a data set |
| MobileLLM-R1 | Leave-one-out: remove one source each time, and look at the loss on probe sets for code, math, and knowledge. Then influence functions estimate the value of each source and calculate the mixture |
| Puro-2B | Continue training from the same Qwen3-0.6B checkpoint for about 8.4B tokens. The share of the candidate data increases linearly from 0 to 80% in the first 1600 steps. Look at the capability vector of 15 benchmarks, then a person sets the mixture |

All of them use one idea: **the same model and the same compute, with only different data; then compare the results**. This chapter runs two such experiments with small models of about 500K parameters. The two models use the same random seed: the same initialization and the same sample positions. Thus their difference comes from the data (a **paired comparison**, the idea of the bootstrap in Chapter 11). The evaluation metric is bits-per-byte:

```python
bpb = (nats * (nbytes > 0)).sum() / (math.log(2) * nbytes.sum())   # 06_quality_ablation.py: bpb_by_hand
```

The numerator is the negative log-likelihood (in nats) of all target tokens. The denominator is the number of UTF-8 bytes that these tokens cover, times ln 2. A special token counts as 0 bytes and is not included. Chapter 7 showed that bpb does not depend on the tokenizer, so you can compare models with different vocabularies. The script also calls the production `zero/data/bpb.py`, and the two results are equal (an `assert` parity check).

**Ablation 1: noisy data vs. filtered data** ([`06_quality_ablation.py`](code/06_quality_ablation.py)). Each of the two models trains for 200 steps × 16 × 128 = 409,600 tokens. One model uses the 1401 raw noisy pages. The other uses the 856 documents that remain after all steps of 02–05:

| Training data | Documents | Tokens | English bpb (seed 0 / 1) | Chinese bpb (seed 0 / 1) | Training loss at the last step (seed 0 / 1) |
|---|---:|---:|---:|---:|---:|
| Raw noisy crawl | 1401 | 1,370,713 | 3.126 / 3.215 | 3.400 / 3.494 | 4.546 / 4.715 |
| Filtered | 856 | 842,535 | 3.086 / 3.187 | 3.316 / 3.412 | 4.659 / 4.779 |
| **Paired difference (noisy − filtered)** | | | **0.040 / 0.028** | **0.084 / 0.081** | |

The training loss is lower for the noisy data, because navigation pages, spam, and duplicates are easy to predict. But on the validation set, the filtered data is better. **A low training loss does not mean a good model.** Always compare data on the same clean validation set. Also note that with the same data and a different seed, bpb can differ by 0.1. This is more than the difference that the data makes. This is why only a paired comparison works, and why real ablations use larger proxy models and many seeds.

**Ablation 2: mixture** ([`07_mixture_ablation.py`](code/07_mixture_ablation.py)). Use the same set of clean data (English, Chinese, code), and train one model for each of three mixtures.

Each mixture trains 2 times, one time with each seed. (With the same seed, the three mixtures have the same initialization: a paired comparison.)

| Mixture (en/zh/code) | English bpb (seed 0 / 1) | Chinese bpb (seed 0 / 1) | Code bpb (seed 0 / 1) | Mean with balanced weights (mean of the two seeds) |
|---|---:|---:|---:|---:|
| Balanced 0.45/0.45/0.10 | 3.192 / 3.155 | 3.486 / 3.525 | 3.760 / 3.744 | 3.381 |
| English-heavy 0.80/0.10/0.10 | 2.928 / 3.018 | 3.818 / 4.057 | 3.652 / 3.794 | 3.482 |
| Code-heavy 0.25/0.25/0.50 | 2.928 / 2.943 | 3.316 / 3.348 | 2.576 / 2.612 | 3.080 |

Three conclusions have the same direction for both seeds:

- **Code bpb is the most sensitive to the mixture.** With 0.50 code, code bpb decreases from about 3.75 to about 2.59. (The cost: the model saw the code 2.37 times.)
- **"English-heavy" makes English better and Chinese clearly worse.** English is about 0.20 lower than with the balanced mixture, and Chinese is about 0.43 higher. (The model saw the Chinese data only 0.08 times.)
- **One result was unexpected.** "Code-heavy" is also better than "balanced" on English and on Chinese (English about 0.24 lower, Chinese about 0.17 lower), with both seeds. We have no reliable explanation. Possibly code helps the model learn general local structure faster (MobileLLM-R1 also found that code data helps math). Possibly it is only an effect of this scale (500K parameters, 400K tokens). **This is why we must be careful when we extrapolate proxy experiments.** Real mixture experiments use much larger proxy models. They look at downstream abilities, not only at bpb, and they make sure that the conclusion does not change with scale. Puro-2B found a trade-off between code ability and general ability.

Which mixture is "best" depends on the weight that you give to each domain. Thus you must first decide the goal of the model. This is also why Puro-2B selects data with a "capability vector", not with a single score.

The **main-line mixture (provisional)** is in [`configs/main/data.toml`](../../configs/main/data.toml):

| Category | Share | Sources | Reason |
|---|---:|---|---|
| English web | 0.45 | FineWeb-Edu 0.35 + DCLM 0.10 | Most open high-quality data is English; web data is the "glue" (the leave-one-out experiment of MobileLLM-R1) |
| Chinese web | 0.30 | FineWeb-2 Chinese 0.20 + Ultra-FineWeb Chinese 0.10 | Bilingual goal (C-Eval, CMMLU); Puro-2B has only 9–12% Chinese, but Chinese is not its goal |
| Code | 0.15 | Stack-Edu | A tool call is structured text generated from a schema; code also helps math (MobileLLM-R1); Llama 3 has about 17% |
| Math | 0.10 | FineMath-3+ | Llama 3 has 25% math and reasoning; more math goes into mid-training |

With a budget of about 400B tokens, the model sees each source at most 1.2 times (FineMath-3+). It sees all other sources less than 1 time (see the comments for each item in `configs/main/data.toml`). These ratios are **provisional**. In Step 2, proxy experiments with the method above will set them (for the plan and the cost, see "Main-line progress").

## 9. Decontamination: the training data must not contain the test questions

If the evaluation questions occur in the training data, the scores are not reliable. The standard method is the **n-gram overlap check**. Normalize each test question (lowercase, remove punctuation; each Chinese character is one word), split it into n-grams, and put them in a set. Then scan each training document. If one n-gram matches, remove the full document. GPT-3 uses 13-grams. Llama 3 uses 8-grams for its contamination analysis. Phi-4 uses a mixed rule with 13-grams and 7-grams, and puts "common 13-grams" (standard phrases of multiple-choice options) on an allowlist.

The core of [`05_decontam.py`](code/05_decontam.py) is only a few lines:

```python
def find_contaminated(docs, eval_items, n=13):
    index = build_index(eval_items, n)                      # n-gram of a test question → question ids
    for i, d in enumerate(docs):
        for g in ngrams(norm_tokens(d["text"]), n):        # look up each n-gram of the document
            found |= index.get(g, set())
```

We use the 871 documents after the quality filter. 13 of them really contain a test question: 10 verbatim, 2 with changed case and punctuation, and 1 paraphrased. The results for different n:

| n | Flagged documents | Verbatim | Changed case and punctuation | Paraphrased | Other hits |
|---:|---:|---:|---:|---:|---:|
| 5 | 28 | 10/10 | 2/2 | 1/1 | 15 |
| 8 | 19 | 10/10 | 2/2 | 0/1 | 7 |
| 13 | 15 | 10/10 | 2/2 | 0/1 | 3 |
| 20 | 14 | 10/10 | 2/2 | 0/1 | 2 |

Three conclusions:

1. Normalization catches the changes in case and punctuation. But **13-grams cannot catch a paraphrased test question**: one word changes in every few words, so no 13 consecutive words match. The Phi-4 report also says that n-gram methods cannot stop paraphrases.
2. A small n gives false positives. At n = 5, a common phrase such as "me to the sight of" also counts as a hit.
3. The 3 "other hits" at n = 13 are **not false positives**. 3 Song ci poems, for example 《御街行》 ("Yu Jie Xing"), occur two times in the corpus. One copy is from 《宋词三百首》 ("Three Hundred Song Ci Poems"), and one is from 《全宋词》 ("Complete Song Ci"). The test question came from one of the copies, so this is a real leak. It also shows that document-level dedup cannot find "the same passage in two different documents".

The main-line decontamination checks all benchmarks of the preregistration in Chapter 11 (BFCL, C-Eval, CMMLU, GSM8K, …), plus our own tool-call development set. For tool-call data, it also checks the function names and the schemas (GOAL.md 3.2). The model card reports the number of test questions that matched.

## 10. Train the main-line tokenizer: how large is the vocabulary?

Chapter 7 discussed the trade-off of the vocabulary size. With a larger vocabulary, the same text gives fewer tokens (more bytes per token). But the embedding has V × d parameters. Chapter 7 used a small 2.5 MB corpus and could only go up to 32K. To decide for the main line, we need a better corpus. The best corpus is **the data that the main line will train on**.

[`configs/vocab/download.toml`](../../configs/vocab/download.toml) uses the downloader of Step 2, `zero.data.download`, to take a small sample from each main-line pretraining source:

- English: FineWeb-Edu, DCLM, and FineMath.
- Chinese: FineWeb-2 (`cmn_Hani`) and Ultra-FineWeb Chinese.
- Code: UltraData-Code-L2 of MiniCPM5 (9 programming languages). Stack-Edu in the main-line configuration has only file ids, and its text must come from Software Heritage with AWS credentials. Thus we use similar data whose text you can download directly.

The mixture is the same as in the main line. All 6 data sets are pinned to their commits of 2026-10-01. The sample has 9,361 documents and 46 MB of text in total. [`09_build_vocab_corpus.py`](code/09_build_vocab_corpus.py) merges them by language, shuffles them, and cuts validation sets (English 1.75 MB, Chinese 0.81 MB, code 1.63 MB). The rest is training text.

The tokenizer training text follows the main-line mixture (English 55%, Chinese 30%, code 15%; 30 MB in total). The training text must not be too small. With only 12 MB, the vocabulary stops at about 90K: no adjacent pairs that occur 2 or more times are left. Thus a larger vocabulary cannot train.

[runs/2026-10-01-vocab-corpus/](../../runs/2026-10-01-vocab-corpus/README.md) records what we downloaded, how much of each source, and the sha256 of each file. The corpus is only for local measurement. It is not in the repository, and we do not use it for training. (The first version of this section used a corpus of technical documentation from GitHub. When we wrote the book, the build environment could not access Hugging Face. The results of that version are after the table below. They show how large the effect of "the same distribution" is.)

We calculate the cost with the shape of the main line (`configs/main/pretrain.toml`: 28 layers, width 1280, tied embeddings) and change only V. Parameters = 605.6M non-embedding + V × 1280, and **the total must not be more than 0.8B**. Training compute per token ≈ 6 × N_matmul (this includes the 1280 × V matmul of lm_head) + attention. What we really must compare is **how much compute it takes to read the same amount of text**:

```
FLOPs / byte = (FLOPs / token) ÷ (bytes / token)
```

These are the results of `08_vocab_size.py --corpus <dir> --refs --train-mb 30`. ("Weighted" uses English 0.55 / Chinese 0.30 / code 0.15. The unit of FLOPs/byte is the FLOPs per token at V = 65,536. On a CPU with 16 threads, the 7 vocabularies took about 2 minutes to train in total.)

| Tokenizer | Vocabulary V | English | Chinese | Code | Weighted bytes/token | Embedding | Total parameters | Compute per token | **FLOPs / byte** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| zero BPE | 16,384 | 3.68 | 3.45 | 3.19 | 3.54 | 21.0M | 626.6M | 6.58 GFLOP | 0.2674 |
| zero BPE | 32,768 | 3.99 | 3.86 | 3.46 | 3.87 | 41.9M | 647.6M | 6.70 GFLOP | 0.2489 |
| zero BPE | 49,152 | 4.13 | 4.09 | 3.59 | 4.04 | 62.9M | 668.5M | 6.83 GFLOP | 0.2431 |
| zero BPE | **65,536** | 4.22 | 4.25 | 3.66 | 4.15 | 83.9M | 689.5M | 6.96 GFLOP | **0.2412** |
| zero BPE | 98,304 | 4.31 | 4.46 | 3.76 | 4.27 | 125.8M | 731.5M | 7.21 GFLOP | 0.2425 |
| zero BPE | 131,072 | 4.36 | 4.62 | 3.81 | 4.36 | 167.8M | 773.4M | 7.46 GFLOP | 0.2460 |
| zero BPE | 151,936 | 4.39 | 4.70 | 3.83 | 4.40 | 194.5M | **800.1M (over the limit)** | 7.62 GFLOP | 0.2491 |
| Qwen2/Qwen3 tokenizer (reference) | 151,936 | 4.39 | 3.90 | 4.00 | 4.18 | 194.5M | 800.1M | 7.62 GFLOP | 0.2618 |
| Llama 3 tokenizer (reference) | 128,256 | 4.53 | 3.27 | 4.05 | 4.08 | 164.2M | 769.8M | 7.44 GFLOP | 0.2619 |

(We rebuilt the reference tokenizers from `ggml-vocab-{qwen2,llama-bpe}.gguf` of llama.cpp. We checked their splits with the test cases that come with llama.cpp. Their parameters and compute are calculated "on the main-line shape".)

How to read this table:

- **The compression gain becomes smaller.** From 16K to 32K, the weighted bytes/token increases by 9.5%. From 64K to 128K, it increases by only 5.1%. Chinese gains the most (Chinese has many "words", and only a large vocabulary has space for them). English and code gain slowly after 64K.
- **Compute per byte is lowest at 64K** and increases on both sides. 96K costs 0.5% more and 48K costs 0.8% more (a small vocabulary gives more tokens). 128K costs 2.0% more and 151,936 costs 3.3% more (lm_head becomes larger). Also, the Qwen vocabulary size at this width gives 800.1M total parameters. This is **more than the 0.8B limit**.
- **Our own tokenizer is better on Chinese, but not on code and English.** With the same vocabulary of 151,936, our tokenizer gives 4.70 bytes/token on Chinese, and Qwen gives only 3.90 (ours is 20% higher). On English, the two are equal (4.39). On code, Qwen is 4.5% higher. Llama 3 is also better than our 128K tokenizer on English and code (3.9% and 6.2% higher). The Qwen and Llama tokenizers trained on data that is orders of magnitude larger than our 30 MB, and much more varied. Our only clear advantage is the high share of Chinese (30%, but Qwen must support 119 languages).
- **The advantage of the same distribution is smaller than it looks.** The first measurement (the collapsed table below) used technical documentation, so the training text and the validation text came from the same type of documents. There, our Chinese was 43% higher than Qwen, and our code was also better. With real web and code samples, the Chinese advantage decreased to 20%, and the code advantage disappeared. The data that you measure on is more important than the precision of the measurement.
- The vocabulary scaling law of Tao et al. (2024) recommends an optimal vocabulary of about 40K for a model with 3B non-vocabulary parameters, trained with the Chinchilla ratio. The main line has about 600M non-vocabulary parameters. But overtraining (about 700 tokens/parameter) moves the optimum up. This agrees in direction with the conclusion of the table: "64K uses the least compute".

<details>
<summary>First measurement: a corpus of technical documentation from GitHub (2026-09; the build environment could not access Hugging Face when we wrote the book)</summary>

Corpus: Chinese text from the Chinese edition of *Dive into Deep Learning*, the Chinese Kubernetes documentation, JavaGuide, CS-Notes, and Python-100-Days (without the code blocks and the original English text in HTML comments). English text from the English edition of *Dive into Deep Learning*, the English Kubernetes documentation, and the CPython documentation. Code from the Python and C source of CPython. About 75 MB in total (for the sources, see the references of this chapter; we did not keep the preparation script). Validation sets: English 1.75 MB, Chinese 0.77 MB, code 1.63 MB.

| Tokenizer | Vocabulary V | English | Chinese | Code | Weighted bytes/token | **FLOPs / byte** |
|---|---:|---:|---:|---:|---:|---:|
| zero BPE | 16,384 | 4.23 | 4.41 | 3.74 | 4.21 | 0.2245 |
| zero BPE | 32,768 | 4.44 | 4.96 | 3.98 | 4.53 | 0.2129 |
| zero BPE | 49,152 | 4.51 | 5.25 | 4.08 | 4.67 | **0.2103** |
| zero BPE | 65,536 | 4.56 | 5.43 | 4.14 | 4.76 | **0.2103** |
| zero BPE | 98,304 | 4.60 | 5.66 | 4.21 | 4.86 | 0.2134 |
| zero BPE | 131,072 | 4.62 | 5.79 | 4.25 | 4.91 | 0.2182 |
| zero BPE | 151,936 | 4.62 | 5.87 | 4.26 | 4.94 | 0.2217 |
| Qwen2/Qwen3 tokenizer (reference) | 151,936 | 4.29 | 4.11 | 4.09 | 4.21 | 0.2603 |
| Llama 3 tokenizer (reference) | 128,256 | 4.38 | 3.75 | 4.18 | 4.16 | 0.2568 |

Technical documentation is much "cleaner" than web pages, so all tokenizers compress it better. The conclusion at that time was "48K and 64K tie for the least compute". This time, on the main-line data sample, 64K alone is the lowest. The choice does not change.

</details>

**The main-line vocabulary is 65,536.** (`configs/main/data.toml` and `configs/main/pretrain.toml` agree; this confirms the "provisional" value of Chapter 7.) These are the reasons:

- On the main-line data sample, it has the lowest compute per byte (it is at the lowest point in both measurements).
- It compresses Chinese 4.0% better than 48K. Chinese is our weak data, so each saved token is valuable.
- Its embedding of 83.9M is only 12% of the total parameters.
- It is a power of 2 and a multiple of 128, so the matmuls are regular on a GPU.

After Step 2 downloads the full data, we will measure again on the full training corpus. This sample comes only from the first few shards of each data set (web pages from 2013–2014; see the record in the runs folder). If the conclusion changes, we will change the choice.

**Pre-tokenization regex**: Chapter 7 left a question open. The regex of zero is the same as the regex of Qwen2/Qwen3, but Qwen3.5 changed the letter sequence from `\p{L}+` to `[\p{L}\p{M}]+`. `\p{M}` is "combining marks": the vowel signs of Devanagari and Thai, and the diacritics of Arabic. At these marks, the old regex cuts a word into pieces:

| Sentence | qwen2 regex | qwen3.5 regex |
|---|---:|---:|
| Hindi (Article 1 of the Universal Declaration of Human Rights) | 36 pieces; `सभी` becomes `सभ` + `ी` | 14 pieces |
| Thai | 16 pieces | 1 piece |
| Chinese / English | 3 / 13 pieces | 3 / 13 pieces (identical) |

On the validation set of the main-line data sample (the first 200K characters of each language), the pieces for English and code are identical. Chinese has only 4 differences. 2 of them are the variation selector U+FE0F after an emoji (it belongs to \p{M}). The other 2 are a passage of Thai text in a Chinese web page: real web pages are never pure single-language text. We trained one tokenizer with a vocabulary of 65,536 for each regex. The validation bytes/token is identical for all three languages (English 4.217, Chinese 4.254, code 3.665). Thus **the main line uses the qwen3.5 regex**. It costs nothing for Chinese, English, and code, it is better for other languages, and it agrees with the latest Qwen. llama.cpp already supports this pre-tokenizer type (`qwen35`): when you export GGUF, set the `pre_tokenizer` parameter to `qwen35`. This choice takes effect only through the configuration (`pretokenize = "qwen3.5"` in `configs/main/data.toml`). The default of `zero/tokenizer.py` does not change, so the existing tests and the tiny pipeline work as before.

## 11. Licenses and provenance: record each step

A data set goes into the model card, and other people must be able to reproduce it. The production pipeline makes a record at each step:

- The downloader ([`zero/data/download.py`](../../zero/data/download.py)) downloads only the sources that are registered in `sources.py`. By default, it refuses a source whose license is "to be verified". Each JSON line contains the data set name, the subset, the revision, the line number, and the metadata of the data set (URL, dump, quality score, `detected_licenses` of code files). The downloader records the sha256 of each shard, and it can resume a download.
- The pipeline ([`zero/data/pipeline.py`](../../zero/data/pipeline.py)) writes the intermediate result of each step to disk as JSONL. At the end, it writes `manifest.json`. The manifest contains these items:
  - the documents and bytes of each source after each step (the funnel);
  - the removal reasons and the number of duplicate clusters;
  - the test questions that decontamination found;
  - the tokenizer hash and the compression of each source;
  - the token counts of the shards, the mixture, and "how many epochs for the budget";
  - the licenses and source addresses, the configuration hash, and the git commit.
- The metadata of each shard contains the tokenizer hash (Chapter 7). This prevents training a model with tokenizer B on data that tokenizer A split.

## 12. Summary

- **Data is the largest lever**: MobileLLM-R1 matches the reasoning ability of Qwen3-0.6B with 11.7% of the tokens. With public data and proxy experiments, Puro-2B matches Qwen2-1.5B for a few thousand US dollars.
- **Pipeline**: extraction → language identification → heuristic rules → exact + near dedup → model-based scoring → (synthetic rephrasing) → decontamination → tokenization and shards. Put the cheap steps first. Rules catch format problems, but not text that "is not language". The thresholds of the rules depend on the domain, so select them by source.
- **MinHash**: P(one position of the signature is equal) = Jaccard. **LSH**: P(candidate) = 1 − (1 − s^r)^b, with the inflection point at (1/b)^(1/r).
- **Model-based scoring**: a large model labels a small part, and a cheap classifier scores all the data. The threshold is a trade-off between precision and recall, and ablations must set it.
- **Mixture**: there is no formula. Use controlled experiments with proxy models. For comparisons, use a paired design and bpb, which does not depend on the tokenizer.
- **Decontamination**: 13-grams catch verbatim leaks and leaks with format changes, but not paraphrases. A small n gives false positives.
- **Tokenizer**: compare "compute per byte", with the constraint "total parameters ≤ 0.8B". For the main-line vocabulary and regex, see Section 10.

---

## From minimal code to production code

| Minimal code (`code/`) | Production code (`zero/`) | What it adds, and why |
|---|---|---|
| `lang_id` in `02` (fraction of Chinese characters / letters) | `langid_stage` in `zero/data/pipeline.py` (the same rough check); Step 2 changes it to fastText lid.176 | Real web pages have hundreds of languages, so a real language identification model is necessary |
| `heuristic_reasons` in `02` (11 rules) | `quality_check` in `zero/data/quality.py` (the `[quality]` section of the TOML can override `QualityThresholds`); `code_quality_reasons` in `pipeline.py` (the three Codex rules) | Configurable thresholds; rules selected by source (`heuristics = "web" / "code" / "none"`); the manifest counts the removal reasons for each rule |
| `MinHash` / `lsh_candidates` / `near_dedup` in `03` | `MinHasher` and `near_dedup` in `zero/data/dedup.py` (the same "low 32 bits" hash, the same union-find) + `exact_dedup` | `num_perm / bands / ngram / threshold` are configurable; after dedup within each source, an optional dedup across sources; a single-process implementation. The scale of Step 2 needs distributed MinHash (for example datatrove). **Not yet verified on large-scale data** |
| Logistic regression in `04` | `score_stage` in `pipeline.py`: a threshold on a score that the data set provides (`score_field` / `score_min`, for example `int_score` of FineWeb-Edu), or any `module:ClassName` classifier (the `QualityClassifier` protocol in `quality.py`) | The main line uses the scores that upstream already calculated; a Chinese quality classifier is a task of Step 2 |
| `find_contaminated` in `05` | `NgramIndex` in `zero/data/decontam.py` (stores n-grams as 8-byte hashes; uses substring matching for short questions) + `decontam_stage` in `pipeline.py` | Reads the evaluation sets from JSONL (by default all string fields, **including tool names and schemas**); writes the details of each hit into the manifest |
| `bpb_by_hand` in `06` | `token_byte_lengths`, `evaluate_bpb`, and `bpb_stats` in `zero/data/bpb.py` (in the style of `evaluate_bpb` of nanochat) | A precalculated "token id → bytes" lookup table; special tokens and ignored positions do not count; with many GPUs, an all_reduce of the numerator and the denominator separately |
| Line-by-line source sampling by mixture in `07` | `MixtureLoader` in `zero/data/mixture.py` (Chapter 14) | Only (seed, rank, line number) decide the sample, so a resumed run is identical byte for byte |
| `train_tokenizer(..., "qwen3.5")` in `08` | `train_tokenizer` and `PRETOKENIZE_PRESETS` in `zero/data/pipeline.py` (`qwen2` is identical to the default of `zero/tokenizer.py`) | The regex changes only through the configuration; the training text is sampled from each source by the mixture (`sample_bytes`) |
| None | `zero/data/download.py` | Streaming download for Step 2: license gate, provenance fields, shard sha256, resume, download planning (`plan_targets`). **Not yet verified** |
| None | `run_pipeline` in `zero/data/pipeline.py` + `configs/{tiny,main}/data.toml` | One command runs all 10 stages and saves each stage to disk. At the end, it writes `manifest.json` and a `pretrain_sources.toml` that you can paste into `pretrain.toml` |

**Parity checks and tests** (`uv run pytest tests/test_bpb.py tests/test_pipeline.py tests/test_download.py tests/test_data.py`):

- `test_bpb.py`: small examples calculated by hand (4 tokens with all logits 0 → bpb exactly 1.0; the analytic value for non-uniform logits); the invariant of a uniform model, bpb = log2(V) ÷ (bytes/token); special tokens, ignore_index, and negative labels do not count; the relation to the normal cross-entropy; the sum of `token_byte_lengths` over all tokens equals the UTF-8 bytes of the original text (also with tokens that contain "half a Chinese character").
- `test_pipeline.py`: the pure functions of each stage; the qwen2 preset gives the same hash as `train_bpe`; for the qwen3.5 preset: round trip, digits split one by one, special-token ids, and the regex is still there after save and load; end to end on a small corpus (text and JSONL inputs, score threshold, exact duplicates, removal of leaked test questions, shards that `PackedDataLoader` can read).
- `test_download.py`: fake data replaces the network. It tests the provenance fields, shard rotation and sha256, resume after a stop, Stack-Edu content by blob_id, the license gate, download planning, and the parsing of the download specs in `configs/main/data.toml`.
- `06_quality_ablation.py` does an `assert` at each evaluation: the bpb by hand equals the bpb of `zero/data/bpb.py`.

**Connection to the training loop** (already in `zero/train/trainer.py`): `Trainer` uses the tokenizer to calculate `token_byte_lengths` one time. `Trainer.evaluate` gives the same validation batches to `bpb_stats`, and writes `val_bpb` together with `val_loss` to the log. (It skips this step for SFT-format data or when there is no tokenizer.) One run on the tiny configuration recorded `val_bpb 3.997` (tiny-configuration demo).

---

## Main-line progress

### Tiny-configuration demo (CPU, `configs/tiny/data.toml`, assets/tiny_corpus)

> The following is a **tiny-configuration demo**. It shows only that each step of the production pipeline really works. The ratios on the 2.5 MB toy corpus do not represent real web data.

```bash
uv run python -m zero.data.pipeline --config configs/tiny/data.toml
```

This is the funnel of one run on this machine (about 10 s of CPU time; 48 s of wall-clock time on a busy machine). Each cell is the number of documents that remain after the step:

| Source (license) | Read | Clean | Language | Rules | Score | Dedup | Decontamination | Train / validation documents | Train tokens | Validation bytes/token |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| shakespeare (public domain) | 615 | 615 | 615 | 528 | 528 | 528 | 528 | 502 / 26 | 351,245 | 2.62 |
| chinese_poetry (MIT) | 225 | 225 | 225 | 211 | 211 | 211 | 211 | 201 / 10 | 422,715 | 2.52 |
| code (to be verified: the repository license is not decided) | 81 | 81 | 81 | 81 | 81 | 81 | 81 | 77 / 4 | 73,511 | 2.01 |

- The manifest records the removal reasons of the rules. Shakespeare: duplicate lines 70, line-end punctuation 13, short lines 4. Poems: duplicate lines 8, and 6 documents with less than 80% "words with a letter". (After the split of Chinese into characters, each punctuation mark also counts as a "word". Ci tunes with many short lines have a high density of punctuation.) Code uses the Codex rules, and no code document was removed.
- Dedup and decontamination both removed 0 documents. The tiny corpus has no duplicate documents. (The three repeated Song ci poems of Section 9 are in different "documents", so document-level dedup cannot see them.) It also does not contain the content of the tool-call development set `tool_dev.jsonl` (100 questions, with function names and schemas).
- The tokenizer has a vocabulary of 2048 (qwen2 regex), and its hash is in the metadata of each shard. You can paste `out/tiny/data_pipeline/pretrain_sources.toml` directly into the tiny pretraining configuration. With it, a 20-step training run (`zero.train.pretrain`, 1.31M parameters) gave a validation loss of 6.80. Then `zero/data/bpb.py` evaluated bpb on the validation shards of the three sources: Shakespeare 3.553, poems 3.749, code 5.146. (Only 20 training steps, so the numbers show only that the interfaces work.)

```python
tb = token_byte_lengths(Tokenizer.load("out/tiny/data_pipeline/shards/tokenizer.json"))
val = PackedDataLoader("out/tiny/data_pipeline/shards/code_val_*.bin", seq_len=128, batch_size=8, shuffle=False)
print(bpb_stats(model, val, tb, steps=8).bpb)
```

### To be added after GPU training

- Step 2: run `download.py` on real data (first with `--max-docs 1000` to check the fields) and `pipeline.py` (change step 5 to distributed MinHash). Write the real funnel, the removal reasons, and the decontamination hits here.
- Proxy experiments for the mixture: start from a checkpoint halfway through the Chapter 12 ladder experiment `configs/ladder/l150m.toml`. Continue training each candidate for about 2B tokens, with the candidate share increasing linearly from 0 to 80%. Look at the bpb on the development set of each domain and at small benchmarks. First compare the sources, then compare 3–4 mixtures, and last set the threshold of the FineWeb-2 Chinese classifier. `zero.tools.estimate_cost` estimates 1.8 GPU hours and about $4 per experiment (assumptions: H100, MFU 0.4, $2.5/GPU hour; **not yet verified on a GPU**). We plan 20–30 experiments. With evaluation, they cost about $100–200. This is within the "small-scale experiments for Chapters 12–13: ~$1,200" of GOAL.md 3.4.
- Measure the vocabulary table of Section 10 again on real samples, and confirm V. After the final decision, update `model.vocab_size` in `configs/main/pretrain.toml`.
- The results of the three license checks that are still open (the upstream sources of Ultra-FineWeb Chinese, Stack-Edu, and "research only" of DCLM).
- The manifest of the final data set (`manifest.json`) is released with the model card.

---

## Frontier notes

- **Automatic mixture optimization**: DoReMi, RegMix (small models fit a regression from "mixture → loss"), and MobileLLM-R1 with influence functions (AutoMixer) calculate the mixture directly. They work in their own papers. But in the technical reports of the leading open models, the final mixture still comes mostly from "proxy experiments + a human decision" (Llama 3 and Puro-2B both say this). Qwen3 says only that it used proxy models for instance-level optimization, and it does not publish the method. Thus this is not yet a consensus.
- **Curriculum learning sorted by quality**: Puro-2B sorts each source by quality score from low to high, so the best data comes last, together with checkpoint averaging. This has the same direction as "change to the best data in mid-training" in Chapter 15. Finer sorting methods are still under research.
- **Semantic dedup**: cluster sentence embeddings and remove documents that "have the same meaning but different words" (SemDeDup). Llama 3 used it on post-training data. We have not yet seen many teams use it on pretraining data.

---

## Adopters and sources

| Technique | Adopters (main versions) | Sources |
|---|---|---|
| Dedup (exact + MinHash near dedup) | Llama 3 (URL level, global MinHash at document level, line level); Nemotron (global fuzzy dedup + exact substring dedup, NeMo Curator); SmolLM (FineWeb / FineWeb-Edu: MinHash within each dump, 5-grams, 14 × 8); OLMo 2 (DCLM-baseline: Bloom filter dedup); Puro-2B (MinHash within each source); MiniCPM (Ultra-FineWeb-L1) | [Llama 3 §3.1.1](https://arxiv.org/abs/2407.21783); [Nemotron-CC §2.1](https://arxiv.org/abs/2412.02595); [FineWeb §3.4, Appendix E](https://arxiv.org/abs/2406.17557); [DCLM data set card](https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0); [OLMo 2 §2.4](https://arxiv.org/abs/2501.00656); [Puro-2B §3.6.2](https://www.alphaxiv.org/abs/2608.27370); [Ultra-FineWeb data set card](https://huggingface.co/datasets/openbmb/Ultra-FineWeb) |
| Model-based quality filtering | Llama 3 (Llama 2 labels → DistilRoberta classifiers for quality, code, and reasoning); Qwen3 (30T tokens labeled in many dimensions, such as educational value); SmolLM2 (FineWeb-Edu, Stack-Edu, and FineMath classifiers); OLMo 2 (DCLM fastText + FineWeb-Edu classifier select Dolmino); Nemotron (ensemble of three classifiers); MiniCPM4 (Ultra-FineWeb fastText); Phi-4 (small classifiers trained on ~10⁶ LLM labels) | [Llama 3 §3.1.1](https://arxiv.org/abs/2407.21783); [Qwen3 §3.1](https://arxiv.org/abs/2505.09388); [SmolLM2](https://arxiv.org/abs/2502.02737); [OLMo 2 §4.3](https://arxiv.org/abs/2501.00656); [Nemotron-CC §2.2](https://arxiv.org/abs/2412.02595); [Ultra-FineWeb](https://arxiv.org/abs/2505.05427); [Phi-4 §2.3](https://arxiv.org/abs/2412.08905) |
| Synthetic rephrasing / synthetic data | Kimi K2 (knowledge rephrasing, math rephrasing as "study notes"); Nemotron (the 1.9T synthetic tokens of Nemotron-CC); Qwen3 (trillions of tokens synthesized with the Qwen2.5 series); Phi-4 (40% synthetic + 15% rewritten web); OLMo 2 (TinyGSM rewritten with MIND in Dolmino); MiniCPM (Ultra-FineWeb-L3 QA generation and rephrasing in many styles) | [Kimi K2 §2.2](https://arxiv.org/abs/2507.20534); [Nemotron-CC §2.3](https://arxiv.org/abs/2412.02595); [Qwen3 §3.1](https://arxiv.org/abs/2505.09388); [Phi-4 §3.2](https://arxiv.org/abs/2412.08905); [OLMo 2 §4.4](https://arxiv.org/abs/2501.00656); [Ultra-FineWeb data set card](https://huggingface.co/datasets/openbmb/Ultra-FineWeb) |
| Decontamination (n-gram overlap) | Llama 3 (8-gram contamination analysis; exact-match dedup of post-training data against benchmark prompts); Phi-4 (mix of 13-grams + 7-grams); OLMo 2 (remove FLAN items with an n-gram overlap ≥10% with the evaluation sets); Puro-2B (13-grams on SFT data); GPT-3 (13-grams, the origin of the method) | [Llama 3 §5.1, §5.2](https://arxiv.org/abs/2407.21783); [Phi-4 Appendix B](https://arxiv.org/abs/2412.08905); [OLMo 2 §4.3](https://arxiv.org/abs/2501.00656); [Puro-2B §3.5](https://www.alphaxiv.org/abs/2608.27370); [GPT-3 Appendix C](https://arxiv.org/abs/2005.14165) |
| Mixture from small-scale experiments | Llama 3 (scaling-law experiments + annealing evaluation); Qwen3 (small proxy models); OLMo 2 (microannealing); Phi-4 (mixture ablations at 1T tokens and 7B scale); MobileLLM-R1 (leave-one-out + influence functions); Puro-2B (continued training of a Qwen3-0.6B proxy) | [Llama 3 §3.1.2–3.1.3](https://arxiv.org/abs/2407.21783); [Qwen3 §3.1](https://arxiv.org/abs/2505.09388); [OLMo 2 §4.4](https://arxiv.org/abs/2501.00656); [Phi-4 §3.2](https://arxiv.org/abs/2412.08905); [MobileLLM-R1 §2](https://arxiv.org/abs/2509.24945); [Puro-2B §3.6.1](https://www.alphaxiv.org/abs/2608.27370) |
| Heuristic rules (Gopher / C4 / FineWeb) | FineWeb (the data of SmolLM); DCLM (RefinedWeb rules, the data of OLMo 2); Llama 3 (duplicated n-grams, dirty words, KL of the token distribution); Nemotron (only on the low-quality part) | [Gopher Appendix A](https://arxiv.org/abs/2112.11446); [C4](https://arxiv.org/abs/1910.10683); [FineWeb §3.3–3.6](https://arxiv.org/abs/2406.17557); [Llama 3 §3.1.1](https://arxiv.org/abs/2407.21783) |

To be verified: the token count of the Chinese subset of FineWeb-2 (the data set card gives only the document count and the parquet size); the original text of the license terms of Nemotron-CC-v2 (we read only the summary in the Puro-2B paper and `license: other` on the HF page); the relation between "research only" on the DCLM data set card and CC-BY-4.0; the exact threshold of "alphanumeric fraction too low" in Section 3.1 of the Codex paper (this chapter uses 0.25; the paper gives no number, to be verified).

---

## Guided questions

Ask Claude Code these questions. Continue until you can explain the answers in your own words:

1. FineWeb found that global dedup over 96 snapshots made the data worse. Why can "more complete dedup" make the data worse? If you have only one snapshot, does this problem still occur?
2. `03_minhash.py` uses 128 hashes, 16 × 8. If you change to 32 × 4, where does the inflection point of the S-curve move? What happens to "reposts with some changed words", and to "two different articles on similar topics"?
3. The most useful feature of the quality classifier in this chapter is "bigram familiarity". If all 250 documents that the annotator labels are Shakespeare and none are Song ci, how does the classifier treat Chinese documents? Does the real FineWeb-Edu classifier have a similar bias? (Hint: with a higher threshold, HellaSwag decreases.)
4. The two ablation experiments trained on only a few hundred thousand tokens. Can such small experiments decide the main-line mixture? What do Puro-2B and Llama 3 do to make proxy experiments more reliable?
5. 13-gram decontamination cannot catch paraphrased test questions. What other methods are there, other than a smaller n? (Hint: the "new questions" evaluation of Phi-4, and the own development set of Chapter 11.)
6. If the vocabulary increases from 64K to 128K, how do the inference speed and the GGUF file size of the main-line model change? For the goal "a tool-call assistant that runs on a laptop", which side is more important?

## Hands-on tasks

For each task, run the code and look at the result.

**Task 1 (basic)**: In `02_heuristic_filter.py`, change the Gopher duplicate-line threshold from 0.3 to 0.5, and run the script again. Record the false-removal rate of Shakespeare and the removal rate of navigation pages. Then think: why can you not tune the threshold only for Shakespeare?

**Task 2 (core)**: In `03_minhash.py`, change `bands` of `near_dedup` to 8, and then to 32 (keep `num_perm` at 128). Run the script again for each value. Use the S-curve table to explain the changes in "redundant copies" and "wrong merges". Then in `near_copy` of `01_noisy_crawl.py`, change the fraction of changed words from 1/60 to 1/10. Can MinHash still catch the copies?

**Task 3 (challenge)**: Add the "allowlist" idea of Phi-4 to `05_decontam.py`. First, count the 13-grams that occur most often in the training documents (for example, in more than 5 documents), and skip them in the check. Then decrease n to 8. Do the "other hits" decrease, while the check still catches all verbatim leaks? Then use the framework of `06_quality_ablation.py` to compare two models, trained "before / after decontamination", by their bpb on the test questions. How much "better" does the leak make the model on the test questions?

---

## Go deeper: CS336

This chapter matches Stanford CS336 (Spring 2026) <https://cs336.stanford.edu/>:

- **Lecture 13: Data (sources and data sets)**. Common Crawl, web extraction, how each open data set was made, and the copyright and license questions of data. This lecture expands Section 2 of this chapter.
- **Lecture 14: Data (filtering, dedup, mixture, synthetic data)**. Language identification, quality classifiers, MinHash/LSH, mixtures, and synthetic data. This lecture follows the same line as Sections 4–9 of this chapter, in a more systematic way.
- **Assignment 4 (Data)**: start from raw Common Crawl data. Implement HTML extraction, language identification, PII masking, harmful-content filtering, quality classifiers, and exact and MinHash dedup yourself. Then train a model on the filtered data and compare the validation loss on a leaderboard. Scripts `02`–`05` of this chapter are a small warm-up for it. Assignment repository: <https://github.com/stanford-cs336/assignment4-data>

---

## References

- Zhao et al. *MobileLLM-R1: Exploring the Limits of Sub-Billion Language Model Reasoners with Open Training Recipes*, ICLR 2026: <https://arxiv.org/abs/2509.24945>
- Luo et al. *PuRo-2B: Poor Lab's Qwen2-1.5B Trained on RTX 5090 within $5090*, 2026: <https://www.alphaxiv.org/abs/2608.27370>
- Penedo et al. *The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale*, 2024: <https://arxiv.org/abs/2406.17557>; FineWeb-Edu data set card: <https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu>
- Penedo et al. *FineWeb2: One Pipeline to Scale Them All — Adapting Pre-Training Data Processing to Every Language*, 2025: <https://arxiv.org/abs/2506.20920>; data set card: <https://huggingface.co/datasets/HuggingFaceFW/fineweb-2>
- Li et al. *DataComp-LM: In search of the next generation of training sets for language models*, 2024: <https://arxiv.org/abs/2406.11794>; data set card: <https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0>
- Su et al. *Nemotron-CC: Transforming Common Crawl into a Refined Long-Horizon Pretraining Dataset*, ACL 2025: <https://arxiv.org/abs/2412.02595>
- Wang et al. *Ultra-FineWeb: Efficient Data Filtering and Verification for High-Quality LLM Training Data*, 2025: <https://arxiv.org/abs/2505.05427>; data set card: <https://huggingface.co/datasets/openbmb/Ultra-FineWeb>
- Allal et al. *SmolLM2: When Smol Goes Big — Data-Centric Training of a Small Language Model* (Stack-Edu, FineMath), 2025: <https://arxiv.org/abs/2502.02737>; <https://huggingface.co/datasets/HuggingFaceTB/stack-edu>, <https://huggingface.co/datasets/HuggingFaceTB/finemath>
- Llama Team. *The Llama 3 Herd of Models*, 2024: <https://arxiv.org/abs/2407.21783>
- Qwen Team. *Qwen3 Technical Report*, 2025: <https://arxiv.org/abs/2505.09388>
- Kimi Team. *Kimi K2: Open Agentic Intelligence*, 2025: <https://arxiv.org/abs/2507.20534>
- Abdin et al. *Phi-4 Technical Report*, 2024: <https://arxiv.org/abs/2412.08905>
- OLMo Team. *2 OLMo 2 Furious*, 2025: <https://arxiv.org/abs/2501.00656>
- Maini et al. *Rephrasing the Web (WRAP)*, 2024: <https://arxiv.org/abs/2401.16380>
- Rae et al. *Scaling Language Models: Methods, Analysis & Insights from Training Gopher* (the quality rules in Appendix A), 2021: <https://arxiv.org/abs/2112.11446>
- Raffel et al. *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer* (C4), 2019: <https://arxiv.org/abs/1910.10683>
- Wenzek et al. *CCNet: Extracting High Quality Monolingual Datasets from Web Crawl Data*, 2019: <https://arxiv.org/abs/1911.00359>
- Lee et al. *Deduplicating Training Data Makes Language Models Better*, 2021: <https://arxiv.org/abs/2107.06499>
- Broder. *On the resemblance and containment of documents* (MinHash), 1997
- Brown et al. *Language Models are Few-Shot Learners* (GPT-3; the 13-gram decontamination in Appendix C), 2020: <https://arxiv.org/abs/2005.14165>
- Chen et al. *Evaluating Large Language Models Trained on Code* (Codex; the code filter rules in Section 3.1), 2021: <https://arxiv.org/abs/2107.03374>
- Muennighoff et al. *Scaling Data-Constrained Language Models*, 2023: <https://arxiv.org/abs/2305.16264>
- Tao et al. *Scaling Laws with Vocabulary: Larger Models Deserve Larger Vocabularies*, NeurIPS 2024: <https://arxiv.org/abs/2407.13623>
- Karpathy. nanochat (`evaluate_bpb` in `nanochat/loss_eval.py`): <https://github.com/karpathy/nanochat>
- `tokenizer.json` of Qwen3.5-0.8B (the pre-tokenization regex, read in 2026-09): <https://huggingface.co/Qwen/Qwen3.5-0.8B/blob/main/tokenizer.json>
- The pre-tokenizer type `qwen35` and the vocab test files of llama.cpp: <https://github.com/ggml-org/llama.cpp> (`src/llama-vocab.cpp`, `models/ggml-vocab-*.gguf`)
- The corpus for the vocabulary measurement in Section 10 (2026-10, samples of the main-line pretraining sources; record in [runs/2026-10-01-vocab-corpus](../../runs/2026-10-01-vocab-corpus/README.md)): [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) (`sample-10BT`), [DCLM-baseline 1.0](https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0), [FineMath](https://huggingface.co/datasets/HuggingFaceTB/finemath) (`finemath-3plus`), [FineWeb-2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) (`cmn_Hani`), [Ultra-FineWeb](https://huggingface.co/datasets/openbmb/Ultra-FineWeb) (`zh`), [UltraData-Code](https://huggingface.co/datasets/openbmb/UltraData-Code) (`UltraData-Code-L2`, the code data of MiniCPM5)
- The corpus for the first vocabulary measurement in Section 10: [d2l-ai/d2l-zh](https://github.com/d2l-ai/d2l-zh), [d2l-ai/d2l-en](https://github.com/d2l-ai/d2l-en), [kubernetes/website](https://github.com/kubernetes/website) (`content/{zh-cn,en}/docs`), [Snailclimb/JavaGuide](https://github.com/Snailclimb/JavaGuide), [CyC2018/CS-Notes](https://github.com/CyC2018/CS-Notes), [jackfrued/Python-100-Days](https://github.com/jackfrued/Python-100-Days), [python/cpython](https://github.com/python/cpython) (`Lib`, `Objects`, `Doc`). Used only for local measurement; not in the repository and not used for training
- CS336 Assignment 4 repository: <https://github.com/stanford-cs336/assignment4-data>

**Next chapter**: Now we have the data and the tokenizer. The next step is to really feed hundreds of billions of tokens into 8 GPUs. Chapter 14, pretraining engineering, covers mixed precision, FlashAttention, data parallelism and FSDP, MFU, loss spikes, and resuming from checkpoints.
