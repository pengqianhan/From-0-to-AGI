# Chapter 7: Language modeling and tokenization — From "predict the next character" to byte-level BPE

**English** · [中文](README.zh.md)

> **Goal**: After this chapter, you can write a byte-level BPE tokenizer by hand. With this tokenizer and a bigram language model from counts, you can calculate the bits-per-byte of a text. You can also explain why we compare models with different tokenizers by bpb, not by the loss or the perplexity.

📺 **Video**: Not published yet. To render it on your computer, run `bash chapters/07-tokenization-language-model/video/build.sh`.
🧪 **Self-check**: After the chapter, type `/ch07-tokenization` in Claude Code.

---

At the end of Part 1, we have a full set of tools: gradient descent, neural networks, automatic differentiation, softmax, and cross-entropy. We also have the methods that make deep networks train stably: initialization, normalization, residual connections, and AdamW. The end of Chapter 5 told us this: **a language model is a classifier over the vocabulary**. It looks at the earlier text and gives a score to each candidate for the "next token". Softmax changes the scores into probabilities, and cross-entropy trains the model.

But Chapter 5 took a shortcut. It used single characters as the classes, and the vocabulary was the 65 characters in one short text. This chapter answers a new question: **how do we split real text into tokens?** By character? By byte? By word?

The method of splitting sets the vocabulary size and the sequence length. It also decides if the model can meet "unknown characters". It even decides if we can use the loss to compare different models.

First, we write language modeling in terms of probability. Then we write a byte-level BPE tokenizer by hand. The tokenizers of GPT-2, Llama 3, Qwen, and DeepSeek are all of this type. Then we use a bigram model, which looks only at the previous token. With it, we calculate all the numbers that evaluate a language model: nats, bits, perplexity, and bits-per-byte.

## 1. Language modeling: give a probability to a full text

A **language model** answers this question: what is the probability of a text `x_1, x_2, …, x_T`?

We cannot give a probability to a full text directly, because the number of possible sentences has no limit. But probability theory has the **chain rule**, which splits this probability into parts:

```
p(x_1, x_2, …, x_T) = p(x_1) · p(x_2 | x_1) · p(x_3 | x_1, x_2) · … · p(x_T | x_1, …, x_{T−1})
                    = Π_t p(x_t | x_<t)
```

This is not an approximation. It is an identity. It changes "give a score to the full text" into T small problems: "predict the next token from the earlier text". Each small problem is a classification problem from Chapter 5. The classes are the V tokens in the vocabulary.

Take the logarithm, and the product becomes a sum. In training, we minimize the mean negative log-likelihood. This is the **cross-entropy of the correct next token at each position**:

```
L = −(1/T) · Σ_t log p(x_t | x_<t)
```

This is the training objective of all GPT-type models. Its name is **next-token prediction**. From attention in Chapter 8 to pretraining in Chapter 14, the models become more complex, but the loss is always this one line.

Models differ in how they calculate `p(x_t | x_<t)`. This chapter uses the simplest method: **look only at the previous token**:

```
p(x_t | x_<t) ≈ p(x_t | x_{t−1})
```

This is a **bigram model**. It cannot do much, but it is sufficient to explain tokenization and evaluation.

Before that, we must solve a more basic problem: what is `x_t`?

## 2. Split by character: simple, but the Chinese vocabulary is large

A model knows only integers. The most direct method is the **character level**: give each different character an id. Chapter 5 used this method.

```bash
uv run python chapters/07-tokenization-language-model/code/01_chars_and_bytes.py
```

The counts on the tiny corpus of this course (`assets/tiny_corpus/`) are:

| Corpus | Characters | UTF-8 bytes | Character-level vocabulary | Bytes / character |
|---|---:|---:|---:|---:|
| English (Shakespeare) | 1,115,394 | 1,115,394 | 65 | 1.00 |
| Chinese (classical poetry) | 428,876 | 1,200,167 | 5,297 | 2.80 |
| Code | 139,005 | 159,673 | 870 | 1.15 |

English needs only 65 characters (uppercase and lowercase letters, punctuation, and whitespace). Chinese is completely different. Chinese writing has no alphabet: each character is a separate symbol for a word or a part of a word. The poetry corpus has more than one million bytes, and it contains 5,297 different characters. Unicode has tens of thousands of Chinese characters. The character level has two problems:

1. **The vocabulary is large, and it has no upper limit.** With Chinese, English, code, and emoji, a character-level vocabulary easily has more than 10,000 entries. And there is always one more character that the model did not see.
2. **What about characters that the model did not see?** Use the first 90% of the Chinese corpus as the training set. Then the last 10% contains 198 characters (129 distinct ones) that do not occur in the training set, for example 亘 and 企. A character-level model must map all of them to one `<unk>` (unknown) symbol. The information is lost.

## 3. Split by byte: the vocabulary is always 256, and nothing is "unknown"

A computer stores text with the **UTF-8** encoding. UTF-8 changes each character into 1 to 4 bytes. A byte is an integer from 0 to 255.

| Character | Code point | UTF-8 bytes |
|---|---|---|
| `A` | U+0041 | `[65]` (1 byte) |
| `é` | U+00E9 | `[195, 169]` (2) |
| `学` ("learn") | U+5B66 | `[229, 173, 166]` (3) |
| `🤖` | U+1F916 | `[240, 159, 164, 150]` (4) |

ASCII characters (English letters, digits, and common punctuation) use 1 byte. **Common Chinese characters use 3 bytes.** The reason: UTF-8 uses 3 bytes for each code point from U+0800 to U+FFFF, and the common Chinese characters are in this range. This gives a second method of splitting, the **byte level**: use the UTF-8 bytes directly as tokens. The vocabulary is always 256.

```python
def byte_ids(text: str) -> list[int]:
    return list(text.encode("utf-8"))     # each byte is the id (0–255); no vocabulary is necessary
```

`学而时习之` (a line from the *Analects* of Confucius: "learn, and practice it often") has 5 characters. It has 5 ids at the character level and 15 ids at the byte level (3 bytes for each character): `[229, 173, 166, 232, 128, 140, …]`. The advantage: the byte level can encode any text, for example rare characters, emoji, corrupted text, or binary data. **`<unk>` never occurs**, and `bytes(ids).decode("utf-8")` gives back the text with no loss.

The disadvantage of the byte level: the sequences become longer. For Chinese, the sequence is 2.8 times as long as at the character level. In Chapter 8, we will see that the compute of attention increases with the square of the sequence length. Long sequences are expensive.

The character level has a large vocabulary and `<unk>`. The byte level has a small vocabulary and long sequences. Is there a method with the advantages of both?

## 4. Byte-level BPE: merge the most frequent pair into a new token

**BPE (Byte Pair Encoding)** was originally a data compression algorithm. In 2015, Sennrich et al. used it for tokenization in machine translation. GPT-2 (2019) changed it to operate on UTF-8 bytes. This is **byte-level BPE**. Its training is one sentence:

> Start from the 256 bytes. **Merge the most frequent pair of adjacent tokens in the corpus into a new token, again and again**, until the vocabulary has the size that you want.

For example, in English, `h` often comes after `t`. So we merge `(t, h)` into a new token `th` (id 256). Then `e` often comes after `th`, so we merge them into `the`, and so on. Each merge adds 1 to the vocabulary and decreases the number of tokens in the corpus.

The final tokenizer has two properties:

- The initial vocabulary contains all 256 bytes. Thus **the tokenizer can encode any text**, and `<unk>` never occurs. We keep the advantage of the byte level.
- Frequent pieces (common words, common Chinese characters, code keywords) become one token, so **the sequences become shorter**. We also get the advantage of the character level.

### 4.1 BPE by hand

The full code is in [`code/02_bpe.py`](code/02_bpe.py). The core of the training part is this loop:

```python
words = Counter(pretokenize(text))                   # keep one copy of each chunk, with its count
seqs = [list(w.encode("utf-8")) for w in words]      # each chunk → a byte sequence
...
for new_id in range(256, vocab_size):
    pair = max(stats, key=stats.get)                 # the most frequent adjacent pair
    self.merges[pair] = new_id                       # remember this merge
    self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]   # bytes of the new token = the two joined
    for i in where.pop(pair):                        # update only the chunks that contain this pair
        ... seqs[i] = merge(seqs[i], pair, new_id)   # replace each (a, b) with new_id, and update the pair counts
```

A chunk is a piece of text from pre-tokenization (Section 4.2). `stats` maps each adjacent pair to its count. `where` records the chunks in which each pair occurs. Thus each merge updates only the affected part, and we do not count the full corpus again.

To encode new text, **replay the merges in the order of training**. Of all the adjacent pairs in the current sequence, find the merge that training learned first, and do it. Continue until no pair can merge. Decoding is simpler: join the bytes of each id, then decode them as UTF-8.

```bash
uv run python chapters/07-tokenization-language-model/code/02_bpe.py
```

The training text is the first 60,000 characters of each corpus (English, Chinese, and code), 305,302 bytes in total. We do 768 merges (vocabulary 256 → 1024). In pure Python, this takes a few seconds (it depends on the machine). The table shows **the first 20 merges**. In the table, `␣` is a space, and `\xef\xbc` is a piece of bytes that cannot show as a character. `，` and `。` are the Chinese comma and full stop.

| id | New token | Count | id | New token | Count |
|---:|---|---:|---:|---|---:|
| 256 | `\xef\xbc` | 7,212 | 266 | `\xe5\x85` | 1,703 |
| 257 | `␣␣` | 6,000 | 267 | `\xe4\xba` | 1,693 |
| 258 | `，` | 4,985 | 268 | `\xe2\x80` | 1,635 |
| 259 | `\xe3\x80` | 4,074 | 269 | `␣t` | 1,595 |
| 260 | `。` | 3,859 | 270 | `子` | 1,429 |
| 261 | `\xe4\xb9` | 2,931 | 271 | `he` | 1,294 |
| 262 | `\xe4\xb8` | 2,901 | 272 | `不` | 1,228 |
| 263 | `\xe5\xad` | 1,871 | 273 | `in` | 1,200 |
| 264 | `␣␣␣␣` | 1,853 | 274 | `\xe4\xbb` | 1,193 |
| 265 | `之` | 1,720 | 275 | `\xe6\x9c` | 1,161 |

Look at this table carefully:

- **The first merge is `\xef\xbc`.** It is not a full character. It is the first two bytes that the full-width punctuation marks share. Chinese text uses full-width punctuation: the comma `，` is `ef bc 8c`, the colon `：` is `ef bc 9a`, and so on. Then id 258 merges `\xef\xbc` with `\x8c` into the full `，`.
- **Many merges are "half characters" of Chinese.** Many common Chinese characters share the same first two bytes, for example `\xe4\xb9` and `\xe4\xb8`. Thus BPE merges these 2-byte prefixes first. Only after them come frequent full characters: 之 (a common word in classical Chinese, "of" or "it"), 子 ("child" or "master"), and 不 ("not").
- **The indentation in code** (`␣␣`, `␣␣␣␣`) merges early. The English pieces `␣t`, `he`, and `in` are also near the top.
- Longer tokens that BPE learns later include `␣return`, `␣import`, `␣Citizen`, `MENENIUS` (the name of a character in a Shakespeare play), and a long `----------------`. **What BPE learns depends completely on what is frequent in the training corpus.**

Encoding examples:

| Text | Bytes | Tokens | Split result |
|---|---:|---:|---|
| `To be, or not to be` | 19 | 7 | `To` `␣be` `,` `␣or` `␣not` `␣to` `␣be` |
| `学而时习之，不亦说乎` | 30 | 12 | `学` `而` `时` `\xe4\xb9` `\xa0` `之` `，` `不` `亦` `\xe8\xaf` `\xb4` `乎` |
| `def forward(self, x):` | 21 | 11 | `def` `␣for` `w` `ard` `(` `self` `,` `␣` `x` `)` `:` |

`学而时习之，不亦说乎` is the full line from the *Analects*: "To learn and to practice it often: is that not a pleasure?" It has 10 characters of 3 bytes each. In this small vocabulary of 1024, BPE did not merge 习 ("practice") and 说 (here "pleasure") into full characters. Thus each of them becomes two tokens: one 2-byte prefix and one single byte. This is the fallback of byte-level BPE: **a character that BPE did not learn falls back to bytes**, and no information is lost.

### 4.2 Pre-tokenization: merges do not cross "word" boundaries

If we run BPE directly on the full text, it learns tokens such as `e␣t` and `.␣The`. These tokens cross the boundaries of words and punctuation, so they waste vocabulary entries. Thus all tokenizers since GPT-2 first use a regular expression for **pre-tokenization**. The regex splits the text into "chunks": runs of letters, digits, runs of punctuation, and whitespace. BPE merges only inside a chunk:

```python
PATTERN = re.compile(r"'(?:s|t|re|ve|m|ll|d)| ?[^\W\d_]+|\d| ?(?:[^\s\w]|_)+|\s+(?!\S)|\s+")
```

The regex splits `"To be, or not to be? 学而时习之。x = 2026\n"` into `'To' ' be' ',' ' or' ' not' ' to' ' be' '?' ' 学而时习之' '。' 'x' ' =' ' ' '2' '0' '2' '6' '\n'`. Note two points. First, the space before a word stays attached to the word (thus tokens such as `␣be` exist). The regex counts Chinese characters as letters, and Chinese text has no spaces between words, so `␣学而时习之` is one chunk.

Second, **digits are split one by one**, so "2026" is always 4 tokens. Qwen splits digits one by one. The advantage: each digit has the same representation everywhere, which helps arithmetic. Llama 3 and DeepSeek-V3 split digits into groups of up to 3 (see "Adopters and sources" at the end of this page).

### 4.3 Compression on the validation set

We tested the hand-written BPE with a vocabulary of 1024 on the last 20,000 characters of each corpus. Training did not see these characters.

| Corpus | Bytes | Tokens | Bytes / token |
|---|---:|---:|---:|
| English | 20,000 | 11,353 | 1.76 |
| Chinese | 55,928 | 32,209 | 1.74 |
| Code | 21,946 | 11,236 | 1.95 |

**Bytes per token** is a common measure of the compression of a tokenizer. A larger value means that the same text needs fewer tokens. Pure byte level gives 1.00. Here, only 768 merges already give 1.7–2.0. What happens with a larger vocabulary?

## 5. Vocabulary size: the trade-off between sequence length and embedding parameters

[`code/04_vocab_and_zero.py`](code/04_vocab_and_zero.py) uses the production trainer (`zero/tokenizer.py`, see below). It trains on the full corpora with a range of vocabulary sizes. Then it measures bytes per token on the validation set:

```bash
uv run python chapters/07-tokenization-language-model/code/04_vocab_and_zero.py
```

| Vocabulary V | English | Chinese | Code | Total validation tokens |
|---:|---:|---:|---:|---:|
| 272 (no merges) | 1.00 | 1.00 | 1.00 | 97,874 |
| 512 | 1.70 | 1.53 | 1.44 | 63,625 |
| 1,024 | 2.05 | 2.02 | 1.72 | 50,102 |
| 2,048 | 2.40 | 2.47 | 1.98 | 42,048 |
| 4,096 | 2.74 | 2.93 | 2.37 | 35,615 |
| 8,192 | 3.01 | 3.35 | 2.82 | 31,116 |
| 16,384 | 3.24 | 3.78 | 3.20 | 27,840 |
| 32,768 | 3.36 | 4.09 | 3.54 | 25,823 |

(272 = 256 bytes + 16 special tokens.) From 272 to 1,024, the sequence length decreases by almost half. After that, each doubling gives a smaller gain. From 16K to 32K, the total number of tokens decreases by only 7%. **Diminishing returns** is the first rule of vocabulary size.

What is the cost? Each token in the vocabulary needs a vector of d dimensions. Thus **the embedding matrix has V × d parameters**. (When the input and output share the embedding, count it only once. Chapter 9 explains this.) The numbers from the `config.json` of each model give:

| Model | Vocabulary V | Width d | Embedding parameters V × d |
|---|---:|---:|---:|
| GPT-2 (124M) | 50,257 | 768 | 38.6M |
| Qwen3-0.6B | 151,936 | 1,024 | 155.6M |
| **Qwen3.5-0.8B** | **248,320** | **1,024** | **254.3M** |
| Llama 3.2 1B | 128,256 | 2,048 | 262.7M |
| Main-line model of this course (not final) | 65,536 | 1,280 | 83.9M |

In Qwen3.5-0.8B, the embedding alone has about 250M parameters. That is a large part of a model with a nominal size of 0.8B. For a large model, these parameters are a small cost. A large vocabulary saves much sequence length, so it is a good trade.

**For a small model, the embedding takes parameters that the Transformer layers could use.** Thus the vocabulary of the main-line model of this course is 65,536 for now, much smaller than the vocabulary of Qwen. This value is not final. Chapter 13 decides it after it measures the compression on real Chinese, English, and code data.

There are two less visible costs. First, with a larger vocabulary, each token occurs fewer times in the training data, so the embeddings of rare tokens do not get sufficient training. Second, the output softmax must calculate over V classes, so a larger V is slower.

## 6. The bigram language model: counting gives the maximum likelihood

Now we have a tokenizer, and we can train a language model. The parameters of a bigram model are a V × V table. Row a, column b holds the probability that the next token is b when the current token is a. Chapter 5 showed this: **for a bigram model, the maximum-likelihood solution is the frequency count**. No gradient descent is necessary:

```
p(b | a) = count(a, b) / Σ_j count(a, j)
```

The full code is in [`code/03_bigram.py`](code/03_bigram.py). The count needs only one line:

```python
def count_bigrams(ids, V):
    a = np.asarray(ids)
    flat = a[:-1] * V + a[1:]                                 # encode (previous, next) as one integer
    return np.bincount(flat, minlength=V * V).reshape(V, V)   # counts[previous, next]
```

The direct frequency has a problem. A pair in the validation set that training did not see gets probability 0, and `−log 0` is infinite. One such pair is sufficient. Thus we need **smoothing**: add a small number α to each cell.

```python
def bigram_prob(counts, prev, nxt, alpha=0.03):
    V = counts.shape[0]
    return (counts[prev, nxt] + alpha) / (counts.sum(axis=1)[prev] + alpha * V)
```

We chose α = 0.03 from 1, 0.3, 0.1, 0.03, and 0.01. We made the choice on a part of the text that training and validation do not use. This part is from the 40,000th-last to the 20,000th-last character of each corpus.

To generate text from a bigram model, sample the next token from the row of the current token, again and again. The training text is each corpus without its last 40,000 characters. Sampling uses no smoothing. In the table, `⏎` is a newline.

| Corpus · tokenization | Sample (starts from a newline) |
|---|---|
| English · character | `In weatee ther?⏎⏎FLES:⏎rure by arort me pre ther t` |
| English · BPE | `MINIUS: I arself.⏎⏎With say.⏎⏎IDUMIRIARYous, all` |
| Chinese · character | `晏殊⏎雪可得见好容。⏎有客长城子⏎张先⏎糜鹿鸣，鞗革有多媚，明。⏎翠绾鬓。⏎报。⏎见桃李藉栽双语。` |
| Chinese · BPE | `凭�。⏎云徊⏎月千花如之终共。⏎拾辱。⏎�` |

The model learned some local patterns. In English, it learned the format of a play: "name + colon + newline". In Chinese, it learned the format "the name of the poet on its own line, then the lines of the poem". (晏殊 Yan Shu and 张先 Zhang Xian are names of poets.) But the full text has no meaning, because **the model looks only at the previous token**.

The Chinese BPE sample also contains `�`. The model generated the first two bytes of a Chinese character, but the next byte did not make valid UTF-8. This occurs only in byte-level models. Real large models also sometimes output this type of corrupted text.

## 7. Evaluation: nats, bits, perplexity, and bits-per-byte

To evaluate a language model, calculate the mean cross-entropy on validation text that the model did not see. Chapter 5 showed three forms of the same quantity:

- **nats / token**: the cross-entropy with the natural logarithm, `−(1/T) Σ ln p(x_t | x_{t−1})`.
- **bits / token**: `nats / ln 2`. It is "the mean number of binary digits that each token still needs".
- **perplexity**: `e^nats = 2^bits`. It is "the number of equally likely candidates that the model hesitates between".

All three numbers are "per token". This causes a problem: **different tokenizers make tokens of different lengths**. One BPE token covers 1.76 bytes on average, and one byte-level token covers only 1 byte. Each BPE token is harder to predict, so its loss is higher. But each BPE token also predicts more content. A comparison of the loss per token is like a comparison of the effort per step of two walkers whose steps have different lengths.

The solution is a different denominator: divide the total loss by the number of **bytes** of the original text.

```
bits-per-byte (bpb) = Σ_t (−ln p(x_t | …)) / (ln 2 × total bytes that these tokens cover)
                    = (bits / token) ÷ (bytes / token)
```

The number of bytes of the same validation text does not change, for all methods of splitting. Thus **bpb does not depend on the tokenizer**, and we can compare models across tokenizers. These lines in the code calculate it:

```python
nats = -np.log(bigram_prob(counts, x, y))                 # −ln p(correct next token) at each position
total_nats, total_bytes = nats.sum(), sum(n_bytes[1:])   # count only the tokens that the model predicts
ce = total_nats / len(y)                                  # nats / token
bits = ce / math.log(2)                                   # bits / token
ppl = math.exp(ce)                                        # perplexity
bpb = total_nats / (math.log(2) * total_bytes)            # bits / byte
```

For each of the three corpora, `03_bigram.py` trains one bigram model for each of the three tokenizations. Then it evaluates each model on the validation set. (The BPE is the tokenizer with a vocabulary of 1024 from Section 4.1.)

| Corpus | Tokenization | Vocabulary V | Validation tokens | nats / token | bits / token | Perplexity | **bpb** |
|---|---|---:|---:|---:|---:|---:|---:|
| English | Byte | 256 | 19,999 | 2.543 | 3.669 | 12.7 | 3.669 |
| English | Character | 66 | 19,999 | 2.543 | 3.668 | 12.7 | 3.668 |
| English | BPE | 1,024 | 11,352 | 3.610 | 5.208 | 37.0 | **2.956** |
| Chinese | Byte | 256 | 55,927 | 2.810 | 4.054 | 16.6 | 4.054 |
| Chinese | Character | 5,172 | 19,999 | 5.507 | 7.945 | 246.4 | **2.841** |
| Chinese | BPE | 1,024 | 32,208 | 3.809 | 5.495 | 45.1 | 3.164 |
| Code | Byte | 256 | 21,945 | 2.567 | 3.704 | 13.0 | 3.704 |
| Code | Character | 815 | 19,999 | 2.683 | 3.871 | 14.6 | 3.528 |
| Code | BPE | 1,024 | 11,235 | 3.844 | 5.546 | 46.7 | **2.839** |

How to read this table:

- **Perplexity alone gives the opposite conclusion.** On English, the byte-level bigram has a perplexity of 12.7, and BPE has 37.0. The byte level seems much better. But in bpb, BPE gives 2.956 and the byte level gives 3.669: BPE is 19% better. The reason: one BPE token covers 1.76 bytes, so "the previous token" of the bigram model contains more of the earlier text.
- **For English, the byte level and the character level are almost the same.** The Shakespeare text is almost all ASCII, so one character is one byte. The two are the same model.
- **For Chinese, the character level is the best (2.841).** A BPE vocabulary of 1024 is too small for Chinese. Many Chinese characters still split into byte pieces (习 and 说 in Section 4.1). Then the "1 token" of context of the bigram model is only half a character, and it is wasted. The character level has 5,172 characters with one token each, which is a good match. But the character level has a small advantage here: the 58 unseen characters in the validation set all map to the same `<unk>`. It is easier to predict `<unk>` than to predict which rare character it is.
- **All bpb values are far below 8.** A uniform random guess of one byte needs 8 bits. A count table that looks only at the previous token already compresses the text to about 3 bits/byte.

bpb is not a toy measure for this chapter only. Karpathy's nanochat evaluates the validation measure `val_bpb` at regular intervals during pretraining. Its function `evaluate_bpb` does the same thing as above: it adds up the loss and the bytes of each target token, but not of special tokens. At the end, it divides the two sums. In Chapter 13, the main-line model of this course compares different vocabulary sizes. There, too, we must use bpb, not the loss.

## 8. Summary

- **Language modeling**: the chain rule splits the probability of a full text into `Π p(x_t | x_<t)`. The training objective is the cross-entropy of the next token at each position. This is the classification over the vocabulary from Chapter 5.
- **Character level**: a large vocabulary (5,000+ characters for Chinese) and `<unk>`. **Byte level**: the vocabulary is always 256 and `<unk>` never occurs, but Chinese sequences are 2.8 times as long.
- **Byte-level BPE**: start from the 256 bytes and merge the most frequent adjacent pair, again and again. Pre-tokenization keeps merges inside chunks. To encode, replay the merges in the order of training.
- **Vocabulary size**: a larger vocabulary gives shorter sequences, but with diminishing returns. The embedding parameters V × d increase linearly. Small models must calculate this cost carefully.
- **Bigram**: it looks only at the previous token. The frequency count is the maximum-likelihood solution. It needs smoothing.
- **Evaluation**: nats, bits, and perplexity are "per token", so we cannot compare them across tokenizers. **bits-per-byte** divides by the number of bytes, so it does not depend on the tokenizer.

The largest problem of the bigram model is in its definition: **it looks only at the previous token**. Look at `学而时习之，不亦说乎` again. To predict the last character 乎 (a question word), the model must know that `不亦说` ("is that not a pleasure") comes before it. To follow the meter of a full poem, the model must remember several earlier lines. If we extend the context from 1 token to 2 or 3 tokens, the table has V³ or V⁴ cells, and soon we cannot store it. Attention, in the next chapter, lets the model see all earlier tokens without a huge table.

---

## From minimal code to production code

Each idea of the minimal code has a matching part in the production code of the main-line model:

| Minimal code | Production code (`zero/`) | What it adds, and why |
|---|---|---|
| `BPE.train` in `02_bpe.py` (pure Python, a few seconds on a 305 KB corpus) | `train_bpe(texts, vocab_size, special_tokens)` in `zero/tokenizer.py`, based on Hugging Face `tokenizers` (Rust) | The main-line tokenizer must train on corpora of several GB, and pure Python is too slow. For the same 768 merges, one measurement gave 0.74 s for zero and 4.55 s for the hand-written version. A new run on a different server in 2026-10 gave 1.08 s vs 1.62 s. On a corpus this small, the ratio changes much with the machine and its load |
| `PATTERN`: a simplified regex for Python `re`, with `[^\W\d_]` for letters | `PRETOKENIZE_REGEX`: exactly the same regex as in the `tokenizer.json` of Qwen2/Qwen3, with `\p{L}` and `\p{N}` | The Unicode categories are more exact. Digits (`\p{N}`) split one by one. It agrees with Qwen, so we can compare the two directly |
| No normalization | `normalizers.NFC()` | It combines forms such as "e + combining accent" into one code point. Then the same character does not become different tokens because it has different forms (the same as Qwen3) |
| No special tokens | 16 special tokens at the fixed ids 0–15: `<|endoftext|>`, `<|im_start|>`/`<|im_end|>` (chat), `<tool_call>`/`</tool_call>`, `<tool_response>`/`</tool_response>` (tool calls), `<think>`/`</think>`, and 7 reserved slots | The chat and tool-call formats (Chapter 16) need boundary markers that BPE does not split. zero encodes `<|im_start|>` as one id (1). The hand-written version splits it into 8 normal tokens. With the reserved slots, we can add special tokens later without a change to the vocabulary size |
| Calculate `bytes / tokens` by hand | `Tokenizer.bytes_per_token(texts)` | Chapter 13 uses it to compare vocabulary sizes on Chinese, English, and code |
| The `bpb` formula in `03_bigram.py` | **zero does not have it yet**: the validation in `zero/train/trainer.py` reports only the loss per token. The plan is to add it in the same way as nanochat `evaluate_bpb` (add up the loss and the bytes of each target token; do not count special tokens) | When Chapter 13 compares vocabulary sizes, the loss is not comparable, but bpb is |
| Encode a short text in memory | `write_shards` in `zero/data/shard.py`: it tokenizes the text and writes `np.uint32` shards, **with one `<|endoftext|>` after each document**. The metadata records the tokenizer hash `Tokenizer.hash()` | Pretraining cannot tokenize at each step. `<|endoftext|>` teaches the model that "the document ends here". The hash prevents training a model with tokenizer B on data from tokenizer A. The shards use uint32 because the vocabulary can be larger than 65,535 |
| None | `Tokenizer.save_hf(out_dir)` | Exports files that `AutoTokenizer` in transformers can read directly. The release needs them (Chapter 20) |

**Parity check**: the same training text (60,000 characters of each of the three corpora) and the same 768 merges. Bytes per token on the validation set:

| | English | Chinese | Code |
|---|---:|---:|---:|
| Hand-written BPE (`02_bpe.py`) | 1.76 | 1.74 | 1.95 |
| `zero/tokenizer.py` | 1.86 | 1.79 | 1.97 |

Both restore the validation set byte for byte, and they split digits in exactly the same way (`x = 2026` → `x` `␣=` `␣` `2` `0` `2` `6`). We expected a small difference in compression. The regexes are different (`\p{L}` vs `[^\W\d_]`). The rules that choose a pair when two counts are equal are also different. Both differences change the merge order a little.

The tests in `tests/test_tokenizer.py` check the quality of the production tokenizer (`uv run pytest tests/test_tokenizer.py`; all 12 tests pass on this machine). The tests check these points:

- **Encode, then decode, gives back the original text** for Chinese, English, code, emoji, full-width spaces, and whitespace at the start and end.
- The ids of the special tokens are fixed, and BPE does not split them.
- `"2026"` encodes as 4 tokens.
- `bytes_per_token` is larger than 1.5 for Chinese, English, and code, and larger than 2.0 for Chinese.
- Save and load do not change the encoding or the hash.
- After export, `AutoTokenizer` in transformers gives exactly the same encoding as zero.

---

## Adopters and sources

| Technique | Adopters (main versions) | Source |
|---|---|---|
| Byte-level BPE (the 256 bytes as the base, no `<unk>`) | GPT-2; Llama 3 (128K vocabulary: 100K from tiktoken + 28K non-English tokens); Qwen (`tokenizer.json` of Qwen2/Qwen3/Qwen3.5: `ByteLevel` pre-tokenizer + `BPE` model); DeepSeek-V3 (128K, `ByteLevel` + `BPE`); OLMo 2 (`ByteLevel` + `BPE`) | GPT-2 paper, Section 2.2; Llama 3 paper; the `tokenizer.json` in the Hugging Face repository of each model (links in the references) |
| Regex pre-tokenization | Since GPT-2; the `tokenizer.json` files of Qwen3, Llama 3.2, DeepSeek-V3, and OLMo 2 all have a `Split` regex | Same as above |
| Digit splitting | Qwen3 / Qwen3.5: `\p{N}` (one by one); Llama 3.2, DeepSeek-V3, OLMo 2: `\p{N}{1,3}` (groups of up to 3) | Same as above (read in 2026-09) |
| Vocabulary size | GPT-2 50,257; Llama 3 family 128,256; Qwen3 151,936; Qwen3.5 248,320 (`vocab_size` in `config.json`) | `config.json` of each model |
| bits-per-byte as a measure across tokenizers | nanochat (`val_bpb`, `evaluate_bpb` in `nanochat/loss_eval.py`) | <https://github.com/karpathy/nanochat> |

The Gemma family uses a SentencePiece tokenizer, so it is not in the table. Is it also a "BPE with byte fallback"? Use its technical report as the reference (to be verified).

---

## Guided questions

Ask Claude Code these questions. Continue until you can explain the answers in your own words:

1. The chain rule `p(x_1…x_T) = Π p(x_t | x_<t)` is an identity, not an approximation. At which step does the bigram model make its approximation? How large is the table of a trigram model, which looks at the previous 2 tokens? Can we store it when V = 65,536?
2. Why is the first merge of BPE `\xef\xbc` and not a Chinese character? If the training corpus is all English, what is the first merge probably? Change `02_bpe.py` to test your answer.
3. "The BPE model with a perplexity of 37 is better than the byte-level model with a perplexity of 12.7." Explain this sentence with the definition of bpb. If two models use the same tokenizer, can a comparison of the loss and a comparison of bpb give different conclusions?
4. In the Chinese bigram of this chapter, the character level (bpb 2.841) is better than BPE with a vocabulary of 1024 (3.164). What do you expect if you increase the BPE vocabulary to 8,192? Why?
5. The embedding of Qwen3.5-0.8B has about 250M parameters. Assume that the total parameter budget is fixed at 0.8B. You decrease the vocabulary from 248K to 65K. What can you do with the parameters that you save? What is the cost?
6. Why do we add `<|endoftext|>` after each document in the pretraining data? If we do not add it, what occurs when the model generates text?

## Hands-on tasks

For each task, run the code and look at the result.

**Task 1 (basic)**: In `02_bpe.py`, change `vocab_size` from 1024 to 2048 and to 4096. Record the training time and the bytes per token on the validation set for the three corpora. Check if 习 and 说 become single tokens.

**Task 2 (core)**: Add a **unigram** baseline to `03_bigram.py`. A unigram model does not look at the earlier text at all: `p(x_t) = count(x_t) / N`. Calculate the bpb of the three tokenizations on the three corpora, and compare it with the bigram. How many bits/byte does "look at the previous token" save?

**Task 3 (challenge)**: In `03_bigram.py`, replace the BPE with a tokenizer with a vocabulary of 8,192 from `zero.tokenizer.train_bpe` (`04_vocab_and_zero.py` shows how). Calculate the bpb on the three corpora again. With the larger vocabulary, the bigram table has 8,192² ≈ 67 million cells, but the training data has only a little more than one million tokens. What do you see? Tune the smoothing α, and explain what you find. This problem, "not sufficient data to fill the table", is one reason why the next chapter uses a neural network (not a count table) as the model.

---

## Go deeper: CS336

This chapter matches Stanford CS336 (Spring 2026), <https://cs336.stanford.edu/>:

- **Lecture 1: Overview and tokenization**. It goes from the character level, the byte level, and the word level to BPE. It follows the same line as this chapter, but it discusses the advantages and disadvantages of each tokenization method more systematically.
- **Lecture 2: PyTorch and resource accounting**. "Embedding parameters = V × d" in Section 5 of this chapter is only a small part of resource accounting. Lecture 2 teaches you to calculate all the parameters, memory, and FLOPs of a model (Chapter 12 of this course uses this again).
- **The BPE part of Assignment 1 (Basics)**: implement the training, encoding, and decoding of byte-level BPE from zero. You must train on TinyStories and OpenWebText within limits of time and memory. This is much more difficult than `02_bpe.py` in this chapter: you need parallel pre-tokenization and incremental updates of the counts. The code of this chapter is a good warm-up. Assignment repository: <https://github.com/stanford-cs336/assignment1-basics>

---

## References

- Sennrich, Haddow, Birch. *Neural Machine Translation of Rare Words with Subword Units* (BPE for tokenization), 2015: <https://arxiv.org/abs/1508.07909>
- Radford et al. *Language Models are Unsupervised Multitask Learners* (GPT-2, byte-level BPE, Section 2.2): <https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf>
- Llama Team, Meta. *The Llama 3 Herd of Models* (128K vocabulary = 100K from tiktoken + 28K non-English tokens), 2024: <https://arxiv.org/abs/2407.21783>
- Karpathy. minbpe (a minimal byte-level BPE, with a reproduction of the GPT-4 tokenizer): <https://github.com/karpathy/minbpe>
- Karpathy. nanochat (`evaluate_bpb` in `nanochat/loss_eval.py`): <https://github.com/karpathy/nanochat>
- Hugging Face `tokenizers` documentation: <https://huggingface.co/docs/tokenizers>
- The tokenizer and configuration files of each model (read in 2026-09):
  - Qwen3-0.6B: <https://huggingface.co/Qwen/Qwen3-0.6B/blob/main/tokenizer.json>, <https://huggingface.co/Qwen/Qwen3-0.6B/blob/main/config.json>
  - Qwen3.5-0.8B: <https://huggingface.co/Qwen/Qwen3.5-0.8B/blob/main/config.json>, <https://huggingface.co/Qwen/Qwen3.5-0.8B/blob/main/tokenizer.json>
  - DeepSeek-V3: <https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/tokenizer.json>
  - Llama 3.2 1B (an unofficial mirror; the official repository needs an access request): <https://huggingface.co/unsloth/Llama-3.2-1B/blob/main/tokenizer.json>
  - OLMo 2: <https://huggingface.co/allenai/OLMo-2-1124-7B/blob/main/tokenizer.json>
  - GPT-2: <https://huggingface.co/openai-community/gpt2/blob/main/config.json>
- UTF-8 encoding rules: <https://en.wikipedia.org/wiki/UTF-8>
- CS336 Assignment 1 repository: <https://github.com/stanford-cs336/assignment1-basics>
- [nanoGPT](https://github.com/karpathy/nanoGPT), [minimind](https://github.com/jingyaogong/minimind): both start language models from character-level or BPE tokenization

**Next chapter**: A bigram model sees only the previous token. To predict 乎 in `不亦说乎`, the model must look back at `不亦`. To write a poem that rhymes, it must remember several earlier lines. In Chapter 8, we start from "a weighted average of all earlier tokens" and derive **attention**. Attention lets each position decide which part of the earlier text to look at.
