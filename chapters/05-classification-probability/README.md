# Chapter 5: Classification and probability — From "predict a number" to "predict a class"

**English** · [中文](README.zh.md)

> **Goal**: After this chapter, you can write softmax and cross-entropy by hand. You can derive that the gradient of the two together is `p − onehot`. You can use numbers to show why classification does not use MSE. You can also explain why "a language model is a classifier over the vocabulary".

📺 **Video**: Not published yet. To render it on your computer, run `bash chapters/05-classification-probability/video/build.sh`.
🧪 **Self-check**: After the chapter, type `/ch05-classification` in Claude Code.

---

In the last chapter, we wrote automatic differentiation ourselves. For any network structure, if we can write the forward pass, autograd calculates the gradients. But until now, all our models output **one number**: a house price, or the y value on a curve. The loss was always the mean squared error.

This chapter solves a new problem: **what if the answer is not a number, but a class?** Does this image show a cat, a dog, or a bird? Is this email spam? And the question that is most important for this course: **what is the next character in this sentence?**

At the end of this chapter, you have the complete training objective of a large language model. The more than twenty chapters after this one become much more complex. But the last layer of the model always does what this chapter shows.

## 1. Intuition: why not predict the class number directly?

The laziest method is this: give cat, dog, and bird the numbers 0, 1, and 2. Then predict this number with regression, as in Chapter 1.

Problems occur immediately:

- **The numbers add an order that does not exist.** What does a prediction of 1.5 mean? "Between dog and bird"? If the model is not sure between cat and bird, regression outputs their mean, 1. But 1 is "dog".
- **What we want is "how sure" the model is.** "70% cat, 26% dog, 4% bird" is much more useful than one number. We can set a threshold, sort the classes, or sample the next character from the probabilities (Chapter 10).

Thus we use a different method: **the model outputs one number for each class, one score per class**. Then we change this set of scores into a probability distribution.

## 2. Logits and softmax: change scores into probabilities

The last layer of the model (`h @ W + b` from Chapters 2 and 3) outputs one real number for each class. These numbers are the **logits**. A logit can be positive or negative, and the logits do not add up to 1. The example for the whole chapter is:

```
logits z = [2.0, 1.0, −1.0]    # cat, dog, bird
```

To change the logits into probabilities, we need two things: each number must be positive, and the numbers must add up to 1. **Softmax** does both in the most direct way. First, it applies the exponential function (the result is always positive). Then it divides by the sum (the result always adds up to 1):

```
p_k = exp(z_k) / Σ_j exp(z_j)
```

Run [`code/01_softmax.py`](code/01_softmax.py):

```bash
uv run python chapters/05-classification-probability/code/01_softmax.py
```

| | Cat | Dog | Bird |
|---|---:|---:|---:|
| logits `z` | 2.0 | 1.0 | −1.0 |
| `exp(z)` | 7.3891 | 2.7183 | 0.3679 |
| softmax `p` | 0.7054 | 0.2595 | 0.0351 |

Remember these properties:

- **Softmax keeps the order**: a higher score gives a higher probability. Softmax is a "soft" version of argmax. It does not select only the largest score. It gives probabilities in proportion to the exponentials. This is the origin of the name.
- **Only the differences are important**: add the same constant to all logits, and the result does not change. The reason: `exp(z + c) = exp(z)·exp(c)`, and `exp(c)` cancels in the numerator and the denominator. The code adds 100 to all three logits. The result is still `[0.7054, 0.2595, 0.0351]`.
- **The exponential makes differences larger**: a difference of 1 in the logits gives a ratio of e ≈ 2.7 in the probabilities. A difference of 3 gives a ratio of about 20.

**Numerical stability: subtract the maximum first.** The exponential grows too fast. In float32, `exp(x)` overflows to `inf` when x is more than about 88.7 (709.8 in float64). When we train a large model, logits of tens or hundreds are not rare. Scale the logits up to `[1000, 500, −500]`. Then the calculation from the definition gives `[nan, 0, 0]`.

The solution is the property "only the differences are important". First, subtract the maximum from all logits. Then the largest term becomes `exp(0) = 1`, and it never overflows:

```python
def softmax(z, temperature=1.0):
    z = np.asarray(z, dtype=np.float64) / temperature
    z = z - z.max(axis=-1, keepdims=True)   # subtract the maximum: same result, but no overflow
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)
```

For `[1000, 500, −500]`, the stable version gives `[1, 0, 0]`. This result is correct. The softmax in all deep-learning frameworks uses this method.

## 3. Maximum likelihood: make the probability of the correct answer as large as possible

With probabilities, the objective of training follows directly: **the higher the probability that the model gives to the correct answer, the better**.

For one sample, the correct class is y, and the model gives it the probability `p_y`. Assume that the samples are independent. Then, for the full data set, the probability that the model "guesses all correct answers" is the product of the probabilities of all samples:

```
likelihood = Π_i p_{i, y_i}
```

Use this product as a function of the parameters, and find the parameters that make it largest. This method is **maximum likelihood estimation (MLE)**.

The product has a practical problem: it **underflows**. The code calculates it for a model that gives each correct answer the probability 0.9:

| Samples | Likelihood 0.9ⁿ | Log-likelihood n·ln 0.9 |
|---:|---:|---:|
| 10 | 3.487 × 10⁻¹ | −1.05 |
| 100 | 2.656 × 10⁻⁵ | −10.54 |
| 1000 | 1.748 × 10⁻⁴⁶ | −105.36 |
| 10000 | 2.470 × 10⁻³²³ | −1053.61 |

With 10000 samples, the product is already near the smallest positive number that float64 can represent. The training set of a language model has trillions of tokens. Thus we take the logarithm: **log changes a product into a sum**. Also, log is monotonically increasing, so to maximize the likelihood is the same as to maximize the log-likelihood. Add a minus sign so that "smaller is better", and divide by N to get the mean. The result is the **negative log-likelihood (NLL)**:

```
L = −1/N · Σ_i log p_{i, y_i}
```

## 4. Cross-entropy and its clean gradient

The NLL above has a more common name: **cross-entropy**. In general, the cross-entropy of the true distribution q and the model distribution p is `H(q, p) = −Σ_k q_k log p_k`. In classification, the true distribution is onehot: 1 for the correct class, 0 for all other classes. Then only one term stays in the sum, `−log p_y`. This term is exactly the NLL. Thus in classification, **"maximum likelihood", "minimize the negative log-likelihood", and "minimize the cross-entropy" are the same thing**.

Go back to the cat-dog-bird example. The table shows the loss for each possible correct answer:

| Correct answer | Probability p from the model | Loss −ln p |
|---|---:|---:|
| Cat | 0.7054 | 0.3490 |
| Dog | 0.2595 | 1.3490 |
| Bird | 0.0351 | 3.3490 |
| (uniform guess over 3 classes) | 1/3 | 1.0986 |

A correct and confident guess gives a small loss. The lower the probability of the correct answer, the larger the loss, with no upper limit: when p → 0, −ln p → ∞. The last row is also useful: **a model that learned nothing must have a loss of about ln(number of classes)**. This is the first check point for errors in training code. For the same reason, the spiral data later in this chapter has an initial loss of 1.0919 ≈ ln 3, and a language model has an initial loss of ≈ ln(vocabulary size).

The implementation must also be numerically stable. Do not calculate softmax first and then take the log: if a probability underflows to 0, the log gives −∞. Calculate **log-softmax** directly: `log p_k = z_k − logsumexp(z)`. Here too, subtract the maximum first:

```python
def log_softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=-1, keepdims=True))

def cross_entropy(logits, y):
    return -log_softmax(logits)[np.arange(len(y)), y].mean()
```

**Gradient: `p − onehot`.** This is the most important result of the chapter. For one sample, put softmax into the loss:

```
L = −log p_y = −z_y + log Σ_j exp(z_j)
```

Take the derivative for the k-th logit. The first term gives −1, but only when k = y. The derivative of the second term is `exp(z_k) / Σ_j exp(z_j)`, which is exactly `p_k`. Thus:

```
∂L/∂z_k = p_k − [k = y]      that is,   ∂L/∂z = p − onehot(y)
```

In the derivative, all the complexity of softmax and log cancels. What stays is "the prediction of the model minus the correct answer". It has the same form as the MSE residual `ŷ − y` in Chapter 1. In the code, it is one line:

```python
def ce_grad(logits, y):
    p = softmax(logits)
    p[np.arange(len(y)), y] -= 1.0      # p − onehot
    return p / len(y)                  # divide by N because the loss is a mean
```

Use the method from Chapters 1 and 4: a **gradient check** with central differences ([`code/02_cross_entropy.py`](code/02_cross_entropy.py)):

| | Cat | Dog | Bird |
|---|---:|---:|---:|
| Analytic gradient `p − onehot` (correct answer = cat) | −0.2946 | 0.2595 | 0.0351 |
| Numerical gradient (central difference) | −0.2946 | 0.2595 | 0.0351 |

The maximum difference is 2.18 × 10⁻¹². For 8 random samples × 5 classes, the maximum difference is 3.63 × 10⁻¹¹. The derivation is correct.

Read this gradient. Gradient descent pushes the logit of the correct class up (the gradient is negative, so to subtract it is to increase the logit). The force of the push is `1 − p_y`: it is equal to the distance that is still missing. Gradient descent pushes the logit of each wrong class down. The force is exactly the probability `p_k` that the model gave to that wrong class.

## 5. Why classification does not use MSE

We have probabilities now. Can we continue to use MSE and make `p` move toward the onehot target, with `L = Σ_k (p_k − t_k)²`? We can calculate this loss, and it can train. But it has a serious weakness: **when the model is "confidently wrong", MSE gives almost no gradient.**

Do an experiment. The true class is 0. The model raises the logit of the wrong class 1 to s, and the other logits are 0. A larger s means that the model is more sure of the wrong answer. The table shows the size of the gradient for the logits, for the two losses (part 4 of `02_cross_entropy.py`):

| Wrong-class logit s | p(correct) | CE loss | ‖CE gradient‖ | MSE loss | ‖MSE gradient‖ |
|---:|---:|---:|---:|---:|---:|
| 0 | 3.33 × 10⁻¹ | 1.099 | 0.8165 | 0.6667 | 5.44 × 10⁻¹ |
| 2 | 1.07 × 10⁻¹ | 2.240 | 1.1954 | 1.4290 | 5.08 × 10⁻¹ |
| 4 | 1.77 × 10⁻² | 4.036 | 1.3769 | 1.8959 | 1.23 × 10⁻¹ |
| 6 | 2.47 × 10⁻³ | 6.005 | 1.4090 | 1.9852 | 1.83 × 10⁻² |
| 8 | 3.35 × 10⁻⁴ | 8.001 | 1.4135 | 1.9980 | 2.51 × 10⁻³ |
| 10 | 4.54 × 10⁻⁵ | 10.000 | 1.4141 | 1.9997 | 3.40 × 10⁻⁴ |

The two gradient columns go in opposite directions:

- **Cross-entropy**: the worse the error, the larger the gradient. At the end, the gradient stays at √2 ≈ 1.414 (the length of `p − onehot` ≈ `[−1, 1, 0]`). The loss also grows linearly with s. The model always "feels the pain".
- **MSE**: the worse the error, the smaller the gradient. At s = 10, the gradient is only 3.4 × 10⁻⁴, more than 4000 times smaller than the cross-entropy gradient. The MSE loss has an upper limit of 2 (two probabilities are each wrong by 1). At this limit, the loss surface is flat.

The reason is in the chain rule. The MSE gradient goes through the derivative of softmax, `∂p/∂z = diag(p) − ppᵀ`. Each term of this matrix is multiplied by some `p_k`. When the probability of the correct class is near 0, this factor also makes the gradient almost 0.

This is the same type of effect as the "saturation" of sigmoid at its two ends, where the derivative goes to 0. (Chapter 3 showed the shape of sigmoid.) In cross-entropy, the `log` cancels the exponential of softmax. Thus the gradient is exactly `p − onehot`, and it does not saturate.

> **Note:** The numbers of the training experiments in this chapter come from one CPU run on the course build machine. Different machines and different versions of the low-level math libraries use a slightly different order of floating-point operations. A few hundred training steps make these small differences larger. On your computer, the numbers can be different from the second or third decimal place. Trust the conclusions below that do not depend on exact values. For a second run on another server in 2026-10, see [runs/2026-10-01-gpu0-check/chapters-01-06.md](../../runs/2026-10-01-gpu0-check/chapters-01-06.md).

This effect is not only in one table. We use the spiral classifier from Section 6. On purpose, we increase the standard deviation of the output-layer weights to 10. Then the model makes "confident random guesses" from the start. We train it once with cross-entropy and once with MSE (the same network and the same learning rate, 1.0).

In the table below, the first three columns are the output on the course build machine. The last column is the MSE result of the same code on another server in 2026-10. In that second run, the cross-entropy column was identical to the build machine, digit for digit.

| Step | Cross-entropy training: accuracy | MSE training: accuracy | MSE training: accuracy (second run, another server) |
|---:|---:|---:|---:|
| 0 | 30.7% | 30.7% | 30.7% |
| 100 | 85.7% | 65.3% | 53.3% |
| 500 | 97.3% | 65.3% | 96.7% |
| 800 | 98.0% | 66.0% | 99.3% |
| 1000 | 99.0% | 95.3% | 99.3% |
| 3000 | 99.3% | 99.3% | 99.0% |

On the build machine, the MSE version **was stuck near 65% for 700 to 800 steps**. At step 500, the cross-entropy version had 0 samples with "a probability below 1% for the correct answer". The MSE version had 103 such samples. One third of the data was confidently wrong, but it almost could not move the parameters.

On the other server, the MSE version was stuck for a much shorter time. At step 100, it had only 53.3%, with 136 samples at p(correct) < 1%. But it got out after 200 to 300 steps, and at step 500 only 7 such samples were left (the cross-entropy version still had 0).

This result is a lesson in itself: **small experiments are very sensitive to numerical details**. The matrix multiplications on the two machines differ only by very small rounding errors. But in the saturated region, MSE pushes the gradient to almost 0. Then these small rounding differences decide which samples "flip" first. Thus the stuck time changed from 700–800 steps to 200–300 steps. Cross-entropy does not saturate, and its trajectory was identical on the two machines, digit for digit.

Do not use "the number of steps that MSE was stuck" as a conclusion. Two results are true in both runs. First, cross-entropy pushes steadily from the first step. Second, MSE starts much more slowly: at first, many samples are confidently wrong, but they cannot move the parameters. This is the case that the gradient table above shows.

In both runs, MSE got out at the end (this problem is small). But for a large model, you cannot pay for these hundreds of steps.

With a normal initialization (the initial predictions are almost uniform), both losses learn the task. After 3000 steps, both have an accuracy of 99.3%. Only the cross-entropy of the MSE version is a little higher (0.0490 vs 0.0293). Thus a more exact statement is: **MSE for classification is not "unusable". But it is weakest when the model most needs a correction.**

Golik et al. made a systematic comparison on speech recognition in 2013. Their conclusion is the same: from a random initialization, a network that trains with the squared error does not converge to a good solution. Cross-entropy also has a clean probabilistic meaning (maximum likelihood). For these reasons, cross-entropy became the default loss for classification.

## 6. Train a classifier: three spirals

Now put all the parts together. The data is three spiral arms that cross each other (a classic toy data set from CS231n). Each class has 100 points. The code makes the data, so you do not need to download anything.

A straight line cannot separate the spirals. Thus the model is the two-layer MLP from Chapter 3: 2 → 64 (ReLU) → 3. We write backpropagation by hand, as in Chapter 4. The only change is that the gradient of the last layer is now `p − onehot`:

```python
logits, cache = forward(params, X)             # z = ReLU(X W1 + b1) W2 + b2
grads = backward(params, cache, ce_grad(logits, y))   # start the backward pass from dlogits = (p − onehot)/N
for k in params:
    params[k] -= lr * grads[k]
```

Run [`code/03_train_classifier.py`](code/03_train_classifier.py) (a few seconds on a CPU):

```bash
uv run python chapters/05-classification-probability/code/03_train_classifier.py
```

| Step | Cross-entropy | Training accuracy |
|---:|---:|---:|
| 0 | 1.0919 | 30.7% |
| 100 | 0.2641 | 93.0% |
| 200 | 0.1513 | 96.3% |
| 500 | 0.0813 | 99.0% |
| 1000 | 0.0526 | 99.3% |
| 3000 | 0.0293 | 99.3% |

- **The initial loss is 1.0919 ≈ ln 3 = 1.0986.** The initial output-layer weights are very small, so at the start the model guesses almost uniformly. This agrees with the expectation from Section 4.
- **On a separate test set** (the same spirals, different random noise), the accuracy is 99.0%.
- **Comparison: remove the hidden layer.** A linear softmax classifier (`logits = XW + b`, also called multinomial logistic regression) gets only 54.0% after 3000 steps. The boundary between each pair of classes is a straight line, and straight lines cannot separate the spirals. Classification also needs the nonlinearity from Chapter 3.

The video shows this process. Each point of the plane has the color of the class that the model predicts. At the start, almost the full plane has one color. During training, the regions of the three colors slowly curl into three spirals.

## 7. Temperature: make the distribution sharper or flatter

Softmax has one more control: divide the logits by a **temperature T** before softmax.

| Temperature T | Cat | Dog | Bird |
|---:|---:|---:|---:|
| 0.5 | 0.8789 | 0.1189 | 0.0022 |
| 1.0 | 0.7054 | 0.2595 | 0.0351 |
| 2.0 | 0.5465 | 0.3315 | 0.1220 |
| 10.0 | 0.3780 | 0.3420 | 0.2800 |

A smaller T makes the differences larger and the distribution "sharper". When T → 0, softmax becomes argmax. A larger T makes the differences smaller and the distribution "flatter". When T → ∞, it becomes the uniform distribution.

In training, T is usually 1. Temperature becomes important in Chapter 10. When a language model generates text, the temperature decides between "always select the most probable character" and "try other characters more boldly".

## 8. A language model is a classifier over the vocabulary

This is the most important section of the chapter. **A language model does one thing: it looks at the previous characters and guesses the next character.** Use "the next character" as the class, and you have a classification problem. The number of classes is the size V of the **vocabulary**.

The last layer of the model outputs one logit for each candidate token. Softmax gives the probability distribution of the next token. The loss is the cross-entropy of the correct next token. That is all.

[`code/04_next_token.py`](code/04_next_token.py) shows this with a short Chinese text. Each Chinese character is one token. The model is the simplest type: a V × V table. The row of the "current character" contains the logits of the next character (Chapter 7 explains this **bigram** model in detail). Training uses the `cross_entropy` and `ce_grad` functions from above, with no changes:

```bash
uv run python chapters/05-classification-probability/code/04_next_token.py
```

The corpus has 104 characters, the vocabulary has V = 65 characters, and there are 103 training pairs:

| Step | Loss (nats) | Loss (bits) | Perplexity |
|---:|---:|---:|---:|
| 0 | 4.1744 | 6.0224 | 65.00 |
| 10 | 0.9503 | 1.3709 | 2.59 |
| 50 | 0.5027 | 0.7253 | 1.65 |
| 500 | 0.4622 | 0.6668 | 1.59 |

For comparison, count the frequencies directly: `p(next character | current character) = count / total count`. This gives a loss of 0.4585, and gradient descent moves toward this value. **For a bigram model, "count the frequencies" is the closed-form maximum-likelihood solution**, like the least-squares solution in Chapter 1. What the model learned also agrees with the counts. After 「分」 ("divide"), the probability of 「数」 is 0.67 and the probability of 「类」 is 0.33. The reason: in the corpus, 分数 ("score") occurs two times and 分类 ("classification") occurs one time.

The table uses three units for the same quantity:

- **nats**: the cross-entropy with the natural logarithm ln. This is the number that we calculated in this whole chapter.
- **bits**: the same quantity with log base 2, `bits = nats / ln 2`. It means "the mean number of binary digits that are still necessary to describe each token". A lower loss means better compression. Chapter 7 uses a related metric, **bits per byte (BPB)**. It divides the bits over the bytes of the raw text. Then we can compare models with different tokenizers fairly.
- **perplexity**: `exp(nats) = 2^bits`. The intuition is "the model hesitates uniformly among this number of candidates". At step 0, the perplexity is 65, which is exactly the vocabulary size: a random guess among 65 characters. After training, it is 1.59: at each step, the model hesitates between only one or two candidates.

At a real scale: the GPT-2 vocabulary has 50257 tokens. The loss of a model before training must be near ln 50257 = 10.82 nats (15.62 bits). Later, you will read training logs. If the loss at the first step is about 10.8, you know that the model "guesses uniformly", and the code is probably correct.

## 9. Summary

- **logits**: the model gives each class a score, which can be any real number.
- **softmax**: `p_k = exp(z_k)/Σexp(z_j)` changes the scores into a probability distribution. In the implementation, subtract the maximum first.
- **Maximum likelihood**: make the probability of the correct answer as large as possible. The log changes the product into a sum. The result is **negative log-likelihood = cross-entropy**, `−log p_y`.
- **Gradient**: `∂L/∂z = p − onehot`, the prediction minus the answer.
- **No MSE**: when the model is confidently wrong, the MSE gradient vanishes. The cross-entropy gradient does not.
- **Temperature**: divide the logits by T to control how sharp the distribution is.
- **Language model**: classification over the vocabulary, and the loss is the cross-entropy. Nats, bits, and perplexity are three forms of the same quantity.

---

## From minimal code to production code

For Chapters 1–6, the production code is the standard PyTorch code. See [`code/05_pytorch_version.py`](code/05_pytorch_version.py):

```python
model = nn.Sequential(nn.Linear(2, 64), nn.ReLU(), nn.Linear(64, 3))
optimizer = torch.optim.SGD(model.parameters(), lr=1.0)

for step in range(3000):
    logits = model(X)                    # 1. forward pass: get the logits (no softmax!)
    loss = F.cross_entropy(logits, Y)    # 2. cross-entropy: log-softmax + NLL, fused inside
    optimizer.zero_grad()                # 3. set the gradients to zero
    loss.backward()                      # 4. backward pass: autograd gets (p − onehot)/N
    optimizer.step()                     # 5. update
```

The results (all parity checks in float64):

| Parity check | PyTorch | Our NumPy version |
|---|---:|---:|
| `F.cross_entropy`, 8 random samples × 5 classes | 3.1764553511 | 3.1764553511 |
| Gradient: autograd vs manual `(p − onehot)/N` | max difference 6.94 × 10⁻¹⁸ | — |
| `label_smoothing=0.1` vs manual `−Σ q_k log p_k` | 3.1633053590 | 3.1633053590 |
| Spiral classifier, cross-entropy after 3000 steps (same initialization) | 0.0293 | 0.0293 (difference 7.85 × 10⁻¹¹) |
| logits `[1000, 500, −500]`, correct answer = class 3 | 1500.0 | the naive code `−log(softmax)` gives `inf` |

The table shows what the production code adds to the minimal code and why:

| Minimal code | Production code | Why |
|---|---|---|
| `softmax`, then `−log` (or a manual `log_softmax`) | `F.cross_entropy(logits, y)` takes the logits directly | It fuses log-softmax and NLL into one operator and uses logsumexp for numerical stability. **The most common bug is to apply softmax first and then pass the result in.** This bug gives no error. It only makes the training quietly bad. |
| Manual `ce_grad` | `loss.backward()` | Autograd gets the same `p − onehot` (identical digit by digit in the table above). |
| The target is onehot | `label_smoothing=ε` | It changes the target to "1 − ε + ε/K for the correct class, ε/K for each other class". This stops the model from pushing a logit to infinity and becoming overconfident (Szegedy et al. 2016). It is optional. The pretraining of the main-line model in this course does not use it. |
| Calculate the loss for every sample | `ignore_index=-100` | Some positions do not count in the loss. The loss mask of SFT in Chapter 16 uses this: calculate the loss only on the answer, not on the question. |
| logits with the shape `(N, K)` | In a language model, the logits have the shape `(B, T, V)`. Flatten them to `(B·T, V)` | Each position of each sequence is one classification. nanoGPT and minimind both have this line in their model code: `F.cross_entropy(logits.view(-1, V), targets.view(-1), ignore_index=...)` |
| float64 | Mixed-precision training (BF16) | Matrix multiplications use BF16. But PyTorch automatic mixed precision runs `cross_entropy`, `log_softmax`, and `softmax` in float32 automatically. The exponential and the log are sensitive to precision (Chapter 14). |

The last part of the script is the code from a language model. The logits have the shape `(2, 4, 50257)` and are all 0 (a uniform guess). After flattening, the loss is 10.8249. This is exactly ln 50257.

---

## Guided questions

Ask Claude Code these questions. Continue until you can explain the answers in your own words:

1. For two classes, softmax has only two logits, `z₁, z₂`. Prove that `p₁ = sigmoid(z₁ − z₂)`. What does this tell you about the relation between sigmoid and softmax?
2. Why does `F.cross_entropy` need logits as input, not probabilities? What happens if you pass the probabilities after softmax? (Hint: then softmax occurs two times. Change `05_pytorch_version.py` and look at the loss and the accuracy.)
3. The difference between the cross-entropy `H(q, p)` and the entropy `H(q)` is the **KL divergence**. In classification, q is onehot. What is its entropy? What does this tell you about the relation between minimizing the cross-entropy and minimizing the KL divergence? (Logit distillation in Chapter 17 and DPO in Chapter 18 use the KL divergence.)
4. Can the cross-entropy on the training set decrease to 0? For a loss of 0, what must the logits look like? How is this related to the problem that label smoothing tries to solve?
5. The intuition for a perplexity of 1.59 is "the model hesitates uniformly among 1.59 candidates". A language model has a perplexity of 20 on Chinese text and 3 on code. What does this tell you? Why can we not compare the perplexity of models with different tokenizers directly? (This is the reason for bits per byte in Chapter 7.)

## Hands-on tasks

For each task, run the code and look at the result.

**Task 1 (basic)**: In `01_softmax.py`, check that softmax goes to the onehot of argmax when T → 0, and to the uniform distribution when T → ∞. Print the results for T = 0.01 and T = 1000. Then explain why T cannot be 0 in `softmax(z / T)`.

**Task 2 (core)**: In `03_train_classifier.py`, change the number of classes to 5 (`make_spirals(n_classes=5)`, and change the network output to 5). First, predict the initial loss. Then run the script to check your prediction. Record the accuracy after 3000 training steps. If the accuracy is much worse, try a larger hidden layer or more training steps.

**Task 3 (challenge)**: The MSE comparison in `02_cross_entropy.py` uses "softmax probabilities vs onehot". Another common wrong method is "regress the logits directly onto the onehot target" (no softmax, `L = Σ(z_k − t_k)²`). Train with this loss on the spiral data and record the accuracy. Then use the terms of this chapter to explain: is its problem the same type as the problem of "softmax + MSE"?

---

## References

- Goodfellow, Bengio, Courville. *Deep Learning*, Section 5.5, "Maximum Likelihood Estimation", and Section 6.2.2, "Softmax Units for Multinoulli Output Distributions": <https://www.deeplearningbook.org/contents/ml.html>, <https://www.deeplearningbook.org/contents/mlp.html>
- Stanford CS231n. *Putting it together: Minimal Neural Network Case Study* (three-class spiral data, a linear softmax classifier vs a two-layer network): <https://cs231n.github.io/neural-networks-case-study/>
- Golik, Doetsch, Ney. *Cross-Entropy vs. Squared Error Training: a Theoretical and Experimental Comparison*, Interspeech 2013: <https://www.isca-archive.org/interspeech_2013/golik13_interspeech.html>
- Szegedy et al. *Rethinking the Inception Architecture for Computer Vision* (the source of label smoothing), 2016: <https://arxiv.org/abs/1512.00567>
- Hinton, Vinyals, Dean. *Distilling the Knowledge in a Neural Network* (softmax with temperature; distillation in Chapter 17 uses it again), 2015: <https://arxiv.org/abs/1503.02531>
- Radford et al. *Language Models are Unsupervised Multitask Learners* (GPT-2, vocabulary of 50257 tokens): <https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf>
- PyTorch documentation, `torch.nn.functional.cross_entropy` (the input is unnormalized logits; the parameters `label_smoothing` and `ignore_index`): <https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.cross_entropy.html>
- PyTorch documentation, *Automatic Mixed Precision* (the list "CUDA Ops that can autocast to float32" includes `cross_entropy`, `log_softmax`, and `softmax`): <https://docs.pytorch.org/docs/stable/amp.html>
- [nanoGPT](https://github.com/karpathy/nanoGPT) `model.py` and [minimind](https://github.com/jingyaogong/minimind) `model/model_minimind.py`: in both, the language-model loss is one line, `F.cross_entropy`.

**Next chapter**: The spiral classifier has only two layers, and it trains without problems. But what if we make the network deeper, with more than ten or even tens of layers? Then the loss either does not move at all, or it suddenly explodes to NaN. The "vanishing gradient" of MSE in this chapter is only a small part of this problem. In Chapter 6, we look at initialization, normalization, residual connections, AdamW, and learning-rate schedules. These methods make deep networks trainable.
