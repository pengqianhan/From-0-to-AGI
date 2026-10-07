---
description: "Chapter 5 self-check: classification and probability — softmax, cross-entropy, maximum likelihood (第 5 章自检：分类与概率——softmax、交叉熵、最大似然)"
---

# Chapter 5 self-check: classification and probability

The learner typed `/ch05-classification`. They finished Chapter 5 (`chapters/05-classification-probability/`). Help them check if they understand it. Do not give them the answers.

**Language**: Use the language of the learner. If the learner writes in Chinese, ask the questions and give feedback in Chinese. The Chinese text of the chapter is in `README.zh.md`. Write short, clear sentences (see `docs/STYLE_GUIDE.md`).

**Method**: Ask the questions below one at a time. Wait for the learner to answer in their own words. Then give feedback: tell them what is correct, what is not correct, and how to make their understanding deeper. Do not ask all the questions at the same time.

---

## Questions (from easy to difficult)

**Level 1: from scores to loss**

Ask the learner:
> A three-class model outputs logits = [2, 1, −1]. The correct answer is class 3. Tell what each step from the logits to the loss does, and estimate the loss. What must the loss be, approximately, for a model that learned nothing?

Expected answer: softmax (apply the exponential, then divide by the sum) gives probabilities of about [0.71, 0.26, 0.04]. The cross-entropy = −ln p(correct class) = −ln 0.035 ≈ 3.35. A uniform random guess gives ln 3 ≈ 1.10. Extra credit: "cross-entropy is the log of the maximum likelihood with a minus sign".

---

**Level 2: intuition for the gradient**

Ask the learner:
> What is the gradient of softmax + cross-entropy for the logits? The correct answer is cat. The model gives cat 0.7, dog 0.26, and bird 0.04. After a gradient-descent step, in which direction does each of the three logits move? Which logit moves more?

Expected answer: gradient = p − onehot = [−0.3, 0.26, 0.04]. After we subtract the gradient, the cat logit increases (with a force of 1 − p = 0.3), and the dog and bird logits decrease. The dog logit decreases more than the bird logit, because the model gave the wrong probability mostly to dog. If the learner only recites the formula, ask: "Why is the force on a wrong class exactly p?"

---

**Level 3: why not MSE**

Ask the learner:
> The model pushed the probability of the correct answer down to 0.0001. It is confidently wrong. Compare the size of the gradient with cross-entropy and with MSE (softmax probabilities vs onehot). Why are they different?

Expected answer: the cross-entropy gradient is still near its maximum (about 1.41 in the example of the chapter). The MSE gradient is almost 0 (3.4 × 10⁻⁴ in the example). The reason: the MSE gradient goes through the derivative of softmax, diag(p) − ppᵀ, which contains factors of the probabilities. When the probability of the correct class is near 0, these factors make the gradient almost 0. In cross-entropy, the log cancels the exponential of softmax, so the gradient is exactly p − onehot. Extra credit: in the spiral experiment of the chapter, MSE started much more slowly. (The stuck time depends much on the machine: on the build machine, MSE was stuck for 700 to 800 steps; on another machine, it got out after 200 to 300 steps.) If the learner says "MSE cannot be used for classification", correct it to "MSE can train, but it is weakest when the model most needs a correction".

---

**Level 4: transfer to language models**

Ask the learner:
> A language model has a vocabulary of 50257 tokens. In the training log, the loss at the first step is 10.8. After training, it is 3.0 (nats). Why is it 10.8 at the first step? What is 3.0 nats in bits and as a perplexity, and what does each mean? What happens if the code applies softmax to the logits first and then passes them to `F.cross_entropy`?

Expected answer: 10.8 ≈ ln 50257, so the model guesses uniformly, and the code is probably correct. 3.0 nats ≈ 4.33 bits. The perplexity is e³ ≈ 20: at each step, the model hesitates among about 20 candidates. `F.cross_entropy` expects logits. If it gets probabilities, softmax occurs two times. There is no error, but the "scores" are between 0 and 1. The second softmax cannot make a sharp distribution, so the loss has a lower limit that it cannot go below (for three classes, −ln(e/(e+2)) ≈ 0.55). The gradient also becomes smaller, and training becomes slower. (Measured on the spiral data: the accuracy at step 1000 is 94.3%, and 99.0% with the correct code. The reported loss stays near 0.58.)

---

## Rules for feedback

- If the answer is correct: say so. Then ask a deeper "why" question.
- If the answer is not correct: do not give the answer. Give a hint. For example, ask the learner to run `code/02_cross_entropy.py` and look at the gradient table in part 4, or to change `code/05_pytorch_version.py` and try it. Then let them think again.
- If the learner says "I do not know": ask them to guess first. A wrong guess is better than no guess.

When the learner passes all four levels, tell them to continue to Chapter 6 (`chapters/06-training-stability/`). After Chapter 6, they can check themselves with `/ch06-training-stability`.
