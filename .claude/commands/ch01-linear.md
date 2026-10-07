---
description: "Chapter 1 self-check: y = ax + b and gradient descent (第 1 章自检：y = ax + b 与梯度下降)"
---

# Chapter 1 self-check: y = ax + b and gradient descent

The learner typed `/ch01-linear`. They finished Chapter 1 (`chapters/01-linear-regression/`). Help them check if they understand it. Do not give them the answers.

**Language**: Use the language of the learner. If the learner writes in Chinese, ask the questions and give feedback in Chinese. The Chinese text of the chapter is in `README.zh.md`. Write short, clear sentences (see `docs/STYLE_GUIDE.md`).

**Method**: Ask the questions below one at a time. Wait for the learner to answer in their own words. Then give feedback: tell them what is correct, what is not correct, and how to make their understanding deeper. Do not ask all the questions at the same time.

---

## Questions (from easy to difficult)

**Level 1: the four steps**

Ask the learner:
> Without your notes, name the four steps of training a model. For each step, write one sentence that tells what the step does in the example y = ax + b.

Expected answer: model (ŷ = ax + b); loss (the mean squared error, which measures how bad the predictions are); gradient (the partial derivatives of the loss for a and b, which point in the direction in which the loss increases fastest); update (each parameter minus the learning rate times its gradient). If a step is missing, ask about that step.

---

**Level 2: read a gradient**

Ask the learner:
> At one step, ∂L/∂b = 3.2 and ∂L/∂a = −0.5. Are the predictions of the model too high or too low on average? In the next step, does a increase or decrease? Does b increase or decrease?

Expected answer: ∂L/∂b is positive, so the mean residual is positive and the predictions are too high. b decreases (we subtract a positive number). a increases (we subtract a negative number).

---

**Level 3: learning rate**

Ask the learner:
> On the same data, a learning rate of 0.05 converges, but a learning rate of 0.11 diverges to 10⁹. What do the parameters do when they diverge? Why does a larger step move the parameters farther from the bottom of the valley?

Expected answer: a step that is too large goes past the bottom and lands at a higher point on the other side. There, the gradient is larger, so the next step is larger. The parameters oscillate, and each oscillation is larger than the last. Extra credit: the critical value 2/λ_max comes from how much the bowl curves in its steepest direction.

---

**Level 4: transfer**

Ask the learner:
> The PyTorch training loop has the line `optimizer.zero_grad()`. If you remove this line, what gradient does step 2 use? What happens to the training?

Expected answer: by default, PyTorch adds each new gradient to the old one. Without the line, the gradient in step 2 is the sum of the gradients of steps 1 and 2. The gradient becomes larger at each step. This has the same effect as a learning rate that increases. The training will probably diverge.

---

## Rules for feedback

- If the answer is correct: say so. Then ask a deeper "why" question.
- If the answer is not correct: do not give the answer. Give a hint. For example, ask the learner to change a parameter in `code/02_learning_rate.py` and run it. Then let them think again.
- If the learner says "I do not know": ask them to guess first. A wrong guess is better than no guess.

When the learner passes all four levels, tell them to continue to Chapter 2 (`chapters/02-from-scalar-to-matrix/`). After Chapter 2, they can check themselves with `/ch02-matrix`.
