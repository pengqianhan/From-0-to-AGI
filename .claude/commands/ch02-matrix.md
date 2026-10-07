---
description: "Chapter 2 self-check: from scalars to matrices — vectors, matrix multiplication, shape rule, y = XW + b (第 2 章自检：从标量到矩阵——向量、矩阵乘法、形状规则、y = XW + b)"
---

# Chapter 2 self-check: from scalars to matrices

The learner typed `/ch02-matrix`. They finished Chapter 2 (`chapters/02-from-scalar-to-matrix/`). Help them check if they understand it. Do not give them the answers.

**Language**: Use the language of the learner. If the learner writes in Chinese, ask the questions and give feedback in Chinese. The Chinese text of the chapter is in `README.zh.md`. Write short, clear sentences (see `docs/STYLE_GUIDE.md`).

**Method**: Ask the questions below one at a time. Wait for the learner to answer in their own words. Then give feedback: tell them what is correct, what is not correct, and how to make their understanding deeper. Do not ask all the questions at the same time.

---

## Questions (from easy to difficult)

**Level 1: clear concepts**

Ask the learner:
> Without your notes, explain the "shape rule" of matrix multiplication in one sentence: when can you multiply two matrices, and what is the shape of the result? Then tell why the inner dimension must be equal.

Expected answer: "(m, k) times (k, n) gives (m, n), and the inner k must be the same". The reason: each cell of the result is the dot product of a row of A and a column of B. Thus the two vectors must have the same length. If the learner cannot give the reason, tell them to go back to `code/02_matrix_multiply.py` and look at the calculation that it prints.

---

**Level 2: intuition**

Ask the learner:
> A neural network has this line of code: `Y = X @ W + b`. X is (32, 128), W is (128, 64), and b is (64,). Without a calculation, give the shape of Y. Explain what 32, 128, and 64 represent. If the next batch has only 7 samples, which shapes change?

Expected answer: Y is (32, 64). 32 is the batch size (the number of samples), 128 is the number of input features, and 64 is the number of outputs. With 7 samples, only the first dimension of X and Y changes to 7. W and b do not change: the shapes of the parameters do not depend on the batch size.

---

**Level 3: find the problem**

Show the learner this code, and ask them where the bug is:

```python
X = np.random.randn(4, 3)
W = np.random.randn(3, 5)
b = np.zeros(3)
y = X @ W + b
```

Expected answer: `X @ W` is (4, 5). The shape of b must be `(5,)` (one bias for each output), not `(3,)`. To broadcast (4, 5) + (3,), NumPy aligns the shapes from the last dimension. 5 ≠ 3, so broadcasting gives an error.

Follow-up question (extra credit): the label `y` has the shape `(4,)`, and the prediction `y_hat` has the shape `(4, 1)`. What does `y_hat - y` give? Expected answer: it broadcasts to (4, 4). There is no error, but the loss is completely wrong. This bug is more dangerous than a bug that gives an error.

---

**Level 4: transfer**

Ask the learner:
> 1. An RGB image has 32×32 pixels. After you flatten it, how many dimensions does the vector have? You want a linear layer that changes it into scores for 10 classes. What is the shape of W? If you use `nn.Linear` in PyTorch, what is the shape of its `weight`?
> 2. The gradient of multivariate linear regression is `∂L/∂W = 2/N · Xᵀ(ŷ − y)`. Do not use the formula from memory. How can you use shapes to check that the formula at least "looks correct"?

Expected answer:
1. 32×32×3 = 3072 dimensions. W is (3072, 10). The `weight` of `nn.Linear(3072, 10)` is (10, 3072), because PyTorch keeps it as (output, input), and the forward pass calculates `x @ weight.T + bias`.
2. Xᵀ is (k, N) and the residual is (N, 1), so the product is (k, 1), the same shape as W. A gradient must have the same shape as its parameter, because each parameter needs its own gradient.

---

## Rules for feedback

- If the answer is correct: say so. Then ask a deeper "why" question. For example: why is vectorization tens to thousands of times faster than a Python loop? Why are two linear layers together still one layer?
- If the answer is not correct: do not give the answer. Give a hint. For example, ask the learner to print `.shape` in Python, or to run `code/01_matrix_basics.py` and read the broadcasting error message. Then let them think again.
- If the learner says "I do not know": ask them to guess first. A wrong guess is better than no guess.

When the learner passes all four levels, tell them to continue to Chapter 3: nonlinearity and neural networks (`chapters/03-neural-network/`). Use this question as an introduction: `03_linear_layer.py` shows that two linear layers are equal to one linear layer. How can a model fit a curve?
