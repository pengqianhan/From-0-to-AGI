---
description: "Chapter 4 self-check: backpropagation and automatic differentiation (第 4 章自检：反向传播与自动微分)"
---

# Chapter 4 self-check: backpropagation and automatic differentiation

The learner typed `/ch04-backprop`. They finished Chapter 4 (`chapters/04-backprop-autograd/`). Help them check if they understand it. Do not give them the answers.

**Language**: Use the language of the learner. If the learner writes in Chinese, ask the questions and give feedback in Chinese. The Chinese text of the chapter is in `README.zh.md`. Write short, clear sentences (see `docs/STYLE_GUIDE.md`).

**Method**: Ask the questions below one at a time. Wait for the learner to answer in their own words. Then give feedback: tell them what is correct, what is not correct, and how to make their understanding deeper. Do not ask all the questions at the same time.

---

## Questions (from easy to difficult)

**Level 1: concepts — what the forward pass and the backward pass do**

Ask the learner:
> Without your notes, tell what the automatic differentiation engine does in the "forward" phase and in the "backward" phase. What is the purpose of `_prev` and of `_backward` in a `Value` node?

Expected answer: in the forward pass, each operation makes a new node. The node calculates its value and records the nodes that it comes from (`_prev`) and its operation. In this way, the engine records the full computational graph. In the backward pass, the engine starts from the loss (gradient set to 1). It calls the `_backward` of each node in reverse topological order. Each `_backward` adds "upstream gradient × local derivative" to the `grad` of the input nodes. If "topological order" or "local derivative" is missing, ask about it.

---

**Level 2: intuition — calculate on the graph by hand**

Ask the learner:
> For `L = (a·b + c)²` with a = 1, b = 2, and c = −1, first calculate the intermediate values of the forward pass. Then go from right to left and calculate ∂L/∂a, ∂L/∂b, and ∂L/∂c. Which local derivative do you multiply by at each step?

Expected answer: d = a·b = 2, e = d + c = 1, L = 1. ∂L/∂e = 2e = 2. The addition sends the gradient without change, so ∂L/∂d = ∂L/∂c = 2. The multiplication multiplies by the other input, so ∂L/∂a = 2 × b = 4 and ∂L/∂b = 2 × a = 2. Ask the learner to verify the result with `01_engine.py`.

---

**Level 3: find the problem — `+=` and `zero_grad`**

Ask the learner:
> Someone changed each `self.grad += ...` in `Value` to `self.grad = ...`. The code still runs, and the MLP still "trains". For `y = x·x + x` (x = 3), what gradient does this person get? Why is it wrong? What is the fastest way to find this bug? Also: `+=` is correct, so why must the training loop call `zero_grad()` at each step?

Expected answer: the graph uses x three times. The correct gradient is the sum of the three paths: 3 + 3 + 1 = 7. With `=`, the gradient that arrives later replaces the gradient that arrived first. Only the contribution of one path stays (the engine of this chapter gives 3). A gradient check (a parity check against central-difference numerical gradients) finds the bug. In the chapter, the engine with the bug has relative errors of about 0.97 / 0.93. `+=` is necessary to add the gradients of several paths in the same backward pass. But the gradients of different training steps must not add up. The framework cannot tell the two cases apart, so each step must set the gradients to zero before the backward pass. The same is true for PyTorch.

---

**Level 4: transfer — from scalars to tensors**

Ask the learner:
> Our scalar engine trains a network with 97 parameters, and each step builds 3720 nodes. How does PyTorch do the same work faster? For `Y = X·W` (X has the shape (N, d_in), W has the shape (d_in, d_out)), the upstream gradient G = ∂L/∂Y is known. Write ∂L/∂X and ∂L/∂W, and check the shapes. Why does deep learning use reverse mode and not forward mode?

Expected answer: in a tensor framework, one matrix multiplication is one node, and its backward pass is also a matrix multiplication (the vector-Jacobian product). The framework never builds the full Jacobian, and BLAS or the GPU does the work. ∂L/∂X = G·Wᵀ ((N, d_out)·(d_out, d_in) = (N, d_in)). ∂L/∂W = Xᵀ·G ((d_in, N)·(N, d_out) = (d_in, d_out)). Training has only one scalar loss but a very large number of parameters. One pass of reverse mode gives the gradients for all parameters. One pass of forward mode gives the derivative for only one input.

---

## Rules for feedback

- If the answer is correct: say so. Then ask a deeper "why" question. For example: "Why must the order be reverse topological order? Can any order work?"
- If the answer is not correct: do not give the answer. Give a hint. For example, ask the learner to print the `grad` of the intermediate nodes in `01_engine.py`, or to run `02_grad_check.py`. Then let them think again.
- If the learner says "I do not know": ask them to guess first. A wrong guess is better than no guess.

When the learner passes all four levels, tell them to continue to Chapter 5 (`chapters/05-*/`, classification and probability: softmax and cross-entropy).
