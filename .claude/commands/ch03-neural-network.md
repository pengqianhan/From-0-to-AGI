---
description: "Chapter 3 self-check: nonlinearity and neural networks — stacked linear layers, activation functions, two-layer MLP (第 3 章自检：非线性与神经网络——线性叠加、激活函数、两层 MLP)"
---

# Chapter 3 self-check: nonlinearity and neural networks

The learner typed `/ch03-neural-network`. They finished Chapter 3 (`chapters/03-neural-network/`). Help them check if they understand it. Do not give them the answers.

**Language**: Use the language of the learner. If the learner writes in Chinese, ask the questions and give feedback in Chinese. The Chinese text of the chapter is in `README.zh.md`. Write short, clear sentences (see `docs/STYLE_GUIDE.md`).

**Method**: Ask the questions below one at a time. Wait for the learner to answer in their own words. Then give feedback: tell them what is correct, what is not correct, and how to make their understanding deeper. Do not ask all the questions at the same time.

---

## Questions (from easy to difficult)

**Level 1: stacked linear layers are still linear**

Ask the learner:
> A network has three linear layers, 1→8→8→1, with 97 parameters in total. It has no activation function between the layers. Can it fit y = sin(2x)? Use one formula to explain why.

Expected answer: no. `(X·W1 + b1)·W2 + b2 = X·(W1·W2) + (b1·W2 + b2)`. A composition of linear layers is still a linear layer. Three layers also merge, one at a time, into `X·W + b`. The graph is a straight line, so the MSE cannot be lower than the best straight line (0.4341 in this chapter). Extra credit: the learner mentions one of these facts. In the code, the layer-by-layer result and the merged result differ by only 10⁻¹⁵. After training, two linear layers stop at 0.4341.

---

**Level 2: what one ReLU unit looks like**

Ask the learner:
> The contribution of one hidden unit to the output is `v · ReLU(w·x + b)`, with w = 2, b = −1, and v = −3. Draw (or describe) this line. Where is the kink? What does the line look like on the left of the kink and on the right of the kink?

Expected answer: the kink is at x = −b/w = 0.5. When x < 0.5, `w·x + b < 0`, so ReLU gives 0, and this part is flat at 0. When x > 0.5, the output is `−3·(2x − 1)`, a straight line that goes down with a slope of −6. The learner may calculate the kink as b/w, or get the direction wrong. Then ask them to put in x = 0 and x = 1 and check.

---

**Level 3: width and dead units**

Ask the learner:
> In this chapter, the loss of the width-2 network stops at 0.3155, only a little better than 0.4341 for the straight line. For width 8, the loss is anywhere from 0.0064 to 0.0628 with different random seeds. Explain each of these two results.

Expected answer: width 2 gives at most two kinks and three straight segments. sin(2x) has four bends in [−3, 3]. The model does not have sufficient capacity, so no amount of training can build the curve. Width 8 is barely sufficient, and it has no extra kinks. Three things affect the result: the start positions of the kinks; whether a unit "dies" (z < 0 on all data, so ReLU′ is always 0 and the parameters never update again); and whether the kinks are close together (with seed 0 in this chapter, 4 kinks are close together in 2.2–3.0). Width 64 has many extra kinks, so it is much more stable.

---

**Level 4: transfer**

Ask the learner:
> Replace ReLU with SiLU (`z · σ(z)`). Is the statement of Section 5 still true ("one unit = one kink, and the output is piecewise linear")? In the code with manual gradients, which line must change, and what is the new line? Why do modern large models use SiLU/GELU more often than ReLU?

Expected answer: the output is no longer strictly piecewise linear. SiLU is smooth everywhere, so one unit adds a "bend with a round corner". But the intuition "many bends add up to a curve" does not change. In the gradient code, only the `act_grad` line must change: replace ReLU′(z) with SiLU′(z) = σ(z) + z·σ(z)·(1 − σ(z)). The rest of the chain rule does not change at all. Reasons to use SiLU/GELU: they are smooth, and the gradient is not exactly 0 when z < 0 (this makes the dying-ReLU problem less severe). Also, the feed-forward layers of large models use SwiGLU (gating + SiLU), which Chapter 9 explains. The learner does not have to remember the derivative of SiLU. It is sufficient to say "change only the line with the derivative of the activation function".

---

## Rules for feedback

- If the answer is correct: say so. Then ask a deeper "why" question.
- If the answer is not correct: do not give the answer. Give a hint. For example, ask the learner to change the width, the seed, or the activation function in `code/03_mlp_numpy.py` and run it. Or ask them to print some points with the `hinge` function in `code/02_activations.py`. Then let them think again.
- If the learner says "I do not know": ask them to guess first. A wrong guess is better than no guess.

When the learner passes all four levels, tell them to continue to Chapter 4 (`chapters/04-backprop-autograd/`). After Chapter 4, they can check themselves with `/ch04-backprop`. Manual gradients take much work; the next chapter lets the computer calculate the gradients automatically.
