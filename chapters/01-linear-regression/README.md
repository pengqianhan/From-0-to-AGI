# Chapter 1: y = ax + b — Learn "training" from a straight line

**English** · [中文](README.zh.md)

> **Goal**: After this chapter, you can explain the four steps of training a model in your own words: model, loss, gradient, and update. You can also fit a straight line with gradient descent.

📺 **Video**: Not published yet. To render it on your computer, run `bash chapters/01-linear-regression/video/build.sh`.
🧪 **Self-check**: After the chapter, type `/ch01-linear` in Claude Code.

---

This is the first chapter of the course. At the end of the course, we discuss large language models with hundreds of billions of parameters. Here, we start with one straight line: `y = ax + b`. The structure of training a large model is the same as the structure of this chapter. The only difference is the number of parameters: here we have 2, and a large model has billions. Learn this chapter well. Each chapter after it asks the same question: "What new problems occur when we make this larger?"

## 1. Intuition: what is "training"?

You have a set of data points. Each point is a pair `(x, y)`. For example, `x` is the area of a house and `y` is its price. On a plot, the points are almost on a straight line, but not exactly. Real data always has noise.

You want a **model**. You give the model a new `x`, and the model predicts `y`. The simplest model is a straight line:

```
ŷ = a·x + b
```

Here, `ŷ` ("y hat") is the **prediction** of the model. `a` is the slope and `b` is the intercept. `a` and `b` are the **parameters** of the model. All the "knowledge" of this model is in these two numbers.

The question is: which values of `a` and `b` are correct?

**Training** lets the data set the parameters. Training has four steps. These four steps occur in every chapter of the course:

1. **Model**: Write the formula for the prediction, `ŷ = a·x + b`.
2. **Loss**: Use one number to measure how bad the predictions are.
3. **Gradient**: Calculate the direction in which the parameters must move to make the loss smaller.
4. **Update**: Move the parameters a small step in that direction. Then go back to step 1 and do it again.

## 2. Loss: change "how bad" into one number

For data point `i`, the prediction of the model is `ŷ_i = a·x_i + b` and the true value is `y_i`. The difference `ŷ_i − y_i` is the **residual**. Square all residuals and calculate their mean. The result is the **mean squared error (MSE)**:

```
L(a, b) = 1/N · Σ (a·x_i + b − y_i)²
```

We square the residuals for two reasons. First, positive and negative errors do not cancel. Second, the square function is smooth everywhere, so it is easy to differentiate. We use this property in the next section.

We write the loss as `L(a, b)`. **The data does not change, so the loss is a function of the parameters.** A different pair `(a, b)` gives a different loss. Plot the loss for all pairs `(a, b)`. The plot has the shape of a bowl. We want to find the bottom of the bowl.

## 3. Gradient: the direction in which the loss decreases fastest

Think of a person on the side of a valley in fog. The person cannot see the bottom. The person can only feel the slope under their feet and take a step in the steepest downhill direction.

The **gradient** is this slope. It is a vector of the partial derivatives of the loss for each parameter. The gradient points in the direction in which the loss **increases** fastest. For the MSE, the chain rule gives the gradient directly:

```
∂L/∂a = 2/N · Σ (ŷ_i − y_i) · x_i
∂L/∂b = 2/N · Σ (ŷ_i − y_i)
```

These two formulas have a direct meaning. If the model predicts values that are too high (most residuals are positive), `∂L/∂b` is positive, so `b` must decrease. The product of the residual and `x` tells us if the slope `a` must increase or decrease. The gradient is not an abstract symbol. It is a summary of where the model is wrong and by how much.

## 4. Update: gradient descent

The gradient points uphill, so we move in the opposite direction:

```
a ← a − η · ∂L/∂a
b ← b − η · ∂L/∂b
```

`η` ("eta") is the **learning rate**. It controls the size of each step. This process is **gradient descent**: calculate the gradient, take a small step in the opposite direction, and do it again.

## 5. Minimal code: fit a straight line in 60 lines

Run:

```bash
uv run python chapters/01-linear-regression/code/01_fit_line.py
```

The core is these lines (the full code is in [`code/01_fit_line.py`](code/01_fit_line.py)):

```python
def gradients(a, b, x, y):
    err = (a * x + b) - y              # residual ŷ_i − y_i
    grad_a = 2 * np.mean(err * x)      # ∂L/∂a
    grad_b = 2 * np.mean(err)          # ∂L/∂b
    return grad_a, grad_b

for _ in range(steps):
    grad_a, grad_b = gradients(a, b, x, y)
    a = a - lr * grad_a                # a ← a − η · ∂L/∂a
    b = b - lr * grad_b                # b ← b − η · ∂L/∂b
```

The true relation in our data is `y = 2x + 1`, plus noise. We start from a bad point, `a = −1, b = 4`, with a learning rate of 0.05. The output is:

| Step | a | b | Loss |
|---:|---:|---:|---:|
| 0 | −1.000 | 4.000 | 44.23 |
| 1 | 0.944 | 4.492 | 3.34 |
| 10 | 1.166 | 3.855 | 2.33 |
| 50 | 1.682 | 2.123 | 0.61 |
| 200 | 2.038 | 0.926 | 0.25 |

Look at these results:

- **After the first step, the loss decreases from 44 to 3.3.** At the start, the parameters are far from the bottom. The slope is steep and the gradient is large, so the first step is long.
- **After that, the steps become smaller.** Near the bottom, the slope is less steep. The gradient becomes smaller, and so do the steps.
- **The training stops at a ≈ 2.04 and b ≈ 0.93, not at exactly 2 and 1.** The reason is the noise in the data. The closed-form least-squares solution (`np.polyfit`) gives a = 2.052 and b = 0.879. Gradient descent moves toward the same point.

> **Note:** Linear regression has a closed-form solution. Why do we use gradient descent? Only linear models have a closed-form solution. From the neural networks in Chapter 3 onward, no formula gives the answer in one step. Gradient descent is the only general method. In this chapter, we use it on a problem with a known answer. Then we can see how it works.

## 6. Learning rate: too small, correct, and too large

The learning rate is the most important setting in training. Run:

```bash
uv run python chapters/01-linear-regression/code/02_learning_rate.py
```

The output (100 steps for each learning rate) is:

| Learning rate η | Loss after 100 steps | Result |
|---:|---:|---|
| 0.005 | 2.34 | Too small: correct direction, but too slow |
| 0.05 | 0.29 | Normal convergence |
| 0.0914 | 0.25 | Near the critical value, fastest convergence |
| 0.1066 | 7.7 × 10⁹ | Too large: the parameters oscillate with larger and larger steps and diverge |

Where does the critical value come from? In this problem, the loss is a quadratic function of `a` and `b`. The data alone sets the shape of the bowl. For a quadratic function, gradient descent converges only when `η < 2/λ_max`. Here, `λ_max` is the largest eigenvalue of the Hessian (the matrix of second derivatives of the loss). It tells us how much the bowl curves in its steepest direction. The code calculates this critical value: 0.1016. The experiment agrees with the theory. A learning rate a little below this value converges, and a learning rate a little above it diverges.

We will see this effect again in later chapters. When the loss of a large model suddenly increases (a loss spike), the cause is often a step that is too large in one direction. Chapter 6 shows warmup and learning-rate decay. Both methods control this setting.

## 7. Summary

- **Model**: `ŷ = a·x + b`. The parameters are `a` and `b`.
- **Loss**: The mean squared error `L(a, b)`. When the data does not change, the loss is a function of the parameters.
- **Gradient**: `∂L/∂a` and `∂L/∂b`. The gradient points in the direction in which the loss increases fastest.
- **Update**: `parameter ← parameter − η · gradient`. Do the update again until the loss stops decreasing.
- **Learning rate**: If it is too small, training is slow. If it is too large, training diverges.

---

## From minimal code to production code

In the minimal code, we derived the gradient formulas by hand. Real deep-learning code does not do this. For a complex model, derivation by hand takes too much work and causes errors. The standard PyTorch code for the same task is in [`code/03_pytorch_version.py`](code/03_pytorch_version.py):

```python
model = nn.Linear(in_features=1, out_features=1)   # the weight is a, the bias is b
optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
loss_fn = nn.MSELoss()

for step in range(201):
    y_hat = model(x)              # 1. forward pass: calculate the predictions
    loss = loss_fn(y_hat, y)      # 2. loss
    optimizer.zero_grad()         # 3. set the gradients from the last step to zero
    loss.backward()               # 4. backward pass: autograd calculates the gradients
    optimizer.step()              # 5. update the parameters
```

The table shows what the production code adds and why:

| Minimal code | Production code | Why |
|---|---|---|
| Derive `∂L/∂a` and `∂L/∂b` by hand | `loss.backward()` calculates the gradients | You cannot derive gradients by hand for billions of parameters. In Chapter 4, you write autograd yourself and see how it works. |
| `a = a - lr * grad_a` | `optimizer.step()` | The update rule becomes more complex (AdamW in Chapter 6). The optimizer controls it in one place. |
| Two floating-point numbers `a` and `b` | The weight tensor in `nn.Linear` | Chapter 2 changes one input into many inputs. Then the weights become a matrix. |
| None | `optimizer.zero_grad()` | By default, PyTorch **adds** each new gradient to the old one. This is useful for gradient accumulation in Chapter 14. Thus, set the gradients to zero at the start of each step. |

At the end, the script compares the PyTorch result with the result of the manual gradients. After 201 steps, both give `a = 2.039, b = 0.925`, **with the same digits**. Each chapter after this one does the same check: the production code must give the same result as the minimal code (a parity check). Only then can we trust the production code.

Remember this five-line training loop. In Chapter 14, we pretrain the main-line model. The loop then has more engineering parts: distributed training, mixed precision, and resume from checkpoints. But the structure of the loop is still these five steps.

---

## Guided questions

Ask Claude Code these questions. Continue until you can explain the answers in your own words:

1. Change the loss from the squared error to the absolute error `|ŷ − y|`. What is the new gradient formula? What problem occurs at the point `ŷ = y`?
2. Why does the gradient point in the direction in which the loss **increases** fastest? Ask Claude Code to explain it with a 2D contour plot.
3. Multiply all `x` values in the data by 10 (for example, change the unit of area from 10 m² to 1 m²). How does the critical learning rate change? Why? How is this related to normalization in Chapter 6?
4. Each step of gradient descent uses **all** 50 data points to calculate the gradient. What problem occurs with 1 billion data points? (Hint: search for "stochastic gradient descent" and "mini-batch".)
5. The loss stops at 0.25. Why can the loss not decrease to 0? How is 0.25 related to the size of the noise in the data?

## Hands-on tasks

For each task, run the code and look at the result.

**Task 1 (basic)**: In `01_fit_line.py`, change the start point to `a = 10, b = −10`. Record the number of steps until the loss is less than 0.3. Compare it with the original start point.

**Task 2 (core)**: In `01_fit_line.py`, add a function that calculates a numerical gradient. Use `(L(a + ε, b) − L(a − ε, b)) / (2ε)` to estimate `∂L/∂a`, with ε = 1e-5. Compare it with the gradient from the formula and print the difference. This method is a **gradient check**. You use it again when you write autograd in Chapter 4.

**Task 3 (challenge)**: Change the model to a quadratic curve, `ŷ = a·x² + b·x + c`. Train it on the same data (the true relation is a straight line). Is the learned `a` near 0? Then change the data to a true quadratic relation, `y = 0.5x² − x + 2`, and train again. Think about this: if the data follows a more complex curve, must we guess the formula each time? Chapter 3 answers this question with neural networks.

---

## References

- Goodfellow, Bengio, Courville. *Deep Learning*, Section 4.3, "Gradient-Based Optimization": <https://www.deeplearningbook.org/contents/numerical.html>
- Andrew Ng. *CS229 Lecture Notes*, Chapter 1, "Linear Regression and Gradient Descent": <https://cs229.stanford.edu/main_notes.pdf>
- 3Blue1Brown. *Gradient descent, how neural networks learn*: <https://www.3blue1brown.com/lessons/gradient-descent>
- PyTorch tutorial *Learning the Basics* (the standard training loop): <https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html>

**Next chapter**: A real house price depends on more than the area. It also depends on the number of bedrooms, the floor, the distance to a train station, and more. The input changes from one number to a set of numbers, and the parameters change from two numbers to a matrix. In Chapter 2, we go from scalars to matrices.
