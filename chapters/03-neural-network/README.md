# Chapter 3: Nonlinearity and neural networks — Build a curve from line segments

**English** · [中文](README.zh.md)

> **Goal**: After this chapter, you can explain why a stack of linear layers is still linear, for any number of layers. You can draw the hinge line of one ReLU hidden unit. You can also train a two-layer MLP with gradient descent and fit the curve `y = sin(2x)`.

📺 **Video**: Not published yet. To render it on your computer, run `bash chapters/03-neural-network/video/build.sh`.
🧪 **Self-check**: After the chapter, type `/ch03-neural-network` in Claude Code.

---

In the last chapter, we changed the input from one number to a set of numbers. We wrote the model as `Ŷ = XW + b`, and we learned the shape rules of matrix multiplication. But for any size of W, this model is still **linear**. When the change in the input doubles, the change in the output also doubles. On a plot, the model is always a straight line or a flat plane. This chapter answers one question: **the data is curved and a straight line cannot fit it, so what do we do?**

The answer is the neural network. Its key part is one small change: put a nonlinear "activation function" between two linear layers. In this chapter, we see why this small change is sufficient. We also see what the network looks like on a plot: **a curve made from many straight-line segments**.

## 1. Intuition: a straight line cannot fit a curve

Task 3 of Chapter 1 asked a question: if the data follows a complex curve, must we guess the formula each time? This chapter answers the question with one specific curve.

The data of this chapter is 100 evenly spaced points for x in [−3, 3], with `y = sin(2x)`. This curve is a wave that goes up and down and has four bends. We did not add noise, because we want to look at only one question: can the model bend?

First, use the method from Chapter 1. Among all straight lines, find the line with the smallest mean squared error (the closed-form least-squares solution). Run:

```bash
uv run python chapters/03-neural-network/code/01_linear_is_not_enough.py
```

```
1) Fit y = sin(2x) with a straight line
   Best straight line: y = -0.165·x + 0.000, MSE = 0.4341
   Reference: variance of y = 0.5178 (the MSE if we always predict the mean)
```

The best straight line is almost flat, and its MSE is 0.4341. A model that learns nothing and always predicts the mean gets 0.5178, which is not much worse. **The cause is not too little training, and it is not a bad learning rate.** This line is already the best of all straight lines. The problem is the model itself: a straight line cannot bend.

## 2. Stack more linear layers? This does not help

A natural idea is: if one layer is not sufficient, stack two layers. In the notation of Chapter 2 (each row is one sample, `h ← h·W + b`), two linear layers are:

```
Ŷ = (X·W1 + b1)·W2 + b2
```

Multiply out the parentheses:

```
Ŷ = X·(W1·W2) + (b1·W2 + b2)
  = X·W + b        where W = W1·W2, b = b1·W2 + b2
```

`W1·W2` is still a matrix, and `b1·W2 + b2` is still a vector. **Two linear layers are equal to one linear layer.** The same is true for any number of layers. Merge them one at a time, and the result is still `X·W + b`. A composition of linear functions is still a linear function.

The function `collapse` in the code does this merge (the full code is in [`code/01_linear_is_not_enough.py`](code/01_linear_is_not_enough.py)):

```python
def collapse(layers):
    W, b = layers[0]
    for W_next, b_next in layers[1:]:
        W, b = W @ W_next, b @ W_next + b_next   # (X·W + b)·W' + b' = X·(W·W') + (b·W' + b')
    return W, b
```

Make two random networks that have only linear layers. Calculate the output layer by layer, with no shortcut. Then calculate it again with the merged single layer, and compare:

```
2) Stacked linear layers are still one linear layer
   2 layers 1→8→1: 25 parameters, merged y = 0.375·x +2.045, max difference layer-by-layer vs merged = 8.9e-16
   3 layers 1→8→8→1: 97 parameters, merged y = -0.145·x -3.210, max difference layer-by-layer vs merged = 4.4e-15
```

The difference is of the order of 10⁻¹⁵. This difference is only floating-point rounding error: mathematically, the two results are equal. With 25 parameters or with 97 parameters, the plot is only a straight line.

Here is more direct evidence. Use gradient descent to train a network with two linear layers and a width of 8 in the middle. This is the code from Section 4, without the activation function. After 1000 steps, the loss decreases to 0.4341. Then the loss **does not move at all**. It is exactly the loss of the best straight line (see the first row of the table in Section 7). More parameters cannot make the model more than a straight line.

## 3. Activation function: add a bend between the two layers

The problem is that every part is linear. The solution needs only one small change. Between the two layers, apply a **nonlinear** function to each number. This function is the **activation function**. The most common activation function is **ReLU (Rectified Linear Unit)**:

```
ReLU(z) = max(0, z)
```

ReLU changes a number less than 0 to 0, and it keeps a number greater than 0 as it is. Its graph is a line with one **kink** (a sharp bend) at 0. Two points are important:

- ReLU operates **element by element**. It calculates each number in the matrix separately, and it does not mix different numbers. Thus it does not change the shape: `(N, H)` goes in, and `(N, H)` comes out.
- This one kink stops the derivation of Section 2 (multiply out the parentheses and merge into one layer). You cannot write `ReLU(X·W1 + b1)·W2` as `X·(some matrix)`, because a matrix product cannot make a kink.

There is more than one activation function. Run:

```bash
uv run python chapters/03-neural-network/code/02_activations.py
```

```
1) Values of the activation functions at some points
   z            -3.0    -1.0     0.0     1.0     3.0
   ReLU        0.000   0.000   0.000   1.000   3.000
   sigmoid     0.047   0.269   0.500   0.731   0.953
   tanh       -0.995  -0.762   0.000   0.762   0.995
   SiLU       -0.142  -0.269   0.000   0.731   2.858
   GELU       -0.004  -0.159   0.000   0.841   2.996
```

| Activation function | Formula | Notes |
|---|---|---|
| sigmoid | `σ(z) = 1 / (1 + e^(−z))` | Common in early neural networks. S-shaped; it squeezes the input into (0, 1). Both ends are too flat: when z moves a little away from 0, the gradient is almost 0. Deep networks with sigmoid are difficult to train |
| tanh | `tanh(z)` | Also S-shaped. It squeezes the input into (−1, 1) and is centered at 0. But both ends are also too flat |
| ReLU | `max(0, z)` | Became popular around 2010 (Nair & Hinton 2010; Glorot et al. 2011). A simple function. On the positive side, the gradient is always 1 |
| SiLU (also called Swish) | `z · σ(z)` | A smooth version of ReLU. For a large positive z, it is about z. For a large negative z, it is about 0. Near 0, it has a smooth bend |
| GELU | `z · Φ(z)`, where Φ is the cumulative distribution function of the standard normal distribution | Another smooth version. Its shape is almost the same as SiLU |

Large models today use the last two. The feed-forward layers of Qwen, DeepSeek, and Llama use **SwiGLU**, and Gemma uses **GeGLU**. These are "gated" structures: one path goes through SiLU (or GELU), and the result is multiplied element by element with a second path. Chapter 9 gives the details of gating. In this chapter, know only this: **SiLU and GELU are smooth versions of ReLU, and they have the same function: they add nonlinearity.** The rest of this chapter uses ReLU, because its kink is the easiest to see.

## 4. Two-layer MLP: linear → ReLU → linear

Connect a linear layer, an activation function, and a linear layer in sequence. The result is the simplest neural network: a two-layer **multi-layer perceptron (MLP)**.

```
Z = X·W1 + b1        (N, 1) → (N, H)     layer 1: linear
A = ReLU(Z)          (N, H)              activation: element by element
Ŷ = A·W2 + b2        (N, H) → (N, 1)     layer 2: linear
L = mean((Ŷ − Y)²)                       loss: the same mean squared error as in Chapter 1
```

You can check the shapes with the rules from Chapter 2. `X` is `(N, 1)`, `W1` is `(1, H)`, and `b1` is `(H,)`. `W2` is `(H, 1)`, and `b2` is `(1,)`. The H numbers in the middle are the **hidden units**. People also often call them neurons. H is the **width** of the network. The total number of parameters is `H + H + H + 1 = 3H + 1`.

These are the three matching lines in the code (the full code is in [`code/03_mlp_numpy.py`](code/03_mlp_numpy.py)):

```python
def forward(p, x, act="relu"):
    z = x @ p["W1"] + p["b1"]         # Z = X·W1 + b1
    a = act_fn(z, act)                # A = ReLU(Z)
    y_hat = a @ p["W2"] + p["b2"]     # Ŷ = A·W2 + b2
    return y_hat, (z, a)
```

With `act="linear"`, `act_fn` returns its input without change. This setting gives the "two linear layers" network from Section 2. The code is the same; the only difference is the activation function.

## 5. One ReLU hidden unit = one kink

Why can an MLP bend? Look only at the contribution of hidden unit j to the output. When the input is one number x, the unit first calculates `w·x + b` (w and b are entry j of W1 and b1). Then the unit applies ReLU. Then it multiplies the result by the second-layer weight v (entry j of W2):

```
v · ReLU(w·x + b)
```

The graph of this contribution is a **hinge line**: a line with one kink. One side is flat (the side where ReLU gives 0), and the other side is sloped. The kink is where `w·x + b = 0`, that is, at **`x = −b/w`**.

Each parameter changes the hinge line in a different way:

- Change b: the kink moves to the left or to the right.
- Change w: the sloped side becomes steeper or less steep. The sign of w sets if the sloped side is on the right or on the left.
- Change v: the whole line scales. When v is negative, the whole hinge line turns upside down.

The output of the network is the sum of H such hinge lines, plus b2:

```
ŷ(x) = Σⱼ vⱼ · ReLU(wⱼ·x + bⱼ) + b2
```

A sum of hinge lines is still a line made of straight segments (a piecewise-linear function), but with more kinks. Part 2 of `02_activations.py` builds shapes from some ReLUs by hand:

```
2) Build shapes from ReLUs (x from −2 to 2)
   x
        -2.0  -1.5  -1.0  -0.5   0.0   0.5   1.0   1.5   2.0
   |x| = ReLU(x)+ReLU(−x)
         2.0   1.5   1.0   0.5   0.0   0.5   1.0   1.5   2.0
   tent = ReLU(x+1)−2ReLU(x)+ReLU(x−1)
         0.0   0.0   0.0   0.5   1.0   0.5   0.0   0.0   0.0
```

Two ReLUs make the absolute value |x| (a V shape). Three ReLUs (kinks at −1, 0, and 1; the middle one is multiplied by −2) make a "tent". The tent is a bump only in [−1, 1], and it is 0 at all other points. With tents, you can put a bump of any height at any position. With sufficient bumps that are sufficiently narrow, the sum can follow any continuous curve closely.

This is the idea of the **universal approximation theorem**. With sufficient hidden units, a network with one hidden layer can approximate a continuous function to any accuracy (Cybenko 1989; Hornik 1991; Leshno et al. 1993 proved that it is also true for nonpolynomial activations such as ReLU).

Note that the theorem only guarantees that such a set of parameters exists. It does not say that gradient descent can find it. It also does not say how many hidden units are necessary. The next sections test this with experiments.

## 6. Training: manual gradients + gradient descent

The four steps of training are the same as in Chapter 1: model, loss, gradient, and update. The only new problem is the gradient. The parameters are now in two layers. To calculate the gradient of the loss for W1, we must go "through" the second layer and the ReLU.

The method is to start at the loss and go back, one layer at a time, with the chain rule:

```python
def gradients(p, x, y, act="relu"):
    n = len(x)
    y_hat, (z, a) = forward(p, x, act)
    loss = float(np.mean((y_hat - y) ** 2))
    d_yhat = 2 * (y_hat - y) / n          # ∂L/∂Ŷ: the same as 2/N·(ŷ − y) in Chapter 1
    d_W2 = a.T @ d_yhat                   # ∂L/∂W2 = Aᵀ · ∂L/∂Ŷ
    d_b2 = d_yhat.sum(axis=0)             # ∂L/∂b2 = Σ ∂L/∂Ŷ
    d_a = d_yhat @ p["W2"].T              # ∂L/∂A  = ∂L/∂Ŷ · W2ᵀ
    d_z = d_a * act_grad(z, act)          # ∂L/∂Z  = ∂L/∂A ⊙ ReLU′(Z)
    d_W1 = x.T @ d_z                      # ∂L/∂W1 = Xᵀ · ∂L/∂Z
    d_b1 = d_z.sum(axis=0)                # ∂L/∂b1 = Σ ∂L/∂Z
    return loss, {"W1": d_W1, "b1": d_b1, "W2": d_W2, "b2": d_b2}
```

Read these lines:

- The second layer `Ŷ = A·W2 + b2` is a linear regression. The only difference is that the input is A, not X. Thus `∂L/∂W2 = Aᵀ·∂L/∂Ŷ`. This formula has the same form as "residual times input" in Chapter 1.
- To send the error back to the hidden layer, multiply by `W2ᵀ`. The "responsibility" of each hidden unit is proportional to the weight that connects it to the output.
- To go through the ReLU, multiply by `ReLU′(Z)`. The derivative is 1 when z > 0 and 0 when z < 0. **A unit that is not active gets a gradient of 0**, so its parameters do not move in this step. (At exactly z = 0, the derivative is not defined. By convention, we use 0. In practice, this case almost never occurs.)
- The first layer is again a "linear regression". Its input is X, and its error is `∂L/∂Z`.

This algorithm goes from the end back to the start and applies the chain rule layer by layer. Its name is **backpropagation**. Chapter 4 explains it in full and lets the computer do it automatically. Here, we first derive the gradients by hand. Then we use the **numerical gradient** from Task 2 of Chapter 1 to look for derivation errors:

```
1) Gradient check (width 8): manual vs numerical gradients, max difference = 1.1e-10
```

The difference is 10⁻¹⁰, so the manual formulas are correct. The update rule is exactly the same as in Chapter 1:

```python
for k in p:
    p[k] -= lr * g[k]             # θ ← θ − η · ∂L/∂θ, the same as in Chapter 1
```

A note about initialization: W1 comes from a standard normal distribution. We choose b1 so that the kink `−b1/W1` of each hidden unit starts at a random point in [−3, 3]. We scale W2 down by `1/√H`. Initialization has many details. Chapter 6 explains them in full.

## 7. Experiment: width 2, 8, and 64

Run:

```bash
uv run python chapters/03-neural-network/code/03_mlp_numpy.py
```

The settings are learning rate 0.01, full-batch gradient descent for 20000 steps, and random seed 0 (about 25 s):

| Model | Parameters | Step 0 | Step 1000 | Step 5000 | Step 20000 |
|---|---:|---:|---:|---:|---:|
| Two linear layers (no activation), width 8 | 25 | 2.0895 | 0.4341 | 0.4341 | **0.4341** |
| ReLU, width 2 | 7 | 0.5175 | 0.3245 | 0.3155 | **0.3155** |
| ReLU, width 8 | 25 | 0.6509 | 0.2731 | 0.0350 | **0.0152** |
| ReLU, width 64 | 193 | 2.7760 | 0.1169 | 0.0133 | **0.0009** |
| Reference: the best straight line | 2 | | | | 0.4341 |

Look at these results:

- **With the same 25 parameters, the ReLU makes a very large difference.** The two linear layers stop at 0.4341 (a straight line). With ReLU, the loss decreases to 0.0152, almost 30 times smaller. The only difference is the one `max(0, z)` in the middle.
- **Width 2 is only a little better than a straight line.** Two hidden units give at most two kinks and three straight segments. But sin(2x) has four bends in [−3, 3], so no combination of three segments can fit it.
- **Width 8 gets the approximate shape, and width 64 is almost perfect.** More kinks make the bends smoother. The loss decreases from 0.0152 to 0.0009.

One experiment can be luck. Run it again with 5 random seeds (0–4):

```
3) 5 random seeds (0–4), loss after 20000 steps
   width  2: 0.3155  0.4337  0.3866  0.3874  0.3866   median 0.3866
   width  8: 0.0152  0.0628  0.0064  0.0146  0.0333   median 0.0152
   width 64: 0.0009  0.0005  0.0003  0.0009  0.0002   median 0.0005
```

The pattern is stable: wider is better. But width 8 is sensitive to the seed (from 0.0064 to 0.0628, almost 10 times different). Two things affect the result: where the kinks start, and if a unit "dies". A dead unit has z < 0 on all data, so its gradient is always 0 and it never moves again. Width 64 has many extra kinks, so its result is much more stable. This result gives a basic intuition for why modern large models prefer to be wide rather than narrow.

Finally, take the width-8 network apart:

```
4) The width-8 network: each hidden unit is one kink
   Kink positions x = −b1/W1: -2.00  -0.79  0.45  0.86  2.20  2.21  2.53  2.99
   Max amplitude of one hinge piece in the data range = 12.17 (the network output has an amplitude of only 1.20: the pieces cancel each other)
   Max difference between (sum of pieces + b2) and the network output = 2.2e-15
```

- The sum of the 8 hinge lines is the network output. The only error is floating-point rounding. The formula from Section 5, `ŷ = Σ vⱼ·ReLU(wⱼx + bⱼ) + b2`, is not an analogy. It is an identity.
- Gradient descent found the kink positions by itself. We did not tell it anything about the sine function. Some kinks are close together on the right, in the range 2.2–3.0. Thus the 8 units are not used "evenly". This is one reason why width 8 is not stable.
- One hinge line can reach an amplitude of 12, but the output reaches only 1.2. The hinge lines cancel each other. The network does not learn a neat division of work, such as "each unit controls one small segment". It learns a set of numbers whose sum is correct.

## 8. Summary

- **Stacked linear layers are still linear**: `(X·W1 + b1)·W2 + b2 = X·(W1·W2) + (b1·W2 + b2)`. More parameters still give only a straight line.
- **The activation function** adds nonlinearity and operates element by element. ReLU = `max(0, z)`. Sigmoid and tanh were the early choices. Modern large models use the smooth versions of ReLU, SiLU and GELU (inside SwiGLU / GeGLU, see Chapter 9).
- **One ReLU hidden unit = one kink**, at `x = −b/w`. The output of a two-layer MLP = the sum of H hinge lines + b2.
- **The width** sets how many kinks the network can make. On sin(2x): width 2 → 0.3155, width 8 → 0.0152, width 64 → 0.0009.
- **The training method did not change**: it still has the four steps model, loss, gradient, and update. The chain rule gives the gradient, from the end back to the start (backpropagation; Chapter 4 explains it in detail).

---

## From minimal code to production code

In the minimal code, we wrote the forward pass and the gradients by hand as matrix operations. In Chapters 1–6, the production code is the standard PyTorch code. It is in [`code/04_pytorch_version.py`](code/04_pytorch_version.py):

```python
model = nn.Sequential(
    nn.Linear(1, 64),    # Z = X·W1ᵀ + b1 (the PyTorch weight has the shape (out, in))
    nn.ReLU(),           # A = ReLU(Z)
    nn.Linear(64, 1),    # Ŷ = A·W2ᵀ + b2
)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

for step in range(20001):
    y_hat = model(x)              # 1. forward pass
    loss = loss_fn(y_hat, y)      # 2. loss
    optimizer.zero_grad()         # 3. set the gradients to zero
    loss.backward()               # 4. backward pass: autograd calculates the gradients of all parameters
    optimizer.step()              # 5. update
```

The five lines of the training loop are **exactly the same** as in Chapter 1. Only `model` changed. Run:

```bash
uv run python chapters/03-neural-network/code/04_pytorch_version.py
```

```
1) Parity check: NumPy initial parameters copied into PyTorch (float64), width 64, SGD learning rate 0.01
   step     0: loss = 2.7760
   step  1000: loss = 0.1169
   step  5000: loss = 0.0133
   step 20000: loss = 0.0009
   PyTorch 0.0009144016  vs  NumPy with manual gradients 0.0009144016, difference = 6.2e-18

2) Default PyTorch initialization + float32, width 64, same training for 20000 steps
   step     0: loss = 0.4166
   step  1000: loss = 0.0927
   step  5000: loss = 0.0320
   step 20000: loss = 0.0019
   193 parameters (W1: 64, b1: 64, W2: 64, b2: 1)
```

- **Parity check**: copy the initial parameters of the NumPy code into PyTorch without change (note the transpose). Both use float64. After 20000 steps, the losses differ by 6.2 × 10⁻¹⁸. Thus the manual gradients and the autograd gradients agree at each step.
- **The default initialization gives a similar result**: with the PyTorch default initialization and float32, the loss after 20000 steps is 0.0019. This loss is of the same order as the 0.0009 from the manual initialization.

The table shows what the production code changes and why:

| Minimal code | Production code | Why |
|---|---|---|
| `x @ W1 + b1` | `nn.Linear(1, 64)` | The module creates, initializes, and registers the parameters. `model.parameters()` gets all parameters in one call and gives them to the optimizer |
| `np.maximum(0, z)` | `nn.ReLU()` | The activation function is also a module. To change to `nn.SiLU()` or `nn.GELU()`, change only one line |
| Three lines that we wrote and connected by hand | `nn.Sequential(...)` | It connects submodules in sequence. When there are more layers, write a custom `nn.Module` (the Transformer in Chapter 9 does this) |
| 6 lines of gradient formulas that we derived by hand | `loss.backward()` | Two layers already need 6 lines. For tens of layers and hundreds of types of operations, manual derivation is not practical. In Chapter 4, you write autograd yourself |
| Our own initialization rule | The default initialization of `nn.Linear` (a type of Kaiming uniform distribution) | The default is sufficient for most networks. Chapter 6 explains why the scale must depend on the input dimension |
| `p[k] -= lr * g[k]` | `optimizer.step()` | In Chapter 6, we change to AdamW, and no line of the training loop changes |

## Adopters and sources

In this chapter, only one mainstream technique needs a check of "who uses it": the activation function in the feed-forward layers of modern large models. We show ReLU, sigmoid, and tanh only as history (consensus rule C: necessary background).

| Technique | Adopters (leading open model families) | Source |
|---|---|---|
| SwiGLU (gating + SiLU) | Llama | LLaMA paper, Section 2.2, "SwiGLU activation function": <https://arxiv.org/abs/2302.13971> |
| | Qwen | Qwen3 technical report, section on the model architecture (GQA, SwiGLU, RoPE, RMSNorm): <https://arxiv.org/abs/2505.09388> |
| | DeepSeek | DeepSeek LLM technical report, section on the architecture (RMSNorm, SwiGLU): <https://arxiv.org/abs/2401.02954> |
| GeGLU (gating + GELU) | Gemma | Gemma technical report, section on the model architecture (GeGLU activations): <https://arxiv.org/abs/2403.08295> |

SwiGLU itself comes from Shazeer 2020, *GLU Variants Improve Transformer* (<https://arxiv.org/abs/2002.05202>). Chapter 9 checks each item and gives the details.

## Guided questions

Ask Claude Code these questions. Continue until you can explain the answers in your own words:

1. Replace ReLU with `f(z) = 2z + 1`. Can the network still fit sin(2x)? What about `f(z) = z²`? Which functions can be activation functions? (Hint: search for the conditions that the "universal approximation theorem" puts on the activation function.)
2. A hidden unit has `z < 0` on all 100 data points. What is its gradient? What happens in training? This problem is the "dying ReLU". Why do SiLU and GELU make it less severe?
3. The output of a two-layer MLP is piecewise linear. What does this fact tell you about the prediction at x = 10 (outside the range of the training data)? Does the prediction continue as a sine wave?
4. The width-64 network has 193 parameters, but the data has only 100 points. There are more parameters than data points. Why did no problem occur? What happens if the data has noise?
5. The input changes from 1 dimension to d dimensions. What geometric shape does the "kink" of one ReLU hidden unit become? (Hint: in 2D, `w·x + b = 0` is a straight line.)

## Hands-on tasks

For each task, run the code and look at the result.

**Task 1 (basic)**: In `03_mlp_numpy.py`, change the learning rate from 0.01 to 0.03. Train the width-8 and width-64 networks again. Compare with the original results: how does the loss change? If you see `nan`, what happened? (Remember the critical learning rate from Chapter 1.)

**Task 2 (core)**: Change `act_fn` and `act_grad` to tanh (the derivative is `1 − tanh²(z)`). First, do a gradient check with `numerical_gradients`. Then train the width-8 and width-64 networks, and compare with the ReLU results. Is the output of the tanh network still piecewise linear?

**Task 3 (challenge)**: Do not use gradient descent. Design **by hand** a ReLU network with a width of 12 or less that approximates sin(2x). Take 12 evenly spaced points in [−3, 3]. Make the piecewise-linear output of the network go exactly through the values of sin(2x) at these points. (Hint: go from left to right. At each kink, how much must the slope change? That change is v·|w| of this unit.) Calculate the MSE of your network. Compare it with the width-8 and width-64 networks from gradient descent.

---

## References

- Goodfellow, Bengio, Courville. *Deep Learning*, Chapter 6, "Deep Feedforward Networks" (Section 6.1 uses XOR to show why linear models are not sufficient; Section 6.3 is about hidden units and ReLU): <https://www.deeplearningbook.org/contents/mlp.html>
- Michael Nielsen. *Neural Networks and Deep Learning*, Chapter 4, "A visual proof that neural nets can compute any function": <http://neuralnetworksanddeeplearning.com/chap4.html>
- 3Blue1Brown. *But what is a neural network?*: <https://www.3blue1brown.com/lessons/neural-networks>
- Cybenko (1989). *Approximation by superpositions of a sigmoidal function*: <https://doi.org/10.1007/BF02551274>
- Leshno, Lin, Pinkus, Schocken (1993). *Multilayer feedforward networks with a nonpolynomial activation function can approximate any function*: <https://doi.org/10.1016/S0893-6080(05)80131-5>
- Nair & Hinton (2010). *Rectified Linear Units Improve Restricted Boltzmann Machines*: <https://www.cs.toronto.edu/~hinton/absps/reluICML.pdf>
- Glorot, Bordes, Bengio (2011). *Deep Sparse Rectifier Neural Networks*: <https://proceedings.mlr.press/v15/glorot11a.html>
- Hendrycks & Gimpel (2016). *Gaussian Error Linear Units (GELUs)*: <https://arxiv.org/abs/1606.08415>
- Elfwing, Uchibe, Doya (2017). *Sigmoid-Weighted Linear Units for Neural Network Function Approximation in Reinforcement Learning* (SiLU): <https://arxiv.org/abs/1702.03118>
- Shazeer (2020). *GLU Variants Improve Transformer* (SwiGLU, GeGLU): <https://arxiv.org/abs/2002.05202>
- PyTorch tutorial *Build the Neural Network* (`nn.Sequential`, `nn.Linear`, `nn.ReLU`): <https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html>
- [Mathematical theory of deep learning](https://arxiv.org/abs/2407.18384) (already in `references.md`; its chapters on universal approximation are good for readers who want the rigorous proofs)

**Next chapter**: For the two-layer network in this chapter, we already had to derive 6 lines of gradients by hand. We also had to check them with numerical gradients. What about a network with tens of layers and hundreds of types of operations? Manual gradients take much work and cause errors. In Chapter 4, we start from the chain rule and write a small automatic differentiation (autograd) tool of a little more than 100 lines. We use it to train the MLP from this chapter again. Then you know what occurs inside `loss.backward()`.
