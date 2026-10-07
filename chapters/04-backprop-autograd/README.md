# Chapter 4: Backpropagation and automatic differentiation — Let the computer calculate the derivatives

**English** · [中文](README.zh.md)

> **Goal**: After this chapter, you can write a scalar automatic differentiation engine of about 150 lines by hand. You can use it to train a multilayer perceptron. You can also use numerical gradients and PyTorch to prove that each gradient from the engine is correct.

📺 **Video**: Not published yet. To render it on your computer, run `bash chapters/04-backprop-autograd/video/build.sh`.
🧪 **Self-check**: After the chapter, type `/ch04-backprop` in Claude Code.

---

In the last chapter, we used a two-layer MLP to fit `y = sin(2x)`. To train it, we derived 6 gradient formulas by hand. We went back from the loss, through the second layer, through ReLU, to the first layer. Then we had to check the formulas with a numerical gradient before we could trust them. This chapter solves this problem: **each change to the network needs a new derivation of the gradients. Can the computer calculate the gradients itself?**

The answer is **automatic differentiation (autograd)**. In the last chapter, we went from back to front and applied the chain rule layer by layer. This algorithm is **backpropagation**. It is the form of automatic differentiation that neural networks use. In Chapter 1, PyTorch had one line, `loss.backward()`. In this chapter, you write the mechanism behind that line yourself.

## 1. The problem: derivation by hand does not scale

Look again at the gradients that we derived:

- **Chapter 1**: `ŷ = a·x + b`, two parameters, two formulas: `∂L/∂a = 2/N · Σ(ŷ_i − y_i)·x_i`, `∂L/∂b = 2/N · Σ(ŷ_i − y_i)`.
- **Chapter 3**: a two-layer network, `A = ReLU(X·W1 + b1)`, `Ŷ = A·W2 + b2`. We derived 6 formulas by hand. As one formula, the gradient of the first-layer weights is:

```
∂L/∂W1 = Xᵀ · [ (2/N · (Ŷ − Y) · W2ᵀ) ⊙ 1(X·W1 + b1 > 0) ]
```

If you add one more layer, the formula gets one more level of nesting. If you replace ReLU with tanh, you must derive the `1(· > 0)` term again. If you replace the mean squared error with the cross-entropy of the next chapter, you must derive the outermost term again. A worse problem is that **an error in the derivation is difficult to find**. If one transpose in the formula is wrong, the code still runs, but the training does not make progress.

At the end of the course, we train a large language model. It has billions of parameters, tens of layers, and hundreds of different operations. Derivation by hand is not possible.

The solution comes from one observation: **however complex a network is, it is a combination of a few simple operations.** These operations are addition, multiplication, power, tanh, exponential, logarithm, and a few more. We know the derivative of each simple operation. If the computer also finds the derivative of the combination automatically, the problem is solved. The mathematical tool for the combination is the chain rule.

## 2. The chain rule: multiply the local derivatives along the path

Think of a set of gears. `u = 3x`: when x turns by a small amount, u turns 3 times as much. `y = u²`: at x = 2 and u = 6, when u turns by a small amount, y turns `2u = 12` times as much. When x turns by a small amount, how much does y turn? 3 × 12 = **36 times** as much.

This is the **chain rule**:

```
dy/dx = dy/du · du/dx
```

Verify it with code (we explain the `Value` class of `01_engine.py` soon):

```python
x = Value(2.0); u = x * 3; y = u**2
y.backward()
x.grad   # 36.0
```

The key point of the chain rule: **each segment needs to know only its own small step.** The segment `u = 3x` only knows "I make the change 3 times larger". It does not need to know if a square or a different operation comes after it. This derivative of one step is the **local derivative**.

A variable can affect the output through **more than one path**. Then we **add** the contributions of all paths (the multivariable chain rule). In Section 5, this rule becomes an important `+=` in the code.

## 3. Computational graph: record each step in the forward pass

Break an expression into its smallest operations, and make each intermediate result a node. The result is a **computational graph**. Take the example of this chapter:

```
L = (a·b + c)², a = 2, b = −3, c = 10
```

It has three steps: `d = a·b`, `e = d + c`, `L = e²`.

```
a ──┐
    (×)──→ d ──┐
b ──┘          (+)──→ e ──→ (²)──→ L
c ─────────────┘
```

In the **forward pass**, the values flow from left to right: `d = −6`, `e = 4`, `L = 16`.

Our engine does one more thing in the forward pass: **each operation makes a new node, and the node records its input nodes and its operation.** After one pass, the engine has recorded the full graph. These are all the data of the `Value` class:

```python
class Value:
    def __init__(self, data, _children=(), _op=""):
        self.data = float(data)
        self.grad = 0.0                   # ∂L/∂(this node); 0 before the backward pass
        self._backward = lambda: None     # sends the gradient to the inputs (a leaf node has none)
        self._prev = _children            # graph edges: the nodes that this node comes from
        self._op = _op
```

## 4. Backpropagation: upstream gradient × local derivative

Now we calculate the gradient of L for each node. We write `v̄ = ∂L/∂v` (say "v bar") and go from right to left:

| Node | Local derivative | Gradient = upstream gradient × local derivative |
|---|---|---|
| L | — | `L̄ = 1` (the derivative of L for itself) |
| e (L = e²) | `∂L/∂e = 2e = 8` | `ē = 1 × 8 = 8` |
| d, c (e = d + c) | `∂e/∂d = 1`, `∂e/∂c = 1` | `d̄ = 8`, `c̄ = 8` |
| a, b (d = a·b) | `∂d/∂a = b = −3`, `∂d/∂b = a = 2` | `ā = 8 × (−3) = −24`, `b̄ = 8 × 2 = 16` |

Run `uv run python chapters/04-backprop-autograd/code/01_engine.py`. The output is the same:

```
Forward:  d = a·b = -6.0   e = d + c = 4.0   L = e² = 16.0
Backward: ∂L/∂e = 8.0  ∂L/∂d = 8.0  ∂L/∂a = -24.0  ∂L/∂b = 16.0  ∂L/∂c = 8.0
```

Each step does **the same action**. It takes the gradient that comes from upstream, multiplies it by the local derivative of the node, and sends the result to the inputs. The local derivative of an addition node is 1, so the node **sends the gradient without change** to both inputs. The local derivative of a multiplication node is **the value of the other input**, so `ā = d̄ · b`.

This is all the mathematics of backpropagation. The rest is engineering: how to write it as code.

## 5. Write each operation only once: `_backward`

The secret of automatic differentiation: **you write the local derivative of each operation only once.** Take multiplication as an example:

```python
def __mul__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data * other.data, (self, other), "*")   # forward: calculate the value, record the two inputs

    def _backward():                  # ∂(a·b)/∂a = b, ∂(a·b)/∂b = a
        self.grad += other.data * out.grad
        other.grad += self.data * out.grad
    out._backward = _backward
    return out
```

`out.grad` is the "upstream gradient", and `other.data` is the "local derivative". Our engine has the operations in the table below. Each operation is "one line for the forward pass + one line for the local derivative":

| Operation | Forward | Local derivative | The line in the code |
|---|---|---|---|
| Addition | `c = a + b` | `∂c/∂a = 1`, `∂c/∂b = 1` | `self.grad += out.grad` |
| Multiplication | `c = a·b` | `∂c/∂a = b`, `∂c/∂b = a` | `self.grad += other.data * out.grad` |
| Constant power | `c = aᵏ` | `k·aᵏ⁻¹` | `self.grad += k * self.data ** (k - 1) * out.grad` |
| ReLU | `c = max(0, a)` | 1 if a > 0, else 0 | `self.grad += (self.data > 0) * out.grad` |
| tanh | `c = tanh a` | `1 − c²` | `self.grad += (1 - t * t) * out.grad` |
| Exponential | `c = eᵃ` | `c` | `self.grad += out.data * out.grad` |
| Logarithm | `c = ln a` | `1/a` | `self.grad += (1 / self.data) * out.grad` |

Subtraction, division, and negation do not need their own derivatives. They are combinations of the operations above. For example, `a − b = a + (b × −1)` and `a / b = a × b⁻¹`:

```python
def __sub__(self, other):      return self + (-other)
def __truediv__(self, other):  return self * other**-1
```

## 6. Fan-out: why the gradient uses `+=`

Each line above uses `+=`, not `=`. Why?

Look at this example: `y = x·x + x`, with x = 3. The graph **uses x three times**: one time for each of the two inputs of the multiplication, and one time for the addition.

```
x ──┬──→ (×) ──→ m = x·x ──┐
    ├──→ (×)                (+)──→ y
    └───────────────────────┘
```

In the backward pass, each of the three paths brings back a part of the gradient. The two inputs of the multiplication each bring back `x × 1 = 3`. The path through the addition brings back `1`. The true gradient of x is the **sum** of these parts: `3 + 3 + 1 = 7`. This is equal to the derivative by hand, `dy/dx = 2x + 1 = 7`. The output of `01_engine.py` is:

```
Fan-out: y = x·x + x, x = 3 → ∂y/∂x = 7.0 (by hand: 2x + 1 = 7)
```

If you write `=`, the gradient that arrives later **replaces** the gradient that arrived first. Only the contribution of the last path stays. This problem is real. In real networks, fan-out occurs everywhere. Each neuron of the first layer uses the same input. Each neuron of the next layer uses the output of a hidden unit. In the gradient check of Section 8, we change `+=` to `=` on purpose and look at the errors that this change causes.

This also answers a question from Chapter 1: **why does PyTorch add the gradients by default, so that each step must start with `optimizer.zero_grad()`?** The reason is that backpropagation needs this sum to work correctly. The framework cannot tell the difference between "a gradient from a different path in the same backward pass" and "a gradient that is left from the last step". Thus, the training loop must set the gradients to zero before each backward pass. In our MLP, `net.zero_grad()` does this.

## 7. Topological sort: the order of the backward pass

The last question: in the backward pass, in which order do we process the nodes?

The rule: **a node must receive all gradients from its downstream nodes before it sends its gradient upstream.** In `y = x·x + x`, suppose that x receives only the 1 from the addition path and then sends its gradient upstream. Then its gradient is wrong.

The method is to do a **topological sort** of the graph first. The sort puts each node before all nodes that use it. Then go through the list **in reverse order**. When the engine processes a node, all nodes that use it are already done, and their gradients are already added. The full `backward` has only these lines:

```python
def backward(self):
    topo, visited = [], set()
    def build(v):
        if v not in visited:
            visited.add(v)
            for child in v._prev:
                build(child)
            topo.append(v)            # add a node only after all its inputs
    build(self)
    self.grad = 1.0                   # start: ∂L/∂L = 1
    for v in reversed(topo):          # reverse topological order: collect all gradients, then send them upstream
        v._backward()
```

Now the engine is complete. It has the `Value` class and three small classes: `Neuron`, `Layer`, and `MLP`. They have the same structure as in Chapter 3. A neuron calculates `act(w·x + b)`. A layer is a set of neurons side by side, which is the `y = xW + b` of Chapter 2. The full file has 194 lines, or about 125 lines without empty lines and comments. The full code is in [`code/01_engine.py`](code/01_engine.py).

> The idea comes from [micrograd](https://github.com/karpathy/micrograd) by Andrej Karpathy (MIT license). The code of this chapter is rewritten to match the method of this course. Its interface is similar to micrograd. We strongly recommend his video. In two and a half hours, he builds this engine from zero.

## 8. Gradient check: how do we know that the gradients are correct?

Task 2 of Chapter 1 showed the **gradient check**. Move a parameter up by a small amount and down by a small amount, and see how much the loss changes:

```
∂L/∂p ≈ (L(p + ε) − L(p − ε)) / (2ε)
```

This **numerical gradient** is slow and not very accurate. But it **cannot have a derivation error**, because it uses only the forward pass. A comparison with the autograd result is the most reliable way to check backpropagation.

```bash
uv run python chapters/04-backprop-autograd/code/02_grad_check.py
```

`02_grad_check.py` first does two numerical checks (ε = 10⁻⁶). It runs each check two times: one time with the correct engine, and one time with an engine that has an intentional bug. The relative error uses the full gradient vector: `‖g_auto − g_num‖ / (‖g_auto‖ + ‖g_num‖)`.

| What we check | Engine | Max absolute error | Relative error |
|---|---|---:|---:|
| An expression that uses all operations (3 inputs) | Correct (`+=`) | 8.7 × 10⁻¹¹ | 1.8 × 10⁻¹¹ |
| The loss of MLP(1, [8, 8, 1]) (97 parameters) | Correct (`+=`) | 2.0 × 10⁻¹⁰ | 1.8 × 10⁻¹⁰ |
| The same expression | With a bug (`=`) | 2.6 | 0.97 |
| The same MLP | With a bug (`=`) | 0.66 | 0.93 |

The expression is `((x·y + eᶻ).ln() − x/z).tanh() · x + relu(y²) + (x − 3)³/10`. The gradients of the three inputs agree digit by digit:

```
  [expression]  autograd: +2.452530, -2.590708, +0.123690
                numerical: +2.452530, -2.590708, +0.123690
```

At the end, `02_grad_check.py` uses `Value` to rebuild the ReLU network of Chapter 3 (width 8, the same initial parameters, the same 100 points). It does a parity check of the autograd gradients against the 6 formulas that we derived by hand in Chapter 3:

```
Parity check with the hand-derived gradients of Chapter 3 (ReLU MLP, width 8, 100 points):
  W1  shape (1, 8)  max difference 4.2e-17
  b1  shape (8,)    max difference 6.9e-17
  W2  shape (8, 1)  max difference 8.9e-16
  b2  shape (1,)    max difference 1.1e-16
```

In the last chapter, we derived a series of formulas by hand, such as `d_W1 = x.T @ d_z`. Now one line, `backward()`, gives the same gradients. The only difference is floating-point rounding error.

How to read the results: the relative error of the correct engine is of the order of 10⁻¹⁰. This is the truncation and rounding error of the central difference itself, so the two methods agree. **The relative error of the engine with the bug is near 1. This means that the gradient is almost completely wrong.** Note this interesting fact: the code with the bug still runs, and the MLP still "trains". Without a gradient check, you can only think "why does this network not learn well?"

After you add a new operation to the engine, always do a gradient check first. PyTorch includes the same tool. See "From minimal code to production code" at the end of this chapter.

## 9. Train an MLP with the engine

Now we use our own engine to train a network from zero. The network of Chapter 3 needed 20000 steps to train. Our scalar engine is slow (Section 10 explains why). Thus, we use a smaller task: fit `y = sin(x)` on 20 points with equal spacing in [−3, 3]. The network is `MLP(1, [8, 8, 1])`: a 1-D input, **two** hidden layers with 8 neurons each, tanh activation, and 97 parameters in total. Compared with Chapter 3, it has one more layer and a different activation function. With derivation by hand, both changes need new derivations. With our engine, we derive no formula.

The training loop is the same as in Chapter 1. Only the gradient step changes to `loss.backward()`:

```python
for step in range(steps + 1):
    loss = mse(net, xs, ys)           # 1. forward pass: also records the full computational graph
    net.zero_grad()                   # 2. set the gradients to zero (backward uses +=)
    loss.backward()                   # 3. backward pass: calculates the gradients of all parameters
    for p in net.parameters():        # 4. update: p ← p − η · ∂L/∂p
        p.data -= lr * p.grad
```

(In the full code, the last step only evaluates and does not update. See [`code/03_train_mlp.py`](code/03_train_mlp.py).) Run:

```bash
uv run python chapters/04-backprop-autograd/code/03_train_mlp.py
```

Learning rate 0.1, full-batch gradient descent, 500 steps:

| Step | 0 | 1 | 10 | 50 | 100 | 200 | 300 | 400 | 500 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Loss | 0.7163 | 0.4323 | 0.1111 | 0.0837 | 0.0478 | 0.0189 | 0.0145 | 0.0115 | 0.0094 |

The predictions after training (one point in every 4):

| x | sin(x) | Prediction |
|---:|---:|---:|
| −3.00 | −0.141 | −0.323 |
| −1.74 | −0.986 | −0.872 |
| −0.47 | −0.456 | −0.560 |
| 0.79 | 0.710 | 0.803 |
| 2.05 | 0.886 | 0.761 |

The model has learned the general shape of the curve. At the edge (x = −3), the error is still large. More training or a different learning rate can make it better; the guided questions discuss this. We also tried learning rates of 0.2 and 0.3. Then the loss jumps up and down. This is the "step that is too large" of Chapter 1. The important point is: **from start to end, we did not derive one gradient by hand**.

## 10. The cost: a scalar computational graph is slow

At the start, `03_train_mlp.py` prints this number:

```
One forward pass (20 samples) records a computational graph with 3720 Value nodes
```

Each product of a weight and an input, and each addition, is one Python object and one closure. This small network has 20 samples and 97 parameters, and each step builds 3720 nodes. The 500 training steps took 12.3 seconds (about 25 ms per step; one measurement, which changes with the machine load). For a model with billions of parameters, this method cannot work.

Real frameworks do **the same thing, but each node is a full tensor**. One matrix multiplication `Y = X·W` is **one** node in the computational graph, and its `_backward` is also a matrix multiplication. Use the shape rules of Chapter 2. Let the upstream gradient be `G = ∂L/∂Y` (with the same shape as Y). Then:

```
∂L/∂X = G · Wᵀ        ∂L/∂W = Xᵀ · G        ∂L/∂b = sum of G over the rows
```

This is the **vector-Jacobian product (VJP)**. The **Jacobian** is the matrix of the partial derivatives of all outputs for all inputs. The full Jacobian of `Y` for `X` is very large. When `X` is 4×3 and `Y` is 4×2, it already has 8 × 12 = 96 numbers, and in a real model the number is astronomically large. But backpropagation **never needs to build the Jacobian**. It needs only the product "upstream gradient × Jacobian", and this product is one matrix multiplication. The last part of `04_pytorch_compare.py` verifies that `torch.autograd.grad(Y, X, grad_outputs=G)` is equal to `G @ W.T`.

The principle is the same as in our scalar engine. The forward pass records the graph. The backward pass goes in reverse topological order, multiplies the upstream gradient by the local derivative, and adds the gradients at fan-out points. The only difference: "multiply by the local derivative" changes from a scalar multiplication to a matrix multiplication. Highly optimized C++/CUDA code does this multiplication.

> **Why "backward"?** In training, we have **one** output (the loss) and **many** inputs (the parameters). One pass back from the output gives the gradients of the loss for all parameters, at a cost of only a few forward passes. The other method goes forward from the inputs (forward-mode automatic differentiation). Each pass of forward mode gives the derivative for only one input. For billions of parameters, forward mode needs billions of passes. This is why all deep-learning frameworks use reverse mode.

## 11. Summary

- **Derivation by hand does not scale**: each change to the model needs a new derivation, and errors are difficult to find.
- **Chain rule**: the derivative of a composite function = the product of the local derivatives along the path. Add the contributions of all paths.
- **Computational graph**: in the forward pass, each operation records one node (value, inputs, operation).
- **Backpropagation**: start from `L̄ = 1`. In reverse topological order, each node adds "upstream gradient × local derivative" to its inputs with `+=`.
- **Automatic differentiation**: write the local derivative of each operation once. Then the engine calculates the gradient of any combination automatically.
- **Gradient check**: a parity check against numerical gradients is the most reliable way to find bugs in backpropagation.
- **Cost**: a scalar graph is too slow. Real frameworks do the same thing on tensors (VJP).

---

## From minimal code to production code

In PyTorch, the same MLP uses the standard `nn.Linear` and `torch.autograd`. [`code/04_pytorch_compare.py`](code/04_pytorch_compare.py) copies the initial weights of the Value MLP into `nn.Sequential` without change. Then it does a parity check item by item:

```bash
uv run python chapters/04-backprop-autograd/code/04_pytorch_compare.py
```

```python
model = nn.Sequential(nn.Linear(1, 8), nn.Tanh(), nn.Linear(8, 8), nn.Tanh(), nn.Linear(8, 1))
loss = ((model(x) - y) ** 2).mean()   # x has shape (20, 1): one pass for the full batch of samples
model.zero_grad()
loss.backward()                        # torch.autograd: backpropagation on tensors
```

**1. Gradient parity check, parameter by parameter** (both sides use double precision):

| Parameter | Shape | allclose | Max difference |
|---|---|---|---:|
| `0.weight` | (8, 1) | True | 8.3 × 10⁻¹⁷ |
| `0.bias` | (8,) | True | 3.1 × 10⁻¹⁷ |
| `2.weight` | (8, 8) | True | 5.6 × 10⁻¹⁷ |
| `2.bias` | (8,) | True | 7.5 × 10⁻¹⁷ |
| `4.weight` | (1, 8) | True | 4.2 × 10⁻¹⁷ |
| `4.bias` | (1,) | True | 5.6 × 10⁻¹⁷ |

Over the 97 parameters, the maximum gradient difference is 8.3 × 10⁻¹⁷. This is the rounding error of double-precision floating point. The initial loss is 0.716258950201 on both sides.

**2. Training parity check**: both sides train for 500 steps with SGD and a learning rate of 0.1. The final loss is **0.0093662407** on both sides (the same 10 decimal places).

**3. Speed** (one forward pass + one backward pass, one measurement; the numbers change with the machine load, but the order of magnitude of the ratio does not change):

| Samples | Value engine (scalar graph) | PyTorch (tensor graph) | Ratio |
|---:|---:|---:|---:|
| 20 | 43.7 ms | 0.355 ms | about 120× |
| 200 | 775.3 ms | 0.414 ms | about 1900× |

When the number of samples increases 10 times, the time of the scalar engine increases at least 10 times, because the number of nodes increases linearly. In this measurement, the time increased about 18 times. A new run on a different server in 2026-10 gave about 10 times. The time of PyTorch almost does not change, because the additional samples only add rows to the matrices. (Other jobs ran on the machine during this measurement. When the same Value training script runs alone, one step takes about 25 ms.)

**4. `torch.autograd.gradcheck`**: the gradient check that PyTorch includes. Its principle is the same as in our `02_grad_check.py`: it compares the numerical gradient from finite differences with the analytic gradient from autograd. The official documentation says that it needs double-precision inputs. We ran it on all 97 parameters, and the result is `True`.

The table shows what the production code adds and why:

| Minimal code (`01_engine.py`) | Production code (`torch.autograd`) | Why |
|---|---|---|
| One node for each scalar | One node for each tensor operation | The number of nodes decreases from the order of "parameters × samples" to the order of "layers". The Python overhead almost disappears. |
| `_backward` is a scalar multiplication | The backward pass is a VJP (`G·Wᵀ`, `Xᵀ·G`) that calls BLAS / CUDA kernels | Matrix multiplication can use SIMD on the CPU and parallel processing on the GPU. |
| Recursive topological sort | A backward engine in C++ that schedules the work by dependencies | A very deep graph does not overflow the Python recursion stack (see Hands-on task 3). |
| All intermediate values stay in memory | Keeps only the intermediate results that the backward pass needs, and by default frees the graph after the backward pass | Saves GPU memory. Activation checkpointing in Chapter 14 goes further. |
| Seven operations | More than a thousand operations, each with a registered backward function | For a custom operation, you can write the forward / backward yourself with `torch.autograd.Function`. |
| You write the gradient check yourself | `torch.autograd.gradcheck` / `gradgradcheck` | The same idea, included in the framework. |
| `net.zero_grad()` | `optimizer.zero_grad()` | The same reason: by default, the gradients add up. |

In each chapter after this one, we write new operations (attention in Chapter 8, RMSNorm in Chapter 9, and others). For them, we use PyTorch autograd directly and do not write the backward pass by hand again. But now you know what occurs inside `loss.backward()`.

## Adopters and sources

Reverse-mode automatic differentiation is the **de facto standard** of deep-learning frameworks (class B in Section 2.1 of GOAL.md). All main open models are trained with it:

- **PyTorch**: `torch.autograd`, [Autograd mechanics](https://docs.pytorch.org/docs/stable/notes/autograd.html). The official training / inference code of open models such as Llama, Qwen, DeepSeek, and OLMo is based on PyTorch.
- **JAX**: `jax.grad` / `jax.vjp`, [Automatic differentiation](https://docs.jax.dev/en/latest/automatic-differentiation.html). The official implementation of Gemma is based on JAX.
- **TensorFlow**: `tf.GradientTape`, [Introduction to gradients and automatic differentiation](https://www.tensorflow.org/guide/autodiff).

> To be verified: the list above tells which framework the training code of each model family uses. This information comes from a general impression of the public repositories of the projects. We did not open the technical reports one by one to check it when we wrote this section. This point does not affect the content of this chapter, because this chapter teaches only the principle.

---

## Guided questions

Ask Claude Code these questions. Continue until you can explain the answers in your own words:

1. In the example `L = (a·b + c)²`, suppose that a also takes the place of c: `L = (a·b + a)²`. What does the computational graph look like now? Calculate `∂L/∂a` by hand, then verify it with `Value`.
2. Why does the gradient check use the central difference `(L(p+ε) − L(p−ε)) / 2ε` and not the one-sided difference `(L(p+ε) − L(p)) / ε`? What happens when ε is too large? What happens when ε is too small? (You can change `EPS` in `02_grad_check.py` and try.)
3. ReLU has no derivative at 0. Our code gives the derivative 0 when `data == 0`. Can this cause a problem? What does the gradient check do at this point?
4. Reverse mode gives the gradients of "one output for all inputs" in one backward pass. Forward mode gives the derivatives of "all outputs for one input" in one pass. When is forward mode the better choice?
5. After 500 training steps, the fit is worst near x = −3. Is the cause the model capacity, the number of training steps, or the learning rate? Design an experiment to test your guess.
6. Backpropagation uses intermediate values from the forward pass (for example, `other.data` in the multiplication). How many intermediate values must a large model with tens of layers keep in the forward pass? How is this related to the memory use during training?

## Hands-on tasks

For each task, run the code and look at the result.

**Task 1 (basic)**: Calculate by hand the gradients of `L = (a·b + c)²` for a, b, and c at `a = 1, b = 2, c = −1`. First draw the computational graph, then fill in the gradients from right to left. Then change the example at the end of `01_engine.py` to verify your answer.

**Task 2 (core)**: Add a `sigmoid` operation to `Value`: the forward pass is `s = 1 / (1 + e⁻ᵃ)`, and the local derivative is `s·(1 − s)`. Add it to the expression in `expression_case` of `02_grad_check.py`, and make sure that the gradient check passes. Then, in `03_train_mlp.py`, change the activation function of the MLP to `"relu"`, and then to your `sigmoid`. Compare the loss after 500 steps.

**Task 3 (challenge)**: In the speed test of `04_pytorch_compare.py`, change the number of samples from 200 to 2000. You see a `RecursionError`: the recursive topological sort overflows the stack on a deep graph (the loss is a chain of 2000 additions). Rewrite `build` in `backward` as a version without recursion that uses a stack. Use `02_grad_check.py` to make sure that the engine still works correctly. Then measure the time for 2000 samples. To go further, write a `Tensor` class that supports only matrix multiplication, addition, tanh, and mean. Use the VJP formulas of Section 10 for its backward pass, and do a parity check of its gradients against PyTorch.

---

## References

- Andrej Karpathy. *micrograd* (MIT license, the prototype of the engine in this chapter): <https://github.com/karpathy/micrograd>
- Andrej Karpathy. *The spelled-out intro to neural networks and backpropagation: building micrograd* (video): <https://www.youtube.com/watch?v=VMj-3S1tku0>
- Stanford CS231n course notes, *Backpropagation, Intuitions* (an intuitive explanation of "gradients add at fan-out points"): <https://cs231n.github.io/optimization-2/>
- Baydin, Pearlmutter, Radul, Siskind. *Automatic Differentiation in Machine Learning: a Survey*, JMLR 18(153), 2018: <https://www.jmlr.org/papers/volume18/17-468/17-468.pdf> (arXiv: <https://arxiv.org/abs/1502.05767>)
- Rumelhart, Hinton, Williams. *Learning representations by back-propagating errors*, Nature 323, 1986: <https://www.nature.com/articles/323533a0>
- 3Blue1Brown. *Backpropagation calculus*: <https://www.3blue1brown.com/lessons/backpropagation-calculus>
- PyTorch documentation, *Autograd mechanics*: <https://docs.pytorch.org/docs/stable/notes/autograd.html>; `torch.autograd.gradcheck`: <https://docs.pytorch.org/docs/stable/generated/torch.autograd.gradcheck.gradcheck.html>

**Next chapter**: With automatic differentiation, a network can be as deep as we want. But until now, we only did regression, which predicts one number. Suppose that we must decide if an image shows a cat or a dog, or which word comes next. Then the output becomes "the probability of each class", and the mean squared error is no longer a good choice. In Chapter 5, we discuss classification and probability: softmax, cross-entropy, and why classification does not use the mean squared error.
