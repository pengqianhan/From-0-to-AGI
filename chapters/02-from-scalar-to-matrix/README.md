# Chapter 2: From scalars to matrices — y = XW + b

**English** · [中文](README.zh.md)

> **Goal**: After this chapter, you can write `Y = XW + b` on paper and give the correct shape of each matrix. You can use the shape rule `(m, k) @ (k, n) → (m, n)` to check if a line of code can run. You can also train a multivariate linear regression with the gradient in matrix form, `2/N · Xᵀ(ŷ − y)`.

📺 **Video**: Not published yet. To render it on your computer, run `bash chapters/02-from-scalar-to-matrix/video/build.sh`.
🧪 **Self-check**: After the chapter, type `/ch02-matrix` in Claude Code.

---

In the last chapter, we used `ŷ = a·x + b` to learn the four steps of training: model, loss, gradient, and update. But that model has only **one input** and **two parameters**. In the real world, a house price does not depend only on the area. In one layer of a large language model, the input is a vector with thousands of dimensions. The layer also processes hundreds or thousands of samples at the same time.

This chapter answers one question: **the input changes from one number to a set of numbers, and the samples change from one to a batch. How do we write the model and the training?** The answer is vectors and matrices. None of the four steps changes. Only the scalars become matrices. After this change, the code is shorter, and it also runs faster.

## 1. Intuition: one input is not enough

We predict house prices again. Chapter 1 used only the area x. Now each house has three features:

| Feature | Example |
|---|---|
| x₁ area | 80 m² |
| x₂ number of bedrooms | 2 |
| x₃ distance to the city center | 5 km |

The most natural extension is this: give each feature a **weight**. The weight tells how much the feature changes the price. Then add a bias b:

```
ŷ = w₁·x₁ + w₂·x₂ + w₃·x₃ + b
```

The slope a of Chapter 1 is "the weight when there is only one feature". We give all prices in units of 10,000 yuan (10k yuan). Assume that each square meter adds 0.8 and each bedroom adds 5. Each 1 km of distance from the city center subtracts 3. The base price is 20. Then the prediction for this house is `0.8×80 + 5×2 − 3×5 + 20 = 79` (10k yuan).

Now there is a problem. What if there are more than 3 features (thousands in a large model)? Must we write all the terms one by one? We need a notation that treats "a set of numbers" as one object.

## 2. Vectors and the dot product

Put the three features in a list. This list is a **vector**: `x = [80, 2, 5]`. Put the three weights in a vector too: `w = [0.8, 5, −3]`. The long sum of products above is the **dot product**: multiply the items at the same position, then add all the products.

```
w · x = w₁x₁ + w₂x₂ + w₃x₃ = Σ wᵢxᵢ
ŷ = w · x + b
```

Run:

```bash
uv run python chapters/02-from-scalar-to-matrix/code/01_matrix_basics.py
```

The code ([`code/01_matrix_basics.py`](code/01_matrix_basics.py)) is:

```python
dot_loop = sum(w[i] * x[i] for i in range(3))   # Σ wᵢxᵢ, one term at a time
dot_np = w @ x                                  # the same calculation in one line of NumPy
```

The output is `w · x = 59.0` and `ŷ = w · x + b = 79.0`.

The dot product also has a geometric meaning. We use it many times in later chapters. When two vectors point in more similar directions, their dot product is larger. When two vectors are perpendicular, their dot product is 0. The attention mechanism in Chapter 8 uses the dot product to measure "how related two words are".

## 3. A batch of samples: the matrix

One house needs one dot product. What about 10,000 houses? Put the feature vector of each house in one **row**. Then put the rows on top of each other. The result is a **matrix** X:

```
      area  rooms dist
X = [[  80,   2,   5.0],    ← house 1
     [ 120,   3,   2.0],    ← house 2
     [  60,   1,   8.0],    ← house 3
     [ 100,   3,   3.5]]    ← house 4         shape (4, 3)
```

**Each row is one sample. Each column is one feature.** We write the shape as `(number of rows, number of columns)` = `(4, 3)`.

Put the weights in one column W (shape `(3, 1)`). Then one **matrix multiplication** `X @ W` calculates the dot products of all four houses at the same time:

```python
scores = X @ w          # (4, 3) @ (3,) → (4,)
scores + b              # [ 79. 125.  49. 104.5]
```

These four numbers are the predicted prices of the four houses. **One matrix multiplication = one dot product for each sample.**

## 4. How to calculate a matrix multiplication, and where the shape rule comes from

The definition of the matrix multiplication `C = A @ B` is only one sentence:

```
C[i][j] = row i of A · column j of B
```

The most direct implementation uses three nested loops ([`code/02_matrix_multiply.py`](code/02_matrix_multiply.py)):

```python
for i in range(m):              # row i of the result
    for j in range(n):          # column j of the result
        for p in range(k):      # dot product of row i and column j
            C[i][j] += A[i][p] * B[p][j]
```

Run `uv run python chapters/02-from-scalar-to-matrix/code/02_matrix_multiply.py`. The script prints the calculation for each cell:

```
C[0][0] = 1×7 + 2×9 + 3×11 = 58
C[0][1] = 1×8 + 2×10 + 3×12 = 64
C[1][0] = 4×7 + 5×9 + 6×11 = 139
C[1][1] = 4×8 + 5×10 + 6×12 = 154
```

The definition gives the most important rule of this chapter, the **shape rule**:

```
(m, k) @ (k, n) → (m, n)
```

- The two inner k values must be equal. A row of A and a column of B make a dot product, so the two vectors must have the same length.
- The number of rows of the result comes from A. The number of columns comes from B. The inner k "disappears" in the sum.

At the end, the script tests three examples:

| Left | Right | Result |
|---|---|---|
| (4, 3) | (3, 1) | (4, 1) |
| (32, 128) | (128, 64) | (32, 64) |
| (3, 2) | (3, 5) | Error: the inner dimensions 2 ≠ 3 |

In deep-learning code, many bugs come from shapes that do not agree. Make this a habit: for each line of matrix code, write the shapes in a comment.

> **Note:** `@` and `*` are two different operations. `A @ B` is matrix multiplication. `A * B` is **element-wise multiplication** (the Hadamard product): it multiplies the items at the same position. In `01_matrix_basics.py`, `M @ N = [[19, 22], [43, 50]]` and `M * N = [[5, 12], [21, 32]]`. When the two shapes are the same, neither operation gives an error, but the results are completely different. This type of bug is the most difficult to find.

## 5. Y = XW + b: give each letter a shape

Extend Section 3 to "many outputs". The result is the most common line in a neural network:

```
Y   =   X    @   W    +   b
(N,n)  (N,k)    (k,n)    (n,)
```

- **N**: the number of samples in this batch. This is the **batch dimension**.
- **k**: the number of input features of each sample.
- **n**: the number of outputs. **Column j** of W holds the k weights that calculate output j.

An important observation: **the shape of W does not depend on N**. You can give the model 4 samples or 10,000 samples at one time: it uses the same W. This is why you can change the batch size during training without a change to the model.

[`code/03_linear_layer.py`](code/03_linear_layer.py) sends the 4 houses through two layers, one after the other:

```python
def linear(X, W, b):
    return X @ W + b            # (batch, k) @ (k, n) + (n,) → (batch, n)
```

```
(4, 3) --[@W1 + b1]--> (4, 2) --[@W2 + b2]--> (4, 1)
```

### Broadcasting: how to add b to each row

`X @ W` has the shape `(N, n)`, but b has only the shape `(n,)`. How can we add two different shapes? NumPy and PyTorch automatically add b to **each row**. The effect is the same as N copies of b, but there are no real copies in memory. This is **broadcasting**.

The rule: **align the two shapes from the last dimension**. Broadcasting is possible only when, in each dimension, the two sizes are equal, or one of them is 1 (or missing). The output of `01_matrix_basics.py` is:

```
(4, 2) matrix + vector of shape (2,) → [10, -1] is added to each row
(4, 2) + (3,) gives an error: operands could not be broadcast together with shapes (4,2) (3,)
```

Thus, the length of b must be equal to the number of outputs n. Broadcasting is convenient, but it is also dangerous. Two shapes can broadcast "by accident". Then the code gives no error, but it can calculate something completely different from what you want (guided question 4).

## 6. The gradient in matrix form

The model is now `ŷ = XW + b`. The loss is still the mean squared error `L = 1/N · Σ (ŷᵢ − yᵢ)²`. How do we calculate the gradient?

Remember Chapter 1: `∂L/∂a = 2/N · Σ (ŷᵢ − yᵢ) · xᵢ`. Now there are k weights. The gradient of weight j has the same form:

```
∂L/∂w_j = 2/N · Σᵢ (ŷᵢ − yᵢ) · x_ij
```

Put the gradients for j = 1…k in one column. The result is exactly the matrix multiplication `Xᵀ @ (ŷ − y)`, because row j of Xᵀ holds feature j of all samples. Thus:

```
∂L/∂W = 2/N · Xᵀ (ŷ − y)        (k, N) @ (N, 1) → (k, 1)
∂L/∂b = 2/N · Σ (ŷ − y)
```

Check the shape: the result is `(k, 1)`, the same as W. **A gradient always has the same shape as its parameter**, because each parameter needs its own gradient. Later, when you see a more complex formula, check the shapes first. This check stops half of all errors.

`Xᵀ` is the **transpose** of X: rows become columns, and columns become rows. `(N, k)` becomes `(k, N)`.

## 7. Minimal code: train a multivariate linear regression

Run:

```bash
uv run python chapters/02-from-scalar-to-matrix/code/04_multivariate_regression.py
```

The core is almost the same as in Chapter 1 ([`code/04_multivariate_regression.py`](code/04_multivariate_regression.py)):

```python
def gradients(W, b, X, y):
    n = len(X)
    err = X @ W + b - y                  # (N, 1): residual ŷ − y
    grad_W = 2 / n * X.T @ err           # ∂L/∂W = 2/N · Xᵀ(ŷ − y), (3, 1)
    grad_b = 2 / n * err.sum(axis=0)     # ∂L/∂b = 2/N · Σ(ŷ − y)
    return grad_W, grad_b

for _ in range(steps):
    grad_W, grad_b = gradients(W, b, X, y)
    W = W - lr * grad_W                  # W ← W − η · ∂L/∂W
    b = b - lr * grad_b                  # b ← b − η · ∂L/∂b
```

We made 200 houses. The true relation is `price = 0.8·area + 5·bedrooms − 3·distance + 20`, plus noise with a standard deviation of 5 (10k yuan).

**Standardize first.** The three features have very different scales: their standard deviations are 30.14, 1.11, and 5.19. Calculate the critical learning rate `2/λ_max` with the method of Chapter 1:

| | Critical learning rate |
|---|---:|
| Original features | 8.45 × 10⁻⁵ |
| After standardization (subtract the mean of each feature, divide by its standard deviation) | 0.879 |

With the original features, the "bowl" is narrow and steep in the direction of the area. Thus the learning rate must be less than 1/10,000. The flattest direction is almost the direction of the bias b. Its curvature is only about 1/240,000 of the curvature in the steepest direction. With such a small learning rate, hundreds of thousands of steps are necessary to get to the bottom of the bowl. In Hands-on task 2, you see this yourself.

**Standardization** makes the curvature about the same in all directions. Then the learning rate can be much larger. This is the answer to guided question 3 of Chapter 1. It is also the predecessor of "normalization" in Chapter 6.

Start from W = 0 and b = 0, with a learning rate of 0.1 (the parameters are in the standardized space):

| Step | w_area | w_rooms | w_dist | b | Loss |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.000 | 0.000 | 0.000 | 0.000 | 7875.146 |
| 1 | 5.014 | 1.332 | −3.077 | 16.682 | 5043.738 |
| 10 | 22.021 | 5.039 | −14.099 | 74.455 | 112.024 |
| 20 | 24.297 | 5.260 | −15.718 | 82.450 | 22.981 |
| 50 | 24.571 | 5.248 | −15.918 | 83.410 | 21.942 |
| 200 | 24.571 | 5.248 | −15.918 | 83.411 | 21.942 |

Converted back to the original units (10k yuan):

| | Gradient descent | Closed-form solution (least squares) | True value |
|---|---:|---:|---:|
| w_area (per m²) | 0.815 | 0.815 | 0.800 |
| w_rooms (per bedroom) | 4.734 | 4.734 | 5.000 |
| w_dist (per km) | −3.070 | −3.070 | −3.000 |
| b | 19.837 | 19.837 | 20.000 |

Look at these results:

- **The training converges in 50 steps**, as in Chapter 1. First, the loss decreases in large steps. Then it decreases more and more slowly.
- **The final loss is 21.94, near the noise variance 5² = 25.** The model learned the relation. The rest is the noise in the data, and no model can remove it.
- **Gradient descent and the closed-form solution agree to three decimal places.** The small differences from the true values come from the noise.
- Use the learned model to predict the price of a new house (100 m², 3 bedrooms, 5 km to the city center). The prediction is 100.2 (10k yuan).

## 8. Loops vs vectorization: how large is the difference?

You can calculate the same formula with a Python loop, one term at a time. You can also calculate it with one matrix operation. The second method is **vectorization**. Run:

```bash
uv run python chapters/02-from-scalar-to-matrix/code/05_loop_vs_vectorized.py
```

The script first makes sure that the different versions give the same result. Then it measures the time (the fastest of many runs). The results of one run on our machine (4-core CPU) are:

**Comparison 1: one forward pass `Y = X @ W`, X is (1000, 100), W is (100, 10)**

| Version | Time | Compared with three loops |
|---|---:|---:|
| Three nested Python loops (`matmul` in `02`) | 62.70 ms | 1× |
| Loop only over samples, one `np.dot` for each row | 1.94 ms | 32× faster |
| One `X @ W` | 0.057 ms | 1091× faster |

**Comparison 2: multivariate linear regression (5000 houses, 3 features), 200 training steps**

| Version | Time |
|---|---:|
| Gradient with a loop over each sample and each feature | 538.99 ms |
| Matrix form `2/N · Xᵀ(ŷ − y)` | 21.73 ms (25× faster) |

The parameters from the two training versions differ by at most 3.6 × 10⁻¹⁵. This is floating-point error. The times change with the machine and its load. On the same machine, in more runs, the speedup of the forward pass was between 950× and 1170×. In 2026-10, a run on a different server gave 1400–1650× for the forward pass and about 50× for training. When the machine does other work at the same time (for example, it renders a video), the code runs even more slowly. Look only at the order of magnitude.

Comparison 2 is only tens of times faster (the exact value depends on the machine). This is much less than the thousands of times in comparison 1, and this difference is important. Here, each matrix operation processes only 5000×3 numbers. There are also 200 iterations of a Python loop, with function calls. **The more work each call gives to the low-level library, the larger the advantage of vectorization.** This is also one reason why GPU training uses large batches.

**Why is vectorization faster?** For each multiply-add, a Python loop must interpret a line of code, check types, and make a new number object. The real calculation is only a small part of the work. `X @ W` gives the full block of data to a linear algebra library (BLAS) that is written in C and assembly. The data is contiguous in memory. One CPU instruction can calculate several numbers at the same time (SIMD), and the library can use many cores.

On a GPU, thousands of cores do multiply-adds at the same time, and matrix multiplication is the operation that GPUs do best. Most of the computation in the training and inference of large models is in matrix multiplications. (In Chapter 12, the estimate of the training compute, `C ≈ 6ND`, counts mainly the multiply-adds in matrix multiplications.) This is why all of deep learning is written in matrix form.

## 9. Summary

- **A set of inputs → a vector**: `ŷ = w · x + b`. The dot product multiplies the items at the same position and adds the products.
- **A batch of samples → a matrix**, with one sample in each row: `Y = XW + b`, shape `(N, k) @ (k, n) + (n,) → (N, n)`.
- **Shape rule** `(m, k) @ (k, n) → (m, n)`: the two inner k values must be equal.
- **Broadcasting**: NumPy aligns b from the last dimension and adds it to each row automatically.
- **Gradient** `∂L/∂W = 2/N · Xᵀ(ŷ − y)`: it has the same shape as W.
- **Vectorization**: for the same calculation, one matrix operation is tens to thousands of times faster than a Python loop.
- The four steps of training (model, loss, gradient, update) did not change.

---

## GPU measurements (one RTX 3090)

> **Note:** All numbers in the text above come from CPU runs. This section uses one NVIDIA GeForce RTX 3090 (24 GB of GPU memory, Ampere architecture). The data sheet gives these values: BF16 tensor-core dense peak about 71 TFLOPS, FP32 about 35.6 TFLOPS, memory bandwidth about 936 GB/s. Software: PyTorch 2.11.0+cu128, CUDA 12.8, October 2026. The server sets the power limit of this card to 240 W (the factory default is 350 W). Under a continuous full load, the card decreases its clock frequency. Thus the absolute values of compute and bandwidth are lower than on a 3090 at full power, and the relative values are more reliable. If you do not have a GPU, skip this section.

Run:

```bash
uv run python chapters/02-from-scalar-to-matrix/code/07_gpu_matmul.py
```

The script calculates the same square matrix multiplication `(N, N) @ (N, N)`, for N from 64 to 8192. Each multiplication has 2N³ floating-point operations. The CPU is an AMD Threadripper PRO 3995WX, with a fixed number of 8 threads. Both sides use float32 (TF32 is off on the GPU). The last column uses the BF16 tensor cores of the GPU. After each GPU call, the script waits for the GPU to finish before it stops the timer. Thus it measures the time "from the start of one matrix multiplication in Python to the result". The data is already in CPU memory or GPU memory before the timing. The times are medians:

| N | CPU float32 | GPU float32 | GPU speedup over CPU | CPU TFLOPS | GPU float32 TFLOPS | GPU BF16 TFLOPS |
|---:|---:|---:|---:|---:|---:|---:|
| 64 | 0.014 ms | 0.024 ms | 0.60× (slower) | 0.037 | 0.02 | 0.02 |
| 128 | 0.025 ms | 0.039 ms | 0.63× (slower) | 0.170 | 0.11 | 0.18 |
| 256 | 0.089 ms | 0.030 ms | 2.93× | 0.376 | 1.10 | 1.05 |
| 512 | 0.860 ms | 0.040 ms | 21.26× | 0.312 | 6.64 | 10.31 |
| 1024 | 5.356 ms | 0.136 ms | 39.46× | 0.401 | 15.82 | 34.82 |
| 2048 | 39.707 ms | 0.776 ms | 51.15× | 0.433 | 22.13 | 56.43 |
| 4096 | 310.047 ms | 8.138 ms | 38.10× | 0.443 | 16.89 | 51.60 |
| 8192 | 2.40 s | 63.960 ms | 37.54× | 0.458 | 17.19 | 50.62 |

This table shows the statement of Section 8 again, but now on a GPU: "the more work each call gives to the low-level library, the larger the advantage of vectorization". For N ≤ 512, the GPU column almost does not change. It stays at 0.02–0.04 ms. This time is the fixed cost to start one GPU calculation and wait for the result. The fixed cost does not depend on the size of the matrix. Thus, for small matrices such as 64 and 128, the CPU is faster.

For N of 1024 and more, the fixed cost becomes a small part of the total time. Then the GPU is 38–51× faster, and the BF16 tensor cores are 2–3× faster than float32. This result supports two statements: "GPU training uses large batches", and "Chapter 14 goes down to BF16" (in the table of "From minimal code to production code"). A GPU needs sufficiently large matrices to be fully used.

One result was unexpected: float32 is fastest at 2048. Larger matrices drop to about 17 TFLOPS, only half of the data-sheet value. The cause is the power limit of this card, 240 W (the factory default is 350 W). In a separate test, we ran large matrix multiplications for several seconds without a pause. The power stayed at 240 W, and the core clock dropped from about 1.7 GHz (idle) to 0.8–1.0 GHz. The longer the calculation, the lower the clock. Thus, on your own card, the numbers for large matrices will probably be better.

## From minimal code to production code

The standard PyTorch code for the same task is in [`code/06_pytorch_version.py`](code/06_pytorch_version.py). The model becomes `nn.Linear(3, 1)`. The five lines of the training loop from Chapter 1 **do not change at all**:

```python
model = nn.Linear(in_features=3, out_features=1)   # linear layer with 3 inputs and 1 output
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
loss_fn = nn.MSELoss()

for step in range(200):
    y_hat = model(x)              # 1. forward pass: (200, 3) → (200, 1), the full batch at one time
    loss = loss_fn(y_hat, t)      # 2. loss
    optimizer.zero_grad()         # 3. set the gradients to zero
    loss.backward()               # 4. backward pass: autograd calculates ∂L/∂W = 2/N·Xᵀ(ŷ−y)
    optimizer.step()              # 5. update
```

Run `uv run python chapters/02-from-scalar-to-matrix/code/06_pytorch_version.py`. The output is:

```
nn.Linear(3, 1): weight shape (1, 3), bias shape (1,)
Input batch (4, 3) → output (4, 1); maximum difference from x @ weight.T + bias 0.0e+00
...
After 200 steps (standardized space):
  PyTorch  W = [ 24.571   5.248 -15.918], b = [83.411]
  NumPy    W = [ 24.571   5.248 -15.918], b = [83.411]
  float64 maximum difference = 0.0e+00
  float32 (default precision of PyTorch) maximum difference = 1.8e-05
```

The table shows what the production code adds and why:

| Minimal code | Production code | Why |
|---|---|---|
| Derive `grad_W = 2/N * X.T @ err` by hand | `loss.backward()` | In this chapter, we can still derive the gradient by hand. Chapter 3 adds a nonlinearity. With many layers, a derivation by hand is no longer possible. In Chapter 4, you write autograd yourself. |
| W has the shape `(k, n)`; calculate `X @ W` | `weight` has the shape `(n, k)` = `(output, input)`; calculate `x @ weight.T + bias` | This is the PyTorch convention. The documentation writes it as `y = xAᵀ + b`. When you read the code of other people or load their weights, be careful with this transpose. |
| The input must be `(N, k)` | The input can be `(*, k)`: any number of leading dimensions | In the Transformer of Chapter 9, the input is `(batch, sequence length, dimension)`. `nn.Linear` operates only on the last dimension. |
| float64 | float32 by default | Single precision uses half the memory and is much faster on a GPU. The cost is a numerical difference of about 10⁻⁵ (the 1.8e-05 above). Chapter 14 goes down to BF16. |

**Parity check**: with float64, the PyTorch version and the NumPy version (matrix gradients by hand) are **exactly the same** after 200 steps (the difference is 0). With the default float32, the difference is about 10⁻⁵. Only when the production code agrees with the minimal code can we trust it.

---

## Guided questions

Ask Claude Code these questions. Continue until you can explain the answers in your own words:

1. What does the dot product of two vectors mean geometrically? If two vectors are perpendicular, what is their dot product? Why does the attention mechanism in Chapter 8 use the dot product to measure how "related" two words are? Ask Claude Code to explain it with a figure.
2. Matrix A is `(2, 3)` and B is `(3, 5)`. What is the shape of `A @ B`? What about `B @ A`? Is matrix multiplication commutative?
3. `03_linear_layer.py` has the line `H = X @ W1 + b1`. What does each **column** of the weight matrix W1 represent? What does each **row** represent? If you initialize all of W1 to 0, what is the difference between the two outputs?
4. The shape of `y` is `(200,)`, and the shape of `y_hat` is `(200, 1)`. What is the shape of `y_hat - y`? Why does this bug not give an error, but silently makes the training wrong? (Guess first, then try it in Python.)
5. In `05`, why is the training comparison only tens of times faster, but the forward-pass comparison thousands of times faster? If you change the number of houses from 5000 to 50, does the speedup become larger or smaller?
6. In Section 7, without standardization, the critical learning rate is only 8.45 × 10⁻⁵. Train with this learning rate. Which weight learns faster, the weight of the area or the weight of the bedrooms? Why?

## Hands-on tasks

For each task, run the code and look at the result.

**Task 1 (basic)**: Write a function that calculates the dot product of two vectors. Use only Python lists, not NumPy. Test it with `[1, 2, 3]` and `[4, 5, 6]`: the result must be 32. Then improve `matmul` in `02_matrix_multiply.py`: when the dimensions do not agree, print a clear error message. For example: "A is (3, 2), B is (3, 5): A has 2 columns ≠ B has 3 rows, so A @ B is not possible".

**Task 2 (core)**: In `04_multivariate_regression.py`, remove the standardization and train on the original features. First, use `critical_lr` to calculate the critical value. Try 0.9 times and 1.05 times this value. Record the loss and the four parameters after 200 steps. Then, for 0.9 times, find the parameter that is farthest from its true value after 200 steps. Increase the number of steps to 20,000 and to 200,000, and look again. Explain the result with this idea: "the bowl curves by different amounts in different directions".

**Task 3 (challenge)**: Change `03_linear_layer.py` to three layers (add a `2-dim → 3-dim` layer in the middle). Print the shape of the tensor after each step. Then merge the weights of the three layers into **one** W and **one** b. Make sure that the merged single layer gives exactly the same output as the three layers. Think about this: if any number of layers is equal to one layer, where does the "deep" in deep learning come from? Chapter 3 answers this question.

---

## References

- NumPy documentation, *Broadcasting* (the official description of the broadcasting rules): <https://numpy.org/doc/stable/user/basics.broadcasting.html>
- NumPy documentation, *What is NumPy?* ("Why is NumPy fast?": vectorization and precompiled C code): <https://numpy.org/doc/stable/user/whatisnumpy.html>
- PyTorch documentation, `torch.nn.Linear` (`y = xAᵀ + b`, input shape `(*, in_features)`): <https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html>
- 3Blue1Brown. *Essence of Linear Algebra* (matrix multiplication = composition of linear transformations): <https://www.3blue1brown.com/topics/linear-algebra>
- Goodfellow, Bengio, Courville. *Deep Learning*, Chapter 2, "Linear Algebra": <https://www.deeplearningbook.org/contents/linear_algebra.html>
- Zhang et al. *Dive into Deep Learning*, Section 3.1, "Linear Regression" (includes a comparison of the speedup from vectorization): <https://d2l.ai/chapter_linear-regression/linear-regression.html>
- Andrew Ng. *CS229 Lecture Notes*, Chapter 1 (multivariate linear regression and the normal equations): <https://cs229.stanford.edu/main_notes.pdf>

**Next chapter**: At the end, `03_linear_layer.py` shows one fact. Two linear layers in a stack are still one matrix, `W1 @ W2`, so two layers are equal to one layer. A stack of any number of linear functions is still linear, and it can never draw a curve. But most relations in the real world are curved. In Chapter 3, we add a small "nonlinearity" between the two layers. We see how it lets the model fit curves. This is a neural network.
