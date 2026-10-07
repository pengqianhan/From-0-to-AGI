# Chapter 6: Stable training — Initialization, normalization, residual connections, AdamW, and learning-rate schedules

**English** · [中文](README.zh.md)

> **Goal**: After this chapter, you can explain why a network with tens of layers "does not train". You can fix it yourself with initialization, RMSNorm, residual connections, AdamW, warmup + decay, and gradient clipping. You can also use comparison experiments to show what each method fixes.

📺 **Video**: Not published yet. To render it on your computer, run `bash chapters/06-training-stability/video/build.sh`.
🧪 **Self-check**: After the chapter, type `/ch06-training-stability` in Claude Code.

---

In the previous chapter, we trained a classifier with softmax and cross-entropy. Now all four steps of training a model (model, loss, gradient, update) are in place, and the network calculates its gradients automatically. But all our networks had only two or three layers. Large language models often have tens of layers: Qwen3-0.6B has 28 layers, and Llama 3 405B has 126 layers.

This chapter solves one problem: **when the network becomes deeper, training fails. What can we do?** First, we see how training fails. Then we fix it with six methods that all large models use today, one method at a time.

This chapter also closes the question from Chapter 1. A learning rate that is too large causes divergence. So how do we set the learning rate when we train a large model?

## 1. The problem: where does the signal go after 30 layers?

Start with the simplest experiment. Build a 30-layer MLP with a width of 256 in each layer. Each layer does "multiply by a matrix, then ReLU":

```
h_l = ReLU(h_{l-1} · W_l)
```

Do not train the network. Send a batch of random inputs (standard deviation 1) through it, and look at two values in each layer. In the forward pass, look at the standard deviation of the output of each layer. In the backward pass, look at the standard deviation of the error signal `∂L/∂h_l`. Run:

```bash
uv run python chapters/06-training-stability/code/01_signal_propagation.py
```

The weights come from `N(0, 1)`. The output is (a part of it; the script prints the full output):

| Layer | 1 | 10 | 20 | 30 |
|---|---:|---:|---:|---:|
| Activation std | 9.35 | 2.59 × 10¹⁰ | 7.66 × 10²⁰ | 2.27 × 10³¹ |
| Error signal std | 1.79 × 10²⁹ | 6.48 × 10¹⁹ | 2.33 × 10⁹ | 0.0622 |

Each layer multiplies the values by about 11. After 30 layers, the values are 10³¹. In the backward pass, the error signal goes back from layer 30, and it also becomes larger at each layer. At layer 1, it is 10²⁹. If we update the parameters with these gradients, the first step overflows to NaN.

Now make the weights smaller, `N(0, 0.01²)`. The activations decrease from 0.0935 to 2.27 × 10⁻²⁹ at layer 30. The error signal at layer 1 is only 1.79 × 10⁻²⁹. This is the **vanishing gradient** problem: the first layers receive almost no learning signal. The first case, with the very large values, is the **exploding gradient** problem.

The cause is basic arithmetic. One layer is one multiplication, so 30 layers are 30 multiplications in a sequence. Let the gain of each layer be a little different from 1, for example 1.1 or 0.9. Then 30 multiplications give 17× or 0.04×. If the gain is far from 1, the result is extremely large or extremely small. **Depth changes any small error in scale into an exponential error.**

The six methods in this chapter all answer one question: how do we keep the signal from exploding or vanishing over tens of layers? And how do we keep the size of each parameter update under control?

## 2. Initialization: keep the scale the same in each layer

### 2.1 Intuition and formula

Let each component of the input `h` have the variance `v`. Multiply `h` by a random `fan_in × fan_out` matrix whose elements have the variance `σ²`. Each output component is a sum of `fan_in` terms, so its variance is `fan_in · σ² · v`. To keep the variance the same, we need `fan_in · σ² = 1`. This is the idea of **Xavier initialization** (Glorot & Bengio, 2010).

ReLU sets half of the values to 0, so the variance becomes half again. Add back this factor of 1/2, and you get **Kaiming initialization** (He et al., 2015):

```
Var(W) = 2 / fan_in, that is, std = sqrt(2 / fan_in)
```

The code is one line:

```python
def kaiming_std(fan_in: int = WIDTH) -> float:
    return math.sqrt(2 / fan_in)      # Var(W) = 2 / fan_in
```

### 2.2 Experiment

We use the same 30-layer network with five initializations (the output of `01_signal_propagation.py`):

| Initialization | Weight std | Gain per layer | Activation std at layer 30 | Error signal, layer 1 / layer 30 |
|---|---:|---:|---:|---:|
| std = 1.0 | 1 | 11.3 | 2.27 × 10³¹ | 2.87 × 10³⁰ |
| std = 0.01 | 0.01 | 0.113 | 2.27 × 10⁻²⁹ | 2.87 × 10⁻²⁸ |
| std = 0.02 | 0.02 | 0.226 | 2.44 × 10⁻²⁰ | 1.54 × 10⁻¹⁹ |
| Xavier, sqrt(1/fan_in) | 0.0625 | 0.707 | 1.71 × 10⁻⁵ | 3.46 × 10⁻⁵ |
| **Kaiming, sqrt(2/fan_in)** | **0.0884** | **1.000** | **0.56** | **0.801** |

The "gain per layer" is the value from theory, `std · sqrt(fan_in / 2)`. Its 30th power explains the orders of magnitude in the table. With Kaiming, the gain is exactly 1. After 30 layers, the activation std is still 0.56, and 80% of the error signal from layer 30 arrives back at layer 1. Xavier is different only by a factor of 2, which gives a gain of 0.707 per layer. After 30 layers, only about 10⁻⁵ of the signal is left.

**Calculate the scale of the initialization from fan_in and the activation function. Do not choose a number without a reason.**

### 2.3 Then why do large models all use 0.02?

Open the configuration file of an open model. You often see `"initializer_range": 0.02`. The `config.json` files of Qwen3-0.6B, Gemma 3 1B, OLMo 2, and SmolLM3 all use 0.02. The OLMo 2 technical report also states that it initializes all parameters from a normal distribution with mean 0 and standard deviation 0.02.

But in the table above, a plain network with std = 0.02 keeps only 10⁻²⁰ of the signal after 30 layers. This is not a contradiction, because a large model is not a plain stack of layers. It has **normalization** and **residual connections** (the next two sections). With these two parts, the scale of the signal no longer depends on weights that are exactly `sqrt(2/fan_in)`. A fixed small std is sufficient. Also, at a width of 1024, `sqrt(1/1024) ≈ 0.031`, so 0.02 and Xavier have almost the same magnitude.

Large models often add one more detail to the initialization. They make **the output projections that write back to the residual stream smaller again, by a factor that depends on the number of layers**. (The residual stream is the main path of `h` through the blocks; see Section 4.) GPT-2 does this, and nanoGPT uses the same method: these matrices use `0.02 / sqrt(2 · number of layers)`. The experiment in Section 4 shows why.

Note that not all models do this. The OLMo 2 report says that they changed from initialization "scaled by layer" back to a uniform 0.02. In their experiments, the uniform value was more stable. The DeepSeek-V3 technical report says that it initializes all parameters with a standard deviation of 0.006.

## 3. Normalization: LayerNorm → RMSNorm

### 3.1 Intuition

Initialization controls only the start. When training starts, the weights change, and the scale of each layer slowly drifts. A more complete solution: **each layer forces its input back to a standard scale**, independent of what occurred before. This is **normalization**.

### 3.2 Formula and code

**LayerNorm** (Ba et al., 2016) operates on the vector `x` (length d) of each sample. It subtracts the mean and divides by the standard deviation. Then it multiplies by a learnable scale `γ` and adds a shift `β`:

```
LayerNorm(x) = (x − μ) / sqrt(σ² + ε) · γ + β,   μ = mean(x), σ² = mean((x − μ)²)
```

**RMSNorm** (Zhang & Sennrich, 2019) removes the "subtract the mean" step and `β`. It only divides by the root mean square:

```
RMSNorm(x) = x / sqrt(mean(x²) + ε) · γ
```

```python
def layer_norm(x, gamma, beta, eps=1e-6):
    mu = x.mean(-1, keepdim=True)
    var = ((x - mu) ** 2).mean(-1, keepdim=True)
    return (x - mu) / torch.sqrt(var + eps) * gamma + beta

def rms_norm(x, gamma, eps=1e-6):
    rms = torch.sqrt((x * x).mean(-1, keepdim=True) + eps)
    return x / rms * gamma
```

Run `02_normalization.py`. It uses a 4-dimensional vector to show the difference between the two:

| | Output | Mean | RMS |
|---|---|---:|---:|
| Input x | [2, 4, 6, 8] | | |
| LayerNorm(x) | [−1.342, −0.447, 0.447, 1.342] | 0.000 | 1.000 |
| RMSNorm(x) | [0.365, 0.73, 1.095, 1.461] | 0.913 | 1.000 |

Both set the scale to 1. The only difference is that LayerNorm also moves the mean to 0. The script also shows two more facts. First, if we subtract the mean from the input, the two outputs are identical (maximum difference 0). Second, if we multiply x by 100, the RMSNorm output almost does not change (maximum change 1.19 × 10⁻⁷). This property is **scale invariance**.

### 3.3 What RMSNorm removes, and why it is sufficient

RMSNorm removes "shift invariance". If we add a constant to all elements of the input, the LayerNorm output does not change, but the RMSNorm output changes. The RMSNorm paper assumes that **scale invariance** (bringing the scale back) makes training stable, and that subtracting the mean has little effect.

In experiments, RMSNorm works as well as LayerNorm. But it calculates one less mean and stores one less set of `β` (the parameters decrease from 2d to d). The paper reports that RMSNorm is 7%–64% faster on different models and frameworks. Today, almost all main open large models use RMSNorm. The list of adopters is at the end of this chapter.

### 3.4 Effect

Put RMSNorm into the 30-layer network of Section 1. Each layer normalizes first, then multiplies by the matrix: `h_l = ReLU(RMSNorm(h_{l-1}) · W_l)`.

| Initialization | Activation at layer 1 | Layer 10 | Layer 20 | Layer 30 | Error signal, layer 1 / layer 30 |
|---|---:|---:|---:|---:|---:|
| std = 1.0 | 9.33 | 10.1 | 9.7 | 9.92 | 1.33 |
| std = 0.01 | 0.0933 | 0.101 | 0.097 | 0.0992 | 1.33 |
| std = 0.02 | 0.187 | 0.201 | 0.194 | 0.198 | 1.33 |
| Kaiming | 0.825 | 0.89 | 0.858 | 0.877 | 1.33 |

With an initialization of 1.0 or of 0.01, the signal keeps the same magnitude over 30 layers. The error signal also no longer changes exponentially. Normalization replaces a fragile requirement, "the scale of each layer must be exactly correct", with "each layer corrects its scale automatically".

### 3.5 Where to put it: Pre-Norm

The position of the normalization is also important. The original Transformer puts LayerNorm **after** the residual addition (Post-Norm). Today, most models put it on the **input** of each sublayer (**Pre-Norm**), and add one more norm before the output:

```
Post-Norm: h ← Norm(h + f(h))
Pre-Norm:  h ← h + f(Norm(h)),   before the output: Norm(h)
```

Xiong et al. (2020) analyzed this. At initialization, Post-Norm has large gradients near the output layer, so it needs warmup to train. Pre-Norm has gradients of similar size in all layers. The Transformer block in Chapter 9 uses the Pre-Norm structure, and so does the experiment network in this chapter. There is an exception: OLMo 2 puts RMSNorm on the **output** of attention and of the MLP, and then adds the result back to the residual stream. The OLMo 2 report says that this is more stable together with QK-Norm.

## 4. Residual connections: a direct path for the signal

### 4.1 Intuition and formula

Normalization controls the scale, but deep networks have one more problem. A **residual connection** (He et al., 2016) adds only one plus sign:

```
Plain stack:         h ← f(h)
Residual connection: h ← h + f(h)
```

Each block only has to learn "a small change to the current value". In the backward pass, the chain rule gives:

```
∂h_{l+1}/∂h_l = I + ∂f/∂h_l
```

The `I` (the identity matrix) has this meaning: whatever `f` learns, the gradient always has a path on which it does not change. With 30 blocks, the gradient has at least one path from the output to block 1 that "always goes through I". This path has no matrix multiplication.

### 4.2 Experiment

`03_residual.py` builds 30 blocks. Each block is `f(h) = ReLU(RMSNorm(h) · W1) · W2`. The script compares three versions. In addition to the scale, we look at two metrics:

- **Cosine similarity**: how similar the random inputs in a batch are to each other, pair by pair. For the inputs, the value is 0.000 (they are orthogonal). A value near 1 means that the network "sees all inputs as the same thing".
- **Direction similarity of the error signal**: for the same sample, how similar the direction of the error signal at block 1 is to its direction at block 30.

| Version | Cosine similarity at block 30 | Residual-stream std at block 30 | Error signal, block 1 / block 30 (size) | Direction similarity |
|---|---:|---:|---:|---:|
| Plain stack | 0.953 | 0.987 | 1.250 | −0.002 |
| Residual (W2 not scaled) | 0.549 | 5.545 | 3.795 | 0.256 |
| **Residual (W2 × 1/√(2L))** | **0.110** | **1.223** | **1.170** | **0.824** |

The table shows three findings:

1. **A plain stack with RMSNorm has a normal scale, but it loses the information.** After 30 blocks, the pairwise similarity of the representations of all inputs is 0.953. The network almost cannot tell the inputs apart. The error signal that arrives back at block 1 has a correct size (1.25). But its direction has no relation to its direction at block 30 (−0.002), as if 30 random matrices mixed it.
2. **With residual connections, the information stays, and the error signal goes straight through.** With W2 made smaller by the number of layers, the similarity is only 0.110. The direction similarity between the error signals at block 1 and block 30 is 0.824. This is the "direct path".
3. **The residual stream becomes larger and larger, so the output projection must be smaller.** Each block adds one `f(h)` to the residual stream. After 30 blocks, the std increases to 5.5, which is about `sqrt(1 + 30)`. With W2 multiplied by `1/sqrt(2L)`, the std increases only to 1.22. This is the origin of the rule "make the residual projection smaller by the number of layers" in Section 2.3.

## 5. Optimizers: SGD → momentum → Adam → AdamW

The signal can now flow. The next question is how the parameters move. The SGD of Chapter 1 uses one learning rate for all parameters. The problem is that the "slope" of different parameters can differ by several orders of magnitude.

### 5.1 A bowl whose steepness differs by 1000×

`04_optimizers.py` uses a 4-dimensional bowl `L(θ) = ½ Σ hᵢ θᵢ²`. The curvatures in the four directions are `h = [100, 10, 1, 0.1]`, and the start point is `θ = [1, 1, 1, 1]`. Chapter 1 showed that the learning rate of SGD must be less than `2 / largest curvature = 0.02`. If it is not, the steepest direction diverges.

**SGD**: `θ ← θ − η · g`

**Momentum**: add up the gradients into a "velocity". In flat directions, the steps become faster and faster. In directions that oscillate, the positive and negative gradients cancel.

```
v ← β · v + g,   θ ← θ − η · v
```

**Adam** (Kingma & Ba, 2014) keeps two moving averages: `m` of the gradient (the direction) and `v` of the squared gradient (the scale). **Adam divides the step of each parameter by the gradient scale of that parameter**:

```
m ← β₁ m + (1 − β₁) g
v ← β₂ v + (1 − β₂) g²
m̂ = m / (1 − β₁ᵗ),  v̂ = v / (1 − β₂ᵗ)          (bias correction: m and v start from 0, so they are too small in the first steps)
θ ← θ − η · m̂ / (√v̂ + ε)
```

```python
m = b1 * m + (1 - b1) * g                   # first moment: direction
v = b2 * v + (1 - b2) * g * g               # second moment: scale
m_hat, v_hat = m / (1 - b1**t), v / (1 - b2**t)
step = -lr * m_hat / (np.sqrt(v_hat) + eps) # each parameter: divide by its own gradient scale
```

The output is:

| | Step 1 \|Δθ₁\| | \|Δθ₂\| | \|Δθ₃\| | \|Δθ₄\| | Loss after 200 steps | Steps until loss < 0.001 |
|---|---:|---:|---:|---:|---:|---:|
| SGD (η = 0.019) | 1.9000 | 0.1900 | 0.0190 | 0.0019 | 2.36 × 10⁻² | More than 200 |
| Momentum (η = 0.019, β = 0.9) | 1.9000 | 0.1900 | 0.0190 | 0.0019 | 4.91 × 10⁻⁶ | 109 |
| Adam (η = 0.05) | 0.0500 | 0.0500 | 0.0500 | 0.0500 | 2.94 × 10⁻⁷ | 48 |

In its first step, SGD moves 1.9 in the steepest direction, but only 0.0019 in the flattest direction. The difference is 1000×. The steepest direction limits the learning rate, so after 200 steps the flattest direction is still at 0.68. Momentum makes the steps in the flat direction faster and faster, and it converges in 109 steps.

In its first step, Adam moves each parameter by exactly η = 0.05. The reason: in the first step, `m̂ / √v̂` equals `g / |g| = ±1`. The division removes the size of the gradient, and only the direction stays. Adam converges in 48 steps.

The Transformer needs exactly this property: "each parameter sets its own step size". Zhang et al. (2024) found that the parameter blocks of a Transformer (the embedding layer, the Q/K/V of attention, the MLP) have very different curvatures. SGD with one learning rate cannot handle this, but Adam can. Conversely, SGD can be as good as Adam on a network whose layers all have a similar structure. The small MLP in Section 8 is such a case.

### 5.2 AdamW: "decouple" the weight decay

**Weight decay** moves the parameters a little toward 0 at each step, so that the weights cannot grow without limit. The traditional method adds `½ λ ‖θ‖²` to the loss, which adds `λθ` to the gradient. This is L2 regularization. For SGD, the two descriptions are equivalent.

For Adam, they are not equivalent. Adam divides the `λθ` in the gradient by `√v`, together with the true gradient. **A parameter with a large gradient has its decay divided by a large number, so it gets almost no decay. A parameter with a small gradient gets a larger decay.** AdamW (Loshchilov & Hutter, 2017) takes the decay out of the gradient and applies it separately:

```
θ ← θ − η · λ · θ               (decoupled weight decay: does not go through √v)
θ ← θ − η · m̂ / (√v̂ + ε)        (the normal Adam update)
```

The second experiment in `04_optimizers.py` uses two groups of parameters. All parameters start at 1. The loss gives no signal, and the gradient is only noise. Group A has a noise std of 0.01, and group B has a noise std of 10. With λ = 0.1 and η = 0.01, the script runs 3000 steps. Ideally, weight decay pulls both groups equally to `(1 − ηλ)^3000 = 0.050`:

| | Group A (small gradients) | Group B (large gradients) |
|---|---:|---:|
| Adam + L2 regularization | 0.000 | 0.753 |
| AdamW (decoupled) | 0.047 | 0.056 |

With Adam + L2 and the same λ, group A goes to 0, and group B almost does not move. With AdamW, both groups are near the theoretical value 0.050. **With AdamW, only λ sets the amount of decay, and the size of the gradient has no effect.** This is why almost all open large models use AdamW for pretraining today. The common hyperparameters are β₁ = 0.9, β₂ = 0.95, and λ = 0.1 (the reports of Llama 2, DeepSeek-V3, and SmolLM3 all use these values).

### 5.3 Which parameters do not get weight decay

In practice, we usually put the parameters into two groups: **matrices get weight decay, and 1-dimensional parameters (the γ of RMSNorm, biases) do not.** The γ sets the scale of the normalized signal, and its default value is 1. There is no reason to pull it toward 0. The `configure_optimizers` function of nanoGPT makes the groups with `p.dim() >= 2`.

OLMo 2 goes one step further and also excludes the **token embeddings** from weight decay. They found that weight decay makes the norm of the embeddings smaller and smaller. Then the gradients of the early layers become larger, and more spikes occur. The SmolLM3 report says that SmolLM3 uses the same method. The code in this chapter applies weight decay only to the matrices:

```python
if p.dim() >= 2:                   # decay only the matrices, not γ
    p.mul_(1 - lr * cfg["wd"])     # decoupled weight decay
p -= lr * m_hat / (v_hat.sqrt() + 1e-8)
```

## 6. Learning-rate schedules: warmup + cosine, and WSD

Chapter 1 showed that training diverges when the learning rate is above the critical value. When we train a large model, the learning rate is not one fixed number. It is a curve that changes with the step.

### 6.1 Why warmup at the start

At the start of training, the parameters are random. Adam's `v` has seen only a few gradients, so its estimate is not accurate. If we use the peak learning rate at this time, each parameter moves a full η in the first step (as in the previous section). The whole network gets a strong push, and a loss spike (a sudden increase of the loss) often occurs. **Warmup** increases the learning rate linearly from almost 0 to the peak during the first several hundred to several thousand steps.

### 6.2 Why decay at the end

When the learning rate is large, the parameters jump from side to side near the bottom of the valley, and the loss does not decrease. If we decrease the learning rate slowly, the parameters can settle at the bottom. The most common method is **cosine decay** (SGDR, by Loshchilov & Hutter). After warmup, the learning rate follows a cosine curve from the peak down to 10% of the peak. Llama 2, Llama 3, and OLMo 2 all use "linear warmup + cosine decay".

```python
def warmup_cosine(step, total, peak, warmup, min_ratio=0.1):
    if step < warmup:
        return peak * (step + 1) / warmup                  # warmup: 0 → peak
    p = (step - warmup) / max(1, total - warmup)           # decay progress 0 → 1
    return peak * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * p)))
```

### 6.3 WSD: stay constant, then decay at the end

Cosine decay has one problem: the shape of the curve depends on the total number of steps. Thus **you must decide the length of training before you start**. If you want to train longer in the middle of the run, you must start again.

MiniCPM (Hu et al., 2024) explicitly proposed **WSD (Warmup-Stable-Decay)**. After warmup, the learning rate **stays constant** (stable). Only in a short final part (often 10%–20%) does it decrease fast (decay). You can add a decay to any checkpoint of the stable part and get a "finished" model. If you want to train more, continue from the stable part.

```python
def wsd(step, total, peak, warmup, decay_frac=0.2, min_ratio=0.0):
    decay_start = int(total * (1 - decay_frac))
    if step < warmup:
        return peak * (step + 1) / warmup                  # W: warmup
    if step < decay_start:
        return peak                                        # S: stable, constant
    p = (step - decay_start) / max(1, total - decay_start)
    return peak * (1 - (1 - min_ratio) * p)                # D: decay, linear
```

`05_lr_schedule.py` prints the two curves (1000 total steps, peak 3 × 10⁻³, warmup 100 steps):

| Step | 0 | 50 | 100 | 300 | 500 | 700 | 800 | 900 | 999 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| warmup + cosine | 3.0e−5 | 1.5e−3 | 3.0e−3 | 2.7e−3 | 1.9e−3 | 9.8e−4 | 6.2e−4 | 3.8e−4 | 3.0e−4 |
| WSD | 3.0e−5 | 1.5e−3 | 3.0e−3 | 3.0e−3 | 3.0e−3 | 3.0e−3 | 3.0e−3 | 1.5e−3 | 1.5e−5 |

Many open models now use this shape: "constant for a long time, then decay at the end". SmolLM3 explicitly says WSD (2000 warmup steps, then a linear decay to 0 in the last 10%). The Kimi K2 report explicitly says WSD (a constant 2e−4 for the first 10T tokens, then a cosine decay to 2e−5 over 5.5T tokens). DeepSeek-V3 does not use the name WSD, but its schedule has the same shape (2K warmup steps, constant until 10T tokens, then a cosine decay to 2.2e−5 over 4.3T tokens).

> **Note:** The numbers of the training experiments in this chapter come from one CPU run on the course build machine. Different machines and different versions of the low-level math libraries do the floating-point operations in a slightly different order. After a few hundred training steps, training makes these small differences larger. Your numbers can be different from the second or third decimal place. Trust the conclusions below that do not depend on exact values. For a second run on another server in 2026-10, see [runs/2026-10-01-gpu0-check/chapters-01-06.md](../../runs/2026-10-01-gpu0-check/chapters-01-06.md).

`07_schedule_experiments.py` compares three schedules on the network of Section 8 (800 steps; a lower validation loss is better):

| Schedule | Validation loss |
|---|---:|
| warmup + cosine | 0.775 |
| WSD (linear decay to 0 in the last 20%) | 0.769 |
| Constant learning rate (warmup only) | 0.806 |

The two schedules that decay at the end give almost the same result. The schedule without decay is noticeably worse. Now look at how WSD can "finish at any time". Run one trunk with a constant learning rate. At step 400 and at step 800, branch off a 100-step decay. Compare the result with cosine runs whose total length is set before the start:

| Total length | End of stable | After 100 decay steps | A separate cosine run |
|---:|---:|---:|---:|
| 500 steps | 0.887 | 0.801 | 0.796 |
| 900 steps | 0.806 | 0.755 | 0.760 |

In the 100 decay steps, the validation loss decreases by 0.05–0.09, to the level of the separate cosine run. WSD trains only 400 + 100 + 400 + 100 = 1000 steps in total, but the two cosine runs need 500 + 900 = 1400 steps. Hägele et al. (2024) did the same comparison on 1B and 8B models and got the same result. A constant learning rate plus a short decay is as good as a tuned cosine schedule, and it saves much compute in scaling experiments. The ladder experiment in Chapter 12 and the "annealing" in Chapter 15 both use this property.

## 7. Gradient clipping: a maximum for the step size

Even when all settings are correct, training sometimes gets a batch of "bad data". An example is the long sequences of repeated characters in the OLMo 2 report. Then the gradient suddenly becomes very large. **Gradient clipping** (Pascanu et al., 2013) puts the gradients of all parameters into one long vector. If the norm of this vector is more than a maximum (almost all large models use 1.0), clipping scales the whole vector down by the same factor. The direction does not change.

```
‖g‖ = sqrt(Σ all gradient components²),   g ← g · min(1, max_norm / ‖g‖)
```

```python
def clip_by_global_norm(grads, max_norm=1.0):
    total = math.sqrt(sum(float((g * g).sum()) for g in grads))
    scale = min(1.0, max_norm / (total + 1e-6))
    for g in grads:
        g *= scale
    return total
```

`07_schedule_experiments.py` adds 5 batches of bad data (inputs multiplied by 30) at steps 300–304:

| | Max gradient norm at the bad data | Max training loss in the next 40 steps | Final validation loss |
|---|---:|---:|---:|
| RMSNorm on, clipping on | 1.1 | 1.096 | 0.776 |
| RMSNorm on, clipping off | 1.1 | 1.096 | 0.776 |
| RMSNorm off, clipping on | 95.9 | 1.118 | 0.747 |
| RMSNorm off, clipping off | 105.9 | 1.977 | 0.773 |

With RMSNorm, the first norm cancels the 30× input, so the gradient norm is only 1.1. Clipping makes no difference. Without RMSNorm, the gradient norm increases to about 100. Without clipping, the training loss after the bad data jumps to 1.977; with clipping, it goes only to 1.118.

**Clipping is the last safety device.** In normal training, it has almost no effect. When a problem occurs, it limits the size of that step. It has almost no cost, so it is standard in the training of large models.

## 8. Comparison experiment: one network, turn on the methods one at a time

Now put all the parts together. The task of `06_ablation.py` is as follows. The input is a random 16-dimensional vector, and a fixed "teacher network" gives the label (10 classes). Each step samples 128 new samples, so the data never runs out, as in pretraining. We keep 2048 more samples to calculate the validation loss. The cross-entropy of a random guess is about ln 10 = 2.30.

The student network has 12 blocks (24 linear layers) with a width of 32, and it trains for 800 steps. All methods are written by hand, and you can turn each one on or off.

```bash
uv run python chapters/06-training-stability/code/06_ablation.py      # about 1.5 min
```

**Add the methods one at a time** (each SGD row runs once with each learning rate, 0.01 / 0.03 / 0.1, and uses the best one):

| Configuration | Learning rate | Validation loss | Accuracy | Result | Validation loss for the 3 learning rates |
|---|---:|---:|---:|---|---|
| A Plain deep network (std = 1 initialization + SGD) | — | nan | — | Diverges | All nan |
| B + Kaiming initialization | 0.1 | 1.931 | 0.279 | Learns slowly | 2.301 / 2.095 / 1.931 |
| C + residual connections | 0.03 | 0.821 | 0.705 | Success | 0.823 / 0.821 / nan |
| D + RMSNorm (Pre-Norm) | 0.1 | 0.811 | 0.700 | Success | 0.838 / 0.814 / 0.811 |
| E SGD → AdamW (constant learning rate) | 0.003 | 0.807 | 0.697 | Success | |
| F + warmup + cosine decay | 0.003 | 0.775 | 0.711 | Success | |
| G + gradient clipping (= full set) | 0.003 | 0.775 | 0.708 | Success | |

**Remove one method at a time from the full set**:

| Configuration | Validation loss | Accuracy | Result |
|---|---:|---:|---|
| Full set, but no residual connections | 1.462 | 0.515 | Learns slowly |
| Full set, but no RMSNorm | 0.747 | 0.726 | Success |
| Full set, but std = 1 initialization | 0.825 | 0.697 | Success |
| Full set, but constant learning rate (no warmup / decay) | 0.803 | 0.704 | Success |

(The "Result" column is a rough grouping by validation loss: nan means "diverges", more than 2.0 means "did not learn", 1.0–2.0 means "learns slowly", and less than 1.0 means "success". Row B, with 1.931, is near the boundary at 2.0. A second run of the same code on another server in 2026-10 gave 2.035 (accuracy 0.168). With the same rule, that result is "did not learn". The two runs show the same thing: with only a better initialization, a plain stack of 24 layers almost cannot learn.)

Read these two tables together with the sections above:

- **Initialization decides if training can start.** In A, the initial loss is already 6 × 10¹⁵. With all three learning rates, the loss becomes NaN at step 2. With Kaiming (B), training no longer diverges. But a plain stack of 24 layers almost cannot learn: the best result is only near 2.0, not far from the 2.30 of a random guess.
- **Residual connections decide if a deep network can learn.** With residual connections (C), the validation loss decreases from about 2.0 to 0.82. Conversely, if we remove only the residual connections from the full set, the loss goes back to 1.462 (the second run gave 1.310). This is still in the "learns slowly" group. Residual connections are the switch with the largest effect in this experiment. This agrees with the finding of Section 4: "a plain stack cannot tell the inputs apart".
- **Normalization gives robustness, but not always a lower loss.** C diverges at a learning rate of 0.1, but D (with RMSNorm) trains well at 0.1. The full set with the "bad" std = 1 initialization also trains to 0.825. But on this small network, when the initialization and the learning rate are both tuned, the loss without RMSNorm is a little lower (0.747). The stress test below shows the value of RMSNorm.
- **AdamW is as good as the tuned SGD here.** E (0.807) and D (0.811) are almost the same. All layers of this MLP have the same structure, which is the case of Section 5.1: "SGD can be as good as Adam". Adam has two advantages. You do not need to tune the learning rate for each case. Also, Adam works on structures like the Transformer, whose parameter blocks are very different.
- **Learning-rate decay gives a consistent improvement.** F is 0.03 better than E. The full set with a constant learning rate also goes back to 0.803.
- **Gradient clipping has no effect in normal training** (F and G are the same). It is for unexpected events, like the one in Section 7.

**High-learning-rate stress test**: multiply the learning rate by 100, from 0.003 to 0.3 (`07_schedule_experiments.py`):

| Configuration | Max training loss in the first 150 steps | Validation loss after 800 steps |
|---|---:|---:|
| Full set | 2.723 | 0.892 |
| No warmup | 6.178 | 0.890 |
| No RMSNorm | 1.1 × 10¹⁰ | 0.991 |

Here we see the effect of warmup and RMSNorm. Without warmup, the loss at the start increases to 6.2. Without RMSNorm, the loss spike goes above 10¹⁰. (The second run gave 2.4 × 10¹². When a spike is this high, its exact magnitude is very sensitive to rounding.) The training recovers at the end, but the validation loss is worse.

Our toy model has only tens of thousands of parameters, and it can recover from these spikes by itself. A large model has billions of parameters and trains for months. For such a model, one spike can mean a rollback to a checkpoint from several days before. Thus large models use all of these safety devices.

## 9. Summary

| Method | Problem that it solves | Evidence in this chapter |
|---|---|---|
| Initialization scaled by fan_in | Repeated multiplication makes the signal explode or vanish exponentially | After 30 layers, the activations change from 10³¹ / 10⁻²⁹ to 0.56 |
| RMSNorm (Pre-Norm) | The scale drifts during training; training is sensitive to the initialization and the learning rate | The signal is stable over 30 layers with any initialization; at η = 0.3, the spike decreases from more than 10¹⁰ to 2.7 |
| Residual connections | Deep layers "cannot tell the inputs apart", and the gradient is mixed | Similarity 0.953 → 0.110; validation loss about 2.0 → 0.82 |
| AdamW | Parameters have very different scales; the size of the gradient distorts the amount of decay | 48 steps vs more than 200 steps for SGD; decay of the two groups 0.047 / 0.056 |
| warmup + cosine / WSD | A strong push at the start causes spikes; at the end, the parameters do not settle at the bottom of the valley | Without warmup, the spike is 6.2; the decay gives an improvement of 0.03–0.09 |
| Gradient clipping | Rare bad data gives very large gradients | Loss spike after the bad data 1.977 → 1.118 |

---

## From minimal code to production code

For Chapters 1–6, the production code is the standard PyTorch code. The same "full set" configuration is in [`code/08_pytorch_version.py`](code/08_pytorch_version.py):

```python
class Block(nn.Module):
    def __init__(self, width, hidden):
        super().__init__()
        self.norm = nn.RMSNorm(width, eps=1e-6)
        self.fc1 = nn.Linear(width, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, width, bias=False)    # output projection of the residual branch

    def forward(self, x):
        return x + self.fc2(F.relu(self.fc1(self.norm(x))))

decay = [p for p in model.parameters() if p.dim() >= 2]
no_decay = [p for p in model.parameters() if p.dim() < 2]
opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1},
                         {"params": no_decay, "weight_decay": 0.0}],
                        lr=3e-3, betas=(0.9, 0.95), eps=1e-8)
scheduler = torch.optim.lr_scheduler.LambdaLR(
    opt, lambda s: warmup_cosine(s, total, 1.0, warmup))   # multiplier function, peak = 1

for step in range(total):
    loss = F.cross_entropy(model(x), y)
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    opt.step()
    scheduler.step()
```

| Minimal code (hand-written) | Standard PyTorch code | Notes |
|---|---|---|
| `rms_norm(x, gamma)` | `nn.RMSNorm(width, eps=1e-6)` | The same formula; part of PyTorch since 2.4. `nn.LayerNorm` also exists |
| `torch.randn(...) * std` | `nn.init.normal_(p, std=0.02)`, and `0.02/√(2L)` for the output projections | `init_llm_style` in the script is the usual method in large models |
| Hand-written m, v, bias correction, decoupled decay | `torch.optim.AdamW` | The update formula is the same; on GPUs, there are also fused kernels |
| Decay only `if p.dim() >= 2` | Two parameter groups, with `weight_decay` 0.1 and 0 | The optimizer manages the hyperparameters by group |
| `warmup_cosine` / `wsd` functions | `LambdaLR(opt, multiplier function)`, and `scheduler.step()` at each step | Save the scheduler state together with the checkpoint (Chapter 14) |
| `clip_by_global_norm` | `torch.nn.utils.clip_grad_norm_` | Returns the norm before clipping; training logs often record it |

**Parity check**: the script copies the initial weights of the hand-written version (06) into the PyTorch version. It runs 200 steps on the same data stream and compares the loss at each step:

| Precision | Max loss difference per step | Loss at step 200 (PyTorch / hand-written) |
|---|---:|---|
| float64 | 8.9 × 10⁻¹⁶ | 0.9041 / 0.9041 |
| float32 | 1.3 × 10⁻² | 0.8676 / 0.8659 |

In double precision, the two versions agree digit for digit, so the formulas are identical. In single precision, the two sides do the operations in a slightly different order. Each step adds a rounding difference of about 10⁻⁷, and training makes the difference larger step by step. After 200 steps, the difference is 10⁻². (The second run gave 3 × 10⁻⁴. The final size depends on the machine and on the kernels that PyTorch selects.)

This is also a lesson in training stability. Training is a dynamical system that makes small disturbances larger. Thus, do parity checks in high precision. Also, "a fixed random seed makes the results reproducible" is true only on the same machine with the same code. (The two precisions use different random data, so the loss values in the two rows are different.)

At the end, the script trains the full set for 800 steps with the large-model-style initialization, std = 0.02. The validation losses of cosine and WSD are both about 0.80 (0.80028 and 0.80029 on the build machine; 0.80101 and 0.79763 in the second run). The difference is within the random variation between machines, so we cannot say which schedule is better.

**In the main-line code**: from Chapter 7, the production code is in `zero/`. The methods of this chapter are at these locations in the main line. Chapter 14 explains them in detail:

- `zero/model.py`: `RMSNorm` (the same as the Hugging Face Qwen3 implementation: it converts to float32 before it normalizes). `Transformer.init_weights()` initializes with `N(0, init_std)`, and it divides the two projections that write back to the residual stream by `sqrt(2 · n_layers)`.
- `zero/train/schedule.py`: `cosine_with_warmup`, `wsd` (the decay shape can be linear, cosine, or 1−sqrt), and `LRScheduler` (it writes the multiplier into the optimizer and saves its state in the checkpoint).
- `zero/train/trainer.py`: `build_optimizer` makes two groups, "matrices / 1-dimensional parameters". A setting controls if the embeddings get weight decay (the OLMo 2 method). In `Trainer.train`, each step calls `clip_grad_norm_` and writes the gradient norm to the log.

---

## Frontier notes

**The Muon optimizer**: Muon orthogonalizes the update of each matrix (it makes all singular values of the update matrix equal). The Kimi K2 report cites experiments from its earlier work, Moonlight: with the same compute and model size, Muon is much better than AdamW. Kimi K2 used an improved version, MuonClip, to pretrain on 15.5T tokens. GOAL.md lists Muon as "to be verified (possibly already a consensus)".

This chapter verified the main open models: Llama, Qwen3, DeepSeek-V3, OLMo 2, and SmolLM3. When this chapter was written, their public reports still used AdamW. Thus the main text of this chapter covers only AdamW. Chapter 12 will verify if Muon meets the rule "adopted by 3 independent families".

---

## Adopters and sources

All entries below are explicit statements in technical reports, model cards, or official `config.json` files.

| Technique | Adopters (at least 3 families) | Sources |
|---|---|---|
| RMSNorm | Llama 2 ("pre-normalization using RMSNorm"); Qwen3 ("RMSNorm with pre-normalization"); gpt-oss ("root mean square normalization"); OLMo 2 (changed from non-parametric LayerNorm to RMSNorm); the configs of DeepSeek-V3 and Gemma 3 have `rms_norm_eps` | [Llama 2 §2.2](https://arxiv.org/abs/2307.09288), [Qwen3 §2](https://arxiv.org/abs/2505.09388), [gpt-oss model card §2.2](https://arxiv.org/abs/2508.10925), [OLMo 2 §2.1](https://arxiv.org/abs/2501.00656), [DeepSeek-V3 config](https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/config.json), [Gemma 3 1B config](https://huggingface.co/google/gemma-3-1b-it/blob/main/config.json) |
| Pre-Norm | Llama 2; Qwen3; gpt-oss ("Pre-LN placement"). Counterexample: OLMo 2 puts the norm on the sublayer output | Same as above; [Xiong et al. 2020](https://arxiv.org/abs/2002.04745) |
| AdamW | Llama 2 (β = 0.9/0.95, λ = 0.1); Llama 3; DeepSeek-V3 (β = 0.9/0.95, λ = 0.1); OLMo 2; SmolLM3 (β = 0.9/0.95, λ = 0.1). Counterexample: Kimi K2 uses MuonClip | [Llama 2](https://arxiv.org/abs/2307.09288), [Llama 3 §3.4.1](https://arxiv.org/abs/2407.21783), [DeepSeek-V3 §4.2](https://arxiv.org/abs/2412.19437), [OLMo 2 §3.4](https://arxiv.org/abs/2501.00656), [SmolLM3 blog](https://github.com/huggingface/blog/blob/main/smollm3.md), [Kimi K2 §2.1](https://arxiv.org/abs/2507.20534) |
| No weight decay for embeddings / 1-dimensional parameters | OLMo 2 (no decay for embeddings); SmolLM3 (uses the OLMo 2 method); nanoGPT (no decay for 1-dimensional parameters) | [OLMo 2 §3.4.2](https://arxiv.org/abs/2501.00656), [SmolLM3 blog](https://github.com/huggingface/blog/blob/main/smollm3.md), [nanoGPT model.py](https://github.com/karpathy/nanoGPT/blob/master/model.py) |
| warmup + cosine decay | Llama 2 (2000 warmup steps, decay to 10% of the peak); Llama 3 (8000 warmup steps, decay to 8e−7); OLMo 2 (2000 warmup steps, decay to 10%) | [Llama 2 §2.2](https://arxiv.org/abs/2307.09288), [Llama 3 §3.4.1](https://arxiv.org/abs/2407.21783), [OLMo 2 §2.3](https://arxiv.org/abs/2501.00656) |
| WSD (warmup-stable-decay) | MiniCPM (proposed it); SmolLM3 (linear decay to 0 in the last 10%); Kimi K2 (the report explicitly says WSD); DeepSeek-V3 (the same shape: constant until 10T tokens, then cosine decay; it does not use the name WSD) | [MiniCPM §4](https://arxiv.org/abs/2404.06395), [SmolLM3 blog](https://github.com/huggingface/blog/blob/main/smollm3.md), [Kimi K2 §2.5](https://arxiv.org/abs/2507.20534), [DeepSeek-V3 §4.2](https://arxiv.org/abs/2412.19437) |
| Gradient clipping (norm 1.0) | Llama 2; DeepSeek-V3; OLMo 2; SmolLM3 | [Llama 2](https://arxiv.org/abs/2307.09288), [DeepSeek-V3 §4.2](https://arxiv.org/abs/2412.19437), [OLMo 2 Table 3](https://arxiv.org/abs/2501.00656), [SmolLM3 blog](https://github.com/huggingface/blog/blob/main/smollm3.md) |
| Initialization std 0.02 | The configs of Qwen3, Gemma 3, and SmolLM3 have `initializer_range: 0.02`; the OLMo 2 report explicitly says 0.02. Note: the DeepSeek-V3 config says 0.02, but the main text of its report says 0.006 | [Qwen3-0.6B config](https://huggingface.co/Qwen/Qwen3-0.6B/blob/main/config.json), [Gemma 3 1B config](https://huggingface.co/google/gemma-3-1b-it/blob/main/config.json), [SmolLM3-3B config](https://huggingface.co/HuggingFaceTB/SmolLM3-3B/blob/main/config.json), [OLMo 2 §3.2](https://arxiv.org/abs/2501.00656) |

To be verified: the Qwen3 technical report says only that "the second stage accelerates the learning-rate decay". It does not give the exact shape of the schedule. Thus Qwen3 is not in the adopter lists for warmup + cosine or WSD. This chapter did not verify the norm position of Gemma 3.

---

## Guided questions

Ask Claude Code these questions. Continue until you can explain the answers in your own words:

1. In the table of 01, look at the ratio "error signal at layer 1 / at layer 30" for std = 1.0 and for std = 0.01. The orders of magnitude of the two ratios are exactly reciprocal. Which property of a ReLU network makes the whole table scale proportionally when you change only the weight scale?
2. The "2" in Kaiming initialization comes from ReLU. If the activation function is tanh, what must this factor be? Why? (Hint: near 0, the slope of tanh is 1.)
3. RMSNorm does not subtract the mean. For which inputs are the outputs of RMSNorm and LayerNorm most different? Make an example and let `02_normalization.py` print it.
4. In the experiment of 03, the plain stack has a normal scale and an error signal of normal size, but it still "cannot learn". Is "a gradient of normal size" the same as "a useful gradient"?
5. Why does Adam move each parameter by exactly η in the first step? How is this related to warmup?
6. The decay phase of WSD is only 10%–20% of the steps, but it decreases the loss by a large amount. Explain why. Use the picture from Chapter 1: "when the learning rate is too large, the parameters jump from side to side near the bottom of the valley".

## Hands-on tasks

For each task, run the code and look at the result.

**Task 1 (basic)**: In `01_signal_propagation.py`, change the activation function from ReLU to tanh. Run the five initializations again. Which one now comes nearest to "each layer does not make the signal larger or smaller"? Does the result agree with your answer to guided question 2?

**Task 2 (core)**: In `06_ablation.py`, change `BLOCKS` from 12 to 24 (48 linear layers). Run the two tables again. For which configurations does the result change? Do residual connections and RMSNorm become more important or less important? (The run time doubles, to about 3 min.)

**Task 3 (challenge)**: The WSD in `05_lr_schedule.py` uses a linear decay. Hägele et al. (2024) found that a decay with the shape `1 − sqrt(progress)` is better. Add a `shape` parameter to the `wsd` function to implement it. Then use the experiment of `07_schedule_experiments.py` to compare the validation loss of the "linear" and "1 − sqrt" decays. Is the difference as large as the paper says? If not, what can be the reason?

---

## After Part 1, you can start CS336

This is the end of Part 1, "Start from a straight line". We started from `y = ax + b` and went through matrices, neural networks, backpropagation, softmax, and cross-entropy. In this chapter, we made a deep network train in a stable way. Stanford [CS336: Language Modeling from Scratch](https://cs336.stanford.edu/) (Spring 2026) expects exactly this knowledge before you start. The first lecture of CS336 starts with tokenization. Assignment 1 asks you to write BPE, a Transformer, AdamW, and a training loop from zero.

The RMSNorm, AdamW, cosine schedule, and gradient clipping that you wrote by hand in this chapter occur there again in the same form. From Chapter 7, each chapter of this course ends with a section "Go deeper: CS336", which points to the related lectures and assignments. You can continue with this course. You can also open the CS336 course page now and study the two together.

---

## References

- Glorot & Bengio. *Understanding the difficulty of training deep feedforward neural networks* (Xavier initialization), AISTATS 2010: <https://proceedings.mlr.press/v9/glorot10a.html>
- He et al. *Delving Deep into Rectifiers* (Kaiming initialization), 2015: <https://arxiv.org/abs/1502.01852>
- He et al. *Deep Residual Learning for Image Recognition* (residual connections), 2015: <https://arxiv.org/abs/1512.03385>
- Ba, Kiros, Hinton. *Layer Normalization*, 2016: <https://arxiv.org/abs/1607.06450>
- Zhang & Sennrich. *Root Mean Square Layer Normalization*, NeurIPS 2019: <https://arxiv.org/abs/1910.07467>
- Xiong et al. *On Layer Normalization in the Transformer Architecture* (Pre-LN and warmup), ICML 2020: <https://arxiv.org/abs/2002.04745>
- Kingma & Ba. *Adam: A Method for Stochastic Optimization*, 2014: <https://arxiv.org/abs/1412.6980>
- Loshchilov & Hutter. *Decoupled Weight Decay Regularization* (AdamW), 2017: <https://arxiv.org/abs/1711.05101>
- Loshchilov & Hutter. *SGDR: Stochastic Gradient Descent with Warm Restarts* (cosine schedule), 2016: <https://arxiv.org/abs/1608.03983>
- Pascanu, Mikolov, Bengio. *On the difficulty of training recurrent neural networks* (gradient clipping), 2012: <https://arxiv.org/abs/1211.5063>
- Zhang et al. *Why Transformers Need Adam: A Hessian Perspective*, NeurIPS 2024: <https://arxiv.org/abs/2402.16788>
- Hu et al. *MiniCPM: Unveiling the Potential of Small Language Models with Scalable Training Strategies* (WSD), 2024: <https://arxiv.org/abs/2404.06395>
- Hägele et al. *Scaling Laws and Compute-Optimal Training Beyond Fixed Training Durations* (constant learning rate + cooldown), NeurIPS 2024: <https://arxiv.org/abs/2405.18392>
- OLMo Team. *2 OLMo 2 Furious* (stability details: initialization, norm position, no weight decay for embeddings, and more), 2024: <https://arxiv.org/abs/2501.00656>
- Karpathy. nanoGPT (parameter groups in `configure_optimizers`, scaled initialization of the residual projections, cosine schedule in `get_lr`): <https://github.com/karpathy/nanoGPT>
- Technical reports and configs of each model: see the "Adopters and sources" table above
- CS336 (Spring 2026): <https://cs336.stanford.edu/>

**Next chapter**: Part 1 ends here. We now have a set of tools that make a deep network train in a stable way. Part 2 starts to build a real language model. The first question is: a model knows only numbers, so how do we change text into numbers? By character, by word, or by byte? In Chapter 7, we start with language modeling and tokenization, and we write a BPE tokenizer by hand.
