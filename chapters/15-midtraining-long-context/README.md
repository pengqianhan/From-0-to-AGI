# Chapter 15: Mid-training and long context — How to train the last stage, and how to read longer text

**English** · [中文](README.zh.md)

> **Goal**: After this chapter, you can explain why "annealing" changes to high-quality data in the decay phase of WSD. You can use "branched decay" to compare two data mixtures at a low cost. You can calculate the wavelength of each dimension pair for any RoPE setting, and explain why a model cannot read text that is longer than its training length. You can write the frequency interpolation of YaRN by hand and do a parity check against the official implementation. You know how the main-line model extends from 4K to 32K, and what Gate 2 must check.

📺 **Video**: Not published yet. To render it on your computer, run `bash chapters/15-midtraining-long-context/video/build.sh`.
🧪 **Self-check**: After the chapter, type `/ch15-midtraining` in Claude Code.

---

In the last chapter, we prepared the pretraining of the main-line model: thousands of GPU-hours and hundreds of billions of tokens. The learning rate follows WSD through warmup and the stable phase. (In `configs/main/pretrain.toml`, `decay_frac = 0`: we keep the decay phase for later on purpose.) This chapter solves two problems at the end of pretraining:

1. **How do we train the last stage?** In the short stage where the learning rate decreases, the training has a surprisingly large effect on the final model. Almost all leading open models change to "the best data" in this stage. This is **annealing** or **mid-training**.
2. **What do we do when the model cannot read long text?** Pretraining uses sequences of length 4K. (The compute of attention increases with the square of the length, so 32K for all of training costs too much.) But the main-line model must call tools. Tool descriptions and a multi-turn conversation often have more than 10,000 tokens. A model that saw only 4K fails when it reads 32K directly. The solution is to change the "rotation speed" of RoPE and then train a little more: a **larger RoPE base frequency** and **YaRN**.

After these two steps, the main-line base model is ready. Then comes **Gate 2**: compare its real evaluation results with the predictions of Gate 1. If the results are clearly lower, diagnose the cause first. Do not hurry into post-training.

The code of this chapter (all of it runs on a CPU):

```bash
uv run python chapters/15-midtraining-long-context/code/01_rope_wavelengths.py   # wavelength table, unseen angles, compute cost of 32K (1 s)
uv run python chapters/15-midtraining-long-context/code/02_yarn_from_scratch.py  # PI / YaRN from scratch, parity check with zero and HF (a few seconds)
uv run python chapters/15-midtraining-long-context/code/03_context_extension.py  # train at length 64 → read 128/256 (about 11 min of CPU time on one thread)
uv run python chapters/15-midtraining-long-context/code/04_anneal_mixture.py     # branched decay: decay × new data (about 8 min of CPU time on one thread)
```

## 1. Mid-training: give the best data in the stage with the lowest learning rate

### 1.1 Intuition: what the model learns last, it remembers best

Chapter 6 did an experiment. From a trunk with a constant learning rate, it branched off a short decay at different points. Each time, the validation loss immediately decreased by a large step. It became equal to a cosine schedule with a total length that was set in advance. MiniCPM (the paper that introduced WSD) saw the same effect on a 0.036B model: **the decay phase is only about 10% of the steps, but the loss decreases sharply**.

Use the picture from Chapter 1 to understand the cause. When the learning rate is large, the parameters jump from side to side near the bottom of the valley. The model learns the "general direction" quickly, but it cannot settle on the details. Only when the learning rate decreases do the parameters slowly go down to the bottom of the valley. In other words, **the decay phase sets where the model stops at the end**. And where the model stops depends on the data that it sees in this phase.

This gives a natural idea. The last stage has a large effect, so keep the best data for this stage: the data that we most want the model to learn. This is mid-training:

```
Pretraining (stable phase): mostly a very large amount of web pages, constant learning rate          ≈ 90–95% of the compute
Mid-training (decay phase): change to high-quality data + math/code + instruction-style data, learning rate decays to 0   ≈ 5–10%
```

The name "mid" means that this stage is between pretraining and post-training (SFT, RL). The objective is still general next-token prediction. But the data starts to move toward "the skills that we want".

### 1.2 Who does this

| Model | Method | Source |
|---|---|---|
| **OLMo 2** | Explicitly calls it mid-training: 5–10% of the total compute. The learning rate decays linearly to 0. The data changes to Dolmino Mix: about half is high-quality filtered web pages, plus FLAN instruction data, academic papers, Wikipedia, StackExchange Q&A, and synthetic math. The 7B model does three runs with 50B tokens each (different data orders), then averages the weights | [OLMo 2 §2.3, §4](https://arxiv.org/abs/2501.00656) |
| **Llama 3** | Annealing at the end of pretraining: the learning rate decays linearly to 0, and the training upsamples data of "very high quality". Llama 3 also uses annealing to **evaluate data**: it anneals a half-trained 8B model on 40B tokens, with the new data set at 30% | [Llama 3 §3.1.3, §3.4.3](https://arxiv.org/abs/2407.21783) |
| **SmolLM3** | In the decay phase of WSD (10T → 11.1T tokens), it upsamples math and code more, and adds instruction and reasoning data such as OpenMathReasoning. After that, it has one more mid-training for "long context + reasoning" | [SmolLM3 blog](https://github.com/huggingface/blog/blob/main/smollm3.md) |
| **MiniCPM** | In the decay phase, it mixes high-quality SFT data into the pretraining data. A controlled experiment shows that "add it in the decay phase" is better than "add it only in the SFT stage" | [MiniCPM §5, §6.2](https://arxiv.org/abs/2404.06395) |
| **Qwen3** | The second stage (about 5T tokens) increases the share of STEM, code, reasoning, and synthetic data, and it "accelerates the learning-rate decay" | [Qwen3 §3.2](https://arxiv.org/abs/2505.09388) |
| **MobileLLM-R1** | Two mid-training stages of 100B tokens each. The learning rate decays linearly to 0. The data is Dolmino plus more math and code. It also uses Llama-3.1-8B for logits distillation | [MobileLLM-R1 §3, Appendix A](https://arxiv.org/abs/2509.24945) |
| **Puro-2B** | The second stage has 960B tokens with linear decay. In each source, it sorts the data by quality score from low to high, so the better data comes later. The sorted order is 1.18 points higher than a random order on the mean of 15 tasks | [Puro-2B §3.4](https://www.alphaxiv.org/abs/2608.27370) |

OLMo 2 shows the effect most directly (Table 9). For the 7B model, mid-training increases the mean score of 10 evaluations from 53.0 to 62.9, and GSM8K from 24.1 to 67.5. Each mid-training run has only 50B tokens, about 1.3% of the 3.9T tokens of pretraining (three runs, then an average).

### 1.3 Why it works: three reasons

1. **A low learning rate "fixes" what the model learns.** As we said above, the decay phase sets where the model stops. The data distribution in this phase has a much larger effect on the final model than the same number of tokens earlier in training.
2. **Good data is rare, so put it where it has the most effect.** High-quality math, code, and instruction data often has only billions to tens of billions of tokens. Spread over a pretraining run of trillions of tokens, it either has a share that is too small to matter, or it repeats many times. MiniCPM gives this argument: put the data together in the decay phase. Then the small data sets do not repeat many times, and they arrive in the phase with the largest effect. The "microannealing" experiments of OLMo 2 also found that data of the target domain helps when it is **present**. Math at 10% and at 35% gave almost the same result (GSM8K subset 61 vs 63.5; before annealing, 28.5).
3. **Data evaluation becomes cheap.** To find out if a new data set is good, you do not need to train a model from zero. Branch off a short decay from a checkpoint of the stable phase, and compare "with the data" and "without the data". Llama 3 uses this method to measure the value of small data sets (the method is similar to Blakeney et al. 2024). OLMo 2 calls it "microannealing". This is the same use of "branched decay" as in Chapter 6, but this time the branches compare data.

### 1.4 Small experiment: decay × new data

`04_anneal_mixture.py` trains the minimal Transformer of Chapter 9 (byte level) on three "sources": English (Shakespeare), Chinese (classical poetry), and code. The mixture of the trunk is English 0.45 / Chinese 0.45 / code 0.10. Code is the kind of data that is "rare, and we want to make that skill stronger", like math in OLMo 2. The trunk trains for 800 steps at a constant learning rate. Then 4 branches start from the same point, and each branch trains for 200 steps:

```python
decay = [PEAK_LR * (1 - (s + 1) / BRANCH_STEPS) for s in range(BRANCH_STEPS)]   # linear decay to 0
const = [PEAK_LR] * BRANCH_STEPS
branches = {
    # 恒定 = constant, 衰减 = decay, 原配比 = old mixture, 新配比 = new mixture
    "A 恒定 + 原配比": (const, MIX_PRETRAIN),
    "B 衰减 + 原配比": (decay, MIX_PRETRAIN),
    "C 恒定 + 新配比": (const, MIX_ANNEAL),     # new mixture: English 0.25 / Chinese 0.25 / code 0.50
    "D 衰减 + 新配比": (decay, MIX_ANNEAL),
}
```

> **Note:** The names of the branches and the sources in the code stay in Chinese. They are keys in the result cache, and the video of this chapter reads these keys. The comment in the code gives their meanings. The script prints the English names.

Output (validation bits-per-byte, lower is better; about 8 min of CPU time on one thread):

| | English | Chinese | Code | Mean |
|---|---:|---:|---:|---:|
| Branch point (trunk, 800 steps) | 2.759 | 3.148 | 2.746 | 2.884 |
| A constant + old mixture | 2.700 | 3.097 | 2.632 | 2.810 |
| B decay + old mixture | 2.549 | 3.037 | 2.472 | 2.686 |
| C constant + new mixture | 2.803 | 3.168 | 2.223 | 2.731 |
| D decay + new mixture | 2.642 | 3.097 | **2.091** | **2.610** |

> **About the numbers:** The numbers of the training experiments in this chapter come from one CPU run on the course build machine. Different machines and different versions of the low-level math libraries do the floating-point operations in a slightly different order. After a few hundred training steps, these small differences become larger. Your numbers can be different from the second or third decimal place. Use the conclusions below, which do not depend on the exact values. For a rerun on another server in 2026-10, see [runs/2026-10-01-gpu0-check/chapters-11-15.md](../../runs/2026-10-01-gpu0-check/chapters-11-15.md).

Use A ("keep the constant learning rate, do not change the data") as the baseline. The bits-per-byte of code decreased by 0.160 with decay only (B), by 0.409 with new data only (C), and by 0.541 with both (D). Look at these points:

- **Decay alone helps all sources**: B is about 0.06–0.17 lower than A on English, Chinese, and code. We saw this effect in Chapter 6.
- **New data is a trade-off**: in C, code becomes much better, but English and Chinese become worse than in A (English 2.803 vs 2.700). The reason is that their shares decreased from 0.45 to 0.25.
- **D adds the two effects together**: it has the best code and the best mean. English and Chinese are not as good as in B (2.642 vs 2.549, 3.097 vs 3.037), but they are still better than or equal to A. Real mid-training must make the same trade-off. This is why high-quality web pages are still about half of Dolmino in OLMo 2, and not all of it changes to math.
- The four branches together added only 4 × 200 = 800 steps. But they answered two questions: "Is decay worth it?" and "Is new data worth it?" This is the value of "branched decay" as a tool for data experiments.

This is a very small experiment with one seed and about 840K parameters (the minimal Transformer of Chapter 9). It only shows the method. Do not extrapolate the size of the differences to the main-line model.

### 1.5 What the mid-training of the main-line model adds

GOAL.md Section 3.3 specifies the mid-training data of the main-line model. In addition to the high-quality web pages, it adds three kinds of data (`configs/main/midtrain.toml`; the branched-decay experiments of Step 2 will set the ratios):

- **Math and code**: FineMath and Stack-Edu (OLMo 2, SmolLM3, and Llama 3 all upsample them in this stage).
- **Instruction-style data**: like FLAN in OLMo 2 and the SFT data in MiniCPM. This data lets the base model learn the "question–answer" format early.
- **Tool-calling format data**: the main-line model needs this data (the goal of Chapters 16 and 19 is tool calling). If a format such as `<tool_call>{...}</tool_call>` occurs at the end of pretraining, the model does not have to learn the format from zero in post-training. **This point is a design choice of this course, not a verified consensus of the field.** In Step 2, a branched-decay experiment will check if it really helps.

Two rules apply. First, the mid-training data also needs 13-gram decontamination (GOAL.md 3.2). Second, OLMo 2 let itself look at only 200 of the 1319 GSM8K questions to tune mid-training. We are stricter: all decisions use only our own development set, and the test benchmarks run only at the gates.

## 2. Long context: why the model cannot read long text

### 2.1 The clock of RoPE

Chapter 9 explained RoPE. It puts the `head_dim` dimensions of q and k into `d/2` pairs. At position m, pair i turns by `m·ω_i` radians:

```
ω_i = θ^(-2i/d)               rotation speed (θ is the base frequency, default 10,000)
λ_i = 2π / ω_i = 2π·θ^(2i/d)  wavelength: the number of tokens for one full turn
```

Think of a clock with 64 hands. Hand 0 is the second hand: it turns 1 radian for each token. The last hand is slower than the hour hand. `01_rope_wavelengths.py` calculates the wavelength of each pair for `head_dim = 128` of the main-line model:

```python
def wavelengths(head_dim: int, theta: float) -> np.ndarray:
    return 2 * math.pi / inv_freq(head_dim, theta)    # λ_i = 2π/ω_i
```

Output (part; unit: tokens):

| Dimension pair i | θ = 10K | θ = 500K (Llama 3) | θ = 1M (Qwen3 long context) |
|---:|---:|---:|---:|
| 0 | 6.3 | 6.3 | 6.3 |
| 16 | 62.8 | 167.1 | 198.7 |
| 32 | 628.3 | 4,443 | 6,283 |
| 48 | 6,283 | 118K | 199K |
| 63 | 54K | 2.6M | 5.1M |

| Pairs that do not make one full turn in length L (λ_i > L), of 64 pairs | θ = 10K | θ = 500K | θ = 1M |
|---|---:|---:|---:|
| L = 4096 | 18 | 32 | 33 |
| L = 32768 | 4 | 22 | 24 |

The wavelength of the fastest pair is always 2π ≈ 6.3 (1 radian for each token). It does not depend on θ. θ sets only how slow the later hands are.

### 2.2 Two reasons why the model cannot read long text

**Reason 1: unseen angles.** With a training length of 4096 and θ = 10K, 18 hands **do not make one full turn** in 4096 tokens. For these dimension pairs, the model saw only one arc of the circle. At 32K, they turn to angles that never occurred in training. Think of a person who saw a clock only between 0:00 and 3:00, and must suddenly say what "9:00" means. The YaRN paper gives this explanation: the dimensions that do not make one full turn carry **absolute position** information. When they go past the training range, the model does not recognize them.

**Reason 2: diluted attention.** The softmax divides the weights among all positions. When the positions increase from 4K to 32K, there are 8 times more candidates. The attention distribution becomes "flatter" (its entropy increases). Attention that should be concentrated spreads over many unrelated positions. The "temperature" of YaRN addresses this problem (Section 3.3).

A small experiment (`03_context_extension.py`) makes the problem small enough for a CPU. It takes the minimal Transformer of Chapter 9 (byte level, `head_dim = 32`, θ = 10K) and trains it for 1500 steps on sequences of length **64** only. Then it calculates the loss directly on validation sequences of length 128 and 256. With causal attention, the prediction at position p sees only `[0, p]`. Thus the loss in each position range shows how the model does "past the training length":

Pretrain on 3,072,000 bytes (tokens). Then **do no training**, and change only the RoPE cos/sin (validation loss in nat/byte, lower is better; the last three columns are position ranges):

| Setting | L = 64 | L = 128 | L = 256 | Positions 0–64 | Positions 64–128 | Positions 128–256 |
|---|---:|---:|---:|---:|---:|---:|
| (a) Change nothing | 1.578 | 1.646 | 1.879 | 1.578 | 1.714 | **2.111** |

With "change nothing", the loss inside the training length (positions 0–64) is 1.578. Past position 64, the loss increases, and at positions 128–256 it is 2.111. A longer context gives more content to use, so the loss should be **lower**. This is the problem "the model cannot read long text".

(The rows for PI, YaRN, and the larger base frequency of the same table are in Section 3.4. We look at them after we explain the three methods.)

## 3. Three methods: larger base frequency, position interpolation, and YaRN

All three methods **change only the rotation speeds of RoPE**. They do not change any learnable parameter. After the change, each method must train a little more on long text.

### 3.1 Larger base frequency (ABF)

The most direct method is to make θ larger. Then all hands become slower together. Xiong et al. (2023, Llama 2 Long) call this method **ABF (adjusted base frequency)**. When θ increases from 10K to 1M, pair i becomes `100^(2i/d)` times slower:

| Dimension pair i | 0 | 16 | 32 | 48 | 63 |
|---|---:|---:|---:|---:|---:|
| Times slower | 1.0 | 3.2 | 10.0 | 31.6 | 93.1 |

The second hand does not change, so the model can still tell near positions apart. The slow hands become much slower. At 32K, **no hand turns to an angle that "training with θ = 10K at length 4K" did not show** (part 3 of `01`: 18 pairs → 0 pairs). The cost: each angle now means a different distance. Thus the model must continue to train with the new base frequency.

Who uses it: in its long-context stage, Qwen3 "uses ABF to increase the base frequency from 10,000 to 1,000,000, as in Qwen2.5" (`config.json`: `rope_theta: 1000000`). SmolLM3 extends in two stages: θ becomes 1.5M for 4K→32K and 5M for 32K→64K (`config.json`: `rope_theta: 5000000.0`). Gemma 3 increases the base frequency of its global attention layers from 10K to 1M. Llama 3 and OLMo 2 use θ = 500K from the start of pretraining. (The Llama 3 report cites the result of Xiong et al.: this value works for lengths up to 32K.)

### 3.2 Position interpolation (PI): the base for YaRN

Position interpolation (PI) by Chen et al. (2023) uses a different idea. It does not let the positions go past the range. Instead, it "compresses" the positions. To extend by s times, it uses m/s in place of position m. This is the same as making all hands s times slower:

```python
def pi_inv_freq(head_dim: int, theta: float, s: float) -> torch.Tensor:
    return rope_inv_freq(head_dim, theta) / s      # make all dimension pairs s times slower
```

This solves the out-of-range problem, but the second hand also becomes slower. Before, two adjacent tokens differ by 1 radian on the second hand. After ÷8, they differ by only 0.125 radian. **Near positions become hard to tell apart.** In the ablation of the YaRN paper, PI ×8 without fine-tuning gives a perplexity above 10 (the original model: about 4). Gemma 3 uses PI-style scaling to extend to 128K (`rope_scaling: {rope_type: "linear", factor: 8.0}`), together with continued training. This course uses PI as the base to understand YaRN (GOAL.md 2.1, rule C).

### 3.3 YaRN: keep the fast hands, interpolate the slow hands, and adjust the temperature

YaRN (Peng et al., 2023) combines the two ideas above: **compress only the slow hands that really go past the range, and do not touch the fast hands**. It puts the dimension pairs into three zones by "the number of turns in the training length L", `r_i = L/λ_i`:

- More than β_fast = 32 turns (high frequency): **keep them as they are**. These pairs encode only relative distance, so they cannot go past the range.
- Less than β_slow = 1 turn (low frequency): **fully interpolate** them, ÷s, as in PI.
- Between the two: a linear ramp.

YaRN also multiplies the attention logits by a temperature factor, `√(1/t) = 0.1·ln(s) + 1`. This factor compensates for "more candidates, flatter attention". The implementation does not change the attention code. It multiplies cos/sin by this number. The rotation applies the factor once to q and once to k, so the logits get its square. The core of `02_yarn_from_scratch.py`:

```python
low = max(math.floor(dim_with_turns(beta_fast)), 0)                # pairs before it: > β_fast turns
high = min(math.ceil(dim_with_turns(beta_slow)), head_dim - 1)     # pairs after it: < β_slow turns
ramp = ((i - low) / (high - low)).clamp(0, 1)   # 0 = high-frequency zone, 1 = low-frequency zone
keep = 1 - ramp                                  # weight of the original frequency
new_w = keep * w + (1 - keep) * w / s            # high: no change; low: ÷ s; between: a mix
mscale = 0.1 * math.log(s) + 1.0                 # √(1/t) = 0.1·ln(s) + 1
```

`02` prints what YaRN does to each dimension pair. It uses the setup of the small experiment of this chapter (`head_dim = 32`, θ = 10K, training length 64, s = 4). Part of the output:

| i | Wavelength λ | Turns in training | Keep weight | ω (original) | ω (PI) | ω (YaRN) |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 6.3 | 10.19 | 1.00 | 1.00000 | 0.25000 | 1.00000 |
| 1 | 11.2 | 5.73 | 0.80 | 0.56234 | 0.14059 | 0.47799 |
| 2 | 19.9 | 3.22 | 0.60 | 0.31623 | 0.07906 | 0.22136 |
| 4 | 62.8 | 1.02 | 0.20 | 0.10000 | 0.02500 | 0.04000 |
| 5 | 111.7 | 0.57 | 0.00 | 0.05623 | 0.01406 | 0.01406 |
| 15 | 35333 | 0.00 | 0.00 | 0.00018 | 0.00004 | 0.00004 |

mscale = 0.1·ln 4 + 1 = 1.1386, so the attention logits are multiplied by 1.2965. Note that the code rounds the "turns" boundaries to whole pair indices (`floor` / `ceil`). With a training length of only 64, pair 0 makes only 10 turns, not 32. But after rounding, it is still in the "keep" zone. This is how the official YaRN code works, and transformers and zero do the same. Equations (10)–(13) of the paper make the ramp linear in the "number of turns". In the ramp zone, this differs from the official code by up to 57% (part 4 of `02`). Use the code as the reference.

For the settings of the main-line model (part 2 of `02`):

| Case | Keep as they are | Ramp | Full interpolation | mscale |
|---|---:|---:|---:|---:|
| θ = 10K, 4K → 32K (YaRN only) | 21 pairs | 25 pairs | 18 pairs | 1.208 |
| θ = 1M, 32K → 128K (the method of the Qwen3 model card) | 24 pairs | 16 pairs | 24 pairs | 1.139 |

Parity check (part 3 of `02`): the from-scratch version agrees with `zero.model.compute_rope_inv_freq` and with `Qwen3RotaryEmbedding` of transformers on 4 settings. The largest frequency difference is ≤ 6.0×10⁻⁸ (float32 rounding), and mscale is exactly the same.

### 3.4 Small experiment: fine-tune for a short time

Zero-shot (no training) is only the start. In real models, all three methods train more. Llama 3 extends from 8K to 128K in 6 stages, with about 800B tokens. The long-context stage of Qwen3 uses "hundreds of billions" of tokens. DeepSeek-V3 uses YaRN in two stages of 1000 steps each (4K→32K→128K). SmolLM3 uses two stages of 50B tokens each. In the small experiment, each setting fine-tunes for 150 steps on sequences of length 256 (1/10 of the pretraining tokens), with exactly the same data order:

First, the full zero-shot table (no training, only a new RoPE):

| Setting | L = 64 | L = 128 | L = 256 | Positions 0–64 | Positions 64–128 | Positions 128–256 |
|---|---:|---:|---:|---:|---:|---:|
| (a) Change nothing | 1.578 | 1.646 | 1.879 | 1.578 | 1.714 | 2.111 |
| (b) Position interpolation PI (÷4) | 3.461 | 3.531 | 3.561 | 3.461 | 3.602 | 3.590 |
| (c) YaRN (s = 4) | 1.654 | 1.660 | 1.673 | 1.654 | 1.667 | **1.685** |
| (d) Larger base frequency (θ = 100K) | 1.626 | 1.643 | 1.851 | 1.626 | 1.660 | 2.058 |

After fine-tuning for 150 steps on sequences of length 256 (307,200 tokens, 1/10 of pretraining):

| Setting | L = 64 | L = 128 | L = 256 | Positions 0–64 | Positions 64–128 | Positions 128–256 |
|---|---:|---:|---:|---:|---:|---:|
| (a) Change nothing | 1.587 | 1.570 | 1.561 | 1.587 | 1.553 | 1.553 |
| (b) Position interpolation PI | 1.701 | 1.688 | 1.679 | 1.701 | 1.676 | 1.669 |
| (c) YaRN | **1.578** | **1.564** | **1.555** | 1.578 | 1.550 | 1.547 |
| (d) Larger base frequency | 1.591 | 1.574 | 1.568 | 1.591 | 1.557 | 1.562 |

How to read the two tables:

- **Zero-shot, YaRN is the most stable**: at positions 128–256, the loss is only 1.685, almost the same as inside the training length. The cost is a slightly worse loss up to length 64 (1.654 vs 1.578), because the slow hands are slower and the temperature changed. This is why the Qwen3 model card warns that "static YaRN can affect short text, so turn it on only when you need it".
- **Zero-shot, PI is the worst**: even up to length 64, the loss increases from 1.578 to 3.461. The 4× compression also makes the second hand slower. The model cannot tell near positions apart, so it cannot read even short text well.
- **Zero-shot, a larger base frequency alone almost does not help** (2.058 vs 2.111): the relation between each angle and each distance changed, so the model must adapt again. This method is designed for use with continued training.
- **After fine-tuning, the differences become much smaller**: YaRN is the lowest at all three lengths. "Change nothing" and "larger base frequency" are close behind (only about 0.01 more). PI is still more than 0.1 behind.
- **One seed only**: the differences among YaRN, "change nothing", and "larger base frequency" are only a few hundredths. They are probably inside the seed noise (Hands-on task 2 asks you to check this with a different seed). A rerun on a different machine also moves the number in a cell by about 0.01, which is the same size as these differences. Only two conclusions are safe: zero-shot, YaRN is much better than the other three; with the same fine-tuning budget, PI is clearly behind.

The scale of this very small experiment (length 64 → 256, fewer than 15,000 steps) is very far from a real model (4K → 32K, billions of tokens). The experiment only shows the mechanism. It does not show which method is better for the main-line model. The main-line model uses a larger base frequency + continued training at 32K, with optional YaRN on top at inference. This choice follows the methods of the leading models in Sections 3.1–3.3.

## 4. Data and cost of long context

**Cost.** For each token, the operations of `QKᵀ` and `AV` in attention are proportional to the sequence length. Part 5 of `01` calculates the operations for each training token of the main-line model. It uses the formula of zero (PaLM Appendix B, no halving for the causal mask):

| Sequence length | Operations per token | Share of attention (`QKᵀ`, `AV`) |
|---:|---:|---:|
| 4096 | 6.96 GFLOP | 41% |
| 32768 | 26.69 GFLOP | 84% |

For the same number of tokens, training with 32K sequences costs 3.84× more. (FlashAttention skips the upper triangle of the causal mask and saves about half of the attention operations. But the relation stays the same.)

Thus nobody trains with long sequences all the time. In the words of Llama 3: "We do not train on long sequences earlier because the compute in self-attention layers grows quadratically in the sequence length." All models put long context at the end of pretraining, and it is only a very short stage.

**Data.** The data of the long-context stage must really be "long":

- **Naturally long documents**: books, code repositories, and long web pages. In the long-context corpus of Qwen3, 75% of the data is 16K–32K tokens long, and the other 25% is 4K–16K. **Keep some shorter data** to prevent a loss of short-text ability.
- **Synthetic long-text tasks**: in the SFT stage, Llama 3 uses the model to make question answering on long documents, hierarchical summaries, and a code task: "delete a source file that many files of the repository use, and let the model write it again". Llama 3 found that 0.1% of such data is enough to keep both long and short abilities.
- More is not always better. SmolLM3 tried extra upsampling of books and code repositories, above the natural length distribution. Its ablations found that this did not improve the RULER and HELMET scores more. "The decay-phase mixture + longer sequences + a larger base frequency" was enough.

**Do not lose the short-text ability.** Llama 3 uses two criteria to decide if an extension stage succeeded. First, the short-context evaluations **recover completely**. Second, needle in a haystack at that length is **all correct**.

## 5. How to evaluate long context

- **Needle in a haystack (NIAH)** (Kamradt, 2023): at some depth of a long text with unrelated content, insert one "needle" sentence (for example, "The password of X is a 7-digit number"). At the end, ask the model. Scan length × depth and draw the results as a table. NIAH is a **smoke test**: if the model fails, there is surely a problem; if the model passes, it does not mean that the model really uses long context. The Llama 3 report says that all NIAH answers were correct, and DeepSeek-V3 passes all tests up to 128K.
- **RULER** (Hsieh et al., 2024, NVIDIA): extends NIAH to 13 synthetic tasks in 4 categories: retrieval (several kinds of needles, several distracting needles), multi-hop tracing (chains of variable assignments), aggregation (find the most common words), and question answering on long documents. It tested 17 models. **Almost all models are near a perfect score on plain NIAH, but their RULER scores decrease clearly with length.** Of the models that claim support for 32K or more, only half still pass at 32K (the pass line is 85.6, the score of Llama2-7B at 4K). The reports of Qwen3, Gemma 3, and SmolLM3 all use RULER for their long-context results.

This chapter adds a needle-in-a-haystack tool to the main-line model: `zero/tools/needle.py` (see "From minimal code to production code"). The tool checks if "the generated text contains the correct number". It also calculates a finer **likelihood gain**: as a control, replace the number in the needle with a different random number. Then measure how much the negative log-likelihood of the correct answer decreases. Only a value > 0 shows that the model really uses the information in the needle.

## 6. Summary

- **Mid-training / annealing**: the decay phase of WSD uses only 5–10% of the compute, but it sets where the model stops at the end. Put high-quality web pages, math and code, and instruction-style data together in this phase. "Branched decay" compares mixtures at a low cost.
- **Why the model cannot read long text**: slow hands that do not make one full turn in the training length turn to unseen angles at longer positions. More candidates also dilute the attention.
- **Larger base frequency**: all hands become slower together, and the fast hands almost do not change. The model must continue to train with the new base frequency.
- **YaRN**: keep the fast hands, divide the slow hands by s, use a ramp between them, and multiply the logits by `(0.1·ln s + 1)²`. In the small experiment, YaRN almost does not lose quality zero-shot. After a short continued training, it is still the best (by a small margin).
- **Cost and evaluation**: at 32K, the compute per token is about 3.8× the compute at 4K, so only a short stage at the end uses long sequences. Needle in a haystack is a smoke test; RULER is the benchmark.

---

## GPU measurements (one RTX 3090)

> **Note:** All numbers in the text above come from CPU runs. This section measures on one NVIDIA GeForce RTX 3090 (24 GB memory, Ampere architecture; data sheet: dense BF16 Tensor Core peak about 71 TFLOPS, FP32 about 35.6 TFLOPS, memory bandwidth about 936 GB/s). Environment: PyTorch 2.11.0+cu128, CUDA 12.8, October 2026. The server sets the power limit of this card to 240 W (the factory default is 350 W). Under a constant full load, the card decreases its clock. Thus the absolute compute and bandwidth are lower than on a 3090 at full power, and the relative values are more reliable. If you do not have a GPU, skip this section.

Section 4 says: "for the same number of tokens, training with 32K sequences costs 3.84× more". That number comes from a formula. This 24 GB card cannot hold one full 32K sequence of the main-line model. (With the memory formula of Chapter 14, the peak of the logits alone needs 12 GiB.) Thus we measure only **one layer**: one Block of the long-context configuration (`configs/main/longctx.toml`, RoPE base frequency 1M). It uses the zero implementation, BF16 autocast, and SDPA attention (on this card, SDPA uses FlashAttention; see the GPU measurements of Chapter 14). Each call gets 32,768 tokens. Only the sequence length T changes, and T × micro batch stays the same. A 3090 is not an H100, so the long-context cost of the main line on 8×H100 is still an estimate. Here we look at the ratio "how many times more expensive" on a real card.

Run:

```bash
uv run python chapters/15-midtraining-long-context/code/05_gpu_long_context_cost.py   # one main-line Block, T = 4K → 32K (about 20 s)
```

Forward + backward, median of 10 runs. "Formula" is the operations per token for one layer, 6·N_layer + 12·q_dim·T (the rule of Section 4, without the output layer). "Causal halved" uses half of the attention term. "Real TFLOPS" uses the operations that the GPU really does after the halving. "MFU by the zero rule" uses the formula without halving. Both are relative to the BF16 data-sheet peak of 71 TFLOPS:

| T | Micro batch | Time | Per token | Measured ratio | Formula, not halved | Formula, causal halved | Real TFLOPS | Share of peak | MFU by the zero rule |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4,096 | 8 | 172.2 ms | 5,255 ns | 1.00× | 1.00× | 1.00× | 34.3 | 48% | 62% |
| 8,192 | 4 | 209.2 ms | 6,385 ns | 1.21× | 1.44× | 1.28× | 36.1 | 51% | 73% |
| 16,384 | 2 | 283.3 ms | 8,644 ns | 1.64× | 2.31× | 1.84× | 38.3 | 54% | 87% |
| 32,768 | 1 | 435.2 ms | 13,282 ns | 2.53× | 4.06× | 2.96× | 40.1 | 56% | 99% |

Memory: at all four lengths, this layer uses 2,943 MiB more at the end of the forward pass. Without the BF16 weight copies, this is 92,868 bytes per token. It differs from the `memory_calc` formula of Chapter 14 (92,840 bytes) by only 0.03%, and it **does not depend on T**.

How to read the results:

- **At 32K, each token costs 2.53× more in the measurement, not 4.06×** (this is for one layer; the 3.84× of Section 4 includes the output layer). The difference has two sources. First, the causal mask: FlashAttention skips the upper triangle, and the halved formula gives 2.96×. Second, the attention kernel is "fuller" on long sequences: with the operations that the GPU really does, the utilization increases from 48% to 56%. The overhead of normalization, RoPE, and element-wise operations depends only on the number of tokens, and it does not increase with T. The full model also has an output layer that does not depend on T, so the ratio becomes a little smaller again. Thus the 3.84× of Section 4 is a conservative upper bound. On this card, the real cost is about 2.5×. The direction of the conclusion does not change: long sequences are expensive, so train only a short stage at the end with them.
- **Check the formula before you compare MFU**: the MFU in the zero log uses the formula without halving (Chapter 14, Section 6). The longer T is, the more of the upper triangle the formula counts, although the GPU does not do it. At 32K, this layer gets 99% by the zero rule, almost the peak. The real utilization is only 56%. To compare the MFU of the long-context stage with the MFU of the pretraining stage, first halve the attention term, and then compare.
- **Memory does not depend on T, but it depends on the number of tokens**: the activations per token do not change with T, because FlashAttention does not store the T × T matrix. The problem at 32K is that one sequence has 32,768 tokens: 28 layers × 92,868 bytes × 32,768 ≈ 79 GiB. The activations of one sequence alone almost fill an 80 GB H100. FSDP splits the parameters, gradients, and optimizer states, but not the activations (Chapter 14, Section 5.3). To fit one 32K sequence, you also need activation checkpointing (Chapter 14, Section 2), or you must split one sequence across several GPUs. This is the calculation that Hands-on task 1 of Chapter 14 asks you to do.

---

## From minimal code to production code

| Minimal code (`code/`) | Production code (`zero/`, `configs/`, `tests/`) | What it adds, and why |
|---|---|---|
| `yarn_inv_freq` in `02` | `zero/model.py`: `compute_rope_inv_freq(head_dim, theta, scaling)` returns `(inv_freq, attention_scaling)`. `RotaryEmbedding` precomputes cos/sin for `[0, max_seq_len)` and multiplies them by `attention_scaling` | cos/sin are **non-persistent buffers**, so they are not in the `state_dict`. After a change to `rope_theta` or `rope_scaling`, build them again, and load all other weights as they are. A position past `max_seq_len` causes an error, so the model does not extrapolate silently |
| Manual replacement of cos/sin in `03` | `zero/config.py`: `ModelConfig.rope_theta`, `ModelConfig.rope_scaling` (checks: only `type = "yarn"` is supported, `factor ≥ 1`, `original_max_position_embeddings` is necessary, and an unknown field causes an error). At export, `zero/hf.py` writes it as the HF `rope_scaling` / `rope_parameters` | Driven by the configuration. After export, transformers, vLLM, and llama.cpp can all run inference with the same YaRN parameters |
| "Branch from the same point + new mixture + linear decay" in `04` | `zero/train/midtrain.py`: `init_from` points to the pretraining checkpoint. `check_compatible` allows changes only to `rope_theta`, `rope_scaling`, and `max_seq_len` (a change to the number of layers or to a dimension causes an error), and it prints "what changed". `[schedule] kind = "wsd"` with `decay_frac = 1.0` is one full decay phase. `[[data.sources]]` changes the mixture (`zero/data/mixture.py` samples by weight) | Uses the same `zero/train/trainer.py` as pretraining (BF16, gradient accumulation, resume from checkpoints, multiple GPUs). After an interruption, run the same command again. It resumes from the checkpoint of this run |
| "Fine-tune at length 256" in `03` | `configs/main/longctx.toml` (see the table below) | 32K sequences, FSDP, 1M tokens per step |
| None | `zero/tools/needle.py`: `make_case` (exact control of the length and of the needle position, with a control prompt), `score_text`, `answer_nll`, `run_grid`, `format_grid`. Command line: `uv run python -m zero.tools.needle --model <ckpt> --lengths ... --depths ...` | Can evaluate any zero checkpoint or exported HF folder. You can also pass your own `generate_fn` to evaluate other models |

**Parity checks**:

- Part 3 of `02`: the from-scratch YaRN agrees with `zero.model.compute_rope_inv_freq` and with `Qwen3RotaryEmbedding` of transformers on 4 settings (including "32K×4", which Qwen3 recommends). The largest frequency difference is ≤ 6×10⁻⁸ (float32 rounding), and mscale is exactly the same.
- The cases `yarn` and `yarn_custom_beta` in [`tests/test_model_hf_parity.py`](../../tests/test_model_hf_parity.py): a randomly initialized HF `Qwen3ForCausalLM` with YaRN. The test copies its weights into zero. On 150 positions (past the original length of 32 / 48), the logits differ by less than 1e-5.
- `test_greedy_batch_and_yarn` in [`tests/test_kv_cache.py`](../../tests/test_kv_cache.py): with YaRN on, greedy generation with the KV cache and without the KV cache gives exactly the same output.
- [`tests/test_needle.py`](../../tests/test_needle.py) (new in this chapter, 9 tests): the prompt length is exactly the requested length; the needle position increases with depth; the control prompt differs only in the number of the needle. A fake model that "reads the needle" gets a full score, and a fake model that says nothing gets 0. `answer_nll` agrees with a token-by-token calculation by hand. A real zero model runs through the full table.

```bash
uv run pytest tests/test_model_hf_parity.py tests/test_kv_cache.py tests/test_needle.py -q
```

Result on our machine: `24 passed in 16.73s` (9 of them are the new tests in `test_needle.py`).

**The main-line configuration, item by item**:

`configs/main/midtrain.toml` (inherits from `pretrain.toml`):

| Setting | Value | Why |
|---|---|---|
| `train.init_from` | `out/main/pretrain/ckpt` | Continue from the last checkpoint of the stable phase of pretraining |
| `train.max_steps` | 50000 (≈ 26B tokens) | About 6.5% of the 400B tokens of pretraining, inside the 5–10% range of OLMo 2; to be decided |
| `[schedule]` | `wsd`, `warmup_steps = 0`, `decay_frac = 1.0`, `min_lr_ratio = 0` | The full stage is the decay phase of WSD: a linear decay from the peak to 0 (OLMo 2, Llama 3, and MobileLLM-R1 all decay linearly to 0) |
| `[[data.sources]]` | fineweb-edu 0.35, fineweb-2-zh 0.25, stack-edu 0.15, finemath 0.15, instruct-toolcall 0.10 | Math and code are about double their pretraining shares of 0.12 / 0.08, plus 10% instruction and tool-calling format data. In Step 2, branched decay sets the exact ratios |
| Other settings (model shape, optimizer, seq_len 4096) | Inherited | Mid-training does not change the model shape. `check_compatible` stops a change by mistake |

`configs/main/longctx.toml` (inherits from `pretrain.toml`, and continues from the result of `midtrain`):

| Setting | Value | Why |
|---|---|---|
| `model.rope_theta` | 1,000,000 | ABF: 10K → 1M, the same as the long-context stage of Qwen3 |
| `model.max_seq_len`, `data.seq_len` | 32768 | Train directly at the target length (do not depend on extrapolation at inference) |
| `micro_batch_size × grad_accum × 8 GPUs × 32768` | 1 × 4 × 8 × 32768 = 1,048,576 tokens/step | The activations of a 32K sequence are very large, so each GPU holds only one sequence at a time |
| `train.parallel` | `fsdp` | Split the parameters, gradients, and optimizer states over 8 GPUs to free memory for the activations. (FSDP2 itself was verified on 2×RTX 3090. But on 24GB cards, FSDP with 2 or 3 GPUs runs out of memory (OOM) at 32K: the logits and activations of each GPU cannot be split. See Section 14.5 of [runs/2026-10-01-gpu0-check](../../runs/2026-10-01-gpu0-check/README.md). It still needs a test on 8×H100) |
| `train.max_steps` | 4000 (≈ 4.2B tokens) | To be decided. DeepSeek-V3 uses 1000 steps per stage, and SmolLM3 uses 50B tokens per stage, so 4.2B is conservative |
| `optim.lr`, `[schedule]` | 1e-4, cosine, 200 warmup steps, decay to 10% | To be decided; see "Open questions" below |
| Longer at inference | Optionally add `model.rope_scaling = {type = "yarn", factor = 4, original_max_position_embeddings = 32768}` | The method that the Qwen3 model card recommends: 32K native, YaRN ×4 to 128K. Turn it on only when necessary (static YaRN has a small effect on short text) |

**Open questions (in Step 2, small-scale experiments decide them)**: `midtrain` already decays the learning rate to 0. Then `longctx` warms up again to 1e-4. This is the same as two annealing stages. The leading models do not use the same order. Qwen3 puts long context in the last stage of pretraining. Llama 3 first extends the length, and then anneals on the last 40M tokens at length 128K. DeepSeek-V3 extends the length with the learning rate from the end of pretraining, 7.3×10⁻⁶. SmolLM3 uses "the decay-phase mixture + longer sequences". Also, zero does not do intra-document attention masking at this time (see "Frontier notes"). Thus a 32K window contains many short documents that are packed together.

## Main-line progress

### Tiny-configuration demo (CPU, `configs/tiny`, about 1.3M parameters)

> **Note:** This is a **tiny-configuration demo**. It shows only that the code runs and that each change has the expected effect. It does not show any result of the main-line model.

First, run the tiny pretraining. (To keep the output folder separate from other chapters, we use `--set` to change `out_dir`. The default command also works.)

```bash
uv run python -m zero.train.pretrain --config configs/tiny/pretrain.toml --set train.out_dir=out/tiny/ch15/pretrain
uv run python -m zero.train.midtrain --config configs/tiny/midtrain.toml \
    --set train.out_dir=out/tiny/ch15/midtrain --set train.init_from=out/tiny/ch15/pretrain/ckpt
```

```
# pretraining (part)
Model parameters 1.31M (non-embedding 0.79M), 2048 tokens per step, 200 steps, device cpu, world_size=1
step    100/200 | loss 6.1065 | lr 3.00e-03 | gnorm 0.38 | 2,709 tok/s | val 6.0048
step    200/200 | loss 5.4736 | lr 3.00e-04 | gnorm 0.38 | 2,608 tok/s | val 5.7116

# mid-training
[midtrain] model.rope_scaling: None → {'type': 'yarn', 'factor': 2.0, 'original_max_position_embeddings': 128, 'beta_fast': 32.0, 'beta_slow': 1.0}
[midtrain] data.seq_len: 128 → 256
[midtrain] data mixture: {'shakespeare': 0.45, 'chinese_poetry': 0.45, 'code': 0.1} → {'shakespeare': 0.3, 'chinese_poetry': 0.6, 'code': 0.1}
Loaded model weights from out/tiny/ch15/pretrain/ckpt (step 200)
step      1/60 | loss 5.5548 | lr 9.83e-04 | gnorm 0.38 | 1,793 tok/s
step     30/60 | loss 5.4730 | lr 5.00e-04 | gnorm 0.42 | 2,434 tok/s | val 5.6488
step     60/60 | loss 5.4976 | lr 0.00e+00 | gnorm 0.38 | 2,229 tok/s | val 5.6101
```

Note three points in this log:

- `check_compatible` prints what this run changed: YaRN ×2 (original length 128), sequence length 128 → 256, and a mixture that moves toward Chinese.
- The learning rate decays linearly from 1e-3 to 0 (`decay_frac = 1.0`). This is the "full decay".
- The pretraining val 5.7116 is on sequences of length 128. The mid-training val 5.6101 is on sequences of length 256. **You cannot compare the two directly.** They show only this: after the RoPE change and the longer sequences, training converges as usual and does not break.

In 2026-10, a rerun of the same two commands on another server gave val 5.6004 at pretraining step 200 and val 5.5353 at mid-training step 60. The numbers are different (see "About the numbers" in Section 1.4), but the three points above are still true.

Needle in a haystack (`uv run python -m zero.tools.needle --model out/tiny/ch15/midtrain/ckpt --lengths 64,128,240 --depths 0,0.5,1 --n 5`):

```
Generation accuracy (fraction of greedy outputs that contain the correct number)
length\depth    0.00    0.50    1.00
          64    0.00    0.00    0.00
         128    0.00    0.00    0.00
         240    0.00    0.00    0.00

Likelihood gain nll_gain = NLL(control) − NLL(true needle); > 0 means that the model uses the needle
length\depth    0.00    0.50    1.00
          64  +0.010  -0.012  +0.000
         128  +0.004  +0.006  -0.064
         240  -0.019  -0.038  -0.027
```

A model with 1.3M parameters that trained for only 260 steps cannot find the needle. All accuracies are 0. The likelihood gain moves above and below 0. (Each cell has only 5 questions, so these differences are noise.) `tests/test_needle.py` makes sure that the tool itself is correct.

### To be added after GPU training

- The real training curves of mid-training and long-context extension, the data mixture and the cost of each stage, and the failures and rework.
- Branched-decay experiments that compare some mixtures for mid-training (especially: "does the tool-calling format data help?").
- The needle-in-a-haystack table at 32K, the RULER results (4K–32K), and whether the short-context evaluations recover.
- The comparison report of **Gate 2** (see the checklist below).

**Cost estimate** (`zero.tools.estimate_cost`; assumptions: H100 SXM, MFU 0.4, $2.5 per GPU-hour; **not verified on a GPU yet**; with 32K sequences, the real MFU can be lower):

| Stage | Tokens | Operations per token | GPU-hours | Cost |
|---|---:|---:|---:|---:|
| Mid-training (`midtrain.toml`, 4K) | 26.2B | 6.96 GFLOP | 127.9 | $320 |
| Long context (`longctx.toml`, 32K) | 4.19B | 26.69 GFLOP | 78.5 | $196 |
| Total | | | 206.4 | $516 (budget in GOAL.md 3.4: $700) |

Both items are more than $100. Thus, as GOAL.md 3.4 specifies, the author must approve the estimate before the runs start.

### Gate 2 checklist (GOAL.md 3.4: end of pretraining)

- [ ] Evaluate the base model after the long-context extension. Use the few-shot base-model benchmarks of the preregistration (`eval/PREREGISTRATION.md`) and our own development set. Report bootstrap 95% confidence intervals.
- [ ] Compare each item with the extrapolated predictions of Gate 1 (loss and benchmark scores). **If the results are clearly lower, diagnose first** (data, learning rate, code bugs, evaluation templates). Do not start post-training.
- [ ] Short-context ability: compare the development-set loss and the few-shot results before and after the long-context extension. Make sure that they "recover completely" (the Llama 3 criterion).
- [ ] Long context: `zero.tools.needle` is all correct at 4K / 8K / 16K / 32K × 5 depths. Report RULER 4K–32K as it is.
- [ ] Tool-calling format: look at the loss on held-out tool-calling format data. Make sure that the model learned the format data of mid-training.
- [ ] Archive the results of the 13-gram decontamination check of the mid-training data and the long-context data.
- [ ] Record the cost in `runs/ledger.md`.
- [ ] Do not use the preregistered test benchmarks to select checkpoints or to tune mixtures (GOAL.md Section 11).

---

## Frontier notes

> **NTK-aware interpolation**: one of the methods before YaRN. It changes to a larger base frequency, `θ' = θ·s^(d/(d−2))`, and spreads "the pressure of interpolation" over all dimensions. (The YaRN paper also puts Code Llama in this group: Code Llama set the base frequency to 1M by hand.) The best base frequency for each factor must be found by trial. The details are not in the main text.
>
> **Intra-document masking**: when several short documents are packed into one long window, each document sees only itself. Llama 3 says that it "has a limited effect in standard pretraining, but it is important in continued pretraining on very long sequences". SmolLM3 also uses it. DeepSeek-V3 says explicitly that it does not use it. At this time, we verified only two adopters, so it is not in the main text yet. The pretraining of zero does not implement it yet. Before the long-context work of Step 2, a small experiment should check it first.

---

## Adopters and sources

| Technique | Adopters (main versions) | Sources |
|---|---|---|
| Mid-training / annealing (high-quality data in the decay phase) | **OLMo 2** (mid-training, Dolmino Mix, LR decays linearly to 0); **Llama 3** (annealing upsamples high-quality data, and annealing evaluates data); **SmolLM3** (the decay phase upsamples math and code, and adds instruction and reasoning data); **MiniCPM** (the decay phase mixes in SFT data); **Qwen3** (stage S2 increases the share of STEM/code/reasoning/synthetic data and accelerates the LR decay); **MobileLLM-R1** (two mid-training stages, LR decays linearly to 0); Puro-2B (a data curriculum sorted by quality in the second stage + linear decay) | [OLMo 2](https://arxiv.org/abs/2501.00656) §2.3, §4; [Llama 3](https://arxiv.org/abs/2407.21783) §3.1.3, §3.4.3; [SmolLM3 blog](https://github.com/huggingface/blog/blob/main/smollm3.md); [MiniCPM](https://arxiv.org/abs/2404.06395) §5; [Qwen3](https://arxiv.org/abs/2505.09388) §3.2; [MobileLLM-R1](https://arxiv.org/abs/2509.24945) §3, Appendix A; [Puro-2B](https://www.alphaxiv.org/abs/2608.27370) §3.4 |
| Larger RoPE base frequency (ABF) | **Qwen3** (long-context stage 10K → 1M; `rope_theta: 1000000`); **SmolLM3** (1.5M → 5M; `rope_theta: 5000000.0`); **Gemma 3** (global layers 10K → 1M); **Llama 3** (θ = 500K from the start of pretraining; `rope_theta: 500000.0`); **OLMo 2** (`rope_theta: 500000`); gpt-oss (`rope_theta: 150000`) | [Qwen3](https://arxiv.org/abs/2505.09388) §3.2; [SmolLM3 blog](https://github.com/huggingface/blog/blob/main/smollm3.md); [Gemma 3](https://arxiv.org/abs/2503.19786) §2, §5.3; [Llama 3](https://arxiv.org/abs/2407.21783) §3.2; Xiong et al. 2023 [arXiv:2309.16039](https://arxiv.org/abs/2309.16039); the `config.json` of each model (links below) |
| YaRN | **DeepSeek-V3** (`rope_scaling: {type: yarn, factor: 40, original_max_position_embeddings: 4096, beta_fast: 32, beta_slow: 1}`, two stages of 1000 steps each, 4K→32K→128K); **gpt-oss** (`factor: 32`, `original_max_position_embeddings: 4096`); **Kimi K2** (`factor: 32`, original length 4096); **Qwen3** (32K native; the model card recommends YaRN `factor: 4` to 128K, and the RULER results use this setting); **SmolLM3** (trains at 64K, extrapolates to 128K with YaRN) | [YaRN](https://arxiv.org/abs/2309.00071); [DeepSeek-V3](https://arxiv.org/abs/2412.19437) §4.3; [Qwen3-8B model card](https://huggingface.co/Qwen/Qwen3-8B); [SmolLM3 blog](https://github.com/huggingface/blog/blob/main/smollm3.md); the `config.json` of each model |
| Position interpolation PI (base for YaRN) | Gemma 3 (`rope_scaling: {rope_type: linear, factor: 8.0}`; the report says that it follows the method of Chen et al.) | [PI](https://arxiv.org/abs/2306.15595); [Gemma 3](https://arxiv.org/abs/2503.19786) §5.3 |
| Long-context evaluation | NIAH: the reports of Llama 3 and DeepSeek-V3; RULER: Qwen3 (Appendix A.1.1), Gemma 3 (Table 15), SmolLM3 (blog) | [Kamradt NIAH](https://github.com/gkamradt/LLMTest_NeedleInAHaystack); [RULER](https://arxiv.org/abs/2404.06654) |

**Consensus decision (GOAL.md 2.1)**: mid-training / annealing, a larger base frequency, and YaRN each have 3 or more independent leading model families that use them explicitly in their main versions. Thus they are in the main text. PI is only the base for YaRN (rule C). NTK-aware interpolation and intra-document masking are in "Frontier notes".

**Model configurations** (read from Hugging Face in 2026-09):
[Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B/blob/main/config.json),
[Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B/blob/main/config.json),
[SmolLM3-3B](https://huggingface.co/HuggingFaceTB/SmolLM3-3B/blob/main/config.json),
[Llama-3.1-8B (unsloth mirror)](https://huggingface.co/unsloth/Meta-Llama-3.1-8B/blob/main/config.json),
[OLMo-2-1124-7B](https://huggingface.co/allenai/OLMo-2-1124-7B/blob/main/config.json),
[DeepSeek-V3](https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/config.json),
[gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b/blob/main/config.json),
[Kimi-K2-Instruct](https://huggingface.co/moonshotai/Kimi-K2-Instruct/blob/main/config.json),
[gemma-3-4b-pt](https://huggingface.co/google/gemma-3-4b-pt/blob/main/config.json).

Notes: the `rope_scaling` of Llama 3.1 is Meta's own `llama3` type (`factor: 8`, `low_freq_factor: 1`, `high_freq_factor: 4`, original length 8192). Its idea is the same as in YaRN ("zones by frequency"), but the formula is different, so we do not count it as a YaRN adopter. In the `config.json` of Kimi K2, `beta_fast` and `beta_slow` are both 1.0. This is different from the 32 / 1 that the YaRN paper recommends; the reason is to be verified. In DeepSeek-V3, YaRN applies only to the decoupled RoPE key in MLA (Chapter 21). In the `config.json` of Qwen3, `rope_scaling` is `null`: the model card recommends that users turn on YaRN when they need it.

---

## Guided questions

Ask Claude Code these questions. Continue until you can explain the answers in your own words:

1. Mid-training often uses only a few percent of the pretraining tokens. Why is its effect so large? If you spread the same high-quality data evenly over all of pretraining, what result do you expect? Do you agree with the two arguments of MiniCPM?
2. In the `04` experiment of this chapter, how much improvement do "new data without decay" (C) and "decay without new data" (B) each give? Why do English and Chinese not become worse in D (or how much worse do they become)? How do you balance "the skill that you want to make stronger" against "the skills that the model already has"?
3. After θ becomes larger, there are no "unseen angles" at 32K. Why must the model still train more? (Hint: each angle now means a different distance.) What happens if θ becomes infinitely large?
4. PI makes all hands s times slower. YaRN makes only the slow hands slower. In the table of `02`, find the dimension pairs that are "fully interpolated" in the small experiment of this chapter. Explain why the second hand (i = 0) must not change.
5. The YaRN temperature `0.1·ln(s) + 1` is an empirical formula from a fit. Ask Claude Code to help you derive it: when the candidate positions increase from L to sL, how much does the entropy of uniform attention increase? In which direction must the temperature move?
6. A model that gets all needle-in-a-haystack answers correct can do badly on RULER. Design a task that shows better than NIAH that a model "really uses long context". How can you score it automatically?

## Hands-on tasks

For each task, run the code and look at the result.

**Task 1 (basic)**: In `01_rope_wavelengths.py`, change `BASES` to models that interest you: gpt-oss (θ = 150K, head_dim 64) and SmolLM3 (θ = 5M, head_dim 128). In their native training lengths, how many hands do not make one full turn?

**Task 2 (core)**: In `03_context_extension.py`, add a setting (e): YaRN **without the temperature** (mscale fixed at 1). Measure it zero-shot and after fine-tuning, and compare it with (c). How much does the temperature help zero-shot? Is it still important after fine-tuning? Then change the random seed (change `train(..., seed=...)`), and check if the difference is larger than the variation between seeds.

**Task 3 (challenge)**: Use the method of `04_anneal_mixture.py` for a "microannealing"-style data valuation. Replace the "code" source with a small text that you find outside `assets/tiny_corpus/` (for example, a public-domain book). Add it only in the decay phase, at 30% (the method of Llama 3). How much does it help the bits-per-byte of its own validation set? Does it hurt the other sources? Then run needle in a haystack once with `uv run python -m zero.tools.needle --model out/tiny/midtrain/ckpt --lengths 64,128,240`, and explain why the tiny model gets 0 everywhere.

---

## Go deeper: CS336

This chapter matches Stanford CS336 (Spring 2026) <https://cs336.stanford.edu/>, with **partial coverage**:

- **Lecture 14: Data (filtering, deduplication, mixtures, synthetic data).** The parts on data mixtures and synthetic data are the theory behind "what data mid-training uses" in this chapter. The filtering and mixture experiments of Assignment 4 (Data) connect directly to the branched decay of this chapter.
- **Lecture 15: Mid-training and post-training (SFT/RLHF).** It explains where mid-training is in the full pipeline, and the boundary between mid-training and SFT.
- CS336 does not go deep into the long-context extension of RoPE (ABF, YaRN) or long-context evaluation. See the references of this chapter.

---

## References

- OLMo Team. *2 OLMo 2 Furious* (mid-training, Dolmino Mix, microannealing, checkpoint soup), 2024: <https://arxiv.org/abs/2501.00656>
- Llama Team. *The Llama 3 Herd of Models* (§3.1.3 annealing data, §3.4.2 long-context pretraining, §3.4.3 annealing, §4.3.4 long-context SFT), 2024: <https://arxiv.org/abs/2407.21783>
- Hugging Face. *SmolLM3: smol, multilingual, long-context reasoner* (blog): <https://github.com/huggingface/blog/blob/main/smollm3.md>
- Hu et al. *MiniCPM: Unveiling the Potential of Small Language Models with Scalable Training Strategies* (WSD, and "add high-quality data in the decay phase"), 2024: <https://arxiv.org/abs/2404.06395>
- Qwen Team. *Qwen3 Technical Report* (§3.2 three-stage pretraining, Appendix A.1.1 RULER), 2025: <https://arxiv.org/abs/2505.09388>
- Zhao et al. *MobileLLM-R1: Exploring the Limits of Sub-Billion Language Model Reasoners with Open Training Recipes*, 2025: <https://arxiv.org/abs/2509.24945>
- Luo et al. *PuRo-2B: Poor Lab's Qwen2-1.5B Trained on RTX 5090 within $5090*, 2026: <https://www.alphaxiv.org/abs/2608.27370> (already in `references.md`)
- Blakeney et al. *Does your data spark joy? Performance gains from domain upsampling at the end of training*, 2024: <https://arxiv.org/abs/2406.03476>
- Peng et al. *YaRN: Efficient Context Window Extension of Large Language Models*, 2023: <https://arxiv.org/abs/2309.00071>
- Chen et al. *Extending Context Window of Large Language Models via Positional Interpolation*, 2023: <https://arxiv.org/abs/2306.15595>
- Xiong et al. *Effective Long-Context Scaling of Foundation Models* (ABF), 2023: <https://arxiv.org/abs/2309.16039>
- Gemma Team. *Gemma 3 Technical Report* (§5.3 long context), 2025: <https://arxiv.org/abs/2503.19786>
- DeepSeek-AI. *DeepSeek-V3 Technical Report* (§4.3 long-context extension), 2024: <https://arxiv.org/abs/2412.19437>
- Hsieh et al. *RULER: What's the Real Context Size of Your Long-Context Language Models?*, 2024: <https://arxiv.org/abs/2404.06654>
- Kamradt. *Needle In A Haystack — Pressure Testing LLMs*, 2023: <https://github.com/gkamradt/LLMTest_NeedleInAHaystack>
- Hägele et al. *Scaling Laws and Compute-Optimal Training Beyond Fixed Training Durations* (WSD and branched decay, already cited in Chapter 6), 2024: <https://arxiv.org/abs/2405.18392>
- [CS336](https://cs336.stanford.edu/) Lectures 14 and 15

**Next chapter**: A base model can continue text, but it cannot "have a conversation" yet, and it cannot call tools in a fixed format. Chapter 16 explains SFT. It shows how to design a chat template (with the tool-calling format), and why we calculate the loss only on the reply. It also makes the main-line base model an assistant that follows instructions, for the first time.
