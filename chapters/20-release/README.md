# Chapter 20: Release — Report against the preregistration, and run the model on a laptop

**English** · [中文](README.zh.md)

> **Goal**: After this chapter, you can use the preregistered protocol to give an "ahead / tie / behind" decision for each item of the final model. You can explain what to write when the model does not reach its target. You can calculate how much memory any model needs in fp16 / Q8_0 / Q4_K_M. You can write block-wise INT8 / INT4 quantization by hand and measure how much it changes the loss. You know what a "release" must deliver (weights in a standard format, GGUF, a model card, a license, intermediate checkpoints, a local demo). You also know how to avoid exaggeration after the release.

📺 **Video**: Not published yet. To render it on your computer, run `bash chapters/20-release/video/build.sh`.
🧪 **Self-check**: After the chapter, type `/ch20-release` in Claude Code.

---

In the last chapter, we used GRPO in a tool-calling environment to train the final version of the model. All training work is now done. But the model is only a checkpoint folder on the hard disk of a server. Other people cannot use it, and nobody knows how good it is. This chapter covers the last part of the path:

1. **How good is the model?** Do the final evaluation with the preregistration protocol from Chapter 11. We locked the exam, the templates, the decoding parameters, and the decision rules before training. Now we only run them and apply them. This is **Gate 3**.
2. **How can other people use the model?** Export the weights to a standard format that all tools know. Then **quantize** them, so that the model runs on a normal laptop. Add a local tool-calling demo to prove that the model is "really usable".
3. **What do we say about the model?** Write a model card: data, licenses, recipe, cost, all evaluation results (also the items that we lost), and known limitations. Do not exaggerate a win. Do not hide a loss.

The code of this chapter (all of it runs on the CPU):

```bash
uv run python chapters/20-release/code/01_blockwise_quant.py     # block-wise INT8/INT4 quantization: error and size table (a few seconds)
uv run python chapters/20-release/code/02_quantize_tiny_model.py # apply it to the small model of Chapter 10 and see how much the loss changes (about 1–4 min on one thread)
uv run python chapters/20-release/code/03_memory_calculator.py   # memory of the main-line model in each format (1 s)
uv run python chapters/20-release/code/04_model_card.py          # make a model card skeleton from the evaluation JSON (1 s)
```

## 1. Gate 3: hand in the exam, do not change the grading

### 1.1 Intuition: the exam is sealed before the test starts

Think of an exam. If you select the subjects and the scoring method after the exam, almost anybody can "come first". You select the subjects that you are good at, and a scoring method that helps you. In model evaluation, this is **cherry-picking** (selecting results after the fact). It is the most common source of inflated claims such as "better than SOTA".

Our method is the **preregistration** from Chapter 11. Before the main-line pretraining starts, we write the items below into `eval/PREREGISTRATION.md` and commit the file. The commit time is the registration time (GOAL.md 3.2):

- the benchmarks and their versions (specialized: BFCL + one tool-calling benchmark with Chinese; general: knowledge, Chinese, math, code, instruction following);
- the evaluation framework and its version number, the prompts and templates, the decoding parameters;
- the list of opponents and the freeze date (open-weight models with an official parameter count between 0.7× and 1.3× of ours; Qwen3.5-0.8B is a required opponent);
- the criterion for "ahead".

Gate 3 does a plain job: **run this file as it is, and do not change one word**. After that, you can add a change only as an "amendment with a date and a reason".

### 1.2 Decision rule: on which side of 0 is the full confidence interval?

For the same set of questions, we and the opponent each have a score for each question: `a_i` and `b_i` (correct = 1, wrong = 0). How much can we trust the difference of the means `d = mean(a) − mean(b)`? Use the **paired bootstrap** from Chapter 11. Sample n questions with replacement (both models use the same indices) and calculate one d\*. Do this 10,000 times. The 2.5% and 97.5% quantiles give the 95% confidence interval `[lo, hi]`:

```
lo > 0                 → ahead  (the full interval is to the right of 0)
hi < 0                 → behind
otherwise (crosses 0)  → tie
```

The `decide` function in `zero/eval/bootstrap.py` does this:

```python
def decide(ci_low: float, ci_high: float) -> str:
    if ci_low > 0:
        return AHEAD      # ahead
    if ci_high < 0:
        return BEHIND     # behind
    return TIE            # tie
```

Two more rules apply (GOAL.md 3.2, item 4). If an opponent has a thinking mode and a non-thinking mode, we test both modes. Then we **use the higher score of the opponent** for the comparison (`compare_to_opponent`). We show the official scores of the opponents next to ours, but we do not use them for the comparison. We run all opponents again ourselves, with the same method.

### 1.3 What a decision table looks like (tiny-configuration demo)

> The content below is a **tiny-configuration demo**: a model with about 1.3M parameters and a few dozen toy questions. It only shows that the decision procedure works. It is not a result of the main-line model.

> **Note:** The numbers in this section come from a smoke test that we ran **before** the fix of the tool-call grader (Chapter 19, Section 6). They are the real output of that run. We ran the smoke test again after the fix (`uv run python -m zero.smoke --out out/smoke_final`). The data, pretraining, mid-training, and SFT stages gave exactly the same results. The number of distillation samples that passed verification changed from 1 (the nonsense sentence) to 0. Thus, the numbers of DPO, GRPO, and evaluation after it changed. For example, the tool-calling call_exact of the same SFT model changed from 0.133 to 0.100. The model did not change; the grader became stricter. GRPO against SFT is still "tie". When you run the test yourself, use the numbers from your run.

In the smoke test (`zero.smoke`), the "opponent" is our own SFT model. We compare the models after DPO and after GRPO with it (real output from `out/smoke/eval/report.md`):

| Model | Baseline | Task | Metric | Model score | Baseline score | Difference | 95% CI | Decision |
|---|---|---|---|---:|---:|---:|---|---|
| dpo | sft | toy_mc | acc | 0.200 | 0.167 | +0.033 | [+0.000, +0.100] | tie |
| dpo | sft | tool_dev | call_exact | 0.100 | 0.133 | −0.033 | [−0.100, +0.000] | tie |
| grpo | sft | toy_mc | acc | 0.300 | 0.167 | +0.133 | [+0.033, +0.267] | ahead |
| grpo | sft | tool_dev | call_exact | 0.100 | 0.133 | −0.033 | [−0.100, +0.000] | tie |

Look at two things in this table:

- **GRPO is "ahead" on the multiple-choice questions, but not on tool calling.** Tool calling is our hard goal. If we report only the third row, that is typical cherry-picking.
- With 30 questions, the confidence intervals are very wide (about ±0.1). In a real evaluation, BFCL has more than a thousand questions, so the intervals are much narrower. But **a Chinese subset with few questions can still fail to show a winner**. For this reason, the preregistration must state the "conditions for the hard goal" and the method for multiple comparisons in advance. Do not look for the subset that won afterwards and report only that subset.

### 1.4 Three outcomes, three ways to write them

| Outcome | What you can say | What you cannot say |
|---|---|---|
| The target group (hard goal) is "ahead" of each opponent | "Ahead of all models in the opponent list, including Qwen3.5-0.8B, on the preregistered BFCL (version X)" | "Better tool calling than all models of the same size" (we tested only these benchmarks) |
| Some items are "tie" | "Tie with Qwen3.5-0.8B (difference +0.8, 95% CI [−1.2, +2.9])" | "Slightly ahead": if the interval crosses 0, there is no evidence |
| Some items are "behind" | List the items where we are behind, and write a gap analysis (data? post-training? size?). GOAL.md 3.4 says that this analysis is also course content | Change the benchmark or the template and run again until we win |

At the end, there is one more section: **"Opponents added after release"**. At release, look again for new models of the same size that came out after the freeze date. Run them if you can, and report the results truthfully, **also if they are better than ours**. The freeze date protects the claim "we did not select the opponents afterwards". It does not mean "we are always first".

## 2. What a "release" delivers

GOAL.md 3.5 lists five items: the models (Base, SFT, final version, and key intermediate checkpoints), the quantized versions, a local demo, a model card, and a reproducible recipe. For the first two items, the key words are **standard format**.

### 2.1 Standard format: tools of other people can use it directly

A Hugging Face model folder is the de facto standard:

| File | Content | Who reads it |
|---|---|---|
| `config.json` | Architecture name (we write `Qwen3ForCausalLM`), number of layers, width, number of heads, RoPE / YaRN parameters | transformers, vLLM, the conversion script of llama.cpp |
| `model.safetensors` | Weights | Same as above |
| `tokenizer.json`, `tokenizer_config.json` | Tokenizer, and the **chat template** (a Jinja template) | `apply_chat_template`, the chat API of inference servers |
| `generation_config.json` | Default end token and sampling parameters | `generate()` |

Three details decide if people can use the model directly:

- **safetensors, not pickle**: The default PyTorch format `.pt/.bin` is pickle, and loading a pickle file can run any code. A safetensors file is only "an 8-byte header length + a JSON header (dtype, shape, and offset of each tensor) + contiguous raw bytes". It cannot hide code, and it supports zero-copy loading with mmap.
- **Use an architecture name that all tools support**: The zero model matches the Qwen3 dense model layer by layer (Chapter 9). Thus, the export writes `Qwen3ForCausalLM` directly. transformers, vLLM, and llama.cpp then recognize the model without a change to one line of code. The cost: we cannot freely invent a new structure. This is one more reason for the GOAL.md 3.3 rule "the main-line model takes no architecture risk".
- **The chat template must be identical to the training template, character by character**: A tool-calling model is very sensitive to the format. One extra newline can stop it from producing `<tool_call>`. We write the training template into `tokenizer_config.json` without changes. The smoke test renders it with `apply_chat_template` of transformers and compares it with our template character by character (`hf_template_identical=True`).

### 2.2 Intermediate checkpoints: for readers without compute

GOAL.md 3.5 requires the release of "key intermediate checkpoints". Then readers can start from our Base model and do the experiments of Chapters 16–19. Fully open models already do this as a convention. OLMo 2 releases the pretraining checkpoints as branches of the same repository (`revision="stage1-step140000-tokens294B"`). SmolLM3 has a separate repository, `SmolLM3-3B-checkpoints`: one checkpoint for each 40,000 pretraining steps and one for each post-training stage. We use the same method: the branch name gives the number of steps and tokens.

## 3. Quantization: why, and how

### 3.1 Intuition: memory = number of parameters × bits per parameter

A normal laptop has 8–16 GB of memory, and the system and other programs also need part of it. For inference, the model must keep at least all its weights in memory:

```
weight memory ≈ number of parameters × number of bits / 8
```

`03_memory_calculator.py` does this calculation tensor by tensor for the main-line model. It uses the provisional shape in `configs/main/pretrain.toml`: 689.5M parameters, of which 83.9M are embedding parameters.

| Format | bits/parameter | Weight size |
|---|---:|---:|
| fp32 | 32.00 | 2.57 GiB |
| bf16 / fp16 | 16.00 | 1.28 GiB |
| Q8_0 | 8.50 | 0.68 GiB |
| Q4_K_M | 5.00 | 0.40 GiB |

Inference also needs the KV cache (the formula from Chapter 21: `2 × layers × KV heads × head_dim × sequence length × 2 bytes`). The KV cache is 112 KiB for each token, 0.44 GiB for a 4K context, and 3.50 GiB for a 32K context. With a long context, the KV cache is much larger than the Q4 weights.

A 0.7B model also fits in memory in bf16. For this model, the main benefit of quantization is **speed**. When the CPU generates token by token, each step reads all weights from memory once. The bottleneck is memory bandwidth (the decode analysis in Chapters 10 and 21). With half the weight bytes, the read is almost 2× faster. For 7B and 70B models, quantization decides if the model fits in memory or not.

### 3.2 Block-wise quantization: one scale for each 32 numbers

The simplest quantization uses a scale factor `scale`. Divide each weight by the scale and round to an integer. To use the weight, multiply back:

```
scale = max|w| / qmax          (INT8: qmax = 127; INT4: qmax = 7)
q     = round(w / scale)       store this integer
ŵ     = scale · q              dequantize when you use it
```

The problem is in "find a scale": a scale for which numbers? Weights often contain a few **outliers** with a very large absolute value. If the whole matrix shares one scale, one outlier makes the scale large. Then millions of numbers of normal size all fall into the one or two levels near 0. The solution is **block-wise** quantization: along each row, each 32 numbers form one block, and each block has its own scale. The scale itself is stored in fp16. Over the numbers of a block, this costs 16/32 = 0.5 bit for each number.

The core of `01_blockwise_quant.py` is these lines:

```python
qmax = 2 ** (bits - 1) - 1                      # INT8 → 127, INT4 → 7
blocks = w.reshape(-1, block)                   # each row is one block (32 numbers)
absmax = np.abs(blocks).max(axis=1, keepdims=True)
scale = (absmax / qmax).astype(np.float16).astype(np.float32)  # store the scale in fp16
q = np.clip(np.round(blocks / scale), -qmax, qmax).astype(np.int8)  # q = round(w / scale)
...
return (q.astype(np.float32) * scale).reshape(shape)  # ŵ = scale · q
```

Try it on a 1024×1024 matrix: Gaussian, standard deviation 0.02, and 0.1% of the elements multiplied by 10 to act as outliers. We made this matrix to show the effect of outliers. The first 8 numbers of the first block:

```
w     : +0.0025 -0.0026 +0.0128 +0.0021 -0.0107 +0.0072 +0.0261 +0.0189
INT8  : scale=0.000366  q=[7, -7, 35, 6, -29, 20, 71, 52]   ŵ almost unchanged
INT4  : scale=0.006641  q=[0, 0, 2, 0, -2, 1, 4, 3]         ŵ=+0.0000 +0.0000 +0.0133 …
```

INT4 has only 15 levels, so both 0.0025 and −0.0026 become 0. The results for the whole matrix:

| Method | bits/weight | Size | Relative weight error | Relative error of the output y = xWᵀ |
|---|---:|---:|---:|---:|
| fp16 | 16.000 | 2048 KiB | 0.02% | 0.02% |
| INT8, one scale for the whole matrix | 8.000 | 1024 KiB | 7.36% | 7.38% |
| INT8, one scale for each row | 8.016 | 1026 KiB | 2.09% | 2.09% |
| INT8, blocks of 32 | 8.500 | 1088 KiB | 0.63% | 0.64% |
| INT4, one scale for the whole matrix | 4.000 | 512 KiB | 92.97% | 93.16% |
| INT4, one scale for each row | 4.016 | 514 KiB | 36.36% | 36.33% |
| INT4, blocks of 32 | 4.500 | 576 KiB | 11.32% | 11.43% |
| INT4, two-level scale (the Q4_K idea) | 4.500 | 576 KiB | 8.28% | 8.30% |

With "INT4, one scale for the whole matrix", the error is 93%. This is almost the same as setting the whole matrix to zero. Blocks cost 0.5 bit more, and the error decreases to 11%.

### 3.3 Blocks in GGUF: Q8_0, Q4_K, Q4_K_M

llama.cpp uses the **GGUF** file format. Its quantization types are engineering versions of the idea above (for the block layouts, see `ggml/src/ggml-common.h` in llama.cpp):

| Type | One block | Layout | bits/weight |
|---|---|---|---:|
| Q8_0 | 32 numbers | 1 fp16 scale + 32 int8 = 34 bytes | 8.5 |
| Q4_0 | 32 numbers | 1 fp16 scale + 16 bytes (32 × 4 bits) | 4.5 |
| Q4_K | super-block of 256 numbers | 8 sub-blocks (32 numbers each); each sub-block has a 6-bit scale and a 6-bit min; the super-block adds 2 fp16 values ("scale of the scales" and "scale of the mins"); 128 bytes of 4-bit integers; 144 bytes in total | 4.5 |
| Q6_K | 256 numbers | 16 sub-blocks, 8-bit sub-block scales + 1 fp16 | 6.5625 |

Q8_0 is the same format as our hand-written "INT8, blocks of 32". Q4_K adds two things. First, each sub-block has a min in addition to the scale (`ŵ = scale·q − min`), so it can represent a block that is not centered at 0. Second, Q4_K uses a **two-level scale**: the sub-block scales are themselves quantized to 6 bits. Thus, with the same 4.5 bits, Q4_K can also store a min. `quant_kstyle` in `01` is a simplified version (the real ggml implementation also searches for a better scale). The error decreases from 11.3% to 8.3%.

The most common type, **Q4_K_M**, does not use Q4_K for all tensors. It follows the rule in `src/llama-quant.cpp` of llama.cpp. The output layer (the token embedding when the embedding is shared) uses Q6_K. attn_v and ffn_down use Q6_K in the "use more bits" layers: the first 1/8, the last 1/8, and every third layer in the middle. All other matrices use Q4_K, and the 1D norm weights stay in f32. `03_memory_calculator.py` calculates each tensor with this rule.

To make sure that the calculator is correct, we used zero to randomly initialize a model with the same structure and 51.4M parameters (dim 512, 16 layers, tied embeddings). We sent it through `export_to_hf_qwen3` → `convert_hf_to_gguf.py` → `llama-quantize`. Then we used `gguf-py` of llama.cpp to read the real type and byte count of each tensor. For f16, Q8_0, and Q4_K_M, the total tensor bytes were **identical to the byte** to the calculator. In Q4_K_M, the type of each tensor also agreed with the calculator (17 Q6_K, 96 Q4_K, 65 F32).

For reference, the llama.cpp documentation gives 4.89 bits/weight for Q4_K_M and 8.50 for Q8_0 of Llama 3.1 8B. For our main-line model, the calculation gives 5.00. The value is a little higher because the tied embedding has 12% of the parameters, and the rule puts it in Q6_K.

### 3.4 Quality for size: measure it on a small model

An error of 11% sounds very bad. But the model does not care if each weight is exact. It cares if the final predictions change. `02_quantize_tiny_model.py` applies quantization to the character-level small model of Chapter 10 (0.86M parameters, dim=128, 4 layers). It quantizes all 2D matrices and keeps the 1D RMSNorm weights in fp32 (the same as llama.cpp). Then it measures the loss on the validation set and the "top-1 agreement". The top-1 agreement tells if the quantized model and the fp32 model predict the same most probable next character at each position:

> **Note:** About the numbers: the numbers of the training experiments in this chapter come from one CPU run on the course build machine. Different machines and different versions of the math libraries use a slightly different order of floating-point operations. After a few hundred training steps, these small differences become larger. Your numbers can be different from the second or third decimal place. Use the conclusions below that do not depend on exact values. For a re-run on another server in 2026-10, see [runs/2026-10-01-gpu0-check/chapters-16-20.md](../../runs/2026-10-01-gpu0-check/chapters-16-20.md).

| Method | Weight size | val loss | Δ loss | top-1 agreement |
|---|---:|---:|---:|---:|
| fp32 | 3365 KiB | 1.8385 | — | 100.0% |
| fp16 | 1685 KiB | 1.8385 | −0.0000 | 100.0% |
| INT8, one scale for the whole matrix | 845 KiB | 1.8386 | +0.0002 | 99.4% |
| INT8, blocks of 32 (≈Q8_0) | 897 KiB | 1.8384 | −0.0001 | 99.6% |
| INT4, one scale for the whole matrix | 425 KiB | 1.8525 | +0.0141 | 90.3% |
| INT4, one scale for each row | 436 KiB | 1.8446 | +0.0061 | 93.5% |
| INT4, blocks of 32 (≈Q4_0) | 477 KiB | 1.8435 | +0.0050 | 94.1% |
| INT4, two-level scale (≈Q4_K) | 486 KiB | 1.8437 | +0.0052 | 94.9% |
| INT3, blocks of 32 | 372 KiB | 1.8776 | +0.0391 | 85.9% |
| INT2, blocks of 32 | 267 KiB | 2.3439 | +0.5055 | 46.4% |

A re-run on another server in 2026-10 gave these results (the Chapter 10 base model was trained again on that machine, so its weights are a little different): fp32 1.8308; INT4 blocks +0.0066, 93.3%; INT4 two-level +0.0046, 94.6%; INT3 +0.0421, 85.4%; INT2 +0.5771, 44.1%. The weight size column is the same.

Some observations:

- **8 bits are almost lossless**: the loss changes in the fourth decimal place. The first 60 characters of greedy generation are identical to fp32.
- **4 bits start to have a cost**: the loss increases by a few thousandths (0.005; 0.007 in the re-run). About 1 position in every 10 to 20 gets a different most probable character (agreement 94.1%; 93.3% in the re-run). Greedy generation soon takes a different path from fp32. Here, the path changes at the 9th character ("And the course…" vs "And the such…"). In the re-run, fp32 generates "Ay so the soul, and the prove…" and INT4 blocks generate "Ay the send the prove…": the path changes already at the 4th character.
- **Below 4 bits is a cliff**: the cost of 3 bits is several times the cost of 4 bits (8× here; 6.4× in the re-run). The 2-bit model is almost useless. The loss jumps from 1.84 to 2.34, and the model generates "is is apeeng". In the re-run, the loss jumps from 1.83 to 2.41, and the model generates "How, would are a worlous…".
- This small model did not train for long, and its weights have no clear outliers. Thus, INT8 with "one scale for the whole matrix" also works. It is also not clear how much better the two-level scale is than normal blocks. In both runs, the two-level scale has a slightly higher agreement. But here, the loss does not show a winner (1.8437 vs 1.8435, agreement 94.9% vs 94.1%). In the re-run, the two-level scale is clearly better (1.8355 vs 1.8374, 94.6% vs 93.3%). These results are different from the conclusions for the matrix with outliers that we made above. **The effect of quantization depends on the specific weights, so you must measure it on your own model.** On a model this small, even "which of two methods is better" can change when you train the model again.

For reference on a large model, the perplexity documentation of llama.cpp gives results for Llama 3 8B on WikiText-2. The perplexity is 6.233 for f16, 6.234 for Q8_0, 6.407 for Q4_K_M (+0.175), and 6.700 for Q4_0 (+0.469). The pattern is the same: "8 bits are almost lossless, 4 bits have a visible cost, and K-quants are better than the old Q4_0". We cannot extrapolate from the 8B results to know if our 0.7B model is more sensitive. Before the release, measure the perplexity with `llama-perplexity` on our own development set. Also compare the scores of bf16 and Q4_K_M on the tool-calling development set, and write them in the model card (`runs/RELEASE_CHECKLIST.md` C7–C8).

### 3.5 Summary: three settings of quantization

- **Number of bits**: 8 bits are almost free. 4 bits are the common trade-off on a laptop. Be careful below 4 bits.
- **Block size**: a smaller block has fewer numbers that one outlier can damage, but the scales cost more (0.5 bit for each number with 32 numbers in each block).
- **Which tensors get more bits**: tensors that are sensitive to errors (the output layer, some attn_v / ffn_down) use a higher precision. This is the "M" in Q4_K_M.

All of this is **post-training quantization** of the weights. It does not change the training. Gemma 3 and other models also release QAT versions, which simulate quantization during training (for example `gemma-3-1b-it-qat-q4_0-gguf`). QAT is outside the scope of this chapter.

## 4. Run the model locally: llama.cpp / Ollama on a laptop, vLLM on a server

| Use case | Tool | Format | How we use it |
|---|---|---|---|
| Laptop, CPU, Mac | **llama.cpp** | GGUF | `llama-cli -m zero-Q4_K_M.gguf`; includes the OpenAI-compatible `llama-server` |
| One-step installation on a laptop | **Ollama** | GGUF | Write `FROM ./zero-Q4_K_M.gguf` in the `Modelfile`, then run `ollama create zero` |
| GPU server, many users at the same time | **vLLM** | HF folder (safetensors) | `vllm serve <folder> --enable-auto-tool-choice --tool-call-parser hermes` (PagedAttention, Chapters 10 and 21) |

The `hermes` parser of vLLM recognizes exactly the format `<tool_call>{"name": …, "arguments": …}</tool_call>`. The tool-calling templates of the Qwen series use this style, and so does our template (`zero/post/chat.py`). Thus, when vLLM serves the same exported folder, it returns structured `tool_calls` directly. We do not need a custom parser (RUNBOOK stage 6, item 10 checks this on a GPU).

The **local demo** is the direct proof that the model is "really usable". `zero/demo/cli.py` is a command-line assistant with this loop: user input → apply the template and generate → parse `<tool_call>` → run the tool locally → send the result back as a tool message → repeat until the model gives an answer without a tool call. The local tools are a calculator (evaluation with an AST allowlist, no `eval`), date calculation, today's date, and `search_files`. `search_files` searches only inside the `--root` folder. The resolved real path must be under root (this blocks `../` escapes). It does not follow symbolic links that point outside, and it reads at most the first 1 MB of each file (`tests/test_demo.py` covers these cases). When a model runs actions on your computer, **the limits of the tools are more important than the intelligence of the model**.

## 5. Model card and license

### 5.1 What a model card contains

A model card (proposed by Mitchell et al. in 2019) is the `README.md` on the front page of a model repository. It starts with a block of YAML metadata (`license`, `language`, `datasets`, `base_model`, `pipeline_tag`, ...), which Hugging Face uses for search and display. The text for people comes after it. The model cards of the leading model families all have these parts: architecture and parameter count, training data, usage, evaluation table, and limitations. (Compare the model cards of Qwen3, Llama 3.2, Gemma 3, OLMo 2, and SmolLM3; the links are at the end of the chapter.) GOAL.md 3.5 requires more: **the recipe and cost of each stage, the preregistration protocol, all evaluation results (also the items where we are behind), and the decontamination check**.

`04_model_card.py` makes a skeleton from the evaluation results JSON. Its rule is **fill in only numbers that have a source**. It makes the evaluation tables directly from the JSON. Where no data exists, it always writes "TBD after training". At the end, the script counts the placeholders that are left (the count must be zero before the release). Here is a run on the smoke-test results (tiny-configuration demo):

```
---
license: other  # undecided
language: [zh, en]
tags: [function-calling, tool-use, from-scratch, gguf]
library_name: transformers
pipeline_tag: text-generation
---
# zero-0.7b (working name)
> ⚠️ Tiny-configuration demo: the numbers below come from a CPU smoke test of a model with about 1.3M parameters...
## Model summary / Usage / Training data and licenses / Recipe and cost of each stage / Preregistration
## Evaluation results (all items, also the items where we are behind)
…Total: ahead 1, tie 5, behind 0 (all items are listed; none are selected).
### Opponents added after release / Decontamination check / Known limitations / Citation and acknowledgments
<!-- 15 'TBD after training' placeholders are left. Fill in all of them before release. -->
```

### 5.2 License: the author decides; here are the options

The author (the project lead) decides the license of the weights. The table below only lists the options and what each option means:

| Option | Who uses it | Meaning |
|---|---|---|
| Apache-2.0 | Qwen3, OLMo 2, SmolLM3 | Permissive: allows commercial use and modification; has a patent grant clause; you must keep the license and the copyright notice |
| MIT | DeepSeek-R1 and others | Shorter and more permissive: you must only keep the copyright notice |
| Custom community license | Llama 3.2 (`llama3.2`), Gemma (`gemma`) | Adds a use policy and conditions (for example, the number of users and the acceptable uses); not an open-source license in the OSI sense |

Our target is "the model, the data recipe, the code, and the intermediate checkpoints are all public" (GOAL.md Section 1). This agrees with the fully open families that use Apache-2.0 (OLMo, SmolLM). But the author confirms the final choice before it goes into the model card.

The **attribution duty for the data** is a different matter from the weight license. We must do it, whatever license the weights get. These are the licenses of the candidate pretraining data (from the Hugging Face data set cards). FineWeb-Edu, FineWeb-2, and FineMath use **ODC-By 1.0** (Open Data Commons Attribution: you can use the data, but you must give attribution). The FineWeb cards also say that use is subject to the Common Crawl terms of use. DCLM-baseline uses **CC-BY-4.0** (it also requires attribution). Stack-Edu comes from The Stack v2; its data set card has no single license tag, so we must follow the license of the original code (to be verified).

For the teacher model of distillation, record its name, its version, and if its license allows "training other models on its outputs" (Chapter 17). The data table of the model card lists all of these.

## 6. After the release: collect feedback, do not exaggerate

- **Make a place to collect failure examples** (an issue template: input, tool definitions, model output, expected output). The tool-calling failures of small models often fall into a few types: wrong argument types, no call when a call is necessary, and made-up tool results. These failures guide the next version better than the evaluation table.
- **When other people get scores that are different from ours**, first align the framework version, the template, and the decoding parameters (this is why the preregistration fixes them). After you confirm the cause, update the model card and **write the date and the reason**.
- **The words in announcements follow the decision table**: say "ahead" only for the "ahead" cells, and give the benchmark, the version, and the opponent list. For "tie", say "tie". Put the "behind" items in the limitations. Never use the numbers of the tiny-configuration demo in announcements.

## 7. Summary

- **Gate 3** = run the preregistered protocol and decide with the confidence interval: `lo > 0` is ahead, `hi < 0` is behind, all other cases are a tie. List all results. If we do not reach the target, do not claim "ahead". Also report the opponents added after the release truthfully.
- **Standard format**: safetensors + config + tokenizer + chat template, with an architecture name that is compatible with Qwen3. transformers / vLLM / llama.cpp can use it directly.
- **Quantization**: memory = parameters × bits. Use blocks with one fp16 scale for each block (Q8_0 = 8.5 bits, Q4_K = 4.5 bits). Q4_K_M gives more bits to sensitive tensors. 8 bits are almost lossless, 4 bits have a cost, and lower is a cliff. Measure it on your own model.
- **Local run**: llama.cpp / Ollama (GGUF) on a laptop, vLLM (safetensors) on a server. The tool limits of the demo must be strict.
- **Model card**: fill in only numbers that have a source; write "TBD after training" where no source exists. The author decides the license. Do the attribution duty for the data.

---

## GPU measurements (one RTX 3090)

> All numbers in the chapter text above come from CPU runs. This section uses one NVIDIA GeForce RTX 3090 (24 GB of GPU memory, Ampere architecture). Data sheet: dense BF16 tensor-core peak about 71 TFLOPS, FP32 about 35.6 TFLOPS, memory bandwidth about 936 GB/s. Environment: PyTorch 2.11.0+cu128, CUDA 12.8, October 2026. The server sets the power limit of this card to 240 W (the factory default is 350 W). Under a continuous full load, the card lowers its clock. Thus, the absolute compute and bandwidth values are lower than on a 3090 at full power, and the relative values are more reliable. If you do not have a GPU, skip this section.

Run:

```bash
uv run python chapters/20-release/code/05_gpu_quant_matvec.py
```

The script builds one batch-1 decode step from all 197 matrices of the main-line model (28 layers × 7 + output layer, 689.4M parameters, without the 1D norm weights). It calculates only the matrix multiplications (for attention and the KV cache, see Chapter 21). A CUDA Graph records the full step for the timing. The weights are random (N(0, 0.02²)), so the error column is only for comparison. For reference, a copy of a 1 GiB tensor on this card measured 851 GB/s (91% of the data sheet value).

| Format and kernel | bits/weight | Weight bytes | Time per step | Effective bandwidth | Speedup vs BF16 cuBLAS | Relative output error |
|---|---:|---:|---:|---:|---:|---:|
| BF16, cuBLAS (`F.linear`) | 16.00 | 1.379 GB | 2.567 ms | 537 GB/s | 1.00× | — |
| BF16, kernel generated by torch.compile | 16.00 | 1.379 GB | 2.589 ms | 533 GB/s | 0.99× | — |
| INT8, one scale for each row, torch.compile fuses dequantization into the multiply-add | 8.01 | 0.690 GB | 2.294 ms | 301 GB/s | 1.12× | 0.82% |
| INT4, groups of 32, tinygemm kernel | 5.00 | 0.431 GB | 1.157 ms | 372 GB/s | 2.22× | 7.83% |
| INT8, dequantize to a BF16 copy of the weights first, then multiply | 8.01 | 0.690 GB | 11.527 ms | 60 GB/s | 0.22× | 0.82% |

Time per step when the model generates B sequences at the same time:

| B | BF16 cuBLAS | INT4 tinygemm | INT4 speedup vs BF16 |
|---:|---:|---:|---:|
| 1 | 2.622 ms | 1.156 ms | 2.27× |
| 8 | 3.440 ms | 1.676 ms | 2.05× |
| 32 | 3.365 ms | 4.084 ms | 0.82× |
| 128 | 5.474 ms | 14.030 ms | 0.39× |
| 512 | 15.380 ms | 54.015 ms | 0.28× |

Effective bandwidth (GB/s) at batch 1 for one matrix type at a time (one product in each of the 28 layers, mean):

| Matrix | Shape N×K | BF16 size | BF16 cuBLAS | INT8 compiled | INT4 tinygemm |
|---|---:|---:|---:|---:|---:|
| wk | 1024×1280 | 2.5 MiB | 429 | 224 | 214 |
| w_up | 3584×1280 | 8.8 MiB | 520 | 285 | 404 |
| lm_head | 65536×1280 | 160.0 MiB | 683 | 314 | 632 |

On a GPU, "smaller weights give faster decode" is true, but the gain depends on the kernel. INT8 reads half the bytes, but with a general compiled kernel it is only 1.12× faster. The optimized tinygemm kernel reads less than one third of the bytes and is 2.22× faster. If you first dequantize all weights to BF16 and then multiply, the step becomes 4.5× slower. The real work of engines such as llama.cpp and vLLM is in kernels that "dequantize while they read".

The small size of the model is another reason for the gap to the theoretical lower bound. Each matrix has only a few MiB, and a smaller matrix uses less of the bandwidth (wk reaches only 429 GB/s, but a copy of a large tensor reaches 851 GB/s). The fixed cost of each kernel uses up part of the savings. With a larger batch, the result reverses: from B = 32, INT4 is slower than BF16, and at B = 512 it reaches only 0.28×. The bottleneck is now compute, and dequantization is extra work. Thus, quantization mainly speeds up "one person who generates token by token on a laptop". A server with many users at the same time (vLLM in Section 4) is a different calculation.

## From minimal code to production code

| Minimal code (`code/`) | Production code (`zero/`) | What it adds, and why |
|---|---|---|
| `01` block-wise quantization by hand | `llama-quantize` of llama.cpp (`quantize` in `zero/export/gguf.py`) | Real k-quants search for better scales, mix types by tensor (Q4_K_M), and have fast CPU/GPU kernels. We do not write them again; we call the official tool |
| — | `export_to_hf_qwen3(model, config, out_dir, tokenizer, dtype, chat=True)` in `zero/hf.py` | Maps the parameter names to the HF Qwen3 names. Does not save `lm_head` a second time when the embedding is shared. Writes `config.json` (both the new and the old RoPE/YaRN fields), `generation_config.json` (the end token of the chat model is `<|im_end|>`), the tokenizer, and the chat template |
| — | `convert_hf_to_gguf` / `build_llama_cpp` / `quantize` / `run_llama` / `llama_tokenize` in `zero/export/gguf.py` | Uses the **official** `convert_hf_to_gguf.py` of llama.cpp with only one runtime patch. The "pre-tokenizer hash" of our own tokenizer is not in the official table. When the script does not recognize the hash, it uses `qwen2` (our pre-tokenizer regex is the same as the Qwen2 regex). We do not change any file of llama.cpp |
| `03` memory calculator | `zero/tools/count_params.py`, `zero/tools/kv_cache_calc.py` | The same calculation, connected to the main-line config |
| `04` model card generator | `results_table` / `comparison_table` / `write_report` in `zero/eval/report.py` | Evaluation results → Markdown tables; the model card and the course text quote them directly |
| — | `zero/eval/harness.py`, `zero/eval/bootstrap.py`, `zero/eval/bfcl.py` | Internal evaluation, paired bootstrap and decisions; BFCL adapter (not verified yet) |
| — | `zero/demo/cli.py` | Local tool-calling assistant: generate → parse → run → send back; `search_files` stays inside root |

**Parity checks** (they make sure that the exported files and the trained model are the same model):

- `tests/test_export_hf.py`: after the export, transformers loads the zero model, and the logits agree with zero (fp32, `rtol=atol=1e-5`). The test covers three configurations: tied embeddings, untied embeddings, and YaRN. zero can also read the exported folder back.
- `tests/test_gguf.py`: `llama-tokenize` encodes four kinds of text (Chinese, English, code, tool calls) with the GGUF from the official script. The token IDs are identical, one by one, to the IDs of our tokenizer. `llama-simple` generates greedily with the f32 GGUF, and the result agrees with zero token by token (the model has a YaRN config). After quantization to Q8_0 and Q4_K_M, the models run.
- `tests/test_demo.py`: `search_files` blocks `../` escapes and symbolic-link escapes; the loop generate → run → send back; the tiny model runs end to end.

While we wrote this chapter, we **checked all of this again** with the existing files in `out/smoke/` and a compiled llama.cpp (commit `81bc6b8`). All of the following is real output:

- The prompt `<|im_start|>user\n3 * (4 + 5) 等于多少？<|im_end|>\n<|im_start|>assistant\n` (the user asks "What is 3 * (4 + 5)?"): on `zero-tiny-Q8_0.gguf`, `llama-tokenize` gives 20 token IDs that are identical to the IDs of the zero tokenizer (`[1, 449, 214, 34, 773, …, 1, 575, 214]`).
- Greedy generation of 24 tokens for the same prompt: zero (fp32 HF folder), `zero-tiny-f16.gguf`, and `zero-tiny-Q8_0.gguf` give **identical** output: `<tool_call>\n{"name":233 = 2 = 14 = 24 = 18)`. The tiny model only learned "output `<tool_call>` first", and the rest is random. This is the expected level for 1.3M parameters.
- One of these 24 steps was very close. At the 23rd token, zero gives `8` a logit of 7.888, and the second choice `4` has 7.862. The difference is only 0.026, and Q8_0 still selects the correct token.
- We also quantized `zero-tiny-f16.gguf` to Q4_K_M. The row length of the tiny model is 128, which is not a multiple of 256, so k-quants cannot be used. `llama-quantize` automatically falls back to 25 Q5_0 + 5 Q8_0 tensors (6.39 bits/weight, 1.12 MB). The model runs, but its output takes a different path from f16 at the 10th token. It outputs `<tool_call>\n{"name":233 =）。` and then stops. On this undertrained small model, even a 5–8-bit mix changes the greedy path. The width of the main-line model is 1280 = 5 × 256, so it has no fallback problem.
- On this shared machine, llama.cpp on one thread runs at only 0.14 token/s (the CPU is very busy). This value **does not show** the speed on a laptop. At release, measure the real speed with checklist item C10.

---

## Main-line progress

### Tiny-configuration demo (CPU, `configs/tiny`, about 1.3M parameters)

> The content below is a **tiny-configuration demo**. It only shows that the code paths of the export, the quantization, and the demo work. It does not show any result of the main-line model.

Real output from `uv run python -m zero.smoke` (`out/smoke/SUMMARY.md`, one thread, export stage 57.7s):

| Item | Result |
|---|---|
| HF export (`hf_chat/`) | After transformers loads the model, the maximum logits difference is `0.0e+00` (fp32 export); `apply_chat_template` is identical to our template, character by character |
| GGUF f16 | 2.7 MB |
| GGUF Q8_0 | 1.47 MB |
| llama-simple test run | Success (`llama_simple_ok=True`) |
| Local demo | Runs, but **the answer is empty** (`answer=`): the model produced the end token directly and did not call a tool |

We must report the empty demo answer truthfully. With a long prompt ("system prompt + tool list + user question"), the tiny model produced `<|im_end|>` as its first token. While we wrote the chapter, we asked the same model again: "3 * (4 + 5) 等于多少？" (`--root chapters/20-release`). The answer was still empty. In the evaluation table, its `tool_dev call_exact` is only 0.10–0.13 (30 questions), which agrees with this result. The demo of Step 1 only proves that the chain "load → generate → parse → run → send back" works. A really useful assistant must wait for the main-line model of Step 2.

### To be added after GPU training

- The final evaluation table with the preregistered protocol (each opponent × each benchmark, with confidence intervals and decisions), whether the hard goal is met, and a gap analysis;
- Opponents added after the release;
- The hit rates and removed counts of the decontamination check;
- The real GGUF size of the main-line model; the PPL, KLD, and tool-calling scores of Q8_0 / Q4_K_M relative to bf16; the memory and speed on a laptop;
- A screen recording of the local demo (examples of success and failure);
- Links to the Hugging Face repositories (Base, SFT, final version, intermediate checkpoints, GGUF) and the final model card;
- The cost (GOAL.md 3.4 gives Chapter 20 a budget of about $200, mainly for the final evaluation).

### Gate 3 checklist (GOAL.md 3.4: before release)

For the full checklist, see [`runs/RELEASE_CHECKLIST.md`](../../runs/RELEASE_CHECKLIST.md) (A evaluation, B decontamination, C weights and formats, D demo, E license, F release files). The main points:

- [ ] Run all evaluations with the benchmarks, framework versions, templates, and decoding parameters that `eval/PREREGISTRATION.md` froze. Save the result of each question.
- [ ] Do a paired bootstrap against each opponent (use the higher of the thinking / non-thinking modes of the opponent). List all "ahead / tie / behind" results.
- [ ] If the model does not reach the hard goal, do not claim "ahead". Write a gap analysis.
- [ ] After the release, look again for new models released after the freeze date. Write "Opponents added after release".
- [ ] Write the decontamination results (13-gram, tool function names / schemas) in the model card.
- [ ] Do the parity checks again on the final weights: HF logits, `llama-tokenize`, and f32 GGUF greedy generation. Measure the PPL and tool-calling scores of Q8_0 / Q4_K_M.
- [ ] Run Q4_K_M on a normal laptop with llama.cpp and Ollama. Record the screen during the demo.
- [ ] The author selects the license, and the data attribution table is complete. No "TBD after training" is left in the model card.
- [ ] Record the cost in `runs/ledger.md`.
- [ ] **Wait for the final confirmation of the project lead before the release.**

---

## Adopters and sources

| Technique / practice | Category (GOAL.md 2.1) | Adopters and sources |
|---|---|---|
| GGUF + llama.cpp local inference and quantization | B industry standard | llama.cpp ([GitHub](https://github.com/ggml-org/llama.cpp); [GGUF specification](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md)). Families that release official GGUF files: Qwen ([Qwen3-0.6B-GGUF](https://huggingface.co/Qwen/Qwen3-0.6B-GGUF)), Google ([gemma-3-1b-it-qat-q4_0-gguf](https://huggingface.co/google/gemma-3-1b-it-qat-q4_0-gguf)), Ai2 ([OLMo-2-0425-1B-GGUF](https://huggingface.co/allenai/OLMo-2-0425-1B-GGUF)); Ollama imports GGUF directly ([import documentation](https://github.com/ollama/ollama/blob/main/docs/import.mdx)) |
| safetensors weight format | B industry standard | [safetensors](https://github.com/huggingface/safetensors); the official repositories of Qwen3, OLMo 2, Llama 3.2, Gemma 3, and SmolLM3 all release safetensors (see the file lists of the model card links below) |
| vLLM serving + tool-call parsing | B industry standard | [vLLM](https://github.com/vllm-project/vllm), Kwon et al. *PagedAttention* ([arXiv:2309.06180](https://arxiv.org/abs/2309.06180)); [tool calling documentation](https://github.com/vllm-project/vllm/blob/main/docs/features/tool_calling.md) (`--enable-auto-tool-choice`, the `hermes` parser for Qwen) |
| Weight quantization (blocks, k-quants) | B industry standard (GOAL.md 2.1, inference row) | Block layouts: llama.cpp `ggml/src/ggml-common.h`; tensor rules of Q4_K_M: `src/llama-quant.cpp`; quality / size data: llama.cpp [`tools/quantize/README.md`](https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md), [`tools/perplexity/README.md`](https://github.com/ggml-org/llama.cpp/blob/master/tools/perplexity/README.md); original k-quants PR [#1684](https://github.com/ggml-org/llama.cpp/pull/1684) |
| Model card (YAML metadata + data, evaluation, limitations) | A adopted by many | [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B), [Llama-3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B), [gemma-3-1b-it](https://huggingface.co/google/gemma-3-1b-it), [OLMo-2-0425-1B](https://huggingface.co/allenai/OLMo-2-0425-1B), [SmolLM3-3B](https://huggingface.co/HuggingFaceTB/SmolLM3-3B); [HF model card documentation](https://huggingface.co/docs/hub/model-cards); Mitchell et al. *Model Cards for Model Reporting* ([arXiv:1810.03993](https://arxiv.org/abs/1810.03993)) |
| Release of intermediate checkpoints | A (fully open families) | OLMo 2 (`revision` branches in the same repository, [model card](https://huggingface.co/allenai/OLMo-2-0425-1B)), SmolLM3 ([SmolLM3-3B-checkpoints](https://huggingface.co/HuggingFaceTB/SmolLM3-3B-checkpoints)), Pythia (Biderman et al., [arXiv:2304.01373](https://arxiv.org/abs/2304.01373), 154 checkpoints for each model) |
| Separate releases of Base / SFT / final version | A adopted by many | OLMo 2 (four repositories: Base, SFT, DPO, Instruct), Qwen3 ([Qwen3-0.6B-Base](https://huggingface.co/Qwen/Qwen3-0.6B-Base) and Qwen3-0.6B), Gemma 3 (`-pt` and `-it`), SmolLM3 (Base and 3B) |

## Frontier notes

> **Quantization-aware training (QAT)**: in the last part of training, the forward pass simulates the quantization error, so that the model "gets used to" 4 bits. Then the team releases the quantized version. Google released QAT versions of Gemma 3 as Q4_0 GGUF files. gpt-oss releases its MoE weights in MXFP4 (the details of its training process are to be verified). QAT needs a changed training process and extra compute, and few teams use it yet. Our main line uses post-training quantization (PTQ).
>
> **Importance matrix (imatrix)**: a short calibration text measures the effect of each weight on the output. Quantization then protects the important weights first. In the perplexity documentation of llama.cpp, imatrix decreases the ΔPPL of Llama 3 8B Q4_K_M from 0.175 to 0.151. Community quantized versions use it widely. But the choice of calibration data adds a new variable (for example, overlap with the evaluation sets). Our release does not use imatrix at first. If we use it, the calibration data must also go through the decontamination check.

## Guided questions

1. Suppose we are "ahead" of Qwen3.5-0.8B on the BFCL total score but "behind" on the Chinese tool-calling benchmark. With the preregistration, how must we write the conclusion? What if the preregistration does not cover this case? (Hint: can an amendment fix the problem?)
2. Why must the paired bootstrap use the same question indices for the two models? If you resample each model independently, does the confidence interval become wider or narrower?
3. In `01_blockwise_quant.py`, change the block size from 32 to 16, 64, and 128. How do the INT4 error and the bits/weight change? Find the trade-off point between error and size.
4. At a 32K context, the KV cache (3.5 GiB) is much larger than the Q4_K_M weights (0.40 GiB). Which methods can make the KV cache smaller? (Hint: Chapter 21; `--cache-type-k q8_0` of llama.cpp.)
5. Why does Q4_K_M give more bits to attn_v, ffn_down, and the output layer? Ask Claude Code to help you keep only one type of tensor at 8 bits in `02`. Find which type has the largest effect on the loss.
6. The model license is Apache-2.0, but the pretraining data uses ODC-By and CC-BY. Do these two conflict? Who has the attribution duty, and where must the attribution be written?

## Hands-on tasks

**Task 1 (basic)**: Run `03_memory_calculator.py configs/ladder/l300m.toml` (the 300M model of the ladder experiment) and get its size in Q4_K_M. Then change the vocabulary of `configs/main/pretrain.toml` from 65,536 to 151,936 (the Qwen3 vocabulary) and calculate again. Do not change the file in the repository; use a copy. How much does Q6_K for the embedding increase the total size?

**Task 2 (core)**: In `02_quantize_tiny_model.py`, add a "mixed precision" method: INT8 blocks for the embedding and INT4 blocks for all other matrices. This copies how Q4_K_M gives more bits to the output layer. Compare its loss and size with pure INT4 blocks. Then try the opposite (embedding INT4, all others INT8). Which method gives the better trade-off?

**Task 3 (challenge)**: Do this task if you ran `uv run python -m zero.smoke` and compiled llama.cpp with `zero/export/gguf.py`. Use `llama-quantize` to quantize `out/smoke/gguf/zero-tiny-f16.gguf` to Q4_0, Q5_0, and Q8_0. Use `llama-simple` to generate greedily on the same prompt. For each format, record the token at which the output takes a different path from f16. Then make a model card with `04_model_card.py --out`, and complete the "Known limitations" section with your results.

## Go deeper: CS336

This chapter matches Stanford CS336 (Spring 2026) <https://cs336.stanford.edu/>:

- **Lecture 10: Inference**. The compute and bandwidth calculation of inference (prefill and decode), the KV cache, and methods such as quantization that decrease the cost of inference. This lecture is the background of "why quantize, and why 4-bit generation is faster" in this chapter.
- **Lecture 12: Evaluation**. How to design benchmarks, how to avoid contamination, and how to compare models. This lecture is the background of Gate 3 and the decision rules in this chapter (Chapter 11 of this course is an introduction to it).
- CS336 does not go deep into the exact GGUF / k-quant formats, model cards and licenses, or the release process. See the references of this chapter.

---

## References

- ggml-org. *llama.cpp*: <https://github.com/ggml-org/llama.cpp> (this chapter checked commit `81bc6b8`; the block layouts in `ggml/src/ggml-common.h`, the Q4_K_M rules in `src/llama-quant.cpp`, and the size and perplexity tables in `tools/quantize/README.md` and `tools/perplexity/README.md`)
- ggml-org. *GGUF file format*: <https://github.com/ggml-org/ggml/blob/master/docs/gguf.md>
- ggml-org. *k-quants* (PR #1684): <https://github.com/ggml-org/llama.cpp/pull/1684>
- Hugging Face. *safetensors*: <https://github.com/huggingface/safetensors>
- Hugging Face. *Model Cards* (Hub documentation): <https://huggingface.co/docs/hub/model-cards>
- Mitchell et al. *Model Cards for Model Reporting*, 2019: <https://arxiv.org/abs/1810.03993>
- Kwon et al. *Efficient Memory Management for Large Language Model Serving with PagedAttention* (vLLM), 2023: <https://arxiv.org/abs/2309.06180>
- vLLM. *Tool Calling*: <https://github.com/vllm-project/vllm/blob/main/docs/features/tool_calling.md>
- Ollama. *Importing a model*: <https://github.com/ollama/ollama/blob/main/docs/import.mdx>
- Dettmers et al. *LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale* (the outlier problem), 2022: <https://arxiv.org/abs/2208.07339>
- Frantar et al. *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers*, 2022: <https://arxiv.org/abs/2210.17323>
- Groeneveld / OLMo Team. *2 OLMo 2 Furious* (fully open release: data, code, checkpoints), 2024: <https://arxiv.org/abs/2501.00656>
- Biderman et al. *Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling*, 2023: <https://arxiv.org/abs/2304.01373>
- Model card examples: [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B), [Llama-3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B), [gemma-3-1b-it](https://huggingface.co/google/gemma-3-1b-it), [OLMo-2-0425-1B](https://huggingface.co/allenai/OLMo-2-0425-1B), [SmolLM3-3B-checkpoints](https://huggingface.co/HuggingFaceTB/SmolLM3-3B-checkpoints)
- Data set licenses: [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) (ODC-By), [FineWeb-2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) (ODC-By), [FineMath](https://huggingface.co/datasets/HuggingFaceTB/finemath) (ODC-By), [DCLM-baseline](https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0) (CC-BY-4.0), [Stack-Edu](https://huggingface.co/datasets/HuggingFaceTB/stack-edu) (to be verified); [ODC-By 1.0 full text](https://opendatacommons.org/licenses/by/1-0/)
- [CS336](https://cs336.stanford.edu/) Lectures 10 and 12

**Next chapter**: Part 4 ends here. The main-line model changed from "a set of weights" into an assistant that calls tools on a laptop. Part 5 goes in a different direction. We calculated above that at a 32K context, the KV cache needs 3.5 GiB, almost 9× the size of the 4-bit weights. Chapter 21 starts from this "ledger of the KV cache": it shows how MQA, GQA, and MLA make the KV cache smaller, step by step.
