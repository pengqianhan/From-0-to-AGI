# GPU0 verification: Chapters 24–26

**English** · [中文](chapters-24-26.zh.md)

Date: 2026-10-01. Branch `gpu0-verification`. Scope: `chapters/24-mixture-of-experts`, `chapters/25-mtp-speculative-decoding`, `chapters/26-open-model-panorama`.

## Environment

- CPU: AMD Ryzen Threadripper PRO 3995WX (Zen 2, AVX2, no AVX-512). This is a shared server. During the runs, the load average changed between 4 and 23 (other agents and video renders ran at the same time).
- PyTorch 2.11.0+cu128, CUDA 12.8. transformers is the version in the .venv of the repository. (`03_meta_params.py` of Chapter 26 recognizes all new architectures; no package is missing.)
- GPU: NVIDIA GeForce RTX 3090, only GPU0 (`CUDA_VISIBLE_DEVICES=0`, in a queue through `gpu0.lock`). Before the first use, I made sure that `torch.cuda.device_count() == 1`. `nvidia-smi -i 0` shows a power limit of **240 W** (default 350 W), so the absolute GPU values can be low. I did not use GPU1 or GPU2.
- CPU reproduction command: `CUDA_VISIBLE_DEVICES= UV_NO_SYNC=1 uv run python chapters/.../code/xx.py`, at most 4 processes at the same time.
- Raw output: `/tmp/claude-1006/-home-phan635-Opensource-From-0-to-AGI/1d40fca9-4469-49a7-88b9-4e96bf8f0051/scratchpad/ch24-26/` (`cpu_*.log`, `cpu_times.txt`; final GPU output `gpu_24_final.log`, `gpu_25_final.log`; `gpu_gmm_profile.log` is the profiler probe for `F.grouped_mm`).

## 1. CPU reproduction table

"CPU time" is the wall-clock time of the full `uv run` process. I measured it in parallel on a busy machine, so use it only for the order of magnitude. At the start, `out/` had no cached weights. Thus Chapter 24 03 and Chapter 25 01/05 were all **trained again from the start** on this machine. All 12 scripts returned rc=0, with **no errors**.

| Ch. | Script | CPU time | Category | Notes |
|---|---|---:|---|---|
| 24 | `01_param_ledger.py` | <0.1 s | Match | The FFN share table (41.9M / 151.0M / 78.3%, 7.9M / 13.8M / 63.6%), the total / active parameters of the 9 MoE models, and the notes on counting differences (351.2B / 32.1B, 5.13B / 3.61B) are all correct |
| 24 | `02_moe_layer.py` | 2.1 s | Mismatch (rounding level) | Only the sigmoid max difference of the float64 parity check: README 1.4e-16 → measured 1.7e-16. The parameter ledger, 1.2α = 0.0120, and the bias-balancing table (4.00 / 3.87 / 1.05, 50.3% / 32.8% / 0.0%) are all correct |
| 24 | `03_train_compare.py` | ≈ 37 min single-thread equivalent (3 processes in parallel, 15 min) + 2 s for the summary | **Mismatch** | After retraining, the validation loss and the load numbers drift a little. One conclusion in the text is no longer true. See Detail 1 |
| 24 | `04_small_vs_large.py` | <0.1 s | Match | The lineups, Qwen3 Table 5, the memory table, and the table of read fractions are all correct |
| 25 | `01_models_and_cost.py` | 118 s (includes 81 s to train the target and 28 s to train the draft) | **Mismatch** + timing | Validation loss of the target model: README 1.838 → measured **1.831**. Draft: 57,600 parameters, 1.966, and 15:1 are all correct. The T table is timing (2.69 / 2.71 / 2.95 / 3.15 / 3.49 / 4.01 ms, c ≈ 0.22). See Detail 2 |
| 25 | `02_greedy_speculative.py` | 51 s | **Mismatch** (small) + timing | For all 4 values of k, "identical to greedy decoding of the target model: True" ✓. The acceptance rate / output per round / target forward passes change a little with the target weights. See Detail 2 |
| 25 | `03_speculative_sampling.py` | 59 s | Parts 1 and 2 match; part 3 is a **mismatch** (small) + timing | One step of rejection sampling (TV 0.0013, chi-square 4.6, p-value 0.46, 0.5722 / 0.5714) and the Markov chain (0.0083, 50.5, 49, 0.41, 1.50) match character for character. The acceptance rate of the real models at temperature 1 changed. See Detail 2 |
| 25 | `04_speedup_formula.py` | <0.1 s | Match | Calculation only. The table for c = 0.05, 3.69×, and 6.86× are all correct |
| 25 | `05_mtp.py` | 376 s (includes training of 3 models) | **Mismatch** + timing | The MTP comparison table and the self-speculation acceptance rate drift a little. The conclusions do not change. See Detail 2 |
| 26 | `01_panorama.py` | 6.7 s | Match | The layer composition, the large comparison table (every cell), 11.5% / 9.62 GiB / 83.88 GiB, the adoption matrix, and the KV ledger of the next version (3.50 GiB / 1.20 GiB / 896.00 MiB / 22.48 MiB) are all correct. The tree and the tables in the README only remove rows and use abbreviations |
| 26 | `02_evolution_tree.py` | 1.7 s | Match | Every node of the tree, the status, and "used by" match. (The "used by" lists in the README omit the two small Qwen models. This is an abbreviation for the layout.) |
| 26 | `03_meta_params.py` | 5.8 s | Match | Every cell of the parameter table (116.829B / 5.71B / 5.13B / 4.55B …, 0.6895B, 2.72T / 48.6B / 5.45T) is correct. The script reads only the local `models.json` and does not use the network |

Totals (12 scripts): 6 match (24-01, 24-04, 25-04, and the 3 scripts of Chapter 26). 1 mismatch at the rounding level (24-02). 5 mismatches from numerical drift after retraining (24-03, 25-01, 25-02, 25-03 part 3, 25-05). 0 errors.

## 2. Details of the mismatches

The common cause: all these numbers come from **small models trained from the start on a CPU**. The README numbers come from training on a different machine. On this machine (Zen 2 / AVX2), the order of the floating-point operations is different. With the same seed and the same code, the trained weights are a little different. The evidence:

- The load distribution of Chapter 24 03 at step 0 and step 100 is **bit-identical** to the README. It starts to differ only at step 300.
- The validation loss of the Chapter 25 draft model (1 layer), 1.966, is the same as in the README. But the 4-layer target model gives 1.831.
- On the same machine, the result is deterministic. My process trained the target model. At the same time, another agent trained the same model and overwrote `chapters/10-inference/code/out/tiny_kv4_s600.pt`. Both models have a validation loss of 1.830811.

This is not a bug in the code.

### Detail 1: Chapter 24 `03_train_compare.py`

| Variant | README validation loss, mean [seed 0, 1] | Measured | README load (mean, worst) | Measured |
|---|---|---|---|---|
| Dense-384 | 1.793 [1.800, 1.787] | 1.789 [1.796, 1.782] | — | — |
| Dense-1536 | 1.815 [1.815, 1.816] | 1.808 [1.810, 1.806] | — | — |
| MoE-no-balance | 1.780 [1.760, 1.799] | 1.782 [1.764, 1.800] | 2.61, 3.63 | 2.63, 3.61 |
| MoE-aux-loss | 1.786 [1.772, 1.799] | 1.779 [1.774, 1.785] | 1.14, 1.26 | 1.14, 1.33 |
| MoE-aux-free | 1.786 [1.774, 1.797] | 1.788 [1.773, 1.803] | 1.12, 1.20 | 1.10, 1.16 |
| Fine-grained+shared | 1.777 [1.778, 1.776] | 1.768 [1.757, 1.778] | 1.15, 1.22 | 1.11, 1.15 |

- The parameter columns and the column "Experts with load < 1%" (1 / 0 / 0 / 0) match. "For the same variant, a different seed changes the validation loss by up to": README 0.039 → measured 0.036.
- The load table of layer 1 in Sections 4 / 6: step 0 and step 100 match exactly. Step 300, no balancing: README `0.24 0.08 0.45 0.05 0.05 0.00 0.02 0.12` (3.61) → measured `0.27 0.08 0.44 0.06 0.03 0.01 0.02 0.09` (3.53). Step 800: README 3.63 → 3.53. Auxiliary loss: 1.76 / 1.34 / 1.05 → 1.76 / 1.28 / 1.07. Bias: 1.08 / 1.18 / 1.20 → **1.29** / 1.18 / 1.16.
- "Max/mean" of the 4 layers at the end: README 2.25, 3.63, 2.46, 2.79 → measured 2.40, 3.53, 2.29, 2.51.
- A check of each conclusion in the text:
  - "Both load-balancing methods work … no expert is idle" and "Without balancing, the worst layer … one expert is fully idle": still true (worst 3.61; the sixth expert has 1%). But "the sixth expert falls from 3% to 0" and "the third expert grows to 45%" are now 1% and 44%.
  - "The bias method balances the load faster (layer 1 reaches 1.08 at step 100)": the direction is still true (at step 100, bias 1.29 vs auxiliary loss 1.76). But the number 1.08 changed.
  - "The five variants with the same active compute are between 1.777 and 1.793, and the differences are smaller than the 0.039 between seeds": now 1.768–1.789 vs 0.036. The conclusion is still true.
  - "Dense-1536 is the worst (with both seeds)": still true (seed 0: 1.810, the worst; seed 1: 1.806, the worst).
  - **The sentence that is no longer true**: at the end of Section 7, "Fine-grained+shared … the results of its two seeds are the closest (1.778, 1.776)". The measured values are 1.757 and 1.778, a difference of 0.021. Except for MoE-no-balance and MoE-aux-free, this is the largest difference. The closest seeds are those of Dense-1536 (0.004). (Fixed later: Section 7 now gives the values of both runs and says that two seeds cannot show whether this variant is more stable.)
- Suggestion (the main workflow decides): either refresh this table and the related sentences with the results of this machine, or add a note: "the numbers depend on the machine; retraining on another machine causes a drift of about 0.01". (Later, the README got the second option: a note in Section 4 says that the numbers depend on the machine and links to this record.)

### Detail 2: Chapter 25 (a chain of changes from different target-model weights)

- `01`: validation loss of the target model 1.838 → **1.831**. (Line 22 of the README and the Chapter 10 README both say 1.838. The source is `01_tiny_model.py` of Chapter 10, which gives 1.831 when retrained on this machine.)
- `02`, greedy table (README → measured):

  | k | α | Output per round | Formula (1) | Target forward passes |
  |---|---|---|---|---|
  | 1 | 0.721 → 0.703 | 1.72 → 1.70 | 1.72 → 1.70 | 466 → 471 |
  | 2 | 0.697 → 0.706 | 2.19 → 2.18 | 2.18 → 2.20 | 368 → 368 |
  | 3 | 0.722 → 0.723 | 2.64 → 2.65 | 2.62 → 2.62 | 306 → **303** |
  | 4 | 0.704 → 0.708 | 2.83 → 2.89 | 2.79 → 2.82 | 286 → 279 |
  | 5 | 0.700 → 0.704 | 2.95 → 3.03 | 2.94 → 2.96 | 275 → 267 |
  | 6 | 0.707 → 0.710 | 3.10 → 3.19 | 3.12 → 3.13 | 262 → 253 |
  | 8 | 0.703 → 0.702 | 3.32 → 3.30 | 3.22 → 3.22 | 244 → 245 |

  The text "at k = 3, it goes from 800 to 306" → 303. "The acceptance rate is 0.70–0.72 for all values of k" is still true. The CPU time and the speedup are timing (normal greedy 1.88 s, c ≈ 0.21, the fastest is k = 3 with 1.30×, formula 1.62×). The trend agrees with the text.
- `03`, part 3 (temperature 1): for k = 1 / 3 / 5, the measured acceptance rate 0.674 / 0.680 / 0.747 → 0.663 / 0.700 / 0.703. The formula mean 0.690 / 0.697 / 0.714 → 0.691 / 0.707 / 0.699. The output per round 1.67 / 2.44 / 3.19 → 1.66 / 2.48 / 2.89. Speedup 1.23 / 1.18 / 1.10× (timing). The README says that k = 5 is slower, but this time it was not slower. The README already says "look only at the trend".
- `05`:

  | Row | README | Measured |
  |---|---|---|
  | Seed 0, λ = 0 | 1.838 / 0.454 | 1.831 / 0.455 |
  | Seed 0, λ = 0.3 | 1.821 / 0.456 / 0.464 | 1.826 / 0.456 / 0.462 |
  | Seed 1, λ = 0 | 1.869 / 0.445 | 1.861 / 0.446 |
  | Seed 1, λ = 0.3 | 1.857 / 0.449 / 0.458 | 1.858 / 0.449 / 0.457 |
  | Self-speculation | Acceptance rate 0.638, 486 forward passes, 1.65 characters per pass; 2.15 s vs 2.22 s, 0.97× | 0.627, 490, 1.63; 2.11 s vs 2.01 s, 1.05× |

  The text says "For both seeds, the main model with MTP has a slightly lower validation loss (by 0.017 and 0.012)". The measured values are only 0.005 and 0.003 lower. The direction does not change. The conclusion "the difference is smaller than the noise of a different seed, so it does not show that MTP is better" becomes even stronger. The MTP module has 246,400 parameters (29%), which matches.
- Also note: the cached weights of Chapter 10, `chapters/10-inference/code/out/tiny_kv4_s600.pt`, are shared by several chapters. In this run, my process and another agent each trained and wrote this file in the same minute (the other agent overwrote it at 09:54:21). The two results were the same, so this caused no problem. But with parallel reproduction, a process can "read a half-written file".

## 3. New GPU scripts and README sections

Both scripts do these things:

- Without CUDA, they print "This script needs a CUDA GPU. …" and exit 0 (verified).
- They use a fixed seed.
- They warm up, measure with CUDA events / `torch.cuda.synchronize()`, and take the median.
- At the start, they print the GPU name and the torch version.
- They pass `ruff check` and `ruff format --check`.
- They change no existing file. (The Chapter 25 script replaces the device of the mask `arange` in `Attention.forward` of Chapter 10, but only inside its own process. The file itself did not change.)

### Chapter 24: `chapters/24-mixture-of-experts/code/05_gpu_moe.py` (6 s on GPU0)

The setup: 8192 tokens at a time, d = 1024, expert width 2048, top-2, E = 8 → 128. The script compares the loop over experts of `02` with "sort into segments + `F.grouped_mm`". It also compares them with a dense SwiGLU of the same compute. The README has a new section, `## GPU measurements (one RTX 3090)`, directly before `## From minimal code to production code`. The numbers of the table come from `gpu_24_final.log`:

| E | Total parameters | GFLOP per token | Tokens per expert | Loop, ms | Grouped, ms | Loop/grouped | Grouped TFLOPS |
|---:|---:|---:|---:|---:|---:|---:|---:|
| Dense 4096 | 13M | 0.0252 | — | — | 4.27 | — | 48.3 |
| 8 | 50M | 0.0252 | 2048 | 6.45 | 6.03 | 1.1× | 34.2 |
| 128 | 805M | 0.0254 | 128 | 28.79 | 10.05 | 2.9× | 20.7 |

(The README has all 5 rows.) Key points:

- The total parameters grow ×16, but the FLOPs do not change. The grouped version grows from 6.03 to 10.05 ms.
- With smaller experts, the compute utilization falls from 48.3 to 20.7 TFLOPS.
- At E = 128, the Python loop is 2.9 times slower. (This factor changes with the CPU load: the same calculation gave 3.1× and 3.7× in the first two trial runs.)

**A finding about the GPU statements in the text (I did not change the text; the main workflow decides)**: Section 9 says that "GPUs use a grouped GEMM (one kernel calculates a group of matrix multiplications with different shapes)". The profiler shows that `F.grouped_mm` of PyTorch 2.11 launches 129 kernels on the 3090 (sm86). 128 of them are cuBLAS GEMMs, one for each expert (`ampere_bf16_s16816gemm…`). They are not one kernel. The concept in the text is correct. But a reader who uses the PyTorch grouped_mm on a consumer card does not get "one kernel". The GPU section of the README states this honestly. (Fixed later: Section 9 now says that whether a grouped GEMM is really one fused kernel depends on the implementation and the hardware.)

At first, I also measured "how many experts each decode step reads" (to check the third ledger of Section 8). But because of the point above, `F.grouped_mm` also takes 2.2 ms at batch 1. That is slower than one read of all 128 experts combined into one dense FFN (1.8 ms). Thus the measurement cannot show "read only the selected experts", and I did not put it into the final script and the README.

### Chapter 25: `chapters/25-mtp-speculative-decoding/code/06_gpu_speculative.py` (96 s on GPU0, about 10 GB of GPU memory)

The script has three parts:

1. The time to feed T tokens at once, with 200 positions already in the KV cache. Two targets: the 0.86M target of this chapter, and a 4.15B target scaled up with the same structure (BF16, random weights).
2. The trained target + draft of this chapter run `speculative_greedy` of `02` on the GPU. The script checks that the output is token-for-token identical and records the number of accepted tokens in each round.
3. The scaled-up target + a scaled-up draft of 1/322 the size "replay" speculative decoding with the real per-round acceptance counts (the first 2 prompts, median of 3 runs). This part measures the wall-clock speedup and compares it with the formula.

The numbers of the README section come from `gpu_25_final.log`:

- T = 1 / 2 / 4 / 8 / 16 / 64 / 256: the scaled-up model goes from 13.385 to 14.597 ms (1.09× at T = 16), 1.51× at 64, and 3.50× at 256. At T = 1, the effective bandwidth is 621 GB/s. The small target of this chapter takes 3.84 ms on the GPU and stays flat. But this is only 0.9 GB/s (kernel launch overhead, about 6.5 µs for each kernel).
- In GPU FP32, k = 1, 2, 3, 4, 6, and 8 are all token-for-token identical. α is 0.702–0.723 (exactly the same as the CPU reproduction on this machine).
- Replay: c ≈ 0.078. For k = 1 / 2 / 3 / 4 / 6 / 8, measured 1.61 / 1.90 / 2.18 / **2.25** / 2.18 / 2.01×, formula 1.60 / 1.89 / 2.15 / 2.20 / 2.15 / 2.01×. The difference is < 3%.

**Findings about the GPU statements in the text (I did not change the text; the main workflow decides)**:

1. **Confirmed**: Sections 1, 5, and 8 say that decode is limited by bandwidth, and that verifying k+1 positions costs about the same as generating 1. The 4.15B model with T ≤ 16 is only ≤ 9% slower.
2. **Confirmed**: Section 5 says that the speedup of the small CPU experiment does not show the situation on a GPU. On the GPU, verification is free, c is small, and the formula agrees almost exactly. The best k = 4 gives 2.25× (on the CPU, at most 1.28×).
3. **Partly contradicted**: Section 5 says "On real GPU systems, the draft is usually two orders of magnitude smaller than the target", and "c can go below 0.05". Here the draft has 1/322 of the parameters of the target, but c = 0.078 (a time ratio of about 1/13). In eager PyTorch, one step of the draft is almost only kernel launch overhead. Thus the number of layers / operators sets c, not the parameter count. In inference engines with CUDA Graph / fused kernels, c can be lower, but this script did not measure < 0.05. (Fixed later: Section 5 now says that c can go below 0.05 only if the kernel launch overhead also becomes small.)
4. Section 3 says that numerical differences between batch sizes on a GPU can make the outputs not fully identical. This FP32 GPU run had no flips (all 6 values of k were token-for-token identical). This does not contradict the text.
5. Also: on the GPU, one forward pass of the small model of this chapter takes 3.84 ms. This is not faster than the single-thread CPU in Section 1 of the README (3.55 ms).

Also, the GPU section says that the acceptance rates "differ from the table of Section 5 by 0.01–0.02". This sentence explains why the GPU α differs from the table of Section 5 (the weights were retrained on this machine). If the main workflow decides to refresh Section 5 with the results of this machine, delete this sentence at the same time.

### Chapter 26: no GPU script

All three scripts read `models.json`, count the parameters on the meta device, and calculate the KV ledger. They do not depend on the GPU. Chapter 21 already measures the GPU part ("KV cache size / decode bandwidth"). No new files; the README did not change.

## 4. Decisions for the main workflow

1. Should the numbers in the text of Chapter 24 03 and Chapter 25 01/02/03/05 change to the results of this machine (after retraining)? At least one sentence is not true on this machine: the end of Section 7 of Chapter 24, "Fine-grained+shared … the results of its two seeds are the closest (1.778, 1.776)". For Chapter 25, the source is 1.838 / 1.831 in Chapter 10. (Fixed later for the Chapter 24 sentence: see Detail 1.)
2. Should Section 9 of Chapter 24, "grouped GEMM (one kernel …)", get a sentence that `F.grouped_mm` of PyTorch on consumer cards actually calls each expert separately? (Fixed later: see Section 3.)
3. Should Section 5 of Chapter 25, "c can go below 0.05", get an addition: "the kernel launch overhead must go away (CUDA Graph and so on); otherwise the number of layers sets c"? (Fixed later: see Section 3.)
4. The numbers of both GPU sections were measured with a power limit of 240 W (effective bandwidth 621 GB/s, dense 48.3 TFLOPS). Add a note about this to all of them.
