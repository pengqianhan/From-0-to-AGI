# GPU0 verification: Chapters 1–6

**English** · [中文](chapters-01-06.zh.md)

Date: 2026-10-01. Branch: `gpu0-verification`. Scope: `chapters/01-*` to `chapters/06-*`.

## Environment

- CPU: AMD Ryzen Threadripper PRO 3995WX (64 cores / 128 threads, Zen 2, AVX2, no AVX-512). This is a shared server. During the runs, the load average was between 5 and 45 (other agents and video renders ran at the same time).
- NumPy 2.4.4 (scipy-openblas 0.3.31, DYNAMIC_ARCH), PyTorch 2.11.0+cu128.
- GPU: NVIDIA GeForce RTX 3090. I used only GPU0 (`CUDA_VISIBLE_DEVICES=0`), with a queue through `gpu0.lock`. Before each run, I made sure that `torch.cuda.device_count() == 1`.
- CPU reproduction command: `CUDA_VISIBLE_DEVICES= UV_NO_SYNC=1 uv run python chapters/.../code/xx.py`, with a maximum of 4 processes in parallel. The raw output is in `/tmp/claude-1006/-home-phan635-Opensource-From-0-to-AGI/1d40fca9-4469-49a7-88b9-4e96bf8f0051/scratchpad/ch01-06/cpu/`. The probe experiments are in `probe/` in the same folder.

## 1. CPU reproduction table

"CPU time" is the wall-clock time of the full `uv run` process. I measured it with 4 runs in parallel on a busy machine, so use it only for the order of magnitude. Categories: `match` / `timing only` / `mismatch` / `error`. All 30 scripts returned rc=0. **There were no errors.**

| Chapter | Script | CPU time | Category | Notes |
|---|---|---:|---|---|
| 1 | `01_fit_line.py` | 0.2 s | match | The table with 5 rows and polyfit 2.052 / 0.879: all correct |
| 1 | `02_learning_rate.py` | 0.2 s | match | Critical value 0.1016; 2.34 / 0.29 / 0.25 / 7.7×10⁹: all correct |
| 1 | `03_pytorch_version.py` | 5.4 s | match | After 201 steps, 2.039 / 0.925 on both sides |
| 2 | `01_matrix_basics.py` | 0.2 s | match | 59.0, 79.0, `[79 125 49 104.5]`, `M@N`, `M*N`, and the broadcasting error message: all correct |
| 2 | `02_matrix_multiply.py` | 0.2 s | match | 58/64/139/154 and the three shape examples: all correct |
| 2 | `03_linear_layer.py` | 0.2 s | match | The main text quotes only shapes: all correct |
| 2 | `04_multivariate_regression.py` | 0.2 s | match | Standard deviations, critical learning rate, training table, table in the original units, and 100.2 (10k yuan): all correct. I also calculated the ratio of the Hessian eigenvalues: 2.42×10⁵. Thus "about 1/240,000" is also correct |
| 2 | `05_loop_vs_vectorized.py` | 2.1 s | timing only | See detail 1 below (the speedup is 2 times the "25×" in the README) |
| 2 | `06_pytorch_version.py` | 5.1 s | match | float64 difference 0.0e+00 and float32 difference 1.8e-05: all correct |
| 3 | `01_linear_is_not_enough.py` | 0.1 s | match | 0.4341 / 0.5178 / 8.9e-16 / 4.4e-15: all correct |
| 3 | `02_activations.py` | 0.1 s | match | The two output blocks are identical, character for character |
| 3 | `03_mlp_numpy.py` | 21.2 s | mismatch (rounding level) | Only one value differs: `Max difference between (sum of pieces + b2) and the network output` is 2.2e-15 in the README and 2.7e-15 in this run. All other losses, kinks, and 12.17 / 1.20 are correct |
| 3 | `04_pytorch_version.py` | 15.4 s | mismatch (rounding level) | `difference = 6.2e-18` in the README; 3.9e-18 in this run. All other values are correct |
| 4 | `01_engine.py` | <0.1 s | match | |
| 4 | `02_grad_check.py` | 6.0 s | match | The four-row error table, the expression gradients, and the 4 rows of the parity check with Chapter 3: all correct |
| 4 | `03_train_mlp.py` | 9.6 s | timing only | Loss table, prediction table, and 3720 nodes: all correct. The README says "the 500 steps took 12.3 seconds (about 25 ms per step)". This run took 9.6 seconds / 19 ms |
| 4 | `04_pytorch_compare.py` | 28.9 s | mismatch (rounding level) + timing | See detail 2 |
| 5 | `01_softmax.py` | 0.2 s | match | |
| 5 | `02_cross_entropy.py` | 0.1 s | match | Likelihood table, loss table, 2.18e-12, 3.63e-11, and the gradient table for CE vs MSE: all correct |
| 5 | `03_train_classifier.py` | 3.6 s | **mismatch** | Comparison 3 (MSE "stuck at 65%") does not reproduce. See detail 3 |
| 5 | `04_next_token.py` | 0.2 s | match | |
| 5 | `05_pytorch_version.py` | 4.4 s | mismatch (rounding level) | Max gradient difference: 6.94e-18 in the README → 1.39e-17 in this run. Difference between the cross-entropies at step 3000: 7.85e-11 → 7.23e-11. All other values are correct |
| 6 | `01_signal_propagation.py` | 2.3 s | match | Both tables are correct |
| 6 | `02_normalization.py` | 2.3 s | match | |
| 6 | `03_residual.py` | 2.2 s | match | |
| 6 | `04_optimizers.py` | 0.7 s | match | |
| 6 | `05_lr_schedule.py` | 0.1 s | match | |
| 6 | `06_ablation.py` | 59.6 s | **mismatch** | The values of the float32 training drift. The result of row B changes from "learns slowly" to "did not learn". See detail 4 |
| 6 | `07_schedule_experiments.py` | 70.0 s | **mismatch** | Same cause: the values drift a little. See detail 4 |
| 6 | `08_pytorch_version.py` | 17.5 s | **mismatch** | The float32 parity-check difference and the final validation losses are different from the README. See detail 4 |

Statistics (30 scripts): match 20. Timing only 2 (Chapter 2: 05, Chapter 4: 03). Mismatch, but only at double-precision rounding level (≤1e-15) 4 (Chapter 3: 03/04, Chapter 4: 04, Chapter 5: 05). Mismatch that affects numbers or conclusions in the main text 4 (Chapter 5: 03, Chapter 6: 06/07/08). Error 0.

## 2. Details of the mismatches and timing differences

### Detail 1: Chapter 2 `05_loop_vs_vectorized.py` (timing only, but it affects one argument)

I ran the script 3 more times separately (machine load ≈ 45):

| | README (4-core CPU) | This machine (3 runs) |
|---|---:|---:|
| Three nested loops | 62.70 ms | 58.1 / 63.1 / 60.3 ms |
| One `np.dot` for each row | 1.94 ms (32×) | 2.34 / 2.43 / 2.42 ms (25–26×) |
| One `X @ W` | 0.057 ms (1091×) | 0.038–0.039 ms (1485–1656×) |
| Gradient with a loop | 538.99 ms | 544.7 / 550.9 / 553.0 ms |
| Matrix form | 21.73 ms (25×) | 10.5 / 11.0 / 11.1 ms (50–52×) |
| Max parameter difference | 3.6e-15 | 3.6e-15 |

The README itself says that the times change with the machine ("the speedup of the forward pass changes between 950× and 1170×"). Thus I put this script in the "timing only" category. But on this machine, the forward speedup is 1400–1650×, outside the range in the README. The training comparison is about 50×, not 25×.

Two places quote the exact value 25. The main text says "Comparison 2 is only 25× faster, much less than comparison 1". Guided question 5 asks "why is the training comparison only 25× faster?". The argument itself (50 ≪ 1500) is still correct. **You need to decide**: change "25×" to a statement that does not depend on the machine, such as "tens of times"?

### Detail 2: Chapter 4 `04_pytorch_compare.py` (rounding level + timing)

| Parameter | README max difference | This run |
|---|---:|---:|
| 0.weight | 8.3e-17 | 5.6e-17 |
| 0.bias | 3.1e-17 | 2.2e-17 |
| 2.weight | 5.6e-17 | 5.6e-17 |
| 2.bias | 7.5e-17 | 7.1e-17 |
| 4.weight | 4.2e-17 | 1.1e-16 |
| 4.bias | 5.6e-17 | 1.4e-17 |
| Max over all 97 parameters | 8.3e-17 | 1.1e-16 |

These values all match: the initial loss 0.716258950201, the loss 0.0093662407 after 500 steps, gradcheck True, and VJP True. Speed table in this run: 20 samples 47.0 ms / 0.257 ms (183×), 200 samples 472.4 ms / 0.294 ms (1608×). The README has 43.7 / 0.355 (about 120×) and 775.3 / 0.414 (about 1900×). Timing differences are expected. The README says "when the number of samples increases 10 times, the time of the scalar engine increases by a little more than 10 times". On this machine, the increase was about 10 times.

Probable cause: all of these are double-precision rounding differences of magnitude 1e-16. The PyTorch double-precision kernels add the numbers in a different order on different CPU instruction sets (this machine: AVX2, no AVX-512). The same cause applies to the rounding-level differences in Chapter 3 (03/04) and Chapter 5 (05).

### Detail 3: Chapter 5 `03_train_classifier.py`, comparison 3 (it affects an argument in the main text)

The table in Section 5 of the README and the output of this machine (output-layer std = 10, lr = 1.0):

| Step | CE accuracy (README / this run) | MSE accuracy (README / this run) |
|---:|---:|---:|
| 0 | 30.7% / 30.7% | 30.7% / 30.7% |
| 100 | 85.7% / 85.7% | **65.3% / 53.3%** |
| 500 | 97.3% / 97.3% | **65.3% / 96.7%** |
| 800 | 98.0% / 98.0% | **66.0% / 99.3%** |
| 1000 | 99.0% / 99.0% | **95.3% / 99.3%** |
| 3000 | 99.3% / 99.3% | **99.3% / 99.0%** |
| Samples with p(correct) < 1% at step 500 | 0 / 0 | **103 / 7** |

The CE column matches digit for digit, but the MSE column is completely different. On this machine, two statements in the README are not true. The first is "the MSE version was stuck near 65% for 700 to 800 steps". The second is "the MSE version has 103 samples that are confidently wrong". In this run, the MSE version reached 96.7% at step 500. At step 800, it was even higher than the CE version.

Investigation:

- Change the number of OpenBLAS threads (1 / 4 / default 128): the result did not change (the run is deterministic on this machine).
- Turn off the AVX2/FMA3 SIMD of NumPy (`NPY_DISABLE_CPU_FEATURES`): the result did not change.
- Change the OpenBLAS kernel (`OPENBLAS_CORETYPE=Prescott`): the MSE trajectory changed much again (step 100: 94.0%, step 500: 97.0%, step 800: 98.7%; 3 samples below 1% at step 500). The `Haswell` kernel gave the same result as the default.

Conclusion: this MSE training trajectory is extremely sensitive to the rounding in matrix multiplication. When the logits are large, softmax saturation makes the MSE gradient very small. Then very small rounding differences can decide which samples "flip" first. The README numbers probably come from a different CPU or BLAS kernel. This is not a code bug.

But **the specific effect that the main text uses as support, "MSE is stuck for 700 to 800 steps", cannot be reproduced**. The video also has "65%, 103 samples", in fact F8 of `video/script.md` and in the storyboard. (`scenes.py` calculates the curves from the code during the render. Thus the curves on the screen will not agree with the narration.) The table "gradient size of CE vs MSE" in Section 5, part 4 uses only formulas and no training, and it agrees completely. The core argument, "the MSE gradient vanishes when the model is confidently wrong", is still correct.

**You need to decide**: replace the demo with one that is stable with all BLAS kernels? For example, use the median of several seeds, or set out_std / the learning rate to a range where MSE is stuck reliably. Then update the README and the video narration to agree. I did not change any file.

### Detail 4: Chapter 6 `06_ablation.py`, `07_schedule_experiments.py`, `08_pytorch_version.py` (float32 drift)

These three scripts train with PyTorch float32 on a single thread. The values differ a little from the README. In a few places, the difference changes a result label in the main text.

`06_ablation.py` (README → this run):

| Configuration | Validation loss | Accuracy | Result / three learning rates |
|---|---|---|---|
| A | nan → nan | | Same |
| B + Kaiming | **1.931 → 2.035** | **0.279 → 0.168** | **"learns slowly" → "did not learn"**; 2.301/2.095/1.931 → 2.301/2.093/2.035 |
| C + residual | 0.821 → 0.822 | 0.705 → 0.703 | 0.823/0.821/nan → 0.823/0.822/nan |
| D + RMSNorm | 0.811 → 0.811 | 0.700 → 0.697 | 0.838/0.814/0.811 → 0.838/0.815/0.811 |
| E AdamW | 0.807 → 0.806 | 0.697 → 0.700 | |
| F + cosine | 0.775 → 0.774 | 0.711 → 0.710 | |
| G full set | 0.775 → 0.775 | 0.708 → 0.709 | |
| No residual | **1.462 → 1.310** | 0.515 → 0.560 | Same result in both: "learns slowly" |
| No RMSNorm | 0.747 → 0.744 | 0.726 → 0.726 | |
| std = 1 | 0.825 → 0.825 | 0.697 → 0.697 | |
| Constant learning rate | 0.803 → 0.802 | 0.704 → 0.701 | |

`07_schedule_experiments.py` (README → this run):

- Schedules: cosine 0.775 → 0.775; WSD 0.769 → 0.770; const 0.806 → 0.808.
- WSD branches: at 500 steps, 0.887→0.801 (cosine 0.796) → this run 0.885→0.801 (0.796). At 900 steps, 0.806→0.755 (0.760) → this run 0.808→0.756 (0.761).
- Stress test: full set 2.723 / 0.892 → 2.723 / 0.886; no warmup 6.178 / 0.890 → 6.178 / 0.888; **no RMSNorm 1.1×10¹⁰ / 0.991 → 2.4×10¹² / 0.999**.
- Gradient clipping (RMSNorm / clipping): on/on 1.1, 1.096, 0.776 → 1.1, 1.095, 0.776; on/off 1.096 → 1.094; **off/on 95.9, 1.118, 0.747 → 106.5, 1.099, 0.747**; off/off 105.9, 1.977, 0.773 → 107.1, 2.000, 0.770.

`08_pytorch_version.py` (README → this run):

- float64 max loss difference per step: 8.9e-16 → 4.4e-16 (rounding level).
- **float32 max loss difference per step: 1.3e-2 → 3.1e-4**. At step 200: 0.8676 / 0.8659 → 0.8659 / 0.8660.
- **Full set with std = 0.02, 800 steps: cosine 0.80028, WSD 0.80029 → 0.80101, 0.79763**.

Probable cause: float32 training makes the rounding differences of about 1e-7 per step larger. The rounding differences depend on the CPU kernels that PyTorch selects. Verification: on the same machine, I set `ATEN_CPU_CAPABILITY=default` (this turns off the vectorized kernels) and ran `08` again. The float32 parity-check difference became 2.4e-07. The final validation losses became 0.80282 / 0.79997, which is a third set of numbers. Section 8 of the README itself says that "a fixed random seed makes the results reproducible only on the same machine with the same code". Thus this is not a bug.

These sentences in the main text are affected (**you need to decide** whether to change them to statements that do not depend on the machine):

- Row B of the table in Section 8 ("1.931 / learns slowly"), "the best result is only 1.931", and "the validation loss decreases directly from 1.931 to 0.821". The summary table in Section 9: "validation loss 1.931 → 0.821". On this machine, row B is above the threshold of 2.0 for "did not learn".
- "If we remove only the residual connections from the full set, the loss goes back to 1.462" (this machine: 1.310).
- Section 7: "with clipping, it goes only to 1.118". The summary table: "1.977 → 1.118" (this machine: 2.000 → 1.099).
- Section 8: "the loss spike goes as high as 10¹⁰". The summary table: "the spike decreases from 10¹⁰ to 2.7" (this machine: 2.4×10¹²).
- In "From minimal code to production code": "after 200 steps, the difference grows to 10⁻²" (this machine: 3×10⁻⁴). Also "the validation losses of cosine and WSD are 0.80028 and 0.80029, almost the same by chance" (this machine: 0.80101 / 0.79763, a difference of 0.003).

All qualitative conclusions are true on this machine. Residual connections have the largest effect. RMSNorm gives robustness. The decay gives an improvement. Clipping limits the spikes.

### Other observations (no effect on numbers)

- `chapters/01-*/code/03_pytorch_version.py`, `chapters/02-*/code/06_pytorch_version.py`, and `chapters/03-*/code/04_pytorch_version.py` do not have `torch.set_num_threads(1)`. Section 3 of `docs/CHAPTER_GUIDE.md` requires this line. The results are correct, but these scripts do not follow the guide.

## 3. GPU measurements

### New files

- `chapters/02-from-scalar-to-matrix/code/07_gpu_matmul.py` (new script): square matrix multiplication `(N, N) @ (N, N)`, N = 64…8192. It compares the time and the TFLOPS of CPU float32 (8 threads), GPU float32 (TF32 off), and GPU BF16. It takes the median after a warmup, and it calls `torch.cuda.synchronize()` for each GPU run. For each size, it does a CPU / GPU / BF16 parity check (assert). Without CUDA, it prints the required message and calls `exit 0` (verified with `CUDA_VISIBLE_DEVICES=`). `ruff check` passes. One run takes about 22 s of wall-clock time.
- `chapters/02-from-scalar-to-matrix/README.md`: a new section `## GPU measurements (one RTX 3090)`, directly before `## From minimal code to production code`. I only added 25 lines and did not change the existing content. The numbers in the table come exactly from the first run of the final version (`scratchpad/ch01-06/gpu/final_run.out`).

Two runs of the final version (one after the other, while the script held the lock one time):

| N | Run 1: CPU / GPU fp32 / speedup / BF16 TFLOPS | Run 2: CPU / GPU fp32 / speedup / BF16 TFLOPS |
|---:|---|---|
| 64 | 0.014 / 0.024 ms / 0.60× / 0.02 | 0.014 / 0.022 ms / 0.63× / 0.02 |
| 128 | 0.025 / 0.039 ms / 0.63× / 0.18 | 0.023 / 0.023 ms / 0.99× / 0.18 |
| 256 | 0.089 / 0.030 ms / 2.93× / 1.05 | 0.164 / 0.024 ms / 6.90× / 1.46 |
| 512 | 0.860 / 0.040 ms / 21.26× / 10.31 | 0.605 / 0.040 ms / 15.09× / 10.61 |
| 1024 | 5.356 / 0.136 ms / 39.46× / 34.82 | 4.941 / 0.135 ms / 36.61× / 34.64 |
| 2048 | 39.707 / 0.776 ms / 51.15× / 56.43 | 37.287 / 0.776 ms / 48.07× / 56.39 |
| 4096 | 310.0 / 8.138 ms / 38.10× / 51.60 | 289.5 / 8.126 ms / 35.62× / 49.14 |
| 8192 | 2.40 s / 63.96 ms / 37.54× / 50.62 | 2.41 s / 63.66 ms / 37.84× / 51.15 |

For N ≥ 512, the two runs agree closely. For the small matrices (N ≤ 256), the CPU load (shared server) and the variation of the launch overhead have an effect. The speedup is between 0.6× and 1.0× for N=128, and between 2.9× and 6.9× for N=256. The conclusion "the GPU is slower for small matrices, and tens of times faster for large matrices" is true in both runs.

An earlier version had a second part: "move the forward pass (1000,100)@(100,10) of comparison 1 in Section 8 to the GPU". I removed this part. On this long, thin shape, torch on the CPU with 8 threads needs 0.11 ms. This is slower than the 0.04–0.06 ms of NumPy in the main text. The two results together would mislead the reader.

### Important finding: the power limit of GPU0 is 240 W (factory default: 350 W)

`nvidia-smi -i 0` gives `power.limit 240 W, power.default_limit 350 W, clocks.max.sm 2100 MHz`. A probe (`scratchpad/ch01-06/probe/clock_probe.py`) ran large matrix multiplications continuously for about 4 s:

| Load | Sustained TFLOPS | Sampled SM clock / power |
|---|---:|---|
| 2048 float32 | 13.0 | 810–960 MHz / 225–239 W |
| 8192 float32 | 15.6 | 960–1020 MHz / 182–240 W |
| 8192 BF16 | 47.4 | 1200–1470 MHz / 158–238 W |

Thus, on this card, the power stays at the 240 W limit under sustained full load. The clock is then only a little more than half of the idle boost clock (about 1.7 GHz). In the benchmark script, 2048 is faster than 4096/8192 because each round takes only slightly more than 10 ms. Thus it still runs at the boost clock.

**This affects the absolute numbers of all GPU measurements in all chapters**, mainly the timings that last more than a few seconds. I recommend that the main workflow state this one time, in the GPU section of each chapter or in the summary. Alternatively, decide whether to set the power limit back to 350 W and measure again. I did not change any GPU setting. The new section in the Chapter 2 README already states this.

### Other chapters: no new GPU scripts

The scripts of Chapters 1 and 3–6 are small NumPy / single-thread PyTorch examples, with tens to tens of thousands of parameters. The main text discusses values and convergence, not speed. On a GPU, these scripts would only repeat the conclusion of Chapter 2: "the GPU is slower for small matrices". A script that adds nothing is worse than no script, so I did not add any.

One candidate that I considered and rejected: in Chapter 5, "From minimal code to production code" says that "autocast promotes cross_entropy to float32". This statement can be verified on CUDA. But it is not in the quick-read text, and it does not change the explanation. Thus I did not do it.

## 4. Decisions needed

1. Chapter 5, 03, comparison 3: "MSE is stuck at 65% for 700 to 800 steps / 103 samples" cannot be reproduced. It is also extremely sensitive to the BLAS kernel (detail 3). Replace it with a more stable demo in Section 5 of the README and in the video? In the video, it is in fact F8 of `video/script.md`. It is also in the screen and the narration of storyboard S09: "在百分之六十五附近躺了七八百步……一百零三个样本" ("lay near sixty-five percent for seven or eight hundred steps … one hundred and three samples").
2. The float32 numbers of Chapter 6 (06/07/08) drift with the CPU kernel. The result of row B crosses the "did not learn" threshold (detail 4). Change the text to statements that do not depend on the machine, or mark the numbers as "numbers from this machine"?
3. Chapter 2, main text and Guided question 5: "the training comparison is 25× faster". On this machine, it is about 50× (detail 1).
4. The power limit of GPU0 is 240 W. This affects the absolute values of all GPU measurements (see above).
