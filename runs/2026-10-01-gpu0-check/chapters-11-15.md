# GPU0 verification: Chapters 11–15

**English** · [中文](chapters-11-15.zh.md)

Date: 2026-10-01. Branch: `gpu0-verification`. Scope: `chapters/11-*` to `chapters/15-*`.

## Environment

- CPU: AMD Ryzen Threadripper PRO 3995WX (AVX2, no AVX-512; `torch.backends.cpu.get_cpu_capability()` = AVX2). This is a shared server: other agents and video rendering ran at the same time.
- PyTorch 2.11.0+cu128, CUDA 12.8.
- GPU: NVIDIA GeForce RTX 3090 (GPU0 only, `CUDA_VISIBLE_DEVICES=0`). Each use waited in the queue of `gpu0.lock`. Before the first use, we confirmed `torch.cuda.device_count() == 1`. The power limit of GPU0 is 240 W (from `nvidia-smi`; the default is 350 W).
- CPU reproduction command: `CUDA_VISIBLE_DEVICES= UV_NO_SYNC=1 uv run python <script>`, with at most 4 processes in parallel. `06_ddp_by_hand.py` starts 2 processes itself, so it ran last and alone.
- Raw output: `/tmp/claude-1006/-home-phan635-Opensource-From-0-to-AGI/1d40fca9-4469-49a7-88b9-4e96bf8f0051/scratchpad/ch11-15/` (`cpu/*.log` is the CPU reproduction, `cpu/_timings.txt` the run times, `gpu/*.log` the GPU runs, and `gpu/_runs.txt` the queue and run times).

**Shadow folder**: 12/03, 04, and 06 write caches to `out/ch12/*.json`. 15/03 and 04 write `chapters/15-*/code/out/*.pt` (the video of Chapter 15 reads these files). We did not want to leave files in the repository or affect the rendering in progress. Thus all CPU scripts ran in a shadow repository in the scratchpad, `scratchpad/ch11-15/shadow/`. The `code/` folders of Chapters 11–15 are unchanged copies, and so is the `code/` folder of Chapter 9, which Chapter 12 loads with importlib. The other folders are symbolic links to the repository. The commands started from the repository root and used the `.venv` of the repository. The folder was new, so all caches were calculated again, and no old cache was read. After the runs, we confirmed that the repository has no new `out/`, `data/`, or `code/out/` files.

## 1. CPU reproduction table

"CPU time" is the wall-clock time of the full `uv run` process. We measured it with 4 runs in parallel on a busy machine, so use it only for the order of magnitude. Categories: `match` / `timing only` / `mismatch` / `mismatch (rounding level)` / `error`. All 30 scripts returned rc=0: **no errors**.

| Ch. | Script | CPU time | Category | Notes |
|---|---|---:|---|---|
| 11 | `01_pick_on_test.py` | 0.8 s | match | The 4 rows of the table are bit-identical |
| 11 | `02_loglik_vs_generate.py` | 0.7 s | match | q13 table, 6-row result table, "英国→巴黎" (UK → Paris), 92 documents / 32 questions / 12 leaked questions |
| 11 | `03_prompt_sensitivity.py` | 0.2 s | match | 6-row table, highest 0.531 / lowest 0.125, share of the most-picked position for letter choices 1.00 |
| 11 | `04_paired_bootstrap.py` | 2.2 s | match | Hand-calculated example [+0.00, +0.50], paired width 0.093 / unpaired 0.160, table of item counts, multiple comparisons 0.07 / 0.26. The table of the tiny-configuration demo is hard-coded in the script. The log prints "(out/smoke/eval/results.json does not exist locally; check skipped)", which is expected. README L20 says "about 10 seconds" |
| 11 | `05_contamination.py` | 0.2 s | match | 12/12, 22 question stems, 0.375 → 0.000, variant table, canary, schema comparison |
| 12 | `01_flops.py` | 5.6 s | match | The table and the main-line block are bit-identical |
| 12 | `02_chinchilla.py` | 0.2 s | match | |
| 12 | `03_lr_sweep.py` | 134 s | **mismatch** | The loss in the high-learning-rate cells differs, so η\*, the power law, and the s5 learning rate change. See Detail 1. README says "about 7 minutes" |
| 12 | `04_mini_ladder.py` | 185 s | **mismatch** | The fitted formula, the errors, the held-out table, and the optimal ratio all changed. 4 qualitative conclusions are no longer true. See Detail 2. README says "about 15 minutes" |
| 12 | `05_plan_budget.py` | <0.1 s | match | The README shows an excerpt: it removed 4 candidate rows and the "MFU 0.55" line of the output, with no mark for the omission |
| 12 | `06_muon.py` | 61 s | match | README says "about 5 minutes" |
| 13 | `01_noisy_crawl.py` | 0.1 s | match | |
| 13 | `02_heuristic_filter.py` | 3.6 s | match | L107 "the rules remove 96% of the code in the tiny corpus" is not an output of this script, so we cannot check it |
| 13 | `03_minhash.py` | 9.3 s | match | |
| 13 | `04_quality_classifier.py` | 5.1 s | match | |
| 13 | `05_decontam.py` | 6.9 s | match | |
| 13 | `06_quality_ablation.py` | 211 s | **mismatch** | The document counts and the token counts are correct. The bpb, the loss of the last step, and the paired differences differ. See Detail 3 |
| 13 | `07_mixture_ablation.py` | 277 s | **mismatch** | Of the 6 training runs, 3 differ from the README by ≤ 0.001, and the other 3 differ by 0.01–0.10. See Detail 3 |
| 13 | `08_vocab_size.py` (default mode) | 54 s | match | All numbers that the default tiny-corpus mode prints are correct. The large table in the text comes from `--corpus DIR --refs` (an external corpus), which we did not reproduce |
| 14 | `01_step_cost.py` | 4.6 s | timing only | ① and ② are all correct. ③: the single-thread FP32 peak of this computer is 56.2 GFLOPS (README 48.8). The MFU step reads the tiny pretraining log `out/ch14/pretrain/log.jsonl` (an output of the tiny-configuration demo), so the script skips it by design |
| 14 | `02_precision.py` | 4.2 s | match | Bit layout table, 100.0030 / 4 / 32, 1.0000 / 0.0000, the 5 values of stochastic rounding and their mean 97.90, inf / 0 / 1.00117e-8, 2.4% / 0.0% / 0.0%, 2.78e-3: all correct |
| 14 | `03_online_softmax.py` | 2.2 s | **mismatch** | The step table, the softmax values, and 5.8e-15 for 512 × 64 are all correct. L218 "10,000 random scores … max difference 1.1 × 10⁻¹⁶": measured **8.3e-15**. See Detail 4 |
| 14 | `04_tiled_attention.py` | 2.4 s | mismatch (rounding level) | Middle block 4,096 / 7,000, 524,288, and 8.0 GiB / 112 GiB are all correct. The float64 max differences for each block size differ from the README at the 10⁻¹⁶ level. See Detail 4 |
| 14 | `05_memory.py` | 6.1 s | match | Table of 16 bytes/parameter, activations 30,251,012 / 24,090,628 / 6,428,676 (100% / 80% / 21%), main-line memory table: all correct |
| 14 | `06_ddp_by_hand.py` | 8.2 s | mismatch (rounding level) | Loss table, 1.8e-15, 1,500,000, and 2.57 / 4.50 GiB are all correct. L298 "max difference of the final parameters 6.9 × 10⁻¹⁷" (two places): measured 5.6e-17 |
| 14 | `07_resume.py` | 3.8 s | match | Resume table, bit-identical, and the max differences of the four "restore one item less" cases: all correct |
| 15 | `01_rope_wavelengths.py` | 2.3 s | match | Wavelength table, number of pairs that do not complete one turn, 6.96 / 26.69 GFLOP, 41% / 84%, 3.84× |
| 15 | `02_yarn_from_scratch.py` | 5.7 s | match | Excerpt table, mscale 1.1386, 1.2965, parity check 6.0e-08 |
| 15 | `03_context_extension.py` | 773 s | **mismatch** | Each cell of the two loss tables differs by 0.000–0.029. The qualitative conclusions are mostly still true. See Detail 5. README says "about 11 minutes" |
| 15 | `04_anneal_mixture.py` | 500 s | **mismatch** | The bpb table differs by at most 0.047, and all qualitative conclusions are still true. Also, "1.3M parameters" does not agree with the source code. See Detail 5 |

Summary: match 20, timing only 1, mismatch (rounding level) 2, mismatch 7, error 0. Of the 7 mismatches, 6 are diverged trajectories of training scripts, and 1 is a single number in 14/03. In addition, 12/03–04 and 15/04 each have one text error that does not depend on the run environment.

The line numbers (L…) in this text refer to the README before this change. In Chapter 14, the new section is after the original L436. In Chapter 15, the new section is after the original L296 (+31 lines), so the original L408 of Chapter 15 is now L439.

### Common cause of the mismatches: floating-point differences between machines (nondeterminism on this machine is excluded)

All training mismatches (12/03, 12/04, 13/06, 13/07, 15/03, 15/04) have the same form. The seed is fixed, and `set_num_threads(1)` is set. The code and the data did not change after the README commit: the deterministic quantities, such as the token counts and the document counts, are bit-identical. But the training loss diverges at the 3rd–4th digit. The runs with a high learning rate diverge most, and some runs are bit-identical.

We wanted to separate two cases: "each run on this machine is different" and "only a different machine gives different results". Thus we ran `12/03_lr_sweep.py` again on the same machine with `--fresh`. The two outputs (without the timings) were **byte-identical**. So these scripts are deterministic in one environment. The README numbers most probably came from a different machine or a different set of CPU kernels (for example, a different instruction set / BLAS). Training over hundreds to thousands of steps amplifies the very small rounding differences. 14/04 also supports this. We ran it again with `ATEN_CPU_CAPABILITY=default`. The SDPA max difference in the `128×32` cell changed from 8.0e-16 to 7.8e-16, and the other values did not change. The same code with different kernels gives different rounding noise.

**Recommendation**: For the numbers of training scripts in the text, do one of two things. Update them from a rerun on this machine, or add the note "on a different machine, the numbers differ from the 3rd digit". Also change the qualitative conclusions that depend on specific values (see Details 2, 3, and 5) into statements that this drift does not affect. The main process decides this; I did not change them.

### Detail 1: 12/03_lr_sweep.py

In the sweep table, the low-learning-rate cells are bit-identical, and the high-learning-rate cells differ (README → measured):

- s1: column 0.02 3.2778\* → 3.2791\*; column 0.04 3.3372 → 3.3057
- s2: column 0.02 3.2353 → 3.2311; column 0.04 3.3323 → 3.3123
- s3: column 0.005 3.2385 → 3.2549; column 0.01 3.3055 → 3.3096; column 0.02 3.3848 → 3.3321; column 0.04 3.4342 → 3.4426
- s4: column 0.0025 3.1401\* → 3.1472\*; column 0.005 3.2746 → 3.2767; column 0.01 3.3019 → 3.3051; column 0.02 3.3805 → 3.4094; column 0.04 3.3845 → 3.3547
- For all 4 sizes, the position of the best grid point did not change.

Downstream: η\* (L182) s1 0.01879 → 0.02144, s2 0.00885 → 0.008938, s3 0.002915 → 0.002783, s4 0.002284 → 0.002263. Power law (L184): 18.6·N^(−0.744) → 32.6·N^(−0.794). s5 learning rate (L185, L240): 0.001123 → 0.001029. Table "use 0.02 for all sizes" (L191–194): s2 +0.0684 → +0.0642, s3 3.3848 / +0.1908 → 3.3321 / +0.1382, s4 3.3805 / +0.2405 → 3.4094 / +0.2622. L197 "s3 and s4 are even worse than s1 and s2" is still true.

### Detail 2: 12/04_mini_ladder.py

04 reads η\* from 03, so it changes together with 03. The same floating-point sensitivity is added on top:

- Fitted formula (L229): 2.557 + 19.23/N^0.52 + 520.6/D^0.58 → 2.565 + 1721/N^0.98 + 427.2/D^0.56
- Fit error (L231–235): max error 1.80% → 2.40%; s1 D=65,536 error +1.79% → +2.40%
- Held-out table (L245–247): errors −0.87% / −0.32% / +1.01% → −1.02% / −0.15% / +1.51%. Interval of D=65,536: 3.3912–3.4535 → 3.3668–3.4585, measured 3.4623
- Compute-optimal (L255): N ≈ 21,194, D ≈ 2,693,780 (127.1 tokens/parameter) → N ≈ 28,997, D ≈ 1,968,928 (67.9 tokens/parameter)

**Qualitative statements that are no longer true**:

- L238 "both exponents are about 0.5": measured α = 0.98, β = 0.56;
- L238 "the max error of the fitted points is 1.8%": measured 2.40% ("looser than both the Delphi standard and the gate 1 standard" is still true);
- L250 "all of them are inside the 95% interval": the measured 3.4623 at D=65,536 is above the upper limit 3.4585 of the interval;
- L258 "the compute-optimal ratio is 127 tokens/parameter": measured 67.9 ("not 20" is still true).

**A text error that does not depend on the run environment**: L216 and the docstring at line 118 of `03_lr_sweep.py` describe the WSD branches. They say that "the 3 budgets cost only about 1.35× the compute of the largest budget (not 1.75×)". In the implementation of `train_wsd_branches`, the trunk runs to 0.8 × 512 = 409 steps, and the three decays take 26 + 52 + 103 steps. The total is 590 steps, which is **1.15×** the largest budget of 512 steps (0.8 + 0.2 × 1.75 = 1.15; 1.35 is probably a calculation error from 1 + 0.2 × 1.75).

### Detail 3: 13/06_quality_ablation.py, 13/07_mixture_ablation.py

06 (README → measured, seed 0 / 1):

- Raw dirty web pages: English bpb 3.126 / 3.215 → 3.115 / 3.207; Chinese 3.400 / 3.494 → 3.382 / 3.492; loss of the last step 4.546 / 4.715 → 4.511 / 4.706
- After filtering: English 3.086 / 3.187 → 3.083 / 3.166; Chinese 3.316 / 3.412 → 3.315 / 3.375; loss of the last step 4.659 / 4.779 → 4.656 / 4.736
- Paired differences: English 0.040 / 0.028 → 0.032 / 0.041; Chinese 0.084 / 0.081 → 0.067 / 0.117
- The qualitative conclusion of L218 is mostly true: all 4 paired differences are positive, training on dirty data gives a lower loss, and a different seed can change the result by about 0.1. But "the difference from a different seed is larger than the difference from the data" is no longer strictly true. The paired difference 0.117 for Chinese with seed 1 is larger than the maximum seed difference 0.110.

07 (README → measured, seed 0 / 1):

- Balanced: English 3.192 / 3.155 → 3.163 / 3.155; Chinese 3.486 → 3.480; code 3.760 → 3.750; mean 3.381 → 3.372
- English-heavy: English 3.018 (seed 1) → 3.115; Chinese 3.818 / 4.057 → 3.818 / 4.061; code 3.652 / 3.794 → 3.651 / 3.825; mean 3.482 → 3.506
- Code-heavy: the differences are within 0.001
- L233 "English is about 0.20 lower than balanced": measured, the mean of the two seeds is only 0.137 lower (seed 0 is 0.235 lower, seed 1 is 0.040 lower). L234 "English is about 0.24 lower": measured 0.224. Both conclusions "the two seeds go in the same direction" are true.
- Note: the docstring of script 07 says "about 8 minutes on one thread", and the README says "about 4 minutes". The measured time was 4.6 minutes.

### Detail 4: three small numbers in Chapter 14

- `03_online_softmax.py` ②: README L218 says "with 10,000 random scores (float64), the max difference is 1.1 × 10⁻¹⁶". Measured: 8.3e-15. The script has used scores with std 10 since its first commit (confirmed with `git log -p`). In the same run, the 5.8e-15 of ③ agrees with the README, so a change of the random numbers is not probable. The 1.1e-16 in the README is probably a copy error or comes from an earlier version. Both values are in the range of float64 rounding error, so the conclusion does not change. Recommendation: change the value to 8.3 × 10⁻¹⁵.
- `04_tiled_attention.py` ① (README → measured): 64×64 vs naive 1.0e-15 → 8.9e-16; 128×32 7.8e-16 / 6.7e-16 → 9.4e-16 / 8.0e-16; 32×128 1.2e-15 / 4.4e-16 → 1.1e-15 / 6.1e-16; 100×70 7.8e-16 / 7.8e-16 → 6.7e-16 / 7.8e-16. ②: max difference with P recalculated 2.8e-16 → 2.2e-16. All of these are float64 rounding noise, and the conclusion "about 10⁻¹⁵" does not change.
- `06_ddp_by_hand.py`: max difference of the final parameters 6.9e-17 → 5.6e-17 (gradient accumulation and 2 processes, two places: L298 and the summary table at L431). This is rounding noise.

### Detail 5: 15/03_context_extension.py, 15/04_anneal_mixture.py

03 (README → measured). The script prints the zero-shot table with the title "No training, only a new RoPE", and the table after fine-tuning with the title "After 150 fine-tuning steps at length 256":

- Zero-shot table: (a) 1.578 / 1.646 / 1.879 / 1.578 / 1.714 / 2.111 → 1.580 / 1.641 / 1.861 / 1.580 / 1.701 / 2.082; (b) PI 3.461 … 3.590 → 3.446 … 3.579; (c) YaRN 1.654 … 1.685 → 1.651 … 1.686; (d) 1.626 … 2.058 → 1.621 … 2.030
- Table after fine-tuning: (a) 1.587 / 1.570 / 1.561 → 1.598 / 1.582 / 1.570; (b) 1.701 / 1.688 / 1.679 → 1.697 / 1.685 / 1.675; (c) 1.578 / 1.564 / 1.555 → 1.591 / 1.575 / 1.565; (d) 1.591 / 1.574 / 1.568 → 1.603 / 1.585 / 1.578
- Quotes in the text: L156 "1.578 … 2.111" → 1.580 … 2.082; L252 "1.685" and "1.654 vs 1.578" → 1.686, and 1.651 vs 1.580; L253 "1.578 increases to 3.461" → 1.580 → 3.446; L254 "2.058 vs 2.111" → 2.030 vs 2.082
- All qualitative conclusions are true. In the zero-shot case, YaRN is the most stable and PI is the worst. A larger base frequency alone helps almost nothing. After fine-tuning, YaRN is the lowest at all three lengths. Two quantitative ranges in L255 are a little off: "a difference of 0.006–0.013" is 0.005–0.013 measured, and "PI is still about 0.12 behind" is 0.106–0.110 measured. The drift between two runs (about 0.01) has the same size as the differences between the methods. This supports L256: "probably within the seed variation".
- Note: L258 says "not even 15,000 steps". The real numbers are 1500 pretraining steps, then 150 fine-tuning steps for each setting. Probably 1500 was written as 15,000 (literally, the statement is not wrong).

04 (README → measured, English / Chinese / code / mean):

- Branch point 2.759 / 3.148 / 2.746 / 2.884 → 2.741 / 3.158 / 2.732 / 2.877; A 2.700 / 3.097 / 2.632 / 2.810 → 2.707 / 3.102 / 2.641 / 2.817; B 2.549 / 3.037 / 2.472 / 2.686 → 2.544 / 3.037 / 2.472 / 2.684; C 2.803 / 3.168 / 2.223 / 2.731 → 2.797 / 3.163 / 2.186 / 2.715; D 2.642 / 3.097 / 2.091 / 2.610 → 2.631 / 3.099 / 2.091 / 2.607
- L88: decrease of the code bpb 0.160 / 0.409 / 0.541 → 0.170 / 0.456 / 0.551. L90 "B is 0.06–0.16 lower than A": measured about 0.07–0.17. All qualitative conclusions are true.
- **A text error that does not depend on the run environment**: L95 says "a tiny experiment at the level of 1.3M parameters". 04 uses `tt.Config(seq_len=...)` of Chapter 9 (dim 128, 4 layers, FFN 352). Its parameter count is 836,992 (about 0.84M; L233 of the Chapter 9 README gives the same value). 1.3M is the parameter count of `configs/tiny`, so the two were probably mixed up.

### Other notes (not mismatches)

- The run-time hints are usually too long. 11/04 "about 10 seconds": measured 2.2 seconds. 12/03 "about 7 minutes": measured 2.2 minutes. 12/04 "about 15 minutes": measured 3.1 minutes. 12/06 "about 5 minutes": measured 1 minute. The README already says that the times were measured on a busy shared CPU, so the hints can stay.
- 13/03: L129 "at s=0.3, the measured candidate rate is 0.22" is a historical number: the author describes an old bug. The current code gives 0.001, which agrees with the formula. The text does this on purpose.
- 13/05: L276 "3 poems, for example 《御街行》 ("Yu Jie Xing")": the log prints only the first one.

## 2. New GPU scripts and README sections

| File | Content | Run time alone (after the lock was acquired) |
|---|---|---:|
| `chapters/14-pretraining-engineering/code/08_gpu_matmul_attention.py` | ① Throughput of square matmuls in FP32 / TF32 / BF16 / FP16 (n = 1024–8192) and the error relative to FP64; ② BF16 autocast on the CPU / GPU for the same Linear as in `02_precision.py` ⑥, with the `allow_bf16_reduced_precision_reduction` switch; ③ attention forward + backward pass: naive code vs SDPA with only FlashAttention allowed, peak memory and time for T = 1K–16K; ④ the profiler shows which CUDA kernels the GQA attention of zero (`enable_gqa=True`) really calls in BF16 / FP32, and whether each forced backend can handle GQA | 13.4 s |
| `chapters/14-pretraining-engineering/code/09_gpu_train_step.py` | Main-line model (689.5M, `Transformer` of zero + fused AdamW), micro batch 1: FP32 / BF16 / BF16 + activation checkpointing × T = 1024 / 4096. It measures the "memory added after the forward pass" (`fwd added`) and the peak of the full step, and compares them with `memory_calc`. It also measures the time per step, MFU, HFU, and the utilization after causal halving. At T = 4096 with checkpointing, it finds the maximum micro batch: formula prediction vs measurement | 53.8 s |
| `chapters/15-midtraining-long-context/code/05_gpu_long_context_cost.py` | One Block of `configs/main/longctx.toml`, 32,768 tokens per call, T = 4K → 32K: the factor of the time per token (compared with two formulas: not halved / causal halving), real TFLOPS, and MFU by the zero definition; activation memory per token, compared with `memory_calc` | 17.4 s |

- The three scripts have these properties. Without CUDA, they print "This script needs a CUDA GPU. Without a GPU, skip it: the chapter shows one result from an RTX 3090." and exit 0 (15/05 prints "a result" in place of "one result"; verified with `CUDA_VISIBLE_DEVICES=`). The seed is fixed. Each timing has a warm-up, uses CUDA events, and takes the median. At the start, they print the GPU name and the torch version. `UV_NO_SYNC=1 uv run ruff check` passes. They import `zero` read-only and do not change `zero/` or `configs/`.
- Each script ran two rounds on GPU0. We used the first round to adjust the output. In 09, the first round showed that FP32 does not fit at T = 2048, so we changed to T = 1024 / 4096. 08 got the cuBLAS reduction switch and the check of the FP32 kernels. 15/05 got a column for the MFU definition. All numbers in the README come word for word from the output of the **second round (final version)**: `gpu/08_try2.log`, `gpu/09_try2.log`, `gpu/15_05_try2.log`. For the same configuration, the two rounds differ by at most about 10% in the same BF16 matmul cell (a lower clock speed because of the power limit). The MFU of one training step differs by less than 1% (55.9% / 55.5%).
- README: Chapters 14 and 15 each got a new section `## GPU measurements (one RTX 3090)`. It is directly before `## From minimal code to production code`, with a `---` between them, as in the section style of the full book. In Chapter 14, the section has 4 subheadings, one for each experiment (matrix multiplication / attention / memory / speed and MFU). The interpretation of MFU describes the effect of the 240 W power limit. These are additions only, with no change to the existing text (`git diff --numstat`: +94 / −0, +31 / −0).
- Chapters 11, 12, and 13 did not get a GPU section. The code of Chapters 11 and 13 is evaluation statistics and data processing, which do not relate to a GPU. The budget of Chapter 12 is fully based on the H100 peak and an MFU assumption. A conversion of 3090 measurements does not give a more useful conclusion (the problem of the MFU definition is already in Chapters 14 and 15). We did not add a section, so that the text stays focused.

## 3. GPU results compared with the README estimates: key points

1. **Mixed precision**: on the 3090, BF16 matmul reaches up to 60.4 TFLOPS (85% of the spec value 71), and FP32 reaches 16.9 (48% of the spec value 35.6). BF16 is 3.6 times faster. In one training step (T = 1024), BF16 is 2.7 times faster than FP32. The relative error against FP64 is 2.9e-3 for BF16 and 3.6e-4 for FP16 (a factor of 8 = 3 more mantissa bits). The same Linear as in 02 ⑥ gives 2.78e-3 on the CPU (the same as in the text) and 3.25e-3 on the GPU. With `allow_bf16_reduced_precision_reduction` of cuBLAS turned off, the GPU also gives 2.78e-3.
2. **Attention**: the peak memory of the naive code is about 2 times the theoretical value of S + P, and it grows with T². T = 16K does not fit. The memory of FlashAttention grows linearly (514 MiB at T = 16K), and FlashAttention is 3.1–5.4 times faster.
3. **SDPA backend**: in BF16, the GQA attention of zero (`enable_gqa=True`) calls `pytorch_flash::flash_fwd_kernel` and three `flash_bwd_*` kernels. The forced memory-efficient backend + GQA gives an error; FlashAttention / cuDNN work. **In FP32, SDPA falls back to the math backend** (no fused kernel, and a separate softmax). This backend stores the T × T matrix explicitly.
4. **memory_calc**: in BF16, the "memory added after the forward pass" (activations + weight copies) differs from the formula by 1–2%. The peak of the full step is 5–16% lower than the formula (the formula is a conservative upper bound). T = 4096, BF16, micro batch 1: the formula gives 24.02 GiB, so by the formula it does not fit on the 23.6 GiB card. The measurement is 22.17 GiB, so it fits. Maximum micro batch with checkpointing: formula prediction 3, measured 3. In FP32, the formula is too low by about 2 GiB (T = 1024), because of the math backend above. Chapter 15: the activations per token per layer are 92,868 bytes (formula 92,840), independent of T.
5. **Activation checkpointing**: the activations decrease from 12.33 to 1.75 GiB (14%). Each step is 27% slower (723 → 919 ms), which agrees with "about 1/3 more compute". MFU 55.5% → 43.6%, HFU 57.1%.
6. **MFU**: for the main-line model on one GPU in eager mode at T = 4096, MFU is 55.5% (the zero / PaLM definition). The real utilization after causal halving is 44.2%. Chapter 15: at 32K, the "MFU" of one layer by the zero definition is 99%, and the real value is 56%. At 32K, one token costs only 2.53 times as much as at 4K (the formula without halving gives 4.06, and with causal halving 2.96; all values are for one layer).

## 4. Items for the main process to decide

The GPU results directly confirm or refute the statements below in the README / configuration. I did **not** change these sentences, and I did not change `zero/` or `configs/`:

1. **Confirmed** (3090 + PyTorch 2.11; check again on the H100 in stage 6). Chapter 14, Section 4.4 ("How zero uses it"): "whether `enable_gqa=True` can use the Flash backend on a GPU is **not verified on a GPU yet**". The row of `04_tiled_attention.py` in the "From minimal code to production code" table: "which backend is really used is **not verified on a GPU yet**". The item "One GPU, BF16 + SDPA" in the verification table of "Main-line progress". Result: in BF16, the FlashAttention kernel is used.
2. **Confirmed (direction)**: Chapter 14, Section 2: "a conservative upper bound in eager mode". On the 3090, the formula gives a full-step peak that is 5–16% too high. "Maximum micro batch about 5" in the main-line progress table can be conservative on the H100; stage 6 decides.
3. **Confirmed**: the activation checkpointing of `zero/model.py` works on CUDA (memory and time as expected; we did not do a gradient parity check on the GPU). Also: the docstring of `zero/tools/memory_calc.py` still says "**zero does not implement activation checkpointing yet**". This is out of date (`zero/model.py` implements it).
4. **Needs an added explanation**: the FP32 formula of `memory_calc` is not correct on CUDA (FP32 + GQA falls back to the math backend, which stores an extra T × T matrix). The main line uses BF16 by default, so the main line is not affected. If you change it, add a sentence to the part "What the estimate includes, and its limits" of `memory_calc`.
5. **Refuted / must be rewritten**: Chapter 15, L269: "for the same number of tokens, training with 32K sequences costs 3.84 times as much (… saves about half of the attention operations, but the ratio stays the same)". The summary at L294: "about 3.8 times the cost of 4K". Measured on one layer, 32K costs only 2.53 times as much. Even the formula with causal halving (2.96 times) gives a value that is too high. Treat 3.84 times as an upper bound. Original L408 (now L439): "the real MFU with 32K sequences can be lower". On one layer of the 3090, the utilization at 32K is higher than at 4K for both definitions (real 48% → 56%; zero definition 62% → 99%). Recommendation: at least add the warning "with long sequences, the MFU in the zero log is too high; halve the attention term before you compare".
6. **MFU definition**: Chapter 14, Sections 1 and 6, and the MFU 0.4 assumption of `estimate_cost` all use the formula without halving. At T = 4096, MFU 0.4 means a real utilization of about 0.32 (5.55 / 6.96). This does not affect the conversion to the $ budget, because the same definition goes in and out. But the measurement in stage 6 must use the same definition for the comparison. In the 32K stage of Chapter 15, an estimate with the MFU of 4K gives a cost that is too high.
7. **Long-context configuration**: `configs/main/longctx.toml` uses FSDP and does not turn on `activation_checkpointing` (it inherits false from pretrain.toml). Its comment says "The activations of 32K sequences are large, so FSDP2 saves memory". The "From minimal code to production code" table of Chapter 15 also says "makes room in memory for the activations". But FSDP does not split the activations. Measured: 92,868 bytes per token per layer × 28 layers × 32,768 ≈ 79 GiB. `memory_calc configs/main/longctx.toml --strategy fsdp` gives 101.24 GiB (more than 80 GiB). Only with `--checkpointing` does it give 29.12 GiB. This run measured that the formula is about 8% conservative. Even with this discount, the model does not fit on an 80 GB H100 without activation checkpointing. Recommendation: write `activation_checkpointing = true` in `longctx.toml` (or verify it first in stage 6).
8. The start of Chapter 14 says "all code in this chapter runs on a CPU". Now code/ has two GPU scripts (08, 09), so this sentence must be made consistent (I did not change it).
9. Please decide two things. First: update the values of the training scripts in Chapters 12, 13, and 15 from the reruns on this machine, or write them in a form that cross-machine drift does not affect (see "Common cause of the mismatches"). Second: the two text errors that do not depend on the run environment ("1.35×" in Chapter 12 must be 1.15×; "1.3M parameters" in Chapter 15 must be about 0.84M).
