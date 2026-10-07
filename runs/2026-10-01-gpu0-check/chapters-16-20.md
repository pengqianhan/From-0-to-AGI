# GPU0 verification: Chapters 16–20

**English** · [中文](chapters-16-20.zh.md)

Date: 2026-10-01. Branch: `gpu0-verification`. Scope: `chapters/16-*` to `chapters/20-*` (SFT / distillation / DPO / RL / release).

## Environment

- CPU: AMD Ryzen Threadripper PRO 3995WX (64 cores, Zen 2, AVX2, no AVX-512). This is a shared server. During the runs, the load average varied between 5 and 40 (other agents and video rendering ran at the same time).
- PyTorch 2.11.0+cu128, CUDA 12.8.
- GPU: NVIDIA GeForce RTX 3090 (GPU0 only, `CUDA_VISIBLE_DEVICES=0`). All GPU commands waited in the queue of `gpu0.lock`. Before the runs, we confirmed `torch.cuda.device_count() == 1`.
- CPU reproduction command: `CUDA_VISIBLE_DEVICES= UV_NO_SYNC=1 uv run python chapters/.../code/xx.py`, with at most 4 processes at the same time. The raw output is in `/tmp/claude-1006/-home-phan635-Opensource-From-0-to-AGI/1d40fca9-4469-49a7-88b9-4e96bf8f0051/scratchpad/ch16-20/logs/` (reruns in `logs_rerun/`, GPU output in `gpu/`, probes in `probe/`).

## 1. CPU reproduction table

"CPU time" is the wall-clock time of the full `uv run` process. We measured it with at most 4 runs in parallel on a busy machine, so use it only for the order of magnitude. Categories: `match` / `timing only` / `mismatch` / `error`. All 23 scripts returned rc=0: **no errors**.

| Ch. | Script | CPU time | Category | Notes |
|---|---|---:|---|---|
| 16 | `01_chat_template.py` | 0.1 s | match | Full rendered text, per-segment table, 1126 / 197 / 17.5%, characters per role, last 40 characters, and the parity check with `render_text` ("full conversation same, generation prompt same"): all correct |
| 16 | `02_loss_mask.py` | 2.2 s | match | 65 → 89, 137 / 53 (39%), 1.091 / 1.500: all correct |
| 16 | `03_sft_tiny.py` | 351 s | **mismatch** | The first run must train the base model of Chapter 10 + 3 SFT models. The trained base model is not the same as the one that the README used, so the downstream numbers drift at the 2nd–3rd decimal place. See Detail 1 (time: the README says "about 17 minutes under heavy load"; here 5.9 minutes) |
| 16 | `04_packing.py` | 3.1 s | **mismatch** | The packing statistics (200 conversations, 130–390, 43% / 91%, 94 windows, 2.1 conversations, position 154) are all correct. The three loss lines of cross-contamination differ, and the **direction is reversed**. See Detail 2 |
| 16 | `05_smoke_samples.py` | 2.2 s | cannot compare (no error) | The repository root has no `out/smoke/`, so by design the script only prints "out/smoke/ not found. First run `uv run python -m zero.smoke` …". As the task instructions said, we did not run the smoke test again |
| 17 | `01_soft_labels.py` | 2.3 s | match | Soft-label table, temperature table, two-row gradient table, 1.16e-11 / 3.10e-11, gradient norms 0.268 / 0.100 / 0.035 / 0.011 (0.6741 / 64 = 0.01053 → 0.011), and the three rows of the zero parity check: all correct |
| 17 | `02_toy_distill.py` | 106 s | **mismatch** | Rows A and R and the teacher (429,665 parameters, 2.439) are bit-identical. Rows B, C, and D changed, B the most (validation 3.362 → 3.382). See Detail 3. The time agrees with the README ("about 2 minutes") |
| 17 | `03_forward_reverse_kl.py` | 9.4 s | match | Teacher mass 0.596 / 0.016 / 0.388 and the three rows of μ, σ, and mass: all correct |
| 17 | `04_rejection_sampling.py` | 2.5 s | match | Funnel 1600 / 1303 / 596 / 497 / 172, 291 → 43, 24 + 19, 68.9%, and 0 kept for the 21 tasks without tools: all correct |
| 17 | `05_shared_vocab.py` | 2.2 s | match | Segmentation, error message, and the three rows of the vocabulary parameter ledger: all correct |
| 18 | `01_bradley_terry.py` | 4.1 s | match | σ table, training table, 0.803 / 0.805, 0.428393: all correct |
| 18 | `02_rlhf_kl.py` | 5.2 s | match | Table of 8 answers, 7 rows of the β sweep, 4 PPO rows, 1.6112, 0.00021: all correct |
| 18 | `03_dpo_derivation.py` | 4.6 s | mismatch (rounding level) | Step (1), "largest difference for 5 random π": README 2.2×10⁻¹⁶, measured 3.3×10⁻¹⁶ (float64 rounding, which depends on the machine / BLAS). (2), (3), ⑤, ⑥, and ⑦ are all correct |
| 18 | `04_toy_dpo.py` | 26 s | **mismatch** (small) | All numbers drift at the 3rd–4th decimal place, and the conclusions do not change. See Detail 4. The time agrees with "about half a minute" |
| 18 | `05_dpo_pitfalls.py` | 35 s | **mismatch** | The tables of Pitfalls 2 and 3 are bit-identical. In Pitfall 1, the two rows lr = 1e-2 and 5e-2 differ a lot, and one sentence of the text is no longer true. See Detail 5 |
| 19 | `01_reinforce_bandit.py` | 1.0 s | match | Three tables and 1589×: all correct |
| 19 | `02_grpo_from_scratch.py` | 7.5 s | match | Imitation table, GRPO table, zero-variance groups 0.56 → 0.94, clipping 0.00–0.02, sample table for ±0.94, parity check 0.00e+00 / −0.046469: all correct |
| 19 | `03_reward_hacking.py` | 13 s | mismatch (one sentence of text only) | The tables and the samples are bit-identical. But "in the last batch, all guess 7" is not correct. See Detail 6 |
| 19 | `04_tool_env_rewards.py` | 0.1 s | match | 7-row reward table: all correct |
| 20 | `01_blockwise_quant.py` | 0.3 s | match | w / q / ŵ of the first block and the 8-row error table: all correct |
| 20 | `02_quantize_tiny_model.py` | 48 s | **mismatch** | Same cause as 16/03 (a different base model of Chapter 10). The full table and the generated samples changed, and some specific sentences of the text are no longer true. See Detail 7 |
| 20 | `03_memory_calculator.py` | 0.1 s | match | 689.5M / 83.9M, 4-row size table, KV 112 KiB / 0.44 / 3.50 GiB: all correct |
| 20 | `04_model_card.py` | 0.1 s | cannot compare (no error) | There is no `out/smoke/`, so by design the output is a skeleton with "TBD after training" in all places (34 placeholders; the README shows the version with smoke-test results, which has 15) |

**Determinism check**: we ran `18/04`, `18/05`, and `17/02` again on this machine, once each. The outputs were identical word for word (only the time differed). So the mismatches above do not come from randomness. **This machine and the machine used for writing calculate different floating-point numbers**. The most probable cause is a different CPU instruction set / different math-library kernels: they give rounding differences, and training amplifies them.

**Handling of `code/out/`**: the first run of `16/03` trains again and overwrites 3 committed `ch16_*.json` files (the curves trained on this machine are different). We copied the new files and the diff to `ch16-20/ch16_out_new/` in the scratchpad, then restored the files with `git checkout`. In `git status`, `chapters/16-sft/code/out/` is clean. The trained files `ch16_masked.pt`, `ch16_unmasked.pt`, and `ch16_small40.pt` (`*.pt` is in gitignore) stay in place. If we delete them, the next process that runs `03` (or the video rendering) trains again and overwrites the committed json files again. `chapters/10-inference/code/out/tiny_kv4_s600.pt` was also trained in this first run (it is also in gitignore).

### Details of the mismatches

**Detail 1: `16-sft/03_sft_tiny.py`** (root cause: the base model of Chapter 10, `tiny_kv4_s600.pt`, is not in git. The first run trains it again, and the model trained on this machine is not the same as the model used for writing)

| README | Measured |
|---|---|
| Base model continues the text (no template): `\nCAMILLANUS:\nI will the son the shall the prove the shall th` | `\nCAMILLA:\nI will the sent the such the shall the prince the ` |
| Base model "answer" with the chat template: `'The shall the then the shally the the the the the the th'` | `'The shall the the the shalleath the the the thear the the th'` |
| Base model loss on the validation set: assistant tokens only 4.640, all tokens 4.160 | 4.669, 4.161 |
| `'Compute 20+55.'` → `"a": 19, "b": 15` | `"a": 19, "b": 12` (the other 3 answers are identical word for word) |
| Table, validation loss: 0.102 / 0.110 / 0.366 | 0.104 / 0.110 / 0.350 (the three columns `format`, `name ok`, and `args ok` are all correct) |
| By type: 310 argument values 1.146, the other 3364 tokens 0.006 | 310 tokens 1.141, 3364 tokens 0.008 |
| Curve of 40 samples: step 1 4.561 / 4.537; 100 0.086 / 0.316; 200 0.046 / 0.343; 300 0.055 / 0.366 | 4.597 / 4.559; 0.084 / 0.316; 0.055 / 0.343; 0.065 / 0.350 |

All conclusions are still true: the model learns the format first; it does not copy the arguments correctly; with mask and without mask are almost equal; with 40 samples, the validation loss goes up again after step 100.

**Detail 2: `16-sft/04_packing.py`** (same root cause: it uses the model that 03 trained)

| | README | Measured |
|---|---:|---:|
| alone | 0.5596 | 0.5095 |
| packed after A (Paris), causal mask | 0.4967, max logits difference 9.90 | **0.7323**, max difference 8.39 |
| packed after A, document mask | 0.5596, max difference 4.92e-05 | 0.5095, max difference 8.15e-05 |

README Section 5.2 says: "the loss even **decreased**: … B 'copies the homework of its neighbor'". On this machine, the loss **increased**. Only one conclusion is robust: "with the normal causal mask, A changes the logits of B a lot (max difference 8–10); the document mask makes the result the same as for B alone". The direction of the loss change depends on the specific weights. Recommendation: change this sentence into a statement that does not depend on the direction. For example: "the loss changed from 0.56 to 0.50 — on a model trained on another machine, it increased to 0.73. The direction is not fixed, but the loss changed". You decide.

**Detail 3: `17-distillation/02_toy_distill.py`**

| Training signal | README (train / validation / per seed) | Measured |
|---|---|---|
| A | 2.678 / 3.679 / 3.626 · 3.739 · 3.673 | Same |
| B | 3.302 / 3.362 / 3.327 · 3.387 · 3.371 | 3.360 / **3.382** / 3.348 · 3.409 · 3.389 |
| C | 2.867 / 3.290 / 3.258 · 3.326 · 3.286 | 2.866 / 3.286 / 3.252 · 3.323 · 3.283 |
| D | 2.693 / 3.344 / 3.326 · 3.374 · 3.332 | 2.692 / 3.341 / 3.321 · 3.369 · 3.333 |
| R | 3.123 / 3.143 | Same |

A and R do not use the teacher, and they are bit-identical. B, C, and D all use the teacher. The teacher validation value 2.439 also agrees (to the 3rd digit). But very small differences of the teacher weights are amplified most in B, because B **samples** 20,000 characters from the teacher. The conclusions of the text do not change: C is the best and 0.39 bit better than A; B and D are both better than A; D is between them. "Only 0.15 bit from 'enough data'" is 0.14 measured.

**Detail 4: `18-preference-alignment/04_toy_dpo.py`**: SFT reference model 0.369 / 1.000 / 0.360 → 0.368 / 1.000 / 0.357. Each row of the training table changes at the 3rd–4th decimal place (for example, at step 150, log π(chosen) −0.551 → −0.550, and log π(rejected) −8.919 → −8.913). Held-out 0.369 → 0.430 becomes 0.368 → 0.430, and correct samples 0.360 → 0.413 become 0.357 → 0.412. The β sweep table also changes at the 3rd digit (β = 0.3: 0.2601 / +1.540 / +5.13 / +0.722 / −4.411 / 0.529 → 0.2599 / +1.544 / +5.15 / +0.723 / −4.425 / 0.530). The conclusions do not change.

**Detail 5: `18-preference-alignment/05_dpo_pitfalls.py`**, table of Pitfall 1:

| lr | README (margin / train acc / held-out P(correct) / well-formed samples / correct samples) | Measured |
|---|---|---|
| SFT | — / — / 0.369 / 1.000 / 0.360 | 0.368 / 1.000 / 0.357 |
| 1e-4 | +0.05 / 0.95 / 0.432 / 1.000 / 0.447 | Same |
| 1e-3 | +0.60 / 0.95 / 0.430 / 0.995 / 0.413 | Correct samples 0.412, the rest the same |
| 1e-2 | +4.26 / 1.00 / 0.071 / **0.473** / 0.063 | +4.20 / 1.00 / 0.068 / **0.618** / 0.068 |
| 5e-2 | +6.61 / 1.00 / 0.031 / 0.300 / 0.032 | +5.73 / 1.00 / 0.000 / 0.172 / 0.000 |

The training in the two rows with a large learning rate is very unstable, so the floating-point differences are amplified. The text says "at lr = 1e-2 … more than half of the sampled answers do not even have the correct format". This is not true on this machine (well-formed 0.618, so fewer than 40% have a wrong format). The conclusion "a high margin ≠ a good model" is still true. Pitfall 2 (0.353 → 0.223 and others) and Pitfall 3 (0.95 / 0.84 vs 0.83 / 0.63) are bit-identical.

**Detail 6: `19-reinforcement-learning/03_reward_hacking.py`**: all numbers and samples are bit-identical, so this machine gives the same result as the machine used for writing. But Section 7.2 says "the answers to the tool tasks are still mostly guesses (in the last batch, all guess 7)". This contradicts the output of the script itself: one printed sample is `2+1 → <call> 字 8 </call>` (`字` is the "word" token of the toy vocabulary). I also counted all 120 tool-task answers in the batch at step 120 with the fixed reward. 77 guess 7 (64%). The others are 0 (14), 8 (7), 1 (6), 5 (6), 3 (5), and other values. Recommendation: change it to "most guess 7 (about two thirds)".

**Detail 7: `20-release/02_quantize_tiny_model.py`** (same root cause as Detail 1)

| Method | README val loss / Δ loss / top-1 agreement | Measured |
|---|---|---|
| fp32 | 1.8385 / — / 100.0% | 1.8308 / — / 100.0% |
| INT8, 1 scale/tensor | 1.8386 / +0.0002 / 99.4% | 1.8310 / +0.0002 / 99.4% |
| INT8 block32 (≈Q8_0) | 1.8384 / −0.0001 / 99.6% | 1.8307 / −0.0001 / 99.5% |
| INT4, 1 scale/tensor | 1.8525 / +0.0141 / 90.3% | 1.8456 / +0.0147 / 91.1% |
| INT4, 1 scale/row | 1.8446 / +0.0061 / 93.5% | 1.8385 / +0.0076 / 91.9% |
| INT4 block32 (≈Q4_0) | 1.8435 / +0.0050 / 94.1% | 1.8374 / +0.0066 / 93.3% |
| INT4 2-level (≈Q4_K) | 1.8437 / +0.0052 / 94.9% | 1.8355 / +0.0046 / 94.6% |
| INT3 block32 | 1.8776 / +0.0391 / 85.9% | 1.8729 / +0.0421 / 85.4% |
| INT2 block32 | 2.3439 / +0.5055 / 46.4% | 2.4079 / +0.5771 / 44.1% |

(The column of weight sizes is all correct.) These specific statements are no longer true:

- "two-level scales and normal blocks also cannot be separated by the loss (1.8437 vs 1.8435, agreement 94.9% vs 94.1%)": on this machine, two-level scales are clearly better (1.8355 vs 1.8374, 94.6% vs 93.3%);
- "the cost of 3 bit is 8 times the cost of 4 bit": on this machine, 0.0421 / 0.0066 ≈ 6.4 times. "The most probable character changes at 1 position in 17": on this machine, at about 1 position in 15;
- "greedy generation diverges from fp32 at the 9th character ('And the course…' vs 'And the such…')": on this machine, fp32 generates `Ay so the soul, and the prove…`, and INT4 block32 generates `Ay the send the prove…`, so they diverge at the 4th character. 2 bit generates `How, would are a worlous…`, not "is is apeeng".
- Still true: "8 bit is almost lossless, and the first 60 characters of greedy generation are identical to fp32", "4 bit starts to have a cost", "below 4 bit is a cliff", and "the effect of quantization depends on the specific weights".

**Recommendation (no code change; you decide)**: in 16/03, 16/04, 20/02, and in other chapters, all numbers that depend on `tiny_kv4_s600.pt` of Chapter 10 do not agree with the text. The reason is that "the first run trains the base model again on this machine". To let readers reproduce the numbers in the text, you can commit these small `.pt` files (about 3.4 MB each) as assets. Or add a note in the text: "on different machines, the 2nd–3rd decimal places differ, and some qualitative descriptions (such as the direction of the loss in 16/04) can also change".

## 2. GPU measurements

All numbers written into the README come from GPU0 (`CUDA_VISIBLE_DEVICES=0`, via `gpu0.lock`). The final version of each script ran once. The raw output is in `ch16-20/gpu/ch16_final.log` and `ch20_final.log` in the scratchpad (the other `*_try*.log` files are from debugging). Both scripts pass `uv run ruff check`. Without CUDA, both print the required message and exit 0 (verified with `CUDA_VISIBLE_DEVICES=`). The coordinator later allowed GPU1 for debugging. But the permission system refused my GPU1 commands, so I used only GPU0 the whole time.

### Chapter 16: new `chapters/16-sft/code/06_gpu_packing.py`, and a new section "GPU measurements (one RTX 3090)" in the README

- It relates to Section 5.1 of the text (padding wastes compute) and Section 5.3 (the trade-off: zero does not isolate packed conversations; the advantage is that zero "can use the fastest causal attention kernel directly").
- ① With the `Transformer` of zero and the main-line shape of `configs/main/sft.toml` (689.5M), a forward + backward pass on the same 200 conversations as in 04: "one per row, pad to 513" takes 14.57 s (3,019 real tokens/s), "one per row, batch max" (padded to the longest in the batch) 10.32 s (1.41×), and "first-fit" packing 7.03 s (**2.07×**). The two passes differ by ≤ 0.2%.
- ② An 8192 window holds 40 conversations, and the document mask has only 2.9% of the pairs of the causal mask. Forward + backward pass of one attention layer: "causal FlashAttention (mixes docs)" 18.20 ms; "doc mask: boolean matrix to SDPA" → PyTorch selects the MATH implementation, 227.95 ms, 16.70 GiB; "doc mask: FlexAttention" 2.48 ms (7.34×); "doc mask: varlen FlashAttention" 2.43 ms (7.49×). The max output difference between normal causal and varlen is 4.62 (cross-contamination). The three document-mask versions differ by ≤ 1.56e-02 (BF16 rounding).
- This makes the "trade-off" in Section 5.3 clearer. In a long window, isolation of conversations with varlen / Flex is not slower: it removes most of the attention compute. Only the "boolean matrix" version is slow. **I did not change Section 5.3 of the text or the label "not implemented yet, not verified on a GPU yet"**. You decide whether to change the plan of zero because of this result.
- Total run time 77.7 s (wall-clock). In part ①, each layout ran only two passes, and we took the median (7–15 s per pass; three passes would take more than 2 minutes). The README and the script both say this.

### Chapter 20: new `chapters/20-release/code/05_gpu_quant_matvec.py`, and a new section "GPU measurements (one RTX 3090)" in the README

- It relates to Section 3.1 of the text ("the bottleneck of decode is bandwidth; with weights of half the size, reading is almost twice as fast") and Section 4 (llama.cpp on a laptop vs vLLM on a server).
- All 197 matrices of the main-line model, batch 1, CUDA Graph: "BF16, cuBLAS (F.linear)" 2.567 ms (537 GB/s); "BF16, torch.compile kernel" 2.589 ms; "INT8, compiled fused dequant" (one scale per row, torch.compile fuses the dequantization) 2.294 ms, **only 1.12× faster**; "INT4, tinygemm kernel" (groups of 32, 5 bit) 1.157 ms, **2.22× faster**; "INT8, dequant to BF16 first" (dequantize first, then multiply) 11.527 ms (0.22×). Reference bandwidth from a copy of a 1 GiB tensor: 851 GB/s.
- Batch sweep: INT4 is 2.27× and 2.05× faster at B = 1 and 8. From B = 32, it is slower (0.82×, 0.39×, 0.28×).
- Bandwidth for each matrix type: wk (2.5 MiB) reaches only 429 GB/s in BF16. The smaller the matrix, the less of the bandwidth it uses.
- Conclusion written in the README: quantization gives a speedup, but the size of the speedup depends on the kernel (INT8 with a general compiled kernel is only 1.12×). The matrices of this 0.7B model are too small, and this also removes part of the gain. With a large batch, quantization is slower.
- Total run time 22 s (wall-clock).
- Another observation during debugging (not in the README): PyTorch's built-in `torch._weight_int8pack_mm` on CUDA takes 13.2 ms at batch 1 (0.20×, only 52 GB/s) and 94 ms at B = 32. This kernel is not optimized for decode, so the final version uses torch.compile to make the INT8 kernel. Also, the BF16 bandwidth of the single lm_head matrix was 560 and 683 GB/s in two runs (the timing of a single kernel is sensitive to the clock frequency and to the algorithm that cuBLAS selects). The interpretation in the README does not quote this cell.
- The power limit of GPU0 is 240 W (the coordinator told us). The absolute bandwidth values can be too low. The ratios (2.22×, 0.28×, and others) should be less affected.

### Chapters 17, 18, 19: no GPU scripts, and the reasons

- **Chapter 17**: the minimal code is a character-level MLP with 8,905 / 429,665 parameters. A GPU only makes it faster; it does not make any argument clearer. "The cost ratio between the teacher forward pass and the student training in online distillation" is FLOP arithmetic (teacher forward pass ≈ 2N_T, student forward + backward pass ≈ 6N_S), and no argument in the text depends on it. Also, the main line uses only sequence-level distillation (the teacher generates on vLLM). A measurement of the teacher forward pass in the training loop does not correspond to the method of the main line.
- **Chapter 18**: a GRU with about 20,000 parameters, and a bandit. The arguments about DPO (β, learning rate, the probability of chosen goes down, implicit reward) do not depend on the device. Only one sentence in the text relates to the GPU: "put chosen / rejected into one batch for the forward pass, and save half of the kernel launches". It is an implementation detail and does not need a separate script.
- **Chapter 19**: GRPO on a bandit and on single-digit addition. This also does not depend on the device. The real thing to measure on a GPU is "how much time the sampling (generation) and the training of the main-line GRPO each take". It relates directly to a statement in the text: "zero sampling has no continuous batching; if the throughput is not sufficient, change to verl". But this measurement depends on the GPU path of GRPO in zero, which another agent is verifying. Recommendation: consider it after that result is available.

## 3. Items for you to decide

1. In 16/04, "the loss even decreased" has the opposite direction on this machine (Detail 2). In 20/02, some specific sentences are no longer true (Detail 7). In 18/05, "more than half do not even have the correct format" is not true (Detail 5). In 19/03, "all guess 7" contradicts the script output (Detail 6). Do you want to change the wording of the text?
2. Do you want to commit `tiny_kv4_s600.pt` of Chapter 10 (and the `ch16_*.pt` files of Chapter 16) as assets, so that readers can reproduce the numbers in the text? Or add a note in the text that there are small differences between machines?
3. The GPU results of Chapter 16 show that document isolation with varlen / FlexAttention is even faster than normal causal attention in an 8192 window. Do you want to move "isolation of packed conversations" earlier, so that it is no longer only "an option for the second step"?

## 4. List of changes

- New: `chapters/16-sft/code/06_gpu_packing.py`, `chapters/20-release/code/05_gpu_quant_matvec.py`, and this file.
- Changed: `chapters/16-sft/README.md` and `chapters/20-release/README.md` (each got a new section before "From minimal code to production code"; the existing text and numbers did not change).
- Not changed: other chapters, the existing scripts in `code/`, the committed json files in `code/out/` (restored), `video/`, `zero/`, `configs/`.
- Files from the runs that are in gitignore: `chapters/10-inference/code/out/tiny_kv4_s600.pt`, `chapters/16-sft/code/out/ch16_{masked,unmasked,small40}.pt`.
