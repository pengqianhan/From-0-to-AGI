# GPU0 verification: Chapters 21–23

**English** · [中文](chapters-21-23.zh.md)

Date: 2026-10-01. Branch `gpu0-verification`. Scope: `chapters/21-kv-cache-ledger`, `chapters/22-local-sparse-attention`, `chapters/23-linear-attention-hybrid`.

## Environment

- CPU: AMD Ryzen Threadripper PRO 3995WX (Zen 2, AVX2, no AVX-512). This is a shared server. Other agents and video renders ran at the same time.
- PyTorch 2.11.0+cu128, CUDA 12.8, Triton 3.6.0. `flash_attn`, `fla` (flash-linear-attention), and `causal_conv1d` are not installed.
- GPU: only GPU0 (NVIDIA GeForce RTX 3090, `CUDA_VISIBLE_DEVICES=0`, in a queue through `gpu0.lock`). Before the first use, I made sure that `torch.cuda.device_count() == 1`. `nvidia-smi` shows a power limit of **240 W** for GPU0 (default 350 W). On the same card, one sequential read (a sum) of 1 GiB measured 873 GB/s (spec 936 GB/s). This reference already includes the effect of the power limit.
- CPU reproduction: `CUDA_VISIBLE_DEVICES= UV_NO_SYNC=1 uv run python ...`, at most 4 processes in parallel.
- 4 scripts train models and cache the weights (21/04, 22/02, 23/04, 23/05; 22/03–05 read the cache of 22/02). By default, they write to `code/out/`. To write nothing into the repository, I ran them through a small wrapper, `scratchpad/ch21-23/run_redirected.py`. The wrapper reads the source of the script and replaces only the line `OUT = ...` with a scratch folder. (For 22/03–05, it sets `m.OUT` after it loads 02.) Then it runs the source with `exec`, as `__main__` and with the original file name. The script files did not change. The weights are in `scratchpad/ch21-23/out/`.
- Raw output: CPU runs in `scratchpad/ch21-23/logs/`, reruns in `logs2/`, final GPU output in `final/` (`scratchpad` = `/tmp/claude-1006/-home-phan635-Opensource-From-0-to-AGI/1d40fca9-4469-49a7-88b9-4e96bf8f0051/scratchpad`).

## 1. CPU reproduction table

"CPU time" is the wall-clock time of the full `uv run` process. I measured it with 4 processes in parallel on a busy machine, so use it only for the order of magnitude. All 14 scripts returned rc=0, with **no errors**.

| Ch. | Script | CPU time | Category | Notes |
|---|---|---:|---|---|
| 21 | `01_kv_ledger.py` | 0.1 s | Match | The two ledger tables match item by item. (The script prints MiB for values smaller than 1 GiB, and the README changes them to GiB: 448 MiB = 0.44 GiB, 1008 MiB = 0.98 GiB, and so on.) The two parity checks against `zero` both give True |
| 21 | `02_prefill_decode.py` | 2.4 s | Match | The prefill table, the decode table, and the table of concurrent conversations are all correct. (For the row 131,072, the script prints 2173.9 ms, and the README writes 2.2 s.) |
| 21 | `03_mla.py` | 2.6 s | Mismatch (rounding level) | "Absorbed path = explicit path": 4.4e-16, a match. "Token by token with a cache = one forward pass over the full sequence": README 2.8e-16, measured 3.9e-16 (float64). All other values match |
| 21 | `04_attention_variants.py` | 1370 s (trains 15 models) | Mismatch (FP32 training drift, the conclusions do not change) | See Detail 1 |
| 22 | `01_masks_and_ledger.py` | 2.7 s | Match | The mask plots, 55/34 pairs, 128×/4,096×, the two receptive-field rows, and the four rows of the 128K ledger are all correct |
| 22 | `02_swa_model.py` | 735 s (trains 6 models) | Only the timing differs | It prints only the parameter counts (449K / 167K), and the README does not quote them. For the run time, see Detail 4 |
| 22 | `03_compare.py` | 84 s | Match | Loss 1.797 / 1.745 / 1.748, the KV position counts, the needle-in-a-haystack accuracy in the three ranges, and the breakdown into 12 groups are all correct |
| 22 | `04_bounded_cache.py` | 13 s | Match | The three True rows and the cache byte counts are all correct |
| 22 | `05_topk_sparse.py` | 236 s | Match | The 7-row table matches character for character |
| 23 | `01_linear_attention.py` | 4.9 s | Mismatch (rounding level) + timing | Max relative error of parallel vs recurrent: README 4.7e-07, measured 4.2e-07. The KV/state table matches. For the decode timing, see Detail 3 |
| 23 | `02_chunked.py` | 6.1 s | Mismatch (rounding level) + timing | Recurrent − parallel: README 1.8e-06, measured 2.0e-06. Recurrent − chunked: 6.0e-07, a match. The decay table matches. For the speed table, see Detail 3 |
| 23 | `03_delta_rule.py` | 2.4 s | Mismatch (rounding level) | Parts 1–3 are all correct. Part 4, output difference / state difference: README 4.8e-07 / 3.6e-07, measured 4.2e-07 / 2.4e-07 |
| 23 | `04_hybrid_lm.py` | 1320 s (trains 4 models) | **Mismatch** (changes one sentence of the text) | See Detail 2 |
| 23 | `05_associative_recall.py` | 400 s (trains 4 models) | Mismatch (±1 percentage point, the conclusions do not change) | See Detail 2 |

Totals (14 scripts): 6 match. 1 differs only in timing (22/02). 4 differ only at the rounding level (21/03, 23/01, 23/02, 23/03; 23/01 and 23/02 also have timing differences). 3 differ because of numerical drift in FP32 training (21/04, 23/04, 23/05; only 23/04 makes one sentence of the text incorrect). 0 errors.

**These differences depend on the machine. They are not random variation on the same machine.** I ran 21/03, 23/01, 23/02, and 23/03 again on this machine, and the numbers were bit-identical. I trained GGGA of 23/04 again in a new folder, and the validation loss was 1.652 again. For Chapter 10, another agent ran `01_tiny_model.py` again on this machine: MHA seed 0 gave 1.831 and seed 1 gave 1.861. These values are exactly the same as MHA seeds 0 and 1 of 21/04 on this machine. Thus the README numbers were probably made on a different CPU (possibly a machine with AVX-512). Its FP32 kernels round differently. After a few hundred training steps, the difference grows to the third decimal place.

## 2. Details of the mismatches and the timing differences

### Detail 1: Chapter 21 `04_attention_variants.py` (FP32 training drift, the conclusions do not change)

| Variant | README mean [three seeds] | Measured mean [three seeds] |
|---|---|---|
| MHA | 1.858 [1.838 / 1.869 / 1.867] | 1.852 [1.831 / 1.861 / 1.865] |
| GQA | 1.856 [1.867 / 1.849 / 1.852] | 1.853 [1.865 / 1.843 / 1.852] |
| MQA | 1.860 [1.860 / 1.858 / 1.862] | 1.859 [1.860 / 1.853 / 1.864] |
| MLA-48 | 1.887 [1.878 / 1.888 / 1.895] | 1.886 [1.878 / 1.887 / 1.894] |
| MLA-16 | 1.886 [1.889 / 1.854 / 1.916] | 1.886 [1.889 / 1.854 / 1.916] (exactly the same) |

All other columns match: the numbers per layer per position, the bytes per token, the measured cache bytes (2,150,400 / 1,075,200 / 537,600 / 537,600 / 268,800), the attention parameter counts, and "Cached = naive". These numbers in the text change (Section 5):

- "The loss of MHA with seed 0, 1.838, is exactly the same as in Chapter 10": on this machine, the value is 1.831, and Chapter 10 on this machine also gives 1.831. "Exactly the same" is still true; only the number changed.
- "MHA alone changes by 0.031 with a different seed (1.838 vs 1.869)" → 0.034 on this machine (1.831 vs 1.865).
- "Their means are between 1.856 and 1.860" → 1.852–1.859 on this machine.
- "All three seeds of MLA-48 (1.878–1.895) are higher than all nine seeds of the first three structures (the highest is 1.869)" → 1.878–1.894 on this machine, and the highest is 1.865. The conclusion is still true. "Its mean is about 0.03 higher than that of MQA" → 0.027 on this machine; still true.

### Detail 2: Chapter 23 `04_hybrid_lm.py` and `05_associative_recall.py`

Validation loss of `04_hybrid_lm.py`:

| Architecture | README | Measured |
|---|---:|---:|
| AAAA | 1.685 | 1.683 |
| LLLL | 1.754 | 1.760 |
| GGGG | 1.661 | **1.648** |
| GGGA | **1.649** | 1.652 |

The parameter counts, the inference cache table, and the Qwen3.5-0.8B cache table all match. **The text that changes** (Section 8.2): "the 3:1 hybrid is the lowest (1.649)" is not true on this machine. On this machine, GGGG is the lowest (1.648), and GGGA is second (1.652). Thus the bold value in the GGGA cell of the table is also incorrect. "Gated DeltaNet is a little better than pure attention (1.661 vs 1.685)" → 1.648 vs 1.683 on this machine; the direction does not change. "Only 0.04 among the top three" (0.035 on this machine) and "at most 0.1 nats" (0.112 on this machine) are still true. The README already says "We used a single random seed … Thus we cannot make a reliable ranking from them". Thus only one specific statement must change, and the conclusion does not change. (Fixed later: the table of Section 8.2 now has a column for the rerun on another server. The text says that the order of GGGA and GGGG changed in the rerun.)

`05_associative_recall.py`: the rows AA and GA match cell by cell. For LL, N = 16 / 20 changed from 24% / 19% to 25% / 18%. For GG, N = 8 / 16 / 20 changed from 62% / 39% / 32% to 63% / 38% / 31%. The three conclusions do not change (at N = 24, 29% → 79% matches). I also found this: the text says that Gated DeltaNet is higher than naive linear attention "by 10–20 percentage points at each N". With the README's own numbers, the difference at N = 4 and 8 is already 21 and 22 percentage points (21 and 23 on this machine). I suggest "12–23 percentage points" or "from ten to more than twenty percentage points". (Fixed later: the README now says "12 to 22 percentage points higher at each N".)

### Detail 3: Chapter 23, the timing of 01 and 02 (only the timing differs)

- `01`, part 3 (milliseconds): README 0.033/0.030, 0.250/0.030, 12.5/0.030, 44.2/0.030; this machine 0.056/0.042, 0.260/0.039, 5.356/0.041, 20.920/0.041. The trend is the same (softmax grows with T, and linear stays constant).
- `02`, speed table (milliseconds): README 256: 33.4/4.1/3.9, 1024: 76.2/12.7/79.6, 4096: 344.1/47.2/2015.0; this machine 256: 10.2/0.8/0.9, 1024: 41.8/3.9/23.3, 4096: 167.0/15.5/678.8. The text says that the chunkwise form "is about 7 times faster than the token-by-token form": 10.8 times on this machine. "More than 40 times faster than the fully parallel form": 43.8 times on this machine, still true. "For short sequences, the fully parallel form is the fastest": on this machine, at T = 256, chunkwise (0.8 ms) is a little faster than parallel (0.9 ms), so the two are equal.

### Detail 4: Chapter 22 `02_swa_model.py`: the statements about the run time do not agree (only timing)

The docstring of the script says "trains all 6 models, about 5 minutes". The README says "the first run took about 40 minutes in total". On this machine, with 4 processes in parallel, the script took 735 s (about 12 minutes). The two statements do not agree. I suggest that we make them the same. (Fixed later: the docstring now says "about 12 minutes on an idle CPU, up to 40 minutes on a busy one".)

## 3. New GPU scripts and README sections

All three scripts pass `UV_NO_SYNC=1 uv run ruff check`. Without CUDA, they print the agreed message and exit 0 (verified with `CUDA_VISIBLE_DEVICES=`). They use a fixed random seed. They warm up first, measure the time with CUDA events, and take the median. At the start, they print the GPU name and the torch version. All numbers in the READMEs come character for character from one run of the final version on GPU0 (`scratchpad/ch21-23/final/*.log`).

| Ch. | New script | GPU0 wall-clock time | New in the README |
|---|---|---:|---|
| 21 | `chapters/21-kv-cache-ledger/code/05_gpu_decode_ledger.py` | 9.0 s | `## GPU measurements (one RTX 3090)`, after the summary of Section 6 and before `## From minimal code to production code` |
| 22 | `chapters/22-local-sparse-attention/code/06_gpu_flex_window.py` | 39.6 s (includes 7 s for the cold compile of FlexAttention) | Same position |
| 23 | `chapters/23-linear-attention-hybrid/code/06_gpu_linear_vs_softmax.py` | 23.6 s | Same position |

### Chapter 21: one decode step ≈ (weights + KV cache) ÷ bandwidth

The script builds a BF16 decode step with random weights in the shape of the main-line model (28 layers, width 1280, 16/8 heads, head_dim 128, FFN 3584, vocabulary 65,536). It fills the KV cache with T positions first, and it replays the step with a CUDA Graph. For the theoretical lower bound, it calls `decode_step` of `02_prefill_decode.py` directly and only replaces `PEAK_FLOPS`/`HBM_BW` with the 3090 specs. Normal attention uses SDPA (FlashAttention). MLA uses a hand-written absorbed path (the weighted average uses split-KV).

Main results:

- With batch 1, from T = 1K to 128K, the step reads 13.89 GiB more KV cache and takes 23.83 ms longer. This is 626 GB/s, about 1.7 ms for each GiB.
- "4 sequences of 32K" and "1 sequence of 128K" both read 14 GiB of KV. One step takes 27.48 vs 27.23 ms.
- At 4K, batch 1→16 increases the throughput from 244 to 936 (about 3.8×; the 3090 ledger predicts 3.3×). At 32K, batch 1→4 increases it only from 108 to 146 (about 1.35×; the ledger gives 1.25×).
- At T = 32K, MHA/GQA/MQA/MLA really allocate 7.00/3.50/0.44/0.98 GiB of KV cache, exactly as in the ledger. One step takes 15.57/9.33/4.36/7.75 ms.
- The measured times are 1.5–2.1 times slower than the theoretical lower bound. With a short context, the effective bandwidth is only 440 GB/s. Each weight matrix of a layer has only 5–18 MB, and the matrix-vector products of batch 1 cannot use the full bandwidth. In a microbenchmark, one matrix-vector product with the size of one layer (1280×5120, 3584×1280, and so on; batch 1) reaches only 240–410 GB/s. But the 1280×65536 lm_head reaches 660–750 GB/s. With a long context, the bandwidth comes near 600 GB/s.
- Without CUDA Graph, one step at T = 1K takes 7.51 ms (with CUDA Graph, 3.40 ms).

Found during debugging: a hand-written `q @ Kᵀ → softmax → @ V` for GQA decode reaches only about 270 GB/s, because the reduction over the length T goes to very few SMs. With SDPA, it reaches 600 GB/s. MLA through SDPA (head_dim 576) falls back to a very slow path (11 ms per layer). Thus MLA uses the hand-written path + split-KV.

### Chapter 22: the sliding window must "really skip" the blocks outside the window

The setup: batch 1, 16 heads, head_dim 128, BF16. Three versions: dense causal SDPA (FlashAttention), boolean-mask SDPA (memory-efficient + a T×T mask), and FlexAttention (block-sparse with `create_block_mask`).

- At T = 16,384, FlexAttention grows from 0.90 ms at W = 128 to 9.37 ms at W = 4,096. Dense causal stays at about 23 ms, and the boolean mask stays at about 70 ms.
- At W = 1,024, for T = 4K/16K/64K, FlexAttention takes 0.94/3.61/12.06 ms, and dense takes 1.62/22.72/473.55 ms (39.3× faster at 64K).
- The max difference between the three versions is 3.9e-03 (the size of BF16 rounding).

**FlexAttention runs on a 3090 + PyTorch 2.11.** There is only one problem. By default, `create_block_mask` builds the full T×T table. At T = 65,536, it requests 32 GiB and runs out of memory (OOM). `torch.compile(create_block_mask)` solves the problem (`_compile=True` is marked as deprecated).

### Chapter 23: softmax vs the chunkwise / recurrent forms of linear attention

The shapes are those of the Gated DeltaNet layer of Qwen3.5-0.8B (16 heads, d_k = d_v = 128). The script calls the functions of 03 and 04 without changes in `with torch.device("cuda")` (FP32, chunk length 64). Softmax uses FlashAttention (BF16).

- On the GPU, chunkwise = recurrent (output difference 6.0e-07, state difference 4.8e-07).
- At T = 4,096, the token-by-token recurrent form takes 813.5 ms, and the chunkwise form takes 43.1 ms (about 19×).
- The chunkwise form grows linearly with T. FlashAttention grows 21× from 16K to 64K. At 64K, the naive linear chunkwise form (180.7 ms) is already faster than FlashAttention (460.1 ms). The Gated DeltaNet chunkwise form (757.5 ms) is not faster yet.
- Decode: the Gated DeltaNet state is always (1, 16, 128, 128), 1 MiB, and one step takes 0.24–0.31 ms. The softmax KV cache grows from 8 MiB to 2,048 MiB, and one step grows from 0.053 to 3.408 ms.
- For short sequences, the fixed overhead of the Python loop limits the pure PyTorch implementation of this chapter. Each chunk costs about 0.7 ms for Gated DeltaNet and about 0.18 ms for naive linear, independent of T. This is why production code uses the fused kernels of fla. (fla is not installed on this machine, so there is no comparison.)

## 4. Effect of the GPU results on the GPU statements in the READMEs (I did not change the original text; please decide on all of them together)

1. **Chapter 22, last row of "From minimal code to production code"**: "On a GPU, to really skip the blocks outside the window, you need `window_size=(W − 1, 0)` of FlashAttention, or a sliding-window `mask_mod` for PyTorch FlexAttention … **these GPU paths are not yet verified on a GPU**". This run confirms the FlexAttention part. FlexAttention runs on a 3090 + PyTorch 2.11, it really skips blocks, and its numbers agree with the masked version. (Note: this is the standalone script of the chapter, not `zero/arch/sliding_window.py`.) We could not verify the `window_size` of FlashAttention (`flash_attn` is not installed). We did not test vLLM / transformers. (Fixed later: the row now gives the FlexAttention measurement and says that the `window_size` of FlashAttention is not verified.)
2. **On a GPU, the current path of `zero/arch/sliding_window.py` is slower than full attention.** It uses SDPA + a boolean mask (lines 229–232). In this run, the same call takes about 70 ms at T = 16K. This is 3 times the dense causal FlashAttention (about 23 ms). It also needs a mask and a bias of size T² (256 MiB + 512 MiB at T = 16K). Its comment already says that only the `window_size` of FlashAttention or FlexAttention really skips the blocks outside the window on a GPU. The data of this run gives a measured basis for this sentence. Please decide whether to change the CUDA path of zero to FlexAttention. (I did not change zero/.)
3. **Chapter 23, Section 3**: "On a GPU, the difference is larger, because the chunkwise form changes the work into matrix multiplications, which GPUs do best". For the pair chunkwise vs token-by-token recurrent, the GPU gives about 19× (T = 4,096). On the CPU, the README says 7×, and this machine gives 10.8×. **Confirmed.** But one limit must be added: on a GPU, the Python launch overhead limits the absolute speed of the pure PyTorch chunk loop of this chapter. For short sequences, it is tens of times slower than FlashAttention. The other half, "more than 40 times faster than the fully parallel form", was not measured on a GPU.
4. **Chapter 21, "From minimal code to production code"**: "The MLA of this course … its performance is … not yet verified on a GPU". The script of this run uses its own MLA absorbed-path decode, not `zero/arch/mla.py`. Thus the MLA of zero is still not verified on a GPU. But the data supports "a real deployment must use special implementations": without fusion, the MLA cache is only 28% of the GQA cache, but one step saves only about 17%.
5. **Chapter 21, Section 1.2** (H100 theory): "at 4K, when the batch grows from 1 to 16, the throughput increases 3.3 times", and at 32K, batching almost stops working. The 3090 measurements show the same trend (4K: 3.8×; 32K, batch 1→4: 1.35×). This agrees with Section 1.2.
6. This run did not test the GPU path of `zero/arch/linear_attention.py` (the scripts use the chapter code).

## 5. Other

- `chapters/22-local-sparse-attention/video/subtitles.srt` shows as modified in `git status`. I did not change it (it is probably the video render).
- `chapters/22-local-sparse-attention/code/out/` and `chapters/23-linear-attention-hybrid/code/out/` (`*.pt`, ignored by .gitignore) were written after 10:29 by the video render (`video/build.sh`; when cache.json is missing, scenes.py trains small models). My processes did not write them. When I started, these two folders did not exist, and I redirected all my training to the scratch folder.
- One rule for the interpretation: the power limit of GPU0 is 240 W. For Chapter 21, I used the pure read bandwidth of 873 GB/s, measured on the same card, as the reference. Thus, in the part "far from the spec peak", I attribute to the small matrix-vector products only the gap beyond the 873 GB/s reference. The interpretations for Chapters 22 and 23 compare only relative times and do not quote ratios to the peak.
- No commit, push, or branch change. No package installed.

## Decisions for you

1. Chapter 23, Section 8.2: "the 3:1 hybrid is the lowest (1.649)" does not reproduce on this machine (GGGG, 1.648, is lower). Should we change the table and this sentence to the numbers of this machine? Or should we only add "on different machines, the ranking can change"? (Fixed later: see Detail 2.)
2. Should we replace the numbers of 21/04, 23/05, and some rounding-level numbers with the results of this machine? (All conclusions stay the same.) (Later, the READMEs of Chapters 21 and 23 kept the original numbers. They added a note about the drift between machines, with a link to this record.)
3. Should we make the run-time statements of Chapter 22, 02, the same: the docstring (about 5 minutes) and the README (about 40 minutes)? (Fixed later: see Detail 4.)
4. Should the CUDA path of `zero/arch/sliding_window.py` change to FlexAttention? (See item 2 of Section 4.)
