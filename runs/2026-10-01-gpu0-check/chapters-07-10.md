# GPU0 verification: Chapters 7–10

**English** · [中文](chapters-07-10.zh.md)

Date: 2026-10-01. Branch: `gpu0-verification`. Scope: `chapters/07-*` to `chapters/10-*`.

## Environment

- CPU: AMD Ryzen Threadripper PRO 3995WX (Zen 2, AVX2, no AVX-512). This is a shared server. During the runs, the load average was between 5 and 54 (other agents and video renders ran at the same time).
- PyTorch 2.11.0+cu128 (on the CPU, it uses MKL / oneDNN, `CPU capability usage: AVX2`), cuDNN 9.19.
- GPU: NVIDIA GeForce RTX 3090. I used only GPU0 (`CUDA_VISIBLE_DEVICES=0`), with a queue through `gpu0.lock`. Before each run, I made sure that `torch.cuda.device_count() == 1`. The power limit of GPU0 is 240 W (default: 350 W). Under sustained full load, the GPU lowers its clock, so the absolute values are low. I did not use GPU1. (Later, the coordinator said that I could use GPU1 for debugging. But the permission check blocked it, because only the user can give this type of permission. Thus all GPU runs were on GPU0.)
- CPU reproduction command: `CUDA_VISIBLE_DEVICES= UV_NO_SYNC=1 uv run python chapters/.../code/xx.py`. I split the chapters into 3 pipelines that ran in parallel (a maximum of 3 CPU processes at the same time). The raw output is in `/tmp/claude-1006/-home-phan635-Opensource-From-0-to-AGI/1d40fca9-4469-49a7-88b9-4e96bf8f0051/scratchpad/ch07-10/cpu/`. The GPU output is in `gpu/` in the same folder, and the probe experiments are in `explore/`.
- Side effect: during the CPU reproduction, the scripts of Chapters 9 and 10 wrote the trained weights into `chapters/09-modern-transformer/code/out/tiny_transformer.pt` and `chapters/10-inference/code/out/tiny_kv*.pt`. The scripts do this by design (`.gitignore` ignores `*.pt`). Later, when you render the videos of these two chapters on this machine, the renders load these files directly and do not train again.

## 1. CPU reproduction table

"CPU time" is the wall-clock time of the full `uv run` process. I measured it on a busy machine, so use it only for the order of magnitude. All 19 scripts returned rc=0. **There were no errors.**

| Chapter | Script | CPU time | Category | Notes |
|---|---|---:|---|---|
| 7 | `01_chars_and_bytes.py` | 0.2 s | match | Corpus table, UTF-8 table, and 198 unseen characters / 129 distinct: all correct |
| 7 | `02_bpe.py` | 1.8 s | timing only | 20 merges, the encoding examples, and the compression table: all correct. The README says "about 4 seconds in pure Python". This run took 1.6 seconds |
| 7 | `03_bigram.py` | 4.6 s | match | The 9-row bpb table, 58 `<unk>`, and the 4 sampled texts: identical, character for character |
| 7 | `04_vocab_and_zero.py` | 55.1 s | timing only | Parity table, vocabulary sweep table, and embedding table: all correct. The README says "zero 0.74 seconds, hand-written 4.55 seconds". This run took 1.08 / 1.62 seconds (the ratio changes from 6.1× to 1.5×) |
| 8 | `01_average_to_attention.py` | 0.2 s | match | The two output blocks are identical, character for character |
| 8 | `02_attention_from_scratch.py` | 2.2 s | mismatch (rounding level) | Max difference from SDPA: 9.7e-08 in the README, 8.9e-08 in this run (in two places: the output block of Section 8 and the summary in Section 10). All other values are correct |
| 8 | `03_why_sqrt_d.py` | 2.1 s | match | Both tables are correct |
| 8 | `04_train_attention.py` | 58.3 s | match | Loss / bpb table, head-role table, heat map, and the argmax at each position: all correct. "About 1 minute" is also correct |
| 8 | `05_zero_parity.py` | 2.1 s | mismatch (rounding level) | ① Max difference: 8.9e-08 in the README, 1.5e-07 in this run. ② All correct |
| 9 | `01_position.py` | 1.7 s | mismatch (rounding level) | Max difference for "No position information": 5.96×10⁻⁸ in the README, 0.00e+00 in this run. (The interpretation in the main text, "the outputs are the same", is still true.) RoPE table, 0.133, and 2.2184: all correct |
| 9 | `02_tiny_transformer.py` | 886.0 s | **mismatch** | The float32 training drifts. From step 400, bpb differs by 0.002–0.011, and the sample text after training is different. See detail 2. The wall-clock time of 14.8 minutes agrees with the 14.7 minutes in the README |
| 9 | `03_shapes.py` | 2.3 s | match | Shape table, main-line data flow, 8.0 GiB, 13.11M, and the tied-embedding table: all correct |
| 9 | `04_parity_with_zero.py` | 6.5 s | mismatch (rounding level) | Max difference after training: 1.43e-05 in the README, 1.72e-05 in this run (`assert_close` still passes). Random initialization: 5.96e-07 in the README, 4.77e-07 in this run (I copied the script to the scratch folder and ran it without weights). Greedy generation identical on the two sides: True |
| 9 | `05_qk_norm.py` | 2.2 s | match | The table is correct |
| 10 | `01_tiny_model.py` | 98.3 s | **mismatch** | Validation loss: 1.838 in the README, 1.831 in this run. The parameter count 861,440 and "about 1.5 minutes" are correct |
| 10 | `02_sampling.py` | 12.4 s | **mismatch** | The numbers and the generated text all changed in the temperature table, the top-p table, and the decoding-strategy table. See detail 3 |
| 10 | `03_kv_cache.py` | 42.9 s | **mismatch** + timing | The two parity-check items are still True, and the cache size of 872,448 bytes is correct. The generated text and the logits difference (4.3e-06→2.9e-06) changed. The speed and prefill/decode results are timing differences. See detail 3 |
| 10 | `04_kv_memory.py` | 2.3 s | match | The three tables and 117,440,512: all correct |
| 10 | `05_gqa.py` | 290.9 s | **mismatch** | The three losses drift a little. On this machine, one sentence in the main text is not true: "a different seed gives a larger difference than the differences between the three variants". See detail 4 |

Statistics (19 scripts): match 8. Timing only 2 (Chapter 7: 02, 04). Mismatch, but only at float32 rounding level (1e-7 to 1e-5) 4 (Chapter 8: 02/05, Chapter 9: 01/04). Mismatch that affects numbers in the main text 5 (Chapter 9: 02, Chapter 10: 01/02/03/05). Error 0.

## 2. Details of the mismatches

### Detail 1: why the values drift (common cause)

The Chapter 10 code did not change after it was added (`git diff 85e12c9 HEAD -- chapters/10-inference/code/` is empty). The seed is fixed, and the code uses one thread. Thus the differences come from the run environment. This machine uses PyTorch 2.11.0+cu128 (MKL + oneDNN) on Zen 2 / AVX2. Its order of floating-point accumulation is different from the "build machine" that produced the README.

In one operation, this is only a difference of about 1e-7 (the rounding-level mismatches in Chapters 8 and 9). But a few hundred training steps make it larger, and the weights become different. Evidence: in Chapter 9, the bpb at training steps 0 and 200 is exactly the same as in the README (8.052, 2.781). The values separate only from step 400. The agent for Chapters 1–6 saw the same effect in Chapter 6.

### Detail 2: Chapter 9 `02_tiny_transformer.py`

| Step | 0 | 200 | 400 | 600 | 800 | 1000 | 1200 |
|---|---:|---:|---:|---:|---:|---:|---:|
| README | 8.052 | 2.781 | 2.518 | 2.391 | 2.289 | 2.243 | 2.221 |
| This run | 8.052 | 2.781 | 2.529 | 2.385 | 2.284 | 2.241 | 2.217 |

After training, the sample still starts with `ROMEO:\nAnd yield for what be`. After that, it is different (this run continues with `deceived my heart.\n\nQUEEN ELIZABETH:` …). The 400-byte sample in the README cannot be reproduced on this machine. The qualitative conclusions do not change: the model learned the format of a play and old English words, and it sometimes makes new words.

### Detail 3: Chapter 10 `01`–`03` (the weights of the small model changed, so all downstream numbers changed)

- Validation loss: 1.838 → 1.831.
- Temperature table in Section 2.2 (the order of the top 8 also changed):

| Temperature | README | This run |
|---|---|---|
| T=0.5 | t .427, a .093, n .073, s .066, h .062, m .060, b .047, w .037; entropy 2.15 | t .370, s .099, n .096, a .068, m .068, h .063, b .060, w .037; entropy 2.26 |
| T=1.0 | t .173, a .081, n .071, s .068, h .066, m .065, b .057, w .051; entropy 3.00 | t .158, s .082, n .081, a .068, m .068, h .065, b .064, w .050; entropy 2.99 |
| T=1.5 | t .106, …, w .047; entropy 3.39 | t .101, s .065, n .064, a .057, m .057, h .056, b .055, w .047; entropy 3.36 |

  The main text says "the probability of t increases from 0.17 to 0.43 … decreases to 0.11". This run gives 0.16 → 0.37 … 0.10.
- Top-p table in Section 2.3: for the certain context `e`, p = 0.546 → 0.572, and top-p=0.9 keeps **5 → 4** candidates. For the uncertain context `A`, p = 0.123 → 0.130, and top-p still keeps 15. (This does not affect the argument. The effect is even clearer.)
- Decoding-strategy table in Sections 2.1 / 2.4 (`distinct 4-gram ratio` / start of the text):
  - greedy 0.27 → 0.29 (the start becomes `the soul, and the prove the provess`, and it still goes around in a loop);
  - T=0.5 0.84 → 0.81; T=1.0 0.95 → 0.98; top-k=5 0.91 → 0.89;
  - top-p=0.9 0.94 → 0.93 (the same start, `thou more the prevenced can that`);
  - T=1.5 0.98 → 1.00 (the garbage word `gMycouuesy` in the main text is `gMycouresy` in this run).
- Parity-check block in Section 4.1: both `same: True` items reproduce. The start texts changed (greedy `'the soul, and the prove the provess\nThat'`, sampling `'thou more the prevenced can that\nO, in I'`). The max logits difference changed from 4.3e-06 to 2.9e-06. The cache size of 872,448 bytes is correct.
- Speed in Section 4.2 (timing differences; all counts of processed positions are correct): 64 / 128 / 256 / 512 → 0.29/0.15 (1.9×), 0.75/0.31 (2.4×), 2.25/0.58 (3.9×), 11.89/1.20 (9.9×).
- Section 4.3: prefill 16.2 ms (15,792 positions/s), decode 487.8 ms (525 positions/s), 30× (README: 16.9 ms / 691.5 ms / 41×; a timing difference).

### Detail 4: Chapter 10 `05_gqa.py` (the conclusion of one sentence is not true on this machine)

| Variant | README validation loss | This run |
|---|---:|---:|
| MHA | 1.838 | 1.831 |
| GQA | 1.867 | 1.865 |
| MQA | 1.860 | 1.860 |
| MHA, seed 1 | 1.869 (difference 0.031) | 1.861 (difference 0.030) |

The parameter counts and the KV cache bytes are all correct. The time to generate 512 characters is 1.28 / 1.18 / 1.36 seconds (timing).

The problem: Section 6.1 says "a different random seed alone changes the loss of the same MHA by 0.031, more than the differences between the three variants". With the README numbers, this is true: the largest difference between the three variants is 0.029 < 0.031. On this machine, the largest difference between the three variants is 0.034 (MHA 1.831 vs GQA 1.865). This is more than the seed difference of 0.030, so **the sentence is not true**.

The conclusion "at this scale, we cannot tell which variant is better" is still mostly correct (the differences have the same magnitude as the seed noise). But the wording must change, for example to "the same order of magnitude as the differences between the three variants". **You need to decide**: change the wording, or update all tables of Chapter 10 (and Chapter 9) with the numbers from this machine.

## 3. GPU measurements: new scripts and README sections

Both scripts pass `UV_NO_SYNC=1 uv run ruff check`. Without CUDA, they print the agreed message and exit with 0. They use a fixed seed. They do a warmup first, and then they take the median with CUDA events. At the start, they print the GPU name and the torch version. Neither script compares with the CPU.

The README sections are before `## From minimal code to production code`, after the existing `---` (the same position as in Chapters 2 and 16). The numbers in the tables come exactly from the logs listed below.

### Chapter 8: `chapters/08-attention/code/06_gpu_sdpa_backends.py` (about 13 seconds)

- Content: the attention shape of the main-line model (batch 1, 16 heads, head_dim 128, BF16, causal), T = 512 … 32K. The script compares two quantities: the time and the peak additional GPU memory. It compares the hand-written `attention` of Chapter 8 with three SDPA backends: math / efficient / flash. (It loads `02` unchanged with importlib. `torch.device("cuda")` makes sure that the mask inside the function is also built on the GPU.) Then it checks which backends work with `enable_gqa=True`.
- README: a new section "GPU measurements (one RTX 3090)" in `chapters/08-attention/README.md`. It has two tables (time, GPU memory), one paragraph about GQA, and 5 sentences of interpretation. The numbers come from `scratchpad/ch07-10/gpu/ch08_06_final.log` (the final version). Two earlier trial runs (`try1/try2`) showed the same trends.
- Key results:
  - The hand-written version uses about 2 times the memory of the score matrix, and this grows with T² (16,704 MiB at 16K; OOM at 32K).
  - flash/efficient use additional memory only for the output (64 MiB at 16K), and this grows linearly with T.
  - The time of flash still grows with T². At 16K, flash is 7.3 times faster than the hand-written version.
  - The math backend is even slower than the hand-written version, and it uses more GPU memory. (The profiler shows that it first converts q, k, v to float32 with `aten::to`.)
- The script sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (with `setdefault` before `import torch`). Without this setting, the hand-written version gets an OOM at 16K because of allocator fragmentation. (Probe: 7.76 GiB of the 16.2 GiB reserved memory was free, but a new 8 GiB block did not fit.) The real requirement is only 16.3 GiB.
- Small defect: the docstring says "about half a minute", but the measured time is about 13 seconds. I tried to change it to "about 15 seconds", but the permission check blocked the edit. I did not try again. Please fix it when convenient, or keep it.

### Chapter 10: `chapters/10-inference/code/06_gpu_prefill_decode.py` (about 53 seconds; peak GPU memory 7.0 GiB allocated / 9.6 GiB reserved)

- Content: the script loads `TinyLM` from `01_tiny_model.py` and the generation functions from `03_kv_cache.py` unchanged with importlib. (They run inside `torch.device("cuda")`, and the original files do not change.) The script changes the size to the shape of one Llama 3 8B layer (d 4096, 32/8 heads, FFN 14336) and stacks 4 layers. The result has 0.87B parameters, BF16, and a random initialization. The script has three parts:
  - ① time per token, naive vs KV cache;
  - ② one forward pass with T = 1 … 2048 tokens (two timings, eager and CUDA graph; it calculates the TFLOPS and the bandwidth of the weight reads);
  - ③ decode with batch 1 … 64, at context 64 and 512.
- README: a new section "GPU measurements (one RTX 3090)" in `chapters/10-inference/README.md`, with three tables and 5 sentences of interpretation. The numbers come from `scratchpad/ch07-10/gpu/ch10_06_try2.log` (the final code; after this run, only the note about GPU memory in the docstring changed). Another run of the same version (`explore/ch10_mem_gpu0.log`, which measured the peak GPU memory) gave numbers within 12%.
- Why this shape: first, I tried the shape of the main-line model (28 layers, d 1280). In eager mode, each decode step took 38 ms, but the GPU did real work for only 4.4 ms (CUDA graph replay). The overhead of Python, which launches the kernels one at a time, hid everything else. Thus the "bandwidth bottleneck" was not visible. So I changed to a wide, shallow model with 4 layers, and I also give the CUDA graph times. A probe showed that the `torch.device("cuda")` context itself increases the launch overhead of each operator from 9.3 µs to 12.1 µs.
- Key results:
  - From T = 1 to 32, the time almost does not change (2.65 → 3.19 ms; weight reads at 550–660 GB/s; 1%–25% of the peak compute).
  - For T ≥ 128, the time grows linearly with T, at about 60% of the peak compute. The bend agrees with "peak compute ÷ bandwidth ≈ 76".
  - At context 64, the throughput increases 29 times from batch 1 to batch 64.
  - At context 512 and batch 64, the KV cache (512 MiB, plus the copies from `torch.cat` and `repeat_interleave`) pushes the throughput down to 3,737.
  - On the GPU, the KV cache is only 1.1–2.7 times faster.
  - Eager mode takes almost 2 times as long as the CUDA graph.

### Chapters 7 and 9: no new scripts

Chapter 7 is about tokenization and counting. A GPU does not help the arguments of the main text. For Chapter 9, the only idea was "move the 11-minute training to the GPU". This does not make any specific argument clearer. The Chapter 8 script and the RUNBOOK verification below already cover the one Chapter 9 sentence about the GPU.

## 4. `enable_gqa` and the flash backend (the "to be verified" in row 1 of the RUNBOOK stage table)

**Conclusion: yes.** On an RTX 3090 with PyTorch 2.11.0+cu128 and BF16 (FP16 is the same), `F.scaled_dot_product_attention(..., enable_gqa=True)` can use the FlashAttention backend. When you do not specify a backend, SDPA also selects FlashAttention by default.

| Backend | BF16/FP16, MHA | BF16/FP16, GQA (16 q / 8 kv, `enable_gqa=True`) | FP32, MHA | FP32, GQA |
|---|---|---|---|---|
| FLASH_ATTENTION | Works | **Works** (difference from math with copied K/V: 7.8e-03, BF16 rounding level) | Does not work (requires half/bf16) | Does not work |
| EFFICIENT_ATTENTION | Works | **Does not work**: "For dense input, both fused kernels require query, key and value to have the same num_heads" | Works | Does not work |
| CUDNN_ATTENTION | Works | Works | Does not work | Does not work |
| MATH | Works | Works (identical to copied K/V) | Works | Works |
| Default selection (`torch._fused_sdp_choice`) | FLASH | FLASH | EFFICIENT | **MATH** |

- I verified the RUNBOOK command exactly as written (`zero.model.Transformer` with `configs/main/pretrain.toml`, `.cuda().bfloat16()`, input (1, 4096), `sdpa_kernel(SDPBackend.FLASH_ATTENTION)`). The forward pass gives no error, and the output is (1, 4096, 65536) BF16. Forward + backward also give no error, with a peak GPU memory of 13.89 GiB. With `EFFICIENT_ATTENTION` instead, the same command gives "No available kernel". Probe scripts: `explore/runbook_flash.py`, `explore/runbook_bwd.py`, `explore/gqa_probe.py`.
- Effect on the RUNBOOK: the original plan, "if not, fall back to the efficient backend", does not work. efficient does not support `enable_gqa`. A fallback must first copy K/V with `repeat_interleave` and then use efficient, or it must use the cuDNN backend.
- A trap to remember: **with FP32 + GQA, the default falls back to the math backend**. This backend stores the full T × T score matrix in GPU memory. If someone runs zero in float32 on a GPU (for example, for tests or evaluation), long sequences will use much GPU memory.
- Speed: at T = 4096, flash + `enable_gqa` and "copy K/V first, then use flash" take about the same time. Three runs gave 1.52/1.47, 1.19/1.35, and 1.17/1.46 ms. The order changes, so we cannot say which one is faster. GPU memory: 16 MiB vs 48 MiB.

## 5. How the GPU results relate to the current statements in the READMEs (I did not change the original text; you decide how to handle them)

1. `chapters/08-attention/README.md`, the `F.scaled_dot_product_attention` row of the table in "From minimal code to production code": "on CUDA, it selects the FlashAttention-2 or memory-efficient kernel automatically … **this GPU path is not yet verified on a GPU**". This is now verified on an RTX 3090. With BF16, both MHA and GQA select flash by default, and `enable_gqa` does not copy K/V (additional GPU memory 16 vs 48 MiB). But memory-efficient does not support `enable_gqa`, so "or memory-efficient kernel" is not true for GQA. With FP32, GQA falls back to math.
2. `chapters/09-modern-transformer/README.md`, the `Attention` row of the table in "From minimal code to production code": "on a GPU, it uses the FlashAttention kernel automatically, Chapter 14, **not yet verified on a GPU**". Same as above: this is now confirmed, including forward + backward with the main-line configuration of zero.
3. `chapters/10-inference/README.md`, Section 4.3: "each decode step … the hardware spends most of its time on moving data, and the compute units are not fully used". Section 6.1: "the speed benefit of GQA … becomes visible only on a GPU where memory bandwidth is the bottleneck, with long contexts and large batches". The GPU results support both sentences. Decode reads the weights at 64–70% of the specified bandwidth, with 1% of the compute. At context 512 and batch 64, the KV cache uses the largest part of the bandwidth. I did not measure the speed difference between GQA and MHA directly.
4. In row 1 of the stage table in `runs/RUNBOOK.md`, "to be verified" can change to "verified" (see Section 4).

## 6. Decisions for you

1. The training numbers of Chapters 9 and 10 cannot be reproduced on this machine (details 2–4). Update the tables with the numbers from this machine, or add a sentence such as "the numbers can differ a little with the CPU / PyTorch version"? At a minimum, the wording of the sentence "more than the differences between the three variants" in Section 6.1 of Chapter 10 must change.
2. Update the rounding-level differences (1e-7 to 1e-5) of Chapter 8 (02/05) and Chapter 9 (01/04) with the numbers from this machine?
3. The three "not yet verified on a GPU" statements in Section 5 above, and the "to be verified" in the RUNBOOK.
4. The "about half a minute" in the docstring of the Chapter 8 GPU script (the measured time is about 13 seconds).
