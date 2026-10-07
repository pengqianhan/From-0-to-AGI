# Stage 6 (RTX 3090 version): verify the GPU paths of zero on 1 and 2 RTX 3090 GPUs

**English** · [中文](README.zh.md)

Date: 2026-10-01. Branch: `gpu0-verification`. This run matches Section 2, "Stage 6: GPU verification run", of [`runs/RUNBOOK.md`](../RUNBOOK.md).
The RUNBOOK writes stage 6 for 8×H100. Sections 0–12 do the "single-GPU version" of each item on **one** RTX 3090.
Later, we reserved two more GPUs. Section 14 then did the cross-GPU items on **2×RTX 3090 (PCIe, no NVLink)**. Section 13 lists the items that still need more GPUs or H100 GPUs. We do not claim that these items are done.

> **Note:** This file covers only `zero/`. In the same check, we also reproduced the minimal code of each chapter on the CPU of this machine, and we added new "GPU measurements". These results are in the same folder:
> [chapters-01-06.md](chapters-01-06.md), [chapters-07-10.md](chapters-07-10.md), [chapters-11-15.md](chapters-11-15.md),
> [chapters-16-20.md](chapters-16-20.md), [chapters-21-23.md](chapters-21-23.md), [chapters-24-26.md](chapters-24-26.md).

## Hardware and software

| Item | Value |
|---|---|
| GPU | **Physical GPU0**: NVIDIA GeForce RTX 3090, 24 GB (23.57 GiB available), sm_86, PCIe; **power cap 240 W** (factory default 350 W) |
| Multiple GPUs (Section 14) | Physical GPU0 + **GPU2** (the same model, also with a power cap of 240 W). The 32K memory test also used GPU1 (power cap 200 W). The three GPUs have no NVLink between them. `nvidia-smi topo -m` shows NODE (the traffic goes through PCIe / the host bridge) |
| Driver / CUDA | 570.195.03 / CUDA 12.8 |
| PyTorch | 2.11.0+cu128 (cuDNN 9.19.0), Python 3.12.12, transformers 5.17.0 |
| CPU | 128 cores (for single-thread tasks such as the smoke test) |
| Basis for peak compute | RTX 3090 dense BF16 Tensor Core (FP32 accumulation): **71 TFLOPS** (GA102 white paper, at the boost clock of 1695 MHz). The peak table of zero does not include the 3090. Thus all configurations in this folder set `train.gpu_peak_tflops = 71`, and the log reports the MFU for the 3090 directly |

**Each GPU number in Sections 0–12 comes from physical GPU0** (`CUDA_VISIBLE_DEVICES=0`; torch sees only 1 GPU; before each command, we confirmed `torch.cuda.device_count() == 1`).
Each of these commands also used the file lock `flock <scratchpad>/gpu0.lock`. The lock puts the command in a queue with other tasks and gives it the full GPU alone. The multi-GPU runs in Section 14 use `CUDA_VISIBLE_DEVICES=0,2` (the 32K test uses `0,1,2`) and hold the lock of each GPU at the same time.
The project lead reserved GPU1 and GPU2 in addition during the verification.

## Read first: three conventions

1. **Data**: we did not download pretraining data. Training uses `assets/tiny_corpus` from the repository (Shakespeare / Tang poems / code, about 935,000 training tokens, a tokenizer with a vocabulary of 2048). `zero.data.prepare` makes these data on the spot in `out/gpu0-check/tiny/`. The vocabulary of the model is still 65,536 (the same as the ladder and the main line), so the memory and compute cost of the logits is real. The loss values show only that the loss "goes down". They do not show any model quality.
   Copies of the configurations are in [`configs/`](configs/). The `*_tiny.toml` files inherit the ladder / main-line configurations of the repository and change only the data. The `*.resolved.json` files are the full output of `--print-config`. The commands in the sections below show all other overrides.
2. **MFU**: zero counts the FLOPs per token as `6·N_matmul + 12·L·q_dim·T` (`zero/model.py`). The attention term is **not halved for the causal mask**. The count also does not include the recomputation of activation checkpointing. In the main-line configuration with T=4096, the attention term is 40% of the total, but the causal kernel of FlashAttention computes only half of it. Thus this MFU is too high for long sequences. Also, the 3090 has a power cap of 240 W, so its real clock can be lower than the 1695 MHz of the white paper.
   **These MFU values show only "how fully the 3090 is used". They do not represent the H100 with NVLink, and they cannot replace the 0.4 in the RUNBOOK directly.**
3. **The default GPU kernels are not deterministic from run to run** (the atomic adds for dQ in the FlashAttention backward pass, the index_add in the embedding backward pass, and others).
   When we run the same command twice, the loss differs by about 1e-5 from step 2. The large learning rate of the ladder configurations (2e-3–3e-3) amplifies this difference to 0.4–0.7 in one step during the first 20–30 steps. Thus, for each item that "compares the loss of two runs", we also make one run with
   [`det_pretrain.py`](det_pretrain.py) (`torch.use_deterministic_algorithms(True)`; no change to the code of zero). With this setting, two runs are bit-identical, and the throughput is only about 7% lower.

Large files (checkpoints, full logs, the JSON results of each script) are in `out/gpu0-check/` (.gitignore excludes this folder).

---

## 0. Baseline

**Unit tests**

```bash
CUDA_VISIBLE_DEVICES= UV_NO_SYNC=1 uv run pytest -q                                   # CPU
flock <gpu0.lock> env CUDA_VISIBLE_DEVICES=0 UV_NO_SYNC=1 PYTHONPATH=runs/2026-10-01-gpu0-check \
    uv run pytest -q -p cuda_test_probe                                               # GPU visible
```

| Time | CPU (`CUDA_VISIBLE_DEVICES=`) | GPU0 visible |
|---|---|---|
| At the start | 319 passed, 5 skipped (llama.cpp not compiled), about 30 s | 319 passed, 5 skipped, the same result |
| At the end (with the 4 new tests of this run) | **321 passed, 7 skipped** (3 new tests need CUDA, so they skip; one gguf test no longer skips, because the smoke test cloned the llama.cpp repository) | **324 passed, 4 skipped** |

[`cuda_test_probe.py`](cuda_test_probe.py) (a pytest plugin) records for each test whether the test allocates CUDA memory. At the start, **no test used CUDA** (all tests hard-code `device="cpu"`). With a visible GPU, the only difference is that `checkpoint.rng_state()` also saves a CUDA random-number state.
At the end, only the 3 new CUDA regression tests of this run allocated GPU memory (see Section 12).

**Smoke test** `zero.smoke`: all `configs/tiny/*.toml` files hard-code `device = "cpu"` and `dtype = "fp32"`, and `pick_device("cpu")` returns the CPU directly. Thus, **when you run `zero.smoke` on a machine with a GPU, all training still runs on the CPU** (the run only initializes the CUDA context).

```bash
flock <gpu0.lock> env CUDA_VISIBLE_DEVICES=0 UV_NO_SYNC=1 uv run python -m zero.smoke --out out/gpu0-check/smoke
```

To really use CUDA, [`smoke_cuda.py`](smoke_cuda.py) changes `[train] device` to `cuda` and `dtype` to `auto` (BF16 autocast on CUDA) when it reads the configuration. All other parts are the same as `zero.smoke` (in zero.smoke, eval / export / demo already run on the CPU):

```bash
flock <gpu0.lock> env CUDA_VISIBLE_DEVICES=0 UV_NO_SYNC=1 uv run python runs/2026-10-01-gpu0-check/smoke_cuda.py --out out/gpu0-check/smoke_cuda
```

| Stage | `zero.smoke` (really on the CPU, single thread) | `smoke_cuda.py` (GPU0, BF16, after the fix of the distillation device bug) |
|---|---|---|
| data+tokenizer | ok, 4.7 s | ok, 4.0 s |
| pretrain (200 steps) | ok, 80.7 s, loss 5.338 / val 5.482 | ok, 6.6 s, loss 5.257 / val 5.457 |
| midtrain (60 steps) | ok, 27.4 s, val 5.453 | ok, 1.5 s, val 5.431 |
| SFT (240 steps) | ok, 290.6 s, val 0.624 | ok, 6.5 s, val 0.621 |
| Distillation (20 steps) | ok, 20.6 s, kd 0.0096 | ok, 5.4 s, kd 0.0129 (student and teacher both on cuda:0) |
| DPO (24 steps) | ok, 23.9 s, acc 0.75 | ok, 5.7 s, acc 0.75 |
| GRPO (10 steps) | ok, 57.5 s; sampling 1.43 s per step, full step 5.74 s | ok, 9.9 s; sampling 0.91 s per step, full step 0.98 s |
| eval / export / demo | ok, 21.8 / 12.7 / 0.0 s; HF logits difference 0.0, template identical character for character | ok, 17.9 / 6.3 / 0.0 s; the same as on the left |
| **Total** | **540 s (all 10 stages passed)** | **64 s (all 10 stages passed)** |

## 1. BF16 + SDPA uses FlashAttention (RUNBOOK #1)

```bash
flock <gpu0.lock> env CUDA_VISIBLE_DEVICES=0 UV_NO_SYNC=1 uv run python runs/2026-10-01-gpu0-check/sdpa_check.py
```

Main-line shapes (16 query heads, 8 K/V heads, head_dim 128, T=4096, BF16, causal), forward + backward pass. FP32 math is the reference for the error:

| Form | FLASH | EFFICIENT | CUDNN | MATH |
|---|---|---|---|---|
| `enable_gqa=True` (the form in zero) | **works**, 4.65 ms, 409 MiB | **No available kernel** | works, 4.59 ms | 55.7 ms, 4432 MiB |
| `repeat_interleave` K/V first | works, 5.56 ms | 17.0 ms | 5.06 ms | 55.0 ms |

- All combinations that run have a maximum absolute error of 7.5e-3 against FP32 math (BF16 rounding).
- **Without any restriction, PyTorch selects Flash**: the profiler shows the kernels `pytorch_flash::flash_fwd_kernel` /
  `flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel` (for both forms: `enable_gqa=True` and copy first).
- `torch.backends.cuda.can_use_*_attention(enable_gqa=True)`: flash True, cudnn True, **efficient False**
  (the warning text: "For dense input, both fused kernels require query, key and value to have the same num_heads").
- The original command of the RUNBOOK (the 689.5M model with `.cuda().bfloat16()`, a forward pass of 1×4096 under `sdpa_kernel(FLASH_ATTENTION)`):
  **passed**, output `[1, 4096, 65536]`. The same command with EFFICIENT gives "No available kernel".

Training steps of the full model (FP32 master weights + BF16 autocast, the same as Trainer; micro batch 1 × 4096; forward + backward pass, without the optimizer):

| Attention | Activation checkpointing off | Activation checkpointing on |
|---|---|---|
| Default (`enable_gqa=True`, Flash selected automatically) | 5,872 tok/s, MFU 57.5%, 19.5 GiB | 4,640 tok/s, 45.5%, 9.6 GiB |
| Flash forced (`enable_gqa=True`) | 5,817 tok/s, 57.0% | 4,627 tok/s, 45.3% |
| `repeat_interleave` first, then Flash | 5,722 tok/s, 56.1%, 20.0 GiB | 4,538 tok/s, 44.5% |
| `repeat_interleave` first, then efficient | 3,876 tok/s, 38.0% | (script limit, see below) |
| efficient forced (`enable_gqa=True`) | No available kernel | No available kernel |

Conclusion: **passed**. On the 3090 (sm_86) with torch 2.11, `enable_gqa=True` goes directly to the FlashAttention-2 backend. It is about 2% faster than copying K/V first. `Attention` needs no change.
Only one point needs attention: the efficient backend does not support `enable_gqa`. If a later change forces efficient (or Flash falls back because it cannot use a shape), the run stops with an error. It does not become slower without a message.
(The failure in the cell "script limit" of the table is a problem of the script. `sdpa_kernel` wraps only the forward pass. In the backward pass, activation checkpointing recomputes with the default Flash kernel. The saved tensors then have different shapes, and `torch.utils.checkpoint` reports inconsistent metadata. zero itself does not use `sdpa_kernel`, so zero is not affected.)

## 2. Training path (single-GPU version of RUNBOOK #2)

```bash
flock <gpu0.lock> bash runs/2026-10-01-gpu0-check/train_run.sh pretrain_l20m runs/2026-10-01-gpu0-check/configs/l20m_tiny.toml \
    --set train.max_steps=200 --set schedule.warmup_steps=20 --set checkpoint.every=100 --set train.eval_every=100 --set logging.every=10
# l60m: the same; l150m: also add --set train.micro_batch_size=4 --set train.grad_accum_steps=4 --set train.max_steps=100
```

[`train_run.sh`](train_run.sh) runs `uv run torchrun --standalone --nproc_per_node=1 -m zero.train.pretrain --config ...`.
At the same time, it records the GPU memory with nvidia-smi every 0.5 s. Each step has 32,768 tokens (= the single-GPU share of the ladder configuration).

| Configuration | Parameters | micro × accumulation | loss (step 1 → last step) | val (step 100 → 200) | tok/s (steady state) | MFU (3090) | Peak memory (nvidia-smi) |
|---|---:|---|---|---|---:|---:|---:|
| l20m | 23.1M | 8 × 2 | 11.135 → 5.429 | 6.221 → 5.627 | ≈ 98,500 | 26% | 17.1 GB |
| l60m | 71.3M | 8 × 2 | 11.175 → 5.354 | 6.218 → 5.627 | ≈ 45,600 | 37% | 21.6 GB |
| l150m | 160.5M | 4 × 4 (100 steps) | 11.227 → 6.243 | 6.316 (step 100) | ≈ 24,400 | 43% | 16.8 GB |
| l150m | 160.5M | 8 × 2 | — | — | — | — | **OOM** (FP32 logits 8×2048×65536 need 4 GiB) |

Key log lines: `Model parameters 23.07M (non-embedding 6.30M), 32768 tokens per step, 200 steps, device cuda:0, world_size=1`;
`step 200/200 | loss 5.4290 | lr 0.00e+00 | gnorm 0.22 | 97,961 tok/s | MFU 26.0% | val 5.6273`.

Conclusion: **the single-GPU path passed** (BF16 autocast, fused AdamW, gradient accumulation, the WSD schedule, validation loss / bpb, and checkpoints all work; the loss goes down smoothly).
Note: with `torchrun --nproc_per_node=1`, `WORLD_SIZE=1`, and `wrap_model` **does not wrap the model in DDP**. Thus this item **did not** verify DDP.
For the DDP wrapper on one GPU, see Section 4. For DDP across GPUs, see Section 14. The default micro batch of the ladder configurations (16 × 2048) is for 80 GB GPUs. On a 24 GB GPU, l20m / l60m fit only with 8, and l150m fits only with 4.

## 3. Resume from a checkpoint (single-GPU version of RUNBOOK #3)

```bash
flock <gpu0.lock> bash runs/2026-10-01-gpu0-check/resume_check.sh                                        # default kernels
flock <gpu0.lock> env DET=1 STEPS=100 EVERY=50 KILL_AT=75 TAG=resume_det bash runs/2026-10-01-gpu0-check/resume_check.sh   # deterministic algorithms
```

[`resume_check.sh`](resume_check.sh) uses l20m. First, it makes a control run without interruption. Then it runs the same command again. When the log shows step KILL_AT, the script sends
`kill -9` to torchrun and the worker. (torchrun starts the worker with `start_new_session`, so a kill of the process group does not reach the worker. Thus the script uses `pkill -9` on the unique output folder in the command line.)
Then the script runs the command again without changes, and the run resumes from the latest checkpoint automatically.

| Comparison | Default kernels (200 steps, save every 100 steps, kill at step 150) | Deterministic algorithms (100 steps, save every 50 steps, kill at step 75) |
|---|---|---|
| Resume log | `Resumed training from out/gpu0-check/resume/kill/ckpt/step_00000100 (step 100)` | `Resumed training from .../resume_det/kill/ckpt/step_00000050 (step 50)` |
| **Same run**: after the resume vs before the kill (both start from the same checkpoint) | Steps 101–108 **bit-identical**; steps 101–150: maximum difference 1.1e-3 | Steps 51–75 **all bit-identical** |
| Resume vs the control without interruption | Maximum difference 0.49 (two independent runs alone differ by up to 2.09 in steps 1–150; see convention 3 in "Read first") | Steps 51–100 **all bit-identical**; step 100 val_loss 6.207895 = 6.207895 |
| Same learning rate at each step | Yes | Yes |

Conclusion: **passed**. With deterministic algorithms, the resume after kill -9 is bit-identical to the run without interruption. (The model, the AdamW state, the WSD schedule, the position of the data loader, and the random-number state are all restored exactly.) With the default kernels, the first 8 steps after the resume are also bit-identical. The later differences come from the non-determinism of the GPU kernels themselves, and they have the same size as the differences between two independent runs.
A separate direct test checks the save / restore of the CUDA random-number state in `checkpoint.py`: `rng_state()` → draw 1000 CUDA random numbers → change the state →
`set_rng_state()` → draw again. The two draws are bit-identical (part 6 of `parity_cuda.py`). (The training loop itself does not use CUDA random numbers, so the consistency of the resume does not depend on this state.)

## 4. Single-GPU path of FSDP2 and DDP (single-GPU version of RUNBOOK #4)

`torchrun --nproc_per_node=1 ... --set train.parallel=fsdp` **does nothing**: with `WORLD_SIZE=1`, `DistInfo.is_distributed`
is False, and `wrap_model` returns the model without changes. The same is true for DDP. Thus [`dist1_check.py`](dist1_check.py) makes its own
NCCL process group with world_size=1 and forces `is_distributed` to True. Then the DDP / FSDP2 branches of `wrap_model` and `checkpoint.py` really run:

```bash
flock <gpu0.lock> env CUDA_VISIBLE_DEVICES=0 UV_NO_SYNC=1 uv run python runs/2026-10-01-gpu0-check/dist1_check.py
```

l20m + tiny data, 50 steps, deterministic algorithms; plain = no wrapper (the same as torchrun on one GPU):

| Comparison | lr 3e-3 (ladder default) | lr 3e-4 |
|---|---|---|
| DDP (`DistributedDataParallel`, NCCL, with `no_sync` gradient accumulation) vs plain | **bit-identical** | **bit-identical** |
| FSDP2 (`fully_shard` + `MixedPrecisionPolicy(bf16, fp32 reduction)`) vs plain, maximum relative difference of the per-step loss | 0.57% | 0.11% |

- FSDP2 checkpoint (`get_model_state_dict` / `get_optimizer_state_dict` gather one full state_dict):
  we deleted everything after step 25 and resumed from step 25 to 50 (`set_model_state_dict` / `set_optimizer_state_dict`). The loss is **bit-identical to the run without interruption**.
- The single-GPU `load_policy` can read the FSDP2 checkpoint back (strict load of 91 tensors). On real validation data, the CE is 6.3398.
  The plain checkpoint of the same step gives 6.3417. The relative difference is 3.1e-4.
- We ran the script twice (between the two runs, we changed only the check method for load_policy). All loss numbers of the two runs are exactly the same. With deterministic algorithms, the results are reproducible.
- The peak memory of the three wrappers is 14.5–14.9 GiB. (On one GPU, FSDP has nothing to shard; the run only goes through the code path.)

Conclusion: **the single-GPU path passed** (the criterion "error < 1%" is met). The result shows that these parts run on a GPU: the call of `fully_shard`, the mixed-precision policy, fused AdamW with DTensor parameters,
`clip_grad_norm_`, and the gather and restore of the FSDP state_dict. **Real sharding (world_size > 1) and consistency across GPUs still need multiple GPUs.**

## 5. torch.compile (single-GPU version of RUNBOOK #5)

```bash
flock <gpu0.lock> bash runs/2026-10-01-gpu0-check/compile_check.sh             # off ×2, on ×1 (default kernels)
flock <gpu0.lock> env PART=det bash runs/2026-10-01-gpu0-check/compile_check.sh  # off and on, 1 run each, with deterministic algorithms
```

l60m + tiny data, 60 steps (warmup 10, `logging.every=1`, `checkpoint.every=0`):

| | compile off (a) | compile off (b) | compile on | off (deterministic) | on (deterministic) |
|---|---:|---:|---:|---:|---:|
| tok/s (mean of steps 21–60) | 46,984 | 46,509 | **64,363** | 43,800 | 61,049 |
| MFU (3090) | 38.3% | 37.9% | **52.5%** | 35.7% | 49.8% |
| Time of step 1 (with compilation) | 1.1 s | 1.1 s | 30.7 s | 1.2 s | 32.2 s |
| step 60 loss | 6.264 | 6.209 | 6.357 | 6.249 | 6.304 |

- Loss comparison: with the same batch and the same initial weights, compile and eager differ by only 1.3e-5 at step 1 (rounding differences from fusion).
  Mean relative difference of the per-step loss over 60 steps: on vs off 0.91%; **off vs off (two eager runs) 1.12%**; on vs off with deterministic algorithms 0.66%.
  The difference from compilation is not larger than the noise between two eager runs.
- Main-line 689.5M shape (T=4096, activation checkpointing, micro batch 3): eager 4,909 tok/s → compile **6,944 tok/s (+41%)**.
  The loss of the first 3 steps differs from eager by ≤ 2e-4 (`mem_probe.py`; see Section 7).

Conclusion: **passed on one GPU**. On one GPU, compile increases the throughput by +37% (l60m) / +41% (main-line shape). We did not verify compile together with DDP / FSDP.

## 6. Summary of MFU measurements (RUNBOOK #6, single-3090 version)

| Model / sequence | Settings | tok/s | MFU (÷71 TFLOPS) | Source |
|---|---|---:|---:|---|
| l20m / 2048 | eager, 8 × 2 | 98,500 | 26% | Section 2 |
| l60m / 2048 | eager, 8 × 2 | 45,600–47,000 | 37–38% | Sections 2, 5 |
| l60m / 2048 | **compile**, 8 × 2 | 64,400 | 52% | Section 5 |
| l150m / 2048 | eager, 4 × 4 | 24,400 | 43% | Section 2 |
| Main line 689.5M / 4096 | eager, activation checkpointing, micro 3 | 4,909 | 48% | Section 7 |
| Main line 689.5M / 4096 | **compile**, activation checkpointing, micro 3 | 6,944 | 68% | Section 7 |
| Main-line shape / 8192 | eager, activation checkpointing, micro 1 | 4,122 | 57% | Section 8 |

How to read the table: the smaller the model, the larger the fraction of lm_head (vocabulary 65,536) and of element-wise operations, and the lower the MFU. At main-line size on the 3090, eager reaches about one half and compile about two thirds.
(This uses the MFU basis of zero: the attention term is not halved, so the values are too high; see convention 2 in "Read first".) These numbers **cannot** be converted directly to the H100:
the H100 has a different ratio of compute to bandwidth, and NVLink and the communication of 8 GPUs add different costs. The MFU of 0.4 in the RUNBOOK still needs a measurement on the H100.

## 7. Memory and batch (RUNBOOK #7, main line 689.5M, T=4096, one 24 GB GPU)

```bash
flock <gpu0.lock> env CUDA_VISIBLE_DEVICES=0 UV_NO_SYNC=1 uv run python runs/2026-10-01-gpu0-check/mem_probe.py main
```

[`mem_probe.py`](mem_probe.py) starts a new process for each probe. The process runs 3 steps with the real `Trainer` (forward + backward pass + fused AdamW, `stop_at=3`).
The configuration is [`configs/main_tiny.toml`](configs/main_tiny.toml) (main-line shape + tiny data). The micro batch increases from 1 until OOM.
The estimate column is `zero.tools.memory_calc.estimate_memory(..., num_gpus=1, strategy="ddp", dtype="bf16")` (on one GPU, there are no DDP buckets).

| Activation checkpointing | compile | micro batch | Result | PyTorch peak allocated | Peak reserved | memory_calc estimate | tok/s | MFU |
|---|---|---:|---|---:|---:|---:|---:|---:|
| off | off | 1 | **OOM** | 22.13 GiB (at OOM) | 23.08 | 24.02 | — | — |
| on | off | 1 | ok | 11.55 GiB | 13.66 | 13.92 | 4,578 | 44.8% |
| on | off | 2 | ok | 15.12 GiB | 18.44 | 17.37 | 4,812 | 47.1% |
| on | off | 3 | ok | 18.71 GiB | 22.98 | 20.82 | 4,909 | 48.1% |
| on | off | 4 | **OOM** (needs another 4 GiB for the FP32 logits) | — | — | 24.27 | — | — |
| on | on | 3 | ok | 18.65 GiB | 21.92 | 20.82 | 6,944 | 68.0% |

- On a 24 GB GPU: **without activation checkpointing, not even micro batch 1 fits; with it, the maximum is 3**. All "fits / does not fit" decisions of memory_calc are correct
  (`max_micro_batch(..., gpu_gib=23.56)` gives 0 and 3).
- memory_calc is 2.1–2.4 GiB higher than the peak that PyTorch really allocates (a conservative upper bound, as its documentation says). At micro batch 3, the reserved
  peak of the caching allocator is another 2.2 GiB above the estimate. Thus the advice in its documentation, "keep another 2–5 GB of margin", is necessary.
- On CUDA, the gradients with activation checkpointing are **bit-identical** to the gradients without it (in both FP32 and BF16 autocast; part 3 of `parity_cuda.py`).
  At the main-line shape, the throughput cost is about 21% (Section 1: 5,872 → 4,640 tok/s).
- The largest part of the memory is lm_head: when micro batch 4 fails, the allocation that fails is exactly the FP32 logits of 4 × 4096 × 65,536 (4 GiB). A chunked cross-entropy
  (which does not make all FP32 logits at once) can save much memory. This is a code change. We did not do it; we only record it here.
- What this means for 80 GB GPUs (our deduction, not measured): for T=4096 on one GPU, memory_calc estimates about 61 GiB for micro 4 without checkpointing. This agrees with the choice in the main-line configuration.
  RUNBOOK #7 probes "micro 16 × accumulation 1". For it, memory_calc (8-GPU DDP) estimates about 214 GiB without checkpointing and about 68 GiB with checkpointing.
  Add the reserved margin that this section shows, and even the run with checkpointing is tight on 80 GB. This needs a measurement on the H100.

## 8. Long sequences, 32K (single-GPU version of RUNBOOK #8)

```bash
flock <gpu0.lock> env CUDA_VISIBLE_DEVICES=0 UV_NO_SYNC=1 uv run python runs/2026-10-01-gpu0-check/mem_probe.py longctx
```

The configuration is [`configs/longctx_tiny.toml`](configs/longctx_tiny.toml). It inherits `configs/main/longctx.toml` (θ=1e6, max_seq_len 32768).
It does not load the midtrain checkpoint and measures only memory. On one GPU, `parallel = "fsdp"` has no effect (see Section 4).

| Sequence length | Activation checkpointing | Result | PyTorch peak allocated | memory_calc estimate |
|---:|---|---|---:|---:|
| 32,768 | on | **OOM** (needs another 8 GiB: FP32 logits 32768 × 65536) | 19.57 GiB (at OOM) | 38.07 |
| 32,768 | off | **OOM** | 21.98 GiB (at OOM) | 111.28 |
| 16,384 | on | **OOM** (needs another 4 GiB) | 17.19 GiB (at OOM) | 24.27 |
| 8,192 | on | ok, loss 11.41 → 11.29, 4,122 tok/s, MFU 56.8% | 15.15 GiB (reserved 18.51) | 17.37 |

Conclusion, recorded as measured: **32K does not fit on one 24 GB GPU, even with micro batch 1 + activation checkpointing**. The upper limit on one GPU is 8K.
The bottleneck is the logits (32K × 65,536 × several BF16/FP32 copies ≈ 20 GiB), not attention (the memory of the Flash kernel is linear in T).
The RUNBOOK plans 8×H100 + FSDP. FSDP shards only the parameters / gradients / optimizer states, and **it cannot shard the logits of each GPU**. By memory_calc,
for 32K, micro 1, and checkpointing on, 8-GPU DDP needs about 40.6 GiB per GPU, and 8-GPU FSDP needs about 29.1 GiB per GPU. This should fit on 80 GB, but it needs a measurement.
To do 32K on smaller GPUs later, we need a chunked cross-entropy.

## 9. Post-training paths (single-GPU version of RUNBOOK #9)

- **SFT / distillation / DPO / GRPO all run on CUDA + BF16 autocast**: see `smoke_cuda.py` (the table in Section 0). The loss / reward of each stage
  has the same order of magnitude as in the CPU smoke test. We fixed one bug: the student of distillation **always trained on the CPU** (Section 12). After the fix, the log shows
  `Model parameters 1.31M ..., 20 steps, device cuda:0`.
- **SFT from the torchrun entry point** (part 1 of [`post_check.sh`](post_check.sh)):

  ```bash
  uv run torchrun --standalone --nproc_per_node=1 -m zero.post.sft --config configs/tiny/sft.toml \
      --set train.device=cuda --set train.dtype=auto --set train.max_steps=20 ... (the paths point to the outputs of smoke_cuda)
  ```

  `Loaded model weights from out/gpu0-check/smoke_cuda/midtrain/ckpt (step 60)`; `step 20/20 | loss 2.4487 | ... | val 2.6198`. Passed.
- **Sampling time per GRPO step**: for the tiny model (1.3M, G=8, 4 problems per step, 64 new tokens), sampling takes 0.91 s per step (1.43 s on the CPU).
  The Python loop over the tokens limits the small model, so the GPU does not make it much faster.
  For the sampling speed of the main-line 689.5M model, we use [`grpo_sample_bench.py`](grpo_sample_bench.py) (random weights; G=16 and `max_new_tokens=512` from `configs/main/grpo.toml`;
  a prompt of 300 tokens; each sample generates all 512 tokens):

  | Form | One group (16 samples × 512 tokens) | Decode throughput | Memory | Converted: 64 problems per step |
  |---|---:|---:|---:|---:|
  | FP32 (the current form in `run_grpo`: `sample_group` is not in autocast) | 17.9 s | 457 tok/s | 6.7 GiB | ≈ 1,150 s |
  | BF16 autocast around generate | 15.0 s | 546 tok/s | 7.3 GiB | ≈ 960 s |
  | Model `.bfloat16()` (the KV cache is also BF16) | 13.7 s | 598 tok/s | 3.3 GiB | ≈ 880 s |

  On one 3090, the sampling alone of one main-line GRPO step takes 15–19 minutes (500 steps take about 130–160 hours). The cause: the autoregressive decoding handles one prompt at a time, with a batch of only G=16.
  This decoding is fully limited by latency / bandwidth. The H100 is several times faster, but the structural problem stays the same: **before GRPO in stage 10, sample all 64 prompts
  together in one large batch, or change to vLLM / verl as the module docstring of zero/post/grpo.py says** (this run did not change the code).
- vLLM, lm-eval, bfcl: **skipped: not installed** (as instructed, we did not install them).

## 10. Export (single-GPU version of RUNBOOK #10)

```bash
uv run python runs/2026-10-01-gpu0-check/export_check.py out/gpu0-check/smoke_cuda/grpo/ckpt out/gpu0-check/export_grpo
uv run python runs/2026-10-01-gpu0-check/export_check.py out/gpu0-check/pretrain_l20m/ckpt out/gpu0-check/export_l20m
```

[`export_check.py`](export_check.py): `load_policy(ckpt, device="cuda")` (the model is directly on the GPU) → `export_to_hf_qwen3(..., chat=True)`
→ `transformers.AutoModelForCausalLM.from_pretrained(...).cuda()`, then a parity check on the GPU.

| Model | Maximum difference of FP32 logits (HF vs zero) | Greedy, 32 tokens | Chat template identical character for character | BF16 export: HF vs zero (both BF16) | Top-1 agreement, BF16 vs FP32 |
|---|---:|---|---|---:|---:|
| GRPO model of smoke_cuda (1.3M, trained on the GPU) | 6.7e-6 (maximum logit 20.0) | identical | yes | 0.0 | 99.1% |
| l20m pretraining, 200 steps (23.1M, trained on the GPU) | 7.6e-6 (maximum logit 11.2) | identical | yes | 0.0625 | 98.5% |

Conclusion: **passed** (HF export, then transformers loads the model on the GPU for the parity check). vLLM loading and the hermes tool-call parser: **skipped: not installed**.
GGUF: in the smoke test, `convert_hf_to_gguf` succeeded (f16, 2.7 MB). llama.cpp is not compiled, so we did not quantize or run the model (this is not related to the GPU).
Note: the first time that `zero.smoke` ran, its export stage made a shallow clone of llama.cpp into
`~/.cache/zero/llama.cpp` automatically, as `zero/export/gguf.py` is designed to do (214 MB of source code, at 10:05). This is the only network download of this run. We downloaded no data sets and no model weights.

## 11. Other modules with the label "not verified on a GPU yet" (CUDA parity check)

```bash
flock <gpu0.lock> env CUDA_VISIBLE_DEVICES=0 UV_NO_SYNC=1 uv run python runs/2026-10-01-gpu0-check/parity_cuda.py
```

[`parity_cuda.py`](parity_cuda.py) uses the same weights and the same batch of tokens (B=4, T=128, dim 128, 4 layers). It does one forward + backward pass each on CPU FP32 / CUDA FP32 / CUDA BF16 autocast,
and it reports the relative error ‖a−b‖/‖b‖ (with CPU FP32 as the reference). The table below shows the results **after the fix of the two bugs in Section 12**:

| Module | CUDA FP32: output / gradient | CUDA BF16: output / gradient | Cached generation = recomputation at each step (CUDA) |
|---|---|---|---|
| Main-line Transformer (control) | 1.3e-6 / 1.8e-6 | 1.5e-2 / 2.1e-2 | Yes (KV cache, 60 tokens) |
| MoE (top-2 of 8 + shared expert) | 1.1e-6 / 1.6e-6 | 5.4e-2 / 7.6e-2 (before the fix: **error**) | — |
| MLA / MLA + q_lora | 1.4e-6 / 2.0e-6 | 1.5e-2 / 2.2e-2 | Yes (absorption + latent-vector cache) |
| Sliding window (window 32, 1:1) | 1.3e-6 / 1.8e-6 | 1.5e-2 / 2.1e-2 | Yes (ring buffer on cuda:0) |
| Gated DeltaNet hybrid (3:1) | 2.5e-6 / 2.0e-5 | 3.8e-2 / **2.0e-1** | Yes (before the fix: **error**) |
| Linear-attention hybrid (3:1) | 1.7e-6 / 4.9e-6 | 2.2e-2 / 5.1e-2 | Yes (before the fix: **error**) |
| MTP (depth 1) | 1.3e-6 / 1.8e-6 | 1.5e-2 / 2.1e-2 | Self-speculative greedy = greedy of the main model |
| Speculative decoding (1-layer draft / draft = target) | — | — | Greedy output identical token for token to the target model; acceptance rate 100% when draft = target |

- CUDA FP32 and CPU FP32 are all within 1e-5. Most BF16 errors have the same size as the main-line control (about 2%). Two errors are larger:
  MoE (in BF16, the top-k of the router selects a different expert when two scores are almost equal) and **the gradient of Gated DeltaNet (20%)**. The pure-PyTorch chunkwise delta rule
  is clearly not precise enough under BF16 autocast. For real training, change to the fla kernel as the file says (the kernel accumulates in FP32 internally), or compute this part in FP32.
- **fused AdamW** (`trainer.py` turns it on by default on CUDA): with the same sequence of gradients for 20 steps, the maximum parameter difference of fused vs foreach is 4.8e-6, and vs CPU AdamW it is also 4.8e-6
  (the total parameter change is 2.3e-2). All GPU training in Sections 2–8 used the fused path. **Passed.**
- **Muon**: on CUDA, NS5 computes in BF16 by default (and returns the dtype of the input). The relative difference from CPU FP32 is about 2.1–2.3% (CUDA FP32 and CPU FP32 differ by < 1.2e-4).
  For non-square matrices, NS5 pushes the singular values into [0.68, 1.15], as NS5 is designed to do. One NS5 on a 1280 × 3584 matrix takes about 3 ms. `MuonAdamW` runs 5 steps on CUDA and 5 steps on the CPU;
  the relative difference of the total update is 1.25%. Training l20m with `optim.name=muon` for 100 steps: loss 11.135 → 4.288, val 4.496, about 94k tok/s
  (AdamW about 99k), memory 17.1 GB (part 3 of `post_check.sh`). **Passed on one GPU**; numerical consistency under DDP needs multiple GPUs.
- **CUDA random-number state**: see Section 3. **Passed.**
- Throughput for reference (dim 768, 12 layers, T=1024, batch 8, BF16 autocast, forward + backward pass). These numbers show only that the modules "run"; they are **not a performance verification**:
  dense GQA 39.3k tok/s; MoE (top-2 of 8, active parameters ≈ dense) 24.0k; MLA (kv_lora 256) 32.9k;
  sliding window (window 512, 1:1) 19.9k (a custom boolean mask cannot use the Flash kernel, so the sliding window is only half as fast as full attention); Gated DeltaNet hybrid 15.1k (pure PyTorch).

## 12. Problems found and fixes

| # | Problem | Scope | Action |
|---|---|---|---|
| 1 | `run_distill` in `zero/post/distill.py` builds the Trainer of the student with `DistInfo()` (CPU by default). With a GPU and `device = "auto"/"cuda"`, **the student still trains on the CPU**. `DistillTrainer` also moves the teacher, which was on the GPU, back to the CPU. There is no error; the run is only slow | Only when CUDA is available | **Fixed**: changed to `DistInfo(device=pick_device(tc.device))` (the results on the CPU do not change); new test `tests/test_distill.py::test_run_distill_trains_student_on_cuda` (skips without CUDA) |
| 2 | `generate_greedy` in `zero/arch/linear_attention.py` always makes the input and the cache on the CPU. When the model is on CUDA, it reports "Expected all tensors to be on the same device" | Only CUDA | **Fixed**: makes the input and the cache on the device of the model; new test `tests/test_arch_linear_attention.py::test_generate_greedy_on_cuda` (skips without CUDA) |
| 3 | `MoEFFN.forward` in `zero/arch/moe.py` reports `index_add_(): self (Float) and source (BFloat16) must have the same scalar type` under BF16 autocast (the expert output is BF16, and the accumulation buffer is FP32). BF16 autocast on the CPU also causes this error | The autocast path; CPU FP32 is not affected | **Fixed**: converts to the precision of the buffer before the accumulation (no operation in FP32); new CPU test `tests/test_arch_moe.py::test_moe_forward_backward_under_bf16_autocast` |
| 4 | GRPO sampling (`sample_group`) is not in autocast, so it runs in FP32 on CUDA. It also decodes one prompt at a time with batch = G. At main-line size on the 3090, the sampling of each step takes 15–19 minutes | Performance | **Not changed** (a large change); see Section 9 |
| 5 | The pure-PyTorch chunkwise implementation of Gated DeltaNet has a relative gradient error of 20% under BF16 autocast | Precision (experimental module) | **Not changed**; see Section 11 |
| 6 | The sliding window uses SDPA with a custom mask. On CUDA, it cannot use Flash, so its throughput is only half of full attention | Performance (experimental module) | **Not changed**; the file already says that production must use window_size of FlashAttention / FlexAttention |
| 7 | The largest part of the memory is the FP32 logits: at main-line T=4096, micro batch 4 does not fit, and 32K does not fit on one GPU | Memory | **Not changed** (chunked cross-entropy is a code change); see Sections 7 and 8 |
| 8 | `configs/tiny/*.toml` hard-codes `device = "cpu"`, so `zero.smoke` does not use the GPU, even on a GPU machine | Coverage of the smoke test | **Not changed** (as instructed, we did not change configs/); `smoke_cuda.py` fills the gap |
| 9 | The documentation of `zero/tools/memory_calc.py` says "zero does not implement activation checkpointing yet". This text is out of date | Documentation | Changed it to "zero implements it" at the same time (in the same paragraph as the GPU label) |

After the changes, `CUDA_VISIBLE_DEVICES= UV_NO_SYNC=1 uv run pytest -q`: 321 passed, 7 skipped; on GPU0: 324 passed, 4 skipped; `ruff check zero tests runs/2026-10-01-gpu0-check` passed.

## 13. Items that are still not verified

- When Sections 0–12 were complete, these items still "needed multiple GPUs": DDP across GPUs, resume across GPUs, FSDP2 sharding on multiple GPUs, NCCL communication, the DDP consistency of Muon, and compile together with DDP.
  At that time, the permission system refused the commands for GPU1 (this agreed with the first rule, "use only GPU0"), so none of these commands ran. Later, the project lead reserved GPU1 and GPU2 personally.
  **Section 14** then did these items on 2×RTX 3090.
- Still not verified: throughput and MFU on **8 GPUs** (the real scale of the RUNBOOK) and with **NVLink**; compile together with FSDP;
  the all_reduce of `update_bias` of MoE in distributed training; multi-GPU versions of DPO / GRPO / distillation (not implemented); vLLM / lm-eval / bfcl (not installed).
- FP8 (`plan_budget.py`): not implemented, and the 3090 does not support it. FlashAttention varlen (isolation for SFT packing), fla / causal-conv1d kernels, and
  grouped GEMM: not implemented / not installed, so they are still "not verified on a GPU yet". (For the speed of the FlexAttention sliding window, see the GPU measurements of Chapter 22.
  But `sliding_window.py` of zero still uses boolean-mask SDPA.)

## 14. Multi-GPU verification (2×RTX 3090, PCIe)

Hardware: physical GPU0 + GPU2 (both with a power cap of 240 W), `CUDA_VISIBLE_DEVICES=0,2`, `torchrun --standalone --nproc_per_node=2`.
There is no NVLink between the two GPUs. **The throughput and MFU here show only what 2 GPUs on PCIe can reach. They do not represent 8×H100 with NVLink.**

### 14.0 NCCL

```bash
CUDA_VISIBLE_DEVICES=0,2 UV_NO_SYNC=1 uv run torchrun --standalone --nproc_per_node=2 runs/2026-10-01-gpu0-check/nccl_probe.py
```

[`nccl_probe.py`](nccl_probe.py): the all-reduce on the two GPUs gives the correct result (3072). 256 MiB takes 57.4 ms, **a bus bandwidth of about 4.7 GB/s**
(the usual magnitude without NVLink and without working P2P; H100 NVLink is more than 100 times faster). `NCCL_P2P_DISABLE` is not necessary, and nothing hangs.
This run also verifies the `torch.cuda.set_device` line in `dist.py` (the line runs only with multiple GPUs).

### 14.1 Step-by-step parity check of DDP / FSDP2 / Muon (RUNBOOK #2, #4)

```bash
CUDA_VISIBLE_DEVICES=0 UV_NO_SYNC=1 uv run python runs/2026-10-01-gpu0-check/dist2_check.py ref     # single-GPU reference
CUDA_VISIBLE_DEVICES=0 UV_NO_SYNC=1 uv run python runs/2026-10-01-gpu0-check/dist2_check.py ctrl    # single-GPU control
CUDA_VISIBLE_DEVICES=0,2 UV_NO_SYNC=1 uv run torchrun --standalone --nproc_per_node=2 \
    runs/2026-10-01-gpu0-check/dist2_check.py multi
```

[`dist2_check.py`](dist2_check.py): l20m shape, 50 steps, deterministic algorithms on in all runs. Each step has 16 sequences × 2048 tokens:
single-GPU reference = micro 8 × accumulation 2; DDP = micro 8 per GPU; `ddp_accum` = micro 4 × accumulation 2 per GPU (uses `no_sync`); FSDP2 = micro 8 per GPU.

**First, a trap in the setup**: the first run used the multi-source `l20m_tiny.toml`. The loss of step 1 (before any update) was 11.1350 on one GPU and 11.1327 on two GPUs,
so the two runs did not use the same samples. The cause: the source sampler of `MixtureLoader` uses `(seed, rank)` as its seed, so each GPU selects its sources separately (`zero/DESIGN.md` already says this;
it is not a bug). Thus the step-by-step parity check against one GPU uses the single-source configuration [`configs/l20m_tiny_1src.toml`](configs/l20m_tiny_1src.toml) instead.
(This configuration merges the shards of the three sources into one `PackedDataLoader`; global sample g goes to rank g % world_size.) We keep the results of the multi-source runs too.
They serve for "2-GPU FSDP vs 2-GPU DDP" (the two runs see the same data).

| Parity check | Result |
|---|---|
| Loss of step 1 (single source) | One GPU 11.155300 = DDP 11.155300; FSDP 11.155274 (the parameters take part in the computation as BF16; difference 2.3e-6) |
| Parameters on each GPU in DDP (after 50 steps of each DDP run; bits read as integers and summed) | **Bit-identical on the two GPUs** (ddp, ddp_accum, ddp lr 3e-4, muon_ddp: all) |
| DDP vs one GPU, steps 2–10 | lr 3e-3: relative difference ≤ 8.4e-5; lr 3e-4: ≤ 1e-6 (only the summation order differs) |
| **Single-GPU control**: micro 4 × 4 vs micro 8 × 2 (mathematically equal; only the summation order differs) | lr 3e-3: above 1e-3 from step 14, maximum 21.9% within 50 steps; lr 3e-4: maximum 0.12% |
| DDP vs one GPU, maximum relative difference within 50 steps | lr 3e-3: 26.7% (also above 1e-3 from step 14); lr 3e-4: 2.5% (at the loss spike of step 27; ≤ 0.5% at all other steps) |
| `ddp_accum` (no_sync) vs one GPU | lr 3e-3: 27.2%, the same order of magnitude as DDP |
| **FSDP2 vs DDP (multi-source data, the same data for both), lr 3e-4** | **Maximum 0.10%** (RUNBOOK criterion < 1%: passed); 7.5% at lr 3e-3 |
| FSDP2 vs DDP (single source), lr 3e-4 | ≤ 1.1e-3 in the first 25 steps; 4.5% at the spike of step 27 |
| Muon: DDP vs one GPU (single source, lr 3e-3) | Maximum 3.7%; the parameters on the two GPUs are bit-identical |
| Peak memory (`max_memory_allocated`, two GPUs) | DDP 14.53 / 14.53 GiB; ddp_accum 7.66 / 7.75; FSDP2 14.33 / 14.33; Muon 15.11 / 14.98 |

How to read the table: the single-source tiny data is very unstable at lr 3e-3 (the single-GPU reference itself jumps from 6.4 back to 11.2 at step 15). A loss spike amplifies a summation-order difference of about 1e-6
to tens of percent. The single-GPU control, which changes only the accumulation, shows the same effect (21.9%). For DDP / FSDP vs one GPU, the step where the difference starts and the size of the amplification
both have the same order as in this control. Also, step 1 is the same, and the parameters on each GPU are bit-identical. The conclusion: **the numerical path of DDP and FSDP2 across GPUs is correct**.
The criterion "loss error < 1% in 50 steps" is useful only in a less chaotic setting: with multi-source data + lr 3e-4, FSDP2 vs DDP has a maximum of 0.10%.

### 14.2 FSDP2 checkpoint: multi-GPU resume and single-GPU read-back (RUNBOOK #4)

This test is also part of `dist2_check.py multi`. We kept only step 25 in the output folder of the 2-GPU FSDP2 run and resumed on 2 GPUs to step 50. **Steps 26–50 are bit-identical to the run without interruption.**
(The gather with `get_model_state_dict` / `get_optimizer_state_dict` and the distribution with `set_model_state_dict` / `set_optimizer_state_dict` both used real 2-GPU sharding.)
The single-GPU `load_policy` **strictly reads back the 91 tensors** of the checkpoint that 2-GPU FSDP2 saved. The validation CE is 6.4852 (the DDP checkpoint gives 6.4594; the two trajectories are different from the start).

### 14.3 Resume across GPUs (RUNBOOK #3)

```bash
env GPU=0,2 NPROC=2 DET=1 STEPS=100 EVERY=50 KILL_AT=75 TAG=resume_det_2gpu \
    bash runs/2026-10-01-gpu0-check/resume_check.sh --set train.grad_accum_steps=1
```

2-GPU DDP, deterministic algorithms. At step 75, the script sends `kill -9` to torchrun and the two workers, then runs the same command again: `Resumed training from .../resume_det_2gpu/kill/ckpt/step_00000050 (step 50)`.
**After the resume, steps 51–75 are bit-identical to the steps before the kill. Steps 51–100 are bit-identical to the 2-GPU run without interruption.** Step 100 val_loss: 6.224730 = 6.224730.

### 14.4 Throughput and compile (2-GPU version of RUNBOOK #2, #5, #6)

```bash
env GPU=0,2 NPROC=2 bash runs/2026-10-01-gpu0-check/train_run.sh pretrain_l20m_ddp2 runs/2026-10-01-gpu0-check/configs/l20m_tiny.toml \
    --set train.max_steps=200 --set schedule.warmup_steps=20 --set checkpoint.every=100 --set train.eval_every=100 --set logging.every=10
# compile: also add --set train.compile=true (output folder pretrain_l20m_ddp2_compile)
```

Default kernels (not deterministic). The work per GPU is the same as the single-GPU l20m in Section 2 (micro 8 × accumulation 2), so each step has 65,536 tokens:

| Run | tok/s (steady state) | Relative to one GPU | MFU per GPU (3090) | Peak memory (nvidia-smi) | loss 1 → 200 / val |
|---|---:|---:|---:|---|---|
| One GPU (Section 2) | ≈ 98,500 | 1× | 26% | 17.1 GB | 11.135 → 5.429 / 5.627 |
| 2-GPU DDP | ≈ 172,000 | **1.75× (scaling efficiency about 88%)** | 23% | GPU0 17,294 / GPU2 17,263 MiB | → 5.487 / 5.522 |
| 2-GPU DDP + `torch.compile` | ≈ 218,000 | 2.2× (about 27% faster than 2-GPU eager) | 29% | GPU2 20,523 MiB | → 5.740 / 5.848 |

The memory of the two GPUs is balanced. compile together with DDP runs, and the loss goes down normally. The difference of the final loss from eager is within the run-to-run noise of the default kernels (see convention 3 in "Read first").
Each step takes about 48 ms more than the same work on one GPU (0.381 s vs 0.333 s). Of this time, the gradient all-reduce over PCIe takes about 20 ms (the gradients of l20m are about 92 MB; estimate at 4.7 GB/s). The rest is the cost of the incomplete overlap of communication with computation, and of the GPUs that wait for each other.

### 14.5 Multi-GPU FSDP for 32K long sequences (multi-GPU version of RUNBOOK #8)

```bash
env GPU=0,2 NPROC=2 bash runs/2026-10-01-gpu0-check/train_run.sh longctx_fsdp2 runs/2026-10-01-gpu0-check/configs/longctx_tiny.toml \
    --set train.activation_checkpointing=true --set train.max_steps=3 --set train.eval_every=0 --set checkpoint.every=0 \
    --set checkpoint.resume=false --set logging.every=1
# 3 GPUs: GPU=0,1,2 NPROC=3 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True, and change the configuration to configs/longctx_tiny_1src.toml
# (the code source of the tiny corpus gives only 2 segments of 32K, which is not enough for 3 ranks)
```

| Setting (main line 689.5M, T = 32,768, micro 1, activation checkpointing) | Result | Allocated at OOM | Additional request |
|---|---|---:|---:|
| One GPU (Section 8) | OOM | 19.57 GiB | 8 GiB |
| 2-GPU FSDP2 (GPU0 + GPU2) | **OOM** | 15.84 GiB | 8 GiB (FP32 logits 32768 × 65536) |
| 3-GPU FSDP2 (GPU0 + GPU1 + GPU2) | **OOM** | 15.40 GiB | 8 GiB |

FSDP shards the parameters / gradients / optimizer states. From 1 GPU to 2 GPUs, it saves about 3.7 GiB. From 2 GPUs to 3 GPUs, it saves only 0.4 GiB more. The rest is the activations and the logits of each GPU, and FSDP cannot shard them.
**To run 32K on 24 GB GPUs, you need a chunked cross-entropy (or larger GPUs). More GPUs do not solve the problem.** By memory_calc, 32K should fit on the 80 GB H100, but this still needs a measurement.

## Summary table

| Item | Status | Evidence |
|---|---|---|
| 0a. Unit tests (CPU / GPU visible) | Passed | Section 0: CPU 321 passed / 7 skipped, GPU0 324 passed / 4 skipped |
| 0b. `zero.smoke` | Passed (but really on the CPU) | Section 0: 10 stages in 540 s; the tiny configurations hard-code the CPU |
| 0c. CUDA version of the smoke test (`smoke_cuda.py`) | Passed | Section 0: 10 stages in 64 s; all training stages on cuda:0 |
| #1 BF16 + SDPA uses Flash, `enable_gqa=True` | Passed | Section 1: the Flash kernel by default; efficient does not support GQA; the original RUNBOOK command passed |
| #2 Training path (torchrun, one GPU) | Passed (one GPU) | Section 2: loss goes down for l20m / l60m / l150m; tok/s, memory, MFU |
| NCCL (2 GPUs, PCIe) | Passed | Section 14.0: all-reduce result correct; bus bandwidth about 4.7 GB/s |
| #2 DDP across GPUs | Passed (2×3090, PCIe) | Sections 14.1, 14.4: parameters bit-identical on each GPU; the difference from one GPU has the same order as in the summation-order control; memory balanced; throughput 1.75× (scaling efficiency about 88%) |
| #3 Resume (kill -9) | Passed (one GPU) | Section 3: with deterministic algorithms, the resume is bit-identical to the run without interruption |
| #3 Resume across GPUs | Passed (2×3090) | Section 14.3: with deterministic algorithms, the resume after kill -9 is bit-identical to the 2-GPU run without interruption |
| #4 FSDP2 / DDP wrapper and checkpoint gather | Passed (single-GPU path + 2×3090) | Section 4 (single-GPU path); Sections 14.1, 14.2: 2-GPU FSDP2 vs DDP maximum 0.10% (lr 3e-4), 2-GPU resume bit-identical, strict read-back with single-GPU load_policy |
| #5 torch.compile | Passed (one GPU, 2-GPU DDP) | Section 5: loss within the run-to-run noise, throughput +37% / +41%; Section 14.4: 2-GPU DDP + compile about 27% faster than eager. Not verified together with FSDP |
| #6 MFU measurements | Passed (3090 numbers) | Section 6; the MFU on the H100 still needs a measurement |
| #7 Memory and batch | Passed | Section 7: on 24 GB, 0 without checkpointing and at most 3 with checkpointing; memory_calc decisions correct, conservative by 2.1–2.4 GiB |
| #8 32K long sequences | Failed (does not fit on 24 GB GPUs) | Section 8: on one GPU, 32K / 16K OOM even with micro 1 + checkpointing, 8K works; Section 14.5: 2-GPU and 3-GPU FSDP2 also OOM (FSDP cannot shard the logits); chunked cross-entropy is necessary |
| #9 SFT / distillation / DPO / GRPO (CUDA + BF16) | Passed (after the fix of the distillation device bug) | Sections 0, 9, 12 |
| #9 GRPO sampling time | Measured | Section 9: at main-line size, sampling takes 15–19 minutes per step (3090); batched sampling or vLLM is necessary |
| #10 HF export + transformers parity check (GPU) | Passed | Section 10: FP32 logits difference < 1e-5; greedy output identical |
| #10 vLLM | Skipped: not installed | — |
| lm-eval / bfcl | Skipped: not installed | — |
| fused AdamW | Passed | Section 11: difference from foreach / CPU 4.8e-6 |
| Activation checkpointing | Passed | Sections 7, 11: gradients bit-identical; memory / throughput measured |
| Save and restore of the CUDA random-number state | Passed | Sections 3, 11 |
| Muon CUDA BF16 NS5 | Passed (one GPU) | Section 11: about 2% difference from CPU FP32; training is normal |
| Muon DDP numerical consistency | Passed (2×3090) | Section 14.1: parameters bit-identical on the two GPUs; the difference from one GPU has the same order as in the summation-order control |
| MoE / MLA / sliding window / linear attention / MTP: CUDA forward and backward pass | Passed (after the fix of 2 bugs) | Sections 11, 12 |
| Speculative decoding / MTP self-speculation: CUDA correctness | Passed | Section 11: greedy output identical |
| GPU performance of the experimental modules above | Not verified (only reference throughput numbers) | Section 11 |
| grouped GEMM / fla / FlexAttention / varlen / FP8 | Skipped: not implemented or not installed | Section 13 |
| 8 GPUs / NVLink throughput and MFU, compile + FSDP, distributed update_bias of MoE | Not verified | Section 13 |
