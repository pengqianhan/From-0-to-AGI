# Step 2 runbook (RUNBOOK)

**English** · [中文](RUNBOOK.zh.md)

> Scope: GOAL.md Section 10, "Step 2: train the main-line model". Step 1 (the current state of this repository) has no GPU. **None of the GPU commands below has run on a GPU yet.**
> `uv run pytest` and `uv run python -m zero.smoke` cover the parts that a CPU can test.
> At the end of each stage, record the cost in [`runs/ledger.md`](ledger.md).

## 0. Rules

- **Spending needs approval**: for any single run with an expected cost of more than **$100**, first send two things to the project lead: the output of `estimate_cost` and the checklist of the related section of this runbook. Start the run only after approval (GOAL.md 3.4).
- **One folder for each run**: `runs/<date>-<stage>-<short name>/`. Put these files in it: a copy of the configuration of the run (the output of `--print-config`), a log summary (the key lines of `log.jsonl`), and the conclusion. Do not put large files (checkpoints, data) into the repository.
- **Small first, then large**: for each new stage, first do a run with `--set train.max_steps=20`. Make sure that the loss, the throughput, and the GPU memory are normal. Then do the full run.
- **Price assumption**: H100 SXM at $2.5 per GPU-hour (GOAL.md 3.4), so 8 GPUs = $20 per hour. The real rental price decides. Change it with `--price`.
- **How we estimate the throughput**: all "expected throughput" values below come from `zero/tools/estimate_cost.py` with MFU 0.4 (H100 dense BF16 peak 989.5 TFLOPS, to be verified). After Stage 6 measures the MFU, calculate all budgets again with the measured value.

Expected throughput (assumption: 8×H100, MFU 0.4; before the measurement, these values are only estimates):

| Stage | Sequence length | FLOPs per token | Total throughput of 8 GPUs | Source |
|---|---:|---:|---:|---|
| Pretraining / mid-training | 4,096 | 6.955e9 | ≈ 455k token/s | `estimate_cost --config configs/main/pretrain.toml` |
| Long-context extension | 32,768 | 2.669e10 | ≈ 119k token/s | `--config configs/main/longctx.toml` |
| SFT / distillation training | 8,192 | 9.774e9 | ≈ 324k token/s (after packing, about 70% are real tokens) | `--seq-len 8192` |
| GRPO / DPO sampling | — | — | Depends on generation, so you cannot estimate it from the training FLOPs; Stage 10 measures it | — |

(Conversion: total throughput = 8 × 989.5e12 × 0.4 / FLOPs per token.)

## 1. Prepare the environment (one time on each new machine)

```bash
git clone <this repository> && cd From-0-to-AGI
uv sync --group dev                       # torch, tokenizers, safetensors, transformers, pytest
uv run python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.device_count())"
nvidia-smi topo -m                        # make sure that NVLink connects all 8 GPUs to each other
uv run pytest -q                          # all CPU tests must pass first (about 3 minutes)
uv run python -m zero.smoke --out /tmp/smoke   # end-to-end smoke test (one CPU thread: about 9 minutes on an idle CPU, longer on a busy CPU)
```

- torch must be a version with CUDA (`uv pip install torch --index-url https://download.pytorch.org/whl/cu12x`, matched to the driver).
- Put the data on a local NVMe disk: `data/tokenizer/`, `data/pretrain/`, `data/midtrain/`, `data/sft/`, and others (the paths are in `configs/main/*.toml`).
- Evaluation: `uv pip install lm-eval==<frozen version> bfcl-eval==<frozen version> vllm==<frozen version>` (write the version numbers into `eval/PREREGISTRATION.md`).
- GGUF export: `uv run python -m zero.export.gguf ...` automatically makes a shallow clone of llama.cpp and compiles it (it needs `cmake`; `apt-get install -y cmake`).

## 2. Stage 6: GPU verification run (≤ $50)

Purpose: verify, item by item, the paths that are "not verified on a GPU yet", and measure the MFU. Budget: 8×H100 for about 1.5 hours ≈ $30, at most $50.

> 2026-10: we already did the one-GPU / 2-GPU RTX 3090 version of this section ([`2026-10-01-gpu0-check/`](2026-10-01-gpu0-check/README.md)). The paths, the agreement across GPUs, and the resume all passed. 32K does not fit on a 24 GB GPU. We must still run each item on 8×H100. The focus is the throughput and the MFU with NVLink, 8-GPU FSDP, and the GPU memory at 32K.

| # | Verification item | Command | Pass criterion |
|---|---|---|---|
| 1 | One GPU, BF16 + SDPA uses FlashAttention | `uv run python -c "import torch; from torch.nn.attention import sdpa_kernel, SDPBackend; from zero.config import load_model_config; from zero.model import Transformer; m=Transformer(load_model_config('configs/main/pretrain.toml')).cuda().bfloat16(); x=torch.randint(0,65536,(1,4096),device='cuda');\nwith sdpa_kernel(SDPBackend.FLASH_ATTENTION): print(m(x).shape)"` | No "No available kernel" error. **Verified** (RTX 3090, 2026-10, see Section 1 of [`2026-10-01-gpu0-check/`](2026-10-01-gpu0-check/README.md)): in BF16, `enable_gqa=True` uses the Flash backend by default. The memory-efficient backend **does not support GQA**, so "fall back to efficient" is not possible. If Flash is not available on the H100, there is only one way: in `Attention`, first `repeat_interleave` K/V, then use the efficient backend. Record the MFU difference |
| 2 | Multi-GPU DDP | `uv run torchrun --standalone --nproc_per_node=8 -m zero.train.pretrain --config configs/main/pretrain.toml --set train.max_steps=200 --set checkpoint.every=100 --set train.out_dir=out/gpu_check/ddp` | The loss decreases; the GPU memory is balanced across the GPUs; `tok/s` and MFU in the log are stable |
| 3 | Resume from a checkpoint | `kill` the previous command at about step 150, then run the same command again | Training resumes from step 100. The loss of steps 101–200 agrees with a control run without interruption (BF16 allows differences of the order of 1e-3; the data order must be exactly the same) |
| 4 | FSDP2 | Same as 2. Add `--set train.parallel=fsdp`, and use a different folder | The loss curve of the first 50 steps agrees with DDP (error < 1%); `load_policy` on one GPU can read the checkpoint back |
| 5 | torch.compile | Same as 2. Run 100 steps with the default `compile = true`, and 100 steps with `--set train.compile=false` | The two runs give the same loss; record the throughput gain from compile |
| 6 | Measured MFU | Use the log of steps 100–200 of item 2 | Record `tok/s` and `mfu`. Run `estimate_cost --mfu <measured>` again with the measured MFU, and update this runbook and the budget |
| 7 | GPU memory and batch | Try `--set train.micro_batch_size=16 --set train.grad_accum_steps=1` | Find the largest micro batch without OOM; keep the tokens per step = 524,288 |
| 8 | Long sequences | `torchrun ... -m zero.train.midtrain --config configs/main/longctx.toml --set train.max_steps=20` (init_from points to the checkpoint of item 2; as a temporary setting, you can use `--set train.init_from=out/gpu_check/ddp/ckpt`) | 32K sequences do not cause OOM (FSDP); the throughput has the same order of magnitude as the estimate |
| 9 | Post-training paths | `torchrun ... -m zero.post.sft --config configs/main/sft.toml --set train.max_steps=20 --set train.init_from=out/gpu_check/ddp/ckpt` (needs a small SFT JSONL file); `python -m zero.post.grpo --config configs/main/grpo.toml --set train.max_steps=2 --set grpo.prompts_per_step=4` | No errors; GRPO records the sampling time of each step, to estimate the budget of Stage 10 |
| 10 | Export | `load_policy` + `export_to_hf_qwen3(..., chat=True)`, then `vllm serve out/gpu_check/hf --enable-auto-tool-choice --tool-call-parser hermes`. Send one request with tools | vLLM can load the model; the response has structured `tool_calls` (bad answers from the model are not a problem; the check is on the format path) |

After the verification: write the measured MFU, throughput, and GPU memory into `runs/<date>-gpu-check/README.md`. Update the throughput table at the top of this runbook, and record the cost.

## 3. Stage 7: opponent reruns and the final preregistration (about $200)

1. Use the rules of `eval/opponents.md` to fix the opponent list and the **freeze date**.
2. Fix the versions of the evaluation frameworks (lm-evaluation-harness, BFCL, ACEBench, and others). Write them into `eval/PREREGISTRATION.md`.
3. For each opponent: use the official template, both modes (thinking / non-thinking, if the model has them), and the same decoding parameters. Run all benchmarks, and save the per-question results.
   - General benchmarks: `lm_eval --model vllm --model_args pretrained=<opponent>,dtype=bfloat16 --tasks <frozen tasks> --batch_size auto --log_samples --output_path eval/results/<opponent>`
   - BFCL: for an opponent, use the handler name that BFCL has built in: `bfcl generate --model <handler name> --test-category <frozen categories> --backend vllm`, then `bfcl evaluate`. For our model, use `python -m zero.eval.bfcl` (see `zero/eval/bfcl.py`; not verified yet).
4. Estimate: about 25 models × ~1 GPU·h each ≈ $60–200. If the cost is more than $100, get approval first.
5. **Stop and wait for the confirmation of the preregistration.** After the confirmation, commit `eval/PREREGISTRATION.md` (the commit time is the registration time).

## 4. Stage 8: ladder experiments and Gate 1 (about $1,200)

```bash
for s in l20m l60m l150m l300m; do
  uv run torchrun --standalone --nproc_per_node=8 -m zero.train.pretrain --config configs/ladder/$s.toml
done
```

| Configuration | Parameters | Tokens | Estimated GPU-hours | Estimated cost |
|---|---:|---:|---:|---:|
| l20m | 23.1M | 0.46B | 0.1 | $0.3 |
| l60m | 71.3M | 1.43B | 0.6 | $1.5 |
| l150m | 160.5M | 3.21B | 2.8 | $7 |
| l300m | 318.8M | 6.38B | 10.8 | $27 |

(`estimate_cost --config configs/ladder/<s>.toml --tokens <token>`, MFU 0.4. The real MFU of small models is usually lower. Correct the values with the measurements of Stage 6.)
One ladder round is cheap. Most of the budget goes to the learning-rate sweeps, the data mixture ablations (Chapter 13), and recipe validation (b) (one full post-training run on an open Base model of the same size).

Metrics to watch: the final val loss of each size (to fit L(N, D)), whether the loss curves are smooth, whether loss spikes occur, and tok/s.

**Gate 1 checklist** (complete all items before you ask for the pretraining budget):

- [ ] The 4 ladder sizes are complete. The fit residual of val loss against (N, D) is < 1%. Extrapolate the loss of the main-line Base at about 400B tokens (with a confidence interval).
- [ ] Extrapolate the general benchmark scores (use the "loss → score" relation of the ladder models). Do not extrapolate benchmarks where the small models are near random; say so honestly.
- [ ] Recipe validation (a): apply the post-training recipe to 2–3 ladder Base models. Get the relation "Base quality → tool-calling score", and extrapolate it to the main line.
- [ ] Recipe validation (b): apply the post-training recipe to an existing open Base model of the same size, to see the upper limit of the recipe (validation only, do not publish).
- [ ] Do the predictions reach the hard goal (the criterion in the preregistration)? Write the answer as the Gate 1 report.
- [ ] Submit the pretraining budget (`estimate_cost --config configs/main/pretrain.toml --tokens 400B --mfu <measured>`) together with the report.
- [ ] **Wait for approval.**

## 5. Stage 9: pretraining, mid-training, long context, and Gate 2 (about $5,700)

```bash
# Pretraining (about 400B tokens, 762,940 steps; estimate 1,952 GPU·h ≈ $4.9K @ MFU 0.4. Calculate again with the measured MFU, then ask for approval)
uv run torchrun --standalone --nproc_per_node=8 -m zero.train.pretrain --config configs/main/pretrain.toml
# Mid-training / annealing (≈ 26B tokens, estimate 127 GPU·h ≈ $317)
uv run torchrun --standalone --nproc_per_node=8 -m zero.train.midtrain --config configs/main/midtrain.toml
# Long-context extension to 32K (≈ 4.2B tokens, estimate 79 GPU·h ≈ $197)
uv run torchrun --standalone --nproc_per_node=8 -m zero.train.midtrain --config configs/main/longctx.toml
```

> **Note:** the first estimate, for 500B tokens, was about $6.1K (MFU 0.4). This is more than the ~$5K that GOAL.md 3.4 gives to pretraining. Thus Chapter 12 decreased the default to about 400B tokens (`max_steps = 762940`, about $4.9K at MFU 0.4). Gate 1 must still fix the final balance of three values: the token count, the MFU, and the rental price. Use the ladder experiments and the measured MFU for this decision.

Metrics to watch:

- `loss` / `val_loss`: they decrease smoothly. Compare them with the extrapolated ladder curve.
- `grad_norm`: it stays stable near 0.2–1. A sudden increase of more than 10 times is an early sign of a spike.
- `tok_per_s`, `mfu`: a drop in speed means that a GPU is slow (check `nvidia-smi` and the NCCL logs).
- `mixture_counts`: the real sampling ratio of each source agrees with the configuration.
- Checkpoints: one each 2,000 steps, `keep_last = 3`. Each day, take one checkpoint and do a few-shot evaluation (the multiple-choice items of `zero.eval.harness` + a small subset of lm-eval).

Failure handling: if the loss becomes NaN or diverges, the training loop stops automatically (`FloatingPointError`). Resume from the last checkpoint. If necessary, skip the bad data segment (change the `data` seed or remove the shard), and decrease the learning rate. If the machine is preempted, run the same command again. Training then resumes automatically.

**Gate 2 checklist**:

- [ ] The real scores of the Base model on the preregistered Base benchmarks vs. the prediction of Gate 1 (list the deviation of each item).
- [ ] If the scores are clearly lower, diagnose first (data, learning rate, evaluation template). Write a diagnosis report, then decide whether to start post-training.
- [ ] Decontamination check: the 13-gram overlap of the training data with all evaluation sets (`zero/data/decontam.py`). Write the result into the draft model card.
- [ ] Record the cost.

## 6. Stage 10: post-training, Gate 3, and release (about $1,500 + $200)

The plan, the proxy track that runs before our base is ready, and the decision rules are in [`runs/POSTTRAIN_PLAN.md`](POSTTRAIN_PLAN.md). The commands below are the main line (track C).

```bash
# SFT (≈ 1B window tokens, estimate 6.9 GPU·h ≈ $17)
uv run torchrun --standalone --nproc_per_node=8 -m zero.post.sft --config configs/main/sft.toml
# Distillation: first start the teacher server (after you verify the license, fill in name / version / license / license_allows_distillation in the configuration)
vllm serve <teacher> --served-model-name teacher --enable-auto-tool-choice --tool-call-parser hermes &
uv run python -m zero.post.distill --config configs/main/distill.toml
# DPO (single-process implementation)
uv run python -m zero.post.dpo --config configs/main/dpo.toml
# GRPO (single-process implementation; if the throughput is too low, first do a parity check against verl, then use verl; see the module docstring of zero/post/grpo.py)
uv run python -m zero.post.grpo --config configs/main/grpo.toml
# Cross-stage on-policy distillation (last stage; teachers = our own distill + GRPO checkpoints, same tokenizer)
uv run python -m zero.post.opd --config configs/main/opd.toml
# Internal evaluation (tool_env dev + toy sets, paired bootstrap)
uv run python -m zero.eval.harness --config configs/main/eval.toml
# Export (the test prompt 你好 in --run means "hello"; it is data, so keep it)
uv run python -c "from zero.post.common import load_policy; from zero.hf import export_to_hf_qwen3; m,t=load_policy('out/main/opd/ckpt'); export_to_hf_qwen3(m,None,'out/main/hf_final',tokenizer=t,chat=True)"
uv run python -m zero.export.gguf --hf-dir out/main/hf_final --out out/main/zero-f16.gguf --quantize Q4_K_M --run "<|im_start|>user\n你好<|im_end|>\n<|im_start|>assistant\n"
```

Budget notes: the cost of teacher data generation depends on the size of the teacher and the number of samples (estimate it after you measure the vLLM throughput). The cost of GRPO is mostly sampling. Stage 6, item 9 measures the time of each step. Then estimate the cost as "seconds per step × steps × $20/hour". If it is more than $100, ask for approval.

Metrics to watch:

- SFT / distillation: `loss` and `val_loss` on the assistant tokens only; the pass rate of the execution check on the distillation data (`teacher.jsonl.meta.json`).
- DPO: `loss` decreases from 0.693; `acc` (the fraction of pairs where the implicit reward of chosen > rejected); `margin`. If `chosen_reward` also decreases, the training "pushes both down together": be careful.
- GRPO: `reward_mean`, `format_rate`, **`call_rate`** (in the smoke test, a hack occurred where the model "stops calling tools"; see item 8 in `tool_env.py`), `resp_len`, `kl`, `clip_frac`, `zero_std_groups` (a high value means that the tasks are too difficult or too easy).
- OPD: `kl` and `kl/<teacher>` go down; `eos_rate` stays near 1 (if it drops, the student rambles); the tool-call score of the GRPO stage must not drop.
- At the end of each stage, run `zero.eval.harness` and a BFCL subset. If any item clearly degrades, go back to the checkpoint of the previous stage.

**Gate 3 checklist** (before the release):

- [ ] Run all evaluations with the benchmarks, framework versions, templates, and decoding parameters that `eval/PREREGISTRATION.md` froze. Save the per-question results.
- [ ] Do a paired bootstrap against each opponent (`zero.eval.bootstrap`; for the opponent, use the higher of thinking / non-thinking). Use the preregistered criteria to give "ahead / tie / behind".
- [ ] If the model does not reach the hard goal, do not claim "ahead". Write an honest gap analysis.
- [ ] Decontamination: the n-gram overlap of the training data (teacher synthetic data included) with the evaluation sets, and the overlap check of tool function names / schemas with BFCL.
- [ ] Model card: data and licenses, the recipe and cost of each stage, the preregistration, all evaluation results (the items where we are behind included), the decontamination results, and the known limitations.
- [ ] The GGUF (Q4_K_M) runs with llama.cpp on a laptop. Record the screen during the local demo (`python -m zero.demo.cli --model out/main/hf_final`).
- [ ] After the release, look again for new models released after the freeze date. Write "Opponents added after release".
- [ ] **Wait for the final confirmation before the release.**

## 7. Budget comparison (GOAL.md 3.4)

| Use | GOAL budget | Current estimate (MFU 0.4) | Notes |
|---|---:|---:|---|
| Stage 6 GPU verification | (inside the reserve) | ≤ $50 | |
| Chapter 11 opponent reruns | ~$200 | $60–200 | Depends on the number of opponents |
| Chapters 12–13 ladder, ablations, recipe validation | ~$1,200 | Ladder itself < $40; most of the cost is ablations | |
| Chapter 14 pretraining | ~$5,000 | ~$4,900 (400B tokens, MFU 0.4) | Fix it after the MFU measurement |
| Chapter 15 mid-training + long context | ~$700 | ~$514 | |
| Chapters 16–19 post-training | ~$1,500 | SFT ~$17 + distillation/DPO/GRPO to be measured | |
| Chapter 20 final evaluation and release | ~$200 | — | |
| Part 5 architecture experiments | ~$400 | — | Optional |
