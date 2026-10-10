# Post-training plan (Chapters 16–19 · Stage 10)

**English** · [中文](POSTTRAIN_PLAN.zh.md)

> Status: **plan**, 2026-10-10. The code and configs are ready and run end to end on a CPU (`uv run python -m zero.smoke`); **not run on a GPU yet**.
> This file says how post-training works, in which order, and what counts as success. The command checklist is in section 6 of [`RUNBOOK.md`](RUNBOOK.md).

## 0. Decisions already made

| Decision | Content | Reason |
|---|---|---|
| Tokenizer | The main line uses **our own** 65,536-token vocabulary | Confirmed by the project lead on 2026-10-10 |
| Pipeline | SFT → sequence-level distillation → [DPO, optional] → GRPO → **cross-stage on-policy distillation (OPD)** | See section 1 |
| OPD teachers | **Only our own checkpoints of the earlier stages** | OPD compares two distributions token by token, so it needs the same vocabulary. All open teachers have a different one |
| External teachers | **Sequence-level distillation only** (the teacher writes text → execution check → SFT data) | Independent of the vocabulary |
| RL algorithm | GRPO with verifiable rewards | Tool-call trajectories are short and their rewards are verifiable, so the group comparison works |
| PPO | Not in the main line; a "frontier observation" in Chapter 19 | A critic for long-horizon agent RL is used by one family only so far (GLM-5.2; see Chapter 19 in `references.md`). Our tasks are not long-horizon |
| Decoupling | Run post-training on an open base before our own base is ready | See tracks A and B in section 2 |

## 1. The pipeline

| # | Stage | Module | Starts from | Data | What it fixes |
|---|---|---|---|---|---|
| 1 | SFT | `zero.post.sft` | base | `data/sft/*.jsonl` (sources and licenses: Chapter 16) | The base only continues text → chat template, tool-call format, when to stop |
| 2 | Sequence-level distillation | `zero.post.distill` | SFT | Teacher trajectories that pass the execution check, mixed with the SFT data | Our own SFT data is not enough → borrow the solutions of a strong teacher |
| 3 | DPO (optional) | `zero.post.dpo` | distill | Online preference pairs | Talks but does not follow → preference alignment. **The ablation decides if it stays** (section 4) |
| 4 | GRPO | `zero.post.grpo` | DPO (or distill) | Tool-call tasks with a verifier | Imitates but cannot get it right alone → reinforce with verifiable rewards |
| 5 | Cross-stage OPD | `zero.post.opd` (**new**) | GRPO | The prompt pool of each teacher | GRPO makes general abilities worse → learn them back from the earlier checkpoints (GLM-5 §3.5) |

Default teachers of stage 5:

| Teacher | Checkpoint | Prompt pool | Weight | Ability it brings back |
|---|---|---|---:|---|
| `distill` | result of stage 2 | prompts of the SFT conversations (`data/sft/train.jsonl`) | 0.5 | General chat, instruction following (the level before GRPO) |
| `grpo` | result of stage 4 | tool-call tasks | 0.5 | Tool calls (what GRPO learned; OPD must not wash it out) |

The default OPD loss is `full_kl`: the reverse KL(p_S ‖ p_T) over the whole vocabulary at each position of the responses that the student sampled. The teachers are local, so the full distribution is cheap.
`sampled` is the GLM-5 form: the advantage A_t = log p_T(y_t) − log p_S(y_t) needs only the log probability of the sampled token. In expectation, its gradient is the gradient of the reverse KL (`tests/test_opd.py` checks this by exact enumeration).

## 2. Three tracks

| Track | Base | Configs | When | Purpose |
|---|---|---|---|---|
| **A strong proxy base** | Qwen3-0.6B-Base (same architecture as the main line: dense, GQA, QK-Norm, tied embeddings) | `configs/proxy/*.toml` | **Now** (needs only GPUs) | Make the pipeline, the code, and the hyperparameters work; same-base comparison with the official Qwen3-0.6B |
| **B weak own base** | An intermediate checkpoint of our own pretraining (same tokenizer and architecture) | `configs/weak/*.toml` | When pretraining has a checkpoint at about 100B tokens | Check that the recipe of track A still works on a weak base; find the hyperparameters that must change |
| **C main line** | `out/main/longctx` after Gate 2 | `configs/main/*.toml` | After Gate 2 | The real model; Gate 3 decides |

The three tracks use **the same code and the same recipe**. The `proxy` and `weak` configs inherit the main line with `base = "../main/<stage>.toml"` and change only the model shape, the tokenizer, and the paths (`tests/test_post_configs.py` checks this). A hyperparameter tuned on A or B changes in one place: the main config.

**Shared data**: `data/sft/*.jsonl` and the teacher data `data/distill/teacher.jsonl` are text and do not depend on the tokenizer. All three tracks share them; generate them once. The packed windows (`packed/`) and the online preference pairs depend on the tokenizer or the policy, so each track has its own.

### Limits of track A

- Qwen3-0.6B-Base was pretrained on about 36T tokens; our main line on about 400B, about 90× fewer. A result on A can fail on a weak base, so track B is required.
- Track A uses the Qwen vocabulary. Its checkpoints **cannot** be OPD teachers of the main line.
- The results of A only validate the pipeline. They are never reported as results of the main-line model.

### Which checkpoint for track B

- A checkpoint at about 100–200B pretraining tokens. Its strength is closer to the final main-line base than to Qwen.
- If it is in the stable phase of WSD (the learning rate is still high), first run a short branched decay (Chapter 15; `configs/main/midtrain.toml` with `max_steps` about 4,000 ≈ 2B tokens, about $25). A checkpoint without decay underestimates the base.
- An intermediate checkpoint has seen only 4,096-token windows, so `configs/weak/` uses `seq_len` 4,096 and keeps the RoPE base at 10,000.

## 3. What counts as success (goes into the preregistration)

### Track A: same-base comparison

> The same Qwen3-0.6B-Base: **our post-training** vs **the official post-trained Qwen3-0.6B**

- Primary endpoints: E1 (BFCL) and E2 (ACEBench, Chinese) of the preregistration. The official Qwen3-0.6B runs in both thinking and non-thinking mode; the higher score counts, per benchmark.
- Decision: paired bootstrap with `zero.eval.bootstrap`, "ahead / tie / behind" by the preregistered rule.
- **Pass**: "tie" or "ahead" on both E1 and E2.
- Each stage must contribute: a stage must not be significantly behind the previous stage on E1. If it is, find the cause before going on.
- The general group (MMLU-Redux, C-Eval, GSM8K, IFEval, ...) is reported honestly. After OPD it must not be significantly worse than after SFT.
- Write this rule into `eval/PREREGISTRATION.md` **before** track A starts.

### Track B: the recipe transfers

- Each of SFT, distillation, GRPO, OPD gives a non-negative, stable gain on E1 (paired bootstrap not behind the previous stage).
- GRPO has a learning signal: the mean `zero_std_groups` of the first 50 steps is below 0.8. Above 0.8, the tasks are too hard or too easy for the weak base; change the task difficulty first (section 6).
- Write the hyperparameters that had to change from track A into `runs/<date>-weak-*/README.md`.

### Track C: Gate 3

The Gate 3 checklist in section 6 of `RUNBOOK.md`, against all opponents.

## 4. Ablations (on A first; confirm the result once on B)

| Ablation | How | Decision |
|---|---|---|
| Keep DPO? | GRPO starts from `distill` or from `dpo`; the later stages are the same | If E1/E2 are not behind and the general group is not worse, **drop DPO** (one stage less) |
| Keep OPD? | Compare the `grpo` and `opd` checkpoints | Keep it if the general group improves and E1 is not behind |
| OPD loss | `--set opd.loss=sampled` vs the default `full_kl` | The better one on E1 + general group, at the same number of steps |
| OPD teacher weights | `distill` 0.5/0.5 vs 0.7/0.3 | Same |
| clip-higher | GRPO `clip_eps_high` 0.28 vs 0 | E1 and the stability of `resp_len` |

When the budget is tight, ablations use half the number of steps.

## 5. Starting hyperparameters (the config defaults; tune them on A)

| Stage | Learning rate | Other | Sweep first |
|---|---|---|---|
| SFT | 5e-5, cosine, warmup 100 | 2,000 steps × 524k tokens, seq 8,192 | lr {2e-5, 5e-5, 1e-4} at 25% of the steps, by `val_loss` |
| Distillation | 3e-5 | 1,500 steps; SFT data mixed in against forgetting | — |
| DPO | 5e-7 | β = 0.1, 64 pairs per step | — |
| GRPO | 1e-6, constant | G = 16, 64 prompts per step, ε 0.2 / 0.28, no KL, token mean | lr {5e-7, 1e-6, 2e-6}, 100 steps each |
| OPD | 1e-6, constant | 256 prompts per step, 1 sample each, 200 steps | — |

## 6. Prerequisites and known gaps

**Must be done before the runs (P0):**

1. **SFT data**: `data/sft/train.jsonl` and `val.jsonl`, built from the sources and licenses of Chapter 16; decontaminated against the function names and schemas of BFCL / ACEBench.
2. **Teacher**: choose the teacher of sequence-level distillation (Apache-2.0 / MIT); fill in name, version, and license in `configs/main/distill.toml`.
3. **Real RL task sets**: **this is the largest gap now.** The tool-call tasks of `grpo.py` and `opd.py` come only from `tool_env`, a toy environment with 6 mock tools. Real training needs real tool-call tasks with a verifier (AST matching for any schema). The code for "read tasks from JSONL + generic scoring" must come before the GRPO run of track A.
4. **The preregistered rule of track A** in `eval/PREREGISTRATION.md` (section 3).
5. **Stage 6, item 9**: measure the time per step of GRPO and OPD on a GPU, to fix the budget of section 7.

**Known risks:**

| Risk | Symptom | Response |
|---|---|---|
| No reward signal on the weak base | `zero_std_groups` near 1, `reward_mean` flat | Remove the tasks that the SFT model always or never solves, by pass@k (the Kimi K2 and OLMo 3 practice; online filtering is not implemented yet, so filter offline first); stronger distillation data |
| Reward hacking in GRPO | `call_rate` goes down while `reward_mean` goes up | The 8 anti-hacking rules of `tool_env.py`; the scorer of the real task sets needs the same tests |
| OPD washes out what GRPO learned | E1 goes down, `kl/grpo` goes up | Larger weight for the `grpo` teacher; fewer steps |
| The single-process code is too slow | GRPO / OPD steps take too long | Parity check against verl as the module docstring of `grpo.py` says, then use verl |
| The student starts to ramble | `eos_rate` down, `resp_len` up | Lower learning rate; check the fraction cut by `max_new_tokens` |

## 7. Budget ($1,500 in total, GOAL.md 3.4)

The GRPO and OPD numbers are **caps**. Recompute them after stage 6, item 9. A single run above $100 needs approval first; record the cost in `runs/ledger.md`.

| Item | Cap | Notes |
|---|---:|---|
| Teacher data generation (shared by the three tracks) | $150 | Fix the number of samples after the vLLM throughput measurement |
| Track A | $550 | SFT ~$15, distillation ~$13, DPO ~$20, GRPO ≤ $150, OPD ≤ $80, ablations (half steps) ≤ $200, BFCL/ACEBench evaluation ~$30 |
| Track B | $280 | Branched decay ~$25, SFT ~$8, distillation ~$7, GRPO ≤ $150, OPD ≤ $80, evaluation ~$10 |
| Track C | $450 | One full pipeline about $300 + room for one rerun |
| Reserve | $70 | |

## 8. Steps of track A

```bash
# 0. Download the base and the official post-trained model (Apache-2.0)
huggingface-cli download Qwen/Qwen3-0.6B-Base --local-dir data/hf/Qwen3-0.6B-Base
huggingface-cli download Qwen/Qwen3-0.6B --local-dir data/hf/Qwen3-0.6B
# 1. Import as a zero checkpoint, and check the model shape of the config
uv run python -m zero.tools.import_hf data/hf/Qwen3-0.6B-Base out/proxy/base --check-config configs/proxy/sft.toml
# 2. Each stage: first a run with --set train.max_steps=20, then the full run ("small first" in the RUNBOOK)
uv run torchrun --standalone --nproc_per_node=8 -m zero.post.sft --config configs/proxy/sft.toml
uv run python -m zero.post.distill --config configs/proxy/distill.toml
uv run python -m zero.post.dpo     --config configs/proxy/dpo.toml
uv run python -m zero.post.grpo    --config configs/proxy/grpo.toml
uv run python -m zero.post.opd     --config configs/proxy/opd.toml
# 3. Internal evaluation (with the official Qwen3-0.6B); the decision uses BFCL / ACEBench (section 3)
uv run python -m zero.eval.harness --config configs/proxy/eval.toml
```

Track B uses the same commands with `configs/weak/` instead of `configs/proxy/`; step 1 becomes a link from the chosen checkpoint to `out/weak/base` (see the comment at the top of `configs/weak/sft.toml`). Track C: section 6 of `RUNBOOK.md`.

Each run gets a directory `runs/<date>-<track>-<stage>/` with a copy of the config, a log summary (key lines of `log.jsonl`), and the conclusion.

## 9. Code added with this plan

| File | Content |
|---|---|
| `zero/post/opd.py` | Cross-stage on-policy distillation: several teachers with the same vocabulary, each with its own prompt pool and weight; `full_kl` / `sampled` losses; resume; KL logged per teacher |
| `zero/tools/import_hf.py` | Import an HF Qwen3 dense model (with its tokenizer) as a zero checkpoint, and check the model shape of a config |
| `configs/{tiny,main}/opd.toml` | OPD configs; `opd` added to `configs/main/eval.toml` |
| `configs/proxy/*.toml` | Track A: the full post-training configs on Qwen3-0.6B-Base |
| `configs/weak/*.toml` | Track B: the full post-training configs on an intermediate checkpoint of our own |
| `tests/test_opd.py`, `tests/test_post_configs.py` | Hand-computed / enumerated checks of the losses, end to end, refusal of a teacher with another tokenizer, HF import round trip, config checks of the three tracks |
| `zero/smoke.py` | The smoke pipeline has an OPD stage; the export now uses the OPD checkpoint |
