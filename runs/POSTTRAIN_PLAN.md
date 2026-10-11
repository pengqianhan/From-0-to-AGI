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

1. **SFT data**: the pipeline code is done (2026-10-10, `zero/post/sft_data.py`, see section 6.2). What is left: verify the license of each source, download the data, export the evaluation questions for decontamination, and tune the mixture by token share.
2. **Teacher**: the code works on real data now (section 6.4); **you choose the teacher** from the verified candidates of section 6.4 and fill in name, version, and license in `configs/main/distill.toml`.
3. **Real RL task sets**: the code is done (2026-10-10, `zero/post/envs/fc_tasks.py`, see section 6.1). What is left: download the data, build the task files, and spot-check the scores locally.
4. **The preregistered rule of track A**: written as a candidate clause in section 6 of `eval/PREREGISTRATION.md` (draft; you confirm it at the freeze). The evaluation pipeline is in section 6.3.
5. **Launch check** on the first GPU hour (`zero.tools.launch_check`, section 6.4): the seconds per step of every stage, to fix the budget of section 7.

### 6.1 Real function-calling tasks (`zero/post/envs/fc_tasks.py`)

**Task format**: one JSON object per line. `tools` (OpenAI function format), `messages` (the prompt, history allowed), `gold_calls` (the calls of the next assistant turn, in any order; an empty list means "no call is correct"). Each argument can have `alternatives` (all accepted values; `""` means it may be left out), the same as the BFCL "possible answer" format. Only the first assistant turn is scored, as in the GRPO of `tool_env`.

**Reward**: the same scale and the same anti-hacking rules as `tool_env.score_tool_calls` (format error −1; a forged tool result, call JSON outside the tags, too long, too many calls are format errors; no call when one is needed 0; a call when none is correct −0.5; correctly no call 0.5; else 0.1 + 0.9 × fraction matched − 0.25 × extra calls − 0.5 × calls that break the schema). New: a **generic schema check** (tool not offered, missing required argument, argument not in the schema, wrong type, value outside the enum) and a **generic argument comparison** (strings stripped and case-insensitive; numbers by value; a left-out argument whose gold value is its default counts as the same call).

**Candidate sources** (licenses checked on the data cards on 2026-10-10; check again before use):

| Source | License | Languages | Converter | Notes |
|---|---|---|---|---|
| NousResearch/hermes-function-calling-v1 | Apache-2.0 | en | `hermes` | ShareGPT + `<tool_call>`; single-turn and multi-turn |
| Team-ACE/ToolACE | Apache-2.0 | en, zh | `toolace` | Python call syntax `[Func(a=1)]`; has turns where a missing parameter means "do not call"; **the main source of Chinese tool-call data** |
| glaiveai/glaive-function-calling-v2 | Apache-2.0 | en | convert to `openai` first | Hermes already has a cleaned 5k subset |
| Salesforce/xlam-function-calling-60k | to be verified (gated) | en | `xlam` | Gated; the data card cannot be read here |

**Build steps**:

```bash
# 1. Convert + validate + deduplicate + decontaminate against BFCL (drop on a shared tool name or 13-gram); keep 500 for dev
uv run python -m zero.post.envs.fc_tasks build \
    --src hermes:data/raw/hermes-function-calling-v1/func-calling-singleturn.json \
    --src hermes:data/raw/hermes-function-calling-v1/func-calling.json \
    --src toolace:data/raw/ToolACE/data.json \
    --exclude-bfcl data/eval/bfcl --exclude-tasks data/eval/acebench_zh.jsonl \
    --out data/rl/fc_all.jsonl --dev-out data/rl/fc_dev.jsonl --dev-size 500
# 2. Split into SFT conversations and RL tasks that do not overlap (a task that SFT showed with its answer is all-correct in an RL group: no signal)
uv run python -m zero.post.envs.fc_tasks split data/rl/fc_all.jsonl \
    --sft-out data/sft/fc_sft.jsonl --rl-out data/rl/fc_train.jsonl --sft-frac 0.5
# 3. Spot-check: pick some tasks, write correct and wrong outputs by hand, and check that the scores make sense
uv run python -m zero.post.envs.fc_tasks score data/rl/fc_dev.jsonl 0 '<tool_call>{...}</tool_call>'
```

`<out>.meta.json` records the count and license of each source, the count of each drop reason, and the counts of "no call" and parallel-call tasks.

**Connected**: `task_files` in `configs/main/grpo.toml`, the prompt pool of the `grpo` OPD teacher in all three tracks, and `fc_tasks` (the dev set, paired bootstrap) in the eval configs of all three tracks. The tiny configs still use `tool_env` (smoke test).

**Not done yet**:
- A converter from ACEBench to this format (now `--exclude-tasks` only reads a file that is already converted).
- Multi-step tasks (they need executable tools).
- The online DPO pairs and the teacher tasks of distillation still come from `tool_env`.
- Decontamination checks only 13-grams of the user text and tool names; near-duplicate function schemas are not checked yet.

### 6.2 SFT data pipeline (`zero/post/sft_data.py`)

One config (`configs/main/sft_data.toml`) lists all sources; one command writes `data/sft/train.jsonl`, `val.jsonl`, and `meta.json`:

```bash
uv run python -m zero.post.sft_data --config configs/main/sft_data.toml
```

**Formats**: `messages` (Tülu 3, SmolTalk, SmolTalk2, including SmolTalk2's `xml_tools` and `custom_instructions`), `sharegpt`, `alpaca`, `sft` (this project's format, for example the output of `fc_tasks split`), `tool_env` (our own synthetic trajectories). `<tool_call>` text in an assistant turn becomes structured `tool_calls`; a row whose call does not parse is dropped.

**License per row**: each row of Tülu 3 and SmolTalk2 carries its upstream subset name (the `source` field). `license_by_source` gives each subset its own license; `drop_sources` removes subsets. NC licenses are always dropped; "待核实" (to be verified) is dropped by default, with a warning when a whole source is dropped.

**Cleaning and filters** (each counted per source in `meta.json`): special-token injection, empty answers, tool calls that do not parse, `<think>` blocks (the main line does not think by default: the thinking part is removed, and a row with only thinking is dropped), language filter (`langs`), too long (tokens of the main-line tokenizer), exact duplicates, near duplicates (MinHash of the first user message), 13-gram decontamination (evaluation questions), and BFCL tool-name decontamination.

**Mixing**: each source is filtered, shuffled, and cut to `max_rows`; then the validation set is taken from the pool (before repetition, so no validation row is trained on); then `repeat` upsamples. `meta.json` reports rows, total tokens, and assistant tokens per source and language. **Judge the mixture by tokens, not rows.** A starting target: tool calls about 25%, Chinese about 30%, general English the rest.

**Sources in the config** (licenses read from the data cards on 2026-10-10):

| Source | License | Status |
|---|---|---|
| `fc-tool-calls` (Hermes + ToolACE, output of `fc_tasks split`) | Apache-2.0 (on each row) | Usable |
| `tool-env` (our own synthetic data) | this project | Usable |
| 6 English no_think subsets of SmolTalk2 | no license metadata on the card; partly generated by Qwen | **To be verified** |
| The Chinese part of SmolTalk2's multilingual subset | same | **To be verified** |
| smoltalk-chinese (OpenCSG) | metadata says Apache-2.0, but the card says commercial use needs permission by email | **To be verified**: email for commercial permission first |
| COIG-CQIA | no license on the card; content from Zhihu, Douban, and other sites | Not used |
| Infinity-Instruct (BAAI) | gated; the card cannot be read here | For you to check |

**Chinese general instruction data is the largest gap of this mixture.** Chinese data with a clean license is scarce. Three ways:
1. Email OpenCSG for commercial permission for smoltalk-chinese.
2. Verify the upstream terms of SmolTalk2's multilingual subset (the Qwen-generated part).
3. Generate it ourselves: an open teacher with a permissive license (Apache-2.0 / MIT) answers Chinese prompts, through the sequence-level distillation of Chapter 17. This is the cleanest way, but it costs teacher inference.

### 6.3 Evaluation pipeline (2026-10-10)

Everything that can run on a CPU is done and checked on real data; only the generation steps that need a GPU and vLLM remain.

**BFCL (E1)**: the three "to be verified" items of `zero/eval/bfcl.py` were checked against the source of `bfcl-eval` 2026.3.23.
- The registry names and the `ModelConfig` fields: they match the source; a model registered at run time is seen by the CLI.
- The structure of `function`: BFCL passes the raw function descriptions with Python type names (`dict`, `float`). Our handler converts them to JSON-schema names, as in our training data (now in section 4 of the preregistration).
- Per-item results: BFCL's score file lists **only the failed items**. Per-item correctness = all ids of the result file − the failed ids of the score file. New: `per_item_results`, `collect`, `compare` (stratified paired bootstrap). The old loose CSV parsing is gone.

**The scorer checked on the real evaluation data**: with each item's gold answer as the output, every item must score exact.
- BFCL v4, 3,641 single-turn items: the first scorer disagreed on 264. Four causes, all fixed: BFCL's function docs contradict their own answers (for example `year` typed integer with the answer `"dontcare"`), `null` for optional arguments, nested possible answers, and more than 5 parallel calls. Now 0 disagree.
- ACEBench, 967 Chinese and 973 English items (Normal + Special, without Agent): 0 disagree after the output-length limit was raised. ACEBench itself has one item with a duplicated tool (`normal_atom_number_48`).
- Each of these cases is a regression test.

**Exports**:
- `fc_tasks export bfcl|acebench`: the evaluation sets as task files, **only for decontamination** and the scorer check; never to pick checkpoints (section 8 of the preregistration). The tool-call dev set is `data/rl/fc_dev.jsonl`, held out from the training sources.
- `zero.eval.export_prompts`: the questions of the general benchmarks (MMLU-Redux, MMLU-Pro, C-Eval, CMMLU, GSM8K, MATH-500, HumanEval+, MBPP+, IFEval) for the 13-gram decontamination of the SFT data. It needs `datasets` and access to Hugging Face; tested here only with fake data. Check the data set ids and splits when the preregistration is frozen.

**Still to do on a GPU**: run BFCL generate / evaluate with vLLM; check that `apply_chat_template` of the exported tokenizer gives the training text inside BFCL; adapt ACEBench's official scripts to a current vLLM.

### 6.4 Teacher data, multi-GPU, difficulty filter, launch check (2026-10-10)

**Teacher data on real data** (`zero/post/distill.py`): besides the toy environment, `task_files` (real function-calling tasks: the teacher's first turns that score exact are kept) and `prompt_files` (prompts without answers, for example Chinese instructions: empty answers, forged turns, tool calls, and answers in another language are dropped; `prompt_license` is written to each row). `concurrency` sends parallel requests to the teacher server. **This is the clean way to fill the Chinese gap of section 6.2**: a teacher whose license allows it answers Chinese prompts. The prompts need a license too.

**Teacher candidates** (license read from the model metadata on 2026-10-10; **the choice is yours**):

| Model | License | Notes |
|---|---|---|
| Qwen3.5-35B-A3B | Apache-2.0 | 3B active: cheap to serve on one GPU (FP8) |
| Qwen3.5-122B-A10B | Apache-2.0 | stronger; hosted by inference providers |
| Qwen3-235B-A22B-Instruct-2507 | Apache-2.0 | non-thinking instruct model |
| gpt-oss-120b | Apache-2.0 | OpenAI also has a usage policy: read it before use |
| DeepSeek-V4.1-Flash | MIT | 763B total; hosted only, in practice |
| GLM-5.2 | MIT | Chinese and English; strong tool calls (MCP-Atlas, Tool-Decathlon) |

When a hosted provider serves the teacher, also read the provider's terms of service.

**Multi-GPU** (`zero/post/common.py`): DPO, GRPO, and OPD are data parallel with torchrun; the batch sizes in the configs are global, so the recipe does not change with the number of GPUs. 2 CPU processes give the same per-step losses and final weights as 1 process (`tests/test_post_ddp.py`). Distillation runs in two commands: the teacher data in one process (`--generate-only`; under torchrun the other ranks would wait for hours in a collective), then the student trains with the SFT `Trainer` (DDP). DPO pairs (`generate_pairs`) and the difficulty filter split their sampling over the ranks. **A DPO bug was fixed on the way**: every micro-step of a step used the same `micro_batch_size` pairs, so most of each step's pairs were never trained on.

**Difficulty filter** (`zero/post/difficulty.py`): with the checkpoint that starts RL, k samples per task; keep the tasks with 1/k ≤ pass ≤ (k−1)/k (optionally a fraction of the all-wrong ones for a weak base). A task whose prompt leaves less than `max_new_tokens` of the context is dropped here and in GRPO (`too_long`; it matters for the 4096-token weak track). Per track, because difficulty depends on the policy: `data/rl/<track>/fc_train_filtered.jsonl`.

**Launch check** (`zero/tools/launch_check.py`): the first hour on rented GPUs. A few steps of every stage of a track (chained in a launch directory), then the median seconds per step, the peak memory, and the projected hours, GPU-hours, and dollars of each full stage; the teacher throughput from a small sample. It checks the inputs before it starts. It replaces "stage 6, item 9" for post-training.

**Known risks:**

| Risk | Symptom | Response |
|---|---|---|
| No reward signal on the weak base | `zero_std_groups` near 1, `reward_mean` flat | `zero/post/difficulty.py`: remove the tasks that the starting checkpoint always or never solves, by pass@k (the Kimi K2 and OLMo 3 practice; online filtering is not implemented); stronger distillation data |
| Reward hacking in GRPO | `call_rate` goes down while `reward_mean` goes up | The 8 anti-hacking rules of `tool_env.py`; the scorer of the real task sets needs the same tests |
| OPD washes out what GRPO learned | E1 goes down, `kl/grpo` goes up | Larger weight for the `grpo` teacher; fewer steps |
| Sampling is too slow | GRPO / OPD steps take too long even with all GPUs (data parallel, but no continuous batching) | Parity check against verl as the module docstring of `grpo.py` says, then use verl |
| The student starts to ramble | `eos_rate` down, `resp_len` up | Lower learning rate; check the fraction cut by `max_new_tokens` |

## 7. Budget ($1,500 in total, GOAL.md 3.4)

The GRPO and OPD numbers are **caps**. Recompute them with the launch check. A single run above $100 needs approval first; record the cost in `runs/ledger.md`.

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
# 2. Launch check: a few steps of every stage → s/step, memory, projected cost (runs/<date>-proxy-launch/)
uv run python -m zero.tools.launch_check --track proxy --nproc 8 --price 2.5
# 3. The full runs (all data parallel; batch sizes in the configs are global)
uv run torchrun --standalone --nproc_per_node=8 -m zero.post.sft     --config configs/proxy/sft.toml
uv run python -m zero.post.distill --config configs/proxy/distill.toml --generate-only   # teacher data, one process
uv run torchrun --standalone --nproc_per_node=8 -m zero.post.distill --config configs/proxy/distill.toml
uv run torchrun --standalone --nproc_per_node=8 -m zero.post.dpo     --config configs/proxy/dpo.toml
uv run torchrun --standalone --nproc_per_node=8 -m zero.post.difficulty --policy out/proxy/dpo/ckpt \
    --tasks data/rl/fc_train.jsonl --k 8 --out data/rl/proxy/fc_train_filtered.jsonl
uv run torchrun --standalone --nproc_per_node=8 -m zero.post.grpo    --config configs/proxy/grpo.toml
uv run torchrun --standalone --nproc_per_node=8 -m zero.post.opd     --config configs/proxy/opd.toml
# 4. Internal evaluation (with the official Qwen3-0.6B); the decision uses BFCL / ACEBench (section 3)
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
| `zero/post/common.py` (data parallel), `grpo.py` / `opd.py` / `dpo.py` / `distill.py` (section 6.4) | DPO, GRPO, OPD, and distillation on N GPUs; DPO batch fix; teacher data on real tasks and prompts |
| `zero/post/difficulty.py`, `zero/tools/launch_check.py` (section 6.4) | Offline difficulty filter of RL tasks; launch check with cost projection |
| `zero/eval/bfcl.py`, `zero/eval/export_prompts.py`, ACEBench converter and `export` in `fc_tasks` (section 6.3) | Evaluation pipeline: BFCL adapter fixed against the source, per-item results and paired bootstrap, scorer checked on all of BFCL and ACEBench, export of evaluation questions (decontamination) |
| `zero/post/sft_data.py`, `configs/main/sft_data.toml` (section 6.2) | SFT data pipeline: 5 source formats, license per row, cleaning, deduplication, decontamination, mixture by tokens |
| `zero/post/envs/fc_tasks.py` (section 6.1) | Real function-calling tasks: format, generic schema check and reward, converters for Hermes / ToolACE / xLAM / OpenAI / BFCL, decontamination, build and SFT/RL split; connected to GRPO (`task_files`) and evaluation (`fc_tasks`) |
