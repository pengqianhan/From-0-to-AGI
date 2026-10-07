# Ladder experiment (scaling ladder) protocol — for Step 2, Stage 8 and Gate 1

**English** · [中文](README.zh.md)

> This file goes with Chapter 12 (`chapters/12-scaling-laws/`). We wrote it in Step 1: **none of the experiments below has run on a GPU yet**.
> The costs are estimates from `zero/tools/estimate_cost.py` at MFU 0.3 (the MFU of small models is usually lower than the 0.4 of the main line). Update them after the measurements of Stage 6.
> The same process on a CPU is in `code/03_lr_sweep.py` and `code/04_mini_ladder.py` of Chapter 12 (tiny-configuration demo).

## 1. Purpose

Use 4 small models to answer the three questions of Gate 1 (GOAL.md 3.4):

1. With the current recipe and a number of tokens inside the budget, **what validation loss can the main-line Base reach** (`configs/main/pretrain.toml`, 605.6M non-embedding)? Give a 95% interval.
2. About what **benchmark scores** does this loss give (the two-step method: loss → score)?
3. With which rules do the learning rate and the other hyperparameters **transfer** from the small models to the main line?

Principles (the text of Chapter 12 gives the full background):

- **Tune first, then fit**: sweep the learning rate separately for each size. Badly tuned small models distort the scaling law (Lourie et al. 2026, arXiv:2608.11859).
- **Fixed recipe**: the data mixture, the tokenizer (the same vocabulary as the main line), the sequence length, the optimizer, the schedule, and the initialization are exactly the same in the ladder and in the main line. If you change any of these items in the ladder, do the fit again. Delphi failed the first time because its recipe did not hold for long training.
- **Cover the region of the main line**: the main line is in the "overtrained" region, at about 600 tokens/parameter. The ladder must have points with tokens/parameter much larger than 20. It must not run only the Chinchilla-optimal points.
- **Held-out test**: the largest size is not in the fit. Use it only to test the extrapolation error (the method of Delphi).

## 2. Sizes and token budgets

All sizes share `configs/ladder/base.toml`: vocabulary 65,536, sequence 2,048, 262,144 tokens per step (16 × 8 GPUs × 2048), and the WSD schedule (linear decay to 0 in the last 20%).

| Configuration | Total parameters | Non-embedding | Token budget (tokens/total parameters) | Total steps (rounded to 1000) | Use |
|---|---:|---:|---|---|---|
| `l20m` | 23.1M | 6.3M | 20× / 80× / 320× | 2,000 / 7,000 / 28,000 | Fit |
| `l60m` | 71.3M | 37.8M | 20× / 80× / 320× | 5,000 / 22,000 / 87,000 | Fit |
| `l150m` | 160.5M | 110.1M | 20× / 80× / 320× | 12,000 / 49,000 / 196,000 | Fit |
| `l300m` | 318.8M | 251.7M | 20× / 80× | 24,000 / 97,000 | **Held-out test** |

Each size trains only **one trunk**. WSD branches give several budgets (Chapter 12, "several points from one training run"):

- Trunk: `max_steps = steps of the largest budget`. Train it to 80% of these steps (the end of the stable phase).
- The branch point of budget k is `0.8 × S_k`. Copy the trunk checkpoint at the branch point into a new folder, and resume with `max_steps = S_k`.
  The learning rate of the first 80% is exactly the same as in the trunk (the WSD stable phase is constant). Only the decay of the last 20% is new.
- Cost ≈ `0.8 × S_max + 0.2 × ΣS_k`, not `ΣS_k`.

Estimate (MFU 0.3, $2.5/GPU-hour, with branches): l20m ≈ $3, l60m ≈ $27, l150m ≈ $160, l300m ≈ $150. One round costs about **$340** in total.
With the learning-rate sweep (see 3) and the repeats with 2 seeds (see 5), the full ladder costs about **$450–550**. This is inside the $1,200 that GOAL.md 3.4 gives to Chapters 12–13.
**A single run of l150m or l300m costs more than $100. Before you start such a run, ask for approval as RUNBOOK says.**

## 3. Learning-rate sweep (do it first)

- On the 20× budget, sweep 5 peak learning rates for each size (each value is 2× the previous value). Center the grid on the value in the configuration file. **If the best value is at an edge of the grid, add one more value on that side**, until the best value is inside.
- l20m and l60m: 5 values each. l150m: 3 values (around the value that is extrapolated from the smaller sizes). l300m: **no sweep**. It uses the extrapolated value, and this is a test itself.
- Fit η\*(N) = c · N^(−k) (a straight line in log coordinates), and extrapolate it to the main line. Delphi also multiplied the learning rate by a correction for training length, (T₀/T)^0.3.
  Use this correction if the branch points of the 80× and 320× budgets clearly lose with the learning rate tuned on 20×. To verify this, run two more learning rates on l20m.
- Fix the other hyperparameters first: β₁=0.9, β₂=0.95, weight decay 0.1, warmup 200 steps, gradient clipping 1.0, batch 262K tokens. Joint scaling of the batch and the learning rate needs more runs (DeepSeek LLM fits B_opt and η_opt as power laws of compute). This budget does not include it. Record it honestly as a known limit.

```bash
# Example: the 20× run of l60m with lr=1e-3 (for other learning rates, change only --set optim.lr and out_dir)
uv run torchrun --standalone --nproc_per_node=8 -m zero.train.pretrain --config configs/ladder/l60m.toml \
  --set train.max_steps=5000 --set optim.lr=1e-3 --set train.out_dir=out/ladder/lr/l60m_1e-3
```

## 4. Commands for the trunk and the branches

```bash
S=l150m; LR=<sweep result>
# Trunk: train to 80% of the largest budget. Save a checkpoint every 800 steps, and keep all of them (the branches need them)
uv run torchrun --standalone --nproc_per_node=8 -m zero.train.pretrain --config configs/ladder/$S.toml \
  --set optim.lr=$LR --set train.max_steps=196000 --set checkpoint.every=800 --set checkpoint.keep_last=0 \
  --set train.out_dir=out/ladder/$S/trunk
# Branch: budget of 49,000 steps → add a decay from step 39,200
mkdir -p out/ladder/$S/b49000/ckpt
cp -r out/ladder/$S/trunk/ckpt/step_00039200 out/ladder/$S/b49000/ckpt/
echo step_00039200 > out/ladder/$S/b49000/ckpt/latest
uv run torchrun --standalone --nproc_per_node=8 -m zero.train.pretrain --config configs/ladder/$S.toml \
  --set optim.lr=$LR --set train.max_steps=49000 --set train.out_dir=out/ladder/$S/b49000
```

- With `max_steps=196000`, the trunk itself is the branch of the largest budget (it trains to the end, so it includes the decay phase).
- All step counts are rounded to multiples of 1000. Thus the branch points are multiples of 800, and they align with `checkpoint.every=800`.
- At resume, the trainer builds `LRScheduler` again with the new `max_steps`. Thus the learning rate before the branch point is the same as in the trunk. The state of the data loader comes back from the checkpoint, so the branch sees the data that the trunk sees "next".

## 5. What to record

The trainer automatically writes `log.jsonl` in each run folder (`out/ladder/...`), with `loss`, `val_loss`, `tokens`, `lr`, `grad_norm`, `tok_per_s`, and `mfu`. Also record these items by hand in `runs/<date>-ladder/README.md`:

- The **final** `val_loss` of each (size, budget): the evaluation after the end of the decay. An intermediate point cannot be the result of a different budget;
- The sweep table (size × learning rate → val_loss) and the selected η\*;
- The measured `tok_per_s` and `mfu` (to correct the main-line budget);
- At least at the 20× point of l60m, run 2 seeds, and record the difference between the seeds (Delphi: the seed difference is about 0.1%, much smaller than the extrapolation interval);
- On each checkpoint, run a few-shot evaluation on the **development sets** that are not in the preregistration (`zero.eval.harness`; do not tune with the preregistered test benchmarks). Also record the "soft metrics": the log probability of the correct option and the bits-per-byte of the reference answer;
- Record all loss spikes, divergences, and reruns honestly.

## 6. Fit and extrapolation

```bash
uv run python -m zero.tools.fit_scaling \
  $(for d in out/ladder/l20m/* out/ladder/l60m/* out/ladder/l150m/*; do echo --run $d; done) \
  $(for d in out/ladder/l300m/*; do echo --holdout $d; done) \
  --target-config configs/main/pretrain.toml --target-D 400B --target-D 500B \
  --bootstrap 500 --out runs/ladder/fit.json
uv run python -m zero.tools.plan_budget --config configs/main/pretrain.toml --mfu <measured> --fit runs/ladder/fit.json
```

Pass criteria (write them in the Gate 1 report):

- The maximum relative error of the fit points is < 1%.
- The l300m held-out points are inside the 95% interval of the extrapolation, and the error is < 1%. If they are not, **do not force a fit**. First, check if the recipe fails for long training (the first time, Delphi did not adjust the learning rate to the training length). Fix the recipe and run the ladder again.
- Write clearly the width of the interval of the extrapolation to the main line. Relative to l150m, the main line is an extrapolation of about 5.5× in N and about 8× in D, so the interval becomes much wider.

## 7. Relation to the tiny-configuration demo of Chapter 12

| Step 2 (this protocol) | CPU demo of Chapter 12 |
|---|---|
| 4 sizes × 3 budgets (WSD branches) | `code/04_mini_ladder.py`: 4 sizes × 3 budgets |
| Learning-rate sweep for each size; if the best value is at an edge, add one more value outside | `code/03_lr_sweep.py` |
| Power-law extrapolation of η\*(N) to the held-out size | Same as above; the held-out size s5 uses the extrapolated learning rate |
| `zero.tools.fit_scaling` (non-negative least squares + grouped bootstrap) | `fit_lnd` / `bootstrap` in `04_mini_ladder.py` |
| l300m held-out test | s5 held-out test (2.3× the largest ladder size) |
