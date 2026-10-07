# Gate 1 report (template) — before the large spending: extrapolated prediction + recipe validation

**English** · [中文](gate1_report_template.zh.md)

> Based on GOAL.md 3.4 "Gate 1" and Chapter 12. At the end of Step 2, Stage 8, copy this file to `runs/<date>-gate1/README.md` and fill it in.
> Give it to the project owner together with `runs/ladder/fit.json` and the budget table. **Do not start the main-line pretraining before the approval.**
> Rules for filling in: write the source of each number (run folder / command / commit hash). For work that you did not run, write "not done". If a result does not agree with the expectation, write it honestly. Do not delete points, and do not pick points.

## 0. Summary (write it last, at most one screen)

- Main-line configuration: `configs/main/pretrain.toml` (commit ____), total parameters ____M, non-embedding ____M
- Planned tokens: ____B (____ tokens/parameter), expected cost $____ (measured MFU ____, price $____/GPU-hour)
- Extrapolated Base validation loss: ____ (95% interval ____–____)
- Extrapolated general benchmarks (Base, few-shot): see the table in Section 3; tool calling (after post-training): see Section 4
- **Conclusion: the prediction CAN / CANNOT reach the preregistered hard goal**; recommendation: start pretraining / adjust the recipe ____ / adjust the goal ____

## 1. Recipe and versions

| Item | Value | Same as the main line? |
|---|---|---|
| Tokenizer (hash of `data/tokenizer/tokenizer.json`) | | Must be the same |
| Data mixture (conclusion of Chapter 13) | | Must be the same |
| Sequence length / tokens per step | | Ladder 2048 / 262K; main line 4096 / 524K (write the effect of the difference) |
| Optimizer (AdamW / Muon), β, weight decay, clipping | | |
| Learning-rate rule (fit formula of η\*(N); with or without the correction for training length) | | |
| Schedule (WSD: warmup, decay fraction, decay shape) | | |
| Precision (BF16 / FP8) | | |
| Code commit | | |

## 2. Ladder results and loss extrapolation

### 2.1 Learning-rate sweep

| Size | Swept learning rates (val_loss) | Selected η\* | Is the best value inside the grid? |
|---|---|---|---|

Fit: η\*(N) = ____ · N^(−____); extrapolated to l300m: ____; extrapolated to the main line: ____.

### 2.2 Fit of L(N, D)

- Fit points: ____ (sizes × budgets). Definition of N: non-embedding / total parameters / FLOPs (select one and give the reason)
- `L(N, D) = E + A/N^α + B/D^β`: E=____, A=____, α=____, B=____, β=____
- Fit residuals: RMSE ____, maximum relative error ____% (criterion: < 1%)
- Seed noise (l60m 20×, 2 seeds): ____%

| Run | N | D | Actual val_loss | Fit | Error |
|---|---:|---:|---:|---:|---:|

### 2.3 Held-out test (l300m, not in the fit)

| Budget | Actual | Extrapolated | 95% interval | Error | Inside the interval? |
|---|---:|---:|---|---:|---|

If a point is outside the interval or the error is > 1%: diagnosis ____, action ____ (fix the recipe and run the ladder again; do not patch the old fit).

### 2.4 Extrapolation to the main line

| Target | N | D | Extrapolated loss | 95% interval | Extrapolation factor (N / D relative to the largest fitted ladder point) |
|---|---:|---:|---:|---|---|
| Main line @ ____B | | | | | |
| Main line @ ____B (budget limit) | | | | | |

Attach the `fit_scaling` command and its original output in `runs/<date>-gate1/fit.log`.

## 3. Benchmark score extrapolation (Base)

Use the two-step method (Chapter 12; the method of Delphi and of the Llama 3 technical report):

1. For the **soft metric** of each benchmark (the log probability of the correct option / the bits-per-byte of the reference answer), fit its relation to the loss (or to the compute) on the ladder.
2. With a set of public models (run again with the same evaluation framework and the same template), fit an S-shaped mapping "soft metric → hard score" (`zero.tools.fit_scaling.fit_loss_to_score` or a similar tool).
   Combine the two steps to get the predicted scores of the main line.

| Benchmark (development-set version, not the preregistered test set) | Random level | Measured on the largest ladder model | Extrapolated to the main line | Interval | Notes on reliability |
|---|---:|---:|---:|---|---|

- A benchmark where all ladder models are near the random level: **do not extrapolate**. Write "no signal at the ladder scale".
- Do **not run** the preregistered test benchmarks at Gate 1 (GOAL.md Section 11). Here, use only our own development sets.

## 4. Recipe validation

### 4.1 (a) The post-training recipe applied to ladder Base models

| Ladder Base | Base val_loss | Tool-calling dev-set score after post-training | General dev-set score |
|---|---:|---:|---:|

Fit "Base loss → tool-calling score": ____ (form, coefficients, residuals). Extrapolate to the predicted loss of the main-line Base (Section 2.4): tool-calling score ____ (interval ____).
After post-training, all small models can be near a score of 0. In that case, write only "no signal at the ladder scale", use (b) as the main evidence, and explain why.

### 4.2 (b) The post-training recipe applied to an existing open Base model of the same size (only for validation, not for release)

- Base: ____ (license ____; gap to the main-line Base: general dev set ____)
- After post-training: tool-calling dev set ____; gap to the preregistered opponents (rerun scores) ____
- Interpretation: this is a reference for the "upper limit of the recipe". If our Base is weaker than this open Base, convert the result with the slope from 4.1.

## 5. Budget

| Item | Value | Source |
|---|---|---|
| Measured MFU / tok/s (Stage 6) | | `runs/<date>-gpu-check/` |
| Price | | Rental contract |
| Main-line tokens | | Output of `plan_budget` |
| Pretraining cost | | Same as above |
| Spent (up to this report) | | `runs/ledger.md` |
| Remaining budget | | |

Attach the original output of `plan_budget` below (with several values of MFU, to show the sensitivity).

## 6. Risks and items not verified

- The extrapolation factor and the interval width; does the recipe still hold for the long training of the main line? (the lesson of Delphi)
- Systematic error from the differences between the ladder and the main line (sequence length, batch)
- Paths not verified on a GPU yet: ____
- Other: ____

## 7. Decision request

- [ ] Approve the main-line pretraining: ____B tokens, budget $____ (limit $____; if the cost goes above the limit, stop first and report)
- [ ] Or: adjust the recipe / goal: ____

> All predictions in this report come from the ladder extrapolation. They are not measured results of the main-line model.
