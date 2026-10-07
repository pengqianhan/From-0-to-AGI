# Preregistration — **Draft / not frozen**

**English** · [中文](PREREGISTRATION.zh.md)

> **Status: DRAFT. Not frozen. This file does not count as a registration yet.**
> According to GOAL.md 3.2, set the opponent list and the freeze date in Step 2, before the main-line pretraining starts. After the project lead confirms them, finalize and commit this file.
> **The commit time is the registration time.** After the file is final, add each change only as an entry with "date + reason" in the section "Amendments" at the end. Do not rewrite the file silently.
> At the freeze, all "TBD" (to be decided) items in this draft must get specific values. An item marked "candidate" is a proposal of this draft. The project lead must confirm it.
> Chapter 11 (`chapters/11-evaluation/README.md`) verified the benchmark facts in this draft (number of questions, categories, scoring method, versions), as of 2026-09.

GOAL.md 3.2 requires that this file states five items. They are in Section 2 (benchmarks and versions), Section 3 (frameworks and versions), Section 4 (prompts, templates, decoding parameters),
Section 5 (opponent list and freeze date), and Section 6 (the criterion for "ahead").

## 1. The claims to test

- **Hard goal (specific)**: on the two **primary endpoints** of Section 2.1, our final model is **ahead** of every model in the opponent list (`eval/opponents.md`), Qwen3.5-0.8B included.
- **Soft goal (general)**: report honestly on the general benchmarks of Section 2.2. Compare our rank with the "fully open recipe" models. Write the gap to the open-weight-only models honestly.
- By default, our model has only a non-thinking mode.

## 2. Benchmarks and versions

### 2.1 Specific group (hard goal)

| Primary endpoint | Benchmark | Version / data snapshot | Included categories | Metric | Status |
|---|---|---|---|---|---|
| **E1** | BFCL | The latest version at the freeze (now V4; the latest `bfcl-eval` on PyPI is 2026.3.23) | Candidate: the four non-Agentic sections. Non-Live (`simple_python`, `simple_java`, `simple_javascript`, `multiple`, `parallel`, `parallel_multiple`), Live (`live_simple`, `live_multiple`, `live_parallel`, `live_parallel_multiple`), Relevance (`irrelevance`, `live_irrelevance`, `live_relevance`), Multi-Turn (`multi_turn_base`, `multi_turn_miss_func`, `multi_turn_miss_param`, `multi_turn_long_context`) | Weighted accuracy: the accuracy of each section, weighted with the official V4 weights 10 / 10 / 10 / 30, renormalized (the scoring rules inside each section are the same as the official rules) | Candidate, to be confirmed |
| **E2** | ACEBench (Chinese) | Repository commit TBD (paper arXiv:2501.12851 v8; on 2025-10-29 the repository still corrected answers; MIT license) | Normal (atom, single-turn, multi-turn, similar API, preference) + Special (incomplete, error, irrelevant) of `data_zh` | Accuracy with the official definition (Normal: AST match; Special: rule-based decision). How to combine the two types: TBD (candidate: the official weighting by "square root of the number of samples") | Candidate, to be confirmed |

Report only, not a hard goal:

| Benchmark | Content | Reason |
|---|---|---|
| BFCL V4 official total score (with the Agentic 40%) | `web_search_*`, `memory_*` | Web search depends on SerpAPI and live web pages, so it is not reproducible. To be verified: whether the memory category can be reproduced offline. Report it next to E1 |
| BFCL `format_sensitivity` | A category without a score | Descriptive report |
| ACEBench Agent (Chinese, English) | multi-turn, multi-step | GPT-4o must play the user: closed source, costs money, not reproducible |
| ACEBench English Normal + Special | — | Descriptive report |
| τ²-bench | airline, retail, telecom (50 / 115 / 114 questions) | A large model simulates the user. The simulated user itself makes errors at a rate of 16%–47% (Table 2 of the paper). Metric: pass^k; k and the user-simulator model: TBD |

### 2.2 General group (report honestly)

| Benchmark | Data / subset | Questions | Scoring | Status |
|---|---|---:|---|---|
| MMLU-Redux | `edinburgh-dawg/mmlu-redux-2.0` (revision TBD). Whether to remove the questions with `error_type ≠ ok`: TBD (candidate: all 5,700 questions for the main result; the version without them as a descriptive report) | 5,700 | Accuracy | Candidate |
| MMLU-Pro | Official test | 12,032 (10 choices, pick 1) | 5-shot CoT, then extract the answer with the official regex | Candidate |
| C-Eval | **val** (the test answers are not public; when we run the opponents again ourselves, we can use only val) | 1,346 | Accuracy | Candidate |
| CMMLU | Official test | 11,528 or 11,582 (the main text and the appendix of the paper do not agree; verify with the data files) | Accuracy | Candidate |
| GSM8K | test | 1,319 | Extract the final number, exact match | Candidate |
| MATH-500 | The 500-question subset of PRM800K | 500 | Answer-equivalence check; checker TBD | Candidate |
| HumanEval+ / MBPP+ | EvalPlus | 164 / 378 | pass@1 (greedy) | Candidate |
| IFEval | `google/IFEval` | 541 | Main result: prompt-level strict. Also report the other three scores | Candidate |
| Base model (for Gate 2) | HellaSwag, ARC-Easy/Challenge, PIQA, WinoGrande; the 5-shot log-likelihood versions of MMLU / C-Eval / CMMLU | — | acc and acc_norm (which one is the main result: TBD) | Candidate |

## 3. Evaluation frameworks and versions (fill in at the freeze)

| Framework | Use | Version | Notes |
|---|---|---|---|
| lm-evaluation-harness | General group, Base benchmarks | TBD (the latest on PyPI now is 0.4.13, 2026-08-31) | `--log_samples` saves the result of each question. Verified: `mmlu_redux`, `mmlu_pro`, `ceval`, `cmmlu`, `gsm8k`, `ifeval`, `hellaswag`, `arc`, `piqa`, `winogrande` have ready-made tasks. Which task or script to use for MATH-500: to be verified |
| bfcl-eval | BFCL | TBD (the latest now is 2026.3.23) | Our model uses `ZeroFCHandler` in `zero/eval/bfcl.py` (with the chat_template from the export folder). **Not verified yet** |
| ACEBench | E2 | Repository commit TBD | The official `requirements.txt` pins `vllm==0.6.1.post1`. New models will probably need an adapter layer (to be verified) |
| EvalPlus | HumanEval+ / MBPP+ | TBD (the latest now is 0.3.1) | |
| vLLM | Inference backend | TBD | The same version for all models |
| This repository | Paired bootstrap, reports | The commit at the freeze | `zero/eval/bootstrap.py`, `zero/eval/report.py`. The stratified / weighted bootstrap that E1 needs is implemented as `stratified_paired_bootstrap`. The overall decision for several opponents × several endpoints is `overall_verdict` (see Section 6; the tests are in `tests/test_eval.py`) |

## 4. Prompts, templates, and decoding parameters

- **Templates**: each model uses its own **official** chat / tool-calling template (the `chat_template` in the HF `tokenizer_config.json`, or the built-in handler of BFCL).
  Our model uses the `chat_template` in the export folder (`CHAT_TEMPLATE` in `zero/post/chat.py`). It is the same as in training, character for character. `tests/test_chat.py` makes sure of this.
  FC mode or Prompt mode in BFCL for each model: TBD (candidate: FC for a model with an official FC handler, else Prompt; if a model has both, take the higher score).
- **System prompt**: the official default of each benchmark. If a benchmark has no official default, do not add a system prompt.
- **Number of few-shot examples and the examples**: the default settings of each benchmark in the framework that we use. At the freeze, write down each one.
- **Thinking mode**: if an opponent has a thinking mode and a non-thinking mode, **test both**. **For each benchmark**, compare with the mode that has the higher score (GOAL.md 3.2, item 4).
  Reason: the stronger mode depends on the benchmark (Qwen3.5-0.8B model card: MMLU-Redux thinking 59.5 / non-thinking 48.5; IFEval thinking 44.0 / non-thinking 52.1).
  Maximum output length in thinking mode: TBD. If the model reaches the limit and gives no answer, count the answer as wrong (candidate). Report the fraction of these answers separately.
- **Decoding parameters** (the same for all models, GOAL.md 3.2, item 4): the candidate is greedy decoding (temperature = 0; the BFCL default is 0.001). max_new_tokens: TBD.
  For metrics that need sampling (for example, pass@k, and pass^k of τ²-bench), fix temperature / top_p / seed / number of samples.
- **For the project lead to decide**: the official evaluation of Qwen3.5 uses sampling (for language benchmarks: temperature 1.0, top_p 0.95, top_k 20, presence_penalty 1.5).
  The model card also warns that 0.8B easily gets stuck in loops in thinking mode. Thus shared greedy decoding can be a disadvantage for this model. Candidate approach: use shared greedy decoding for the main comparison.
  Then run again with the officially recommended parameters of each model, and use the higher of the two scores as the opponent score. (This extends "take the higher mode" to the decoding parameters.)

## 5. Opponent list and freeze date

See `eval/opponents.md`. The list has all public-weight models with an official parameter count between 0.7 and 1.3 times the size of our model. (The count includes the embedding and uses the count of the publisher. For a multimodal model, count the language-model part.)
All models released before the freeze date count. Qwen3.5-0.8B is always compared. With the current 689.5M of `configs/main/pretrain.toml`, the range is [482.7M, 896.4M]. (Chapter 12 decides the final size. Calculate the range again at the freeze.)
Freeze date: **TBD**. At release, check again for new models released after the freeze date. Write them in the section "Opponents added after release", even if they are stronger than our model.

## 6. The criterion for "ahead"

- **Single comparison**: on the same set of questions, do a **paired bootstrap** of the per-question scores (10,000 resamples, 95% confidence interval with the percentile method, seed TBD). Difference = ours − opponent
  (for the opponent, take the higher score with the rules of Section 4). Lower bound of the interval > 0 → **ahead**. Upper bound < 0 → **behind**. The interval includes 0 → **tie** (`decide` in `zero/eval/bootstrap.py`).
- **Resampling for E1**: E1 is a score weighted by category. Thus do the resampling inside each category separately (stratified bootstrap). Then combine the categories with the weights of Section 2.1. Implementation: `stratified_paired_bootstrap(a_by, b_by, weights, ...)` in `zero/eval/bootstrap.py`. (With only one category, it gives the same result as `paired_bootstrap`.)
- **Condition for the hard goal** (candidate): for **every** opponent in the frozen list of `eval/opponents.md`, E1 and E2 **both** give "ahead".
  This is an intersection-union test. Each single comparison must be significant at the 5% level. Then the type I error rate of the overall conclusion is not more than 5%. Thus we do not apply an additional correction for multiple comparisons.
- **All other scores** (each category, each subset, the general group, the report-only benchmarks) are only descriptive. They are not part of the decision for "ahead". They cannot become primary endpoints after the fact.
- We show the officially published scores next to our scores, but they are not the basis for the comparison.

## 7. Decontamination method

- **n-gram overlap**: check the 13-gram overlap between all training data (pretraining, mid-training, SFT, synthetic teacher data, preference data, RL tasks) and all evaluation sets of Section 2
  (`zero/data/decontam.py`; a "word" = an English alphanumeric string / one Chinese character; a question with fewer than 13 words must match as a full question). Delete each training document with a hit, or mask the full passage. Write the hit rate in the model card.
  What to check (the question / the question + the answer) and the value of n for short Chinese questions: TBD. (`code/05_contamination.py` of Chapter 11 shows that a match on a short question stem is not necessarily a leak.)
- **canary**: scan the training corpus for known canary strings (for example, the GUID of BIG-bench). If a document contains one, delete the full document.
- **Tool-calling check**: compare the function names and parameter schemas of the training data with those of BFCL and ACEBench. Remove all functions with the same name and the same parameters, and count them. Check the functions with the same name but different parameters by hand. Record the numbers.
- **Our own environment**: deduplicate the training tasks of `zero/post/envs/tool_env.py` against its fixed dev set by question text. (The question space of the two types "weather" and "small talk" is too small for deduplication. Report them separately.)
- Write the results and the methods in the model card.

## 8. Roles of the development set and the test set

- Use only the development set to pick checkpoints, tune hyperparameters, and select prompts: `zero/eval/tasks/tool_dev.jsonl` (a frozen file) and a general development set, TBD (candidate: C-Eval dev, the validation split of each benchmark).
- Run the test benchmarks of Section 2 only at the gates (GOAL.md 3.4) and in the final evaluation. Record each run in `runs/ledger.md`.

## 9. Reporting the results

- Write all results (also the items where we are behind) in the model card and in Chapter 20. Publish the per-question results and the evaluation logs with the release.
- If we do not reach the hard goal, do not claim "ahead". Write an honest gap analysis.

## Amendments

(After the freeze, add changes only here: the date, the change, and the reason.)
