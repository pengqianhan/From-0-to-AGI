# Release checklist (Step 2, Stage 10 · Gate 3 and release)

**English** · [中文](RELEASE_CHECKLIST.zh.md)

> Related documents: Chapter 20 [`chapters/20-release/`](../chapters/20-release/README.md), Section 6 of [`runs/RUNBOOK.md`](RUNBOOK.md),
> [`eval/PREREGISTRATION.md`](../eval/PREREGISTRATION.md). We wrote this checklist in Step 1. **No item has been done yet.**
> Check the items one by one. For each item, write a file path or a link in the "Evidence" column. If you cannot check an item, write the reason. Do not skip it.
> Before the release, **wait for the final confirmation of the project lead** (GOAL.md Section 10, Stage 10).

## A. Gate 3: final evaluation (follow the preregistered protocol exactly, with no change)

| # | Item | Evidence | Status |
|---|---|---|---|
| A1 | Make sure that `eval/PREREGISTRATION.md` is final, and record the registration commit. Each change after the freeze is in the section "Amendments", with a date and a reason | commit hash | [ ] |
| A2 | The versions of the evaluation frameworks agree with the preregistration (lm-evaluation-harness, bfcl-eval, vLLM, evalplus). Archive the output of `pip freeze` | `runs/<date>-final-eval/env.txt` | [ ] |
| A3 | Evaluate **the released weights**: the exported HF folder (`export_to_hf_qwen3(..., chat=True)`), not a training checkpoint. Record the sha256 of `model.safetensors` | sha256 | [ ] |
| A4 | Templates: our model uses the `chat_template` in the exported folder; each opponent uses its own official template. The decoding parameters are the same as in the preregistration | Evaluation logs | [ ] |
| A5 | Run all special benchmarks (BFCL + Chinese tool-calling benchmarks) and all general benchmarks. Save the **per-question** results (`--log_samples`) | `eval/results/final/` | [ ] |
| A6 | Do a paired bootstrap for each opponent and each benchmark (`zero.eval.bootstrap`, 10,000 resamples, 95% CI; for the opponent, use the higher of thinking / non-thinking) | Comparison table | [ ] |
| A7 | Use the preregistered criteria to write "ahead / tie / behind" in each cell. **List all cells.** Do not select benchmarks | Table from `zero.eval.report` | [ ] |
| A8 | Is the hard goal reached? Check each item of the "conditions for the hard goal" in the preregistration. If the goal is not reached, **do not claim "ahead"**, and write a gap analysis | Conclusion paragraph | [ ] |
| A9 | Show the official scores that the opponents published side by side, but only as a reference | Table | [ ] |
| A10 | Opponents added after release: look for models of 0.7–1.3 times our size that were released after the freeze date. Run them if you can. Write them into the section "Opponents added after release", even if they are better than our model | List + results | [ ] |
| A11 | The test benchmarks ran only one time, here. We did not use them to select checkpoints or to tune hyperparameters (GOAL.md Section 11) | Statement | [ ] |

## B. Decontamination

| # | Item | Evidence | Status |
|---|---|---|---|
| B1 | Do the 13-gram overlap check of all training data (pretraining, mid-training, SFT, teacher synthetic data, preference data, RL tasks) with all evaluation sets (`zero/data/decontam.py`) | Table of hit rates | [ ] |
| B2 | Compare the function names and argument schemas of the tool-calling data with BFCL and the other evaluation sets. Remove the overlaps and count them | Count | [ ] |
| B3 | Write the results into the section "Decontamination check" of the model card | Model card | [ ] |

## C. Weights and formats

| # | Item | Evidence | Status |
|---|---|---|---|
| C1 | HF folder: `config.json` (`Qwen3ForCausalLM`), `generation_config.json`, `model.safetensors` (bf16), `tokenizer.json`, `tokenizer_config.json` (with `chat_template`) | Folder listing | [ ] |
| C2 | After `transformers` loads the model, the logits agree with zero (use the method of `tests/test_export_hf.py`, and run it again on the final weights) | Maximum error | [ ] |
| C3 | The output of `apply_chat_template` is identical, character by character, to the template in training (an example with tools) | Empty diff | [ ] |
| C4 | vLLM: `vllm serve <dir> --enable-auto-tool-choice --tool-call-parser hermes` can load the model, and a request with tools returns structured `tool_calls` | Archived request / response | [ ] |
| C5 | GGUF: `python -m zero.export.gguf --hf-dir ... --outtype f16`. `llama-tokenize` agrees token by token with our tokenizer (one sample each for Chinese, English, code, and a tool call) | Parity-check output | [ ] |
| C6 | The greedy output of the f32 (or f16) GGUF agrees token by token with zero (YaRN configuration included) | Parity-check output | [ ] |
| C7 | Quantization: Q8_0, Q4_K_M. Use `llama-perplexity` on our own development set to measure PPL and KLD (against f16). Write them into the model card | PPL / KLD table | [ ] |
| C8 | Scores of the quantized versions (at least Q4_K_M) on the tool-calling development set, compared with bf16. If the score drops clearly, say so in the model card, and recommend Q8_0 | Table | [ ] |
| C9 | Ollama: the `Modelfile` contains `FROM ./zero-Q4_K_M.gguf` + the chat template. After `ollama create`, the model can chat | Screen recording / log | [ ] |
| C10 | Run Q4_K_M on a normal laptop (record the laptop model, the memory, and the operating system). Record the memory use and token/s | Table | [ ] |

## D. Local demo

| # | Item | Evidence | Status |
|---|---|---|---|
| D1 | `python -m zero.demo.cli --model <final HF folder> --root <demo folder>`: record one video for each of three kinds of questions: calculator, date, and file search | Screen recording | [ ] |
| D2 | Also record the failure examples (wrong tool, wrong arguments, made-up results). Write them into "Known limitations" in the model card | Screen recording | [ ] |
| D3 | File search stays inside `--root` (`tests/test_demo.py` covers escapes with `../` and with symbolic links). The README tells users not to set the root folder to their home folder | Statement | [ ] |

## E. License and attribution (the author decides; the executor prepares the material)

| # | Item | Evidence | Status |
|---|---|---|---|
| E1 | Weights license: the author selects one of the options listed in Chapter 20. Write it into `license:` in the YAML of the model card and into the `LICENSE` of the repository | Decision record | [ ] |
| E2 | For each training data set: the name, the version, the license, and the attribution requirement (ODC-By data requires attribution; CC-BY requires attribution; code data follows its original license). Put them in a table | Data table | [ ] |
| E3 | Distillation teacher: the name, the version, and the terms in the license text about "use of the outputs to train other models" | Extract of the terms | [ ] |
| E4 | Write the license of the code repository and the license of the model separately | LICENSE file | [ ] |

## F. Release items (GOAL.md 3.5)

| # | Item | Evidence | Status |
|---|---|---|---|
| F1 | Hugging Face: three repositories for Base, SFT, and the final version (or one repository with several branches). Publish the important intermediate checkpoints as `revision` branches (the name gives the number of steps and the number of tokens) | Link | [ ] |
| F2 | GGUF repository: Q8_0, Q4_K_M (f16 is optional) | Link | [ ] |
| F3 | Model card: use `chapters/20-release/code/04_model_card.py` to make the skeleton from the evaluation JSON. Then add the text by hand. **No "TBD after training" placeholder can be left** | Model card | [ ] |
| F4 | Reproducible recipe: `configs/main/`, the data-processing scripts, the training log summaries, and the costs in `runs/ledger.md` | Link | [ ] |
| F5 | Publish the evaluation report (per-question results + logs) together with the model | Link | [ ] |
| F6 | Put the results back into the course: add the real results to "Main-line progress" in Chapters 11–20. In the Chapter 20 video, replace the shots labeled "tiny-configuration demo" with real data, and render them again | PR | [ ] |
| F7 | Record the costs in `runs/ledger.md` | Line number | [ ] |
| F8 | **Wait for the final confirmation of the project lead** before you make the repository public | Confirmation record | [ ] |

## G. After the release

- Open an issue template / a discussion area to collect feedback: failed tool calls (with the input, the tool definitions, and the output), problems with the quantized versions, and license problems.
- If other people get evaluation scores that differ from ours, first check the framework versions, the templates, and the decoding parameters. Then update the model card, and **write the date and the reason of each change**.
- Do not change the evaluation tables silently because of feedback after the release. Add new evaluation results as "revisions".
