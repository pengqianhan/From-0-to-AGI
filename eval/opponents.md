# Opponent list (candidates, **not frozen**)

**English** · [中文](opponents.zh.md)

> Status: **draft**. In Stage 7 of Step 2, set the freeze date. After the project lead confirms it, finalize the list (GOAL.md 3.2, item 3).
> After the list is final, add each change only as an "amendment" with a date and a reason.

## Rules (GOAL.md 3.2)

- **Scope**: all public-weight models with an official parameter count between **0.7 and 1.3 times** the size of our model. All models released before the freeze date count.
  The parameter count includes the embedding and uses the count of the publisher. For a multimodal model, use the parameter count of the language-model part.
- Our model: `configs/main/pretrain.toml`, **689.5M** (`uv run python -m zero.tools.count_params configs/main/pretrain.toml`;
  the vocabulary of 65,536 is provisional; if Chapter 13 changes the vocabulary, calculate again) → range **[482.7M, 896.4M]**.
- **Benchmark model**: whatever size class the final model has, **Qwen3.5-0.8B must be compared**.
- Starting point: `small-llms-under-5b-2026-08-30/inventory.md` (updated 2026-08-30). Before the freeze, also check for new releases after that date.
- If an opponent has a thinking mode and a non-thinking mode, test both modes and use the higher score. We run all opponents again ourselves, with the same framework.

## Candidate list (selected from inventory.md)

The column "Size class" gives the name of the size class in the inventory. A row marked "to be verified" has a size-class name inside the range. But before the freeze, compare its exact parameter count (embedding included) with the model card of the publisher. This is most important for models with a nominal size of "0.5B": the lower bound of the range is 482.7M, so the exact count can be outside the range.

| Organization | Model | Size class | Ratio to 689.5M (by size class) | Notes |
|---|---|---:|---:|---|
| Alibaba / Qwen | **Qwen3.5-0.8B** | 0.8B | 1.16 | **Benchmark model, always compared**. Multimodal, counted by the language-model part (model card: Language Model 0.8B, vocabulary 248,320, tied with the output layer). Test both thinking and non-thinking. The model card states that the **default is non-thinking**, and warns that the model easily gets stuck in loops in thinking mode (Chapter 11, verified 2026-09) |
| Alibaba / Qwen | Qwen3-0.6B | 0.6B | 0.87 | Model card: 0.6B (0.44B without embedding). Test both thinking and non-thinking (verified 2026-09) |
| Alibaba / Qwen | Qwen2.5-0.5B(-Instruct) | 0.49B | 0.71 | Model card: 0.49B (0.36B without embedding) → 490M, **inside the range**, but near the lower bound 482.7M. If the final size changes, calculate again (verified 2026-09) |
| Alibaba / Qwen | Qwen2.5-Coder-0.5B | 0.5B | 0.73 | Parameter count to be verified; code model |
| Alibaba / Qwen | Qwen2-0.5B | 0.5B | 0.73 | Parameter count to be verified |
| Alibaba / Qwen | Qwen1.5-0.5B | 0.5B | 0.73 | Parameter count to be verified |
| Meta research | MobileLLM-600M | 600M | 0.87 | |
| Liquid AI | LFM2-700M | 700M | 1.02 | Hybrid architecture |
| CyberAgent | OpenCALM-medium (830M) | 830M | 1.20 | Japanese model |
| BigScience | BLOOM-560M / BLOOMZ-560M | 560M | 0.81 | Multilingual; BLOOMZ is the instruction-tuned version |
| Cerebras | Cerebras-GPT-590M | 590M | 0.86 | |
| Tencent | Hunyuan-0.5B | 0.5B | 0.73 | Parameter count to be verified |
| OpenBMB | MiniCPM4-0.5B | 0.5B | 0.73 | Parameter count to be verified |
| H2O.ai | H2O-Danube3-500M | 0.5B | 0.73 | Parameter count to be verified |
| TII | Falcon-H1-0.5B | 0.5B | 0.73 | Hybrid attention / SSM; parameter count to be verified |
| MBZUAI | MobiLlama-0.5B | 0.5B | 0.73 | Parameter count to be verified |
| Swiss AI | Apertus-v1.1-0.5B | 0.5B | 0.73 | Fully open; parameter count to be verified |
| RWKV Foundation | RWKV 4/5/6/7, all versions | ~169M–2.9B | — | Verify each version: does a size inside the range exist? |

### Outside the range, but near the bounds (not opponents; check the parameter count again before the freeze)

| Model | Size class | Ratio | Notes |
|---|---:|---:|---|
| OpenELM-450M | 450M | 0.65 | Below the lower bound |
| Pythia-410M, OpenCALM-small(410M), InkubaLM-0.4B | ~0.4B | ~0.6 | Below the lower bound |
| SmolLM / SmolLM2-360M | 360M | 0.52 | Below the lower bound |
| Gemma 3 1B, OLMo (2) 1B, Pythia-1B, MobileLLM-1B, MiniCPM5-1B, Falcon3-1B, PLaMo-2-1B, and other models of the 1B class | 1B | 1.45 | Above the upper bound (if the exact parameter count is < 896.4M, include the model; check before the freeze) |

## Rerun plan and cost for Step 2

- Follow Stage 7 of `runs/RUNBOOK.md`. First, set the freeze date and the list, and pin the framework versions. Then do a trial run with 1–2 opponents (verify the open items of `zero/eval/bfcl.py`, and verify if ACEBench runs on a new version of vLLM).
- For each opponent, use the official template, both modes (if available), and the same decoding parameters. Run all benchmarks of Section 2 of `eval/PREREGISTRATION.md`, and save the per-question results.
- Cost estimate (RUNBOOK): about 25 models × about 1 GPU-hour each ≈ $60–200 (the GOAL.md 3.4 budget is about $200). If one run costs more than $100, get approval first. Record the costs in `runs/ledger.md`.

## Information to add at the freeze

For each opponent: the exact parameter count with a link to its source, the release date, the license, the HF repository and commit, the official chat / tool-calling template, and whether it has a thinking mode.
Also give the name of the handler for BFCL (if BFCL has a built-in handler, use it; if not, write an adapter layer).
