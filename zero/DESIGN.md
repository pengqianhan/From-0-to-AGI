# zero: production code of the main-line model

**English** · [中文](DESIGN.zh.md)

`zero` is the production code of the main-line model of this course (working name). It covers the same ideas as the minimal code in the `code/` folder of each chapter, but it can really train a model with 0.6–0.8B parameters. The requirements are in `GOAL.md` 9.1.

## Design principles

1. **Readability first**: follow the style of nanochat: one node, pure PyTorch, no black-box training framework. Each file starts with a paragraph that tells which chapter the file belongs to. Comments, docstrings, and printed output are in English (see Section 4 of `docs/STYLE_GUIDE.md`).
2. **The architecture is compatible with Qwen3 dense models**: Pre-Norm RMSNorm, SwiGLU, RoPE, GQA, QK-Norm, tied embeddings, no bias. Advantages:
   - We can check the correctness directly with a parity check: initialize a small Hugging Face `Qwen3ForCausalLM` at random, and move its weights into `zero`. The logits must agree.
   - After export, transformers / vLLM / llama.cpp support the model directly. The GGUF conversion uses the official llama.cpp script.
3. **Configuration-driven**: all hyperparameters are in `configs/*.toml`. The code does not hard-code them. There are three levels: `configs/tiny/` (CPU smoke test), `configs/ladder/` (ladder experiments), and `configs/main/` (main-line training).
4. **Tests first**: Step 1 has no GPU, so the trust in the code comes only from `tests/`. In the code, mark the paths that a CPU cannot test (multiple GPUs, FlashAttention kernels, FP8) with the comment `# not verified on GPU yet`.
5. **Runs offline**: the smoke test and the unit tests do not need a network. The tiny corpus is in `assets/tiny_corpus/` (committed to the repository, with license notes).

## Modules and chapters

| Module | Content | Chapters |
|---|---|---|
| `zero/config.py` | `ModelConfig`, `TrainConfig`, and other dataclasses; `load_config(path)` reads TOML | 9, 12 |
| `zero/model.py` | `RMSNorm`, `RotaryEmbedding` (with YaRN scaling), `Attention` (GQA + QK-Norm + SDPA + KV cache), `SwiGLU`, `Block`, `Transformer`; `Transformer.forward(tokens, kv_cache=None, start_pos=0) -> logits` | 8, 9, 10, 15 |
| `zero/kv_cache.py` | `KVCache`: a preallocated K/V cache | 10, 21 |
| `zero/generate.py` | `sample_next(logits, temperature, top_p, generator=None)`, `generate(model, prompt_ids, max_new_tokens, temperature=1.0, top_p=1.0, use_cache=True, eos_id=None, seed=None)` (returns the new tokens, without the prompt and without eos), `generate_stream(...)` (yields one step at a time) | 10 |
| `zero/hf.py` | `load_from_hf_qwen3(hf_model_or_state_dict, config=None) -> Transformer` (also accepts an export folder), `export_to_hf_qwen3(model, config, out_dir, tokenizer=None, dtype=torch.bfloat16)`, `config_to_hf_qwen3(config)`, `config_from_hf_qwen3(hf_config)` | 9, 20 |
| `zero/tokenizer.py` | `train_bpe(texts, vocab_size, special_tokens) -> Tokenizer` (a byte-level BPE based on HF `tokenizers`); `Tokenizer.encode/decode/save/load`; the compression statistic `bytes_per_token` | 7, 13 |
| `zero/data/` | `sources.py` (data set registry: name, address, license, language), `clean.py`, `dedup.py` (exact hash + MinHash), `quality.py` (heuristic filters + classifier interface), `decontam.py` (n-gram decontamination), `shard.py` (writes tokenized uint32 shards), `loader.py` (`PackedDataLoader`: distributed, with a `state_dict` for resume), `mixture.py` (samples from multiple sources by ratio), `prepare.py` (for tiny: raw text → tokenizer + training/validation shards) | 13, 14 |
| `zero/train/` | `dist.py` (DDP/FSDP initialization), `schedule.py` (cosine, WSD), `checkpoint.py` (model, optimizer, scheduler, data loader, random-number state), `trainer.py` (general training loop: BF16 autocast, gradient accumulation, gradient clipping, logs, evaluation, MFU estimate), `pretrain.py`, `midtrain.py` (entry point: `python -m zero.train.pretrain --config ...`) | 6, 12, 14, 15 |
| `zero/post/` | `chat.py` (renders and parses the chat template and the tool-calling format), `sft.py` (loss mask, packing), `distill.py` (teacher data generation + execution check + logits distillation loss), `dpo.py`, `grpo.py`, `envs/tool_env.py` (simulated APIs and verifiable rewards) | 16–19 |
| `zero/eval/` | `harness.py` (few-shot log-likelihood multiple choice, generative exact match), `bootstrap.py` (confidence intervals), `bfcl.py` (the adapter to the official BFCL for Step 2), `report.py` | 11, 20 |
| `zero/export/` | `gguf.py` (calls `convert_hf_to_gguf.py` of llama.cpp) | 20 |
| `zero/arch/` | The experiment modules of Part 5, **not for the main line**: `mla.py`, `sliding_window.py`, `linear_attention.py` (with the Gated DeltaNet recurrence), `moe.py`, `mtp.py`, `speculative.py` | 21–25 |
| `zero/tools/` | `estimate_cost.py` (estimate of GPU-hours and cost), `count_params.py`, `kv_cache_calc.py` | 12, 21 |
| `zero/demo/cli.py` | A local tool-calling command-line assistant | 20 |
| `zero/smoke.py` | The end-to-end smoke test: `uv run python -m zero.smoke` | All |

## Data format conventions

- Pretraining shards: `<name>_<idx>.bin` (a `np.uint32` token sequence, no file header, one `<|endoftext|>` **after each document**) + `<name>.json` (metadata: tokenizer hash, token count, document count, sources, token count of each shard).
- Chat data (SFT/DPO/RL): JSONL. Each line is `{"messages": [{"role": "system|user|assistant|tool", "content": "...", "tool_calls": [...]}], "tools": [...]}`. DPO data also has `chosen`/`rejected`.
- Tool-calling format (the same as the Qwen/Hermes style, so that inference frameworks recognize it after export): in the assistant message, use
  `<tool_call>{"name": "...", "arguments": {...}}</tool_call>`. Put the tool result in a message with `role="tool"`.

## Tests (`tests/`)

The tests match the list in GOAL.md 9.1: parity of the model logits, KV cache consistency, tokenizer round trip and compression ratio, parity of the SFT/DPO/GRPO losses with hand calculations, consistency of resume from a checkpoint, round trip of the tool-calling template, and the reward functions. All tests finish on a CPU in a few minutes: `uv run pytest`.

## Implementation notes (added when the zero core was done)

- **Configuration**: `load_config(path, overrides=None) -> Config`, `Config.model: ModelConfig`, `Config.train: TrainConfig` (it contains `data/optim/schedule/checkpoint/logging`). In TOML, write these sections as top-level tables `[data]`, `[optim]`, …. The loader supports inheritance with `base = "xxx.toml"` and command-line overrides with `--set section.key=value`.
- **Model**: `Transformer.forward(tokens, kv_cache=None, start_pos=0) -> logits`; `Transformer.loss(tokens, targets, ignore_index=-100)`; `num_params(non_embedding=False)`; `flops_per_token(seq_len)`. Module-level functions: `count_params(config)` (does not allocate memory) and `estimate_flops_per_token(config, seq_len)` (= 6·N_matmul + 12·L·q_dim·T). Parameter names: `tok_emb`, `layers.i.{attn_norm, attn.{wq,wk,wv,wo,q_norm,k_norm}, ffn_norm, ffn.{w_gate,w_up,w_down}}`, `norm`, `lm_head`. The mapping table to the HF names is at the start of `zero/hf.py`.
- **Special tokens of the tokenizer** (their ids are fixed at the start of the vocabulary): 0 `<|endoftext|>`, 1 `<|im_start|>`, 2 `<|im_end|>`, 3 `<tool_call>`, 4 `</tool_call>`, 5 `<tool_response>`, 6 `</tool_response>`, 7 `<think>`, 8 `</think>`, 9–15 `<|reserved_0..6|>`. `Tokenizer.eos_id` is `<|endoftext|>` for the Base model. `Tokenizer.save_hf(out_dir)` writes files that `AutoTokenizer` can read.
- **Training**: `Trainer(cfg, info=None).train(stop_at=None) -> list[dict]`. `run_training(cfg)` is the wrapper for the entry scripts (initialize the distributed setup → rank 0 prepares the data → train). The data loader `PackedDataLoader(paths, seq_len, batch_size, rank, world_size, seed, shuffle, device)` has the same interface as `MixtureLoader`: `next_batch() -> (x, y)`, `state_dict()`, `load_state_dict()`.
- **checkpoint**: `<ckpt_dir>/step_XXXXXXXX/{model.pt, optim.pt, meta.json, rank{r}.pt}` + `latest`. The code writes to a temporary folder first, and then renames it atomically.

## Implementation notes (added when post-training, evaluation, export, and the demo were done)

- **Chat template** (`zero/post/chat.py`): `render(messages, tools=None, add_generation_prompt=False, tokenizer=None, enable_thinking=False)`.
  Without a tokenizer, it returns `(text, per-character mask)`. With a tokenizer, it returns `(ids, per-token mask)` (it encodes the full text once, and then uses the character offsets from `Tokenizer.encode_with_offsets` to look up the mask).
  Other functions: `render_text`, `render_segments`, `assistant_text(msg)`, `encode_prompt_response(messages, response, tok, tools)` (marks only the response), `parse_assistant(text) -> ParsedAssistant(content, tool_calls, reasoning_content, errors)`, `format_tool_call`.
  `CHAT_TEMPLATE` is the equivalent Jinja template. The format is the same as the Qwen3 format, with one difference: when thinking is off, the template does not add an empty `<think></think>` after the generation prompt.
- **Export**: `export_to_hf_qwen3(..., chat=False, chat_template=None)`. If you give a tokenizer, it writes `CHAT_TEMPLATE` into `tokenizer_config.json`.
  With `chat=True`, eos is `<|im_end|>` (in generation_config, it is `[<|im_end|>, <|endoftext|>]`). `Tokenizer.save_hf(..., chat_template=None)`.
- **Configuration**: `DataConfig.format`: `"packed"` (default, pretraining shards) | `"sft"` (chat windows + mask, `zero.data.loader.MaskedWindowLoader`) | `"none"`
  (DPO / GRPO manage their own data, so `[[data.sources]]` can be empty). `zero.post.common.load_post_config(src, {name: dataclass}, overrides)` parses the sections of each stage (`[sft]`, `[teacher]`, `[distill]`, `[dpo]`, `[grpo]`).
  `src` can be a path or a dict (the smoke test uses a dict). The evaluation configuration has only `[eval]`
  (`zero.eval.harness.load_eval_config`).
- **Training loop**: SFT and distillation reuse `Trainer` (with `format = "sft"`, it reads `MaskedWindowLoader`; a new hook `Trainer.extra_metrics()`
  lets the distillation subclass log `ce` / `kd`). DPO / GRPO / on-policy distillation use `zero.post.common.LoopState` (optimizer, schedule, clipping, logs, checkpoint, resume).
  At this time, they are **single-process** implementations (CPU / one GPU). Multiple GPUs are not implemented.
- **Model loading**: `zero.post.common.load_policy(path)` supports both a zero checkpoint (it reads the tokenizer path from meta.json) and an HF folder.
- **SFT data files**: `<shard_dir>/{train,val}.bin` (uint32, n_windows × (seq_len+1)) + a `.mask` file with the same name (uint8) + `.json` metadata.
- **Tool environment** (`zero/post/envs/tool_env.py`): `TOOLS`, `execute_call`, `validate_arguments`, `Task`, `generate_tasks(n, seed, split)`,
  `dev_tasks(n)` (a fixed dev set; the training tasks automatically exclude the same questions), `make_splits`, `reference_messages(task)`,
  `score_tool_calls(task, text) -> Reward`, `score_final_answer`, `run_episode(policy, task)`. The frozen dev set file is `zero/eval/tasks/tool_dev.jsonl`.
- **Loss functions**: `zero.post.dpo.dpo_loss`, `zero.post.grpo.group_advantages` / `grpo_loss`, `zero.post.distill.kd_loss` / `reverse_kl_loss`.
- **Evaluation**: `zero.eval.harness` (`eval_multiple_choice`, `eval_exact_match`, `eval_tool_calls`, `run_eval`),
  `zero.eval.bootstrap.paired_bootstrap` / `compare_to_opponent`, `zero.eval.report.write_report`, `zero.eval.bfcl` (not verified yet).
- **GGUF**: `zero.export.gguf.convert_hf_to_gguf / build_llama_cpp / quantize / run_llama / llama_tokenize`.
  Before the code calls the official `convert_hf_to_gguf.py`, it applies a runtime patch: if the pre-tokenizer rule is unknown, treat it as `qwen2` (the regex is the same; `tests/test_gguf.py` does a parity check with `llama-tokenize`).
- **demo**: `zero.demo.cli` (`chat_turn`, `search_files`, `execute_demo_tool`). **Smoke test**: `zero.smoke`.

## Additional notes (after Chapters 12–14)

- **Multi-source mixing and multiple GPUs**: `MixtureLoader` draws the source separately for each `(seed, rank)`. Thus, with multi-source data, "2 GPUs = 1 GPU with a double batch" is not true bit for bit (it is true for a single source; see tests/test_ddp_cpu.py). The expected source ratio on each rank is the same.
- **Activation checkpointing**: with `train.activation_checkpointing = true`, each Block keeps only its input and calculates the rest again in the backward pass (tests/test_activation_checkpointing.py makes sure that the gradients agree). We verified it on one RTX 3090 (2026-10, see runs/2026-10-01-gpu0-check/). On CUDA, the gradients with and without it are identical bit for bit. With the main-line configuration and T=4096, a 24 GB GPU cannot fit micro batch 1 without it. With it, the GPU fits a maximum of 3.
- **Optimizer**: `optim.name = "muon"` changes to `zero/train/muon.py` (Muon for 2D weights, AdamW for the other parameters). Chapter 12 verified that Muon satisfies the consensus rule. The ladder experiments decide if the main line uses it.
- **Validation bits-per-byte**: if `data.tokenizer` is set, the pretraining evaluation also records `val_bpb` (`zero/data/bpb.py`).
- **GPU memory of the main-line configuration**: micro batch 4 × accumulation 4 (in Chapter 14, `zero/tools/memory_calc.py` estimated that micro batch 8 needs more than 80 GB).
