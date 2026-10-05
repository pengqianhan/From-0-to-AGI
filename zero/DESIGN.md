# zero：主线模型的生产级代码

`zero` 是本课主线模型的生产级代码（暂名）。它和各章 `code/` 里的极简代码讲同一件事，但要能真正训练 0.6–0.8B 的模型。规范见 `GOAL.md` 9.1。

## 设计原则

1. **可读优先**：参考 nanochat 的风格——单节点、纯 PyTorch、没有黑盒训练框架。每个文件开头一段中文说明它对应哪一章。
2. **架构与 Qwen3 稠密模型兼容**：Pre-Norm RMSNorm、SwiGLU、RoPE、GQA、QK-Norm、共享 embedding、无 bias。好处：
   - 正确性可以直接对拍：随机初始化一个小的 Hugging Face `Qwen3ForCausalLM`，把权重搬进 `zero`，logits 必须一致；
   - 导出后直接被 transformers / vLLM / llama.cpp 支持，GGUF 转换用 llama.cpp 官方脚本。
3. **配置驱动**：所有超参在 `configs/*.toml` 里，代码不写死。三档：`configs/tiny/`（CPU 冒烟）、`configs/ladder/`（阶梯实验）、`configs/main/`（主线训练）。
4. **测试先行**：第一步没有 GPU，可信度全靠 `tests/`。CPU 上测不了的路径（多卡、FlashAttention kernel、FP8）在代码里标注 `# 尚未在 GPU 上验证`。
5. **离线可跑**：冒烟测试和单元测试不依赖网络。极小语料放在 `assets/tiny_corpus/`（提交进仓库，附许可证说明）。

## 模块与对应章节

| 模块 | 内容 | 对应章节 |
|---|---|---|
| `zero/config.py` | `ModelConfig`、`TrainConfig` 等 dataclass；`load_config(path)` 读 TOML | 9、12 |
| `zero/model.py` | `RMSNorm`、`RotaryEmbedding`（含 YaRN 缩放）、`Attention`（GQA + QK-Norm + SDPA + KV cache）、`SwiGLU`、`Block`、`Transformer`；`Transformer.forward(tokens, kv_cache=None, start_pos=0) -> logits` | 8、9、10、15 |
| `zero/kv_cache.py` | `KVCache`：预分配的 K/V 缓存 | 10、21 |
| `zero/generate.py` | `sample_next(logits, temperature, top_p, generator=None)`、`generate(model, prompt_ids, max_new_tokens, temperature=1.0, top_p=1.0, use_cache=True, eos_id=None, seed=None)`（返回新 token，不含提示词和 eos）、`generate_stream(...)`（逐步产出） | 10 |
| `zero/hf.py` | `load_from_hf_qwen3(hf_model_or_state_dict, config=None) -> Transformer`（也可传导出目录）、`export_to_hf_qwen3(model, config, out_dir, tokenizer=None, dtype=torch.bfloat16)`、`config_to_hf_qwen3(config)`、`config_from_hf_qwen3(hf_config)` | 9、20 |
| `zero/tokenizer.py` | `train_bpe(texts, vocab_size, special_tokens) -> Tokenizer`（基于 HF `tokenizers` 的 byte-level BPE）；`Tokenizer.encode/decode/save/load`；压缩率统计 `bytes_per_token` | 7、13 |
| `zero/data/` | `sources.py`（数据集登记表：名称、地址、许可证、语言）、`clean.py`、`dedup.py`（精确哈希 + MinHash）、`quality.py`（启发式过滤 + 分类器接口）、`decontam.py`（n-gram 去污染）、`shard.py`（分词后写成 uint32 分片）、`loader.py`（`PackedDataLoader`：分布式、可断点续传的 `state_dict`）、`mixture.py`（多来源按比例采样）、`prepare.py`（tiny 用：原始文本 → 分词器 + 训练/验证分片） | 13、14 |
| `zero/train/` | `dist.py`（DDP/FSDP 初始化）、`schedule.py`（cosine、WSD）、`checkpoint.py`（模型、优化器、调度器、数据加载器、随机数状态）、`trainer.py`（通用训练循环：BF16 autocast、梯度累积、梯度裁剪、日志、评估、MFU 估计）、`pretrain.py`、`midtrain.py`（入口：`python -m zero.train.pretrain --config ...`） | 6、12、14、15 |
| `zero/post/` | `chat.py`（对话模板与工具调用格式的渲染/解析）、`sft.py`（loss mask、打包）、`distill.py`（教师数据生成 + 执行验证 + logits 蒸馏损失）、`dpo.py`、`grpo.py`、`envs/tool_env.py`（模拟 API 与可验证奖励） | 16–19 |
| `zero/eval/` | `harness.py`（少样本对数似然选择题、生成式精确匹配）、`bootstrap.py`（置信区间）、`bfcl.py`（第二步对接官方 BFCL 的适配层）、`report.py` | 11、20 |
| `zero/export/` | `gguf.py`（调用 llama.cpp 的 `convert_hf_to_gguf.py`） | 20 |
| `zero/arch/` | 第五部分的实验模块，**不用于主线**：`mla.py`、`sliding_window.py`、`linear_attention.py`（含 Gated DeltaNet 递推）、`moe.py`、`mtp.py`、`speculative.py` | 21–25 |
| `zero/tools/` | `estimate_cost.py`（卡时与费用估算）、`count_params.py`、`kv_cache_calc.py` | 12、21 |
| `zero/demo/cli.py` | 本地工具调用命令行助手 | 20 |
| `zero/smoke.py` | 端到端冒烟测试：`uv run python -m zero.smoke` | 全部 |

## 数据格式约定

- 预训练分片：`<name>_<idx>.bin`（`np.uint32` token 序列，无文件头，**每篇文档后面**跟一个 `<|endoftext|>`）+ `<name>.json`（元数据：分词器哈希、token 数、文档数、来源、各分片 token 数）。
- 对话数据（SFT/DPO/RL）：JSONL，每行 `{"messages": [{"role": "system|user|assistant|tool", "content": "...", "tool_calls": [...]}], "tools": [...]}`；DPO 额外有 `chosen`/`rejected`。
- 工具调用格式（与 Qwen/Hermes 风格一致，便于导出后被推理框架识别）：助手消息里用
  `<tool_call>{"name": "...", "arguments": {...}}</tool_call>`，工具结果放在 `role="tool"` 的消息里。

## 测试（`tests/`）

对应 GOAL.md 9.1 的清单：模型 logits 对拍、KV cache 一致性、分词器往返与压缩率、SFT/DPO/GRPO 损失手算对拍、断点续训一致性、工具调用模板往返与奖励函数。全部在 CPU 上几分钟内跑完：`uv run pytest`。

## 实现备注（zero core 完成时补充）

- **配置**：`load_config(path, overrides=None) -> Config`，`Config.model: ModelConfig`、`Config.train: TrainConfig`（内含 `data/optim/schedule/checkpoint/logging`）。TOML 里这些小节写成顶层表 `[data]`、`[optim]`……；支持 `base = "xxx.toml"` 继承和命令行 `--set section.key=value` 覆盖。
- **模型**：`Transformer.forward(tokens, kv_cache=None, start_pos=0) -> logits`；`Transformer.loss(tokens, targets, ignore_index=-100)`；`num_params(non_embedding=False)`；`flops_per_token(seq_len)`。模块级函数 `count_params(config)`（不分配内存）、`estimate_flops_per_token(config, seq_len)`（= 6·N_matmul + 12·L·q_dim·T）。参数名：`tok_emb`、`layers.i.{attn_norm, attn.{wq,wk,wv,wo,q_norm,k_norm}, ffn_norm, ffn.{w_gate,w_up,w_down}}`、`norm`、`lm_head`，与 HF 的对照表在 `zero/hf.py` 开头。
- **分词器特殊 token**（id 固定在词表最前面）：0 `<|endoftext|>`、1 `<|im_start|>`、2 `<|im_end|>`、3 `<tool_call>`、4 `</tool_call>`、5 `<tool_response>`、6 `</tool_response>`、7 `<think>`、8 `</think>`、9–15 `<|reserved_0..6|>`。`Tokenizer.eos_id` 为 Base 模型的 `<|endoftext|>`；`Tokenizer.save_hf(out_dir)` 写出 `AutoTokenizer` 可读的文件。
- **训练**：`Trainer(cfg, info=None).train(stop_at=None) -> list[dict]`；`run_training(cfg)` 是入口脚本用的封装（初始化分布式 → rank 0 准备数据 → 训练）。数据加载器 `PackedDataLoader(paths, seq_len, batch_size, rank, world_size, seed, shuffle, device)` 与 `MixtureLoader` 接口相同：`next_batch() -> (x, y)`、`state_dict()`、`load_state_dict()`。
- **checkpoint**：`<ckpt_dir>/step_XXXXXXXX/{model.pt, optim.pt, meta.json, rank{r}.pt}` + `latest`；先写临时目录再原子改名。

## 实现备注（后训练、评测、导出、demo 完成时补充）

- **对话模板**（`zero/post/chat.py`）：`render(messages, tools=None, add_generation_prompt=False, tokenizer=None, enable_thinking=False)`
  不给分词器返回 `(text, 逐字符 mask)`，给了返回 `(ids, 逐 token mask)`（整段一次编码，再用 `Tokenizer.encode_with_offsets` 的字符偏移查 mask）；
  `render_text`、`render_segments`、`assistant_text(msg)`、`encode_prompt_response(messages, response, tok, tools)`（只标回复部分）、
  `parse_assistant(text) -> ParsedAssistant(content, tool_calls, reasoning_content, errors)`、`format_tool_call`；
  `CHAT_TEMPLATE` 是等价的 Jinja 模板。格式与 Qwen3 一致，差别只有一处：关闭思考时不在生成提示后塞空的 `<think></think>`。
- **导出**：`export_to_hf_qwen3(..., chat=False, chat_template=None)`：给了分词器就把 `CHAT_TEMPLATE` 写进 `tokenizer_config.json`；
  `chat=True` 时 eos 为 `<|im_end|>`（generation_config 里是 `[<|im_end|>, <|endoftext|>]`）。`Tokenizer.save_hf(..., chat_template=None)`。
- **配置**：`DataConfig.format`：`"packed"`（默认，预训练分片）| `"sft"`（对话窗口 + mask，`zero.data.loader.MaskedWindowLoader`）| `"none"`
  （DPO / GRPO 自己管数据，`[[data.sources]]` 可空）。各阶段自己的小节（`[sft]`、`[teacher]`、`[distill]`、`[dpo]`、`[grpo]`）由
  `zero.post.common.load_post_config(src, {名: dataclass}, overrides)` 解析，`src` 可以是路径或 dict（冒烟测试用）。评测配置只有 `[eval]`
  （`zero.eval.harness.load_eval_config`）。
- **训练循环**：SFT 与蒸馏复用 `Trainer`（`format = "sft"` 时读 `MaskedWindowLoader`；新增钩子 `Trainer.extra_metrics()`，
  蒸馏子类用它记 `ce` / `kd`）。DPO / GRPO / 在线蒸馏用 `zero.post.common.LoopState`（优化器、调度、裁剪、日志、checkpoint、续训），
  目前是**单进程**实现（CPU / 单卡），多卡未实现。
- **模型加载**：`zero.post.common.load_policy(path)` 同时支持 zero checkpoint（分词器路径从 meta.json 读）与 HF 目录。
- **SFT 数据文件**：`<shard_dir>/{train,val}.bin`（uint32，n_windows × (seq_len+1)）+ 同名 `.mask`（uint8）+ `.json` 元数据。
- **工具环境**（`zero/post/envs/tool_env.py`）：`TOOLS`、`execute_call`、`validate_arguments`、`Task`、`generate_tasks(n, seed, split)`、
  `dev_tasks(n)`（固定 dev 集，训练任务自动排除同问题）、`make_splits`、`reference_messages(task)`、
  `score_tool_calls(task, text) -> Reward`、`score_final_answer`、`run_episode(policy, task)`。冻结的 dev 集文件：`zero/eval/tasks/tool_dev.jsonl`。
- **损失函数**：`zero.post.dpo.dpo_loss`、`zero.post.grpo.group_advantages` / `grpo_loss`、`zero.post.distill.kd_loss` / `reverse_kl_loss`。
- **评测**：`zero.eval.harness`（`eval_multiple_choice`、`eval_exact_match`、`eval_tool_calls`、`run_eval`）、
  `zero.eval.bootstrap.paired_bootstrap` / `compare_to_opponent`、`zero.eval.report.write_report`、`zero.eval.bfcl`（尚未验证）。
- **GGUF**：`zero.export.gguf.convert_hf_to_gguf / build_llama_cpp / quantize / run_llama / llama_tokenize`。
  调用官方 `convert_hf_to_gguf.py` 前打一个运行时补丁：预切分规则不认识时按 `qwen2` 处理（正则相同，`tests/test_gguf.py` 用 `llama-tokenize` 对拍）。
- **demo**：`zero.demo.cli`（`chat_turn`、`search_files`、`execute_demo_tool`）。**冒烟测试**：`zero.smoke`。

## 补充说明（第 12–14 章后）

- **多来源混合与多卡**：`MixtureLoader` 按 `(seed, rank)` 各自抽签选来源，所以多来源数据下"2 卡 = 1 卡双倍 batch"不逐位成立（单来源时成立，见 tests/test_ddp_cpu.py）。每个 rank 的来源比例在期望上一致。
- **激活检查点**：`train.activation_checkpointing = true` 时每个 Block 只保存输入、反向时重算（tests/test_activation_checkpointing.py 保证梯度一致）；已在单张 RTX 3090 上验证（2026-10，见 runs/2026-10-01-gpu0-check/：CUDA 上开关前后梯度逐位相同，主线配置 T=4096 时 24GB 卡上不开放不下 micro batch 1、开了最多放 3）。
- **优化器**：`optim.name = "muon"` 切换到 `zero/train/muon.py`（二维权重用 Muon，其余 AdamW；第 12 章核实 Muon 满足共识规则，是否用于主线由阶梯实验决定）。
- **验证集 bits-per-byte**：配置了 `data.tokenizer` 的预训练评估会同时记录 `val_bpb`（`zero/data/bpb.py`）。
- **主线配置显存**：micro batch 4 × 累积 4（第 14 章 `zero/tools/memory_calc.py` 估算 micro batch 8 超过 80GB）。
