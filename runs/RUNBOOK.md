# 第二步运行手册（RUNBOOK）

> 适用范围：GOAL.md 第 10 节"第二步：训练主线模型"。第一步（本仓库当前状态）没有 GPU，下面所有 GPU 命令
> **都还没有在 GPU 上跑过**；CPU 上能测的部分已由 `uv run pytest` 和 `uv run python -m zero.smoke` 覆盖。
> 每个阶段跑完，把花费记进 [`runs/ledger.md`](ledger.md)。

## 0. 规矩

- **花钱要批准**：任何单次预计超过 **$100** 的运行，先把 `estimate_cost` 的输出和本手册对应小节的检查清单发给项目负责人，批准后再开跑（GOAL.md 3.4）。
- **每次运行一个目录**：`runs/<日期>-<阶段>-<简称>/`，放本次用的配置副本（`--print-config` 的输出）、日志摘要（`log.jsonl` 的关键行）、结论。大文件（checkpoint、数据）不进仓库。
- **先小后大**：每个新阶段先用 `--set train.max_steps=20` 跑通，确认 loss、吞吐、显存正常，再跑全量。
- **价格假设**：H100 SXM $2.5/卡时（GOAL.md 3.4），8 卡 = $20/小时；以实际租价为准，改 `--price`。
- **吞吐估算的口径**：下文的"预期吞吐"都由 `zero/tools/estimate_cost.py` 按 MFU 0.4 推出（H100 稠密 BF16 峰值 989.5 TFLOPS，待核实）。阶段 6 实测 MFU 后，用实测值重算全部预算。

预期吞吐一览（8×H100、MFU 0.4 的假设；实测前只是估计）：

| 阶段 | 序列长度 | 每 token FLOPs | 8 卡总吞吐 | 来源 |
|---|---:|---:|---:|---|
| 预训练 / 中期训练 | 4,096 | 6.955e9 | ≈ 455k token/s | `estimate_cost --config configs/main/pretrain.toml` |
| 长上下文扩展 | 32,768 | 2.669e10 | ≈ 119k token/s | `--config configs/main/longctx.toml` |
| SFT / 蒸馏训练 | 8,192 | 9.774e9 | ≈ 324k token/s（打包后约 70% 是真实 token） | `--seq-len 8192` |
| GRPO / DPO 采样 | — | — | 取决于生成，不能按训练 FLOPs 估；阶段 10 实测 | — |

（换算：总吞吐 = 8 × 989.5e12 × 0.4 / 每 token FLOPs。）

## 1. 环境准备（每台新机器做一次）

```bash
git clone <本仓库> && cd From-0-to-AGI
uv sync --group dev                       # torch、tokenizers、safetensors、transformers、pytest
uv run python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.device_count())"
nvidia-smi topo -m                        # 确认 8 卡 NVLink 全互联
uv run pytest -q                          # CPU 测试必须先全绿（约 3 分钟）
uv run python -m zero.smoke --out /tmp/smoke   # 端到端冒烟（CPU 单线程：空闲时约 9 分钟，CPU 被占用时更久）
```

- torch 需要带 CUDA 的版本（`uv pip install torch --index-url https://download.pytorch.org/whl/cu12x`，与驱动匹配）。
- 数据放本地 NVMe：`data/tokenizer/`、`data/pretrain/`、`data/midtrain/`、`data/sft/`……（路径见 `configs/main/*.toml`）。
- 评测：`uv pip install lm-eval==<冻结版本> bfcl-eval==<冻结版本> vllm==<冻结版本>`（版本号写进 `eval/PREREGISTRATION.md`）。
- 导出 GGUF：`uv run python -m zero.export.gguf ...` 会自动浅克隆并编译 llama.cpp（需要 `cmake`；`apt-get install -y cmake`）。

## 2. 阶段 6：GPU 验证运行（≤ $50）

目的：把"尚未在 GPU 上验证"的路径逐项验证掉，并实测 MFU。预算：8×H100 约 1.5 小时 ≈ $30，上限 $50。

> 2026-10：本节的单卡 / 2 卡 RTX 3090 版已经做过（[`2026-10-01-gpu0-check/`](2026-10-01-gpu0-check/README.md)：通路、跨卡一致性、续训都通过；32K 在 24GB 的卡上放不下）。8×H100 上仍要逐项跑一遍，重点是 NVLink 下的吞吐与 MFU、8 卡 FSDP 和 32K 的显存。

| # | 验证项 | 命令 | 通过标准 |
|---|---|---|---|
| 1 | 单卡 BF16 + SDPA 走 FlashAttention | `uv run python -c "import torch; from torch.nn.attention import sdpa_kernel, SDPBackend; from zero.config import load_model_config; from zero.model import Transformer; m=Transformer(load_model_config('configs/main/pretrain.toml')).cuda().bfloat16(); x=torch.randint(0,65536,(1,4096),device='cuda');\nwith sdpa_kernel(SDPBackend.FLASH_ATTENTION): print(m(x).shape)"` | 不报 "No available kernel"。**已核实**（RTX 3090，2026-10，见 [`2026-10-01-gpu0-check/`](2026-10-01-gpu0-check/README.md) 第 1 节）：BF16 下 `enable_gqa=True` 默认就走 Flash 后端。memory-efficient 后端**不支持 GQA**，所以不能"退回 efficient"；万一 H100 上 Flash 不可用，只能在 `Attention` 里先 `repeat_interleave` K/V 再走 efficient，并记录 MFU 差异 |
| 2 | 多卡 DDP | `uv run torchrun --standalone --nproc_per_node=8 -m zero.train.pretrain --config configs/main/pretrain.toml --set train.max_steps=200 --set checkpoint.every=100 --set train.out_dir=out/gpu_check/ddp` | loss 下降；各卡显存均衡；日志里 `tok/s` 与 MFU 稳定 |
| 3 | 断点续训 | 上一条跑到 ~150 步时 `kill`，再执行同一条命令 | 从 step 100 续训，第 101–200 步 loss 与不中断的对照运行一致（BF16 下允许 1e-3 级差异；数据顺序必须完全一致） |
| 4 | FSDP2 | 同 2，加 `--set train.parallel=fsdp`，另起目录 | 与 DDP 的前 50 步 loss 曲线一致（误差 < 1%）；checkpoint 能被单卡 `load_policy` 读回 |
| 5 | torch.compile | 同 2，默认 `compile = true` 与 `--set train.compile=false` 各跑 100 步 | 两者 loss 一致；记录 compile 带来的吞吐提升 |
| 6 | MFU 实测 | 取 2 的第 100–200 步日志 | 记录 `tok/s`、`mfu`；用实测 MFU 重跑 `estimate_cost --mfu <实测>`，更新本手册和预算 |
| 7 | 显存与 batch | `--set train.micro_batch_size=16 --set train.grad_accum_steps=1` 试探 | 找到不 OOM 的最大 micro batch，保持每步 token 数 = 524,288 |
| 8 | 长序列 | `torchrun ... -m zero.train.midtrain --config configs/main/longctx.toml --set train.max_steps=20`（init_from 指向 2 的 checkpoint，可临时 `--set train.init_from=out/gpu_check/ddp/ckpt`） | 32K 序列不 OOM（FSDP），吞吐与估算同量级 |
| 9 | 后训练通路 | `torchrun ... -m zero.post.sft --config configs/main/sft.toml --set train.max_steps=20 --set train.init_from=out/gpu_check/ddp/ckpt`（需要一份小 SFT JSONL）；`python -m zero.post.grpo --config configs/main/grpo.toml --set train.max_steps=2 --set grpo.prompts_per_step=4` | 不报错；GRPO 记录每步采样耗时，用来估算阶段 10 的预算 |
| 10 | 导出 | `load_policy` + `export_to_hf_qwen3(..., chat=True)`，再 `vllm serve out/gpu_check/hf --enable-auto-tool-choice --tool-call-parser hermes`，发一个带 tools 的请求 | vLLM 能加载；返回结构化的 `tool_calls`（模型乱答没关系，看格式通路） |

验证完：把实测 MFU、吞吐、显存写进 `runs/<日期>-gpu-check/README.md`，更新本手册顶部的吞吐表，并记账。

## 3. 阶段 7：对手重跑与预注册定稿（约 $200）

1. 按 `eval/opponents.md` 的规则确定对手清单与**冻结日期**；
2. 固定评测框架版本（lm-evaluation-harness、BFCL、ACEBench 等），写进 `eval/PREREGISTRATION.md`；
3. 对每个对手：官方模板、两种模式（思考 / 非思考，如有）、相同解码参数，跑全部基准，保存逐题结果；
   - 通用基准：`lm_eval --model vllm --model_args pretrained=<对手>,dtype=bfloat16 --tasks <冻结的任务> --batch_size auto --log_samples --output_path eval/results/<对手>`
   - BFCL：对手用 BFCL 内置的 handler 名：`bfcl generate --model <handler 名> --test-category <冻结的类别> --backend vllm`，再 `bfcl evaluate`；我们的模型用 `python -m zero.eval.bfcl`（见 `zero/eval/bfcl.py`，尚未验证）
4. 估算：约 25 个模型 × 每个 ~1 GPU·h ≈ $60–200，超过 $100 先报批；
5. **停下来等确认预注册内容**，确认后提交 `eval/PREREGISTRATION.md`（commit 时间即登记时间）。

## 4. 阶段 8：阶梯实验与闸门 1（约 $1,200）

```bash
for s in l20m l60m l150m l300m; do
  uv run torchrun --standalone --nproc_per_node=8 -m zero.train.pretrain --config configs/ladder/$s.toml
done
```

| 配置 | 参数量 | token | 估算卡时 | 估算费用 |
|---|---:|---:|---:|---:|
| l20m | 23.1M | 0.46B | 0.1 | $0.3 |
| l60m | 71.3M | 1.43B | 0.6 | $1.5 |
| l150m | 160.5M | 3.21B | 2.8 | $7 |
| l300m | 318.8M | 6.38B | 10.8 | $27 |

（`estimate_cost --config configs/ladder/<s>.toml --tokens <token>`，MFU 0.4；小模型的实际 MFU 通常更低，按阶段 6 实测修正。）
一轮阶梯很便宜，预算主要花在学习率扫描、数据配比消融（第 13 章）和配方验证 (b)（在一个同尺寸开源 Base 上跑一遍完整后训练）。

需要盯的指标：各尺寸的最终 val loss（拟合 L(N, D)）、loss 曲线是否平滑、有无 loss spike、tok/s。

**闸门 1 检查清单**（全部完成才申请预训练预算）：

- [ ] 阶梯 4 个尺寸跑完，val loss 与 (N, D) 的拟合残差 < 1%，外推出主线 Base 在约 400B token 的 loss（附置信区间）
- [ ] 外推出通用基准分数（用阶梯模型的"loss → 分数"关系；小模型接近随机的基准不外推，如实说明）
- [ ] 配方验证 (a)：后训练配方套在 2–3 个阶梯 Base 上，得到"Base 质量 → 工具调用得分"的关系并外推到主线
- [ ] 配方验证 (b)：后训练配方套在一个现成同尺寸开源 Base 上，看配方上限（只做验证，不发布）
- [ ] 预测结果是否达到硬目标（预注册的判定标准），写成闸门 1 报告
- [ ] 预训练预算（`estimate_cost --config configs/main/pretrain.toml --tokens 400B --mfu <实测>`）随报告一起提交
- [ ] **等批准**

## 5. 阶段 9：预训练、中期训练、长上下文与闸门 2（约 $5,700）

```bash
# 预训练（约 400B token，762,940 步；估算 1,952 GPU·h ≈ $4.9K @ MFU 0.4 —— 以实测 MFU 重算后报批）
uv run torchrun --standalone --nproc_per_node=8 -m zero.train.pretrain --config configs/main/pretrain.toml
# 中期训练 / 退火（≈ 26B token，估算 127 GPU·h ≈ $317）
uv run torchrun --standalone --nproc_per_node=8 -m zero.train.midtrain --config configs/main/midtrain.toml
# 长上下文扩展到 32K（≈ 4.2B token，估算 79 GPU·h ≈ $197）
uv run torchrun --standalone --nproc_per_node=8 -m zero.train.midtrain --config configs/main/longctx.toml
```

注意：最初按 500B token 估算约 $6.1K（MFU 0.4），超过 GOAL.md 3.4 给预训练的 ~$5K；第 12 章据此把默认值降到约 400B token（`max_steps = 762940`，MFU 0.4 时约 $4.9K）。闸门 1 仍需用阶梯实验和实测 MFU 在"token 数、MFU、租价"三者之间定稿。

需要盯的指标：

- `loss` / `val_loss`：平滑下降；与阶梯外推曲线对比；
- `grad_norm`：稳定在 0.2–1 附近，突然放大 10 倍以上是 spike 前兆；
- `tok_per_s`、`mfu`：掉速说明有卡慢（查 `nvidia-smi`、NCCL 日志）；
- `mixture_counts`：各来源实际采样比例与配置一致；
- checkpoint 每 2,000 步一个，`keep_last = 3`；每天抽一个 checkpoint 做少样本评测（`zero.eval.harness` 的选择题 + lm-eval 的小子集）。

故障处理：loss 变 NaN / 发散 → 训练循环会自动停下（`FloatingPointError`）；从上一个 checkpoint 续训，必要时跳过出问题的数据段（改 `data` 的种子或剔除分片）、降低学习率。机器被抢占 → 同一条命令重跑自动续训。

**闸门 2 检查清单**：

- [ ] Base 模型在预注册的 Base 基准上的实际分数 vs 闸门 1 的预测（逐项列出偏差）
- [ ] 明显偏低时先诊断（数据、学习率、评测模板），写诊断报告，再决定是否进入后训练
- [ ] 去污染检查：训练数据与全部评测集的 13-gram 重叠（`zero/data/decontam.py`），结果写进模型卡草稿
- [ ] 花费记账

## 6. 阶段 10：后训练、闸门 3 与发布（约 $1,500 + $200）

```bash
# SFT（≈ 1B 窗口 token，估算 6.9 GPU·h ≈ $17）
uv run torchrun --standalone --nproc_per_node=8 -m zero.post.sft --config configs/main/sft.toml
# 蒸馏：先起教师服务（许可证核实后在配置里填 name / version / license / license_allows_distillation）
vllm serve <教师> --served-model-name teacher --enable-auto-tool-choice --tool-call-parser hermes &
uv run python -m zero.post.distill --config configs/main/distill.toml
# DPO（单进程实现）
uv run python -m zero.post.dpo --config configs/main/dpo.toml
# GRPO（单进程实现；吞吐不够时先与 verl 对拍，再用 verl，见 zero/post/grpo.py 模块说明）
uv run python -m zero.post.grpo --config configs/main/grpo.toml
# 内部评测（tool_env dev + 玩具集，配对 bootstrap）
uv run python -m zero.eval.harness --config configs/main/eval.toml
# 导出
uv run python -c "from zero.post.common import load_policy; from zero.hf import export_to_hf_qwen3; m,t=load_policy('out/main/grpo/ckpt'); export_to_hf_qwen3(m,None,'out/main/hf_final',tokenizer=t,chat=True)"
uv run python -m zero.export.gguf --hf-dir out/main/hf_final --out out/main/zero-f16.gguf --quantize Q4_K_M --run "<|im_start|>user\n你好<|im_end|>\n<|im_start|>assistant\n"
```

预算说明：教师数据生成的成本取决于教师大小和样本数（vLLM 吞吐实测后估算）；GRPO 的成本主要是采样，阶段 6 第 9 项实测每步耗时后，用"每步秒数 × 步数 × $20/小时"估算，超过 $100 报批。

需要盯的指标：

- SFT / 蒸馏：只在助手 token 上的 `loss`、`val_loss`；蒸馏数据的执行验证通过率（`teacher.jsonl.meta.json`）；
- DPO：`loss` 从 0.693 下降、`acc`（隐式奖励 chosen > rejected 的比例）、`margin`；`chosen_reward` 也在下降说明在"一起压低"，要警惕；
- GRPO：`reward_mean`、`format_rate`、**`call_rate`**（冒烟测试里出现过"不再调用工具"的作弊，见 `tool_env.py` 第 8 条）、`resp_len`、`kl`、`clip_frac`、`zero_std_groups`（太高说明任务太难或太简单）；
- 每个阶段结束都跑一次 `zero.eval.harness` 和 BFCL 子集，任何一项明显退化就回退到上一阶段的 checkpoint。

**闸门 3 检查清单**（发布前）：

- [ ] 按 `eval/PREREGISTRATION.md` 冻结的基准、框架版本、模板、解码参数跑完全部评测，保存逐题结果
- [ ] 与每个对手做配对 bootstrap（`zero.eval.bootstrap`，对手取思考 / 非思考中较高者），按预注册的判定标准给出"超过 / 持平 / 落后"
- [ ] 没达到硬目标就不宣称"超过"，如实写差距分析
- [ ] 去污染：训练数据（含教师合成数据）与评测集的 n-gram 重叠、工具函数名 / schema 与 BFCL 的重合检查
- [ ] 模型卡：数据与许可证、各阶段配方与花费、预注册、全部评测结果（含落后项）、去污染结果、已知局限
- [ ] GGUF 在笔记本上用 llama.cpp 跑通（Q4_K_M），本地 demo（`python -m zero.demo.cli --model out/main/hf_final`）录屏
- [ ] 发布后再查冻结日之后的新模型，写"发布后新增对手"
- [ ] **等最后确认再发布**

## 7. 预算对照（GOAL.md 3.4）

| 用途 | GOAL 预算 | 当前估算（MFU 0.4） | 备注 |
|---|---:|---:|---|
| 阶段 6 GPU 验证 | （预留内） | ≤ $50 | |
| 第 11 章对手重跑 | ~$200 | $60–200 | 取决于对手数量 |
| 第 12–13 章阶梯、消融、配方验证 | ~$1,200 | 阶梯本身 < $40，主要是消融 | |
| 第 14 章预训练 | ~$5,000 | ~$4,900（400B token，MFU 0.4） | 需实测 MFU 后定稿 |
| 第 15 章中期训练 + 长上下文 | ~$700 | ~$514 | |
| 第 16–19 章后训练 | ~$1,500 | SFT ~$17 + 蒸馏/DPO/GRPO 待实测 | |
| 第 20 章最终评测与发布 | ~$200 | — | |
| 第五部分架构实验 | ~$400 | — | 可选 |
