# 阶段 6（RTX 3090 版）：在单卡与 2 卡 RTX 3090 上验证 zero 的 GPU 路径

[English](README.md) · **中文**

日期：2026-10-01。分支：`gpu0-verification`。本次运行对应 [`runs/RUNBOOK.zh.md`](../RUNBOOK.zh.md) 第 2 节"阶段 6：GPU 验证运行"。
RUNBOOK 的阶段 6 按 8×H100 编写。第 0–12 节在**一张** RTX 3090 上做每一项的"单卡版本"。
之后又预约到另外两张卡，第 14 节在 **2×RTX 3090（PCIe，无 NVLink）** 上补做了跨卡的项。仍然需要更多卡或 H100 的项列在第 13 节。我们不把这些项写成已完成。

> **注意：**本文件只管 `zero/`。同一轮检查还在本机 CPU 上复现了各章的极简代码，并新增了"GPU 实测"。这些结果记录在同一目录：
> [chapters-01-06.zh.md](chapters-01-06.zh.md)、[chapters-07-10.zh.md](chapters-07-10.zh.md)、[chapters-11-15.zh.md](chapters-11-15.zh.md)、
> [chapters-16-20.zh.md](chapters-16-20.zh.md)、[chapters-21-23.zh.md](chapters-21-23.zh.md)、[chapters-24-26.zh.md](chapters-24-26.zh.md)。

## 硬件与软件

| 项 | 值 |
|---|---|
| GPU | **物理 GPU0**：NVIDIA GeForce RTX 3090，24 GB（23.57 GiB 可用），sm_86，PCIe；**功耗上限 240 W**（出厂默认 350 W） |
| 多卡（第 14 节） | 物理 GPU0 + **GPU2**（同型号，功耗上限同为 240 W）。32K 显存试验另加 GPU1（功耗上限 200 W）。三张卡之间没有 NVLink，`nvidia-smi topo -m` 显示 NODE（走 PCIe / 主机桥） |
| 驱动 / CUDA | 570.195.03 / CUDA 12.8 |
| PyTorch | 2.11.0+cu128（cuDNN 9.19.0），Python 3.12.12，transformers 5.17.0 |
| CPU | 128 核（用于冒烟测试等单线程任务） |
| 峰值算力口径 | RTX 3090 稠密 BF16 Tensor Core（FP32 累加）**71 TFLOPS**（GA102 白皮书，按 1695 MHz boost 频率）。zero 的峰值表里没有 3090。所以本目录的配置都设 `train.gpu_peak_tflops = 71`，让日志直接按 3090 报告 MFU |

**第 0–12 节的每一个 GPU 数字都来自物理 GPU0**（`CUDA_VISIBLE_DEVICES=0`，torch 只看到 1 张卡；每条命令前都确认过 `torch.cuda.device_count() == 1`）。
这些命令都用文件锁 `flock <scratchpad>/gpu0.lock` 与其他任务排队，并独占整张卡。第 14 节的多卡运行用 `CUDA_VISIBLE_DEVICES=0,2`（32K 试验用 `0,1,2`），同时持有各卡的锁。
GPU1、GPU2 是项目负责人在验证过程中追加预约的。

## 先读：三条口径

1. **数据**：没有下载预训练数据。训练用仓库自带的 `assets/tiny_corpus`（莎士比亚 / 唐诗 / 代码，约 93.5 万训练 token，分词器的词表为 2048）。`zero.data.prepare` 现场把它生成到 `out/gpu0-check/tiny/`。模型的词表仍是 65,536（与阶梯 / 主线相同），所以 logits 的显存和算力开销是真实的。loss 数值只说明"在下降"，不说明任何模型质量。
   配置副本在 [`configs/`](configs/)。`*_tiny.toml` 继承仓库里的阶梯 / 主线配置，只换数据。`*.resolved.json` 是 `--print-config` 的完整输出。其余覆盖项都写在下面各节的命令里。
2. **MFU**：zero 按 `6·N_matmul + 12·L·q_dim·T` 计算每个 token 的 FLOPs（`zero/model.py`）。注意力项**不因因果 mask 减半**，也不计激活检查点的重算。
   T=4096 的主线配置里，注意力项占 40%，而 FlashAttention 的因果内核实际只算一半。所以长序列下这个 MFU 偏高。另外 3090 的功耗上限是 240 W，实际频率可能低于白皮书的 1695 MHz。
   **这些 MFU 只能说明"3090 跑得多满"，不能代表 H100 NVLink 的情况，也不能直接替换 RUNBOOK 里的 0.4。**
3. **GPU 默认内核在两次运行之间不确定**（FlashAttention 反向传播里 dQ 的原子加、embedding 反向传播里的 index_add 等）。
   同一条命令跑两次，loss 从第 2 步起就差 1e-5 量级。阶梯配置的学习率大（2e-3–3e-3），前 20–30 步会把差异放大到单步 0.4–0.7。所以凡是"对比两次运行的 loss"的项，都另用
   [`det_pretrain.py`](det_pretrain.py) 跑一遍（`torch.use_deterministic_algorithms(True)`，不改 zero 的代码）。打开后，两次运行逐位相同，吞吐只慢约 7%。

大文件（checkpoint、完整日志、各脚本的 JSON 结果）在 `out/gpu0-check/`（.gitignore 忽略这个目录）。

---

## 0. 基线

**单元测试**

```bash
CUDA_VISIBLE_DEVICES= UV_NO_SYNC=1 uv run pytest -q                                   # CPU
flock <gpu0.lock> env CUDA_VISIBLE_DEVICES=0 UV_NO_SYNC=1 PYTHONPATH=runs/2026-10-01-gpu0-check \
    uv run pytest -q -p cuda_test_probe                                               # GPU 可见
```

| 时间点 | CPU（`CUDA_VISIBLE_DEVICES=`） | GPU0 可见 |
|---|---|---|
| 开始时 | 319 passed, 5 skipped（llama.cpp 未编译），约 30 s | 319 passed, 5 skipped，结果相同 |
| 结束时（含本次新增的 4 个测试） | **321 passed, 7 skipped**（3 个新测试需要 CUDA，因此跳过；冒烟测试克隆了 llama.cpp 仓库，所以有一个 gguf 测试不再跳过） | **324 passed, 4 skipped** |

[`cuda_test_probe.py`](cuda_test_probe.py)（pytest 插件）统计每个测试有没有分配 CUDA 显存。开始时**没有任何测试走 CUDA**（所有测试都写死 `device="cpu"`）。GPU 可见时，唯一的区别是 `checkpoint.rng_state()` 会多存一份 CUDA 随机数状态。
结束时，只有本次新增的 3 个 CUDA 回归测试分配了显存（见第 12 节）。

**冒烟测试** `zero.smoke`：`configs/tiny/*.toml` 全部写死 `device = "cpu"`、`dtype = "fp32"`，`pick_device("cpu")` 直接返回 CPU。所以**在有 GPU 的机器上跑 `zero.smoke`，训练仍然全在 CPU 上**（只初始化了 CUDA 上下文）。

```bash
flock <gpu0.lock> env CUDA_VISIBLE_DEVICES=0 UV_NO_SYNC=1 uv run python -m zero.smoke --out out/gpu0-check/smoke
```

为了真正走 CUDA，[`smoke_cuda.py`](smoke_cuda.py) 在读配置时把 `[train] device` 换成 `cuda`、`dtype` 换成 `auto`（CUDA 上即 BF16 autocast）。其余部分与 `zero.smoke` 完全相同（eval / export / demo 在 zero.smoke 里本来就在 CPU 上）：

```bash
flock <gpu0.lock> env CUDA_VISIBLE_DEVICES=0 UV_NO_SYNC=1 uv run python runs/2026-10-01-gpu0-check/smoke_cuda.py --out out/gpu0-check/smoke_cuda
```

| 阶段 | `zero.smoke`（实际在 CPU，单线程） | `smoke_cuda.py`（GPU0，BF16，修复蒸馏的设备 bug 之后） |
|---|---|---|
| data+tokenizer | ok，4.7 s | ok，4.0 s |
| pretrain（200 步） | ok，80.7 s，loss 5.338 / val 5.482 | ok，6.6 s，loss 5.257 / val 5.457 |
| midtrain（60 步） | ok，27.4 s，val 5.453 | ok，1.5 s，val 5.431 |
| SFT（240 步） | ok，290.6 s，val 0.624 | ok，6.5 s，val 0.621 |
| 蒸馏（20 步） | ok，20.6 s，kd 0.0096 | ok，5.4 s，kd 0.0129（学生、教师都在 cuda:0） |
| DPO（24 步） | ok，23.9 s，acc 0.75 | ok，5.7 s，acc 0.75 |
| GRPO（10 步） | ok，57.5 s；每步采样 1.43 s、整步 5.74 s | ok，9.9 s；每步采样 0.91 s、整步 0.98 s |
| eval / export / demo | ok，21.8 / 12.7 / 0.0 s；HF logits 差 0.0，模板逐字相同 | ok，17.9 / 6.3 / 0.0 s；同左 |
| **合计** | **540 s（10 个阶段全部通过）** | **64 s（10 个阶段全部通过）** |

## 1. BF16 + SDPA 走 FlashAttention（RUNBOOK #1）

```bash
flock <gpu0.lock> env CUDA_VISIBLE_DEVICES=0 UV_NO_SYNC=1 uv run python runs/2026-10-01-gpu0-check/sdpa_check.py
```

主线形状（16 个查询头、8 个 K/V 头、head_dim 128、T=4096、BF16、因果），前向 + 反向传播。误差以 FP32 math 为基准：

| 写法 | FLASH | EFFICIENT | CUDNN | MATH |
|---|---|---|---|---|
| `enable_gqa=True`（zero 的写法） | **能用**，4.65 ms，409 MiB | **No available kernel** | 能用，4.59 ms | 55.7 ms，4432 MiB |
| 先 `repeat_interleave` K/V | 能用，5.56 ms | 17.0 ms | 5.06 ms | 55.0 ms |

- 所有能跑的组合与 FP32 math 的最大绝对误差都是 7.5e-3（BF16 舍入）。
- **不加任何限制时，PyTorch 选的就是 Flash**：profiler 里看到的 kernel 是 `pytorch_flash::flash_fwd_kernel` /
  `flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel`（`enable_gqa=True` 与先复制两种写法都是）。
- `torch.backends.cuda.can_use_*_attention(enable_gqa=True)`：flash True、cudnn True、**efficient False**
  （警告原文："For dense input, both fused kernels require query, key and value to have the same num_heads"）。
- RUNBOOK 的原命令（689.5M 模型 `.cuda().bfloat16()`，在 `sdpa_kernel(FLASH_ATTENTION)` 下前向 1×4096）：
  **通过**，输出 `[1, 4096, 65536]`。同样的命令换成 EFFICIENT，报 "No available kernel"。

整个模型的训练步（FP32 主权重 + BF16 autocast，与 Trainer 相同；micro batch 1 × 4096，前向 + 反向传播，不含优化器）：

| 注意力 | 不开激活检查点 | 开激活检查点 |
|---|---|---|
| 默认（`enable_gqa=True`，自动选到 Flash） | 5,872 tok/s，MFU 57.5%，19.5 GiB | 4,640 tok/s，45.5%，9.6 GiB |
| 强制 Flash（`enable_gqa=True`） | 5,817 tok/s，57.0% | 4,627 tok/s，45.3% |
| 先 `repeat_interleave` 再 Flash | 5,722 tok/s，56.1%，20.0 GiB | 4,538 tok/s，44.5% |
| 先 `repeat_interleave` 再 efficient | 3,876 tok/s，38.0% | （脚本限制，见下） |
| 强制 efficient（`enable_gqa=True`） | No available kernel | No available kernel |

结论：**通过**。在 3090（sm_86）+ torch 2.11 上，`enable_gqa=True` 直接走 FlashAttention-2 后端，比先复制 K/V 快约 2%。`Attention` 不需要改。
只有一点要注意：efficient 后端不支持 `enable_gqa`。如果以后强制 efficient（或 Flash 因形状不可用而回退），程序会直接报错，而不是悄悄变慢。
（表中"脚本限制"那一格的失败是脚本的问题。`sdpa_kernel` 只包住了前向传播。激活检查点在反向传播重算时回到默认的 Flash 内核，保存的张量形状不同，`torch.utils.checkpoint` 报 metadata 不一致。zero 本身不用 `sdpa_kernel`，不受影响。）

## 2. 训练通路（RUNBOOK #2 的单卡版）

```bash
flock <gpu0.lock> bash runs/2026-10-01-gpu0-check/train_run.sh pretrain_l20m runs/2026-10-01-gpu0-check/configs/l20m_tiny.toml \
    --set train.max_steps=200 --set schedule.warmup_steps=20 --set checkpoint.every=100 --set train.eval_every=100 --set logging.every=10
# l60m 同上；l150m 另加 --set train.micro_batch_size=4 --set train.grad_accum_steps=4 --set train.max_steps=100
```

[`train_run.sh`](train_run.sh) 就是 `uv run torchrun --standalone --nproc_per_node=1 -m zero.train.pretrain --config ...`。
它同时每 0.5 s 用 nvidia-smi 记一次显存。每步 32,768 token（= 阶梯配置单卡的份额）。

| 配置 | 参数 | micro × 累积 | loss（step 1 → 末步） | val（step 100 → 200） | tok/s（稳定后） | MFU（3090） | 显存峰值（nvidia-smi） |
|---|---:|---|---|---|---:|---:|---:|
| l20m | 23.1M | 8 × 2 | 11.135 → 5.429 | 6.221 → 5.627 | ≈ 98,500 | 26% | 17.1 GB |
| l60m | 71.3M | 8 × 2 | 11.175 → 5.354 | 6.218 → 5.627 | ≈ 45,600 | 37% | 21.6 GB |
| l150m | 160.5M | 4 × 4（100 步） | 11.227 → 6.243 | 6.316（step 100） | ≈ 24,400 | 43% | 16.8 GB |
| l150m | 160.5M | 8 × 2 | — | — | — | — | **OOM**（FP32 logits 8×2048×65536 要 4 GiB） |

关键日志行：`Model parameters 23.07M (non-embedding 6.30M), 32768 tokens per step, 200 steps, device cuda:0, world_size=1`；
`step 200/200 | loss 5.4290 | lr 0.00e+00 | gnorm 0.22 | 97,961 tok/s | MFU 26.0% | val 5.6273`。

结论：**单卡通路通过**（BF16 autocast、fused AdamW、梯度累积、WSD 调度、验证集 loss / bpb、checkpoint 都正常，loss 平滑下降）。
注意：`torchrun --nproc_per_node=1` 时 `WORLD_SIZE=1`，`wrap_model` **不包 DDP**。所以这一项**没有**验证 DDP。
单卡上的 DDP 包装见第 4 节，跨卡 DDP 见第 14 节。阶梯配置默认的 micro batch 16 × 2048 是按 80 GB 卡定的。24 GB 卡上，l20m / l60m 用 8、l150m 用 4 才放得下。

## 3. 断点续训（RUNBOOK #3 的单卡版）

```bash
flock <gpu0.lock> bash runs/2026-10-01-gpu0-check/resume_check.sh                                        # 默认内核
flock <gpu0.lock> env DET=1 STEPS=100 EVERY=50 KILL_AT=75 TAG=resume_det bash runs/2026-10-01-gpu0-check/resume_check.sh   # 确定性算法
```

[`resume_check.sh`](resume_check.sh) 用 l20m。它先跑一遍不中断的对照，再用同一条命令跑一遍。日志出现 step KILL_AT 时，脚本
`kill -9` torchrun 和 worker。（torchrun 用 `start_new_session` 启动 worker，杀进程组杀不到，所以脚本按命令行里唯一的输出目录 `pkill -9`。）
然后原样重跑，程序自动从最近的 checkpoint 续训。

| 对比 | 默认内核（200 步，每 100 步存，第 150 步 kill） | 确定性算法（100 步，每 50 步存，第 75 步 kill） |
|---|---|---|
| 续训日志 | `Resumed training from out/gpu0-check/resume/kill/ckpt/step_00000100 (step 100)` | `Resumed training from .../resume_det/kill/ckpt/step_00000050 (step 50)` |
| **同一次运行**：续训后 vs 被杀前（起点是同一个 checkpoint） | 第 101–108 步**逐位相同**，101–150 步最大差 1.1e-3 | 51–75 步**全部逐位相同** |
| 续训 vs 不中断的对照 | 最大差 0.49（两次独立运行本身在 1–150 步就差到 2.09，见"口径"第 3 条） | 51–100 步**全部逐位相同**，step 100 val_loss 6.207895 = 6.207895 |
| 学习率逐步相同 | 是 | 是 |

结论：**通过**。在确定性算法下，kill -9 之后续训与不中断的运行逐位相同（模型、AdamW 状态、WSD 调度、数据加载器位置、随机数状态全部精确恢复）。默认内核下，续训后前 8 步也逐位相同。之后的差异来自 GPU 内核本身的非确定性，量级与两次独立运行相同。
`checkpoint.py` 里 CUDA 随机数状态的保存 / 恢复另有直接测试：`rng_state()` → 抽 1000 个 CUDA 随机数 → 打乱 →
`set_rng_state()` → 再抽，两次结果逐位相同（`parity_cuda.py` 第 6 部分）。（训练循环本身不消耗 CUDA 随机数，所以续训一致性不依赖它。）

## 4. FSDP2 与 DDP 的单卡通路（RUNBOOK #4 的单卡版）

`torchrun --nproc_per_node=1 ... --set train.parallel=fsdp` **什么都不做**：`WORLD_SIZE=1` 时 `DistInfo.is_distributed`
为 False，`wrap_model` 原样返回模型，DDP 也一样。所以这里用 [`dist1_check.py`](dist1_check.py) 自己建一个
world_size=1 的 NCCL 进程组，并把 `is_distributed` 强制为 True。这样 `wrap_model` 与 `checkpoint.py` 的 DDP / FSDP2 分支才真的执行：

```bash
flock <gpu0.lock> env CUDA_VISIBLE_DEVICES=0 UV_NO_SYNC=1 uv run python runs/2026-10-01-gpu0-check/dist1_check.py
```

l20m + tiny 数据，50 步，确定性算法；plain = 不包装（与 torchrun 单卡相同）：

| 对比 | lr 3e-3（阶梯默认） | lr 3e-4 |
|---|---|---|
| DDP（`DistributedDataParallel`，NCCL，含 `no_sync` 梯度累积）vs plain | **逐位相同** | **逐位相同** |
| FSDP2（`fully_shard` + `MixedPrecisionPolicy(bf16, fp32 规约)`）vs plain，逐步 loss 最大相对差 | 0.57% | 0.11% |

- FSDP2 的 checkpoint（`get_model_state_dict` / `get_optimizer_state_dict` 聚合成完整 state_dict）：
  删掉 step 25 之后的 checkpoint，从 step 25 续训到 50（`set_model_state_dict` / `set_optimizer_state_dict`）。loss **与不中断的运行逐位相同**。
- 单卡 `load_policy` 能读回 FSDP2 的 checkpoint（strict 加载 91 个张量）。在真实验证数据上 CE 为 6.3398，
  plain 同一步的 checkpoint 为 6.3417，相对差 3.1e-4。
- 脚本跑了两遍（中间只改了 load_policy 的检查方式），两遍的全部 loss 数字完全相同。确定性算法下结果可复现。
- 三种包装的显存峰值为 14.5–14.9 GiB（单卡上 FSDP 没有可切的对象，只是走通路）。

结论：**单卡通路通过**（满足误差 < 1% 的标准）。这说明下面这些在 GPU 上都能跑：`fully_shard` 的调用方式、混合精度策略、fused AdamW 与 DTensor 参数、
`clip_grad_norm_`、FSDP 的 state_dict 聚合与恢复。**真正的切分（world_size > 1）和跨卡一致性仍需多卡。**

## 5. torch.compile（RUNBOOK #5 的单卡版）

```bash
flock <gpu0.lock> bash runs/2026-10-01-gpu0-check/compile_check.sh             # 关 ×2、开 ×1（默认内核）
flock <gpu0.lock> env PART=det bash runs/2026-10-01-gpu0-check/compile_check.sh  # 确定性算法下关、开各 1 次
```

l60m + tiny 数据，60 步（warmup 10，`logging.every=1`，`checkpoint.every=0`）：

| | 关 compile（a） | 关 compile（b） | 开 compile | 关（确定性） | 开（确定性） |
|---|---:|---:|---:|---:|---:|
| tok/s（21–60 步平均） | 46,984 | 46,509 | **64,363** | 43,800 | 61,049 |
| MFU（3090） | 38.3% | 37.9% | **52.5%** | 35.7% | 49.8% |
| 第 1 步耗时（含编译） | 1.1 s | 1.1 s | 30.7 s | 1.2 s | 32.2 s |
| step 60 loss | 6.264 | 6.209 | 6.357 | 6.249 | 6.304 |

- loss 对比：同一批数据、同一份初始权重，第 1 步 compile 与 eager 只差 1.3e-5（融合带来的舍入差）。
  60 步逐步 loss 的平均相对差：开 vs 关 0.91%，**关 vs 关（两次 eager 本身）1.12%**，确定性算法下开 vs 关 0.66%。
  编译带来的差异不超过两次 eager 运行之间的噪声。
- 主线 689.5M 形状（T=4096，激活检查点，micro batch 3）：eager 4,909 tok/s → compile **6,944 tok/s（+41%）**。
  前 3 步 loss 与 eager 相差 ≤ 2e-4（`mem_probe.py`，见第 7 节）。

结论：**单卡通过**。compile 在单卡上带来 +37%（l60m）/ +41%（主线形状）的吞吐。compile 与 DDP / FSDP 的组合没有验证。

## 6. MFU 实测汇总（RUNBOOK #6，单卡 3090 版）

| 模型 / 序列 | 设置 | tok/s | MFU（÷71 TFLOPS） | 来源 |
|---|---|---:|---:|---|
| l20m / 2048 | eager，8 × 2 | 98,500 | 26% | 第 2 节 |
| l60m / 2048 | eager，8 × 2 | 45,600–47,000 | 37–38% | 第 2、5 节 |
| l60m / 2048 | **compile**，8 × 2 | 64,400 | 52% | 第 5 节 |
| l150m / 2048 | eager，4 × 4 | 24,400 | 43% | 第 2 节 |
| 主线 689.5M / 4096 | eager，激活检查点，micro 3 | 4,909 | 48% | 第 7 节 |
| 主线 689.5M / 4096 | **compile**，激活检查点，micro 3 | 6,944 | 68% | 第 7 节 |
| 主线形状 / 8192 | eager，激活检查点，micro 1 | 4,122 | 57% | 第 8 节 |

读法：模型越小，lm_head（65,536 词表）和逐元素运算的比例越大，MFU 越低。主线尺寸在 3090 上，eager 约一半，compile 约三分之二。
（这是按 zero 的 MFU 口径：注意力项未减半，所以偏高，见"口径"第 2 条。）这些数字**不能**直接换算到 H100：
H100 的算力 / 带宽比、NVLink 和 8 卡通信开销都不同。RUNBOOK 的 MFU 0.4 仍需在 H100 上实测。

## 7. 显存与 batch（RUNBOOK #7，主线 689.5M、T=4096、单卡 24 GB）

```bash
flock <gpu0.lock> env CUDA_VISIBLE_DEVICES=0 UV_NO_SYNC=1 uv run python runs/2026-10-01-gpu0-check/mem_probe.py main
```

[`mem_probe.py`](mem_probe.py) 每次试探起一个新进程，用真正的 `Trainer` 跑 3 步（前向 + 反向传播 + fused AdamW，`stop_at=3`）。
配置是 [`configs/main_tiny.toml`](configs/main_tiny.toml)（主线形状 + tiny 数据）。micro batch 从 1 往上加，直到 OOM。
估算列是 `zero.tools.memory_calc.estimate_memory(..., num_gpus=1, strategy="ddp", dtype="bf16")`（单卡时没有 DDP 桶）。

| 激活检查点 | compile | micro batch | 结果 | PyTorch 分配峰值 | reserved 峰值 | memory_calc 估算 | tok/s | MFU |
|---|---|---:|---|---:|---:|---:|---:|---:|
| 关 | 关 | 1 | **OOM** | 22.13 GiB（OOM 时） | 23.08 | 24.02 | — | — |
| 开 | 关 | 1 | ok | 11.55 GiB | 13.66 | 13.92 | 4,578 | 44.8% |
| 开 | 关 | 2 | ok | 15.12 GiB | 18.44 | 17.37 | 4,812 | 47.1% |
| 开 | 关 | 3 | ok | 18.71 GiB | 22.98 | 20.82 | 4,909 | 48.1% |
| 开 | 关 | 4 | **OOM**（要再分配 4 GiB 的 FP32 logits） | — | — | 24.27 | — | — |
| 开 | 开 | 3 | ok | 18.65 GiB | 21.92 | 20.82 | 6,944 | 68.0% |

- 24 GB 卡上：**不开激活检查点，连 micro batch 1 都放不下；开了最多 3**。memory_calc 的"放不放得下"判断全部正确
  （`max_micro_batch(..., gpu_gib=23.56)` 给出 0 和 3）。
- memory_calc 比 PyTorch 实际分配的峰值高 2.1–2.4 GiB（保守上界，符合它文档里的口径）。micro batch 3 时，缓存分配器的 reserved
  峰值比估算再高 2.2 GiB。所以它文档里"再留 2–5 GB 余量"的建议是必要的。
- 在 CUDA 上，开激活检查点时的梯度与不开时**逐位相同**（FP32 与 BF16 autocast 都是，`parity_cuda.py` 第 3 部分）。
  主线形状下，吞吐代价约 21%（第 1 节：5,872 → 4,640 tok/s）。
- 显存大头是 lm_head：micro batch 4 失败时，要分配的正是 4 × 4096 × 65,536 的 FP32 logits（4 GiB）。分块计算交叉熵
  （不一次性生成全部 FP32 logits）能明显省显存。这是代码改动，没有做，只记在这里。
- 对 80 GB 卡的含义（推断，未实测）：在 T=4096、单卡下，memory_calc 估算 micro 4 不开检查点约 61 GiB，与主线配置的选择一致。
  RUNBOOK #7 试探"micro 16 × 累积 1"。按 memory_calc（8 卡 DDP），它不开检查点约 214 GiB，开检查点约 68 GiB。
  再加上本节看到的 reserved 余量，80 GB 上开检查点也很紧，需要在 H100 上实测。

## 8. 长序列 32K（RUNBOOK #8 的单卡版）

```bash
flock <gpu0.lock> env CUDA_VISIBLE_DEVICES=0 UV_NO_SYNC=1 uv run python runs/2026-10-01-gpu0-check/mem_probe.py longctx
```

配置是 [`configs/longctx_tiny.toml`](configs/longctx_tiny.toml)，它继承 `configs/main/longctx.toml`（θ=1e6、max_seq_len 32768）。
它不从 midtrain checkpoint 加载，只测显存。单卡时 `parallel = "fsdp"` 不起作用（见第 4 节）。

| 序列长度 | 激活检查点 | 结果 | PyTorch 分配峰值 | memory_calc 估算 |
|---:|---|---|---:|---:|
| 32,768 | 开 | **OOM**（要再分配 8 GiB：FP32 logits 32768 × 65536） | 19.57 GiB（OOM 时） | 38.07 |
| 32,768 | 关 | **OOM** | 21.98 GiB（OOM 时） | 111.28 |
| 16,384 | 开 | **OOM**（要再分配 4 GiB） | 17.19 GiB（OOM 时） | 24.27 |
| 8,192 | 开 | ok，loss 11.41 → 11.29，4,122 tok/s，MFU 56.8% | 15.15 GiB（reserved 18.51） | 17.37 |

结论：**如实记录：32K 在单张 24 GB 卡上放不下，即使 micro batch 1 + 激活检查点也不行**。单卡能跑的上限是 8K。
瓶颈是 logits（32K × 65,536 × BF16/FP32 多份 ≈ 20 GiB），不是注意力（Flash 内核的显存与 T 成线性）。
RUNBOOK 设想用 8×H100 + FSDP。FSDP 只切参数 / 梯度 / 优化器状态，**切不掉每卡自己的 logits**。按 memory_calc，
32K、micro 1、开检查点时，8 卡 DDP 每卡约 40.6 GiB，8 卡 FSDP 每卡约 29.1 GiB。这在 80 GB 上应该放得下，但需要实测。
如果以后要在更小的卡上做 32K，需要分块交叉熵。

## 9. 后训练通路（RUNBOOK #9 的单卡版）

- **SFT / 蒸馏 / DPO / GRPO 在 CUDA + BF16 autocast 下都能跑**：见 `smoke_cuda.py`（第 0 节的表）。各阶段的 loss / 奖励
  与 CPU 冒烟测试同量级。修复了一个 bug：蒸馏的学生原来**固定在 CPU 上训练**（第 12 节）。修复后日志为
  `Model parameters 1.31M ..., 20 steps, device cuda:0`。
- **torchrun 入口的 SFT**（[`post_check.sh`](post_check.sh) 第 1 段）：

  ```bash
  uv run torchrun --standalone --nproc_per_node=1 -m zero.post.sft --config configs/tiny/sft.toml \
      --set train.device=cuda --set train.dtype=auto --set train.max_steps=20 ...（路径指向 smoke_cuda 的产物）
  ```

  `Loaded model weights from out/gpu0-check/smoke_cuda/midtrain/ckpt (step 60)`；`step 20/20 | loss 2.4487 | ... | val 2.6198`。通过。
- **GRPO 每步采样耗时**：tiny 模型（1.3M，G=8，每步 4 题，64 个新 token）每步采样 0.91 s（CPU 上 1.43 s）。
  小模型受 Python 逐 token 循环限制，GPU 快不了多少。
  主线 689.5M 的采样速度用 [`grpo_sample_bench.py`](grpo_sample_bench.py) 测（随机权重；按 `configs/main/grpo.toml`
  取 G=16、`max_new_tokens=512`；提示词 300 token；每条都生成满 512）：

  | 写法 | 一组（16 条 × 512 token） | 解码吞吐 | 显存 | 换算：每步 64 题 |
  |---|---:|---:|---:|---:|
  | FP32（`run_grpo` 现在的写法：`sample_group` 不在 autocast 里） | 17.9 s | 457 tok/s | 6.7 GiB | ≈ 1,150 s |
  | BF16 autocast 包住 generate | 15.0 s | 546 tok/s | 7.3 GiB | ≈ 960 s |
  | 模型 `.bfloat16()`（KV cache 也是 BF16） | 13.7 s | 598 tok/s | 3.3 GiB | ≈ 880 s |

  在一张 3090 上，主线 GRPO 一步光采样就要 15–19 分钟（500 步约 130–160 小时）。原因是自回归解码逐个提示词进行，batch 只有 G=16。
  这种解码完全受延迟 / 带宽限制。H100 会快几倍，但结构性问题不变：**阶段 10 做 GRPO 之前，应该把 64 个提示词
  拼成一个大 batch 采样，或者按 zero/post/grpo.py 模块说明直接换 vLLM / verl**（本次没有改代码）。
- vLLM、lm-eval、bfcl：**跳过：未安装**（按要求不安装）。

## 10. 导出（RUNBOOK #10 的单卡版）

```bash
uv run python runs/2026-10-01-gpu0-check/export_check.py out/gpu0-check/smoke_cuda/grpo/ckpt out/gpu0-check/export_grpo
uv run python runs/2026-10-01-gpu0-check/export_check.py out/gpu0-check/pretrain_l20m/ckpt out/gpu0-check/export_l20m
```

[`export_check.py`](export_check.py)：`load_policy(ckpt, device="cuda")`（模型直接在 GPU 上）→ `export_to_hf_qwen3(..., chat=True)`
→ `transformers.AutoModelForCausalLM.from_pretrained(...).cuda()`，然后在 GPU 上对拍。

| 模型 | FP32 logits 最大差（HF vs zero） | 贪心 32 token | 对话模板逐字相同 | BF16 导出：HF vs zero（都是 BF16） | BF16 vs FP32 的 top-1 一致率 |
|---|---:|---|---|---:|---:|
| smoke_cuda 的 GRPO 模型（1.3M，GPU 上训练） | 6.7e-6（logits 最大 20.0） | 逐字相同 | 是 | 0.0 | 99.1% |
| l20m 预训练 200 步（23.1M，GPU 上训练） | 7.6e-6（logits 最大 11.2） | 逐字相同 | 是 | 0.0625 | 98.5% |

结论：**通过**（HF 导出 + transformers 在 GPU 上加载对拍）。vLLM 加载与 hermes 工具调用解析器：**跳过：未安装**。
GGUF：冒烟测试里 `convert_hf_to_gguf` 成功（f16，2.7 MB）。llama.cpp 未编译，所以没有量化和试跑（与 GPU 无关）。
注意：第一次跑 `zero.smoke` 时，它的 export 阶段按 `zero/export/gguf.py` 的设计，自动把 llama.cpp 浅克隆到
`~/.cache/zero/llama.cpp`（214 MB 源码，10:05）。这是本次唯一的网络下载。没有下载数据集或模型权重。

## 11. 其他标注了"尚未在 GPU 上验证"的模块（CUDA 对拍）

```bash
flock <gpu0.lock> env CUDA_VISIBLE_DEVICES=0 UV_NO_SYNC=1 uv run python runs/2026-10-01-gpu0-check/parity_cuda.py
```

[`parity_cuda.py`](parity_cuda.py) 用同一份权重、同一批 token（B=4、T=128、dim 128、4 层），在 CPU FP32 / CUDA FP32 / CUDA BF16 autocast 上
各做一次前向 + 反向传播，报告相对误差 ‖a−b‖/‖b‖（以 CPU FP32 为基准）。下表是**修复第 12 节的两个 bug 之后**的结果：

| 模块 | CUDA FP32：输出 / 梯度 | CUDA BF16：输出 / 梯度 | 缓存生成 = 每步重算（CUDA） |
|---|---|---|---|
| 主线 Transformer（对照组） | 1.3e-6 / 1.8e-6 | 1.5e-2 / 2.1e-2 | 是（KV cache，60 token） |
| MoE（8 选 2 + 共享专家） | 1.1e-6 / 1.6e-6 | 5.4e-2 / 7.6e-2（修复前：**报错**） | — |
| MLA / MLA + q_lora | 1.4e-6 / 2.0e-6 | 1.5e-2 / 2.2e-2 | 是（吸收 + 潜向量缓存） |
| 滑动窗口（窗口 32，1:1） | 1.3e-6 / 1.8e-6 | 1.5e-2 / 2.1e-2 | 是（环形缓冲区在 cuda:0） |
| Gated DeltaNet 混合（3:1） | 2.5e-6 / 2.0e-5 | 3.8e-2 / **2.0e-1** | 是（修复前：**报错**） |
| 线性注意力混合（3:1） | 1.7e-6 / 4.9e-6 | 2.2e-2 / 5.1e-2 | 是（修复前：**报错**） |
| MTP（深度 1） | 1.3e-6 / 1.8e-6 | 1.5e-2 / 2.1e-2 | 自推测贪心 = 主模型贪心 |
| 推测解码（1 层草稿 / 草稿 = 目标） | — | — | 贪心与目标模型逐字相同；草稿 = 目标时接受率 100% |

- CUDA FP32 与 CPU FP32 全部在 1e-5 以内。BF16 的误差大多与主线对照组同量级（约 2%）。有两个偏大：
  MoE（BF16 下，路由的 top-k 在接近平局时会换专家）和 **Gated DeltaNet 的梯度（20%）**。纯 PyTorch 的分块 delta 规则
  在 BF16 autocast 下精度明显不够。真正训练时，应按文件说明换 fla 的 kernel（内部用 FP32 累加），或者让这部分在 FP32 下算。
- **fused AdamW**（`trainer.py` 在 CUDA 上默认打开）：同一串梯度走 20 步，fused vs foreach 参数最大差 4.8e-6，vs CPU AdamW 同为 4.8e-6
  （参数总变化 2.3e-2）。所有 GPU 训练（第 2–8 节）都走 fused 路径。**通过**。
- **Muon**：CUDA 上 NS5 默认用 BF16 计算（返回输入的 dtype），与 CPU FP32 的相对差约 2.1–2.3%（CUDA FP32 与 CPU FP32 相差 < 1.2e-4）。
  非方阵的奇异值被推到 [0.68, 1.15]，符合 NS5 的设计。1280 × 3584 的矩阵做一次 NS5 约 3 ms。`MuonAdamW` 在 CUDA 与 CPU 上
  各走 5 步，总更新量相对差 1.25%。用 `optim.name=muon` 训练 l20m 100 步：loss 11.135 → 4.288，val 4.496，约 94k tok/s
  （AdamW 约 99k），显存 17.1 GB（`post_check.sh` 第 3 段）。**单卡通过**；DDP 下的数值一致性需要多卡。
- **CUDA 随机数状态**：见第 3 节，**通过**。
- 吞吐参考（dim 768、12 层、T=1024、batch 8、BF16 autocast、前向 + 反向传播；只说明"跑得动"，**不是性能验证**）：
  稠密 GQA 39.3k tok/s；MoE（8 选 2，激活参数 ≈ 稠密）24.0k；MLA（kv_lora 256）32.9k；
  滑动窗口（窗口 512，1:1）19.9k（自定义布尔 mask 用不了 Flash 内核，反而只有全注意力的一半）；Gated DeltaNet 混合 15.1k（纯 PyTorch）。

## 12. 发现的问题与修复

| # | 问题 | 影响范围 | 处理 |
|---|---|---|---|
| 1 | `zero/post/distill.py` 的 `run_distill` 用 `DistInfo()`（默认 CPU）构造学生的 Trainer。有 GPU、`device = "auto"/"cuda"` 时，**学生仍在 CPU 上训练**。本来放到 GPU 的教师也被 `DistillTrainer` 搬回 CPU。程序不报错，只是慢 | 只影响有 CUDA 时 | **已修**：改成 `DistInfo(device=pick_device(tc.device))`（CPU 上结果不变）；新增 `tests/test_distill.py::test_run_distill_trains_student_on_cuda`（没有 CUDA 时跳过） |
| 2 | `zero/arch/linear_attention.py` 的 `generate_greedy` 把输入和缓存固定建在 CPU 上。模型在 CUDA 上时，报 "Expected all tensors to be on the same device" | 只影响 CUDA | **已修**：按模型所在设备建输入和缓存；新增 `tests/test_arch_linear_attention.py::test_generate_greedy_on_cuda`（没有 CUDA 时跳过） |
| 3 | `zero/arch/moe.py` 的 `MoEFFN.forward` 在 BF16 autocast 下报 `index_add_(): self (Float) and source (BFloat16) must have the same scalar type`（专家输出是 BF16，累加缓冲是 FP32）。CPU 上的 BF16 autocast 同样会触发 | autocast 路径；CPU FP32 不受影响 | **已修**：累加前转成缓冲的精度（FP32 时是空操作）；新增 CPU 测试 `tests/test_arch_moe.py::test_moe_forward_backward_under_bf16_autocast` |
| 4 | GRPO 采样（`sample_group`）不在 autocast 里，CUDA 上是 FP32。它还逐个提示词、以 batch = G 解码。主线尺寸在 3090 上每步采样 15–19 分钟 | 性能 | **没改**（改动较大），见第 9 节 |
| 5 | Gated DeltaNet 的纯 PyTorch 分块实现在 BF16 autocast 下梯度相对误差 20% | 精度（实验模块） | **没改**，见第 11 节 |
| 6 | 滑动窗口用自定义 mask 走 SDPA。CUDA 上用不了 Flash，吞吐只有全注意力的一半 | 性能（实验模块） | **没改**，文件里已写明生产上要用 FlashAttention 的 window_size / FlexAttention |
| 7 | 显存大头是 FP32 logits：主线 T=4096 时 micro batch 4 放不下，32K 单卡放不下 | 显存 | **没改**（分块交叉熵是代码改动），见第 7、8 节 |
| 8 | `configs/tiny/*.toml` 写死 `device = "cpu"`，所以 `zero.smoke` 在 GPU 机器上也不走 GPU | 冒烟测试的覆盖面 | **没改**（按要求不动 configs/），用 `smoke_cuda.py` 补上 |
| 9 | `zero/tools/memory_calc.py` 的文档说"zero 目前没有实现激活检查点"，已过时 | 文档 | 顺手改成"zero 已实现"（与 GPU 标注在同一段） |

改完后，`CUDA_VISIBLE_DEVICES= UV_NO_SYNC=1 uv run pytest -q`：321 passed, 7 skipped；GPU0 上 324 passed, 4 skipped；`ruff check zero tests runs/2026-10-01-gpu0-check` 通过。

## 13. 仍然没有验证的项

- 第 0–12 节做完时，这些项都还"需要多卡"：跨卡 DDP、跨卡断点续训、FSDP2 多卡切分、NCCL 通信、Muon 的 DDP 一致性、compile 与 DDP 的组合。
  当时权限系统拒绝了 GPU1 的命令（与最初"只用 GPU0"的规则一致），所以这些命令一次都没跑。之后，项目负责人本人追加预约了 GPU1、GPU2。
  这些项在**第 14 节**的 2×RTX 3090 上补做了。
- 仍未验证：**8 卡**（RUNBOOK 的真实规模）与 **NVLink** 下的吞吐和 MFU；compile 与 FSDP 的组合；
  MoE 的 `update_bias` 在分布式下的 all_reduce；DPO / GRPO / 蒸馏的多卡版本（本来就未实现）；vLLM / lm-eval / bfcl（未安装）。
- FP8（`plan_budget.py`）：没有实现，3090 也不支持。FlashAttention varlen（SFT 打包隔离）、fla / causal-conv1d kernel、
  grouped GEMM：都没有实现 / 没有安装，仍是"尚未在 GPU 上验证"。（FlexAttention 滑动窗口的速度见第 22 章的 GPU 实测。
  但 zero 的 `sliding_window.py` 仍用布尔 mask SDPA。）

## 14. 多卡验证（2×RTX 3090，PCIe）

硬件：物理 GPU0 + GPU2（功耗上限都是 240 W），`CUDA_VISIBLE_DEVICES=0,2`，`torchrun --standalone --nproc_per_node=2`。
两卡之间没有 NVLink。**这里的吞吐和 MFU 只说明 PCIe 上的 2 卡能到多少，不能代表 8×H100 NVLink。**

### 14.0 NCCL

```bash
CUDA_VISIBLE_DEVICES=0,2 UV_NO_SYNC=1 uv run torchrun --standalone --nproc_per_node=2 runs/2026-10-01-gpu0-check/nccl_probe.py
```

[`nccl_probe.py`](nccl_probe.py)：两卡 all-reduce 结果正确（3072）。256 MiB 用时 57.4 ms，**总线带宽约 4.7 GB/s**
（这是没有 NVLink、P2P 也没走通时的量级；H100 NVLink 是它的上百倍）。不需要 `NCCL_P2P_DISABLE`，没有卡住。
`dist.py` 里 `torch.cuda.set_device` 那一行（只在多卡时执行）也随之验证。

### 14.1 DDP / FSDP2 / Muon 的逐步对拍（RUNBOOK #2、#4）

```bash
CUDA_VISIBLE_DEVICES=0 UV_NO_SYNC=1 uv run python runs/2026-10-01-gpu0-check/dist2_check.py ref     # 单卡参考
CUDA_VISIBLE_DEVICES=0 UV_NO_SYNC=1 uv run python runs/2026-10-01-gpu0-check/dist2_check.py ctrl    # 单卡对照
CUDA_VISIBLE_DEVICES=0,2 UV_NO_SYNC=1 uv run torchrun --standalone --nproc_per_node=2 \
    runs/2026-10-01-gpu0-check/dist2_check.py multi
```

[`dist2_check.py`](dist2_check.py)：l20m 形状，50 步，全部打开确定性算法。每步都是 16 条 × 2048 token：
单卡参考 = micro 8 × 累积 2；DDP = 每卡 micro 8；`ddp_accum` = 每卡 micro 4 × 累积 2（走 `no_sync`）；FSDP2 = 每卡 micro 8。

**先说一个口径上的坑**：第一次用多来源的 `l20m_tiny.toml` 跑，第 1 步的 loss（还没有任何更新）单卡是 11.1350，双卡是 11.1327。
两边不是同一批样本。原因是 `MixtureLoader` 的来源采样器按 `(seed, rank)` 播种，每张卡各自抽来源（`zero/DESIGN.zh.md` 已写明，
不是 bug）。所以和单卡的逐步对拍改用单来源配置 [`configs/l20m_tiny_1src.toml`](configs/l20m_tiny_1src.toml)。
（它把三个来源的分片合成一个 `PackedDataLoader`，第 g 个全局样本分给 rank g % world_size。）多来源的那组结果仍保留，
用来做"2 卡 FSDP vs 2 卡 DDP"（两者看到的数据相同）。

| 对拍 | 结果 |
|---|---|
| 第 1 步 loss（单来源） | 单卡 11.155300 = DDP 11.155300；FSDP 11.155274（参数以 BF16 参与计算，差 2.3e-6） |
| DDP 各卡参数（每个 DDP 运行 50 步后，按位解释成整数求和） | **两张卡逐位相同**（ddp、ddp_accum、ddp lr 3e-4、muon_ddp 全部） |
| DDP vs 单卡，第 2–10 步 | lr 3e-3：相对差 ≤ 8.4e-5；lr 3e-4：≤ 1e-6（只有求和顺序不同） |
| **单卡对照**：micro 4 × 4 vs micro 8 × 2（数学上等价，只有求和顺序不同） | lr 3e-3：第 14 步起超过 1e-3，50 步内最大 21.9%；lr 3e-4：最大 0.12% |
| DDP vs 单卡，50 步内最大相对差 | lr 3e-3：26.7%（同样第 14 步起超过 1e-3）；lr 3e-4：2.5%（出现在第 27 步的损失尖峰，其余步 ≤ 0.5%） |
| `ddp_accum`（no_sync）vs 单卡 | lr 3e-3：27.2%，与 DDP 同一量级 |
| **FSDP2 vs DDP（多来源数据，两者同数据），lr 3e-4** | **最大 0.10%**（RUNBOOK 标准 < 1%，通过）；lr 3e-3 为 7.5% |
| FSDP2 vs DDP（单来源），lr 3e-4 | 前 25 步 ≤ 1.1e-3，第 27 步尖峰处 4.5% |
| Muon：DDP vs 单卡（单来源，lr 3e-3） | 最大 3.7%；两张卡参数逐位相同 |
| 显存峰值（`max_memory_allocated`，两卡） | DDP 14.53 / 14.53 GiB；ddp_accum 7.66 / 7.75；FSDP2 14.33 / 14.33；Muon 15.11 / 14.98 |

怎么读：单来源的 tiny 数据在 lr 3e-3 下很不稳定（单卡参考自己在第 15 步就从 6.4 跳回 11.2）。损失尖峰会把 1e-6 级的求和顺序差
放大到百分之几十。单卡上只换累积方式的对照也一样（21.9%）。DDP / FSDP 与单卡的差异开始出现的时间、放大的幅度，
都和这个对照同一量级。再加上第 1 步相同、各卡参数逐位相同，结论是**跨卡 DDP 与 FSDP2 的数值通路正确**。
"50 步 loss 误差 < 1%"这条标准只在不那么混沌的设置下有意义：多来源数据 + lr 3e-4 时，FSDP2 vs DDP 最大 0.10%。

### 14.2 FSDP2 checkpoint：多卡续训与单卡读回（RUNBOOK #4）

这一项同在 `dist2_check.py multi` 里。把 2 卡 FSDP2 的输出目录只留 step 25，在 2 卡上续训到 50：**第 26–50 步与不中断的运行逐位相同**。
（`get_model_state_dict` / `get_optimizer_state_dict` 的聚合、`set_model_state_dict` / `set_optimizer_state_dict` 的分发都走了真正的 2 卡切分。）
2 卡 FSDP2 存的 checkpoint 用单卡 `load_policy` **严格读回 91 个张量**，验证集 CE 为 6.4852（DDP 的 checkpoint 是 6.4594，两条轨迹本来就不同）。

### 14.3 跨卡断点续训（RUNBOOK #3）

```bash
env GPU=0,2 NPROC=2 DET=1 STEPS=100 EVERY=50 KILL_AT=75 TAG=resume_det_2gpu \
    bash runs/2026-10-01-gpu0-check/resume_check.sh --set train.grad_accum_steps=1
```

2 卡 DDP，确定性算法。第 75 步 `kill -9` torchrun 和两个 worker，再原样重跑：`Resumed training from .../resume_det_2gpu/kill/ckpt/step_00000050 (step 50)`。
**续训后第 51–75 步与被杀前逐位相同；第 51–100 步与不中断的 2 卡运行逐位相同**，step 100 的 val_loss 6.224730 = 6.224730。

### 14.4 吞吐与 compile（RUNBOOK #2、#5、#6 的 2 卡版）

```bash
env GPU=0,2 NPROC=2 bash runs/2026-10-01-gpu0-check/train_run.sh pretrain_l20m_ddp2 runs/2026-10-01-gpu0-check/configs/l20m_tiny.toml \
    --set train.max_steps=200 --set schedule.warmup_steps=20 --set checkpoint.every=100 --set train.eval_every=100 --set logging.every=10
# compile：另加 --set train.compile=true（输出目录 pretrain_l20m_ddp2_compile）
```

默认内核（非确定性）。每卡的工作量与第 2 节单卡 l20m 相同（micro 8 × 累积 2），所以每步 65,536 token：

| 运行 | tok/s（稳定后） | 相对单卡 | 每卡 MFU（3090） | 显存峰值（nvidia-smi） | loss 1 → 200 / val |
|---|---:|---:|---:|---|---|
| 单卡（第 2 节） | ≈ 98,500 | 1× | 26% | 17.1 GB | 11.135 → 5.429 / 5.627 |
| 2 卡 DDP | ≈ 172,000 | **1.75×（扩展效率约 88%）** | 23% | GPU0 17,294 / GPU2 17,263 MiB | → 5.487 / 5.522 |
| 2 卡 DDP + `torch.compile` | ≈ 218,000 | 2.2×（比 2 卡 eager 快约 27%） | 29% | GPU2 20,523 MiB | → 5.740 / 5.848 |

两卡显存均衡。compile 与 DDP 的组合能跑，loss 正常下降。最终 loss 与 eager 的差别在默认内核的运行间噪声内（见"口径"第 3 条）。
每步比单卡同样的工作量多约 48 ms（0.381 s 对 0.333 s）。其中 PCIe 上的梯度 all-reduce 约 20 ms（l20m 的梯度约 92 MB，按 4.7 GB/s 估算）。其余是通信与计算没能完全重叠、各卡互相等待的开销。

### 14.5 32K 长序列上多卡 FSDP（RUNBOOK #8 的多卡版）

```bash
env GPU=0,2 NPROC=2 bash runs/2026-10-01-gpu0-check/train_run.sh longctx_fsdp2 runs/2026-10-01-gpu0-check/configs/longctx_tiny.toml \
    --set train.activation_checkpointing=true --set train.max_steps=3 --set train.eval_every=0 --set checkpoint.every=0 \
    --set checkpoint.resume=false --set logging.every=1
# 3 卡：GPU=0,1,2 NPROC=3 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True，配置换成 configs/longctx_tiny_1src.toml
# （tiny 语料的 code 来源只切得出 2 个 32K 片段，3 个 rank 分不过来）
```

| 设置（主线 689.5M，T = 32,768，micro 1，激活检查点） | 结果 | OOM 时已分配 | 还要申请 |
|---|---|---:|---:|
| 单卡（第 8 节） | OOM | 19.57 GiB | 8 GiB |
| 2 卡 FSDP2（GPU0 + GPU2） | **OOM** | 15.84 GiB | 8 GiB（FP32 logits 32768 × 65536） |
| 3 卡 FSDP2（GPU0 + GPU1 + GPU2） | **OOM** | 15.40 GiB | 8 GiB |

FSDP 把参数 / 梯度 / 优化器状态切开。从 1 卡到 2 卡省了约 3.7 GiB，从 2 卡到 3 卡只再省 0.4 GiB。剩下的是每张卡自己的激活和 logits，切不掉。
**在 24 GB 的卡上跑 32K，需要分块交叉熵（或更大的卡），加卡解决不了。** 按 memory_calc，80 GB 的 H100 上应能放下，但仍需实测。

## 总表

| 验证项 | 状态 | 证据 |
|---|---|---|
| 0a. 单元测试（CPU / GPU 可见） | 通过 | 第 0 节：CPU 321 passed / 7 skipped，GPU0 324 passed / 4 skipped |
| 0b. `zero.smoke` | 通过（但实际在 CPU 上） | 第 0 节：10 个阶段 540 s；tiny 配置写死 CPU |
| 0c. 冒烟测试的 CUDA 版（`smoke_cuda.py`） | 通过 | 第 0 节：10 个阶段 64 s，训练阶段全在 cuda:0 |
| #1 BF16 + SDPA 走 Flash，`enable_gqa=True` | 通过 | 第 1 节：默认即 Flash 内核；efficient 不支持 GQA；RUNBOOK 原命令通过 |
| #2 训练通路（torchrun 单卡） | 通过（单卡） | 第 2 节：l20m / l60m / l150m loss 下降，tok/s、显存、MFU |
| NCCL（2 卡，PCIe） | 通过 | 第 14.0 节：all-reduce 结果正确，总线带宽约 4.7 GB/s |
| #2 跨卡 DDP | 通过（2×3090，PCIe） | 第 14.1、14.4 节：各卡参数逐位相同，与单卡的差异和求和顺序对照同量级；显存均衡；吞吐 1.75×（扩展效率约 88%） |
| #3 断点续训（kill -9） | 通过（单卡） | 第 3 节：确定性算法下续训与不中断逐位相同 |
| #3 跨卡续训 | 通过（2×3090） | 第 14.3 节：确定性算法下，kill -9 后续训与不中断的 2 卡运行逐位相同 |
| #4 FSDP2 / DDP 包装与 checkpoint 聚合 | 通过（单卡通路 + 2×3090） | 第 4 节（单卡通路）；第 14.1、14.2 节：2 卡 FSDP2 vs DDP 最大 0.10%（lr 3e-4），2 卡续训逐位相同，单卡 load_policy 严格读回 |
| #5 torch.compile | 通过（单卡、2 卡 DDP） | 第 5 节：loss 在运行间噪声内，吞吐 +37% / +41%；第 14.4 节：2 卡 DDP + compile 比 eager 快约 27%。与 FSDP 的组合未验证 |
| #6 MFU 实测 | 通过（3090 数字） | 第 6 节；H100 的 MFU 仍需实测 |
| #7 显存与 batch | 通过 | 第 7 节：24 GB 上不开检查点 0、开检查点最多 3；memory_calc 判断正确，偏保守 2.1–2.4 GiB |
| #8 32K 长序列 | 失败（24 GB 卡上放不下） | 第 8 节：单卡 32K / 16K 即使 micro 1 + 检查点也 OOM，8K 可以；第 14.5 节：2 卡、3 卡 FSDP2 也 OOM（logits 切不掉），需要分块交叉熵 |
| #9 SFT / 蒸馏 / DPO / GRPO（CUDA + BF16） | 通过（修复蒸馏设备 bug 后） | 第 0、9、12 节 |
| #9 GRPO 采样耗时 | 已测 | 第 9 节：主线尺寸每步采样 15–19 分钟（3090），需要批量采样或 vLLM |
| #10 HF 导出 + transformers 对拍（GPU） | 通过 | 第 10 节：FP32 logits 差 < 1e-5，贪心逐字相同 |
| #10 vLLM | 跳过：未安装 | — |
| lm-eval / bfcl | 跳过：未安装 | — |
| fused AdamW | 通过 | 第 11 节：与 foreach / CPU 差 4.8e-6 |
| 激活检查点 | 通过 | 第 7、11 节：梯度逐位相同，显存 / 吞吐实测 |
| CUDA 随机数状态保存恢复 | 通过 | 第 3、11 节 |
| Muon CUDA BF16 NS5 | 通过（单卡） | 第 11 节：与 CPU FP32 差约 2%，训练正常 |
| Muon DDP 数值一致性 | 通过（2×3090） | 第 14.1 节：两张卡参数逐位相同，与单卡的差异和求和顺序对照同量级 |
| MoE / MLA / 滑动窗口 / 线性注意力 / MTP：CUDA 前向反向传播 | 通过（修复 2 个 bug 后） | 第 11、12 节 |
| 推测解码 / MTP 自推测：CUDA 正确性 | 通过 | 第 11 节：贪心逐字相同 |
| 上述实验模块的 GPU 性能 | 未验证（只有吞吐参考数字） | 第 11 节 |
| grouped GEMM / fla / FlexAttention / varlen / FP8 | 跳过：未实现或未安装 | 第 13 节 |
| 8 卡 / NVLink 吞吐与 MFU、compile + FSDP、MoE 分布式 update_bias | 未验证 | 第 13 节 |
