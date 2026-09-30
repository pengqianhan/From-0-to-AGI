# GPU0 验证：第 21–23 章

日期：2026-10-01。分支 `gpu0-verification`。负责范围：`chapters/21-kv-cache-ledger`、`chapters/22-local-sparse-attention`、`chapters/23-linear-attention-hybrid`。

## 环境

- CPU：AMD Ryzen Threadripper PRO 3995WX（Zen 2，AVX2，无 AVX-512），共享服务器，其他 agent 与视频渲染同时在跑。
- PyTorch 2.11.0+cu128，CUDA 12.8，Triton 3.6.0；没有安装 `flash_attn`、`fla`（flash-linear-attention）、`causal_conv1d`。
- GPU：只用 GPU0（NVIDIA GeForce RTX 3090，`CUDA_VISIBLE_DEVICES=0`，经 `gpu0.lock` 排队；首次使用前确认 `torch.cuda.device_count() == 1`）。`nvidia-smi` 显示 GPU0 功耗上限 **240 W**（默认 350 W）。同一张卡上把 1 GiB 连续读一遍（求和）实测 873 GB/s（规格 936 GB/s），这个参照已经包含了功耗上限的影响。
- CPU 复现：`CUDA_VISIBLE_DEVICES= UV_NO_SYNC=1 uv run python ...`，最多 4 个进程并行。
- 会训练并缓存权重的 4 个脚本（21/04、22/02、23/04、23/05；22/03–05 读 22/02 的缓存）默认写 `code/out/`。为了不往仓库里写东西，我用一个小包装器 `scratchpad/ch21-23/run_redirected.py` 执行：读入脚本源码，只把 `OUT = ...` 那一行换成草稿目录（22/03–05 是在加载 02 之后设 `m.OUT`），再以 `__main__`、原文件名 `exec`。脚本文件本身没有改动。权重在 `scratchpad/ch21-23/out/`。
- 原始输出：CPU 在 `scratchpad/ch21-23/logs/`，复跑在 `logs2/`，GPU 最终输出在 `final/`（`scratchpad` = `/tmp/claude-1006/-home-phan635-Opensource-From-0-to-AGI/1d40fca9-4469-49a7-88b9-4e96bf8f0051/scratchpad`）。

## 一、CPU 复现表

"CPU 耗时"是 `uv run` 整个进程的墙钟时间（4 个并行、机器繁忙时测得，只作量级参考）。14 个脚本全部 rc=0，**没有报错**。

| 章 | 脚本 | CPU 耗时 | 分类 | 说明 |
|---|---|---:|---|---|
| 21 | `01_kv_ledger.py` | 0.1 s | 一致 | 两张账本表逐项一致（脚本对小于 1 GiB 的量打印 MiB，README 换成 GiB：448 MiB = 0.44 GiB、1008 MiB = 0.98 GiB 等）；zero 对拍两项 True |
| 21 | `02_prefill_decode.py` | 2.4 s | 一致 | prefill 表、decode 表、并发条数表全对（131,072 行脚本打印 2173.9 ms，README 写 2.2 s） |
| 21 | `03_mla.py` | 2.6 s | 不一致（舍入级） | 吸收 = 显式 4.4e-16 一致；"带缓存逐个喂 = 一次性"README 2.8e-16，实测 3.9e-16（float64）；其余一致 |
| 21 | `04_attention_variants.py` | 1370 s（训练 15 个模型） | 不一致（FP32 训练漂移，结论不变） | 见细节 1 |
| 22 | `01_masks_and_ledger.py` | 2.7 s | 一致 | 掩码图、55/34 对、128×/4,096×、感受野两行、128K 账本四行全对 |
| 22 | `02_swa_model.py` | 735 s（训练 6 个模型） | 仅计时不同 | 只打印参数量（449K / 167K），README 没引用；耗时见细节 4 |
| 22 | `03_compare.py` | 84 s | 一致 | loss 1.797 / 1.745 / 1.748、KV 位置数、捞针三段与 12 段细分全对 |
| 22 | `04_bounded_cache.py` | 13 s | 一致 | 三行 True 与缓存字节数全对 |
| 22 | `05_topk_sparse.py` | 236 s | 一致 | 7 行表逐字一致 |
| 23 | `01_linear_attention.py` | 4.9 s | 不一致（舍入级）+ 计时 | 并行 vs 递推最大相对误差 README 4.7e-07，实测 4.2e-07；KV/状态表一致；decode 计时见细节 3 |
| 23 | `02_chunked.py` | 6.1 s | 不一致（舍入级）+ 计时 | 递推 − 并行 README 1.8e-06，实测 2.0e-06；递推 − 分块 6.0e-07 一致；衰减表一致；速度表见细节 3 |
| 23 | `03_delta_rule.py` | 2.4 s | 不一致（舍入级） | 第 1–3 部分全对；第 4 部分输出差 / 状态差 README 4.8e-07 / 3.6e-07，实测 4.2e-07 / 2.4e-07 |
| 23 | `04_hybrid_lm.py` | 1320 s（训练 4 个模型） | **不一致**（影响一句正文） | 见细节 2 |
| 23 | `05_associative_recall.py` | 400 s（训练 4 个模型） | 不一致（±1 个百分点，结论不变） | 见细节 2 |

统计（14 个脚本）：一致 6；仅计时不同 1（22/02）；不一致但只是舍入级 4（21/03、23/01、23/02、23/03，其中 23/01、23/02 同时有计时差异）；不一致、FP32 训练数值漂移 3（21/04、23/04、23/05，只有 23/04 让正文里一句话不再成立）；报错 0。

**这些差异是机器相关的，不是同一台机器上的随机性**：21/03、23/01、23/02、23/03 在本机复跑一次，数字逐位相同；23/04 的 GGGA 在新目录里重新训练一次，验证 loss 仍是 1.652。第 10 章（另一个 agent）在本机复跑 `01_tiny_model.py` 得到的 MHA 种子 0 是 1.831、种子 1 是 1.861，和本机 21/04 的 MHA 种子 0、1 完全相同。所以 README 的数字大概是在另一种 CPU（可能是 AVX-512 的机器）上生成的，FP32 kernel 的舍入不同，训练几百步后放大到小数点后第三位。

## 二、不一致与计时差异的细节

### 细节 1：第 21 章 `04_attention_variants.py`（FP32 训练漂移，结论不变）

| 方案 | README 均值 [三个种子] | 实测均值 [三个种子] |
|---|---|---|
| MHA | 1.858 [1.838 / 1.869 / 1.867] | 1.852 [1.831 / 1.861 / 1.865] |
| GQA | 1.856 [1.867 / 1.849 / 1.852] | 1.853 [1.865 / 1.843 / 1.852] |
| MQA | 1.860 [1.860 / 1.858 / 1.862] | 1.859 [1.860 / 1.853 / 1.864] |
| MLA-48 | 1.887 [1.878 / 1.888 / 1.895] | 1.886 [1.878 / 1.887 / 1.894] |
| MLA-16 | 1.886 [1.889 / 1.854 / 1.916] | 1.886 [1.889 / 1.854 / 1.916]（完全相同） |

每层每位置个数、每 token 字节数、实测缓存字节数（2,150,400 / 1,075,200 / 537,600 / 537,600 / 268,800）、注意力参数量、"缓存版 = 朴素版"全部一致。受影响的正文数字（第 5 节）：

- "MHA 种子 0 的 loss 1.838 与第 10 章完全相同"：本机是 1.831，第 10 章本机也是 1.831，"完全相同"仍成立，只是数字变了；
- "MHA 自己换种子就能差 0.031（1.838 对 1.869）"→ 本机 0.034（1.831 对 1.865）；
- "三者均值在 1.856–1.860 之间"→ 本机 1.852–1.859；
- "MLA-48 的三个种子（1.878–1.895）全部高于前三种结构的全部九个种子（最高 1.869）"→ 本机 1.878–1.894、最高 1.865，结论仍成立；"均值比 MQA 高约 0.03"→ 本机 0.027，仍成立。

### 细节 2：第 23 章 `04_hybrid_lm.py`、`05_associative_recall.py`

`04_hybrid_lm.py` 的验证 loss：

| 结构 | README | 实测 |
|---|---:|---:|
| AAAA | 1.685 | 1.683 |
| LLLL | 1.754 | 1.760 |
| GGGG | 1.661 | **1.648** |
| GGGA | **1.649** | 1.652 |

参数量、推理缓存表、Qwen3.5-0.8B 缓存表全部一致。**受影响的正文**（第 8.2 节）："3:1 混合最低（1.649）"在本机不成立——本机最低的是 GGGG（1.648），GGGA 1.652 第二；表里 GGGA 那格的加粗也随之不对。"Gated DeltaNet 比纯注意力略好（1.661 vs 1.685）"→ 本机 1.648 vs 1.683，方向不变。"前三名之间只差 0.04"（本机 0.035）、"最大 0.1 nats"（本机 0.112）仍成立。README 已经写了"单一随机种子……不能据此排出可靠的名次"，所以只是一句具体表述要改，结论不受影响。

`05_associative_recall.py`：AA、GA 两行逐格一致；LL 的 N = 16 / 20 从 24% / 19% 变成 25% / 18%；GG 的 N = 8 / 16 / 20 从 62% / 39% / 32% 变成 63% / 38% / 31%。三条结论不变（N = 24 时 29% → 79% 一致）。顺带发现：正文"Gated DeltaNet 比朴素线性……每个 N 上都高 10–20 个百分点"用 README 自己的数字算，N = 4、8 已经是 21、22 个百分点（本机 21、23），建议改成"12–23 个百分点"或"十到二十几个百分点"。

### 细节 3：第 23 章 01、02 的计时（仅计时不同）

- `01` 第 3 部分（毫秒）：README 0.033/0.030、0.250/0.030、12.5/0.030、44.2/0.030；本机 0.056/0.042、0.260/0.039、5.356/0.041、20.920/0.041。趋势相同（softmax 随 T 涨，线性是常数）。
- `02` 速度表（毫秒）：README 256：33.4/4.1/3.9，1024：76.2/12.7/79.6，4096：344.1/47.2/2015.0；本机 256：10.2/0.8/0.9，1024：41.8/3.9/23.3，4096：167.0/15.5/678.8。正文"分块比逐 token 快约 7 倍"本机是 10.8 倍；"比完全并行快 40 多倍"本机 43.8 倍，成立；"短序列时完全并行最快"本机 T = 256 时分块 0.8 ms 略快于并行 0.9 ms，两者持平。

### 细节 4：第 22 章 `02_swa_model.py` 的耗时说法不统一（仅计时）

脚本 docstring 写"训练全部 6 个模型，约 5 分钟"，README 写"首次共用了约 40 分钟"。本机 4 个进程并行时用了 735 s（约 12 分钟）。两处说法不一致，建议统一。

## 三、新增的 GPU 脚本和 README 小节

三个脚本都通过 `UV_NO_SYNC=1 uv run ruff check`；没有 CUDA 时打印约定的提示并 exit 0（已用 `CUDA_VISIBLE_DEVICES=` 验证）；固定随机种子；先预热，用 CUDA event 计时取中位数；开头打印 GPU 名称和 torch 版本。README 里的数字全部逐字取自最终版本在 GPU0 上的一次运行（`scratchpad/ch21-23/final/*.log`）。

| 章 | 新脚本 | GPU0 墙钟 | README 新增 |
|---|---|---:|---|
| 21 | `chapters/21-kv-cache-ledger/code/05_gpu_decode_ledger.py` | 9.0 s | `## GPU 实测（单张 RTX 3090）`，第 6 节小结之后、`## 从极简到生产级` 之前 |
| 22 | `chapters/22-local-sparse-attention/code/06_gpu_flex_window.py` | 39.6 s（含 FlexAttention 冷编译 7 s） | 同上位置 |
| 23 | `chapters/23-linear-attention-hybrid/code/06_gpu_linear_vs_softmax.py` | 23.6 s | 同上位置 |

### 第 21 章：decode 一步 ≈（权重 + KV cache）÷ 带宽

按主线模型形状（28 层、宽 1280、16/8 头、head_dim 128、FFN 3584、词表 65,536）搭随机权重的 BF16 decode 步，KV cache 预填 T 个位置，CUDA Graph 重放；理论下限直接调用 `02_prefill_decode.py` 的 `decode_step`，只把 `PEAK_FLOPS`/`HBM_BW` 换成 3090 规格。普通注意力用 SDPA（FlashAttention），MLA 手写吸收路径（加权平均用 split-KV）。

主要结果：batch 1 从 T = 1K 到 128K，KV cache 多读 13.89 GiB、一步多 23.83 ms（折合 626 GB/s，每 GiB 约 1.7 ms）；"4 条 32K"与"1 条 128K"都读 14 GiB KV，一步 27.48 vs 27.23 ms；4K 时 batch 1→16 吞吐 244→936（约 3.8×，3090 账本预测 3.3×），32K 时 batch 1→4 只有 108→146（约 1.35×，账本 1.25×）。T = 32K 时 MHA/GQA/MQA/MLA 的 KV cache 实际分配 7.00/3.50/0.44/0.98 GiB，与账本完全一致；一步 15.57/9.33/4.36/7.75 ms。实测比理论下限慢 1.5–2.1 倍：短上下文时等效带宽只有 440 GB/s（每层权重矩阵只有 5–18 MB，batch 1 的矩阵-向量乘跑不满带宽；微基准里每层大小的单个矩阵-向量乘（1280×5120、3584×1280 等，batch 1）只有 240–410 GB/s，而 1280×65536 的 lm_head 能到 660–750 GB/s），长上下文靠近 600 GB/s。不用 CUDA Graph 时 T = 1K 一步 7.51 ms（用了 3.40 ms）。

调试中发现：手写 `q @ Kᵀ → softmax → @ V` 做 GQA decode 只有约 270 GB/s（长度 T 的归约只分给很少几个 SM），换成 SDPA 后 600 GB/s；MLA 走 SDPA（head_dim 576）会退到极慢的路径（11 ms/层），所以 MLA 用手写 + split-KV。

### 第 22 章：滑动窗口要"真的跳过"窗口外的块

batch 1、16 头、head_dim 128、BF16；稠密因果 SDPA（FlashAttention）/ 布尔掩码 SDPA（memory-efficient + T×T 掩码）/ FlexAttention（`create_block_mask` 块稀疏）。T = 16,384 时 FlexAttention 从 W = 128 的 0.90 ms 涨到 W = 4,096 的 9.37 ms，稠密因果恒为约 23 ms，布尔掩码恒为约 70 ms；W = 1,024 时 T = 4K/16K/64K 下 FlexAttention 0.94/3.61/12.06 ms，稠密 1.62/22.72/473.55 ms（64K 时快 39.3×）。三者最大差 3.9e-03（BF16 舍入量级）。**FlexAttention 在 3090 + PyTorch 2.11 上能跑**，唯一的坑是 `create_block_mask` 默认会铺开整张 T×T 表，T = 65,536 时要申请 32 GiB 而 OOM；用 `torch.compile(create_block_mask)`（`_compile=True` 已被标为弃用）解决。

### 第 23 章：softmax vs 线性注意力的分块 / 递推形式

形状取 Qwen3.5-0.8B 的 Gated DeltaNet 层（16 头，d_k = d_v = 128）；03、04 的函数原样在 `with torch.device("cuda")` 里调用（FP32、块长 64），softmax 用 FlashAttention（BF16）。GPU 上分块 = 递推（输出差 6.0e-07、状态差 4.8e-07）。T = 4,096 时逐 token 递推 813.5 ms、分块 43.1 ms（约 19×）；分块随 T 线性涨，FlashAttention 16K→64K 涨 21×，64K 时朴素线性分块 180.7 ms 已快于 FlashAttention 460.1 ms（Gated DeltaNet 分块 757.5 ms 还没追上）。decode：Gated DeltaNet 状态始终是 (1, 16, 128, 128)、1 MiB，一步 0.24–0.31 ms；softmax 的 KV cache 从 8 MiB 涨到 2,048 MiB，一步从 0.053 涨到 3.408 ms。短序列上本章纯 PyTorch 实现被 Python 循环的固定开销压住（Gated DeltaNet 每块约 0.7 ms、朴素线性每块约 0.18 ms，与 T 无关），这是生产上要用 fla 融合 kernel 的原因（本机没装 fla，没有对比）。

## 四、GPU 结果对 README 里 GPU 说法的影响（均未改动原文，请统一处理）

1. **第 22 章"从极简到生产级"最后一行**："GPU 上要真正跳过窗口外的块，需要 FlashAttention 的 `window_size=(W − 1, 0)`，或 PyTorch FlexAttention 的滑动窗口 `mask_mod`……**这些 GPU 路径尚未在 GPU 上验证**"。本次证实了 FlexAttention 这一条：在 3090 + PyTorch 2.11 上能跑、真的按块跳过、数值与掩码版一致（注意是章节里的独立脚本，不是 `zero/arch/sliding_window.py`）。FlashAttention 的 `window_size` 没法验证（没装 `flash_attn`），vLLM / transformers 也没测。
2. **`zero/arch/sliding_window.py` 在 GPU 上的现行路径会比全注意力还慢**：它（第 229–232 行）用 SDPA + 布尔掩码，本次的同款调用在 T = 16K 时约 70 ms，是稠密因果 FlashAttention（约 23 ms）的 3 倍，还要 T² 大小的掩码和偏置（T = 16K 时 256 MiB + 512 MiB）。它的注释已经写了"GPU 上用 FlashAttention 的 window_size 或 FlexAttention 才能真正跳过"，本次数据给这句话提供了实测依据。是否把 zero 的 CUDA 路径换成 FlexAttention，请你决定（zero/ 我没有改）。
3. **第 23 章第 3 节**"GPU 上的差距会更大，因为分块把工作变成了 GPU 最擅长的矩阵乘法"：分块 vs 逐 token 递推这一对，GPU 上是约 19×（T = 4,096），CPU 上 README 写 7×、本机 10.8×，**证实**。但要补一句限定：本章纯 PyTorch 的分块循环在 GPU 上绝对速度受 Python 发射开销限制，短序列比 FlashAttention 慢几十倍。"比完全并行快 40 多倍"那一半 GPU 上没有测。
4. **第 21 章"从极简到生产级"**"本课的 MLA……尚未在 GPU 上验证性能"：本次脚本是自己写的 MLA 吸收路径 decode，不是 `zero/arch/mla.py`，所以 zero 的 MLA 仍未在 GPU 上验证；但数据支持"真正上线要用专门实现"：不融合时 MLA 缓存只有 GQA 的 28%，一步只省约 17%。
5. **第 21 章第 1.2 节**（H100 理论值）："4K 时 batch 1→16 吞吐涨 3.3 倍、32K 时拼批几乎失灵"——3090 实测同一趋势（4K：3.8×；32K batch 1→4：1.35×），与第 1.2 节的说法一致。
6. `zero/arch/linear_attention.py` 的 GPU 路径本次没有测（脚本用的是章节代码）。

## 五、其他

- `chapters/22-local-sparse-attention/video/subtitles.srt` 在 `git status` 里显示已修改，不是我改的（应是视频渲染）。
- `chapters/22-local-sparse-attention/code/out/` 和 `chapters/23-linear-attention-hybrid/code/out/`（`*.pt`，被 .gitignore 忽略）是 10:29 之后视频渲染（`video/build.sh`，scenes.py 缺 cache.json 时会训练小模型）写的，不是我的进程：我开始时这两个目录不存在，我的训练全部重定向到了草稿目录。
- 解读时的一个口径：GPU0 功耗上限 240 W。第 21 章我用同卡实测的 873 GB/s 纯读带宽作参照，所以"离规格峰值远"那部分我只把超出 873 GB/s 参照的差距归到小矩阵-向量乘上；第 22、23 章的解读只比较相对耗时，没有引用离峰值的比例。
- 没有 commit、push、切分支；没有安装任何包。

## 需要你决定的事

1. 第 23 章第 8.2 节"3:1 混合最低（1.649）"在本机复现不出来（GGGG 1.648 更低），要不要按本机数字改表、改这句话，或者只加一句"不同机器上名次会互换"。
2. 第 21/04、23/05 和几个舍入级数字是否统一换成本机结果（结论都不变）。
3. 第 22 章 02 的 docstring（约 5 分钟）与 README（约 40 分钟）的耗时说法要不要统一。
4. `zero/arch/sliding_window.py` 的 CUDA 路径要不要换成 FlexAttention（见第四节第 2 条）。
