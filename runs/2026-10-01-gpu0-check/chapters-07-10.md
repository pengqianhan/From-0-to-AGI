# GPU0 验证：第 7–10 章

日期：2026-10-01。分支 `gpu0-verification`。负责范围：`chapters/07-*` 到 `chapters/10-*`。

## 环境

- CPU：AMD Ryzen Threadripper PRO 3995WX（Zen 2，AVX2，无 AVX-512），共享服务器，运行期间 load average 在 5–54 之间浮动（其他 agent 与视频渲染同时在跑）。
- PyTorch 2.11.0+cu128（CPU 走 MKL / oneDNN，`CPU capability usage: AVX2`），cuDNN 9.19。
- GPU：NVIDIA GeForce RTX 3090（只用 GPU0，`CUDA_VISIBLE_DEVICES=0`，经 `gpu0.lock` 排队；运行前确认 `torch.cuda.device_count() == 1`）。GPU0 功耗上限 240 W（默认 350 W），持续满载时降频，绝对值偏低。没有使用 GPU1（协调者后来说可以用 GPU1 调试，但权限检查拦下了，这类授权需要用户本人确认，所以所有 GPU 运行都在 GPU0 上）。
- CPU 复现命令：`CUDA_VISIBLE_DEVICES= UV_NO_SYNC=1 uv run python chapters/.../code/xx.py`，按章分 3 条流水线并行（同时最多 3 个 CPU 进程）。原始输出在 `/tmp/claude-1006/-home-phan635-Opensource-From-0-to-AGI/1d40fca9-4469-49a7-88b9-4e96bf8f0051/scratchpad/ch07-10/cpu/`，GPU 输出在同目录的 `gpu/`，探针实验在 `explore/`。
- 副作用：CPU 复现时第 9、10 章脚本按设计把训练好的权重写进了 `chapters/09-modern-transformer/code/out/tiny_transformer.pt` 和 `chapters/10-inference/code/out/tiny_kv*.pt`（`*.pt` 已被 .gitignore 忽略）。之后在这台机器上渲染这两章视频时会直接加载它们，不再重训。

## 一、CPU 复现表

"CPU 耗时"是 `uv run` 整个进程的墙钟时间（机器繁忙时测得，只作量级参考）。19 个脚本全部 rc=0，**没有报错**。

| 章 | 脚本 | CPU 耗时 | 分类 | 说明 |
|---|---|---:|---|---|
| 7 | `01_chars_and_bytes.py` | 0.2 s | 一致 | 语料表、UTF-8 表、198 个 / 129 种未见字符全对 |
| 7 | `02_bpe.py` | 1.8 s | 仅计时不同 | 20 个合并、编码例子、压缩率表全对；"纯 Python 用时约 4 秒"实测 1.6 秒 |
| 7 | `03_bigram.py` | 4.6 s | 一致 | 9 行 bpb 表、58 个 `<unk>`、4 段抽样文本逐字一致 |
| 7 | `04_vocab_and_zero.py` | 55.1 s | 仅计时不同 | 对拍表、词表扫描表、embedding 表全对；"zero 0.74 秒、手写 4.55 秒"实测 1.08 / 1.62 秒（倍数从 6.1× 变成 1.5×） |
| 8 | `01_average_to_attention.py` | 0.2 s | 一致 | 两个输出块逐字一致 |
| 8 | `02_attention_from_scratch.py` | 2.2 s | 不一致（舍入级） | 与 SDPA 最大差 README 9.7e-08，实测 8.9e-08（第 8 节输出块和第 10 节小结两处）；其余全对 |
| 8 | `03_why_sqrt_d.py` | 2.1 s | 一致 | 两张表全对 |
| 8 | `04_train_attention.py` | 58.3 s | 一致 | 损失 / bpb 表、头分工表、热力图、逐位置 argmax 全对；"约 1 分钟"也对 |
| 8 | `05_zero_parity.py` | 2.1 s | 不一致（舍入级） | ① 最大差 README 8.9e-08，实测 1.5e-07；② 全对 |
| 9 | `01_position.py` | 1.7 s | 不一致（舍入级） | "无位置信息"最大差 README 5.96×10⁻⁸，实测 0.00e+00（正文的解读"等于完全相同"仍成立）；RoPE 表、0.133、2.2184 全对 |
| 9 | `02_tiny_transformer.py` | 886.0 s | **不一致** | float32 训练漂移，bpb 从第 400 步起差 0.002–0.011，训练后样本文字不同，见细节 2；墙钟 14.8 分钟与 README 的 14.7 分钟相符 |
| 9 | `03_shapes.py` | 2.3 s | 一致 | 形状表、主线数据流、8.0 GiB、13.11M、共享 embedding 表全对 |
| 9 | `04_parity_with_zero.py` | 6.5 s | 不一致（舍入级） | 训练后最大差 README 1.43e-05，实测 1.72e-05（`assert_close` 仍通过）；随机初始化 README 5.96e-07，实测 4.77e-07（另拷贝到草稿目录、无权重时运行）；贪心生成两边相同 True |
| 9 | `05_qk_norm.py` | 2.2 s | 一致 | 表全对 |
| 10 | `01_tiny_model.py` | 98.3 s | **不一致** | 验证 loss README 1.838，实测 1.831；参数量 861,440 与"约 1.5 分钟"对 |
| 10 | `02_sampling.py` | 12.4 s | **不一致** | 温度表、top-p 表、解码策略表的数字和生成文字都变了，见细节 3 |
| 10 | `03_kv_cache.py` | 42.9 s | **不一致** + 计时 | 对拍两项仍是 True、缓存 872,448 字节对；生成文字、logits 差 4.3e-06→2.9e-06 变了；测速与 prefill/decode 属计时差异，见细节 3 |
| 10 | `04_kv_memory.py` | 2.3 s | 一致 | 三张表、117,440,512 全对 |
| 10 | `05_gqa.py` | 290.9 s | **不一致** | 三个 loss 小幅漂移，正文"换种子的差距比三者之间的差距还大"一句在本机不成立，见细节 4 |

统计（19 个脚本）：一致 8；仅计时不同 2（第 7 章 02、04）；不一致但只是 float32 舍入级（1e-7 ~ 1e-5）4（第 8 章 02/05、第 9 章 01/04）；不一致且影响正文数字 5（第 9 章 02、第 10 章 01/02/03/05）；报错 0。

## 二、不一致的细节

### 细节 1：为什么会漂移（共同原因）

第 10 章的代码自加入以来没改过（`git diff 85e12c9 HEAD -- chapters/10-inference/code/` 为空），种子固定、单线程，所以差异来自运行环境：本机是 Zen 2 / AVX2 上的 PyTorch 2.11.0+cu128（MKL + oneDNN），和写 README 的"构建机"浮点累加顺序不同。单步运算里这只是 1e-7 量级的差别（第 8、9 章那几处舍入级不一致），但几百步训练会把它放大成不同的权重。证据：第 9 章训练第 0、200 步的 bpb 与 README 一模一样（8.052、2.781），从第 400 步起才分开。第 1–6 章的 agent 在第 6 章看到的是同一现象。

### 细节 2：第 9 章 `02_tiny_transformer.py`

| 步数 | 0 | 200 | 400 | 600 | 800 | 1000 | 1200 |
|---|---:|---:|---:|---:|---:|---:|---:|
| README | 8.052 | 2.781 | 2.518 | 2.391 | 2.289 | 2.243 | 2.221 |
| 实测 | 8.052 | 2.781 | 2.529 | 2.385 | 2.284 | 2.241 | 2.217 |

训练后样本开头仍是 `ROMEO:\nAnd yield for what be`，之后不同（实测接的是 `deceived my heart.\n\nQUEEN ELIZABETH:` …）。README 里贴的 400 字节样本在本机复现不出来。定性结论（学会剧本格式、古英语用词、偶尔造词）不变。

### 细节 3：第 10 章 `01`–`03`（小模型权重变了，下游数字全变）

- 验证 loss：1.838 → 1.831。
- 第 2.2 节温度表（前 8 名顺序也变了）：

| 温度 | README | 实测 |
|---|---|---|
| T=0.5 | t .427, a .093, n .073, s .066, h .062, m .060, b .047, w .037；熵 2.15 | t .370, s .099, n .096, a .068, m .068, h .063, b .060, w .037；熵 2.26 |
| T=1.0 | t .173, a .081, n .071, s .068, h .066, m .065, b .057, w .051；熵 3.00 | t .158, s .082, n .081, a .068, m .068, h .065, b .064, w .050；熵 2.99 |
| T=1.5 | t .106, …, w .047；熵 3.39 | t .101, s .065, n .064, a .057, m .057, h .056, b .055, w .047；熵 3.36 |

  正文"t 从 0.17 涨到 0.43 … 降到 0.11"实测是 0.16 → 0.37 … 0.10。
- 第 2.3 节 top-p 表：确定的上下文 `e` p = 0.546 → 0.572，top-p=0.9 留下 **5 → 4** 个；不确定的 `A` p = 0.123 → 0.130，仍留 15 个（论点不受影响，反而更明显）。
- 第 2.1 / 2.4 节解码策略表（不重复 4-gram 占比 / 开头）：贪心 0.27 → 0.29（开头变成 `the soul, and the prove the provess`，仍在打转）；T=0.5 0.84 → 0.81；T=1.0 0.95 → 0.98；top-k=5 0.91 → 0.89；top-p=0.9 0.94 → 0.93（开头 `thou more the prevenced can that` 相同）；T=1.5 0.98 → 1.00（正文引用的乱码 `gMycouuesy` 实测是 `gMycouresy`）。
- 第 4.1 节对拍块：两项"一致：True"复现；开头文字变了（贪心 `'the soul, and the prove the provess\nThat'`，采样 `'thou more the prevenced can that\nO, in I'`）；logits 最大差 4.3e-06 → 2.9e-06；缓存 872,448 字节对。
- 第 4.2 节测速（计时差异，处理位置数全对）：64 / 128 / 256 / 512 → 0.29/0.15（1.9×）、0.75/0.31（2.4×）、2.25/0.58（3.9×）、11.89/1.20（9.9×）。
- 第 4.3 节：prefill 16.2 ms（15,792 位置/秒），decode 487.8 ms（525 位置/秒），30 倍（README 16.9 ms / 691.5 ms / 41 倍，计时差异）。

### 细节 4：第 10 章 `05_gqa.py`（一句话的结论在本机不成立）

| 方案 | README 验证 loss | 实测 |
|---|---:|---:|
| MHA | 1.838 | 1.831 |
| GQA | 1.867 | 1.865 |
| MQA | 1.860 | 1.860 |
| MHA 换种子 1 | 1.869（差 0.031） | 1.861（差 0.030） |

参数量、KV cache 字节数全对；生成 512 字的时间 1.28 / 1.18 / 1.36 秒（计时）。问题：第 6.1 节写"只换一个随机种子，同样的 MHA 就差了 0.031——比三者之间的差距还大"。README 的数里三者最大差 0.029 < 0.031 成立；本机三者最大差 0.034（MHA 1.831 vs GQA 1.865）> 换种子的 0.030，**这句话不成立**。"这个规模上分不出谁好"的结论大体仍对（差距与种子噪声同一量级），但措辞需要改，例如改成"和三者之间的差距是同一个量级"。**需要你决定**：改措辞，还是用本机数字整体更新第 10 章（以及第 9 章）的表。

## 三、GPU 实测：新增的脚本与 README 小节

两个脚本都通过 `UV_NO_SYNC=1 uv run ruff check`，没有 CUDA 时打印约定的提示并 exit 0，固定种子，先预热再用 CUDA event 取中位数，开头打印 GPU 名称和 torch 版本；都不和 CPU 对比。README 小节放在 `## 从极简到生产级` 之前（和第 2、16 章的放法一样，在原有 `---` 之后），表里的数字逐字来自下面列出的日志。

### 第 8 章：`chapters/08-attention/code/06_gpu_sdpa_backends.py`（约 13 秒）

- 内容：主线模型的注意力形状（batch 1、16 头、head_dim 128、BF16、因果），T = 512 … 32K；比较第 8 章的手写 `attention`（importlib 原样加载 `02`，用 `torch.device("cuda")` 让它内部的 mask 也建在 GPU 上）与 SDPA 的 math / efficient / flash 三个后端的耗时和峰值额外显存；再检查 `enable_gqa=True` 能走哪些后端。
- README：`chapters/08-attention/README.md` 新增"GPU 实测（单张 RTX 3090）"，两张表（耗时、显存）+ GQA 一段 + 5 句解读。数字来自 `scratchpad/ch07-10/gpu/ch08_06_final.log`（最终版本；此前两次试跑 `try1/try2` 的趋势相同）。
- 要点：手写版显存 ≈ 分数矩阵的 2 倍，随 T² 涨（16K 时 16,704 MiB，32K OOM）；flash/efficient 只多占输出（16K 时 64 MiB），随 T 线性涨；flash 耗时仍随 T² 涨，16K 比手写快 7.3 倍；math 后端比手写还慢还费显存（profiler 看到它先把 q、k、v `aten::to` 成 float32）。
- 脚本里设了 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`（`import torch` 之前 `setdefault`）：不设时手写版在 16K 会因为分配器碎片 OOM（探针：已预留 16.2 GiB 里空着 7.76 GiB，却放不下新的 8 GiB 块），真实需要只有 16.3 GiB。
- 小瑕疵：docstring 写的是"约半分钟"，实测约 13 秒；我想改成"约 15 秒"时被权限检查拦下了，没有再改，请顺手处理或保留。

### 第 10 章：`chapters/10-inference/code/06_gpu_prefill_decode.py`（约 53 秒，峰值显存 7.0 GiB 已分配 / 9.6 GiB 预留）

- 内容：importlib 原样加载 `01_tiny_model.py` 的 `TinyLM` 和 `03_kv_cache.py` 的生成函数（在 `torch.device("cuda")` 里运行，不改原文件），尺寸换成 Llama 3 8B 一层的形状（d 4096、32/8 头、FFN 14336）叠 4 层，0.87B 参数、BF16、随机初始化。三部分：① 朴素 vs KV cache 每 token 耗时；② 一次前向喂 T = 1 … 2048 个 token（eager 与 CUDA graph 两种计时，算 TFLOPS 与读权重带宽）；③ decode batch 1 … 64，上下文 64 与 512。
- README：`chapters/10-inference/README.md` 新增"GPU 实测（单张 RTX 3090）"，三张表 + 5 句解读。数字来自 `scratchpad/ch07-10/gpu/ch10_06_try2.log`（最终代码；之后只改了 docstring 里的显存说明）。另一次同版本运行（`explore/ch10_mem_gpu0.log`，用来量峰值显存）数字差别在 12% 以内。
- 选型说明：先试过主线模型的形状（28 层、d 1280），eager 下每步 decode 要 38 ms，其中 GPU 真正干活只有 4.4 ms（CUDA graph 重放），全被 Python 逐个发射 kernel 的开销淹没，看不出"带宽瓶颈"；所以改成宽而浅的 4 层，并同时给出 CUDA graph 时间。探针显示 `torch.device("cuda")` 上下文本身让每个算子的发射开销从 9.3 µs 涨到 12.1 µs。
- 要点：T = 1 → 32 耗时几乎不变（2.65 → 3.19 ms，读权重 550–660 GB/s，算力 1%–25%），T ≥ 128 随 T 线性涨、算力约六成——拐点和"峰值算力 ÷ 带宽 ≈ 76"对得上；上下文 64 时 batch 1 → 64 吞吐涨 29 倍，上下文 512、batch 64 时 KV cache（512 MiB，外加 `torch.cat`、`repeat_interleave` 的复制）把吞吐压到 3,737；GPU 上 KV cache 只快 1.1–2.7 倍；eager 比 CUDA graph 慢近一倍。

### 第 7、9 章：没有新增

第 7 章是分词和计数，GPU 帮不上正文的论点。第 9 章能想到的只有"把 11 分钟的训练挪到 GPU 上"，不能让任何一个具体论点更清楚；第 9 章里那句关于 GPU 的话，第 8 章的脚本和下面的 RUNBOOK 核实已经覆盖了。

## 四、`enable_gqa` 与 flash 后端（RUNBOOK 阶段表第 1 行的"待核实"）

**结论：能。** RTX 3090、PyTorch 2.11.0+cu128、BF16（FP16 相同）下，`F.scaled_dot_product_attention(..., enable_gqa=True)` 可以走 FlashAttention 后端，而且不指定后端时默认选的就是它。

| 后端 | BF16/FP16，MHA | BF16/FP16，GQA（16 q / 8 kv，`enable_gqa=True`） | FP32，MHA | FP32，GQA |
|---|---|---|---|---|
| FLASH_ATTENTION | 能跑 | **能跑**（与复制 K/V 的 math 结果差 7.8e-03，BF16 舍入量级） | 不能（要求 half/bf16） | 不能 |
| EFFICIENT_ATTENTION | 能跑 | **不能**："For dense input, both fused kernels require query, key and value to have the same num_heads" | 能跑 | 不能 |
| CUDNN_ATTENTION | 能跑 | 能跑 | 不能 | 不能 |
| MATH | 能跑 | 能跑（与复制 K/V 完全相同） | 能跑 | 能跑 |
| 默认选择（`torch._fused_sdp_choice`） | FLASH | FLASH | EFFICIENT | **MATH** |

- RUNBOOK 那条命令原样核实过（`configs/main/pretrain.toml` 的 `zero.model.Transformer`，`.cuda().bfloat16()`，输入 (1, 4096)，`sdpa_kernel(SDPBackend.FLASH_ATTENTION)`）：前向不报错，输出 (1, 4096, 65536) BF16；前向 + 反向也不报错，峰值显存 13.89 GiB。同一条命令换成 `EFFICIENT_ATTENTION` 会报 "No available kernel"。探针脚本：`explore/runbook_flash.py`、`explore/runbook_bwd.py`、`explore/gqa_probe.py`。
- 对 RUNBOOK 的影响：原计划"若不能，退回 efficient 后端"行不通：efficient 不支持 `enable_gqa`，要退就得先 `repeat_interleave` K/V 再用 efficient，或者用 cuDNN 后端。
- 值得记一笔的坑：**FP32 + GQA 时默认会退回 math 后端**，要把 T × T 的分数矩阵整个存进显存。如果有人在 GPU 上用 float32 跑 zero（例如测试或评测），长序列会很费显存。
- 速度：T = 4096 时 flash + `enable_gqa` 与"先复制 K/V 再用 flash"耗时相近（三次运行 1.52/1.47、1.19/1.35、1.17/1.46 ms，互有先后，不能说谁更快）；显存 16 MiB 对 48 MiB。

## 五、GPU 结果与 README 现有说法的关系（没有改原文，由你统一处理）

1. `chapters/08-attention/README.md`"从极简到生产级"表 `F.scaled_dot_product_attention` 那一行："在 CUDA 上会自动选用 FlashAttention-2 或 memory-efficient 内核……**这条 GPU 路径尚未在 GPU 上验证**"——已在 RTX 3090 上验证：BF16 下 MHA 和 GQA 默认都选 flash，`enable_gqa` 不复制 K/V（额外显存 16 vs 48 MiB）。但 memory-efficient 不支持 `enable_gqa`，"或 memory-efficient 内核"对 GQA 不成立；FP32 下 GQA 会退回 math。
2. `chapters/09-modern-transformer/README.md`"从极简到生产级"表 `Attention` 行："GPU 上自动走 FlashAttention 内核，第 14 章，**尚未在 GPU 上验证**"——同上，已证实（包括 zero 主线配置前向 + 反向）。
3. `chapters/10-inference/README.md` 第 4.3 节"decode 每一步……硬件大部分时间花在搬数据上，算力吃不饱"与第 6.1 节"GQA 的速度收益……要在显存带宽成为瓶颈的 GPU 上、长上下文和大 batch 时才明显"——GPU 结果支持这两句（decode 读权重达规格带宽的 64–70%、算力 1%；上下文 512、batch 64 时 KV cache 成了带宽大头）。没有直接测 GQA 与 MHA 的速度差。
4. `runs/RUNBOOK.md` 阶段表第 1 行的"待核实"可以改成已核实（见第四节）。

## 六、需要你决定的事

1. 第 9、10 章的训练相关数字在本机复现不出来（细节 2–4）。是用本机数字更新表格，还是加一句"数字因 CPU / PyTorch 版本会略有不同"？至少第 10 章 6.1 节"比三者之间的差距还大"那句需要改措辞。
2. 第 8 章 02/05、第 9 章 01/04 的舍入级差异（1e-7 ~ 1e-5）要不要更新成本机数字。
3. 上面第五节的三处"尚未在 GPU 上验证"与 RUNBOOK 的"待核实"。
4. 第 8 章 GPU 脚本 docstring 的"约半分钟"（实测约 13 秒）。
