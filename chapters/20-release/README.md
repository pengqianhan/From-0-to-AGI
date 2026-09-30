# 第 20 章：发布 —— 按预注册交卷，把模型装进笔记本

> **一句话目标**：读完这一章，你能按预注册协议对最终模型逐项给出"超过 / 持平 / 落后"，并说清楚没达标时该怎么写；能算出任意模型在 fp16 / Q8_0 / Q4_K_M 下要多少内存，手写分块 INT8 / INT4 量化并量出它让 loss 变了多少；知道一次"发布"要交出哪些东西（标准格式权重、GGUF、模型卡、许可证、中间 checkpoint、本地 demo），以及发布之后怎样不夸大。

📺 **本章视频**：待发布（本地渲染：`bash chapters/20-release/video/build.sh`）
🧪 **本章自检**：学完后在 Claude Code 里输入 `/ch20-release`

---

上一章我们用 GRPO 在工具调用环境里把模型训成了最终版。到这里，训练的活儿都干完了，但模型还只是服务器硬盘上的一个 checkpoint 目录：别人用不了，也没人知道它到底好不好。这一章要解决最后一段路：

1. **它到底好不好？** 按第 11 章写好的预注册协议做最终评测——考卷、模板、解码参数、判定规则在训练前就锁死了，现在只是照着跑、照着判。这就是**闸门 3**。
2. **别人怎么用？** 把权重导出成大家都认识的标准格式；再**量化**，让它能在普通笔记本上跑；配一个本地工具调用 demo，证明它"真正可用"。
3. **怎么说？** 写模型卡：数据、许可证、配方、花费、全部评测结果（包括输的）、已知局限。赢了不夸大，输了不藏。

本章代码（都在 CPU 上跑）：

```bash
uv run python chapters/20-release/code/01_blockwise_quant.py     # 分块 INT8/INT4 量化：误差与大小表（几秒）
uv run python chapters/20-release/code/02_quantize_tiny_model.py # 套在第 10 章的小模型上，看 loss 变多少（单线程约 1–4 分钟）
uv run python chapters/20-release/code/03_memory_calculator.py   # 主线模型在各种格式下的内存（1 秒）
uv run python chapters/20-release/code/04_model_card.py          # 从评测 JSON 生成模型卡骨架（1 秒）
```

## 1. 闸门 3：交卷，而不是改卷

### 1.1 直觉：考卷在开考前就封好了

想象一场考试：如果考完才决定考哪几科、每科怎么算分，那几乎谁都能"考第一"——挑自己擅长的科目、换一种对自己有利的评分方式就行。模型评测里这件事叫**事后挑选（cherry-picking）**，是"超过 SOTA"这类说法最常见的水分来源。

我们的办法是第 11 章的**预注册（preregistration）**：在主线预训练开始之前，把下面这些写进 `eval/PREREGISTRATION.md` 并提交，commit 时间就是登记时间（GOAL.md 3.2）：

- 考哪些基准、什么版本（专项：BFCL + 一个含中文的工具调用基准；通用：知识、中文、数学、代码、指令遵循）；
- 评测框架和版本号、提示词和模板、解码参数；
- 对手清单与冻结日期（官方参数量在我们的 0.7–1.3 倍之间的公开权重模型，Qwen3.5-0.8B 必比）；
- "超过"的判定标准。

闸门 3 做的事情很朴素：**照着这份文件跑一遍，一个字都不改**。之后任何修改只能以"带日期和理由的修订"追加。

### 1.2 判定规则：置信区间整体在 0 的哪一边

对同一组题，我们和对手各有逐题得分 `a_i`、`b_i`（对 = 1，错 = 0）。平均分之差 `d = mean(a) − mean(b)` 有多可信？用第 11 章的**配对 bootstrap**：有放回地抽 n 道题（两个模型用同一组下标），算一次 d\*，重复一万次，取 2.5% 和 97.5% 分位数作为 95% 置信区间 `[lo, hi]`：

```
lo > 0        → 超过（整个区间都在 0 右边）
hi < 0        → 落后
否则（跨过 0） → 持平
```

对应代码 `zero/eval/bootstrap.py` 的 `decide`：

```python
def decide(ci_low: float, ci_high: float) -> str:
    if ci_low > 0:
        return AHEAD      # 超过
    if ci_high < 0:
        return BEHIND     # 落后
    return TIE            # 持平
```

两条附加规则（GOAL.md 3.2 第 4 条）：对手有思考 / 非思考两种模式时两种都测，**取对手较高的那个**来比（`compare_to_opponent`）；对手的官方分数并列展示，但不作为比较依据——所有对手由我们用同一把尺子重跑。

### 1.3 一张判定表长什么样（极小配置演示）

> 以下是**极小配置演示**：约 1.3M 参数的模型、几十道玩具题，只说明判定流程是通的，不是主线模型的结果。

> 注：本节数字来自修复工具调用判分器（第 19 章第 6 节）**之前**的那次冒烟测试，是当时的真实输出。修复后重跑（`uv run python -m zero.smoke --out out/smoke_final`），数据、预训练、中期训练、SFT 各阶段完全一致；蒸馏通过验证的样本从 1 条（那句胡话）变为 0 条，下游的 DPO、GRPO 和评测数字随之变化（例如同一个 SFT 模型的工具调用 call_exact 从 0.133 变为 0.100——模型没变，是判分更严了；GRPO 相对 SFT 仍判"持平"）。你自己运行时以运行结果为准。

冒烟测试（`zero.smoke`）里，"对手"是我们自己的 SFT 模型，拿 DPO 和 GRPO 后的模型和它比（`out/smoke/eval/report.md` 的真实输出）：

| 模型 | 基线 | 任务 | 指标 | 模型 | 基线 | 差值 | 95% CI | 判定 |
|---|---|---|---|---:|---:|---:|---|---|
| dpo | sft | toy_mc | acc | 0.200 | 0.167 | +0.033 | [+0.000, +0.100] | 持平 |
| dpo | sft | tool_dev | call_exact | 0.100 | 0.133 | −0.033 | [−0.100, +0.000] | 持平 |
| grpo | sft | toy_mc | acc | 0.300 | 0.167 | +0.133 | [+0.033, +0.267] | 超过 |
| grpo | sft | tool_dev | call_exact | 0.100 | 0.133 | −0.033 | [−0.100, +0.000] | 持平 |

读这张表要注意两件事：

- **GRPO 在选择题上"超过"了，但在工具调用上没有**——而工具调用才是我们的硬目标。如果只报第三行，就是典型的挑选。
- 30 道题的置信区间宽得吓人（±0.1 量级）。真实评测里 BFCL 有上千道题，区间会窄很多，但**题目少的中文子集照样可能判不出胜负**。这就是为什么预注册里要事先写好"硬目标成立的条件"和多重比较的处理办法，而不是事后看哪个子集赢了就报哪个。

### 1.4 三种结局，三种写法

| 结局 | 能说什么 | 不能说什么 |
|---|---|---|
| 硬目标组对每个对手都"超过" | "在预注册的 BFCL（版本 X）上超过了对手清单里的全部模型，包括 Qwen3.5-0.8B" | "工具调用能力超过所有同尺寸模型"（只在这些基准上测过） |
| 有"持平" | "与 Qwen3.5-0.8B 持平（差值 +0.8，95% CI [−1.2, +2.9]）" | "略微领先"——区间跨过 0 就是没有证据 |
| 有"落后" | 把落后的项列出来，写差距分析（数据？后训练？尺寸？）；GOAL.md 3.4 说这本身就是课程内容 | 换一个基准、换一种模板再跑，直到赢为止 |

最后还有一节**"发布后新增对手"**：发布时再查一遍冻结日之后新出的同尺寸模型，能跑就跑，照实写——**即使它们比我们强**。冻结日期保护的是"我们没有事后挑对手"，不是"我们永远第一"。

## 2. "发布"到底交出什么

GOAL.md 3.5 列了五样：模型（Base、SFT、最终版和关键中间 checkpoint）、量化版、本地 demo、模型卡、可复现配方。前两样的关键词是**标准格式**。

### 2.1 标准格式：让别人的工具直接能用

一个 Hugging Face 模型目录就是事实标准：

| 文件 | 内容 | 谁读它 |
|---|---|---|
| `config.json` | 架构名（我们写 `Qwen3ForCausalLM`）、层数、宽度、头数、RoPE / YaRN 参数 | transformers、vLLM、llama.cpp 的转换脚本 |
| `model.safetensors` | 权重 | 同上 |
| `tokenizer.json`、`tokenizer_config.json` | 分词器，以及 **chat template**（Jinja 模板） | `apply_chat_template`、推理服务的对话接口 |
| `generation_config.json` | 默认的结束符和采样参数 | `generate()` |

两个细节决定了"能不能直接用"：

- **safetensors 而不是 pickle**：PyTorch 默认的 `.pt/.bin` 是 pickle，加载时可以执行任意代码；safetensors 只是"8 字节头长度 + JSON 头（每个张量的 dtype、形状、偏移）+ 连续的原始字节"，不能藏代码，还能零拷贝地 mmap 加载。
- **架构名选一个大家都支持的**：zero 的模型和 Qwen3 稠密模型逐层对应（第 9 章），导出时直接写成 `Qwen3ForCausalLM`，transformers、vLLM、llama.cpp 不用改一行代码就认识。代价是我们不能随便发明新结构——这也是 GOAL.md 3.3 说"主线模型不冒架构风险"的另一个理由。
- **chat template 必须和训练时逐字一致**：工具调用模型对格式极其敏感，多一个换行都可能让它不再输出 `<tool_call>`。我们把训练用的模板原样写进 `tokenizer_config.json`，冒烟测试里用 transformers 的 `apply_chat_template` 渲染，和我们的模板逐字比对（`hf_template_identical=True`）。

### 2.2 中间 checkpoint：给没有算力的读者

GOAL.md 3.5 要求发布"关键的中间 checkpoint"，让读者可以直接从我们的 Base 开始做第 16–19 章的实验。这在完全开放的模型里已经是惯例：OLMo 2 把预训练过程中的 checkpoint 作为同一个仓库的分支发布（`revision="stage1-step140000-tokens294B"`），SmolLM3 单独开了一个 `SmolLM3-3B-checkpoints` 仓库，预训练每 4 万步一个、后训练每个阶段一个。我们沿用"分支名写清步数与 token 数"的做法。

## 3. 量化：为什么，以及怎么做

### 3.1 直觉：内存 = 参数量 × 每个参数的 bit 数

一台普通笔记本有 8–16 GB 内存，还要留给系统和别的程序。模型推理时至少要把全部权重放进内存：

```
权重内存 ≈ 参数量 × bit 数 / 8
```

`03_memory_calculator.py` 逐个张量算主线模型（`configs/main/pretrain.toml` 的暂定形状，689.5M 参数，其中 embedding 83.9M）：

| 格式 | bit/参数 | 权重大小 |
|---|---:|---:|
| fp32 | 32.00 | 2.57 GiB |
| bf16 / fp16 | 16.00 | 1.28 GiB |
| Q8_0 | 8.50 | 0.68 GiB |
| Q4_K_M | 5.00 | 0.40 GiB |

推理时还有 KV cache（第 21 章的公式 `2 × 层数 × KV 头数 × head_dim × 序列长 × 2 字节`）：每个 token 112 KiB，4K 上下文 0.44 GiB，32K 上下文 3.50 GiB——长上下文时 KV cache 比 Q4 的权重还大得多。

0.7B 的模型用 bf16 其实也放得下。量化对它的意义主要是**快**：CPU 上逐 token 生成时，每一步都要把全部权重从内存读一遍，瓶颈是内存带宽（第 10、21 章的 decode 分析），权重小一半，读得就快近一倍。对 7B、70B 的模型，量化则是"放得下 / 放不下"的区别。

### 3.2 分块量化：每 32 个数一个 scale

最简单的量化：找一个缩放系数 `scale`，把每个权重除以它、四舍五入成整数，用的时候再乘回去：

```
scale = max|w| / qmax          （INT8：qmax = 127；INT4：qmax = 7）
q     = round(w / scale)       存这个整数
ŵ     = scale · q              用的时候反量化
```

问题出在"找一个 scale"——给谁找？权重里常有少数绝对值特别大的**离群值（outlier）**。如果整张矩阵共用一个 scale，一个离群值就把 scale 撑大，其余几百万个普通大小的数全被挤进 0 附近那一两个格子里。解决办法是**分块（block-wise）**：沿每一行每 32 个数一块，每块单独一个 scale。scale 本身用 fp16 存，摊到每个数上是 16/32 = 0.5 bit。

`01_blockwise_quant.py` 的核心就这几行：

```python
qmax = 2 ** (bits - 1) - 1                      # INT8 → 127，INT4 → 7
blocks = w.reshape(-1, block)                   # 每行是一块（32 个数）
absmax = np.abs(blocks).max(axis=1, keepdims=True)
scale = (absmax / qmax).astype(np.float16).astype(np.float32)  # scale 用 fp16 存
q = np.clip(np.round(blocks / scale), -qmax, qmax).astype(np.int8)  # q = round(w / scale)
...
return (q.astype(np.float32) * scale).reshape(shape)  # ŵ = scale · q
```

拿一个 1024×1024 的矩阵试（高斯、标准差 0.02，再把 0.1% 的元素放大 10 倍当离群值——这是造出来的矩阵，用来演示离群值的影响）。第一块的前 8 个数：

```
w     : +0.0025 -0.0026 +0.0128 +0.0021 -0.0107 +0.0072 +0.0261 +0.0189
INT8  : scale=0.000366  q=[7, -7, 35, 6, -29, 20, 71, 52]   ŵ 几乎不变
INT4  : scale=0.006641  q=[0, 0, 2, 0, -2, 1, 4, 3]         ŵ=+0.0000 +0.0000 +0.0133 …
```

INT4 只有 15 个格子，0.0025 和 −0.0026 都被压成了 0。整张矩阵的结果：

| 方案 | bit/权重 | 大小 | 权重相对误差 | 输出 y = xWᵀ 相对误差 |
|---|---:|---:|---:|---:|
| fp16 | 16.000 | 2048 KiB | 0.02% | 0.02% |
| INT8 整张一个 scale | 8.000 | 1024 KiB | 7.36% | 7.38% |
| INT8 每行一个 scale | 8.016 | 1026 KiB | 2.09% | 2.09% |
| INT8 分块 32 | 8.500 | 1088 KiB | 0.63% | 0.64% |
| INT4 整张一个 scale | 4.000 | 512 KiB | 92.97% | 93.16% |
| INT4 每行一个 scale | 4.016 | 514 KiB | 36.36% | 36.33% |
| INT4 分块 32 | 4.500 | 576 KiB | 11.32% | 11.43% |
| INT4 两级 scale（Q4_K 思路） | 4.500 | 576 KiB | 8.28% | 8.30% |

"INT4 整张一个 scale"误差 93%，几乎等于把整张矩阵清零；分块后多花 0.5 bit，误差降到 11%。

### 3.3 GGUF 里的块：Q8_0、Q4_K、Q4_K_M

llama.cpp 用的 **GGUF** 文件格式里，量化类型就是上面这个思路的工程化版本（块结构见 llama.cpp 的 `ggml/src/ggml-common.h`）：

| 类型 | 一块 | 结构 | bit/权重 |
|---|---|---|---:|
| Q8_0 | 32 个数 | 1 个 fp16 scale + 32 个 int8 = 34 字节 | 8.5 |
| Q4_0 | 32 个数 | 1 个 fp16 scale + 16 字节（32 个 4 bit） | 4.5 |
| Q4_K | 256 个数的超级块 | 8 个子块（每块 32 个数），每个子块一个 6 bit 的 scale 和一个 6 bit 的 min；超级块再用 2 个 fp16（"scale 的 scale"和"min 的 scale"）；4 bit 的整数 128 字节；共 144 字节 | 4.5 |
| Q6_K | 256 个数 | 16 个子块，8 bit 子块 scale + 1 个 fp16 | 6.5625 |

Q8_0 和我们手写的"INT8 分块 32"是同一种格式。Q4_K 多做了两件事：每个子块除了 scale 还有一个 min（`ŵ = scale·q − min`，能表示不以 0 为中心的块），以及**两级 scale**——子块的 scale 本身也量化成 6 bit，所以同样 4.5 bit 能多存一个 min。`01` 里的 `quant_kstyle` 是它的简化版（ggml 真正的实现还会搜索更好的 scale），误差从 11.3% 降到 8.3%。

最常用的 **Q4_K_M** 不是"全部用 Q4_K"：按 llama.cpp `src/llama-quant.cpp` 的规则，输出层（共享 embedding 时就是 token embedding）用 Q6_K，attn_v 和 ffn_down 在"要多给 bit 的层"（前 1/8、后 1/8 以及中间每 3 层一个）用 Q6_K，其余 Q4_K，一维的 norm 权重保持 f32。`03_memory_calculator.py` 就是照这条规则逐张量算的；为了确认它算得对，我们用 zero 随机初始化了一个 51.4M 参数的同结构模型（dim 512、16 层、共享 embedding），走一遍 `export_to_hf_qwen3` → `convert_hf_to_gguf.py` → `llama-quantize`，再用 llama.cpp 的 `gguf-py` 读出每个张量的实际类型和字节数：f16、Q8_0、Q4_K_M 三种格式的张量总字节数与计算器**逐字节相同**，Q4_K_M 里每个张量的类型也全部和计算器一致（17 个 Q6_K、96 个 Q4_K、65 个 F32）。

作为参照，llama.cpp 文档里 Llama 3.1 8B 的 Q4_K_M 是 4.89 bit/权重、Q8_0 是 8.50；我们的主线模型算出来是 5.00，高一点是因为共享的 embedding 占了 12% 的参数、而它按规则用了 Q6_K。

### 3.4 质量换大小：在小模型上量一量

误差 11% 听起来很吓人，但模型关心的不是每个权重准不准，而是最后的预测变没变。`02_quantize_tiny_model.py` 把量化套在第 10 章的字符级小模型上（0.86M 参数，dim=128、4 层），所有二维矩阵量化，RMSNorm 的一维权重保持 fp32（和 llama.cpp 一样），然后在验证集上测 loss 和"top-1 一致率"（量化模型和 fp32 模型在每个位置最可能的下一个字是否相同）：

| 方案 | 权重大小 | val loss | Δ loss | top-1 一致 |
|---|---:|---:|---:|---:|
| fp32 | 3365 KiB | 1.8385 | — | 100.0% |
| fp16 | 1685 KiB | 1.8385 | −0.0000 | 100.0% |
| INT8 整张一个 scale | 845 KiB | 1.8386 | +0.0002 | 99.4% |
| INT8 分块 32（≈Q8_0） | 897 KiB | 1.8384 | −0.0001 | 99.6% |
| INT4 整张一个 scale | 425 KiB | 1.8525 | +0.0141 | 90.3% |
| INT4 每行一个 scale | 436 KiB | 1.8446 | +0.0061 | 93.5% |
| INT4 分块 32（≈Q4_0） | 477 KiB | 1.8435 | +0.0050 | 94.1% |
| INT4 两级 scale（≈Q4_K） | 486 KiB | 1.8437 | +0.0052 | 94.9% |
| INT3 分块 32 | 372 KiB | 1.8776 | +0.0391 | 85.9% |
| INT2 分块 32 | 267 KiB | 2.3439 | +0.5055 | 46.4% |

几条观察：

- **8 bit 基本无损**：loss 的变化在第四位小数，贪心生成的前 60 个字符和 fp32 一字不差。
- **4 bit 开始有代价**：loss 升 0.005，每 17 个位置就有 1 个最可能的字变了；贪心生成从第 9 个字符起就和 fp32 分叉（"And the course…" vs "And the such…"）。
- **4 bit 往下是悬崖**：3 bit 的代价是 4 bit 的 8 倍，2 bit 的模型基本废了（loss 从 1.84 跳到 2.34，生成的是 "is is apeeng"）。
- 这个小模型训练得不久，权重里没有明显的离群值，所以"整张一个 scale"的 INT8 也还行，两级 scale 和普通分块在 loss 上也分不出高下（1.8437 vs 1.8435，一致率 94.9% vs 94.1%）——和上面造出来的带离群值矩阵结论不同。**量化的效果取决于具体的权重，必须在自己的模型上量**。

大模型上的参照：llama.cpp 的 perplexity 文档给出了 Llama 3 8B 在 WikiText-2 上的结果，f16 的困惑度 6.233，Q8_0 6.234，Q4_K_M 6.407（+0.175），Q4_0 6.700（+0.469）——同样是"8 bit 几乎无损，4 bit 有可见代价，K-quant 比老式 Q4_0 好"。我们的 0.7B 模型会不会更敏感，不能从 8B 的结果外推，发布前用 `llama-perplexity` 在自己的开发集上测，并且在工具调用开发集上比较 bf16 和 Q4_K_M 的得分，写进模型卡（`runs/RELEASE_CHECKLIST.md` C7–C8）。

### 3.5 小结：量化的三个旋钮

- **位数**：8 bit 几乎免费，4 bit 是笔记本上的常用折中，更低要谨慎；
- **分块大小**：块越小，离群值连累的数越少，但 scale 的开销越大（每块 32 个数时摊到 0.5 bit）；
- **哪些张量多给 bit**：对误差敏感的张量（输出层、部分 attn_v / ffn_down）用更高精度，这就是 Q4_K_M 里的"M"。

这里讲的都是**训练后的权重量化**（post-training quantization），不改训练。Gemma 3 等模型还发布了在训练中模拟量化的 QAT 版本（如 `gemma-3-1b-it-qat-q4_0-gguf`），不在本章范围内。

## 4. 本地运行：笔记本用 llama.cpp / Ollama，服务器用 vLLM

| 场景 | 工具 | 格式 | 我们的用法 |
|---|---|---|---|
| 笔记本、CPU、Mac | **llama.cpp** | GGUF | `llama-cli -m zero-Q4_K_M.gguf`；自带 OpenAI 兼容的 `llama-server` |
| 笔记本上一键安装 | **Ollama** | GGUF | `Modelfile` 里写 `FROM ./zero-Q4_K_M.gguf`，`ollama create zero` |
| GPU 服务器、高并发 | **vLLM** | HF 目录（safetensors） | `vllm serve <目录> --enable-auto-tool-choice --tool-call-parser hermes`（第 10、21 章的 PagedAttention） |

vLLM 的 `hermes` 解析器识别的正是 `<tool_call>{"name": …, "arguments": …}</tool_call>` 这种格式——Qwen 系列的工具调用模板就是这种风格，我们的模板（`zero/post/chat.py`）也是。所以同一份导出目录，vLLM 起服务后直接返回结构化的 `tool_calls`，不需要写自定义解析器（RUNBOOK 阶段 6 第 10 项在 GPU 上验证）。

**本地 demo** 是"真正可用"的直接证明：`zero/demo/cli.py` 是一个命令行助手，循环是"用户输入 → 套模板生成 → 解析 `<tool_call>` → 在本地执行 → 结果作为 tool 消息喂回 → 直到模型给出不含工具调用的回答"。本地工具有计算器（AST 白名单求值，不用 `eval`）、日期计算、今天的日期，以及 `search_files`——它只在 `--root` 目录内搜索：解析后的真实路径必须在 root 之下（挡住 `../` 逃逸），不跟随指向外部的符号链接，单个文件只读前 1 MB（`tests/test_demo.py` 覆盖这几种情况）。让一个模型在你电脑上执行动作时，**工具的边界比模型的聪明更重要**。

## 5. 模型卡与许可证

### 5.1 模型卡写什么

模型卡（model card，Mitchell 等人 2019 年提出）就是模型仓库首页的 `README.md`：开头一段 YAML 元数据（`license`、`language`、`datasets`、`base_model`、`pipeline_tag`……，Hugging Face 用来检索和展示），后面是给人看的正文。头部模型家族的模型卡都有这几块：架构与参数量、训练数据、使用方法、评测表、局限（Qwen3、Llama 3.2、Gemma 3、OLMo 2、SmolLM3 的模型卡都可以对照看，链接在文末）。GOAL.md 3.5 在此基础上要求得更多：**每个阶段的配方与花费、预注册协议、全部评测结果（包括落后的项）、去污染检查**。

`04_model_card.py` 从评测结果 JSON 生成一份骨架，原则是**只填有来源的数**：评测表直接从 JSON 生成，没有数据的地方一律写"待训练"，脚本最后数一遍还剩几处占位（发布前必须清零）。用冒烟测试的结果跑一遍（极小配置演示）：

```
---
license: other  # 待定
language: [zh, en]
tags: [function-calling, tool-use, from-scratch, gguf]
library_name: transformers
pipeline_tag: text-generation
---
# zero-0.7b（暂名）
> ⚠️ 极小配置演示：下面的数字来自 CPU 上约 1.3M 参数的冒烟测试……
## 模型概要 / 使用方法 / 训练数据与许可证 / 各阶段配方与花费 / 预注册
## 评测结果（全部列出，包括落后的项）
…合计：超过 1 项、持平 5 项、落后 0 项（全部列出，不挑选）。
### 发布后新增对手 / 去污染检查 / 已知局限 / 引用与致谢
<!-- 还有 15 处'待训练'，发布前必须全部填上 -->
```

### 5.2 许可证：作者决定，这里摆出选项

权重的许可证由作者（项目负责人）决定，下面只列选项和各自的含义：

| 选项 | 谁在用 | 含义 |
|---|---|---|
| Apache-2.0 | Qwen3、OLMo 2、SmolLM3 | 宽松，允许商用和修改；有专利授权条款；需要保留许可证和版权声明 |
| MIT | DeepSeek-R1 等 | 更短更宽松，只要求保留版权声明 |
| 自定义社区许可证 | Llama 3.2（`llama3.2`）、Gemma（`gemma`） | 附加使用政策和条件（如用户规模、可接受用途），不是 OSI 意义上的开源许可证 |

我们的目标是"模型、数据配方、代码、中间 checkpoint 全部公开"（GOAL.md 1 节），和 Apache-2.0 的完全开放家族（OLMo、SmolLM）一致；但最终怎么选，写进模型卡前由作者确认。

**数据的署名义务**和权重许可证是两回事，不管权重选什么都要履行。预训练候选数据的许可证（Hugging Face 数据集卡）：FineWeb-Edu、FineWeb-2、FineMath 是 **ODC-By 1.0**（Open Data Commons Attribution：可以用，但要署名；FineWeb 系列的卡片还写明使用同时受 Common Crawl 使用条款约束），DCLM-baseline 是 **CC-BY-4.0**（同样要求署名），Stack-Edu 来自 The Stack v2，数据集卡没有统一的许可证标签，要按原始代码的许可证处理（待核实）。蒸馏的教师模型要记录名称、版本，以及它的许可证是否允许"用输出训练其他模型"（第 17 章）。这些全部列进模型卡的数据表。

## 6. 发布之后：收反馈，不夸大

- **开一个收集失败例子的地方**（issue 模板：输入、工具定义、模型输出、期望输出）。小模型的工具调用失败往往集中在几类（参数类型错、该调不调、编造工具结果），比评测表更能指导下一版。
- **别人复现的分数和我们不一样时**，先对齐框架版本、模板、解码参数（这正是预注册里写死它们的原因），确认后更新模型卡，**写明日期和原因**。
- **宣传用语跟着判定表走**：只在"超过"的格子上说"超过"，并注明基准、版本、对手清单；"持平"就说持平；"落后"写进局限。极小配置演示的数字永远不出现在宣传里。

## 7. 小结

- **闸门 3** = 按预注册协议跑、按置信区间判：`lo > 0` 超过，`hi < 0` 落后，其余持平；全部列出，没达标不宣称"超过"，发布后新增的对手也照实写。
- **标准格式**：safetensors + config + 分词器 + chat template，架构名兼容 Qwen3，transformers / vLLM / llama.cpp 直接能用。
- **量化**：内存 = 参数 × bit；分块 + 每块一个 fp16 scale（Q8_0 = 8.5 bit，Q4_K = 4.5 bit），Q4_K_M 给敏感张量多留 bit；8 bit 几乎无损，4 bit 有代价，更低是悬崖——要在自己的模型上量。
- **本地运行**：笔记本 llama.cpp / Ollama（GGUF），服务器 vLLM（safetensors）；demo 的工具边界要严。
- **模型卡**：只填有来源的数，没有的写"待训练"；许可证由作者定，数据署名义务照办。

---

## 从极简到生产级

| 极简版（`code/`） | 生产级（`zero/`） | 多做了什么、为什么 |
|---|---|---|
| `01` 手写分块量化 | llama.cpp 的 `llama-quantize`（`zero/export/gguf.py` 的 `quantize`） | 真正的 k-quant 会搜索更优的 scale、按张量混合类型（Q4_K_M），还有高效的 CPU/GPU 内核；我们不重写，调官方工具 |
| — | `zero/hf.py` 的 `export_to_hf_qwen3(model, config, out_dir, tokenizer, dtype, chat=True)` | 参数名映射成 HF Qwen3 的名字；共享 embedding 时不重复保存 `lm_head`；写 `config.json`（同时写新旧两种 RoPE/YaRN 字段）、`generation_config.json`（对话模型的结束符是 `<|im_end|>`）、分词器和 chat template |
| — | `zero/export/gguf.py` 的 `convert_hf_to_gguf` / `build_llama_cpp` / `quantize` / `run_llama` / `llama_tokenize` | 用 llama.cpp **官方**的 `convert_hf_to_gguf.py`，只打一个运行时补丁：我们自训分词器的"预切分规则哈希"不在官方表里，不认识时按 `qwen2` 处理（我们的预切分正则与 Qwen2 相同），不改 llama.cpp 的任何文件 |
| `03` 内存计算器 | `zero/tools/count_params.py`、`zero/tools/kv_cache_calc.py` | 同样的账，接主线配置 |
| `04` 模型卡生成器 | `zero/eval/report.py` 的 `results_table` / `comparison_table` / `write_report` | 评测结果 → Markdown 表，模型卡和课程正文直接引用 |
| — | `zero/eval/harness.py`、`zero/eval/bootstrap.py`、`zero/eval/bfcl.py` | 内部评测、配对 bootstrap 与判定；BFCL 适配层（尚未验证） |
| — | `zero/demo/cli.py` | 本地工具调用助手：生成 → 解析 → 执行 → 喂回；`search_files` 限制在 root 内 |

**对拍**（保证导出的东西和训练的模型是同一个模型）：

- `tests/test_export_hf.py`：zero 模型导出后用 transformers 加载，logits 与 zero 一致（fp32 下 `rtol=atol=1e-5`，含共享 / 不共享 embedding、YaRN 三种配置）；导出目录也能读回 zero。
- `tests/test_gguf.py`：官方脚本转出的 GGUF 用 `llama-tokenize` 编码中文、英文、代码、工具调用四种文本，token id 与我们的分词器逐个相同；f32 GGUF 用 `llama-simple` 贪心生成，与 zero 逐 token 一致（模型带 YaRN 配置）；Q8_0、Q4_K_M 量化后能跑。
- `tests/test_demo.py`：`search_files` 挡住 `../` 和符号链接逃逸；生成 → 执行 → 喂回的循环；tiny 模型端到端跑通。

本章写作时我们用 `out/smoke/` 里已有的文件和编译好的 llama.cpp（commit `81bc6b8`）**重新核实了一遍**，全部是真实输出：

- 提示词 `<|im_start|>user\n3 * (4 + 5) 等于多少？<|im_end|>\n<|im_start|>assistant\n`：`llama-tokenize` 在 `zero-tiny-Q8_0.gguf` 上给出的 20 个 token id 与 zero 分词器完全相同（`[1, 449, 214, 34, 773, …, 1, 575, 214]`）；
- 同一提示词贪心生成 24 个 token：zero（fp32 HF 目录）、`zero-tiny-f16.gguf`、`zero-tiny-Q8_0.gguf` 三者输出**逐字相同**：`<tool_call>\n{"name":233 = 2 = 14 = 24 = 18)`（tiny 模型只学会了"先输出 `<tool_call>`"，后面是乱的——这是 1.3M 参数应有的水平）；
- 这 24 步里有一步非常"险"：第 23 个 token，zero 给 `8` 的 logit 是 7.888，第二名 `4` 是 7.862，只差 0.026，Q8_0 仍然选对了；
- 我们另把 `zero-tiny-f16.gguf` 量化成 Q4_K_M：tiny 模型的行长 128 不是 256 的倍数，k-quant 用不了，`llama-quantize` 自动回退成 25 个 Q5_0 + 5 个 Q8_0 张量（6.39 bit/权重，1.12 MB）。它能跑，但生成到第 10 个 token 就和 f16 分叉，输出 `<tool_call>\n{"name":233 =）。` 后结束——即使是 5–8 bit 的混合，在这个没训好的小模型上也会改变贪心路径。主线模型的宽度 1280 = 5 × 256，没有回退问题。
- 这台共享机器上 llama.cpp 单线程只有 0.14 token/s（CPU 被大量占用），**不代表**笔记本上的速度；真实速度在发布时按清单 C10 测。

---

## 主线进度

### 极小配置演示（CPU，`configs/tiny`，约 1.3M 参数）

> 以下是**极小配置演示**：只说明导出、量化、demo 的代码通路是通的，不代表主线模型的任何结果。

来自 `uv run python -m zero.smoke` 的真实输出（`out/smoke/SUMMARY.md`，单线程，导出阶段 57.7s）：

| 项目 | 结果 |
|---|---|
| HF 导出（`hf_chat/`） | transformers 加载后 logits 最大误差 `0.0e+00`（fp32 导出）；`apply_chat_template` 与我们的模板逐字一致 |
| GGUF f16 | 2.7 MB |
| GGUF Q8_0 | 1.47 MB |
| llama-simple 试跑 | 成功（`llama_simple_ok=True`） |
| 本地 demo | 跑通，但**回答是空的**（`answer=`）：模型直接输出了结束符，没有调用工具 |

demo 的空回答要如实说：tiny 模型在"系统提示 + 工具列表 + 用户问题"这样的长提示下，第一个 token 就生成了 `<|im_end|>`。我们写作时用同一个模型再问了一次"3 * (4 + 5) 等于多少？"（`--root chapters/20-release`），结果仍然是空回答。评测表里它的 `tool_dev call_exact` 只有 0.10–0.13（30 题），与此一致。第一步的 demo 只证明"加载 → 生成 → 解析 → 执行 → 喂回"这条链路能跑，真正好用要等第二步的主线模型。

### 待 GPU 训练后补充

- 按预注册协议的最终评测表（每个对手 × 每个基准，含置信区间与判定）、硬目标是否成立、差距分析；
- 发布后新增对手；
- 去污染检查的命中率与剔除数；
- 主线模型的 GGUF 实际大小、Q8_0 / Q4_K_M 相对 bf16 的 PPL、KLD 与工具调用得分，笔记本上的内存与速度；
- 本地 demo 的录屏（成功和失败的例子）；
- Hugging Face 上的仓库链接（Base、SFT、最终版、中间 checkpoint、GGUF）与最终模型卡；
- 花费（GOAL.md 3.4 给第 20 章的预算约 $200，主要是最终评测）。

### 闸门 3 检查清单（GOAL.md 3.4：发布前）

完整清单见 [`runs/RELEASE_CHECKLIST.md`](../../runs/RELEASE_CHECKLIST.md)（A 评测、B 去污染、C 权重与格式、D demo、E 许可证、F 发布物），要点：

- [ ] 按 `eval/PREREGISTRATION.md` 冻结的基准、框架版本、模板、解码参数跑完全部评测，保存逐题结果；
- [ ] 与每个对手做配对 bootstrap（对手取思考 / 非思考中较高者），全部列出"超过 / 持平 / 落后"；
- [ ] 没达到硬目标就不宣称"超过"，写差距分析；
- [ ] 发布后再查冻结日之后的新模型，写"发布后新增对手"；
- [ ] 去污染（13-gram、工具函数名 / schema）结果写进模型卡；
- [ ] 最终权重再做一遍 HF logits 对拍、`llama-tokenize` 对拍、f32 GGUF 贪心对拍；Q8_0 / Q4_K_M 的 PPL 与工具调用得分；
- [ ] Q4_K_M 在普通笔记本上用 llama.cpp 和 Ollama 跑通，demo 录屏；
- [ ] 许可证由作者选定，数据署名表完成；模型卡里"待训练"清零；
- [ ] 花费记入 `runs/ledger.md`；
- [ ] **等项目负责人最后确认再发布**。

---

## 采用方与来源

| 技术 / 做法 | 类别（GOAL.md 2.1） | 采用方与来源 |
|---|---|---|
| GGUF + llama.cpp 本地推理与量化 | B 行业标准 | llama.cpp（[GitHub](https://github.com/ggml-org/llama.cpp)；[GGUF 规范](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md)）。官方发布 GGUF 的家族：Qwen（[Qwen3-0.6B-GGUF](https://huggingface.co/Qwen/Qwen3-0.6B-GGUF)）、Google（[gemma-3-1b-it-qat-q4_0-gguf](https://huggingface.co/google/gemma-3-1b-it-qat-q4_0-gguf)）、Ai2（[OLMo-2-0425-1B-GGUF](https://huggingface.co/allenai/OLMo-2-0425-1B-GGUF)）；Ollama 直接导入 GGUF（[import 文档](https://github.com/ollama/ollama/blob/main/docs/import.mdx)） |
| safetensors 权重格式 | B 行业标准 | [safetensors](https://github.com/huggingface/safetensors)；Qwen3、OLMo 2、Llama 3.2、Gemma 3、SmolLM3 的官方仓库都以 safetensors 发布（见下方模型卡链接的文件列表） |
| vLLM 服务 + 工具调用解析 | B 行业标准 | [vLLM](https://github.com/vllm-project/vllm)，Kwon 等 *PagedAttention*（[arXiv:2309.06180](https://arxiv.org/abs/2309.06180)）；[工具调用文档](https://github.com/vllm-project/vllm/blob/main/docs/features/tool_calling.md)（`--enable-auto-tool-choice`、Qwen 用 `hermes` 解析器） |
| 权重量化（分块、k-quant） | B 行业标准（GOAL.md 2.1 推理行） | 块结构：llama.cpp `ggml/src/ggml-common.h`；Q4_K_M 的张量规则：`src/llama-quant.cpp`；质量 / 大小数据：llama.cpp [`tools/quantize/README.md`](https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md)、[`tools/perplexity/README.md`](https://github.com/ggml-org/llama.cpp/blob/master/tools/perplexity/README.md)；k-quants 原始 PR [#1684](https://github.com/ggml-org/llama.cpp/pull/1684) |
| 模型卡（YAML 元数据 + 数据、评测、局限） | A 多家采用 | [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)、[Llama-3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B)、[gemma-3-1b-it](https://huggingface.co/google/gemma-3-1b-it)、[OLMo-2-0425-1B](https://huggingface.co/allenai/OLMo-2-0425-1B)、[SmolLM3-3B](https://huggingface.co/HuggingFaceTB/SmolLM3-3B)；[HF 模型卡文档](https://huggingface.co/docs/hub/model-cards)；Mitchell 等 *Model Cards for Model Reporting*（[arXiv:1810.03993](https://arxiv.org/abs/1810.03993)） |
| 发布中间 checkpoint | A（完全开放家族） | OLMo 2（同一仓库的 `revision` 分支，[模型卡](https://huggingface.co/allenai/OLMo-2-0425-1B)）、SmolLM3（[SmolLM3-3B-checkpoints](https://huggingface.co/HuggingFaceTB/SmolLM3-3B-checkpoints)）、Pythia（Biderman 等，[arXiv:2304.01373](https://arxiv.org/abs/2304.01373)，每个模型 154 个 checkpoint） |
| Base / SFT / 最终版分开发布 | A 多家采用 | OLMo 2（Base、SFT、DPO、Instruct 四个仓库）、Qwen3（[Qwen3-0.6B-Base](https://huggingface.co/Qwen/Qwen3-0.6B-Base) 与 Qwen3-0.6B）、Gemma 3（`-pt` 与 `-it`）、SmolLM3（Base 与 3B） |

## 前沿观察

> **量化感知训练（QAT）**：训练的最后一段在前向里模拟量化误差，让模型"习惯"4 bit，再发布量化版。Google 为 Gemma 3 发布了 QAT 版的 Q4_0 GGUF，gpt-oss 的 MoE 权重以 MXFP4 发布（待核实其训练流程细节）。它需要改训练流程、额外的算力，采用方还不多，我们的主线用训练后量化（PTQ）。
>
> **重要性矩阵（imatrix）**：用一小段校准文本统计每个权重对输出的影响，量化时优先保护重要的权重。llama.cpp 的 perplexity 文档里，Llama 3 8B 的 Q4_K_M 加 imatrix 后 ΔPPL 从 0.175 降到 0.151。社区量化版本广泛使用，但校准数据的选择会引入新的变量（例如和评测集的重叠），我们的发布版先不用；若使用，校准数据也要进去污染检查。

## 引导问题

1. 如果 BFCL 总分我们"超过"了 Qwen3.5-0.8B，但中文工具调用基准"落后"，按预注册应该怎么写结论？如果预注册里没写这种情况怎么办（提示：修订记录能不能用来补救？）
2. 配对 bootstrap 为什么要两个模型用同一组题下标？如果分别独立重抽，置信区间会变宽还是变窄？
3. `01_blockwise_quant.py` 里把块大小从 32 改成 16、64、128，INT4 的误差和 bit/权重会怎样变？找出误差和大小的折中点。
4. KV cache 在 32K 上下文时（3.5 GiB）比 Q4_K_M 的权重（0.40 GiB）大得多。有哪些办法压 KV cache？（提示：第 21 章；llama.cpp 的 `--cache-type-k q8_0`）
5. 为什么 Q4_K_M 偏偏给 attn_v、ffn_down 和输出层多留 bit？可以让 Claude Code 帮你在 `02` 里只把某一类张量保持 8 bit，看哪一类对 loss 影响最大。
6. 模型许可证选 Apache-2.0，但预训练数据是 ODC-By、CC-BY——这两件事冲突吗？署名义务落在谁身上、写在哪里？

## 动手任务

**任务 1（基础）**：运行 `03_memory_calculator.py configs/ladder/l300m.toml`（阶梯实验的 300M 模型），算出它在 Q4_K_M 下的大小；再把 `configs/main/pretrain.toml` 的词表从 65,536 改成 151,936（Qwen3 的词表）在内存里重算一遍（不要改仓库里的文件，复制一份），看 embedding 的 Q6_K 让总大小涨了多少。

**任务 2（核心）**：在 `02_quantize_tiny_model.py` 里加一个"混合精度"方案：embedding 用 INT8 分块，其余矩阵用 INT4 分块（模仿 Q4_K_M 给输出层多留 bit），和纯 INT4 分块比较 loss 与大小。再试试反过来（embedding INT4、其余 INT8），哪一种更划算？

**任务 3（挑战）**：如果你跑过 `uv run python -m zero.smoke`，并按 `zero/export/gguf.py` 编译了 llama.cpp：用 `llama-quantize` 把 `out/smoke/gguf/zero-tiny-f16.gguf` 量化成 Q4_0、Q5_0、Q8_0，用 `llama-simple` 在同一个提示词上贪心生成，记下每种格式和 f16 从第几个 token 开始分叉。再用 `04_model_card.py --out` 生成一张模型卡，把"已知局限"一节按你看到的结果补上。

## 想深入：CS336

本章对应斯坦福 CS336（Spring 2026）<https://cs336.stanford.edu/>：

- **第 10 讲：推理（Inference）**。推理的算力与带宽账（prefill 与 decode）、KV cache、量化等降低推理成本的方法，是本章"为什么量化、为什么 4 bit 生成更快"的背景。
- **第 12 讲：评测（Evaluation）**。基准怎么设计、怎么避免污染、怎么比较，是本章闸门 3 与判定规则的背景（本课第 11 章是它的入门）。
- GGUF / k-quant 的具体格式、模型卡与许可证、发布流程，CS336 未深入，见本章参考文献。

---

## 本章参考文献

- ggml-org. *llama.cpp*：<https://github.com/ggml-org/llama.cpp>（本章核对的 commit `81bc6b8`；`ggml/src/ggml-common.h` 的块结构，`src/llama-quant.cpp` 的 Q4_K_M 规则，`tools/quantize/README.md`、`tools/perplexity/README.md` 的大小与困惑度表）
- ggml-org. *GGUF 文件格式*：<https://github.com/ggml-org/ggml/blob/master/docs/gguf.md>
- ggml-org. *k-quants*（PR #1684）：<https://github.com/ggml-org/llama.cpp/pull/1684>
- Hugging Face. *safetensors*：<https://github.com/huggingface/safetensors>
- Hugging Face. *Model Cards*（Hub 文档）：<https://huggingface.co/docs/hub/model-cards>
- Mitchell et al. *Model Cards for Model Reporting*，2019：<https://arxiv.org/abs/1810.03993>
- Kwon et al. *Efficient Memory Management for Large Language Model Serving with PagedAttention*（vLLM），2023：<https://arxiv.org/abs/2309.06180>
- vLLM. *Tool Calling*：<https://github.com/vllm-project/vllm/blob/main/docs/features/tool_calling.md>
- Ollama. *Importing a model*：<https://github.com/ollama/ollama/blob/main/docs/import.mdx>
- Dettmers et al. *LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale*（离群值问题），2022：<https://arxiv.org/abs/2208.07339>
- Frantar et al. *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers*，2022：<https://arxiv.org/abs/2210.17323>
- Groeneveld / OLMo Team. *2 OLMo 2 Furious*（完全开放发布：数据、代码、checkpoint），2024：<https://arxiv.org/abs/2501.00656>
- Biderman et al. *Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling*，2023：<https://arxiv.org/abs/2304.01373>
- 模型卡示例：[Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)、[Llama-3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B)、[gemma-3-1b-it](https://huggingface.co/google/gemma-3-1b-it)、[OLMo-2-0425-1B](https://huggingface.co/allenai/OLMo-2-0425-1B)、[SmolLM3-3B-checkpoints](https://huggingface.co/HuggingFaceTB/SmolLM3-3B-checkpoints)
- 数据集许可证：[FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)（ODC-By）、[FineWeb-2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2)（ODC-By）、[FineMath](https://huggingface.co/datasets/HuggingFaceTB/finemath)（ODC-By）、[DCLM-baseline](https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0)（CC-BY-4.0）、[Stack-Edu](https://huggingface.co/datasets/HuggingFaceTB/stack-edu)（待核实）；[ODC-By 1.0 原文](https://opendatacommons.org/licenses/by/1-0/)
- [CS336](https://cs336.stanford.edu/) 第 10、12 讲

**下一章**：第四部分到这里结束，主线模型从"一堆权重"变成了能在笔记本上调用工具的助手。第五部分换一个方向：我们刚才算出，32K 上下文时 KV cache 要 3.5 GiB，比 4 bit 的权重大近 9 倍。第 21 章从这本"KV cache 的账本"讲起：MQA、GQA、MLA 是怎样一步步把它压小的。
