# 第 25 章：多 token 预测与推测解码 —— 让小模型先猜，大模型一次改完

> **一句话目标**：读完这一章，你能写出推测解码的"猜—验—回滚"循环，并验证贪心时输出与目标模型逐字相同；能从 `min(1, p/q)` 和残差分布 `max(0, p − q)` 推出"采样时分布一点不变"，并用卡方检验在小词表上验证它；能用 `(1 − α^{k+1}) / ((1 − α)(1 + k·c))` 估算加速比；还能说清楚 DeepSeek-V3 的 MTP 模块长什么样、为什么它既能让训练更好、又能在推理时当草稿。

📺 **本章视频**：待发布（本地渲染：`bash chapters/25-mtp-speculative-decoding/video/build.sh`）
🧪 **本章自检**：学完后在 Claude Code 里输入 `/ch25-speculative`

---

上一章的混合专家（MoE）让模型参数很多、每个 token 却只用其中一小部分，省下了每个 token 的计算。但不管用的是稠密模型还是 MoE，生成时都还是**一个 token、一个 token 地往外蹦**：第 10 章说过，decode 每一步只算一个新 token，却要把全部权重和 KV cache 从显存里读一遍，硬件大部分时间在等数据。这一章要解决的问题是：**能不能让大模型一次前向就吐出好几个 token，而且输出的质量一点都不变？** 答案是推测解码（speculative decoding）：让一个便宜的草稿先猜几个 token，大模型一次前向把它们全部检查一遍。草稿从哪来？可以是一个小模型，也可以是大模型自己训练时顺手学会的"多 token 预测"（Multi-Token Prediction，MTP）模块——DeepSeek-V3 之后，千问、GLM、MiniMax、小米 MiMo 的主力模型都带上了它。

本章代码：

```bash
uv run python chapters/25-mtp-speculative-decoding/code/01_models_and_cost.py    # 目标/草稿模型 + "验证 k 个和生成 1 个一样贵"
uv run python chapters/25-mtp-speculative-decoding/code/02_greedy_speculative.py # 贪心推测解码：逐字相同 + 接受率 + 速度
uv run python chapters/25-mtp-speculative-decoding/code/03_speculative_sampling.py # 拒绝采样保持分布：卡方检验 + 真实模型
uv run python chapters/25-mtp-speculative-decoding/code/04_speedup_formula.py    # 加速比公式与最优 k（纯计算，瞬间）
uv run python chapters/25-mtp-speculative-decoding/code/05_mtp.py                # MTP 模块：训练 + 自推测解码
```

目标模型直接用第 10 章训练好的小模型（4 层、宽 128、字符级 65 个字符、0.86M 参数，验证集 loss 1.838）；草稿模型是同一套结构、**同一个词表**，只有 1 层、宽 64（57,600 个参数，约为目标的 1/15，验证集 loss 1.966）。第一次运行会训练草稿模型并缓存到 `code/out/`。

> 本章所有计时都在一台被多个任务共享的 4 核机器上单线程测得（测量时负载在 16–33 之间）。为了减少"排队等 CPU"的干扰，计时用的是**进程 CPU 时间**（`time.process_time()`）而不是墙钟时间，但仍有明显波动；表里的计时只用来看**数量级和趋势**。接受率、前向次数、分布检验这些数固定随机种子，可复现。

## 1. 问题：decode 在等数据，不在算

回忆第 10 章的一个数字：同一个小模型，prefill 一次喂 256 个 token 的吞吐是 decode 一次喂 1 个的 41 倍。第 21 章把原因算清楚了：decode 每生成一个 token，要把全部权重和整个 KV cache 读一遍，却只做很少的乘加，"算力 ÷ 读取字节"（算术强度）只有 1–5，而 H100 要到约 295 才能把算力吃满。**decode 阶段，GPU 在等数据**。

这就留下了一个空子：既然读一遍权重的时间是固定的，那**一次多喂几个 token 几乎不花额外时间**——读的字节没变，多出来的那点计算本来就闲着。`01_models_and_cost.py` 在 CPU 上测了这件事：已经有 200 个位置的 KV cache，目标模型一次前向喂 T 个新 token 要多久：

| 一次喂 T 个 token | 1 | 2 | 3 | 5 | 9 | 17 |
|---|---|---|---|---|---|---|
| 目标模型耗时（ms） | 3.55 | 4.13 | 4.44 | 4.60 | 4.85 | 6.08 |
| 相对 T = 1 | 1.00× | 1.16× | 1.25× | 1.30× | 1.36× | 1.71× |
| 草稿模型耗时（ms） | 0.88 | 0.88 | 1.05 | 0.92 | 1.07 | 1.06 |

计算量涨了 17 倍，时间只涨了 71%。（在这个极小模型、单线程 CPU 上，时间主要花在 Python 和 PyTorch 每个算子的固定开销上，而不是显存带宽；但结论的形状一样：**检查一段话比写一段话便宜得多**。）

问题是：我们不知道后面几个 token 是什么，没法"一次喂好几个"。除非——有人先猜。

## 2. 推测解码：草稿先猜，目标一次检查

推测解码（Leviathan 等 2023；DeepMind 的 Chen 等 2023 同时提出，叫 speculative sampling）借用了 CPU 的"推测执行"：先按猜测往下跑，猜错了再撤回。每一轮：

1. **草稿（draft）猜 k 个**：一个便宜的模型自回归地生成 k 个 token `d₁ … d_k`；
2. **目标（target）一次检查**：把"最后一个已确定的 token + k 个草稿"一次喂给目标模型，一次前向得到 k+1 个位置的预测；
3. **从左往右比**：`d_i` 和目标在这个位置的答案一致就接受；遇到第一个不一致的，停下，改用目标自己的答案（**纠正**）；k 个全对时，目标在最后一个位置的答案白送一个（**奖励**）；
4. **回滚**：被拒绝的草稿已经写进了两个模型的 KV cache，要扔掉。

`02_greedy_speculative.py` 里，贪心版本的核心就是这几行：

```python
feed = seq[len(tc) :] + drafts                                   # 缓存里还没有的已确定 token + k 个草稿
p_logits = target(torch.tensor([feed]), tc)[0, -(k + 1) :]       # 一次前向，最后 k+1 个位置
choice = p_logits.argmax(-1).tolist()                            # 目标在每个位置的答案
m = 0
while m < k and drafts[m] == choice[m]:                           # 从左往右接受
    m += 1
seq += drafts[:m] + [choice[m]]                                  # m 个草稿 + 1 个纠正（或奖励）
truncate(tc, len(seq) - 1)                                       # 回滚：扔掉被拒草稿的 K/V
```

这里有一个真实的例子（视频 S03 用的就是它）：提示词是验证集里的一段莎士比亚，草稿猜了 4 个字符，目标模型接受了前两个，第三个不同意、换成了自己的答案——这一轮只跑了一次目标模型，就产出了 3 个字符。

**每一轮至少产出 1 个 token**（最坏情况第一个草稿就被拒，拿到目标自己的答案，和普通解码一样），**最多 k+1 个**。目标模型每轮只跑一次，所以目标的前向次数只会减少、不会增加；多花的只是草稿的时间。

## 3. 贪心：为什么输出逐字相同

先看最简单的情况：目标模型用贪心解码（每步取 argmax）。推测解码的每个输出 token，要么是"和目标 argmax 一致的草稿"，要么是"目标的 argmax"——**全都是目标模型在同样前缀下会选的那个 token**。所以输出和目标模型自己一步一步贪心解码**逐字相同**。草稿只决定"一次能前进几步"，不决定"写什么"。

这句话要靠代码验证，因为很容易写错——最常见的错误是 KV cache 没回滚干净，被拒草稿的 K/V 留在缓存里，后面的注意力"看见"了从未发生的历史。`02` 在 4 段提示词、每段 200 个字符、k = 1、3、5、8 下对拍：

```
k=1：4 段提示词 × 200 个 token，与目标模型贪心解码逐字相同：True
k=3：4 段提示词 × 200 个 token，与目标模型贪心解码逐字相同：True
k=5：4 段提示词 × 200 个 token，与目标模型贪心解码逐字相同：True
k=8：4 段提示词 × 200 个 token，与目标模型贪心解码逐字相同：True
```

回滚的写法取决于缓存的实现。第 10 章的极简缓存用 `torch.cat` 拼接，回滚就是切片：`cache.k[layer] = cache.k[layer][:, :, :n]`。`zero` 的缓存是预分配的（第 10 章），`KVCache.update(layer, start_pos, k, v)` 按位置覆盖写入、只返回 `[0, start_pos + T)`，所以回滚连数据都不用动，只要把"有效长度"退回去，下次从那个位置开始写就行。

> 严格说"逐字相同"要求浮点计算也完全一样。目标模型一次喂 k+1 个 token 和一次喂 1 个 token，矩阵乘法的分块顺序可能不同，末位舍入可能不同；如果两个候选的 logit 恰好几乎相等，argmax 可能翻转。本章的 FP32 小模型上没有遇到；vLLM 的文档也专门说明，它的实现"在算法上无损"，但 GPU 上不同 batch 大小的数值差异可能导致输出不完全一致。

## 4. 采样：拒绝采样让分布一点不变

贪心好办，采样呢？目标模型在某个位置给出分布 `p`，草稿给出的是另一个分布 `q`。草稿按 `q` 抽出一个 token `x`，我们希望最后留下的 token 服从 `p`，而不是 `q`。

规则只有两条：

1. 以概率 `min(1, p(x) / q(x))` **接受** `x`；
2. 被拒绝时，从**残差分布** `p'(x) = max(0, p(x) − q(x)) / Σ max(0, p − q)` 里重抽一个，本轮到此为止。

直觉：`p(x) ≥ q(x)` 的 token，草稿给得"不够多"，一律接受；`p(x) < q(x)` 的 token，草稿给得"太多了"，按比例砍掉多出来的部分。砍掉的概率质量去哪了？正好补给那些草稿给得不够的 token——残差分布只在 `p > q` 的地方有质量。

**推导**：最终抽到某个 token `x` 的概率 = "草稿抽到 x 并被接受" + "被拒绝后从残差里抽到 x"：

```
P(草稿抽到 x 且接受) = q(x) · min(1, p(x)/q(x)) = min(p(x), q(x))
P(被拒绝)            = 1 − Σ_x min(p(x), q(x)) = 1 − α
P(被拒后抽到 x)       = (1 − α) · max(0, p(x) − q(x)) / (1 − α)  = p(x) − min(p(x), q(x))
两项相加              = p(x)                                                  ✓
```

（倒数第二行用到 `Σ max(0, p − q) = Σ (p − min(p, q)) = 1 − α`。）顺便得到一个重要的量：**一次接受的概率** `α = Σ_x min(p(x), q(x)) = 1 − TV(p, q)`，TV 是两个分布的总变差距离。草稿和目标越像，α 越高。

`03_speculative_sampling.py` 里，这条规则就是：

```python
def accept_or_resample(p, q, x, g):
    if torch.rand((), generator=g) < torch.clamp(p[x] / q[x], max=1.0):  # 以 min(1, p/q) 接受
        return True, x
    residual = torch.clamp(p - q, min=0)                                  # 残差 max(0, p − q)
    return False, int(torch.multinomial(residual / residual.sum(), 1, generator=g))
```

**实验一：一步，6 个词，抽 20 万次。** ⟨TOY_TABLE⟩

**实验二：完整算法。** 上面只验证了一个位置。完整的推测解码还有"前面的草稿被接受了、后面的才会被检查"和"全部接受时的奖励 token"。`03` 用一对马尔可夫链当"语言模型"（下一个 token 只看上一个，目标和草稿是两个随机转移矩阵），k = 2，生成长度 3 的序列 6 万次，和目标模型的精确联合分布（4³ = 64 个格子）比：TV 距离 0.0083，卡方统计量 50.5（自由度 49，合并了期望次数小于 5 的格子），p 值 0.41；平均每轮产出 1.50 个 token。完整算法同样没有改变分布。

**实验三：真实模型，温度 1.0。** ⟨REAL_SAMPLING⟩

两个实用细节：

- **温度和 top-p 要对两边同样处理**。`p`、`q` 指的是"经过同样温度、top-p 处理之后"的分布；推测解码保证的是输出服从"处理后的目标分布"，也就是你不用推测解码、用同样参数采样时得到的分布。`zero/arch/speculative.py` 的 `warp_probs` 就是这样做的。
- **放宽接受条件会损失精确性**。有些实现为了提高接受率，把规则放宽（例如 Leviathan 等论文附录里的 lenience 参数、Medusa 的 typical acceptance），换来更快，但输出分布不再严格等于目标分布。本章只讲严格版本。

## 5. 能快多少：α、k 和 c

假设每个草稿 token 被接受的概率都是 α、相互独立。一轮里：第 1 个产出 token 一定有；第 2 个要第 1 个草稿被接受（概率 α）；第 3 个要前两个都被接受（α²）……所以

```
每轮期望产出 E = 1 + α + α² + … + α^k = (1 − α^{k+1}) / (1 − α)
```

每轮的代价是目标模型 1 次前向加草稿 k 次前向。记 `c` = 草稿一步 ÷ 目标一步，**假设目标验证 k+1 个位置和生成 1 个一样快**，加速比就是（Leviathan 等 2023，定理 3.8）：

```
加速比 = (1 − α^{k+1}) / ((1 − α)(1 + k·c))
```

`04_speedup_formula.py` 把它算成表（c = 0.05 时的一部分）：

| α \ k | 1 | 2 | 3 | 4 | 6 | 8 | 最优 k |
|---|---|---|---|---|---|---|---|
| 0.5 | 1.43 | 1.59 | 1.63 | 1.61 | 1.53 | 1.43 | 3（1.63×） |
| 0.7 | 1.62 | 1.99 | 2.20 | 2.31 | 2.35 | 2.28 | 6（2.35×） |
| 0.9 | 1.81 | 2.46 | 2.99 | 3.41 | 4.01 | 4.38 | 13（4.67×） |

几条规律：α 越高、c 越小，越值得多猜；α 低时猜多了只是白白浪费草稿的算力（α = 0.5 时 k 超过 3 反而变慢）；c = 0 时加速比的上限是 `1/(1 − α)`。这个公式和论文对得上：α = 0.8、k = 5、c = 0 时是 3.69 倍，正是 Leviathan 等表 1 的数字。

**实测（贪心，4 段 × 200 个字符，`02_greedy_speculative.py`）**：普通贪心解码 800 个字符用了 2.73 秒 CPU 时间，成本系数 c ≈ 0.20。

| k | 逐 token 接受率 α | 每轮产出（实测） | 公式 (1) | 目标前向次数（普通解码 800） | CPU 时间 s | 加速比 | 公式预测 |
|---|---|---|---|---|---|---|---|
| 1 | 0.721 | 1.72 | 1.72 | 466 | 2.35 | 1.17× | 1.43× |
| 2 | 0.697 | 2.19 | 2.18 | 368 | 2.24 | 1.22× | 1.56× |
| 3 | 0.722 | 2.64 | 2.62 | 306 | 2.13 | 1.28× | 1.63× |
| 4 | 0.704 | 2.83 | 2.79 | 286 | 2.28 | 1.20× | 1.55× |
| 5 | 0.700 | 2.95 | 2.94 | 275 | 2.47 | 1.10× | 1.47× |
| 6 | 0.707 | 3.10 | 3.12 | 262 | 2.51 | 1.09× | 1.41× |
| 8 | 0.703 | 3.32 | 3.22 | 244 | 2.76 | 0.99× | 1.23× |

（"逐 token 接受率"= 接受数 ÷ 被比较过的草稿数；第一个被拒之后的草稿没有被比较，不计入。）

这张表里有三件事值得看：

1. **公式 (1) 几乎完全对得上**：实测的每轮产出和 `(1 − α^{k+1})/(1 − α)` 差不到 0.1。接受率在不同 k 下都在 0.70–0.72，说明"每个草稿独立地以 α 被接受"在这里是个不错的近似。
2. **目标模型的前向次数实打实地少了**：k = 3 时从 800 次降到 306 次。这个数和硬件无关。
3. **CPU 上的加速远小于前向次数的减少**：最快的 k = 3 只快了 1.28 倍，比公式预测的 1.63 倍还低。原因都在公式的假设里：一是 c 不小——1 层的草稿一步要花目标的约五分之一，因为在这么小的模型上，时间主要是每个算子的固定开销，层数少了、宽度小了，开销却省不了多少；二是验证并不完全免费——第 1 节的表里一次喂 4 个 token 要慢约 1.3 倍，把它代进去（`2.64 / (1.28 + 3 × 0.2) ≈ 1.40`）就接近实测了；剩下的差距是 Python 循环、回滚、草稿补喂 token 的开销。k 再大，草稿的开销越积越多，k = 8 时已经比不用推测解码还慢。

所以：**这个 CPU 小实验能验证正确性、接受率和前向次数，但它的加速比不代表 GPU 上的情况**。

真实的 GPU 系统上，草稿通常比目标小两个数量级，c 能到 0.05 以下，验证 k+1 个 token 也几乎真的免费。Leviathan 等在 TPU 上用 T5-small（77M）给 T5-XXL（11B）当草稿，α 在 0.62–0.75，实测加速 2.3–3.4 倍。

**什么时候推测解码不划算**：一是 α 太低（草稿和目标差太远，或者采样温度很高、分布很平）；二是**大批量高并发**时——服务端把几十条请求拼成一批 decode，算术强度已经上来了，"多验证几个位置几乎免费"不再成立，多出来的验证计算会挤占别人的算力。vLLM 的文档就写明推测解码主要用于"中低 QPS、带宽受限"的场景，也提供了按负载动态调整猜测长度的选项。

## 6. 草稿从哪来

推测解码对草稿**没有任何正确性要求**——草稿再烂，输出分布都不变，只是变慢。所以草稿可以五花八门，只有一条硬约束：**和目标模型用同一个分词器**（草稿提出的 token id，目标要能直接打分）。常见的三类：

1. **同家族的小模型**。同一个分词器、同样的训练数据，天然"想法相近"。Leviathan 等发现草稿比目标小两个数量级左右时，α 和 c 平衡得最好；CS336 的推理讲座举的搭配是 70B 配 8B、8B 配 1B，并建议用蒸馏让草稿更像目标。Google 给 Gemma 4 的 E2B、E4B、12B、26B-A4B、31B 都发布了专门的 MTP 草稿模型（`gemma-4-*-it-assistant`；31B 的草稿只有 4 层、与目标模型共享 KV cache），模型卡写的是"最多约 3 倍加速、输出质量与标准生成完全相同"。
2. **提示词查找（prompt lookup / n-gram）**。在已经有的文本里找和末尾几个 token 相同的片段，把它后面的 token 当草稿。代价几乎为零（c ≈ 0）。改写代码、总结文档、工具调用里重复参数名这类"大量复制上下文"的任务上 α 很高。Leviathan 等甚至发现一个 bigram 小表都能让 T5-XXL 翻译快 1.25 倍。`zero/arch/speculative.py` 在 `draft=None` 时就用它。
3. **模型自带的草稿头**。与其另外训练、部署一个模型，不如在目标模型身上长出一个"猜后面几个 token"的小部件。它能直接读目标模型最后一层的表示，信息比独立的小模型多得多，所以猜得准。这一类里最主流的是下一节的 MTP；Medusa、EAGLE 等方法见章末"前沿观察"。

## 7. MTP：训练时多学一步，推理时自带草稿

### 7.1 从"多个头"到"顺序的模块"

多 token 预测最早的形式（Gloeckle 等 2024，Meta）很直接：共享的主干后面接 n 个独立的输出头，第 j 个头预测往后第 j 个 token，损失加起来。DeepSeek-V3 做了一个关键改动：**按顺序预测，保留完整的因果链**。深度为 k 的 MTP 模块在位置 i 做的事是：

```
h'ᵏᵢ = Mₖ [RMSNorm(hᵏ⁻¹ᵢ) ; RMSNorm(Emb(t_{i+k}))]       # 上一层的表示 + 第 i+k 个 token 的 embedding
hᵏ   = TRMₖ(h'ᵏ)                                          # 一个 Transformer block（因果注意力）
Pᵏ_{i+k+1} = OutHead(hᵏᵢ)                                 # 与主模型共享的输出头
L_MTP = λ/D · Σₖ CrossEntropy(Pᵏ, t)                      # 加到主损失上
```

几个要点：

- **Embedding 和输出头与主模型共享**，MTP 模块自己只有两个 RMSNorm、一个 `2d → d` 的投影 `Mₖ` 和一个 Transformer block。在本章的小模型上，MTP 模块的参数是主模型的 ⟨MTP_PARAM_FRAC⟩。
- 模块 k 的输入里有 `t_{i+k}` 的 embedding——也就是说，预测 `t_{i+2}` 时它**知道** `t_{i+1}` 是什么（训练时是真实的 token，推理时是主模型刚选出的 token）。所以它不是在"盲猜两步之后"，而是在主模型的表示基础上**再多走一步**。这正是它当草稿格外准的原因。
- DeepSeek-V3 用 D = 1（只多预测一个 token），λ 在前 10T token 取 0.3、之后取 0.1。GLM-4.5 用了同样的 λ 设置（前 15T token 0.3、之后 0.1）。

### 7.2 训练：信号更密

MTP 的第一个目的是**让主模型训练得更好**：每个位置除了"下一个 token"，还要为"下下个 token"提供有用的表示，训练信号更密（DeepSeek 原文："densifies the training signals"），也可能让模型学会"提前规划"。DeepSeek-V3 的消融（表 4）在 15.7B 和 228.7B 两个规模的 MoE 上各训练一对模型，唯一区别是加不加 1 层 MTP：加了之后大多数基准变好（例如小模型 HumanEval 20.7 → 26.8、GSM8K 25.4 → 31.4），推理时丢掉 MTP 模块，成本完全相同。

`05_mtp.py` 在第 10 章的小模型上做了同样的事：同样的初始化、同样的数据顺序、同样的 600 步，只多了 `0.3 × L_MTP`。第 10 章缓存的两个模型（种子 0、1）正好是 λ = 0 的对照组：

⟨MTP_TABLE⟩

⟨MTP_DISCUSS⟩

`05` 里 MTP 模块的核心：

```python
x = self.proj(torch.cat([self.enorm(emb_next), self.hnorm(h)], dim=-1))  # [RMSNorm(Emb(t_{i+1})); RMSNorm(h_i)] → 投影
return self.norm(self.block(x, cos, sin, cache, 0))                       # 一个 block + RMSNorm，再乘共享的 Embᵀ
```

训练时一行：`loss = l_main + lam * l_mtp`，其中 `l_mtp` 的目标是 `y[:, 1:]`（y 是左移一位的目标序列，再左移一位就是"下下个"）。

### 7.3 推理：丢掉，或者当草稿

MTP 的第二个用途是**自推测解码**（self-speculative decoding）：主模型给出下一个 token `t_{i+1}` 的同时，MTP 模块拿主模型的表示和 `t_{i+1}` 的 embedding 猜 `t_{i+2}`；下一次前向把 `[t_{i+1}, 草稿]` 一起喂进去，既验证草稿，又顺手得到再下一个 token。k = 1 时每次前向期望产出 `1 + α` 个 token。DeepSeek-V3 报告第二个 token 的接受率在 85%–90% 之间，解码速度（TPS）提到约 1.8 倍——代入公式，1 + α = 1.85–1.90，扣掉 MTP 模块自己的开销，1.8 倍很合理。

`05` 的自推测实验：⟨MTP_SPEC⟩

实现上有一个小细节：MTP 模块里的 Transformer block 也要看历史，所以它有自己的一份 KV cache；它在位置 i 需要"主模型在 i 的表示"和"第 i+1 个 token"，所以总比主模型慢一步，只处理已经确定的位置，**不需要回滚**。

### 7.4 谁在用 MTP

按 GOAL.md 2.1 的规则 A 核对（详见"采用方与来源"）：DeepSeek（V3、V4 的 `num_nextn_predict_layers: 1`）、千问（Qwen3-Next 模型卡写明 MTP；Qwen3.5 全系列 config 里 `mtp_num_hidden_layers: 1`，连 0.8B 都有）、智谱 GLM（GLM-4.5 技术报告"加一层 MoE 作为 MTP 层，用于推理时的推测解码"；GLM-5 config 同样有 1 层）、MiniMax（M2 的 config：`use_mtp: true`、`num_mtp_modules: 3`）、小米 MiMo（MiMo-7B 的 `num_nextn_predict_layers: 1`；MiMo-V2-Flash 开源了 3 层 MTP 权重，模型卡称"输出速度提到 3 倍"）、NVIDIA（Nemotron 3 Super 的 `num_nextn_predict_layers: 1`）。远超 3 家，满足共识条件，进正文。反例也有：Kimi K2 的 config 里 `num_nextn_predict_layers: 0`（沿用 DeepSeek-V3 的结构但没有带 MTP 层）。推测解码本身则是推理引擎的行业标准：vLLM 和 SGLang 都内置了草稿模型、n-gram、MTP、EAGLE 等多种草稿方式。

## 8. 小结

- **前提**：decode 受带宽限制，目标模型一次验证 k+1 个位置的时间和生成 1 个差不多。
- **算法**：草稿猜 k 个 → 目标一次前向 → 从左往右接受、遇错纠正、全对奖励 → 回滚 KV cache。每轮 1 到 k+1 个 token。
- **质量不变**：贪心时逐字相同；采样时按 `min(1, p/q)` 接受、被拒从 `max(0, p − q)` 重抽，输出恰好服从目标分布。一次接受的概率 `α = Σ min(p, q) = 1 − TV(p, q)`。
- **能快多少**：`(1 − α^{k+1}) / ((1 − α)(1 + k·c))`。α 高、c 小才快；大批量高并发时收益变小。
- **草稿**：同分词器的小模型、提示词查找、模型自带的草稿头。
- **MTP**：顺序的 MTP 模块（两个 RMSNorm + 投影 + 一个 block，共享 embedding 和输出头），训练时多一个 λ·L_MTP 让信号更密，推理时丢掉或当草稿（DeepSeek-V3：接受率 85%–90%，约 1.8 倍）。

---

## 从极简到生产级

生产级实现在 `zero/arch/speculative.py` 和 `zero/arch/mtp.py`（第五部分的实验模块，**不用于主线模型的训练**）。

| 极简版（`code/`） | 生产级（`zero/arch/`） | 多做了什么、为什么 |
|---|---|---|
| `02` 的 `speculative_greedy`、`03` 的 `speculative_sample`，第 10 章的 `TinyLM` + 拼接式 KV cache | `speculative.py` 的 `speculative_generate(target, draft, prompt_ids, max_new_tokens, k, temperature, top_p, seed, eos_id)` → `SpeculativeResult` | 用 `zero.model.Transformer` 和预分配的 `zero.kv_cache.KVCache`：回滚只需 `rollback(cache, n)` 把有效长度退回去（`KVCache.update` 按 `start_pos` 覆盖写入、只返回 `[0, start_pos + T)`）；贪心与采样同一套代码；`eos_id` 提前停止；返回接受数、被比较数、每轮接受数，`acceptance_rate`、`tokens_per_round` 直接可读；检查两个模型词表一致 |
| `03` 的 `accept_or_resample`、`speculative_step` | `verify(p_logits, drafts, q_probs, temperature, top_p, generator)` | 支持 **top-p**：`warp_probs` 对 p、q 做同样的温度和 top-p 处理（与 `zero.generate.sample_next` 同一个分布）；草稿分布可以是 None（确定性草稿，按 one-hot 处理，接受概率就是 p(x)）；残差数值上为 0 时的兜底 |
| 无 | `prompt_lookup_draft(seq, k, max_ngram)`；`speculative_generate(..., draft=None)` | 零成本的 n-gram 草稿（vLLM 的 `ngram` 方法同类） |
| `04` 的公式 | `expected_tokens_per_round(α, k)`、`expected_speedup(α, k, c)` | 同一个公式，给评估脚本用 |
| `05` 的 `MTPHead` | `mtp.py` 的 `MTPModule`（`enorm`、`hnorm`、`eh_proj`、`block`、`norm`） | 命名与 vLLM / SGLang 的 DeepSeek MTP 实现一致（拼接顺序 `[enorm(emb); hnorm(h)]`，送进 MTP 的是主模型最后一个 RMSNorm 之后的表示）；block 直接用主线的 `zero.model.Block`（GQA + QK-Norm + RoPE + SwiGLU） |
| `05` 的 `train_mtp` 里那一行损失 | `MTPTransformer(config, n_mtp)`（主模型原样是 `zero.model.Transformer`）+ `mtp_loss(model, tokens, targets, lam)` | 支持 D 个顺序模块（深度 k 在长度 T−k 上计算）、`ignore_index`（SFT 的 loss mask 也能用）；返回总损失、主损失和各深度损失，方便记日志 |
| `05` 的 `mtp_self_speculative`（贪心） | `mtp_speculative_generate(model, prompt_ids, max_new_tokens, temperature, top_p, seed, eos_id)` | 贪心与采样；MTP block 有自己的预分配 KV cache，只处理已确定的位置，不需要回滚；复用 `verify` |

**对拍**（`uv run pytest tests/test_arch_speculative.py tests/test_arch_mtp.py`，本机 ⟨N_TESTS⟩ 项全部通过）：

- `test_arch_speculative.py`：k = 1、3、6 时贪心推测解码与 `zero.generate.generate(target, ..., temperature=0)` **逐字相同**；提示词查找草稿同样逐字相同；**草稿 = 目标**时（贪心和采样）接受率 100%、每轮正好 k 个草稿 + 1 个奖励；**缓存回滚**：先把 4 个"错误草稿"写进缓存、回滚、再写正确的 token，logits 与一次性全量前向在 1e-5 内一致；单个位置的 `verify` 抽 4 万次，结果分布与 p 的 TV 距离 < 0.01、接受率与 Σ min(p, q) 相差 < 0.01；固定种子的采样可复现；top-p 规则与 `sample_next` 一致；公式的边界情况。
- `test_arch_mtp.py`：形状（D = 2 时两层 MTP 的 logits 长度分别是 T−1、T−2），且主模型 logits 与 `Transformer.forward` 完全一致；**深度 1 的损失手算**：MTP 损失 = 对 `tokens[i+2]` 的交叉熵，总损失 = 主损失 + λ · MTP 损失；**因果性**：改动位置 7 以后的 token 不影响位置 0–5 的 MTP 输出，改动位置 7 会影响位置 6（它用了 Emb(t₇)）；**梯度**流到 MTP 模块、共享的 embedding / 输出头和主模型的层；几步训练 loss 下降；MTP 自推测解码（贪心）与主模型贪心解码逐字相同。

**真正上线服务**不会用这样的 Python 循环。行业做法：

- **vLLM**：`--speculative-config` 支持草稿模型（`draft_model`）、`ngram`、`mtp`、`eagle` 等方法（以及 suffix decoding、MLP speculator 等），`num_speculative_tokens` 就是本章的 k；它的拒绝采样器有专门的"收敛到目标分布"测试和"贪心逐字相同"的端到端测试。文档示例里 `method: "mtp"` 用的正是小米 MiMo-7B。
- **SGLang**：`speculative_algorithm` 可选 `EAGLE` / `EAGLE3`、`STANDALONE`（独立的草稿模型）、`NGRAM` 等，DeepSeek 等模型的 MTP 层用别名 `NEXTN` 走 EAGLE 这条路径（`python/sglang/srt/speculative/spec_info.py`）；另有训练 EAGLE 草稿头的 SpecForge 工具。
- 服务端的难点是**批量**：一批里每条序列接受的个数不同，KV cache 的回滚、下一轮的输入长度都参差不齐；还常用"树状验证"（一次验证多条候选路径）进一步提高每轮产出。这些都在推理引擎的调度器里完成，本章的 batch = 1 实现不涉及。

`zero/arch/speculative.py` 和 `mtp.py` 的 GPU 路径**尚未在 GPU 上验证**：本章只在 CPU 上验证了算法的正确性，没有测过 GPU 上的速度。

**主线模型怎么用这一章**。主线模型**不带 MTP**（GOAL.md 3.3：主线不冒架构风险；在 0.6B 这个规模上，MTP 对主模型质量的影响也没有公开的可靠证据）。第二步发布时可以这样给它加速，前提是先在 GPU 上实测：

1. **另训一个小草稿**：用主线的分词器和同一份数据，训练一个约 5000 万到 1 亿参数的 `zero` 模型（`configs/` 里加一档配置即可，代码不用改），再用主线模型的输出做一轮蒸馏（第 17 章）提高 α；
2. **提示词查找**：工具调用里大量复制函数名、参数名和 JSON 键，零成本，值得先试；
3. **事后加 MTP**：冻结主线模型，只训练一个 `zero.arch.mtp.MTPModule`（DeepSeek 的消融显示联合训练能改进主模型，但只拿它当草稿的话，事后训练也可以——这一点本课没有验证，**待实验**）。

哪种方式、k 取多少、在 llama.cpp / vLLM 上各快多少，都要等第二步在真实硬件上测了再写。

---

## 前沿观察

> **不算共识、只在这里提一句的技术**
>
> - **Medusa**（Cai 等 2024）：在目标模型最后一层上并联几个独立的头，第 j 个头直接预测往后第 j 个 token（不看中间 token，类似 Gloeckle 等的并行头），再用"树状注意力"一次验证多条候选。
> - **EAGLE / EAGLE-2 / EAGLE-3**（Li 等 2024–2025）：一个很小的自回归草稿网络，输入是目标模型的隐藏特征（EAGLE-3 用多层特征），配合动态草稿树，接受率很高。它是目前推理引擎里最常用的草稿方式之一（vLLM 文档把它和 MTP 并列为收益最高的方法），Hugging Face 上也有大量 EAGLE-3 草稿头（NVIDIA 给 gpt-oss-120b、社区给 Llama、Qwen3、Kimi K2.6、MiniMax 等训练的）。但这些几乎都是推理厂商或社区**事后**训练的，头部模型家族在主力版本的技术报告或模型卡里**自己**发布 EAGLE 草稿的，我没能核实到 3 家，所以放在这里。它和 MTP 的思路很像——DeepSeek-V3 的原文就说"保持因果链"这一点和 EAGLE 相似，区别在于 MTP 的首要目的是改进训练。
> - **树状验证**（SpecInfer、Medusa、EAGLE-2 等）：一次验证一棵候选树而不是一条链，每轮期望产出更高，但注意力掩码和缓存管理更复杂。
> - **放宽的接受规则**（lenience、typical acceptance 等）：换更高的接受率，代价是输出分布不再严格等于目标分布。

## 采用方与来源

| 技术 | 采用方（主力版本） | 来源 |
|---|---|---|
| MTP（顺序的多 token 预测模块，训练目标 + 推理草稿） | **DeepSeek**：V3（D = 1，λ = 0.3 → 0.1，第二个 token 接受率 85%–90%，TPS 约 1.8 倍）、V4-Pro（`num_nextn_predict_layers: 1`）；**千问**：Qwen3-Next-80B-A3B（模型卡："MTP 提升预训练效果并加速推理"）、Qwen3.5 全系列（0.8B 的 `mtp_num_hidden_layers: 1`）；**智谱**：GLM-4.5 / 4.5-Air（1 层 MoE 作为 MTP 层，λ = 0.3 → 0.1）、GLM-5（`num_nextn_predict_layers: 1`）；**MiniMax**：M2（`use_mtp: true`、`num_mtp_modules: 3`）；**小米**：MiMo-7B（`num_nextn_predict_layers: 1`）、MiMo-V2-Flash（开源 3 层 MTP 权重）；**NVIDIA**：Nemotron 3 Super 120B-A12B（`num_nextn_predict_layers: 1`） | [DeepSeek-V3 技术报告 arXiv:2412.19437](https://arxiv.org/abs/2412.19437)（第 2.2、4.5.1、5.4.3 节）；config.json：[DeepSeek-V3](https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/config.json)、[DeepSeek-V4-Pro](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/config.json)、[Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B/blob/main/config.json)、[GLM-4.5](https://huggingface.co/zai-org/GLM-4.5/blob/main/config.json)、[GLM-5](https://huggingface.co/zai-org/GLM-5/blob/main/config.json)、[MiniMax-M2](https://huggingface.co/MiniMaxAI/MiniMax-M2/blob/main/config.json)、[MiMo-7B-Base](https://huggingface.co/XiaomiMiMo/MiMo-7B-Base/blob/main/config.json)、[Nemotron-3-Super-120B-A12B](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16/blob/main/config.json)；模型卡：[Qwen3-Next-80B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct)、[MiMo-V2-Flash](https://huggingface.co/XiaomiMiMo/MiMo-V2-Flash)；[GLM-4.5 技术报告 arXiv:2508.06471](https://arxiv.org/abs/2508.06471)（第 2.1、2.4 节）；[MiMo-7B 技术报告 arXiv:2505.07608](https://arxiv.org/abs/2505.07608) |
| 推测解码（草稿模型 / n-gram / MTP 草稿 + 拒绝采样验证） | 行业标准（GOAL.md 2.1 的 B 类）：**vLLM**、**SGLang** 内置；模型方发布专用草稿：**Google Gemma 4**（E2B、E4B、12B、26B-A4B、31B 各有 `*-it-assistant` MTP 草稿模型，模型卡称"最多约 3 倍、质量完全相同"）；**DeepSeek-V3**、**GLM-4.5**、**MiMo** 在技术报告/模型卡里说明用 MTP 做推测解码 | Leviathan 等 2023；Chen 等 2023；[vLLM 推测解码文档（源文件）](https://github.com/vllm-project/vllm/blob/main/docs/features/speculative_decoding/README.md)与 [MTP 文档](https://github.com/vllm-project/vllm/blob/main/docs/features/speculative_decoding/mtp.md)；[SGLang](https://github.com/sgl-project/sglang)；[gemma-4-31B-it-assistant 模型卡](https://huggingface.co/google/gemma-4-31B-it-assistant) |

说明：

- 所有 config 与模型卡均在 2026-09 通过 Hugging Face 读取。Qwen3-Next 的 config.json 里没有 MTP 相关字段（MTP 权重的存放方式与 Qwen3.5 不同），采用的是模型卡的明文说法。
- Kimi K2 的 config 是 `num_nextn_predict_layers: 0`（GLM-4.5 技术报告表 1 也列为 0 层 MTP），不计入采用方。
- MiMo-7B 技术报告里 MTP 的具体用法（预训练几层、推理用几层、接受率）本章没有逐条核对，只引用了 config 与 vLLM 文档的示例，**待核实**。
- Gemma 4 的 MTP 草稿是独立的小模型（`Gemma4AssistantForCausalLM`，4 层，与目标共享 KV cache），和 DeepSeek 式"主模型内部的 MTP 模块"不完全一样；这里把它计入"推测解码 + 模型方发布草稿"，不计入"DeepSeek 式 MTP 模块"。Gemma 4 技术报告（模型卡链接为 arXiv:2607.02770）本章未阅读，**待核实**其中关于 MTP 训练方式的描述。

---

## 引导问题

带着这些问题去问 Claude Code，直到你能用自己的话讲清楚：

1. 推测解码"验证 k+1 个位置和生成 1 个一样快"的前提在什么情况下不成立？试着让 Claude Code 帮你估算：主线模型在 H100 上、batch = 1 和 batch = 64 时，一次 decode 的算术强度各是多少？k = 4 时验证一轮的算术强度又是多少？（提示：第 21 章的 `02_prefill_decode.py`。）
2. 采样时的推导里，如果草稿抽到的 token 有 `q(x) = 0`（草稿认为不可能），会发生什么？如果目标认为不可能（`p(x) = 0`）呢？温度 → 0 时，采样版的规则会退化成贪心版吗？
3. 本章说"一次接受的概率 α = 1 − TV(p, q)"。为什么是 TV 距离，而不是 KL 散度？如果你要**训练**一个草稿让 α 尽量高，损失函数应该选什么？（提示：搜索 DistillSpec。）
4. DeepSeek-V3 的 MTP 模块预测 `t_{i+2}` 时能看到 `t_{i+1}` 的 embedding；Gloeckle 等的并行头和 Medusa 看不到。这对训练时的"信号加密"和推理时的接受率分别意味着什么？
5. `05_mtp.py` 里 MTP 模块的"下下个字符准确率"和主模型的"下一个字符准确率"差不多高，这是不是说明预测两步之后和预测一步之后一样容易？
6. 为什么 MiniMax 的博客把"和推测解码的配合"列为线性注意力 / 混合架构的未解决问题之一？（提示：想想回滚。线性注意力的状态是累加出来的，被拒的草稿已经加进去了，怎么撤回？）

## 动手任务

每个任务都要真的运行代码、看到结果。

**任务 1（基础）**：在 `04_speedup_formula.py` 里把 c 设成 `01_models_and_cost.py` 在你机器上测出来的值，再用 `02` 测出的 α，找出最优 k；然后在 `02` 里只跑这个 k 和它两边的 k，看实测的最快 k 和公式预测的是否一致。如果不一致，找出公式没考虑到的开销。

**任务 2（核心）**：给 `02_greedy_speculative.py` 加一个提示词查找草稿（在 `seq` 里找与末尾 3 个字符相同的最近片段，取后面 k 个字符），与 1 层草稿模型比较接受率和目标前向次数。再换一个"会重复"的提示词（比如把一段台词复制两遍，让模型续写第三遍），两者的差距怎么变？对照 `zero/arch/speculative.py` 的 `prompt_lookup_draft`。

**任务 3（挑战）**：用蒸馏提高草稿的接受率：在 `01_models_and_cost.py` 里新增一个训练函数，损失改成草稿分布与目标分布之间的 KL 散度（目标模型冻结，第 17 章的 logits 蒸馏），训练同样的步数，比较蒸馏前后温度 1.0 时的实测接受率（`03` 第 3 部分）和贪心时的接受率（`02`）。再试试把损失换成 TV 距离，哪个对 α 更有效？

---

## 想深入：CS336

本章对应斯坦福 CS336（Spring 2026）<https://cs336.stanford.edu/>：

- **第 10 讲：推理**。这一讲的 speculative sampling 一节（讲义源码 `lecture_10.py` 的 `speculative_sampling()`，见 <https://github.com/stanford-cs336/lectures>）从"prefill 并行、decode 受带宽限制，所以检查比生成快"讲起，给出了和本章第 4 节相同的两词表证明、70B 配 8B / 8B 配 1B 的草稿搭配、用蒸馏让草稿更像目标，以及 Medusa、EAGLE 两种改进草稿的方法。注意 CS336 讲义里**草稿记作 p、目标记作 q**，和本章（以及 Leviathan 等原论文）正好相反。
- **CS336 未深入**：MTP 作为训练目标（DeepSeek-V3 的顺序 MTP 模块、损失权重、消融）以及推理引擎里的实现细节，这一讲没有展开，见本章参考文献。

---

## 本章参考文献

- Leviathan, Kalman, Matias. *Fast Inference from Transformers via Speculative Decoding*，ICML 2023：<https://arxiv.org/abs/2211.17192>
- Chen, Borgeaud, Irving, Lespiau, Sifre, Jumper. *Accelerating Large Language Model Decoding with Speculative Sampling*，2023：<https://arxiv.org/abs/2302.01318>
- Stern, Shazeer, Uszkoreit. *Blockwise Parallel Decoding for Deep Autoregressive Models*（贪心的"并行猜、再验证"的前身），NeurIPS 2018：<https://arxiv.org/abs/1811.03115>
- Gloeckle, Idrissi, Rozière, Lopez-Paz, Synnaeve. *Better & Faster Large Language Models via Multi-token Prediction*，ICML 2024：<https://arxiv.org/abs/2404.19737>
- DeepSeek-AI. *DeepSeek-V3 Technical Report*，2024：<https://arxiv.org/abs/2412.19437>
- GLM-4.5 Team. *GLM-4.5: Agentic, Reasoning, and Coding (ARC) Foundation Models*，2025：<https://arxiv.org/abs/2508.06471>
- Xiaomi LLM-Core Team. *MiMo: Unlocking the Reasoning Potential of Language Model – From Pretraining to Posttraining*，2025：<https://arxiv.org/abs/2505.07608>
- Cai et al. *Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads*，2024：<https://arxiv.org/abs/2401.10774>
- Li, Wei, Zhang, Zhang. *EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty*，2024：<https://arxiv.org/abs/2401.15077>；*EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test*，2025：<https://arxiv.org/abs/2503.01840>
- Zhou et al. *DistillSpec: Improving Speculative Decoding via Knowledge Distillation*，2023：<https://arxiv.org/abs/2310.08461>
- Saxena. *Prompt Lookup Decoding*：<https://github.com/apoorvumang/prompt-lookup-decoding>
- vLLM 推测解码文档（源文件）：<https://github.com/vllm-project/vllm/blob/main/docs/features/speculative_decoding/README.md>；vLLM 的 DeepSeek MTP 实现：<https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/deepseek_mtp.py>
- SGLang：<https://github.com/sgl-project/sglang>
- 模型配置与模型卡（2026-09 读取）：见上方"采用方与来源"表中的链接；[Kimi-K2-Instruct config](https://huggingface.co/moonshotai/Kimi-K2-Instruct/blob/main/config.json)（`num_nextn_predict_layers: 0`）
- Google Research. *Looking back at speculative decoding*（CS336 讲义引用的回顾博客）：<https://research.google/blog/looking-back-at-speculative-decoding/>
- CS336 Spring 2026 讲义源码：<https://github.com/stanford-cs336/lectures>；课程主页：<https://cs336.stanford.edu/>

**下一章**：第五部分的架构实验到这里就做完了：更小的 KV cache（GQA、MLA）、更便宜的长上下文（滑动窗口、线性注意力混合）、更省的计算（MoE），以及更快的生成（推测解码、MTP）。最后一章把这些拼图放回真实的模型里：拆解几个写作时最新的开源旗舰，画出架构演化树，把我们的主线模型也放进同一张表。第 26 章，当前最先进开源模型全景。
