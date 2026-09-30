# 第 9 章：现代 Transformer —— 把注意力搭成一个会写字的模型

> **一句话目标**：读完这一章，你能画出现代 Transformer 一次前向的完整张量形状数据流，说清楚 Pre-Norm RMSNorm、RoPE、SwiGLU、QK-Norm、共享 embedding 各自解决什么问题、对应代码里哪一行，并亲手在 CPU 上训练一个能写出"莎士比亚腔"的小模型。

📺 **本章视频**：待发布（本地渲染：`bash chapters/09-modern-transformer/video/build.sh`）
🧪 **本章自检**：学完后在 Claude Code 里输入 `/ch09-transformer`

---

上一章我们从"加权平均"推出了注意力：每个位置用 Q 去和所有位置的 K 比相似度，按相似度把 V 加权平均起来，因果 mask 保证只看过去，多个头各看各的。但一层注意力还不是一个语言模型：字节（或 token）怎么变成向量？位置信息从哪来？每个位置拿到"别人的信息"之后怎么加工？几十层叠起来怎么不训炸？最后怎么变回"下一个词的概率"？

这一章要解决的问题就是：**把注意力装进一个完整的、能训练的语言模型**。我们会逐个装上今天几乎所有开源大模型都在用的零件，写出一个 200 行左右的单文件模型，在 CPU 上训练十几分钟，看它从乱码学会写剧本；最后把它的权重原封不动地搬进主线模型的生产级代码 `zero/`，证明两层代码算出来的是同一个东西——而 `zero` 又和千问 Qwen3 的官方实现逐项对得上。

## 1. 全景：查表 → N 个 Block → 打分

先看整体。一个 Decoder-only Transformer 做三件事：

1. **查表（embedding）**：每个 token 的编号去一张 `V × d` 的表里取出一行，变成 `d` 维向量。本章的极简模型按字节切分，`V = 256`，`d = 128`。
2. **N 个 Block**：每个 Block 里有两个子层。
   - **注意力（Attention）**：在**位置之间**交换信息——第 t 个位置从前面的位置那里"读"东西。
   - **前馈网络（FFN）**：在**每个位置内部**加工——同一套参数对每个位置单独做一次非线性变换（第 3 章的 MLP）。
3. **打分（LM head）**：最后一个位置的向量乘上一个 `d × V` 的矩阵，得到 V 个分数（logits），softmax 之后就是下一个 token 的概率（第 5 章）。

每个子层都包在同一个套路里——**Pre-Norm + 残差**（第 6 章）：

```
x = x + Attention(RMSNorm(x))      # 位置之间交换信息
x = x + FFN(RMSNorm(x))            # 每个位置各自加工
```

对应极简代码 [`code/02_tiny_transformer.py`](code/02_tiny_transformer.py) 里的 `Block.forward`：

```python
def forward(self, x, cos, sin):
    x = x + self.attn(self.attn_norm(x), cos, sin)  # 位置之间交换信息
    x = x + self.ffn(self.ffn_norm(x))              # 每个位置各自加工
    return x
```

一个好用的比喻是**残差流（residual stream）**：`x` 是一条从输入一路流到输出的"主干道"，宽度始终是 `d`。每个子层只是从主干道上读一份拷贝（先归一化），算出一点"修改意见"，再加回主干道。第 6 章讲过，这样梯度可以沿着加法直通前面的层；Pre-Norm 把归一化放在子层入口而不是主干道上，主干道本身保持"干净的恒等映射"，几十层也训得动。原始 Transformer（2017）是 Post-Norm（`x = Norm(x + 子层(x))`），GPT-2 起改成了 Pre-Norm。

下面按数据流动的顺序，把每个零件讲清楚。

## 2. 位置：注意力分不清顺序

### 2.1 问题

第 8 章的注意力只看"内容像不像"：分数是 `q·k`，和 k 在第几个位置无关。后果是什么？做个实验（[`code/01_position.py`](code/01_position.py)）：把"狗咬人了"和"人咬狗了"送进同一个因果注意力，看最后一个字"了"的输出：

```bash
uv run python chapters/09-modern-transformer/code/01_position.py
```

| | "狗咬人了" vs "人咬狗了"，"了"的输出最大差 |
|---|---:|
| 无位置信息 | 5.96 × 10⁻⁸（浮点误差，等于完全相同） |
| 加 RoPE | 0.133 |

没有位置信息时，注意力把前文当成一个**集合**而不是**序列**：加权平均和相加的顺序无关，谁咬谁对它来说一模一样。语言模型必须知道顺序，所以要把位置写进去。

GPT-2 的做法是再学一张"位置表"（learned absolute position embedding）：第 m 个位置加上第 m 行向量。问题是它只认训练时见过的位置（GPT-2 是 1024 个），而且位置信息混在内容向量里，到了深层就被稀释了。今天的主流做法是 **RoPE**。

### 2.2 RoPE：把 q、k 按位置旋转

**旋转位置编码（Rotary Position Embedding, RoPE）**（Su et al., 2021）的想法很几何：把 q、k 的维度两两配对，每一对看成平面上的一个二维向量；在位置 m，把这个二维向量**旋转 m·ω 的角度**。

一对维度 `(a, b)` 旋转角度 θ：

```
(a, b)  →  (a·cosθ − b·sinθ,  b·cosθ + a·sinθ)
```

不同的维度对转速不同：第 i 对的转速是 `ω_i = θ_base^(−2i/d)`，`θ_base` 默认 10000。前面的维度对转得快（每个位置转 1 弧度），后面的转得很慢（转一圈要几千个位置）——像钟表的秒针、分针、时针。

为什么这样就能编码位置？关键在点积。q 在位置 m 转了 `mω`，k 在位置 n 转了 `nω`，两个向量的点积只取决于它们**夹角的变化**，也就是 `(m − n)ω`：

```
RoPE(q, m) · RoPE(k, n) = 只和 (m − n) 有关的函数
```

绝对位置被"消掉"了，只剩**相对位置**。这正是语言需要的："前一个词"在句首和句中是同一种关系。脚本里直接验证（同一对 q、k 放到不同位置）：

| m | n | m − n | 旋转后的点积 q_m·k_n |
|---:|---:|---:|---:|
| 3 | 1 | +2 | 2.4426 |
| 10 | 8 | +2 | 2.4426 |
| 50 | 48 | +2 | 2.4426 |
| 5 | 1 | +4 | 0.8593 |
| 40 | 36 | +4 | 0.8593 |
| 1 | 3 | −2 | 0.7563 |

相隔 2 的三组点积完全一样，相隔 4 的两组也一样；−2 和 +2 不同，说明它分得清前后。另外旋转不改变长度（脚本里 `|q| = 2.2184`，转到位置 37 之后还是 2.2184），所以 RoPE 只改"方向"不改"大小"，不会干扰注意力分数的尺度。

代码只有两个函数：

```python
def rope_cos_sin(head_dim, seq_len, theta=10000.0):
    inv_freq = theta ** (-torch.arange(0, head_dim, 2).float() / head_dim)  # ω_i = θ^(-2i/d)
    angles = torch.outer(torch.arange(seq_len).float(), inv_freq)           # (T, d/2)：m·ω_i
    angles = torch.cat([angles, angles], dim=-1)                            # (T, d)
    return angles.cos(), angles.sin()

def apply_rope(x, cos, sin):
    """维度 i 与 i + d/2 配成一对，做二维旋转：(a, b) → (a·cos − b·sin, b·cos + a·sin)。"""
    a, b = x.chunk(2, dim=-1)
    return x * cos + torch.cat([-b, a], dim=-1) * sin
```

注意配对方式：这里把第 i 维和第 i + d/2 维配成一对（"前后两半"），而 RoFormer 论文写的是相邻两维配对。两种写法数学上等价（只是维度换了个排列），但**权重不通用**：Hugging Face 的 Llama、Qwen 实现都用"前后两半"，我们照它来，第 7 节的对拍才能对上。

RoPE 只加在 q、k 上，不加在 v 上（位置只影响"看谁"，不影响"读到什么"），也没有任何可学参数。第 15 章做长上下文时，调的就是 `θ_base` 和各维度的转速（YaRN）。

## 3. FFN：SwiGLU

注意力负责"搬运"信息，搬来之后的加工在 FFN 里。GPT-2 的 FFN 是第 3 章的两层 MLP：`W₂ · GELU(W₁ x)`，中间宽度 `4d`。

现代模型换成了**门控线性单元（Gated Linear Unit）**的一个变体 **SwiGLU**（Shazeer, 2020）：

```
FFN(x) = W_down · ( SiLU(W_gate · x) ⊙ (W_up · x) )
SiLU(z) = z · sigmoid(z)
```

两条并行的路：`W_up x` 是"内容"，`SiLU(W_gate x)` 是"门"，逐元素相乘（⊙）——门接近 0 的维度被关掉，门大的维度被放大。第 3 章说 SiLU 是平滑版的 ReLU；门控的意思是：**关不关，由输入自己决定**，而且决定得很平滑，比单纯的 ReLU 表达力更强。Shazeer 的实验里，同样参数量下 GLU 系列的困惑度比 ReLU/GELU 的 MLP 更低，LLaMA 采用后成了事实标准。

```python
def forward(self, x):
    return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))
```

多了一个矩阵，参数不就变多了？所以中间宽度从 `4d` 缩到 `8/3·d`：两个 `d × 4d` 矩阵是 `8d²`，三个 `d × (8/3)d` 矩阵也是 `8d²`。[`code/03_shapes.py`](code/03_shapes.py) 在 `d = 1280` 上算出两者都是 13.11M。LLaMA 论文原话是"用 2/3·4d 而不是 PaLM 的 4d"。这个比例是**起点而不是定律**：本章极简模型取 352（`8/3 × 128 ≈ 341`，取 32 的倍数），Qwen3-0.6B 取 3072（3d），Llama 3.2 1B 取 8192（4d），主线模型取 3584（2.8d），都是各家按硬件对齐和实验调出来的。

## 4. QK-Norm：给注意力分数上保险

注意力分数是 `q·k / √d`。训练中 q、k 的长度会慢慢变大，分数跟着变大，softmax 就会"一边倒"：几乎全部权重压在一个位置上，梯度变得又尖又不稳定，大模型训练中的 loss 尖峰常常和这件事有关（OLMo 2 报告专门分析过）。

**QK-Norm**（Dehghani et al., 2023，最早用于 220 亿参数的视觉 Transformer）的办法很直接：做点积之前，对**每个头**的 q、k 各做一次 RMSNorm。[`code/05_qk_norm.py`](code/05_qk_norm.py) 把同一组随机 q、k 整体放大 s 倍，模拟训练中范数变大：

```bash
uv run python chapters/09-modern-transformer/code/05_qk_norm.py
```

| 放大倍数 s | 不加 QK-Norm：最大分数 | 平均最大权重 | 熵 | 加 QK-Norm：最大分数 | 平均最大权重 | 熵 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 4.9 | 0.222 | 2.42 | 2.8 | 0.213 | 2.43 |
| 2 | 19.5 | 0.667 | 0.99 | 2.8 | 0.213 | 2.43 |
| 4 | 77.8 | 0.923 | 0.21 | 2.8 | 0.213 | 2.43 |
| 8 | 311.3 | 0.996 | 0.02 | 2.8 | 0.213 | 2.43 |
| 16 | 1245.0 | 1.000 | 0.00 | 2.8 | 0.213 | 2.43 |

（16 个位置均匀分布的熵是 ln 16 = 2.77。）不加 QK-Norm，放大 4 倍时注意力已经 92% 压在一个位置上；加了之后分数的尺度与 q、k 的长度无关。真实模型里 RMSNorm 带可学的缩放权重，分数仍然可以变大，但变大要靠"权重慢慢学"，而不是被激活值的漂移推着走。

在代码里就是两行，放在拆头之后、RoPE 之前：

```python
q = self.q_norm(self.wq(x).view(B, T, self.h, self.hd)).transpose(1, 2)  # (B, H, T, hd)
k = self.k_norm(self.wk(x).view(B, T, self.h, self.hd)).transpose(1, 2)
```

## 5. 输入与输出：共享 embedding

模型两头各有一个 `V × d` 的矩阵：输入端的 embedding 表（token → 向量），输出端的 LM head（向量 → 每个 token 的分数）。**共享 embedding（tied embeddings）**（Press & Wolf, 2017）让它们用同一个矩阵：

```python
self.lm_head.weight = self.tok_emb.weight  # 共享 embedding：同一个矩阵用两次
```

直觉：输入时"猫"这个词对应一个向量；输出时，一个位置的向量和"猫"的向量越像，就越应该预测"猫"。一张表两用，说得通，也省参数。省多少？取决于词表占模型的比重（`code/03_shapes.py` 的输出，Qwen3-0.6B 按官方 `config.json` 的超参计算）：

| 模型 | 词表矩阵 V×d | 共享时总参数 | 词表占比 | 若不共享 | 实际 |
|---|---:|---:|---:|---:|---|
| 本章极简模型（V=256） | 0.03M | 0.84M | 3.9% | 0.87M | 共享 |
| `configs/tiny`（V=2048） | 0.26M | 1.05M | 25.0% | 1.31M | 不共享 |
| Qwen3-0.6B（V=151,936） | 155.58M | 596.05M | 26.1% | 751.63M | 共享 |
| `configs/main` 主线（V=65,536） | 83.89M | 689.52M | 12.2% | 773.40M | 共享 |

小模型里词表是一大块：Qwen3-0.6B 要是不共享，会多出 1.56 亿参数、总量涨 26%。这就是为什么**小模型普遍共享、大模型普遍不共享**：Qwen3 技术报告的表 1 里 0.6B、1.7B、4B 共享，8B 及以上不共享；DeepSeek-V3、gpt-oss 这样的大模型都不共享（见文末来源）。主线模型的词表暂定 65,536，共享后 embedding 占 83.9M / 689.5M。`configs/tiny` 是个例外：它的注释写明，在这个 1M 参数的冒烟规模上实测不共享时 loss 更快走出"只会猜高频词"的平台期，所以演示配置关掉了共享——这类取舍都写在配置里，不写死在代码里。

## 6. 完整张量形状数据流

把零件串起来，看一次前向里每一步的形状。[`code/03_shapes.py`](code/03_shapes.py) 在极简模型上挂钩子（forward hook）打印真实形状（B = 2 条序列，T = 16 个字节）：

| 模块 | 输入形状 | 输出形状 | 说明 |
|---|---|---|---|
| `tok_emb` | (2, 16) | (2, 16, 128) | 整数 → 向量 |
| `layers.0.attn_norm` | (2, 16, 128) | (2, 16, 128) | RMSNorm 不改形状 |
| `layers.0.attn.wq` | (2, 16, 128) | (2, 16, 128) | 之后 view 成 4 个头 |
| `layers.0.attn.q_norm` | (2, 16, 4, 32) | (2, 16, 4, 32) | QK-Norm 在每个头的 32 维上做 |
| `layers.0.attn` | (2, 16, 128) | (2, 16, 128) | 内部分数矩阵 (2, 4, 16, 16) |
| `layers.0.ffn.w_gate` / `w_up` | (2, 16, 128) | (2, 16, 352) | SwiGLU 先变宽 |
| `layers.0.ffn.w_down` | (2, 16, 352) | (2, 16, 128) | 再变回 d |
| `layers.0` … `layers.3` | (2, 16, 128) | (2, 16, 128) | 每个 Block 进出形状相同，所以能叠 |
| `norm` | (2, 16, 128) | (2, 16, 128) | 最后一次 RMSNorm |
| `lm_head` | (2, 16, 128) | (2, 16, 256) | 每个位置对下一个字节的 256 个打分 |

同一张图换成主线模型的尺寸（`configs/main/pretrain.toml`，每张卡一个 micro batch：B = 8, T = 4096）：

```
token id                   (8, 4096)
embedding 查表              (8, 4096, 1280)
× 28 层  RMSNorm            (8, 4096, 1280)
  q = x·Wq → 拆头           (8, 16, 4096, 128)
  k, v = x·Wk, x·Wv         (8, 8, 4096, 128)   ← GQA：8 个 K/V 头（第 10 章）
  QK-Norm + RoPE            形状不变
  注意力分数 QKᵀ            (8, 16, 4096, 4096)
  加权求和 → 合并头 → Wo     (8, 4096, 2048) → (8, 4096, 1280)
  残差相加                   (8, 4096, 1280)
  SwiGLU: gate、up          (8, 4096, 3584)
  down → 残差相加            (8, 4096, 1280)
最后 RMSNorm                 (8, 4096, 1280)
lm_head → logits             (8, 4096, 65536)
```

两个值得记住的地方：一是**残差流的形状从头到尾不变**，所有子层都是"读一份、算一点、加回去"；二是**两头最贵**：注意力分数是 T × T 的，最后的 logits 有 8 × 4096 × 65536 ≈ 2.15G 个数，按 float32 存要 8.0 GiB——第 14 章的 FlashAttention 和分块交叉熵就是为这两处准备的。另外主线模型的 q 合起来是 16 × 128 = 2048 维，比 d = 1280 还宽，这是 Qwen3 允许的（`head_dim` 不必等于 `d / n_heads`），所以 `Wo` 是 2048 → 1280。

## 7. 训练：从乱码到剧本

一切就位，训练。数据是 `assets/tiny_corpus/shakespeare.txt`（约 1.1MB，公有领域），按字节切分，前 90% 训练、后 10% 验证。第 7 章讲过，按字节算的交叉熵除以 ln 2 就是 **bits-per-byte**；词表是 256，瞎猜就是 8 bit/字节。训练循环就是第 1 章的五行，加上第 6 章的 AdamW、warmup + 余弦衰减、梯度裁剪：

```python
x, y = get_batch(train_data, cfg, batch_size, g)          # y = x 右移一位
loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten())
opt.zero_grad()
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
opt.step()
```

```bash
uv run python chapters/09-modern-transformer/code/02_tiny_transformer.py
```

模型 836,992 个参数（4 层、d = 128、4 个头、FFN 352），batch 32 × 128 字节，学习率峰值 3e-3，1200 步。验证集上的 bits-per-byte：

| 步数 | 0 | 200 | 400 | 600 | 800 | 1000 | 1200 |
|---|---:|---:|---:|---:|---:|---:|---:|
| val bit/字节 | 8.052 | 2.781 | 2.518 | 2.391 | 2.289 | 2.243 | 2.221 |

训练前（随机初始化，温度 0.8 采样）从 `ROMEO:\n` 往后写 200 个字节，是一堆非法 UTF-8 和控制字符，终端里显示为乱码。训练后写 400 个字节（节选）：

```
ROMEO:
And yield for what be a child.

JULIET:
No determina, hear my master warners:
Whose three live well said an your honour:
I having alonged against thee, good both
Your words of all now, and thou thy death,
She was not had succetting with worm dead!
And say we did sound the friends of England,
And doth to well here dew thy justice king.

Second Servingman:
Then I be the heart of your blood than name
```

它学会了剧本格式（角色名大写 + 冒号 + 换行）、大部分英文单词的拼写、`thou`/`thy` 这种古英语用词和大致的韵律；意思还不通，偶尔造词（`succetting`、`warners`）。这就是 84 万参数、看了约 490 万个字节（1200 步 × 32 × 128，约 4.9 遍训练集）能做到的。耗时：在本机（4 核、同时有其他任务在跑）单线程墙钟 14.7 分钟、CPU 时间 10.8 分钟；赶时间可以加 `--steps 400`（学习率按 400 步重新安排，效果会差一些，数字以你自己的运行为准）。

生成时注意 `generate` 函数里的这行注释：

```python
for _ in range(n_new):  # 每生成一个字节，都把整段重新算一遍 —— 慢！第 10 章用 KV cache 解决
```

## 8. 小结

- **Transformer = 查表 → N 个 Block → 打分**；每个 Block = 注意力（位置之间交换信息）+ FFN（每个位置各自加工），都包在 Pre-Norm RMSNorm + 残差里，残差流的形状 (B, T, d) 从头到尾不变。
- **RoPE**：把 q、k 的维度两两配对按位置旋转，点积只剩相对位置 m − n。
- **SwiGLU**：`W_down(SiLU(W_gate x) ⊙ W_up x)`，门控 FFN；中间宽度约 8/3·d 让参数量与 4d 的 MLP 持平。
- **QK-Norm**：点积前对每个头的 q、k 做 RMSNorm，注意力分数不再随激活值的漂移变大。
- **共享 embedding**：输入表和输出头用同一个矩阵；词表占比大的小模型收益明显。

### 历史对照：GPT-2（2019）与今天

| 零件 | GPT-2 | 今天的主流（Qwen3 / Llama 3 / 本课主线） |
|---|---|---|
| 归一化 | LayerNorm（减均值、除标准差、带 bias） | RMSNorm（只缩放） |
| 归一化位置 | Pre-Norm（GPT-2 已把 LN 移到子层入口；2017 年原版 Transformer 是 Post-Norm） | Pre-Norm |
| 位置编码 | 学出来的绝对位置表，最多 1024 个位置 | RoPE，无参数，编码相对位置 |
| FFN | `4d` 两层 MLP + GELU | SwiGLU，约 8/3·d ~ 4d |
| 注意力 | 多头注意力 | GQA（第 10 章）+ QK-Norm（Qwen3、Gemma 3、OLMo 2） |
| bias | 线性层和 LayerNorm 都带 bias | 大多去掉（Qwen3 报告写明去掉了 Qwen2 的 QKV bias） |
| 共享 embedding | 共享 | 小模型共享，大模型多数不共享 |

GPT-2 的具体结构可以在 transformers 的 `GPT2Config`（`activation_function="gelu_new"`、`n_inner` 默认 4 × hidden、`tie_word_embeddings=True`）和 `modeling_gpt2.py`（`wpe` 位置表、`ln_1` 在注意力之前）里核对；gpt-oss 模型卡也写明"与 GPT-2 一样使用 Pre-LN"。

---

## 从极简到生产级

主线模型的模型代码在 [`zero/model.py`](../../zero/model.py)，和极简版逐个零件对应：

| 极简版（`code/02_tiny_transformer.py`） | 生产级（`zero/`） | 生产级多做了什么，为什么 |
|---|---|---|
| `RMSNorm` | `zero/model.py` 的 `RMSNorm` | 先转 float32 算归一化再转回原精度（BF16 训练时数值更稳），与 HF 的 `Qwen3RMSNorm` 逐行一致 |
| `rope_cos_sin` + `apply_rope` | `compute_rope_inv_freq`、`RotaryEmbedding`、`apply_rope` | cos/sin 预计算成不进 checkpoint 的 buffer；支持 **YaRN** 缩放（第 15 章长上下文），`reset_buffers` 在换设备后重建 |
| `Attention`（手写 softmax + mask） | `Attention` | **GQA**（`n_kv_heads < n_heads`，第 10 章）；`F.scaled_dot_product_attention`（GPU 上 BF16 自动走 FlashAttention 内核，第 14 章；已在 RTX 3090 上验证，见 [runs/2026-10-01-gpu0-check](../../runs/2026-10-01-gpu0-check/README.md) 第 1 节）；**KV cache** 与分块 prefill 的 mask（第 10 章）；`head_dim` 可以不等于 `d / n_heads` |
| `SwiGLU` | `SwiGLU` | 相同（参数名 `w_gate/w_up/w_down` 故意取成一样） |
| `Block` | `Block` | 多传 `kv_cache`、`start_pos`，其余相同 |
| `TinyTransformer` | `Transformer` | `tie_embeddings` 可配置；`loss()` 在 float32 上算并支持 `ignore_index`（第 16 章 SFT 的 loss mask）；`num_params`、`flops_per_token`（第 12、14 章算 MFU 和成本） |
| `Config` dataclass，写死在文件里 | `zero/config.py` 的 `ModelConfig` + `configs/*.toml` | 超参全在 TOML 里：`configs/tiny`（CPU 冒烟）、`configs/ladder`（阶梯实验）、`configs/main`（主线）；读配置时就校验（例如 `n_heads` 必须能被 `n_kv_heads` 整除） |
| 手写训练循环 | `zero/train/`（`python -m zero.train.pretrain`） | 数据混合、BF16、梯度累积、DDP/FSDP、断点续训、日志、checkpoint（第 14 章） |
| 无 | `zero/hf.py` | 与 Hugging Face Qwen3 格式互转：`load_from_hf_qwen3`、`export_to_hf_qwen3` |

### 两层对拍

**第一层对拍：极简版 = zero。** [`code/04_parity_with_zero.py`](code/04_parity_with_zero.py) 按极简模型的超参建一个 `zero.Transformer`，把训练好的权重直接 `load_state_dict` 进去（参数名一一对应，严格模式也能过），同一段文本比较 logits：

```bash
uv run python chapters/09-modern-transformer/code/04_parity_with_zero.py
```

```
参数量：极简 836,992  zero 836,992
1) 极简 vs zero：logits 形状 (1, 66, 256)，最大绝对差 1.43e-05，下一个字节的预测相同的位置 100%
2) 极简 vs transformers.Qwen3ForCausalLM：最大绝对差 1.43e-05
3) 贪心生成 80 个字节，两边完全相同：True
```

对齐只需要两处配置：极简版没有 GQA，所以 zero 设 `n_kv_heads = n_heads`（每个查询头有自己的 K/V，就是普通多头注意力）；极简版没有 bias、共享 embedding、`θ_base = 10000`、`eps = 1e-6`，zero 同样设置。剩下的差别是计算顺序：极简版手写 `softmax(QKᵀ/√d)·V`，zero 调用 PyTorch 的 SDPA，数学相同、浮点运算顺序不同，所以是**浮点误差内一致**（训练后最大差 1.4 × 10⁻⁵，`torch.testing.assert_close(rtol=1e-5, atol=1e-5)` 通过），不是逐位相同；随机初始化权重时最大差是 5.96 × 10⁻⁷。第 3 项还顺带检查了第 10 章的内容：zero 用 KV cache 贪心生成的 80 个字节，和极简版每步重算整段的结果一字不差。

**第二层对拍：zero = Qwen3 官方实现。** 第 2 项把 zero 模型用 `export_to_hf_qwen3` 导出成 Hugging Face 目录，再用 transformers 官方的 `Qwen3ForCausalLM.from_pretrained` 读回来，logits 一致。这一层的系统性证据在 [`tests/test_model_hf_parity.py`](../../tests/test_model_hf_parity.py)：随机初始化一个小的 `Qwen3ForCausalLM`（连 RMSNorm 的权重也随机化，让对拍更严），权重搬进 zero，覆盖共享/不共享 embedding、默认 RoPE、两种 YaRN 缩放、无 GQA 且 `n_heads × head_dim ≠ d` 五种情况，logits 误差都在 1e-5 以内；另外核对参数量与 HF 完全相同、loss 与 HF 一致：

```bash
uv run pytest -q tests/test_model_hf_parity.py      # 本机：8 passed in 4.24s
```

所以 **zero 就是一个 Qwen3 结构（稠密版）的实现**：训练出来的模型用 `zero/hf.py` 导出后，transformers、vLLM、llama.cpp 都能直接加载（第 20 章）。这也是本章极简模型的最终身份：一个 4 层、词表 256 的"迷你 Qwen3"。

### 生产级训练入口

同一个模型结构，在 `zero` 里由配置驱动训练：

```bash
uv run python -m zero.train.pretrain --config configs/tiny/pretrain.toml
```

本机实跑（为了不覆盖仓库里已有的 `out/tiny/pretrain`，加了 `--set train.out_dir=<临时目录>`）：`configs/tiny` 是 1.31M 参数（4 层、d = 128、GQA 4/2、不共享 embedding）、词表 2048 的 BPE（在 `assets/tiny_corpus` 的中、英、代码三份语料上训练；分片和分词器不存在时由配置里的 `[data.prepare]` 现场生成），三个来源按 0.45 / 0.45 / 0.1 混合，每步 2048 个 token，200 步：

| 步数 | 1 | 25 | 50 | 100 | 150 | 200 |
|---|---:|---:|---:|---:|---:|---:|
| 训练 loss（nat/token） | 7.651 | 6.384 | 6.394 | 6.107 | 5.869 | 5.474 |
| 验证 loss | | | | 6.005 | | 5.712 |

墙钟 1 分 25 秒，吞吐约 3,650–6,240 token/秒（单线程，机器繁忙）。这是冒烟测试：它只说明"同一份生产代码能端到端跑通、loss 在降"。它的 loss 按 BPE token 计（词表 2048），不能和上面按字节算的 bits-per-byte 直接比较；换算需要知道每个 token 平均几个字节（第 7 章），这里不做。

## 前沿观察

下面这些出现在最新的模型里，但还没形成本课规则下的共识，不进正文：

- **不用 RoPE 的层（NoPE）**：SmolLM3 的模型卡写明用了"GQA 和 NoPE（3:1）"，`config.json` 里每 4 层有 1 层不加 RoPE。因果 mask 本身就泄露了一点位置信息（第 i 个位置能看到 i 个 token），所以完全不加位置编码的 decoder 也能训。目前只看到少数家族这样做。
- **归一化放在子层输出上**：OLMo 2 把 RMSNorm 从子层入口挪到了子层输出（`h = x + RMSNorm(Attn(x))`，报告称之为 reordered norm），Gemma 3 则入口、输出都放（报告写"post-norm and pre-norm with RMSNorm"）。这说明"Pre-Norm 之外再加一道"在大规模训练里有人在试，但主流（Llama、Qwen、DeepSeek、gpt-oss）仍是纯 Pre-Norm。
- **只旋转部分维度（partial RoPE）**：Qwen3.5-0.8B 的 `config.json` 里 `partial_rotary_factor = 0.25`，GLM-4.5-Air 是 0.5。
- **GeGLU**：Gemma 系列用 GELU 做门（`hidden_activation = gelu_pytorch_tanh`），结构与 SwiGLU 相同，只换了激活函数。

## 采用方与来源

| 技术 | 采用方（头部开源模型家族） | 来源 |
|---|---|---|
| RMSNorm | Llama | LLaMA 论文 2.2 节"Pre-normalization … We use the RMSNorm"：<https://arxiv.org/abs/2302.13971> |
| | Qwen3 | Qwen3 技术报告第 2 节"RMSNorm with pre-normalization"：<https://arxiv.org/abs/2505.09388> |
| | Gemma 3 | Gemma 3 技术报告第 2 节"post-norm and pre-norm with RMSNorm"：<https://arxiv.org/abs/2503.19786> |
| | OLMo 2 | OLMo 2 报告 2.1 节（从非参数 LayerNorm 换成 RMSNorm）：<https://arxiv.org/abs/2501.00656> |
| | gpt-oss | 模型卡 2.2 节"root mean square normalization … before each attention and MoE block"：<https://arxiv.org/abs/2508.10925> |
| Pre-Norm | Llama | LLaMA 论文 2.2 节"normalize the input of each transformer sub-layer"：<https://arxiv.org/abs/2302.13971> |
| | Qwen3 | Qwen3 技术报告第 2 节"with pre-normalization"：<https://arxiv.org/abs/2505.09388> |
| | gpt-oss | 模型卡 2.2 节"Similar to GPT-2 we use Pre-LN placement"：<https://arxiv.org/abs/2508.10925> |
| | DeepSeek-V3 | 技术报告图 2（每个 Block 里 RMSNorm 在注意力和 FFN 之前）：<https://arxiv.org/abs/2412.19437> |
| SwiGLU | Llama | LLaMA 论文 2.2 节"SwiGLU activation function … 2/3·4d"：<https://arxiv.org/abs/2302.13971> |
| | Qwen3 | Qwen3 技术报告第 2 节：<https://arxiv.org/abs/2505.09388>；`Qwen/Qwen3-0.6B` 的 `config.json`（`hidden_act: silu`） |
| | OLMo 2 | OLMo 2 报告 2.1 节"SwiGLU … approximately 8/3·d"：<https://arxiv.org/abs/2501.00656> |
| | gpt-oss | 模型卡 2.2 节"The MoE blocks use the gated SwiGLU activation"（带截断和残差的非标准实现）：<https://arxiv.org/abs/2508.10925> |
| RoPE | Llama | LLaMA 论文 2.2 节"Rotary Embeddings"：<https://arxiv.org/abs/2302.13971> |
| | Qwen3 | Qwen3 技术报告第 2 节：<https://arxiv.org/abs/2505.09388> |
| | Gemma 3 | Gemma 3 技术报告第 2 节（全局层 RoPE 基频 1M、局部层 10k）：<https://arxiv.org/abs/2503.19786> |
| | DeepSeek-V3 | 技术报告 2.1.1 节（MLA 中单独携带 RoPE 的解耦 key）：<https://arxiv.org/abs/2412.19437> |
| | gpt-oss | 模型卡 2.2 节"rotary position embeddings … YaRN"：<https://arxiv.org/abs/2508.10925> |
| QK-Norm | Qwen3 | Qwen3 技术报告第 2 节"introduce QK-Norm … to ensure stable training"：<https://arxiv.org/abs/2505.09388> |
| | Gemma 3 | Gemma 3 技术报告第 2 节"we replace the soft-capping of Gemma 2 with QK-norm"：<https://arxiv.org/abs/2503.19786> |
| | OLMo 2 | OLMo 2 报告 2.1、3.3.2 节"This avoids attention logits being too large"：<https://arxiv.org/abs/2501.00656> |
| 小模型共享 embedding | Qwen3 | 技术报告表 1：0.6B / 1.7B / 4B 共享，8B 及以上不共享：<https://arxiv.org/abs/2505.09388>；`Qwen/Qwen3-0.6B` 的 `config.json`：`tie_word_embeddings: true` |
| | Llama 3.2 | `Llama-3.2-1B` 的 `config.json`：`tie_word_embeddings: true`（官方仓库需申请访问，这里读的是 `unsloth/Llama-3.2-1B` 镜像：<https://huggingface.co/unsloth/Llama-3.2-1B/blob/main/config.json>） |
| | SmolLM | `HuggingFaceTB/SmolLM3-3B` 与 `SmolLM2-135M` 的 `config.json`：`tie_word_embeddings: true`（<https://huggingface.co/HuggingFaceTB/SmolLM3-3B>） |
| | Gemma 3 | `google/gemma-3-1b-it` 的 `config.json` 未写该字段，transformers 的 `Gemma3TextConfig` 默认 `tie_word_embeddings=True`；技术报告表 1 只列了一份 302M 的 embedding 参数（= 262,144 × 1,152） |

以上 `config.json` 均于 2026-09-26 通过 Hugging Face Hub 读取。反例同样核对过：OLMo-2-0425-1B（`tie_word_embeddings: false`）、DeepSeek-V3、gpt-oss-20b 不共享。

## 引导问题

带着这些问题去问 Claude Code，直到你能用自己的话讲清楚：

1. 把 Block 里的 Pre-Norm 改成 Post-Norm（`x = RMSNorm(x + Attn(x))`），残差流上还剩一条"干净的恒等路径"吗？为什么这会让深层网络更难训？（提示：回到第 6 章的残差实验。）
2. RoPE 的点积只和 m − n 有关。那为什么还要让不同维度对用不同的转速？如果所有维度对都用 ω = 1，位置 1 和位置 1 + 2π·k 会怎样？
3. 为什么 RoPE 只转 q 和 k，不转 v？如果把 v 也转了，会发生什么？
4. 共享 embedding 让 `lm_head.weight` 和 `tok_emb.weight` 是同一个张量。反向传播时，这个矩阵的梯度从哪两处来？`weight decay` 应该对它怎么处理？
5. 把本章模型从字节级换成第 7 章的 BPE（比如词表 2048），参数量、每步看到的文本量、bits-per-byte 分别会怎么变？
6. 生成 400 个字节要调用模型 400 次，每次都把前面整段重新算一遍。哪些计算是重复的？（这就是第 10 章的起点。）

## 动手任务

**任务 1（基础）**：运行 `01_position.py`，把 `apply_rope` 的配对方式改成"相邻两维配对"（第 0、1 维一对，第 2、3 维一对……），验证点积仍然只和 m − n 有关；再想一想，为什么这样改之后 `04_parity_with_zero.py` 会对不上。

**任务 2（核心）**：在 `02_tiny_transformer.py` 里做消融，每次只改一处、都训练 400 步（`--steps 400`），记录验证集 bits-per-byte：(a) 去掉 RoPE；(b) 把 SwiGLU 换成 `4d` 的 GELU MLP；(c) 去掉 QK-Norm；(d) 取消共享 embedding。哪一项影响最大？结果和你的预期一样吗？（在这么小的规模上，有的差别可能在噪声以内——换一个随机种子再跑一次看看。）

**任务 3（挑战）**：把训练数据换成 `assets/tiny_corpus/chinese_poetry.txt`。一个汉字在 UTF-8 里是 3 个字节，同样 128 字节的上下文只能看到约 40 个字。训练后生成的诗是否经常出现半个汉字（显示为 �）？试着把 `seq_len` 调大或者换成第 7 章的 BPE，比较效果和速度。

## 想深入：CS336

本章对应 [CS336](https://cs336.stanford.edu/)（Spring 2026）的**第 3 讲"架构与超参"**：归一化的位置、激活函数与门控、位置编码、FFN 宽度、头数、词表大小这些选择，以及各家模型为什么这样选，是本章"采用方"表格的进阶版。**作业 1（Basics）**要求从零写出 BPE、Transformer 语言模型、AdamW 和训练循环——做完本章再去做它，会轻松很多。课程页有每讲的讲义和录像。

## 本章参考文献

- Vaswani et al. (2017). *Attention Is All You Need*：<https://arxiv.org/abs/1706.03762>
- Radford et al. (2019). *Language Models are Unsupervised Multitask Learners*（GPT-2）：<https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf>
- Xiong et al. (2020). *On Layer Normalization in the Transformer Architecture*（Pre-LN vs Post-LN）：<https://arxiv.org/abs/2002.04745>
- Zhang & Sennrich (2019). *Root Mean Square Layer Normalization*：<https://arxiv.org/abs/1910.07467>
- Su et al. (2021). *RoFormer: Enhanced Transformer with Rotary Position Embedding*：<https://arxiv.org/abs/2104.09864>
- Shazeer (2020). *GLU Variants Improve Transformer*：<https://arxiv.org/abs/2002.05202>
- Dehghani et al. (2023). *Scaling Vision Transformers to 22 Billion Parameters*（QK-Norm）：<https://arxiv.org/abs/2302.05442>
- Wortsman et al. (2023). *Small-scale proxies for large-scale Transformer training instabilities*：<https://arxiv.org/abs/2309.14322>
- Press & Wolf (2017). *Using the Output Embedding to Improve Language Models*（共享 embedding）：<https://arxiv.org/abs/1608.05859>
- Touvron et al. (2023). *LLaMA*：<https://arxiv.org/abs/2302.13971>
- Qwen Team (2025). *Qwen3 Technical Report*：<https://arxiv.org/abs/2505.09388>
- Gemma Team (2025). *Gemma 3 Technical Report*：<https://arxiv.org/abs/2503.19786>
- OLMo Team (2025). *2 OLMo 2 Furious*：<https://arxiv.org/abs/2501.00656>
- OpenAI (2025). *gpt-oss-120b & gpt-oss-20b Model Card*：<https://arxiv.org/abs/2508.10925>
- DeepSeek-AI (2024). *DeepSeek-V3 Technical Report*：<https://arxiv.org/abs/2412.19437>
- nanoGPT（单文件 GPT-2 结构的训练代码，可以和本章的现代结构对照读）：<https://github.com/karpathy/nanoGPT>
- minimind（中文、从零训练的小型 Llama 结构模型）：<https://github.com/jingyaogong/minimind>
- 1.5 万字速通 LLM 主流模型结构（Llama、Qwen、GLM、DeepSeek……）：<https://zhuanlan.zhihu.com/p/2060741715095560795>
- CS336 Language Modeling from Scratch：<https://cs336.stanford.edu/>

**下一章**：我们的模型会写字了，但写得很慢：每生成一个字节，都把前面整段重新算一遍，其中绝大部分是上一步刚算过的。第 10 章把算过的 K、V 存起来（KV cache），再让多个查询头共享一组 K、V（GQA），让缓存变小——顺便把温度、top-p 这些采样旋钮讲清楚。
