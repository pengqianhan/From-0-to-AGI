# 第 16 章：SFT —— 教底座模型回答问题、调用工具

[English](README.md) · **中文**

> **目标**：读完这一章，你能把一段含工具调用的多轮对话手写成 ChatML 格式的一整串文本，并指出哪些 token 计入损失。你能写出"只在助手 token 上计算损失"的训练代码，并解释 `<|im_end|>` 为什么必须计入损失。你能说清把多条对话打包进一个窗口省了什么、代价是什么。你还能为主线模型挑出许可证合适的指令数据和工具调用数据。

📺 **本章视频**：还没有发布。要在本机渲染，运行 `bash chapters/16-sft/video/build.sh`。
🧪 **本章自检**：学完后，在 Claude Code 里输入 `/ch16-sft`。

---

第三部分结束时，我们有了主线的底座模型（base model）。预训练给它喂了几千亿 token。中期训练用高质量数据给它收尾。长上下文扩展让它能读 32K token。按第 11 章的考卷，它能做选择题，也能续写代码。但是问它"北京今天天气怎么样？"，它不会回答。它会**接着写**：也许再写几个问题，也许写一段新闻，也许写一首诗。

**这一章要解决一个问题：只会续写的底座，怎样变成一个助手？这个助手要按格式回答，在需要工具时写出正确的工具调用。** 答案是监督微调（supervised fine-tuning，SFT）。SFT 是第四部分"后训练"的第一步。后面的蒸馏（第 17 章）、DPO（第 18 章）、GRPO（第 19 章）都从它出发。

SFT 的算法几乎没有新东西。它还是第 7 章以来一直在用的 next-token 交叉熵。新东西全在数据的形式上。**对话模板**（chat template）把对话变成文本。**loss mask** 决定哪些 token 计入损失。**打包**（packing）把长短不一的对话高效地装进训练窗口。这三样里错一个字符，部署后的模型就会出奇怪的问题。所以本章把它们一个字符一个字符地拆开看。

## 1. 直觉：底座只会续写

先看底座的真实表现。第 10 章我们在莎士比亚文本上训练过一个 0.86M 参数的字符级小模型。把一个问题直接输入给它（`code/03_sft_tiny.py`）：

```
Question:      Is it raining in Dallas?
Continuation:  \nCAMILLANUS:\nI will the son the shall the prove the shall th
```

它没有"回答"的概念。预训练的目标从头到尾都是"猜下一个 token"。在它见过的文本里，问句后面最常出现的是下一个角色的台词，所以它写台词。大模型也一样。底座见过的网页里，问句后面可能是答案，也可能是更多问句、广告或评论区。**模型会不会回答，不取决于它懂不懂，而取决于它认为接下来最可能出现什么。**

SFT 的想法很朴素。模型会模仿它见过的文本，那就给它看大量"问题后面紧跟着好回答"的文本。见得多了，"问题 → 回答"就成了模型眼中最可能的续写。具体做法是：准备一批对话，写明用户说什么、助手怎样回答、该调用什么工具。按固定格式把每条对话拼成文本，在这些文本上继续训练几千步。训练目标和预训练相同，只改两件事：

1. 数据从"网页文本"换成"格式化的对话"。
2. 只在**助手说的话**上计算损失。

写成公式：一条对话渲染成 token 序列 `x_1 … x_T`，其中属于助手输出的位置集合是 `A`：

```
L_SFT = − 1/|A| · Σ_{t ∈ A} log p_θ(x_t | x_<t)
```

和预训练的 `−1/T · Σ_t log p_θ(x_t | x_<t)` 相比，只是求和范围从全部位置缩小到 `A`。`code/02_loss_mask.py` 里的这几行做的就是这件事。`y` 里不计入损失的位置换成 `−100`。交叉熵的 `ignore_index` 跳过这些位置，并在剩下的位置上求平均：

```python
def masked_targets(ids, mask, use_mask=True):
    x = torch.tensor(ids[:-1])
    y = torch.tensor(ids[1:])
    if use_mask:
        y = torch.where(torch.tensor(mask[1:]), y, torch.full_like(y, -100))
    return x, y
# loss = F.cross_entropy(logits, y, ignore_index=-100)
```

注意是 `mask[1:]`。第 t 个位置预测的是第 t+1 个 token。所以这个位置算不算损失，看的是**被预测的那个 token** 是不是助手的。这是 SFT 代码里最常见的差一错误（off-by-one error）。

## 2. 对话模板：把一段对话变成一串文本

模型只认识一串 token。一段对话有多个角色、多轮，还有工具调用。所以要先约定一种格式，把对话压成一维文本。这个约定叫**对话模板**（chat template）。训练和推理必须使用**逐字相同**的模板。部署时，推理框架必须拼出和训练时相同的格式。差一个换行，模型看到的就是训练分布以外的输入。

### 2.1 ChatML：角色标记

今天开源模型里最常见的格式源自 ChatML（Chat Markup Language）。OpenAI 在 2023 年公开了它。每条消息用两个特殊 token 包起来：

```
<|im_start|>角色
内容<|im_end|>
```

角色是 `system`、`user`、`assistant` 之一。第 7 章讲过，`<|im_start|>` 和 `<|im_end|>` 是**特殊 token**（special token）。它们在词表里各占一个 id。BPE 不会把它们拆开，普通文本也不会意外拼出它们。zero 的分词器把它们固定在 id 1 和 2。

我们在 Hugging Face 上逐一核对了原始的 `tokenizer_config.json` / `chat_template.jinja`（2026-09 读取）。Qwen3、Qwen3.5、SmolLM3、Hermes 3、OLMo 3 的 Instruct 模型都用这套角色标记。它们在别的细节上各有不同，2.4 节会讲。

### 2.2 工具调用：三个约定

工具调用在 ChatML 上加了三个约定。这些约定源自 Nous Research 的 Hermes 函数调用格式。Qwen3 的官方模板也用这一套：

1. **工具清单写在 system 里。** 每个工具是一个 JSON Schema（名字、说明、参数类型），一行一个，放在 `<tools></tools>` 之间。清单前后各有一段固定的英文说明。
2. **助手的调用写成 `<tool_call>{json}</tool_call>`。** JSON 里只有 `name` 和 `arguments` 两个字段。要调用几次，就写几段。
3. **工具的返回值放进一个 user 轮。** 每条结果包在 `<tool_response></tool_response>` 里。连续几条结果合并进同一个 user 轮。

下面是一条真实的训练样本。它是 `zero/post/envs/tool_env.py` 生成的标准解答轨迹，即冒烟测试的 `out/smoke/sft/train.jsonl` 第 1 行。这里的工具清单只留了用到的 `get_weather`。代码块是 `code/01_chat_template.py` 渲染出的**完整字符串**：

```
<|im_start|>system
你是一个会使用工具的助手。需要时调用工具，拿到结果后用一句话回答。

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name": "get_weather", "description": "查询城市今天的天气", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call><|im_end|>
<|im_start|>user
成都和广州今天哪个更热？<|im_end|>
<|im_start|>assistant
<tool_call>
{"name": "get_weather", "arguments": {"city": "成都"}}
</tool_call>
<tool_call>
{"name": "get_weather", "arguments": {"city": "广州"}}
</tool_call><|im_end|>
<|im_start|>user
<tool_response>
{"city": "成都", "condition": "阴", "temp_c": 21, "humidity": 65}
</tool_response>
<tool_response>
{"city": "广州", "condition": "雷阵雨", "temp_c": 31, "humidity": 85}
</tool_response><|im_end|>
<|im_start|>assistant
广州更热（成都 21°C，广州 31°C）。<|im_end|>
```

读一遍这段文本，就读懂了一个工具调用模型的"一生"。系统提示告诉它有哪些工具、调用格式是什么。用户提问。它输出两段 `<tool_call>`。推理框架（如 vLLM 的 `hermes` 解析器）解析出这两段 JSON，真的去调用天气 API。框架把结果包成 `<tool_response>`，放回对话。模型再读结果，用一句话回答。

推理时，模板把提示词渲染到最后一条用户消息为止。然后加上 `<|im_start|>assistant\n` 作为"生成提示"（generation prompt）。模型从这里接着写，写到 `<|im_end|>` 就停：

```
Last 40 characters of the prompt at inference time: '今天哪个更热？<|im_end|>\n<|im_start|>assistant\n'
```

手写的渲染函数 `render()` 大约 40 行。核心是这一段（`code/01_chat_template.py`）：

```python
elif role == "assistant":
    segs.append((f"{IM_START}assistant\n", "assistant", False))  # role header: the prompt gives it, so do not learn it
    text = m.get("content") or ""
    for j, tc in enumerate(m.get("tool_calls") or []):
        if text or j > 0:
            text += "\n"
        text += ('<tool_call>\n{"name": ' + tojson(tc["name"]) + ', "arguments": '
                 + tojson(tc["arguments"]) + "}\n</tool_call>")
    segs.append((text + IM_END, "assistant", True))  # ← only this segment is in the loss (with <|im_end|>)
```

它返回的不只是文本。它返回一段段 `(文本, 角色, 是否计入损失)`。脚本最后和生产级的 `zero.post.chat.render_text` 对拍。完整对话和生成提示两种情况下，两边的字符串都**逐字相同**。

### 2.3 模板要"一字不差"

两个事实说明格式细节很重要：

- Tülu 3 技术报告（表 13）在同一份中间 SFT 数据上只改模板：把助手消息末尾的换行换成 eos、去掉换行、换成 Zephyr 或 Llama 3 模板。平均分在 51.6 到 53.0 之间变动。几个模板之间只差几个字符。
- zero 的模板写了两份。Python 版在训练时渲染，同时计算 mask。Jinja 版在导出时写进 `tokenizer_config.json`，给 transformers / vLLM / llama.cpp 使用。`tests/test_chat.py` 用 transformers 的 Jinja 引擎对 12 种组合逐字对拍。冒烟测试导出 Hugging Face 格式后，再次确认两份模板一致（`hf_template_identical=True`）。

### 2.4 各家的差别：角色标记一致，调用语法不一

核对官方模板时，我们也看到了分歧。这里列出来，以免你以为"大家都一样"：

| 模型 | 角色标记 | 工具清单 | 调用语法 | 工具返回 |
|---|---|---|---|---|
| Qwen3 | ChatML | system 里的 `<tools>` | `<tool_call>` 包 JSON（`name`、`arguments`） | user 轮里的 `<tool_response>` |
| Hermes 3（Llama 3.1 底座） | ChatML | system 里的 `<tools>` | `<tool_call>` 包 JSON（`arguments` 在前） | 独立的 `tool` 角色 + `<tool_response>` |
| SmolLM3 | ChatML | system 里的 `<tools>`（`xml_tools`） | `<tool_call>` 包 JSON | 放进 user 轮 |
| Qwen3.5 | ChatML | system 里的 `<tools>` | `<tool_call>` 里是 XML：`<function=名字><parameter=参数>值</parameter></function>` | user 轮里的 `<tool_response>` |
| OLMo 3 Instruct | ChatML | system 里的 `<functions>` | `<function_calls>` 里是 Python 调用 `name(a=1)` | 独立的 `environment` 角色 |

角色标记（ChatML）是共识。工具调用的外壳多数用 `<tool_call>`。各家的不同在于：里面写 JSON 还是 XML 参数，工具结果放在哪个角色里。主线模型选 Qwen3 / Hermes 风格的 JSON。理由是 vLLM 的 `hermes` 解析器和 llama.cpp 都直接支持它。所以第 20 章导出后，不用写任何适配代码。

zero 的模板与 Qwen3 只差一处。Qwen3 在"关闭思考"时，会在生成提示后面加一个空的 `<think>\n\n</think>\n\n`。我们不加，因为主线模型从没见过思考格式。

## 3. loss mask：只学助手说的话

### 3.1 哪些 token 计入损失

`01_chat_template.py` 逐段打印出每一段是否计入损失：

```
  no loss  system    '<|im_start|>system\n你是一个会使用工具的助手。…
  no loss  user      '<|im_start|>user\n成都和广州今天哪个更热？<|im_end|>\n'
  no loss  assistant '<|im_start|>assistant\n'
★ in loss  assistant '<tool_call>\n{"name": "get_weather", "arguments": {"city": "成都"}}\n</…
  no loss  assistant '\n'
  no loss  tool      '<|im_start|>user\n<tool_response>\n{"city": "成都", …
  no loss  tool      '\n<tool_response>\n{"city": "广州", …
  no loss  assistant '<|im_start|>assistant\n'
★ in loss  assistant '广州更热（成都 21°C，广州 31°C）。<|im_end|>'
  no loss  assistant '\n'

Total characters: 1126. In the loss: 197 (17.5%)
Characters per role: {'system': 622, 'user': 40, 'assistant': 243, 'tool': 221}
```

规则只有三条：

1. **助手的输出计入损失**，包括工具调用的每一个字符。
2. **助手末尾的 `<|im_end|>` 也计入损失。** 模型要学会"什么时候停"。少了它，模型推理时会一直写下去：写完回答，接着替用户编下一轮。
3. **其余都不计入损失**：system、user、工具返回，以及 `<|im_start|>assistant\n` 这个角色头。推理时角色头是提示词的一部分，不需要模型生成。

### 3.2 为什么不算用户的话

直觉上，"多学一点没有坏处"。那为什么要扔掉 80% 以上的文本？

- **目标不同。** 我们要的模型是"给定上下文，写出好回复"，不是"模仿用户提问"。用户的话和工具的返回，是模型在推理时**读**的，不是它要**写**的。
- **梯度预算。** 这条样本里，system 提示占 622 个字符，助手只有 197 个。如果全部计入损失，每一步的梯度大半花在"背诵每条样本都一样的工具说明"上。我们用 zero 的分词器编码了冒烟测试的全部 1,500 条训练对话。它们共 502,756 个 token，其中助手 token 只有 67,191 个，占 **13.4%**。
- **不该学工具返回。** 模型如果学会"生成" `<tool_response>` 里的内容，就是在学编造 API 的返回值。第 19 章的奖励函数会专门检查"伪造工具结果"（`tool_env.py` 的 `_forged`）。SFT 阶段不该教它这个。

Llama 2 的论文写得很直接："we zero-out the loss on tokens from the user prompt, so as a result, we backpropagate only on answer tokens"。下面这些都用同一个做法：Tülu 3 的训练代码 open-instruct、Hugging Face TRL 的 `assistant_only_loss`、nanochat 的 SFT 脚本、CS336 作业 5 的 `response_mask`。

### 3.3 损失怎样平均

`L_SFT` 除以的是 `|A|`：一个 batch 里所有助手 token 一起求平均。`02_loss_mask.py` 演示了一个容易犯的错误。样本 A 的助手只有 3 个 token（每个损失 2.0），样本 B 有 30 个（每个损失 1.0）：

```
Mean per token: 1.091; mean of each sample, then mean over samples: 1.500
```

梯度累积可能引入这个错误。做法是：每个 micro-batch 各自按"本批助手 token 数"求平均，再在 micro-batch 之间求平均。这样结果就会从前者悄悄变成后者。这时短回复的权重被放大。Tülu 3 报告（4.3.2 节）专门写了这个问题。他们改用求和损失（sum loss），并调整学习率。zero 目前按常见做法实现（`zero/post/sft.py` 开头的注释写明了这一点）。第二步如果发现短回复权重过大，可以改成跨 micro-batch 按 token 数加权。

## 4. 小实验：把小底座 SFT 成"会发工具调用"的模型

`code/03_sft_tiny.py` 把上面几件事串起来做一遍。任务是一个玩具工具调用，有两个工具：`get_weather(city)` 和 `add(a, b)`。问题从模板里随机生成（"Is it raining in {c}?"、"What is {a} plus {b}?"……），标准回复是一次工具调用。2,000 条训练对话用 30 个城市。测试用 **10 个训练里从没出现过的城市**，数字全是随机的。这样测的是"照抄"能力，而不是背城市名。

### 4.1 扩词表

底座的词表只有莎士比亚文本里的 65 个字符。没有 `{ } " _`，没有大部分数字，更没有 `<|im_start|>`。所以第一步是扩词表：追加 20 个新字符和 4 个特殊 token（65 → 89）。新行的 embedding 用旧行的平均值加一点噪声初始化。这样开始时，新 token 的 logits 不会过大或过小。

```
One toy sample (the tokens in [square brackets] are in the loss):
<|im_start|>system
Tools: get_weather(city), add(a, b)<|im_end|>
<|im_start|>user
What is 97 plus 53?<|im_end|>
<|im_start|>assistant
[<tool_call>
{"name": "add", "arguments": {"a": 97, "b": 53}}
</tool_call><|im_end|>]
137 tokens in total, 53 in the loss (39%)
```

这也是 zero 的分词器在第 7 章就把 `<|im_start|>`、`<tool_call>` 等特殊 token 预留在 id 0–15 的原因。预训练时它们就在词表里，所以 SFT 时不用改模型形状。

### 4.2 结果

> **关于数字**：本章训练类实验的数字来自课程构建机上的一次 CPU 运行。不同机器、不同版本的底层数学库，浮点运算的顺序略有不同。训练几百步后，这些微小差异会被放大。你本机跑出的数字可能从小数点后第二、三位开始就不一样。请以下文不依赖具体数值的结论为准。2026-10 在另一台服务器上的复跑对照见 [runs/2026-10-01-gpu0-check/chapters-16-20.md](../../runs/2026-10-01-gpu0-check/chapters-16-20.md)。

设置如下：全参数微调，AdamW，学习率 1e-3（小模型用的值；大模型要小得多，见第 7 节）。warmup 20 步，然后余弦衰减到 10%。batch 8，300 步（约 1.2 个 epoch），不做权重衰减。底座与 SFT 后的模型在同样的题目上：

```
Base model 'answer' with the chat template: 'The shall the then the shally the the the the the the th'

Answers after SFT (with mask):
  'What is 7 plus 11?'               → <tool_call>\n{"name": "add", "arguments": {"a": 1, "b": 12}}\n</tool_call>
  'Is it raining in Dallas?'         → <tool_call>\n{"name": "get_weather", "arguments": {"city": "Denver"}}\n</tool_call>
  'What is the weather in Nairobi?'  → <tool_call>\n{"name": "get_weather", "arguments": {"city": "Denver"}}\n</tool_call>
  'Compute 20+55.'                   → <tool_call>\n{"name": "add", "arguments": {"a": 19, "b": 15}}\n</tool_call>
```

50 道测试题上的统计（贪心生成，遇到 `<|im_end|>` 停）：

| 设置 | 格式对 | 函数名对 | 参数全对 | 验证损失（只算助手 token） |
|---|---:|---:|---:|---:|
| 底座（扩词表后，未 SFT；20 道题） | 0.00 | 0.00 | 0.00 | 4.640 |
| 有 mask，2,000 条 | 1.00 | 1.00 | 0.00 | 0.102 |
| 无 mask，2,000 条 | 1.00 | 1.00 | 0.00 | 0.110 |
| 有 mask，只有 40 条（约 60 个 epoch） | 1.00 | 1.00 | 0.00 | 0.366 |

把有 mask 模型的验证损失按 token 类型拆开：

```
Argument values (city names, numbers): 310 tokens, mean loss 1.146
Rest (format, function name):          3364 tokens, mean loss 0.006
```

换成训练里见过的城市（数字仍随机），参数全对的比例也只有 0.04。

三点观察：

1. **SFT 首先教会的是格式。** 300 步之后，JSON 结构、标签、函数名、在哪里停，几乎零误差（损失 0.006）。函数名也全选对了：模型读懂了"天气"和"加法"两类问题。
2. **最难的是参数。** 参数要从用户问题里**照抄**。这要求注意力学会"回头找到那个城市名，并逐字复制"。这个 0.86M 的小模型 300 步没学会。它的策略是：写一个训练里常见的城市（Denver），写一个看起来像数字的数字。损失几乎全集中在这 310 个参数 token 上。下面"主线进度"里，zero 冒烟测试的 tiny 模型表现**完全相同**：格式全对，函数名基本对，参数抄错。主线模型要在这里补足。它需要更大的模型和更多更杂的数据（第 17 章蒸馏）。它还需要强化学习，奖励是"调用是否真的执行正确"（第 19 章）。
3. **在这个玩具任务上，有没有 mask 差别很小**（验证损失 0.102 对 0.110）。这里的用户问题很短、很有规律。无 mask 版"顺便"背下问题，代价不大。真实数据里，system 提示和工具返回常常占 80% 以上的 token，差别会大得多。我们如实报告这个小结果，不把它说成"mask 的效果立即明显"。

另外，mask 还决定了"训练损失"这个数字的含义。同一个底座在同一批验证数据上，只算助手 token 是 4.640，全部 token 都算是 4.160。不说清楚算的是哪些 token，两个损失值就没法比较。

### 4.3 过拟合

表的最后一行只给 40 条数据，训练同样多的步数（约 60 个 epoch）：

```
step    1  train loss 4.561  val loss 4.537
step  100  train loss 0.086  val loss 0.316
step  200  train loss 0.046  val loss 0.343
step  300  train loss 0.055  val loss 0.366
```

训练损失一路降到 0.05。验证损失在第 100 步之后开始**回升**：模型在背那 40 条数据。SFT 的数据量通常比预训练小几个数量级，所以过拟合是常态风险。对策有三个：更多、更多样的数据；少训几个 epoch；按留出的验证集挑选 checkpoint，而不是按训练损失挑选。

## 5. 打包：把多条对话装进一个窗口

### 5.1 为什么要打包

对话长短不一。最简单的做法是一条对话占一行，补齐（pad）到窗口长度。补齐的位置不计入损失，但前向传播和反向传播照样要处理它们。这全是浪费。`code/04_packing.py` 造了 200 条长 130–390 token 的对话，窗口长 513（seq_len 512 + 1）：

```
No packing (one per row, padded to 513): 200 rows, real tokens 43%
First-fit packing: 94 windows, real tokens 91%, mean 2.1 conversations per window
```

**首次适配**（first-fit）的做法是：按顺序处理每条对话，放进第一个还装得下的窗口。都装不下，就开一个新窗口。有一条原则不能破：**一条对话不切开**。如果切开，后半段的助手回复就看不到前半段的问题，等于教模型"凭空回答"。代码直接丢弃放不下的超长对话，并记下数量。Llama 2 的 SFT 也把样本拼接起来，填满 4096 的窗口。nanochat 和 TRL 用的是装得更紧的最佳适配（best-fit）。

### 5.2 代价：串门

打包之后，在普通因果注意力下，同一窗口里第二条对话的 token 能**看到**第一条对话。`04_packing.py` 用 SFT 过的小模型量了一下。它把对话 A（问 Paris）和对话 B（问 Kyoto）装进同一窗口，看 B 的输出：

```
Loss on the assistant tokens of conversation B (Kyoto):
  alone                                0.5596
  packed after A (Paris), causal mask  0.4967   max logits difference 9.90e+00
  packed after A, document mask        0.5596   max logits difference 4.92e-05
```

在普通因果 mask 下，B 的 logits 最多变了 9.9，损失**降**了。A 是一条格式相同的工具调用，B 像是"抄了邻居的作业"。但这个方向并不可靠。2026-10 我们在另一台服务器上复跑了这个实验。第 10 章的底座和这里的 SFT 模型都在那台机器上重新训练，权重只有细微差别。同样三行是：

```
  alone                                0.5095
  packed after A (Paris), causal mask  0.7323   max logits difference 8.39e+00
  packed after A, document mask        0.5095   max logits difference 8.15e-05
```

这次损失反而**升**了，邻居成了干扰。小实验对权重很敏感，"升还是降"这种细节换一台机器就可能反过来。两次都成立的只有两点：在普通因果 mask 下，一条无关的对话大幅改变了 B 的输出（logits 最大差 8–10）；文档 mask 让 B 的输出和单独计算时一样。这就是串门（cross-contamination）。训练时，上下文里无关的对话影响了模型的预测。推理时却没有这个邻居。所以训练和推理的条件不完全相同。

解决办法叫**文档 mask**（document masking）。把因果 mask 换成块对角的下三角，让每条对话只能看自己：

```python
def document_mask(doc_ids):
    """Causal + only the same conversation: a block-diagonal lower triangle."""
    return causal(len(doc_ids)) & (doc_ids[:, None] == doc_ids[None, :])
```

加上文档 mask 以后，B 的损失与单独计算完全相同。logits 的差不到 1e-4，是浮点误差。还有一个细节：B 在窗口里从第 154 个位置开始。但 RoPE 只看相对位置（第 9 章）。只要注意力被隔开，绝对位置的偏移就不影响结果。

### 5.3 zero 的选择

zero 的 SFT **目前不隔离**打包在一起的对话。它用 PyTorch 的 SDPA 加普通因果 mask，与 nanochat 的 SFT 相同。这是一个有意的取舍：

- **好处**：实现简单，能直接使用最快的因果注意力 kernel。另外，SFT 数据里每条对话都以 `<|im_start|>system` 开头，模型很容易学会"新对话从这里开始"。冒烟测试里一条工具对话约 335 个 token，所以 512 的窗口平均只装得下 1.06 条（1,500 条装进 1,415 个窗口）。串门本来就很少。
- **代价**：主线 SFT 的窗口是 8192。大量短对话会被打包在一起，串门变多。
- **第二步的选项**：使用 FlashAttention 的变长（varlen）接口。传入每条对话的边界，它就能在不浪费算力的前提下隔离注意力。（TRL 的文档写明，它的打包只用于配合 FlashAttention 的设置。）本章脚本 `06_gpu_packing.py` 已在 RTX 3090 上实测了 varlen（见下文"GPU 实测"）。但 zero **还没有实现这条路径**。是否值得做，先用小实验比较"隔离与不隔离"对工具调用得分的影响，再做决定。

## 6. 数据：从哪来，许可证是什么

SFT 的效果主要由数据决定。Llama 2 的报告说，他们没有使用数百万条第三方数据。他们只用了 27,540 条自己标注的高质量样本，结果明显更好（"Quality Is All You Need"）。LIMA 论文的标题就是"Less Is More for Alignment"。但对小模型和工具调用这项具体能力，数据量仍然重要。Tülu 3 的消融实验显示，SFT 数据从 5% 加到 100%，平均分一直在涨（图 4）。

### 6.1 通用指令数据

| 数据集 | 规模 | 许可证（数据卡原文） | 备注 |
|---|---|---|---|
| [Tülu 3 SFT mixture](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture) | 939,344 条 | 整体 ODC-BY-1.0，但**子集各有许可证**："Some portions of the dataset are non-commercial" | 例如 No Robots 是 CC-BY-NC-4.0。还含第三方模型生成的输出，受这些模型各自条款的约束 |
| [SmolTalk](https://huggingface.co/datasets/HuggingFaceTB/smoltalk) | 约 1M 条 | 新造的四个子集是 Apache-2.0；混入的公开数据集："refer to the original dataset" | SmolLM2-Instruct 用它训练。另有面向 135M/360M 小模型的 [smol-smoltalk](https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk) |
| [SmolTalk2](https://huggingface.co/datasets/HuggingFaceTB/smoltalk2) | Mid / SFT / Preference 三部分 | 待核实（数据卡元数据未标许可证） | SmolLM3 后训练三个阶段的数据 |

规矩（GOAL.md 3.3）：**逐个子集记录来源和许可证**。非商用的子集不进主线。对闭源模型生成的数据，还要看生成模型的使用条款是否允许"用输出训练别的模型"。数据卡里常常写不清这一项。一律先记"待核实"，再逐项查。

### 6.2 工具调用数据

| 数据集 | 规模 | 许可证 | 生成方式 |
|---|---|---|---|
| [xLAM function-calling 60k](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k)（APIGen） | 60,000 条 | CC-BY-4.0（需申请访问） | APIGen 流水线合成，经三级验证：格式检查、**真实执行**、语义检查 |
| [ToolACE](https://huggingface.co/datasets/Team-ACE/ToolACE) | 1 万–10 万条 | Apache-2.0 | 自演化合成 26,507 个 API；多智能体生成对话；规则 + 模型双层验证；中英双语 |
| [Hermes Function-Calling V1](https://huggingface.co/datasets/NousResearch/hermes-function-calling-v1) | 5 个子集 | Apache-2.0 | Hermes 2 Pro 的函数调用与 JSON 模式数据，格式就是本章的 `<tool_call>` |

SmolTalk 本身也混入了 8 万条 APIGen 函数调用数据（Synth-APIGen + xLAM）。这三份数据都是**合成**的。生成它们的模型及其条款，要逐一核实（待核实）。

### 6.3 自己合成：可执行、可验证

主线的另一路数据来自自己的工具环境。`zero/post/envs/tool_env.py` 有 6 个模拟 API：计算器、天气、单位换算、日期加减、日期间隔、星期几。`reference_messages(task)` 生成"用户 → 助手调用 → 工具结果 → 助手回答"的标准解答轨迹。本章 2.2 节那条样本就是它生成的。这类数据的好处是：每一条调用都能**真的执行**，程序可以自动判定结果对不对。这正是 xLAM 强调的"执行验证"思路。第 17 章会用更大的教师模型在更多工具上生成轨迹，再用同样的执行验证筛掉错误样本。

### 6.4 去污染

GOAL.md 3.2 要求所有训练数据与评测集做 13-gram 重叠检查。工具调用数据还要检查**函数名和 schema** 是否和 BFCL 等评测集重合。公开的函数调用数据集常常覆盖同一批流行 API，所以和 BFCL 重合的风险比通用文本高得多。Tülu 3 的附录 B.2 也列出了多份公开数据集与常用评测的重叠。

## 7. 超参数：小学习率、少 epoch

SFT 是在一个已经很好的模型上"轻轻推一下"。所以学习率比预训练小得多，epoch 也很少：

| 来源 | 模型 | 学习率 | epoch | 其他 |
|---|---|---|---|---|
| Llama 2 论文 | 7B–70B | 2e-5（余弦） | 2 | 权重衰减 0.1，batch 64，序列 4096 |
| Tülu 3 报告 表 11 | Llama 3.1 8B / 70B | 5e-6 / 2e-6（线性） | 2 | 有效 batch 128，最大长度 4096，warmup 3%；报告说训练更久没有继续变好 |
| SmolTalk 数据卡 | SmolLM2（1.7B）对比实验 | 3e-4 | 2 | 序列 8192，全局 batch 16 |
| zero `configs/main/sft.toml` | 主线 0.69B | 5e-5（待调） | ≈2（待定） | 见下一节 |

有两个规律。第一，模型越大，学习率越小。Tülu 3 从 8B 到 70B 把学习率从 5e-6 降到 2e-6，到 405B 再降。报告把这称为更轻的一推（"lighter touch"）。第二，epoch 数多在 1–3。Tülu 3 还发现，只换随机种子，8B SFT 模型的平均分就会在 59.8–60.1 之间变动。所以不要把 0.2 分的差别当成超参数的功劳。

学习率太大或 epoch 太多，有两种后果。一是 4.3 节那样的过拟合。二是**遗忘**预训练学到的通用能力。SFT 前后都要测第 11 章的通用基准，确认没有明显掉分。

### 7.1 LoRA：简要说明

LoRA（Low-Rank Adaptation）冻结原模型的权重。它只在每个线性层旁边训练一对低秩矩阵 `ΔW = B·A`（秩通常是 8–64）。可训练参数少几个数量级，显存需求大大降低。一张消费级显卡就能微调 7B 模型。LoRA 是社区微调的事实标准。Hugging Face PEFT 和 TRL 都内置了它。TRL 文档建议 LoRA 的学习率约 1e-4，比全参数微调高一个数量级。

主线模型**不用 LoRA，做全参数微调**。理由很实际。模型只有 0.69B 参数，全参数的优化器状态在 8 张 H100 上没有问题。我们的目标是把工具调用能力推到极限。没有必要为了省显存，接受低秩约束可能带来的上限损失。读者想在自己的笔记本上用我们发布的底座模型做实验时，LoRA 是很好的选择。

## 8. 小结

- 底座只会续写。SFT 用同一个 next-token 交叉熵，在格式化的对话上继续训练。这样"问题 → 好回答"就成了模型眼中最可能的续写。
- 对话模板把多角色、多轮、工具调用压成一串文本。ChatML 的 `<|im_start|>角色\n…<|im_end|>` 是共识。工具清单写在 system 里，调用写成 `<tool_call>{json}</tool_call>`，结果包在 `<tool_response>` 里。训练和推理必须逐字一致。
- loss mask 只让助手输出（含工具调用、含 `<|im_end|>`）计入损失。在真实的工具对话里，这部分只占 13–18% 的 token。
- 打包把多条对话装进一个窗口，真实 token 的占比从 43% 提到 91%。代价是同一窗口的对话会串门。文档 mask 可以消除串门。zero 目前不隔离，这是有意的取舍。
- 数据决定上限。逐个子集核实许可证。工具调用数据优先选"可执行验证"的。用小学习率、1–3 个 epoch，按验证集挑选 checkpoint。
- 小实验和冒烟测试给出同一个信号：SFT 很快教会格式和函数名；最难的是从问题里照抄正确的参数。

---

## GPU 实测（单张 RTX 3090）

> 上面正文里的数字都来自 CPU 运行。本节换到一张 NVIDIA GeForce RTX 3090 上实测（24 GB 显存，Ampere 架构；规格表：BF16 张量核稠密峰值约 71 TFLOPS，FP32 约 35.6 TFLOPS，显存带宽约 936 GB/s）。环境：PyTorch 2.11.0+cu128、CUDA 12.8，2026 年 10 月。服务器把这张卡的功耗上限设成了 240 W（出厂默认 350 W），持续满载时它会降频。所以算力和带宽的绝对值比满功耗的 3090 偏低，看相对关系更可靠。没有 GPU 可以跳过本节。

运行：

```bash
uv run python chapters/16-sft/code/06_gpu_packing.py
```

**① 补齐与打包，改成计时。** 我们用主线模型的形状（`configs/main/sft.toml`，689.5M 参数，BF16 autocast，eager 模式）。把第 5.1 节那 200 条对话（44,007 个真实 token）做一遍前向 + 反向传播。损失只算助手 token，不含优化器的更新。每批 8 行。每种排法跑两遍，取中位数（两遍相差不到 0.2%）：

| 排法 | 批数 | 算过的位置 | 真实占比 | 耗时 | 真实 token/s | 相对第一行 |
|---|---:|---:|---:|---:|---:|---:|
| 一条一行，补齐到 513 | 25 | 102,400 | 43% | 14.57 s | 3,019 | 1.00× |
| 一条一行，补齐到批内最长 | 25 | 67,608 | 65% | 10.32 s | 4,262 | 1.41× |
| 首次适配打包（94 个窗口） | 12 | 48,128 | 91% | 7.03 s | 6,256 | 2.07× |

**② 隔离对话要花多少（5.3 节的取舍）。** 把同样的对话按主线 SFT 的 8192 窗口首次适配打包。第一个窗口装进 40 条对话（8,165 个 token）。文档 mask 里要算的（查询, 键）对只有因果 mask 的 2.9%。单测一层注意力（16 个查询头 / 8 个 K/V 头，head_dim 128，BF16）的前向 + 反向传播：

| 写法 | 前向 + 反向 | × 28 层 | 相对因果 | 显存峰值 | 与 varlen 输出的最大差 |
|---|---:|---:|---:|---:|---:|
| 普通因果（FlashAttention，会串门；zero 现在的做法） | 18.20 ms | 510 ms | 1.00× | 0.77 GiB | 4.62e+00 |
| 文档 mask：布尔矩阵传给 SDPA（PyTorch 选了 math 实现） | 227.95 ms | 6382 ms | 0.08× | 16.70 GiB | 1.56e-02 |
| 文档 mask：FlexAttention | 2.48 ms | 69 ms | 7.34× | 0.58 GiB | 7.81e-03 |
| 文档 mask：varlen FlashAttention | 2.43 ms | 68 ms | 7.49× | 0.84 GiB | 0.00e+00 |

补齐的位置在 GPU 上一点也不省：耗时基本跟着"算过的位置"走。打包后，同一张卡每秒处理的真实 token 是"补齐到 513"的 2.07 倍。只补齐到批内最长，能拿回一部分（1.41 倍）。5.3 节说，不隔离的好处是"能直接使用最快的因果注意力 kernel"。这只在和"把文档 mask 写成布尔矩阵"比较时成立。那种写法让 PyTorch 退回 math 实现：一层慢 12 倍，显存 16.7 GiB。FlexAttention 和 varlen FlashAttention 只算每条对话自己的那块下三角。它们反而比普通因果快 7 倍多，输出和 varlen 一致到 BF16 舍入。普通因果与 varlen 的最大差是 4.6，这就是串门。所以在 8192 这么长、装了几十条短对话的窗口里，隔离对话不用拿速度去换。代价在实现上：代码要把对话边界传给 kernel。zero 的这条路径仍然尚未实现（见 5.3 节）。

## 从极简代码到生产级代码

| 极简代码 | 生产级代码（`zero/`） | 多做了什么、为什么 |
|---|---|---|
| `01_chat_template.py` 的 `render()`：约 40 行，返回 (文本, 角色, 是否计入损失) | `zero/post/chat.py` 的 `render_segments` / `render` / `render_text` | 支持 OpenAI 风格的 `{"type": "function", "function": {...}}`、字符串形式的 `arguments`、同时有文字和调用的助手消息、思考内容（`enable_thinking`，默认关）。给了分词器时，它**整段一次编码**，再用每个 token 的字符起点查逐字符的 mask。它不分段编码再拼接，因为段边界处 BPE 的合并结果可能不同。只有整段编码，才与推理框架"拼好整串再编码"的做法一致 |
| （没有） | `zero/post/chat.py` 的 `CHAT_TEMPLATE`（Jinja） | 同一格式的第二份实现。导出时写进 `tokenizer_config.json`，让 transformers / vLLM / llama.cpp 拼出逐字相同的提示词。`tests/test_chat.py` 用 transformers 的 Jinja 引擎对拍：3 种对话 × 有无工具 × 有无生成提示 |
| `01` 的 `parse_tool_calls()`：一个正则 + `json.loads` | `zero/post/chat.py` 的 `parse_assistant` → `ParsedAssistant(content, tool_calls, reasoning_content, errors)` | 模型可能输出任何东西：标签不配对、JSON 坏了、多余字段、`arguments` 写成字符串……函数不抛异常，把错误记在 `errors` 里。第 19 章的奖励函数据此扣分。`test_render_parse_roundtrip` 保证渲染与解析互逆 |
| `02_loss_mask.py` 的 `masked_targets()`：mask 为 0 的目标换成 -100 | `zero/data/loader.py` 的 `MaskedWindowLoader.next_batch()` | 同样是 `np.where(mask[:, 1:], tok[:, 1:], -100)`。另外支持多卡分片（第 g 个样本给 rank `g % world_size`）、由 (seed, epoch) 确定的打乱、用 `state_dict` 精确续训。`test_masked_loss_matches_hand_computation` 与手算对拍，`test_window_loader_targets_and_resume` 测试续训 |
| `03_sft_tiny.py` 的 `batchify()`：右侧补齐 | `zero/post/sft.py` 的 `encode_example` + `pack_examples` | 每条对话末尾加一个 `<|endoftext|>`（mask=0）作分隔。首次适配打包：只回看最近 64 个没满的窗口，剩余不到 16 个位置的窗口就关闭。丢弃超长样本和没有助手 token 的样本，并计数。写成 `train.bin`（uint32）+ `train.mask`（uint8）+ 元数据（分词器哈希、seq_len）。换了分词器或长度，就拒绝旧文件 |
| `04_packing.py` 的 `pack_first_fit()` + `document_mask()` | `pack_examples`；**没有**文档 mask | 见 5.3 节的取舍。`test_pack_examples_keeps_conversations_whole` 保证对话不被切开 |
| `03` 的训练循环：几十行 | `zero/post/sft.py` 的 `run_sft` → 复用预训练的 `Trainer`（`[data] format = "sft"`） | BF16、梯度累积、DDP、断点续训、日志、checkpoint 全部复用。`[train] init_from` 指向长上下文阶段的 checkpoint。`test_run_sft_end_to_end_and_resume` 端到端训练，并验证续训 |
| `03` 用第 10 章的底座，并临时扩词表 | 第 7 章的分词器已经把特殊 token 预留在 id 0–15 | 预训练时它们就在词表里。SFT 不改模型形状，也没有"新 embedding 怎样初始化"的问题 |
| `03` 手写 2,000 条玩具样本 | `zero/post/sft.py` 的 `env_conversations` + `tool_env.reference_messages` | 配置了 `generate_train = N` 且数据文件不存在时，用工具环境生成标准解答轨迹。train 和 dev 互不重叠 |

在构建机上运行 `uv run pytest tests/test_chat.py tests/test_sft.py`：27 项全部通过（7.2 秒）。

### `configs/main/sft.toml` 逐项说明

| 配置 | 值 | 为什么 |
|---|---|---|
| `[model]` | 与 `configs/main/longctx.toml` 相同（689.5M，`rope_theta = 1e6`，`max_seq_len = 32768`） | SFT 从长上下文阶段的 checkpoint 接着训练，形状必须一致 |
| `init_from` | `out/main/longctx/ckpt` | 第 15 章的产物 |
| `seq_len` | 8192 | 工具说明 + 多轮对话 + 工具返回，一条就可能上千 token。代码丢弃更长的样本，并计数 |
| `micro_batch_size × grad_accum × 8 卡` | 4 × 2 × 8 | 每步 64 个窗口 × 8192 = 524,288 token（打包后约 70% 是真实 token） |
| `max_steps` | 2000 | ≈ 1.05B token；假设 SFT 数据约 0.5B token，约为 2 个 epoch（待定） |
| `lr` | 5e-5，余弦，warmup 100，最低 10% | 小模型 SFT 常用 1e-5 ~ 1e-4（待阶梯实验调） |
| `weight_decay` | 0 | 步数少，不需要 |
| `grad_clip` | 1.0 | 与预训练一致 |
| `eval_every` / `eval_batches` | 200 / 20 | 按验证集挑选 checkpoint（4.3 节） |

成本：`uv run python -m zero.tools.estimate_cost --config configs/main/pretrain.toml --tokens 1B --seq-len 8192` → 6.9 GPU·h，约 $17。这里的假设是 H100、MFU 0.4、每卡时 $2.5，**尚未在 GPU 上验证**。SFT 在整个预算里很便宜。贵的是第 17 章用教师模型生成数据。

---

## 主线进度

### 极小配置演示（CPU，`python -m zero.smoke`，约 1.3M 参数）

> **注意：**本节数字来自修复工具调用判分器（第 19 章第 6 节）**之前**的那次冒烟测试，是当时的真实输出。修复后我们重跑了一次（`uv run python -m zero.smoke --out out/smoke_final`）。数据、预训练、中期训练、SFT 各阶段的结果完全一致。蒸馏中通过验证的样本从 1 条（那句胡话）变为 0 条。下游的 DPO、GRPO 和评测数字随之变化。例如，同一个 SFT 模型的工具调用 call_exact 从 0.133 变为 0.100：模型没变，是判分更严了。GRPO 相对 SFT 仍判"持平"。你自己运行时，以运行结果为准。

> 以下是**极小配置演示**：只说明生产级代码的通路是通的，不代表主线模型的任何结果。

冒烟测试的 SFT 阶段从 tiny 中期训练的 checkpoint 出发，用 `tool_env` 生成的 1,500 条工具调用对话训练（`configs/tiny/sft.toml`；冒烟测试把 seq_len 改为 512、micro batch 改为 8、步数改为 240）。数字来自 `out/smoke/summary.json`、`out/smoke/sft/data/train.json` 和 `out/smoke/eval/results.json`。用 `code/05_smoke_samples.py` 可以重新打印：

| 指标 | 值 |
|---|---|
| 训练对话 / 打包后窗口 | 1,500 条 → 1,415 个窗口（seq_len 512，填充率 69%），没有丢弃 |
| 计入损失的 token | 67,191 个（全部 token 的 13.4%） |
| 训练 240 步后 | 训练损失 0.560，验证损失 0.620，用时 448 秒（CPU 单线程） |
| dev 集工具调用（30 题） | 格式正确 1.00，调用完全正确 0.133，参数 AST 一致 0.10，最终回答正确 0.167 |

模型写出来的是什么样？下面是 `05_smoke_samples.py` 的输出（dev 集前 6 题，贪心生成）：

```
Q: 深圳和巴黎今天哪个更热？
  reference: get_weather(city=深圳), get_weather(city=巴黎)
  model: <tool_call>{"name": "get_weather", "arguments": {"city": "东京"}}</tool_call> ×2
Q: Convert 68 F to C.
  reference: convert_units(value=68, from_unit=F, to_unit=C)
  model: <tool_call>{"name": "convert_units", "arguments": {"value": 14, "from_unit": "C", "to_unit": "F"}}</tool_call>
Q: 2027-07-20 是星期几？
  reference: weekday(date=2027-07-20)
  model: <tool_call>{"name": "weekday", "arguments": {"date": "2023-01-11"}}</tool_call>
```

结果与第 4 节的小实验一样：**格式完全正确，函数名基本正确，参数几乎都是编的**（一个训练里常见的城市，一个常见的日期）。1.3M 参数的模型没有学会从问题里照抄参数。30 题里调用完全正确的有 4 题，其中 3 题本来就不需要调用工具。

### 待 GPU 训练后补充

- 主线 SFT 的数据配方：通用指令数据、工具调用数据、自合成轨迹各占多少；每个子集的来源与许可证清单；BFCL 函数名 / schema 去污染的结果。
- 训练曲线、学习率与 epoch 的小规模扫描、按验证集挑出的 checkpoint。
- SFT 前后在第 11 章通用基准上的对比（确认没有明显遗忘），以及工具调用开发集上的得分。
- "打包是否隔离"的对比实验（5.3 节）。
- 实际花费（记入 `runs/ledger.md`）、失败与返工。

---

## 前沿观察

> **工具调用的语法还没有统一。** 角色标记（ChatML），以及"工具写在 system 里、调用包在 `<tool_call>` 里"，已经是多家的共识。但标签里面写什么正在分化。Qwen3、Hermes、SmolLM3 写 JSON。Qwen3.5 改成了 XML 风格的 `<function=名字><parameter=参数>值</parameter>`：参数值可以跨多行，写代码这类长参数时不用转义。OLMo 3 写 Python 函数调用。哪种格式对小模型最友好？我们没有找到公开的对照实验。主线沿用 JSON，理由是推理框架对它的支持最成熟。
>
> **SFT 打包时的文档 mask。** TRL 的打包只用于配合 FlashAttention 的设置（按对话边界隔离，其实现细节待核实）。nanochat 和 zero 不隔离。我们核实到的明确采用方不足三家，所以暂不写进正文，第二步用小实验决定（见 5.3 节）。

---

## 采用方与来源

**ChatML 风格的对话模板 + `<tool_call>` 工具调用格式**

- Qwen3：[`Qwen/Qwen3-0.6B` tokenizer_config.json 的 chat_template](https://huggingface.co/Qwen/Qwen3-0.6B/blob/main/tokenizer_config.json)（JSON 调用，与 zero 的模板只差关闭思考时的空 `<think>`）；[Qwen3 技术报告](https://arxiv.org/abs/2505.09388)
- Qwen3.5：[`Qwen/Qwen3.5-0.8B` chat_template.jinja](https://huggingface.co/Qwen/Qwen3.5-0.8B/blob/main/chat_template.jinja)（ChatML + `<tool_call>`，内部为 XML 参数）
- SmolLM3（Hugging Face）：[`HuggingFaceTB/SmolLM3-3B` chat_template.jinja](https://huggingface.co/HuggingFaceTB/SmolLM3-3B/blob/main/chat_template.jinja)
- Hermes 3（Nous Research，Llama 3.1 底座）：[模型卡 Prompt Format 一节](https://huggingface.co/NousResearch/Hermes-3-Llama-3.1-8B)；[技术报告](https://arxiv.org/abs/2408.11857)
- OLMo 3（Ai2）：[`allenai/Olmo-3-7B-Instruct` chat_template.jinja](https://huggingface.co/allenai/Olmo-3-7B-Instruct/blob/main/chat_template.jinja)（ChatML 角色标记，工具调用用 `<function_calls>`）
- ChatML 的原始说明：[openai-python `chatml.md`](https://github.com/openai/openai-python/blob/release-v0.28.0/chatml.md)

**只在助手 token 上计算损失（loss mask）**

- Llama 2：论文 3.1 节"zero-out the loss on tokens from the user prompt"，<https://arxiv.org/abs/2307.09288>
- Tülu 3 / OLMo（Ai2）：训练代码 [open-instruct](https://github.com/allenai/open-instruct) 的 `dataset_transformation.py` 把非助手消息的 labels 设为 -100；[Tülu 3 报告](https://arxiv.org/abs/2411.15124)
- SmolLM3：模板里用 `{% generation %}…{% endgeneration %}` 标出助手输出，transformers 据此返回助手 mask；[TRL 文档](https://huggingface.co/docs/trl/sft_trainer#train-on-assistant-messages-only)以它为 `assistant_only_loss` 的示例
- nanochat：[`scripts/chat_sft.py`](https://github.com/karpathy/nanochat/blob/master/scripts/chat_sft.py)（`render_conversation` 返回 ids 与 loss mask）
- CS336 作业 5：`run_tokenize_prompt_and_output` 返回 `response_mask`，<https://github.com/stanford-cs336/assignment5-alignment>

**SFT 数据与超参实践（公开的数据 + 小学习率 + 约 2 个 epoch）**

- Tülu 3：数据 [tulu-3-sft-mixture](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture)，超参见报告表 11（5e-6 / 2e-6，2 epoch）
- SmolLM2：数据 [SmolTalk](https://huggingface.co/datasets/HuggingFaceTB/smoltalk)，[SmolLM2 论文](https://arxiv.org/abs/2502.02737)
- Llama 2：27,540 条高质量标注、2e-5、2 epoch（论文 3.1 节）
- Hermes：函数调用数据 [hermes-function-calling-v1](https://huggingface.co/datasets/NousResearch/hermes-function-calling-v1)

**打包（多条样本拼进一个窗口）**：Llama 2（论文 3.1 节）、nanochat（`chat_sft.py` 的 best-fit 打包）、TRL（[Packing 文档](https://huggingface.co/docs/trl/reducing_memory_usage#packing)，默认 Best-Fit Decreasing）。

---

## 引导问题

向 Claude Code 提出这些问题。一直问到你能用自己的话讲清楚答案：

1. 把 2.2 节那条样本里的 `<|im_end|>` 全部从损失里拿掉，模型推理时会出什么问题？反过来，如果把 `<|im_start|>assistant\n` 也计入损失，会有害吗？
2. 第 4 节的小实验里，"有 mask"和"无 mask"几乎打平。设计一个玩具数据，让两者的差别变得明显。（提示：拉长 system 提示，或让用户消息里出现"像助手回复"的文本。）
3. 为什么 zero 的 `render` 要"整段一次编码，再按字符偏移查 mask"，而不是每段分别编码再拼接？构造一个例子，让 BPE 在段边界处的合并结果不同。
4. Qwen3.5 把工具调用参数从 JSON 改成了 `<parameter=…>` 的 XML 形式。设想一个调用，它的参数是一段多行 Python 代码。两种格式各要模型生成哪些额外的字符？哪种更容易出错？
5. 你手里有 Tülu 3 SFT mixture 和 xLAM 60k 两份数据，主线模型要商用发布。按本章的规矩，列出你要逐项核实的许可证与条款问题。
6. SFT 的验证损失已经降到 0.1，工具调用的参数却几乎全错（第 4 节）。为什么只看损失会被误导？你会加什么指标来监控训练？

## 动手任务

**任务 1（基础）**：运行 `01_chat_template.py`。在对话里再加一轮：用户说"谢谢"，助手回复"不客气"。预测新增的哪些字符计入损失，再看打印结果验证。用 `zero.post.chat.render_text` 对拍。

**任务 2（核心）**：修改 `02_loss_mask.py` 里的 `masked_targets`，故意把 `mask[1:]` 写成 `mask[:-1]`。用 `03_sft_tiny.py` 训练一次。观察模型在 `<|im_end|>` 处的行为和测试得分有什么变化，并解释原因。

**任务 3（挑战）**：让小模型学会照抄参数。可以试这些办法：把训练步数加到 2,000 步；把城市换成更多样的随机字母串；在 `03` 里加一个"只在参数 token 上加大权重"的损失；或者把第 10 章的模型换成更深的模型。记录每种办法在测试集上的"参数全对"。再检查 40 条数据时的过拟合是否更严重。全部在 CPU 上完成。

## 想深入：CS336

本章对应斯坦福 CS336（Spring 2026）<https://cs336.stanford.edu/>：

- **第 15 讲：中期训练与后训练（SFT / RLHF）。** 讲 SFT 的数据从哪里来、为什么"少而精"。还讲指令数据的风格与安全问题，以及它和后面 RLHF 的衔接（讲义与录像见课程页）。
- **作业 5（Alignment）**：`run_tokenize_prompt_and_output` 要求你把提示词和回复拼起来，构造与 labels 对齐的 `response_mask`。这正是本章 `masked_targets` 做的事。`get_packed_sft_dataset` 要求把指令数据打包成定长窗口。可选的补充作业涉及指令微调与 RLHF。作业仓库：<https://github.com/stanford-cs336/assignment5-alignment>

---

## 本章参考文献

- Touvron et al. *Llama 2: Open Foundation and Fine-Tuned Chat Models*，2023：<https://arxiv.org/abs/2307.09288>
- Lambert et al. *Tülu 3: Pushing Frontiers in Open Language Model Post-Training*，2024：<https://arxiv.org/abs/2411.15124>；训练代码 open-instruct：<https://github.com/allenai/open-instruct>
- Allal et al. *SmolLM2: When Smol Goes Big — Data-Centric Training of a Small Language Model*，2025：<https://arxiv.org/abs/2502.02737>
- Zhou et al. *LIMA: Less Is More for Alignment*，2023：<https://arxiv.org/abs/2305.11206>
- Ouyang et al. *Training language models to follow instructions with human feedback*（InstructGPT），2022：<https://arxiv.org/abs/2203.02155>
- Teknium et al. *Hermes 3 Technical Report*，2024：<https://arxiv.org/abs/2408.11857>
- Liu et al. *APIGen: Automated Pipeline for Generating Verifiable and Diverse Function-Calling Datasets*，2024：<https://arxiv.org/abs/2406.18518>
- Liu et al. *ToolACE: Winning the Points of LLM Function Calling*，2024：<https://arxiv.org/abs/2409.00920>
- Hu et al. *LoRA: Low-Rank Adaptation of Large Language Models*，2021：<https://arxiv.org/abs/2106.09685>
- Qwen Team. *Qwen3 Technical Report*，2025：<https://arxiv.org/abs/2505.09388>
- OpenAI. ChatML 说明（openai-python v0.28）：<https://github.com/openai/openai-python/blob/release-v0.28.0/chatml.md>
- Hugging Face TRL 文档：[SFT Trainer](https://huggingface.co/docs/trl/sft_trainer)、[Reducing Memory Usage（打包）](https://huggingface.co/docs/trl/reducing_memory_usage)
- 数据卡（2026-09 读取）：[tulu-3-sft-mixture](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture)、[smoltalk](https://huggingface.co/datasets/HuggingFaceTB/smoltalk)、[smoltalk2](https://huggingface.co/datasets/HuggingFaceTB/smoltalk2)、[xlam-function-calling-60k](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k)、[ToolACE](https://huggingface.co/datasets/Team-ACE/ToolACE)、[hermes-function-calling-v1](https://huggingface.co/datasets/NousResearch/hermes-function-calling-v1)
- 对话模板（2026-09 读取）：[Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B/blob/main/tokenizer_config.json)、[Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B/blob/main/chat_template.jinja)、[SmolLM3-3B](https://huggingface.co/HuggingFaceTB/SmolLM3-3B/blob/main/chat_template.jinja)、[Hermes-3-Llama-3.1-8B](https://huggingface.co/NousResearch/Hermes-3-Llama-3.1-8B)、[Olmo-3-7B-Instruct](https://huggingface.co/allenai/Olmo-3-7B-Instruct/blob/main/chat_template.jinja)
- [nanochat](https://github.com/karpathy/nanochat) 的 `scripts/chat_sft.py`；[minimind](https://github.com/jingyaogong/minimind) 的 SFT 部分（中文小模型的完整 SFT 流程）
- CS336 作业 5：<https://github.com/stanford-cs336/assignment5-alignment>

**下一章**：SFT 让模型学会了格式，但参数抄不对。一个办法是给它更多、更好的示范，而最好的示范来自一个更强的模型。第 17 章讲蒸馏：为什么 Llama 3.2、Gemma、Qwen3 的小模型都靠大模型"喂"出来；怎样用教师模型生成工具调用轨迹，再用执行验证筛掉错误的轨迹；以及选教师时最容易忽略的一件事：许可证。
