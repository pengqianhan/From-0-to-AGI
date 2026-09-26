# 第 16 章：SFT —— 把只会续写的底座，教成会回答、会调工具的助手

> **一句话目标**：读完这一章，你能把一段含工具调用的多轮对话手写成 ChatML 格式的整串文本，指出其中哪些 token 要算 loss；能写出"只在助手 token 上算 loss"的训练代码并解释 `<|im_end|>` 为什么必须算；能说清把多条对话打包进一个窗口省了什么、代价是什么；能为主线模型挑出许可证合适的指令数据和工具调用数据。

📺 **本章视频**：待发布（本地渲染：`bash chapters/16-sft/video/build.sh`）
🧪 **本章自检**：学完后在 Claude Code 里输入 `/ch16-sft`

---

第三部分结束时，我们手里有了主线的 Base 模型：预训练喂了它几千亿 token，中期训练用高质量数据给它收了尾，长上下文扩展让它能读 32K。按第 11 章的考卷，它能做选择题、能续写代码。但是你问它"北京今天天气怎么样？"，它不会回答——它会**接着写**：也许再写几个问题，也许写一段新闻，也许写一首诗。**这一章要解决的问题是：一个只会续写的底座，怎么变成一个会按格式回答、会在该调用工具的时候写出正确工具调用的助手？** 答案叫监督微调（Supervised Fine-Tuning，SFT）。它是第四部分"后训练"的第一步，后面的蒸馏（第 17 章）、DPO（第 18 章）、GRPO（第 19 章）都站在它上面。

SFT 本身的算法几乎没有新东西：还是第 7 章以来一直在用的 next-token 交叉熵。新东西全在"数据长什么样"：一套把对话变成文本的**对话模板**（chat template），一张决定哪些 token 算 loss 的**loss mask**，以及把长短不一的对话高效装进训练窗口的**打包**（packing）。这三样做错一个字符，模型就会在部署时出怪毛病，所以本章会把它们一个字符一个字符地拆开看。

## 1. 直觉：底座只会续写

先看底座的真实表现。第 10 章我们在莎士比亚上训练过一个 0.86M 参数的字符级小模型，把一个问题直接喂给它（`code/03_sft_tiny.py`）：

```
问题：   Is it raining in Dallas?
续写：   \nCAMILLANUS:\nI will the son the shall the prove the shall th
```

它没有"回答"的概念。预训练的目标从头到尾就是"猜下一个 token"，它见过的文本里，一个问句后面最常出现的是下一个角色的台词，那它就写台词。大模型也是一样：Base 模型见过的网页里，问句后面可能是答案，也可能是更多问句、广告、评论区。**会不会回答，不取决于模型懂不懂，而取决于它认为"接下来最可能出现什么"。**

SFT 的想法朴素到有点可笑：既然模型会模仿它见过的文本，那就给它看大量"问题后面紧跟着好回答"的文本。见得多了，"问题 → 回答"就成了它心里最可能的续写。具体说，我们准备一批对话（用户说什么、助手该怎么答、该调什么工具），按固定格式拼成文本，在这些文本上继续训练几千步。训练目标和预训练一模一样，只改了两件事：

1. 数据从"网页文本"换成"格式化的对话"；
2. loss 只在**助手说的话**上算。

写成公式，设一条对话渲染成 token 序列 `x_1 … x_T`，其中属于助手输出的位置集合是 `A`：

```
L_SFT = − 1/|A| · Σ_{t ∈ A} log p_θ(x_t | x_<t)
```

和预训练的 `−1/T · Σ_t log p_θ(x_t | x_<t)` 相比，只是求和范围从全部位置缩到了 `A`。对应 `code/02_loss_mask.py` 里的这几行：`y` 里不算 loss 的位置换成 `−100`，交叉熵的 `ignore_index` 会跳过它们，剩下的位置自动求平均：

```python
def masked_targets(ids, mask, use_mask=True):
    x = torch.tensor(ids[:-1])
    y = torch.tensor(ids[1:])
    if use_mask:
        y = torch.where(torch.tensor(mask[1:]), y, torch.full_like(y, -100))
    return x, y
# loss = F.cross_entropy(logits, y, ignore_index=-100)
```

注意是 `mask[1:]`：第 t 个位置预测的是第 t+1 个 token，要不要算 loss，看的是**被预测的那个 token** 是不是助手的。这是 SFT 代码里最常见的差一错误。

## 2. 对话模板：把一段对话变成一串文本

模型只认识一串 token。一段对话有多个角色、多轮、还有工具调用，得先约定一种格式把它压成一维文本。这个约定叫**对话模板**（chat template）。它必须在训练和推理时**逐字一致**：训练时用什么格式，部署时推理框架就得拼出同样的格式，差一个换行，模型看到的就是分布外的输入。

### 2.1 ChatML：角色标记

今天开源模型里最常见的格式源自 OpenAI 2023 年公开的 ChatML（Chat Markup Language）：每条消息用两个特殊 token 包起来，

```
<|im_start|>角色
内容<|im_end|>
```

角色是 `system`、`user`、`assistant` 之一。第 7 章讲过，`<|im_start|>` 和 `<|im_end|>` 是**特殊 token**：在词表里各占一个 id，BPE 不会把它们拆开，也不会被普通文本意外拼出来。zero 的分词器把它们固定在 id 1 和 2。

我们用 Hugging Face 上的原始 `tokenizer_config.json` / `chat_template.jinja` 逐一核对过（2026-09 读取）：Qwen3、Qwen3.5、SmolLM3、Hermes 3、OLMo 3 的 Instruct 模型都用这套角色标记。它们在别的细节上各有不同，这一点 2.4 节会讲。

### 2.2 工具调用：三个约定

工具调用在 ChatML 上加了三个约定（源自 Nous Research 的 Hermes 函数调用格式，Qwen3 的官方模板也是这一套）：

1. **工具清单写在 system 里**。每个工具是一个 JSON Schema（名字、说明、参数类型），一行一个，放在 `<tools></tools>` 之间，前后有一段固定的英文说明；
2. **助手的调用写成 `<tool_call>{json}</tool_call>`**，JSON 里只有 `name` 和 `arguments` 两个字段；一次要调几个就写几段；
3. **工具的返回值放进一个 user 轮**，每条结果包在 `<tool_response></tool_response>` 里；连续几条结果合并进同一个 user 轮。

下面是一条真实训练样本：`zero/post/envs/tool_env.py` 生成的标准解答轨迹（冒烟测试的 `out/smoke/sft/train.jsonl` 第 1 行，工具清单只留了用到的 `get_weather`），`code/01_chat_template.py` 渲染出的**完整字符串**：

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

读一遍这段文本，就读懂了一个工具调用模型的"一生"：系统提示告诉它有哪些工具、调用格式是什么；用户提问；它输出两段 `<tool_call>`；推理框架（如 vLLM 的 `hermes` 解析器）把这两段 JSON 解析出来，真的去调用天气 API；结果包成 `<tool_response>` 塞回对话；它再读结果、用一句话回答。

推理时，提示词渲染到最后一条用户消息为止，再加上 `<|im_start|>assistant\n` 作为"生成提示"（generation prompt），模型从这里接着写，写到 `<|im_end|>` 就停：

```
推理时提示词的最后 40 个字符： '今天哪个更热？<|im_end|>\n<|im_start|>assistant\n'
```

手写的渲染函数 `render()` 大约 40 行，核心是这一段（`code/01_chat_template.py`）：

```python
elif role == "assistant":
    segs.append((f"{IM_START}assistant\n", "assistant", False))  # 角色头：提示词给的，不学
    text = m.get("content") or ""
    for j, tc in enumerate(m.get("tool_calls") or []):
        if text or j > 0:
            text += "\n"
        text += ('<tool_call>\n{"name": ' + tojson(tc["name"]) + ', "arguments": '
                 + tojson(tc["arguments"]) + "}\n</tool_call>")
    segs.append((text + IM_END, "assistant", True))  # ← 只有这一段算 loss（含 <|im_end|>）
```

它返回的不只是文本，而是一段段 `(文本, 角色, 是否算 loss)`。脚本最后和生产级的 `zero.post.chat.render_text` 对拍：完整对话和生成提示两种情况都**逐字一致**。

### 2.3 模板要"一字不差"

有两个事实说明格式细节不是小事：

- Tülu 3 技术报告（表 13）在同一份中间 SFT 数据上只改模板：把助手消息末尾的换行换成 eos、去掉换行、换成 Zephyr 或 Llama 3 模板，平均分在 51.6 到 53.0 之间变动。只是几个字符的差别。
- zero 的模板用 Python（训练时渲染、顺带算 mask）和 Jinja（导出到 `tokenizer_config.json`，给 transformers / vLLM / llama.cpp 用）写了两份，`tests/test_chat.py` 用 transformers 的 Jinja 引擎逐字对拍 12 种组合；冒烟测试导出 Hugging Face 格式后再次确认模板一致（`hf_template_identical=True`）。

### 2.4 各家的差别：角色标记一致，调用语法不一

核对官方模板时我们也看到了分歧，写在这里免得你以为"大家都一样"：

| 模型 | 角色标记 | 工具清单 | 调用语法 | 工具返回 |
|---|---|---|---|---|
| Qwen3 | ChatML | system 里 `<tools>` | `<tool_call>` 包 JSON（`name`、`arguments`） | user 轮里的 `<tool_response>` |
| Hermes 3（Llama 3.1 底座） | ChatML | system 里 `<tools>` | `<tool_call>` 包 JSON（`arguments` 在前） | 独立的 `tool` 角色 + `<tool_response>` |
| SmolLM3 | ChatML | system 里 `<tools>`（`xml_tools`） | `<tool_call>` 包 JSON | 放进 user 轮 |
| Qwen3.5 | ChatML | system 里 `<tools>` | `<tool_call>` 里是 XML：`<function=名字><parameter=参数>值</parameter></function>` | user 轮里的 `<tool_response>` |
| OLMo 3 Instruct | ChatML | system 里 `<functions>` | `<function_calls>` 里是 Python 调用 `name(a=1)` | 独立的 `environment` 角色 |

角色标记（ChatML）是共识；工具调用的外壳多数用 `<tool_call>`；里面写 JSON 还是 XML 参数、工具结果放哪个角色，各家不同。主线模型选 Qwen3 / Hermes 风格的 JSON，理由是 vLLM 的 `hermes` 解析器、llama.cpp 都直接支持，第 20 章导出后不用写任何适配代码。

zero 的模板与 Qwen3 只差一处：Qwen3 在"关闭思考"时会在生成提示后面塞一个空的 `<think>\n\n</think>\n\n`，我们不塞，因为主线模型从没见过思考格式。

## 3. loss mask：只学助手说的话

### 3.1 哪些 token 算

`01_chat_template.py` 逐段打印出每一段算不算 loss：

```
  不算  system    '<|im_start|>system\n你是一个会使用工具的助手。…
  不算  user      '<|im_start|>user\n成都和广州今天哪个更热？<|im_end|>\n'
  不算  assistant '<|im_start|>assistant\n'
★ 算    assistant '<tool_call>\n{"name": "get_weather", "arguments": {"city": "成都"}}\n</…
  不算  assistant '\n'
  不算  tool      '<|im_start|>user\n<tool_response>\n{"city": "成都", …
  不算  tool      '\n<tool_response>\n{"city": "广州", …
  不算  assistant '<|im_start|>assistant\n'
★ 算    assistant '广州更热（成都 21°C，广州 31°C）。<|im_end|>'
  不算  assistant '\n'

总字符 1126，其中算 loss 的 197（17.5%）
各角色字符数： {'system': 622, 'user': 40, 'assistant': 243, 'tool': 221}
```

规则只有三条：

1. **助手的输出算**，包括工具调用的每一个字符；
2. **助手末尾的 `<|im_end|>` 也算**。模型要学会"什么时候停"：少了它，推理时模型会一直写下去，写完回答接着替用户编下一轮；
3. **其余都不算**：system、user、工具返回，以及 `<|im_start|>assistant\n` 这个角色头（推理时它是提示词的一部分，不需要模型生成）。

### 3.2 为什么不算用户的话

直觉上"多学一点总没坏处"，为什么要把 80% 以上的文本扔掉？

- **目标不同**。我们要的是一个"给定上下文，写出好回复"的模型，不是一个"模仿用户提问"的模型。用户的话、工具的返回，是模型在推理时**读**的，不是它要**写**的。
- **梯度预算**。这条样本里 system 提示占了 622 个字符，助手只有 197 个。如果全部都算，每一步的梯度大半花在"背诵那段每条样本都一样的工具说明"上。冒烟测试的全部 1,500 条训练对话用 zero 的分词器编码后共 502,756 个 token，助手 token 只有 67,191 个，**13.4%**。
- **工具返回不该学**。模型如果学会"生成" `<tool_response>` 里的内容，就是在学编造 API 的返回值。第 19 章的奖励函数会专门检查"伪造工具结果"（`tool_env.py` 的 `_forged`），SFT 阶段就不该教它这个。

Llama 2 的论文写得很直白："we zero-out the loss on tokens from the user prompt, so as a result, we backpropagate only on answer tokens"。Tülu 3 的训练代码 open-instruct、Hugging Face TRL 的 `assistant_only_loss`、nanochat 的 SFT 脚本、CS336 作业 5 的 `response_mask` 都是同一个做法。

### 3.3 loss 怎么平均

`L_SFT` 里除以的是 `|A|`：一个 batch 里所有助手 token 一起平均。`02_loss_mask.py` 演示了一个容易踩的坑：样本 A 的助手只有 3 个 token（每个 loss 2.0），样本 B 有 30 个（每个 1.0）：

```
按 token 平均：1.091；先各自平均再按样本平均：1.500
```

梯度累积时，如果每个 micro-batch 各自按"本批助手 token 数"求平均，再在 micro-batch 之间平均，就会从前者悄悄变成后者，短回复的权重被放大。Tülu 3 报告（4.3.2 节）专门写了这个问题，他们改用求和 loss（sum loss）配合调整学习率。zero 目前按常见做法实现（`zero/post/sft.py` 开头的注释写明了这一点），第二步如果发现短回复权重过大，可以改成跨 micro-batch 按 token 数加权。

## 4. 小实验：把小底座 SFT 成"会发工具调用"的模型

`code/03_sft_tiny.py` 把上面几件事串起来做一遍。任务是一个玩具工具调用：两个工具 `get_weather(city)` 和 `add(a, b)`，问题从模板里随机生成（"Is it raining in {c}?"、"What is {a} plus {b}?"……），标准回复是一次工具调用。2,000 条训练对话用 30 个城市；测试用 **10 个训练里从没出现过的城市**，数字全是随机的。这样测的是"照抄"能力，而不是背城市名。

### 4.1 扩词表

底座的词表只有莎士比亚里的 65 个字符，没有 `{ } " _`、没有大部分数字，更没有 `<|im_start|>`。所以第一步是扩词表：追加 20 个新字符和 4 个特殊 token（65 → 89），新行的 embedding 用旧行的平均值加一点噪声初始化，这样初始时新 token 的 logits 不会离谱。

```
一条玩具样本（[方括号] 里是算 loss 的 token）：
<|im_start|>system
Tools: get_weather(city), add(a, b)<|im_end|>
<|im_start|>user
What is 97 plus 53?<|im_end|>
<|im_start|>assistant
[<tool_call>
{"name": "add", "arguments": {"a": 97, "b": 53}}
</tool_call><|im_end|>]
共 137 个 token，算 loss 的 53 个（39%）
```

这也是为什么 zero 的分词器在第 7 章就把 `<|im_start|>`、`<tool_call>` 等特殊 token 预留在 id 0–15：预训练时它们就在词表里，SFT 时不用改模型形状。

### 4.2 结果

全参数微调，AdamW，学习率 1e-3（小模型；大模型要小得多，见第 6 节），warmup 20 步后余弦衰减到 10%，batch 8，300 步（约 1.2 个 epoch），不做权重衰减。底座与 SFT 后在同一道题上：

```
底座套上对话模板后的"回答"：'The shall the then the shally the the the the the the the th'

SFT 之后（有 mask）：
  'What is 7 plus 11?'               → <tool_call>\n{"name": "add", "arguments": {"a": 1, "b": 12}}\n</tool_call>
  'Is it raining in Dallas?'         → <tool_call>\n{"name": "get_weather", "arguments": {"city": "Denver"}}\n</tool_call>
  'What is the weather in Nairobi?'  → <tool_call>\n{"name": "get_weather", "arguments": {"city": "Denver"}}\n</tool_call>
  'Compute 20+55.'                   → <tool_call>\n{"name": "add", "arguments": {"a": 19, "b": 15}}\n</tool_call>
```

50 道测试题上的统计（贪心生成，遇到 `<|im_end|>` 停）：

| 设置 | 格式对 | 函数名对 | 参数全对 | 验证 loss（只算助手 token） |
|---|---:|---:|---:|---:|
| 底座（扩词表后，未 SFT；20 道题） | 0.00 | 0.00 | 0.00 | 4.640 |
| 有 mask，2,000 条 | 1.00 | 1.00 | 0.00 | 0.102 |
| 无 mask，2,000 条 | 1.00 | 1.00 | 0.00 | 0.110 |
| 有 mask，只有 40 条（约 60 个 epoch） | 1.00 | 1.00 | 0.00 | 0.366 |

把有 mask 模型的验证 loss 按 token 类型拆开：

```
参数值（城市名、数字）310 个 token，平均 loss 1.146
其余（格式、函数名）3364 个 token，平均 loss 0.006
```

换成训练里见过的城市（数字仍随机），参数全对也只有 0.04。

三点观察：

1. **SFT 教会的首先是格式**。300 步之后，JSON 结构、标签、函数名、在哪里停，几乎零误差（loss 0.006）；函数名的选择也全对，模型读懂了"天气"和"加法"两类问题。
2. **最难的是参数**。参数要从用户问题里**照抄**，这要求注意力学会"回头找到那个城市名并逐字复制"。这个 0.86M 的小模型 300 步没学会，它的策略是写一个训练里常见的城市（Denver）、一个看起来像数字的数字。loss 几乎全集中在这 310 个参数 token 上。这和下面"主线进度"里 zero 冒烟测试的 tiny 模型表现**一模一样**：格式全对、函数名基本对、参数抄错。这正是主线模型要靠更大的模型、更多更杂的数据（第 17 章蒸馏）、再加上以"调用是否真的执行正确"为奖励的强化学习（第 19 章）来补的地方。
3. **有没有 mask，在这个玩具上差别很小**（验证 loss 0.102 对 0.110）。这里的用户问题很短、很有规律，无 mask 版"顺便"学背问题并不费事。真实数据里，system 提示和工具返回动辄占 80% 以上的 token，差别会大得多。我们如实报告这个小结果，而不把它说成"mask 立竿见影"。

另外，mask 还决定了"训练 loss 这个数字是什么意思"：同一个底座在同一批验证数据上，只算助手 token 是 4.640，全部 token 都算是 4.160。不说清楚算的是哪些 token，两个 loss 就没法比。

### 4.3 过拟合

最后一行只给 40 条数据、训同样多的步数（约 60 个 epoch）：

```
step    1  训练 loss 4.561  验证 loss 4.537
step  100  训练 loss 0.086  验证 loss 0.316
step  200  训练 loss 0.046  验证 loss 0.343
step  300  训练 loss 0.055  验证 loss 0.366
```

训练 loss 一路降到 0.05，验证 loss 在第 100 步之后开始**回升**：模型在背那 40 条。SFT 的数据量通常比预训练小几个数量级，过拟合是常态风险。对策是：更多、更多样的数据；少训几个 epoch；按留出的验证集挑 checkpoint，而不是按训练 loss。

## 5. 打包：把多条对话装进一个窗口

### 5.1 为什么要打包

对话长短不一。最简单的做法是一条对话一行，补齐（pad）到窗口长度。补齐的位置不算 loss，但照样要算前向和反向，全是浪费。`code/04_packing.py` 造了 200 条长 130–390 token 的对话，窗口长 513（seq_len 512 + 1）：

```
不打包（一条一行，补齐到 513）：200 行，真实 token 占 43%
首次适配打包：94 个窗口，真实 token 占 91%，每个窗口平均装 2.1 条
```

**首次适配**（first-fit）：按顺序处理每条对话，放进第一个还装得下的窗口，都装不下就开新窗口。有一条原则不能破：**一条对话不切开**。切开的话，后半段的助手回复看不到前半段的问题，等于在教模型"凭空回答"。放不下的超长对话直接丢弃并计数。Llama 2 的 SFT 也是把样本拼接起来填满 4096 的窗口；nanochat 和 TRL 用的是装箱更紧的最佳适配（best-fit）。

### 5.2 代价：串门

打包之后，同一窗口里第二条对话的 token，在普通因果注意力下能**看到**第一条对话。`04_packing.py` 用 SFT 过的小模型量了一下：把对话 A（问 Paris）和对话 B（问 Kyoto）装进同一窗口，看 B 的输出：

```
对话 B（Kyoto）的助手 token loss：
  单独一条                              0.5596
  打包在 A（Paris）后面，普通因果 mask     0.4967   logits 最大差 9.90e+00
  打包在 A 后面，文档 mask                0.5596   logits 最大差 4.92e-05
```

普通因果 mask 下，B 的 logits 最多变了 9.9，loss 反而**降**了：A 就是一条格式相同的工具调用，B"抄了邻居的作业"。这就是串门（cross-contamination）：训练时模型能借助上下文里无关的对话，推理时却没有这个邻居，训练和推理的条件不完全一样。

解决办法叫**文档 mask**（document masking）：把因果 mask 换成块对角的下三角，每条对话只能看自己：

```python
def document_mask(doc_ids):
    """因果 + 只能看同一条对话：块对角的下三角。"""
    return causal(len(doc_ids)) & (doc_ids[:, None] == doc_ids[None, :])
```

加上它以后，B 的 loss 与单独计算完全一样（logits 差 5e-5，是浮点误差）。一个细节：B 在窗口里是从第 154 个位置开始的，但 RoPE 只看相对位置（第 9 章），只要注意力被隔开，绝对位置的偏移不影响结果。

### 5.3 zero 的选择

zero 的 SFT **目前不隔离**打包在一起的对话：它用 PyTorch 的 SDPA 加普通因果 mask，与 nanochat 的 SFT 一样。这是一个明确的取舍：

- **好处**：实现简单，能直接用最快的因果注意力 kernel；而 SFT 数据里每条对话都以 `<|im_start|>system` 开头，模型很容易学会"新对话从这里开始"。冒烟测试里一条工具对话约 335 个 token，512 的窗口平均只装得下 1.06 条（1,500 条装进 1,415 个窗口），串门本来就很少；
- **代价**：主线 SFT 的窗口是 8192，短对话会被大量打包在一起，串门变多；
- **第二步的选项**：用 FlashAttention 的变长（varlen）接口，传入每条对话的边界，在不浪费算力的前提下隔离注意力（TRL 的文档写明它的打包只用于配合 FlashAttention 的设置）。zero 的这条路径**尚未实现、尚未在 GPU 上验证**，是否值得做，先用小实验比较"隔离 vs 不隔离"对工具调用得分的影响再定。

## 6. 数据：从哪来，许可证是什么

SFT 的效果主要由数据决定。Llama 2 的报告说他们放下了数百万条第三方数据，只用了 27,540 条自己标注的高质量样本，结果明显更好（"Quality Is All You Need"）；LIMA 的标题干脆叫"Less Is More for Alignment"。但对小模型和工具调用这个具体能力，量仍然重要：Tülu 3 的消融显示，SFT 数据从 5% 加到 100%，平均分一直在涨（图 4）。

### 6.1 通用指令数据

| 数据集 | 规模 | 许可证（数据卡原文） | 备注 |
|---|---|---|---|
| [Tülu 3 SFT mixture](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture) | 939,344 条 | 整体 ODC-BY-1.0，**子集各有许可证**，"Some portions of the dataset are non-commercial" | 例如 No Robots 是 CC-BY-NC-4.0；还含第三方模型生成的输出，受各自条款约束 |
| [SmolTalk](https://huggingface.co/datasets/HuggingFaceTB/smoltalk) | 约 1M 条 | 新造的四个子集 Apache-2.0；混入的公开数据集"refer to the original dataset" | SmolLM2-Instruct 用它训练；另有面向 135M/360M 小模型的 [smol-smoltalk](https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk) |
| [SmolTalk2](https://huggingface.co/datasets/HuggingFaceTB/smoltalk2) | Mid / SFT / Preference 三部分 | 待核实（数据卡元数据未标许可证） | SmolLM3 后训练三阶段的数据 |

规矩（GOAL.md 3.3）：**逐个子集记录来源和许可证**，非商用的子集不进主线；由闭源模型生成的数据，还要看生成模型的使用条款是否允许"用输出训练别的模型"，这一项在数据卡里常常写不清，一律记"待核实"再逐项查。

### 6.2 工具调用数据

| 数据集 | 规模 | 许可证 | 生成方式 |
|---|---|---|---|
| [xLAM function-calling 60k](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k)（APIGen） | 60,000 条 | CC-BY-4.0（数据集需申请访问） | APIGen 流水线合成，经三级验证：格式检查、**真实执行**、语义检查 |
| [ToolACE](https://huggingface.co/datasets/Team-ACE/ToolACE) | 1 万–10 万条 | Apache-2.0 | 自演化合成 26,507 个 API，多智能体生成对话，规则 + 模型双层验证；中英双语 |
| [Hermes Function-Calling V1](https://huggingface.co/datasets/NousResearch/hermes-function-calling-v1) | 5 个子集 | Apache-2.0 | Hermes 2 Pro 的函数调用与 JSON 模式数据，格式就是本章的 `<tool_call>` |

SmolTalk 本身也混入了 8 万条 APIGen 函数调用数据（Synth-APIGen + xLAM）。三份数据都是**合成**的，生成用的模型及其条款要逐一核实（待核实）。

### 6.3 自己合成：可执行、可验证

主线的另一路数据来自自己的工具环境。`zero/post/envs/tool_env.py` 有 6 个模拟 API（计算器、天气、单位换算、日期加减、日期间隔、星期几），`reference_messages(task)` 生成"用户 → 助手调用 → 工具结果 → 助手回答"的标准解答轨迹，本章 2.2 节那条样本就是它生成的。这类数据的好处是每一条调用都能**真的执行**，结果对不对可以自动判定，正是 xLAM 强调的"执行验证"思路。第 17 章会用更大的教师模型在更多工具上生成轨迹，再用同样的执行验证筛掉错误样本。

### 6.4 去污染

GOAL.md 3.2 要求所有训练数据与评测集做 13-gram 重叠检查；工具调用数据还要检查**函数名和 schema**有没有和 BFCL 等评测集重合。公开的函数调用数据集常常覆盖同一批流行 API，和 BFCL 撞车的风险比通用文本高得多。Tülu 3 的附录 B.2 也列出了多份公开数据集与常用评测的重叠。

## 7. 超参数：小学习率、少 epoch

SFT 是在一个已经很好的模型上"轻轻推一下"，所以学习率比预训练小得多，epoch 很少：

| 来源 | 模型 | 学习率 | epoch | 其他 |
|---|---|---|---|---|
| Llama 2 论文 | 7B–70B | 2e-5（余弦） | 2 | 权重衰减 0.1，batch 64，序列 4096 |
| Tülu 3 报告 表 11 | Llama 3.1 8B / 70B | 5e-6 / 2e-6（线性） | 2 | 有效 batch 128，最大长度 4096，warmup 3%；报告说训更久没有继续变好 |
| SmolTalk 数据卡 | SmolLM2（1.7B）对比实验 | 3e-4 | 2 | 序列 8192，全局 batch 16 |
| zero `configs/main/sft.toml` | 主线 0.69B | 5e-5（待调） | ≈2（待定） | 见下一节 |

两个规律：模型越大，学习率越小（Tülu 3 从 8B 到 70B 把学习率从 5e-6 降到 2e-6，405B 再降，报告称之为更轻的一推，"lighter touch"）；epoch 数多在 1–3。Tülu 3 还发现，只换随机种子，8B SFT 模型的平均分就能在 59.8–60.1 之间变动，所以别把 0.2 分的差别当成超参数的功劳。

太大的学习率或太多的 epoch 有两种后果：一是 4.3 节那样的过拟合；二是**遗忘**预训练学到的通用能力。第 11 章的通用基准在 SFT 前后都要测，确认没有明显掉分。

### 7.1 LoRA：一笔带过

LoRA（Low-Rank Adaptation）冻结原模型的权重，只在每个线性层旁边训练一对低秩矩阵 `ΔW = B·A`（秩通常 8–64），可训练参数少几个数量级，显存需求大减，一张消费级显卡就能微调 7B 模型，是社区微调的事实标准（Hugging Face PEFT、TRL 都内置；TRL 文档建议 LoRA 的学习率约 1e-4，比全参数微调高一个量级）。

主线模型**不用 LoRA，做全参数微调**，理由很实际：模型只有 0.69B，全参数的优化器状态在 8 张 H100 上毫无压力；我们的目标是把工具调用能力推到极限，没必要为了省显存接受低秩约束可能带来的上限损失。读者想在自己的笔记本上用我们发布的 Base 模型做实验时，LoRA 是很好的选择。

## 8. 小结

- 底座只会续写。SFT 用同一个 next-token 交叉熵，在格式化的对话上继续训练，把"问题 → 好回答"变成模型心里最可能的续写。
- 对话模板把多角色、多轮、工具调用压成一串文本。ChatML 的 `<|im_start|>角色\n…<|im_end|>` 是共识；工具清单写在 system 里，调用写成 `<tool_call>{json}</tool_call>`，结果包在 `<tool_response>` 里。训练和推理必须逐字一致。
- loss mask 只让助手输出（含工具调用、含 `<|im_end|>`）算 loss；真实工具对话里这只占 13–18% 的 token。
- 打包把多条对话装进一个窗口，真实 token 占比从 43% 提到 91%；代价是同一窗口的对话会串门，文档 mask 可以消除它。zero 目前不隔离，这是有意的取舍。
- 数据决定上限：逐子集核实许可证，工具调用数据优先选"可执行验证"的；小学习率、1–3 个 epoch，按验证集挑 checkpoint。
- 小实验和冒烟测试给出同一个信号：SFT 很快教会格式和函数名，最难的是从问题里照抄正确的参数。

---

## 从极简到生产级

| 极简代码 | 生产级（`zero/`） | 多做了什么、为什么 |
|---|---|---|
| `01_chat_template.py` 的 `render()`：约 40 行，返回 (文本, 角色, 算不算 loss) | `zero/post/chat.py` 的 `render_segments` / `render` / `render_text` | 支持 OpenAI 风格的 `{"type": "function", "function": {...}}` 与字符串形式的 `arguments`、助手消息同时有文字和调用、思考内容（`enable_thinking`，默认关）；给了分词器时**整段一次编码**，再用每个 token 的字符起点查逐字符 mask，而不是分段编码再拼接（段边界处 BPE 的合并结果可能不同，与推理框架"拼好整串再编码"才能一致） |
| （没有） | `zero/post/chat.py` 的 `CHAT_TEMPLATE`（Jinja） | 同一格式的第二份实现，导出时写进 `tokenizer_config.json`，让 transformers / vLLM / llama.cpp 拼出逐字相同的提示词。`tests/test_chat.py` 用 transformers 的 Jinja 引擎对拍 3 种对话 × 有无工具 × 有无生成提示 |
| `01` 的 `parse_tool_calls()`：一个正则 + `json.loads` | `zero/post/chat.py` 的 `parse_assistant` → `ParsedAssistant(content, tool_calls, reasoning_content, errors)` | 模型输出什么都可能：标签不配对、JSON 坏了、多余字段、`arguments` 写成字符串……不抛异常，把错误记在 `errors` 里，第 19 章的奖励函数据此扣分；`test_render_parse_roundtrip` 保证渲染与解析互逆 |
| `02_loss_mask.py` 的 `masked_targets()`：mask 为 0 的目标换成 -100 | `zero/data/loader.py` 的 `MaskedWindowLoader.next_batch()` | 同样是 `np.where(mask[:, 1:], tok[:, 1:], -100)`；另外支持多卡分片（第 g 个样本给 rank `g % world_size`）、按 (seed, epoch) 确定的打乱、`state_dict` 精确续训。`test_masked_loss_matches_hand_computation` 与手算对拍，`test_window_loader_targets_and_resume` 测续训 |
| `03_sft_tiny.py` 的 `batchify()`：右侧补齐 | `zero/post/sft.py` 的 `encode_example` + `pack_examples` | 每条对话末尾加一个 `<|endoftext|>`（mask=0）作分隔；首次适配打包（只回看最近 64 个没满的窗口，剩余不到 16 个位置的窗口关闭）；超长样本、没有助手 token 的样本丢弃并计数；写成 `train.bin`（uint32）+ `train.mask`（uint8）+ 元数据（分词器哈希、seq_len），换了分词器或长度会拒绝旧文件 |
| `04_packing.py` 的 `pack_first_fit()` + `document_mask()` | `pack_examples`；**没有**文档 mask | 见 5.3 节的取舍；`test_pack_examples_keeps_conversations_whole` 保证对话不被切开 |
| `03` 的训练循环：几十行 | `zero/post/sft.py` 的 `run_sft` → 复用预训练的 `Trainer`（`[data] format = "sft"`） | BF16、梯度累积、DDP、断点续训、日志、checkpoint 全部复用，`[train] init_from` 指向长上下文阶段的 checkpoint；`test_run_sft_end_to_end_and_resume` 端到端训练并验证续训 |
| `03` 用第 10 章的底座并临时扩词表 | 特殊 token 在第 7 章的分词器里就预留在 id 0–15 | 预训练时它们就在词表里，SFT 不改模型形状，也不需要"新 embedding 怎么初始化"的问题 |
| `03` 手写 2,000 条玩具样本 | `zero/post/sft.py` 的 `env_conversations` + `tool_env.reference_messages` | 配置 `generate_train = N` 且数据文件不存在时，用工具环境生成标准解答轨迹；train / dev 互不重叠 |

本机运行 `uv run pytest tests/test_chat.py tests/test_sft.py`：27 项全部通过（7.2 秒）。

### `configs/main/sft.toml` 逐项说明

| 配置 | 值 | 为什么 |
|---|---|---|
| `[model]` | 与 `configs/main/longctx.toml` 相同（689.5M，`rope_theta = 1e6`，`max_seq_len = 32768`） | SFT 从长上下文阶段的 checkpoint 接着训，形状必须一致 |
| `init_from` | `out/main/longctx/ckpt` | 第 15 章的产物 |
| `seq_len` | 8192 | 工具说明 + 多轮对话 + 工具返回，一条就可能上千 token；更长的样本丢弃并计数 |
| `micro_batch_size × grad_accum × 8 卡` | 4 × 2 × 8 | 每步 64 个窗口 × 8192 = 524,288 token（打包后约 70% 是真实 token） |
| `max_steps` | 2000 | ≈ 1.05B token，假设 SFT 数据约 0.5B token 时约 2 个 epoch（待定） |
| `lr` | 5e-5，余弦，warmup 100，最低 10% | 小模型 SFT 常见 1e-5 ~ 1e-4（待阶梯实验调） |
| `weight_decay` | 0 | 步数少，不需要 |
| `grad_clip` | 1.0 | 与预训练一致 |
| `eval_every` / `eval_batches` | 200 / 20 | 按验证集挑 checkpoint（4.3 节） |

成本：`uv run python -m zero.tools.estimate_cost --config configs/main/pretrain.toml --tokens 1B --seq-len 8192` → 6.9 GPU·h、约 $17（H100、MFU 0.4、$2.5/卡时的假设，**尚未在 GPU 上验证**）。SFT 在整个预算里很便宜，贵的是第 17 章用教师模型生成数据。

---

## 主线进度

### 极小配置演示（CPU，`python -m zero.smoke`，约 1.3M 参数）

> 以下是**极小配置演示**：只说明生产级代码通路是通的，不代表主线模型的任何结果。

冒烟测试的 SFT 阶段从 tiny 中期训练的 checkpoint 出发，用 `tool_env` 生成的 1,500 条工具调用对话训练（`configs/tiny/sft.toml`，冒烟测试把 seq_len 改为 512、micro batch 8、240 步）。数字来自 `out/smoke/summary.json`、`out/smoke/sft/data/train.json` 和 `out/smoke/eval/results.json`，用 `code/05_smoke_samples.py` 可以重新打印：

| 指标 | 值 |
|---|---|
| 训练对话 / 打包后窗口 | 1,500 条 → 1,415 个窗口（seq_len 512，填充率 69%），没有丢弃 |
| 算 loss 的 token | 67,191 个（全部 token 的 13.4%） |
| 训练 240 步后 | 训练 loss 0.560，验证 loss 0.620，用时 448 秒（CPU 单线程） |
| dev 集工具调用（30 题） | 格式正确 1.00，调用完全正确 0.133，参数 AST 一致 0.10，最终回答正确 0.167 |

模型写出来的是什么样（`05_smoke_samples.py`，dev 集前 6 题，贪心生成）：

```
问： 深圳和巴黎今天哪个更热？
  标准： get_weather(city=深圳), get_weather(city=巴黎)
  模型： <tool_call>{"name": "get_weather", "arguments": {"city": "东京"}}</tool_call> ×2
问： Convert 68 F to C.
  标准： convert_units(value=68, from_unit=F, to_unit=C)
  模型： <tool_call>{"name": "convert_units", "arguments": {"value": 14, "from_unit": "C", "to_unit": "F"}}</tool_call>
问： 2027-07-20 是星期几？
  标准： weekday(date=2027-07-20)
  模型： <tool_call>{"name": "weekday", "arguments": {"date": "2023-01-11"}}</tool_call>
```

与第 4 节的小实验一样：**格式完全正确、函数名基本正确、参数几乎都是编的**（一个训练里常见的城市、一个常见的日期）。1.3M 参数的模型没有学会从问题里照抄参数。30 题里调用完全正确的 4 题，有 3 题是本来就不需要调用工具的问题。

### 待 GPU 训练后补充

- 主线 SFT 的数据配方：通用指令数据、工具调用数据、自合成轨迹各占多少，每个子集的来源与许可证清单，BFCL 函数名 / schema 去污染的结果；
- 训练曲线、学习率与 epoch 的小规模扫描、按验证集挑出的 checkpoint；
- SFT 前后在第 11 章通用基准上的对比（确认没有明显遗忘），工具调用开发集上的得分；
- "打包是否隔离"的对比实验（5.3 节）；
- 实际花费（记入 `runs/ledger.md`）、失败与返工。

---

## 前沿观察

> **工具调用的语法还没有统一**。角色标记（ChatML）和"工具写在 system 里、调用包在 `<tool_call>` 里"已经是多家共识，但标签里面写什么正在分化：Qwen3、Hermes、SmolLM3 写 JSON；Qwen3.5 改成了 XML 风格的 `<function=名字><parameter=参数>值</parameter>`（参数值可以跨多行，写代码之类的长参数时不用转义）；OLMo 3 写 Python 函数调用。哪种格式对小模型最友好，我们没有找到公开的对照实验，主线沿用 JSON，理由是推理框架支持最成熟。
>
> **SFT 打包时的文档 mask**。TRL 的打包只用于配合 FlashAttention 的设置（按对话边界隔离，待核实其实现细节），nanochat 和 zero 不隔离；核实到的明确采用方不足三家，暂不进正文，第二步用小实验决定（见 5.3 节）。

---

## 采用方与来源

**ChatML 风格的对话模板 + `<tool_call>` 工具调用格式**

- Qwen3：[`Qwen/Qwen3-0.6B` tokenizer_config.json 的 chat_template](https://huggingface.co/Qwen/Qwen3-0.6B/blob/main/tokenizer_config.json)（JSON 调用，与 zero 的模板只差关闭思考时的空 `<think>`）；[Qwen3 技术报告](https://arxiv.org/abs/2505.09388)
- Qwen3.5：[`Qwen/Qwen3.5-0.8B` chat_template.jinja](https://huggingface.co/Qwen/Qwen3.5-0.8B/blob/main/chat_template.jinja)（ChatML + `<tool_call>`，内部为 XML 参数）
- SmolLM3（Hugging Face）：[`HuggingFaceTB/SmolLM3-3B` chat_template.jinja](https://huggingface.co/HuggingFaceTB/SmolLM3-3B/blob/main/chat_template.jinja)
- Hermes 3（Nous Research，Llama 3.1 底座）：[模型卡 Prompt Format 一节](https://huggingface.co/NousResearch/Hermes-3-Llama-3.1-8B)；[技术报告](https://arxiv.org/abs/2408.11857)
- OLMo 3（Ai2）：[`allenai/Olmo-3-7B-Instruct` chat_template.jinja](https://huggingface.co/allenai/Olmo-3-7B-Instruct/blob/main/chat_template.jinja)（ChatML 角色标记，工具调用用 `<function_calls>`）
- ChatML 的原始说明：[openai-python `chatml.md`](https://github.com/openai/openai-python/blob/release-v0.28.0/chatml.md)

**只在助手 token 上算 loss（loss mask）**

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

1. 把 2.2 节那条样本里的 `<|im_end|>` 全部从 loss 里拿掉，模型推理时会出什么毛病？如果反过来把 `<|im_start|>assistant\n` 也算进 loss，会有害吗？
2. 第 4 节的小实验里"有 mask"和"无 mask"几乎打平。设计一个玩具数据，让两者的差别变得明显（提示：拉长 system 提示，或让用户消息里出现"像助手回复"的文本）。
3. 为什么 zero 的 `render` 要"整段一次编码再按字符偏移查 mask"，而不是每段分别编码再拼接？构造一个 BPE 在段边界处合并结果不同的例子。
4. Qwen3.5 把工具调用参数从 JSON 改成了 `<parameter=…>` 的 XML 形式。对于"参数是一段多行 Python 代码"的调用，两种格式各要模型生成哪些额外的字符？哪种更容易出错？
5. 你手里有 Tülu 3 SFT mixture 和 xLAM 60k 两份数据，主线模型要商用发布。按本章的规矩，列出你要逐项核实的许可证与条款问题。
6. SFT 的验证 loss 已经降到 0.1，工具调用的参数却几乎全错（第 4 节）。只看 loss 为什么会被骗？你会加什么指标来盯训练？

## 动手任务

- **基础**：运行 `01_chat_template.py`，在对话里再加一轮"用户说谢谢 → 助手回复不客气"，预测新增的哪些字符算 loss，再看打印结果验证；用 `zero.post.chat.render_text` 对拍。
- **核心**：修改 `02_loss_mask.py` 里的 `masked_targets`，故意把 `mask[1:]` 写成 `mask[:-1]`，用 `03_sft_tiny.py` 训一次，观察模型在 `<|im_end|>` 处的行为和测试得分发生了什么，解释原因。
- **挑战**：让小模型学会照抄参数。可以试：把训练步数加到 2,000 步、把城市换成更多样的随机字母串、在 `03` 里加一个"只在参数 token 上加大权重"的 loss，或者把第 10 章的模型换成更深的。记录每种办法下测试集的"参数全对"，并检查 40 条数据时的过拟合是否更严重。全部在 CPU 上完成。

## 想深入：CS336

本章对应斯坦福 CS336（Spring 2026）<https://cs336.stanford.edu/>：

- **第 15 讲：中期训练与后训练（SFT / RLHF）**。讲 SFT 的数据从哪里来、为什么"少而精"、指令数据的风格与安全问题，以及它和后面 RLHF 的衔接（讲义与录像见课程页）。
- **作业 5（Alignment）**：`run_tokenize_prompt_and_output` 要求你把提示词和回复拼起来、构造与 labels 对齐的 `response_mask`，正是本章 `masked_targets` 做的事；`get_packed_sft_dataset` 要求把指令数据打包成定长窗口；可选的补充作业涉及指令微调与 RLHF。作业仓库：<https://github.com/stanford-cs336/assignment5-alignment>

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

**下一章**：SFT 让模型学会了格式，但参数抄不对。一个办法是给它更多、更好的示范，而最好的示范来自一个更强的模型。第 17 章讲蒸馏：为什么 Llama 3.2、Gemma、Qwen3 的小模型都靠大模型"喂"出来；怎么用教师模型生成工具调用轨迹，再用执行验证把错的筛掉；以及选教师时最容易忽略的一件事——许可证。
