# Chapter 16: SFT — Teach a base model to answer questions and to call tools

**English** · [中文](README.zh.md)

> **Goal**: After this chapter, you can write a multi-turn conversation with tool calls by hand as one ChatML string. You can also point to the tokens that are in the loss. You can write training code that calculates the loss only on assistant tokens, and you can explain why `<|im_end|>` must be in the loss. You can explain what packing several conversations into one window saves and what it costs. You can also select instruction data and tool-call data with suitable licenses for the main-line model.

📺 **Video**: Not published yet. To render it on your computer, run `bash chapters/16-sft/video/build.sh`.
🧪 **Self-check**: After the chapter, type `/ch16-sft` in Claude Code.

---

At the end of Part 3, we have the base model of the main line. Pretraining gave it hundreds of billions of tokens. Mid-training finished it with high-quality data. The long-context extension lets it read 32K tokens. On the evaluation of Chapter 11, it can answer multiple-choice questions and continue code. But ask it "What is the weather in Beijing today?", and it does not answer. It **continues the text**: maybe with more questions, maybe with a news article, maybe with a poem.

**This chapter solves one problem: how does a base model that only continues text become an assistant? The assistant must answer in the correct format, and it must write a correct tool call when a tool is necessary.** The answer is supervised fine-tuning (SFT). SFT is the first step of Part 4, "Post-training". Distillation (Chapter 17), DPO (Chapter 18), and GRPO (Chapter 19) all start from it.

The SFT algorithm has almost nothing new. It is the same next-token cross-entropy that we have used since Chapter 7. All the new parts are in the form of the data. A **chat template** turns a conversation into text. A **loss mask** selects the tokens that are in the loss. **Packing** puts conversations of different lengths efficiently into training windows. One wrong character in any of the three can cause strange problems in the deployed model. So this chapter examines them character by character.

## 1. Intuition: a base model only continues text

First, look at what a base model really does. In Chapter 10, we trained a small character-level model with 0.86M parameters on Shakespeare. We give it a question directly (`code/03_sft_tiny.py`):

```
Question:      Is it raining in Dallas?
Continuation:  \nCAMILLANUS:\nI will the son the shall the prove the shall th
```

The model has no concept of an "answer". From start to end, the pretraining objective is "guess the next token". In the text that the model saw, the most common text after a question is the next line of the next character. So the model writes the next line of a play. Large models are the same. In the web pages that a base model saw, an answer can come after a question, but also more questions, an advertisement, or comments. **Whether a model answers does not depend on what the model knows. It depends on what the model thinks is most likely to come next.**

The idea of SFT is very plain. The model imitates the text that it saw. So show it a large amount of text in which a good answer comes directly after each question. After many examples, "question → answer" becomes the most likely continuation for the model. In practice, we prepare a set of conversations: what the user says, how the assistant must answer, and which tools it must call. We join each conversation into text in a fixed format, and we continue training on this text for a few thousand steps. The training objective is the same as in pretraining. Only two things change:

1. The data changes from "web text" to "formatted conversations".
2. We calculate the loss only on **what the assistant says**.

As a formula: a conversation renders to the token sequence `x_1 … x_T`, and `A` is the set of positions that belong to the assistant output:

```
L_SFT = − 1/|A| · Σ_{t ∈ A} log p_θ(x_t | x_<t)
```

Compare it with the pretraining loss `−1/T · Σ_t log p_θ(x_t | x_<t)`. The only change is the range of the sum: all positions become `A`. These lines in `code/02_loss_mask.py` do it. Each position of `y` that is not in the loss becomes `−100`. The `ignore_index` of the cross-entropy skips these positions and calculates the mean over the remaining positions:

```python
def masked_targets(ids, mask, use_mask=True):
    x = torch.tensor(ids[:-1])
    y = torch.tensor(ids[1:])
    if use_mask:
        y = torch.where(torch.tensor(mask[1:]), y, torch.full_like(y, -100))
    return x, y
# loss = F.cross_entropy(logits, y, ignore_index=-100)
```

Note `mask[1:]`. Position t predicts token t+1. So **the predicted token** decides if the position is in the loss: is that token an assistant token? This off-by-one error is the most common error in SFT code.

## 2. Chat template: turn a conversation into one string

The model reads only a sequence of tokens. A conversation has several roles, several turns, and also tool calls. So we must first agree on a format that flattens the conversation into one-dimensional text. This agreement is the **chat template**. Training and inference must use **exactly the same** template, character for character. At deployment, the inference framework must build the same format as in training. If one newline is different, the model sees an input outside its training distribution.

### 2.1 ChatML: role markers

Today, the most common format in open models comes from ChatML (Chat Markup Language), which OpenAI published in 2023. Two special tokens wrap each message:

```
<|im_start|>role
content<|im_end|>
```

The role is one of `system`, `user`, and `assistant`. Chapter 7 explained that `<|im_start|>` and `<|im_end|>` are **special tokens**. Each one has one id in the vocabulary. BPE never splits them, and normal text cannot produce them by accident. The zero tokenizer fixes them at ids 1 and 2.

We checked the original `tokenizer_config.json` / `chat_template.jinja` files on Hugging Face one by one (read in 2026-09). The Instruct models of Qwen3, Qwen3.5, SmolLM3, Hermes 3, and OLMo 3 all use these role markers. They differ in other details. Section 2.4 shows the differences.

### 2.2 Tool calls: three conventions

Tool calls add three conventions to ChatML. The conventions come from the Hermes function-calling format of Nous Research. The official Qwen3 template uses the same set:

1. **The tool list goes in the system message.** Each tool is a JSON Schema (name, description, parameter types), one per line, between `<tools></tools>`. A fixed English instruction text comes before and after the list.
2. **The assistant writes each call as `<tool_call>{json}</tool_call>`.** The JSON has only two fields: `name` and `arguments`. For several calls, the assistant writes several blocks.
3. **The tool results go into a user turn.** Each result is wrapped in `<tool_response></tool_response>`. Consecutive results go into the same user turn.

Below is a real training sample. It is a reference trajectory from `zero/post/envs/tool_env.py`: line 1 of `out/smoke/sft/train.jsonl` from the smoke test. Its tool list here keeps only `get_weather`, the tool that the sample uses. The block shows the **full string** that `code/01_chat_template.py` renders. The sample is data, so its Chinese text stays as it is. The system prompt means "You are an assistant that can use tools. Call a tool when necessary. After you get the result, answer in one sentence." The tool description means "Get the weather of a city today". The user asks "Which is hotter today, Chengdu or Guangzhou?" The weather conditions are "overcast" (阴) and "thunderstorm" (雷阵雨). The final answer means "Guangzhou is hotter (Chengdu 21°C, Guangzhou 31°C)."

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

Read this text once, and you know the full "life" of a tool-calling model. The system prompt tells the model which tools exist and what the call format is. The user asks a question. The model writes two `<tool_call>` blocks. The inference framework (for example, the `hermes` parser of vLLM) parses the two JSON objects and really calls the weather API. The framework wraps the results in `<tool_response>` and puts them back into the conversation. Then the model reads the results and answers in one sentence.

At inference time, the template renders the prompt up to the last user message. Then it adds `<|im_start|>assistant\n` as the "generation prompt". The model continues from there and stops when it writes `<|im_end|>`:

```
Last 40 characters of the prompt at inference time: '今天哪个更热？<|im_end|>\n<|im_start|>assistant\n'
```

The hand-written render function `render()` has about 40 lines. Its core is this part (`code/01_chat_template.py`):

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

The function returns more than text. It returns a list of segments `(text, role, in_loss)`. At the end, the script does a parity check with the production function `zero.post.chat.render_text`. For both the full conversation and the generation prompt, the two strings are **identical, character for character**.

### 2.3 The template must match character for character

Two facts show that format details are important:

- The Tülu 3 technical report (Table 13) changes only the template on the same intermediate SFT data. The changes: replace the newline at the end of each assistant message with eos, remove the newline, or use the Zephyr or Llama 3 template. The average score moves between 51.6 and 53.0. The difference between the templates is only a few characters.
- zero has two copies of its template. The Python copy renders at training time and also calculates the mask. The Jinja copy goes into `tokenizer_config.json` at export, for transformers / vLLM / llama.cpp. `tests/test_chat.py` uses the Jinja engine of transformers for a character-level parity check of 12 combinations. After the smoke test exports the Hugging Face format, it confirms again that the templates are identical (`hf_template_identical=True`).

### 2.4 Differences between models: the same role markers, different call syntax

When we checked the official templates, we also found differences. We list them here, so that you do not think that "all models are the same":

| Model | Role markers | Tool list | Call syntax | Tool results |
|---|---|---|---|---|
| Qwen3 | ChatML | `<tools>` in system | JSON in `<tool_call>` (`name`, `arguments`) | `<tool_response>` in a user turn |
| Hermes 3 (Llama 3.1 base) | ChatML | `<tools>` in system | JSON in `<tool_call>` (`arguments` first) | Separate `tool` role + `<tool_response>` |
| SmolLM3 | ChatML | `<tools>` in system (`xml_tools`) | JSON in `<tool_call>` | In a user turn |
| Qwen3.5 | ChatML | `<tools>` in system | XML in `<tool_call>`: `<function=name><parameter=param>value</parameter></function>` | `<tool_response>` in a user turn |
| OLMo 3 Instruct | ChatML | `<functions>` in system | Python call `name(a=1)` in `<function_calls>` | Separate `environment` role |

The role markers (ChatML) are the consensus. Most models wrap the tool call in `<tool_call>`. The models differ in the content of the call (JSON or XML arguments) and in the role that holds the tool results. The main-line model uses the JSON style of Qwen3 / Hermes. The reason: the `hermes` parser of vLLM and llama.cpp support this style directly. So after the export in Chapter 20, we do not need any adapter code.

The zero template differs from Qwen3 in only one place. When "thinking" is off, Qwen3 adds an empty `<think>\n\n</think>\n\n` after the generation prompt. We do not add it, because the main-line model never saw the thinking format.

## 3. Loss mask: learn only what the assistant says

### 3.1 Which tokens are in the loss

`01_chat_template.py` prints each segment and shows if it is in the loss:

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

There are only three rules:

1. **The assistant output is in the loss**, with each character of the tool calls.
2. **The `<|im_end|>` at the end of each assistant message is also in the loss.** The model must learn "when to stop". Without it, the model does not stop at inference time. After the answer, it continues and invents the next user turn.
3. **Nothing else is in the loss**: system, user, tool results, and the role header `<|im_start|>assistant\n`. At inference time, the role header is part of the prompt, so the model does not need to generate it.

### 3.2 Why the user text is not in the loss

Intuition says that "more learning cannot hurt". So why do we discard more than 80% of the text?

- **The objective is different.** We want a model that "writes a good reply for a given context". We do not want a model that "imitates the questions of users". At inference time, the model **reads** the user text and the tool results. It does not **write** them.
- **Gradient budget.** In this sample, the system prompt has 622 characters, and the assistant has only 197. If all text is in the loss, most of the gradient at each step goes to "memorizing the same tool instructions in every sample". We encoded all 1,500 training conversations of the smoke test with the zero tokenizer. They had 502,756 tokens, and only 67,191 of them were assistant tokens: **13.4%**.
- **The model must not learn tool results.** If the model learns to "generate" the content of `<tool_response>`, it learns to invent API results. In Chapter 19, the reward function checks for "forged tool results" (`_forged` in `tool_env.py`). The SFT stage must not teach this behavior.

The Llama 2 paper says it directly: "we zero-out the loss on tokens from the user prompt, so as a result, we backpropagate only on answer tokens". Many projects do the same. Examples: open-instruct (the training code of Tülu 3), `assistant_only_loss` in Hugging Face TRL, the SFT script of nanochat, and `response_mask` in CS336 Assignment 5.

### 3.3 How to average the loss

`L_SFT` divides by `|A|`: all assistant tokens in a batch share one mean. `02_loss_mask.py` shows an easy mistake. Sample A has only 3 assistant tokens (loss 2.0 each), and sample B has 30 (loss 1.0 each):

```
Mean per token: 1.091; mean of each sample, then mean over samples: 1.500
```

Gradient accumulation can cause this mistake. Each micro-batch takes its own mean over "its own assistant tokens", and then the code averages the micro-batches. Then the first value silently becomes the second, and short replies get a larger weight. The Tülu 3 report (Section 4.3.2) discusses this problem. Its authors changed to a sum loss and adjusted the learning rate. zero now uses the common method (the comment at the top of `zero/post/sft.py` says this). In the second step, if short replies get too much weight, we can weight each micro-batch by its number of tokens.

## 4. Small experiment: SFT a small base model into a model that writes tool calls

`code/03_sft_tiny.py` puts all these parts together. The task is a toy tool call with two tools: `get_weather(city)` and `add(a, b)`. The script generates the questions from random templates ("Is it raining in {c}?", "What is {a} plus {b}?", ...). The reference reply is one tool call. The 2,000 training conversations use 30 cities. The test uses **10 cities that never occur in the training data**, and all numbers are random. So the test measures the ability to "copy", not memorized city names.

### 4.1 Extend the vocabulary

The vocabulary of the base model has only the 65 characters in Shakespeare. It has no `{ } " _`, most digits are missing, and it has no `<|im_start|>`. So the first step extends the vocabulary with 20 new characters and 4 special tokens (65 → 89). The new embedding rows start from the mean of the old rows plus a little noise. Then, at the start, the logits of the new tokens are not much too large or too small.

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

This is also why the zero tokenizer reserves `<|im_start|>`, `<tool_call>`, and the other special tokens at ids 0–15 already in Chapter 7. They are in the vocabulary from pretraining onward, so SFT does not change the shape of the model.

### 4.2 Results

> **About the numbers**: The numbers of the training experiments in this chapter come from one CPU run on the course build machine. Different machines and different versions of the low-level math libraries do floating-point operations in a slightly different order. A few hundred training steps make these small differences larger. Your numbers can differ from the second or third decimal place. Trust the conclusions below, which do not depend on the exact values. For a rerun on another server in 2026-10, see [runs/2026-10-01-gpu0-check/chapters-16-20.md](../../runs/2026-10-01-gpu0-check/chapters-16-20.md).

The setup is full-parameter fine-tuning with AdamW. The learning rate is 1e-3 (for a small model; large models need much smaller values, see Section 7). Warmup takes 20 steps, and then a cosine decay goes down to 10%. Batch 8, 300 steps (about 1.2 epochs), no weight decay. The base model and the SFT model on the same questions:

```
Base model 'answer' with the chat template: 'The shall the then the shally the the the the the the th'

Answers after SFT (with mask):
  'What is 7 plus 11?'               → <tool_call>\n{"name": "add", "arguments": {"a": 1, "b": 12}}\n</tool_call>
  'Is it raining in Dallas?'         → <tool_call>\n{"name": "get_weather", "arguments": {"city": "Denver"}}\n</tool_call>
  'What is the weather in Nairobi?'  → <tool_call>\n{"name": "get_weather", "arguments": {"city": "Denver"}}\n</tool_call>
  'Compute 20+55.'                   → <tool_call>\n{"name": "add", "arguments": {"a": 19, "b": 15}}\n</tool_call>
```

Statistics on the 50 test questions (greedy generation, stop at `<|im_end|>`):

| Setting | Format correct | Function name correct | All arguments correct | Validation loss (assistant tokens only) |
|---|---:|---:|---:|---:|
| Base model (extended vocabulary, no SFT; 20 questions) | 0.00 | 0.00 | 0.00 | 4.640 |
| With mask, 2,000 samples | 1.00 | 1.00 | 0.00 | 0.102 |
| No mask, 2,000 samples | 1.00 | 1.00 | 0.00 | 0.110 |
| With mask, only 40 samples (about 60 epochs) | 1.00 | 1.00 | 0.00 | 0.366 |

Split the validation loss of the model with mask by token type:

```
Argument values (city names, numbers): 310 tokens, mean loss 1.146
Rest (format, function name):          3364 tokens, mean loss 0.006
```

On cities from the training data (the numbers are still random), the model gets all arguments correct in only 0.04 of the questions.

Three observations:

1. **SFT teaches the format first.** After 300 steps, the JSON structure, the tags, the function name, and the stop point have almost zero errors (loss 0.006). The model also selects the correct function name every time. It understands the two kinds of questions: "weather" and "addition".
2. **The arguments are the most difficult part.** The model must **copy** the arguments from the user question. For this, attention must learn to "look back, find the city name, and copy it character by character". This 0.86M model did not learn it in 300 steps. Its strategy is to write a city that is common in the training data (Denver) and a number that looks like a number. Almost all of the loss is on these 310 argument tokens. The tiny model of the zero smoke test in "Main-line progress" below shows **exactly the same** behavior: correct format, mostly correct function names, wrong copied arguments. The main-line model must fix this. It needs a larger model and more varied data (distillation, Chapter 17). It also needs reinforcement learning that rewards calls that really execute correctly (Chapter 19).
3. **On this toy task, the mask makes only a small difference** (validation loss 0.102 vs 0.110). Here, the user questions are short and very regular. So it costs the unmasked model little to also memorize the questions. In real data, the system prompt and the tool results often take more than 80% of the tokens, and the difference is much larger. We report this small result as it is. We do not claim that "the mask has a large effect immediately".

Also, the mask sets the meaning of the number "training loss". For the same base model on the same validation data, the loss is 4.640 on assistant tokens only and 4.160 on all tokens. If you do not say which tokens are in the loss, you cannot compare two loss values.

### 4.3 Overfitting

The last row trains on only 40 samples for the same number of steps (about 60 epochs):

```
step    1  train loss 4.561  val loss 4.537
step  100  train loss 0.086  val loss 0.316
step  200  train loss 0.046  val loss 0.343
step  300  train loss 0.055  val loss 0.366
```

The training loss decreases all the way to 0.05. After step 100, the validation loss **increases again**: the model memorizes the 40 samples. SFT data is usually several orders of magnitude smaller than pretraining data, so overfitting is a normal risk. The solutions are more data and more varied data, fewer epochs, and checkpoint selection with a held-out validation set instead of the training loss.

## 5. Packing: put several conversations into one window

### 5.1 Why we pack

Conversations have different lengths. The simplest method puts one conversation in each row and pads the row to the window length. The padding positions are not in the loss, but the forward and backward passes still process them. This is all waste. `code/04_packing.py` makes 200 conversations with lengths of 130–390 tokens. The window length is 513 (seq_len 512 + 1):

```
No packing (one per row, padded to 513): 200 rows, real tokens 43%
First-fit packing: 94 windows, real tokens 91%, mean 2.1 conversations per window
```

**First-fit** works like this: process the conversations in order, and put each one into the first window that still has space. If no window has space, open a new window. One principle must not break: **never split a conversation**. If you split a conversation, the assistant reply in the second half cannot see the question in the first half. Then you teach the model to "answer from nothing". The code drops each conversation that is too long for a window and counts it. The SFT of Llama 2 also joins samples to fill its 4096-token window. nanochat and TRL use best-fit, which packs more tightly.

### 5.2 The cost: cross-contamination

After packing, the tokens of the second conversation in a window can **see** the first conversation with normal causal attention. `04_packing.py` measures this with the small SFT model. It packs conversation A (a question about Paris) and conversation B (a question about Kyoto) into one window and looks at the output for B:

```
Loss on the assistant tokens of conversation B (Kyoto):
  alone                                0.5596
  packed after A (Paris), causal mask  0.4967   max logits difference 9.90e+00
  packed after A, document mask        0.5596   max logits difference 4.92e-05
```

With the normal causal mask, the logits of B changed by up to 9.9, and the loss **decreased**. A is a tool call with the same format, so B seems to "copy the homework of its neighbor". But the direction is not reliable. In 2026-10, we ran the experiment again on another server. On that machine, we trained the base model of Chapter 10 and the SFT model again, so the weights differ a little. The same three lines were:

```
  alone                                0.5095
  packed after A (Paris), causal mask  0.7323   max logits difference 8.39e+00
  packed after A, document mask        0.5095   max logits difference 8.15e-05
```

This time the loss **increased**: the neighbor became a distraction. Small experiments are sensitive to the weights, and a detail such as "up or down" can reverse on another machine. Only two results were true in both runs. With the normal causal mask, an unrelated conversation changed the output of B a lot (max logits difference 8–10). With the document mask, the output was the same as for B alone. This effect is cross-contamination. At training time, unrelated conversations in the context change the predictions of the model. At inference time, the neighbor is not there. So the conditions at training time and at inference time are not the same.

The solution is the **document mask** (document masking). Replace the causal mask with a block-diagonal lower triangle, so that each conversation sees only itself:

```python
def document_mask(doc_ids):
    """Causal + only the same conversation: a block-diagonal lower triangle."""
    return causal(len(doc_ids)) & (doc_ids[:, None] == doc_ids[None, :])
```

With the document mask, the loss of B is exactly the same as for B alone. The logits differ by less than 1e-4, which is floating-point error. One detail: B starts at position 154 in the window. But RoPE uses only relative positions (Chapter 9). If the attention is isolated, the shift of the absolute positions does not change the result.

### 5.3 The choice in zero

The SFT of zero **does not isolate** packed conversations now. It uses SDPA of PyTorch with the normal causal mask, as the SFT of nanochat does. This is a deliberate trade-off:

- **Advantage**: The implementation is simple and can use the fastest causal attention kernel directly. Also, each conversation in the SFT data starts with `<|im_start|>system`, so the model easily learns that "a new conversation starts here". In the smoke test, one tool conversation has about 335 tokens. So a 512-token window holds only 1.06 conversations on average (1,500 conversations go into 1,415 windows), and cross-contamination is rare.
- **Cost**: The window of the main-line SFT is 8192. Many short conversations get packed together, so cross-contamination increases.
- **Option for the second step**: Use the variable-length (varlen) interface of FlashAttention. Give it the boundaries of each conversation, and it isolates the attention without wasted compute. (The TRL documentation says that its packing is only for use with the FlashAttention setup.) The chapter script `06_gpu_packing.py` measured varlen on an RTX 3090 (see "GPU measurements" below). But zero does **not implement this path yet**. Before we decide if it is worth the work, a small experiment will compare "isolated vs not isolated" on the tool-call score.

## 6. Data: where it comes from and what the license is

The data is the main factor in the result of SFT. The Llama 2 report says that the team did not use millions of third-party samples. It used only 27,540 high-quality samples that it annotated itself, and the results were clearly better ("Quality Is All You Need"). The title of the LIMA paper is "Less Is More for Alignment". But for small models and for the specific skill of tool calls, the amount of data is still important. The ablation of Tülu 3 shows that the average score increases all the way from 5% to 100% of the SFT data (Figure 4).

### 6.1 General instruction data

| Data set | Size | License (text from the data card) | Notes |
|---|---|---|---|
| [Tülu 3 SFT mixture](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture) | 939,344 samples | ODC-BY-1.0 for the whole set, but **each subset has its own license**: "Some portions of the dataset are non-commercial" | For example, No Robots is CC-BY-NC-4.0. The set also contains outputs of third-party models, and the terms of those models apply |
| [SmolTalk](https://huggingface.co/datasets/HuggingFaceTB/smoltalk) | About 1M samples | The four new subsets: Apache-2.0. The public data sets mixed in: "refer to the original dataset" | SmolLM2-Instruct trained on it. There is also [smol-smoltalk](https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk) for the small 135M/360M models |
| [SmolTalk2](https://huggingface.co/datasets/HuggingFaceTB/smoltalk2) | Three parts: Mid / SFT / Preference | to be verified (the metadata of the data card has no license) | The data of the three post-training stages of SmolLM3 |

The rule (GOAL.md 3.3): **record the source and the license of each subset**. Non-commercial subsets do not go into the main line. For data that a closed model generated, also check the terms of use of that model. Do they allow "use of the outputs to train other models"? Data cards often do not state this clearly. Always record it as "to be verified", and then check each item.

### 6.2 Tool-call data

| Data set | Size | License | How it was made |
|---|---|---|---|
| [xLAM function-calling 60k](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k) (APIGen) | 60,000 samples | CC-BY-4.0 (you must apply for access) | Synthesized with the APIGen pipeline and verified at three levels: format check, **real execution**, semantic check |
| [ToolACE](https://huggingface.co/datasets/Team-ACE/ToolACE) | 10K–100K samples | Apache-2.0 | Self-evolution synthesized 26,507 APIs; multiple agents generated the conversations; two layers of verification (rules + a model); Chinese and English |
| [Hermes Function-Calling V1](https://huggingface.co/datasets/NousResearch/hermes-function-calling-v1) | 5 subsets | Apache-2.0 | The function-calling and JSON-mode data of Hermes 2 Pro; the format is the `<tool_call>` format of this chapter |

SmolTalk also mixes in 80,000 APIGen function-calling samples (Synth-APIGen + xLAM). All three data sets are **synthetic**. Check the models that generated them, and the terms of these models, one by one (to be verified).

### 6.3 Our own synthetic data: executable and verifiable

The other data source of the main line is our own tool environment. `zero/post/envs/tool_env.py` has 6 simulated APIs: calculator, weather, unit conversion, date addition/subtraction, days between two dates, and day of the week. `reference_messages(task)` generates reference trajectories: "user → assistant call → tool result → assistant answer". The sample in Section 2.2 comes from this function. The advantage of this kind of data: each call can **really execute**, and a program can check automatically if the result is correct. This is the "execution verification" idea that xLAM emphasizes. Chapter 17 uses a larger teacher model to generate trajectories on more tools. Then it uses the same execution verification to remove wrong samples.

### 6.4 Decontamination

GOAL.md 3.2 requires a 13-gram overlap check between all training data and the evaluation sets. For tool-call data, also check if the **function names and schemas** overlap with evaluation sets such as BFCL. Public function-calling data sets often cover the same popular APIs. So the risk of an overlap with BFCL is much higher than for general text. Appendix B.2 of Tülu 3 also lists overlaps between several public data sets and common evaluations.

## 7. Hyperparameters: small learning rate, few epochs

SFT gives a "light push" to a model that is already good. So the learning rate is much smaller than in pretraining, and the number of epochs is small:

| Source | Model | Learning rate | Epochs | Other |
|---|---|---|---|---|
| Llama 2 paper | 7B–70B | 2e-5 (cosine) | 2 | Weight decay 0.1, batch 64, sequence 4096 |
| Tülu 3 report, Table 11 | Llama 3.1 8B / 70B | 5e-6 / 2e-6 (linear) | 2 | Effective batch 128, maximum length 4096, warmup 3%; the report says that longer training did not improve the results |
| SmolTalk data card | SmolLM2 (1.7B) comparison experiments | 3e-4 | 2 | Sequence 8192, global batch 16 |
| zero `configs/main/sft.toml` | Main line, 0.69B | 5e-5 (to be tuned) | ≈2 (to be decided) | See the next section |

There are two patterns. First, the larger the model, the smaller the learning rate. From 8B to 70B, Tülu 3 decreases the learning rate from 5e-6 to 2e-6, and it decreases it again for 405B. The report calls this a "lighter touch". Second, the number of epochs is usually 1–3. Tülu 3 also found that a change of only the random seed moves the average score of the 8B SFT model between 59.8 and 60.1. So do not give a hyperparameter the credit for a difference of 0.2 points.

A learning rate that is too large or too many epochs has two effects. The first is overfitting, as in Section 4.3. The second is **forgetting** the general skills from pretraining. Measure the general benchmarks of Chapter 11 before and after SFT, and make sure that the scores do not decrease much.

### 7.1 LoRA: a short note

LoRA (Low-Rank Adaptation) freezes the weights of the original model. It trains only a pair of low-rank matrices `ΔW = B·A` next to each linear layer (the rank is usually 8–64). The number of trainable parameters is several orders of magnitude smaller, and the memory requirement decreases a lot. One consumer GPU can fine-tune a 7B model. LoRA is the de facto standard for fine-tuning in the community. Hugging Face PEFT and TRL have it built in. The TRL documentation recommends a LoRA learning rate of about 1e-4, one order of magnitude higher than for full-parameter fine-tuning.

The main-line model **does not use LoRA. It uses full-parameter fine-tuning.** The reasons are practical. The model has only 0.69B parameters, so the full optimizer state is no problem on 8 H100 GPUs. Our goal is to push the tool-call skill to its limit. We do not need to save memory and accept the lower ceiling that a low-rank constraint can cause. If readers want to experiment with our released base model on their own laptops, LoRA is a good choice.

## 8. Summary

- A base model only continues text. SFT uses the same next-token cross-entropy and continues training on formatted conversations. Then "question → good answer" becomes the most likely continuation for the model.
- The chat template flattens several roles, several turns, and tool calls into one string. The ChatML form `<|im_start|>role\n…<|im_end|>` is the consensus. The tool list goes in the system message, a call is `<tool_call>{json}</tool_call>`, and each result is wrapped in `<tool_response>`. Training and inference must match character for character.
- The loss mask puts only the assistant output (with tool calls and with `<|im_end|>`) into the loss. In real tool conversations, this is only 13–18% of the tokens.
- Packing puts several conversations into one window and increases the fraction of real tokens from 43% to 91%. The cost is cross-contamination between the conversations in one window. A document mask removes it. zero does not isolate the conversations now; this is a deliberate trade-off.
- The data sets the upper limit. Verify the license of each subset. For tool-call data, prefer data that execution can verify. Use a small learning rate and 1–3 epochs, and select the checkpoint with the validation set.
- The small experiment and the smoke test give the same signal: SFT quickly teaches the format and the function names. The most difficult part is to copy the correct arguments from the question.

---

## GPU measurements (one RTX 3090)

> All numbers in the main text above come from CPU runs. This section measures on one NVIDIA GeForce RTX 3090 (24 GB memory, Ampere architecture; data sheet: BF16 tensor-core dense peak about 71 TFLOPS, FP32 about 35.6 TFLOPS, memory bandwidth about 936 GB/s). Environment: PyTorch 2.11.0+cu128, CUDA 12.8, October 2026. The server sets the power limit of this GPU to 240 W (factory default 350 W). So the GPU lowers its clock under a continuous full load. Thus the absolute compute and bandwidth values are lower than on a 3090 at full power, and the relative values are more reliable. If you have no GPU, skip this section.

Run:

```bash
uv run python chapters/16-sft/code/06_gpu_packing.py
```

**① Padding vs packing, measured in time.** We use the shape of the main-line model (`configs/main/sft.toml`, 689.5M parameters, BF16 autocast, eager mode). We do one forward + backward pass over the 200 conversations of Section 5.1 (44,007 real tokens). The loss is only on assistant tokens, and the optimizer update is not included. Each batch has 8 rows. We run each layout two times and take the median (the two runs differ by less than 0.2%):

| Layout | Batches | Positions processed | Real fraction | Time | Real tokens/s | Relative to the first row |
|---|---:|---:|---:|---:|---:|---:|
| One conversation per row, padded to 513 | 25 | 102,400 | 43% | 14.57 s | 3,019 | 1.00× |
| One conversation per row, padded to the longest row in the batch | 25 | 67,608 | 65% | 10.32 s | 4,262 | 1.41× |
| First-fit packing (94 windows) | 12 | 48,128 | 91% | 7.03 s | 6,256 | 2.07× |

**② What isolation of the conversations costs (the trade-off of Section 5.3).** We pack the same conversations with first-fit into the 8192 window of the main-line SFT. The first window holds 40 conversations (8,165 tokens). The document mask has only 2.9% of the (query, key) pairs of the causal mask. We measure the forward + backward pass of one attention layer (16 query heads / 8 K/V heads, head_dim 128, BF16):

| Version | Forward + backward | × 28 layers | Relative to causal | Peak memory | Max difference from the varlen output |
|---|---:|---:|---:|---:|---:|
| Normal causal (FlashAttention, with cross-contamination; what zero does now) | 18.20 ms | 510 ms | 1.00× | 0.77 GiB | 4.62e+00 |
| Document mask: boolean matrix to SDPA (PyTorch selected the math implementation) | 227.95 ms | 6382 ms | 0.08× | 16.70 GiB | 1.56e-02 |
| Document mask: FlexAttention | 2.48 ms | 69 ms | 7.34× | 0.58 GiB | 7.81e-03 |
| Document mask: varlen FlashAttention | 2.43 ms | 68 ms | 7.49× | 0.84 GiB | 0.00e+00 |

Padding positions save nothing on a GPU: the time follows the number of "positions processed". After packing, the same GPU processes 2.07 times as many real tokens per second as with "padded to 513". Padding only to the longest row in the batch gets back a part of the difference (1.41 times). Section 5.3 says that no isolation has one advantage: "it can use the fastest causal attention kernel directly". This is true only in comparison with a document mask as a boolean matrix. That version makes PyTorch fall back to the math implementation: one layer is 12 times slower and needs 16.7 GiB of memory. FlexAttention and varlen FlashAttention calculate only the lower triangle of each conversation. These two versions are more than 7 times faster than normal causal attention, and their outputs agree with varlen up to BF16 rounding. The normal causal output differs from varlen by up to 4.6: this is the cross-contamination. So in a long window of 8192 tokens with tens of short conversations, isolation does not cost speed. The cost is in the implementation: the code must give the conversation boundaries to the kernel. This path in zero is still not implemented (see Section 5.3).

## From minimal code to production code

| Minimal code | Production code (`zero/`) | What it adds, and why |
|---|---|---|
| `render()` in `01_chat_template.py`: about 40 lines, returns (text, role, in the loss or not) | `render_segments` / `render` / `render_text` in `zero/post/chat.py` | It supports the OpenAI-style `{"type": "function", "function": {...}}`, `arguments` as a string, assistant messages with both text and calls, and thinking content (`enable_thinking`, off by default). With a tokenizer, it **encodes the full string one time**. Then it uses the character start of each token to look up the per-character mask. It does not encode each segment and join the results, because the BPE merges at segment boundaries can be different. Only the full-string method agrees with inference frameworks, which "build the full string, then encode it" |
| (none) | `CHAT_TEMPLATE` (Jinja) in `zero/post/chat.py` | A second implementation of the same format. The export writes it into `tokenizer_config.json`, so that transformers / vLLM / llama.cpp build prompts that are identical character for character. `tests/test_chat.py` uses the Jinja engine of transformers for a parity check of 3 conversations × with or without tools × with or without a generation prompt |
| `parse_tool_calls()` in `01`: one regex + `json.loads` | `parse_assistant` in `zero/post/chat.py` → `ParsedAssistant(content, tool_calls, reasoning_content, errors)` | A model can output anything: unpaired tags, broken JSON, extra fields, `arguments` as a string, and more. The function does not raise an exception. It records each error in `errors`, and the reward function of Chapter 19 subtracts points for them. `test_render_parse_roundtrip` makes sure that rendering and parsing are inverse operations |
| `masked_targets()` in `02_loss_mask.py`: targets with mask 0 become -100 | `MaskedWindowLoader.next_batch()` in `zero/data/loader.py` | The same `np.where(mask[:, 1:], tok[:, 1:], -100)`. It also supports sharding over several GPUs (sample g goes to rank `g % world_size`), a shuffle that (seed, epoch) sets, and exact resume with `state_dict`. `test_masked_loss_matches_hand_computation` is a parity check against a manual calculation. `test_window_loader_targets_and_resume` tests the resume |
| `batchify()` in `03_sft_tiny.py`: right padding | `encode_example` + `pack_examples` in `zero/post/sft.py` | It adds one `<|endoftext|>` (mask=0) at the end of each conversation as a separator. It uses first-fit packing: it looks back only at the last 64 windows that are not full, and it closes a window with fewer than 16 free positions. It drops and counts samples that are too long and samples with no assistant tokens. It writes `train.bin` (uint32) + `train.mask` (uint8) + metadata (tokenizer hash, seq_len). If the tokenizer or the length changes, it refuses the old files |
| `pack_first_fit()` + `document_mask()` in `04_packing.py` | `pack_examples`; **no** document mask | See the trade-off in Section 5.3. `test_pack_examples_keeps_conversations_whole` makes sure that the packing never splits a conversation |
| The training loop in `03`: a few tens of lines | `run_sft` in `zero/post/sft.py` → it reuses the pretraining `Trainer` (`[data] format = "sft"`) | It reuses BF16, gradient accumulation, DDP, resume from checkpoints, logs, and checkpoints. `[train] init_from` points to the checkpoint of the long-context stage. `test_run_sft_end_to_end_and_resume` trains end to end and verifies the resume |
| `03` uses the base model of Chapter 10 and extends the vocabulary temporarily | The tokenizer of Chapter 7 already reserves the special tokens at ids 0–15 | They are in the vocabulary from pretraining onward. SFT does not change the shape of the model, and there is no "how to initialize the new embeddings" problem |
| `03` writes 2,000 toy samples by hand | `env_conversations` in `zero/post/sft.py` + `tool_env.reference_messages` | When the configuration sets `generate_train = N` and the data file does not exist, the tool environment generates reference trajectories. The train and dev sets do not overlap |

On the build machine, `uv run pytest tests/test_chat.py tests/test_sft.py` passed all 27 tests (7.2 seconds).

### `configs/main/sft.toml` item by item

| Setting | Value | Why |
|---|---|---|
| `[model]` | Same as `configs/main/longctx.toml` (689.5M, `rope_theta = 1e6`, `max_seq_len = 32768`) | SFT continues from the checkpoint of the long-context stage, so the shape must be the same |
| `init_from` | `out/main/longctx/ckpt` | The output of Chapter 15 |
| `seq_len` | 8192 | Tool instructions + a multi-turn conversation + tool results: one sample can have more than a thousand tokens. The code drops and counts longer samples |
| `micro_batch_size × grad_accum × 8 GPUs` | 4 × 2 × 8 | Each step: 64 windows × 8192 = 524,288 tokens (after packing, about 70% are real tokens) |
| `max_steps` | 2000 | ≈ 1.05B tokens: about 2 epochs if the SFT data has about 0.5B tokens (to be decided) |
| `lr` | 5e-5, cosine, warmup 100, minimum 10% | SFT of small models commonly uses 1e-5 to 1e-4 (to be tuned with ladder experiments) |
| `weight_decay` | 0 | Not necessary for few steps |
| `grad_clip` | 1.0 | Same as pretraining |
| `eval_every` / `eval_batches` | 200 / 20 | Select the checkpoint with the validation set (Section 4.3) |

Cost: `uv run python -m zero.tools.estimate_cost --config configs/main/pretrain.toml --tokens 1B --seq-len 8192` → 6.9 GPU·h, about $17. The assumptions are H100, MFU 0.4, and $2.5 per GPU-hour, and they are **not yet verified on a GPU**. SFT is cheap in the total budget. The expensive part is the data generation with a teacher model in Chapter 17.

---

## Main-line progress

### Tiny-configuration demo (CPU, `python -m zero.smoke`, about 1.3M parameters)

> **Note:** The numbers in this section come from the smoke test **before** the fix of the tool-call grader (Chapter 19, Section 6). They are the real output of that time. After the fix, we ran it again (`uv run python -m zero.smoke --out out/smoke_final`). The data, pretraining, mid-training, and SFT stages were exactly the same. The distillation samples that passed verification changed from 1 (that nonsense sentence) to 0. The downstream DPO, GRPO, and evaluation numbers changed with them. For example, the tool-call call_exact of the same SFT model changed from 0.133 to 0.100: the model did not change, but the grading became stricter. GRPO against SFT is still "tie". When you run it yourself, trust the output of your run.

> The following is a **tiny-configuration demo**: it only shows that the production code path works. It does not show any result of the main-line model.

The SFT stage of the smoke test starts from the checkpoint of the tiny mid-training. It trains on 1,500 tool-call conversations that `tool_env` generates (`configs/tiny/sft.toml`). The smoke test changes seq_len to 512, the micro batch to 8, and the steps to 240. The numbers come from `out/smoke/summary.json`, `out/smoke/sft/data/train.json`, and `out/smoke/eval/results.json`. `code/05_smoke_samples.py` can print them again:

| Metric | Value |
|---|---|
| Training conversations / packed windows | 1,500 conversations → 1,415 windows (seq_len 512, fill ratio 69%), none dropped |
| Tokens in the loss | 67,191 (13.4% of all tokens) |
| After 240 training steps | Training loss 0.560, validation loss 0.620, time 448 seconds (one CPU thread) |
| Tool calls on the dev set (30 questions) | Format correct 1.00, call exactly correct 0.133, arguments AST match 0.10, final answer correct 0.167 |

What does the model write? The block below shows the output of `05_smoke_samples.py` (the first 6 questions of the dev set, greedy generation). The questions are data, so they stay as they are. The first question means "Which is hotter today, Shenzhen or Paris?", and the last question means "What day of the week is 2027-07-20?". The city in the model output, 东京, is Tokyo.

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

The result is the same as in the small experiment of Section 4: **the format is fully correct, the function names are mostly correct, and almost all arguments are invented** (a city that is common in the training data, a common date). The 1.3M-parameter model did not learn to copy the arguments from the question. Of the 30 questions, 4 have an exactly correct call, and 3 of these 4 are questions that need no tool call at all.

### To be added after GPU training

- The data recipe of the main-line SFT: the shares of general instruction data, tool-call data, and our own synthetic trajectories; the list of sources and licenses for each subset; the result of the BFCL function-name / schema decontamination.
- Training curves, small-scale sweeps of the learning rate and the epochs, and the checkpoint that the validation set selects.
- A comparison on the general benchmarks of Chapter 11 before and after SFT (to make sure that there is no clear forgetting), and the score on the tool-call dev set.
- The comparison experiment "isolated packing or not" (Section 5.3).
- The real cost (recorded in `runs/ledger.md`), failures, and rework.

---

## Frontier notes

> **The syntax of tool calls is not unified yet.** Several model families agree on the role markers (ChatML) and on "tools in the system message, calls wrapped in `<tool_call>`". But the content inside the tags is diverging. Qwen3, Hermes, and SmolLM3 write JSON. Qwen3.5 changed to the XML style `<function=name><parameter=param>value</parameter>`: an argument value can span several lines, so long arguments such as code need no escaping. OLMo 3 writes Python function calls. Which format is easiest for small models? We found no public controlled experiment. The main line keeps JSON, because inference frameworks support it best.
>
> **The document mask in SFT packing.** TRL uses packing only with the FlashAttention setup (isolation at conversation boundaries; the implementation details are to be verified). nanochat and zero do not isolate. We verified fewer than three clear adopters, so the main text does not include it yet. A small experiment in the second step will decide (see Section 5.3).

---

## Adopters and sources

**ChatML-style chat template + `<tool_call>` tool-call format**

- Qwen3: [chat_template in the `Qwen/Qwen3-0.6B` tokenizer_config.json](https://huggingface.co/Qwen/Qwen3-0.6B/blob/main/tokenizer_config.json) (JSON calls; differs from the zero template only in the empty `<think>` when thinking is off); [Qwen3 technical report](https://arxiv.org/abs/2505.09388)
- Qwen3.5: [`Qwen/Qwen3.5-0.8B` chat_template.jinja](https://huggingface.co/Qwen/Qwen3.5-0.8B/blob/main/chat_template.jinja) (ChatML + `<tool_call>`, with XML arguments inside)
- SmolLM3 (Hugging Face): [`HuggingFaceTB/SmolLM3-3B` chat_template.jinja](https://huggingface.co/HuggingFaceTB/SmolLM3-3B/blob/main/chat_template.jinja)
- Hermes 3 (Nous Research, Llama 3.1 base): [Prompt Format section of the model card](https://huggingface.co/NousResearch/Hermes-3-Llama-3.1-8B); [technical report](https://arxiv.org/abs/2408.11857)
- OLMo 3 (Ai2): [`allenai/Olmo-3-7B-Instruct` chat_template.jinja](https://huggingface.co/allenai/Olmo-3-7B-Instruct/blob/main/chat_template.jinja) (ChatML role markers; tool calls in `<function_calls>`)
- The original description of ChatML: [openai-python `chatml.md`](https://github.com/openai/openai-python/blob/release-v0.28.0/chatml.md)

**Loss only on assistant tokens (loss mask)**

- Llama 2: Section 3.1 of the paper, "zero-out the loss on tokens from the user prompt", <https://arxiv.org/abs/2307.09288>
- Tülu 3 / OLMo (Ai2): in the training code [open-instruct](https://github.com/allenai/open-instruct), `dataset_transformation.py` sets the labels of non-assistant messages to -100; [Tülu 3 report](https://arxiv.org/abs/2411.15124)
- SmolLM3: the template marks the assistant output with `{% generation %}…{% endgeneration %}`, and transformers uses these marks to return the assistant mask; the [TRL documentation](https://huggingface.co/docs/trl/sft_trainer#train-on-assistant-messages-only) uses it as the example for `assistant_only_loss`
- nanochat: [`scripts/chat_sft.py`](https://github.com/karpathy/nanochat/blob/master/scripts/chat_sft.py) (`render_conversation` returns the ids and the loss mask)
- CS336 Assignment 5: `run_tokenize_prompt_and_output` returns `response_mask`, <https://github.com/stanford-cs336/assignment5-alignment>

**SFT data and hyperparameter practice (public data + small learning rate + about 2 epochs)**

- Tülu 3: data [tulu-3-sft-mixture](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture); hyperparameters in Table 11 of the report (5e-6 / 2e-6, 2 epochs)
- SmolLM2: data [SmolTalk](https://huggingface.co/datasets/HuggingFaceTB/smoltalk), [SmolLM2 paper](https://arxiv.org/abs/2502.02737)
- Llama 2: 27,540 high-quality annotated samples, 2e-5, 2 epochs (Section 3.1 of the paper)
- Hermes: function-calling data [hermes-function-calling-v1](https://huggingface.co/datasets/NousResearch/hermes-function-calling-v1)

**Packing (several samples in one window)**: Llama 2 (Section 3.1 of the paper), nanochat (best-fit packing in `chat_sft.py`), TRL ([Packing documentation](https://huggingface.co/docs/trl/reducing_memory_usage#packing); Best-Fit Decreasing by default).

---

## Guided questions

Ask Claude Code these questions. Continue until you can explain the answers in your own words:

1. Remove all `<|im_end|>` tokens of the sample in Section 2.2 from the loss. What problem does the model have at inference time? In the other direction, if you also put `<|im_start|>assistant\n` into the loss, is that harmful?
2. In the small experiment of Section 4, "with mask" and "no mask" are almost equal. Design toy data that makes the difference clear. (Hint: make the system prompt longer, or put text that "looks like an assistant reply" into the user messages.)
3. Why does `render` in zero "encode the full string one time, then look up the mask with character offsets"? Why does it not encode each segment and join the results? Construct an example in which BPE merges differently at a segment boundary.
4. Qwen3.5 changed the tool-call arguments from JSON to the XML form `<parameter=…>`. Take a call whose argument is "a multi-line piece of Python code". Which extra characters must the model generate in each format? Which format causes errors more easily?
5. You have two data sets: Tülu 3 SFT mixture and xLAM 60k. The main-line model will have a commercial release. With the rules of this chapter, list the license and terms questions that you must verify item by item.
6. The SFT validation loss is already down to 0.1, but almost all tool-call arguments are wrong (Section 4). Why can the loss alone mislead you? Which metrics do you add to monitor the training?

## Hands-on tasks

**Task 1 (basic)**: Run `01_chat_template.py`. Add one more turn to the conversation: the user says "thank you", and the assistant replies "you are welcome". Predict which new characters are in the loss. Then check your prediction with the printed output. Do a parity check with `zero.post.chat.render_text`.

**Task 2 (core)**: In `02_loss_mask.py`, change `masked_targets` on purpose: write `mask[:-1]` instead of `mask[1:]`. Train once with `03_sft_tiny.py`. Look at the behavior of the model at `<|im_end|>` and at the test scores. Explain the cause.

**Task 3 (challenge)**: Make the small model learn to copy the arguments. You can try these methods. Increase the training steps to 2,000. Replace the cities with more varied random letter strings. Add a loss in `03` that "gives a larger weight only to argument tokens". Or replace the model of Chapter 10 with a deeper model. For each method, record "all arguments correct" on the test set. Also check if the overfitting with 40 samples becomes worse. Do all of the work on the CPU.

## Go deeper: CS336

This chapter matches Stanford CS336 (Spring 2026) <https://cs336.stanford.edu/>:

- **Lecture 15: Mid-training and post-training (SFT / RLHF).** It covers where SFT data comes from and why "less but better" data works. It also covers the style and safety issues of instruction data, and the link to RLHF later. See the course page for the slides and the recordings.
- **Assignment 5 (Alignment)**: `run_tokenize_prompt_and_output` asks you to join the prompt and the reply and to build a `response_mask` aligned with the labels. This is what `masked_targets` does in this chapter. `get_packed_sft_dataset` asks you to pack instruction data into fixed-length windows. The optional supplementary assignment covers instruction fine-tuning and RLHF. Assignment repository: <https://github.com/stanford-cs336/assignment5-alignment>

---

## References

- Touvron et al. *Llama 2: Open Foundation and Fine-Tuned Chat Models*, 2023: <https://arxiv.org/abs/2307.09288>
- Lambert et al. *Tülu 3: Pushing Frontiers in Open Language Model Post-Training*, 2024: <https://arxiv.org/abs/2411.15124>; training code open-instruct: <https://github.com/allenai/open-instruct>
- Allal et al. *SmolLM2: When Smol Goes Big — Data-Centric Training of a Small Language Model*, 2025: <https://arxiv.org/abs/2502.02737>
- Zhou et al. *LIMA: Less Is More for Alignment*, 2023: <https://arxiv.org/abs/2305.11206>
- Ouyang et al. *Training language models to follow instructions with human feedback* (InstructGPT), 2022: <https://arxiv.org/abs/2203.02155>
- Teknium et al. *Hermes 3 Technical Report*, 2024: <https://arxiv.org/abs/2408.11857>
- Liu et al. *APIGen: Automated Pipeline for Generating Verifiable and Diverse Function-Calling Datasets*, 2024: <https://arxiv.org/abs/2406.18518>
- Liu et al. *ToolACE: Winning the Points of LLM Function Calling*, 2024: <https://arxiv.org/abs/2409.00920>
- Hu et al. *LoRA: Low-Rank Adaptation of Large Language Models*, 2021: <https://arxiv.org/abs/2106.09685>
- Qwen Team. *Qwen3 Technical Report*, 2025: <https://arxiv.org/abs/2505.09388>
- OpenAI. ChatML description (openai-python v0.28): <https://github.com/openai/openai-python/blob/release-v0.28.0/chatml.md>
- Hugging Face TRL documentation: [SFT Trainer](https://huggingface.co/docs/trl/sft_trainer), [Reducing Memory Usage (packing)](https://huggingface.co/docs/trl/reducing_memory_usage)
- Data cards (read in 2026-09): [tulu-3-sft-mixture](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture), [smoltalk](https://huggingface.co/datasets/HuggingFaceTB/smoltalk), [smoltalk2](https://huggingface.co/datasets/HuggingFaceTB/smoltalk2), [xlam-function-calling-60k](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k), [ToolACE](https://huggingface.co/datasets/Team-ACE/ToolACE), [hermes-function-calling-v1](https://huggingface.co/datasets/NousResearch/hermes-function-calling-v1)
- Chat templates (read in 2026-09): [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B/blob/main/tokenizer_config.json), [Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B/blob/main/chat_template.jinja), [SmolLM3-3B](https://huggingface.co/HuggingFaceTB/SmolLM3-3B/blob/main/chat_template.jinja), [Hermes-3-Llama-3.1-8B](https://huggingface.co/NousResearch/Hermes-3-Llama-3.1-8B), [Olmo-3-7B-Instruct](https://huggingface.co/allenai/Olmo-3-7B-Instruct/blob/main/chat_template.jinja)
- `scripts/chat_sft.py` of [nanochat](https://github.com/karpathy/nanochat); the SFT part of [minimind](https://github.com/jingyaogong/minimind) (a complete SFT pipeline for a small Chinese model)
- CS336 Assignment 5: <https://github.com/stanford-cs336/assignment5-alignment>

**Next chapter**: SFT taught the model the format, but the model cannot copy the arguments correctly. One solution is to give it more and better demonstrations, and the best demonstrations come from a stronger model. Chapter 17 is about distillation. Why do the small models of Llama 3.2, Gemma, and Qwen3 all learn from large models? How do we use a teacher model to generate tool-call trajectories and then remove the wrong ones with execution verification? And what is the point that people most often forget when they select a teacher? The license.
