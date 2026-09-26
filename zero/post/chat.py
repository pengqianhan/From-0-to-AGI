"""对话模板与工具调用格式：渲染、解析、loss mask（对应第 16 章 SFT，第 19、20 章也用）。

格式与 Qwen3 的 ChatML 模板一致（便于导出后被 transformers / vLLM / llama.cpp 直接使用）：

    <|im_start|>system
    {系统提示}

    # Tools

    You may call one or more functions to assist with the user query.

    You are provided with function signatures within <tools></tools> XML tags:
    <tools>
    {"type": "function", "function": {...}}
    </tools>

    For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
    <tool_call>
    {"name": <function-name>, "arguments": <args-json-object>}
    </tool_call><|im_end|>
    <|im_start|>user
    北京今天天气怎么样？<|im_end|>
    <|im_start|>assistant
    <tool_call>
    {"name": "get_weather", "arguments": {"city": "北京"}}
    </tool_call><|im_end|>
    <|im_start|>user
    <tool_response>
    {"city": "北京", "condition": "晴", "temp_c": 25}
    </tool_response><|im_end|>
    <|im_start|>assistant
    北京今天晴，25°C。<|im_end|>

要点：

- 工具（JSON schema）写在 system 里的 `<tools></tools>` 中，每行一个；
- 助手的工具调用写成 `<tool_call>{"name": ..., "arguments": {...}}</tool_call>`（Hermes / Qwen 风格），
  vLLM 用 `--tool-call-parser hermes` 就能解析出结构化的 tool_calls；
- 工具结果是 `role="tool"` 的消息，连续几条合并进一个 user 轮，各自包在 `<tool_response>` 里；
- **loss mask**：只有助手"自己说的话"算 loss——从 `<|im_start|>assistant\\n` 之后到 `<|im_end|>`
  （含 `<|im_end|>`，模型要学会什么时候停；含工具调用的 token）。系统提示、用户话、工具返回都不算；
- 思考（`<think>…</think>`）默认关闭：`enable_thinking=False` 时历史里的 `reasoning_content` 不渲染。
  与 Qwen3 的一处差别：Qwen3 关闭思考时会在生成提示后塞一个空的 `<think>\\n\\n</think>\\n\\n`，
  本模板不塞（我们的模型从没见过思考格式，塞了反而是分布外输入）。

同一份格式有两种实现，必须逐字一致（`tests/test_chat.py` 用 transformers 的 `apply_chat_template` 对拍）：

1. Python：`render()`，训练时用，顺带算出 loss mask；
2. Jinja：`CHAT_TEMPLATE`，导出时写进 `tokenizer_config.json`，推理框架用它拼提示词。
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from zero.tokenizer import Tokenizer

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"

TOOLS_PREAMBLE = (
    "# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
    "You are provided with function signatures within <tools></tools> XML tags:\n<tools>"
)
TOOLS_POSTAMBLE = (
    "\n</tools>\n\nFor each function call, return a json object with function name and arguments "
    "within <tool_call></tool_call> XML tags:\n<tool_call>\n"
    '{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call>'
)


def tojson(x: Any) -> str:
    """与 transformers 模板环境里的 `tojson` 过滤器完全相同（不转义非 ASCII，默认分隔符）。"""
    return json.dumps(x, ensure_ascii=False)


def _jinja_str(s: str) -> str:
    """Python 字符串 → Jinja 单引号字面量。"""
    return "'" + s.replace("\\", "\\\\").replace("'", "\\'").replace("\n", "\\n") + "'"


# ---------------------------------------------------------------------------
# 渲染（Python）
# ---------------------------------------------------------------------------


def normalize_tool_call(tc: dict[str, Any]) -> dict[str, Any]:
    """OpenAI 风格 {"type": "function", "function": {...}} 或 {"name", "arguments"} → {"name", "arguments"}。"""
    if "function" in tc and isinstance(tc["function"], dict):
        tc = tc["function"]
    return {"name": tc.get("name"), "arguments": tc.get("arguments", {})}


def format_tool_call(tc: dict[str, Any]) -> str:
    tc = normalize_tool_call(tc)
    args = tc["arguments"]
    args_s = args if isinstance(args, str) else tojson(args)
    return '<tool_call>\n{"name": ' + tojson(tc["name"]) + ', "arguments": ' + args_s + "}\n</tool_call>"


def _content(msg: dict[str, Any]) -> str:
    c = msg.get("content")
    if c is None:
        return ""
    return c if isinstance(c, str) else tojson(c)


def render_segments(
    messages: Sequence[dict[str, Any]],
    tools: Sequence[dict[str, Any]] | None = None,
    add_generation_prompt: bool = False,
    enable_thinking: bool = False,
) -> list[tuple[str, bool]]:
    """把对话渲染成若干段 (文本, 是否算 loss)。拼起来就是完整的提示词。"""
    segs: list[tuple[str, bool]] = []
    msgs = list(messages)

    def emit(s: str, train: bool = False) -> None:
        if s:
            segs.append((s, train))

    has_system = bool(msgs) and msgs[0].get("role") == "system"
    if tools:
        emit(f"{IM_START}system\n")
        if has_system:
            emit(_content(msgs[0]) + "\n\n")
        emit(TOOLS_PREAMBLE)
        for tool in tools:
            emit("\n" + tojson(tool))
        emit(TOOLS_POSTAMBLE + f"{IM_END}\n")
    elif has_system:
        emit(f"{IM_START}system\n" + _content(msgs[0]) + f"{IM_END}\n")

    for i, m in enumerate(msgs):
        role = m.get("role")
        if role == "system" and i == 0:
            continue
        if role in ("user", "system"):
            emit(f"{IM_START}{role}\n" + _content(m) + f"{IM_END}\n")
        elif role == "assistant":
            emit(f"{IM_START}assistant\n")
            body = ""
            reasoning = m.get("reasoning_content")
            if enable_thinking and isinstance(reasoning, str) and reasoning:
                body += "<think>\n" + reasoning.strip() + "\n</think>\n\n"
            content = m.get("content") if isinstance(m.get("content"), str) else ""
            body += content
            for j, tc in enumerate(m.get("tool_calls") or []):
                if (j == 0 and content) or j > 0:
                    body += "\n"
                body += format_tool_call(tc)
            emit(body + IM_END, train=True)
            emit("\n")
        elif role == "tool":
            if i == 0 or msgs[i - 1].get("role") != "tool":
                emit(f"{IM_START}user")
            emit("\n<tool_response>\n" + _content(m) + "\n</tool_response>")
            if i == len(msgs) - 1 or msgs[i + 1].get("role") != "tool":
                emit(f"{IM_END}\n")
        else:
            raise ValueError(f"未知的 role：{role!r}")
    if add_generation_prompt:
        emit(f"{IM_START}assistant\n")
    return segs


def render(
    messages: Sequence[dict[str, Any]],
    tools: Sequence[dict[str, Any]] | None = None,
    add_generation_prompt: bool = False,
    tokenizer: Tokenizer | None = None,
    enable_thinking: bool = False,
) -> tuple[str, list[bool]] | tuple[list[int], list[bool]]:
    """渲染对话。

    - 不给 tokenizer：返回 (text, mask)，mask 是逐字符的布尔列表；
    - 给 tokenizer：返回 (ids, mask)，mask 是逐 token 的：True 表示这个 token 是助手输出，
      训练时要预测它（SFT 的 loss mask）。

    分词时对**整段文本**一次编码（与推理框架拼好提示词再编码完全一致），再用每个 token 的字符
    起点查逐字符 mask——而不是一段一段分别编码再拼接（段边界处的 BPE 合并可能不同）。
    """
    segs = render_segments(messages, tools, add_generation_prompt, enable_thinking)
    text = "".join(s for s, _ in segs)
    char_mask: list[bool] = []
    for s, train in segs:
        char_mask.extend([train] * len(s))
    if tokenizer is None:
        return text, char_mask
    ids, offsets = tokenizer.encode_with_offsets(text)
    mask = [char_mask[a] if a < len(char_mask) else False for a, _ in offsets]
    return ids, mask


def render_text(
    messages: Sequence[dict[str, Any]],
    tools: Sequence[dict[str, Any]] | None = None,
    add_generation_prompt: bool = False,
    enable_thinking: bool = False,
) -> str:
    return "".join(s for s, _ in render_segments(messages, tools, add_generation_prompt, enable_thinking))


# ---------------------------------------------------------------------------
# 解析助手输出
# ---------------------------------------------------------------------------

_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)


@dataclass
class ParsedAssistant:
    content: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    reasoning_content: str | None = None
    errors: list[str] = field(default_factory=list)  # 格式错误（奖励函数据此扣分）

    def to_message(self) -> dict[str, Any]:
        m: dict[str, Any] = {"role": "assistant", "content": self.content}
        if self.tool_calls:
            m["tool_calls"] = [dict(tc) for tc in self.tool_calls]
        if self.reasoning_content is not None:
            m["reasoning_content"] = self.reasoning_content
        return m

    def __getitem__(self, key: str) -> Any:  # 允许 parsed["content"] / parsed["tool_calls"]
        return getattr(self, key)


def parse_tool_call_json(body: str) -> tuple[dict[str, Any] | None, str | None]:
    """解析 <tool_call> 里的 JSON，返回 (call, error)。"""
    try:
        obj = json.loads(body.strip())
    except json.JSONDecodeError as e:
        return None, f"tool_call 不是合法 JSON：{e.msg}"
    if not isinstance(obj, dict):
        return None, "tool_call 必须是 JSON 对象"
    name = obj.get("name")
    args = obj.get("arguments", {})
    if not isinstance(name, str) or not name:
        return None, "tool_call 缺少字符串字段 name"
    if isinstance(args, str):  # 有的模型把 arguments 写成 JSON 字符串
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            return None, "arguments 是字符串但不是合法 JSON"
    if not isinstance(args, dict):
        return None, "arguments 必须是 JSON 对象"
    extra = set(obj) - {"name", "arguments"}
    if extra:
        return None, f"tool_call 有多余字段 {sorted(extra)}"
    return {"name": name, "arguments": args}, None


def parse_assistant(text: str) -> ParsedAssistant:
    """把助手生成的文本（不含 `<|im_start|>assistant\\n`）解析成 {content, tool_calls}。

    与 `render` 互为逆操作：`parse_assistant(渲染出的助手文本)` 还原出原消息。
    解析失败的地方记在 `errors` 里，而不是抛异常——模型输出什么都有可能。
    """
    out = ParsedAssistant()
    if IM_END in text:
        text = text[: text.index(IM_END)]
    if text.startswith("<think>"):
        end = text.find("</think>")
        if end < 0:
            out.errors.append("<think> 没有闭合")
            out.reasoning_content = text[len("<think>") :].strip()
            return out
        out.reasoning_content = text[len("<think>") : end].strip()
        text = text[end + len("</think>") :].lstrip("\n")
    matches = list(_TOOL_CALL_RE.finditer(text))
    if not matches:
        if "<tool_call>" in text or "</tool_call>" in text:
            out.errors.append("<tool_call> 标签不配对")
        out.content = text
        return out
    prefix = text[: matches[0].start()]
    if prefix.endswith("\n"):
        prefix = prefix[:-1]
    between = [text[a.end() : b.start()] for a, b in zip(matches, matches[1:])]
    suffix = text[matches[-1].end() :]
    for gap in between:
        if gap.strip():
            out.errors.append("两个 tool_call 之间有多余文本")
    if "<tool_call>" in suffix or "</tool_call>" in suffix:
        out.errors.append("<tool_call> 标签不配对")
    if "<tool_call>" in prefix or "</tool_call>" in prefix:
        out.errors.append("<tool_call> 标签嵌套或不配对")
    for m in matches:
        call, err = parse_tool_call_json(m.group(1))
        if err:
            out.errors.append(err)
        else:
            out.tool_calls.append(call)  # type: ignore[arg-type]
    out.content = prefix + (suffix if suffix.strip() else "")
    return out


# ---------------------------------------------------------------------------
# Jinja 模板（导出到 tokenizer_config.json 的 chat_template）
# ---------------------------------------------------------------------------

CHAT_TEMPLATE = (
    """{%- if tools %}
    {{- '<|im_start|>system\\n' }}
    {%- if messages[0].role == 'system' %}
        {{- messages[0].content + '\\n\\n' }}
    {%- endif %}
    {{- """
    + _jinja_str(TOOLS_PREAMBLE)
    + """ }}
    {%- for tool in tools %}
        {{- '\\n' }}
        {{- tool | tojson }}
    {%- endfor %}
    {{- """
    + _jinja_str(TOOLS_POSTAMBLE + "<|im_end|>\n")
    + """ }}
{%- elif messages[0].role == 'system' %}
    {{- '<|im_start|>system\\n' + messages[0].content + '<|im_end|>\\n' }}
{%- endif %}
{%- for message in messages %}
    {%- if message.role == 'system' and loop.first %}
    {%- elif message.role == 'user' or message.role == 'system' %}
        {{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>\\n' }}
    {%- elif message.role == 'assistant' %}
        {%- if message.content is string %}
            {%- set content = message.content %}
        {%- else %}
            {%- set content = '' %}
        {%- endif %}
        {{- '<|im_start|>assistant\\n' }}
        {%- if enable_thinking is defined and enable_thinking and message.reasoning_content is string and message.reasoning_content %}
            {{- '<think>\\n' + message.reasoning_content.strip() + '\\n</think>\\n\\n' }}
        {%- endif %}
        {{- content }}
        {%- if message.tool_calls %}
            {%- for tool_call in message.tool_calls %}
                {%- if (loop.first and content) or (not loop.first) %}
                    {{- '\\n' }}
                {%- endif %}
                {%- if tool_call.function is defined and tool_call.function %}
                    {%- set tool_call = tool_call.function %}
                {%- endif %}
                {{- '<tool_call>\\n{"name": ' + (tool_call.name | tojson) + ', "arguments": ' }}
                {%- if tool_call.arguments is string %}
                    {{- tool_call.arguments }}
                {%- else %}
                    {{- tool_call.arguments | tojson }}
                {%- endif %}
                {{- '}\\n</tool_call>' }}
            {%- endfor %}
        {%- endif %}
        {{- '<|im_end|>\\n' }}
    {%- elif message.role == 'tool' %}
        {%- if loop.first or messages[loop.index0 - 1].role != 'tool' %}
            {{- '<|im_start|>user' }}
        {%- endif %}
        {{- '\\n<tool_response>\\n' }}
        {%- if message.content is string %}
            {{- message.content }}
        {%- else %}
            {{- message.content | tojson }}
        {%- endif %}
        {{- '\\n</tool_response>' }}
        {%- if loop.last or messages[loop.index0 + 1].role != 'tool' %}
            {{- '<|im_end|>\\n' }}
        {%- endif %}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\\n' }}
{%- endif %}
"""
)
