"""Chat template and tool-call format: render, parse, and loss mask.

Chapter 16 (SFT) uses this module. Chapters 19 and 20 also use it.

The format is the same as the Qwen3 ChatML template. Then transformers, vLLM, and llama.cpp can use
the exported model directly:

    <|im_start|>system
    {system prompt}

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
    What's the weather like in Beijing today?<|im_end|>
    <|im_start|>assistant
    <tool_call>
    {"name": "get_weather", "arguments": {"city": "Beijing"}}
    </tool_call><|im_end|>
    <|im_start|>user
    <tool_response>
    {"city": "Beijing", "condition": "sunny", "temp_c": 25}
    </tool_response><|im_end|>
    <|im_start|>assistant
    Beijing is sunny today, 25°C.<|im_end|>

Main points:

- The tools (JSON schema) go in the system turn, inside `<tools></tools>`, one tool on each line.
- An assistant tool call has the form `<tool_call>{"name": ..., "arguments": {...}}</tool_call>`
  (Hermes / Qwen style). vLLM with `--tool-call-parser hermes` parses it into structured tool_calls.
- A tool result is a message with `role="tool"`. Consecutive tool messages go into one user turn.
  Each result is inside its own `<tool_response>`.
- **loss mask**: only the text that the assistant "says" counts in the loss: from after
  `<|im_start|>assistant\\n` to `<|im_end|>`. This includes `<|im_end|>`, because the model must learn
  when to stop. It also includes the tokens of tool calls. The system prompt, the user text, and the
  tool results do not count.
- Thinking (`<think>…</think>`) is off by default. With `enable_thinking=False`, the template does not
  render `reasoning_content` in the history. One difference from Qwen3: when thinking is off, Qwen3
  adds an empty `<think>\\n\\n</think>\\n\\n` after the generation prompt. This template does not
  add it. Our model never saw the thinking format, so the empty block would be out-of-distribution input.

The same format has two implementations. They must give the same text, character by character.
`tests/test_chat.py` does a parity check with `apply_chat_template` of transformers:

1. Python: `render()`. Training uses it, and it also gives the loss mask.
2. Jinja: `CHAT_TEMPLATE`. The export writes it into `tokenizer_config.json`. Inference frameworks
   use it to build the prompt.
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
    """Same as the `tojson` filter in the transformers template environment.

    It does not escape non-ASCII characters, and it uses the default separators.
    """
    return json.dumps(x, ensure_ascii=False)


def _jinja_str(s: str) -> str:
    """Python string → Jinja single-quoted literal."""
    return "'" + s.replace("\\", "\\\\").replace("'", "\\'").replace("\n", "\\n") + "'"


# ---------------------------------------------------------------------------
# Rendering (Python)
# ---------------------------------------------------------------------------


def normalize_tool_call(tc: dict[str, Any]) -> dict[str, Any]:
    """OpenAI style {"type": "function", "function": {...}} or {"name", "arguments"} → {"name", "arguments"}."""
    if "function" in tc and isinstance(tc["function"], dict):
        tc = tc["function"]
    return {"name": tc.get("name"), "arguments": tc.get("arguments", {})}


def format_tool_call(tc: dict[str, Any]) -> str:
    tc = normalize_tool_call(tc)
    args = tc["arguments"]
    args_s = args if isinstance(args, str) else tojson(args)
    return (
        '<tool_call>\n{"name": '
        + tojson(tc["name"])
        + ', "arguments": '
        + args_s
        + "}\n</tool_call>"
    )


def _content(msg: dict[str, Any]) -> str:
    c = msg.get("content")
    if c is None:
        return ""
    return c if isinstance(c, str) else tojson(c)


def assistant_text(m: dict[str, Any], enable_thinking: bool = False) -> str:
    """Assistant message → the text that the model must generate.

    The text does not include `<|im_start|>assistant\\n` and `<|im_end|>`.
    """
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
    return body


def render_segments(
    messages: Sequence[dict[str, Any]],
    tools: Sequence[dict[str, Any]] | None = None,
    add_generation_prompt: bool = False,
    enable_thinking: bool = False,
) -> list[tuple[str, bool]]:
    """Render the conversation as segments (text, counts in the loss). Together they are the full prompt."""
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
            emit(assistant_text(m, enable_thinking) + IM_END, train=True)
            emit("\n")
        elif role == "tool":
            if i == 0 or msgs[i - 1].get("role") != "tool":
                emit(f"{IM_START}user")
            emit("\n<tool_response>\n" + _content(m) + "\n</tool_response>")
            if i == len(msgs) - 1 or msgs[i + 1].get("role") != "tool":
                emit(f"{IM_END}\n")
        else:
            raise ValueError(f"Unknown role: {role!r}")
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
    """Render a conversation.

    - Without a tokenizer: return (text, mask). The mask is a list of booleans, one per character.
    - With a tokenizer: return (ids, mask). The mask has one value per token. True means that the token
      is assistant output, and training must predict it (the loss mask of SFT).

    The tokenizer encodes the **full text** at one time. This is the same as an inference framework,
    which builds the prompt and then encodes it. Then the start character of each token gives its
    value in the character mask. We do not encode each segment separately and join the results,
    because the BPE merges at a segment boundary can be different.
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
    return "".join(
        s for s, _ in render_segments(messages, tools, add_generation_prompt, enable_thinking)
    )


def encode_prompt_response(
    messages: Sequence[dict[str, Any]],
    response: dict[str, Any] | Sequence[dict[str, Any]],
    tokenizer: Tokenizer,
    tools: Sequence[dict[str, Any]] | None = None,
    enable_thinking: bool = False,
) -> tuple[list[int], list[bool]]:
    """Prompt + response (one assistant message, or a list of next messages) → (ids, mask).

    The mask marks only the assistant tokens of the **response**. DPO (chosen / rejected) and the
    student samples of distillation use this function: the assistant tokens of earlier turns in the
    prompt do not count. This works because the text of render(prompt + response) always starts with
    render(prompt, add_generation_prompt=True)."""
    resp = [response] if isinstance(response, dict) else list(response)
    prefix = render_text(
        messages, tools, add_generation_prompt=True, enable_thinking=enable_thinking
    )
    segs = render_segments(list(messages) + resp, tools, False, enable_thinking)
    text = "".join(s for s, _ in segs)
    if not text.startswith(prefix):
        raise ValueError("The response must start with an assistant message")
    char_mask: list[bool] = []
    for s, train in segs:
        char_mask.extend([train] * len(s))
    for i in range(len(prefix)):
        char_mask[i] = False
    ids, offsets = tokenizer.encode_with_offsets(text)
    return ids, [char_mask[a] if a < len(char_mask) else False for a, _ in offsets]


# ---------------------------------------------------------------------------
# Parsing the assistant output
# ---------------------------------------------------------------------------

_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)


@dataclass
class ParsedAssistant:
    content: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    reasoning_content: str | None = None
    errors: list[str] = field(default_factory=list)  # format errors (the reward function uses them to reduce the score)

    def to_message(self) -> dict[str, Any]:
        m: dict[str, Any] = {"role": "assistant", "content": self.content}
        if self.tool_calls:
            m["tool_calls"] = [dict(tc) for tc in self.tool_calls]
        if self.reasoning_content is not None:
            m["reasoning_content"] = self.reasoning_content
        return m

    def __getitem__(self, key: str) -> Any:  # allows parsed["content"] / parsed["tool_calls"]
        return getattr(self, key)


def parse_tool_call_json(body: str) -> tuple[dict[str, Any] | None, str | None]:
    """Parse the JSON inside <tool_call>. Return (call, error)."""
    try:
        obj = json.loads(body.strip())
    except json.JSONDecodeError as e:
        return None, f"tool_call is not valid JSON: {e.msg}"
    if not isinstance(obj, dict):
        return None, "tool_call must be a JSON object"
    name = obj.get("name")
    args = obj.get("arguments", {})
    if not isinstance(name, str) or not name:
        return None, "tool_call has no string field name"
    if isinstance(args, str):  # some models write arguments as a JSON string
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            return None, "arguments is a string but not valid JSON"
    if not isinstance(args, dict):
        return None, "arguments must be a JSON object"
    extra = set(obj) - {"name", "arguments"}
    if extra:
        return None, f"tool_call has extra fields {sorted(extra)}"
    return {"name": name, "arguments": args}, None


def parse_assistant(text: str) -> ParsedAssistant:
    """Parse the text that the assistant generated into {content, tool_calls}.

    The text does not include `<|im_start|>assistant\\n`. This function is the inverse of `render`:
    `parse_assistant(rendered assistant text)` gives back the original message. Parse failures go into
    `errors`; the function does not raise an exception, because the model can output anything.
    """
    out = ParsedAssistant()
    if IM_END in text:
        text = text[: text.index(IM_END)]
    if text.startswith("<think>"):
        end = text.find("</think>")
        if end < 0:
            out.errors.append("<think> is not closed")
            out.reasoning_content = text[len("<think>") :].strip()
            return out
        out.reasoning_content = text[len("<think>") : end].strip()
        text = text[end + len("</think>") :].lstrip("\n")
    matches = list(_TOOL_CALL_RE.finditer(text))
    if not matches:
        if "<tool_call>" in text or "</tool_call>" in text:
            out.errors.append("Unpaired <tool_call> tags")
        out.content = text
        return out
    prefix = text[: matches[0].start()]
    if prefix.endswith("\n"):
        prefix = prefix[:-1]
    between = [text[a.end() : b.start()] for a, b in zip(matches, matches[1:])]
    suffix = text[matches[-1].end() :]
    for gap in between:
        if gap.strip():
            out.errors.append("Extra text between two tool_calls")
    if "<tool_call>" in suffix or "</tool_call>" in suffix:
        out.errors.append("Unpaired <tool_call> tags")
    if "<tool_call>" in prefix or "</tool_call>" in prefix:
        out.errors.append("Nested or unpaired <tool_call> tags")
    for m in matches:
        call, err = parse_tool_call_json(m.group(1))
        if err:
            out.errors.append(err)
        else:
            out.tool_calls.append(call)  # type: ignore[arg-type]
    out.content = prefix + (suffix if suffix.strip() else "")
    return out


# ---------------------------------------------------------------------------
# Jinja template (the chat_template that the export writes to tokenizer_config.json)
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
