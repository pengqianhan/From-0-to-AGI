"""Chapter 16 · Minimal code 1: write a ChatML chat template by hand (with tool calls).

A conversation (a list of messages with the roles system / user / assistant / tool) → one string.
The format is the same as the official Qwen3 template:
- Each message is `<|im_start|>role\\ncontent<|im_end|>\\n`.
- The tool list goes into <tools></tools> in the system message.
- The assistant writes each call as <tool_call>{json}</tool_call>.
- Each tool result is wrapped in <tool_response> and goes into one user turn.

The renderer also marks each segment: "is it in the loss at training time?" Only the text that
the assistant writes is in the loss (with the tool calls and with <|im_end|>).
At the end, a parity check with the production code zero/post/chat.py: the two rendered strings
must be identical, character for character.

Run: uv run python chapters/16-sft/code/01_chat_template.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repository root: the parity check at the end imports zero

IM_START, IM_END = "<|im_start|>", "<|im_end|>"

# Fixed text of the tool instructions (identical to the chat_template of Qwen3)
TOOLS_HEAD = (
    "# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
    "You are provided with function signatures within <tools></tools> XML tags:\n<tools>"
)
TOOLS_TAIL = (
    "\n</tools>\n\nFor each function call, return a json object with function name and arguments "
    "within <tool_call></tool_call> XML tags:\n<tool_call>\n"
    '{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call>'
)

# A real training sample: a reference trajectory from zero/post/envs/tool_env.py (line 1 of
# out/smoke/sft/train.jsonl). The tool list of the original sample also has convert_units and
# weekday, which the sample does not use. We keep only get_weather to make the printout short.
TOOLS = [
    {"type": "function", "function": {"name": "get_weather", "description": "查询城市今天的天气",
     "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                    "required": ["city"]}}},
]
MESSAGES = [
    {"role": "system", "content": "你是一个会使用工具的助手。需要时调用工具，拿到结果后用一句话回答。"},
    {"role": "user", "content": "成都和广州今天哪个更热？"},
    {"role": "assistant", "content": "", "tool_calls": [
        {"name": "get_weather", "arguments": {"city": "成都"}},
        {"name": "get_weather", "arguments": {"city": "广州"}}]},
    {"role": "tool", "content": '{"city": "成都", "condition": "阴", "temp_c": 21, "humidity": 65}'},
    {"role": "tool", "content": '{"city": "广州", "condition": "雷阵雨", "temp_c": 31, "humidity": 85}'},
    {"role": "assistant", "content": "广州更热（成都 21°C，广州 31°C）。"},
]


def tojson(x) -> str:
    return json.dumps(x, ensure_ascii=False)  # keep Chinese characters unescaped, as the tojson filter of transformers does


def render(messages, tools=None, add_generation_prompt=False):
    """Return [(text, role, in_loss), ...]. Join the texts to get the full string for the model."""
    segs = []
    msgs = list(messages)
    has_sys = msgs and msgs[0]["role"] == "system"
    if tools:  # the tool list goes into the system turn
        body = (msgs[0]["content"] + "\n\n" if has_sys else "") + TOOLS_HEAD
        body += "".join("\n" + tojson(t) for t in tools) + TOOLS_TAIL
        segs.append((f"{IM_START}system\n{body}{IM_END}\n", "system", False))
    elif has_sys:
        segs.append((f"{IM_START}system\n{msgs[0]['content']}{IM_END}\n", "system", False))
    for i, m in enumerate(msgs):
        role = m["role"]
        if role == "system" and i == 0:
            continue
        if role in ("user", "system"):
            segs.append((f"{IM_START}{role}\n{m['content']}{IM_END}\n", role, False))
        elif role == "assistant":
            segs.append((f"{IM_START}assistant\n", "assistant", False))  # role header: the prompt gives it, so do not learn it
            text = m.get("content") or ""
            for j, tc in enumerate(m.get("tool_calls") or []):
                if text or j > 0:
                    text += "\n"
                text += ('<tool_call>\n{"name": ' + tojson(tc["name"]) + ', "arguments": '
                         + tojson(tc["arguments"]) + "}\n</tool_call>")
            segs.append((text + IM_END, "assistant", True))  # ← only this segment is in the loss (with <|im_end|>)
            segs.append(("\n", "assistant", False))
        elif role == "tool":  # consecutive tool results go into the same user turn
            first = i == 0 or msgs[i - 1]["role"] != "tool"
            last = i == len(msgs) - 1 or msgs[i + 1]["role"] != "tool"
            s = (f"{IM_START}user" if first else "") + "\n<tool_response>\n" + m["content"]
            s += "\n</tool_response>" + (f"{IM_END}\n" if last else "")
            segs.append((s, "tool", False))
    if add_generation_prompt:
        segs.append((f"{IM_START}assistant\n", "assistant", False))
    return segs


def parse_tool_calls(text: str):
    """Model output → list of tool calls (the inverse of rendering)."""
    return [json.loads(b) for b in re.findall(r"<tool_call>\n(.*?)\n</tool_call>", text, re.S)]


if __name__ == "__main__":
    segs = render(MESSAGES, TOOLS)
    text = "".join(s for s, _, _ in segs)
    print("=" * 30, "Rendered result (the full string for the model)", "=" * 30)
    print(text)
    print("=" * 30, "Per segment: in the loss? / role", "=" * 30)
    for s, role, train in segs:
        print(f"{'★ in loss' if train else '  no loss'}  {role:9s} {s!r:.70}")
    n_train = sum(len(s) for s, _, t in segs if t)
    print(f"\nTotal characters: {len(text)}. In the loss: {n_train} ({100 * n_train / len(text):.1f}%)")
    by_role = {}
    for s, role, _ in segs:
        by_role[role] = by_role.get(role, 0) + len(s)
    print("Characters per role:", by_role)

    # At inference time, the prompt ends with "<|im_start|>assistant\n". The model continues from there.
    prompt = "".join(s for s, _, _ in render(MESSAGES[:2], TOOLS, add_generation_prompt=True))
    print("\nLast 40 characters of the prompt at inference time:", repr(prompt[-40:]))

    # Parse: rendered assistant text → the tool calls again
    first_reply = [s for s, _, t in segs if t][0]
    print("Parsed first assistant reply:", parse_tool_calls(first_reply))

    # Parity check with the production code
    try:
        from zero.post.chat import render_text
    except ImportError:
        print("\n(Package zero not found, so there is no parity check.)")
    else:
        same = render_text(MESSAGES, TOOLS) == text
        same_gen = render_text(MESSAGES[:2], TOOLS, add_generation_prompt=True) == prompt
        print(f"\nParity check with zero.post.chat.render_text: full conversation {'same' if same else 'DIFFERENT'}, "
              f"generation prompt {'same' if same_gen else 'DIFFERENT'}")
