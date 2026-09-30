"""第 16 章 · 极简代码 1：手写一个 ChatML 对话模板（含工具调用）

一段对话（system / user / assistant / tool 几种角色的消息列表）→ 一整串文本。
格式与 Qwen3 的官方模板相同（`<|im_start|>角色\\n内容<|im_end|>\\n`，工具写在 system 里的
<tools></tools> 中，助手的调用写成 <tool_call>{json}</tool_call>，工具结果包在 <tool_response> 里，
放进一个 user 轮）。

渲染的同时给每一段标上"训练时要不要算 loss"：只有助手自己说的话（含工具调用、含 <|im_end|>）算。
最后和生产级的 zero/post/chat.py 对拍：两边渲染出的字符串必须逐字相同。

运行：uv run python chapters/16-sft/code/01_chat_template.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # 仓库根目录，最后对拍时导入 zero

IM_START, IM_END = "<|im_start|>", "<|im_end|>"

# 工具说明的固定文字（与 Qwen3 的 chat_template 一字不差）
TOOLS_HEAD = (
    "# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
    "You are provided with function signatures within <tools></tools> XML tags:\n<tools>"
)
TOOLS_TAIL = (
    "\n</tools>\n\nFor each function call, return a json object with function name and arguments "
    "within <tool_call></tool_call> XML tags:\n<tool_call>\n"
    '{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call>'
)

# 一条真实的训练样本：zero/post/envs/tool_env.py 生成的标准解答轨迹（out/smoke/sft/train.jsonl 第 1 行；
# 原样本的工具清单里还有 convert_units、weekday 两个没用到的工具，这里为了打印简短只留 get_weather）
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
    return json.dumps(x, ensure_ascii=False)  # 中文不转义，与 transformers 的 tojson 过滤器一致


def render(messages, tools=None, add_generation_prompt=False):
    """返回 [(文本, 角色, 是否算 loss), ...]；把文本拼起来就是喂给模型的整串。"""
    segs = []
    msgs = list(messages)
    has_sys = msgs and msgs[0]["role"] == "system"
    if tools:  # 工具清单写进 system 轮
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
            segs.append((f"{IM_START}assistant\n", "assistant", False))  # 角色头：提示词给的，不学
            text = m.get("content") or ""
            for j, tc in enumerate(m.get("tool_calls") or []):
                if text or j > 0:
                    text += "\n"
                text += ('<tool_call>\n{"name": ' + tojson(tc["name"]) + ', "arguments": '
                         + tojson(tc["arguments"]) + "}\n</tool_call>")
            segs.append((text + IM_END, "assistant", True))  # ← 只有这一段算 loss（含 <|im_end|>）
            segs.append(("\n", "assistant", False))
        elif role == "tool":  # 连续几条工具结果合并进同一个 user 轮
            first = i == 0 or msgs[i - 1]["role"] != "tool"
            last = i == len(msgs) - 1 or msgs[i + 1]["role"] != "tool"
            s = (f"{IM_START}user" if first else "") + "\n<tool_response>\n" + m["content"]
            s += "\n</tool_response>" + (f"{IM_END}\n" if last else "")
            segs.append((s, "tool", False))
    if add_generation_prompt:
        segs.append((f"{IM_START}assistant\n", "assistant", False))
    return segs


def parse_tool_calls(text: str):
    """模型输出 → 工具调用列表（渲染的逆操作）。"""
    return [json.loads(b) for b in re.findall(r"<tool_call>\n(.*?)\n</tool_call>", text, re.S)]


if __name__ == "__main__":
    segs = render(MESSAGES, TOOLS)
    text = "".join(s for s, _, _ in segs)
    print("=" * 30, "渲染结果（喂给模型的整串文本）", "=" * 30)
    print(text)
    print("=" * 30, "逐段：角色 / 是否算 loss", "=" * 30)
    for s, role, train in segs:
        print(f"{'★ 算  ' if train else '  不算'}  {role:9s} {s!r:.70}")
    n_train = sum(len(s) for s, _, t in segs if t)
    print(f"\n总字符 {len(text)}，其中算 loss 的 {n_train}（{100 * n_train / len(text):.1f}%）")
    by_role = {}
    for s, role, _ in segs:
        by_role[role] = by_role.get(role, 0) + len(s)
    print("各角色字符数：", by_role)

    # 推理时：提示词以 "<|im_start|>assistant\n" 结尾，模型从这里接着写
    prompt = "".join(s for s, _, _ in render(MESSAGES[:2], TOOLS, add_generation_prompt=True))
    print("\n推理时提示词的最后 40 个字符：", repr(prompt[-40:]))

    # 解析：渲染出的助手文本 → 还原出工具调用
    first_reply = [s for s, _, t in segs if t][0]
    print("解析第一条助手回复：", parse_tool_calls(first_reply))

    # 与生产级实现对拍
    try:
        from zero.post.chat import render_text
    except ImportError:
        print("\n（没找到 zero 包，跳过对拍）")
    else:
        same = render_text(MESSAGES, TOOLS) == text
        same_gen = render_text(MESSAGES[:2], TOOLS, add_generation_prompt=True) == prompt
        print(f"\n与 zero.post.chat.render_text 对拍：完整对话 {'一致' if same else '不一致'}，"
              f"生成提示 {'一致' if same_gen else '不一致'}")
