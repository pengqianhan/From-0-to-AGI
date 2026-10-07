"""Command-line assistant with local tool calls (Chapter 20: the direct proof that the model is "really usable").

    uv run python -m zero.demo.cli --model out/tiny/grpo/ckpt --root .          # interactive mode
    uv run python -m zero.demo.cli --model out/tiny/hf_chat --once "3 * (4 + 5) 等于多少？"

The model can be a zero checkpoint directory or an exported Hugging Face directory. The loop:

    user input → generate with the chat template → parse <tool_call> → run it locally →
    send the result back as a tool message → …
    → until the model gives an answer without a tool call (at most --max-turns turns)

Local tools:

- `calculator(expression)`: arithmetic (evaluation with an AST allowlist, no eval).
- `date_add` / `days_between` / `weekday`: date calculations; `today()`: today's date (from the
  system clock).
- `search_files(query, glob="*")`: search the `--root` directory by file name and content, and
  return only the first matches. The paths stay inside root (the resolved real path must be
  below root, which stops a `../` escape). It does not follow symbolic links that point outside,
  and it reads only the first 1 MB of each file.

**The output of the tiny model is mostly noise** (1.3M parameters, a few minutes of CPU training).
In Step 1, this demo shows only that the chain "load → generate → parse → run → send back" works.
For a really useful assistant, wait for the main-line model of Step 2.

The model sees some strings in this file as input: the system prompt, the tool descriptions, and
the tool results (including the `ToolError` messages and the "match" value). These strings are
data. They stay in Chinese, the same as in zero/post/envs/tool_env.py.
"""

from __future__ import annotations

import argparse
import datetime as dt
import fnmatch
import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from zero.post.chat import parse_assistant
from zero.post.envs.tool_env import TOOLS, ToolError, execute_call

MAX_FILE_BYTES = 1 << 20
MAX_RESULTS = 8

DEMO_TOOLS: list[dict[str, Any]] = [
    TOOLS["calculator"],
    TOOLS["date_add"],
    TOOLS["days_between"],
    TOOLS["weekday"],
    {
        "type": "function",
        "function": {
            "name": "today",
            "description": "今天的日期",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_files",
            "description": "在本地目录里按文件名和内容搜索",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "glob": {"type": "string", "description": "文件名通配，如 *.py"},
                },
                "required": ["query"],
            },
        },
    },
]
SYSTEM = (
    "你是一个本地助手，可以使用计算器、日期工具和文件搜索。需要时调用工具，拿到结果后简洁回答。"
)


def search_files(root: Path, query: str, glob: str = "*") -> dict[str, Any]:
    """Find files below root whose name or content contains query (case-insensitive); return the first MAX_RESULTS."""
    if not query:
        raise ToolError("query 不能为空")
    root = root.resolve()
    q = query.lower()
    hits: list[dict[str, Any]] = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        for fn in filenames:
            p = Path(dirpath) / fn
            try:
                real = p.resolve()
            except OSError:
                continue
            if not real.is_relative_to(root) or not fnmatch.fnmatch(fn, glob):
                continue
            rel = str(real.relative_to(root))
            if q in fn.lower():
                hits.append({"path": rel, "match": "文件名"})
            else:
                try:
                    with open(real, "rb") as f:
                        text = f.read(MAX_FILE_BYTES).decode("utf-8", errors="ignore")
                except OSError:
                    continue
                for i, line in enumerate(text.splitlines(), 1):
                    if q in line.lower():
                        hits.append({"path": rel, "line": i, "text": line.strip()[:120]})
                        break
            if len(hits) >= MAX_RESULTS:
                return {"results": hits, "truncated": True}
    return {"results": hits, "truncated": False}


def execute_demo_tool(name: str, args: dict[str, Any], root: Path) -> dict[str, Any]:
    try:
        if name == "today":
            return {"date": dt.date.today().isoformat()}
        if name == "search_files":
            if not isinstance(args.get("query"), str):
                raise ToolError("query 必须是字符串")
            return search_files(root, args["query"], str(args.get("glob") or "*"))
        if name in ("calculator", "date_add", "days_between", "weekday"):
            return execute_call(name, args)
        raise ToolError(f"没有这个工具：{name}")
    except ToolError as e:
        return {"error": str(e)}


def chat_turn(
    generate_fn: Callable[[list[dict[str, Any]], list[dict[str, Any]]], str],
    messages: list[dict[str, Any]],
    root: Path,
    max_turns: int = 4,
    on_event: Callable[[str, Any], None] | None = None,
) -> str:
    """Process one user input (messages already contains this user message).

    Append the assistant/tool messages in place, and return the text of the final answer.
    """
    emit = on_event or (lambda *_: None)
    for _ in range(max_turns):
        text = generate_fn(messages, DEMO_TOOLS)
        parsed = parse_assistant(text)
        if parsed.errors:
            emit("format_error", parsed.errors)
            messages.append({"role": "assistant", "content": text})
            return text
        messages.append(parsed.to_message())
        if not parsed.tool_calls:
            return parsed.content
        for c in parsed.tool_calls:
            emit("call", c)
            res = execute_demo_tool(c["name"], c["arguments"], root)
            emit("result", res)
            messages.append({"role": "tool", "content": json.dumps(res, ensure_ascii=False)})
    return "(Reached the maximum number of turns; no final answer)"


def main(argv: list[str] | None = None) -> None:
    import torch

    from zero.post.common import chat_complete, load_policy

    ap = argparse.ArgumentParser(description="Assistant with local tool calls (Chapter 20)")
    ap.add_argument("--model", required=True, help="zero checkpoint directory or HF directory")
    ap.add_argument("--root", default=".", help="root directory for search_files")
    ap.add_argument("--once", default="", help="ask only this one question, then exit (not interactive)")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-turns", type=int, default=4)
    ap.add_argument("--threads", type=int, default=1)
    args = ap.parse_args(argv)
    torch.set_num_threads(args.threads)
    model, tok = load_policy(args.model)
    model.eval()
    root = Path(args.root)

    def gen(messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> str:
        return chat_complete(model, tok, messages, tools, args.max_new_tokens, args.temperature)[0]

    def show(kind: str, obj: Any) -> None:
        tag = {"call": "→ call", "result": "← result", "format_error": "✗ format error"}[kind]
        print(f"  {tag} {json.dumps(obj, ensure_ascii=False)[:300]}")

    messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM}]
    questions = [args.once] if args.once else None
    print(
        f"[zero demo] model {args.model} ({model.num_params() / 1e6:.1f}M parameters), search root {root.resolve()}"
    )
    while True:
        if questions is not None:
            if not questions:
                break
            q = questions.pop(0)
            print(f"You: {q}")
        else:
            try:
                q = input("You: ").strip()
            except EOFError:
                break
            if q in ("", "exit", "quit"):
                break
        messages.append({"role": "user", "content": q})
        ans = chat_turn(gen, messages, root, args.max_turns, show)
        print(f"Assistant: {ans}")


if __name__ == "__main__":
    main()
