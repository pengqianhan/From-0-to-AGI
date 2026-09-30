"""本地工具调用命令行助手（对应第 20 章：“真正可用”的直接证明）。

    uv run python -m zero.demo.cli --model out/tiny/grpo/ckpt --root .          # 交互模式
    uv run python -m zero.demo.cli --model out/tiny/hf_chat --once "3 * (4 + 5) 等于多少？"

模型可以是 zero 的 checkpoint 目录，也可以是导出的 Hugging Face 目录。循环：

    用户输入 → 套对话模板生成 → 解析 <tool_call> → 在本地执行 → 结果作为 tool 消息喂回 → …
    → 直到模型给出不含工具调用的回答（最多 --max-turns 轮）

本地工具：

- `calculator(expression)`：算术（AST 白名单求值，不用 eval）；
- `date_add` / `days_between` / `weekday`：日期计算；`today()`：今天的日期（读系统时钟）；
- `search_files(query, glob="*")`：在 `--root` 目录里按文件名和内容搜索，只返回前几条匹配；
  路径限制在 root 之内（解析后的真实路径必须在 root 下，防止 `../` 逃逸），不跟随指向外部的符号链接，
  单个文件只读前 1 MB。

**tiny 模型的输出基本是乱的**（1.3M 参数、几分钟 CPU 训练），这个 demo 在第一步只证明"加载 → 生成 →
解析 → 执行 → 喂回"这条链路是通的；真正好用要等第二步训练出的主线模型。
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
    """在 root 下找文件名或内容包含 query 的文件（大小写不敏感），返回前 MAX_RESULTS 条。"""
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
    """处理一轮用户输入（messages 已经含这条 user 消息），原地追加助手/工具消息，返回最终回答文本。"""
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
    return "（达到最大轮数，没有得到最终回答）"


def main(argv: list[str] | None = None) -> None:
    import torch

    from zero.post.common import chat_complete, load_policy

    ap = argparse.ArgumentParser(description="本地工具调用助手（第 20 章）")
    ap.add_argument("--model", required=True, help="zero checkpoint 目录或 HF 目录")
    ap.add_argument("--root", default=".", help="search_files 的搜索根目录")
    ap.add_argument("--once", default="", help="只问一个问题就退出（非交互）")
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
        tag = {"call": "→ 调用", "result": "← 结果", "format_error": "✗ 格式错误"}[kind]
        print(f"  {tag} {json.dumps(obj, ensure_ascii=False)[:300]}")

    messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM}]
    questions = [args.once] if args.once else None
    print(
        f"[zero demo] 模型 {args.model}（{model.num_params() / 1e6:.1f}M 参数），搜索根目录 {root.resolve()}"
    )
    while True:
        if questions is not None:
            if not questions:
                break
            q = questions.pop(0)
            print(f"你：{q}")
        else:
            try:
                q = input("你：").strip()
            except EOFError:
                break
            if q in ("", "exit", "quit"):
                break
        messages.append({"role": "user", "content": q})
        ans = chat_turn(gen, messages, root, args.max_turns, show)
        print(f"助手：{ans}")


if __name__ == "__main__":
    main()
