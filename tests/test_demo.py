"""本地 demo：工具执行、文件搜索限制在根目录内、生成 → 解析 → 执行 → 喂回的循环（第 20 章）。"""

from __future__ import annotations

import json
import os
from pathlib import Path

from zero.demo.cli import chat_turn, execute_demo_tool, search_files
from zero.post.chat import format_tool_call


def test_search_files_confined_to_root(tmp_path: Path) -> None:
    root = tmp_path / "root"
    (root / "sub").mkdir(parents=True)
    (root / "sub" / "notes.md").write_text("第一行\nTODO: 买牛奶\n")
    (root / "todo_list.txt").write_text("x")
    outside = tmp_path / "secret.txt"
    outside.write_text("TODO: 不该被搜到")
    os.symlink(outside, root / "link.txt")
    r = search_files(root, "todo")
    paths = {h["path"] for h in r["results"]}
    assert paths == {"todo_list.txt", "sub/notes.md"}
    hit = next(h for h in r["results"] if h["path"] == "sub/notes.md")
    assert hit["line"] == 2
    assert search_files(root, "todo", glob="*.md")["results"][0]["path"] == "sub/notes.md"
    assert "error" in execute_demo_tool("search_files", {"query": ""}, root)
    assert execute_demo_tool("calculator", {"expression": "2 ** 10"}, root) == {"result": 1024}
    assert "error" in execute_demo_tool("rm", {}, root)


def test_chat_turn_loop(tmp_path: Path) -> None:
    calls = []

    def fake_generate(messages, tools):  # noqa: ANN001, ANN202
        calls.append(messages[-1]["role"])
        if messages[-1]["role"] == "user":
            return format_tool_call({"name": "calculator", "arguments": {"expression": "12 * (3 + 4)"}})
        res = json.loads(messages[-1]["content"])
        return f"结果是 {res['result']}。"

    msgs = [{"role": "user", "content": "12 * (3 + 4)?"}]
    events = []
    ans = chat_turn(fake_generate, msgs, tmp_path, on_event=lambda k, o: events.append(k))
    assert ans == "结果是 84。" and events == ["call", "result"] and calls == ["user", "tool"]
    assert [m["role"] for m in msgs] == ["user", "assistant", "tool", "assistant"]
    bad = chat_turn(lambda m, t: '<tool_call>{"name"', [{"role": "user", "content": "x"}], tmp_path)
    assert bad.startswith("<tool_call>")


def test_demo_cli_runs_with_tiny_model(tiny_ckpt, capsys) -> None:  # noqa: ANN001
    from zero.demo.cli import main

    main(["--model", str(tiny_ckpt), "--once", "你好", "--max-new-tokens", "8", "--max-turns", "2"])
    out = capsys.readouterr().out
    assert "你：你好" in out and "助手：" in out
