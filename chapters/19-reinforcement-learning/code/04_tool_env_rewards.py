"""Chapter 19 · 04: see how the production scorer zero/post/envs/tool_env.py scores different outputs
(runs in seconds).

The "naive reward / fixed reward" in the minimal code 03 is a toy version. The main-line model uses
score_tool_calls. Here we take one calculator task and one chat task. We give the scorer a normal output
and some typical hacking outputs, and print the score and the reason for each.

    uv run python chapters/19-reinforcement-learning/code/04_tool_env_rewards.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # the repository root

from zero.post.chat import format_tool_call  # noqa: E402
from zero.post.envs.tool_env import generate_tasks, safe_eval, score_tool_calls  # noqa: E402


def main() -> None:
    tasks = generate_tasks(200, seed=0, split="train")
    calc = next(t for t in tasks if t.gold_calls and t.gold_calls[0]["name"] == "calculator")
    chat = next(t for t in tasks if not t.gold_calls)
    gold = calc.gold_calls[0]
    good = format_tool_call(gold)
    bare = json.dumps(gold, ensure_ascii=False)
    answer = str(safe_eval(gold["arguments"]["expression"]))  # the answer that the model calculates "in its head"

    print(f"Tool task: {calc.query}    gold call: {json.dumps(gold, ensure_ascii=False)}")
    cases = {
        "correct call": good,
        "broken JSON in tags": '<tool_call>{"name": "calculator", "arguments": {"expression": </tool_call>',
        "bare JSON (guard #8)": bare,
        "answer only (guard #2)": format_tool_call({"name": "calculator", "arguments": {"expression": answer}}),
        "call twice (guard #1)": good + "\n" + good,
        "fake result (guard #3)": good + "\n<tool_response>{\"result\": " + answer + "}</tool_response>",
    }
    for name, text in cases.items():
        r = score_tool_calls(calc, text)
        print(f"  {name:<22} reward {r.total:+.2f}  {'; '.join(r.details[:1])}")

    print(f"\nChat task: {chat.query}")
    for name, text in {"normal answer": "不客气！", "empty reply": "", "bare JSON": bare, "random tool call": good}.items():
        r = score_tool_calls(chat, text)
        print(f"  {name:<22} reward {r.total:+.2f}  {'; '.join(r.details[:1])}")


if __name__ == "__main__":
    main()
