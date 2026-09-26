"""第 19 章 · 04：看一眼生产级判分器 zero/post/envs/tool_env.py 怎么给各种输出打分（秒级）

极简代码 03 里的"天真奖励 / 修好的奖励"是玩具版；主线模型真正用的是 score_tool_calls。
这里拿一道计算器题和一道闲聊题，把正常输出和几种典型的作弊输出都喂进去，打印分数和理由。

    uv run python chapters/19-reinforcement-learning/code/04_tool_env_rewards.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # 仓库根目录

from zero.post.chat import format_tool_call  # noqa: E402
from zero.post.envs.tool_env import generate_tasks, safe_eval, score_tool_calls  # noqa: E402


def main() -> None:
    tasks = generate_tasks(200, seed=0, split="train")
    calc = next(t for t in tasks if t.gold_calls and t.gold_calls[0]["name"] == "calculator")
    chat = next(t for t in tasks if not t.gold_calls)
    gold = calc.gold_calls[0]
    good = format_tool_call(gold)
    bare = json.dumps(gold, ensure_ascii=False)
    answer = str(safe_eval(gold["arguments"]["expression"]))  # 模型"心算"出的答案

    print(f"工具题：{calc.query}    标准调用：{json.dumps(gold, ensure_ascii=False)}")
    cases = {
        "正确调用": good,
        "标签里的 JSON 坏了": '<tool_call>{"name": "calculator", "arguments": {"expression": </tool_call>',
        "去掉标签的裸 JSON（守卫 #8）": bare,
        "心算后只传答案（守卫 #2）": format_tool_call({"name": "calculator", "arguments": {"expression": answer}}),
        "同一个调用喷两遍（守卫 #1）": good + "\n" + good,
        "伪造工具结果（守卫 #3）": good + "\n<tool_response>{\"result\": " + answer + "}</tool_response>",
    }
    for name, text in cases.items():
        r = score_tool_calls(calc, text)
        print(f"  {name:<22} 奖励 {r.total:+.2f}  {'；'.join(r.details[:1])}")

    print(f"\n闲聊题：{chat.query}")
    for name, text in {"正常回答": "不客气！", "空回复": "", "裸 JSON": bare, "乱调用工具": good}.items():
        r = score_tool_calls(chat, text)
        print(f"  {name:<22} 奖励 {r.total:+.2f}  {'；'.join(r.details[:1])}")


if __name__ == "__main__":
    main()
