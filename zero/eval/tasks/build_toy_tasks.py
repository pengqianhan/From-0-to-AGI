"""Make the toy evaluation sets of the repository (Chapter 11). The output JSONL files are in the repository, so usually you do not need to run this again:

    uv run python -m zero.eval.tasks.build_toy_tasks

- toy_mc.jsonl: multiple choice (arithmetic + general knowledge in Chinese and English). It tests
  the code path of "few-shot log-likelihood choice".
- toy_gen.jsonl: generative exact match (arithmetic, repeat a word).
- tool_dev.jsonl: the tool-call dev set (the first 100 tasks of the fixed dev set in
  zero/post/envs/tool_env.py), frozen in a file. A later change to the task generator cannot
  change the evaluation set without notice.

All of these are toy data that we wrote, only to verify the evaluation code. The real benchmarks
(MMLU, C-Eval, BFCL, ...) run in Step 2 with the official evaluation frameworks, as
eval/PREREGISTRATION.md specifies.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent

FACTS = [
    ("中国的首都是哪座城市？", ["北京", "上海", "广州", "南京"], 0),
    ("一年有几个月？", ["十二个", "十个", "十一个", "十三个"], 0),
    ("水在标准大气压下的沸点是多少摄氏度？", ["100", "50", "0", "200"], 0),
    ("太阳从哪个方向升起？", ["东方", "西方", "南方", "北方"], 0),
    ("一周有几天？", ["七天", "五天", "六天", "八天"], 0),
    ("熊猫最爱吃什么？", ["竹子", "鱼", "苹果", "米饭"], 0),
    ("“春眠不觉晓”的下一句是？", ["处处闻啼鸟", "夜来风雨声", "花落知多少", "疑是地上霜"], 0),
    ("三角形有几条边？", ["三条", "四条", "五条", "两条"], 0),
    ("地球绕着什么转？", ["太阳", "月亮", "火星", "木星"], 0),
    ("冬天之后是哪个季节？", ["春天", "夏天", "秋天", "冬天"], 0),
    ("What is the capital of France?", ["Paris", "London", "Berlin", "Rome"], 0),
    ("How many legs does a spider have?", ["eight", "six", "four", "ten"], 0),
    ("What color is the sky on a clear day?", ["blue", "green", "red", "black"], 0),
    ("Which animal says 'moo'?", ["cow", "cat", "dog", "duck"], 0),
    ("What is the opposite of 'hot'?", ["cold", "warm", "big", "fast"], 0),
    ("Water freezes at how many degrees Celsius?", ["0", "100", "10", "50"], 0),
    ("How many days are in a week?", ["seven", "five", "ten", "three"], 0),
    ("Which planet do we live on?", ["Earth", "Mars", "Venus", "Jupiter"], 0),
    ("What do bees make?", ["honey", "milk", "silk", "paper"], 0),
    ("Who wrote 'Romeo and Juliet'?", ["Shakespeare", "Dickens", "Tolstoy", "Homer"], 0),
]


def build(seed: int = 0) -> None:
    rng = random.Random(seed)
    mc = []
    for i in range(20):
        a, b = rng.randint(2, 50), rng.randint(2, 50)
        ans = a + b
        choices = [ans, ans + 1, ans - 1, ans + 10]
        rng.shuffle(choices)
        mc.append(
            {
                "id": f"arith-{i}",
                "question": f"{a} + {b} = ?",
                "choices": [str(c) for c in choices],
                "answer": choices.index(ans),
            }
        )
    for i, (q, ch, a) in enumerate(FACTS):
        order = list(range(len(ch)))
        rng.shuffle(order)
        mc.append(
            {
                "id": f"fact-{i}",
                "question": q,
                "choices": [ch[j] for j in order],
                "answer": order.index(a),
            }
        )
    gen = []
    for i in range(20):
        a, b = rng.randint(1, 30), rng.randint(1, 30)
        gen.append({"id": f"add-{i}", "prompt": f"{a} + {b} =", "answer": str(a + b)})
    words = ["apple", "river", "moon", "花", "山", "code", "tiger", "海", "light", "春"]
    for i, w in enumerate(words):
        gen.append({"id": f"copy-{i}", "prompt": f"Repeat: {w} ->", "answer": w})

    from zero.post.envs.tool_env import dev_tasks

    tools = [t.to_dict() for t in dev_tasks(100)]
    for name, rows in (("toy_mc.jsonl", mc), ("toy_gen.jsonl", gen), ("tool_dev.jsonl", tools)):
        with open(HERE / name, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{name}: {len(rows)} rows")


if __name__ == "__main__":
    build()
