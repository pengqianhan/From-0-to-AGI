"""第 17 章 · 极简代码 4：拒绝采样 + 执行验证——多采几个，只留验证通过的

对工具调用任务（zero/post/envs/tool_env.py 的模拟 API），教师每个任务采样 K 次，然后逐关筛：

  关 1 格式：输出能解析（JSON 完整、标签配对、没有伪造工具结果）
  关 2 调用：函数名 + 规范化后的参数与标准调用一致（计算器另接受执行结果一致）→ 调用满分
  关 3 执行 + 回答：真的去执行工具、把结果喂回，教师写最终回答，要点必须都在 → score_final_answer
  关 4 去重：每个任务最多留 keep 条

这里的"教师"是一个**规则模拟器**，不是语言模型：它以固定概率犯几类真实教师常犯的错（参数写错、
选错工具、JSON 写坏、不调工具直接编答案、多调一个、工具结果对了但回答抄错）。这样能在 CPU 上
几秒内看到漏斗的每一关筛掉了什么。筛选本身用的就是生产级代码 zero.post.distill.teacher_trajectories。

最后看不需要工具的任务（打招呼、鼓励）：旧版验证器只检查"有没有乱调工具"，不检查回答内容——
冒烟测试里唯一"通过验证"的那条教师数据就是一句胡话。修好后，这类回答只拿 0.5 分、标记为"无法核对"，
不进蒸馏数据；编出题目里没有的数字直接 0 分。

运行：uv run python chapters/17-distillation/code/04_rejection_sampling.py      （约 5 秒）
"""

import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from zero.post.chat import format_tool_call, parse_assistant  # noqa: E402
from zero.post.distill import teacher_trajectories  # noqa: E402
from zero.post.envs.tool_env import (  # noqa: E402
    TOOLS,
    execute_safely,
    generate_tasks,
    score_final_answer,
    score_tool_calls,
)

N_TASKS, K, KEEP = 200, 8, 1
# 模拟教师第一轮输出的错误类型与概率（数字是随手设的，只为演示漏斗，不代表任何真实模型）
FIRST_TURN = [
    ("正确", 0.40),
    ("参数写错", 0.20),
    ("选错工具", 0.10),
    ("JSON 写坏", 0.10),
    ("不调工具直接答", 0.10),
    ("多调一个", 0.10),
]
FINAL_WRONG = 0.15  # 工具结果对了，最终回答却抄错数字的概率
NONSENSE = ["坚下云，气温 28°C。", "好的。", "Paris is sunny."]  # 不需要工具的任务上的胡话


def corrupt_args(args: dict, rng: random.Random) -> dict:
    a = dict(args)
    k = rng.choice(list(a))
    v = a[k]
    if isinstance(v, int | float) and not isinstance(v, bool):
        a[k] = v + rng.choice([1, 7, 10])
    elif k == "expression":
        a[k] = v.replace(v.strip()[0], str((int(v.strip()[0]) + 1) % 10) if v.strip()[0].isdigit() else "1", 1)
    elif k in ("date", "start_date", "end_date"):
        a[k] = v[:-1] + str((int(v[-1]) + 3) % 10)
    else:
        a[k] = v + "市" if not v.endswith("市") else "上海"
    return a


class SimTeacher:
    """和 LocalTeacher / OpenAITeacher 同样的接口：complete(messages, tools, n, seed) → n 段助手文本。"""

    def __init__(self, tasks: list) -> None:
        self.by_query = {t.query: t for t in tasks}
        self.first_kinds: Counter = Counter()

    def complete(self, messages: list, tools: list, n: int, seed: int) -> list[str]:
        rng = random.Random(seed)
        task = self.by_query[next(m["content"] for m in messages if m["role"] == "user")]
        if messages[-1]["role"] == "tool":  # 第二轮：看工具结果写最终回答
            if rng.random() < FINAL_WRONG:
                return ["结果是 12345。"] * n
            return [task.gold_answer] * n
        outs = []
        self.last_kinds = []
        for _ in range(n):
            if not task.gold_calls:  # 不需要工具的任务：一半正常回答，一半胡话
                kind = "正确" if rng.random() < 0.5 else "胡话"
                self.first_kinds[kind] += 1
                self.last_kinds.append(kind)
                outs.append(task.gold_answer if kind == "正确" else rng.choice(NONSENSE))
                continue
            kind = rng.choices([k for k, _ in FIRST_TURN], [p for _, p in FIRST_TURN])[0]
            self.first_kinds[kind] += 1
            self.last_kinds.append(kind)
            calls = [dict(c) for c in task.gold_calls]
            if kind == "参数写错":
                calls[0] = {"name": calls[0]["name"], "arguments": corrupt_args(calls[0]["arguments"], rng)}
            elif kind == "选错工具":
                offered = [t["function"]["name"] for t in task.tools if t["function"]["name"] != calls[0]["name"]]
                other = rng.choice(offered)
                calls[0] = {"name": other, "arguments": calls[0]["arguments"]}
            elif kind == "多调一个":
                calls.append(dict(calls[0]))
            if kind == "JSON 写坏":
                outs.append(format_tool_call(calls[0])[:-20])
            elif kind == "不调工具直接答":
                outs.append("答案是 42。")
            else:
                outs.append("\n".join(format_tool_call(c) for c in calls))
        return outs


def funnel(tasks: list, teacher: SimTeacher) -> dict:
    """极简版：把 teacher_trajectories 的判断逐关拆开计数（每个候选都走完，不提前停）。"""
    c = Counter()
    for i, task in enumerate(tasks):
        texts = teacher.complete(task.messages, task.tools, K, seed=i)
        kinds = list(teacher.last_kinds)
        for j, text in enumerate(texts):
            c["候选"] += 1
            c[("候选", kinds[j])] += 1
            r = score_tool_calls(task, text)
            if not r.format_ok:
                continue
            c["关1 格式正确"] += 1
            if r.total < 0.999:
                continue
            c["关2 调用正确"] += 1
            parsed = parse_assistant(text)
            msgs = [dict(m) for m in task.messages] + [parsed.to_message()]
            if parsed.tool_calls:
                for call in parsed.tool_calls:
                    res = execute_safely(call["name"], call["arguments"])
                    msgs.append({"role": "tool", "content": json.dumps(res, ensure_ascii=False)})
                final = teacher.complete(msgs, task.tools, 1, seed=i * 7919 + j)[0]
            else:
                final = text
            if not score_final_answer(task, final).answer_ok:
                continue
            c["关3 执行+回答正确"] += 1
            c[("通过", kinds[j])] += 1
            if kinds[j] == "参数写错":
                c[("参数写错却通过", task.kind)] += 1
    return c


def main() -> None:
    tasks = generate_tasks(N_TASKS, seed=17, split="train")
    kinds = Counter(t.kind for t in tasks)
    print(f"{N_TASKS} 个 tool_env 训练任务：" + "，".join(f"{k} {v}" for k, v in kinds.most_common()))
    print(f"可用工具 {len(TOOLS)} 个；每个任务采样 K = {K} 次，最多保留 {KEEP} 条\n")

    # 1) 极简版漏斗：每一关还剩多少
    teacher = SimTeacher(tasks)
    c = funnel(tasks, teacher)
    print("模拟教师第一轮输出的类型 → 最终通过验证的条数：")
    for k, v in teacher.first_kinds.most_common():
        print(f"  {k:<10}{v:>5} → {c[('通过', k)]:>4}")
    sneaky = {kk[1]: v for kk, v in c.items() if isinstance(kk, tuple) and kk[0] == "参数写错却通过"}
    print(f"  （'参数写错'却通过的，按任务类型：{sneaky}——执行结果与标准调用相同，见 README）")
    print("\n漏斗（逐关累计）：")
    total = c["候选"]
    for stage in ["候选", "关1 格式正确", "关2 调用正确", "关3 执行+回答正确"]:
        print(f"  {stage:<14}{c[stage]:>6}  ({c[stage] / total:6.1%})")

    # 2) 生产级：zero.post.distill.teacher_trajectories（每个任务保留 KEEP 条）
    teacher2 = SimTeacher(tasks)
    kept_rows, n_cand, covered = [], 0, 0
    for i, task in enumerate(tasks):
        kept, n = teacher_trajectories(teacher2, task, K, KEEP, seed=i)
        n_cand += n
        covered += bool(kept)
        kept_rows += [(task, m) for m in kept]
    print(f"\nzero.post.distill.teacher_trajectories：{n_cand} 个候选 → 保留 {len(kept_rows)} 条"
          f"（{covered}/{N_TASKS} 个任务至少有一条；关 4 每任务最多 {KEEP} 条）")
    print(f"  如果不筛，{total - c['关3 执行+回答正确']} / {total} = {1 - c['关3 执行+回答正确'] / total:.1%} 的候选没有通过验证"
          "（绝大多数是错的；不需要工具的任务无法核对，也算在里面）。")

    # 3) 不需要工具的任务：回答无法自动核对，不进蒸馏数据
    n_no_tool = sum(1 for t in tasks if not t.gold_calls)
    no_tool = [(t, m) for t, m in kept_rows if not t.gold_calls]
    print(f"\n不需要工具的任务 {n_no_tool} 个（{n_no_tool * K} 个候选，其中胡话 {teacher.first_kinds['胡话']} 个）"
          f"→ 保留 {len(no_tool)} 条。")
    print("  这类回答没有标准答案可核对（奖励 NO_TOOL_REWARD = 0.5，不算验证通过），"
          "\"该不调就不调\"的示范交给人工审过的 SFT 数据。")


if __name__ == "__main__":
    main()
