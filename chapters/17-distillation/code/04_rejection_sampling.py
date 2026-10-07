"""Chapter 17 · Minimal code 4: rejection sampling + execution check. Sample several times, keep only the verified ones

For tool-call tasks (the simulated APIs in zero/post/envs/tool_env.py), the teacher samples K times
for each task. Then a sequence of gates filters the samples:

  Gate 1, format: the output can be parsed (complete JSON, matched tags, no fake tool result)
  Gate 2, call: the function name + normalized arguments are the same as the gold call
          (for the calculator, the same execution result is also accepted) → full call score
  Gate 3, execution + answer: really run the tool and give the result back; the teacher writes
          the final answer, and all key points must be in it → score_final_answer
  Gate 4, deduplication: keep at most `keep` samples for each task

The "teacher" here is a **rule-based simulator**, not a language model. With fixed probabilities, it makes
the errors that real teachers often make: wrong arguments, wrong tool, broken JSON, an invented answer
with no tool call, one extra call, and a final answer that copies a wrong number from a correct tool result.
Then we can see in a few seconds on a CPU what each gate of the funnel removes. The filter itself is
the production code zero.post.distill.teacher_trajectories.

At the end, we look at the tasks that need no tool (greetings, encouragement). The old verifier only
checked "no unnecessary tool call". It did not check the content of the answer. In the smoke test,
the only teacher sample that "passed verification" was a nonsense sentence. After the fix, such an answer
gets only 0.5 and is marked "cannot be checked", so it does not go into the distillation data.
An answer with a number that is not in the question gets 0.

Run: uv run python chapters/17-distillation/code/04_rejection_sampling.py      (about 5 s)
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
# Error types and probabilities of the first-turn output of the simulated teacher.
# The numbers are arbitrary. They only demonstrate the funnel and do not represent a real model.
FIRST_TURN = [
    ("correct", 0.40),
    ("wrong args", 0.20),
    ("wrong tool", 0.10),
    ("bad JSON", 0.10),
    ("no call", 0.10),  # answers directly with no tool call
    ("extra call", 0.10),
]
FINAL_WRONG = 0.15  # probability that the tool result is correct but the final answer copies a wrong number
# Nonsense answers on tasks that need no tool. They are teacher outputs (data), so they stay in Chinese:
# "坚下云，气温 28°C。" = "<nonsense words>, temperature 28°C."; "好的。" = "OK."
NONSENSE = ["坚下云，气温 28°C。", "好的。", "Paris is sunny."]


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
        # Add "市" ("city") to a city name, for example "北京" → "北京市", or change it to "上海" (Shanghai)
        a[k] = v + "市" if not v.endswith("市") else "上海"
    return a


class SimTeacher:
    """The same interface as LocalTeacher / OpenAITeacher: complete(messages, tools, n, seed) → n assistant texts."""

    def __init__(self, tasks: list) -> None:
        self.by_query = {t.query: t for t in tasks}
        self.first_kinds: Counter = Counter()

    def complete(self, messages: list, tools: list, n: int, seed: int) -> list[str]:
        rng = random.Random(seed)
        task = self.by_query[next(m["content"] for m in messages if m["role"] == "user")]
        if messages[-1]["role"] == "tool":  # second turn: read the tool result and write the final answer
            if rng.random() < FINAL_WRONG:
                return ["结果是 12345。"] * n  # teacher output (data): "The result is 12345."
            return [task.gold_answer] * n
        outs = []
        self.last_kinds = []
        for _ in range(n):
            if not task.gold_calls:  # a task that needs no tool: half normal answers, half nonsense
                kind = "correct" if rng.random() < 0.5 else "nonsense"
                self.first_kinds[kind] += 1
                self.last_kinds.append(kind)
                outs.append(task.gold_answer if kind == "correct" else rng.choice(NONSENSE))
                continue
            kind = rng.choices([k for k, _ in FIRST_TURN], [p for _, p in FIRST_TURN])[0]
            self.first_kinds[kind] += 1
            self.last_kinds.append(kind)
            calls = [dict(c) for c in task.gold_calls]
            if kind == "wrong args":
                calls[0] = {"name": calls[0]["name"], "arguments": corrupt_args(calls[0]["arguments"], rng)}
            elif kind == "wrong tool":
                offered = [t["function"]["name"] for t in task.tools if t["function"]["name"] != calls[0]["name"]]
                other = rng.choice(offered)
                calls[0] = {"name": other, "arguments": calls[0]["arguments"]}
            elif kind == "extra call":
                calls.append(dict(calls[0]))
            if kind == "bad JSON":
                outs.append(format_tool_call(calls[0])[:-20])
            elif kind == "no call":
                outs.append("答案是 42。")  # teacher output (data): "The answer is 42."
            else:
                outs.append("\n".join(format_tool_call(c) for c in calls))
        return outs


def funnel(tasks: list, teacher: SimTeacher) -> dict:
    """Minimal version: split the checks of teacher_trajectories into gates and count each gate.
    Each candidate goes through all gates (no early stop).

    The keys of the counter for the gates stay in Chinese, because the video (video/scenes.py) reads them:
    "候选" = candidates, "关1 格式正确" = gate 1, correct format, "关2 调用正确" = gate 2, correct call,
    "关3 执行+回答正确" = gate 3, correct execution + answer."""
    c = Counter()
    for i, task in enumerate(tasks):
        texts = teacher.complete(task.messages, task.tools, K, seed=i)
        kinds = list(teacher.last_kinds)
        for j, text in enumerate(texts):
            c["候选"] += 1
            c[("candidate", kinds[j])] += 1
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
            c[("passed", kinds[j])] += 1
            if kinds[j] == "wrong args":
                c[("wrong args but passed", task.kind)] += 1
    return c


# Display names of the gates for the printed output
STAGE_EN = {"候选": "candidates", "关1 格式正确": "gate 1 format", "关2 调用正确": "gate 2 call",
            "关3 执行+回答正确": "gate 3 answer"}


def main() -> None:
    tasks = generate_tasks(N_TASKS, seed=17, split="train")
    kinds = Counter(t.kind for t in tasks)
    print(f"{N_TASKS} tool_env training tasks: " + ", ".join(f"{k} {v}" for k, v in kinds.most_common()))
    print(f"{len(TOOLS)} tools available; K = {K} samples for each task, keep at most {KEEP}\n")

    # 1) Minimal funnel: how many candidates are left after each gate
    teacher = SimTeacher(tasks)
    c = funnel(tasks, teacher)
    print("Type of the first-turn output of the simulated teacher → number that passed verification:")
    for k, v in teacher.first_kinds.most_common():
        print(f"  {k:<10}{v:>5} → {c[('passed', k)]:>4}")
    sneaky = {kk[1]: v for kk, v in c.items() if isinstance(kk, tuple) and kk[0] == "wrong args but passed"}
    print(f"  ('wrong args' that passed, by task type: {sneaky}. The execution result is the same as for the gold call; see README)")
    print("\nFunnel (cumulative, gate by gate):")
    total = c["候选"]
    for stage in ["候选", "关1 格式正确", "关2 调用正确", "关3 执行+回答正确"]:
        print(f"  {STAGE_EN[stage]:<14}{c[stage]:>6}  ({c[stage] / total:6.1%})")

    # 2) Production code: zero.post.distill.teacher_trajectories (keeps KEEP samples for each task)
    teacher2 = SimTeacher(tasks)
    kept_rows, n_cand, covered = [], 0, 0
    for i, task in enumerate(tasks):
        kept, n = teacher_trajectories(teacher2, task, K, KEEP, seed=i)
        n_cand += n
        covered += bool(kept)
        kept_rows += [(task, m) for m in kept]
    print(f"\nzero.post.distill.teacher_trajectories: {n_cand} candidates → kept {len(kept_rows)}"
          f" ({covered}/{N_TASKS} tasks have at least one; gate 4 keeps at most {KEEP} for each task)")
    print(f"  Without the filter, {total - c['关3 执行+回答正确']} / {total} = {1 - c['关3 执行+回答正确'] / total:.1%} of the candidates did not pass verification"
          " (almost all of them are wrong; the tasks that need no tool cannot be checked, and they are included too).")

    # 3) Tasks that need no tool: the answer cannot be checked automatically, so it does not go into the distillation data
    n_no_tool = sum(1 for t in tasks if not t.gold_calls)
    no_tool = [(t, m) for t, m in kept_rows if not t.gold_calls]
    print(f"\n{n_no_tool} tasks need no tool ({n_no_tool * K} candidates, {teacher.first_kinds['nonsense']} of them nonsense)"
          f" → kept {len(no_tool)}.")
    print("  These answers have no gold answer to check them against (reward NO_TOOL_REWARD = 0.5, not counted as verified). "
          "Examples of \"no tool call when no tool is needed\" come from the human-reviewed SFT data.")


if __name__ == "__main__":
    main()
