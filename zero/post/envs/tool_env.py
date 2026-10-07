"""Tool-calling environment: mock APIs, task generation, and verifiable rewards.

Chapter 19 uses this module. Chapters 16–18 and 20 also use it.

GRPO needs an environment that can check answers automatically. Everything here is deterministic.
The environment does not use the network or the real clock.

The model sees some strings in this file as input: the system prompt, the tool descriptions, the
task texts, the city and unit names, and the `ToolError` messages (they go back to the model as tool
results). These strings are data. They stay in Chinese, because a change would change the training
data and the results.

**Mock APIs** (`TOOLS`; the JSON schema follows the OpenAI / Qwen function format)

| Tool | What it does |
|---|---|
| `calculator(expression)` | Evaluates an arithmetic expression. It accepts only numbers, + - * / // % **, and parentheses. It uses an AST allowlist, not eval. |
| `get_weather(city)` | Looks up the weather in a fixed table. It accepts Chinese and English city names. |
| `convert_units(value, from_unit, to_unit)` | Converts length, mass, and temperature. It accepts common aliases: km / 公里 / kilometer. |
| `date_add(date, days)` | Adds a number of days to a date. |
| `days_between(start_date, end_date)` | Gives the number of days between two dates. |
| `weekday(date)` | Gives the day of the week of a date. |

**Task** (`Task`): a tool list (the necessary tools + distractor tools), a user question, the gold
calls, and the answer facts (strings that the final answer must contain). `make_splits()` makes
train / dev sets that do not overlap. About 10% of the tasks need no tool call (for example, a
greeting). These tasks test that the model makes no call when no call is necessary. BFCL calls this
"irrelevance".

**Reward** (`score_tool_calls` for the single-step call in GRPO; `score_final_answer` for a full episode)

    Format error (broken JSON, unpaired tags, forged <tool_response>, too long, too many calls) → -1
    A tool call when no call is necessary → -0.5; no call when a call is necessary → 0
    Otherwise  0.1 (format score) + 0.9 × Σ score of each gold call / number of gold calls
               − 0.25 × number of extra calls − 0.5 × number of calls that do not match the schema
          Score of each gold call: function name + arguments equal by AST or by execution result → 1;
          only the function name is correct → 0.2
    Then clip to [-1, 1]

**Design against reward hacking.** Each item has a test in tests/test_tool_env.py:

1. **One-to-one matching**: the predicted calls and the gold calls are matched one to one. A repeated
   call cannot get more score. Each extra call loses score, so "spray all possible calls" loses reward.
2. **An execution match needs the same numbers**: when the calculator results are equal, the numbers in
   the expression (as a multiset) must also be equal to the gold call. Otherwise the model can
   calculate the answer itself and call `calculator("42")` to get the execution score.
3. **No forged tool results**: `<tool_response>` or `<|im_start|>` in the output is a format error.
   This stops the model from writing a fake tool result and then answering "from the result".
4. **Limits on length and number of calls**: output longer than `MAX_OUTPUT_CHARS`, or more than
   `MAX_CALLS` calls, is a format error.
5. **Safe execution**: calculator accepts only the AST nodes in the allowlist. The exponent and the
   size of numbers have limits, so `9**9**9` cannot freeze the scorer.
6. **Final answer**: all facts must be present. The count of numbers in the answer has a limit
   (`_too_many_numbers`). This stops the model from listing many candidate numbers to guess.
7. **Schema check**: a missing field, an extra field, or a wrong type means that the call does not
   match the schema. Such a call is not executed and gets no score.
8. **A call without tags is a format error**: JSON such as `{"name": ...` / `"arguments":` outside the
   tags gets -1. We **observed this hack** in the smoke test. The first reward checked only the content
   in the `<tool_call>` tags. Almost all calls of the tiny model were broken JSON (-1). In 10 GRPO
   steps, the model learned to "remove the tags and output the JSON as before". The format errors
   stopped, the score went from -1 to 0, and the mean reward curve went up nicely. But the model could
   not make a single call. With this rule, such output gets -1 again.
   Also, on a task that needs no tool, an empty reply gets 0, not 1.

Limit: for tasks that need no tool, the scorer checks only that the model makes no wrong calls. It does
not judge the quality of the reply. That needs a reward model or rule-based scoring.
"""

from __future__ import annotations

import ast
import datetime as dt
import functools
import json
import math
import operator
import random
import re
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from zero.post.chat import parse_assistant

MAX_CALLS = 5
MAX_OUTPUT_CHARS = 2000
SYSTEM_PROMPT = "你是一个会使用工具的助手。需要时调用工具，拿到结果后用一句话回答。"


class ToolError(ValueError):
    """The tool arguments are wrong, or the execution failed.

    The message goes back to the model as the tool result (`execute_safely`). It is data, so the
    messages of this error stay in Chinese.
    """


# ---------------------------------------------------------------------------
# Mock APIs
# ---------------------------------------------------------------------------


def _fn(name: str, desc: str, props: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": desc,
            "parameters": {"type": "object", "properties": props, "required": required},
        },
    }


TOOLS: dict[str, dict[str, Any]] = {
    "calculator": _fn(
        "calculator",
        "计算算术表达式",
        {"expression": {"type": "string", "description": "如 3 * (4 + 5)"}},
        ["expression"],
    ),
    "get_weather": _fn("get_weather", "查询城市今天的天气", {"city": {"type": "string"}}, ["city"]),
    "convert_units": _fn(
        "convert_units",
        "单位换算（长度、质量、温度）",
        {
            "value": {"type": "number"},
            "from_unit": {"type": "string"},
            "to_unit": {"type": "string"},
        },
        ["value", "from_unit", "to_unit"],
    ),
    "date_add": _fn(
        "date_add",
        "日期加上若干天",
        {"date": {"type": "string", "description": "YYYY-MM-DD"}, "days": {"type": "integer"}},
        ["date", "days"],
    ),
    "days_between": _fn(
        "days_between",
        "两个日期相差的天数",
        {"start_date": {"type": "string"}, "end_date": {"type": "string"}},
        ["start_date", "end_date"],
    ),
    "weekday": _fn("weekday", "某个日期是星期几", {"date": {"type": "string"}}, ["date"]),
}

# A fixed weather table (invented data, so that the results are deterministic)
WEATHER: dict[str, dict[str, Any]] = {
    "北京": {"condition": "晴", "temp_c": 25, "humidity": 30},
    "上海": {"condition": "多云", "temp_c": 28, "humidity": 70},
    "广州": {"condition": "雷阵雨", "temp_c": 31, "humidity": 85},
    "深圳": {"condition": "阵雨", "temp_c": 30, "humidity": 80},
    "杭州": {"condition": "小雨", "temp_c": 23, "humidity": 75},
    "成都": {"condition": "阴", "temp_c": 21, "humidity": 65},
    "东京": {"condition": "晴", "temp_c": 19, "humidity": 50},
    "伦敦": {"condition": "小雨", "temp_c": 14, "humidity": 80},
    "纽约": {"condition": "多云", "temp_c": 17, "humidity": 55},
    "巴黎": {"condition": "阴", "temp_c": 16, "humidity": 60},
}
CITY_ALIASES: dict[str, str] = {
    "beijing": "北京",
    "shanghai": "上海",
    "guangzhou": "广州",
    "shenzhen": "深圳",
    "hangzhou": "杭州",
    "chengdu": "成都",
    "tokyo": "东京",
    "london": "伦敦",
    "new york": "纽约",
    "paris": "巴黎",
}
CITY_EN = {v: k.title() for k, v in CITY_ALIASES.items()}

# Units: canonical name → (category, factor to the base unit). Temperature has its own formula.
_UNITS: dict[str, tuple[str, float]] = {
    "m": ("length", 1.0),
    "km": ("length", 1000.0),
    "cm": ("length", 0.01),
    "mile": ("length", 1609.344),
    "ft": ("length", 0.3048),
    "inch": ("length", 0.0254),
    "kg": ("mass", 1.0),
    "g": ("mass", 0.001),
    "lb": ("mass", 0.45359237),
    "oz": ("mass", 0.028349523125),
    "C": ("temperature", 1.0),
    "F": ("temperature", 1.0),
    "K": ("temperature", 1.0),
}
UNIT_ALIASES: dict[str, str] = {
    **{u.lower(): u for u in _UNITS},
    "meter": "m",
    "meters": "m",
    "米": "m",
    "kilometer": "km",
    "kilometers": "km",
    "公里": "km",
    "千米": "km",
    "centimeter": "cm",
    "厘米": "cm",
    "miles": "mile",
    "英里": "mile",
    "feet": "ft",
    "foot": "ft",
    "英尺": "ft",
    "inches": "inch",
    "英寸": "inch",
    "kilogram": "kg",
    "千克": "kg",
    "公斤": "kg",
    "gram": "g",
    "克": "g",
    "pound": "lb",
    "pounds": "lb",
    "磅": "lb",
    "ounce": "oz",
    "盎司": "oz",
    "celsius": "C",
    "摄氏度": "C",
    "°c": "C",
    "fahrenheit": "F",
    "华氏度": "F",
    "°f": "F",
    "kelvin": "K",
    "开尔文": "K",
}
WEEKDAYS_ZH = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
WEEKDAYS_EN = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

_BIN_OPS: dict[type, Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_MAX_ABS = 1e15
_MAX_EXPR_CHARS = 200


def safe_eval(expr: str) -> float | int:
    """Evaluate an expression without eval.

    Only numbers, + - * /, floor division, modulo, power, parentheses, and signs are allowed.
    """
    if not isinstance(expr, str) or not expr.strip():
        raise ToolError("expression 必须是非空字符串")
    if len(expr) > _MAX_EXPR_CHARS:
        raise ToolError("表达式太长")
    expr = expr.replace("×", "*").replace("÷", "/").replace("^", "**")
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise ToolError(f"表达式语法错误：{expr}") from e

    def ev(node: ast.AST) -> float | int:
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return node.value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub | ast.UAdd):
            v = ev(node.operand)
            return -v if isinstance(node.op, ast.USub) else v
        if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
            a, b = ev(node.left), ev(node.right)
            if isinstance(node.op, ast.Pow) and (abs(b) > 64 or abs(a) > 1e6):
                raise ToolError("乘方太大")
            try:
                r = _BIN_OPS[type(node.op)](a, b)
            except ZeroDivisionError as e:
                raise ToolError("除以零") from e
            if isinstance(r, complex) or abs(r) > _MAX_ABS:
                raise ToolError("结果超出范围")
            return r
        raise ToolError(f"不支持的表达式成分：{type(node).__name__}")

    return ev(tree)


def _num(x: float | int) -> float | int:
    """Make the result a JSON-friendly number: an int for an integer value, else 6 decimal places."""
    if isinstance(x, float) and x.is_integer() and abs(x) < 1e15:
        return int(x)
    return round(float(x), 6) if isinstance(x, float) else x


def _parse_date(s: Any, field_name: str) -> dt.date:
    if not isinstance(s, str):
        raise ToolError(f"{field_name} 必须是 YYYY-MM-DD 字符串")
    try:
        return dt.date.fromisoformat(s.strip())
    except ValueError as e:
        raise ToolError(f"{field_name} 不是合法日期：{s}") from e


def _canon_unit(u: Any) -> str:
    if not isinstance(u, str):
        raise ToolError("单位必须是字符串")
    key = u.strip()
    canon = UNIT_ALIASES.get(key.lower()) or UNIT_ALIASES.get(key)
    if canon is None:
        raise ToolError(f"不认识的单位：{u}")
    return canon


def canon_city(city: Any) -> str:
    if not isinstance(city, str):
        raise ToolError("city 必须是字符串")
    c = city.strip()
    c = CITY_ALIASES.get(c.lower(), c).removesuffix("市")
    if c not in WEATHER:
        raise ToolError(f"没有 {city} 的天气数据")
    return c


def convert(value: float, from_unit: str, to_unit: str) -> float:
    fu, tu = _canon_unit(from_unit), _canon_unit(to_unit)
    fk, tk = _UNITS[fu][0], _UNITS[tu][0]
    if fk != tk:
        raise ToolError(f"{from_unit} 和 {to_unit} 不是同一类单位")
    if fk == "temperature":
        c = {"C": value, "F": (value - 32) * 5 / 9, "K": value - 273.15}[fu]
        return {"C": c, "F": c * 9 / 5 + 32, "K": c + 273.15}[tu]
    return value * _UNITS[fu][1] / _UNITS[tu][1]


def validate_arguments(name: str, args: Any) -> list[str]:
    """Check the arguments against the JSON schema: missing fields, extra fields, wrong types.

    Return a list of errors (empty means valid). The errors also go back to the model through
    `ToolError`, so they are data.
    """
    if name not in TOOLS:
        return [f"没有这个工具：{name}"]
    if not isinstance(args, dict):
        return ["arguments 必须是对象"]
    params = TOOLS[name]["function"]["parameters"]
    props: dict[str, Any] = params["properties"]
    errors = [f"缺少参数 {r}" for r in params["required"] if r not in args]
    errors += [f"多余参数 {k}" for k in args if k not in props]
    for k, v in args.items():
        if k not in props:
            continue
        t = props[k]["type"]
        ok = {
            "string": isinstance(v, str),
            "number": isinstance(v, int | float) and not isinstance(v, bool),
            "integer": isinstance(v, int) and not isinstance(v, bool),
            "boolean": isinstance(v, bool),
        }.get(t, True)
        if not ok:
            errors.append(f"参数 {k} 应为 {t}")
    return errors


def execute_call(name: str, args: dict[str, Any]) -> dict[str, Any]:
    """Run one tool call and return a JSON-serializable result. Raise ToolError for wrong arguments."""
    errs = validate_arguments(name, args)
    if errs:
        raise ToolError("；".join(errs))
    if name == "calculator":
        return {"result": _num(safe_eval(args["expression"]))}
    if name == "get_weather":
        c = canon_city(args["city"])
        return {"city": c, **WEATHER[c]}
    if name == "convert_units":
        v = convert(float(args["value"]), args["from_unit"], args["to_unit"])
        return {"value": round(v, 4), "unit": _canon_unit(args["to_unit"])}
    if name == "date_add":
        d = _parse_date(args["date"], "date")
        if abs(args["days"]) > 100_000:
            raise ToolError("days 太大")
        return {"date": (d + dt.timedelta(days=args["days"])).isoformat()}
    if name == "days_between":
        a = _parse_date(args["start_date"], "start_date")
        b = _parse_date(args["end_date"], "end_date")
        return {"days": (b - a).days}
    if name == "weekday":
        d = _parse_date(args["date"], "date")
        return {"weekday": WEEKDAYS_ZH[d.weekday()], "weekday_en": WEEKDAYS_EN[d.weekday()]}
    raise ToolError(f"没有这个工具：{name}")


def execute_safely(name: str, args: dict[str, Any]) -> dict[str, Any]:
    """Run the call. On an error, return {"error": ...} (the tool result that goes back to the model)."""
    try:
        return execute_call(name, args)
    except ToolError as e:
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------


@dataclass
class Task:
    id: str
    kind: str
    lang: str
    tools: list[dict[str, Any]]
    messages: list[dict[str, Any]]  # system + user
    gold_calls: list[dict[str, Any]]  # [{"name", "arguments"}]; empty means that no tool call is necessary
    answer_facts: list[str]  # facts that the final answer must contain
    gold_answer: str  # the reference final answer (for SFT)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Task:
        return cls(**{k: d[k] for k in cls.__dataclass_fields__})

    @property
    def query(self) -> str:
        return self.messages[-1]["content"]


def _fmt_expr(rng: random.Random) -> str:
    a, b = rng.randint(2, 999), rng.randint(2, 99)
    op = rng.choice(["+", "-", "*", "*", "+"])
    if rng.random() < 0.3:
        c = rng.randint(2, 20)
        return f"({a} {op} {b}) * {c}"
    return f"{a} {op} {b}"


def _random_date(rng: random.Random) -> str:
    d = dt.date(2020, 1, 1) + dt.timedelta(days=rng.randint(0, 365 * 8))
    return d.isoformat()


def _fmt_num(x: float | int) -> str:
    x = _num(x) if isinstance(x, float) else x
    return str(x)


_NO_TOOL_QUERIES = [
    ("zh", "你好！", "你好！有什么可以帮你的？"),
    ("zh", "谢谢你的帮助。", "不客气！"),
    ("zh", "你是谁？", "我是一个会使用工具的助手。"),
    ("en", "Hello!", "Hello! How can I help you?"),
    ("en", "Thanks a lot.", "You're welcome!"),
    ("zh", "讲一句鼓励的话。", "坚持下去，你一定可以的！"),
]


def make_task(rng: random.Random, idx: int, split: str) -> Task:
    """Generate one random task."""
    lang = "zh" if rng.random() < 0.7 else "en"
    r = rng.random()
    calls: list[dict[str, Any]]
    if r < 0.1:
        lang, q, ans = rng.choice(_NO_TOOL_QUERIES)
        kind, calls, facts = "no_tool", [], []
    elif r < 0.3:
        kind = "calculator"
        e = _fmt_expr(rng)
        v = _num(safe_eval(e))
        q = rng.choice(
            [f"帮我算一下 {e} 等于多少？", f"{e} 是多少？", f"计算 {e}。"]
            if lang == "zh"
            else [f"What is {e}?", f"Please compute {e}."]
        )
        calls = [{"name": "calculator", "arguments": {"expression": e}}]
        facts = [_fmt_num(v)]
        ans = f"{e} = {_fmt_num(v)}。" if lang == "zh" else f"{e} = {_fmt_num(v)}."
    elif r < 0.45:
        kind = "weather"
        city = rng.choice(list(WEATHER))
        w = WEATHER[city]
        name = city if lang == "zh" else CITY_EN[city]
        q = (
            rng.choice([f"{name}今天天气怎么样？", f"查一下{name}的天气。"])
            if lang == "zh"
            else f"What's the weather like in {name} today?"
        )
        calls = [{"name": "get_weather", "arguments": {"city": name}}]
        facts = [str(w["temp_c"])]
        ans = (
            f"{city}今天{w['condition']}，气温 {w['temp_c']}°C。"
            if lang == "zh"
            else f"{name}: {w['condition']}, {w['temp_c']}°C."
        )
        if lang == "zh":
            facts.append(w["condition"])
    elif r < 0.55:
        kind = "weather_compare"
        c1, c2 = rng.sample(list(WEATHER), 2)
        t1, t2 = WEATHER[c1]["temp_c"], WEATHER[c2]["temp_c"]
        if t1 == t2:
            c2 = next(c for c in WEATHER if WEATHER[c]["temp_c"] != t1)
            t2 = WEATHER[c2]["temp_c"]
        lang = "zh"
        q = f"{c1}和{c2}今天哪个更热？"
        calls = [
            {"name": "get_weather", "arguments": {"city": c1}},
            {"name": "get_weather", "arguments": {"city": c2}},
        ]
        hot = c1 if t1 > t2 else c2
        facts = [hot]
        ans = f"{hot}更热（{c1} {t1}°C，{c2} {t2}°C）。"
    elif r < 0.7:
        kind = "convert"
        pairs = [
            ("km", "mile", "公里", "英里"),
            ("mile", "km", "英里", "公里"),
            ("kg", "lb", "千克", "磅"),
            ("lb", "kg", "磅", "千克"),
            ("C", "F", "摄氏度", "华氏度"),
            ("F", "C", "华氏度", "摄氏度"),
            ("m", "ft", "米", "英尺"),
            ("inch", "cm", "英寸", "厘米"),
        ]
        fu, tu, fz, tz = rng.choice(pairs)
        val = rng.randint(1, 500)
        out = round(convert(val, fu, tu), 2)
        q = f"{val} {fz}等于多少{tz}？" if lang == "zh" else f"Convert {val} {fu} to {tu}."
        calls = [
            {"name": "convert_units", "arguments": {"value": val, "from_unit": fu, "to_unit": tu}}
        ]
        facts = [_fmt_num(out)]
        ans = (
            f"{val} {fz}约等于 {_fmt_num(out)} {tz}。"
            if lang == "zh"
            else f"{val} {fu} ≈ {_fmt_num(out)} {tu}."
        )
    elif r < 0.8:
        kind = "date_add"
        d = _random_date(rng)
        n = rng.randint(1, 400)
        res = (dt.date.fromisoformat(d) + dt.timedelta(days=n)).isoformat()
        q = f"{d} 之后 {n} 天是哪一天？" if lang == "zh" else f"What date is {n} days after {d}?"
        calls = [{"name": "date_add", "arguments": {"date": d, "days": n}}]
        facts = [res]
        ans = f"是 {res}。" if lang == "zh" else f"It is {res}."
    elif r < 0.9:
        kind = "days_between"
        d1 = _random_date(rng)
        d2 = (dt.date.fromisoformat(d1) + dt.timedelta(days=rng.randint(1, 900))).isoformat()
        n = (dt.date.fromisoformat(d2) - dt.date.fromisoformat(d1)).days
        q = (
            f"从 {d1} 到 {d2} 有多少天？"
            if lang == "zh"
            else f"How many days are there from {d1} to {d2}?"
        )
        calls = [{"name": "days_between", "arguments": {"start_date": d1, "end_date": d2}}]
        facts = [str(n)]
        ans = f"相差 {n} 天。" if lang == "zh" else f"{n} days."
    else:
        kind = "weekday"
        d = _random_date(rng)
        wd = dt.date.fromisoformat(d).weekday()
        q = f"{d} 是星期几？" if lang == "zh" else f"What day of the week is {d}?"
        calls = [{"name": "weekday", "arguments": {"date": d}}]
        facts = [WEEKDAYS_ZH[wd] if lang == "zh" else WEEKDAYS_EN[wd]]
        ans = f"{d} 是{facts[0]}。" if lang == "zh" else f"{d} is a {facts[0]}."

    needed = sorted({c["name"] for c in calls})
    others = [t for t in TOOLS if t not in needed]
    n_distract = rng.randint(1, 2) if needed else rng.randint(1, 3)
    names = needed + rng.sample(others, n_distract)
    rng.shuffle(names)
    return Task(
        id=f"{split}-{idx:06d}",
        kind=kind,
        lang=lang,
        tools=[TOOLS[n] for n in names],
        messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": q}],
        gold_calls=calls,
        answer_facts=facts,
        gold_answer=ans,
    )


# The fixed dev set has a fixed seed. The training tasks exclude each task with the same question
# text as a dev task, so the evaluation set does not leak into training.
# Exception: weather (about 30 question forms) and greetings (6) have a question space that is too
# small to avoid overlap with dev. For these two kinds, dev tests only "uses the correct tool /
# calls or does not call". The report shows them separately.
DEV_SEED = 0
N_DEV_MAX = 300
SMALL_SPACE_KINDS = frozenset({"weather", "no_tool"})


def _generate(n: int, seed: int, split: str, exclude: frozenset[str], dedup: bool) -> list[Task]:
    rng = random.Random(f"tool_env-{split}-{seed}")
    out: list[Task] = []
    seen: set[str] = set()
    tries = 0
    while len(out) < n:
        tries += 1
        if tries > n * 50 + 1000:
            raise RuntimeError("Cannot generate enough unique tasks")
        t = make_task(rng, len(out), split)
        if t.kind not in SMALL_SPACE_KINDS:
            if t.query in exclude or (dedup and t.query in seen):
                continue
        seen.add(t.query)
        out.append(t)
    return out


def dev_tasks(n: int = 200) -> list[Task]:
    """Return the fixed dev set (the first n tasks; a larger n gives a superset of a smaller n)."""
    if n > N_DEV_MAX:
        raise ValueError(f"The dev set has at most {N_DEV_MAX} tasks")
    return _generate(n, DEV_SEED, "dev", frozenset(), dedup=True)


@functools.lru_cache(maxsize=1)
def _dev_queries() -> frozenset[str]:
    return frozenset(t.query for t in dev_tasks(N_DEV_MAX) if t.kind not in SMALL_SPACE_KINDS)


def generate_tasks(n: int, seed: int = 0, split: str = "train") -> list[Task]:
    """Generate n tasks.

    With split="train", exclude all tasks with the same question as a task in the fixed dev set.
    With split="dev", return the fixed dev set.
    """
    if split == "dev":
        return dev_tasks(n)
    return _generate(n, seed, split, _dev_queries(), dedup=False)


def make_splits(n_train: int, n_dev: int, seed: int = 0) -> tuple[list[Task], list[Task]]:
    """Return (training tasks, the first n_dev tasks of the fixed dev set)."""
    return generate_tasks(n_train, seed, "train"), dev_tasks(n_dev)


def reference_messages(task: Task) -> list[dict[str, Any]]:
    """The full conversation of the gold solution.

    user → assistant(tool_calls) → tool results → assistant(final answer).
    """
    msgs = [dict(m) for m in task.messages]
    if task.gold_calls:
        msgs.append({"role": "assistant", "content": "", "tool_calls": task.gold_calls})
        for c in task.gold_calls:
            msgs.append(
                {
                    "role": "tool",
                    "content": json.dumps(
                        execute_call(c["name"], c["arguments"]), ensure_ascii=False
                    ),
                }
            )
    msgs.append({"role": "assistant", "content": task.gold_answer})
    return msgs


# ---------------------------------------------------------------------------
# Rewards
# ---------------------------------------------------------------------------


@dataclass
class Reward:
    total: float
    format_ok: bool
    n_calls: int = 0
    ast_match: float = 0.0  # fraction of gold calls with AST-equal arguments
    exec_match: float = 0.0  # fraction of gold calls with equivalent (or AST-equal) arguments; calculator: by result
    answer_ok: bool | None = None
    details: list[str] = field(default_factory=list)
    exact: bool = False  # correct calls and no extra call; or no call (when none is necessary) and nothing invented


_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")
# A chat task that needs no tool: the content cannot be scored automatically. The task gets a
# partial score only for "correctly made no call". The full score is for verifiable tasks.
NO_TOOL_REWARD = 0.5


def _norm_value(v: Any) -> Any:
    if isinstance(v, str):
        return re.sub(r"\s+", "", v).lower()
    if isinstance(v, bool):
        return v
    if isinstance(v, int | float):
        return round(float(v), 6)
    if isinstance(v, dict):
        return {k: _norm_value(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_norm_value(x) for x in v]
    return v


def ast_equal(pred: dict[str, Any], gold: dict[str, Any]) -> bool:
    """The function names are equal, and the arguments are equal item by item after normalization.

    Strings: no whitespace, case-insensitive. Numbers: compared by value.
    """
    return pred["name"] == gold["name"] and _norm_value(pred["arguments"]) == _norm_value(
        gold["arguments"]
    )


def _results_equal(a: dict[str, Any], b: dict[str, Any]) -> bool:
    if set(a) != set(b):
        return False
    for k in a:
        x, y = a[k], b[k]
        if isinstance(x, int | float) and isinstance(y, int | float):
            if not math.isclose(x, y, rel_tol=1e-6, abs_tol=1e-6):
                return False
        elif x != y:
            return False
    return True


def exec_equal(pred: dict[str, Any], gold: dict[str, Any]) -> bool:
    """The execution results are equal.

    For calculator, the multisets of numbers must also be equal. Then the model cannot calculate the
    answer itself and pass only the answer.
    """
    if pred["name"] != gold["name"]:
        return False
    try:
        rp = execute_call(pred["name"], pred["arguments"])
        rg = execute_call(gold["name"], gold["arguments"])
    except ToolError:
        return False
    if pred["name"] == "calculator":
        nums = lambda s: Counter(_NUM_RE.findall(str(s)))  # noqa: E731
        if nums(pred["arguments"]["expression"]) != nums(gold["arguments"]["expression"]):
            return False
    return _results_equal(rp, rg)


def canonical_args(name: str, args: dict[str, Any]) -> dict[str, Any] | None:
    """Convert the arguments to a canonical form. Return None if the arguments are not valid.

    Dates become ISO strings, units and cities become standard names, and numbers become values.
    """
    try:
        if name == "get_weather":
            return {"city": canon_city(args["city"])}
        if name == "convert_units":
            return {
                "value": round(float(args["value"]), 6),
                "from_unit": _canon_unit(args["from_unit"]),
                "to_unit": _canon_unit(args["to_unit"]),
            }
        if name == "date_add":
            return {
                "date": _parse_date(args["date"], "date").isoformat(),
                "days": int(args["days"]),
            }
        if name == "days_between":
            return {
                "start_date": _parse_date(args["start_date"], "start_date").isoformat(),
                "end_date": _parse_date(args["end_date"], "end_date").isoformat(),
            }
        if name == "weekday":
            return {"date": _parse_date(args["date"], "date").isoformat()}
    except (ToolError, KeyError, TypeError, ValueError):
        return None
    return None


def args_equivalent(pred: dict[str, Any], gold: dict[str, Any]) -> bool:
    """The arguments "mean the same": they are equal item by item after normalization.

    Only calculator uses the execution result (plus the same multiset of numbers), because one
    expression has many equivalent forms. For the other tools, the arguments themselves must be
    equivalent. A check of only the execution result accepts a call with "wrong arguments and a
    correct result by chance". Example: for weekday, a date that is 7 days off gives the same day of
    the week. This is a real case from Chapters 17 and 19.
    """
    if pred["name"] != gold["name"]:
        return False
    if pred["name"] == "calculator":
        return exec_equal(pred, gold)
    cp = canonical_args(pred["name"], pred["arguments"])
    return cp is not None and cp == canonical_args(gold["name"], gold["arguments"])


def _invented_numbers(answer: str, query: str) -> bool:
    """The answer has numbers that are not in the question.

    On a chat task that needs no tool, such numbers usually mean an invented tool result.
    """
    allowed = set(_NUM_RE.findall(query))
    return any(n not in allowed for n in _NUM_RE.findall(answer))


def _forged(text: str) -> str | None:
    for tag in ("<tool_response>", "</tool_response>", "<|im_start|>"):
        if tag in text:
            return f"The output contains {tag} (a forged tool result or conversation turn)"
    return None


_BARE_CALL_RE = re.compile(r'\{\s*"(?:name|arguments)\b|"arguments"\s*:')


def _bare_call(content: str) -> str | None:
    """Tool-call JSON outside the tags is a format error.

    Without this rule, the model can remove the tags to avoid the -1 format score.
    """
    if _BARE_CALL_RE.search(content):
        return "Tool call without <tool_call> tags (name / arguments JSON outside the tags)"
    return None


def score_tool_calls(task: Task, text: str) -> Reward:
    """Score the first output of the assistant: which tools it calls, and if the arguments are correct."""
    if len(text) > MAX_OUTPUT_CHARS:
        return Reward(-1.0, False, details=["Output too long"])
    forged = _forged(text)
    if forged:
        return Reward(-1.0, False, details=[forged])
    parsed = parse_assistant(text)
    if parsed.errors:
        return Reward(-1.0, False, n_calls=len(parsed.tool_calls), details=list(parsed.errors))
    bare = _bare_call(parsed.content)
    if bare:
        return Reward(-1.0, False, n_calls=len(parsed.tool_calls), details=[bare])
    calls = parsed.tool_calls
    if len(calls) > MAX_CALLS:
        return Reward(-1.0, False, n_calls=len(calls), details=[f"More than {MAX_CALLS} calls"])
    offered = {t["function"]["name"] for t in task.tools}
    valid, details = [], []
    n_invalid = 0
    for c in calls:
        errs = validate_arguments(c["name"], c["arguments"])
        if c["name"] not in offered:
            errs = [f"Called a tool that was not offered: {c['name']}", *errs]
        if errs:
            n_invalid += 1
            details.extend(errs)
        else:
            valid.append(c)
    format_ok = n_invalid == 0
    gold = task.gold_calls
    if not gold:
        if not calls:
            content = parsed.content.strip()
            if not content:
                return Reward(0.0, True, 0, details=["Empty answer"])
            if _invented_numbers(content, task.query):
                return Reward(
                    0.0,
                    True,
                    0,
                    answer_ok=False,
                    details=["No tool call, but the answer has numbers that are not in the question (possibly invented)"],
                )
            # No tool call is correct. But the chat content cannot be checked automatically, so the
            # score is partial and answer_ok is "unknown".
            return Reward(
                NO_TOOL_REWARD,
                True,
                0,
                1.0,
                1.0,
                answer_ok=None,
                details=["Correctly made no tool call (the answer content cannot be checked automatically)"],
                exact=True,
            )
        return Reward(-0.5, format_ok, len(calls), details=["Called a tool when no tool was necessary", *details])
    if not calls:
        return Reward(0.0, True, 0, details=["A tool call was necessary, but there was no call"])

    # One-to-one matching: first the AST-equal calls, then the equivalent calls, then the calls
    # with only the same function name
    remaining = list(range(len(valid)))
    n_ast = n_exec = 0
    score = 0.0
    unmatched_gold = []
    for g in gold:
        hit = next((i for i in remaining if ast_equal(valid[i], g)), None)
        if hit is not None:
            remaining.remove(hit)
            n_ast += 1
            n_exec += 1
            score += 1.0
            continue
        unmatched_gold.append(g)
    still = []
    for g in unmatched_gold:
        hit = next((i for i in remaining if args_equivalent(valid[i], g)), None)
        if hit is not None:
            remaining.remove(hit)
            n_exec += 1
            score += 1.0
        else:
            still.append(g)
    for g in still:
        hit = next((i for i in remaining if valid[i]["name"] == g["name"]), None)
        if hit is not None:
            remaining.remove(hit)
            score += 0.2
            details.append(f"Wrong arguments for {g['name']}")
        else:
            details.append(f"Missing call {g['name']}")
    n_extra = len(remaining)
    if n_extra:
        details.append(f"Extra calls: {n_extra}")
    total = 0.1 + 0.9 * score / len(gold) - 0.25 * n_extra - 0.5 * n_invalid
    total = max(-1.0, min(1.0, total))
    return Reward(
        total,
        format_ok,
        len(calls),
        n_ast / len(gold),
        n_exec / len(gold),
        details=details,
        exact=total >= 0.999,
    )


def _too_many_numbers(answer: str, facts: Sequence[str]) -> bool:
    allowed = sum(len(_NUM_RE.findall(f)) for f in facts) + 6  # numbers repeated from the question are normal
    return len(_NUM_RE.findall(answer)) > allowed


def score_final_answer(task: Task, text: str) -> Reward:
    """Score the final answer.

    The answer must not call a tool, and it must contain all facts. It must not list many numbers to guess.
    """
    if len(text) > MAX_OUTPUT_CHARS:
        return Reward(-1.0, False, answer_ok=False, details=["Output too long"])
    forged = _forged(text)
    parsed = parse_assistant(text)
    if forged or parsed.errors:
        return Reward(-1.0, False, answer_ok=False, details=[forged or "", *parsed.errors])
    if parsed.tool_calls:
        return Reward(
            0.0, True, len(parsed.tool_calls), answer_ok=False, details=["The final answer still calls a tool"]
        )
    bare = _bare_call(parsed.content)
    if bare:
        return Reward(-1.0, False, answer_ok=False, details=[bare])
    ans = parsed.content.strip()
    if not ans:
        return Reward(0.0, True, answer_ok=False, details=["Empty answer"])
    if not task.answer_facts:
        if _invented_numbers(ans, task.query):
            return Reward(
                0.0, True, answer_ok=False, details=["The answer has numbers that are not in the question (possibly invented)"]
            )
        return Reward(NO_TOOL_REWARD, True, answer_ok=None, details=["The answer content cannot be checked automatically"])
    if _too_many_numbers(ans, task.answer_facts):
        return Reward(0.0, True, answer_ok=False, details=["Too many numbers in the answer"])
    norm = ans.replace(" ", "").lower()
    ok = all(f.replace(" ", "").lower() in norm for f in task.answer_facts)
    return Reward(1.0 if ok else 0.0, True, answer_ok=ok, details=[] if ok else ["Missing facts"])


# ---------------------------------------------------------------------------
# A full episode: generate → parse → execute → feed back → final answer
# ---------------------------------------------------------------------------

Policy = Callable[[list[dict[str, Any]], list[dict[str, Any]]], str]


@dataclass
class Episode:
    messages: list[dict[str, Any]]
    call_reward: Reward
    answer_reward: Reward
    turns: int

    @property
    def success(self) -> bool:
        # The content of a chat answer cannot be checked (answer_ok is None). The episode is a success
        # if the call behavior is correct and the answer is not judged as invented.
        return self.call_reward.exact and self.answer_reward.answer_ok is not False


def run_episode(policy: Policy, task: Task, max_turns: int = 3) -> Episode:
    """Let the policy (input: messages and tools; output: assistant text) do one task.

    `score_tool_calls` scores the first output. Then the calls in the output are executed, and the
    results go back to the model as tool messages. This continues until a turn makes no tool call
    (the final answer). `score_final_answer` scores that turn.
    """
    msgs = [dict(m) for m in task.messages]
    call_reward: Reward | None = None
    answer_reward = Reward(0.0, True, answer_ok=False, details=["No turns left"])
    turns = 0
    while turns < max_turns:
        turns += 1
        text = policy(msgs, task.tools)
        if call_reward is None:
            call_reward = score_tool_calls(task, text)
        parsed = parse_assistant(text)
        if parsed.errors or _forged(text):
            msgs.append({"role": "assistant", "content": text})
            answer_reward = Reward(-1.0, False, answer_ok=False, details=["Format error; the episode stops"])
            break
        msgs.append(parsed.to_message())
        if not parsed.tool_calls:
            answer_reward = score_final_answer(task, text)
            break
        for c in parsed.tool_calls[:MAX_CALLS]:
            res = execute_safely(c["name"], c["arguments"])
            msgs.append({"role": "tool", "content": json.dumps(res, ensure_ascii=False)})
    assert call_reward is not None
    return Episode(msgs, call_reward, answer_reward, turns)
