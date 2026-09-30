"""工具调用环境：模拟 API + 任务生成 + 可验证奖励（对应第 19 章，第 16–18、20 章也用）。

GRPO 需要一个"能自动判对错"的环境。这里的一切都是确定性的，不联网、不依赖真实时钟：

**模拟 API**（`TOOLS`，JSON schema 与 OpenAI / Qwen 的 function 格式一致）

| 工具 | 作用 |
|---|---|
| `calculator(expression)` | 算术表达式求值（只允许数字、+ - * / // % ** 和括号，用 AST 白名单而不是 eval） |
| `get_weather(city)` | 从固定表里查天气（中英文城市名都认） |
| `convert_units(value, from_unit, to_unit)` | 长度、质量、温度换算（认常见别名：km / 公里 / kilometer） |
| `date_add(date, days)` | 日期加减天数 |
| `days_between(start_date, end_date)` | 两个日期相差几天 |
| `weekday(date)` | 某天是星期几 |

**任务**（`Task`）：工具列表（需要的工具 + 干扰工具）、用户问题、标准调用（gold calls）、
标准答案要点（final answer 里必须出现的字符串）。`make_splits()` 生成互不重叠的 train / dev。
约 10% 的任务不需要调用工具（例如打招呼），考模型"该不调就不调"（BFCL 里叫 irrelevance）。

**奖励**（`score_tool_calls`，用于 GRPO 的单步调用；`score_final_answer` 用于完整一轮）

    格式错误（JSON 坏了、标签不配对、伪造 <tool_response>、超长、调用过多）→ -1
    不该调工具却调了 → -0.5；该调却没调 → 0
    否则  0.1（格式分）+ 0.9 × Σ 每个标准调用的得分 / 标准调用数 − 0.25 × 多余调用数 − 0.5 × 不合 schema 的调用数
          每个标准调用的得分：函数名 + 参数 AST 一致或执行结果一致 → 1；只有函数名对 → 0.2
    最后裁剪到 [-1, 1]

**防奖励作弊（reward hacking）的设计**，每条都有对应测试（tests/test_tool_env.py）：

1. **一对一匹配**：预测调用和标准调用做一对一匹配，重复同一个调用不能多拿分；多出来的调用扣分，
   所以"把所有可能的调用都喷一遍"是亏的；
2. **执行匹配要求同样的数字**：calculator 的执行结果一致时，还要求表达式里的数字（多重集合）与标准
   一致——否则模型可以自己心算出答案、调用 `calculator("42")` 骗到执行分；
3. **禁止伪造工具结果**：输出里出现 `<tool_response>` 或 `<|im_start|>` 直接判格式错误，
   防止模型自己编一段工具返回再"根据结果"作答；
4. **长度和调用数上限**：输出超过 `MAX_OUTPUT_CHARS` 或调用超过 `MAX_CALLS` 判格式错误；
5. **安全执行**：calculator 只接受 AST 白名单节点，幂指数和数值大小有上限，防止 `9**9**9` 卡死判分器；
6. **最终答案**：要点必须全部出现，且答案里出现的数字个数有上限（`_too_many_numbers`），
   防止把一堆候选数字全列出来碰运气；
7. **schema 校验**：参数缺字段、多字段、类型不对都算不合 schema，不能执行，也不能得分；
8. **不带标签的调用算格式错误**：标签外面出现 `{"name": ...` / `"arguments":` 这样的 JSON 判 -1。
   这一条是冒烟测试里**真实观察到**的作弊：最初的奖励只检查 `<tool_call>` 标签里的内容，tiny 模型的
   调用几乎都是坏 JSON（-1 分），GRPO 十步之内就学会了"去掉标签、照样输出 JSON"——不再有格式错误，
   得分从 -1 升到 0，平均奖励曲线漂亮地上升，而模型一个调用都不会了。加上这条之后这种输出回到 -1。
   另外，"不该调用工具"的任务回复为空时得 0 分而不是 1 分。

局限：不需要工具的任务只检查"有没有乱调用"，不判断回答内容好不好（需要奖励模型或规则判分）。
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
    """工具参数不对或执行失败。"""


# ---------------------------------------------------------------------------
# 模拟 API
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

# 固定的天气表（虚构数据，保证确定性）
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

# 单位：规范名 → (类别, 换算到基准单位的系数)；温度单独处理
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
    """只允许数字、四则运算、整除、取模、乘方、括号和正负号的求值器（不用 eval）。"""
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
    """把结果规整成 JSON 友好的数：整数值用 int，其余保留 6 位小数。"""
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
    """按 JSON schema 检查参数：缺字段、多字段、类型不对。返回错误列表（空表示合法）。"""
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
    """执行一次工具调用，返回 JSON 可序列化的结果；参数不对抛 ToolError。"""
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
    """执行，出错时返回 {"error": ...}（喂回给模型的工具结果）。"""
    try:
        return execute_call(name, args)
    except ToolError as e:
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# 任务
# ---------------------------------------------------------------------------


@dataclass
class Task:
    id: str
    kind: str
    lang: str
    tools: list[dict[str, Any]]
    messages: list[dict[str, Any]]  # system + user
    gold_calls: list[dict[str, Any]]  # [{"name", "arguments"}]；空表示不该调用工具
    answer_facts: list[str]  # 最终回答里必须出现的要点
    gold_answer: str  # 参考的最终回答（SFT 用）

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
    """随机生成一个任务。"""
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


# 固定的 dev 集：种子固定，所有训练任务都排除与它问题文本相同的任务（防止评测集泄漏进训练）。
# 例外：天气（约 30 种问法）和打招呼（6 种）的问题空间太小，无法与 dev 不重叠——这两类在 dev 里
# 只考"会不会用对工具 / 该不该调用"，报告里单列。
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
            raise RuntimeError("生成不出足够多的不重复任务")
        t = make_task(rng, len(out), split)
        if t.kind not in SMALL_SPACE_KINDS:
            if t.query in exclude or (dedup and t.query in seen):
                continue
        seen.add(t.query)
        out.append(t)
    return out


def dev_tasks(n: int = 200) -> list[Task]:
    """固定的 dev 集（前 n 个；n 越大，结果是更小 n 的超集）。"""
    if n > N_DEV_MAX:
        raise ValueError(f"dev 集最多 {N_DEV_MAX} 个任务")
    return _generate(n, DEV_SEED, "dev", frozenset(), dedup=True)


@functools.lru_cache(maxsize=1)
def _dev_queries() -> frozenset[str]:
    return frozenset(t.query for t in dev_tasks(N_DEV_MAX) if t.kind not in SMALL_SPACE_KINDS)


def generate_tasks(n: int, seed: int = 0, split: str = "train") -> list[Task]:
    """生成 n 个任务。split="train" 时排除所有与固定 dev 集问题相同的任务；split="dev" 返回固定 dev 集。"""
    if split == "dev":
        return dev_tasks(n)
    return _generate(n, seed, split, _dev_queries(), dedup=False)


def make_splits(n_train: int, n_dev: int, seed: int = 0) -> tuple[list[Task], list[Task]]:
    """(训练任务, 固定 dev 集的前 n_dev 个)。"""
    return generate_tasks(n_train, seed, "train"), dev_tasks(n_dev)


def reference_messages(task: Task) -> list[dict[str, Any]]:
    """标准解答的完整对话：user → assistant(tool_calls) → tool 结果 → assistant(最终回答)。"""
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
# 奖励
# ---------------------------------------------------------------------------


@dataclass
class Reward:
    total: float
    format_ok: bool
    n_calls: int = 0
    ast_match: float = 0.0  # 标准调用里参数 AST 一致的比例
    exec_match: float = 0.0  # 标准调用里参数等价（或 AST 一致）的比例；calculator 按执行结果
    answer_ok: bool | None = None
    details: list[str] = field(default_factory=list)
    exact: bool = False  # 调用行为完全正确：该调的全调对且没多余调用；不该调的没调、也没编造


_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")
# 不需要工具的闲聊任务：内容无法自动判分，只因"正确地没调工具"给部分分（满分留给可验证的任务）
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
    """函数名相同，且参数在规范化后逐项相等（字符串去空白、忽略大小写；数字按数值比）。"""
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
    """执行结果一致（calculator 另要求数字的多重集合一致，防止心算后直接传答案）。"""
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
    """把参数换成规范形式（日期解析成 ISO、单位和城市换成标准名、数字按数值）；参数非法返回 None。"""
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
    """参数"意思相同"：规范化后逐项相等。

    只有 calculator 用执行结果判（外加数字多重集合一致），因为等价的算式写法很多；
    其余工具必须参数本身等价——只看执行结果会放过"参数错、结果碰巧对"的调用
    （例如 weekday 的日期差 7 天，星期几一样；第 17、19 章的真实案例）。
    """
    if pred["name"] != gold["name"]:
        return False
    if pred["name"] == "calculator":
        return exec_equal(pred, gold)
    cp = canonical_args(pred["name"], pred["arguments"])
    return cp is not None and cp == canonical_args(gold["name"], gold["arguments"])


def _invented_numbers(answer: str, query: str) -> bool:
    """回答里出现了问题里没有的数字：不需要工具的闲聊任务上，这通常是在编造工具结果。"""
    allowed = set(_NUM_RE.findall(query))
    return any(n not in allowed for n in _NUM_RE.findall(answer))


def _forged(text: str) -> str | None:
    for tag in ("<tool_response>", "</tool_response>", "<|im_start|>"):
        if tag in text:
            return f"输出里出现了 {tag}（伪造工具结果或对话轮次）"
    return None


_BARE_CALL_RE = re.compile(r'\{\s*"(?:name|arguments)\b|"arguments"\s*:')


def _bare_call(content: str) -> str | None:
    """标签外面出现了工具调用样子的 JSON：判格式错误（否则去掉标签就能躲开 -1 的格式分）。"""
    if _BARE_CALL_RE.search(content):
        return "工具调用缺少 <tool_call> 标签（标签外出现了 name / arguments JSON）"
    return None


def score_tool_calls(task: Task, text: str) -> Reward:
    """给"助手的第一轮输出"打分：该调用哪些工具、参数对不对。"""
    if len(text) > MAX_OUTPUT_CHARS:
        return Reward(-1.0, False, details=["输出过长"])
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
        return Reward(-1.0, False, n_calls=len(calls), details=[f"调用超过 {MAX_CALLS} 个"])
    offered = {t["function"]["name"] for t in task.tools}
    valid, details = [], []
    n_invalid = 0
    for c in calls:
        errs = validate_arguments(c["name"], c["arguments"])
        if c["name"] not in offered:
            errs = [f"调用了未提供的工具 {c['name']}", *errs]
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
                return Reward(0.0, True, 0, details=["空回答"])
            if _invented_numbers(content, task.query):
                return Reward(
                    0.0,
                    True,
                    0,
                    answer_ok=False,
                    details=["没调工具，但回答里有问题中没有的数字（疑似编造）"],
                )
            # 没调工具是对的；但闲聊内容本身无法自动核对，只给部分分，answer_ok 记为"未知"
            return Reward(
                NO_TOOL_REWARD,
                True,
                0,
                1.0,
                1.0,
                answer_ok=None,
                details=["正确地没有调用工具（回答内容无法自动核对）"],
                exact=True,
            )
        return Reward(-0.5, format_ok, len(calls), details=["不需要工具却调用了", *details])
    if not calls:
        return Reward(0.0, True, 0, details=["需要调用工具但没有调用"])

    # 一对一匹配：先配 AST 一致的，再配执行一致的，最后配只有函数名一致的
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
            details.append(f"{g['name']} 参数不对")
        else:
            details.append(f"缺少调用 {g['name']}")
    n_extra = len(remaining)
    if n_extra:
        details.append(f"多余调用 {n_extra} 个")
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
    allowed = sum(len(_NUM_RE.findall(f)) for f in facts) + 6  # 复述问题里的数字是正常的
    return len(_NUM_RE.findall(answer)) > allowed


def score_final_answer(task: Task, text: str) -> Reward:
    """给最终回答打分：不能再调用工具；所有要点都出现；不能靠罗列一堆数字碰运气。"""
    if len(text) > MAX_OUTPUT_CHARS:
        return Reward(-1.0, False, answer_ok=False, details=["输出过长"])
    forged = _forged(text)
    parsed = parse_assistant(text)
    if forged or parsed.errors:
        return Reward(-1.0, False, answer_ok=False, details=[forged or "", *parsed.errors])
    if parsed.tool_calls:
        return Reward(
            0.0, True, len(parsed.tool_calls), answer_ok=False, details=["最终回答里还在调用工具"]
        )
    bare = _bare_call(parsed.content)
    if bare:
        return Reward(-1.0, False, answer_ok=False, details=[bare])
    ans = parsed.content.strip()
    if not ans:
        return Reward(0.0, True, answer_ok=False, details=["空回答"])
    if not task.answer_facts:
        if _invented_numbers(ans, task.query):
            return Reward(
                0.0, True, answer_ok=False, details=["回答里有问题中没有的数字（疑似编造）"]
            )
        return Reward(NO_TOOL_REWARD, True, answer_ok=None, details=["回答内容无法自动核对"])
    if _too_many_numbers(ans, task.answer_facts):
        return Reward(0.0, True, answer_ok=False, details=["回答里数字过多"])
    norm = ans.replace(" ", "").lower()
    ok = all(f.replace(" ", "").lower() in norm for f in task.answer_facts)
    return Reward(1.0 if ok else 0.0, True, answer_ok=ok, details=[] if ok else ["缺少要点"])


# ---------------------------------------------------------------------------
# 完整一轮：生成 → 解析 → 执行 → 喂回 → 最终回答
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
        # 闲聊任务的回答内容无法核对（answer_ok 为 None），只要调用行为对、没被判为编造就算成功
        return self.call_reward.exact and self.answer_reward.answer_ok is not False


def run_episode(policy: Policy, task: Task, max_turns: int = 3) -> Episode:
    """让 policy（输入 messages 和 tools，输出助手文本）完成一个任务。

    第一轮输出按 `score_tool_calls` 打分；之后执行其中的调用、把结果作为 tool 消息喂回，
    直到某一轮不再调用工具（即最终回答），按 `score_final_answer` 打分。
    """
    msgs = [dict(m) for m in task.messages]
    call_reward: Reward | None = None
    answer_reward = Reward(0.0, True, answer_ok=False, details=["轮数用完"])
    turns = 0
    while turns < max_turns:
        turns += 1
        text = policy(msgs, task.tools)
        if call_reward is None:
            call_reward = score_tool_calls(task, text)
        parsed = parse_assistant(text)
        if parsed.errors or _forged(text):
            msgs.append({"role": "assistant", "content": text})
            answer_reward = Reward(-1.0, False, answer_ok=False, details=["格式错误，结束"])
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
