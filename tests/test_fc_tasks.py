"""Real function-calling tasks (Chapter 19): format, schema check, reward, converters, decontamination.

The reward keeps the scale and the anti-hacking rules of tool_env. Each converter is checked on a
hand-written row in the published format of its data set. GRPO and the eval harness run on a task file.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from tests.conftest import post_config
from zero.post.envs.fc_tasks import (
    FCTask,
    build,
    call_matches,
    decontaminate_tasks,
    from_bfcl,
    from_openai_messages,
    from_xlam,
    load_fc_tasks,
    load_task_pool,
    schema_errors,
    score_any,
    score_fc,
    task_errors,
    values_match,
)

WEATHER = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Weather of a city",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "default": "celsius"},
                "days": {"type": "integer"},
            },
            "required": ["city"],
        },
    },
}
STOCK = {
    "type": "function",
    "function": {
        "name": "get_stock",
        "parameters": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string"},
                "tags": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["symbol"],
        },
    },
}


def call(name: str, **args) -> str:  # noqa: ANN003
    return "<tool_call>\n" + json.dumps({"name": name, "arguments": args}) + "\n</tool_call>"


def task(
    gold: list[dict], tools=(WEATHER, STOCK), q: str = "What is the weather in Paris?"
) -> FCTask:  # noqa: ANN001
    return FCTask("t", list(tools), [{"role": "user", "content": q}], gold)


PARIS = {
    "name": "get_weather",
    "arguments": {"city": "Paris"},
    "alternatives": {"unit": ["celsius", ""]},
}


# --------------------------------------------------------------------- schema and values


def test_schema_errors() -> None:
    p = WEATHER["function"]["parameters"]
    assert schema_errors({"city": "Paris"}, p) == []
    assert any("missing required" in e for e in schema_errors({}, p))
    assert any("unknown argument" in e for e in schema_errors({"city": "x", "country": "FR"}, p))
    assert any("wrong type" in e for e in schema_errors({"city": 3}, p))
    assert any("enum" in e for e in schema_errors({"city": "x", "unit": "kelvin"}, p))
    assert schema_errors({"city": "x", "days": 3.0}, p) == []  # 3.0 is an integer value
    assert any(
        "wrong type" in e for e in schema_errors({"city": "x", "days": True}, p)
    )  # bool is not an int
    s = STOCK["function"]["parameters"]
    assert any("[1]" in e for e in schema_errors({"symbol": "A", "tags": ["x", 2]}, s))
    # BFCL type names
    assert (
        schema_errors(
            {"a": 1.5, "b": {}},
            {"type": "dict", "properties": {"a": {"type": "float"}, "b": {"type": "dict"}}},
        )
        == []
    )


def test_values_match() -> None:
    assert values_match(" Paris ", "paris") and values_match(1, 1.0) and not values_match(True, 1)
    assert values_match([1, "A"], [1.0, "a"]) and not values_match([1, 2], [2, 1])
    assert values_match({"x": "A"}, {"x": "a"}) and not values_match({"x": 1}, {"x": 1, "y": 2})


def test_call_matches_alternatives_and_defaults() -> None:
    f = WEATHER["function"]
    assert call_matches({"name": "get_weather", "arguments": {"city": "paris"}}, PARIS, f)
    assert call_matches(
        {"name": "get_weather", "arguments": {"city": "Paris", "unit": "celsius"}}, PARIS, f
    )
    assert not call_matches(
        {"name": "get_weather", "arguments": {"city": "Paris", "unit": "fahrenheit"}}, PARIS, f
    )
    gold = {"name": "get_weather", "arguments": {"city": "Paris"}}  # no alternatives
    # An argument that the gold call does not mention: accepted only with its default value
    assert call_matches(
        {"name": "get_weather", "arguments": {"city": "Paris", "unit": "celsius"}}, gold, f
    )
    assert not call_matches(
        {"name": "get_weather", "arguments": {"city": "Paris", "days": 3}}, gold, f
    )
    assert not call_matches({"name": "get_weather", "arguments": {}}, gold, f)
    # The gold call gives an optional argument at its default value: leaving it out is the same call
    explicit = {"name": "get_weather", "arguments": {"city": "Paris", "unit": "celsius"}}
    assert call_matches({"name": "get_weather", "arguments": {"city": "Paris"}}, explicit, f)
    days = {
        "name": "get_weather",
        "arguments": {"city": "Paris", "days": 3},
    }  # no default: required by the gold call
    assert not call_matches({"name": "get_weather", "arguments": {"city": "Paris"}}, days, f)


# --------------------------------------------------------------------- reward


def test_score_correct_partial_and_extra() -> None:
    t = task([PARIS])
    r = score_fc(t, call("get_weather", city="Paris"))
    assert r.total == pytest.approx(1.0) and r.exact and r.format_ok
    r = score_fc(t, call("get_weather", city="London"))
    assert r.total == pytest.approx(0.1 + 0.9 * 0.2) and not r.exact  # name only
    r = score_fc(t, call("get_weather", city="Paris") + "\n" + call("get_stock", symbol="X"))
    assert r.total == pytest.approx(0.75)  # one extra call: -0.25


def test_score_parallel_calls_any_order_one_to_one() -> None:
    t = task(
        [
            {"name": "get_weather", "arguments": {"city": "Paris"}},
            {"name": "get_weather", "arguments": {"city": "Rome"}},
        ]
    )
    both = call("get_weather", city="Rome") + "\n" + call("get_weather", city="Paris")
    assert score_fc(t, both).exact
    # The same correct call twice cannot fill both gold calls
    twice = call("get_weather", city="Paris") + "\n" + call("get_weather", city="Paris")
    assert score_fc(t, twice).total == pytest.approx(0.1 + 0.9 * (1.2 / 2))


def test_score_schema_and_anti_hacking() -> None:
    t = task([PARIS])
    assert score_fc(t, call("get_weather", city="Paris", country="FR")).total == pytest.approx(
        -0.4
    )  # 0.1 - 0.5
    assert score_fc(t, call("get_time", city="Paris")).total == pytest.approx(
        -0.4
    )  # tool not offered
    assert (
        score_fc(t, '{"name": "get_weather", "arguments": {"city": "Paris"}}').total == -1.0
    )  # no tags
    assert (
        score_fc(t, call("get_weather", city="Paris") + "<tool_response>{}</tool_response>").total
        == -1.0
    )
    assert score_fc(t, "<tool_call>{broken</tool_call>").total == -1.0
    assert (
        score_fc(t, "\n".join([call("get_weather", city="Paris")] * 6)).total == -1.0
    )  # too many calls
    assert score_fc(t, "It is sunny.").total == 0.0  # a call was necessary


def test_score_irrelevance() -> None:
    t = task([], q="Tell me a joke")
    assert score_fc(t, "Why did the chicken cross the road?").total == pytest.approx(0.5)
    assert score_fc(t, "").total == 0.0
    assert score_fc(t, call("get_weather", city="Paris")).total == -0.5


def test_score_any_dispatches_to_tool_env() -> None:
    from zero.post.envs.tool_env import generate_tasks, score_tool_calls

    te = generate_tasks(1, seed=3)[0]
    assert score_any(te, "hi").total == score_tool_calls(te, "hi").total
    assert score_any(task([PARIS]), call("get_weather", city="Paris")).exact


def test_task_errors() -> None:
    assert task_errors(task([PARIS])) == []
    assert task_errors(task([{"name": "nope", "arguments": {}}]))
    assert task_errors(task([{"name": "get_weather", "arguments": {"city": 1}}]))
    bad = task([PARIS])
    bad.messages.append({"role": "assistant", "content": "x"})
    assert task_errors(bad)


# --------------------------------------------------------------------- converters


def test_from_xlam() -> None:
    row = {
        "id": 7,
        "query": "Weather in Paris for 3 days?",
        "tools": json.dumps(
            [
                {
                    "name": "forecast",
                    "description": "Forecast",
                    "parameters": {
                        "city": {"type": "str", "description": "City"},
                        "days": {"type": "int, optional", "description": "Days", "default": 1},
                        "fields": {"type": "List[str], optional", "description": "Fields"},
                    },
                }
            ]
        ),
        "answers": json.dumps([{"name": "forecast", "arguments": {"city": "Paris", "days": 3}}]),
    }
    t = from_xlam(row, 0)
    p = t.tools[0]["function"]["parameters"]
    assert p["required"] == ["city"]
    assert p["properties"]["days"] == {"type": "integer", "description": "Days", "default": 1}
    assert p["properties"]["fields"]["type"] == "array"
    assert t.id == "xlam-7" and task_errors(t) == []
    assert score_fc(t, call("forecast", city="Paris", days=3)).exact


def test_from_openai_messages() -> None:
    row = {
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Price of AAPL?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {"name": "get_stock", "arguments": '{"symbol": "AAPL"}'},
                    }
                ],
            },
            {"role": "tool", "content": "{}"},
        ],
        "tools": [STOCK],
    }
    t = from_openai_messages(row, 0, "src", "Apache-2.0")
    assert [m["role"] for m in t.messages] == ["system", "user"]
    assert t.gold_calls == [{"name": "get_stock", "arguments": {"symbol": "AAPL"}}]
    assert (
        from_openai_messages({"messages": [{"role": "user", "content": "x"}]}, 0, "s", "") is None
    )


def test_from_bfcl_with_possible_answers() -> None:
    q = {
        "id": "simple_0",
        "question": [
            [{"role": "user", "content": "Area of a triangle with base 10 and height 5?"}]
        ],
        "function": [
            {
                "name": "calc_area",
                "description": "Area",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "base": {"type": "integer"},
                        "height": {"type": "integer"},
                        "unit": {"type": "string"},
                    },
                    "required": ["base", "height"],
                },
            }
        ],
    }
    a = {
        "id": "simple_0",
        "ground_truth": [{"calc_area": {"base": [10], "height": [5], "unit": ["units", ""]}}],
    }
    t = from_bfcl(q, a)
    assert t.tools[0]["function"]["parameters"]["type"] == "object"
    assert t.gold_calls[0]["arguments"] == {"base": 10, "height": 5, "unit": "units"}
    assert task_errors(t) == []
    assert score_fc(t, call("calc_area", base=10, height=5)).exact
    assert score_fc(t, call("calc_area", base=10, height=5, unit="units")).exact
    assert not score_fc(t, call("calc_area", base=10, height=6)).exact
    assert from_bfcl(q, None).gold_calls == []  # irrelevance item


# --------------------------------------------------------------------- build and decontamination


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_decontaminate_by_tool_name_and_ngram() -> None:
    ev = [
        task(
            [PARIS],
            tools=(WEATHER,),
            q="What is the weather like in the beautiful old city of Paris today please tell me",
        )
    ]
    same_tool = task([PARIS], tools=(WEATHER,), q="weather?")
    same_text = task(
        [{"name": "get_stock", "arguments": {"symbol": "A"}}],
        tools=(STOCK,),
        q="Hi! What is the weather like in the beautiful old city of Paris today please tell me now",
    )
    clean = task(
        [{"name": "get_stock", "arguments": {"symbol": "A"}}], tools=(STOCK,), q="Price of A?"
    )
    kept, dropped = decontaminate_tasks([same_tool, same_text, clean], ev)
    assert kept == [clean] and dropped == {"tool_name": 1, "ngram": 1}


def test_build_end_to_end(tmp_path: Path) -> None:
    rows = [
        {
            "query": f"Weather in city {i}?",
            "tools": [{"name": "forecast", "parameters": {"city": {"type": "str"}}}],
            "answers": [{"name": "forecast", "arguments": {"city": f"c{i}"}}],
        }
        for i in range(10)
    ]
    rows.append(rows[0])  # duplicate
    rows.append(
        {
            "query": "bad",
            "tools": [{"name": "f", "parameters": {}}],
            "answers": [{"name": "g", "arguments": {}}],
        }
    )
    _write(tmp_path / "x.jsonl", rows)
    meta = build(
        [("xlam", str(tmp_path / "x.jsonl"), "CC-BY-4.0")],
        tmp_path / "train.jsonl",
        dev_out=tmp_path / "dev.jsonl",
        dev_size=3,
    )
    assert meta["n_train"] == 7 and meta["n_dev"] == 3
    assert meta["dropped"] == {"duplicate": 1, "invalid": 1}
    assert meta["licenses"] == {"xlam": "CC-BY-4.0"}
    assert len(load_fc_tasks(tmp_path / "train.jsonl", strict=True)) == 7
    assert json.loads((tmp_path / "train.jsonl.meta.json").read_text())["n_dev"] == 3


def test_cli_build_parses_license(tmp_path: Path) -> None:
    from zero.post.envs.fc_tasks import main

    _write(
        tmp_path / "x.jsonl",
        [
            {
                "query": "q",
                "tools": [{"name": "f", "parameters": {}}],
                "answers": [{"name": "f", "arguments": {}}],
            }
        ],
    )
    main(["build", "--src", f"xlam:{tmp_path / 'x.jsonl'}:MIT", "--out", str(tmp_path / "o.jsonl")])
    assert load_fc_tasks(tmp_path / "o.jsonl")[0].license == "MIT"


# --------------------------------------------------------------------- training and evaluation


def _fc_file(tmp_path: Path, n: int = 6) -> Path:
    tasks = [
        task([{"name": "get_weather", "arguments": {"city": f"C{i}"}}], q=f"Weather in C{i}?")
        for i in range(n)
    ]
    tasks.append(task([], q="Hello"))
    p = tmp_path / "fc.jsonl"
    _write(p, [t.to_dict() for t in tasks])
    return p


def test_load_task_pool(tmp_path: Path) -> None:
    assert len(load_task_pool([str(_fc_file(tmp_path))], 0, 0)) == 7
    assert len(load_task_pool([], 5, 0)) == 5  # no files: tool_env


def test_run_grpo_on_task_file(tmp_path: Path, chat_tok, chat_tok_path, tiny_ckpt) -> None:  # noqa: ANN001
    from zero.post.grpo import run_grpo

    d = post_config(
        tmp_path,
        chat_tok_path,
        tiny_ckpt,
        chat_tok.vocab_size,
        grpo={
            "group_size": 3,
            "prompts_per_step": 2,
            "max_new_tokens": 10,
            "task_files": [str(_fc_file(tmp_path))],
        },
    )
    h = run_grpo(d, log=lambda _: None)[-1]
    assert h["step"] == 2 and math.isfinite(h["reward_mean"]) and -1.0 <= h["reward_mean"] <= 1.0


def test_eval_harness_fc_tasks(tmp_path: Path, tiny_ckpt) -> None:  # noqa: ANN001
    from zero.eval.harness import EvalConfig, EvalModel, run_eval

    ec = EvalConfig(
        models=[EvalModel("a", str(tiny_ckpt)), EvalModel("b", str(tiny_ckpt))],
        fc_tasks=[str(_fc_file(tmp_path))],
        max_new_tokens=8,
        out_dir=str(tmp_path / "eval"),
        baseline="a",
        n_boot=200,
    )
    out = run_eval(ec, log=lambda _: None)
    r = out["results"]["a"]["fc"]
    assert r["type"] == "fc" and r["n"] == 7 and 0.0 <= r["call_exact"] <= 1.0
    assert out["comparisons"][0]["task"] == "fc" and out["comparisons"][0]["diff"] == 0.0


# --------------------------------------------------------------------- Hermes and ToolACE (rows copied from the data sets)

HERMES_ROW = {
    "id": "89ef3c87",
    "conversations": [
        {
            "from": "system",
            "value": "You are a function calling AI model. ... <tools>\n[]\n</tools>",
        },
        {
            "from": "human",
            "value": "Initialize my smart home with a Nest Thermostat and group it as 'Main Control Group'.",
        },
        {
            "from": "gpt",
            "value": '<tool_call>\n{"name": "initialize_smart_home_system", "arguments": {"device_list": ["Nest Thermostat"]}}\n</tool_call>\n'
            "<tool_call>\n{'arguments': {'group_name': 'Main Control Group', 'devices': ['Nest Thermostat']}, 'name': 'create_device_group'}\n</tool_call>\n",
        },
        {
            "from": "tool",
            "value": '<tool_response>\n{"name": "initialize_smart_home_system", "content": {"ok": true}}\n</tool_response>\n',
        },
        {"from": "gpt", "value": "Done: the system is initialized and the group is created."},
    ],
    "tools": json.dumps(
        [
            {
                "type": "function",
                "function": {
                    "name": "initialize_smart_home_system",
                    "description": "Init.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "device_list": {"type": "array", "items": {"type": "string"}}
                        },
                        "required": ["device_list"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "create_device_group",
                    "description": "Group.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "group_name": {"type": "string"},
                            "devices": {"type": "array", "items": {"type": "string"}},
                        },
                        "required": ["group_name", "devices"],
                    },
                },
            },
        ]
    ),
}


def test_from_hermes_one_task_per_assistant_turn() -> None:
    from zero.post.envs.fc_tasks import from_hermes

    t1, t2 = from_hermes(HERMES_ROW, 0)
    assert [m["role"] for m in t1.messages] == [
        "user"
    ]  # the data set's own system prompt is dropped
    assert {c["name"] for c in t1.gold_calls} == {
        "initialize_smart_home_system",
        "create_device_group",
    }
    assert task_errors(t1) == [] and task_errors(t2) == []
    assert [m["role"] for m in t2.messages] == ["user", "assistant", "tool"] and t2.gold_calls == []
    assert t2.messages[2]["content"].startswith('{"name"')  # <tool_response> tags removed
    both = (
        call("create_device_group", group_name="Main Control Group", devices=["Nest Thermostat"])
        + "\n"
        + call("initialize_smart_home_system", device_list=["Nest Thermostat"])
    )
    assert score_fc(t1, both).exact
    # The history renders with our template (tool_calls in the assistant turn)
    from zero.post.chat import render_text

    assert "<tool_response>" in render_text(t2.messages, t2.tools, add_generation_prompt=True)


TOOLACE_ROW = {
    "system": "You are an expert in composing functions. ...\nHere is a list of functions in JSON format that you can invoke:\n"
    + json.dumps(
        [
            {
                "name": "Market Trends API",
                "description": "Trends.",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "trend_type": {
                            "description": "Trend type.",
                            "type": "string",
                            "enum": ["MARKET_INDEXES", "CRYPTO"],
                        },
                        "country": {"description": "Country.", "type": "string", "default": "us"},
                    },
                    "required": ["trend_type"],
                },
                "required": None,
            },
            {
                "name": "SEC Filings",
                "description": "Filings.",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "identifier": {
                            "description": "Symbol.",
                            "type": "string",
                            "default": "aapl",
                        }
                    },
                    "required": ["identifier"],
                },
                "required": None,
            },
        ]
    )
    + ". \nShould you decide to return the function call(s). \nPut it in the format of [func1(params_name=params_value, params_name2=params_value2...), func2(params)]\n\nNO other text MUST be included. \n",
    "conversations": [
        {"from": "user", "value": "Top market trends in the US?"},
        {
            "from": "assistant",
            "value": '[Market Trends API(trend_type="MARKET_INDEXES", country="us")]',
        },
        {"from": "tool", "value": '[{"name": "Market Trends API", "results": {}}]'},
        {"from": "assistant", "value": "Here are the trends."},
        {"from": "user", "value": "苹果公司的 SEC 文件呢？再看看加密货币趋势。"},
        {
            "from": "assistant",
            "value": '[SEC Filings(identifier="AAPL"), Market Trends API(trend_type="CRYPTO")]',
        },
    ],
}


def test_parse_python_calls() -> None:
    from zero.post.envs.fc_tasks import parse_python_calls

    assert parse_python_calls('[Get Data(a="x, y)", b=[1, 2], c={"k": True}), g()]') == [
        {"name": "Get Data", "arguments": {"a": "x, y)", "b": [1, 2], "c": {"k": True}}},
        {"name": "g", "arguments": {}},
    ]
    assert parse_python_calls("Here are the trends.") is None
    assert parse_python_calls("[f(__import__('os'))]") is None  # positional / code: refused
    assert parse_python_calls("[f(a=open('x'))]") is None  # not a literal


def test_from_toolace() -> None:
    from zero.post.envs.fc_tasks import from_toolace

    t1, t2, t3 = from_toolace(TOOLACE_ROW, 0)
    assert t1.tools[0]["function"]["parameters"]["type"] == "object"  # "dict" converted
    assert t1.gold_calls == [
        {
            "name": "Market Trends API",
            "arguments": {"trend_type": "MARKET_INDEXES", "country": "us"},
        }
    ]
    assert t2.gold_calls == [] and t2.messages[-1]["role"] == "tool"
    assert t3.lang == "zh" and [c["name"] for c in t3.gold_calls] == [
        "SEC Filings",
        "Market Trends API",
    ]
    assert all(task_errors(t) == [] for t in (t1, t2, t3))
    assert score_fc(
        t1, call("Market Trends API", trend_type="MARKET_INDEXES")
    ).exact  # country="us" is the default


def test_build_hermes_and_toolace(tmp_path: Path) -> None:
    (tmp_path / "h.json").write_text(json.dumps([HERMES_ROW]))
    (tmp_path / "t.json").write_text(json.dumps([TOOLACE_ROW]))
    meta = build(
        [("hermes", str(tmp_path / "h.json"), ""), ("toolace", str(tmp_path / "t.json"), "")],
        tmp_path / "o.jsonl",
    )
    assert meta["per_source"] == {"hermes": 2, "toolace": 3}
    assert meta["licenses"] == {"hermes": "Apache-2.0", "toolace": "Apache-2.0"}
    assert meta["n_irrelevance"] == 2 and meta["n_parallel"] == 2


def test_to_sft_renders_and_split_does_not_overlap(tmp_path: Path, chat_tok) -> None:  # noqa: ANN001
    from zero.post.chat import render
    from zero.post.envs.fc_tasks import main, split_sft_rl, to_sft

    t = task([PARIS])
    row = to_sft(t)
    ids, mask = render(row["messages"], row["tools"], tokenizer=chat_tok)
    assert any(mask) and "get_weather" in chat_tok.decode([i for i, m in zip(ids, mask) if m])
    assert to_sft(task([])) is None
    tasks = [
        task([{"name": "get_weather", "arguments": {"city": f"C{i}"}}], q=f"q{i}")
        for i in range(10)
    ]
    for i, t in enumerate(tasks):
        t.id = f"t{i}"
    sft, rl = split_sft_rl(tasks, 0.3, seed=1)
    assert len(sft) == 3 and len(rl) == 7
    assert not {r["task_id"] for r in sft} & {t.id for t in rl}
    p = tmp_path / "all.jsonl"
    _write(p, [t.to_dict() for t in tasks])
    main(
        [
            "split",
            str(p),
            "--sft-out",
            str(tmp_path / "s.jsonl"),
            "--rl-out",
            str(tmp_path / "r.jsonl"),
            "--sft-frac",
            "0.5",
        ]
    )
    assert len(load_fc_tasks(tmp_path / "r.jsonl")) == 5
