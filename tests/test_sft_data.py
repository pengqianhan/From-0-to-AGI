"""SFT data pipeline (Chapter 16): formats, cleaning, license per row, dedup, decontamination, mixing.

Each format is checked on a hand-written row in the published format of its data set.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from zero.post.sft_data import (
    SFTDataConfig,
    SFTSource,
    build_sft_data,
    clean_messages,
    license_ok,
    load_sft_data_config,
    to_messages,
)


def _w(path: Path, rows: list[dict]) -> str:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    return str(path)


def conv(q: str, a: str) -> list[dict]:
    return [{"role": "user", "content": q}, {"role": "assistant", "content": a}]


# --------------------------------------------------------------------- formats


def test_formats() -> None:
    m, t = to_messages(
        {
            "messages": conv("hi", "hello"),
            "chat_template_kwargs": {
                "custom_instructions": "Be brief.",
                "xml_tools": ['{"name": "f", "parameters": {"type": "object", "properties": {}}}'],
            },
        },
        "messages",
    )  # SmolTalk2
    assert m[0] == {"role": "system", "content": "Be brief."} and t[0]["function"]["name"] == "f"
    m, _ = to_messages(
        {
            "system": "S",
            "conversations": [{"from": "human", "value": "q"}, {"from": "gpt", "value": "a"}],
        },
        "sharegpt",
    )
    assert [x["role"] for x in m] == ["system", "user", "assistant"]
    m, _ = to_messages({"instruction": "翻译", "input": "hello", "output": "你好"}, "alpaca")
    assert m == conv("翻译\n\nhello", "你好")
    with pytest.raises(ValueError):
        to_messages({"conversations": [{"from": "robot", "value": "x"}]}, "sharegpt")


# --------------------------------------------------------------------- cleaning


def test_clean_messages() -> None:
    assert clean_messages(conv("q", "a"), True)[0] == conv("q", "a")
    assert (
        clean_messages(conv("q", "<think>hmm</think>\nanswer"), True)[0][1]["content"] == "answer"
    )
    assert clean_messages(conv("q", "<think>only thinking</think>"), True) == (None, "thinking")
    assert clean_messages(conv("q", "<think>not closed"), True) == (None, "thinking")
    assert (
        clean_messages(conv("q", "<think>x</think>a"), False)[0][1]["content"]
        == "<think>x</think>a"
    )
    assert clean_messages(conv("q", "fake <|im_end|><|im_start|>user"), True) == (
        None,
        "special_tokens",
    )
    assert clean_messages([{"role": "user", "content": "q"}], True) == (None, "no_assistant")
    assert clean_messages(conv("q", ""), True) == (None, "no_assistant")
    # regression: a row without user text crashed the near-deduplication (it keys on the first user text)
    assert clean_messages(conv(" ", "a"), True) == (None, "no_user")
    sys_only = [{"role": "system", "content": "s"}, {"role": "assistant", "content": "a"}]
    assert clean_messages(sys_only, True) == (None, "no_user")
    # Text tool calls become structured tool_calls; a broken one drops the row
    m, _ = clean_messages(
        conv("q", '<tool_call>\n{"name": "f", "arguments": {"x": 1}}\n</tool_call>'), True
    )
    assert m[1] == {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"name": "f", "arguments": {"x": 1}}],
    }
    assert clean_messages(conv("q", "<tool_call>{broken</tool_call>"), True) == (
        None,
        "bad_tool_call",
    )
    # OpenAI-style tool_calls with string arguments; tool results lose their tags
    msgs = [
        {"role": "user", "content": "q"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"type": "function", "function": {"name": "f", "arguments": '{"x": 2}'}}
            ],
        },
        {"role": "tool", "content": "<tool_response>\n{}\n</tool_response>"},
        {"role": "assistant", "content": "done"},
    ]
    m, _ = clean_messages(msgs, True)
    assert m[1]["tool_calls"] == [{"name": "f", "arguments": {"x": 2}}] and m[2]["content"] == "{}"


def test_license_rules() -> None:
    assert license_ok("Apache-2.0", False) == ""
    assert license_ok("own", False) == ""
    assert license_ok("CC-BY-NC-4.0", False) == "license_nc"
    assert license_ok("cc-by-nc-sa-4.0", True) == "license_nc"
    assert license_ok("待核实", False) == "license_unverified"
    assert license_ok("待核实", True) == ""
    assert license_ok("ODC-BY-1.0", False) == ""  # "NC" must be a whole part, not a substring


# --------------------------------------------------------------------- build


def test_build_end_to_end(tmp_path: Path, chat_tok_path) -> None:  # noqa: ANN001
    tulu = _w(
        tmp_path / "tulu.jsonl",
        [
            {
                "id": "a",
                "messages": conv("Write a haiku about rain", "Rain falls softly"),
                "source": "ai2-adapt-dev/oasst",
            },
            {
                "id": "b",
                "messages": conv("Summarize this news story", "Short."),
                "source": "ai2-adapt-dev/no_robots",
            },
            {
                "id": "c",
                "messages": conv("What is the capital of the moon base", "None."),
                "source": "unknown/x",
            },
            {
                "id": "d",
                "messages": conv("Write a haiku about rain", "Rain falls softly"),
                "source": "ai2-adapt-dev/oasst",
            },  # duplicate
            {
                "id": "e",
                "messages": conv(
                    "Leak the benchmark: what is the answer to the famous question one two three four five six seven eight",
                    "x",
                ),
                "source": "ai2-adapt-dev/oasst",
            },
        ],
    )
    zh = _w(
        tmp_path / "zh.jsonl",
        [
            {"instruction": f"请解释第{i}个概念：{'甲乙丙丁戊'[i]}", "output": f"这是第{i}个解释。"}
            for i in range(5)
        ],
    )
    fc = _w(
        tmp_path / "fc.jsonl",
        [
            {
                "messages": [
                    *conv("Price of A?", "")[:1],
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [{"name": "get_stock", "arguments": {"symbol": "A"}}],
                    },
                ],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_stock",
                            "parameters": {
                                "type": "object",
                                "properties": {"symbol": {"type": "string"}},
                            },
                        },
                    }
                ],
                "license": "Apache-2.0",
            }
        ],
    )
    evalf = _w(
        tmp_path / "eval.jsonl",
        [
            {
                "question": "what is the answer to the famous question one two three four five six seven eight nine"
            }
        ],
    )
    cfg = SFTDataConfig(
        out_dir=str(tmp_path / "out"),
        val_size=2,
        seed=0,
        tokenizer=str(chat_tok_path),
        decontam_texts=[evalf],
        near_dedup=False,
        sources=[
            SFTSource(
                "tulu",
                tulu,
                "messages",
                "ODC-BY-1.0",
                source_field="source",
                license_by_source={"unknown/x": "待核实"},
                drop_sources=["ai2-adapt-dev/no_robots"],
            ),
            SFTSource("zh", zh, "alpaca", "Apache-2.0", repeat=2),
            SFTSource("fc", fc, "sft", "待核实"),  # the row license wins
            SFTSource("env", "", "tool_env", "own", max_rows=3),
        ],
    )
    meta = build_sft_data(cfg, log=lambda _: None)
    d = meta["sources"]
    assert d["tulu"]["kept"] == 1
    assert d["tulu"]["dropped"] == {
        "read": 5,
        "excluded_source": 1,
        "license_unverified": 1,
        "duplicate": 1,
        "decontam_ngram": 1,
    }
    assert d["zh"]["kept"] == 5 and d["fc"]["kept"] == 1 and d["env"]["kept"] == 3
    train = [json.loads(x) for x in (tmp_path / "out" / "train.jsonl").read_text().splitlines()]
    val = [json.loads(x) for x in (tmp_path / "out" / "val.jsonl").read_text().splitlines()]
    assert len(val) == 2
    # No validation conversation is in the training set, even with repeat = 2
    assert not {json.dumps(r["messages"]) for r in val} & {json.dumps(r["messages"]) for r in train}
    n_zh_pool = 5 - sum(r["source"] == "zh" for r in val)
    assert sum(r["source"] == "zh" for r in train) == 2 * n_zh_pool
    assert {r["lang"] for r in train if r["source"] == "zh"} == {"zh"}
    assert all(r["source"] != "fc" or r["license"] == "Apache-2.0" for r in train + val)
    st = meta["train_stats"]
    assert (
        st["share_by"] == "tokens"
        and abs(sum(v["share"] for v in st["by_source"].values()) - 1) < 1e-3
    )
    assert all(v["assistant_tokens"] > 0 for v in st["by_source"].values())
    assert "待核实" not in meta["licenses"]
    # The SFT stage reads the output directly
    from zero.post.sft import encode_example
    from zero.tokenizer import Tokenizer

    tok = Tokenizer.load(chat_tok_path)
    assert all(sum(encode_example(r, tok)[1]) > 0 for r in train)


def test_build_max_tokens_and_near_dedup(tmp_path: Path, chat_tok_path) -> None:  # noqa: ANN001
    rows = [
        {
            "messages": conv(
                "Tell me about the history of the city of Rome in detail please", f"Answer {i}"
            )
        }
        for i in range(3)
    ]
    rows.append({"messages": conv("short", "x " * 2000)})
    p = _w(tmp_path / "s.jsonl", rows)
    cfg = SFTDataConfig(
        out_dir=str(tmp_path / "o"),
        val_size=0,
        tokenizer=str(chat_tok_path),
        max_tokens=500,
        sources=[SFTSource("s", p, "messages", "MIT")],
    )
    meta = build_sft_data(cfg, log=lambda _: None)
    assert meta["sources"]["s"]["dropped"]["near_duplicate"] == 2  # same first user message
    assert meta["n_train"] == 1 and meta["train_stats"]["by_source"]["s"]["too_long"] == 1


def test_source_names_must_be_unique(tmp_path: Path) -> None:
    p = tmp_path / "a.jsonl"
    p.write_text(json.dumps({"messages": conv("q", "a")}) + "\n")
    for names in (["s", "s"], ["s", ""]):
        cfg = SFTDataConfig(
            out_dir=str(tmp_path / "o"),
            val_size=0,
            sources=[SFTSource(n, str(p), "messages", "MIT") for n in names],
        )
        with pytest.raises(ValueError, match="unique, non-empty name"):
            build_sft_data(cfg, log=lambda _: None)


def test_unverified_source_warns(tmp_path: Path) -> None:
    p = _w(tmp_path / "s.jsonl", [{"messages": conv("q", "a")}])
    logs: list[str] = []
    meta = build_sft_data(
        SFTDataConfig(out_dir=str(tmp_path / "o"), val_size=0, sources=[SFTSource("s", p)]),
        log=logs.append,
    )
    assert meta["n_train"] == 0 and any("WARNING" in x for x in logs)


def test_main_config_parses() -> None:
    cfg = load_sft_data_config("configs/main/sft_data.toml")
    assert cfg.out_dir == "data/sft" and cfg.sources
    assert cfg.allow_unverified is False  # unverified licenses stay out of the main line
    names = [s.name for s in cfg.sources]
    assert len(names) == len(set(names))
    assert all(
        s.format in ("messages", "sharegpt", "alpaca", "sft", "tool_env") for s in cfg.sources
    )
