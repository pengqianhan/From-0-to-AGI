"""Chat template (Chapter 16).

Rendering and parsing are inverse operations. The loss mask marks only assistant tokens. The Jinja
template and the Python code give the same text, character by character.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from zero.post.chat import (
    CHAT_TEMPLATE,
    assistant_text,
    encode_prompt_response,
    parse_assistant,
    render,
    render_text,
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": '查询 "城市" 天气',
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {"name": "calculator", "parameters": {"type": "object", "properties": {}}},
    },
]

CONVERSATIONS = {
    "tools_multi_call": [
        {"role": "system", "content": "你是助手。"},
        {"role": "user", "content": "北京和上海哪个热？"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": {"city": "北京"}},
                },
                {"name": "get_weather", "arguments": '{"city": "上海"}'},
            ],
        },
        {"role": "tool", "content": '{"temp_c": 25}'},
        {"role": "tool", "content": {"temp_c": 28}},
        {"role": "assistant", "content": "上海更热。"},
    ],
    "content_and_call": [
        {"role": "user", "content": "算一下 1+1"},
        {
            "role": "assistant",
            "content": "好的，我来算。",
            "tool_calls": [{"name": "calculator", "arguments": {"expression": "1+1"}}],
        },
        {"role": "tool", "content": "2"},
        {"role": "assistant", "content": "等于 2。"},
        {"role": "user", "content": "谢谢"},
    ],
    "plain": [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ],
}


def _hf_render(messages, tools, add_generation_prompt):  # noqa: ANN001, ANN202
    utils = pytest.importorskip("transformers.utils.chat_template_utils")
    tmpl = utils._compile_jinja_template(CHAT_TEMPLATE)
    kw = {"tools": tools} if tools else {}
    return tmpl.render(messages=messages, add_generation_prompt=add_generation_prompt, **kw)


@pytest.mark.parametrize("name", list(CONVERSATIONS))
@pytest.mark.parametrize("with_tools", [True, False])
@pytest.mark.parametrize("gen", [True, False])
def test_jinja_template_matches_python(name: str, with_tools: bool, gen: bool) -> None:
    msgs = CONVERSATIONS[name]
    tools = TOOLS if with_tools else None
    assert _hf_render(msgs, tools, gen) == render_text(msgs, tools, gen)


def test_rendered_format_is_qwen_style() -> None:
    text = render_text(CONVERSATIONS["tools_multi_call"], TOOLS)
    assert text.startswith("<|im_start|>system\n你是助手。\n\n# Tools\n")
    assert "<tools>\n" + json.dumps(TOOLS[0], ensure_ascii=False) + "\n" in text
    assert (
        '<tool_call>\n{"name": "get_weather", "arguments": {"city": "北京"}}\n</tool_call>\n<tool_call>'
        in text
    )
    # two consecutive tool messages go into the same user turn
    assert text.count("<|im_start|>user\n<tool_response>") == 1
    assert '<tool_response>\n{"temp_c": 28}\n</tool_response><|im_end|>' in text


@pytest.mark.parametrize("name", list(CONVERSATIONS))
def test_render_parse_roundtrip(name: str) -> None:
    for m in CONVERSATIONS[name]:
        if m["role"] != "assistant":
            continue
        parsed = parse_assistant(assistant_text(m))
        assert parsed.errors == []
        assert parsed.content == m["content"]
        want = [
            {
                "name": (tc.get("function") or tc)["name"],
                "arguments": (lambda a: json.loads(a) if isinstance(a, str) else a)(
                    (tc.get("function") or tc)["arguments"]
                ),
            }
            for tc in m.get("tool_calls", [])
        ]
        assert parsed.tool_calls == want
        # render the parsed result again: the text does not change (string arguments become an
        # object with the same JSON)
        assert assistant_text(parsed.to_message()) == assistant_text(
            {**m, "tool_calls": [{"name": w["name"], "arguments": w["arguments"]} for w in want]}
        )


def test_parse_think_and_malformed() -> None:
    p = parse_assistant("<think>\n想一想\n</think>\n\n答案<|im_end|>多余")
    assert p.reasoning_content == "想一想" and p.content == "答案" and not p.errors
    assert parse_assistant('<tool_call>{"name": "x", "arguments": {</tool_call>').errors
    assert parse_assistant('<tool_call>{"name": "x"').errors  # not closed
    assert parse_assistant('<tool_call>["x"]</tool_call>').errors  # not an object
    assert parse_assistant('<tool_call>{"arguments": {}}</tool_call>').errors  # no name
    assert parse_assistant('<tool_call>{"name": "x", "arguments": [1]}</tool_call>').errors
    ok = parse_assistant('<tool_call>{"name": "x", "arguments": {"a": 1}}</tool_call>')
    assert not ok.errors and ok.tool_calls == [{"name": "x", "arguments": {"a": 1}}]


def test_thinking_off_by_default() -> None:
    msgs = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "a", "reasoning_content": "r"},
    ]
    assert "<think>" not in render_text(msgs)
    assert "<think>\nr\n</think>\n\na<|im_end|>" in render_text(msgs, enable_thinking=True)
    utils = pytest.importorskip("transformers.utils.chat_template_utils")
    tmpl = utils._compile_jinja_template(CHAT_TEMPLATE)
    assert tmpl.render(messages=msgs, enable_thinking=True) == render_text(
        msgs, enable_thinking=True
    )


def test_loss_mask_covers_only_assistant_tokens(chat_tok) -> None:  # noqa: ANN001
    msgs = CONVERSATIONS["tools_multi_call"]
    ids, mask = render(msgs, TOOLS, tokenizer=chat_tok)
    assert len(ids) == len(mask)
    text = render_text(msgs, TOOLS)
    assert chat_tok.decode(ids) == text
    # the marked tokens together are exactly the two assistant outputs (each ends with <|im_end|>)
    masked = chat_tok.decode([i for i, m in zip(ids, mask) if m])
    a1 = assistant_text(msgs[2]) + "<|im_end|>"
    a2 = assistant_text(msgs[5]) + "<|im_end|>"
    assert masked == a1 + a2
    # the <tool_call> of the two tool calls are in the mask (the two in the format text of the system
    # prompt are not); <|im_start|> and the tool results are not in the mask
    tc = chat_tok.special_id("<tool_call>")
    tr = chat_tok.special_id("<tool_response>")
    assert sum(1 for i, m in zip(ids, mask) if i == tc and m) == 2
    assert sum(1 for i, m in zip(ids, mask) if i == tc and not m) == 2
    assert not any(m for i, m in zip(ids, mask) if i in (tr, chat_tok.im_start_id))
    n_end = sum(1 for i, m in zip(ids, mask) if m and i == chat_tok.im_end_id)
    assert n_end == 2


def test_encode_prompt_response_masks_only_response(chat_tok) -> None:  # noqa: ANN001
    msgs = CONVERSATIONS["content_and_call"][:4]  # the history already has two assistant messages
    resp = {"role": "assistant", "content": "不客气"}
    ids, mask = encode_prompt_response(msgs + [{"role": "user", "content": "谢谢"}], resp, chat_tok)
    assert chat_tok.decode([i for i, m in zip(ids, mask) if m]) == "不客气<|im_end|>"


def test_export_writes_chat_template(tmp_path: Path, chat_tok) -> None:  # noqa: ANN001
    transformers = pytest.importorskip("transformers")
    from zero.config import ModelConfig
    from zero.hf import export_to_hf_qwen3
    from zero.model import Transformer

    m = Transformer(
        ModelConfig(
            vocab_size=chat_tok.vocab_size,
            dim=32,
            n_layers=1,
            n_heads=2,
            n_kv_heads=1,
            ffn_dim=32,
            max_seq_len=256,
        )
    )
    out = export_to_hf_qwen3(m, None, tmp_path / "hf", tokenizer=chat_tok, chat=True)
    cfg = json.loads((out / "tokenizer_config.json").read_text())
    assert cfg["chat_template"] == CHAT_TEMPLATE and cfg["eos_token"] == "<|im_end|>"
    gen = json.loads((out / "generation_config.json").read_text())
    assert gen["eos_token_id"] == [chat_tok.im_end_id, chat_tok.eot_id]
    hf_tok = transformers.AutoTokenizer.from_pretrained(str(out))
    msgs = CONVERSATIONS["tools_multi_call"]
    s = hf_tok.apply_chat_template(msgs, tools=TOOLS, tokenize=False, add_generation_prompt=True)
    assert s == render_text(msgs, TOOLS, add_generation_prompt=True)
    ids = hf_tok.apply_chat_template(msgs, tools=TOOLS, tokenize=True, add_generation_prompt=True)
    if isinstance(ids, dict) or hasattr(ids, "input_ids"):
        ids = ids["input_ids"]
    assert list(ids) == render(msgs, TOOLS, add_generation_prompt=True, tokenizer=chat_tok)[0]
    assert torch.is_tensor(torch.tensor(ids))
