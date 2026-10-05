"""蒸馏：KL 损失手算、top-k、分词器检查、教师后端（本地 / OpenAI 兼容假服务器）、执行验证与元数据（第 17 章）。"""

from __future__ import annotations

import json
import math
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest
import torch

from tests.conftest import post_config
from zero.post.chat import format_tool_call
from zero.post.distill import (
    DistillConfig,
    OpenAITeacher,
    TeacherConfig,
    check_license,
    generate_kd_data,
    kd_loss,
    openai_message_to_text,
    reverse_kl_loss,
    run_distill,
    to_openai_messages,
)
from zero.post.envs.tool_env import generate_tasks


def test_kd_loss_hand_computed() -> None:
    s = torch.tensor([[[1.0, 0.0, -1.0]]])
    t = torch.tensor([[[0.0, 1.0, 0.0]]])
    mask = torch.tensor([[True]])
    ps = torch.softmax(s[0, 0], -1)
    pt = torch.softmax(t[0, 0], -1)
    hand = float((pt * (pt.log() - ps.log())).sum())
    assert kd_loss(s, t, mask).item() == pytest.approx(hand, abs=1e-6)
    # 温度 τ：两边除以 τ 再 softmax，结果乘 τ²
    ps2, pt2 = torch.softmax(s[0, 0] / 2, -1), torch.softmax(t[0, 0] / 2, -1)
    hand2 = 4 * float((pt2 * (pt2.log() - ps2.log())).sum())
    assert kd_loss(s, t, mask, temperature=2.0).item() == pytest.approx(hand2, abs=1e-6)
    # top-1：教师分布变成 one-hot（第 1 个 token），损失 = −log p_S(1)
    assert kd_loss(s, t, mask, topk=1).item() == pytest.approx(-math.log(float(ps[1])), abs=1e-6)
    # 反向 KL
    hand_r = float((ps * (ps.log() - pt.log())).sum())
    assert reverse_kl_loss(s, t, mask).item() == pytest.approx(hand_r, abs=1e-6)


def test_kd_loss_mask_zero_and_shape_check() -> None:
    x = torch.randn(2, 4, 7)
    mask = torch.tensor([[1, 1, 0, 0], [1, 0, 0, 0]], dtype=torch.bool)
    assert kd_loss(x, x, mask).item() == pytest.approx(0.0, abs=1e-6)
    y = x.clone()
    y[0, 3] += 5.0  # mask 之外的位置不影响损失
    assert kd_loss(x, y, mask).item() == pytest.approx(0.0, abs=1e-6)
    with pytest.raises(ValueError, match="同一个分词器"):
        kd_loss(torch.randn(1, 2, 7), torch.randn(1, 2, 9), torch.ones(1, 2, dtype=torch.bool))


def test_license_guard() -> None:
    check_license(TeacherConfig(backend="local", path="x"), is_self=True)
    with pytest.raises(PermissionError):
        check_license(TeacherConfig(backend="openai", model="big"), is_self=False)
    check_license(
        TeacherConfig(
            backend="openai", model="big", license="Apache-2.0", license_allows_distillation=True
        ),
        is_self=False,
    )


def test_openai_message_conversion() -> None:
    msgs = [
        {"role": "user", "content": "q"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"name": "calculator", "arguments": {"expression": "1+1"}}],
        },
        {"role": "tool", "content": "2"},
    ]
    o = to_openai_messages(msgs)
    assert o[1]["tool_calls"][0]["function"]["arguments"] == '{"expression": "1+1"}'
    assert o[2]["tool_call_id"] == o[1]["tool_calls"][0]["id"]
    text = openai_message_to_text(
        {
            "content": None,
            "tool_calls": [
                {
                    "id": "c",
                    "type": "function",
                    "function": {"name": "calculator", "arguments": '{"expression": "1+1"}'},
                }
            ],
        }
    )
    assert text == format_tool_call({"name": "calculator", "arguments": {"expression": "1+1"}})


class _FakeOpenAI(BaseHTTPRequestHandler):
    """假的 OpenAI 兼容服务：对任何任务都返回"正确"的调用 / 回答（从请求里的用户问题查标准答案）。"""

    tasks: dict = {}
    requests: list = []

    def do_POST(self) -> None:  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _FakeOpenAI.requests.append(body)
        q = next(m["content"] for m in body["messages"] if m["role"] == "user")
        task = _FakeOpenAI.tasks[q]
        if body["messages"][-1]["role"] == "tool" or not task.gold_calls:
            msg = {"role": "assistant", "content": task.gold_answer}
        else:
            msg = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": f"c{i}",
                        "type": "function",
                        "function": {
                            "name": c["name"],
                            "arguments": json.dumps(c["arguments"], ensure_ascii=False),
                        },
                    }
                    for i, c in enumerate(task.gold_calls)
                ],
            }
        # 第二个样本故意给错
        bad = {"role": "assistant", "content": '<tool_call>{"name": "calculator"'}
        data = {"choices": [{"message": msg}] + [{"message": bad}] * (body["n"] - 1)}
        out = json.dumps(data, ensure_ascii=False).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *a) -> None:  # noqa: ANN002
        pass


def test_openai_teacher_with_fake_server(tmp_path: Path) -> None:
    tasks = generate_tasks(6, seed=9)
    _FakeOpenAI.tasks = {t.query: t for t in tasks}
    srv = HTTPServer(("127.0.0.1", 0), _FakeOpenAI)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    try:
        tc = TeacherConfig(
            backend="openai",
            base_url=f"http://127.0.0.1:{srv.server_port}/v1",
            model="fake-teacher",
            name="FakeTeacher",
            version="v0",
            license="Apache-2.0",
            license_allows_distillation=True,
        )
        dc = DistillConfig(
            out_jsonl=str(tmp_path / "kd.jsonl"), n_tasks=6, samples_per_task=2, env_seed=9
        )
        meta = generate_kd_data(OpenAITeacher(tc), tc, dc, log=lambda _: None)
    finally:
        srv.shutdown()
    assert (
        meta["n_candidates"] == 12 and meta["n_verified"] == 5
    )  # 每个任务一对一错，错的被执行验证筛掉；闲聊任务的回答内容无法核对，也不收进蒸馏数据
    assert meta["teacher"] == {
        "name": "FakeTeacher",
        "version": "v0",
        "license": "Apache-2.0",
        "license_allows_distillation": True,
        "backend": "openai",
        "path_or_model": "fake-teacher",
    }
    rows = [json.loads(x) for x in (tmp_path / "kd.jsonl").read_text().splitlines()]
    assert all(r["teacher"]["name"] == "FakeTeacher" for r in rows)
    assert json.loads((tmp_path / "kd.jsonl.meta.json").read_text())["pass_rate"] == pytest.approx(
        5 / 12
    )
    assert _FakeOpenAI.requests[0]["tools"] and _FakeOpenAI.requests[0]["n"] == 2


def test_run_distill_local_self_teacher(tmp_path: Path, chat_tok, chat_tok_path, tiny_ckpt) -> None:  # noqa: ANN001
    from zero.post.common import write_jsonl
    from zero.post.sft import env_conversations

    write_jsonl(tmp_path / "sft.jsonl", env_conversations(8, 0, "train"))
    d = post_config(
        tmp_path,
        chat_tok_path,
        tiny_ckpt,
        chat_tok.vocab_size,
        data={"format": "sft", "seq_len": 512},
        teacher={"backend": "local", "path": str(tiny_ckpt), "name": "self", "max_new_tokens": 12},
        distill={
            "out_jsonl": str(tmp_path / "kd.jsonl"),
            "n_tasks": 3,
            "samples_per_task": 2,
            "mix_sft_jsonl": str(tmp_path / "sft.jsonl"),
            "mix_sft_max": 8,
            "kd_alpha": 0.5,
            "on_policy_steps": 1,
            "on_policy_batch": 2,
        },
    )
    d["model"]["max_seq_len"] = 512
    s = run_distill(d, log=lambda _: None)
    h = s["history"][-1]
    assert h["step"] == 2 and "kd" in h and "ce" in h
    # 学生与教师初始相同：第一步 KL 为 0
    assert s["history"][0]["kd"] == pytest.approx(0.0, abs=1e-5)
    assert len(s["on_policy"]) == 1
    from zero.train.checkpoint import find_latest

    assert find_latest(tmp_path / "run" / "ckpt").name == "step_00000003"


def test_logits_kd_requires_same_tokenizer(
    tmp_path: Path, chat_tok, chat_tok_path, tiny_ckpt, tiny_texts
) -> None:  # noqa: ANN001
    from zero.post.common import write_jsonl
    from zero.post.sft import env_conversations
    from zero.tokenizer import train_bpe

    other = train_bpe([tiny_texts["en"][:5000]], vocab_size=chat_tok.vocab_size)
    other.save(tmp_path / "other.json")
    write_jsonl(tmp_path / "kd.jsonl", env_conversations(3, 0, "train"))
    (tmp_path / "kd.jsonl.meta.json").write_text(json.dumps({"n_verified": 3}))
    d = post_config(
        tmp_path,
        tmp_path / "other.json",
        tiny_ckpt,
        chat_tok.vocab_size,
        data={"format": "sft", "seq_len": 512},
        teacher={"backend": "local", "path": str(tiny_ckpt)},
        distill={"out_jsonl": str(tmp_path / "kd.jsonl")},
    )
    d["model"]["max_seq_len"] = 512
    with pytest.raises(ValueError, match="同一个分词器"):
        run_distill(d, log=lambda _: None)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 CUDA")
def test_run_distill_trains_student_on_cuda(
    tmp_path: Path, chat_tok, chat_tok_path, tiny_ckpt
) -> None:  # noqa: ANN001
    """[train] device = "cuda" 时学生必须在 GPU 上训练（原来固定 DistInfo()，学生和教师都留在 CPU；2026-10 发现）。"""
    from zero.post.common import write_jsonl
    from zero.post.sft import env_conversations

    write_jsonl(tmp_path / "sft.jsonl", env_conversations(8, 0, "train"))
    d = post_config(
        tmp_path,
        chat_tok_path,
        tiny_ckpt,
        chat_tok.vocab_size,
        train={"device": "cuda", "dtype": "auto"},
        data={"format": "sft", "seq_len": 512},
        teacher={"backend": "local", "path": str(tiny_ckpt), "name": "self", "max_new_tokens": 12},
        distill={
            "out_jsonl": str(tmp_path / "kd.jsonl"),
            "n_tasks": 2,
            "samples_per_task": 2,
            "mix_sft_jsonl": str(tmp_path / "sft.jsonl"),
            "mix_sft_max": 8,
            "kd_alpha": 0.5,
        },
    )
    d["model"]["max_seq_len"] = 512
    logs: list[str] = []
    s = run_distill(d, log=logs.append)
    assert s["history"][-1]["step"] == 2
    assert any("设备 cuda" in m for m in logs), logs
