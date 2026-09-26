"""评测：配对 bootstrap 判定、对手取较高模式、选择题 / 精确匹配 / 工具调用评测的通路、报告表格、BFCL 适配层（第 11、20 章）。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from zero.eval.bootstrap import AHEAD, BEHIND, TIE, compare_to_opponent, decide, paired_bootstrap
from zero.eval.harness import (
    EvalConfig,
    EvalModel,
    continuation_logprob,
    eval_exact_match,
    eval_multiple_choice,
    eval_tool_calls,
    load_eval_config,
    run_eval,
)
from zero.eval.report import comparison_table, results_table


def test_decision_rule() -> None:
    assert decide(0.01, 0.2) == AHEAD == "超过"
    assert decide(-0.01, 0.2) == TIE == "持平"
    assert decide(-0.3, -0.02) == BEHIND == "落后"


def test_paired_bootstrap_cases() -> None:
    rng = np.random.default_rng(0)
    base = (rng.random(400) < 0.5).astype(float)
    better = base.copy()
    better[rng.choice(400, 60, replace=False)] = 1.0  # 明显更好
    r = paired_bootstrap(better, base, n_boot=2000, seed=1)
    assert (
        r.decision == "超过"
        and r.ci_low > 0
        and r.diff == pytest.approx(better.mean() - base.mean())
    )
    assert paired_bootstrap(base, better, n_boot=2000).decision == "落后"
    noisy = base.copy()  # 3 道由对变错、4 道由错变对：差值很小，区间跨过 0
    ones, zeros = np.flatnonzero(base == 1), np.flatnonzero(base == 0)
    noisy[ones[:3]] = 0
    noisy[zeros[:4]] = 1
    assert paired_bootstrap(noisy, base, n_boot=2000).decision == "持平"
    same = paired_bootstrap(base, base, n_boot=500)
    assert same.ci_low == same.ci_high == 0.0 and same.decision == "持平"
    # 可复现
    assert paired_bootstrap(better, base, n_boot=500, seed=3) == paired_bootstrap(
        better, base, n_boot=500, seed=3
    )
    with pytest.raises(ValueError):
        paired_bootstrap([1, 0], [1])


def test_opponent_higher_mode_is_used() -> None:
    ours = [1, 1, 1, 0] * 50
    mode, r = compare_to_opponent(
        ours, {"non_thinking": [0, 1, 0, 0] * 50, "thinking": [1, 1, 1, 0] * 50}, n_boot=500
    )
    assert mode == "thinking" and r.decision == "持平"


def test_continuation_logprob_and_mc(chat_tok, tiny_ckpt) -> None:  # noqa: ANN001
    from zero.post.common import load_policy

    model, tok = load_policy(tiny_ckpt)
    lp, n = continuation_logprob(model, tok, "Question: 1+1?\nAnswer:", " 2")
    ctx, cont = tok.encode("Question: 1+1?\nAnswer:"), tok.encode(" 2")
    ids = torch.tensor([ctx + cont])
    with torch.no_grad():
        logp = F.log_softmax(model(ids[:, :-1]).float(), -1)[0]
    hand = sum(float(logp[len(ctx) - 1 + i, cont[i]]) for i in range(len(cont)))
    assert n == len(cont) and lp == pytest.approx(hand, abs=1e-4)
    items = [{"question": f"q{i}", "choices": ["a", "bb", "c"], "answer": i % 3} for i in range(8)]
    r = eval_multiple_choice(model, tok, items, fewshot=2)
    assert r["n"] == 6 and len(r["items"]) == 6 and 0 <= r["acc"] <= 1
    g = eval_exact_match(
        model, tok, [{"prompt": "1 + 1 =", "answer": "2"}] * 3, fewshot=1, max_new_tokens=4
    )
    assert g["n"] == 2 and all(isinstance(it["output"], str) for it in g["items"])


def test_tool_eval_with_oracle() -> None:
    from zero.post.chat import format_tool_call
    from zero.post.envs.tool_env import dev_tasks

    tasks = dev_tasks(20)
    by_q = {t.query: t for t in tasks}

    def oracle(messages, tools):  # noqa: ANN001, ANN202
        t = by_q[next(m["content"] for m in messages if m["role"] == "user")]
        if messages[-1]["role"] == "user" and t.gold_calls:
            return "\n".join(format_tool_call(c) for c in t.gold_calls)
        return t.gold_answer

    r = eval_tool_calls(oracle, tasks)
    assert r["call_exact"] == 1.0 and r["answer_ok"] == 1.0 and r["success"] == 1.0 and r["n"] == 20
    r = eval_tool_calls(lambda m, t: "不知道", tasks)
    assert r["call_exact"] == pytest.approx(sum(t.kind == "no_tool" for t in tasks) / 20)


def test_shipped_task_files_are_valid() -> None:
    from zero.eval.harness import TASK_DIR
    from zero.post.envs.tool_env import Task, dev_tasks

    mc = [json.loads(x) for x in (TASK_DIR / "toy_mc.jsonl").read_text().splitlines()]
    assert len(mc) == 40 and all(0 <= it["answer"] < len(it["choices"]) for it in mc)
    tool = [
        Task.from_dict(json.loads(x))
        for x in (TASK_DIR / "tool_dev.jsonl").read_text().splitlines()
    ]
    assert [t.query for t in tool] == [t.query for t in dev_tasks(100)]  # 冻结文件与生成器一致


def test_run_eval_and_report(tmp_path: Path, tiny_ckpt) -> None:  # noqa: ANN001
    ec = EvalConfig(
        models=[EvalModel("a", str(tiny_ckpt)), EvalModel("b", str(tiny_ckpt))],
        mc_tasks=["toy_mc.jsonl"],
        gen_tasks=["toy_gen.jsonl"],
        tool_tasks="tool_dev.jsonl",
        max_items=3,
        max_new_tokens=8,
        max_turns=2,
        out_dir=str(tmp_path),
        baseline="a",
        n_boot=200,
    )
    p = run_eval(ec, log=lambda _: None)
    assert set(p["results"]) == {"a", "b"}
    assert {c["task"] for c in p["comparisons"]} == {"toy_mc", "toy_gen", "tool_dev"}
    assert all(c["decision"] == "持平" and c["diff"] == 0 for c in p["comparisons"])  # 同一个模型
    md = (tmp_path / "report.md").read_text()
    assert "| a |" in md and "95% CI" in md


def test_report_tables() -> None:
    res = {
        "m1": {"t": {"type": "mc", "acc": 0.5, "acc_norm": 0.25, "n": 4}},
        "m2": {"t": {"type": "mc", "acc": 0.75, "acc_norm": 0.5, "n": 4}},
    }
    tab = results_table(res)
    assert "| m2 | 0.750 | 0.500 |" in tab and "t n=4" in tab
    c = comparison_table(
        [
            {
                "model": "m2",
                "baseline": "m1",
                "task": "t",
                "metric": "acc",
                "mean_a": 0.75,
                "mean_b": 0.5,
                "diff": 0.25,
                "ci_low": -0.1,
                "ci_high": 0.5,
                "decision": "持平",
            }
        ]
    )
    assert "[-0.100, +0.500] | 持平" in c


def test_eval_config_file_parses() -> None:
    ec = load_eval_config("configs/tiny/eval.toml")
    assert ec.models[0].name == "sft" and ec.baseline == "sft" and ec.tool_tasks == "tool_dev.jsonl"


def test_bfcl_adapter_without_package(tmp_path: Path) -> None:
    from zero.eval import bfcl

    if not bfcl.bfcl_available():
        with pytest.raises(SystemExit):
            bfcl.run_cli("generate", str(tmp_path), "zero-FC", [])
    (tmp_path / "score").mkdir()
    (tmp_path / "score" / "data_overall.csv").write_text(
        "Rank,Model,Overall Acc\n1,zero-FC,12.5\n2,other,50\n"
    )
    got = bfcl.collect_scores(tmp_path / "score", "zero-FC")
    assert got == {"data_overall": {"Rank": "1", "Model": "zero-FC", "Overall Acc": "12.5"}}
