"""Evaluation: paired bootstrap decision, the higher mode of the opponent, the code paths of multiple choice / exact match / tool-call evaluation, report tables, BFCL adapter (Chapters 11 and 20)."""

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
    assert decide(0.01, 0.2) == AHEAD == "ahead"
    assert decide(-0.01, 0.2) == TIE == "tie"
    assert decide(-0.3, -0.02) == BEHIND == "behind"


def test_paired_bootstrap_cases() -> None:
    rng = np.random.default_rng(0)
    base = (rng.random(400) < 0.5).astype(float)
    better = base.copy()
    better[rng.choice(400, 60, replace=False)] = 1.0  # clearly better
    r = paired_bootstrap(better, base, n_boot=2000, seed=1)
    assert (
        r.decision == "ahead"
        and r.ci_low > 0
        and r.diff == pytest.approx(better.mean() - base.mean())
    )
    assert paired_bootstrap(base, better, n_boot=2000).decision == "behind"
    noisy = base.copy()  # 3 items change from correct to wrong, 4 from wrong to correct: small difference, the interval contains 0
    ones, zeros = np.flatnonzero(base == 1), np.flatnonzero(base == 0)
    noisy[ones[:3]] = 0
    noisy[zeros[:4]] = 1
    assert paired_bootstrap(noisy, base, n_boot=2000).decision == "tie"
    same = paired_bootstrap(base, base, n_boot=500)
    assert same.ci_low == same.ci_high == 0.0 and same.decision == "tie"
    # Reproducible
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
    assert mode == "thinking" and r.decision == "tie"


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
    assert [t.query for t in tool] == [t.query for t in dev_tasks(100)]  # the frozen file matches the generator


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
    assert all(c["decision"] == "tie" and c["diff"] == 0 for c in p["comparisons"])  # the same model
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
                "decision": "tie",
            }
        ]
    )
    assert "[-0.100, +0.500] | tie" in c


def test_eval_config_file_parses() -> None:
    ec = load_eval_config("configs/tiny/eval.toml")
    assert ec.models[0].name == "sft" and ec.baseline == "sft" and ec.tool_tasks == "tool_dev.jsonl"


def _bfcl_project(root: Path, model: str, cat: str, group: str, ids: list[str], failed: list[str]) -> None:
    """Write the files that bfcl-eval 2026.3.23 writes: every item in result, header + failures in score."""
    res = root / "result" / model.replace("/", "_") / group
    sc = root / "score" / model.replace("/", "_") / group
    res.mkdir(parents=True, exist_ok=True)
    sc.mkdir(parents=True, exist_ok=True)
    (res / f"BFCL_v4_{cat}_result.json").write_text("".join(json.dumps({"id": i, "result": "x"}) + "\n" for i in ids))
    header = {"accuracy": 1 - len(failed) / len(ids), "correct_count": len(ids) - len(failed), "total_count": len(ids)}
    rows = [header] + [{"id": i, "model_name": model, "valid": False, "error": ["wrong"]} for i in failed]
    (sc / f"BFCL_v4_{cat}_score.json").write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_bfcl_adapter_per_item_and_compare(tmp_path: Path) -> None:
    from zero.eval import bfcl

    if not bfcl.bfcl_available():
        with pytest.raises(SystemExit):
            bfcl.run_cli("generate", str(tmp_path), "zero-FC", [])
    ids = [f"simple_python_{i}" for i in range(20)]
    _bfcl_project(tmp_path, "zero-FC", "simple_python", "non_live", ids, ids[:5])
    _bfcl_project(tmp_path, "Qwen/Qwen3-0.6B-FC", "simple_python", "non_live", ids, ids[:10])
    lids = [f"live_simple_{i}-0-0" for i in range(10)]
    _bfcl_project(tmp_path, "zero-FC", "live_simple", "live", lids, [])
    _bfcl_project(tmp_path, "Qwen/Qwen3-0.6B-FC", "live_simple", "live", lids, lids[:1])

    items = bfcl.per_item_results(tmp_path, "zero-FC")
    assert items["simple_python"][ids[0]] == 0.0 and items["simple_python"][ids[9]] == 1.0
    assert bfcl.collect_scores(tmp_path, "zero-FC") == {
        "live_simple": {"accuracy": 1.0, "n": 10},
        "simple_python": {"accuracy": 0.75, "n": 20},
    }
    r = bfcl.compare_models(tmp_path, "zero-FC", "Qwen/Qwen3-0.6B-FC", n_boot=500)
    assert r["categories"] == ["live_simple", "simple_python"]
    assert r["diff"] == pytest.approx((0.1 + 0.25) / 2)  # equal weights: mean of the category differences
    # A score header that disagrees with the result file is an error, not a silent mismatch
    (tmp_path / "result" / "zero-FC" / "live" / "BFCL_v4_live_simple_result.json").write_text('{"id": "x"}\n')
    with pytest.raises(ValueError, match="total_count"):
        bfcl.per_item_results(tmp_path, "zero-FC")


def test_bfcl_prompt_types_become_json_schema() -> None:
    from zero.post.envs.fc_tasks import to_json_schema

    f = {"name": "calc", "parameters": {"type": "dict", "properties": {"x": {"type": "float"}, "pts": {"type": "tuple", "items": {"type": "integer"}}}}}
    out = to_json_schema(f)
    assert out["parameters"]["type"] == "object"
    assert out["parameters"]["properties"]["x"]["type"] == "number"
    assert out["parameters"]["properties"]["pts"]["type"] == "array"
    assert f["parameters"]["type"] == "dict"  # the input is not changed


def test_stratified_single_stratum_matches_paired():
    import numpy as np

    from zero.eval.bootstrap import paired_bootstrap, stratified_paired_bootstrap

    rng = np.random.default_rng(1)
    a = rng.integers(0, 2, 50).astype(float)
    b = rng.integers(0, 2, 50).astype(float)
    r1 = paired_bootstrap(a, b, n_boot=2000, seed=3, chunk=2000)
    r2 = stratified_paired_bootstrap({"s": a}, {"s": b}, {"s": 1.0}, n_boot=2000, seed=3)
    assert abs(r1.ci_low - r2.ci_low) < 1e-12 and abs(r1.ci_high - r2.ci_high) < 1e-12
    assert r1.decision == r2.decision


def test_stratified_weighted_means():
    from zero.eval.bootstrap import stratified_paired_bootstrap

    a = {"x": [1.0, 1.0], "y": [0.0, 0.0]}
    b = {"x": [0.0, 0.0], "y": [0.0, 0.0]}
    r = stratified_paired_bootstrap(a, b, {"x": 1.0, "y": 3.0}, n_boot=200, seed=0)
    assert abs(r.mean_a - 0.25) < 1e-12 and abs(r.diff - 0.25) < 1e-12


def test_overall_verdict_requires_every_comparison():
    from zero.eval.bootstrap import overall_verdict

    assert overall_verdict({("q", "e1"): "ahead", ("q", "e2"): "ahead"}) == "ahead"
    assert overall_verdict({("q", "e1"): "ahead", ("q", "e2"): "tie"}) == "tie"
    assert overall_verdict({("q", "e1"): "ahead", ("m", "e1"): "behind"}) == "behind"


def test_export_prompts_with_fake_loader(tmp_path: Path) -> None:
    from zero.eval.export_prompts import SPECS, PromptSpec, export_spec

    data = {
        ("x/cmmlu", "anatomy", "test"): [{"Question": "人体最大的器官是什么？"}, {"Question": "人体最大的器官是什么？"}],
        ("x/cmmlu", "law", "test"): [{"Question": "法律问题"}, {"Question": ""}],
    }

    def loader(hf_id, config, split):  # noqa: ANN001, ANN202
        if (hf_id, config, split) not in data:
            raise ValueError("no such split")
        return data[(hf_id, config, split)]

    spec = PromptSpec("cmmlu", "x/cmmlu", ("test", "dev"), ("Question", "question"))
    r = export_spec(spec, tmp_path, loader, lambda _: ["anatomy", "law"])
    assert r["n"] == 2 and r["missing"] == ["anatomy/dev", "law/dev"]  # duplicates and empty questions skipped
    rows = [json.loads(x) for x in (tmp_path / "cmmlu.jsonl").read_text().splitlines()]
    assert rows[0] == {"question": "人体最大的器官是什么？", "config": "anatomy", "split": "test"}
    # The output feeds the SFT decontamination (zero.post.sft_data reads "question")
    from zero.post.sft_data import _eval_texts

    assert _eval_texts([str(tmp_path / "*.jsonl")]) == ["人体最大的器官是什么？", "法律问题"]
    assert len({s.name for s in SPECS}) == len(SPECS)
