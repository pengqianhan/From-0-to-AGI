"""评测框架：少样本对数似然选择题、生成式精确匹配、工具调用评测（对应第 11、20 章）。

    uv run python -m zero.eval.harness --config configs/tiny/eval.toml

三类评测（每一类都返回**逐题得分**，供 `bootstrap.py` 做配对 bootstrap）：

1. **选择题（对数似然）**：提示词 = k 个示例 + 问题，对每个选项算"续写成这个选项"的 log 概率之和，
   取最大的为答案。与 lm-evaluation-harness 的做法一致：
   - `acc`：直接比 log 概率之和；
   - `acc_norm`：除以选项的 UTF-8 字节数再比（长选项天然概率低，归一化后更公平）；
   - 上下文和续写分开编码再拼接（续写前面的空格归续写），与 lm-eval 的默认做法一致。
   Base 模型常用这种评测：不需要模型会"按格式作答"。
2. **生成式精确匹配**：k 个示例 + 提示词，贪心生成到换行为止，去掉首尾空白后与答案逐字比较。
3. **工具调用**：用对话模板把 tool_env 的 dev 任务喂给模型，走完"生成 → 解析 → 执行 → 喂回 →
   最终回答"（`run_episode`），记录第一轮调用的奖励、AST 匹配、执行匹配、最终回答是否正确。

这里的数据都是仓库自带的玩具集（`zero/eval/tasks/`），只用来验证代码。正式基准按
eval/PREREGISTRATION.md 用官方评测框架跑（lm-evaluation-harness、BFCL 等，见 `bfcl.py`）。
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch

from zero.model import Transformer
from zero.post.common import load_policy, make_policy, read_jsonl, token_logprobs
from zero.tokenizer import Tokenizer

TASK_DIR = Path(__file__).resolve().parent / "tasks"


@dataclass
class EvalModel:
    name: str = ""
    path: str = ""  # zero checkpoint 目录或 HF 目录


@dataclass
class EvalConfig:
    models: list[EvalModel] = field(default_factory=list)
    mc_tasks: list[str] = field(default_factory=list)  # 选择题 JSONL
    gen_tasks: list[str] = field(default_factory=list)  # 生成式 JSONL
    tool_tasks: str = ""  # 工具调用 dev 集 JSONL（tool_env.Task 的字典）
    fewshot: int = 3
    max_items: int = 0  # 每个任务最多评多少题；0 = 全部
    max_new_tokens: int = 96
    max_turns: int = 3
    out_dir: str = "out/eval"
    baseline: str = ""  # 其余模型都与它做配对 bootstrap 比较
    n_boot: int = 2000
    seed: int = 0
    cpu_threads: int = 0


# ---------------------------------------------------------------------------
# 选择题
# ---------------------------------------------------------------------------


def _mc_prompt(item: dict[str, Any]) -> str:
    return f"Question: {item['question']}\nAnswer:"


@torch.no_grad()
def continuation_logprob(
    model: Transformer, tok: Tokenizer, context: str, continuation: str
) -> tuple[float, int]:
    """log P(continuation | context) 以及续写的 token 数。上下文过长时从左边截断。"""
    ctx = tok.encode(context)
    cont = tok.encode(continuation)
    ids = (ctx + cont)[-(model.config.max_seq_len + 1) :]
    n = len(cont)
    x = torch.tensor([ids[:-1]], device=next(model.parameters()).device)
    y = torch.tensor([ids[1:]], device=x.device)
    lp = token_logprobs(model(x), y)[0, -n:]
    return float(lp.sum()), n


def eval_multiple_choice(
    model: Transformer,
    tok: Tokenizer,
    items: Sequence[dict[str, Any]],
    fewshot: int = 3,
    max_items: int = 0,
) -> dict[str, Any]:
    """前 fewshot 题做示例，其余题评测。返回 {"acc", "acc_norm", "n", "items": [...逐题...]}。"""
    shots = list(items[:fewshot])
    rest = list(items[fewshot:])
    if max_items > 0:
        rest = rest[:max_items]
    prefix = "".join(f"{_mc_prompt(s)} {s['choices'][s['answer']]}\n\n" for s in shots)
    model.eval()
    per = []
    for it in rest:
        ctx = prefix + _mc_prompt(it)
        scores, norms = [], []
        for ch in it["choices"]:
            lp, _ = continuation_logprob(model, tok, ctx, " " + ch)
            scores.append(lp)
            norms.append(lp / max(len((" " + ch).encode("utf-8")), 1))
        pred = max(range(len(scores)), key=lambda i: scores[i])
        pred_n = max(range(len(norms)), key=lambda i: norms[i])
        per.append(
            {
                "id": it.get("id"),
                "pred": pred,
                "correct": float(pred == it["answer"]),
                "correct_norm": float(pred_n == it["answer"]),
            }
        )
    n = max(len(per), 1)
    return {
        "acc": sum(p["correct"] for p in per) / n,
        "acc_norm": sum(p["correct_norm"] for p in per) / n,
        "n": len(per),
        "items": per,
    }


# ---------------------------------------------------------------------------
# 生成式精确匹配
# ---------------------------------------------------------------------------


@torch.no_grad()
def greedy_until(
    model: Transformer, tok: Tokenizer, prompt: str, max_new_tokens: int = 16, stop: str = "\n"
) -> str:
    from zero.generate import generate_stream

    ids = tok.encode(prompt)[-(model.config.max_seq_len - max_new_tokens) :]
    out: list[int] = []
    for t in generate_stream(model, ids, max_new_tokens, temperature=0.0, eos_id=tok.eot_id):
        tid = int(t[0])
        if tid == tok.eot_id:
            break
        out.append(tid)
        if stop in tok.decode(out):
            break
    text = tok.decode(out)
    return text.split(stop)[0] if stop else text


def eval_exact_match(
    model: Transformer,
    tok: Tokenizer,
    items: Sequence[dict[str, Any]],
    fewshot: int = 3,
    max_items: int = 0,
    max_new_tokens: int = 16,
) -> dict[str, Any]:
    shots = list(items[:fewshot])
    rest = list(items[fewshot:])
    if max_items > 0:
        rest = rest[:max_items]
    prefix = "".join(f"{s['prompt']} {s['answer']}\n" for s in shots)
    per = []
    for it in rest:
        out = greedy_until(model, tok, prefix + it["prompt"], max_new_tokens).strip()
        answers = it["answer"] if isinstance(it["answer"], list) else [it["answer"]]
        per.append(
            {
                "id": it.get("id"),
                "output": out,
                "correct": float(out in [a.strip() for a in answers]),
            }
        )
    n = max(len(per), 1)
    return {"em": sum(p["correct"] for p in per) / n, "n": len(per), "items": per}


# ---------------------------------------------------------------------------
# 工具调用
# ---------------------------------------------------------------------------


def eval_tool_calls(
    policy: Callable[[list[dict[str, Any]], list[dict[str, Any]]], str],
    tasks: Sequence[Any],
    max_turns: int = 3,
) -> dict[str, Any]:
    """policy(messages, tools) -> 助手文本。对每个任务跑完整一轮，返回汇总与逐题结果。"""
    from zero.post.envs.tool_env import run_episode

    per = []
    for t in tasks:
        ep = run_episode(policy, t, max_turns)
        cr = ep.call_reward
        per.append(
            {
                "id": t.id,
                "kind": t.kind,
                "call_reward": cr.total,
                "format_ok": float(cr.format_ok),
                "call_exact": float(
                    cr.total >= 0.999
                ),  # 该调的全调对、没有多余调用（不该调的没调）
                "ast_match": cr.ast_match,
                "exec_match": cr.exec_match,
                "answer_ok": float(bool(ep.answer_reward.answer_ok)),
                "success": float(ep.success),
                "turns": ep.turns,
            }
        )
    n = max(len(per), 1)
    keys = [
        "call_reward",
        "format_ok",
        "call_exact",
        "ast_match",
        "exec_match",
        "answer_ok",
        "success",
    ]
    return {**{k: sum(p[k] for p in per) / n for k in keys}, "n": len(per), "items": per}


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------


def load_eval_config(path: str | Path, overrides: Sequence[str] | None = None) -> EvalConfig:
    from zero.config import _apply_override, _from_dict, _read_toml_with_base

    data = _read_toml_with_base(Path(path))
    for o in overrides or []:
        _apply_override(data, o)
    return _from_dict(EvalConfig, data.get("eval", {}), "[eval]")


def _resolve(p: str) -> Path:
    q = Path(p)
    return q if q.exists() else TASK_DIR / p


def evaluate_model(
    model: Transformer, tok: Tokenizer, ec: EvalConfig, log: Callable[[str], None] = print
) -> dict[str, Any]:
    from zero.post.envs.tool_env import Task

    res: dict[str, Any] = {}
    for p in ec.mc_tasks:
        name = Path(p).stem
        t0 = time.time()
        res[name] = {
            "type": "mc",
            **eval_multiple_choice(model, tok, read_jsonl(_resolve(p)), ec.fewshot, ec.max_items),
        }
        log(
            f"  {name}: acc {res[name]['acc']:.3f} acc_norm {res[name]['acc_norm']:.3f}（{time.time() - t0:.1f}s）"
        )
    for p in ec.gen_tasks:
        name = Path(p).stem
        t0 = time.time()
        res[name] = {
            "type": "gen",
            **eval_exact_match(model, tok, read_jsonl(_resolve(p)), ec.fewshot, ec.max_items),
        }
        log(f"  {name}: em {res[name]['em']:.3f}（{time.time() - t0:.1f}s）")
    if ec.tool_tasks:
        t0 = time.time()
        tasks = [Task.from_dict(r) for r in read_jsonl(_resolve(ec.tool_tasks))]
        if ec.max_items > 0:
            tasks = tasks[: ec.max_items]
        policy = make_policy(model, tok, ec.max_new_tokens, temperature=0.0)
        r = eval_tool_calls(policy, tasks, ec.max_turns)
        res["tool_dev"] = {"type": "tool", **r}
        log(
            f"  tool_dev: call_reward {r['call_reward']:+.3f} call_exact {r['call_exact']:.3f} "
            f"format {r['format_ok']:.3f} answer {r['answer_ok']:.3f}（{time.time() - t0:.1f}s）"
        )
    return res


PRIMARY_METRIC = {"mc": "acc", "gen": "em", "tool": "call_exact"}
ITEM_KEY = {"acc": "correct", "acc_norm": "correct_norm", "em": "correct"}


def item_scores(task_result: dict[str, Any], metric: str) -> list[float]:
    key = ITEM_KEY.get(metric, metric)
    return [float(it[key]) for it in task_result["items"]]


def run_eval(ec: EvalConfig, log: Callable[[str], None] = print) -> dict[str, Any]:
    from zero.eval.bootstrap import paired_bootstrap
    from zero.eval.report import write_report

    if ec.cpu_threads > 0:
        torch.set_num_threads(ec.cpu_threads)
    results: dict[str, dict[str, Any]] = {}
    for m in ec.models:
        log(f"[eval] {m.name} ← {m.path}")
        model, tok = load_policy(m.path)
        model.eval()
        results[m.name] = evaluate_model(model, tok, ec, log)
    comparisons = []
    if ec.baseline:
        if ec.baseline not in results:
            raise ValueError(f"[eval] baseline={ec.baseline!r} 不在 models 里")
        for name in results:
            if name == ec.baseline:
                continue
            for task, r in results[name].items():
                metric = PRIMARY_METRIC[r["type"]]
                a = item_scores(r, metric)
                b = item_scores(results[ec.baseline][task], metric)
                br = paired_bootstrap(a, b, n_boot=ec.n_boot, seed=ec.seed)
                comparisons.append(
                    {
                        "model": name,
                        "baseline": ec.baseline,
                        "task": task,
                        "metric": metric,
                        **asdict(br),
                    }
                )
    out = Path(ec.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = {"config": asdict(ec), "results": results, "comparisons": comparisons}
    (out / "results.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    write_report(out / "report.md", results, comparisons, title="评测结果（玩具集，仅验证代码）")
    log(f"[eval] 结果：{out / 'results.json'}，报告：{out / 'report.md'}")
    return payload


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="评测（第 11、20 章）")
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = ap.parse_args(argv)
    run_eval(load_eval_config(args.config, args.set))


if __name__ == "__main__":
    main()
