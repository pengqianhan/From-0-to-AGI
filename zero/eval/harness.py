"""Evaluation framework: few-shot log-likelihood multiple choice, generative exact match, tool-call evaluation (Chapters 11 and 20).

    uv run python -m zero.eval.harness --config configs/tiny/eval.toml

There are three types of evaluation. Each type returns **per-item scores** for the paired bootstrap
in `bootstrap.py`:

1. **Multiple choice (log-likelihood)**: prompt = k examples + the question. For each choice,
   compute the sum of the log probabilities of "the continuation is this choice". The choice with
   the highest value is the answer. This is the method of lm-evaluation-harness:
   - `acc`: compare the sums of the log probabilities directly.
   - `acc_norm`: divide by the number of UTF-8 bytes of the choice, then compare. A long choice
     always has a lower probability, so the normalized comparison is fairer.
   - Encode the context and the continuation separately, then join them (the space before the
     continuation belongs to the continuation). This is the default of lm-eval.
   Base models often use this evaluation, because the model does not need to "answer in a format".
2. **Generative exact match**: k examples + the prompt. Generate greedily until a line break.
   Remove the white space at the start and the end, then compare with the answer character by character.
3. **Tool calls**: the chat template gives the dev tasks of tool_env to the model. The full
   sequence "generate → parse → execute → feed back → final answer" runs (`run_episode`).
   Record the reward of the first-turn call, the AST match, the execution match, and if the final
   answer is correct.

All data here are toy sets in the repository (`zero/eval/tasks/`), only to verify the code. The real
benchmarks run with the official evaluation frameworks, as eval/PREREGISTRATION.md specifies
(lm-evaluation-harness, BFCL, and others; see `bfcl.py`).
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
    path: str = ""  # zero checkpoint directory or HF directory


@dataclass
class EvalConfig:
    models: list[EvalModel] = field(default_factory=list)
    mc_tasks: list[str] = field(default_factory=list)  # multiple-choice JSONL files
    gen_tasks: list[str] = field(default_factory=list)  # generative JSONL files
    tool_tasks: str = ""  # tool-call dev set JSONL (dicts of tool_env.Task)
    fc_tasks: list[str] = field(default_factory=list)  # function-calling dev sets (zero/post/envs/fc_tasks.py format); first turn only
    fewshot: int = 3
    max_items: int = 0  # maximum number of items for each task; 0 = all
    max_new_tokens: int = 96
    max_turns: int = 3
    out_dir: str = "out/eval"
    baseline: str = ""  # the paired bootstrap compares all other models with this model
    n_boot: int = 2000
    seed: int = 0
    cpu_threads: int = 0


# ---------------------------------------------------------------------------
# Multiple choice
# ---------------------------------------------------------------------------


def _mc_prompt(item: dict[str, Any]) -> str:
    return f"Question: {item['question']}\nAnswer:"


@torch.no_grad()
def continuation_logprob(
    model: Transformer, tok: Tokenizer, context: str, continuation: str
) -> tuple[float, int]:
    """log P(continuation | context) and the number of tokens of the continuation. A context that is too long is cut from the left."""
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
    """The first `fewshot` items are the examples; the other items are evaluated.

    Returns {"acc", "acc_norm", "n", "items": [...one entry per item...]}.
    """
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
# Generative exact match
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
# Tool calls
# ---------------------------------------------------------------------------


def eval_tool_calls(
    policy: Callable[[list[dict[str, Any]], list[dict[str, Any]]], str],
    tasks: Sequence[Any],
    max_turns: int = 3,
) -> dict[str, Any]:
    """policy(messages, tools) -> assistant text. Run one full episode for each task, and return the summary and the per-item results."""
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
                "call_exact": float(cr.exact),  # all necessary calls are correct and there are no extra calls (no call when none is necessary, no invented call)
                "ast_match": cr.ast_match,
                "exec_match": cr.exec_match,
                "answer_ok": float(
                    ep.answer_reward.answer_ok is not False
                ),  # a chat task that cannot be checked does not count as wrong
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


def eval_fc_tasks(
    policy: Callable[[list[dict[str, Any]], list[dict[str, Any]]], str],
    tasks: Sequence[Any],
) -> dict[str, Any]:
    """Real function-calling tasks: score the first assistant turn with `fc_tasks.score_fc`."""
    from zero.post.envs.fc_tasks import score_fc

    per = []
    for t in tasks:
        r = score_fc(t, policy(t.messages, t.tools))
        per.append(
            {
                "id": t.id,
                "relevance": "irrelevance" if not t.gold_calls else "call",
                "call_reward": r.total,
                "format_ok": float(r.format_ok),
                "call_exact": float(r.exact),
                "ast_match": r.ast_match,
            }
        )
    n = max(len(per), 1)
    keys = ["call_reward", "format_ok", "call_exact", "ast_match"]
    return {**{k: sum(p[k] for p in per) / n for k in keys}, "n": len(per), "items": per}


# ---------------------------------------------------------------------------
# Entry point
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
            f"  {name}: acc {res[name]['acc']:.3f} acc_norm {res[name]['acc_norm']:.3f} ({time.time() - t0:.1f}s)"
        )
    for p in ec.gen_tasks:
        name = Path(p).stem
        t0 = time.time()
        res[name] = {
            "type": "gen",
            **eval_exact_match(model, tok, read_jsonl(_resolve(p)), ec.fewshot, ec.max_items),
        }
        log(f"  {name}: em {res[name]['em']:.3f} ({time.time() - t0:.1f}s)")
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
            f"format {r['format_ok']:.3f} answer {r['answer_ok']:.3f} ({time.time() - t0:.1f}s)"
        )
    for p in ec.fc_tasks:
        from zero.post.envs.fc_tasks import load_fc_tasks

        name = Path(p).stem
        t0 = time.time()
        tasks = load_fc_tasks(_resolve(p))
        if ec.max_items > 0:
            tasks = tasks[: ec.max_items]
        policy = make_policy(model, tok, ec.max_new_tokens, temperature=0.0)
        r = eval_fc_tasks(policy, tasks)
        res[name] = {"type": "fc", **r}
        log(
            f"  {name}: call_exact {r['call_exact']:.3f} call_reward {r['call_reward']:+.3f} "
            f"format {r['format_ok']:.3f} (n={r['n']}, {time.time() - t0:.1f}s)"
        )
    return res


PRIMARY_METRIC = {"mc": "acc", "gen": "em", "tool": "call_exact", "fc": "call_exact"}
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
            raise ValueError(f"[eval] baseline={ec.baseline!r} is not in models")
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
    write_report(out / "report.md", results, comparisons, title="Evaluation results (toy sets, only to verify the code)")
    log(f"[eval] results: {out / 'results.json'}, report: {out / 'report.md'}")
    return payload


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Evaluation (Chapters 11 and 20)")
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = ap.parse_args(argv)
    run_eval(load_eval_config(args.config, args.set))


if __name__ == "__main__":
    main()
