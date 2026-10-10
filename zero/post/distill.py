"""Distillation: teacher data generation + execution check + sequence-level / logits distillation.

Chapter 17 uses this module.

    uv run python -m zero.post.distill --config configs/tiny/distill.toml

**Two kinds of distillation** (both are "consensus" in GOAL.md 2.1):

1. **Sequence-level distillation** (sequence-level KD): the teacher model solves the tasks. Only the
   trajectories that **pass the execution check** stay, and they train the student as SFT data.
   The teacher only needs to "generate text". It can be any model with any tokenizer, or a remote
   HTTP service.
2. **Logits distillation**: at each position of the same text, the next-token distribution of the
   student moves toward the distribution of the teacher:
       KL(p_T ‖ p_S) = Σ_v p_T(v) · (log p_T(v) − log p_S(v))      (forward KL: "cover all that the teacher can say")
   Both logits are divided by the temperature τ before the softmax, and the loss is multiplied by τ²
   (Hinton et al. 2015; this keeps the gradient magnitude).
   With `kd_topk > 0`, only the k tokens with the highest teacher probability count. The teacher
   distribution is normalized again over the k tokens; the student distribution is not. This saves
   memory and storage. Step 2 uses it when it stores the teacher logits offline.
   **Logits distillation needs the same tokenizer for the teacher and the student** (the same
   vocabulary and the same splitting). A position-by-position comparison needs two distributions on
   the same set of tokens. This module compares the hashes of the two tokenizers and raises an error
   if they are different. A teacher with a different tokenizer can do only sequence-level distillation.

Final loss = (1 − α) · cross-entropy (the SFT loss of the student on the teacher trajectories) + α · KL,
with α = `kd_alpha`.

**Teacher backends** (`[teacher] backend`):

- `"local"`: a local model, a zero checkpoint or an exported HF folder (Qwen3 dense architecture, for
  example the open teacher of Step 2). It samples with our own `zero.generate` (with the KV cache).
  Only this backend can do logits distillation.
- `"openai"`: any OpenAI-compatible HTTP interface. In Step 2, vLLM serves the teacher:
  `vllm serve <teacher> --enable-auto-tool-choice --tool-call-parser hermes`. Sequence-level
  distillation only. The unit tests use a local fake server and no network.

**Execution check**: for each task, the first-turn tool calls of the teacher must get the full
`score_tool_calls` score (correct function name + arguments). Then the tools run, and the results go
back to the teacher. The final answer must pass `score_final_answer`. The other trajectories are dropped.
Each row and the metadata file (`<out_jsonl>.meta.json`) record the name, version, and license of the
teacher (GOAL.md 3.3). With `backend = "openai"`, or when the teacher is not a model of this project,
the config must set `license_allows_distillation = true`. This means that the license allows the use of
the outputs to train other models (for example Apache-2.0 / MIT). Otherwise the program does not run.

**The teacher in the smoke test is a stand-in.** GOAL.md asks for a small open model that runs on a CPU
as the teacher of the smoke test. But this environment cannot access huggingface.co, so it cannot
download open weights. Thus `configs/tiny/distill.toml` uses the tiny SFT model **itself** as the
teacher (self-distillation / rejection-sampling fine-tuning). The sequence-level data is its own
samples, filtered by the execution check. The logits distillation term is the KL to "the frozen SFT
model", and it is 0 at the start. This verifies only the code path. It does not show the effect of
distillation.
To use a real teacher in Step 2, change only `[teacher]`: `backend = "local"` +
`path = <HF folder of an open Qwen3-family teacher>`, or `backend = "openai"` + the address of the
vLLM service. Logits distillation needs the same tokenizer. Our own tokenizer is different from all
open teachers, so in practice we do only sequence-level distillation.

**On-policy distillation** (GOAL.md 2.1 lists it as "to be verified"; off by default): the student
samples. On the sequences of the student, the reverse KL(p_S ‖ p_T) is minimized at each position
(the GKD of Agarwal et al. 2023, the special case λ = 1). With `on_policy_steps > 0`, this many steps
run after offline distillation, with this one teacher.
Chapter 17 found that on-policy distillation is a consensus method. The main line uses it in a
separate last stage, with several of our own checkpoints as teachers: `zero/post/opd.py`
(cross-stage on-policy distillation, the GLM-5 recipe).
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.request
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from zero.post.chat import assistant_text
from zero.post.common import (
    chat_complete,
    load_policy,
    load_post_config,
    pad_batch,
    write_jsonl,
)
from zero.tokenizer import Tokenizer


@dataclass
class TeacherConfig:
    backend: str = (
        "local"  # "local" (zero checkpoint / HF folder) | "openai" (OpenAI-compatible HTTP)
    )
    path: str = ""  # local: checkpoint folder or HF folder
    base_url: str = ""  # openai: for example http://localhost:8000/v1
    model: str = ""  # openai: the model name on the server
    api_key_env: str = "OPENAI_API_KEY"  # read the API key from this environment variable (a local vLLM does not need it)
    name: str = ""  # for the record: teacher name, for example "Qwen3-8B"
    version: str = ""  # for the record: version / commit / release date
    license: str = ""  # for the record: license, for example "Apache-2.0"
    license_allows_distillation: bool = (
        False  # the license allows the use of the outputs to train other models
    )
    temperature: float = 0.7
    top_p: float = 0.95
    max_new_tokens: int = 96
    timeout_s: float = 120.0


@dataclass
class DistillConfig:
    out_jsonl: str = ""  # sequence-level distillation data (the trajectories that passed the check)
    shard_dir: str = ""  # packed windows
    n_tasks: int = 64  # number of tool_env training tasks
    samples_per_task: int = 4  # number of teacher samples for each task
    keep_per_task: int = 1  # maximum number of verified trajectories to keep for each task
    env_seed: int = 0
    mix_sft_jsonl: str = (
        ""  # optional: mix in the original SFT data (prevents forgetting with little teacher data)
    )
    mix_sft_max: int = 0  # maximum number of rows to mix in
    logits_kd: bool = (
        True  # add the logits distillation term (needs the local backend + the same tokenizer)
    )
    kd_alpha: float = 0.5
    kd_temperature: float = 1.0
    kd_topk: int = 0
    # Real data (2026-10-10). Each source is optional; n_tasks = 0 turns the toy environment off.
    task_files: list[str] = field(
        default_factory=list
    )  # function-calling tasks: keep teacher answers that score exact
    prompt_files: list[str] = field(
        default_factory=list
    )  # prompts without answers (e.g. Chinese chat): filtered teacher answers
    prompt_license: str = ""  # license of the prompts in prompt_files (written to each row); empty: the teacher license
    max_prompts: int = 0  # per prompt file; 0 = all
    max_tasks: int = 0  # per task file; 0 = all
    match_lang: bool = (
        True  # prompt_files: the answer must be in the language of the prompt (zh / en)
    )
    concurrency: int = (
        1  # parallel requests to an OpenAI-compatible teacher server (the local backend uses 1)
    )
    on_policy_steps: int = 0  # >0: this many on-policy distillation steps after offline distillation (to be verified; off by default)
    on_policy_lr: float = 1e-4
    on_policy_batch: int = 4
    overwrite: bool = False


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------


def kd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    mask: torch.Tensor,
    temperature: float = 1.0,
    topk: int = 0,
) -> torch.Tensor:
    """Forward KL(p_T ‖ p_S), mean over the mask positions, × τ². logits: (B, T, V), mask: (B, T).

    **Both must come from the same tokenizer** (the same vocabulary). Otherwise dimension v is not the
    same token in both, and the comparison has no meaning."""
    if student_logits.shape != teacher_logits.shape:
        raise ValueError(
            f"The student logits {tuple(student_logits.shape)} and the teacher logits {tuple(teacher_logits.shape)} have different shapes: "
            "logits distillation needs the same tokenizer for both"
        )
    tau = temperature
    s = student_logits.float() / tau
    t = teacher_logits.float() / tau
    if topk > 0:
        t_top, idx = t.topk(topk, dim=-1)
        logp_t = F.log_softmax(t_top, dim=-1)
        logp_s = F.log_softmax(s, dim=-1).gather(-1, idx)
    else:
        logp_t = F.log_softmax(t, dim=-1)
        logp_s = F.log_softmax(s, dim=-1)
    kl = (logp_t.exp() * (logp_t - logp_s)).sum(-1)
    m = mask.float()
    return (kl * m).sum() / m.sum().clamp(min=1.0) * tau * tau


def reverse_kl_loss(
    student_logits: torch.Tensor, teacher_logits: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Reverse KL(p_S ‖ p_T), mean over the mask positions.

    On-policy distillation uses it. The gradient goes only to the student.
    """
    logp_s = F.log_softmax(student_logits.float(), dim=-1)
    logp_t = F.log_softmax(teacher_logits.float(), dim=-1)
    kl = (logp_s.exp() * (logp_s - logp_t)).sum(-1)
    m = mask.float()
    return (kl * m).sum() / m.sum().clamp(min=1.0)


# ---------------------------------------------------------------------------
# Teacher backends
# ---------------------------------------------------------------------------


class LocalTeacher:
    """A local model (zero checkpoint or HF folder with the Qwen3 architecture). It samples with zero.generate."""

    def __init__(self, cfg: TeacherConfig, device: str | torch.device = "cpu") -> None:
        self.cfg = cfg
        self.model, self.tok = load_policy(cfg.path, device=device)
        self.model.eval()

    def complete(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]], n: int, seed: int
    ) -> list[str]:
        c = self.cfg
        return chat_complete(
            self.model, self.tok, messages, tools, c.max_new_tokens, c.temperature, c.top_p, n, seed
        )


def to_openai_messages(messages: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Our message format → OpenAI Chat Completions format.

    In tool_calls, the arguments are JSON strings, and each call has an id.
    """
    out: list[dict[str, Any]] = []
    call_ids: list[str] = []
    k = 0
    for m in messages:
        m = dict(m)
        if m.get("role") == "assistant" and m.get("tool_calls"):
            tcs = []
            for tc in m["tool_calls"]:
                f = tc.get("function", tc)
                args = f.get("arguments", {})
                cid = f"call_{k}"
                k += 1
                call_ids.append(cid)
                tcs.append(
                    {
                        "id": cid,
                        "type": "function",
                        "function": {
                            "name": f["name"],
                            "arguments": args
                            if isinstance(args, str)
                            else json.dumps(args, ensure_ascii=False),
                        },
                    }
                )
            m["tool_calls"] = tcs
            m["content"] = m.get("content") or None
        elif m.get("role") == "tool":
            m.setdefault("tool_call_id", call_ids.pop(0) if call_ids else "call_0")
        out.append(m)
    return out


def openai_message_to_text(msg: dict[str, Any]) -> str:
    """Assistant message from OpenAI → the text that the assistant must generate in our template."""
    calls = []
    for tc in msg.get("tool_calls") or []:
        f = tc.get("function", tc)
        args = f.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                pass  # keep the original string: it renders as broken JSON, and the check removes it
        calls.append({"name": f.get("name"), "arguments": args})
    return assistant_text(
        {"role": "assistant", "content": msg.get("content") or "", "tool_calls": calls}
    )


class OpenAITeacher:
    """OpenAI-compatible HTTP interface (vLLM / SGLang / vendor APIs). It uses only urllib from the standard library."""

    def __init__(self, cfg: TeacherConfig) -> None:
        if not cfg.base_url or not cfg.model:
            raise ValueError("[teacher] backend=openai needs base_url and model")
        self.cfg = cfg

    def complete(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]], n: int, seed: int
    ) -> list[str]:
        c = self.cfg
        payload = {
            "model": c.model,
            "messages": to_openai_messages(messages),
            **({"tools": tools} if tools else {}),
            "n": n,
            "temperature": c.temperature,
            "top_p": c.top_p,
            "max_tokens": c.max_new_tokens,
            "seed": seed,
        }
        headers = {"Content-Type": "application/json"}
        key = os.environ.get(c.api_key_env, "")
        if key:
            headers["Authorization"] = f"Bearer {key}"
        req = urllib.request.Request(
            c.base_url.rstrip("/") + "/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=c.timeout_s) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return [openai_message_to_text(ch["message"]) for ch in data.get("choices", [])]


def build_teacher(
    cfg: TeacherConfig, device: str | torch.device = "cpu"
) -> LocalTeacher | OpenAITeacher:
    if cfg.backend == "local":
        return LocalTeacher(cfg, device)
    if cfg.backend == "openai":
        return OpenAITeacher(cfg)
    raise ValueError(f"[teacher] backend must be local / openai, not {cfg.backend!r}")


def check_license(cfg: TeacherConfig, is_self: bool) -> None:
    """GOAL.md 3.3: the license of the teacher must allow "the use of the outputs to train other models".

    Exception: a model of this project as a stand-in.
    """
    if is_self:
        return
    if not cfg.license_allows_distillation:
        raise PermissionError(
            f"The license of the teacher {cfg.name or cfg.path or cfg.model!r} is not confirmed to allow distillation. "
            "Verify the license (for example Apache-2.0 / MIT), then set license and license_allows_distillation = true in [teacher]"
        )


# ---------------------------------------------------------------------------
# Teacher data: sample → execute → check
# ---------------------------------------------------------------------------


def teacher_trajectories(
    teacher: LocalTeacher | OpenAITeacher,
    task: Any,
    n: int,
    keep: int,
    seed: int,
) -> tuple[list[list[dict[str, Any]]], int]:
    """For one task, sample the first turn of the teacher n times (one request / one batch).

    Only the samples with a full tool-call score continue: run the tools, feed back the results, and let
    the teacher write the final answer (greedy). Keep a sample only if its final answer also passes.
    Return (list of the full conversations that passed, number of candidates)."""
    from zero.post.chat import parse_assistant
    from zero.post.envs.tool_env import execute_safely, score_final_answer, score_tool_calls

    firsts = teacher.complete(task.messages, task.tools, n, seed)
    kept: list[list[dict[str, Any]]] = []
    for j, text in enumerate(firsts):
        if len(kept) >= keep:
            break
        if score_tool_calls(task, text).total < 0.999:
            continue
        parsed = parse_assistant(text)
        msgs = [dict(m) for m in task.messages] + [parsed.to_message()]
        if parsed.tool_calls:
            for c in parsed.tool_calls:
                msgs.append(
                    {
                        "role": "tool",
                        "content": json.dumps(
                            execute_safely(c["name"], c["arguments"]), ensure_ascii=False
                        ),
                    }
                )
            final = teacher.complete(msgs, task.tools, 1, seed + 7919 * (j + 1))[0]
            if not score_final_answer(task, final).answer_ok:
                continue
            msgs.append(parse_assistant(final).to_message())
        elif not score_final_answer(task, text).answer_ok:
            continue
        kept.append(msgs)
    return kept, len(firsts)


def fc_task_answers(
    teacher: Any, task: Any, n: int, keep: int, seed: int
) -> tuple[list[list[dict[str, Any]]], int]:
    """Rejection sampling on a real function-calling task: keep the first turns that score exact.

    For a "no call" task, exact means: no call and a non-empty reply (the teacher's own wording).
    """
    from zero.post.chat import parse_assistant
    from zero.post.envs.fc_tasks import score_fc

    outs = teacher.complete(task.messages, task.tools, n, seed)
    kept = []
    for text in outs:
        if len(kept) >= keep:
            break
        if score_fc(task, text).exact:
            kept.append([dict(m) for m in task.messages] + [parse_assistant(text).to_message()])
    return kept, len(outs)


def prompt_answers(
    teacher: Any, messages: list[dict[str, Any]], n: int, keep: int, seed: int, match_lang: bool
) -> tuple[list[list[dict[str, Any]]], int, Counter]:
    """Teacher answers to a prompt without a reference answer, with simple filters.

    Dropped: empty answers, special tokens (forged turns), unclosed <think>, tool calls (no tools were
    offered), and, with match_lang, answers in another language than the prompt (zh / en).
    """
    from zero.post.chat import parse_assistant
    from zero.post.envs.fc_tasks import guess_lang
    from zero.post.sft_data import FORBIDDEN

    outs = teacher.complete(messages, [], n, seed)
    users = [m.get("content") or "" for m in messages if m.get("role") == "user"]
    want = guess_lang(users[-1]) if users else ""
    kept, why = [], Counter()
    for text in outs:
        if len(kept) >= keep:
            break
        parsed = parse_assistant(text)
        if not parsed.content.strip():
            why["empty"] += 1
        elif any(t in text for t in FORBIDDEN):
            why["special_tokens"] += 1
        elif parsed.errors:
            why["format"] += 1
        elif parsed.tool_calls:
            why["tool_call"] += 1
        elif match_lang and want and guess_lang(parsed.content) != want:
            why["language"] += 1
        else:
            kept.append(
                [dict(m) for m in messages] + [{"role": "assistant", "content": parsed.content}]
            )
    return kept, len(outs), why


def _distill_jobs(dc: DistillConfig) -> list[tuple[str, Any]]:
    """All teacher jobs, in a fixed order: (kind, item). kind: env | fc | prompt."""
    from types import SimpleNamespace

    from zero.post.common import read_jsonl
    from zero.post.envs.fc_tasks import load_fc_tasks
    from zero.post.envs.tool_env import generate_tasks
    from zero.post.opd import conversation_prompt

    jobs: list[tuple[str, Any]] = []
    if dc.n_tasks > 0:
        jobs += [("env", t) for t in generate_tasks(dc.n_tasks, seed=dc.env_seed, split="train")]
    for f in dc.task_files:
        fc = load_fc_tasks(f)
        jobs += [("fc", t) for t in (fc[: dc.max_tasks] if dc.max_tasks > 0 else fc)]
    for f in dc.prompt_files:
        rows = read_jsonl(f)
        if dc.max_prompts > 0:
            rows = rows[: dc.max_prompts]
        for i, r in enumerate(rows):
            if "prompt" in r and "messages" not in r:
                r = {"messages": [{"role": "user", "content": r["prompt"]}]}
            p = conversation_prompt(r)
            if p is not None:
                jobs.append(
                    (
                        "prompt",
                        SimpleNamespace(messages=p.messages, id=f"{Path(f).stem}-{r.get('id', i)}"),
                    )
                )
    return jobs


def generate_kd_data(
    teacher: LocalTeacher | OpenAITeacher,
    tcfg: TeacherConfig,
    dc: DistillConfig,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Generate the sequence-level distillation data, write out_jsonl and out_jsonl.meta.json, and return the metadata.

    Sources (each optional): the toy tool environment (`n_tasks`, executed and checked end to end),
    real function-calling tasks (`task_files`, first turn must score exact), and prompts without
    answers (`prompt_files`, filtered). Requests to an OpenAI-compatible server run `concurrency` at a
    time; the order of the output does not depend on it (each job has a fixed seed).
    """
    from concurrent.futures import ThreadPoolExecutor

    jobs = _distill_jobs(dc)
    if not jobs:
        raise ValueError("[distill] no teacher jobs: set n_tasks > 0, task_files, or prompt_files")
    t0 = time.time()
    teacher_meta = {
        "name": tcfg.name,
        "version": tcfg.version,
        "license": tcfg.license,
        "license_allows_distillation": tcfg.license_allows_distillation,
        "backend": tcfg.backend,
        "path_or_model": tcfg.path or tcfg.model,
    }

    def run(i: int) -> tuple[list[list[dict[str, Any]]], int, Counter]:
        kind, item = jobs[i]
        seed = dc.env_seed * 1_000_003 + i
        if kind == "env":
            kept, n = teacher_trajectories(
                teacher, item, dc.samples_per_task, dc.keep_per_task, seed
            )
            return kept, n, Counter()
        if kind == "fc":
            kept, n = fc_task_answers(teacher, item, dc.samples_per_task, dc.keep_per_task, seed)
            return kept, n, Counter()
        return prompt_answers(
            teacher, item.messages, dc.samples_per_task, dc.keep_per_task, seed, dc.match_lang
        )

    workers = dc.concurrency if isinstance(teacher, OpenAITeacher) else 1
    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(run, range(len(jobs))))
    else:
        results = []
        for i in range(len(jobs)):
            results.append(run(i))
            if (i + 1) % 1000 == 0:
                log(f"[distill] {i + 1}/{len(jobs)} teacher jobs")
    rows = []
    by_kind: dict[str, Counter] = {}
    for (kind, item), (kept, n, why) in zip(jobs, results):
        c = by_kind.setdefault(kind, Counter())
        c.update(candidates=n, verified=len(kept), jobs=1, jobs_with_verified=int(bool(kept)))
        c.update({f"drop_{k}": v for k, v in why.items()})
        if kind == "prompt":
            lic, tools = dc.prompt_license or tcfg.license, None
        else:
            lic, tools = getattr(item, "license", "") or "own", item.tools
        for msgs in kept:
            row = {
                "messages": msgs,
                "task_id": item.id,
                "source": f"distill-{kind}",
                "license": lic,
                "teacher": teacher_meta,
            }
            if tools:
                row["tools"] = tools
            rows.append(row)
    write_jsonl(dc.out_jsonl, rows)
    n_cand = sum(c["candidates"] for c in by_kind.values())
    meta = {
        "teacher": teacher_meta,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "generation": {
            "temperature": tcfg.temperature,
            "top_p": tcfg.top_p,
            "max_new_tokens": tcfg.max_new_tokens,
        },
        "env": {"n_tasks": dc.n_tasks, "env_seed": dc.env_seed, "split": "train"},
        "task_files": list(dc.task_files),
        "prompt_files": list(dc.prompt_files),
        "by_source": {k: dict(v) for k, v in by_kind.items()},
        "n_candidates": n_cand,
        "n_verified": len(rows),
        "tasks_with_verified": len({r["task_id"] for r in rows}),
        "pass_rate": len(rows) / max(n_cand, 1),
        "filter": {
            "env": "score_tool_calls == 1 and score_final_answer.answer_ok (zero/post/envs/tool_env.py)",
            "fc": "score_fc exact on the first turn (zero/post/envs/fc_tasks.py)",
            "prompt": "non-empty, no special tokens, no tool calls, same language as the prompt"
            + ("" if dc.match_lang else " (language check off)"),
        },
        "seconds": round(time.time() - t0, 1),
    }
    Path(str(dc.out_jsonl) + ".meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2)
    )
    log(
        f"[distill] teacher {tcfg.name or tcfg.path}: {n_cand} candidates → {len(rows)} kept"
        f" (pass rate {meta['pass_rate']:.1%}, {meta['seconds']}s) {meta['by_source']}"
    )
    return meta


# ---------------------------------------------------------------------------
# Training: sequence-level (SFT loss) + logits distillation
# ---------------------------------------------------------------------------


def _make_distill_trainer_cls():  # noqa: ANN202
    from zero.train.trainer import Trainer

    class DistillTrainer(Trainer):
        """Change only one thing in the general Trainer: loss = (1-α)·CE + α·KL(teacher ‖ student)."""

        def __init__(
            self, cfg: Any, teacher_model: torch.nn.Module, dc: DistillConfig, **kw: Any
        ) -> None:
            super().__init__(cfg, **kw)
            self.teacher = teacher_model.to(self.device).eval()
            for p in self.teacher.parameters():
                p.requires_grad_(False)
            self.dc = dc
            self.last_kd = 0.0
            self.last_ce = 0.0

        def _forward_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            from zero.model import cross_entropy_loss

            with self.autocast():
                logits = self.model(x)
                with torch.no_grad():
                    t_logits = self.teacher(x)
            ce = cross_entropy_loss(logits, y)
            kd = kd_loss(logits, t_logits, y != -100, self.dc.kd_temperature, self.dc.kd_topk)
            self.last_kd = float(kd.detach())
            self.last_ce = float(ce.detach())
            a = self.dc.kd_alpha
            return (1 - a) * ce + a * kd

        def extra_metrics(self) -> dict[str, Any]:
            return {"ce": self.last_ce, "kd": self.last_kd}

    return DistillTrainer


def on_policy_distill(
    student: torch.nn.Module,
    teacher: torch.nn.Module,
    tok: Tokenizer,
    tasks: Sequence[Any],
    steps: int,
    lr: float = 1e-4,
    batch: int = 4,
    max_new_tokens: int = 64,
    temperature: float = 1.0,
    seed: int = 0,
    log: Callable[[str], None] = print,
) -> list[dict[str, float]]:
    """On-policy distillation (to be verified; see the module docstring).

    The student samples → minimize the reverse KL(p_S ‖ p_T) on the samples of the student.
    """
    from zero.post.grpo import sample_group

    opt = torch.optim.AdamW(
        [p for p in student.parameters() if p.requires_grad], lr=lr, weight_decay=0.0
    )
    hist = []
    for step in range(steps):
        task = tasks[(seed + step) % len(tasks)]
        p_ids, resps = sample_group(
            student, tok, task, batch, max_new_tokens, temperature, 1.0, seed * 1000 + step
        )
        seqs = [p_ids + r for r in resps]
        masks = [[False] * len(p_ids) + [True] * len(r) for r in resps]
        ids, mask = pad_batch(seqs, masks, tok.eot_id, next(student.parameters()).device)
        student.train()
        s_logits = student(ids[:, :-1])
        with torch.no_grad():
            t_logits = teacher(ids[:, :-1])
        loss = reverse_kl_loss(s_logits, t_logits, mask[:, 1:])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        hist.append({"step": step + 1, "reverse_kl": float(loss.detach())})
        log(f"[on-policy] step {step + 1}/{steps} | reverse KL {float(loss.detach()):.4f}")
    return hist


def run_distill(
    src: str | os.PathLike | dict[str, Any],
    overrides: Sequence[str] | None = None,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Teacher data generation → (mix in SFT data) → packing → train the student.

    CPU, 1 GPU, or N GPUs with torchrun: rank 0 generates and packs the data, then every rank trains
    with the same `Trainer` as SFT (DDP). Return {"meta", "history", ...}.
    """
    from zero.train.dist import cleanup, init_distributed

    cfg, sec = load_post_config(
        src, {"teacher": TeacherConfig, "distill": DistillConfig}, overrides
    )
    info = init_distributed(cfg.train.device)
    try:
        return _distill(cfg, sec["teacher"], sec["distill"], info, log)
    finally:
        cleanup()


def _distill(
    cfg: Any, tcfg: TeacherConfig, dc: DistillConfig, info: Any, log: Callable[[str], None]
) -> dict[str, Any]:
    from zero.config import DataSourceConfig
    from zero.post.common import read_jsonl
    from zero.post.sft import build_sft_shards
    from zero.train.dist import barrier
    from zero.train.trainer import Trainer

    tc = cfg.train
    if tc.cpu_threads > 0:
        torch.set_num_threads(tc.cpu_threads)
    if cfg.train.data.format != "sft":
        raise ValueError(
            'The distillation config needs [data] format = "sft" (the teacher trajectories are packed as SFT data)'
        )
    if dc.on_policy_steps > 0 and info.world_size > 1:
        raise ValueError(
            "[distill] on_policy_steps runs in one process; for N GPUs use zero.post.opd"
        )
    tok = Tokenizer.load(tc.data.tokenizer)

    # Is the teacher "a model of this project" (the stand-in of the smoke test)?
    # Yes for the local backend + a zero checkpoint folder.
    is_self = tcfg.backend == "local" and not (Path(tcfg.path) / "config.json").exists()
    check_license(tcfg, is_self)
    # Logits distillation needs a local teacher on every rank; data generation needs it only on rank 0.
    use_logits = dc.logits_kd and tcfg.backend == "local"
    if dc.logits_kd and not use_logits:
        log(
            "[distill] the teacher is not a local model, so no logits: sequence-level distillation only"
        )
    meta_path = Path(str(dc.out_jsonl) + ".meta.json")
    reuse = Path(dc.out_jsonl).exists() and meta_path.exists() and not dc.overwrite
    teacher = None
    if use_logits or (info.is_main and not reuse):
        # With CUDA, the local teacher is on this rank's GPU. Verified on one RTX 3090
        # (2026-10, see runs/2026-10-01-gpu0-check/).
        teacher = build_teacher(tcfg, info.device)
    # First check if logits distillation is possible. For different tokenizers, raise the error
    # early, before the generation of the teacher data.
    if use_logits and teacher.tok.hash() != tok.hash():  # type: ignore[union-attr]
        raise ValueError(
            "Logits distillation needs the same tokenizer for the teacher and the student (the hashes are different). For a teacher with a different tokenizer, set [distill] logits_kd = false"
        )

    shard_dir = Path(dc.shard_dir or Path(tc.out_dir) / "data")
    meta: dict[str, Any] = {}
    stats: dict[str, Any] = {}
    if info.is_main:
        if reuse:
            meta = json.loads(meta_path.read_text())
            log(
                f"[distill] reusing the existing teacher data {dc.out_jsonl} ({meta['n_verified']} rows)"
            )
        else:
            meta = generate_kd_data(teacher, tcfg, dc, log)  # type: ignore[arg-type]
        rows = read_jsonl(dc.out_jsonl)
        if dc.mix_sft_jsonl and dc.mix_sft_max > 0:
            extra = read_jsonl(dc.mix_sft_jsonl)[: dc.mix_sft_max]
            rows = rows + extra
            log(
                f"[distill] mixed in {len(extra)} rows of the original SFT data ({len(rows)} rows in total)"
            )
        if not rows:
            raise ValueError(
                "[distill] no training data: no teacher trajectory passed the check, and no SFT data was mixed in"
            )
        mixed = shard_dir / "train_mix.jsonl"
        write_jsonl(mixed, rows)
        stats = build_sft_shards(mixed, tok, tc.data.seq_len, shard_dir / "train.bin")
        log(f"[distill] packing: {stats}")
    barrier()  # the other ranks wait for the packed windows

    tc.data.sources = [DataSourceConfig(name="distill", path=str(shard_dir / "train.bin"))]
    tc.data.val = ""
    tc.eval_every = 0
    os.makedirs(tc.out_dir, exist_ok=True)
    # [train] device selects the device (before 2026-10 the code always used the CPU; found on an RTX 3090,
    # see runs/2026-10-01-gpu0-check/).
    if use_logits:
        trainer = _make_distill_trainer_cls()(cfg, teacher.model, dc, info=info, log=log)  # type: ignore[union-attr]
    else:
        trainer = Trainer(cfg, info, log)
    history = trainer.train()

    on_policy_hist: list[dict[str, float]] = []
    if dc.on_policy_steps > 0 and isinstance(teacher, LocalTeacher):
        from zero.post.envs.tool_env import generate_tasks
        from zero.train.checkpoint import save_checkpoint

        student = trainer.raw_model
        tasks = generate_tasks(64, seed=dc.env_seed + 1, split="train")
        on_policy_hist = on_policy_distill(
            student,
            teacher.model,
            tok,
            tasks,
            dc.on_policy_steps,
            dc.on_policy_lr,
            dc.on_policy_batch,
            tcfg.max_new_tokens,
            1.0,
            tc.seed,
            log,
        )
        save_checkpoint(
            tc.checkpoint_dir,
            trainer.step + dc.on_policy_steps,
            student,
            meta={"config": cfg.to_dict(), "on_policy_steps": dc.on_policy_steps},
        )
    summary = {
        "meta": meta,
        "pack": stats,
        "history": history,
        "on_policy": on_policy_hist,
        "config": {"teacher": asdict(tcfg), "distill": asdict(dc)},
    }
    if info.is_main:
        Path(tc.out_dir, "distill_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, default=str)
        )
    return summary


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Distillation (Chapter 17)")
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = ap.parse_args(argv)
    run_distill(args.config, args.set)


if __name__ == "__main__":
    main()
