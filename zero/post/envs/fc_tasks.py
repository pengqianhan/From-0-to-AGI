"""Real function-calling tasks: one JSONL format, a generic verifier, converters, decontamination.

Chapter 19 uses this module. `tool_env.py` is a toy environment with 6 mock tools; this module lets
GRPO and on-policy distillation train on tasks with **any** tool schema, from open data sets.

    # Build a task file from open data sets, without the items that overlap with the evaluation sets
    uv run python -m zero.post.envs.fc_tasks build \\
        --src hermes:data/raw/hermes-function-calling-v1/func-calling-singleturn.json \\
        --src toolace:data/raw/ToolACE/data.json \\
        --exclude-bfcl data/eval/bfcl \\
        --out data/rl/fc_all.jsonl --dev-out data/rl/fc_dev.jsonl --dev-size 500
    # Source kinds: xlam, hermes, toolace, openai (messages + tools), fc (this format).
    # A license can follow the path: --src openai:path.jsonl:Apache-2.0
    # Split into SFT conversations and RL tasks that do not overlap
    uv run python -m zero.post.envs.fc_tasks split data/rl/fc_all.jsonl \\
        --sft-out data/sft/fc_sft.jsonl --rl-out data/rl/fc_train.jsonl --sft-frac 0.5
    # Evaluation data as task files, for decontamination (--exclude-tasks, sft_data decontam_tasks) and for
    # the one-off check that this scorer agrees with the benchmark's answers. Never train on them, and do
    # not use them to select checkpoints: test sets run only at the gates (eval/PREREGISTRATION.md, 8).
    uv run python -m zero.post.envs.fc_tasks export bfcl <site-packages>/bfcl_eval/data --out data/eval/bfcl_v4.jsonl
    uv run python -m zero.post.envs.fc_tasks export acebench ACEBench/data_all/data_zh --out data/eval/acebench_zh.jsonl
    # Score a model output against one task (for debugging)
    uv run python -m zero.post.envs.fc_tasks score data/rl/fc_dev.jsonl 0 '<tool_call>...</tool_call>'

**Task format** (one JSON object per line; the same fields as `tool_env.Task` where they overlap):

    {"id": "xlam-000123", "source": "xlam", "license": "CC-BY-4.0", "lang": "en",
     "tools": [{"type": "function", "function": {"name": ..., "description": ...,
                "parameters": {"type": "object", "properties": {...}, "required": [...]}}}],
     "messages": [{"role": "user", "content": "..."}],          # the prompt (may include history)
     "gold_calls": [{"name": "get_weather", "arguments": {"city": "Paris"},
                     "alternatives": {"unit": ["celsius", ""]}}]}

- `gold_calls` is the set of calls of the next assistant turn. The order does not matter (parallel
  calls). An empty list means that **no** call is correct (BFCL calls this "irrelevance").
- `alternatives` (optional, per argument): all accepted values. `""` in the list means "the argument
  may be left out". This is the "possible answer" format of BFCL, so BFCL items convert without loss.
  An argument in `arguments` without an entry in `alternatives` accepts only that one value.
- Only the **first** assistant turn is scored (a single-step task, as in the GRPO of `tool_env`).
  Multi-step tasks with tool execution need executable tools; they are not in this format.

**Reward** (`score_fc`): the same scale and the same anti-hacking rules as `tool_env.score_tool_calls`,
so the GRPO logs and thresholds stay comparable:

    Format error (broken JSON, unpaired tags, forged <tool_response>, call JSON outside the tags,
      too long, too many calls) → -1
    A call when no call is correct → -0.5; no call when a call is necessary → 0
    No call, correctly → 0.5 (the content of the reply cannot be checked)
    Otherwise 0.1 + 0.9 × Σ score of each gold call / number of gold calls
              − 0.25 × number of extra calls − 0.5 × number of calls that break the schema
        Score of each gold call (one-to-one matching): name + all arguments accepted → 1;
        only the name → 0.2
    Then clip to [-1, 1]

**Order**: the calls are first matched one to one against the gold calls; only the calls that match
no gold call are schema-checked. A call that matches an accepted answer is correct even if the tool
description disagrees with it (BFCL's function docs sometimes contradict their own answers, for
example `year` typed integer with the answer "dontcare"; checked on all 3,641 BFCL v4 single-turn
items, where every gold answer scores exact). The call limit is max(MAX_CALLS, 2 × gold calls).

**Schema check** (`schema_errors`): a call to a tool that was not offered, a missing required
argument, an argument that is not in the schema, a wrong type, or a value outside `enum` breaks the
schema. Such a call gets no score and a penalty. The types follow JSON schema; the Python-style names
of BFCL (`dict`, `float`, `tuple`, `any`) are accepted too.

**Value comparison** (`values_match`): strings without surrounding whitespace and case-insensitive;
numbers by value (1 == 1.0); lists in order; objects key by key. This is the AST match of BFCL, with a
slightly looser string rule (BFCL also ignores case and some punctuation).

**Decontamination**: `build --exclude-bfcl <dir>` drops a training task when (a) one of its tool names
is a function name of BFCL, or (b) its user text shares a 13-gram with a BFCL question
(`zero.data.decontam`). The dropped counts go into `<out>.meta.json`. `--exclude-tasks` does the same
with any task file in this format (for example ACEBench after conversion).
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from zero.post.chat import parse_assistant
from zero.post.envs.tool_env import (
    MAX_CALLS,
    NO_TOOL_REWARD,
    Reward,
    _bare_call,
    _forged,
)

# Real tasks have long arguments (an ACEBench parallel-call answer is > 2,000 characters), so the output
# limit is larger than the 2,000 characters of the toy environment. It still stops endless outputs.
MAX_OUTPUT_CHARS = 8000

# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------


@dataclass
class FCTask:
    id: str
    tools: list[dict[str, Any]]
    messages: list[dict[str, Any]]
    gold_calls: list[dict[str, Any]]
    source: str = ""
    license: str = ""
    lang: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> FCTask:
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})

    @property
    def query(self) -> str:
        users = [m.get("content") or "" for m in self.messages if m.get("role") == "user"]
        return users[-1] if users else ""

    def tool(self, name: str) -> dict[str, Any] | None:
        for t in self.tools:
            f = t.get("function", t)
            if f.get("name") == name:
                return f
        return None


def load_fc_tasks(
    path: str | Path, strict: bool = False, check_schema: bool = True
) -> list[FCTask]:
    """Read a task file. Invalid tasks (`task_errors`) are skipped, or raise with strict=True.

    check_schema=False for evaluation files (BFCL / ACEBench docs sometimes contradict their answers).
    """
    out = []
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            t = FCTask.from_dict(json.loads(line))
            errs = task_errors(t, check_schema)
            if errs:
                if strict:
                    raise ValueError(f"{path}:{n} ({t.id}): {'; '.join(errs)}")
                continue
            out.append(t)
    return out


def task_errors(t: FCTask, check_schema: bool = True) -> list[str]:
    """Problems that make a task unusable: no user message, a gold call that breaks its own schema, ...

    check_schema=False skips the schema check of the gold calls (for evaluation sets such as BFCL,
    whose function docs sometimes contradict their own answers; the scorer trusts the answers).
    """
    errs = []
    if not t.messages or t.messages[-1].get("role") not in ("user", "tool"):
        errs.append("the prompt must end with a user (or tool) message")
    names = [(t.get("function") or t).get("name") for t in t.tools]
    if len(set(names)) != len(names):
        errs.append("duplicate tool names")
    for g in t.gold_calls:
        f = t.tool(g.get("name", ""))
        if f is None:
            errs.append(f"gold call {g.get('name')!r} is not among the tools")
            continue
        se = (
            schema_errors(g.get("arguments", {}), f.get("parameters") or {}, g.get("alternatives"))
            if check_schema
            else []
        )
        if se:
            errs.append(f"gold call {g['name']} breaks its schema: {'; '.join(se)}")
    return errs


# ---------------------------------------------------------------------------
# Schema check and value comparison
# ---------------------------------------------------------------------------

_TYPE_ALIASES = {
    "dict": "object",
    "float": "number",
    "int": "integer",
    "str": "string",
    "bool": "boolean",
    "list": "array",
    "tuple": "array",
}


def _norm_type(t: Any) -> str:
    t = str(t).lower()
    return _TYPE_ALIASES.get(t, t)


def type_ok(value: Any, schema: dict[str, Any]) -> bool:
    t = schema.get("type")
    if t is None:
        return True
    types = [_norm_type(x) for x in (t if isinstance(t, list) else [t])]
    for ty in types:
        if ty == "any":
            return True
        if ty == "string" and isinstance(value, str):
            return True
        if ty == "boolean" and isinstance(value, bool):
            return True
        if ty == "integer" and isinstance(value, int) and not isinstance(value, bool):
            return True
        if ty == "integer" and isinstance(value, float) and value.is_integer():
            return True  # 3.0 for an integer argument: same value; JSON does not tell them apart
        if ty == "number" and isinstance(value, int | float) and not isinstance(value, bool):
            return True
        if ty == "array" and isinstance(value, list):
            return True
        if ty == "object" and isinstance(value, dict):
            return True
        if ty == "null" and value is None:
            return True
    return False


def _value_errors(value: Any, schema: dict[str, Any], where: str, depth: int = 0) -> list[str]:
    if not type_ok(value, schema):
        return [f"{where}: wrong type (want {schema.get('type')}, got {type(value).__name__})"]
    errs = []
    if "enum" in schema and not any(values_match(value, e) for e in schema["enum"]):
        errs.append(f"{where}: value not in enum")
    if depth < 4:
        if isinstance(value, list) and isinstance(schema.get("items"), dict):
            for i, v in enumerate(value):
                errs += _value_errors(v, schema["items"], f"{where}[{i}]", depth + 1)
        if isinstance(value, dict) and "properties" in schema:
            errs += _object_errors(value, schema, where, depth + 1)
    return errs


def _object_errors(
    args: dict[str, Any], schema: dict[str, Any], where: str, depth: int = 0
) -> list[str]:
    props = schema.get("properties") or {}
    errs = [
        f"{where}: missing required {r!r}" for r in schema.get("required") or [] if r not in args
    ]
    required = set(schema.get("required") or [])
    for k, v in args.items():
        if k not in props:
            errs.append(f"{where}: unknown argument {k!r}")
        elif v is None and k not in required:
            continue  # null for an optional argument: "not given" (BFCL accepts it)
        else:
            errs += _value_errors(v, props[k], f"{where}.{k}", depth)
    return errs


def schema_errors(
    args: Any, parameters: dict[str, Any], alternatives: dict[str, Any] | None = None
) -> list[str]:
    """Why `args` does not satisfy the `parameters` schema of a tool (empty list: it does).

    `alternatives` (of a gold call) only makes "required" arguments with "" optional.
    """
    if not isinstance(args, dict):
        return ["arguments is not an object"]
    if not parameters:
        return [] if not args else [f"unknown argument {k!r}" for k in args]
    schema = dict(parameters)
    if alternatives:
        schema["required"] = [
            r for r in schema.get("required") or [] if "" not in (alternatives.get(r) or [])
        ]
    return _object_errors(args, schema, "args")


def values_match(pred: Any, gold: Any) -> bool:
    """Equal after normalization: strings stripped and case-insensitive, numbers by value."""
    if isinstance(gold, bool) or isinstance(pred, bool):
        return isinstance(gold, bool) and isinstance(pred, bool) and pred == gold
    if isinstance(gold, int | float) and isinstance(pred, int | float):
        return abs(float(pred) - float(gold)) <= 1e-6 * max(1.0, abs(float(gold)))
    if isinstance(gold, str) and isinstance(pred, str):
        return pred.strip().lower() == gold.strip().lower()
    if isinstance(gold, list) and isinstance(pred, list):
        return len(gold) == len(pred) and all(values_match(p, g) for p, g in zip(pred, gold))
    if isinstance(gold, dict) and isinstance(pred, dict):
        return set(gold) == set(pred) and all(values_match(pred[k], gold[k]) for k in gold)
    return pred == gold


NESTED = (
    "$alt"  # {"$alt": {key: [accepted values]}}: a dict argument with alternatives per key (BFCL)
)


def _accepts(v: Any, a: Any) -> bool:
    """One accepted value `a` (maybe a nested-alternatives dict) accepts the predicted value `v`."""
    if isinstance(a, dict) and set(a) == {NESTED}:
        if not isinstance(v, dict):
            return False
        inner = a[NESTED]
        if set(v) - set(inner):
            return False
        for k, acc in inner.items():
            if k not in v:
                if "" in acc:
                    continue
                return False
            if not any(x != "" and _accepts(v[k], x) for x in acc):
                return False
        return True
    return values_match(v, a)


def call_matches(pred: dict[str, Any], gold: dict[str, Any], tool: dict[str, Any] | None) -> bool:
    """`pred` is an accepted answer for the gold call (name and every argument)."""
    if pred["name"] != gold["name"]:
        return False
    alts = gold.get("alternatives") or {}
    gargs = gold.get("arguments") or {}
    pargs = pred["arguments"]
    if not isinstance(pargs, dict):
        return False
    defaults = {
        k: s["default"]
        for k, s in ((tool or {}).get("parameters", {}).get("properties") or {}).items()
        if isinstance(s, dict) and "default" in s
    }
    for k in set(gargs) | set(alts) | set(pargs):
        accepted = list(alts.get(k, [])) or ([gargs[k]] if k in gargs else [])
        if k not in pargs:
            if not accepted or "" in accepted:
                continue
            # Left out, but the gold value is the default value: the same call
            if k in defaults and any(a != "" and values_match(defaults[k], a) for a in accepted):
                continue
            return False
        v = pargs[k]
        if not accepted:
            # An argument that the gold call does not mention: accepted only with its default value
            if k in defaults and values_match(v, defaults[k]):
                continue
            return False
        if not any(a != "" and _accepts(v, a) for a in accepted):
            return False
    return True


# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------


def score_fc(task: FCTask, text: str) -> Reward:
    """Score the first assistant output on one task. Same scale as `tool_env.score_tool_calls`."""
    if len(text) > MAX_OUTPUT_CHARS:
        return Reward(-1.0, False, details=["Output too long"])
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
    limit = max(MAX_CALLS, 2 * len(task.gold_calls))
    if len(calls) > limit:
        return Reward(-1.0, False, n_calls=len(calls), details=[f"More than {limit} calls"])

    gold = task.gold_calls
    details: list[str] = []
    # 1) One-to-one matching against the gold calls. A call that matches an accepted answer is correct
    #    even if the tool description disagrees with it: the possible answers decide (as in BFCL, whose
    #    function docs sometimes contradict their own answers).
    remaining = list(range(len(calls)))
    score = 0.0
    n_match = 0
    still = []
    for g in gold:
        f = task.tool(g["name"])
        hit = next((i for i in remaining if call_matches(calls[i], g, f)), None)
        if hit is None:
            still.append(g)
            continue
        remaining.remove(hit)
        n_match += 1
        score += 1.0
    # 2) The other calls must at least satisfy the schema of an offered tool
    n_invalid = 0
    unmatched_valid = []
    for i in remaining:
        c = calls[i]
        f = task.tool(c["name"])
        errs = (
            [f"Called a tool that was not offered: {c['name']}"]
            if f is None
            else schema_errors(c["arguments"], f.get("parameters") or {})
        )
        if errs:
            n_invalid += 1
            details.extend(errs)
        else:
            unmatched_valid.append(i)
    format_ok = n_invalid == 0
    if not gold:
        if not calls:
            if not parsed.content.strip():
                return Reward(0.0, True, 0, details=["Empty answer"])
            return Reward(
                NO_TOOL_REWARD,
                True,
                0,
                1.0,
                1.0,
                details=["Correctly made no tool call"],
                exact=True,
            )
        return Reward(
            -0.5,
            format_ok,
            len(calls),
            details=["Called a tool when no call was correct", *details],
        )
    if not calls:
        return Reward(0.0, True, 0, details=["A tool call was necessary, but there was no call"])
    # 3) Partial credit: the right function name with wrong arguments (schema-valid calls only)
    for g in still:
        hit = next((i for i in unmatched_valid if calls[i]["name"] == g["name"]), None)
        if hit is not None:
            unmatched_valid.remove(hit)
            score += 0.2
            details.append(f"Wrong arguments for {g['name']}")
        else:
            details.append(f"Missing call {g['name']}")
    n_extra = len(unmatched_valid)
    if n_extra:
        details.append(f"Extra calls: {n_extra}")
    total = 0.1 + 0.9 * score / len(gold) - 0.25 * n_extra - 0.5 * n_invalid
    total = max(-1.0, min(1.0, total))
    frac = n_match / len(gold)
    return Reward(total, format_ok, len(calls), frac, frac, details=details, exact=total >= 0.999)


def score_any(task: Any, text: str) -> Reward:
    """Score a task of either environment (tool_env.Task or FCTask)."""
    if isinstance(task, FCTask):
        return score_fc(task, text)
    from zero.post.envs.tool_env import score_tool_calls

    return score_tool_calls(task, text)


def load_task_pool(
    task_files: Sequence[str], n_tool_env: int, env_seed: int, split: str = "train"
) -> list[Any]:
    """The task pool of GRPO / OPD: the task files, or the toy environment if there are none."""
    if task_files:
        pool: list[Any] = []
        for p in task_files:
            pool += load_fc_tasks(p)
        if not pool:
            raise ValueError(f"No valid task in {list(task_files)}")
        return pool
    from zero.post.envs.tool_env import generate_tasks

    return list(generate_tasks(n_tool_env, seed=env_seed, split=split))


# ---------------------------------------------------------------------------
# Converters
# ---------------------------------------------------------------------------

_XLAM_TYPES = {
    "str": "string",
    "string": "string",
    "int": "integer",
    "integer": "integer",
    "float": "number",
    "number": "number",
    "bool": "boolean",
    "boolean": "boolean",
    "list": "array",
    "dict": "object",
    "set": "array",
    "tuple": "array",
    "any": "any",
}


def _xlam_param(spec: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    """One xLAM parameter ({"type": "List[int], optional", ...}) → (JSON schema, required)."""
    raw = str(spec.get("type", "any"))
    optional = "optional" in raw.lower()
    base = re.split(r"[,\[]", raw)[0].strip().lower()
    out: dict[str, Any] = {"type": _XLAM_TYPES.get(base, "any")}
    if spec.get("description"):
        out["description"] = spec["description"]
    if "default" in spec and spec["default"] not in (None, ""):
        out["default"] = spec["default"]
        optional = True
    return out, not optional


def from_xlam(row: dict[str, Any], idx: int, license: str = "CC-BY-4.0") -> FCTask:
    """A row of Salesforce xLAM / APIGen (query, tools, answers; tools/answers may be JSON strings).

    The license default is what the data card says at the time of writing; verify it before use.
    """
    tools_raw = row["tools"] if isinstance(row["tools"], list) else json.loads(row["tools"])
    answers = row["answers"] if isinstance(row["answers"], list) else json.loads(row["answers"])
    tools = []
    for t in tools_raw:
        props, required = {}, []
        for name, spec in (t.get("parameters") or {}).items():
            props[name], req = _xlam_param(spec if isinstance(spec, dict) else {})
            if req:
                required.append(name)
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": {"type": "object", "properties": props, "required": required},
                },
            }
        )
    gold = [{"name": a["name"], "arguments": a.get("arguments") or {}} for a in answers]
    return FCTask(
        id=f"xlam-{row.get('id', idx)}",
        tools=tools,
        messages=[{"role": "user", "content": row["query"]}],
        gold_calls=gold,
        source="xlam",
        license=license,
        lang="en",
    )


def from_openai_messages(row: dict[str, Any], idx: int, source: str, license: str) -> FCTask | None:
    """A conversation in the OpenAI format (messages with assistant tool_calls + tools).

    The prompt is everything before the first assistant turn; the gold calls are the tool_calls of that
    turn (none: an "irrelevance" task). Returns None if the conversation has no assistant turn.
    """
    msgs = row.get("messages") or []
    first = next((i for i, m in enumerate(msgs) if m.get("role") == "assistant"), None)
    if first is None or first == 0:
        return None
    gold = []
    for tc in msgs[first].get("tool_calls") or []:
        f = tc.get("function", tc)
        args = f.get("arguments", {})
        if isinstance(args, str):
            args = json.loads(args)
        gold.append({"name": f["name"], "arguments": args})
    tools = [
        t if "function" in t else {"type": "function", "function": t}
        for t in row.get("tools") or []
    ]
    return FCTask(
        id=f"{source}-{row.get('id', idx)}",
        tools=tools,
        messages=[
            {k: v for k, v in m.items() if k in ("role", "content", "tool_calls", "name")}
            for m in msgs[:first]
        ],
        gold_calls=gold,
        source=source,
        license=license,
        lang=row.get("lang", ""),
    )


def _bfcl_schema(s: Any) -> Any:
    """BFCL uses Python-style type names ("dict", "float", "tuple"); convert them to JSON schema."""
    if isinstance(s, dict):
        out = {k: _bfcl_schema(v) for k, v in s.items()}
        if "type" in out and isinstance(out["type"], str):
            out["type"] = _norm_type(out["type"])
        return out
    if isinstance(s, list):
        return [_bfcl_schema(x) for x in s]
    return s


def to_json_schema(f: dict[str, Any]) -> dict[str, Any]:
    """A BFCL function description with Python type names → the same with JSON-schema type names."""
    return _bfcl_schema(f)


def _bfcl_alt(x: Any) -> Any:
    """A BFCL accepted value; a dict whose values are all lists is a nested possible answer."""
    if isinstance(x, dict) and x and all(isinstance(v, list) for v in x.values()):
        return {NESTED: {k: [_bfcl_alt(y) for y in v] for k, v in x.items()}}
    return x


def _bfcl_first(acc: list[Any]) -> Any:
    """The first accepted value, as a plain value (nested possible answers resolved recursively)."""
    x = next((a for a in acc if a != ""), "")
    if isinstance(x, dict) and x and all(isinstance(v, list) for v in x.values()):
        out = {k: _bfcl_first(v) for k, v in x.items()}
        return {k: v for k, v in out.items() if v != ""}
    return x


def from_bfcl(question: dict[str, Any], answer: dict[str, Any] | None) -> FCTask:
    """One BFCL item (question file row + possible-answer row; no answer: irrelevance).

    **For decontamination and local dev checks only. Never train on BFCL.**
    ground_truth: [{func_name: {arg: [accepted values]}}]. The first accepted value is the gold value;
    all accepted values go into `alternatives`.
    """
    turns = question["question"]
    msgs = turns[0] if turns and isinstance(turns[0], list) else turns
    tools = [{"type": "function", "function": _bfcl_schema(f)} for f in question["function"]]
    gold = []
    for gt in (answer or {}).get("ground_truth", []):
        for name, args in gt.items():
            gold.append(
                {
                    "name": name,
                    "arguments": {
                        k: _bfcl_first(v) for k, v in args.items() if _bfcl_first(v) != ""
                    },
                    "alternatives": {k: [_bfcl_alt(x) for x in v] for k, v in args.items()},
                }
            )
    return FCTask(
        id=f"bfcl-{question['id']}",
        tools=tools,
        messages=[{"role": m["role"], "content": m["content"]} for m in msgs],
        gold_calls=gold,
        source="bfcl",
        license="Apache-2.0",
        lang="en",
    )


def read_bfcl_dir(root: str | Path) -> list[FCTask]:
    """All BFCL_v*_*.json question files under root, with the matching possible_answer files."""
    root = Path(root)
    out = []
    for q in sorted(root.rglob("BFCL_*.json")):
        if "possible_answer" in q.parts or "unused_datasets" in q.parts:
            continue
        if q.read_text("utf-8").lstrip().startswith("{\n"):
            continue  # not JSON Lines (format_sensitivity.json is a pretty-printed list of test ids)
        ans_path = q.parent / "possible_answer" / q.name
        answers = {}
        if ans_path.exists():
            for line in ans_path.read_text("utf-8").splitlines():
                if line.strip():
                    a = json.loads(line)
                    answers[a["id"]] = a
        for line in q.read_text("utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                if "function" in row and "question" in row:
                    out.append(from_bfcl(row, answers.get(row["id"])))
    return out


_CJK_RE = re.compile(r"[\u4e00-\u9fff]")


def guess_lang(text: str) -> str:
    return "zh" if len(_CJK_RE.findall(text)) >= 2 else "en"


def _tool_entry(f: dict[str, Any]) -> dict[str, Any]:
    """One function description → {"type": "function", "function": {...}} with JSON-schema types."""
    f = f.get("function", f)
    out = {"name": f["name"], "description": f.get("description", "")}
    params = _bfcl_schema(f.get("parameters") or {"type": "object", "properties": {}})
    params.setdefault("type", "object")
    params.setdefault("properties", {})
    out["parameters"] = params
    return {"type": "function", "function": out}


def _loads_loose(s: str) -> Any:
    """JSON, or a Python literal (some data sets write dicts with single quotes and True/False)."""
    import ast

    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return ast.literal_eval(s)


_SHAREGPT_ROLE = {
    "system": "system",
    "human": "user",
    "user": "user",
    "gpt": "assistant",
    "assistant": "assistant",
    "tool": "tool",
}
_HERMES_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)
_HERMES_TOOLS_RE = re.compile(r"<tools>\s*(.*?)\s*</tools>", re.S)


# What a broken JSON / Python literal in a data set can raise
_PARSE_ERRORS = (
    ValueError,
    TypeError,
    KeyError,
    AttributeError,
    SyntaxError,
    MemoryError,
    RecursionError,
)


def _hermes_calls(text: str) -> list[dict[str, Any]] | None:
    """The <tool_call> bodies of one Hermes turn → calls. None if a body does not parse."""
    calls = []
    for body in _HERMES_CALL_RE.findall(text):
        try:
            obj = _loads_loose(body)
            args = obj.get("arguments") or {}
            if isinstance(args, str):
                args = _loads_loose(args)
        except _PARSE_ERRORS:
            return None
        if not isinstance(obj.get("name"), str) or not isinstance(args, dict):
            return None
        calls.append({"name": obj["name"], "arguments": args})
    return calls


def from_hermes(row: dict[str, Any], idx: int, license: str = "Apache-2.0") -> list[FCTask]:
    """NousResearch hermes-function-calling-v1 (ShareGPT; calls are <tool_call> JSON or Python dicts).

    One task per assistant turn: the prompt is the history before it (without the data set's own system
    prompt: our template writes the tools section itself), the gold calls are the calls of that turn
    (none: a "no call" task). Earlier calls and tool results stay in the history.
    A row whose tool list does not parse gives no task. A turn whose calls do not parse ends the
    conversation: the tasks before it stay, the later turns would have a broken history.
    """
    conv = row.get("conversations") or []
    try:
        tools_raw = row.get("tools")
        if isinstance(tools_raw, str) and tools_raw.strip():
            tools_raw = _loads_loose(tools_raw)
        if not tools_raw:
            sys_text = next((c["value"] for c in conv if c.get("from") == "system"), "")
            m = _HERMES_TOOLS_RE.search(sys_text)
            tools_raw = _loads_loose(m.group(1)) if m else []
        tools = [_tool_entry(t) for t in tools_raw]
    except _PARSE_ERRORS:
        return []
    out, hist = [], []
    for c in conv:
        role = _SHAREGPT_ROLE.get(c.get("from", ""), "")
        text = c.get("value", "")
        if role == "system" or not role:
            continue
        if role == "assistant":
            calls = _hermes_calls(text)
            if calls is None:
                break
            if hist and hist[-1]["role"] in ("user", "tool"):
                out.append(
                    FCTask(
                        id=f"hermes-{row.get('id', idx)}-{len(out)}",
                        tools=tools,
                        messages=[dict(m) for m in hist],
                        gold_calls=calls,
                        source="hermes",
                        license=license,
                        lang=guess_lang(hist[-1].get("content", "")),
                    )
                )
            msg: dict[str, Any] = {"role": "assistant", "content": "" if calls else text}
            if calls:
                msg["tool_calls"] = calls
            hist.append(msg)
        elif role == "tool":
            body = re.sub(r"</?tool_response>", "", text).strip()
            hist.append({"role": "tool", "content": body})
        else:
            hist.append({"role": "user", "content": text})
    return out


def _looks_like_call_list(text: str) -> bool:
    t = text.strip()
    return t.startswith("[") and t.endswith("]") and "(" in t


def parse_python_calls(text: str) -> list[dict[str, Any]] | None:
    """`[Func Name(a="x", b=1), other(c=[1, 2])]` → calls. None if the text is not such a list.

    The function names of ToolACE can contain spaces, so the names are cut by hand; the arguments are
    parsed with the Python AST (literals only, no evaluation of code).
    """
    import ast

    if not _looks_like_call_list(text):
        return None
    t = text.strip()
    t = t[1:-1].strip()
    calls, depth, quote, start = [], 0, "", 0
    parts = []
    for i, ch in enumerate(t):
        if quote:
            if ch == quote and t[i - 1] != "\\":
                quote = ""
        elif ch in "\"'":
            quote = ch
        elif ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        elif ch == "," and depth == 0:
            parts.append(t[start:i])
            start = i + 1
    parts.append(t[start:])
    for p in parts:
        p = p.strip()
        k = p.find("(")
        if k <= 0 or not p.endswith(")"):
            return None
        name = p[:k].strip()
        try:
            node = ast.parse(f"f({p[k + 1 : -1]})", mode="eval").body
            if not isinstance(node, ast.Call) or node.args:
                return None
            args = {kw.arg: ast.literal_eval(kw.value) for kw in node.keywords if kw.arg}
        except _PARSE_ERRORS:
            return None
        calls.append({"name": name, "arguments": args})
    return calls


def from_toolace(row: dict[str, Any], idx: int, license: str = "Apache-2.0") -> list[FCTask]:
    """Team-ACE/ToolACE (system prompt with a JSON function list; calls as `[Func(arg=...)]`).

    One task per assistant turn, as in `from_hermes`. An assistant turn that is not a call list is a
    "no call" task (ToolACE has turns that must point out missing parameters instead of calling).
    A turn that looks like a call list but does not parse is not a "no call" task: it ends the
    conversation (the tasks before it stay). A row whose function list does not parse gives no task.
    """
    system = row.get("system", "")
    k = system.find("[", system.find("invoke:") + 1) if "invoke:" in system else -1
    if k < 0:
        return []
    try:
        # the JSON list ends before ". Should you ..."
        funcs, _ = json.JSONDecoder().raw_decode(system[k:])
        tools = [_tool_entry(f) for f in funcs]
    except _PARSE_ERRORS:
        return []
    out, hist = [], []
    for c in row.get("conversations") or []:
        role = _SHAREGPT_ROLE.get(c.get("from", ""), "")
        text = c.get("value", "")
        if role == "assistant":
            calls = parse_python_calls(text)
            if calls is None and _looks_like_call_list(text):
                break
            if hist and hist[-1]["role"] in ("user", "tool"):
                out.append(
                    FCTask(
                        id=f"toolace-{idx}-{len(out)}",
                        tools=tools,
                        messages=[dict(x) for x in hist],
                        gold_calls=calls or [],
                        source="toolace",
                        license=license,
                        lang=guess_lang(hist[-1].get("content", "")),
                    )
                )
            msg: dict[str, Any] = {"role": "assistant", "content": "" if calls else text}
            if calls:
                msg["tool_calls"] = calls
            hist.append(msg)
        elif role in ("user", "tool"):
            hist.append({"role": role, "content": text})
    return out


ACEBENCH_SCORED = (
    "normal_atom_bool",
    "normal_atom_enum",
    "normal_atom_list",
    "normal_atom_number",
    "normal_atom_object_deep",
    "normal_atom_object_short",
    "normal_multi_turn_user_adjust",
    "normal_multi_turn_user_switch",
    "normal_preference",
    "normal_similar_api",
    "normal_single_turn_parallel_function",
    "normal_single_turn_single_function",
    "special_error_param",
    "special_incomplete",
    "special_irrelevant",
)  # E2 of eval/PREREGISTRATION.md: Normal + Special (the Agent categories need a simulated user)

_ACE_TURN_RE = re.compile(r"^(user|system):\s?", re.M)


def _ace_messages(text: str) -> list[dict[str, Any]]:
    """ACEBench writes the conversation as text: "user: ...\nsystem: ...". "system" is the assistant."""
    parts = _ACE_TURN_RE.split(text)
    msgs = []
    for role, content in zip(parts[1::2], parts[2::2]):
        msgs.append({"role": "user" if role == "user" else "assistant", "content": content.strip()})
    return msgs


def from_acebench(question: dict[str, Any], answer: dict[str, Any] | None) -> FCTask:
    """One ACEBench item (Normal or Special) → FCTask. **For decontamination and local dev checks only.**

    - `time` and `profile` go into a system message (the official prompt also gives them to the model).
    - Normal: `ground_truth` is {func: args} (or a list with one such dict). Repeated calls of one
      function are written as func_1, func_2: the suffix is removed when only the base name is a tool.
    - Special (incomplete / error_param / irrelevant): the correct behavior is to make **no** call. The
      official scorer also checks that the reply names the missing or wrong parameter; this local
      approximation checks only "no call".
    """
    tools = [_tool_entry(f) for f in question["function"]]
    names = {t["function"]["name"] for t in tools}
    system = []
    if question.get("time"):
        system.append(question["time"])
    if question.get("profile"):
        system.append("用户画像：" + json.dumps(question["profile"], ensure_ascii=False))
    msgs = ([{"role": "system", "content": "\n".join(system)}] if system else []) + _ace_messages(
        question["question"]
    )
    gold = []
    category = re.sub(r"_\d+(_\d+)?$", "", question["id"])
    gt = (answer or {}).get("ground_truth")
    if not category.startswith("special") and gt:
        if isinstance(gt, list):
            gt = gt[0]
        for name, args in gt.items():
            if name not in names:
                base = re.sub(r"_\d+$", "", name)
                name = base if base in names else name
            gold.append({"name": name, "arguments": args})
    return FCTask(
        id=f"acebench-{question['id']}",
        tools=tools,
        messages=msgs,
        gold_calls=gold,
        source="acebench",
        license="MIT",
        lang=guess_lang(question["question"]),
    )


def read_acebench_dir(
    root: str | Path, categories: Sequence[str] = ACEBENCH_SCORED
) -> list[FCTask]:
    """data_all/data_zh (or data_en) of the ACEBench repository → tasks of the given categories."""
    root = Path(root)
    out = []
    for c in categories:
        q = root / f"data_{c}.json"
        if not q.exists():
            continue
        answers = {}
        a = root / "possible_answer" / q.name
        if a.exists():
            for line in a.read_text("utf-8").splitlines():
                if line.strip():
                    r = json.loads(line)
                    answers[r["id"]] = r
        for line in q.read_text("utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                out.append(from_acebench(row, answers.get(row["id"])))
    return out


def _read_rows(path: str | Path) -> Iterable[dict[str, Any]]:
    text = Path(path).read_text("utf-8")
    if text.lstrip().startswith("["):
        yield from json.loads(text)
    else:
        for line in text.splitlines():
            if line.strip():
                yield json.loads(line)


CONVERTERS = {
    "xlam": lambda path, lic: (
        from_xlam(r, i, lic or "CC-BY-4.0") for i, r in enumerate(_read_rows(path))
    ),
    "hermes": lambda path, lic: (
        t for i, r in enumerate(_read_rows(path)) for t in from_hermes(r, i, lic or "Apache-2.0")
    ),
    "toolace": lambda path, lic: (
        t for i, r in enumerate(_read_rows(path)) for t in from_toolace(r, i, lic or "Apache-2.0")
    ),
    "openai": lambda path, lic: (
        from_openai_messages(r, i, Path(path).stem, lic) for i, r in enumerate(_read_rows(path))
    ),
    "fc": lambda path, lic: (FCTask.from_dict(r) for r in _read_rows(path)),
}


# ---------------------------------------------------------------------------
# Decontamination and build
# ---------------------------------------------------------------------------


def tool_names(t: FCTask) -> set[str]:
    return {(x.get("function") or x).get("name", "") for x in t.tools}


def decontaminate_tasks(
    tasks: Sequence[FCTask], eval_tasks: Sequence[FCTask], n: int = 13
) -> tuple[list[FCTask], Counter]:
    """Drop training tasks that share a tool name or a user-text n-gram with the evaluation tasks."""
    from zero.data.decontam import NgramIndex

    names = set().union(*(tool_names(t) for t in eval_tasks)) if eval_tasks else set()
    index = NgramIndex(n)
    index.add_eval_set("eval", [t.query for t in eval_tasks])
    kept, dropped = [], Counter()
    for t in tasks:
        if tool_names(t) & names:
            dropped["tool_name"] += 1
        elif index.check(t.query):
            dropped["ngram"] += 1
        else:
            kept.append(t)
    return kept, dropped


def build(
    sources: Sequence[tuple[str, str, str]],
    out: str | Path,
    exclude: Sequence[FCTask] = (),
    dev_out: str | Path | None = None,
    dev_size: int = 0,
    seed: int = 0,
    max_tools: int = 32,
) -> dict[str, Any]:
    """sources: (kind, path, license) → validated, deduplicated, decontaminated task file(s)."""
    tasks: list[FCTask] = []
    dropped: Counter = Counter()
    per_source: Counter = Counter()
    licenses: dict[str, str] = {}
    seen: set[str] = set()
    for kind, path, lic in sources:
        if kind not in CONVERTERS:
            raise ValueError(f"Unknown source kind {kind!r}; known: {sorted(CONVERTERS)}")
        for t in CONVERTERS[kind](path, lic):
            if t is None:
                dropped["no_assistant_turn"] += 1
                continue
            if task_errors(t):
                dropped["invalid"] += 1
                continue
            if len(t.tools) > max_tools:
                dropped["too_many_tools"] += 1
                continue
            key = json.dumps(
                [t.messages, sorted(tool_names(t))], sort_keys=True, ensure_ascii=False
            )
            if key in seen:
                dropped["duplicate"] += 1
                continue
            seen.add(key)
            tasks.append(t)
            per_source[t.source] += 1
            licenses[t.source] = t.license
    if exclude:
        tasks, dc = decontaminate_tasks(tasks, exclude)
        for k, v in dc.items():
            dropped[f"decontam_{k}"] += v
    rng = random.Random(seed)
    rng.shuffle(tasks)
    dev = tasks[:dev_size] if dev_out and dev_size else []
    train = tasks[len(dev) :]

    def write(path: str | Path, rows: Sequence[FCTask]) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            for t in rows:
                f.write(json.dumps(t.to_dict(), ensure_ascii=False) + "\n")

    write(out, train)
    if dev:
        write(dev_out, dev)  # type: ignore[arg-type]
    meta = {
        "n_train": len(train),
        "n_dev": len(dev),
        "per_source": dict(per_source),
        "licenses": licenses,
        "dropped": dict(dropped),
        "n_irrelevance": sum(1 for t in train if not t.gold_calls),
        "n_parallel": sum(1 for t in train if len(t.gold_calls) > 1),
        "n_exclude_tasks": len(exclude),
        "sources": [list(s) for s in sources],
    }
    Path(str(out) + ".meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    return meta


def to_sft(task: FCTask) -> dict[str, Any] | None:
    """A task with gold calls → one SFT conversation that ends with the gold call turn.

    "No call" tasks are skipped: their reference reply is not in the task.
    """
    if not task.gold_calls:
        return None
    calls = [{"name": g["name"], "arguments": g["arguments"]} for g in task.gold_calls]
    msgs = [*task.messages, {"role": "assistant", "content": "", "tool_calls": calls}]
    return {
        "messages": msgs,
        "tools": task.tools,
        "task_id": task.id,
        "source": task.source,
        "license": task.license,
    }


def split_sft_rl(
    tasks: Sequence[FCTask], sft_frac: float, seed: int = 0
) -> tuple[list[dict[str, Any]], list[FCTask]]:
    """Split the tasks into SFT conversations and RL tasks that do not overlap.

    RL on tasks that SFT already showed with the answer gives little signal (the group is all correct).
    A "no call" task drawn for SFT has no reference reply (`to_sft` skips it): it goes to RL, where the
    verifier scores it, instead of being lost.
    """
    idx = list(range(len(tasks)))
    random.Random(seed).shuffle(idx)
    k = int(len(idx) * sft_frac)
    sft, rl = [], []
    for n, i in enumerate(idx):
        r = to_sft(tasks[i]) if n < k else None
        if r is not None:
            sft.append(r)
        else:
            rl.append(i)
    return sft, [tasks[i] for i in sorted(rl)]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Function-calling task files (Chapter 19)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="convert, validate, deduplicate, decontaminate")
    b.add_argument("--src", action="append", required=True, metavar="KIND:PATH[:LICENSE]")
    b.add_argument("--exclude-bfcl", action="append", default=[], metavar="DIR")
    b.add_argument("--exclude-acebench", action="append", default=[], metavar="DIR")
    b.add_argument("--exclude-tasks", action="append", default=[], metavar="JSONL")
    b.add_argument("--out", required=True)
    b.add_argument("--dev-out")
    b.add_argument("--dev-size", type=int, default=0)
    b.add_argument("--seed", type=int, default=0)
    sp = sub.add_parser("split", help="split a task file into SFT conversations and RL tasks")
    sp.add_argument("tasks")
    sp.add_argument("--sft-out", required=True)
    sp.add_argument("--rl-out", required=True)
    sp.add_argument("--sft-frac", type=float, default=0.5)
    sp.add_argument("--seed", type=int, default=0)
    ex = sub.add_parser(
        "export",
        help="evaluation data (BFCL / ACEBench) → a task file (dev checks, decontamination)",
    )
    ex.add_argument("kind", choices=["bfcl", "acebench"])
    ex.add_argument("root", help="BFCL: <bfcl_eval>/data; ACEBench: data_all/data_zh or data_en")
    ex.add_argument("--out", required=True)
    s = sub.add_parser("score", help="score one output on one task")
    s.add_argument("tasks")
    s.add_argument("index", type=int)
    s.add_argument("text")
    args = ap.parse_args(argv)
    if args.cmd == "build":
        srcs = []
        for spec in args.src:
            kind, _, rest = spec.partition(":")
            if ":" in rest and not Path(rest).exists():
                path, _, lic = rest.rpartition(":")
            else:
                path, lic = rest, ""
            srcs.append((kind, path, lic))
        excl: list[FCTask] = []
        for d in args.exclude_bfcl:
            excl += read_bfcl_dir(d)
        for d in args.exclude_acebench:
            excl += read_acebench_dir(d)
        for p in args.exclude_tasks:
            excl += load_fc_tasks(p, check_schema=False)
        meta = build(srcs, args.out, excl, args.dev_out, args.dev_size, args.seed)
        print(json.dumps(meta, ensure_ascii=False, indent=2))
    elif args.cmd == "export":
        tasks = read_bfcl_dir(args.root) if args.kind == "bfcl" else read_acebench_dir(args.root)
        p = Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            for t in tasks:
                f.write(json.dumps(t.to_dict(), ensure_ascii=False) + "\n")
        print(json.dumps({"tasks": len(tasks), "out": str(p)}))
    elif args.cmd == "split":
        sft, rl = split_sft_rl(load_fc_tasks(args.tasks), args.sft_frac, args.seed)
        for path, rows in ((args.sft_out, sft), (args.rl_out, [t.to_dict() for t in rl])):
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(json.dumps({"sft": len(sft), "rl": len(rl)}))
    else:
        t = load_fc_tasks(args.tasks, strict=True)[args.index]
        r = score_fc(t, args.text)
        print(json.dumps(asdict(r), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
