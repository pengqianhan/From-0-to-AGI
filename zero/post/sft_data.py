"""SFT data pipeline: sources → one cleaned, licensed, decontaminated mixture (Chapter 16).

    uv run python -m zero.post.sft_data --config configs/main/sft_data.toml

The output is what `zero.post.sft` reads (`data/sft/train.jsonl`, `val.jsonl`): one conversation per
line, `{"messages": [...], "tools": [...], "id", "source", "license", "lang"}`. The rows are text, so the
three tracks of `runs/POSTTRAIN_PLAN.md` share them; only the packing depends on the tokenizer.

**Source formats** (`[[sources]] format`):

| format | Row | Examples |
|---|---|---|
| `messages` | `{"messages": [{"role", "content"}...], "tools"?}`; SmolTalk2's `chat_template_kwargs` (`xml_tools`, `custom_instructions`) is read too | Tülu 3, SmolTalk, SmolTalk2 |
| `sharegpt` | `{"conversations": [{"from", "value"}...], "system"?}` | OpenHermes, many Chinese sets |
| `alpaca` | `{"instruction", "input", "output"}` | many Chinese sets |
| `sft` | this format (a row `license` overrides the source license) | `fc_tasks split --sft-out`, our own data |
| `tool_env` | no file: `max_rows` reference trajectories of the toy tool environment | our own synthetic data |

Tool calls written as text inside an assistant turn (`<tool_call>{...}</tool_call>`) become structured
`tool_calls`; a turn that does not parse is dropped, because our template would render it differently.

**License per row** (GOAL.md 3.3): mixtures such as Tülu 3 and SmolTalk2 tag each row with its upstream
`source`, and the subsets have different licenses. With `source_field`, the license of a row is
`license_by_source[row[source_field]]`, else the source license. Rows from `drop_sources` are dropped.
A row is kept only if its license is not non-commercial (`NC` in the name) and is not "待核实" (to be
verified; the same marker as `zero/data/sources.py`), unless `allow_unverified = true`. `own` marks data
that this project made.

**Cleaning**, each with a drop counter in `meta.json`:
- `no_assistant`: no assistant turn, or the last assistant turn is empty.
- `special_tokens`: `<|im_start|>`, `<|im_end|>`, or `<|endoftext|>` in any text (they would forge turns).
- `bad_tool_call`: tool-call text that does not parse.
- `thinking`: with `drop_thinking`, `<think>…</think>` is removed from the assistant turns (the main-line
  model has no thinking mode by default); a turn that becomes empty drops the row.
- `other_language`: with `langs`, a row in another language (the guess knows only zh / en: a French row
  counts as "en", so `langs = ["zh"]` keeps the Chinese part of a multilingual set).
- `too_long`: more than `max_tokens` tokens with `tokenizer` (the SFT packing drops them anyway, but
  here they are counted per source).
- `duplicate` / `near_duplicate`: the same conversation; MinHash on the first user message (Chapter 13).
- `decontam_ngram` / `decontam_tool_name`: a user text that shares a 13-gram with the evaluation texts, or
  a tool name of BFCL (GOAL.md 3.2).

**Mixing**: after the filters, each source is shuffled and cut to `max_rows` (0: all), then the validation
set is taken from the pool (`val_size`, before any repetition), then the training rows of a source are
repeated `repeat` times. `meta.json` reports the rows and (with a tokenizer) the tokens and assistant
tokens of each source and language, so the mixture is checked by tokens, not by rows.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import random
import re
from collections import Counter
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

UNVERIFIED = "待核实"  # same marker as zero/data/sources.py and zero/data/download.py
FORBIDDEN = ("<|im_start|>", "<|im_end|>", "<|endoftext|>")
_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.S)


@dataclass
class SFTSource:
    name: str = ""
    path: str = ""  # file or glob (.jsonl / .json / .parquet); not used by format "tool_env"
    format: str = "messages"  # messages | sharegpt | alpaca | sft | tool_env
    license: str = UNVERIFIED
    lang: str = "auto"  # "auto": guess from the text (zh if it has Chinese characters)
    max_rows: int = 0  # after filtering; 0 = all
    repeat: int = 1  # training rows are repeated this many times (upsampling)
    source_field: str = (
        ""  # the row field with the upstream subset name (Tülu 3, SmolTalk2: "source")
    )
    license_by_source: dict[str, str] = field(default_factory=dict)
    drop_sources: list[str] = field(default_factory=list)
    langs: list[str] = field(
        default_factory=list
    )  # keep only these languages ("zh" / "en"); empty: all
    notes: str = ""


@dataclass
class SFTDataConfig:
    out_dir: str = "data/sft"
    val_size: int = 2000
    seed: int = 0
    tokenizer: str = ""  # for token counts and max_tokens; empty: no token counts
    max_tokens: int = 0  # 0: no limit
    drop_thinking: bool = True
    allow_unverified: bool = False
    near_dedup: bool = True
    near_threshold: float = 0.85
    decontam_texts: list[str] = field(
        default_factory=list
    )  # globs of evaluation prompts (.jsonl / .txt)
    decontam_bfcl: list[str] = field(
        default_factory=list
    )  # BFCL data folders (tool names + questions)
    decontam_n: int = 13
    sources: list[SFTSource] = field(default_factory=list)


def load_sft_data_config(path: str | Path, overrides: Sequence[str] | None = None) -> SFTDataConfig:
    from zero.config import _apply_override, _from_dict, _read_toml_with_base

    data = _read_toml_with_base(Path(path))
    for o in overrides or []:
        _apply_override(data, o)
    return _from_dict(
        SFTDataConfig, {**data.get("build", {}), "sources": data.get("sources", [])}, "[build]"
    )


# ---------------------------------------------------------------------------
# Reading and converting rows
# ---------------------------------------------------------------------------


def read_rows(pattern: str) -> Iterator[dict[str, Any]]:
    paths = sorted(glob.glob(pattern)) or ([pattern] if Path(pattern).exists() else [])
    if not paths:
        raise FileNotFoundError(f"No file matches {pattern}")
    for p in paths:
        if p.endswith(".parquet"):
            try:
                import pyarrow.parquet as pq
            except ImportError as e:  # pragma: no cover
                raise ImportError("Reading .parquet needs pyarrow: uv sync --extra data") from e
            yield from pq.read_table(p).to_pylist()
            continue
        text = Path(p).read_text("utf-8")
        if text.lstrip().startswith("["):
            yield from json.loads(text)
        else:
            for line in text.splitlines():
                if line.strip():
                    yield json.loads(line)


_ROLE = {
    "system": "system",
    "human": "user",
    "user": "user",
    "gpt": "assistant",
    "assistant": "assistant",
    "tool": "tool",
    "function": "tool",
    "observation": "tool",
}


def _maybe_json(x: Any) -> Any:
    if isinstance(x, str) and x.strip()[:1] in "[{":
        try:
            return json.loads(x)
        except json.JSONDecodeError:
            return x
    return x


def _tools_of(row: dict[str, Any]) -> list[dict[str, Any]] | None:
    tools = _maybe_json(row.get("tools"))
    kw = row.get("chat_template_kwargs") or {}
    if not tools and kw.get("xml_tools"):
        tools = [_maybe_json(t) for t in kw["xml_tools"]]
    if not tools:
        return None
    return [
        t if "function" in t else {"type": "function", "function": t}
        for t in tools
        if isinstance(t, dict)
    ]


def to_messages(
    row: dict[str, Any], fmt: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]] | None]:
    """One source row → (messages, tools), not yet cleaned."""
    if fmt in ("messages", "sft"):
        msgs = [dict(m) for m in row.get("messages") or []]
        kw = row.get("chat_template_kwargs") or {}
        if kw.get("custom_instructions") and not any(m.get("role") == "system" for m in msgs):
            msgs.insert(0, {"role": "system", "content": kw["custom_instructions"]})
        return msgs, _tools_of(row)
    if fmt == "sharegpt":
        msgs = []
        if row.get("system"):
            msgs.append({"role": "system", "content": row["system"]})
        for c in row.get("conversations") or []:
            role = _ROLE.get(c.get("from", ""))
            if role is None:
                raise ValueError(f"unknown ShareGPT role {c.get('from')!r}")
            msgs.append({"role": role, "content": c.get("value", "")})
        return msgs, _tools_of(row)
    if fmt == "alpaca":
        prompt = row.get("instruction", "")
        if row.get("input"):
            prompt = f"{prompt}\n\n{row['input']}" if prompt else row["input"]
        return [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": row.get("output", "")},
        ], None
    raise ValueError(f"Unknown source format {fmt!r}")


def clean_messages(
    msgs: list[dict[str, Any]], drop_thinking: bool
) -> tuple[list[dict[str, Any]] | None, str]:
    """Normalize one conversation. Returns (messages, "") or (None, drop reason)."""
    from zero.post.chat import parse_assistant

    out = []
    for m in msgs:
        role = m.get("role")
        content = m.get("content") or ""
        if not isinstance(content, str):
            return None, "bad_content"
        if any(t in content for t in FORBIDDEN):
            return None, "special_tokens"
        msg: dict[str, Any] = {"role": role, "content": content}
        if role == "assistant":
            had_think = "<think>" in content
            if drop_thinking and had_think:
                content = _THINK_RE.sub("", content)
                if "<think>" in content:  # not closed
                    return None, "thinking"
            calls = list(m.get("tool_calls") or [])
            if "<tool_call>" in content or "</tool_call>" in content:
                parsed = parse_assistant(content)
                if parsed.errors:
                    return None, "bad_tool_call"
                content, calls = parsed.content, calls + parsed.tool_calls
            norm_calls = []
            for tc in calls:
                f = tc.get("function", tc)
                args = _maybe_json(f.get("arguments", {}))
                if not isinstance(args, dict) or not f.get("name"):
                    return None, "bad_tool_call"
                norm_calls.append({"name": f["name"], "arguments": args})
            msg = {"role": "assistant", "content": content.strip() if norm_calls else content}
            if norm_calls:
                msg["tool_calls"] = norm_calls
            if not msg["content"].strip() and not norm_calls:
                return None, "thinking" if had_think else "no_assistant"
        elif role == "tool":
            msg["content"] = re.sub(r"</?tool_response>", "", content).strip()
        elif role not in ("system", "user"):
            return None, "bad_role"
        out.append(msg)
    if not any(m["role"] == "assistant" for m in out) or out[-1]["role"] != "assistant":
        return None, "no_assistant"
    if not any(m["role"] == "user" for m in out):
        return None, "no_assistant"
    return out, ""


def row_license(row: dict[str, Any], src: SFTSource) -> tuple[str | None, str]:
    """(license, "") or (None, drop reason)."""
    sub = str(row.get(src.source_field, "")) if src.source_field else ""
    if sub and sub in src.drop_sources:
        return None, "excluded_source"
    lic = (
        (src.license_by_source.get(sub) if sub else None)
        or (row.get("license") if src.format == "sft" else None)
        or src.license
    )
    return lic, ""


def license_ok(lic: str, allow_unverified: bool) -> str:
    """ "" if allowed, else the drop reason."""
    if re.search(r"(^|[-_ ])NC([-_ ]|$)", lic.upper()):
        return "license_nc"
    if lic == UNVERIFIED and not allow_unverified:
        return "license_unverified"
    return ""


def guess_lang(msgs: Sequence[dict[str, Any]]) -> str:
    from zero.post.envs.fc_tasks import guess_lang as g

    return g(" ".join((m.get("content") or "")[:500] for m in msgs[:3]))


def user_texts(msgs: Sequence[dict[str, Any]]) -> list[str]:
    return [m["content"] for m in msgs if m["role"] == "user" and m["content"]]


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


def _eval_texts(globs: Sequence[str]) -> list[str]:
    texts = []
    for g in globs:
        for p in sorted(glob.glob(g)):
            if p.endswith(".txt"):
                texts += [x for x in Path(p).read_text("utf-8").splitlines() if x.strip()]
                continue
            for r in read_rows(p):
                for k in ("question", "prompt", "query", "text", "input", "instruction"):
                    if isinstance(r.get(k), str):
                        texts.append(r[k])
                        break
    return texts


def iter_source(src: SFTSource, cfg: SFTDataConfig, drops: Counter) -> Iterator[dict[str, Any]]:
    """Converted, cleaned, licensed rows of one source (deduplication and decontamination come later)."""
    if src.format == "tool_env":
        from zero.post.sft import env_conversations

        for i, r in enumerate(env_conversations(src.max_rows or 1000, cfg.seed, "train")):
            yield {
                "messages": r["messages"],
                "tools": r["tools"],
                "id": f"{src.name}-{i}",
                "source": src.name,
                "license": src.license,
                "lang": guess_lang(r["messages"]),
            }
        return
    rows: Iterable[dict[str, Any]] = read_rows(src.path)
    for i, row in enumerate(rows):
        drops["read"] += 1
        lic, why = row_license(row, src)
        why = why or license_ok(lic or "", cfg.allow_unverified)
        if why:
            drops[why] += 1
            continue
        try:
            msgs, tools = to_messages(row, src.format)
        except (ValueError, KeyError, TypeError):
            drops["bad_format"] += 1
            continue
        msgs, why = clean_messages(msgs, cfg.drop_thinking)
        if msgs is None:
            drops[why] += 1
            continue
        lang = guess_lang(msgs) if src.lang == "auto" else src.lang
        if src.langs and lang not in src.langs:
            drops["other_language"] += 1
            continue
        sub = str(row.get(src.source_field, "")) if src.source_field else ""
        yield {
            "messages": msgs,
            **({"tools": tools} if tools else {}),
            "id": f"{src.name}-{row.get('id', i)}",
            "source": src.name + (f"/{sub}" if sub else ""),
            "license": lic,
            "lang": lang,
        }


def _key(r: dict[str, Any]) -> str:
    return hashlib.sha1(
        json.dumps([r["messages"], r.get("tools")], sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def build_sft_data(cfg: SFTDataConfig, log: Any = print) -> dict[str, Any]:
    rng = random.Random(cfg.seed)
    tok = None
    if cfg.tokenizer:
        from zero.tokenizer import Tokenizer

        tok = Tokenizer.load(cfg.tokenizer)
    index = None
    eval_tool_names: set[str] = set()
    if cfg.decontam_texts or cfg.decontam_bfcl:
        from zero.data.decontam import NgramIndex

        texts = _eval_texts(cfg.decontam_texts)
        if cfg.decontam_bfcl:
            from zero.post.envs.fc_tasks import read_bfcl_dir, tool_names

            for d in cfg.decontam_bfcl:
                for t in read_bfcl_dir(d):
                    texts.append(t.query)
                    eval_tool_names |= tool_names(t)
        index = NgramIndex(cfg.decontam_n)
        index.add_eval_set("eval", texts)
        log(
            f"[sft_data] decontamination: {len(texts)} evaluation texts, {len(eval_tool_names)} tool names"
        )

    seen: set[str] = set()
    per: dict[str, dict[str, Any]] = {}
    pools: dict[str, list[dict[str, Any]]] = {}
    for src in cfg.sources:
        drops: Counter = Counter()
        kept = []
        for r in iter_source(src, cfg, drops):
            k = _key(r)
            if k in seen:
                drops["duplicate"] += 1
                continue
            if index is not None:
                if r.get("tools") and eval_tool_names & {
                    (t.get("function") or t).get("name") for t in r["tools"]
                }:
                    drops["decontam_tool_name"] += 1
                    continue
                if any(index.check(u) for u in user_texts(r["messages"])):
                    drops["decontam_ngram"] += 1
                    continue
            seen.add(k)
            kept.append(r)
        if cfg.near_dedup and len(kept) > 1:
            from zero.data.dedup import near_dedup

            firsts = [user_texts(r["messages"])[0] for r in kept]
            keep, _ = near_dedup(firsts, threshold=cfg.near_threshold)
            keep_idx = set(keep)
            drops["near_duplicate"] += len(kept) - len(keep_idx)
            kept = [r for i, r in enumerate(kept) if i in keep_idx]
        rng.shuffle(kept)
        if src.max_rows and src.format != "tool_env":
            drops["over_max_rows"] += max(0, len(kept) - src.max_rows)
            kept = kept[: src.max_rows]
        pools[src.name] = kept
        per[src.name] = {
            "license": src.license,
            "kept": len(kept),
            "dropped": dict(drops),
            "repeat": src.repeat,
        }
        if drops.get("license_unverified") and not kept:
            log(
                f"[sft_data] WARNING {src.name}: every row is dropped because the license is {UNVERIFIED}. Verify it and set license."
            )
        log(f"[sft_data] {src.name}: kept {len(kept)}, dropped {dict(drops)}")

    # Validation set: from the pool before repetition, so no validation row is ever trained on
    allrows = [(name, r) for name, rows in pools.items() for r in rows]
    rng.shuffle(allrows)
    val = [r for _, r in allrows[: cfg.val_size]]
    val_ids = {id(r) for r in val}
    train = []
    for name, rows in pools.items():
        rep = next(s.repeat for s in cfg.sources if s.name == name)
        train += [r for r in rows if id(r) not in val_ids] * max(rep, 1)
    rng.shuffle(train)

    stats = _stats(train, tok, cfg.max_tokens)
    if cfg.max_tokens and tok is not None:
        train = [r for r in train if not r.get("_too_long")]
        val = [r for r in val if _ntok(r, tok)[0] <= cfg.max_tokens]
    for r in train:
        r.pop("_too_long", None)
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train.jsonl", train), ("val.jsonl", val)):
        with open(out / name, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    meta = {
        "n_train": len(train),
        "n_val": len(val),
        "sources": per,
        "train_stats": stats,
        "licenses": sorted({r["license"] for r in train}),
        "config": asdict(cfg),
    }
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    log(f"[sft_data] train {len(train)}, val {len(val)} → {out}")
    return meta


def _ntok(r: dict[str, Any], tok: Any) -> tuple[int, int]:
    from zero.post.chat import render

    ids, mask = render(r["messages"], r.get("tools"), tokenizer=tok)
    return len(ids), sum(mask)


def _stats(rows: list[dict[str, Any]], tok: Any, max_tokens: int) -> dict[str, Any]:
    """Rows (and tokens, with a tokenizer) per source and per language; marks rows over max_tokens."""
    by_src: dict[str, Counter] = {}
    by_lang: dict[str, Counter] = {}
    cache: dict[int, tuple[int, int]] = {}  # repeated rows are the same object
    for r in rows:
        c = Counter(rows=1)
        if tok is not None:
            if id(r) not in cache:
                cache[id(r)] = _ntok(r, tok)
            n, a = cache[id(r)]
            if max_tokens and n > max_tokens:
                r["_too_long"] = True
                c = Counter(too_long=1)
            else:
                c.update(tokens=n, assistant_tokens=a)
        top = r["source"].split("/")[0]
        by_src.setdefault(top, Counter()).update(c)
        by_lang.setdefault(r.get("lang", ""), Counter()).update(c)
    total = sum(c["tokens"] for c in by_src.values()) or sum(c["rows"] for c in by_src.values())
    key = "tokens" if tok is not None else "rows"
    return {
        "by_source": {
            k: {**dict(v), "share": round(v[key] / max(total, 1), 4)} for k, v in by_src.items()
        },
        "by_lang": {
            k: {**dict(v), "share": round(v[key] / max(total, 1), 4)} for k, v in by_lang.items()
        },
        "share_by": key,
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Build the SFT data mixture (Chapter 16)")
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = ap.parse_args(argv)
    build_sft_data(load_sft_data_config(args.config, args.set))


if __name__ == "__main__":
    main()
