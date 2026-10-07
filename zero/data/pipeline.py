"""Pretraining data pipeline: an end-to-end recipe that one TOML file controls (Chapter 13).

    uv run python -m zero.data.pipeline --config configs/tiny/data.toml    # about half a minute on a CPU
    uv run python -m zero.data.pipeline --config configs/main/data.toml    # Step 2, not verified yet

Stages (each stage writes its output to disk as JSONL, so you can check and run each stage again separately):

    1. read        the inputs of [[sources]]: plain text (split at blank lines and joined into
                   documents) or the JSONL shards from download.py
    2. clean       clean.py: Unicode NFC, white space and control characters, remove literal special tokens
    3. language    rough check by script (fraction of Chinese characters): remove non-Chinese
                   documents from Chinese sources (Step 2 uses fastText lid.176 instead)
    4. quality     heuristic rules (web: Gopher + C4 + FineWeb; code: the three rules of Codex) + an
                   optional score threshold (the classifier score of the data set, for example
                   int_score of FineWeb-Edu) or a custom classifier
    5. dedup       exact hash + MinHash LSH (dedup.py), first in each source, then optionally once
                   across sources
    6. decontam    n-gram overlap check (default 13) with the evaluation sets (decontam.py);
                   remove each document with a hit
    7. split       for each source, take a validation set with a fixed seed
    8. tokenizer   sample text from the sources by mixture weight and train a byte-level BPE
                   (pre-tokenization regex: qwen2 / qwen3.5)
    9. shards      shard.py: <source>_train_*.bin / <source>_val_*.bin
   10. manifest    manifest.json: for each step and source, the documents/bytes that stay (the
                   funnel), the removal reasons, the decontamination hits, the tokenizer hash and
                   the compression of each source, the shard token counts, the mixture and "epochs
                   needed for the budget", the licenses and the source URLs

The tiny configuration runs the full pipeline on assets/tiny_corpus. The main configuration
describes the real recipe of Step 2 (configs/main/data.toml).
At the scale of Step 2 (billions of documents), the MinHash of step 5 must change to a distributed
implementation (for example MinhashDedup of datatrove). The single-process implementation here only
makes sure that the algorithm is correct (tests/test_data.py, tests/test_pipeline.py). It is **not
verified on large data yet**: it keeps all documents of a source in memory. Step 2 must stream the
input shards (stages 1–4 and 6 can stream document by document; step 5 needs a distributed MinHash).
"""

from __future__ import annotations

import argparse
import dataclasses
import glob
import gzip
import hashlib
import importlib
import json
import os
import random
import subprocess
import time
import tomllib
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tokenizers import Regex, pre_tokenizers, trainers

from zero.data.clean import clean_document, split_into_documents
from zero.data.decontam import NgramIndex
from zero.data.dedup import exact_dedup, near_dedup, text_hash
from zero.data.quality import QualityThresholds, cjk_ratio, quality_check
from zero.data.shard import write_shards
from zero.data.sources import SOURCES
from zero.tokenizer import (
    DEFAULT_SPECIAL_TOKENS,
    ENDOFTEXT,
    PRETOKENIZE_REGEX,
    Tokenizer,
    _build_empty_bpe,
    train_bpe,
)

# ---------------------------------------------------------------------------
# Tokenizer: two presets of the pre-tokenization regex
# ---------------------------------------------------------------------------

#: The regex in tokenizer.json of Qwen3.5 (read in 2026-09). It has one difference from Qwen2/Qwen3:
#: letter runs change from \p{L}+ to [\p{L}\p{M}]+, and punctuation runs also exclude \p{M}.
#: \p{M} is the "combining marks" (vowel signs of Devanagari and Thai, diacritics of Arabic, ...).
#: The old regex splits a word at these marks. The new regex keeps "letter + combining mark" in one piece.
QWEN35_PRETOKENIZE_REGEX = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}"
    r"| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)

PRETOKENIZE_PRESETS: dict[str, str] = {
    "qwen2": PRETOKENIZE_REGEX,  # default of zero/tokenizer.py, the same as Qwen2/Qwen3
    "qwen3": PRETOKENIZE_REGEX,
    "qwen3.5": QWEN35_PRETOKENIZE_REGEX,
}

#: The llama.cpp pre-tokenizer type name for the GGUF export (the pre_tokenizer parameter of zero/export/gguf.py)
GGUF_PRE_TOKENIZER = {"qwen2": "qwen2", "qwen3": "qwen2", "qwen3.5": "qwen35"}


def resolve_pretokenize(name_or_regex: str) -> str:
    """Preset name (qwen2 / qwen3 / qwen3.5) → regex. Any other string is used as the regex itself."""
    return PRETOKENIZE_PRESETS.get(name_or_regex, name_or_regex)


def train_tokenizer(
    texts_or_files: Iterable[str | os.PathLike],
    vocab_size: int,
    pretokenize: str = "qwen2",
    special_tokens: Sequence[str] | None = None,
    min_frequency: int = 2,
) -> Tokenizer:
    """Train a byte-level BPE, with a choice of pre-tokenization regex.

    The default (qwen2) calls `zero.tokenizer.train_bpe` directly, and the result is identical.
    With a different regex, only the pre-tokenization step changes. The normalization (NFC), the
    byte level, and the layout of the special tokens are the same as in `train_bpe`. So
    `Tokenizer.save_hf`, `bytes_per_token`, and `hash` work as usual (the regex is in
    tokenizer.json, and loading restores it).
    """
    regex = resolve_pretokenize(pretokenize)
    if regex == PRETOKENIZE_REGEX:
        return train_bpe(texts_or_files, vocab_size, special_tokens, min_frequency)
    special = list(special_tokens) if special_tokens is not None else list(DEFAULT_SPECIAL_TOKENS)
    if ENDOFTEXT not in special:
        special = [ENDOFTEXT, *special]
    if vocab_size < 256 + len(special):
        raise ValueError(f"vocab_size={vocab_size} is too small")
    tok = _build_empty_bpe()
    tok.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Split(Regex(regex), behavior="isolated", invert=False),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
        ]
    )
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )

    def iterate() -> Iterator[str]:
        for item in texts_or_files:
            if isinstance(item, os.PathLike):
                with open(item, encoding="utf-8") as f:
                    while chunk := f.read(1 << 20):
                        yield chunk
            else:
                yield item

    tok.train_from_iterator(iterate(), trainer=trainer)
    return Tokenizer(tok)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class SourceSpec:
    name: str  # shard name prefix (<name>_train_*.bin), also the key in the mixture
    inputs: list[str]  # list of globs: plain text files, or .jsonl / .jsonl.gz files from download.py
    registry: str = ""  # registry name in zero/data/sources.py (the license and the URL come from there)
    format: str = "text"  # "text" | "jsonl"
    text_field: str = "text"
    weight: float = 1.0  # pretraining mixture weight (normalized)
    lang: str = ""  # "zh" | "en" | "code" | "": rough language check
    heuristics: str = "web"  # "web" | "code" | "none"
    score_field: str = ""  # quality score field of the data set (for example int_score of FineWeb-Edu)
    score_min: float | None = None
    classifier: str = ""  # "module:ClassName"; the class must have score(texts) -> list[float]
    classifier_min: float = 3.0
    doc_chars: int = 4000  # plain text input: split at blank lines, then join into documents of about this length
    min_chars: int = 50
    max_docs: int | None = None  # read only the first N documents (for debugging)


@dataclass
class DedupSpec:
    exact: bool = True
    near: bool = True
    threshold: float = 0.8
    num_perm: int = 128
    bands: int = 16
    ngram: int = 5
    cross_source: bool = False  # after the dedup in each source, dedup once more across sources
    n_jobs: int = 1  # processes for the MinHash signatures (same result as 1, only faster; the 16.8GB ladder data takes about 5 hours in one process)


@dataclass
class EvalSetSpec:
    name: str
    path: str  # .jsonl (the strings in `fields`; default: all string fields) or .txt (one item per line)
    fields: list[str] = field(default_factory=list)


@dataclass
class DecontamSpec:
    n: int = 13
    min_tokens: int = 8  # skip items with fewer normalized "words" than this (substring matches of very short items give too many false hits)
    eval_sets: list[EvalSetSpec] = field(default_factory=list)


@dataclass
class TokenizerSpec:
    vocab_size: int = 2048
    pretokenize: str = "qwen2"  # "qwen2" | "qwen3" | "qwen3.5" | custom regex
    sample_bytes: int = 0  # bytes of text to train the tokenizer (sampled from the sources by mixture weight); 0 = all training text
    min_frequency: int = 2
    path: str = ""  # an existing tokenizer.json: if given, do not train; use it to make the shards


@dataclass
class PipelineSpec:
    name: str = "data"
    work_dir: str = "out/data"  # intermediate results (the JSONL of each stage)
    out_dir: str = ""  # shard output directory; default <work_dir>/shards
    seed: int = 0
    val_fraction: float = 0.05
    val_max_docs: int = 0  # maximum number of validation documents for each source (0 = no limit)
    shard_tokens: int = 100_000_000
    token_budget: float = 0.0  # pretraining token budget (the manifest computes "the number of passes over each source")
    write_stages: bool = True  # write the result of each stage to disk as JSONL (easy checks for tiny; with the large data of main, write only the last step)


@dataclass
class PipelineConfig:
    pipeline: PipelineSpec
    sources: list[SourceSpec]
    quality: QualityThresholds = field(default_factory=QualityThresholds)
    dedup: DedupSpec = field(default_factory=DedupSpec)
    decontam: DecontamSpec = field(default_factory=DecontamSpec)
    tokenizer: TokenizerSpec = field(default_factory=TokenizerSpec)
    raw: dict[str, Any] = field(default_factory=dict)  # the original TOML (written into the manifest)


def _build(cls: type, d: dict[str, Any], where: str) -> Any:
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = set(d) - names
    if unknown:
        raise ValueError(f"[{where}] unknown fields: {sorted(unknown)}")
    return cls(**d)


def config_from_dict(d: dict[str, Any]) -> PipelineConfig:
    """TOML dict → PipelineConfig. Only these sections are read: [pipeline] [[sources]] [quality] [dedup] [decontam] [tokenizer].

    download.py reads the other sections of the main configuration, such as [download]. This function ignores them.
    """
    known = {"pipeline", "sources", "quality", "dedup", "decontam", "tokenizer"}
    if not d.get("sources"):
        raise ValueError("At least one [[sources]] is necessary")
    sources = []
    for s in d["sources"]:
        s = {k: v for k, v in s.items() if k != "download"}  # [sources.download] is for download.py
        sources.append(_build(SourceSpec, s, "sources"))
    names = [s.name for s in sources]
    if len(set(names)) != len(names):
        raise ValueError(f"Duplicate source names: {names}")
    dec = dict(d.get("decontam", {}))
    dec["eval_sets"] = [
        _build(EvalSetSpec, e, "decontam.eval_sets") for e in dec.get("eval_sets", [])
    ]
    return PipelineConfig(
        pipeline=_build(PipelineSpec, d.get("pipeline", {}), "pipeline"),
        sources=sources,
        quality=_build(QualityThresholds, d.get("quality", {}), "quality"),
        dedup=_build(DedupSpec, d.get("dedup", {}), "dedup"),
        decontam=_build(DecontamSpec, dec, "decontam"),
        tokenizer=_build(TokenizerSpec, d.get("tokenizer", {}), "tokenizer"),
        raw={k: v for k, v in d.items() if k in known},
    )


def load_pipeline_config(path: str | os.PathLike) -> PipelineConfig:
    with open(path, "rb") as f:
        return config_from_dict(tomllib.load(f))


# ---------------------------------------------------------------------------
# Read and write documents (JSONL: {"id", "text", "source", ...metadata})
# ---------------------------------------------------------------------------

Doc = dict[str, Any]


def _open(path: str | os.PathLike, mode: str = "rt") -> Any:
    return (
        gzip.open(path, mode, encoding="utf-8")
        if str(path).endswith(".gz")
        else open(path, mode, encoding="utf-8")
    )


def read_jsonl(path: str | os.PathLike) -> Iterator[Doc]:
    with _open(path) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(docs: Iterable[Doc], path: str | os.PathLike) -> int:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    n = 0
    with _open(tmp, "wt") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
            n += 1
    os.replace(tmp, p)
    return n


def _expand(patterns: Sequence[str]) -> list[Path]:
    files: list[Path] = []
    for pat in patterns:
        matches = sorted(glob.glob(pat))
        if not matches:
            raise FileNotFoundError(f"Input not found: {pat}")
        files.extend(Path(m) for m in matches)
    return files


def read_source(spec: SourceSpec) -> Iterator[Doc]:
    """Read all raw documents of one source.

    Plain text is split at blank lines and joined into documents of about doc_chars characters. JSONL is read directly.
    """
    n = 0
    for f in _expand(spec.inputs):
        if spec.format == "text":
            rows: Iterable[Doc] = (
                {"id": f"{f.name}#{i}", "text": t}
                for i, t in enumerate(split_into_documents(f.read_text("utf-8"), spec.doc_chars))
            )
        elif spec.format == "jsonl":
            rows = read_jsonl(f)
        else:
            raise ValueError(f"Source {spec.name}: unknown format={spec.format!r}")
        for i, row in enumerate(rows):
            if spec.max_docs is not None and n >= spec.max_docs:
                return
            text = row.get(spec.text_field)
            if not isinstance(text, str):
                continue
            doc = {k: v for k, v in row.items() if k != spec.text_field}
            doc["text"] = text
            doc.setdefault("id", f"{f.name}#{i}")
            doc["source"] = spec.name
            n += 1
            yield doc


# ---------------------------------------------------------------------------
# Stages (pure functions: a list of documents in, the kept documents + statistics out)
# ---------------------------------------------------------------------------


@dataclass
class StageResult:
    kept: list[Doc]
    removed: Counter = field(default_factory=Counter)  # removal reason → number of documents

    @property
    def n_removed(self) -> int:
        return sum(self.removed.values())


def clean_stage(docs: Iterable[Doc], min_chars: int = 1) -> StageResult:
    kept, removed = [], Counter()
    for d in docs:
        t = clean_document(d["text"], min_chars=min_chars)
        if t is None:
            removed["too_short"] += 1
            continue
        kept.append({**d, "text": t})
    return StageResult(kept, removed)


def detect_script_lang(text: str) -> str:
    """A minimal rough language check. If Chinese characters are > 30% of the non-white-space characters, return zh; if not, other.

    Step 2 uses fastText lid.176 instead.
    """
    return "zh" if cjk_ratio(text) > 0.3 else "other"


def langid_stage(docs: Iterable[Doc], lang: str) -> StageResult:
    kept, removed = [], Counter()
    for d in docs:
        if lang == "zh" and detect_script_lang(d["text"]) != "zh":
            removed["not_zh"] += 1
        elif lang == "en" and cjk_ratio(d["text"]) > 0.1:
            removed["not_en"] += 1
        else:
            kept.append(d)
    return StageResult(kept, removed)


def code_quality_reasons(text: str) -> list[str]:
    """The three rules from Section 3.1 of the Codex paper (arXiv:2107.03374) that filter code files.

    Mean line length > 100, longest line > 1000, and a fraction of alphanumeric characters that is
    too low (< 0.25 here). Such files are usually generated or minified.
    """
    lines = text.split("\n") or [""]
    reasons = []
    if sum(len(ln) for ln in lines) / len(lines) > 100:
        reasons.append("code_mean_line_len")
    if max(len(ln) for ln in lines) > 1000:
        reasons.append("code_max_line_len")
    if sum(c.isalnum() for c in text) / max(len(text), 1) < 0.25:
        reasons.append("code_alnum_frac")
    return reasons


def heuristic_stage(docs: Iterable[Doc], mode: str, th: QualityThresholds) -> StageResult:
    """mode="web": the Gopher/C4/FineWeb rules of quality.py; "code": the Codex rules; "none": no filter.

    A document can fail several rules. `removed` counts only the first reason (for the funnel chart).
    """
    kept, removed = [], Counter()
    for d in docs:
        if mode == "none":
            reasons: list[str] = []
        elif mode == "web":
            reasons = quality_check(d["text"], th).reasons
        elif mode == "code":
            reasons = code_quality_reasons(d["text"])
        else:
            raise ValueError(f"Unknown heuristics mode: {mode!r}")
        if reasons:
            removed[reasons[0]] += 1
        else:
            kept.append(d)
    return StageResult(kept, removed)


def load_classifier(path: str) -> Any:
    """'module:ClassName' → an instance (the constructor gets no arguments)."""
    mod, _, cls = path.partition(":")
    return getattr(importlib.import_module(mod), cls)()


def score_stage(docs: Sequence[Doc], spec: SourceSpec) -> StageResult:
    """Quality score thresholds: first the score field of the data set, then the custom classifier."""
    kept, removed = list(docs), Counter()
    if spec.score_field and spec.score_min is not None:
        nxt = []
        for d in kept:
            s = d.get(spec.score_field)
            if s is None or float(s) < spec.score_min:
                removed[f"{spec.score_field}<{spec.score_min}"] += 1
            else:
                nxt.append(d)
        kept = nxt
    if spec.classifier:
        clf = load_classifier(spec.classifier)
        scores = clf.score([d["text"] for d in kept])
        nxt = []
        for d, s in zip(kept, scores):
            if s < spec.classifier_min:
                removed["classifier"] += 1
            else:
                nxt.append({**d, "quality_score": float(s)})
        kept = nxt
    return StageResult(kept, removed)


def dedup_stage(docs: Sequence[Doc], spec: DedupSpec) -> tuple[StageResult, list[list[str]]]:
    """Exact dedup + MinHash LSH near-dedup.

    Returns the kept documents and the near-duplicate clusters (document ids; the first one is kept).
    """
    kept = list(docs)
    removed: Counter = Counter()
    clusters: list[list[str]] = []
    if spec.exact:
        keep = exact_dedup([d["text"] for d in kept])
        removed["exact_duplicate"] += len(kept) - len(keep)
        kept = [kept[i] for i in keep]
    if spec.near and len(kept) > 1:
        keep, cl = near_dedup(
            [d["text"] for d in kept],
            threshold=spec.threshold,
            num_perm=spec.num_perm,
            bands=spec.bands,
            ngram=spec.ngram,
            n_jobs=spec.n_jobs,
        )
        removed["near_duplicate"] += len(kept) - len(keep)
        clusters = [[kept[i]["id"] for i in c] for c in cl]
        kept = [kept[i] for i in keep]
    return StageResult(kept, removed), clusters


def _strings(obj: Any) -> Iterator[str]:
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _strings(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _strings(v)


def load_eval_texts(es: EvalSetSpec) -> list[str]:
    """Evaluation set → one text for each item.

    For JSONL, take the fields in `fields` (default: all strings, including tool names and schemas).
    """
    p = Path(es.path)
    if p.suffix == ".txt":
        return [ln.strip() for ln in p.read_text("utf-8").splitlines() if ln.strip()]
    out = []
    for row in read_jsonl(p):
        parts = [row.get(k) for k in es.fields] if es.fields else [row]
        out.append("\n".join(s for part in parts for s in _strings(part)))
    return out


def build_eval_index(spec: DecontamSpec) -> tuple[NgramIndex, dict[str, int]]:
    from zero.data.decontam import normalize_tokens

    index = NgramIndex(spec.n)
    sizes = {}
    for es in spec.eval_sets:
        texts = [t for t in load_eval_texts(es) if len(normalize_tokens(t)) >= spec.min_tokens]
        index.add_eval_set(es.name, texts)
        sizes[es.name] = len(texts)
    return index, sizes


def decontam_stage(docs: Sequence[Doc], index: NgramIndex) -> tuple[StageResult, list[dict]]:
    kept, removed, hits = [], Counter(), []
    for d in docs:
        h = index.check(d["text"])
        if h:
            removed["contaminated"] += 1
            hits.append(
                {
                    "doc": d["id"],
                    "hits": [{"eval_set": k[0], "item": k[1], "ngrams": v} for k, v in h.items()],
                }
            )
        else:
            kept.append(d)
    return StageResult(kept, removed), hits


def split_train_val(
    docs: Sequence[Doc], val_fraction: float, seed: int, name: str, val_max: int = 0
) -> tuple[list[Doc], list[Doc]]:
    docs = list(docs)
    random.Random(f"{seed}-{name}").shuffle(docs)
    n_val = max(1, int(len(docs) * val_fraction)) if len(docs) > 1 else 0
    if val_max:
        n_val = min(n_val, val_max)
    return docs[n_val:], docs[:n_val]


def sample_for_tokenizer(
    train: dict[str, list[Doc]], weights: dict[str, float], total_bytes: int, seed: int
) -> list[str]:
    """Sample the text for the tokenizer training from the sources by mixture weight.

    Source i gives at most total_bytes × w_i bytes (total_bytes=0 means all).
    """
    out = []
    for name, docs in train.items():
        budget = total_bytes * weights[name] if total_bytes else float("inf")
        order = list(range(len(docs)))
        random.Random(f"{seed}-tok-{name}").shuffle(order)
        used = 0
        for i in order:
            if used >= budget:
                break
            out.append(docs[i]["text"])
            used += len(docs[i]["text"].encode("utf-8"))
    return out


# ---------------------------------------------------------------------------
# Main flow
# ---------------------------------------------------------------------------


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5, check=True
        ).stdout.strip()
    except Exception:  # noqa: BLE001 - also runs outside a git repository
        return ""


def _nbytes(docs: Iterable[Doc]) -> int:
    return sum(len(d["text"].encode("utf-8")) for d in docs)


def run_pipeline(cfg: PipelineConfig, log: Callable[[str], None] = print) -> dict[str, Any]:
    """Run the full pipeline. Return the manifest (also written to <work_dir>/manifest.json)."""
    t0 = time.time()
    P = cfg.pipeline
    work = Path(P.work_dir)
    out_dir = Path(P.out_dir) if P.out_dir else work / "shards"
    work.mkdir(parents=True, exist_ok=True)
    weights_raw = {s.name: float(s.weight) for s in cfg.sources}
    wsum = sum(weights_raw.values())
    weights = {k: v / wsum for k, v in weights_raw.items()}

    funnel: dict[str, list[dict[str, Any]]] = {}
    removed_all: dict[str, dict[str, dict[str, int]]] = {}
    clusters_all: dict[str, int] = {}
    per_source: dict[str, list[Doc]] = {}

    def record(src: str, stage: str, docs: Sequence[Doc], res: StageResult | None = None) -> None:
        funnel.setdefault(src, []).append(
            {"stage": stage, "docs": len(docs), "bytes": _nbytes(docs)}
        )
        if res is not None and res.removed:
            removed_all.setdefault(src, {})[stage] = dict(res.removed)
        if P.write_stages or stage == "decontam":
            write_jsonl(docs, work / f"{len(funnel[src]):02d}_{stage}" / f"{src}.jsonl")

    def took(t: float) -> str:  # time of each stage: in a large run, it shows which stage is slow
        return f"{time.time() - t:.0f}s"

    # 1–5: for each source
    for spec in cfg.sources:
        t = time.time()
        docs = list(read_source(spec))
        record(spec.name, "raw", docs)
        log(f"[pipeline] {spec.name}: read {len(docs)} documents ({took(t)})")
        for stage, fn in [
            ("clean", lambda d, s=spec: clean_stage(d, s.min_chars)),
            ("langid", lambda d, s=spec: langid_stage(d, s.lang)),
            ("heuristics", lambda d, s=spec: heuristic_stage(d, s.heuristics, cfg.quality)),
            ("score", lambda d, s=spec: score_stage(d, s)),
        ]:
            t = time.time()
            res = fn(docs)
            docs = res.kept
            record(spec.name, stage, docs, res)
            log(f"[pipeline] {spec.name}: {stage} → {len(docs)} documents ({took(t)})")
        t = time.time()
        res, clusters = dedup_stage(docs, cfg.dedup)
        docs = res.kept
        clusters_all[spec.name] = len(clusters)
        record(spec.name, "dedup", docs, res)
        per_source[spec.name] = docs
        log(f"[pipeline] {spec.name}: {funnel[spec.name][0]['docs']} → {len(docs)} documents (after dedup; dedup took {took(t)})")

    # 5': cross-source dedup (optional), in the order of the sources: a source that comes first keeps its documents
    t = time.time()
    if cfg.dedup.cross_source and len(per_source) > 1:
        seen: set[str] = set()
        for name, docs in per_source.items():
            keep = []
            for d in docs:
                h = text_hash(d["text"])
                if h not in seen:
                    seen.add(h)
                    keep.append(d)
            if len(keep) < len(docs):
                removed_all.setdefault(name, {})["cross_source"] = {
                    "exact_duplicate": len(docs) - len(keep)
                }
            per_source[name] = keep

    log(f"[pipeline] cross-source dedup ({took(t)})")

    # 6: decontamination
    index, eval_sizes = build_eval_index(cfg.decontam)
    contamination: dict[str, list[dict]] = {}
    for name in per_source:
        t = time.time()
        res, hits = decontam_stage(per_source[name], index)
        per_source[name] = res.kept
        contamination[name] = hits
        record(name, "decontam", res.kept, res)
        log(f"[pipeline] {name}: decontam → {len(res.kept)} documents ({took(t)}, with the disk write)")

    # 7: split
    train: dict[str, list[Doc]] = {}
    val: dict[str, list[Doc]] = {}
    for name, docs in per_source.items():
        train[name], val[name] = split_train_val(docs, P.val_fraction, P.seed, name, P.val_max_docs)
        if not train[name]:
            raise ValueError(f"Source {name} has no training documents after the filters. Check if the rules are correct for this source")

    # 8: tokenizer
    T = cfg.tokenizer
    if T.path and Path(T.path).exists():
        tok = Tokenizer.load(T.path)
        log(f"[pipeline] using the existing tokenizer {T.path}")
    else:
        texts = sample_for_tokenizer(train, weights, T.sample_bytes, P.seed)
        log(
            f"[pipeline] training the tokenizer: vocab={T.vocab_size}, pretokenize={T.pretokenize}, "
            f"{sum(len(t.encode()) for t in texts) / 1e6:.2f} MB of text"
        )
        t = time.time()
        tok = train_tokenizer(texts, T.vocab_size, T.pretokenize, min_frequency=T.min_frequency)
        log(f"[pipeline] tokenizer trained ({took(t)})")
    tok_path = Path(T.path) if T.path else out_dir / "tokenizer.json"
    tok.save(tok_path)
    compression = {
        name: round(tok.bytes_per_token([d["text"] for d in docs]), 4)
        for name, docs in val.items()
        if docs
    }

    # 9: shards
    shards: dict[str, dict[str, Any]] = {}
    for name in per_source:
        t = time.time()
        info: dict[str, Any] = {}
        for split, docs in (("train", train[name]), ("val", val[name])):
            for stale in out_dir.glob(f"{name}_{split}_*.bin"):  # shards from the last run (there can be more of them)
                stale.unlink()
            paths = write_shards(
                (d["text"] for d in docs),
                tok,
                out_dir,
                f"{name}_{split}",
                shard_tokens=P.shard_tokens,
                source=name,
            )
            meta = json.loads((out_dir / f"{name}_{split}.json").read_text())
            info[split] = {
                "docs": len(docs),
                "tokens": meta["num_tokens"],
                "files": [p.name for p in paths],
            }
        shards[name] = info
        log(f"[pipeline] {name}: shards {info['train']['tokens']:,} + {info['val']['tokens']:,} tokens ({took(t)})")

    # 10: manifest
    mixture = {}
    for name, w in weights.items():
        tokens = shards[name]["train"]["tokens"]
        m: dict[str, Any] = {"weight": round(w, 6), "train_tokens": tokens}
        if P.token_budget:
            m["budget_tokens"] = w * P.token_budget
            m["epochs_at_budget"] = round(w * P.token_budget / max(tokens, 1), 4)
        mixture[name] = m
    provenance = {}
    for spec in cfg.sources:
        reg = SOURCES.get(spec.registry)
        provenance[spec.name] = {
            "registry": spec.registry,
            "url": reg.url if reg else "",
            "license": reg.license if reg else "未登记",
            "languages": list(reg.languages) if reg else [],
            "inputs": spec.inputs,
            "input_sha256": _inputs_digest(spec),
        }
    manifest = {
        "name": P.name,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git_commit": _git_commit(),
        "config_sha256": hashlib.sha256(
            json.dumps(cfg.raw, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest()[:16],
        "config": cfg.raw,
        "funnel": funnel,
        "removed": removed_all,
        "near_dup_clusters": clusters_all,
        "decontam": {
            "n": cfg.decontam.n,
            "eval_sets": eval_sizes,
            "contaminated_docs": {k: len(v) for k, v in contamination.items()},
            "hits": contamination,
        },
        "tokenizer": {
            "path": str(tok_path),
            "hash": tok.hash(),
            "vocab_size": tok.vocab_size,
            "pretokenize": T.pretokenize,
            "gguf_pre_tokenizer": GGUF_PRE_TOKENIZER.get(T.pretokenize, "unknown"),
            "val_bytes_per_token": compression,
        },
        "shards": {"dir": str(out_dir), **shards},
        "mixture": mixture,
        "provenance": provenance,
        "seconds": round(time.time() - t0, 1),
    }
    (work / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False))
    (work / "pretrain_sources.toml").write_text(pretrain_sources_toml(manifest))
    log(f"[pipeline] done in {manifest['seconds']}s, manifest: {work / 'manifest.json'}")
    return manifest


def _inputs_digest(spec: SourceSpec) -> str:
    """Hash of the input files: the content hash (small files), or the hash of name + size + modification time (files > 64MB)."""
    h = hashlib.sha256()
    for f in _expand(spec.inputs):
        st = f.stat()
        h.update(f.name.encode())
        if st.st_size <= 64 << 20:
            h.update(f.read_bytes())
        else:
            h.update(f"{st.st_size}-{int(st.st_mtime)}".encode())
    return h.hexdigest()[:16]


def pretrain_sources_toml(manifest: dict[str, Any]) -> str:
    """Make a [data] section that you can paste into configs/*/pretrain.toml."""
    d = manifest["shards"]["dir"]
    lines = [
        f"# Made by zero.data.pipeline ({manifest['name']}, config hash {manifest['config_sha256']})",
        "[data]",
        f'tokenizer = "{manifest["tokenizer"]["path"]}"',
        f'val = "{d}/*_val_*.bin"',
        "",
    ]
    for name, m in manifest["mixture"].items():
        lines += [
            "[[data.sources]]",
            f'name = "{name}"',
            f'path = "{d}/{name}_train_*.bin"',
            f"weight = {m['weight']}",
            "",
        ]
    return "\n".join(lines)


def funnel_table(manifest: dict[str, Any]) -> str:
    """Show the funnel as a text table: the number of documents left after each step, for each source."""
    stages = [s["stage"] for s in next(iter(manifest["funnel"].values()))]
    rows = [["source", *stages]]
    for src, steps in manifest["funnel"].items():
        rows.append([src, *[str(s["docs"]) for s in steps]])
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    return "\n".join("  ".join(c.rjust(w) for c, w in zip(r, widths)) for r in rows)


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Pretraining data pipeline (Chapter 13)")
    ap.add_argument("--config", required=True)
    args = ap.parse_args(argv)
    cfg = load_pipeline_config(args.config)
    m = run_pipeline(cfg, log=lambda s: print(s, flush=True))  # flush at once: a large run takes several hours
    print(funnel_table(m))
    for name, mix in m["mixture"].items():
        print(
            f"  {name:>16}: train {mix['train_tokens']:>10,} tokens, weight {mix['weight']:.3f}, "
            f"validation bytes/token {m['tokenizer']['val_bytes_per_token'].get(name, float('nan')):.3f}"
        )


if __name__ == "__main__":
    main()
