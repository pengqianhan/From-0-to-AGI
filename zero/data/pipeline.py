"""预训练数据流水线：一个 TOML 驱动的端到端配方（对应第 13 章）。

    uv run python -m zero.data.pipeline --config configs/tiny/data.toml    # CPU 上约半分钟
    uv run python -m zero.data.pipeline --config configs/main/data.toml    # 第二步，尚未验证

阶段（每个阶段的输出都落盘成 JSONL，方便单独检查、单独重跑）：

    1. 读取      [[sources]] 的 inputs：纯文本（按空行切段拼成文档）或 download.py 写出的 JSONL 分片
    2. 清洗      clean.py：Unicode NFC、空白与控制字符、去掉特殊 token 字面量
    3. 语言检查  按文字（汉字占比）粗检：中文来源里混进来的非中文文档丢掉（第二步换 fastText lid.176）
    4. 质量      启发式规则（web：Gopher + C4 + FineWeb；code：Codex 的三条规则）+ 可选的分数阈值
                 （数据集自带的分类器分数，如 FineWeb-Edu 的 int_score）或自定义分类器
    5. 去重      精确哈希 + MinHash LSH（dedup.py），先在每个来源内部做，可选再跨来源做一次
    6. 去污染    与评测集做 n-gram（默认 13）重叠检查（decontam.py），撞了的文档整篇删除
    7. 切分      每个来源按固定种子切出验证集
    8. 分词器    按配比从各来源采样文本，训练 byte-level BPE（预切分正则可选 qwen2 / qwen3.5）
    9. 分片      shard.py：<来源>_train_*.bin / <来源>_val_*.bin
   10. 清单      manifest.json：每一步每个来源留下多少文档/字节（漏斗）、删除原因、去污染命中、
                 分词器哈希与各来源压缩率、分片 token 数、配比与"按预算需要几个 epoch"、许可证与来源地址

tiny 配置把 assets/tiny_corpus 走一遍全流程；main 配置描述第二步的真实配方（configs/main/data.toml）。
第二步的规模（数十亿篇文档）下，第 5 步的 MinHash 需要换成分布式实现（如 datatrove 的 MinhashDedup），
这里的单进程实现只保证算法正确（tests/test_data.py、tests/test_pipeline.py），**尚未在大规模数据上验证**：
它把一个来源的全部文档放在内存里，第二步要按输入分片流式处理（阶段 1–4、6 天然可以逐篇流式，
第 5 步需要分布式 MinHash）。
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
# 分词器：预切分正则的两个预设
# ---------------------------------------------------------------------------

#: Qwen3.5 的 tokenizer.json 里的正则（2026-09 读取）：和 Qwen2/Qwen3 只差一处——字母串从 \p{L}+
#: 变成 [\p{L}\p{M}]+，标点串也把 \p{M} 排除在外。\p{M} 是"组合符号"（天城文、泰文的元音符号、
#: 阿拉伯文的变音符号……）。旧正则会在这些符号处把一个词切开，新正则让"字母 + 组合符号"留在一个词块里。
QWEN35_PRETOKENIZE_REGEX = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}"
    r"| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)

PRETOKENIZE_PRESETS: dict[str, str] = {
    "qwen2": PRETOKENIZE_REGEX,  # zero/tokenizer.py 的默认值，与 Qwen2/Qwen3 相同
    "qwen3": PRETOKENIZE_REGEX,
    "qwen3.5": QWEN35_PRETOKENIZE_REGEX,
}

#: 导出 GGUF 时 llama.cpp 对应的预切分类型名（zero/export/gguf.py 的 pre_tokenizer 参数）
GGUF_PRE_TOKENIZER = {"qwen2": "qwen2", "qwen3": "qwen2", "qwen3.5": "qwen35"}


def resolve_pretokenize(name_or_regex: str) -> str:
    """预设名（qwen2 / qwen3 / qwen3.5）→ 正则；不是预设名就当作正则本身。"""
    return PRETOKENIZE_PRESETS.get(name_or_regex, name_or_regex)


def train_tokenizer(
    texts_or_files: Iterable[str | os.PathLike],
    vocab_size: int,
    pretokenize: str = "qwen2",
    special_tokens: Sequence[str] | None = None,
    min_frequency: int = 2,
) -> Tokenizer:
    """训练 byte-level BPE，可以换预切分正则。

    默认（qwen2）直接调用 `zero.tokenizer.train_bpe`，结果完全相同；换正则时只替换预切分这一步，
    规范化（NFC）、字节级、特殊 token 的排布都和 `train_bpe` 一致，所以 `Tokenizer.save_hf`、
    `bytes_per_token`、`hash` 照常可用（正则写在 tokenizer.json 里，加载时自动恢复）。
    """
    regex = resolve_pretokenize(pretokenize)
    if regex == PRETOKENIZE_REGEX:
        return train_bpe(texts_or_files, vocab_size, special_tokens, min_frequency)
    special = list(special_tokens) if special_tokens is not None else list(DEFAULT_SPECIAL_TOKENS)
    if ENDOFTEXT not in special:
        special = [ENDOFTEXT, *special]
    if vocab_size < 256 + len(special):
        raise ValueError(f"vocab_size={vocab_size} 太小")
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
# 配置
# ---------------------------------------------------------------------------


@dataclass
class SourceSpec:
    name: str  # 分片名前缀（<name>_train_*.bin），也是配比里的键
    inputs: list[str]  # glob 列表：纯文本文件，或 download.py 写出的 .jsonl / .jsonl.gz
    registry: str = ""  # zero/data/sources.py 里的登记名（许可证、地址从那里取）
    format: str = "text"  # "text" | "jsonl"
    text_field: str = "text"
    weight: float = 1.0  # 预训练配比（会归一化）
    lang: str = ""  # "zh" | "en" | "code" | ""：语言粗检
    heuristics: str = "web"  # "web" | "code" | "none"
    score_field: str = ""  # 数据集自带的质量分数字段（如 FineWeb-Edu 的 int_score）
    score_min: float | None = None
    classifier: str = ""  # "模块:类名"，类要有 score(texts) -> list[float]
    classifier_min: float = 3.0
    doc_chars: int = 4000  # 纯文本输入：按空行切段后拼成约这么长的文档
    min_chars: int = 50
    max_docs: int | None = None  # 只读前 N 篇（调试用）


@dataclass
class DedupSpec:
    exact: bool = True
    near: bool = True
    threshold: float = 0.8
    num_perm: int = 128
    bands: int = 16
    ngram: int = 5
    cross_source: bool = False  # 来源内部去重之后，是否再跨来源去一次


@dataclass
class EvalSetSpec:
    name: str
    path: str  # .jsonl（取 fields 里的字符串，默认全部字符串字段）或 .txt（每行一题）
    fields: list[str] = field(default_factory=list)


@dataclass
class DecontamSpec:
    n: int = 13
    min_tokens: int = 8  # 规范化后不足这么多"词"的题目跳过（太短的题目用子串匹配误报太多）
    eval_sets: list[EvalSetSpec] = field(default_factory=list)


@dataclass
class TokenizerSpec:
    vocab_size: int = 2048
    pretokenize: str = "qwen2"  # "qwen2" | "qwen3" | "qwen3.5" | 自定义正则
    sample_bytes: int = 0  # 训练分词器用多少字节（按配比从各来源采样）；0 = 全部训练文本
    min_frequency: int = 2
    path: str = ""  # 已有的 tokenizer.json：给了就不训练，直接用它切分片


@dataclass
class PipelineSpec:
    name: str = "data"
    work_dir: str = "out/data"  # 中间结果（每阶段的 JSONL）
    out_dir: str = ""  # 分片输出目录；默认 <work_dir>/shards
    seed: int = 0
    val_fraction: float = 0.05
    val_max_docs: int = 0  # 每个来源验证集最多多少篇（0 = 不限）
    shard_tokens: int = 100_000_000
    token_budget: float = 0.0  # 预训练 token 预算（manifest 里算"每个来源要过几遍"）
    write_stages: bool = True  # 每个阶段的结果都落盘成 JSONL（tiny 方便检查；main 数据量大时只写最后一步）


@dataclass
class PipelineConfig:
    pipeline: PipelineSpec
    sources: list[SourceSpec]
    quality: QualityThresholds = field(default_factory=QualityThresholds)
    dedup: DedupSpec = field(default_factory=DedupSpec)
    decontam: DecontamSpec = field(default_factory=DecontamSpec)
    tokenizer: TokenizerSpec = field(default_factory=TokenizerSpec)
    raw: dict[str, Any] = field(default_factory=dict)  # 原始 TOML（写进 manifest）


def _build(cls: type, d: dict[str, Any], where: str) -> Any:
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = set(d) - names
    if unknown:
        raise ValueError(f"[{where}] 不认识的字段：{sorted(unknown)}")
    return cls(**d)


def config_from_dict(d: dict[str, Any]) -> PipelineConfig:
    """TOML 字典 → PipelineConfig。只认识 [pipeline] [[sources]] [quality] [dedup] [decontam]
    [tokenizer] 这几节（main 配置里的 [download] 等小节由 download.py 读取，这里忽略）。"""
    known = {"pipeline", "sources", "quality", "dedup", "decontam", "tokenizer"}
    if not d.get("sources"):
        raise ValueError("至少要有一个 [[sources]]")
    sources = []
    for s in d["sources"]:
        s = {k: v for k, v in s.items() if k != "download"}  # [sources.download] 给 download.py
        sources.append(_build(SourceSpec, s, "sources"))
    names = [s.name for s in sources]
    if len(set(names)) != len(names):
        raise ValueError(f"来源名重复：{names}")
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
# 读写文档（JSONL：{"id", "text", "source", ...元数据}）
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
            raise FileNotFoundError(f"找不到输入：{pat}")
        files.extend(Path(m) for m in matches)
    return files


def read_source(spec: SourceSpec) -> Iterator[Doc]:
    """读一个来源的全部原始文档。纯文本按空行切段、拼成约 doc_chars 的文档；JSONL 直接读。"""
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
            raise ValueError(f"来源 {spec.name}：不认识的 format={spec.format!r}")
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
# 各阶段（纯函数：输入文档列表，输出保留的文档 + 统计）
# ---------------------------------------------------------------------------


@dataclass
class StageResult:
    kept: list[Doc]
    removed: Counter = field(default_factory=Counter)  # 删除原因 → 篇数

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
    """极简的语言粗检：汉字占非空白字符 > 30% 记为 zh，否则 other（第二步换 fastText lid.176）。"""
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
    """Codex 论文（arXiv:2107.03374）第 3.1 节过滤代码文件的三条规则：平均行长 > 100、
    最长行 > 1000、字母数字字符占比太低（这里取 < 0.25）。多半是自动生成或压缩过的文件。"""
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
    """mode="web"：quality.py 的 Gopher/C4/FineWeb 规则；"code"：Codex 规则；"none"：不过滤。
    一篇文档可能同时违反多条规则，removed 里按第一条原因计数（漏斗图用）。"""
    kept, removed = [], Counter()
    for d in docs:
        if mode == "none":
            reasons: list[str] = []
        elif mode == "web":
            reasons = quality_check(d["text"], th).reasons
        elif mode == "code":
            reasons = code_quality_reasons(d["text"])
        else:
            raise ValueError(f"不认识的 heuristics 模式：{mode!r}")
        if reasons:
            removed[reasons[0]] += 1
        else:
            kept.append(d)
    return StageResult(kept, removed)


def load_classifier(path: str) -> Any:
    """'模块:类名' → 实例（无参数构造）。"""
    mod, _, cls = path.partition(":")
    return getattr(importlib.import_module(mod), cls)()


def score_stage(docs: Sequence[Doc], spec: SourceSpec) -> StageResult:
    """质量分数阈值：先看数据集自带的分数字段，再看自定义分类器。"""
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
    """精确去重 + MinHash LSH 近似去重；返回保留的文档，以及近似重复簇（文档 id，第一篇是保留的）。"""
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
    """评测集 → 每题一段文本。JSONL 取 fields 指定的字段（默认全部字符串，包括工具名和 schema）。"""
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
    """按配比从各来源采样训练分词器的文本：来源 i 最多拿 total_bytes × w_i 字节（total_bytes=0 表示全部）。"""
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
# 主流程
# ---------------------------------------------------------------------------


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5, check=True
        ).stdout.strip()
    except Exception:  # noqa: BLE001 - 不在 git 仓库里也照常运行
        return ""


def _nbytes(docs: Iterable[Doc]) -> int:
    return sum(len(d["text"].encode("utf-8")) for d in docs)


def run_pipeline(cfg: PipelineConfig, log: Callable[[str], None] = print) -> dict[str, Any]:
    """跑完整条流水线，返回 manifest（同时写到 <work_dir>/manifest.json）。"""
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

    # 1–5：逐来源
    for spec in cfg.sources:
        docs = list(read_source(spec))
        record(spec.name, "raw", docs)
        for stage, fn in [
            ("clean", lambda d, s=spec: clean_stage(d, s.min_chars)),
            ("langid", lambda d, s=spec: langid_stage(d, s.lang)),
            ("heuristics", lambda d, s=spec: heuristic_stage(d, s.heuristics, cfg.quality)),
            ("score", lambda d, s=spec: score_stage(d, s)),
        ]:
            res = fn(docs)
            docs = res.kept
            record(spec.name, stage, docs, res)
        res, clusters = dedup_stage(docs, cfg.dedup)
        docs = res.kept
        clusters_all[spec.name] = len(clusters)
        record(spec.name, "dedup", docs, res)
        per_source[spec.name] = docs
        log(f"[pipeline] {spec.name}: {funnel[spec.name][0]['docs']} → {len(docs)} 篇（去重后）")

    # 5'：跨来源去重（可选）：按来源顺序，先出现的来源优先保留
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

    # 6：去污染
    index, eval_sizes = build_eval_index(cfg.decontam)
    contamination: dict[str, list[dict]] = {}
    for name in per_source:
        res, hits = decontam_stage(per_source[name], index)
        per_source[name] = res.kept
        contamination[name] = hits
        record(name, "decontam", res.kept, res)

    # 7：切分
    train: dict[str, list[Doc]] = {}
    val: dict[str, list[Doc]] = {}
    for name, docs in per_source.items():
        train[name], val[name] = split_train_val(docs, P.val_fraction, P.seed, name, P.val_max_docs)
        if not train[name]:
            raise ValueError(f"来源 {name} 过滤后没有训练文档了，检查规则是否对它适用")

    # 8：分词器
    T = cfg.tokenizer
    if T.path and Path(T.path).exists():
        tok = Tokenizer.load(T.path)
        log(f"[pipeline] 使用已有分词器 {T.path}")
    else:
        texts = sample_for_tokenizer(train, weights, T.sample_bytes, P.seed)
        log(
            f"[pipeline] 训练分词器：vocab={T.vocab_size}，预切分={T.pretokenize}，"
            f"{sum(len(t.encode()) for t in texts) / 1e6:.2f} MB 文本"
        )
        tok = train_tokenizer(texts, T.vocab_size, T.pretokenize, min_frequency=T.min_frequency)
    tok_path = Path(T.path) if T.path else out_dir / "tokenizer.json"
    tok.save(tok_path)
    compression = {
        name: round(tok.bytes_per_token([d["text"] for d in docs]), 4)
        for name, docs in val.items()
        if docs
    }

    # 9：分片
    shards: dict[str, dict[str, Any]] = {}
    for name in per_source:
        info: dict[str, Any] = {}
        for split, docs in (("train", train[name]), ("val", val[name])):
            for stale in out_dir.glob(f"{name}_{split}_*.bin"):  # 上次运行留下的分片（可能更多）
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

    # 10：清单
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
    log(f"[pipeline] 完成，用时 {manifest['seconds']}s，清单：{work / 'manifest.json'}")
    return manifest


def _inputs_digest(spec: SourceSpec) -> str:
    """输入文件的内容哈希（小文件）或 名字+大小+修改时间 的哈希（> 64MB 的大文件）。"""
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
    """生成可以贴进 configs/*/pretrain.toml 的 [data] 片段。"""
    d = manifest["shards"]["dir"]
    lines = [
        f"# 由 zero.data.pipeline 生成（{manifest['name']}，配置哈希 {manifest['config_sha256']}）",
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
    """把漏斗打印成文本表：每个来源每一步剩多少篇。"""
    stages = [s["stage"] for s in next(iter(manifest["funnel"].values()))]
    rows = [["来源", *stages]]
    for src, steps in manifest["funnel"].items():
        rows.append([src, *[str(s["docs"]) for s in steps]])
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    return "\n".join("  ".join(c.rjust(w) for c, w in zip(r, widths)) for r in rows)


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="预训练数据流水线（第 13 章）")
    ap.add_argument("--config", required=True)
    args = ap.parse_args(argv)
    cfg = load_pipeline_config(args.config)
    m = run_pipeline(cfg)
    print(funnel_table(m))
    for name, mix in m["mixture"].items():
        print(
            f"  {name:>16}: 训练 {mix['train_tokens']:>10,} token，配比 {mix['weight']:.3f}，"
            f"验证集字节/token {m['tokenizer']['val_bytes_per_token'].get(name, float('nan')):.3f}"
        )


if __name__ == "__main__":
    main()
