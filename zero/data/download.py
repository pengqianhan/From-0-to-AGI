"""第二步的数据下载器：从 Hugging Face 流式读取登记过的数据集，写成带出处的 JSONL 分片（对应第 13 章）。

    uv sync --extra data                                       # 装 datasets（以及 Stack-Edu 要用的 boto3）
    uv run python -m zero.data.download --config configs/main/data.toml --sources fineweb-edu
    uv run python -m zero.data.download --config configs/main/data.toml --dry-run   # 只打印计划

**尚未验证**：本构建环境访问不了 huggingface.co，下面的网络路径（`datasets` 流式读取、Software Heritage
S3）一次都没有真正跑过；只有纯函数（配置解析、记录构造、分片写入与续传、下载量规划、许可证检查）有单元测试
（tests/test_download.py 用假数据代替网络）。第二步第一次下载时，先对每个来源跑 `--max-docs 1000`
核对字段名和内容，再放开。

设计要点：
- **只下登记过的数据**：每个来源必须在 zero/data/sources.py 里登记；许可证还是"待核实"的来源默认拒绝下载
  （`--allow-unverified` 才放行），对应 GOAL.md 3.3"只用许可证允许的公开数据"。
- **出处跟着每一条记录走**：每行 JSON 带 source、hf_repo、hf_config、hf_split、hf_revision、行号，
  以及数据集自带的元数据（url、dump、质量分数……），模型卡里的数据清单从这里汇总。
- **可续传**：每个来源一个 `_manifest.json`，记录写完的分片（文件名、行数、字节数、sha256）；
  重跑时跳过已完成的行数，从下一个分片接着写。
- **下多少由预算决定**：`plan_targets` 按 token 预算 × 配比 × 每 token 字节数 × 过滤损耗余量，
  算出每个来源要下载的原始字节数。
- **Stack-Edu 只有文件 id**：内容要按 blob_id 从 Software Heritage 的 S3 桶取（数据集卡给出的方式），
  需要 AWS 凭证，见 `fetch_swh_content`。
"""

from __future__ import annotations

import argparse
import dataclasses
import gzip
import hashlib
import json
import os
import time
import tomllib
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from zero.data.sources import SOURCES

# ---------------------------------------------------------------------------
# 规格
# ---------------------------------------------------------------------------


@dataclass
class DownloadSpec:
    """一个来源怎么下载。对应 configs/main/data.toml 里每个 [[sources]] 下的 [sources.download]。"""

    name: str  # 输出目录名 / 分片前缀（与流水线的来源名一致）
    registry: str  # zero/data/sources.py 的登记名
    repo: str  # Hugging Face 数据集仓库，如 "HuggingFaceFW/fineweb-edu"
    config: str | None = None  # 子集（name=），如 "sample-100BT"、"cmn_Hani"
    split: str = "train"
    revision: str | None = None  # 固定到某个 commit，保证可复现（第二步定稿时填）
    data_files: str | None = None
    text_field: str = "text"
    id_field: str = "id"
    keep_fields: list[str] = field(default_factory=list)  # 额外保留的元数据字段
    swh_content: bool = False  # Stack-Edu：text 需要按 blob_id 从 Software Heritage 取
    target_bytes: int = 0  # 下载多少原始字节就停（0 = 由 plan_targets 计算或不限）
    max_docs: int = 0  # 最多下载多少篇（0 = 不限）
    docs_per_shard: int = 100_000

    @classmethod
    def from_dict(cls, name: str, registry: str, d: dict[str, Any]) -> DownloadSpec:
        names = {f.name for f in dataclasses.fields(cls)} - {"name", "registry"}
        unknown = set(d) - names
        if unknown:
            raise ValueError(f"[{name}.download] 不认识的字段：{sorted(unknown)}")
        return cls(name=name, registry=registry, **d)


def specs_from_config(cfg: dict[str, Any]) -> list[DownloadSpec]:
    """从数据配置（configs/main/data.toml 解析出的字典）里取出所有带 [sources.download] 的来源。"""
    out = []
    for s in cfg.get("sources", []):
        if "download" in s:
            out.append(DownloadSpec.from_dict(s["name"], s.get("registry", s["name"]), s["download"]))
    return out


def check_license(spec: DownloadSpec, allow_unverified: bool = False) -> str:
    """返回登记的许可证；来源没登记、或许可证"待核实"且没有显式放行时报错。"""
    if spec.registry not in SOURCES:
        raise KeyError(f"{spec.name}：{spec.registry!r} 没有在 zero/data/sources.py 登记，不能下载")
    lic = SOURCES[spec.registry].license
    if "待核实" in lic and not allow_unverified:
        raise PermissionError(
            f"{spec.name}：许可证{lic}。先人工核实并更新 sources.py，或加 --allow-unverified"
        )
    return lic


def plan_targets(
    weights: dict[str, float],
    token_budget: float,
    bytes_per_token: dict[str, float],
    keep_rate: dict[str, float] | None = None,
    margin: float = 1.1,
) -> dict[str, int]:
    """每个来源要下载多少原始字节。

    需要的训练 token = 预算 × 配比；换成字节 = × 每 token 字节数（用主线分词器在该来源样本上测）；
    再除以流水线的保留率（去重、过滤会删掉一部分），乘一点余量。
    """
    tot = sum(weights.values())
    keep_rate = keep_rate or {}
    out = {}
    for name, w in weights.items():
        need = token_budget * w / tot * bytes_per_token[name]
        out[name] = int(need / keep_rate.get(name, 1.0) * margin)
    return out


# ---------------------------------------------------------------------------
# 记录与分片
# ---------------------------------------------------------------------------


def make_record(row: dict[str, Any], spec: DownloadSpec, index: int) -> dict[str, Any] | None:
    """数据集的一行 → 我们的 JSONL 记录（text + 出处 + 保留的元数据）。text 为空返回 None。"""
    text = row.get(spec.text_field)
    if not isinstance(text, str) or not text.strip():
        return None
    rec: dict[str, Any] = {
        "id": str(row.get(spec.id_field, f"{spec.name}-{index}")),
        "text": text,
        "source": spec.name,
        "hf_repo": spec.repo,
        "hf_config": spec.config,
        "hf_split": spec.split,
        "hf_revision": spec.revision,
        "row": index,
    }
    for k in spec.keep_fields:
        if k in row:
            rec[k] = row[k]
    return rec


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


class ShardWriter:
    """把记录写成 `<out>/<name>/<name>-00000.jsonl.gz`，每 docs_per_shard 行换一个文件；
    每写完一个分片就更新 `_manifest.json`（先写临时文件再改名，中途被杀也不会留下半个清单）。"""

    def __init__(self, out_dir: str | os.PathLike, spec: DownloadSpec, license: str) -> None:
        self.dir = Path(out_dir) / spec.name
        self.dir.mkdir(parents=True, exist_ok=True)
        self.spec = spec
        self.manifest_path = self.dir / "_manifest.json"
        if self.manifest_path.exists():
            self.manifest = json.loads(self.manifest_path.read_text())
        else:
            self.manifest = {
                "source": spec.name,
                "registry": spec.registry,
                "license": license,
                "url": SOURCES[spec.registry].url if spec.registry in SOURCES else "",
                "spec": dataclasses.asdict(spec),
                "shards": [],
                "complete": False,
            }
        self._f: Any = None
        self._tmp: Path | None = None
        self._rows = 0
        self._bytes = 0

    # ---- 续传信息 ----
    @property
    def rows_done(self) -> int:
        """已完成分片覆盖的数据集行数（续传时跳过这么多行）。"""
        return sum(s["rows"] for s in self.manifest["shards"])

    @property
    def docs_done(self) -> int:
        return sum(s["docs"] for s in self.manifest["shards"])

    @property
    def bytes_done(self) -> int:
        return sum(s["bytes"] for s in self.manifest["shards"])

    def _open(self) -> None:
        idx = len(self.manifest["shards"])
        self._name = f"{self.spec.name}-{idx:05d}.jsonl.gz"
        self._tmp = self.dir / (self._name + ".tmp")
        self._f = gzip.open(self._tmp, "wt", encoding="utf-8")
        self._rows = self._docs = self._bytes = 0

    def add(self, rec: dict[str, Any] | None) -> None:
        """写一条记录；rec 为 None 表示这一行被跳过（仍计入行数，续传时对齐数据集行号）。"""
        if self._f is None:
            self._open()
        self._rows += 1
        if rec is not None:
            self._f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            self._docs += 1
            self._bytes += len(rec["text"].encode("utf-8"))
        if self._docs >= self.spec.docs_per_shard:
            self.flush()

    def flush(self) -> None:
        if self._f is None or self._rows == 0:
            return
        self._f.close()
        final = self.dir / self._name
        os.replace(self._tmp, final)
        self.manifest["shards"].append(
            {
                "file": self._name,
                "rows": self._rows,
                "docs": self._docs,
                "bytes": self._bytes,
                "sha256": _sha256(final),
                "written": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            }
        )
        self._f = None
        self._save()

    def finish(self, complete: bool) -> dict[str, Any]:
        self.flush()
        self.manifest["complete"] = complete
        self._save()
        return self.manifest

    def _save(self) -> None:
        tmp = self.manifest_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.manifest, indent=1, ensure_ascii=False))
        os.replace(tmp, self.manifest_path)


# ---------------------------------------------------------------------------
# 网络部分（尚未验证）
# ---------------------------------------------------------------------------


def hf_rows(spec: DownloadSpec, skip: int = 0) -> Iterator[dict[str, Any]]:  # pragma: no cover
    """用 `datasets` 流式读取（不把整个数据集下载到本地）。尚未验证。"""
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError("需要 datasets：uv sync --extra data") from e
    kw: dict[str, Any] = {"split": spec.split, "streaming": True}
    if spec.config:
        kw["name"] = spec.config
    if spec.revision:
        kw["revision"] = spec.revision
    if spec.data_files:
        kw["data_files"] = spec.data_files
    ds = load_dataset(spec.repo, **kw)
    if skip:
        ds = ds.skip(skip)
    yield from ds


def fetch_swh_content(blob_id: str, s3: Any = None) -> str | None:  # pragma: no cover
    """按 blob_id 从 Software Heritage 的 S3 桶取文件内容（Stack-Edu 数据集卡给出的方式）。尚未验证。

    需要 boto3 和 AWS 凭证；取不到（NoSuchKey）返回 None。
    """
    if s3 is None:
        import boto3  # 可选依赖：uv sync --extra data

        s3 = boto3.client("s3")
    try:
        obj = s3.get_object(Bucket="softwareheritage", Key=f"content/{blob_id}")
    except Exception as e:  # noqa: BLE001
        if getattr(e, "response", {}).get("Error", {}).get("Code") == "NoSuchKey":
            return None
        raise
    with gzip.GzipFile(fileobj=obj["Body"]) as f:
        return f.read().decode("utf-8", errors="ignore")


def download_source(
    spec: DownloadSpec,
    out_dir: str | os.PathLike,
    rows: Iterable[dict[str, Any]] | None = None,
    allow_unverified: bool = False,
    fetch: Callable[[str], str | None] | None = None,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """下载一个来源到 out_dir/<name>/。rows 为 None 时从 Hugging Face 流式读取（测试时传假数据）。

    停止条件：数据集读完（complete=True），或达到 target_bytes / max_docs（complete=False，下次续传）。
    """
    lic = check_license(spec, allow_unverified)
    w = ShardWriter(out_dir, spec, lic)
    if w.manifest.get("complete"):
        log(f"[download] {spec.name} 已完成，跳过")
        return w.manifest
    skip = w.rows_done
    it = iter(rows) if rows is not None else hf_rows(spec, skip=skip)
    if rows is not None and skip:  # 假数据/本地数据：手动跳过已完成的行
        for _ in range(skip):
            next(it, None)
    if spec.swh_content and fetch is None:
        fetch = fetch_swh_content
    docs, nbytes = w.docs_done, w.bytes_done
    index = skip
    complete = True
    for row in it:
        if (spec.max_docs and docs >= spec.max_docs) or (
            spec.target_bytes and nbytes >= spec.target_bytes
        ):
            complete = False
            break
        if spec.swh_content:
            row = {**row, spec.text_field: fetch(row["blob_id"])}  # type: ignore[misc]
        rec = make_record(row, spec, index)
        w.add(rec)
        if rec is not None:
            docs += 1
            nbytes += len(rec["text"].encode("utf-8"))
        index += 1
    m = w.finish(complete)
    log(f"[download] {spec.name}: {docs:,} 篇，{nbytes / 1e9:.2f} GB，complete={complete}")
    return m


def main(argv: Sequence[str] | None = None) -> None:  # pragma: no cover
    ap = argparse.ArgumentParser(description="第二步：下载预训练数据（尚未验证）")
    ap.add_argument("--config", required=True, help="configs/main/data.toml")
    ap.add_argument("--out", default="data/raw")
    ap.add_argument("--sources", nargs="*", help="只下载这些来源（默认全部）")
    ap.add_argument("--max-docs", type=int, default=0, help="每个来源最多下载多少篇（试跑用）")
    ap.add_argument("--allow-unverified", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="只打印计划，不联网")
    args = ap.parse_args(argv)
    with open(args.config, "rb") as f:
        cfg = tomllib.load(f)
    specs = specs_from_config(cfg)
    if args.sources:
        specs = [s for s in specs if s.name in set(args.sources)]
    for s in specs:
        if args.max_docs:
            s.max_docs = args.max_docs
        lic = SOURCES[s.registry].license if s.registry in SOURCES else "未登记"
        print(
            f"{s.name:>18}  {s.repo}  config={s.config}  split={s.split}  许可证={lic}  "
            f"目标 {s.target_bytes / 1e9:.0f} GB"
        )
        if not args.dry_run:
            download_source(s, args.out, allow_unverified=args.allow_unverified)


if __name__ == "__main__":  # pragma: no cover
    main()
