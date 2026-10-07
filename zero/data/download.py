"""Data downloader for Step 2: stream registered data sets from Hugging Face and write JSONL shards with provenance (Chapter 13).

    uv sync --extra data                                       # install datasets (and boto3, which Stack-Edu needs)
    uv run python -m zero.data.download --config configs/main/data.toml --sources fineweb-edu
    uv run python -m zero.data.download --config configs/main/data.toml --dry-run   # print the plan only

**Verification status**: the build environment of Step 1 had no access to huggingface.co. At that
time, only the pure functions had unit tests (config parsing, record construction, shard writing
and resume, download planning, license check). tests/test_download.py uses fake data instead of
the network. On 2026-10-01, the downloader used the network for the first time: `datasets`
streaming + stop at `target_bytes`. It downloaded a few MB each from FineWeb-Edu, DCLM, FineMath,
FineWeb-2 Chinese, Ultra-FineWeb Chinese, and UltraData-Code (configs/vocab/download.toml; the
records are in runs/2026-10-01-vocab-corpus/). We checked the field names and the provenance, and
we fixed a bug: keep_fields overwrote a provenance field. **Not verified yet**: the Stack-Edu
content from the Software Heritage S3 (needs AWS credentials), and the throughput and resume of
large downloads. Known problem: after `datasets` streaming, the interpreter reports the fatal error
"PyGILState_Release" at exit (exit code 134). All data is already written before this error.
Trust `_manifest.json`, not the exit code.
For the first download of Step 2, first run `--max-docs 1000` for each source to check the field
names and the content. Then remove the limit.

Design points:
- **Download only registered data**: each source must be registered in zero/data/sources.py.
  By default, the downloader refuses a source whose license is still "待核实" (to be verified).
  `--allow-unverified` lets it through. This follows GOAL.md 3.3: "use only public data whose
  license allows it".
- **The provenance goes with each record**: each JSON line has source, hf_repo, hf_config,
  hf_split, hf_revision, the row number, and the metadata of the data set (url, dump, quality
  score, ...). The data list of the model card is made from these fields.
- **Resume**: each source has a `_manifest.json` that lists the completed shards (file name, rows,
  bytes, sha256). A new run skips the completed rows and continues with the next shard.
- **The budget decides the download size**: `plan_targets` computes the raw bytes to download for
  each source: token budget × mixture weight × bytes per token × margin for filter losses.
- **Stack-Edu has only file ids**: the content comes from the Software Heritage S3 bucket by
  blob_id (the method of the data set card). This needs AWS credentials. See `fetch_swh_content`.
"""

from __future__ import annotations

import argparse
import dataclasses
import gzip
import hashlib
import json
import os
import sys
import time
import tomllib
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from zero.data.sources import SOURCES

# ---------------------------------------------------------------------------
# Specification
# ---------------------------------------------------------------------------


@dataclass
class DownloadSpec:
    """How to download one source. Corresponds to [sources.download] under each [[sources]] in configs/main/data.toml."""

    name: str  # output directory name / shard prefix (the same as the source name in the pipeline)
    registry: str  # registry name in zero/data/sources.py
    repo: str  # Hugging Face data set repository, for example "HuggingFaceFW/fineweb-edu"
    # Subset (name=), for example "sample-100BT" or "cmn_Hani". It can be a list (Stack-Edu has one
    # subset for each programming language); the subsets are read in sequence.
    config: str | list[str] | None = None
    split: str = "train"
    revision: str | None = None  # pin to one commit for reproducibility (fill in when Step 2 is final)
    data_files: str | None = None
    text_field: str = "text"
    id_field: str = "id"
    keep_fields: list[str] = field(default_factory=list)  # more metadata fields to keep
    swh_content: bool = False  # Stack-Edu: get the text by blob_id from Software Heritage
    target_bytes: int = 0  # stop after this many raw bytes (0 = computed by plan_targets, or no limit)
    max_docs: int = 0  # maximum number of documents to download (0 = no limit)
    docs_per_shard: int = 100_000

    @classmethod
    def from_dict(cls, name: str, registry: str, d: dict[str, Any]) -> DownloadSpec:
        names = {f.name for f in dataclasses.fields(cls)} - {"name", "registry"}
        unknown = set(d) - names
        if unknown:
            raise ValueError(f"[{name}.download] unknown fields: {sorted(unknown)}")
        return cls(name=name, registry=registry, **d)


def specs_from_config(cfg: dict[str, Any]) -> list[DownloadSpec]:
    """Get all sources with [sources.download] from the data config (the dict parsed from configs/main/data.toml)."""
    out = []
    for s in cfg.get("sources", []):
        if "download" in s:
            out.append(
                DownloadSpec.from_dict(s["name"], s.get("registry", s["name"]), s["download"])
            )
    return out


def check_license(spec: DownloadSpec, allow_unverified: bool = False) -> str:
    """Return the registered license.

    Raise an error if the source is not registered, or if the license is "待核实" (to be verified)
    and allow_unverified is not set.
    """
    if spec.registry not in SOURCES:
        raise KeyError(f"{spec.name}: {spec.registry!r} is not registered in zero/data/sources.py; cannot download it")
    lic = SOURCES[spec.registry].license
    if "待核实" in lic and not allow_unverified:
        raise PermissionError(
            f"{spec.name}: the license is not verified (license={lic}). Verify it by hand and update sources.py, or add --allow-unverified"
        )
    return lic


def plan_targets(
    weights: dict[str, float],
    token_budget: float,
    bytes_per_token: dict[str, float],
    keep_rate: dict[str, float] | None = None,
    margin: float = 1.1,
) -> dict[str, int]:
    """The number of raw bytes to download for each source.

    Training tokens needed = budget × mixture weight. In bytes: × bytes per token (measured with the
    main-line tokenizer on a sample of the source). Then divide by the keep rate of the pipeline
    (deduplication and filters remove a part), and multiply by a small margin.
    """
    tot = sum(weights.values())
    keep_rate = keep_rate or {}
    out = {}
    for name, w in weights.items():
        need = token_budget * w / tot * bytes_per_token[name]
        out[name] = int(need / keep_rate.get(name, 1.0) * margin)
    return out


# ---------------------------------------------------------------------------
# Records and shards
# ---------------------------------------------------------------------------


def make_record(row: dict[str, Any], spec: DownloadSpec, index: int) -> dict[str, Any] | None:
    """One row of the data set → our JSONL record (text + provenance + kept metadata). Return None if the text is empty."""
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
    if "hf_subset" in row:
        rec["hf_config"] = row["hf_subset"]
    reserved = set(rec)
    for k in spec.keep_fields:
        if k in row:
            # A field of the data set can have the same name as one of our provenance fields
            # ("source" in Ultra-FineWeb is the name of the upstream corpus). Keep it with a new name;
            # do not overwrite the provenance. Found in the first real download in 2026-10.
            rec[f"orig_{k}" if k in reserved else k] = row[k]
    return rec


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


class ShardWriter:
    """Write the records to `<out>/<name>/<name>-00000.jsonl.gz`, with a new file every docs_per_shard rows.

    After each shard, update `_manifest.json`. It writes a temporary file first and then renames it,
    so a killed process never leaves half a manifest.
    """

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

    # ---- resume information ----
    @property
    def rows_done(self) -> int:
        """Number of data set rows that the completed shards cover (a resume skips this many rows)."""
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
        """Write one record. rec = None means that this row is skipped.

        A skipped row still counts as a row, so that a resume stays aligned with the row numbers of the data set.
        """
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
# Network part (HF streaming: verified at small scale in 2026-10; Software Heritage S3: not verified yet)
# ---------------------------------------------------------------------------


def hf_rows(spec: DownloadSpec, skip: int = 0) -> Iterator[dict[str, Any]]:  # pragma: no cover
    """Stream with `datasets` (do not download the full data set). Verified in 2026-10 with small downloads from 6 data sets."""
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError("datasets is necessary: uv sync --extra data") from e
    import itertools

    configs = spec.config if isinstance(spec.config, list) else [spec.config]

    def stream(config: str | None) -> Iterator[dict[str, Any]]:
        kw: dict[str, Any] = {"split": spec.split, "streaming": True}
        if config:
            kw["name"] = config
        if spec.revision:
            kw["revision"] = spec.revision
        if spec.data_files:
            kw["data_files"] = spec.data_files
        for row in load_dataset(spec.repo, **kw):
            yield {**row, "hf_subset": config} if len(configs) > 1 else row

    # The subsets follow each other in sequence. A resume skips the first `skip` rows. The skipped rows
    # are still read from the network; if this is too slow in Step 2, record the progress per subset.
    yield from itertools.islice(itertools.chain.from_iterable(map(stream, configs)), skip, None)


def fetch_swh_content(blob_id: str, s3: Any = None) -> str | None:  # pragma: no cover
    """Get the file content by blob_id from the Software Heritage S3 bucket (the method of the Stack-Edu data set card). Not verified yet.

    Needs boto3 and AWS credentials. Returns None if the object does not exist (NoSuchKey).
    """
    if s3 is None:
        import boto3  # optional dependency: uv sync --extra data

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
    """Download one source to out_dir/<name>/. If rows is None, stream from Hugging Face (tests give fake data).

    The download stops when the data set ends (complete=True), or at target_bytes / max_docs
    (complete=False; the next run resumes).
    """
    lic = check_license(spec, allow_unverified)
    w = ShardWriter(out_dir, spec, lic)
    if w.manifest.get("complete"):
        log(f"[download] {spec.name} is complete, skipped")
        return w.manifest
    # The target is already reached (the last run stopped at target_bytes / max_docs). Do not open the
    # stream again: to skip the downloaded rows, a resume must read them again from the network,
    # and that is the same as a second download for no purpose.
    if (spec.max_docs and w.docs_done >= spec.max_docs) or (
        spec.target_bytes and w.bytes_done >= spec.target_bytes
    ):
        log(f"[download] {spec.name} reached the download target, skipped")
        return w.manifest
    skip = w.rows_done
    it = iter(rows) if rows is not None else hf_rows(spec, skip=skip)
    if rows is not None and skip:  # fake or local data: skip the completed rows by hand
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
    log(f"[download] {spec.name}: {docs:,} documents, {nbytes / 1e9:.2f} GB, complete={complete}")
    return m


def main(argv: Sequence[str] | None = None) -> None:  # pragma: no cover
    ap = argparse.ArgumentParser(description="Step 2: download the pretraining data (large downloads are not verified yet)")
    ap.add_argument("--config", required=True, help="configs/main/data.toml")
    ap.add_argument("--out", default="data/raw")
    ap.add_argument("--sources", nargs="*", help="download only these sources (default: all)")
    ap.add_argument("--max-docs", type=int, default=0, help="maximum number of documents for each source (for test runs)")
    ap.add_argument("--allow-unverified", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="print the plan only; do not use the network")
    args = ap.parse_args(argv)
    with open(args.config, "rb") as f:
        cfg = tomllib.load(f)
    specs = specs_from_config(cfg)
    if args.sources:
        specs = [s for s in specs if s.name in set(args.sources)]
    for s in specs:
        if args.max_docs:
            s.max_docs = args.max_docs
        lic = SOURCES[s.registry].license if s.registry in SOURCES else "unregistered"
        b = s.target_bytes
        target = f"{b / 1e9:.0f} GB" if b >= 1e9 else f"{b / 1e6:.1f} MB"
        print(
            f"{s.name:>18}  {s.repo}  config={s.config}  split={s.split}  license={lic}  target {target}"
        )
        if not args.dry_run:
            download_source(s, args.out, allow_unverified=args.allow_unverified)


if __name__ == "__main__":  # pragma: no cover
    main()
    # The shards and _manifest.json are written and on disk. The background threads of `datasets`
    # streaming can make the interpreter crash at exit (PyGILState_Release, exit code 134), or hang
    # (in 2026-10, during parallel downloads, all 8 processes hung and blocked all parallel slots).
    # So end the process directly and skip the cleanup of the interpreter.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
