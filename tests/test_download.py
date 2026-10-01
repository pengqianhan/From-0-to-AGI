"""第二步下载器（zero/data/download.py）的纯函数：用假数据代替网络。"""

from __future__ import annotations

import gzip
import json
import tomllib
from pathlib import Path

import pytest

from zero.data.download import (
    DownloadSpec,
    ShardWriter,
    check_license,
    download_source,
    make_record,
    plan_targets,
    specs_from_config,
)

REPO = Path(__file__).resolve().parent.parent


def fake_rows(n: int) -> list[dict]:
    rows = [
        {
            "id": f"doc{i}",
            "text": f"document number {i} " * 5,
            "url": f"http://x/{i}",
            "score": i % 5,
        }
        for i in range(n)
    ]
    rows[3]["text"] = "   "  # 空文本：跳过，但仍占一个行号
    return rows


def spec(**kw) -> DownloadSpec:  # noqa: ANN003
    base = dict(
        name="fw",
        registry="fineweb-edu",
        repo="HuggingFaceFW/fineweb-edu",
        config="sample-10BT",
        keep_fields=["url", "score"],
        docs_per_shard=4,
    )
    base.update(kw)
    return DownloadSpec(**base)


def read_all(d: Path) -> list[dict]:
    out = []
    for f in sorted(d.glob("*.jsonl.gz")):
        with gzip.open(f, "rt", encoding="utf-8") as fh:
            out += [json.loads(line) for line in fh]
    return out


def test_make_record_keeps_provenance() -> None:
    r = make_record({"id": 7, "text": "hello", "url": "u", "junk": 1}, spec(), 12)
    assert r == {
        "id": "7",
        "text": "hello",
        "source": "fw",
        "hf_repo": "HuggingFaceFW/fineweb-edu",
        "hf_config": "sample-10BT",
        "hf_split": "train",
        "hf_revision": None,
        "row": 12,
        "url": "u",
    }
    assert make_record({"text": ""}, spec(), 0) is None
    assert make_record({"content": "x"}, spec(text_field="content"), 0)["text"] == "x"


def test_make_record_keep_field_does_not_clobber_provenance() -> None:
    # Ultra-FineWeb 自带一个 "source" 字段（上游语料名，如 "Tele"），不能覆盖我们的出处
    r = make_record({"text": "x", "source": "Tele", "score": 0.5}, spec(keep_fields=["source", "score"]), 0)
    assert r["source"] == "fw"
    assert r["orig_source"] == "Tele"
    assert r["score"] == 0.5


def test_download_writes_shards_and_resumes(tmp_path: Path) -> None:
    rows = fake_rows(10)
    # 第一次：最多 5 篇就停（模拟中途停下），complete=False
    m = download_source(spec(max_docs=5), tmp_path, rows=rows, log=lambda s: None)
    assert m["complete"] is False and m["license"] == "ODC-By-1.0"
    assert [s["docs"] for s in m["shards"]] == [4, 1]
    # 续传：跳过已完成分片覆盖的行，接着写，直到读完
    m = download_source(spec(), tmp_path, rows=rows, log=lambda s: None)
    assert m["complete"] is True
    recs = read_all(tmp_path / "fw")
    assert [r["row"] for r in recs] == [0, 1, 2, 4, 5, 6, 7, 8, 9]  # 第 3 行是空文本
    assert all(r["source"] == "fw" and "url" in r and "score" in r for r in recs)
    assert sum(s["rows"] for s in m["shards"]) == 10
    # 已完成就不再下载
    assert download_source(spec(), tmp_path, rows=rows, log=lambda s: None)["complete"] is True
    # sha256 与文件一致
    import hashlib

    for s in m["shards"]:
        assert hashlib.sha256((tmp_path / "fw" / s["file"]).read_bytes()).hexdigest() == s["sha256"]


def test_target_bytes_stops_early(tmp_path: Path) -> None:
    m = download_source(spec(target_bytes=100), tmp_path, rows=fake_rows(10), log=lambda s: None)
    assert m["complete"] is False
    assert sum(s["bytes"] for s in m["shards"]) >= 100


def test_swh_content_fetch(tmp_path: Path) -> None:
    rows = [{"blob_id": "a", "text": None}, {"blob_id": "b", "text": None}]
    s = spec(name="se", registry="fineweb-edu", swh_content=True, id_field="blob_id")
    download_source(
        s, tmp_path, rows=rows, fetch=lambda b: None if b == "b" else "print(1)", log=lambda x: None
    )
    recs = read_all(tmp_path / "se")
    assert [(r["id"], r["text"]) for r in recs] == [("a", "print(1)")]


def test_license_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    from zero.data import sources

    fake = sources.DatasetSource(
        name="fake-unverified", url="u", license="待核实", languages=("en",), stage="pretrain"
    )
    monkeypatch.setitem(sources.SOURCES, "fake-unverified", fake)
    assert check_license(spec()) == "ODC-By-1.0"
    with pytest.raises(PermissionError):
        check_license(spec(registry="fake-unverified"))
    assert "待核实" in check_license(spec(registry="fake-unverified"), allow_unverified=True)
    with pytest.raises(KeyError):
        check_license(spec(registry="not-registered"))


def test_plan_targets() -> None:
    t = plan_targets(
        {"en": 3, "zh": 1},
        token_budget=1e9,
        bytes_per_token={"en": 4.0, "zh": 2.0},
        keep_rate={"zh": 0.5},
        margin=1.0,
    )
    assert t == {"en": int(0.75e9 * 4), "zh": int(0.25e9 * 2 / 0.5)}


def test_main_config_download_specs_parse() -> None:
    with open(REPO / "configs" / "main" / "data.toml", "rb") as f:
        cfg = tomllib.load(f)
    specs = specs_from_config(cfg)
    assert specs, "configs/main/data.toml 里应当有 [sources.download]"
    for s in specs:
        assert s.repo.count("/") == 1
        check_license(s, allow_unverified=True)  # 每个来源都登记过
    with pytest.raises(ValueError):
        DownloadSpec.from_dict("x", "fineweb-edu", {"repo": "a/b", "bogus": 1})


def test_shard_writer_rotation(tmp_path: Path) -> None:
    w = ShardWriter(tmp_path, spec(docs_per_shard=2), "ODC-By-1.0")
    for i in range(5):
        w.add(make_record({"text": f"t{i}"}, spec(), i))
    m = w.finish(complete=True)
    assert [s["docs"] for s in m["shards"]] == [2, 2, 1]
    assert not list((tmp_path / "fw").glob("*.tmp"))
