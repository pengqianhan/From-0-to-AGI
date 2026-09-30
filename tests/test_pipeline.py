"""数据流水线（zero/data/pipeline.py）：各阶段的纯函数 + 小语料端到端 + 预切分正则预设。"""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

import numpy as np
import pytest

from zero.data.loader import PackedDataLoader
from zero.data.pipeline import (
    PRETOKENIZE_PRESETS,
    DedupSpec,
    SourceSpec,
    code_quality_reasons,
    config_from_dict,
    decontam_stage,
    dedup_stage,
    heuristic_stage,
    langid_stage,
    load_pipeline_config,
    run_pipeline,
    sample_for_tokenizer,
    score_stage,
    split_train_val,
    train_tokenizer,
)
from zero.data.quality import QualityThresholds
from zero.tokenizer import PRETOKENIZE_REGEX, Tokenizer

REPO = Path(__file__).resolve().parent.parent

PROSE = (
    "The quick brown fox jumps over the lazy dog, and that is the whole story of the day. "
    "It was a cold morning with a bright sun over the hills. We walked to the river and sat "
    "down to watch the boats go by, talking about the books we had read and the places we "
    "wanted to see. Nobody was in a hurry to go back home.\n"
)


def doc(i: int, text: str, **kw) -> dict:  # noqa: ANN003
    return {"id": f"d{i}", "text": text, "source": "s", **kw}


# ---------------------------------------------------------------------------
# 分词器：预切分正则预设
# ---------------------------------------------------------------------------


def test_qwen2_preset_is_zero_default(tiny_texts: dict[str, str]) -> None:
    assert PRETOKENIZE_PRESETS["qwen2"] == PRETOKENIZE_REGEX == PRETOKENIZE_PRESETS["qwen3"]
    a = train_tokenizer(list(tiny_texts.values()), 400, "qwen2")
    from zero.tokenizer import train_bpe

    assert a.hash() == train_bpe(list(tiny_texts.values()), 400).hash()


def test_qwen35_preset_roundtrip_and_marks(tiny_texts: dict[str, str], tmp_path: Path) -> None:
    tok = train_tokenizer(list(tiny_texts.values()), 400, "qwen3.5")
    hindi = "सभी मनुष्यों को गौरव"
    for text in ["学而时习之，不亦说乎？", "def f(x):\n    return x + 1\n", hindi, "x = 2026"]:
        assert tok.decode(tok.encode(text)) == text
    assert len(tok.encode("2026")) == 4  # 数字仍然逐个切
    assert tok.special_id("<|im_start|>") == 1
    # 存取后正则还在（写在 tokenizer.json 里）
    p = tok.save(tmp_path / "t.json")
    assert "\\p{M}" in p.read_text() and Tokenizer.load(p).encode(hindi) == tok.encode(hindi)


# ---------------------------------------------------------------------------
# 各阶段
# ---------------------------------------------------------------------------


def test_langid_and_heuristics() -> None:
    zh = "学而时习之，不亦说乎？有朋自远方来，不亦乐乎？人不知而不愠，不亦君子乎？" * 3
    res = langid_stage([doc(0, zh), doc(1, PROSE)], "zh")
    assert [d["id"] for d in res.kept] == ["d0"] and res.removed["not_zh"] == 1
    nav = "\n".join(["Home | About | Contact | Login"] * 40)
    res = heuristic_stage(
        [doc(0, " ".join([PROSE.strip()] * 3)), doc(1, nav)], "web", QualityThresholds()
    )
    assert [d["id"] for d in res.kept] == ["d0"] and res.n_removed == 1
    assert heuristic_stage([doc(1, nav)], "none", QualityThresholds()).n_removed == 0


def test_code_rules() -> None:
    good = "def add(a, b):\n    return a + b\n\nprint(add(1, 2))\n"
    assert code_quality_reasons(good) == []
    assert "code_max_line_len" in code_quality_reasons("x = [" + "1, " * 600 + "]")
    assert "code_alnum_frac" in code_quality_reasons("{}[]();;;;!!!!@@@@####\n" * 5)


def test_score_stage_threshold() -> None:
    spec = SourceSpec(name="s", inputs=[], score_field="int_score", score_min=3)
    res = score_stage([doc(0, "a", int_score=4), doc(1, "b", int_score=2), doc(2, "c")], spec)
    assert [d["id"] for d in res.kept] == ["d0"] and res.n_removed == 2


def test_dedup_stage() -> None:
    a = PROSE * 4
    near = a.replace("cold", "warm").replace("river", "lake")
    other = "Completely different text about compilers and parsers and grammars. " * 12
    docs = [doc(0, a), doc(1, a + "  "), doc(2, near), doc(3, other)]
    res, clusters = dedup_stage(docs, DedupSpec())
    assert [d["id"] for d in res.kept] == ["d0", "d3"]
    assert res.removed == {"exact_duplicate": 1, "near_duplicate": 1}
    assert clusters == [["d0", "d2"]]


def test_decontam_stage_and_eval_loading(tmp_path: Path) -> None:
    item = "Which planet is known as the red planet and has two small moons named Phobos and Deimos"
    ev = tmp_path / "ev.jsonl"
    ev.write_text(json.dumps({"id": 1, "question": item, "answer": "Mars"}) + "\n")
    cfg = config_from_dict(
        {
            "sources": [{"name": "s", "inputs": ["x"]}],
            "decontam": {"n": 13, "eval_sets": [{"name": "toy", "path": str(ev)}]},
        }
    )
    from zero.data.pipeline import build_eval_index

    index, sizes = build_eval_index(cfg.decontam)
    assert sizes == {"toy": 1}
    leak = (
        PROSE
        + "Quiz: which PLANET is known as the red planet, and has two small moons named Phobos"
    )
    res, hits = decontam_stage([doc(0, PROSE), doc(1, leak)], index)
    assert [d["id"] for d in res.kept] == ["d0"]
    assert hits[0]["doc"] == "d1" and hits[0]["hits"][0]["eval_set"] == "toy"


def test_split_and_tokenizer_sampling() -> None:
    docs = [doc(i, "x" * 100) for i in range(40)]
    tr, va = split_train_val(docs, 0.1, 0, "s")
    assert len(va) == 4 and len(tr) == 36 and not {d["id"] for d in tr} & {d["id"] for d in va}
    assert split_train_val(docs, 0.1, 0, "s") == (tr, va)  # 确定性
    s = sample_for_tokenizer({"a": docs, "b": docs}, {"a": 0.75, "b": 0.25}, 2000, 0)
    assert len(s) == 15 + 5


def test_config_validation() -> None:
    with pytest.raises(ValueError):
        config_from_dict({"sources": []})
    with pytest.raises(ValueError):
        config_from_dict({"sources": [{"name": "a", "inputs": [], "bogus": 1}]})
    with pytest.raises(ValueError):
        config_from_dict({"sources": [{"name": "a", "inputs": []}, {"name": "a", "inputs": []}]})


def test_repo_configs_parse() -> None:
    tiny = load_pipeline_config(REPO / "configs" / "tiny" / "data.toml")
    assert {s.name for s in tiny.sources} == {"shakespeare", "chinese_poetry", "code"}
    main = load_pipeline_config(REPO / "configs" / "main" / "data.toml")
    assert abs(sum(s.weight for s in main.sources) - 1.0) < 1e-6
    assert main.tokenizer.pretokenize in PRETOKENIZE_PRESETS
    from zero.data.sources import SOURCES

    assert all(s.registry in SOURCES for s in main.sources)
    # 词表是 128 的倍数（GPU 上矩阵乘更整齐），且装得下 256 个字节 + 16 个特殊 token
    assert main.tokenizer.vocab_size % 128 == 0 and main.tokenizer.vocab_size > 272
    with open(REPO / "configs" / "main" / "pretrain.toml", "rb") as f:
        pre = tomllib.load(f)
    assert pre["model"]["vocab_size"] % 128 == 0


# ---------------------------------------------------------------------------
# 端到端（小语料，几秒）
# ---------------------------------------------------------------------------


def test_end_to_end_small(tmp_path: Path, tiny_texts: dict[str, str]) -> None:
    (tmp_path / "en.txt").write_text(tiny_texts["en"][:40_000])
    (tmp_path / "zh.txt").write_text(tiny_texts["zh"][:30_000])
    # 一个 JSONL 来源（download.py 的格式），带质量分数和一篇重复
    rows = [{"id": i, "text": PROSE * (2 + i % 3), "int_score": i % 5} for i in range(12)]
    rows.append({"id": 99, "text": PROSE * 2, "int_score": 4})  # 与 id=0 完全相同
    quiz = "Which planet is known as the red planet and has two small moons named Phobos and Deimos"
    rows.append({"id": 100, "text": PROSE + quiz + "?\n" + PROSE, "int_score": 4})  # 泄漏的考题
    (tmp_path / "web.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    ev = tmp_path / "ev.txt"
    ev.write_text(quiz + "\n")
    cfg = config_from_dict(
        {
            "pipeline": {"name": "t", "work_dir": str(tmp_path / "work"), "val_fraction": 0.1},
            "sources": [
                {
                    "name": "en",
                    "inputs": [str(tmp_path / "en.txt")],
                    "doc_chars": 2000,
                    "weight": 2,
                    "lang": "en",
                },
                {
                    "name": "zh",
                    "inputs": [str(tmp_path / "zh.txt")],
                    "doc_chars": 2000,
                    "weight": 1,
                    "lang": "zh",
                },
                {
                    "name": "web",
                    "inputs": [str(tmp_path / "web.jsonl")],
                    "format": "jsonl",
                    "weight": 1,
                    "heuristics": "none",
                    "score_field": "int_score",
                    "score_min": 2,
                },
            ],
            "decontam": {"eval_sets": [{"name": "ev", "path": str(ev)}]},
            "tokenizer": {"vocab_size": 400, "pretokenize": "qwen3.5"},
        }
    )
    m = run_pipeline(cfg, log=lambda s: None)
    assert m["mixture"]["en"]["weight"] == pytest.approx(0.5)
    assert m["decontam"]["contaminated_docs"] == {"en": 0, "zh": 0, "web": 1}
    web = {s["stage"]: s["docs"] for s in m["funnel"]["web"]}
    assert (
        web["raw"] == 14
        and web["decontam"] == web["dedup"] - 1
        and web["score"] < 14
        and web["dedup"] <= web["score"]
    )
    assert m["tokenizer"]["gguf_pre_tokenizer"] == "qwen35"
    tok = Tokenizer.load(m["tokenizer"]["path"])
    assert tok.hash() == m["tokenizer"]["hash"]
    shard_dir = Path(m["shards"]["dir"])
    loader = PackedDataLoader(str(shard_dir / "en_train_*.bin"), seq_len=16, batch_size=2)
    x, y = loader.next_batch()
    assert x.shape == (2, 16) and int(x.max()) < tok.vocab_size
    assert (Path(cfg.pipeline.work_dir) / "pretrain_sources.toml").exists()
    # 每一步的中间结果都落盘
    assert list((Path(cfg.pipeline.work_dir)).glob("*_dedup/en.jsonl"))
    assert np.fromfile(shard_dir / "zh_val_00000.bin", dtype=np.uint32).size > 0
