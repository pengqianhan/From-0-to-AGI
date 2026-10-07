"""Data pipeline: cleaning, deduplication, quality filters, decontamination, shards and loaders, multi-source mixture."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from zero.data.clean import clean_document, normalize_text, split_into_documents
from zero.data.decontam import decontaminate, find_contamination
from zero.data.dedup import MinHasher, estimate_jaccard, exact_dedup, jaccard, near_dedup, shingles
from zero.data.loader import PackedDataLoader
from zero.data.mixture import MixtureLoader, MixtureSampler
from zero.data.quality import KeywordClassifier, filter_by_classifier, quality_check
from zero.data.shard import load_metadata, read_shard, write_shards
from zero.data.sources import get_source, list_sources, unverified_licenses
from zero.tokenizer import train_bpe

# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------


def test_normalize_idempotent() -> None:
    raw = "Café\r\nline  \t\n\n\n\n​next　word\x00<|endoftext|>"
    once = normalize_text(raw)
    assert once == normalize_text(once)
    assert "\r" not in once and "\x00" not in once and "​" not in once and "\n\n\n" not in once
    assert once.startswith("Café")
    assert "<|endoftext|>" not in clean_document(raw)
    assert clean_document("   ") is None


def test_split_into_documents() -> None:
    text = "\n\n".join(f"para {i} " + "x" * 50 for i in range(20))
    docs = split_into_documents(text, doc_chars=200)
    assert len(docs) > 3
    assert "\n\n".join(docs) == text


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


def _article(seed: int, n: int = 200) -> str:
    # "Words" of random letters: texts with different seeds have almost no common character 5-grams
    rng = np.random.default_rng(seed)
    letters = np.array(list("abcdefghijklmnopqrstuvwxyz"))
    words = ["".join(rng.choice(letters, size=rng.integers(3, 9))) for _ in range(n)]
    words[0] = "alpha"
    return " ".join(words)


def test_exact_dedup() -> None:
    texts = ["a b c", "a b c  ", "x", "a b c"]
    assert exact_dedup(texts) == [0, 2]  # after normalization, documents 2 and 4 are the same as document 1


def test_minhash_estimates_jaccard() -> None:
    a, b = _article(0), _article(0)[:-150] + _article(1)[:150]
    sa, sb = shingles(a), shingles(b)
    h = MinHasher(num_perm=256)
    est = estimate_jaccard(h.signature(a), h.signature(b))
    assert abs(est - jaccard(sa, sb)) < 0.1


def test_near_dedup_finds_modified_copies() -> None:
    base = [_article(i) for i in range(10)]
    near_copy = base[3].replace("alpha", "ALPHA", 1) + " 结尾多了几个字"
    texts = [*base, near_copy, base[7] + " !"]
    keep, clusters = near_dedup(texts, threshold=0.7)
    assert 10 not in keep and 11 not in keep
    assert sorted(keep) == list(range(10))
    assert [3, 10] in clusters and [7, 11] in clusters


# ---------------------------------------------------------------------------
# Quality filters
# ---------------------------------------------------------------------------

GOOD_EN = (
    "The history of the printing press is a story of how ideas spread. Before it was invented, books were "
    "copied by hand, which made them rare and expensive. With the press, a single workshop could produce "
    "hundreds of copies in the time it once took to make one.\n"
    "This change had effects that reached far beyond the workshop. Literacy rose, and people began to share "
    "arguments about science and religion with a speed that had never been possible.\n"
    "Historians still debate how quickly these changes happened, but most agree that the press was one of the "
    "most important inventions of its age, and that its influence can still be felt today."
)
GOOD_ZH = (
    "印刷术的发明改变了知识传播的方式。在此之前，书籍都要靠人工抄写，数量少而且价格昂贵。"
    "有了印刷术以后，一个作坊在很短的时间里就可以印出成百上千本书。\n"
    "这种变化的影响远远超出了作坊本身。识字的人越来越多，人们也开始更快地交流关于科学和思想的争论。\n"
    "历史学家对这些变化发生得有多快仍有争议，但大多数人都同意，印刷术是那个时代最重要的发明之一。"
)


def test_quality_good_documents_pass() -> None:
    for doc in (GOOD_EN, GOOD_ZH):
        r = quality_check(doc)
        assert r.keep, r.reasons


def test_quality_bad_documents_fail() -> None:
    assert "too_few_words" in quality_check("Click here. Login.").reasons
    menu = "\n".join(["Home", "About us", "Contact", "Products", "Login", "Sign up"] * 20)
    r = quality_check(menu)
    assert not r.keep and "dup_lines" in r.reasons
    spam = "buy now " * 200
    assert "top_2gram" in quality_check(spam).reasons
    symbols = " ".join(["#tag"] * 100)
    assert "symbol_word_ratio" in quality_check(symbols).reasons


def test_classifier_interface() -> None:
    texts = ["because of this, for example, therefore " * 5, "nothing to see here " * 20]
    assert filter_by_classifier(texts, KeywordClassifier(), threshold=1.0) == [0]


# ---------------------------------------------------------------------------
# Decontamination
# ---------------------------------------------------------------------------


def test_decontamination() -> None:
    question = (
        "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. "
        "How many clips did Natalia sell altogether in April and May?"
    )
    short_zh = "床前明月光，疑是地上霜"
    train = [
        "Some unrelated text about cooking pasta and tomatoes for dinner tonight with friends and family.",
        "Forum post: natalia SOLD clips to 48 of her friends in april, and then she sold half as many... cool",
        "唐诗选读：床前明月光，疑是地上霜。举头望明月，低头思故乡。",
        "Natalia sold clips.",  # too short for a 13-gram, and it does not contain the short item
    ]
    report = find_contamination(train, {"gsm8k": [question], "zh": [short_zh]}, n=13)
    assert report.contaminated_docs == [1, 2]
    assert report.eval_items_hit("gsm8k") == [0]
    keep, _ = decontaminate(train, {"gsm8k": [question], "zh": [short_zh]})
    assert keep == [0, 3]
    assert "2/4" in report.summary()


# ---------------------------------------------------------------------------
# Shards + loader
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tok(tiny_texts: dict[str, str]):  # noqa: ANN201
    return train_bpe([tiny_texts["en"][:20000], tiny_texts["zh"][:10000]], vocab_size=600)


def test_shard_roundtrip(tok, tmp_path: Path) -> None:  # noqa: ANN001
    docs = [f"文档 {i}：" + "学而时习之 hello world " * (i % 7 + 1) for i in range(300)]
    paths = write_shards(docs, tok, tmp_path, "demo", shard_tokens=1000, source="unit-test")
    assert len(paths) > 1 and all(p.name.startswith("demo_") for p in paths)
    meta = load_metadata(tmp_path / "demo.json")
    assert meta["num_documents"] == 300 and meta["tokenizer_hash"] == tok.hash()
    stream = np.concatenate([np.asarray(read_shard(p)) for p in paths])
    assert len(stream) == meta["num_tokens"] == sum(s["num_tokens"] for s in meta["shards"])
    # Split at <|endoftext|>: decoding each document must give the original text
    ends = np.flatnonzero(stream == tok.eot_id)
    assert len(ends) == 300
    starts = np.concatenate([[0], ends[:-1] + 1])
    decoded = [tok.decode(stream[s:e].tolist()) for s, e in zip(starts, ends)]
    assert decoded == docs
    assert load_metadata(paths[0])["name"] == "demo"


def test_loader_covers_every_chunk_once_per_epoch(tmp_path: Path, random_shards) -> None:  # noqa: ANN001
    pattern = random_shards(tmp_path, "d", n_shards=3, tokens_per_shard=1000 + 1, vocab=50, seed=0)
    ld = PackedDataLoader(pattern, seq_len=100, batch_size=5, seed=1)
    assert ld.total_chunks == 30
    seen = {ld.locate(g)[1:] for g in range(30)}
    assert len(seen) == 30  # in one epoch, each (shard, chunk) occurs exactly once
    x, y = ld.next_batch()
    assert x.shape == (5, 100) and y.shape == (5, 100)
    assert (x[:, 1:] == y[:, :-1]).all()  # y is x shifted by one position
    assert ld.locate(30)[0] == 1  # second epoch
    # The order of the second epoch is different from the first (a new shuffle)
    assert [ld.locate(g)[1:] for g in range(30)] != [ld.locate(30 + g)[1:] for g in range(30)]


def test_loader_resume_exact(tmp_path: Path, random_shards) -> None:  # noqa: ANN001
    pattern = random_shards(tmp_path, "d", n_shards=4, tokens_per_shard=3000, vocab=100, seed=1)
    ref = PackedDataLoader(pattern, seq_len=64, batch_size=4, seed=7)
    ref_batches = [ref.next_batch()[0] for _ in range(40)]  # across several epochs
    ld = PackedDataLoader(pattern, seq_len=64, batch_size=4, seed=7)
    for _ in range(13):
        ld.next_batch()
    state = ld.state_dict()
    ld2 = PackedDataLoader(pattern, seq_len=64, batch_size=4, seed=7)
    ld2.load_state_dict(state)
    for i in range(13, 40):
        assert (ld2.next_batch()[0] == ref_batches[i]).all()
    bad = PackedDataLoader(pattern, seq_len=32, batch_size=4, seed=7)
    with pytest.raises(ValueError):
        bad.load_state_dict(state)


def test_loader_distributed_sharding(tmp_path: Path, random_shards) -> None:  # noqa: ANN001
    """2 ranks with B rows each == 1 rank with 2B rows (the same global samples), and the two ranks do not overlap."""
    pattern = random_shards(tmp_path, "d", n_shards=3, tokens_per_shard=2000, vocab=100, seed=2)
    single = PackedDataLoader(pattern, seq_len=50, batch_size=6, seed=3)
    r0 = PackedDataLoader(pattern, seq_len=50, batch_size=3, rank=0, world_size=2, seed=3)
    r1 = PackedDataLoader(pattern, seq_len=50, batch_size=3, rank=1, world_size=2, seed=3)
    for _ in range(5):
        s = single.next_samples(6)
        a, b = r0.next_samples(3), r1.next_samples(3)
        as_set = lambda arr: {tuple(r) for r in arr.tolist()}  # noqa: E731
        assert as_set(s) == as_set(a) | as_set(b)
        assert not (as_set(a) & as_set(b))


def test_loader_no_shuffle_is_sequential(tmp_path: Path, random_shards) -> None:  # noqa: ANN001
    pattern = random_shards(tmp_path, "d", n_shards=1, tokens_per_shard=501, vocab=100, seed=4)
    ld = PackedDataLoader(pattern, seq_len=100, batch_size=5, shuffle=False)
    raw = np.fromfile(sorted(tmp_path.glob("d_*.bin"))[0], dtype=np.uint32)
    x, y = ld.next_batch()
    assert (x.numpy().reshape(-1) == raw[:500]).all()
    assert (y.numpy().reshape(-1) == raw[1:501]).all()


# ---------------------------------------------------------------------------
# Multi-source mixture
# ---------------------------------------------------------------------------


def test_mixture_sampler_deterministic_and_proportional() -> None:
    w = {"web": 0.6, "code": 0.3, "math": 0.1}
    a = MixtureSampler(w, seed=5).draw(20000)
    b = MixtureSampler(w, seed=5).draw(20000)
    assert a == b
    assert MixtureSampler(w, seed=6).draw(200) != a[:200]
    frac = {k: a.count(k) / len(a) for k in w}
    for k, v in w.items():
        assert abs(frac[k] - v) < 0.02
    # two draws == one draw of all rows
    s = MixtureSampler(w, seed=5)
    assert s.draw(1500) + s.draw(18500) == a


def test_mixture_loader_resume(tmp_path: Path, random_shards) -> None:  # noqa: ANN001
    pa = random_shards(tmp_path, "a", n_shards=2, tokens_per_shard=2000, vocab=100, seed=10)
    pb = random_shards(tmp_path, "b", n_shards=1, tokens_per_shard=2000, vocab=100, seed=11)

    def make() -> MixtureLoader:
        loaders = {
            "a": PackedDataLoader(pa, seq_len=32, batch_size=1, seed=1),
            "b": PackedDataLoader(pb, seq_len=32, batch_size=1, seed=2),
        }
        return MixtureLoader(loaders, {"a": 0.7, "b": 0.3}, batch_size=4, seed=9)

    ref = make()
    ref_batches = [ref.next_batch()[0] for _ in range(30)]
    m = make()
    for _ in range(11):
        m.next_batch()
    state = m.state_dict()
    m2 = make()
    m2.load_state_dict(state)
    for i in range(11, 30):
        assert (m2.next_batch()[0] == ref_batches[i]).all()
    assert sum(ref.counts.values()) == 120 and ref.counts["a"] > ref.counts["b"]


def test_sources_registry() -> None:
    assert get_source("fineweb-edu").license == "ODC-By-1.0"
    assert all(s.stage == "pretrain" for s in list_sources(stage="pretrain"))
    assert "stack-edu" in unverified_licenses()
    with pytest.raises(KeyError):
        get_source("nope")


def test_near_dedup_parallel_matches_serial() -> None:
    # Signatures in multiple processes ([dedup] n_jobs of the pipeline) are only faster: the kept indices and duplicate clusters must be identical to one process
    rng = np.random.default_rng(0)
    words = [f"w{i}" for i in range(500)]
    base = [" ".join(rng.choice(words, 60)) for _ in range(2600)]
    texts = base + [t + " extra" for t in base[:300]]  # 300 near duplicates
    serial = near_dedup(texts, n_jobs=1)
    parallel = near_dedup(texts, n_jobs=4)
    assert serial == parallel
    assert len(serial[1]) >= 250  # most near duplicates are found
