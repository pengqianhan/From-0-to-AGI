"""Bits-per-byte evaluation (zero/data/bpb.py): small hand-computed examples, the uniform-model invariant, special tokens and ignore positions are not counted."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from zero.data.bpb import BpbStats, bpb_stats, evaluate_bpb, token_byte_lengths, token_bytes_of
from zero.data.loader import PackedDataLoader
from zero.data.shard import write_shards
from zero.tokenizer import Tokenizer, train_bpe


@pytest.fixture(scope="module")
def tok(tiny_texts: dict[str, str]) -> Tokenizer:
    return train_bpe(list(tiny_texts.values()), vocab_size=600)


class FixedLogits(torch.nn.Module):
    """Output the same logits for any input (for hand computations)."""

    def __init__(self, logits: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("row", logits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.row.expand(*x.shape, -1).clone()


TEXTS = [
    "学而时习之，不亦说乎？",
    "To be, or not to be.",
    "def f(x):\n    return x + 1\n",
    "🤖 café",
]


def test_token_byte_lengths_match_utf8(tok: Tokenizer) -> None:
    tb = token_byte_lengths(tok)
    assert tb.shape == (tok.vocab_size,)
    assert all(int(tb[i]) == 0 for i in tok.special_tokens.values())
    assert int(tb.min()) >= 0 and int(tb[[i for i in range(tok.vocab_size) if i >= 16]].min()) >= 1
    for text in TEXTS:
        ids = tok.encode(text)
        # The sum of the bytes of each token is exactly the UTF-8 byte count of the text (also for tokens that hold "half a Chinese character")
        assert int(tb[ids].sum()) == len(text.encode("utf-8"))
        assert b"".join(token_bytes_of(tok.id_to_token(i)) for i in ids) == text.encode("utf-8")


def test_hand_computed_example() -> None:
    # Vocabulary of 4: id 0 is a special token (0 bytes); ids 1/2/3 cover 1/2/3 bytes. All logits are 0 → each token has probability 1/4
    token_bytes = torch.tensor([0, 1, 2, 3])
    model = FixedLogits(torch.zeros(4))
    x = torch.zeros(1, 6, dtype=torch.long)
    y = torch.tensor([[1, 2, 0, 3, -100, -1]])  # special tokens, ignore_index, and negative labels are not counted
    s = bpb_stats(model, [(x, y)], token_bytes)
    # numerator = 3 tokens × ln 4 nats; denominator = ln 2 × (1+2+3) bytes → bpb = 3·2 / 6 = 1.0
    assert s.tokens == 3 and s.bytes == 6
    assert s.nats == pytest.approx(3 * math.log(4))
    assert s.bpb == pytest.approx(1.0)

    # Not uniform: p(3) = e^2 / (3 + e^2); count only token 3 → bpb = −log2 p(3) / 3
    model2 = FixedLogits(torch.tensor([0.0, 0.0, 0.0, 2.0]))
    p3 = math.exp(2) / (3 + math.exp(2))
    got = evaluate_bpb(model2, [(x[:, :1], torch.tensor([[3]]))], token_bytes)
    assert got == pytest.approx(-math.log2(p3) / 3)


def test_uniform_model_invariance(tok: Tokenizer, tiny_texts: dict[str, str]) -> None:
    """A model with a uniform distribution: bpb = log2(V) × (counted tokens / bytes) = log2(V) / (bytes/token)."""
    V = tok.vocab_size
    ids = tok.encode(tiny_texts["zh"][:3000] + tiny_texts["en"][:3000])
    t = torch.tensor(ids)
    x, y = t[:-1].unsqueeze(0), t[1:].unsqueeze(0)
    tb = token_byte_lengths(tok)
    s = bpb_stats(FixedLogits(torch.zeros(V)), [(x, y)], tb)
    assert s.tokens == len(ids) - 1
    assert s.bytes == int(tb[y].sum())
    assert s.bpb == pytest.approx(math.log2(V) / s.bytes_per_token, rel=1e-6)  # the cross-entropy is computed in float32
    # A model that "predicts better" (more probability on the true next token) must have a lower bpb
    assert s.bpb > 0


def test_loader_interface_and_specials(tok: Tokenizer, tmp_path: Path) -> None:
    docs = ["学而时习之，不亦说乎？" * 20, "To be, or not to be, that is the question. " * 20]
    write_shards(docs, tok, tmp_path, "val")
    loader = PackedDataLoader(str(tmp_path / "val_*.bin"), seq_len=32, batch_size=2, shuffle=False)
    tb = token_byte_lengths(tok)
    model = FixedLogits(torch.zeros(tok.vocab_size))
    s = bpb_stats(model, loader, tb, steps=3)
    # The shards contain <|endoftext|>. It has 0 bytes and is not counted: counted tokens = non-special tokens in the targets
    loader2 = PackedDataLoader(str(tmp_path / "val_*.bin"), seq_len=32, batch_size=2, shuffle=False)
    ys = torch.cat([loader2.next_batch()[1].reshape(-1) for _ in range(3)])
    assert s.tokens == int((tb[ys] > 0).sum()) and s.bytes == int(tb[ys].sum())
    with pytest.raises(ValueError):
        bpb_stats(model, loader, tb)  # a loader needs steps
    # an iterable list of batches + a cut at steps
    batches = [(torch.zeros(1, 4, dtype=torch.long), torch.tensor([[20, 21, 22, 23]]))] * 5
    assert bpb_stats(model, batches, tb, steps=2).tokens == 8


def test_bpbstats_properties() -> None:
    s = BpbStats(nats=math.log(2) * 10, bytes=5, tokens=4)
    assert s.bpb == pytest.approx(2.0)
    assert s.bytes_per_token == pytest.approx(1.25)
    assert BpbStats(0.0, 0, 0).bpb == float("inf")


def test_matches_mean_cross_entropy_on_real_model(
    tok: Tokenizer, tiny_texts: dict[str, str]
) -> None:
    """Without special tokens: bpb × ln2 × bytes / tokens = the usual mean cross-entropy."""
    from zero.config import ModelConfig
    from zero.model import Transformer

    torch.manual_seed(0)
    cfg = ModelConfig(
        vocab_size=tok.vocab_size,
        dim=32,
        n_layers=1,
        n_heads=2,
        n_kv_heads=1,
        head_dim=16,
        ffn_dim=64,
        max_seq_len=64,
    )
    model = Transformer(cfg)
    ids = torch.tensor(tok.encode(tiny_texts["en"][:2000])[:65])
    x, y = ids[:-1].unsqueeze(0), ids[1:].unsqueeze(0)
    tb = token_byte_lengths(tok)
    s = bpb_stats(model, [(x, y)], tb)
    with torch.no_grad():
        ce = torch.nn.functional.cross_entropy(model(x).reshape(-1, tok.vocab_size), y.reshape(-1))
    assert s.bpb * math.log(2) * s.bytes / s.tokens == pytest.approx(float(ce), rel=1e-5)
    assert model.training  # after the evaluation, the original train/eval mode is restored
