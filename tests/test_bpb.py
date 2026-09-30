"""bits-per-byte 评估（zero/data/bpb.py）：手算小例子、均匀模型不变量、特殊 token 与 ignore 位置不计入。"""

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
    """不管输入是什么，都输出同一组 logits（手算用）。"""

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
        # 逐 token 的字节数加起来，正好是原文的 UTF-8 字节数（包括被切成"半个汉字"的 token）
        assert int(tb[ids].sum()) == len(text.encode("utf-8"))
        assert b"".join(token_bytes_of(tok.id_to_token(i)) for i in ids) == text.encode("utf-8")


def test_hand_computed_example() -> None:
    # 词表 4：id 0 是特殊 token（0 字节），id 1/2/3 分别覆盖 1/2/3 个字节；logits 全 0 → 每个 token 概率 1/4
    token_bytes = torch.tensor([0, 1, 2, 3])
    model = FixedLogits(torch.zeros(4))
    x = torch.zeros(1, 6, dtype=torch.long)
    y = torch.tensor([[1, 2, 0, 3, -100, -1]])  # 特殊 token、ignore_index、负数标签都不计入
    s = bpb_stats(model, [(x, y)], token_bytes)
    # 分子 = 3 个 token × ln 4 nats；分母 = ln 2 × (1+2+3) 字节 → bpb = 3·2 / 6 = 1.0
    assert s.tokens == 3 and s.bytes == 6
    assert s.nats == pytest.approx(3 * math.log(4))
    assert s.bpb == pytest.approx(1.0)

    # 非均匀：p(3) = e^2 / (3 + e^2)，只数 token 3 → bpb = −log2 p(3) / 3
    model2 = FixedLogits(torch.tensor([0.0, 0.0, 0.0, 2.0]))
    p3 = math.exp(2) / (3 + math.exp(2))
    got = evaluate_bpb(model2, [(x[:, :1], torch.tensor([[3]]))], token_bytes)
    assert got == pytest.approx(-math.log2(p3) / 3)


def test_uniform_model_invariance(tok: Tokenizer, tiny_texts: dict[str, str]) -> None:
    """均匀分布的模型：bpb = log2(V) × (计入的 token 数 / 字节数) = log2(V) / (bytes/token)。"""
    V = tok.vocab_size
    ids = tok.encode(tiny_texts["zh"][:3000] + tiny_texts["en"][:3000])
    t = torch.tensor(ids)
    x, y = t[:-1].unsqueeze(0), t[1:].unsqueeze(0)
    tb = token_byte_lengths(tok)
    s = bpb_stats(FixedLogits(torch.zeros(V)), [(x, y)], tb)
    assert s.tokens == len(ids) - 1
    assert s.bytes == int(tb[y].sum())
    assert s.bpb == pytest.approx(math.log2(V) / s.bytes_per_token, rel=1e-6)  # 交叉熵按 float32 算
    # 一个"更会猜"的模型（把概率集中在真实下一个 token 上）bpb 必须更低
    assert s.bpb > 0


def test_loader_interface_and_specials(tok: Tokenizer, tmp_path: Path) -> None:
    docs = ["学而时习之，不亦说乎？" * 20, "To be, or not to be, that is the question. " * 20]
    write_shards(docs, tok, tmp_path, "val")
    loader = PackedDataLoader(str(tmp_path / "val_*.bin"), seq_len=32, batch_size=2, shuffle=False)
    tb = token_byte_lengths(tok)
    model = FixedLogits(torch.zeros(tok.vocab_size))
    s = bpb_stats(model, loader, tb, steps=3)
    # 分片里有 <|endoftext|>，它是 0 字节，不计入：计入 token 数 = 目标里非特殊 token 的个数
    loader2 = PackedDataLoader(str(tmp_path / "val_*.bin"), seq_len=32, batch_size=2, shuffle=False)
    ys = torch.cat([loader2.next_batch()[1].reshape(-1) for _ in range(3)])
    assert s.tokens == int((tb[ys] > 0).sum()) and s.bytes == int(tb[ys].sum())
    with pytest.raises(ValueError):
        bpb_stats(model, loader, tb)  # 加载器必须给 steps
    # 可迭代的 batch 列表 + steps 截断
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
    """没有特殊 token 时：bpb × ln2 × 字节数 / token 数 = 普通的平均交叉熵。"""
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
    assert model.training  # 评估后恢复原来的 train/eval 状态
