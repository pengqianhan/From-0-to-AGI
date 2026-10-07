"""Correctness tests for the needle-in-a-haystack tool (zero/tools/needle.py, Chapter 15)."""

from __future__ import annotations

import re

import pytest
import torch

from zero.config import ModelConfig
from zero.model import Transformer
from zero.tools.needle import answer_nll, format_grid, make_case, run_grid, score_text


@pytest.mark.parametrize("lang", ["en", "zh"])
@pytest.mark.parametrize("length", [96, 200])
def test_case_length_and_needle_position(chat_tok, lang: str, length: int) -> None:
    for depth in [0.0, 0.5, 1.0]:
        case = make_case(chat_tok, length, depth, seed=3, lang=lang)
        assert len(case.prompt_ids) == length == len(case.control_ids)
        text = chat_tok.decode(case.prompt_ids)
        assert case.answer in text and case.key in text
        # The control prompt has no correct answer, only the decoy. All tokens except the number
        # in the needle are the same.
        ctrl = chat_tok.decode(case.control_ids)
        assert case.answer not in ctrl and case.decoy in ctrl
        diff = [i for i, (a, b) in enumerate(zip(case.prompt_ids, case.control_ids)) if a != b]
        assert diff and min(diff) >= case.needle_pos
    # At depth 0, the needle is at the start. At depth 1, it is at the end of the haystack
    # (immediately before the question). A larger depth gives a later position.
    pos = [make_case(chat_tok, 200, d, seed=3).needle_pos for d in (0.0, 0.25, 0.5, 0.75, 1.0)]
    assert pos[0] == 0 and pos == sorted(pos) and pos[-1] > pos[1]


def test_deterministic_and_too_short(chat_tok) -> None:
    a = make_case(chat_tok, 128, 0.5, seed=11)
    b = make_case(chat_tok, 128, 0.5, seed=11)
    assert a.prompt_ids == b.prompt_ids and a.answer == b.answer
    assert make_case(chat_tok, 128, 0.5, seed=12).answer != a.answer
    with pytest.raises(ValueError):
        make_case(chat_tok, 10, 0.5)
    with pytest.raises(ValueError):
        make_case(chat_tok, 128, 1.5)


def test_custom_haystack(chat_tok, tiny_texts) -> None:
    case = make_case(chat_tok, 256, 0.3, seed=1, haystack_text=tiny_texts["en"][:5000])
    assert len(case.prompt_ids) == 256
    assert "grass is green" not in chat_tok.decode(case.prompt_ids)


def test_score_text() -> None:
    assert score_text(" 1234567. And then", "1234567") == 1.0
    assert score_text(" 123456", "1234567") == 0.0


def test_grid_with_oracle_and_blank_generators(chat_tok) -> None:
    """Test the scoring path with fake generators: an "oracle" that reads the needle gets the full score; an empty output gets 0."""

    def oracle(prompt_ids: list[int]) -> str:
        text = chat_tok.decode(prompt_ids)
        return " " + re.findall(r"is (\d{7})\.", text)[0]

    res = run_grid(None, chat_tok, [96, 160], [0.0, 1.0], n=3, generate_fn=oracle, with_nll=False)
    assert len(res) == 4 and all(r.accuracy == 1.0 for r in res)
    res0 = run_grid(None, chat_tok, [96], [0.5], n=2, generate_fn=lambda ids: "", with_nll=False)
    assert res0[0].accuracy == 0.0
    assert "length" in format_grid(res)


def test_answer_nll_matches_manual_and_runs_on_zero_model(chat_tok) -> None:
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=chat_tok.vocab_size, dim=32, n_layers=2, n_heads=4, n_kv_heads=2,
                      ffn_dim=64, max_seq_len=160)  # fmt: skip
    model = Transformer(cfg).eval()
    case = make_case(chat_tok, 120, 0.5, seed=5)
    nll = answer_nll(model, case.prompt_ids, case.answer_ids)
    # Manual calculation: log softmax token by token
    ids = torch.tensor([case.prompt_ids + case.answer_ids])
    with torch.no_grad():
        logp = model(ids).log_softmax(-1)[0]
    n = len(case.answer_ids)
    manual = (
        -sum(logp[len(case.prompt_ids) - 1 + j, case.answer_ids[j]].item() for j in range(n)) / n
    )
    assert abs(nll - manual) < 1e-4
    # One pass with a real zero model (random initialization: accuracy is 0, but the pipeline must run)
    res = run_grid(model, chat_tok, [120], [0.0, 1.0], n=1, max_new_tokens=8)
    assert all(0.0 <= r.accuracy <= 1.0 and r.nll > 0 and r.nll_control > 0 for r in res)
