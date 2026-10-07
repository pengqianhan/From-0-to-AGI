"""After export to the Hugging Face format and a load with transformers, the logits must be the same (Chapter 20)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

transformers = pytest.importorskip("transformers")

from zero.config import ModelConfig  # noqa: E402
from zero.hf import export_to_hf_qwen3, load_from_hf_qwen3  # noqa: E402
from zero.model import Transformer  # noqa: E402
from zero.tokenizer import train_bpe  # noqa: E402


def _model(**kw) -> Transformer:
    cfg = dict(
        vocab_size=300,
        dim=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        ffn_dim=96,
        max_seq_len=128,
        init_std=0.08,
    )
    cfg.update(kw)
    torch.manual_seed(0)
    m = Transformer(ModelConfig(**cfg)).eval()
    with torch.no_grad():  # make the RMSNorm weights not all 1, so the parity check is stricter
        for p in m.parameters():
            if p.dim() == 1:
                p.uniform_(0.5, 1.5)
    return m


@pytest.mark.parametrize(
    "kw",
    [
        {},
        {"tie_embeddings": False},
        {
            "rope_scaling": {"type": "yarn", "factor": 4.0, "original_max_position_embeddings": 32},
            "rope_theta": 5e4,
        },
    ],
    ids=["tied", "untied", "yarn"],
)
def test_export_then_load_with_transformers(tmp_path: Path, kw: dict) -> None:
    model = _model(**kw)
    out = export_to_hf_qwen3(model, None, tmp_path, dtype=torch.float32)
    cfg = json.loads((out / "config.json").read_text())
    assert cfg["model_type"] == "qwen3" and cfg["architectures"] == ["Qwen3ForCausalLM"]
    hf = transformers.AutoModelForCausalLM.from_pretrained(str(out), dtype=torch.float32).eval()
    tokens = torch.randint(0, 300, (2, 100), generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        torch.testing.assert_close(hf(tokens).logits, model(tokens), rtol=1e-5, atol=1e-5)
    # zero can also read the directory back directly
    back = load_from_hf_qwen3(out).eval()
    with torch.no_grad():
        torch.testing.assert_close(back(tokens), model(tokens))


def test_export_bf16_with_tokenizer(tmp_path: Path, tiny_texts: dict[str, str]) -> None:
    tok = train_bpe([tiny_texts["en"][:20000], tiny_texts["zh"][:10000]], vocab_size=300)
    model = _model()
    out = export_to_hf_qwen3(model, model.config, tmp_path, tokenizer=tok)  # default bf16
    assert {
        "tokenizer.json",
        "tokenizer_config.json",
        "generation_config.json",
        "model.safetensors",
    } <= {p.name for p in out.iterdir()}
    hf = transformers.AutoModelForCausalLM.from_pretrained(str(out))
    hf_tok = transformers.AutoTokenizer.from_pretrained(str(out))
    text = "学而时习之 To be or not"
    ids = hf_tok(text, return_tensors="pt", add_special_tokens=False).input_ids
    assert ids[0].tolist() == tok.encode(text)
    with torch.no_grad():
        ref = model(ids)
        got = hf(ids).logits.float()
    assert (got - ref).abs().max() < 0.1  # bf16 weights: only approximate agreement is required
    assert got[0, -1].argmax() == ref[0, -1].argmax() or (got - ref).abs().max() < 0.05
    assert hf.config.eos_token_id == tok.eos_id
