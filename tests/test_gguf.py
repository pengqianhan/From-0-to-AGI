"""GGUF 导出：llama.cpp 官方转换脚本 + 分词对拍 + 贪心生成对拍 + 量化（第 20 章）。

需要本地已有 llama.cpp 仓库（默认 ~/.cache/zero/llama.cpp，或环境变量 ZERO_LLAMA_CPP）；
没有就跳过（测试不联网）。编译好的 llama-tokenize / llama-simple / llama-quantize 存在时再做对拍和量化。
"""

from __future__ import annotations

import pytest
import torch

from zero.export import gguf

LLAMA = gguf.default_llama_cpp_dir()
pytestmark = pytest.mark.skipif(
    not (LLAMA / "convert_hf_to_gguf.py").exists(), reason=f"没有 llama.cpp 仓库（{LLAMA}）"
)
pytest.importorskip("sentencepiece")  # 官方转换脚本 import 它


@pytest.fixture(scope="module")
def exported(tmp_path_factory, chat_tok):  # noqa: ANN001, ANN201
    from zero.config import ModelConfig
    from zero.hf import export_to_hf_qwen3
    from zero.model import Transformer

    torch.manual_seed(0)
    cfg = ModelConfig(
        vocab_size=chat_tok.vocab_size,
        dim=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        ffn_dim=128,
        max_seq_len=512,
        tie_embeddings=False,
        init_std=0.1,
        rope_scaling={"type": "yarn", "factor": 2.0, "original_max_position_embeddings": 256},
    )
    model = Transformer(cfg).eval()
    d = tmp_path_factory.mktemp("gguf")
    hf = export_to_hf_qwen3(
        model, None, d / "hf", tokenizer=chat_tok, dtype=torch.float32, chat=True
    )
    f32 = gguf.convert_hf_to_gguf(hf, d / "m-f32.gguf", "f32")
    return model, f32, d


def test_convert_produces_gguf(exported) -> None:  # noqa: ANN001
    _, f32, _ = exported
    assert f32.read_bytes()[:4] == b"GGUF"


@pytest.mark.skipif(gguf.find_binary("llama-tokenize") is None, reason="llama.cpp 未编译")
def test_llama_cpp_tokenizer_matches_ours(exported, chat_tok) -> None:  # noqa: ANN001
    _, f32, _ = exported
    for text in [
        "<|im_start|>user\n北京今天天气怎么样？ What's 3+4?<|im_end|>\n<|im_start|>assistant\n",
        '<tool_call>\n{"name": "calculator", "arguments": {"expression": "12 * (3 + 4)"}}\n</tool_call>',
        "  Hello,   world!\n\n\tdef f(x): return x**2  # 注释 1234567",
    ]:
        assert gguf.llama_tokenize(f32, text) == chat_tok.encode(text), text


@pytest.mark.skipif(gguf.find_binary("llama-simple") is None, reason="llama.cpp 未编译")
def test_llama_cpp_greedy_matches_zero(exported, chat_tok) -> None:  # noqa: ANN001
    from zero.generate import generate

    model, f32, _ = exported
    prompt = "<|im_start|>user\n你好，今天星期几？<|im_end|>\n<|im_start|>assistant\n"
    ours = generate(model, chat_tok.encode(prompt), 12, temperature=0.0)
    out = gguf.run_llama(f32, prompt, 12)
    assert out.strip().startswith(prompt.strip()[:10])
    assert chat_tok.decode(ours).strip() in out  # f32 GGUF 的贪心输出与 zero 逐 token 一致


@pytest.mark.skipif(gguf.find_binary("llama-quantize") is None, reason="llama.cpp 未编译")
@pytest.mark.parametrize("qtype", ["Q8_0", "Q4_K_M"])
def test_quantize_and_run(exported, qtype: str) -> None:  # noqa: ANN001
    _, f32, d = exported
    q = gguf.quantize(f32, d / f"m-{qtype}.gguf", qtype)
    assert q.stat().st_size < f32.stat().st_size
    assert gguf.run_llama(q, "hello", 4)
