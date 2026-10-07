"""Chapter 20 · Minimal code 2: quantize a real small model and measure how much the loss changes

Model: the trained character-level small model of Chapter 10 (Pre-Norm RMSNorm, RoPE, SwiGLU,
shared embedding, dim=128, 4 layers, about 0.8M parameters). The first run trains it for about
1–2 minutes. Later runs load it from the cache.

We do the same as llama.cpp: quantize all 2D weight matrices (embedding, Q/K/V/O, FFN), and keep
the 1D RMSNorm weights in fp32. For each method, we quantize and dequantize the weights (fake quant),
run the validation set, and record:

- the validation loss and its change Δ from fp32;
- the top-1 agreement: at each position of the validation set, do the quantized model and the
  fp32 model give the same most probable next character?
- the weight size (from the bits/weight of each method);
- the first 60 characters of greedy generation, to look at the quality.

Run: uv run python chapters/20-release/code/02_quantize_tiny_model.py
"""

from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

torch.set_num_threads(1)  # many jobs share the CPU of the build machine; on your computer, you can remove this line

HERE = Path(__file__).resolve().parent
CH10 = HERE.parents[1] / "10-inference" / "code" / "01_tiny_model.py"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


tiny = _load("ch10_tiny_model", CH10)
quant = _load("ch20_blockwise_quant", HERE / "01_blockwise_quant.py")

# Methods: display name → quantization method from 01 (None = no quantization)
SCHEMES = [
    ("fp32", None),
    ("fp16", "fp16"),
    ("INT8, 1 scale/tensor", "int8-tensor"),
    ("INT8 block32 (≈Q8_0)", "int8-block32"),
    ("INT4, 1 scale/tensor", "int4-tensor"),
    ("INT4, 1 scale/row", "int4-row"),
    ("INT4 block32 (≈Q4_0)", "int4-block32"),
    ("INT4 2-level (≈Q4_K)", "int4-kquant"),
    ("INT3 block32", "int3-block32"),
    ("INT2 block32", "int2-block32"),
]


def bpw(scheme: str | None, shape: tuple[int, int]) -> float:
    if scheme is None:
        return 32.0
    return quant.bits_per_weight(scheme, shape)


def quantize_model(model: torch.nn.Module, scheme: str | None) -> tuple[torch.nn.Module, float]:
    """Return (a quantized copy of the model, total weight bytes)."""
    m = copy.deepcopy(model)
    total_bits = 0.0
    with torch.no_grad():
        for _, p in m.named_parameters():
            if p.dim() == 2:  # matrix: quantize
                w = p.detach().numpy().astype(np.float32)
                total_bits += w.size * bpw(scheme, w.shape)
                if scheme is not None:
                    p.copy_(torch.from_numpy(quant.fake_quant(w, scheme)))
            else:             # 1D RMSNorm weight: keep fp32 (llama.cpp does the same)
                total_bits += p.numel() * 32
    return m.eval(), total_bits / 8


@torch.no_grad()
def evaluate(model, ref_model, data, n_batches: int = 20, seq: int = 64) -> tuple[float, float]:
    """Validation loss, and the top-1 agreement with the reference model (on the same data)."""
    g = torch.Generator().manual_seed(1234)
    loss, agree, n = 0.0, 0, 0
    for _ in range(n_batches):
        x, y = data.batch("val", 32, seq, g)
        logits = model(x)
        loss += F.cross_entropy(logits.flatten(0, 1), y.flatten()).item()
        agree += (logits.argmax(-1) == ref_model(x).argmax(-1)).sum().item()
        n += y.numel()
    return loss / n_batches, agree / n


@torch.no_grad()
def greedy(model, data, prompt: str = "ROMEO:\n", n: int = 60) -> str:
    ids = data.encode(prompt)
    for _ in range(n):
        ids.append(int(model(torch.tensor([ids]))[0, -1].argmax()))
    return data.decode(ids[len(prompt):])


def run() -> list[dict]:
    data = tiny.CharData()
    base = tiny.load_or_train(4)
    rows = []
    for name, scheme in SCHEMES:
        m, nbytes = quantize_model(base, scheme)
        loss, agree = evaluate(m, base, data)
        rows.append(dict(name=name, scheme=scheme or "fp32", kib=nbytes / 1024, loss=loss,
                         agree=agree, text=greedy(m, data)))
    return rows


def main() -> None:
    rows = run()
    n_params = sum(p.numel() for p in tiny.load_or_train(4).parameters())
    print(f"Small model of Chapter 10: {n_params / 1e6:.2f}M parameters\n")
    ref = rows[0]["loss"]
    print(f"{'Method':<24}{'W KiB':>9}{'val loss':>10}{'Δ loss':>9}{'top-1 agr':>11}")
    for r in rows:
        print(f"{r['name']:<24}{r['kib']:>9.0f}{r['loss']:>10.4f}{r['loss'] - ref:>+9.4f}"
              f"{r['agree']:>11.1%}")
    print("\nGreedy generation (prompt 'ROMEO:\\n', first 60 characters):")
    for r in rows:
        if r["scheme"] in ("fp32", "int8-block32", "int4-block32", "int4-tensor", "int2-block32"):
            print(f"--- {r['name']}\n{r['text']}")


if __name__ == "__main__":
    main()
