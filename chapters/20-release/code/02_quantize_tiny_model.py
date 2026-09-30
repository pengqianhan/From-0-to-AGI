"""第 20 章 · 极简代码 2：把量化套在一个真的小模型上，看 loss 变了多少

模型：第 10 章训练好的字符级小模型（Pre-Norm RMSNorm、RoPE、SwiGLU、共享 embedding，
dim=128、4 层，约 0.8M 参数；第一次运行会先训练约一两分钟，之后从缓存加载）。

做法和 llama.cpp 一样：所有二维权重矩阵（embedding、Q/K/V/O、FFN）量化；RMSNorm 的一维权重保持 fp32。
每种方案都"量化再反量化"（fake quant）后跑验证集，记录：

- 验证集 loss 和相对 fp32 的变化 Δ；
- top-1 一致率：在验证集的每个位置，量化模型和 fp32 模型"最可能的下一个字"是否相同；
- 权重大小（按每种方案的 bit/权重算）；
- 贪心生成的前 60 个字符，看一眼质量。

运行：uv run python chapters/20-release/code/02_quantize_tiny_model.py
"""

from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

torch.set_num_threads(1)  # 构建环境里多个任务共享 CPU；读者本机可以删掉这行

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

# 方案：名字 → 用 01 里的哪种量化（None = 不量化）
SCHEMES = [
    ("fp32", None),
    ("fp16", "fp16"),
    ("INT8 整张一个 scale", "int8-tensor"),
    ("INT8 分块 32（≈Q8_0）", "int8-block32"),
    ("INT4 整张一个 scale", "int4-tensor"),
    ("INT4 每行一个 scale", "int4-row"),
    ("INT4 分块 32（≈Q4_0）", "int4-block32"),
    ("INT4 两级 scale（≈Q4_K）", "int4-kquant"),
    ("INT3 分块 32", "int3-block32"),
    ("INT2 分块 32", "int2-block32"),
]


def bpw(scheme: str | None, shape: tuple[int, int]) -> float:
    if scheme is None:
        return 32.0
    return quant.bits_per_weight(scheme, shape)


def quantize_model(model: torch.nn.Module, scheme: str | None) -> tuple[torch.nn.Module, float]:
    """返回 (量化后的模型副本, 权重总字节数)。"""
    m = copy.deepcopy(model)
    total_bits = 0.0
    with torch.no_grad():
        for _, p in m.named_parameters():
            if p.dim() == 2:  # 矩阵：量化
                w = p.detach().numpy().astype(np.float32)
                total_bits += w.size * bpw(scheme, w.shape)
                if scheme is not None:
                    p.copy_(torch.from_numpy(quant.fake_quant(w, scheme)))
            else:             # RMSNorm 的一维权重：保持 fp32（llama.cpp 也这样）
                total_bits += p.numel() * 32
    return m.eval(), total_bits / 8


@torch.no_grad()
def evaluate(model, ref_model, data, n_batches: int = 20, seq: int = 64) -> tuple[float, float]:
    """验证集 loss，以及与参考模型的 top-1 一致率（同一批数据）。"""
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
    print(f"第 10 章的小模型：{n_params / 1e6:.2f}M 参数\n")
    ref = rows[0]["loss"]
    print(f"{'方案':<24}{'权重 KiB':>9}{'val loss':>10}{'Δ loss':>9}{'top-1 一致':>11}")
    for r in rows:
        print(f"{r['name']:<24}{r['kib']:>9.0f}{r['loss']:>10.4f}{r['loss'] - ref:>+9.4f}"
              f"{r['agree']:>11.1%}")
    print("\n贪心生成（提示词 'ROMEO:\\n'，前 60 个字符）：")
    for r in rows:
        if r["scheme"] in ("fp32", "int8-block32", "int4-block32", "int4-tensor", "int2-block32"):
            print(f"--- {r['name']}\n{r['text']}")


if __name__ == "__main__":
    main()
