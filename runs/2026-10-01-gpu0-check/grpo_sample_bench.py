"""阶段 6 第 9 项的补充：主线 689.5M 模型在单张 RTX 3090 上做 GRPO 采样要多久（随机权重，只测速度）。

    CUDA_VISIBLE_DEVICES=0 UV_NO_SYNC=1 uv run python runs/2026-10-01-gpu0-check/grpo_sample_bench.py

`zero.post.grpo.sample_group` 对每个提示词调用一次 `zero.generate.generate`（batch = G，带 KV cache），
而且**不在 autocast 里**，所以 CUDA 上采样是 FP32。这里按 configs/main/grpo.toml 的 G=16、max_new_tokens=512，
提示词长度取 300（带工具说明的对话大约这么长），比较：
  - FP32（与 run_grpo 现在的写法相同）
  - BF16 autocast 包住 generate（KV cache 仍是 FP32）
  - 模型整体 .bfloat16()（KV cache 也是 BF16）
随机权重不会提前遇到 eos，每条都生成满 512 个 token，是"最长回复"的上界。
"""

from __future__ import annotations

import contextlib
import json
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from zero.config import load_model_config  # noqa: E402
from zero.generate import generate  # noqa: E402
from zero.model import Transformer  # noqa: E402


def bench(model, G: int, prompt_len: int, new: int, autocast: bool) -> dict:
    prompt = torch.randint(0, model.config.vocab_size, (G, prompt_len), device="cuda")
    ctx = torch.autocast("cuda", dtype=torch.bfloat16) if autocast else contextlib.nullcontext()
    with ctx:
        generate(model, prompt, 8, temperature=1.0, seed=0)  # 预热
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        out = generate(model, prompt, new, temperature=1.0, seed=0)
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    n = sum(len(o) for o in out)
    return {
        "seconds": round(dt, 2),
        "new_tokens": n,
        "tok_per_s": round(n / dt),
        "peak_mem_GiB": round(torch.cuda.max_memory_allocated() / 2**30, 2),
    }


def main() -> None:
    assert torch.cuda.device_count() == 1, "只允许看到 1 张卡（CUDA_VISIBLE_DEVICES=0）"
    cfg = load_model_config(REPO / "configs/main/grpo.toml")
    cfg.max_seq_len = 1024  # 只需要 300 + 512；RoPE 表小一点
    torch.manual_seed(0)
    G, P, NEW = 16, 300, 512
    res: dict = {"G": G, "prompt_len": P, "max_new_tokens": NEW}
    m = Transformer(cfg).cuda().eval()
    res["fp32"] = bench(m, G, P, NEW, autocast=False)
    print("fp32", res["fp32"], flush=True)
    res["bf16_autocast"] = bench(m, G, P, NEW, autocast=True)
    print("bf16 autocast", res["bf16_autocast"], flush=True)
    m = m.bfloat16()
    res["bf16_weights"] = bench(m, G, P, NEW, autocast=False)
    print("bf16 weights", res["bf16_weights"], flush=True)
    # 按 configs/main/grpo.toml：每步 64 个提示词 × G=16，逐个提示词采样
    for k in ("fp32", "bf16_autocast", "bf16_weights"):
        res[k]["per_grpo_step_sampling_s(64 prompts)"] = round(res[k]["seconds"] * 64, 1)
    out = REPO / "out/gpu0-check/grpo_sample_bench.json"
    out.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
