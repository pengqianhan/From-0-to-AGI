"""Addition to Stage 6, item 9: time of GRPO sampling for the main-line 689.5M model on one RTX 3090.

The weights are random. The script measures only the speed.

    CUDA_VISIBLE_DEVICES=0 UV_NO_SYNC=1 uv run python runs/2026-10-01-gpu0-check/grpo_sample_bench.py

`zero.post.grpo.sample_group` calls `zero.generate.generate` one time for each prompt (batch = G,
with KV cache). The call is **not** inside autocast, so sampling on CUDA is FP32. This script uses
G=16 and max_new_tokens=512 from configs/main/grpo.toml, and a prompt length of 300
(a conversation with tool descriptions has about this length). It compares:
  - FP32 (the same as the current code of run_grpo)
  - BF16 autocast around generate (the KV cache is still FP32)
  - the full model in .bfloat16() (the KV cache is also BF16)
With random weights, generation does not stop early at eos. Each sequence gets all 512 tokens,
so the result is the upper bound for the "longest response".
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
        generate(model, prompt, 8, temperature=1.0, seed=0)  # warmup
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
    assert torch.cuda.device_count() == 1, "Only 1 GPU may be visible (CUDA_VISIBLE_DEVICES=0)"
    cfg = load_model_config(REPO / "configs/main/grpo.toml")
    cfg.max_seq_len = 1024  # 300 + 512 is sufficient; the RoPE table is then smaller
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
    # As in configs/main/grpo.toml: 64 prompts × G=16 per step, sampled one prompt at a time
    for k in ("fp32", "bf16_autocast", "bf16_weights"):
        res[k]["per_grpo_step_sampling_s(64 prompts)"] = round(res[k]["seconds"] * 64, 1)
    out = REPO / "out/gpu0-check/grpo_sample_bench.json"
    out.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
