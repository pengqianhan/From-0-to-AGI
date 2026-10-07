"""Stage 6, item 1: can BF16 + SDPA use FlashAttention, and which backend does `enable_gqa=True` use?

The test uses one RTX 3090.

    CUDA_VISIBLE_DEVICES=0 UV_NO_SYNC=1 uv run python runs/2026-10-01-gpu0-check/sdpa_check.py

Three parts:
A. Attention only (main-line shape: 16 query heads, 8 K/V heads, head_dim 128, T=4096, BF16, causal):
   for each backend × {enable_gqa=True, repeat_interleave K/V first}: does it run, the forward+backward time,
   and the error against FP32 math. Then `torch.backends.cuda.can_use_*_attention(..., debug=True)` asks
   PyTorch for its own decision.
B. The original command in the RUNBOOK: the main-line 689.5M model `.cuda().bfloat16()`, forward pass
   at T=4096 under `sdpa_kernel(FLASH_ATTENTION)`.
C. Training step of the full model (FP32 master weights + BF16 autocast, forward+backward, micro batch 1 × 4096):
   default selection / forced Flash / forced efficient / repeat_interleave and then Flash.
   Compare throughput and MFU (3090 dense BF16 peak 71 TFLOPS).
"""

from __future__ import annotations

import contextlib
import json
import sys
import time
import warnings
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from zero.config import load_model_config  # noqa: E402
from zero.model import Transformer  # noqa: E402

PEAK_3090 = 71e12  # RTX 3090 dense BF16 Tensor Core peak (FP32 accumulation), from the GA102 white paper
BACKENDS = {
    "flash": SDPBackend.FLASH_ATTENTION,
    "efficient": SDPBackend.EFFICIENT_ATTENTION,
    "cudnn": SDPBackend.CUDNN_ATTENTION,
    "math": SDPBackend.MATH,
}


def _time(fn, iters: int = 5) -> float:
    fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def part_a(B: int = 1, H: int = 16, Hkv: int = 8, T: int = 4096, D: int = 128) -> dict:
    torch.manual_seed(0)
    dev = "cuda"
    q = torch.randn(B, H, T, D, device=dev, dtype=torch.bfloat16)
    k = torch.randn(B, Hkv, T, D, device=dev, dtype=torch.bfloat16)
    v = torch.randn(B, Hkv, T, D, device=dev, dtype=torch.bfloat16)
    rep = H // Hkv
    # Reference: FP32 math (K/V are copied explicitly first)
    with sdpa_kernel(SDPBackend.MATH):
        ref = F.scaled_dot_product_attention(
            q.float(),
            k.float().repeat_interleave(rep, 1),
            v.float().repeat_interleave(rep, 1),
            is_causal=True,
        )

    # PyTorch's own decision (with debug=True, PyTorch gives the reasons for "cannot use" as warnings)
    ask = {}
    for mode, (kk, vv, gqa) in {
        "enable_gqa": (k, v, True),
        "repeat_interleave": (k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1), False),
    }.items():
        params = torch.backends.cuda.SDPAParams(q, kk, vv, None, 0.0, True, gqa)
        for name, f in {
            "flash": torch.backends.cuda.can_use_flash_attention,
            "efficient": torch.backends.cuda.can_use_efficient_attention,
            "cudnn": torch.backends.cuda.can_use_cudnn_attention,
        }.items():
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                ok = bool(f(params, True))
            reasons = [str(x.message).strip().splitlines()[0][:200] for x in w]
            ask[f"{mode}/{name}"] = {"can_use": ok, "reasons": reasons}

    rows = []
    # Which kernel PyTorch selects with no restriction: the profiler shows the names of the CUDA kernels
    # that really start
    default_kernels = {}
    for mode, (kk, vv, gqa) in {
        "enable_gqa": (k, v, True),
        "repeat_interleave": (k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1), False),
    }.items():
        qq = q.detach().clone().requires_grad_(True)
        kk = kk.detach().clone().requires_grad_(True)
        vv = vv.detach().clone().requires_grad_(True)
        from torch.profiler import ProfilerActivity, profile

        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            o = F.scaled_dot_product_attention(qq, kk, vv, is_causal=True, enable_gqa=gqa)
            o.float().sum().backward()
            torch.cuda.synchronize()
        names = sorted(
            {
                e.key
                for e in prof.key_averages()
                if any(s in e.key.lower() for s in ("flash", "fmha", "attention", "cudnn", "sdpa"))
            }
        )
        default_kernels[mode] = [n[:120] for n in names]
    for mode in ("enable_gqa", "repeat_interleave"):
        for name, be in BACKENDS.items():
            qq = q.detach().clone().requires_grad_(True)
            kk = k.detach().clone().requires_grad_(True)
            vv = v.detach().clone().requires_grad_(True)

            def run(qq=qq, kk=kk, vv=vv, be=be, mode=mode):
                with sdpa_kernel(be):
                    if mode == "enable_gqa":
                        o = F.scaled_dot_product_attention(
                            qq, kk, vv, is_causal=True, enable_gqa=True
                        )
                    else:
                        o = F.scaled_dot_product_attention(
                            qq,
                            kk.repeat_interleave(rep, 1),
                            vv.repeat_interleave(rep, 1),
                            is_causal=True,
                        )
                o.float().sum().backward()
                return o

            row = {"mode": mode, "backend": name}
            try:
                torch.cuda.reset_peak_memory_stats()
                o = run()
                row["max_abs_err_vs_fp32_math"] = float((o.float() - ref).abs().max())
                row["fwd_bwd_ms"] = round(_time(run) * 1e3, 2)
                row["peak_mem_MiB"] = round(torch.cuda.max_memory_allocated() / 2**20)
                row["ok"] = True
            except RuntimeError as e:
                row["ok"] = False
                row["error"] = str(e).strip().splitlines()[0][:160]
            rows.append(row)
            torch.cuda.empty_cache()
    return {
        "shape": dict(B=B, H=H, Hkv=Hkv, T=T, D=D),
        "can_use": ask,
        "default_dispatch_kernels": default_kernels,
        "runs": rows,
    }


def part_b(cfg) -> dict:
    """The original command of item 1 in the RUNBOOK."""
    m = Transformer(cfg).cuda().bfloat16().eval()
    x = torch.randint(0, cfg.vocab_size, (1, 4096), device="cuda")
    out: dict = {}
    for name in ("flash", "efficient"):
        try:
            with torch.no_grad(), sdpa_kernel(BACKENDS[name]):
                y = m(x)
            out[name] = {"ok": True, "shape": list(y.shape)}
        except RuntimeError as e:
            out[name] = {"ok": False, "error": str(e).strip().splitlines()[0][:160]}
    del m
    torch.cuda.empty_cache()
    return out


def _gqa_repeat_wrapper(orig):
    """Change a call with enable_gqa=True to "repeat_interleave K/V first, then call".

    The replacement occurs only in this script. The code in zero does not change.
    """

    def sdpa(q, k, v, *args, enable_gqa=False, **kw):
        if enable_gqa and k.shape[1] != q.shape[1]:
            rep = q.shape[1] // k.shape[1]
            k = k.repeat_interleave(rep, 1)
            v = v.repeat_interleave(rep, 1)
        return orig(q, k, v, *args, **kw)

    return sdpa


def part_c(cfg, T: int = 4096, iters: int = 4) -> dict:
    """Forward+backward of the full model (FP32 master weights + BF16 autocast, as in Trainer), without the optimizer."""
    torch.manual_seed(0)
    m = Transformer(cfg).cuda().train()
    m.activation_checkpointing = False
    fpt = m.flops_per_token(T)
    x = torch.randint(0, cfg.vocab_size, (1, T + 1), device="cuda")
    inp, tgt = x[:, :-1], x[:, 1:]
    orig = F.scaled_dot_product_attention
    rows = []
    variants = [
        ("default(enable_gqa)", None, False),
        ("flash(enable_gqa)", "flash", False),
        ("efficient(enable_gqa)", "efficient", False),
        ("flash(repeat_interleave)", "flash", True),
        ("efficient(repeat_interleave)", "efficient", True),
    ]
    for name, be, repeat in variants:
        for ckpt in (False, True):
            m.activation_checkpointing = ckpt

            def step(be=be):
                ctx = sdpa_kernel(BACKENDS[be]) if be else contextlib.nullcontext()
                with ctx, torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = m.loss(inp, tgt)
                loss.backward()
                return loss

            row = {"variant": name, "activation_checkpointing": ckpt}
            try:
                if repeat:
                    F.scaled_dot_product_attention = _gqa_repeat_wrapper(orig)
                m.zero_grad(set_to_none=True)
                torch.cuda.reset_peak_memory_stats()
                dt = _time(step, iters)
                tok_s = T / dt
                row.update(
                    ok=True,
                    step_s=round(dt, 4),
                    tok_per_s=round(tok_s),
                    mfu_3090=round(fpt * tok_s / PEAK_3090, 4),
                    peak_mem_GiB=round(torch.cuda.max_memory_allocated() / 2**30, 2),
                )
            except (RuntimeError, torch.OutOfMemoryError) as e:
                row.update(ok=False, error=str(e).strip().splitlines()[0][:160])
            finally:
                F.scaled_dot_product_attention = orig
                m.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
    return {"T": T, "flops_per_token": fpt, "runs": rows}


def main() -> None:
    assert torch.cuda.device_count() == 1, "Only 1 GPU may be visible (CUDA_VISIBLE_DEVICES=0)"
    cfg = load_model_config(REPO / "configs/main/pretrain.toml")
    env = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(0),
        "capability": torch.cuda.get_device_capability(0),
    }
    print(json.dumps(env), flush=True)
    res = {"env": env}
    res["A_attention_only"] = part_a()
    print(json.dumps(res["A_attention_only"], ensure_ascii=False, indent=1), flush=True)
    res["B_runbook_command"] = part_b(cfg)
    print(json.dumps(res["B_runbook_command"], ensure_ascii=False), flush=True)
    res["C_model_train_step"] = part_c(cfg)
    out = REPO / "out/gpu0-check/sdpa_check.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
