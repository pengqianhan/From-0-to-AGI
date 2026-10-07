"""Chapter 14 · Minimal code 9 (GPU measurement): one training step of the main-line model on one GPU. Memory, activation checkpointing, and MFU.

Section 2 of the chapter estimates the memory of the main-line model with zero/tools/memory_calc.py,
and Section 6 defines MFU. In both places, the main-line numbers are estimates from formulas.
Here, the script puts the main-line model (689.5M, configs/main/pretrain.toml, the zero Transformer +
fused AdamW, the same as in the zero training loop) on one GPU and really trains some steps:
  ① Memory, micro batch 1: three settings, FP32, BF16 autocast, and BF16 + activation checkpointing
     ("ckpt"), at T = 1024 and 4096. It measures "the memory added at the end of the forward pass"
     (activations saved for the backward pass + the BF16 weight copies of autocast) and the peak memory
     of the full step. It compares them with the formulas of memory_calc (one GPU, no DDP communication buckets);
  ② Speed: time per step (CUDA event timing after warmup, median) → throughput →
     MFU = throughput × FLOPs per token / spec peak.
     MFU does not count the extra forward pass of activation checkpointing; HFU counts it.
     The last column calculates the utilization again with the attention term halved
     ("causal-halved"), because the causal mask skips the upper triangle;
  ③ At T = 4096 with activation checkpointing: how many sequences fit on one GPU, formula vs measurement.

The inputs are random tokens. (The loss values have no meaning. The memory and the time are the same as with real data.)

Run: uv run python chapters/14-pretraining-engineering/code/09_gpu_train_step.py   (needs a CUDA GPU with about 24 GB of memory; about 1 min on an RTX 3090)
"""

import contextlib
import gc
import statistics
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))  # so that `import zero` works from any folder of the repository

from zero.config import load_model_config  # noqa: E402
from zero.model import Transformer, count_params, estimate_flops_per_token  # noqa: E402
from zero.tools import memory_calc as mc  # noqa: E402
from zero.tools.estimate_cost import peak_tflops_for_device_name  # noqa: E402

SPEC_BF16 = {"3090": 71.0}  # dense BF16 Tensor Core peak of the RTX 3090 from the spec sheet (TFLOPS, FP32 accumulation)
GiB = mc.GiB
SETTINGS = {"FP32": (False, False), "BF16 autocast": (True, False), "BF16 + ckpt": (True, True)}


def main():
    if not torch.cuda.is_available():
        print("This script needs a CUDA GPU. Without a GPU, skip it: the chapter shows one result from an RTX 3090.")
        sys.exit(0)
    name = torch.cuda.get_device_name()
    total_gib = torch.cuda.get_device_properties(0).total_memory / GiB
    peak = next((v for k, v in SPEC_BF16.items() if k in name), None) or peak_tflops_for_device_name(name)
    print(f"GPU: {name}, {total_gib:.1f} GiB; PyTorch {torch.__version__}, CUDA {torch.version.cuda}")

    def pct(tflops):
        return f"{tflops / peak:.1%}" if peak else "—"

    cfg = load_model_config(ROOT / "configs/main/pretrain.toml")
    P = count_params(cfg)["total"]
    per_layer, _ = mc.matmul_params(cfg)          # parameters in matmuls in each layer
    torch.manual_seed(0)
    with torch.device("cuda"):
        model = Transformer(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1, fused=True)
    gen = torch.Generator(device="cuda").manual_seed(0)

    def batch(mb, T):
        return (torch.randint(0, cfg.vocab_size, (mb, T), device="cuda", generator=gen),
                torch.randint(0, cfg.vocab_size, (mb, T), device="cuda", generator=gen))

    def step(x, y, bf16, ckpt):
        model.activation_checkpointing = ckpt
        with torch.autocast("cuda", dtype=torch.bfloat16) if bf16 else contextlib.nullcontext():
            loss = model.loss(x, y)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)

    step(*batch(1, 512), True, False)            # first step: AdamW creates m and v
    torch.cuda.synchronize()
    static = torch.cuda.memory_allocated()
    print(f"\nMain-line model, {P / 1e6:.1f}M parameters. Parameters + AdamW m and v stay in memory: {static / GiB:.2f} GiB"
          f" ({12 * P / GiB:.2f} GiB at 12 bytes/parameter). Gradients exist only from the backward pass to the update: 4 more bytes/parameter")

    def measure(mb, T, bf16, ckpt, reps=5):
        """Return (memory added at the end of the forward pass, peak memory of the full step, seconds per step).

        Return None if there is not sufficient memory.
        """
        x, y = batch(mb, T)
        try:
            for _ in range(2):                   # warmup
                step(x, y, bf16, ckpt)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            base = torch.cuda.memory_allocated()
            model.activation_checkpointing = ckpt
            with torch.autocast("cuda", dtype=torch.bfloat16) if bf16 else contextlib.nullcontext():
                loss = model.loss(x, y)
            torch.cuda.synchronize()
            saved = torch.cuda.memory_allocated() - base
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            del loss
            torch.cuda.synchronize()
            peak_mem = torch.cuda.max_memory_allocated()
            times = []
            for _ in range(reps):
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                step(x, y, bf16, ckpt)
                end.record()
                torch.cuda.synchronize()
                times.append(start.elapsed_time(end) / 1e3)
            return saved, peak_mem, statistics.median(times)
        except torch.OutOfMemoryError:
            return None
        finally:
            opt.zero_grad(set_to_none=True)
            gc.collect()
            torch.cuda.empty_cache()

    results = {}
    for T in (1024, 4096):
        for label, (bf16, ckpt) in SETTINGS.items():
            results[T, label] = measure(1, T, bf16, ckpt)

    print("\n① Memory, micro batch 1: measured vs the formulas of memory_calc (one GPU, G = GiB)")
    print(f"  {'T':>5} {'setting':<14} {'fwd added':>9} {'formula':>7} {'peak':>8} {'formula':>7}")
    for (T, label), r in results.items():
        bf16, ckpt = SETTINGS[label]
        est = mc.estimate_memory(cfg, 1, T, num_gpus=1, strategy="ddp",
                                 dtype="bf16" if bf16 else "fp32", checkpointing=ckpt)
        est_saved = est.activations + est.weight_copies
        if r is None:
            print(f"  {T:>5} {label:<14} {'OOM':>8} {est_saved / GiB:>6.2f}G {'OOM':>7} {est.total / GiB:>6.2f}G")
        else:
            print(f"  {T:>5} {label:<14} {r[0] / GiB:>8.2f}G {est_saved / GiB:>6.2f}G "
                  f"{r[1] / GiB:>7.2f}G {est.total / GiB:>6.2f}G")
    print("  (fwd added = the extra memory at the moment when the forward pass has calculated the loss; formula = activations + BF16 weight copies of memory_calc.")
    print("   peak = the highest memory_allocated in forward + backward + update; formula = total of memory_calc. OOM = out of memory.)")

    print(f"\n② Speed, micro batch 1 (MFU from the BF16 peak of {peak} TFLOPS)")
    print(f"  {'T':>5} {'setting':<14} {'per step':>8} {'token/s':>8} {'MFU':>7} {'HFU':>7} {'causal-halved':>14}")
    for (T, label), r in results.items():
        if r is None:
            continue
        _, ckpt = SETTINGS[label]
        tps = T / r[2]
        fpt = estimate_flops_per_token(cfg, T)                  # 6N + 12·L·q_dim·T (the convention of zero / PaLM)
        real = fpt - 6 * cfg.n_layers * cfg.q_dim * T          # the causal mask skips the upper triangle: halve the attention term
        # activation checkpointing: before the backward pass, each layer does its forward pass again (2·N_layer + 4·L·q_dim·T)
        hfu = pct(tps * (fpt + 2 * cfg.n_layers * per_layer + 4 * cfg.n_layers * cfg.q_dim * T) / 1e12) \
            if ckpt else ""
        print(f"  {T:>5} {label:<14} {r[2] * 1e3:>6.0f}ms {tps:>8,.0f} {pct(tps * fpt / 1e12):>7} "
              f"{hfu:>7} {pct(tps * real / 1e12):>14}")

    T = 4096
    pred = mc.max_micro_batch(cfg, T, total_gib, num_gpus=1, strategy="ddp", checkpointing=True)
    print(f"\n③ T = {T}, BF16 + activation checkpointing: memory_calc predicts that a {total_gib:.1f} GiB GPU holds at most {pred} sequences")
    print(f"  {'micro batch':>11} {'peak':>8} {'formula':>7} {'per step':>8} {'token/s':>8} {'MFU':>7}")
    fpt = estimate_flops_per_token(cfg, T)
    for mb in range(1, pred + 3):
        est = mc.estimate_memory(cfg, mb, T, num_gpus=1, strategy="ddp", checkpointing=True)
        r = results.get((T, "BF16 + ckpt")) if mb == 1 else measure(mb, T, True, True, reps=3)
        if r is None:
            print(f"  {mb:>11} {'OOM':>7} {est.total / GiB:>6.2f}G")
            break
        tps = mb * T / r[2]
        print(f"  {mb:>11} {r[1] / GiB:>7.2f}G {est.total / GiB:>6.2f}G {r[2] * 1e3:>6.0f}ms "
              f"{tps:>8,.0f} {pct(tps * fpt / 1e12):>7}")


if __name__ == "__main__":
    main()
