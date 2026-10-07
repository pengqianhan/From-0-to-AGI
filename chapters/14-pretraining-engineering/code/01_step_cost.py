"""Chapter 14 · Minimal code 1: the compute and the time of one pretraining step, and how to calculate MFU.

Chapter 12 derived the floating-point operations to train one token (forward pass + backward pass):
    FLOPs per token = 6·N_matmul + 12·L·q_dim·T
N_matmul is the number of parameters in matrix multiplications (lm_head included). The second term
is for the two matrix multiplications in attention, QKᵀ and AV. These two have no parameters.

The script does three things:
  ① It uses this formula to calculate the FLOPs of one step and of the full run (about 400B tokens)
     for the main-line model (configs/main/pretrain.toml).
  ② It converts the FLOPs to wall-clock time and cost on 8×H100 (MFU = 0.3 / 0.4 / 0.5).
  ③ MFU (Model FLOPs Utilization) = measured throughput × FLOPs per token / hardware peak.
     It measures the peak matmul speed of one CPU thread on this computer. Then it reads tok/s
     from the log of the tiny pretraining run and calculates the MFU on the CPU.

Run: uv run python chapters/14-pretraining-engineering/code/01_step_cost.py   (about 10 s)
(③ needs the tiny pretraining run first: uv run python -m zero.train.pretrain --config configs/tiny/pretrain.toml
  --set train.out_dir=out/ch14/pretrain. If there is no log, the script skips ③.)
"""

import json
import math
import statistics
import time
import tomllib
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
H100_BF16_DENSE = 989.5e12  # dense BF16 peak of the H100 SXM (table in zero/tools/estimate_cost.py, to be verified)
PRICE = 2.5  # USD per GPU-hour (assumption in GOAL.md 3.4)


def shape(cfg_path: Path) -> dict:
    with open(cfg_path, "rb") as f:
        c = tomllib.load(f)
    m = c["model"]
    m.setdefault("head_dim", m["dim"] // m["n_heads"])
    return {"model": m, "train": c["train"], "seq_len": c["data"]["seq_len"]}


def flops_per_token(m: dict, T: int) -> tuple[float, int, int]:
    d, L, hd = m["dim"], m["n_layers"], m["head_dim"]
    q_dim, kv_dim = m["n_heads"] * hd, m["n_kv_heads"] * hd
    per_layer = d * q_dim + 2 * d * kv_dim + q_dim * d + 3 * d * m["ffn_dim"]
    n_matmul = L * per_layer + d * m["vocab_size"]  # lm_head does its matmul, also when it shares weights with the embedding
    n_total = L * (per_layer + 2 * d + 2 * hd) + d + d * m["vocab_size"]
    if not m.get("tie_embeddings", True):
        n_total += d * m["vocab_size"]
    return 6 * n_matmul + 12 * L * q_dim * T, n_matmul, n_total


def cpu_peak_flops(n: int = 1024, reps: int = 10, trials: int = 5) -> float:
    """The fastest speed of an fp32 matmul on one thread.

    Take the fastest of several trials, so that other processes have less effect on the result.
    """
    a, b = torch.randn(n, n), torch.randn(n, n)
    for _ in range(3):
        a @ b
    best = 0.0
    for _ in range(trials):
        t = time.perf_counter()
        for _ in range(reps):
            a @ b
        best = max(best, 2 * n**3 * reps / (time.perf_counter() - t))
    return best


def main():
    torch.set_num_threads(1)
    main_cfg = shape(ROOT / "configs/main/pretrain.toml")
    m, tr, T = main_cfg["model"], main_cfg["train"], main_cfg["seq_len"]
    fpt, n_matmul, n_total = flops_per_token(m, T)
    print("① Compute of one step of the main-line model (configs/main/pretrain.toml)")
    print(f"  Total parameters N = {n_total / 1e6:.1f}M, parameters in matmuls N_matmul = {n_matmul / 1e6:.1f}M, T = {T}")
    print(f"  FLOPs per token = 6·N_matmul + 12·L·q_dim·T = {fpt:.4g}")
    gpus = 8
    tok_step = tr["micro_batch_size"] * tr["grad_accum_steps"] * gpus * T
    print(f"  Tokens per step = {tr['micro_batch_size']} × {tr['grad_accum_steps']} × {gpus} GPUs × {T} = {tok_step:,}")
    print(f"  FLOPs per step = {fpt * tok_step:.4g}")

    D = 400e9  # pretraining token budget from Chapter 12 (about 400B; gate 1 sets the final value)
    print(f"\n② How long {D / 1e9:.0f}B tokens take on 8×H100 (peak 989.5 TFLOPS per GPU, to be verified)")
    total = fpt * D
    print(f"  Total FLOPs = {total:.4g}; {D / 1e9:.0f}B / {tok_step:,} rounded up = {math.ceil(D / tok_step):,} steps"
          f" (max_steps in configs/main/pretrain.toml = {tr['max_steps']:,})")
    print(f"  {'MFU':>5} {'8-GPU tok/s':>14} {'s/step':>8} {'days':>6} {'GPU-h':>8} {'cost':>8}")
    for mfu in (0.3, 0.4, 0.5):
        tps = gpus * H100_BF16_DENSE * mfu / fpt
        hours = D / tps / 3600
        print(f"  {mfu:>5} {tps:>14,.0f} {tok_step / tps:>8.2f} {hours / 24:>6.2f} "
              f"{hours * gpus:>8,.0f} ${hours * gpus * PRICE:>7,.0f}")
    old = 500e9 / (gpus * H100_BF16_DENSE * 0.4 / fpt) / 3600
    print(f"  Reference: the original plan of 500B tokens at MFU 0.4 takes {old / 24:.2f} days and ${old * gpus * PRICE:,.0f}."
          f" This is more than the ~$5K for pretraining in GOAL.md 3.4")
    cost04 = D / (gpus * H100_BF16_DENSE * 0.4 / fpt) / 3600 * gpus * PRICE
    print(f"  The cost is inversely proportional to MFU: {D / 1e9:.0f}B costs more than $5,000 when MFU is below {0.4 * cost04 / 5000:.3f}")

    print("\n③ MFU: measured throughput × FLOPs per token / peak (one CPU thread on this computer)")
    peak = cpu_peak_flops()
    print(f"  Peak fp32 matmul speed on one thread (1024×1024, fastest run): {peak / 1e9:.1f} GFLOPS")
    tiny = shape(ROOT / "configs/tiny/pretrain.toml")
    tfpt, _, tn = flops_per_token(tiny["model"], tiny["seq_len"])
    log = ROOT / "out/ch14/pretrain/log.jsonl"
    if not log.exists():
        print(f"  Cannot find {log.relative_to(ROOT)}. Run the tiny pretraining first (see the top of this file)")
        return
    recs = [json.loads(line) for line in log.read_text().splitlines()]
    tps = statistics.median(r["tok_per_s"] for r in recs if r["step"] > 1)
    mfu = tps * tfpt / peak
    print(f"  Tiny model: {tn / 1e6:.2f}M parameters, FLOPs per token = {tfpt:.4g}")
    print(f"  Throughput in the tiny pretraining log (median): {tps:,.0f} tok/s")
    print(f"  → effective compute {tps * tfpt / 1e9:.2f} GFLOPS, MFU ≈ {mfu:.1%}")
    print("  (Tiny-configuration demo: the denominator is the measured peak of one CPU thread on this computer."
          " Do not compare it directly with MFU on a GPU.)")


if __name__ == "__main__":
    main()
