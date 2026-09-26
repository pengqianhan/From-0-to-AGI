"""第 14 章 · 极简代码 1：一步预训练要花多少算力、多少时间 —— 以及 MFU 是怎么算的

第 12 章推过：训练一个 token 的浮点运算量（前向 + 反向）
    每 token FLOPs = 6·N_matmul + 12·L·q_dim·T
其中 N_matmul 是参与矩阵乘的参数（含 lm_head），后一项是注意力里 QKᵀ 和 AV 两个"没有参数"的矩阵乘。

这里做三件事：
  ① 用这个公式算主线模型（configs/main/pretrain.toml）一步、全程要多少 FLOPs；
  ② 换算成 8×H100 上的墙钟时间和费用（MFU 取 0.3 / 0.4 / 0.5）；
  ③ MFU（Model FLOPs Utilization）= 实际吞吐 × 每 token FLOPs / 硬件峰值：
     在本机单线程 CPU 上实测矩阵乘峰值，再读 tiny 预训练日志里的 tok/s，算出 CPU 上的 MFU。

运行：uv run python chapters/14-pretraining-engineering/code/01_step_cost.py   （约 10 秒）
（③ 需要先跑过 tiny 预训练：uv run python -m zero.train.pretrain --config configs/tiny/pretrain.toml
  --set train.out_dir=out/ch14/pretrain；没有日志时跳过。）
"""

import json
import statistics
import time
import tomllib
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
H100_BF16_DENSE = 989.5e12  # H100 SXM 稠密 BF16 峰值（zero/tools/estimate_cost.py 的表，待核实）
PRICE = 2.5  # 美元 / 卡时（GOAL.md 3.4 的假设）


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
    n_matmul = L * per_layer + d * m["vocab_size"]  # lm_head 即使和 embedding 共享，乘法照算
    n_total = L * (per_layer + 2 * d + 2 * hd) + d + d * m["vocab_size"]
    if not m.get("tie_embeddings", True):
        n_total += d * m["vocab_size"]
    return 6 * n_matmul + 12 * L * q_dim * T, n_matmul, n_total


def cpu_peak_flops(n: int = 1024, reps: int = 10, trials: int = 5) -> float:
    """单线程 fp32 矩阵乘能跑到的最快速度（取多次里最快的一次，减少被其他进程打扰的影响）。"""
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
    print("① 主线模型一步的算力（configs/main/pretrain.toml）")
    print(f"  总参数 N = {n_total / 1e6:.1f}M，参与矩阵乘 N_matmul = {n_matmul / 1e6:.1f}M，T = {T}")
    print(f"  每 token FLOPs = 6·N_matmul + 12·L·q_dim·T = {fpt:.4g}")
    gpus = 8
    tok_step = tr["micro_batch_size"] * tr["grad_accum_steps"] * gpus * T
    print(f"  每步 token = {tr['micro_batch_size']} × {tr['grad_accum_steps']} × {gpus} 卡 × {T} = {tok_step:,}")
    print(f"  每步 FLOPs = {fpt * tok_step:.4g}")

    print("\n② 500B token 在 8×H100 上要多久（峰值 989.5 TFLOPS/卡，待核实）")
    D = 500e9
    total = fpt * D
    print(f"  总 FLOPs = {total:.4g}")
    print(f"  {'MFU':>5} {'8 卡吞吐 tok/s':>14} {'每步秒数':>8} {'天数':>6} {'卡时':>8} {'费用':>8}")
    for mfu in (0.3, 0.4, 0.5):
        tps = gpus * H100_BF16_DENSE * mfu / fpt
        hours = D / tps / 3600
        print(f"  {mfu:>5} {tps:>14,.0f} {tok_step / tps:>8.2f} {hours / 24:>6.2f} "
              f"{hours * gpus:>8,.0f} ${hours * gpus * PRICE:>7,.0f}")

    print("\n③ MFU：实测吞吐 × 每 token FLOPs / 峰值（本机单线程 CPU）")
    peak = cpu_peak_flops()
    print(f"  单线程 fp32 矩阵乘峰值（1024×1024，取最快一次）：{peak / 1e9:.1f} GFLOPS")
    tiny = shape(ROOT / "configs/tiny/pretrain.toml")
    tfpt, _, tn = flops_per_token(tiny["model"], tiny["seq_len"])
    log = ROOT / "out/ch14/pretrain/log.jsonl"
    if not log.exists():
        print(f"  找不到 {log.relative_to(ROOT)}，先跑 tiny 预训练（见文件开头）")
        return
    recs = [json.loads(line) for line in log.read_text().splitlines()]
    tps = statistics.median(r["tok_per_s"] for r in recs if r["step"] > 1)
    mfu = tps * tfpt / peak
    print(f"  tiny 模型：{tn / 1e6:.2f}M 参数，每 token FLOPs = {tfpt:.4g}")
    print(f"  tiny 预训练日志的吞吐（中位数）：{tps:,.0f} tok/s")
    print(f"  → 有效算力 {tps * tfpt / 1e9:.2f} GFLOPS，MFU ≈ {mfu:.1%}")
    print("  （极小配置演示：分母是本机单线程 CPU 的实测峰值，和 GPU 上的 MFU 不可直接比较）")


if __name__ == "__main__":
    main()
