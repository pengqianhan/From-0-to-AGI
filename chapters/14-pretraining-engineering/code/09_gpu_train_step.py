"""第 14 章 · 极简代码 9（GPU 实测）：主线模型在一张 GPU 上训练一步 —— 显存、激活检查点、MFU

正文第 2 节用 zero/tools/memory_calc.py 估算主线模型的显存，第 6 节定义了 MFU，
这两处的主线数字都是公式估算。这里把主线模型（689.5M，configs/main/pretrain.toml，
zero 的 Transformer + fused AdamW，和 zero 的训练循环一样）放到一张 GPU 上真的训练几步：
  ① 显存，micro batch 1：FP32、BF16 autocast、BF16 + 激活检查点三种设置，T = 1024 和 4096。
     量"前向结束时多出来的显存"（为反向保存的激活 + autocast 的 BF16 权重副本）和整步的峰值显存，
     和 memory_calc 的公式（单卡，没有 DDP 通信桶）对照；
  ② 速度：每步耗时（预热后 CUDA event 计时，取中位数）→ 吞吐 → MFU = 吞吐 × 每 token FLOPs / 规格峰值。
     激活检查点多算的那一遍前向不计入 MFU，另算 HFU；再按"因果掩码的上三角其实没算"换算一次；
  ③ T = 4096、打开激活检查点时，一张卡最多放得下几条序列：公式的预测 vs 实际。

输入是随机 token（loss 的数值没有意义，显存和耗时与真实数据相同）。

运行：uv run python chapters/14-pretraining-engineering/code/09_gpu_train_step.py   （需要约 24 GB 显存的 CUDA GPU，RTX 3090 上约 1 分钟）
"""

import contextlib
import gc
import statistics
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))  # 让 `import zero` 在仓库任意位置运行都能找到

from zero.config import load_model_config  # noqa: E402
from zero.model import Transformer, count_params, estimate_flops_per_token  # noqa: E402
from zero.tools import memory_calc as mc  # noqa: E402
from zero.tools.estimate_cost import peak_tflops_for_device_name  # noqa: E402

SPEC_BF16 = {"3090": 71.0}  # RTX 3090 规格表的 BF16 Tensor Core 稠密峰值（TFLOPS，FP32 累加）
GiB = mc.GiB
SETTINGS = {"FP32": (False, False), "BF16 autocast": (True, False), "BF16 + 检查点": (True, True)}


def main():
    if not torch.cuda.is_available():
        print("本脚本需要 CUDA GPU；没有 GPU 可以跳过，正文里贴了一次 RTX 3090 上的结果。")
        sys.exit(0)
    name = torch.cuda.get_device_name()
    total_gib = torch.cuda.get_device_properties(0).total_memory / GiB
    peak = next((v for k, v in SPEC_BF16.items() if k in name), None) or peak_tflops_for_device_name(name)
    print(f"GPU：{name}，{total_gib:.1f} GiB；PyTorch {torch.__version__}，CUDA {torch.version.cuda}")

    def pct(tflops):
        return f"{tflops / peak:.1%}" if peak else "—"

    cfg = load_model_config(ROOT / "configs/main/pretrain.toml")
    P = count_params(cfg)["total"]
    per_layer, _ = mc.matmul_params(cfg)          # 每层参与矩阵乘的参数
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

    step(*batch(1, 512), True, False)            # 第一步：让 AdamW 建好 m、v
    torch.cuda.synchronize()
    static = torch.cuda.memory_allocated()
    print(f"\n主线模型 {P / 1e6:.1f}M 参数。参数 + AdamW 的 m、v 常驻显存 {static / GiB:.2f} GiB"
          f"（按 12 字节/参数算是 {12 * P / GiB:.2f} GiB）；梯度只在反向到更新之间存在，另 4 字节/参数")

    def measure(mb, T, bf16, ckpt, reps=5):
        """返回 (前向结束时新增的显存, 整步峰值显存, 每步秒数)；显存不够返回 None。"""
        x, y = batch(mb, T)
        try:
            for _ in range(2):                   # 预热
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

    print("\n① 显存，micro batch 1：实测 vs memory_calc 的公式（单卡，G = GiB）")
    print(f"  {'T':>5} {'设置':<14} {'前向后新增':>9} {'公式':>7} {'整步峰值':>8} {'公式':>7}")
    for (T, label), r in results.items():
        bf16, ckpt = SETTINGS[label]
        est = mc.estimate_memory(cfg, 1, T, num_gpus=1, strategy="ddp",
                                 dtype="bf16" if bf16 else "fp32", checkpointing=ckpt)
        est_saved = est.activations + est.weight_copies
        if r is None:
            print(f"  {T:>5} {label:<14} {'显存不够':>8} {est_saved / GiB:>6.2f}G {'显存不够':>7} {est.total / GiB:>6.2f}G")
        else:
            print(f"  {T:>5} {label:<14} {r[0] / GiB:>8.2f}G {est_saved / GiB:>6.2f}G "
                  f"{r[1] / GiB:>7.2f}G {est.total / GiB:>6.2f}G")
    print("  （前向后新增 = 前向算完 loss 那一刻比开始时多出的显存；公式 = memory_calc 的激活 + BF16 权重副本。")
    print("   整步峰值 = 前向 + 反向 + 更新里 memory_allocated 的最高点；公式 = memory_calc 的合计。）")

    print(f"\n② 速度，micro batch 1（MFU 按 BF16 峰值 {peak} TFLOPS）")
    print(f"  {'T':>5} {'设置':<14} {'每步':>8} {'token/s':>8} {'MFU':>7} {'HFU':>7} {'因果减半后的利用率':>14}")
    for (T, label), r in results.items():
        if r is None:
            continue
        _, ckpt = SETTINGS[label]
        tps = T / r[2]
        fpt = estimate_flops_per_token(cfg, T)                  # 6N + 12·L·q_dim·T（zero / PaLM 口径）
        real = fpt - 6 * cfg.n_layers * cfg.q_dim * T          # 因果掩码的上三角不用算：注意力项减半
        # 激活检查点：反向前每层再做一遍前向（2·N_层 + 4·L·q_dim·T）
        hfu = pct(tps * (fpt + 2 * cfg.n_layers * per_layer + 4 * cfg.n_layers * cfg.q_dim * T) / 1e12) \
            if ckpt else ""
        print(f"  {T:>5} {label:<14} {r[2] * 1e3:>6.0f}ms {tps:>8,.0f} {pct(tps * fpt / 1e12):>7} "
              f"{hfu:>7} {pct(tps * real / 1e12):>14}")

    T = 4096
    pred = mc.max_micro_batch(cfg, T, total_gib, num_gpus=1, strategy="ddp", checkpointing=True)
    print(f"\n③ T = {T}、BF16 + 激活检查点：memory_calc 预测 {total_gib:.1f} GiB 的卡最多放 {pred} 条")
    print(f"  {'micro batch':>11} {'整步峰值':>8} {'公式':>7} {'每步':>8} {'token/s':>8} {'MFU':>7}")
    fpt = estimate_flops_per_token(cfg, T)
    for mb in range(1, pred + 3):
        est = mc.estimate_memory(cfg, mb, T, num_gpus=1, strategy="ddp", checkpointing=True)
        r = results.get((T, "BF16 + 检查点")) if mb == 1 else measure(mb, T, True, True, reps=3)
        if r is None:
            print(f"  {mb:>11} {'显存不够':>7} {est.total / GiB:>6.2f}G")
            break
        tps = mb * T / r[2]
        print(f"  {mb:>11} {r[1] / GiB:>7.2f}G {est.total / GiB:>6.2f}G {r[2] * 1e3:>6.0f}ms "
              f"{tps:>8,.0f} {pct(tps * fpt / 1e12):>7}")


if __name__ == "__main__":
    main()
