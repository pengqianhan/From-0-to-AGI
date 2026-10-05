"""第 22 章 · GPU 实测：滑动窗口要"真的跳过"窗口外的块，才会变快

同一个滑动窗口注意力（mask = (j ≤ i) & (i − j < W)，01 的 sliding_mask），在 GPU 上三种算法：
  1. 稠密因果 SDPA（FlashAttention kernel）：窗口外照算不误，作为基准；
  2. 布尔掩码 SDPA（memory-efficient kernel + attn_mask）：窗口外的分数算出来再掩掉，
     还要先造一张 T × T 的布尔掩码；
  3. FlexAttention 的块稀疏滑动窗口：create_block_mask 先按 128 × 128 的块标出哪些块全在窗口外，
     kernel 直接跳过它们，只算带子经过的块。
先固定 T 改窗口 W，再固定 W 改 T，看耗时怎么变；最后确认三者算出来的是同一个东西。
FlexAttention 要 torch.compile 生成 Triton kernel，第一次调用要先编译（本机几秒，之后读缓存）。
没有 GPU 可以跳过；正文里贴了一次 RTX 3090 上的结果。
运行：uv run python chapters/22-local-sparse-attention/code/06_gpu_flex_window.py
"""

from __future__ import annotations

import sys
import time

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

B, H, D = 1, 16, 128  # 主线模型的查询头数和 head_dim
WINDOW = None  # 当前窗口（GPU 上的一个整数张量）；mask_mod 读它，换窗口不用重新编译


def sliding_mask_mod(b, h, q_idx, kv_idx):
    """FlexAttention 的 mask_mod：和 01 的 sliding_mask 同一个条件。"""
    return (kv_idx <= q_idx) & (q_idx - kv_idx < WINDOW)


def bool_mask(T: int, W: int) -> torch.Tensor:
    i = torch.arange(T, device="cuda")[:, None]
    j = torch.arange(T, device="cuda")[None, :]
    return (j <= i) & (i - j < W)


def cuda_ms(fn, reps: int = 10) -> float:
    for _ in range(2):  # 预热
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        torch.cuda.synchronize()
        times.append(a.elapsed_time(b))
    return sorted(times)[reps // 2]


def peak_extra_mib(fn) -> float:
    """fn 运行时在已有张量之外额外占了多少显存（MiB）。"""
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    return (torch.cuda.max_memory_allocated() - base) / 2**20


def main() -> None:
    global WINDOW
    if not torch.cuda.is_available():
        print("本脚本需要 CUDA GPU；没有 GPU 可以跳过，正文里贴了一次 RTX 3090 上的结果。")
        sys.exit(0)
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention

    torch.manual_seed(0)
    print(f"GPU：{torch.cuda.get_device_name(0)}，PyTorch {torch.__version__}，CUDA {torch.version.cuda}")
    print(f"形状：batch {B}，{H} 个头，head_dim {D}，BF16；FlexAttention 块大小 128 × 128")
    flex = torch.compile(flex_attention)
    # 造 block mask 也编译：按块生成，不在显存里铺开 T × T 的整张表（T = 65,536 时那要 32 GiB）
    make_block_mask = torch.compile(create_block_mask)
    WINDOW = torch.tensor(1, device="cuda")

    def run_all(T: int, W: int, with_mask: bool = True):
        q, k, v = (torch.randn(B, H, T, D, device="cuda", dtype=torch.bfloat16) for _ in range(3))
        WINDOW.fill_(W)
        bm = make_block_mask(sliding_mask_mod, None, None, T, T, device="cuda")
        n_blocks = (T // 128) * (T // 128 + 1) // 2  # 因果下三角里的块数
        frac = (1 - bm.sparsity() / 100) * (T // 128) ** 2 / n_blocks  # 实际要算的块 / 因果全部块

        def dense():
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                return F.scaled_dot_product_attention(q, k, v, is_causal=True)

        def flex_run():
            return flex(q, k, v, block_mask=bm)

        r = dict(frac=frac, dense=cuda_ms(dense), flex=cuda_ms(flex_run))
        if with_mask:
            m = bool_mask(T, W)  # 一张 T × T 的布尔表，一次前向里各层可以共用，不计入耗时
            r["mask_mib"] = m.nbytes / 2**20

            def masked():
                with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
                    return F.scaled_dot_product_attention(q, k, v, attn_mask=m)

            r["masked"] = cuda_ms(masked)
            r["masked_mem"] = peak_extra_mib(masked)
            ref = masked()
            r["diff_flex"] = (flex_run() - ref).abs().max().item()
            if W >= T:
                r["diff_dense"] = (dense() - ref).abs().max().item()
        r["flex_mem"] = peak_extra_mib(flex_run)
        r["dense_mem"] = peak_extra_mib(dense)
        return r

    t0 = time.time()
    run_all(1024, 256, with_mask=False)  # 触发一次编译
    print(f"（FlexAttention 首次编译用了 {time.time() - t0:.0f} s）")

    T = 16384
    print(f"\n1. 固定 T = {T:,}，改窗口 W（毫秒，10 次取中位数）")
    print("       W   flex 要算的块   稠密因果 SDPA   布尔掩码 SDPA   FlexAttention   flex 比稠密快")
    diffs = []
    for W in (128, 512, 1024, 4096, T):
        r = run_all(T, W)
        diffs.append((W, r))
        print(f"   {W:6,d}   {r['frac']:10.1%}   {r['dense']:10.2f}   {r['masked']:12.2f}"
              f"   {r['flex']:12.2f}   {r['dense'] / r['flex']:8.1f}×")
    r = diffs[2][1]
    print(f"   运行时在 q、k、v 之外额外占的显存（W = 1,024）：稠密 {r['dense_mem']:.0f} MiB；"
          f"布尔掩码 SDPA 先要一张 {r['mask_mib']:.0f} MiB 的掩码，运行时再占 {r['masked_mem']:.0f} MiB；"
          f"flex {r['flex_mem']:.0f} MiB")

    W = 1024
    print(f"\n2. 固定 W = {W:,}，改序列长 T（毫秒）")
    print("        T   稠密因果 SDPA   布尔掩码 SDPA   FlexAttention   flex 比稠密快")
    for T in (4096, 16384, 65536):
        r = run_all(T, W, with_mask=T <= 16384)
        masked = f"{r['masked']:12.2f}" if "masked" in r else f"{'（跳过）':>10}"
        print(f"   {T:6,d}   {r['dense']:10.2f}   {masked}   {r['flex']:12.2f}"
              f"   {r['dense'] / r['flex']:8.1f}×")
    print("   （T = 65,536 时布尔掩码本身就要 4 GiB，转成加性偏置再翻一倍，跳过）")

    print("\n3. 三者算的是同一个东西（与布尔掩码 SDPA 的最大绝对差，BF16）")
    for W, r in diffs:
        extra = f"；W ≥ T 时稠密因果 SDPA {r['diff_dense']:.1e}" if "diff_dense" in r else ""
        print(f"   T = 16,384、W = {W:>6,d}：FlexAttention {r['diff_flex']:.1e}{extra}")


if __name__ == "__main__":
    main()
