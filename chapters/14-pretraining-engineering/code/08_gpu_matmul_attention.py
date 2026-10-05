"""第 14 章 · 极简代码 8（GPU 实测）：矩阵乘和注意力，在一张真 GPU 上有多快、占多少显存

正文第 3、4 节的结论是在 CPU 上讲出来的：
  - BF16 让 Tensor Core 算得更快，但只有 7 位尾数，精度约 2–3 位有效数字；
  - 朴素注意力要把 T×T 的 S 和 P 写进显存，FlashAttention 分块 + online softmax，从不写出它们。
这个脚本在一张 CUDA GPU 上把它们测出来：
  ① 方阵矩阵乘 C = A·B：FP32、TF32、BF16、FP16 各自的实测 TFLOPS（CUDA event 计时，取中位数），
     以及各自和 FP64 结果的相对误差；
  ② 02_precision.py ⑥ 的同一个 Linear、同一批输入，在 CPU 和 GPU 上各做一次 BF16 autocast；
  ③ 注意力前向 + 反向（1 条序列，16 个头 × head_dim 128，BF16，因果）：朴素写法 vs SDPA 的
     FlashAttention 后端，峰值显存和耗时随序列长度 T 怎么涨；
  ④ zero 的注意力（GQA：16 个查询头、8 个 KV 头，enable_gqa=True）在 GPU 上实际调用了哪个内核，
     BF16 autocast 和 FP32 各看一次。

运行：uv run python chapters/14-pretraining-engineering/code/08_gpu_matmul_attention.py   （需要 CUDA GPU，RTX 3090 上约 15 秒）
"""

import contextlib
import dataclasses
import math
import re
import statistics
import sys
import warnings
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.profiler import ProfilerActivity, profile

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))  # 让 `import zero` 在仓库任意位置运行都能找到
warnings.filterwarnings("ignore", message=".*no current CUDA context")    # 无害的提示，免得打乱输出
warnings.filterwarnings("ignore", message=".*Profiler clears events")

# 规格表的稠密峰值（TFLOPS）。RTX 3090：NVIDIA Ampere GA102 白皮书；Tensor Core 按 FP32 累加计，
# TF32 与 FP32 的规格峰值都是 35.6（GeForce 卡上 TF32 不比 FP32 快）。其他卡不查表，只报实测值。
SPEC = {"3090": {"FP32": 35.6, "TF32": 35.6, "BF16": 71.0, "FP16": 71.0}}
FORMATS = [("FP32", torch.float32, False), ("TF32", torch.float32, True),
           ("BF16", torch.bfloat16, False), ("FP16", torch.float16, False)]
H, D = 16, 128  # 主线模型：16 个查询头 × head_dim 128


def cuda_time(fn, warmup: int = 3, reps: int = 10) -> float:
    """fn() 一次的耗时（秒）：先预热，再用 CUDA event 计时 reps 次，取中位数。"""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(reps):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) / 1e3)
    return statistics.median(times)


def matmul_tflops(n: int, dtype: torch.dtype, tf32: bool) -> float:
    torch.backends.cuda.matmul.allow_tf32 = tf32
    a = torch.randn(n, n, device="cuda").to(dtype)
    b = torch.randn(n, n, device="cuda").to(dtype)
    inner = max(1, int(1e11 // (2 * n**3)))  # 每次计时至少约 1000 亿次运算，减小计时误差

    def run():
        for _ in range(inner):
            a @ b

    t = cuda_time(run)
    torch.backends.cuda.matmul.allow_tf32 = False
    return 2 * n**3 * inner / t / 1e12


def naive_attention(q, k, v, mask):
    s = q @ k.transpose(-2, -1) / math.sqrt(D)       # (1, H, T, T)：完整写进显存
    s = s.masked_fill(mask, float("-inf"))
    return torch.softmax(s, dim=-1) @ v               # P 也是 (1, H, T, T)，为反向保存


def flash_attention(q, k, v, mask):
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):     # 只允许 FlashAttention 后端，不满足条件就报错
        return F.scaled_dot_product_attention(q, k, v, is_causal=True)


def attention_cost(fn, T: int) -> tuple[float | None, int | None]:
    """前向 + 反向一次的耗时（秒）和峰值显存增量（字节，不含 q、k、v 本身）；显存不够返回 None。"""
    torch.cuda.empty_cache()
    q, k, v = (torch.randn(1, H, T, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
               for _ in range(3))
    grad_out = torch.randn_like(q)
    mask = torch.ones(T, T, dtype=torch.bool, device="cuda").triu(1)

    def step():
        fn(q, k, v, mask).backward(grad_out)
        q.grad = k.grad = v.grad = None

    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    try:
        t = cuda_time(step, warmup=2, reps=5)
    except torch.OutOfMemoryError:
        return None, None
    return t, torch.cuda.max_memory_allocated() - base


def attention_kernels(cfg, T: int, bf16: bool) -> tuple[list[str], bool]:
    """跑一层 zero 的 Transformer（前向 + 反向），返回 (调用到的融合注意力内核, 有没有单独的 softmax 内核)。"""
    from zero.model import Transformer

    model = Transformer(cfg).cuda()
    tokens = torch.randint(0, cfg.vocab_size, (1, T), device="cuda")

    def step():
        with torch.autocast("cuda", dtype=torch.bfloat16) if bf16 else contextlib.nullcontext():
            logits = model(tokens)
        logits.float().square().mean().backward()    # 故意不用交叉熵：它自带 softmax，会混淆判断

    step()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        step()
        torch.cuda.synchronize()
    keys = ("flash", "fmha", "efficient", "attention", "cudnn")
    names = {e.name for e in prof.events() if e.device_type.name == "CUDA"}
    fused = {re.sub(r"<.*", "", n).removeprefix("void ") for n in names if any(k in n.lower() for k in keys)}
    return sorted(fused), any("softmax" in n.lower() for n in names)   # 只留函数名，去掉模板参数


def main():
    if not torch.cuda.is_available():
        print("本脚本需要 CUDA GPU；没有 GPU 可以跳过，正文里贴了一次 RTX 3090 上的结果。")
        sys.exit(0)
    torch.manual_seed(0)
    name = torch.cuda.get_device_name()
    spec = next((v for k, v in SPEC.items() if k in name), None)
    print(f"GPU：{name}，{torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GiB；"
          f"PyTorch {torch.__version__}，CUDA {torch.version.cuda}")

    print("\n① 方阵矩阵乘 C = A·B 的实测吞吐（TFLOPS，CUDA event 计时，10 次取中位数）")
    sizes = (1024, 2048, 4096, 8192)
    print(f"  {'格式':<6}" + "".join(f"{f'n={n}':>10}" for n in sizes) + f"{'规格峰值':>10}{'最高/峰值':>10}")
    best = {}
    for fmt, dtype, tf32 in FORMATS:
        row = [matmul_tflops(n, dtype, tf32) for n in sizes]
        best[fmt] = max(row)
        peak = spec[fmt] if spec else None
        tail = f"{peak:>10.1f}{best[fmt] / peak:>10.0%}" if peak else f"{'—':>10}{'—':>10}"
        print(f"  {fmt:<6}" + "".join(f"{x:>10.1f}" for x in row) + tail)
    print(f"  BF16 比 FP32 快 {best['BF16'] / best['FP32']:.1f} 倍（各取最快的尺寸）")

    n = 4096
    a = torch.randn(n, n, device="cuda")
    b = torch.randn(n, n, device="cuda")
    ref = a.double() @ b.double()
    print(f"\n  同一对 {n}×{n} 随机矩阵，与 FP64 结果的相对误差 ‖C − C₆₄‖ / ‖C₆₄‖：")
    for fmt, dtype, tf32 in FORMATS:
        torch.backends.cuda.matmul.allow_tf32 = tf32
        c = (a.to(dtype) @ b.to(dtype)).double()
        torch.backends.cuda.matmul.allow_tf32 = False
        print(f"    {fmt:<5} {((c - ref).norm() / ref.norm()).item():.1e}")

    print("\n② 02_precision.py ⑥ 的同一个 Linear 与输入：BF16 autocast 在 CPU 和 GPU 上")
    torch.manual_seed(0)
    torch.randn(100_000)                         # 02_precision.py 在 ⑥ 之前只抽过这一次全局随机数
    lin = torch.nn.Linear(1024, 1024, bias=False)
    xin = torch.randn(64, 1024)
    ref_cpu = lin(xin)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out_cpu = lin(xin)
    lin_gpu, x_gpu = lin.cuda(), xin.cuda()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out_gpu = lin_gpu(x_gpu)
    def rel(out):
        return ((out.float().cpu() - ref_cpu).norm() / ref_cpu.norm()).item()

    same = (out_gpu.cpu() == out_cpu).float().mean().item()
    print(f"  与 FP32 结果的相对误差：CPU {rel(out_cpu):.2e}，GPU {rel(out_gpu):.2e}；"
          f"输出 dtype {out_gpu.dtype}；两边 BF16 输出逐元素相同的比例 {same:.1%}")
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False  # 默认 True
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out_gpu2 = lin_gpu(x_gpu)
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    print(f"  GPU 上禁止 cuBLAS 用 BF16 做中间规约（allow_bf16_reduced_precision_reduction=False）："
          f"相对误差 {rel(out_gpu2):.2e}")

    print(f"\n③ 注意力前向 + 反向（1 条序列，{H} 个头 × {D}，BF16，因果；5 次取中位数）")
    print(f"  {'T':>6} {'S+P 理论':>9} {'朴素 峰值显存':>13} {'Flash 峰值显存':>14} "
          f"{'朴素 耗时':>10} {'Flash 耗时':>10} {'Flash 快':>8}")
    for T in (1024, 2048, 4096, 8192, 16384):
        sp = 2 * H * T * T * 2                                   # S 和 P 各 H·T² 个 BF16
        tn, mn = attention_cost(naive_attention, T)
        tf, mf = attention_cost(flash_attention, T)
        naive_mem = f"{mn / 2**20:>12,.0f}M" if mn is not None else f"{'显存不够':>9}"
        naive_t = f"{tn * 1e3:>8.2f}ms" if tn is not None else f"{'—':>10}"
        speed = f"{tn / tf:>7.1f}×" if tn is not None else f"{'—':>8}"
        print(f"  {T:>6} {sp / 2**20:>8,.0f}M {naive_mem} {mf / 2**20:>13,.0f}M "
              f"{naive_t} {tf * 1e3:>8.2f}ms {speed}")
    print("  （峰值显存 = 一次前向 + 反向里比 q、k、v 多出来的最高点；M = MiB）")

    print("\n④ zero 的注意力在 GPU 上走哪个内核（主线形状取 1 层：16 个查询头、8 个 KV 头、enable_gqa=True）")
    from zero.config import load_model_config

    cfg = dataclasses.replace(load_model_config(ROOT / "configs/main/pretrain.toml"), n_layers=1)
    for label, bf16 in (("BF16 autocast", True), ("FP32", False)):
        fused, softmax = attention_kernels(cfg, 2048, bf16)
        print(f"  {label}：单独的 softmax 内核{'有' if softmax else '没有'}；融合注意力内核{'：' if fused else '没有'}")
        for kname in fused:
            print(f"    {kname}")
    q = torch.randn(1, H, 256, D, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(1, H // 2, 256, D, device="cuda", dtype=torch.bfloat16)
    for backend in (SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.CUDNN_ATTENTION):
        try:
            with warnings.catch_warnings(), sdpa_kernel(backend):
                warnings.simplefilter("ignore")        # 不支持时 PyTorch 会先打印一串原因
                F.scaled_dot_product_attention(q, kv, kv, is_causal=True, enable_gqa=True)
            status = "可以"
        except RuntimeError:
            status = "不支持（报错）"
        print(f"  只允许 {backend.name:<20} + enable_gqa=True：{status}")


if __name__ == "__main__":
    main()
