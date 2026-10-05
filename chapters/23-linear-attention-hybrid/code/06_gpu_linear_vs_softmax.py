"""第 23 章 · GPU 实测：softmax 注意力 vs 线性注意力的分块形式、递推形式

把本章 03、04 里的函数原封不动搬到 GPU 上（在 `with torch.device("cuda")` 里调用，
函数内部新建的张量也就落在 GPU 上），和 PyTorch 自带的 FlashAttention 比一比：
  1. 数值：Gated DeltaNet 的分块形式 == 递推形式（GPU，FP32）；
  2. 整段处理（训练 / prefill）：softmax 注意力用 SDPA 的 FlashAttention kernel（BF16）；
     线性注意力用 04 的 linear_chunked、Gated DeltaNet 用 03 的 gated_delta_chunked
     （纯 PyTorch、FP32、块长 64，Python 逐块循环）；逐 token 的递推形式只测短序列。比耗时和峰值显存；
  3. decode 一步：softmax 注意力要读全部 T 个 K、V；Gated DeltaNet 先用分块形式把 T 个 token
     压成状态，再递推一步——状态是多大、这一步多久。
形状取 Qwen3.5-0.8B 的 Gated DeltaNet 层：batch 1，16 个头，d_k = d_v = 128；softmax 注意力也用 16 × 128。
没有 GPU 可以跳过；正文里贴了一次 RTX 3090 上的结果。
运行：uv run python chapters/23-linear-attention-hybrid/code/06_gpu_linear_vs_softmax.py
"""

from __future__ import annotations

import importlib.util
import sys
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

HERE = Path(__file__).resolve().parent
B, H, D = 1, 16, 128
C = 64  # 分块形式的块长


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def inputs(T: int, dtype=torch.float32):
    """q、k 做 L2 归一化（本章第 3 节起的写法）；g 是对数衰减，beta 是写入强度。"""
    q = F.normalize(torch.randn(B, H, T, D), dim=-1).to(dtype)
    k = F.normalize(torch.randn(B, H, T, D), dim=-1).to(dtype)
    v = torch.randn(B, H, T, D).to(dtype)
    g, beta = -torch.rand(B, H, T) * 0.05, torch.rand(B, H, T)
    return q, k, v, g, beta


def cuda_ms(fn, reps: int) -> float:
    fn()  # 预热
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


def peak_mib(fn) -> float:
    """fn 运行时在输入之外额外占的峰值显存（MiB）。"""
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    return (torch.cuda.max_memory_allocated() - base) / 2**20


def softmax_prefill(q, k, v):
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        return F.scaled_dot_product_attention(q, k, v, is_causal=True)


def main() -> None:
    if not torch.cuda.is_available():
        print("本脚本需要 CUDA GPU；没有 GPU 可以跳过，正文里贴了一次 RTX 3090 上的结果。")
        sys.exit(0)
    lm = _load("hybrid_lm", HERE / "04_hybrid_lm.py")  # 04 里又加载了 03（lm.delta）
    delta = lm.delta
    torch.manual_seed(0)
    print(f"GPU：{torch.cuda.get_device_name(0)}，PyTorch {torch.__version__}，CUDA {torch.version.cuda}")
    print(f"形状：batch {B}，{H} 个头，d_k = d_v = {D}；线性注意力 FP32、块长 {C}；softmax 注意力 BF16")

    with torch.device("cuda"):
        print("\n1. Gated DeltaNet：分块形式 == 递推形式（GPU，FP32，T = 1,024）")
        q, k, v, g, beta = inputs(1024)
        o1, s1 = delta.gated_delta_recurrent(q, k, v, g, beta)
        o2, s2 = delta.gated_delta_chunked(q, k, v, g, beta, C=C)
        print(f"   max|输出差| = {(o1 - o2).abs().max():.1e}   max|最终状态差| = {(s1 - s2).abs().max():.1e}")

        print("\n2. 整段处理 T 个 token（毫秒，取中位数；括号里是输入之外额外占的峰值显存）")
        print("        T   softmax（FlashAttention）   线性注意力·分块   Gated DeltaNet·分块"
              "   Gated DeltaNet·递推")
        for T in (1024, 4096, 16384, 65536):
            q, k, v, g, beta = inputs(T)
            fns = [partial(softmax_prefill, q.bfloat16(), k.bfloat16(), v.bfloat16()),
                   partial(lm.linear_chunked, q, k, v, C=C),
                   partial(delta.gated_delta_chunked, q, k, v, g, beta, C=C)]
            reps = 5 if T >= 16384 else 10
            cells = [f"{cuda_ms(fn, reps):9.1f}（{peak_mib(fn):5.0f} MiB）" for fn in fns]
            if T <= 4096:  # 逐 token 循环：T 步，每步几个小 kernel；更长的序列太慢，不测
                rec = partial(delta.gated_delta_recurrent, q, k, v, g, beta)
                cells.append(f"{cuda_ms(rec, 3):9.1f}")
            else:
                cells.append(f"{'—':>9}")
            print(f"   {T:6,d}   " + "   ".join(cells))

        print("\n3. decode 一步：已有 T 个 token 的上下文，再来 1 个（毫秒，50 次取中位数）")
        print("          T   softmax 的 KV cache   softmax 一步   Gated DeltaNet 的状态   GDN 一步")
        for T in (1024, 16384, 65536, 262144):
            kv_k = torch.randn(B, H, T, D, dtype=torch.bfloat16)  # softmax：T 个位置的 K、V
            kv_v = torch.randn(B, H, T, D, dtype=torch.bfloat16)
            q1 = torch.randn(B, H, 1, D, dtype=torch.bfloat16)
            soft_step = partial(F.scaled_dot_product_attention, q1, kv_k, kv_v)
            _, S = delta.gated_delta_chunked(*inputs(T), C=C)  # prefill：T 个 token 压成状态 S
            gdn_step = partial(delta.gated_delta_recurrent, *inputs(1), S=S)  # 从 S 出发再走一步
            kv_mib = (kv_k.nbytes + kv_v.nbytes) / 2**20
            print(f"   {T:8,d}   {kv_mib:12,.0f} MiB   {cuda_ms(soft_step, 50):9.3f}"
                  f"   {tuple(S.shape)} {S.nbytes / 2**20:.0f} MiB   {cuda_ms(gdn_step, 50):7.3f}")


if __name__ == "__main__":
    main()
