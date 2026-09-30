"""第 8 章 · GPU 脚本 6：手写注意力 vs SDPA 的三个后端——比速度，更比显存

02 里手写的 softmax(QKᵀ/√d) 会把 (B, H, T, T) 的分数矩阵整个放进显存：T 翻一倍，这块显存涨 4 倍。
PyTorch 的 F.scaled_dot_product_attention（SDPA）在 CUDA 上有三个后端，数学完全相同：
  - math：PyTorch 的参考实现，和手写版一样要存 T × T（BF16 输入还会先转成 float32 再算）；
  - efficient：memory-efficient attention（xFormers 的内核），分块计算，不存 T × T；
  - flash：FlashAttention-2，分块 + online softmax，也不存 T × T（原理见第 14 章）。

这个脚本做三件事：
  1. 对拍：同一组 q、k、v，四种写法的输出是否一致；
  2. 扫描序列长度 T = 512 … 32K，记录每种写法的耗时（CUDA event 计时，取中位数）和峰值显存；
  3. GQA（第 10 章）：K/V 头少于查询头时，enable_gqa=True 能走哪些后端。

形状取主线模型的注意力：batch 1、16 个查询头、head_dim 128、BF16、因果 mask。
需要 CUDA GPU（显存 ≥ 20 GB 才能跑完手写版的 16K），约半分钟。
运行：uv run python chapters/08-attention/code/06_gpu_sdpa_backends.py
"""

from __future__ import annotations

import importlib.util
import os
import statistics
import sys
import warnings
from pathlib import Path

# 让显存分配器按需扩展、少留碎片：这样"放不放得下"只取决于真正需要多少显存（要在 import torch 之前设）
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.nn.attention import SDPBackend, sdpa_kernel  # noqa: E402

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("attn02", HERE / "02_attention_from_scratch.py")
attn02 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(attn02)

B, H, D = 1, 16, 128  # 主线模型：16 个查询头，head_dim 128
H_KV = 8  # GQA：8 个 K/V 头
DTYPE = torch.bfloat16
LENGTHS = [512, 1024, 2048, 4096, 8192, 16384, 32768]
MiB = 2**20


def handwritten(q, k, v):
    """原样调用 02 的 attention；torch.device("cuda") 让它内部新建的因果 mask 也放在 GPU 上。"""
    with torch.device(q.device):
        return attn02.attention(q, k, v, causal=True)[0]


def sdpa(backend: SDPBackend):
    def fn(q, k, v):
        with sdpa_kernel(backend):
            return F.scaled_dot_product_attention(q, k, v, is_causal=True)

    return fn


METHODS = {
    "手写(02)": handwritten,
    "SDPA math": sdpa(SDPBackend.MATH),
    "SDPA efficient": sdpa(SDPBackend.EFFICIENT_ATTENTION),
    "SDPA flash": sdpa(SDPBackend.FLASH_ATTENTION),
}


def qkv(T: int, kv_heads: int = H, seed: int = 0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(B, H, T, D, device="cuda", dtype=DTYPE, generator=g)
    k = torch.randn(B, kv_heads, T, D, device="cuda", dtype=DTYPE, generator=g)
    v = torch.randn(B, kv_heads, T, D, device="cuda", dtype=DTYPE, generator=g)
    return q, k, v


@torch.no_grad()
def measure(fn, q, k, v, reps: int):
    """返回（耗时中位数 ms，峰值额外显存 MiB）；显存放不下时返回 None。

    峰值额外显存 = 调用过程中显存占用的最高点 − 调用前的占用（q、k、v 不算在内，输出算在内）。
    """
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    try:
        out = fn(q, k, v)  # 第一次：量峰值显存，顺便预热
        torch.cuda.synchronize()
        peak = (torch.cuda.max_memory_allocated() - base) / MiB
        del out
        fn(q, k, v)  # 再预热一次
        times = []
        for _ in range(reps):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            fn(q, k, v)
            end.record()
            end.synchronize()
            times.append(start.elapsed_time(end))
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache()
        return None
    return statistics.median(times), peak


def fmt_len(T: int) -> str:
    return f"{T // 1024}K" if T >= 1024 else str(T)


def main() -> None:
    if not torch.cuda.is_available():
        print("本脚本需要 CUDA GPU；没有 GPU 可以跳过，正文里贴了一次 RTX 3090 上的结果。")
        sys.exit(0)
    torch.manual_seed(0)
    free, total = (x / 2**30 for x in torch.cuda.mem_get_info())
    print(f"GPU：{torch.cuda.get_device_name(0)}（{total:.1f} GiB，空闲 {free:.1f} GiB）"
          f"  torch {torch.__version__}  CUDA {torch.version.cuda}")
    print(f"形状：B={B}，H={H} 个查询头，head_dim={D}，{str(DTYPE).replace('torch.', '')}，因果 mask\n")

    # ① 对拍：以 float32 的手写版为基准
    q, k, v = qkv(1024)
    ref = handwritten(q.float(), k.float(), v.float())
    print("1. 对拍（T=1024，与 float32 手写版的最大差；BF16 只有 8 位尾数，1e-2 量级是正常的）")
    with torch.no_grad():
        for name, fn in METHODS.items():
            print(f"   {name:<15} {float((fn(q, k, v).float() - ref).abs().max()):.1e}")

    # ② 扫描序列长度
    results = {}
    for T in LENGTHS:
        q, k, v = qkv(T)
        reps = 20 if T <= 4096 else 5
        results[T] = {name: measure(fn, q, k, v, reps) for name, fn in METHODS.items()}
        del q, k, v

    names = list(METHODS)
    print("\n2a. 耗时（毫秒，中位数；OOM = 显存不够）")
    print("   T     " + "".join(f"{n:>16}" for n in names))
    for T, row in results.items():
        cells = "".join(f"{'OOM' if r is None else f'{r[0]:.2f}':>16}" for r in row.values())
        print(f"   {fmt_len(T):<6}{cells}")

    flash = {T: row["SDPA flash"] for T, row in results.items() if row["SDPA flash"]}
    tflops = "，".join(f"{fmt_len(T)} {2 * B * H * T * T * D / r[0] / 1e9:.1f}" for T, r in flash.items())
    print(f"   flash 的有效算力（因果注意力约 2·H·T²·d 次浮点运算，TFLOPS）：{tflops}")

    print(f"\n2b. 峰值额外显存（MiB；{H} 个头的 T×T 分数矩阵按 BF16 算有多大，列在第二列）")
    print("   T     " + f"{'T×T 分数矩阵':>14}" + "".join(f"{n:>16}" for n in names))
    for T, row in results.items():
        scores = B * H * T * T * 2 / MiB
        cells = "".join(f"{'OOM' if r is None else f'{r[1]:.0f}':>16}" for r in row.values())
        print(f"   {fmt_len(T):<6}{scores:>14.0f}{cells}")

    # ③ GQA：16 个查询头共享 8 组 K/V
    T = 4096
    q, k, v = qkv(T, kv_heads=H_KV, seed=1)
    g = H // H_KV
    k_rep, v_rep = k.repeat_interleave(g, dim=1), v.repeat_interleave(g, dim=1)  # 第 10 章的手工复制
    with torch.no_grad(), sdpa_kernel(SDPBackend.MATH):
        ref = F.scaled_dot_product_attention(q, k_rep, v_rep, is_causal=True)
    print(f"\n3. GQA：q 有 {H} 个头，k、v 只有 {H_KV} 个头（T={T}），enable_gqa=True")
    for name, be in [("math", SDPBackend.MATH), ("efficient", SDPBackend.EFFICIENT_ATTENTION),
                     ("flash", SDPBackend.FLASH_ATTENTION)]:
        with warnings.catch_warnings(record=True) as caught:  # 后端被拒时 PyTorch 会用警告说明原因
            warnings.simplefilter("always")
            try:
                with torch.no_grad(), sdpa_kernel(be):
                    out = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
                diff = float((out.float() - ref.float()).abs().max())
                print(f"   {name:<10} 能跑    与'先复制 K/V 再算'的最大差 {diff:.1e}")
            except RuntimeError as e:
                why = [str(w.message).split(". ")[0] for w in caught if "num_heads" in str(w.message)]
                print(f"   {name:<10} 不能跑  {str(e).splitlines()[0]}")
                if why:
                    print(f"              原因：{why[0]}")
    if hasattr(torch, "_fused_sdp_choice"):  # 私有函数，只用来看"不指定后端时"PyTorch 会选谁
        for label, (kk, vv, gqa) in {"MHA": (k_rep, v_rep, False), "GQA": (k, v, True)}.items():
            choice = SDPBackend(torch._fused_sdp_choice(q, kk, vv, is_causal=True, enable_gqa=gqa))
            print(f"   不指定后端时，{label} 默认选：{choice.name}")

    def flash_gqa(q, k, v):
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            return F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)

    def flash_copy(q, k, v):
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            k2, v2 = k.repeat_interleave(g, dim=1), v.repeat_interleave(g, dim=1)
            return F.scaled_dot_product_attention(q, k2, v2, is_causal=True)

    del k_rep, v_rep, ref
    t_gqa, m_gqa = measure(flash_gqa, q, k, v, 20)
    t_copy, m_copy = measure(flash_copy, q, k, v, 20)
    print(f"   flash + enable_gqa：{t_gqa:.2f} ms，峰值额外显存 {m_gqa:.0f} MiB")
    print(f"   flash + 先复制 K/V：{t_copy:.2f} ms，峰值额外显存 {m_copy:.0f} MiB")


if __name__ == "__main__":
    main()
