"""第 24 章 · GPU 实测：专家翻倍、每个 token 的算力不变——以及为什么 GPU 上不能按专家写 Python 循环

正文的代码都在 CPU 上跑。这个脚本把 02_moe_layer.py 的 MoE 层原样搬到一张 GPU 上（BF16，需要 CUDA）：
一次 8192 个 token（像训练或 prefill 时的一个批次），d = 1024，每个专家宽 2048，每个 token 选 2 个——
激活宽度 2 × 2048 = 4096，和一个宽 4096 的稠密 SwiGLU 同算力。专家数 E 从 8 加到 128：总参数翻 16 倍，
每个 token 的 FLOPs 不变。比三种写法的耗时：
  - 稠密 SwiGLU（宽 4096，同激活算力的基准）；
  - 02 的 MoE.forward：Python 里按专家循环，每个专家一次 nonzero（要等 GPU 算完才知道有哪些 token）
    + 三个小矩阵乘 + index_add_；
  - 排序分段 + F.grouped_mm：token 按专家排好序，每个矩阵一次 grouped GEMM 调用算完所有专家，
    整个前向不需要等 GPU（生产级 zero/arch/moe.py 的 MoEFFN 也是"按专家堆叠权重 + 排序分段"的思路）。

路由器、专家都是随机初始化（只测时间；每个 E 都先把两种 MoE 写法的输出对拍一遍，确认算的是同一件事）。
运行：uv run python chapters/24-mixture-of-experts/code/05_gpu_moe.py（约 15 秒）
"""

from __future__ import annotations

import importlib.util
import statistics
import sys
import time
import warnings
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
DEV = "cuda"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


moe_mod = _load("moe_ch24", HERE / "02_moe_layer.py")


def build(cls, *args, **kw):
    """直接在 GPU 上建模块（省掉 CPU 上初始化几亿参数的时间），再转 BF16。"""
    with torch.device(DEV):
        m = cls(*args, **kw)
    return m.to(torch.bfloat16).eval()


def stack_experts(m) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """把 E 个专家的权重按专家堆成三维张量：(E, d, h)、(E, d, h)、(E, h, d)。生产级实现直接这样存。"""
    wg = torch.stack([e.w_gate.weight.t() for e in m.experts]).contiguous()
    wu = torch.stack([e.w_up.weight.t() for e in m.experts]).contiguous()
    wd = torch.stack([e.w_down.weight.t() for e in m.experts]).contiguous()
    return wg, wu, wd


def grouped_moe(m, x: torch.Tensor, w) -> torch.Tensor:
    """与 02 的 MoE.forward 同样的路由和门控；专家计算换成排序分段 + grouped GEMM，全程不用等 GPU。"""
    wg, wu, wd = w
    s = m.router(x).sigmoid() if m.score == "sigmoid" else m.router(x).softmax(-1)
    idx = (s + m.bias).topk(m.K, dim=-1).indices  # (T, K)
    g = s.gather(-1, idx)
    g = g / g.sum(-1, keepdim=True)
    flat = idx.flatten()  # (T·K,) 每一次"分配"去哪个专家
    order = flat.argsort(stable=True)  # 按专家排序：同一个专家的 token 连成一段
    offs = torch.bincount(flat, minlength=m.E).cumsum(0).to(torch.int32)  # 每一段的结尾
    tok = order // m.K  # 排序后的第 i 行来自哪个 token
    xs = x[tok]  # dispatch：(T·K, d)
    h = F.silu(F.grouped_mm(xs, wg, offs=offs)) * F.grouped_mm(xs, wu, offs=offs)
    y = F.grouped_mm(h, wd, offs=offs)  # 每一段乘自己那个专家的权重
    out = torch.zeros_like(x)
    out.index_add_(0, tok, y * g.flatten()[order, None])  # combine：按门控权重加回原位置
    return out


@torch.no_grad()
def gpu_ms(fn, reps: int = 30, warmup: int = 5) -> float:
    """CUDA event 计时，预热后取中位数（毫秒）。"""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(reps):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return statistics.median(times)


def launch_us(n: int = 2000) -> float:
    """这台机器上，Python 里下发一个最小的 GPU 算子要多少微秒（一个元素的加法，GPU 本身几乎不花时间）。"""
    x = torch.zeros(1, device=DEV)
    for _ in range(100):
        x.add_(1)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        x.add_(1)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1e6


@torch.no_grad()
def kernels_per_grouped_mm(E: int = 128, d: int = 1024, h: int = 2048, T: int = 1024) -> dict:
    """用 profiler 看一次 F.grouped_mm（E 个专家）在这张卡上实际发了哪些 GPU kernel、各几次。"""
    from torch.profiler import ProfilerActivity, profile

    xs = torch.randn(T, d, device=DEV, dtype=torch.bfloat16)
    wg = torch.randn(E, d, h, device=DEV, dtype=torch.bfloat16)
    offs = (torch.arange(1, E + 1, device=DEV) * (T // E)).to(torch.int32)
    F.grouped_mm(xs, wg, offs=offs)
    torch.cuda.synchronize()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            F.grouped_mm(xs, wg, offs=offs)
            torch.cuda.synchronize()
    counts: dict = {}
    for ev in prof.events():
        if ev.device_type == torch.autograd.DeviceType.CUDA:
            counts[ev.name] = counts.get(ev.name, 0) + 1
    return counts


def n_params(module) -> int:
    return sum(p.numel() for p in module.parameters())


@torch.no_grad()
def main(d: int = 1024, h: int = 2048, K: int = 2, T: int = 8192) -> None:
    print(
        f"一次 {T} 个 token，d = {d}，每个专家宽 {h}，每个 token 选 {K} 个；"
        "BF16，CUDA event 计时，30 次中位数"
    )
    x = torch.randn(T, d, device=DEV, dtype=torch.bfloat16)
    dense = build(moe_mod.Expert, d, K * h)
    flops_tok = 2 * 3 * d * K * h  # 每个 token：激活的专家里三个矩阵，每个参数一次乘加
    t_dense = gpu_ms(lambda dense=dense: dense(x))
    print(
        f"{'':13}{'总参数':>8}{'每 token GFLOP':>15}{'每专家 token 数':>14}{'逐专家循环 ms':>14}"
        f"{'grouped ms':>12}{'循环/grouped':>12}{'grouped TFLOPS':>16}"
    )
    print(
        f"{'稠密 宽 ' + str(K * h):13}{n_params(dense) / 1e6:7.0f}M{flops_tok / 1e9:15.4f}"
        f"{'':>18}{'':>14}{t_dense:12.2f}{'':>13}{flops_tok * T / t_dense / 1e9:16.1f}"
    )
    del dense
    for E in (8, 16, 32, 64, 128):
        m = build(moe_mod.MoE, d, E, K, h, score="sigmoid")
        w = stack_experts(m)
        ref, out = m(x), grouped_moe(m, x, w)
        rel = float((ref - out).abs().max() / ref.abs().max())
        assert rel < 2e-2, f"两种写法对不上：相对差异 {rel:.1e}"
        fl = flops_tok + 2 * d * E  # 再加上路由器（很小）
        t_loop = gpu_ms(lambda m=m: m(x))
        t_grp = gpu_ms(lambda m=m, w=w: grouped_moe(m, x, w))
        print(
            f"{'MoE E=' + str(E):13}{n_params(m) / 1e6:7.0f}M{fl / 1e9:15.4f}{T * K / E:18.0f}"
            f"{t_loop:14.2f}{t_grp:12.2f}{t_loop / t_grp:12.1f}×{fl * T / t_grp / 1e9:16.1f}"
        )
        del m, w, ref, out
        torch.cuda.empty_cache()
    print(
        "（每个 E 都先对拍：两种 MoE 写法输出的最大差异 < 最大输出的 2%，是 BF16 舍入误差的量级。）"
    )
    print(f"这台机器上 Python 每下发一个最小的 GPU 算子约 {launch_us():.1f} µs")
    kern = kernels_per_grouped_mm()
    print(
        f"一次 F.grouped_mm（128 个专家）在这张卡上发出的 GPU kernel：共 {sum(kern.values())} 个，"
        f"其中最多的一种 {max(kern.values())} 个（{max(kern, key=kern.get)[:40]}…）"
    )


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("本脚本需要 CUDA GPU；没有 GPU 可以跳过，正文里贴了一次 RTX 3090 上的结果。")
        sys.exit(0)
    t0 = time.perf_counter()
    torch.manual_seed(0)
    print(
        f"GPU：{torch.cuda.get_device_name(0)}，PyTorch {torch.__version__}，CUDA {torch.version.cuda}\n"
    )
    main()
    print(f"总用时 {time.perf_counter() - t0:.0f} s")
