"""第 20 章 · GPU 实测：权重少一半，decode 就快近一倍吗？

第 3.1 节说：逐 token 生成（decode）时，每一步都要把全部权重从内存读一遍，瓶颈是带宽，
所以权重小一半，读得就快近一倍。这里在一张 GPU 上实测这笔账（需要 CUDA）。

按主线模型（configs/main/pretrain.toml：28 层、宽 1280、FFN 3584、16 个查询头 / 8 个 K/V 头、
共享 embedding 的输出层 65,536 × 1280）的全部矩阵搭一个 decode 步：batch 1，一个 token 的向量
乘过 28 × 7 个矩阵和输出层。只保留矩阵乘（注意力、KV cache、norm 读的字节很少，第 21 章另算）。
权重是随机的（N(0, 0.02²)，和初始化一样），这里只测速度，误差一列只作对照。几种格式和内核：

  BF16，cuBLAS              F.linear 的默认路径                                          16 bit/权重
  BF16，torch.compile       同一个矩阵–向量乘写成"逐元素乘再求和"，让 torch.compile 生成一个内核
  INT8，每行一个 scale        同样写法，但权重存 int8：编译出的内核直接读 int8、在芯片上转成 BF16
                            再乘加，最后乘 scale（反量化融合进矩阵乘）                        ≈ 8 bit
  INT4，每 32 个数一组        _weight_int4pack_mm（tinygemm 内核，torchao / gpt-fast 用的就是它）：
                            每组一个 bf16 scale + 一个 bf16 zero，ŵ = (q − 8)·scale + zero     5 bit
  INT8，先反量化再乘          w.to(bf16) * scale 先写出一份完整的 BF16 权重，再 F.linear（不融合）

一步有近 200 个小 kernel，从 Python 逐个发射的开销（每个几微秒）会盖过读显存的时间，所以用 CUDA Graph
把整步录下来、一次提交（推理引擎也是这么做的）。再看 batch 变大（一次同时生成 B 条）时 BF16 和 INT4 这笔账怎么变，
最后单看几种大小不同的矩阵，各自能跑到多少带宽。
没有 GPU 可以跳过；正文里贴了一次 RTX 3090 上的结果。
运行：uv run python chapters/20-release/code/05_gpu_quant_matvec.py
"""

from __future__ import annotations

import statistics
import sys
import time
import tomllib
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
CFG = ROOT / "configs" / "main" / "pretrain.toml"
HBM_BW = 936e9  # RTX 3090 显存带宽（规格表），字节/秒
GROUP = 32  # INT4 每 32 个数一组


def shapes(m: dict) -> list[tuple[str, int, int]]:
    """decode 一步要乘的全部矩阵：(名字, 输出维 N, 输入维 K)，和 nn.Linear 的 weight 一样是 N × K。"""
    d, hd, f = m["dim"], m["head_dim"], m["ffn_dim"]
    q, kv = m["n_heads"] * hd, m["n_kv_heads"] * hd
    layer = [("wq", q, d), ("wk", kv, d), ("wv", kv, d), ("wo", d, q),
             ("w_gate", f, d), ("w_up", f, d), ("w_down", d, f)]
    return layer * m["n_layers"] + [("lm_head", m["vocab_size"], d)]


# ── 量化：INT8 每行一个 scale；INT4 按 tinygemm 的格式每 32 个数一组 ─────────────────
def int8_rowwise(w: torch.Tensor):
    scale = w.float().abs().amax(1) / 127                                   # scale = max|w| / 127
    q = torch.round(w.float() / scale[:, None]).clamp(-127, 127).to(torch.int8)
    return q, scale.to(torch.bfloat16)


def int4_groupwise(w: torch.Tensor):
    """每 GROUP 个数一组：q = round((w − min) / scale) ∈ [0, 15]，ŵ = (q − 8)·scale + zero，zero = min + 8·scale。"""
    N, K = w.shape
    g = w.float().view(N, K // GROUP, GROUP)
    mn, mx = g.amin(-1), g.amax(-1)
    scale = ((mx - mn) / 15).clamp(min=1e-8)
    q = torch.round((g - mn[..., None]) / scale[..., None]).clamp(0, 15).to(torch.int32)
    zero = mn + 8 * scale
    deq = ((q - 8) * scale[..., None] + zero[..., None]).view(N, K)         # 参考：反量化后的权重
    q = q.view(N, K)
    packed = torch.ops.aten._convert_weight_to_int4pack((q[:, ::2] << 4 | q[:, 1::2]).to(torch.uint8), 8)
    sz = torch.stack([scale, zero], -1).transpose(0, 1).contiguous().to(torch.bfloat16)  # (K/32, N, 2)
    return packed, sz, deq


# ── batch 1 的矩阵–向量乘写成"逐元素乘再沿 K 求和"，torch.compile 把它编译成一个内核 ──────
@torch.compile(dynamic=False)
def matvec_bf16(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:  # x (1, K)，w (N, K) → (N,)
    return (x * w).sum(-1)


@torch.compile(dynamic=False)
def matvec_int8(x: torch.Tensor, q: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    return (x * q.to(x.dtype)).sum(-1) * s  # 读 int8 → 在芯片上转 BF16 → 乘加 → 乘 scale，一个内核


# ── 计时：CUDA Graph 录下整步，重放多次取中位数 ─────────────────────────────────
def graph_ms(fn, budget_s: float = 0.3, rounds: int = 5) -> float:
    """录下 fn 为一个 CUDA Graph；每轮重放若干次取平均，共 rounds 轮，返回每步毫秒数的中位数。
    每轮的重放次数按单步耗时定，让每轮大约 budget_s 秒（慢的格式少放几次，总时间可控）。"""
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):  # 录制前先在旁路 stream 上跑两遍（预热、编译、让 cuBLAS 选好算法）
        for _ in range(2):
            fn()
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    graph.replay()  # 预热一次
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    graph.replay()
    b.record()
    torch.cuda.synchronize()
    reps = max(2, min(50, int(budget_s * 1e3 / a.elapsed_time(b))))
    times = []
    for _ in range(rounds):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(reps):
            graph.replay()
        b.record()
        torch.cuda.synchronize()
        times.append(a.elapsed_time(b) / reps)
    del graph
    return statistics.median(times)


def copy_bandwidth(dev: str) -> float:
    """复制一个 1 GiB 的 BF16 张量：读 1 GiB + 写 1 GiB，返回字节/秒。"""
    src = torch.empty(2**29, device=dev, dtype=torch.bfloat16)
    dst = torch.empty_like(src)
    return 2 * src.numel() * 2 / (graph_ms(lambda: dst.copy_(src)) / 1e3)


def main() -> None:
    if not torch.cuda.is_available():
        print("本脚本需要 CUDA GPU；没有 GPU 可以跳过，正文里贴了一次 RTX 3090 上的结果。")
        sys.exit(0)
    T0 = time.time()
    torch.manual_seed(0)
    dev = "cuda"
    print(f"GPU：{torch.cuda.get_device_name(0)}，PyTorch {torch.__version__}，CUDA {torch.version.cuda}")

    m = tomllib.loads(CFG.read_text(encoding="utf-8"))["model"]
    mats = shapes(m)
    n_params = sum(n * k for _, n, k in mats)
    n_rows = sum(n for _, n, _ in mats)
    W, Q8, Q4 = [], [], []
    err = {"int8": [], "int4": []}
    xs = {k: torch.randn(1, k, device=dev, dtype=torch.bfloat16) for k in {k for _, _, k in mats}}
    for _, n, k in mats:
        w = (torch.randn(n, k, device=dev) * 0.02).to(torch.bfloat16)
        q8, s8 = int8_rowwise(w)
        p4, sz, deq4 = int4_groupwise(w)
        x = xs[k]
        ref = x.float() @ w.float().T                                        # BF16 权重的精确输出
        y8 = (x.float() @ q8.float().T) * s8.float()
        y4 = torch.ops.aten._weight_int4pack_mm(x, p4, GROUP, sz).float()
        y4_ref = x.float() @ deq4.T                                          # 用反量化权重手算
        assert (y4 - y4_ref).norm() / y4_ref.norm() < 1e-2, "INT4 打包格式不对"
        err["int8"].append(float((y8 - ref).norm() / ref.norm()))
        err["int4"].append(float((y4 - ref).norm() / ref.norm()))
        W.append(w)
        Q8.append((q8, s8))
        Q4.append((p4, sz))
    print(f"主线模型的全部矩阵：28 层 × 7 + 输出层 = {len(mats)} 个，{n_params / 1e6:.1f}M 参数"
          "（不含一维的 norm 权重）")

    def run(fmt: str, B: int = 1, sel: list[int] | None = None):
        """录下"乘过 sel 里的矩阵"（默认全部 197 个）为一步，返回毫秒数。"""
        x = {k: torch.randn(B, k, device=dev, dtype=torch.bfloat16) for k in xs}
        idx = range(len(mats)) if sel is None else sel

        def step():
            for i in idx:
                (_, _, k), w, (q8, s8), (p4, sz) = mats[i], W[i], Q8[i], Q4[i]
                if fmt == "bf16":
                    F.linear(x[k], w)
                elif fmt == "bf16c":
                    matvec_bf16(x[k], w)
                elif fmt == "int8c":
                    matvec_int8(x[k], q8, s8)
                elif fmt == "int4":
                    torch.ops.aten._weight_int4pack_mm(x[k], p4, GROUP, sz)
                else:  # 先反量化成一份完整的 BF16 权重，再乘
                    F.linear(x[k], q8.to(torch.bfloat16) * s8[:, None])

        return graph_ms(step)

    b8 = n_params + 2 * n_rows  # int8 权重 + 每行一个 bf16 scale
    b4 = n_params // 2 + n_params // GROUP * 4  # 4 bit 权重 + 每组两个 bf16
    rows = [  # (名字, 格式, 每步要读的权重字节, 误差)
        ("BF16，cuBLAS（F.linear）", "bf16", 2 * n_params, None),
        ("BF16，torch.compile 生成的内核", "bf16c", 2 * n_params, None),
        ("INT8，torch.compile 融合反量化", "int8c", b8, "int8"),
        ("INT4，tinygemm 内核", "int4", b4, "int4"),
        ("INT8，先反量化成 BF16 再乘", "naive", b8, "int8"),
    ]
    copy_bw = copy_bandwidth(dev)
    print(f"参照：这张卡上复制一个 1 GiB 的张量（读 + 写各 1 GiB），实测 {copy_bw / 1e9:.0f} GB/s"
          f"（规格表 936 GB/s 的 {100 * copy_bw / HBM_BW:.0f}%），这是实际能跑到的带宽上限的一个参照")
    print("\n① batch 1 的 decode 一步（只有矩阵乘）：")
    print(f"   {'格式与内核':28s} {'bit/权重':>8s} {'权重字节':>9s} {'一步耗时':>9s} {'等效带宽':>10s} "
          f"{'占936GB/s':>9s} {'比cuBLAS快':>9s} {'输出相对误差':>10s}")
    base = None
    for name, fmt, nbytes, e in rows:
        ms = run(fmt)
        base = base or ms
        bw = nbytes / (ms / 1e3)
        es = "—" if e is None else f"{100 * statistics.mean(err[e]):.2f}%"
        print(f"   {name:28s} {8 * nbytes / n_params:8.2f} {nbytes / 1e9:7.3f}GB {ms:7.3f}ms "
              f"{bw / 1e9:6.0f}GB/s {100 * bw / HBM_BW:8.0f}% {base / ms:8.2f}× {es:>10s}")
    print(f"   （理论下限 = 权重字节 ÷ 936 GB/s：BF16 {2 * n_params / HBM_BW * 1e3:.3f} ms，"
          f"INT8 {b8 / HBM_BW * 1e3:.3f} ms，INT4 {b4 / HBM_BW * 1e3:.3f} ms。"
          "等效带宽 = 权重字节 ÷ 耗时；最后一行实际还要多写、多读一份 BF16 权重）")

    print("\n② 一次同时生成 B 条（batch B），一步耗时（毫秒）：")
    print(f"   {'B':>4s} {'BF16 cuBLAS':>12s} {'INT4 tinygemm':>14s} {'INT4 比 BF16 快':>14s}")
    for B in (1, 8, 32, 128, 512):
        t16, t4 = run("bf16", B), run("int4", B)
        print(f"   {B:4d} {t16:12.3f} {t4:14.3f} {t16 / t4:13.2f}×")
    print("\n③ 为什么离理论下限还远：单看一种矩阵（28 层各乘一次，取平均），batch 1 的等效带宽（GB/s）")
    print(f"   {'矩阵':8s} {'形状 N×K':>12s} {'BF16 大小':>10s} {'BF16 cuBLAS':>12s} {'INT8 编译':>10s} "
          f"{'INT4 tinygemm':>14s}")
    for nm in ("wk", "w_up", "lm_head"):
        sel = [i for i, (name, _, _) in enumerate(mats) if name == nm]
        _, n, k = mats[sel[0]]
        gbs = []
        for fmt, nbytes in (("bf16", 2 * n * k), ("int8c", n * k + 2 * n), ("int4", n * k // 2 + n * k // GROUP * 4)):
            ms = run(fmt, sel=sel) / len(sel)
            gbs.append(nbytes / (ms / 1e3) / 1e9)
        print(f"   {nm:8s} {f'{n}×{k}':>12s} {2 * n * k / 2**20:8.1f}MiB {gbs[0]:12.0f} {gbs[1]:10.0f} "
              f"{gbs[2]:14.0f}")

    print(f"\n显存峰值 {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB，"
          f"总耗时 {time.time() - T0:.0f}s")


if __name__ == "__main__":
    main()
