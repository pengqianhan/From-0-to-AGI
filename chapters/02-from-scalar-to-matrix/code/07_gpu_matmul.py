"""第 2 章 · GPU 实测：矩阵乘法搬到 GPU 上，到底快多少？

第 8 节说"每次交给底层库的活越大，向量化的优势越明显"，又说 GPU 最擅长的恰恰是矩阵乘法。
这个脚本把同一个方阵乘法 C = A @ B（A、B 都是 N×N）分别放在 CPU 和 GPU 上算，N 从 64 扫到 8192：
- 一次 N×N 矩阵乘法有 N³ 次乘加 = 2N³ 次浮点运算（FLOP）；TFLOPS = 2N³ / 耗时 / 10¹²。
- CPU 和 GPU 都用 float32（GPU 关掉 TF32，是真正的单精度）；最后一列是 GPU 上的 BF16（张量核）。
- 计时：先预热，再重复多次取中位数。GPU 每次都 torch.cuda.synchronize()，测的是
  "从 Python 发起一次矩阵乘，到结果算完"的时间，包含启动 GPU 计算的固定开销。
- 数据事先放好在各自的内存/显存里，不算 CPU 和 GPU 之间来回拷贝的时间。
没有 CUDA GPU 时直接退出。CPU 线程数固定为 CPU_THREADS。
运行：uv run python chapters/02-from-scalar-to-matrix/code/07_gpu_matmul.py
"""

import platform
import statistics
import sys
import time

import torch

CPU_THREADS = 8                                      # CPU 用几个线程：普通台式机大约这么多核
SIZES = [64, 128, 256, 512, 1024, 2048, 4096, 8192]


def median_time(fn, repeat: int, warmup: int = 3, sync: bool = False) -> float:
    """先预热 warmup 次，再跑 repeat 次，返回耗时的中位数（秒）。sync=True 时每次都等 GPU 算完。"""
    for _ in range(warmup):
        fn()
    if sync:
        torch.cuda.synchronize()
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        if sync:
            torch.cuda.synchronize()                 # GPU 是异步的：不等它算完，计时就是假的
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def cpu_name() -> str:
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "未知 CPU"


def fmt_time(t: float) -> str:
    return f"{t * 1e3:9.3f} ms" if t < 1 else f"{t:9.2f} s "


def sweep():
    """扫描矩阵规模。返回 [(N, CPU 秒, GPU FP32 秒, GPU BF16 秒)]。"""
    rows = []
    for n in SIZES:
        a = torch.randn(n, n)                        # (N, N)，float32，在 CPU 上
        b = torch.randn(n, n)
        a_gpu, b_gpu = a.cuda(), b.cuda()            # 同样的数据拷一份到显存
        a_bf, b_bf = a_gpu.bfloat16(), b_gpu.bfloat16()

        big = n >= 4096
        t_cpu = median_time(lambda a=a, b=b: a @ b, repeat=3 if big else 20, warmup=1 if big else 3)
        t_gpu = median_time(lambda a=a_gpu, b=b_gpu: a @ b, repeat=20, sync=True)
        t_bf = median_time(lambda a=a_bf, b=b_bf: a @ b, repeat=20, sync=True)

        # 对拍：CPU、GPU float32、GPU BF16 算的是同一个矩阵乘法（BF16 只有约 3 位有效数字，只看误差量级）
        ref = a @ b
        scale = ref.abs().max().item()
        err32 = ((a_gpu @ b_gpu).cpu() - ref).abs().max().item() / scale
        err16 = ((a_bf @ b_bf).float().cpu() - ref).abs().max().item() / scale
        assert err32 < 1e-4 and err16 < 5e-2, (n, err32, err16)
        rows.append((n, t_cpu, t_gpu, t_bf))
    return rows


def main() -> None:
    if not torch.cuda.is_available():
        print("本脚本需要 CUDA GPU；没有 GPU 可以跳过，正文里贴了一次 RTX 3090 上的结果。")
        sys.exit(0)

    torch.manual_seed(0)
    torch.set_num_threads(CPU_THREADS)
    torch.set_float32_matmul_precision("highest")    # 不用 TF32：GPU 这一列是真正的 float32
    print(f"GPU：{torch.cuda.get_device_name(0)}；PyTorch {torch.__version__}，CUDA {torch.version.cuda}")
    print(f"CPU：{cpu_name()}，用 {torch.get_num_threads()} 个线程")
    torch.ones(1).cuda()                             # 先把 CUDA 初始化掉，不算进计时

    rows = sweep()
    print("\n方阵乘法 (N, N) @ (N, N)，每次 2N³ 次浮点运算；耗时取中位数")
    print("      N    CPU float32    GPU float32   GPU 比 CPU 快   CPU TFLOPS  GPU TFLOPS  GPU BF16 TFLOPS")
    for n, t_cpu, t_gpu, t_bf in rows:
        flop = 2 * n**3
        print(f"  {n:>5}  {fmt_time(t_cpu)}   {fmt_time(t_gpu)}   {t_cpu / t_gpu:9.2f} 倍"
              f"   {flop / t_cpu / 1e12:10.3f}  {flop / t_gpu / 1e12:10.2f}  {flop / t_bf / 1e12:15.2f}")


if __name__ == "__main__":
    main()
