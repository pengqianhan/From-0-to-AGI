"""Chapter 2 · GPU measurement: how much faster is matrix multiplication on a GPU?

Section 8 says: "The more work each call gives to the low-level library, the larger the advantage of
vectorization." It also says that a GPU is best at matrix multiplication.
This script calculates the same square matrix multiplication C = A @ B (A and B are both N×N)
on the CPU and on the GPU, for N from 64 to 8192:
- One N×N matrix multiplication has N³ multiply-adds = 2N³ floating-point operations (FLOP).
  TFLOPS = 2N³ / time / 10¹².
- The CPU and the GPU both use float32 (TF32 is off on the GPU, so it is true single precision).
  The last column is BF16 on the GPU (tensor cores).
- Timing: first a warmup, then many repeats; the result is the median. After each GPU call, the script
  calls torch.cuda.synchronize(). Thus it measures the time "from the start of one matrix multiplication
  in Python to the finished result". This time includes the fixed cost to start a GPU calculation.
- The data is already in CPU memory or GPU memory before the timing starts.
  The timing does not include copies between the CPU and the GPU.
Without a CUDA GPU, the script stops immediately. The number of CPU threads is fixed at CPU_THREADS.
Run: uv run python chapters/02-from-scalar-to-matrix/code/07_gpu_matmul.py
"""

import platform
import statistics
import sys
import time

import torch

CPU_THREADS = 8                                      # number of CPU threads: about the number of cores in a usual desktop computer
SIZES = [64, 128, 256, 512, 1024, 2048, 4096, 8192]


def median_time(fn, repeat: int, warmup: int = 3, sync: bool = False) -> float:
    """Run warmup times first, then run repeat times. Return the median time (seconds). With sync=True, wait for the GPU to finish each time."""
    for _ in range(warmup):
        fn()
    if sync:
        torch.cuda.synchronize()
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        if sync:
            torch.cuda.synchronize()                 # The GPU is asynchronous: if we do not wait for it, the time is wrong
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
    return platform.processor() or "unknown CPU"


def fmt_time(t: float) -> str:
    return f"{t * 1e3:9.3f} ms" if t < 1 else f"{t:9.2f} s "


def sweep():
    """Sweep the matrix size. Return [(N, CPU seconds, GPU FP32 seconds, GPU BF16 seconds)]."""
    rows = []
    for n in SIZES:
        a = torch.randn(n, n)                        # (N, N), float32, on the CPU
        b = torch.randn(n, n)
        a_gpu, b_gpu = a.cuda(), b.cuda()            # copy the same data to GPU memory
        a_bf, b_bf = a_gpu.bfloat16(), b_gpu.bfloat16()

        big = n >= 4096
        t_cpu = median_time(lambda a=a, b=b: a @ b, repeat=3 if big else 20, warmup=1 if big else 3)
        t_gpu = median_time(lambda a=a_gpu, b=b_gpu: a @ b, repeat=20, sync=True)
        t_bf = median_time(lambda a=a_bf, b=b_bf: a @ b, repeat=20, sync=True)

        # Parity check: CPU, GPU float32, and GPU BF16 calculate the same matrix multiplication
        # (BF16 has only about 3 significant digits, so we check only the order of magnitude of the error)
        ref = a @ b
        scale = ref.abs().max().item()
        err32 = ((a_gpu @ b_gpu).cpu() - ref).abs().max().item() / scale
        err16 = ((a_bf @ b_bf).float().cpu() - ref).abs().max().item() / scale
        assert err32 < 1e-4 and err16 < 5e-2, (n, err32, err16)
        rows.append((n, t_cpu, t_gpu, t_bf))
    return rows


def main() -> None:
    if not torch.cuda.is_available():
        print("This script needs a CUDA GPU. Without a GPU, skip it. The chapter text shows the results of one run on an RTX 3090.")
        sys.exit(0)

    torch.manual_seed(0)
    torch.set_num_threads(CPU_THREADS)
    torch.set_float32_matmul_precision("highest")    # no TF32: the GPU column is true float32
    print(f"GPU: {torch.cuda.get_device_name(0)}; PyTorch {torch.__version__}, CUDA {torch.version.cuda}")
    print(f"CPU: {cpu_name()}, {torch.get_num_threads()} threads")
    torch.ones(1).cuda()                             # initialize CUDA first, so that the timing does not include it

    rows = sweep()
    print("\nSquare matrix multiplication (N, N) @ (N, N), 2N³ floating-point operations each; median time")
    print("      N    CPU float32    GPU float32   GPU vs CPU   CPU TFLOPS  GPU TFLOPS  GPU BF16 TFLOPS")
    for n, t_cpu, t_gpu, t_bf in rows:
        flop = 2 * n**3
        print(f"  {n:>5}  {fmt_time(t_cpu)}   {fmt_time(t_gpu)}   {t_cpu / t_gpu:9.2f}×"
              f"   {flop / t_cpu / 1e12:10.3f}  {flop / t_gpu / 1e12:10.2f}  {flop / t_bf / 1e12:15.2f}")


if __name__ == "__main__":
    main()
