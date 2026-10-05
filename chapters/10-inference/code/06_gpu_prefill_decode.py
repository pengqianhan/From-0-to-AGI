"""第 10 章 · GPU 脚本 6：在 GPU 上看 KV cache，和"prefill 吃算力、decode 吃带宽"

CPU 上（03_kv_cache.py）看到了两件事：KV cache 让生成快了好几倍；prefill 的吞吐是 decode 的几十倍。
这里换一张 GPU、一个大得多的模型再看一遍：
  1. 朴素生成 vs KV cache：平均每生成一个 token 要多久（直接调用 03 的 generate_naive / generate_cached）；
  2. 一次前向喂 T 个 token（就是 prefill T 个 token），T 从 1 到 2048：
     T 小的时候耗时几乎不变——瓶颈是把全部权重从显存读一遍；T 大了耗时才随 T 线性增长——瓶颈变成算力；
  3. decode 每条序列每步只喂 1 个 token，把 batch 从 1 加到 64：读一遍权重，服务 64 条序列。

计时有两种：
  - 逐个发射（eager）：就是平常那样调用模型，Python 每调用一个算子就向 GPU 发射一个 kernel；
  - CUDA graph：先把一次前向的全部 kernel 录下来，之后整张图一次性重放，不再经过 Python。
    量到的是 GPU 自己干活的时间（vLLM 等推理引擎在 decode 阶段也用这招省掉发射开销）。

模型：01_tiny_model.py 里的 TinyLM，代码原样 importlib 加载，只换尺寸：取 Llama 3 8B 一层的形状
（d=4096、32 个查询头、8 个 KV 头、FFN 14336），叠 4 层，约 0.87B 参数，BF16。
层数少、每层宽，是为了让"读权重"而不是"Python 发射 kernel"成为主要开销。
权重是随机初始化的——这里只测速度，不看生成的内容。
需要 CUDA GPU（约 10 GB 显存），1 分钟左右。
运行：uv run python chapters/10-inference/code/06_gpu_prefill_decode.py
"""

from __future__ import annotations

import importlib.util
import statistics
import sys
import time
from pathlib import Path

import torch


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


tiny = _load("tiny_model", "01_tiny_model.py")
kvc = _load("kv_cache_demo", "03_kv_cache.py")

SIZE = dict(dim=4096, n_layers=4, n_heads=32, n_kv_heads=8, ffn_dim=14336, max_seq_len=2560)
GEN_LENGTHS = [64, 256, 512]
PREFILL_LENGTHS = [1, 8, 32, 128, 512, 2048]
BATCHES = [1, 8, 32, 64]
CONTEXTS = [64, 512]  # decode 测速时缓存里已有的位置数
SPEC = {"RTX 3090": (71.0, 936.0)}  # 规格表：BF16 张量核稠密峰值 TFLOPS、显存带宽 GB/s


def cuda_time(fn, reps: int) -> float:
    """CUDA event 计时，先预热两次，返回中位数（毫秒）。"""
    fn()
    fn()
    times = []
    for _ in range(reps):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    return statistics.median(times)


def graph_time(fn, reps: int) -> float:
    """把 fn 的全部 kernel 录成一张 CUDA graph，重放计时（中位数，毫秒）。"""
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):  # 录制前先在旁路 stream 上预热
        fn()
        fn()
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    ms = cuda_time(graph.replay, reps)
    del graph
    torch.cuda.empty_cache()
    return ms


def wall_time(fn, reps: int) -> float:
    """整段生成（每步都要把 token 取回 CPU）用墙钟计时，返回中位数（秒）。"""
    times = []
    for _ in range(reps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def pct(x: float, peak: float | None) -> str:
    return f"（{x / peak:4.0%}）" if peak else ""


def main() -> None:
    if not torch.cuda.is_available():
        print("本脚本需要 CUDA GPU；没有 GPU 可以跳过，正文里贴了一次 RTX 3090 上的结果。")
        sys.exit(0)
    name = torch.cuda.get_device_name(0)
    print(f"GPU：{name}  torch {torch.__version__}  CUDA {torch.version.cuda}")
    peak_tf, peak_bw = next((v for k, v in SPEC.items() if k in name), (None, None))
    torch.manual_seed(0)

    with torch.device("cuda"), torch.no_grad():  # 01/03 里新建的张量（ids、掩码）也都放到 GPU 上
        data = tiny.CharData()
        c = tiny.Config(vocab_size=data.vocab_size, **SIZE)
        model = tiny.TinyLM(c).to(torch.bfloat16).eval()
        n_params = sum(p.numel() for p in model.parameters())
        w_bytes = n_params * 2
        d_attn = c.n_heads * c.head_dim
        print(f"模型：{c.n_layers} 层，d={c.dim}，{c.n_heads} 个查询头 / {c.n_kv_heads} 个 KV 头，"
              f"head_dim={c.head_dim}，FFN {c.ffn_dim}；{n_params / 1e9:.2f}B 参数，"
              f"BF16 权重 {w_bytes / 2**30:.2f} GiB")
        if peak_tf:
            print(f"规格表峰值：BF16 {peak_tf:.0f} TFLOPS，显存带宽 {peak_bw:.0f} GB/s；"
                  f"两者之比 ≈ {peak_tf * 1e12 / (peak_bw * 1e9):.0f} FLOP/字节。"
                  f"读一遍权重至少要 {w_bytes / (peak_bw * 1e9) * 1e3:.2f} ms")

        # ① 朴素 vs KV cache（03 的生成函数，每步都要把 token 取回 CPU，只能 eager）
        prompt = data.encode(kvc.PROMPT)
        kvc.generate_naive(model, prompt, 8, temperature=0)  # 预热
        kvc.generate_cached(model, prompt, 8, temperature=0)
        print(f"\n1. 朴素生成 vs KV cache（贪心，提示词 {len(prompt)} 个字符，eager，每项 3 次取中位数）")
        print("   新生成   朴素 ms/token   KV cache ms/token   加速")
        for n in GEN_LENGTHS:
            t_naive = wall_time(lambda n=n: kvc.generate_naive(model, prompt, n, temperature=0), 3)
            t_cache = wall_time(lambda n=n: kvc.generate_cached(model, prompt, n, temperature=0), 3)
            print(f"   {n:6d}   {t_naive / n * 1e3:13.2f}   {t_cache / n * 1e3:17.2f}"
                  f"   {t_naive / t_cache:5.1f}×")

        # ② 一次前向喂 T 个 token（prefill）
        print("\n2. 一次前向喂 T 个 token（prefill；每项 10 次取中位数；后三列按 CUDA graph 的时间算）")
        print("      T   eager ms   graph ms      token/s   算力 TFLOPS（占峰值）   读权重 GB/s（占峰值）")
        for T in PREFILL_LENGTHS:
            ids = torch.randint(0, c.vocab_size, (1, T))
            eager = cuda_time(lambda ids=ids: model(ids), 10)
            ms = graph_time(lambda ids=ids: model(ids), 10)
            flops = 2 * n_params * T + 2 * c.n_layers * d_attn * T * T  # 矩阵乘法 + 因果注意力
            tflops, gbps = flops / ms / 1e9, w_bytes / ms / 1e6
            print(f"   {T:5d}  {eager:8.2f}  {ms:8.2f}  {T / ms * 1e3:11,.0f}   {tflops:8.1f}{pct(tflops, peak_tf)}"
                  f"   {gbps:10.0f}{pct(gbps, peak_bw)}")

        # ③ decode：缓存里已有 ctx 个位置，每条序列每步喂 1 个新 token
        print("\n3. decode：每条序列每步喂 1 个 token，batch 从 1 到 64（CUDA graph，每项 20 次取中位数）")
        print("   上下文  batch   每步 ms      token/s   KV cache MiB   读显存 GB/s（占峰值）")
        for ctx in CONTEXTS:
            for B in BATCHES:
                cache = tiny.KVCache(c.n_layers)
                model(torch.randint(0, c.vocab_size, (B, ctx)), cache)
                k0, v0 = list(cache.k), list(cache.v)
                x = torch.randint(0, c.vocab_size, (B, 1))

                def step(cache=cache, k0=k0, v0=v0, x=x):
                    cache.k, cache.v = list(k0), list(v0)  # 每次都从同样的 ctx 个位置开始
                    model(x, cache)

                ms = graph_time(step, 20)
                kv_bytes = sum(t.numel() * t.element_size() for t in k0 + v0)
                gbps = (w_bytes + kv_bytes) / ms / 1e6  # 至少要读：全部权重 + 全部 KV cache
                print(f"   {ctx:6d}  {B:5d}  {ms:8.2f}  {B / ms * 1e3:11,.0f}   {kv_bytes / 2**20:12.0f}"
                      f"   {gbps:10.0f}{pct(gbps, peak_bw)}")
                del cache, k0, v0


if __name__ == "__main__":
    main()
