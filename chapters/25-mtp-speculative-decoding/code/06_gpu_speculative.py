"""第 25 章 · GPU 实测：decode 在等数据，所以"一次验证 k 个"几乎免费；推测解码在 GPU 上能快多少

正文的计时都来自单线程 CPU。这个脚本把同样的实验搬到一张 GPU 上（需要 CUDA）：
  1. 已有 200 个位置的 KV cache，目标模型一次前向喂 T 个新 token 要多久？
     - 本章的小目标模型（0.86M 参数，FP32）；
     - 同一个 TinyLM 结构放大到约 4.1B 参数（宽 8192、6 层，BF16）："放大版目标"。读一遍权重要 8.3 GB，
       落在 decode 的带宽受限区。放大版是随机初始化的——这里只测时间，前向耗时和权重的数值无关；
  2. 本章训练好的目标 + 草稿在 GPU 上跑一遍贪心推测解码（4 段 × 200 个字符，用的就是 02 的
     speculative_greedy）：GPU 上是否仍然逐字相同，并记下每一轮接受了几个；
  3. 放大版目标 + 放大版草稿（1 层、宽 1024，约为目标的 1/300）：按第 2 部分每一轮真实的接受个数，
     把推测解码的全部前向、同步和回滚"回放"一遍，测墙钟加速比，和公式
     (1 − α^{k+1}) / ((1 − α)(1 + k·c)) 对照。（随机权重的两个模型之间谈不上"猜得准不准"，
     所以接受几个照抄真实模型那一轮的结果；为了控制总时长，只回放前 2 段提示词。）

第 10 章的注意力用 torch.arange 在 CPU 上建因果掩码；这里只把这一处改成建在 GPU 上，其余原样复用。
运行：uv run python chapters/25-mtp-speculative-decoding/code/06_gpu_speculative.py
（先运行 01，让目标 / 草稿的权重缓存在 out/ 里；之后约 1–2 分钟，需要约 10 GB 显存）
"""

from __future__ import annotations

import importlib.util
import math
import statistics
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
DEV = "cuda"
BIG_TARGET = dict(dim=8192, n_layers=6, n_heads=64, n_kv_heads=8, ffn_dim=22016)
BIG_DRAFT = dict(dim=1024, n_layers=1, n_heads=8, n_kv_heads=8, ffn_dim=2816)
KS = (1, 2, 3, 4, 6, 8)


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def patch_attention(ch10) -> None:
    """第 10 章 Attention.forward 的原样拷贝，只把因果掩码的 arange 建在 x 所在的设备上。"""

    def forward(self, x, cos, sin, cache, layer):
        B, T, _ = x.shape
        c = self.c
        q = self.wq(x).view(B, T, c.n_heads, c.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, c.n_kv_heads, c.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, c.n_kv_heads, c.head_dim).transpose(1, 2)
        q, k = ch10.apply_rope(q, cos, sin), ch10.apply_rope(k, cos, sin)
        if cache is not None:
            k, v = cache.append(layer, k, v)
        S = k.shape[2]
        g = c.n_heads // c.n_kv_heads
        k, v = k.repeat_interleave(g, dim=1), v.repeat_interleave(g, dim=1)
        att = q @ k.transpose(-2, -1) / math.sqrt(c.head_dim)
        i = torch.arange(T, device=x.device)[:, None] + (S - T)  # ← 唯一的改动：device=x.device
        j = torch.arange(S, device=x.device)[None, :]
        att = att.masked_fill(j > i, float("-inf")).softmax(-1)
        return self.wo((att @ v).transpose(1, 2).reshape(B, T, -1))

    ch10.Attention.forward = forward


class OnGPU(torch.nn.Module):
    """把 CPU 上的 token id 搬到 GPU 再前向——这样 02 的 greedy_generate / speculative_greedy 可以原样调用。"""

    def __init__(self, model) -> None:
        super().__init__()
        self.model, self.c = model, model.c

    def forward(self, ids, cache=None):
        return self.model(ids.to(DEV), cache)


def build_big(ch10, vocab_size: int, cfg: dict):
    """直接在 GPU 上、直接用 BF16 建放大版（4B 参数先建 FP32 再转会超出 24 GB 显存）。"""
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    with torch.device(DEV):
        model = ch10.TinyLM(ch10.Config(vocab_size=vocab_size, **cfg))
    torch.set_default_dtype(old)
    return model.to(torch.bfloat16).eval()  # RoPE 表也转成 BF16，和激活值同一精度


def n_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


def n_bytes(model) -> int:
    return sum(p.numel() * p.element_size() for p in model.parameters())


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
def forward_ms(model, ctx: int, n_new: int, reps: int = 30) -> float:
    """已有 ctx 个位置的缓存时，一次前向喂 n_new 个新 token 的耗时（毫秒，CUDA event 计时，中位数）。"""
    g = torch.Generator(device=DEV).manual_seed(0)
    cache = m1.KVCache(model.c.n_layers)
    model(torch.randint(0, model.c.vocab_size, (1, ctx), generator=g, device=DEV), cache)
    new = torch.randint(0, model.c.vocab_size, (1, n_new), generator=g, device=DEV)
    times = []
    for i in range(reps + 5):  # 前 5 次是预热
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        model(new, cache)
        end.record()
        torch.cuda.synchronize()
        if i >= 5:
            times.append(start.elapsed_time(end))
        m1.truncate(cache, ctx)  # 每次都回到同样的起点
    return statistics.median(times)


def wall(fn, *args):
    """墙钟时间（秒）：前后都等 GPU 算完。"""
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn(*args)
    torch.cuda.synchronize()
    return time.perf_counter() - t0, out


@torch.no_grad()
def replay(target, draft, prompt: list[int], k: int, accepted: list[int]) -> int:
    """按给定的逐轮接受数，把 02 的 speculative_greedy 的每一步重走一遍：草稿补喂 + 猜 k 个、
    目标一次前向验证、取 argmax（同样会等 GPU）、回滚两个缓存。只有"接受几个"不靠比较，而是照抄。"""
    seq = list(prompt)
    tc, dc = m1.KVCache(target.c.n_layers), m1.KVCache(draft.c.n_layers)
    for m in accepted:
        logits = draft(torch.tensor([seq[len(dc) :]]), dc)[0, -1]
        drafts = []
        for i in range(k):
            drafts.append(int(logits.argmax()))
            if i < k - 1:
                logits = draft(torch.tensor([[drafts[-1]]]), dc)[0, -1]
        choice = (
            target(torch.tensor([seq[len(tc) :] + drafts]), tc)[0, -(k + 1) :].argmax(-1).tolist()
        )
        seq += drafts[:m] + [choice[m]]
        m1.truncate(tc, len(seq) - 1)
        m1.truncate(dc, min(len(dc), len(seq) - 1))
    return len(seq) - len(prompt)


def greedy_all(model, P, N):
    return [m2.greedy_generate(model, p, N) for p in P]


def replay_all(target, draft, P, k, traces_k):
    return sum(replay(target, draft, p, k, tr) for p, tr in zip(P, traces_k))


def expected_tokens(alpha: float, k: int) -> float:
    return (1 - alpha ** (k + 1)) / (1 - alpha)


def alpha_of(traces_k: list[list[int]], k: int) -> float:
    """逐 token 接受率 = 接受数 ÷ 被比较过的草稿数（和 02 的算法一样）。"""
    acc = sum(m for tr in traces_k for m in tr)
    examined = sum(m + (1 if m < k else 0) for tr in traces_k for m in tr)
    return acc / examined


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("本脚本需要 CUDA GPU；没有 GPU 可以跳过，正文里贴了一次 RTX 3090 上的结果。")
        sys.exit(0)
    t_start = time.perf_counter()
    torch.manual_seed(0)
    m1 = _load("ch25_models", "01_models_and_cost.py")
    m2 = _load("ch25_greedy", "02_greedy_speculative.py")
    ch10 = m1.ch10
    patch_attention(ch10)
    print(
        f"GPU：{torch.cuda.get_device_name(0)}，PyTorch {torch.__version__}，CUDA {torch.version.cuda}"
    )

    target, draft = m1.load_target().to(DEV), m1.load_draft().to(DEV)  # 权重来自 01 的缓存（FP32）
    big_t = build_big(ch10, target.c.vocab_size, BIG_TARGET)
    big_d = build_big(ch10, target.c.vocab_size, BIG_DRAFT)
    print(
        f"本章目标 {n_params(target) / 1e6:.2f}M 参数（FP32，{n_bytes(target) / 1e6:.1f} MB）；"
        f"放大版目标 {n_params(big_t) / 1e9:.2f}B 参数（BF16，{n_bytes(big_t) / 1e9:.2f} GB），"
        f"放大版草稿 {n_params(big_d) / 1e6:.1f}M 参数（目标的 1/{n_params(big_t) / n_params(big_d):.0f}）"
    )
    print(f"这台机器上 Python 每下发一个最小的 GPU 算子约 {launch_us():.1f} µs")

    # ── 1. 一次前向喂 T 个 token ──
    print(
        "\n── 1. 已有 200 个位置的 KV cache，一次前向喂 T 个新 token 的耗时（CUDA event，30 次中位数）──"
    )
    print(f"{'T':>4} {'本章目标 ms':>11} {'相对 T=1':>8} {'放大版目标 ms':>13} {'相对 T=1':>8}")
    base_s = base_b = None
    for T in (1, 2, 4, 8, 16, 64, 256):
        ts, tb = forward_ms(target, 200, T), forward_ms(big_t, 200, T)
        base_s, base_b = base_s or ts, base_b or tb
        print(f"{T:4d} {ts:11.3f} {ts / base_s:7.2f}× {tb:13.3f} {tb / base_b:7.2f}×")
    print(
        f"放大版 T=1：{n_bytes(big_t) / 1e9:.2f} GB 权重 / {base_b:.2f} ms = 有效带宽 "
        f"{n_bytes(big_t) / base_b / 1e6:.0f} GB/s；本章目标 T=1：{n_bytes(target) / 1e6:.1f} MB / "
        f"{base_s:.2f} ms = {n_bytes(target) / base_s / 1e6:.1f} GB/s"
    )

    # ── 2. 本章的目标 + 草稿：GPU 上的真实推测解码（正确性 + 逐轮接受数）──
    P, N = m2.prompts(), 200
    T_, D_ = OnGPU(target), OnGPU(draft)
    base_out = greedy_all(T_, P, N)
    print(f"\n── 2. 本章的目标 + 草稿搬到 GPU（FP32，贪心，4 段 × {N} 个字符）──")
    print(f"{'k':>2} {'与目标贪心逐字相同':>12} {'接受率α':>7} {'每轮产出':>7} {'目标前向次数':>9}")
    traces = {}
    for k in KS:
        trace: list = []
        res = [m2.speculative_greedy(T_, D_, p, N, k, trace) for p in P]
        same = all(r[0] == b for r, b in zip(res, base_out))
        # trace 按提示词顺序连在一起：按每段的轮数切开，得到每段的逐轮接受数
        cuts = [0]
        for r in res:
            cuts.append(cuts[-1] + r[1]["rounds"])
        traces[k] = [[m for _, _, m in trace[a:b]] for a, b in zip(cuts, cuts[1:])]
        rounds = cuts[-1]
        print(
            f"{k:2d} {str(same):>17} {alpha_of(traces[k], k):9.3f} "
            f"{(sum(map(sum, traces[k])) + rounds) / rounds:10.2f} "
            f"{rounds:14d}"
        )

    # ── 3. 放大版：按真实的逐轮接受数回放 ──
    BT, BD = OnGPU(big_t), OnGPU(big_d)
    P3 = P[:2]
    c_big = forward_ms(big_d, 200, 1) / forward_ms(big_t, 200, 1)
    replay(BT, BD, P3[0], 3, traces[3][0][:5])  # 预热
    tb_big, ts_big = [], {k: [] for k in KS}
    for _ in range(3):
        tb_big.append(wall(greedy_all, BT, P3, N)[0])
        for k in KS:
            ts_big[k].append(wall(replay_all, BT, BD, P3, k, traces[k][: len(P3)])[0])
    tbb = statistics.median(tb_big)
    print(
        f"\n── 3. 放大版目标（{n_params(big_t) / 1e9:.2f}B）+ 放大版草稿（{n_params(big_d) / 1e6:.0f}M），"
        f"BF16，按第 2 部分前 {len(P3)} 段的逐轮接受数回放，墙钟时间 3 次中位数 ──"
    )
    print(f"普通贪心解码 {len(P3) * N} 个 token：{tbb:.2f} s；成本系数 c ≈ {c_big:.3f}")
    print(f"{'k':>2} {'接受率α':>7} {'每轮产出':>7} {'墙钟 s':>7} {'加速比':>6} {'公式预测':>7}")
    for k in KS:
        a = alpha_of(traces[k][: len(P3)], k)
        rounds = sum(len(tr) for tr in traces[k][: len(P3)])
        ts = statistics.median(ts_big[k])
        print(
            f"{k:2d} {a:9.3f} {(sum(map(sum, traces[k][: len(P3)])) + rounds) / rounds:10.2f} "
            f"{ts:8.2f} {tbb / ts:7.2f}× {expected_tokens(a, k) / (1 + k * c_big):8.2f}×"
        )
    print(f"\n总用时 {time.perf_counter() - t_start:.0f} s")
