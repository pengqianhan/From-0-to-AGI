"""第 16 章 · GPU 实测：打包省下多少时间，隔离对话又要花多少

第 5 节在 CPU 上数的是 token：同样 200 条对话，一条一行补齐到 513 时真实 token 只占 43%，
首次适配打包后占 91%。这里在一张 GPU 上换成掐表（需要 CUDA）：

1. 用主线模型的形状（configs/main/sft.toml，689.5M 参数；BF16 autocast、eager 模式）对这 200 条对话
   做一遍前向 + 反向（loss 只算助手 token；不含优化器那一步，它和怎么排无关），三种排法：
     一条一行，补齐到 513；
     一条一行，按原顺序每 8 条一批，只补齐到这一批里最长的那条（常见的"动态补齐"）；
     首次适配打包进 94 个窗口（04 的 pack_first_fit）。
   对话就是 04 造的那 200 条；字符级 token id 直接当主线词表里的 id 用，只为了让长度和 mask 一样。
2. 第 5.3 节的取舍：zero 用最快的因果注意力 kernel，不隔离打包在一起的对话。把同样的对话按主线 SFT 的
   窗口（8192）首次适配打包，取第一个窗口，单测一层注意力（16 个查询头 / 8 个 K/V 头、head_dim 128）
   的前向 + 反向，四种写法：
     普通因果（SDPA 的 is_causal，走 FlashAttention）——zero 现在的做法，会串门；
     文档 mask 写成一个 8192 × 8192 的布尔矩阵传给 SDPA——最直接的写法；
     FlexAttention + 文档 mask——按 128 × 128 的块，整块被遮住就跳过不算；
     变长（varlen）FlashAttention：传入每条对话的起止位置——5.3 节说的"第二步的选项"。
没有 GPU 可以跳过；正文里贴了一次 RTX 3090 上的结果。
运行：uv run python chapters/16-sft/code/06_gpu_packing.py
"""

from __future__ import annotations

import importlib.util
import random
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))  # 让脚本能 import 仓库根目录下的 zero


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


pk = load("ch16_packing", HERE / "04_packing.py")  # pack_first_fit，以及 lm（02）、ch10
lm, ch10 = pk.lm, pk.ch10
MICRO = 8  # 每个 micro-batch 8 行


def conversations() -> list[tuple[list[int], list[bool]]]:
    """和 04_packing.run() 一模一样的 200 条对话（同一个随机种子）。"""
    tok = lm.ChatTok(ch10.CharData().chars)
    rng = random.Random(3)
    convs = []
    for _ in range(200):
        msgs = lm.make_example(rng)["messages"]
        for _ in range(rng.choice([0, 0, 1, 2])):
            e2 = lm.make_example(rng)
            msgs = msgs[:1] + e2["messages"][1:] + msgs[1:]
        convs.append(lm.encode_with_mask(tok, msgs))
    return convs


def rows_to_batches(rows: list[tuple[list[int], list[bool]]], pad_to: int | None) -> list[tuple]:
    """每行 (ids, mask) → x = ids[:-1]，y = ids[1:]（非助手位置 −100），每 MICRO 行一个 batch。
    pad_to=None：补齐到这一批里最长的一行。"""
    out = []
    for i in range(0, len(rows), MICRO):
        chunk = rows[i : i + MICRO]
        L = (pad_to or max(len(ids) for ids, _ in chunk)) - 1
        x = torch.zeros(len(chunk), L, dtype=torch.long)
        y = torch.full((len(chunk), L), -100, dtype=torch.long)
        for r, (ids, mask) in enumerate(chunk):
            xi, yi = lm.masked_targets(ids, mask)
            x[r, : len(xi)], y[r, : len(yi)] = xi, yi
        out.append((x.cuda(), y.cuda()))
    return out


def time_pass(model, batches, reps: int = 2) -> tuple[float, float]:
    """整份数据前向 + 反向一遍（梯度累积，不更新），重复 reps 次，返回（秒数的中位数，最大 / 最小 − 1）。
    每遍 7–15 秒，两遍之间只差百分之零点几，所以只跑两遍（中位数即平均），总时间控制在 2 分钟内。"""
    times = []
    for _ in range(reps):
        model.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for x, y in batches:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model.loss(x, y)
            loss.backward()
        b.record()
        torch.cuda.synchronize()
        times.append(a.elapsed_time(b) / 1e3)
    return statistics.median(times), max(times) / min(times) - 1


def part1(convs) -> None:
    from zero.config import load_model_config
    from zero.model import Transformer

    lengths = [len(ids) for ids, _ in convs]
    real = sum(lengths)
    window = 513
    bins = pk.pack_first_fit(lengths, window)
    packed_rows = []
    for b in bins:  # 把一个窗口里的几条对话首尾相接；mask 跟着走，窗口边界处的预测自然不算 loss
        ids = [t for i in b for t in convs[i][0]]
        mask = [m for i in b for m in convs[i][1]]
        packed_rows.append((ids, mask))
    layouts = [
        ("一条一行，补齐到 513", rows_to_batches(convs, window)),
        ("一条一行，补齐到批内最长", rows_to_batches(convs, None)),
        (f"首次适配打包（{len(bins)} 个窗口）", rows_to_batches(packed_rows, window)),
    ]
    cfg = load_model_config(ROOT / "configs" / "main" / "sft.toml")
    with torch.device("cuda"):  # 直接在 GPU 上建模型、初始化（CPU 单线程初始化 6.9 亿个参数很慢）
        model = Transformer(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"① 主线模型（{n_params / 1e6:.1f}M 参数）对同样 200 条对话（{real:,} 个真实 token）"
          f"前向 + 反向一遍，每批 {MICRO} 行：")
    time_pass(model, layouts[0][1][:3], reps=1)  # 预热：cuBLAS 选算法、分配显存
    print(f"   {'排法':24s} {'批数':>4s} {'算过的位置':>10s} {'真实占比':>8s} {'耗时':>8s} "
          f"{'真实 token/s':>12s} {'相对第一行':>10s} {'两遍相差':>8s}")
    base = None
    for name, batches in layouts:
        positions = sum(x.numel() for x, _ in batches)
        sec, spread = time_pass(model, batches)
        base = base or sec
        print(f"   {name:24s} {len(batches):4d} {positions:10,d} {100 * real / positions:7.0f}% "
              f"{sec:7.2f}s {real / sec:12,.0f} {base / sec:9.2f}× {100 * spread:7.1f}%")
    del model
    torch.cuda.empty_cache()


# ── 2. 一层注意力：因果 vs 三种文档 mask ────────────────────────────────────────────
def bench(fn, grad_out, reps: int = 10) -> float:
    """前向 + 反向，预热 3 次后重复 reps 次，返回毫秒中位数。"""
    for _ in range(3):
        fn().backward(grad_out)
    torch.cuda.synchronize()
    times = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn().backward(grad_out)
        b.record()
        torch.cuda.synchronize()
        times.append(a.elapsed_time(b))
    return statistics.median(times)


def part2(convs) -> None:
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention
    from torch.nn.attention.varlen import varlen_attn

    T, H, KV, D = 8192, 16, 8, 128
    lengths = [len(ids) for ids, _ in convs]
    first = pk.pack_first_fit(lengths, T)[0]
    docs = [lengths[i] for i in first]
    n_conv = len(docs)
    if sum(docs) < T:
        docs.append(T - sum(docs))  # 窗口剩下的补齐位置单独算一段
    dev = "cuda"
    doc_id = torch.repeat_interleave(torch.arange(len(docs)), torch.tensor(docs)).to(dev)
    useful = sum(n * (n + 1) // 2 for n in docs[:n_conv]) / (T * (T + 1) // 2)
    print(f"\n② 主线 SFT 窗口 {T}：首次适配装进 {n_conv} 条对话（{sum(docs[:n_conv]):,} 个 token），"
          f"单测一层注意力（{H} 个查询头 / {KV} 个 K/V 头，head_dim {D}，BF16）前向 + 反向。")
    print(f"   文档 mask 里要算的 (查询, 键) 对只有因果 mask 的 {100 * useful:.1f}%")

    torch.manual_seed(0)
    q = torch.randn(1, H, T, D, device=dev, dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(1, KV, T, D, device=dev, dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(1, KV, T, D, device=dev, dtype=torch.bfloat16, requires_grad=True)
    go = torch.randn(1, H, T, D, device=dev, dtype=torch.bfloat16)
    allow = (doc_id[:, None] == doc_id[None, :]) & torch.ones(T, T, dtype=torch.bool, device=dev).tril()

    def causal():
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            return F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)

    def boolmask():
        return F.scaled_dot_product_attention(q, k, v, attn_mask=allow, enable_gqa=True)

    def same_doc(b, h, qi, ki):
        return (doc_id[qi] == doc_id[ki]) & (qi >= ki)

    block_mask = create_block_mask(same_doc, None, None, T, T, device=dev)
    flex = torch.compile(flex_attention)

    def flexdoc():
        return flex(q, k, v, block_mask=block_mask, enable_gqa=True)

    cu = torch.tensor([0, *torch.tensor(docs).cumsum(0).tolist()], device=dev, dtype=torch.int32)

    def varlen():  # varlen 的输入是 (总 token 数, 头数, head_dim)，没有 batch 维
        o = varlen_attn(q[0].transpose(0, 1).contiguous(), k[0].transpose(0, 1).contiguous(),
                        v[0].transpose(0, 1).contiguous(), cu, cu, max(docs), max(docs),
                        window_size=(-1, 0))
        return o.transpose(0, 1)[None]

    try:  # PyTorch 给"布尔 mask + GQA"选了哪种实现
        chosen = SDPBackend(torch._fused_sdp_choice(q, k, v, attn_mask=allow, enable_gqa=True)).name
    except Exception:
        chosen = "?"
    rows = [("普通因果（FlashAttention，会串门）", causal),
            ("文档 mask：布尔矩阵传给 SDPA", boolmask),
            ("文档 mask：FlexAttention", flexdoc),
            ("文档 mask：varlen FlashAttention", varlen)]
    ref = None
    with torch.no_grad():
        outs = {}
        for name, fn in rows:
            try:
                outs[name] = fn().float()
            except Exception as e:  # 某种写法在这块卡 / 这个 PyTorch 版本上不可用时，照实打印
                outs[name] = e
        ref = outs[rows[3][0]] if not isinstance(outs[rows[3][0]], Exception) else None
    print(f"   {'写法':34s} {'前向+反向':>10s} {'× 28 层':>9s} {'相对因果':>8s} {'显存峰值':>9s} "
          f"{'与 varlen 输出最大差':>18s}")
    base = None
    for name, fn in rows:
        if isinstance(outs[name], Exception):
            print(f"   {name:34s} 不可用：{type(outs[name]).__name__}: {str(outs[name])[:80]}")
            continue
        q.grad = k.grad = v.grad = None
        torch.cuda.reset_peak_memory_stats()
        try:
            ms = bench(fn, go)
        except torch.OutOfMemoryError:
            print(f"   {name:34s} 显存不够（24 GB 放不下它的反向）")
            continue
        peak = torch.cuda.max_memory_allocated() / 2**30
        base = base or ms
        diff = "—" if ref is None else f"{float((outs[name] - ref).abs().max()):.2e}"
        print(f"   {name:34s} {ms:8.2f}ms {28 * ms:7.0f}ms {base / ms:7.2f}× {peak:7.2f}GiB {diff:>18s}")
    print(f"   （布尔矩阵那一行，PyTorch 选的实现是 {chosen}。普通因果与 varlen 的差就是串门："
          "同一窗口里后面的对话看到了前面的对话；三种文档 mask 之间的差是 BF16 舍入）")


def main() -> None:
    if not torch.cuda.is_available():
        print("本脚本需要 CUDA GPU；没有 GPU 可以跳过，正文里贴了一次 RTX 3090 上的结果。")
        sys.exit(0)
    torch.manual_seed(0)
    print(f"GPU：{torch.cuda.get_device_name(0)}，PyTorch {torch.__version__}，CUDA {torch.version.cuda}\n")
    convs = conversations()
    t0 = time.time()
    part1(convs)
    part2(convs)
    print(f"\n总耗时 {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
