"""第 21 章 · GPU 实测：decode 一步 ≈（权重 + KV cache）÷ 显存带宽

把第 1.2、3 节的账本搬到一张真 GPU 上（需要 CUDA）：
  1. 按主线模型（configs/main）的形状搭一个随机权重的 28 层 decode 步（BF16），KV cache 预先填满
     T 个位置，测"再生成 1 个 token"要多久；理论值用 02_prefill_decode.py 的 decode_step，
     只把硬件换成 RTX 3090 的规格：
         一步 ≥ max(算力 ÷ 峰值, (权重 + B × T × 每 token KV) ÷ 显存带宽)
  2. 同一上下文下换 MHA / GQA / MQA / MLA（吸收路径，512 + 64）：KV cache 实际占多少显存、
     decode 一步多久。

一步 decode 有几百个小 kernel，逐个从 Python 发射的开销（每个几微秒）会盖过读显存的时间，
所以用 CUDA Graph 把整步录下来、一次提交——vLLM 这类推理引擎也是这么做的。
普通注意力用 PyTorch 的 SDPA（FlashAttention kernel）读 KV cache；MLA 没有现成 kernel，
用几次矩阵乘手写吸收路径。省略了 RoPE、QK-Norm 这些逐元素运算：它们几乎不读显存，不影响账本。
没有 GPU 可以跳过；正文里贴了一次 RTX 3090 上的结果。
运行：uv run python chapters/21-kv-cache-ledger/code/05_gpu_decode_ledger.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
PEAK_FLOPS = 71e12  # RTX 3090 稠密 BF16 张量核峰值（规格表）
HBM_BW = 936e9  # RTX 3090 显存带宽，字节/秒（规格表）
BF16 = 2


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def build(c, kind: str, B: int, T: int, dev: str = "cuda") -> dict:
    """随机权重 + 填满 T 个位置的缓存。kind: MHA / GQA / MQA / MLA。"""
    D, H, hd, L = c.dim, c.n_heads, c.head_dim, c.n_layers
    r, dr = 512, 64  # MLA：kv_lora_rank 512 + qk_rope_head_dim 64（DeepSeek-V3 同形）

    def w(*shape):
        return torch.randn(*shape, device=dev, dtype=torch.bfloat16) * 0.02

    m = dict(kind=kind, B=B, T=T, H=H, hd=hd, D=D, layers=[])
    m["lm_head"] = w(c.vocab_size, D)  # 与 embedding 共享；decode 时整份读一遍
    before = torch.cuda.memory_allocated()
    if kind == "MLA":
        m["cache"] = torch.randn(L, B, T, r + dr, device=dev, dtype=torch.bfloat16)  # [c_KV ; k_R]
    else:
        m["n_kv"] = {"MHA": H, "GQA": c.n_kv_heads, "MQA": 1}[kind]
        m["cache"] = torch.randn(L, 2, B, m["n_kv"], T, hd, device=dev, dtype=torch.bfloat16)
    m["cache_alloc"] = torch.cuda.memory_allocated() - before  # 分配器实际给出的字节
    for _ in range(L):
        lay = dict(n1=torch.ones(D, device=dev, dtype=torch.bfloat16),
                   n2=torch.ones(D, device=dev, dtype=torch.bfloat16),
                   w_gu=w(D, 2 * c.ffn_dim), w_down=w(c.ffn_dim, D), wo=w(H * hd, D))
        if kind == "MLA":  # 每个头 q：128（不带位置）+ 64（RoPE）；W_UK、W_UV 各 (H, 128, 512)
            lay.update(wq=w(D, H * (hd + dr)), wkv_a=w(D, r + dr), w_uk=w(H, hd, r),
                       w_uv=w(H, r, hd))
        else:
            lay.update(wqkv=w(D, (H + 2 * m["n_kv"]) * hd))
        m["layers"].append(lay)
    m["weight_bytes"] = sum(t.numel() * BF16 for lay in m["layers"] for k, t in lay.items()
                            if k.startswith("w")) + m["lm_head"].numel() * BF16
    return m


def attend_gqa(m, lay, h, cache_l):
    """普通注意力的 decode：新 K/V 写进最后一个槽，再用 SDPA（FlashAttention）读全部 T 个 K、V。"""
    B, H, hd, n_kv = m["B"], m["H"], m["hd"], m["n_kv"]
    q, k, v = (h @ lay["wqkv"]).split([H * hd, n_kv * hd, n_kv * hd], dim=-1)
    cache_l[0, :, :, -1] = k.view(B, n_kv, hd)
    cache_l[1, :, :, -1] = v.view(B, n_kv, hd)
    o = F.scaled_dot_product_attention(q.view(B, H, 1, hd), cache_l[0], cache_l[1],
                                       enable_gqa=n_kv < H)  # 共用一组 K/V 的查询头不复制 K/V
    return o.reshape(B, H * hd)


def attend_mla(m, lay, h, cache_l, n_split: int = 32):
    """MLA 吸收路径（03_mla.py 同一公式）：query 投进潜空间，直接和缓存的 576 维潜向量点积。"""
    B, H, hd, T = m["B"], m["H"], m["hd"], m["T"]
    q = (h @ lay["wq"]).view(B, H, hd + 64)
    q_nope, q_pe = q.split([hd, 64], dim=-1)
    cache_l[:, -1] = h @ lay["wkv_a"]  # 新位置的 [c_KV ; k_R] 写进最后一个槽
    q_lat = (q_nope.transpose(0, 1) @ lay["w_uk"]).transpose(0, 1)  # (B, H, 512)：W_UK 吸收进 query
    q_full = torch.cat([q_lat, q_pe], dim=-1)  # (B, H, 576)
    s = (q_full @ cache_l.transpose(1, 2)).float() / (hd + 64) ** 0.5  # (B, H, T)：所有头读同一份缓存
    p = s.softmax(-1).to(torch.bfloat16)
    # 在潜空间里加权平均。沿 T 切成 n_split 段分头算再相加（flash-decoding 的 split-KV）：
    # 否则一个长度 T 的求和只能摊给很少几个 SM，读显存快不起来
    p = p.view(B, H, n_split, T // n_split).transpose(1, 2)  # (B, n_split, H, T/n_split)
    c = cache_l[..., :512].view(B, n_split, T // n_split, 512)
    o_lat = (p @ c).sum(1)  # (B, H, 512)
    o = (o_lat.transpose(0, 1) @ lay["w_uv"]).transpose(0, 1)  # (B, H, 128)：再用 W_UV 投回去
    return o.reshape(B, H * hd)


def decode_step(m, x):
    D = m["D"]
    attend = attend_mla if m["kind"] == "MLA" else attend_gqa
    for li, lay in enumerate(m["layers"]):
        h = F.rms_norm(x, (D,), lay["n1"])
        x = x + attend(m, lay, h, m["cache"][li]) @ lay["wo"]
        h = F.rms_norm(x, (D,), lay["n2"])
        gate, up = (h @ lay["w_gu"]).chunk(2, dim=-1)
        x = x + (F.silu(gate) * up) @ lay["w_down"]
    return x @ m["lm_head"].T  # (B, vocab)


def cuda_ms(run, reps: int = 30) -> float:
    """用 CUDA event 计时，返回中位数（毫秒）。"""
    for _ in range(3):  # 预热
        run()
    times = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        run()
        b.record()
        torch.cuda.synchronize()
        times.append(a.elapsed_time(b))
    return sorted(times)[reps // 2]


def time_step(m, graph: bool = True) -> float:
    """decode 一步的中位数耗时（毫秒）。"""
    x = torch.randn(m["B"], m["D"], device="cuda", dtype=torch.bfloat16)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):  # 预热（CUDA Graph 要求先在旁路流上跑几次）
        for _ in range(3):
            decode_step(m, x)
    torch.cuda.current_stream().wait_stream(side)
    if not graph:
        return cuda_ms(lambda: decode_step(m, x))
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        decode_step(m, x)
    return cuda_ms(g.replay)


def free(m) -> None:
    m.clear()
    torch.cuda.empty_cache()


def main() -> None:
    if not torch.cuda.is_available():
        print("本脚本需要 CUDA GPU；没有 GPU 可以跳过，正文里贴了一次 RTX 3090 上的结果。")
        sys.exit(0)
    torch.manual_seed(0)
    pd = _load("prefill_decode", HERE / "02_prefill_decode.py")
    pd.PEAK_FLOPS, pd.HBM_BW = PEAK_FLOPS, HBM_BW  # 账本公式不变，只换硬件规格
    c, n_params, n_matmul, kv_tok = pd.model_numbers()
    print(f"GPU：{torch.cuda.get_device_name(0)}，PyTorch {torch.__version__}，CUDA {torch.version.cuda}")
    print(f"硬件规格：BF16 峰值 {PEAK_FLOPS / 1e12:.0f} TFLOPS，显存带宽 {HBM_BW / 1e9:.0f} GB/s"
          f"（脊点 {PEAK_FLOPS / HBM_BW:.0f} 次/字节）")

    buf = torch.empty(2**29, device="cuda", dtype=torch.bfloat16)  # 1 GiB
    read_bw = buf.nbytes / (cuda_ms(buf.sum, 10) / 1e3)
    print(f"对照：同一张卡上把 1 GiB 连续读一遍（求和）的实测带宽 {read_bw / 1e9:.0f} GB/s")
    del buf

    print("\n1. 主线模型（GQA 16/8，28 层，BF16）decode 一步：实测 vs 账本（CUDA Graph，30 次取中位数）")
    print("        T     B    读权重    读 KV cache    理论下限    实测     实测/理论   实测带宽   吞吐 token/s")
    rows = [(1024, 1), (4096, 1), (4096, 16), (32768, 1), (32768, 4), (131072, 1)]
    got = {}
    for T, B in rows:
        m = build(c, "GQA", B, T)
        d = pd.decode_step(c, n_params, n_matmul, kv_tok, T, B)
        ms = time_step(m)
        got[T, B] = (ms, m["cache"].nbytes)
        bw = (m["weight_bytes"] + m["cache"].nbytes) / (ms / 1e3)
        print(f"   {T:7,d} {B:4d}   {d['weights'] / 2**30:5.2f} GiB  {d['kv'] / 2**30:6.2f} GiB"
              f"   {d['t'] * 1e3:6.2f} ms  {ms:6.2f} ms   {ms / (d['t'] * 1e3):5.2f}×"
              f"   {bw / 1e9:5.0f} GB/s  {B / ms * 1e3:9,.0f}")
        if (T, B) == (1024, 1):
            eager = time_step(m, graph=False)
            print(f"           （同一步不用 CUDA Graph、逐个发射 kernel：{eager:.2f} ms）")
        free(m)
    (t0, b0), (t1, b1) = got[1024, 1], got[131072, 1]
    print(f"   batch 1 从 T = 1,024 到 131,072：KV cache 多读 {(b1 - b0) / 2**30:.2f} GiB，"
          f"一步多花 {t1 - t0:.2f} ms，折合 {(b1 - b0) / ((t1 - t0) / 1e3) / 1e9:.0f} GB/s")

    T, B = 32768, 1
    print(f"\n2. 同一个上下文 T = {T:,}、batch {B}：换注意力后的 KV cache 与 decode 一步")
    print("   方案    每层每位置    账本 KV     实际分配      权重      理论下限    实测     实测带宽")
    for kind in ("MHA", "GQA", "MQA", "MLA"):
        m = build(c, kind, B, T)
        per = 576 if kind == "MLA" else 2 * m["n_kv"] * c.head_dim
        ledger = per * c.n_layers * T * B * BF16
        # 实际要读的字节：普通注意力 K、V 各读一遍；MLA 算分数读 576 维、加权平均再读前 512 维
        read_kv = ledger if kind != "MLA" else (576 + 512) * c.n_layers * T * B * BF16
        t_theory = (m["weight_bytes"] + read_kv) / HBM_BW * 1e3
        ms = time_step(m)
        print(f"   {kind:4}  {per:6,d} 个数  {ledger / 2**30:6.2f} GiB  {m['cache_alloc'] / 2**30:6.2f} GiB"
              f"  {m['weight_bytes'] / 2**30:5.2f} GiB  {t_theory:6.2f} ms  {ms:6.2f} ms"
              f"  {(m['weight_bytes'] + read_kv) / (ms / 1e3) / 1e9:5.0f} GB/s")
        free(m)
    print("   （MLA 的理论下限按实际读的字节算：分数读 576 维潜向量，加权平均再读一遍前 512 维；"
          "FlashMLA 这类融合 kernel 只读一遍）")


if __name__ == "__main__":
    main()
