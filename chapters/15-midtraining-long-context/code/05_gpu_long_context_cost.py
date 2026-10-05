"""第 15 章 · 极简代码 5（GPU 实测）：32K 序列每个 token 到底贵几倍 —— 主线模型一层 Block 的实测

正文第 4 节按 zero 的口径（PaLM 附录 B，不为因果掩码减半）算：主线模型每个训练 token
4K 时 6.96 GFLOP、32K 时 26.69 GFLOP，贵 3.84 倍；FlashAttention 会跳过因果掩码的上三角，
实际的注意力运算约省一半。到底贵几倍，在 GPU 上直接量：

  取长上下文配置（configs/main/longctx.toml）里的一个 Block —— zero 的实现：RMSNorm、GQA（16 个查询头、
  8 个 KV 头 × 128）、QK-Norm、RoPE（基频 100 万）、SwiGLU（FFN 3584），BF16 autocast，SDPA 注意力。
  每次都喂同样多的 token（T × micro batch = 32,768），T 从 4096 加到 32768：
  ① 前向 + 反向的每 token 耗时，相对 T = 4096 贵几倍；和两种公式（不减半 / 因果减半）对照；
     再按两种口径换算算力利用率：真正做了的运算（因果减半），和 zero 日志里 MFU 用的不减半公式；
  ② 这一层为反向保存了多少显存，每 token 多少字节（FlashAttention 不存 T×T 的矩阵，应当和 T 无关）。

只量一层：整个模型 28 层叠起来就是 28 倍，另加一个和 T 无关的输出层。输入是随机向量。

运行：uv run python chapters/15-midtraining-long-context/code/05_gpu_long_context_cost.py   （需要 CUDA GPU，RTX 3090 上约 20 秒）
"""

import statistics
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]   # 仓库根目录，用来 import zero
sys.path.insert(0, str(ROOT))

from zero.config import load_model_config  # noqa: E402
from zero.model import Block, RotaryEmbedding  # noqa: E402
from zero.tools.memory_calc import layer_activation_bytes_per_token, matmul_params  # noqa: E402

TOKENS = 32768            # 每次前向 + 反向喂的 token 数（T × micro batch）
SEQ_LENS = [4096, 8192, 16384, 32768]
SPEC_BF16 = {"3090": 71.0}  # RTX 3090 规格表的 BF16 Tensor Core 稠密峰值（TFLOPS，FP32 累加）


def main():
    if not torch.cuda.is_available():
        print("本脚本需要 CUDA GPU；没有 GPU 可以跳过，正文里贴了一次 RTX 3090 上的结果。")
        sys.exit(0)
    name = torch.cuda.get_device_name()
    peak = next((v for k, v in SPEC_BF16.items() if k in name), None)
    print(f"GPU：{name}；PyTorch {torch.__version__}，CUDA {torch.version.cuda}")

    cfg = load_model_config(ROOT / "configs/main/longctx.toml")
    torch.manual_seed(0)
    with torch.device("cuda"):
        block = Block(cfg, layer_idx=0)
        rope = RotaryEmbedding(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta, cfg.rope_scaling)
    per_layer, _ = matmul_params(cfg)
    print(f"一层 Block：dim {cfg.dim}，{cfg.n_heads} 个查询头 / {cfg.n_kv_heads} 个 KV 头 × {cfg.head_dim}，"
          f"FFN {cfg.ffn_dim}，RoPE 基频 {cfg.rope_theta:g}；每次 {TOKENS:,} 个 token，BF16 autocast")

    def flops(T, causal_half):
        """一层每 token 的训练运算量：6·N_层 + 12·q_dim·T（因果减半时注意力项取一半）。"""
        attn = 12 * cfg.q_dim * T
        return 6 * per_layer + (attn / 2 if causal_half else attn)

    def measure(T):
        """返回 (每次前向 + 反向的秒数, 前向结束时多占的显存字节)。"""
        mb = TOKENS // T
        cos, sin = rope(0, T)
        x = torch.randn(mb, T, cfg.dim, device="cuda", requires_grad=True)
        grad = torch.randn(mb, T, cfg.dim, device="cuda")

        def step():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                y = block(x, cos, sin)
            y.backward(grad)
            x.grad = None
            block.zero_grad(set_to_none=True)

        for _ in range(3):                        # 预热
            step()
        torch.cuda.synchronize()
        base = torch.cuda.memory_allocated()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            y = block(x, cos, sin)
        torch.cuda.synchronize()
        saved = torch.cuda.memory_allocated() - base   # 含这一层的输出（也就是下一层的输入）
        del y
        times = []
        for _ in range(10):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            step()
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end) / 1e3)
        return statistics.median(times), saved

    rows = [(T, TOKENS // T, *measure(T)) for T in SEQ_LENS]

    def pct(tflops):
        return f"{tflops / peak:.0%}" if peak else "—"

    t0 = rows[0][2]
    print("\n① 前向 + 反向，每次 32,768 个 token（10 次取中位数）")
    print(f"  {'T':>6} {'micro batch':>11} {'耗时':>8} {'每 token':>8} {'实测倍数':>7} "
          f"{'公式·不减半':>9} {'公式·因果减半':>10} {'实际 TFLOPS':>11} {'占峰值':>6} {'按 zero 口径的 MFU':>16}")
    for T, mb, sec, _ in rows:
        real = TOKENS * flops(T, True) / sec / 1e12       # 按真正做了的运算（因果减半）
        booked = TOKENS * flops(T, False) / sec / 1e12    # 按 zero / PaLM 的记账口径（不减半）
        print(f"  {T:>6} {mb:>11} {sec * 1e3:>6.1f}ms {sec / TOKENS * 1e9:>6.0f}ns {sec / t0:>7.2f}× "
              f"{flops(T, False) / flops(SEQ_LENS[0], False):>10.2f}× "
              f"{flops(T, True) / flops(SEQ_LENS[0], True):>12.2f}× {real:>11.1f} {pct(real):>6} {pct(booked):>16}")
    print("  （实际 TFLOPS 按因果减半后真正做的运算量算；zero 日志里的 MFU 用不减半的公式，T 越长越虚高）")

    print("\n② 这一层前向结束时多占的显存（为反向保存的激活 + BF16 权重副本 + 输出）")
    formula = layer_activation_bytes_per_token(cfg, "bf16")
    copies = 2 * per_layer                        # autocast 为反向保存的 BF16 权重副本，和 token 数无关
    print(f"  memory_calc 的公式：每 token {formula:,} 字节 × {TOKENS:,} + 权重副本 {copies / 2**20:.0f} MiB"
          f" = {(formula * TOKENS + copies) / 2**20:,.0f} MiB")
    for T, _, _, saved in rows:
        print(f"  T = {T:>6}：实测 {saved / 2**20:>6,.0f} MiB，扣掉权重副本后每 token "
              f"{(saved - copies) / TOKENS:>8,.0f} 字节")


if __name__ == "__main__":
    main()
