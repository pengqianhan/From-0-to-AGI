"""第 14 章 · 极简代码 2：FP32、BF16、FP16 —— 少了的那些位去哪了

一个浮点数 = 符号位 + 指数位 + 尾数位：
    FP32：1 + 8 + 23     BF16：1 + 8 + 7      FP16：1 + 5 + 10
指数位决定"能表示多大、多小"（范围），尾数位决定"相邻两个数隔多远"（精度）。
BF16 保留了 FP32 的 8 位指数（范围一样），只砍尾数；FP16 两头都砍，范围小得多。

这个脚本用真实数字看：
  ① 三种格式的位布局、最大值、最小正规数、机器 epsilon；
  ② 把 0.01 连加一万次（正确答案 100）：低精度累加器会"卡住"；
  ③ 权重更新 w ← w − η·g：BF16 的权重吃不下小步长 → 所以要 FP32 主权重；
  ④ 随机舍入（stochastic rounding）：BF16 累加器平均意义上不再卡住；
  ⑤ FP16 的范围问题：大数溢出成 inf、小梯度下溢成 0（所以 FP16 训练要 loss scaling，BF16 不用）；
  ⑥ PyTorch 的 autocast：矩阵乘在 BF16 里算，参数、梯度仍是 FP32。

运行：uv run python chapters/14-pretraining-engineering/code/02_precision.py   （几秒）
"""

import struct

import torch

torch.manual_seed(0)
FORMATS = {"fp32": (torch.float32, 8, 23), "bf16": (torch.bfloat16, 8, 7), "fp16": (torch.float16, 5, 10)}


def bits(x: float, dtype: torch.dtype) -> str:
    """把 x 转成 dtype，再按"符号 | 指数 | 尾数"打印它的二进制位。"""
    t = torch.tensor(x, dtype=dtype)
    if dtype == torch.float32:
        raw = struct.unpack(">I", struct.pack(">f", float(t)))[0]
        n, e = 32, 8
    else:
        raw = int(t.view(torch.int16).item()) & 0xFFFF
        n, e = 16, 8 if dtype == torch.bfloat16 else 5
    b = format(raw, f"0{n}b")
    return f"{b[0]} | {b[1:1 + e]} | {b[1 + e:]}"


def repeat_add(value: float, n: int, dtype: torch.dtype) -> float:
    acc = torch.zeros((), dtype=dtype)
    v = torch.tensor(value, dtype=dtype)
    for _ in range(n):
        acc = acc + v  # 每一步的结果都舍入回 dtype
    return float(acc)


def to_bf16_stochastic(x: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """随机舍入到 BF16：在 FP32 的低 16 位上加一个 [0, 2^16) 的随机数，再截断。
    离上面那个 BF16 越近，进位的概率越大 —— 期望值恰好等于原数。"""
    raw = x.float().view(torch.int32)
    noise = torch.randint(0, 1 << 16, raw.shape, generator=gen, dtype=torch.int32)
    return ((raw + noise) & ~0xFFFF).view(torch.float32).to(torch.bfloat16)


def main():
    torch.set_num_threads(1)
    print("① 位布局（符号 | 指数 | 尾数）与范围")
    x = 3.14159
    print(f"  {'':5} {'π 的二进制位':<42} {'存下来的值':>12} {'最大值':>10} {'最小正规数':>11} {'epsilon':>9}")
    for name, (dt, _, _) in FORMATS.items():
        fi = torch.finfo(dt)
        stored = float(torch.tensor(x, dtype=dt))
        print(f"  {name:5} {bits(x, dt):<42} {stored:>12.7f} {fi.max:>10.3g} {fi.tiny:>11.3g} {fi.eps:>9.3g}")
    print("  epsilon = 1 和下一个可表示数的距离 = 2^-尾数位数：FP32 2^-23，BF16 2^-7，FP16 2^-10")

    print("\n② 把 0.01 连加 10,000 次（正确答案 100）")
    for name, (dt, _, _) in FORMATS.items():
        print(f"  {name} 累加器：{repeat_add(0.01, 10_000, dt):.4f}")
    print("  BF16 只有 7 位尾数：在 [4, 8) 里相邻两数相距 4 × 2^-7 = 0.03125，")
    print("  0.01 不到间隔的一半，每次加完又被舍回原值 —— 累加器卡在 4；FP16 同理卡在 32")

    print("\n③ 权重更新：w = 1.0，每步 w ← w − η·g，η·g = 1e-3，走 1000 步（正确答案 0.0）")
    for name, dt in [("bf16 权重", torch.bfloat16), ("fp32 权重", torch.float32)]:
        w = torch.tensor(1.0, dtype=dt)
        for _ in range(1000):
            w = w - torch.tensor(1e-3, dtype=dt)
        print(f"  {name}：{float(w):.4f}")
    print("  → 优化器里要保留一份 FP32 的主权重（master weights），BF16 只用于前向/反向的计算")

    print("\n④ 随机舍入：BF16 累加器 + stochastic rounding（0.01 × 10,000，5 个种子）")
    results = []
    for seed in range(5):
        gen = torch.Generator().manual_seed(seed)
        acc = torch.zeros((), dtype=torch.bfloat16)
        for _ in range(10_000):
            acc = to_bf16_stochastic(acc.float() + 0.01, gen)
        results.append(float(acc))
    print("  " + "  ".join(f"{r:.2f}" for r in results) + f"   平均 {sum(results) / 5:.2f}")

    print("\n⑤ FP16 的范围问题（BF16 对照）")
    for v in (70000.0, 1e-8):
        print(f"  {v:>8g} → fp16: {float(torch.tensor(v, dtype=torch.float16)):<10g}"
              f" bf16: {float(torch.tensor(v, dtype=torch.bfloat16)):g}")
    grads = torch.randn(100_000) * 1e-6  # 一批很小的梯度
    for name, dt in [("fp16", torch.float16), ("bf16", torch.bfloat16)]:
        zero_frac = (grads.to(dt) == 0).float().mean().item()
        print(f"  std 1e-6 的梯度转成 {name}：{zero_frac:.1%} 变成 0")
    scaled = (grads * 1024).to(torch.float16)
    print(f"  FP16 先乘 1024（loss scaling）再转：{(scaled == 0).float().mean().item():.1%} 变成 0")

    print("\n⑥ autocast：矩阵乘用 BF16 算，参数和梯度保持 FP32")
    lin = torch.nn.Linear(1024, 1024, bias=False)
    xin = torch.randn(64, 1024)
    ref = lin(xin)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = lin(xin)
    out.float().pow(2).mean().backward()
    rel = ((out.float() - ref).norm() / ref.norm()).item()
    print(f"  输出 dtype：{out.dtype}；参数 dtype：{lin.weight.dtype}；梯度 dtype：{lin.weight.grad.dtype}")
    print(f"  与 FP32 结果的相对误差：{rel:.2e}（BF16 epsilon 的量级 {torch.finfo(torch.bfloat16).eps:.1e}）")


if __name__ == "__main__":
    main()
