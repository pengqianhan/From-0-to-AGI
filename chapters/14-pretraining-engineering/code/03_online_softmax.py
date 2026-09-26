"""第 14 章 · 极简代码 3：online softmax —— 只扫一遍，边走边改正

标准的"安全 softmax"要扫三遍：① 找最大值 m；② 求和 l = Σ exp(x_i − m)；③ 输出 exp(x_i − m) / l。
如果数据一块一块地流过来（FlashAttention 就是这样），我们不想等全部看完才知道 m。

online softmax（Milakov & Gimelshein, 2018）只扫一遍，维护两个"到目前为止"的量：
    m_j = max(m_{j−1}, x_j)                                   —— 当前最大值
    l_j = l_{j−1} · exp(m_{j−1} − m_j) + exp(x_j − m_j)         —— 以当前最大值为基准的指数和
新的最大值出现时，旧的和是按旧的 m 算的，乘一个 exp(m_旧 − m_新) ≤ 1 就"改正"回新基准。
扫完以后 softmax_i = exp(x_i − m_N) / l_N，和三遍的结果完全一样。

再往前一步：注意力的输出是 Σ softmax_i · v_i，也可以边走边改正：
    o_j = o_{j−1} · exp(m_{j−1} − m_j) + exp(x_j − m_j) · v_j，   最后输出 o_N / l_N
这就是 FlashAttention 的核心，下一个脚本把它按块（tile）来做。

运行：uv run python chapters/14-pretraining-engineering/code/03_online_softmax.py   （1 秒）
"""

import math

import torch

torch.manual_seed(0)
EXAMPLE = [1.0, 3.0, 2.0, 5.0, 4.0]  # 视频里逐步演示用的一行分数


def softmax_three_pass(x: list[float]) -> list[float]:
    m = max(x)                                   # 第 1 遍：最大值
    l = sum(math.exp(xi - m) for xi in x)        # 第 2 遍：指数和
    return [math.exp(xi - m) / l for xi in x]    # 第 3 遍：归一化


def online_softmax_stats(x: list[float]) -> tuple[list[tuple[float, float]], float, float]:
    """只扫一遍，返回每一步的 (m_j, l_j)，以及最终的 m、l。"""
    m, l, trace = -math.inf, 0.0, []
    for xi in x:
        m_new = max(m, xi)
        l = l * math.exp(m - m_new) + math.exp(xi - m_new)   # 旧的和按新最大值改正
        m = m_new
        trace.append((m, l))
    return trace, m, l


def online_attention_row(scores: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """一行分数 scores (n,) 与 v (n, d)：一遍扫完直接得到 Σ softmax_i · v_i。"""
    m, l = -math.inf, 0.0
    o = torch.zeros(v.shape[1], dtype=v.dtype)
    for s, vj in zip(scores.tolist(), v):
        m_new = max(m, s)
        scale = math.exp(m - m_new)
        p = math.exp(s - m_new)
        l = l * scale + p
        o = o * scale + p * vj                   # 输出也按新最大值改正
        m = m_new
    return o / l


def main():
    print("① 一遍扫描的中间状态（x = " + str(EXAMPLE) + "）")
    trace, m, l = online_softmax_stats(EXAMPLE)
    print(f"  {'j':>2} {'x_j':>5} {'m_j':>5} {'l_j':>9}  说明")
    prev_m = -math.inf
    for j, (xj, (mj, lj)) in enumerate(zip(EXAMPLE, trace), 1):
        note = f"新最大值：旧的和乘 exp({prev_m:g} − {mj:g}) = {math.exp(prev_m - mj):.4f}" \
            if mj > prev_m and j > 1 else ""
        print(f"  {j:>2} {xj:>5g} {mj:>5g} {lj:>9.4f}  {note}")
        prev_m = mj
    online = [math.exp(xi - m) / l for xi in EXAMPLE]
    ref = softmax_three_pass(EXAMPLE)
    print("  三遍 softmax：" + ", ".join(f"{p:.4f}" for p in ref))
    print("  一遍 softmax：" + ", ".join(f"{p:.4f}" for p in online))
    print(f"  最大差：{max(abs(a - b) for a, b in zip(ref, online)):.1e}")

    print("\n② 随机的长向量（float64）")
    x = (torch.randn(10_000, dtype=torch.float64) * 10).tolist()
    _, m, l = online_softmax_stats(x)
    diff = max(abs(math.exp(xi - m) / l - r) for xi, r in zip(x, softmax_three_pass(x)))
    print(f"  n = 10,000，分数 std 10：一遍与三遍的最大差 {diff:.1e}")

    print("\n③ 一遍算出注意力输出 Σ softmax_i · v_i")
    scores = torch.randn(512, dtype=torch.float64) * 3
    v = torch.randn(512, 64, dtype=torch.float64)
    ref_o = torch.softmax(scores, 0) @ v
    diff_o = (online_attention_row(scores, v) - ref_o).abs().max().item()
    print(f"  512 个位置、d = 64：与 softmax(scores) @ V 的最大差 {diff_o:.1e}")


if __name__ == "__main__":
    main()
