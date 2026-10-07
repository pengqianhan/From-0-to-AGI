"""Chapter 14 · Minimal code 3: online softmax. Scan only once, and correct the result during the scan.

The standard "safe softmax" scans three times: ① find the maximum m; ② calculate the sum
l = Σ exp(x_i − m); ③ output exp(x_i − m) / l.
If the data arrives block by block (as in FlashAttention), we do not want to wait for all blocks to know m.

Online softmax (Milakov & Gimelshein, 2018) scans only once. It keeps two values "up to now":
    m_j = max(m_{j−1}, x_j)                                   — the current maximum
    l_j = l_{j−1} · exp(m_{j−1} − m_j) + exp(x_j − m_j)         — the sum of exponentials, relative to the current maximum
When a new maximum occurs, the old sum still uses the old m. Multiply it by exp(m_old − m_new) ≤ 1
to "correct" it to the new reference. After the scan, softmax_i = exp(x_i − m_N) / l_N.
The result is the same as the result of the three scans.

One more step: the attention output is Σ softmax_i · v_i. We can also correct it during the scan:
    o_j = o_{j−1} · exp(m_{j−1} − m_j) + exp(x_j − m_j) · v_j,   and the final output is o_N / l_N
This is the core of FlashAttention. The next script does it block by block (in tiles).

Run: uv run python chapters/14-pretraining-engineering/code/03_online_softmax.py   (1 s)
"""

import math

import torch

torch.manual_seed(0)
EXAMPLE = [1.0, 3.0, 2.0, 5.0, 4.0]  # one row of scores for the step-by-step demo in the video


def softmax_three_pass(x: list[float]) -> list[float]:
    m = max(x)                                   # scan 1: the maximum
    l = sum(math.exp(xi - m) for xi in x)        # scan 2: the sum of exponentials
    return [math.exp(xi - m) / l for xi in x]    # scan 3: normalize


def online_softmax_stats(x: list[float]) -> tuple[list[tuple[float, float]], float, float]:
    """Scan only once. Return (m_j, l_j) for each step, and the final m and l."""
    m, l, trace = -math.inf, 0.0, []
    for xi in x:
        m_new = max(m, xi)
        l = l * math.exp(m - m_new) + math.exp(xi - m_new)   # correct the old sum to the new maximum
        m = m_new
        trace.append((m, l))
    return trace, m, l


def online_attention_row(scores: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """One row of scores (n,) and v (n, d): one scan gives Σ softmax_i · v_i directly."""
    m, l = -math.inf, 0.0
    o = torch.zeros(v.shape[1], dtype=v.dtype)
    for s, vj in zip(scores.tolist(), v):
        m_new = max(m, s)
        scale = math.exp(m - m_new)
        p = math.exp(s - m_new)
        l = l * scale + p
        o = o * scale + p * vj                   # also correct the output to the new maximum
        m = m_new
    return o / l


def main():
    print("① State during one scan (x = " + str(EXAMPLE) + ")")
    trace, m, l = online_softmax_stats(EXAMPLE)
    print(f"  {'j':>2} {'x_j':>5} {'m_j':>5} {'l_j':>9}  note")
    prev_m = -math.inf
    for j, (xj, (mj, lj)) in enumerate(zip(EXAMPLE, trace), 1):
        note = f"new maximum: multiply the old sum by exp({prev_m:g} − {mj:g}) = {math.exp(prev_m - mj):.4f}" \
            if mj > prev_m and j > 1 else ""
        print(f"  {j:>2} {xj:>5g} {mj:>5g} {lj:>9.4f}  {note}")
        prev_m = mj
    online = [math.exp(xi - m) / l for xi in EXAMPLE]
    ref = softmax_three_pass(EXAMPLE)
    print("  three-scan softmax: " + ", ".join(f"{p:.4f}" for p in ref))
    print("  one-scan softmax:   " + ", ".join(f"{p:.4f}" for p in online))
    print(f"  max difference: {max(abs(a - b) for a, b in zip(ref, online)):.1e}")

    print("\n② A long random vector (float64)")
    x = (torch.randn(10_000, dtype=torch.float64) * 10).tolist()
    _, m, l = online_softmax_stats(x)
    diff = max(abs(math.exp(xi - m) / l - r) for xi, r in zip(x, softmax_three_pass(x)))
    print(f"  n = 10,000, scores with std 10: max difference between one scan and three scans {diff:.1e}")

    print("\n③ One scan gives the attention output Σ softmax_i · v_i")
    scores = torch.randn(512, dtype=torch.float64) * 3
    v = torch.randn(512, 64, dtype=torch.float64)
    ref_o = torch.softmax(scores, 0) @ v
    diff_o = (online_attention_row(scores, v) - ref_o).abs().max().item()
    print(f"  512 positions, d = 64: max difference from softmax(scores) @ V {diff_o:.1e}")


if __name__ == "__main__":
    main()
