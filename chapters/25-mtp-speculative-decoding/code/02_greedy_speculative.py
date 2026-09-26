"""第 25 章 · 极简代码 2：贪心推测解码——输出和目标模型逐字相同，但更快

每一轮：
  1. 草稿模型自回归地猜 k 个 token（便宜：它很小）；
  2. 目标模型把"最后一个已确定的 token + k 个草稿"一次喂进去（一次前向，k+1 个位置并行打分）；
  3. 从左往右比：草稿 d_i 等于目标在这个位置的 argmax 就接受，遇到第一个不等的就停下，
     改用目标自己的 argmax（纠正）；k 个全对时，目标在最后一个位置还白送一个 token（奖励）。
  4. 被拒绝的草稿 token 已经写进了两个模型的 KV cache，要回滚（truncate）。
每一轮至少产出 1 个 token（纠正或奖励），最多 k+1 个；目标模型每一轮只跑一次。

运行：uv run python chapters/25-mtp-speculative-decoding/code/02_greedy_speculative.py
"""

from __future__ import annotations

import importlib.util
import statistics
import sys
import time
from pathlib import Path

import torch

torch.set_num_threads(1)
HERE = Path(__file__).resolve().parent


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


m1 = _load("ch25_models", "01_models_and_cost.py")
KVCache, truncate = m1.KVCache, m1.truncate


@torch.no_grad()
def greedy_generate(model, prompt: list[int], n_new: int) -> list[int]:
    """基线：目标模型带 KV cache 的普通贪心解码，每步一次前向、一个 token。"""
    cache = KVCache(model.c.n_layers)
    logits = model(torch.tensor([prompt]), cache)[0, -1]
    out = []
    for _ in range(n_new):
        nxt = int(logits.argmax())
        out.append(nxt)
        logits = model(torch.tensor([[nxt]]), cache)[0, -1]
    return out


@torch.no_grad()
def speculative_greedy(target, draft, prompt: list[int], n_new: int, k: int, trace=None):
    """贪心推测解码。返回 (新 token, 统计)。trace 是列表时，逐轮记下 (草稿, 目标的答案, 接受数)。"""
    seq = list(prompt)
    tc, dc = KVCache(target.c.n_layers), KVCache(draft.c.n_layers)
    stats = dict(rounds=0, proposed=0, accepted=0, examined=0)
    while len(seq) - len(prompt) < n_new:
        # ── 1. 草稿：先把缓存里还没有的 token 补进去，再一个一个猜 k 个 ──
        logits = draft(torch.tensor([seq[len(dc) :]]), dc)[0, -1]
        drafts = []
        for i in range(k):
            drafts.append(int(logits.argmax()))
            if i < k - 1:
                logits = draft(torch.tensor([[drafts[-1]]]), dc)[0, -1]
        # ── 2. 目标：一次前向验证全部 k 个草稿（外加缓存里还没有的已确定 token）──
        feed = seq[len(tc) :] + drafts
        p_logits = target(torch.tensor([feed]), tc)[0, -(k + 1) :]  # 最后 k+1 个位置
        choice = p_logits.argmax(-1).tolist()  # choice[i] = 目标在"第 i 个草稿"位置的答案
        # ── 3. 从左往右接受，遇到第一个不一致就停 ──
        m = 0
        while m < k and drafts[m] == choice[m]:
            m += 1
        seq += drafts[:m] + [choice[m]]  # m 个草稿 + 1 个纠正（m == k 时是奖励 token）
        if trace is not None:
            trace.append((drafts, choice, m))
        stats["rounds"] += 1
        stats["proposed"] += k
        stats["accepted"] += m
        stats["examined"] += m + (1 if m < k else 0)  # 被比较过的草稿个数
        # ── 4. 回滚：缓存里只留下"已确定且不是最后一个"的位置 ──
        truncate(tc, len(seq) - 1)
        truncate(dc, min(len(dc), len(seq) - 1))
    return seq[len(prompt) :][:n_new], stats


def prompts(n: int = 4, length: int = 40) -> list[list[int]]:
    """从验证集里截几段做提示词。"""
    data = m1.ch10.CharData()
    val = data.val.tolist()
    step = len(val) // (n + 1)
    return [val[(i + 1) * step : (i + 1) * step + length] for i in range(n)]


def timed(fn, *args) -> tuple[float, object]:
    """返回 (本进程 CPU 时间, 结果)。共享机器上墙钟时间主要反映排队，见 01 的说明。"""
    t0 = time.process_time()
    out = fn(*args)
    return time.process_time() - t0, out


def expected_tokens(alpha: float, k: int) -> float:
    """Leviathan 等人的公式 (1)：每轮期望产出 (1 − α^{k+1}) / (1 − α) 个 token。"""
    return (1 - alpha ** (k + 1)) / (1 - alpha)


if __name__ == "__main__":
    target, draft = m1.load_target(), m1.load_draft()
    data = m1.ch10.CharData()
    P = prompts()
    N = 200

    # ── 正确性：贪心推测解码的输出必须和目标模型自己贪心解码逐字相同 ──
    base_out = [greedy_generate(target, p, N) for p in P]
    for k in (1, 3, 5, 8):
        same = all(speculative_greedy(target, draft, p, N, k)[0] == b for p, b in zip(P, base_out))
        print(f"k={k}：{len(P)} 段提示词 × {N} 个 token，与目标模型贪心解码逐字相同：{same}")
    print("\n示例（提示词 + 生成）：\n" + data.decode(P[0]) + "|" + data.decode(base_out[0][:120]))

    # ── 接受率与速度：交替重复测 3 轮，取中位数，减轻 CPU 负载波动 ──
    ks = (1, 2, 3, 4, 5, 6, 8)
    t_base, t_spec, st = [], {k: [] for k in ks}, {}
    for _ in range(3):
        t_base.append(sum(timed(greedy_generate, target, p, N)[0] for p in P))
        for k in ks:
            tot, agg = 0.0, dict(rounds=0, proposed=0, accepted=0, examined=0)
            for p in P:
                dt, (_, s) = timed(speculative_greedy, target, draft, p, N, k)
                tot += dt
                for key in agg:
                    agg[key] += s[key]
            t_spec[k].append(tot)
            st[k] = agg
    tb = statistics.median(t_base)
    c = m1.forward_time(draft, 200, 1) / m1.forward_time(target, 200, 1)
    print(
        f"\n目标模型普通贪心解码：{len(P) * N} 个 token 用时 {tb:.2f} s（中位数）；成本系数 c ≈ {c:.2f}"
    )
    print(
        f"{'k':>2} {'逐token接受率α':>13} {'每轮产出(实测)':>13} {'公式(1)':>8} "
        f"{'目标前向次数':>11} {'CPU 时间 s':>7} {'加速比':>7} {'公式预测':>8}"
    )
    for k in ks:
        s = st[k]
        alpha = s["accepted"] / s["examined"]
        per_round = (s["accepted"] + s["rounds"]) / s["rounds"]  # 每轮 = 接受数 + 1
        ts = statistics.median(t_spec[k])
        pred = expected_tokens(alpha, k) / (k * c + 1)
        print(
            f"{k:2d} {alpha:13.3f} {per_round:13.2f} {expected_tokens(alpha, k):8.2f} "
            f"{s['rounds']:11d} {ts:7.2f} {tb / ts:6.2f}× {pred:7.2f}×"
        )
    print(
        "（'目标前向次数'是 4 段 × 200 个 token 一轮测量的次数；普通解码需要 800 次。"
        "\n  计时是进程 CPU 时间，仍有波动，加速比只看趋势。公式预测假设验证 k+1 个 token 和生成 1 个一样贵。）"
    )
