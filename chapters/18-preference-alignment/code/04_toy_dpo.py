"""第 18 章 · 极简代码 4：在一个小语言模型上跑 DPO

任务：提示词 "a+b="（a、b 是 0–9），回答是和的数字，最后以 ";" 结束。
① "SFT"：用**质量参差不齐**的数据训练一个小 GRU 语言模型——同一道题，40% 的示范是对的，
   60% 是随便写的错数字。它学会了格式，也学会了"经常答错"。这就是参考模型 π_ref。
② 偏好数据：70 道题，每道 4 对（chosen = 正确答案，rejected = π_ref 会写出的错答案）。
   另外 30 道题留出，训练时从不出现。
③ DPO：看训练中隐式奖励的 margin、准确率、chosen / rejected 的 log 概率怎么动；
   留出题上"答对的概率"、采样的格式正确率。
④ 扫 β。
学习率太大、chosen 的概率也一起掉等坑，见 05_dpo_pitfalls.py。

运行：uv run python chapters/18-preference-alignment/code/04_toy_dpo.py   （CPU 时间约半分钟；机器繁忙时墙钟几分钟）
"""

from __future__ import annotations

import copy
import random
from functools import lru_cache

import torch
import torch.nn.functional as F

torch.set_num_threads(1)  # 构建机多任务共享 CPU（本机可删）

VOCAB = list("0123456789+=;")
STOI = {c: i for i, c in enumerate(VOCAB)}
EOS = STOI[";"]
PROMPTS = [(a, b) for a in range(10) for b in range(10)]
_rng = random.Random(0)
_order = PROMPTS[:]
_rng.shuffle(_order)
TRAIN_PROMPTS, HELDOUT_PROMPTS = _order[:70], _order[70:]
P_CORRECT_SFT = 0.4


def encode(s: str) -> list[int]:
    return [STOI[c] for c in s]


def wrong_answers(a: int, b: int, mistakes: str = "random") -> list[int]:
    """错答案。"random"：0–18 里任何一个错数字；"near"：只差 1 或 2（05 用它演示一个坑）。"""
    if mistakes == "near":
        return [a + b + d for d in (-2, -1, 1, 2) if a + b + d >= 0]
    return [x for x in range(19) if x != a + b]


class TinyLM(torch.nn.Module):
    """字符级 GRU 语言模型（约 2.0 万参数）。换成 Transformer，下面的 DPO 代码一行都不用改。"""

    def __init__(self, d: int = 32, h: int = 64) -> None:
        super().__init__()
        self.emb = torch.nn.Embedding(len(VOCAB), d)
        self.rnn = torch.nn.GRU(d, h, batch_first=True)
        self.head = torch.nn.Linear(h, len(VOCAB))

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(self.emb(ids))
        return self.head(out)


def pad(seqs: list[list[int]]) -> tuple[torch.Tensor, torch.Tensor]:
    L = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), L), EOS, dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, : len(s)] = torch.tensor(s)
    return ids, torch.tensor([len(s) for s in seqs])


def response_logps(model: TinyLM, prompts: list[tuple[int, int]], answers: list[int]) -> torch.Tensor:
    """每条"提示词 + 回答"里**只对回答部分**求 log 概率之和（提示词不算，和 zero 一样）。"""
    seqs, starts = [], []
    for (a, b), ans in zip(prompts, answers):
        p = encode(f"{a}+{b}=")
        seqs.append(p + encode(f"{ans};"))
        starts.append(len(p))
    ids, lens = pad(seqs)
    logp = torch.log_softmax(model(ids[:, :-1]), -1).gather(-1, ids[:, 1:, None]).squeeze(-1)
    pos = torch.arange(ids.shape[1] - 1)[None, :] + 1  # logp[:, t] 预测的是第 t+1 个 token
    mask = (pos >= torch.tensor(starts)[:, None]) & (pos < lens[:, None])
    return (logp * mask).sum(-1)


@lru_cache(maxsize=2)
def sft_model(mistakes: str = "random") -> TinyLM:
    """① 在参差不齐的示范上做 SFT：答对 40%，答错 60%。"""
    torch.manual_seed(0)
    g = random.Random(1)
    model = TinyLM()
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    for _ in range(1500):
        batch = [g.choice(PROMPTS) for _ in range(64)]
        ans = [a + b if g.random() < P_CORRECT_SFT else g.choice(wrong_answers(a, b, mistakes))
               for a, b in batch]
        loss = -response_logps(model, batch, ans).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model.eval()


def make_pairs(mistakes: str = "random", seed: int = 0, per_prompt: int = 4) -> list[tuple[tuple[int, int], int, int]]:
    """② 偏好对：(提示词, chosen = 正确答案, rejected = 一个 π_ref 会犯的错)。"""
    g = random.Random(seed)
    return [((a, b), a + b, g.choice(wrong_answers(a, b, mistakes)))
            for a, b in TRAIN_PROMPTS for _ in range(per_prompt)]


@torch.no_grad()
def sample(model: TinyLM, prompts: list[tuple[int, int]], n: int, seed: int, max_new: int = 4) -> list[str]:
    """温度 1 采样，返回生成的字符串（不含提示词）。"""
    g = torch.Generator().manual_seed(seed)
    outs = []
    for a, b in prompts:
        ids = torch.tensor([encode(f"{a}+{b}=")] * n)
        gen = [""] * n
        done = torch.zeros(n, dtype=torch.bool)
        for _ in range(max_new):
            probs = torch.softmax(model(ids)[:, -1], -1)
            nxt = torch.multinomial(probs, 1, generator=g)
            ids = torch.cat([ids, nxt], 1)
            for i in range(n):
                if not done[i]:
                    gen[i] += VOCAB[int(nxt[i])]
            done |= nxt[:, 0] == EOS
            if done.all():
                break
        outs.extend(gen)
    return outs


def well_formed(s: str) -> bool:
    return s.endswith(";") and 1 <= len(s) - 1 <= 2 and s[:-1].isdigit()


@torch.no_grad()
def evaluate(model: TinyLM, ref: TinyLM, n_samples: int = 20, mistakes: str = "random") -> dict:
    """留出题：答对的概率、采样的格式正确率与正确率、隐式奖励的偏好准确率（β 无关，只看符号）。"""
    prompts = HELDOUT_PROMPTS
    p_correct = response_logps(model, prompts, [a + b for a, b in prompts]).exp().mean()
    outs = sample(model, prompts, n_samples, seed=123)
    truth = [f"{a + b};" for a, b in prompts for _ in range(n_samples)]
    fmt = sum(well_formed(s) for s in outs) / len(outs)
    acc = sum(s == t for s, t in zip(outs, truth)) / len(outs)
    g = random.Random(7)
    wrong = [g.choice(wrong_answers(a, b, mistakes)) for a, b in prompts]
    right = [a + b for a, b in prompts]
    h = (response_logps(model, prompts, right) - response_logps(ref, prompts, right)) - (
        response_logps(model, prompts, wrong) - response_logps(ref, prompts, wrong))
    return {"p_correct": float(p_correct), "format": fmt, "sample_acc": acc,
            "pref_acc": float((h > 0).float().mean())}


def run_dpo(beta: float = 0.1, lr: float = 1e-3, steps: int = 150, bsz: int = 32, seed: int = 0,
            log_every: int = 0, mistakes: str = "random") -> tuple[TinyLM, list[dict]]:
    """③ DPO 训练。参考模型的 log 概率预先算好（zero 的 ref_mode = "precompute"）。"""
    ref = sft_model(mistakes)
    policy = copy.deepcopy(ref).train()
    pairs = make_pairs(mistakes)
    prompts = [p for p, _, _ in pairs]
    yw = [w for _, w, _ in pairs]
    yl = [l for _, _, l in pairs]
    with torch.no_grad():
        ref_w, ref_l = response_logps(ref, prompts, yw), response_logps(ref, prompts, yl)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    g = torch.Generator().manual_seed(seed)
    hist = []
    for step in range(steps + 1):
        if log_every and step % log_every == 0:
            with torch.no_grad():
                pw, pl = response_logps(policy, prompts, yw), response_logps(policy, prompts, yl)
            h = beta * ((pw - ref_w) - (pl - ref_l))
            hist.append({"step": step, "loss": float(-F.logsigmoid(h).mean()),
                         "margin": float(h.mean()), "acc": float((h > 0).float().mean()),
                         "logp_w": float(pw.mean()), "logp_l": float(pl.mean())})
        if step == steps:
            break
        idx = torch.randint(0, len(pairs), (bsz,), generator=g)
        pw = response_logps(policy, [prompts[i] for i in idx], [yw[i] for i in idx])
        pl = response_logps(policy, [prompts[i] for i in idx], [yl[i] for i in idx])
        h = beta * ((pw - ref_w[idx]) - (pl - ref_l[idx]))  # 隐式奖励差
        loss = -F.logsigmoid(h).mean()                       # DPO 损失
        opt.zero_grad()
        loss.backward()
        opt.step()
    return policy.eval(), hist


def main_results() -> dict:
    """04 的全部数字（视频 scenes.py 也调用它）。"""
    ref = sft_model()
    base = evaluate(ref, ref)
    policy, hist = run_dpo(beta=0.1, lr=1e-3, steps=150, log_every=25)
    after = evaluate(policy, ref)
    sweep = []
    for beta in (0.03, 0.1, 0.3, 1.0):
        pol, hh = (policy, [hist[0], hist[-1]]) if beta == 0.1 else run_dpo(beta=beta, log_every=150)
        e = evaluate(pol, ref)
        sweep.append({"beta": beta, "loss": hh[-1]["loss"], "margin": hh[-1]["margin"],
                      "d_logp_w": hh[-1]["logp_w"] - hh[0]["logp_w"],
                      "d_logp_l": hh[-1]["logp_l"] - hh[0]["logp_l"], "p_correct": e["p_correct"],
                      "sample_acc": e["sample_acc"]})
    return {"base": base, "hist": hist, "after": after, "n_pairs": len(make_pairs()), "beta_sweep": sweep}


def main() -> None:
    R = main_results()
    base, hist, after = R["base"], R["hist"], R["after"]
    print(f"① SFT 参考模型（示范里 {P_CORRECT_SFT:.0%} 是对的）在 30 道留出题上：")
    print(f"   答对的概率 {base['p_correct']:.3f} | 采样格式正确 {base['format']:.3f} | 采样答对 {base['sample_acc']:.3f}")
    print(f"\n② 偏好数据：{R['n_pairs']} 对（70 道题 × 4），chosen = 正确答案，rejected = 错答案")
    print("③ DPO（β = 0.1，lr = 1e-3，每步 32 对，150 步），训练集上的指标：")
    print(f"   {'步':>4} | {'损失':>6} | {'margin':>7} | {'acc':>5} | {'log π(chosen)':>13} | {'log π(rejected)':>15}")
    for h in hist:
        print(f"   {h['step']:>4} | {h['loss']:.4f} | {h['margin']:>+7.3f} | {h['acc']:.2f} | "
              f"{h['logp_w']:>13.3f} | {h['logp_l']:>15.3f}")
    print(f"   留出题：答对的概率 {base['p_correct']:.3f} → {after['p_correct']:.3f} | "
          f"采样答对 {base['sample_acc']:.3f} → {after['sample_acc']:.3f} | "
          f"格式 {base['format']:.3f} → {after['format']:.3f} | 隐式奖励排序正确 {after['pref_acc']:.2f}")

    print("\n④ 扫 β（lr = 1e-3，150 步）：")
    print("   margin 是隐式奖励之差（已乘 β）；margin/β 是 log 比值之差——策略离 π_ref 走了多远")
    print(f"   {'β':>5} | {'最终损失':>7} | {'margin':>7} | {'margin/β':>8} | {'Δlog π(chosen)':>14} | {'Δlog π(rejected)':>16} | {'留出答对概率':>10}")
    for r in R["beta_sweep"]:
        print(f"   {r['beta']:>5} | {r['loss']:>7.4f} | {r['margin']:>+7.3f} | {r['margin'] / r['beta']:>+8.2f} | "
              f"{r['d_logp_w']:>+14.3f} | {r['d_logp_l']:>+16.3f} | {r['p_correct']:>10.3f}")


if __name__ == "__main__":
    main()
