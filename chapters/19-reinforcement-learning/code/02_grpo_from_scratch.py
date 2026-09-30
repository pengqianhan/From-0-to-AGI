"""第 19 章 · 02：从零实现 GRPO —— 模仿到头了，让验证器接着教（PyTorch，CPU 约一分钟）

任务：两个个位数相加，a + b，输出答案的数字再输出 <eos>（例如 7+5 → "1" "2" <eos>）。
答案对不对，一个函数就能判断——这就是"可验证奖励"（verifiable reward）。

1. **模仿（SFT）**：老师是个"不太会进位"的模型——不进位的题全对；进位的题只有 30% 写对，
   70% 忘了写进位（7+5 写成 "2"）。学生照着老师的示范学，最后连老师的错误也一起学会了。
2. **GRPO**：同一道题采样 G 个回答，用验证器打分，组内归一化得到优势
   A_i = (r_i − mean(r)) / std(r)，再做带裁剪的策略梯度（外加对 SFT 模型的 k3 KL）。
   没有价值模型（critic），基线就是"同一道题其他回答的平均分"。
3. 和生产级代码对拍：同一批数据上，这里的优势与 zero.post.grpo.group_advantages、
   损失与 zero.post.grpo.grpo_loss 逐项一致。

    uv run python chapters/19-reinforcement-learning/code/02_grpo_from_scratch.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(1)  # 构建环境多个任务共享 CPU；读者本机可删掉

EOS, BOS, V, T = 10, 11, 11, 3  # 词表：0–9 是数字，10 是 <eos>；最多生成 3 个 token
PROMPTS = [(a, b) for a in range(10) for b in range(10)]


def target(a: int, b: int) -> list[int]:
    return [int(c) for c in str(a + b)] + [EOS]


def carry(a: int, b: int) -> bool:
    return a + b >= 10


class TinyPolicy(nn.Module):
    """极小的自回归策略：下一个 token 的分布只看 (a, b, 位置, 上一个 token)。"""

    def __init__(self, d: int = 32, h: int = 128):
        super().__init__()
        self.ea, self.eb = nn.Embedding(10, d), nn.Embedding(10, d)
        self.ep, self.eprev = nn.Embedding(T, d), nn.Embedding(12, d)
        self.mlp = nn.Sequential(nn.Linear(d, h), nn.GELU(), nn.Linear(h, V))

    def forward(self, a, b, prev):  # a, b: (B,)；prev: (B, T) 每个位置的"上一个 token"
        pos = torch.arange(prev.shape[1])
        x = self.ea(a)[:, None] + self.eb(b)[:, None] + self.ep(pos)[None] + self.eprev(prev)
        return self.mlp(x)  # (B, T, V)


def token_logps(model, a, b, seq):
    """seq: (B, T) 回复 token（<eos> 之后用 <eos> 填充）→ 每个位置的 log π(y_t | ·)。"""
    prev = torch.cat([torch.full_like(seq[:, :1], BOS), seq[:, :-1]], dim=1)
    logits = model(a, b, prev)
    return torch.log_softmax(logits, -1).gather(-1, seq[..., None]).squeeze(-1)


def response_mask(seq):
    """回复 token 的掩码：直到并包括第一个 <eos>。"""
    is_eos = (seq == EOS).int()
    before = torch.cumsum(is_eos, 1) - is_eos  # 这个位置之前出现过几个 <eos>
    return before == 0


@torch.no_grad()
def sample(model, a, b, greedy=False, gen=None):
    seq = torch.full((len(a), T), EOS, dtype=torch.long)
    done = torch.zeros(len(a), dtype=torch.bool)
    for t in range(T):
        prevs = torch.cat([torch.full((len(a), 1), BOS), seq[:, :t]], 1)
        prevs = torch.cat([prevs, torch.full((len(a), T - t - 1), EOS)], 1)
        logits = model(a, b, prevs)[:, t]
        nxt = logits.argmax(-1) if greedy else torch.multinomial(F.softmax(logits, -1), 1, generator=gen)[:, 0]
        nxt = torch.where(done, torch.full_like(nxt, EOS), nxt)
        seq[:, t] = nxt
        done |= nxt == EOS
    return seq


def verify(a: int, b: int, seq: list[int]) -> float:
    """可验证奖励：<eos> 之前的数字恰好等于 a + b，且确实输出了 <eos>，得 1 分，否则 0 分。"""
    if EOS not in seq:
        return 0.0
    return 1.0 if seq[: seq.index(EOS) + 1] == target(a, b) else 0.0


# ---------------------------------------------------------------------------
# GRPO 的两个核心函数（和 zero/post/grpo.py 对拍）
# ---------------------------------------------------------------------------


def group_advantages(rewards: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """rewards: (P, G) → A = (r − 组均值) / (组标准差 + eps)。全对或全错的组，A 全是 0。"""
    mean = rewards.mean(1, keepdim=True)
    std = rewards.std(1, keepdim=True)  # 无偏标准差，与 TRL / verl / zero 一致
    return (rewards - mean) / (std + eps)


def grpo_loss(logp, old_logp, ref_logp, adv, mask, eps_clip=0.2, beta=0.02):
    """逐 token：−min(ρA, clip(ρ, 1−ε, 1+ε)A) + β·k3，再对所有回复 token 取平均（token_mean）。"""
    ratio = torch.exp(logp - old_logp)  # ρ_t = π_θ / π_old
    A = adv[:, None]
    per_tok = torch.maximum(-ratio * A, -torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * A)
    d = ref_logp - logp
    kl = torch.exp(d) - d - 1  # k3 估计：非负，期望等于 KL(π_θ ‖ π_ref)
    per_tok = per_tok + beta * kl
    m = mask.float()
    loss = (per_tok * m).sum() / m.sum()
    kl = kl.detach()
    clipped = ((ratio < 1 - eps_clip) | (ratio > 1 + eps_clip)).float()
    return loss, float((kl * m).sum() / m.sum()), float((clipped * m).sum() / m.sum())


# ---------------------------------------------------------------------------


def teacher_demo(a: int, b: int, gen: torch.Generator) -> list[int]:
    """不太会进位的老师：进位题 70% 忘了写进位（7+5 → "2"）。"""
    if carry(a, b) and torch.rand(1, generator=gen).item() < 0.7:
        return [(a + b) % 10, EOS]
    return target(a, b)


def pad(seq: list[int]) -> list[int]:
    return seq + [EOS] * (T - len(seq))


def evaluate(model, gen):
    a = torch.tensor([p[0] for p in PROMPTS])
    b = torch.tensor([p[1] for p in PROMPTS])
    g = sample(model, a, b, greedy=True).tolist()
    greedy = [verify(x, y, s) for (x, y), s in zip(PROMPTS, g)]
    rep = 20  # 采样准确率：每题采 20 次
    s = sample(model, a.repeat(rep), b.repeat(rep), gen=gen).tolist()
    samp = [verify(x, y, q) for (x, y), q in zip(PROMPTS * rep, s)]
    car = [i for i, p in enumerate(PROMPTS) if carry(*p)]
    return {
        "greedy": sum(greedy) / len(greedy),
        "greedy_carry": sum(greedy[i] for i in car) / len(car),
        "sampled": sum(samp) / len(samp),
    }


def sft(model, gen, steps=600, n_demo=20):
    data = [(a, b, pad(teacher_demo(a, b, gen))) for a, b in PROMPTS for _ in range(n_demo)]
    teacher_acc = sum(verify(a, b, s) for a, b, s in data) / len(data)
    A = torch.tensor([d[0] for d in data])
    B = torch.tensor([d[1] for d in data])
    S = torch.tensor([d[2] for d in data])
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    for _ in range(steps):
        idx = torch.randint(len(data), (256,), generator=gen)
        lp = token_logps(model, A[idx], B[idx], S[idx])
        m = response_mask(S[idx]).float()
        loss = -(lp * m).sum() / m.sum()  # SFT：只在回复 token 上算交叉熵
        opt.zero_grad()
        loss.backward()
        opt.step()
    return teacher_acc


def grpo(model, ref, gen, steps=60, P=16, G=8, lr=1e-3, mu=2, log_every=5, crosscheck=None):
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    hist, example = [], None
    for step in range(1, steps + 1):
        idx = torch.randint(len(PROMPTS), (P,), generator=gen)
        a = torch.tensor([PROMPTS[i][0] for i in idx]).repeat_interleave(G)  # 每题复制 G 份
        b = torch.tensor([PROMPTS[i][1] for i in idx]).repeat_interleave(G)
        seq = sample(model, a, b, gen=gen)  # 1. 采样
        r = torch.tensor([verify(x, y, s) for x, y, s in zip(a.tolist(), b.tolist(), seq.tolist())])
        rewards = r.view(P, G)  # 2. 打分
        adv = group_advantages(rewards).view(-1)  # 3. 组内归一化的优势
        mask = response_mask(seq)
        with torch.no_grad():
            old = token_logps(model, a, b, seq)  # π_old：采样时的策略
            reflp = token_logps(ref, a, b, seq)  # π_ref：SFT 模型（冻结）
        if example is None:  # 记下第一组"有对有错"的样本，给视频用
            for j in range(P):
                if rewards[j].std() > 0 and carry(int(a[j * G]), int(b[j * G])):
                    example = {
                        "prompt": [int(a[j * G]), int(b[j * G])],
                        "responses": [
                            "".join(str(t) for t in s[: s.index(EOS)]) if EOS in s else "…"
                            for s in seq[j * G : (j + 1) * G].tolist()
                        ],
                        "rewards": rewards[j].tolist(),
                        "adv": group_advantages(rewards[j : j + 1])[0].tolist(),
                    }
                    break
        for _ in range(mu):  # 4. 同一批样本更新 μ 次；第二次起 ρ ≠ 1，裁剪开始起作用
            logp = token_logps(model, a, b, seq)
            loss, kl, clip = grpo_loss(logp, old, reflp, adv, mask)
            if crosscheck is not None and step == 1:
                crosscheck(rewards, adv, logp, old, reflp, mask, loss)
                crosscheck = None
            opt.zero_grad()
            loss.backward()
            opt.step()
        hist.append({"step": step, "reward": float(r.mean()), "kl": kl, "clip": clip,
                     "zero_std": float((rewards.std(1) == 0).float().mean())})
        if step % log_every == 0:
            ev = evaluate(model, gen)
            hist[-1].update(ev)
            print(f"  第 {step:>3} 步 | 批平均奖励 {r.mean():.2f} | 贪心准确率 {ev['greedy']:.2f}"
                  f"（进位题 {ev['greedy_carry']:.2f}）| 采样准确率 {ev['sampled']:.2f}"
                  f" | KL {kl:.3f} | 裁剪比例 {clip:.2f} | 零方差组 {hist[-1]['zero_std']:.2f}")
    return hist, example


def crosscheck_with_zero(rewards, adv, logp, old, ref, mask, loss):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # 仓库根目录，才能 import zero
    from zero.post.grpo import group_advantages as z_adv
    from zero.post.grpo import grpo_loss as z_loss

    d_adv = (z_adv(rewards).view(-1) - adv).abs().max().item()
    z, _ = z_loss(logp, old, adv, mask, clip_eps=0.2, ref_logp=ref, kl_coef=0.02, loss_agg="token_mean")
    print(f"  对拍 zero.post.grpo：优势最大差 {d_adv:.2e}，损失 {loss.item():+.6f} vs {z.item():+.6f}"
          f"（差 {abs(loss.item() - z.item()):.2e}）")


def run(seed: int = 0, verbose: bool = True):
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    model = TinyPolicy()
    teacher_acc = sft(model, gen)
    ev_sft = evaluate(model, gen)
    if verbose:
        print(f"== 1. 模仿：老师示范的准确率 {teacher_acc:.3f}（不进位全对，进位只对 30%） ==")
        print(f"  SFT 之后：贪心准确率 {ev_sft['greedy']:.2f}（进位题 {ev_sft['greedy_carry']:.2f}），"
              f"采样准确率 {ev_sft['sampled']:.2f}")
        print("\n== 2. GRPO：G=8，每步 16 道题，ε=0.2，β=0.02，μ=2 ==")
    ref = TinyPolicy()
    ref.load_state_dict(model.state_dict())
    ref.requires_grad_(False)
    hist, example = grpo(model, ref, gen, crosscheck=crosscheck_with_zero if verbose else None)
    hist.insert(0, {"step": 0, **ev_sft})
    return {"teacher_acc": teacher_acc, "sft": ev_sft, "hist": hist, "example": example}


def main() -> None:
    out = run()
    ex = out["example"]
    print(f"\n== 3. 一组样本长什么样（第 1 步，题目 {ex['prompt'][0]}+{ex['prompt'][1]}） ==")
    for resp, r, A in zip(ex["responses"], ex["rewards"], ex["adv"]):
        print(f"  回答 {resp:>3}  奖励 {r:.0f}  优势 {A:+.2f}")


if __name__ == "__main__":
    main()
