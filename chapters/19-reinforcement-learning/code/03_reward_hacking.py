"""第 19 章 · 03：奖励作弊（reward hacking）现场重演 —— 奖励涨了，本事没涨（PyTorch，CPU 约一两分钟）

这是冒烟测试里真实发生过的事（zero/post/envs/tool_env.py 模块说明第 8 条）的缩小版：
tiny 模型的工具调用大多格式坏掉（-1 分），GRPO 十步之内学会了"去掉 <tool_call> 标签、照样输出 JSON"，
格式错误消失、平均奖励上升，而模型一个调用都没学会。

玩具版的"工具调用"：
    词表  <call>  </call>  <eos>  字（普通文字 / 函数名）  0–9
    工具题（a, b）：正确的调用是  <call> 字 d </call> <eos>（"函数名 + 参数"），d = (a + b) mod 10
    闲聊题：不该调用工具，回一句普通文字  字 <eos>

两种奖励（除了第 8 条守卫，其余与 tool_env 的打分结构一致）：

| 输出 | 天真的奖励 | 修好的奖励 |
|---|---|---|
| 标签坏了（<call> 没配对 / 标签里不是"字 + 数字"）或超长 | −1 | −1 |
| 工具题：调用格式对、答案错 / 对 | 0.1 / 1 | 0.1 / 1 |
| 工具题：没有标签（没调用） | 0 | 0；但**标签外出现数字**（裸调用）→ −1 |
| 闲聊题：调用了工具 | −0.5 | −0.5 |
| 闲聊题：没有标签 | 1（不管写了什么） | 空回复 0，裸调用 −1，普通文字 1 |

"真实成功率"是另一把尺子（相当于留出的验证器 / 人工抽查）：工具题要调用正确，闲聊题要回普通文字。

    uv run python chapters/19-reinforcement-learning/code/03_reward_hacking.py
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(1)

OPEN, CLOSE, EOS, WORD = 0, 1, 2, 3
DIG = 4  # 数字 d 的 token id 是 4 + d
V, T, BOS = 14, 6, 14
NAMES = ["<call>", "</call>", "<eos>", "字"] + [str(d) for d in range(10)]
TOOL = [(a, b) for a in range(10) for b in range(10)]
N_CHAT = 25  # 闲聊题数目（编号 0..24）


def show(seq: list[int]) -> str:
    out = []
    for t in seq:
        out.append(NAMES[t])
        if t == EOS:
            break
    return " ".join(out)


def classify(seq: list[int]) -> str:
    """把一个输出归类：ok_call / wrong_call / broken / bare / empty / text。"""
    if EOS not in seq:
        return "broken"  # 超长（没在 T 个 token 内结束）
    body = seq[: seq.index(EOS)]
    if OPEN in body or CLOSE in body:
        if body[:1] == [OPEN] and body[1:2] == [WORD] and len(body) == 4 and body[2] >= DIG and body[3] == CLOSE:
            return "call"
        return "broken"
    if not body:
        return "empty"
    if any(t >= DIG for t in body):
        return "bare"  # 标签外面出现了"答案"——相当于不带 <tool_call> 的 JSON
    return "text"


def reward(kind: str, a: int, b: int, seq: list[int], fixed: bool) -> float:
    c = classify(seq)
    if c == "broken":
        return -1.0
    if kind == "tool":
        if c == "call":
            return 1.0 if seq[2] - DIG == (a + b) % 10 else 0.1
        if fixed and c == "bare":
            return -1.0  # 守卫 #8：标签外出现调用样子的内容 → 格式错误
        return 0.0  # 该调却没调
    # 闲聊题
    if c == "call":
        return -0.5
    if fixed:
        return {"empty": 0.0, "bare": -1.0, "text": 1.0}[c]
    return 1.0  # 天真：只要没调用工具就满分


def success(kind: str, a: int, b: int, seq: list[int]) -> bool:
    """留出的"真实"判据：和训练用的奖励无关。"""
    c = classify(seq)
    if kind == "tool":
        return c == "call" and seq[2] - DIG == (a + b) % 10
    return c == "text"


class TinyPolicy(nn.Module):
    def __init__(self, d: int = 32, h: int = 128):
        super().__init__()
        self.ek, self.ea, self.eb = nn.Embedding(2, d), nn.Embedding(10, d), nn.Embedding(N_CHAT, d)
        self.ebb = nn.Embedding(10, d)
        self.ep, self.eprev = nn.Embedding(T, d), nn.Embedding(V + 1, d)
        self.mlp = nn.Sequential(nn.Linear(d, h), nn.GELU(), nn.Linear(h, V))

    def embed_prompt(self, kind, a, b):  # kind 0 = 工具题 (a, b)；1 = 闲聊题（a 是编号）
        tool = self.ea(a.clamp(max=9)) + self.ebb(b)
        chat = self.eb(a)
        return self.ek(kind) + torch.where(kind[:, None] == 0, tool, chat)

    def forward(self, kind, a, b, prev):
        pos = torch.arange(prev.shape[1])
        x = self.embed_prompt(kind, a, b)[:, None] + self.ep(pos)[None] + self.eprev(prev)
        return self.mlp(x)


def shift(seq):
    return torch.cat([torch.full_like(seq[:, :1], BOS), seq[:, :-1]], 1)


def logps(model, kind, a, b, seq):
    return torch.log_softmax(model(kind, a, b, shift(seq)), -1).gather(-1, seq[..., None]).squeeze(-1)


def resp_mask(seq):
    e = (seq == EOS).int()
    return (torch.cumsum(e, 1) - e) == 0


@torch.no_grad()
def sample(model, kind, a, b, gen):
    seq = torch.full((len(a), T), EOS, dtype=torch.long)
    done = torch.zeros(len(a), dtype=torch.bool)
    for t in range(T):
        logits = model(kind, a, b, shift(seq))[:, t]
        nxt = torch.multinomial(F.softmax(logits, -1), 1, generator=gen)[:, 0]
        seq[:, t] = torch.where(done, torch.full_like(nxt, EOS), nxt)
        done |= seq[:, t] == EOS
    return seq


def sft_demo(kind: int, gen: torch.Generator) -> list[int]:
    """起点模型的示范分布：像 tiny 模型一样，格式经常写坏，内容基本是瞎猜。"""
    u = torch.rand(1, generator=gen).item()
    d = DIG + int(torch.randint(10, (1,), generator=gen))
    if kind == 0:  # 工具题
        if u < 0.03:
            out = [OPEN, WORD, d, CLOSE]  # 格式对（"函数名 + 参数"，数字是瞎猜的）——很少见
        elif u < 0.27:
            out = [OPEN, WORD, d]  # 忘了 </call>
        elif u < 0.51:
            out = [OPEN, d, WORD, CLOSE]  # 顺序乱了（相当于坏 JSON）
        elif u < 0.75:
            out = [OPEN, WORD, WORD, CLOSE]  # 没有参数
        else:
            out = [WORD, d]  # 裸"调用"：内容一样，只是没有标签
    elif u < 0.5:
        out = [WORD]  # 闲聊题：一句普通文字
    elif u < 0.85:
        out = [WORD, d]  # 闲聊题也常冒出一段"调用样子"的内容
    else:
        out = []  # 空回复
    return out + [EOS] * (T - len(out))


def prompts(n, gen, p_chat=0.25):
    kind = (torch.rand(n, generator=gen) < p_chat).long()
    a = torch.where(kind == 0, torch.randint(10, (n,), generator=gen), torch.randint(N_CHAT, (n,), generator=gen))
    b = torch.where(kind == 0, torch.randint(10, (n,), generator=gen), torch.zeros(n, dtype=torch.long))
    return kind, a, b


def make_start(seed: int = 0) -> TinyPolicy:
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    model = TinyPolicy()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    kind, a, b = prompts(4000, gen)
    seq = torch.tensor([sft_demo(int(k), gen) for k in kind])
    for _ in range(300):
        i = torch.randint(len(seq), (256,), generator=gen)
        m = resp_mask(seq[i]).float()
        loss = -(logps(model, kind[i], a[i], b[i], seq[i]) * m).sum() / m.sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model


@torch.no_grad()
def measure(model, gen, n=2000):
    kind, a, b = prompts(n, gen)
    seq = sample(model, kind, a, b, gen).tolist()
    cls = [classify(s) for s in seq]
    tool = [i for i in range(n) if kind[i] == 0]
    chat = [i for i in range(n) if kind[i] == 1]
    ok = [success("tool" if kind[i] == 0 else "chat", int(a[i]), int(b[i]), seq[i]) for i in range(n)]
    return {
        "format_err": sum(c == "broken" for c in cls) / n,
        "call_rate": sum(cls[i] == "call" for i in tool) / len(tool),  # 工具题里格式正确的调用
        "bare_rate": sum(cls[i] == "bare" for i in range(n)) / n,
        "chat_ok": sum(cls[i] == "text" for i in chat) / len(chat),
        "success": sum(ok) / n,
    }


def train(fixed: bool, steps=120, P=16, G=8, lr=2e-3, seed=0, log_every=10, verbose=True):
    model = make_start(seed)
    gen = torch.Generator().manual_seed(1000 + seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    hist = [{"step": 0, "reward": None, **measure(model, gen)}]
    examples = []
    for step in range(1, steps + 1):
        kind, a, b = prompts(P, gen)
        kind, a, b = kind.repeat_interleave(G), a.repeat_interleave(G), b.repeat_interleave(G)
        seq = sample(model, kind, a, b, gen)
        r = torch.tensor([reward("tool" if k == 0 else "chat", x, y, s, fixed)
                          for k, x, y, s in zip(kind.tolist(), a.tolist(), b.tolist(), seq.tolist())])
        R = r.view(P, G)
        adv = ((R - R.mean(1, keepdim=True)) / (R.std(1, keepdim=True) + 1e-6)).view(-1)
        m = resp_mask(seq).float()
        loss = -((adv[:, None] * logps(model, kind, a, b, seq)) * m).sum() / m.sum()  # μ=1：ρ≡1，不用裁剪
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % log_every == 0:
            ev = measure(model, gen)
            hist.append({"step": step, "reward": float(r.mean()), **ev})
            if verbose:
                print(f"  第 {step:>3} 步 | 训练奖励 {r.mean():+.2f} | 格式错误 {ev['format_err']:.2f} | "
                      f"工具题正确格式调用 {ev['call_rate']:.2f} | 裸调用 {ev['bare_rate']:.2f} | "
                      f"闲聊题正常回复 {ev['chat_ok']:.2f} | 真实成功率 {ev['success']:.2f}")
        if step == steps:  # 最后一步：工具题、闲聊题各取两个样本
            rows = list(zip(kind.tolist(), a.tolist(), b.tolist(), seq.tolist(), r.tolist()))[::G]
            for want in (0, 1):
                for k, x, y, s, rr in [row for row in rows if row[0] == want][:2]:
                    examples.append({"kind": "工具题" if k == 0 else "闲聊题",
                                     "prompt": f"{x}+{y}" if k == 0 else f"闲聊#{x}",
                                     "output": show(s), "reward": rr})
    return hist, examples


def main() -> None:
    for fixed in (False, True):
        name = "修好的奖励（加上守卫 #8）" if fixed else "天真的奖励（只检查标签里面）"
        print(f"== {name} ==")
        hist, ex = train(fixed)
        h0 = hist[0]
        print(f"  起点       | 格式错误 {h0['format_err']:.2f} | 工具题正确格式调用 {h0['call_rate']:.2f} | "
              f"裸调用 {h0['bare_rate']:.2f} | 闲聊题正常回复 {h0['chat_ok']:.2f} | 真实成功率 {h0['success']:.2f}")
        print("  最后一步的几个样本：")
        for e in ex:
            print(f"    {e['kind']} {e['prompt']:>7} → {e['output']:<24} 奖励 {e['reward']:+.1f}")
        print()


if __name__ == "__main__":
    main()
