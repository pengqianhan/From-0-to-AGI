"""Chapter 19 · 03: reward hacking, replayed. The reward goes up, the skill does not
(PyTorch, about 1–2 min on a CPU).

This is a small replay of a real event in the smoke test (item 8 in the module docstring of
zero/post/envs/tool_env.py). Most tool calls of the tiny model had a broken format (reward −1).
In 10 steps, GRPO learned to "remove the <tool_call> tags and output the JSON anyway".
The format errors went away and the mean reward went up, but the model did not learn one correct call.

The toy "tool call":
    vocabulary  <call>  </call>  <eos>  字 (a "word" token: plain text or a function name)  0–9
    tool task (a, b): the correct call is  <call> 字 d </call> <eos>  ("function name + argument"),
                      with d = (a + b) mod 10
    chat task: do not call a tool; reply with plain text  字 <eos>

Two rewards (they differ only in guard 8; the rest has the same structure as the tool_env scoring):

| Output | Naive reward | Fixed reward |
|---|---|---|
| Broken tags (<call> without a pair / not "字 + digit" in the tags) or too long | −1 | −1 |
| Tool task: call with the correct format, wrong / correct answer | 0.1 / 1 | 0.1 / 1 |
| Tool task: no tags (no call) | 0 | 0; but **a digit outside the tags** (bare call) → −1 |
| Chat task: calls a tool | −0.5 | −0.5 |
| Chat task: no tags | 1 (for any text) | empty reply 0, bare call −1, plain text 1 |

The "true success rate" is a different measure (like a held-out verifier or a human spot check):
a tool task needs a correct call, and a chat task needs a plain-text reply.

    uv run python chapters/19-reinforcement-learning/code/03_reward_hacking.py
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(1)

OPEN, CLOSE, EOS, WORD = 0, 1, 2, 3
DIG = 4  # the token ID of digit d is 4 + d
V, T, BOS = 14, 6, 14
NAMES = ["<call>", "</call>", "<eos>", "字"] + [str(d) for d in range(10)]
TOOL = [(a, b) for a in range(10) for b in range(10)]
N_CHAT = 25  # number of chat tasks (IDs 0..24)


def show(seq: list[int]) -> str:
    out = []
    for t in seq:
        out.append(NAMES[t])
        if t == EOS:
            break
    return " ".join(out)


def classify(seq: list[int]) -> str:
    """Put one output into a class: call / broken / bare / empty / text (reward() checks if a call is correct)."""
    if EOS not in seq:
        return "broken"  # too long (did not end within T tokens)
    body = seq[: seq.index(EOS)]
    if OPEN in body or CLOSE in body:
        if body[:1] == [OPEN] and body[1:2] == [WORD] and len(body) == 4 and body[2] >= DIG and body[3] == CLOSE:
            return "call"
        return "broken"
    if not body:
        return "empty"
    if any(t >= DIG for t in body):
        return "bare"  # an "answer" outside the tags: like JSON without <tool_call>
    return "text"


def reward(kind: str, a: int, b: int, seq: list[int], fixed: bool) -> float:
    c = classify(seq)
    if c == "broken":
        return -1.0
    if kind == "tool":
        if c == "call":
            return 1.0 if seq[2] - DIG == (a + b) % 10 else 0.1
        if fixed and c == "bare":
            return -1.0  # guard #8: content that looks like a call outside the tags → format error
        return 0.0  # should call, but did not
    # chat task
    if c == "call":
        return -0.5
    if fixed:
        return {"empty": 0.0, "bare": -1.0, "text": 1.0}[c]
    return 1.0  # naive: full score for any output without a tool call


def success(kind: str, a: int, b: int, seq: list[int]) -> bool:
    """The held-out "true" criterion: it does not depend on the training reward."""
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

    def embed_prompt(self, kind, a, b):  # kind 0 = tool task (a, b); 1 = chat task (a is the ID)
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
    """The demo distribution of the start model: like the tiny model, the format is often broken
    and the content is mostly a guess."""
    u = torch.rand(1, generator=gen).item()
    d = DIG + int(torch.randint(10, (1,), generator=gen))
    if kind == 0:  # tool task
        if u < 0.03:
            out = [OPEN, WORD, d, CLOSE]  # correct format ("function name + argument", the digit is a guess): rare
        elif u < 0.27:
            out = [OPEN, WORD, d]  # forgets </call>
        elif u < 0.51:
            out = [OPEN, d, WORD, CLOSE]  # wrong order (like broken JSON)
        elif u < 0.75:
            out = [OPEN, WORD, WORD, CLOSE]  # no argument
        else:
            out = [WORD, d]  # bare "call": the same content, but no tags
    elif u < 0.5:
        out = [WORD]  # chat task: one line of plain text
    elif u < 0.85:
        out = [WORD, d]  # chat tasks also often get content that looks like a call
    else:
        out = []  # empty reply
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
        "call_rate": sum(cls[i] == "call" for i in tool) / len(tool),  # calls with the correct format on tool tasks
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
        loss = -((adv[:, None] * logps(model, kind, a, b, seq)) * m).sum() / m.sum()  # μ=1: ρ≡1, no clipping
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % log_every == 0:
            ev = measure(model, gen)
            hist.append({"step": step, "reward": float(r.mean()), **ev})
            if verbose:
                print(f"  step {step:>3} | train reward {r.mean():+.2f} | format errors {ev['format_err']:.2f} | "
                      f"tool: correct-format call {ev['call_rate']:.2f} | bare call {ev['bare_rate']:.2f} | "
                      f"chat: normal reply {ev['chat_ok']:.2f} | true success {ev['success']:.2f}")
        if step == steps:  # last step: take two samples of tool tasks and two of chat tasks
            rows = list(zip(kind.tolist(), a.tolist(), b.tolist(), seq.tolist(), r.tolist()))[::G]
            for want in (0, 1):
                for k, x, y, s, rr in [row for row in rows if row[0] == want][:2]:
                    examples.append({"kind": "tool" if k == 0 else "chat",
                                     "prompt": f"{x}+{y}" if k == 0 else f"chat#{x}",
                                     "output": show(s), "reward": rr})
    return hist, examples


def main() -> None:
    for fixed in (False, True):
        name = "Fixed reward (with guard #8)" if fixed else "Naive reward (checks only inside the tags)"
        print(f"== {name} ==")
        hist, ex = train(fixed)
        h0 = hist[0]
        print(f"  start    | format errors {h0['format_err']:.2f} | tool: correct-format call {h0['call_rate']:.2f} | "
              f"bare call {h0['bare_rate']:.2f} | chat: normal reply {h0['chat_ok']:.2f} | true success {h0['success']:.2f}")
        print("  Samples from the last step:")
        for e in ex:
            print(f"    {e['kind']} {e['prompt']:>7} → {e['output']:<24} reward {e['reward']:+.1f}")
        print()


if __name__ == "__main__":
    main()
