"""第 8 章 · 极简代码 4：在小语料上训练一个单层注意力模型，看它到底在"看"哪里

三个模型，结构只差"怎么混合前面的 token"这一步，其余完全相同（字符级，Tiny Shakespeare）：

    bigram     h = emb(x) + pos                 只看当前字符（第 7 章的 bigram 加上位置）
    average    h = h + Wo · 均匀平均(Wv h)       前缀平均：看得到前文，但每个字一样重要
    attention  h = h + MultiHeadAttention(h)    权重由 Q·K 决定（本章）
    logits = lm_head(h)

每个模型在 CPU 上训练约 20–40 秒。训练完打印验证集损失（nats 和 bits-per-byte），
再把注意力模型在一段文本上的注意力矩阵画成字符热力图，并统计每个头平均看向哪里。

运行：uv run python chapters/08-attention/code/04_train_attention.py
"""

from __future__ import annotations

import importlib.util
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

HERE = Path(__file__).resolve().parent
CORPUS = HERE.parents[2] / "assets" / "tiny_corpus" / "shakespeare.txt"

# 这么小的模型，单线程反而最快（多线程的调度开销比计算还大，机器忙时尤其明显）
torch.set_num_threads(1)

_spec = importlib.util.spec_from_file_location("attn02", HERE / "02_attention_from_scratch.py")
attn02 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(attn02)

BLOCK = 64       # 上下文长度 T
C = 64           # 通道数
HEADS = 4        # 头数，每头 d = 16
BATCH = 32
STEPS = 2000
LR = 3e-3
SAMPLE = "First Citizen:\nBefore we proceed any further, hear me speak."


def load_data():
    text = CORPUS.read_text(encoding="utf-8")
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    n = int(0.9 * len(data))
    return chars, stoi, data[:n], data[n:]


def get_batch(data: torch.Tensor, g: torch.Generator):
    ix = torch.randint(len(data) - BLOCK - 1, (BATCH,), generator=g)
    x = torch.stack([data[i : i + BLOCK] for i in ix])
    y = torch.stack([data[i + 1 : i + BLOCK + 1] for i in ix])
    return x, y


class TinyLM(nn.Module):
    def __init__(self, vocab: int, mode: str) -> None:
        super().__init__()
        self.mode = mode
        self.tok = nn.Embedding(vocab, C)
        self.pos = nn.Embedding(BLOCK, C)   # 位置向量（第 9 章换成 RoPE）
        if mode == "attention":
            self.mix = attn02.MultiHeadAttention(C, HEADS)
        elif mode == "average":
            self.wv = nn.Linear(C, C, bias=False)
            self.wo = nn.Linear(C, C, bias=False)
        self.head = nn.Linear(C, vocab)

    def forward(self, idx: torch.Tensor, return_weights: bool = False):
        T = idx.shape[1]
        h = self.tok(idx) + self.pos(torch.arange(T))
        w = None
        if self.mode == "attention":
            out, w = self.mix(h, return_weights=True)
            h = h + out                                        # 残差连接（第 6 章）
        elif self.mode == "average":
            W = torch.tril(torch.ones(T, T))
            W = W / W.sum(1, keepdim=True)                     # 固定的均匀权重
            h = h + self.wo(W @ self.wv(h))
        logits = self.head(h)
        return (logits, w) if return_weights else logits


@torch.no_grad()
def evaluate(model: TinyLM, data: torch.Tensor, batches: int = 40) -> float:
    model.eval()
    g = torch.Generator().manual_seed(1234)   # 三个模型用同一批验证数据
    losses = []
    for _ in range(batches):
        x, y = get_batch(data, g)
        losses.append(F.cross_entropy(model(x).flatten(0, 1), y.flatten()).item())
    model.train()
    return sum(losses) / len(losses)


def train(mode: str, steps: int = STEPS, seed: int = 0, log_every: int = 500):
    torch.manual_seed(seed)
    chars, _, train_data, val_data = load_data()
    model = TinyLM(len(chars), mode)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    g = torch.Generator().manual_seed(seed)
    hist = []
    for step in range(steps + 1):
        if step % log_every == 0:
            hist.append((step, evaluate(model, val_data)))
        if step == steps:
            break
        x, y = get_batch(train_data, g)
        loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten())
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model, hist


@torch.no_grad()
def attention_on(model: TinyLM, text: str) -> torch.Tensor:
    """返回注意力权重 (H, T, T)。"""
    _, stoi, _, _ = load_data()
    idx = torch.tensor([[stoi[c] for c in text]])
    model.eval()
    _, w = model(idx, return_weights=True)
    return w[0]


@torch.no_grad()
def head_profile(model: TinyLM, data: torch.Tensor, batches: int = 20) -> torch.Tensor:
    """每个头平均把多少权重放在"往前数 k 个"的位置上：返回 (H, 4)，列是 k = 0, 1, 2, ≥3。"""
    model.eval()
    g = torch.Generator().manual_seed(99)
    acc = torch.zeros(HEADS, 4)
    T = BLOCK
    offset = torch.arange(T)[:, None] - torch.arange(T)[None, :]   # 查询位置 − 键位置
    for _ in range(batches):
        x, _ = get_batch(data, g)
        _, w = model(x, return_weights=True)                         # (B, H, T, T)
        w = w[:, :, 8:, :].mean(0)                                   # 跳过开头 8 个位置（前文太短）
        off = offset[8:]
        for k in range(3):
            acc[:, k] += (w * (off == k)).sum(-1).mean(-1)
        acc[:, 3] += (w * (off >= 3)).sum(-1).mean(-1)
    return acc / batches


def show_char(c: str) -> str:
    return {"\n": "⏎", " ": "␣"}.get(c, c)


def heatmap(w: torch.Tensor, text: str) -> str:
    """把 (T, T) 权重画成字符热力图：行 = 查询（当前位置），列 = 键（被看的位置）。"""
    shades = " .:-=+*#%@"
    lines = ["    " + "".join(show_char(c) for c in text)]
    for i, c in enumerate(text):
        row = "".join(shades[min(9, int(w[i, j] * 10))] if j <= i else " " for j in range(len(text)))
        lines.append(f"  {show_char(c)} {row}")
    return "\n".join(lines)


def main() -> None:
    chars, _, _, val_data = load_data()
    print(f"语料：Tiny Shakespeare，{len(chars)} 种字符（全是 ASCII，1 字符 = 1 字节）；"
          f"上下文 T={BLOCK}，C={C}，H={HEADS}，训练 {STEPS} 步 × batch {BATCH}\n")
    models, results = {}, []
    for mode in ("bigram", "average", "attention"):
        t0 = time.time()
        model, hist = train(mode)
        dt = time.time() - t0
        models[mode] = model
        final = hist[-1][1]
        n_params = sum(p.numel() for p in model.parameters())
        results.append((mode, n_params, final, final / math.log(2), dt))
        curve = "  ".join(f"{s}:{v:.3f}" for s, v in hist)
        print(f"[{mode:>9}] 验证损失（步数:nats）{curve}   用时 {dt:.0f}s")

    print("\n| 模型 | 参数量 | 验证损失（nats/字符） | bits-per-byte |")
    print("|---|---:|---:|---:|")
    for mode, n, loss, bpb, _ in results:
        print(f"| {mode} | {n:,} | {loss:.3f} | {bpb:.3f} |")

    model = models["attention"]
    prof = head_profile(model, val_data)
    print("\n注意力模型：每个头平均把权重放在哪里（验证集，查询位置 ≥ 8）")
    print("| 头 | 自己 (k=0) | 前 1 个 | 前 2 个 | 更早 (k≥3) |")
    print("|---|---:|---:|---:|---:|")
    for h in range(HEADS):
        print(f"| {h} | " + " | ".join(f"{v:.2f}" for v in prof[h]) + " |")

    w = attention_on(model, SAMPLE)
    text = SAMPLE
    for h in range(HEADS):
        print(f"\n头 {h} 在样例文本上的注意力（行 = 当前字符，列 = 被看的字符；越深越重，␣ 是空格，⏎ 是换行）")
        print(heatmap(w[h, 15:, 15:], text[15:]))   # 只画第二行，保持宽度可读
    # 几个具体位置：看得最重的前 3 个字符
    print("\n几个位置最关注的前 3 个字符（全部 4 个头平均）：")
    wm = w.mean(0)
    for pos in [text.index("proceed") + 6, text.index("further") + 6, len(text) - 2]:
        top = torch.topk(wm[pos, : pos + 1], 3)
        desc = "，".join(f"{pos - int(j)} 前「{show_char(text[int(j)])}」{float(v):.2f}"
                        for v, j in zip(top.values, top.indices))
        print(f"  位置 {pos}「{show_char(text[pos])}」（前文 …{text[max(0, pos - 12): pos + 1]!r}）：{desc}")


if __name__ == "__main__":
    main()
