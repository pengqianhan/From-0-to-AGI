"""第 17 章 · 极简代码 2：玩具蒸馏实验——硬标签 vs 序列级蒸馏 vs logits 蒸馏

任务：字符级语言模型（莎士比亚文本，assets/tiny_corpus/shakespeare.txt），看前 8 个字符猜下一个字符。

- 教师：大 MLP（约 43 万参数），在全部训练文本（约 100 万字符）上训练；
- 学生：小 MLP（约 9 千参数），**只能看到 2 万字符**的训练文本（模拟"学生的数据不够多"），
  同样的初始化、同样的步数、同样的批大小，比较四种训练信号：
    A. 硬标签：真实的下一个字符（one-hot 交叉熵），就是普通的预训练 / SFT；
    B. 序列级蒸馏：让教师自己**写** 2 万字符，学生在教师写的文本上做普通交叉熵
       （= "教师数据 SFT"，DeepSeek-R1-Distill 的做法）；
    C. logits 蒸馏：在同样 2 万字符的每个位置上，让学生贴教师的整个分布，τ²·KL(p_T^τ ‖ p_S^τ)；
    D. C + A 混合：0.5·硬标签 + 0.5·KD（zero 默认 kd_alpha = 0.5）。
  另外给一个参照 R：学生结构不变，但能看到全部 100 万字符的真实文本（"数据管够"时的上限）。
- 指标：在留出的真实验证文本上的交叉熵，换算成 bits/char（越低越好）。3 个随机种子。

运行：uv run python chapters/17-distillation/code/02_toy_distill.py      （单线程 CPU 时间约 2 分钟；机器繁忙时墙钟时间会长得多）
"""

import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(1)
ROOT = Path(__file__).resolve().parents[3]
CORPUS = ROOT / "assets" / "tiny_corpus" / "shakespeare.txt"

CTX = 8  # 看前 8 个字符
SMALL_N = 20_000  # 学生能看到的训练字符数
TEACHER = dict(emb=32, hidden=512, layers=2)
STUDENT = dict(emb=8, hidden=64, layers=1)
TEACHER_STEPS, STUDENT_STEPS, BATCH = 3000, 2000, 256
TAU = 2.0
SEEDS = [0, 1, 2]


class MLPLM(nn.Module):
    """最简单的"语言模型"：把前 CTX 个字符的 embedding 拼起来，过几层 MLP，输出下一个字符的 logits。"""

    def __init__(self, vocab: int, emb: int, hidden: int, layers: int) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocab, emb)
        dims = [CTX * emb] + [hidden] * layers
        self.body = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:]))
        self.head = nn.Linear(hidden, vocab)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, CTX) → (B, V)
        h = self.emb(x).flatten(1)
        for lin in self.body:
            h = torch.tanh(lin(h))
        return self.head(h)


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def windows(ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """把一串字符 id 切成 (上文 CTX 个字符, 下一个字符) 的训练样本。"""
    x = ids.unfold(0, CTX, 1)[:-1]
    y = ids[CTX:]
    return x, y


@torch.no_grad()
def bits_per_char(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    model.eval()
    ce = F.cross_entropy(model(x), y).item()
    model.train()
    return ce / math.log(2)


def kd_loss(z_s: torch.Tensor, z_t: torch.Tensor, tau: float) -> torch.Tensor:
    """τ²·KL(p_T^τ ‖ p_S^τ)，对 batch 取平均（与 01 和 zero.post.distill.kd_loss 同一个公式）。"""
    logp_t = F.log_softmax(z_t / tau, -1)
    logp_s = F.log_softmax(z_s / tau, -1)
    return (logp_t.exp() * (logp_t - logp_s)).sum(-1).mean() * tau * tau


def train(model: nn.Module, x: torch.Tensor, y: torch.Tensor | None, steps: int, seed: int,
          t_logits: torch.Tensor | None = None, alpha: float = 0.0, lr: float = 3e-3) -> nn.Module:
    """alpha = 0：纯硬标签；alpha = 1：纯 KD；中间：混合。每步从 (x, y) 里随机抽一个 batch。"""
    g = torch.Generator().manual_seed(1000 + seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    for step in range(steps):
        idx = torch.randint(0, len(x), (BATCH,), generator=g)
        z = model(x[idx])
        loss = torch.zeros(())
        if alpha < 1:
            loss = loss + (1 - alpha) * F.cross_entropy(z, y[idx])
        if alpha > 0:
            loss = loss + alpha * kd_loss(z, t_logits[idx], TAU)
        for pg in opt.param_groups:  # 余弦衰减到 10%
            pg["lr"] = lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / steps)))
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model


@torch.no_grad()
def teacher_write(teacher: nn.Module, prompt: torch.Tensor, n: int, seed: int) -> torch.Tensor:
    """让教师自回归地"写" n 个字符（温度 1 采样），开头用一段真实文本。"""
    teacher.eval()
    g = torch.Generator().manual_seed(seed)
    # 一次并行写 20 段，每段 n/20 个字符，省时间
    k = 20
    starts = torch.randint(0, len(prompt) - CTX, (k,), generator=g)
    ctx = torch.stack([prompt[s:s + CTX] for s in starts])
    seqs = [[] for _ in range(k)]
    for _ in range(n // k):
        p = F.softmax(teacher(ctx), -1)
        nxt = torch.multinomial(p, 1, generator=g)
        for i in range(k):
            seqs[i].append(int(nxt[i]))
        ctx = torch.cat([ctx[:, 1:], nxt], 1)
    out = [c for s in seqs for c in s]
    return torch.tensor(out)


def run_experiment(log=print) -> dict:
    """跑完整个实验，返回结果字典（视频 scenes.py 也调用它，并缓存到 video/out/cache.json）。"""
    t0 = time.time()
    text = CORPUS.read_text()
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    ids = torch.tensor([stoi[c] for c in text])
    n_train = int(0.9 * len(ids))
    train_ids, val_ids = ids[:n_train], ids[n_train:]
    x_full, y_full = windows(train_ids)
    x_val, y_val = windows(val_ids)
    x_small, y_small = windows(train_ids[:SMALL_N])
    V = len(chars)
    log(f"语料 {len(text):,} 字符，字符表 {V}；训练 {n_train:,}，验证 {len(val_ids):,}；学生只看前 {SMALL_N:,} 个")

    # ── 教师 ────────────────────────────────────────────────────────────
    torch.manual_seed(0)
    teacher = MLPLM(V, **TEACHER)
    train(teacher, x_full, y_full, TEACHER_STEPS, seed=0)
    teacher.eval()
    t_bpc = bits_per_char(teacher, x_val, y_val)
    log(f"教师：{n_params(teacher):,} 参数，全部训练文本 {TEACHER_STEPS} 步 → 验证 {t_bpc:.3f} bits/char")

    with torch.no_grad():
        t_logits_small = teacher(x_small)  # 教师在学生那 2 万字符每个位置上的 logits（C、D 用）
    written = teacher_write(teacher, train_ids[:SMALL_N], SMALL_N, seed=123)
    x_seq, y_seq = windows(written)  # B 用：教师写的文本
    sample = "".join(chars[i] for i in written[:160].tolist())
    log(f"教师写的文本（前 160 字符）：{sample!r}")

    # ── 学生：同样的初始化、步数、批大小，只换训练信号 ───────────────────────
    arms = {
        "A 硬标签（2 万真实字符）": dict(x=x_small, y=y_small),
        "B 序列级蒸馏（教师写的 2 万字符）": dict(x=x_seq, y=y_seq),
        f"C logits 蒸馏（τ={TAU:g}，同 2 万字符）": dict(x=x_small, y=y_small, t_logits=t_logits_small, alpha=1.0),
        "D 0.5·硬标签 + 0.5·KD": dict(x=x_small, y=y_small, t_logits=t_logits_small, alpha=0.5),
        "R 参照：硬标签（全部 100 万字符）": dict(x=x_full, y=y_full),
    }
    results: dict[str, list[tuple[float, float]]] = {k: [] for k in arms}
    for seed in SEEDS:
        for name, kw in arms.items():
            torch.manual_seed(seed)
            student = MLPLM(V, **STUDENT)
            train(student, kw["x"], kw["y"], STUDENT_STEPS, seed, kw.get("t_logits"), kw.get("alpha", 0.0))
            tr = bits_per_char(student, x_small, y_small)
            va = bits_per_char(student, x_val, y_val)
            results[name].append((tr, va))
    s_params = n_params(MLPLM(V, **STUDENT))
    return {
        "teacher_params": n_params(teacher), "student_params": s_params, "teacher_bpc": t_bpc,
        "uniform_bpc": math.log2(V), "vocab": V, "sample": sample, "seeds": SEEDS,
        "steps": STUDENT_STEPS, "batch": BATCH, "tau": TAU,
        "arms": {name: {"train": [r[0] for r in rs], "val": [r[1] for r in rs]} for name, rs in results.items()},
        "seconds": time.time() - t0,
    }


def main() -> None:
    r = run_experiment()
    print(f"\n学生：{r['student_params']:,} 参数，每组 {r['steps']} 步 × batch {r['batch']}，种子 {r['seeds']}")
    print(f"{'训练信号':<30}{'2万训练字符上':>12}{'验证 bits/char':>16}  各种子（验证）")
    for name, a in r["arms"].items():
        tr = sum(a["train"]) / len(a["train"])
        va = sum(a["val"]) / len(a["val"])
        per = " / ".join(f"{x:.3f}" for x in a["val"])
        print(f"{name:<30}{tr:>12.3f}{va:>16.3f}  {per}")
    print(f"（教师验证 {r['teacher_bpc']:.3f}；均匀乱猜 log2({r['vocab']}) = {r['uniform_bpc']:.3f}）")
    print(f"用时 {r['seconds']:.0f}s")


if __name__ == "__main__":
    main()
