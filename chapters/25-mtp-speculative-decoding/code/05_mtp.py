"""第 25 章 · 极简代码 5：多 token 预测（MTP）——多学一个"下下个 token"，再拿它当草稿

DeepSeek-V3 式的 MTP 模块（深度 D = 1）接在第 10 章的小模型后面：
    h'_i = W [RMSNorm(Emb(t_{i+1})) ; RMSNorm(h_i)]      # 主模型在位置 i 的表示 + 第 i+1 个 token 的 embedding
    h''  = Block(h')                                     # 一个 Transformer block（因果注意力）
    P(t_{i+2}) = softmax(RMSNorm(h''_i) · Embᵀ)          # 与主模型共享的 embedding / 输出头
    L = L_main + λ · L_MTP                               # λ = 0.3（DeepSeek-V3 前 10T token 的取值）

训练：与第 10 章完全相同的初始化、数据顺序和超参（600 步），只多了 MTP 损失；
第 10 章缓存的模型（种子 0 和 1）正好就是 λ = 0 的对照组。
推理：MTP 模块当草稿做自推测解码——主模型每次前向验证 1 个草稿，并多给 1 个 token。

运行：uv run python chapters/25-mtp-speculative-decoding/code/05_mtp.py
（首次要训练两个带 MTP 的模型，单线程每个约两三分钟，之后读缓存 code/out/*.pt）
"""

from __future__ import annotations

import importlib.util
import math
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(1)
HERE = Path(__file__).resolve().parent
OUT = HERE / "out"


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


m1 = _load("ch25_models", "01_models_and_cost.py")
ch10, KVCache, truncate = m1.ch10, m1.KVCache, m1.truncate


class MTPHead(nn.Module):
    """一个 MTP 模块：两个 RMSNorm → 拼接后投影 2d → d → 一个 Transformer block → RMSNorm。"""

    def __init__(self, c) -> None:
        super().__init__()
        self.enorm, self.hnorm = ch10.RMSNorm(c.dim), ch10.RMSNorm(c.dim)
        self.proj = nn.Linear(2 * c.dim, c.dim, bias=False)
        self.block = ch10.Block(c)
        self.norm = ch10.RMSNorm(c.dim)

    def forward(self, h, emb_next, cos, sin, cache=None):
        x = self.proj(torch.cat([self.enorm(emb_next), self.hnorm(h)], dim=-1))
        return self.norm(self.block(x, cos, sin, cache, 0))


def hidden(lm, ids, cache=None):
    """主模型最后一个 RMSNorm 之后的表示（乘 Embᵀ 就是 logits）。与 TinyLM.forward 相同，只是不乘输出头。"""
    T = ids.shape[1]
    start = len(cache) if cache is not None else 0
    cos, sin = lm.cos[start : start + T], lm.sin[start : start + T]
    x = lm.emb(ids)
    for layer, blk in enumerate(lm.blocks):
        x = blk(x, cos, sin, cache, layer)
    return lm.norm(x)


def mtp_forward(lm, head, x):
    """x: (B, T) → 主模型 logits (B, T, V)、MTP logits (B, T−1, V)。MTP 在位置 i 预测 x[i+2]。"""
    h = hidden(lm, x)
    T = x.shape[1]
    hm = head(h[:, :-1], lm.emb(x[:, 1:]), lm.cos[: T - 1], lm.sin[: T - 1])
    return h @ lm.emb.weight.T, hm @ lm.emb.weight.T


def train_mtp(seed: int, lam: float = 0.3, steps: int = 600, bsz: int = 16, seq: int = 64,
              lr: float = 3e-3):
    """与 ch10.train 同样的初始化 / 数据 / 学习率，只多了 λ · L_MTP。"""
    data = ch10.CharData()
    c = ch10.Config(vocab_size=data.vocab_size)
    torch.manual_seed(seed)
    lm = ch10.TinyLM(c)  # 先建主模型：随机数用法与 ch10.train 完全相同，初始权重一致
    head = MTPHead(c)  # 线性层用 PyTorch 默认初始化，和第 10 章的 Block 一样
    params = list(lm.parameters()) + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.1)
    g = torch.Generator().manual_seed(seed)
    t0 = time.time()
    for step in range(steps + 1):
        for pg in opt.param_groups:
            pg["lr"] = lr * min(1, (step + 1) / 100) * 0.5 * (1 + math.cos(math.pi * step / steps))
        x, y = data.batch("train", bsz, seq, g)
        logits, mlogits = mtp_forward(lm, head, x)
        l_main = F.cross_entropy(logits.flatten(0, 1), y.flatten())
        l_mtp = F.cross_entropy(mlogits.flatten(0, 1), y[:, 1:].flatten())  # 目标：下下个字符
        loss = l_main + lam * l_mtp
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if step % 200 == 0:
            print(f"  step {step:4d}  主损失 {l_main.item():.3f}  MTP 损失 {l_mtp.item():.3f}  "
                  f"({time.time() - t0:.0f}s)")
    return lm.eval(), head.eval()


def load_mtp(seed: int):
    c = ch10.Config(vocab_size=ch10.CharData().vocab_size)
    path = OUT / f"mtp_lam0.3_seed{seed}.pt"
    lm, head = ch10.TinyLM(c), MTPHead(c)
    if path.exists():
        sd = torch.load(path, weights_only=True)
        lm.load_state_dict(sd["lm"])
        head.load_state_dict(sd["head"])
        return lm.eval(), head.eval()
    print(f"训练带 MTP 的模型（种子 {seed}，600 步，只需一次）")
    lm, head = train_mtp(seed)
    OUT.mkdir(exist_ok=True)
    torch.save({"lm": lm.state_dict(), "head": head.state_dict()}, path)
    return lm, head


@torch.no_grad()
def evaluate(lm, head=None, n_batches: int = 20, seq: int = 64):
    """验证集：主模型 loss / 下一个字符准确率；有 MTP 时再报 MTP 的"下下个字符"准确率。"""
    data = ch10.CharData()
    g = torch.Generator().manual_seed(1234)  # 与 ch10.val_loss 同样的验证批次
    loss = acc1 = acc2 = 0.0
    for _ in range(n_batches):
        x, y = data.batch("val", 32, seq, g)
        if head is None:
            logits = lm(x)
        else:
            logits, mlogits = mtp_forward(lm, head, x)
            acc2 += (mlogits.argmax(-1) == y[:, 1:]).float().mean().item()
        loss += F.cross_entropy(logits.flatten(0, 1), y.flatten()).item()
        acc1 += (logits.argmax(-1) == y).float().mean().item()
    return loss / n_batches, acc1 / n_batches, acc2 / n_batches


@torch.no_grad()
def greedy(lm, prompt, n_new):
    cache = KVCache(lm.c.n_layers)
    logits = lm(torch.tensor([prompt]), cache)[0, -1]
    out = []
    for _ in range(n_new):
        out.append(int(logits.argmax()))
        logits = lm(torch.tensor([[out[-1]]]), cache)[0, -1]
    return out


@torch.no_grad()
def mtp_self_speculative(lm, head, prompt, n_new):
    """贪心自推测：每轮 MTP 猜 1 个，主模型一次前向验证它并多给 1 个。返回 (新 token, 轮数, 接受数)。"""
    mc, hc = KVCache(lm.c.n_layers), KVCache(1)
    h = hidden(lm, torch.tensor([prompt]), mc)[0]  # prefill
    seq = list(prompt) + [int((h[-1] @ lm.emb.weight.T).argmax())]
    pending, rounds, accepted = h, 0, 0  # pending：还没喂给 MTP 的主模型表示
    while len(seq) - len(prompt) < n_new:
        L = len(seq)
        # MTP 在位置 i 需要 h_i 和 seq[i+1]：补上 len(hc)..L−2，最后一个位置的输出就是草稿
        s = len(hc)
        hm = head(pending[None], lm.emb(torch.tensor([seq[s + 1 : L]])),
                  lm.cos[s : L - 1], lm.sin[s : L - 1], hc)
        d = int((hm[0, -1] @ lm.emb.weight.T).argmax())
        # 主模型一次前向：[seq[L−1], d] → 两个位置的预测
        hn = hidden(lm, torch.tensor([[seq[L - 1], d]]), mc)
        choice = (hn[0] @ lm.emb.weight.T).argmax(-1).tolist()
        ok = choice[0] == d
        seq += [d, choice[1]] if ok else [choice[0]]
        rounds, accepted = rounds + 1, accepted + ok
        truncate(mc, len(seq) - 1)  # 被拒时扔掉草稿的 K/V
        pending = hn[0, : 1 + ok]
    return seq[len(prompt) :][:n_new], rounds, accepted


if __name__ == "__main__":
    m2 = _load("ch25_greedy", "02_greedy_speculative.py")
    print(f"{'模型':<22} {'验证 loss':>9} {'下一个字符准确率':>13} {'MTP 下下个字符准确率':>17}")
    rows = {}
    for seed in (0, 1):
        base = ch10.load_or_train(n_kv_heads=4, steps=600, seed=seed, verbose=True)
        lm, head = load_mtp(seed)
        rows[seed] = (evaluate(base), evaluate(lm, head))
        (lb, ab, _), (lm_, am, a2) = rows[seed]
        print(f"{'λ=0（第 10 章）种子 ' + str(seed):<22} {lb:9.3f} {ab:16.3f} {'—':>20}")
        print(f"{'λ=0.3（加 MTP）种子 ' + str(seed):<22} {lm_:9.3f} {am:16.3f} {a2:20.3f}")
    print("（同一种子的两行：初始化和数据顺序完全相同，只差 MTP 损失。第 10 章发现换一个种子 loss 就差 0.03，"
          "\n  在这个规模上 MTP 对主模型的影响要和这个噪声比。）")

    lm, head = load_mtp(0)
    P, N = m2.prompts(), 200
    ref = [greedy(lm, p, N) for p in P]
    outs = [mtp_self_speculative(lm, head, p, N) for p in P]
    same = all(o[0] == r for o, r in zip(outs, ref))
    rounds = sum(o[1] for o in outs)
    acc = sum(o[2] for o in outs)
    print(f"\nMTP 自推测解码（种子 0，贪心，4 段 × {N} 个字符）：与主模型贪心解码逐字相同：{same}")
    print(f"草稿接受率 {acc / rounds:.3f}，主模型前向 {rounds} 次（普通解码 {len(P) * N} 次），"
          f"每次前向产出 {(len(P) * N) / rounds:.2f} 个字符")
    tb, ts = [], []
    for _ in range(3):
        t0 = time.process_time()
        for p in P:
            greedy(lm, p, N)
        tb.append(time.process_time() - t0)
        t0 = time.process_time()
        for p in P:
            mtp_self_speculative(lm, head, p, N)
        ts.append(time.process_time() - t0)
    tb, ts = statistics.median(tb), statistics.median(ts)
    print(f"CPU 时间：普通贪心 {tb:.2f} s，MTP 自推测 {ts:.2f} s，加速 {tb / ts:.2f}×")
    n_head = sum(p.numel() for p in head.parameters())
    n_lm = sum(p.numel() for p in lm.parameters())
    print(f"MTP 模块参数 {n_head:,}（主模型 {n_lm:,} 的 {n_head / n_lm:.0%}；embedding 与输出头共享，不另算）")
