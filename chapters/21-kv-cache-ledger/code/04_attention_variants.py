"""第 21 章 · 极简代码 4：同一个小模型，换五种注意力，训练同样的步数

MHA（4 个 KV 头）/ GQA（2 个）/ MQA（1 个）/ MLA（潜向量 48 维，缓存和 MQA 一样大）/ MLA（潜向量 16 维）。
除了注意力，其余完全相同：第 10 章的字符级莎士比亚小模型（4 层、宽 128、4 个查询头、head_dim 32），
同样的数据顺序、600 步、AdamW + warmup + cosine。每种跑 3 个随机种子，报告验证 loss 的均值和范围，
以及生成 512 个字符后真实测得的缓存字节数（FP32）。

注意：这是百万参数、几分钟训练的极小实验，只能说明"代码对、趋势大致如何"，
不能推出大模型上的结论（DeepSeek-V2 论文附录 D、GLM-5 报告 2.1 节做的才是认真的对比）。
第一次运行要训练 15 个小模型（单线程约 20–30 分钟），权重缓存在 code/out/*.pt，之后几十秒跑完。
运行：uv run python chapters/21-kv-cache-ledger/code/04_attention_variants.py
"""

from __future__ import annotations

import importlib.util
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

torch.set_num_threads(1)
HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
CH10 = HERE.parents[1] / "10-inference" / "code"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


mla_mod = _load("mla_demo", HERE / "03_mla.py")
tiny = mla_mod.tiny
kvc = _load("kv_cache_demo", CH10 / "03_kv_cache.py")

# 名称, n_kv_heads, MLA 潜向量维度（None = 普通注意力）
VARIANTS = [("MHA", 4, None), ("GQA", 2, None), ("MQA", 1, None),
            ("MLA-48", 4, 48), ("MLA-16", 4, 16)]
SEEDS = (0, 1, 2)
ROPE_DIM = 16
GEN = 512


def build(n_kv: int, rank: int | None) -> tiny.TinyLM:
    c = tiny.Config(vocab_size=tiny.CharData().vocab_size, n_kv_heads=n_kv)
    model = tiny.TinyLM(c)
    if rank is not None:  # 把每层注意力换成 MLA：nope 32 + rope 16 的 key，value 32
        for blk in model.blocks:
            blk.attn = mla_mod.MLA(c.dim, c.n_heads, rank, c.head_dim, ROPE_DIM, c.head_dim,
                                   c.max_seq_len)
    return model


def train(n_kv: int, rank: int | None, seed: int, steps: int = 600, bsz: int = 16,
          seq: int = 64, lr: float = 3e-3):
    """与第 10 章 01_tiny_model.train 完全相同的训练循环，只是模型由 build 构造。"""
    data = tiny.CharData()
    torch.manual_seed(seed)
    model = build(n_kv, rank)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
    g = torch.Generator().manual_seed(seed)  # 同一个种子 → 所有变体看到同样的数据顺序
    for step in range(steps + 1):
        for pg in opt.param_groups:  # warmup + cosine
            pg["lr"] = lr * min(1, (step + 1) / 100) * 0.5 * (1 + math.cos(math.pi * step / steps))
        x, y = data.batch("train", bsz, seq, g)
        loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten())
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    return model.eval()


def load_or_train(name: str, n_kv: int, rank: int | None, seed: int):
    path = OUT / f"{name}_s{seed}.pt"
    if path.exists():
        model = build(n_kv, rank)
        model.load_state_dict(torch.load(path, weights_only=True))
        return model.eval()
    t0 = time.time()
    model = train(n_kv, rank, seed)
    OUT.mkdir(exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"  训练 {name} 种子 {seed}：{time.time() - t0:.0f}s", flush=True)
    return model


def run() -> list[dict]:
    res_path = OUT / "results.pt"  # 结果也存成 .pt（被 .gitignore 忽略），视频直接读
    if res_path.exists():
        return torch.load(res_path, weights_only=False)
    data = tiny.CharData()
    prompt = data.encode(kvc.PROMPT)
    rows = []
    for name, n_kv, rank in VARIANTS:
        losses, models = [], []
        for s in SEEDS:
            m = load_or_train(name, n_kv, rank, s)
            losses.append(tiny.val_loss(m))
            models.append(m)
        m = models[0]
        with torch.no_grad():
            out_c, _, cache = kvc.generate_cached(m, prompt, GEN, temperature=0)
            out_n, _ = kvc.generate_naive(m, prompt, 100, temperature=0)  # 对拍：缓存版 = 朴素版
        attn = sum(p.numel() for n, p in m.named_parameters() if ".attn." in n)
        per_layer = rank + ROPE_DIM if rank else 2 * n_kv * m.c.head_dim
        rows.append(dict(name=name, n_kv=n_kv, rank=rank, losses=losses,
                         mean=sum(losses) / len(losses), per_layer=per_layer,
                         per_token_bytes=per_layer * m.c.n_layers * 4, cache=cache.nbytes(),
                         attn=attn, total=sum(p.numel() for p in m.parameters()),
                         same=out_c[:100] == out_n, sample=data.decode(out_c[:48])))
    torch.save(rows, res_path)
    return rows


if __name__ == "__main__":
    rows = run()
    mha = rows[0]
    print(f"4 层、4 个查询头、head_dim 32，训练 600 步 × 3 个种子；缓存在生成 {GEN} 个字符后测量（FP32）")
    print("  方案     每层每位置  每 token   KV cache 实测        注意力参数   验证 loss 均值"
          "  [三个种子]                 缓存版=朴素版")
    for r in rows:
        ls = " ".join(f"{x:.3f}" for x in r["losses"])
        print(f"  {r['name']:7} {r['per_layer']:6d} 个数  {r['per_token_bytes']:6,d} B  "
              f"{r['cache']:9,d} B ({r['cache'] / mha['cache']:.3f}×)  {r['attn']:9,d}   "
              f"{r['mean']:.3f}  [{ls}]   {r['same']}")
    spread = max(max(r["losses"]) - min(r["losses"]) for r in rows)
    print(f"  同一方案换种子，loss 最多相差 {spread:.3f}；方案之间均值差距小于这个量级时不能当真")
    for r in rows:
        print(f"  {r['name']:7} 贪心生成开头：{r['sample']!r}")
