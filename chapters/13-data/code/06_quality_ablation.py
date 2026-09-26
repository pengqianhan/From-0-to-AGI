"""数据消融①：同样的模型、同样的步数，只换数据——脏网页原样 vs 走完整条过滤流水线。

两个小模型（约 50 万参数，zero 的 Transformer）各训练 200 步、看同样多的 token，
在同一份留出的干净文本（英文、中文）上比较 bits-per-byte。每种数据用 2 个随机种子各训练一次，
看差距是否大于种子带来的波动。bpb 用本章的公式手算一遍，再和生产级的 zero/data/bpb.py 对拍。

    uv run python chapters/13-data/code/06_quality_ablation.py     # CPU 约 5–8 分钟
"""

from __future__ import annotations

import importlib.util
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

torch.set_num_threads(1)  # 构建环境里多个任务共享 CPU（读者本机可以删掉这行）

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))

from zero.config import ModelConfig  # noqa: E402
from zero.data.bpb import bpb_stats, token_byte_lengths  # noqa: E402
from zero.model import Transformer  # noqa: E402
from zero.tokenizer import Tokenizer, train_bpe  # noqa: E402


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


SEQ, BATCH, STEPS, LR = 128, 16, 200, 3e-3
SEEDS = (0, 1)
MODEL = dict(dim=96, n_layers=3, n_heads=4, n_kv_heads=2, head_dim=24, ffn_dim=256,
             max_seq_len=SEQ, tie_embeddings=False)


def make_tokenizer(texts: list[str], vocab: int = 1024) -> Tokenizer:
    return train_bpe(texts, vocab_size=vocab)


def pack(tok: Tokenizer, docs: list[str]) -> np.ndarray:
    """文档分词后首尾相接，每篇后面加 <|endoftext|>（和 zero/data/shard.py 一样）。"""
    ids: list[int] = []
    for d in docs:
        ids += tok.encode(d) + [tok.eot_id]
    return np.array(ids, dtype=np.int64)


def bpb_by_hand(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor, tb: torch.Tensor) -> float:
    """bpb = Σ(−ln p(目标 token)) / (ln 2 × Σ 目标 token 的字节数)；特殊 token 0 字节，不计入。"""
    with torch.no_grad():
        nats = F.cross_entropy(model(x).flatten(0, 1), y.flatten(), reduction="none")
    nbytes = tb[y.flatten()]
    return float((nats * (nbytes > 0)).sum() / (math.log(2) * nbytes.sum()))


def val_batches(arr: np.ndarray, n_max: int = 48) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """验证集：不重叠地切成长度 SEQ+1 的片段，每批 BATCH 条。"""
    n = min((len(arr) - 1) // SEQ, n_max * BATCH)
    w = np.stack([arr[i * SEQ : i * SEQ + SEQ + 1] for i in range(n)])
    t = torch.from_numpy(w)
    return [(t[i : i + BATCH, :-1], t[i : i + BATCH, 1:]) for i in range(0, n, BATCH)]


def train(
    train_arrays: dict[str, np.ndarray],
    weights: dict[str, float],
    vocab: int,
    seed: int = 0,
    steps: int = STEPS,
    log_every: int = 50,
    tag: str = "",
) -> tuple[Transformer, list[tuple[int, float]]]:
    """按配比逐行抽来源、随机取窗口，训练 steps 步（AdamW + warmup + 余弦衰减）。"""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = Transformer(ModelConfig(vocab_size=vocab, **MODEL))
    opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95), weight_decay=0.1)
    names = list(weights)
    p = np.array([weights[n] for n in names], dtype=float)
    p /= p.sum()
    curve = []
    t0 = time.time()
    for step in range(1, steps + 1):
        lr = LR * min(1.0, step / 20) * (0.55 + 0.45 * math.cos(math.pi * step / steps))
        for g in opt.param_groups:
            g["lr"] = lr
        rows = []
        for src in rng.choice(len(names), size=BATCH, p=p):
            arr = train_arrays[names[src]]
            i = rng.integers(0, len(arr) - SEQ - 1)
            rows.append(arr[i : i + SEQ + 1])
        b = torch.from_numpy(np.stack(rows))
        loss = model.loss(b[:, :-1], b[:, 1:])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        curve.append((step, loss.item()))
        if log_every and step % log_every == 0:
            print(f"  [{tag}] step {step:>4}  loss {loss.item():.3f}  ({time.time() - t0:.0f}s)")
    return model, curve


def evaluate(model: Transformer, val: dict[str, np.ndarray], tb: torch.Tensor) -> dict[str, float]:
    out = {}
    for name, arr in val.items():
        batches = val_batches(arr)
        prod = bpb_stats(model, batches, tb).bpb  # 生产级实现
        x = torch.cat([b[0] for b in batches])
        y = torch.cat([b[1] for b in batches])
        mine = bpb_by_hand(model, x, y, tb)
        assert abs(prod - mine) < 1e-4, (prod, mine)  # 对拍：手算与 zero/data/bpb.py 一致
        out[name] = prod
    return out


def run() -> dict:
    crawl_mod = _load("01_noisy_crawl")
    q = _load("04_quality_classifier")
    dc = _load("05_decontam")
    crawl = crawl_mod.build_crawl()
    raw = crawl["docs"]
    cleaned = q.quality_filter(q.after_dedup())
    hits = dc.find_contaminated(cleaned, crawl["eval"], 13)
    cleaned = [d for i, d in enumerate(cleaned) if i not in hits]

    # 分词器：在过滤后的训练文本上训练（主线也是这样），两个模型共用
    tok = make_tokenizer([d["text"] for d in cleaned])
    tb = token_byte_lengths(tok)
    val = {lang: pack(tok, docs) for lang, docs in crawl["heldout"].items()}
    res: dict = {"tokens": {}, "bpb": {}, "curves": {}}
    for name, docs in [("原样脏网页", raw), ("过滤后", cleaned)]:
        arr = pack(tok, [d["text"] for d in docs])
        res["tokens"][name] = len(arr)
        runs = []
        for seed in SEEDS:
            model, curve = train({"all": arr}, {"all": 1.0}, tok.vocab_size, seed=seed,
                                 tag=f"{name} seed {seed}")
            runs.append(evaluate(model, val, tb))
            if seed == SEEDS[0]:
                res["curves"][name] = curve[::5]
        res["bpb"][name] = {k: [r[k] for r in runs] for k in runs[0]}  # 每个种子一个值
    res["docs"] = {"原样脏网页": len(raw), "过滤后": len(cleaned)}
    return res


def main() -> None:
    res = run()
    print(f"\n{'训练数据':<10}{'文档数':>7}{'token 数':>11}{'英文 bpb（种子 0 / 1）':>24}{'中文 bpb（种子 0 / 1）':>24}")
    for name in res["bpb"]:
        b = res["bpb"][name]
        en = " / ".join(f"{v:.3f}" for v in b["en"])
        zh = " / ".join(f"{v:.3f}" for v in b["zh"])
        print(f"{name:<10}{res['docs'][name]:>7}{res['tokens'][name]:>11,}{en:>24}{zh:>24}")
    a, b = res["bpb"]["原样脏网页"], res["bpb"]["过滤后"]
    d_en = np.mean(a["en"]) - np.mean(b["en"])
    d_zh = np.mean(a["zh"]) - np.mean(b["zh"])
    noise = max(abs(v[0] - v[1]) for r in res["bpb"].values() for v in r.values())
    print(f"\n同样训练 {STEPS} 步 × {BATCH} × {SEQ} = {STEPS * BATCH * SEQ:,} 个 token："
          f"过滤后的数据让英文 bpb 平均低 {d_en:.3f}，中文低 {d_zh:.3f}"
          f"（两个种子之间的最大差别 {noise:.3f}）")


if __name__ == "__main__":
    main()
