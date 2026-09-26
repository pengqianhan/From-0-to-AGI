"""第 25 章 · 极简代码 1：目标模型、草稿模型，以及"验证 k 个 token 和生成 1 个一样贵"

- 目标模型（target）：直接复用第 10 章训练好的小模型（4 层、宽 128、字符级 65 个字符，0.86M 参数）；
- 草稿模型（draft）：同一套结构、**同一个分词器**（同样的 65 个字符），但只有 1 层、宽 64，
  在同样的语料上训练。推测解码要求两者词表完全一致：草稿提出的 token id，目标模型要能直接打分。

然后测一件事：在已有 KV cache 的情况下，目标模型一次前向喂 T 个新 token 要多久？
decode 阶段一次只喂 1 个 token，算力吃不饱（第 10、21 章），所以喂 1 个和喂 5 个的时间差不多——
这就是推测解码能成立的物理前提。

权重缓存在 code/out/*.pt（已被 .gitignore 忽略）；第 10 章的权重不存在时会先训练它。
运行：uv run python chapters/25-mtp-speculative-decoding/code/01_models_and_cost.py
"""

from __future__ import annotations

import importlib.util
import statistics
import sys
import time
from pathlib import Path

import torch

torch.set_num_threads(1)  # 构建机多任务共享 CPU：单线程更稳（读者本机可删掉）
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = HERE / "out"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # dataclass 需要能在 sys.modules 里找到自己的模块
    spec.loader.exec_module(mod)
    return mod


# 第 10 章的 TinyLM / Config / CharData / KVCache / train，原样复用
ch10 = _load("ch10_tiny_model", ROOT / "chapters" / "10-inference" / "code" / "01_tiny_model.py")
KVCache = ch10.KVCache
DRAFT_CFG = dict(dim=64, n_layers=1, n_heads=2, n_kv_heads=2, ffn_dim=192)


def load_target():
    """第 10 章的 4 层小模型（MHA，600 步）。"""
    return ch10.load_or_train(n_kv_heads=4, steps=600, verbose=True)


def load_draft(steps: int = 1500):
    """1 层、宽 64 的草稿模型：同一份字符级数据、同一个词表。"""
    c = ch10.Config(vocab_size=ch10.CharData().vocab_size, **DRAFT_CFG)
    path = OUT / f"draft_1x64_s{steps}.pt"
    model = ch10.TinyLM(c)
    if path.exists():
        model.load_state_dict(torch.load(path, weights_only=True))
        return model.eval()
    print(f"训练草稿模型（{steps} 步，只需一次，之后从 {path.name} 加载）")
    model = ch10.train(c, steps=steps, seed=0, verbose=True)
    OUT.mkdir(exist_ok=True)
    torch.save(model.state_dict(), path)
    return model


def truncate(cache: KVCache, n: int) -> None:
    """KV cache 回滚：只保留前 n 个位置（被拒绝的草稿 token 的 K/V 要扔掉）。"""
    for layer in range(len(cache.k)):
        if cache.k[layer] is not None:
            cache.k[layer] = cache.k[layer][:, :, :n]
            cache.v[layer] = cache.v[layer][:, :, :n]


@torch.no_grad()
def forward_time(model, ctx_len: int, n_new: int, reps: int = 60) -> float:
    """已有 ctx_len 个位置的缓存时，一次前向喂 n_new 个新 token 的耗时（毫秒，取中位数）。

    用 time.process_time()（本进程占用的 CPU 时间）而不是墙钟时间：构建机上几十个任务抢 4 个核，
    墙钟时间主要反映"排队等 CPU"，CPU 时间才反映这次前向本身的工作量（单线程时两者本应相同）。"""
    g = torch.Generator().manual_seed(0)
    cache = KVCache(model.c.n_layers)
    model(torch.randint(0, model.c.vocab_size, (1, ctx_len), generator=g), cache)
    new = torch.randint(0, model.c.vocab_size, (1, n_new), generator=g)
    times = []
    for _ in range(reps):
        t0 = time.process_time()
        model(new, cache)
        times.append(time.process_time() - t0)
        truncate(cache, ctx_len)  # 每次都回到同样的起点
    return statistics.median(times) * 1e3


def n_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    target, draft = load_target(), load_draft()
    data = ch10.CharData()
    print(f"目标模型：{n_params(target):,} 参数，验证集 loss {ch10.val_loss(target):.3f}")
    print(f"草稿模型：{n_params(draft):,} 参数，验证集 loss {ch10.val_loss(draft):.3f}")
    print(
        f"参数量之比 {n_params(target) / n_params(draft):.0f} : 1，共用同一个 {data.vocab_size} 字符的词表"
    )

    print(
        "\n已有 200 个位置的 KV cache，一次前向喂 T 个新 token 的耗时（单线程，进程 CPU 时间，中位数）："
    )
    print(f"{'T':>4} {'目标模型 ms':>12} {'相对 T=1':>9} {'草稿模型 ms':>12}")
    base = None
    for T in (1, 2, 3, 5, 9, 17):
        # 交替测量，减轻机器负载波动对比较的影响
        tt = forward_time(target, 200, T)
        td = forward_time(draft, 200, T)
        base = base or tt
        print(f"{T:4d} {tt:12.2f} {tt / base:8.2f}× {td:12.2f}")
    c = forward_time(draft, 200, 1) / forward_time(target, 200, 1)
    print(f"\n成本系数 c = 草稿一步 / 目标一步 ≈ {c:.2f}")
    print("（这台机器被多个任务共享，计时每次都会波动；看数量级和趋势即可。）")
