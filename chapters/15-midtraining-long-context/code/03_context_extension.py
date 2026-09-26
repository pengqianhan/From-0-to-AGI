"""第 15 章 · 极简代码 3：训练长度 64 的小模型，能不能读 128、256？

1. 用第 9 章的极简 Transformer（字节级，head_dim 32，θ = 1 万），在莎士比亚上只用长度 64 的片段训练；
2. 不再训练，直接在长度 64 / 128 / 256 的验证片段上算 loss，四种 RoPE 设置：
     (a) 什么都不改        (b) 位置内插 PI（÷4）
     (c) YaRN（s = 4）     (d) 调大基频（θ = 10 万）
3. 每种设置再用长度 256 的片段微调一小段（150 步，约为预训练的 1/10），重新评测。

所有设置都只改 RoPE 的 cos/sin，不改任何可学参数；评测时同一个模型对所有长度用同一套 cos/sin（"静态缩放"）。
运行：uv run python chapters/15-midtraining-long-context/code/03_context_extension.py
      （单线程约 11 分钟 CPU 时间，机器繁忙时墙钟更长；结果缓存在 code/out/context_extension.pt，视频直接读它；加 --fresh 重跑）
"""

import argparse
import copy
import importlib.util
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

torch.set_num_threads(1)

HERE = Path(__file__).resolve().parent
CACHE = HERE / "out" / "context_extension.pt"
_spec = importlib.util.spec_from_file_location(
    "tiny_transformer", HERE.parents[1] / "09-modern-transformer/code/02_tiny_transformer.py")
tt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tt)
_spec2 = importlib.util.spec_from_file_location("yarn", HERE / "02_yarn_from_scratch.py")
yarn = importlib.util.module_from_spec(_spec2)
_spec2.loader.exec_module(yarn)

TRAIN_LEN, MAX_LEN, S = 64, 256, 4.0
EVAL_LENS = (64, 128, 256)
BUCKETS = ((0, 64), (64, 128), (128, 256))
HEAD_DIM, THETA, ABF_THETA = 32, 10_000.0, 100_000.0


def rope_tables(variant: str) -> tuple[torch.Tensor, torch.Tensor]:
    """返回长度 MAX_LEN 的 cos/sin 表。mscale 直接乘进 cos/sin（与 zero 一样），q·k 因此放大 mscale²。"""
    mscale = 1.0
    if variant == "none":
        w = yarn.rope_inv_freq(HEAD_DIM, THETA)
    elif variant == "pi":
        w = yarn.pi_inv_freq(HEAD_DIM, THETA, S)
    elif variant == "yarn":
        w, _, mscale = yarn.yarn_inv_freq(HEAD_DIM, THETA, S, TRAIN_LEN)
    elif variant == "abf":
        w = yarn.rope_inv_freq(HEAD_DIM, ABF_THETA)
    else:
        raise ValueError(variant)
    angles = torch.outer(torch.arange(MAX_LEN, dtype=torch.float32), w)
    angles = torch.cat([angles, angles], dim=-1)
    return angles.cos() * mscale, angles.sin() * mscale


def set_rope(model, variant: str):
    model.cos, model.sin = rope_tables(variant)
    return model


def batches(data, seq_len, batch_size, steps, seed):
    g = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        ix = torch.randint(len(data) - seq_len - 1, (batch_size,), generator=g)
        x = torch.stack([data[i:i + seq_len] for i in ix])
        y = torch.stack([data[i + 1:i + seq_len + 1] for i in ix])
        yield x, y


def train(model, data, seq_len, batch_size, steps, lr, warmup, seed, log_every=0):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
    curve = []
    model.train()
    for step, (x, y) in enumerate(batches(data, seq_len, batch_size, steps, seed)):
        # warmup + 余弦衰减到 10%（第 6 章）
        f = min(1.0, (step + 1) / warmup) * (0.1 + 0.45 * (1 + math.cos(math.pi * step / steps)))
        for gr in opt.param_groups:
            gr["lr"] = lr * f
        loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten())
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        curve.append(loss.item())
        if log_every and (step + 1) % log_every == 0:
            print(f"   step {step + 1:5d} | train loss {sum(curve[-log_every:]) / log_every:.3f}")
    model.eval()
    return curve


@torch.no_grad()
def per_position_loss(model, val, n_windows=256, seed=7) -> torch.Tensor:
    """在 n_windows 个长度 MAX_LEN 的验证片段上，算每个位置的平均 loss（nat/字节），形状 (MAX_LEN,)。
    因果注意力下，位置 p 的预测只看 [0, p]，所以"长度 L 的 loss"就是前 L 个位置的平均。"""
    g = torch.Generator().manual_seed(seed)
    ix = torch.randint(len(val) - MAX_LEN - 1, (n_windows,), generator=g)
    total = torch.zeros(MAX_LEN)
    for chunk in ix.split(32):
        x = torch.stack([val[i:i + MAX_LEN] for i in chunk])
        y = torch.stack([val[i + 1:i + MAX_LEN + 1] for i in chunk])
        loss = F.cross_entropy(model(x).transpose(1, 2), y, reduction="none")  # (B, T)
        total += loss.sum(0)
    return total / n_windows


def summarize(pos_loss: torch.Tensor) -> dict:
    return {
        "len": {L: pos_loss[:L].mean().item() for L in EVAL_LENS},
        "bucket": {f"{a}-{b}": pos_loss[a:b].mean().item() for a, b in BUCKETS},
        "curve": pos_loss.tolist(),
    }


def run(fresh: bool = False, pre_steps: int = 1500, ft_steps: int = 150) -> dict:
    if CACHE.exists() and not fresh:
        return torch.load(CACHE, weights_only=False)
    torch.manual_seed(1337)
    train_data, val_data = tt.load_data()
    cfg = tt.Config(seq_len=MAX_LEN)            # 预计算到 256 的 cos/sin；训练只用前 64 个位置
    base = tt.TinyTransformer(cfg)
    set_rope(base, "none")
    t0 = time.time()
    print(f"预训练：长度 {TRAIN_LEN}，{pre_steps} 步，batch 32")
    pre_curve = train(base, train_data, TRAIN_LEN, 32, pre_steps, lr=3e-3, warmup=100, seed=1, log_every=300)
    print(f"   用时 {time.time() - t0:.0f}s")

    results = {"zero_shot": {}, "finetuned": {}, "ft_curves": {}}
    variants = ["none", "pi", "yarn", "abf"]
    for v in variants:
        results["zero_shot"][v] = summarize(per_position_loss(set_rope(base, v), val_data))
    for v in variants:
        t0 = time.time()
        m = set_rope(copy.deepcopy(base), v)
        # 所有微调用同一份数据顺序（seed=2），只有 RoPE 设置不同
        results["ft_curves"][v] = train(m, train_data, MAX_LEN, 8, ft_steps, lr=1e-3, warmup=10, seed=2)
        results["finetuned"][v] = summarize(per_position_loss(m, val_data))
        print(f"   微调 {v:<5} 用时 {time.time() - t0:.0f}s")
    set_rope(base, "none")
    results.update(pre_curve=pre_curve, pre_steps=pre_steps, ft_steps=ft_steps,
                   tokens_pre=pre_steps * 32 * TRAIN_LEN, tokens_ft=ft_steps * 8 * MAX_LEN)
    CACHE.parent.mkdir(exist_ok=True)
    torch.save(results, CACHE)
    return results


NAMES = {"none": "(a) 什么都不改", "pi": "(b) 位置内插 PI", "yarn": "(c) YaRN", "abf": "(d) 调大基频 θ=10万"}


def report(r: dict) -> None:
    print(f"\n预训练 {r['tokens_pre']:,} 个 token（长度 {TRAIN_LEN}）；每种设置微调 {r['tokens_ft']:,} 个 token（长度 {MAX_LEN}）")
    for key, title in [("zero_shot", "不训练，直接换 RoPE"), ("finetuned", f"再用长度 {MAX_LEN} 微调 {r['ft_steps']} 步")]:
        print(f"\n{title}：验证 loss（nat/字节，越低越好）")
        print(f"   {'设置':<18}" + "".join(f"{'L=' + str(L):>9}" for L in EVAL_LENS)
              + "".join(f"{'位置' + k:>12}" for k in r[key]["none"]["bucket"]))
        for v, name in NAMES.items():
            s = r[key][v]
            print(f"   {name:<16}" + "".join(f"{s['len'][L]:>9.3f}" for L in EVAL_LENS)
                  + "".join(f"{x:>12.3f}" for x in s["bucket"].values()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fresh", action="store_true", help="忽略缓存，重新训练")
    args = ap.parse_args()
    report(run(fresh=args.fresh))
