"""第 15 章 · 极简代码 4：退火——在 WSD 的衰减段换数据，比一比

第 6 章做过"分叉衰减"：一条恒定学习率的主干，随时可以分出一小段衰减得到一个"训练完成"的模型。
这里在同一个分叉点分出 4 条支路，只改两件事：学习率衰不衰减 × 数据配比换不换。

  数据：三个"来源"——英文（莎士比亚）、中文（古诗词）、代码，按字节建模（词表 256）。
  主干：配比 英 0.45 / 中 0.45 / 代码 0.10（代码是"少而想补强"的那一类，好比 OLMo 2 里的数学）。
  支路的新配比：英 0.25 / 中 0.25 / 代码 0.50（把目标能力的数据上采样）。

运行：uv run python chapters/15-midtraining-long-context/code/04_anneal_mixture.py
      （单线程约 8 分钟 CPU 时间，机器繁忙时墙钟更长；结果缓存在 code/out/anneal_mixture.pt，视频直接读它；加 --fresh 重跑）
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
ROOT = HERE.parents[2]
CACHE = HERE / "out" / "anneal_mixture.pt"
_spec = importlib.util.spec_from_file_location(
    "tiny_transformer", HERE.parents[1] / "09-modern-transformer/code/02_tiny_transformer.py")
tt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tt)

SOURCES = {"英文": "shakespeare.txt", "中文": "chinese_poetry.txt", "代码": "code.txt"}
MIX_PRETRAIN = {"英文": 0.45, "中文": 0.45, "代码": 0.10}
MIX_ANNEAL = {"英文": 0.25, "中文": 0.25, "代码": 0.50}
SEQ, BATCH, PEAK_LR, WARMUP = 64, 32, 3e-3, 100
TRUNK_STEPS, BRANCH_STEPS = 800, 200


def load_sources():
    train, val = {}, {}
    for name, fn in SOURCES.items():
        raw = torch.tensor(list((ROOT / "assets/tiny_corpus" / fn).read_bytes()), dtype=torch.long)
        n = int(0.9 * len(raw))
        train[name], val[name] = raw[:n], raw[n:]
    return train, val


class MixedStream:
    """每条序列先按配比抽一个来源，再在该来源里随机取一段。配比可以中途更换。"""

    def __init__(self, data: dict, mix: dict, seed: int):
        self.data, self.g = data, torch.Generator().manual_seed(seed)
        self.set_mix(mix)

    def set_mix(self, mix: dict):
        self.names = list(mix)
        self.probs = torch.tensor([mix[n] for n in self.names])

    def next(self):
        src = torch.multinomial(self.probs, BATCH, replacement=True, generator=self.g)
        xs, ys = [], []
        for s in src.tolist():
            d = self.data[self.names[s]]
            i = torch.randint(len(d) - SEQ - 1, (1,), generator=self.g).item()
            xs.append(d[i:i + SEQ])
            ys.append(d[i + 1:i + SEQ + 1])
        return torch.stack(xs), torch.stack(ys)


def train_steps(model, opt, stream, lrs):
    model.train()
    for lr in lrs:
        for gr in opt.param_groups:
            gr["lr"] = lr
        x, y = stream.next()
        loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten())
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    model.eval()


@torch.no_grad()
def evaluate(model, val: dict, n=96) -> dict:
    """每个来源的验证 bits-per-byte（固定的 n 个片段）。"""
    out = {}
    for name, d in val.items():
        g = torch.Generator().manual_seed(11)
        ix = torch.randint(len(d) - SEQ - 1, (n,), generator=g)
        x = torch.stack([d[i:i + SEQ] for i in ix])
        y = torch.stack([d[i + 1:i + SEQ + 1] for i in ix])
        out[name] = F.cross_entropy(model(x).flatten(0, 1), y.flatten()).item() / math.log(2)
    out["平均"] = sum(out.values()) / len(out)
    return out


def run(fresh: bool = False) -> dict:
    if CACHE.exists() and not fresh:
        return torch.load(CACHE, weights_only=False)
    torch.manual_seed(1337)
    train_data, val_data = load_sources()
    model = tt.TinyTransformer(tt.Config(seq_len=SEQ))
    opt = torch.optim.AdamW(model.parameters(), lr=PEAK_LR, betas=(0.9, 0.95), weight_decay=0.1)
    stream = MixedStream(train_data, MIX_PRETRAIN, seed=3)
    t0 = time.time()
    # 主干：warmup 后恒定学习率（WSD 的 W + S）
    trunk_lrs = [PEAK_LR * min(1.0, (s + 1) / WARMUP) for s in range(TRUNK_STEPS)]
    train_steps(model, opt, stream, trunk_lrs)
    trunk_eval = evaluate(model, val_data)
    print(f"主干 {TRUNK_STEPS} 步（恒定学习率）用时 {time.time() - t0:.0f}s")

    decay = [PEAK_LR * (1 - (s + 1) / BRANCH_STEPS) for s in range(BRANCH_STEPS)]   # 线性降到 0
    const = [PEAK_LR] * BRANCH_STEPS
    branches = {
        "A 恒定 + 原配比": (const, MIX_PRETRAIN),
        "B 衰减 + 原配比": (decay, MIX_PRETRAIN),
        "C 恒定 + 新配比": (const, MIX_ANNEAL),
        "D 衰减 + 新配比": (decay, MIX_ANNEAL),
    }
    out = {}
    for name, (lrs, mix) in branches.items():
        t0 = time.time()
        # 从同一个点分叉：复制模型、优化器状态（m、v）和数据流的随机数状态
        m = copy.deepcopy(model)
        o = torch.optim.AdamW(m.parameters(), lr=PEAK_LR, betas=(0.9, 0.95), weight_decay=0.1)
        o.load_state_dict(copy.deepcopy(opt.state_dict()))
        st = MixedStream(train_data, mix, seed=0)
        st.g.set_state(stream.g.get_state())
        train_steps(m, o, st, lrs)
        out[name] = evaluate(m, val_data)
        print(f"支路 {name} 用时 {time.time() - t0:.0f}s")
    res = {"trunk": trunk_eval, "branches": out, "trunk_lrs": trunk_lrs, "decay_lrs": decay,
           "trunk_steps": TRUNK_STEPS, "branch_steps": BRANCH_STEPS}
    CACHE.parent.mkdir(exist_ok=True)
    torch.save(res, CACHE)
    return res


def report(r: dict) -> None:
    cols = ["英文", "中文", "代码", "平均"]
    print(f"\n验证 bits-per-byte（越低越好）。主干 {r['trunk_steps']} 步，每条支路再训 {r['branch_steps']} 步")
    print(f"   {'':<16}" + "".join(f"{c:>8}" for c in cols))
    print(f"   {'分叉点（主干）':<14}" + "".join(f"{r['trunk'][c]:>8.3f}" for c in cols))
    for name, ev in r["branches"].items():
        print(f"   {name:<14}" + "".join(f"{ev[c]:>8.3f}" for c in cols))
    b = r["branches"]
    print("\n拆开看（代码的 bits-per-byte 下降多少）：")
    print(f"   只衰减（B − A）：{b['A 恒定 + 原配比']['代码'] - b['B 衰减 + 原配比']['代码']:+.3f}")
    print(f"   只换数据（C − A）：{b['A 恒定 + 原配比']['代码'] - b['C 恒定 + 新配比']['代码']:+.3f}")
    print(f"   两者一起（D − A）：{b['A 恒定 + 原配比']['代码'] - b['D 衰减 + 新配比']['代码']:+.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fresh", action="store_true", help="忽略缓存，重新训练")
    args = ap.parse_args()
    report(run(fresh=args.fresh))
