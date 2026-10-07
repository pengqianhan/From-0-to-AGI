"""Chapter 15 · Minimal code 4: annealing. Change the data in the decay stage of WSD, and compare.

Chapter 6 showed "branched decay": from a trunk with a constant learning rate, you can branch off a
short decay at any time and get a "finished" model. Here, 4 branches start from the same branch
point. They change only two things: decay the learning rate or not × change the data mixture or not.

  Data: three "sources": English (Shakespeare), Chinese (classical poetry), and code.
        The model works on bytes (vocabulary 256).
  Trunk: mixture English 0.45 / Chinese 0.45 / code 0.10. Code is the data that is "rare, and we
         want more of the skill", like math in OLMo 2.
  New mixture of the branches: English 0.25 / Chinese 0.25 / code 0.50 (upsample the data of the
         target skill).

Run: uv run python chapters/15-midtraining-long-context/code/04_anneal_mixture.py
     (about 8 min of CPU time on one thread; the wall time is longer on a busy machine. The results
     are cached in code/out/anneal_mixture.pt, and the video reads this file. Add --fresh to run again.)
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

# The names of the sources and branches stay in Chinese. They are keys in the result cache, and the
# video of this chapter reads these keys. NAME_EN gives the English names for the printed output.
NAME_EN = {
    "英文": "English", "中文": "Chinese", "代码": "code", "平均": "mean",
    "A 恒定 + 原配比": "A const + old mix", "B 衰减 + 原配比": "B decay + old mix",
    "C 恒定 + 新配比": "C const + new mix", "D 衰减 + 新配比": "D decay + new mix",
}
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
    """For each sequence, pick a source by the mixture, then take a random piece of that source.
    The mixture can change during training."""

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
    """Validation bits-per-byte of each source (on n fixed pieces)."""
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
    # Trunk: a constant learning rate after warmup (the W + S of WSD)
    trunk_lrs = [PEAK_LR * min(1.0, (s + 1) / WARMUP) for s in range(TRUNK_STEPS)]
    train_steps(model, opt, stream, trunk_lrs)
    trunk_eval = evaluate(model, val_data)
    print(f"trunk {TRUNK_STEPS} steps (constant learning rate), time {time.time() - t0:.0f}s")

    decay = [PEAK_LR * (1 - (s + 1) / BRANCH_STEPS) for s in range(BRANCH_STEPS)]   # linear decay to 0
    const = [PEAK_LR] * BRANCH_STEPS
    branches = {
        # 恒定 = constant, 衰减 = decay, 原配比 = old mixture, 新配比 = new mixture
        "A 恒定 + 原配比": (const, MIX_PRETRAIN),
        "B 衰减 + 原配比": (decay, MIX_PRETRAIN),
        "C 恒定 + 新配比": (const, MIX_ANNEAL),
        "D 衰减 + 新配比": (decay, MIX_ANNEAL),
    }
    out = {}
    for name, (lrs, mix) in branches.items():
        t0 = time.time()
        # Branch from the same point: copy the model, the optimizer state (m, v), and the RNG state of the data stream
        m = copy.deepcopy(model)
        o = torch.optim.AdamW(m.parameters(), lr=PEAK_LR, betas=(0.9, 0.95), weight_decay=0.1)
        o.load_state_dict(copy.deepcopy(opt.state_dict()))
        st = MixedStream(train_data, mix, seed=0)
        st.g.set_state(stream.g.get_state())
        train_steps(m, o, st, lrs)
        out[name] = evaluate(m, val_data)
        print(f"branch {NAME_EN[name]} time {time.time() - t0:.0f}s")
    res = {"trunk": trunk_eval, "branches": out, "trunk_lrs": trunk_lrs, "decay_lrs": decay,
           "trunk_steps": TRUNK_STEPS, "branch_steps": BRANCH_STEPS}
    CACHE.parent.mkdir(exist_ok=True)
    torch.save(res, CACHE)
    return res


def report(r: dict) -> None:
    cols = ["英文", "中文", "代码", "平均"]
    print(f"\nValidation bits-per-byte (lower is better). Trunk {r['trunk_steps']} steps, then {r['branch_steps']} more steps on each branch")
    print(f"   {'':<20}" + "".join(f"{NAME_EN[c]:>8}" for c in cols))
    print(f"   {'trunk: branch point':<20}" + "".join(f"{r['trunk'][c]:>8.3f}" for c in cols))
    for name, ev in r["branches"].items():
        print(f"   {NAME_EN.get(name, name):<20}" + "".join(f"{ev[c]:>8.3f}" for c in cols))
    b = r["branches"]
    print("\nOne effect at a time (how much the bits-per-byte of code decreases):")
    print(f"   decay only (B − A):       {b['A 恒定 + 原配比']['代码'] - b['B 衰减 + 原配比']['代码']:+.3f}")
    print(f"   new data only (C − A):    {b['A 恒定 + 原配比']['代码'] - b['C 恒定 + 新配比']['代码']:+.3f}")
    print(f"   both together (D − A):    {b['A 恒定 + 原配比']['代码'] - b['D 衰减 + 新配比']['代码']:+.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fresh", action="store_true", help="ignore the cache and train again")
    args = ap.parse_args()
    report(run(fresh=args.fresh))
