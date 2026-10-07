"""Data ablation 1: the same model and the same number of steps; only the data changes.
The raw noisy crawl vs. the output of the full filter pipeline.

Two small models (about 500K parameters, the zero Transformer) train for 200 steps each and see
the same number of tokens. We compare their bits-per-byte on the same held-out clean text
(English, Chinese). Each data set trains with 2 random seeds. Then we can see if the difference
is larger than the variation from the seed. We calculate bpb by hand with the formula of this
chapter, and do a parity check with the production code zero/data/bpb.py.

    uv run python chapters/13-data/code/06_quality_ablation.py     # about 3 min of CPU time on one thread
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

torch.set_num_threads(1)  # many jobs share the CPU in the build environment (you can remove this line on your computer)

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
# Display names for the printed output. The data names in run() stay in Chinese (see there).
DATA_EN = {"原样脏网页": "noisy", "过滤后": "filtered"}
MODEL = dict(dim=96, n_layers=3, n_heads=4, n_kv_heads=2, head_dim=24, ffn_dim=256,
             max_seq_len=SEQ, tie_embeddings=False)


def make_tokenizer(texts: list[str], vocab: int = 1024) -> Tokenizer:
    return train_bpe(texts, vocab_size=vocab)


def pack(tok: Tokenizer, docs: list[str]) -> np.ndarray:
    """Tokenize the documents and join them end to end, with <|endoftext|> after each document
    (the same as zero/data/shard.py)."""
    ids: list[int] = []
    for d in docs:
        ids += tok.encode(d) + [tok.eot_id]
    return np.array(ids, dtype=np.int64)


def bpb_by_hand(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor, tb: torch.Tensor) -> float:
    """bpb = Σ(−ln p(target token)) / (ln 2 × Σ bytes of the target tokens).

    A special token has 0 bytes and does not count.
    """
    with torch.no_grad():
        nats = F.cross_entropy(model(x).flatten(0, 1), y.flatten(), reduction="none")
    nbytes = tb[y.flatten()]
    return float((nats * (nbytes > 0)).sum() / (math.log(2) * nbytes.sum()))


def val_batches(arr: np.ndarray, n_max: int = 48) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Validation set: split into windows of length SEQ+1 with no overlap, BATCH windows per batch."""
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
    """For each row, sample a source by the mixture weights and take a random window. Train for
    `steps` steps (AdamW + warmup + cosine decay)."""
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
        prod = bpb_stats(model, batches, tb).bpb  # production code
        x = torch.cat([b[0] for b in batches])
        y = torch.cat([b[1] for b in batches])
        mine = bpb_by_hand(model, x, y, tb)
        assert abs(prod - mine) < 1e-4, (prod, mine)  # parity check: the bpb by hand equals zero/data/bpb.py
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

    # Tokenizer: train it on the filtered training text (the main line does the same).
    # The two models use the same tokenizer.
    tok = make_tokenizer([d["text"] for d in cleaned])
    tb = token_byte_lengths(tok)
    val = {lang: pack(tok, docs) for lang, docs in crawl["heldout"].items()}
    res: dict = {"tokens": {}, "bpb": {}, "curves": {}}
    # The names stay in Chinese ("raw noisy crawl", "filtered"): they are keys in the JSON that
    # video/scenes.py reads.
    for name, docs in [("原样脏网页", raw), ("过滤后", cleaned)]:
        arr = pack(tok, [d["text"] for d in docs])
        res["tokens"][name] = len(arr)
        runs = []
        for seed in SEEDS:
            model, curve = train({"all": arr}, {"all": 1.0}, tok.vocab_size, seed=seed,
                                 tag=f"{DATA_EN[name]} seed {seed}")
            runs.append(evaluate(model, val, tb))
            if seed == SEEDS[0]:
                res["curves"][name] = curve[::5]
        res["bpb"][name] = {k: [r[k] for r in runs] for k in runs[0]}  # one value per seed
    res["docs"] = {"原样脏网页": len(raw), "过滤后": len(cleaned)}
    return res


def main() -> None:
    import argparse
    import json

    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=Path, default=None, help="also save the results as JSON (for the video; write it under video/out/)")
    args = ap.parse_args()
    res = run()
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(f"\n{'Data':<10}{'Docs':>7}{'Tokens':>11}{'EN bpb (seed 0 / 1)':>24}{'ZH bpb (seed 0 / 1)':>24}")
    for name in res["bpb"]:
        b = res["bpb"][name]
        en = " / ".join(f"{v:.3f}" for v in b["en"])
        zh = " / ".join(f"{v:.3f}" for v in b["zh"])
        print(f"{DATA_EN.get(name, name):<10}{res['docs'][name]:>7}{res['tokens'][name]:>11,}{en:>24}{zh:>24}")
    a, b = res["bpb"]["原样脏网页"], res["bpb"]["过滤后"]
    # Paired comparison: with the same seed, the two data sets have the same initialization and
    # the same sample positions. Thus the difference comes from the data.
    d_en = [x - y for x, y in zip(a["en"], b["en"])]
    d_zh = [x - y for x, y in zip(a["zh"], b["zh"])]
    print(f"\nThe same training of {STEPS} steps × {BATCH} × {SEQ} = {STEPS * BATCH * SEQ:,} tokens. "
          "The filtered data decreases bpb by (paired difference, seed 0 / 1):")
    print(f"  English {d_en[0]:.3f} / {d_en[1]:.3f}, Chinese {d_zh[0]:.3f} / {d_zh[1]:.3f}")
    print(f"  Reference: with the same data and a different seed, bpb differs by up to {max(abs(v[0] - v[1]) for r in res['bpb'].values() for v in r.values()):.3f}."
          " Thus compare only two runs with the same seed (paired). Do not compare results from different seeds")

if __name__ == "__main__":
    main()
