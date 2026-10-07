"""Data ablation 2: the mixture. Use the same clean data (English, Chinese, code). Train one
small model for each of three mixtures, and look at bits-per-byte on the validation set of
each of the three domains.

Real work (Llama 3, OLMo 2, Puro-2B, MobileLLM-R1, ...) does the same: much smaller proxy models
run controlled experiments on the candidate mixtures. The teams look at the change in each
ability, and then select the mixture for the large model. Here, the proxy model has only about
500K parameters and trains on a few hundred thousand tokens.

    uv run python chapters/13-data/code/07_mixture_ablation.py     # about 8 min of CPU time on one thread (3 mixtures × 2 seeds)
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# The names stay in Chinese ("balanced", "English-heavy", "code-heavy"): they are keys in the
# JSON that video/scenes.py reads.
MIXTURES = {
    "均衡": {"en": 0.45, "zh": 0.45, "code": 0.10},
    "英文为主": {"en": 0.80, "zh": 0.10, "code": 0.10},
    "代码为主": {"en": 0.25, "zh": 0.25, "code": 0.50},
}
# Display names for the printed output
MIXTURE_EN = {"均衡": "balanced", "英文为主": "English-heavy", "代码为主": "code-heavy"}


def run() -> dict:
    crawl_mod = _load("01_noisy_crawl")
    abl = _load("06_quality_ablation")
    crawl = crawl_mod.build_crawl()
    # Clean English and Chinese: the pool of "good documents" from 01 (without the held-out
    # validation set). Code: code.txt from tiny_corpus; the last 10% is the validation set.
    good = [d for d in crawl["docs"] if d["kind"] == "good"]
    code_docs = crawl_mod.split_docs((REPO / "assets" / "tiny_corpus" / "code.txt").read_text("utf-8"))
    n_val = len(code_docs) // 10
    train_docs = {
        "en": [d["text"] for d in good if d["lang"] == "en"],
        "zh": [d["text"] for d in good if d["lang"] == "zh"],
        "code": code_docs[:-n_val],
    }
    val_docs = {"en": crawl["heldout"]["en"], "zh": crawl["heldout"]["zh"], "code": code_docs[-n_val:]}
    tok = abl.make_tokenizer([t for v in train_docs.values() for t in v])
    tb = abl.token_byte_lengths(tok)
    train_arr = {k: abl.pack(tok, v) for k, v in train_docs.items()}
    val_arr = {k: abl.pack(tok, v) for k, v in val_docs.items()}
    res = {"train_tokens": {k: len(v) for k, v in train_arr.items()}, "bpb": {}, "bpb_seeds": {}}
    for name, w in MIXTURES.items():
        runs = []
        for seed in abl.SEEDS:  # with the same seed, the three mixtures have the same initialization: a paired comparison
            model, _ = abl.train(train_arr, w, tok.vocab_size, seed=seed, tag=f"{MIXTURE_EN[name]} seed {seed}")
            runs.append(abl.evaluate(model, val_arr, tb))
        res["bpb_seeds"][name] = {k: [r[k] for r in runs] for k in runs[0]}
        res["bpb"][name] = {k: sum(r[k] for r in runs) / len(runs) for k in runs[0]}  # mean over the seeds
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
    abl = _load("06_quality_ablation")
    total = abl.STEPS * abl.BATCH * abl.SEQ
    print(f"\nAvailable training tokens: {res['train_tokens']}; each model trains on {total:,} tokens")
    print("\nbpb for each seed (seed 0 / 1):")
    for name, b in res["bpb_seeds"].items():
        print(f"  {MIXTURE_EN.get(name, name)}: " + ", ".join(f"{k} " + " / ".join(f"{v:.3f}" for v in vs) for k, vs in b.items()))
    print(f"\nMean of the two seeds:\n{'Mixture (en/zh/code)':<28}{'EN bpb':>9}{'ZH bpb':>9}{'Code bpb':>9}{'Balanced mean':>16}")
    target = MIXTURES["均衡"]
    for name, w in MIXTURES.items():
        b = res["bpb"][name]
        avg = sum(target[k] * b[k] for k in b)
        mix = f"{MIXTURE_EN[name]} {w['en']:.2f}/{w['zh']:.2f}/{w['code']:.2f}"
        print(f"{mix:<28}{b['en']:>9.3f}{b['zh']:>9.3f}{b['code']:>9.3f}{avg:>16.3f}")
    epochs = {
        name: {k: w[k] * total / res["train_tokens"][k] for k in w} for name, w in MIXTURES.items()
    }
    print("\nEpochs: how many times the model saw each source:")
    for name, e in epochs.items():
        print(f"  {MIXTURE_EN[name]}: " + ", ".join(f"{k} {v:.2f}" for k, v in e.items()))
    print("\nNote: with the same seed, the three mixtures have the same initialization, but each step"
          " samples different windows. Seeds can differ by 0.1 bpb (see 06). Use only the differences"
          " that have the same direction for both seeds")


if __name__ == "__main__":
    main()
