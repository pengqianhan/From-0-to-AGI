"""数据消融②：配比。同一堆干净数据（英文、中文、代码），换三种配比各训练一个小模型，
在三个领域的验证集上分别看 bits-per-byte。

真实做法（Llama 3、OLMo 2、Puro-2B、MobileLLM-R1……）也是这样：用小得多的代理模型（proxy）在候选配比上
做对照实验，看各项能力的变化，再决定大模型的配比。这里的代理模型只有约 50 万参数、训练几十万 token。

    uv run python chapters/13-data/code/07_mixture_ablation.py     # CPU 约 5 分钟
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


MIXTURES = {
    "均衡": {"en": 0.45, "zh": 0.45, "code": 0.10},
    "英文为主": {"en": 0.80, "zh": 0.10, "code": 0.10},
    "代码为主": {"en": 0.25, "zh": 0.25, "code": 0.50},
}


def run() -> dict:
    crawl_mod = _load("01_noisy_crawl")
    abl = _load("06_quality_ablation")
    crawl = crawl_mod.build_crawl()
    # 干净的中英文：01 里的"好文档"池（不含留出的验证集）；代码：tiny_corpus 的 code.txt，最后 10% 做验证
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
    res = {"train_tokens": {k: len(v) for k, v in train_arr.items()}, "bpb": {}}
    for name, w in MIXTURES.items():
        model, _ = abl.train(train_arr, w, tok.vocab_size, seed=0, tag=name)
        res["bpb"][name] = abl.evaluate(model, val_arr, tb)
    return res


def main() -> None:
    res = run()
    abl = _load("06_quality_ablation")
    total = abl.STEPS * abl.BATCH * abl.SEQ
    print(f"\n可用的训练 token：{res['train_tokens']}；每个模型训练 {total:,} 个 token")
    print(f"\n{'配比（英/中/代码）':<22}{'英文 bpb':>9}{'中文 bpb':>9}{'代码 bpb':>9}{'按均衡权重的平均':>16}")
    target = MIXTURES["均衡"]
    for name, w in MIXTURES.items():
        b = res["bpb"][name]
        avg = sum(target[k] * b[k] for k in b)
        mix = f"{name} {w['en']:.2f}/{w['zh']:.2f}/{w['code']:.2f}"
        print(f"{mix:<22}{b['en']:>9.3f}{b['zh']:>9.3f}{b['code']:>9.3f}{avg:>16.3f}")
    epochs = {
        name: {k: w[k] * total / res["train_tokens"][k] for k in w} for name, w in MIXTURES.items()
    }
    print("\n每个来源被看了几遍（epoch）：")
    for name, e in epochs.items():
        print(f"  {name}：" + "，".join(f"{k} {v:.2f}" for k, v in e.items()))
    print(f"\n（{np.round(list(res['bpb']['均衡'].values()), 3)} 这样的差别要和随机种子的波动比，"
          "见 06 的两个种子）")


if __name__ == "__main__":
    main()
