"""Vocabulary size of the main-line tokenizer: compression (bytes/token) × the cost in
parameters and compute. Also the choice of the pre-tokenization regex.

    uv run python chapters/13-data/code/08_vocab_size.py                      # on the tiny corpus (about 1 min)
    uv run python chapters/13-data/code/08_vocab_size.py --corpus DIR --refs  # the large-corpus measurement in the text

--corpus DIR: the directory contains {zh,en,code}_{train,val}.txt. (The first version in the
    text used about 75 MB of Chinese, English, and code text from some open-source
    repositories. The current corpus comes from 09_build_vocab_corpus.py; see README Section 10.)
--refs: also measure the compression of the Qwen2/Qwen3 and Llama 3 tokenizers on the same
    validation text. This needs models/ggml-vocab-*.gguf from the llama.cpp repository
    (zero/export/gguf.py clones llama.cpp to ~/.cache/zero/llama.cpp).

How we calculate the cost: the shape of the main-line model does not change
(configs/main/pretrain.toml: 28 layers, width 1280, shared embedding). Only the vocabulary
size V changes:
- Parameters: embedding = V × 1280 (input and output share it, so it counts one time). The
  total must not be more than 0.8B.
- Training compute: per token ≈ 6 × N_matmul (this includes the 1280 × V matmul of lm_head)
  + attention.
- What we really want to know is "how much compute it takes to read the same amount of text":
  FLOPs / byte = (FLOPs / token) ÷ (bytes / token).
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))

from tokenizers import Regex, pre_tokenizers  # noqa: E402

from zero.config import load_config  # noqa: E402
from zero.data.pipeline import PRETOKENIZE_PRESETS, train_tokenizer  # noqa: E402
from zero.model import count_params, estimate_flops_per_token  # noqa: E402

# Target mixture of the main line (configs/main/data.toml): English web + math count as
# English 0.55, Chinese 0.30, code 0.15
MIX = {"en": 0.55, "zh": 0.30, "code": 0.15}
LLAMA_CPP = Path.home() / ".cache" / "zero" / "llama.cpp"

# Scripts with combining marks (\p{M}): Devanagari (Hindi), Thai, and decomposed Vietnamese
MARK_SAMPLES = {
    "Hindi": "सभी मनुष्यों को गौरव और अधिकारों के मामले में जन्मजात स्वतन्त्रता प्राप्त है।",
    "Thai": "มนุษย์ทั้งหลายเกิดมามีอิสระและเสมอกันในเกียรติศักดิ์และสิทธิ",
    "Chinese": "人人生而自由，在尊严和权利上一律平等。",
    "English": "All human beings are born free and equal in dignity and rights.",
}


def load_corpus(corpus: Path | None, train_mb: float = 20.0) -> tuple[dict[str, str], dict[str, str]]:
    if corpus is None:  # tiny: the last 10% of each corpus is the validation set
        files = {"en": "shakespeare.txt", "zh": "chinese_poetry.txt", "code": "code.txt"}
        train, val = {}, {}
        for k, f in files.items():
            t = (REPO / "assets" / "tiny_corpus" / f).read_text("utf-8")
            cut = int(len(t) * 0.9)
            train[k], val[k] = t[:cut], t[cut:]
        return train, val
    # Take the tokenizer training text by the target mixture: English 55%, Chinese 30%, code 15%
    # (train_mb MB in total). The mixture sets which language gets the merge rules. If code
    # takes too much, the Chinese words do not get into the vocabulary.
    train = {}
    for k, w in MIX.items():
        t = (corpus / f"{k}_train.txt").read_text("utf-8")
        budget = int(train_mb * 1e6 * w)
        b = t.encode("utf-8")[:budget].decode("utf-8", errors="ignore")
        train[k] = b[: b.rfind("\n\n")] if len(t.encode()) > budget else t
    val = {k: (corpus / f"{k}_val.txt").read_text("utf-8") for k in MIX}
    return train, val


def bytes_per_token(tok, text: str) -> float:  # noqa: ANN001
    return len(text.encode("utf-8")) / max(len(tok.encode(text)), 1)


def cost_row(V: int, bpt: dict[str, float]) -> dict[str, float]:
    cfg = load_config(REPO / "configs" / "main" / "pretrain.toml").model
    cfg = dataclasses.replace(cfg, vocab_size=V)
    p = count_params(cfg)
    fpt = estimate_flops_per_token(cfg, 4096)
    weighted = sum(MIX[k] * bpt[k] for k in MIX) / sum(MIX.values())
    return {"emb": p["embedding"] / 1e6, "total": p["total"] / 1e6, "fpt": fpt,
            "bpt_mix": weighted, "flops_per_byte": fpt / weighted}


def ref_tokenizers() -> dict:
    """Rebuild the reference tokenizers from the vocab-only GGUF files of llama.cpp.

    They split text the same way as the official tokenizer.json. We checked this with the
    .inp/.out test cases that come with llama.cpp.
    """
    sys.path.insert(0, str(LLAMA_CPP / "gguf-py"))
    from gguf import GGUFReader  # noqa: PLC0415
    from tokenizers import Tokenizer, decoders, models, normalizers  # noqa: PLC0415

    llama3 = PRETOKENIZE_PRESETS["qwen2"].replace(r"\p{N}|", r"\p{N}{1,3}|")
    out = {}
    for label, fname, regex, nfc in [
        ("Qwen2/Qwen3", "qwen2", PRETOKENIZE_PRESETS["qwen2"], True),
        ("Llama 3", "llama-bpe", llama3, False),
    ]:
        r = GGUFReader(str(LLAMA_CPP / "models" / f"ggml-vocab-{fname}.gguf"))

        def field(name: str, r=r) -> list[str]:
            f = r.fields[name]
            return [bytes(f.parts[i]).decode("utf-8") for i in f.data]

        toks = field("tokenizer.ggml.tokens")
        tt = r.fields["tokenizer.ggml.token_type"]
        types = [int(tt.parts[i][0]) for i in tt.data]
        vocab = {t: i for i, t in enumerate(toks) if types[i] == 1}
        merges = [tuple(m.split(" ", 1)) for m in field("tokenizer.ggml.merges")]
        tok = Tokenizer(models.BPE(vocab=vocab, merges=merges))
        if nfc:
            tok.normalizer = normalizers.NFC()
        tok.pre_tokenizer = pre_tokenizers.Sequence(
            [pre_tokenizers.Split(Regex(regex), "isolated"),
             pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)])
        tok.decoder = decoders.ByteLevel()

        class Wrap:  # like the zero Tokenizer, show only encode
            def __init__(self, t) -> None:  # noqa: ANN001
                self.t = t

            def encode(self, s: str) -> list[int]:
                return self.t.encode(s, add_special_tokens=False).ids

        out[label] = (len(toks), Wrap(tok))
    return out


def pieces(regex: str, text: str) -> list[str]:
    return [p for p, _ in pre_tokenizers.Split(Regex(regex), "isolated").pre_tokenize_str(text)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, default=None)
    ap.add_argument("--refs", action="store_true")
    ap.add_argument("--sizes", type=int, nargs="*", default=None)
    ap.add_argument("--train-mb", type=float, default=20.0)
    ap.add_argument("--json", type=Path, default=None, help="also save the numbers of the table as JSON (for the video)")
    args = ap.parse_args()
    train, val = load_corpus(args.corpus, args.train_mb)
    sizes = args.sizes or ([1024, 2048, 4096, 8192, 16384] if args.corpus is None
                           else [16384, 32768, 49152, 65536, 98304, 131072, 151936])
    print("Training text: " + ", ".join(f"{k} {len(v.encode()) / 1e6:.1f} MB" for k, v in train.items())
          + "; validation text: " + ", ".join(f"{k} {len(v.encode()) / 1e6:.2f} MB" for k, v in val.items()))

    rows = []
    for V in sizes:
        t0 = time.time()
        tok = train_tokenizer(list(train.values()), V)
        print(f"  Train V={V:,}: {time.time() - t0:.0f}s, actual vocabulary {tok.vocab_size:,}")
        if tok.vocab_size < V:  # too little training text: no more adjacent pairs occur ≥2 times, so the vocabulary cannot reach V
            print(f"    (not enough training text: the vocabulary reached only {tok.vocab_size:,}; this row is not in the table)")
            continue
        bpt = {k: bytes_per_token(tok, v) for k, v in val.items()}
        rows.append((f"zero BPE {V // 1024}K" if V % 1024 == 0 else f"zero BPE {V:,}", V, bpt))
    if args.refs and LLAMA_CPP.exists():
        for label, (V, tok) in ref_tokenizers().items():
            rows.append((label, V, {k: bytes_per_token(tok, v) for k, v in val.items()}))

    base = cost_row(65536, rows[0][2])["fpt"]  # only for normalization; it does not change the comparison
    print(f"\n{'Tokenizer':<16}{'Vocab V':>9}{'EN':>7}{'ZH':>7}{'Code':>7}{'Mix':>7}"
          f"{'embedding':>11}{'Total':>9}{'FLOPs/byte(rel)':>17}")
    for label, V, bpt in rows:
        c = cost_row(V, bpt)
        print(f"{label:<16}{V:>9,}{bpt['en']:>7.2f}{bpt['zh']:>7.2f}{bpt['code']:>7.2f}"
              f"{c['bpt_mix']:>7.2f}{c['emb']:>10.1f}M{c['total']:>8.1f}M"
              f"{c['flops_per_byte'] / base:>17.3f}")
    print("(The unit of FLOPs/byte is the FLOPs per token at V=65,536. Smaller = more text for the same compute)")
    if args.json:
        import json  # noqa: PLC0415

        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(
            [{"label": lb, "V": V, "bpt": bpt, **cost_row(V, bpt), "fpb_rel": cost_row(V, bpt)["flops_per_byte"] / base}
             for lb, V, bpt in rows], ensure_ascii=False, indent=1))

    # Pre-tokenization regex: qwen2 (= Qwen3, the zero default) vs qwen3.5
    q2, q35 = PRETOKENIZE_PRESETS["qwen2"], PRETOKENIZE_PRESETS["qwen3.5"]
    print("\nPre-tokenization regexes: the number of pieces in one sentence")
    for lang, s in MARK_SAMPLES.items():
        a, b = pieces(q2, s), pieces(q35, s)
        print(f"  {lang:<8} qwen2 {len(a):>3} pieces   qwen3.5 {len(b):>3} pieces   e.g. {a[:4]} → {b[:3]}")
    same = {k: "identical" if pieces(q2, v[:200_000]) == pieces(q35, v[:200_000]) else "different"
            for k, v in val.items()}
    print(f"  Pieces from the two regexes on the validation set of this corpus (first 200K characters of each): {same}")
    V = 8192 if args.corpus is None else 65536
    t2 = train_tokenizer(list(train.values()), V, "qwen2")
    t35 = train_tokenizer(list(train.values()), V, "qwen3.5")
    print(f"  The same training with V={V:,}: " + ", ".join(
        f"{k} {bytes_per_token(t2, v):.3f} vs {bytes_per_token(t35, v):.3f}" for k, v in val.items()))


if __name__ == "__main__":
    main()
