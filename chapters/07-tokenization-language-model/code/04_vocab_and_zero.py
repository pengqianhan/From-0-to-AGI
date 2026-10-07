"""Chapter 7 · Minimal code 4: the vocabulary-size trade-off, and a parity check with the production tokenizer zero/tokenizer.py

1. Same training text, same 768 merges: hand-written BPE vs zero.tokenizer.train_bpe (HF tokenizers,
   written in Rust). Compare the compression (bytes/token) on the validation set and the training time.
   Both must give back the original text with no loss.
2. Sweep the vocabulary size with the zero trainer: a larger vocabulary gives shorter sequences,
   but more embedding parameters V × d.
Run: uv run python chapters/07-tokenization-language-model/code/04_vocab_and_zero.py
"""

import importlib.util
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # so that `import zero` finds the repository root
from zero.tokenizer import DEFAULT_SPECIAL_TOKENS, IM_START, train_bpe  # noqa: E402

_spec = importlib.util.spec_from_file_location("bpe_mod", Path(__file__).with_name("02_bpe.py"))
bpe_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bpe_mod)

# Vocabulary size and width of real models (from the config.json of each model on Hugging Face; see the README references)
REAL = [
    ("GPT-2 (124M)", 50_257, 768),
    ("Qwen3-0.6B", 151_936, 1024),
    ("Qwen3.5-0.8B", 248_320, 1024),
    ("Llama 3.2 1B", 128_256, 2048),
    ("Main line (plan)", 65_536, 1280),
]


def ratio(encode, texts: dict[str, str]) -> dict[str, float]:
    out = {}
    for k, v in texts.items():
        ids = encode(v)
        out[k] = len(v.encode("utf-8")) / len(ids)
    return out


if __name__ == "__main__":
    train, val = bpe_mod.load_splits()
    text = "\n".join(train.values())
    n_special = len(DEFAULT_SPECIAL_TOKENS)

    # ── 1. Parity check: hand-written BPE vs zero ─────────────────────────
    t0 = time.time()
    mine = bpe_mod.BPE().train(text, vocab_size=256 + 768)
    t_mine = time.time() - t0
    t0 = time.time()
    prod = train_bpe([text], vocab_size=256 + 768 + n_special)  # the zero vocabulary also has 16 special tokens
    t_prod = time.time() - t0
    for v in val.values():
        assert mine.decode(mine.encode(v)) == v and prod.decode(prod.encode(v)) == v

    r_mine, r_prod = ratio(mine.encode, val), ratio(prod.encode, val)
    print("Same training text, same 768 merges. Bytes/token on the validation set (larger = fewer tokens):")
    print(f" {'':<22}{'en':>7}{'zh':>7}{'code':>7}   training time")
    print(f"  {'hand-written (02_bpe)':<20}" + "".join(f"{r_mine[k]:>7.2f}" for k in val)
          + f"   {t_mine:.2f} s")
    print(f"  {'zero/tokenizer.py':<21}" + "".join(f"{r_prod[k]:>7.2f}" for k in val)
          + f"   {t_prod:.2f} s")
    print("  Both encode and then decode the validation text back to the original, byte for byte.")

    s = "x = 2026"
    print(f"\nDigit split: hand-written {[mine.vocab[i].decode() for i in mine.encode(s)]}; "
          f"zero {[prod.decode([i]) for i in prod.encode(s)]}")
    chat = f"{IM_START}user\n你好"
    ids = prod.encode(chat)
    print(f"Special token: zero encodes {chat!r} as {ids[:3]}…; <|im_start|> is one id = {prod.im_start_id}. "
          f"The hand-written version splits it into {len(mine.encode(IM_START))} normal tokens")

    # ── 2. Vocabulary size: sequence length vs embedding parameters ───────
    big = {k: (bpe_mod.CORPUS / f).read_text("utf-8")[:-40_000] for k, f in bpe_mod.FILES.items()}
    print("\nVocabulary-size sweep (zero trainer; training text = the three corpora without their last 40,000 characters; "
          "validation = the last 20,000 characters):")
    print(f"  {'vocab V':>8}{'en':>7}{'zh':>7}{'code':>7}   val tokens")
    total_bytes = sum(len(v.encode()) for v in val.values())
    for V in [272, 512, 1024, 2048, 4096, 8192, 16384, 32768]:
        tok = train_bpe(list(big.values()), vocab_size=V)
        r = ratio(tok.encode, val)
        n = sum(len(tok.encode(v)) for v in val.values())
        print(f"  {tok.vocab_size:>8,}" + "".join(f"{r[k]:>7.2f}" for k in val) + f"   {n:>10,}")
    print(f"  (the validation set has {total_bytes:,} bytes in total; V = 272 = 256 bytes + 16 special tokens, that is, no merges)")

    print("\nEmbedding parameters of real models = V × d (count one copy when the input and output share the embedding):")
    for name, V, d in REAL:
        print(f"  {name:<16} V = {V:>7,}  d = {d:>4}  →  {V * d / 1e6:6.1f}M")
