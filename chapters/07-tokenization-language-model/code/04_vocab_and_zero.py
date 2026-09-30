"""第 7 章 · 极简代码 4：词表大小的取舍，以及和生产级分词器 zero/tokenizer.py 对拍

1. 同一份训练文本、同样 768 次合并：手写 BPE vs zero.tokenizer.train_bpe（HF tokenizers，Rust 实现），
   比较验证集上的压缩率（字节/token）和训练用时；两者都必须能无损还原原文。
2. 用 zero 的训练器扫一遍词表大小：词表越大，序列越短，但 embedding 参数 V × d 越多。
运行：uv run python chapters/07-tokenization-language-model/code/04_vocab_and_zero.py
"""

import importlib.util
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # 让 `import zero` 能找到仓库根目录
from zero.tokenizer import DEFAULT_SPECIAL_TOKENS, IM_START, train_bpe  # noqa: E402

_spec = importlib.util.spec_from_file_location("bpe_mod", Path(__file__).with_name("02_bpe.py"))
bpe_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bpe_mod)

# 真实模型的词表与宽度（来自各模型在 Hugging Face 上的 config.json，见 README 参考文献）
REAL = [
    ("GPT-2 (124M)", 50_257, 768),
    ("Qwen3-0.6B", 151_936, 1024),
    ("Qwen3.5-0.8B", 248_320, 1024),
    ("Llama 3.2 1B", 128_256, 2048),
    ("本课主线（暂定）", 65_536, 1280),
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

    # ── 1. 对拍：手写 BPE vs zero ─────────────────────────────────────────
    t0 = time.time()
    mine = bpe_mod.BPE().train(text, vocab_size=256 + 768)
    t_mine = time.time() - t0
    t0 = time.time()
    prod = train_bpe([text], vocab_size=256 + 768 + n_special)  # zero 的词表里还有 16 个特殊 token
    t_prod = time.time() - t0
    for v in val.values():
        assert mine.decode(mine.encode(v)) == v and prod.decode(prod.encode(v)) == v

    r_mine, r_prod = ratio(mine.encode, val), ratio(prod.encode, val)
    print("同一份训练文本、同样 768 次合并，验证集上的字节/token（越大越省 token）：")
    print(f"  {'':<22}{'en':>7}{'zh':>7}{'code':>7}   训练用时")
    print(f"  {'手写 BPE (02_bpe.py)':<20}" + "".join(f"{r_mine[k]:>7.2f}" for k in val)
          + f"   {t_mine:.2f} 秒")
    print(f"  {'zero/tokenizer.py':<21}" + "".join(f"{r_prod[k]:>7.2f}" for k in val)
          + f"   {t_prod:.2f} 秒")
    print("  两者编码再解码都能逐字节还原验证集原文。")

    s = "x = 2026"
    print(f"\n数字切分：手写 {[mine.vocab[i].decode() for i in mine.encode(s)]}；"
          f"zero {[prod.decode([i]) for i in prod.encode(s)]}")
    chat = f"{IM_START}user\n你好"
    ids = prod.encode(chat)
    print(f"特殊 token：zero 把 {chat!r} 编成 {ids[:3]}…，<|im_start|> 是一个 id = {prod.im_start_id}；"
          f"手写版会把它拆成 {len(mine.encode(IM_START))} 个普通 token")

    # ── 2. 词表大小：序列长度 vs embedding 参数 ─────────────────────────────
    big = {k: (bpe_mod.CORPUS / f).read_text("utf-8")[:-40_000] for k, f in bpe_mod.FILES.items()}
    print("\n词表大小扫描（zero 训练器，训练文本 = 三份语料除去最后 40,000 字符；验证 = 最后 20,000 字符）：")
    print(f"  {'词表 V':>8}{'en':>7}{'zh':>7}{'code':>7}   验证集总 token 数")
    total_bytes = sum(len(v.encode()) for v in val.values())
    for V in [272, 512, 1024, 2048, 4096, 8192, 16384, 32768]:
        tok = train_bpe(list(big.values()), vocab_size=V)
        r = ratio(tok.encode, val)
        n = sum(len(tok.encode(v)) for v in val.values())
        print(f"  {tok.vocab_size:>8,}" + "".join(f"{r[k]:>7.2f}" for k in val) + f"   {n:>10,}")
    print(f"  （验证集共 {total_bytes:,} 字节；V = 272 = 256 字节 + 16 个特殊 token，即不做任何合并）")

    print("\n真实模型的 embedding 参数 = V × d（输入输出共享 embedding 时只算一份）：")
    for name, V, d in REAL:
        print(f"  {name:<16} V = {V:>7,}  d = {d:>4}  →  {V * d / 1e6:6.1f}M")
