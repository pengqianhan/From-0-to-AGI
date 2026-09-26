"""主线分词器的词表大小：压缩率（字节/token）× 参数与算力代价，外加预切分正则的选择。

    uv run python chapters/13-data/code/08_vocab_size.py                      # 在 tiny 语料上（约 1 分钟）
    uv run python chapters/13-data/code/08_vocab_size.py --corpus DIR --refs  # 正文里的大语料测量

--corpus DIR：目录里放 {zh,en,code}_{train,val}.txt（正文用的是从几个开源仓库整理的约 75MB 中英代码文本，
    整理方法见 README 第 8 节）。
--refs：顺便测 Qwen2/Qwen3、Llama 3 分词器在同一份验证文本上的压缩率。需要 llama.cpp 仓库里的
    models/ggml-vocab-*.gguf（zero/export/gguf.py 会把 llama.cpp 克隆到 ~/.cache/zero/llama.cpp）。

代价的算法：主线模型的形状固定（configs/main/pretrain.toml：28 层、宽 1280、共享 embedding），
只换词表大小 V：
- 参数：embedding = V × 1280（输入输出共享，只算一份），总量不能超过 0.8B；
- 训练算力：每个 token ≈ 6 × N_matmul（lm_head 的 1280 × V 矩阵乘也在里面）+ 注意力；
- 我们真正关心的是"读完同样多的文本要花多少算力"：FLOPs / 字节 = (FLOPs / token) ÷ (字节 / token)。
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

# 主线的目标配比（configs/main/data.toml）：英文网页 + 数学算英文 0.55，中文 0.30，代码 0.15
MIX = {"en": 0.55, "zh": 0.30, "code": 0.15}
LLAMA_CPP = Path.home() / ".cache" / "zero" / "llama.cpp"

# 带组合符号（\p{M}）的几种文字：天城文（印地语）、泰文，以及拆开写的越南语
MARK_SAMPLES = {
    "印地语": "सभी मनुष्यों को गौरव और अधिकारों के मामले में जन्मजात स्वतन्त्रता प्राप्त है।",
    "泰语": "มนุษย์ทั้งหลายเกิดมามีอิสระและเสมอกันในเกียรติศักดิ์และสิทธิ",
    "中文": "人人生而自由，在尊严和权利上一律平等。",
    "English": "All human beings are born free and equal in dignity and rights.",
}


def load_corpus(corpus: Path | None) -> tuple[dict[str, str], dict[str, str]]:
    if corpus is None:  # tiny：每份语料最后 10% 做验证
        files = {"en": "shakespeare.txt", "zh": "chinese_poetry.txt", "code": "code.txt"}
        train, val = {}, {}
        for k, f in files.items():
            t = (REPO / "assets" / "tiny_corpus" / f).read_text("utf-8")
            cut = int(len(t) * 0.9)
            train[k], val[k] = t[:cut], t[cut:]
        return train, val
    train = {k: (corpus / f"{k}_train.txt").read_text("utf-8") for k in MIX}
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
    """从 llama.cpp 的 vocab-only GGUF 重建参考分词器（和官方 tokenizer.json 的切分一致，
    用 llama.cpp 自带的 .inp/.out 测试用例核对过）。"""
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

        class Wrap:  # 和 zero 的 Tokenizer 一样只暴露 encode
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
    args = ap.parse_args()
    train, val = load_corpus(args.corpus)
    sizes = args.sizes or ([1024, 2048, 4096, 8192, 16384] if args.corpus is None
                           else [16384, 32768, 49152, 65536, 98304, 131072, 151936])
    print("训练文本：" + "，".join(f"{k} {len(v.encode()) / 1e6:.1f} MB" for k, v in train.items())
          + "；验证文本：" + "，".join(f"{k} {len(v.encode()) / 1e6:.2f} MB" for k, v in val.items()))

    rows = []
    for V in sizes:
        t0 = time.time()
        tok = train_tokenizer(list(train.values()), V)
        bpt = {k: bytes_per_token(tok, v) for k, v in val.items()}
        rows.append((f"zero BPE {V // 1024}K" if V % 1024 == 0 else f"zero BPE {V:,}", V, bpt))
        print(f"  训练 V={V:,}：{time.time() - t0:.0f}s，实际词表 {tok.vocab_size:,}")
    if args.refs and LLAMA_CPP.exists():
        for label, (V, tok) in ref_tokenizers().items():
            rows.append((label, V, {k: bytes_per_token(tok, v) for k, v in val.items()}))

    base = cost_row(65536, rows[0][2])["fpt"]  # 只用来归一化，不影响比较
    print(f"\n{'分词器':<16}{'词表 V':>9}{'英文':>7}{'中文':>7}{'代码':>7}{'加权':>7}"
          f"{'embedding':>11}{'总参数':>9}{'FLOPs/字节(相对)':>17}")
    for label, V, bpt in rows:
        c = cost_row(V, bpt)
        print(f"{label:<16}{V:>9,}{bpt['en']:>7.2f}{bpt['zh']:>7.2f}{bpt['code']:>7.2f}"
              f"{c['bpt_mix']:>7.2f}{c['emb']:>10.1f}M{c['total']:>8.1f}M"
              f"{c['flops_per_byte'] / base:>17.3f}")
    print("（FLOPs/字节 以 V=65,536 时每 token 的 FLOPs 为单位；越小 = 同样算力读到的文本越多）")

    # 预切分正则：qwen2（= Qwen3，zero 默认）vs qwen3.5
    q2, q35 = PRETOKENIZE_PRESETS["qwen2"], PRETOKENIZE_PRESETS["qwen3.5"]
    print("\n预切分正则对比：一句话被切成几个词块")
    for lang, s in MARK_SAMPLES.items():
        a, b = pieces(q2, s), pieces(q35, s)
        print(f"  {lang:<8} qwen2 {len(a):>3} 块   qwen3.5 {len(b):>3} 块   例：{a[:4]} → {b[:3]}")
    same = {k: "完全相同" if pieces(q2, v[:200_000]) == pieces(q35, v[:200_000]) else "有差别"
            for k, v in val.items()}
    print(f"  本语料验证集（每份前 20 万字符）上两种正则切出的词块：{same}")
    V = 8192 if args.corpus is None else 65536
    t2 = train_tokenizer(list(train.values()), V, "qwen2")
    t35 = train_tokenizer(list(train.values()), V, "qwen3.5")
    print(f"  同样训练 V={V:,}：" + "，".join(
        f"{k} {bytes_per_token(t2, v):.3f} vs {bytes_per_token(t35, v):.3f}" for k, v in val.items()))


if __name__ == "__main__":
    main()
