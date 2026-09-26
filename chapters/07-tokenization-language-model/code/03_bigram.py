"""第 7 章 · 极简代码 3：数出来的 bigram 语言模型 + bits-per-byte

语言模型 = 给定前文，预测下一个 token 的概率分布。bigram 只看前 1 个 token：
    p(x_t | x_<t) ≈ p(x_t | x_{t-1}) = (count[x_{t-1}, x_t] + α) / (Σ_j count[x_{t-1}, j] + α·V)
"数次数再除以总数"就是最大似然解（第 5 章 04_next_token.py 已经验证过）；加 α 是平滑，
免得验证集里出现一次训练时没见过的组合，概率就是 0、损失变成无穷大。

评估：交叉熵（nats/token）→ bits/token → 困惑度 → bits-per-byte。
bpb = 总损失(nats) / (ln 2 × 这些 token 覆盖的总字节数)，和分词器无关，可以横向比较。
运行：uv run python chapters/07-tokenization-language-model/code/03_bigram.py
"""

import importlib.util
import math
from pathlib import Path

import numpy as np

_spec = importlib.util.spec_from_file_location("bpe_mod", Path(__file__).with_name("02_bpe.py"))
bpe_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bpe_mod)

ALPHA = 0.03  # 加 α 平滑：在每份语料的 [-40000:-20000] 这段（不参与训练和验证）上从 1、0.3、0.1、0.03、0.01 里挑的


class ByteTok:
    """字节级：id = UTF-8 字节。"""
    V = 256

    def encode(self, s: str) -> list[int]:
        return list(s.encode("utf-8"))

    def token_bytes(self, i: int) -> bytes:
        return bytes([i])


class CharTok:
    """字符级：词表 = 训练集里出现过的字符 + 1 个 <unk>（没见过的字符都映射到它）。"""

    def __init__(self, train_text: str) -> None:
        self.chars = sorted(set(train_text))
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.V = len(self.chars) + 1

    def encode(self, s: str) -> list[int]:
        return [self.stoi.get(c, self.V - 1) for c in s]

    def token_bytes(self, i: int) -> bytes:
        return self.chars[i].encode("utf-8") if i < self.V - 1 else b"?"


class BPETok:
    """第 2 个脚本里手写的 BPE。"""

    def __init__(self, bpe) -> None:
        self.bpe, self.V = bpe, len(bpe.vocab)

    def encode(self, s: str) -> list[int]:
        return self.bpe.encode(s)

    def token_bytes(self, i: int) -> bytes:
        return self.bpe.vocab[i]


def count_bigrams(ids: list[int], V: int) -> np.ndarray:
    a = np.asarray(ids)
    flat = a[:-1] * V + a[1:]                                 # 把 (前一个, 下一个) 编成一个整数
    return np.bincount(flat, minlength=V * V).reshape(V, V).astype(np.float64)  # counts[前, 后]


def bigram_probs(counts: np.ndarray, alpha: float = ALPHA) -> np.ndarray:
    """每一行归一化：P[a, b] = p(下一个是 b | 当前是 a)。整行都是 0（没见过的 a）时退回均匀分布。"""
    c = counts + alpha
    rows = c.sum(axis=1, keepdims=True)
    return np.where(rows > 0, c / np.maximum(rows, 1e-12), 1.0 / len(c))


def evaluate(P: np.ndarray, ids: list[int], n_bytes: list[int]) -> dict:
    """在一段 token 序列上算交叉熵。n_bytes[i] 是第 i 个 token 在原文里占几个字节。"""
    x, y = np.array(ids[:-1]), np.array(ids[1:])
    nats = -np.log(P[x, y])                                   # 每个位置的 −ln p(正确的下一个 token)
    total_nats, total_bytes = nats.sum(), sum(n_bytes[1:])   # 只统计被预测的那些 token
    ce = total_nats / len(y)                                  # nats / token
    return {
        "tokens": len(y),
        "nats": ce,
        "bits": ce / math.log(2),                             # bits / token
        "ppl": math.exp(ce),                                  # 困惑度 = e^nats = 2^bits
        "bpb": total_nats / (math.log(2) * total_bytes),     # bits / byte
    }


def run(tok, train_text: str, val_text: str, n_bytes_fn) -> dict:
    tr, va = tok.encode(train_text), tok.encode(val_text)
    P = bigram_probs(count_bigrams(tr, tok.V))
    return evaluate(P, va, n_bytes_fn(va))


def sample(P: np.ndarray, tok, start: int, n: int, seed: int = 0) -> str:
    rng = np.random.default_rng(seed)
    ids = [start]
    for _ in range(n):
        ids.append(int(rng.choice(len(P), p=P[ids[-1]])))    # 按 p(下一个 | 当前) 抽样
    return b"".join(tok.token_bytes(i) for i in ids).decode("utf-8", errors="replace")


def splits(key: str) -> tuple[str, str]:
    """bigram 训练用整份语料除去最后 40,000 字符；验证用最后 20,000 字符（与 02 相同）。"""
    t = (bpe_mod.CORPUS / bpe_mod.FILES[key]).read_text("utf-8")
    return t[:-40_000], t[-20_000:]


if __name__ == "__main__":
    train, _ = bpe_mod.load_splits()
    bpe = bpe_mod.BPE().train("\n".join(train.values()), vocab_size=1024)  # 分词器只用各 60,000 字符训练

    print(f"bigram 训练：各语料除去最后 40,000 字符；验证：最后 20,000 字符。平滑 α = {ALPHA}\n")
    print(f"{'语料':<5}{'分词':<14}{'V':>6}{'val token数':>12}{'nats/tok':>10}{'bits/tok':>10}"
          f"{'困惑度':>9}{'bpb':>8}")
    for k in ["en", "zh", "code"]:
        tr, va = splits(k)
        char = CharTok(tr)
        n_unk = sum(i == char.V - 1 for i in char.encode(va))
        for name, tok in [("字节", ByteTok()), ("字符", char), ("BPE", BPETok(bpe))]:
            if isinstance(tok, CharTok):  # <unk> 按原字符的真实字节数计
                nb = lambda ids, v=va: [len(c.encode()) for c in v]  # noqa: E731
            else:
                nb = lambda ids, t=tok: [len(t.token_bytes(i)) for i in ids]  # noqa: E731
            r = run(tok, tr, va, nb)
            print(f"{k:<6}{name:<12}{tok.V:>8,}{r['tokens']:>11,}{r['nats']:>10.3f}{r['bits']:>10.3f}"
                  f"{r['ppl']:>9.1f}{r['bpb']:>8.3f}")
        print(f"      （字符级：验证集里有 {n_unk} 个字符训练时没见过，都记成 <unk>）\n")

    print("对照：均匀乱猜一个字节 = 8 bits/byte。\n")

    print("从 bigram 里抽样（只看前 1 个 token；抽样用不平滑的计数）：")
    for k in ["en", "zh"]:
        tr, _ = splits(k)
        for name, tok, n in [("字符", CharTok(tr), 50), ("BPE", BPETok(bpe), 30)]:
            P = bigram_probs(count_bigrams(tok.encode(tr), tok.V), alpha=0.0)
            s = sample(P, tok, tok.encode("\n")[0], n, seed=1)
            print(f"  [{k} · {name}] {s.strip()!r}")
