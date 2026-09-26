"""第 7 章 · 极简代码 2：手写 byte-level BPE（约 100 行）

BPE（Byte Pair Encoding）的训练只有一句话：**反复把语料里最常见的相邻两个符号合并成一个新符号**。
  1. 预切分：先用正则把文本切成"词块"（字母串、单个数字、标点串、空白），合并不跨越词块边界；
  2. 每个词块变成 UTF-8 字节序列，初始词表就是 256 个字节；
  3. 统计所有相邻 id 对的出现次数，把最多的那一对合并成新 id（256、257、……），重复直到词表够大。
编码新文本时，按学到的先后顺序重放这些合并；解码时把每个 id 对应的字节拼起来再按 UTF-8 解码。
运行：uv run python chapters/07-tokenization-language-model/code/02_bpe.py
"""

import re
import time
from collections import Counter, defaultdict
from pathlib import Path

CORPUS = Path(__file__).resolve().parents[3] / "assets" / "tiny_corpus"
FILES = {"en": "shakespeare.txt", "zh": "chinese_poetry.txt", "code": "code.txt"}

# 预切分正则（GPT-2 / Qwen 思路的简化版，Python re 没有 \p{L}，用 [^\W\d_] 表示"字母"，含汉字）：
# 英文缩写 | 可带一个前导空格的字母串 | 单个数字 | 可带前导空格的标点串（含下划线）| 空白
# 数字逐个切开，和 Qwen 的做法一致
PATTERN = re.compile(r"'(?:s|t|re|ve|m|ll|d)| ?[^\W\d_]+|\d| ?(?:[^\s\w]|_)+|\s+(?!\S)|\s+")


def pretokenize(text: str) -> list[str]:
    return PATTERN.findall(text)


def merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """把 ids 里所有相邻的 pair 替换成 new_id。"""
    out, i = [], 0
    while i < len(ids):
        if i + 1 < len(ids) and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class BPE:
    def __init__(self) -> None:
        self.merges: dict[tuple[int, int], int] = {}          # (a, b) -> 新 id，按学习顺序
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        self.history: list[tuple[int, bytes, int]] = []      # (新 id, 新 token 的字节, 合并时的次数)

    def train(self, text: str, vocab_size: int) -> "BPE":
        words = Counter(pretokenize(text))                   # 相同词块只存一份，记次数
        seqs = [list(w.encode("utf-8")) for w in words]
        freqs = list(words.values())
        stats: Counter = Counter()                           # 相邻对 -> 加权次数
        where = defaultdict(set)                             # 相邻对 -> 包含它的词块编号
        for i, s in enumerate(seqs):
            for p in zip(s, s[1:]):
                stats[p] += freqs[i]
                where[p].add(i)
        for new_id in range(256, vocab_size):
            if not stats:
                break
            pair = max(stats, key=stats.get)                 # 最常见的相邻对
            count = stats[pair]
            if count < 2:
                break
            self.merges[pair] = new_id
            self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]
            self.history.append((new_id, self.vocab[new_id], count))
            for i in where.pop(pair):                        # 只更新受影响的词块
                s, f = seqs[i], freqs[i]
                for p in zip(s, s[1:]):
                    stats[p] -= f
                s = seqs[i] = merge(s, pair, new_id)
                for p in zip(s, s[1:]):
                    stats[p] += f
                    where[p].add(i)
            stats = +stats                                   # 丢掉次数 ≤ 0 的对
        return self

    def _encode_chunk(self, chunk: str) -> list[int]:
        ids = list(chunk.encode("utf-8"))
        while len(ids) >= 2:
            # 在所有相邻对里找"最早学到的"那个合并，先做它（和训练时的顺序一致）
            pair = min(zip(ids, ids[1:]), key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            ids = merge(ids, pair, self.merges[pair])
        return ids

    def encode(self, text: str) -> list[int]:
        cache: dict[str, list[int]] = {}
        out: list[int] = []
        for chunk in pretokenize(text):
            if chunk not in cache:
                cache[chunk] = self._encode_chunk(chunk)
            out.extend(cache[chunk])
        return out

    def decode(self, ids: list[int]) -> str:
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")


def show(b: bytes) -> str:
    """把 token 的字节显示出来：能按 UTF-8 解码就显示文字（空格显示成 ␣），否则显示十六进制。"""
    try:
        return b.decode("utf-8").replace(" ", "␣").replace("\n", "↵")
    except UnicodeDecodeError:
        return "".join(f"\\x{c:02x}" for c in b)


def load_splits(n_train: int = 60_000, n_val: int = 20_000) -> tuple[dict, dict]:
    """每份语料取开头 n_train 个字符训练，结尾 n_val 个字符做验证（互不重叠）。"""
    train, val = {}, {}
    for key, f in FILES.items():
        t = (CORPUS / f).read_text("utf-8")
        train[key], val[key] = t[:n_train], t[-n_val:]
    return train, val


if __name__ == "__main__":
    train, val = load_splits()
    text = "\n".join(train.values())
    print(f"训练文本：英文 + 中文 + 代码各 60,000 字符，共 {len(text.encode()):,} 字节")
    print("预切分示例：", pretokenize("To be, or not to be? 学而时习之。x = 2026\n"))

    t0 = time.time()
    bpe = BPE().train(text, vocab_size=1024)
    print(f"训练 {len(bpe.merges)} 次合并（词表 256 → {len(bpe.vocab)}），用时 {time.time() - t0:.1f} 秒\n")

    print("最先学到的 20 个合并：")
    for new_id, b, count in bpe.history[:20]:
        print(f"  {new_id}: {show(b):<10} 出现 {count:,} 次")
    print("\n后面学到的一些较长 token：")
    longest = sorted(bpe.history, key=lambda h: -len(h[1]))[:10]
    print("  " + "  ".join(show(b) for _, b, _ in longest))

    print("\n编码示例：")
    for s in ["To be, or not to be", "学而时习之，不亦说乎", "def forward(self, x):"]:
        ids = bpe.encode(s)
        assert bpe.decode(ids) == s
        print(f"  {s!r}: {len(s.encode())} 字节 → {len(ids)} 个 token：",
              " | ".join(show(bpe.vocab[i]) for i in ids))

    print("\n验证集（每份语料结尾 20,000 字符，训练时没见过）上的压缩率：")
    print("  语料     字节数   token 数   字节/token")
    for k, v in val.items():
        ids = bpe.encode(v)
        assert bpe.decode(ids) == v                          # 编码再解码必须还原原文
        nb = len(v.encode())
        print(f"  {k:<6} {nb:>8,} {len(ids):>9,}   {nb / len(ids):8.2f}")
