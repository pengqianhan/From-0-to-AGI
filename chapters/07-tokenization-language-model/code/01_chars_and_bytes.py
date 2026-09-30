"""第 7 章 · 极简代码 1：按字符切，还是按字节切？

模型只认识整数。把文本变成整数序列，最直接的两种办法：
  - 字符级：每个 Unicode 字符一个 id，词表 = 语料里出现过的所有字符；
  - 字节级：先用 UTF-8 编码成字节，每个字节（0–255）一个 id，词表固定 256。
这里在 assets/tiny_corpus 的三份语料上统计两种切法的词表大小和序列长度。
运行：uv run python chapters/07-tokenization-language-model/code/01_chars_and_bytes.py
"""

from pathlib import Path

CORPUS = Path(__file__).resolve().parents[3] / "assets" / "tiny_corpus"
FILES = {"英文（莎士比亚）": "shakespeare.txt", "中文（诗词）": "chinese_poetry.txt", "代码": "code.txt"}


def load(name: str) -> str:
    return (CORPUS / name).read_text("utf-8")


def char_ids(text: str):
    """字符级：词表 = 出现过的字符，按码位排序。"""
    vocab = sorted(set(text))
    stoi = {c: i for i, c in enumerate(vocab)}
    return [stoi[c] for c in text], vocab


def byte_ids(text: str) -> list[int]:
    """字节级：UTF-8 编码后的每个字节就是 id（0–255），不需要任何词表。"""
    return list(text.encode("utf-8"))


if __name__ == "__main__":
    for s in ["A", "é", "学", "🤖"]:
        b = s.encode("utf-8")
        print(f"「{s}」 码位 U+{ord(s):04X}  UTF-8 {len(b)} 个字节：{list(b)}")

    s = "学而时习之"
    print(f"\n「{s}」 字符级 {len(s)} 个 id，字节级 {len(byte_ids(s))} 个 id：{byte_ids(s)}")
    assert bytes(byte_ids(s)).decode("utf-8") == s  # 字节序列可以无损还原

    print("\n语料            字符数      字节数   字符词表  字节词表  字节/字符")
    for label, f in FILES.items():
        text = load(f)
        ids, vocab = char_ids(text)
        nb = len(byte_ids(text))
        print(f"{label:<10} {len(ids):>10,} {nb:>11,} {len(vocab):>8,} {256:>8}  {nb / len(ids):8.2f}")

    # 字符级的麻烦：训练时没见过的字符怎么办？
    zh = load("chinese_poetry.txt")
    cut = int(len(zh) * 0.9)
    seen = set(zh[:cut])
    unseen = [c for c in zh[cut:] if c not in seen]
    print(f"\n中文前 90% 当训练集：后 10% 里有 {len(unseen)} 个字符（{len(set(unseen))} 种）"
          f"在训练集里没出现过，例如 {''.join(sorted(set(unseen))[:10])}")
    print("字节级不存在这个问题：任何文本都只由 256 种字节组成。")
