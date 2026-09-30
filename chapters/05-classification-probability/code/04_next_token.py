"""第 5 章 · 极简代码 4：语言模型 = 在词表上做分类

"看前一个字，猜下一个字"就是一个分类问题：类别数 = 词表大小。
这里用最简单的模型——一张 V×V 的 logits 表（第 7 章会正式讲这个 bigram 模型）——
和 02 里完全相同的 softmax + 交叉熵 + 梯度 p − onehot 来训练。
然后把损失换算成 bits 和困惑度（perplexity）。
只用 NumPy，CPU 上一两秒跑完。
运行：uv run python chapters/05-classification-probability/code/04_next_token.py
"""

import importlib.util
from pathlib import Path

import numpy as np

_spec = importlib.util.spec_from_file_location("ce_mod", Path(__file__).with_name("02_cross_entropy.py"))
ce = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ce)

TEXT = (
    "我们从一条直线出发。直线拟合不了曲线，所以有了神经网络。"
    "神经网络输出一组分数，把分数变成概率，交叉熵衡量概率给错了多少。"
    "语言模型做的事情也一样：看前面的字，猜下一个字。"
    "猜下一个字，就是在所有的字里做一次分类。"
)


def build_dataset(text: str):
    vocab = sorted(set(text))
    stoi = {c: i for i, c in enumerate(vocab)}
    ids = np.array([stoi[c] for c in text])
    return vocab, stoi, ids[:-1], ids[1:]  # 输入：当前字；标签：下一个字


def train_bigram(x: np.ndarray, y: np.ndarray, V: int, lr: float = 50.0, steps: int = 500):
    """logits = W[x]：每个"当前字"对应一行 V 个分数。等价于 onehot(x) @ W，一个线性分类器。"""
    W = np.zeros((V, V))  # 全 0 → 初始预测是均匀分布，损失 = ln V
    log = []
    for step in range(steps + 1):
        logits = W[x]
        if step in (0, 10, 50, 100, 500):
            log.append((step, ce.cross_entropy(logits, y)))
        g = ce.ce_grad(logits, y)       # (p − onehot) / N
        np.add.at(W, x, -lr * g)        # 把梯度加回到对应的行
    return W, log


def count_mle_loss(x: np.ndarray, y: np.ndarray, V: int) -> float:
    """最大似然的"解析解"：直接数频率，p(下一个字 | 当前字) = 次数 / 总次数。
    梯度下降训练的 bigram 最终会逼近这个损失。"""
    counts = np.zeros((V, V))
    np.add.at(counts, (x, y), 1)
    p = counts / counts.sum(axis=1, keepdims=True)
    return float(-np.log(p[x, y]).mean())


def to_bits_and_ppl(nats: float):
    """nats（自然对数）→ bits（以 2 为底）；困惑度 = e^loss = 2^bits。"""
    return nats / np.log(2), float(np.exp(nats))


if __name__ == "__main__":
    vocab, stoi, x, y = build_dataset(TEXT)
    V = len(vocab)
    print(f"语料 {len(TEXT)} 个字，词表 V = {V} 个不同的字，训练样本 {len(x)} 对（当前字 → 下一个字）")
    print(f"均匀乱猜的损失 = ln V = {np.log(V):.4f} nats\n")

    W, log = train_bigram(x, y, V)
    print("  步数   损失(nats)  损失(bits)   困惑度")
    for step, nats in log:
        bits, ppl = to_bits_and_ppl(nats)
        print(f"{step:>6}   {nats:9.4f}   {bits:9.4f}   {ppl:8.2f}")

    print(f"对照：直接数频率（最大似然的解析解）的损失 = {count_mle_loss(x, y, V):.4f} nats")

    print("\n模型学到的：给定当前字，下一个字的概率（前 3 名）")
    for c in ["神", "下", "分"]:
        p = ce.softmax(W[stoi[c]])
        top = np.argsort(-p)[:3]
        print(f"  「{c}」→ " + "，".join(f"{vocab[i]} {p[i]:.2f}" for i in top))

    print("\n换成真实词表的规模：")
    for name, v in [("本例", V), ("GPT-2 的 BPE 词表", 50257)]:
        print(f"  {name}：V = {v}，均匀乱猜的损失 ln V = {np.log(v):.2f} nats = {np.log2(v):.2f} bits，"
              f"困惑度 = {v}")
