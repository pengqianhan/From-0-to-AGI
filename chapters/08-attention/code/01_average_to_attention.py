"""第 8 章 · 极简代码 1：从"平均前面的词"到"注意力"

bigram 只看前一个 token。想看更多上下文，最朴素的办法是：把前面所有 token 的向量取平均。
这一步有一个经典技巧：用一个下三角矩阵做一次矩阵乘法，就能同时算出每个位置的"前缀平均"。
再往前一步：把固定的均匀权重换成"由数据决定的权重"（点积相似度 → softmax），就是注意力的雏形。

只用 NumPy，CPU 上瞬间跑完。
运行：uv run python chapters/08-attention/code/01_average_to_attention.py
"""

import numpy as np

np.set_printoptions(precision=2, suppress=True)

TOKENS = ["我", "爱", "吃", "苹", "果"]


def make_x(T: int = 5, C: int = 2, seed: int = 0) -> np.ndarray:
    """T 个 token，每个是一个 C 维向量（第 7 章的 embedding 查表得到的就是这样的向量）。"""
    rng = np.random.default_rng(seed)
    return rng.normal(size=(T, C))


def prefix_mean_loop(x: np.ndarray) -> np.ndarray:
    """最直白的写法：第 t 个位置 = 前 t+1 个向量的平均（包含自己，不含未来）。"""
    out = np.zeros_like(x)
    for t in range(len(x)):
        out[t] = x[: t + 1].mean(axis=0)
    return out


def uniform_weights(T: int) -> np.ndarray:
    """经典技巧：下三角全 1 矩阵，每行除以行和 → 每行是一组"均匀"权重。"""
    w = np.tril(np.ones((T, T)))
    return w / w.sum(axis=1, keepdims=True)


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=-1, keepdims=True)  # 第 5 章的数值稳定写法
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def causal_softmax(scores: np.ndarray) -> np.ndarray:
    """因果 mask：上三角（未来）填 −∞，softmax 之后那些位置的权重恰好是 0。"""
    T = scores.shape[0]
    future = np.triu(np.ones((T, T), dtype=bool), k=1)
    return softmax(np.where(future, -np.inf, scores))


def dot_product_weights(x: np.ndarray) -> np.ndarray:
    """数据决定的权重：位置 t 和位置 s 的"相关程度" = 两个向量的点积（第 2 章），再过 softmax。"""
    scores = x @ x.T  # (T, T)，scores[t, s] = x_t · x_s
    return causal_softmax(scores)


def main() -> None:
    x = make_x()
    T = len(x)
    print("输入：5 个 token，每个是一个 2 维向量 x（形状", x.shape, "）")
    for tok, row in zip(TOKENS, x):
        print(f"  {tok}: {row}")

    # ① 循环版前缀平均
    loop = prefix_mean_loop(x)
    # ② 下三角矩阵乘法版：一次 W @ x 算出所有位置
    w_uni = uniform_weights(T)
    mat = w_uni @ x
    print("\n① 均匀权重矩阵 W（下三角，每行和为 1）：")
    print(w_uni)
    print("② W @ x 与循环版的最大差：", f"{np.abs(mat - loop).max():.1e}")

    # ③ 同一个 W 也可以写成：分数全 0 → mask 掉未来 → softmax
    w_soft = causal_softmax(np.zeros((T, T)))
    print("③ softmax(全 0 分数 + 因果 mask) 与 W 的最大差：", f"{np.abs(w_soft - w_uni).max():.1e}")

    # ④ 把全 0 分数换成点积相似度：权重开始由数据决定
    w_dot = dot_product_weights(x)
    print("\n④ 点积分数 x @ xᵀ：")
    print(x @ x.T)
    print("   因果 mask + softmax 后的权重（每行和为 1）：")
    print(w_dot)
    print("   每行的和：", w_dot.sum(axis=1))
    last = TOKENS[-1]
    top = int(np.argmax(w_dot[-1]))
    print(
        f"   最后一个位置「{last}」最关注「{TOKENS[top]}」，权重 {w_dot[-1, top]:.2f}"
        f"（均匀平均时每个都是 {w_uni[-1, 0]:.2f}）"
    )
    print("   加权平均后的输出 w_dot @ x 的最后一行：", (w_dot @ x)[-1])
    self_top = sum(int(np.argmax(w_dot[t]) == t) for t in range(T))
    print(
        f"   {self_top}/{T} 个位置权重最大的都是自己：x·x = |x|² 往往最大，"
        "所以要用 Q、K 两个不同的投影"
    )


if __name__ == "__main__":
    main()
