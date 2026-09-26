"""第 2 章 · 极简代码 1：向量、点积、矩阵、形状、广播

只用 NumPy，CPU 上瞬间跑完。
运行：uv run python chapters/02-from-scalar-to-matrix/code/01_matrix_basics.py
"""

import numpy as np

# ─── 1. 向量：一套房子的三个特征 ─────────────────────────────────────────────

print("=" * 50)
print("1. 向量与点积")
print("=" * 50)

x = np.array([80.0, 2.0, 5.0])   # 面积(㎡)、卧室数、距市中心(km)
w = np.array([0.8, 5.0, -3.0])   # 每个特征的权重（后面由训练学出来，这里先手写）
b = 20.0

print(f"x = {x}，形状 {x.shape}")  # (3,)
print(f"w = {w}，形状 {w.shape}")

# 点积：对应位置相乘再求和 —— 第 1 章的 a·x 变成了 w₁x₁ + w₂x₂ + w₃x₃
dot_loop = sum(w[i] * x[i] for i in range(3))
dot_np = w @ x                     # 等价于 np.dot(w, x)
print(f"\nw · x（循环）= {dot_loop}")
print(f"w · x（@）   = {dot_np}")
print(f"ŷ = w · x + b = {dot_np + b}")

v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])
print(f"\n[1,2,3] · [4,5,6] = {v1 @ v2}")   # 1×4 + 2×5 + 3×6 = 32

# ─── 2. 矩阵：很多套房子摞在一起 ─────────────────────────────────────────────

print("\n" + "=" * 50)
print("2. 矩阵与形状")
print("=" * 50)

X = np.array([
    [80,  2, 5.0],
    [120, 3, 2.0],
    [60,  1, 8.0],
    [100, 3, 3.5],
])
print(f"X（4 套房子 × 3 个特征）:\n{X}")
print(f"X 的形状: {X.shape}")          # (4, 3) → 4 行 3 列
print(f"X 的转置 Xᵀ 的形状: {X.T.shape}")  # (3, 4)

# ─── 3. 元素级运算 vs 矩阵乘法 ───────────────────────────────────────────────

print("\n" + "=" * 50)
print("3. 元素级运算 vs 矩阵乘法")
print("=" * 50)

M = np.array([[1, 2],
              [3, 4]])
N = np.array([[5, 6],
              [7, 8]])
print(f"M + N（对应位置相加）=\n{M + N}\n")
print(f"M * 2（标量乘法）=\n{M * 2}\n")
print(f"M * N（元素级乘法，Hadamard 乘积）=\n{M * N}\n")
print(f"M @ N（矩阵乘法）=\n{M @ N}")
print("注意：* 和 @ 是完全不同的两件事，混用是最常见的 bug 之一。")

# ─── 4. 广播：把 b 加到每一行 ────────────────────────────────────────────────

print("\n" + "=" * 50)
print("4. 广播（broadcasting）")
print("=" * 50)

scores = X @ w                       # (4, 3) @ (3,) → (4,)
print(f"X @ w = {scores}，形状 {scores.shape}")
print(f"X @ w + b = {scores + b}  ← 标量 b 被加到每一个元素上")

Z = np.zeros((4, 2))
b2 = np.array([10.0, -1.0])          # 形状 (2,)
print(f"\n(4, 2) 的矩阵 + 形状 {b2.shape} 的向量 =\n{Z + b2}")
print("形状 (2,) 的 b 被“复制”到了每一行，但内存里并没有真的复制。")

try:
    Z + np.array([1.0, 2.0, 3.0])    # (4, 2) + (3,)：最后一维 2 ≠ 3
except ValueError as e:
    print(f"\n(4, 2) + (3,) 会报错：{e}")

# ─── 5. 特殊矩阵 ─────────────────────────────────────────────────────────────

print("\n" + "=" * 50)
print("5. 特殊矩阵")
print("=" * 50)
print(f"3×3 单位矩阵:\n{np.eye(3)}")
rng = np.random.default_rng(42)
print(f"\n随机初始化的权重矩阵 W（3×2）:\n{rng.standard_normal((3, 2)).round(3)}")
print("神经网络的权重矩阵就是这样随机初始化的，训练就是不断调整这些数字。")
