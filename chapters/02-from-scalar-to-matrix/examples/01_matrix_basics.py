"""
第 1 章 示例 1：矩阵基础操作
运行方式：python 01_matrix_basics.py
"""

import numpy as np

# ─── 1. 创建向量 ────────────────────────────────────────────────────────────

print("=" * 50)
print("1. 向量")
print("=" * 50)

v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])

print(f"v1 = {v1}")
print(f"v2 = {v2}")
print(f"v1 的形状: {v1.shape}")  # (3,)

# 向量点积
dot = np.dot(v1, v2)
print(f"\nv1 · v2 = {dot}")
# 手动验证：1×4 + 2×5 + 3×6 = 4 + 10 + 18 = 32

# ─── 2. 创建矩阵 ────────────────────────────────────────────────────────────

print("\n" + "=" * 50)
print("2. 矩阵")
print("=" * 50)

A = np.array([[1, 2, 3],
              [4, 5, 6]])

print(f"矩阵 A:\n{A}")
print(f"A 的形状: {A.shape}")  # (2, 3) → 2行3列

# 矩阵转置
A_T = A.T
print(f"\nA 的转置 Aᵀ:\n{A_T}")
print(f"Aᵀ 的形状: {A_T.shape}")  # (3, 2)

# ─── 3. 矩阵基本运算 ─────────────────────────────────────────────────────────

print("\n" + "=" * 50)
print("3. 矩阵运算")
print("=" * 50)

M = np.array([[1, 2],
              [3, 4]])

N = np.array([[5, 6],
              [7, 8]])

print(f"M =\n{M}\n")
print(f"N =\n{N}\n")

# 加法（对应位置相加）
print(f"M + N =\n{M + N}\n")

# 标量乘法
print(f"M × 2 =\n{M * 2}\n")

# 元素级乘法（Hadamard 乘积，注意不是矩阵乘法）
print(f"M ⊙ N（元素级乘法）=\n{M * N}\n")

# ─── 4. 特殊矩阵 ─────────────────────────────────────────────────────────────

print("=" * 50)
print("4. 特殊矩阵")
print("=" * 50)

# 零矩阵
zeros = np.zeros((3, 3))
print(f"3×3 零矩阵:\n{zeros}\n")

# 单位矩阵（对角线为 1，其余为 0）
identity = np.eye(3)
print(f"3×3 单位矩阵:\n{identity}\n")

# 随机矩阵（神经网络权重初始化就是这样做的）
rng = np.random.default_rng(42)
random_W = rng.standard_normal((4, 3))
print(f"随机初始化的权重矩阵 W（4×3）:\n{random_W.round(3)}")
print(f"\n提示：神经网络的权重矩阵就是这样随机初始化的，")
print(f"      训练的过程就是不断调整这些数字。")
