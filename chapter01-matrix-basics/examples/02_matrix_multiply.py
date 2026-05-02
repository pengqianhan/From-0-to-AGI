"""
第 1 章 示例 2：手写矩阵乘法（不用 NumPy）
目的：真正理解矩阵乘法的计算过程
运行方式：python 02_matrix_multiply.py
"""


def matmul(A, B):
    """
    手写矩阵乘法
    A: m×k 矩阵（列表的列表）
    B: k×n 矩阵
    返回: m×n 矩阵
    """
    m = len(A)      # A 的行数
    k = len(A[0])   # A 的列数（= B 的行数）
    n = len(B[0])   # B 的列数

    # 验证维度是否匹配
    assert len(B) == k, f"维度不匹配：A 有 {k} 列，但 B 有 {len(B)} 行"

    # 初始化结果矩阵（全零）
    C = [[0.0] * n for _ in range(m)]

    # 三重循环：i=行，j=列，p=求和维度
    for i in range(m):
        for j in range(n):
            for p in range(k):
                C[i][j] += A[i][p] * B[p][j]
                # C[i][j] = A 第 i 行 与 B 第 j 列 的点积

    return C


def print_matrix(M, name="矩阵"):
    print(f"{name}:")
    for row in M:
        print("  [" + "  ".join(f"{x:6.2f}" for x in row) + "]")
    print()


# ─── 测试 ────────────────────────────────────────────────────────────────────

print("=" * 50)
print("手写矩阵乘法演示")
print("=" * 50)

A = [[1, 2],
     [3, 4]]

B = [[5, 6],
     [7, 8]]

print_matrix(A, "A (2×2)")
print_matrix(B, "B (2×2)")

C = matmul(A, B)
print_matrix(C, "C = A × B (2×2)")

print("手动验证：")
print(f"  C[0][0] = 1×5 + 2×7 = {1*5 + 2*7} ✓" if C[0][0] == 19 else "  C[0][0] 错误")
print(f"  C[1][1] = 3×6 + 4×8 = {3*6 + 4*8} ✓" if C[1][1] == 50 else "  C[1][1] 错误")

# ─── 更直观的可视化 ──────────────────────────────────────────────────────────

print("\n" + "=" * 50)
print("逐步计算可视化（A 的行 × B 的列）")
print("=" * 50)

A = [[1, 2, 3],
     [4, 5, 6]]    # 2×3

B = [[7,  8],
     [9,  10],
     [11, 12]]     # 3×2

print_matrix(A, "A (2×3)")
print_matrix(B, "B (3×2)")

m, k, n = 2, 3, 2
print("计算过程：")
for i in range(m):
    for j in range(n):
        terms = " + ".join(f"{A[i][p]}×{B[p][j]}" for p in range(k))
        result = sum(A[i][p] * B[p][j] for p in range(k))
        print(f"  C[{i}][{j}] = {terms} = {result}")

C = matmul(A, B)
print()
print_matrix(C, "C = A × B (2×2)")

# ─── 与 NumPy 对比验证 ────────────────────────────────────────────────────────

import numpy as np

print("=" * 50)
print("与 NumPy 对比（验证正确性）")
print("=" * 50)

A_np = np.array(A)
B_np = np.array(B)
C_np = A_np @ B_np  # @ 是 NumPy 的矩阵乘法运算符

C_manual = np.array(C)
print(f"我们的结果:\n{C_manual}\n")
print(f"NumPy 结果:\n{C_np}\n")
print(f"结果一致: {np.allclose(C_manual, C_np)} ✓")
print()
print("说明：在实际深度学习中，我们用 NumPy 或 PyTorch 的矩阵乘法，")
print("      它们底层经过高度优化，在 GPU 上可以极速运行。")
