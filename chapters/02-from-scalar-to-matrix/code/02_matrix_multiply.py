"""第 2 章 · 极简代码 2：手写矩阵乘法（不用 NumPy）

目的：真正理解矩阵乘法的计算过程，以及形状规则 (m, k) @ (k, n) → (m, n)。
运行：uv run python chapters/02-from-scalar-to-matrix/code/02_matrix_multiply.py
"""


def matmul(A, B):
    """手写矩阵乘法。A: m×k（列表的列表），B: k×n，返回 m×n。"""
    m = len(A)      # A 的行数
    k = len(A[0])   # A 的列数（必须等于 B 的行数）
    n = len(B[0])   # B 的列数

    # 形状规则：中间的 k 必须对上
    assert len(B) == k, f"维度不匹配：A 有 {k} 列，但 B 有 {len(B)} 行"

    C = [[0.0] * n for _ in range(m)]
    # 三重循环：i = 行，j = 列，p = 求和维度
    for i in range(m):
        for j in range(n):
            for p in range(k):
                C[i][j] += A[i][p] * B[p][j]   # C[i][j] = A 第 i 行 · B 第 j 列
    return C


def print_matrix(M, name="矩阵"):
    print(f"{name}:")
    for row in M:
        print("  [" + "  ".join(f"{x:6.1f}" for x in row) + "]")
    print()


if __name__ == "__main__":
    import numpy as np

    print("=" * 50)
    print("手写矩阵乘法：C[i][j] = A 第 i 行 · B 第 j 列")
    print("=" * 50)

    A = [[1, 2, 3],
         [4, 5, 6]]    # (2, 3)
    B = [[7, 8],
         [9, 10],
         [11, 12]]     # (3, 2)
    print_matrix(A, "A (2×3)")
    print_matrix(B, "B (3×2)")

    print("计算过程：")
    for i in range(2):
        for j in range(2):
            terms = " + ".join(f"{A[i][p]}×{B[p][j]}" for p in range(3))
            result = sum(A[i][p] * B[p][j] for p in range(3))
            print(f"  C[{i}][{j}] = {terms} = {result}")

    C = matmul(A, B)
    print()
    print_matrix(C, "C = A @ B (2×2)")

    print("=" * 50)
    print("与 NumPy 对比")
    print("=" * 50)
    C_np = np.array(A) @ np.array(B)   # @ 是 NumPy 的矩阵乘法运算符
    print(f"NumPy 结果:\n{C_np}")
    print(f"结果一致: {np.allclose(np.array(C), C_np)}")

    print("\n" + "=" * 50)
    print("形状规则：(m, k) @ (k, n) → (m, n)")
    print("=" * 50)
    for sa, sb in [((4, 3), (3, 1)), ((32, 128), (128, 64)), ((3, 2), (3, 5))]:
        try:
            out = (np.zeros(sa) @ np.zeros(sb)).shape
            print(f"  {sa} @ {sb} → {out}")
        except ValueError:
            print(f"  {sa} @ {sb} → 报错：中间的 {sa[1]} ≠ {sb[0]}")
