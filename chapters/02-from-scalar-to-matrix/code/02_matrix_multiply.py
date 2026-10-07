"""Chapter 2 · Minimal code 2: matrix multiplication by hand (without NumPy)

Purpose: understand each calculation in a matrix multiplication, and the shape rule (m, k) @ (k, n) → (m, n).
Run: uv run python chapters/02-from-scalar-to-matrix/code/02_matrix_multiply.py
"""


def matmul(A, B):
    """Matrix multiplication by hand. A: m×k (a list of lists), B: k×n. Returns m×n."""
    m = len(A)      # number of rows of A
    k = len(A[0])   # number of columns of A (must be equal to the number of rows of B)
    n = len(B[0])   # number of columns of B

    # Shape rule: the inner dimension k must agree
    assert len(B) == k, f"Dimensions do not agree: A has {k} columns, but B has {len(B)} rows"

    C = [[0.0] * n for _ in range(m)]
    # Three nested loops: i = row, j = column, p = the dimension that we sum over
    for i in range(m):
        for j in range(n):
            for p in range(k):
                C[i][j] += A[i][p] * B[p][j]   # C[i][j] = row i of A · column j of B
    return C


def print_matrix(M, name="Matrix"):
    print(f"{name}:")
    for row in M:
        print("  [" + "  ".join(f"{x:6.1f}" for x in row) + "]")
    print()


if __name__ == "__main__":
    import numpy as np

    print("=" * 50)
    print("Matrix multiplication by hand: C[i][j] = row i of A · column j of B")
    print("=" * 50)

    A = [[1, 2, 3],
         [4, 5, 6]]    # (2, 3)
    B = [[7, 8],
         [9, 10],
         [11, 12]]     # (3, 2)
    print_matrix(A, "A (2×3)")
    print_matrix(B, "B (3×2)")

    print("Calculation:")
    for i in range(2):
        for j in range(2):
            terms = " + ".join(f"{A[i][p]}×{B[p][j]}" for p in range(3))
            result = sum(A[i][p] * B[p][j] for p in range(3))
            print(f"  C[{i}][{j}] = {terms} = {result}")

    C = matmul(A, B)
    print()
    print_matrix(C, "C = A @ B (2×2)")

    print("=" * 50)
    print("Compare with NumPy")
    print("=" * 50)
    C_np = np.array(A) @ np.array(B)   # @ is the matrix multiplication operator of NumPy
    print(f"NumPy result:\n{C_np}")
    print(f"Results agree: {np.allclose(np.array(C), C_np)}")

    print("\n" + "=" * 50)
    print("Shape rule: (m, k) @ (k, n) → (m, n)")
    print("=" * 50)
    for sa, sb in [((4, 3), (3, 1)), ((32, 128), (128, 64)), ((3, 2), (3, 5))]:
        try:
            out = (np.zeros(sa) @ np.zeros(sb)).shape
            print(f"  {sa} @ {sb} → {out}")
        except ValueError:
            print(f"  {sa} @ {sb} → error: the inner dimensions {sa[1]} ≠ {sb[0]}")
