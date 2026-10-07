"""Chapter 2 · Minimal code 1: vectors, dot products, matrices, shapes, and broadcasting

Uses only NumPy. It finishes on the CPU in less than one second.
Run: uv run python chapters/02-from-scalar-to-matrix/code/01_matrix_basics.py
"""

import numpy as np

# ─── 1. Vector: the three features of one house ──────────────────────────────

print("=" * 50)
print("1. Vectors and the dot product")
print("=" * 50)

x = np.array([80.0, 2.0, 5.0])   # area (m²), number of bedrooms, distance to the city center (km)
w = np.array([0.8, 5.0, -3.0])   # one weight for each feature (training learns them later; here we write them by hand)
b = 20.0

print(f"x = {x}, shape {x.shape}")  # (3,)
print(f"w = {w}, shape {w.shape}")

# Dot product: multiply the items at the same position, then add the products.
# The a·x of Chapter 1 becomes w₁x₁ + w₂x₂ + w₃x₃.
dot_loop = sum(w[i] * x[i] for i in range(3))
dot_np = w @ x                     # the same as np.dot(w, x)
print(f"\nw · x (loop) = {dot_loop}")
print(f"w · x (@)    = {dot_np}")
print(f"ŷ = w · x + b = {dot_np + b}")

v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])
print(f"\n[1,2,3] · [4,5,6] = {v1 @ v2}")   # 1×4 + 2×5 + 3×6 = 32

# ─── 2. Matrix: many houses, one on top of the other ─────────────────────────

print("\n" + "=" * 50)
print("2. Matrices and shapes")
print("=" * 50)

X = np.array([
    [80,  2, 5.0],
    [120, 3, 2.0],
    [60,  1, 8.0],
    [100, 3, 3.5],
])
print(f"X (4 houses × 3 features):\n{X}")
print(f"Shape of X: {X.shape}")          # (4, 3) → 4 rows, 3 columns
print(f"Shape of the transpose Xᵀ: {X.T.shape}")  # (3, 4)

# ─── 3. Element-wise operations vs matrix multiplication ─────────────────────

print("\n" + "=" * 50)
print("3. Element-wise operations vs matrix multiplication")
print("=" * 50)

M = np.array([[1, 2],
              [3, 4]])
N = np.array([[5, 6],
              [7, 8]])
print(f"M + N (add the items at the same position) =\n{M + N}\n")
print(f"M * 2 (scalar multiplication) =\n{M * 2}\n")
print(f"M * N (element-wise multiplication, Hadamard product) =\n{M * N}\n")
print(f"M @ N (matrix multiplication) =\n{M @ N}")
print("Note: * and @ are two different operations. Using one in place of the other is one of the most common bugs.")

# ─── 4. Broadcasting: add b to each row ──────────────────────────────────────

print("\n" + "=" * 50)
print("4. Broadcasting")
print("=" * 50)

scores = X @ w                       # (4, 3) @ (3,) → (4,)
print(f"X @ w = {scores}, shape {scores.shape}")
print(f"X @ w + b = {scores + b}  ← NumPy adds the scalar b to each item")

Z = np.zeros((4, 2))
b2 = np.array([10.0, -1.0])          # shape (2,)
print(f"\n(4, 2) matrix + vector of shape {b2.shape} =\n{Z + b2}")
print("NumPy \"copies\" the b of shape (2,) to each row. It does not really make copies in memory.")

try:
    Z + np.array([1.0, 2.0, 3.0])    # (4, 2) + (3,): last dimension 2 ≠ 3
except ValueError as e:
    print(f"\n(4, 2) + (3,) gives an error: {e}")

# ─── 5. Special matrices ─────────────────────────────────────────────────────

print("\n" + "=" * 50)
print("5. Special matrices")
print("=" * 50)
print(f"3×3 identity matrix:\n{np.eye(3)}")
rng = np.random.default_rng(42)
print(f"\nRandomly initialized weight matrix W (3×2):\n{rng.standard_normal((3, 2)).round(3)}")
print("The weight matrices of a neural network start with random values like these. Training changes these numbers again and again.")
