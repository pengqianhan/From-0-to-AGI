"""
第 1 章 示例 3：用矩阵乘法实现神经网络线性层
目的：看清楚 "一层神经网络" 本质上就是一次矩阵乘法
运行方式：python 03_linear_layer.py
"""

import numpy as np

# ─── 场景：预测房价 ───────────────────────────────────────────────────────────
#
# 输入特征（每套房子有 3 个特征）：
#   - 面积（平方米）
#   - 卧室数量
#   - 到市中心距离（公里）
#
# 我们想把这 3 个特征变换成 2 个"隐藏特征"（让网络自己学习有用的中间表示）
# 最终再变成 1 个数：预测价格


print("=" * 60)
print("场景：用神经网络预测房价")
print("=" * 60)

# ─── 1. 准备输入数据 ──────────────────────────────────────────────────────────

# 4 套房子的数据，每套 3 个特征
# 形状：(4, 3)  →  4 个样本，每个样本 3 维特征
X = np.array([
    [80,  2, 5.0],   # 80㎡，2室，距市中心5km
    [120, 3, 2.0],   # 120㎡，3室，距市中心2km
    [60,  1, 8.0],   # 60㎡，1室，距市中心8km
    [100, 3, 3.5],   # 100㎡，3室，距市中心3.5km
], dtype=float)

print(f"\n输入 X（4套房子，3个特征）:")
print(f"  形状: {X.shape}")
feature_names = ["面积(㎡)", "卧室数", "距市中心(km)"]
print(f"  特征: {feature_names}")
print(X)

# ─── 2. 第一层：3维 → 2维 ────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("第一层：线性变换 3维 → 2维")
print("=" * 60)

# 权重矩阵 W1：形状 (3, 2)
# 把 3 个特征压缩成 2 个"隐藏特征"
# 真实训练中这些数字由梯度下降学习得到，这里手动设置演示
rng = np.random.default_rng(42)
W1 = rng.standard_normal((3, 2)) * 0.1   # 形状 (3, 2)
b1 = np.zeros(2)                           # 偏置，形状 (2,)

print(f"\n权重矩阵 W1（形状 {W1.shape}）:")
print(W1.round(4))
print(f"\n偏置 b1（形状 {b1.shape}）: {b1}")

# 前向传播：Y = X @ W1 + b1
H = X @ W1 + b1   # 形状：(4, 3) @ (3, 2) = (4, 2)

print(f"\n计算：H = X @ W1 + b1")
print(f"  X 形状：{X.shape}")
print(f"  W1 形状：{W1.shape}")
print(f"  → H 形状：{H.shape}  （4个样本，每个2个隐藏特征）")
print(f"\nH（隐藏层输出）:")
print(H.round(4))

# ─── 3. 第二层：2维 → 1维（预测价格）───────────────────────────────────────

print("\n" + "=" * 60)
print("第二层：线性变换 2维 → 1维（输出预测价格）")
print("=" * 60)

W2 = rng.standard_normal((2, 1)) * 0.1   # 形状 (2, 1)
b2 = np.zeros(1)

print(f"\n权重矩阵 W2（形状 {W2.shape}）:")
print(W2.round(4))

# 输出层
Y_pred = H @ W2 + b2   # 形状：(4, 2) @ (2, 1) = (4, 1)

print(f"\n计算：Y = H @ W2 + b2")
print(f"  H 形状：{H.shape}")
print(f"  W2 形状：{W2.shape}")
print(f"  → Y 形状：{Y_pred.shape}")
print(f"\n预测价格（万元）:")
for i, price in enumerate(Y_pred.flatten()):
    print(f"  房子 {i+1}: {price:.2f} 万元")

# ─── 4. 完整流程总结 ──────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("完整流程总结")
print("=" * 60)
print("""
输入 X         第一层              隐藏层 H       第二层            输出 Y
(4, 3) ---[@W1 + b1]---> (4, 2) ---[@W2 + b2]---> (4, 1)

关键维度变化：
  输入：4个样本 × 3个特征
  隐藏：4个样本 × 2个隐藏特征（由网络学习最有用的特征组合）
  输出：4个样本 × 1个预测值

每一层都只是：矩阵乘法 + 加法
这就是神经网络的核心计算！
""")

# ─── 5. 用 PyTorch 对比验证 ──────────────────────────────────────────────────

try:
    import torch
    import torch.nn as nn

    print("=" * 60)
    print("PyTorch 版本对比（实际项目中这样写）")
    print("=" * 60)

    model = nn.Sequential(
        nn.Linear(3, 2),   # 3维 → 2维
        nn.Linear(2, 1),   # 2维 → 1维
    )

    X_tensor = torch.tensor(X, dtype=torch.float32)
    Y_torch = model(X_tensor)

    print(f"\n模型结构:\n{model}")
    print(f"\nPyTorch 预测结果形状: {Y_torch.shape}")
    print("（数值不同是因为权重随机初始化不同，但计算结构完全一样）")
    print("\n✓ nn.Linear 内部就是：output = input @ weight.T + bias")

except ImportError:
    print("\n提示：安装 PyTorch 后可运行 PyTorch 对比部分")
    print("      pip install torch")
