"""第 17 章 · 极简代码 1：软标签、温度、KD 损失与它的梯度，并与 zero 对拍

1. one-hot 标签 vs 教师的软标签：软标签里有多少"信息"（熵，单位 bit）；
2. 温度 τ：把教师 logits 除以 τ 再 softmax，分布变平，"暗知识"（错误答案之间的相对大小）浮出来；
3. KD 损失 = τ² · KL(p_T^τ ‖ p_S^τ)，手推梯度 ∂L/∂z_S = τ · (p_S^τ − p_T^τ)，用中心差分检验；
4. 与生产级 zero.post.distill.kd_loss（含 top-k 版本）对拍。

运行：uv run python chapters/17-distillation/code/01_soft_labels.py      （约 2 秒）
"""

import sys
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(1)
ROOT = Path(__file__).resolve().parents[3]  # 仓库根目录，用来 import zero

# 一个"下一个词"的小例子：上文是"今天天气很"，词表只有 6 个候选
VOCAB = ["好", "热", "冷", "晴", "猫", "跑"]
TEACHER_LOGITS = np.array([4.0, 3.0, 2.2, 1.8, -2.0, -3.0])  # 教师：好 > 热 > 冷 > 晴 >> 猫、跑
STUDENT_LOGITS = np.array([2.0, 0.5, 1.5, -0.5, 0.0, -1.0])  # 一个还没学好的学生
TEMPS = [0.5, 1.0, 2.0, 4.0]


def softmax(z: np.ndarray, tau: float = 1.0) -> np.ndarray:
    z = z / tau
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def entropy_bits(p: np.ndarray) -> float:
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum()) + 0.0  # +0.0：把 -0.0 变成 0.0


def kd_loss(z_s: np.ndarray, z_t: np.ndarray, tau: float) -> float:
    """τ² · KL(p_T^τ ‖ p_S^τ) = τ² · Σ_v p_T(v) · (log p_T(v) − log p_S(v))。"""
    p_t, p_s = softmax(z_t, tau), softmax(z_s, tau)
    return float(tau * tau * (p_t * (np.log(p_t) - np.log(p_s))).sum())


def kd_grad(z_s: np.ndarray, z_t: np.ndarray, tau: float) -> np.ndarray:
    """∂L/∂z_S = τ² · (1/τ) · (p_S^τ − p_T^τ) = τ · (p_S^τ − p_T^τ)。"""
    return tau * (softmax(z_s, tau) - softmax(z_t, tau))


def numeric_grad(z_s: np.ndarray, z_t: np.ndarray, tau: float, h: float = 1e-5) -> np.ndarray:
    g = np.zeros_like(z_s)
    for k in range(len(z_s)):
        e = np.zeros_like(z_s)
        e[k] = h
        g[k] = (kd_loss(z_s + e, z_t, tau) - kd_loss(z_s - e, z_t, tau)) / (2 * h)
    return g


def main() -> None:
    # ── 1. one-hot vs 软标签 ─────────────────────────────────────────────
    onehot = np.eye(len(VOCAB))[0]
    p_t = softmax(TEACHER_LOGITS)
    print("1) 同一个位置的两种标签（上文：今天天气很 __）")
    print("   词     " + "  ".join(f"{w:>5}" for w in VOCAB))
    print("   one-hot" + "  ".join(f"{x:6.3f}" for x in onehot))
    print("   教师p_T " + "  ".join(f"{x:6.3f}" for x in p_t))
    print(f"   熵：one-hot = {entropy_bits(onehot):.3f} bit，教师软标签 = {entropy_bits(p_t):.3f} bit")
    print("   one-hot 只说'答案是好'；软标签还说'热、冷、晴也说得通，猫、跑不行'。")

    # ── 2. 温度 ─────────────────────────────────────────────────────────
    print("\n2) 温度 τ 让教师分布变平（p_T^τ = softmax(z_T / τ)）")
    print("   τ     " + "  ".join(f"{w:>5}" for w in VOCAB) + "   熵(bit)  p(晴)/p(猫)")
    for tau in TEMPS:
        p = softmax(TEACHER_LOGITS, tau)
        ratio = p[3] / p[4]
        print(f"   {tau:<4} " + "  ".join(f"{x:6.3f}" for x in p) + f"   {entropy_bits(p):6.3f}  {ratio:8.1f}")
    print("   τ 越大，排名靠后的候选拿到的概率越多；但它们之间的相对顺序（晴 > 猫）始终不变。")

    # ── 3. KD 损失与梯度 ────────────────────────────────────────────────
    print("\n3) KD 损失 L = τ²·KL(p_T^τ ‖ p_S^τ)，梯度 ∂L/∂z_S = τ·(p_S^τ − p_T^τ)")
    for tau in [1.0, 2.0]:
        ana = kd_grad(STUDENT_LOGITS, TEACHER_LOGITS, tau)
        num = numeric_grad(STUDENT_LOGITS, TEACHER_LOGITS, tau)
        print(f"   τ={tau}: L = {kd_loss(STUDENT_LOGITS, TEACHER_LOGITS, tau):.4f}")
        print("     解析梯度 " + "  ".join(f"{x:+.4f}" for x in ana))
        print("     数值梯度 " + "  ".join(f"{x:+.4f}" for x in num))
        print(f"     最大差 {np.abs(ana - num).max():.2e}")
    ce_grad = softmax(STUDENT_LOGITS) - onehot
    print("   对照：硬标签交叉熵的梯度 p_S − onehot = " + "  ".join(f"{x:+.4f}" for x in ce_grad))
    print("   硬标签只把'好'往上推、其余一律往下压；KD 按教师的意见分别推、压每一个词。")

    # 为什么乘 τ²：当 τ 很大时 p ≈ 1/V + z/(Vτ)，梯度 ≈ (z_S − z_T)/(V·τ²) · τ²
    print("\n   为什么乘 τ²：不乘时梯度量级随 τ 按 1/τ² 缩小（下表是不乘 τ² 时的梯度范数）")
    for tau in [1.0, 2.0, 4.0, 8.0]:
        g = kd_grad(STUDENT_LOGITS, TEACHER_LOGITS, tau) / (tau * tau)
        print(f"     τ={tau:<4} |∇| = {np.linalg.norm(g):.4f}   × τ² = {np.linalg.norm(g) * tau * tau:.4f}")

    # ── 4. 与 zero 对拍 ────────────────────────────────────────────────
    sys.path.insert(0, str(ROOT))
    from zero.post.distill import kd_loss as zero_kd_loss

    torch.manual_seed(0)
    b, t, v = 2, 5, 11
    zs, zt = torch.randn(b, t, v) * 2, torch.randn(b, t, v) * 2
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.bool)
    print("\n4) 与 zero.post.distill.kd_loss 对拍（随机 logits，2×5 个位置，词表 11，mask 掉 2 个位置）")
    for tau in [1.0, 2.0]:
        ours = np.mean([kd_loss(zs[i, j].numpy(), zt[i, j].numpy(), tau)
                        for i in range(b) for j in range(t) if mask[i, j]])
        prod = float(zero_kd_loss(zs, zt, mask, temperature=tau))
        print(f"   τ={tau}: 极简 {ours:.6f}   zero {prod:.6f}   差 {abs(ours - prod):.1e}")
    # top-k：教师只保留概率最大的 k 个 token 并重新归一化，学生不归一化
    k = 3
    ours_k = []
    for i in range(b):
        for j in range(t):
            if not mask[i, j]:
                continue
            idx = np.argsort(-zt[i, j].numpy())[:k]
            p_t_k = softmax(zt[i, j].numpy()[idx])
            logp_s = np.log(softmax(zs[i, j].numpy()))[idx]
            ours_k.append(float((p_t_k * (np.log(p_t_k) - logp_s)).sum()))
    prod_k = float(zero_kd_loss(zs, zt, mask, topk=k))
    print(f"   top-{k}: 极简 {np.mean(ours_k):.6f}   zero {prod_k:.6f}   差 {abs(np.mean(ours_k) - prod_k):.1e}")


if __name__ == "__main__":
    main()
