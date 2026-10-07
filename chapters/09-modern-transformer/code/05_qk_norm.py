"""Chapter 9 · Minimal code 5: QK-Norm keeps the attention scores under control, for any length of q and k

Attention score = q·k / √d. During training, the length (norm) of q and k can increase more and more.
Then the scores also increase, and softmax becomes "one-sided": almost all the weight goes to one
position. The gradients become sharp and unstable.
Before the dot product, QK-Norm applies RMSNorm (Chapter 6) to q and k of each head.
This brings the length back to a fixed scale.

Experiment: multiply the same random q, k by s (this simulates a norm that increases during training).
Look at:
  - the largest attention score (logit)
  - the largest weight after softmax, and the entropy of the attention distribution
    (a smaller entropy means a more "one-sided" distribution)
Run: uv run python chapters/09-modern-transformer/code/05_qk_norm.py
"""

import math

import torch

torch.set_num_threads(1)  # many jobs share the CPU of the build machine; on your computer you can remove this line

torch.manual_seed(0)
T, D = 16, 32  # 16 positions, head_dim = 32


def rms_norm(x, eps=1e-6):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


def stats(q, k):
    logits = q @ k.T / math.sqrt(D)                    # (T, T) attention scores
    p = logits.softmax(-1)
    entropy = -(p * p.clamp_min(1e-12).log()).sum(-1).mean()  # entropy of each row, then the mean
    return logits.abs().max().item(), p.max(-1).values.mean().item(), entropy.item()


q0, k0 = torch.randn(T, D), torch.randn(T, D)
print(f"Entropy H of the uniform distribution (no preference) = ln {T} = {math.log(T):.2f}\n")
print(f"{'scale s':>10} | {' no QK-Norm: max score':>20} {'avg max weight':>10} {'H':>6} |"
      f" {'  QK-Norm: max score':>18} {'avg max weight':>10} {'H':>6}")
for s in [1, 2, 4, 8, 16]:
    q, k = q0 * s, k0 * s
    a = stats(q, k)
    b = stats(rms_norm(q), rms_norm(k))
    print(f"{s:>10} | {a[0]:>22.1f} {a[1]:>14.3f} {a[2]:>6.2f} | {b[0]:>20.1f} {b[1]:>14.3f} {b[2]:>6.2f}")
