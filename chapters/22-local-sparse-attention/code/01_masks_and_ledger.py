"""Chapter 22 · Minimal code 1: three masks, the receptive field, and the KV cache that a sliding window saves.

Three parts. None of them needs training:
  1. Draw three attention masks: full causal, sliding window (W=4), and local-global interleaving.
     Count how many query-key pairs each mask must calculate.
  2. Receptive field: with sliding windows only, each layer moves information back by at most W-1 more
     positions. One global layer reaches the start in one step.
  3. KV cache ledger: use the real configurations of some public models (config.json, read in 2026-09).
     Calculate the memory at a 128K context for full attention and for the real local-global configuration.

Run: uv run python chapters/22-local-sparse-attention/code/01_masks_and_ledger.py
"""

from __future__ import annotations

import torch

torch.set_num_threads(1)


# ── 1. Masks ───────────────────────────────────────────────────────────────
def causal_mask(T: int) -> torch.Tensor:
    i = torch.arange(T)[:, None]
    j = torch.arange(T)[None, :]
    return j <= i  # see only itself and the past


def sliding_mask(T: int, W: int) -> torch.Tensor:
    i = torch.arange(T)[:, None]
    j = torch.arange(T)[None, :]
    return (j <= i) & (i - j < W)  # see only the last W positions (itself included)


def show(mask: torch.Tensor, title: str) -> None:
    print(f"{title} (■ = visible, {int(mask.sum())} pairs)")
    for row in mask.tolist():
        print("  " + " ".join("■" if v else "·" for v in row))


# ── 2. Receptive field: after L layers, how far back the last token gets information ──
def receptive_field(masks: list[torch.Tensor]) -> list[int]:
    """reach[i, j] = can information flow from position j to position i through these layers (product of boolean matrices)."""
    T = masks[0].shape[0]
    reach = torch.eye(T, dtype=torch.bool)
    out = []
    for m in masks:
        reach = (m.float() @ reach.float()) > 0  # in this layer, i sees k, and k already collected j
        far = int(torch.nonzero(reach[-1])[0])  # the earliest position that the last token can "see"
        out.append(T - 1 - far)  # the largest distance
    return out


# ── 3. KV cache ledger ─────────────────────────────────────────────────────
def kv_bytes(n_full: int, n_sliding: int, W: int, T: int, kv_heads: int, head_dim: int) -> int:
    """2 (K and V) × positions stored per layer × KV heads × head_dim × 2 bytes (BF16)."""
    per_pos = 2 * kv_heads * head_dim * 2
    return (n_full * T + n_sliding * min(W, T)) * per_pos


# The numbers come from the config.json of each model (the links are in the README, "Adopters and sources").
MODELS = [
    # name, layers, global layers among them, window, KV heads, head_dim
    ("Mistral-7B-v0.1（全部滑动）", 32, 0, 4096, 8, 128),
    ("Gemma-3-27B（5 局部:1 全局）", 62, 10, 1024, 16, 128),
    ("gpt-oss-120b（1:1 交替）", 36, 18, 128, 8, 64),
    ("OLMo-3-7B（3 局部:1 全局）", 32, 8, 4096, 32, 128),
]
# Display names for the printed output. The names in MODELS stay in Chinese, because video/scenes.py
# uses them (it shows the part before "（").
MODEL_EN = {
    "Mistral-7B-v0.1（全部滑动）": "Mistral-7B-v0.1 (all sliding)",
    "Gemma-3-27B（5 局部:1 全局）": "Gemma-3-27B (5 local:1 global)",
    "gpt-oss-120b（1:1 交替）": "gpt-oss-120b (1:1 alternating)",
    "OLMo-3-7B（3 局部:1 全局）": "OLMo-3-7B (3 local:1 global)",
}


def main() -> None:
    T, W = 10, 4
    full = causal_mask(T)
    swa = sliding_mask(T, W)
    show(full, f"Full causal attention T={T}")
    show(swa, f"Sliding window W={W}")
    for T_big in (1024, 32768):
        n_full = T_big * (T_big + 1) // 2
        n_swa = int(sliding_mask(T_big, W).sum()) if T_big <= 4096 else W * T_big - W * (W - 1) // 2
        print(
            f"T={T_big:>6}: full causal {n_full:>12,} pairs, sliding window W={W} {n_swa:>9,} pairs"
            f" ({n_full / n_swa:,.0f}×)"
        )

    print("\nReceptive field: after layer l, the largest distance from which information reaches the last token (T=64, W=4)")
    T = 64
    L = 6
    all_swa = [sliding_mask(T, W)] * L
    interleave = [sliding_mask(T, W) if (l + 1) % 3 else causal_mask(T) for l in range(L)]
    print("  layer              :", " ".join(f"{l + 1:>3}" for l in range(L)))
    print("  all sliding window :", " ".join(f"{d:>3}" for d in receptive_field(all_swa)))
    print("  2 local + 1 global :", " ".join(f"{d:>3}" for d in receptive_field(interleave)))
    print(f"  Theory: with sliding windows only, the receptive field after l layers = l × (W − 1) = l × {W - 1}")

    T = 131072
    print(f"\nKV cache ledger: context of {T:,} (128K) tokens, batch 1, BF16")
    print(f"  {'Model':<32}{'If all full attn':>14}{' Actual config':>12}{'  Savings':>8}")
    for name, n_layers, n_global, win, kvh, hd in MODELS:
        full_b = kv_bytes(n_layers, 0, win, T, kvh, hd)
        real_b = kv_bytes(n_global, n_layers - n_global, win, T, kvh, hd)
        print(
            f"  {MODEL_EN[name]:<32}{full_b / 2**30:>12.2f} GiB{real_b / 2**30:>10.2f} GiB"
            f"{1 - real_b / full_b:>9.1%}"
        )


if __name__ == "__main__":
    main()
