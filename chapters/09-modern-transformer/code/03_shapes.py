"""第 9 章 · 极简代码 3：一次前向的完整张量形状 + 参数都花在哪

1. 在极简模型上挂钩子（forward hook），把一次前向里每一步的真实形状打印出来；
2. 同一张数据流换成主线模型的尺寸（configs/main/pretrain.toml），逐步写出形状；
3. 参数账本：SwiGLU 为什么取 8/3·d；共享 embedding 在不同规模上省了多少。
运行：uv run python chapters/09-modern-transformer/code/03_shapes.py
"""

import importlib.util
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from zero.config import ModelConfig, load_model_config  # noqa: E402
from zero.model import count_params  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "tiny", Path(__file__).with_name("02_tiny_transformer.py"))
tiny = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tiny)


def shape(t):
    return "(" + ", ".join(str(s) for s in t.shape) + ")"


# ── 1. 极简模型上的真实形状 ──────────────────────────────────────────────────
torch.manual_seed(0)
model = tiny.TinyTransformer(tiny.Config()).eval()
B, T = 2, 16
rows = []
watch = ["tok_emb", "layers.0.attn_norm", "layers.0.attn.wq", "layers.0.attn.wk",
         "layers.0.attn.q_norm", "layers.0.attn.wo", "layers.0.attn", "layers.0.ffn_norm",
         "layers.0.ffn.w_gate", "layers.0.ffn.w_up", "layers.0.ffn.w_down", "layers.0",
         "layers.3", "norm", "lm_head"]
mods = dict(model.named_modules())
for name in watch:
    mods[name].register_forward_hook(
        lambda m, i, o, name=name: rows.append((name, shape(i[0]), shape(o))))
with torch.no_grad():
    model(torch.randint(0, 256, (B, T)))
print(f"1) 极简模型一次前向（B={B}, T={T}, D=128, H=4, head_dim=32, FFN=352, V=256）：")
print(f"   {'模块':<22}{'输入形状':<18}输出形状")
for name, i, o in rows:
    print(f"   {name:<22}{i:<18}{o}")

# ── 2. 主线模型的尺寸 ────────────────────────────────────────────────────────
m = load_model_config(str(ROOT / "configs/main/pretrain.toml"))
Bm, Tm = 8, 4096
D, H, KV, hd, F, V = m.dim, m.n_heads, m.n_kv_heads, m.head_dim, m.ffn_dim, m.vocab_size
print(f"\n2) 主线模型（configs/main/pretrain.toml，micro batch B={Bm}, T={Tm}）：")
flow = [
    ("token id", f"({Bm}, {Tm})"),
    ("embedding 查表", f"({Bm}, {Tm}, {D})"),
    (f"× {m.n_layers} 层  RMSNorm", f"({Bm}, {Tm}, {D})"),
    ("  q = x·Wq → 拆头", f"({Bm}, {H}, {Tm}, {hd})"),
    ("  k, v = x·Wk, x·Wv", f"({Bm}, {KV}, {Tm}, {hd})  ← GQA：{KV} 个 K/V 头（第 10 章）"),
    ("  QK-Norm + RoPE", "形状不变"),
    ("  注意力分数 QKᵀ", f"({Bm}, {H}, {Tm}, {Tm})"),
    ("  加权求和 → 合并头 → Wo", f"({Bm}, {Tm}, {H * hd}) → ({Bm}, {Tm}, {D})"),
    ("  残差相加", f"({Bm}, {Tm}, {D})"),
    ("  SwiGLU: gate、up", f"({Bm}, {Tm}, {F})"),
    ("  down → 残差相加", f"({Bm}, {Tm}, {D})"),
    ("最后 RMSNorm", f"({Bm}, {Tm}, {D})"),
    ("lm_head → logits", f"({Bm}, {Tm}, {V})"),
]
for step, s in flow:
    print(f"   {step:<26}{s}")
gb = Bm * Tm * V * 4 / 2**30
print(f"   注意：logits 有 {Bm * Tm * V / 1e9:.2f}G 个数，按 float32 存要 {gb:.1f} GiB —— 词表越大，最后一步越贵")

# ── 3. 参数账本 ──────────────────────────────────────────────────────────────
d = 1280
print(f"\n3) FFN 参数（d={d}）：经典 MLP 4d 两个矩阵 = {2 * d * 4 * d / 1e6:.2f}M；"
      f"SwiGLU 8/3·d 三个矩阵 = {3 * d * (8 * d // 3) / 1e6:.2f}M（几乎相等，这就是 8/3 的来历）")
print(f"   主线模型实际取 ffn_dim={F}（= {F / D:.2f}·d）")

qwen3_06b = ModelConfig(vocab_size=151936, dim=1024, n_layers=28, n_heads=16, n_kv_heads=8,
                        head_dim=128, ffn_dim=3072, tie_embeddings=True)  # 来自官方 config.json
cases = [("本章极简模型", tiny_cfg := ModelConfig(vocab_size=256, dim=128, n_layers=4, n_heads=4,
                                                  n_kv_heads=4, ffn_dim=352, tie_embeddings=True)),
         ("configs/tiny", load_model_config(str(ROOT / "configs/tiny/pretrain.toml"))),
         ("Qwen3-0.6B（官方配置）", qwen3_06b), ("configs/main（主线）", m)]
print(f"\n   {'模型':<20}{'词表矩阵 V×d':>13}{'共享时总参数':>13}{'占比':>7}{'不共享时':>10}   实际")
for name, cfg in cases:
    actual = "共享" if cfg.tie_embeddings else "不共享"
    cfg.tie_embeddings = True
    c = count_params(cfg)
    tot, emb = c["total"], c["embedding"]
    print(f"   {name:<20}{emb / 1e6:>12.2f}M{tot / 1e6:>12.2f}M{emb / tot:>7.1%}{(tot + emb) / 1e6:>9.2f}M   {actual}")
assert count_params(tiny_cfg)["total"] == sum(p.numel() for p in model.parameters())
