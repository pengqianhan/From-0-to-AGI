"""Chapter 9 · Minimal code 3: all tensor shapes in one forward pass + where the parameters go

1. Put hooks (forward hooks) on the minimal model. Print the real shape at each step of one forward pass;
2. Show the same data flow with the sizes of the main-line model (configs/main/pretrain.toml),
   step by step;
3. Parameter ledger: why SwiGLU uses 8/3·d; how many parameters tied embeddings save at different sizes.
Run: uv run python chapters/09-modern-transformer/code/03_shapes.py
"""

import importlib.util
import sys
from pathlib import Path

import torch

torch.set_num_threads(1)  # many jobs share the CPU of the build machine; on your computer you can remove this line
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


# ── 1. Real shapes in the minimal model ──────────────────────────────────────
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
print(f"1) One forward pass of the minimal model (B={B}, T={T}, D=128, H=4, head_dim=32, FFN=352, V=256):")
print(f"   {'module':<22}{'input shape':<18}output shape")
for name, i, o in rows:
    print(f"   {name:<22}{i:<18}{o}")

# ── 2. Sizes of the main-line model ──────────────────────────────────────────
m = load_model_config(str(ROOT / "configs/main/pretrain.toml"))
Bm, Tm = 8, 4096
D, H, KV, hd, F, V = m.dim, m.n_heads, m.n_kv_heads, m.head_dim, m.ffn_dim, m.vocab_size
print(f"\n2) Main-line model (configs/main/pretrain.toml, micro batch B={Bm}, T={Tm}):")
flow = [
    ("token id", f"({Bm}, {Tm})"),
    ("embedding lookup", f"({Bm}, {Tm}, {D})"),
    (f"× {m.n_layers} layers  RMSNorm", f"({Bm}, {Tm}, {D})"),
    ("  q = x·Wq → split heads", f"({Bm}, {H}, {Tm}, {hd})"),
    ("  k, v = x·Wk, x·Wv", f"({Bm}, {KV}, {Tm}, {hd})  ← GQA: {KV} K/V heads (Chapter 10)"),
    ("  QK-Norm + RoPE", "shape does not change"),
    ("  attention scores QKᵀ", f"({Bm}, {H}, {Tm}, {Tm})"),
    ("  weights·V → merge → Wo", f"({Bm}, {Tm}, {H * hd}) → ({Bm}, {Tm}, {D})"),
    ("  add to residual", f"({Bm}, {Tm}, {D})"),
    ("  SwiGLU: gate, up", f"({Bm}, {Tm}, {F})"),
    ("  down → add to residual", f"({Bm}, {Tm}, {D})"),
    ("final RMSNorm", f"({Bm}, {Tm}, {D})"),
    ("lm_head → logits", f"({Bm}, {Tm}, {V})"),
]
for step, s in flow:
    print(f"   {step:<26}{s}")
gb = Bm * Tm * V * 4 / 2**30
print(f"   Note: the logits have {Bm * Tm * V / 1e9:.2f}G numbers. In float32, they need {gb:.1f} GiB. A larger vocabulary makes the last step more expensive.")

# ── 3. Parameter ledger ──────────────────────────────────────────────────────
d = 1280
print(f"\n3) FFN parameters (d={d}): classic MLP, 4d, two matrices = {2 * d * 4 * d / 1e6:.2f}M; "
      f"SwiGLU, 8/3·d, three matrices = {3 * d * (8 * d // 3) / 1e6:.2f}M (almost equal: this is where 8/3 comes from)")
print(f"   The main-line model uses ffn_dim={F} (= {F / D:.2f}·d)")

qwen3_06b = ModelConfig(vocab_size=151936, dim=1024, n_layers=28, n_heads=16, n_kv_heads=8,
                        head_dim=128, ffn_dim=3072, tie_embeddings=True)  # from the official config.json
cases = [("Ch. 9 minimal model", tiny_cfg := ModelConfig(vocab_size=256, dim=128, n_layers=4, n_heads=4,
                                                  n_kv_heads=4, ffn_dim=352, tie_embeddings=True)),
         ("configs/tiny", load_model_config(str(ROOT / "configs/tiny/pretrain.toml"))),
         ("Qwen3-0.6B official", qwen3_06b), ("main (configs/main)", m)]
print(f"\n   {'model':<20}{'vocab V×d':>13}{'tied total':>13}{'share':>7}{'if untied':>10}   actual")
for name, cfg in cases:
    actual = "tied" if cfg.tie_embeddings else "untied"
    cfg.tie_embeddings = True
    c = count_params(cfg)
    tot, emb = c["total"], c["embedding"]
    print(f"   {name:<20}{emb / 1e6:>12.2f}M{tot / 1e6:>12.2f}M{emb / tot:>7.1%}{(tot + emb) / 1e6:>9.2f}M   {actual}")
assert count_params(tiny_cfg)["total"] == sum(p.numel() for p in model.parameters())
