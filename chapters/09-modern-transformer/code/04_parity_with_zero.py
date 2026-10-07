"""Chapter 9 · Minimal code 4: a parity check on two levels. Minimal model = zero.Transformer = Hugging Face Qwen3

1. Build a zero.Transformer (production code, zero/model.py) with the same hyperparameters.
   Copy the weights of the minimal model into it without changes (the parameter names are the same
   on purpose, so state_dict loads directly). Compare the logits on the same input;
2. Use zero/hf.py to export the model to the Hugging Face Qwen3 format. Load it again with the
   official Qwen3ForCausalLM from transformers, and compare the logits again;
3. Greedy generation: the minimal code (calculates the full sequence again at each step) and zero
   (with a KV cache, Chapter 10) give the same bytes.

The structures of the minimal code and zero have only two differences. The configuration can align both:
  - The minimal code has no GQA → set n_kv_heads = n_heads in zero (each query head has its own K/V.
    This is normal multi-head attention);
  - The minimal code writes softmax(QKᵀ/√d)·V by hand → zero calls F.scaled_dot_product_attention.
    The math is the same, but the order of operations is different. Thus the results agree
    "within floating-point error" (~1e-6). They are not identical bit for bit.
Run: uv run python chapters/09-modern-transformer/code/04_parity_with_zero.py
(First run 02_tiny_transformer.py to get the trained weights. Without them, the parity check uses random initial weights.)
"""

import importlib.util
import sys
import tempfile
from pathlib import Path

import torch

torch.set_num_threads(1)  # many jobs share the CPU of the build machine; on your computer you can remove this line
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from zero.config import ModelConfig  # noqa: E402
from zero.generate import generate as zero_generate  # noqa: E402
from zero.hf import export_to_hf_qwen3  # noqa: E402
from zero.model import Transformer  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "tiny", Path(__file__).with_name("02_tiny_transformer.py"))
tiny = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tiny)

torch.manual_seed(0)
if tiny.CKPT.exists():
    mini, _ = tiny.load_trained()
    print("Minimal model: uses the trained weights from", tiny.CKPT.relative_to(ROOT))
else:
    mini = tiny.TinyTransformer(tiny.Config()).eval()
    print("Minimal model: no trained weights found; uses random initialization")
c = mini.cfg

# ── 1. The same hyperparameters → the ModelConfig of zero ─────────────────────
zcfg = ModelConfig(
    vocab_size=c.vocab_size, dim=c.dim, n_layers=c.n_layers,
    n_heads=c.n_heads, n_kv_heads=c.n_heads,    # no GQA: number of K/V heads = number of query heads
    head_dim=c.dim // c.n_heads, ffn_dim=c.ffn_dim, rope_theta=c.rope_theta,
    max_seq_len=c.seq_len, norm_eps=c.eps, qk_norm=True, tie_embeddings=True,
)
prod = Transformer(zcfg).eval()
prod.load_state_dict(mini.state_dict())         # each parameter name has a match, so strict=True also passes
n_mini = sum(p.numel() for p in mini.parameters())
print(f"Parameters: minimal {n_mini:,}  zero {prod.num_params():,}")

text = b"ROMEO:\nBut soft, what light through yonder window breaks?\nJULIET:\n"
tokens = torch.tensor([list(text)])
with torch.no_grad():
    l_mini, l_prod = mini(tokens), prod(tokens)
diff = (l_mini - l_prod).abs().max().item()
print(f"\n1) minimal vs zero: logits shape {tuple(l_mini.shape)}, max absolute difference {diff:.2e}, "
      f"positions with the same next-byte prediction {(l_mini.argmax(-1) == l_prod.argmax(-1)).float().mean():.0%}")
torch.testing.assert_close(l_prod, l_mini, rtol=1e-5, atol=1e-5)

# ── 2. Export to HF Qwen3; load it again with the official transformers code ──
try:
    from transformers import Qwen3ForCausalLM
except ImportError:
    Qwen3ForCausalLM = None
if Qwen3ForCausalLM is not None:
    with tempfile.TemporaryDirectory() as d:
        export_to_hf_qwen3(prod, zcfg, d, dtype=torch.float32)
        hf = Qwen3ForCausalLM.from_pretrained(d).eval()
        with torch.no_grad():
            l_hf = hf(tokens).logits
    diff_hf = (l_hf - l_mini).abs().max().item()
    print(f"2) minimal vs transformers.Qwen3ForCausalLM: max absolute difference {diff_hf:.2e}")
    torch.testing.assert_close(l_hf, l_mini, rtol=1e-5, atol=1e-5)

# ── 3. Greedy generation gives the same bytes ─────────────────────────────────
prompt = list(b"ROMEO:\n")
ids = torch.tensor([prompt])
with torch.no_grad():
    for _ in range(80):  # minimal code: calculate the full sequence again at each step, take the most probable byte
        ids = torch.cat([ids, mini(ids)[:, -1].argmax(-1, keepdim=True)], dim=1)
out_mini = ids[0, len(prompt):].tolist()
out_prod = zero_generate(prod, prompt, 80, temperature=0.0, use_cache=True)
print(f"3) greedy generation of 80 bytes, the two sides are identical: {out_mini == out_prod}")
print("   " + bytes(out_prod).decode("utf-8", errors="replace").replace("\n", "\n   "))
