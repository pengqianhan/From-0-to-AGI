"""第 9 章 · 极简代码 4：两层对拍 —— 极简模型 = zero.Transformer = Hugging Face Qwen3

1. 按同样的超参建一个 zero.Transformer（生产级，zero/model.py），把极简模型的权重原样搬进去
   （参数名是故意取成一样的，state_dict 直接能加载）；同一批输入，比较 logits；
2. 再用 zero/hf.py 把它导出成 Hugging Face Qwen3 格式，用 transformers 官方的 Qwen3ForCausalLM
   读回来，再比一次 logits；
3. 贪心生成：极简版（每步重算整段）与 zero（带 KV cache，第 10 章）逐字节一致。

极简版与 zero 的结构差别只有两处，都能用配置对齐：
  - 极简版没有 GQA → zero 里设 n_kv_heads = n_heads（每个查询头有自己的 K/V，就是普通多头注意力）；
  - 极简版手写 softmax(QKᵀ/√d)·V → zero 调用 F.scaled_dot_product_attention，数学相同、
    运算顺序不同，所以是"浮点误差内一致"（~1e-6），不是逐位相同。
运行：uv run python chapters/09-modern-transformer/code/04_parity_with_zero.py
（先运行 02_tiny_transformer.py 得到训练好的权重；没有的话用随机初始化的权重对拍）
"""

import importlib.util
import sys
import tempfile
from pathlib import Path

import torch

torch.set_num_threads(1)  # 构建环境多任务共享 CPU；本机可以删掉这行
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
    print("极简模型：读取训练好的权重", tiny.CKPT.relative_to(ROOT))
else:
    mini = tiny.TinyTransformer(tiny.Config()).eval()
    print("极简模型：没找到训练好的权重，用随机初始化")
c = mini.cfg

# ── 1. 同一份超参 → zero 的 ModelConfig ─────────────────────────────────────
zcfg = ModelConfig(
    vocab_size=c.vocab_size, dim=c.dim, n_layers=c.n_layers,
    n_heads=c.n_heads, n_kv_heads=c.n_heads,    # 无 GQA：K/V 头数 = 查询头数
    head_dim=c.dim // c.n_heads, ffn_dim=c.ffn_dim, rope_theta=c.rope_theta,
    max_seq_len=c.seq_len, norm_eps=c.eps, qk_norm=True, tie_embeddings=True,
)
prod = Transformer(zcfg).eval()
prod.load_state_dict(mini.state_dict())         # 参数名一一对应，strict=True 也能过
n_mini = sum(p.numel() for p in mini.parameters())
print(f"参数量：极简 {n_mini:,}  zero {prod.num_params():,}")

text = b"ROMEO:\nBut soft, what light through yonder window breaks?\nJULIET:\n"
tokens = torch.tensor([list(text)])
with torch.no_grad():
    l_mini, l_prod = mini(tokens), prod(tokens)
diff = (l_mini - l_prod).abs().max().item()
print(f"\n1) 极简 vs zero：logits 形状 {tuple(l_mini.shape)}，最大绝对差 {diff:.2e}，"
      f"下一个字节的预测相同的位置 {(l_mini.argmax(-1) == l_prod.argmax(-1)).float().mean():.0%}")
torch.testing.assert_close(l_prod, l_mini, rtol=1e-5, atol=1e-5)

# ── 2. 导出成 HF Qwen3，用 transformers 官方实现读回来 ───────────────────────
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
    print(f"2) 极简 vs transformers.Qwen3ForCausalLM：最大绝对差 {diff_hf:.2e}")
    torch.testing.assert_close(l_hf, l_mini, rtol=1e-5, atol=1e-5)

# ── 3. 贪心生成逐字节一致 ────────────────────────────────────────────────────
prompt = list(b"ROMEO:\n")
ids = torch.tensor([prompt])
with torch.no_grad():
    for _ in range(80):  # 极简版：每一步把整段重新算一遍，取最大概率的字节
        ids = torch.cat([ids, mini(ids)[:, -1].argmax(-1, keepdim=True)], dim=1)
out_mini = ids[0, len(prompt):].tolist()
out_prod = zero_generate(prod, prompt, 80, temperature=0.0, use_cache=True)
print(f"3) 贪心生成 80 个字节，两边完全相同：{out_mini == out_prod}")
print("   " + bytes(out_prod).decode("utf-8", errors="replace").replace("\n", "\n   "))
