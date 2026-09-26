"""第 14 章 · 极简代码 5：训练时显存里装了什么 —— 数出来，而不是背公式

训练一步，显存里有四类东西：
  参数（FP32 主权重 4 字节）+ 梯度（4 字节）+ AdamW 的 m 和 v（各 4 字节）= 每个参数 16 字节，
  再加上前向时为反向保存的激活值（activations），它和 micro batch × 序列长度成正比。

这个脚本在 CPU 上用 zero 的真实模型（tiny 形状）把它们一项项数出来：
  ① 走一步 AdamW 之后，统计参数、梯度、优化器状态各占多少字节；
  ② 用 torch.autograd.graph.saved_tensors_hooks 记录反向要用的每个张量，
     比较 FP32、BF16 autocast、BF16 + 激活检查点（activation checkpointing）三种情况；
  ③ 把同样的账算到主线模型（689.5M，configs/main/pretrain.toml）上，看 micro batch 怎么选。

运行：uv run python chapters/14-pretraining-engineering/code/05_memory.py   （几秒）
"""

import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))  # 让 `import zero` 在仓库任意位置运行都能找到

from zero.config import ModelConfig, load_model_config  # noqa: E402
from zero.model import Transformer, count_params, cross_entropy_loss  # noqa: E402
from zero.tools.memory_calc import GiB, estimate_memory  # noqa: E402

TINY = ModelConfig(vocab_size=2048, dim=128, n_layers=4, n_heads=4, n_kv_heads=2, head_dim=32,
                   ffn_dim=384, max_seq_len=256, tie_embeddings=True)
B, T = 4, 128


def nbytes(ts) -> int:
    return sum(t.numel() * t.element_size() for t in ts)


def saved_activation_bytes(model: nn.Module, tokens: torch.Tensor, autocast: bool) -> int:
    """前向时，autograd 为反向保存了哪些张量？按存储去重，不算参数和 buffer 本身。"""
    own = {p.untyped_storage().data_ptr() for p in model.parameters()}
    own |= {b.untyped_storage().data_ptr() for b in model.buffers()}
    seen = {}

    def pack(t):
        key = t.untyped_storage().data_ptr()
        if key not in own and key not in seen:
            seen[key] = t.untyped_storage().nbytes()
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
            logits = model(tokens)
        loss = cross_entropy_loss(logits, tokens)
    loss.backward()
    return sum(seen.values())


class CheckpointedBlock(nn.Module):
    """激活检查点：前向只存这一块的输入，反向时把整块重算一遍。"""

    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(self, x, cos, sin, kv_cache=None, start_pos=0):
        return checkpoint(self.block, x, cos, sin, use_reentrant=True)


def main():
    torch.set_num_threads(1)
    torch.manual_seed(0)
    model = Transformer(TINY)
    P = sum(p.numel() for p in model.parameters())
    tokens = torch.randint(0, TINY.vocab_size, (B, T))

    print(f"① 参数 + 梯度 + 优化器状态（tiny：{P:,} 个参数，走一步 AdamW 之后）")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    cross_entropy_loss(model(tokens), tokens).backward()
    opt.step()
    rows = [
        ("参数（FP32）", nbytes(model.parameters())),
        ("梯度（FP32）", nbytes(p.grad for p in model.parameters())),
        ("AdamW m（FP32）", nbytes(s["exp_avg"] for s in opt.state.values())),
        ("AdamW v（FP32）", nbytes(s["exp_avg_sq"] for s in opt.state.values())),
    ]
    for name, b in rows:
        print(f"  {name:<16} {b:>12,} 字节   {b / P:.0f} 字节/参数")
    total = sum(b for _, b in rows)
    print(f"  {'合计':<16} {total:>12,} 字节   {total / P:.0f} 字节/参数")
    opt.zero_grad(set_to_none=True)

    print(f"\n② 为反向保存的激活（micro batch {B} × 序列 {T} = {B * T} 个 token，{TINY.n_layers} 层）")
    cases = [("FP32", model, False), ("BF16 autocast", model, True)]
    ckpt_model = Transformer(TINY)
    ckpt_model.load_state_dict(model.state_dict())
    ckpt_model.layers = nn.ModuleList([CheckpointedBlock(b) for b in ckpt_model.layers])
    cases.append(("BF16 + 激活检查点", ckpt_model, True))
    base = None
    for name, m, ac in cases:
        b = saved_activation_bytes(m, tokens, ac)
        base = base or b
        print(f"  {name:<18} {b:>11,} 字节 = {b / (B * T):>8,.0f} 字节/token   ({b / base:.0%})")
    print("  BF16 autocast 里还包括前向转出的 BF16 权重副本；激活检查点每层只剩块的输入（4·dim 字节/token），")
    print("  代价是反向时每层多算一遍前向（约多 1/3 的计算量）")

    print("\n③ 同样的账算到主线模型（8×H100 80GB，BF16，seq 4096；zero/tools/memory_calc.py）")
    main_cfg = load_model_config(ROOT / "configs/main/pretrain.toml")
    Pm = count_params(main_cfg)["total"]
    print(f"  参数 {Pm / 1e6:.1f}M × 16 字节 = {16 * Pm / GiB:.1f} GiB（每张卡都存一份：DDP）")
    print(f"  {'micro batch':>11} {'激活':>8} {'DDP 合计':>9} {'FSDP 合计':>10} {'DDP + 检查点':>12}")
    for mb in (1, 2, 4, 8):
        ddp = estimate_memory(main_cfg, mb, 4096, num_gpus=8, strategy="ddp")
        fsdp = estimate_memory(main_cfg, mb, 4096, num_gpus=8, strategy="fsdp")
        ck = estimate_memory(main_cfg, mb, 4096, num_gpus=8, strategy="ddp", checkpointing=True)
        print(f"  {mb:>11} {ddp.activations / GiB:>7.1f}G {ddp.total / GiB:>8.1f}G "
              f"{fsdp.total / GiB:>9.1f}G {ck.total / GiB:>11.1f}G")
    print("  （保守上界，eager 模式；GPU 上尚未验证。micro batch 8 超过 80 GB → 用梯度累积凑大 batch）")


if __name__ == "__main__":
    main()
