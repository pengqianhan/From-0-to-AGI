"""第 6 章 · 从极简到生产级：同一套技巧的 PyTorch 标准写法

和 06_ablation.py 的"全套"完全一样，只是每个手写的零件都换成 PyTorch 自带的：
  手写 rms_norm              → nn.RMSNorm
  手写 AdamW 循环             → torch.optim.AdamW（两个参数组：矩阵衰减，γ 不衰减）
  手写 warmup_cosine / wsd    → torch.optim.lr_scheduler.LambdaLR
  手写 clip_by_global_norm    → torch.nn.utils.clip_grad_norm_
先用和极简版相同的初始权重、相同的数据跑 200 步，逐步对拍损失；
再换成大模型常用的"std=0.02 + 残差输出投影按层数缩小"初始化完整训练一次。
运行：uv run python chapters/06-training-stability/code/08_pytorch_version.py
"""

import importlib.util
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

torch.set_num_threads(1)



def load_ablation(dtype=torch.float32):
    """加载 06_ablation.py。dtype=float64 时，它的数据和权重都用双精度创建（对拍用）。"""
    old = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    spec = importlib.util.spec_from_file_location(f"ablation_{dtype}",
                                                  Path(__file__).with_name("06_ablation.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    torch.set_default_dtype(old)
    return mod


ab = load_ablation()


class Block(nn.Module):
    """Pre-Norm 残差块：x + W2·ReLU(W1·RMSNorm(x))。第 9 章的 Transformer 块也是这个骨架。"""

    def __init__(self, width: int, hidden: int):
        super().__init__()
        self.norm = nn.RMSNorm(width, eps=1e-6)
        self.fc1 = nn.Linear(width, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, width, bias=False)   # 残差分支的输出投影

    def forward(self, x):
        return x + self.fc2(F.relu(self.fc1(self.norm(x))))


class DeepNet(nn.Module):
    def __init__(self, d_in=ab.D_IN, width=ab.WIDTH, hidden=ab.HIDDEN, blocks=ab.BLOCKS,
                 n_cls=ab.N_CLS):
        super().__init__()
        self.inp = nn.Linear(d_in, width, bias=False)
        self.blocks = nn.ModuleList(Block(width, hidden) for _ in range(blocks))
        self.norm = nn.RMSNorm(width, eps=1e-6)           # 最终 norm
        self.out = nn.Linear(width, n_cls, bias=False)

    def forward(self, x):
        h = self.inp(x)
        for b in self.blocks:
            h = b(h)
        return self.out(self.norm(h))


def init_llm_style(model: DeepNet, std: float = 0.02):
    """大模型常见写法：所有矩阵 N(0, 0.02²)；残差分支的输出投影再除以 √(2·层数)。"""
    n = len(model.blocks)
    for name, p in model.named_parameters():
        if p.dim() >= 2:
            s = std / math.sqrt(2 * n) if name.endswith("fc2.weight") else std
            nn.init.normal_(p, mean=0.0, std=s)


def copy_from_minimal(model: DeepNet, m: dict):
    """把 06 手写版的初始权重搬进来（nn.Linear 存的是转置）。"""
    with torch.no_grad():
        model.inp.weight.copy_(m["inp"].T)
        model.out.weight.copy_(m["out"].T)
        for blk, b in zip(model.blocks, m["blocks"]):
            blk.fc1.weight.copy_(b["w1"].T)
            blk.fc2.weight.copy_(b["w2"].T)


def make_optimizer(model: nn.Module, lr: float, wd: float = 0.1):
    """AdamW + 两个参数组：二维以上的矩阵做权重衰减，一维的（RMSNorm 的 γ、偏置）不做。"""
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    groups = [{"params": decay, "weight_decay": wd}, {"params": no_decay, "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.95), eps=1e-8)


def make_scheduler(opt, total: int, warmup: int, kind: str = "cosine"):
    """LambdaLR：每步把学习率设为 峰值 × lambda(step)。lambda 直接复用 05 的函数（峰值传 1）。"""
    if kind == "cosine":
        f = lambda s: ab.sched.warmup_cosine(s, total, 1.0, warmup)  # noqa: E731
    else:
        f = lambda s: ab.sched.wsd(s, total, 1.0, warmup)            # noqa: E731
    return torch.optim.lr_scheduler.LambdaLR(opt, f)


def train(model, steps: int, cfg=ab.FULL, kind: str = "cosine", seed: int = 0, data_mod=ab):
    opt = make_optimizer(model, cfg["lr"], cfg["wd"])
    scheduler = make_scheduler(opt, steps, cfg["warmup"], kind)
    data = torch.Generator().manual_seed(seed + 100)       # 和极简版同一条数据流
    losses = []
    for _ in range(steps):
        x, y = data_mod.make_batch(ab.BATCH, data)
        loss = F.cross_entropy(model(x), y)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg["clip"])  # 梯度裁剪
        opt.step()
        scheduler.step()                                   # 学习率走一步
        losses.append(loss.item())
    return losses


@torch.no_grad()
def evaluate(model, data_mod=ab):
    logits = model(data_mod.X_VAL)
    return F.cross_entropy(logits, data_mod.Y_VAL).item()


def cross_check(dtype, steps: int = 200):
    """同样的初始权重、同样的数据：PyTorch 版和 06 的手写版逐步比较损失。"""
    mod = load_ablation(dtype)
    old = torch.get_default_dtype()
    torch.set_default_dtype(dtype)                         # 整个对拍都在这个精度下进行
    ref = mod.run(mod.FULL, steps=steps)                   # 06 的手写版
    model = DeepNet()
    copy_from_minimal(model, mod.init_model(mod.FULL))
    losses = train(model, steps, data_mod=mod)
    torch.set_default_dtype(old)
    diff = max(abs(a - b) for a, b in zip(losses, ref["losses"]))
    return diff, losses[-1], ref["losses"][-1]


if __name__ == "__main__":
    # ── 对拍：同样的初始权重、同样的数据，200 步 ──────────────────────────────
    for dtype in [torch.float64, torch.float32]:
        diff, mine, ref = cross_check(dtype)
        print(f"对拍 200 步（{str(dtype).replace('torch.', '')}）：逐步损失的最大差 {diff:.1e}；"
              f"第 200 步损失 PyTorch {mine:.4f} / 手写 {ref:.4f}")

    model = DeepNet()
    opt = make_optimizer(model, 3e-3)
    n_dec = sum(p.numel() for p in opt.param_groups[0]["params"])
    n_nod = sum(p.numel() for p in opt.param_groups[1]["params"])
    print(f"参数组：做权重衰减的矩阵 {n_dec} 个参数，不衰减的 γ {n_nod} 个参数")

    # ── 大模型风格的初始化，完整训练 800 步 ─────────────────────────────────
    for kind in ["cosine", "wsd"]:
        torch.manual_seed(0)
        model = DeepNet()
        init_llm_style(model)
        losses = train(model, ab.STEPS, kind=kind)
        print(f"std=0.02 初始化 + {kind:<6s}：{ab.STEPS} 步后验证损失 {evaluate(model):.5f}")
