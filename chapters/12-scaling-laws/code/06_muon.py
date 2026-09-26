"""第 12 章 · 极简代码 6：Muon —— 把矩阵的更新"正交化"，和 AdamW 在同样的 token 数下比一比

Muon 对每个隐藏层权重矩阵：
    M ← μ·M + G                 （动量）
    U ← NS5(μ·M + G)            （5 步 Newton–Schulz，把 G = UΣVᵀ 变成约 UVᵀ：所有方向步长相同）
    W ← W·(1 − ηλ) − η·0.2·√max(m,n)·U   （0.2·√max(m,n)：让更新的 RMS 和 AdamW 相当，学习率可以沿用）
embedding、lm_head、RMSNorm 仍然用 AdamW（Kimi K2、GLM-4.5、DeepSeek-V4 的分组方式）。

实验：阶梯里的 s2、s3 两个尺寸，和 03 脚本同样的 131,072 个字节，Muon 扫 3 个学习率，
和 03 脚本里调好的 AdamW 最优值比较。这是"极小配置演示"：几万参数上的结果不能直接推到 0.7B。

运行：uv run python chapters/12-scaling-laws/code/06_muon.py
      （需要先跑 03_lr_sweep.py；单线程约 2–4 分钟，结果缓存在 out/ch12/muon.json）
生产级实现：zero/train/muon.py（MuonAdamW + build_muon_optimizer），测试 tests/test_muon.py。
"""

import argparse
import importlib.util
import json
import math
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("lr_sweep", HERE / "03_lr_sweep.py")
sw = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sw)

MUON_LRS = [2.5e-3, 5e-3, 1e-2]
SIZES = ["s2", "s3"]


def newton_schulz5(G, steps=5):
    """G → 约 UVᵀ。系数来自 Keller Jordan 的参考实现：5 步就把奇异值推到 1 附近。"""
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G / (G.norm() + 1e-7)  # 最大奇异值 ≤ 1，迭代才收敛
    tall = X.shape[0] > X.shape[1]
    if tall:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        X = a * X + (b * A + c * A @ A) @ X
    return X.T if tall else X


class Muon(torch.optim.Optimizer):
    """极简版：组里 muon=True 的走 Muon，否则走 torch 自带的 AdamW 公式。"""

    def __init__(self, groups, lr, wd=0.1, momentum=0.95):
        super().__init__(groups, dict(lr=lr, wd=wd, momentum=momentum, muon=False))
        self.adam = torch.optim.AdamW([p for g in self.param_groups if not g["muon"] for p in g["params"]],
                                      lr=lr, betas=(0.9, 0.95), weight_decay=0.0)

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            if not g["muon"]:
                continue
            for p in g["params"]:
                buf = self.state[p].setdefault("m", torch.zeros_like(p))
                buf.mul_(g["momentum"]).add_(p.grad)
                U = newton_schulz5(p.grad + g["momentum"] * buf)  # Nesterov
                p.mul_(1 - g["lr"] * g["wd"])
                p.add_(U, alpha=-g["lr"] * 0.2 * math.sqrt(max(p.shape)))
        for ga in self.adam.param_groups:  # AdamW 部分跟着调度走同一个学习率
            ga["lr"] = self.param_groups[0]["lr"]
        self.adam.step()

    def zero_grad(self, set_to_none=True):
        super().zero_grad(set_to_none)

    def state_dict(self):  # 分叉时复制优化器状态用
        return {"muon": super().state_dict(), "adam": self.adam.state_dict()}

    def load_state_dict(self, sd):
        super().load_state_dict(sd["muon"])
        self.adam.load_state_dict(sd["adam"])


def make_muon(model, lr):
    hidden = [p for n, p in model.named_parameters() if n.startswith("layers.") and p.ndim == 2]
    other = [p for n, p in model.named_parameters() if not (n.startswith("layers.") and p.ndim == 2)]
    return Muon([{"params": hidden, "muon": True}, {"params": other, "muon": False}], lr=lr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fresh", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(1)
    G = torch.randn(64, 32)
    s = torch.linalg.svdvals(newton_schulz5(G))
    print(f"NS5 检查：随机 64×32 梯度的奇异值 {torch.linalg.svdvals(G / G.norm()).min():.3f}–"
          f"{torch.linalg.svdvals(G / G.norm()).max():.3f} → 正交化后 {s.min():.3f}–{s.max():.3f}")

    sweep = sw.run_sweep()
    best = sw.best_lrs(sweep)
    path = sw.OUT / "muon.json"
    rows = [] if args.fresh or not path.exists() else json.loads(path.read_text())
    done = {(r["size"], r["lr"]) for r in rows}
    data = sw.load_bytes()
    for name, dim, L in sw.LADDER:
        if name not in SIZES:
            continue
        for lr in MUON_LRS:
            if (name, lr) in done:
                continue
            bpb = sw.train_wsd_branches(dim, L, lr, [sw.SWEEP_TOKENS], data=data, make_opt=make_muon)[sw.SWEEP_TOKENS]
            rows.append({"size": name, "N": best[name]["N"], "lr": lr, "val_bpb": bpb})
            print(f"  Muon {name} lr={lr:g}: {bpb:.4f}")
            path.write_text(json.dumps(rows, indent=1))
    print(f"\n同样 {sw.SWEEP_TOKENS:,} 字节，验证 loss（bit/字节）：")
    print(f"{'尺寸':>4} {'N':>8} | {'AdamW 最优':>14} | {'Muon 最优':>14} | 差")
    for name in SIZES:
        mu = min((r for r in rows if r["size"] == name), key=lambda r: r["val_bpb"])
        ad = best[name]
        print(f"{name:>4} {ad['N']:>8,} | {ad['val_bpb']:.4f} (η={ad['lr']:<6g}) | {mu['val_bpb']:.4f} (η={mu['lr']:<6g}) | "
              f"{mu['val_bpb'] - ad['val_bpb']:+.4f}")


if __name__ == "__main__":
    main()
