"""第 6 章 · 极简代码 6：同一个深网络，把稳定训练的技巧一个个打开/关掉

任务：输入是 16 维随机向量，标签由一个固定的"老师网络"给出（10 类）。每一步都采一批新数据
（像预训练一样数据取之不尽），所以训练损失就接近真实水平；另留 2048 个样本做验证。
学生网络：12 个块（24 个线性层），宽 32。所有技巧都手写：
  初始化、RMSNorm（Pre-Norm）、残差、AdamW、warmup + 余弦、梯度裁剪。
运行：uv run python chapters/06-training-stability/code/06_ablation.py      （约 1.5 分钟）
"""

import copy
import importlib.util
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

torch.set_num_threads(1)


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


norm = _load("norm", "02_normalization.py")
sched = _load("sched", "05_lr_schedule.py")

D_IN, N_CLS, WIDTH, HIDDEN, BLOCKS, BATCH, STEPS = 16, 10, 32, 64, 12, 128, 800

# ── 数据：固定的老师网络 ─────────────────────────────────────────────────────
_tg = torch.Generator().manual_seed(1234)
T1 = torch.randn(D_IN, 64, generator=_tg) / math.sqrt(D_IN)
T2 = torch.randn(64, N_CLS, generator=_tg) / math.sqrt(64)


def make_batch(n: int, g: torch.Generator):
    x = torch.randn(n, D_IN, generator=g)
    y = (torch.tanh(2 * x @ T1) @ T2).argmax(1)
    return x, y


X_VAL, Y_VAL = make_batch(2048, torch.Generator().manual_seed(1))

# ── 配置：每个开关对应一个技巧 ───────────────────────────────────────────────
FULL = dict(
    init="kaiming",      # "kaiming"：按 fan_in 缩放；"std1"：N(0, 1)
    residual=True,       # h ← h + f(h)，否则 h ← f(h)
    norm=True,           # f 的输入先过 RMSNorm（Pre-Norm），输出头前再过一次
    opt="adamw",         # "adamw" 或 "sgd"
    lr=3e-3,
    schedule="cosine",   # "cosine"（带 warmup）、"wsd"（带 warmup）或 "const"
    warmup=100,
    clip=1.0,            # 梯度全局范数上限；None 表示不裁剪
    wd=0.1,
)


def init_model(cfg, seed: int = 0):
    g = torch.Generator().manual_seed(seed)

    def w(fan_in, fan_out, std):
        return (torch.randn(fan_in, fan_out, generator=g) * std).requires_grad_()

    kaiming = cfg["init"] == "kaiming"
    blocks = []
    for _ in range(BLOCKS):
        s1 = math.sqrt(2 / WIDTH) if kaiming else 1.0
        s2 = math.sqrt(1 / HIDDEN) if kaiming else 1.0
        if kaiming and cfg["residual"]:
            s2 /= math.sqrt(2 * BLOCKS)          # 残差分支的输出投影按层数缩小（见 03）
        blocks.append(dict(w1=w(WIDTH, HIDDEN, s1), w2=w(HIDDEN, WIDTH, s2),
                           g=torch.ones(WIDTH, requires_grad=True)))
    return dict(inp=w(D_IN, WIDTH, 1 / math.sqrt(D_IN)), blocks=blocks,
                g_final=torch.ones(WIDTH, requires_grad=True),
                out=w(WIDTH, N_CLS, 1 / math.sqrt(WIDTH)))


def parameters(model, cfg):
    """返回 [(名字, 张量)]。没用归一化时，γ 不参与计算，也就不交给优化器。"""
    ps = [("inp", model["inp"]), ("out", model["out"])]
    for i, b in enumerate(model["blocks"]):
        ps += [(f"b{i}.w1", b["w1"]), (f"b{i}.w2", b["w2"])]
        if cfg["norm"]:
            ps.append((f"b{i}.g", b["g"]))
    if cfg["norm"]:
        ps.append(("g_final", model["g_final"]))
    return ps


def forward(model, x, cfg):
    h = x @ model["inp"]
    for b in model["blocks"]:
        z = norm.rms_norm(h, b["g"]) if cfg["norm"] else h     # Pre-Norm：先归一化再进分支
        f = torch.relu(z @ b["w1"]) @ b["w2"]
        h = h + f if cfg["residual"] else torch.relu(f)       # 残差：h + f(h)
    if cfg["norm"]:
        h = norm.rms_norm(h, model["g_final"])                 # 输出头前的最终 norm
    return h @ model["out"]


def lr_fn(cfg, total: int):
    peak, warm = cfg["lr"], cfg["warmup"]
    if cfg["schedule"] == "cosine":
        return lambda s: sched.warmup_cosine(s, total, peak, warm)
    if cfg["schedule"] == "wsd":
        return lambda s: sched.wsd(s, total, peak, warm)
    return lambda s: peak * min(1.0, (s + 1) / warm) if warm else peak


def new_state(model, cfg, seed: int = 0):
    ps = parameters(model, cfg)
    return dict(step=0, data=torch.Generator().manual_seed(seed + 100),
                m=[torch.zeros_like(p) for _, p in ps], v=[torch.zeros_like(p) for _, p in ps])


def train_steps(model, state, cfg, lr_of_step, n_steps: int, bad_steps=(), log_every=0):
    """训练 n_steps 步。返回每步的 (loss, 裁剪前的梯度范数)；发散（NaN/inf）就提前停。"""
    ps = parameters(model, cfg)
    hist = []
    b1, b2 = 0.9, 0.95
    for _ in range(n_steps):
        t = state["step"]
        x, y = make_batch(BATCH, state["data"])
        if t in bad_steps:
            x = x * 30.0                                        # 模拟一批坏数据（07 用）
        loss = F.cross_entropy(forward(model, x, cfg), y)
        for _, p in ps:
            p.grad = None
        loss.backward()
        grads = [p.grad for _, p in ps]
        if cfg["clip"]:
            gnorm = sched.clip_by_global_norm(grads, cfg["clip"])   # 梯度裁剪
        else:
            gnorm = math.sqrt(sum(float((g * g).sum()) for g in grads))
        if not (math.isfinite(loss.item()) and math.isfinite(gnorm)):
            hist.append((float("nan"), float("nan")))
            break
        lr = lr_of_step(t)
        with torch.no_grad():
            for i, (name, p) in enumerate(ps):
                g = p.grad
                if cfg["opt"] == "sgd":
                    p -= lr * g                                     # θ ← θ − η·g
                    continue
                state["m"][i].mul_(b1).add_(g, alpha=1 - b1)        # m ← β₁m + (1−β₁)g
                state["v"][i].mul_(b2).addcmul_(g, g, value=1 - b2) # v ← β₂v + (1−β₂)g²
                m_hat = state["m"][i] / (1 - b1 ** (t + 1))
                v_hat = state["v"][i] / (1 - b2 ** (t + 1))
                if p.dim() >= 2:                                    # 只衰减矩阵，不衰减 γ
                    p.mul_(1 - lr * cfg["wd"])                      # 解耦权重衰减
                p -= lr * m_hat / (v_hat.sqrt() + 1e-8)
        hist.append((loss.item(), gnorm))
        state["step"] += 1
        if log_every and t % log_every == 0:
            print(f"    step {t:4d}  loss {loss.item():.3f}  |g| {gnorm:.2f}  lr {lr:.2e}")
    return hist


@torch.no_grad()
def evaluate(model, cfg):
    logits = forward(model, X_VAL, cfg)
    if not torch.isfinite(logits).all():
        return float("nan"), float("nan")
    return F.cross_entropy(logits, Y_VAL).item(), (logits.argmax(1) == Y_VAL).float().mean().item()


def run(cfg, steps: int = STEPS, seed: int = 0, **kw):
    model = init_model(cfg, seed)
    state = new_state(model, cfg, seed)
    hist = train_steps(model, state, cfg, lr_fn(cfg, steps), steps, **kw)
    val_loss, val_acc = evaluate(model, cfg)
    losses = [h[0] for h in hist]
    return dict(losses=losses, grad_norms=[h[1] for h in hist], val_loss=val_loss,
                val_acc=val_acc, diverged=not math.isfinite(val_loss) or len(hist) < steps,
                model=model, state=state)


def verdict(r) -> str:
    if r["diverged"]:
        return "发散（NaN）"
    if r["val_loss"] > 2.0:
        return "没学会"
    if r["val_loss"] > 1.0:
        return "学得很慢"
    return "训练成功"


BASE = dict(FULL, init="std1", residual=False, norm=False, opt="sgd", schedule="const",
            warmup=0, clip=None)
LADDER = [  # (名字, 配置, 要试的学习率)；SGD 各行都在三个学习率里挑最好的
    ("A 普通深网络（std=1 初始化 + SGD）", BASE, [0.01, 0.03, 0.1]),
    ("B + Kaiming 初始化", dict(BASE, init="kaiming"), [0.01, 0.03, 0.1]),
    ("C + 残差连接", dict(BASE, init="kaiming", residual=True), [0.01, 0.03, 0.1]),
    ("D + RMSNorm（Pre-Norm）", dict(BASE, init="kaiming", residual=True, norm=True),
     [0.01, 0.03, 0.1]),
    ("E SGD → AdamW", dict(FULL, schedule="const", warmup=0, clip=None), [3e-3]),
    ("F + warmup + 余弦衰减", dict(FULL, clip=None), [3e-3]),
    ("G + 梯度裁剪（= 全套）", FULL, [3e-3]),
]
LEAVE_ONE_OUT = [
    ("全套，但去掉残差", dict(FULL, residual=False)),
    ("全套，但去掉 RMSNorm", dict(FULL, norm=False)),
    ("全套，但初始化用 std=1", dict(FULL, init="std1")),
    ("全套，但学习率恒定（无 warmup/衰减）", dict(FULL, schedule="const", warmup=0)),
]


def best_of(cfg, lrs):
    """每个学习率各跑一次，返回 (最好的学习率, 它的结果, 全部结果)。"""
    results = [(lr, run(dict(cfg, lr=lr))) for lr in lrs]
    ok = [x for x in results if not x[1]["diverged"]]
    lr, r = min(ok, key=lambda x: x[1]["val_loss"]) if ok else results[-1]
    return lr, r, results


def fmt(v: float) -> str:
    return "  nan" if not math.isfinite(v) else f"{v:5.3f}"


if __name__ == "__main__":
    t0 = time.time()
    print(f"网络：{BLOCKS} 块 × 2 个线性层，宽 {WIDTH}；每步 {BATCH} 个新样本，共 {STEPS} 步")
    print(f"随机猜的交叉熵约 ln10 = {math.log(10):.2f}\n")
    print("一、逐个加上技巧（SGD 行在 0.01/0.03/0.1 中取最好的学习率）")
    print(f"  {'配置':<34s}{'学习率':>8s}{'验证损失':>9s}{'准确率':>8s}   结论")
    for name, cfg, lrs in LADDER:
        lr, r, allr = best_of(cfg, lrs)
        detail = "" if len(allr) == 1 else "   [" + "  ".join(
            f"{x:g}→{fmt(y['val_loss']).strip()}" for x, y in allr) + "]"
        print(f"  {name:<30s}{lr:>10.3g}{fmt(r['val_loss']):>10s}{fmt(r['val_acc']):>9s}   {verdict(r)}{detail}")
    print("\n二、从全套里每次只拿掉一个")
    for name, cfg in LEAVE_ONE_OUT:
        r = run(cfg)
        print(f"  {name:<30s}{cfg['lr']:>10.3g}{fmt(r['val_loss']):>10s}{fmt(r['val_acc']):>9s}   {verdict(r)}")
    print(f"\n用时 {time.time() - t0:.0f} 秒")
