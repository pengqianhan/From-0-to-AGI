"""Chapter 6 · Minimal code 6: one deep network; turn the methods for stable training on and off

Task: the input is a random 16-dimensional vector. A fixed "teacher network" gives the label
(10 classes). Each step samples a new batch (the data never runs out, as in pretraining),
so the training loss is near the true level. We keep 2048 more samples for validation.
Student network: 12 blocks (24 linear layers), width 32. All methods are written by hand:
  initialization, RMSNorm (Pre-Norm), residual connections, AdamW, warmup + cosine, gradient clipping.
Run: uv run python chapters/06-training-stability/code/06_ablation.py      (about 1.5 min)
"""

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

# ── Data: a fixed teacher network ──────────────────────────────────────────────
_tg = torch.Generator().manual_seed(1234)
T1 = torch.randn(D_IN, 64, generator=_tg) / math.sqrt(D_IN)
T2 = torch.randn(64, N_CLS, generator=_tg) / math.sqrt(64)


def make_batch(n: int, g: torch.Generator):
    x = torch.randn(n, D_IN, generator=g)
    y = (torch.tanh(2 * x @ T1) @ T2).argmax(1)
    return x, y


X_VAL, Y_VAL = make_batch(2048, torch.Generator().manual_seed(1))

# ── Configuration: each switch controls one method ─────────────────────────────
FULL = dict(
    init="kaiming",      # "kaiming": scale with fan_in; "std1": N(0, 1)
    residual=True,       # h ← h + f(h); if False, h ← f(h)
    norm=True,           # RMSNorm on the input of f (Pre-Norm), and again before the output head
    opt="adamw",         # "adamw" or "sgd"
    lr=3e-3,
    schedule="cosine",   # "cosine" (with warmup), "wsd" (with warmup), or "const"
    warmup=100,
    clip=1.0,            # maximum global gradient norm; None means no clipping
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
            s2 /= math.sqrt(2 * BLOCKS)          # residual-branch output projection: smaller for more layers (see 03)
        blocks.append(dict(w1=w(WIDTH, HIDDEN, s1), w2=w(HIDDEN, WIDTH, s2),
                           g=torch.ones(WIDTH, requires_grad=True)))
    return dict(inp=w(D_IN, WIDTH, 1 / math.sqrt(D_IN)), blocks=blocks,
                g_final=torch.ones(WIDTH, requires_grad=True),
                out=w(WIDTH, N_CLS, 1 / math.sqrt(WIDTH)))


def parameters(model, cfg):
    """Return [(name, tensor)]. Without normalization, γ is not used, so the optimizer does not get it."""
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
        z = norm.rms_norm(h, b["g"]) if cfg["norm"] else h     # Pre-Norm: normalize before the branch
        f = torch.relu(z @ b["w1"]) @ b["w2"]
        h = h + f if cfg["residual"] else torch.relu(f)       # residual: h + f(h)
    if cfg["norm"]:
        h = norm.rms_norm(h, model["g_final"])                 # the final norm before the output head
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
    """Train for n_steps steps. Return (loss, gradient norm before clipping) for each step.
    Stop early if training diverges (NaN/inf)."""
    ps = parameters(model, cfg)
    hist = []
    b1, b2 = 0.9, 0.95
    for _ in range(n_steps):
        t = state["step"]
        x, y = make_batch(BATCH, state["data"])
        if t in bad_steps:
            x = x * 30.0                                        # simulate a bad batch (used by 07)
        loss = F.cross_entropy(forward(model, x, cfg), y)
        for _, p in ps:
            p.grad = None
        loss.backward()
        grads = [p.grad for _, p in ps]
        if cfg["clip"]:
            gnorm = sched.clip_by_global_norm(grads, cfg["clip"])   # gradient clipping
        else:
            gnorm = math.sqrt(sum(float((g * g).sum()) for g in grads))
        if not (math.isfinite(loss.item()) and math.isfinite(gnorm)):
            hist.append((float("nan"), float("nan")))
            break
        lr = lr_of_step(t)
        with torch.no_grad():
            for i, (_name, p) in enumerate(ps):
                g = p.grad
                if cfg["opt"] == "sgd":
                    p -= lr * g                                     # θ ← θ − η·g
                    continue
                state["m"][i].mul_(b1).add_(g, alpha=1 - b1)        # m ← β₁m + (1−β₁)g
                state["v"][i].mul_(b2).addcmul_(g, g, value=1 - b2) # v ← β₂v + (1−β₂)g²
                m_hat = state["m"][i] / (1 - b1 ** (t + 1))
                v_hat = state["v"][i] / (1 - b2 ** (t + 1))
                if p.dim() >= 2:                                    # decay only the matrices, not γ
                    p.mul_(1 - lr * cfg["wd"])                      # decoupled weight decay
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
        return "diverged (NaN)"
    if r["val_loss"] > 2.0:
        return "did not learn"
    if r["val_loss"] > 1.0:
        return "learns slowly"
    return "success"


BASE = dict(FULL, init="std1", residual=False, norm=False, opt="sgd", schedule="const",
            warmup=0, clip=None)
LADDER = [  # (name, configuration, learning rates to try); each SGD row uses the best of 3
    ("A plain net (std=1 init + SGD)", BASE, [0.01, 0.03, 0.1]),
    ("B + Kaiming init", dict(BASE, init="kaiming"), [0.01, 0.03, 0.1]),
    ("C + residual connections", dict(BASE, init="kaiming", residual=True), [0.01, 0.03, 0.1]),
    ("D + RMSNorm (Pre-Norm)", dict(BASE, init="kaiming", residual=True, norm=True),
     [0.01, 0.03, 0.1]),
    ("E SGD → AdamW", dict(FULL, schedule="const", warmup=0, clip=None), [3e-3]),
    ("F + warmup + cosine decay", dict(FULL, clip=None), [3e-3]),
    ("G + gradient clipping (= full)", FULL, [3e-3]),
]
LEAVE_ONE_OUT = [
    ("Full, but no residual", dict(FULL, residual=False)),
    ("Full, but no RMSNorm", dict(FULL, norm=False)),
    ("Full, but init std=1", dict(FULL, init="std1")),
    ("Full, but no warmup or decay", dict(FULL, schedule="const", warmup=0)),
]


def best_of(cfg, lrs):
    """Run once for each learning rate. Return (best learning rate, its result, all results)."""
    results = [(lr, run(dict(cfg, lr=lr))) for lr in lrs]
    ok = [x for x in results if not x[1]["diverged"]]
    lr, r = min(ok, key=lambda x: x[1]["val_loss"]) if ok else results[-1]
    return lr, r, results


def fmt(v: float) -> str:
    return "  nan" if not math.isfinite(v) else f"{v:5.3f}"


if __name__ == "__main__":
    t0 = time.time()
    print(f"Network: {BLOCKS} blocks × 2 linear layers, width {WIDTH}; {BATCH} new samples per step, {STEPS} steps")
    print(f"Cross-entropy of a random guess: about ln10 = {math.log(10):.2f}\n")
    print("1. Add the methods one at a time (each SGD row uses the best learning rate of 0.01/0.03/0.1)")
    print(f"  {'Configuration':<34s}{'LR  ':>8s}{'Val loss ':>9s}{'Acc':>8s}   Result")
    for name, cfg, lrs in LADDER:
        lr, r, allr = best_of(cfg, lrs)
        detail = "" if len(allr) == 1 else "   [" + "  ".join(
            f"{x:g}→{fmt(y['val_loss']).strip()}" for x, y in allr) + "]"
        print(f"  {name:<30s}{lr:>10.3g}{fmt(r['val_loss']):>10s}{fmt(r['val_acc']):>9s}   {verdict(r)}{detail}")
    print("\n2. Remove one method at a time from the full set")
    for name, cfg in LEAVE_ONE_OUT:
        r = run(cfg)
        print(f"  {name:<30s}{cfg['lr']:>10.3g}{fmt(r['val_loss']):>10s}{fmt(r['val_acc']):>9s}   {verdict(r)}")
    print(f"\nTime: {time.time() - t0:.0f} s")
