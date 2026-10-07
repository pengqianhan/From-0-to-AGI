"""Chapter 6 · From minimal code to production code: the standard PyTorch code for the same methods

This is the same "full set" as in 06_ablation.py. Each hand-written part becomes the PyTorch version:
  hand-written rms_norm              → nn.RMSNorm
  hand-written AdamW loop            → torch.optim.AdamW (two parameter groups: decay the
                                       matrices, do not decay γ)
  hand-written warmup_cosine / wsd   → torch.optim.lr_scheduler.LambdaLR
  hand-written clip_by_global_norm   → torch.nn.utils.clip_grad_norm_
First, run 200 steps with the same initial weights and the same data as the minimal code,
and compare the loss at each step (a parity check).
Then do one full training run with the initialization that large models often use:
"std=0.02, and a smaller residual output projection for more layers".
Run: uv run python chapters/06-training-stability/code/08_pytorch_version.py
"""

import importlib.util
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

torch.set_num_threads(1)



def load_ablation(dtype=torch.float32):
    """Load 06_ablation.py. With dtype=float64, it makes its data and weights in double precision
    (for the parity check)."""
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
    """Pre-Norm residual block: x + W2·ReLU(W1·RMSNorm(x)). The Transformer block in Chapter 9
    has the same structure."""

    def __init__(self, width: int, hidden: int):
        super().__init__()
        self.norm = nn.RMSNorm(width, eps=1e-6)
        self.fc1 = nn.Linear(width, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, width, bias=False)   # output projection of the residual branch

    def forward(self, x):
        return x + self.fc2(F.relu(self.fc1(self.norm(x))))


class DeepNet(nn.Module):
    def __init__(self, d_in=ab.D_IN, width=ab.WIDTH, hidden=ab.HIDDEN, blocks=ab.BLOCKS,
                 n_cls=ab.N_CLS):
        super().__init__()
        self.inp = nn.Linear(d_in, width, bias=False)
        self.blocks = nn.ModuleList(Block(width, hidden) for _ in range(blocks))
        self.norm = nn.RMSNorm(width, eps=1e-6)           # final norm
        self.out = nn.Linear(width, n_cls, bias=False)

    def forward(self, x):
        h = self.inp(x)
        for b in self.blocks:
            h = b(h)
        return self.out(self.norm(h))


def init_llm_style(model: DeepNet, std: float = 0.02):
    """The usual method in large models: all matrices N(0, 0.02²); the output projection of
    each residual branch is also divided by √(2·number of layers)."""
    n = len(model.blocks)
    for name, p in model.named_parameters():
        if p.dim() >= 2:
            s = std / math.sqrt(2 * n) if name.endswith("fc2.weight") else std
            nn.init.normal_(p, mean=0.0, std=s)


def copy_from_minimal(model: DeepNet, m: dict):
    """Copy the initial weights of the hand-written version in 06 (nn.Linear stores the transpose)."""
    with torch.no_grad():
        model.inp.weight.copy_(m["inp"].T)
        model.out.weight.copy_(m["out"].T)
        for blk, b in zip(model.blocks, m["blocks"]):
            blk.fc1.weight.copy_(b["w1"].T)
            blk.fc2.weight.copy_(b["w2"].T)


def make_optimizer(model: nn.Module, lr: float, wd: float = 0.1):
    """AdamW with two parameter groups: weight decay for matrices (2 or more dimensions),
    no weight decay for 1-dimensional parameters (the γ of RMSNorm, biases)."""
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    groups = [{"params": decay, "weight_decay": wd}, {"params": no_decay, "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.95), eps=1e-8)


def make_scheduler(opt, total: int, warmup: int, kind: str = "cosine"):
    """LambdaLR: at each step, set the learning rate to peak × lambda(step).
    The lambda uses the functions of 05 (with peak = 1)."""
    if kind == "cosine":
        f = lambda s: ab.sched.warmup_cosine(s, total, 1.0, warmup)  # noqa: E731
    else:
        f = lambda s: ab.sched.wsd(s, total, 1.0, warmup)            # noqa: E731
    return torch.optim.lr_scheduler.LambdaLR(opt, f)


def train(model, steps: int, cfg=ab.FULL, kind: str = "cosine", seed: int = 0, data_mod=ab):
    opt = make_optimizer(model, cfg["lr"], cfg["wd"])
    scheduler = make_scheduler(opt, steps, cfg["warmup"], kind)
    data = torch.Generator().manual_seed(seed + 100)       # the same data stream as the minimal code
    losses = []
    for _ in range(steps):
        x, y = data_mod.make_batch(ab.BATCH, data)
        loss = F.cross_entropy(model(x), y)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg["clip"])  # gradient clipping
        opt.step()
        scheduler.step()                                   # move the learning rate one step
        losses.append(loss.item())
    return losses


@torch.no_grad()
def evaluate(model, data_mod=ab):
    logits = model(data_mod.X_VAL)
    return F.cross_entropy(logits, data_mod.Y_VAL).item()


def cross_check(dtype, steps: int = 200):
    """Same initial weights, same data: compare the loss of the PyTorch version and the
    hand-written version of 06 at each step."""
    mod = load_ablation(dtype)
    old = torch.get_default_dtype()
    torch.set_default_dtype(dtype)                         # the full parity check uses this precision
    ref = mod.run(mod.FULL, steps=steps)                   # the hand-written version of 06
    model = DeepNet()
    copy_from_minimal(model, mod.init_model(mod.FULL))
    losses = train(model, steps, data_mod=mod)
    torch.set_default_dtype(old)
    diff = max(abs(a - b) for a, b in zip(losses, ref["losses"]))
    return diff, losses[-1], ref["losses"][-1]


if __name__ == "__main__":
    # ── Parity check: same initial weights, same data, 200 steps ─────────────────
    for dtype in [torch.float64, torch.float32]:
        diff, mine, ref = cross_check(dtype)
        print(f"Parity check, 200 steps ({str(dtype).replace('torch.', '')}): max loss difference per step {diff:.1e}; "
              f"loss at step 200: PyTorch {mine:.4f} / hand-written {ref:.4f}")

    model = DeepNet()
    opt = make_optimizer(model, 3e-3)
    n_dec = sum(p.numel() for p in opt.param_groups[0]["params"])
    n_nod = sum(p.numel() for p in opt.param_groups[1]["params"])
    print(f"Parameter groups: matrices with weight decay {n_dec} parameters, γ without weight decay {n_nod} parameters")

    # ── Initialization in the style of large models, full training of 800 steps ──
    for kind in ["cosine", "wsd"]:
        torch.manual_seed(0)
        model = DeepNet()
        init_llm_style(model)
        losses = train(model, ab.STEPS, kind=kind)
        print(f"std=0.02 init + {kind:<6s}: val loss after {ab.STEPS} steps {evaluate(model):.5f}")
