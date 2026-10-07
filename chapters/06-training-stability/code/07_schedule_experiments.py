"""Chapter 6 · Minimal code 7: when do the learning-rate schedule, warmup, and gradient clipping help?

Three groups of experiments with the network and the training loop of 06:
  1. cosine vs WSD vs a constant learning rate. Also, WSD can "finish at any time": from one
     stable run, branch off a 100-step decay at step 400 and at step 800, and compare with
     cosine runs whose total length is set before the start;
  2. high-learning-rate stress test: what occurs without warmup, or without RMSNorm;
  3. gradient clipping: 5 bad batches (inputs × 30) in the middle of training.
Run: uv run python chapters/06-training-stability/code/07_schedule_experiments.py   (about 1.5 min)
"""

import copy
import importlib.util
import math
import time
from pathlib import Path

import torch

torch.set_num_threads(1)

_spec = importlib.util.spec_from_file_location("ablation", Path(__file__).with_name("06_ablation.py"))
ab = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ab)
sched = ab.sched


def clone(model, state):
    """Copy the model and the optimizer state (also the random state of the data stream),
    so that several branches can start from the same checkpoint."""
    g = torch.Generator()
    g.set_state(state["data"].get_state())
    st = dict(step=state["step"], data=g, m=[t.clone() for t in state["m"]],
              v=[t.clone() for t in state["v"]])
    return copy.deepcopy(model), st


def wsd_branching(cfg=ab.FULL, decay=100, branch_at=(400, 800)):
    """One stable run. At each point in branch_at, branch off a linear decay of `decay` steps.

    Return {branch point: (val loss at the end of stable, val loss after the decay,
    training-loss curve of the decay)}, and the training-loss curve of the stable trunk.
    """
    peak, warm = cfg["lr"], cfg["warmup"]
    model = ab.init_model(cfg)
    state = ab.new_state(model, cfg)
    stable_lr = lambda s: peak * min(1.0, (s + 1) / warm)          # noqa: E731
    out, trunk, done = {}, [], 0
    for b in branch_at:
        trunk += ab.train_steps(model, state, cfg, stable_lr, b - done)
        done = b
        m2, s2 = clone(model, state)
        before = ab.evaluate(m2, cfg)[0]
        total = b + decay
        lr_decay = lambda s, total=total: sched.wsd(s, total, peak, warm, decay_frac=decay / total)  # noqa: E731
        tail = ab.train_steps(m2, s2, cfg, lr_decay, decay)
        out[b] = (before, ab.evaluate(m2, cfg)[0], [h[0] for h in tail])
    return out, [h[0] for h in trunk]


def cosine_run(total: int, cfg=ab.FULL):
    r = ab.run(dict(cfg, schedule="cosine"), steps=total)
    return r["val_loss"], r["losses"]


def stress(cfg, lr):
    r = ab.run(dict(cfg, lr=lr))
    early = [x for x in r["losses"][:150] if math.isfinite(x)]
    return max(early) if early else float("nan"), r


if __name__ == "__main__":
    t0 = time.time()
    fmt = ab.fmt
    print("1. Learning-rate schedules (full configuration, 800 steps)")
    for name in ["cosine", "wsd", "const"]:
        r = ab.run(dict(ab.FULL, schedule=name))
        print(f"  {name:<7s} val loss {fmt(r['val_loss'])}")

    print("\n  WSD can finish at any time: one stable trunk, with a 100-step decay branch at step 400 and 800")
    branches, _ = wsd_branching()
    for b, (before, after, _) in branches.items():
        cos, _ = cosine_run(b + 100)
        print(f"    total {b + 100:>4d} steps: end of stable {fmt(before)} → after decay {fmt(after)}"
              f"   reference: a separate {b + 100}-step cosine run {fmt(cos)}")
    print("    WSD trains 400+100+400+100 = 1000 steps in total; two cosine runs need 500+900 = 1400 steps")

    print("\n2. High-learning-rate stress test: η = 0.3 (normal: 0.003)")
    print("  max training loss in the first 150 steps / val loss after 800 steps")
    rows = [("full", ab.FULL), ("no warmup", dict(ab.FULL, warmup=0)),
            ("no RMSNorm", dict(ab.FULL, norm=False))]
    for name, cfg in rows:
        peak, r = stress(cfg, lr=0.3)
        peak_s = fmt(peak) if peak < 1e3 else f"{peak:.1e}"
        print(f"  {name:<12s}{peak_s:>10s} / {fmt(r['val_loss'])}" + (" (diverged)" if r["diverged"] else ""))

    print("\n3. Gradient clipping: 5 bad batches at steps 300–304 (inputs × 30)")
    bad = tuple(range(300, 305))
    for norm_on in [True, False]:
        for clip in [1.0, None]:
            cfg = dict(ab.FULL, norm=norm_on, clip=clip)
            r = ab.run(cfg, bad_steps=bad)
            after = max(r["losses"][305:345])
            print(f"  RMSNorm {'on' if norm_on else 'off'}, clipping {'on' if clip else 'off'}: "
                  f"max training loss in the 40 steps after the bad data {fmt(after)}, "
                  f"final val loss {fmt(r['val_loss'])}"
                  f", max gradient norm at the bad steps {max(r['grad_norms'][300:305]):.1f}")
    print(f"\nTime: {time.time() - t0:.0f} s")
