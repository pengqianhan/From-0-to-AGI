"""Chapter 25 · Minimal code 5: multi-token prediction (MTP). Learn one more token (the token
after next), then use it as a draft.

A DeepSeek-V3-style MTP module (depth D = 1) after the small model of Chapter 10:
    h'_i = W [RMSNorm(Emb(t_{i+1})) ; RMSNorm(h_i)]      # main-model state at i + embedding of token i+1
    h''  = Block(h')                                     # one Transformer block (causal attention)
    P(t_{i+2}) = softmax(RMSNorm(h''_i) · Embᵀ)          # embedding / output head shared with the main model
    L = L_main + λ · L_MTP                               # λ = 0.3 (DeepSeek-V3 value for the first 10T tokens)

Training: the same initialization, data order, and hyperparameters as Chapter 10 (600 steps).
The only addition is the MTP loss. The cached models of Chapter 10 (seeds 0 and 1) are the
control group with λ = 0.
Inference: the MTP module is the draft for self-speculative decoding. Each forward pass of the main
model verifies 1 draft and gives 1 more token.

Run: uv run python chapters/25-mtp-speculative-decoding/code/05_mtp.py
(The first run trains two models with MTP, about 2–3 minutes each with 1 thread.
Later runs read the cache code/out/*.pt.)
"""

from __future__ import annotations

import importlib.util
import math
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(1)
HERE = Path(__file__).resolve().parent
OUT = HERE / "out"


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


m1 = _load("ch25_models", "01_models_and_cost.py")
ch10, KVCache, truncate = m1.ch10, m1.KVCache, m1.truncate


class MTPHead(nn.Module):
    """One MTP module: two RMSNorms → concatenate and project 2d → d → one Transformer block → RMSNorm."""

    def __init__(self, c) -> None:
        super().__init__()
        self.enorm, self.hnorm = ch10.RMSNorm(c.dim), ch10.RMSNorm(c.dim)
        self.proj = nn.Linear(2 * c.dim, c.dim, bias=False)
        self.block = ch10.Block(c)
        self.norm = ch10.RMSNorm(c.dim)

    def forward(self, h, emb_next, cos, sin, cache=None):
        x = self.proj(torch.cat([self.enorm(emb_next), self.hnorm(h)], dim=-1))
        return self.norm(self.block(x, cos, sin, cache, 0))


def hidden(lm, ids, cache=None):
    """The main-model state after its last RMSNorm (multiply by Embᵀ to get the logits).

    The same as TinyLM.forward, but without the output head."""
    T = ids.shape[1]
    start = len(cache) if cache is not None else 0
    cos, sin = lm.cos[start : start + T], lm.sin[start : start + T]
    x = lm.emb(ids)
    for layer, blk in enumerate(lm.blocks):
        x = blk(x, cos, sin, cache, layer)
    return lm.norm(x)


def mtp_forward(lm, head, x):
    """x: (B, T) → main-model logits (B, T, V), MTP logits (B, T−1, V). At position i, MTP predicts x[i+2]."""
    h = hidden(lm, x)
    T = x.shape[1]
    hm = head(h[:, :-1], lm.emb(x[:, 1:]), lm.cos[: T - 1], lm.sin[: T - 1])
    return h @ lm.emb.weight.T, hm @ lm.emb.weight.T


def train_mtp(
    seed: int, lam: float = 0.3, steps: int = 600, bsz: int = 16, seq: int = 64, lr: float = 3e-3
):
    """The same initialization / data / learning rate as ch10.train. The only addition is λ · L_MTP."""
    data = ch10.CharData()
    c = ch10.Config(vocab_size=data.vocab_size)
    torch.manual_seed(seed)
    lm = ch10.TinyLM(c)  # Build the main model first: the same RNG use as ch10.train gives the same initial weights
    head = MTPHead(c)  # The linear layers use the PyTorch default initialization, as the Chapter 10 Block does
    params = list(lm.parameters()) + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.1)
    g = torch.Generator().manual_seed(seed)
    t0 = time.time()
    for step in range(steps + 1):
        for pg in opt.param_groups:
            pg["lr"] = lr * min(1, (step + 1) / 100) * 0.5 * (1 + math.cos(math.pi * step / steps))
        x, y = data.batch("train", bsz, seq, g)
        logits, mlogits = mtp_forward(lm, head, x)
        l_main = F.cross_entropy(logits.flatten(0, 1), y.flatten())
        l_mtp = F.cross_entropy(mlogits.flatten(0, 1), y[:, 1:].flatten())  # Target: the character after next
        loss = l_main + lam * l_mtp
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if step % 200 == 0:
            print(
                f"  step {step:4d}  main loss {l_main.item():.3f}  MTP loss {l_mtp.item():.3f}  "
                f"({time.time() - t0:.0f}s)"
            )
    return lm.eval(), head.eval()


def load_mtp(seed: int):
    c = ch10.Config(vocab_size=ch10.CharData().vocab_size)
    path = OUT / f"mtp_lam0.3_seed{seed}.pt"
    lm, head = ch10.TinyLM(c), MTPHead(c)
    if path.exists():
        sd = torch.load(path, weights_only=True)
        lm.load_state_dict(sd["lm"])
        head.load_state_dict(sd["head"])
        return lm.eval(), head.eval()
    print(f"Training the model with MTP (seed {seed}, 600 steps, only once)")
    lm, head = train_mtp(seed)
    OUT.mkdir(exist_ok=True)
    torch.save({"lm": lm.state_dict(), "head": head.state_dict()}, path)
    return lm, head


@torch.no_grad()
def evaluate(lm, head=None, n_batches: int = 20, seq: int = 64):
    """Validation set: main-model loss / next-character accuracy.

    With MTP, also report the MTP accuracy for the character after next."""
    data = ch10.CharData()
    g = torch.Generator().manual_seed(1234)  # The same validation batches as ch10.val_loss
    loss = acc1 = acc2 = 0.0
    for _ in range(n_batches):
        x, y = data.batch("val", 32, seq, g)
        if head is None:
            logits = lm(x)
        else:
            logits, mlogits = mtp_forward(lm, head, x)
            acc2 += (mlogits.argmax(-1) == y[:, 1:]).float().mean().item()
        loss += F.cross_entropy(logits.flatten(0, 1), y.flatten()).item()
        acc1 += (logits.argmax(-1) == y).float().mean().item()
    return loss / n_batches, acc1 / n_batches, acc2 / n_batches


@torch.no_grad()
def greedy(lm, prompt, n_new):
    cache = KVCache(lm.c.n_layers)
    logits = lm(torch.tensor([prompt]), cache)[0, -1]
    out = []
    for _ in range(n_new):
        out.append(int(logits.argmax()))
        logits = lm(torch.tensor([[out[-1]]]), cache)[0, -1]
    return out


@torch.no_grad()
def mtp_self_speculative(lm, head, prompt, n_new):
    """Greedy self-speculation: in each round, MTP guesses 1 token. One forward pass of the main model
    verifies it and gives 1 more token. Returns (new tokens, number of rounds, number accepted)."""
    mc, hc = KVCache(lm.c.n_layers), KVCache(1)
    h = hidden(lm, torch.tensor([prompt]), mc)[0]  # prefill
    seq = list(prompt) + [int((h[-1] @ lm.emb.weight.T).argmax())]
    pending, rounds, accepted = h, 0, 0  # pending: main-model states that MTP did not get yet
    while len(seq) - len(prompt) < n_new:
        L = len(seq)
        # At position i, MTP needs h_i and seq[i+1]: feed len(hc)..L−2. The output at the last position is the draft.
        s = len(hc)
        hm = head(
            pending[None],
            lm.emb(torch.tensor([seq[s + 1 : L]])),
            lm.cos[s : L - 1],
            lm.sin[s : L - 1],
            hc,
        )
        d = int((hm[0, -1] @ lm.emb.weight.T).argmax())
        # One forward pass of the main model: [seq[L−1], d] → predictions at two positions
        hn = hidden(lm, torch.tensor([[seq[L - 1], d]]), mc)
        choice = (hn[0] @ lm.emb.weight.T).argmax(-1).tolist()
        ok = choice[0] == d
        seq += [d, choice[1]] if ok else [choice[0]]
        rounds, accepted = rounds + 1, accepted + ok
        truncate(mc, len(seq) - 1)  # If the draft is rejected, discard its K/V
        pending = hn[0, : 1 + ok]
    return seq[len(prompt) :][:n_new], rounds, accepted


if __name__ == "__main__":
    m2 = _load("ch25_greedy", "02_greedy_speculative.py")
    print(f"{'model':<22} {'val loss':>9} {'acc. (next char)':>13} {'MTP acc (after next)':>17}")
    rows = {}
    for seed in (0, 1):
        base = ch10.load_or_train(n_kv_heads=4, steps=600, seed=seed, verbose=True)
        lm, head = load_mtp(seed)
        rows[seed] = (evaluate(base), evaluate(lm, head))
        (lb, ab, _), (lm_, am, a2) = rows[seed]
        print(f"{'λ=0 (ch. 10) seed ' + str(seed):<22} {lb:9.3f} {ab:16.3f} {'—':>20}")
        print(f"{'λ=0.3 (+MTP) seed ' + str(seed):<22} {lm_:9.3f} {am:16.3f} {a2:20.3f}")
    print(
        "(The two rows of one seed have the same initialization and data order; only the MTP loss is different."
        "\n  Chapter 10 found that a different seed changes the loss by 0.03. At this scale, compare the effect"
        "\n  of MTP on the main model with this noise.)"
    )

    lm, head = load_mtp(0)
    P, N = m2.prompts(), 200
    ref = [greedy(lm, p, N) for p in P]
    outs = [mtp_self_speculative(lm, head, p, N) for p in P]
    same = all(o[0] == r for o, r in zip(outs, ref))
    rounds = sum(o[1] for o in outs)
    acc = sum(o[2] for o in outs)
    print(f"\nMTP self-speculative decoding (seed 0, greedy, 4 prompts × {N} characters): identical to greedy decoding of the main model: {same}")
    print(
        f"draft acceptance rate {acc / rounds:.3f}, {rounds} forward passes of the main model (normal decoding: {len(P) * N}), "
        f"{(len(P) * N) / rounds:.2f} characters per forward pass"
    )
    tb, ts = [], []
    for _ in range(3):
        t0 = time.process_time()
        for p in P:
            greedy(lm, p, N)
        tb.append(time.process_time() - t0)
        t0 = time.process_time()
        for p in P:
            mtp_self_speculative(lm, head, p, N)
        ts.append(time.process_time() - t0)
    tb, ts = statistics.median(tb), statistics.median(ts)
    print(f"CPU time: normal greedy {tb:.2f} s, MTP self-speculation {ts:.2f} s, speedup {tb / ts:.2f}×")
    n_head = sum(p.numel() for p in head.parameters())
    n_lm = sum(p.numel() for p in lm.parameters())
    print(
        f"MTP module parameters {n_head:,} ({n_head / n_lm:.0%} of the main model {n_lm:,}; the shared embedding and output head are not counted again)"
    )
