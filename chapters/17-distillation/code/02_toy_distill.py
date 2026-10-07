"""Chapter 17 · Minimal code 2: a toy distillation experiment. Hard labels vs sequence-level KD vs logits KD

Task: a character-level language model on Shakespeare text (assets/tiny_corpus/shakespeare.txt).
The model sees the previous 8 characters and predicts the next character.

- Teacher: a large MLP (about 430k parameters), trained on all training text (about 1M characters).
- Student: a small MLP (about 9k parameters). It **sees only 20k characters** of training text
  (this simulates "the student does not have enough data").
  Same initialization, same number of steps, same batch size. We compare four training signals:
    A. Hard labels: the true next character (one-hot cross-entropy). This is normal pretraining / SFT.
    B. Sequence-level KD: the teacher **writes** 20k characters. The student trains with normal
       cross-entropy on the teacher text (= "SFT on teacher data", the method of DeepSeek-R1-Distill).
    C. Logits KD: at each position of the same 20k characters, the student matches the full
       distribution of the teacher, τ²·KL(p_T^τ ‖ p_S^τ).
    D. A mix of C and A: 0.5·hard label + 0.5·KD (the zero default is kd_alpha = 0.5).
  We also give a reference R: the same student, but it sees all 1M characters of real text
  (the upper limit when "there is enough data").
- Metric: cross-entropy on held-out real validation text, in bits/char (lower is better). 3 random seeds.

Run: uv run python chapters/17-distillation/code/02_toy_distill.py      (about 2 min of single-thread CPU time; on a busy machine the wall-clock time is much longer)
"""

import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(1)
ROOT = Path(__file__).resolve().parents[3]
CORPUS = ROOT / "assets" / "tiny_corpus" / "shakespeare.txt"

CTX = 8  # look at the previous 8 characters
SMALL_N = 20_000  # number of training characters that the student can see
TEACHER = dict(emb=32, hidden=512, layers=2)
STUDENT = dict(emb=8, hidden=64, layers=1)
TEACHER_STEPS, STUDENT_STEPS, BATCH = 3000, 2000, 256
TAU = 2.0
SEEDS = [0, 1, 2]


class MLPLM(nn.Module):
    """The simplest "language model": concatenate the embeddings of the previous CTX characters,
    apply a few MLP layers, and output the logits of the next character."""

    def __init__(self, vocab: int, emb: int, hidden: int, layers: int) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocab, emb)
        dims = [CTX * emb] + [hidden] * layers
        self.body = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:]))
        self.head = nn.Linear(hidden, vocab)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, CTX) → (B, V)
        h = self.emb(x).flatten(1)
        for lin in self.body:
            h = torch.tanh(lin(h))
        return self.head(h)


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def windows(ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Cut a sequence of character ids into training samples (CTX context characters, next character)."""
    x = ids.unfold(0, CTX, 1)[:-1]
    y = ids[CTX:]
    return x, y


@torch.no_grad()
def bits_per_char(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    model.eval()
    ce = F.cross_entropy(model(x), y).item()
    model.train()
    return ce / math.log(2)


def kd_loss(z_s: torch.Tensor, z_t: torch.Tensor, tau: float) -> torch.Tensor:
    """τ²·KL(p_T^τ ‖ p_S^τ), mean over the batch (the same formula as 01 and zero.post.distill.kd_loss)."""
    logp_t = F.log_softmax(z_t / tau, -1)
    logp_s = F.log_softmax(z_s / tau, -1)
    return (logp_t.exp() * (logp_t - logp_s)).sum(-1).mean() * tau * tau


def train(model: nn.Module, x: torch.Tensor, y: torch.Tensor | None, steps: int, seed: int,
          t_logits: torch.Tensor | None = None, alpha: float = 0.0, lr: float = 3e-3) -> nn.Module:
    """alpha = 0: hard labels only; alpha = 1: KD only; between: a mix. Each step samples a random batch from (x, y)."""
    g = torch.Generator().manual_seed(1000 + seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    for step in range(steps):
        idx = torch.randint(0, len(x), (BATCH,), generator=g)
        z = model(x[idx])
        loss = torch.zeros(())
        if alpha < 1:
            loss = loss + (1 - alpha) * F.cross_entropy(z, y[idx])
        if alpha > 0:
            loss = loss + alpha * kd_loss(z, t_logits[idx], TAU)
        for pg in opt.param_groups:  # cosine decay to 10%
            pg["lr"] = lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / steps)))
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model


@torch.no_grad()
def teacher_write(teacher: nn.Module, prompt: torch.Tensor, n: int, seed: int) -> torch.Tensor:
    """The teacher "writes" n characters autoregressively (sampling at temperature 1). It starts from real text."""
    teacher.eval()
    g = torch.Generator().manual_seed(seed)
    # Write 20 pieces in parallel, n/20 characters each, to save time
    k = 20
    starts = torch.randint(0, len(prompt) - CTX, (k,), generator=g)
    ctx = torch.stack([prompt[s:s + CTX] for s in starts])
    seqs = [[] for _ in range(k)]
    for _ in range(n // k):
        p = F.softmax(teacher(ctx), -1)
        nxt = torch.multinomial(p, 1, generator=g)
        for i in range(k):
            seqs[i].append(int(nxt[i]))
        ctx = torch.cat([ctx[:, 1:], nxt], 1)
    out = [c for s in seqs for c in s]
    return torch.tensor(out)


def run_experiment(log=print) -> dict:
    """Run the full experiment and return a dict of results (the video scenes.py also calls it and caches the result in video/out/cache.json)."""
    t0 = time.time()
    text = CORPUS.read_text()
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    ids = torch.tensor([stoi[c] for c in text])
    n_train = int(0.9 * len(ids))
    train_ids, val_ids = ids[:n_train], ids[n_train:]
    x_full, y_full = windows(train_ids)
    x_val, y_val = windows(val_ids)
    x_small, y_small = windows(train_ids[:SMALL_N])
    V = len(chars)
    log(f"Corpus {len(text):,} characters, {V} distinct characters; train {n_train:,}, validation {len(val_ids):,}; the student sees only the first {SMALL_N:,}")

    # ── Teacher ─────────────────────────────────────────────────────────
    torch.manual_seed(0)
    teacher = MLPLM(V, **TEACHER)
    train(teacher, x_full, y_full, TEACHER_STEPS, seed=0)
    teacher.eval()
    t_bpc = bits_per_char(teacher, x_val, y_val)
    log(f"Teacher: {n_params(teacher):,} parameters, {TEACHER_STEPS} steps on all training text → validation {t_bpc:.3f} bits/char")

    with torch.no_grad():
        t_logits_small = teacher(x_small)  # teacher logits at each position of the 20k student characters (for C and D)
    written = teacher_write(teacher, train_ids[:SMALL_N], SMALL_N, seed=123)
    x_seq, y_seq = windows(written)  # for B: the text that the teacher wrote
    sample = "".join(chars[i] for i in written[:160].tolist())
    log(f"Text written by the teacher (first 160 characters): {sample!r}")

    # ── Student: same initialization, steps, and batch size; only the training signal changes ──
    arms = {
        "A hard label (20k real chars)": dict(x=x_small, y=y_small),
        "B seq-level KD (teacher text)": dict(x=x_seq, y=y_seq),
        f"C logits KD (τ={TAU:g}, same 20k)": dict(x=x_small, y=y_small, t_logits=t_logits_small, alpha=1.0),
        "D 0.5·hard label + 0.5·KD": dict(x=x_small, y=y_small, t_logits=t_logits_small, alpha=0.5),
        "R reference (hard, all 1M)": dict(x=x_full, y=y_full),
    }
    results: dict[str, list[tuple[float, float]]] = {k: [] for k in arms}
    for seed in SEEDS:
        for name, kw in arms.items():
            torch.manual_seed(seed)
            student = MLPLM(V, **STUDENT)
            train(student, kw["x"], kw["y"], STUDENT_STEPS, seed, kw.get("t_logits"), kw.get("alpha", 0.0))
            tr = bits_per_char(student, x_small, y_small)
            va = bits_per_char(student, x_val, y_val)
            results[name].append((tr, va))
    s_params = n_params(MLPLM(V, **STUDENT))
    return {
        "teacher_params": n_params(teacher), "student_params": s_params, "teacher_bpc": t_bpc,
        "uniform_bpc": math.log2(V), "vocab": V, "sample": sample, "seeds": SEEDS,
        "steps": STUDENT_STEPS, "batch": BATCH, "tau": TAU,
        "arms": {name: {"train": [r[0] for r in rs], "val": [r[1] for r in rs]} for name, rs in results.items()},
        "seconds": time.time() - t0,
    }


def main() -> None:
    r = run_experiment()
    print(f"\nStudent: {r['student_params']:,} parameters, {r['steps']} steps × batch {r['batch']} for each signal, seeds {r['seeds']}")
    print(f"{'training signal':<30}{'on 20k train':>12}{'val bits/char':>16}  per seed (val)")
    for name, a in r["arms"].items():
        tr = sum(a["train"]) / len(a["train"])
        va = sum(a["val"]) / len(a["val"])
        per = " / ".join(f"{x:.3f}" for x in a["val"])
        print(f"{name:<30}{tr:>12.3f}{va:>16.3f}  {per}")
    print(f"(teacher validation {r['teacher_bpc']:.3f}; uniform random guess log2({r['vocab']}) = {r['uniform_bpc']:.3f})")
    print(f"Time {r['seconds']:.0f}s")


if __name__ == "__main__":
    main()
