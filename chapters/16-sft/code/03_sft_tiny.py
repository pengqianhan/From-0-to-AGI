"""Chapter 16 · Minimal code 3: use SFT to turn a small base model that only continues text
into a model that writes tool calls.

Base model: the character-level small model of Chapter 10, trained on Shakespeare (0.86M
parameters). If there is no cache, the script first trains it (one or two minutes).
Steps:
  1. See how the base model "answers" a question: it only continues the text. It does not answer.
  2. Extend the vocabulary: add new characters ({ } " _ digits, ...) and 4 special tokens.
     The new rows start from the mean of the old embeddings.
  3. Full-parameter SFT with the same data and the same number of steps, two times: once with
     the loss only on assistant tokens (with mask), once with the loss on all tokens (no mask).
  4. Greedy generation on 50 questions with **cities and numbers that the model never saw**.
     Count the fraction with the correct format / correct function name / all arguments correct.
  5. Extra: train for the same number of steps on only 40 samples (about 60 epochs) to see
     overfitting.

The script trains each model only once. It caches the weights in code/out/ (.gitignore ignores *.pt).
Run: uv run python chapters/16-sft/code/03_sft_tiny.py
(The first run trains 3 models, 300 steps each. On the heavily loaded build machine, this took
about 17 minutes in total. On an idle laptop, it is much faster. After that, the script loads the
models from the cache, and only generates and scores.)
"""

from __future__ import annotations

import importlib.util
import json
import math
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

torch.set_num_threads(1)  # the build machine shares its CPU between many jobs; on your computer, you can remove this line
HERE = Path(__file__).resolve().parent
OUT = HERE / "out"


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


ch10 = load("ch10_tiny_model", HERE.parents[1] / "10-inference" / "code" / "01_tiny_model.py")
lm = load("ch16_loss_mask", HERE / "02_loss_mask.py")

STEPS, BSZ, LR, WARMUP = 300, 8, 1e-3, 20
N_TRAIN, N_SMALL, N_TEST = 2000, 40, 50


# ── Base model + vocabulary extension ───────────────────────────────────────────
def base_model():
    return ch10.load_or_train(4)  # the small MHA model of Chapter 10 (loads from the cache if it exists)


def extend_vocab(model, tok: lm.ChatTok):
    """Extend the vocabulary of the shared embedding from 65 to len(tok.itos).

    Each new token vector = the mean of the old vectors (+ a little noise). Then, at the start,
    the logits of the new tokens are not much too large or much too small.
    """
    old = model.emb.weight.data
    g = torch.Generator().manual_seed(0)
    new_rows = old.mean(0, keepdim=True) + 0.01 * torch.randn(len(tok.itos) - len(old), old.shape[1],
                                                              generator=g)
    model.emb = torch.nn.Embedding(len(tok.itos), old.shape[1])
    model.emb.weight.data = torch.cat([old, new_rows])
    model.c.vocab_size = len(tok.itos)
    return model


# ── Data ──────────────────────────────────────────────────────────────────────
def dataset(n: int, seed: int, split: str):
    rng = random.Random(seed)
    return [lm.make_example(rng, split) for _ in range(n)]


def batchify(tok, examples, use_mask: bool):
    """Pad on the right to make one batch. Padding positions and positions not in the loss get target -100."""
    rows = [lm.encode_with_mask(tok, ex["messages"]) for ex in examples]
    L = max(len(ids) for ids, _ in rows) - 1
    x = torch.zeros(len(rows), L, dtype=torch.long)
    y = torch.full((len(rows), L), -100, dtype=torch.long)
    for i, (ids, mask) in enumerate(rows):
        xi, yi = lm.masked_targets(ids, mask, use_mask)
        x[i, : len(xi)], y[i, : len(yi)] = xi, yi
    return x, y


@torch.no_grad()
def assistant_loss(model, tok, examples) -> float:
    """Validation metric: always on assistant tokens only, with or without a mask in training.

    Only then can we compare the two kinds of training.
    """
    x, y = batchify(tok, examples, use_mask=True)
    return F.cross_entropy(model(x).flatten(0, 1), y.flatten(), ignore_index=-100).item()


@torch.no_grad()
def loss_split(model, tok, examples) -> dict:
    """Split the assistant tokens into two groups and calculate the mean loss of each group.

    The groups: argument values (city names and numbers, copied from the question) / the rest
    (format, function name).
    """
    tot = {"args": [0.0, 0], "rest": [0.0, 0]}
    for ex in examples:
        ids, mask = lm.encode_with_mask(tok, ex["messages"])
        text = "".join(s for s, _, _ in lm.tmpl.render(ex["messages"]))
        arg_pos = set()
        start = text.index('"arguments": ')
        for v in ex["call"]["arguments"].values():
            v = json.dumps(v)
            v = v[1:-1] if v.startswith('"') else v  # a string value without its quotes
            p = text.index(v, start)
            a, b = len(tok.encode(text[:p])), len(tok.encode(text[: p + len(v)]))
            arg_pos |= set(range(a, b))
            start = p + len(v)
        logits = model(torch.tensor([ids[:-1]]))[0]
        nll = F.cross_entropy(logits, torch.tensor(ids[1:]), reduction="none")
        for t in range(1, len(ids)):  # position t-1 predicts token t
            if mask[t]:
                k = "args" if t in arg_pos else "rest"
                tot[k][0] += nll[t - 1].item()
                tot[k][1] += 1
    return {k: v[0] / v[1] for k, v in tot.items()} | {"n_args": tot["args"][1], "n_rest": tot["rest"][1]}


# ── Training ──────────────────────────────────────────────────────────────────
def sft(tok, train_set, val_set, use_mask: bool, steps: int = STEPS, seed: int = 0, log=print):
    torch.manual_seed(seed)
    model = extend_vocab(base_model(), tok).train()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0)  # few steps, so no weight decay
    rng = random.Random(seed)
    order: list[int] = []
    hist = []
    t0 = time.time()
    for step in range(1, steps + 1):
        lr = LR * min(1.0, step / WARMUP) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / steps)))
        for pg in opt.param_groups:
            pg["lr"] = lr
        if len(order) < BSZ:  # shuffle again after each epoch
            order += rng.sample(range(len(train_set)), len(train_set))
        idx, order = order[:BSZ], order[BSZ:]
        x, y = batchify(tok, [train_set[i] for i in idx], use_mask)
        loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten(), ignore_index=-100)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 50 == 0 or step == 1:
            model.eval()
            v = assistant_loss(model, tok, val_set)
            model.train()
            hist.append({"step": step, "train_loss": round(loss.item(), 4), "val_asst_loss": round(v, 4)})
            log(f"    step {step:4d}  train loss {loss.item():.3f}  val (assistant tokens) {v:.3f}"
                f"  ({time.time() - t0:.0f}s)")
    return model.eval(), hist


# ── Generation and scoring ──────────────────────────────────────────────────
@torch.no_grad()
def reply(model, tok, messages, max_new: int = 64) -> str:
    """Generate the assistant reply greedily. Stop at <|im_end|>."""
    ids, _ = lm.encode_with_mask(tok, messages, add_generation_prompt=True)
    cache = ch10.KVCache(model.c.n_layers)  # KV cache of Chapter 10: process the prompt once, then only the new token at each step
    logits = model(torch.tensor([ids]), cache)
    out = []
    for _ in range(max_new):
        nxt = int(logits[0, -1].argmax())
        if nxt == tok.im_end:
            break
        out.append(nxt)
        logits = model(torch.tensor([[nxt]]), cache)
    return tok.decode(out)


def score(text: str, gold: dict) -> dict:
    calls = []
    try:
        calls = lm.tmpl.parse_tool_calls(text)
    except json.JSONDecodeError:
        pass
    ok = len(calls) == 1 and isinstance(calls[0], dict)
    return {"format": ok, "name": ok and calls[0].get("name") == gold["name"],
            "args": ok and calls[0].get("name") == gold["name"]
            and calls[0].get("arguments") == gold["arguments"]}


def evaluate(model, tok, test_set) -> dict:
    s = {"format": 0, "name": 0, "args": 0}
    for ex in test_set:
        r = score(reply(model, tok, ex["messages"][:2]), ex["call"])
        for k in s:
            s[k] += r[k]
    return {k: v / len(test_set) for k, v in s.items()}


@torch.no_grad()
def continue_text(model, tok, text: str, n: int = 60) -> str:
    ids = tok.encode(text)
    cache = ch10.KVCache(model.c.n_layers)
    logits, out = model(torch.tensor([ids]), cache), []
    for _ in range(n):
        out.append(int(logits[0, -1][: tok.n_base].argmax()))  # the base model knows only the old characters
        logits = model(torch.tensor([[out[-1]]]), cache)
    return tok.decode(out)


def get_or_train(name: str, tok, train_set, val_set, use_mask: bool, log=print):
    path = OUT / f"ch16_{name}.pt"
    hist_path = OUT / f"ch16_{name}.json"
    if path.exists() and hist_path.exists():
        model = extend_vocab(base_model(), tok)
        model.load_state_dict(torch.load(path, weights_only=True))
        return model.eval(), json.loads(hist_path.read_text())
    log(f"  Train {name} ({STEPS} steps, batch {BSZ}, lr {LR}). Only once: later runs load {path.name}.")
    model, hist = sft(tok, train_set, val_set, use_mask, log=log)
    OUT.mkdir(exist_ok=True)
    torch.save(model.state_dict(), path)
    hist_path.write_text(json.dumps(hist))
    return model, hist


def run(log=print) -> dict:
    """Do all experiments. Return the numbers for the video and the text."""
    tok = lm.ChatTok(ch10.CharData().chars)
    train_set, val_set = dataset(N_TRAIN, 0, "train"), dataset(64, 1, "train")
    test_set = dataset(N_TEST, 2, "test")  # unseen cities, random numbers
    base = base_model()
    demo = next(ex for ex in test_set if ex["call"]["name"] == "get_weather")  # no digits in the question, so the base model knows each character
    q = demo["messages"][1]["content"]
    res = {"n_params": sum(p.numel() for p in base.parameters()), "vocab_base": tok.n_base,
           "vocab_new": len(tok.itos), "question": q,
           "base_continue": continue_text(base, tok, q + "\n")}
    base_x = extend_vocab(base_model(), tok).eval()
    res["base_reply"] = reply(base_x, tok, demo["messages"][:2], max_new=60)
    res["base_eval"] = evaluate(base_x, tok, test_set[:20])
    res["base_val_asst_loss"] = assistant_loss(base_x, tok, val_set)
    # On the same data: the loss "on assistant tokens only" and "on all tokens" (base model, extended vocabulary)
    x, y_all = batchify(tok, val_set, use_mask=False)
    res["base_val_all_loss"] = F.cross_entropy(base_x(x).flatten(0, 1), y_all.flatten(),
                                               ignore_index=-100).item()
    for name, use_mask, data in (("masked", True, train_set), ("unmasked", False, train_set),
                                 ("small40", True, train_set[:N_SMALL])):
        model, hist = get_or_train(name, tok, data, val_set, use_mask, log)
        res[name] = {"hist": hist, "eval": evaluate(model, tok, test_set),
                     "val_asst_loss": assistant_loss(model, tok, val_set)}
        if name == "masked":
            res["masked"]["split"] = loss_split(model, tok, val_set)
            res["masked"]["eval_seen"] = evaluate(model, tok, dataset(N_TEST, 3, "train"))
            res["examples"] = [{"q": ex["messages"][1]["content"],
                                "reply": reply(model, tok, ex["messages"][:2])}
                               for ex in test_set[:4]]
    return res


if __name__ == "__main__":
    t0 = time.time()
    r = run()
    print(f"\nBase model: {r['n_params'] / 1e6:.2f}M parameters, vocabulary {r['vocab_base']} → {r['vocab_new']}")
    print(f"Question: {r['question']!r}")
    print(f"Base model continues the text (no template): {r['base_continue']!r}")
    print(f"Base model 'answer' with the chat template: {r['base_reply']!r}")
    print(f"Base model on 20 test questions: {r['base_eval']}")
    print(f"Base model loss on the validation set: assistant tokens only {r['base_val_asst_loss']:.3f}, "
          f"all tokens {r['base_val_all_loss']:.3f}")
    print("\nAnswers after SFT (with mask):")
    for e in r["examples"]:
        print(f"  {e['q']!r:40s} → {e['reply']!r}")
    print(f"\n{N_TEST} test questions (no city was in the training data):")
    print(f"{'setting':24s} {'format':>6s} {'name ok':>8s} {'args ok':>8s} {'val loss (asst)':>16s}")
    for name, label in (("masked", "with mask, 2000 samples"), ("unmasked", "no mask, 2000 samples"),
                        ("small40", "with mask, 40 samples")):
        e = r[name]["eval"]
        print(f"{label:24s} {e['format']:6.2f} {e['name']:8.2f} {e['args']:8.2f} "
              f"{r[name]['val_asst_loss']:16.3f}")
    sp = r["masked"]["split"]
    print(f"Model with mask, validation assistant tokens by type: argument values {sp['n_args']}, mean loss {sp['args']:.3f}; "
          f"rest (format, function name) {sp['n_rest']}, mean loss {sp['rest']:.3f}")
    e = r["masked"]["eval_seen"]
    print(f"Control: model with mask on cities seen in training (numbers still random): format {e['format']:.2f}, "
          f"name ok {e['name']:.2f}, args ok {e['args']:.2f}")
    print(f"\nCurve: {N_SMALL} samples, {STEPS} steps (about {STEPS * BSZ // N_SMALL} epochs):")
    for h in r["small40"]["hist"]:
        print(f"  step {h['step']:4d}  train loss {h['train_loss']:.3f}  val loss {h['val_asst_loss']:.3f}")
    print(f"\nTotal time {time.time() - t0:.0f}s")
