"""Chapter 16 · Minimal code 4: packing and the "cross-contamination" problem.

1. Packing: put conversations of different lengths into fixed-length windows with first-fit.
   Compare the fraction of real tokens with "one conversation per row, padded to the window length".
2. Cross-contamination: in one window, with a normal causal attention mask, the tokens of the second
   conversation can see the first conversation. Use the small model from script 3 (after SFT) and
   compare the logits of conversation B in three cases: alone / packed after A (normal causal mask) /
   packed after A with a document mask (document masking: each conversation sees only itself).

Run: uv run python chapters/16-sft/code/04_packing.py
(It needs the model that script 3 trains. If the model does not exist, the script trains it first.)
"""

from __future__ import annotations

import importlib.util
import math
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

torch.set_num_threads(1)  # the build machine shares its CPU between many jobs; on your computer, you can remove this line
HERE = Path(__file__).resolve().parent


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


sft = load("ch16_sft_tiny", HERE / "03_sft_tiny.py")
lm, ch10 = sft.lm, sft.ch10


# ── 1. First-fit packing ───────────────────────────────────────────────────────
def pack_first_fit(lengths: list[int], window: int) -> list[list[int]]:
    """Return the indices of the conversations in each window.

    Never split a conversation. If it does not fit into an open window, open a new window.
    """
    bins: list[list[int]] = []
    used: list[int] = []
    for i, n in enumerate(lengths):
        for b in range(len(bins)):
            if used[b] + n <= window:
                bins[b].append(i)
                used[b] += n
                break
        else:
            bins.append([i])
            used.append(n)
    return bins


# ── 2. Forward pass with any attention mask. It is a copy of TinyLM.forward of Chapter 10;
# only the causal mask changes: the caller gives the mask. ──
def forward_with_mask(model, ids: torch.Tensor, allow: torch.Tensor) -> torch.Tensor:
    """ids (1, T); allow (T, T) is a boolean matrix: allow[i, j] = position i can see position j."""
    c, T = model.c, ids.shape[1]
    cos, sin = model.cos[:T], model.sin[:T]
    x = model.emb(ids)
    for blk in model.blocks:
        a, h = blk.attn, blk.n1(x)
        q = a.wq(h).view(1, T, c.n_heads, c.head_dim).transpose(1, 2)
        k = a.wk(h).view(1, T, c.n_kv_heads, c.head_dim).transpose(1, 2)
        v = a.wv(h).view(1, T, c.n_kv_heads, c.head_dim).transpose(1, 2)
        q, k = ch10.apply_rope(q, cos, sin), ch10.apply_rope(k, cos, sin)
        att = (q @ k.transpose(-2, -1) / math.sqrt(c.head_dim)).masked_fill(~allow, float("-inf"))
        x = x + a.wo((att.softmax(-1) @ v).transpose(1, 2).reshape(1, T, -1))
        h = blk.n2(x)
        x = x + blk.w_down(F.silu(blk.w_gate(h)) * blk.w_up(h))
    return model.norm(x) @ model.emb.weight.T


def causal(T: int) -> torch.Tensor:
    return torch.ones(T, T, dtype=torch.bool).tril()


def document_mask(doc_ids: torch.Tensor) -> torch.Tensor:
    """Causal + only the same conversation: a block-diagonal lower triangle."""
    return causal(len(doc_ids)) & (doc_ids[:, None] == doc_ids[None, :])


def run() -> dict:
    """Do the two experiments. Return the numbers for the text and the video."""
    tok = lm.ChatTok(ch10.CharData().chars)
    rng = random.Random(3)
    # Different lengths: some samples get extra turns at the start (like multi-turn conversations in real data)
    convs = []
    for _ in range(200):
        ex = lm.make_example(rng)
        msgs = ex["messages"]
        for _ in range(rng.choice([0, 0, 1, 2])):
            e2 = lm.make_example(rng)
            msgs = msgs[:1] + e2["messages"][1:] + msgs[1:]
        convs.append(lm.encode_with_mask(tok, msgs)[0])
    lengths = [len(c) for c in convs]
    window = 513  # seq_len 512 + 1 (as in zero: the window keeps one more token to shift the targets by one)
    bins = pack_first_fit(lengths, window)
    r = {"n": len(convs), "min": min(lengths), "max": max(lengths), "real": sum(lengths),
         "window": window, "n_bins": len(bins),
         "bins_preview": [[lengths[i] for i in b] for b in bins[:6]],
         "pad_pct": 100 * sum(lengths) / (len(convs) * window),
         "pack_pct": 100 * sum(lengths) / (len(bins) * window)}

    # ── Cross-contamination ───────────────────────────────────────────────────
    model, _ = sft.get_or_train("masked", tok, sft.dataset(sft.N_TRAIN, 0, "train"),
                                sft.dataset(64, 1, "train"), True)

    def conv(q, city):
        return lm.encode_with_mask(tok, [{"role": "system", "content": lm.SYSTEM},
                                         {"role": "user", "content": q},
                                         {"role": "assistant", "content": "", "tool_calls": [
                                             {"name": "get_weather", "arguments": {"city": city}}]}])

    a, b = conv("What is the weather in Paris?", "Paris"), conv("Is it raining in Kyoto?", "Kyoto")
    ids_a, ids_b = a[0], b[0]
    packed = torch.tensor([ids_a + ids_b])
    doc = torch.tensor([0] * len(ids_a) + [1] * len(ids_b))
    with torch.no_grad():
        alone = forward_with_mask(model, torch.tensor([ids_b]), causal(len(ids_b)))[0]
        mixed = forward_with_mask(model, packed, causal(packed.shape[1]))[0, len(ids_a):]
        isolated = forward_with_mask(model, packed, document_mask(doc))[0, len(ids_a):]
        same = torch.allclose(model(torch.tensor([ids_b]))[0], alone, atol=1e-5)

    def asst_loss(logits):
        y = torch.tensor(ids_b[1:])
        keep = torch.tensor(b[1][1:])
        return F.cross_entropy(logits[:-1][keep], y[keep]).item()

    r.update(same_as_ch10=bool(same), len_a=len(ids_a), len_b=len(ids_b),
             loss_alone=asst_loss(alone), loss_mixed=asst_loss(mixed),
             loss_isolated=asst_loss(isolated),
             diff_mixed=float((mixed - alone).abs().max()),
             diff_isolated=float((isolated - alone).abs().max()))
    return r


if __name__ == "__main__":
    r = run()
    print(f"{r['n']} conversations, length {r['min']}–{r['max']} tokens, total {r['real']}")
    print(f"No packing (one per row, padded to {r['window']}): {r['n']} rows, real tokens {r['pad_pct']:.0f}%")
    print(f"First-fit packing: {r['n_bins']} windows, real tokens {r['pack_pct']:.0f}%, "
          f"mean {r['n'] / r['n_bins']:.1f} conversations per window")
    print(f"\nOur forward pass with a mask gives the same result as TinyLM.forward of Chapter 10: {r['same_as_ch10']}")
    print("Loss on the assistant tokens of conversation B (Kyoto):")
    print(f"  alone                                {r['loss_alone']:.4f}")
    print(f"  packed after A (Paris), causal mask  {r['loss_mixed']:.4f}"
          f"   max logits difference {r['diff_mixed']:.2e}")
    print(f"  packed after A, document mask        {r['loss_isolated']:.4f}"
          f"   max logits difference {r['diff_isolated']:.2e}")
    print(f"(RoPE uses only relative positions. B starts at position {r['len_a']} in the window, but this "
          "does not change the result, if the attention is isolated.)")
