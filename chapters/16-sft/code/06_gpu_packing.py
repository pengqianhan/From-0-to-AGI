"""Chapter 16 · GPU measurements: how much time packing saves, and how much isolation of the
conversations costs.

Section 5 counted tokens on the CPU. For the same 200 conversations, real tokens are only 43% with
one conversation per row padded to 513, and 91% after first-fit packing. Here we measure the time
on one GPU (CUDA is necessary):

1. Use the shape of the main-line model (configs/main/sft.toml, 689.5M parameters; BF16 autocast,
   eager mode). Do one forward + backward pass over the 200 conversations. The loss is only on
   assistant tokens. The optimizer step is not included, because it does not depend on the layout.
   Three layouts:
     one conversation per row, padded to 513;
     one conversation per row, in the original order, 8 per batch, padded only to the longest
     row in the batch (the common "dynamic padding");
     first-fit packing into 94 windows (pack_first_fit of 04).
   The conversations are the 200 from 04. We use the character-level token ids as ids in the
   main-line vocabulary. This only makes the lengths and the masks the same.
2. The trade-off of Section 5.3: zero uses the fastest causal attention kernel and does not isolate
   packed conversations. Pack the same conversations with first-fit into the main-line SFT window
   (8192) and take the first window. Measure the forward + backward pass of one attention layer
   (16 query heads / 8 K/V heads, head_dim 128) in four versions:
     normal causal (is_causal of SDPA, which uses FlashAttention): what zero does now; it has
       cross-contamination;
     document mask as one 8192 × 8192 boolean matrix for SDPA: the most direct version;
     FlexAttention + document mask: it works in 128 × 128 blocks and skips each fully masked block;
     variable-length (varlen) FlashAttention: give it the start and end of each conversation.
       This is the "option for the second step" in Section 5.3.
Without a GPU, skip this script. The text shows one result from an RTX 3090.
Run: uv run python chapters/16-sft/code/06_gpu_packing.py
"""

from __future__ import annotations

import importlib.util
import random
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))  # lets the script import zero from the repository root


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


pk = load("ch16_packing", HERE / "04_packing.py")  # pack_first_fit, and also lm (02) and ch10
lm, ch10 = pk.lm, pk.ch10
MICRO = 8  # 8 rows per micro-batch


def conversations() -> list[tuple[list[int], list[bool]]]:
    """The same 200 conversations as in 04_packing.run() (same random seed)."""
    tok = lm.ChatTok(ch10.CharData().chars)
    rng = random.Random(3)
    convs = []
    for _ in range(200):
        msgs = lm.make_example(rng)["messages"]
        for _ in range(rng.choice([0, 0, 1, 2])):
            e2 = lm.make_example(rng)
            msgs = msgs[:1] + e2["messages"][1:] + msgs[1:]
        convs.append(lm.encode_with_mask(tok, msgs))
    return convs


def rows_to_batches(rows: list[tuple[list[int], list[bool]]], pad_to: int | None) -> list[tuple]:
    """Each row (ids, mask) → x = ids[:-1], y = ids[1:] (−100 at non-assistant positions).

    One batch for each MICRO rows. pad_to=None: pad to the longest row in the batch.
    """
    out = []
    for i in range(0, len(rows), MICRO):
        chunk = rows[i : i + MICRO]
        L = (pad_to or max(len(ids) for ids, _ in chunk)) - 1
        x = torch.zeros(len(chunk), L, dtype=torch.long)
        y = torch.full((len(chunk), L), -100, dtype=torch.long)
        for r, (ids, mask) in enumerate(chunk):
            xi, yi = lm.masked_targets(ids, mask)
            x[r, : len(xi)], y[r, : len(yi)] = xi, yi
        out.append((x.cuda(), y.cuda()))
    return out


def time_pass(model, batches, reps: int = 2) -> tuple[float, float]:
    """One forward + backward pass over all data (gradient accumulation, no update), reps times.

    Return (median of the seconds, max / min − 1). Each pass takes 7–15 seconds, and two passes
    differ by less than 1%. So we run only two passes (the median is the mean) and keep the total
    time under 2 minutes.
    """
    times = []
    for _ in range(reps):
        model.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for x, y in batches:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model.loss(x, y)
            loss.backward()
        b.record()
        torch.cuda.synchronize()
        times.append(a.elapsed_time(b) / 1e3)
    return statistics.median(times), max(times) / min(times) - 1


def part1(convs) -> None:
    from zero.config import load_model_config
    from zero.model import Transformer

    lengths = [len(ids) for ids, _ in convs]
    real = sum(lengths)
    window = 513
    bins = pk.pack_first_fit(lengths, window)
    packed_rows = []
    for b in bins:  # join the conversations of one window end to end; the mask follows them, so the predictions
        # at the boundaries are not in the loss
        ids = [t for i in b for t in convs[i][0]]
        mask = [m for i in b for m in convs[i][1]]
        packed_rows.append((ids, mask))
    layouts = [
        ("one per row, pad to 513", rows_to_batches(convs, window)),
        ("one per row, batch max", rows_to_batches(convs, None)),
        (f"first-fit, {len(bins)} windows", rows_to_batches(packed_rows, window)),
    ]
    cfg = load_model_config(ROOT / "configs" / "main" / "sft.toml")
    with torch.device("cuda"):  # build and initialize the model directly on the GPU (one CPU thread is slow for 690M parameters)
        model = Transformer(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"① Main-line model ({n_params / 1e6:.1f}M parameters), the same 200 conversations ({real:,} real tokens), "
          f"one forward + backward pass, {MICRO} rows per batch:")
    time_pass(model, layouts[0][1][:3], reps=1)  # warmup: cuBLAS selects its algorithms and allocates GPU memory
    print(f"   {'layout':24s} {'#bat':>4s} {'positions':>10s} {'real %':>8s} {'time':>8s} "
          f"{'real token/s':>12s} {'vs row 1':>10s} {'spread':>8s}")
    base = None
    for name, batches in layouts:
        positions = sum(x.numel() for x, _ in batches)
        sec, spread = time_pass(model, batches)
        base = base or sec
        print(f"   {name:24s} {len(batches):4d} {positions:10,d} {100 * real / positions:7.0f}% "
              f"{sec:7.2f}s {real / sec:12,.0f} {base / sec:9.2f}× {100 * spread:7.1f}%")
    del model
    torch.cuda.empty_cache()


# ── 2. One attention layer: causal vs three document masks ─────────────────────────
def bench(fn, grad_out, reps: int = 10) -> float:
    """Forward + backward pass. Warm up 3 times, then repeat reps times. Return the median in ms."""
    for _ in range(3):
        fn().backward(grad_out)
    torch.cuda.synchronize()
    times = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn().backward(grad_out)
        b.record()
        torch.cuda.synchronize()
        times.append(a.elapsed_time(b))
    return statistics.median(times)


def part2(convs) -> None:
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention
    from torch.nn.attention.varlen import varlen_attn

    T, H, KV, D = 8192, 16, 8, 128
    lengths = [len(ids) for ids, _ in convs]
    first = pk.pack_first_fit(lengths, T)[0]
    docs = [lengths[i] for i in first]
    n_conv = len(docs)
    if sum(docs) < T:
        docs.append(T - sum(docs))  # the padding positions left in the window are one more segment
    dev = "cuda"
    doc_id = torch.repeat_interleave(torch.arange(len(docs)), torch.tensor(docs)).to(dev)
    useful = sum(n * (n + 1) // 2 for n in docs[:n_conv]) / (T * (T + 1) // 2)
    print(f"\n② Main-line SFT window {T}: first-fit puts {n_conv} conversations ({sum(docs[:n_conv]):,} tokens) into it. "
          f"Forward + backward pass of one attention layer ({H} query heads / {KV} K/V heads, head_dim {D}, BF16).")
    print(f"   The document mask has only {100 * useful:.1f}% of the (query, key) pairs of the causal mask")

    torch.manual_seed(0)
    q = torch.randn(1, H, T, D, device=dev, dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(1, KV, T, D, device=dev, dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(1, KV, T, D, device=dev, dtype=torch.bfloat16, requires_grad=True)
    go = torch.randn(1, H, T, D, device=dev, dtype=torch.bfloat16)
    allow = (doc_id[:, None] == doc_id[None, :]) & torch.ones(T, T, dtype=torch.bool, device=dev).tril()

    def causal():
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            return F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)

    def boolmask():
        return F.scaled_dot_product_attention(q, k, v, attn_mask=allow, enable_gqa=True)

    def same_doc(b, h, qi, ki):
        return (doc_id[qi] == doc_id[ki]) & (qi >= ki)

    block_mask = create_block_mask(same_doc, None, None, T, T, device=dev)
    flex = torch.compile(flex_attention)

    def flexdoc():
        return flex(q, k, v, block_mask=block_mask, enable_gqa=True)

    cu = torch.tensor([0, *torch.tensor(docs).cumsum(0).tolist()], device=dev, dtype=torch.int32)

    def varlen():  # the input of varlen is (total tokens, heads, head_dim), with no batch dimension
        o = varlen_attn(q[0].transpose(0, 1).contiguous(), k[0].transpose(0, 1).contiguous(),
                        v[0].transpose(0, 1).contiguous(), cu, cu, max(docs), max(docs),
                        window_size=(-1, 0))
        return o.transpose(0, 1)[None]

    try:  # which implementation PyTorch selected for "boolean mask + GQA"
        chosen = SDPBackend(torch._fused_sdp_choice(q, k, v, attn_mask=allow, enable_gqa=True)).name
    except Exception:
        chosen = "?"
    rows = [("causal FlashAttention (mixes docs)", causal),
            ("doc mask: boolean matrix to SDPA", boolmask),
            ("doc mask: FlexAttention", flexdoc),
            ("doc mask: varlen FlashAttention", varlen)]
    ref = None
    with torch.no_grad():
        outs = {}
        for name, fn in rows:
            try:
                outs[name] = fn().float()
            except Exception as e:  # if a version is not available on this GPU / this PyTorch version, print the error as it is
                outs[name] = e
        ref = outs[rows[3][0]] if not isinstance(outs[rows[3][0]], Exception) else None
    print(f"   {'version':34s} {'fwd+bwd':>10s} {'28 layers':>9s} {'speedup':>8s} {'peak mem':>9s} "
          f"{'max diff vs varlen':>18s}")
    base = None
    for name, fn in rows:
        if isinstance(outs[name], Exception):
            print(f"   {name:34s} not available: {type(outs[name]).__name__}: {str(outs[name])[:80]}")
            continue
        q.grad = k.grad = v.grad = None
        torch.cuda.reset_peak_memory_stats()
        try:
            ms = bench(fn, go)
        except torch.OutOfMemoryError:
            print(f"   {name:34s} out of GPU memory (its backward pass does not fit in 24 GB)")
            continue
        peak = torch.cuda.max_memory_allocated() / 2**30
        base = base or ms
        diff = "—" if ref is None else f"{float((outs[name] - ref).abs().max()):.2e}"
        print(f"   {name:34s} {ms:8.2f}ms {28 * ms:7.0f}ms {base / ms:7.2f}× {peak:7.2f}GiB {diff:>18s}")
    print(f"   (For the boolean-matrix row, PyTorch selected the {chosen} implementation. The difference "
          "between causal and varlen is the cross-contamination: "
          "later conversations in a window see the earlier ones. The differences between the three "
          "document masks are BF16 rounding.)")


def main() -> None:
    if not torch.cuda.is_available():
        print("This script needs a CUDA GPU. Without a GPU, skip it. The text shows one result from an RTX 3090.")
        sys.exit(0)
    torch.manual_seed(0)
    print(f"GPU: {torch.cuda.get_device_name(0)}, PyTorch {torch.__version__}, CUDA {torch.version.cuda}\n")
    convs = conversations()
    t0 = time.time()
    part1(convs)
    part2(convs)
    print(f"\nTotal time {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
