"""Stage 6, items 7 and 8 (1-GPU version): memory limits of the main-line 689.5M configuration on 24 GB.

It finds the largest micro batch that does not cause OOM, with and without activation checkpointing.
It also checks if a 32K sequence fits. It compares the results with the estimate of zero/tools/memory_calc.py.

    CUDA_VISIBLE_DEVICES=0 UV_NO_SYNC=1 uv run python runs/2026-10-01-gpu0-check/mem_probe.py main
    CUDA_VISIBLE_DEVICES=0 UV_NO_SYNC=1 uv run python runs/2026-10-01-gpu0-check/mem_probe.py longctx

Each trial runs the real `Trainer` for 3 steps in a new subprocess (forward + backward + fused AdamW;
stop_at=3, no checkpoint). It records `torch.cuda.max_memory_allocated` (the tensors that PyTorch allocates)
and `max_memory_reserved` (the memory that the caching allocator really holds).
It also records tok/s and MFU of step 3 (MFU uses the 3090 dense BF16 peak of 71 TFLOPS).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
GiB = 1024**3


def one(config: str, mb: int, ckpt: bool, compile_: bool, seq_len: int) -> dict:
    import torch

    sys.path.insert(0, str(REPO))
    os.chdir(REPO)
    from zero.config import load_config
    from zero.train.trainer import Trainer

    overrides = [
        f"train.micro_batch_size={mb}",
        "train.grad_accum_steps=1",
        f"train.activation_checkpointing={str(ckpt).lower()}",
        f"train.compile={str(compile_).lower()}",
        "train.max_steps=100",
        "train.eval_every=0",
        "checkpoint.every=0",
        "checkpoint.resume=false",
        "logging.every=1",
        'train.init_from=""',
        f"data.seq_len={seq_len}",
        f'train.out_dir="out/gpu0-check/mem_probe/{Path(config).stem}_mb{mb}_ck{int(ckpt)}_c{int(compile_)}_T{seq_len}"',
    ]
    cfg = load_config(config, overrides)
    row: dict = {"mb": mb, "ckpt": ckpt, "compile": compile_, "seq_len": seq_len}
    try:
        trainer = Trainer(cfg, log=lambda m: None)
        torch.cuda.reset_peak_memory_stats()
        hist = trainer.train(stop_at=3)
        row.update(
            ok=True,
            loss=[round(h["loss"], 4) for h in hist],
            tok_per_s=round(hist[-1]["tok_per_s"]),
            mfu_3090=round(hist[-1]["mfu"], 4),
        )
    except torch.OutOfMemoryError as e:
        row.update(ok=False, error=str(e).splitlines()[0][:200])
    row["peak_alloc_GiB"] = round(torch.cuda.max_memory_allocated() / GiB, 2)
    row["peak_reserved_GiB"] = round(torch.cuda.max_memory_reserved() / GiB, 2)
    return row


def estimate(config: str, mb: int, ckpt: bool, seq_len: int) -> float:
    sys.path.insert(0, str(REPO))
    from zero.config import load_model_config
    from zero.tools.memory_calc import estimate_memory

    cfg = load_model_config(REPO / config)
    est = estimate_memory(cfg, mb, seq_len, 1, "ddp", "bf16", ckpt)
    return round(est.total / GiB, 2)


def trial(config: str, mb: int, ckpt: bool, compile_: bool, seq_len: int) -> dict:
    cmd = [sys.executable, __file__, "--one", config, str(mb), str(int(ckpt)), str(int(compile_))]
    cmd.append(str(seq_len))
    p = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO)
    lines = [ln for ln in p.stdout.splitlines() if ln.startswith("{")]
    if not lines:
        row = {"mb": mb, "ckpt": ckpt, "compile": compile_, "seq_len": seq_len, "ok": False}
        row["error"] = (p.stderr.strip().splitlines() or ["?"])[-1][:200]
    else:
        row = json.loads(lines[-1])
    row["memory_calc_GiB"] = estimate(config, mb, ckpt, seq_len)
    print(json.dumps(row, ensure_ascii=False), flush=True)
    return row


def search(config: str, ckpt: bool, compile_: bool, seq_len: int, max_mb: int = 16) -> list:
    rows = []
    for mb in range(1, max_mb + 1):
        r = trial(config, mb, ckpt, compile_, seq_len)
        rows.append(r)
        if not r["ok"]:
            break
    return rows


def main() -> None:
    if sys.argv[1] == "--one":
        config, mb, ck, co, T = sys.argv[2:7]
        print(json.dumps(one(config, int(mb), ck == "1", co == "1", int(T)), ensure_ascii=False))
        return
    which = sys.argv[1]
    res: dict = {}
    if which == "main":
        cfg = "runs/2026-10-01-gpu0-check/configs/main_tiny.toml"
        res["eager_no_ckpt"] = search(cfg, False, False, 4096)
        res["eager_ckpt"] = search(cfg, True, False, 4096)
        best = max([r["mb"] for r in res["eager_ckpt"] if r["ok"]] or [1])
        res["compile_ckpt"] = [trial(cfg, best, True, True, 4096)]
        ok_nock = [r["mb"] for r in res["eager_no_ckpt"] if r["ok"]]
        if ok_nock:
            res["compile_no_ckpt"] = [trial(cfg, max(ok_nock), False, True, 4096)]
    elif which == "longctx":
        cfg = "runs/2026-10-01-gpu0-check/configs/longctx_tiny.toml"
        rows = []
        for T in (32768, 16384, 8192):
            r = trial(cfg, 1, True, False, T)
            rows.append(r)
            if r["ok"]:
                break
        res["ckpt_mb1"] = rows
        res["no_ckpt_mb1_T32768"] = [trial(cfg, 1, False, False, 32768)]
    out = REPO / f"out/gpu0-check/mem_probe_{which}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
