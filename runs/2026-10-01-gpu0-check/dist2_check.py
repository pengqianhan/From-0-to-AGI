"""Multi-GPU version of Stage 6, items 2 and 4, and of Muon: really run DDP / FSDP2 on 2 RTX 3090 GPUs.

The GPUs connect through PCIe, with no NVLink.

    # 1) 1-GPU reference (physical GPU0): no wrapper, micro 8 × accumulation 2, 16 sequences × 2048 tokens per step
    CUDA_VISIBLE_DEVICES=0 UV_NO_SYNC=1 uv run python runs/2026-10-01-gpu0-check/dist2_check.py ref
    # 1b) 1-GPU control: micro 4 × accumulation 4 (mathematically equal to 1); only the order of the sums differs)
    CUDA_VISIBLE_DEVICES=0 UV_NO_SYNC=1 uv run python runs/2026-10-01-gpu0-check/dist2_check.py ctrl
    # 2) 2 GPUs (physical GPU0 + GPU2, both with a power limit of 240 W): also 16 sequences per step
    CUDA_VISIBLE_DEVICES=0,2 UV_NO_SYNC=1 uv run torchrun --standalone --nproc_per_node=2 \
        runs/2026-10-01-gpu0-check/dist2_check.py multi

The script uses the single-source configuration configs/l20m_tiny_1src.toml. PackedDataLoader gives
global sample g to rank g % world_size. Thus "2 GPUs with 8 sequences each, no accumulation",
"2 GPUs with 4 sequences each × accumulation 2", and "1 GPU with 8 sequences × accumulation 2" see the same
16 samples at each step. The gradients are mathematically the same (only the order of the sums differs).
All runs turn on torch.use_deterministic_algorithms. This removes the run-to-run differences of the
GPU kernels (see item 3 of "Read first" in the README of this directory).

(With a multi-source configuration, the source sampler of MixtureLoader is seeded with (seed, rank).
 Each GPU samples its own sources, so the samples are not the same batch as on 1 GPU. To reproduce that
 comparison, use DIST2_CFG=.../l20m_tiny.toml DIST2_OUT=out/gpu0-check/dist2_mixture.)

The multi phase runs these in sequence (50 steps each, l20m shape + tiny data):
  ddp        DDP, micro 8 × 1            ddp_accum  DDP, micro 4 × 2 (gradient accumulation with no_sync)
  fsdp       FSDP2, micro 8 × 1          muon_ddp   optim.name=muon, DDP, micro 8 × 1
  ddp / fsdp also have a group at lr 3e-4 (10 times smaller than the ladder default; the early phase is
  less "chaotic", so the numerical differences of the code paths are easier to see)
After each DDP run, the script reads the bits of each parameter as integers, sums them, and compares the
sums between the 2 GPUs (are the parameters bit-identical?).
It reads the FSDP checkpoint back on 1 GPU with load_policy (strict) and computes the CE on the validation data.
Then it keeps only step 25 in the fsdp directory, resumes on 2 GPUs to step 50, and compares the loss with
the uninterrupted run (this tests the state_dict gather / scatter of FSDP2).
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
os.chdir(REPO)

from zero.config import load_config  # noqa: E402
from zero.post.common import load_policy  # noqa: E402
from zero.train.dist import DistInfo, init_distributed, unwrap_model  # noqa: E402
from zero.train.trainer import Trainer  # noqa: E402

# Default: the single-source configuration. The multi-source MixtureLoader samples sources with
# (seed, rank), so the samples on 2 GPUs and on 1 GPU are different, and a per-step parity check is not possible.
CFG = os.environ.get("DIST2_CFG", "runs/2026-10-01-gpu0-check/configs/l20m_tiny_1src.toml")
OUT = Path(os.environ.get("DIST2_OUT", "out/gpu0-check/dist2"))
STEPS = 50


def make_cfg(name: str, micro: int, accum: int, parallel: str = "ddp", extra: tuple = ()):
    return load_config(
        CFG,
        list(extra)
        + [
            f"train.max_steps={STEPS}",
            f"train.micro_batch_size={micro}",
            f"train.grad_accum_steps={accum}",
            f"train.parallel={parallel}",
            f'train.out_dir="{OUT / name}"',
            "train.eval_every=0",
            "checkpoint.every=25",
            "checkpoint.keep_last=0",
            "logging.every=1",
            "schedule.warmup_steps=10",
        ],
    )


def run(name: str, info: DistInfo, micro: int, accum: int, parallel: str = "ddp", extra=()) -> dict:
    torch.cuda.reset_peak_memory_stats()
    trainer = Trainer(make_cfg(name, micro, accum, parallel, extra), info, log=lambda m: None)
    hist = trainer.train()
    res = {
        "loss": [h["loss"] for h in hist],
        "wrapped": type(trainer.model).__name__,
        "peak_gib": torch.cuda.max_memory_allocated() / 2**30,
    }
    if info.world_size > 1:
        # Peak GPU memory of each GPU
        peaks = [None] * info.world_size
        dist.all_gather_object(peaks, res["peak_gib"])
        res["peak_gib_per_rank"] = peaks
        if parallel == "ddp":
            # Bitwise comparison of the parameters: read the bits of each parameter as integers and sum them
            # (a bit-level checksum). The sums must be identical on the 2 GPUs.
            sums = torch.stack(
                [
                    p.detach().contiguous().view(torch.int16 if p.element_size() == 2 else torch.int32)
                    .to(torch.int64)
                    .sum()
                    for p in unwrap_model(trainer.model).parameters()
                ]
            )
            gathered = [torch.zeros_like(sums) for _ in range(info.world_size)]
            dist.all_gather(gathered, sums)
            res["params_bitwise_equal_across_ranks"] = all(torch.equal(gathered[0], g) for g in gathered)
    if info.rank == 0:
        print(
            f"[{name}] {res['wrapped']}, loss step1 {res['loss'][0]:.4f} → step{STEPS} {res['loss'][-1]:.4f}, "
            f"peak GPU memory {res.get('peak_gib_per_rank', res['peak_gib'])}, "
            f"parameters bit-identical across GPUs: {res.get('params_bitwise_equal_across_ranks', '—')}",
            flush=True,
        )
    del trainer
    torch.cuda.empty_cache()
    return res


def rel(a: list[float], b: list[float]) -> float:
    return max(abs(x - y) / abs(y) for x, y in zip(a, b, strict=True))


def val_ce(ckpt: Path, dev: torch.device) -> tuple[float, int]:
    from zero.data.loader import PackedDataLoader

    model, _ = load_policy(ckpt, device=dev)
    val = PackedDataLoader(
        "out/gpu0-check/tiny/data/*_val_*.bin", seq_len=2048, batch_size=8, shuffle=False, device=dev
    )
    xv, yv = val.next_batch()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        return float(model.eval().loss(xv, yv)), len(model.state_dict())


def phase_ref() -> None:
    assert torch.cuda.device_count() == 1, "Only 1 GPU may be visible for the reference run"
    torch.use_deterministic_algorithms(True)
    OUT.mkdir(parents=True, exist_ok=True)
    info = DistInfo(device=torch.device("cuda", 0))
    res = {
        "adamw": run("ref_adamw", info, 8, 2),
        "adamw_lr3e-4": run("ref_adamw_lr3e-4", info, 8, 2, extra=("optim.lr=3e-4",)),
        "muon": run("ref_muon", info, 8, 2, extra=('optim.name="muon"',)),
    }
    (OUT / "ref.json").write_text(json.dumps(res, indent=1))


def phase_ctrl() -> None:
    """Control: on 1 GPU, micro 4 × accumulation 4 is mathematically equal to micro 8 × accumulation 2.

    Only the order of the sums differs. The divergence from the 1-GPU reference shows how much training
    amplifies a different summation order alone. We expect the DDP / FSDP differences to be of the same
    order of magnitude.
    """
    assert torch.cuda.device_count() == 1, "Only 1 GPU may be visible for the control run"
    torch.use_deterministic_algorithms(True)
    info = DistInfo(device=torch.device("cuda", 0))
    ref = json.loads((OUT / "ref.json").read_text())
    res = {
        "adamw_4x4": run("ctrl_adamw_4x4", info, 4, 4),
        "adamw_lr3e-4_4x4": run("ctrl_adamw_lr3e-4_4x4", info, 4, 4, extra=("optim.lr=3e-4",)),
    }
    res["compare"] = {
        "4x4_vs_8x2_max_rel": rel(res["adamw_4x4"]["loss"], ref["adamw"]["loss"]),
        "4x4_vs_8x2_lr3e-4_max_rel": rel(res["adamw_lr3e-4_4x4"]["loss"], ref["adamw_lr3e-4"]["loss"]),
    }
    (OUT / "ctrl.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res["compare"], indent=1), flush=True)


def phase_multi() -> None:
    info = init_distributed("cuda")
    assert info.world_size == 2, info
    torch.use_deterministic_algorithms(True)
    dev = info.device
    ref = json.loads((OUT / "ref.json").read_text())
    res: dict = {"world_size": info.world_size}
    res["ddp"] = run("ddp", info, 8, 1)
    res["ddp_accum"] = run("ddp_accum", info, 4, 2)
    res["fsdp"] = run("fsdp", info, 8, 1, "fsdp")
    res["ddp_lr3e-4"] = run("ddp_lr3e-4", info, 8, 1, extra=("optim.lr=3e-4",))
    res["fsdp_lr3e-4"] = run("fsdp_lr3e-4", info, 8, 1, "fsdp", extra=("optim.lr=3e-4",))
    res["muon_ddp"] = run("muon_ddp", info, 8, 1, extra=('optim.name="muon"',))

    # FSDP resume: keep only step 25, then resume on 2 GPUs to step 50
    if info.rank == 0:
        src, dst = OUT / "fsdp", OUT / "fsdp_resume"
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        shutil.rmtree(dst / "ckpt" / "step_00000050")
        (dst / "ckpt" / "latest").write_text("step_00000025")
        (dst / "log.jsonl").unlink()
    dist.barrier()
    trainer = Trainer(make_cfg("fsdp_resume", 8, 1, "fsdp"), info, log=lambda m: None)
    start = trainer.step
    resumed = [h["loss"] for h in trainer.train()]
    del trainer

    if info.rank == 0:
        cmp = {
            "ddp_vs_ref_max_rel": rel(res["ddp"]["loss"], ref["adamw"]["loss"]),
            "ddp_accum_vs_ref_max_rel": rel(res["ddp_accum"]["loss"], ref["adamw"]["loss"]),
            "ddp_accum_vs_ddp_max_rel": rel(res["ddp_accum"]["loss"], res["ddp"]["loss"]),
            "step1_loss_ref_ddp_fsdp": [ref["adamw"]["loss"][0], res["ddp"]["loss"][0], res["fsdp"]["loss"][0]],
            "fsdp_vs_ddp_max_rel": rel(res["fsdp"]["loss"], res["ddp"]["loss"]),
            "fsdp_vs_ref_max_rel": rel(res["fsdp"]["loss"], ref["adamw"]["loss"]),
            "ddp_lr3e-4_vs_ref_max_rel": rel(res["ddp_lr3e-4"]["loss"], ref["adamw_lr3e-4"]["loss"]),
            "fsdp_lr3e-4_vs_ddp_max_rel": rel(res["fsdp_lr3e-4"]["loss"], res["ddp_lr3e-4"]["loss"]),
            "muon_ddp_vs_ref_max_rel": rel(res["muon_ddp"]["loss"], ref["muon"]["loss"]),
            "ddp_bitwise_equal_ref": res["ddp"]["loss"] == ref["adamw"]["loss"],
            "fsdp_resume_from": start,
            "fsdp_resume_max_abs_diff": max(
                abs(a - b) for a, b in zip(resumed, res["fsdp"]["loss"][start:], strict=True)
            ),
        }
        ce_f, n_f = val_ce(OUT / "fsdp" / "ckpt", dev)
        ce_d, _ = val_ce(OUT / "ddp" / "ckpt", dev)
        cmp["fsdp_ckpt_load_policy_keys"] = n_f
        cmp["val_ce_fsdp_vs_ddp_ckpt"] = [ce_f, ce_d]
        res["compare"] = cmp
        res["fsdp_resume_losses"] = resumed
        (OUT / "multi.json").write_text(json.dumps(res, indent=1))
        print(json.dumps(cmp, indent=1), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    {"ref": phase_ref, "ctrl": phase_ctrl, "multi": phase_multi}[sys.argv[1]]()
