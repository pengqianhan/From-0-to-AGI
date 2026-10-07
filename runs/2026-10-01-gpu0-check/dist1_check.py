"""Stage 6, item 4 (1-GPU version): really run the DDP / FSDP2 code paths on 1 GPU.

With `torchrun --nproc_per_node=1`, WORLD_SIZE=1, `DistInfo.is_distributed` is False, and `wrap_model`
returns the model unchanged. Thus `--set train.parallel=fsdp` does **nothing** on 1 GPU, and the same is
true for DDP. To make the FSDP branches in wrap_model / checkpoint really run, this script makes its own
NCCL process group with world_size=1 and forces `is_distributed` to True:

    CUDA_VISIBLE_DEVICES=0 UV_NO_SYNC=1 uv run python runs/2026-10-01-gpu0-check/dist1_check.py

The same configuration (l20m shape + tiny data) runs for 50 steps, with torch.use_deterministic_algorithms
on, one group at learning rate 3e-3 and one group at 3e-4:
  plain (no wrapper, the same as torchrun on 1 GPU) / ddp (DDP, NCCL, 1 rank) / fsdp (fully_shard + MixedPrecisionPolicy)
The script compares the loss at each step. It reads the FSDP checkpoint (gathered with get_model_state_dict)
back on 1 GPU with `load_policy` and does a parity check of the logits. Then it resumes from the FSDP
checkpoint of step 25 to step 50 (set_model_state_dict / set_optimizer_state_dict) and compares the loss
with the uninterrupted run. The result shows only that "the 1-GPU code path runs and the values are
reasonable". Sharding and communication across GPUs still need a check on several GPUs.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
os.chdir(REPO)

from zero.config import load_config  # noqa: E402
from zero.post.common import load_policy  # noqa: E402
from zero.train.dist import DistInfo  # noqa: E402
from zero.train.trainer import Trainer  # noqa: E402

CFG = "runs/2026-10-01-gpu0-check/configs/l20m_tiny.toml"
OUT = Path("out/gpu0-check/dist1")
STEPS = 50


@dataclass
class ForcedDistInfo(DistInfo):
    """world_size=1, but it acts as distributed, so wrap_model / checkpoint take the DDP and FSDP branches."""

    @property
    def is_distributed(self) -> bool:  # type: ignore[override]
        return True


def make_cfg(mode: str, steps: int = STEPS, lr: float | None = None):
    parallel = "fsdp" if mode.startswith("fsdp") else "ddp"
    extra = [f"optim.lr={lr}"] if lr else []
    return load_config(
        CFG,
        extra
        + [
            f"train.max_steps={steps}",
            f"train.parallel={parallel}",
            f'train.out_dir="{OUT / mode}"',
            "train.eval_every=0",
            "checkpoint.every=25",
            "checkpoint.keep_last=0",
            "logging.every=1",
            "schedule.warmup_steps=10",
        ],
    )


def run(mode: str, info: DistInfo, lr: float | None = None) -> list[float]:
    cfg = make_cfg(mode, lr=lr)
    torch.cuda.reset_peak_memory_stats()
    trainer = Trainer(cfg, info, log=lambda m: None)
    hist = trainer.train()
    print(
        f"[{mode}] type after wrapping {type(trainer.model).__name__}; optimizer fused="
        f"{trainer.optimizer.defaults.get('fused')}; peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB",
        flush=True,
    )
    losses = [h["loss"] for h in hist]
    print(f"[{mode}] loss step1 {losses[0]:.4f} → step{STEPS} {losses[-1]:.4f}", flush=True)
    return losses


def main() -> None:
    assert torch.cuda.device_count() == 1, "Only 1 GPU may be visible (CUDA_VISIBLE_DEVICES=0)"
    if OUT.exists():
        shutil.rmtree(OUT)
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29533")
    torch.cuda.set_device(0)
    dist.init_process_group("nccl", rank=0, world_size=1, device_id=torch.device("cuda", 0))
    dev = torch.device("cuda", 0)
    # Deterministic algorithms: the default GPU kernels (FlashAttention backward and others) differ
    # between runs. That difference would hide the numerical differences of the DDP / FSDP code paths.
    torch.use_deterministic_algorithms(True)
    plain = DistInfo(device=dev)
    forced = ForcedDistInfo(rank=0, local_rank=0, world_size=1, device=dev, backend="nccl")

    def rel(a: list[float], b: list[float]) -> float:
        return max(abs(x - y) / abs(y) for x, y in zip(a, b, strict=True))

    res: dict = {}
    # Learning rate 3e-4: 10 times smaller than the ladder default 3e-3. The early phase is less "chaotic",
    # so the numerical differences of the code paths are easier to see.
    res["lr3e-4"] = {
        m: run(f"{m}_lr3e-4", i, 3e-4)
        for m, i in (("plain", plain), ("ddp", forced), ("fsdp", forced))
    }
    for k in ("ddp", "fsdp"):
        res[f"lr3e-4_max_rel_diff_{k}_vs_plain"] = rel(res["lr3e-4"][k], res["lr3e-4"]["plain"])
    res["plain"] = run("plain", plain)
    res["ddp"] = run("ddp", forced)
    res["fsdp"] = run("fsdp", forced)
    res["max_rel_diff_ddp_vs_plain"] = rel(res["ddp"], res["plain"])
    res["max_rel_diff_fsdp_vs_plain"] = rel(res["fsdp"], res["plain"])
    res["ddp_bitwise_equal_plain"] = res["ddp"] == res["plain"]
    print(
        f"Max relative difference of the per-step loss (lr 3e-3): ddp vs plain {res['max_rel_diff_ddp_vs_plain']:.2e}"
        f" (bit-identical: {res['ddp_bitwise_equal_plain']}), fsdp vs plain {res['max_rel_diff_fsdp_vs_plain']:.2e}; "
        f"(lr 3e-4) ddp {res['lr3e-4_max_rel_diff_ddp_vs_plain']:.2e}, fsdp {res['lr3e-4_max_rel_diff_fsdp_vs_plain']:.2e}",
        flush=True,
    )

    # Read the FSDP checkpoint back on 1 GPU with load_policy (strict load), compute the CE on real
    # validation data, and compare it with the plain checkpoint of the same step.
    from zero.data.loader import PackedDataLoader

    m_fsdp, _ = load_policy(OUT / "fsdp" / "ckpt", device=dev)
    m_plain, _ = load_policy(OUT / "plain" / "ckpt", device=dev)
    val = PackedDataLoader(
        "out/gpu0-check/tiny/data/*_val_*.bin",
        seq_len=2048,
        batch_size=8,
        shuffle=False,
        device=dev,
    )
    xv, yv = val.next_batch()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        ce_f, ce_p = float(m_fsdp.eval().loss(xv, yv)), float(m_plain.eval().loss(xv, yv))
    res["fsdp_ckpt_loaded_keys"] = len(m_fsdp.state_dict())
    res["val_ce_fsdp_ckpt_vs_plain_ckpt"] = [ce_f, ce_p]
    print(
        f"FSDP checkpoint read back with load_policy: {res['fsdp_ckpt_loaded_keys']} tensors (strict); "
        f"validation CE {ce_f:.4f} (FSDP) vs {ce_p:.4f} (plain), relative difference {abs(ce_f - ce_p) / ce_p:.2e}",
        flush=True,
    )

    # FSDP resume: copy the fsdp directory, keep only step 25, then train to step 50
    src, dst = OUT / "fsdp", OUT / "fsdp_resume"
    shutil.copytree(src, dst)
    shutil.rmtree(dst / "ckpt" / "step_00000050")
    (dst / "ckpt" / "latest").write_text("step_00000025")
    (dst / "log.jsonl").unlink()
    cfg = make_cfg("fsdp_resume")
    trainer = Trainer(cfg, forced, log=lambda m: print(f"[fsdp_resume] {m}", flush=True))
    assert trainer.step == 25, trainer.step
    hist = trainer.train()
    resumed = [h["loss"] for h in hist]
    res["fsdp_resume_steps_26_50"] = resumed
    res["fsdp_resume_max_abs_diff"] = max(
        abs(a - b) for a, b in zip(resumed, res["fsdp"][25:], strict=True)
    )
    print(
        f"FSDP resumed from step 25 to 50: max absolute difference from the uninterrupted loss {res['fsdp_resume_max_abs_diff']:.2e}",
        flush=True,
    )

    dist.destroy_process_group()
    (OUT / "result.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if not isinstance(v, list)}, indent=1))


if __name__ == "__main__":
    main()
