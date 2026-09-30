"""阶段 6 第 2、4 项与 Muon 的多卡版：在 2 张 RTX 3090（PCIe，无 NVLink）上真正跑 DDP / FSDP2。

    # 1) 单卡参考（物理 GPU0）：不包装，micro 8 × 累积 2，每步 16 条 × 2048 token
    CUDA_VISIBLE_DEVICES=0 UV_NO_SYNC=1 uv run python runs/2026-10-01-gpu0-check/dist2_check.py ref
    # 1b) 单卡对照：micro 4 × 累积 4（与 1) 数学上等价，只有求和顺序不同）
    CUDA_VISIBLE_DEVICES=0 UV_NO_SYNC=1 uv run python runs/2026-10-01-gpu0-check/dist2_check.py ctrl
    # 2) 双卡（物理 GPU0 + GPU2，两张卡功耗上限都是 240 W）：同样每步 16 条
    CUDA_VISIBLE_DEVICES=0,2 UV_NO_SYNC=1 uv run torchrun --standalone --nproc_per_node=2 \
        runs/2026-10-01-gpu0-check/dist2_check.py multi

用单来源配置 configs/l20m_tiny_1src.toml：PackedDataLoader 把第 g 个全局样本分给 rank g % world_size，所以"2 卡各 8 条、不累积""2 卡各 4 条 × 累积 2"
与"1 卡 8 条 × 累积 2"每步看到的是同一批 16 条样本，梯度在数学上相同（只有求和顺序不同）。
全部打开 torch.use_deterministic_algorithms，排除 GPU 内核本身的运行间差异（见本目录 README"口径"第 3 条）。

（多来源配置下 MixtureLoader 的来源采样器按 (seed, rank) 播种，各卡各抽来源，与单卡不是同一批样本；
 用 DIST2_CFG=.../l20m_tiny.toml DIST2_OUT=out/gpu0-check/dist2_mixture 可以复现那组对照。）

multi 阶段依次跑（每个 50 步，l20m 形状 + tiny 数据）：
  ddp        DDP，micro 8 × 1            ddp_accum  DDP，micro 4 × 2（走 no_sync 梯度累积）
  fsdp       FSDP2，micro 8 × 1          muon_ddp   optim.name=muon，DDP，micro 8 × 1
  ddp / fsdp 另有 lr 3e-4 一组（比阶梯默认小 10 倍，早期不那么"混沌"，更容易看出通路本身的数值差）
每个 DDP 运行结束后，把每个参数按位解释成整数求和，在两张卡之间比较（参数是否逐位相同）；
FSDP 的 checkpoint 用 load_policy 在单卡上读回（strict），在验证数据上算 CE；
再把 fsdp 目录只留 step 25、在 2 卡上续训到 50，与不中断的 loss 比较（FSDP2 的 state_dict 聚合 / 分发）。
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

# 默认用单来源配置：多来源的 MixtureLoader 按 (seed, rank) 抽来源，2 卡与 1 卡的样本组成不同，无法逐步对拍
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
        # 各卡显存峰值
        peaks = [None] * info.world_size
        dist.all_gather_object(peaks, res["peak_gib"])
        res["peak_gib_per_rank"] = peaks
        if parallel == "ddp":
            # 参数逐位比较：把每个参数按位解释成整数求和（位级校验和），两张卡上必须完全相同
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
            f"[{name}] {res['wrapped']}，loss step1 {res['loss'][0]:.4f} → step{STEPS} {res['loss'][-1]:.4f}，"
            f"显存峰值 {res.get('peak_gib_per_rank', res['peak_gib'])}，"
            f"参数跨卡逐位相同：{res.get('params_bitwise_equal_across_ranks', '—')}",
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
    assert torch.cuda.device_count() == 1, "参考运行只允许看到 1 张卡"
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
    """对照：单卡上 micro 4 × 累积 4 与 micro 8 × 累积 2 数学上等价，只是求和顺序不同。
    它与单卡参考的分叉幅度，就是"求和顺序不同"本身会被训练放大到的程度，DDP / FSDP 的差异应在同一量级。"""
    assert torch.cuda.device_count() == 1, "对照运行只允许看到 1 张卡"
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

    # FSDP 续训：只留 step 25，在 2 卡上续训到 50
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
