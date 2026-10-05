"""阶段 6 第 4 项（单卡版）：在 1 张卡上真正走一遍 DDP / FSDP2 的代码路径。

`torchrun --nproc_per_node=1` 时 WORLD_SIZE=1，`DistInfo.is_distributed` 为 False，`wrap_model` 原样返回模型——
`--set train.parallel=fsdp` 在单卡上**什么都不做**，DDP 也一样。为了让 wrap_model / checkpoint 里 FSDP 的分支
真的执行，这里自己建一个 world_size=1 的 NCCL 进程组，并把 `is_distributed` 强制为 True：

    CUDA_VISIBLE_DEVICES=0 UV_NO_SYNC=1 uv run python runs/2026-10-01-gpu0-check/dist1_check.py

同一份配置（l20m 形状 + tiny 数据）跑 50 步（打开 torch.use_deterministic_algorithms，学习率 3e-3 与 3e-4 各一组）：
  plain（不包装，与 torchrun 单卡相同）/ ddp（DDP，NCCL，1 个 rank）/ fsdp（fully_shard + MixedPrecisionPolicy）
比较逐步 loss；FSDP 的 checkpoint（get_model_state_dict 聚合）用 `load_policy` 在单卡上读回并对拍 logits；
再从 FSDP 的 step 25 checkpoint 续训到 50（set_model_state_dict / set_optimizer_state_dict），与不中断的 loss 对比。
只能说明"单卡通路能跑、数值合理"，多卡切分与通信仍需多卡验证。
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
    """world_size=1 但假装是分布式，让 wrap_model / checkpoint 走 DDP、FSDP 分支。"""

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
        f"[{mode}] 包装后类型 {type(trainer.model).__name__}；optimizer fused="
        f"{trainer.optimizer.defaults.get('fused')}；显存峰值 {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB",
        flush=True,
    )
    losses = [h["loss"] for h in hist]
    print(f"[{mode}] loss step1 {losses[0]:.4f} → step{STEPS} {losses[-1]:.4f}", flush=True)
    return losses


def main() -> None:
    assert torch.cuda.device_count() == 1, "只允许看到 1 张卡（CUDA_VISIBLE_DEVICES=0）"
    if OUT.exists():
        shutil.rmtree(OUT)
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29533")
    torch.cuda.set_device(0)
    dist.init_process_group("nccl", rank=0, world_size=1, device_id=torch.device("cuda", 0))
    dev = torch.device("cuda", 0)
    # 确定性算法：GPU 默认内核（FlashAttention 反向等）有运行间差异，会淹没 DDP / FSDP 通路本身的数值差
    torch.use_deterministic_algorithms(True)
    plain = DistInfo(device=dev)
    forced = ForcedDistInfo(rank=0, local_rank=0, world_size=1, device=dev, backend="nccl")

    def rel(a: list[float], b: list[float]) -> float:
        return max(abs(x - y) / abs(y) for x, y in zip(a, b, strict=True))

    res: dict = {}
    # 学习率 3e-4：比阶梯默认的 3e-3 小 10 倍，早期不那么"混沌"，更容易看出通路本身的数值差
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
        f"逐步 loss 最大相对差（lr 3e-3）：ddp vs plain {res['max_rel_diff_ddp_vs_plain']:.2e}"
        f"（逐位相同：{res['ddp_bitwise_equal_plain']}），fsdp vs plain {res['max_rel_diff_fsdp_vs_plain']:.2e}；"
        f"（lr 3e-4）ddp {res['lr3e-4_max_rel_diff_ddp_vs_plain']:.2e}，fsdp {res['lr3e-4_max_rel_diff_fsdp_vs_plain']:.2e}",
        flush=True,
    )

    # FSDP checkpoint 用 load_policy 在单卡上读回（strict 加载），在真实验证数据上算 CE，与 plain 的同步 checkpoint 比较
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
        f"FSDP checkpoint 经 load_policy 读回：{res['fsdp_ckpt_loaded_keys']} 个张量（strict）；"
        f"验证集 CE {ce_f:.4f}（FSDP）vs {ce_p:.4f}（plain），相对差 {abs(ce_f - ce_p) / ce_p:.2e}",
        flush=True,
    )

    # FSDP 续训：把 fsdp 的目录复制一份，只留 step 25，再训练到 50
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
        f"FSDP 从 step 25 续训到 50：与不中断的 loss 最大绝对差 {res['fsdp_resume_max_abs_diff']:.2e}",
        flush=True,
    )

    dist.destroy_process_group()
    (OUT / "result.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if not isinstance(v, list)}, indent=1))


if __name__ == "__main__":
    main()
