"""The general training loop (Chapters 6, 12, 14, and 15).

Pretraining and mid-training use this one loop. They differ only in two things:
the config (data mixture, learning-rate schedule, sequence length, RoPE), and
whether the training starts from existing weights (`train.init_from`).
One training step does these things:

1. Set the learning rate from the schedule.
2. Gradient accumulation: take grad_accum_steps micro-batches in sequence. Do a forward pass and
   a backward pass for each one, and add the gradients. The code divides each loss by the number
   of accumulation steps first, so the sum is the mean gradient of the large batch.
   With DDP, only the last micro-batch synchronizes the gradients (`no_sync`).
   This removes communication that is not necessary.
3. Gradient clipping: if the global gradient norm is more than grad_clip, scale all gradients down
   by the same factor. This prevents a rare large gradient from breaking the training.
4. AdamW update. Weight decay applies only to matrix weights. RMSNorm weights get no decay:
   decay pulls them toward 0, and that makes the normalization weaker.
   By default, the embedding also gets no decay (decay_embeddings changes this).
5. At regular intervals: print and record the log (loss, learning rate, tokens, throughput, MFU),
   compute the validation loss, and save a checkpoint.

Precision: on CUDA, use BF16 autocast. Matrix multiplications run in BF16, and the parameters
and the optimizer state stay in FP32. On CPU, the default is FP32.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import random
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from zero.config import Config
from zero.data.loader import MaskedWindowLoader, PackedDataLoader
from zero.data.mixture import MixtureLoader
from zero.model import Transformer, cross_entropy_loss
from zero.train.checkpoint import find_latest, load_checkpoint, load_model_weights, save_checkpoint
from zero.train.dist import DistInfo, all_reduce_mean, init_distributed, unwrap_model, wrap_model
from zero.train.schedule import LRScheduler


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_optimizer(model: nn.Module, cfg: Any, device: torch.device) -> torch.optim.AdamW:
    """Make AdamW with two parameter groups: with weight decay and without weight decay.

    If optim.name = "muon", use Muon instead (Chapter 12).
    """
    if getattr(cfg, "name", "adamw") == "muon":
        from .muon import build_muon_optimizer

        return build_muon_optimizer(model, cfg, device)
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_embedding = "tok_emb" in name or "lm_head" in name
        if p.dim() < 2 or (is_embedding and not cfg.decay_embeddings):
            no_decay.append(p)
        else:
            decay.append(p)
    groups = [
        {"params": decay, "weight_decay": cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    kwargs: dict[str, Any] = {}
    if device.type == "cuda":
        # Fused AdamW kernel: verified on one RTX 3090 (2026-10, see runs/2026-10-01-gpu0-check/).
        kwargs["fused"] = True
    return torch.optim.AdamW(groups, lr=cfg.lr, betas=(cfg.beta1, cfg.beta2), eps=cfg.eps, **kwargs)


def build_train_loader(
    cfg: Config, info: DistInfo
) -> PackedDataLoader | MixtureLoader | MaskedWindowLoader:
    d = cfg.train.data
    bsz = cfg.train.micro_batch_size
    if d.format == "sft":
        # SFT: conversation windows + loss mask (Chapter 16). The loader concatenates all sources;
        # their weights have no effect.
        return MaskedWindowLoader(
            [s.path for s in d.sources],
            d.seq_len,
            bsz,
            rank=info.rank,
            world_size=info.world_size,
            seed=cfg.train.seed,
            shuffle=d.shuffle,
            device=info.device,
        )
    if d.format != "packed":
        raise ValueError(f"[data] format={d.format!r}: the general training loop does not read this data format")
    common = dict(
        seq_len=d.seq_len,
        rank=info.rank,
        world_size=info.world_size,
        shuffle=d.shuffle,
        device=info.device,
    )
    if len(d.sources) == 1:
        s = d.sources[0]
        return PackedDataLoader(s.path, batch_size=bsz, seed=cfg.train.seed, **common)  # type: ignore[arg-type]
    loaders = {
        s.name: PackedDataLoader(s.path, batch_size=1, seed=cfg.train.seed + i, **common)  # type: ignore[arg-type]
        for i, s in enumerate(d.sources)
    }
    weights = {s.name: s.weight for s in d.sources}
    return MixtureLoader(
        loaders, weights, batch_size=bsz, seed=cfg.train.seed, rank=info.rank, device=info.device
    )


def autocast_context(device: torch.device, dtype_pref: str) -> contextlib.AbstractContextManager:
    if device.type == "cuda" and dtype_pref in ("auto", "bf16"):
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if device.type == "cpu" and dtype_pref == "bf16":
        return torch.autocast(device_type="cpu", dtype=torch.bfloat16)
    return contextlib.nullcontext()


class Trainer:
    def __init__(
        self, cfg: Config, info: DistInfo | None = None, log: Callable[[str], None] | None = None
    ) -> None:
        self.cfg = cfg
        self.info = info or init_distributed(cfg.train.device)
        self._print = log or print
        tc = cfg.train
        if tc.cpu_threads > 0:
            torch.set_num_threads(tc.cpu_threads)
        seed_everything(tc.seed)  # All ranks use the same seed, so the initial parameters are the same.
        self.device = self.info.device

        model = Transformer(cfg.model).to(self.device)
        model.activation_checkpointing = cfg.train.activation_checkpointing
        self.init_meta: dict[str, Any] | None = None
        resume_path = find_latest(tc.checkpoint_dir) if tc.checkpoint.resume else None
        if tc.init_from and resume_path is None:
            # Mid-training: load only the weights. The RoPE cos/sin are non-persistent buffers,
            # so the model computes them again from the new config (new theta / YaRN).
            self.init_meta = load_model_weights(tc.init_from, model)
            self.log(f"Loaded model weights from {tc.init_from} (step {self.init_meta.get('step')})")
        self.raw_model = model
        if tc.compile:
            # Verified with 1 GPU and with 2-GPU DDP on RTX 3090 (2026-10, see runs/2026-10-01-gpu0-check/).
            # The combination with FSDP is not verified on GPU yet.
            model = torch.compile(model)  # type: ignore[assignment]
        self.model = wrap_model(model, self.info, tc.parallel)
        self.optimizer = build_optimizer(self.model, tc.optim, self.device)
        self.scheduler = LRScheduler(self.optimizer, tc.schedule, tc.optim.lr, tc.max_steps)
        self.loader = build_train_loader(cfg, self.info)
        self.autocast = lambda: autocast_context(self.device, tc.dtype)

        self.step = 0
        self.tokens_seen = 0
        self.history: list[dict[str, Any]] = []
        if resume_path is not None:
            meta = load_checkpoint(
                resume_path,
                self.model,
                self.optimizer,
                self.scheduler,
                self.loader,
                self.info,
                tc.parallel,
            )
            self.step = int(meta["step"])
            self.tokens_seen = int(meta.get("tokens_seen", 0))
            self.log(f"Resumed training from {meta['path']} (step {self.step})")

        self.tokens_per_step = (
            tc.micro_batch_size * tc.grad_accum_steps * self.info.world_size * tc.data.seq_len
        )
        self.flops_per_token = unwrap_model(self.raw_model).flops_per_token(tc.data.seq_len)  # type: ignore[operator]
        self.peak_flops = self._peak_flops()

    # ---- Helpers ----
    def log(self, msg: str) -> None:
        if self.info.is_main:
            self._print(msg)

    def _peak_flops(self) -> float | None:
        if self.cfg.train.gpu_peak_tflops > 0:
            return self.cfg.train.gpu_peak_tflops * 1e12
        if self.device.type == "cuda":
            from zero.tools.estimate_cost import peak_tflops_for_device_name

            t = peak_tflops_for_device_name(torch.cuda.get_device_name(self.device))
            return t * 1e12 if t else None
        return None  # No MFU on CPU.

    def _write_jsonl(self, record: dict[str, Any]) -> None:
        if not self.info.is_main:
            return
        path = Path(self.cfg.train.log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def extra_metrics(self) -> dict[str, Any]:
        """Return extra metrics for the log. Subclasses (for example, distillation) override this."""
        return {}

    def _forward_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        with self.autocast():
            logits = self.model(x)
        return cross_entropy_loss(logits, y)

    # ---- Evaluation ----
    @torch.no_grad()
    def evaluate(self) -> float | None:
        d = self.cfg.train.data
        if not d.val:
            return None
        val_cls = MaskedWindowLoader if d.format == "sft" else PackedDataLoader
        val = val_cls(
            d.val,
            seq_len=d.seq_len,
            batch_size=self.cfg.train.micro_batch_size,
            rank=self.info.rank,
            world_size=self.info.world_size,
            shuffle=False,
            device=self.device,
        )
        self.model.eval()
        losses = []
        n = min(
            self.cfg.train.eval_batches,
            max(val.total_chunks // (val.batch_size * self.info.world_size), 1),
        )
        batches = [val.next_batch() for _ in range(n)]
        for x, y in batches:
            losses.append(self._forward_loss(x, y).float())
        self.model.train()
        loss = all_reduce_mean(torch.stack(losses).mean())
        self.last_val_bpb = self._val_bpb(batches)
        return float(loss.item())

    def _val_bpb(self, batches) -> float | None:
        """Validation bits per byte (Chapters 7 and 13).

        This metric does not depend on the tokenizer, so you can compare different vocabularies.
        Compute it only for pretraining-format data when the config gives a tokenizer path.
        Otherwise, return None.
        """
        d = self.cfg.train.data
        if d.format == "sft" or not d.tokenizer:
            return None
        if getattr(self, "_token_bytes", None) is None:
            from ..data.bpb import token_byte_lengths
            from ..tokenizer import Tokenizer

            self._token_bytes = token_byte_lengths(Tokenizer.load(d.tokenizer), self.device)
        from ..data.bpb import bpb_stats

        stats = bpb_stats(
            self.model, batches, self._token_bytes, autocast=getattr(self, "autocast", None)
        )
        return stats.bpb if stats.bytes > 0 else None

    def save(self) -> Path:
        tc = self.cfg.train
        return save_checkpoint(
            tc.checkpoint_dir,
            self.step,
            self.model,
            self.optimizer,
            self.scheduler,
            self.loader,
            meta={"tokens_seen": self.tokens_seen, "config": self.cfg.to_dict()},
            info=self.info,
            parallel=tc.parallel,
            keep_last=tc.checkpoint.keep_last,
        )

    # ---- Main loop ----
    def train(self, stop_at: int | None = None) -> list[dict[str, Any]]:
        """Train until max_steps.

        stop_at is for tests of "the training stops halfway": at this step, return
        (and do not save an extra checkpoint). If stop_at is not given, use train.stop_step
        from the config (0 = do not stop early).
        """
        tc = self.cfg.train
        if stop_at is None and tc.stop_step > 0:
            stop_at = tc.stop_step
        accum = tc.grad_accum_steps
        use_no_sync = self.info.is_distributed and tc.parallel == "ddp"
        self.model.train()
        if self.step == 0 and self.info.is_main:
            m = self.raw_model
            self.log(
                f"Model parameters {m.num_params() / 1e6:.2f}M (non-embedding {m.num_params(non_embedding=True) / 1e6:.2f}M), "
                f"{self.tokens_per_step} tokens per step, {tc.max_steps} steps, device {self.device}, world_size={self.info.world_size}"
            )
        t_last = time.perf_counter()
        steps_since_log = 0
        t_start = time.perf_counter()
        while self.step < tc.max_steps:
            if stop_at is not None and self.step >= stop_at:
                break
            lr = self.scheduler.apply(self.step)
            loss_sum = torch.zeros((), device=self.device)
            for micro in range(accum):
                x, y = self.loader.next_batch()
                sync_ctx = (
                    self.model.no_sync()
                    if use_no_sync and micro < accum - 1
                    else contextlib.nullcontext()  # type: ignore[operator]
                )
                with sync_ctx:
                    loss = self._forward_loss(x, y)
                    (loss / accum).backward()
                loss_sum += loss.detach()
            if tc.optim.grad_clip > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), tc.optim.grad_clip
                )
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), float("inf"))
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.step += 1
            self.tokens_seen += self.tokens_per_step
            steps_since_log += 1

            is_last = self.step == tc.max_steps
            do_log = self.step % tc.logging.every == 0 or is_last or self.step == 1
            do_eval = tc.eval_every > 0 and (self.step % tc.eval_every == 0 or is_last)
            record: dict[str, Any] | None = None
            if do_log or do_eval:
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                dt = time.perf_counter() - t_last
                loss_val = float(all_reduce_mean(loss_sum / accum).item())
                tok_per_s = self.tokens_per_step * steps_since_log / max(dt, 1e-9)
                mfu = (
                    self.flops_per_token * tok_per_s / (self.peak_flops * self.info.world_size)
                    if self.peak_flops
                    else None
                )
                record = {
                    "step": self.step,
                    "loss": loss_val,
                    "lr": lr,
                    "grad_norm": float(grad_norm),
                    "tokens": self.tokens_seen,
                    "tok_per_s": tok_per_s,
                    "mfu": mfu,
                    "elapsed_s": time.perf_counter() - t_start,
                }
                if self.device.type == "cuda":  # for the launch check (zero/tools/launch_check.py)
                    record["max_mem_gb"] = torch.cuda.max_memory_allocated(self.device) / 2**30
                if isinstance(self.loader, MixtureLoader):
                    record["mixture_counts"] = dict(self.loader.counts)
                extra = self.extra_metrics()
                record.update(extra)
                if do_eval:
                    record["val_loss"] = self.evaluate()
                    if getattr(self, "last_val_bpb", None) is not None:
                        record["val_bpb"] = self.last_val_bpb
                if not math.isfinite(loss_val):
                    self.log(f"step {self.step}: loss is {loss_val}, training stops")
                    raise FloatingPointError(f"Loss diverged: {loss_val}")
                self.history.append(record)
                self._write_jsonl(record)
                msg = (
                    f"step {self.step:>6}/{tc.max_steps} | loss {loss_val:.4f} | lr {lr:.2e} | "
                    f"gnorm {float(grad_norm):.2f} | {tok_per_s:,.0f} tok/s"
                )
                if mfu is not None:
                    msg += f" | MFU {mfu:.1%}"
                for k, v in extra.items():
                    msg += f" | {k} {v:.4f}"
                if record.get("val_loss") is not None:
                    msg += f" | val {record['val_loss']:.4f}"
                self.log(msg)
                t_last = time.perf_counter()
                steps_since_log = 0

            every = tc.checkpoint.every
            if (every > 0 and self.step % every == 0) or is_last:
                path = self.save()
                self.log(f"checkpoint saved: {path}")
                t_last = time.perf_counter()
        return self.history


def run_training(cfg: Config, log: Callable[[str], None] | None = None) -> list[dict[str, Any]]:
    """Entry point for scripts: init distributed → prepare data (rank 0 only) → train → clean up."""
    from zero.data.prepare import prepare_data
    from zero.train.dist import barrier, cleanup

    info = init_distributed(cfg.train.device)
    try:
        if cfg.train.data.prepare is not None:
            if info.is_main:
                prepare_data(cfg.train.data, seed=cfg.train.seed, log=log or print)
            barrier()
        os.makedirs(cfg.train.out_dir, exist_ok=True)
        trainer = Trainer(cfg, info, log)
        return trainer.train()
    finally:
        cleanup()
