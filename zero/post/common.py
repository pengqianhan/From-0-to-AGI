"""Small helpers that all post-training stages share (Chapters 16–19).

- `load_post_config`: read the TOML file. `zero.config` parses the common sections ([model] [train]
  [data] [optim] [schedule] [checkpoint] [logging]). This module parses the sections of each stage
  ([sft] [distill] [teacher] [dpo] [grpo] [eval]) into dataclasses, with the same check for unknown fields.
- `load_policy`: load (model, tokenizer) from a zero checkpoint folder or an exported Hugging Face folder.
- `chat_complete`: apply the chat template → generate with the KV cache → decode.
- `token_logprobs` / `pad_batch`: DPO, GRPO, and distillation all need "the log probability of each token".
- Read and write JSONL.
- Data parallelism for the stages that do not use `Trainer` (DPO, GRPO, on-policy distillation):
  `rank_share`, `all_reduce_sum` / `all_reduce_max`, and the gradient all-reduce in `LoopState`.
  For the slow waits (with a long timeout, see `SLOW_WAIT_MIN`): `run_on_rank0` for work that only
  rank 0 does (packing, writing files), `all_gather_objects`, `wait_all`.

**Data parallelism of DPO / GRPO / OPD** (`torchrun --nproc_per_node=N -m zero.post.grpo ...`):
every rank holds a full copy of the policy (0.7B + AdamW fits on one 80GB GPU). The batch sizes in
the config are **global** (prompts per step, pairs per step); the ranks split them, so the recipe does
not change with the number of GPUs. Each rank divides its loss by the *global* normalizer (all
response tokens, or all pairs, of the step), and `LoopState.optimizer_step` **sums** the gradients
over the ranks. The sum of the per-rank gradients is then exactly the gradient of one process on the
whole batch (`tests/test_post_ddp.py` checks this with 2 CPU processes). Samples are seeded by their
global index, not by the rank, so the rollouts do not depend on the number of GPUs either.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any, TypeVar

import torch
import torch.nn.functional as F

from zero.config import (
    Config,
    ConfigError,
    ModelConfig,
    _apply_override,
    _from_dict,
    _read_toml_with_base,
    config_from_dict,
)
from zero.model import Transformer
from zero.post.chat import render
from zero.tokenizer import Tokenizer

T = TypeVar("T")

CORE_SECTIONS = {"model", "train", "data", "optim", "schedule", "checkpoint", "logging"}


def load_post_config(
    src: str | os.PathLike | dict[str, Any],
    sections: dict[str, type],
    overrides: Sequence[str] | None = None,
) -> tuple[Config, dict[str, Any]]:
    """Read the configuration. Return (common Config, {section name: dataclass instance}).

    src can be a path or a dict that is already expanded.
    """
    if isinstance(src, dict):
        data = json.loads(json.dumps(src))  # deep copy
        source = "<dict>"
    else:
        data = _read_toml_with_base(Path(src))
        source = str(src)
    for item in overrides or []:
        _apply_override(data, item)
    extra = {k: data.pop(k) for k in list(data) if k not in CORE_SECTIONS}
    unknown = set(extra) - set(sections)
    if unknown:
        raise ConfigError(
            f"Unknown config section {sorted(unknown)}. This stage accepts: {sorted(sections)}"
        )
    parsed = {
        name: _from_dict(cls, extra.get(name, {}), f"[{name}]") for name, cls in sections.items()
    }
    return config_from_dict(data, source_path=source), parsed


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_policy(
    path: str | os.PathLike,
    tokenizer_path: str | os.PathLike | None = None,
    device: str | torch.device = "cpu",
    model_overrides: dict[str, Any] | None = None,
) -> tuple[Transformer, Tokenizer]:
    """Load (model, tokenizer).

    path can be:
    - an exported HF folder (with config.json + *.safetensors + tokenizer.json);
    - a zero checkpoint folder (`<out>/ckpt` or one `step_xxx`). The tokenizer path comes from the
      configuration in meta.json, or from tokenizer_path.
    model_overrides: for example {"max_seq_len": 1024} (change only the fields related to RoPE).
    """
    from zero.hf import load_from_hf_qwen3
    from zero.train.checkpoint import find_latest

    p = Path(path)
    if (p / "config.json").exists():
        model = load_from_hf_qwen3(p)
        if model_overrides:
            cfg = ModelConfig(**{**model.config.__dict__, **model_overrides})
            sd = model.state_dict()
            model = Transformer(cfg)
            model.load_state_dict(sd)
        tok = Tokenizer.load(Path(tokenizer_path) if tokenizer_path else p / "tokenizer.json")
        return model.to(device), tok
    ckpt = find_latest(p)
    if ckpt is None:
        raise FileNotFoundError(f"{path} is not an HF folder, and it contains no zero checkpoint")
    meta = json.loads((ckpt / "meta.json").read_text())
    mcfg = dict(meta["config"]["model"])
    mcfg.update(model_overrides or {})
    model = Transformer(ModelConfig(**mcfg))
    sd = torch.load(ckpt / "model.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(sd)
    tp = Path(tokenizer_path or meta["config"]["train"]["data"]["tokenizer"])
    if not tp.is_absolute() and not tp.exists():
        # The configuration stores a path relative to the working folder of the training run (usually
        # the repository root). When we run from a different folder, resolve it from the repository root.
        repo_root = Path(__file__).resolve().parents[2]
        if (repo_root / tp).exists():
            tp = repo_root / tp
    return model.to(device), Tokenizer.load(tp)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


@torch.no_grad()
def chat_complete(
    model: Transformer,
    tok: Tokenizer,
    messages: Sequence[dict[str, Any]],
    tools: Sequence[dict[str, Any]] | None = None,
    max_new_tokens: int = 128,
    temperature: float = 0.0,
    top_p: float = 1.0,
    n: int = 1,
    seed: int | None = None,
) -> list[str]:
    """Generate n assistant replies (text, without <|im_end|>) for one conversation.

    The n samples run in parallel as one batch with the KV cache. They share the same prompt and have
    the same length, so no padding is necessary."""
    from zero.generate import generate

    ids, _ = render(messages, tools, add_generation_prompt=True, tokenizer=tok)
    budget = model.config.max_seq_len - len(ids)
    if budget <= 0:
        return ["" for _ in range(n)]
    device = next(model.parameters()).device
    prompt = torch.tensor([ids] * n, dtype=torch.long, device=device)
    outs = generate(
        model,
        prompt,
        min(max_new_tokens, budget),
        temperature=temperature,
        top_p=top_p,
        eos_id=tok.im_end_id,
        seed=seed,
    )
    return [tok.decode(o) for o in outs]  # type: ignore[arg-type]


def make_policy(
    model: Transformer, tok: Tokenizer, max_new_tokens: int = 128, temperature: float = 0.0
):  # noqa: ANN201
    """Wrap the model as policy(messages, tools) -> text, which tool_env.run_episode needs."""

    def policy(messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> str:
        return chat_complete(model, tok, messages, tools, max_new_tokens, temperature)[0]

    return policy


# ---------------------------------------------------------------------------
# Log probabilities
# ---------------------------------------------------------------------------


def pad_batch(
    seqs: Sequence[Sequence[int]],
    masks: Sequence[Sequence[bool]],
    pad_id: int,
    device: str | torch.device = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad on the right. Return ids (B, T) and mask (B, T) (mask=False on the padding).

    With causal attention, padding on the right has no effect on the earlier positions.
    """
    L = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), L), pad_id, dtype=torch.long)
    m = torch.zeros((len(seqs), L), dtype=torch.bool)
    for i, (s, mk) in enumerate(zip(seqs, masks)):
        ids[i, : len(s)] = torch.tensor(list(s), dtype=torch.long)
        m[i, : len(mk)] = torch.tensor(list(mk), dtype=torch.bool)
    return ids.to(device), m.to(device)


def token_logprobs(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """logits (B, T, V), targets (B, T) → log probability of each target token (B, T), float32."""
    return torch.gather(F.log_softmax(logits.float(), dim=-1), -1, targets.unsqueeze(-1)).squeeze(
        -1
    )


def sequence_token_logprobs(
    model: torch.nn.Module, ids: torch.Tensor, mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Do one forward pass on the full sequence. Return (logp, target_mask), both with shape (B, T-1):
    logp[:, t] = log π(ids[:, t+1] | ids[:, :t+1]), target_mask = mask[:, 1:]."""
    logits = model(ids[:, :-1])
    return token_logprobs(logits, ids[:, 1:]), mask[:, 1:]


# ---------------------------------------------------------------------------
# JSONL
# ---------------------------------------------------------------------------


def read_jsonl(path: str | os.PathLike) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: str | os.PathLike, rows: Iterable[dict[str, Any]]) -> int:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def set_threads(n: int) -> None:
    if n > 0:
        torch.set_num_threads(n)


# ---------------------------------------------------------------------------
# Data parallelism helpers (see the module docstring)
# ---------------------------------------------------------------------------


def rank_share(items: Sequence[T], info: Any) -> list[tuple[int, T]]:
    """This rank's part of a global list, with the global index of each item: items[rank::world]."""
    return [(i, x) for i, x in enumerate(items) if i % info.world_size == info.rank]


def all_reduce_sum(values: Sequence[float], info: Any) -> list[float]:
    """Sum of each value over the ranks (float64). With one process, the values unchanged."""
    if not getattr(info, "is_distributed", False):
        return [float(v) for v in values]
    import torch.distributed as dist

    t = torch.tensor([float(v) for v in values], dtype=torch.float64, device=info.device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t.tolist()


def all_reduce_max(values: Sequence[float], info: Any) -> list[float]:
    if not getattr(info, "is_distributed", False):
        return [float(v) for v in values]
    import torch.distributed as dist

    t = torch.tensor([float(v) for v in values], dtype=torch.float64, device=info.device)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return t.tolist()


def rank0_log(info: Any, log: Callable[[str], None]) -> Callable[[str], None]:
    """log on rank 0; nothing on the other ranks (with 8 GPUs, every line would print 8 times)."""
    return log if info.is_main else (lambda _msg: None)


# Slow waits: rank 0 packs data while the others wait, or the ranks finish their shares of sampling at
# different times. The stages with such waits call init_distributed(slow_wait_min=SLOW_WAIT_MIN); the
# helpers below wait on `info.wait_group` (gloo, this timeout) instead of the default group (10 minutes
# with NCCL, which stays short for training).
SLOW_WAIT_MIN = 120.0


def wait_all(info: Any) -> None:
    """A barrier for a slow wait (on `info.wait_group` when there is one)."""
    if getattr(info, "is_distributed", False):
        import torch.distributed as dist

        dist.barrier(group=getattr(info, "wait_group", None))


def all_gather_objects(obj: Any, info: Any) -> list[Any]:
    """The obj of every rank, in rank order, on every rank (pickled). With one process, [obj].

    Every rank waits for the slowest one (on `info.wait_group` when there is one).
    """
    if not getattr(info, "is_distributed", False):
        return [obj]
    import torch.distributed as dist

    out: list[Any] = [None] * info.world_size
    dist.all_gather_object(out, obj, group=getattr(info, "wait_group", None))
    return out


def run_on_rank0(info: Any, fn: Callable[[], T]) -> T | None:
    """Run fn() on rank 0 only; the other ranks wait (on `info.wait_group`) and get None.

    An error on rank 0 reaches every rank: the others raise too, instead of waiting in a barrier until
    the collective timeout hides the real error.
    """
    if not getattr(info, "is_distributed", False):
        return fn()
    import torch.distributed as dist

    out, err = None, None
    if info.is_main:
        try:
            out = fn()
        except Exception as e:  # noqa: BLE001 - re-raised below, after the other ranks are told
            err = e
    group = getattr(info, "wait_group", None)
    device = "cpu" if group is not None else info.device  # the wait group is gloo
    flag = torch.tensor([0.0 if err is None else 1.0], device=device)
    dist.all_reduce(flag, group=group)
    if err is not None:
        raise err
    if flag.item() > 0:
        raise RuntimeError("rank 0 failed (see its log), so this rank stops too")
    return out


def all_reduce_grads(model: torch.nn.Module, info: Any, bucket_numel: int = 1 << 25) -> None:
    """Sum the gradients over the ranks, in buckets of at most bucket_numel elements (128MB in FP32).

    A parameter without a gradient on this rank (for example, the rank had no tokens this step) gets
    zeros, so that every rank sends the same buckets.
    """
    if not getattr(info, "is_distributed", False):
        return
    import torch.distributed as dist
    from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

    grads = []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        if p.grad is None:
            p.grad = torch.zeros_like(p)
        grads.append(p.grad)
    bucket: list[torch.Tensor] = []
    size = 0

    def flush() -> None:
        if not bucket:
            return
        flat = _flatten_dense_tensors(bucket)
        dist.all_reduce(flat, op=dist.ReduceOp.SUM)
        for g, f in zip(bucket, _unflatten_dense_tensors(flat, bucket)):
            g.copy_(f)
        bucket.clear()

    for g in grads:
        if size + g.numel() > bucket_numel and bucket:
            flush()
            size = 0
        bucket.append(g)
        size += g.numel()
    flush()


# ---------------------------------------------------------------------------
# The training-loop skeleton that DPO, GRPO, and on-policy distillation share: optimizer, learning
# rate, gradient clipping, logging, checkpoint, and resume
# ---------------------------------------------------------------------------


class LoopState:
    """The shared parts of the training stages that do not use `Trainer` (their data is not a fixed token stream).

    Usage:
        loop = LoopState(cfg, model, log)       # make optimizer/scheduler; resume if a checkpoint exists
        while loop.step < max_steps:
            lr = loop.begin_step()
            ... loss.backward() ...
            gnorm = loop.end_step()             # clip + update + step += 1
            loop.record({...})                  # log, write JSONL, save a checkpoint when necessary
    """

    def __init__(
        self,
        cfg: Config,
        model: torch.nn.Module,
        log: Any = print,
        info: Any = None,
    ) -> None:
        from zero.train.checkpoint import find_latest, load_checkpoint
        from zero.train.dist import DistInfo
        from zero.train.schedule import LRScheduler
        from zero.train.trainer import build_optimizer

        self.cfg = cfg
        self.info = info or DistInfo(device=next(model.parameters()).device)
        self.model = model
        self._log = log
        tc = cfg.train
        self.optimizer = build_optimizer(model, tc.optim, self.info.device)
        self.scheduler = LRScheduler(self.optimizer, tc.schedule, tc.optim.lr, tc.max_steps)
        self.step = 0
        self.history: list[dict[str, Any]] = []
        self.resumed_from: str | None = None
        latest = find_latest(tc.checkpoint_dir) if tc.checkpoint.resume else None
        if latest is not None:
            meta = load_checkpoint(latest, model, self.optimizer, self.scheduler, None, self.info)
            self.step = int(meta["step"])
            self.resumed_from = meta["path"]
            self.log(f"Resumed training from {meta['path']} (step {self.step})")

    def log(self, msg: str) -> None:
        if self.info.is_main and self._log is not None:
            self._log(msg)

    def begin_step(self) -> float:
        return self.scheduler.apply(self.step)

    def optimizer_step(self) -> float:
        """Gradient all-reduce (sum, with several processes) + clipping + one parameter update.

        This does not increase the step count. With GRPO ppo_epochs > 1, one step has several updates.
        """
        all_reduce_grads(self.model, self.info)
        clip = self.cfg.train.optim.grad_clip
        gnorm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), clip if clip > 0 else float("inf")
        )
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        return float(gnorm)

    def end_step(self) -> float:
        gnorm = self.optimizer_step()
        self.step += 1
        return gnorm

    def record(self, rec: dict[str, Any], fmt: str = "") -> None:
        tc = self.cfg.train
        rec = {"step": self.step, **rec}
        if self.info.device.type == "cuda":  # for the launch check (zero/tools/launch_check.py)
            rec["max_mem_gb"] = torch.cuda.max_memory_allocated(self.info.device) / 2**30
        is_last = self.step >= tc.max_steps
        if self.step % tc.logging.every == 0 or is_last or self.step == 1:
            self.history.append(rec)
            if self.info.is_main:
                p = Path(tc.log_path)
                p.parent.mkdir(parents=True, exist_ok=True)
                with open(p, "a") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            self.log(fmt.format(**rec) if fmt else json.dumps(rec, ensure_ascii=False))
        every = tc.checkpoint.every
        if (every > 0 and self.step % every == 0) or is_last:
            self.save()

    def save(self, extra_meta: dict[str, Any] | None = None) -> Path:
        from zero.train.checkpoint import save_checkpoint

        tc = self.cfg.train
        return save_checkpoint(
            tc.checkpoint_dir,
            self.step,
            self.model,
            self.optimizer,
            self.scheduler,
            None,
            meta={"config": self.cfg.to_dict(), **(extra_meta or {})},
            info=self.info,
            keep_last=tc.checkpoint.keep_last,
        )


def build_model_from_init(cfg: Config, device: str | torch.device = "cpu") -> Transformer:
    """Build the model from cfg.model, and load the weights from cfg.train.init_from (a zero checkpoint)."""
    from zero.train.checkpoint import load_model_weights

    model = Transformer(cfg.model).to(device)
    if cfg.train.init_from:
        load_model_weights(cfg.train.init_from, model)
    return model
