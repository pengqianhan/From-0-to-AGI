"""Configuration: all hyperparameters of the model and the training (Chapters 9 and 12).

All hyperparameters are in `configs/*.toml`. The code does not hard-code them. This module does
three things:

1. It defines the "shape" of the config with dataclasses: `ModelConfig` (model structure) and
   `TrainConfig` (training procedure, with the data / optim / schedule / checkpoint / logging
   sections).
2. `load_config(path)` reads TOML with the standard library `tomllib`. A config can inherit from
   another config with `base = "xxx.toml"` (a mid-training config contains only the differences
   from pretraining).
3. Validation: unknown fields, wrong types, and values that conflict (for example, n_heads must be
   divisible by n_kv_heads) cause an error when the config is read. The error gives a hint such as
   "did you mean xxx". Thus a bad config does not crash in the middle of training.

The TOML layout (each section is a top-level table):

    base = "pretrain.toml"      # optional: read this file first, then override it with this file
    [model]      -> ModelConfig
    [train]      -> scalar fields of TrainConfig
    [data]       -> DataConfig (with [[data.sources]] and the optional [data.prepare])
    [optim]      -> OptimConfig
    [schedule]   -> ScheduleConfig
    [checkpoint] -> CheckpointConfig
    [logging]    -> LoggingConfig

All path fields are relative to the repository root (the current directory when you run a command).
"""

from __future__ import annotations

import dataclasses
import difflib
import tomllib
import types
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    """Raised when a config file has a problem. The message names the field."""


# ---------------------------------------------------------------------------
# Model structure
# ---------------------------------------------------------------------------

_ROPE_SCALING_KEYS = {
    "type",
    "factor",
    "original_max_position_embeddings",
    "beta_fast",
    "beta_slow",
    "attention_factor",
}


@dataclass
class ModelConfig:
    """Hyperparameters of the Transformer structure.

    Each field matches one field of the Hugging Face `Qwen3Config` (see `zero/hf.py`).
    """

    vocab_size: int = 32000
    dim: int = 512  # hidden dimension d_model (HF: hidden_size)
    n_layers: int = 8
    n_heads: int = 8  # number of query heads
    n_kv_heads: int = 4  # number of K/V heads (GQA: a group of query heads shares one K/V head)
    head_dim: int | None = None  # default dim // n_heads; Qwen3 allows n_heads*head_dim != dim
    ffn_dim: int = 1536  # SwiGLU intermediate dimension (HF: intermediate_size)
    rope_theta: float = 10000.0
    # RoPE scaling such as YaRN: {"type": "yarn", "factor": 4.0, "original_max_position_embeddings": 4096,
    #                             "beta_fast": 32, "beta_slow": 1}; None means no scaling
    rope_scaling: dict[str, Any] | None = None
    max_seq_len: int = 2048  # maximum number of positions that RoPE precomputes; also the default KV cache length
    norm_eps: float = 1e-6
    qk_norm: bool = True  # QK-Norm of Qwen3: RMSNorm on q and k of each head, over head_dim
    tie_embeddings: bool = True  # the input embedding and the output lm_head share weights
    init_std: float = 0.02

    def __post_init__(self) -> None:
        if self.head_dim is None and self.n_heads > 0:
            self.head_dim = self.dim // self.n_heads

    def validate(self) -> None:
        def need(cond: bool, msg: str) -> None:
            if not cond:
                raise ConfigError(f"[model] {msg}")

        for name in (
            "vocab_size",
            "dim",
            "n_layers",
            "n_heads",
            "n_kv_heads",
            "ffn_dim",
            "max_seq_len",
        ):
            need(getattr(self, name) > 0, f"{name} must be a positive integer, got {getattr(self, name)}")
        need(
            self.n_heads % self.n_kv_heads == 0,
            f"n_heads={self.n_heads} must be divisible by n_kv_heads={self.n_kv_heads} (GQA groups)",
        )
        assert self.head_dim is not None
        need(
            self.head_dim > 0 and self.head_dim % 2 == 0,
            f"head_dim={self.head_dim} must be a positive even number (RoPE rotates pairs)",
        )
        need(self.rope_theta > 0, "rope_theta must be > 0")
        need(self.norm_eps > 0, "norm_eps must be > 0")
        need(self.init_std > 0, "init_std must be > 0")
        if self.rope_scaling is not None:
            rs = self.rope_scaling
            unknown = set(rs) - _ROPE_SCALING_KEYS
            need(
                not unknown,
                f"rope_scaling has unknown fields {sorted(unknown)}; available: {sorted(_ROPE_SCALING_KEYS)}",
            )
            need(
                rs.get("type") == "yarn",
                f"rope_scaling.type supports only 'yarn' at this time, got {rs.get('type')!r}",
            )
            need(float(rs.get("factor", 0)) >= 1.0, "rope_scaling.factor must be >= 1")
            need(
                int(rs.get("original_max_position_embeddings", 0)) > 0,
                "rope_scaling.original_max_position_embeddings is required (the context length of pretraining)",
            )

    @property
    def q_dim(self) -> int:
        assert self.head_dim is not None
        return self.n_heads * self.head_dim

    @property
    def kv_dim(self) -> int:
        assert self.head_dim is not None
        return self.n_kv_heads * self.head_dim


# ---------------------------------------------------------------------------
# Training procedure
# ---------------------------------------------------------------------------


@dataclass
class DataSourceConfig:
    """One data source: a set of shards + a sampling weight.

    With more than one source, the loader mixes them by weight (see zero/data/mixture.py).
    """

    name: str = ""
    path: str = ""  # glob of the shards, for example "out/tiny/data/shakespeare_train_*.bin"
    weight: float = 1.0


@dataclass
class DataPrepareConfig:
    """Optional: if the shards do not exist, train a tokenizer on the raw text and make the shards.

    Use this only for the tiny smoke test.
    """

    raw_files: list[str] = field(default_factory=list)  # glob of the raw text; the file name without the extension is the source name
    out_dir: str = ""  # output directory of the shards
    vocab_size: int = 2048  # vocabulary size of the tokenizer to train (only if data.tokenizer does not exist)
    val_fraction: float = 0.05  # fraction of the documents of each source for the validation set
    doc_chars: int = 4000  # split the raw text at empty lines, then join the parts into "documents" of about this length


@dataclass
class DataConfig:
    tokenizer: str = ""  # path of tokenizer.json
    seq_len: int = 1024  # training sequence length (each sample takes seq_len+1 tokens)
    sources: list[DataSourceConfig] = field(default_factory=list)
    val: str = ""  # glob of the validation shards; empty means no evaluation
    shuffle: bool = True
    prepare: DataPrepareConfig | None = None
    # Data format: "packed" = pretraining shards (uint32 token stream, see PackedDataLoader in zero/data/loader.py);
    # "sft" = packed chat windows + loss mask (zero/post/sft.py writes them, MaskedWindowLoader reads them);
    # "none" = the training loop manages its own data (DPO / GRPO / distillation); [[data.sources]] can be empty
    format: str = "packed"


@dataclass
class OptimConfig:
    name: str = "adamw"  # "adamw" | "muon" (Muon is for 2D weight matrices; the other parameters still use AdamW; see Chapter 12)
    lr: float = 3e-4  # peak learning rate
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    grad_clip: float = 1.0  # clipping of the global gradient norm; <=0 means no clipping
    decay_embeddings: bool = False  # apply weight decay to the embedding or not (norm weights never get weight decay)


@dataclass
class ScheduleConfig:
    kind: str = "cosine"  # "cosine" | "wsd" | "constant"
    warmup_steps: int = 100
    min_lr_ratio: float = 0.1  # minimum learning rate = lr * min_lr_ratio
    decay_frac: float = 0.2  # WSD: the fraction of the steps at the end that decays
    decay_shape: str = "linear"  # shape of the WSD decay phase: "linear" | "cosine" | "sqrt"


@dataclass
class CheckpointConfig:
    every: int = 1000  # save every N steps; <=0 means save only at the end
    keep_last: int = 3  # keep only the last N (<=0 keeps all)
    resume: bool = True  # resume automatically if the directory contains a checkpoint
    dir: str = ""  # default <out_dir>/ckpt


@dataclass
class LoggingConfig:
    every: int = 10  # print/log every N steps
    jsonl: str = ""  # default <out_dir>/log.jsonl


@dataclass
class TrainConfig:
    seed: int = 1337
    max_steps: int = 1000
    micro_batch_size: int = 8  # sequences per forward pass on each GPU
    grad_accum_steps: int = 1  # gradient accumulation: effective batch = micro * accum * world_size
    device: str = "auto"  # "auto" | "cpu" | "cuda"
    dtype: str = "auto"  # "auto" (bf16 on CUDA, fp32 on CPU) | "bf16" | "fp32"
    # torch.compile: verified on RTX 3090 with one GPU and with 2-GPU DDP (2026-10, see runs/2026-10-01-gpu0-check/).
    # Not verified on GPUs together with FSDP yet.
    compile: bool = False
    # Activation checkpointing: each layer stores only its input and computes again in the backward pass
    # (Chapter 14). Verified on one RTX 3090 (2026-10, see runs/2026-10-01-gpu0-check/).
    activation_checkpointing: bool = False
    cpu_threads: int = (
        0  # number of PyTorch threads on the CPU; 0 means the default. If other processes use all CPUs, 1 is the fastest
    )
    parallel: str = "ddp"  # parallel method with more than one process: "ddp" | "fsdp"; ignored with one process
    out_dir: str = "out/run"
    # Stop at this step (0 = train to max_steps). The learning-rate schedule still uses max_steps, so a stop
    # and a resume give the same result as one full run. The successive halving of the ladder experiments
    # uses it: train to 2/8, then continue only the configs that remain (runs/ladder-3090).
    # Put the stop step on a multiple of checkpoint.every, so that a resume can continue from this step
    stop_step: int = 0
    init_from: str = ""  # mid-training: load the model weights from this checkpoint directory (or the latest one in this parent directory)
    eval_every: int = 100  # compute the validation loss every N steps; <=0 means no evaluation
    eval_batches: int = 10
    gpu_peak_tflops: float = 0.0  # for MFU; 0 means look up the device name in a table (no MFU if not found)
    data: DataConfig = field(default_factory=DataConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    def validate(self) -> None:
        def need(cond: bool, msg: str) -> None:
            if not cond:
                raise ConfigError(msg)

        need(self.max_steps > 0, "[train] max_steps must be > 0")
        need(self.stop_step >= 0, "[train] stop_step must be >= 0 (0 = train to max_steps)")
        need(self.micro_batch_size > 0, "[train] micro_batch_size must be > 0")
        need(self.grad_accum_steps > 0, "[train] grad_accum_steps must be > 0")
        need(
            self.device in ("auto", "cpu", "cuda"),
            f"[train] device must be auto/cpu/cuda, got {self.device!r}",
        )
        need(
            self.dtype in ("auto", "bf16", "fp32"),
            f"[train] dtype must be auto/bf16/fp32, got {self.dtype!r}",
        )
        need(
            self.parallel in ("ddp", "fsdp"),
            f"[train] parallel must be ddp/fsdp, got {self.parallel!r}",
        )
        need(self.data.seq_len > 0, "[data] seq_len must be > 0")
        need(
            self.data.format in ("packed", "sft", "none"),
            f"[data] format must be packed/sft/none, got {self.data.format!r}",
        )
        need(
            len(self.data.sources) > 0 or self.data.format != "packed",
            "[data] needs at least one [[data.sources]]",
        )
        for s in self.data.sources:
            need(bool(s.name) and bool(s.path), "[[data.sources]] each entry needs a name and a path")
            need(s.weight > 0, f"[[data.sources]] the weight of {s.name} must be > 0")
        names = [s.name for s in self.data.sources]
        need(len(set(names)) == len(names), f"[[data.sources]] duplicate names: {names}")
        need(
            self.schedule.kind in ("cosine", "wsd", "constant"),
            f"[schedule] kind must be cosine/wsd/constant, got {self.schedule.kind!r}",
        )
        need(
            self.schedule.decay_shape in ("linear", "cosine", "sqrt"),
            f"[schedule] decay_shape must be linear/cosine/sqrt, got {self.schedule.decay_shape!r}",
        )
        need(0.0 <= self.schedule.min_lr_ratio <= 1.0, "[schedule] min_lr_ratio must be in [0, 1]")
        need(0.0 <= self.schedule.decay_frac <= 1.0, "[schedule] decay_frac must be in [0, 1]")
        need(self.schedule.warmup_steps >= 0, "[schedule] warmup_steps must be >= 0")
        need(self.optim.lr > 0, "[optim] lr must be > 0")
        if self.data.prepare is not None:
            p = self.data.prepare
            need(bool(p.raw_files) and bool(p.out_dir), "[data.prepare] needs raw_files and out_dir")
            need(0.0 < p.val_fraction < 1.0, "[data.prepare] val_fraction must be in (0, 1)")

    @property
    def checkpoint_dir(self) -> str:
        return self.checkpoint.dir or str(Path(self.out_dir) / "ckpt")

    @property
    def log_path(self) -> str:
        return self.logging.jsonl or str(Path(self.out_dir) / "log.jsonl")


@dataclass
class Config:
    """A full config: model + training."""

    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    source_path: str = ""  # the file that the config came from (only for logs)

    def validate(self) -> None:
        self.model.validate()
        self.train.validate()
        if self.train.data.seq_len > self.model.max_seq_len:
            raise ConfigError(
                f"[data] seq_len={self.train.data.seq_len} is larger than [model] max_seq_len={self.model.max_seq_len}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {"model": dataclasses.asdict(self.model), "train": dataclasses.asdict(self.train)}


# ---------------------------------------------------------------------------
# Read TOML: dict -> dataclass, with validation
# ---------------------------------------------------------------------------


def _type_name(tp: Any) -> str:
    return getattr(tp, "__name__", str(tp))


def _coerce(value: Any, tp: Any, where: str) -> Any:
    """Check/convert a value from TOML to the type of the dataclass field."""
    origin = typing.get_origin(tp)
    args = typing.get_args(tp)
    # Optional[X] / X | None
    if origin in (typing.Union, types.UnionType):
        if value is None and type(None) in args:
            return None
        non_none = [a for a in args if a is not type(None)]
        errors = []
        for a in non_none:
            try:
                return _coerce(value, a, where)
            except ConfigError as e:
                errors.append(str(e))
        raise ConfigError(errors[0] if errors else f"{where}: wrong type")
    if dataclasses.is_dataclass(tp):
        if not isinstance(value, dict):
            raise ConfigError(f"{where}: must be a table, got {type(value).__name__}")
        return _from_dict(tp, value, where)
    if origin is list:
        if not isinstance(value, list):
            raise ConfigError(f"{where}: must be a list, got {type(value).__name__}")
        (inner,) = args or (Any,)
        return [_coerce(v, inner, f"{where}[{i}]") for i, v in enumerate(value)]
    if origin is dict:
        if not isinstance(value, dict):
            raise ConfigError(f"{where}: must be a table (dict), got {type(value).__name__}")
        return dict(value)
    if tp is Any:
        return value
    if tp is bool:
        if not isinstance(value, bool):
            raise ConfigError(f"{where}: must be true/false, got {value!r}")
        return value
    if tp is int:
        if isinstance(value, bool) or not isinstance(value, int):
            # allow notation such as 1e6 for an integer
            if isinstance(value, float) and value.is_integer():
                return int(value)
            raise ConfigError(f"{where}: must be an integer, got {value!r}")
        return value
    if tp is float:
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ConfigError(f"{where}: must be a number, got {value!r}")
        return float(value)
    if tp is str:
        if not isinstance(value, str):
            raise ConfigError(f"{where}: must be a string, got {value!r}")
        return value
    raise ConfigError(f"{where}: unsupported field type {_type_name(tp)}")


def _from_dict(cls: type, data: dict[str, Any], where: str) -> Any:
    hints = typing.get_type_hints(cls)
    names = [f.name for f in dataclasses.fields(cls)]
    kwargs = {}
    for key, value in data.items():
        if key not in hints:
            close = difflib.get_close_matches(key, names, n=1)
            hint = f", did you mean {close[0]!r}?" if close else f". Available fields: {', '.join(names)}"
            raise ConfigError(f"{where}: unknown field {key!r}{hint}")
        kwargs[key] = _coerce(value, hints[key], f"{where}.{key}")
    return cls(**kwargs)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _read_toml_with_base(path: Path, seen: tuple[Path, ...] = ()) -> dict[str, Any]:
    path = path.resolve()
    if path in seen:
        raise ConfigError(f"Config inheritance has a cycle: {' -> '.join(str(p) for p in (*seen, path))}")
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")
    try:
        with open(path, "rb") as f:
            data = tomllib.load(f)
    except tomllib.TOMLDecodeError as e:
        raise ConfigError(f"{path}: TOML syntax error: {e}") from e
    base = data.pop("base", None)
    if base is not None:
        parent = _read_toml_with_base((path.parent / base), (*seen, path))
        data = _deep_merge(parent, data)
    return data


def _apply_override(data: dict[str, Any], item: str) -> None:
    """Command-line override: `train.max_steps=10`, `model.rope_theta=1e6`, `data.seq_len=256`."""
    if "=" not in item:
        raise ConfigError(f"Override {item!r} must have the format section.key=value")
    key, raw = item.split("=", 1)
    try:
        value = tomllib.loads(f"v = {raw}")["v"]
    except tomllib.TOMLDecodeError:
        value = raw  # use it as a plain string
    parts = key.strip().split(".")
    d = data
    for p in parts[:-1]:
        d = d.setdefault(p, {})
    d[parts[-1]] = value


def config_from_dict(data: dict[str, Any], source_path: str = "") -> Config:
    """Build and validate a Config from a dict (after the merge)."""
    data = dict(data)
    known = {"model", "train", "data", "optim", "schedule", "checkpoint", "logging"}
    unknown = set(data) - known
    if unknown:
        k = sorted(unknown)[0]
        close = difflib.get_close_matches(k, sorted(known), n=1)
        hint = f", did you mean [{close[0]}]?" if close else ""
        raise ConfigError(f"Unknown config section [{k}]{hint}")
    model = _from_dict(ModelConfig, data.get("model", {}), "[model]")
    train_dict = dict(data.get("train", {}))
    for sec in ("data", "optim", "schedule", "checkpoint", "logging"):
        if sec in data:
            if sec in train_dict:
                raise ConfigError(f"[{sec}] and [train.{sec}] cannot both be present")
            train_dict[sec] = data[sec]
    train = _from_dict(TrainConfig, train_dict, "[train]")
    cfg = Config(model=model, train=train, source_path=source_path)
    cfg.validate()
    return cfg


def load_config(path: str | Path, overrides: list[str] | None = None) -> Config:
    """Read a TOML config (with base inheritance and command-line overrides) and validate it."""
    data = _read_toml_with_base(Path(path))
    for item in overrides or []:
        _apply_override(data, item)
    return config_from_dict(data, source_path=str(path))


def read_toml(path: str | Path) -> dict[str, Any]:
    """Read TOML, resolve the base inheritance, and return the raw dict (no validation)."""
    return _read_toml_with_base(Path(path))


def load_model_config(path: str | Path) -> ModelConfig:
    """Read only the [model] section (for tool scripts, for example to count parameters or estimate cost)."""
    data = _read_toml_with_base(Path(path))
    cfg = _from_dict(ModelConfig, data.get("model", {}), "[model]")
    cfg.validate()
    return cfg
