"""配置：模型与训练的全部超参数（对应第 9、12 章）。

所有超参都写在 `configs/*.toml` 里，代码里不写死。这个模块做三件事：

1. 用 dataclass 定义配置的"形状"：`ModelConfig`（模型结构）、`TrainConfig`（训练流程，
   内含 data / optim / schedule / checkpoint / logging 几个小节）；
2. `load_config(path)` 用标准库 `tomllib` 读 TOML，支持 `base = "xxx.toml"` 继承另一份配置
   （中期训练只写和预训练不同的地方）；
3. 校验：未知字段、类型错误、互相矛盾的取值（比如 n_heads 不能被 n_kv_heads 整除），
   都在读配置时就报错，并给出"你是不是想写 xxx"的提示，而不是训练到一半才崩。

TOML 的布局（各节都是顶层表）：

    base = "pretrain.toml"      # 可选：先读这份，再用本文件覆盖
    [model]      -> ModelConfig
    [train]      -> TrainConfig 的标量字段
    [data]       -> DataConfig（含 [[data.sources]] 与可选的 [data.prepare]）
    [optim]      -> OptimConfig
    [schedule]   -> ScheduleConfig
    [checkpoint] -> CheckpointConfig
    [logging]    -> LoggingConfig

路径字段一律相对于仓库根目录（也就是运行命令时的当前目录）。
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
    """配置文件有问题时抛出，信息里会指出是哪个字段。"""


# ---------------------------------------------------------------------------
# 模型结构
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
    """Transformer 结构超参。字段含义与 Hugging Face `Qwen3Config` 一一对应（见 `zero/hf.py`）。"""

    vocab_size: int = 32000
    dim: int = 512  # 隐藏维度 d_model（HF: hidden_size）
    n_layers: int = 8
    n_heads: int = 8  # 查询头数
    n_kv_heads: int = 4  # K/V 头数（GQA：若干个查询头共享一组 K/V）
    head_dim: int | None = None  # 默认 dim // n_heads；Qwen3 允许 n_heads*head_dim != dim
    ffn_dim: int = 1536  # SwiGLU 中间维度（HF: intermediate_size）
    rope_theta: float = 10000.0
    # YaRN 等 RoPE 缩放：{"type": "yarn", "factor": 4.0, "original_max_position_embeddings": 4096,
    #                    "beta_fast": 32, "beta_slow": 1}；None 表示不缩放
    rope_scaling: dict[str, Any] | None = None
    max_seq_len: int = 2048  # RoPE 预计算的最大位置数，也是 KV cache 默认长度
    norm_eps: float = 1e-6
    qk_norm: bool = True  # Qwen3 的 QK-Norm：对每个头的 q、k 在 head_dim 上做 RMSNorm
    tie_embeddings: bool = True  # 输入 embedding 与输出 lm_head 共享权重
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
            need(getattr(self, name) > 0, f"{name} 必须是正整数，当前 {getattr(self, name)}")
        need(
            self.n_heads % self.n_kv_heads == 0,
            f"n_heads={self.n_heads} 必须能被 n_kv_heads={self.n_kv_heads} 整除（GQA 分组）",
        )
        assert self.head_dim is not None
        need(
            self.head_dim > 0 and self.head_dim % 2 == 0,
            f"head_dim={self.head_dim} 必须是正偶数（RoPE 两两旋转）",
        )
        need(self.rope_theta > 0, "rope_theta 必须 > 0")
        need(self.norm_eps > 0, "norm_eps 必须 > 0")
        need(self.init_std > 0, "init_std 必须 > 0")
        if self.rope_scaling is not None:
            rs = self.rope_scaling
            unknown = set(rs) - _ROPE_SCALING_KEYS
            need(
                not unknown,
                f"rope_scaling 有未知字段 {sorted(unknown)}，可用 {sorted(_ROPE_SCALING_KEYS)}",
            )
            need(
                rs.get("type") == "yarn",
                f"rope_scaling.type 目前只支持 'yarn'，当前 {rs.get('type')!r}",
            )
            need(float(rs.get("factor", 0)) >= 1.0, "rope_scaling.factor 必须 >= 1")
            need(
                int(rs.get("original_max_position_embeddings", 0)) > 0,
                "rope_scaling.original_max_position_embeddings 必须给出（预训练时的上下文长度）",
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
# 训练流程
# ---------------------------------------------------------------------------


@dataclass
class DataSourceConfig:
    """一个数据来源：一组分片 + 采样权重（多来源时按权重混合，见 zero/data/mixture.py）。"""

    name: str = ""
    path: str = ""  # 分片的 glob，例如 "out/tiny/data/shakespeare_train_*.bin"
    weight: float = 1.0


@dataclass
class DataPrepareConfig:
    """可选：分片不存在时，从原始文本现场训练分词器并切分片（只用于 tiny 冒烟）。"""

    raw_files: list[str] = field(default_factory=list)  # 原始文本 glob；文件名去掉扩展名就是来源名
    out_dir: str = ""  # 分片输出目录
    vocab_size: int = 2048  # 要训练的分词器词表大小（data.tokenizer 不存在时才训练）
    val_fraction: float = 0.05  # 每个来源留多少比例的文档做验证集
    doc_chars: int = 4000  # 把原始文本按空行切段后，再拼成约这么长的"文档"


@dataclass
class DataConfig:
    tokenizer: str = ""  # tokenizer.json 路径
    seq_len: int = 1024  # 训练序列长度（每个样本取 seq_len+1 个 token）
    sources: list[DataSourceConfig] = field(default_factory=list)
    val: str = ""  # 验证集分片 glob；空表示不做评估
    shuffle: bool = True
    prepare: DataPrepareConfig | None = None
    # 数据格式："packed" = 预训练分片（uint32 token 流，见 zero/data/loader.py 的 PackedDataLoader）；
    # "sft" = 对话打包窗口 + loss mask（zero/post/sft.py 生成，MaskedWindowLoader 读取）；
    # "none" = 训练循环自己管数据（DPO / GRPO / 蒸馏），[[data.sources]] 可以为空
    format: str = "packed"


@dataclass
class OptimConfig:
    lr: float = 3e-4  # 峰值学习率
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    grad_clip: float = 1.0  # 全局梯度范数裁剪；<=0 表示不裁剪
    decay_embeddings: bool = False  # embedding 是否做权重衰减（norm 的权重永远不衰减）


@dataclass
class ScheduleConfig:
    kind: str = "cosine"  # "cosine" | "wsd" | "constant"
    warmup_steps: int = 100
    min_lr_ratio: float = 0.1  # 最低学习率 = lr * min_lr_ratio
    decay_frac: float = 0.2  # WSD：最后多少比例的步数用来衰减
    decay_shape: str = "linear"  # WSD 衰减段的形状："linear" | "cosine" | "sqrt"


@dataclass
class CheckpointConfig:
    every: int = 1000  # 每多少步存一次；<=0 表示只在结束时存
    keep_last: int = 3  # 只保留最近几个（<=0 全留）
    resume: bool = True  # 目录里有 checkpoint 时自动续训
    dir: str = ""  # 默认 <out_dir>/ckpt


@dataclass
class LoggingConfig:
    every: int = 10  # 每多少步打印/记录一次
    jsonl: str = ""  # 默认 <out_dir>/log.jsonl


@dataclass
class TrainConfig:
    seed: int = 1337
    max_steps: int = 1000
    micro_batch_size: int = 8  # 每张卡每次前向的序列数
    grad_accum_steps: int = 1  # 梯度累积：有效 batch = micro * accum * world_size
    device: str = "auto"  # "auto" | "cpu" | "cuda"
    dtype: str = "auto"  # "auto"（CUDA 用 bf16，CPU 用 fp32）| "bf16" | "fp32"
    compile: bool = False  # torch.compile（尚未在 GPU 上验证）
    cpu_threads: int = (
        0  # CPU 上 PyTorch 的线程数；0 表示用默认值。机器被别的进程占满时设 1 反而最快
    )
    parallel: str = "ddp"  # 多进程时的并行方式："ddp" | "fsdp"；单进程忽略
    out_dir: str = "out/run"
    init_from: str = ""  # 中期训练：从这个 checkpoint 目录（或其父目录里最新的）加载模型权重
    eval_every: int = 100  # 每多少步算一次验证 loss；<=0 不评估
    eval_batches: int = 10
    gpu_peak_tflops: float = 0.0  # 用于 MFU；0 表示按设备名自动查表（查不到就不报 MFU）
    data: DataConfig = field(default_factory=DataConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    def validate(self) -> None:
        def need(cond: bool, msg: str) -> None:
            if not cond:
                raise ConfigError(msg)

        need(self.max_steps > 0, "[train] max_steps 必须 > 0")
        need(self.micro_batch_size > 0, "[train] micro_batch_size 必须 > 0")
        need(self.grad_accum_steps > 0, "[train] grad_accum_steps 必须 > 0")
        need(
            self.device in ("auto", "cpu", "cuda"),
            f"[train] device 只能是 auto/cpu/cuda，当前 {self.device!r}",
        )
        need(
            self.dtype in ("auto", "bf16", "fp32"),
            f"[train] dtype 只能是 auto/bf16/fp32，当前 {self.dtype!r}",
        )
        need(
            self.parallel in ("ddp", "fsdp"),
            f"[train] parallel 只能是 ddp/fsdp，当前 {self.parallel!r}",
        )
        need(self.data.seq_len > 0, "[data] seq_len 必须 > 0")
        need(
            self.data.format in ("packed", "sft", "none"),
            f"[data] format 只能是 packed/sft/none，当前 {self.data.format!r}",
        )
        need(
            len(self.data.sources) > 0 or self.data.format != "packed",
            "[data] 至少要有一个 [[data.sources]]",
        )
        for s in self.data.sources:
            need(bool(s.name) and bool(s.path), "[[data.sources]] 每项都要有 name 和 path")
            need(s.weight > 0, f"[[data.sources]] {s.name} 的 weight 必须 > 0")
        names = [s.name for s in self.data.sources]
        need(len(set(names)) == len(names), f"[[data.sources]] 名字重复：{names}")
        need(
            self.schedule.kind in ("cosine", "wsd", "constant"),
            f"[schedule] kind 只能是 cosine/wsd/constant，当前 {self.schedule.kind!r}",
        )
        need(
            self.schedule.decay_shape in ("linear", "cosine", "sqrt"),
            f"[schedule] decay_shape 只能是 linear/cosine/sqrt，当前 {self.schedule.decay_shape!r}",
        )
        need(0.0 <= self.schedule.min_lr_ratio <= 1.0, "[schedule] min_lr_ratio 必须在 [0, 1]")
        need(0.0 <= self.schedule.decay_frac <= 1.0, "[schedule] decay_frac 必须在 [0, 1]")
        need(self.schedule.warmup_steps >= 0, "[schedule] warmup_steps 必须 >= 0")
        need(self.optim.lr > 0, "[optim] lr 必须 > 0")
        if self.data.prepare is not None:
            p = self.data.prepare
            need(bool(p.raw_files) and bool(p.out_dir), "[data.prepare] 需要 raw_files 和 out_dir")
            need(0.0 < p.val_fraction < 1.0, "[data.prepare] val_fraction 必须在 (0, 1)")

    @property
    def checkpoint_dir(self) -> str:
        return self.checkpoint.dir or str(Path(self.out_dir) / "ckpt")

    @property
    def log_path(self) -> str:
        return self.logging.jsonl or str(Path(self.out_dir) / "log.jsonl")


@dataclass
class Config:
    """一份完整配置：模型 + 训练。"""

    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    source_path: str = ""  # 从哪个文件读来的（只用于日志）

    def validate(self) -> None:
        self.model.validate()
        self.train.validate()
        if self.train.data.seq_len > self.model.max_seq_len:
            raise ConfigError(
                f"[data] seq_len={self.train.data.seq_len} 超过了 [model] max_seq_len={self.model.max_seq_len}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {"model": dataclasses.asdict(self.model), "train": dataclasses.asdict(self.train)}


# ---------------------------------------------------------------------------
# 读 TOML：dict -> dataclass，带校验
# ---------------------------------------------------------------------------


def _type_name(tp: Any) -> str:
    return getattr(tp, "__name__", str(tp))


def _coerce(value: Any, tp: Any, where: str) -> Any:
    """把 TOML 里读到的值按 dataclass 字段类型检查/转换。"""
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
        raise ConfigError(errors[0] if errors else f"{where}: 类型不对")
    if dataclasses.is_dataclass(tp):
        if not isinstance(value, dict):
            raise ConfigError(f"{where}: 应该是一个表（table），实际是 {type(value).__name__}")
        return _from_dict(tp, value, where)
    if origin is list:
        if not isinstance(value, list):
            raise ConfigError(f"{where}: 应该是列表，实际是 {type(value).__name__}")
        (inner,) = args or (Any,)
        return [_coerce(v, inner, f"{where}[{i}]") for i, v in enumerate(value)]
    if origin is dict:
        if not isinstance(value, dict):
            raise ConfigError(f"{where}: 应该是表（dict），实际是 {type(value).__name__}")
        return dict(value)
    if tp is Any:
        return value
    if tp is bool:
        if not isinstance(value, bool):
            raise ConfigError(f"{where}: 应该是 true/false，实际是 {value!r}")
        return value
    if tp is int:
        if isinstance(value, bool) or not isinstance(value, int):
            # 允许 1e6 这种写法表示整数
            if isinstance(value, float) and value.is_integer():
                return int(value)
            raise ConfigError(f"{where}: 应该是整数，实际是 {value!r}")
        return value
    if tp is float:
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ConfigError(f"{where}: 应该是数字，实际是 {value!r}")
        return float(value)
    if tp is str:
        if not isinstance(value, str):
            raise ConfigError(f"{where}: 应该是字符串，实际是 {value!r}")
        return value
    raise ConfigError(f"{where}: 不支持的字段类型 {_type_name(tp)}")


def _from_dict(cls: type, data: dict[str, Any], where: str) -> Any:
    hints = typing.get_type_hints(cls)
    names = [f.name for f in dataclasses.fields(cls)]
    kwargs = {}
    for key, value in data.items():
        if key not in hints:
            close = difflib.get_close_matches(key, names, n=1)
            hint = f"，你是不是想写 {close[0]!r}？" if close else f"。可用字段：{', '.join(names)}"
            raise ConfigError(f"{where}: 未知字段 {key!r}{hint}")
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
        raise ConfigError(f"配置继承出现循环：{' -> '.join(str(p) for p in (*seen, path))}")
    if not path.exists():
        raise ConfigError(f"找不到配置文件：{path}")
    try:
        with open(path, "rb") as f:
            data = tomllib.load(f)
    except tomllib.TOMLDecodeError as e:
        raise ConfigError(f"{path}: TOML 语法错误：{e}") from e
    base = data.pop("base", None)
    if base is not None:
        parent = _read_toml_with_base((path.parent / base), (*seen, path))
        data = _deep_merge(parent, data)
    return data


def _apply_override(data: dict[str, Any], item: str) -> None:
    """命令行覆盖：`train.max_steps=10`、`model.rope_theta=1e6`、`data.seq_len=256`。"""
    if "=" not in item:
        raise ConfigError(f"覆盖项 {item!r} 格式应为 section.key=value")
    key, raw = item.split("=", 1)
    try:
        value = tomllib.loads(f"v = {raw}")["v"]
    except tomllib.TOMLDecodeError:
        value = raw  # 当作裸字符串
    parts = key.strip().split(".")
    d = data
    for p in parts[:-1]:
        d = d.setdefault(p, {})
    d[parts[-1]] = value


def config_from_dict(data: dict[str, Any], source_path: str = "") -> Config:
    """从一个（已经合并好的）dict 构建并校验 Config。"""
    data = dict(data)
    known = {"model", "train", "data", "optim", "schedule", "checkpoint", "logging"}
    unknown = set(data) - known
    if unknown:
        k = sorted(unknown)[0]
        close = difflib.get_close_matches(k, sorted(known), n=1)
        hint = f"，你是不是想写 [{close[0]}]？" if close else ""
        raise ConfigError(f"未知的配置节 [{k}]{hint}")
    model = _from_dict(ModelConfig, data.get("model", {}), "[model]")
    train_dict = dict(data.get("train", {}))
    for sec in ("data", "optim", "schedule", "checkpoint", "logging"):
        if sec in data:
            if sec in train_dict:
                raise ConfigError(f"[{sec}] 和 [train.{sec}] 不能同时出现")
            train_dict[sec] = data[sec]
    train = _from_dict(TrainConfig, train_dict, "[train]")
    cfg = Config(model=model, train=train, source_path=source_path)
    cfg.validate()
    return cfg


def load_config(path: str | Path, overrides: list[str] | None = None) -> Config:
    """读取 TOML 配置（支持 base 继承与命令行覆盖）并校验。"""
    data = _read_toml_with_base(Path(path))
    for item in overrides or []:
        _apply_override(data, item)
    return config_from_dict(data, source_path=str(path))


def read_toml(path: str | Path) -> dict[str, Any]:
    """读 TOML 并展开 base 继承，返回原始 dict（不校验）。"""
    return _read_toml_with_base(Path(path))


def load_model_config(path: str | Path) -> ModelConfig:
    """只读 [model] 一节（工具脚本用，比如数参数量、估算成本）。"""
    data = _read_toml_with_base(Path(path))
    cfg = _from_dict(ModelConfig, data.get("model", {}), "[model]")
    cfg.validate()
    return cfg
