"""后训练各阶段共用的小工具（对应第 16–19 章）。

- `load_post_config`：读 TOML。通用部分（[model] [train] [data] [optim] [schedule] [checkpoint]
  [logging]）交给 `zero.config`，各阶段自己的小节（[sft] [distill] [teacher] [dpo] [grpo] [eval]）
  在这里按 dataclass 解析，同样做未知字段检查；
- `load_policy`：从 zero 的 checkpoint 目录或导出的 Hugging Face 目录加载 (模型, 分词器)；
- `chat_complete`：套对话模板 → 带 KV cache 生成 → 解码；
- `token_logprobs` / `pad_batch`：DPO、GRPO、蒸馏都要算"每个 token 的 log 概率"；
- JSONL 读写。
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Sequence
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
    """读配置：返回 (通用 Config, {小节名: dataclass 实例})。src 可以是路径或已经展开好的 dict。"""
    if isinstance(src, dict):
        data = json.loads(json.dumps(src))  # 深拷贝
        source = "<dict>"
    else:
        data = _read_toml_with_base(Path(src))
        source = str(src)
    for item in overrides or []:
        _apply_override(data, item)
    extra = {k: data.pop(k) for k in list(data) if k not in CORE_SECTIONS}
    unknown = set(extra) - set(sections)
    if unknown:
        raise ConfigError(f"未知的配置节 {sorted(unknown)}，本阶段可用：{sorted(sections)}")
    parsed = {
        name: _from_dict(cls, extra.get(name, {}), f"[{name}]") for name, cls in sections.items()
    }
    return config_from_dict(data, source_path=source), parsed


# ---------------------------------------------------------------------------
# 模型加载
# ---------------------------------------------------------------------------


def load_policy(
    path: str | os.PathLike,
    tokenizer_path: str | os.PathLike | None = None,
    device: str | torch.device = "cpu",
    model_overrides: dict[str, Any] | None = None,
) -> tuple[Transformer, Tokenizer]:
    """加载 (模型, 分词器)。

    path 可以是：
    - 导出的 HF 目录（有 config.json + *.safetensors + tokenizer.json）；
    - zero 的 checkpoint 目录（`<out>/ckpt` 或某个 `step_xxx`），分词器路径从 meta.json 里的配置读，
      也可用 tokenizer_path 指定。
    model_overrides：比如 {"max_seq_len": 1024}（只允许改 RoPE 相关字段）。
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
        raise FileNotFoundError(f"{path} 既不是 HF 目录，也找不到 zero checkpoint")
    meta = json.loads((ckpt / "meta.json").read_text())
    mcfg = dict(meta["config"]["model"])
    mcfg.update(model_overrides or {})
    model = Transformer(ModelConfig(**mcfg))
    sd = torch.load(ckpt / "model.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(sd)
    tp = Path(tokenizer_path or meta["config"]["train"]["data"]["tokenizer"])
    if not tp.is_absolute() and not tp.exists():
        # 配置里记的是相对训练时工作目录（通常是仓库根目录）的路径；从别的目录调用时按仓库根目录解析
        repo_root = Path(__file__).resolve().parents[2]
        if (repo_root / tp).exists():
            tp = repo_root / tp
    return model.to(device), Tokenizer.load(tp)


# ---------------------------------------------------------------------------
# 生成
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
    """对一段对话生成 n 个助手回复（文本，不含 <|im_end|>）。n 个样本一个 batch 并行（同一提示词，
    长度相同，不需要 padding），用 KV cache。"""
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
    """包装成 tool_env.run_episode 需要的 policy(messages, tools) -> text。"""

    def policy(messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> str:
        return chat_complete(model, tok, messages, tools, max_new_tokens, temperature)[0]

    return policy


# ---------------------------------------------------------------------------
# log 概率
# ---------------------------------------------------------------------------


def pad_batch(
    seqs: Sequence[Sequence[int]],
    masks: Sequence[Sequence[bool]],
    pad_id: int,
    device: str | torch.device = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """右侧补齐。返回 ids (B, T) 与 mask (B, T)（补齐部分 mask=False）。因果注意力下右侧补齐不影响前面的位置。"""
    L = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), L), pad_id, dtype=torch.long)
    m = torch.zeros((len(seqs), L), dtype=torch.bool)
    for i, (s, mk) in enumerate(zip(seqs, masks)):
        ids[i, : len(s)] = torch.tensor(list(s), dtype=torch.long)
        m[i, : len(mk)] = torch.tensor(list(mk), dtype=torch.bool)
    return ids.to(device), m.to(device)


def token_logprobs(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """logits (B, T, V)、targets (B, T) → 每个目标 token 的 log 概率 (B, T)，float32。"""
    return torch.gather(F.log_softmax(logits.float(), dim=-1), -1, targets.unsqueeze(-1)).squeeze(
        -1
    )


def sequence_token_logprobs(
    model: torch.nn.Module, ids: torch.Tensor, mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """整段前向一次，返回 (logp, target_mask)，形状都是 (B, T-1)：
    logp[:, t] = log π(ids[:, t+1] | ids[:, :t+1])，target_mask = mask[:, 1:]。"""
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
# DPO / GRPO / 在线蒸馏共用的"训练循环骨架"：优化器、学习率、梯度裁剪、日志、checkpoint、续训
# ---------------------------------------------------------------------------


class LoopState:
    """不走 `Trainer`（数据不是固定的 token 流）的训练阶段共用的部分。

    用法：
        loop = LoopState(cfg, model, log)       # 建优化器/调度器；有 checkpoint 就续训
        while loop.step < max_steps:
            lr = loop.begin_step()
            ... loss.backward() ...
            gnorm = loop.end_step()             # 裁剪 + 更新 + step += 1
            loop.record({...})                  # 打日志、写 JSONL、按需存 checkpoint
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
            self.log(f"从 {meta['path']} 续训（step {self.step}）")

    def log(self, msg: str) -> None:
        if self.info.is_main and self._log is not None:
            self._log(msg)

    def begin_step(self) -> float:
        return self.scheduler.apply(self.step)

    def optimizer_step(self) -> float:
        """梯度裁剪 + 一次参数更新（不增加步数；GRPO 的 ppo_epochs > 1 时一步里会更新多次）。"""
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
    """按 cfg.model 建模型，从 cfg.train.init_from（zero checkpoint）加载权重。"""
    from zero.train.checkpoint import load_model_weights

    model = Transformer(cfg.model).to(device)
    if cfg.train.init_from:
        load_model_weights(cfg.train.init_from, model)
    return model
