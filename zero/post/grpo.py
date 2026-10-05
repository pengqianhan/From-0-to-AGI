"""GRPO：组内相对策略优化 + 可验证奖励（对应第 19 章）。

    uv run python -m zero.post.grpo --config configs/tiny/grpo.toml

每一步：

1. 取 P 个任务（`zero/post/envs/tool_env.py`），每个任务用当前策略采样 G 个回复
   （同一提示词复制 G 份组成一个 batch，`zero.generate` 带 KV cache 生成，遇到 `<|im_end|>` 停）；
2. 用可验证奖励 `score_tool_calls` 给每个回复打分 r_1..r_G；
3. **组内归一化的优势**：A_i = (r_i − mean(r)) / (std(r) + ε)（std 取无偏估计，与 TRL / verl 一致；
   `scale_rewards = false` 时只减均值，即 Dr. GRPO 的做法）。一组里全对或全错时 A 全为 0，这组不提供梯度；
4. **裁剪的重要性比**（PPO-clip）：对回复里的每个 token t，
       ρ_t = π_θ(y_t | ·) / π_old(y_t | ·)
       ℓ_t = −min(ρ_t · A, clip(ρ_t, 1−ε_low, 1+ε_high) · A)
   π_old 是采样时的策略（在更新前用同一份权重算一遍 log 概率）；`ppo_epochs = 1` 时 ρ ≡ 1，
   但梯度 ∇ρ = ∇log π 不为零；`ppo_epochs > 1` 时同一批数据更新多次，裁剪才真正起作用；
5. **可选 KL**：kl_coef > 0 时加 β · KL(π_θ ‖ π_ref)，逐 token 用 k3 估计
   exp(ref − logp) − (ref − logp) − 1（DeepSeekMath 的写法，非负、无偏）；
6. **token 级聚合**：`loss_agg = "token_mean"`（默认）——一个 batch 里所有回复 token 的 ℓ 直接平均，
   长回复里的每个 token 和短回复里的权重一样（DAPO 的做法，避免"长回复的 token 被稀释"）；
   `"seq_mean_token_mean"` 是原始 GRPO 论文的写法：先在每条回复内平均，再在回复之间平均。

日志：平均奖励、格式正确率（没有格式错误的回复比例；不调用工具的纯文本也算格式正确）、
调用率（含至少一个格式正确的工具调用的比例）、回复长度、KL、裁剪比例、"零方差组"比例。

**与 verl 对拍（第二步，尚未进行）**：吞吐不够时可换用 verl（见 references.md：XiaomiMiMo/verl）。
对拍方法：同一个导出的 HF 模型、同一批 tool_env 任务（固定种子导出 JSONL）、同样的 G / ε / β /
学习率 / 聚合方式（verl 里 `algorithm.adv_estimator=grpo`、`actor_rollout_ref.actor.loss_agg_mode=token-mean`、
`use_kl_loss` 与 `kl_loss_type=low_var_kl` 对应这里的 k3），奖励函数用 verl 的自定义 reward 接口
包一层 `score_tool_calls`；先比第一步的优势与损失（同一批样本应逐位一致），再比前 50 步的平均奖励曲线。
（verl 的配置项名称按其 0.9 版文档整理，待核实。）
"""

from __future__ import annotations

import argparse
import copy
import os
import random
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from zero.post.chat import render
from zero.post.common import (
    LoopState,
    build_model_from_init,
    load_post_config,
    pad_batch,
    sequence_token_logprobs,
    set_threads,
)
from zero.tokenizer import Tokenizer


@dataclass
class GRPOConfig:
    group_size: int = 8  # G：每个提示词采样几个回复
    prompts_per_step: int = 4  # P：每步几个提示词
    max_new_tokens: int = 64
    temperature: float = 1.0
    top_p: float = 1.0
    clip_eps: float = 0.2  # ε_low
    clip_eps_high: float = 0.0  # ε_high；0 表示与 clip_eps 相同（>ε_low 即 DAPO 的 clip-higher）
    kl_coef: float = 0.0  # β；0 表示不加 KL，也不需要参考模型
    ppo_epochs: int = 1  # 同一批样本更新几次
    loss_agg: str = "token_mean"  # "token_mean" | "seq_mean_token_mean"
    scale_rewards: bool = True  # 优势是否除以组内标准差
    n_train_tasks: int = 2000
    env_seed: int = 0
    forward_batch: int = 16  # 算 log 概率时每次前向多少条序列（省内存）


# ---------------------------------------------------------------------------
# 优势与损失
# ---------------------------------------------------------------------------


def group_advantages(rewards: torch.Tensor, scale: bool = True, eps: float = 1e-6) -> torch.Tensor:
    """rewards: (P, G) → 优势 (P, G)。A = (r − 组均值) / (组标准差 + eps)，标准差用无偏估计。"""
    mean = rewards.mean(dim=1, keepdim=True)
    adv = rewards - mean
    if scale:
        std = rewards.std(dim=1, keepdim=True) if rewards.shape[1] > 1 else torch.zeros_like(mean)
        adv = adv / (std + eps)
    return adv


def grpo_loss(
    logp: torch.Tensor,
    old_logp: torch.Tensor,
    advantages: torch.Tensor,
    mask: torch.Tensor,
    clip_eps: float = 0.2,
    clip_eps_high: float | None = None,
    ref_logp: torch.Tensor | None = None,
    kl_coef: float = 0.0,
    loss_agg: str = "token_mean",
    num_tokens: float | None = None,
    num_seqs: float | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """GRPO 损失。

    logp / old_logp / ref_logp / mask：(B, T)，只在 mask 为真的回复 token 上计算；advantages：(B,)。
    num_tokens / num_seqs：整个 batch 的回复 token 总数 / 序列数（分块前向时传入，保证分块求和
    等于整批一次算）；不传则用本块自己的。
    """
    hi = clip_eps if not clip_eps_high else clip_eps_high
    m = mask.float()
    ratio = torch.exp(logp - old_logp)
    adv = advantages[:, None]
    pg1 = -adv * ratio
    pg2 = -adv * torch.clamp(ratio, 1.0 - clip_eps, 1.0 + hi)
    per_tok = torch.maximum(pg1, pg2)
    kl = torch.zeros_like(per_tok)
    if ref_logp is not None and kl_coef > 0:
        d = ref_logp - logp
        kl = torch.exp(d) - d - 1.0
        per_tok = per_tok + kl_coef * kl
    if loss_agg == "token_mean":
        denom = num_tokens if num_tokens is not None else m.sum().clamp(min=1.0)
        loss = (per_tok * m).sum() / denom
    elif loss_agg == "seq_mean_token_mean":
        per_seq = (per_tok * m).sum(-1) / m.sum(-1).clamp(min=1.0)
        denom = num_seqs if num_seqs is not None else per_tok.shape[0]
        loss = per_seq.sum() / denom
    else:
        raise ValueError(f"未知的 loss_agg：{loss_agg}")
    with torch.no_grad():
        n = m.sum().clamp(min=1.0)
        clipped = ((ratio < 1.0 - clip_eps) | (ratio > 1.0 + hi)).float()
        metrics = {
            "kl": float((kl * m).sum() / n),
            "clip_frac": float((clipped * m).sum() / n),
            "ratio_mean": float((ratio * m).sum() / n),
        }
    return loss, metrics


# ---------------------------------------------------------------------------
# 采样
# ---------------------------------------------------------------------------


@dataclass
class Rollout:
    task_idx: int
    prompt_ids: list[int]
    response_ids: list[int]  # 含结尾的 <|im_end|>（若生成到了）
    text: str
    reward: float
    format_ok: bool
    n_calls: int = 0


@torch.no_grad()
def sample_group(
    model: torch.nn.Module,
    tok: Tokenizer,
    task: Any,
    G: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    seed: int,
) -> tuple[list[int], list[list[int]]]:
    """同一提示词复制 G 份，带 KV cache 生成。返回 (提示词 ids, G 个回复 ids)，回复以 <|im_end|> 结尾（若停下了）。"""
    from zero.generate import generate

    ids, _ = render(task.messages, task.tools, add_generation_prompt=True, tokenizer=tok)
    budget = min(max_new_tokens, model.config.max_seq_len - len(ids))
    device = next(model.parameters()).device
    prompt = torch.tensor([ids] * G, dtype=torch.long, device=device)
    outs = generate(model, prompt, budget, temperature, top_p, eos_id=tok.im_end_id, seed=seed)
    resp = []
    for o in outs:  # generate 去掉了 eos；没用满预算说明是遇到 eos 停下的，把它补回来参与训练
        o = list(o)  # type: ignore[arg-type]
        resp.append(o + [tok.im_end_id] if len(o) < budget else o)
    return list(ids), resp


# ---------------------------------------------------------------------------
# 训练
# ---------------------------------------------------------------------------


def _logps_chunked(
    model: torch.nn.Module,
    seqs: list[list[int]],
    masks: list[list[bool]],
    pad_id: int,
    device: torch.device,
    chunk: int,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    out = []
    for i in range(0, len(seqs), chunk):
        ids, mask = pad_batch(seqs[i : i + chunk], masks[i : i + chunk], pad_id, device)
        out.append(sequence_token_logprobs(model, ids, mask))
    return out


def run_grpo(
    src: str | os.PathLike | dict[str, Any],
    overrides: Sequence[str] | None = None,
    log: Callable[[str], None] = print,
) -> list[dict[str, Any]]:
    """单进程实现（CPU / 单卡）。单卡已在 RTX 3090 上验证（2026-10）；多卡与 vLLM 采样尚未在 GPU 上验证。"""
    from zero.post.envs.tool_env import generate_tasks, score_tool_calls

    cfg, sec = load_post_config(src, {"grpo": GRPOConfig}, overrides)
    gc: GRPOConfig = sec["grpo"]
    tc = cfg.train
    set_threads(tc.cpu_threads)
    torch.manual_seed(tc.seed)
    # CUDA 上用 BF16 autocast（与 Trainer 相同的规则；采样 sample_group 不在 autocast 里，是 FP32）：已在单张 RTX 3090 上验证（2026-10，见 runs/2026-10-01-gpu0-check/）
    device = torch.device(
        "cuda" if tc.device in ("auto", "cuda") and torch.cuda.is_available() else "cpu"
    )
    from zero.train.trainer import autocast_context

    def ac():  # noqa: ANN202
        return autocast_context(device, tc.dtype)

    tok = Tokenizer.load(tc.data.tokenizer)
    model = build_model_from_init(cfg, device)
    ref_model = None
    if gc.kl_coef > 0:
        ref_model = copy.deepcopy(model).eval()
        for p in ref_model.parameters():
            p.requires_grad_(False)
    tasks = generate_tasks(gc.n_train_tasks, seed=gc.env_seed, split="train")
    loop = LoopState(cfg, model, log)
    G, P = gc.group_size, gc.prompts_per_step
    log(
        f"[grpo] {len(tasks)} 个训练任务，每步 {P} 个提示词 × G={G}，ε={gc.clip_eps}，β={gc.kl_coef}，聚合 {gc.loss_agg}"
    )

    while loop.step < tc.max_steps:
        t0 = time.perf_counter()
        lr = loop.begin_step()
        rng = random.Random(
            tc.seed * 7919 + loop.step
        )  # 任务选择只由 (seed, step) 决定，续训可复现
        chosen = rng.sample(range(len(tasks)), P)

        # 1-2. 采样 + 打分
        rollouts: list[Rollout] = []
        for j, ti in enumerate(chosen):
            task = tasks[ti]
            p_ids, resps = sample_group(
                model,
                tok,
                task,
                G,
                gc.max_new_tokens,
                gc.temperature,
                gc.top_p,
                seed=tc.seed * 100_003 + loop.step * 101 + j,
            )
            for r in resps:
                text = tok.decode([t for t in r if t != tok.im_end_id])
                rw = score_tool_calls(task, text)
                rollouts.append(Rollout(ti, p_ids, r, text, rw.total, rw.format_ok, rw.n_calls))
        t_gen = time.perf_counter() - t0

        # 3. 组内优势
        rewards = torch.tensor([r.reward for r in rollouts], dtype=torch.float32).view(P, G)
        adv = group_advantages(rewards, gc.scale_rewards).view(-1).to(device)

        seqs = [r.prompt_ids + r.response_ids for r in rollouts]
        masks = [[False] * len(r.prompt_ids) + [True] * len(r.response_ids) for r in rollouts]
        n_tok = float(sum(len(r.response_ids) for r in rollouts))
        keep = [i for i in range(len(rollouts)) if len(rollouts[i].response_ids) > 0]

        # π_old 与 π_ref 的 log 概率（不求梯度）
        model.eval()
        with torch.no_grad(), ac():
            old = _logps_chunked(model, seqs, masks, tok.eot_id, device, gc.forward_batch)
            ref = (
                _logps_chunked(ref_model, seqs, masks, tok.eot_id, device, gc.forward_batch)
                if ref_model is not None
                else None
            )
        model.train()

        # 4-6. 更新（ppo_epochs 次；每次对整批做分块前向 + 反向，最后一起 step）
        metrics: dict[str, float] = {}
        loss_total = 0.0
        for epoch in range(gc.ppo_epochs):
            if epoch > 0:
                loop.optimizer_step()  # 前一次的梯度先更新掉，再用新参数重算 ρ
                metrics = {}
            loss_total = 0.0
            for ci, i in enumerate(range(0, len(seqs), gc.forward_batch)):
                sl = slice(i, i + gc.forward_batch)
                ids, mask = pad_batch(seqs[sl], masks[sl], tok.eot_id, device)
                with ac():
                    logp, tmask = sequence_token_logprobs(model, ids, mask)
                o_lp, _ = old[ci]
                r_lp = ref[ci][0] if ref is not None else None
                loss, m = grpo_loss(
                    logp,
                    o_lp,
                    adv[sl],
                    tmask,
                    gc.clip_eps,
                    gc.clip_eps_high,
                    r_lp,
                    gc.kl_coef,
                    gc.loss_agg,
                    num_tokens=max(n_tok, 1.0),
                    num_seqs=float(len(seqs)),
                )
                if keep:
                    loss.backward()
                loss_total += float(loss.detach())
                w = tmask.float().sum().item() / max(n_tok, 1.0)
                for k, v in m.items():
                    metrics[k] = metrics.get(k, 0.0) + v * w
        gnorm = loop.end_step()

        fmt_rate = sum(r.format_ok for r in rollouts) / len(
            rollouts
        )  # 纯文本（没调用）也算格式正确
        call_rate = sum(r.n_calls > 0 and r.format_ok for r in rollouts) / len(rollouts)
        zero_std = float((rewards.std(dim=1) == 0).float().mean())
        loop.record(
            {
                "loss": loss_total,
                "reward_mean": float(rewards.mean()),
                "reward_max": float(rewards.max()),
                "format_rate": fmt_rate,
                "call_rate": call_rate,
                "resp_len": n_tok / len(rollouts),
                "zero_std_groups": zero_std,
                **metrics,
                "lr": lr,
                "grad_norm": gnorm,
                "gen_s": t_gen,
                "step_s": time.perf_counter() - t0,
            },
            "step {step:>4} | reward {reward_mean:+.3f} | format {format_rate:.2f} | call {call_rate:.2f} | len {resp_len:.1f} "
            "| kl {kl:.4f} | clip {clip_frac:.2f} | loss {loss:+.4f} | {step_s:.1f}s",
        )
    return loop.history


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="GRPO（第 19 章）")
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = ap.parse_args(argv)
    run_grpo(args.config, args.set)


if __name__ == "__main__":
    main()
