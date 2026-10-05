"""DPO：直接偏好优化（对应第 18 章）。

    uv run python -m zero.post.dpo --config configs/tiny/dpo.toml

**损失**（Rafailov et al. 2023, arXiv:2305.18290）：对同一个提示词 x 的一对回复 (y_w 更好, y_l 更差)，

    L = -log σ( β · [ (log π(y_w|x) - log π_ref(y_w|x)) - (log π(y_l|x) - log π_ref(y_l|x)) ] )

- log π(y|x) 是回复里**每个 token 的 log 概率之和**（提示词部分不算，见 `encode_prompt_response`）；
- π_ref 是训练开始时的模型（SFT 之后），冻结不动。`ref_mode = "precompute"` 在训练前把参考模型
  对全部数据的 log 概率算好存起来（省一份模型的显存）；`"online"` 每步用一份冻结的拷贝现算；
- β 控制"离参考模型能走多远"：β 越大越保守；
- 隐式奖励 r(y) = β·(log π(y|x) − log π_ref(y|x))；日志里的 `acc` 是 r(y_w) > r(y_l) 的比例、
  `margin` 是 r(y_w) − r(y_l) 的平均值。

**偏好数据格式**（JSONL，每行一对）：

    {"messages": [...提示词...], "tools": [...],
     "chosen":   {"role": "assistant", "content": ..., "tool_calls": [...]},
     "rejected": {"role": "assistant", "content": ...}}

chosen / rejected 也可以是消息列表（多轮的后续）。`[dpo] generate_pairs = N` 且文件不存在时，
`make_env_preferences` 用工具环境现场造偏好对：对每个任务从当前策略采样若干回复、用可验证奖励打分，
得分最高的（不够好就用标准解答）当 chosen，得分最低的当 rejected——"on-policy 偏好数据"。
"""

from __future__ import annotations

import argparse
import copy
import os
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from zero.post.chat import encode_prompt_response, format_tool_call
from zero.post.common import (
    LoopState,
    build_model_from_init,
    chat_complete,
    load_post_config,
    pad_batch,
    read_jsonl,
    sequence_token_logprobs,
    set_threads,
    write_jsonl,
)
from zero.tokenizer import Tokenizer


@dataclass
class DPOConfig:
    train_jsonl: str = ""
    beta: float = 0.1
    ref_mode: str = "precompute"  # "precompute" | "online"
    # 文件不存在时用工具环境造偏好对
    generate_pairs: int = 0
    samples_per_prompt: int = 4
    gen_temperature: float = 1.0
    max_new_tokens: int = 96
    env_seed: int = 0


# ---------------------------------------------------------------------------
# 损失
# ---------------------------------------------------------------------------


def dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    beta: float = 0.1,
) -> tuple[torch.Tensor, dict[str, float]]:
    """输入都是 (B,) 的序列 log 概率之和。返回 (标量损失, 指标)。"""
    chosen_rewards = beta * (policy_chosen_logps - ref_chosen_logps)
    rejected_rewards = beta * (policy_rejected_logps - ref_rejected_logps)
    logits = chosen_rewards - rejected_rewards
    loss = -F.logsigmoid(logits).mean()
    metrics = {
        "loss": float(loss.detach()),
        "acc": float((logits > 0).float().mean()),
        "margin": float(logits.detach().mean()),
        "chosen_reward": float(chosen_rewards.detach().mean()),
        "rejected_reward": float(rejected_rewards.detach().mean()),
    }
    return loss, metrics


# ---------------------------------------------------------------------------
# 数据
# ---------------------------------------------------------------------------


@dataclass
class PreferencePair:
    chosen_ids: list[int]
    chosen_mask: list[bool]
    rejected_ids: list[int]
    rejected_mask: list[bool]


def encode_pair(row: dict[str, Any], tok: Tokenizer, max_len: int) -> PreferencePair | None:
    msgs, tools = row["messages"], row.get("tools")
    c_ids, c_mask = encode_prompt_response(msgs, row["chosen"], tok, tools)
    r_ids, r_mask = encode_prompt_response(msgs, row["rejected"], tok, tools)
    if max(len(c_ids), len(r_ids)) > max_len or not any(c_mask) or not any(r_mask):
        return None
    return PreferencePair(c_ids, c_mask, r_ids, r_mask)


def batch_logps(
    model: torch.nn.Module, pairs: Sequence[PreferencePair], pad_id: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """chosen 和 rejected 拼成一个 batch 前向一次，返回两者的序列 log 概率之和 (B,), (B,)。"""
    seqs = [p.chosen_ids for p in pairs] + [p.rejected_ids for p in pairs]
    masks = [p.chosen_mask for p in pairs] + [p.rejected_mask for p in pairs]
    ids, mask = pad_batch(seqs, masks, pad_id, device)
    logp, tmask = sequence_token_logprobs(model, ids, mask)
    seq = (logp * tmask).sum(-1)
    return seq[: len(pairs)], seq[len(pairs) :]


def corrupt_call(call: dict[str, Any], rng: random.Random) -> str:
    """把标准调用改坏（错参数 / JSON 破损），作为兜底的 rejected。"""
    c = copy.deepcopy(call)
    if rng.random() < 0.5 and c["arguments"]:
        k = rng.choice(list(c["arguments"]))
        v = c["arguments"][k]
        c["arguments"][k] = (v + 1) if isinstance(v, int | float) else (str(v) + "0")
        return format_tool_call(c)
    return format_tool_call(c)[:-14]  # 截掉 "}\n</tool_call>"：格式错误


def make_env_preferences(
    model: torch.nn.Module,
    tok: Tokenizer,
    n_pairs: int,
    samples_per_prompt: int = 4,
    temperature: float = 1.0,
    max_new_tokens: int = 96,
    seed: int = 0,
    log: Callable[[str], None] = print,
) -> list[dict[str, Any]]:
    """用工具环境 + 当前策略造偏好对（只针对"第一轮：该调用什么工具"）。"""
    from zero.post.chat import parse_assistant
    from zero.post.envs.tool_env import generate_tasks, score_tool_calls

    rng = random.Random(seed)
    tasks = generate_tasks(n_pairs, seed=seed + 101, split="train")
    rows = []
    n_policy_chosen = 0
    for i, task in enumerate(tasks):
        outs = chat_complete(
            model,
            tok,
            task.messages,
            task.tools,
            max_new_tokens,
            temperature,
            n=samples_per_prompt,
            seed=seed * 1000 + i,
        )
        scored = sorted(((score_tool_calls(task, o).total, o) for o in outs), key=lambda x: -x[0])
        gold = (
            "\n".join(format_tool_call(c) for c in task.gold_calls)
            if task.gold_calls
            else task.gold_answer
        )
        best_r, best = scored[0]
        worst_r, worst = scored[-1]
        if best_r >= 0.999:
            chosen_text, src = best, "policy"
            n_policy_chosen += 1
        else:
            chosen_text, src = gold, "gold"
        if worst_r >= 0.999:  # 全都对：用改坏的标准答案当 rejected
            worst = (
                corrupt_call(task.gold_calls[0], rng)
                if task.gold_calls
                else format_tool_call({"name": task.tools[0]["function"]["name"], "arguments": {}})
            )
        rows.append(
            {
                "messages": task.messages,
                "tools": task.tools,
                "chosen": _as_message(chosen_text, parse_assistant),
                "rejected": _as_message(worst, parse_assistant),
                "chosen_source": src,
                "task_id": task.id,
            }
        )
    log(
        f"[dpo] 造了 {len(rows)} 对偏好数据，其中 chosen 来自策略自身采样的 {n_policy_chosen} 对，其余用标准解答"
    )
    return rows


def _as_message(text: str, parse: Callable[[str], Any]) -> dict[str, Any]:
    """助手文本 → 消息。格式坏掉的文本原样放进 content（渲染出来还是同一段文本）。"""
    p = parse(text)
    if p.errors:
        return {"role": "assistant", "content": text}
    return p.to_message()


# ---------------------------------------------------------------------------
# 训练
# ---------------------------------------------------------------------------


def run_dpo(
    src: str | os.PathLike | dict[str, Any],
    overrides: Sequence[str] | None = None,
    log: Callable[[str], None] = print,
) -> list[dict[str, Any]]:
    """单进程实现（CPU / 单卡）。单卡已在 RTX 3090 上验证（2026-10）；多卡 DDP 尚未在 GPU 上验证，第二步按 RUNBOOK 的说明改用 torchrun。"""
    cfg, sec = load_post_config(src, {"dpo": DPOConfig}, overrides)
    dc: DPOConfig = sec["dpo"]
    tc = cfg.train
    set_threads(tc.cpu_threads)
    torch.manual_seed(tc.seed)
    # CUDA 上用 BF16 autocast（与 Trainer 相同的规则）：已在单张 RTX 3090 上验证（2026-10，见 runs/2026-10-01-gpu0-check/）
    device = torch.device(
        "cuda" if tc.device in ("auto", "cuda") and torch.cuda.is_available() else "cpu"
    )
    from zero.train.trainer import autocast_context

    def ac():  # noqa: ANN202
        return autocast_context(device, tc.dtype)

    tok = Tokenizer.load(tc.data.tokenizer)
    model = build_model_from_init(cfg, device)

    if not os.path.exists(dc.train_jsonl):
        if dc.generate_pairs <= 0:
            raise FileNotFoundError(f"[dpo] 找不到 {dc.train_jsonl}，也没有设置 generate_pairs")
        rows = make_env_preferences(
            model,
            tok,
            dc.generate_pairs,
            dc.samples_per_prompt,
            dc.gen_temperature,
            dc.max_new_tokens,
            dc.env_seed,
            log,
        )
        write_jsonl(dc.train_jsonl, rows)
    rows = read_jsonl(dc.train_jsonl)
    pairs = [p for p in (encode_pair(r, tok, tc.data.seq_len) for r in rows) if p is not None]
    if not pairs:
        raise ValueError("[dpo] 没有可用的偏好对（都超长？）")
    log(
        f"[dpo] {len(pairs)} 对偏好数据（丢弃 {len(rows) - len(pairs)} 对），β = {dc.beta}，参考模型：{dc.ref_mode}"
    )

    # 参考模型：训练开始时的策略
    ref_model = None
    ref_c = ref_r = None
    bsz = tc.micro_batch_size
    if dc.ref_mode == "precompute":
        model.eval()
        with torch.no_grad():
            cs, rs = [], []
            for i in range(0, len(pairs), bsz):
                with ac():
                    c, r = batch_logps(model, pairs[i : i + bsz], tok.eot_id, device)
                cs.append(c)
                rs.append(r)
        ref_c, ref_r = torch.cat(cs), torch.cat(rs)
    elif dc.ref_mode == "online":
        ref_model = copy.deepcopy(model).eval()
        for p in ref_model.parameters():
            p.requires_grad_(False)
    else:
        raise ValueError(f"[dpo] ref_mode 只能是 precompute / online，当前 {dc.ref_mode!r}")

    loop = LoopState(cfg, model, log)
    model.train()
    per_step = bsz * tc.grad_accum_steps
    order: list[int] = []
    while loop.step < tc.max_steps:
        lr = loop.begin_step()
        agg: dict[str, float] = {}
        for _ in range(tc.grad_accum_steps):
            # 样本顺序由 (seed, epoch) 决定，续训后接得上
            start = (loop.step * per_step) % len(pairs)
            epoch = (loop.step * per_step) // len(pairs)
            order = list(range(len(pairs)))
            random.Random(tc.seed + epoch).shuffle(order)
            idx = [order[(start + j) % len(pairs)] for j in range(bsz)]
            batch = [pairs[j] for j in idx]
            with ac():
                pc, pr = batch_logps(model, batch, tok.eot_id, device)
            if ref_model is not None:
                with torch.no_grad():
                    with ac():
                        rc, rr = batch_logps(ref_model, batch, tok.eot_id, device)
            else:
                assert ref_c is not None and ref_r is not None
                rc, rr = ref_c[idx], ref_r[idx]
            loss, m = dpo_loss(pc, pr, rc, rr, dc.beta)
            (loss / tc.grad_accum_steps).backward()
            for k, v in m.items():
                agg[k] = agg.get(k, 0.0) + v / tc.grad_accum_steps
        gnorm = loop.end_step()
        loop.record(
            {**agg, "lr": lr, "grad_norm": gnorm},
            "step {step:>5} | dpo loss {loss:.4f} | acc {acc:.2f} | margin {margin:+.3f} | lr {lr:.2e}",
        )
    return loop.history


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="DPO（第 18 章）")
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = ap.parse_args(argv)
    run_dpo(args.config, args.set)


if __name__ == "__main__":
    main()
