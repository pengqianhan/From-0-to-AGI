"""GRPO: group relative policy optimization + verifiable rewards (Chapter 19).

    uv run python -m zero.post.grpo --config configs/tiny/grpo.toml

Each step:

1. Take P tasks (`zero/post/envs/tool_env.py`). For each task, sample G responses from the current
   policy. The same prompt is copied G times to make one batch. `zero.generate` generates with the
   KV cache and stops at `<|im_end|>`.
2. The verifiable reward `score_tool_calls` gives each response a score r_1..r_G.
3. **Group-normalized advantage**: A_i = (r_i − mean(r)) / (std(r) + ε). std is the unbiased
   estimate, the same as TRL / verl. With `scale_rewards = false`, only the mean is subtracted (the
   Dr. GRPO method). When all responses in a group are correct, or all are wrong, all A are 0, and the
   group gives no gradient.
4. **Clipped importance ratio** (PPO-clip): for each token t in the response,
       ρ_t = π_θ(y_t | ·) / π_old(y_t | ·)
       ℓ_t = −min(ρ_t · A, clip(ρ_t, 1−ε_low, 1+ε_high) · A)
   π_old is the policy at sampling time (the same weights calculate the log probabilities once before
   the update). With `ppo_epochs = 1`, ρ ≡ 1, but the gradient ∇ρ = ∇log π is not zero. With
   `ppo_epochs > 1`, the same batch gives several updates, and only then does the clipping have an effect.
5. **Optional KL**: with kl_coef > 0, add β · KL(π_θ ‖ π_ref). The k3 estimate for each token is
   exp(ref − logp) − (ref − logp) − 1 (the DeepSeekMath form; it is non-negative and unbiased).
6. **Token-level aggregation**: `loss_agg = "token_mean"` (default) takes the mean of ℓ over all
   response tokens in a batch. Each token of a long response has the same weight as a token of a short
   response (the DAPO method; it prevents "the tokens of long responses get diluted").
   `"seq_mean_token_mean"` is the form of the original GRPO paper: first the mean in each response,
   then the mean over the responses.

Log: mean reward, format rate (fraction of responses without format errors; plain text without a tool
call also has a correct format), call rate (fraction with at least one tool call in the correct
format), response length, KL, clip fraction, and the fraction of "zero-variance groups".

**Parity check with verl (Step 2, not done yet)**: if the throughput is not sufficient, use verl
(see references.md: XiaomiMiMo/verl). Parity-check method: the same exported HF model, the same batch of
tool_env tasks (exported to JSONL with a fixed seed), and the same G / ε / β / learning rate /
aggregation. In verl, `algorithm.adv_estimator=grpo`, `actor_rollout_ref.actor.loss_agg_mode=token-mean`,
and `use_kl_loss` with `kl_loss_type=low_var_kl` match the k3 here. The reward function wraps
`score_tool_calls` in the custom reward interface of verl. First compare the advantages and the loss of
the first step (the same batch of samples must give bit-identical values). Then compare the mean reward
curve of the first 50 steps. (The verl config names come from its 0.9 documentation; to be verified.)
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
    group_size: int = 8  # G: number of responses sampled for each prompt
    prompts_per_step: int = 4  # P: number of prompts in each step
    max_new_tokens: int = 64
    temperature: float = 1.0
    top_p: float = 1.0
    clip_eps: float = 0.2  # ε_low
    clip_eps_high: float = 0.0  # ε_high; 0 means the same as clip_eps (>ε_low is the clip-higher of DAPO)
    kl_coef: float = 0.0  # β; 0 means no KL term and no reference model
    ppo_epochs: int = 1  # number of updates on the same batch of samples
    loss_agg: str = "token_mean"  # "token_mean" | "seq_mean_token_mean"
    scale_rewards: bool = True  # divide the advantage by the standard deviation of the group
    n_train_tasks: int = 2000
    env_seed: int = 0
    forward_batch: int = 16  # sequences in each forward pass for the log probabilities (saves memory)


# ---------------------------------------------------------------------------
# Advantage and loss
# ---------------------------------------------------------------------------


def group_advantages(rewards: torch.Tensor, scale: bool = True, eps: float = 1e-6) -> torch.Tensor:
    """rewards: (P, G) → advantages (P, G).

    A = (r − group mean) / (group standard deviation + eps). The standard deviation is the unbiased estimate.
    """
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
    """GRPO loss.

    logp / old_logp / ref_logp / mask: (B, T). Only the response tokens where mask is true count.
    advantages: (B,).
    num_tokens / num_seqs: the total number of response tokens / sequences in the full batch. Give them
    for a chunked forward pass, so that the sum over the chunks is equal to one pass on the full batch.
    If they are not given, the function uses the values of this chunk.
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
        raise ValueError(f"Unknown loss_agg: {loss_agg}")
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
# Sampling
# ---------------------------------------------------------------------------


@dataclass
class Rollout:
    task_idx: int
    prompt_ids: list[int]
    response_ids: list[int]  # includes the final <|im_end|> (if the model generated it)
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
    """Copy the same prompt G times and generate with the KV cache.

    Return (prompt ids, G response ids). A response ends with <|im_end|> if the generation stopped there.
    """
    from zero.generate import generate

    ids, _ = render(task.messages, task.tools, add_generation_prompt=True, tokenizer=tok)
    budget = min(max_new_tokens, model.config.max_seq_len - len(ids))
    device = next(model.parameters()).device
    prompt = torch.tensor([ids] * G, dtype=torch.long, device=device)
    outs = generate(model, prompt, budget, temperature, top_p, eos_id=tok.im_end_id, seed=seed)
    resp = []
    for o in outs:  # generate removes the eos. If the response is shorter than the budget, it stopped at the eos:
        # add the eos back, so that training uses it
        o = list(o)  # type: ignore[arg-type]
        resp.append(o + [tok.im_end_id] if len(o) < budget else o)
    return list(ids), resp


# ---------------------------------------------------------------------------
# Training
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
    """Single-process implementation (CPU / 1 GPU).

    1 GPU was verified on an RTX 3090 (2026-10). Multi-GPU and vLLM sampling are not verified on GPUs yet.
    """
    from zero.post.envs.tool_env import generate_tasks, score_tool_calls

    cfg, sec = load_post_config(src, {"grpo": GRPOConfig}, overrides)
    gc: GRPOConfig = sec["grpo"]
    tc = cfg.train
    set_threads(tc.cpu_threads)
    torch.manual_seed(tc.seed)
    # On CUDA, use BF16 autocast (the same rule as Trainer; sample_group is not in autocast, it is FP32).
    # Verified on one RTX 3090 (2026-10, see runs/2026-10-01-gpu0-check/).
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
        f"[grpo] {len(tasks)} training tasks, {P} prompts × G={G} per step, ε={gc.clip_eps}, β={gc.kl_coef}, aggregation {gc.loss_agg}"
    )

    while loop.step < tc.max_steps:
        t0 = time.perf_counter()
        lr = loop.begin_step()
        rng = random.Random(
            tc.seed * 7919 + loop.step
        )  # only (seed, step) selects the tasks, so a resumed run is reproducible
        chosen = rng.sample(range(len(tasks)), P)

        # 1-2. Sample + score
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

        # 3. Group advantages
        rewards = torch.tensor([r.reward for r in rollouts], dtype=torch.float32).view(P, G)
        adv = group_advantages(rewards, gc.scale_rewards).view(-1).to(device)

        seqs = [r.prompt_ids + r.response_ids for r in rollouts]
        masks = [[False] * len(r.prompt_ids) + [True] * len(r.response_ids) for r in rollouts]
        n_tok = float(sum(len(r.response_ids) for r in rollouts))
        keep = [i for i in range(len(rollouts)) if len(rollouts[i].response_ids) > 0]

        # Log probabilities of π_old and π_ref (no gradient)
        model.eval()
        with torch.no_grad(), ac():
            old = _logps_chunked(model, seqs, masks, tok.eot_id, device, gc.forward_batch)
            ref = (
                _logps_chunked(ref_model, seqs, masks, tok.eot_id, device, gc.forward_batch)
                if ref_model is not None
                else None
            )
        model.train()

        # 4-6. Update (ppo_epochs times; each time a chunked forward + backward pass on the full batch,
        # then one step at the end)
        metrics: dict[str, float] = {}
        loss_total = 0.0
        for epoch in range(gc.ppo_epochs):
            if epoch > 0:
                loop.optimizer_step()  # apply the previous gradient first, then calculate ρ again with the new parameters
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
        )  # plain text (no call) also has a correct format
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
    ap = argparse.ArgumentParser(description="GRPO (Chapter 19)")
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = ap.parse_args(argv)
    run_grpo(args.config, args.set)


if __name__ == "__main__":
    main()
