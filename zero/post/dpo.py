"""DPO: direct preference optimization (Chapter 18).

    uv run python -m zero.post.dpo --config configs/tiny/dpo.toml

**Loss** (Rafailov et al. 2023, arXiv:2305.18290): for a pair of responses to the same prompt x
(y_w is better, y_l is worse),

    L = -log σ( β · [ (log π(y_w|x) - log π_ref(y_w|x)) - (log π(y_l|x) - log π_ref(y_l|x)) ] )

- log π(y|x) is the **sum of the log probabilities of each token** in the response. The prompt does
  not count (see `encode_prompt_response`).
- π_ref is the model at the start of training (after SFT), and it is frozen. `ref_mode = "precompute"`
  calculates the log probabilities of the reference model on all data before training and stores
  them (this saves the memory of one model). `"online"` calculates them at each step with a frozen copy.
- β controls "how far the model can move from the reference model": a larger β is more conservative.
- Implicit reward r(y) = β·(log π(y|x) − log π_ref(y|x)). In the log, `acc` is the fraction with
  r(y_w) > r(y_l), and `margin` is the mean of r(y_w) − r(y_l).

**Preference data format** (JSONL, one pair on each line):

    {"messages": [...prompt...], "tools": [...],
     "chosen":   {"role": "assistant", "content": ..., "tool_calls": [...]},
     "rejected": {"role": "assistant", "content": ...}}

chosen / rejected can also be a list of messages (the next turns of a multi-turn conversation).
If the config sets `[dpo] generate_pairs = N` and the file does not exist, `make_env_preferences` makes
preference pairs with the tool environment. For each task, it samples some responses from the current
policy and scores them with the verifiable reward. The response with the highest score is chosen (if it
is not good enough, the gold solution is chosen). The response with the lowest score is rejected. This
is "on-policy preference data".
"""

from __future__ import annotations

import argparse
import copy
import os
import random
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
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
    # if the file does not exist, make preference pairs with the tool environment
    generate_pairs: int = 0
    samples_per_prompt: int = 4
    gen_temperature: float = 1.0
    max_new_tokens: int = 96
    env_seed: int = 0
    # Online pairs from real function-calling tasks (zero/post/envs/fc_tasks.py); empty: tool_env
    task_files: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------


def dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    beta: float = 0.1,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Each input is (B,): the sum of the log probabilities of a sequence. Return (scalar loss, metrics)."""
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
# Data
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
    """Put chosen and rejected into one batch and do one forward pass.

    Return the sums of the sequence log probabilities of both: (B,), (B,).
    """
    seqs = [p.chosen_ids for p in pairs] + [p.rejected_ids for p in pairs]
    masks = [p.chosen_mask for p in pairs] + [p.rejected_mask for p in pairs]
    ids, mask = pad_batch(seqs, masks, pad_id, device)
    logp, tmask = sequence_token_logprobs(model, ids, mask)
    seq = (logp * tmask).sum(-1)
    return seq[: len(pairs)], seq[len(pairs) :]


def corrupt_call(call: dict[str, Any], rng: random.Random) -> str:
    """Break the gold call (wrong argument / broken JSON) to use as a fallback rejected response."""
    c = copy.deepcopy(call)
    if rng.random() < 0.5 and c["arguments"]:
        k = rng.choice(list(c["arguments"]))
        v = c["arguments"][k]
        c["arguments"][k] = (v + 1) if isinstance(v, int | float) else (str(v) + "0")
        return format_tool_call(c)
    return format_tool_call(c)[:-14]  # cut off "}\n</tool_call>": a format error


def make_env_preferences(
    model: torch.nn.Module,
    tok: Tokenizer,
    n_pairs: int,
    samples_per_prompt: int = 4,
    temperature: float = 1.0,
    max_new_tokens: int = 96,
    seed: int = 0,
    log: Callable[[str], None] = print,
    task_files: Sequence[str] = (),
) -> list[dict[str, Any]]:
    """Make preference pairs with the current policy on tool-call tasks (first turn: "which tool to call").

    The tasks come from `task_files` (real function-calling tasks, `zero/post/envs/fc_tasks.py`) or, if
    there are none, from the toy tool environment. A real "no call" task has no reference reply; if no
    sample of the policy is correct on it, the task gives no pair.
    """
    from zero.post.chat import parse_assistant
    from zero.post.envs.fc_tasks import load_task_pool, score_any

    rng = random.Random(seed)
    pool = load_task_pool(task_files, n_pairs, seed + 101)
    tasks = rng.sample(pool, min(n_pairs, len(pool))) if task_files else pool
    rows = []
    n_policy_chosen = n_skipped = 0
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
        scored = [(score_any(task, o), o) for o in outs]
        scored.sort(key=lambda x: -x[0].total)
        gold = (
            "\n".join(
                format_tool_call({"name": c["name"], "arguments": c["arguments"]})
                for c in task.gold_calls
            )
            if task.gold_calls
            else getattr(task, "gold_answer", None)
        )
        (best_rw, best), (worst_rw, worst) = scored[0], scored[-1]
        if best_rw.exact:
            chosen_text, src = best, "policy"
            n_policy_chosen += 1
        elif gold:
            chosen_text, src = gold, "gold"
        else:
            n_skipped += 1  # a real "no call" task without a correct sample: no reference reply
            continue
        if worst_rw.exact:  # all are correct: use a broken gold answer as rejected
            if task.gold_calls:
                g0 = task.gold_calls[0]
                worst = corrupt_call({"name": g0["name"], "arguments": g0["arguments"]}, rng)
            elif task.tools:
                worst = format_tool_call(
                    {
                        "name": (task.tools[0].get("function") or task.tools[0])["name"],
                        "arguments": {},
                    }
                )
            else:
                n_skipped += 1
                continue
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
        f"[dpo] made {len(rows)} preference pairs: chosen is a policy sample in {n_policy_chosen} pairs, "
        f"the gold solution in the others; {n_skipped} tasks gave no pair"
    )
    return rows


def _as_message(text: str, parse: Callable[[str], Any]) -> dict[str, Any]:
    """Assistant text → message.

    Text with a broken format goes into content as it is, so the rendered text stays the same.
    """
    p = parse(text)
    if p.errors:
        return {"role": "assistant", "content": text}
    return p.to_message()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def run_dpo(
    src: str | os.PathLike | dict[str, Any],
    overrides: Sequence[str] | None = None,
    log: Callable[[str], None] = print,
) -> list[dict[str, Any]]:
    """CPU, 1 GPU, or N GPUs with torchrun (data parallel; see the docstring of `zero.post.common`).

    **Batch**: one optimizer step uses `micro_batch_size × grad_accum_steps` different pairs (global,
    independent of the number of GPUs). The ranks split them; one forward pass holds at most
    `micro_batch_size` pairs. (Before 2026-10-10, every micro-step of a step reused the same
    `micro_batch_size` pairs, so the rest of the step's pairs were never trained on.)
    """
    from zero.train.dist import cleanup, init_distributed

    cfg, sec = load_post_config(src, {"dpo": DPOConfig}, overrides)
    info = init_distributed(cfg.train.device)
    try:
        return _dpo_loop(cfg, sec["dpo"], info, log)
    finally:
        cleanup()


def step_pair_indices(step: int, per_step: int, n: int, seed: int) -> list[int]:
    """The pair indices of one optimizer step: a window of a per-epoch shuffle.

    Only (seed, step) decides it, so a resumed run continues at the same place, on every rank.
    """
    out = []
    orders: dict[int, list[int]] = {}  # at most two epochs per step
    for j in range(per_step):
        g = step * per_step + j  # global sample number since the start of training
        epoch, pos = divmod(g, n)
        if epoch not in orders:
            order = list(range(n))
            random.Random(seed + epoch).shuffle(order)
            orders[epoch] = order
        out.append(orders[epoch][pos])
    return out


def _dpo_loop(
    cfg: Any, dc: DPOConfig, info: Any, log: Callable[[str], None]
) -> list[dict[str, Any]]:
    import torch.distributed as dist

    from zero.post.common import all_reduce_sum, rank_share
    from zero.train.dist import barrier
    from zero.train.trainer import autocast_context

    tc = cfg.train
    set_threads(tc.cpu_threads)
    torch.manual_seed(tc.seed)
    # On CUDA, use BF16 autocast (the same rule as Trainer). Verified on one RTX 3090
    # (2026-10, see runs/2026-10-01-gpu0-check/).
    device = info.device

    def ac():  # noqa: ANN202
        return autocast_context(device, tc.dtype)

    tok = Tokenizer.load(tc.data.tokenizer)
    model = build_model_from_init(cfg, device)

    if not os.path.exists(dc.train_jsonl):
        if dc.generate_pairs <= 0:
            raise FileNotFoundError(
                f"[dpo] {dc.train_jsonl} not found, and generate_pairs is not set"
            )
        if info.is_main:  # one process writes the file; the others wait and read it
            rows = make_env_preferences(
                model,
                tok,
                dc.generate_pairs,
                dc.samples_per_prompt,
                dc.gen_temperature,
                dc.max_new_tokens,
                dc.env_seed,
                log,
                task_files=dc.task_files,
            )
            write_jsonl(dc.train_jsonl, rows)
    barrier()
    rows = read_jsonl(dc.train_jsonl)
    pairs = [p for p in (encode_pair(r, tok, tc.data.seq_len) for r in rows) if p is not None]
    if not pairs:
        raise ValueError("[dpo] no usable preference pairs (are all of them too long?)")
    bsz = tc.micro_batch_size
    per_step = bsz * tc.grad_accum_steps
    log(
        f"[dpo] {len(pairs)} preference pairs ({len(rows) - len(pairs)} dropped), {per_step} pairs per step "
        f"on {info.world_size} process(es), β = {dc.beta}, reference model: {dc.ref_mode}"
    )

    # Reference model: the policy at the start of training
    ref_model = None
    ref_c = ref_r = None
    if dc.ref_mode == "precompute":
        # Each rank computes its share; zeros elsewhere, so a sum over the ranks fills the full vector
        model.eval()
        ref_c = torch.zeros(len(pairs), dtype=torch.float32, device=device)
        ref_r = torch.zeros(len(pairs), dtype=torch.float32, device=device)
        mine = [i for i, _ in rank_share(range(len(pairs)), info)]
        with torch.no_grad():
            for c in range(0, len(mine), bsz):
                idx = mine[c : c + bsz]
                with ac():
                    cc, rr = batch_logps(model, [pairs[i] for i in idx], tok.eot_id, device)
                ref_c[idx] = cc.float()
                ref_r[idx] = rr.float()
        if info.is_distributed:
            dist.all_reduce(ref_c)
            dist.all_reduce(ref_r)
    elif dc.ref_mode == "online":
        ref_model = copy.deepcopy(model).eval()
        for p in ref_model.parameters():
            p.requires_grad_(False)
    else:
        raise ValueError(f"[dpo] ref_mode must be precompute / online, not {dc.ref_mode!r}")

    loop = LoopState(cfg, model, log, info=info)
    model.train()
    keys = ("loss", "acc", "margin", "chosen_reward", "rejected_reward")
    while loop.step < tc.max_steps:
        t0 = time.perf_counter()
        lr = loop.begin_step()
        step_idx = step_pair_indices(loop.step, per_step, len(pairs), tc.seed)
        mine = [i for _, i in rank_share(step_idx, info)]
        sums = dict.fromkeys(keys, 0.0)
        for c in range(0, len(mine), bsz):
            idx = mine[c : c + bsz]
            batch = [pairs[j] for j in idx]
            with ac():
                pc, pr = batch_logps(model, batch, tok.eot_id, device)
            if ref_model is not None:
                with torch.no_grad(), ac():
                    rc, rr = batch_logps(ref_model, batch, tok.eot_id, device)
            else:
                assert ref_c is not None and ref_r is not None
                rc, rr = ref_c[idx], ref_r[idx]
            loss, m = dpo_loss(pc, pr, rc, rr, dc.beta)
            (loss * (len(idx) / per_step)).backward()  # the mean over all pairs of the step
            for k in keys:
                sums[k] += m[k] * len(idx)
        gnorm = loop.end_step()
        tot = all_reduce_sum([sums[k] for k in keys], info)
        loop.record(
            {
                **{k: v / per_step for k, v in zip(keys, tot)},
                "lr": lr,
                "grad_norm": gnorm,
                "step_s": time.perf_counter() - t0,
            },
            "step {step:>5} | dpo loss {loss:.4f} | acc {acc:.2f} | margin {margin:+.3f} | lr {lr:.2e}",
        )
    return loop.history


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="DPO (Chapter 18)")
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = ap.parse_args(argv)
    run_dpo(args.config, args.set)


if __name__ == "__main__":
    main()
