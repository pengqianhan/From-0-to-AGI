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
    # if the file does not exist, make preference pairs with the tool environment
    generate_pairs: int = 0
    samples_per_prompt: int = 4
    gen_temperature: float = 1.0
    max_new_tokens: int = 96
    env_seed: int = 0


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
) -> list[dict[str, Any]]:
    """Make preference pairs with the tool environment + the current policy.

    The pairs cover only the first turn: "which tool to call".
    """
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
        if worst_r >= 0.999:  # all are correct: use a broken gold answer as rejected
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
        f"[dpo] made {len(rows)} preference pairs: chosen is a policy sample in {n_policy_chosen} pairs, the gold solution in the others"
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
    """Single-process implementation (CPU / 1 GPU).

    1 GPU was verified on an RTX 3090 (2026-10). Multi-GPU DDP is not verified on GPUs yet. In Step 2,
    change to torchrun as RUNBOOK describes.
    """
    cfg, sec = load_post_config(src, {"dpo": DPOConfig}, overrides)
    dc: DPOConfig = sec["dpo"]
    tc = cfg.train
    set_threads(tc.cpu_threads)
    torch.manual_seed(tc.seed)
    # On CUDA, use BF16 autocast (the same rule as Trainer). Verified on one RTX 3090
    # (2026-10, see runs/2026-10-01-gpu0-check/).
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
            raise FileNotFoundError(f"[dpo] {dc.train_jsonl} not found, and generate_pairs is not set")
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
        raise ValueError("[dpo] no usable preference pairs (are all of them too long?)")
    log(
        f"[dpo] {len(pairs)} preference pairs ({len(rows) - len(pairs)} dropped), β = {dc.beta}, reference model: {dc.ref_mode}"
    )

    # reference model: the policy at the start of training
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
        raise ValueError(f"[dpo] ref_mode must be precompute / online, not {dc.ref_mode!r}")

    loop = LoopState(cfg, model, log)
    model.train()
    per_step = bsz * tc.grad_accum_steps
    order: list[int] = []
    while loop.step < tc.max_steps:
        lr = loop.begin_step()
        agg: dict[str, float] = {}
        for _ in range(tc.grad_accum_steps):
            # (seed, epoch) sets the sample order, so a resumed run continues at the same place
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
    ap = argparse.ArgumentParser(description="DPO (Chapter 18)")
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = ap.parse_args(argv)
    run_dpo(args.config, args.set)


if __name__ == "__main__":
    main()
