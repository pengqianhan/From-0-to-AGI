"""Cross-stage on-policy distillation (OPD): the last post-training stage (Chapter 17).

    uv run python -m zero.post.opd --config configs/tiny/opd.toml

**Why this stage.** Each post-training stage optimizes one goal, and the model can forget what the
earlier stages taught it. GRPO on tool calls can, for example, make general chat worse. GLM-5 (§3.5,
"On-Policy Cross-Stage Distillation") adds one last stage: the final checkpoints of the earlier stages
are the **teachers**, and the student learns back each skill from the teacher that was best at it.
DeepSeek-V4, MiMo-V2-Flash, and K2-Horizon also use on-policy distillation as the last step, to merge
several domain experts into one model (Chapter 17, "Adopters and sources").

**Why the teachers are our own checkpoints.** On-policy distillation compares the teacher and the
student token by token, so both must use the same vocabulary. The main-line model uses its own
tokenizer, and no open teacher has it. The checkpoints of our own earlier stages have the same tokenizer
by construction. This module checks the tokenizer hash of each teacher and refuses a different one.

**Each step:**

1. Take P prompts. Each teacher has its own prompt pool (for example: the GRPO teacher → tool_env
   tasks; the SFT/distill teacher → the prompts of the SFT conversations). `weight` sets the share of
   each pool. Only (seed, step) selects the prompts, so a resumed run is reproducible.
2. The **student** samples `samples_per_prompt` responses for each prompt (on-policy: the student
   trains on its own outputs, so there is no train/inference mismatch of the inputs).
3. The teacher of the prompt scores each token of the response. Two losses:
   - `loss = "full_kl"` (default): the exact reverse KL over the whole vocabulary at each response
     position, KL(p_S ‖ p_T) = Σ_v p_S(v)·(log p_S(v) − log p_T(v)). We have the teacher locally, so
     the full distribution is cheap. This is the GKD form of Agarwal et al. 2023 with λ = 1.
   - `loss = "sampled"`: the GLM-5 / Thinking Machines form. Each sampled token gets the advantage
     A_t = sg(log p_T(y_t) − log p_S(y_t)), and loss = −A_t · log p_S(y_t) (a policy gradient with a
     per-token reward). Its expectation has the same gradient as the reverse KL, but it needs only the
     log probability of the sampled token. Use it when the teacher runs in an inference server that
     returns only log probabilities.
   Both losses are a token-level mean over all response tokens of the step (the same aggregation as
   `grpo.py`, `loss_agg = "token_mean"`).

There is no reward and no group: GLM-5 sets the group size to 1 in this stage, because the advantage
comes from the gap to the teacher, not from a group of samples.

Log: the reverse KL estimate on the sampled tokens for each teacher (`kl/<name>`, the mean of
log p_S − log p_T), the response length, and the fraction of responses that end with `<|im_end|>`
(`eos_rate`; if it goes down, the student has started to ramble).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

from zero.post.chat import render
from zero.post.common import (
    LoopState,
    build_model_from_init,
    load_policy,
    load_post_config,
    pad_batch,
    read_jsonl,
    set_threads,
    token_logprobs,
)
from zero.post.distill import reverse_kl_loss
from zero.tokenizer import Tokenizer


@dataclass
class OPDTeacher:
    name: str = ""  # used in the log: kl/<name>
    path: str = ""  # zero checkpoint folder of an earlier stage (same tokenizer as the student)
    prompts: str = "tool_env"  # "tool_env", or a JSONL file of conversations ({"messages": [...], "tools": [...]})
    weight: float = 1.0  # share of this teacher's prompts in each step
    max_prompts: int = 0  # 0 = use all rows of the JSONL file


@dataclass
class OPDConfig:
    teachers: list[OPDTeacher] = field(default_factory=list)
    prompts_per_step: int = 32  # P
    samples_per_prompt: int = 1  # GLM-5 uses 1 in this stage
    max_new_tokens: int = 256
    temperature: float = 1.0
    top_p: float = 1.0
    loss: str = "full_kl"  # "full_kl" | "sampled"
    n_tool_tasks: int = 2000  # size of the tool_env pool (for teachers with prompts = "tool_env")
    env_seed: int = 0
    forward_batch: int = 16  # sequences in each forward pass (saves memory)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------


def sampled_opd_loss(
    logp_s: torch.Tensor, logp_t: torch.Tensor, mask: torch.Tensor, num_tokens: float
) -> torch.Tensor:
    """Per-token policy gradient with the advantage A_t = sg(log p_T − log p_S).

    logp_s (with gradient), logp_t: (B, T), log probabilities of the sampled tokens. mask: (B, T).
    Returns the sum over the masked tokens divided by num_tokens (the token count of the whole step).
    """
    adv = (logp_t - logp_s).detach()
    return -(adv * logp_s * mask.float()).sum() / max(num_tokens, 1.0)


# ---------------------------------------------------------------------------
# Prompt pools
# ---------------------------------------------------------------------------


def conversation_prompt(row: dict[str, Any]) -> SimpleNamespace | None:
    """A conversation → its prompt: the messages before the last assistant message.

    A conversation without an assistant message is used as it is. Returns None if nothing is left.
    """
    msgs = list(row.get("messages") or [])
    last = max((i for i, m in enumerate(msgs) if m.get("role") == "assistant"), default=None)
    if last is not None:
        msgs = msgs[:last]
    if not msgs:
        return None
    return SimpleNamespace(messages=msgs, tools=row.get("tools"))


def load_prompt_pool(t: OPDTeacher, oc: OPDConfig) -> list[Any]:
    """The prompt pool of one teacher. Each item has .messages and .tools (like a tool_env task)."""
    if t.prompts == "tool_env":
        from zero.post.envs.tool_env import generate_tasks

        return list(generate_tasks(oc.n_tool_tasks, seed=oc.env_seed, split="train"))
    rows = read_jsonl(t.prompts)
    if t.max_prompts > 0:
        rows = rows[: t.max_prompts]
    pool = [p for p in (conversation_prompt(r) for r in rows) if p is not None]
    if not pool:
        raise ValueError(f"[opd] teacher {t.name!r}: no usable prompt in {t.prompts}")
    return pool


def pick_prompts(
    pools: Sequence[Sequence[Any]], weights: Sequence[float], n: int, rng: random.Random
) -> list[tuple[int, int]]:
    """n draws of (teacher index, prompt index). The teacher of each draw follows the weights."""
    idx = rng.choices(range(len(pools)), weights=weights, k=n)
    return [(k, rng.randrange(len(pools[k]))) for k in idx]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def run_opd(
    src: str | os.PathLike | dict[str, Any],
    overrides: Sequence[str] | None = None,
    log: Callable[[str], None] = print,
) -> list[dict[str, Any]]:
    """CPU, 1 GPU, or N GPUs with torchrun (data parallel; see the docstring of `zero.post.common`)."""
    from zero.train.dist import cleanup, init_distributed

    cfg, sec = load_post_config(src, {"opd": OPDConfig}, overrides)
    info = init_distributed(cfg.train.device)
    try:
        return _opd_loop(cfg, sec["opd"], info, log)
    finally:
        cleanup()


def _opd_loop(
    cfg: Any, oc: OPDConfig, info: Any, log: Callable[[str], None]
) -> list[dict[str, Any]]:
    from zero.post.common import all_reduce_sum, rank_share
    from zero.post.grpo import sample_group
    from zero.train.trainer import autocast_context

    tc = cfg.train
    if not oc.teachers:
        raise ValueError("[opd] needs at least one [[opd.teachers]]")
    if oc.loss not in ("full_kl", "sampled"):
        raise ValueError(f'[opd] loss must be "full_kl" or "sampled", not {oc.loss!r}')
    if not tc.init_from:
        raise ValueError(
            "[opd] set train.init_from to the student checkpoint (usually the last RL stage)"
        )
    if oc.prompts_per_step < info.world_size:
        raise ValueError(
            f"[opd] prompts_per_step = {oc.prompts_per_step} < {info.world_size} processes"
        )
    set_threads(tc.cpu_threads)
    torch.manual_seed(tc.seed)
    device = info.device

    def ac():  # noqa: ANN202
        return autocast_context(device, tc.dtype)

    tok = Tokenizer.load(tc.data.tokenizer)
    model = build_model_from_init(cfg, device)

    teachers, pools = [], []
    for t in oc.teachers:
        t_model, t_tok = load_policy(t.path, device=device)
        if t_tok.hash() != tok.hash():
            raise ValueError(
                f"[opd] teacher {t.name!r} ({t.path}) has a different tokenizer from the student. "
                "On-policy distillation compares the two distributions token by token, so it needs the same vocabulary."
            )
        if t_model.config.vocab_size != model.config.vocab_size:
            raise ValueError(
                f"[opd] teacher {t.name!r}: vocab_size {t_model.config.vocab_size} != student {model.config.vocab_size}"
            )
        t_model.eval()
        for p in t_model.parameters():
            p.requires_grad_(False)
        teachers.append(t_model)
        pools.append(load_prompt_pool(t, oc))
    names = [t.name or f"t{i}" for i, t in enumerate(oc.teachers)]
    weights = [t.weight for t in oc.teachers]
    loop = LoopState(cfg, model, log, info=info)
    log(
        f"[opd] teachers {', '.join(f'{n} ({len(p)} prompts, w={w})' for n, p, w in zip(names, pools, weights))}; "
        f"{oc.prompts_per_step} prompts × {oc.samples_per_prompt} samples per step on {info.world_size} process(es), loss {oc.loss}"
    )

    limit = model.config.max_seq_len - 8
    while loop.step < tc.max_steps:
        t0 = time.perf_counter()
        lr = loop.begin_step()
        rng = random.Random(tc.seed * 7919 + loop.step)  # the same on every rank

        # 1-2. Prompts → student samples (this rank's prompts, seeded by their global index j)
        seqs: list[list[int]] = []
        masks: list[list[bool]] = []
        owner: list[int] = []
        n_eos = 0
        skipped = 0
        model.eval()
        for j, (k, pi) in rank_share(pick_prompts(pools, weights, oc.prompts_per_step, rng), info):
            task = pools[k][pi]
            ids, _ = render(task.messages, task.tools, add_generation_prompt=True, tokenizer=tok)
            if len(ids) >= limit:
                skipped += 1  # too long for the context
                continue
            p_ids, resps = sample_group(
                model,
                tok,
                task,
                oc.samples_per_prompt,
                oc.max_new_tokens,
                oc.temperature,
                oc.top_p,
                seed=tc.seed * 100_003 + loop.step * 101 + j,
            )
            for r in resps:
                if not r:
                    continue
                n_eos += int(r[-1] == tok.im_end_id)
                seqs.append(p_ids + r)
                masks.append([False] * len(p_ids) + [True] * len(r))
                owner.append(k)
        t_gen = time.perf_counter() - t0
        (n_tok,) = all_reduce_sum([float(sum(sum(m) for m in masks))], info)  # global normalizer
        model.train()

        # 3. Loss per teacher (each sequence is scored by the teacher of its prompt), token-level mean
        loss_total = 0.0
        kl_sum = [0.0] * len(teachers)
        tok_cnt = [0.0] * len(teachers)
        for k, t_model in enumerate(teachers):
            mine = [i for i, o in enumerate(owner) if o == k]
            for c in range(0, len(mine), oc.forward_batch):
                sel = mine[c : c + oc.forward_batch]
                ids, mask = pad_batch(
                    [seqs[i] for i in sel], [masks[i] for i in sel], tok.eot_id, device
                )
                tmask = mask[:, 1:]
                with ac():
                    s_logits = model(ids[:, :-1])
                    with torch.no_grad():
                        t_logits = t_model(ids[:, :-1])
                # Only the response positions: the prompts (tool schemas, history) can be much longer
                # than the responses, and each position costs a full row of vocabulary logits.
                s_r, t_r = s_logits[tmask], t_logits[tmask]  # (N, V)
                y = ids[:, 1:][tmask]  # (N,)
                lp_s = token_logprobs(s_r.unsqueeze(0), y.unsqueeze(0))[0]
                lp_t = token_logprobs(t_r.unsqueeze(0), y.unsqueeze(0))[0]
                c_tok = float(y.numel())
                ones = torch.ones_like(y, dtype=torch.bool)
                if oc.loss == "full_kl":
                    loss = reverse_kl_loss(s_r.unsqueeze(0), t_r.unsqueeze(0), ones.unsqueeze(0))
                    loss = loss * (c_tok / max(n_tok, 1.0))
                else:
                    loss = sampled_opd_loss(lp_s, lp_t, ones, n_tok)
                loss.backward()
                loss_total += float(loss.detach())
                kl_sum[k] += float((lp_s.detach() - lp_t).sum())
                tok_cnt[k] += c_tok
        gnorm = loop.end_step()

        nt = len(teachers)
        sums = all_reduce_sum([loss_total, n_eos, skipped, len(seqs), *kl_sum, *tok_cnt], info)
        loss_g, eos_g, skip_g, nseq_g = sums[:4]
        kl_g, cnt_g = sums[4 : 4 + nt], sums[4 + nt :]
        rec: dict[str, Any] = {
            "loss": loss_g,
            "kl": sum(kl_g) / max(sum(cnt_g), 1.0),
            **{f"kl/{n}": kl_g[k] / cnt_g[k] for k, n in enumerate(names) if cnt_g[k] > 0},
            "resp_len": n_tok / max(nseq_g, 1),
            "eos_rate": eos_g / max(nseq_g, 1),
            "n_seqs": int(nseq_g),
            "skipped": int(skip_g),
            "lr": lr,
            "grad_norm": gnorm,
            "gen_s": t_gen,
            "step_s": time.perf_counter() - t0,
        }
        loop.record(
            rec,
            "step {step:>4} | kl {kl:.4f} | len {resp_len:.1f} | eos {eos_rate:.2f} | loss {loss:+.4f} | {step_s:.1f}s",
        )
    if info.is_main:
        Path(tc.out_dir).mkdir(parents=True, exist_ok=True)
        Path(tc.out_dir, "opd_summary.json").write_text(
            json.dumps(
                {"config": asdict(oc), "history": loop.history},
                ensure_ascii=False,
                indent=2,
                default=str,
            )
        )
    return loop.history


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Cross-stage on-policy distillation (Chapter 17)")
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = ap.parse_args(argv)
    run_opd(args.config, args.set)


if __name__ == "__main__":
    main()
