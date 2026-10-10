"""Offline difficulty filter for RL tasks: keep the tasks that the policy solves sometimes (Chapter 19).

    uv run torchrun --standalone --nproc_per_node=8 -m zero.post.difficulty \\
        --policy out/main/distill/ckpt --tasks data/rl/fc_train.jsonl --k 8 \\
        --out data/rl/fc_train_filtered.jsonl

**Why.** GRPO learns from the differences inside a group. A task that the policy always solves, or
never solves, gives every sample the same reward: all advantages are 0 and the task costs sampling
time for no gradient (`zero_std_groups` in the GRPO log). Kimi K2 keeps only medium tasks by the
pass@k of the SFT model; OLMo 3 filters zero-gradient groups; Qwen3 keeps queries that are
"learnable for the cold-start model" (Chapter 19, "Adopters and sources"). This tool is the offline
form: run it once with the checkpoint that starts RL, before the GRPO stage. (Online filtering inside
the GRPO loop is not implemented.)

**What it does.** For each task: k samples from the policy (temperature 1.0 by default), each scored
with the verifier of the task (`fc_tasks.score_any`). pass = number of exact answers / k. A task is
kept if `min_pass <= pass <= max_pass` (default: at least one exact and at least one wrong answer).
`--keep-hard` keeps that fraction of the all-wrong tasks anyway (useful for a weak base: they become
learnable after some RL; chosen by a fixed seed).

**Outputs.** `--out`: the kept tasks (same format as the input). `<out>.stats.jsonl`: per task id,
pass, mean reward. `<out>.meta.json`: counts, the histogram of pass over all tasks, the settings.

**Data parallel.** With torchrun, each rank scores tasks[rank::world]; samples are seeded by the
global task index, so the result does not depend on the number of processes. Rank 0 merges.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import torch

from zero.post.common import rank_share


def score_task(
    model: torch.nn.Module,
    tok: Any,
    task: Any,
    k: int,
    max_new_tokens: int,
    temperature: float,
    seed: int,
) -> dict[str, Any]:
    from zero.post.envs.fc_tasks import score_any
    from zero.post.grpo import sample_group

    _, resps = sample_group(model, tok, task, k, max_new_tokens, temperature, 1.0, seed)
    rewards = [score_any(task, tok.decode([t for t in r if t != tok.im_end_id])) for r in resps]
    n_exact = sum(r.exact for r in rewards)
    return {
        "id": task.id,
        "pass": n_exact / k,
        "n_exact": n_exact,
        "mean_reward": sum(r.total for r in rewards) / k,
    }


def keep_decision(
    stats: Sequence[dict[str, Any]], min_pass: float, max_pass: float, keep_hard: float, seed: int
) -> tuple[set[str], Counter]:
    """The ids to keep, and the counts by reason."""
    rng = random.Random(seed)
    keep, why = set(), Counter()
    for s in sorted(stats, key=lambda x: x["id"]):
        p = s["pass"]
        if min_pass <= p <= max_pass:
            keep.add(s["id"])
            why["kept"] += 1
        elif p > max_pass:
            why["too_easy"] += 1
        elif p == 0 and rng.random() < keep_hard:
            keep.add(s["id"])
            why["kept_hard"] += 1
        else:
            why["too_hard"] += 1
    return keep, why


def run_filter(
    policy: str,
    tasks_path: str,
    out: str,
    k: int = 8,
    min_pass: float | None = None,
    max_pass: float | None = None,
    keep_hard: float = 0.0,
    max_new_tokens: int = 512,
    temperature: float = 1.0,
    seed: int = 0,
    device: str = "auto",
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    from zero.post.common import load_policy
    from zero.post.envs.fc_tasks import load_fc_tasks
    from zero.train.dist import barrier, cleanup, init_distributed

    lo = 1.0 / k if min_pass is None else min_pass  # at least one exact answer
    hi = (k - 1) / k if max_pass is None else max_pass  # at least one wrong answer
    info = init_distributed(device)
    try:
        model, tok = load_policy(policy, device=info.device)
        model.eval()
        tasks = load_fc_tasks(tasks_path)
        part = Path(f"{out}.part{info.rank}")
        part.parent.mkdir(parents=True, exist_ok=True)
        with open(part, "w", encoding="utf-8") as f:
            for n, (j, t) in enumerate(rank_share(tasks, info)):
                s = score_task(model, tok, t, k, max_new_tokens, temperature, seed * 1_000_003 + j)
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
                if info.is_main and (n + 1) % 100 == 0:
                    log(f"[difficulty] rank 0: {n + 1} tasks scored")
        barrier()
        meta: dict[str, Any] = {}
        if info.is_main:
            stats = []
            for r in range(info.world_size):
                p = Path(f"{out}.part{r}")
                stats += [json.loads(x) for x in p.read_text("utf-8").splitlines() if x.strip()]
                p.unlink()
            keep, why = keep_decision(stats, lo, hi, keep_hard, seed)
            with open(out, "w", encoding="utf-8") as f:
                for t in tasks:
                    if t.id in keep:
                        f.write(json.dumps(t.to_dict(), ensure_ascii=False) + "\n")
            by_id = {s["id"]: s for s in stats}
            with open(f"{out}.stats.jsonl", "w", encoding="utf-8") as f:
                for t in tasks:
                    f.write(json.dumps(by_id[t.id], ensure_ascii=False) + "\n")
            hist = Counter(s["n_exact"] for s in stats)
            meta = {
                "n_tasks": len(tasks),
                **dict(why),
                "pass_histogram": {f"{i}/{k}": hist.get(i, 0) for i in range(k + 1)},
                "mean_pass": sum(s["pass"] for s in stats) / max(len(stats), 1),
                "settings": {
                    "policy": policy,
                    "tasks": tasks_path,
                    "k": k,
                    "min_pass": lo,
                    "max_pass": hi,
                    "keep_hard": keep_hard,
                    "temperature": temperature,
                    "max_new_tokens": max_new_tokens,
                    "seed": seed,
                    "world_size": info.world_size,
                },
            }
            Path(f"{out}.meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
            log(
                f"[difficulty] {json.dumps({k_: v for k_, v in meta.items() if k_ != 'settings'}, ensure_ascii=False)}"
            )
        barrier()
        return meta
    finally:
        cleanup()


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Offline difficulty filter for RL tasks (Chapter 19)")
    ap.add_argument(
        "--policy", required=True, help="zero checkpoint or HF folder (the model that starts RL)"
    )
    ap.add_argument("--tasks", required=True, help="task file (zero/post/envs/fc_tasks.py format)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument(
        "--min-pass", type=float, default=None, help="default 1/k: at least one exact answer"
    )
    ap.add_argument(
        "--max-pass", type=float, default=None, help="default (k-1)/k: at least one wrong answer"
    )
    ap.add_argument(
        "--keep-hard", type=float, default=0.0, help="fraction of all-wrong tasks to keep"
    )
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    a = ap.parse_args(argv)
    run_filter(
        a.policy,
        a.tasks,
        a.out,
        a.k,
        a.min_pass,
        a.max_pass,
        a.keep_hard,
        a.max_new_tokens,
        a.temperature,
        a.seed,
        a.device,
    )


if __name__ == "__main__":
    main()
