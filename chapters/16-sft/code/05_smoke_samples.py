"""Chapter 16 · Tiny-configuration demo: what does the tiny model (about 1.3M parameters) write
after SFT in the smoke test?

The script reads out/smoke/sft/ckpt, which `uv run python -m zero.smoke` writes. It generates
greedily on the first 6 questions of the fixed dev set and prints each output next to the reference
answer. Then it prints the SFT metrics that the smoke test recorded. If out/smoke does not exist,
the script tells you to run the smoke test first (15 minutes or more).

This is a **tiny-configuration demo**: it only shows that the production code path works.
It does not show any result of the main-line model.
Run: uv run python chapters/16-sft/code/05_smoke_samples.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

torch.set_num_threads(1)  # the build machine shares its CPU between many jobs; on your computer, you can remove this line
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
SMOKE = ROOT / "out" / "smoke"


def smoke_numbers() -> dict | None:
    p = SMOKE / "summary.json"
    if not p.exists():
        return None
    stages = {s["stage"]: s for s in json.loads(p.read_text())["stages"]}
    out = {"sft": stages.get("sft"), "eval": None, "data": None}
    ev = SMOKE / "eval" / "results.json"
    if ev.exists():
        t = json.loads(ev.read_text())["results"]["sft"]["tool_dev"]
        out["eval"] = {k: t[k] for k in ("n", "format_ok", "call_exact", "ast_match", "answer_ok")}
    meta = SMOKE / "sft" / "data" / "train.json"
    if meta.exists():
        out["data"] = json.loads(meta.read_text())
    return out


def samples(n: int = 6) -> list[dict]:
    from zero.post.common import chat_complete, load_policy
    from zero.post.envs.tool_env import dev_tasks

    model, tok = load_policy(SMOKE / "sft" / "ckpt")
    model.eval()
    rows = []
    for t in dev_tasks(30)[:n]:
        out = chat_complete(model, tok, t.messages, t.tools, max_new_tokens=96)[0]
        rows.append({"q": t.messages[-1]["content"],
                     "gold": [{"name": c["name"], "arguments": c["arguments"]} for c in t.gold_calls],
                     "out": out})
    return rows


if __name__ == "__main__":
    nums = smoke_numbers()
    if nums is None or not (SMOKE / "sft" / "ckpt").exists():
        print("out/smoke/ not found. First run `uv run python -m zero.smoke` (15 minutes or more on a CPU).")
        raise SystemExit(0)
    print("[Tiny-configuration demo] SFT stage of the smoke test:", nums["sft"])
    print("Packing statistics:", nums["data"])
    print("Tool calls on the dev set (30 questions):", nums["eval"])
    print()
    for r in samples():
        print("Q:", r["q"])
        print("  reference:", json.dumps(r["gold"], ensure_ascii=False))
        print("  model:", repr(r["out"]))
