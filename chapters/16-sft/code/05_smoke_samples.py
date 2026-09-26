"""第 16 章 · 极小配置演示：看看冒烟测试里 SFT 过的 tiny 模型（约 1.3M 参数）到底写出了什么

读取 `uv run python -m zero.smoke` 留下的 out/smoke/sft/ckpt，在固定 dev 集的前 6 道题上贪心生成，
和标准答案并排打印；再打印冒烟测试记录的 SFT 指标。没有 out/smoke 就提示先跑冒烟测试（约 15 分钟以上）。

这是**极小配置演示**：只说明生产级代码通路是通的，不代表主线模型的任何结果。
运行：uv run python chapters/16-sft/code/05_smoke_samples.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

torch.set_num_threads(1)  # 构建环境多任务共享 CPU；读者本机可以删掉这行
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
        print("没有找到 out/smoke/：先运行 `uv run python -m zero.smoke`（CPU 上约 15 分钟以上）")
        raise SystemExit(0)
    print("【极小配置演示】冒烟测试的 SFT 阶段：", nums["sft"])
    print("打包统计：", nums["data"])
    print("dev 集工具调用（30 题）：", nums["eval"])
    print()
    for r in samples():
        print("问：", r["q"])
        print("  标准：", json.dumps(r["gold"], ensure_ascii=False))
        print("  模型：", repr(r["out"]))
