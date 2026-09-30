"""第 20 章 · 极简代码 4：从评测结果 JSON 生成模型卡（model card）

模型卡是模型仓库首页的 README.md：开头一段 YAML 元数据（许可证、语言、数据集、标签……，
Hugging Face 用它做检索和展示），后面是给人看的正文。GOAL.md 3.5 规定我们的模型卡必须写：
训练数据与许可证、每个阶段的配方与花费、预注册协议、全部评测结果（包括落后的项）、去污染检查、已知局限。

这个脚本的原则：**只填有来源的数**。评测表直接从结果 JSON 生成（格式与 zero/eval/harness.py 的
results.json 相同：results[模型][任务] = {指标: 值, n: 题数}，comparisons = 配对 bootstrap 的判定）；
没有数据的地方一律写"待训练"，而不是留空或估一个数。

    uv run python chapters/20-release/code/04_model_card.py                 # 默认读 out/smoke（极小配置演示）
    uv run python chapters/20-release/code/04_model_card.py --results path/to/results.json --out card.md

out/smoke 由 `uv run python -m zero.smoke` 生成；没有这个目录时，脚本输出一张全是"待训练"的骨架。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TODO = "待训练"

# 元数据：发布时逐项填写；None 表示还没定，卡片里显示"待定"
META = {
    "name": "zero-0.7b（暂名）",
    "license": None,  # 权重许可证：作者决定（见正文"许可证"一节的选项）
    "language": ["zh", "en"],
    "library_name": "transformers",
    "pipeline_tag": "text-generation",
    "tags": ["function-calling", "tool-use", "from-scratch", "gguf"],
    "datasets": [],   # 发布时填 HF 数据集 id，例如预训练用到的 FineWeb-Edu
    "architecture": "Qwen3ForCausalLM 兼容的稠密 Transformer（Pre-Norm RMSNorm、SwiGLU、RoPE、GQA、QK-Norm、共享 embedding）",
    "params": "689.5M（configs/main/pretrain.toml 暂定形状）",
}

STAGES = ["预训练", "中期训练", "长上下文", "SFT", "蒸馏", "DPO", "GRPO"]


def yaml_front_matter(meta: dict) -> str:
    lines = ["---", f"license: {meta['license'] or 'other  # 待定'}"]
    for key in ("language", "tags", "datasets"):
        if meta[key]:
            lines.append(f"{key}:")
            lines += [f"- {v}" for v in meta[key]]
    lines += [f"library_name: {meta['library_name']}", f"pipeline_tag: {meta['pipeline_tag']}", "---"]
    return "\n".join(lines)


def results_table(results: dict) -> str:
    cols: list[tuple[str, str]] = []
    for per_task in results.values():
        for task, r in per_task.items():
            for k, v in r.items():
                if isinstance(v, (int, float)) and k != "n" and (task, k) not in cols:
                    cols.append((task, k))
    out = ["| 模型 | " + " | ".join(f"{t} / {m}" for t, m in cols) + " |",
           "|---|" + "---:|" * len(cols)]
    for model, per_task in results.items():
        cells = [f"{per_task.get(t, {}).get(m, float('nan')):.3f}" for t, m in cols]
        out.append(f"| {model} | " + " | ".join(cells) + " |")
    ns = sorted({f"{t} n={r.get('n')}" for pt in results.values() for t, r in pt.items()})
    return "\n".join(out) + "\n\n题数：" + "，".join(ns)


def comparison_table(comps: list[dict]) -> str:
    out = ["| 我们 | 对手 | 基准 | 指标 | 我们 | 对手 | 差值 | 95% CI | 判定 |",
           "|---|---|---|---|---:|---:|---:|---|---|"]
    for c in comps:
        out.append(f"| {c['model']} | {c['baseline']} | {c['task']} | {c['metric']} | {c['mean_a']:.3f} | "
                   f"{c['mean_b']:.3f} | {c['diff']:+.3f} | [{c['ci_low']:+.3f}, {c['ci_high']:+.3f}] | "
                   f"**{c['decision']}** |")
    tally = {d: sum(c["decision"] == d for c in comps) for d in ("超过", "持平", "落后")}
    out.append("")
    out.append(f"合计：超过 {tally['超过']} 项、持平 {tally['持平']} 项、落后 {tally['落后']} 项（全部列出，不挑选）。")
    return "\n".join(out)


def stage_table(summary: dict | None) -> str:
    rows = ["| 阶段 | token / 步数 | 关键指标 | 花费 |", "|---|---|---|---|"]
    by_name = {s["stage"]: s for s in (summary or {}).get("stages", [])}
    alias = {"预训练": "pretrain", "中期训练": "midtrain", "SFT": "sft", "蒸馏": "distill",
             "DPO": "dpo", "GRPO": "grpo"}
    for st in STAGES:
        s = by_name.get(alias.get(st, ""))
        if s is None:
            rows.append(f"| {st} | {TODO} | {TODO} | {TODO} |")
            continue
        metrics = "；".join(f"{k}={v}" for k, v in s.items()
                           if k not in ("stage", "status", "seconds", "steps"))
        rows.append(f"| {st} | {s.get('steps', '—')} 步 | {metrics} | CPU {s['seconds']:.0f}s，$0 |")
    return "\n".join(rows)


def build_card(results_json: dict | None, summary: dict | None, demo: bool) -> str:
    parts = [yaml_front_matter(META), "", f"# {META['name']}", ""]
    if demo:
        parts += ["> ⚠️ **极小配置演示**：下面的数字来自 CPU 上约 1.3M 参数的冒烟测试（`zero.smoke`），",
                  "> 只说明流水线是通的，**不是主线模型的结果**。", ""]
    parts += [
        "## 模型概要", "",
        f"- 架构：{META['architecture']}",
        f"- 参数量：{META['params']}",
        "- 用途：中英双语、工具调用（function calling）的小模型；可在笔记本上用 llama.cpp / Ollama 运行",
        f"- 权重许可证：{META['license'] or '待定（作者决定）'}",
        "- 发布内容：Base、SFT、最终版，以及关键中间 checkpoint（HF safetensors）；GGUF（Q8_0、Q4_K_M）",
        "",
        "## 使用方法", "",
        "```python",
        "from transformers import AutoModelForCausalLM, AutoTokenizer",
        "tok = AutoTokenizer.from_pretrained(\"<仓库名>\")",
        "model = AutoModelForCausalLM.from_pretrained(\"<仓库名>\")",
        "ids = tok.apply_chat_template(messages, tools=tools, add_generation_prompt=True, return_tensors=\"pt\")",
        "```",
        "",
        "```bash",
        "llama-cli -m zero-Q4_K_M.gguf          # llama.cpp；Ollama 可直接加载同一个 GGUF",
        "vllm serve <仓库名> --enable-auto-tool-choice --tool-call-parser hermes",
        "```",
        "",
        "## 训练数据与许可证", "",
        "| 数据集 | 用在哪个阶段 | token 数 | 许可证 | 署名要求 |", "|---|---|---|---|---|",
        f"| {TODO} | {TODO} | {TODO} | {TODO} | {TODO} |",
        "",
        "教师模型（蒸馏）：名称、版本、许可证是否允许用输出训练其他模型——" + TODO + "。",
        "",
        "## 各阶段配方与花费", "",
        stage_table(summary), "",
        "完整配置见 `configs/main/`，花费明细见 `runs/ledger.md`。", "",
        "## 预注册", "",
        "评测协议在训练前登记：`eval/PREREGISTRATION.md`（登记 commit：" + TODO + "）。",
        "判定规则：配对 bootstrap 95% 置信区间整体 > 0 为超过，整体 < 0 为落后，跨过 0 为持平。", "",
        "## 评测结果（全部列出，包括落后的项）", "",
    ]
    if results_json:
        parts += [results_table(results_json["results"]), "",
                  "### 配对比较", "", comparison_table(results_json.get("comparisons", [])), ""]
    else:
        parts += [f"{TODO}（BFCL、中文工具调用基准、通用基准；每个对手、每个基准一行）", ""]
    parts += [
        "官方公布的对手分数并列展示，但不作为比较依据。", "",
        "### 发布后新增对手", "",
        f"冻结日期之后发布的同尺寸模型：{TODO}（即使它们比我们强，也列在这里）。", "",
        "## 去污染检查", "",
        f"- 13-gram 重叠：训练数据（含教师合成数据）vs 全部评测集，命中率 {TODO}",
        f"- 工具函数名 / 参数 schema 与 BFCL 等评测集的重合：剔除 {TODO} 个", "",
        "## 已知局限", "",
        "- 参数少，知识量有限，事实类问题会编造；",
        "- 工具调用只在预注册的基准和我们自己的环境里评测过，真实场景的工具与参数分布可能不同；",
        f"- 其他：{TODO}", "",
        "## 引用与致谢", "",
        "代码与课程：<https://github.com/…/From-0-to-AGI>（" + TODO + "）",
    ]
    return "\n".join(parts) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(ROOT / "out/smoke/eval/results.json"))
    ap.add_argument("--summary", default=str(ROOT / "out/smoke/summary.json"))
    ap.add_argument("--out", default="", help="写到文件；默认打印到终端")
    args = ap.parse_args()
    res_p, sum_p = Path(args.results), Path(args.summary)
    results = json.loads(res_p.read_text(encoding="utf-8")) if res_p.exists() else None
    summary = json.loads(sum_p.read_text(encoding="utf-8")) if sum_p.exists() else None
    demo = "smoke" in str(res_p) and results is not None
    card = build_card(results, summary, demo)
    if args.out:
        Path(args.out).write_text(card, encoding="utf-8")
        print(f"模型卡 → {args.out}")
    else:
        print(card)
    n_todo = card.count(TODO)
    print(f"<!-- 还有 {n_todo} 处'{TODO}'，发布前必须全部填上 -->")


if __name__ == "__main__":
    main()
