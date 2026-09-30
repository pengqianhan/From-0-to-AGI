"""BFCL 适配层：用官方 Berkeley Function Calling Leaderboard 评测我们导出的模型（对应第 11、20 章）。

**状态：尚未验证。** 这台机器没有 GPU，也下载不了评测数据以外的模型；下面的代码按 `bfcl-eval`
2026.3.23 版（PyPI）的源码写成，第二步在 GPU 机器上第一次跑通后再去掉这句话。

为什么需要适配：BFCL 的本地模型走"prompt 模式"——评测框架自己拼提示词（`_format_prompt`），
再调 vLLM / SGLang 的 `/v1/completions`。它内置的 `QwenFCHandler` 手写了一份 Qwen3 模板，关闭思考时
会在生成提示后塞空的 `<think>\\n\\n</think>\\n\\n`，而我们的模型没见过思考格式。所以这里注册一个
`ZeroFCHandler`：继承 `QwenFCHandler`（复用它解析 `<tool_call>` 的逻辑），只把拼提示词换成
**导出目录里的 chat_template**（`tokenizer.apply_chat_template`），保证评测时的提示词与训练时逐字一致。

用法（第二步，GPU 机器上）：

    uv pip install bfcl-eval==2026.3.23 vllm            # 版本在 eval/PREREGISTRATION.md 冻结
    # 1) 生成（框架自动起 vLLM 服务；或者先自己起服务，再加 --skip-server-setup）
    uv run python -m zero.eval.bfcl generate --hf-dir out/main/hf_final --name zero-0.7b-FC \\
        --test-category simple_python,multiple,parallel,parallel_multiple,irrelevance,live,multi_turn \\
        --backend vllm --num-gpus 1
    # 2) 判分
    uv run python -m zero.eval.bfcl evaluate --hf-dir out/main/hf_final --name zero-0.7b-FC \\
        --test-category simple_python,multiple,parallel,parallel_multiple,irrelevance,live,multi_turn
    # 3) 读分数（BFCL 写在 $BFCL_PROJECT_ROOT/score/ 下的 CSV），转成我们的报告格式
    uv run python -m zero.eval.bfcl collect --score-dir $BFCL_PROJECT_ROOT/score --name zero-0.7b-FC

对手模型（Qwen3.5-0.8B 等）直接用 BFCL 内置的 handler 名（各自官方模板），同一版本、同样的解码参数
（BFCL 默认 temperature=0.001）。逐题结果在 `$BFCL_PROJECT_ROOT/score/<model>/` 里，
配对 bootstrap 用 `zero.eval.bootstrap.paired_bootstrap`。

待核实（第二步第一次运行时逐条确认）：
- `local_inference_model_map` / `MODEL_CONFIG_MAPPING` 两个注册表的名字与 `ModelConfig` 字段；
- `_format_prompt(messages, function)` 里 `function` 的结构（这里包成 `{"type": "function", "function": f}`）；
- 分数 CSV 的文件名与列名（`collect` 只做了宽松解析）。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any


def bfcl_available() -> bool:
    try:
        import bfcl_eval  # noqa: F401
    except ImportError:
        return False
    return True


def make_handler_class():  # noqa: ANN201
    """返回 ZeroFCHandler 类（需要已安装 bfcl-eval）。尚未验证。"""
    from bfcl_eval.model_handler.local_inference.qwen_fc import QwenFCHandler

    class ZeroFCHandler(QwenFCHandler):
        def _format_prompt(
            self, messages: list[dict[str, Any]], function: list[dict[str, Any]]
        ) -> str:
            tools = [
                f if "function" in f else {"type": "function", "function": f} for f in function
            ]
            return self.tokenizer.apply_chat_template(
                messages, tools=tools or None, add_generation_prompt=True, tokenize=False
            )

    return ZeroFCHandler


def register_zero_model(name: str, hf_dir: str | os.PathLike, display_name: str = "") -> None:
    """把我们的模型登记进 BFCL 的模型表（进程内生效）。尚未验证。"""
    from bfcl_eval.constants import model_config as mc

    cfg = mc.ModelConfig(
        model_name=str(hf_dir),
        display_name=display_name or f"{name} (FC)",
        url="https://github.com/（发布后填写）",
        org="From-0-to-AGI",
        license="apache-2.0",
        model_handler=make_handler_class(),
        input_price=None,
        output_price=None,
        is_fc_model=True,
        underscore_to_dot=False,
    )
    mc.local_inference_model_map[name] = cfg
    mc.MODEL_CONFIG_MAPPING[name] = cfg


def run_cli(cmd: str, hf_dir: str, name: str, extra: list[str]) -> None:
    """注册模型后，调用 bfcl 的命令行入口（typer app）。尚未验证。"""
    if not bfcl_available():
        raise SystemExit("没有安装 bfcl-eval：uv pip install bfcl-eval==<预注册冻结的版本>")
    register_zero_model(name, hf_dir)
    from bfcl_eval.__main__ import cli

    argv = [cmd, "--model", name, *extra]
    if cmd == "generate":
        argv += ["--local-model-path", str(hf_dir)]
    sys.argv = ["bfcl", *argv]
    cli()


def collect_scores(score_dir: str | os.PathLike, name: str) -> dict[str, Any]:
    """从 BFCL 的分数目录里找出 name 这一行（CSV），返回 {列名: 值}。宽松解析，待核实。"""
    out: dict[str, Any] = {}
    for p in sorted(Path(score_dir).rglob("*.csv")):
        with open(p, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if any(name in str(v) for v in row.values()):
                    out[p.stem] = row
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="BFCL 适配（第二步用，尚未验证）")
    ap.add_argument("command", choices=["generate", "evaluate", "collect"])
    ap.add_argument("--hf-dir", default="")
    ap.add_argument("--name", default="zero-FC")
    ap.add_argument("--score-dir", default="")
    args, extra = ap.parse_known_args(argv)
    if args.command == "collect":
        print(json.dumps(collect_scores(args.score_dir, args.name), ensure_ascii=False, indent=2))
        return
    run_cli(args.command, args.hf_dir, args.name, extra)


if __name__ == "__main__":
    main()
