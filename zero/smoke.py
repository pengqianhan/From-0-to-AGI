"""端到端冒烟测试：一条命令在 CPU 上把整条流水线跑通（对应全书；GOAL.md 9.1）。

    uv run python -m zero.smoke                     # 输出到 out/smoke（先清空）
    uv run python -m zero.smoke --out /tmp/smoke    # 换目录

阶段（全部用 configs/tiny/*.toml，路径里的 out/tiny 换成 --out 目录，另有几处为了速度的覆盖，见下）：

 1. data：生成工具调用格式的文本（tool_env 的标准解答对话）
 2. tokenizer：在 tiny_corpus（莎士比亚 / 唐诗 / 代码）+ 工具调用文本上训练 byte-level BPE
    （让分词器见过 JSON 和对话格式，一条带工具说明的对话从约 700 token 降到约 330 token）
 3. pretrain：configs/tiny/pretrain.toml
 4. midtrain：configs/tiny/midtrain.toml，数据混合里加入工具调用格式数据（GOAL.md 3.3 的中期训练）
 5. SFT：configs/tiny/sft.toml（seq_len 1024 → 512）
 6. distill：configs/tiny/distill.toml（替身教师 = tiny SFT 模型自己，见 zero/post/distill.py 的说明）
 7. DPO：configs/tiny/dpo.toml
 8. GRPO：configs/tiny/grpo.toml
 9. eval：configs/tiny/eval.toml（玩具集 + 工具调用 dev 集，SFT / DPO / GRPO 三个模型，配对 bootstrap）
10. export：导出 HF 目录（transformers 加载对拍 logits、apply_chat_template 与我们的模板逐字一致）
    + GGUF（llama.cpp 官方转换脚本；已编译 llama.cpp 时再量化成 Q8_0 并用 llama-simple 生成几个 token）
11. demo：本地工具调用助手问一个问题

最后打印每个阶段的耗时和关键指标（loss / 奖励），并写 `<out>/SUMMARY.md` 与 `summary.json`。
tiny 模型只有约 1.3M 参数，所有分数都只说明"代码通路是通的"，不说明任何方法的效果。
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

REPO = Path(__file__).resolve().parent.parent
CONFIGS = REPO / "configs" / "tiny"


def _rewrite(obj: Any, out: str) -> Any:
    """把配置里所有以 out/tiny 开头的路径换成 out。"""
    if isinstance(obj, dict):
        return {k: _rewrite(v, out) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_rewrite(v, out) for v in obj]
    if isinstance(obj, str) and obj.startswith("out/tiny"):
        return out + obj[len("out/tiny") :]
    return obj


def load_tiny(name: str, out: str) -> dict[str, Any]:
    from zero.config import read_toml

    d = _rewrite(read_toml(CONFIGS / f"{name}.toml"), out)
    # 相对仓库根目录的输入路径（assets/…）改成绝对路径，这样可以在任何目录下运行
    if "data" in d and "prepare" in d["data"]:
        d["data"]["prepare"]["raw_files"] = [
            str(REPO / p) for p in d["data"]["prepare"]["raw_files"]
        ]
    return d


def _set_seq(d: dict[str, Any], seq_len: int, factor: float) -> None:
    d["data"]["seq_len"] = seq_len
    d["model"]["max_seq_len"] = seq_len
    d["model"]["rope_scaling"]["factor"] = factor


class Smoke:
    def __init__(self, out: Path, log: Callable[[str], None] = print) -> None:
        self.out = out
        self.o = str(out)
        self.log = log
        self.rows: list[dict[str, Any]] = []

    def stage(self, name: str, fn: Callable[[], dict[str, Any]]) -> dict[str, Any]:
        self.log(f"\n{'=' * 20} [{len(self.rows) + 1}] {name} {'=' * 20}")
        t0 = time.perf_counter()
        try:
            info = fn()
            status = "ok"
        except Exception as e:  # noqa: BLE001 - 冒烟测试要把失败记进表里，而不是直接崩掉
            traceback.print_exc()
            info = {"error": f"{type(e).__name__}: {e}"[:300]}
            status = "FAILED"
        dt = time.perf_counter() - t0
        row = {"stage": name, "status": status, "seconds": round(dt, 1), **info}
        self.rows.append(row)
        self.log(
            f"[smoke] {name}: {status}，{dt:.1f}s，{ {k: v for k, v in info.items() if k != 'error'} }"
        )
        if status != "ok":
            raise RuntimeError(f"阶段 {name} 失败：{info.get('error')}")
        return info

    # ---- 各阶段 ----
    def data_and_tokenizer(self) -> dict[str, Any]:
        from zero.post.chat import render_text
        from zero.post.envs.tool_env import generate_tasks, reference_messages
        from zero.tokenizer import train_bpe

        tasks = generate_tasks(1000, seed=11, split="train")
        convs = [render_text(reference_messages(t), t.tools) for t in tasks]
        corpus = self.out / "corpus"
        corpus.mkdir(parents=True, exist_ok=True)
        (corpus / "toolcall.txt").write_text("\n\n".join(convs), encoding="utf-8")
        (corpus / "toolcall.jsonl").write_text(
            "\n".join(json.dumps({"text": c}, ensure_ascii=False) for c in convs), encoding="utf-8"
        )
        files = sorted((REPO / "assets" / "tiny_corpus").glob("*.txt")) + [corpus / "toolcall.txt"]
        tok = train_bpe(files, vocab_size=2048)
        tok.save(self.out / "tokenizer.json")
        n = [len(tok.encode(c)) for c in convs[:100]]
        return {
            "toolcall_convs": len(convs),
            "tokens_per_conv": round(sum(n) / len(n), 1),
            "vocab": tok.vocab_size,
        }

    def pretrain(self) -> dict[str, Any]:
        from zero.config import config_from_dict
        from zero.train.trainer import run_training

        d = load_tiny("pretrain", self.o)
        cfg = config_from_dict(d)
        hist = run_training(cfg, log=self.log)
        return {
            "loss": round(hist[-1]["loss"], 4),
            "val_loss": _r(hist[-1].get("val_loss")),
            "steps": hist[-1]["step"],
        }

    def midtrain(self) -> dict[str, Any]:
        from zero.config import config_from_dict
        from zero.data.shard import write_shards
        from zero.tokenizer import Tokenizer
        from zero.train.midtrain import check_compatible
        from zero.train.trainer import run_training

        tok = Tokenizer.load(self.out / "tokenizer.json")
        rows = [
            json.loads(x) for x in (self.out / "corpus" / "toolcall.jsonl").read_text().splitlines()
        ]
        # 对话文本里的 <|im_start|> 等要保留成特殊 token，所以不走 clean（clean 会去掉特殊 token 字面量）
        write_shards(
            [r["text"] for r in rows], tok, self.out / "data", "toolcall_train", source="tool_env"
        )
        d = load_tiny("midtrain", self.o)
        d["data"]["sources"].append(
            {"name": "toolcall", "path": f"{self.o}/data/toolcall_train_*.bin", "weight": 0.4}
        )
        cfg = config_from_dict(d)
        for c in check_compatible(cfg):
            self.log(f"[midtrain] {c}")
        hist = run_training(cfg, log=self.log)
        return {
            "loss": round(hist[-1]["loss"], 4),
            "val_loss": _r(hist[-1].get("val_loss")),
            "steps": hist[-1]["step"],
        }

    def sft(self) -> dict[str, Any]:
        from zero.post.sft import run_sft

        d = load_tiny("sft", self.o)
        _set_seq(d, 512, 4.0)
        d["train"]["micro_batch_size"] = 8
        d["train"]["max_steps"] = 240
        d["train"]["eval_every"] = 120
        hist = run_sft(d, log=self.log)
        return {
            "loss": round(hist[-1]["loss"], 4),
            "val_loss": _r(hist[-1].get("val_loss")),
            "steps": hist[-1]["step"],
        }

    def distill(self) -> dict[str, Any]:
        from zero.post.distill import run_distill

        d = load_tiny("distill", self.o)
        _set_seq(d, 512, 4.0)
        d["distill"]["n_tasks"] = 24
        d["train"]["max_steps"] = 20
        s = run_distill(d, log=self.log)
        h = s["history"][-1]
        return {
            "teacher_candidates": s["meta"]["n_candidates"],
            "verified": s["meta"]["n_verified"],
            "loss": round(h["loss"], 4),
            "ce": _r(h.get("ce")),
            "kd": _r(h.get("kd")),
        }

    def dpo(self) -> dict[str, Any]:
        from zero.post.dpo import run_dpo

        d = load_tiny("dpo", self.o)
        _set_seq(d, 512, 4.0)
        d["dpo"]["generate_pairs"] = 32
        d["train"]["max_steps"] = 24
        hist = run_dpo(d, log=self.log)
        h = hist[-1]
        return {
            "loss": round(h["loss"], 4),
            "acc": _r(h["acc"]),
            "margin": _r(h["margin"]),
            "steps": h["step"],
        }

    def grpo(self) -> dict[str, Any]:
        from zero.post.grpo import run_grpo

        d = load_tiny("grpo", self.o)
        _set_seq(d, 512, 4.0)
        d["train"]["max_steps"] = 10
        hist = run_grpo(d, log=self.log)
        rew = [h["reward_mean"] for h in hist]
        return {
            "reward_first": _r(rew[0]),
            "reward_last": _r(rew[-1]),
            "format_rate_last": _r(hist[-1]["format_rate"]),
            "call_rate_last": _r(hist[-1]["call_rate"]),
            "resp_len_last": _r(hist[-1]["resp_len"]),
            "steps": hist[-1]["step"],
        }

    def evaluate(self) -> dict[str, Any]:
        from zero.eval.harness import EvalConfig, EvalModel, run_eval

        ec = EvalConfig(
            models=[
                EvalModel("sft", f"{self.o}/sft/ckpt"),
                EvalModel("dpo", f"{self.o}/dpo/ckpt"),
                EvalModel("grpo", f"{self.o}/grpo/ckpt"),
            ],
            mc_tasks=["toy_mc.jsonl"],
            gen_tasks=["toy_gen.jsonl"],
            tool_tasks="tool_dev.jsonl",
            max_items=30,
            out_dir=f"{self.o}/eval",
            baseline="sft",
            n_boot=2000,
            cpu_threads=1,
        )
        payload = run_eval(ec, self.log)
        r = payload["results"]
        info: dict[str, Any] = {}
        for m in ("sft", "grpo"):
            info[f"{m}_mc_acc"] = _r(r[m]["toy_mc"]["acc"])
            info[f"{m}_tool_call_exact"] = _r(r[m]["tool_dev"]["call_exact"])
        g = [c for c in payload["comparisons"] if c["model"] == "grpo" and c["task"] == "tool_dev"][
            0
        ]
        info["grpo_vs_sft_tool"] = (
            f"{g['diff']:+.3f} [{g['ci_low']:+.3f},{g['ci_high']:+.3f}] {g['decision']}"
        )
        return info

    def export(self) -> dict[str, Any]:
        from zero.export import gguf
        from zero.hf import export_to_hf_qwen3
        from zero.post.chat import render_text
        from zero.post.common import load_policy
        from zero.post.envs.tool_env import dev_tasks, reference_messages

        model, tok = load_policy(f"{self.o}/grpo/ckpt")
        hf_dir = export_to_hf_qwen3(
            model, None, self.out / "hf_chat", tokenizer=tok, dtype=torch.float32, chat=True
        )
        info: dict[str, Any] = {"hf_dir": "hf_chat"}
        try:
            import transformers

            hf = transformers.AutoModelForCausalLM.from_pretrained(
                str(hf_dir), dtype=torch.float32
            ).eval()
            hf_tok = transformers.AutoTokenizer.from_pretrained(str(hf_dir))
            t = dev_tasks(1)[0]
            msgs = reference_messages(t)
            same = hf_tok.apply_chat_template(msgs, tools=t.tools, tokenize=False) == render_text(
                msgs, t.tools
            )
            ids = torch.tensor([tok.encode(render_text(msgs, t.tools))])
            with torch.no_grad():
                diff = (hf(ids).logits - model.eval()(ids)).abs().max().item()
            info.update({"hf_template_identical": same, "hf_logits_maxdiff": f"{diff:.1e}"})
        except ImportError:
            info["hf_check"] = "transformers 未安装，跳过"
        try:
            f16 = gguf.convert_hf_to_gguf(hf_dir, self.out / "gguf" / "zero-tiny-f16.gguf", "f16")
            info["gguf_f16_MB"] = round(f16.stat().st_size / 1e6, 2)
            if gguf.find_binary("llama-quantize") and gguf.find_binary("llama-simple"):
                q = gguf.quantize(f16, self.out / "gguf" / "zero-tiny-Q8_0.gguf", "Q8_0")
                info["gguf_q8_MB"] = round(q.stat().st_size / 1e6, 2)
                txt = gguf.run_llama(
                    q, "<|im_start|>user\n你好<|im_end|>\n<|im_start|>assistant\n", 8
                )
                info["llama_simple_ok"] = bool(txt.strip())
            else:
                info["llama_cpp_binaries"] = (
                    "未编译，跳过量化与试跑（python -m zero.export.gguf --quantize Q8_0 会编译）"
                )
        except Exception as e:  # noqa: BLE001 - GGUF 依赖外部仓库，失败只记录不中断
            info["gguf_error"] = f"{type(e).__name__}: {e}"[:200]
        return info

    def demo(self) -> dict[str, Any]:
        from zero.demo.cli import SYSTEM, chat_turn
        from zero.post.common import chat_complete, load_policy

        model, tok = load_policy(self.out / "hf_chat")
        model.eval()
        events: list[str] = []
        msgs = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": "帮我算一下 12 * (3 + 4) 等于多少？"},
        ]
        ans = chat_turn(
            lambda m, t: chat_complete(model, tok, m, t, 64, 0.0)[0],
            msgs,
            REPO / "zero",
            max_turns=3,
            on_event=lambda k, _: events.append(k),
        )
        return {"events": ",".join(events) or "none", "answer": ans[:60].replace("\n", " ")}


def _r(x: Any) -> Any:
    return round(float(x), 4) if isinstance(x, int | float) else x


def summary_table(rows: list[dict[str, Any]]) -> str:
    lines = ["| 阶段 | 状态 | 耗时 (s) | 指标 |", "|---|---|---:|---|"]
    for r in rows:
        metrics = "; ".join(
            f"{k}={v}" for k, v in r.items() if k not in ("stage", "status", "seconds")
        )
        lines.append(f"| {r['stage']} | {r['status']} | {r['seconds']} | {metrics} |")
    total = sum(r["seconds"] for r in rows)
    lines.append(f"| **合计** | | {total:.1f} | |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="端到端冒烟测试（CPU）")
    ap.add_argument("--out", default="out/smoke")
    ap.add_argument(
        "--keep", action="store_true", help="不清空输出目录（已有 checkpoint 会被续用）"
    )
    ap.add_argument("--threads", type=int, default=1)
    args = ap.parse_args(argv)
    torch.set_num_threads(args.threads)
    out = Path(args.out).resolve()
    if out.exists() and not args.keep:
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    os.chdir(REPO)
    s = Smoke(out)
    t0 = time.perf_counter()
    ok = True
    try:
        s.stage("data+tokenizer", s.data_and_tokenizer)
        s.stage("pretrain", s.pretrain)
        s.stage("midtrain", s.midtrain)
        s.stage("sft", s.sft)
        s.stage("distill", s.distill)
        s.stage("dpo", s.dpo)
        s.stage("grpo", s.grpo)
        s.stage("eval", s.evaluate)
        s.stage("export (HF + GGUF)", s.export)
        s.stage("demo", s.demo)
    except RuntimeError as e:
        ok = False
        print(f"[smoke] 中止：{e}", file=sys.stderr)
    table = summary_table(s.rows)
    total = time.perf_counter() - t0
    md = (
        f"# 冒烟测试结果\n\n输出目录：`{out}`，总耗时 {total:.0f}s，torch 线程数 {args.threads}。\n\n{table}\n\n"
        "tiny 模型约 1.3M 参数；分数只说明代码通路是通的。蒸馏的教师是 tiny SFT 模型自己（替身，见 zero/post/distill.py）。\n"
    )
    (out / "SUMMARY.md").write_text(md, encoding="utf-8")
    (out / "summary.json").write_text(
        json.dumps({"ok": ok, "total_s": total, "stages": s.rows}, ensure_ascii=False, indent=2)
    )
    print("\n" + md)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
