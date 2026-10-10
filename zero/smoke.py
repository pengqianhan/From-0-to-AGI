"""End-to-end smoke test: one command runs the full pipeline on a CPU (all chapters; GOAL.md 9.1).

    uv run python -m zero.smoke                     # output to out/smoke (cleared first)
    uv run python -m zero.smoke --out /tmp/smoke    # a different directory

Stages (all use configs/tiny/*.toml; out/tiny in the paths becomes the --out directory; some
overrides make the run faster, see below):

 1. data: generate text in the tool-call format (the reference conversations of tool_env)
 2. tokenizer: train a byte-level BPE on tiny_corpus (Shakespeare / Tang poems / code) + the
    tool-call text (when the tokenizer has seen JSON and the chat format, one conversation with
    tool descriptions decreases from about 700 tokens to about 330 tokens)
 3. pretrain: configs/tiny/pretrain.toml
 4. midtrain: configs/tiny/midtrain.toml; the data mixture adds tool-call format data
    (the mid-training of GOAL.md 3.3)
 5. SFT: configs/tiny/sft.toml (seq_len 1024 → 512)
 6. distill: configs/tiny/distill.toml (stand-in teacher = the tiny SFT model itself; see the
    notes in zero/post/distill.py)
 7. DPO: configs/tiny/dpo.toml
 8. GRPO: configs/tiny/grpo.toml
 9. OPD: configs/tiny/opd.toml (cross-stage on-policy distillation; teachers = the tiny distill and
    GRPO checkpoints)
10. eval: configs/tiny/eval.toml (toy sets + tool-call dev set; the SFT / DPO / GRPO / OPD models;
    paired bootstrap)
11. export: export an HF directory (transformers loads it for a parity check of the logits;
    apply_chat_template is identical to our template, character by character)
    + GGUF (the official llama.cpp conversion script; if llama.cpp is built, also quantize to
    Q8_0 and generate some tokens with llama-simple)
12. demo: ask the local tool-calling assistant one question

At the end, the script prints the time and the key metrics (loss / reward) of each stage, and
writes `<out>/SUMMARY.md` and `summary.json`.
The tiny model has only about 1.3M parameters. All scores show only that "the code paths work".
They do not show the effect of any method.
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
    """Replace each path in the config that starts with out/tiny with out."""
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
    # Make the input paths that are relative to the repository root (assets/…) absolute,
    # so the script can run from any directory
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
        except Exception as e:  # noqa: BLE001 - the smoke test records a failure in the table; it does not crash
            traceback.print_exc()
            info = {"error": f"{type(e).__name__}: {e}"[:300]}
            status = "FAILED"
        dt = time.perf_counter() - t0
        row = {"stage": name, "status": status, "seconds": round(dt, 1), **info}
        self.rows.append(row)
        self.log(
            f"[smoke] {name}: {status}, {dt:.1f}s, { {k: v for k, v in info.items() if k != 'error'} }"
        )
        if status != "ok":
            raise RuntimeError(f"Stage {name} failed: {info.get('error')}")
        return info

    # ---- Stages ----
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
        # <|im_start|> and the other markers in the chat text must stay special tokens. Thus skip clean
        # (clean removes literal special tokens).
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

    def opd(self) -> dict[str, Any]:
        from zero.post.opd import run_opd

        d = load_tiny("opd", self.o)
        _set_seq(d, 512, 4.0)
        d["train"]["max_steps"] = 5
        hist = run_opd(d, log=self.log)
        return {
            "kl_first": _r(hist[0]["kl"]),
            "kl_last": _r(hist[-1]["kl"]),
            "eos_rate_last": _r(hist[-1]["eos_rate"]),
            "steps": hist[-1]["step"],
        }

    def evaluate(self) -> dict[str, Any]:
        from zero.eval.harness import EvalConfig, EvalModel, run_eval

        ec = EvalConfig(
            models=[
                EvalModel("sft", f"{self.o}/sft/ckpt"),
                EvalModel("dpo", f"{self.o}/dpo/ckpt"),
                EvalModel("grpo", f"{self.o}/grpo/ckpt"),
                EvalModel("opd", f"{self.o}/opd/ckpt"),
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

        model, tok = load_policy(f"{self.o}/opd/ckpt")
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
            info["hf_check"] = "transformers is not installed; skipped"
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
                    "not built; skipped quantization and test run (python -m zero.export.gguf --quantize Q8_0 builds them)"
                )
        except Exception as e:  # noqa: BLE001 - GGUF needs an external repository; record a failure, do not stop
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
    lines = ["| Stage | Status | Time (s) | Metrics |", "|---|---|---:|---|"]
    for r in rows:
        metrics = "; ".join(
            f"{k}={v}" for k, v in r.items() if k not in ("stage", "status", "seconds")
        )
        lines.append(f"| {r['stage']} | {r['status']} | {r['seconds']} | {metrics} |")
    total = sum(r["seconds"] for r in rows)
    lines.append(f"| **Total** | | {total:.1f} | |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="End-to-end smoke test (CPU)")
    ap.add_argument("--out", default="out/smoke")
    ap.add_argument(
        "--keep", action="store_true", help="do not clear the output directory (existing checkpoints are used again)"
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
        s.stage("opd", s.opd)
        s.stage("eval", s.evaluate)
        s.stage("export (HF + GGUF)", s.export)
        s.stage("demo", s.demo)
    except RuntimeError as e:
        ok = False
        print(f"[smoke] Stopped: {e}", file=sys.stderr)
    table = summary_table(s.rows)
    total = time.perf_counter() - t0
    md = (
        f"# Smoke test results\n\nOutput directory: `{out}`, total time {total:.0f}s, torch threads {args.threads}.\n\n{table}\n\n"
        "The tiny model has about 1.3M parameters; the scores show only that the code paths work. The distillation teacher is the tiny SFT model itself (a stand-in, see zero/post/distill.py).\n"
    )
    (out / "SUMMARY.md").write_text(md, encoding="utf-8")
    (out / "summary.json").write_text(
        json.dumps({"ok": ok, "total_s": total, "stages": s.rows}, ensure_ascii=False, indent=2)
    )
    print("\n" + md)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
