"""蒸馏：教师数据生成 + 执行验证 + 序列级 / logits 蒸馏（对应第 17 章）。

    uv run python -m zero.post.distill --config configs/tiny/distill.toml

**两种蒸馏**（都属于 GOAL.md 2.1 的"共识"）：

1. **序列级蒸馏**（sequence-level KD）：让教师模型解题，只保留**执行验证通过**的轨迹，当 SFT 数据训学生。
   对教师只需要"能生成文本"，教师可以是任何模型、任何分词器，也可以是远程的 HTTP 服务；
2. **logits 蒸馏**：在同一段文本的每个位置，让学生的下一个 token 分布去贴教师的分布：
       KL(p_T ‖ p_S) = Σ_v p_T(v) · (log p_T(v) − log p_S(v))      （前向 KL，"覆盖教师的全部可能"）
   两边的 logits 先除以温度 τ 再 softmax，损失乘 τ²（Hinton et al. 2015，保持梯度量级）；
   `kd_topk > 0` 时只用教师概率最大的 k 个 token（教师分布在 k 个上重新归一化，学生不归一化），
   省显存/省存储，第二步离线存教师 logits 时用。
   **logits 蒸馏要求教师和学生用同一个分词器**（同一个词表、同样的切分）：两个分布必须定义在同一组
   token 上才能逐位比较。本模块会检查两边分词器的哈希，不一致直接报错——换了分词器的教师只能做序列级蒸馏。

最终损失 = (1 − α) · 交叉熵（学生在教师轨迹上的 SFT 损失）+ α · KL，α = `kd_alpha`。

**教师后端**（`[teacher] backend`）：

- `"local"`：本地模型，zero 的 checkpoint 或导出的 HF 目录（Qwen3 稠密结构，比如第二步的开源教师），
  用我们自己的 `zero.generate`（带 KV cache）采样；只有它能做 logits 蒸馏；
- `"openai"`：任何 OpenAI 兼容的 HTTP 接口（第二步用 vLLM 起教师服务：
  `vllm serve <教师> --enable-auto-tool-choice --tool-call-parser hermes`），只做序列级蒸馏。
  单元测试用本地假服务器测它，不联网。

**执行验证**：对每个任务，教师第一轮的工具调用必须拿满 `score_tool_calls`（函数名 + 参数对），
执行工具、把结果喂回后，最终回答必须通过 `score_final_answer`；不通过的丢掉。
每条数据和元数据文件（`<out_jsonl>.meta.json`）都记录教师的名称、版本、许可证（GOAL.md 3.3）。
`backend = "openai"` 或教师不是本项目自己的模型时，必须在配置里写明 `license_allows_distillation = true`
（许可证允许用输出训练其他模型，如 Apache-2.0 / MIT），否则拒绝运行。

**冒烟测试里的教师是替身**：GOAL.md 希望冒烟测试用一个能在 CPU 上跑的小开源模型当教师，但这个环境
访问不了 huggingface.co，下载不了任何开源权重，所以 `configs/tiny/distill.toml` 用 tiny SFT 模型
**自己**当教师（自蒸馏 / 拒绝采样微调）：序列级数据来自它自己采样、再经执行验证筛过的轨迹；
logits 蒸馏项是对"冻结的 SFT 模型"的 KL，起步时为 0。这只验证代码通路，不代表蒸馏的效果。
第二步换真教师只需改 `[teacher]`：`backend = "local"` + `path = <开源 Qwen3 系教师的 HF 目录>`
（同分词器时才能开 logits 蒸馏——我们自训的分词器和任何开源教师都不同，所以实际只做序列级），
或 `backend = "openai"` + vLLM 服务地址。

**在线策略蒸馏**（on-policy distillation，GOAL.md 2.1 里列为"待核实"，默认关闭）：学生自己采样，
在学生生成的序列上逐位置最小化反向 KL(p_S ‖ p_T)（Agarwal et al. 2023 的 GKD，λ = 1 的特例）。
`on_policy_steps > 0` 时在离线蒸馏之后跑这么多步。它是否已达到"共识"待第 17 章写作时核实，
正文不讲，只放"前沿观察"。
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from zero.post.chat import assistant_text
from zero.post.common import (
    chat_complete,
    load_policy,
    load_post_config,
    pad_batch,
    write_jsonl,
)
from zero.tokenizer import Tokenizer


@dataclass
class TeacherConfig:
    backend: str = "local"  # "local"（zero checkpoint / HF 目录）| "openai"（OpenAI 兼容 HTTP）
    path: str = ""  # local：checkpoint 目录或 HF 目录
    base_url: str = ""  # openai：如 http://localhost:8000/v1
    model: str = ""  # openai：服务端的模型名
    api_key_env: str = "OPENAI_API_KEY"  # 从这个环境变量读 API key（本地 vLLM 可以不设）
    name: str = ""  # 记录用：教师名称，如 "Qwen3-8B"
    version: str = ""  # 记录用：版本 / commit / 发布日期
    license: str = ""  # 记录用：许可证，如 "Apache-2.0"
    license_allows_distillation: bool = False  # 许可证是否允许用输出训练其他模型
    temperature: float = 0.7
    top_p: float = 0.95
    max_new_tokens: int = 96
    timeout_s: float = 120.0


@dataclass
class DistillConfig:
    out_jsonl: str = ""  # 序列级蒸馏数据（验证通过的轨迹）
    shard_dir: str = ""  # 打包后的窗口
    n_tasks: int = 64  # 用多少个 tool_env 训练任务
    samples_per_task: int = 4  # 每个任务教师采样几次
    keep_per_task: int = 1  # 每个任务最多保留几条通过验证的轨迹
    env_seed: int = 0
    mix_sft_jsonl: str = ""  # 可选：把原 SFT 数据混进来（防止只有少量教师数据时遗忘）
    mix_sft_max: int = 0  # 最多混多少条
    logits_kd: bool = True  # 是否加 logits 蒸馏项（需要 local 后端 + 同一个分词器）
    kd_alpha: float = 0.5
    kd_temperature: float = 1.0
    kd_topk: int = 0
    on_policy_steps: int = 0  # >0：离线蒸馏后再做这么多步在线策略蒸馏（待核实，默认关）
    on_policy_lr: float = 1e-4
    on_policy_batch: int = 4
    overwrite: bool = False


# ---------------------------------------------------------------------------
# 损失
# ---------------------------------------------------------------------------


def kd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    mask: torch.Tensor,
    temperature: float = 1.0,
    topk: int = 0,
) -> torch.Tensor:
    """前向 KL(p_T ‖ p_S) 在 mask 位置上的平均值 × τ²。logits: (B, T, V)，mask: (B, T)。

    **两者必须来自同一个分词器**（同一个词表），否则第 v 维不是同一个 token，比较没有意义。"""
    if student_logits.shape != teacher_logits.shape:
        raise ValueError(
            f"学生 logits {tuple(student_logits.shape)} 与教师 {tuple(teacher_logits.shape)} 形状不同："
            "logits 蒸馏要求两者用同一个分词器"
        )
    tau = temperature
    s = student_logits.float() / tau
    t = teacher_logits.float() / tau
    if topk > 0:
        t_top, idx = t.topk(topk, dim=-1)
        logp_t = F.log_softmax(t_top, dim=-1)
        logp_s = F.log_softmax(s, dim=-1).gather(-1, idx)
    else:
        logp_t = F.log_softmax(t, dim=-1)
        logp_s = F.log_softmax(s, dim=-1)
    kl = (logp_t.exp() * (logp_t - logp_s)).sum(-1)
    m = mask.float()
    return (kl * m).sum() / m.sum().clamp(min=1.0) * tau * tau


def reverse_kl_loss(
    student_logits: torch.Tensor, teacher_logits: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """反向 KL(p_S ‖ p_T) 在 mask 位置上的平均值（在线策略蒸馏用，梯度只流向学生）。"""
    logp_s = F.log_softmax(student_logits.float(), dim=-1)
    logp_t = F.log_softmax(teacher_logits.float(), dim=-1)
    kl = (logp_s.exp() * (logp_s - logp_t)).sum(-1)
    m = mask.float()
    return (kl * m).sum() / m.sum().clamp(min=1.0)


# ---------------------------------------------------------------------------
# 教师后端
# ---------------------------------------------------------------------------


class LocalTeacher:
    """本地模型（zero checkpoint 或 Qwen3 结构的 HF 目录），用 zero.generate 采样。"""

    def __init__(self, cfg: TeacherConfig, device: str | torch.device = "cpu") -> None:
        self.cfg = cfg
        self.model, self.tok = load_policy(cfg.path, device=device)
        self.model.eval()

    def complete(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]], n: int, seed: int
    ) -> list[str]:
        c = self.cfg
        return chat_complete(
            self.model, self.tok, messages, tools, c.max_new_tokens, c.temperature, c.top_p, n, seed
        )


def to_openai_messages(messages: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """我们的消息格式 → OpenAI Chat Completions 格式（tool_calls 的 arguments 是 JSON 字符串，带 id）。"""
    out: list[dict[str, Any]] = []
    call_ids: list[str] = []
    k = 0
    for m in messages:
        m = dict(m)
        if m.get("role") == "assistant" and m.get("tool_calls"):
            tcs = []
            for tc in m["tool_calls"]:
                f = tc.get("function", tc)
                args = f.get("arguments", {})
                cid = f"call_{k}"
                k += 1
                call_ids.append(cid)
                tcs.append(
                    {
                        "id": cid,
                        "type": "function",
                        "function": {
                            "name": f["name"],
                            "arguments": args
                            if isinstance(args, str)
                            else json.dumps(args, ensure_ascii=False),
                        },
                    }
                )
            m["tool_calls"] = tcs
            m["content"] = m.get("content") or None
        elif m.get("role") == "tool":
            m.setdefault("tool_call_id", call_ids.pop(0) if call_ids else "call_0")
        out.append(m)
    return out


def openai_message_to_text(msg: dict[str, Any]) -> str:
    """OpenAI 返回的 assistant message → 我们模板里助手应该生成的文本。"""
    calls = []
    for tc in msg.get("tool_calls") or []:
        f = tc.get("function", tc)
        args = f.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                pass  # 保留原字符串：渲染出来就是坏的 JSON，验证会把它筛掉
        calls.append({"name": f.get("name"), "arguments": args})
    return assistant_text(
        {"role": "assistant", "content": msg.get("content") or "", "tool_calls": calls}
    )


class OpenAITeacher:
    """OpenAI 兼容的 HTTP 接口（vLLM / SGLang / 各家 API）。只用标准库 urllib。"""

    def __init__(self, cfg: TeacherConfig) -> None:
        if not cfg.base_url or not cfg.model:
            raise ValueError("[teacher] backend=openai 需要 base_url 和 model")
        self.cfg = cfg

    def complete(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]], n: int, seed: int
    ) -> list[str]:
        c = self.cfg
        payload = {
            "model": c.model,
            "messages": to_openai_messages(messages),
            "tools": tools,
            "n": n,
            "temperature": c.temperature,
            "top_p": c.top_p,
            "max_tokens": c.max_new_tokens,
            "seed": seed,
        }
        headers = {"Content-Type": "application/json"}
        key = os.environ.get(c.api_key_env, "")
        if key:
            headers["Authorization"] = f"Bearer {key}"
        req = urllib.request.Request(
            c.base_url.rstrip("/") + "/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=c.timeout_s) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return [openai_message_to_text(ch["message"]) for ch in data.get("choices", [])]


def build_teacher(
    cfg: TeacherConfig, device: str | torch.device = "cpu"
) -> LocalTeacher | OpenAITeacher:
    if cfg.backend == "local":
        return LocalTeacher(cfg, device)
    if cfg.backend == "openai":
        return OpenAITeacher(cfg)
    raise ValueError(f"[teacher] backend 只能是 local / openai，当前 {cfg.backend!r}")


def check_license(cfg: TeacherConfig, is_self: bool) -> None:
    """GOAL.md 3.3：教师的许可证必须允许"用输出训练其他模型"。本项目自己的模型当替身时例外。"""
    if is_self:
        return
    if not cfg.license_allows_distillation:
        raise PermissionError(
            f"教师 {cfg.name or cfg.path or cfg.model!r} 的许可证未确认允许蒸馏："
            "核实许可证（如 Apache-2.0 / MIT）后在 [teacher] 里写 license、license_allows_distillation = true"
        )


# ---------------------------------------------------------------------------
# 教师数据：采样 → 执行 → 验证
# ---------------------------------------------------------------------------


def teacher_trajectories(
    teacher: LocalTeacher | OpenAITeacher,
    task: Any,
    n: int,
    keep: int,
    seed: int,
) -> tuple[list[list[dict[str, Any]]], int]:
    """对一个任务：教师第一轮采样 n 次（一次请求 / 一个 batch），只有工具调用拿满分的继续——执行工具、
    喂回结果、让教师写最终回答（贪心），最终回答也通过才保留。返回 (通过的完整对话列表, 候选数)。"""
    from zero.post.chat import parse_assistant
    from zero.post.envs.tool_env import execute_safely, score_final_answer, score_tool_calls

    firsts = teacher.complete(task.messages, task.tools, n, seed)
    kept: list[list[dict[str, Any]]] = []
    for j, text in enumerate(firsts):
        if len(kept) >= keep:
            break
        if score_tool_calls(task, text).total < 0.999:
            continue
        parsed = parse_assistant(text)
        msgs = [dict(m) for m in task.messages] + [parsed.to_message()]
        if parsed.tool_calls:
            for c in parsed.tool_calls:
                msgs.append(
                    {
                        "role": "tool",
                        "content": json.dumps(
                            execute_safely(c["name"], c["arguments"]), ensure_ascii=False
                        ),
                    }
                )
            final = teacher.complete(msgs, task.tools, 1, seed + 7919 * (j + 1))[0]
            if not score_final_answer(task, final).answer_ok:
                continue
            msgs.append(parse_assistant(final).to_message())
        elif not score_final_answer(task, text).answer_ok:
            continue
        kept.append(msgs)
    return kept, len(firsts)


def generate_kd_data(
    teacher: LocalTeacher | OpenAITeacher,
    tcfg: TeacherConfig,
    dc: DistillConfig,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """生成序列级蒸馏数据，写 out_jsonl 和 out_jsonl.meta.json，返回元数据。"""
    from zero.post.envs.tool_env import generate_tasks

    tasks = generate_tasks(dc.n_tasks, seed=dc.env_seed, split="train")
    t0 = time.time()
    rows = []
    n_cand = 0
    teacher_meta = {
        "name": tcfg.name,
        "version": tcfg.version,
        "license": tcfg.license,
        "license_allows_distillation": tcfg.license_allows_distillation,
        "backend": tcfg.backend,
        "path_or_model": tcfg.path or tcfg.model,
    }
    for i, task in enumerate(tasks):
        kept, n = teacher_trajectories(
            teacher, task, dc.samples_per_task, dc.keep_per_task, dc.env_seed * 1_000_003 + i
        )
        n_cand += n
        for msgs in kept:
            rows.append(
                {"messages": msgs, "tools": task.tools, "task_id": task.id, "teacher": teacher_meta}
            )
    write_jsonl(dc.out_jsonl, rows)
    meta = {
        "teacher": teacher_meta,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "generation": {
            "temperature": tcfg.temperature,
            "top_p": tcfg.top_p,
            "max_new_tokens": tcfg.max_new_tokens,
        },
        "env": {"n_tasks": dc.n_tasks, "env_seed": dc.env_seed, "split": "train"},
        "n_candidates": n_cand,
        "n_verified": len(rows),
        "tasks_with_verified": len({r["task_id"] for r in rows}),
        "pass_rate": len(rows) / max(n_cand, 1),
        "filter": "score_tool_calls == 1 且 score_final_answer.answer_ok（zero/post/envs/tool_env.py）",
        "seconds": round(time.time() - t0, 1),
    }
    Path(str(dc.out_jsonl) + ".meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2)
    )
    log(
        f"[distill] 教师 {tcfg.name or tcfg.path}：{n_cand} 个候选 → 验证通过 {len(rows)} 条"
        f"（通过率 {meta['pass_rate']:.1%}，{meta['seconds']}s）"
    )
    return meta


# ---------------------------------------------------------------------------
# 训练：序列级（SFT 损失）+ logits 蒸馏
# ---------------------------------------------------------------------------


def _make_distill_trainer_cls():  # noqa: ANN202
    from zero.train.trainer import Trainer

    class DistillTrainer(Trainer):
        """在通用 Trainer 上只改一处：损失 = (1-α)·CE + α·KL(教师 ‖ 学生)。"""

        def __init__(
            self, cfg: Any, teacher_model: torch.nn.Module, dc: DistillConfig, **kw: Any
        ) -> None:
            super().__init__(cfg, **kw)
            self.teacher = teacher_model.to(self.device).eval()
            for p in self.teacher.parameters():
                p.requires_grad_(False)
            self.dc = dc
            self.last_kd = 0.0
            self.last_ce = 0.0

        def _forward_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            from zero.model import cross_entropy_loss

            with self.autocast():
                logits = self.model(x)
                with torch.no_grad():
                    t_logits = self.teacher(x)
            ce = cross_entropy_loss(logits, y)
            kd = kd_loss(logits, t_logits, y != -100, self.dc.kd_temperature, self.dc.kd_topk)
            self.last_kd = float(kd.detach())
            self.last_ce = float(ce.detach())
            a = self.dc.kd_alpha
            return (1 - a) * ce + a * kd

        def extra_metrics(self) -> dict[str, Any]:
            return {"ce": self.last_ce, "kd": self.last_kd}

    return DistillTrainer


def on_policy_distill(
    student: torch.nn.Module,
    teacher: torch.nn.Module,
    tok: Tokenizer,
    tasks: Sequence[Any],
    steps: int,
    lr: float = 1e-4,
    batch: int = 4,
    max_new_tokens: int = 64,
    temperature: float = 1.0,
    seed: int = 0,
    log: Callable[[str], None] = print,
) -> list[dict[str, float]]:
    """在线策略蒸馏（待核实，见模块说明）：学生采样 → 在学生的样本上最小化反向 KL(p_S ‖ p_T)。"""
    from zero.post.grpo import sample_group

    opt = torch.optim.AdamW(
        [p for p in student.parameters() if p.requires_grad], lr=lr, weight_decay=0.0
    )
    hist = []
    for step in range(steps):
        task = tasks[(seed + step) % len(tasks)]
        p_ids, resps = sample_group(
            student, tok, task, batch, max_new_tokens, temperature, 1.0, seed * 1000 + step
        )
        seqs = [p_ids + r for r in resps]
        masks = [[False] * len(p_ids) + [True] * len(r) for r in resps]
        ids, mask = pad_batch(seqs, masks, tok.eot_id, next(student.parameters()).device)
        student.train()
        s_logits = student(ids[:, :-1])
        with torch.no_grad():
            t_logits = teacher(ids[:, :-1])
        loss = reverse_kl_loss(s_logits, t_logits, mask[:, 1:])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        hist.append({"step": step + 1, "reverse_kl": float(loss.detach())})
        log(f"[on-policy] step {step + 1}/{steps} | reverse KL {float(loss.detach()):.4f}")
    return hist


def run_distill(
    src: str | os.PathLike | dict[str, Any],
    overrides: Sequence[str] | None = None,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """教师数据生成 →（混入 SFT 数据）→ 打包 → 训练学生。返回 {"meta", "history", ...}。"""
    from zero.post.common import read_jsonl
    from zero.post.sft import build_sft_shards
    from zero.train.dist import DistInfo
    from zero.train.trainer import Trainer

    cfg, sec = load_post_config(
        src, {"teacher": TeacherConfig, "distill": DistillConfig}, overrides
    )
    tcfg: TeacherConfig = sec["teacher"]
    dc: DistillConfig = sec["distill"]
    tc = cfg.train
    if tc.cpu_threads > 0:
        torch.set_num_threads(tc.cpu_threads)
    if cfg.train.data.format != "sft":
        raise ValueError('蒸馏配置需要 [data] format = "sft"（教师轨迹按 SFT 方式打包）')
    tok = Tokenizer.load(tc.data.tokenizer)

    # 教师是不是"本项目自己的模型"（冒烟测试的替身）：local 后端 + zero checkpoint 目录
    is_self = tcfg.backend == "local" and not (Path(tcfg.path) / "config.json").exists()
    check_license(tcfg, is_self)
    # 尚未在 GPU 上验证：有 CUDA 时本地教师放到 GPU 上
    t_device = "cuda" if tc.device in ("auto", "cuda") and torch.cuda.is_available() else "cpu"
    teacher = build_teacher(tcfg, t_device)
    # 先检查能不能做 logits 蒸馏（分词器不同就尽早报错，别等教师数据生成完）
    use_logits = dc.logits_kd and isinstance(teacher, LocalTeacher)
    if dc.logits_kd and not use_logits:
        log("[distill] 教师不是本地模型，拿不到 logits：只做序列级蒸馏")
    if use_logits and teacher.tok.hash() != tok.hash():  # type: ignore[union-attr]
        raise ValueError(
            "logits 蒸馏要求教师与学生使用同一个分词器（哈希不同）；换了分词器的教师请设 [distill] logits_kd = false"
        )

    meta_path = Path(str(dc.out_jsonl) + ".meta.json")
    if Path(dc.out_jsonl).exists() and meta_path.exists() and not dc.overwrite:
        meta = json.loads(meta_path.read_text())
        log(f"[distill] 复用已有的教师数据 {dc.out_jsonl}（{meta['n_verified']} 条）")
    else:
        meta = generate_kd_data(teacher, tcfg, dc, log)

    rows = read_jsonl(dc.out_jsonl)
    if dc.mix_sft_jsonl and dc.mix_sft_max > 0:
        extra = read_jsonl(dc.mix_sft_jsonl)[: dc.mix_sft_max]
        rows = rows + extra
        log(f"[distill] 混入 {len(extra)} 条原 SFT 数据（共 {len(rows)} 条）")
    if not rows:
        raise ValueError("[distill] 没有任何训练数据：教师轨迹全部没通过验证，且没有混入 SFT 数据")
    shard_dir = Path(dc.shard_dir or Path(tc.out_dir) / "data")
    mixed = shard_dir / "train_mix.jsonl"
    write_jsonl(mixed, rows)
    stats = build_sft_shards(mixed, tok, tc.data.seq_len, shard_dir / "train.bin")
    log(f"[distill] 打包：{stats}")
    from zero.config import DataSourceConfig

    tc.data.sources = [DataSourceConfig(name="distill", path=str(shard_dir / "train.bin"))]
    tc.data.val = ""
    tc.eval_every = 0

    os.makedirs(tc.out_dir, exist_ok=True)
    if use_logits:
        trainer = _make_distill_trainer_cls()(cfg, teacher.model, dc, info=DistInfo(), log=log)  # type: ignore[union-attr]
    else:
        trainer = Trainer(cfg, DistInfo(), log)
    history = trainer.train()

    on_policy_hist: list[dict[str, float]] = []
    if dc.on_policy_steps > 0 and isinstance(teacher, LocalTeacher):
        from zero.post.envs.tool_env import generate_tasks
        from zero.train.checkpoint import save_checkpoint

        student = trainer.raw_model
        tasks = generate_tasks(64, seed=dc.env_seed + 1, split="train")
        on_policy_hist = on_policy_distill(
            student,
            teacher.model,
            tok,
            tasks,
            dc.on_policy_steps,
            dc.on_policy_lr,
            dc.on_policy_batch,
            tcfg.max_new_tokens,
            1.0,
            tc.seed,
            log,
        )
        save_checkpoint(
            tc.checkpoint_dir,
            trainer.step + dc.on_policy_steps,
            student,
            meta={"config": cfg.to_dict(), "on_policy_steps": dc.on_policy_steps},
        )
    summary = {
        "meta": meta,
        "pack": stats,
        "history": history,
        "on_policy": on_policy_hist,
        "config": {"teacher": asdict(tcfg), "distill": asdict(dc)},
    }
    Path(tc.out_dir, "distill_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str)
    )
    return summary


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="蒸馏（第 17 章）")
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = ap.parse_args(argv)
    run_distill(args.config, args.set)


if __name__ == "__main__":
    main()
