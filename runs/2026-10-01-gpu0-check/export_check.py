"""阶段 6 第 10 项（单卡版）：HF 导出 + transformers 在 GPU 上加载，与 zero 对拍 logits 和贪心生成。

    CUDA_VISIBLE_DEVICES=0 UV_NO_SYNC=1 uv run python runs/2026-10-01-gpu0-check/export_check.py \
        <zero checkpoint 目录> <输出目录>

1. `load_policy(ckpt, device="cuda")`（模型直接在 GPU 上）→ `export_to_hf_qwen3(..., chat=True)`，
   分别导出 FP32 和 BF16 两份；
2. `transformers.AutoModelForCausalLM.from_pretrained(...).cuda()`：FP32 与 zero 的 FP32 对拍 logits；
   BF16 与"从同一 HF 目录读回的 zero 模型（BF16）"对拍；
3. 贪心生成 32 个 token：HF `generate` 与 `zero.generate` 在 GPU（FP32）上是否逐字相同；
4. `apply_chat_template` 与 `zero.post.chat.render_text` 逐字相同。
vLLM 未安装，不测。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from zero.generate import generate  # noqa: E402
from zero.hf import export_to_hf_qwen3  # noqa: E402
from zero.post.chat import render_text  # noqa: E402
from zero.post.common import load_policy  # noqa: E402
from zero.post.envs.tool_env import dev_tasks, reference_messages  # noqa: E402


def main() -> None:
    import transformers

    assert torch.cuda.device_count() == 1, "只允许看到 1 张卡（CUDA_VISIBLE_DEVICES=0）"
    ckpt, out = Path(sys.argv[1]), Path(sys.argv[2])
    dev = torch.device("cuda", 0)
    model, tok = load_policy(ckpt, device=dev)
    model.eval()
    res: dict = {
        "ckpt": str(ckpt),
        "params_M": round(model.num_params() / 1e6, 2),
        "transformers": transformers.__version__,
    }

    t = dev_tasks(1)[0]
    msgs = reference_messages(t)
    text = render_text(msgs, t.tools)
    ids = torch.tensor([tok.encode(text)[: model.config.max_seq_len]], device=dev)
    res["n_tokens"] = ids.shape[1]

    # FP32 导出 → HF FP32（GPU）对拍
    d32 = export_to_hf_qwen3(
        model, None, out / "hf_fp32", tokenizer=tok, dtype=torch.float32, chat=True
    )
    hf32 = (
        transformers.AutoModelForCausalLM.from_pretrained(str(d32), dtype=torch.float32)
        .to(dev)
        .eval()
    )
    with torch.no_grad():
        a, b = hf32(ids).logits, model(ids)
    res["fp32_logits_maxdiff"] = float((a - b).abs().max())
    res["fp32_logits_max"] = float(b.abs().max())
    hf_tok = transformers.AutoTokenizer.from_pretrained(str(d32))
    res["chat_template_identical"] = (
        hf_tok.apply_chat_template(msgs, tools=t.tools, tokenize=False) == text
    )

    # 贪心生成：HF generate vs zero.generate（GPU，FP32）
    prompt = ids[:, : min(64, ids.shape[1])]
    with torch.no_grad():
        hf_out = hf32.generate(
            prompt, max_new_tokens=32, do_sample=False, eos_token_id=None, pad_token_id=0
        )[0, prompt.shape[1] :].tolist()
    z_out = generate(model, prompt[0].tolist(), 32, temperature=0.0)
    res["greedy_32_identical"] = z_out[: len(hf_out)] == hf_out  # HF 遇到 eos 会提前停
    res["greedy_lens"] = [len(hf_out), len(z_out)]
    del hf32
    torch.cuda.empty_cache()

    # BF16 导出 → HF BF16 vs zero（从 HF 目录读回，转 BF16）
    d16 = export_to_hf_qwen3(
        model, None, out / "hf_bf16", tokenizer=tok, dtype=torch.bfloat16, chat=True
    )
    hf16 = (
        transformers.AutoModelForCausalLM.from_pretrained(str(d16), dtype=torch.bfloat16)
        .to(dev)
        .eval()
    )
    z16, _ = load_policy(d16, device=dev)
    z16 = z16.to(torch.bfloat16).eval()
    with torch.no_grad():
        a16, b16 = hf16(ids).logits.float(), z16(ids).float()
    res["bf16_logits_maxdiff"] = float((a16 - b16).abs().max())
    res["bf16_vs_fp32_logits_maxdiff"] = float((a16 - b.float()).abs().max())
    res["bf16_top1_agree_with_fp32"] = float((a16.argmax(-1) == b.argmax(-1)).float().mean())
    res["hf_bf16_dir_MB"] = round(sum(f.stat().st_size for f in d16.iterdir()) / 1e6, 1)
    print(json.dumps(res, ensure_ascii=False, indent=1))
    (out / "export_check.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
