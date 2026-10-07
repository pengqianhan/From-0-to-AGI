"""Stage 6, item 10 (1-GPU version): HF export, then transformers loads the model on the GPU.

Parity check of the logits and of greedy generation against zero.

    CUDA_VISIBLE_DEVICES=0 UV_NO_SYNC=1 uv run python runs/2026-10-01-gpu0-check/export_check.py \
        <zero checkpoint directory> <output directory>

1. `load_policy(ckpt, device="cuda")` (the model is directly on the GPU) → `export_to_hf_qwen3(..., chat=True)`,
   one export in FP32 and one in BF16;
2. `transformers.AutoModelForCausalLM.from_pretrained(...).cuda()`: parity check of the FP32 logits against
   the FP32 zero model; parity check of BF16 against "the zero model read back from the same HF directory (BF16)";
3. greedy generation of 32 tokens: are HF `generate` and `zero.generate` on the GPU (FP32) token-for-token identical?
4. are the texts from `apply_chat_template` and `zero.post.chat.render_text` character-for-character identical?
vLLM is not installed, so the script does not test it.
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

    assert torch.cuda.device_count() == 1, "Only 1 GPU may be visible (CUDA_VISIBLE_DEVICES=0)"
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

    # FP32 export → parity check with HF FP32 (GPU)
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

    # Greedy generation: HF generate vs zero.generate (GPU, FP32)
    prompt = ids[:, : min(64, ids.shape[1])]
    with torch.no_grad():
        hf_out = hf32.generate(
            prompt, max_new_tokens=32, do_sample=False, eos_token_id=None, pad_token_id=0
        )[0, prompt.shape[1] :].tolist()
    z_out = generate(model, prompt[0].tolist(), 32, temperature=0.0)
    res["greedy_32_identical"] = z_out[: len(hf_out)] == hf_out  # HF stops early at eos
    res["greedy_lens"] = [len(hf_out), len(z_out)]
    del hf32
    torch.cuda.empty_cache()

    # BF16 export → HF BF16 vs zero (read back from the HF directory, converted to BF16)
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
