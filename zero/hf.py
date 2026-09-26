"""与 Hugging Face Qwen3 格式互转（对应第 9、20 章）。

`zero` 的模型结构与 `Qwen3ForCausalLM`（稠密版）一一对应，所以可以：

- `load_from_hf_qwen3`：把官方实现的权重搬进 `zero.Transformer`，用来做 logits 对拍
  （`tests/test_model_hf_parity.py`）；
- `export_to_hf_qwen3`：把训练好的 `zero` 模型导出成 transformers / vLLM / llama.cpp 都认识的
  目录（`config.json` + `model.safetensors` + 可选的分词器文件）。

导出本身只依赖 `safetensors`，不需要安装 transformers；加载 HF 模型对象时才需要。

参数名对照（i 为层号）：

| zero | Hugging Face Qwen3 |
|---|---|
| tok_emb.weight | model.embed_tokens.weight |
| layers.i.attn_norm.weight | model.layers.i.input_layernorm.weight |
| layers.i.attn.wq/wk/wv/wo.weight | model.layers.i.self_attn.q_proj/k_proj/v_proj/o_proj.weight |
| layers.i.attn.q_norm/k_norm.weight | model.layers.i.self_attn.q_norm/k_norm.weight |
| layers.i.ffn_norm.weight | model.layers.i.post_attention_layernorm.weight |
| layers.i.ffn.w_gate/w_up/w_down.weight | model.layers.i.mlp.gate_proj/up_proj/down_proj.weight |
| norm.weight | model.norm.weight |
| lm_head.weight | lm_head.weight（共享 embedding 时不单独保存） |
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from zero.config import ModelConfig
from zero.model import Transformer

if TYPE_CHECKING:
    from zero.tokenizer import Tokenizer

_LAYER_MAP = {
    "attn_norm.weight": "input_layernorm.weight",
    "attn.wq.weight": "self_attn.q_proj.weight",
    "attn.wk.weight": "self_attn.k_proj.weight",
    "attn.wv.weight": "self_attn.v_proj.weight",
    "attn.wo.weight": "self_attn.o_proj.weight",
    "attn.q_norm.weight": "self_attn.q_norm.weight",
    "attn.k_norm.weight": "self_attn.k_norm.weight",
    "ffn_norm.weight": "post_attention_layernorm.weight",
    "ffn.w_gate.weight": "mlp.gate_proj.weight",
    "ffn.w_up.weight": "mlp.up_proj.weight",
    "ffn.w_down.weight": "mlp.down_proj.weight",
}


def zero_to_hf_name(name: str) -> str:
    """zero 参数名 → HF 参数名。"""
    if name == "tok_emb.weight":
        return "model.embed_tokens.weight"
    if name == "norm.weight":
        return "model.norm.weight"
    if name == "lm_head.weight":
        return "lm_head.weight"
    if name.startswith("layers."):
        _, idx, rest = name.split(".", 2)
        if rest in _LAYER_MAP:
            return f"model.layers.{idx}.{_LAYER_MAP[rest]}"
    raise KeyError(f"无法映射的参数名：{name}")


# ---------------------------------------------------------------------------
# 配置互转
# ---------------------------------------------------------------------------


def config_to_hf_qwen3(config: ModelConfig, **extra: Any) -> dict[str, Any]:
    """ModelConfig → Qwen3 的 config.json 内容（dict）。

    同时写新旧两种 RoPE 字段：transformers>=5 读 `rope_parameters`，旧版本读 `rope_theta`/`rope_scaling`。
    """
    rope_parameters: dict[str, Any] = {
        "rope_type": "default",
        "rope_theta": float(config.rope_theta),
    }
    rope_scaling = None
    if config.rope_scaling is not None:
        rs = {k: v for k, v in config.rope_scaling.items() if k != "type"}
        rope_parameters = {"rope_type": "yarn", "rope_theta": float(config.rope_theta), **rs}
        rope_scaling = {"rope_type": "yarn", "type": "yarn", **rs}
    d: dict[str, Any] = {
        "architectures": ["Qwen3ForCausalLM"],
        "model_type": "qwen3",
        "vocab_size": config.vocab_size,
        "hidden_size": config.dim,
        "intermediate_size": config.ffn_dim,
        "num_hidden_layers": config.n_layers,
        "num_attention_heads": config.n_heads,
        "num_key_value_heads": config.n_kv_heads,
        "head_dim": config.head_dim,
        "hidden_act": "silu",
        "max_position_embeddings": config.max_seq_len,
        "initializer_range": config.init_std,
        "rms_norm_eps": config.norm_eps,
        "tie_word_embeddings": config.tie_embeddings,
        "rope_theta": float(config.rope_theta),
        "rope_scaling": rope_scaling,
        "rope_parameters": rope_parameters,
        "attention_bias": False,
        "attention_dropout": 0.0,
        "use_sliding_window": False,
        "sliding_window": None,
        "max_window_layers": config.n_layers,
        "use_cache": True,
    }
    d.update(extra)
    return d


def config_from_hf_qwen3(hf_config: Any) -> ModelConfig:
    """HF 的 Qwen3Config（对象或 dict）→ ModelConfig。"""
    get = (
        hf_config.get
        if isinstance(hf_config, dict)
        else (lambda k, default=None: getattr(hf_config, k, default))
    )
    rope = get("rope_parameters") or {}
    rope_scaling_old = get("rope_scaling") or {}
    theta = rope.get("rope_theta", get("rope_theta", 10000.0))
    rope_type = (
        rope.get("rope_type")
        or rope_scaling_old.get("rope_type")
        or rope_scaling_old.get("type")
        or "default"
    )
    scaling = None
    if rope_type == "yarn":
        src = rope if rope.get("rope_type") == "yarn" else rope_scaling_old
        scaling = {"type": "yarn"}
        for k in (
            "factor",
            "original_max_position_embeddings",
            "beta_fast",
            "beta_slow",
            "attention_factor",
        ):
            if src.get(k) is not None:
                scaling[k] = src[k]
    elif rope_type != "default":
        raise ValueError(f"暂不支持的 rope_type：{rope_type}")
    if get("attention_bias", False):
        raise ValueError("zero 的模型没有 bias，不能加载 attention_bias=True 的模型")
    return ModelConfig(
        vocab_size=get("vocab_size"),
        dim=get("hidden_size"),
        n_layers=get("num_hidden_layers"),
        n_heads=get("num_attention_heads"),
        n_kv_heads=get("num_key_value_heads") or get("num_attention_heads"),
        head_dim=get("head_dim") or get("hidden_size") // get("num_attention_heads"),
        ffn_dim=get("intermediate_size"),
        rope_theta=float(theta),
        rope_scaling=scaling,
        max_seq_len=get("max_position_embeddings"),
        norm_eps=get("rms_norm_eps", 1e-6),
        qk_norm=True,
        tie_embeddings=bool(get("tie_word_embeddings", False)),
        init_std=get("initializer_range", 0.02),
    )


# ---------------------------------------------------------------------------
# 加载
# ---------------------------------------------------------------------------


def _read_hf_dir(path: Path) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    from safetensors.torch import load_file

    with open(path / "config.json") as f:
        cfg = json.load(f)
    sd: dict[str, torch.Tensor] = {}
    files = sorted(path.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"{path} 里没有 .safetensors 文件")
    for fp in files:
        sd.update(load_file(str(fp)))
    return sd, cfg


def load_from_hf_qwen3(
    hf_model_or_state_dict: Any, config: ModelConfig | None = None, strict: bool = True
) -> Transformer:
    """从 HF Qwen3 加载权重，返回 `zero.Transformer`。

    参数可以是：`Qwen3ForCausalLM` 对象、它的 state_dict、或一个导出的目录路径（含 config.json 与 safetensors）。
    传 state_dict 时必须同时给 config。
    """
    if isinstance(hf_model_or_state_dict, str | os.PathLike):
        sd, hf_cfg = _read_hf_dir(Path(hf_model_or_state_dict))
        config = config or config_from_hf_qwen3(hf_cfg)
    elif isinstance(hf_model_or_state_dict, dict):
        sd = hf_model_or_state_dict
        if config is None:
            raise ValueError("传 state_dict 时需要同时传 config")
    else:
        sd = hf_model_or_state_dict.state_dict()
        config = config or config_from_hf_qwen3(hf_model_or_state_dict.config)

    model = Transformer(config)
    new_sd = {}
    for name in model.state_dict():
        if name == "lm_head.weight" and config.tie_embeddings:
            continue  # 共享权重：跟着 tok_emb 走
        hf_name = zero_to_hf_name(name)
        if hf_name not in sd:
            if name == "lm_head.weight" and "model.embed_tokens.weight" in sd:
                new_sd[name] = sd["model.embed_tokens.weight"]
                continue
            raise KeyError(f"HF 权重里缺少 {hf_name}（对应 zero 的 {name}）")
        new_sd[name] = sd[hf_name]
    missing, unexpected = model.load_state_dict(new_sd, strict=False)
    missing = [m for m in missing if not (m == "lm_head.weight" and config.tie_embeddings)]
    if strict and (missing or unexpected):
        raise RuntimeError(f"加载不完整：missing={missing} unexpected={unexpected}")
    return model


# ---------------------------------------------------------------------------
# 导出
# ---------------------------------------------------------------------------


def export_to_hf_qwen3(
    model: Transformer,
    config: ModelConfig | None,
    out_dir: str | os.PathLike,
    tokenizer: Tokenizer | None = None,
    dtype: torch.dtype = torch.bfloat16,
) -> Path:
    """导出成 HF 目录：config.json、generation_config.json、model.safetensors，以及（给了分词器时）
    tokenizer.json / tokenizer_config.json / special_tokens_map.json。返回目录路径。"""
    from safetensors.torch import save_file

    config = config or model.config
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    tensors: dict[str, torch.Tensor] = {}
    for name, tensor in model.state_dict().items():
        if name == "lm_head.weight" and config.tie_embeddings:
            continue  # safetensors 不允许共享存储；HF 通过 tie_word_embeddings=true 自动绑回去
        tensors[zero_to_hf_name(name)] = tensor.detach().to("cpu", dtype).contiguous()
    save_file(tensors, str(out / "model.safetensors"), metadata={"format": "pt"})

    extra: dict[str, Any] = {
        "torch_dtype": str(dtype).removeprefix("torch."),
        "dtype": str(dtype).removeprefix("torch."),
    }
    gen: dict[str, Any] = {"do_sample": True, "temperature": 0.7, "top_p": 0.8}
    if tokenizer is not None:
        eos = tokenizer.eos_id
        extra.update({"eos_token_id": eos, "bos_token_id": None, "pad_token_id": tokenizer.eot_id})
        gen.update({"eos_token_id": eos, "pad_token_id": tokenizer.eot_id})
    with open(out / "config.json", "w") as f:
        json.dump(config_to_hf_qwen3(config, **extra), f, indent=2, ensure_ascii=False)
    with open(out / "generation_config.json", "w") as f:
        json.dump(gen, f, indent=2)
    if tokenizer is not None:
        tokenizer.save_hf(out, model_max_length=config.max_seq_len)
    return out
