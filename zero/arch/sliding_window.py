"""滑动窗口注意力与局部-全局交替（对应第 22 章）。第五部分的实验模块，**不用于主线模型**。

三样东西：

- `sliding_window_mask(...)`：布尔掩码。全局位置 p 的查询只能看见位置 j 满足
  `p - window < j <= p` 的键（window=None 表示普通因果掩码）。和 Hugging Face
  （Mistral / Gemma / gpt-oss 的 `sliding_window` 字段）的约定一致：窗口**包含自己**，
  每个 token 最多看见 window 个位置。
- `make_layer_types(n_layers, global_every)`：生成每层的类型列表，取值与 HF `config.layer_types`
  相同（"sliding_attention" / "full_attention"）。`global_every=6` 就是 Gemma 3 的 5 局部 : 1 全局，
  `global_every=2` 是 Gemma 2 / gpt-oss 的 1:1 交替，`global_every=4` 是 OLMo 3 的 3:1。
- `SlidingWindowAttention`：继承 `zero.model.Attention`（参数名完全相同，可以直接搬权重），
  只改掩码；配套的 `SlidingWindowKVCache` 对滑动窗口层只保留最近 window 个位置（环形缓冲区），
  所以这些层的 KV cache 不随序列长度增长。

用法：

    model = Transformer(cfg)
    layer_types = make_layer_types(cfg.n_layers, global_every=4)
    convert_to_sliding_window(model, layer_types, window=128)   # 原地替换每层的注意力
    cache = SlidingWindowKVCache.from_model(model, batch_size=1, max_seq_len=4096)
    out = generate_greedy(model, prompt, 100, cache=cache)

在 CPU 上用 PyTorch SDPA + 自定义布尔掩码实现。生产推理的对应物（均**尚未在 GPU 上验证**）：
FlashAttention 2/3 的 `flash_attn_func(..., causal=True, window_size=(window - 1, 0))`；
PyTorch FlexAttention 的 `sliding_window` mask_mod；vLLM / transformers 读取 `sliding_window` 与
`layer_types` 后自动选择对应 kernel 和分层缓存。`tests/test_arch_sliding_window.py` 负责对拍。
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F

from zero.config import ModelConfig
from zero.kv_cache import KVCache
from zero.model import Attention, Transformer, apply_rope

SLIDING = "sliding_attention"
FULL = "full_attention"


# ---------------------------------------------------------------------------
# 掩码与层类型
# ---------------------------------------------------------------------------


def sliding_window_mask(
    q_pos: torch.Tensor, k_pos: torch.Tensor, window: int | None
) -> torch.Tensor:
    """q_pos: (Tq,) 查询的全局位置；k_pos: (Tk,) 键的全局位置（可以乱序，负数表示空槽）。

    返回 (Tq, Tk) 的布尔掩码，True = 可以看见。条件：`0 <= k <= q` 且（有窗口时）`q - k < window`。
    """
    q = q_pos[:, None]
    k = k_pos[None, :]
    mask = (k >= 0) & (k <= q)
    if window is not None:
        mask &= (q - k) < window
    return mask


def make_layer_types(
    n_layers: int, global_every: int | None = None, all_sliding: bool = False
) -> list[str]:
    """每层是滑动窗口还是全注意力。

    - `global_every=k`：第 k、2k、3k……层（从 1 数）是全注意力，其余是滑动窗口，
      与 HF Gemma 3 的 `(i + 1) % sliding_window_pattern == 0` 约定一致；
    - `all_sliding=True`：全部是滑动窗口（Mistral 7B v0.1 的做法）；
    - 都不给：全部是全注意力。
    """
    if all_sliding:
        return [SLIDING] * n_layers
    if global_every is None:
        return [FULL] * n_layers
    if global_every < 1:
        raise ValueError(f"global_every 必须 >= 1，当前 {global_every}")
    return [FULL if (i + 1) % global_every == 0 else SLIDING for i in range(n_layers)]


# ---------------------------------------------------------------------------
# KV cache：滑动窗口层用环形缓冲区
# ---------------------------------------------------------------------------


class SlidingWindowKVCache:
    """分层的 KV cache：全注意力层按 max_seq_len 预分配，滑动窗口层只分配 window 个槽位。

    环形缓冲区：位置 p 的 K/V 写进槽位 `p % window`，覆盖掉 window 步之前的旧值。
    因为 K 在写入前已经做过 RoPE（位置信息已经"转"进向量里），注意力对键的排列顺序不敏感，
    所以槽位乱序没关系；只需要额外记住每个槽位现在存的是哪个位置（`pos`，空槽为 -1），用来构造掩码。

    `update` 的返回值比 `zero.kv_cache.KVCache` 多一个 `k_pos`（每个键的全局位置）。
    """

    def __init__(
        self,
        layer_types: Sequence[str],
        window: int,
        batch_size: int,
        max_seq_len: int,
        n_kv_heads: int,
        head_dim: int,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.layer_types = list(layer_types)
        self.window = window
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.k: list[torch.Tensor] = []
        self.v: list[torch.Tensor] = []
        self.pos: list[torch.Tensor] = []
        for t in self.layer_types:
            slots = min(window, max_seq_len) if t == SLIDING else max_seq_len
            shape = (batch_size, n_kv_heads, slots, head_dim)
            self.k.append(torch.zeros(shape, device=device, dtype=dtype))
            self.v.append(torch.zeros(shape, device=device, dtype=dtype))
            self.pos.append(torch.full((slots,), -1, dtype=torch.long, device=device))

    @classmethod
    def from_model(
        cls,
        model: Transformer,
        batch_size: int,
        max_seq_len: int | None = None,
        dtype: torch.dtype | None = None,
    ) -> SlidingWindowKVCache:
        layer_types, window = get_layer_types(model)
        cfg = model.config
        assert cfg.head_dim is not None
        param = next(model.parameters())
        return cls(
            layer_types,
            window or (max_seq_len or cfg.max_seq_len),
            batch_size,
            max_seq_len or cfg.max_seq_len,
            cfg.n_kv_heads,
            cfg.head_dim,
            device=param.device,
            dtype=dtype or param.dtype,
        )

    def update(
        self, layer_idx: int, start_pos: int, k: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """写入位置 [start_pos, start_pos+T) 的新 K/V，返回 (K, V, k_pos) 供本次注意力使用。

        滑动窗口层：返回"缓存里的旧值 + 这次的新值"（新值还没写进环形缓冲区之前先拼上，
        这样一次喂进超过 window 个 token 的分块 prefill 也正确），然后只把最后 window 个写回。
        """
        bsz, _, t, _ = k.shape
        end = start_pos + t
        if end > self.max_seq_len:
            raise ValueError(f"KV cache 溢出：需要 {end} 个位置，max_seq_len={self.max_seq_len}")
        new_pos = torch.arange(start_pos, end, device=k.device)
        ck, cv, cpos = self.k[layer_idx], self.v[layer_idx], self.pos[layer_idx]
        k = k.to(ck.dtype)
        v = v.to(cv.dtype)
        if self.layer_types[layer_idx] == FULL:
            ck[:bsz, :, start_pos:end] = k
            cv[:bsz, :, start_pos:end] = v
            cpos[start_pos:end] = new_pos
            return ck[:bsz, :, :end], cv[:bsz, :, :end], cpos[:end]

        slots = ck.shape[2]
        valid = cpos >= 0
        k_all = torch.cat([ck[:bsz][:, :, valid], k], dim=2)
        v_all = torch.cat([cv[:bsz][:, :, valid], v], dim=2)
        pos_all = torch.cat([cpos[valid], new_pos])
        keep = slice(max(0, t - slots), t)  # 只有最后 slots 个新位置需要留下来
        idx = new_pos[keep] % slots
        ck[:bsz, :, idx] = k[:, :, keep]
        cv[:bsz, :, idx] = v[:, :, keep]
        cpos[idx] = new_pos[keep]
        return k_all, v_all, pos_all

    def nbytes(self) -> int:
        """K 和 V 合计占用的字节数（滑动窗口层只有 window 个槽位）。"""
        return sum(t.numel() * t.element_size() for t in self.k + self.v)


# ---------------------------------------------------------------------------
# 注意力
# ---------------------------------------------------------------------------


class SlidingWindowAttention(Attention):
    """与 `zero.model.Attention` 相同的参数（wq/wk/wv/wo/q_norm/k_norm），只多一个窗口。

    window=None 时就是普通的全注意力层（局部-全局交替模型里的"全局层"）。
    """

    def __init__(self, config: ModelConfig, window: int | None) -> None:
        super().__init__(config)
        self.window = window

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: SlidingWindowKVCache | KVCache | None = None,
        layer_idx: int = 0,
        start_pos: int = 0,
    ) -> torch.Tensor:
        bsz, seqlen, _ = x.shape
        q = self.q_norm(self.wq(x).view(bsz, seqlen, self.n_heads, self.head_dim)).transpose(1, 2)
        k = self.k_norm(self.wk(x).view(bsz, seqlen, self.n_kv_heads, self.head_dim))
        k = k.transpose(1, 2)
        v = self.wv(x).view(bsz, seqlen, self.n_kv_heads, self.head_dim).transpose(1, 2)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        q_pos = torch.arange(start_pos, start_pos + seqlen, device=x.device)
        if isinstance(kv_cache, SlidingWindowKVCache):
            k, v, k_pos = kv_cache.update(layer_idx, start_pos, k, v)
        elif kv_cache is not None:  # 普通 KVCache：键的位置就是 0..kv_len-1
            k, v = kv_cache.update(layer_idx, start_pos, k, v)
            k_pos = torch.arange(k.shape[2], device=x.device)
        else:
            k_pos = q_pos
        k, v = k.to(q.dtype), v.to(q.dtype)

        mask = sliding_window_mask(q_pos, k_pos, self.window)
        # 自定义布尔掩码走 SDPA 的通用路径（CUDA 上能跑，但带掩码用不了 Flash 内核，RTX 3090 上训练吞吐只有
        # 同尺寸全注意力的一半左右，2026-10）；GPU 上用 FlashAttention 的 window_size 或
        # FlexAttention 才能真正跳过窗口外的块（尚未在 GPU 上验证）
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, enable_gqa=self.n_kv_heads != self.n_heads
        )
        out = out.transpose(1, 2).reshape(bsz, seqlen, self.n_heads * self.head_dim)
        return self.wo(out)


def convert_to_sliding_window(
    model: Transformer, layer_types: Sequence[str], window: int
) -> Transformer:
    """原地把 model 每一层的注意力换成 SlidingWindowAttention（保留原有权重），返回 model。"""
    if len(layer_types) != len(model.layers):
        raise ValueError(f"layer_types 有 {len(layer_types)} 项，模型有 {len(model.layers)} 层")
    if window < 1:
        raise ValueError(f"window 必须 >= 1，当前 {window}")
    for block, t in zip(model.layers, layer_types, strict=True):
        if t not in (SLIDING, FULL):
            raise ValueError(f"未知的层类型 {t!r}，可用 {SLIDING!r} / {FULL!r}")
        old = block.attn
        new = SlidingWindowAttention(model.config, window if t == SLIDING else None)
        new.load_state_dict(old.state_dict())
        new.to(device=next(old.parameters()).device, dtype=next(old.parameters()).dtype)
        block.attn = new
    return model


def get_layer_types(model: Transformer) -> tuple[list[str], int | None]:
    """读出每层的类型和（滑动窗口层的）窗口大小；没转换过的层算全注意力。"""
    types, window = [], None
    for block in model.layers:
        w = getattr(block.attn, "window", None)
        types.append(SLIDING if w is not None else FULL)
        window = w if w is not None else window
    return types, window


# ---------------------------------------------------------------------------
# 生成（贪心；用来对拍"有界缓存"和"不用缓存"）
# ---------------------------------------------------------------------------


@torch.no_grad()
def generate_greedy(
    model: Transformer,
    prompt_ids: Sequence[int],
    max_new_tokens: int,
    cache: SlidingWindowKVCache | None = None,
) -> list[int]:
    """贪心生成 max_new_tokens 个新 token。cache=None 时每步把整段序列重新过一遍模型。"""
    was_training = model.training
    model.eval()
    try:
        device = next(model.parameters()).device
        seq = torch.tensor([list(prompt_ids)], dtype=torch.long, device=device)
        out: list[int] = []
        pos, nxt_in = 0, seq
        for _ in range(max_new_tokens):
            if cache is not None:
                logits = model(nxt_in, kv_cache=cache, start_pos=pos)  # type: ignore[arg-type]
                pos += nxt_in.shape[1]
            else:
                logits = model(seq)
            nxt = logits[:, -1].argmax(-1, keepdim=True)
            out.append(int(nxt))
            seq = torch.cat([seq, nxt], dim=1)
            nxt_in = nxt
        return out
    finally:
        model.train(was_training)


def kv_cache_bytes(
    layer_types: Sequence[str],
    window: int,
    seq_len: int,
    n_kv_heads: int,
    head_dim: int,
    bytes_per_elem: int = 2,
    batch_size: int = 1,
) -> int:
    """KV cache 的理论字节数：全注意力层存 seq_len 个位置，滑动窗口层存 min(window, seq_len) 个。"""
    per_pos = 2 * n_kv_heads * head_dim * bytes_per_elem * batch_size
    positions = sum(seq_len if t == FULL else min(window, seq_len) for t in layer_types)
    return positions * per_pos
