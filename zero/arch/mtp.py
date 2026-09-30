"""多 token 预测 MTP（Multi-Token Prediction，对应第 25 章）。实验模块，不用于主线模型。

DeepSeek-V3（arXiv:2412.19437 第 2.2 节）的做法：在主模型后面串接 D 个**顺序**的 MTP 模块，
第 k 个模块预测"再往后第 k 个" token，并保留完整的因果链：

    h'^k_i = M_k [RMSNorm(h^{k-1}_i) ; RMSNorm(Emb(t_{i+k}))]   # 上一层的表示 + 第 i+k 个 token 的 embedding
    h^k_{1:T-k} = TRM_k(h'^k_{1:T-k})                           # 一个 Transformer block（因果注意力）
    P^k_{i+k+1} = OutHead(h^k_i)                                # 与主模型共享的输出头
    L_MTP = λ / D · Σ_k CE(P^k, t)                               # 加在主损失上

- Emb 和 OutHead 与主模型**共享**（不增加词表大小的参数）；k = 1 时 h^0 是主模型的表示。
- 训练：每个位置除了"下一个 token"还多学"下下个 token"，训练信号更密（densify）；
  DeepSeek-V3 的消融（表 4）显示主模型在多数基准上变好。推理时可以直接丢掉 MTP 模块。
- 推理：也可以把 MTP 模块当草稿做**自推测解码**（self-speculative decoding）：主模型给出下一个
  token 的同时，MTP 模块猜下下个，下一次前向一起验证（`mtp_speculative_generate`）。
  DeepSeek-V3 报告第二个 token 的接受率 85%–90%，解码 TPS 提到 1.8 倍（第 5.4.3 节）。

实现细节与 vLLM / SGLang 的 DeepSeek MTP 推理实现对齐（DeepSeek 的训练代码没有开源）：
- 拼接顺序是 [enorm(embedding) ; hnorm(hidden)]，投影 `eh_proj`（即论文的 M_k）；
- 送进 MTP 的 h^0 是主模型**最后一个 RMSNorm 之后**的隐藏状态（也就是送进 lm_head 的那个）；
- 每个 MTP 模块有自己的最终 RMSNorm（vLLM 里叫 shared_head.norm），然后乘共享的 lm_head。
主模型用 `zero.model.Transformer` 原样不动；MTP block 就是 `zero.model.Block`（GQA + QK-Norm + RoPE + SwiGLU）。
本文件只追求可读和正确，尚未在 GPU 上验证性能。
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from zero.arch.speculative import SpeculativeResult, rollback, verify, warp_probs
from zero.config import ModelConfig
from zero.kv_cache import KVCache
from zero.model import Block, RMSNorm, Transformer, cross_entropy_loss


class MTPModule(nn.Module):
    """第 k 个 MTP 模块：enorm / hnorm → eh_proj(2d → d) → 一个 Transformer block → 自己的最终 norm。"""

    def __init__(self, config: ModelConfig, index: int = 0) -> None:
        super().__init__()
        self.enorm = RMSNorm(config.dim, config.norm_eps)
        self.hnorm = RMSNorm(config.dim, config.norm_eps)
        self.eh_proj = nn.Linear(2 * config.dim, config.dim, bias=False)
        # layer_idx = index：MTP 的 KV cache 单独一份（`MTPTransformer.new_mtp_cache`），第 index 层属于它
        self.block = Block(config, index)
        self.norm = RMSNorm(config.dim, config.norm_eps)

    def forward(
        self,
        h_prev: torch.Tensor,
        emb_next: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: KVCache | None = None,
        start_pos: int = 0,
    ) -> torch.Tensor:
        """h_prev, emb_next: (B, T, d) → 本层输出（已过最终 norm，直接乘 lm_head 得 logits），形状 (B, T, d)。"""
        x = self.eh_proj(torch.cat([self.enorm(emb_next), self.hnorm(h_prev)], dim=-1))
        x = self.block(x, cos, sin, kv_cache, start_pos)
        return self.norm(x)


class MTPTransformer(nn.Module):
    """主模型 `Transformer` + D 个顺序 MTP 模块（共享 tok_emb 和 lm_head）。"""

    def __init__(self, config: ModelConfig, n_mtp: int = 1) -> None:
        super().__init__()
        self.config = config
        self.model = Transformer(config)
        self.mtp = nn.ModuleList([MTPModule(config, i) for i in range(n_mtp)])
        self._init_mtp()

    @torch.no_grad()
    def _init_mtp(self) -> None:
        std = self.config.init_std
        for mod in self.mtp:
            for name, p in mod.named_parameters():
                if name.endswith("weight") and p.dim() == 2:
                    nn.init.normal_(p, mean=0.0, std=std)
                else:
                    nn.init.ones_(p)  # RMSNorm 权重

    @property
    def n_mtp(self) -> int:
        return len(self.mtp)

    def new_mtp_cache(self, max_seq_len: int, device=None, dtype=torch.float32) -> KVCache:
        assert self.config.head_dim is not None
        return KVCache(
            n_layers=self.n_mtp,
            batch_size=1,
            max_seq_len=max_seq_len,
            n_kv_heads=self.config.n_kv_heads,
            head_dim=self.config.head_dim,
            device=device or next(self.parameters()).device,
            dtype=dtype,
        )

    def hidden(
        self, tokens: torch.Tensor, kv_cache: KVCache | None = None, start_pos: int = 0
    ) -> torch.Tensor:
        """主模型最后一个 RMSNorm 之后的隐藏状态 (B, T, d)；lm_head(hidden) 就是主模型的 logits。
        与 `Transformer.forward` 同一套计算，只是少乘 lm_head。"""
        m = self.model
        cos, sin = m.rope(start_pos, tokens.shape[1])
        h = m.tok_emb(tokens)
        for layer in m.layers:
            h = layer(h, cos, sin, kv_cache, start_pos)
        return m.norm(h)

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """训练用前向。tokens: (B, T) → (主模型 logits (B, T, V), [第 k 个 MTP 的 logits (B, T−k, V)])。

        第 k 个 MTP 在位置 i（0 ≤ i < T−k）用 h^{k-1}_i 和 Emb(tokens[i+k])，预测 tokens[i+k+1]
        （即 targets[i+k]，targets 是 tokens 左移一位）。"""
        m = self.model
        T = tokens.shape[1]
        h = self.hidden(tokens)
        logits = m.lm_head(h)
        mtp_logits = []
        for k, mod in enumerate(self.mtp, start=1):
            if T - k <= 0:
                break
            cos, sin = m.rope(0, T - k)
            h = mod(h[:, : T - k], m.tok_emb(tokens[:, k:]), cos, sin)
            mtp_logits.append(m.lm_head(h))
        return logits, mtp_logits


def mtp_loss(
    model: MTPTransformer,
    tokens: torch.Tensor,
    targets: torch.Tensor,
    lam: float = 0.3,
    ignore_index: int = -100,
) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
    """L = L_main + λ/D · Σ_k L_k（DeepSeek-V3 式 (25)；V3 预训练前 10T token 用 λ = 0.3，之后 0.1）。

    targets[:, i] 是 tokens[:, i] 的下一个 token；第 k 个 MTP 的目标是 targets[:, k:]。
    返回 (总损失, 主损失, [各深度的 MTP 损失])。"""
    logits, mtp_logits = model(tokens)
    main = cross_entropy_loss(logits, targets, ignore_index)
    losses = [
        cross_entropy_loss(lg, targets[:, k:], ignore_index)
        for k, lg in enumerate(mtp_logits, start=1)
    ]
    total = main
    if losses:
        total = main + lam / len(losses) * torch.stack(losses).sum()
    return total, main, losses


@torch.no_grad()
def mtp_speculative_generate(
    model: MTPTransformer,
    prompt_ids: Sequence[int] | torch.Tensor,
    max_new_tokens: int,
    temperature: float = 0.0,
    top_p: float = 1.0,
    seed: int | None = None,
    eos_id: int | None = None,
) -> SpeculativeResult:
    """用第 1 个 MTP 模块当草稿做自推测解码（每轮 1 个草稿，和 DeepSeek-V3 的用法相同）。batch = 1。

    状态：主模型缓存有效到 t_len；MTP 缓存有效到 m_len（MTP 在位置 i 需要 h_i 和 seq[i+1]，
    所以总比主模型慢一步）；pending 保存还没喂给 MTP 的主模型隐藏状态。
    每一轮：
      1. MTP 把"已确定、还没处理"的位置补上，最后一个位置的输出就是对 seq[L] 之后那个 token 的草稿 d；
      2. 主模型一次喂 [seq[L−1], d]，得到两行 logits：验证 d，并多给一个 token；
      3. 被拒时主模型缓存回滚一格；MTP 缓存里只有已确定的位置，不用回滚。
    """
    if model.n_mtp < 1:
        raise ValueError("模型没有 MTP 模块")
    prompt = (
        prompt_ids.flatten().tolist() if isinstance(prompt_ids, torch.Tensor) else list(prompt_ids)
    )
    m, mtp = model.model, model.mtp[0]
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    max_len = min(len(prompt) + max_new_tokens + 2, model.config.max_seq_len)
    max_new_tokens = min(max_new_tokens, max_len - len(prompt) - 1)
    gen = None
    if seed is not None:
        gen = torch.Generator(device=device)
        gen.manual_seed(seed)

    was_training = model.training
    model.eval()
    try:
        t_cache = KVCache.from_config(model.config, 1, max_len, device, dtype)
        m_cache = model.new_mtp_cache(max_len, device, dtype)
        # prefill：主模型处理整段提示词，得到下一个 token
        h = model.hidden(torch.tensor([prompt], device=device), t_cache, 0)[0]  # (L, d)
        first = verify(m.lm_head(h[-1:]), [], None, temperature, top_p, gen)[1]
        seq = prompt + first
        t_len, m_len = len(prompt), 0
        pending = h  # 位置 m_len.. 的主模型隐藏状态
        res = SpeculativeResult(tokens=[])
        while len(seq) - len(prompt) < max_new_tokens and not (
            eos_id is not None and eos_id in seq[len(prompt) :]
        ):
            L = len(seq)
            # ── 1. MTP 补上位置 m_len..L-2（每个位置配上它的下一个 token），最后一个位置给出草稿 ──
            n_feed = L - 1 - m_len
            emb = m.tok_emb(torch.tensor([seq[m_len + 1 : L]], device=device))
            cos, sin = m.rope(m_len, n_feed)
            h_mtp = mtp(pending[None, :n_feed], emb, cos, sin, m_cache, m_len)
            m_len = L - 1
            q_logits = m.lm_head(h_mtp[0, -1])
            if temperature <= 0:
                d, q_probs = int(q_logits.argmax()), None
            else:
                q = warp_probs(q_logits, temperature, top_p)
                d, q_probs = int(torch.multinomial(q, 1, generator=gen)), q[None]
            drafts = [d] if L - len(prompt) < max_new_tokens else []  # 最后一个 token 不必猜
            # ── 2. 主模型一次前向：[seq[L-1], d] ──
            h_new = model.hidden(
                torch.tensor([[seq[L - 1]] + drafts], device=device), t_cache, t_len
            )[0]
            acc, new = verify(
                m.lm_head(h_new), drafts, q_probs if drafts else None, temperature, top_p, gen
            )
            # ── 3. 记账、回滚 ──
            res.rounds += 1
            res.proposed += len(drafts)
            res.accepted += acc
            res.examined += len(drafts)
            res.accepted_per_round.append(acc)
            seq += new
            t_len = len(seq) - 1
            rollback(t_cache, t_len)
            pending = h_new[: 1 + acc]  # 位置 L-1（以及被接受时的 L）的隐藏状态，下一轮喂给 MTP
    finally:
        model.train(was_training)

    out = seq[len(prompt) :][:max_new_tokens]
    if eos_id is not None and eos_id in out:
        out = out[: out.index(eos_id)]
    res.tokens = out
    return res
