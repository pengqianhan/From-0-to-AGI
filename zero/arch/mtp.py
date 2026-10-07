"""Multi-Token Prediction, MTP (Chapter 25). Experiment module; the main-line model does not use it.

The DeepSeek-V3 method (arXiv:2412.19437, Section 2.2): add D **sequential** MTP modules after the
main model. Module k predicts the token "k positions further ahead", and keeps the full causal chain:

    h'^k_i = M_k [RMSNorm(h^{k-1}_i) ; RMSNorm(Emb(t_{i+k}))]   # previous representation + embedding of token i+k
    h^k_{1:T-k} = TRM_k(h'^k_{1:T-k})                           # one Transformer block (causal attention)
    P^k_{i+k+1} = OutHead(h^k_i)                                # output head shared with the main model
    L_MTP = λ / D · Σ_k CE(P^k, t)                               # added to the main loss

- Emb and OutHead are **shared** with the main model (no new vocabulary-size parameters).
  For k = 1, h^0 is the representation of the main model.
- Training: each position learns "the next token" and also "the token after the next". This makes
  the training signal denser. The DeepSeek-V3 ablation (Table 4) shows that the main model improves
  on most benchmarks. At inference, you can remove the MTP modules.
- Inference: you can also use the MTP module as a draft for **self-speculative decoding**:
  when the main model gives the next token, the MTP module guesses the token after it, and the next
  forward pass verifies both (`mtp_speculative_generate`). DeepSeek-V3 reports an acceptance rate
  of 85%–90% for the second token, and 1.8 times the decode TPS (Section 5.4.3).

The implementation details match the DeepSeek MTP inference code of vLLM / SGLang
(DeepSeek did not release its training code):
- The concatenation order is [enorm(embedding) ; hnorm(hidden)], with the projection `eh_proj`
  (the M_k of the paper).
- The h^0 input of the MTP is the main-model hidden state **after the last RMSNorm**
  (the same one that goes into lm_head).
- Each MTP module has its own final RMSNorm (vLLM calls it shared_head.norm), then multiplies by
  the shared lm_head.
The main model is `zero.model.Transformer` without changes. The MTP block is `zero.model.Block`
(GQA + QK-Norm + RoPE + SwiGLU).
This file has only two goals: easy to read and correct. On RTX 3090, the forward / backward parity
check of CUDA against CPU passed, and self-speculative greedy output is identical to the main model
(2026-10, see runs/2026-10-01-gpu0-check/). The performance is not verified on GPU yet.
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
    """MTP module k: enorm / hnorm → eh_proj(2d → d) → one Transformer block → its own final norm."""

    def __init__(self, config: ModelConfig, index: int = 0) -> None:
        super().__init__()
        self.enorm = RMSNorm(config.dim, config.norm_eps)
        self.hnorm = RMSNorm(config.dim, config.norm_eps)
        self.eh_proj = nn.Linear(2 * config.dim, config.dim, bias=False)
        # layer_idx = index: the MTP has a separate KV cache (`MTPTransformer.new_mtp_cache`),
        # and layer `index` of that cache belongs to this module.
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
        """h_prev, emb_next: (B, T, d) → output of this module, shape (B, T, d).

        The output is after the final norm: multiply it by lm_head to get the logits.
        """
        x = self.eh_proj(torch.cat([self.enorm(emb_next), self.hnorm(h_prev)], dim=-1))
        x = self.block(x, cos, sin, kv_cache, start_pos)
        return self.norm(x)


class MTPTransformer(nn.Module):
    """Main model `Transformer` + D sequential MTP modules (shared tok_emb and lm_head)."""

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
                    nn.init.ones_(p)  # RMSNorm weights

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
        """Hidden state (B, T, d) of the main model after the last RMSNorm.

        lm_head(hidden) gives the logits of the main model. The computation is the same as in
        `Transformer.forward`, without the multiplication by lm_head.
        """
        m = self.model
        cos, sin = m.rope(start_pos, tokens.shape[1])
        h = m.tok_emb(tokens)
        for layer in m.layers:
            h = layer(h, cos, sin, kv_cache, start_pos)
        return m.norm(h)

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Forward pass for training.
        tokens: (B, T) → (main-model logits (B, T, V), [logits of MTP k (B, T−k, V)]).

        At position i (0 ≤ i < T−k), MTP k uses h^{k-1}_i and Emb(tokens[i+k]) to predict
        tokens[i+k+1] (that is, targets[i+k]; targets is tokens shifted left by one).
        """
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
    """L = L_main + λ/D · Σ_k L_k (DeepSeek-V3 Eq. (25); V3 pretraining used λ = 0.3 for the
    first 10T tokens, then 0.1).

    targets[:, i] is the next token after tokens[:, i]. The target of MTP k is targets[:, k:].
    Return (total loss, main loss, [MTP loss at each depth]).
    """
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
    """Self-speculative decoding with MTP module 1 as the draft (1 draft per round, as in
    DeepSeek-V3). batch = 1.

    State: the main-model cache is valid up to t_len. The MTP cache is valid up to m_len
    (at position i, the MTP needs h_i and seq[i+1], so it is always one step behind the main model).
    pending holds the main-model hidden states that the MTP did not process yet.
    Each round:
      1. The MTP processes the positions that are "final but not processed yet". The output at the
         last position is the draft d for the token after seq[L].
      2. The main model gets [seq[L−1], d] in one pass and gives two rows of logits:
         they verify d and give one more token.
      3. If d is rejected, roll back the main-model cache by one position. The MTP cache contains
         only final positions, so it needs no rollback.
    """
    if model.n_mtp < 1:
        raise ValueError("The model has no MTP module")
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
        # prefill: the main model processes the full prompt and gives the next token.
        h = model.hidden(torch.tensor([prompt], device=device), t_cache, 0)[0]  # (L, d)
        first = verify(m.lm_head(h[-1:]), [], None, temperature, top_p, gen)[1]
        seq = prompt + first
        t_len, m_len = len(prompt), 0
        pending = h  # main-model hidden states for positions m_len..
        res = SpeculativeResult(tokens=[])
        while len(seq) - len(prompt) < max_new_tokens and not (
            eos_id is not None and eos_id in seq[len(prompt) :]
        ):
            L = len(seq)
            # ── 1. The MTP processes positions m_len..L-2 (each with its next token); the last position gives the draft ──
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
            drafts = [d] if L - len(prompt) < max_new_tokens else []  # no guess is necessary for the last token
            # ── 2. One forward pass of the main model: [seq[L-1], d] ──
            h_new = model.hidden(
                torch.tensor([[seq[L - 1]] + drafts], device=device), t_cache, t_len
            )[0]
            acc, new = verify(
                m.lm_head(h_new), drafts, q_probs if drafts else None, temperature, top_p, gen
            )
            # ── 3. Bookkeeping and rollback ──
            res.rounds += 1
            res.proposed += len(drafts)
            res.accepted += acc
            res.examined += len(drafts)
            res.accepted_per_round.append(acc)
            seq += new
            t_len = len(seq) - 1
            rollback(t_cache, t_len)
            pending = h_new[: 1 + acc]  # hidden states of position L-1 (and L if accepted); the MTP gets them in the next round
    finally:
        model.train(was_training)

    out = seq[len(prompt) :][:max_new_tokens]
    if eos_id is not None and eos_id in out:
        out = out[: out.index(eos_id)]
    res.tokens = out
    return res
