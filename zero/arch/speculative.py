"""推测解码（speculative decoding，对应第 25 章）。实验模块，不用于主线模型的训练。

decode 阶段一次只喂 1 个 token，算力吃不饱、时间花在读权重和 KV cache 上（第 10、21 章），
所以目标模型一次前向验证 k+1 个位置，和生成 1 个 token 的时间差不多。推测解码利用这一点：

1. 便宜的草稿（draft）自回归地提出 k 个 token；
2. 目标模型（target）把它们一次喂进去，得到 k+1 个位置的分布 p_1..p_{k+1}；
3. 从左往右逐个验证草稿 x_i（草稿分布 q_i）：
   - 贪心（temperature <= 0）：x_i == argmax p_i 就接受；
   - 采样：以概率 min(1, p_i(x_i) / q_i(x_i)) 接受；被拒时从残差分布
     norm(max(0, p_i − q_i)) 重抽一个，本轮结束；
   k 个全部接受时，再从 p_{k+1} 抽一个"奖励" token；
4. 被拒绝的草稿已经写进了两个模型的 KV cache：回滚（只需把"有效长度"退回去——
   `zero.kv_cache.KVCache.update` 按 start_pos 覆盖写入、只返回 [0, start_pos+T)，
   后面残留的旧数据不会被读到）。

**质量不变**：贪心时输出与目标模型自己贪心解码逐字相同；采样时每个 token 的分布恰好是目标模型
（经过同样的 temperature / top-p 处理后）的分布（Leviathan 等 2023 附录 A.1；Chen 等 2023）。
一次接受的概率 α = Σ_x min(p(x), q(x))。

草稿有两种来源：
- 另一个同词表的小模型（`draft=Transformer`）；
- `draft=None`：提示词查找（prompt lookup / n-gram）——在已有文本里找和末尾 n 个 token 相同的片段，
  把它后面的 token 当草稿；草稿分布是 one-hot，采样时接受概率就是 p(x)。
MTP 模块当草稿（自推测）见 `zero/arch/mtp.py` 的 `mtp_speculative_generate`，复用本文件的 `verify`。

只支持 batch = 1：批量推测解码要处理每条序列接受个数不同带来的"参差"，vLLM / SGLang 在调度器里做这件事。
本文件只追求可读和正确，尚未在 GPU 上验证性能。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import torch

from zero.kv_cache import KVCache
from zero.model import Transformer


@dataclass
class SpeculativeResult:
    """生成结果和接受统计。"""

    tokens: list[int]  # 新 token（不含提示词；遇到 eos 时不含 eos）
    rounds: int = 0  # 目标模型的验证前向次数（不含 prefill）
    proposed: int = 0  # 草稿一共提出的 token 数
    accepted: int = 0  # 被接受的草稿数
    examined: int = 0  # 真正被比较过的草稿数（第一个被拒之后的草稿不算）
    accepted_per_round: list[int] = field(default_factory=list)

    @property
    def acceptance_rate(self) -> float:
        """逐 token 接受率 α 的估计：接受数 / 被比较数（在 α 独立同分布的假设下是最大似然估计）。"""
        return self.accepted / self.examined if self.examined else 0.0

    @property
    def tokens_per_round(self) -> float:
        """目标模型每次前向平均产出几个 token（普通解码是 1）。"""
        return (self.accepted + self.rounds) / self.rounds if self.rounds else 0.0


def expected_tokens_per_round(alpha: float, k: int) -> float:
    """Leviathan 等 2023 的公式 (1)：(1 − α^{k+1}) / (1 − α)。"""
    if alpha >= 1.0:
        return float(k + 1)
    return (1 - alpha ** (k + 1)) / (1 - alpha)


def expected_speedup(alpha: float, k: int, c: float) -> float:
    """Leviathan 等 2023 的定理 3.8：(1 − α^{k+1}) / ((1 − α)(k·c + 1))，c = 草稿一步 / 目标一步的耗时。
    假设目标模型验证 k+1 个位置和生成 1 个 token 一样快。"""
    return expected_tokens_per_round(alpha, k) / (k * c + 1)


def warp_probs(logits: torch.Tensor, temperature: float, top_p: float = 1.0) -> torch.Tensor:
    """logits (..., V) → 经过 temperature 和 top-p 处理后的概率（与 `zero.generate.sample_next` 同一个分布）。

    推测解码的保证是针对"处理后的目标分布"的：p 和 q 要用同样的处理。"""
    probs = torch.softmax(logits.float() / temperature, dim=-1)
    if top_p < 1.0:
        sorted_probs, idx = torch.sort(probs, dim=-1, descending=True)
        keep_sorted = (torch.cumsum(sorted_probs, dim=-1) - sorted_probs) < top_p
        keep = torch.zeros_like(keep_sorted).scatter(-1, idx, keep_sorted)
        probs = probs * keep
        probs = probs / probs.sum(dim=-1, keepdim=True)
    return probs


def verify(
    p_logits: torch.Tensor,
    drafts: Sequence[int],
    q_probs: torch.Tensor | None,
    temperature: float,
    top_p: float = 1.0,
    generator: torch.Generator | None = None,
) -> tuple[int, list[int]]:
    """一轮验证。

    p_logits: (k+1, V) 目标模型在"每个草稿位置 + 最后一个位置"的 logits；
    drafts:   k 个草稿 token；
    q_probs:  (k, V) 草稿分布（已经过同样的 temperature / top-p）；None 表示草稿是确定性的（one-hot）。
    返回 (接受个数 m, 本轮确定的新 token：m 个草稿 + 1 个纠正或奖励 token)。
    """
    k = len(drafts)
    if temperature <= 0:
        choice = p_logits.argmax(dim=-1).tolist()
        m = 0
        while m < k and drafts[m] == choice[m]:
            m += 1
        return m, list(drafts[:m]) + [choice[m]]

    p = warp_probs(p_logits, temperature, top_p)
    for i, x in enumerate(drafts):
        if q_probs is None:
            q_i = torch.zeros_like(p[i])
            q_i[x] = 1.0
        else:
            q_i = q_probs[i].float()
        ratio = p[i, x] / q_i[x] if q_i[x] > 0 else torch.tensor(0.0)
        if torch.rand((), generator=generator, device=p.device) < torch.clamp(ratio, max=1.0):
            continue  # 接受
        residual = torch.clamp(p[i] - q_i, min=0.0)
        if residual.sum() <= 0:  # p == q（数值上）时残差为 0；此时拒绝本不会发生，退回按 p 抽
            residual = p[i]
        tok = int(torch.multinomial(residual / residual.sum(), 1, generator=generator))
        return i, list(drafts[:i]) + [tok]
    bonus = int(torch.multinomial(p[k], 1, generator=generator))
    return k, list(drafts) + [bonus]


def prompt_lookup_draft(seq: Sequence[int], k: int, max_ngram: int = 3) -> list[int]:
    """提示词查找（n-gram）草稿：在 seq 里从后往前找与末尾 n 个 token 相同的片段（n 从大到小），
    返回它后面最多 k 个 token；找不到返回空列表。代价几乎为零（c ≈ 0）。"""
    L = len(seq)
    for n in range(min(max_ngram, L - 1), 0, -1):
        tail = list(seq[L - n :])
        for start in range(L - n - 1, -1, -1):  # 最近的匹配优先
            if list(seq[start : start + n]) == tail:
                cont = list(seq[start + n : start + n + k])
                if cont:
                    return cont
    return []


def rollback(cache: KVCache, length: int) -> None:
    """KV cache 回滚到前 length 个位置。预分配缓存不需要清数据，只记录有效长度。"""
    cache.seq_len = min(cache.seq_len, length)


@torch.no_grad()
def speculative_generate(
    target: Transformer,
    draft: Transformer | None,
    prompt_ids: Sequence[int] | torch.Tensor,
    max_new_tokens: int,
    k: int = 4,
    temperature: float = 0.0,
    top_p: float = 1.0,
    seed: int | None = None,
    eos_id: int | None = None,
    max_ngram: int = 3,
) -> SpeculativeResult:
    """用草稿模型（或 draft=None 时用提示词查找）给目标模型做推测解码。batch = 1。

    temperature <= 0：贪心，输出与 `zero.generate.generate(target, ..., temperature=0)` 逐字相同。
    temperature > 0：采样，每个 token 的分布与目标模型（同样 temperature / top-p）的分布相同，
    但具体抽到的序列与 `generate` 不同（随机数的用法不同）。
    """
    if k < 1:
        raise ValueError("k 至少为 1")
    if draft is not None and draft.config.vocab_size != target.config.vocab_size:
        raise ValueError("草稿模型和目标模型必须共用同一个词表")
    prompt = (
        prompt_ids.flatten().tolist() if isinstance(prompt_ids, torch.Tensor) else list(prompt_ids)
    )
    if not prompt:
        raise ValueError("提示词不能为空")
    device = next(target.parameters()).device
    dtype = next(target.parameters()).dtype
    max_len = min(len(prompt) + max_new_tokens + k + 1, target.config.max_seq_len)
    if draft is not None:
        max_len = min(max_len, draft.config.max_seq_len)
    max_new_tokens = min(max_new_tokens, max_len - len(prompt) - 1)
    gen = None
    if seed is not None:
        gen = torch.Generator(device=device)
        gen.manual_seed(seed)

    t_cache = KVCache.from_config(target.config, 1, max_len, device, dtype)
    d_cache = KVCache.from_config(draft.config, 1, max_len, device, dtype) if draft else None
    t_len = d_len = 0  # 两个缓存里有效的位置数
    seq = list(prompt)
    res = SpeculativeResult(tokens=[])
    models = [target] + ([draft] if draft is not None else [])
    was_training = [m.training for m in models]
    for m in models:
        m.eval()
    try:
        while len(seq) - len(prompt) < max_new_tokens:
            remaining = max_new_tokens - (len(seq) - len(prompt))
            kk = min(k, remaining - 1, max_len - len(seq) - 1)  # 留出纠正/奖励 token 的位置

            # ── 1. 草稿提出 kk 个 token ──
            drafts: list[int] = []
            q_rows: list[torch.Tensor] = []
            if kk > 0 and draft is not None:
                assert d_cache is not None
                inp = torch.tensor([seq[d_len:]], device=device)
                logits = draft(inp, kv_cache=d_cache, start_pos=d_len)[0, -1]
                d_len = len(seq)
                for i in range(kk):
                    if temperature <= 0:
                        x = int(logits.argmax())
                    else:
                        q = warp_probs(logits, temperature, top_p)
                        q_rows.append(q)
                        x = int(torch.multinomial(q, 1, generator=gen))
                    drafts.append(x)
                    if i < kk - 1:
                        logits = draft(
                            torch.tensor([[x]], device=device), kv_cache=d_cache, start_pos=d_len
                        )[0, -1]
                        d_len += 1
            elif kk > 0:
                drafts = prompt_lookup_draft(seq, kk, max_ngram)

            # ── 2. 目标模型一次前向验证 ──
            n = len(drafts)
            inp = torch.tensor([seq[t_len:] + drafts], device=device)
            p_logits = target(inp, kv_cache=t_cache, start_pos=t_len)[0, -(n + 1) :]
            q_probs = torch.stack(q_rows) if q_rows else None
            m, new = verify(p_logits, drafts, q_probs, temperature, top_p, gen)

            # ── 3. 记账、接上新 token、回滚缓存 ──
            res.rounds += 1
            res.proposed += n
            res.accepted += m
            res.examined += m + (1 if m < n else 0)
            res.accepted_per_round.append(m)
            seq += new
            t_len = len(seq) - 1  # 最后一个新 token 还没进缓存
            rollback(t_cache, t_len)
            if d_cache is not None:
                d_len = min(d_len, len(seq) - 1)
                rollback(d_cache, d_len)
            if eos_id is not None and eos_id in new:
                break
    finally:
        for mdl, was in zip(models, was_training):
            mdl.train(was)

    out = seq[len(prompt) :][:max_new_tokens]
    if eos_id is not None and eos_id in out:
        out = out[: out.index(eos_id)]
    res.tokens = out
    return res
