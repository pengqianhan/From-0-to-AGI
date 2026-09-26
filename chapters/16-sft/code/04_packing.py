"""第 16 章 · 极简代码 4：打包（packing）与"串门"问题

1. 打包：把多条长短不一的对话首次适配（first-fit）装进定长窗口，与"一条一行、补齐到窗口长度"比，
   真实 token 占比高多少；
2. 串门：同一个窗口里，后一条对话的 token 在普通因果注意力下能看到前一条对话。用第 3 个脚本 SFT 过的
   小模型，比较对话 B 的 logits：单独算 / 打包在 A 后面（普通因果 mask）/ 打包在 A 后面但加文档 mask
   （document masking：每条对话只看自己）。

运行：uv run python chapters/16-sft/code/04_packing.py   （需要第 3 个脚本训练出的模型，没有会先训练）
"""

from __future__ import annotations

import importlib.util
import math
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

torch.set_num_threads(1)  # 构建环境多任务共享 CPU；读者本机可以删掉这行
HERE = Path(__file__).resolve().parent


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


sft = load("ch16_sft_tiny", HERE / "03_sft_tiny.py")
lm, ch10 = sft.lm, sft.ch10


# ── 1. 首次适配打包 ──────────────────────────────────────────────────────────
def pack_first_fit(lengths: list[int], window: int) -> list[list[int]]:
    """返回每个窗口里装了哪几条（下标）。一条对话不切开；放不下就开新窗口。"""
    bins: list[list[int]] = []
    used: list[int] = []
    for i, n in enumerate(lengths):
        for b in range(len(bins)):
            if used[b] + n <= window:
                bins[b].append(i)
                used[b] += n
                break
        else:
            bins.append([i])
            used.append(n)
    return bins


# ── 2. 带任意注意力 mask 的前向（照抄第 10 章 TinyLM.forward，只把因果 mask 换成传进来的）──
def forward_with_mask(model, ids: torch.Tensor, allow: torch.Tensor) -> torch.Tensor:
    """ids (1, T)；allow (T, T) 布尔矩阵，allow[i, j] = 第 i 个位置能不能看第 j 个位置。"""
    c, T = model.c, ids.shape[1]
    cos, sin = model.cos[:T], model.sin[:T]
    x = model.emb(ids)
    for blk in model.blocks:
        a, h = blk.attn, blk.n1(x)
        q = a.wq(h).view(1, T, c.n_heads, c.head_dim).transpose(1, 2)
        k = a.wk(h).view(1, T, c.n_kv_heads, c.head_dim).transpose(1, 2)
        v = a.wv(h).view(1, T, c.n_kv_heads, c.head_dim).transpose(1, 2)
        q, k = ch10.apply_rope(q, cos, sin), ch10.apply_rope(k, cos, sin)
        att = (q @ k.transpose(-2, -1) / math.sqrt(c.head_dim)).masked_fill(~allow, float("-inf"))
        x = x + a.wo((att.softmax(-1) @ v).transpose(1, 2).reshape(1, T, -1))
        h = blk.n2(x)
        x = x + blk.w_down(F.silu(blk.w_gate(h)) * blk.w_up(h))
    return model.norm(x) @ model.emb.weight.T


def causal(T: int) -> torch.Tensor:
    return torch.ones(T, T, dtype=torch.bool).tril()


def document_mask(doc_ids: torch.Tensor) -> torch.Tensor:
    """因果 + 只能看同一条对话：块对角的下三角。"""
    return causal(len(doc_ids)) & (doc_ids[:, None] == doc_ids[None, :])


if __name__ == "__main__":
    tok = lm.ChatTok(ch10.CharData().chars)
    rng = random.Random(3)
    # 长短不一：一部分样本前面多一轮闲聊（模拟真实数据里的多轮对话）
    convs = []
    for _ in range(200):
        ex = lm.make_example(rng)
        msgs = ex["messages"]
        for _ in range(rng.choice([0, 0, 1, 2])):
            e2 = lm.make_example(rng)
            msgs = msgs[:1] + e2["messages"][1:] + msgs[1:]
        convs.append(lm.encode_with_mask(tok, msgs)[0])
    lengths = [len(c) for c in convs]
    window = 513  # seq_len 512 + 1（与 zero 一样，窗口多存一个 token 用来错位出目标）
    bins = pack_first_fit(lengths, window)
    real = sum(lengths)
    print(f"200 条对话，长度 {min(lengths)}–{max(lengths)} 个 token，合计 {real}")
    print(f"不打包（一条一行，补齐到 {window}）：{len(convs)} 行，真实 token 占 "
          f"{100 * real / (len(convs) * window):.0f}%")
    print(f"首次适配打包：{len(bins)} 个窗口，真实 token 占 {100 * real / (len(bins) * window):.0f}%，"
          f"每个窗口平均装 {len(convs) / len(bins):.1f} 条")

    # ── 串门 ───────────────────────────────────────────────────────────────
    model, _ = sft.get_or_train("masked", tok, sft.dataset(sft.N_TRAIN, 0, "train"),
                                sft.dataset(64, 1, "train"), True)
    a = lm.encode_with_mask(tok, [{"role": "system", "content": lm.SYSTEM},
                                  {"role": "user", "content": "What is the weather in Paris?"},
                                  {"role": "assistant", "content": "", "tool_calls": [
                                      {"name": "get_weather", "arguments": {"city": "Paris"}}]}])
    b = lm.encode_with_mask(tok, [{"role": "system", "content": lm.SYSTEM},
                                  {"role": "user", "content": "Is it raining in Kyoto?"},
                                  {"role": "assistant", "content": "", "tool_calls": [
                                      {"name": "get_weather", "arguments": {"city": "Kyoto"}}]}])
    ids_a, ids_b = a[0], b[0]
    packed = torch.tensor([ids_a + ids_b])
    doc = torch.tensor([0] * len(ids_a) + [1] * len(ids_b))
    with torch.no_grad():
        alone = forward_with_mask(model, torch.tensor([ids_b]), causal(len(ids_b)))[0]
        mixed = forward_with_mask(model, packed, causal(packed.shape[1]))[0, len(ids_a):]
        isolated = forward_with_mask(model, packed, document_mask(doc))[0, len(ids_a):]
        same_as_ch10 = torch.allclose(model(torch.tensor([ids_b]))[0], alone, atol=1e-5)

    def asst_loss(logits):
        y = torch.tensor(ids_b[1:])
        keep = torch.tensor(b[1][1:])
        return F.cross_entropy(logits[:-1][keep], y[keep]).item()

    print(f"\n自己写的带 mask 前向与第 10 章 TinyLM.forward 一致：{same_as_ch10}")
    print("对话 B（Kyoto）的助手 token loss：")
    print(f"  单独一条                         {asst_loss(alone):.4f}")
    print(f"  打包在 A（Paris）后面，普通因果 mask  {asst_loss(mixed):.4f}"
          f"   logits 最大差 {(mixed - alone).abs().max():.2e}")
    print(f"  打包在 A 后面，文档 mask          {asst_loss(isolated):.4f}"
          f"   logits 最大差 {(isolated - alone).abs().max():.2e}")
    print("（RoPE 只看相对位置，所以 B 在窗口里从第 {} 个位置开始也不影响结果——只要注意力被隔开）"
          .format(len(ids_a)))
