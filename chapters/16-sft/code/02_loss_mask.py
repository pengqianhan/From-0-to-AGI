"""第 16 章 · 极简代码 2：loss mask —— 只在助手说的话上算 loss

三件事：
  1. 一个玩具"工具调用"任务 + 字符级分词器（特殊 token 各占一个 id），第 3、4 个脚本也用它；
  2. 渲染一条对话，得到逐 token 的 mask：助手 token 为 1，其余为 0；打印出来看看；
  3. 在同一批数据上对比"只算助手 token"与"全部 token 都算"的 loss，
     并演示 loss 怎么在不同长度的样本之间平均（按 token 平均 vs 按样本平均）。

运行：uv run python chapters/16-sft/code/02_loss_mask.py
"""

from __future__ import annotations

import importlib.util
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


tmpl = load("ch16_chat_template", HERE / "01_chat_template.py")  # 第 1 个脚本里的 render()

# ── 玩具任务：两个工具，问题模板 + 城市 / 数字 ─────────────────────────────────
SYSTEM = "Tools: get_weather(city), add(a, b)"
TRAIN_CITIES = ["Paris", "London", "Tokyo", "Berlin", "Madrid", "Rome", "Cairo", "Dublin",
                "Oslo", "Vienna", "Prague", "Lisbon", "Athens", "Warsaw", "Seoul", "Delhi",
                "Lima", "Quito", "Sydney", "Boston", "Denver", "Austin", "Miami", "Chicago",
                "Toronto", "Munich", "Milan", "Naples", "Geneva", "Zurich"]
TEST_CITIES = ["Venice", "Hanoi", "Dallas", "Nairobi", "Bogota", "Havana", "Kyoto", "Porto",
               "Helsinki", "Manila"]  # 训练时从没出现过：考的是"照抄"能力，而不是背城市名
WEATHER_Q = ["What is the weather in {c}?", "How is the weather in {c} today?",
             "Is it raining in {c}?", "Weather for {c}, please."]
ADD_Q = ["What is {a} plus {b}?", "Add {a} and {b}.", "What is {a} + {b}?", "Compute {a}+{b}."]


def make_example(rng: random.Random, split: str = "train") -> dict:
    """一条 (system, user, assistant) 对话，助手的回复是一次工具调用。"""
    if rng.random() < 0.5:
        c = rng.choice(TRAIN_CITIES if split == "train" else TEST_CITIES)
        q, call = rng.choice(WEATHER_Q).format(c=c), {"name": "get_weather", "arguments": {"city": c}}
    else:
        a, b = rng.randint(0, 99), rng.randint(0, 99)
        q, call = rng.choice(ADD_Q).format(a=a, b=b), {"name": "add", "arguments": {"a": a, "b": b}}
    return {"messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": q},
                         {"role": "assistant", "content": "", "tool_calls": [call]}],
            "call": call}


# ── 字符级分词器：第 10 章的莎士比亚字符表 + 新字符 + 特殊 token ─────────────────
SPECIALS = ["<|im_start|>", "<|im_end|>", "<tool_call>", "</tool_call>"]
_SPECIAL_RE = "(" + "|".join(s.replace("|", r"\|") for s in SPECIALS) + ")"


class ChatTok:
    """base_chars：底座模型的词表（第 10 章：莎士比亚里出现过的 65 个字符）。
    SFT 数据里有底座没见过的字符（{ } " _ 数字……）和特殊 token，追加在词表末尾。"""

    def __init__(self, base_chars: list[str]) -> None:
        import re

        self._re = re
        extra = sorted({ch for ch in '{}"_<>/|0123456789+()' if ch not in base_chars})
        self.itos = list(base_chars) + extra + SPECIALS
        self.stoi = {s: i for i, s in enumerate(self.itos)}
        self.n_base = len(base_chars)
        self.im_end = self.stoi["<|im_end|>"]

    def encode(self, text: str) -> list[int]:
        ids = []
        for part in self._re.split(_SPECIAL_RE, text):
            if part in self.stoi and part in SPECIALS:
                ids.append(self.stoi[part])  # 特殊 token 整体一个 id，不会被拆开
            else:
                ids.extend(self.stoi[ch] for ch in part)
        return ids

    def decode(self, ids) -> str:
        return "".join(self.itos[i] for i in ids)


def encode_with_mask(tok: ChatTok, messages, add_generation_prompt: bool = False):
    """渲染 → (ids, mask)。逐段编码：mask 直接跟着段走（字符级分词，段边界不会合并）。"""
    ids, mask = [], []
    for text, _role, train in tmpl.render(messages, add_generation_prompt=add_generation_prompt):
        t = tok.encode(text)
        ids += t
        mask += [train] * len(t)
    return ids, mask


def shakespeare_chars() -> list[str]:
    text = (HERE.parents[2] / "assets" / "tiny_corpus" / "shakespeare.txt").read_text("utf-8")
    return sorted(set(text))


def masked_targets(ids: list[int], mask: list[bool], use_mask: bool = True):
    """x = ids[:-1]，y = ids[1:]；y 里不算 loss 的位置换成 -100（cross_entropy 的 ignore_index）。
    注意是 mask[1:]：第 t 个位置预测的是第 t+1 个 token，要不要算 loss 看的是"被预测的那个 token"。"""
    x = torch.tensor(ids[:-1])
    y = torch.tensor(ids[1:])
    if use_mask:
        y = torch.where(torch.tensor(mask[1:]), y, torch.full_like(y, -100))
    return x, y


if __name__ == "__main__":
    tok = ChatTok(shakespeare_chars())
    print(f"词表：底座 {tok.n_base} 个字符 + 新字符 {len(tok.itos) - tok.n_base - len(SPECIALS)} 个"
          f" + 特殊 token {len(SPECIALS)} 个 = {len(tok.itos)}")
    ex = make_example(random.Random(0))
    ids, mask = encode_with_mask(tok, ex["messages"])
    print("\n一条玩具样本（[方括号] 里是算 loss 的 token）：")
    out, inside = "", False
    for i, m in zip(ids, mask):
        if m and not inside:
            out += "["
        if not m and inside:
            out += "]"
        inside = m
        out += tok.decode([i])
    print(out + ("]" if inside else ""))
    print(f"共 {len(ids)} 个 token，算 loss 的 {sum(mask)} 个（{100 * sum(mask) / len(ids):.0f}%）")

    # 真实样本（第 1 个脚本里那条工具对话）按字符数算：助手只占一小部分
    segs = tmpl.render(tmpl.MESSAGES, tmpl.TOOLS)
    n = sum(len(s) for s, _, _ in segs)
    k = sum(len(s) for s, _, t in segs if t)
    print(f"\n第 1 个脚本里的真实工具对话：{n} 个字符，算 loss 的 {k} 个（{100 * k / n:.1f}%）")

    # ── 同一批数据，两种 loss ───────────────────────────────────────────────
    # 模型用一个随机初始化的 embedding + 线性层代替（只为演示算法，不需要训练）
    torch.manual_seed(0)
    V = len(tok.itos)
    emb, head = torch.nn.Embedding(V, 16), torch.nn.Linear(16, V)
    rng = random.Random(1)
    batch = [encode_with_mask(tok, make_example(rng)["messages"]) for _ in range(4)]
    tot_all, tot_asst, n_all, n_asst = 0.0, 0.0, 0, 0
    for ids, mask in batch:
        x, y_full = masked_targets(ids, mask, use_mask=False)
        _, y_mask = masked_targets(ids, mask, use_mask=True)
        logits = head(emb(x))
        nll = F.cross_entropy(logits, y_full, reduction="none")  # 每个位置的 -log p
        keep = y_mask != -100
        tot_all, n_all = tot_all + nll.sum().item(), n_all + len(nll)
        tot_asst, n_asst = tot_asst + nll[keep].sum().item(), n_asst + int(keep.sum())
        # 用 ignore_index 算出来的，就是只在 keep 位置上的平均
        assert torch.allclose(F.cross_entropy(logits, y_mask, ignore_index=-100), nll[keep].mean())
    print(f"\n随机初始化的小模型，4 条样本：全部 token 的平均 loss {tot_all / n_all:.3f}"
          f"（{n_all} 个位置），只算助手 token {tot_asst / n_asst:.3f}（{n_asst} 个位置）")
    print("ignore_index=-100 的结果与手算的'只在助手位置上平均'一致 ✓")

    # ── 长短不一的样本怎么平均 ──────────────────────────────────────────────
    # 样本 A 助手有 3 个 token、每个 loss 2.0；样本 B 有 30 个、每个 loss 1.0
    a, b = torch.full((3,), 2.0), torch.full((30,), 1.0)
    per_token = torch.cat([a, b]).mean()
    per_sample = (a.mean() + b.mean()) / 2
    print(f"\n按 token 平均：{per_token:.3f}；先各自平均再按样本平均：{per_sample:.3f}"
          "（梯度累积时如果每个 micro-batch 各自求平均，就会从前者悄悄变成后者）")
