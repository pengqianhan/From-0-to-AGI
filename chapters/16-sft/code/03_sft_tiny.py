"""第 16 章 · 极简代码 3：把一个只会续写的小底座，SFT 成会发工具调用的模型

底座：第 10 章在莎士比亚上训练的字符级小模型（0.86M 参数；没有缓存就先花一两分钟训练它）。
步骤：
  1. 看底座怎么"回答"问题：它只会续写，不会回答；
  2. 扩词表：加上 { } " _ 数字等新字符和 4 个特殊 token（新行用旧 embedding 的平均值初始化）；
  3. 全参数 SFT：同样的数据、同样的步数，一次只在助手 token 上算 loss（有 mask），一次所有 token
     都算（无 mask）；
  4. 在 50 道**没见过的城市和数字**上贪心生成，统计格式对 / 函数名对 / 参数全对的比例；
  5. 附加：只给 40 条数据训同样多步（约 60 个 epoch），看过拟合。

所有模型只在这里训练一次，权重缓存在 code/out/（*.pt 已被 .gitignore 忽略）。
运行：uv run python chapters/16-sft/code/03_sft_tiny.py
（首次要训练 3 个模型，每个 300 步；构建机负载很重时共用了约 17 分钟，空闲的笔记本上会快得多。
之后模型从缓存加载，只剩生成和判分。）
"""

from __future__ import annotations

import importlib.util
import json
import math
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

torch.set_num_threads(1)  # 构建环境多任务共享 CPU；读者本机可以删掉这行
HERE = Path(__file__).resolve().parent
OUT = HERE / "out"


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


ch10 = load("ch10_tiny_model", HERE.parents[1] / "10-inference" / "code" / "01_tiny_model.py")
lm = load("ch16_loss_mask", HERE / "02_loss_mask.py")

STEPS, BSZ, LR, WARMUP = 300, 8, 1e-3, 20
N_TRAIN, N_SMALL, N_TEST = 2000, 40, 50


# ── 底座 + 扩词表 ────────────────────────────────────────────────────────────
def base_model():
    return ch10.load_or_train(4)  # 第 10 章的 MHA 小模型（有缓存直接加载）


def extend_vocab(model, tok: lm.ChatTok):
    """共享 embedding 的词表从 65 扩到 len(tok.itos)。新 token 的向量 = 旧向量的平均（+ 一点噪声），
    这样初始时新 token 的 logits 不会离谱地大或小。"""
    old = model.emb.weight.data
    g = torch.Generator().manual_seed(0)
    new_rows = old.mean(0, keepdim=True) + 0.01 * torch.randn(len(tok.itos) - len(old), old.shape[1],
                                                              generator=g)
    model.emb = torch.nn.Embedding(len(tok.itos), old.shape[1])
    model.emb.weight.data = torch.cat([old, new_rows])
    model.c.vocab_size = len(tok.itos)
    return model


# ── 数据 ────────────────────────────────────────────────────────────────────
def dataset(n: int, seed: int, split: str):
    rng = random.Random(seed)
    return [lm.make_example(rng, split) for _ in range(n)]


def batchify(tok, examples, use_mask: bool):
    """右侧补齐成一个 batch；补齐位置和不算 loss 的位置，目标都是 -100。"""
    rows = [lm.encode_with_mask(tok, ex["messages"]) for ex in examples]
    L = max(len(ids) for ids, _ in rows) - 1
    x = torch.zeros(len(rows), L, dtype=torch.long)
    y = torch.full((len(rows), L), -100, dtype=torch.long)
    for i, (ids, mask) in enumerate(rows):
        xi, yi = lm.masked_targets(ids, mask, use_mask)
        x[i, : len(xi)], y[i, : len(yi)] = xi, yi
    return x, y


@torch.no_grad()
def assistant_loss(model, tok, examples) -> float:
    """验证指标：不管训练时用没用 mask，都只在助手 token 上算（两种训练才可比）。"""
    x, y = batchify(tok, examples, use_mask=True)
    return F.cross_entropy(model(x).flatten(0, 1), y.flatten(), ignore_index=-100).item()


# ── 训练 ────────────────────────────────────────────────────────────────────
def sft(tok, train_set, val_set, use_mask: bool, steps: int = STEPS, seed: int = 0, log=print):
    torch.manual_seed(seed)
    model = extend_vocab(base_model(), tok).train()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0)  # 步数少，不做权重衰减
    rng = random.Random(seed)
    order: list[int] = []
    hist = []
    t0 = time.time()
    for step in range(1, steps + 1):
        lr = LR * min(1.0, step / WARMUP) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / steps)))
        for pg in opt.param_groups:
            pg["lr"] = lr
        if len(order) < BSZ:  # 一轮（epoch）用完就重新打乱
            order += rng.sample(range(len(train_set)), len(train_set))
        idx, order = order[:BSZ], order[BSZ:]
        x, y = batchify(tok, [train_set[i] for i in idx], use_mask)
        loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten(), ignore_index=-100)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 50 == 0 or step == 1:
            model.eval()
            v = assistant_loss(model, tok, val_set)
            model.train()
            hist.append({"step": step, "train_loss": round(loss.item(), 4), "val_asst_loss": round(v, 4)})
            log(f"    step {step:4d}  训练 loss {loss.item():.3f}  验证（助手 token）{v:.3f}"
                f"  ({time.time() - t0:.0f}s)")
    return model.eval(), hist


# ── 生成与判分 ──────────────────────────────────────────────────────────────
@torch.no_grad()
def reply(model, tok, messages, max_new: int = 64) -> str:
    """贪心生成助手回复，遇到 <|im_end|> 停。"""
    ids, _ = lm.encode_with_mask(tok, messages, add_generation_prompt=True)
    cache = ch10.KVCache(model.c.n_layers)  # 第 10 章的 KV cache：提示词算一次，之后每步只算新 token
    logits = model(torch.tensor([ids]), cache)
    out = []
    for _ in range(max_new):
        nxt = int(logits[0, -1].argmax())
        if nxt == tok.im_end:
            break
        out.append(nxt)
        logits = model(torch.tensor([[nxt]]), cache)
    return tok.decode(out)


def score(text: str, gold: dict) -> dict:
    calls = []
    try:
        calls = lm.tmpl.parse_tool_calls(text)
    except json.JSONDecodeError:
        pass
    ok = len(calls) == 1 and isinstance(calls[0], dict)
    return {"format": ok, "name": ok and calls[0].get("name") == gold["name"],
            "args": ok and calls[0].get("name") == gold["name"]
            and calls[0].get("arguments") == gold["arguments"]}


def evaluate(model, tok, test_set) -> dict:
    s = {"format": 0, "name": 0, "args": 0}
    for ex in test_set:
        r = score(reply(model, tok, ex["messages"][:2]), ex["call"])
        for k in s:
            s[k] += r[k]
    return {k: v / len(test_set) for k, v in s.items()}


@torch.no_grad()
def continue_text(model, tok, text: str, n: int = 60) -> str:
    ids = tok.encode(text)
    cache = ch10.KVCache(model.c.n_layers)
    logits, out = model(torch.tensor([ids]), cache), []
    for _ in range(n):
        out.append(int(logits[0, -1][: tok.n_base].argmax()))  # 底座只认得旧字符
        logits = model(torch.tensor([[out[-1]]]), cache)
    return tok.decode(out)


def get_or_train(name: str, tok, train_set, val_set, use_mask: bool, log=print):
    path = OUT / f"ch16_{name}.pt"
    hist_path = OUT / f"ch16_{name}.json"
    if path.exists() and hist_path.exists():
        model = extend_vocab(base_model(), tok)
        model.load_state_dict(torch.load(path, weights_only=True))
        return model.eval(), json.loads(hist_path.read_text())
    log(f"  训练 {name}（{STEPS} 步，batch {BSZ}，lr {LR}，只需一次，之后从 {path.name} 加载）")
    model, hist = sft(tok, train_set, val_set, use_mask, log=log)
    OUT.mkdir(exist_ok=True)
    torch.save(model.state_dict(), path)
    hist_path.write_text(json.dumps(hist))
    return model, hist


def run(log=print) -> dict:
    """做全部实验，返回视频和正文要用的数字。"""
    tok = lm.ChatTok(ch10.CharData().chars)
    train_set, val_set = dataset(N_TRAIN, 0, "train"), dataset(64, 1, "train")
    test_set = dataset(N_TEST, 2, "test")  # 没见过的城市、随机数字
    base = base_model()
    demo = next(ex for ex in test_set if ex["call"]["name"] == "get_weather")  # 问题里没有数字，底座认得每个字符
    q = demo["messages"][1]["content"]
    res = {"n_params": sum(p.numel() for p in base.parameters()), "vocab_base": tok.n_base,
           "vocab_new": len(tok.itos), "question": q,
           "base_continue": continue_text(base, tok, q + "\n")}
    base_x = extend_vocab(base_model(), tok).eval()
    res["base_reply"] = reply(base_x, tok, demo["messages"][:2], max_new=60)
    res["base_eval"] = evaluate(base_x, tok, test_set[:20])
    res["base_val_asst_loss"] = assistant_loss(base_x, tok, val_set)
    # 同一批数据上，"只算助手"与"全部都算"两种 loss 各是多少（底座、扩词表后）
    x, y_all = batchify(tok, val_set, use_mask=False)
    res["base_val_all_loss"] = F.cross_entropy(base_x(x).flatten(0, 1), y_all.flatten(),
                                               ignore_index=-100).item()
    for name, use_mask, data in (("masked", True, train_set), ("unmasked", False, train_set),
                                 ("small40", True, train_set[:N_SMALL])):
        model, hist = get_or_train(name, tok, data, val_set, use_mask, log)
        res[name] = {"hist": hist, "eval": evaluate(model, tok, test_set),
                     "val_asst_loss": assistant_loss(model, tok, val_set)}
        if name == "masked":
            res["masked"]["eval_seen"] = evaluate(model, tok, dataset(N_TEST, 3, "train"))
            res["examples"] = [{"q": ex["messages"][1]["content"],
                                "reply": reply(model, tok, ex["messages"][:2])}
                               for ex in test_set[:4]]
    return res


if __name__ == "__main__":
    t0 = time.time()
    r = run()
    print(f"\n底座：{r['n_params'] / 1e6:.2f}M 参数，词表 {r['vocab_base']} → {r['vocab_new']}")
    print(f"问题：{r['question']!r}")
    print(f"底座直接续写（不套模板）：{r['base_continue']!r}")
    print(f"底座套上对话模板后的'回答'：{r['base_reply']!r}")
    print(f"底座在 20 道测试题上：{r['base_eval']}")
    print(f"底座在验证集上的 loss：只算助手 token {r['base_val_asst_loss']:.3f}，"
          f"全部 token {r['base_val_all_loss']:.3f}")
    print("\nSFT 之后（有 mask）的回答：")
    for e in r["examples"]:
        print(f"  {e['q']!r:40s} → {e['reply']!r}")
    print(f"\n{N_TEST} 道测试题（城市都没在训练里出现过）：")
    print(f"{'设置':24s} {'格式对':>6s} {'函数名对':>8s} {'参数全对':>8s} {'验证 loss（助手）':>16s}")
    for name, label in (("masked", "有 mask，2000 条"), ("unmasked", "无 mask，2000 条"),
                        ("small40", "有 mask，只有 40 条")):
        e = r[name]["eval"]
        print(f"{label:24s} {e['format']:6.2f} {e['name']:8.2f} {e['args']:8.2f} "
              f"{r[name]['val_asst_loss']:16.3f}")
    e = r["masked"]["eval_seen"]
    print(f"对照：有 mask 的模型换成训练里见过的城市（数字仍随机）：格式 {e['format']:.2f}，"
          f"函数名 {e['name']:.2f}，参数全对 {e['args']:.2f}")
    print(f"\n{N_SMALL} 条数据训 {STEPS} 步（约 {STEPS * BSZ // N_SMALL} 个 epoch）的曲线：")
    for h in r["small40"]["hist"]:
        print(f"  step {h['step']:4d}  训练 loss {h['train_loss']:.3f}  验证 loss {h['val_asst_loss']:.3f}")
    print(f"\n总耗时 {time.time() - t0:.0f}s")
