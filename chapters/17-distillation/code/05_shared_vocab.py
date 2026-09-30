"""第 17 章 · 极简代码 5：logits 蒸馏为什么要求同一个词表，以及换词表要付出多少参数

1. 两个分词器切同一句话，切出来的 token 个数、边界都不一样：第 t 个位置不是同一个"下一个词"，
   第 v 维 logit 也不是同一个 token——两个分布没法逐位比较。zero 的 kd_loss 遇到形状不同直接报错。
2. 主线模型（configs/main/pretrain.toml：28 层、宽 1280、共享 embedding）用自训的 65,536 词表。
   如果为了能从千问教师做 logits 蒸馏而改用千问的分词器，参数要涨多少？

运行：uv run python chapters/17-distillation/code/05_shared_vocab.py      （约 5 秒）
"""

import dataclasses
import sys
from pathlib import Path

import torch

torch.set_num_threads(1)
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from zero.config import load_config  # noqa: E402
from zero.model import count_params  # noqa: E402
from zero.post.distill import kd_loss  # noqa: E402

# 词表大小（2026-09 读取各模型的 config.json）：Qwen3-0.6B 151,936；Qwen3.5-0.8B 248,320
VOCABS = {"主线（自训 BPE，第 13 章）": 65_536, "Qwen3 分词器": 151_936, "Qwen3.5 分词器": 248_320}
CAP = 800_000_000  # GOAL.md 3.3：主线模型不超过 0.8B


def toy_tokenize(text: str, vocab: list[str]) -> list[str]:
    """最长匹配切词（只为演示"不同词表切法不同"）。"""
    out, i = [], 0
    while i < len(text):
        for L in range(min(4, len(text) - i), 0, -1):
            if text[i:i + L] in vocab or L == 1:
                out.append(text[i:i + L])
                i += L
                break
    return out


def main() -> None:
    text = "今天天气很好"
    a = toy_tokenize(text, ["今天", "天气"])
    b = toy_tokenize(text, ["今天天", "气很", "好"])
    print(f"1) 同一句话「{text}」，两个词表：")
    print(f"   学生词表切成 {len(a)} 个 token：{a}")
    print(f"   教师词表切成 {len(b)} 个 token：{b}")
    print("   位置数不同、边界不同：学生第 2 步要猜'天气'，教师第 2 步在猜'气很'——没有逐位可比的分布。")
    try:
        kd_loss(torch.randn(1, len(a), 65_536), torch.randn(1, len(b), 151_936), torch.ones(1, len(a), dtype=torch.bool))
    except ValueError as e:
        print(f"   zero.post.distill.kd_loss 报错：{e}")

    cfg = load_config(ROOT / "configs" / "main" / "pretrain.toml").model
    print(f"\n2) 主线模型形状不变（{cfg.n_layers} 层、宽 {cfg.dim}、共享 embedding），只换词表：")
    print(f"   {'分词器':<22}{'词表':>9}{'embedding':>14}{'总参数':>14}  ≤ 0.8B?")
    base = None
    for name, v in VOCABS.items():
        c = count_params(dataclasses.replace(cfg, vocab_size=v))
        base = base or c["total"]
        ok = "是" if c["total"] <= CAP else "否"
        extra = "" if c["total"] == base else f"（+{(c['total'] - base) / 1e6:.1f}M）"
        print(f"   {name:<20}{v:>9,}{c['embedding'] / 1e6:>12.1f}M{c['total'] / 1e6:>12.1f}M  {ok} {extra}")


if __name__ == "__main__":
    main()
