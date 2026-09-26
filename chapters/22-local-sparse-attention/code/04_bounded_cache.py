"""第 22 章 · 极简代码 4：滑动窗口层的 KV cache 有上限

用 02 训练好的语言模型生成 300 个字符，比较：
  - 不用缓存：每一步把整段序列重新过一遍模型；
  - 用截断缓存：滑动窗口层只保留最近 W 个位置的 K/V（见 02 的 KVCache.append）。
两者生成的文字必须逐字相同；同时打印缓存大小随生成长度的变化。

运行：uv run python chapters/22-local-sparse-attention/code/04_bounded_cache.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

torch.set_num_threads(1)
_spec = importlib.util.spec_from_file_location(
    "swa_model", Path(__file__).resolve().parent / "02_swa_model.py"
)
m = importlib.util.module_from_spec(_spec)
sys.modules["swa_model"] = m  # dataclass 需要能在 sys.modules 里找到所在模块
_spec.loader.exec_module(m)


@torch.no_grad()
def generate(model, ids: list[int], n: int, use_cache: bool, report_at=()):
    ids = list(ids)
    cache = m.KVCache(model.c.n_layers) if use_cache else None
    sizes = {}
    nxt_in, start = torch.tensor([ids]), 0
    for step in range(1, n + 1):
        if use_cache:
            logits = model(nxt_in, cache, start)
            start += nxt_in.shape[1]
        else:
            logits = model(torch.tensor([ids]))
        ids.append(int(logits[0, -1].argmax()))
        nxt_in = torch.tensor([[ids[-1]]])
        if use_cache and start in report_at:  # start = 已经缓存了多少个位置
            sizes[start] = cache.nbytes()
    return ids, sizes


def main() -> None:
    data = m.CharData()
    prompt = data.encode("ROMEO:\n")
    report_at = (16, 64, 128, 256, 306)
    print(f"提示词 'ROMEO:\\n'，贪心生成 300 个字符；窗口 W = {m.W}\n")
    print(f"{'配置':<11}{'与不用缓存逐字相同':>18}   KV cache 字节数（已缓存 " +
          " / ".join(str(t) for t in report_at) + " 个位置）")
    texts = {}
    for variant in m.VARIANTS:
        model = m.load_or_train("lm", variant)
        a, sizes = generate(model, prompt, 300, use_cache=True, report_at=report_at)
        b, _ = generate(model, prompt, 300, use_cache=False)
        texts[variant] = data.decode(a)
        print(f"{variant:<11}{str(a == b):>18}   " + " / ".join(f"{sizes[t]:,}" for t in report_at))
    print("\nsliding 的生成结果（前 200 个字符）：\n" + texts["sliding"][:200])


if __name__ == "__main__":
    main()
