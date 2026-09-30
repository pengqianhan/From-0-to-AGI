"""第 10 章 · 极简代码 5：GQA / MQA——让多个查询头共享一组 K/V

同一个小模型（4 个查询头），只改 n_kv_heads：
  4 = MHA（每个查询头有自己的 K/V）；2 = GQA（每 2 个查询头共享一组）；1 = MQA（全部共享一组）。
用同样的种子、同样的数据顺序、同样的 600 步训练，比较：验证 loss、参数量、KV cache 大小、生成速度。
另外用 MHA 换一个随机种子再训一次，看看"只是换个种子" loss 会差多少——小于这个数的差别不能当真。
第一次运行要训练 3 个新模型（单线程约 4–5 分钟），之后从 code/out/ 加载，几十秒跑完。
运行：uv run python chapters/10-inference/code/05_gqa.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


tiny = _load("tiny_model", "01_tiny_model.py")
kvc = _load("kv_cache_demo", "03_kv_cache.py")

VARIANTS = [("MHA", 4), ("GQA", 2), ("MQA", 1)]
GEN = 512


def run():
    data = tiny.CharData()
    prompt = data.encode(kvc.PROMPT)
    rows = []
    for name, kv in VARIANTS:
        model = tiny.load_or_train(kv)
        attn = sum(p.numel() for n, p in model.named_parameters() if ".attn." in n)
        total = sum(p.numel() for p in model.parameters())
        out, _, cache = kvc.generate_cached(model, prompt, GEN, temperature=0)
        t = kvc.best_time(lambda m=model: kvc.generate_cached(m, prompt, GEN, temperature=0))
        rows.append(dict(name=name, kv=kv, val=tiny.val_loss(model), attn=attn, total=total,
                         cache=cache.nbytes(), time=t, sample=data.decode(out[:60])))
    noise = tiny.val_loss(tiny.load_or_train(4, seed=1))
    return rows, noise


if __name__ == "__main__":
    rows, noise = run()
    mha = rows[0]
    print(f"4 个查询头，head_dim 32，训练 600 步；KV cache 在生成 {GEN} 个字符后测量（FP32）")
    print("  方案  KV头  验证loss  注意力参数  总参数    KV cache       生成512字(秒)")
    for r in rows:
        print(f"  {r['name']}   {r['kv']:3d}   {r['val']:7.3f}   {r['attn']:8,d}  {r['total']:8,d}  "
              f"{r['cache']:9,d} B ({r['cache'] / mha['cache']:.2f}×)   {r['time']:5.2f}")
    print(f"  对照：MHA 换随机种子 1 重训，验证 loss {noise:.3f}"
          f"（和种子 0 差 {abs(noise - mha['val']):.3f}）")
    for r in rows:
        print(f"  {r['name']} 贪心生成开头：{r['sample']!r}")
