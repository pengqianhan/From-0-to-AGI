# 第 22 章：局部与稀疏注意力 —— 只看附近，也不丢掉远处

> **一句话目标**：读完这一章，你能写出滑动窗口注意力的掩码和"有上限"的 KV cache，算出 L 层滑动窗口的感受野；能按 `config.json` 里的 `layer_types` 和 `sliding_window` 说清楚 Gemma、gpt-oss、OLMo 3 这些模型把多少层换成了局部、省了多少 KV cache；还能用一个小实验指出纯滑动窗口在哪里失效、插一层全局注意力为什么能修好它，以及稀疏注意力"按内容挑键"和滑动窗口"按位置挑键"的区别。

📺 **本章视频**：待发布（本地渲染：`bash chapters/22-local-sparse-attention/video/build.sh`）
🧪 **本章自检**：学完后在 Claude Code 里输入 `/ch22-local-attention`

---

上一章我们给 KV cache 记了一本账：长上下文贵在两处——prefill 的注意力算力随长度平方增长，decode 时每一步都要把全部历史 K/V 从显存里读一遍。MQA、GQA、MLA 压的是"每个位置存多少"。这一章要解决的问题是：**能不能让一个位置不必看全部历史？** 最直接的办法是只看附近的一段——**滑动窗口注意力（sliding window attention, SWA）**；它的代价是看不到远处，于是有了"大部分层局部、少数层全局"的**局部-全局交替**。再往前一步，与其固定地看"最近的"，不如让模型自己挑"最相关的"几个——这就是**稀疏注意力（sparse attention）**。

本章代码（都在 CPU 上跑）：

```bash
uv run python chapters/22-local-sparse-attention/code/01_masks_and_ledger.py  # 掩码、感受野、4 个公开模型的 KV 账（几秒）
uv run python chapters/22-local-sparse-attention/code/02_swa_model.py         # 训练 6 个小模型（首次约 25 分钟，之后读缓存）
uv run python chapters/22-local-sparse-attention/code/03_compare.py           # 全注意力 / 滑动窗口 / 交替：loss、KV、大海捞针
uv run python chapters/22-local-sparse-attention/code/04_bounded_cache.py     # 截断缓存生成 = 不用缓存生成，缓存大小封顶
uv run python chapters/22-local-sparse-attention/code/05_topk_sparse.py       # 同样 k 个键：按位置挑 vs 按内容挑
```

__BODY__
