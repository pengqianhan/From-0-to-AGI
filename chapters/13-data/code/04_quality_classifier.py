"""基于模型的质量过滤（玩具版）：一个"昂贵的评审员"标一小部分，训练一个便宜的分类器给全部打分。

FineWeb-Edu 的做法：让 Llama-3-70B-Instruct 给约 46 万个网页的"教育价值"打 0–5 分，在这些标注上训练一个
小分类器（Snowflake-arctic-embed-m + 线性回归头），再给全部 15T token 打分，只留 ≥ 3 分的。
DCLM 的做法类似，只是正例换成指令数据和 ELI5 高赞回答、分类器换成 fastText。

这里缩小成：
- "评审员"：我们手里的真实标签（kind 是 good/contaminated 算好，其余算差）。**这一步在真实流程里是
  大模型打分**，我们没有大模型，就用标签代替——这是本脚本唯一"作弊"的地方；
- 只给 250 篇"请评审员打分"，其余文档用分类器预测；
- 分类器：8 个手工特征 + 逻辑回归（NumPy 梯度下降），特征里最有用的是"相邻词对有多眼熟"
  （和 CCNet 用 KenLM 困惑度过滤是一个思路）。

    uv run python chapters/13-data/code/04_quality_classifier.py
"""

from __future__ import annotations

import importlib.util
import math
import random
import re
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


CJK = re.compile(r"[一-鿿]")
FEATURES = ["行尾标点占比", "平均词长", "停用词占比", "不同词占比", "词对眼熟度", "符号占比", "log 长度", "汉字占比"]
GOOD_KINDS = ("good", "contaminated", "exact_dup", "near_dup")
STOP = set("的了是在和有也就不都而与之其以") | {"the", "be", "to", "of", "and", "that", "have", "with",
                                            "i", "you", "my", "a", "in", "is", "it", "not"}


def tokens(text: str) -> list[str]:
    """中文按字、英文按空白切（和启发式规则的"词"一致）。"""
    return re.findall(r"[一-鿿]|[^\s一-鿿]+", text.lower())


def bigrams(text: str) -> list[tuple[str, str]]:
    out = []
    for ln in text.split("\n"):
        t = tokens(ln)
        out += list(zip(t, t[1:]))
    return out


def features(text: str, ref: Counter, own: Counter | None = None) -> list[float]:
    """ref：好文档里每个相邻词对出现的次数；own：这篇文档自己贡献的次数（标注集里的文档要扣掉自己，
    否则"眼熟度"在训练集上虚高，分类器学到的阈值到了新文档上就不对了）。"""
    t = tokens(text)
    n = max(len(t), 1)
    lines = [ln for ln in text.split("\n") if ln.strip()]
    bg = bigrams(text)
    non_space = max(sum(not c.isspace() for c in text), 1)
    return [
        sum(ln.rstrip().endswith(tuple(".!?。！？，,;:；：")) for ln in lines) / max(len(lines), 1),
        sum(map(len, t)) / n,
        sum(w in STOP for w in t) / n,
        len(set(t)) / n,
        sum(ref[x] - (own[x] if own else 0) > 0 for x in bg) / max(len(bg), 1),  # 相邻词对见过没有
        sum(not c.isalnum() and not c.isspace() for c in text) / non_space,
        math.log(len(text) + 1),
        len(CJK.findall(text)) / non_space,
    ]


def train_logreg(X: np.ndarray, y: np.ndarray, lr: float = 0.5, steps: int = 2000, l2: float = 1e-3):
    """逻辑回归：p = sigmoid(X·w + b)，最小化交叉熵（第 5 章），全批量梯度下降。"""
    w, b = np.zeros(X.shape[1]), 0.0
    for _ in range(steps):
        p = 1 / (1 + np.exp(-(X @ w + b)))
        w -= lr * (X.T @ (p - y) / len(y) + l2 * w)
        b -= lr * float(np.mean(p - y))
    return w, b


def classify(docs: list[dict], n_annotate: int = 250, seed: int = 0) -> dict:
    """评审员标 n_annotate 篇 → 训练逻辑回归 → 给全部文档打分。返回分数、标签、权重等。"""
    # 去重后留下的那一份转载，内容本身是好的；差文档 = 乱序文本等"不像话"的页面
    label = np.array([d["kind"] in GOOD_KINDS for d in docs], dtype=float)
    rng = random.Random(seed)
    idx = list(range(len(docs)))
    rng.shuffle(idx)
    ann, rest = idx[:n_annotate], idx[n_annotate:]
    ref = Counter(b for i in ann if label[i] == 1 for b in bigrams(docs[i]["text"]))
    ann_good = {i for i in ann if label[i] == 1}
    X = np.array([features(d["text"], ref, Counter(bigrams(d["text"])) if i in ann_good else None)
                  for i, d in enumerate(docs)])
    mu, sd = X[ann].mean(0), X[ann].std(0) + 1e-9
    Xs = (X - mu) / sd

    w, b = train_logreg(Xs[ann], label[ann])
    p = 1 / (1 + np.exp(-(Xs @ w + b)))
    return {"p": p, "label": label, "ann": ann, "rest": rest, "w": w}


def after_dedup() -> list[dict]:
    """01 → 02 → 03：脏网页经过启发式过滤和去重之后的文档。"""
    crawl_mod = _load("01_noisy_crawl")
    heur = _load("02_heuristic_filter")
    mh = _load("03_minhash")
    docs = crawl_mod.build_crawl()["docs"]
    docs, _, _ = heur.heuristic_filter(docs)
    docs = [docs[i] for i in mh.exact_dedup([d["text"] for d in docs])]
    keep, _ = mh.near_dedup([d["text"] for d in docs])
    return [docs[i] for i in keep]


def quality_filter(docs: list[dict], threshold: float = 0.5) -> list[dict]:
    r = classify(docs)
    return [d for d, p in zip(docs, r["p"]) if p >= threshold]


def main() -> None:
    crawl_mod = _load("01_noisy_crawl")
    docs = after_dedup()
    r = classify(docs)
    p, label, ann, rest, w = r["p"], r["label"], r["ann"], r["rest"], r["w"]
    print(f"启发式 + 去重之后还剩 {len(docs)} 篇，其中差文档 {int((1 - label).sum())} 篇："
          f"{dict(Counter(d['kind'] for d in docs if d['kind'] not in GOOD_KINDS))}")
    print(f"\n评审员标注 {len(ann)} 篇（其中差文档 {int((1 - label[ann]).sum())} 篇），训练逻辑回归。权重（标准化后）：")
    for name, wi in sorted(zip(FEATURES, w), key=lambda x: -abs(x[1])):
        print(f"  {name:<8}{wi:+.2f}")

    rr = np.array(rest)
    print(f"\n在没标注的 {len(rest)} 篇上，换不同的阈值（分数 < 阈值 就删）：")
    print(f"{'阈值':>6}{'抓到差文档':>10}{'漏掉':>6}{'误伤好文档':>10}{'精确率':>8}{'召回率':>8}")
    for th in [0.3, 0.5, 0.7, 0.9]:
        bad = p[rr] < th
        tp = int((bad & (label[rr] == 0)).sum())
        fp = int((bad & (label[rr] == 1)).sum())
        fn = int((~bad & (label[rr] == 0)).sum())
        print(f"{th:>6}{tp:>10}{fn:>6}{fp:>10}{tp / max(tp + fp, 1):>8.2f}{tp / max(tp + fn, 1):>8.2f}")
    pred_bad = p < 0.5

    kept = [d for d, bad in zip(docs, pred_bad) if not bad]
    print(f"\n质量过滤：{len(docs)} → {len(kept)} 篇")
    before, after = Counter(d["kind"] for d in docs), Counter(d["kind"] for d in kept)
    for k in crawl_mod.KINDS:
        if before[k]:
            print(f"  {k:<14}{before[k]:>5} → {after[k]:>5}")
    # 分数最低的一篇
    worst = int(np.argmin(p))
    print(f"\n分数最低的文档（p={p[worst]:.3f}，{docs[worst]['kind']}）："
          + docs[worst]["text"][:90].replace("\n", "⏎"))


if __name__ == "__main__":
    main()
