"""造一份"脏网页"：真实文档 + 重复 + 导航页 + 广告 + 乱码 + 乱序文本 + 泄漏的考题。

真实网页数据下载不了（也太大），所以本章用 assets/tiny_corpus 里的莎士比亚和古诗词当"好文档"，
再按网页上常见的几类垃圾，自己往里掺坏东西。每篇文档都带一个标签 kind，后面几个脚本用它来
检查每一步过滤到底删对了没有——真实世界里没有这个标签，这正是数据工作难的地方。

    uv run python chapters/13-data/code/01_noisy_crawl.py
"""

from __future__ import annotations

import random
import re
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
CORPUS = REPO / "assets" / "tiny_corpus"
SEED = 0
DOC_CHARS = 1500


def split_docs(text: str, doc_chars: int = DOC_CHARS) -> list[str]:
    """按空行切段，再把相邻段落拼成约 doc_chars 长的"文档"。"""
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    docs, cur = [], ""
    for p in paras:
        if cur and len(cur) + len(p) > doc_chars:
            docs.append(cur)
            cur = ""
        cur = f"{cur}\n\n{p}" if cur else p
    return docs + ([cur] if cur else [])


def load_real() -> dict[str, list[str]]:
    return {
        "en": split_docs((CORPUS / "shakespeare.txt").read_text("utf-8")),
        "zh": split_docs((CORPUS / "chinese_poetry.txt").read_text("utf-8")),
    }


# ---------------------------------------------------------------------------
# 几类垃圾
# ---------------------------------------------------------------------------

NAV_EN = ["Home", "About Us", "Contact", "Login", "Sign up", "Privacy Policy", "Terms of Use",
          "Cart (0)", "FAQ", "Blog", "Careers", "Sitemap", "Follow us", "Subscribe"]
NAV_ZH = ["首页", "关于我们", "联系方式", "登录", "注册", "隐私政策", "用户协议", "购物车", "帮助中心",
          "新闻中心", "加入我们", "网站地图", "关注我们", "订阅"]
SPAM_EN = ["cheap", "best price", "free shipping", "buy now", "discount", "watches", "casino",
           "bonus", "limited offer", "click here", "top rated", "online"]
SPAM_ZH = ["免费下载", "高清", "在线观看", "最新", "破解版", "优惠", "包邮", "点击进入", "官方正版", "秒杀"]


def nav_page(rng: random.Random, lang: str) -> str:
    items = NAV_ZH if lang == "zh" else NAV_EN
    lines = [" | ".join(rng.sample(items, 5)) for _ in range(rng.randint(6, 12))]
    lines += rng.sample(items, 8)
    lines.append("© 2024 版权所有" if lang == "zh" else "© 2024 All rights reserved")
    return "\n".join(lines)


def spam_page(rng: random.Random, lang: str) -> str:
    words = SPAM_ZH if lang == "zh" else SPAM_EN
    sep = "" if lang == "zh" else " "
    lines = []
    for _ in range(rng.randint(15, 30)):
        lines.append(sep.join(rng.choice(words) for _ in range(rng.randint(4, 9))))
    return "\n".join(lines)


def garble(rng: random.Random, doc: str) -> str:
    """两种常见的乱码：UTF-8 字节被当成 Latin-1 解码（mojibake，只对非 ASCII 的中文有效），或者一堆符号。"""
    if not doc.isascii() and rng.random() < 0.7:
        return doc.encode("utf-8").decode("latin-1")
    sym = "#@$%^&*~<>{}[]|\\/=+_0123456789"
    return "\n".join("".join(rng.choice(sym) for _ in range(rng.randint(20, 60))) for _ in range(30))


def word_salad(rng: random.Random, doc: str, lang: str) -> str:
    """把一篇真文档每行里的词（中文按字）打乱，行尾的标点留在原位：
    字符统计、标点、行长几乎都不变，启发式规则很难发现，但已经不是话了。"""
    lines = doc.split("\n")
    out = []
    for ln in lines:
        toks = list(ln) if lang == "zh" else ln.split(" ")
        body, last = toks[:-1], toks[-1:]
        rng.shuffle(body)
        out.append(("" if lang == "zh" else " ").join(body + last))
    return "\n".join(out)


def near_copy(rng: random.Random, doc: str, lang: str) -> str:
    """转载：改几个词，前后加上站点的页眉页脚。"""
    toks = list(doc) if lang == "zh" else doc.split(" ")
    for _ in range(max(1, len(toks) // 60)):
        i = rng.randrange(len(toks))
        toks[i] = rng.choice(["的", "了", "是"]) if lang == "zh" else rng.choice(["the", "a", "and"])
    body = ("" if lang == "zh" else " ").join(toks)
    head = "转载自：诗词网" if lang == "zh" else "Reposted from poetry-archive.example"
    foot = "分享到：微博 | 微信" if lang == "zh" else "Share this: Facebook | Twitter"
    return f"{head}\n{body}\n{foot}"


def exact_copy(rng: random.Random, doc: str) -> str:
    """原样转载：只有空白不同（Windows 换行、行尾空格），规范化之后与原文完全相同。"""
    return doc.replace("\n", "  \r\n") if rng.random() < 0.5 else doc + "\n\n"


# ---------------------------------------------------------------------------
# 考题（评测集）与泄漏
# ---------------------------------------------------------------------------


def make_eval_items(rng: random.Random, heldout: dict[str, list[str]], n_per_lang: int = 30) -> list[dict]:
    """从留出的文档里摘一段当"考题"（填空题的题干）：英文约 30 个词，中文约 40 个字。"""
    items = []
    for lang, docs in heldout.items():
        for k in range(n_per_lang):
            doc = docs[k % len(docs)]
            if lang == "en":
                words = doc.split()
                s = rng.randrange(max(1, len(words) - 30))
                q = " ".join(words[s : s + 30])
            else:
                flat = doc.replace("\n", "")
                s = rng.randrange(max(1, len(flat) - 40))
                q = flat[s : s + 40]
            items.append({"id": f"{lang}-{k}", "lang": lang, "question": q})
    return items


def paraphrase(rng: random.Random, q: str, lang: str) -> str:
    """"改写"：每隔几个词换一个，13-gram 就再也对不上了。"""
    toks = list(q) if lang == "zh" else q.split(" ")
    for i in range(3, len(toks), 6):
        toks[i] = "之" if lang == "zh" else "thus"
    return ("" if lang == "zh" else " ").join(toks)


def build_crawl(seed: int = SEED) -> dict:
    """返回 {"docs": [{"text", "kind", "lang"}], "eval": [考题], "heldout": {lang: [文档]}}。

    kind：good（好文档）、contaminated（好文档里夹了一道考题）、exact_dup、near_dup、nav、spam、
    garbled、salad。
    """
    rng = random.Random(seed)
    real = load_real()
    heldout, pool = {}, {}
    for lang, docs in real.items():
        docs = docs[:]
        rng.shuffle(docs)
        n_hold = len(docs) // 10  # 10% 留出：既是考题的来源，也是后面训练小模型时的验证集
        heldout[lang], pool[lang] = docs[:n_hold], docs[n_hold:]
    eval_items = make_eval_items(rng, heldout)

    out = [{"text": d, "kind": "good", "lang": lang} for lang, docs in pool.items() for d in docs]
    for i, d in enumerate(out):
        d["origin"] = i  # 内容的"出处"：副本的 origin 等于原文的 origin，去重统计用
    goods = [d for d in out if d["kind"] == "good"]

    # 泄漏：把 16 道考题塞进 16 篇好文档中间（12 道原样，2 道改了大小写和标点，2 道改写过）
    leaked = rng.sample(range(len(eval_items)), 16)
    for j, ei in enumerate(leaked):
        item = eval_items[ei]
        cands = [d for d in goods if d["lang"] == item["lang"] and d["kind"] == "good"]
        d = rng.choice(cands)
        q = item["question"]
        variant = "verbatim" if j < 12 else ("case_punct" if j < 14 else "paraphrase")
        if variant == "case_punct":
            q = q.upper().replace(",", ";") if item["lang"] == "en" else q.replace("，", ",")
        elif variant == "paraphrase":
            q = paraphrase(rng, q, item["lang"])
        lines = d["text"].split("\n")
        k = len(lines) // 2
        d["text"] = "\n".join(lines[:k] + [q] + lines[k:])
        d["kind"] = "contaminated"
        d["leak"] = {"eval_id": item["id"], "variant": variant}

    goods = [d for d in out if d["kind"] == "good"]  # 垃圾只从没泄漏考题的好文档里造
    junk = []
    for _ in range(80):
        src = rng.choice(goods)
        junk.append({"text": exact_copy(rng, src["text"]), "kind": "exact_dup", "lang": src["lang"],
                     "origin": src["origin"]})
    for _ in range(80):
        src = rng.choice(goods)
        junk.append({"text": near_copy(rng, src["text"], src["lang"]), "kind": "near_dup",
                     "lang": src["lang"], "origin": src["origin"]})
    for _ in range(60):
        lang = rng.choice(["en", "zh"])
        junk.append({"text": nav_page(rng, lang), "kind": "nav", "lang": lang})
    for _ in range(40):
        lang = rng.choice(["en", "zh"])
        junk.append({"text": spam_page(rng, lang), "kind": "spam", "lang": lang})
    for _ in range(30):
        src = rng.choice(goods)
        junk.append({"text": garble(rng, src["text"]), "kind": "garbled", "lang": src["lang"]})
    for _ in range(80):
        src = rng.choice(goods)
        junk.append({"text": word_salad(rng, src["text"], src["lang"]), "kind": "salad",
                     "lang": src["lang"]})
    docs = out + junk
    rng.shuffle(docs)
    for i, d in enumerate(docs):
        d["id"] = i
    return {"docs": docs, "eval": eval_items, "heldout": heldout}


KINDS = ["good", "contaminated", "exact_dup", "near_dup", "nav", "spam", "garbled", "salad"]


def kind_table(docs: list[dict]) -> Counter:
    return Counter(d["kind"] for d in docs)


def main() -> None:
    crawl = build_crawl()
    docs = crawl["docs"]
    c = kind_table(docs)
    print(f"脏网页一共 {len(docs)} 篇，{sum(len(d['text'].encode()) for d in docs) / 1e6:.2f} MB")
    print(f"{'类型':<14}{'篇数':>6}  例子（前 60 个字符）")
    for k in KINDS:
        ex = next(d for d in docs if d["kind"] == k)["text"][:60].replace("\n", "⏎")
        print(f"{k:<14}{c[k]:>6}  {ex}")
    print(f"考题：{len(crawl['eval'])} 道（英文、中文各 30），其中 16 道泄漏进了训练数据")
    print(f"留出的干净文档（验证集）：英文 {len(crawl['heldout']['en'])} 篇，中文 {len(crawl['heldout']['zh'])} 篇")


if __name__ == "__main__":
    main()
