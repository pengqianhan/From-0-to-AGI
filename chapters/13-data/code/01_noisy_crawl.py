"""Make a "noisy crawl": real documents + duplicates + navigation pages + spam + garbled text
+ word salad + leaked test questions.

We cannot download real web data here (and it is too large). Thus this chapter uses Shakespeare
and classical Chinese poetry from assets/tiny_corpus as the "good documents". Then it adds the
types of junk that are common on web pages. Each document has a label `kind`. The scripts after
this one use the label to check if each filter step removed the correct documents. The real
world has no such label. This is why data work is difficult.

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
    """Split the text into paragraphs at empty lines.

    Then join adjacent paragraphs into "documents" of about doc_chars characters.
    """
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
# Types of junk
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
    """Two common types of garbled text: UTF-8 bytes decoded as Latin-1 (mojibake; this works only
    on non-ASCII Chinese text), or a block of random symbols."""
    if not doc.isascii() and rng.random() < 0.7:
        return doc.encode("utf-8").decode("latin-1")
    sym = "#@$%^&*~<>{}[]|\\/=+_0123456789"
    return "\n".join("".join(rng.choice(sym) for _ in range(rng.randint(20, 60))) for _ in range(30))


def word_salad(rng: random.Random, doc: str, lang: str) -> str:
    """Shuffle the words in each line of a real document (for Chinese, the characters).
    The punctuation at the end of each line stays in place. The character statistics, the
    punctuation, and the line lengths almost do not change, so heuristic rules cannot easily
    find this text. But the text is no longer language."""
    lines = doc.split("\n")
    out = []
    for ln in lines:
        toks = list(ln) if lang == "zh" else ln.split(" ")
        body, last = toks[:-1], toks[-1:]
        rng.shuffle(body)
        out.append(("" if lang == "zh" else " ").join(body + last))
    return "\n".join(out)


def near_copy(rng: random.Random, doc: str, lang: str) -> str:
    """A repost: change some words, and add the header and footer of the site."""
    toks = list(doc) if lang == "zh" else doc.split(" ")
    for _ in range(max(1, len(toks) // 60)):
        i = rng.randrange(len(toks))
        toks[i] = rng.choice(["的", "了", "是"]) if lang == "zh" else rng.choice(["the", "a", "and"])
    body = ("" if lang == "zh" else " ").join(toks)
    head = "转载自：诗词网" if lang == "zh" else "Reposted from poetry-archive.example"
    foot = "分享到：微博 | 微信" if lang == "zh" else "Share this: Facebook | Twitter"
    return f"{head}\n{body}\n{foot}"


def exact_copy(rng: random.Random, doc: str) -> str:
    """An exact repost: only the whitespace is different (Windows line ends, spaces at line ends).
    After normalization, the text is identical to the original."""
    return doc.replace("\n", "  \r\n") if rng.random() < 0.5 else doc + "\n\n"


# ---------------------------------------------------------------------------
# Test questions (the evaluation set) and leaks
# ---------------------------------------------------------------------------


def make_eval_items(rng: random.Random, heldout: dict[str, list[str]], n_per_lang: int = 30) -> list[dict]:
    """Take a passage from the held-out documents as a "test question" (the stem of a fill-in
    question): about 30 words in English, about 40 characters in Chinese."""
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
    """A "paraphrase": replace one word in every few words. Then no 13-gram matches any more."""
    toks = list(q) if lang == "zh" else q.split(" ")
    for i in range(3, len(toks), 6):
        toks[i] = "之" if lang == "zh" else "thus"
    return ("" if lang == "zh" else " ").join(toks)


def build_crawl(seed: int = SEED) -> dict:
    """Return {"docs": [{"text", "kind", "lang"}], "eval": [questions], "heldout": {lang: [docs]}}.

    kind: good (a good document), contaminated (a good document that contains a test question),
    exact_dup, near_dup, nav, spam, garbled, salad.
    """
    rng = random.Random(seed)
    real = load_real()
    heldout, pool = {}, {}
    for lang, docs in real.items():
        docs = docs[:]
        rng.shuffle(docs)
        # Hold out 10%: the source of the test questions, and later the validation set
        n_hold = len(docs) // 10
        heldout[lang], pool[lang] = docs[:n_hold], docs[n_hold:]
    eval_items = make_eval_items(rng, heldout)

    out = [{"text": d, "kind": "good", "lang": lang} for lang, docs in pool.items() for d in docs]
    for i, d in enumerate(out):
        d["origin"] = i  # A copy gets the "origin" of its original; the dedup statistics use it
    goods = [d for d in out if d["kind"] == "good"]

    # Leaks: put 16 test questions into the middle of 16 good documents
    # (12 verbatim, 2 with changed case and punctuation, 2 paraphrased)
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

    goods = [d for d in out if d["kind"] == "good"]  # make junk only from documents without a leak
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
    print(f"Noisy crawl: {len(docs)} documents, {sum(len(d['text'].encode()) for d in docs) / 1e6:.2f} MB")
    print(f"{'Type':<14}{'Docs':>6}  Example (first 60 characters)")
    for k in KINDS:
        ex = next(d for d in docs if d["kind"] == k)["text"][:60].replace("\n", "⏎")
        print(f"{k:<14}{c[k]:>6}  {ex}")
    print(f"Test questions: {len(crawl['eval'])} (30 English, 30 Chinese); 16 of them leaked into the training data")
    print(f"Held-out clean documents (validation set): {len(crawl['heldout']['en'])} English, {len(crawl['heldout']['zh'])} Chinese")


if __name__ == "__main__":
    main()
