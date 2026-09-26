"""提示词格式敏感性：同一个模型、同一批题，只改提示词的"长相"，分数能差出好几倍。

    uv run python chapters/11-evaluation/code/03_prompt_sensitivity.py

复用 02 的玩具世界和字符级最长后缀模型，只换 6 种提示词格式（意思完全一样），都用对数似然判分。
真实的大模型没有这么极端，但方向一样：Sclar 等（ICLR 2024）在 LLaMA-2-13B 上发现，
意思相同的格式之间准确率最多差 76 个百分点。所以预注册必须把模板一字不差地写死。
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _load(name: str):  # noqa: ANN202
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # dataclass 需要能在 sys.modules 里找到模块
    spec.loader.exec_module(mod)
    return mod


toy = _load("02_loglik_vs_generate")
LETTERS = "ABCD"

# 名字 → (提示词函数, 选项怎么写)。"letter" 表示让模型输出选项字母，而不是选项内容。
FORMATS = {
    "完形填空": (lambda it: it.stem, "text"),
    "完形填空 + 末尾空格": (lambda it: it.stem + " ", "text"),
    "问答（练习册格式）": (lambda it: f"问：{it.question}\n答：", "text"),
    "问答（英文标签）": (lambda it: f"Q: {it.question}\nA: ", "text"),
    "问答（冒号换空格）": (lambda it: f"问 {it.question}\n答 ", "text"),
    "字母选择题": (
        lambda it: f"问：{it.question}\n"
        + "".join(f"{LETTERS[i]}. {c}\n" for i, c in enumerate(it.choices))
        + "答：",
        "letter",
    ),
}


def score_format(lm, items, prompt_fn, mode: str) -> list[dict]:  # noqa: ANN001
    per = []
    for it in items:
        ctx = prompt_fn(it)
        options = list(LETTERS[: len(it.choices)]) if mode == "letter" else it.choices
        lps = [lm.logprob(ctx, o) for o in options]
        pred = max(range(len(lps)), key=lambda i: lps[i])
        per.append({"id": it.id, "correct": float(pred == it.answer), "leaked": it.leaked,
                    "pred": pred})
    return per


def run_formats() -> list[dict]:
    """返回每种格式在 全部 / 泄漏 / 干净 三组题上的正确率（视频也用这个函数）。"""
    world = toy.build_world()
    lm = toy.SuffixLM(world.corpus)
    rows = []
    for name, (fn, mode) in FORMATS.items():
        per = score_format(lm, world.items, fn, mode)
        leak, clean = toy.split_acc(per)
        preds = [p["pred"] for p in per]
        rows.append({
            "name": name,
            "acc": sum(p["correct"] for p in per) / len(per),
            "leaked": leak,
            "clean": clean,
            "most_common_pred_share": max(preds.count(k) for k in range(4)) / len(preds),
        })
    return rows


def main() -> None:
    rows = run_formats()
    print("同一个模型（02 的字符级最长后缀模型）、同样 32 道四选一题、同样的对数似然判分，只换提示词格式：\n")
    print("| 提示词格式 | 全部 32 题 | 泄漏的 12 题 | 干净的 20 题 | 最常选的那个位置占比 |")
    print("|---|---:|---:|---:|---:|")
    for r in rows:
        print(f"| {r['name']} | {r['acc']:.3f} | {r['leaked']:.3f} | {r['clean']:.3f} | "
              f"{r['most_common_pred_share']:.2f} |")
    accs = [r["acc"] for r in rows]
    print(f"\n最高 {max(accs):.3f}，最低 {min(accs):.3f}，相差 {max(accs) - min(accs):.3f}"
          "（随机猜的期望是 0.25）")
    print("最后一列：字母选择题里，模型几乎总选同一个字母——小模型还不会'看选项、报字母'，"
          "所以 Base 小模型通常用完形填空式的对数似然评测。")


if __name__ == "__main__":
    main()
