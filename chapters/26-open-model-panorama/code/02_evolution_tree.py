"""Chapter 26 · minimal code 2: the architecture evolution tree of the whole course.

Each node of the tree is one technique: its source (paper + year), the chapter of this course
that explains it, and its status now ("consensus" or "frontier").
We do not write "who uses it" by hand. The features() function of 01_panorama.py reads it from
the configs of the 6 newest flagships in models.json. The adopters that earlier chapters verified
(01_panorama.EARLIER) are added.

The script prints a text tree. The video (video/scenes.py) also calls build_tree() here to draw
the same tree.

Run: uv run python chapters/26-open-model-panorama/code/02_evolution_tree.py
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _panorama():
    spec = importlib.util.spec_from_file_location("panorama", HERE / "01_panorama.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["panorama"] = mod
    spec.loader.exec_module(mod)
    return mod


# (id, parent node, name, source and year, chapter in this course, status, key in features())
# Status: base = start / part of the consensus block; consensus = consensus per GOAL 2.1;
# new = new consensus; frontier = frontier note.
# The chapter labels stay in Chinese because the Chinese video shows them. chapter_en() gives the
# English form for the printed output.
NODES = [
    ("gpt2", None, "GPT-2", "Radford et al. 2019", "第 8 章", "base", None),
    (
        "llama",
        "gpt2",
        "Modern dense block (Llama style)",
        "Llama, arXiv:2302.13971, 2023",
        "第 9 章",
        "base",
        None,
    ),
    (
        "rmsnorm",
        "llama",
        "Pre-Norm + RMSNorm",
        "arXiv:1910.07467, 2019",
        "第 6、9 章",
        "consensus",
        "RMSNorm 前置",
    ),
    (
        "rope",
        "llama",
        "RoPE (+ YaRN to extend)",
        "arXiv:2104.09864, 2021; YaRN arXiv:2309.00071",
        "第 9、15 章",
        "consensus",
        "RoPE",
    ),
    ("swiglu", "llama", "SwiGLU", "arXiv:2002.05202, 2020", "第 9 章", "consensus", "SwiGLU/GLU"),
    ("gqa", "llama", "GQA", "arXiv:2305.13245, 2023", "第 10 章", "consensus", "GQA"),
    ("qknorm", "llama", "QK-Norm", "arXiv:2010.04245, 2020", "第 9 章", "consensus", "QK-Norm"),
    (
        "tie",
        "llama",
        "Tied input and output embedding (small models)",
        "arXiv:1608.05859, 2017",
        "第 9 章",
        "consensus",
        "Tied embedding",
    ),
    # Branch 1: the feed-forward layer becomes sparse
    (
        "moe",
        "llama",
        "MoE: fine-grained + shared experts",
        "DeepSeekMoE, arXiv:2401.06066, 2024",
        "第 24 章",
        "consensus",
        "MoE",
    ),
    (
        "auxfree",
        "moe",
        "Auxiliary-loss-free balancing",
        "arXiv:2408.15664, 2024",
        "第 24 章",
        "consensus",
        "无辅助损失均衡",
    ),
    (
        "latentmoe",
        "moe",
        "Latent MoE (experts compute in a latent space)",
        "Kimi K3 model card, 2026",
        "本章",
        "frontier",
        None,
    ),
    # Branch 2: KV compression
    (
        "mla",
        "gqa",
        "MLA: K/V compressed into a latent vector",
        "DeepSeek-V2, arXiv:2405.04434, 2024",
        "第 21 章",
        "consensus",
        "MLA",
    ),
    (
        "csa",
        "mla",
        "CSA/HCA: shared KV compressed along the tokens",
        "DeepSeek-V4, arXiv:2606.19348, 2026",
        "本章",
        "frontier",
        "CSA/HCA compression",
    ),
    # Branch 3: look at only a part
    (
        "swa",
        "llama",
        "Sliding window / local-global interleaving",
        "Mistral 7B arXiv:2310.06825; Gemma 2 arXiv:2408.00118",
        "第 22 章",
        "consensus",
        "滑动窗口",
    ),
    (
        "sparse",
        "swa",
        "Sparse attention (select top-k by content)",
        "DeepSeek-V3.2 DSA, 2025",
        "第 22 章、本章",
        "new",
        "稀疏注意力",
    ),
    # Branch 4: hybrid with linear attention
    (
        "hybrid",
        "llama",
        "Hybrid linear attention (about 3:1)",
        "Gated DeltaNet arXiv:2412.06464; Qwen3-Next 2025",
        "第 23 章",
        "consensus",
        "混合线性注意力",
    ),
    (
        "kda",
        "hybrid",
        "KDA (per-channel gate)",
        "Kimi Linear, arXiv:2510.26692, 2025",
        "第 23 章",
        "frontier",
        None,
    ),
    # Branch 5: training objective
    (
        "mtp",
        "llama",
        "MTP (predict one more token)",
        "DeepSeek-V3, arXiv:2412.19437, 2024",
        "第 25 章",
        "consensus",
        "MTP",
    ),
    # Branch 6: the residual stream itself (a new direction that started only in 2026)
    (
        "mhc",
        "llama",
        "Residual-stream changes: mHC / AttnRes",
        "DeepSeek-V4 2026; Kimi K3 2026",
        "本章",
        "frontier",
        "Hyper-connection mHC",
    ),
]

STATUS_ZH = {
    "base": "start",
    "consensus": "consensus",
    "new": "new consensus (reached rule A only in 2026)",
    "frontier": "frontier note",
}


def chapter_en(ch: str) -> str:
    """English display form of a chapter label: "第 6、9 章" → "Ch. 6, 9", "本章" → "this chapter"."""
    ch = re.sub(r"第 ([\d、]+) 章", lambda m: "Ch. " + m.group(1), ch)
    return ch.replace("本章", "this chapter").replace("、", ", ")


def build_tree() -> dict:
    """Return {nodes: [...], flagships: [...]}. Each node also lists the newest flagships that use it."""
    pan = _panorama()
    ms = pan.load_models()
    shown = pan.FLAGSHIPS + ["qwen3-0.6b", "qwen3.5-0.8b"]  # tied embedding occurs only in small models
    feats = {mid: pan.features(ms[mid]) for mid in shown}
    extra_users = {  # nodes that the config does not show; we add the users by hand
        "latentmoe": ["Kimi-K3"],
        "kda": ["Kimi-K3"],
        "mhc": ["DeepSeek-V4-Pro", "Kimi-K3"],
    }
    nodes = []
    for nid, parent, name, src, ch, status, key in NODES:
        users = [ms[mid]["name"] for mid in shown if key and key in feats[mid]]
        for u in extra_users.get(nid, []):
            if u not in users:
                users.append(u)
        earlier = pan.EARLIER.get(key, []) if key else []
        nodes.append(
            dict(
                id=nid,
                parent=parent,
                name=name,
                source=src,
                chapter=ch,
                status=status,
                users=users,
                earlier=earlier,
            )
        )
    main_feats = pan.features(ms["main"])
    return dict(
        nodes=nodes, flagships=[ms[m]["name"] for m in pan.FLAGSHIPS], main_uses=list(main_feats)
    )


def print_tree(tree: dict) -> None:
    kids: dict[str | None, list[dict]] = {}
    for n in tree["nodes"]:
        kids.setdefault(n["parent"], []).append(n)

    def walk(node: dict, prefix: str, last: bool) -> None:
        branch = "" if node["parent"] is None else ("└─ " if last else "├─ ")
        users = f" | used by: {', '.join(node['users'])}" if node["users"] else ""
        print(
            f"{prefix}{branch}{node['name']}  [{STATUS_ZH[node['status']]}, {chapter_en(node['chapter'])}]{users}"
        )
        ch = kids.get(node["id"], [])
        for i, c in enumerate(ch):
            walk(
                c,
                prefix + ("" if node["parent"] is None else ("   " if last else "│  ")),
                i == len(ch) - 1,
            )

    for root in kids[None]:
        walk(root, "", True)


def main() -> None:
    tree = build_tree()
    print(
        "Architecture evolution tree (status per GOAL.md 2.1; 'used by' lists only the 6 newest flagships in the table of this chapter + 2 small Qwen models:\n  "
        + ", ".join(tree["flagships"])
        + ", Qwen3-0.6B, Qwen3.5-0.8B)\n"
    )
    print_tree(tree)
    print("\nSources:")
    for n in tree["nodes"]:
        print(f"  {n['name']}: {n['source']}")
    print("\nThe main-line model uses: " + ", ".join(_panorama().en(k) for k in tree["main_uses"]))


if __name__ == "__main__":
    main()
