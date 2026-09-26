"""第 26 章 · 极简代码 2：整门课的架构演化树

树上每个节点是一项技术：出处（论文 + 年份）、在本课哪一章讲、现在是"共识"还是"前沿"。
"谁在用"不手写：用 01_panorama.py 的 features() 从 models.json 里 6 个最新旗舰的 config 读出来，
再加上前面章节核实过的采用方（01_panorama.EARLIER）。

打印一棵文字树；视频（video/scenes.py）也调用这里的 build_tree() 画同一棵树。

运行：uv run python chapters/26-open-model-panorama/code/02_evolution_tree.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _panorama():
    spec = importlib.util.spec_from_file_location("panorama", HERE / "01_panorama.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["panorama"] = mod
    spec.loader.exec_module(mod)
    return mod


# (id, 父节点, 名称, 出处与年份, 本课章节, 状态, 对应 features() 里的键)
# 状态：base = 起点 / 共识块的组成；consensus = 按 GOAL 2.1 已是共识；frontier = 前沿观察
NODES = [
    ("gpt2", None, "GPT-2", "Radford 等 2019", "第 8 章", "base", None),
    (
        "llama",
        "gpt2",
        "现代稠密块（Llama 式）",
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
        "RoPE（+ YaRN 扩长）",
        "arXiv:2104.09864, 2021；YaRN arXiv:2309.00071",
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
        "共享输入输出 embedding（小模型）",
        "arXiv:1608.05859, 2017",
        "第 9 章",
        "consensus",
        "共享 embedding",
    ),
    # 分支 1：前馈层变稀疏
    (
        "moe",
        "llama",
        "MoE：细粒度 + 共享专家",
        "DeepSeekMoE, arXiv:2401.06066, 2024",
        "第 24 章",
        "consensus",
        "MoE",
    ),
    (
        "auxfree",
        "moe",
        "无辅助损失均衡",
        "arXiv:2408.15664, 2024",
        "第 24 章",
        "consensus",
        "无辅助损失均衡",
    ),
    (
        "latentmoe",
        "moe",
        "Latent MoE（专家在潜空间里算）",
        "Kimi K3 模型卡, 2026",
        "本章",
        "frontier",
        None,
    ),
    # 分支 2：KV 压缩
    (
        "mla",
        "gqa",
        "MLA：K/V 压成潜向量",
        "DeepSeek-V2, arXiv:2405.04434, 2024",
        "第 21 章",
        "consensus",
        "MLA",
    ),
    (
        "csa",
        "mla",
        "CSA/HCA：按 token 压缩的共享 KV",
        "DeepSeek-V4, arXiv:2606.19348, 2026",
        "本章",
        "frontier",
        "压缩注意力（CSA/HCA）",
    ),
    # 分支 3：只看一部分
    (
        "swa",
        "llama",
        "滑动窗口 / 局部-全局交替",
        "Mistral 7B arXiv:2310.06825；Gemma 2 arXiv:2408.00118",
        "第 22 章",
        "consensus",
        "滑动窗口",
    ),
    (
        "sparse",
        "swa",
        "稀疏注意力（按内容挑 top-k）",
        "DeepSeek-V3.2 DSA, 2025",
        "第 22 章、本章",
        "new",
        "稀疏注意力",
    ),
    # 分支 4：线性注意力混合
    (
        "hybrid",
        "llama",
        "混合线性注意力（约 3:1）",
        "Gated DeltaNet arXiv:2412.06464；Qwen3-Next 2025",
        "第 23 章",
        "consensus",
        "混合线性注意力",
    ),
    (
        "kda",
        "hybrid",
        "KDA（逐通道门控）",
        "Kimi Linear, arXiv:2510.26692, 2025",
        "第 23 章",
        "frontier",
        None,
    ),
    # 分支 5：训练目标
    (
        "mtp",
        "llama",
        "MTP（多预测一个 token）",
        "DeepSeek-V3, arXiv:2412.19437, 2024",
        "第 25 章",
        "consensus",
        "MTP",
    ),
    # 分支 6：残差流本身（2026 年才出现的新方向）
    (
        "mhc",
        "llama",
        "残差流改造：mHC / AttnRes",
        "DeepSeek-V4 2026；Kimi K3 2026",
        "本章",
        "frontier",
        "超连接 mHC",
    ),
]

STATUS_ZH = {
    "base": "起点",
    "consensus": "共识",
    "new": "新晋共识（2026 刚满 3 家）",
    "frontier": "前沿观察",
}


def build_tree() -> dict:
    """返回 {nodes: [...], flagships: [...]}，每个节点带上"最新旗舰里谁在用"。"""
    pan = _panorama()
    ms = pan.load_models()
    shown = pan.FLAGSHIPS + ["qwen3-0.6b", "qwen3.5-0.8b"]  # 共享 embedding 只在小模型里出现
    feats = {mid: pan.features(ms[mid]) for mid in shown}
    extra_users = {  # config 看不出来、需要额外说明的节点
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
        users = f"｜在用：{'、'.join(node['users'])}" if node["users"] else ""
        print(
            f"{prefix}{branch}{node['name']}  [{STATUS_ZH[node['status']]}，{node['chapter']}]{users}"
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
        "架构演化树（状态按 GOAL.md 2.1 判定；'在用'只列本章表里的 6 个最新旗舰 + 2 个千问小模型：\n  "
        + "、".join(tree["flagships"])
        + "、Qwen3-0.6B、Qwen3.5-0.8B）\n"
    )
    print_tree(tree)
    print("\n出处：")
    for n in tree["nodes"]:
        print(f"  {n['name']}：{n['source']}")
    print("\n主线模型用了：" + "、".join(tree["main_uses"]))


if __name__ == "__main__":
    main()
