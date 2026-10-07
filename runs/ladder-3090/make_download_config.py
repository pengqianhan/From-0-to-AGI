"""生成阶梯实验的下载配置 configs/ladder3090/download.toml（第 0 阶段）。

    uv run python runs/ladder-3090/make_download_config.py

目标：约 3.6B token（按 4.15 字节/token 约 15GB 正文），配比同 configs/main/data.toml（代码用 UltraData-Code-L2）。
为了不只拿到最早的网页（流式读取从文件开头读，文件按抓取时间排列；见 runs/2026-10-01-vocab-corpus），
每个数据集按文件序号均匀挑若干个文件，每个文件从头读 target / k 字节；每个文件是一个独立的下载来源
（名字 <来源>-<序号>），方便并行下载，流水线再用通配符把它们合回一个来源。
"""

from __future__ import annotations

from pathlib import Path

from huggingface_hub import HfApi

GB = 1e9
MARGIN = 1.1  # 留给过滤损耗和验证集
# (来源名, 登记名, 仓库, revision, 文件前缀, 正文字段, 目标字节, 挑几个文件, 额外余量)
PLAN = [
    ("fineweb-edu", "fineweb-edu", "HuggingFaceFW/fineweb-edu", "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9",
     "sample/10BT/", "text", 0.35 * 15 * GB, 14, 1.0),
    ("dclm-baseline", "dclm-baseline", "mlfoundations/dclm-baseline-1.0", "a3b142c183aebe5af344955ae20836eb34dcf69b",
     "global-shard", "text", 0.10 * 15 * GB, 40, 1.0),
    ("finemath", "finemath", "HuggingFaceTB/finemath", "e92b25a616738fe95dc186b64dfb19f9c8525594",
     "finemath-3plus/", "text", 0.10 * 15 * GB, 16, 1.0),
    # FineWeb-2 中文还要过一遍我们的启发式规则（configs/main/data.toml：heuristics = "web"），多留 10%
    ("fineweb-2-zh", "fineweb-2-zh", "HuggingFaceFW/fineweb-2", "af9c13333eb981300149d5ca60a8e9d659b276b9",
     "data/cmn_Hani/train/", "text", 0.20 * 15 * GB, 24, 1.1),
    ("ultra-fineweb-zh", "ultra-fineweb", "openbmb/Ultra-FineWeb", "02c85641e3d19a854be2e09139c25adaa9518063",
     "data/ultrafineweb_zh/", "content", 0.10 * 15 * GB, 16, 1.0),
]
# 代码 0.15：Python 30%，JavaScript / Java / C++ 各 15%，Go / Rust / Shell / C# / PHP 各 5%（同 configs/vocab）
CODE_REV = "85182d829f2ce7ea07cca72ebfc509deea1d9f5f"
for lang, share, k in [("py", 0.30, 8), ("js", 0.15, 4), ("java", 0.15, 4), ("cpp", 0.15, 4),
                       ("go", 0.05, 2), ("rust", 0.05, 2), ("sh", 0.05, 2), ("cs", 0.05, 2), ("php", 0.05, 2)]:
    PLAN.append((f"ultradata-code-{lang}", "ultradata-code", "openbmb/UltraData-Code", CODE_REV,
                 f"data/UltraData-Code-L2/{lang}/", "content", 0.15 * share * 15 * GB, k, 1.0))


# 保留的元数据（流水线要用 FineWeb-Edu 的 int_score 做阈值；其余留作出处记录）
KEEP = {
    "fineweb-edu": ["url", "dump", "score", "int_score", "token_count", "language_score"],
    "dclm-baseline": ["url"],
    "finemath": ["url", "score", "int_score", "token_count"],
    "fineweb-2-zh": ["url", "dump", "language_score"],
    "ultra-fineweb": ["score", "source"],
    "ultradata-code": ["repo_name", "relative_path", "category", "quality_score"],
}
ID_FIELD = {"ultradata-code": "uuid"}
# 没有默认子集的数据集，按文件读时也要给子集名（FineWeb-2 有上千个语言子集）
CONFIG = {"fineweb-2-zh": "cmn_Hani"}


def spread(files: list[str], k: int) -> list[str]:
    """从排好序的文件里均匀挑 k 个（覆盖头、中、尾）。"""
    if k >= len(files):
        return files
    return [files[round(i * (len(files) - 1) / (k - 1))] for i in range(k)] if k > 1 else files[:1]


def main() -> None:
    api = HfApi()
    out = [
        "# Download config for the training data of the ladder experiment (runs/ladder-3090).",
        "# runs/ladder-3090/make_download_config.py writes this file. Do not edit it by hand.",
        "# Each [[sources]] is one file of a dataset. The download reads target_bytes bytes of text from the start of the file.",
        "# The pipeline (configs/ladder3090/data.toml) uses the wildcard <source>-*/ to join the files of one dataset into one source again.",
        "",
    ]
    total = 0.0
    for name, registry, repo, rev, prefix, text_field, target, k, extra in PLAN:
        files = sorted(
            f for f in api.list_repo_files(repo, repo_type="dataset", revision=rev)
            if f.startswith(prefix) and f.endswith((".parquet", ".jsonl.zst"))
        )
        if not files:
            raise SystemExit(f"{repo}@{rev[:10]} 下没有以 {prefix} 开头的数据文件")
        chosen = spread(files, k)
        per_file = int(target * MARGIN * extra / len(chosen))
        total += per_file * len(chosen)
        print(f"{name:22s} {len(files):6d} 个文件里挑 {len(chosen):2d} 个，每个 {per_file / 1e6:7.1f} MB")
        for i, f in enumerate(chosen):
            out += [
                "[[sources]]",
                f'name = "{name}-{i:02d}"',
                f'registry = "{registry}"',
                "  [sources.download]",
                f'  repo = "{repo}"',
                f'  revision = "{rev}"',
                *([f'  config = "{CONFIG[registry]}"'] if registry in CONFIG else []),
                f'  data_files = "{f}"',
                f'  text_field = "{text_field}"',
                f'  id_field = "{ID_FIELD.get(registry, "id")}"',
                f"  keep_fields = {KEEP[registry]!r}".replace("'", '"'),
                f"  target_bytes = {per_file}",
                "",
            ]
    path = Path(__file__).resolve().parents[2] / "configs" / "ladder3090" / "download.toml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(out), encoding="utf-8")
    print(f"共 {total / GB:.2f} GB 正文 → {path}")


if __name__ == "__main__":
    main()
