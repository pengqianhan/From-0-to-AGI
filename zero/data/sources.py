"""数据集登记表：名称、地址、许可证、语言（对应第 13 章）。

GOAL.md 3.3 要求"只用许可证允许的公开数据，每个数据集都记录来源和许可证"。这里就是那张表：
预训练前先在这里登记，模型卡里的数据清单也从这里生成。

许可证一栏的来源：Hugging Face 数据集页面的 license 元数据（2026-09-26 查询）。
页面没有标注、或者数据集本身混合了多种上游许可证的，写"待核实"，第二步下载前人工确认。
这个模块只是元数据，不做任何下载。
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class DatasetSource:
    name: str
    url: str
    license: str  # SPDX 风格的标识；拿不准写"待核实"
    languages: tuple[str, ...]
    stage: str  # "pretrain" | "midtrain" | "tiny"
    description: str = ""
    notes: str = ""
    tags: tuple[str, ...] = field(default_factory=tuple)


SOURCES: dict[str, DatasetSource] = {
    s.name: s
    for s in [
        DatasetSource(
            name="fineweb-edu",
            url="https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu",
            license="ODC-By-1.0",
            languages=("en",),
            stage="pretrain",
            description="FineWeb 经教育价值分类器筛选后的英文网页，约 1.3T token（arXiv:2406.17557）。",
            notes="ODC-By 要求署名；同时受 CommonCrawl 使用条款约束。",
            tags=("web", "edu"),
        ),
        DatasetSource(
            name="dclm-baseline",
            url="https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0",
            license="CC-BY-4.0",
            languages=("en",),
            stage="pretrain",
            description="DCLM-baseline 1.0：约 4T token 的英文网页（arXiv:2406.11794）。",
            tags=("web",),
            notes="数据集卡同时写了 CC-BY-4.0 和“仅供研究”，两者关系待核实（第 13 章）。",
        ),
        DatasetSource(
            name="fineweb-2-zh",
            url="https://huggingface.co/datasets/HuggingFaceFW/fineweb-2",
            license="ODC-By-1.0",
            languages=("zh",),
            stage="pretrain",
            description="FineWeb2 多语言网页中的中文子集（cmn_Hani），arXiv:2506.20920。",
            notes="中文子集名为 cmn_Hani（第 13 章已在数据集卡上确认）；token 数待核实。",
            tags=("web",),
        ),
        DatasetSource(
            name="ultra-fineweb",
            url="https://huggingface.co/datasets/openbmb/Ultra-FineWeb",
            license="Apache-2.0",
            languages=("en", "zh"),
            stage="pretrain",
            description="对 FineWeb 与中文 FineWeb-edu-v2 做验证式高质量过滤后的中英网页（arXiv:2505.05427）。",
            notes="页面标注 Apache-2.0，但中文部分来自多个上游语料（第 13 章列出），各上游许可证待核实。",
            tags=("web", "edu"),
        ),
        DatasetSource(
            name="stack-edu",
            url="https://huggingface.co/datasets/HuggingFaceTB/stack-edu",
            license="待核实",
            languages=("code",),
            stage="pretrain",
            description="从 The Stack v2（StarCoder2Data）中用教育价值分类器筛出的代码，约 125B token。",
            notes="数据集页面未标注 license 元数据；数据只含 blob_id，正文要从 Software Heritage 的 S3 取，受 The Stack v2 条款与每个文件 detected_licenses 约束（待核实）。",
            tags=("code", "edu"),
        ),
        DatasetSource(
            name="finemath",
            url="https://huggingface.co/datasets/HuggingFaceTB/finemath",
            license="ODC-By-1.0",
            languages=("en",),
            stage="pretrain",
            description="从 CommonCrawl 中用数学分类器筛出的数学教育内容，FineMath-3+ 约 34B token。",
            tags=("math",),
        ),
        DatasetSource(
            name="ultradata-code",
            url="https://huggingface.co/datasets/openbmb/UltraData-Code",
            license="Apache-2.0",
            languages=("code",),
            stage="pretrain",
            description="MiniCPM5 的代码数据（2026-09）：L2 是从约 1.92 亿个公开 GitHub 仓库清洗、去重后按算法相关性"
            "与质量筛出的自然代码，约 400B token、11 种语言；L3 是由 L2 合成的编程练习，约 150B token。"
            "正文直接在 parquet 的 content 字段里，不需要 AWS。",
            notes="数据集卡标 Apache-2.0；单个文件原始仓库的许可证数据卡没有逐条给出（待核实）。"
            "数据卡称 1B 模型续训 10B token 时 L2 比 Stack-Edu 在 EvalPlus / MultiPL-E 上高 4.37 / 3.05 分，"
            "可作为主线代码来源 Stack-Edu 的候选替代（第 13 章）。2026-10 先用于第 13 章词表测量的抽样。",
            tags=("code",),
        ),
        # ---- 仓库内置的极小语料（离线冒烟测试用，见 assets/tiny_corpus/LICENSES.md）----
        DatasetSource(
            name="tiny-shakespeare",
            url="https://github.com/karpathy/char-rnn/blob/master/data/tinyshakespeare/input.txt",
            license="Public-Domain",
            languages=("en",),
            stage="tiny",
            description="莎士比亚作品选段，约 1.1MB。",
        ),
        DatasetSource(
            name="tiny-chinese-poetry",
            url="https://github.com/chinese-poetry/chinese-poetry",
            license="MIT",
            languages=("zh",),
            stage="tiny",
            description="chinese-poetry 仓库中论语、诗经、四书、蒙学、唐诗、宋词的一部分，转成纯文本。",
        ),
        DatasetSource(
            name="tiny-code",
            url="https://github.com/pengqianhan/From-0-to-AGI",
            license="待核实",
            languages=("code",),
            stage="tiny",
            description="本仓库自己的 Python 代码（zero/ 下的部分文件）。",
            notes="本仓库尚未声明许可证，待仓库作者确定。",
        ),
    ]
}


def get_source(name: str) -> DatasetSource:
    if name not in SOURCES:
        raise KeyError(f"未登记的数据集：{name}。已登记：{', '.join(SOURCES)}")
    return SOURCES[name]


def list_sources(stage: str | None = None, language: str | None = None) -> list[DatasetSource]:
    out = list(SOURCES.values())
    if stage is not None:
        out = [s for s in out if s.stage == stage]
    if language is not None:
        out = [s for s in out if language in s.languages]
    return out


def unverified_licenses() -> list[str]:
    """许可证还没核实的数据集——第二步下载之前必须清零。"""
    return [s.name for s in SOURCES.values() if "待核实" in s.license]
