"""Data set registry: name, URL, license, languages (Chapter 13).

GOAL.md 3.3 says: "use only public data whose license allows it, and record the source and license
of each data set". This module is that table. Register each data set here before pretraining.
The data list of the model card is also made from this table.

Source of the license column: the license metadata of the Hugging Face data set page (read on 2026-09-26).
If the page has no license, or if the data set mixes several upstream licenses, the value is
"待核实" (to be verified). A person must confirm it before the download of Step 2.
`zero/data/download.py` and the tests look for this exact string, so keep it in Chinese.
This module holds only metadata. It does not download anything.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class DatasetSource:
    name: str
    url: str
    license: str  # SPDX-style identifier; if not sure, write "待核实" (to be verified)
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
            description="English web pages from FineWeb, filtered by an educational-value classifier; about 1.3T tokens (arXiv:2406.17557).",
            notes="ODC-By requires attribution. The CommonCrawl terms of use also apply.",
            tags=("web", "edu"),
        ),
        DatasetSource(
            name="dclm-baseline",
            url="https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0",
            license="CC-BY-4.0",
            languages=("en",),
            stage="pretrain",
            description="DCLM-baseline 1.0: English web pages, about 4T tokens (arXiv:2406.11794).",
            tags=("web",),
            notes="The data set card says both CC-BY-4.0 and \"for research only\". The relation between the two is to be verified (Chapter 13).",
        ),
        DatasetSource(
            name="fineweb-2-zh",
            url="https://huggingface.co/datasets/HuggingFaceFW/fineweb-2",
            license="ODC-By-1.0",
            languages=("zh",),
            stage="pretrain",
            description="The Chinese subset (cmn_Hani) of the FineWeb2 multilingual web pages, arXiv:2506.20920.",
            notes="The name of the Chinese subset is cmn_Hani (confirmed on the data set card in Chapter 13). The number of tokens is to be verified.",
            tags=("web",),
        ),
        DatasetSource(
            name="ultra-fineweb",
            url="https://huggingface.co/datasets/openbmb/Ultra-FineWeb",
            license="Apache-2.0",
            languages=("en", "zh"),
            stage="pretrain",
            description="Chinese and English web pages from FineWeb and the Chinese FineWeb-edu-v2, after verification-based high-quality filtering (arXiv:2505.05427).",
            notes="The page says Apache-2.0, but the Chinese part comes from several upstream corpora (listed in Chapter 13). The license of each upstream corpus is to be verified.",
            tags=("web", "edu"),
        ),
        DatasetSource(
            name="stack-edu",
            url="https://huggingface.co/datasets/HuggingFaceTB/stack-edu",
            license="待核实",
            languages=("code",),
            stage="pretrain",
            description="Code from The Stack v2 (StarCoder2Data), filtered by an educational-value classifier; about 125B tokens.",
            notes="The data set page has no license metadata. The data holds only blob_id; the content comes from the Software Heritage S3. The terms of The Stack v2 and the detected_licenses of each file apply (to be verified).",
            tags=("code", "edu"),
        ),
        DatasetSource(
            name="finemath",
            url="https://huggingface.co/datasets/HuggingFaceTB/finemath",
            license="ODC-By-1.0",
            languages=("en",),
            stage="pretrain",
            description="Math education content from CommonCrawl, filtered by a math classifier; FineMath-3+ has about 34B tokens.",
            tags=("math",),
        ),
        DatasetSource(
            name="ultradata-code",
            url="https://huggingface.co/datasets/openbmb/UltraData-Code",
            license="Apache-2.0",
            languages=("code",),
            stage="pretrain",
            description="The code data of MiniCPM5 (2026-09). L2 is natural code from about 192 million public GitHub "
            "repositories, after cleaning, deduplication, and filtering by algorithmic relevance and quality: about 400B "
            "tokens in 11 languages. L3 is programming exercises synthesized from L2: about 150B tokens. "
            "The content is directly in the parquet field `content`; AWS is not necessary.",
            notes="The data set card says Apache-2.0. The card does not give the license of the original repository "
            "of each file (to be verified). The card says: in a 1B model with 10B tokens of continued training, L2 "
            "is 4.37 / 3.05 points better than Stack-Edu on EvalPlus / MultiPL-E. So it is a candidate to replace "
            "Stack-Edu as the main-line code source (Chapter 13). In 2026-10, it was first used for the samples of "
            "the vocabulary measurement in Chapter 13.",
            tags=("code",),
        ),
        # ---- Tiny corpora in the repository (for offline smoke tests; see assets/tiny_corpus/LICENSES.md) ----
        DatasetSource(
            name="tiny-shakespeare",
            url="https://github.com/karpathy/char-rnn/blob/master/data/tinyshakespeare/input.txt",
            license="Public-Domain",
            languages=("en",),
            stage="tiny",
            description="Selected passages from the works of Shakespeare, about 1.1MB.",
        ),
        DatasetSource(
            name="tiny-chinese-poetry",
            url="https://github.com/chinese-poetry/chinese-poetry",
            license="MIT",
            languages=("zh",),
            stage="tiny",
            description="Parts of the Analects, the Book of Songs, the Four Books, the primers (Mengxue), Tang poems, and Song ci poems from the chinese-poetry repository, as plain text.",
        ),
        DatasetSource(
            name="tiny-code",
            url="https://github.com/pengqianhan/From-0-to-AGI",
            license="待核实",
            languages=("code",),
            stage="tiny",
            description="Python code of this repository (some files in zero/).",
            notes="This repository has no license yet. The author of the repository will decide it.",
        ),
    ]
}


def get_source(name: str) -> DatasetSource:
    if name not in SOURCES:
        raise KeyError(f"Data set not registered: {name}. Registered: {', '.join(SOURCES)}")
    return SOURCES[name]


def list_sources(stage: str | None = None, language: str | None = None) -> list[DatasetSource]:
    out = list(SOURCES.values())
    if stage is not None:
        out = [s for s in out if s.stage == stage]
    if language is not None:
        out = [s for s in out if language in s.languages]
    return out


def unverified_licenses() -> list[str]:
    """Data sets whose license is not verified yet. This list must be empty before the download of Step 2."""
    return [s.name for s in SOURCES.values() if "待核实" in s.license]
