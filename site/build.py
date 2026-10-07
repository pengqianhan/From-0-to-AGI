"""Build the course website (MkDocs Material + mkdocs-static-i18n). English is the default language.

    uv pip install --python .venv/bin/python "mkdocs-material>=9.7" "mkdocs-static-i18n>=1.3"   # or: pip install ...
    python site/build.py            # prepare site/build/ (docs + mkdocs.yml)
    python site/build.py --serve    # preview at http://127.0.0.1:8000
    python site/build.py --build    # write the static site to site/build/site/

The Markdown files in the repository are the only source. This script copies them into site/build/docs/:
- README.md → index.md (English), README.zh.md → index.zh.md (Chinese), for the home page and each chapter;
- links to code and data files point to the files on GitHub, because the website shows only the text;
- images are copied into the website;
- links between pages stay links between pages;
- the "English · 中文" line under each title is removed, because the website has a language button.
A chapter goes on the website only when it has both README.md and README.zh.md (the bilingual pages).
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
BUILD = REPO / "site" / "build"
DOCS = BUILD / "docs"
GITHUB = "https://github.com/pengqianhan/From-0-to-AGI/blob/main/"
LANG_LINE = re.compile(
    r"^(\*\*English\*\* · \[中文\]\([^)]*\)|\[English\]\([^)]*\) · \*\*中文\*\*)\s*$", re.M
)
LINK = re.compile(r"(!?\[[^\]]*\])\(([^)\s]+)\)")


def pages() -> dict[Path, Path]:
    """Map each English source page (path in the repository) to its page in the website."""
    out = {
        REPO / "README.md": DOCS / "index.md",
        REPO / "docs" / "STYLE_GUIDE.md": DOCS / "style-guide.md",
    }
    for ch in sorted((REPO / "chapters").iterdir()):
        if (ch / "README.md").exists() and (ch / "README.zh.md").exists():
            out[ch / "README.md"] = DOCS / "chapters" / ch.name / "index.md"
    return out


def zh(path: Path) -> Path:
    """README.md → README.zh.md, index.md → index.zh.md."""
    return path.with_name(path.stem + ".zh" + path.suffix)


def rewrite(text: str, src: Path, dst: Path, site_pages: dict[Path, Path]) -> str:
    text = LANG_LINE.sub("", text)

    def repl(m: re.Match) -> str:
        label, target = m.group(1), m.group(2)
        if re.match(r"^[a-z]+:|^#", target):
            return m.group(0)  # external link or anchor
        path, _, anchor = target.partition("#")
        resolved = (src.parent / path).resolve()
        english = resolved.with_name(resolved.name.replace(".zh.md", ".md"))
        if english in site_pages:  # a page on the website (the i18n plugin picks the language)
            rel = os.path.relpath(site_pages[english], dst.parent)
            return f"{label}({rel}{'#' + anchor if anchor else ''})"
        try:
            repo_path = resolved.relative_to(REPO)
        except ValueError:
            return m.group(0)
        # An image: copy it into the website, because a GitHub page does not show as an image.
        if label.startswith("!") and resolved.is_file():
            copy = DOCS / "assets" / "images" / repo_path
            copy.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(resolved, copy)
            return f"{label}({os.path.relpath(copy, dst.parent)})"
        kind = "tree" if resolved.is_dir() else "blob"
        url = GITHUB.replace("/blob/", f"/{kind}/") + repo_path.as_posix()
        return f"{label}({url}{'#' + anchor if anchor else ''})"

    return LINK.sub(repl, text)


def prepare() -> Path:
    if DOCS.exists():
        shutil.rmtree(DOCS)
    site_pages = pages()
    nav = []
    for src_en, dst_en in site_pages.items():
        for src, dst in ((src_en, dst_en), (zh(src_en), zh(dst_en))):
            if not src.exists():
                print(f"  missing translation: {src.relative_to(REPO)}", file=sys.stderr)
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text(
                rewrite(src.read_text(encoding="utf-8"), src, dst, site_pages), encoding="utf-8"
            )
        nav.append(dst_en.relative_to(DOCS).as_posix())
    cfg = yaml.safe_load((REPO / "site" / "mkdocs.base.yml").read_text(encoding="utf-8"))
    cfg["docs_dir"] = str(DOCS)
    cfg["site_dir"] = str(BUILD / "site")
    chapters = [p for p in nav if p.startswith("chapters/")]
    cfg["nav"] = ["index.md", {"Chapters": chapters}, "style-guide.md"]
    (BUILD / "mkdocs.yml").write_text(
        yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    print(f"prepared {len(nav)} pages ({len(chapters)} chapters) in {DOCS.relative_to(REPO)}")
    return BUILD / "mkdocs.yml"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--build", action="store_true", help="build the static site")
    ap.add_argument("--serve", action="store_true", help="preview the site")
    args = ap.parse_args()
    cfg = prepare()
    if args.build or args.serve:
        cmd = [sys.executable, "-m", "mkdocs", "serve" if args.serve else "build", "-f", str(cfg)]
        if args.build:
            cmd.append("--strict")
        sys.exit(subprocess.run(cmd).returncode)


if __name__ == "__main__":
    main()
