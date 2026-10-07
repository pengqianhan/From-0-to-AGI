# Course website

The course website shows the text of each chapter in English and Chinese. English is the default language. The language button in the header opens the same page in the other language.

The Markdown files in the repository are the only source. `site/build.py` copies them into `site/build/docs/` and writes `site/build/mkdocs.yml`. Do not edit the files in `site/build/`.

## Build the website on your computer

1. Install the tools:

   ```bash
   uv pip install --python .venv/bin/python -e ".[site]"
   ```

2. Preview the website at http://127.0.0.1:8000:

   ```bash
   .venv/bin/python site/build.py --serve
   ```

3. Build the static website into `site/build/site/`:

   ```bash
   .venv/bin/python site/build.py --build
   ```

`--build` uses the strict mode of MkDocs. A broken link stops the build.

## What the build script does

- `README.md` becomes the English page. `README.zh.md` becomes the Chinese page.
- A chapter goes on the website only when it has both `README.md` and `README.zh.md`.
- Links between pages stay links between pages.
- Links to code and data go to the files on GitHub.
- Images are copied into the website.
- The `**English** · [中文](README.zh.md)` line under each title is removed, because the website has a language button.

## Publish the website

The workflow `.github/workflows/site.yml` builds the website and publishes it on GitHub Pages. It runs after each push to `main` that changes the text, and you can also start it by hand.

> **Note:** Before the first run, open the repository settings on GitHub. Set **Pages > Source** to **GitHub Actions**.

The configuration is in `site/mkdocs.base.yml`. The writing rules are in [`docs/STYLE_GUIDE.md`](../docs/STYLE_GUIDE.md).
