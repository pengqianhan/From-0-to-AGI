# tiny_corpus

**English** · [中文](README.zh.md)

This folder contains a very small corpus (about 2.5 MB) for offline smoke tests. It has Chinese text, English text, and code: `shakespeare.txt` (English, public domain), `chinese_poetry.txt` (Chinese, from the chinese-poetry repository, MIT), and `code.txt` (code from this repository). [LICENSES.md](LICENSES.md) gives the sources and the licenses. `build_corpus.py` shows how to make the files. `configs/tiny/pretrain.toml` uses the corpus to train a tokenizer and make the shards on the spot.
