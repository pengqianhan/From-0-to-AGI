# tiny_corpus

[English](README.md) · **中文**

这个目录是离线冒烟测试用的极小语料（约 2.5 MB），含中文、英文和代码：`shakespeare.txt`（英文，公有领域）、`chinese_poetry.txt`（中文，来自 chinese-poetry 仓库，MIT）、`code.txt`（本仓库的代码）。来源与许可证见 [LICENSES.zh.md](LICENSES.zh.md)。生成方法见 `build_corpus.py`。`configs/tiny/pretrain.toml` 用这份语料现场训练分词器并切分片。
