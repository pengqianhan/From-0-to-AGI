# tiny_corpus 来源与许可证

[English](LICENSES.md) · **中文**

这个目录是提交进仓库的极小语料，总计约 2.5 MB。它只用于离线冒烟测试、单元测试和章节演示，**不是**主线模型的训练数据。重新生成的方法见 `build_corpus.py`。

| 文件 | 内容 | 来源 | 许可证 |
|---|---|---|---|
| `shakespeare.txt` | 莎士比亚剧作选段（Tiny Shakespeare，约 1.1 MB，原样复制） | https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt | 公有领域（莎士比亚原作已进入公有领域；char-rnn 仓库本身用 MIT 许可证） |
| `chinese_poetry.txt` | 《论语》《诗经》、水墨唐诗、宋词三百首，以及全宋词的前若干首（均为简体，约 1.2 MB）。从 JSON 转成纯文本，每篇之间空一行 | https://github.com/chinese-poetry/chinese-poetry （2026-09-26 克隆的 master） | MIT License，Copyright (c) 2016 JackeyGao；原典为古籍，属公有领域 |
| `code.txt` | 本仓库 `zero/` 目录下部分 Python 源码的拼接 | 本仓库 https://github.com/pengqianhan/From-0-to-AGI | 与本仓库相同（仓库许可证待作者确定，待核实） |

chinese-poetry 的 MIT 许可证全文：

```
The MIT License (MIT)

Copyright (c) 2016 JackeyGao

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
