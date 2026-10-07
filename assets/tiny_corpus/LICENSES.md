# tiny_corpus: sources and licenses

**English** · [中文](LICENSES.zh.md)

This folder contains a very small corpus (about 2.5 MB in total) that is committed to the repository. Use it only for offline smoke tests, unit tests, and chapter demos. It is **not** training data for the main-line model. `build_corpus.py` shows how to make the files again.

| File | Content | Source | License |
|---|---|---|---|
| `shakespeare.txt` | Excerpts from Shakespeare's plays (Tiny Shakespeare, about 1.1 MB, copied unchanged) | https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt | Public domain (the original works of Shakespeare are in the public domain; the char-rnn repository itself uses the MIT License) |
| `chinese_poetry.txt` | The Analects (Lunyu), the Book of Songs (Shijing), Shuimo Tangshi (Tang poems), Three Hundred Song Ci Poems, and the first poems of the Complete Song Ci (all in simplified Chinese, about 1.2 MB). Converted from JSON to plain text, with an empty line between two texts | https://github.com/chinese-poetry/chinese-poetry (master, cloned on 2026-09-26) | MIT License, Copyright (c) 2016 JackeyGao; the original texts are ancient books in the public domain |
| `code.txt` | Concatenation of some Python source files in the `zero/` folder of this repository | This repository: https://github.com/pengqianhan/From-0-to-AGI | The same as this repository (the author has not chosen the repository license yet; to be verified) |

Full text of the MIT License of chinese-poetry:

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
