# 词表测量语料：从主线预训练来源抽样（2026-10-01）

用途：第 13 章第 10 节"词表选多大"。在主线预训练数据的样本上测分词器的压缩率，代替原来用 GitHub 技术文档拼的语料（那时的构建环境访问不了 Hugging Face）。**只用于本地测量，不用于训练，也不进仓库。**

这也是 `zero/data/download.py` 第一次真正联网运行（之前只有用假数据的单元测试）。

## 存放位置

| 内容 | 路径 |
|---|---|
| 原始下载（每个来源一个目录：`*.jsonl.gz` 分片 + `_manifest.json`，每条记录带出处和数据集 commit） | `/mnt/DataSets/phan635/From-0-to-AGI/raw_vocab/`（下载日志 `download.log`） |
| 整理后的测量语料（`{en,zh,code}_{train,val}.txt` + `manifest.json`） | `/mnt/DataSets/phan635/From-0-to-AGI/vocab_corpus/`（整理日志 `../vocab_corpus.build.log`） |
| Hugging Face 缓存 | `HF_HOME=/mnt/DataSets/phan635/huggingface/`（流式读取，不会把整个数据集下载到本地） |

## 命令

```bash
uv pip install --python .venv/bin/python "datasets>=3.0" "zstandard>=0.22"   # 即 pyproject 的 data extra（不用 uv sync，免得换掉 cu128 的 torch）
uv run python -m zero.data.download --config configs/vocab/download.toml --out /mnt/DataSets/phan635/From-0-to-AGI/raw_vocab
uv run python chapters/13-data/code/09_build_vocab_corpus.py --raw /mnt/DataSets/phan635/From-0-to-AGI/raw_vocab \
    --out /mnt/DataSets/phan635/From-0-to-AGI/vocab_corpus
RAYON_NUM_THREADS=16 uv run python chapters/13-data/code/08_vocab_size.py --corpus /mnt/DataSets/phan635/From-0-to-AGI/vocab_corpus \
    --refs --train-mb 30 --json chapters/13-data/video/out/vocab_big.json
```

环境：datasets 5.0.1、huggingface_hub 1.33.0、pyarrow 25.0.1、zstandard 0.25.0、tokenizers 0.23.2，Python 3.12。下载时间 2026-10-01 15:37–15:45（NZDT），约 7.5 分钟。

## 下载了什么、下载了多少

配比和主线一致（`configs/main/data.toml`）：英文 0.55 = FineWeb-Edu 0.35 + DCLM 0.10 + FineMath 0.10，中文 0.30 = FineWeb-2 0.20 + Ultra-FineWeb 0.10，代码 0.15。**代码用 UltraData-Code-L2 代替主线的 Stack-Edu**：Stack-Edu 只有文件 id，正文要用 AWS 凭证从 Software Heritage 取；UltraData-Code（MiniCPM5 的代码数据）的正文直接在 parquet 里，许可证 Apache-2.0。每个来源按 `target_bytes` 下载到字节数就停（`complete=False` 表示"按计划停下"，不是失败）。数据集版本都固定到下载当天的 commit（完整 sha 见 `configs/vocab/download.toml`）。

| 来源 | 数据集 | config / split | commit | 许可证 | 文档 | 正文 MB | 目标 MB | 磁盘 MB（gz） |
|---|---|---|---|---|---:|---:|---:|---:|
| fineweb-edu | [HuggingFaceFW/fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) | sample-10BT / train | `87f09149ef` | ODC-By-1.0 | 3,162 | 15.30 | 15.30 | 6.19 |
| dclm-baseline | [mlfoundations/dclm-baseline-1.0](https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0) | — / train | `a3b142c183` | CC-BY-4.0 | 790 | 4.36 | 4.35 | 1.77 |
| finemath | [HuggingFaceTB/finemath](https://huggingface.co/datasets/HuggingFaceTB/finemath) | finemath-3plus / train | `e92b25a616` | ODC-By-1.0 | 439 | 4.39 | 4.35 | 1.07 |
| fineweb-2-zh | [HuggingFaceFW/fineweb-2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) | cmn_Hani / train | `af9c13333e` | ODC-By-1.0 | 2,182 | 8.72 | 8.70 | 4.39 |
| ultra-fineweb-zh | [openbmb/Ultra-FineWeb](https://huggingface.co/datasets/openbmb/Ultra-FineWeb) | — / zh | `02c85641e3` | Apache-2.0 | 992 | 4.30 | 4.30 | 1.97 |
| ultradata-code-py | [openbmb/UltraData-Code](https://huggingface.co/datasets/openbmb/UltraData-Code) | UltraData-Code-L2 / py | `85182d829f` | Apache-2.0 | 543 | 2.72 | 2.70 | 0.74 |
| ultradata-code-js | 同上 | UltraData-Code-L2 / js | `85182d829f` | Apache-2.0 | 185 | 1.35 | 1.35 | 0.35 |
| ultradata-code-java | 同上 | UltraData-Code-L2 / java | `85182d829f` | Apache-2.0 | 236 | 1.35 | 1.35 | 0.30 |
| ultradata-code-cpp | 同上 | UltraData-Code-L2 / cpp | `85182d829f` | Apache-2.0 | 402 | 1.35 | 1.35 | 0.38 |
| ultradata-code-go | 同上 | UltraData-Code-L2 / go | `85182d829f` | Apache-2.0 | 137 | 0.45 | 0.45 | 0.15 |
| ultradata-code-rust | 同上 | UltraData-Code-L2 / rust | `85182d829f` | Apache-2.0 | 86 | 0.46 | 0.45 | 0.12 |
| ultradata-code-sh | 同上 | UltraData-Code-L2 / sh | `85182d829f` | Apache-2.0 | 91 | 0.45 | 0.45 | 0.14 |
| ultradata-code-cs | 同上 | UltraData-Code-L2 / cs | `85182d829f` | Apache-2.0 | 66 | 0.46 | 0.45 | 0.09 |
| ultradata-code-php | 同上 | UltraData-Code-L2 / php | `85182d829f` | Apache-2.0 | 50 | 0.46 | 0.45 | 0.11 |
| **合计** | 6 个数据集、14 个来源 | | | | **9,361** | **46.13** | | **17.78** |

（"正文 MB"是 UTF-8 正文字节数，按 1 MB = 10⁶ 字节；每个分片的行数、字节数、sha256 和写入时间在各自的 `_manifest.json` 里。）

## 整理后的测量语料

`09_build_vocab_corpus.py` 按语言合并、固定种子打乱，先按字节数切出验证集（大小与正文原来的验证集相当），其余为训练集（`08_vocab_size.py --train-mb 30` 从训练集取前 30MB × 配比）：

| 语言 | 集合 | 文档 | 字节 | 各来源字节占比 |
|---|---|---:|---:|---|
| en | train | 4,072 | 22,295,373 | fineweb-edu 63%、finemath 19%、dclm-baseline 18% |
| en | val | 319 | 1,750,543 | fineweb-edu 74%、dclm-baseline 17%、finemath 9% |
| zh | train | 2,923 | 12,209,078 | fineweb-2-zh 67%、ultra-fineweb-zh 33% |
| zh | val | 251 | 814,096 | fineweb-2-zh 65%、ultra-fineweb-zh 35% |
| code | train | 1,474 | 7,427,020 | py 28%、cpp 16%、js 16%、java 14%、go 6%、sh 5%、cs 5%、php 5%、rust 4% |
| code | val | 322 | 1,626,863 | py 39%、java 18%、js 11%、cpp 11%、rust 9%、php / cs / sh / go 各 3% |

每个文件的 sha256 记在 `vocab_corpus/manifest.json`。验证集只有几百篇文档，来源占比和训练集有偏差（例如代码验证集里 Python 占 39%）：单篇代码文件可能有几十 KB，几百篇的抽样就会这样。

## 已知问题与局限

- **下载器的一个 bug（已修）**：Ultra-FineWeb 自带一个 `source` 字段（上游语料名，如 "Tele"），`keep_fields` 里保留它时会覆盖我们自己的出处字段 `source`。现在重名字段改存为 `orig_<字段名>`（`zero/data/download.py` 的 `make_record`，测试 `test_make_record_keep_field_does_not_clobber_provenance`）。主线配置 `configs/main/data.toml` 的 Ultra-FineWeb 也保留了 `source`，第二步正式下载时会用到这个修复。
- **退出时崩溃（未修，不影响数据）**：用 `datasets` 流式读取后，Python 解释器退出时会报 `Fatal Python error: PyGILState_Release`，退出码 134。所有分片和清单都在此之前写完并核对过；关掉 `hf_xet`（`HF_HUB_DISABLE_XET=1`）、关掉 torch 联动（`USE_TORCH=0`）都不能避免。脚本串联时不要依赖这个命令的退出码，以 `_manifest.json` 为准。
- **样本偏向最早的网页**：流式读取从每个数据集的第一个 parquet 文件开始读，而文件按 dump（抓取时间）排列。实测 FineWeb-Edu 的 3,162 篇**全部**来自 `CC-MAIN-2013-20`——`sample-10BT` 虽然是随机抽样的子集，文件仍按 dump 排序，所以只读开头并没有带来时间上的分散；FineWeb-2 中文来自 2013–2014 年的 4 个 dump（`CC-MAIN-2014-15` 774 篇、`2014-10` 547 篇、`2013-48` 517 篇、其余 344 篇）。也就是说网页部分缺少近几年的新词。用于比较不同词表大小的取舍问题不大，但不是"当前网页"的无偏样本；第二步正式下载会读完整数据集，到时用主线分词器重测一次。改进办法：用 `DownloadSpec.data_files` 指定分散在不同年份的几个 parquet 文件。
- **许可证**：上表是各数据集卡标注的许可证。DCLM 数据集卡同时写了"仅供研究"、Ultra-FineWeb 中文部分的上游许可证不一、UltraData-Code 没有逐文件给出原始仓库的许可证——这些在 `zero/data/sources.py` 里都标了"待核实"的说明；本次只在本地做测量。

## 用这份语料测出的结果

`08_vocab_size.py --refs --train-mb 30`（CPU 16 线程，共 3 分 40 秒；完整输出 `/mnt/DataSets/phan635/From-0-to-AGI/vocab_size.log`，视频用的数字在 `chapters/13-data/video/out/vocab_big.json`）：FLOPs/字节在 V = 65,536 最低（0.2412），98,304 贵 0.5%、49,152 贵 0.8%、131,072 贵 2.0%、151,936 贵 3.3%——**主线词表 65,536 的选择不变**。和第一版（GitHub 技术文档语料）相比，所有分词器的压缩率都更低（网页比技术文档杂），我们的分词器在中文上相对 Qwen 的优势从 43% 缩到 20%，代码上的优势消失（Qwen 反而高 4.5%）。详见第 13 章第 10 节。
