# Corpus for the vocabulary measurement: a sample from the main-line pretraining sources (2026-10-01)

**English** · [中文](README.zh.md)

Use: Chapter 13, Section 10 "how large is the vocabulary". We measure the compression of tokenizers on a sample of the main-line pretraining data. This sample replaces the old corpus, which we made from GitHub technical documents (at that time, the build environment could not access Hugging Face). **We use this corpus only for local measurements. We do not use it for training, and it does not go into the repository.**

This is also the first real run of `zero/data/download.py` with the network. Before this run, only unit tests with fake data existed.

## Storage locations

| Content | Path |
|---|---|
| Raw download (one folder for each source: `*.jsonl.gz` shards + `_manifest.json`; each record has its provenance and the data set commit) | `/mnt/DataSets/phan635/From-0-to-AGI/raw_vocab/` (download log `download.log`) |
| Prepared measurement corpus (`{en,zh,code}_{train,val}.txt` + `manifest.json`) | `/mnt/DataSets/phan635/From-0-to-AGI/vocab_corpus/` (build log `../vocab_corpus.build.log`) |
| Hugging Face cache | `HF_HOME=/mnt/DataSets/phan635/huggingface/` (streaming read: it does not download the full data set to the local disk) |

## Commands

```bash
uv pip install --python .venv/bin/python "datasets>=3.0" "zstandard>=0.22"   # = the data extra of pyproject (do not use uv sync: it would replace the cu128 torch)
uv run python -m zero.data.download --config configs/vocab/download.toml --out /mnt/DataSets/phan635/From-0-to-AGI/raw_vocab
uv run python chapters/13-data/code/09_build_vocab_corpus.py --raw /mnt/DataSets/phan635/From-0-to-AGI/raw_vocab \
    --out /mnt/DataSets/phan635/From-0-to-AGI/vocab_corpus
RAYON_NUM_THREADS=16 uv run python chapters/13-data/code/08_vocab_size.py --corpus /mnt/DataSets/phan635/From-0-to-AGI/vocab_corpus \
    --refs --train-mb 30 --json chapters/13-data/video/out/vocab_big.json
```

Environment: datasets 5.0.1, huggingface_hub 1.33.0, pyarrow 25.0.1, zstandard 0.25.0, tokenizers 0.23.2, Python 3.12. Download time: 2026-10-01 15:37–15:45 (NZDT), about 7.5 minutes.

## What we downloaded, and how much

The mixture is the same as in the main line (`configs/main/data.toml`): English 0.55 = FineWeb-Edu 0.35 + DCLM 0.10 + FineMath 0.10; Chinese 0.30 = FineWeb-2 0.20 + Ultra-FineWeb 0.10; code 0.15. **For code, UltraData-Code-L2 replaces Stack-Edu of the main line.** Stack-Edu has only file ids, and the text must come from Software Heritage with AWS credentials. The text of UltraData-Code (the code data of MiniCPM5) is directly in the parquet files, and its license is Apache-2.0. The download of each source stops when it reaches `target_bytes` (`complete=False` means "stopped as planned", not a failure). Each data set version is pinned to the commit of the download day (the full sha is in `configs/vocab/download.toml`).

| Source | Data set | config / split | commit | License | Documents | Text MB | Target MB | Disk MB (gz) |
|---|---|---|---|---|---:|---:|---:|---:|
| fineweb-edu | [HuggingFaceFW/fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) | sample-10BT / train | `87f09149ef` | ODC-By-1.0 | 3,162 | 15.30 | 15.30 | 6.19 |
| dclm-baseline | [mlfoundations/dclm-baseline-1.0](https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0) | — / train | `a3b142c183` | CC-BY-4.0 | 790 | 4.36 | 4.35 | 1.77 |
| finemath | [HuggingFaceTB/finemath](https://huggingface.co/datasets/HuggingFaceTB/finemath) | finemath-3plus / train | `e92b25a616` | ODC-By-1.0 | 439 | 4.39 | 4.35 | 1.07 |
| fineweb-2-zh | [HuggingFaceFW/fineweb-2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) | cmn_Hani / train | `af9c13333e` | ODC-By-1.0 | 2,182 | 8.72 | 8.70 | 4.39 |
| ultra-fineweb-zh | [openbmb/Ultra-FineWeb](https://huggingface.co/datasets/openbmb/Ultra-FineWeb) | — / zh | `02c85641e3` | Apache-2.0 | 992 | 4.30 | 4.30 | 1.97 |
| ultradata-code-py | [openbmb/UltraData-Code](https://huggingface.co/datasets/openbmb/UltraData-Code) | UltraData-Code-L2 / py | `85182d829f` | Apache-2.0 | 543 | 2.72 | 2.70 | 0.74 |
| ultradata-code-js | Same as above | UltraData-Code-L2 / js | `85182d829f` | Apache-2.0 | 185 | 1.35 | 1.35 | 0.35 |
| ultradata-code-java | Same as above | UltraData-Code-L2 / java | `85182d829f` | Apache-2.0 | 236 | 1.35 | 1.35 | 0.30 |
| ultradata-code-cpp | Same as above | UltraData-Code-L2 / cpp | `85182d829f` | Apache-2.0 | 402 | 1.35 | 1.35 | 0.38 |
| ultradata-code-go | Same as above | UltraData-Code-L2 / go | `85182d829f` | Apache-2.0 | 137 | 0.45 | 0.45 | 0.15 |
| ultradata-code-rust | Same as above | UltraData-Code-L2 / rust | `85182d829f` | Apache-2.0 | 86 | 0.46 | 0.45 | 0.12 |
| ultradata-code-sh | Same as above | UltraData-Code-L2 / sh | `85182d829f` | Apache-2.0 | 91 | 0.45 | 0.45 | 0.14 |
| ultradata-code-cs | Same as above | UltraData-Code-L2 / cs | `85182d829f` | Apache-2.0 | 66 | 0.46 | 0.45 | 0.09 |
| ultradata-code-php | Same as above | UltraData-Code-L2 / php | `85182d829f` | Apache-2.0 | 50 | 0.46 | 0.45 | 0.11 |
| **Total** | 6 data sets, 14 sources | | | | **9,361** | **46.13** | | **17.78** |

("Text MB" is the number of UTF-8 text bytes, with 1 MB = 10⁶ bytes. The line count, the byte count, the sha256, and the write time of each shard are in the `_manifest.json` of its source.)

## The prepared measurement corpus

`09_build_vocab_corpus.py` merges the documents by language and shuffles them with a fixed seed. Then it first cuts off a validation set by byte count (about the same size as the old validation set of the chapter text). The rest is the training set (`08_vocab_size.py --train-mb 30` takes the first 30MB × mixture from the training set):

| Language | Set | Documents | Bytes | Byte share of each source |
|---|---|---:|---:|---|
| en | train | 4,072 | 22,295,373 | fineweb-edu 63%, finemath 19%, dclm-baseline 18% |
| en | val | 319 | 1,750,543 | fineweb-edu 74%, dclm-baseline 17%, finemath 9% |
| zh | train | 2,923 | 12,209,078 | fineweb-2-zh 67%, ultra-fineweb-zh 33% |
| zh | val | 251 | 814,096 | fineweb-2-zh 65%, ultra-fineweb-zh 35% |
| code | train | 1,474 | 7,427,020 | py 28%, cpp 16%, js 16%, java 14%, go 6%, sh 5%, cs 5%, php 5%, rust 4% |
| code | val | 322 | 1,626,863 | py 39%, java 18%, js 11%, cpp 11%, rust 9%, php / cs / sh / go 3% each |

The sha256 of each file is in `vocab_corpus/manifest.json`. The validation sets have only a few hundred documents. Thus their source shares are different from the training sets. For example, Python is 39% of the code validation set. A single code file can have tens of KB, so a sample of a few hundred documents gives this result.

## Known problems and limits

- **A bug in the downloader (fixed)**: Ultra-FineWeb has its own `source` field (the name of the upstream corpus, for example "Tele"). When `keep_fields` kept this field, it overwrote our own provenance field `source`. Now the downloader stores a field with the same name as `orig_<field name>` (`make_record` in `zero/data/download.py`, test `test_make_record_keep_field_does_not_clobber_provenance`). The main-line configuration `configs/main/data.toml` also keeps `source` for Ultra-FineWeb, so the real download in Step 2 will use this fix.
- **Crash at exit (not fixed, no effect on the data)**: after a streaming read with `datasets`, the Python interpreter reports `Fatal Python error: PyGILState_Release` at exit, with exit code 134. All shards and manifests are written and checked before this point. Disabling `hf_xet` (`HF_HUB_DISABLE_XET=1`) does not prevent the crash. Disabling the torch integration (`USE_TORCH=0`) also does not prevent it. When you chain scripts, do not rely on the exit code of this command. Use `_manifest.json` as the reference.
- **The sample prefers the oldest web pages**: the streaming read starts from the first parquet file of each data set, and the files are sorted by dump (crawl time). We measured that **all** 3,162 FineWeb-Edu documents come from `CC-MAIN-2013-20`. `sample-10BT` is a random subset, but its files are still sorted by dump. Thus reading only the start gives no spread in time. The FineWeb-2 Chinese documents come from 4 dumps of 2013–2014 (`CC-MAIN-2014-15` 774 documents, `2014-10` 547, `2013-48` 517, the others 344). Thus the web part has no new words from recent years. For the comparison of vocabulary sizes, this is a small problem. But the sample is not an unbiased sample of "current web pages". The real download in Step 2 reads the full data sets; then measure again with the main-line tokenizer. Improvement: use `DownloadSpec.data_files` to select several parquet files from different years.
- **Licenses**: the table above gives the licenses on the data set cards. The DCLM data set card also says "for research only". The upstream licenses of the Chinese part of Ultra-FineWeb are not all the same. UltraData-Code does not give the license of the original repository for each file. `zero/data/sources.py` marks all of these items with "to be verified" notes. This time, we only do measurements locally.

## Results measured with this corpus

`08_vocab_size.py --refs --train-mb 30` (CPU, 16 threads, 3 min 40 s in total; full output in `/mnt/DataSets/phan635/From-0-to-AGI/vocab_size.log`; the numbers for the video are in `chapters/13-data/video/out/vocab_big.json`): FLOPs/byte is lowest at V = 65,536 (0.2412). 98,304 costs 0.5% more, 49,152 costs 0.8% more, 131,072 costs 2.0% more, and 151,936 costs 3.3% more. **The choice of 65,536 for the main-line vocabulary does not change.** Compared with the first version (the corpus of GitHub technical documents), all tokenizers have a lower compression ratio, because web pages are more mixed than technical documents. On Chinese, the advantage of our tokenizer over Qwen decreased from 43% to 20%. On code, the advantage disappeared (now Qwen is 4.5% higher). For details, see Chapter 13, Section 10.
