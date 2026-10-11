# References

**English** · [中文](references.zh.md)

This course does not start from nothing. It uses the work of many people before us. These are some important references:
- [CS336](https://cs336.stanford.edu/)
- [minimind](https://github.com/jingyaogong/minimind)
- [nanoGPT](https://github.com/karpathy/nanoGPT)
- [Hands-On Modern Reinforcement Learning](https://walkinglabs.github.io/hands-on-modern-rl/preface/intro)
- [Mathematical theory of deep learning](https://arxiv.org/abs/2407.18384)
- [Scaling Laws That Extrapolate 300× Past the Fit](https://openathena.ai/blog/delphi/)
- [Is 12 GB of GPU memory enough? A complete path for LLM algorithm research on consumer GPUs](https://x.com/dashen_wang/status/2038970877732380924) (in Chinese)
- [Reproducing all of Schmidhuber’s papers (1990-2025)](https://github.com/cybertronai/schmidhuber-problems/blob/main/VISUAL_TOUR.md)
- [The evolution of AI research methods](https://github.com/owlman/CS_StudyNotes/blob/master/02_%E5%9F%BA%E7%A1%80%E7%90%86%E8%AE%BA%E5%AD%A6%E4%B9%A0/%E4%BA%BA%E5%B7%A5%E6%99%BA%E8%83%BD/01.%E5%AD%A6%E4%B9%A0%E7%AC%94%E8%AE%B0/AI%20%E7%A0%94%E7%A9%B6%E6%96%B9%E6%B3%95%E7%9A%84%E6%BC%94%E5%8F%98.md) (in Chinese): the evolution of AI research methods from 1950. It covers the path from symbolism to connectionism, and then to modern deep learning.
- [Advanced Natural Language Processing / Spring 2026 from CMU](https://cmu-l3.github.io/anlp-spring2026/) [code](https://github.com/cmu-l3/anlp-spring2026-code)
- [Maths, CS & AI Compendium](https://github.com/HenryNdubuaku/maths-cs-ai-compendium/tree/main)
- [Pi Book](https://github.com/antinomie-lab/pi-book/tree/main): the format and the interaction style of this book are good for learning. Its content about the Pi agent is also a useful reference.
- [Pretraining a Mini Kimi K3](https://books.vizuara.ai/book/pretraining-a-mini-k3): a reference for the pretraining of the new K3 architecture.
- [The main LLM architectures in 15,000 characters (Llama, Qwen, GLM, DeepSeek…)](https://zhuanlan.zhihu.com/p/2060741715095560795) (in Chinese): this article describes and compares the structures of the main LLMs in detail. It covers the architecture features and the use cases of Llama, Qwen, GLM, DeepSeek, and other models.
- [Small-Scale Experiments: Are We There Yet?](https://arxiv.org/pdf/2608.11859): this paper is about the scaling law of small LLMs. Small models are very sensitive to hyperparameters, and this sensitivity hides the real scaling law. Small-scale experiments reliably predict large-scale results only after sufficient tuning.
- Data sets: fine-web edu, ultra-fineweb-L1
- [Marin 535B-A23B training livestream](https://wandb.ai/marin-community/marin_moe/reports/535B-A23B-18T-Token-Hero-Run-Scaling-Ladder--VmlldzoxNzc2MDM5Ng), [github details](https://github.com/marin-community/marin/issues/8435), [data composition:](https://storage.googleapis.com/marin-public/held/harrier-k40-cluster-overview/2026.08.18/index.html?revision=uniform-sampling)
- [UNDERSTANDING TRANSFORMERS AND ATTENTIONMECHANISMS: AN INTRODUCTION FOR APPLIEDMATHEMATICIANS](https://arxiv.org/pdf/2604.00965)
- [Puro-2B](https://www.alphaxiv.org/abs/2608.27370): the paper *PuRo-2B: Poor Lab’s Qwen2-1.5B Trained on RTX 5090 within $5090*
- [Awesome Claude Opus 5.5 Videos](https://github.com/athemeroy/awesome-opus-5-5-videos): a collection of examples of videos that people made with Claude Opus 5.5. Look mainly at the "educational explainer" path (Manim/Remotion + TTS) and the [seven types of reusable prompt templates](https://github.com/athemeroy/awesome-opus-5-5-videos/blob/main/docs/prompt-playbook.zh-CN.md) (in Chinese). The production process of the video for each chapter uses this collection as a reference.
- [XiaomiMiMo/verl](https://github.com/XiaomiMiMo/verl): the open-source training code of Xiaomi MiMo for agentic reinforcement learning (based on verl 0.9.0.dev, Apache-2.0). It reproduces the RL recipe in Section 7 of the MiMo-V2.6 technical report. It contains five types of RL environment: code (graded by executable tests), cybersecurity, general knowledge work (graded by a rubric), web development (graded visually), and music. The training data [MiMo-V2.6-RL-oss](https://huggingface.co/datasets/XiaomiMiMo/MiMo-V2.6-RL-oss) is also public. Use this repository as a reference for the RL environments and the reward design in Chapter 19. The repository is also a framework that we can evaluate for the real RL training in Step 2.

## Fully open small models (weights + training data + training code)

Criteria for this list: the weights are available for download, **and** the pretraining data is public, **and** the training code or the recipe is public. For the pretraining data, the release gives a link to the data set, or a pipeline that rebuilds the data deterministically from public data. The size is about ≤ 10B parameters, with a focus on ≤ 5B. In 2026-10, we opened each link and verified each license and each release, one by one. Models that meet only some of the criteria are in the "Partly open" list at the end of this section. Each entry in that list says what is missing.

### OpenBMB MiniCPM
- **MiniCPM5-1B / MiniCPM5-2B** (1.08B / 2.52B, a standard dense `LlamaForCausalLM` structure; 1B was released in 2026-05, 2B on 2026-09-07; the checkpoints of each stage are also released: Base, Midtrain (2B only), SFT)
  - Models: [MiniCPM5-1B](https://huggingface.co/openbmb/MiniCPM5-1B), [MiniCPM5-1B-Base](https://huggingface.co/openbmb/MiniCPM5-1B-Base), [MiniCPM5-2B](https://huggingface.co/openbmb/MiniCPM5-2B), [MiniCPM5-2B-Base](https://huggingface.co/openbmb/MiniCPM5-2B-Base), [MiniCPM5-2B-Midtrain](https://huggingface.co/openbmb/MiniCPM5-2B-Midtrain); collection [MiniCPM5](https://huggingface.co/collections/openbmb/minicpm5)
  - Pretraining data (the UltraData tiered data system, collection [UltraData](https://huggingface.co/collections/openbmb/ultradata)): 1B uses [Ultra-FineWeb](https://huggingface.co/datasets/openbmb/Ultra-FineWeb) (L2 selected web pages, about 1T English + 120B Chinese tokens; its L1 base material is [Ultra-FineWeb-L1](https://huggingface.co/datasets/openbmb/Ultra-FineWeb-L1)), [Ultra-FineWeb-L3](https://huggingface.co/datasets/openbmb/Ultra-FineWeb-L3) (synthetic data from Q&A generation + rephrasing in many styles, 400B+ English + 200B+ Chinese tokens, used in the decay phase), and [UltraData-Math](https://huggingface.co/datasets/openbmb/UltraData-Math) (290B+ tokens, three levels L1/L2/L3). 2B adds [UltraX-Preview](https://huggingface.co/datasets/openbmb/UltraX-Preview) (web data refined by item-by-item edits through function calls, 5 parts of about 20B tokens each) and [UltraData-Code](https://huggingface.co/datasets/openbmb/UltraData-Code) (code data in tiers L0–L3)
  - Post-training data: [UltraData-SFT-2605](https://huggingface.co/datasets/openbmb/UltraData-SFT-2605) (all core-domain SFT data of 1B-SFT; request access on HF), [UltraData-SFT-Agent-2609](https://huggingface.co/datasets/openbmb/UltraData-SFT-Agent-2609) (about 500,000 agent trajectories), [UltraData-RL-2609](https://huggingface.co/datasets/openbmb/UltraData-RL-2609) (the RL data of 2B, 80,000+ tasks with verifiable rewards). The RL of 1B also used public DAPO-Math-17k, TriviaQA, NQ-Open, and others, plus synthetic RLVR data and pairwise RLHF signals that are not released separately
  - Code / recipe: the "Training Recipe" section and the flowchart in [MiniCPM GitHub](https://github.com/OpenBMB/MiniCPM) give a stage-level recipe (stable → short decay 4K → long decay 32K/128K → mid-training → SFT → several RL teachers → on-policy distillation OPD; the decay, mid-training, and SFT stages show token counts). The repository also has fine-tuning scripts and fine-tuning cookbooks for TRL / LLaMA-Factory and others. The code for the data is in [UltraX](https://github.com/openbmb/UltraX) and [UltraData-Math](https://github.com/UltraData-OpenBMB/UltraData-Math). **The pretraining code, the token count of the stable phase, and the data mixture of MiniCPM5 are not public.** The OpenBMB pretraining framework [ForgeTrain](https://github.com/OpenBMB/ForgeTrain) (Apache-2.0) covers only MiniCPM4-0.5B/8B now; a MiniCPM5-1B version is on the roadmap
  - Technical report: no dedicated MiniCPM5 report yet. For the data system, see [Data Science and Technology Towards AGI Part I: Tiered Data Management](https://arxiv.org/abs/2602.09003). Also see [Ultra-FineWeb](https://arxiv.org/abs/2505.05427), [UltraX](https://arxiv.org/abs/2607.08646), [MiniCPM4](https://arxiv.org/abs/2506.07900)
  - License: model Apache-2.0. All the data set cards above say Apache-2.0 (the UltraX data card says that you must also obey the license of each source data set)
  - Use in this course: "L0–L4 tiered management" of bilingual Chinese-English pretraining data, and synthetic rephrasing → Chapter 13; stable + decay in segments (WSD) → Chapters 6/12; decay in segments by context length → Chapter 15; several RL teachers + on-policy distillation → Chapters 17/19; a large amount of Chinese data (Ultra-FineWeb has about 120B Chinese tokens, and the L3 synthetic data has 200B+ Chinese tokens), so it is the first-choice reference for the Chinese data of the main-line model → main-line model

### Hugging Face Smol
As of 2026-10, the newest Smol text model from HuggingFaceTB is still SmolLM3. We did not find SmolLM4 or a newer official release.
- **SmolLM3** (3B, 2025-07; three-stage pretraining on 11.2T tokens + long context 4K→32K→64K + reasoning mid-training on 140B tokens + SFT + APO)
  - Models: [SmolLM3-3B-Base](https://huggingface.co/HuggingFaceTB/SmolLM3-3B-Base), [SmolLM3-3B](https://huggingface.co/HuggingFaceTB/SmolLM3-3B), intermediate checkpoints [SmolLM3-3B-checkpoints](https://huggingface.co/HuggingFaceTB/SmolLM3-3B-checkpoints)
  - Pretraining data: [SmolLM3 pretraining data collection](https://huggingface.co/collections/HuggingFaceTB/smollm3-pretraining-datasets-685a7353fdc01aecde51b1d9) (15 data sets: FineWeb-Edu, DCLM, FineWeb2 / FineWeb2-HQ, FineMath, MegaMath, Stack-Edu, The Stack v2, dolmino-mix, and others); post-training data [smoltalk2](https://huggingface.co/datasets/HuggingFaceTB/smoltalk2) (three subsets: Mid / SFT / Preference)
  - Code / recipe: [huggingface/smollm](https://github.com/huggingface/smollm/tree/main/text/pretraining/smollm3) (5 nanotron configs that give the data weights of each stage), [smollm3-configs](https://huggingface.co/datasets/HuggingFaceTB/smollm3-configs), post-training [alignment-handbook recipes/smollm3](https://github.com/huggingface/alignment-handbook/tree/main/recipes/smollm3)
  - Technical report: [SmolLM3 blog](https://huggingface.co/blog/smollm3) (no arXiv paper); companion long article [The Smol Training Playbook](https://huggingface.co/spaces/HuggingFaceTB/smol-training-playbook)
  - License: model Apache-2.0; the data follows each source (the FineWeb family is ODC-BY, DCLM is CC-BY-4.0, The Stack v2 requires that you agree to its terms, and natural_reasoning in stage 3 is CC-BY-NC-4.0)
  - Note: the paths in the configs point to tokenized data on an internal S3. Only stage 1 gives the matching HF data sets. In the later stages, a few data names have no matching public repository
  - Use in this course: multi-stage data mixture (web pages → a larger share of code/math) → Chapter 13; long-context extension in stages → Chapter 15; mid-training + SFT + APO → Chapters 15/16/18; the ablation methods and the training troubleshooting in the Playbook → Chapters 12/14
- **SmolLM2** (135M / 360M / 1.7B, 2024-10; 1.7B trained on 11T tokens in four stages, 360M/135M trained on 4T/2T tokens)
  - Models: [SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M), [SmolLM2-360M](https://huggingface.co/HuggingFaceTB/SmolLM2-360M), [SmolLM2-1.7B](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B), and an intermediate-checkpoints repository for each size
  - Data: [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu), [DCLM-Edu](https://huggingface.co/datasets/HuggingFaceTB/dclm-edu), [FineMath](https://huggingface.co/datasets/HuggingFaceTB/finemath), [Stack-Edu](https://huggingface.co/datasets/HuggingFaceTB/stack-edu) (only Software Heritage IDs; download the file contents separately under the terms of The Stack v2), [SmolLM-Corpus](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus) (includes Cosmopedia v2); post-training [smoltalk](https://huggingface.co/datasets/HuggingFaceTB/smoltalk)
  - Code / recipe: [smollm/text/pretraining/smollm2](https://github.com/huggingface/smollm/tree/main/text/pretraining/smollm2) (nanotron configs, but the mixture of each stage is only in the paper), [alignment-handbook recipes/smollm2](https://github.com/huggingface/alignment-handbook/tree/main/recipes/smollm2)
  - Technical report: [SmolLM2: When Smol Goes Big](https://arxiv.org/abs/2502.02737)
  - License: model Apache-2.0; data ODC-BY / CC-BY-4.0 / The Stack v2 terms
  - Use in this course: multi-stage training that changes the mixture "online" from evaluations during training → Chapter 13; overtraining (1.7B trained on 11T) → Chapter 12; the small sizes (135M/360M) are good for controlled experiments on one GPU → main-line model
- **SmolLM** (135M / 360M / 1.7B, 2024-07; 600B / 600B / 1T tokens)
  - Models: [SmolLM-135M](https://huggingface.co/HuggingFaceTB/SmolLM-135M), [SmolLM-360M](https://huggingface.co/HuggingFaceTB/SmolLM-360M), [SmolLM-1.7B](https://huggingface.co/HuggingFaceTB/SmolLM-1.7B)
  - Data: [SmolLM-Corpus](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus) (Cosmopedia v2 28B + FineWeb-Edu-dedup 220B + Python-Edu 4B tokens, ODC-BY); the real configs also mix in small amounts of open-web-math, StackOverflow, and other data
  - Code / recipe: [smollm/text/pretraining/smollm1](https://github.com/huggingface/smollm/tree/main/text/pretraining/smollm1) (nanotron YAML files for the three sizes)
  - Technical report: [SmolLM blog](https://huggingface.co/blog/smollm)
  - License: model Apache-2.0, data ODC-BY
  - Use in this course: a small amount of data and a simple config. It is the simplest starting point to reproduce a "full-pipeline small model" → main-line model, Chapter 14
- **Related data and code**
  - [FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb) (ODC-BY, about 18.5T tokens; paper [arXiv 2406.17557](https://arxiv.org/abs/2406.17557)), [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) (ODC-BY), [FineWeb2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) (ODC-BY, 1000+ languages; paper [arXiv 2506.20920](https://arxiv.org/abs/2506.20920)), [Cosmopedia](https://huggingface.co/datasets/HuggingFaceTB/cosmopedia) (Apache-2.0, synthetic textbooks) → Chapter 13
  - [nanotron](https://github.com/huggingface/nanotron) (Apache-2.0, a pretraining framework with 3D parallelism) → Chapter 14; [huggingface/smollm](https://github.com/huggingface/smollm) (Apache-2.0; contains the FineWeb-Edu classifier, the FineMath pipeline, and scripts for decontamination and evaluation) → Chapters 11/13

### NVIDIA
We checked each model. NVIDIA has **no** model under 10B that can go into the main list. Nemotron Nano 2 and Nemotron 3 Nano 4B released only "most of" their pretraining data, and access requires a request (see "Partly open" below). The model cards of Nemotron-H, Hymba, Nemotron-Flash, Minitron, Nemotron-Mini, and others do not give the complete training data for download. OpenReasoning-Nemotron and OpenCodeReasoning-Nemotron are fine-tuned from Qwen2.5. But the data sets that NVIDIA released are very useful references:
- **Nemotron pretraining data** (released together with Nano 2, about 6.6T tokens in total)
  - Data: [Nemotron-CC v1](https://data.commoncrawl.org/contrib/Nemotron/Nemotron-CC/index.html) (6.3T tokens, hosted on Common Crawl, under the Common Crawl terms of use; paper [Nemotron-CC](https://arxiv.org/abs/2412.02595)), [Nemotron-CC-v2](https://huggingface.co/datasets/nvidia/Nemotron-CC-v2), [Nemotron-CC-Math-v1](https://huggingface.co/datasets/nvidia/Nemotron-CC-Math-v1) (133B tokens; paper [arXiv 2508.15096](https://arxiv.org/abs/2508.15096)), [Nemotron-Pretraining-Code-v1](https://huggingface.co/datasets/nvidia/Nemotron-Pretraining-Code-v1) (for GitHub code, it gives only the repo / commit / path metadata; you must crawl the code again yourself), [Nemotron-Pretraining-SFT-v1](https://huggingface.co/datasets/nvidia/Nemotron-Pretraining-SFT-v1); collection [Nemotron pre-training dataset](https://huggingface.co/collections/nvidia/nemotron-pre-training-dataset-689d9de36f84279d83786b35), a sample with no access request [Nemotron-Pretraining-Dataset-sample](https://huggingface.co/datasets/nvidia/Nemotron-Pretraining-Dataset-sample)
  - New in the Nemotron 3 period: [Nemotron-CC-v2.1](https://huggingface.co/datasets/nvidia/Nemotron-CC-v2.1), [Nemotron-CC-Code-v1](https://huggingface.co/datasets/nvidia/Nemotron-CC-Code-v1), [Nemotron-Pretraining-Code-v2](https://huggingface.co/datasets/nvidia/Nemotron-Pretraining-Code-v2) (access request required), and [Nemotron-Pretraining-Specialized-v1](https://huggingface.co/datasets/nvidia/Nemotron-Pretraining-Specialized-v1) with no access request (270.7B tokens, CC-BY-4.0; the Wiki-rephrasing subset is CC-BY-SA, and the scientific-code subset is GFDL)
  - Post-training data: [Nemotron-Post-Training-Dataset-v1](https://huggingface.co/datasets/nvidia/Nemotron-Post-Training-Dataset-v1), [Nemotron-Post-Training-Dataset-v2](https://huggingface.co/datasets/nvidia/Nemotron-Post-Training-Dataset-v2) (access request required), [Llama-Nemotron-Post-Training-Dataset](https://huggingface.co/datasets/nvidia/Llama-Nemotron-Post-Training-Dataset) (all CC-BY-4.0)
  - License: for the data sets with the gated mark, you must agree to the "NVIDIA Data Agreement for Model Training". It allows use only for model training. It is not a CC or OSI license
  - Use in this course: the "classifier-ensemble scoring + synthetic rephrasing of low-quality data" of Nemotron-CC → Chapter 13; the extraction of math web pages in Nemotron-CC-Math → Chapter 13; the post-training data sets → Chapters 16/17
- Training frameworks: [Megatron-LM](https://github.com/NVIDIA/Megatron-LM), [Megatron-Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge), [NeMo-RL](https://github.com/NVIDIA-NeMo/RL) (all Apache-2.0) → Chapters 14/19. The recipe repository [NVIDIA-NeMo/Nemotron](https://github.com/NVIDIA-NeMo/Nemotron) now gives the full pipeline only for large models such as 30B-A3B. It says that it uses only an open-source subset of the data, so it does not reproduce the scores in the report

### AllenAI
The page [allenai.org/open-models](https://allenai.org/open-models) is old (it lists only OLMo 2, Molmo, and the OLMoE App). For a summary of the newest models, data, and code, see [allenai.org/olmo](https://allenai.org/olmo). All Ai2 models come with all intermediate checkpoints and training logs.
- **Olmo 3** (7B, 2025-11; a 32B version is outside the scope): pretraining on 5.9T tokens → Dolmino mid-training on 100B tokens → Longmino long context on 50B tokens (65K). The post-training has three lines: Think / Instruct / RL-Zero
  - Models: [Olmo-3-1025-7B](https://huggingface.co/allenai/Olmo-3-1025-7B) (Base), [Olmo-3-7B-Think](https://huggingface.co/allenai/Olmo-3-7B-Think), [Olmo-3-7B-Instruct](https://huggingface.co/allenai/Olmo-3-7B-Instruct), each with checkpoints of the SFT / DPO stages; the RL-Zero series (Math / Code / IF / General / Mix, RL directly from Base)
  - Data: pretraining [dolma3_mix-6T-1025-7B](https://huggingface.co/datasets/allenai/dolma3_mix-6T-1025-7B) (made specially to reproduce the 7B model; some olmOCR PDF text is replaced with `[REMOVED]`), mid-training [dolma3_dolmino_mix-100B-1025](https://huggingface.co/datasets/allenai/dolma3_dolmino_mix-100B-1025), long context [dolma3_longmino_mix-50B-1025](https://huggingface.co/datasets/allenai/dolma3_longmino_mix-50B-1025), the raw pool before mixing [dolma3_pool](https://huggingface.co/datasets/allenai/dolma3_pool) (about 9.3T); post-training data from the Dolci series (for example [Dolci-Instruct-SFT](https://huggingface.co/datasets/allenai/Dolci-Instruct-SFT))
  - Code / recipe: [OLMo-core](https://github.com/allenai/OLMo-core) (`src/scripts/official/OLMo3/`), [open-instruct](https://github.com/allenai/open-instruct), data rebuild [dolma3](https://github.com/allenai/dolma3), evaluation [OLMES](https://github.com/allenai/olmes)
  - Technical report: [Olmo 3](https://arxiv.org/abs/2512.13961)
  - License: model Apache-2.0; Dolma 3 / Dolci are ODC-BY (some Dolci subsets follow the license of their source)
  - Use in this course: the three stages pretraining → mid-training → long context are a complete comparison for Chapters 14/15 of this course; the RL-Zero and Think lines → Chapter 19; the decontamination tool decon → Chapter 11. 7B is larger than the main-line size, but it is now the most complete example of a "full-pipeline reproduction" → main-line model
- **OLMo 2** (1B / 7B; 7B was released in 2024-11, 1B between 2025-04 and 05; 13B and 32B are outside the scope)
  - Models: [OLMo-2-0425-1B](https://huggingface.co/allenai/OLMo-2-0425-1B) (1.48B, 4T tokens), [OLMo-2-1124-7B](https://huggingface.co/allenai/OLMo-2-1124-7B) (4T tokens), both with SFT / DPO / Instruct checkpoints
  - Data: pretraining [olmo-mix-1124](https://huggingface.co/datasets/allenai/olmo-mix-1124), mid-training [dolmino-mix-1124](https://huggingface.co/datasets/allenai/dolmino-mix-1124), SFT [tulu-3-sft-olmo-2-mixture](https://huggingface.co/datasets/allenai/tulu-3-sft-olmo-2-mixture)
  - Code / recipe: [OLMo](https://github.com/allenai/OLMo) (the stage configs for 1B / 7B are in `configs/official-0425/` and `configs/official-1124/`; this repository is now marked as deprecated, and the new code is in OLMo-core), [open-instruct](https://github.com/allenai/open-instruct)
  - Technical report: [2 OLMo 2 Furious](https://arxiv.org/abs/2501.00656) (already cited in the Chapter 6 entries)
  - License: model Apache-2.0; olmo-mix / dolmino-mix are ODC-BY (the DCLM part is CC-BY-4.0, and the StackExchange part is CC-BY-SA)
  - Use in this course: the stability recipe RMSNorm + QK-Norm + z-loss → Chapters 6/9; a change to high-quality data in the annealing phase, then a "model soup" (the average of several annealing results) → Chapter 15; RLVR → Chapter 19. The size of the 1B version is the nearest to the main-line model of this course → main-line model
- **OLMo / OLMo 0424 / OLMo 0724** (1B, 7B, 2024)
  - Models: [OLMo-1B](https://huggingface.co/allenai/OLMo-1B), [OLMo-7B](https://huggingface.co/allenai/OLMo-7B), [OLMo-7B-0424-hf](https://huggingface.co/allenai/OLMo-7B-0424-hf), [OLMo-1B-0724-hf](https://huggingface.co/allenai/OLMo-1B-0724-hf), [OLMo-7B-0724-hf](https://huggingface.co/allenai/OLMo-7B-0724-hf)
  - Data: [Dolma](https://huggingface.co/datasets/allenai/dolma) (v1.5 / v1.7, ODC-BY); data toolkit [dolma](https://github.com/allenai/dolma)
  - Code / recipe: [OLMo](https://github.com/allenai/OLMo)
  - Technical report: [OLMo: Accelerating the Science of Language Models](https://arxiv.org/abs/2402.00838), [Dolma](https://arxiv.org/abs/2402.00159)
  - License: model Apache-2.0, data ODC-BY
  - Use in this course: one checkpoint every 1000 steps, good for observing training dynamics → Chapters 12/14; the cleaning pipeline of Dolma → Chapter 13
- **OLMoE** (an MoE with 6.9B total parameters and 1.3B active parameters, 2024-09 / 2025-01)
  - Models: [OLMoE-1B-7B-0924](https://huggingface.co/allenai/OLMoE-1B-7B-0924), [OLMoE-1B-7B-0125](https://huggingface.co/allenai/OLMoE-1B-7B-0125)
  - Data: [OLMoE-mix-0924](https://huggingface.co/datasets/allenai/OLMoE-mix-0924) (ODC-BY, mostly DCLM-Baseline); the annealing of the 0125 version also uses dolmino-mix-1124
  - Code / recipe: [allenai/OLMoE](https://github.com/allenai/OLMoE) (Apache-2.0)
  - Technical report: [OLMoE: Open Mixture-of-Experts Language Models](https://arxiv.org/abs/2409.02060)
  - License: model Apache-2.0, data ODC-BY
  - Use in this course: a fully open small MoE, with public data for the ablations of routing, load balancing, and the number of experts → Chapter 24
- **Olmo Hybrid** (7.4B, 2026-03): a hybrid architecture with 3 Gated DeltaNet layers for each attention layer
  - Models: [Olmo-Hybrid-7B](https://huggingface.co/allenai/Olmo-Hybrid-7B) (also SFT and DPO versions of Instruct / Think)
  - Data: [dolma3_mix-6T](https://huggingface.co/datasets/allenai/dolma3_mix-6T) (5.5T tokens) + dolmino 100B + longmino 50B (the same as Olmo 3)
  - Code / recipe: [OLMo-core](https://github.com/allenai/OLMo-core) (`src/scripts/official/OLMo-hybrid/`)
  - Technical report: [Olmo Hybrid: From Theory to Practice and Back](https://arxiv.org/abs/2604.03444)
  - License: model Apache-2.0, data ODC-BY
  - Use in this course: a comparison of a linear-attention hybrid architecture with pure attention on the same data → Chapter 23
- **Bolmo** (1B / 7B, 2025-12): OLMo-2-0425-1B and Olmo-3-1025-7B, converted into byte-level models
  - Models: [Bolmo-1B](https://huggingface.co/allenai/Bolmo-1B), [Bolmo-7B](https://huggingface.co/allenai/Bolmo-7B); data [bolmo_mix](https://huggingface.co/datasets/allenai/bolmo_mix) (ODC-BY); code [bolmo-core](https://github.com/allenai/bolmo-core); report [arXiv 2512.15586](https://arxiv.org/abs/2512.15586); model Apache-2.0
  - Use in this course: a method that models bytes directly, with no BPE tokenization → Chapter 7

### Other models
- **nanochat (Karpathy)** (d32 about 1.9B, d34 about 2.2B, 2025-10 / 2025-11): a complete minimal implementation on a single 8×H100 node, from the tokenizer, pretraining, SFT, and evaluation to a chat interface
  - Models: [nanochat-d32](https://huggingface.co/karpathy/nanochat-d32), [nanochat-d34](https://huggingface.co/karpathy/nanochat-d34) (native `.pt` checkpoints, not the transformers format)
  - Data: pretraining [fineweb-edu-100b-shuffle](https://huggingface.co/datasets/karpathy/fineweb-edu-100b-shuffle) (since 2026-03, the repository uses [climbmix-400b-shuffle](https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle) by default); SFT uses SmolTalk and other public data
  - Code / recipe: [karpathy/nanochat](https://github.com/karpathy/nanochat) (one script, `runs/speedrun.sh`, runs the full pipeline; you change only one setting, `--depth`, and the code calculates the other hyperparameters from the compute-optimal rule)
  - License: MIT
  - Use in this course: the full-pipeline reference implementation that is nearest to the main-line model of this course in size and budget → main-line model; the "GPT-2 speedrun" leaderboard records the speedup from each training improvement → Chapter 14
- **MobileLLM-R1 (Meta)** (140M / 360M / 950M, 2025-09; models under 1B for on-device reasoning)
  - Models: [MobileLLM-R1-950M](https://huggingface.co/facebook/MobileLLM-R1-950M), [MobileLLM-R1-950M-base](https://huggingface.co/facebook/MobileLLM-R1-950M-base), [MobileLLM-R1-360M](https://huggingface.co/facebook/MobileLLM-R1-360M), [MobileLLM-R1-140M](https://huggingface.co/facebook/MobileLLM-R1-140M) (a download requires manual approval on HF)
  - Data: the data for pretraining, mid-training, and SFT (FineWeb-Edu, StarCoderData, FineMath, dolmino-mix-1124, Nemotron-CC-Math-v1, OpenMathReasoning, tulu-3-sft-olmo-2-mixture, and others) are all public data sets. The README gives the mixture of each stage. There is no packaged corpus, and the NVIDIA data requires an access request
  - Code / recipe: [facebookresearch/MobileLLM-R1](https://github.com/facebookresearch/MobileLLM-R1) (the reproduction code and the data mixtures for pretraining, mid-training, and SFT; the links to the intermediate checkpoints say "coming soon")
  - Technical report: [MobileLLM-R1: Exploring the Limits of Sub-Billion Language Model Reasoners with Open Training Recipes](https://arxiv.org/abs/2509.24945)
  - License: **FAIR Noncommercial Research License**; the data follows each source
  - Use in this course: distillation in mid-training with Llama-3.1-8B-Instruct as the teacher, which minimizes the KL of the logits → Chapter 17; the reasoning-data recipe for models under 1B → Chapters 15/16
- **K2-Horizon (LLM360 / IFM)** (nominal 0.9B / 3.7B / 7B; the total parameters that HF counts are 1.08B / 5.06B / 9.0B; 2026-09): 0.9B is pretrained on 5T tokens, and 3.7B / 7B on 22.9T tokens. Then two mid-training stages extend the context to 128K. Then several RL experts → weight merging → on-policy distillation
  - Models: [K2-Horizon-0.9B](https://huggingface.co/IFM/K2-Horizon-0.9B), [K2-Horizon-3.7B](https://huggingface.co/IFM/K2-Horizon-3.7B), [K2-Horizon-7B](https://huggingface.co/IFM/K2-Horizon-7B) (the checkpoints of pretraining, mid-training, and each RL expert are in branches, for example `pretrain_600000`)
  - Data: [TxT360-v2](https://huggingface.co/datasets/IFM/TxT360-v2) (CC-BY-4.0), plus four related data sets: Code-Reasoning, Math-Reasoning, SFT-Reasoning, Pretrain-Behaviors
  - Code / recipe: [xLLM](https://github.com/ifm-ai/xllm) (Apache-2.0, a long-context training framework); the model cards list the steps, the token count, and the sequence length of each stage, with W&B training logs
  - Technical report: the model card says "in preparation"; as of 2026-10-01, the report is not released. Whether the public data covers the full mixture of the 22.9T / 5T tokens is **to be verified**
  - License: model Apache-2.0; data CC-BY-4.0
  - Use in this course: the table of the token budget of each stage; merging RL experts, then on-policy distillation → Chapters 15/17/19; the size of the 0.9B version is near the main-line model
- **YuLan-Mini (Renmin University of China, RUC-GSAI)** (2.4B, 2024-12; 1.08T tokens): bilingual Chinese and English, relatively strong in math and code
  - Models: [YuLan-Mini](https://huggingface.co/yulan-team/YuLan-Mini)
  - Data: [YuLan-Mini-Datasets](https://huggingface.co/datasets/yulan-team/YuLan-Mini-Datasets) (includes the data mixture of each stage)
  - Code / recipe: [YuLan-Mini/pretrain](https://github.com/RUC-GSAI/YuLan-Mini/tree/main/pretrain) (`train.py`, launch scripts for each stage, code for data preprocessing and synthesis)
  - Technical report: [YuLan-Mini: An Open Data-efficient Language Model](https://arxiv.org/abs/2412.17743)
  - License: model MIT; the data follows each source
  - Use in this course: the data-efficient recipe that makes a strong 2B model from only 1T tokens, the handling of training stability, and annealing → Chapters 6/13/15; Chinese data → main-line model
- **Instella (AMD)** (3B, 2025-03; also Long-Instruct and Math versions)
  - Models: [Instella-3B](https://huggingface.co/amd/Instella-3B) (also Stage1, SFT, and Instruct checkpoints)
  - Data: the first stage (4.07T tokens) uses [OLMoE-mix-0924](https://huggingface.co/datasets/allenai/OLMoE-mix-0924); the second stage uses dolmino-mix-1124 and other public data, plus [Instella-GSM8K-synthetic](https://huggingface.co/datasets/amd/Instella-GSM8K-synthetic), which AMD synthesized itself
  - Code / recipe: [AMD-AIG-AIMA/Instella](https://github.com/AMD-AIG-AIMA/Instella) (includes data-preparation scripts and configs)
  - Technical report: [Instella: Fully Open Language Models with Stellar Performance](https://arxiv.org/abs/2511.10628)
  - License: **RESEARCH-ONLY RAIL-MS, research use only**
  - Use in this course: an example that reproduces a 3B model on non-NVIDIA hardware with only public data from other people (the data of Ai2) → Chapter 14
- **Ettin decoders (JHU)** (17M / 32M / 68M / 150M / 400M / 1B, 2025-07; released in pairs with encoders that use the same recipe)
  - Models: for example [ettin-decoder-400m](https://huggingface.co/jhu-clsp/ettin-decoder-400m), [ettin-decoder-1b](https://huggingface.co/jhu-clsp/ettin-decoder-1b)
  - Data: [ettin-pretraining-data](https://huggingface.co/datasets/jhu-clsp/ettin-pretraining-data) (also the data of the extension and decay phases, and the batch-by-batch training order [ettin-data-order](https://huggingface.co/datasets/jhu-clsp/ettin-data-order))
  - Code / recipe: [ettin-encoder-vs-decoder](https://github.com/JHU-CLSP/ettin-encoder-vs-decoder)
  - Technical report: [Seq vs Seq: An Open Suite of Paired Encoders and Decoders](https://arxiv.org/abs/2507.11412)
  - License: MIT
  - Use in this course: 6 sizes with the same data and the same recipe, good for small scaling experiments on one GPU → Chapter 12
- **TinyLlama** (1.1B, 2023–2024; 3T tokens)
  - Models: [TinyLlama-1.1B-intermediate-step-1431k-3T](https://huggingface.co/TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T), [TinyLlama_v1.1](https://huggingface.co/TinyLlama/TinyLlama_v1.1)
  - Data: SlimPajama (the official repository cerebras/SlimPajama-627B is no longer available; now there is only a community re-upload, [SlimPajama-627B_Reupload](https://huggingface.co/datasets/gmongaras/SlimPajama-627B_Reupload)) + [starcoderdata](https://huggingface.co/datasets/bigcode/starcoderdata) (you must agree to the terms)
  - Code / recipe: [jzhang38/TinyLlama](https://github.com/jzhang38/TinyLlama) (based on lit-gpt, includes PRETRAIN.md)
  - Technical report: [TinyLlama: An Open-Source Small Language Model](https://arxiv.org/abs/2401.02385)
  - License: Apache-2.0
  - Use in this course: an example of overtraining (a 1B model trained to 3T tokens); throughput optimization on one machine with many GPUs → Chapters 12/14
- **Pythia (EleutherAI)** (14M–12B; the sizes ≤5B are 14M / 31M / 70M / 160M / 410M / 1B / 1.4B / 2.8B; each size has a standard version and a deduplicated version; 2023)
  - Models: [pythia-1b](https://huggingface.co/EleutherAI/pythia-1b), [pythia-1b-deduped](https://huggingface.co/EleutherAI/pythia-1b-deduped), and others; each model has 154 checkpoint branches (`step0` … `step143000`)
  - Data: the original Pile was taken down because of copyright problems. You can still get the tokenized data in training order, [pile-deduped-pythia-preshuffled](https://huggingface.co/datasets/EleutherAI/pile-deduped-pythia-preshuffled) and [pile-standard-pythia-preshuffled](https://huggingface.co/datasets/EleutherAI/pile-standard-pythia-preshuffled), and the raw text [the_pile_deduplicated](https://huggingface.co/datasets/EleutherAI/the_pile_deduplicated) (it still contains disputed parts such as Books3)
  - Code / recipe: [pythia](https://github.com/EleutherAI/pythia) (the configs of each size), [gpt-neox](https://github.com/EleutherAI/gpt-neox)
  - Technical report: [Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling](https://arxiv.org/abs/2304.01373)
  - License: model Apache-2.0; the data follows each source, and the licenses are not uniform
  - Use in this course: the complete training trajectories of 8 sizes on the same data in the same order; the best choice for class experiments on scaling and training dynamics → Chapter 12
- **DCLM-Baseline models** (1.4B / 6.9B, 2024)
  - Models: [TRI-ML/DCLM-1B](https://huggingface.co/TRI-ML/DCLM-1B) (4.3T tokens, Apache-2.0), [apple/DCLM-7B](https://huggingface.co/apple/DCLM-7B) (2.5T tokens, Apple Sample Code License)
  - Data: [dclm-baseline-1.0](https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0) (CC-BY-4.0), also mixed with StarCoderData and [proof-pile-2](https://huggingface.co/datasets/EleutherAI/proof-pile-2)
  - Code / recipe: [mlfoundations/dclm](https://github.com/mlfoundations/dclm) (MIT, based on OpenLM; it has configs for each scale, but no config is marked as the one for the official 7B training)
  - Technical report: [DataComp-LM](https://arxiv.org/abs/2406.11794)
  - Use in this course: the design of a "data competition" that fixes the model and the training recipe and compares only the data-filtering methods; the fastText quality classifier → Chapter 13
- **Luciole (OpenLLM-France / LINAGORA)** (1B / 8B, 2026-02; about 5T tokens, 30% French)
  - Models: [Luciole-1B-Base](https://huggingface.co/OpenLLM-France/Luciole-1B-Base), [Luciole-8B-Base](https://huggingface.co/OpenLLM-France/Luciole-8B-Base) (8B has a hybrid Mamba-attention structure)
  - Data: [Luciole-Training-Dataset](https://huggingface.co/datasets/OpenLLM-France/Luciole-Training-Dataset) (about 4.65T tokens, CC-BY-SA-4.0)
  - Code / recipe: [Luciole-Training](https://github.com/OpenLLM-France/Luciole-Training) (GPL-3.0; data processing, tokenizer, NeMo pretraining, SFT, evaluation)
  - Technical report: the model card says "coming soon"; for the previous model Lucie-7B, see [arXiv 2503.12294](https://arxiv.org/abs/2503.12294) (its data [Lucie-Training-Dataset](https://huggingface.co/datasets/OpenLLM-France/Lucie-Training-Dataset) is CC-BY-NC-SA)
  - License: model Apache-2.0; data CC-BY-SA-4.0
  - Use in this course: the data mixture and the tokenizer design when a non-English language has a large share; compare with the Chinese case → Chapters 7/13
- **Marin (Stanford CRFM / marin-community)** (8B, 2025-05; 12.7T tokens)
  - Models: [marin-8b-base](https://huggingface.co/marin-community/marin-8b-base), [marin-8b-instruct](https://huggingface.co/marin-community/marin-8b-instruct)
  - Data: all from public data sets (DCLM-Baseline, StarCoderData, ProofPile2, Nemotron-CC, Dolma, dolmino-mix-1124, FineMath, and others); the exact mixture of each stage is in [exp600_tootsie.py](https://github.com/marin-community/marin/blob/ee163702c5bc71c9bbba3238db84b6ee86e826a7/experiments/tootsie/exp600_tootsie.py); the data licenses follow each source
  - Code / recipe: [marin](https://github.com/marin-community/marin) (Apache-2.0; each experiment, from data processing to training, is an executable script), training framework [levanter](https://github.com/stanford-crfm/levanter); the [Marin 8B retrospective](https://marin.readthedocs.io/en/latest/reports/marin-8b-retro/) records honestly why each stage changed and the bugs during the run (a wrong RoPE setting and others)
  - License: model Apache-2.0
  - Use in this course: a real record of decisions that changed the mixture and the learning rate during training, based on observations → Chapters 12/14; the Delphi scaling suite of the same project (IsoFLOP checkpoints from 157M to 25B, for example [delphi-1e21-3.4Bparams-46.3Btokens](https://huggingface.co/marin-community/delphi-1e21-3.4Bparams-46.3Btokens)) contains the models of the Delphi blog post in the general references at the start of this page → Chapter 12
- **Apertus (Swiss AI)** (8B, 2025-09; 70B is outside the scope)
  - Models: [Apertus-8B-2509](https://huggingface.co/swiss-ai/Apertus-8B-2509), [Apertus-8B-Instruct-2509](https://huggingface.co/swiss-ai/Apertus-8B-Instruct-2509) (before the download, you must click to accept the use policy)
  - Data: **Swiss AI does not release the corpus directly. It releases a pipeline that rebuilds the corpus deterministically**: [pretrain-data](https://github.com/swiss-ai/pretrain-data) (rebuilds from public data such as FineWeb-Edu, FineWeb-2, DCLM-Edu, FineMath, MegaMath, and StarCoder) + lists of domains that are removed retroactively by the robots.txt of 2025-01 (for example [robots-txt-blocked-domains-english](https://huggingface.co/datasets/swiss-ai/robots-txt-blocked-domains-english))
  - Code / recipe: [pretrain-code](https://github.com/swiss-ai/pretrain-code)
  - Technical report: [Apertus: Democratizing Open and Compliant LLMs for Global Language Environments](https://arxiv.org/abs/2509.14233)
  - License: model Apache-2.0; the data follows each source
  - Use in this course: the method for data compliance (retroactive removal of the websites that opted out through robots.txt), and how many tokens the compliance filter loses (about 8% for English) → Chapter 13
- **MAP-Neo (M-A-P)** (7.8B, plus 2B and small scaling-law models of 250M / 460M / 980M; 2024): bilingual Chinese and English
  - Models: [neo_7b](https://huggingface.co/m-a-p/neo_7b) (also checkpoints of the intermediate and decay phases), [neo_2b_general](https://huggingface.co/m-a-p/neo_2b_general)
  - Data: [Matrix](https://huggingface.co/datasets/m-a-p/Matrix) (4.69T tokens, bilingual Chinese and English)
  - Code / recipe: [MAP-NEO](https://github.com/multimodal-art-projection/MAP-NEO) (MIT, based on Megatron, includes the data-processing pipeline)
  - Technical report: [MAP-Neo: Highly Capable and Transparent Bilingual Large Language Model Series](https://arxiv.org/abs/2405.19327)
  - License: model Apache-2.0; the data set card says Apache-2.0
  - Use in this course: one of the few fully open Chinese pretraining data sets with its cleaning pipeline → Chapter 13; the small scaling-law models → Chapter 12
- **Amber / Crystal (LLM360; the HF organization is now named IFM)** (6.7B / 7B, 2023-12)
  - Models: [Amber](https://huggingface.co/IFM/Amber) (1.26T tokens, 360 checkpoints), [Crystal](https://huggingface.co/IFM/Crystal) (about 1.4T tokens, SlimPajama + StarCoderData)
  - Data: [AmberDatasets](https://huggingface.co/datasets/IFM/AmberDatasets), [CrystalCoderDatasets](https://huggingface.co/datasets/IFM/CrystalCoderDatasets) (both are tokenized data in training order, ODC-BY)
  - Code / recipe: [amber-train](https://github.com/LLM360/amber-train), [amber-data-prep](https://github.com/LLM360/amber-data-prep), [crystalcoder-train](https://github.com/LLM360/crystalcoder-train) (based on Cerebras hardware; difficult to reproduce on a different platform), analysis tool [Analysis360](https://github.com/LLM360/Analysis360)
  - Technical report: [LLM360: Towards Fully Transparent Open-Source LLMs](https://arxiv.org/abs/2312.06550)
  - License: model Apache-2.0; data ODC-BY
  - Use in this course: what a "fully transparent release" must contain (checkpoints, data order, logs, analysis code) → Chapter 20
- **Comma v0.1 (EleutherAI and others, Common Pile)** (7B, 2025): trained only on openly licensed text
  - Models: [comma-v0.1-1t](https://huggingface.co/common-pile/comma-v0.1-1t), [comma-v0.1-2t](https://huggingface.co/common-pile/comma-v0.1-2t)
  - Data: [comma_v0.1_training_dataset](https://huggingface.co/datasets/common-pile/comma_v0.1_training_dataset) (each source has an open license and keeps its original license); data-processing code [r-three/common-pile](https://github.com/r-three/common-pile)
  - Code / recipe: trained with [Meta Lingua](https://github.com/facebookresearch/lingua); the config files and the training metrics of the 2T version are in [comma-v0.1-2t-checkpoints](https://huggingface.co/common-pile/comma-v0.1-2t-checkpoints)
  - Technical report: [The Common Pile v0.1](https://arxiv.org/abs/2506.05209)
  - License: model Apache-2.0
  - Use in this course: the level that a model can reach with only openly licensed data, and how to check the data licenses source by source → Chapter 13

### Partly open (not in the main list; each entry says what is missing)
- **MiniCPM4** ([0.5B](https://huggingface.co/openbmb/MiniCPM4-0.5B) / [8B](https://huggingface.co/openbmb/MiniCPM4-8B), Apache-2.0): the pretraining framework [ForgeTrain](https://github.com/OpenBMB/ForgeTrain) and Ultra-FineWeb are public, but the other pretraining data and the SFT data (UltraChat v2) are not fully released
- **Nemotron Nano 2** ([NVIDIA-Nemotron-Nano-9B-v2](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-9B-v2), pruned and distilled from 12B-Base; [arXiv 2508.14444](https://arxiv.org/abs/2508.14444); NVIDIA Open Model License): "most of" the pretraining data is public (about 6.6T tokens, see the NVIDIA section above), but you must agree to the NVIDIA data agreement. For GitHub code, only the metadata is given. The multilingual web-crawl data and two private third-party data sets are not public. There are only the Megatron-LM / NeMo-RL frameworks, with no end-to-end reproduction recipe
- **Nemotron 3 Nano 4B** ([NVIDIA-Nemotron-3-Nano-4B-BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16), 2026-03; NVIDIA Nemotron Open Model License): compressed from Nano-9B-v2 with [Nemotron Elastic](https://arxiv.org/abs/2511.16664), so it has the same data gaps as above. The data subset for the distillation is not stated, and the Elastic routing code is not public
- **Apertus v1.1** ([0.5B](https://huggingface.co/swiss-ai/Apertus-v1.1-0.5B) / [1.5B](https://huggingface.co/swiss-ai/Apertus-v1.1-1.5B) / [4B](https://huggingface.co/swiss-ai/Apertus-v1.1-4B), 2026-02 to 04; Apache-2.0): pretraining distillation with 1.7T tokens on the stage-5 data of the 8B model ([arXiv 2605.29128](https://arxiv.org/abs/2605.29128)). But the distillation code repository in the model card (swiss-ai/Megatron-LM-Distill) was not available on 2026-10-01
- **Tülu 3 8B** ([Llama-3.1-Tulu-3-8B](https://huggingface.co/allenai/Llama-3.1-Tulu-3-8B), Llama 3.1 license): the post-training is fully open ([tulu-3-sft-mixture](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture), [open-instruct](https://github.com/allenai/open-instruct), [Tülu 3 report](https://arxiv.org/abs/2411.15124)), but the base model is Meta's Llama 3.1, and its pretraining data is not public; a reference for the SFT → DPO → RLVR recipe → Chapters 16/18/19
- **StarCoder2** ([3B](https://huggingface.co/bigcode/starcoder2-3b) / 7B / 15B, BigCode OpenRAIL-M; [arXiv 2402.19173](https://arxiv.org/abs/2402.19173)): [The Stack v2](https://huggingface.co/datasets/bigcode/the-stack-v2) requires an access request and contains only Software Heritage IDs; you must bulk-download the file contents separately. The GitHub repository has only fine-tuning and inference code, with no pretraining recipe
- **OpenCoder** ([OpenCoder-1.5B-Base](https://huggingface.co/infly/OpenCoder-1.5B-Base) / 8B; [arXiv 2411.04905](https://arxiv.org/abs/2411.04905)): for the code pretraining corpus RefineCode, only the metadata is released ([RefineCode-code-corpus-meta](https://huggingface.co/datasets/OpenCoder-LLM/RefineCode-code-corpus-meta)); the model license is a custom license
- **Zamba2** ([1.2B](https://huggingface.co/Zyphra/Zamba2-1.2B) / 2.7B / 7B, Zyphra, Apache-2.0): the stage-1 data [Zyda-2](https://huggingface.co/datasets/Zyphra/Zyda-2) (ODC-BY) is public; the annealing data of about 50B tokens and the training code are not public. ZAYA1-8B (2026) gives only the share of each data category
- **AMD-OLMo-1B** ([AMD-OLMo-1B](https://huggingface.co/amd/AMD-OLMo-1B), Apache-2.0): trained on a 1.3T-token subset of Dolma v1.7, but we did not find how AMD selected the subset
- **Salamandra** ([2B](https://huggingface.co/BSC-LT/salamandra-2b) / 7B, BSC): the model card says that the corpus "will not be released"; only the list of sources and the training scripts are public
- **EuroLLM** ([1.7B](https://huggingface.co/utter-project/EuroLLM-1.7B) / 9B, Apache-2.0): some data (EuroWeb) and the Megatron fork are public; we could not confirm whether the synthetic math data and the translated Cosmopedia are released
- **Baguettotron / Monad** ([Baguettotron](https://huggingface.co/PleIAs/Baguettotron) 321M, Monad 56M, Pleias, 2025-11, Apache-2.0): trained only on the fully synthetic data [SYNTH](https://huggingface.co/datasets/PleIAs/SYNTH) (CC-BY-4.0), but we did not find the training code; an example of fully synthetic pretraining → Chapter 13
- **OpenLLaMA / RedPajama-INCITE** ([open_llama_3b_v2](https://huggingface.co/openlm-research/open_llama_3b_v2), [RedPajama-INCITE-Base-3B-v1](https://huggingface.co/togethercomputer/RedPajama-INCITE-Base-3B-v1), 2023, Apache-2.0): the data (RedPajama and others) is public, but the exact training config and the training code are not released
- Also: the official weights of Cerebras-GPT and BTLM, and the official repository of SlimPajama (cerebras/SlimPajama-627B), are no longer available, so this list does not include them

## References added while we wrote the course (by chapter)

We read and cited these sources when we wrote each chapter. For the full citations, see the references in the README of each chapter.

### Chapter 2
- NumPy Broadcasting: https://numpy.org/doc/stable/user/basics.broadcasting.html
- NumPy "What is NumPy? / Why is NumPy fast?": https://numpy.org/doc/stable/user/whatisnumpy.html
- PyTorch torch.nn.Linear: https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html
- 3Blue1Brown, Essence of Linear Algebra: https://www.3blue1brown.com/topics/linear-algebra
- Deep Learning book, ch. 2 Linear Algebra: https://www.deeplearningbook.org/contents/linear_algebra.html
- Dive into Deep Learning §3.1 Linear Regression: https://d2l.ai/chapter_linear-regression/linear-regression.html

### Chapter 3
- Nielsen, Neural Networks and Deep Learning ch4 (a visual proof of universal approximation): http://neuralnetworksanddeeplearning.com/chap4.html
- Goodfellow et al., Deep Learning ch6: https://www.deeplearningbook.org/contents/mlp.html
- 3Blue1Brown, Neural networks: https://www.3blue1brown.com/lessons/neural-networks
- Shazeer, GLU Variants Improve Transformer (SwiGLU): https://arxiv.org/abs/2002.05202

### Chapter 4
- micrograd (MIT): https://github.com/karpathy/micrograd
- Karpathy, The spelled-out intro to neural networks and backpropagation: https://www.youtube.com/watch?v=VMj-3S1tku0
- CS231n backprop notes: https://cs231n.github.io/optimization-2/
- Baydin et al., Automatic Differentiation in Machine Learning: a Survey: https://arxiv.org/abs/1502.05767
- PyTorch Autograd mechanics: https://docs.pytorch.org/docs/stable/notes/autograd.html

### Chapter 5
- Hinton, Vinyals, Dean, Distilling the Knowledge in a Neural Network (softmax temperature): https://arxiv.org/abs/1503.02531
- Szegedy et al., Rethinking the Inception Architecture (label smoothing): https://arxiv.org/abs/1512.00567
- CS231n neural network case study (classification of spiral data): https://cs231n.github.io/neural-networks-case-study/

### Chapter 6
- MiniCPM (WSD learning-rate schedule): https://arxiv.org/abs/2404.06395
- Hägele et al., Scaling Laws and Compute-Optimal Training Beyond Fixed Training Durations (constant learning rate + cooldown): https://arxiv.org/abs/2405.18392
- OLMo 2: https://arxiv.org/abs/2501.00656
- SmolLM3 blog: https://huggingface.co/blog/smollm3
- Xiong et al., On Layer Normalization in the Transformer Architecture (Pre-LN): https://arxiv.org/abs/2002.04745
- Loshchilov & Hutter, Decoupled Weight Decay Regularization (AdamW): https://arxiv.org/abs/1711.05101

### Chapter 7
- Sennrich et al., Neural Machine Translation of Rare Words with Subword Units (BPE): https://arxiv.org/abs/1508.07909
- Llama 3 Herd of Models: https://arxiv.org/abs/2407.21783
- Karpathy minbpe: https://github.com/karpathy/minbpe
- CS336 Assignment 1 Basics: https://github.com/stanford-cs336/assignment1-basics

### Chapter 8
- Vaswani et al., Attention Is All You Need: https://arxiv.org/abs/1706.03762
- Bahdanau et al., Neural Machine Translation by Jointly Learning to Align and Translate: https://arxiv.org/abs/1409.0473
- Ainslie et al., GQA: https://arxiv.org/abs/2305.13245
- Dao et al., FlashAttention: https://arxiv.org/abs/2205.14135
- Gemma 3 Technical Report: https://arxiv.org/abs/2503.19786
- gpt-oss-120b & gpt-oss-20b Model Card: https://arxiv.org/abs/2508.10925
- Karpathy, ng-video-lecture (Let's build GPT): https://github.com/karpathy/ng-video-lecture

### Chapter 9
- Su et al., RoFormer (RoPE): https://arxiv.org/abs/2104.09864
- Dehghani et al., Scaling Vision Transformers to 22B (QK-Norm): https://arxiv.org/abs/2302.05442
- Wortsman et al., Small-scale proxies for large-scale Transformer training instabilities: https://arxiv.org/abs/2309.14322
- Press & Wolf, Using the Output Embedding to Improve Language Models: https://arxiv.org/abs/1608.05859
- Qwen3 Technical Report: https://arxiv.org/abs/2505.09388
- DeepSeek-V3 Technical Report: https://arxiv.org/abs/2412.19437

### Chapter 10
- Holtzman et al., The Curious Case of Neural Text Degeneration (top-p): https://arxiv.org/abs/1904.09751
- Shazeer, Fast Transformer Decoding: One Write-Head is All You Need (MQA): https://arxiv.org/abs/1911.02150
- Kwon et al., Efficient Memory Management for LLM Serving with PagedAttention (vLLM): https://arxiv.org/abs/2309.06180
- Orca (continuous batching, OSDI'22): https://www.usenix.org/conference/osdi22/presentation/yu

### Chapter 11
- MMLU-Redux: https://arxiv.org/abs/2406.04127 ; MMLU-Pro: https://arxiv.org/abs/2406.01574
- C-Eval: https://arxiv.org/abs/2305.08322 ; CMMLU: https://arxiv.org/abs/2306.09212
- EvalPlus (HumanEval+/MBPP+): https://arxiv.org/abs/2305.01210 ; IFEval: https://arxiv.org/abs/2311.07911
- BFCL: https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard
- ACEBench: https://arxiv.org/abs/2501.12851 (https://github.com/chenchen0103/ACEBench)
- τ²-bench: https://arxiv.org/abs/2506.07982
- Miller, Adding Error Bars to Evals: https://arxiv.org/abs/2411.00640
- Sclar et al., Quantifying Language Models' Sensitivity to Spurious Features in Prompt Design: https://arxiv.org/abs/2310.11324

### Chapter 12
- Besiroglu et al., Chinchilla Scaling: A replication attempt: https://arxiv.org/abs/2404.10102
- Porian et al., Resolving Discrepancies in Compute-Optimal Scaling: https://arxiv.org/abs/2406.19146
- Sardana & Frankle, Beyond Chinchilla-Optimal (inference cost and overtraining): https://arxiv.org/abs/2401.00448
- Moonlight (Muon at scale): https://arxiv.org/abs/2502.16982
- GLM-4.5: https://arxiv.org/abs/2508.06471
- Keller Jordan, Muon: https://kellerjordan.github.io/posts/muon/

### Chapter 13
- FineWeb: https://arxiv.org/abs/2406.17557 ; FineWeb2: https://arxiv.org/abs/2506.20920
- DCLM: https://arxiv.org/abs/2406.11794
- Nemotron-CC: https://arxiv.org/abs/2412.02595
- Ultra-FineWeb: https://arxiv.org/abs/2505.05427
- Tao et al., Scaling Laws with Vocabulary: https://arxiv.org/abs/2407.13623
- Lee et al., Deduplicating Training Data Makes Language Models Better: https://arxiv.org/abs/2107.06499
- CS336 Assignment 4 Data: https://github.com/stanford-cs336/assignment4-data

### Chapter 14
- Micikevicius et al., Mixed Precision Training: https://arxiv.org/abs/1710.03740
- Milakov & Gimelshein, Online normalizer calculation for softmax: https://arxiv.org/abs/1805.02867
- FlashAttention-2: https://arxiv.org/abs/2307.08691 ; FlashAttention-3: https://arxiv.org/abs/2407.08608
- Chen et al., Training Deep Nets with Sublinear Memory Cost (activation checkpointing): https://arxiv.org/abs/1604.06174
- ZeRO: https://arxiv.org/abs/1910.02054 ; PyTorch FSDP: https://arxiv.org/abs/2304.11277
- PaLM (MFU, loss spike, z-loss): https://arxiv.org/abs/2204.02311
- OLMo-core: https://github.com/allenai/OLMo-core

### Chapter 15
- YaRN: https://arxiv.org/abs/2309.00071 ; Position Interpolation: https://arxiv.org/abs/2306.15595
- Xiong et al., Effective Long-Context Scaling (ABF): https://arxiv.org/abs/2309.16039
- RULER: https://arxiv.org/abs/2404.06654 ; Needle in a Haystack: https://github.com/gkamradt/LLMTest_NeedleInAHaystack
- Blakeney et al., Does your data spark joy? (data for the annealing phase): https://arxiv.org/abs/2406.03476

### Chapter 16
- Touvron et al. *Llama 2: Open Foundation and Fine-Tuned Chat Models*, 2023: https://arxiv.org/abs/2307.09288
- Lambert et al. *Tülu 3: Pushing Frontiers in Open Language Model Post-Training*, 2024: https://arxiv.org/abs/2411.15124; training code open-instruct: https://github.com/allenai/open-instruct
- Allal et al. *SmolLM2: When Smol Goes Big — Data-Centric Training of a Small Language Model*, 2025: https://arxiv.org/abs/2502.02737
- Zhou et al. *LIMA: Less Is More for Alignment*, 2023: https://arxiv.org/abs/2305.11206
- Ouyang et al. *Training language models to follow instructions with human feedback* (InstructGPT), 2022: https://arxiv.org/abs/2203.02155
- Teknium et al. *Hermes 3 Technical Report*, 2024: https://arxiv.org/abs/2408.11857
- Liu et al. *APIGen: Automated Pipeline for Generating Verifiable and Diverse Function-Calling Datasets*, 2024: https://arxiv.org/abs/2406.18518
- Liu et al. *ToolACE: Winning the Points of LLM Function Calling*, 2024: https://arxiv.org/abs/2409.00920
- Hu et al. *LoRA: Low-Rank Adaptation of Large Language Models*, 2021: https://arxiv.org/abs/2106.09685
- OpenAI. ChatML description (openai-python v0.28): https://github.com/openai/openai-python/blob/release-v0.28.0/chatml.md
- Hugging Face TRL documentation: SFT Trainer: https://huggingface.co/docs/trl/sft_trainer, Reducing Memory Usage (packing): https://huggingface.co/docs/trl/reducing_memory_usage
- Data set cards (read in 2026-09): tulu-3-sft-mixture: https://huggingface.co/datasets/allenai/tulu-3-sft-mixture, smoltalk: https://huggingface.co/datasets/HuggingFaceTB/smoltalk, smoltalk2: https://huggingface.co/datasets/HuggingFaceTB/smoltalk2, xlam-function-calling-60k: https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k, ToolACE: https://huggingface.co/datasets/Team-ACE/ToolACE, hermes-function-calling-v1: https://huggingface.co/datasets/NousResearch/hermes-function-calling-v1
- Chat templates (read in 2026-09): Qwen3-0.6B: https://huggingface.co/Qwen/Qwen3-0.6B/blob/main/tokenizer_config.json, Qwen3.5-0.8B: https://huggingface.co/Qwen/Qwen3.5-0.8B/blob/main/chat_template.jinja, SmolLM3-3B: https://huggingface.co/HuggingFaceTB/SmolLM3-3B/blob/main/chat_template.jinja, Hermes-3-Llama-3.1-8B: https://huggingface.co/NousResearch/Hermes-3-Llama-3.1-8B, Olmo-3-7B-Instruct: https://huggingface.co/allenai/Olmo-3-7B-Instruct/blob/main/chat_template.jinja
- nanochat: `scripts/chat_sft.py` in https://github.com/karpathy/nanochat ; minimind: the SFT part of https://github.com/jingyaogong/minimind (a complete SFT pipeline for a small Chinese model)
- CS336 Assignment 5: https://github.com/stanford-cs336/assignment5-alignment

### Chapter 17
- Kim & Rush, Sequence-Level Knowledge Distillation: https://arxiv.org/abs/1606.07947
- Agarwal et al., On-Policy Distillation of Language Models (GKD): https://arxiv.org/abs/2306.13649
- Thinking Machines, On-Policy Distillation: https://thinkingmachines.ai/blog/on-policy-distillation
- Gemma 2: https://arxiv.org/abs/2408.00118
- Minitron (pruning + distillation): https://arxiv.org/abs/2407.14679
- GLM-5 Team. *GLM-5* §3.5 On-Policy Cross-Stage Distillation, 2026: https://arxiv.org/abs/2602.15763. On-policy distillation is the **last** post-training step (SFT → reasoning RL → agentic RL → general RL → OPD). The teachers are the model's own checkpoints from the earlier stages, so they use the same vocabulary. It recovers the abilities that the model forgot in the RL stages. Group size 1, batch size 1024. The main-line model uses its own tokenizer, so it cannot use OPD with an external teacher. This self-distillation form is the OPD that the main line can use.

### Chapter 18
- Rafailov et al. *Direct Preference Optimization: Your Language Model is Secretly a Reward Model*, 2023: https://arxiv.org/abs/2305.18290
- Stiennon et al. *Learning to summarize from human feedback*, 2020: https://arxiv.org/abs/2009.01325
- Schulman et al. *Proximal Policy Optimization Algorithms*, 2017: https://arxiv.org/abs/1707.06347
- Gao, Schulman, Hilton. *Scaling Laws for Reward Model Overoptimization*, 2022: https://arxiv.org/abs/2210.10760
- Qwen Team. *Qwen2 Technical Report* (§4.3), 2024: https://arxiv.org/abs/2407.10671; *Qwen2.5 Technical Report* (§4.2–4.3), 2024: https://arxiv.org/abs/2412.15115
- DeepSeek-AI. *DeepSeek LLM: Scaling Open-Source Language Models with Longtermism* (§4), 2024: https://arxiv.org/abs/2401.02954
- NVIDIA. *Nemotron-4 340B Technical Report* (§3.3.2 DPO and RPO), 2024: https://arxiv.org/abs/2406.11704
- Tunstall et al. *Zephyr: Direct Distillation of LM Alignment*, 2023: https://arxiv.org/abs/2310.16944
- Hugging Face. *SmolLM3: smol, multilingual, long-context reasoner* (blog): https://github.com/huggingface/blog/blob/main/smollm3.md
- D'Oosterlinck et al. *Anchored Preference Optimization and Contrastive Revisions* (APO), 2024: https://arxiv.org/abs/2408.06266
- Razin et al. *Unintentional Unalignment: Likelihood Displacement in Direct Preference Optimization*, 2024: https://arxiv.org/abs/2410.08847
- Pal et al. *Smaug: Fixing Failure Modes of Preference Optimisation with DPO-Positive*, 2024: https://arxiv.org/abs/2402.13228
- Park et al. *Disentangling Length from Quality in Direct Preference Optimization*, 2024: https://arxiv.org/abs/2403.19159
- Singhal et al. *A Long Way to Go: Investigating Length Correlations in RLHF*, 2023: https://arxiv.org/abs/2310.03716
- Variants (observations of new work): IPO https://arxiv.org/abs/2310.12036; KTO https://arxiv.org/abs/2402.01306; SimPO https://arxiv.org/abs/2405.14734; ORPO https://arxiv.org/abs/2403.07691
- Preference data sets: UltraFeedback: https://huggingface.co/datasets/openbmb/UltraFeedback (MIT), HelpSteer3: https://huggingface.co/datasets/nvidia/HelpSteer3 (CC-BY-4.0), Tülu 3 8B preference mixture: https://huggingface.co/datasets/allenai/llama-3.1-tulu-3-8b-preference-mixture (ODC-BY-1.0; some subsets do not allow commercial use)

### Chapter 19
- Williams. *Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning* (REINFORCE), 1992: https://link.springer.com/article/10.1007/BF00992696
- Shao et al. *DeepSeekMath* (GRPO, k3 KL, the unified view of gradient coefficients), 2024: https://arxiv.org/abs/2402.03300
- DeepSeek-AI. *DeepSeek-R1*, 2025: https://arxiv.org/abs/2501.12948
- Qwen Team. *Qwen3 Technical Report*, 2025: https://arxiv.org/abs/2505.09388; *Group Sequence Policy Optimization*: https://arxiv.org/abs/2507.18071
- Olmo Team. *Olmo 3*, 2025: https://arxiv.org/abs/2512.13961
- Yu et al. *DAPO*, 2025: https://arxiv.org/abs/2503.14476
- Liu et al. *Understanding R1-Zero-Like Training: A Critical Perspective* (Dr. GRPO), 2025: https://arxiv.org/abs/2503.20783
- Schulman. *Approximating KL Divergence*: http://joschu.net/blog/kl-approx.html
- Lilian Weng. *Reward Hacking in Reinforcement Learning*, 2024: https://lilianweng.github.io/posts/2024-11-28-reward-hacking/
- verl (HybridFlow): https://github.com/verl-project/verl
- GLM-5 Team. *GLM-5* §3.2 (reasoning RL: GRPO + IcePop, no KL term, group size 32, ε_low 0.2 / ε_high 0.28) and §4.1 (asynchronous agentic RL: token-in-token-out, direct double-sided importance sampling), 2026: https://arxiv.org/abs/2602.15763
- GLM-5.2 (2026-06): [model card](https://huggingface.co/zai-org/GLM-5.2), [official blog](https://z.ai/blog/glm-5.2). As reported from the official blog, the **long-horizon** RL stage replaces GRPO with **critic-based PPO**. After compaction, the sub-trajectories of one task have different numbers and lengths, so a group of comparable samples cannot be formed; a value model gives token-level advantages. It also adds a two-stage guard against reward hacking (rules, then an LLM judge; a blocked tool call gets a meaningless fake response, and the trajectory continues). Only one family uses it now → "frontier observation" box in Chapter 19 (rule A needs 3 families). The main line keeps GRPO: tool-call trajectories are short and the rewards are verifiable.
- 机器之心. *GRPO过时了吗？* (Is GRPO outdated?, in Chinese), 2026-06-21: https://www.163.com/dy/article/KVVC29AK0511AQHO.html (36Kr repost: https://www.36kr.com/p/3862288768570377). A second-hand summary of the GLM-5.2 change and the community discussion. It notes that DeepSeek-V4 still uses GRPO for its domain experts.
- *对 GLM-5.2 PPO 优化的思考：Strong Value Model 如何引导真实场景 RL 训练* (in Chinese, Zhihu): https://zhuanlan.zhihu.com/p/2052145684040701422
- *Learning Without Critics? Revisiting GRPO in Classical Reinforcement Learning Environments*, 2025: https://arxiv.org/abs/2511.03527. In long-horizon tasks without early termination, critic-free methods are worse than PPO with a learned value function; only in short tasks such as CartPole are they equal.

### Chapter 20
- GGUF specification: https://github.com/ggml-org/ggml/blob/master/docs/gguf.md
- llama.cpp k-quants (PR #1684): https://github.com/ggml-org/llama.cpp/pull/1684
- Mitchell et al., Model Cards for Model Reporting: https://arxiv.org/abs/1810.03993
- Dettmers et al., LLM.int8(): https://arxiv.org/abs/2208.07339 ; Frantar et al., GPTQ: https://arxiv.org/abs/2210.17323

### Chapter 21
- DeepSeek-V2 (MLA): https://arxiv.org/abs/2405.04434
- Kimi K2 Technical Report: https://arxiv.org/abs/2507.20534
- GLM-5 Technical Report: https://arxiv.org/abs/2602.15763
- FlashMLA: https://github.com/deepseek-ai/FlashMLA

### Chapter 22
- Child, Gray, Radford, Sutskever. *Generating Long Sequences with Sparse Transformers*, 2019: https://arxiv.org/abs/1904.10509
- Beltagy, Peters, Cohan. *Longformer: The Long-Document Transformer* (sliding window + global tokens), 2020: https://arxiv.org/abs/2004.05150
- Jiang et al. *Mistral 7B* (sliding window, rolling buffer cache, theoretical span), 2023: https://arxiv.org/abs/2310.06825
- Xiao et al. *Efficient Streaming Language Models with Attention Sinks* (StreamingLLM), 2023: https://arxiv.org/abs/2309.17453
- Yuan et al. *Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention* (NSA), 2025: https://arxiv.org/abs/2502.11089
- Lu et al. *MoBA: Mixture of Block Attention for Long-Context LLMs*, 2025: https://arxiv.org/abs/2502.13189
- DeepSeek-AI. *DeepSeek-V3.2* (DSA) technical report: https://huggingface.co/deepseek-ai/DeepSeek-V3.2/blob/main/assets/paper.pdf
- DeepSeek-AI. *DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence*, 2026: https://arxiv.org/abs/2606.19348
- Lai et al. *MiniMax Sparse Attention*, 2026: https://arxiv.org/abs/2606.13392
- MiniCPM Team. *MiniCPM4*, 2025: https://arxiv.org/abs/2506.07900; *InfLLM-V2*, 2025: https://arxiv.org/abs/2509.24663
- Dao. FlashAttention (the `window_size` parameter): https://github.com/Dao-AILab/flash-attention; PyTorch FlexAttention blog (a sliding-window `mask_mod` example): https://pytorch.org/blog/flexattention/

### Chapter 23
- Katharopoulos et al., Transformers are RNNs (linear attention): https://arxiv.org/abs/2006.16236
- Yang et al., Parallelizing Linear Transformers with the Delta Rule: https://arxiv.org/abs/2406.06484
- Gated Delta Networks: https://arxiv.org/abs/2412.06464
- Kimi Linear: https://arxiv.org/abs/2510.26692
- MiniMax-01: https://arxiv.org/abs/2501.08313
- Nemotron-H: https://arxiv.org/abs/2504.03624
- flash-linear-attention: https://github.com/fla-org/flash-linear-attention

### Chapter 24
- Shazeer et al. *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer* (the MoE layer and the routing-collapse problem), 2017: https://arxiv.org/abs/1701.06538
- Lepikhin et al. *GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding* (top-2 routing, capacity, auxiliary loss, expert parallelism), 2020: https://arxiv.org/abs/2006.16668
- Fedus, Zoph, Shazeer. *Switch Transformers* (top-1 routing, load-balancing loss, capacity factor), 2021: https://arxiv.org/abs/2101.03961
- Gale et al. *MegaBlocks: Efficient Sparse Training with Mixture-of-Experts* (dropless, block-sparse matrix multiplication), 2022: https://arxiv.org/abs/2211.15841
- Jiang et al. *Mixtral of Experts*, 2024: https://arxiv.org/abs/2401.04088
- Dai et al. *DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models* (fine-grained experts, shared experts), 2024: https://arxiv.org/abs/2401.06066
- Wang et al. *Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts*, 2024: https://arxiv.org/abs/2408.15664
- NVIDIA. *Nemotron 3 Nano: Open, Efficient Mixture-of-Experts Hybrid Mamba-Transformer Model for Agentic Reasoning* (Sections 2.1 and 2.4), 2025: https://arxiv.org/abs/2512.20848
- Muennighoff et al. *OLMoE: Open Mixture-of-Experts Language Models*, 2024: https://arxiv.org/abs/2409.02060
- The MoE implementations in Hugging Face transformers (`models/qwen3_moe`, `models/deepseek_v3`, `models/llama4`): https://github.com/huggingface/transformers/tree/main/src/transformers/models
- DeepSeek. DeepEP (a communication library for expert parallelism): https://github.com/deepseek-ai/DeepEP

### Chapter 25
- Leviathan et al., Fast Inference from Transformers via Speculative Decoding: https://arxiv.org/abs/2211.17192
- Chen et al., Accelerating LLM Decoding with Speculative Sampling: https://arxiv.org/abs/2302.01318
- Gloeckle et al., Better & Faster LLMs via Multi-token Prediction: https://arxiv.org/abs/2404.19737
- MiMo: https://arxiv.org/abs/2505.07608
- CS336 lecture code (Spring 2026): https://github.com/stanford-cs336/lectures

### Chapter 26
- Kimi Team. *Kimi K3 Technical Report*: https://github.com/MoonshotAI/Kimi-K3/blob/main/k3_tech_report.pdf
- GLM-5 Team. *GLM-5: from Vibe Coding to Agentic Engineering*, 2026: https://arxiv.org/abs/2602.15763; IndexShare: https://arxiv.org/abs/2603.12201
- Qwen Team. *Qwen3.8-Max* blog: https://qwen.ai/blog?id=qwen3.8
- Sources of the nodes in the evolution tree: GPT-2 (Radford et al. 2019); Llama https://arxiv.org/abs/2302.13971; RMSNorm https://arxiv.org/abs/1910.07467; RoPE https://arxiv.org/abs/2104.09864; YaRN https://arxiv.org/abs/2309.00071; GLU variants https://arxiv.org/abs/2002.05202; GQA https://arxiv.org/abs/2305.13245; QK-Norm https://arxiv.org/abs/2010.04245; tied embeddings https://arxiv.org/abs/1608.05859; DeepSeekMoE https://arxiv.org/abs/2401.06066; auxiliary-loss-free balancing https://arxiv.org/abs/2408.15664; MLA (DeepSeek-V2) https://arxiv.org/abs/2405.04434; Mistral 7B https://arxiv.org/abs/2310.06825; Gemma 2 https://arxiv.org/abs/2408.00118; Gated DeltaNet https://arxiv.org/abs/2412.06464; MTP (DeepSeek-V3) https://arxiv.org/abs/2412.19437
