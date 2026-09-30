这个教程不是从头构建，参考了很多前人的工作，以下是一些重要的参考资料：
- [CS336](https://cs336.stanford.edu/)
- [minimind](https://github.com/jingyaogong/minimind)
- [nanoGPT](https://github.com/karpathy/nanoGPT)
- [Hands-On Modern Reinforcement Learning](https://walkinglabs.github.io/hands-on-modern-rl/preface/intro)
- [Mathematical theory of deep learning](https://arxiv.org/abs/2407.18384)
- [Scaling Laws That Extrapolate 300× Past the Fit](https://openathena.ai/blog/delphi/)
-[12G 显存，够吗？——消费级 GPU 上做 LLM 算法研究的完整路线](https://x.com/dashen_wang/status/2038970877732380924)
- [Reproducing all of Schmidhuber’s papers (1990-2025)](https://github.com/cybertronai/schmidhuber-problems/blob/main/VISUAL_TOUR.md)
- [AI 研究方法的演变](https://github.com/owlman/CS_StudyNotes/blob/master/02_%E5%9F%BA%E7%A1%80%E7%90%86%E8%AE%BA%E5%AD%A6%E4%B9%A0/%E4%BA%BA%E5%B7%A5%E6%99%BA%E8%83%BD/01.%E5%AD%A6%E4%B9%A0%E7%AC%94%E8%AE%B0/AI%20%E7%A0%94%E7%A9%B6%E6%96%B9%E6%B3%95%E7%9A%84%E6%BC%94%E5%8F%98.md): 从1950年开始的一个AI研究方法的演变，涵盖了从符号主义到连接主义，再到现代深度学习的历程。
- [Advanced Natural Language Processing / Spring 2026 from CMU](https://cmu-l3.github.io/anlp-spring2026/)[code](https://github.com/cmu-l3/anlp-spring2026-code)
- [Maths, CS & AI Compendium](https://github.com/HenryNdubuaku/maths-cs-ai-compendium/tree/main)
- [Pi Book](https://github.com/antinomie-lab/pi-book/tree/main)： 这本书的格式和交互方式很适合学习，关于Pi agent 的内容也可以参考
- [Pretraining a Mini Kimi K3](https://books.vizuara.ai/book/pretraining-a-mini-k3) 对于k3这个新架构的预训练可以参考
- 1.5万字速通LLM主流模型结构（Llama、Qwen、GLM、Deepseek...](https://zhuanlan.zhihu.com/p/2060741715095560795): 这篇文章对主流的LLM模型结构进行了详细的介绍和比较，涵盖了Llama、Qwen、GLM、Deepseek等模型的架构特点和应用场景。
- [Small-Scale Experiments: Are We There Yet?](https://arxiv.org/pdf/2608.11859):这篇文章介绍了小规模LLM的scaling law,小模型对超参数极度敏感掩盖了缩放定律的真实存在,只有充分调优后小规模实验才能可靠预测大规模结果。
- datasets: fine-web edu,ultra-fineweb-L1
- [Marin 535B-A23B 训练直播](https://wandb.ai/marin-community/marin_moe/reports/535B-A23B-18T-Token-Hero-Run-Scaling-Ladder--VmlldzoxNzc2MDM5Ng),[github details](https://github.com/marin-community/marin/issues/8435)，[data composition:](https://storage.googleapis.com/marin-public/held/harrier-k40-cluster-overview/2026.08.18/index.html?revision=uniform-sampling)
- [UNDERSTANDING TRANSFORMERS AND ATTENTIONMECHANISMS: AN INTRODUCTION FOR APPLIEDMATHEMATICIANS](https://arxiv.org/pdf/2604.00965)
- [Puro-2B](https://www.alphaxiv.org/abs/2608.27370):Puro-2B：穷实验室在RTX 5090上以5090美元预算训练的Qwen2-1.5B
- [Awesome Claude Opus 5.5 Videos](https://github.com/athemeroy/awesome-opus-5-5-videos): 用 Claude Opus 5.5 做视频的案例合集，重点看"教育讲解片"路径（Manim/Remotion + TTS）和 [七类可复用提示词模板](https://github.com/athemeroy/awesome-opus-5-5-videos/blob/main/docs/prompt-playbook.zh-CN.md)，每章配套视频的制作流程参考这里。
- [XiaomiMiMo/verl](https://github.com/XiaomiMiMo/verl)：小米 MiMo 开源的智能体强化学习训练代码（基于 verl 0.9.0.dev，Apache-2.0），复现 MiMo-V2.6 技术报告第 7 节的 RL 配方，包含代码（可执行测试判分）、网络安全、通用知识工作（rubric 判分）、网页开发（视觉判分）、音乐五类 RL 环境，并公开了训练数据 [MiMo-V2.6-RL-oss](https://huggingface.co/datasets/XiaomiMiMo/MiMo-V2.6-RL-oss)。可作为第 19 章 RL 环境与奖励设计的参考，也是第二步实际 RL 训练可评估使用的框架。

## 课程写作中新增的参考资料（按章节）

以下资料在编写各章时查阅并引用，完整出处见各章 README 的参考文献。

### 第 2 章
- NumPy Broadcasting: https://numpy.org/doc/stable/user/basics.broadcasting.html
- NumPy "What is NumPy? / Why is NumPy fast?": https://numpy.org/doc/stable/user/whatisnumpy.html
- PyTorch torch.nn.Linear: https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html
- 3Blue1Brown, Essence of Linear Algebra: https://www.3blue1brown.com/topics/linear-algebra
- Deep Learning book, ch. 2 Linear Algebra: https://www.deeplearningbook.org/contents/linear_algebra.html
- Dive into Deep Learning §3.1 Linear Regression: https://d2l.ai/chapter_linear-regression/linear-regression.html

### 第 3 章
- Nielsen, Neural Networks and Deep Learning ch4（万能逼近的可视化证明）: http://neuralnetworksanddeeplearning.com/chap4.html
- Goodfellow et al., Deep Learning ch6: https://www.deeplearningbook.org/contents/mlp.html
- 3Blue1Brown, Neural networks: https://www.3blue1brown.com/lessons/neural-networks
- Shazeer, GLU Variants Improve Transformer (SwiGLU): https://arxiv.org/abs/2002.05202

### 第 4 章
- micrograd（MIT）: https://github.com/karpathy/micrograd
- Karpathy, The spelled-out intro to neural networks and backpropagation: https://www.youtube.com/watch?v=VMj-3S1tku0
- CS231n backprop notes: https://cs231n.github.io/optimization-2/
- Baydin et al., Automatic Differentiation in Machine Learning: a Survey: https://arxiv.org/abs/1502.05767
- PyTorch Autograd mechanics: https://docs.pytorch.org/docs/stable/notes/autograd.html

### 第 5 章
- Hinton, Vinyals, Dean, Distilling the Knowledge in a Neural Network（softmax 温度）: https://arxiv.org/abs/1503.02531
- Szegedy et al., Rethinking the Inception Architecture（label smoothing）: https://arxiv.org/abs/1512.00567
- CS231n neural network case study（螺旋数据分类）: https://cs231n.github.io/neural-networks-case-study/

### 第 6 章
- MiniCPM（WSD 学习率调度）: https://arxiv.org/abs/2404.06395
- Hägele et al., Scaling Laws and Compute-Optimal Training Beyond Fixed Training Durations（恒定学习率 + 冷却）: https://arxiv.org/abs/2405.18392
- OLMo 2: https://arxiv.org/abs/2501.00656
- SmolLM3 博客: https://huggingface.co/blog/smollm3
- Xiong et al., On Layer Normalization in the Transformer Architecture（Pre-LN）: https://arxiv.org/abs/2002.04745
- Loshchilov & Hutter, Decoupled Weight Decay Regularization（AdamW）: https://arxiv.org/abs/1711.05101

### 第 7 章
- Sennrich et al., Neural Machine Translation of Rare Words with Subword Units（BPE）: https://arxiv.org/abs/1508.07909
- Llama 3 Herd of Models: https://arxiv.org/abs/2407.21783
- Karpathy minbpe: https://github.com/karpathy/minbpe
- CS336 Assignment 1 Basics: https://github.com/stanford-cs336/assignment1-basics

### 第 8 章
- Vaswani et al., Attention Is All You Need: https://arxiv.org/abs/1706.03762
- Bahdanau et al., Neural Machine Translation by Jointly Learning to Align and Translate: https://arxiv.org/abs/1409.0473
- Ainslie et al., GQA: https://arxiv.org/abs/2305.13245
- Dao et al., FlashAttention: https://arxiv.org/abs/2205.14135
- Gemma 3 Technical Report: https://arxiv.org/abs/2503.19786
- gpt-oss-120b & gpt-oss-20b Model Card: https://arxiv.org/abs/2508.10925
- Karpathy, ng-video-lecture（Let's build GPT）: https://github.com/karpathy/ng-video-lecture

### 第 9 章
- Su et al., RoFormer（RoPE）: https://arxiv.org/abs/2104.09864
- Dehghani et al., Scaling Vision Transformers to 22B（QK-Norm）: https://arxiv.org/abs/2302.05442
- Wortsman et al., Small-scale proxies for large-scale Transformer training instabilities: https://arxiv.org/abs/2309.14322
- Press & Wolf, Using the Output Embedding to Improve Language Models: https://arxiv.org/abs/1608.05859
- Qwen3 Technical Report: https://arxiv.org/abs/2505.09388
- DeepSeek-V3 Technical Report: https://arxiv.org/abs/2412.19437

### 第 10 章
- Holtzman et al., The Curious Case of Neural Text Degeneration（top-p）: https://arxiv.org/abs/1904.09751
- Shazeer, Fast Transformer Decoding: One Write-Head is All You Need（MQA）: https://arxiv.org/abs/1911.02150
- Kwon et al., Efficient Memory Management for LLM Serving with PagedAttention（vLLM）: https://arxiv.org/abs/2309.06180
- Orca（continuous batching, OSDI'22）: https://www.usenix.org/conference/osdi22/presentation/yu

### 第 11 章
- MMLU-Redux: https://arxiv.org/abs/2406.04127 ; MMLU-Pro: https://arxiv.org/abs/2406.01574
- C-Eval: https://arxiv.org/abs/2305.08322 ; CMMLU: https://arxiv.org/abs/2306.09212
- EvalPlus（HumanEval+/MBPP+）: https://arxiv.org/abs/2305.01210 ; IFEval: https://arxiv.org/abs/2311.07911
- BFCL: https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard
- ACEBench: https://arxiv.org/abs/2501.12851 （https://github.com/chenchen0103/ACEBench）
- τ²-bench: https://arxiv.org/abs/2506.07982
- Miller, Adding Error Bars to Evals: https://arxiv.org/abs/2411.00640
- Sclar et al., Quantifying Language Models' Sensitivity to Spurious Features in Prompt Design: https://arxiv.org/abs/2310.11324

### 第 12 章
- Besiroglu et al., Chinchilla Scaling: A replication attempt: https://arxiv.org/abs/2404.10102
- Porian et al., Resolving Discrepancies in Compute-Optimal Scaling: https://arxiv.org/abs/2406.19146
- Sardana & Frankle, Beyond Chinchilla-Optimal（推理成本与过训练）: https://arxiv.org/abs/2401.00448
- Moonlight（Muon 规模化）: https://arxiv.org/abs/2502.16982
- GLM-4.5: https://arxiv.org/abs/2508.06471
- Keller Jordan, Muon: https://kellerjordan.github.io/posts/muon/

### 第 13 章
- FineWeb: https://arxiv.org/abs/2406.17557 ; FineWeb2: https://arxiv.org/abs/2506.20920
- DCLM: https://arxiv.org/abs/2406.11794
- Nemotron-CC: https://arxiv.org/abs/2412.02595
- Ultra-FineWeb: https://arxiv.org/abs/2505.05427
- Tao et al., Scaling Laws with Vocabulary: https://arxiv.org/abs/2407.13623
- Lee et al., Deduplicating Training Data Makes Language Models Better: https://arxiv.org/abs/2107.06499
- CS336 Assignment 4 Data: https://github.com/stanford-cs336/assignment4-data

### 第 14 章
- Micikevicius et al., Mixed Precision Training: https://arxiv.org/abs/1710.03740
- Milakov & Gimelshein, Online normalizer calculation for softmax: https://arxiv.org/abs/1805.02867
- FlashAttention-2: https://arxiv.org/abs/2307.08691 ; FlashAttention-3: https://arxiv.org/abs/2407.08608
- Chen et al., Training Deep Nets with Sublinear Memory Cost（激活检查点）: https://arxiv.org/abs/1604.06174
- ZeRO: https://arxiv.org/abs/1910.02054 ; PyTorch FSDP: https://arxiv.org/abs/2304.11277
- PaLM（MFU、loss spike、z-loss）: https://arxiv.org/abs/2204.02311
- OLMo-core: https://github.com/allenai/OLMo-core

### 第 15 章
- YaRN: https://arxiv.org/abs/2309.00071 ; Position Interpolation: https://arxiv.org/abs/2306.15595
- Xiong et al., Effective Long-Context Scaling（ABF）: https://arxiv.org/abs/2309.16039
- RULER: https://arxiv.org/abs/2404.06654 ; Needle in a Haystack: https://github.com/gkamradt/LLMTest_NeedleInAHaystack
- Blakeney et al., Does your data spark joy?（退火阶段的数据）: https://arxiv.org/abs/2406.03476

### 第 16 章
- Touvron et al. *Llama 2: Open Foundation and Fine-Tuned Chat Models*，2023：https://arxiv.org/abs/2307.09288
- Lambert et al. *Tülu 3: Pushing Frontiers in Open Language Model Post-Training*，2024：https://arxiv.org/abs/2411.15124；训练代码 open-instruct：https://github.com/allenai/open-instruct
- Allal et al. *SmolLM2: When Smol Goes Big — Data-Centric Training of a Small Language Model*，2025：https://arxiv.org/abs/2502.02737
- Zhou et al. *LIMA: Less Is More for Alignment*，2023：https://arxiv.org/abs/2305.11206
- Ouyang et al. *Training language models to follow instructions with human feedback*（InstructGPT），2022：https://arxiv.org/abs/2203.02155
- Teknium et al. *Hermes 3 Technical Report*，2024：https://arxiv.org/abs/2408.11857
- Liu et al. *APIGen: Automated Pipeline for Generating Verifiable and Diverse Function-Calling Datasets*，2024：https://arxiv.org/abs/2406.18518
- Liu et al. *ToolACE: Winning the Points of LLM Function Calling*，2024：https://arxiv.org/abs/2409.00920
- Hu et al. *LoRA: Low-Rank Adaptation of Large Language Models*，2021：https://arxiv.org/abs/2106.09685
- OpenAI. ChatML 说明（openai-python v0.28）：https://github.com/openai/openai-python/blob/release-v0.28.0/chatml.md
- Hugging Face TRL 文档：SFT Trainer: https://huggingface.co/docs/trl/sft_trainer、Reducing Memory Usage（打包）: https://huggingface.co/docs/trl/reducing_memory_usage
- 数据卡（2026-09 读取）：tulu-3-sft-mixture: https://huggingface.co/datasets/allenai/tulu-3-sft-mixture、smoltalk: https://huggingface.co/datasets/HuggingFaceTB/smoltalk、smoltalk2: https://huggingface.co/datasets/HuggingFaceTB/smoltalk2、xlam-function-calling-60k: https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k、ToolACE: https://huggingface.co/datasets/Team-ACE/ToolACE、hermes-function-calling-v1: https://huggingface.co/datasets/NousResearch/hermes-function-calling-v1
- 对话模板（2026-09 读取）：Qwen3-0.6B: https://huggingface.co/Qwen/Qwen3-0.6B/blob/main/tokenizer_config.json、Qwen3.5-0.8B: https://huggingface.co/Qwen/Qwen3.5-0.8B/blob/main/chat_template.jinja、SmolLM3-3B: https://huggingface.co/HuggingFaceTB/SmolLM3-3B/blob/main/chat_template.jinja、Hermes-3-Llama-3.1-8B: https://huggingface.co/NousResearch/Hermes-3-Llama-3.1-8B、Olmo-3-7B-Instruct: https://huggingface.co/allenai/Olmo-3-7B-Instruct/blob/main/chat_template.jinja
- nanochat: https://github.com/karpathy/nanochat 的 `scripts/chat_sft.py`；minimind: https://github.com/jingyaogong/minimind 的 SFT 部分（中文小模型的完整 SFT 流程）
- CS336 作业 5：https://github.com/stanford-cs336/assignment5-alignment

### 第 17 章
- Kim & Rush, Sequence-Level Knowledge Distillation: https://arxiv.org/abs/1606.07947
- Agarwal et al., On-Policy Distillation of Language Models（GKD）: https://arxiv.org/abs/2306.13649
- Thinking Machines, On-Policy Distillation: https://thinkingmachines.ai/blog/on-policy-distillation
- Gemma 2: https://arxiv.org/abs/2408.00118
- Minitron（剪枝 + 蒸馏）: https://arxiv.org/abs/2407.14679

### 第 18 章
- Rafailov et al. *Direct Preference Optimization: Your Language Model is Secretly a Reward Model*，2023：https://arxiv.org/abs/2305.18290
- Stiennon et al. *Learning to summarize from human feedback*，2020：https://arxiv.org/abs/2009.01325
- Schulman et al. *Proximal Policy Optimization Algorithms*，2017：https://arxiv.org/abs/1707.06347
- Gao, Schulman, Hilton. *Scaling Laws for Reward Model Overoptimization*，2022：https://arxiv.org/abs/2210.10760
- Qwen Team. *Qwen2 Technical Report*（§4.3），2024：https://arxiv.org/abs/2407.10671；*Qwen2.5 Technical Report*（§4.2–4.3），2024：https://arxiv.org/abs/2412.15115
- DeepSeek-AI. *DeepSeek LLM: Scaling Open-Source Language Models with Longtermism*（§4），2024：https://arxiv.org/abs/2401.02954
- NVIDIA. *Nemotron-4 340B Technical Report*（§3.3.2 DPO 与 RPO），2024：https://arxiv.org/abs/2406.11704
- Tunstall et al. *Zephyr: Direct Distillation of LM Alignment*，2023：https://arxiv.org/abs/2310.16944
- Hugging Face. *SmolLM3: smol, multilingual, long-context reasoner*（博客）：https://github.com/huggingface/blog/blob/main/smollm3.md
- D'Oosterlinck et al. *Anchored Preference Optimization and Contrastive Revisions*（APO），2024：https://arxiv.org/abs/2408.06266
- Razin et al. *Unintentional Unalignment: Likelihood Displacement in Direct Preference Optimization*，2024：https://arxiv.org/abs/2410.08847
- Pal et al. *Smaug: Fixing Failure Modes of Preference Optimisation with DPO-Positive*，2024：https://arxiv.org/abs/2402.13228
- Park et al. *Disentangling Length from Quality in Direct Preference Optimization*，2024：https://arxiv.org/abs/2403.19159
- Singhal et al. *A Long Way to Go: Investigating Length Correlations in RLHF*，2023：https://arxiv.org/abs/2310.03716
- 变体（前沿观察）：IPO https://arxiv.org/abs/2310.12036；KTO https://arxiv.org/abs/2402.01306；SimPO https://arxiv.org/abs/2405.14734；ORPO https://arxiv.org/abs/2403.07691
- 偏好数据集：UltraFeedback: https://huggingface.co/datasets/openbmb/UltraFeedback（MIT）、HelpSteer3: https://huggingface.co/datasets/nvidia/HelpSteer3（CC-BY-4.0）、Tülu 3 8B 偏好混合: https://huggingface.co/datasets/allenai/llama-3.1-tulu-3-8b-preference-mixture（ODC-BY-1.0，部分子集不可商用）

### 第 19 章
- Williams. *Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning* (REINFORCE), 1992：https://link.springer.com/article/10.1007/BF00992696
- Shao et al. *DeepSeekMath*（GRPO、k3 KL、统一的梯度系数视角）, 2024：https://arxiv.org/abs/2402.03300
- DeepSeek-AI. *DeepSeek-R1*, 2025：https://arxiv.org/abs/2501.12948
- Qwen Team. *Qwen3 Technical Report*, 2025：https://arxiv.org/abs/2505.09388；*Group Sequence Policy Optimization*：https://arxiv.org/abs/2507.18071
- Olmo Team. *Olmo 3*, 2025：https://arxiv.org/abs/2512.13961
- Yu et al. *DAPO*, 2025：https://arxiv.org/abs/2503.14476
- Liu et al. *Understanding R1-Zero-Like Training: A Critical Perspective*（Dr. GRPO）, 2025：https://arxiv.org/abs/2503.20783
- Schulman. *Approximating KL Divergence*：http://joschu.net/blog/kl-approx.html
- Lilian Weng. *Reward Hacking in Reinforcement Learning*, 2024：https://lilianweng.github.io/posts/2024-11-28-reward-hacking/
- verl（HybridFlow）：https://github.com/verl-project/verl

### 第 20 章
- GGUF 规范: https://github.com/ggml-org/ggml/blob/master/docs/gguf.md
- llama.cpp k-quants（PR #1684）: https://github.com/ggml-org/llama.cpp/pull/1684
- Mitchell et al., Model Cards for Model Reporting: https://arxiv.org/abs/1810.03993
- Dettmers et al., LLM.int8(): https://arxiv.org/abs/2208.07339 ; Frantar et al., GPTQ: https://arxiv.org/abs/2210.17323

### 第 21 章
- DeepSeek-V2（MLA）: https://arxiv.org/abs/2405.04434
- Kimi K2 Technical Report: https://arxiv.org/abs/2507.20534
- GLM-5 Technical Report: https://arxiv.org/abs/2602.15763
- FlashMLA: https://github.com/deepseek-ai/FlashMLA

### 第 22 章
- Child, Gray, Radford, Sutskever. *Generating Long Sequences with Sparse Transformers*，2019：https://arxiv.org/abs/1904.10509
- Beltagy, Peters, Cohan. *Longformer: The Long-Document Transformer*（滑动窗口 + 全局 token），2020：https://arxiv.org/abs/2004.05150
- Jiang et al. *Mistral 7B*（滑动窗口、滚动缓冲区缓存、理论跨度），2023：https://arxiv.org/abs/2310.06825
- Xiao et al. *Efficient Streaming Language Models with Attention Sinks*（StreamingLLM），2023：https://arxiv.org/abs/2309.17453
- Yuan et al. *Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention*（NSA），2025：https://arxiv.org/abs/2502.11089
- Lu et al. *MoBA: Mixture of Block Attention for Long-Context LLMs*，2025：https://arxiv.org/abs/2502.13189
- DeepSeek-AI. *DeepSeek-V3.2*（DSA）技术报告：https://huggingface.co/deepseek-ai/DeepSeek-V3.2/blob/main/assets/paper.pdf
- DeepSeek-AI. *DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence*，2026：https://arxiv.org/abs/2606.19348
- Lai et al. *MiniMax Sparse Attention*，2026：https://arxiv.org/abs/2606.13392
- MiniCPM Team. *MiniCPM4*，2025：https://arxiv.org/abs/2506.07900；*InfLLM-V2*，2025：https://arxiv.org/abs/2509.24663
- Dao. FlashAttention（`window_size` 参数）：https://github.com/Dao-AILab/flash-attention；PyTorch FlexAttention 博客（滑动窗口 `mask_mod` 示例）：https://pytorch.org/blog/flexattention/

### 第 23 章
- Katharopoulos et al., Transformers are RNNs（线性注意力）: https://arxiv.org/abs/2006.16236
- Yang et al., Parallelizing Linear Transformers with the Delta Rule: https://arxiv.org/abs/2406.06484
- Gated Delta Networks: https://arxiv.org/abs/2412.06464
- Kimi Linear: https://arxiv.org/abs/2510.26692
- MiniMax-01: https://arxiv.org/abs/2501.08313
- Nemotron-H: https://arxiv.org/abs/2504.03624
- flash-linear-attention: https://github.com/fla-org/flash-linear-attention

### 第 24 章
- Shazeer et al. *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer*（MoE 层与路由坍缩问题），2017：https://arxiv.org/abs/1701.06538
- Lepikhin et al. *GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding*（top-2 路由、容量、辅助损失、专家并行），2020：https://arxiv.org/abs/2006.16668
- Fedus, Zoph, Shazeer. *Switch Transformers*（top-1 路由、负载均衡损失、容量因子），2021：https://arxiv.org/abs/2101.03961
- Gale et al. *MegaBlocks: Efficient Sparse Training with Mixture-of-Experts*（dropless、块稀疏矩阵乘），2022：https://arxiv.org/abs/2211.15841
- Jiang et al. *Mixtral of Experts*，2024：https://arxiv.org/abs/2401.04088
- Dai et al. *DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models*（细粒度专家、共享专家），2024：https://arxiv.org/abs/2401.06066
- Wang et al. *Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts*，2024：https://arxiv.org/abs/2408.15664
- NVIDIA. *Nemotron 3 Nano: Open, Efficient Mixture-of-Experts Hybrid Mamba-Transformer Model for Agentic Reasoning*（2.1、2.4 节），2025：https://arxiv.org/abs/2512.20848
- Muennighoff et al. *OLMoE: Open Mixture-of-Experts Language Models*，2024：https://arxiv.org/abs/2409.02060
- Hugging Face transformers 的 MoE 实现（`models/qwen3_moe`、`models/deepseek_v3`、`models/llama4`）：https://github.com/huggingface/transformers/tree/main/src/transformers/models
- DeepSeek. DeepEP（专家并行通信库）：https://github.com/deepseek-ai/DeepEP

### 第 25 章
- Leviathan et al., Fast Inference from Transformers via Speculative Decoding: https://arxiv.org/abs/2211.17192
- Chen et al., Accelerating LLM Decoding with Speculative Sampling: https://arxiv.org/abs/2302.01318
- Gloeckle et al., Better & Faster LLMs via Multi-token Prediction: https://arxiv.org/abs/2404.19737
- MiMo: https://arxiv.org/abs/2505.07608
- CS336 lectures 代码（Spring 2026）: https://github.com/stanford-cs336/lectures

### 第 26 章
- Kimi Team. *Kimi K3 Technical Report*：https://github.com/MoonshotAI/Kimi-K3/blob/main/k3_tech_report.pdf
- GLM-5 Team. *GLM-5: from Vibe Coding to Agentic Engineering*，2026：https://arxiv.org/abs/2602.15763；IndexShare：https://arxiv.org/abs/2603.12201
- Qwen Team. *Qwen3.8-Max* 博客：https://qwen.ai/blog?id=qwen3.8
- 演化树上各节点的出处：GPT-2（Radford 等 2019）；Llama https://arxiv.org/abs/2302.13971；RMSNorm https://arxiv.org/abs/1910.07467；RoPE https://arxiv.org/abs/2104.09864；YaRN https://arxiv.org/abs/2309.00071；GLU 变体 https://arxiv.org/abs/2002.05202；GQA https://arxiv.org/abs/2305.13245；QK-Norm https://arxiv.org/abs/2010.04245；共享 embedding https://arxiv.org/abs/1608.05859；DeepSeekMoE https://arxiv.org/abs/2401.06066；无辅助损失均衡 https://arxiv.org/abs/2408.15664；MLA（DeepSeek-V2）https://arxiv.org/abs/2405.04434；Mistral 7B https://arxiv.org/abs/2310.06825；Gemma 2 https://arxiv.org/abs/2408.00118；Gated DeltaNet https://arxiv.org/abs/2412.06464；MTP（DeepSeek-V3）https://arxiv.org/abs/2412.19437
