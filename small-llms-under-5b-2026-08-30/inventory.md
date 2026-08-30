# Publicly documented small generative language models under 5B parameters

Cutoff: **strictly fewer than 5 billion total parameters**. Updated 2026-08-30.

This is a family-level inventory, not a list of every checkpoint. Base/chat/instruct variants, quantizations, and ordinary third-party fine-tunes are collapsed into one row. Included: publicly documented autoregressive text/code models and multimodal models with a qualifying language backbone/total count. Excluded: encoder-only models, embedding/reranker models, and MoEs with fewer than 5B *active* but at least 5B *total* parameters.

| Company / organization | Model family | Qualifying sizes | Notes / primary source |
|---|---|---:|---|
| Meta | OPT | 125M, 350M, 1.3B, 2.7B | Text; [model family](https://huggingface.co/docs/transformers/model_doc/opt) |
| Meta | Galactica | 125M, 1.3B | Scientific text; [paper](https://arxiv.org/abs/2211.09085) |
| Meta | Llama 3.2 | 1B, 3B | Text variants; [model card](https://huggingface.co/meta-llama/Llama-3.2-1B) |
| Meta research | MobileLLM | 125M, 350M, 600M, 1B, 1.5B | On-device; [paper](https://arxiv.org/abs/2402.14905) |
| Microsoft | Phi-1 / Phi-1.5 | 1.3B | Code/text; [paper](https://arxiv.org/abs/2306.11644) |
| Microsoft | Phi-2 | 2.7B | Text; [release](https://www.microsoft.com/en-us/research/blog/phi-2-the-surprising-power-of-small-language-models/) |
| Microsoft | Phi-3 Mini / Phi-3.5 Mini | 3.8B | Text; [technical report](https://www.microsoft.com/en-us/research/publication/phi-3-technical-report-a-highly-capable-language-model-locally-on-your-phone/) |
| Microsoft | Phi-4 Mini / Mini Reasoning | 3.8B | Text/reasoning; [model card](https://huggingface.co/microsoft/Phi-4-mini-instruct) |
| Google DeepMind | Gemma | 2B | Text; [release](https://blog.google/technology/developers/gemma-open-models/) |
| Google DeepMind | RecurrentGemma | 2B | Griffin recurrent architecture; [model card](https://huggingface.co/google/recurrentgemma-2b) |
| Google DeepMind | Gemma 2 | 2B | Text; [release](https://blog.google/technology/developers/google-gemma-2/) |
| Google DeepMind | Gemma 3 | 270M, 1B, 4B | 270M/1B text; 4B multimodal; [model card](https://ai.google.dev/gemma/docs/core/model_card_3) |
| Google | Gemini Nano 1 | 1.8B, 3.25B | Closed/on-device; [technical report](https://arxiv.org/abs/2403.05530) |
| Apple | OpenELM | 270M, 450M, 1.1B, 3B | Text; [research page](https://machinelearning.apple.com/research/openelm) |
| Apple | Apple Foundation Model (on-device) | ~3B | Closed/on-device; [technical report](https://machinelearning.apple.com/research/apple-intelligence-foundation-language-models) |
| IBM | Granite 3.x | 1B, 2B, 3B | Text; [model docs](https://www.ibm.com/granite/docs/models/granite) |
| IBM | Granite 4.x H-1B / H-Micro / Micro | 1.5B, 3B | Dense total counts; [model docs](https://www.ibm.com/granite/docs/models/granite4-0) |
| NVIDIA | Minitron | 4B | Pruned text model; [model card](https://huggingface.co/nvidia/Minitron-4B-Base) |
| NVIDIA | Nemotron Mini | 4B | Text/instruct; [model card](https://build.nvidia.com/nvidia/nemotron-mini-4b-instruct/modelcard) |
| NVIDIA | Nemotron 3 Nano | 4B | Hybrid/MoE family member with 4B total; [family page](https://research.nvidia.com/labs/nemotron/Nemotron-3/) |
| NVIDIA | Hymba | 1.5B | Hybrid-head text model; [model card](https://huggingface.co/nvidia/Hymba-1.5B-Base) |
| Alibaba / Qwen | Qwen 1 / 1.5 | 0.5B, 1.8B, 4B | Text; [repository](https://github.com/QwenLM/Qwen1.5) |
| Alibaba / Qwen | Qwen 2 | 0.5B, 1.5B | Text; [release](https://qwenlm.github.io/blog/qwen2/) |
| Alibaba / Qwen | Qwen 2.5 | 0.5B, 1.5B, 3B | Text; [release](https://qwenlm.github.io/blog/qwen2.5/) |
| Alibaba / Qwen | Qwen 2.5 Coder | 0.5B, 1.5B, 3B | Code; [technical report](https://arxiv.org/abs/2409.12186) |
| Alibaba / Qwen | Qwen 2.5 Math | 1.5B | Math; [release](https://qwenlm.github.io/blog/qwen2.5/) |
| Alibaba / Qwen | Qwen 3 | 0.6B, 1.7B, 4B | Text/reasoning; [release](https://qwenlm.github.io/blog/qwen3/) |
| Alibaba / Qwen | Qwen 3.5 | 0.8B, 2B, 4B | Multimodal; [official repository](https://github.com/QwenLM/Qwen3.5) |
| Tencent | Hunyuan dense | 0.5B, 1.8B, 4B | Text; [official model card](https://huggingface.co/tencent/Hunyuan-0.5B-Pretrain) |
| DeepSeek | DeepSeek LLM | 1.3B | Text; [official repository](https://github.com/deepseek-ai/DeepSeek-LLM) |
| DeepSeek | DeepSeek Coder | 1.3B | Code; [official repository](https://github.com/deepseek-ai/DeepSeek-Coder) |
| DeepSeek | DeepSeek-R1-Distill-Qwen | 1.5B | Reasoning derivative; [model card](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B) |
| Shanghai AI Laboratory | InternLM 2 / 2.5 | 1.8B | Text/chat; [official repository](https://github.com/InternLM/InternLM) |
| OpenBMB / Tsinghua NLP | MiniCPM 1/2 | 1.2B, 2.4B | Text; [official repository](https://github.com/OpenBMB/MiniCPM) |
| OpenBMB / Tsinghua NLP | MiniCPM 3 | 4B | Text; [model card](https://huggingface.co/openbmb/MiniCPM3-4B) |
| OpenBMB / Tsinghua NLP | MiniCPM 4 | 0.5B | Text; [model card](https://huggingface.co/openbmb/MiniCPM4-0.5B) |
| OpenBMB / Tsinghua NLP | MiniCPM 5 | 1B | Text/reasoning; [model card](https://huggingface.co/openbmb/MiniCPM5-1B) |
| 01.AI | Yi-Coder | 1.5B | Code; [model card](https://huggingface.co/01-ai/Yi-Coder-1.5B-Chat) |
| LG AI Research | EXAONE 3.5 / EXAONE Deep | 2.4B | Text/reasoning; [collection](https://huggingface.co/collections/LGAI-EXAONE/exaone-35) |
| Kakao | Kanana Nano / Kanana 2 | 2.1B, 3B | Korean/English text; [organization](https://huggingface.co/kakaocorp) |
| Nanbeige | Nanbeige 4.1 | 3B | Text; [model card](https://huggingface.co/Nanbeige/Nanbeige4.1-3B) |
| CyberAgent | OpenCALM | 160M, 410M, 830M, 1.4B, 3B | Japanese; [official repository](https://github.com/cyberagentailab/open-calm) |
| Preferred Networks | PLaMo 2 | 1B | Japanese/English; [model card](https://huggingface.co/pfnet/plamo-2-1b) |
| Rakuten | Rakuten AI 2.0 Mini | 1.5B | Japanese/English; [organization](https://huggingface.co/Rakuten) |
| rinna | Japanese/Bilingual GPT-NeoX | 300M, 1.3B, 3.6B | Japanese/English; [organization](https://huggingface.co/rinna) |
| Sakana AI | TinySwallow | 1.5B | Japanese derivative; [model card](https://huggingface.co/SakanaAI/TinySwallow-1.5B-Instruct) |
| Sarvam AI | Sarvam-1 | 2B | Indian languages; [model card](https://huggingface.co/sarvamai/sarvam-1) |
| Lelapa AI | InkubaLM | 0.4B | African languages; [model card](https://huggingface.co/lelapa/InkubaLM-0.4B) |
| Technology Innovation Institute | Falcon-RW | 1B | Text; [model card](https://huggingface.co/tiiuae/falcon-rw-1b) |
| Technology Innovation Institute | Falcon 3 | 1B, 3B | Text; [official repository](https://github.com/tiiuae/Falcon3) |
| Technology Innovation Institute | Falcon-H1 | 0.5B, 1.5B, 1.5B-Deep, 3B | Hybrid attention/SSM; [release](https://falcon-lm.github.io/blog/falcon-h1/) |
| H2O.ai | H2O-Danube / Danube 2 | 1.8B | Text; [release](https://h2o.ai/company/press-releases/2024/H2O-Danube2-1-8B-Achieves-Top-Ranking-for-2-Billion-Parameters-Range/) |
| H2O.ai | H2O-Danube 3 | 0.5B, 4B | Text; [organization](https://huggingface.co/h2oai) |
| Liquid AI | LFM2 | 350M, 700M, 1.2B, 2.6B | Hybrid foundation models; [family page](https://www.liquid.ai/) |
| Liquid AI | LFM2.5 | 1.2B, 2.6B | Text/reasoning; [model card](https://huggingface.co/LiquidAI/LFM2.5-2.6B) |
| Mistral AI | Ministral 3 | 3B | Text+vision; [official docs](https://docs.mistral.ai/models/ministral-3-3b-25-12) |
| Stability AI | StableLM Alpha | 3B | Text; [official repository](https://github.com/Stability-AI/StableLM) |
| Stability AI | StableLM 2 | 1.6B | Text; [model card](https://huggingface.co/stabilityai/stablelm-2-1_6b) |
| Stability AI | StableCode | 3B | Code; [official repository](https://github.com/Stability-AI/StableCode) |
| Hugging Face | SmolLM | 135M, 360M, 1.7B | Text; [release](https://huggingface.co/blog/smollm) |
| Hugging Face | SmolLM2 | 135M, 360M, 1.7B | Text; [release](https://huggingface.co/blog/smollm2) |
| Hugging Face | SmolLM3 | 3B | Text/reasoning; [release](https://huggingface.co/blog/smollm3) |
| Allen Institute for AI | OLMo | 1B | Fully open text model; [release](https://allenai.org/olmo) |
| Allen Institute for AI | OLMo 2 | 1B | Fully open text model; [release](https://allenai.org/blog/olmo2) |
| EleutherAI | GPT-Neo | 125M, 1.3B, 2.7B | Text; [official repository](https://github.com/EleutherAI/gpt-neo) |
| EleutherAI | Pythia | 70M, 160M, 410M, 1B, 1.4B, 2.8B | Text/research suite; [official repository](https://github.com/EleutherAI/pythia) |
| BigScience | BLOOM / BLOOMZ | 560M, 1.1B, 1.7B, 3B | Multilingual text; [model family](https://huggingface.co/bigscience/bloom-3b) |
| Cerebras | Cerebras-GPT | 111M, 256M, 590M, 1.3B, 2.7B | Text/research suite; [model family](https://huggingface.co/cerebras/Cerebras-GPT-2.7B) |
| Databricks | Dolly v2 | 3B | Instruction model; [official repository](https://github.com/databrickslabs/dolly) |
| Salesforce | CodeGen / CodeGen2 | 350M, 1B, 2B, 3.7B | Code; [official repository](https://github.com/salesforce/CodeGen) |
| BigCode (ServiceNow + Hugging Face) | SantaCoder / StarCoderBase / StarCoder2 | 1.1B, 1B, 3B | Code; [BigCode models](https://huggingface.co/bigcode) |
| RWKV Foundation | RWKV 4/5/6/7 | ~169M–2.9B variants | Recurrent language models; [official repository](https://github.com/BlinkDL/RWKV-LM) |
| MBZUAI / collaborators | MobiLlama | 0.5B, 1B | Mobile text model; [official repository](https://github.com/mbzuai-oryx/MobiLlama) |
| Princeton NLP | Sheared-LLaMA | 1.3B, 2.7B | Structured-pruned Llama; [official repository](https://github.com/princeton-nlp/LLM-Shearing) |
| TensorOpera AI | Fox | 1.6B | Text; [model card](https://huggingface.co/tensoropera/Fox-1-1.6B) |
| M.A.P. | CT-LLM | 2B | Chinese/English; [model card](https://huggingface.co/m-a-p/CT-LLM-Base) |
| EuroLLM consortium / Unbabel | EuroLLM | 1.7B | European multilingual; [organization](https://huggingface.co/utter-project) |
| Barcelona Supercomputing Center | Salamandra | 2B | Iberian/European multilingual; [model card](https://huggingface.co/BSC-LT/salamandra-2b) |
| Common Corpus consortium | CroissantLLM | 1.3B | English/French; [model card](https://huggingface.co/croissantllm/CroissantLLMBase) |
| Swiss AI Initiative (ETH Zurich, EPFL, CSCS) | Apertus v1.1 | 0.5B, 1.5B, 4B | Fully open multilingual; [model card](https://huggingface.co/swiss-ai/Apertus-v1.1-1.5B) |
| Multiverse Computing | LittleLamb | 0.3B | Compressed/on-device family; [release](https://multiversecomputing.com/resources/introducing-the-littlelamb-0.3b-model-family) |

## Important exclusions and edge cases

- **Exactly 5B is excluded**, because the request says “under 5B.”
- **Gemma 4 E2B/E4B are excluded:** Google documents 5.1B and 8B total parameters respectively; E2B/E4B are effective counts.
- **Gemma 3n E2B/E4B are excluded** under the same total-parameter rule: their deployment/effective counts are below 5B, but total stored parameters are not.
- **Qwen3-30B-A3B, Granite H-Tiny, LFM2.5-8B-A1B, DeepSeek MoEs, and similar models are excluded:** active parameters are small, total parameters are not.
- Model counts are not globally standardized. Some cards exclude embeddings, vision/audio encoders, or tied parameters. This inventory follows the publisher’s documented total where available and flags the most consequential naming traps above.
- “All” cannot be proved literally: private models, undocumented parameter counts, deleted releases, and thousands of third-party derivatives make the universe open-ended. This inventory targets identifiable first-party or research-origin model families with public parameter documentation.

## Research basis

The starting benchmark was the 2024 academic survey of 70 open decoder-only SLMs in the 100M–5B range: [Small Language Models: Survey, Measurements, and Insights](https://doi.org/10.48550/arxiv.2409.15790). It was extended through first-party releases and model cards through 2026-08-30. A current community cross-check was [Awesome Small Language Models](https://github.com/agi-templar/Awesome-Small-Language-Model), but rows above prefer first-party sources.
