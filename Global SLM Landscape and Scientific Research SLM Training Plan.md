# **全球端侧与小参数大语言模型（SLM）产业生态、技术演进与科研专用智能体构建全景指南**

人工智能领域的演进轨迹长期以来被“缩放定律”（Scaling Law）所主导，该定律指出模型性能随着参数规模、数据量和计算资源的增加而呈现可预测的提升。然而，随着云端算力成本的指数级飙升、能源消耗的加剧，以及智能手机、个人电脑、智能座舱和具身机器人等边缘终端对低延迟、高隐私数据处理需求的爆发，全球人工智能产业正在经历一场深刻的范式转移。这场转移的核心，是从盲目追求万亿参数的超大语言模型，转向探索在极低功耗和受限内存下依然能展现出卓越推理能力的“小参数大语言模型”（Small Large Language Models, SLMs）及端侧智能技术。  
本研究报告旨在全面剖析全球小语言模型领域的商业格局与核心技术脉络。报告以中国端侧大模型的拓荒者面壁智能（ModelBest）为切入点，辐射至微软（Microsoft）、谷歌（Google）等全球科技巨头的SLM战略，系统性梳理当前小语言模型在数据工程、训练框架、后训练对齐、模型蒸馏等维度的前沿技术方案。在深度解析多份重磅技术报告的基础上，本报告最终将落脚于一个极具前瞻性的应用命题：如何利用最先进的SLM底层技术与智能体（Agent）框架，从零开始训练并部署一个专门用于辅助科学研究、具备长文本解析与深度逻辑推演能力的专业科研小语言模型。

## **全球小语言模型（SLM）公司的商业与技术竞争格局**

在端侧智能的赛道上，初创企业与国际科技巨头正在通过截然不同却又殊途同归的路径，重新定义“智能密度”的边界。这些企业的商业化进程与技术突破，共同构成了当前SLM领域的全景生态。

### **面壁智能（ModelBest）：端侧智能的商业范本与资本锚点**

面壁智能脱胎于清华大学自然语言处理（NLP）实验室，由前谷歌中国创始员工、知乎CTO李大海担任首席执行官，清华大学计算机系教授刘知远担任首席科学家。在“百模大战”初期，当行业普遍陷入云端算力焦虑时，面壁智能前瞻性地确立了“以小博大、高效低成本”的端侧大模型战略，致力于让小模型释放大能量。这一战略使其在资本市场获得了极高的估值溢价与产业认可。2026年上半年，面壁智能累计融资规模已超过50亿元人民币，投后估值突破200亿元，成为端侧智能领域估值最大的独角兽企业，其投资方阵容涵盖了中国电信、深创投、华为哈勃、春华创投、知乎等产业资本与顶尖财务投资机构。更为标志性的是，面壁智能已于2026年8月在中信证券的辅导下，正式启动A股IPO进程，这不仅是公司发展的里程碑，更标志着端侧大模型的商业闭环正在接受公开市场的最终检验。  
在商业化落地方面，面壁智能构建了极具渗透力的生态网络。在智能终端领域，其自主研发的MiniCPM系列端侧模型（开源仓库及技术报告链接：https://github.com/openbmb/minicpm）已成功搭载于三星Galaxy Z Fold 8等旗舰机型中，成为首批通过国家网信办端侧生成式人工智能服务备案的模型之一。同时，公司与高通达成深度合作，基于第五代骁龙8至尊版移动平台推出了创新的端侧智能体AgentCPM，推动AI手机进入原生智能时代。在智能汽车与智能座舱赛道，面壁智能发布了全球首个落地车端的纯端侧智能助手“cpmGO”，基于车规级SoC完成语音、视觉与车控推理的本地处理，并与车联天下、长安马自达、上汽大众等达成战略合作。此外，其端侧芯算融合大模型项目正式通过了Automotive SPICE（ASPICE）L2级国际评估，成为国内首个获此车规级认证的大模型企业。在垂直行业应用中，面壁智能参与构建了国家级基础设施“法信法律基座大模型”，其司法审判辅助系统已在深圳中院实现从立案到结案的全流程覆盖；在教育领域，其与清华大学联合推出的AI伴学助手“清小搭”已全面服务于高校师生，赋能全自动课堂模式。

### **微软（Microsoft）：合成数据压榨与多模态端侧矩阵**

微软在小语言模型领域构建了多层次的立体矩阵，向业界证明了数据质量在SLM训练中的决定性作用。其著名的Phi-4系列（Phi-4技术报告链接：https://arxiv.org/abs/2603.03975）彻底颠覆了依赖网络爬虫有机数据的预训练范式，通过多智能体提示和自我修正工作流生成的高质量合成数据，大幅提升了模型在STEM领域的复杂逻辑推理能力。最新的Phi-4系列进一步细分出了多个极具针对性的专家模型1：

* **Phi-4-mini-reasoning (3.8B):** 专为数学和结构化推理构建。  
* **Phi-4-multimodal:** 首次在Phi家族中支持文本、音频和视觉，具备超过20种语言的20万词汇量，可进行语音识别、翻译及图表理解。  
* **Phi-4-reasoning-vision-15B:** 紧凑的15B视觉推理模型，非常擅长文档理解、数学图像解析和UI屏幕元素的精准定位导航。

此外，微软更是推出了主打特定任务的**MAI系列端侧大模型**（系列资源主页：https://huggingface.co/microsoft）。MAI系列涵盖了不同尺寸的高效智能体模型，包括：专为GitHub Copilot与VS Code深度定制的MAI-Code-1.1-Flash；拥有强大推理能力的中量级模型MAI-Thinking-1；以及针对计算机视觉自动化与浏览器控制专门设计的Fara-7B与9B旗舰端侧智能体模型。

### **谷歌（Google）：Gemma 4 的智能密度跃升与多模态原生**

谷歌的Gemma系列已跨入**Gemma 4**时代，该系列强调“单位参数智能”（intelligence-per-parameter）和专为智能体工作流设计1。Gemma 4 提供了多个针对不同硬件预算的尺寸：

* **E2B (Effective 2B)与E4B (Effective 4B):** 专为移动端及IoT设备设计，支持高达128K上下文。它们原生支持文本、图像和音频输入，实现零延迟离线运行。  
* **12B 多模态:** 首个无编码器（Encoder-free）的中等尺寸多模态模型，视觉和音频信号直接进入LLM主干，非常适合在16GB显存的普通笔记本电脑上本地运行。  
* **26B MoE 与 31B Dense:** 面向工作站的旗舰推理模型。其中26B MoE模型在推理时仅激活约3.8B参数，而31B模型支持高达256K的上下文窗口。  
  Gemma 4 通过局部注意力与全局注意力交替（并引入p-RoPE、K=V等优化），以及原生支持结构化JSON输出与函数调用，极大拓宽了端侧智能体开发生态。

### **Mistral AI：Ministral 3 系列的端侧突围**

欧洲开源巨头 Mistral AI 推出了专为边缘计算和本地部署打造的 **Ministral 3** 系列1。该系列提供 **3B、8B、14B** 三种尺寸，且每种尺寸均包含基础版（Base）、指令微调版（Instruct）和专门针对复杂问题设计、能够生成更多思考Token的推理版（Reasoning）。Ministral 3 模型拥有 256K 超大上下文窗口，同时支持文本与图像模态，并在性价比与多语言本地应用中展现出极高竞争力。

### **NVIDIA：Nemotron 3 Nano 的混合MoE端侧架构**

NVIDIA 推出的 **Nemotron 3 Nano** 和 **Nemotron 3 Nano Omni** 重新定义了端侧大模型的设计思路1。

* 该系列虽然拥有 **30B 的总参数量**，但由于采用了混合 Mamba-Transformer 专家混合（MoE）架构，推理时**仅激活约 3B 参数**。  
* 支持高达 **100万（1M）Tokens** 的上下文。  
* 其中 **Omni 版本** 将视频、音频、图像和文本推理统一在单个高效系统中，避免了在智能体架构中拼接多个不同模型的低效。非常适合长文本RAG、代码生成及多模态端侧智能体环路。

### **Cohere Labs：Tiny Aya 与 North Mini Code 专家模型**

Cohere 在轻量化和专业化上推出了两款核心模型1：

* **Tiny Aya (3.35B):** 专为解决多语言和地域性覆盖而生。虽然只有不到 4B 参数，但它支持超过70种语言，为计算资源受限地区提供了极具包容性的本地化多语言AI方案。  
* **North Mini Code:** 一款专为软件Agent设计的 **30B总参数/3B激活参数 MoE** 编程代码大模型。它不是通用对话助手，而是专门针对端侧或本地自主编程系统（如执行终端命令、修改代码库）优化的利器。

### **阿里巴巴（Alibaba）：Qwen3与“思考模式”的端侧下放**

阿里巴巴的通义千问（Qwen）团队在小参数模型领域的迭代极为迅速，目前已发布了全面的**Qwen3**系列（开源及技术文档链接：https://huggingface.co/Qwen）。Qwen3涵盖了参数规模从0.6B、1.7B、4B、8B、14B、32B到235B（MoE）的多个版本1。 Qwen3架构中的一项核心创新是将“思考模式”（Thinking mode，适用于复杂多步推理）和“非思考模式”（Non-thinking mode，适用于快速响应）融合进了一个统一的框架中，支持基于任务复杂度的动态计算资源分配（思考预算机制）。此外，Qwen3将多语言支持从上一代（Qwen2.5）的29种语言扩展到了119种语言及方言，其0.6B到8B等极小参数模型在代码生成、数学推理与Agent任务中均展现出超越同级甚至更大参数模型的SOTA表现。

### **Meta：Llama 3.2 极致边缘设备的普及者**

Meta 推出的 **Llama 3.2 (1B 和 3B)** 虽然属于纯文本模型，但其因极致轻量化和庞大的生态支持，已成为移动端、边缘设备上的经典选择。它们拥有128K上下文，极适合在智能手机、PC上执行本地RAG、文本摘要及轻量级指令响应1。

### **IBM：Granite 4.1 的企业级精准赋能**

IBM 开源的 **Granite 4.1** 系列（包含 3B、8B、30B 版本）深度契合企业级开发诉求，专注在代码生成、指令遵循以及高度可预测的商业工作流处理上，为企业在受限算力下部署安全私有模型提供了坚实底座1。

### **Liquid AI：基于非Transformer架构的创新探索**

作为前沿基础模型架构探索者，Liquid AI 突破了传统 Transformer 的架构限制，推出了 Liquid Foundation Models (LFM) 系列（技术主页链接：https://www.liquid.ai/liquid-foundation-models），涵盖 LFM-1B、3B 和 7B 等小参数版本。其中 LFM-7B 展现出卓越的对话能力（包括支持阿拉伯语和日语等），由于其基于特殊的 Liquid 架构底层原理，能够在边缘终端上实现极低的内存占用和极快的推理速度，为端侧低算力硬件提供了一种极具成本效益的非 Transformer 替代方案。

### **HuggingFace：SmolLM 家族的极简本地化**

代表开源社区极致轻量化探索的 HuggingFace 推出了 SmolLM 家族。除了专为极受限环境设计的 SmolLM2 (如 1.7B) 外，HuggingFace 最新的 **SmolLM3** (3B参数) 同样开源发布。它支持高达 128K Tokens 上下文，是开源研究、本地多语言微调以及快速构建本地RAG系统的理想轻量级利器1。

### **蚂蚁集团（Ant Group）：混合推理MoE的端侧突破**

2026年8月，蚂蚁集团旗下的百灵大模型（Ling）团队在HuggingFace平台正式开源了轻量级混合推理MoE模型——**Ling-3.0-tiny**（开源及技术报告地址：https://huggingface.co/inclusionAI/Ling-3.0-tiny）。该模型专为本地部署与低成本推理设计，总参数量为7.9B，但在推理时每Token仅激活1.3B参数。在底层架构上，Ling-3.0-tiny采用了创新的混合线性架构，将KDA与MLA按3:1的比例交替堆叠，并配备了包含128个路由专家的稀疏MoE前馈网络。这种极致的架构设计使其在Apple M系列芯片（如M1/M4 Pro）上能够达到86至90 Token/s的高效推理速度，且在8K上下文长度下峰值内存占用约为8.34 GiB。更重要的是，该模型原生支持快速响应与多步骤推理（混合推理）模式的灵活切换，并同步提供了BF16、FP8和INT4三种权重版本，极大降低了边缘设备的适配门槛。

### **Cactus Compute：极微缩端侧工具调用与结构化提取 (Needle)**

Cactus Compute 推出了专为极端受限边缘设备（如智能手表、VR头显及微控制器）设计的超微型工具调用模型 **Needle** 系列。不同于追求通用对话能力的传统模型，最新的 Needle 2 仅有 45M 参数，被整体打包成一个 14MB 的独立二进制文件，运行时仅需约 28MB 的会话内存。其技术核心在于采用了“简单注意力网络（Simple Attention Network）”架构，用哈达玛多层感知机（Hadamard MLP）完全替代了传统的前馈网络（FFN），并结合了 CQ2-bit 极致量化技术。Needle 模型仅接受自然语言输入并输出经过字节级语法严格约束的 JSON 格式结构化数据，附带置信度门控。在 Raspberry Pi 5 上，其解码速度可达 500 Token/s（开源主页：https://github.com/cactus-compute/needle，技术论文：https://arxiv.org/abs/2607.18363）。

| 公司/机构名称 | 核心小语言模型系列 | 核心技术路径与创新点 | 关键商业落地场景 |
| :---- | :---- | :---- | :---- |
| **面壁智能** | MiniCPM, AgentCPM, VoxCPM | 极致轻量化架构，多模态端到端融合，InfLLM-V2稀疏注意力。 | 三星AI手机，汽车智能座舱，司法与教育垂直领域。 |
| **微软** | Phi-4系列 (Mini/Vision), MAI | 强依赖合成数据，支持屏幕UI定位推理与浏览器自动化。 | GitHub Copilot本地辅助，网页自动化，STEM推理。 |
| **谷歌** | Gemma 4 (E2B/E4B/12B/26B) | 局部/全局注意力交替，支持无编码器多模态输入，极速端侧推理。 | 移动端多模态部署，跨语言，全模态原生Agent。 |
| **Mistral AI** | Ministral 3 (3B/8B/14B) | 256K长上下文，极高性价比基础模型与推理（Reasoning）版本组合。 | 边缘计算助手，高精度多步推理任务，本地服务器。 |
| **NVIDIA** | Nemotron 3 Nano/Omni | 30B总参数/3B激活，混合Mamba-Transformer架构，支持多模态同框。 | 视频理解Agent，高吞吐本地RAG，多模态子智能体。 |
| **Cohere Labs** | Tiny Aya (3.3B), North Mini | 覆盖超70种语言的高深度小模型；专为终端执行优化的代码MoE。 | 区域性多语言本地化部署，Agentic软件工程与控制。 |
| **阿里巴巴** | Qwen3 (0.6B/1.7B/4B/8B等) | 统一思考/非思考模式框架，思考预算控制，支持119种多语言。 | 跨设备生态集成，本地代码补全，复杂数学推理辅助。 |
| **Meta** | Llama 3.2 (1B/3B) | 大模型知识蒸馏与剪枝，超强生态兼容性。 | 轻量级移动设备，基础摘要与指令遵循。 |
| **IBM** | Granite 4.1 (3B/8B/30B) | 高可预测性，针对企业代码和逻辑严密优化。 | 企业内部工具流整合，私有安全部署。 |
| **Liquid AI** | LFM系列 (1B/3B/7B) | 采用Liquid Foundation架构（非传统Transformer），超低显存占用。 | 资源极度受限环境的对话系统，高频边缘推理。 |
| **HuggingFace** | SmolLM2 / SmolLM3 (3B) | 数据集极度提纯与高效率训练，在极小尺寸保留强基础能力。 | 个人PC本地原型开发，轻量级本地多语言微调。 |
| **蚂蚁集团** | Ling-3.0-tiny (7.9B/激活1.3B) | 混合线性架构(KDA+MLA交替)，原生支持多步混合推理模式。 | 本地低显存部署，PC端侧极致推理速度验证。 |
| **Cactus Compute** | Needle 2 (45M) | 无FFN简单注意力网络，CQ2-bit极限量化，14MB二进制体积。 | 智能可穿戴设备、IoT微控制器离线API调用结构化提取。 |

## **小语言模型（SLM）的底层理论与全栈技术方案**

要理解SLM缘何能够在参数规模缩减两个数量级的情况下依然保持强大的智能，必须深入剖析其背后的理论突破、数据工程、训练架构以及智能体演进框架。

### **理论基石：从“缩放定律”到“密度定律”（Densing Law）**

当前SLM飞速发展的核心理论指导，来自于面壁智能与清华大学联合提出的“大模型密度定律”（Densing Law of LLMs），该研究成果已作为封面文章发表于国际顶级学术期刊《Nature Machine Intelligence》。在过去几年中，业界普遍信奉“缩放定律”，认为模型智能仅与参数和数据规模的增加呈正相关，但这导致了模型部署成本的失控。“密度定律”创新性地引入了“能力密度”（Capability Density）这一度量标准，用于评估单位参数所蕴含的智能质量。  
该研究通过对广泛的开源基础模型进行严谨的实证分析，揭示了一个指数级增长规律：大模型的能力密度随着时间推移呈指数级提升。具体而言，模型最大能力密度 ![][image1] 的对数与时间 ![][image2] 存在线性关系：![][image3]。计算得出系数 ![][image4]，进而推导出倍增时间 ![][image5] 天。这意味着，开源大模型的能力密度大约每3.3至3.5个月就会翻一番，达到特定智能水平所需的参数量将呈指数级下降。这一发现不仅颠覆了传统认知，更证明了通过优化数据质量与训练算法来提升“智能密度”，是实现端侧大模型可持续发展的唯一正确路径。

### **数据工程：合成数据的规模化与高质化**

在SLM的预训练与微调中，数据质量的重要性远超数据数量。以微软Phi-4模型为例，其在训练过程中战略性地将合成数据作为核心语料，彻底改变了以往依赖低质量网络有机数据的模式。  
合成数据的生成依赖于高度复杂的工作流。首先是“多智能体提示”（Multi-Agent Prompting），利用多个先进的大型语言模型相互协作，模拟各种复杂的对话、逻辑推理和代码编写场景，从而生成具有丰富语境和深度逻辑的数据集。其次是“指令逆向”（Instruction Reversal），即将现有的高质量代码片段或解答过程输入模型，要求其反向推导出原始的问题指令，这极大地增强了SLM遵循复杂指令的能力。最后是“自我修正工作流”（Self-Revision Workflows），即在数据生成阶段引入反馈循环，让生成模型自主批判并修改其初步输出，确保最终纳入训练集的合成数据在数学推理和事实准确性上达到极高标准。

### **训练框架与长文本处理机制**

端侧设备的内存瓶颈是制约SLM处理长文本的最大障碍。在生成长篇内容时，键值缓存（KV Cache）的内存占用会随着上下文长度的增加而呈平方级爆炸。业界对此提出了多种突破性的架构优化方案。  
面壁智能推出的InfLLM-V2框架是一种无需增加任何额外模型参数的“稠密-稀疏”（Dense-Sparse）切换注意力机制。它采用两阶段架构：第一阶段为Top-K上下文选择，利用压缩后的语义核（Semantic Kernels）对历史文档块进行相关性评分与聚合；第二阶段仅对筛选出的高相关性文档块执行稀疏注意力计算。这种机制允许模型在处理短文本时无缝使用高精度的稠密注意力，在处理长文本时平滑切换至高效的稀疏注意力。实验表明，通过仅5B长文本Token的训练，模型即可在保持98.1%的稠密模型长文本理解性能的基础上，将推理速度提升约4倍。  
另一条路径是谷歌Gemma 4（及其前代）采用的交替注意力机制1。Gemma 4在全局自注意力层中大幅提升RoPE（旋转位置编码）基础频率，同时高比例穿插局部滑动窗口注意力，显著限制了需要长时间保留在内存中的KV Cache数量，成功将其端侧模型扩展至128K至256K上下文。为了在多样化的硬件上高效运行，推理框架如FlagOS被广泛应用，它提供了统一的多芯片后端支持，确保SLM能在各种端侧计算芯片上实现极致的吞吐量。

### **多模态架构的端到端演进**

SLM的视觉与语音能力正逐渐摆脱拼凑式的外挂模块，走向深度的端到端融合。在视觉理解方面，谷歌Gemma 4 的12B多模态版本采取了无编码器（Encoder-free）架构，直接将视觉和音频输入接入模型主干进行原生处理，取代了以往独立的重型多模态编码器1。微软的 Phi-4-reasoning-vision-15B 则采用基于 SigLIP-2 的动态分辨率编码器，允许模型对超高分辨率（如屏幕截图）的细节特征保留极高敏锐度1。  
在语音生成领域，面壁智能开源的VoxCPM系列彻底颠覆了传统的离散分词器（Tokenizer）范式。VoxCPM采用了纯端到端的分层语义-声学扩散建模架构。它引入了可微的量化瓶颈（Differentiable Quantization Bottleneck），将文本-语义语言模型（TSLM）与残差声学模型（RALM分离。整个模型在条件流匹配（Conditional Flow-Matching）目标下进行端到端优化，无需依赖任何外部的语音Tokenizer。这种架构不仅消除了离散化带来的声学细节损失，还使模型能够理解文本上下文并自主生成具有丰富情感与起伏的韵律，支持30种语言及零样本（Zero-shot）真声克隆，且在消费级显卡上可实现低至0.17的实时率（RTF）。

### **后训练对齐与智能体（Agent）生态系统**

让SLM具备真正生产力的关键在于后训练（Post-Training）阶段的对齐技术以及将其封装为自主智能体。在对齐策略上，除了传统的监督微调（SFT）与基于人类反馈的强化学习（RLHF），基于“枢轴Token搜索”（Pivotal Token Search）的直接偏好优化（DPO）技术开始在SLM中普及。该技术通过分析生成序列中对最终逻辑正确性起决定作用的关键Token，构建更精准的拒绝/接受数据对，从而极大提升小模型在关键决策点上的鲁棒性。  
在智能体生态层面，SLM正在突破单体交互的限制。例如，阿里Qwen3系列通过原生提供思考模式，可以直接应对复杂环境任务，而微软MAI系列（如Fara-7B）则在极小的参数量下能够直接根据自然语言生成具体的UI交互坐标来进行浏览器自动化。不仅如此，针对长篇内容生成的逻辑连贯性问题，面壁智能的AgentCPM-Report智能体引入了WARP（Writing As Reasoning Policy，写作即推理）策略。该策略摒弃了僵化的“先计划后写作”模式，允许模型在长达百轮的思维链推演中，交替进行证据起草与逻辑深化，动态修改大纲，从而在8B参数下产出逻辑严密、洞察深刻的万字长文。

## **面向科学研究的专用小语言模型（Scientific AI Agent）全栈训练蓝图**

科学研究是一项高度复杂、对事实严谨性要求极高且需要处理海量多模态数据的智力活动。通用的大语言模型往往受限于长文献的内存崩溃、垂直领域知识的幻觉以及缺乏严密的推理规划能力。为了在本地受限算力（如实验室工作站或个人PC）上部署一个专门辅助文献综述、深度逻辑推演、图表解析及报告生成的“科研专用小语言模型”，我们提出以下包含六大核心步骤的详细技术方案蓝图。

### **阶段一：高密度基座模型筛选与硬件环境适配**

在算力资源有限的约束下，选择具有高初始“智能密度”的基座模型是成功的关键。  
**实施方案**：采用参数规模在1.7B至8B（或稀疏架构下活跃参数在3B左右）的先进开源SLM作为底座。除了面壁智能 MiniCPM4.1-8B、最新的阿里 Qwen3-4B 或微软 Phi-4-mini-reasoning (3.8B) 外，当前也有其他绝佳选择：如果极度看中多模态同框推理，可选择 NVIDIA 的 Nemotron 3 Nano Omni（激活仅3B）；如果注重长篇逻辑演绎，可选择 Mistral AI 提供的 Ministral 3 8B 推理版；如果面向非英语母语的跨国科研环境，则推荐 Cohere Labs 的 Tiny Aya。这些模型在数学、代码及STEM基础能力上已通过密集的预训练达到甚至超越了许多早期大模型。在硬件适配层面，应底层集成 **FlagOS** 多芯片后端插件或使用 vLLM 等推理框架，以实现高吞吐量和极致的内存管理优化。

### **阶段二：学术长上下文能力的内化扩展（Context Extension）**

科研工作不可避免地需要同时输入十余篇甚至数十篇长篇学术论文，这要求模型具备处理超100K Token的上下文能力，且推理速度不能发生断崖式下跌。  
**实施方案**：在基座模型基础上，引入并融合 **InfLLM-V2** 稀疏注意力框架或借鉴Gemma 4的局部-全局交替注意力机制进行长文本领域的持续预训练（Continual Pre-training）。

1. **领域数据准备**：清洗并收集约5B至10B Token的高质量学术语料库（涵盖arXiv、PubMed等平台的开源PDF解析文本）。  
2. **注意力机制重构**：利用InfLLM-V2框架，在不增加原有网络参数的基础上，建立两阶段上下文选择机制。模型在处理长篇文献时，首先通过压缩的语义核计算Query与各个文献数据块的相似度，随后仅对筛选出的高相关度学术段落执行稀疏注意力计算。

### **阶段三：基于合成数据与多模态图表解析的监督微调（SFT）**

科研论文中充斥着大量的图表、公式与复杂引用关系，传统的爬虫数据质量低下，容易导致严重的幻觉。  
**实施方案**：利用自动化流水线构建极高密度的合成数据，并通过工具链实现多模态对齐。

1. **私有知识库自动化处理**：部署 **UltraRAG** 框架，自动化处理含有公式与图表的学术PDF。该系统能够精准切割领域语料，构建高质量的向量索引与图文对齐数据。  
2. **多智能体合成数据生成**：借鉴Phi-4的训练哲学，利用GPT-4o等强力教师模型，采用多智能体提示与指令逆向技术。例如，将一篇前沿医学论文输入系统，要求智能体反向推导出“为了得出该结论，研究者必须先解决哪些子问题”，从而生成大量的“问题-深度推导过程（CoT）-答案”三元组。  
3. **高密度SFT**：使用这些包含严密思维链的合成数据集对模型进行监督微调，迫使小语言模型内化学术推理的逻辑范式。

### **阶段四：赋予跨数据库检索与工具调用的行动力（Tool Learning）**

一个实用的科研Agent不能仅限于内部知识，它必须能够实时查询最新文献、调用计算工具处理实验数据。  
**实施方案**：基于 **ToolLLM** 框架强化模型的外部工具调用能力。

1. **集成科研API库**：利用ToolBench数据集及定制化拓展，赋予模型调用arXiv检索、化学分子式渲染（如RDKit）、Python沙盒代码执行等核心科研工具的API权限。  
2. **引入DFSDT算法抗错训练**：在训练阶段强制模型学习深度优先搜索决策树（DFSDT）算法。在科研检索中极易遇到API查询无结果或超时的情况，通过DFSDT，模型在遇到死胡同时不会直接崩溃，而是学会评估多条推理轨迹，自主决定状态回溯（Backtrack），从而极大提升复杂检索任务的鲁棒性。  
3. **离线边缘工具路由**：如果科研环境对算力与内存的限制达到了极致（例如便携式现场采样设备或传感器），可部署 Cactus Compute 开发的 **Needle 2 (45M)** 作为专门的离线工具调用路由层，它在仅消耗约 28MB 会话内存的极低门槛下，就能根据自然语言输入，高鲁棒性地输出外部硬件或微控制器所需的结构化 API JSON 指令。

### **阶段五：实施WARP策略与枢轴偏好对齐（DPO），攻克深度调研报告**

科研综述或调研报告的撰写是一个动态认知与推演的过程，僵化的长文本生成往往导致逻辑发散或自相矛盾。  
**实施方案**：引入高度专精的后训练（Post-Training）对齐策略。

1. **枢轴Token定向DPO优化**：在模型生成的科研报告序列中，识别导致逻辑滑坡或公式错误的“枢轴Token”（Pivotal Tokens）。基于这些关键节点的概率分布，构建高质量的接受/拒绝偏好对，使用直接偏好优化（DPO）算法微调模型，消除学术推理过程中的幻觉。  
2. **WARP框架内化**：严格贯彻AgentCPM-Report中所展示的“写作即推理”（Writing As Reasoning Policy, WARP）策略。打破“先检索后写作”的线性模型，训练智能体在长文本生成中不断循环执行“基于证据的起草”与“驱动推理的深化”两个步骤。模型在撰写过程中动态修正大纲，如果在某一步发现逻辑断层，会主动触发ToolLLM进行二次文献检索，从而在端侧小参数规模下产出万字级别的专业深度调研报告。

### **阶段六：本地物理隔离部署与跨模态智能体协同（IoA）**

科研数据往往涉及极高的机密性，科研机构排斥云端大模型，迫切需要在本地运行SLM。  
**实施方案**：建立安全、可扩展的分布式科研智能体工作流。

1. **敏捷本地隔离部署**：结合AgentDock沙盒环境，利用Docker-compose在一台普通实验室工作站上实现全套服务（包含基座模型推理、UltraRAG检索模块、Milvus向量数据库）的一键式完全离线部署，确保核心机密数据“不出域”。  
2. **接入IoA（智能体互联网）构建群体智能**：当面临极端复杂的跨学科任务时，通过部署面壁智能与清华大学推出的 **InterSAGE** 协议，赋予科研智能体网络身份与能力发现机制。该科研SLM可通过类似即时通讯（IM）的架构，向同一内网下部署的其他专用智能体（如数据清洗Agent、制图Agent）发送协作请求与任务委派，通过结构化的对话流组织群体智慧，从而突破小模型自身的算力边界，实现前所未有的本地化 AI 辅助科研生产力突变。

#### **Works cited**

> 1. [https://www.turingpost.com/p/slmslist](https://www.turingpost.com/p/slmslist)

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADAAAAAhCAYAAACfiCi5AAACa0lEQVR4Xu2WTUjVQRTFT/RBkhSRqJHgpoXQoqAUQguCiITcJBJoGGRQC4NIMgqCNi7aRgRZFOWmIpAWkbrJlUqbWqhFEGgEUWJBUGDSxzne//DG8b2njxT+iznwg/funTdv7sy9dwaIioqKioqKiloRbSBN5CbpIFvnu9OrVeQImSa3yV7STb6SGm9catVIfpBjsGCkEjJGRsimxJZK7SBT5DFZ69mLySAssN2ePVXSbt8gf8jhwFdG3iPlAWwjEwn67Kua/CTfyc7AlxodhO3+E7Im8J0mf5HyGrgIW+T5wK5gFJR8ZwJfoVKabklwDWJZtJ48g53AocC3j8yQAVgxH4ClkurhFKzVPoClnmrnKrkL61qqKdcM9pA3sE3oIq/JrsRXS4Zhc6jW9pNr5AMZJ83JuJyqIB9hu6ye7/50O3lHRkllYpM0fpLcQWbsfVgHUyeTVDca42qmjXzzviuIPlKUfNeJdMIC1/z15IrnzyuX/8/JPfKKPIJdZtfJxszQOelW1uKOejYFMAg7JUndSjvoupYWuA4WsLpaK2wO/4aXTy28n9xCZq5FdRmZ/Hd5qomVWtnkAmjwbIsFUA5b3EtygpzFwgCkKtjGtQf2nHL5P0vqAl8uFRqA/qMX8y/IFtgcSjXXtrV550gP+YQlPl3y9f9cKjQA9xTRSUtaqIpUc6ioNY9ser4oC1aTS7DTUjbklXZdu5+t/2eT6uULLOX0wLtAHpJf5DcZIseDMVqkbJ9hL1stXin0lrwgJ73xqsNS8tT7vebX6zir9GMNVJdYaSl9tDj/nfXf0nFtxjJfLFFRUVFL0j/nh4cGNeAqzgAAAABJRU5ErkJggg==>

[image2]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAkAAAAiCAYAAACN+vPlAAAAw0lEQVR4XmNgGAVYQDEQfwRiU3QJGBAE4tNA/BaINdHk4AAkAVIAUgjSgBUEAfF/IJ6ELiENxCFQvJEBoqgPyncAYhaQogwgfgTEz6AKQBjEBoktAGIOkCIYgLnnMBDzIksgg2gGHO5BBiBJkCKQ47ECkPEga/CGD1GKiHI0tkAEaUxngIYTCMAcDfIhCLAC8SwGNE+AFP0GYhsgZgTiBCCezABRDAe6DJAQPg7E54B4HhBzIyuAAZAucSAWQJcYBfQGACP2KI4BmnTtAAAAAElFTkSuQmCC>

[image3]: <data:image/png;base64,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>

[image4]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAHAAAAAhCAYAAAABMi8ZAAAEr0lEQVR4Xu2ZW+jlUxTHv0KRWyJ3GdIUJTEuD+65hPKCGmXywAN5oJRhXjSTxINbPBBqXJLckweXPPyNIpcXD1IuhURoEiF31qd1trN/+/z2+e19zhjnn/2pbzPnt9b/tH977b3W2vtIjUaj0Wg0StnBtIdpm9TQWHy2N200vW3aPbE16mADMIfbpoZ/kzNNf5g+Ne2b2BaJnUy3mL41/WX6ynStacfYqZATTG/Iv4d3f9V0eMejS6k/8/eJ6TXTBtOFGR0/8p+b3Uyb5APbbDqsa14YGOcr8iyxYvTsUNMHphdG9lIuMP1oulq+U9A608+m8yO/QI3/kabv5fM5TbeFP5iXK0zfmf6UD3JV17wwXGf6zXRa8pzPPMdewoGmD03PyktHgP/zDBs+gVr/c0y/mz43fdajX01vyfuNuWEFv2u6TL5qCOIZHY/FYE/Te6aPTXsnNlIWqb+0fl8u3wHXpAb5IsB2cfRsFv+bos8xYb6PSw2zQLG903SjaX/5JDCY82KnQviulfKcf5fpLHVXa46jVVZzT5TvsiXTzl2TdpHXm59Mxya2lO1MTyn/nqRDbI/K36nWH3j/NK0C435JnvG2CCfJi/JeGq9wBsMKqoFA3SPfvV/LUwdF/ht5czTtWMIKLknZrHDGtqTJAPKZ59j7Ji6GHcpOzQWEZ9hYECyMWv8czMFa070qW9iD0LU9bVo9+hxPQmktCVwk78gOjp4xSCadbvEZ0wGRLUCxpylh8QwRUtWSpgdwaOwh3Q4FBB98a/1z0HG+o3HzNTdrTM9p3H7TnjOZDOaB4FQAKYaUQW7vY1fTQ/Ld+bC8wJ8u/xsaJ7q7EpZzAMP41ncfzw67gdSZpi4mmsHwbykMbr18AeQgfbACw1EFfSQPJrYSlnMASet099TxuWHCaFoeMe2n8SDRk/LBsBOnBeS/YLkGkAz3ovq755k4yvSl+s8oPyg/SUPsY7pffsbhO9jhJ2t4h10qr4VD0MLnxhYHsK/Vj2ESmcyhgIQJr/VPIcux+5Y0Oe5qaCweV76NDZOUW005DjK9Lz+ccp6kqblPfkuRa2CA+nmH6ZDU0APptiSA+E0jrvXTAhKyUK1/SjhD1pSlLOTilzU5AYEwGHZnbtL7uEHeIqe7jQbmbvn57Hp1X5CrKK6lHpQHcgiu9zarf6WHNIe95BqQBiq3W0OqxidQ6x9gPjgfbpEActbjPMO5LEcIYM11GkG5VdNvQE6RNy2cDd+Ur1bSNXeY7N4SQi3pC1IILtda/CQWYDdSGm5W9+x1trwr7pt0njHOU6Nntf6BODPMFEBWAPdtBIP0xpmMTih+SeDlWNWkPm47EMEk6EM/ifBd52p4F/E9XNHxwqRXdh87tAYWH5O1TuPdHpqyX+QXE4E49fHeR0Q2FsPz8t28InrOUegL0xPqBrzWPxDXz5qj2T+ElckXxEpTQah9fRqqKVsTgnWlPCVzpuSi4DF5rb1kZI+h/mDbqMmfmygRr8tT71Uj0dwRdBqylFp/iG+3hrrj/xVklTWm2+WZoq95KIGsQHbaIK/hKzW5CGJq/eEYedNYm20ajUaj0Wg0Go1Go9FoNBpbjb8BoCBw9NMYkgQAAAAASUVORK5CYII=>

[image5]: <data:image/png;base64,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>