# Chapter 26: The state of open models — Put the architectures of the course in one tree

**English** · [中文](README.zh.md)

> **Goal**: After this chapter, you can take the `config.json` of any open flagship model and read it layer by layer: which kinds of attention it uses, how its FFN is sparse, and if it has MTP. You can calculate its total parameters, its active parameters, and its KV cache at 128K context. On the evolution tree of this chapter, you can show the chapter that each of its parts comes from. You can tell which parts are consensus and which parts still diverge. You can also explain why our main-line model uses only the dense consensus block, and which addition is the best for its "next version".

📺 **Video**: Not published yet. To render it on your computer, run `bash chapters/26-open-model-panorama/video/build.sh`.
🧪 **Self-check**: After the chapter, type `/ch26-panorama` in Claude Code.

---

In the last chapter, we finished MTP and speculative decoding. We have now taken apart, put together, and tested on the CPU all five parts of Part 5: the KV cache ledger, local and sparse attention, hybrid linear attention, MoE, and MTP. This chapter asks: **When we put these parts back into real models, what do the strongest open models look like now (September 2026)? Which methods does everyone use, and which methods still go in different directions?**

First, we find out which models are the newest. Next, we draw an evolution tree that grows from GPT-2. Then we dissect four representative models and put them in one table with our main-line model and with Qwen3.5-0.8B, the model that we compare against. At the end, we look back at the whole course: each step from `y = ax + b` to here is on this tree.

The code of this chapter (all of it runs on the CPU, in a few seconds to 20 s):

```bash
uv run python chapters/26-open-model-panorama/code/01_panorama.py        # dissection + large table + adoption matrix + ledger for the "next version" of the main line
uv run python chapters/26-open-model-panorama/code/02_evolution_tree.py  # architecture evolution tree (the video also uses it)
uv run python chapters/26-open-model-panorama/code/03_meta_params.py     # count parameters with the official HF model classes, without the weights
```

All model numbers come from [`code/models.json`](code/models.json). We **copied each field one by one** from a source of the model (read on 2026-09-26). The source is the `config.json` in the Hugging Face repository of the model, the model card, or the technical report. Each model has its source links. If we did not read a value, we did not write it.

## 1. First, find out which models are the newest

The list of "state-of-the-art" models changes fast. If you write it from memory, it will be wrong. This chapter uses this method. On Hugging Face, list the repositories of some leading organizations by creation time: `deepseek-ai`, `Qwen`, `moonshotai`, `openai`, `zai-org`, `MiniMaxAI`. Then find the newest flagship of each organization. The result on 2026-09-26:

| Family | Flagship in this chapter | Other models of the same generation | Role in this chapter |
|---|---|---|---|
| DeepSeek | **DeepSeek-V4-Pro** (1.6T total parameters / 49B active, MIT) | V4-Flash (284B / 13B), two updates V4-Pro-0813 and V4-Flash-0731, V4.1-Flash with vision | Dissect |
| Qwen | **Qwen3.8-2.4T-A95B** (2.4T / 95B) | Qwen3.8-27B (dense), Qwen3.8-Flash-Next (`model_type: qwen4_exp`, experimental structure) | Dissect |
| Kimi | **Kimi-K3** (2.8T / 104B) | Kimi-K2.6, K2.7-Code | Dissect |
| OpenAI | **gpt-oss-120b** (117B / 5.1B, Apache-2.0) | gpt-oss-20b; no new generation of open weights as of 2026-09 | Dissect (the most "conservative" one) |
| Zhipu GLM | **GLM-5.3** (same base model as GLM-5.2) | GLM-5.3-Flash | Comparison in the same generation |
| MiniMax | **MiniMax-M3** (about 428B / about 23B) | M2.7 and others | Comparison in the same generation |

We dissect four of them. DeepSeek, Qwen, and Kimi represent three different paths to long context in 2026. gpt-oss represents "use only the most stable parts". GLM-5.3 and MiniMax-M3 go into the comparison table, where we count "how many families use it".

## 2. One tree: the architecture evolution of the whole course

Run `02_evolution_tree.py`. It prints the architecture evolution tree of the whole course. We did not write the "used by" part by hand. The program reads it from the config fields of each model in `models.json`:

```
GPT-2  [start, Ch. 8]
└─ Modern dense block (Llama style)  [start, Ch. 9]
   ├─ Pre-Norm + RMSNorm  [consensus, Ch. 6, 9] | used by: all 6 flagships + the two small Qwen models
   ├─ RoPE (+ YaRN to extend)  [consensus, Ch. 9, 15] | used by: the 5 flagships other than Kimi-K3
   ├─ SwiGLU  [consensus, Ch. 9] | used by: all 6 flagships
   ├─ GQA  [consensus, Ch. 10] | used by: Qwen3.8, gpt-oss, MiniMax-M3
   │  └─ MLA: K/V compressed into a latent vector  [consensus, Ch. 21] | used by: Kimi-K3, GLM-5.3
   │     └─ CSA/HCA: shared KV compressed along the tokens  [frontier note, this chapter] | used by: DeepSeek-V4-Pro
   ├─ QK-Norm  [consensus, Ch. 9] | used by: DeepSeek-V4-Pro, Qwen3.8, MiniMax-M3
   ├─ Tied input and output embedding (small models)  [consensus, Ch. 9] | used by: Qwen3-0.6B, Qwen3.5-0.8B
   ├─ MoE: fine-grained + shared experts  [consensus, Ch. 24] | used by: all 6 flagships
   │  ├─ Auxiliary-loss-free balancing  [consensus, Ch. 24] | used by: DeepSeek-V4-Pro, Kimi-K3, GLM-5.3, MiniMax-M3
   │  └─ Latent MoE (experts compute in a latent space)  [frontier note, this chapter] | used by: Kimi-K3
   ├─ Sliding window / local-global interleaving  [consensus, Ch. 22] | used by: DeepSeek-V4-Pro, gpt-oss-120b
   │  └─ Sparse attention (select top-k by content)  [new consensus (reached rule A only in 2026), Ch. 22, this chapter] | used by: DeepSeek-V4-Pro, GLM-5.3, MiniMax-M3
   ├─ Hybrid linear attention (about 3:1)  [consensus, Ch. 23] | used by: Qwen3.8, Kimi-K3, Qwen3.5-0.8B
   │  └─ KDA (per-channel gate)  [frontier note, Ch. 23] | used by: Kimi-K3
   ├─ MTP (predict one more token)  [consensus, Ch. 25] | used by: DeepSeek-V4-Pro, Qwen3.8, GLM-5.3, MiniMax-M3, Qwen3.5-0.8B
   └─ Residual-stream changes: mHC / AttnRes  [frontier note, this chapter] | used by: DeepSeek-V4-Pro, Kimi-K3
```

(The output uses the full model names. Here we shorten them for the layout.)

How to read the tree:

- **The root of the tree is GPT-2 (2019).** The model of Chapter 8 has the form of GPT-2: multi-head attention, learned absolute positions, an MLP with GELU, and LayerNorm. Its KV cache is 36 KiB per token, for a maximum of 1024 tokens.
- **The trunk is the modern dense block of Chapter 9**: Pre-Norm + RMSNorm, RoPE, SwiGLU, GQA, QK-Norm, tied embedding in small models, and no bias. All flagships today still grow on this trunk: all 6 families use RMSNorm and a gated FFN, and 5 families use RoPE. **Our main-line model is this trunk itself.** It has the same shape as Qwen3-0.6B.
- **Six branches.** Each branch answers the same question: "How do we make the model larger, longer, and cheaper?"
  1. **The feed-forward layer becomes sparse** (Chapter 24): MoE. All 6 flagships use it, and 5 of them have shared experts.
  2. **Store less for each position** (Chapters 10, 21): GQA → MLA → the compressed attention of DeepSeek-V4.
  3. **Look at only some positions** (Chapter 22): a sliding window selects by position, and sparse attention selects by content.
  4. **Do not store K/V at all** (Chapter 23): a hybrid of linear attention layers and a few full-attention layers.
  5. **A denser training objective** (Chapter 25): MTP, which can also make the drafts for speculative decoding.
  6. **The residual stream itself** (new in 2026): mHC from DeepSeek and AttnRes from Kimi. These are still at the frontier.

## 3. Dissect four models

Part 1 of `01_panorama.py` draws the layers of each model as one line of letters (G full attention, s sliding window, l linear attention, M MLA, c CSA, h HCA, g GQA+block-sparse):

```
DeepSeek-V4-Pro         61 layers: 31 HCA + 30 CSA
                           hhchchchchchchchchchchchchchchchchchchchchchchchchchchchchchc
Qwen3.8-2.4T-A95B       92 layers: 69 GDN + 23 GQA
                           lllGlllGlllG... (23 groups)
Kimi-K3                 93 layers: 69 KDA + 24 MLA
                           lllMlllMlllM...lllMM
gpt-oss-120b            36 layers: 18 SWA + 18 GQA
                           sGsGsGsGsGsGsGsGsGsGsGsGsGsGsGsGsGsG
Main-line model         28 layers: 28 GQA
```

All four want to "make long context cheap", but the four families give four different arrangements of layers.

### 3.1 DeepSeek-V4-Pro: it left MLA, its own invention

DeepSeek introduced MLA in V2 (Chapter 21) and used it in V3 and V3.2. In V4, it **replaced MLA**. From Section 2.3 of the V4 technical report and the `config.json`:

- **MQA with shared K=V**: `num_key_value_heads: 1`, `head_dim: 512`. Each position stores only one 512-dimensional vector, which is both the key and the value. All 128 query heads read it.
- **Two kinds of compressed attention that alternate** (`compress_ratios`): The first two layers are **HCA** (Heavily Compressed Attention). HCA compresses the KV of each 128 tokens into 1 entry and then does dense attention over all compressed entries. After these two layers, CSA and HCA alternate. **CSA** (Compressed Sparse Attention) compresses each 4 tokens into 1 entry. Then the "lightning indexer" of V3.2 selects the top 1024 compressed entries (`index_topk: 1024`) for attention. This is the "select by content" sparse attention of Chapter 22, but here it selects compressed blocks. The count is 30 CSA layers and 31 HCA layers.
- **Each layer also has a sliding-window branch of 128 tokens** (`sliding_window: 128`). It fills a gap: inside a compressed block, the model cannot see the most recent tokens. There is also an attention sink. RMSNorm on the queries and on the KV entries has the same effect as QK-Norm. RoPE rotates only the last 64 dimensions.
- **FFN**: 6 of 384 routed experts + 1 shared expert. The first 3 layers use hash routing (`num_hash_layers: 3`). The score function is `sqrtsoftplus`, with auxiliary-loss-free balancing (`topk_method: noaux_tc`). There is 1 MTP layer.
- **Residual stream**: mHC (`hc_mult: 4`). It widens the residual stream to 4 streams. A mixing matrix connects the layers. mHC constrains this matrix to be "doubly stochastic" (Section 2.2 of the report).
- **Precision and optimizer**: DeepSeek releases the routed experts in FP4, and most other weights in FP8. Pretraining uses the Muon optimizer (model card).

How do we count the KV cache? A CSA layer stores 512 / 4 = 128 numbers per original token. An HCA layer stores 512 / 128 = 4 numbers. The sliding-window branch stores at most 128 positions. The function `dsv4_layout` in `01_panorama.py` builds this ledger directly with `KVLayout` / `LayerSpec` from the production code `zero.tools.kv_cache_calc`:

```python
for r in c["compress_ratios"][: c["num_hidden_layers"]]:
    if r:                                   # compressed layer: head_dim / r numbers per token
        layers.append(LayerSpec("full", hd // r))
        if with_indexer and r == 4:         # the indexer of a CSA layer also stores compressed keys
            layers.append(LayerSpec("full", c["index_head_dim"] // r))
    layers.append(LayerSpec("sliding", hd, window=win))   # sliding branch, window 128 (K=V, stored once)
```

Result: 7.7 KiB per token and 998.62 MiB at 128K context. V3.2 needs 68.6 KiB and 8.58 GiB. The report says that at 1M context, the KV cache of V4-Pro is only 10% of V3.2. With the same precision (BF16) and with the indexer keys on both sides, our ledger gives **11.5%** (V4-Pro 9.62 GiB vs V3.2 83.88 GiB). This is the same order of magnitude. The small difference has two causes: the report stores the KV entries in FP8 except the RoPE dimensions, and the compression rounds exactly.

### 3.2 Qwen3.8-2.4T-A95B: the 0.8B structure scaled up to 2.4T

This is the largest Qwen model of this generation. Its `architectures` field is `Qwen3_5MoeForCausalLM`: **the same architecture class as Qwen3.5**. The model card also says "Built on the architectural foundation of Qwen3.5". It has the same design as Qwen3.5-0.8B, the model that we compare against, but it is about 3000 times larger:

| | Qwen3.5-0.8B | Qwen3.8-2.4T-A95B |
|---|---|---|
| Layers | 24 = 6 × (3 GDN + 1 full attention) | 92 = 23 × (3 GDN + 1 full attention) |
| Full attention | GQA 8 Q / 2 KV, head_dim 256, output gate | GQA 64 Q / 4 KV, head_dim 256, output gate |
| Linear layers (Gated DeltaNet) | 16 key heads, 16 value heads | 16 key heads, 128 value heads |
| FFN | dense SwiGLU | 10 of 512 experts + 1 shared; auxiliary loss 0.001 |
| RoPE | rotates only 25% of the dimensions, θ = 10⁷ | same |
| MTP | 1 layer | 1 layer (model card: "trained with multiple steps") |
| Vocabulary / tied embedding | 248,320 / yes | 248,320 / no |
| Context | 262,144 | 262,144 (model card: can extend to 1,010,000) |

With hybrid linear attention, only 23 of the 92 layers need a KV cache that grows with length: 92 KiB per token and 11.50 GiB at 128K. The linear layers also have a fixed state of about 568 MiB (estimate). Note that its load balancing uses an **auxiliary loss** (`router_aux_loss_coef: 0.001`), not a DeepSeek-style bias. Chapter 24 said that both methods work in large models.

### 3.3 Kimi-K3: hybrid linear attention + MLA + a sparser MoE

Of the 93 layers of K3, layers 4, 8, ..., 92 and the last layer 93 are **MLA with a gate** (24 layers). The other 69 layers are **KDA** (Kimi Delta Attention, Chapter 23: Gated DeltaNet with a per-channel gate). This is an example of "the three paths can be combined" from Chapter 21. MLA makes each full-attention layer store only 576 numbers per position, and the hybrid makes only 1/4 of the layers store K/V. The result is 27 KiB per token and 3.38 GiB at 128K.

Some other points are important:

- **The full-attention layers have no position encoding** (`mla_use_nope: true`). In the Kimi Linear report, the MLA layers use NoPE, and the KDA layers carry all position information. K3 is the only one of the 6 flagships that does not use RoPE.
- **A sparser MoE**: 16 of 896 experts + 2 shared experts. The model card says "Latent MoE Dimension 3584": the routed experts do not compute in the hidden size of 7168. They first project the input to a latent space of 3584 dimensions. `03_meta_params.py` counts this block of routed experts by hand. In the latent space, the count is 2.72T. Directly in 7168 dimensions, it would be 5.45T. The 2.8T total parameters on the model card agree only with the first number.
- **A new activation function**: `hidden_act: situ` (the model card calls it SiTU-GLU). It is still a kind of gated FFN.
- **Residual stream**: Attention Residuals (`attn_res_block_size: 12`).
- **No MTP** (`num_nextn_predict_layers: 0`), the same as K2.
- MXFP4 weights and quantization-aware training (model card).

### 3.4 gpt-oss-120b: only the most stable parts

OpenAI released gpt-oss in August 2025. The other five models in this chapter are newer. But OpenAI has not released a new generation of open weights since then, so gpt-oss is still the newest model of this family. It is almost a textbook example of "consensus block + MoE + sliding window":

- 36 layers. Sliding window (128 tokens) and full attention alternate 1:1. GQA 64 Q / 8 KV, head_dim 64. Each head has one learnable attention sink.
- 4 of 128 experts, no shared expert, with an auxiliary loss (`router_aux_loss_coef: 0.9`).
- RoPE θ = 150,000. YaRN extends 4K by a factor of 32 to 128K.
- SwiGLU with a clamp (`swiglu_limit: 7.0`). The MoE weights are released in MXFP4, so the model fits on one 80 GB GPU.
- No MLA, no linear attention, no MTP.

The KV cache is 36 KiB per token (only the 18 full-attention layers grow) and 4.50 GiB at 128K.

### 3.5 Two other families of the same generation

- **GLM-5.3**: All 78 layers are MLA (512 + 64) + DeepSeek-style sparse attention (`index_topk: 2048`). IndexShare, from GLM-5.2, lets each 4 sparse-attention layers share one indexer (`indexer_types` in the config: 21 layers `full`, 57 layers `shared`). 8 of 256 experts + 1 shared, auxiliary-loss-free balancing, 1 MTP layer. 87.8 KiB per token.
- **MiniMax-M3**: MiniMax-M2 uses full attention (Chapter 23). M3 changes to GQA (64 Q / 4 KV) + **block-sparse attention MSA**. The first 3 layers use normal attention. The other 57 layers split the history into blocks of 128 tokens, and each query selects 16 blocks (`sparse_block_size: 128`, `sparse_topk_blocks: 16`). The model card says that at 1M context, prefill is 9 times faster and decode is 15 times faster (compared with M2). Block-sparse attention **does not make the KV cache smaller**: it is still 120 KiB per token and 15.00 GiB at 128K. This is the largest of the flagships in this chapter.

## 4. One table

This is Part 2 of `01_panorama.py`. All KV cache values use BF16 and batch 1, and they count only the part that grows with length. "KV per token" is the growth for each additional token at long context (the sliding-window layers are full long before, so they do not count):

| Model | Total | Active | Layers | Layer composition | Vocabulary | Context | Tied emb | KV per token | 32K KV | 128K KV | Linear-layer state |
|---|---:|---:|---:|---|---:|---:|---|---:|---:|---:|---:|
| GPT-2 | 0.12B | dense | 12 | 12 MHA | 50,257 | 1K | yes | 36.0 KiB | — | — | — |
| Qwen3-0.6B | 0.60B | dense | 28 | 28 GQA | 151,936 | 40K | yes | 112.0 KiB | 3.50 GiB | — | — |
| **Main-line model (this course)** | **689.5M** | **dense** | **28** | **28 GQA** | **65,536** | **32K** | **yes** | **112.0 KiB** | **3.50 GiB** | — | — |
| Qwen3.5-0.8B (comparison target) | 0.75B | dense | 24 | 18 GDN + 6 GQA | 248,320 | 256K | yes | 12.0 KiB | 384.00 MiB | 1.50 GiB | 19.27 MiB |
| gpt-oss-120b | 117B | 5.1B | 36 | 18 SWA + 18 GQA | 201,088 | 128K | no | 36.0 KiB | 1.13 GiB | 4.50 GiB | — |
| DeepSeek-V3.2 | 671B | 37B | 61 | 61 MLA+DSA | 129,280 | 160K | no | 68.6 KiB | 2.14 GiB | 8.58 GiB | — |
| DeepSeek-V4-Pro | 1.6T | 49B | 61 | 31 HCA + 30 CSA | 129,280 | 1M | no | 7.7 KiB | 255.38 MiB | 998.62 MiB | — |
| Qwen3.8-2.4T-A95B | 2.4T | 95B | 92 | 69 GDN + 23 GQA | 248,320 | 256K | no | 92.0 KiB | 2.88 GiB | 11.50 GiB | 568.17 MiB |
| Kimi-K3 | 2.8T | 104B | 93 | 69 KDA + 24 MLA | 163,840 | 1M | no | 27.0 KiB | 864.00 MiB | 3.38 GiB | 221.55 MiB |
| GLM-5.3 | 744B | 40B | 78 | 78 MLA+DSA | 154,880 | 1M | no | 87.8 KiB | 2.74 GiB | 10.97 GiB | — |
| MiniMax-M3 | ~428B | ~23B | 60 | 3 GQA + 57 GQA+block-sparse | 200,064 | 1M | no | 120.0 KiB | 3.75 GiB | 15.00 GiB | — |

Notes: The total / active parameters of the flagships come from the model cards. `03_meta_params.py` counts the total parameters of the small dense models. The GLM-5.3 model card does not give the parameter count. Its config has exactly the same shapes as GLM-5, so we use 744B / 40B from the GLM-5 model card. "—" means beyond the context of the model (for the main-line model, we use the 32K of the long-context stage). The linear-layer state is an estimate (with `mamba_ssm_dtype: float32`; the format depends on the implementation).

**How we check the parameter counts.** `03_meta_params.py` gives each config to the **official model class** of its architecture in Hugging Face transformers (`DeepseekV4ForCausalLM`, `Qwen3_5MoeForCausalLM`, `GlmMoeDsaForCausalLM`, ...). It builds the model on the `meta` device: only shapes, no memory allocation, so a 2.4T model builds in one second. Then it counts the parameters:

```python
with torch.device("meta"):
    model = AutoModelForCausalLM.from_config(cfg)
total = sum(p.numel() for p in model.parameters())
experts = sum(p.numel() for n, p in model.named_parameters() if ".experts." in n)
active = total - experts * (1 - k / E)      # each token uses only k of the E routed experts
```

| Model | Counted total | Active (all) | Without input embedding | Without the whole vocabulary | Model card |
|---|---:|---:|---:|---:|---|
| gpt-oss-120b | 116.829B | 5.71B | **5.13B** | 4.55B | 117B / 5.1B |
| DeepSeek-V3.2 | 671.878B | 38.40B | 37.48B | 36.55B | 671B / 37B |
| DeepSeek-V4-Pro | 1572.997B | **49.78B** | 48.85B | 47.93B | 1.6T / 49B |
| Qwen3.8-2.4T-A95B | 2419.805B | **95.29B** | 93.25B | 91.22B | 2.4T / 95B |
| GLM-5.3 | 743.377B | 41.25B | **40.30B** | 39.35B | 744B / 40B |
| MiniMax-M3 | 426.175B | 25.96B | 24.73B | **23.50B** | ~428B / ~23B |
| Main-line model | 0.6895B (zero formula = HF `Qwen3ForCausalLM`) | dense | | | |

All total parameters agree with the model cards. (The HF classes of DeepSeek do not include the MTP layer, so their counts are a little smaller.) An interesting point: **there is no common definition of active parameters.** The 5.1B of OpenAI does not count the input embedding. The about 23B of MiniMax does not count the output head either. The 95B of Qwen and the 49B of DeepSeek count all embeddings. (Bold marks the definition that is nearest to the model card.) The "active parameters" of two families can differ by 10–20% only because of the definition.

Some points to stop and look at:

- **The KV cache is now almost independent of the parameter count.** The 2.8T Kimi-K3 at 128K needs a smaller KV cache (3.38 GiB) than the 0.7B main-line model at 32K (3.50 GiB). The 1.6T DeepSeek-V4-Pro needs less than 1 GiB. The KV cache depends on **how many layers store K/V, and how much each layer stores per position**. This is the whole content of Part 5.
- **None of the six flagships uses normal full attention in every layer.** In gpt-oss, half of the layers use a sliding window. In Qwen and Kimi, 3/4 of the layers are linear. DeepSeek, and GLM with MLA, compress the KV. MiniMax uses sparse attention, which saves compute but not KV.
- **On the small-model side**, Qwen3.5-0.8B uses a 3:1 hybrid to make the KV cache at 32K only 384 MiB. This is one ninth of the main-line model (3.50 GiB). It is the strongest reference for the "next version" of the main-line model (Section 6).
- **Vocabularies become larger**: the flagships use between 130K and 250K. Qwen3.5-0.8B uses a vocabulary of 248K with tied embedding, and the vocabulary alone is one third of its 750M parameters. The 65,536 of the main-line model is the compromise for small models from Chapter 13.

## 5. What is consensus, and what still diverges

Part 3 of `01_panorama.py` reads from the configs of the 6 flagships which techniques each one uses. Then it adds the other families that earlier chapters verified. It decides with rule A of GOAL.md 2.1 (at least 3 independent leading families):

```
Technique               DS-V4  Qwen3.8       K3  gpt-oss  GLM-5.3       M3 Uses  other adopters verified in earlier chapters → decision
Pre-Norm RMSNorm            ●        ●        ●        ●        ●        ●    6  ≥3 families: consensus
RoPE                        ●        ●        ·        ●        ●        ●    5  ≥3 families: consensus
YaRN                        ●        ·        ·        ●        ·        ·    2  + Kimi K2, Qwen3, SmolLM3 (Chapter 15) → ≥3 families: consensus
SwiGLU/GLU                  ●        ●        ●        ●        ●        ●    6  ≥3 families: consensus
QK-Norm                     ●        ●        ·        ·        ·        ●    3  ≥3 families: consensus
Sliding window              ●        ·        ·        ●        ·        ·    2  + Gemma 3, OLMo 3 (Chapter 22) → ≥3 families: consensus
Sparse attention            ●        ·        ·        ·        ●        ●    3  + Meituan LongCat-2.0 (Chapter 22) → ≥3 families: consensus
MoE                         ●        ●        ●        ●        ●        ●    6  ≥3 families: consensus
Shared experts              ●        ●        ●        ·        ●        ●    5  ≥3 families: consensus
Aux-free balancing          ●        ·        ●        ·        ●        ●    4  ≥3 families: consensus
SwiGLU clamp                ●        ·        ·        ●        ·        ●    3  ≥3 families: consensus
MTP                         ●        ●        ·        ·        ●        ●    4  ≥3 families: consensus
GQA                         ·        ●        ·        ●        ·        ●    3  ≥3 families: consensus
Hybrid linear attn          ·        ●        ●        ·        ·        ·    2  + NVIDIA Nemotron 3 (Chapter 23) → ≥3 families: consensus
MLA                         ·        ·        ●        ·        ●        ·    2  + Mistral Large 3 (Chapter 21) → ≥3 families: consensus
CSA/HCA compression, MQA (shared K=V), Hyper-connection mHC, Attention residuals, NoPE (full-attn): 1 family each → fewer than 3 → frontier note
```

(We removed some lines for the layout. The full output is in the run result.)

We split the results into three groups:

**1. A very stable trunk.** Pre-norm RMSNorm, the gated FFN (SwiGLU and its variants), and MoE are 6/6 in the 6 families. RoPE is 5/6 (the only exception, Kimi-K3, gives the position information to its linear layers). Shared experts are 5/6, MTP is 4/6, and auxiliary-loss-free balancing is 4/6. The consensus block of Chapter 9, plus MoE and MTP from Chapters 24 and 25, is the "default configuration" of a 2026 flagship.

**2. "Make long context cheap" has split into three schools. Each school passed the threshold of 3 families.**

- **Hybrid linear attention**: Qwen (Gated DeltaNet 3:1), Kimi (KDA about 3:1), NVIDIA Nemotron 3 (Mamba-2 hybrid, Chapter 23).
- **Sparse attention**: DeepSeek (DSA in V3.2 → CSA in V4), GLM (DSA + IndexShare), MiniMax (block-sparse MSA in M3). When we wrote Chapter 21, we had verified only two families, DeepSeek and GLM. In 2026, MiniMax-M3 joined (and Chapter 22 also verified Meituan LongCat-2.0). The 6 flagships of this chapter alone already give 3 families. Thus the tree marks it as "new consensus", the same decision as Chapter 22.

  But the methods of the families are different. DSA selects the top-k single tokens, CSA compresses first and then selects, and MSA selects blocks of 128 tokens. **The idea "select by content" is consensus, but the specific methods have not converged.** Also, no small model uses it.
- **Store less / compress**: MLA (Kimi-K3 and GLM-5.3, plus Mistral Large 3 from Chapter 21). Also the sliding window (gpt-oss and the sliding branch of DeepSeek-V4, plus Gemma 3 and OLMo 3 from Chapter 22).

Note the position of MLA: its inventor, DeepSeek, replaced it in V4 with compressed attention + MQA with shared K=V. By the count, MLA still satisfies rule A. But it is no longer a consensus that "more and more families use".

**3. Small tricks that pass the count but do not need their own section.** The SwiGLU clamp (10 in DeepSeek-V4, 7 in gpt-oss and MiniMax-M3) has exactly 3 families. It is a stability trick that prevents outlier activations.

There are also two observations outside the flagship configs. The first is **low-precision release**: MXFP4 in gpt-oss, FP4 experts in DeepSeek-V4, and MXFP4 quantization-aware training in Kimi-K3 (all on the model cards). The second is the **Muon optimizer**: the DeepSeek-V4 model card, MuonClip in the Kimi Linear report, and the GLM-5 report (see Chapter 21). These belong to the topics of Chapter 20 (quantization) and Chapter 12 (optimizer), so here we only record them.

**Still diverging** (only 1–2 families each; they go into "Frontier notes" at the end): the CSA/HCA compressed attention and the MQA with shared K=V of DeepSeek-V4, residual-stream changes (mHC, AttnRes), Latent MoE, the per-channel gate of KDA, NoPE in full-attention layers, and n-gram embedding tables.

**One more trend**: almost all models apply RoPE to only some of the dimensions. Qwen rotates 25% (`partial_rotary_factor: 0.25`), and MiniMax rotates 50%. The MLA / compressed attention of DeepSeek and GLM rotates only 64 dimensions. Only gpt-oss rotates all dimensions, and the full-attention layers of Kimi-K3 do not rotate at all. RoPE on all dimensions, as in Chapter 9, is still the standard method for small models (Qwen3-0.6B, the main-line model). But most large models now "keep some dimensions without position".

## 6. Why the main-line model uses only the dense consensus block, and what the next version adds

Put the main-line model back into the table. Among the MoE models with hundreds of billions or trillions of parameters, it looks "plain": 28 full-attention layers, GQA 16/8, QK-Norm, SwiGLU, RMSNorm, RoPE, and tied embedding, with the same shape as Qwen3-0.6B. This is a deliberate choice of GOAL.md 3.3. Each chapter of Part 5 calculated the reasons:

| Technique | The flagships use it. Why does the main line not use it? |
|---|---|
| MoE | MoE "trades GPU memory for compute": the GPU memory holds all parameters, and the saving is the compute per token. In the product lines of Chapter 24, Section 8, the models of a family below some tens of B are almost all dense, and the smallest MoE has more than 20B total parameters. If we cut 0.7B into tens of experts, the matrix of each expert is too small to keep the GPU busy. |
| MLA | Now it occurs only in MoE flagships with tens of billions of parameters or more. At inference, MLA never reconstructs K, so it is not compatible with QK-Norm (Chapter 21). It needs special kernels. |
| Sparse / compressed attention | It saves attention compute above 100K tokens. The target context of the main line is 32K. The methods have not converged. |
| Hybrid linear attention | **The best candidate to consider seriously**: Qwen3.5-0.8B uses it at our size. But it needs additional kernels, and the rollback in speculative decoding is more difficult (Chapter 25, guided question 6). The small experiment in Chapter 23 also showed that pure linear layers are worse at "recall" tasks, and tool calling must copy parameter names exactly. |
| MTP | It adds one block in training, and it can make drafts at inference. But the numbers of Chapter 25 show that its gain is mainly in inference speed. It does not change "how accurate the tool calls are". |

A more basic reason: **at this scale and context length, the architecture makes a much smaller difference than the data and the post-training.** The main-line model must win at tool calling. The result depends on the data mixture of Chapter 13 and on SFT, distillation, DPO, and GRPO in Chapters 16–19. Choose the most stable architecture, and keep the risk in the data and the training.

Then what does the "next version" try first (after the training in Step 2 gives a real baseline)? Part 5 of `01_panorama.py` uses the production calculator to keep the ledger for some variants of the main-line model (32K context, BF16):

| Option | KV cache at 32K | Other |
|---|---:|---|
| Now: 28 full-attention GQA layers | 3.50 GiB | — |
| 3:1 local-global (window 4096) | 1.20 GiB | The parts exist: `zero/arch/sliding_window.py` |
| 3:1 hybrid linear attention (Qwen3.5 style) | 896.00 MiB | Fixed state of the linear layers about 22.48 MiB; the parts exist: `HybridTransformer` in `zero/arch/linear_attention.py` |

With the consensus rules and our goal, I put the experiments of the "next version" in this order:

1. **1 MTP layer** (`zero/arch/mtp.py`). The cost is small, Chapter 25 already has the speculative decoding code, and it is useful for the long JSON outputs of tool calls.
2. **3:1 hybrid linear attention**. The KV cache becomes one quarter, on the same path as Qwen3.5-0.8B. But first compare it on an "exact copy" evaluation of tool calling.
3. **Local-global interleaving**, as the conservative alternative to item 2.

MoE, MLA, and sparse attention do not pay off at 0.7B and 32K, so they do not go into the next version. For each item, first do a comparison with the same compute on the ladder configuration of Chapter 12. Decide with the preregistration method of Chapter 11. Only then decide if the item goes into the main line.

## 7. Summary

- **The newest list** (2026-09): DeepSeek-V4-Pro, Qwen3.8-2.4T-A95B, Kimi-K3, gpt-oss-120b, GLM-5.3, MiniMax-M3. To find the "newest" models, look them up now. Do not use your memory.
- **Evolution tree**: GPT-2 → modern dense block (RMSNorm, RoPE, SwiGLU, GQA, QK-Norm) → six branches: MoE, KV compression (GQA → MLA → compressed attention), look at only a part (sliding window → sparse), hybrid linear attention, MTP, and residual-stream changes.
- **Consensus**: dense block + MoE (fine-grained + shared experts) + MTP is the default configuration of a 2026 flagship. Long context has split into three schools: hybrid linear, sparse, and store less. Each school passed the threshold of 3 families, but the specific methods still diverge.
- **Ledger**: the KV cache no longer depends on the parameter count. The 2.8T Kimi-K3 at 128K needs less than the 0.7B main-line model at 32K. Each family defines active parameters differently.
- **Main-line model**: it uses only the dense consensus block on purpose. For the "next version", the best candidates are MTP and 3:1 hybrid linear attention.

---

## Back to the start: the whole course is one path

We finished 26 chapters. Look back: each chapter grew from a problem of the previous chapter (GOAL.md 2.2):

| Part | The problems that we solved | The part of today's flagships |
|---|---|---|
| Part 1 (Chapters 1–6) | Straight line → neural network → backpropagation → classification → stable training | The training loop of each flagship: model, loss, gradient, update, the same four steps. RMSNorm, residual connections, and AdamW / warmup have not changed. |
| Part 2 (Chapters 7–10) | Tokenization, attention, modern Transformer, inference | The trunk of the evolution tree: all 6 flagships grow on the dense block of Chapter 9. The KV cache of Chapter 10 is the start of Part 5. |
| Part 3 (Chapters 11–15) | Evaluation, scaling laws, data, pretraining engineering, long context | Each score table, each "32T tokens", and each YaRN factor on the model cards |
| Part 4 (Chapters 16–20) | SFT, distillation, preference alignment, reinforcement learning, release | "Expert SFT + GRPO → on-policy distillation" of DeepSeek-V4, the thinking modes of each family, FP4 / MXFP4 releases |
| Part 5 (Chapters 21–26) | KV cache, local / sparse, linear hybrid, MoE, MTP, overview | The whole tree of this chapter |

Chapter 1 said: "The structure of training a large model is the same as the structure of this chapter. The only difference is the number of parameters: here we have 2, and a large model has billions." Now we can add the numbers. Kimi-K3 has 2.8 trillion parameters, and each token uses 104 billion of them. But each step still does the same work: calculate the loss, calculate the gradient, and subtract the learning rate times the gradient.

**Next is Step 2.** We finished the course, and the production code of the main-line model runs end to end on the CPU with a tiny configuration. When GPUs are available, train the 689.5M dense model for real with the gate process of GOAL.md Sections 3 and 10. First, do one GPU verification run for ≤ $50. Then run the ladder, set the hyperparameters, and do pretraining, mid-training, long context, SFT, distillation, DPO, and GRPO. Finally, compare tool calling with Qwen3.5-0.8B on the preregistered exam.

Then we come back to this table and replace "To be added after GPU training" in the "main-line model" row with real numbers.

---

## From minimal code to production code

| Minimal code (`code/`) | Production code / industry reference implementation | What it adds, and why |
|---|---|---|
| `layout` / `kv_numbers` in `01_panorama.py` | `zero/tools/kv_cache_calc.py`: `layout_from_config`, `kv_cache_bytes`, `fixed_state_bytes`, `breakdown`; command line `uv run python -m zero.tools.kv_cache_calc <config.json> --seq 131072` | All KV numbers in this chapter, except DeepSeek-V4, call it directly. It can read the nested HF `text_config`, `layer_types`, the `linear_attn_config` of Kimi, and the MLA fields. It does not know the `compress_ratios` of DeepSeek-V4 yet, so this chapter builds that ledger itself with its `KVLayout` / `LayerSpec` (see the suggested improvement below). |
| `meta_count` in `03_meta_params.py` | `zero/tools/count_params.py` (main-line model, with the formula, no memory allocation); the official model classes in HF transformers (`DeepseekV4ForCausalLM`, `Qwen3_5MoeForCausalLM`, `KimiLinearForCausalLM`, `GptOssForCausalLM`, `GlmMoeDsaForCausalLM`, `MiniMaxM3VLForCausalLM`) | Main-line model: the zero formula and the HF `Qwen3ForCausalLM` with the same shapes on the meta device both give 0.6895B, exactly the same. The HF classes are the reference implementations of each architecture. To understand a new architecture, the fastest way is to read their `modeling_*.py` (for example, the comments of `DeepseekV4CSACache` explain the CSA cache clearly). |
| Each branch of the evolution tree | `zero/arch/`: `mla.py` (Chapter 21), `sliding_window.py` (Chapter 22), `linear_attention.py` (Chapter 23, `HybridTransformer`, parity check with HF `Qwen3_5GatedDeltaNet`), `moe.py` (Chapter 24), `mtp.py` and `speculative.py` (Chapter 25) | All are experimental modules that the main line does not use. They have the same interface as `zero.model`, and they can go directly into the `Transformer` of the main line. We verified their correctness on CUDA on an RTX 3090 (we fixed one bug in MoE and one in linear attention). We did not verify their performance on the GPU yet. See Sections 11 and 12 of [runs/2026-10-01-gpu0-check](../../runs/2026-10-01-gpu0-check/README.md). |
| The main-line model row | `configs/main/pretrain.toml`, `zero/model.py` | Dense consensus block. `tests/test_model_hf_parity.py` makes sure that the logits are the same as the logits of the HF Qwen3 implementation. |

**Parity check**: `uv run pytest tests/test_kv_cache_calc.py tests/test_config.py tests/test_arch_*.py`. On this machine, all 109 tests passed (about 44 s). The three scripts of this chapter call these production functions directly. They do not write a second set of formulas.

**Suggested improvements to `zero/`** (by the working agreement, we only suggest them here; we did not change the code): `zero/tools/kv_cache_calc.layout_from_config` can learn the `compress_ratios` of DeepSeek-V4 (4 → CSA, 128 → HCA, 0 → sliding window only; shared K=V, `head_dim` numbers per entry, plus the `sliding_window` branch; CSA also has the indexer keys of `index_head_dim`). The logic can copy `dsv4_layout` from this chapter. Also, the GPT-2 field names (`n_layer`, `n_head`, `n_embd`) can go into the aliases of `_get`.

---

## Frontier notes

> **Techniques that are not consensus. We only mention them here** (we found only 1–2 leading families for each)
>
> - **Compressed attention CSA / HCA** (DeepSeek-V4): it compresses the KV of each 4 or 128 tokens into one entry, and it adds a sliding branch with a window of 128. MQA with shared K=V, head_dim 512. The KV cache is only about one tenth of V3.2. Now only DeepSeek uses it. DeepSeek replaced MLA after many families had adopted MLA. This is the change in this chapter that is the most important to watch.
> - **Residual-stream changes**: mHC in DeepSeek-V4 and Attention Residuals in Kimi-K3 (`attn_res_block_size: 12`). mHC widens the residual stream to 4 streams and constrains the mixing matrix between layers to be doubly stochastic. The ideas of the two families are different. The config of the experimental Qwen3.8-Flash-Next (`model_type: qwen4_exp`) also has `hc_count: 4`, but it is not a main version.
> - **Latent MoE** (Kimi-K3): the routed experts compute in a latent space of 3584 dimensions. This is how 896 experts fit.
> - **NoPE in full-attention layers** (Kimi Linear / K3): the linear attention layers carry all position information.
> - **n-gram embedding tables**: the config of DeepSeek-V4.1-Flash has `engram_*` fields, and Qwen3.8-Flash-Next has `ngram_*` fields. They add "memory" to the model with a table lookup. In both families, they occur only in non-flagship or experimental versions.
> - **The per-channel gate of KDA**: see Chapter 23.
> - **Attention output gate** (`attn_output_gate` of Qwen, Gated MLA of Kimi-K3) and **attention sink** (gpt-oss, DeepSeek-V4): 2 families each.

---

## Adopters and sources

The adopters of each technique among the 6 flagship families of this chapter (the evidence is a config field, or the code / report given; the full details are in the output of Part 3 of `01_panorama.py`):

| Technique | Adopters among the 6 flagships of this chapter | Additional adopters from earlier chapters | Decision |
|---|---|---|---|
| Pre-Norm + RMSNorm | All 6 families (`rms_norm_eps`; MiniMax-M3 uses the Gemma style `1+w`) | — | Consensus |
| RoPE | DeepSeek-V4 (only 64 dimensions), Qwen3.8 (25%), gpt-oss (all dimensions, θ 150,000), GLM-5.3 (64 dimensions), MiniMax-M3 (50%) | Qwen3, Llama 3, and others (Chapter 9) | Consensus |
| YaRN | DeepSeek-V4 (×16, original length 65,536), gpt-oss (×32, original length 4,096) | Kimi K2, Qwen3, SmolLM3 (Chapter 15) | Consensus |
| SwiGLU / gated FFN | All 6 families (`silu`; MiniMax `swigluoai`; Kimi `situ`) | — | Consensus |
| GQA | Qwen3.8 (64/4), gpt-oss (64/8), MiniMax-M3 (64/4) | Qwen3, Llama 3, and others (Chapters 10, 21) | Consensus |
| QK-Norm | DeepSeek-V4 (report, Section 2.3.3), Qwen3.8 (HF `Qwen3_5MoeAttention.q_norm/k_norm`), MiniMax-M3 (`use_qk_norm: true`) | Gemma 3, OLMo 2 (Chapter 9) | Consensus |
| MLA | Kimi-K3, GLM-5.3 (`kv_lora_rank: 512`) | Mistral Large 3, DeepSeek-V3/V3.2 (Chapter 21) | Consensus (DeepSeek-V4 has replaced it) |
| Sliding window / local-global | gpt-oss (1:1, window 128), DeepSeek-V4 (sliding branch of 128 in each layer) | Gemma 3, OLMo 3 (Chapter 22) | Consensus |
| Sparse attention | DeepSeek-V4 (`index_topk: 1024`), GLM-5.3 (`index_topk: 2048` + IndexShare), MiniMax-M3 (MSA, `sparse_topk_blocks: 16`) | DeepSeek-V3.2 (Chapter 21), Meituan LongCat-2.0 (Chapter 22) | **New consensus in 2026; the methods have not converged** |
| Hybrid linear attention | Qwen3.8 (69 GDN : 23), Kimi-K3 (69 KDA : 24) | NVIDIA Nemotron 3 (Chapter 23), the whole Qwen3.5 series | Consensus |
| MoE (fine-grained) | All 6 families: 6 of 384, 10 of 512, 16 of 896, 4 of 128, 8 of 256, 4 of 128 | Chapter 24 | Consensus |
| Shared experts | DeepSeek-V4 (1), Qwen3.8 (1), Kimi-K3 (2), GLM-5.3 (1), MiniMax-M3 (1); gpt-oss does not use them | Chapter 24 | Consensus |
| Auxiliary-loss-free balancing | DeepSeek-V4, Kimi-K3, GLM-5.3 (`noaux_tc`), MiniMax-M3 (`use_routing_bias`); Qwen and gpt-oss use an auxiliary loss | Chapter 24 | Consensus |
| MTP | DeepSeek-V4, GLM-5.3 (`num_nextn_predict_layers: 1`), Qwen3.8 (`mtp_num_hidden_layers: 1`), MiniMax-M3 (`num_nextn_predict_layers: 1`, and also `num_mtp_modules: 7`); Kimi-K3 and gpt-oss have no MTP | Chapter 25 | Consensus |
| Tied input and output embedding | No flagship ties them; small models tie them: Qwen3-0.6B, Qwen3.5-0.8B | Chapter 9 | Consensus (small models only) |

**Model configurations and model cards** (read through Hugging Face on 2026-09-26):

- DeepSeek-V4-Pro: [config.json](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/config.json), [model card](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro), technical report [arXiv:2606.19348](https://arxiv.org/abs/2606.19348) (Section 2.3: CSA/HCA and other details; Section 2.2: mHC; Section 3.5.1: KV cache layout; Section 4.2.1: model settings)
- DeepSeek-V3.2: [config.json](https://huggingface.co/deepseek-ai/DeepSeek-V3.2/blob/main/config.json) (671B / 37B from the base-model comparison table on the V4 model card)
- DeepSeek-V4.1-Flash: [config.json](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/config.json) (`engram_*` fields)
- Qwen3.8-2.4T-A95B: [config.json](https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B/blob/main/config.json), [model card](https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B), [official blog](https://qwen.ai/blog?id=qwen3.8)
- Qwen3.8-Flash-Next: [config.json](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/config.json) (`qwen4_exp`; `hc_count`, `ngram_*`, `indexer_*`)
- Qwen3.5-0.8B: [config.json](https://huggingface.co/Qwen/Qwen3.5-0.8B/blob/main/config.json); Qwen3-0.6B: [config.json](https://huggingface.co/Qwen/Qwen3-0.6B/blob/main/config.json)
- Kimi-K3: [config.json](https://huggingface.co/moonshotai/Kimi-K3/blob/main/config.json), [model card](https://huggingface.co/moonshotai/Kimi-K3), [technical report PDF](https://github.com/MoonshotAI/Kimi-K3/blob/main/k3_tech_report.pdf) (we did not read all of it for this chapter; see "To be verified"); Kimi Linear report [arXiv:2510.26692](https://arxiv.org/abs/2510.26692) (KDA, 3:1, NoPE in MLA layers, MuonClip)
- gpt-oss-120b: [config.json](https://huggingface.co/openai/gpt-oss-120b/blob/main/config.json), [model card](https://huggingface.co/openai/gpt-oss-120b), [OpenAI model card arXiv:2508.10925](https://arxiv.org/abs/2508.10925)
- GLM-5.3: [config.json](https://huggingface.co/zai-org/GLM-5.3/blob/main/config.json), [model card](https://huggingface.co/zai-org/GLM-5.3); GLM-5 [config.json](https://huggingface.co/zai-org/GLM-5/blob/main/config.json) and [model card](https://huggingface.co/zai-org/GLM-5) (744B / 40B); [GLM-5.2 model card](https://huggingface.co/zai-org/GLM-5.2) (IndexShare [arXiv:2603.12201](https://arxiv.org/abs/2603.12201)); GLM-5 technical report [arXiv:2602.15763](https://arxiv.org/abs/2602.15763)
- MiniMax-M3: [config.json](https://huggingface.co/MiniMaxAI/MiniMax-M3/blob/main/config.json), [model card](https://huggingface.co/MiniMaxAI/MiniMax-M3), technical report [arXiv:2606.13392](https://arxiv.org/abs/2606.13392), [MSA code](https://github.com/MiniMax-AI/MSA)
- GPT-2: [config.json](https://huggingface.co/openai-community/gpt2/blob/main/config.json)

**To be verified**:

- We did not read all of the Kimi-K3 technical report (the PDF on GitHub) for this chapter. We read `mla_use_nope: true` as "the MLA layers use no position encoding", as in the Kimi Linear report. We did not verify the specific structure of Latent MoE: where the projection is, and if the experts share the projection. We checked only the parameter count (2.72T agrees with the 2.8T on the model card). We did not verify the details of Attention Residuals.
- The total / active parameters of GLM-5.3: the model card does not give them. The config has the same shapes as GLM-5, so we use 744B / 40B from the GLM-5 model card. The HF class counts 743.4B.
- The MiniMax-M3 config has both `num_mtp_modules: 7` and `num_nextn_predict_layers: 1`. We did not verify how many MTP layers inference actually uses.
- Item 62 (0) of the DeepSeek-V4 `compress_ratios` is the MTP layer. We inferred this from the mapping in HF `DeepseekV4Config` (0 = sliding window only) and from "61 layers + 1 MTP layer".
- We estimated the KV cache of DeepSeek-V4 as "head_dim numbers per compressed entry, shared over the tokens". The estimate does not include the buffered tokens that do not fill a compressed block yet. The report gives the exact storage format: RoPE dimensions in BF16, the rest in FP8, and the indexer computes in FP4. We mention this format only qualitatively in the comparison.
- Does Qwen3.8 have a formal technical report? When we wrote this chapter, we found only the model card and the blog link.

---

## Guided questions

Ask Claude Code these questions. Continue until you can explain the answers in your own words:

1. DeepSeek invented MLA and then replaced it in V4. Compare the costs of MLA in Chapter 21 (decode compute, not compatible with QK-Norm, needs special kernels) with the CSA/HCA method of this chapter. Which problem did DeepSeek most probably want to solve? What new cost does "shared K=V, head_dim 512" in V4 bring back?
2. Qwen3.8 and Kimi-K3 both use hybrid linear attention of about 3:1. One uses GQA in its full-attention layers, and the other uses MLA. If you change the 23 full-attention layers of Qwen3.8 to MLA (512 + 64), what is the KV cache at 128K? Calculate it with the method of `01_panorama.py`.
3. The block-sparse attention of MiniMax-M3 does not save KV cache. Why can it still make decode 15 times faster? (Hint: what does decode read in Chapter 21? At each step, does sparse attention read all of the KV, or only the selected blocks?)
4. This chapter says "there is no common definition of active parameters". To compare the compute per token of gpt-oss-120b and MiniMax-M3 fairly, which column must you use? The forward-pass compute per token is about how many times the active parameters?
5. By GOAL.md 2.1, sparse attention passed the threshold of 3 families only in 2026. "Several families use the same idea, but with different methods": is this consensus? If you were the author of this course, would you put it in the main text or keep it in the frontier notes?
6. Suppose that the next version of the main-line model can add only one new technique. Do you choose MTP, 3:1 hybrid linear attention, or local-global interleaving? List your reasons and the expected gain. Also list the tests that you add to the exam of Chapter 11 to show that it does not harm tool calling.

## Hands-on tasks

For each task, run the code and look at the result.

**Task 1 (basic)**: On Hugging Face, select a model that this chapter does not include (for example DeepSeek-V4-Flash, Qwen3.8-27B, GLM-5.3-Flash, NVIDIA Nemotron 3, or the newest Gemma). Read its `config.json`. Add an entry in the format of `models.json` (each field must come from a file that you read, with a link). Run `01_panorama.py` and `03_meta_params.py` again. What makes up its layers? Does the total parameter count agree with the model card? What is its KV cache at 128K? For which techniques in the adoption matrix does it change the count of families?

**Task 2 (core)**: Write a "local patch" for `zero.tools.kv_cache_calc.layout_from_config` (copy it into your own script first; do not change `zero/`). Make it read the `compress_ratios` of DeepSeek-V4 directly, and add a switch that includes or excludes the keys of the CSA indexer. Use it to calculate the KV cache of V4-Pro at 32K, 128K, and 1M. Do a parity check with `dsv4_layout` in `01_panorama.py`. Then count the bytes of each KV entry as "448 dimensions in FP8 + 64 dimensions in BF16". Is the result nearer to the "about 10%" of the report?

**Task 3 (challenge)**: Design your own "next-generation small model" (total parameters ≤ 0.8B, target context 32K, focus on tool calling). Write its config (number of layers, width, type of each layer, number of KV heads, MTP or not, vocabulary). Calculate its KV cache at 32K with `zero.tools.kv_cache_calc`. Calculate its total parameters with `zero.tools.count_params` (or with an HF class on the meta device). Then, for each choice that is different from the main-line model, write one sentence: which item of GOAL.md 2.1 it satisfies, who the adopters are, and why it pays off at 0.8B / 32K. Level 4 of the self-check skill `/ch26-panorama` discusses this design with you.

---

## Go deeper: CS336

This chapter matches Stanford CS336 (Spring 2026) <https://cs336.stanford.edu/>:

- **Lecture 3: architectures and hyperparameters.** The architecture choices of the modern Transformer: the position of the normalization, the activation function, the position encoding, and how attention heads share K/V. Also the usual values of the hyperparameters. The trunk of the evolution tree and the part "A very stable trunk" of this chapter show where this lecture lands in the 2026 flagships. The slides and videos are on the course page. (The build environment of this course cannot open the course page. The slides give the exact coverage.)
- **Lecture 4: alternatives to attention, and MoE.** Linear attention, state space models, sparse / local attention, and MoE. These match branches 1, 3, and 4 of this chapter.
- **Not covered in depth by CS336**: this course adds these topics with the models of September 2026. They are: checking the 6 newest flagships item by item from their configs, counting parameters with the official HF classes on the meta device, the layer-by-layer KV cache ledger, the consensus decision with "at least 3 families", the compressed attention and mHC of DeepSeek-V4, and the Latent MoE and AttnRes of Kimi-K3. See the references of this chapter.

---

## References

- DeepSeek-AI. *DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence*, 2026: <https://arxiv.org/abs/2606.19348>
- Kimi Team. *Kimi Linear: An Expressive, Efficient Attention Architecture* (KDA, 3:1 hybrid, NoPE in MLA layers), 2025: <https://arxiv.org/abs/2510.26692>
- Kimi Team. *Kimi K3 Technical Report*: <https://github.com/MoonshotAI/Kimi-K3/blob/main/k3_tech_report.pdf>
- GLM-5 Team. *GLM-5: from Vibe Coding to Agentic Engineering*, 2026: <https://arxiv.org/abs/2602.15763>; IndexShare: <https://arxiv.org/abs/2603.12201>
- MiniMax. *MiniMax-M3* technical report (MiniMax Sparse Attention), 2026: <https://arxiv.org/abs/2606.13392>
- OpenAI. *gpt-oss-120b & gpt-oss-20b Model Card*, 2025: <https://arxiv.org/abs/2508.10925>
- Qwen Team. *Qwen3.8-Max* blog: <https://qwen.ai/blog?id=qwen3.8>
- Sources of the nodes of the evolution tree: GPT-2 (Radford et al. 2019); Llama <https://arxiv.org/abs/2302.13971>; RMSNorm <https://arxiv.org/abs/1910.07467>; RoPE <https://arxiv.org/abs/2104.09864>; YaRN <https://arxiv.org/abs/2309.00071>; GLU variants <https://arxiv.org/abs/2002.05202>; GQA <https://arxiv.org/abs/2305.13245>; QK-Norm <https://arxiv.org/abs/2010.04245>; tied embedding <https://arxiv.org/abs/1608.05859>; DeepSeekMoE <https://arxiv.org/abs/2401.06066>; auxiliary-loss-free balancing <https://arxiv.org/abs/2408.15664>; MLA (DeepSeek-V2) <https://arxiv.org/abs/2405.04434>; Mistral 7B <https://arxiv.org/abs/2310.06825>; Gemma 2 <https://arxiv.org/abs/2408.00118>; Gated DeltaNet <https://arxiv.org/abs/2412.06464>; MTP (DeepSeek-V3) <https://arxiv.org/abs/2412.19437>
- [The main LLM architectures in 15,000 characters (Llama, Qwen, GLM, DeepSeek…)](https://zhuanlan.zhihu.com/p/2060741715095560795) (in Chinese): a side-by-side comparison of the structures of each family. Read it together with the evolution tree of this chapter (already in `references.md`).
- [Pretraining a Mini Kimi K3](https://books.vizuara.ai/book/pretraining-a-mini-k3): pretrain a small K3 with a "KDA + MLA hybrid" by hand. This is the best hands-on extension after Chapters 23 and 26 (already in `references.md`).
- [CS336](https://cs336.stanford.edu/) Lectures 3 and 4
- The links to the model configurations are in "Adopters and sources" above.

**Next step**: This is the last chapter of the course. Step 1 (write the course, and make the production code run end to end on the CPU) is complete here. Step 2 is to train the main-line model for real with the gate process when GPUs are available. Start from `runs/RUNBOOK.md`.
