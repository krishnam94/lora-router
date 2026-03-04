# LoRA Router - Competitive Analysis & Research Knowledge Base

Last updated: 2026-03-03

---

## 1. Competitive Landscape

### The Standard Benchmark: FLAN v2 48-Task LoRA Routing

Every serious LoRA routing paper evaluates on this benchmark. It is the de facto standard.

**Setup:**
- 48 tasks from FLAN v2 (36 NLU, 4 struct-to-text, 8 translation)
- One LoRA adapter per task, trained on LLaMA-2-7B (r=8, alpha=16, target: q_proj + v_proj)
- 50 test samples per task, mixed and shuffled (2,400 total)
- Pre-trained adapters available: `Styxxxx/llama2_7b_lora-{task_name}` on HuggingFace
- Retriever model: `Styxxxx/lora_retriever` (fine-tuned Instructor-XL)
- Training data: `lorahub/flanv2` on HuggingFace
- Reference implementation: [github.com/StyxXuan/LoraRetriever](https://github.com/StyxXuan/LoraRetriever)

**10 Semantic Clusters:**

| Cluster | Tasks | Metric |
|---------|-------|--------|
| Closed-book QA (4) | arc_challenge, arc_easy, natural_questions, trivia_qa | EM |
| Commonsense (4) | copa, hellaswag, piqa, story_cloze | EM |
| Coreference (2) | definite_pronoun_resolution, wsc | EM |
| NLI (10) | anli_r1/r2/r3, cb, mnli_matched/mismatched, qnli, rte, snli, wnli | EM |
| Paraphrase (4) | glue_mrpc, glue_qqp, paws_wiki, stsb | EM |
| Reading Comp + Commonsense (2) | cosmos_qa, record | EM |
| Reading Comprehension (6) | bool_q, drop, multirc, openbookqa, squad_v1, squad_v2 | EM |
| Sentiment (4) | imdb_reviews, sentiment140, sst2, yelp_polarity_reviews | EM |
| Struct-to-Text (4) | common_gen, dart, e2e_nlg, web_nlg_en | ROUGE |
| Translation (8) | para_crawl_enes, wmt14_enfr, wmt16_translate_* (6 pairs) | BLEU |

**Evaluation Regimes:**
- **Non-OOD**: Ground-truth adapter available in pool
- **Semi-OOD** (LORAUTER only): Adapter removed but task validation data remains
- **OOD**: Ground-truth adapter removed from pool entirely

**Oracle**: The adapter trained on the same task as the test query (task-aligned, not empirically best).

**Normalized Score**: `mean(method_score_i / oracle_score_i)` across all 48 tasks. 100% = matching oracle.

---

### Current Leaderboard (LLaMA2-7B, 48 FLAN v2 Tasks)

| Rank | Method | Venue | Non-OOD | OOD | Training-Free? | Data Needed | Composition |
|------|--------|-------|---------|-----|----------------|-------------|-------------|
| 1 | **LORAUTER** | arXiv Jan 2026 | **101.2%** | **88.4%** | Yes | Task validation sets (200 samples) | Output-space weighted fusion (K=3) |
| 2 | LoraRetriever | ACL 2024 | 92.9% | 83.2% | No (contrastive training) | 20 training samples/adapter | Output mixture (uniform 1/k) |
| 3 | ARROW | ICML 2024 | 82.5% | 82.0% | Yes | None (weights only) | Softmax-weighted top-k per token/layer |
| 4 | SpectR | COLM 2025 | 74.2% | 66.3% | Yes | None (weights only) | Uniform top-k per token/layer |
| 5 | LoraHub | COLM 2024 | 68.6% | 67.8% | Gradient-free opt | 5 validation examples/task | CMA-ES weighted linear combo |

---

### Method Deep Dives

#### LORAUTER (Current SOTA) - arXiv:2601.21795
**Authors**: Dhasade et al. (EPFL SaCS Lab)
**No public code.**

**How it works:**
1. Each task gets a representation = mean embedding of up to 200 validation samples using `Styxxxx/lora_retriever` encoder
2. Query embedding via same encoder with instruction prefix: "Represent the sentence for similar task retrieval"
3. Cosine similarity -> softmax with temperature tau=0.2 -> top-K=3 tasks
4. Output-space fusion: `h' = Wx + SUM(w_i * B_i * A_i * x)` where w_i = softmax similarity scores
5. Task-to-adapter pairing via Successive Halving (reduces eval budget 2x vs exhaustive)

**Key numbers:**
- Non-OOD: 101.2% (beats oracle via composition benefits)
- OOD: 88.4% (+5.2 over LoraRetriever)
- Scales to 1,567 HuggingFace adapters with only ~3 point drop

**Critical ablation**: Swapping just the retrieval component (LORAUTER retrieval + LoraRetriever composition) gets 89.9% OOD - higher than full LORAUTER (88.4%). Retrieval is the bigger contributor.

**Exploitable weaknesses:**
1. No per-layer routing - same mixture across all layers
2. No per-token routing - full query determines mixture
3. Fixed K=3 for all inputs - no adaptive selection
4. Reuses someone else's encoder (LoraRetriever's) - not optimized for routing
5. No wall-clock latency numbers reported anywhere
6. Only tested on LLaMA-2 (7B, 13B) - no modern architectures
7. No code released - can't verify claims
8. Translation is a weak spot - underperforms LoraRetriever on some translation tasks
9. Clustering sensitivity - optimal K varies wildly (12 for OOD vs 96 for non-OOD)

#### LoraRetriever - ACL 2024 Findings, arXiv:2402.09997
**Authors**: Zhao et al.
**Code**: [github.com/StyxXuan/LoraRetriever](https://github.com/StyxXuan/LoraRetriever)

**How it works:**
1. Each adapter represented by mean embedding of ~20 training samples via fine-tuned Instructor-XL
2. FAISS IndexFlatIP for retrieval
3. Best mode = "Mixture": run all top-k adapters independently, average outputs (uniform 1/k weights)
4. Batch inference via einsum: different items in batch can use different adapter subsets

**Exploitable weaknesses:**
1. Requires adapter training data (20 samples) - not available for public adapters
2. Uniform 1/k weighting - ignores similarity scores
3. Contrastive retriever needs retraining for new adapters
4. 63% top-1 retrieval accuracy (at 40% training) - wrong adapter a third of the time
5. Fusion mode collapses with heterogeneous tasks
6. 10-point OOD gap (92.9% -> 83.2%)

#### ARROW - ICML 2024
**Code**: [github.com/microsoft/mttl](https://github.com/microsoft/mttl)

**How it works:**
1. Per expert at each layer: SVD of A*B^T, extract first right singular vector as prototype
2. Per token: |dot(prototype, hidden_state)| -> softmax -> top-k weighted merge of LoRA params
3. Per-token, per-layer routing - finest granularity among all methods

**Weakness**: Only uses top-1 singular vector regardless of rank. Semantically ungrounded - routes on parameter values, not task meaning.

#### SpectR - COLM 2025
**No public code.**

Full-spectrum generalization of ARROW. Uses full SVD, computes routing score as L2 norm of projection into each adapter's subspace. Same quality as ARROW but uses all rank dimensions. Still underperforms semantic methods significantly (66.3% OOD).

---

## 2. Recent Papers (2025-2026) - New Techniques

### Tier 1: Highest Relevance

| Paper | Date | Core Idea | Relevance |
|-------|------|-----------|-----------|
| **SEQR** (arXiv:2509.18093) | Sep 2025 | QR-based routing - same quality as SpectR, fewer FLOPs. Provably identifies norm-maximizing adapter. | VERY HIGH - implement as efficient spectral strategy |
| **LoGo** (arXiv:2511.07129) | Nov 2025 | Training-free probe-based selection. Single forward pass through all adapters to score relevance. No data needed. | HIGH - zero-setup routing strategy |
| **Task-Aware VDB** (arXiv:2602.21222) | Feb 2026 | Vector DB retrieval + TIES/DARE/Linear merge. Nucleus sampling for similarity weights. | VERY HIGH - closest to our architecture |
| **HiLoRA** (arXiv:2510.12266) | Oct 2025 | Hierarchical rank-one component routing. Training-free with theoretical error bounds. 55% accuracy gains. | HIGH - sub-adapter granularity routing |
| **RouterDC** (NeurIPS 2024, arXiv:2409.19886) | Sep 2024 | Dual contrastive loss - handles ambiguous queries where multiple adapters are relevant. | HIGH - better contrastive training |
| **Geometry-Aware** (arXiv:2410.09908) | Oct 2024 | L1-sparse optimization for adapter blending. Preserves task manifold geometry. | HIGH - principled composition |
| **RAMoLE** (arXiv:2406.16989) | Jun 2024 | Full system: retrieval + MoLE gating + batch inference for heterogeneous requests. | HIGH - system design reference |
| **LoRA Soups** (COLING 2025, arXiv:2410.13025) | Oct 2024 | CAT (concatenation) with optimal weights beats model merging by 43%. | HIGH - composition method |

### Tier 2: Useful Architectural Ideas

| Paper | Date | Key Idea for Us |
|-------|------|-----------------|
| **LoRA-Mixer** (arXiv:2507.00029) | Jun 2025 | Plug-and-play mode over frozen public LoRAs. Attention-layer routing. |
| **HMoRA** (ICLR 2025) | 2025 | Layer depth determines routing granularity - shallow=task-level, deep=token-level |
| **DynMoLE** (arXiv:2504.00661) | Apr 2025 | Tsallis entropy for adaptive top-k - auto-selects number of adapters per input |
| **CoMoL** (arXiv:2603.00573) | Feb 2026 | Core-space representation for compact expert storage |
| **SMoRA** (arXiv:2501.15103) | Jan 2025 | Rank-as-expert - theoretical equivalence between multi-adapter routing and within-adapter rank routing |
| **S'MoRE** (NeurIPS 2025, arXiv:2504.06426) | Apr 2025 | Tree-structured residual experts. Already in HuggingFace PEFT. |
| **LoRA.rar** (ICCV 2025 Best Paper) | Dec 2024 | Hypernetwork-learned merge weights. 4000x faster than optimization-based. |

---

## 3. What We Need to Beat LORAUTER

### Our Advantages to Exploit

1. **Adaptive K**: LORAUTER always uses K=3. We route with confidence-based adaptive selection:
   - High confidence (>0.8) -> single adapter (fast)
   - Medium (0.4-0.8) -> compose top-2-3
   - Low (<0.4) -> base model fallback
   - DynMoLE's Tsallis entropy approach could inform this

2. **Better composition**: LORAUTER does simple output-space weighted average. We can:
   - Use TIES-Merging to handle sign conflicts
   - Use DARE to prune redundant parameters before merging
   - Use LoRA Soups' CAT strategy
   - Use similarity-proportional weights (not just softmax temperature)

3. **Multi-signal routing**: LORAUTER uses only text embeddings. We can ensemble:
   - Semantic similarity (their approach, our baseline)
   - Spectral signals via SEQR (adapter weight structure)
   - Trained classifier (when training data available)
   - Combine signals for more robust routing

4. **Better encoder**: They reuse LoraRetriever's 2024 encoder. Modern alternatives:
   - `all-MiniLM-L6-v2` (fast, good quality)
   - `gte-large-en-v1.5` or `e5-large-v2` (better quality)
   - Fine-tune on routing-specific contrastive data with RouterDC's dual loss

5. **Production features LORAUTER lacks**:
   - Actual latency measurements and optimization
   - Structured logging of routing decisions
   - OpenAI-compatible proxy for vLLM integration
   - YAML-based adapter registry
   - Confidence scores exposed to callers

6. **Modern architecture support**: Test on LLaMA-3, Mistral, Qwen (they only tested LLaMA-2)

### Our Target Numbers

| Setting | LORAUTER | Our Target | How |
|---------|----------|------------|-----|
| Non-OOD | 101.2% | >101% | Match via similarity + smart composition |
| OOD | 88.4% | >90% | Multi-signal ensemble + adaptive K |
| Latency overhead | Not reported | <50ms routing | Efficient embedding + FAISS |
| Scale (1000+ adapters) | ~3pt drop | <2pt drop | Better clustering + caching |

---

## 4. Other Benchmarks to Target

| Benchmark | What it Tests | Priority |
|-----------|--------------|----------|
| **LoraRetriever 48-task** | LoRA routing accuracy | MUST HAVE |
| **RouterBench** (ICML 2024) | LLM routing (405K samples, 64 tasks) | SHOULD HAVE |
| **MT-Bench** | Multi-turn quality across 8 categories | SHOULD HAVE |
| **LoRALib** (EMNLP 2025) | 40 tasks, 680 LoRA modules | NICE TO HAVE |
| **GSM8K** | Math reasoning (end-to-end quality) | SHOULD HAVE |
| **HumanEval** | Code generation (end-to-end quality) | SHOULD HAVE |

---

## 5. Training Datasets for Domain-Specific Adapters

| Domain | Dataset | Size | HuggingFace ID |
|--------|---------|------|----------------|
| Code | Magicoder-Evol-Instruct-110K | 110K | `ise-uiuc/Magicoder-Evol-Instruct-110K` |
| Math | MetaMathQA | 395K | `meta-math/MetaMathQA` |
| Summarization | CNN/DailyMail | 300K+ | `abisee/cnn_dailymail` |
| Reasoning | SlimOrca | ~500K | `Open-Orca/SlimOrca` |
| Creative | WritingPrompts (Reddit) | ~300K | Available on HF |
| Multilingual | Aya Dataset | 513M across 114 langs | `CohereForAI/aya_dataset` |

---

## 6. Key Sources

### Papers (by relevance)
- LORAUTER: https://arxiv.org/abs/2601.21795
- LoraRetriever: https://arxiv.org/abs/2402.09997 | https://aclanthology.org/2024.findings-acl.263/
- ARROW: https://arxiv.org/abs/2405.11157 | Code: https://github.com/microsoft/mttl
- SpectR: https://arxiv.org/abs/2504.03454
- LoraHub: https://arxiv.org/abs/2307.13269 | Code: https://github.com/sail-sg/lorahub
- SEQR: https://arxiv.org/abs/2509.18093
- LoGo: https://arxiv.org/abs/2511.07129
- RouterDC: https://arxiv.org/abs/2409.19886 | Code: https://github.com/shuhao02/RouterDC
- Task-Aware VDB: https://arxiv.org/abs/2602.21222
- HiLoRA: https://arxiv.org/abs/2510.12266
- Geometry-Aware Composition: https://arxiv.org/abs/2410.09908 | Code: https://github.com/Jinpf314/GeometryAwareAdapterComposition
- RAMoLE: https://arxiv.org/abs/2406.16989
- LoRA Soups: https://arxiv.org/abs/2410.13025 | Code: https://github.com/aksh555/LoRA-Soups
- LoRA-Mixer: https://arxiv.org/abs/2507.00029
- HMoRA: https://openreview.net/forum?id=lTkHiXeuDl | Code: https://github.com/LiaoMengqi/HMoRA
- DynMoLE: https://arxiv.org/abs/2504.00661
- CoMoL: https://arxiv.org/abs/2603.00573
- SMoRA: https://arxiv.org/abs/2501.15103
- S'MoRE: https://arxiv.org/abs/2504.06426 | Code: https://github.com/ZimpleX/SMoRE-LLM
- LoRA.rar: https://arxiv.org/abs/2412.05148
- RouterBench: https://arxiv.org/abs/2403.12031 | Code: https://github.com/withmartian/routerbench
- LoRALib: https://arxiv.org/abs/2509.18137

### Benchmarks & Data
- LoraRetriever adapters: https://huggingface.co/Styxxxx (66 adapters per model size)
- LoraRetriever encoder: https://huggingface.co/Styxxxx/lora_retriever
- FLAN v2 data: https://huggingface.co/datasets/lorahub/flanv2
- LoraRetriever code: https://github.com/StyxXuan/LoraRetriever
- RouterBench: https://github.com/withmartian/routerbench

### Reference Implementations
- microsoft/mttl (ARROW): https://github.com/microsoft/mttl
- sail-sg/lorahub: https://github.com/sail-sg/lorahub
- StyxXuan/LoraRetriever: https://github.com/StyxXuan/LoraRetriever
- shuhao02/RouterDC: https://github.com/shuhao02/RouterDC
