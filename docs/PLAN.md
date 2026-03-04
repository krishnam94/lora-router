# Revised Plan: Build `lora-router` (v2 - Benchmark-Aware)

**Date**: 2026-03-03
**Status**: Pending approval
**Research**: See `docs/research/competitive_analysis.md` for full competitive analysis.

---

## What Changed From v1

The original plan designed a generic routing library. This revised plan is designed to **beat LORAUTER's 88.4% OOD score** on the standard FLAN v2 48-task benchmark, while also being a production-quality pip-installable library.

Key changes:
1. **Strategies redesigned** to target specific competitive weaknesses
2. **Composition upgraded** from simple PEFT merge to multi-method (TIES/DARE/CAT/weighted)
3. **Adaptive K** replaces fixed top-3 - confidence drives how many adapters to compose
4. **Ensemble strategy** combines semantic + spectral signals (no competitor does this)
5. **Benchmark-first** - evaluation module built in Session 1, not Session 3
6. **FLAN v2 48-task eval** as first-class development target
7. **New strategies added**: SEQR (spectral, training-free), LoGo-style probe routing

---

## Architecture (Revised)

```
User Query
    |
    v
[LoRARouter]
    |
    +---> [Strategy] ----+--- SimilarityStrategy (embedding cosine sim - zero setup)
    |                    +--- SEQRStrategy (QR-based spectral - zero data needed)
    |                    +--- ClassifierStrategy (trained sklearn/contrastive)
    |                    +--- EnsembleStrategy (multi-signal voting/weighted)
    |         |
    |         v
    |    AdapterSelection[] (name + confidence + scores)
    |         |
    +---> [SmartComposer] (confidence-based adaptive decision)
    |         |
    |         +-- confidence > threshold_high --> single adapter (fast path)
    |         +-- threshold_low < conf < threshold_high --> compose top-k
    |         |       +-- TIES-Merging (resolves sign conflicts)
    |         |       +-- DARE (prunes redundant params before merge)
    |         |       +-- Linear (simple weighted average)
    |         |       +-- CAT (concatenation a la LoRA Soups)
    |         +-- confidence < threshold_low --> base model fallback
    |
    v
[Output] (or forward to vLLM via proxy)
```

### Design Decisions That Beat Competitors

| Decision | Why | Beats |
|----------|-----|-------|
| Adaptive K (1-5 based on confidence) | LORAUTER always uses K=3 even for easy queries | LORAUTER |
| Similarity-proportional weights | LoraRetriever uses uniform 1/k | LoraRetriever |
| TIES/DARE composition | Everyone uses simple linear merge | All |
| Ensemble (semantic + spectral) | No one combines these signals | All |
| SEQR strategy (training-free, data-free) | Works without any adapter training data | LoraRetriever |
| Confidence scores exposed in API | No competitor surfaces routing confidence | All |
| Pluggable encoder (swap sentence-transformers models) | LORAUTER locked to one encoder | LORAUTER |

---

## File Structure (Revised)

```
~/Desktop/projects/lora-router/
├── pyproject.toml
├── src/lora_router/
│   ├── __init__.py                   # Public API + __version__
│   ├── types.py                      # Pydantic models
│   ├── registry.py                   # AdapterRegistry: register, list, YAML, embeddings
│   ├── router.py                     # LoRARouter: route(), route_batch()
│   ├── strategies/
│   │   ├── __init__.py
│   │   ├── base.py                   # BaseStrategy ABC
│   │   ├── similarity.py            # Embedding cosine sim (default)
│   │   ├── seqr.py                  # QR-based spectral routing (training-free)
│   │   ├── classifier.py            # Trained sklearn or contrastive classifier
│   │   └── ensemble.py              # Multi-strategy weighted voting
│   ├── composition/
│   │   ├── __init__.py
│   │   ├── merger.py                 # MergeMethod: TIES, DARE, linear, CAT
│   │   └── composer.py              # SmartComposer: confidence-based adaptive
│   ├── middleware/
│   │   ├── __init__.py
│   │   ├── proxy.py                  # VLLMProxy (FastAPI, OpenAI-compatible)
│   │   └── logging.py               # Structured routing decision logs
│   └── eval/
│       ├── __init__.py
│       ├── metrics.py               # normalized_oracle_score, routing_accuracy, MRR, NDCG
│       ├── benchmarks.py            # RoutingBenchmark, FlanV2Benchmark
│       └── report.py               # Markdown tables + matplotlib plots
├── scripts/
│   ├── train_adapters.py            # Train 6 LoRAs with Unsloth
│   ├── prepare_benchmark_data.py    # Generate test queries
│   └── download_flan_adapters.py    # Download Styxxxx/llama2_7b_lora-* adapters
├── benchmarks/
│   ├── configs/                     # Benchmark config YAMLs
│   ├── queries/                     # Test queries per domain
│   └── run_benchmark.py             # Full benchmark runner
├── examples/
│   ├── quickstart.py
│   ├── vllm_deployment.py
│   ├── custom_strategy.py
│   └── flan_v2_eval.py             # Run the standard 48-task eval
├── tests/
│   ├── conftest.py
│   ├── unit/
│   │   ├── test_types.py
│   │   ├── test_registry.py
│   │   ├── test_similarity.py
│   │   ├── test_seqr.py
│   │   ├── test_classifier.py
│   │   ├── test_ensemble.py
│   │   ├── test_router.py
│   │   ├── test_merger.py
│   │   ├── test_composer.py
│   │   ├── test_metrics.py
│   │   └── test_benchmarks.py
│   └── integration/
│       ├── test_similarity_real.py
│       ├── test_flan_routing.py
│       └── test_proxy.py
├── docs/
│   ├── PLAN.md                      # This file
│   └── research/
│       └── competitive_analysis.md  # Full competitive analysis with sources
├── .github/workflows/ci.yml
├── README.md
└── LICENSE                          # Apache 2.0
```

---

## Dependencies (Revised)

**Core**: `torch`, `transformers`, `peft>=0.8`, `sentence-transformers>=2.2`, `pydantic>=2.0`, `pyyaml`, `numpy`, `scikit-learn`, `faiss-cpu`

**Extras**:
- `[serve]`: fastapi, uvicorn, httpx
- `[eval]`: matplotlib, rouge-score, tabulate, nltk (for BLEU)
- `[dev]`: pytest, pytest-cov, ruff, mypy, pre-commit
- `[train]`: unsloth, datasets, trl

---

## Session Plan (Revised)

### Session 1: Library Core + Eval Foundation
**Build everything needed to run routing experiments without GPU.**

1. **Scaffold**: pyproject.toml, git init, full package structure
2. **types.py**: AdapterInfo (with description, domain, example_queries, embedding fields), AdapterSelection (name, confidence, scores dict), RoutingDecision (selections list, strategy_used, composition_method, latency_ms), MergeConfig
3. **registry.py**: AdapterRegistry - register with metadata + optional example queries, compute/cache embeddings, YAML import/export, list/get/remove
4. **strategies/base.py**: BaseStrategy ABC with `route(query, registry) -> list[AdapterSelection]` and `route_batch(queries, registry)`
5. **strategies/similarity.py**: SimilarityStrategy - pluggable encoder (default: all-MiniLM-L6-v2), cosine similarity, returns ranked selections with confidence = similarity score, FAISS index for scale
6. **strategies/seqr.py**: SEQRStrategy - QR decomposition of adapter B matrices, route by max activation norm, no training data needed (implement from SEQR paper arXiv:2509.18093)
7. **strategies/classifier.py**: ClassifierStrategy - train sklearn classifier on adapter example queries, predict with calibrated probabilities as confidence
8. **strategies/ensemble.py**: EnsembleStrategy - weighted combination of multiple strategies, configurable weights, returns merged rankings
9. **router.py**: LoRARouter - takes registry + strategy, `route()` returns RoutingDecision, `route_batch()` for efficiency
10. **composition/merger.py**: MergeMethod enum (TIES, DARE, LINEAR, CAT), merge function wrapping PEFT's merge utilities + custom TIES/DARE
11. **composition/composer.py**: SmartComposer - confidence thresholds (configurable), picks single/compose/fallback, selects merge method
12. **eval/metrics.py**: normalized_oracle_score, routing_accuracy, mrr, ndcg, per_cluster_scores
13. **tests/conftest.py**: MockRegistry, MockEmbedder, MockStrategy fixtures, sample adapter configs
14. **Unit tests** (~60): cover types, registry, all 4 strategies, router, merger, composer, metrics
15. **CI**: .github/workflows/ci.yml (ruff + mypy + pytest on 3.10/3.11/3.12)

**Verify**: `pip install -e ".[dev]"` works, `pytest tests/unit/ -v` all green, `ruff check src/` clean, `mypy src/` passes

### Session 2: FLAN v2 Benchmark Setup + Adapter Training
**Set up the exact evaluation protocol used by LORAUTER/LoraRetriever.**

1. **scripts/download_flan_adapters.py**: Download all 48 `Styxxxx/llama2_7b_lora-*` adapters from HuggingFace (use existing pre-trained adapters instead of training from scratch for benchmark parity)
2. **benchmarks/configs/flan_v2.yaml**: Full 48-task config with task names, clusters, metrics, adapter HF IDs
3. **eval/benchmarks.py**: FlanV2Benchmark class - loads mixed test set, runs routing, evaluates per-task and per-cluster, computes normalized oracle scores for all three regimes (Non-OOD, Semi-OOD, OOD)
4. **scripts/train_adapters.py**: Train 6 custom LoRAs (code, math, summarize, reasoning, creative, multilingual) on Unsloth for our demo/quickstart (separate from FLAN v2 eval)
5. **scripts/prepare_benchmark_data.py**: Generate test queries for custom adapters
6. Run FLAN v2 benchmark with SimilarityStrategy as baseline - establish our starting numbers
7. Run with SEQRStrategy - compare
8. Run with EnsembleStrategy - compare
9. Iterate on strategy parameters to maximize OOD score

**Verify**: Can reproduce LoraRetriever's ~83% OOD baseline, our similarity strategy scores above that, ensemble approaches LORAUTER's 88.4%

### Session 3: Optimize for Benchmark + Composition
**Tune routing to beat LORAUTER. Add smart composition.**

1. Test different encoders: all-MiniLM-L6-v2 vs gte-large vs e5-large vs the LoraRetriever encoder
2. Implement RouterDC-style dual contrastive training for ClassifierStrategy (if encoder alone isn't enough)
3. Tune ensemble weights between semantic + spectral signals
4. Test adaptive K vs fixed K=3 - measure impact on OOD score
5. Test TIES vs DARE vs Linear vs CAT composition on the FLAN v2 benchmark
6. Tune confidence thresholds for SmartComposer
7. Run full ablation study and generate results tables
8. eval/report.py: Generate markdown report with comparison tables + plots

**Verify**: OOD score > 88.4% (beat LORAUTER), latency overhead < 50ms, results reproducible

### Session 4: Middleware + Ship
**Production-ready features and launch.**

1. **middleware/proxy.py**: VLLMProxy - FastAPI server, OpenAI-compatible /v1/chat/completions, auto-routes to best adapter
2. **middleware/logging.py**: Structured JSON logging of routing decisions (query, selected adapter, confidence, latency, strategy used)
3. **Examples**: quickstart.py, vllm_deployment.py, custom_strategy.py, flan_v2_eval.py
4. **README.md**: Architecture diagram, quickstart, benchmark results table, comparison to prior work, installation
5. Integration tests: real sentence-transformers, FastAPI TestClient
6. Final CI run, push to GitHub as `krishnam94/lora-router`
7. GitHub topics: lora, peft, routing, adapter, llm, mixture-of-experts

**Verify**: Fresh `pip install` works, all CI green, README renders, proxy starts, examples run

---

## Benchmark Strategy Summary

### Phase 1: Reproduce baselines (Session 2)
- Download Styxxxx adapters (no training cost)
- Run our SimilarityStrategy on FLAN v2 48-task
- Compare against LoraRetriever's reported 83.2% OOD
- Establish our baseline number

### Phase 2: Beat LORAUTER (Session 3)
- Primary attack vector: multi-signal ensemble (semantic + spectral)
- Secondary: adaptive K instead of fixed K=3
- Tertiary: better composition (TIES/DARE vs linear)
- Target: >90% OOD on FLAN v2 48-task

### Phase 3: Additional benchmarks (Session 4)
- MT-Bench per-category with custom 6-domain adapters
- GSM8K (math adapter quality)
- HumanEval (code adapter quality)
- Latency benchmarks

---

## Progress Tracking

Tasks tracked via Claude Code task list. Sprint updates in `~/Desktop/docs/tasks/todo.md`.

Each session = parent task with sub-tasks for each deliverable. Complete only after verification passes.

---

## Risk Mitigations (Updated)

| Risk | Mitigation |
|------|-----------|
| Can't beat LORAUTER OOD | Ensemble (semantic + spectral) is our unique signal combo. If that's not enough, add RouterDC contrastive training. |
| Styxxxx adapters not available | They're published on HuggingFace. Fallback: train our own on FLAN v2 with LoraRetriever's exact config. |
| SEQR doesn't help ensemble | Keep it as a standalone strategy. The ensemble can still win on semantic-only if encoder is better. |
| TIES/DARE composition slower | Benchmark latency. If too slow, default to linear merge with TIES as optional. |
| Encoder quality bottleneck | Test 4+ encoders. Fine-tune if needed (Session 3). |
| PyPI name taken | Check early. Backups: `lorarouter`, `adapter-router`, `lora-routing`. |
