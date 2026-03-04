# LoRA Router - Learning Guide

A ground-up explanation of everything in this project, written for someone who knows basic LoRA concepts.

---

## 1. LoRA Recap (What You Already Know)

**LoRA (Low-Rank Adaptation)** adds small trainable matrices to a frozen LLM. Instead of fine-tuning all parameters, you add two small matrices A and B to specific layers:

```
Original: y = Wx
With LoRA: y = Wx + BAx

Where W is frozen (e.g., 4096 x 4096)
A is small: rank x 4096 (e.g., 8 x 4096)
B is small: 4096 x rank (e.g., 4096 x 8)
```

So instead of training 16M parameters per layer, you train 65K. The `rank` (r) controls the capacity - higher rank = more expressiveness but more parameters.

**Key LoRA settings:**
- `r` (rank): Usually 4-16. Higher = more capacity.
- `alpha`: Scaling factor. The LoRA output is multiplied by `alpha/r`. Usually `alpha = 2*r`.
- `target_modules`: Which attention matrices get LoRA. Common: `["q_proj", "v_proj"]`.

---

## 2. The Problem We're Solving

You can train many LoRA adapters for different tasks:
- A **code** adapter (trained on coding data)
- A **math** adapter (trained on math problems)
- A **summarization** adapter (trained on summarization)
- etc.

**The question**: When a user sends a query, which adapter should you use?

Today, every serving system (vLLM, LoRAX, S-LoRA) requires the **caller to manually specify** which adapter. There's no automatic routing.

**lora-router** solves this: given a query, automatically pick the best adapter(s).

---

## 3. Core Concepts in This Project

### 3.1 Routing Strategy

A **routing strategy** takes a query and ranks all available adapters by relevance. We implement four:

#### Similarity Strategy (our default)
The simplest approach. Works with zero training.

1. **Embed** each adapter's description/examples into a vector (using a sentence-transformer model)
2. **Embed** the incoming query into the same vector space
3. **Cosine similarity**: find which adapter embeddings are closest to the query embedding
4. Rank by similarity score

**Cosine similarity** measures angle between two vectors. If vectors point the same direction, similarity = 1. Perpendicular = 0. Opposite = -1.

```
cos(a, b) = (a . b) / (|a| * |b|)
```

**Sentence-transformers** are models trained to map text to meaningful vectors. Similar texts get similar vectors. We use `all-MiniLM-L6-v2` (fast, 384-dim) by default.

**FAISS** (Facebook AI Similarity Search) is a library for fast nearest-neighbor search. When you have 1000+ adapters, brute-force cosine similarity is slow. FAISS builds an index for O(log n) search.

#### SEQR Strategy (training-free, data-free)
Based on the SEQR paper (Sep 2025). The insight: the adapter's weight matrices themselves encode what kind of inputs they're good at.

1. For each adapter, take the B and A weight matrices
2. Compute **QR decomposition** of B*A - this extracts the "directions" the adapter transforms
3. For a query, project it into each adapter's subspace
4. The adapter whose subspace captures the most "energy" (highest activation norm) is the best fit

**QR decomposition**: Factors a matrix M into Q (orthogonal) and R (upper triangular). Q gives you the basis vectors of the column space. R gives you the coefficients. It's like SVD but faster.

**SVD (Singular Value Decomposition)**: Factors M = U * S * V^T. The singular vectors (columns of V) represent directions of maximum variance. The singular values (diagonal of S) tell you how important each direction is. SEQR/SpectR use this to characterize what an adapter "does" in the weight space.

**Activation norm**: After projecting a query into an adapter's subspace, the L2 norm of the projection tells you "how much of the query's information this adapter captures." Higher norm = better fit.

#### Classifier Strategy (trained)
If you have example queries per adapter, train a classifier:

1. Embed all example queries
2. Train a **LogisticRegression** to predict adapter-name from embedding
3. At inference, classify the query - probabilities become confidence scores

**Calibrated probabilities**: Raw classifier scores aren't true probabilities. **Platt scaling** (CalibratedClassifierCV) adjusts them so a 0.8 confidence really means the model is right 80% of the time. Important for our confidence-based composition.

#### Ensemble Strategy (our differentiator)
No competitor combines multiple routing signals. We combine:
- Semantic signal (from SimilarityStrategy)
- Spectral signal (from SEQRStrategy)
- Optionally: classifier signal

**Weighted voting**: Each strategy votes for adapters. Votes are weighted by strategy importance. Final ranking merges all votes.

### 3.2 Smart Composition

After routing picks the best adapters, we decide what to do. This is where **SmartComposer** comes in.

Three actions based on confidence:

```
confidence > 0.8  -->  SINGLE   (just use the top adapter)
0.3 < conf < 0.8  -->  COMPOSE  (merge top-k adapters together)
confidence < 0.3  -->  FALLBACK (use the base model without any adapter)
```

**Why not always compose?** LORAUTER always merges K=3 adapters. But if the router is 95% confident about one adapter, merging in two weaker ones adds noise. And if confidence is very low, no adapter is relevant - better to use the base model.

**Why this beats competitors**: LORAUTER's fixed K=3 is suboptimal. Our adaptive K uses 1 adapter when confident (faster inference) and more when uncertain (better quality). Best of both worlds.

### 3.3 Merge Methods

When composing multiple adapters, you need to combine their weights. Four methods:

#### LINEAR (simple average)
```python
merged_A = w1 * A1 + w2 * A2 + w3 * A3
merged_B = w1 * B1 + w2 * B2 + w3 * B3
```
Fast but naive. If two adapters learned opposite things for the same parameter, they cancel out.

#### TIES (Trim, Elect Sign, Merge)
Handles the "cancellation" problem from the TIES-Merging paper (arXiv:2306.01708):

1. **Trim**: Zero out small-magnitude parameters (keep only top-k% by magnitude). The `density` parameter controls how many to keep.
2. **Elect Sign**: For each parameter position, vote on the sign (positive or negative). The sign with more total magnitude wins.
3. **Merge**: Average only the parameters that agree with the elected sign.

This prevents the "averaging to zero" problem when adapters disagree.

#### DARE (Drop And REscale)
From arXiv:2311.03099:

1. **Drop**: Randomly zero out (1-density)% of parameters
2. **Rescale**: Multiply remaining by 1/density to maintain expected magnitude
3. **Merge**: Weighted average of the rescaled adapters

The intuition: most LoRA parameters are redundant. Dropping random ones before merging reduces interference between adapters.

#### CAT (Concatenation)
From LoRA Soups paper (arXiv:2410.13025):

Instead of averaging, concatenate the A matrices along the rank dimension:
```
A_merged = [A1; A2; A3]    shape: (3*rank, hidden_dim)
B_merged = [B1, B2, B3]    shape: (hidden_dim, 3*rank)
```

Result: a higher-rank adapter that preserves all adapters' knowledge. No information loss, but increases compute.

### 3.4 Confidence Calibration

Raw cosine similarities aren't great confidence scores (they might all be 0.3-0.5 with tiny differences). We use **softmax with temperature** to spread them out:

```python
confidence_i = exp(similarity_i / temperature) / sum(exp(similarity_j / temperature))
```

**Temperature** (tau) controls spread:
- tau = 1.0: standard softmax, moderate spread
- tau = 0.2: sharper, top scores get much higher confidence (what LORAUTER uses)
- tau = 5.0: flatter, all scores more uniform

Low temperature makes the router more decisive (higher confidence for the best match).

---

## 4. The Benchmark We're Targeting

### FLAN v2 48-Task Benchmark

The standard eval for LoRA routing papers. Setup:

- **48 tasks** from FLAN v2 dataset (NLU, translation, text generation)
- **10 semantic clusters**: NLI, Sentiment, Commonsense, Reading Comprehension, etc.
- **One LoRA adapter per task** trained on LLaMA-2-7B
- **2,400 test queries** (50 per task), mixed and shuffled
- The router must figure out which adapter to use based only on the query text

**Three evaluation regimes:**
- **Non-OOD (In-Domain)**: The correct adapter is in the pool. Can we find it?
- **Semi-OOD**: The correct adapter is removed, but similar ones exist.
- **OOD (Out-of-Domain)**: The correct adapter is removed entirely. Can we find the next best?

**Normalized Oracle Score**: The primary metric. For each task:
```
score_i = (your method's score on task i) / (oracle's score on task i) * 100
```
Then average across all 48 tasks. 100% = matching the perfect per-task adapter. >100% is possible when composing multiple adapters outperforms any single one.

### Current Leaderboard (OOD - the hard setting)

| Method | OOD Score | Our target |
|--------|-----------|------------|
| LORAUTER | 88.4% | Beat this |
| LoraRetriever | 83.2% | - |
| ARROW | 82.0% | - |
| LoraHub | 67.8% | - |
| SpectR | 66.3% | - |

Our target: **>90% OOD** via ensemble routing + adaptive composition.

---

## 5. How Our Code Maps to These Concepts

| Concept | Code location | What it does |
|---------|--------------|--------------|
| Adapter metadata | `types.py: AdapterInfo` | Name, path, description, domain, examples |
| Routing result | `types.py: AdapterSelection` | Adapter name + confidence + per-strategy scores |
| Full decision | `types.py: RoutingDecision` | Selections + action + latency + strategy name |
| Adapter storage | `registry.py: AdapterRegistry` | Register/get/list adapters, cache embeddings, YAML I/O |
| Similarity routing | `strategies/similarity.py` | Cosine sim with sentence-transformers + optional FAISS |
| Spectral routing | `strategies/seqr.py` | QR decomposition of adapter weights |
| Trained routing | `strategies/classifier.py` | Sklearn classifier on example queries |
| Multi-signal routing | `strategies/ensemble.py` | Combine multiple strategies with weighted voting |
| Adaptive composition | `composition/composer.py` | Confidence thresholds -> SINGLE/COMPOSE/FALLBACK |
| Weight merging | `composition/merger.py` | LINEAR, TIES, DARE, CAT merge methods |
| Main interface | `router.py: LoRARouter` | Combines strategy + composer, measures latency |
| Evaluation | `eval/metrics.py` | Routing accuracy, MRR, NDCG, normalized oracle score |

---

## 6. Key Jargon Reference

| Term | Meaning |
|------|---------|
| **Adapter** | A LoRA module - the A and B matrices that modify a frozen model |
| **Routing** | Deciding which adapter to use for a given query |
| **Composition / Merging** | Combining multiple adapters' weights into one |
| **Confidence** | How sure the router is about its selection (0-1) |
| **Oracle** | Perfect routing - always picks the adapter trained on that exact task |
| **OOD** | Out-of-distribution - the correct adapter isn't available |
| **Normalized score** | Method score / Oracle score * 100 |
| **Top-k** | Selecting the k highest-scored items |
| **Softmax** | Converts raw scores to probabilities that sum to 1 |
| **Temperature** | Controls how "peaked" the softmax distribution is |
| **Cosine similarity** | Angle-based similarity between vectors (-1 to 1) |
| **FAISS** | Facebook's fast similarity search library |
| **Sentence-transformer** | Model that converts text to a dense vector |
| **QR decomposition** | Matrix factorization: M = QR (orthogonal * upper triangular) |
| **SVD** | Singular Value Decomposition: M = USV^T |
| **TIES** | Trim, Elect Sign, Merge - handles parameter sign conflicts |
| **DARE** | Drop And REscale - sparsifies before merging |
| **CAT** | Concatenation of LoRA matrices along rank dimension |
| **Platt scaling** | Calibrates classifier outputs to true probabilities |
| **MRR** | Mean Reciprocal Rank - 1/position of correct answer, averaged |
| **NDCG** | Normalized Discounted Cumulative Gain - rewards correct answers at higher ranks |
| **FLAN v2** | Google's instruction-tuning dataset with many NLP tasks |
| **PEFT** | Parameter-Efficient Fine-Tuning library by HuggingFace |
| **Activation norm** | L2 norm of a vector after projection - measures "how much energy" |
| **Subspace** | The "directions" in weight space that an adapter occupies |
| **Rank** | Dimensionality of the LoRA decomposition (r in A: r x d, B: d x r) |
| **Spectral routing** | Using eigenvalues/singular values of weight matrices for routing |
