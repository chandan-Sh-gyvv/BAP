# BAP — Budgeted Activation Propagation Engine

https://doi.org/10.5281/zenodo.18862351

> A graph-based reasoning engine that replaces LLM chain-of-thought with structured activation propagation over a ThoughtSpace.
> **47.5× token reduction** vs standard CoT. Deterministic. Interpretable. Fast.

Copyright 2026 Chandan S H — Licensed under the [Apache License 2.0](LICENSE)

---

## What Is BAP?

BAP is a reasoning kernel that traverses a **ThoughtSpace** — a graph of atomic reasoning primitives — using a physics-inspired energy budget equation instead of generating tokens with an LLM.

The core idea: reasoning is not generation. Reasoning is **retrieval + propagation + pruning**. The LLM is fuel, not the engine.

```
Query → OOV Detection → SHSRS Index Search → Activation Scoring → Pruning → Path
```

BAP produces a deterministic reasoning path in 4–6 steps for queries that would cost an LLM 240–360 tokens of chain-of-thought.

---

## The Activation Equation

Every candidate node at each step is scored by:

```
A(j) = A_source · sim(i,j) · gate(j) · decay(j) · log(1 + budget) · novelty(j)

gate(j)    = sigmoid(GATE_SCALE · (strength_j − harm_j))
decay(j)   = exp(−age_j · λ)
novelty(j) = 1 − max cosine(embed_j, any path node)
```

This equation is the physics of the engine. It is never replaced.

---

## Architecture

```
Query
  │
  ▼
Query Expansion          5 reformulations, averaged embedding
  │
  ▼
OOV Detection            Scan query terms — define missing concepts before traversal
  │
  ▼
SHSRS Search             Spherical Hierarchical Semantic Retrieval System
  │                      Own research index — KMeans + IGAR + per-cluster HNSW
  ▼
Activation Scoring       A(j) = sim · gate · decay · energy · novelty
  │
  ▼
Pruning Gates            harm > 0.5 → prune
                         contradiction cosine < −0.15 → prune
                         activation < 0.008 → prune
  │
  ▼
Competitive Inhibition   budget > 50% → top-2 (explore)
                         budget ≤ 50% → top-1 (exploit)
  │
  ▼
Retrieval Miss?          Drift detected?    → LLM bridge injection
                         Stagnation?        → LLM bridge injection
                         Miss?              → LLM contextual node
                         OOV pre-check      → LLM definitional node
  │
  ▼
Reinforcement            strength += REINFORCE_MAX_DELTA · relevance  (success)
                         strength −= REINFORCE_MIN_DELTA · relevance  (failure)
  │
  ▼
Persist                  nodes.json — index grows and learns across sessions
```

---

## Quick Start

```python
from bap_engine import load_thought_space, BAPEngine, save_updated_nodes

# Load a domain
space  = load_thought_space("fibonacci_reasoning")
engine = BAPEngine(space)

# Run a query
result = engine.run("What is F(7)?")
print(result.summary())

# Persist learned strengths
save_updated_nodes(space)
```

**Multi-domain (auto-routing):**

```python
from bap_engine import MultiBAPEngine

multi  = MultiBAPEngine()                        # loads all 3 domains
result = multi.run("What is F(7)?")              # auto-routes to fibonacci
result = multi.run("What is 17 mod 5?")          # auto-routes to arithmetic
result = multi.run("Who discovered penicillin?") # auto-routes to multihop_facts
multi.save_all()
```

---

## Installation

```bash
pip install sentence-transformers numpy faiss-cpu
```

Requires Python 3.10+. No GPU required.

---

## File Structure

```
bap_engine.py                    Main engine — all logic lives here
populate_thought_space.py        Build ThoughtSpaces from curated node definitions
shsrs/
  engine.py                      SHSRS index implementation
thought_space_fibonacci_reasoning/
  nodes.json                     Persisted node graph (grows across sessions)
  [SHSRS index files]
thought_space_multihop_facts/
  nodes.json
  [SHSRS index files]
thought_space_arithmetic/
  nodes.json
  [SHSRS index files]
test_groq_backend.py             Real-LLM validation suite (4 tests)
test_all_domains.py              3-domain benchmark
```

---

## Domains

| Domain | Nodes | Composition | Avg TRR | Hit Rate |
|--------|-------|-------------|---------|----------|
| `fibonacci_reasoning` | 206+ | Fibonacci properties, Lucas numbers, Pisano period, Binet formula, phi, Pascal's triangle | 0.204 | 100% |
| `arithmetic` | 203+ | Modular arithmetic, prime factorisation, GCD, sequences, number theory | 0.251 | 100% |
| `multihop_facts` | 235+ | Science, history, geography, technology, arts — multi-hop fact chains | 0.370 | 100% |

**TRR** (Token Reduction Ratio) = BAP steps / CoT tokens. Target < 0.40. Lower is better.

---

## Benchmarks

### Estimated TRR vs CoT baseline (30–80 token estimates)

| Domain | Avg TRR | Hit Rate | Status |
|--------|---------|----------|--------|
| fibonacci_reasoning | 0.204 | 100% | PASS |
| arithmetic | 0.251 | 100% | PASS |
| multihop_facts | 0.370 | 100% | PASS |

### True TRR vs real LLM (Groq llama-3.1-8b-instant, measured tokens)

| Domain | BAP steps | CoT tokens | True TRR |
|--------|-----------|------------|----------|
| fibonacci_reasoning | 5.1 avg | 349 avg | 0.0167 |
| arithmetic | 5.0 avg | 356 avg | 0.0191 |
| multihop_facts | 4.8 avg | 241 avg | 0.0273 |
| **Global** | **5.0 avg** | **316 avg** | **0.0210** |

**47.5× token reduction** over real CoT. BAP uses graph traversal steps; CoT uses generated tokens.

---

## LLM Backends

BAP calls the LLM only on retrieval miss, OOV detection, drift, or stagnation — never to generate the full path. The LLM produces one atomic primitive per call.

```python
# Mock (default — no API key, deterministic, for testing)
import bap_engine
# DYNAMIC_GEN_MODEL = "mock"  (default)

# Groq (free tier — recommended)
import os, bap_engine
os.environ["GROQ_API_KEY"] = "gsk_..."
bap_engine.DYNAMIC_GEN_MODEL = "groq"

# Ollama (local)
bap_engine.DYNAMIC_GEN_MODEL = "ollama"
bap_engine.DYNAMIC_GEN_OLLAMA_MODEL = "llama3.2"

# OpenAI
os.environ["OPENAI_API_KEY"] = "sk-..."
bap_engine.DYNAMIC_GEN_MODEL = "openai"

# Anthropic
os.environ["ANTHROPIC_API_KEY"] = "sk-ant-..."
bap_engine.DYNAMIC_GEN_MODEL = "anthropic"

# Custom / foundational model
from bap_engine import LLMBackend, register_backend

class MyModel(LLMBackend):
    def available(self) -> bool: return True
    def generate_primitive(self, prompt: str) -> str:
        return your_model.generate(prompt)

register_backend("my_model", MyModel())
bap_engine.DYNAMIC_GEN_MODEL = "my_model"
```

All backends implement exactly two methods: `available()` and `generate_primitive()`.

---

## Failure Policy

Control engine behaviour on edge-case failures:

```python
from bap_engine import BAPEngine, FailurePolicy

engine = BAPEngine(space, failure_policy=FailurePolicy.PARTIAL)   # default: silent best-effort
engine = BAPEngine(space, failure_policy=FailurePolicy.WARN)      # log warning, continue
engine = BAPEngine(space, failure_policy=FailurePolicy.ABSTAIN)   # raise ValueError on failure
```

Applied at: budget exhaustion with empty path, drift + stagnation both fired with end novelty < 0.10, OOV harm score 0.40–0.50, and low-confidence domain routing.

---

## Key Configuration

All constants are at the top of `bap_engine.py`:

```python
GLOBAL_BUDGET            = 1.0      # total energy per query
EMBED_MODEL              = "all-MiniLM-L6-v2"
SEARCH_K                 = 10       # top-k candidates per SHSRS search
SEARCH_PROBE_AUTO        = True     # auto-scale probe count to domain size
GOAL_ATTRACTION_WEIGHT   = 0.30     # blend query_vec into search vec each step
NOVELTY_WEIGHT           = 0.50     # novelty term weight in activation equation
OOV_THRESHOLD            = 0.38     # cosine below this = OOV unigram
BIGRAM_OOV_THRESHOLD     = 0.50     # tighter threshold for bigram OOV detection
DYNAMIC_GEN_MODEL        = "mock"   # LLM backend: mock | groq | ollama | openai | anthropic
DYNAMIC_GEN_MAX_PER_SESSION = 5     # max LLM calls per query
DYNAMIC_INDEX_REBUILD_EVERY = 0     # 0 = auto-scale; >0 = manual override
NODE_RETIRE_ENABLED      = True     # prune weak dynamic nodes across sessions
STRENGTH_CEIL            = 0.95     # hard ceiling — prevents gate saturation
STRENGTH_FLOOR           = 0.05     # hard floor
ROUTER_COLD_START_BOOST  = 1.25     # routing score multiplier for small domains
```

---

## Adding a New Domain

1. Define nodes in `populate_thought_space.py` — follow the existing domain structure.
2. Run `python populate_thought_space.py` to build the SHSRS index and `nodes.json`.
3. Register the domain in `MultiBAPEngine.ALL_DOMAINS`.

Node types: `fact` | `rule` | `function` | `constraint`

```python
{"id": 0, "text": "F(7) = 13.", "type": "fact", "strength": 0.5, "age": 0.0, "harm": 0.0}
```

---

## SHSRS — The Search Index

BAP uses **SHSRS** (Spherical Hierarchical Semantic Retrieval System) as its retrieval layer — a custom high-performance ANN index built for sentence embeddings.

- Architecture: KMeans partition → IGAR boundary refinement → per-cluster HNSW (FAISS)
- Benchmarked at 95.1% Recall@10 on 150K Wikipedia vectors at 1.75ms per query
- Gap-adaptive centroid routing: probe count scales automatically with query confidence
- Published: DOI [10.5281/zenodo.18733029](https://doi.org/10.5281/zenodo.18733029)

---

## Running Tests

```bash
# 3-domain benchmark (no API key required)
python test_all_domains.py

# Real LLM validation (Groq)
set GROQ_API_KEY=gsk_...
python test_groq_backend.py

# Syntax check
python -c "import ast; ast.parse(open('bap_engine.py').read()); print('OK')"
```

---

## Session 8 — Hardening Features

Production-hardening added in Session 8:

- **Auto-scaling SHSRS probe and cluster count** — `_compute_search_probe()` scales 2→20 by domain size; `_compute_n_clusters()` prevents MiniBatchKMeans ghost centroids
- **Adaptive index rebuild interval** — `_compute_rebuild_interval()` scales 5→20 steps by corpus size
- **FailurePolicy enum** — WARN / ABSTAIN / PARTIAL at all failure points
- **Node retirement** — weak dynamic nodes (strength < 0.15, age > 5.0) pruned automatically
- **Embedding model version guard** — `load_thought_space()` raises `RuntimeError` on model mismatch
- **Strength renormalization** — `_renormalize_strengths()` clamps to [0.05, 0.95]; compresses or lifts when >20% of nodes saturate at ceiling or floor
- **Cross-session text dedup** — hash index prevents duplicate dynamic node injection across sessions; persisted in `nodes.json`
- **Atomic writes** — `nodes.json` written via `.tmp` → `os.replace()` with `.bak` backup
- **Index/nodes.json sync validation** — count mismatch > 5 triggers rebuild; > 20% raises `RuntimeError`
- **OOV prompt sanitization** — strips non-alphanumeric characters, truncates to 60 chars, blocks 7 injection keywords before any term reaches an LLM prompt
- **Domain bleed detection** — logs warning when top-1/top-2 routing margin < 0.15
- **Cold-start domain boost** — domains with fewer than 100 nodes receive a ×1.25 routing score multiplier
- **Concurrent write protection** — platform-specific file lock (fcntl on POSIX, msvcrt on Windows) with 5-second timeout; uses a sentinel `.json.lck` file to avoid same-process read conflicts

---

## Design Principles

1. **BAP is the reasoning engine. The LLM is fuel.** The LLM generates one atomic primitive on demand. It never generates the path.
2. **The activation equation is the physics.** It is never replaced or approximated.
3. **The budget is real.** `GLOBAL_BUDGET = 1.0`. It is never inflated.
4. **Every backend implements exactly two methods.** `available()` and `generate_primitive()`. Nothing more.
5. **Persistence is mandatory.** `save_updated_nodes()` runs after every session. The graph learns.

---

## License

Copyright 2026 Chandan S H

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for the full text.
