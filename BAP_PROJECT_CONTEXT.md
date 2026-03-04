# BAP Project Context
## For Claude Code — Read This First Before Touching Anything

---

## What This Project Is

BAP (Budgeted Activation Propagation) is a reasoning engine built as an alternative to LLM chain-of-thought generation.

Instead of generating reasoning token-by-token with an LLM, BAP traverses a structured ThoughtSpace — a graph of atomic reasoning primitives — using an energy-budget-controlled activation equation. The result is deterministic, interpretable, fast reasoning at 5-8x lower token cost than standard CoT.

The long-term goal: BAP as a kernel for foundational AI. The LLM is not the reasoning engine — it is the fuel (abstraction generator) that BAP governs.

---

## Core Architecture

    Query
      |
      v
    Query Expansion (5 reformulations, averaged embedding)
      |
      v
    OOV Detection (scan query terms against corpus, define missing concepts first)
      |
      v
    SHSRS Index Search (Spherical Hierarchical Semantic Retrieval System)
      |
      v
    Activation Scoring:
      A(j) = A_source * sim(i,j) * gate(j) * decay(j) * log(1+budget) * novelty(j)
      gate(j)    = sigmoid(GATE_SCALE * (strength_j - harm_j))
      decay(j)   = exp(-age_j * LAMBDA_DECAY)
      novelty(j) = 1 - max_cosine(embed_j, any path node)
      |
      v
    Pruning Gates:
      harm cosine > 0.5     -> prune
      contradiction < -0.15 -> prune
      activation < 0.008    -> prune
      |
      v
    Competitive Inhibition:
      budget > 50%  -> top-2 winners (explore)
      budget <= 50% -> top-1 winner  (exploit)
      |
      v
    Retrieval Miss? -> Dynamic Generation
      OOV pre-check: define missing concept BEFORE traversal starts
      Miss fallback: generate contextual continuation node
      |
      v
    Outcome-Weighted Reinforcement (post-query, query-relevance weighted):
      relevance     = max(0, cosine(node.embed, query_vec))
      success=True  -> activated nodes strength += REINFORCE_MAX_DELTA * relevance
      success=False -> activated nodes strength -= REINFORCE_MIN_DELTA * relevance
      |
      v
    Persist to disk (nodes.json) — index grows session-to-session

---

## File Structure

    project/
    |-- bap_engine.py              <- THE MAIN FILE. All engine logic lives here.
    |-- populate_thought_space.py  <- builds initial ThoughtSpace from node defs
    |-- shsrs/
    |   |-- engine.py              <- SHSRS index (do not modify carelessly)
    |-- thought_space_fibonacci_reasoning/
    |   |-- nodes.json             <- persisted nodes (grows over sessions; includes macro fields)
    |   |-- path_registry.json     <- path frequency tracker (macro session)
    |   |-- [SHSRS index files]
    |-- thought_space_multihop_facts/
    |   |-- nodes.json
    |   |-- path_registry.json
    |   |-- [SHSRS index files]
    |-- thought_space_arithmetic/
    |   |-- nodes.json             <- 3rd domain (200+ nodes)
    |   |-- path_registry.json
    |   |-- [SHSRS index files]
    |-- test_groq_backend.py       <- real-LLM validation suite (4 tests)

---

## Key Classes in bap_engine.py

LLMBackend (ABC, line ~273)
  Abstract interface. Every LLM backend implements exactly TWO methods:
    available() -> bool
    generate_primitive(prompt: str) -> str
  Do NOT add methods to this interface. BAP only calls these two.

MockBackend (line ~287)
  Testing backend. No network. No API keys.
  Uses curated pools + semantic similarity for realistic behaviour.
  Extra methods for DynamicGenerator: generate_from_context(), define_term()

OllamaBackend (line ~364)
  Local LLM via Ollama. Configurable model, host, temperature, max_tokens.

OpenAIBackend (line ~396)
  OpenAI API. Reads OPENAI_API_KEY from env or constructor.

AnthropicBackend (line ~427)
  Claude API. Reads ANTHROPIC_API_KEY from env or constructor.

GroqBackend (line ~460)
  Groq free-tier API (llama-3.1-8b-instant). OpenAI-compatible endpoint.
  Reads GROQ_API_KEY from env or constructor. No extra dependencies (uses urllib).
  Validated end-to-end: OOV injection + retrieval miss paths both confirmed working.

register_backend(name, backend) (line ~490)
  Runtime plug-in. Register any model:
    from bap_engine import LLMBackend, register_backend
    import bap_engine
    class MyModel(LLMBackend):
        def available(self): return True
        def generate_primitive(self, prompt): return my_model.call(prompt)
    register_backend("my_model", MyModel())
    bap_engine.DYNAMIC_GEN_MODEL = "my_model"

DynamicGenerator (line ~520)
  Thin orchestration. Session counting, harm filtering, JSON parsing, backend routing.
  Does NOT contain LLM logic — that belongs in the backend.

BAPEngine (line ~615)
  The core engine. Key methods:
    run(query, max_steps, confidence_threshold, expected_contains)
    _detect_oov_term(query)
    _expand_query(query)
    _infer_type_bias(query)
    _activation(source, sim, node, path_embeds)  <- THE CORE EQUATION
    _gate(), _decay(), _energy(), _novelty(), _is_contradiction()
    _update_node_stats(success)
    _add_dynamic_node(), _rebuild_index(), _brute_force_search()

DomainRouter (after BAPEngine)
  Routes queries to the best-matching ThoughtSpace.
  Strategy: k=1 SHSRS search per registered domain; pick domain with highest top-1 cosine score.
    register(space)
    route(query, verbose=True) -> (domain_name, space, score)

MultiBAPEngine (after DomainRouter)
  Single entry point across all domains. Loads one BAPEngine per domain.
    ALL_DOMAINS = ["fibonacci_reasoning", "arithmetic", "multihop_facts"]
    run(query, max_steps, **kwargs) -> BAPResult   (auto-routes via DomainRouter)
    save_all()                                     (persists all domains)
    spaces  (property -> Dict[str, ThoughtSpace])

---

## Config Constants (top of bap_engine.py)

    EMBED_MODEL          = "all-MiniLM-L6-v2"
    GLOBAL_BUDGET        = 1.0
    LAMBDA_DECAY         = 0.01
    GATE_SCALE           = 3.0
    NOVELTY_WEIGHT       = 0.50
    CONTRADICTION_THRESH = -0.15
    EXPLORE_K            = 2        # beam width while budget > 50%
    EXPLOIT_K            = 1        # beam width while budget <= 50%
    MIN_ACTIVATION       = 0.008
    SEARCH_K             = 10        # raised from 5 for wider candidate pool on 200+ node domains
    SEARCH_PROBE         = 12       # probe 12 clusters; clips to n_clusters on small domains
    GOAL_ATTRACTION_WEIGHT = 0.30   # blend query_vec into search_vec each step (prevent drift)
    OOV_THRESHOLD        = 0.38     # unigram: cosine below this = OOV term
    BIGRAM_OOV_THRESHOLD = 0.50     # bigrams need tighter corpus match to be considered in-vocab
    REINFORCE_MAX_DELTA  = 0.08     # max strength delta on success (at relevance = 1.0)
    REINFORCE_MIN_DELTA  = 0.02     # max strength delta on failure (at relevance = 1.0)

    DYNAMIC_GEN_ENABLED         = True
    DYNAMIC_GEN_MODEL           = "mock"  # -> ollama / openai / anthropic / groq / custom
    DYNAMIC_GEN_OLLAMA_MODEL    = "llama3.2"
    DYNAMIC_GEN_OPENAI_MODEL    = "gpt-3.5-turbo"
    DYNAMIC_GEN_ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"
    DYNAMIC_GEN_GROQ_MODEL      = "llama-3.1-8b-instant"
    DYNAMIC_GEN_MAX_PER_SESSION = 5
    DYNAMIC_INDEX_REBUILD_EVERY = 0         # 0 = auto via _compute_rebuild_interval(); >0 = override

    # Auto-scaling (Session 8)
    SEARCH_PROBE_AUTO    = True             # dynamically scale SHSRS probe count to domain size
    ROUTER_MARGIN_WARN_THRESHOLD     = 0.15 # top1-top2 score margin below this -> domain bleed risk
    ROUTER_COLD_START_THRESHOLD      = 100  # domains with fewer nodes get routing score boost
    ROUTER_COLD_START_BOOST          = 1.25 # cold-start multiplier
    NODE_RETIRE_ENABLED      = True
    NODE_RETIRE_MIN_AGE      = 5.0         # DYN node must have aged at least this before retirement
    NODE_RETIRE_MAX_STRENGTH = 0.15        # DYN node must be below this strength to be retired
    STRENGTH_CEIL               = 0.95
    STRENGTH_FLOOR              = 0.05
    STRENGTH_COMPRESS_THRESHOLD = 0.20     # >20% nodes at ceiling -> compress all by 0.90
    STRENGTH_LIFT_THRESHOLD     = 0.20     # >20% nodes at floor -> lift all by +0.05

    # Macro node compilation (Macro Session)
    MACRO_ENABLED            = True
    MACRO_MIN_FREQUENCY      = 3       # path must be seen >= this many times to compile
    MACRO_MIN_LENGTH         = 3       # path must have >= this many nodes to compile
    MACRO_MIN_SUCCESS_RATE   = 0.95   # path must succeed >= 95% of runs to compile
    MACRO_MATCH_THRESHOLD    = 0.82   # anchor_vec cosine threshold for fuzzy macro retrieval
    MACRO_SCORE_BOOST        = 1.35   # activation score multiplier for matched macro candidates
    PATH_REGISTRY_MAX_VECS   = 10     # max stored query_vecs per path registry entry

---

## What Has Been Built and Validated

Core Engine
  [x] Activation propagation: sim * gate * decay * energy * novelty
  [x] Contradiction gate (hard prune, cosine < -0.15)
  [x] Harm gate (hard prune, cosine > 0.5)
  [x] Novelty penalty (weight 0.50)
  [x] Budget control: explore k=2 above 50%, exploit k=1 below 50%
  [x] Query expansion (5 reformulations, averaged vector)
  [x] Type-bias first hop
  [x] Text + ID deduplication

Memory and Learning
  [x] Outcome-weighted reinforcement (success +0.08, failure -0.02)
  [x] Post-session strength and age persisted to nodes.json
  [x] Index grows across sessions
  [x] Node counter fixed (+N new)
  [x] Frequency bias fix: _update_node_stats(success, query_vec) weights each node's delta by cosine(node.embed, query_vec). Hub nodes activated on tangential paths receive near-zero update; directly relevant nodes receive full REINFORCE_MAX/MIN_DELTA. Backward-compatible: query_vec=None -> relevance=1.0 (uniform).
  [x] REINFORCE_MAX_DELTA=0.08, REINFORCE_MIN_DELTA=0.02 in config

Dynamic Generation
  [x] Retrieval miss detection and fallback
  [x] OOV detection: proper nouns + bigrams + technical suffixes
  [x] OOV pre-check before traversal (define + 0.6/0.4 re-embed)
  [x] Brute-force first hop after OOV injection
  [x] Single-attempt guard (_last_gen_failed)
  [x] Generated node strength = 0.65
  [x] GEN_MIN_ACTIVATION = MIN_ACTIVATION * 0.5
  [x] Generation boost 1.4x
  [x] OOV bigram ranking fix: bigrams always ranked before unigrams in OOV scan
  [x] BIGRAM_OOV_THRESHOLD = 0.50 (tighter than unigram 0.38 — prevents false in-vocab)
  [x] False OOV filter: only proper nouns + tech suffixes pass bigram filter (no catch-all)
  [x] OOV surfacing fix: brute-force step uses OOV node's own embed as search_vec (sim=1.0 wins)
  [x] _oov_node_id lifecycle: initialised in __init__, reset per run(), cleared after step 1

Search and Retrieval
  [x] Goal attraction: search_vec = 0.70*current_vec + 0.30*query_vec each step
  [x] _search() docstring placement fixed (was misplaced after force_brute block)
  [x] SHSRS cluster formula: N//20 replaces N//8 — avoids ghost centroids in MiniBatchKMeans
  [x] SEARCH_PROBE raised to 12 — covers all 10 clusters on 200-node domain (was 2, covered only 8% of corpus)
  [x] SEARCH_K raised 5->10 — wider candidate pool prevents correct node being excluded on 200+ node domains
  [x] _expand_query restructured — 4 separated branches: next/after value intent, binet phi/psi expansion, pascal diagonal, recurrence exact fragment. Prevents branch collision on large corpus.
  [x] _infer_type_bias updated — pascal/diagonal -> fact, next/after+number/value/term -> fact, binet checked before generic formula catch-all
  [x] SHSRS boundary dedup fix: scored_ids set added to candidate scoring loop. SHSRS returns some nodes twice (cluster boundary duplicates); without dedup, both slots of EXPLORE_K=2 were filled by the same node, leaving the second unique winner unreachable. scored_ids prevents any node_id from entering scored[] more than once per step.

Multi-Domain Routing
  [x] DomainRouter: register() + route() -> (domain, space, score, low_confidence_bool, routing_margin)  # 5-tuple from Session 8
  [x] MultiBAPEngine: loads all 3 domains, auto-routes, exposes run(**kwargs) + save_all()
  [x] ROUTER_LOW_CONFIDENCE_THRESHOLD = 0.30 — low-confidence route handled via FailurePolicy
  [x] Routing accuracy 7/7 after corpus expansion (telephone: 0.763 multihop, 17 mod 5: 0.781 arithmetic)
  [x] test_multi_domain.py — 7-query validation covering all 3 domains
  [x] Domain bleed detection: log if margin < ROUTER_MARGIN_WARN_THRESHOLD = 0.15 (Session 8)
  [x] Cold-start boost: domains with n_nodes < 100 get ×1.25 routing score (Session 8)

Corpus Scaling
  [x] fibonacci_reasoning: 48 -> 200 nodes (80 fact / 50 rule / 40 fn / 30 constraint). Unicode fixed (phi, sqrt, psi).
  [x] multihop_facts: 43 -> 200 nodes (100 fact / 50 rule / 30 fn / 20 constraint). Technology, science, history, arts.
  [x] arithmetic: 200 -> 203 nodes (added 3 mod-specific anchors to prevent cross-domain routing regression)

Drift Pattern Detection
  [x] _detect_drift(path_embeds): mean pairwise cosine of last DRIFT_WINDOW=3 nodes; fires if > DRIFT_THRESHOLD=0.85
  [x] DRIFT_WINDOW=3, DRIFT_THRESHOLD=0.85 in config
  [x] Drift triggers LLM bridge injection immediately (action="drift_bridge"), even without retrieval miss
  [x] _drift_injected guard: fires at most once per run() call
  [x] Path stagnation detection — _detect_stagnation(recent_novelties, recent_types): fires only when BOTH conditions hold simultaneously over last STAGNATION_WINDOW=4 steps: mean novelty < STAGNATION_NOV_FLOOR=0.15 AND repeated node types
  [x] STAGNATION_WINDOW=4, STAGNATION_NOV_FLOOR=0.15 in config
  [x] Stagnation triggers LLM bridge injection (action="stagnation_bridge"), guard _stagnation_injected fires at most once per run()
  [x] Ordering: drift check fires first (semantic loop), stagnation fires second (novelty plateau) — mutually exclusive in same step
  [x] _detect_stagnation is purely functional (no state, takes only lists) — trivially testable
  [x] Novelty captured before node joins path_embeds — correctly measures novelty relative to existing path

Stagnation Pattern Detection
  [x] _detect_stagnation(recent_novelties, recent_types): BOTH mean novelty < STAGNATION_NOV_FLOOR=0.15 AND all types identical in last STAGNATION_WINDOW=4 nodes
  [x] STAGNATION_WINDOW=4, STAGNATION_NOV_FLOOR=0.15 in config
  [x] Stagnation fires after drift check, before SHSRS search; injects bridge (action="stagnation_bridge") with 1.4x boost
  [x] _stagnation_injected guard: fires at most once per run() call
  [x] path_novelties + path_types tracked per accepted node in run(); stagnation bridge node also appended to both lists
  [x] Distinct from drift: drift = semantic looping (embedding similarity); stagnation = novelty plateau + type mono-culture

Plug-and-Play LLM Layer
  [x] LLMBackend ABC (2 methods only)
  [x] MockBackend, OllamaBackend, OpenAIBackend, AnthropicBackend
  [x] GroqBackend (free-tier, llama-3.1-8b-instant, stdlib urllib only)
  [x] register_backend() for runtime custom model registration
  [x] _resolve_backend() router
  [x] DynamicGenerator decoupled from backend
  [x] Real LLM validated end-to-end: OOV path + retrieval miss path (Groq)

Macro Reasoning Nodes (Macro Session)
  [x] Path Frequency Tracker: path_registry in ThoughtSpace — keyed by "|"-joined node IDs, stores count/successes/last 10 query_vecs/last_seen. Persisted to path_registry.json per domain via _save_path_registry() (atomic write).
  [x] Macro Node Type: ThoughtNode extended with is_macro, subpath_ids, macro_freq, anchor_vec. Type "macro" accepted by DynamicGenerator._parse(). Macros are is_static=True (never retired) + explicit `not node.is_macro` guard in _retire_weak_nodes().
  [x] Macro Compilation: _compile_macros() — fires after every run(). Qualifies paths with count>=3, length>=3, success_rate>=0.95. Depth-check: rejects paths with nested macro depth >2. LLM summarizes chain via _MACRO_PROMPT (<20 words). Falls back to "Compiled: {first_node[:40]}...". Compiled macro strength = max(0.70, min(0.90, mean_constituent_strengths)).
  [x] Fuzzy Macro Matching: _search() scans macro nodes by anchor_vec cosine against query_vec. Threshold MACRO_MATCH_THRESHOLD=0.82. Matched macros appended to SHSRS candidates with MACRO_SCORE_BOOST=1.35.
  [x] Macro Expansion on Hit: when macro wins step 1, its subpath_ids expanded into path as action="macro_expanded". Budget cost = len(subpath)*(GLOBAL_BUDGET/max_steps)*0.5 (half-cost). current_vec updated to last subpath node embed.
  [x] Macro Invalidation Watcher: _validate_macros() called at run() start. Invalidates macros whose constituent nodes are retired/missing (strength=0, moved to _retired_node_ids). Recomputes strength as mean constituent strengths if drifted below 0.50.
  [x] Nested Macro Depth Limit: _compile_macros() checks each constituent — if constituent.is_macro, counts depth. Reject if depth would reach 3+.
  [x] Macro Statistics: BAPResult.macro_hits (expanded macros this run) + macros_available (total in space). summary() shows "Macro hits: N | Macro nodes in space: M".
  [x] MultiBAPEngine integration: save_all() calls _save_path_registry(engine.space) per domain. Reports total macro count. run() triggers macro logic implicitly via BAPEngine.run().
  [x] _save_path_registry(): module-level function, atomic write (.tmp->os.replace()), same pattern as save_updated_nodes().
  [x] save_updated_nodes() updated: persists is_macro/subpath_ids/macro_freq/anchor_vec for existing nodes. New nodes use asdict() which already captures all dataclass fields.
  [x] load_thought_space() updated: loads macro fields from nodes.json; loads path_registry.json if present (empty dict if absent).

Hardening (Session 8)
  [x] Auto-scaling SEARCH_PROBE: _compute_search_probe(n) = 2/6/12/20 by domain size; SEARCH_PROBE_AUTO=True
  [x] Auto-scaling n_clusters: _compute_n_clusters(n) = min(max(n//20,5),50); used in _rebuild_index()
  [x] Adaptive rebuild interval: _compute_rebuild_interval(n) = 5/10/20 by domain size; DYNAMIC_INDEX_REBUILD_EVERY=0 for auto
  [x] FailurePolicy enum: WARN / ABSTAIN / PARTIAL (default). Applied at: budget exhaustion with empty path, drift+stagnation both fired+novelty<0.10, OOV harm 0.40–0.50, low-confidence routing
  [x] Node retirement: _retire_weak_nodes() removes DYN nodes (is_static=False) with strength<0.15 AND age>5.0. Never removes static corpus nodes. IDs tracked in _retired_node_ids, marked retired=true in JSON.
  [x] Embedding model version check: load_thought_space() raises RuntimeError on embed_model mismatch. Writes model name on first load.
  [x] Strength renormalization: _renormalize_strengths() hard-clamps to [0.05,0.95], compresses by 0.90 if >20% above 0.90, lifts by +0.05 if >20% below 0.10. Called after every _update_node_stats().
  [x] Cross-session text dedup: hash(text.lower().strip()) checked before _add_dynamic_node(). Persisted as text_hashes array in nodes.json. Duplicate injections silently skipped.
  [x] Atomic write: save_updated_nodes() writes to .tmp then os.replace() (atomic POSIX+Windows). Keeps .bak backup.
  [x] Index/nodes.json sync validation: count mismatch >5 triggers rebuild; >20% raises RuntimeError.
  [x] OOV prompt sanitization: _sanitize_oov_term() strips non-[a-zA-Z0-9 _-], truncates to 60 chars, rejects 7 injection keywords. Called before any OOV term enters a prompt.
  [x] Domain bleed detection: routing margin = top1-top2 score; logged if < 0.15 with both domain scores.
  [x] Cold-start boost: domains with n_nodes<100 get ×1.25 routing score. Prevents large domains drowning small new ones.
  [x] Concurrent write protection: sentinel lock file (nodes.json.lck), platform-specific lock (fcntl POSIX / msvcrt Windows), 5s timeout, FailurePolicy on lock failure.

Validated Benchmarks
  Fibonacci domain:  avg TRR 0.204  |  hit rate 100%  — PASS  (Session 8)
  Arithmetic domain: avg TRR 0.251  |  hit rate 100%  — PASS  (Session 8)
  Multihop domain:   avg TRR 0.370  |  hit rate 100%  — PASS  (Session 8)
  All domains:       TRR < 0.40     |  hit rate 100%  — ALL PASS
  Target:            TRR < 0.40     |  hit rate > 95%
  NOTE: above TRR uses estimated CoT tokens (30–80). True TRR vs real LLM below.

  True CoT Baseline  (llama-3.1-8b-instant, "Think step by step and answer: {query}")
  Fibonacci domain:  avg BAP 5.1 steps  |  avg CoT 349.3 tokens  |  True TRR 0.0167
  Arithmetic domain: avg BAP 5.0 steps  |  avg CoT 356.4 tokens  |  True TRR 0.0191
  Multihop domain:   avg BAP 4.8 steps  |  avg CoT 241.2 tokens  |  True TRR 0.0273
  GLOBAL:            avg BAP 5.0 steps  |  avg CoT 315.6 tokens  |  True TRR 0.0210
  Improvement over real CoT: 47.5x token reduction
  Results saved: benchmark_cot_results.json

---

## What the LLM Layer Is For

BAP has hit the semantic horizon of purely local propagation.
The engine is stable. The missing piece is generative abstraction capacity.

  BAP = scheduler / control layer
  LLM = abstraction engine / fuel

LLM is called ONLY when:
  1. OOV concept detected in query
  2. Retrieval miss (no candidate above threshold)
  3. Drift pattern detected (semantic looping)
  4. Path stagnation detected (novelty plateau + type mono-culture)

LLM outputs ONE atomic rule OR ONE atomic bridge. No paragraphs. No CoT.
Small injections. Controlled fuel.

DO NOT replace BAP with LLM.
DO NOT let LLM generate the full reasoning path.

---

## Pending Work

  [x] Real LLM backend validation — done (Groq, llama-3.1-8b-instant)
  [x] Goal attraction term — done (GOAL_ATTRACTION_WEIGHT = 0.30)
  [x] Domain scaling — arithmetic expanded 45 -> 200 nodes (80 fact / 50 rule / 40 fn / 30 constraint). Hit rate 100%, TRR 0.040.
  [x] Multi-domain routing — DomainRouter + MultiBAPEngine. 7/7 routing. ROUTER_LOW_CONFIDENCE_THRESHOLD=0.30 via FailurePolicy.
  [x] Drift pattern detection — _detect_drift(path_embeds): mean pairwise cosine of last DRIFT_WINDOW=3 nodes > DRIFT_THRESHOLD=0.85 triggers LLM bridge injection. _drift_injected guard (once per run). action="drift_bridge" in steps_log.
  [x] Path stagnation detection — _detect_stagnation(recent_novelties, recent_types): mean novelty < 0.15 AND all same type in last 4 nodes. Injects stagnation_bridge (1.4x boost, once per run).
  [x] Frequency bias fix — _update_node_stats(success, query_vec): delta scaled by cosine(node.embed, query_vec). REINFORCE_MAX_DELTA=0.08, REINFORCE_MIN_DELTA=0.02. SHSRS scored_ids dedup fix bundled.
  [x] True CoT baseline — True TRR: fibonacci 0.0167, arithmetic 0.0191, multihop 0.0273, global 0.0210. 47.5x reduction. Results: benchmark_cot_results.json.
  [x] Session 8 foundational hardening — 13 tasks (auto-scaling, FailurePolicy, node retirement, embed version check, strength renorm, text dedup, atomic write, sync validation, OOV sanitization, domain bleed, cold-start boost, concurrent write protection). All benchmarks pass.
  [x] Macro-reasoning node session — 9 tasks. Self-optimizing path compression: frequent successful paths compiled into macro nodes. Fuzzy anchor_vec retrieval. Macro expansion on step 1 hit. Invalidation watcher. Nested depth limit (max 2). Macro stats in BAPResult. All 3 domains PASS (fibonacci 0.192, arithmetic 0.167, multihop 0.328, all 100% hit rate).
  [ ] Foundational model integration via register_backend()

---

## Rules Claude Code Must Not Violate

1. NEVER replace the activation equation.
   A(j) = A_source * sim * gate * decay * energy * novelty
   This is the physics. Breaking it breaks everything.

2. NEVER remove the budget system. GLOBAL_BUDGET = 1.0. Do not inflate it.

3. NEVER let LLM generate the full path.
   One node at a time, on miss or OOV only.

4. NEVER break the backend interface.
   LLMBackend has exactly 2 methods. Do not add required methods.

5. ALWAYS run syntax check after edits:
   python3 -c "import ast; ast.parse(open('bap_engine.py').read())"

6. ALWAYS preserve persistence. save_updated_nodes() must run after every session.

7. ALWAYS preserve the three activation thresholds:
   - MIN_ACTIVATION = 0.008 for static nodes
   - GEN_MIN_ACTIVATION = MIN_ACTIVATION * 0.5 for generated nodes
   - OOV nodes use brute-force first hop, not SHSRS

8. NEVER merge backends into DynamicGenerator.
   DynamicGenerator orchestrates. Backends implement.

---

## How to Switch to a Real LLM Backend

    # Ollama (local)
    import bap_engine
    bap_engine.DYNAMIC_GEN_MODEL = "ollama"
    bap_engine.DYNAMIC_GEN_OLLAMA_MODEL = "llama3.2"
    # prerequisite: ollama pull llama3.2

    # Anthropic Claude
    import os, bap_engine
    os.environ["ANTHROPIC_API_KEY"] = "sk-ant-..."
    bap_engine.DYNAMIC_GEN_MODEL = "anthropic"

    # Groq (free tier — recommended for testing, no install required)
    import os, bap_engine
    os.environ["GROQ_API_KEY"] = "gsk_..."
    bap_engine.DYNAMIC_GEN_MODEL = "groq"
    # optional: bap_engine.DYNAMIC_GEN_GROQ_MODEL = "llama-3.1-8b-instant"
    # run: python test_groq_backend.py   (validates all 4 paths)

    # Custom / foundational model
    from bap_engine import LLMBackend, register_backend
    import bap_engine

    class MyFoundationalModel(LLMBackend):
        def available(self): return True
        def generate_primitive(self, prompt: str) -> str:
            return your_model.generate(prompt, max_tokens=80, temperature=0.3)

    register_backend("my_model", MyFoundationalModel())
    bap_engine.DYNAMIC_GEN_MODEL = "my_model"

---

## The Three Prompts (Do Not Change Without Care)

_GENERATE_PROMPT
  Used on retrieval miss. Provides context path + last node.
  Asks for ONE atomic continuation.
  Returns JSON: {"text": "...", "type": "fact|rule|function|constraint"}
  Constraints: under 25 words, self-contained, no CoT.

_DEFINE_PROMPT
  Used on OOV detection. Provides missing term only (no context).
  Asks for ONE definitional primitive. Same JSON format.

_MACRO_PROMPT
  Used in _compile_macros() to summarize a compiled reasoning chain.
  Provides the full chain as newline-joined node texts.
  Asks for ONE summary sentence under 20 words.
  Returns JSON: {"text": "...", "type": "fact|rule|function|constraint"}
  Fallback if LLM fails: "Compiled: {first_node[:40]}..."

---

## Terminology

  ThoughtSpace    = graph of atomic reasoning nodes for one domain
  ThoughtNode     = one atomic primitive: text, type, embedding, strength, age, harm
  Node types      = fact | rule | function | constraint | macro
  Activation A(j) = energy score for candidate node j
  Gate            = sigmoid(strength - harm)
  Decay           = exp(-age * lambda)
  Novelty         = 1 - max cosine similarity to path nodes
  Budget          = energy remaining (starts 1.0)
  TRR             = Token Reduction Ratio = BAP steps / CoT tokens (target < 0.40)
  OOV             = Out-of-vocabulary term with no close corpus match
  SHSRS           = Spherical Hierarchical Semantic Retrieval System
  DYN / GEN       = dynamically generated node
  Miss            = retrieval miss, no candidate above MIN_ACTIVATION
  Backend         = LLM implementation behind the plug-and-play interface
  Macro           = compiled reasoning path node; is_macro=True, subpath_ids=list of constituent IDs, anchor_vec=mean query embedding of seen queries, macro_freq=hit count
  path_registry   = Dict[path_sig -> {count, successes, query_vecs, last_seen}]; persisted to path_registry.json

---

Last updated: 2026-03-04 (macro session)
Engine: BAP v1 — plug-and-play LLM layer, goal attraction, OOV fixes, Groq validated, all 3 domains at 200 nodes, multi-domain routing (7/7 accuracy), drift + stagnation detection, frequency bias fix (relevance-weighted reinforcement), SHSRS boundary dedup fix. True CoT baseline: 47.5x token reduction vs llama-3.1-8b-instant. Session 8 hardening: FailurePolicy enum, auto-scaling SHSRS probe/clusters/rebuild, node retirement, embed model version check, strength renormalization, cross-session text dedup, atomic write (.tmp→os.replace()), index sync validation, OOV prompt sanitization, domain bleed detection, cold-start boost, concurrent write lock (sentinel file). Benchmarks post-hardening: fibonacci 0.204 / arithmetic 0.251 / multihop 0.370, all 100% hit rate. Macro session: self-optimizing path compression, fuzzy macro retrieval, macro expansion on step 1, invalidation watcher, nested depth limit (max 2), macro stats in BAPResult. Benchmarks post-macro: fibonacci 0.192 / arithmetic 0.167 / multihop 0.328, all 100% hit rate.
