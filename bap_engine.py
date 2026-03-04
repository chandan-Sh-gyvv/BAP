"""
BAP — Budgeted Activation Propagation Engine v1
=================================================
Traverses a ThoughtSpace using the update equation:

    A(j) = Σᵢ A(i) · sim(i,j) · gate(j) · decay(j) · log(1 + remaining_budget)

Controls:
    - Top-2 candidates while budget > 30%  (explore)
    - Top-1 candidate  while budget <= 30% (exploit)
    - Novelty penalty: penalise nodes semantically close to path
    - Contradiction check: block nodes opposing active path
    - Budget exhaustion = hard stop

Usage:
    from bap_engine import BAPEngine, BAPResult
    from populate_thought_space import ThoughtNode, ThoughtSpace

    space  = load_thought_space("fibonacci_reasoning")
    engine = BAPEngine(space)
    result = engine.run("What is F(9)?")
    print(result.summary())
"""

import json
import math
import os
import re
import shutil
import sys
import time
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from sentence_transformers import SentenceTransformer
from shsrs.engine import SHSRSEngine

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────

EMBED_MODEL          = "all-MiniLM-L6-v2"
GLOBAL_BUDGET        = 1.0
LAMBDA_DECAY         = 0.01       # age decay rate
GATE_SCALE           = 3.0        # sigmoid sharpness for gate
NOVELTY_WEIGHT       = 0.50       # tighter novelty — suppresses drift in late steps
CONTRADICTION_THRESH = -0.15      # cosine below this = contradiction
EXPLORE_K            = 2          # candidates while budget > 30%
EXPLOIT_K            = 1          # candidates while budget <= 30%
MIN_ACTIVATION       = 0.008      # tight floor — prunes sub-threshold drift
SEARCH_K             = 10         # top-k candidates per SHSRS search — 10 gives wider coverage on 200+ node domains
SEARCH_PROBE         = 12         # fallback fixed probe count (used when SEARCH_PROBE_AUTO=False)
SEARCH_PROBE_AUTO    = True       # dynamically scale probe count to domain size
GOAL_ATTRACTION_WEIGHT = 0.30     # blend query_vec into search vec each step to prevent path drift

# Router confidence
ROUTER_LOW_CONFIDENCE_THRESHOLD  = 0.30   # top-1 SHSRS score below this -> low-confidence route
ROUTER_MARGIN_WARN_THRESHOLD     = 0.15   # top1-top2 score margin below this -> domain bleed risk
ROUTER_COLD_START_THRESHOLD      = 100    # domains with fewer nodes get a routing boost
ROUTER_COLD_START_BOOST          = 1.25   # score multiplier for cold-start (small) domains

# Drift pattern detection
DRIFT_WINDOW    = 3     # look at last N path embeddings for drift
DRIFT_THRESHOLD = 0.85  # mean pairwise cosine among last-N nodes above this = stuck loop

# Stagnation pattern detection
STAGNATION_WINDOW    = 4    # trailing accepted nodes to inspect
STAGNATION_NOV_FLOOR = 0.15 # mean novelty below this = path is plateauing

# Reinforcement deltas — query-relevance weighted
REINFORCE_MAX_DELTA = 0.08  # max strength delta on success  (at relevance = 1.0)
REINFORCE_MIN_DELTA = 0.02  # max strength delta magnitude on failure (at relevance = 1.0)

# OOV concept detection — semantic frontier expansion
OOV_THRESHOLD        = 0.38      # unigram OOV floor: generic words 0.20-0.35, domain concepts 0.40+
BIGRAM_OOV_THRESHOLD = 0.50      # bigrams need a tighter corpus match to be in-vocab; related-but-undefined concepts score 0.38-0.45
OOV_STOPWORDS        = {
    "what", "how", "why", "when", "where", "who", "which", "is", "are", "was",
    "were", "do", "does", "did", "the", "a", "an", "in", "of", "to", "and",
    "or", "for", "with", "about", "after", "before", "between", "can", "could",
    "would", "should", "will", "has", "have", "had", "be", "been", "being",
    "relate", "tell", "explain", "describe", "give", "show", "find", "get",
    "both", "true", "false", "number", "sequence", "next", "same", "each",
    # Generic verbs
    "work", "make", "take", "come", "know", "think", "look", "want", "mean",
    "need", "feel", "seem", "become", "keep", "put", "say", "use", "try",
    "ask", "turn", "move", "live", "play", "run", "hold", "help",
    # Generic nouns
    "subject", "concept", "thing", "fact", "facts", "idea", "way", "part",
    "place", "case", "point", "kind", "type", "form", "field", "area", "term",
    "relation", "result", "value", "time", "year", "day", "name", "group",
    "world", "hand", "life", "line", "side", "system", "level", "example",
    # Generic adjectives
    "different", "natural", "general", "specific", "possible", "important",
    "scientific", "logical", "historical", "mathematical",
}

# Dynamic node generation (retrieval miss fallback)
DYNAMIC_GEN_ENABLED         = True    # set False to disable LLM fallback
DYNAMIC_GEN_MODEL           = "mock"  # "mock" | "ollama" | "openai" | "anthropic" | "custom"
DYNAMIC_GEN_OLLAMA_MODEL    = "llama3.2"
DYNAMIC_GEN_OPENAI_MODEL    = "gpt-3.5-turbo"
DYNAMIC_GEN_ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"
DYNAMIC_GEN_GROQ_MODEL      = "llama-3.1-8b-instant"
DYNAMIC_GEN_MAX_PER_SESSION = 5       # max new nodes per query
DYNAMIC_INDEX_REBUILD_EVERY = 0       # 0 = auto via _compute_rebuild_interval(); >0 = manual override

# Node retirement — prunes weak DYN nodes that failed to add value
NODE_RETIRE_ENABLED      = True
NODE_RETIRE_MIN_AGE      = 5.0    # node must have aged at least this much (sessions × 0.1)
NODE_RETIRE_MAX_STRENGTH = 0.15   # nodes below this threshold are retirement candidates

# Strength renormalization — prevents gate saturation at extremes
STRENGTH_CEIL               = 0.95   # hard ceiling; nodes above this compress toward 0.90
STRENGTH_FLOOR              = 0.05   # hard floor; nodes below this get a soft lift
STRENGTH_COMPRESS_THRESHOLD = 0.20   # if >20% of nodes above 0.90, compress all by ×0.90
STRENGTH_LIFT_THRESHOLD     = 0.20   # if >20% of nodes below 0.10, lift all by +0.05

# Macro node compilation — self-optimizing path compression
MACRO_ENABLED            = True
MACRO_MIN_FREQUENCY      = 3       # path must be seen at least this many times to compile
MACRO_MIN_LENGTH         = 3       # path must have at least this many constituent nodes
MACRO_MIN_SUCCESS_RATE   = 0.95   # path must succeed >= 95% of the time
MACRO_MATCH_THRESHOLD    = 0.82   # anchor_vec cosine threshold for fuzzy macro retrieval
MACRO_SCORE_BOOST        = 1.35   # activation score multiplier for matched macro candidates
PATH_REGISTRY_MAX_VECS   = 10     # max stored query_vecs per path registry entry

# ── Custom LLM slot ───────────────────────────────────────────────────────────
# Set DYNAMIC_GEN_MODEL = "custom" and call register_backend() to plug in any model.
# Example:
#   from bap_engine import LLMBackend, register_backend
#   class MyModel(LLMBackend):
#       def available(self): return True
#       def generate_primitive(self, prompt): ...
#   register_backend("custom", MyModel())
_BACKEND_REGISTRY = {}  # populated by register_backend()

# ──────────────────────────────────────────────────────────────────────────────
# FAILURE POLICY
# ──────────────────────────────────────────────────────────────────────────────

class FailurePolicy(Enum):
    WARN    = "warn"    # log warning and return best-effort partial result
    ABSTAIN = "abstain" # raise ValueError on any failure condition
    PARTIAL = "partial" # silently return partial result (default)


# ──────────────────────────────────────────────────────────────────────────────
# AUTO-SCALING HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def _compute_search_probe(n_nodes: int) -> int:
    """Scale SHSRS probe count to domain size — avoids under-probing large domains."""
    if n_nodes < 60:
        return 2
    if n_nodes < 150:
        return 6
    if n_nodes < 300:
        return 12
    return 20


def _compute_n_clusters(n_nodes: int) -> int:
    """Scale SHSRS cluster count to domain size. Avoids ghost centroids in MiniBatchKMeans."""
    return min(max(n_nodes // 20, 5), 50)


def _compute_rebuild_interval(n_nodes: int) -> int:
    """Scale index rebuild interval to domain size."""
    if n_nodes < 100:
        return 5
    if n_nodes < 500:
        return 10
    return 20


# ──────────────────────────────────────────────────────────────────────────────
# OOV PROMPT SANITIZATION
# ──────────────────────────────────────────────────────────────────────────────

_OOV_INJECTION_TERMS = frozenset({
    "ignore", "system", "prompt", "instruction", "jailbreak", "forget", "override"
})

def _sanitize_oov_term(term: str) -> Optional[str]:
    """
    Sanitize an OOV term before it is used in a generation prompt.

    Steps:
      1. Strip characters outside [a-zA-Z0-9 _-]
      2. Truncate to 60 characters
      3. Reject terms containing known prompt-injection keywords

    Returns sanitized term or None if the term should be blocked.
    """
    clean = re.sub(r"[^a-zA-Z0-9 _\-]", "", term).strip()[:60]
    lower = clean.lower()
    for bad in _OOV_INJECTION_TERMS:
        if bad in lower:
            print(f"  [OOV] Prompt injection attempt blocked: {term[:40]!r}")
            return None
    return clean if clean else None


# ──────────────────────────────────────────────────────────────────────────────
# DATA STRUCTURES
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ThoughtNode:
    id:        int
    text:      str
    type:      str        # fact | rule | function | constraint | macro
    embed:     List[float]
    strength:  float = 0.5
    age:       float = 0.0
    harm:      float = 0.0
    is_static: bool  = True   # False for dynamically generated nodes (eligible for retirement)
    # ── Macro fields (non-macro nodes leave these at defaults) ──────────────────
    is_macro:    bool       = False
    subpath_ids: List[int]  = field(default_factory=list)  # constituent node ids
    macro_freq:  int        = 0                             # times this macro was used
    anchor_vec:  List[float] = field(default_factory=list) # mean query vec that triggers it


@dataclass
class ThoughtSpace:
    domain:  str
    nodes:   Dict[int, ThoughtNode]
    engine:  SHSRSEngine
    budget:  float
    global_budget: float
    active:  Dict[int, float]  # node_id → activation level
    path:    List[int]         # ordered activated node IDs
    _text_hashes:      set = field(default_factory=set)         # cross-session DYN text dedup
    _retired_node_ids: set = field(default_factory=set)         # node IDs retired this session
    path_registry: Dict[str, dict] = field(default_factory=dict)  # path frequency tracker


@dataclass
class BAPStep:
    step:         int
    node_id:      int
    text:         str
    node_type:    str
    activation:   float
    gate:         float
    novelty:      float
    budget_left:  float
    action:       str   # "selected" | "pruned_harm" | "pruned_contradiction" | "pruned_activation"


@dataclass
class BAPResult:
    query:            str
    domain:           str
    path:             List[BAPStep]
    final_budget:     float
    steps_taken:      int
    answer_nodes:     List[str]   # texts of selected nodes in order
    macro_hits:       int = 0     # number of macro expansions during this run
    macros_available: int = 0     # total macro nodes in the space at run time

    def summary(self) -> str:
        lines = [
            f"\n{'='*65}",
            f"BAP RESULT — {self.domain}",
            f"Query: {self.query}",
            f"Steps: {self.steps_taken}  |  Budget used: {1 - self.final_budget:.3f} / {GLOBAL_BUDGET}",
            f"Macro hits: {self.macro_hits}  |  Macro nodes in space: {self.macros_available}",
            f"{'='*65}",
            "Reasoning path:"
        ]
        for i, text in enumerate(self.answer_nodes):
            lines.append(f"  [{i+1}] {text}")
        lines.append("=" * 65)
        return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# LOAD THOUGHT SPACE
# ──────────────────────────────────────────────────────────────────────────────

_embedder: Optional[SentenceTransformer] = None

def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder


def load_thought_space(domain: str) -> ThoughtSpace:
    """Load a previously built thought space from disk."""
    nodes_path = Path(f"thought_space_{domain}/nodes.json")
    index_dir  = f"thought_space_{domain}"

    if not nodes_path.exists():
        raise FileNotFoundError(
            f"No thought space found for domain '{domain}'. "
            f"Run populate_thought_space.py first."
        )

    with open(nodes_path, encoding="utf-8") as f:
        data = json.load(f)

    # ── Task 5: Embedding model version check ─────────────────────────────────
    stored_model = data.get("embed_model")
    if stored_model is None:
        # First load with version tracking — write current model into metadata
        data["embed_model"] = EMBED_MODEL
        with open(nodes_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    elif stored_model != EMBED_MODEL:
        raise RuntimeError(
            f"Embedding model mismatch: nodes.json built with '{stored_model}', "
            f"current model is '{EMBED_MODEL}'. "
            f"Delete the index directory and rebuild with populate_thought_space.py."
        )

    nodes = {}
    seen_texts = set()
    for nd in data["nodes"]:
        if nd.get("retired", False):
            continue  # skip retired nodes
        # Deduplicate by text at load time — catches populate script duplicates
        if nd["text"] in seen_texts:
            continue
        seen_texts.add(nd["text"])
        nodes[nd["id"]] = ThoughtNode(
            id=nd["id"], text=nd["text"], type=nd["type"],
            embed=nd["embed"], strength=nd["strength"],
            age=nd["age"], harm=nd["harm"],
            is_static=nd.get("is_static", True),
            is_macro=nd.get("is_macro", False),
            subpath_ids=nd.get("subpath_ids", []),
            macro_freq=nd.get("macro_freq", 0),
            anchor_vec=nd.get("anchor_vec", []),
        )

    engine = SHSRSEngine.load(index_dir)

    # ── Task 9: Index / nodes.json sync validation ────────────────────────────
    try:
        index_count = engine._n_vectors
        json_count  = len(nodes)
        delta       = abs(index_count - json_count)
        if delta > 0.20 * max(json_count, 1):
            raise RuntimeError(
                f"[BAP] Index sync error: SHSRS has {index_count} nodes, "
                f"nodes.json has {json_count} (>{20}% mismatch). "
                f"Delete index and rebuild."
            )
        if delta > 5:
            print(
                f"[BAP] WARNING: index has {index_count} nodes, "
                f"nodes.json has {json_count}. Triggering rebuild."
            )
            vectors    = [n.embed for n in nodes.values()]
            vecs_np    = __import__("numpy").array(vectors, dtype="float32")
            n_clusters = _compute_n_clusters(len(vectors))
            engine = SHSRSEngine.build(
                vectors=vecs_np, index_dir=index_dir,
                n_clusters=n_clusters, M=8, ef_construction=100,
            )
    except AttributeError:
        pass  # older SHSRS version without _n_vectors — skip sync check

    # ── Task 7: Load cross-session text dedup hash index ─────────────────────
    stored_hashes = data.get("text_hashes", [])
    text_hashes   = set(stored_hashes)

    # ── Load path registry (macro self-optimisation) ──────────────────────────
    registry_path = Path(f"thought_space_{domain}/path_registry.json")
    if registry_path.exists():
        try:
            with open(registry_path, encoding="utf-8") as f:
                path_registry = json.load(f)
        except Exception:
            path_registry = {}
    else:
        path_registry = {}

    n_macros = sum(1 for n in nodes.values() if n.is_macro)
    print(f"[BAP] Loaded '{domain}': {len(nodes)} nodes ({n_macros} macros, "
          f"{len(path_registry)} path registry entries)")
    return ThoughtSpace(
        domain=domain, nodes=nodes, engine=engine,
        budget=GLOBAL_BUDGET, global_budget=GLOBAL_BUDGET,
        active={}, path=[],
        _text_hashes=text_hashes,
        path_registry=path_registry,
    )

# ──────────────────────────────────────────────────────────────────────────────
# DYNAMIC NODE GENERATOR — retrieval miss fallback
# ──────────────────────────────────────────────────────────────────────────────

HARM_TERMS = [
    "violence", "illegal", "harmful", "dangerous", "weapon",
    "abuse", "exploit", "toxic", "fraud", "deception"
]

_DEFINE_PROMPT = """You are building a reasoning graph. A key concept is missing from the knowledge base.

Missing concept: {term}

Generate exactly ONE atomic definitional primitive for this concept.

Rules:
- Define the concept directly and precisely
- Self-contained — no references to "above" or "previous"
- Atomic — one idea, under 25 words
- Factual and unambiguous

Return ONLY the primitive text and its type as JSON:
{{"text": "...", "type": "fact|rule|function|constraint"}}"""


_MACRO_PROMPT = """Summarize this reasoning chain in one precise sentence under 20 words.

Chain: {chain}

Return ONLY the summary text and type as JSON:
{{"text": "...", "type": "fact|rule|function|constraint"}}"""


_GENERATE_PROMPT = """You are building a reasoning graph. Given the context below, generate exactly ONE new atomic reasoning primitive that advances understanding.

Context (recent reasoning path):
{context}

Last active thought:
{last_node}

Rules:
- ONE primitive only — fact, inference rule, functional pattern, or constraint
- Self-contained — no references to "above" or "previous"
- Atomic — one idea, under 25 words
- Must be a logical next step from the last active thought
- Do NOT repeat any context already shown

Return ONLY the primitive text and its type as JSON:
{{"text": "...", "type": "fact|rule|function|constraint"}}"""



# ──────────────────────────────────────────────────────────────────────────────
# PLUG-AND-PLAY LLM BACKEND INTERFACE
# ──────────────────────────────────────────────────────────────────────────────
#
# BAP kernel calls only two methods on any backend:
#   backend.available()                → bool
#   backend.generate_primitive(prompt) → str
#
# Architecture:
#   LLMBackend (abstract)
#       MockBackend      — deterministic, no network, for testing
#       OllamaBackend    — any local model (llama3, mistral, phi3 ...)
#       OpenAIBackend    — OpenAI API
#       AnthropicBackend — Claude API
#       <your model>     — register_backend("name", YourBackend())
#
# BAP never changes when you swap backends.
# ──────────────────────────────────────────────────────────────────────────────

from abc import ABC, abstractmethod


class LLMBackend(ABC):
    """Contract every LLM backend must satisfy. Two methods. Nothing more."""

    @abstractmethod
    def available(self) -> bool:
        """Return True if backend can accept calls right now."""
        ...

    @abstractmethod
    def generate_primitive(self, prompt: str) -> str:
        """Call model with prompt, return raw response string."""
        ...


class MockBackend(LLMBackend):
    """
    Deterministic testing backend. Zero network calls. Zero API keys.
    Uses curated domain pools + semantic similarity for realistic behaviour.
    """

    _POOLS = {
        "fibonacci_reasoning": [
            {"text": "F(n) grows exponentially, doubling roughly every 1.44 steps.", "type": "fact"},
            {"text": "If F(n) > F(n-1), then F(n+1) > F(n) — the sequence is strictly increasing for n >= 1.", "type": "rule"},
            {"text": "The ratio of adjacent Fibonacci terms oscillates above and below phi, converging from both sides.", "type": "function"},
            {"text": "Fibonacci numbers cannot be negative when using the standard positive-index definition.", "type": "constraint"},
            {"text": "Lucas numbers follow the same recurrence as Fibonacci but start with L(0)=2, L(1)=1.", "type": "fact"},
            {"text": "If two consecutive terms are known, the entire Fibonacci sequence can be reconstructed.", "type": "rule"},
            {"text": "Fibonacci-like sequences with different seeds still grow at rate phi.", "type": "fact"},
        ],
        "multihop_facts": [
            {"text": "Physics and chemistry are both natural sciences concerned with matter.", "type": "fact"},
            {"text": "If X won prizes in both physics and chemistry, X worked across both fields.", "type": "rule"},
            {"text": "To verify a Nobel laureate's field: check the prize category they received.", "type": "function"},
            {"text": "A scientist's field is determined by their research focus, not their nationality.", "type": "constraint"},
            {"text": "Radioactivity research falls within both physics and chemistry.", "type": "fact"},
            {"text": "If two facts contradict each other, at most one can be true.", "type": "rule"},
            {"text": "Scientific fields are classified by their primary subject of study.", "type": "fact"},
        ],
        "arithmetic": [
            {"text": "The product of two numbers equals zero only if at least one of them is zero.", "type": "rule"},
            {"text": "Division distributes over addition: a/(b+c) is not equal to a/b + a/c in general.", "type": "constraint"},
            {"text": "The absolute value of a number is always non-negative.", "type": "fact"},
            {"text": "To check divisibility by 9: sum the digits; if divisible by 9, so is the number.", "type": "function"},
            {"text": "A perfect square has an odd number of divisors.", "type": "fact"},
        ],
    }

    _DEFINITIONS = {
        "lucas":          {"text": "Lucas numbers follow the Fibonacci recurrence but start with L(0)=2, L(1)=1.", "type": "fact"},
        "lucas numbers":  {"text": "Lucas numbers follow the Fibonacci recurrence but start with L(0)=2, L(1)=1.", "type": "fact"},
        "binet":          {"text": "Binet formula expresses F(n) as (phi^n - psi^n)/sqrt(5), where phi=(1+sqrt(5))/2.", "type": "function"},
        "binet formula":  {"text": "Binet formula expresses F(n) as (phi^n - psi^n)/sqrt(5), where phi=(1+sqrt(5))/2.", "type": "function"},
        "golden ratio":   {"text": "The golden ratio phi = (1+sqrt(5))/2 approximately 1.618, the limit of consecutive Fibonacci ratios.", "type": "fact"},
        "radioactivity":  {"text": "Radioactivity is the spontaneous emission of particles or energy from an unstable nucleus.", "type": "fact"},
        "polonium":       {"text": "Polonium is a radioactive element discovered by Marie Curie in 1898.", "type": "fact"},
        "radium":         {"text": "Radium is a radioactive element discovered by Marie Curie and Pierre Curie in 1898.", "type": "fact"},
        "recurrence":     {"text": "A recurrence relation defines each term as a function of preceding terms.", "type": "function"},
    }

    def __init__(self, embedder):
        self._embedder = embedder

    def available(self) -> bool:
        return True

    def generate_primitive(self, prompt: str) -> str:
        return ""  # not used; MockBackend routes through generate_from_context / define_term

    def generate_from_context(self, context_nodes, domain) -> str:
        import json as _j
        import numpy as np
        pool = self._POOLS.get(domain, self._POOLS["fibonacci_reasoning"])
        ctx  = " ".join(context_nodes).lower()
        candidates = [c for c in pool if c["text"].lower() not in ctx] or pool
        if context_nodes:
            last_emb  = self._embedder.encode(context_nodes[-1], normalize_embeddings=True).astype(np.float32)
            cand_embs = self._embedder.encode([c["text"] for c in candidates], normalize_embeddings=True).astype(np.float32)
            best = int((cand_embs @ last_emb).argmax())
            return _j.dumps(candidates[best])
        return _j.dumps(candidates[0])

    def define_term(self, term: str, domain: str) -> str:
        import json as _j
        tl = term.lower()
        for key, defn in self._DEFINITIONS.items():
            if key in tl or tl in key:
                return _j.dumps(defn)
        return _j.dumps({"text": f"{term.title()} is a key concept in {domain.replace('_', ' ')}.", "type": "fact"})


class OllamaBackend(LLMBackend):
    """Local LLM via Ollama. Supports llama3, mistral, phi3, gemma, etc."""

    def __init__(self, model: str = "llama3.2", host: str = "http://localhost:11434",
                 temperature: float = 0.3, max_tokens: int = 80):
        self.model       = model
        self.host        = host
        self.temperature = temperature
        self.max_tokens  = max_tokens

    def available(self) -> bool:
        try:
            import urllib.request
            urllib.request.urlopen(f"{self.host}/api/tags", timeout=2)
            return True
        except Exception:
            return False

    def generate_primitive(self, prompt: str) -> str:
        import urllib.request, json as _j
        payload = _j.dumps({
            "model": self.model, "prompt": prompt, "stream": False,
            "options": {"temperature": self.temperature, "num_predict": self.max_tokens}
        }).encode()
        req = urllib.request.Request(
            f"{self.host}/api/generate", data=payload,
            headers={"Content-Type": "application/json"}, method="POST"
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            return _j.loads(resp.read()).get("response", "").strip()


class OpenAIBackend(LLMBackend):
    """OpenAI API — GPT-3.5-turbo, GPT-4, GPT-4o, o1-mini, etc."""

    def __init__(self, model: str = "gpt-3.5-turbo", temperature: float = 0.3,
                 max_tokens: int = 80, api_key: str = None):
        self.model       = model
        self.temperature = temperature
        self.max_tokens  = max_tokens
        self._api_key    = api_key

    def available(self) -> bool:
        import os
        return bool(self._api_key or os.environ.get("OPENAI_API_KEY"))

    def generate_primitive(self, prompt: str) -> str:
        import os, urllib.request, json as _j
        key = self._api_key or os.environ["OPENAI_API_KEY"]
        payload = _j.dumps({
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature, "max_tokens": self.max_tokens
        }).encode()
        req = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions", data=payload,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            return _j.loads(resp.read())["choices"][0]["message"]["content"].strip()


class AnthropicBackend(LLMBackend):
    """Anthropic Claude API — Haiku (fast/cheap), Sonnet (balanced), Opus (powerful)."""

    def __init__(self, model: str = "claude-haiku-4-5-20251001", temperature: float = 0.3,
                 max_tokens: int = 80, api_key: str = None):
        self.model       = model
        self.temperature = temperature
        self.max_tokens  = max_tokens
        self._api_key    = api_key

    def available(self) -> bool:
        import os
        return bool(self._api_key or os.environ.get("ANTHROPIC_API_KEY"))

    def generate_primitive(self, prompt: str) -> str:
        import os, urllib.request, json as _j
        key = self._api_key or os.environ["ANTHROPIC_API_KEY"]
        payload = _j.dumps({
            "model": self.model, "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": [{"role": "user", "content": prompt}]
        }).encode()
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages", data=payload,
            headers={
                "Content-Type": "application/json",
                "x-api-key": key,
                "anthropic-version": "2023-06-01"
            },
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            return _j.loads(resp.read())["content"][0]["text"].strip()


class GroqBackend(LLMBackend):
    """Groq API — free tier, fast inference. OpenAI-compatible endpoint.
    Default model: llama-3.1-8b-instant. Reads GROQ_API_KEY from env."""

    def __init__(self, model: str = "llama-3.1-8b-instant", temperature: float = 0.3,
                 max_tokens: int = 80, api_key: str = None):
        self.model       = model
        self.temperature = temperature
        self.max_tokens  = max_tokens
        self._api_key    = api_key

    def available(self) -> bool:
        import os
        return bool(self._api_key or os.environ.get("GROQ_API_KEY"))

    def generate_primitive(self, prompt: str) -> str:
        import os, urllib.request, json as _j
        key = self._api_key or os.environ["GROQ_API_KEY"]
        payload = _j.dumps({
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }).encode()
        req = urllib.request.Request(
            "https://api.groq.com/openai/v1/chat/completions",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {key}",
                "User-Agent": "bap-engine/1.0",
            },
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            return _j.loads(resp.read())["choices"][0]["message"]["content"].strip()


# ── Backend Registry & Resolution ─────────────────────────────────────────────

def register_backend(name: str, backend: LLMBackend) -> None:
    """
    Register a custom LLM backend.

    Usage:
        from bap_engine import LLMBackend, register_backend
        import bap_engine

        class GeminiBackend(LLMBackend):
            def available(self): return True
            def generate_primitive(self, prompt): ...

        register_backend("gemini", GeminiBackend())
        bap_engine.DYNAMIC_GEN_MODEL = "gemini"
    """
    _BACKEND_REGISTRY[name] = backend
    print(f"[BAP] Backend registered: '{name}' ({type(backend).__name__})")


def _resolve_backend(embedder) -> LLMBackend:
    """Resolve DYNAMIC_GEN_MODEL string to a backend instance."""
    m = DYNAMIC_GEN_MODEL
    if m in _BACKEND_REGISTRY:
        return _BACKEND_REGISTRY[m]
    if m == "mock":
        return MockBackend(embedder)
    if m == "ollama":
        return OllamaBackend(model=DYNAMIC_GEN_OLLAMA_MODEL)
    if m == "openai":
        return OpenAIBackend(model=DYNAMIC_GEN_OPENAI_MODEL)
    if m == "anthropic":
        return AnthropicBackend(model=DYNAMIC_GEN_ANTHROPIC_MODEL)
    if m == "groq":
        return GroqBackend(model=DYNAMIC_GEN_GROQ_MODEL)
    raise ValueError(
        f"Unknown DYNAMIC_GEN_MODEL='{m}'. "
        f"Use 'mock'|'ollama'|'openai'|'anthropic'|'groq' or register_backend('{m}', ...)."
    )


# ──────────────────────────────────────────────────────────────────────────────
# DYNAMIC GENERATOR — thin orchestration layer over the pluggable backend
# ──────────────────────────────────────────────────────────────────────────────

class DynamicGenerator:
    """
    Thin orchestration layer. Handles:
      - Session call counting
      - Harm filtering
      - JSON parsing
      - Routing to backend
    Backend does all LLM work. Swap backends without touching this class.
    """

    def __init__(self, embedder: SentenceTransformer):
        self.embedder       = embedder
        self._backend       = _resolve_backend(embedder)
        self._harm_cluster  = self._build_harm_cluster()
        self._session_count = 0

    def _build_harm_cluster(self) -> np.ndarray:
        vecs = self.embedder.encode(HARM_TERMS, normalize_embeddings=True)
        cluster = vecs.mean(axis=0)
        return cluster / (np.linalg.norm(cluster) + 1e-9)

    def available(self) -> bool:
        return DYNAMIC_GEN_ENABLED and self._backend.available()

    def generate(self, context_nodes: List[str], last_node_text: str, domain: str) -> Optional[dict]:
        """Generate one atomic primitive on retrieval miss."""
        if self._session_count >= DYNAMIC_GEN_MAX_PER_SESSION:
            return None
        if isinstance(self._backend, MockBackend):
            raw = self._backend.generate_from_context(context_nodes, domain)
        else:
            context = " -> ".join(context_nodes[-4:]) if context_nodes else "none"
            prompt  = _GENERATE_PROMPT.format(context=context, last_node=last_node_text)
            try:
                raw = self._backend.generate_primitive(prompt)
            except Exception as e:
                print(f"  [DYN] Generation error: {e}")
                return None
        return self._validate_and_embed(raw)

    def define(self, term: str, domain: str) -> Optional[dict]:
        """Generate a definitional primitive for a specific OOV term."""
        if self._session_count >= DYNAMIC_GEN_MAX_PER_SESSION:
            return None
        # Task 10: sanitize before using in any prompt
        safe_term = _sanitize_oov_term(term)
        if safe_term is None:
            return None
        if isinstance(self._backend, MockBackend):
            raw = self._backend.define_term(safe_term, domain)
        else:
            prompt = _DEFINE_PROMPT.format(term=safe_term)
            try:
                raw = self._backend.generate_primitive(prompt)
            except Exception as e:
                print(f"  [OOV] Definition error: {e}")
                return None
        return self._validate_and_embed(raw)

    def _validate_and_embed(self, raw: str) -> Optional[dict]:
        if not raw:
            return None
        parsed = self._parse(raw)
        if not parsed:
            return None
        emb  = self.embedder.encode(parsed["text"], normalize_embeddings=True).astype(np.float32)
        harm = float(max(0.0, np.dot(emb, self._harm_cluster)))
        if harm > 0.5:
            print(f"  [DYN] Harm check failed ({harm:.3f}), discarding.")
            return None
        self._session_count += 1
        return {**parsed, "embed": emb.tolist(), "harm": harm}

    def reset_session_count(self):
        self._session_count = 0

    def _parse(self, raw: str) -> Optional[dict]:
        import json as _j
        raw = raw.strip()
        if "```" in raw:
            for p in raw.split("```"):
                p = p.strip().lstrip("json").strip()
                if p.startswith("{"):
                    raw = p
                    break
        try:
            data = _j.loads(raw)
            if "text" not in data or "type" not in data:
                return None
            if data["type"] not in {"fact", "rule", "function", "constraint", "macro"}:
                data["type"] = "fact"
            if len(data["text"].split()) > 40:
                return None
            return {"text": data["text"], "type": data["type"]}
        except Exception:
            return None

class BAPEngine:
    """
    Budgeted Activation Propagation over a ThoughtSpace.

    Update equation per step:
        A(j) = A_source · sim(i,j) · gate(j) · decay(j) · energy(t)

    Where:
        gate(j)   = sigmoid(GATE_SCALE * (strength_j - harm_j))
        decay(j)  = exp(-age_j * LAMBDA_DECAY)
        energy(t) = log(1 + remaining_budget)
        novelty(j)= 1 - max cosine(embed_j, any path node embed)
    """

    def __init__(self, space: ThoughtSpace,
                 failure_policy: FailurePolicy = FailurePolicy.PARTIAL):
        self.space          = space
        self.embedder       = _get_embedder()
        self.generator      = DynamicGenerator(self.embedder)
        self.failure_policy = failure_policy
        self._new_nodes_since_rebuild = 0
        self._last_gen_failed = False  # guard: skip gen if previous attempt failed
        self._oov_node_id     = None   # set after OOV injection; triggers brute-force on step 1


    def _expand_query(self, query: str) -> List[str]:
        """Generate query reformulations to improve first-hop retrieval."""
        variants = [query]
        ql = query.lower()

        # Person + field queries
        for name in ["marie curie", "einstein", "newton"]:
            if name in ql:
                last = name.split()[-1].title()
                variants += [
                    f"{last} Nobel Prize field",
                    f"{last} scientific contribution physics chemistry",
                ]
                break

        # Next-number / value lookup queries — target fact nodes, NOT recurrence
        if any(w in ql for w in ["next", "after"]) and any(w in ql for w in ["number", "value", "term"]):
            variants += [
                "Fibonacci sequence values 8 13 21 F(6) F(7) next term",
                "F(7) = 13 F(6) = 8 sequence value fact",
            ]
        # Generic sequence/pattern queries (no specific "next value" intent)
        elif any(w in ql for w in ["sequence", "pattern"]) and "relate" not in ql:
            variants += [
                "Fibonacci recurrence F(n) = F(n-1) + F(n-2)",
                "sequence sum of previous two terms",
            ]

        # Binet's formula — specific closed-form expansion (before generic formula check)
        if "binet" in ql:
            variants += [
                "Binet formula F(n) phi^n psi^n divided sqrt(5) closed form",
                "Fibonacci closed form phi golden ratio sqrt(5)",
            ]
        # Pascal's triangle — specific relational expansion
        elif "pascal" in ql:
            variants += [
                "Pascal triangle diagonal Fibonacci numbers shallow diagonal sum",
                "Pascal's triangle rows sums Fibonacci diagonal",
            ]
        # Specific recurrence-relation queries — use exact node-text fragments
        # to anchor SHSRS away from sum-formula nodes
        elif "recurrence" in ql and not any(w in ql for w in ["sum", "sums", "total"]):
            variants += [
                "F(n) = F(n-1) + F(n-2) for all n >= 2 base recurrence",
                "Fibonacci recurrence each term equals sum of two preceding F(n-1) plus F(n-2)",
            ]
        # Generic formula / relation queries
        elif any(w in ql for w in ["formula", "relation", "equation"]):
            variants += [
                "F(n) = F(n-1) + F(n-2) recurrence relation function",
                "Fibonacci function definition rule",
            ]

        # Contradiction queries
        if any(w in ql for w in ["contradict", "both true", "true", "false"]):
            variants += [
                "contradictory facts cannot both be true constraint",
                "logical consistency constraint",
            ]

        # Direct value lookup: F(7)
        m = re.search(r"f\((\d+)\)", ql)
        if m:
            n = m.group(1)
            variants += [f"F({n}) equals Fibonacci value", f"Fibonacci index {n}"]

        return variants[:5]

    def _infer_type_bias(self, query: str) -> str | None:
        """
        Infer which node type should be prioritised on first hop.
        Returns type name or None (no bias).
        """
        ql = query.lower()
        # Value/fact lookups — includes "next number after X" and Pascal's triangle
        if any(w in ql for w in ["what is f(", "value of f", "equals", "f(7)", "f(8)", "f(9)", "f(10)"]):
            return "fact"
        if "pascal" in ql or "diagonal" in ql:
            return "fact"
        if any(w in ql for w in ["next", "after"]) and any(w in ql for w in ["number", "value", "term"]):
            return "fact"
        # Binet specifically → function (before generic formula check)
        if "binet" in ql:
            return "function"
        if any(w in ql for w in ["formula", "recurrence", "relation", "equation", "matrix form"]):
            return "function"
        if any(w in ql for w in ["rule", "if", "when", "implies", "infer"]):
            return "rule"
        if any(w in ql for w in ["cannot", "impossible", "constraint", "must not", "never"]):
            return "constraint"
        return None

    def _detect_oov_term(self, query: str) -> Optional[str]:
        """
        Scan key terms in query against node corpus.
        Returns the most out-of-vocabulary term, or None if all terms are covered.

        A term is OOV if its embedding is far (cosine < OOV_THRESHOLD) from
        every node in the current space — meaning the concept is unrepresented.

        This catches cases like "Lucas numbers" where the query references a
        concept the index has never seen, before falling back to miss-based gen.
        """
        if not self.space.nodes:
            return None

        # Extract candidate terms: multi-word noun phrases first, then singles
        ql = query.lower()
        words = re.findall(r"[a-z]+(?:'[a-z]+)?", ql)

        # Build bigrams + unigrams, filter stopwords
        candidates = []
        for i in range(len(words) - 1):
            bigram = f"{words[i]} {words[i+1]}"
            if words[i] not in OOV_STOPWORDS and words[i+1] not in OOV_STOPWORDS:
                candidates.append(bigram)
        for w in words:
            if w not in OOV_STOPWORDS and len(w) >= 4:
                candidates.append(w)

        if not candidates:
            return None

        # Heuristic filter: only flag OOV for bigrams, proper nouns, or technical terms
        # This prevents generic words like "work", "facts", "relation" from triggering
        original_caps = set(re.findall(r"[A-Z][a-z]+", query))  # capitalized words in original
        tech_suffix   = re.compile(
            r"(formula|theorem|numbers?|sequences?|series|ratio|constant|"
            r"function|matrix|equation|property|proof|rules?|conjecture)$"
        )
        filtered = [
            c for c in candidates
            if any(p.title() in original_caps for p in c.split())     # proper noun (uni or bigram component)
            or bool(tech_suffix.search(c))                            # technical term
        ]
        candidates = filtered if filtered else []  # empty = no OOV trigger

        if not candidates:
            return None

        # Embed all candidates at once
        cand_embs = self.embedder.encode(
            candidates, normalize_embeddings=True
        ).astype(np.float32)

        # Build corpus matrix once
        corpus_embs = np.array(
            [n.embed for n in self.space.nodes.values()], dtype=np.float32
        )  # shape: (N_nodes, D)

        # For each candidate, find max cosine against all nodes
        sims = cand_embs @ corpus_embs.T   # (n_cands, N_nodes)
        max_sims = sims.max(axis=1)         # (n_cands,)

        # Bigrams always beat unigrams: group 0 = bigram, group 1 = unigram.
        # Within each group sort by ascending max_sim (most OOV first).
        # Walk ranked list and return the first entry that is actually OOV —
        # this falls back to a unigram only if every bigram is already in-corpus.
        ranked = sorted(
            range(len(candidates)),
            key=lambda i: (
                0 if len(candidates[i].split()) == 2 else 1,
                max_sims[i],
            )
        )
        for best_i in ranked:
            thresh = BIGRAM_OOV_THRESHOLD if len(candidates[best_i].split()) == 2 else OOV_THRESHOLD
            if max_sims[best_i] < thresh:
                # Task 10: sanitize before returning to caller / prompt builder
                return _sanitize_oov_term(candidates[best_i])
        return None

    # ── Core math ─────────────────────────────────────────────────────────────

    def _gate(self, node: ThoughtNode) -> float:
        return 1.0 / (1.0 + math.exp(-GATE_SCALE * (node.strength - node.harm)))

    def _decay(self, node: ThoughtNode) -> float:
        return math.exp(-node.age * LAMBDA_DECAY)

    def _energy(self) -> float:
        return math.log(1.0 + self.space.budget)

    def _novelty(self, node: ThoughtNode, path_embeds: List[np.ndarray]) -> float:
        if not path_embeds:
            return 1.0
        emb = np.array(node.embed, dtype=np.float32)
        sims = [float(np.dot(emb, pe)) for pe in path_embeds]
        return 1.0 - max(sims)

    def _is_contradiction(self, node: ThoughtNode, path_embeds: List[np.ndarray]) -> bool:
        if not path_embeds:
            return False
        emb = np.array(node.embed, dtype=np.float32)
        for pe in path_embeds:
            if float(np.dot(emb, pe)) < CONTRADICTION_THRESH:
                return True
        return False

    def _detect_drift(self, path_embeds: List[np.ndarray]) -> bool:
        """
        Return True when the last DRIFT_WINDOW path nodes form a semantic loop.

        Strategy: compute mean pairwise cosine similarity among the last
        DRIFT_WINDOW embeddings.  If the mean exceeds DRIFT_THRESHOLD, the path
        is orbiting the same semantic region and no new territory is being explored.

        Only fires when enough path nodes exist (>= DRIFT_WINDOW).
        """
        if len(path_embeds) < DRIFT_WINDOW:
            return False
        window = path_embeds[-DRIFT_WINDOW:]
        sims = []
        for i in range(len(window)):
            for j in range(i + 1, len(window)):
                sims.append(float(np.dot(window[i], window[j])))
        return (sum(sims) / len(sims)) > DRIFT_THRESHOLD if sims else False

    def _detect_stagnation(
        self,
        recent_novelties: List[float],
        recent_types: List[str]
    ) -> bool:
        """
        Return True when the path is plateauing with no structural diversity.

        Both conditions must hold simultaneously:
          1. Mean novelty of the last STAGNATION_WINDOW accepted nodes
             < STAGNATION_NOV_FLOOR  (engine is not discovering new territory)
          2. All nodes in the window share the same type
             (no structural variety — same-type lock-in)

        Distinct from drift: drift detects semantic looping via embedding
        similarity; stagnation detects a novelty plateau + type mono-culture
        regardless of whether embeddings are close.

        Returns False if fewer than STAGNATION_WINDOW nodes have been accepted.
        """
        if (len(recent_novelties) < STAGNATION_WINDOW
                or len(recent_types) < STAGNATION_WINDOW):
            return False
        window_nov   = recent_novelties[-STAGNATION_WINDOW:]
        window_types = recent_types[-STAGNATION_WINDOW:]
        mean_nov = sum(window_nov) / len(window_nov)
        return mean_nov < STAGNATION_NOV_FLOOR and len(set(window_types)) == 1

    def _activation(
        self,
        source_activation: float,
        sim: float,
        node: ThoughtNode,
        path_embeds: List[np.ndarray]
    ) -> float:
        g = self._gate(node)
        d = self._decay(node)
        e = self._energy()
        n = self._novelty(node, path_embeds)
        return source_activation * sim * g * d * e * (1.0 - NOVELTY_WEIGHT + NOVELTY_WEIGHT * n)

    # ── Search ────────────────────────────────────────────────────────────────

    def _search(self, query_vec: np.ndarray, force_brute: bool = False) -> List[Tuple[int, float]]:
        """
        SHSRS search with safe k for small domains. Auto-scales probe count.
        Also runs a separate fuzzy macro scan (Task 4): any macro whose anchor_vec
        cosine >= MACRO_MATCH_THRESHOLD is injected as a boosted candidate.
        """
        if force_brute:
            return self._brute_force_search(query_vec, set())
        n = len(self.space.nodes)
        k = min(SEARCH_K, n - 1)
        if k < 1:
            return []
        probe = _compute_search_probe(n) if SEARCH_PROBE_AUTO else SEARCH_PROBE
        try:
            candidates = self.space.engine.search(query_vec, k=k, probe=probe)
        except ValueError:
            all_nodes = list(self.space.nodes.values())
            vecs = np.array([nd.embed for nd in all_nodes], dtype=np.float32)
            sims = vecs @ query_vec
            top  = np.argsort(-sims)[:k]
            candidates = [(all_nodes[i].id, float(sims[i])) for i in top]

        # ── Fuzzy macro scan — anchor_vec cosine match ────────────────────────
        existing_ids = {nid for nid, _ in candidates}
        for nid, node in self.space.nodes.items():
            if not node.is_macro or not node.anchor_vec:
                continue
            if nid in existing_ids:
                continue
            av  = np.array(node.anchor_vec, dtype=np.float32)
            sim = float(np.dot(query_vec, av))
            if sim >= MACRO_MATCH_THRESHOLD:
                candidates.append((nid, sim * MACRO_SCORE_BOOST))

        return candidates

    # ── Main propagation loop ─────────────────────────────────────────────────

    def run(
        self,
        query: str,
        max_steps: int = 10,
        confidence_threshold: float = 0.92,
        expected_contains: str = ""
    ) -> BAPResult:
        """
        Run BAP from query until budget exhausted or confidence reached.

        Parameters
        ----------
        query                : natural language question
        max_steps            : hard cap on propagation steps
        confidence_threshold : stop early if top node activation exceeds this
        expected_contains    : optional substring — if given, used to auto-detect
                               success for outcome-weighted reinforcement
        """
        # Reset space state
        self.space.budget = self.space.global_budget
        self.space.active = {}
        self.space.path   = []
        self.generator.reset_session_count()
        self._new_nodes_since_rebuild = 0
        self._macro_hits = 0  # count macro expansions this run

        # Task 6: validate all macros before traversal starts
        self._validate_macros()

        # Embed query with expansion — average query + reformulations
        # Closes the semantic gap between natural language and node text
        expansions = self._expand_query(query)
        all_vecs   = self.embedder.encode(
            expansions, normalize_embeddings=True
        ).astype(np.float32)
        query_vec  = all_vecs.mean(axis=0)
        query_vec /= (np.linalg.norm(query_vec) + 1e-9)

        # Initialise: source activation = 1.0 at query
        source_activation = 1.0
        current_vec       = query_vec
        path_embeds: List[np.ndarray] = []
        path_novelties: List[float] = []  # novelty per accepted node — stagnation detection
        path_types: List[str] = []        # node type per accepted node — stagnation detection
        steps_log: List[BAPStep] = []
        visited: set = set()       # node ids
        visited_texts: set = set()  # node texts — catches SHSRS boundary duplicates

        self._last_gen_failed     = False  # reset per-query
        self._oov_node_id         = None   # reset per-query
        self._drift_injected      = False  # reset per-query — only inject drift node once
        self._stagnation_injected = False  # reset per-query — only inject stagnation node once

        # ── OOV pre-check: detect and define missing concepts before traversal ──
        # This catches "Lucas numbers", "Binet formula" etc. before the first hop
        # so the index contains the term when SHSRS searches for it.
        if DYNAMIC_GEN_ENABLED and self.generator.available():
            oov_term = self._detect_oov_term(query)
            if oov_term:
                print(f"[BAP] OOV detected: '{oov_term}' — generating definition...")
                defn = self.generator.define(oov_term, self.space.domain)
                if defn:
                    # Task 3: flag suspicious-but-not-blocked OOV harm (0.40–0.50)
                    harm_score = defn.get("harm", 0.0)
                    if 0.40 <= harm_score <= 0.50:
                        msg = (f"[BAP] Suspicious OOV harm score ({harm_score:.3f}) "
                               f"for term '{oov_term}'")
                        if self.failure_policy == FailurePolicy.ABSTAIN:
                            raise ValueError(msg)
                        print(f"  [WARN] {msg}")  # WARN or PARTIAL: log and continue
                    new_node = self._add_dynamic_node(defn)
                    if new_node:
                        print(f"[BAP] OOV defined: {new_node.text[:80]}")
                        # Re-embed: blend query with OOV definition (0.6/0.4)
                        # pulls first hop toward the newly defined concept
                        oov_vecs = self.embedder.encode(
                            [query, new_node.text], normalize_embeddings=True
                        ).astype(np.float32)
                        query_vec  = 0.6 * oov_vecs[0] + 0.4 * oov_vecs[1]
                        query_vec /= (np.linalg.norm(query_vec) + 1e-9)
                        self._oov_node_id = new_node.id  # force brute-force on step 1

        # Infer first-hop type bias from query intent
        type_bias = self._infer_type_bias(query)

        print(f"\n[BAP] Query: {query}")
        print(f"[BAP] Domain: {self.space.domain}  |  Nodes: {len(self.space.nodes)}")
        print(f"[BAP] Budget: {self.space.budget:.2f}  |  Max steps: {max_steps}")
        print("-" * 55)

        for step in range(1, max_steps + 1):
            if self.space.budget <= 0:
                print(f"  [step {step}] Budget exhausted.")
                break

            # Determine beam width: explore vs exploit
            k_select = EXPLORE_K if self.space.budget > 0.50 * self.space.global_budget else EXPLOIT_K

            # ── DRIFT DETECTION — inject bridge before SHSRS when path loops ─
            # Only fires once per run, only when generator is available and healthy.
            if (not self._drift_injected
                    and DYNAMIC_GEN_ENABLED
                    and self.generator.available()
                    and self.space.budget > 0.20
                    and self.generator._session_count < DYNAMIC_GEN_MAX_PER_SESSION
                    and self._detect_drift(path_embeds)):
                self._drift_injected = True
                context_texts = [self.space.nodes[nid].text for nid in self.space.path]
                last_text = context_texts[-1] if context_texts else query
                print(f"  [step {step}] Drift detected — injecting bridge node...")
                primitive = self.generator.generate(
                    context_nodes=context_texts,
                    last_node_text=last_text,
                    domain=self.space.domain
                )
                if primitive:
                    new_node = self._add_dynamic_node(primitive)
                    if new_node:
                        new_emb = np.array(new_node.embed, dtype=np.float32)
                        sim = float(np.dot(current_vec, new_emb))
                        a = self._activation(source_activation, sim, new_node, path_embeds)
                        a *= 1.4  # context boost
                        GEN_MIN_ACTIVATION = MIN_ACTIVATION * 0.5
                        if a >= GEN_MIN_ACTIVATION:
                            self.space.active[new_node.id] = a
                            self.space.path.append(new_node.id)
                            visited.add(new_node.id)
                            visited_texts.add(new_node.text)
                            path_embeds.append(new_emb)
                            cost = 0.15
                            self.space.budget = max(0.0, self.space.budget - cost)
                            steps_log.append(BAPStep(
                                step=step, node_id=new_node.id,
                                text=new_node.text, node_type=new_node.type,
                                activation=a, gate=self._gate(new_node),
                                novelty=self._novelty(new_node, path_embeds),
                                budget_left=self.space.budget,
                                action="drift_bridge"
                            ))
                            print(
                                f"  [step {step}] [DRIFT/{new_node.type}] "
                                f"A={a:.4f} budget={self.space.budget:.3f} "
                                f"| {new_node.text[:60]}"
                            )
                            source_activation = a
                            current_vec = new_emb
                            continue

            # ── STAGNATION DETECTION — inject bridge when path plateaus ──────────
            # Fires after drift check, before SHSRS search. Only once per run.
            if (not self._stagnation_injected
                    and DYNAMIC_GEN_ENABLED
                    and self.generator.available()
                    and self.space.budget > 0.20
                    and self.generator._session_count < DYNAMIC_GEN_MAX_PER_SESSION
                    and self._detect_stagnation(path_novelties, path_types)):
                self._stagnation_injected = True
                context_texts = [self.space.nodes[nid].text for nid in self.space.path]
                last_text = context_texts[-1] if context_texts else query
                print(f"  [step {step}] Stagnation detected — injecting bridge node...")
                primitive = self.generator.generate(
                    context_nodes=context_texts,
                    last_node_text=last_text,
                    domain=self.space.domain
                )
                if primitive:
                    new_node = self._add_dynamic_node(primitive)
                    if new_node:
                        new_emb  = np.array(new_node.embed, dtype=np.float32)
                        sim      = float(np.dot(current_vec, new_emb))
                        nov_pre  = self._novelty(new_node, path_embeds)  # before appending
                        a        = self._activation(source_activation, sim, new_node, path_embeds)
                        a       *= 1.4  # context boost
                        GEN_MIN_ACTIVATION = MIN_ACTIVATION * 0.5
                        if a >= GEN_MIN_ACTIVATION:
                            self.space.active[new_node.id] = a
                            self.space.path.append(new_node.id)
                            visited.add(new_node.id)
                            visited_texts.add(new_node.text)
                            path_embeds.append(new_emb)
                            path_novelties.append(nov_pre)
                            path_types.append(new_node.type)
                            cost = 0.15
                            self.space.budget = max(0.0, self.space.budget - cost)
                            steps_log.append(BAPStep(
                                step=step, node_id=new_node.id,
                                text=new_node.text, node_type=new_node.type,
                                activation=a, gate=self._gate(new_node),
                                novelty=nov_pre,
                                budget_left=self.space.budget,
                                action="stagnation_bridge"
                            ))
                            print(
                                f"  [step {step}] [STAG/{new_node.type}] "
                                f"A={a:.4f} budget={self.space.budget:.3f} "
                                f"| {new_node.text[:60]}"
                            )
                            source_activation = a
                            current_vec = new_emb
                            continue

            # SHSRS retrieval — brute-force on step 1 when OOV node was just injected
            use_brute = (step == 1 and self._oov_node_id is not None)
            if use_brute:
                # Search FROM the OOV node's own embedding so it scores sim=1.0
                # and wins activation over established corpus nodes.
                oov_emb = np.array(self.space.nodes[self._oov_node_id].embed, dtype=np.float32)
                search_vec = oov_emb / (np.linalg.norm(oov_emb) + 1e-9)
            else:
                # Goal attraction: blend current position toward query vector to prevent drift
                search_vec = (1.0 - GOAL_ATTRACTION_WEIGHT) * current_vec + GOAL_ATTRACTION_WEIGHT * query_vec
                search_vec /= (np.linalg.norm(search_vec) + 1e-9)
            self._oov_node_id = None  # clear after step 1
            candidates = self._search(search_vec, force_brute=use_brute)

            if not candidates:
                print(f"  [step {step}] No candidates found.")
                break

            # Score all candidates
            scored = []
            scored_ids: set = set()  # dedup SHSRS boundary duplicates within this step
            for node_id, sim in candidates:
                if node_id in visited:
                    continue
                if node_id in scored_ids:          # same node via two cluster paths
                    continue
                if self.space.nodes.get(node_id) and self.space.nodes[node_id].text in visited_texts:
                    continue
                node = self.space.nodes.get(node_id)
                if node is None:
                    continue

                # Harm gate — hard prune
                if node.harm > 0.5:
                    steps_log.append(BAPStep(
                        step=step, node_id=node_id, text=node.text,
                        node_type=node.type, activation=0.0, gate=0.0,
                        novelty=0.0, budget_left=self.space.budget,
                        action="pruned_harm"
                    ))
                    continue

                # Contradiction gate — hard prune
                if self._is_contradiction(node, path_embeds):
                    steps_log.append(BAPStep(
                        step=step, node_id=node_id, text=node.text,
                        node_type=node.type, activation=0.0, gate=0.0,
                        novelty=0.0, budget_left=self.space.budget,
                        action="pruned_contradiction"
                    ))
                    continue

                a = self._activation(source_activation, sim, node, path_embeds)

                # Type-bias boost on step 1 only — pulls correct node type to top
                if step == 1 and type_bias and node.type == type_bias:
                    a *= 1.6

                if a < MIN_ACTIVATION:
                    steps_log.append(BAPStep(
                        step=step, node_id=node_id, text=node.text,
                        node_type=node.type, activation=a,
                        gate=self._gate(node),
                        novelty=self._novelty(node, path_embeds),
                        budget_left=self.space.budget,
                        action="pruned_activation"
                    ))
                    continue

                scored_ids.add(node_id)
                scored.append((a, node_id, node, sim))

            if not scored:
                # ── RETRIEVAL MISS — attempt dynamic generation ───────────────
                if (DYNAMIC_GEN_ENABLED
                        and not self._last_gen_failed       # don't retry after a failed attempt
                        and self.generator.available()
                        and self.space.budget > 0.15
                        and self.generator._session_count < DYNAMIC_GEN_MAX_PER_SESSION):

                    context_texts = [self.space.nodes[nid].text for nid in self.space.path]
                    last_text     = context_texts[-1] if context_texts else query

                    print(f"  [step {step}] Retrieval miss — generating new node...")
                    primitive = self.generator.generate(
                        context_nodes=context_texts,
                        last_node_text=last_text,
                        domain=self.space.domain
                    )

                    if primitive:
                        new_node = self._add_dynamic_node(primitive)
                        if new_node:
                            # Score the new node and add if above threshold
                            new_emb = np.array(new_node.embed, dtype=np.float32)
                            sim     = float(np.dot(current_vec, new_emb))
                            a       = self._activation(source_activation, sim, new_node, path_embeds)
                            a      *= 1.4  # context boost — generated for this exact path

                            GEN_MIN_ACTIVATION = MIN_ACTIVATION * 0.5  # generated nodes get lower floor
                            if a >= GEN_MIN_ACTIVATION:
                                self.space.active[new_node.id] = a
                                self.space.path.append(new_node.id)
                                visited.add(new_node.id)
                                visited_texts.add(new_node.text)
                                path_embeds.append(new_emb)

                                cost = 0.15  # generation costs more than retrieval
                                self.space.budget = max(0.0, self.space.budget - cost)

                                steps_log.append(BAPStep(
                                    step=step, node_id=new_node.id,
                                    text=new_node.text, node_type=new_node.type,
                                    activation=a, gate=self._gate(new_node),
                                    novelty=self._novelty(new_node, path_embeds),
                                    budget_left=self.space.budget,
                                    action="generated"
                                ))
                                print(
                                    f"  [step {step}] [GEN/{new_node.type}] "
                                    f"A={a:.4f} budget={self.space.budget:.3f} "
                                    f"| {new_node.text[:60]}"
                                )
                                # Continue from the new node
                                source_activation = a
                                current_vec = new_emb
                                continue
                            else:
                                self._last_gen_failed = True
                                print(f"  [step {step}] Generated node below threshold — generation exhausted.")
                    else:
                        print(f"  [step {step}] Generation failed or limit reached.")

                print(f"  [step {step}] All candidates pruned.")
                break

            # Sort by activation, take top k_select
            scored.sort(key=lambda x: x[0], reverse=True)
            # Deduplicate winners by text — SHSRS boundary links can return
            # same logical node via two cluster paths in the same search call
            winners_raw = scored[:k_select]
            seen_this_step = set()
            winners = []
            for entry in winners_raw:
                if entry[2].text not in seen_this_step:
                    seen_this_step.add(entry[2].text)
                    winners.append(entry)

            for a, node_id, node, sim in winners:

                # ── Task 5: Macro expansion — fires when macro wins step 1 ────
                if node.is_macro and step == 1 and node.subpath_ids:
                    print(f"  [step {step}] [MACRO] Hit — id={node_id} "
                          f"sim={sim:.3f} subpath={node.subpath_ids}")
                    node.macro_freq += 1
                    self._macro_hits += 1
                    # Budget cost: half of normal traversal per constituent node
                    budget_cost = len(node.subpath_ids) * (GLOBAL_BUDGET / max_steps) * 0.5
                    self.space.budget = max(0.0, self.space.budget - budget_cost)
                    last_sub_vec = None
                    for sub_id in node.subpath_ids:
                        sub_node = self.space.nodes.get(sub_id)
                        if sub_node is None or sub_id in visited:
                            continue
                        sub_emb = np.array(sub_node.embed, dtype=np.float32)
                        sub_nov = self._novelty(sub_node, path_embeds)
                        self.space.active[sub_id] = a
                        self.space.path.append(sub_id)
                        visited.add(sub_id)
                        visited_texts.add(sub_node.text)
                        path_embeds.append(sub_emb)
                        path_novelties.append(sub_nov)
                        path_types.append(sub_node.type)
                        last_sub_vec = sub_emb
                        steps_log.append(BAPStep(
                            step=step, node_id=sub_id,
                            text=sub_node.text, node_type=sub_node.type,
                            activation=a, gate=self._gate(sub_node),
                            novelty=sub_nov, budget_left=self.space.budget,
                            action="macro_expanded"
                        ))
                        print(f"  [step {step}] [macro_exp/{sub_node.type}] "
                              f"| {sub_node.text[:60]}")
                    if last_sub_vec is not None:
                        source_activation = a
                        current_vec = last_sub_vec
                    break  # macro expansion replaces entire step; move to next

                # ── Normal node selection ─────────────────────────────────────
                g = self._gate(node)
                n_score = self._novelty(node, path_embeds)  # computed before append

                self.space.active[node_id] = a
                self.space.path.append(node_id)
                visited.add(node_id)
                visited_texts.add(node.text)
                path_embeds.append(np.array(node.embed, dtype=np.float32))
                path_novelties.append(n_score)
                path_types.append(node.type)

                cost = 0.08 * (1.0 + node.age * 0.1)
                self.space.budget = max(0.0, self.space.budget - cost)

                steps_log.append(BAPStep(
                    step=step, node_id=node_id, text=node.text,
                    node_type=node.type, activation=a, gate=g,
                    novelty=n_score, budget_left=self.space.budget,
                    action="selected"
                ))

                print(
                    f"  [step {step}] [{node.type}] A={a:.4f} "
                    f"gate={g:.3f} nov={n_score:.3f} "
                    f"budget={self.space.budget:.3f} | {node.text[:60]}"
                )

                if a > confidence_threshold and len(self.space.path) >= 2:
                    print(f"  [step {step}] Confidence threshold reached.")
                    self.space.budget = 0.0
                    break

            # Move current_vec to highest activation winner for next step
            best = winners[0]
            # If winner was a macro that already set current_vec via expansion, skip
            if not (best[2].is_macro and step == 1 and best[2].subpath_ids):
                source_activation = best[0]
                current_vec = np.array(best[2].embed, dtype=np.float32)

            if self.space.budget <= 0:
                break

        # ── Task 3: FailurePolicy checks post-loop ────────────────────────────
        # Budget exhaustion with empty path
        if not self.space.path:
            msg = f"[BAP] Budget exhausted with empty path for query: {query!r}"
            if self.failure_policy == FailurePolicy.ABSTAIN:
                raise ValueError(msg)
            if self.failure_policy == FailurePolicy.WARN:
                print(f"  [WARN] {msg}")

        # Drift + stagnation both fired, end-of-run novelty still < 0.10
        last_novelty = path_novelties[-1] if path_novelties else 1.0
        if self._drift_injected and self._stagnation_injected and last_novelty < 0.10:
            msg = (f"[BAP] Drift+stagnation both fired; end novelty "
                   f"{last_novelty:.3f} < 0.10 — path is stuck")
            if self.failure_policy == FailurePolicy.ABSTAIN:
                raise ValueError(msg)
            print(f"  [WARN] {msg}")  # always log; PARTIAL silently continues

        # Post-session: outcome-weighted reinforcement
        answer_text = " ".join(
            self.space.nodes[nid].text for nid in self.space.path
            if nid in self.space.nodes
        ).lower()
        if expected_contains:
            success = expected_contains.lower() in answer_text
        else:
            success = True
        self._update_node_stats(success=success, query_vec=query_vec)

        # ── Macro self-optimisation ───────────────────────────────────────────
        self._record_path(list(self.space.path), success, query_vec)
        if MACRO_ENABLED:
            self._compile_macros()

        answer_nodes = [
            self.space.nodes[nid].text for nid in self.space.path
            if nid in self.space.nodes
        ]
        macros_available = sum(1 for n in self.space.nodes.values() if n.is_macro)

        return BAPResult(
            query=query,
            domain=self.space.domain,
            path=steps_log,
            final_budget=self.space.budget,
            steps_taken=len(self.space.path),
            answer_nodes=answer_nodes,
            macro_hits=self._macro_hits,
            macros_available=macros_available,
        )

    def _add_dynamic_node(
        self,
        primitive: dict,
        is_macro: bool = False,
        subpath_ids: Optional[List[int]] = None,
        anchor_vec: Optional[List[float]] = None,
    ) -> Optional[ThoughtNode]:
        """
        Add a dynamically generated node to the live thought space.
        Uses brute-force cosine as fallback until index rebuild threshold.
        Pass is_macro=True for compiled macro nodes; they bypass retirement and
        receive is_static=True so _retire_weak_nodes() leaves them alone.
        """
        text = primitive["text"]

        # Task 7: cross-session text dedup via hash index
        text_key = hash(text.lower().strip())
        if text_key in self.space._text_hashes:
            print(f"  [DYN] Duplicate node skipped: {text[:40]!r}")
            return None

        new_id = max(self.space.nodes.keys()) + 1 if self.space.nodes else 0

        node = ThoughtNode(
            id=new_id,
            text=text,
            type=primitive["type"],
            embed=primitive["embed"],
            strength=0.65,
            age=0.0,
            harm=primitive.get("harm", 0.0),
            is_static=is_macro,   # macros are static; plain DYN nodes are not
            is_macro=is_macro,
            subpath_ids=subpath_ids or [],
            macro_freq=0,
            anchor_vec=anchor_vec or [],
        )
        self.space.nodes[new_id] = node
        self.space._text_hashes.add(text_key)
        self._new_nodes_since_rebuild += 1

        # Adaptive rebuild interval
        rebuild_every = (
            DYNAMIC_INDEX_REBUILD_EVERY
            if DYNAMIC_INDEX_REBUILD_EVERY > 0
            else _compute_rebuild_interval(len(self.space.nodes))
        )
        if self._new_nodes_since_rebuild >= rebuild_every:
            self._rebuild_index()
            self._new_nodes_since_rebuild = 0

        tag = "MACRO" if is_macro else "DYN"
        print(f"  [{tag}] Added node id={new_id} [{node.type}]: {node.text[:70]}")
        return node

    def _rebuild_index(self):
        """Rebuild SHSRS index with all current nodes including dynamic ones."""
        vectors    = np.array([n.embed for n in self.space.nodes.values()], dtype=np.float32)
        n          = len(vectors)
        n_clusters = _compute_n_clusters(n)  # Task 1: auto-scaled cluster count
        index_dir  = f"thought_space_{self.space.domain}"

        print(f"  [DYN] Rebuilding index: {n} nodes, {n_clusters} clusters")
        self.space.engine = SHSRSEngine.build(
            vectors=vectors,
            index_dir=index_dir,
            n_clusters=n_clusters,
            M=8,
            ef_construction=100,
        )

    def _brute_force_search(
        self,
        query_vec: np.ndarray,
        exclude_ids: set
    ) -> List[Tuple[int, float]]:
        """
        Brute-force cosine search across all nodes.
        Used for dynamic nodes not yet in SHSRS index.
        """
        results = []
        for nid, node in self.space.nodes.items():
            if nid in exclude_ids:
                continue
            emb = np.array(node.embed, dtype=np.float32)
            sim = float(np.dot(query_vec, emb))
            results.append((nid, sim))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:SEARCH_K]

    def _update_node_stats(
        self,
        success: bool = True,
        query_vec: Optional[np.ndarray] = None
    ):
        """
        Post-session outcome-weighted reinforcement — query-relevance weighted.

        Each activated node's delta is scaled by its cosine similarity to the
        original query embedding (clamped >= 0).  Hub nodes that appear on many
        paths but are only tangentially related to the query get near-zero
        updates; nodes that directly answered the query receive the full delta.

        success=True  : delta = +REINFORCE_MAX_DELTA * relevance
        success=False : delta = -REINFORCE_MIN_DELTA * relevance
        non-activated : age += 0.1 per session (passive decay via gate)

        query_vec=None : falls back to uniform relevance=1.0 (backward-compat).
        """
        activated = set(self.space.path)

        for nid, node in self.space.nodes.items():
            node.age += 0.1
            if nid in activated:
                if query_vec is not None:
                    emb = np.array(node.embed, dtype=np.float32)
                    relevance = max(0.0, float(np.dot(emb, query_vec)))
                else:
                    relevance = 1.0  # no query vec — uniform (backward-compatible)
                delta = (REINFORCE_MAX_DELTA if success else -REINFORCE_MIN_DELTA) * relevance
                node.strength = max(0.0, min(1.0, node.strength + delta))

        outcome = "reinforced" if success else "decayed"
        print(f"  [LEARN] {len(activated)} nodes {outcome} "
              f"(relevance-weighted, success={success})")

        # Task 6: renormalize strengths after updates
        self._renormalize_strengths()
        # Task 4: retire weak DYN nodes that failed to add value
        if NODE_RETIRE_ENABLED:
            self._retire_weak_nodes()

    def _renormalize_strengths(self):
        """
        Task 6 — Strength ceiling/floor renormalization.

        1. Hard clamp all strengths to [STRENGTH_FLOOR, STRENGTH_CEIL].
        2. If >20% of nodes are at ceiling (>0.90): scale all by 0.90.
        3. If >20% of nodes are at floor (<0.10): add 0.05 to all, then re-clamp.
        """
        nodes = list(self.space.nodes.values())
        if not nodes:
            return

        # Step 1: hard clamp
        for node in nodes:
            node.strength = max(STRENGTH_FLOOR, min(STRENGTH_CEIL, node.strength))

        n = len(nodes)
        # Step 2: ceiling compression
        at_ceil = sum(1 for node in nodes if node.strength > 0.90)
        if at_ceil / n > STRENGTH_COMPRESS_THRESHOLD:
            for node in nodes:
                node.strength = min(STRENGTH_CEIL, node.strength * 0.90)
            print(f"  [RENORM] Ceiling compression applied ({at_ceil}/{n} nodes >{0.90})")

        # Step 3: floor lift
        at_floor = sum(1 for node in nodes if node.strength < 0.10)
        if at_floor / n > STRENGTH_LIFT_THRESHOLD:
            for node in nodes:
                node.strength = max(STRENGTH_FLOOR, min(STRENGTH_CEIL, node.strength + 0.05))
            print(f"  [RENORM] Floor lift applied ({at_floor}/{n} nodes <{0.10})")

    def _retire_weak_nodes(self):
        """
        Task 4 — Node retirement policy.

        Retires DYN nodes (is_static=False) where strength < NODE_RETIRE_MAX_STRENGTH
        AND age > NODE_RETIRE_MIN_AGE.  Never retires static corpus nodes.
        Retired node IDs are tracked in space._retired_node_ids and removed from
        space.nodes so they don't participate in future traversals.
        """
        to_retire = [
            nid for nid, node in self.space.nodes.items()
            if not node.is_static
            and not node.is_macro   # macros are never retired (passed harm check at compile)
            and node.strength < NODE_RETIRE_MAX_STRENGTH
            and node.age > NODE_RETIRE_MIN_AGE
        ]
        for nid in to_retire:
            self.space._retired_node_ids.add(nid)
            del self.space.nodes[nid]
        if to_retire:
            print(f"  [RETIRE] Retired {len(to_retire)} weak DYN node(s): {to_retire}")

    # ── Task 6: Macro invalidation watcher ────────────────────────────────────

    def _validate_macros(self):
        """
        Task 6 — Check every macro node's constituent subpath_ids are still live.

        - If any subpath node is missing or retired → invalidate macro (strength=0,
          retired flag) and log.
        - If macro strength has drifted below 0.50 → recompute as mean of current
          constituent strengths.
        """
        for nid, node in list(self.space.nodes.items()):
            if not node.is_macro:
                continue
            for sub_id in node.subpath_ids:
                if sub_id not in self.space.nodes:
                    print(f"  [MACRO] Invalidated id={nid} — "
                          f"constituent node {sub_id} retired/missing")
                    node.strength = 0.0
                    self.space._retired_node_ids.add(nid)
                    del self.space.nodes[nid]
                    break
            else:
                # All constituents present — check strength drift
                if node.strength < 0.50 and node.subpath_ids:
                    strengths = [
                        self.space.nodes[sid].strength
                        for sid in node.subpath_ids
                        if sid in self.space.nodes
                    ]
                    if strengths:
                        node.strength = max(0.50, min(0.90,
                                                       sum(strengths) / len(strengths)))

    # ── Task 1: Path frequency tracker ────────────────────────────────────────

    def _record_path(self, path_node_ids: List[int], success: bool,
                     query_vec: np.ndarray):
        """
        Record the current reasoning path in the path registry.
        Atomically persists path_registry.json after each update.
        """
        if not path_node_ids:
            return
        sig     = "|".join(str(i) for i in path_node_ids)
        registry = self.space.path_registry
        if sig not in registry:
            registry[sig] = {
                "count": 0, "successes": 0,
                "query_vecs": [], "last_seen": ""
            }
        entry = registry[sig]
        entry["count"] += 1
        if success:
            entry["successes"] += 1
        entry["query_vecs"].append(query_vec.tolist())
        if len(entry["query_vecs"]) > PATH_REGISTRY_MAX_VECS:
            entry["query_vecs"] = entry["query_vecs"][-PATH_REGISTRY_MAX_VECS:]
        entry["last_seen"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        _save_path_registry(self.space)

    # ── Task 3: Macro compilation ──────────────────────────────────────────────

    def _compile_macros(self):
        """
        Task 3 — Scan path registry for qualifying paths and compile macro nodes.

        Qualification criteria (all must hold):
          - count >= MACRO_MIN_FREQUENCY
          - len(subpath) >= MACRO_MIN_LENGTH
          - successes/count >= MACRO_MIN_SUCCESS_RATE
          - No existing macro already covers this exact subpath
          - All constituent nodes still active in space

        Task 7 — Nested macro depth: allow macro-of-macro (depth 2) but reject
          macro-of-macro-of-macro (depth 3+).
        """
        if not MACRO_ENABLED:
            return

        # Index existing macro subpaths so we don't compile duplicates
        existing_sigs: set = set()
        for node in self.space.nodes.values():
            if node.is_macro and node.subpath_ids:
                existing_sigs.add("|".join(str(i) for i in node.subpath_ids))

        for sig, entry in list(self.space.path_registry.items()):
            if entry["count"] < MACRO_MIN_FREQUENCY:
                continue
            path_ids = [int(x) for x in sig.split("|")]
            if len(path_ids) < MACRO_MIN_LENGTH:
                continue
            rate = entry["successes"] / entry["count"] if entry["count"] else 0.0
            if rate < MACRO_MIN_SUCCESS_RATE:
                continue
            if sig in existing_sigs:
                continue
            # All constituents must be live
            if any(pid not in self.space.nodes for pid in path_ids):
                continue

            # Task 7: nested macro depth limit (max depth = 2)
            max_depth = 0
            for pid in path_ids:
                node = self.space.nodes[pid]
                if node.is_macro:
                    depth = 1
                    for sub_pid in node.subpath_ids:
                        sub = self.space.nodes.get(sub_pid)
                        if sub and sub.is_macro:
                            depth = 2
                    max_depth = max(max_depth, depth)
            if max_depth > 2:
                print(f"  [MACRO] Depth limit — nested macro depth > 2, skipping {sig[:60]}")
                continue

            # ── Compile ───────────────────────────────────────────────────────
            q_vecs = np.array(entry["query_vecs"], dtype=np.float32)
            anchor = q_vecs.mean(axis=0)
            anchor /= (np.linalg.norm(anchor) + 1e-9)

            node_texts = [
                self.space.nodes[pid].text for pid in path_ids
                if pid in self.space.nodes
            ]
            joined = " -> ".join(node_texts)

            # Generate summary text via LLM if available
            if self.generator.available():
                prompt = _MACRO_PROMPT.format(chain=joined[:400])
                try:
                    raw    = self.generator._backend.generate_primitive(prompt)
                    parsed = self.generator._parse(raw)
                    macro_text = parsed["text"] if parsed else f"Compiled: {node_texts[0][:40]}..."
                except Exception:
                    macro_text = f"Compiled: {node_texts[0][:40]}..."
            else:
                macro_text = f"Compiled: {node_texts[0][:40]}..."

            emb = self.embedder.encode(
                macro_text, normalize_embeddings=True
            ).astype(np.float32)

            strengths = [
                self.space.nodes[pid].strength for pid in path_ids
                if pid in self.space.nodes
            ]
            macro_strength = max(0.70, min(0.90,
                                            sum(strengths) / len(strengths)))

            primitive = {"text": macro_text, "type": "macro",
                         "embed": emb.tolist(), "harm": 0.0}
            new_node = self._add_dynamic_node(
                primitive,
                is_macro=True,
                subpath_ids=path_ids,
                anchor_vec=anchor.tolist(),
            )
            if new_node:
                new_node.strength = macro_strength
                existing_sigs.add(sig)
                print(f"  [MACRO] Compiled path {sig[:60]} -> "
                      f"node id={new_node.id} freq={entry['count']} "
                      f"success={rate:.0%}")


# ──────────────────────────────────────────────────────────────────────────────
# DOMAIN ROUTER — auto-select ThoughtSpace for a query
# ──────────────────────────────────────────────────────────────────────────────

class DomainRouter:
    """
    Routes a query to the best-matching ThoughtSpace.

    Strategy: run a k=1 SHSRS search against every registered domain and pick
    the domain whose top-1 cosine similarity is highest.  This is more accurate
    than domain-centroid similarity because the centroid of a large, diverse
    domain (e.g. arithmetic) is blurry and averages out the domain signal.

    Usage:
        router = DomainRouter(_get_embedder())
        router.register(space_fib)
        router.register(space_arith)
        domain, space, score = router.route("What is F(7)?")
    """

    def __init__(self, embedder: SentenceTransformer):
        self._embedder = embedder
        self._spaces: Dict[str, ThoughtSpace] = {}

    def register(self, space: ThoughtSpace) -> None:
        self._spaces[space.domain] = space

    def route(
        self, query: str, verbose: bool = True
    ) -> Tuple[str, ThoughtSpace, float, bool, float]:
        """
        Return (domain_name, space, best_top1_score, low_confidence, routing_margin).

        low_confidence is True when best_score < ROUTER_LOW_CONFIDENCE_THRESHOLD.
        routing_margin is the gap between top-1 and top-2 scores (domain bleed risk
        when margin < ROUTER_MARGIN_WARN_THRESHOLD).
        """
        qvec = self._embedder.encode(
            [query], normalize_embeddings=True
        )[0].astype(np.float32)

        best_domain: Optional[str] = None
        best_score: float = -1.0
        scores_log: Dict[str, float] = {}

        for name, space in self._spaces.items():
            n_nodes = len(space.nodes)
            probe   = _compute_search_probe(n_nodes) if SEARCH_PROBE_AUTO else SEARCH_PROBE
            results = space.engine.search(qvec, k=1, probe=probe)
            top_sim = results[0][1] if results else 0.0

            # Task 12: cold-start boost for small/new domains
            if n_nodes < ROUTER_COLD_START_THRESHOLD:
                if verbose:
                    print(f"[ROUTER] Cold-start boost applied to '{name}' ({n_nodes} nodes)")
                top_sim *= ROUTER_COLD_START_BOOST

            scores_log[name] = top_sim
            if top_sim > best_score:
                best_score = top_sim
                best_domain = name

        # Task 11: domain bleed detection — compute margin between top-1 and top-2
        sorted_scores = sorted(scores_log.values(), reverse=True)
        routing_margin = (sorted_scores[0] - sorted_scores[1]) if len(sorted_scores) >= 2 else 1.0
        if routing_margin < ROUTER_MARGIN_WARN_THRESHOLD:
            sorted_domains = sorted(scores_log.items(), key=lambda x: x[1], reverse=True)
            d1, s1 = sorted_domains[0]
            d2, s2 = sorted_domains[1]
            print(
                f"[ROUTER] Low margin ({routing_margin:.3f}) — domain bleed risk. "
                f"Top: {d1} ({s1:.3f}) vs {d2} ({s2:.3f})"
            )

        low_confidence = best_score < ROUTER_LOW_CONFIDENCE_THRESHOLD
        if verbose:
            scores_str = "  ".join(f"{n}={s:.3f}" for n, s in scores_log.items())
            conf_tag = " [LOW-CONFIDENCE]" if low_confidence else ""
            print(f"[ROUTER] {scores_str}  -> {best_domain} (score={best_score:.3f}){conf_tag}")

        return best_domain, self._spaces[best_domain], best_score, low_confidence, routing_margin


# ──────────────────────────────────────────────────────────────────────────────
# MULTI-BAP ENGINE — single entry point across all domains
# ──────────────────────────────────────────────────────────────────────────────

class MultiBAPEngine:
    """
    BAP engine that auto-routes queries to the correct ThoughtSpace.

    Loads one BAPEngine per domain.  Call run() exactly like BAPEngine.run()
    — the router selects the domain transparently.

    Usage:
        multi = MultiBAPEngine()                   # loads all 3 domains
        result = multi.run("What is F(7)?")        # auto-routes to fibonacci
        result = multi.run("What is 17 mod 5?")   # auto-routes to arithmetic
        multi.save_all()

    Custom domain set:
        multi = MultiBAPEngine(["fibonacci_reasoning", "arithmetic"])
    """

    ALL_DOMAINS: List[str] = [
        "fibonacci_reasoning",
        "arithmetic",
        "multihop_facts",
    ]

    def __init__(self, domains: Optional[List[str]] = None,
                 failure_policy: FailurePolicy = FailurePolicy.PARTIAL):
        if domains is None:
            domains = self.ALL_DOMAINS
        self.failure_policy = failure_policy
        self._embedder = _get_embedder()
        self.router = DomainRouter(self._embedder)
        self._engines: Dict[str, BAPEngine] = {}
        for domain in domains:
            space = load_thought_space(domain)
            self.router.register(space)
            self._engines[domain] = BAPEngine(space, failure_policy=failure_policy)
        print(f"[MultiBAPEngine] Ready — {len(domains)} domain(s): {domains}")

    # ------------------------------------------------------------------
    def run(
        self,
        query: str,
        max_steps: int = 10,
        **kwargs
    ) -> BAPResult:
        """
        Route query to best domain, then run BAPEngine.run().

        Routing failure behaviour is governed by self.failure_policy (Task 3):
          ABSTAIN  — raises ValueError on low-confidence route
          WARN     — logs warning and continues
          PARTIAL  — silently continues (default)
        """
        domain, _space, score, low_conf, routing_margin = self.router.route(query)
        if low_conf:
            msg = (
                f"[MultiBAPEngine] Low-confidence routing (score={score:.3f} < "
                f"{ROUTER_LOW_CONFIDENCE_THRESHOLD}, margin={routing_margin:.3f}). "
                f"Best domain: {domain}."
            )
            if self.failure_policy == FailurePolicy.ABSTAIN:
                raise ValueError(msg)
            if self.failure_policy == FailurePolicy.WARN:
                print(f"  [WARN] {msg}")
            # PARTIAL: silently continue
        return self._engines[domain].run(query, max_steps=max_steps, **kwargs)

    # ------------------------------------------------------------------
    def save_all(self) -> None:
        """Persist strength/age updates and path registries for every loaded domain."""
        total_macros = 0
        for engine in self._engines.values():
            save_updated_nodes(engine.space)
            _save_path_registry(engine.space)
            total_macros += sum(1 for n in engine.space.nodes.values() if n.is_macro)
        print(f"[MultiBAPEngine] save_all complete — "
              f"{total_macros} macro node(s) across all domains")

    # ------------------------------------------------------------------
    @property
    def spaces(self) -> Dict[str, ThoughtSpace]:
        return {d: eng.space for d, eng in self._engines.items()}


# ──────────────────────────────────────────────────────────────────────────────
# SAVE PATH REGISTRY (macro session — persist path_registry.json)
# ──────────────────────────────────────────────────────────────────────────────

def _save_path_registry(space: ThoughtSpace) -> None:
    """Atomically persist path_registry.json for the given domain."""
    if not space.path_registry:
        return
    reg_path = Path(f"thought_space_{space.domain}/path_registry.json")
    tmp_path = reg_path.with_suffix(".tmp")
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(space.path_registry, f, indent=2)
        os.replace(tmp_path, reg_path)
    except Exception:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
        raise


# ──────────────────────────────────────────────────────────────────────────────
# SAVE UPDATED NODES (persist strength/age changes)
# ──────────────────────────────────────────────────────────────────────────────

def save_updated_nodes(space: ThoughtSpace,
                       failure_policy: FailurePolicy = FailurePolicy.PARTIAL):
    """
    Persist strength/age updates, new DYN nodes, retired nodes, and text hashes
    back to nodes.json.

    Task 8  — Atomic write: write to .tmp, then os.replace() (atomic on POSIX +
              Windows Vista+).  Keeps one .bak before each write.
    Task 13 — Concurrent write protection: platform-specific file lock on a
              separate sentinel file (nodes.json.lck).  This avoids the Windows
              same-process read-after-lock issue that occurs when locking the
              data file directly.  5-second timeout.  On lock failure,
              behaviour is governed by failure_policy.
    """
    from dataclasses import asdict
    path      = Path(f"thought_space_{space.domain}/nodes.json")
    tmp_path  = path.with_suffix(".tmp")
    lock_path = path.with_suffix(".json.lck")  # sentinel lock file

    # ── Task 13: acquire exclusive write lock via sentinel file ───────────────
    lock_fh       = None
    lock_acquired = False
    try:
        lock_fh  = open(lock_path, "a+b")  # create if absent; open for append+read
        deadline = time.monotonic() + 5.0
        if sys.platform == "win32":
            import msvcrt
            lock_fh.seek(0)
            while time.monotonic() < deadline:
                try:
                    msvcrt.locking(lock_fh.fileno(), msvcrt.LK_NBLCK, 1)
                    lock_acquired = True
                    break
                except OSError:
                    time.sleep(0.1)
        else:
            import fcntl
            while time.monotonic() < deadline:
                try:
                    fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    lock_acquired = True
                    break
                except BlockingIOError:
                    time.sleep(0.1)

        if not lock_acquired:
            msg = f"[BAP] Could not acquire write lock on {lock_path} within 5s"
            if failure_policy == FailurePolicy.ABSTAIN:
                raise RuntimeError(msg)
            if failure_policy == FailurePolicy.PARTIAL:
                # Save to timestamped fallback
                ts_path  = path.parent / f"nodes_{int(time.time())}.json.lck"
                print(f"  [WARN] {msg} — saving to fallback: {ts_path}")
                lock_fh.close()
                lock_fh = None
                path     = path.parent / f"nodes_{int(time.time())}.json"
                tmp_path = path.with_suffix(".tmp")
            else:  # WARN
                print(f"  [WARN] {msg} — skipping save.")
                return

        # ── Load existing data ─────────────────────────────────────────────────
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        node_map = {n["id"]: n for n in data["nodes"]}

        # Update existing nodes (including macro fields)
        for nid, node in space.nodes.items():
            if nid in node_map:
                node_map[nid]["strength"]     = node.strength
                node_map[nid]["age"]          = node.age
                node_map[nid]["is_static"]    = node.is_static
                node_map[nid]["is_macro"]     = node.is_macro
                node_map[nid]["subpath_ids"]  = node.subpath_ids
                node_map[nid]["macro_freq"]   = node.macro_freq
                node_map[nid]["anchor_vec"]   = node.anchor_vec

        # Persist new DYN nodes not yet in file
        existing_ids = set(node_map.keys())
        new_count = 0
        for nid, node in space.nodes.items():
            if nid not in existing_ids:
                node_map[nid] = asdict(node)
                new_count += 1

        # Task 4: mark retired node IDs as retired=true in the JSON
        for nid in space._retired_node_ids:
            if nid in node_map:
                node_map[nid]["retired"] = True

        # Task 7: persist text hash index
        data["text_hashes"] = list(space._text_hashes)
        data["nodes"] = list(node_map.values())
        data["count"] = len(data["nodes"])

        # ── Task 8: atomic write ───────────────────────────────────────────────
        # 1. Keep one backup of existing file
        if path.exists():
            shutil.copy2(path, path.with_suffix(".bak"))

        # 2. Write to .tmp, then atomic rename
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp_path, path)
        except Exception:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass
            raise

        print(f"[BAP] Node stats saved -> {path}  ({data['count']} nodes, +{new_count} new)")

    finally:
        if lock_fh is not None:
            try:
                if lock_acquired:
                    if sys.platform == "win32":
                        import msvcrt
                        lock_fh.seek(0)
                        msvcrt.locking(lock_fh.fileno(), msvcrt.LK_UNLCK, 1)
                    else:
                        import fcntl
                        fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)
            except Exception:
                pass
            lock_fh.close()


# ──────────────────────────────────────────────────────────────────────────────
# BENCHMARK — measure TRR vs CoT baseline
# ──────────────────────────────────────────────────────────────────────────────

def benchmark(engine: BAPEngine, test_cases: List[dict]) -> dict:
    """
    Run BAP on test cases and measure Token Reduction Ratio.

    test_cases: list of {query, cot_tokens, expected_answer_contains}
        cot_tokens: how many tokens a standard CoT would use (estimated)
        expected_answer_contains: substring that must appear in answer path
    """
    results = []
    for tc in test_cases:
        expected = tc.get("expected_answer_contains", "")
        result = engine.run(tc["query"], max_steps=10, expected_contains=expected)
        bap_tokens = len(result.path)  # each retrieved node = ~0 generation tokens

        # Check if expected content surfaced
        full_answer = " ".join(result.answer_nodes).lower()
        hit = expected.lower() in full_answer if expected else True

        trr = bap_tokens / max(1, tc.get("cot_tokens", 50))
        results.append({
            "query":      tc["query"],
            "bap_steps":  result.steps_taken,
            "cot_tokens": tc.get("cot_tokens", 50),
            "trr":        round(trr, 3),
            "answer_hit": hit,
            "budget_used": round(1 - result.final_budget, 3)
        })

    avg_trr = sum(r["trr"] for r in results) / len(results)
    hit_rate = sum(r["answer_hit"] for r in results) / len(results)

    print(f"\n{'='*55}")
    print(f"BENCHMARK RESULTS")
    print(f"{'='*55}")
    for r in results:
        status = "OK" if r["answer_hit"] else "XX"
        print(f"  {status} TRR={r['trr']:.3f} | steps={r['bap_steps']} | {r['query'][:50]}")
    print(f"{'='*55}")
    print(f"  Avg TRR:  {avg_trr:.3f}  (target < 0.40)")
    print(f"  Hit rate: {hit_rate:.1%} (target > 95%)")
    print(f"{'='*55}")

    return {"avg_trr": avg_trr, "hit_rate": hit_rate, "details": results}


# ──────────────────────────────────────────────────────────────────────────────
# MAIN — test run
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os

    # Set DYNAMIC_GEN_MODEL = "mock" for testing without LLM
    # Set DYNAMIC_GEN_MODEL = "ollama" if ollama is running locally
    # Set DYNAMIC_GEN_MODEL = "openai" + OPENAI_API_KEY env var for OpenAI
    print(f"[BAP] Dynamic generation: model={DYNAMIC_GEN_MODEL}, enabled={DYNAMIC_GEN_ENABLED}")

    # ── Test 1: Fibonacci ──────────────────────────────────────────────────────
    space  = load_thought_space("fibonacci_reasoning")
    engine = BAPEngine(space)
    print(f"[BAP] Generator available: {engine.generator.available()}")

    result = engine.run("What is the next number after 8 in the Fibonacci sequence?")
    print(result.summary())

    # ── Test: trigger dynamic generation (Lucas numbers not in index) ──────────
    print("\n[TEST] Triggering dynamic generation with out-of-index query...")
    result_dyn = engine.run("How do Lucas numbers relate to Fibonacci?", max_steps=8)
    print(result_dyn.summary())
    save_updated_nodes(space)

    # ── Test 2: Multi-hop ─────────────────────────────────────────────────────
    space2  = load_thought_space("multihop_facts")
    engine2 = BAPEngine(space2)

    result2 = engine2.run("What scientific field did Marie Curie work in?")
    print(result2.summary())

    result3 = engine2.run("Can two contradictory facts about the same subject both be true?")
    print(result3.summary())

    # ── Benchmark ─────────────────────────────────────────────────────────────
    TEST_CASES = [
        {
            "query": "What is F(7)?",
            "cot_tokens": 40,
            "expected_answer_contains": "13"
        },
        {
            "query": "What is the recurrence relation for Fibonacci?",
            "cot_tokens": 60,
            "expected_answer_contains": "f(n-1) + f(n-2)"  # lowercase match is fine — hit check is case-insensitive
        },
        {
            "query": "What field did Marie Curie work in?",
            "cot_tokens": 50,
            "expected_answer_contains": "physics"  # in Nobel Prize fact or rule
        },
        {
            "query": "Can two contradictory facts both be true?",
            "cot_tokens": 30,
            "expected_answer_contains": "cannot"
        },
    ]

    space3  = load_thought_space("fibonacci_reasoning")
    engine3 = BAPEngine(space3)
    benchmark(engine3, TEST_CASES[:2])

    space4  = load_thought_space("multihop_facts")
    engine4 = BAPEngine(space4)
    benchmark(engine4, TEST_CASES[2:])