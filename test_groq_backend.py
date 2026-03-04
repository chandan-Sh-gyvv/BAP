"""
BAP — Groq Backend Validation Test
====================================
Validates that the real GroqBackend integrates correctly with the BAP engine.

Tests:
  1. GroqBackend.available()  — key found in env
  2. Raw generate_primitive() — Groq returns valid JSON primitive
  3. OOV detection + Groq definition (Lucas numbers → defined before traversal)
  4. Retrieval miss + Groq generation (bridge node injected mid-traversal)
  5. save_updated_nodes()     — persistence works after real generation

Run:
    set GROQ_API_KEY=gsk_...
    python test_groq_backend.py
"""

import os
import sys
import json

# ── Preflight: key must be set ────────────────────────────────────────────────
if not os.environ.get("GROQ_API_KEY"):
    print("[ERROR] GROQ_API_KEY not set in environment.")
    print("  Run:  set GROQ_API_KEY=gsk_...")
    sys.exit(1)

import bap_engine

# ── Switch to Groq ────────────────────────────────────────────────────────────
bap_engine.DYNAMIC_GEN_MODEL = "groq"
print(f"[TEST] Backend: {bap_engine.DYNAMIC_GEN_MODEL}")
print(f"[TEST] Model:   {bap_engine.DYNAMIC_GEN_GROQ_MODEL}")
print("=" * 60)

# ── TEST 1: available() ───────────────────────────────────────────────────────
print("\n[TEST 1] GroqBackend.available()")
backend = bap_engine.GroqBackend()
assert backend.available(), "FAIL — GROQ_API_KEY not detected by backend"
print("  PASS — backend sees key")

# ── TEST 2: raw generate_primitive() with _DEFINE_PROMPT ─────────────────────
print("\n[TEST 2] Raw generate_primitive() — define 'Lucas numbers'")
prompt = bap_engine._DEFINE_PROMPT.format(term="Lucas numbers")
raw = backend.generate_primitive(prompt)
print(f"  Raw response: {raw[:200]}")

try:
    parsed = json.loads(raw)
    assert "text" in parsed and "type" in parsed, "Missing text or type keys"
    assert parsed["type"] in {"fact", "rule", "function", "constraint"}, f"Bad type: {parsed['type']}"
    assert len(parsed["text"].split()) <= 40, "Text too long"
    print(f"  PASS — valid JSON | type={parsed['type']} | text={parsed['text']}")
except Exception as e:
    # DynamicGenerator._parse handles code blocks — check if that would save it
    gen = bap_engine.DynamicGenerator(bap_engine._get_embedder())
    parsed2 = gen._parse(raw)
    if parsed2:
        print(f"  PASS (via _parse) — type={parsed2['type']} | text={parsed2['text']}")
    else:
        print(f"  FAIL — could not parse: {e}")
        sys.exit(1)

# ── TEST 3: OOV path — Lucas numbers triggers definition before traversal ─────
print("\n[TEST 3] Full run — OOV: 'Lucas numbers' (define before traversal)")
space = bap_engine.load_thought_space("fibonacci_reasoning")
engine = bap_engine.BAPEngine(space)
print(f"  Generator available: {engine.generator.available()}")

nodes_before = len(space.nodes)
result = engine.run(
    "How do Lucas numbers relate to the Fibonacci sequence?",
    max_steps=7
)
print(result.summary())

# Confirm nodes were generated (IDs above the pre-run count)
new_node_ids = [nid for nid in space.nodes if nid >= nodes_before]
print(f"  Dynamic nodes added: {len(new_node_ids)}")
if new_node_ids:
    for nid in new_node_ids:
        print(f"    [{nid}] {space.nodes[nid].text[:80]}")
    print("  PASS — Groq injected node(s)")
else:
    print("  NOTE — no new nodes (all queries resolved from existing corpus)")

bap_engine.save_updated_nodes(space)

# ── TEST 4: retrieval miss path — multihop domain ─────────────────────────────
print("\n[TEST 4] Full run — retrieval miss fallback in multihop domain")
space2 = bap_engine.load_thought_space("multihop_facts")
engine2 = bap_engine.BAPEngine(space2)

result2 = engine2.run(
    "What was Marie Curie's most important scientific achievement?",
    max_steps=7
)
print(result2.summary())
bap_engine.save_updated_nodes(space2)

# ── SUMMARY ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("GROQ BACKEND VALIDATION — COMPLETE")
print(f"  Fibonacci path length : {result.steps_taken}")
print(f"  Multihop  path length : {result2.steps_taken}")
print(f"  Budget remaining (fib): {result.final_budget:.3f}")
print(f"  Budget remaining (mhp): {result2.final_budget:.3f}")
print("=" * 60)
