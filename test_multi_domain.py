"""
BAP — Multi-Domain Routing Validation
======================================
Tests that MultiBAPEngine correctly routes queries to the right domain.

Expected routing:
  Fibonacci queries  -> fibonacci_reasoning
  Arithmetic queries -> arithmetic
  Multihop queries   -> multihop_facts

Run:
    python test_multi_domain.py
"""

import bap_engine

print("Loading MultiBAPEngine (all 3 domains)...")
multi = bap_engine.MultiBAPEngine()
print("=" * 60)

# ── Test cases: (query, expected_domain, hint) ─────────────────────────────
TEST_CASES = [
    # Fibonacci
    ("What is the next number after 8 in the Fibonacci sequence?",
     "fibonacci_reasoning", "fib value"),
    ("What is the recurrence relation for Fibonacci numbers?",
     "fibonacci_reasoning", "fib rule"),
    # Arithmetic
    ("What is 17 mod 5?",
     "arithmetic", "modular arithmetic"),
    ("How do you compute the mean of a dataset?",
     "arithmetic", "statistics"),
    ("What is the quadratic formula?",
     "arithmetic", "algebra"),
    # Multihop / facts
    ("What scientific field did Marie Curie work in?",
     "multihop_facts", "curie"),
    ("Who invented the telephone?",
     "multihop_facts", "multihop fact"),
]

passed = 0
failed = 0

for query, expected_domain, hint in TEST_CASES:
    print(f"\n[QUERY] {query}")
    result = multi.run(query, max_steps=8)
    routed_domain = result.domain
    ok = routed_domain == expected_domain
    status = "PASS" if ok else "FAIL"
    if ok:
        passed += 1
    else:
        failed += 1
    print(f"  {status} routed={routed_domain}  expected={expected_domain}  hint={hint}")
    print(f"       steps={result.steps_taken}  budget_left={result.final_budget:.3f}")
    if result.answer_nodes:
        print(f"       answer: {result.answer_nodes[0][:80]}")

print("\n" + "=" * 60)
print(f"ROUTING RESULTS: {passed}/{len(TEST_CASES)} correct")
print(f"  Pass: {passed}   Fail: {failed}")
print("=" * 60)
