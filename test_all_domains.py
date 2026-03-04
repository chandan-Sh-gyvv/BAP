"""
BAP — Full 3-Domain Benchmark
==============================
Runs benchmark() on fibonacci_reasoning, arithmetic, and multihop_facts,
then persists strength/age updates via save_updated_nodes().

Usage:
    python test_all_domains.py
"""

import bap_engine

# ── Test cases: (query, cot_tokens, expected_contains) ────────────────────────

FIBONACCI_CASES = [
    {"query": "What is F(7)?",
     "cot_tokens": 40, "expected_answer_contains": "13"},
    {"query": "What is the next number after 8 in the Fibonacci sequence?",
     "cot_tokens": 40, "expected_answer_contains": "13"},
    {"query": "What is the recurrence relation for Fibonacci numbers?",
     "cot_tokens": 60, "expected_answer_contains": "f(n-1) + f(n-2)"},
    {"query": "What is Binet's formula?",
     "cot_tokens": 80, "expected_answer_contains": "sqrt(5)"},
    {"query": "What is the golden ratio phi?",
     "cot_tokens": 50, "expected_answer_contains": "phi"},
    {"query": "How does the Fibonacci sequence relate to Pascal's triangle?",
     "cot_tokens": 60, "expected_answer_contains": "pascal"},
    {"query": "What are Lucas numbers?",
     "cot_tokens": 60, "expected_answer_contains": "lucas"},
    {"query": "What is the Pisano period?",
     "cot_tokens": 60, "expected_answer_contains": "pisano"},
    {"query": "Are consecutive Fibonacci numbers coprime?",
     "cot_tokens": 40, "expected_answer_contains": "coprime"},
    {"query": "What is the Zeckendorf representation?",
     "cot_tokens": 60, "expected_answer_contains": "zeckendorf"},
]

ARITHMETIC_CASES = [
    {"query": "What is 17 mod 5?",
     "cot_tokens": 30, "expected_answer_contains": "remainder"},
    {"query": "How do you compute the mean of a dataset?",
     "cot_tokens": 40, "expected_answer_contains": "mean"},
    {"query": "What is the quadratic formula?",
     "cot_tokens": 50, "expected_answer_contains": "sqrt"},
    {"query": "What is the Pythagorean theorem?",
     "cot_tokens": 40, "expected_answer_contains": "right"},
    {"query": "How do you add fractions with different denominators?",
     "cot_tokens": 50, "expected_answer_contains": "denominator"},
    {"query": "What are the rules for exponents?",
     "cot_tokens": 60, "expected_answer_contains": "exponent"},
    {"query": "What is a prime number?",
     "cot_tokens": 40, "expected_answer_contains": "prime"},
    {"query": "How do logarithms work?",
     "cot_tokens": 60, "expected_answer_contains": "log"},
    {"query": "What is the sum of the first n natural numbers?",
     "cot_tokens": 50, "expected_answer_contains": "n * (n + 1)"},
    {"query": "What is standard deviation?",
     "cot_tokens": 50, "expected_answer_contains": "variance"},
]

MULTIHOP_CASES = [
    {"query": "What scientific field did Marie Curie work in?",
     "cot_tokens": 50, "expected_answer_contains": "physics"},
    {"query": "Who invented the telephone?",
     "cot_tokens": 40, "expected_answer_contains": "bell"},
    {"query": "What is the capital of France?",
     "cot_tokens": 30, "expected_answer_contains": "paris"},
    {"query": "When did World War II end?",
     "cot_tokens": 30, "expected_answer_contains": "1945"},
    {"query": "Who discovered penicillin?",
     "cot_tokens": 40, "expected_answer_contains": "fleming"},
    {"query": "What did Einstein receive the Nobel Prize for?",
     "cot_tokens": 50, "expected_answer_contains": "photoelectric"},
    {"query": "Who wrote Hamlet?",
     "cot_tokens": 30, "expected_answer_contains": "shakespeare"},
    {"query": "When was the first FIFA World Cup held?",
     "cot_tokens": 40, "expected_answer_contains": "1930"},
    {"query": "What is the tallest mountain in the world?",
     "cot_tokens": 40, "expected_answer_contains": "everest"},
    {"query": "Can two contradictory facts about the same subject both be true?",
     "cot_tokens": 30, "expected_answer_contains": "cannot"},
]

# ── Run benchmarks ─────────────────────────────────────────────────────────────

print("=" * 60)
print("BAP FULL DOMAIN BENCHMARK")
print("=" * 60)

all_results = {}

for domain, cases in [
    ("fibonacci_reasoning", FIBONACCI_CASES),
    ("arithmetic",          ARITHMETIC_CASES),
    ("multihop_facts",      MULTIHOP_CASES),
]:
    print(f"\n{'='*60}")
    print(f"DOMAIN: {domain}  ({len(cases)} queries)")
    print(f"{'='*60}")

    space  = bap_engine.load_thought_space(domain)
    engine = bap_engine.BAPEngine(space)

    result = bap_engine.benchmark(engine, cases)
    all_results[domain] = result

    bap_engine.save_updated_nodes(space)

# ── Global summary ─────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("GLOBAL SUMMARY")
print("=" * 60)
print(f"  {'Domain':<28}  {'Avg TRR':>8}  {'Hit Rate':>10}  {'Status'}")
print(f"  {'-'*28}  {'-'*8}  {'-'*10}  {'-'*6}")
for domain, r in all_results.items():
    trr_ok  = "OK" if r["avg_trr"]  < 0.40 else "MISS"
    hit_ok  = "OK" if r["hit_rate"] > 0.95 else "MISS"
    status  = "PASS" if trr_ok == "OK" and hit_ok == "OK" else "FAIL"
    print(
        f"  {domain:<28}  {r['avg_trr']:>8.3f}  {r['hit_rate']:>10.1%}  {status}"
    )
print(f"  Target:                           TRR < 0.40   hit > 95%")
print("=" * 60)
